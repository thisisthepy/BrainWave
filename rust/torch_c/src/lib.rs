//! `torch._C` -- the one native piece of PyTorch this project replaces.
//!
//! DESIGN.md §2: PyTorch is mostly Python. `nn/modules/*.py`, `nn/functional.py`,
//! `_tensor.py`, `_decomp/` are all vendored from upstream under BSD; the only
//! part that is native, and therefore the only part that has to be rebuilt, is
//! `_C` -- ATen tensors, the dispatcher, autograd. §5 settled how: candle under
//! a PyO3 adapter (option A), because option B's mobile CMake path force-sets
//! `BUILD_PYTHON=OFF` and so structurally cannot produce a `torch._C` at all.
//!
//! Layout:
//!
//! | module | what |
//! |---|---|
//! | `tensor` | `TensorBase` -- identity of a tensor: shape, dtype, device |
//! | `dtype` | `torch.float32` and friends, as `_C`-owned instances |
//! | `device` | `torch.device`, a label rather than a live backend handle |
//! | `aten` | the single dispatch entrance, and the ops behind it |
//! | `capture` | recording a straight-line region at that entrance, and replaying it |
//! | `quant` | block-quantised weights: candle's `QTensor` behind the third `Repr` arm |
//! | `err` | the message shapes; §6's discovery mechanism lives on these |
//!
//! This is a floor, not a coverage effort. Three ops are implemented. Everything
//! else raises with its own name, so running a model produces the work queue by
//! itself, in frequency order (§6). Details in docs/TORCH_C.md.
// The module must be named `_C` -- that is the name Python imports. rustc's
// snake-case lint has no opinion worth honouring here.
#![allow(non_snake_case)]

use candle_core::Tensor;
use pyo3::prelude::*;
use pyo3::types::PyModule;

mod aten;
mod capture;
mod device;
mod dtype;
mod err;
mod flash;
mod info;
mod quant;
mod reduced;
mod rng;
mod storage;
mod tape;
mod tensor;
mod vulkan;

use crate::device::PyDevice;
use crate::dtype::PyDtype;
use crate::err::candle_err;
use crate::tensor::PyTensorBase;

/// Scaffolding, not torch. There is no aten op that takes a Python list of
/// numbers -- `torch.tensor(...)` is a Python-layer factory that lowers to
/// `lift_fresh`/`_to_copy`, and `lift_fresh` is one of the two ops CORE_ATEN §0
/// found to be neither Core ATen nor covered by the decomposition table. Until
/// that is decided, tests need *some* way to get real data in, so this exists
/// with a leading underscore and no aten name, and is expected to be deleted
/// rather than promoted.
#[pyfunction]
#[pyo3(signature = (values, shape, dtype = None, device = None))]
fn _tensor_from_flat(
    py: Python<'_>,
    values: Vec<f64>,
    shape: Vec<usize>,
    dtype: Option<PyDtype>,
    device: Option<PyDevice>,
) -> PyResult<Py<PyAny>> {
    let expected: usize = shape.iter().product();
    if values.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "_tensor_from_flat: shape {shape:?} needs {expected} values, got {}",
            values.len()
        )));
    }
    let device = device.unwrap_or_else(PyDevice::cpu).resolve()?;
    // BOOL.md §6.3 lists this function as one of the two ways the `torch.bool`
    // invariant could be broken quietly, since arbitrary `f64`s come in here.
    // It used to refuse the tag outright. It now *normalises* instead, for the
    // reason docs/OVERLOAD.md §6.7 gave when `_tensor_new_from_data` was added:
    // the invariant is kept by construction, not by hope, as long as every byte
    // under a `bool` tag goes through `PyTensorBase::boolean` after being
    // reduced to 0/1. `!= 0` is that reduction, and it is also what torch
    // guarantees a bool tensor *reads* as (BOOL.md §2.6).
    //
    // The refusal had become load-bearing in the wrong direction:
    // `tools/golden/build.py` builds every operand through this function, so a
    // mask could not be built at all, and `masked_fill` and `index.Tensor`
    // -- both of which take a `torch.bool` argument -- could not be compared
    // against upstream. An op that cannot be golden-checked is worse than a
    // constructor that normalises.
    if dtype.is_some_and(|d| d.tag() == crate::dtype::TorchDType::Bool) {
        let bytes: Vec<u8> = values.iter().map(|v| u8::from(*v != 0.0)).collect();
        let tensor = Tensor::from_vec(bytes, shape, &device)
            .map_err(|e| candle_err("_tensor_from_flat", e))?;
        return crate::tensor::promote(
            py,
            PyTensorBase::boolean(tensor)?.into_pyobject(py)?.into_any().unbind(),
        );
    }
    let target = dtype
        .map(|d| d.storage("_tensor_from_flat"))
        .transpose()?
        .unwrap_or(candle_core::DType::F32);
    let tensor = Tensor::from_vec(values, shape, &device)
        .and_then(|t| t.to_dtype(target))
        .map_err(|e| candle_err("_tensor_from_flat", e))?;
    crate::tensor::promote(
        py,
        PyTensorBase::new(tensor)?.into_pyobject(py)?.into_any().unbind(),
    )
}

/// One element of a nested Python sequence handed to `torch.tensor(...)`,
/// kept in the category it arrived as -- that is what decides the dtype.
enum Leaf {
    Bool(bool),
    Int(i64),
    Float(f64),
}

fn sequence_items<'py>(value: &Bound<'py, PyAny>) -> Option<Vec<Bound<'py, PyAny>>> {
    if let Ok(list) = value.cast::<pyo3::types::PyList>() {
        return Some(list.iter().collect());
    }
    if let Ok(tuple) = value.cast::<pyo3::types::PyTuple>() {
        return Some(tuple.iter().collect());
    }
    None
}

fn ragged() -> PyErr {
    pyo3::exceptions::PyValueError::new_err(
        "torch._C shim: torch.tensor() got a ragged nested sequence",
    )
}

fn walk_data(
    value: &Bound<'_, PyAny>,
    depth: usize,
    shape: &mut Vec<usize>,
    leaf_depth: &mut Option<usize>,
    out: &mut Vec<Leaf>,
) -> PyResult<()> {
    if let Some(items) = sequence_items(value) {
        if shape.len() == depth {
            shape.push(items.len());
        } else if shape[depth] != items.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected sequence of length {} at dim {} (got {})",
                shape[depth],
                depth,
                items.len()
            )));
        }
        for item in items {
            walk_data(&item, depth + 1, shape, leaf_depth, out)?;
        }
        return Ok(());
    }
    // Every leaf has to sit at the same depth. Counting elements is not
    // enough: `[[1], 2]` walks to shape `[2, 1]` and produces exactly the two
    // values that shape asks for, so the count check passes and the answer is
    // wrong.
    match leaf_depth {
        Some(seen) if *seen != depth => return Err(ragged()),
        _ => *leaf_depth = Some(depth),
    }
    // `bool` before `int`: it is a subclass, and the difference is the whole
    // point -- `torch.tensor([True])` is `torch.bool`, `torch.tensor([1])` is
    // `torch.int64`.
    if value.is_instance_of::<pyo3::types::PyBool>() {
        out.push(Leaf::Bool(value.extract()?));
    } else if value.is_instance_of::<pyo3::types::PyInt>() {
        out.push(Leaf::Int(value.extract()?));
    } else if value.is_instance_of::<pyo3::types::PyFloat>() {
        out.push(Leaf::Float(value.extract()?));
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "torch._C shim: torch.tensor() takes nested sequences of Python \
             bool/int/float, and got {}",
            value.get_type().name().map(|n| n.to_string()).unwrap_or_default()
        )));
    }
    Ok(())
}

/// The storage half of `torch.tensor(...)`.
///
/// Upstream's `torch.tensor` is not an aten op either: it is
/// `THPVariable_tensor` -> `internal_new_from_data`, a `_C` function the
/// dispatcher never sees, and the only aten record a real
/// `torch.tensor([1, 2])` produces is `aten.lift_fresh.default` (measured with
/// a `TorchDispatchMode` logger). So this mirrors upstream's shape rather than
/// inventing an `aten::tensor` call -- `bootstrap.py` builds the data here and
/// then passes the result through `lift_fresh`, which *is* dispatched.
///
/// Distinct from `_tensor_from_flat`, which stays what it is: scaffolding due
/// for deletion that takes an already-flat `f64` list and refuses `torch.bool`
/// outright (BOOL.md §6.3). This one has to accept booleans, because
/// `torch.tensor([True, False])` is how a mask is written -- and it does so
/// through `PyTensorBase::boolean`, the one tagging constructor, so the 0/1
/// invariant holds by construction here too.
#[pyfunction]
#[pyo3(signature = (data, dtype = None, device = None))]
fn _tensor_new_from_data(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    dtype: Option<PyDtype>,
    device: Option<PyDevice>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "torch.tensor";

    let mut shape: Vec<usize> = Vec::new();
    let mut leaves: Vec<Leaf> = Vec::new();
    let mut leaf_depth: Option<usize> = None;
    walk_data(data, 0, &mut shape, &mut leaf_depth, &mut leaves)?;
    // An empty list has no leaves and a legal shape of `[0]`; anything with
    // leaves must have them exactly at the bottom of the shape it declared.
    let expected: usize = shape.iter().product();
    if leaf_depth.is_some_and(|d| d != shape.len()) || leaves.len() != expected {
        return Err(ragged());
    }

    // torch's inference, in category order: all booleans give `torch.bool`,
    // all integers give `int64`, anything with a float in it gives the default
    // float dtype. An empty list gives the default float dtype.
    let all_bool = !leaves.is_empty() && leaves.iter().all(|l| matches!(l, Leaf::Bool(_)));
    let any_float = leaves.iter().any(|l| matches!(l, Leaf::Float(_)));
    let inferred = if all_bool {
        crate::dtype::TorchDType::Bool
    } else if any_float || leaves.is_empty() {
        crate::dtype::default_float()
    } else {
        crate::dtype::TorchDType::Int64
    };
    let tag = dtype.map(|d| d.tag()).unwrap_or(inferred);
    let label = device.unwrap_or_else(PyDevice::cpu);
    // `torch.tensor([1., 2.], device="meta")` keeps the shape and dtype the
    // data implies and throws the data away, which is what upstream does
    // (measured: `tensor(..., device='meta', size=(2,))`). The walk above still
    // runs -- a ragged nested sequence is a `ValueError` on meta too -- because
    // the shape *is* the answer here, and a shape that had not been validated
    // would be worth nothing.
    //
    // This is also on the `with torch.device("meta")` path and not only the
    // explicit one: `torch.get_default_device()` inside a device context is
    // implemented upstream as `torch.tensor([]).device`
    // (`torch/__init__.py:1222`), so the context manager cannot report itself
    // without this branch.
    if label.is_meta() {
        return crate::tensor::promote(
            py,
            PyTensorBase::meta(shape, tag).into_pyobject(py)?.into_any().unbind(),
        );
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let device = label.resolve()?;

    let tensor = if tag == crate::dtype::TorchDType::Bool {
        let bytes: Vec<u8> = leaves
            .iter()
            .map(|leaf| {
                u8::from(match leaf {
                    Leaf::Bool(v) => *v,
                    Leaf::Int(v) => *v != 0,
                    Leaf::Float(v) => *v != 0.0,
                })
            })
            .collect();
        Tensor::from_vec(bytes, shape, &device)
    } else if storage.is_int() {
        let values: Vec<i64> = leaves
            .iter()
            .map(|leaf| match leaf {
                Leaf::Bool(v) => i64::from(*v),
                Leaf::Int(v) => *v,
                Leaf::Float(v) => *v as i64,
            })
            .collect();
        Tensor::from_vec(values, shape, &device).and_then(|t| t.to_dtype(storage))
    } else {
        let values: Vec<f64> = leaves
            .iter()
            .map(|leaf| match leaf {
                Leaf::Bool(v) => f64::from(u8::from(*v)),
                Leaf::Int(v) => *v as f64,
                Leaf::Float(v) => *v,
            })
            .collect();
        Tensor::from_vec(values, shape, &device).and_then(|t| t.to_dtype(storage))
    }
    .map_err(|e| candle_err(OP, e))?;

    let wrapped = if tag == crate::dtype::TorchDType::Bool {
        PyTensorBase::boolean(tensor)?
    } else {
        PyTensorBase::new(tensor)?
    };
    crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())
}

/// `torch.frombuffer(buffer, *, dtype, count=-1, offset=0, requires_grad=False)`.
///
/// Not an aten op, and deliberately not routed through `_aten_dispatch`:
/// upstream has no `aten::frombuffer` (checked -- `torch.ops.aten.frombuffer`
/// raises `AttributeError` on torch 2.13.0). It is a `_C` binding,
/// `THPVariable_frombuffer` in `torch/csrc/utils/tensor_new.cpp`, reachable
/// only as `torch._C._VariableFunctions.frombuffer`. So this is the same split
/// `_tensor_factory` already makes for `torch.tensor`: `_C` builds the data,
/// and the door stays one door because there is no aten call to make.
///
/// It is here because it is the *entire* cost of reading a safetensors
/// checkpoint. `safetensors.torch.load(bytes)` decodes the container in its own
/// Rust extension and then calls exactly one torch function per tensor --
/// `torch.frombuffer(v["data"], dtype=dtype).reshape(v["shape"])`
/// (`safetensors/torch.py:468`). Measured: with this function and nothing else,
/// that path goes from its first wall to a full state dict. See docs/CKPT.md.
///
/// **This copies; upstream aliases.** `torch.frombuffer` upstream returns a
/// tensor that shares memory with the buffer -- writing to the buffer changes
/// the tensor, which is measured and is why upstream warns about non-writable
/// buffers. candle owns its storage, so the bytes are copied in and the two are
/// independent afterwards. For loading a checkpoint that difference is
/// invisible (the buffer is read once and dropped), and it is recorded rather
/// than fixed because fixing it means a storage concept candle does not have.
/// Anything that relies on the aliasing gets wrong answers quietly, so it is
/// written down here and in docs/CKPT.md rather than left to be discovered.
///
/// The `ValueError` messages are upstream's, transcribed from torch 2.13.0 by
/// running each failing case. Behaviour, not just wording: `count == 0` is an
/// error even when the buffer is non-empty, and any `count < 0` -- not just
/// `-1` -- means "all the rest".
#[pyfunction]
#[pyo3(signature = (buffer, *, dtype, count = -1, offset = 0, requires_grad = false))]
fn _frombuffer(
    py: Python<'_>,
    buffer: &Bound<'_, PyAny>,
    dtype: PyDtype,
    count: i64,
    offset: i64,
    requires_grad: bool,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "torch.frombuffer";

    // The buffer protocol rather than a `bytes`/`bytearray` downcast: upstream
    // takes anything that implements it, and `safetensors` hands over a
    // `bytearray` while pickle frames arrive as `memoryview`. `PyBuffer` is
    // available under the Limited API from 3.11 (`pyo3/src/buffer.rs`), and
    // this crate's floor is 3.13.
    let view = pyo3::buffer::PyBuffer::<u8>::get(buffer)?;
    let bytes = view.to_vec(py)?;
    let len = bytes.len() as i64;
    let itemsize = dtype.tag().itemsize() as i64;

    if offset < 0 || offset > (len - 1).max(0) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "offset ({offset} bytes) must be non-negative and no greater than \
             buffer length ({len} bytes) minus 1"
        )));
    }
    if len == 0 || count == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "both buffer length ({len}) and count ({count}) must not be 0"
        )));
    }

    let numel = if count < 0 {
        if (len - offset) % itemsize != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "buffer length ({len} bytes) after offset ({offset} bytes) must \
                 be a multiple of element size ({itemsize})"
            )));
        }
        (len - offset) / itemsize
    } else {
        if count * itemsize + offset > len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "requested buffer length ({count} * {itemsize} bytes) after \
                 offset ({offset} bytes) must not be greater than actual buffer \
                 length ({len} bytes)"
            )));
        }
        count
    };

    let start = offset as usize;
    let end = start + (numel * itemsize) as usize;
    let slice = &bytes[start..end];

    // The bytes-to-tensor step is shared with `TensorBase.set_`, which reads
    // the same little-endian payload out of a storage instead of out of a
    // buffer. Sharing it means the dtype narrowing and the `torch.bool`
    // normalisation cannot drift between the safetensors path and the
    // `torch.load` path -- both are checkpoint readers, and the two agreeing is
    // exactly what docs/CKPT.md §1 measures (worst difference: 0.0).
    let wrapped = crate::tensor::from_le_bytes(OP, slice, &[numel as usize], dtype.tag())?;
    let out = crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())?;
    carry_requires_grad(py, out, requires_grad)
}

/// `requires_grad=True` on a factory that builds its tensor in Rust.
///
/// The two here (`frombuffer`, `asarray`) used to refuse it, alongside the five
/// Python-level doors `bootstrap.py::_strip_python_only_kwargs` covers. Those
/// five now carry the flag (docs/BACKWARD2.md §4.1: the refusal protected a
/// *spelling*, since `.requires_grad_(True)` reaches the same tensor), and
/// leaving these two refusing would re-create the inconsistency one door over.
///
/// The dtype rule and its wording come from `TensorBase.set_requires_grad`, so
/// there is exactly one statement of it on this path.
fn carry_requires_grad(
    py: Python<'_>,
    out: Py<PyAny>,
    requires_grad: bool,
) -> PyResult<Py<PyAny>> {
    if requires_grad {
        out.bind(py).setattr("requires_grad", true)?;
    }
    Ok(out)
}

/// `torch.asarray(obj, *, dtype=None, device=None, copy=None, requires_grad=False)`,
/// narrowed to the one shape that reaches it: a storage.
///
/// Like `frombuffer` above, not an aten op -- `torch.ops.aten.asarray` does not
/// exist on 2.13.0, the function is `torch::utils::asarray` behind
/// `_C._VariableFunctions.asarray` -- so it is here rather than behind the
/// dispatcher, for the same reason and with the same shared byte reader.
///
/// **Who calls it.** safetensors' default `mmap` backend, once per tensor.
/// Measured by wrapping `torch.asarray` and running `safe_open(..., backend=
/// "mmap")`: it is called with a `torch.UntypedStorage` and
/// `dtype=torch.uint8`, plus `device="cpu"` on the `get_tensor` path. The
/// result is then `.view(real_dtype).reshape(shape)`. Together with
/// `UntypedStorage.from_file` and `aten::view.dtype` this is what the default
/// `from_pretrained` route to a safetensors checkpoint costs; docs/CKPT2.md §4.
///
/// **The narrowing is real and is refused by name.** Upstream's `asarray` also
/// takes tensors, sequences, scalars, numpy arrays and buffer objects, and
/// implementing those would be re-deriving `torch.tensor`'s conversion rules
/// with no measured caller -- docs/E2E_REAL.md §1.2 is about exactly that kind
/// of speculative surface. Anything but a storage stops here with the type it
/// was given and a pointer at the two functions that do take those.
///
/// **This copies; upstream aliases**, the same divergence `_frombuffer`
/// records, and invisible for the same reason: the storage is read once while
/// a checkpoint is being loaded and nothing writes through either handle.
#[pyfunction]
#[pyo3(signature = (obj, *, dtype = None, device = None, copy = None, requires_grad = false))]
fn _asarray(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    dtype: Option<PyDtype>,
    device: Option<&Bound<'_, PyAny>>,
    copy: Option<bool>,
    requires_grad: bool,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "torch.asarray";

    // `requires_grad` is carried at the end -- see `carry_requires_grad`.
    // `copy=False` still refuses, and the two are not the same case: the copy
    // names an aliasing this shim's tensors cannot express.
    if copy == Some(false) {
        return Err(crate::err::not_implemented(format!(
            "{OP}(copy=False) -- upstream would alias the source's memory, and \
             this shim's tensors own theirs (see `_frombuffer`). Refusing rather \
             than copying, because a caller that asked for no copy asked for the \
             aliasing, not for the values"
        )));
    }
    if let Some(d) = device {
        let label = d.str()?.to_string();
        if label != "cpu" {
            return Err(crate::err::not_implemented(format!(
                "{OP}(device={label:?}) -- storages in this shim are CPU byte \
                 buffers, so there is nothing to read on another device"
            )));
        }
    }

    let Ok(storage) = obj.extract::<PyRef<'_, crate::storage::PyStorageBase>>() else {
        return Err(crate::err::not_implemented(format!(
            "{OP}: torch._C shim implements this only for a torch.UntypedStorage, \
             which is the form safetensors calls it with; got {}. For a buffer use \
             torch.frombuffer, for a sequence or scalar use torch.tensor",
            obj.get_type().name()?,
        )));
    };

    let tag = dtype
        .ok_or_else(|| {
            crate::err::not_implemented(format!(
                "{OP}(storage) without dtype -- an untyped storage is bytes, so \
                 there is no dtype to infer. safetensors always passes one"
            ))
        })?
        .tag();
    let bytes = storage.bytes();
    let itemsize = tag.itemsize();
    if bytes.len() % itemsize != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{OP}: storage of {} bytes is not a whole number of torch.{} elements \
             ({itemsize} bytes each)",
            bytes.len(),
            tag.name()
        )));
    }
    // Same byte reader as `frombuffer` and `TensorBase.set_`; see the comment
    // at the end of `_frombuffer`.
    let wrapped = crate::tensor::from_le_bytes(OP, bytes, &[bytes.len() / itemsize], tag)?;
    let out = crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())?;
    carry_requires_grad(py, out, requires_grad)
}

/// The triple this artefact was built for. Three targets are cross-compiled and
/// the results are indistinguishable once renamed to `_C.so`, so the build
/// records it here rather than leaving it to be guessed from a file path.
#[pyfunction]
fn _shim_target() -> &'static str {
    env!("TORCH_C_TARGET")
}

/// Reads, and optionally sets, whether `sdpa` computes with `crate::flash` --
/// upstream's blocked kernel reproduced bit for bit -- instead of candle's
/// tensor ops. Answers the value that was in force before the call, so a
/// caller can restore it.
///
/// Off by default: the reference kernel costs 20x at T=512. `crate::flash`
/// carries the measurement, the argument, and -- in `apply_env` -- why this is
/// a name of ours rather than one of upstream's.
///
/// A setter and not only `BW_SDPA_REFERENCE`, because the things that have to
/// run with it on are individual cases inside a suite that must stay fast:
/// `pytests/test_shim.py`'s three tolerance-free sdpa tests and
/// `tools/golden/cases.py`'s sixteen block-boundary cases, both of which flip
/// it and flip it back. An env-only switch would mean a second process for
/// them, which neither harness has a place to spawn.
#[pyfunction]
#[pyo3(signature = (enabled=None))]
fn _shim_sdpa_reference(enabled: Option<bool>) -> bool {
    match enabled {
        Some(enabled) => crate::flash::set_reference(enabled),
        None => crate::flash::reference_enabled(),
    }
}

/// The name surface the vendored tree expects `_C` to present, extracted from
/// the tree's own `.pyi` stubs by `vendor/gen_surface.py` and compiled in so
/// the artefact needs nothing on disk at runtime. See `bootstrap.py`.
const SURFACE: &str = include_str!("surface.json");

/// The signature list `torch.<op>(...)` resolves against, per op, in order.
/// Unlike `SURFACE` this is not generated from the vendored tree -- the tree
/// carries aten overload *names* and Python-level signatures but nothing that
/// joins them (docs/OVERLOAD.md §2) -- so it is transcribed and checked by
/// `pytests/verify_schemas.py` against an installed upstream torch. Compiled
/// in the same way: nothing is read from disk at runtime.
const OVERLOADS: &str = include_str!("overloads.json");

/// The same, for `tensor.<method>(...)`. A separate table because upstream has
/// a separate binding: `THPVariable_mul` (a `TensorBase` method) and the
/// `_VariableFunctions` entry are different C functions with different
/// signature lists, which is why docs/C_SURFACE.md counted the two surfaces
/// apart -- 50 `TensorBase` members called against 13 hoisted functions.
/// Checked by the same `pytests/verify_schemas.py`.
const METHODS: &str = include_str!("methods.json");

/// Everything that is a name rather than a behaviour is built in Python, from
/// `SURFACE`, at module init.
///
/// Why Python and not Rust: what has to be built is ~1,700 Python callables,
/// ~200 heap types with chosen metaclasses, and 27 entries in `sys.modules`.
/// All of that is dynamic Python object construction either way; doing it in
/// Rust would be the same operations spelled through `Bound<'_, PyAny>`, at
/// several times the length, with no more type safety, and it would still be
/// executing at exactly this moment. Keeping it in one readable file also
/// keeps the *reason* each name exists next to the name.
///
/// What stays in Rust is everything with behaviour: dtypes, devices, tensors,
/// and the aten dispatcher. `bootstrap.py` never computes anything -- it wires
/// names to the one door in `aten.rs`.
fn run_bootstrap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let code = std::ffi::CString::new(include_str!("bootstrap.py"))?;
    let boot = PyModule::from_code(
        py,
        code.as_c_str(),
        c"torch_c_bootstrap.py",
        c"_torch_c_bootstrap",
    )?;
    boot.getattr("install")?.call1((m, SURFACE, OVERLOADS, METHODS))?;
    Ok(())
}

/// How much arithmetic a matmul must contain before `gemm` is allowed to spread
/// it over threads. `gemm`'s own default is `48 * 48 * 256 = 589_824`, and on
/// both machines this repository can measure, that number is too low: it hands
/// a job that takes tens of microseconds on one core to four or eight cores and
/// pays more in wakeups than it saves.
///
/// `docs/PERF_ANDROID.md` §4 has the sweep. On the host (M1, 8 cores, idle) a
/// 96x96x96 matmul is 0.0247 ms on one thread and 0.042-0.049 ms threaded --
/// threading loses 1.7-2.0x -- while 192x192x192 and up win 2-3.4x. The
/// crossover sits at 2-4 M multiply-adds. On the Android device the crossover
/// is much higher still (128x128x128 is 0.051 ms single, 0.35 ms threaded, a
/// 6.9x loss), so a value chosen from the host is the conservative one.
///
/// This cannot change any result. `gemm` parallelises by splitting the *output
/// columns* between threads; the `k` accumulation loop is outside that split
/// and runs identically either way, so every output element is the same
/// sequence of operations. That is checked rather than assumed -- §5 of the
/// same document hashes `mm` output over n = 96..512 at three thresholds
/// (fully parallel, this value, fully serial) and gets one digest. The golden
/// suite cannot check it: every shape in it is below even gemm's own default,
/// so it is single-threaded on both sides of the change.
const GEMM_THREADING_THRESHOLD: usize = 4_000_000;

/// `candle` hands every matmul to `gemm` with `Parallelism::Rayon(n)` and lets
/// `gemm` decide whether to use the threads. That decision is a process-global
/// `AtomicUsize` with a public setter, so this is the whole of the fix.
///
/// `BW_GEMM_THREADING_THRESHOLD` overrides it so the measurement in
/// `docs/PERF_ANDROID.md` can be re-run without a rebuild.
fn apply_gemm_threading_threshold() {
    let value = std::env::var("BW_GEMM_THREADING_THRESHOLD")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(GEMM_THREADING_THRESHOLD);
    gemm::set_threading_threshold(value);
}

#[pymodule]
fn _C(m: &Bound<'_, PyModule>) -> PyResult<()> {
    apply_gemm_threading_threshold();
    flash::apply_env();
    dtype::register(m)?;
    device::register(m)?;
    info::register(m)?;
    tensor::register(m)?;
    aten::register(m)?;
    capture::register(m)?;
    tape::register(m)?;
    rng::register(m)?;
    storage::register(m)?;
    quant::register(m)?;
    vulkan::register(m)?;
    m.add_function(wrap_pyfunction!(_tensor_from_flat, m)?)?;
    m.add_function(wrap_pyfunction!(_tensor_new_from_data, m)?)?;
    m.add_function(wrap_pyfunction!(_frombuffer, m)?)?;
    m.add_function(wrap_pyfunction!(_asarray, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_target, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_sdpa_reference, m)?)?;
    run_bootstrap(m)?;
    Ok(())
}
