//! `torch._C.TensorBase` -- the candle tensor, seen from Python.
//!
//! The name is not arbitrary. The vendored Python tree's `torch/_tensor.py`
//! opens with `class Tensor(torch._C.TensorBase)`, and `torch._tensor` is one of
//! the ten modules IMPORT_WALLS §5 measured as actually executing during
//! inference. So `TensorBase` is the exact name the Python layer will subclass,
//! and getting it right now costs nothing while getting it wrong later means
//! patching a vendored file.
//!
//! What lives here is only the *identity* of a tensor -- shape, dtype, device.
//! Arithmetic does not: it goes through the aten dispatcher (`aten.rs`) so that
//! every operation passes the one choke point where an unimplemented op names
//! itself. A convenience method here would be a second, unmeasured entrance.
use std::sync::Arc;

use candle_core::quantized::QTensor;
use candle_core::{CpuStorage, DType, InplaceOp1, Layout, Tensor};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::device::PyDevice;
use crate::dtype::{PyDtype, TorchDType};
use crate::err::{candle_err, not_implemented};

/// What a `TensorBase` is made of.
///
/// **The reason this is an enum is `meta`.** Upstream's meta tensor has shape,
/// dtype and stride and *no bytes* -- `torch.zeros(2, 3, device="meta")
/// .data_ptr()` is `0`, measured. candle has no such thing: every
/// `candle_core::Tensor` owns storage, and `Tensor::zeros` allocates. So a
/// meta tensor cannot be a candle tensor wearing a label; allocating and
/// calling it `meta` would invert the one property meta exists for.
/// docs/devices/META.md §3.
///
/// The cost of the enum is paid once, at `tensor()`: it returns a `PyResult`
/// instead of a `&Tensor`, so **no kernel can read storage off a meta tensor
/// without handling the failure.** That is the point -- the refusal is
/// structural rather than a check each of the 96 kernels has to remember, which
/// is the same argument `check_devices_agree` makes for living at the door.
#[derive(Clone)]
pub enum Repr {
    Dense(Tensor),
    /// `meta`: shape and dtype, no storage.
    ///
    /// Stride is deliberately absent. `TensorBase` has no `.stride()` in this
    /// shim (dense tensors do not report one either), so modelling strides here
    /// would give meta a surface the dense side does not have. Recorded in
    /// docs/devices/META.md §6 as a narrowing, since upstream's meta *does* carry
    /// stride (`torch.zeros(2,3,device="meta").t().stride()` is `(1, 3)`).
    ///
    /// The device label is not stored either, and that is measured rather than
    /// assumed: upstream normalises every meta index away --
    /// `torch.zeros(2, device="meta:7").device` is `device(type='meta')`, same
    /// as `meta:0` and bare `meta`. So there is exactly one meta device and the
    /// label is a constant. If a device kind ever arrives where the index
    /// *survives*, this is the field that has to appear, and
    /// docs/devices/DEVICE_ABS.md §3.2 is the argument for it.
    ///
    /// `storage_id` is the identity of the storage this meta tensor would
    /// have. It is **not** an address: upstream's meta storage answers
    /// `data_ptr() == 0` (measured on 2.13.0 -- every meta storage, base or
    /// view, answers zero), so there is no address to carry. What upstream
    /// *does* carry is a distinct `_cdata` per storage which two views of one
    /// base share, and that is what this token is. It is drawn from a
    /// process-wide counter at construction and **propagated by the one meta
    /// kernel that is a view** (`aten.view.default`), so `x` and `x.view(-1)`
    /// answer with one storage here as they do upstream, and two separately
    /// constructed meta tensors do not. docs/graph/EXPORT5.md §2.
    Meta { shape: Vec<usize>, storage_id: usize },
    /// A GGML block-quantised weight.
    ///
    /// **The reason this is a third arm and not a `Tensor` wearing a label is
    /// that candle's quantisation is not a `DType`.** `QTensor` lives in a
    /// separate type system (`candle_core::quantized`) with its own element
    /// enumeration (`GgmlDType`), its own storage, and its own matmul
    /// (`QMatMul`); it is not convertible to `&Tensor` without dequantising,
    /// which allocates and throws away the whole point. docs/graph/QUANT.md §5.1 and
    /// docs/numerics/DTYPE.md §6.3.
    ///
    /// So `tensor()` refuses on this arm exactly as it refuses on `Meta`, and
    /// for the same structural reason: **no kernel can read dense storage off
    /// a quantised tensor by forgetting to check.** The 96 kernels in
    /// `aten.rs` inherit the refusal from the type rather than from a rule
    /// each of them has to remember. Only the ops taught the quantised arm by
    /// name (`quant.rs`) can compute on one.
    ///
    /// `Arc` rather than the value: `QTensor` is not `Clone` (it carries a
    /// `OnceLock` cache of the repacked blocks that `cpu_fwd` fills on first
    /// use), and `QMatMul::from_arc` wants an `Arc` anyway. Sharing it also
    /// means that cache survives across calls, which is where the repacked
    /// Q4K path's cost is amortised.
    Quantized(Arc<QTensor>),
    /// A tensor whose bytes are in a `VkBuffer` on the `vulkan` device.
    ///
    /// **The fourth arm exists for the same structural reason as the third.**
    /// `PyDevice::resolve()` returns a `candle_core::Device`, which is a closed
    /// enum of `Cpu | Cuda | Metal` with nowhere to put a Vulkan handle, so a
    /// Vulkan tensor cannot be a `candle::Tensor` wearing a label any more than
    /// a GGML weight can. It has to live outside candle, and this is where.
    /// docs/devices/VULKAN2.md §5.1 sized it; docs/devices/VULKAN3.md is what came of that.
    ///
    /// And it inherits `Quantized`'s safety property, which is the whole point
    /// of putting it here rather than anywhere else: `tensor()` refuses on this
    /// arm, and `tensor()` has 396 call sites. **No kernel can read CPU storage
    /// off a Vulkan tensor by forgetting to check** -- it would have to handle
    /// a `PyResult` whose only content is a refusal. A silent CPU fallback is
    /// therefore not a discipline anyone has to keep; it is unrepresentable.
    /// Ops opt in one at a time in `vulkan::dispatch`, by name, the way the
    /// twenty `Repr::Quant` sites did.
    Vulkan(crate::vulkan::VkTensor),
    /// A complex tensor, held as **two real tensors** rather than one
    /// interleaved buffer.
    ///
    /// **The fifth arm exists for the same structural reason as the third and
    /// the fourth: candle cannot hold the thing.** `candle_core::DType`
    /// enumerates fourteen real dtypes and no complex one, in 0.11.0 and on
    /// `main` alike, and -- unlike `torch.int8` (docs/numerics/INT8.md) -- it cannot get
    /// one by adding an arm. `WithDType` is bounded on `std::cmp::PartialOrd`
    /// and `cpu::kernels::VecOps` requires `min`/`max`; the complex numbers
    /// are not ordered, so a complex `DType` means *removing* a bound from
    /// candle's core numeric trait and re-bounding every comparison,
    /// reduction, sort, clamp and argmax generic over it, across three
    /// backends, carried against upstream forever. docs/kernels/COMPLEX.md §2.2 sized
    /// that and refused it; no `[patch]` is offered and none should be added.
    ///
    /// **A pair, not interleaving.** Upstream's own representation interleaves
    /// real and imaginary in a trailing dimension of size 2 -- which is exactly
    /// what `view_as_complex` views -- and that is the wrong choice *here*:
    /// `Repr::Dense`'s shape is candle's shape, so an interleaved complex
    /// tensor would report a trailing `2` in `.shape` that then has to be
    /// hidden at every shape-reporting site. Hiding a dimension in one place
    /// and not another is the silent-wrong-answer shape. With a pair,
    /// `re.shape()` **is** the complex tensor's shape, by construction.
    ///
    /// The invariants, established once in `PyTensorBase::complex` and relied
    /// on everywhere below: `re` and `im` have the same dims, the same candle
    /// dtype, and the same device, and that dtype is one of `F16`/`F32`/`F64`
    /// so that `TorchDType::to_complex` names the tag.
    ///
    /// And it inherits the property that is the whole reason it lives here:
    /// **`tensor()` refuses on this arm**, and `tensor()` has ~400 call sites.
    /// No kernel can read a complex tensor's storage and get back a real
    /// buffer that happens to be the real part -- it would have to handle a
    /// `PyResult` whose only content is a refusal. Dropping the imaginary part
    /// and returning plausible numbers is therefore not a discipline anyone
    /// has to keep; it is unrepresentable, which is the standard
    /// docs/devices/VULKAN2.md set. Ops opt in one at a time, by name, in
    /// `complex_ops` below.
    Complex { re: Tensor, im: Tensor },
}

#[pyclass(name = "TensorBase", module = "torch._C", subclass, from_py_object)]
pub struct PyTensorBase {
    inner: Repr,
    /// The torch-level dtype. Not derivable from `inner.dtype()`: `torch.bool`
    /// and `torch.uint8` share candle's `U8` storage and differ only here.
    /// BOOL.md §5-B is the decision; §6.3 is the invariant that comes with it
    /// (the bytes under a `bool` tag are 0 or 1), which is why `boolean()` is
    /// the only way to attach that tag.
    tag: TorchDType,
    /// **Inert.** Stored and reported, read by nothing. There is no autograd
    /// here (DESIGN.md §3 stage 0), so no graph node is ever created from it
    /// and `grad_fn` is always `None`. It exists because `from_config` calls
    /// `TensorBase.requires_grad_` before it calls anything else interesting,
    /// and the alternative was stopping there. `backward()` stays a raising
    /// stub so that code which really depends on the flag meaning something
    /// fails by name rather than silently getting nothing. Recorded as a
    /// papered-over item in docs/bindings/TENSORBASE.md, not as an implementation.
    ///
    /// Inert is not the same as unconstrained, and that distinction was missed
    /// for as long as this comment has existed. `docs/training/BACKWARD2.md` §1.4
    /// measured the one place where this shim was *more* permissive than
    /// upstream rather than less: an integer tensor could be told to require
    /// gradients here and cannot upstream. `set_requires_grad` states that rule
    /// now, at the site upstream states it -- `tape.rs`'s `wrt_set` had been
    /// carrying it alone, one layer down, where docs/training/BACKWARD.md §4.1 records
    /// having to add it after the reverse walk asked for the derivative of a
    /// token id.
    requires_grad: bool,
    /// **Inert**, for the same reason as `requires_grad`: there is no autograd
    /// here, so nothing ever fires a backward hook. It is a real slot rather
    /// than a refusing property because `torch/_utils.py:246
    /// _rebuild_tensor_v2` *assigns* to it on every tensor of every
    /// `torch.load`, unconditionally and before anyone could have registered a
    /// hook -- so on this path the value written is always the empty
    /// `OrderedDict()` that `_rebuild_tensor_v2`'s own comment insists on
    /// ("we must give an EMPTY OrderedDict(), if you pass a None you'll run
    /// afoul #12219"). Refusing the assignment stopped `torch.load`; accepting
    /// it stores something nothing reads. Recorded in docs/models/CKPT.md §6 as
    /// papered over, not implemented.
    backward_hooks: Option<Py<PyAny>>,
    /// The accumulated gradient, and **no longer inert**.
    ///
    /// `docs/training/AUTOGRAD.md` §7 argued for leaving this a read-only `None`, and
    /// the argument was right at the time and is quoted here rather than
    /// paraphrased: *"making `.grad` writable while nothing writes to it would
    /// move the shim from 'honestly reports no gradient' to 'has a slot that is
    /// always empty'"*. `docs/training/BACKWARD.md` is what changed the antecedent --
    /// the tape writes here, so the slot is no longer always empty, and every
    /// `torch.optim` step in this shim reads it.
    ///
    /// A `Py<PyAny>` rather than a `PyTensorBase` because that is what
    /// `optimizer.zero_grad(set_to_none=True)` writes (`None`) and what a
    /// `Parameter`'s gradient is (a `Tensor`, i.e. a *subclass* instance whose
    /// Python identity a caller may hold on to).
    grad: Option<Py<PyAny>>,
    /// **The one autograd fact this shim computes rather than asserts.**
    ///
    /// `Some(op)` means "this tensor was produced by `op`, under grad mode,
    /// from an operand that required a gradient" -- which is exactly upstream's
    /// condition for `grad_fn is not None`, and therefore for `is_leaf` being
    /// `False`. `None` means leaf. docs/training/BACKWARD4.md §2 is why the two cannot
    /// be separated: upstream does not store `is_leaf`, it *is*
    /// `grad_fn is None`, so a truthful `is_leaf` and a truthful `grad_fn`
    /// nullness are one field and not two.
    ///
    /// It holds the aten op name and nothing else. There is no node, no
    /// `next_functions`, no saved operand and no `apply` -- `bootstrap.py`'s
    /// `_grad_fn_node` builds an opaque object from this string on read, whose
    /// only faithful attribute is its class name, because
    /// `torch/_tensor_str.py:646` reads `type(grad_fn).__name__` and is the
    /// only caller on an exercised path that reaches past nullness
    /// (docs/training/BACKWARD4.md §1.3). `Tensor.backward()` still refuses at
    /// `_ImperativeEngine.run_backward`; **this field is a description of what
    /// happened, not a promise that it can be undone.**
    from_op: Option<Box<str>>,
    /// `t.retain_grad()` was called. Unreachable before this round, because
    /// upstream's `param.is_leaf or param.retains_grad` short-circuited on an
    /// `is_leaf` that was always `True` -- docs/training/BACKWARD3.md §1.2.
    retains_grad: bool,
    /// **The `as_strided` write barrier's keep-alive handle.** `None` for every
    /// tensor that is not an `as_strided` result.
    ///
    /// `Some(barrier)` means this tensor was produced by
    /// `aten.as_strided.default`, which in this shim is a **gather** and not
    /// the two-way view upstream returns (docs/kernels/STRIDED.md). While it is held,
    /// `storage.rs`'s registry bars in-place writes to this tensor's storage
    /// *and* to the base's, so the divergence a gather would otherwise have --
    /// a write lost in either direction, silently -- is a named refusal
    /// instead.
    ///
    /// The field's only job is to own the `Arc`: dropping the last handle is
    /// what unregisters both keys, and holding it is what keeps their
    /// addresses reserved so that neither can be reused while it is still
    /// meaningful. docs/kernels/TAIL4.md §1.2 rejected an address-keyed poison set
    /// precisely because it had no such handle.
    strided: Option<std::sync::Arc<crate::storage::StridedBarrier>>,
    /// `torch._C._set_throw_on_mutable_data_ptr(t)` -- the per-tensor bit that
    /// makes `t.data_ptr()` refuse.
    ///
    /// `fake_tensor.py:943` sets it on every `FakeTensor` at construction: a
    /// fake tensor has no bytes, so handing back an address for them would be
    /// worse than refusing, and upstream would rather the caller fail at the
    /// `data_ptr()` than at whatever it did with the number.
    ///
    /// docs/graph/EXPORT.md §3.2 called this "Rust" and it was right about the
    /// reason: a Python side-table keyed by identity is a *different*
    /// guarantee. A `FakeTensor` is not reliably weak-referenceable, and the
    /// bit has to survive `Tensor._make_subclass`, which builds a new Python
    /// object around the same `PyTensorBase`. As a field it survives both by
    /// construction.
    ///
    /// `AtomicBool` rather than `bool` because the setter runs on an object
    /// Python already holds -- `_set_throw_on_mutable_data_ptr` takes `&self`,
    /// so there is no `&mut` to be had -- and `Cell` is not `Sync`.
    /// `Relaxed` is the right ordering: the bit guards nothing but its own
    /// read, and it is written once at construction before the tensor is
    /// shared.
    throw_on_mutable_data_ptr: std::sync::atomic::AtomicBool,
    /// `torch._C._set_warn_deprecated_on_mutable_data_ptr(t)` -- the *softer*
    /// half of the pair above, and a genuinely different behaviour rather than
    /// a second name for it.
    ///
    /// Upstream **returns the pointer and warns**; it does not refuse. Measured
    /// on 2.13.0, the warning is a `UserWarning` beginning "Accessing the data
    /// pointer of FakeTensor is deprecated". `fake_tensor.py` uses this one for
    /// tensors whose `data_ptr()` is legal-but-suspect and the throwing one for
    /// tensors that have no storage at all, so collapsing the two into a single
    /// refusal would turn a warning into an error for callers upstream still
    /// serves.
    warn_deprecated_on_mutable_data_ptr: std::sync::atomic::AtomicBool,
}

/// Hand-written rather than derived: `backward_hooks` is a `Py<PyAny>`, and
/// incrementing a Python refcount needs the interpreter attached, which
/// `#[derive(Clone)]` has no way to ask for.
impl Clone for PyTensorBase {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            inner: self.inner.clone(),
            tag: self.tag,
            requires_grad: self.requires_grad,
            backward_hooks: self.backward_hooks.as_ref().map(|h| h.clone_ref(py)),
            // Deliberately dropped, not cloned. A clone is a *new* tensor and
            // a gradient belongs to the leaf it was accumulated into; carrying
            // it across would make `p.clone().grad` report a gradient nothing
            // ever computed for that object. Upstream does the same -- a
            // non-leaf has no `.grad` at all.
            grad: None,
            // Dropped for the same reason, and it is upstream's answer too. A
            // clone is a *new* tensor: if it came out of `aten.clone.default`
            // the door marks it itself, and if it came out of
            // `Tensor._make_subclass` -- which is how every `nn.Parameter` is
            // born -- it is a leaf, exactly as `nn.Parameter(non_leaf)` is a
            // leaf upstream.
            from_op: None,
            retains_grad: false,
            // Carried, not dropped. A `PyTensorBase` clone points at the *same*
            // candle tensor and therefore at the same storage, so it is one
            // more handle on a barred buffer and not a new tensor. Dropping
            // the barrier here would let `y = x.as_strided(...)` be laundered
            // into a writable tensor by any path that clones the wrapper.
            strided: self.strided.clone(),
            // Carried, for `strided`'s reason rather than `grad`'s. The bit
            // says "these bytes are not real"; a clone points at the same
            // (absent) bytes, so laundering it through `.clone()` into a
            // tensor whose `data_ptr()` answers would defeat the refusal.
            throw_on_mutable_data_ptr: std::sync::atomic::AtomicBool::new(
                self.throw_on_mutable_data_ptr
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
            warn_deprecated_on_mutable_data_ptr: std::sync::atomic::AtomicBool::new(
                self.warn_deprecated_on_mutable_data_ptr
                    .load(std::sync::atomic::Ordering::Relaxed),
            ),
        })
    }
}

/// The refusal every read of a meta tensor's bytes ends at.
///
/// Upstream's own wording, character for character, measured on torch 2.13.0:
/// `torch.zeros(2, device="meta").tolist()`, `.cpu()`, `.to("cpu")` and
/// `torch.zeros(2).copy_(meta)` all raise
/// `NotImplementedError: Cannot copy out of meta tensor; no data!`.
pub fn no_data() -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err("Cannot copy out of meta tensor; no data!")
}

/// The refusal every dense read of a quantised tensor ends at.
///
/// Not upstream's wording, because upstream has no equivalent: its quantised
/// tensors *do* have dense storage (an `int8` buffer plus a scale), and
/// `.int_repr()` hands it over. A GGML block format has no such buffer -- the
/// bytes are interleaved scales and packed sub-byte quants -- so there is
/// nothing to hand over and the honest answer names the format and points at
/// the one operation that does produce numbers.
pub fn no_dense_storage(format: &str) -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(format!(
        "torch._C shim: this tensor is block-quantised ({format}); it has no dense \
         storage for a kernel to read. Use torch._C._dequantize(t) to materialise \
         float32, or torch._C._quantized_linear(x, w, b) to compute against it."
    ))
}

/// Why a kernel cannot read a Vulkan tensor's bytes.
///
/// The mirror of `no_dense_storage`, and the message says the same thing about
/// a different reason: the storage exists, it is simply not on this side of the
/// PCIe/unified boundary and not in a layout candle can address. It names the
/// way out (`.cpu()`) because there is exactly one.
pub fn no_host_storage() -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(
        "torch._C shim: this tensor is on the vulkan device; its storage is a \
         VkBuffer, not something a CPU kernel can read. Only the ops taught the \
         vulkan device by name compute on one (torch._C._vulkan_ops()); bring it \
         back with .cpu() for anything else. docs/devices/VULKAN3.md",
    )
}

/// The refusal that carries `Repr::Complex`'s safety, and the reason it is
/// worded the way it is.
///
/// A complex tensor *does* have two dense buffers behind it, and either one of
/// them would satisfy a caller's type. That is precisely why this cannot
/// return one: a kernel handed `re` alone computes a plausible answer with the
/// imaginary part dropped, which survives a smoke test, survives a shape check
/// and is only caught by an element-wise comparison nobody runs. So `tensor()`
/// refuses here, and the message says which half would have been lost rather
/// than "no storage", because the reader's next question is whether their gap
/// is the dtype or the operator. docs/kernels/COMPLEX2.md §2.
pub fn no_real_storage(tag: TorchDType) -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(format!(
        "torch._C shim: this is a {} tensor, held as a pair of real tensors \
         (torch._C shim has no complex candle dtype -- docs/kernels/COMPLEX.md §2). \
         Handing a kernel its real part alone would drop the imaginary part \
         and return plausible numbers, so there is no dense storage to read. \
         Only the ops taught the complex representation by name compute on \
         one (torch._C._complex_ops()); torch.view_as_real(z) is the way back \
         to a real tensor.",
        tag.name()
    ))
}

/// Identities for meta storages, handed out in sequence.
///
/// A counter rather than an address because a meta storage **has** no address
/// -- upstream answers `data_ptr() == 0` for every one of them. What the number
/// has to do is be distinct per storage and shared between a tensor and its
/// views, and a counter does both without pretending to be a pointer. It starts
/// at 1 so that `0` stays available as "not a meta storage", the same
/// convention `PyStorageBase::origin` already uses.
static META_STORAGE_IDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);

fn next_meta_storage_id() -> usize {
    META_STORAGE_IDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl PyTensorBase {
    /// The identity of the storage this tensor would have, if it is a meta
    /// tensor. `None` for every other representation -- a dense tensor's
    /// storage identity is candle's buffer address and comes from
    /// `storage_snapshot`.
    pub fn meta_storage_id(&self) -> Option<usize> {
        match &self.inner {
            Repr::Meta { storage_id, .. } => Some(*storage_id),
            _ => None,
        }
    }

    /// A tensor whose torch dtype is whatever candle is already storing.
    pub fn new(inner: Tensor) -> PyResult<Self> {
        let tag = TorchDType::from_storage(inner.dtype()).ok_or_else(|| {
            not_implemented(format!(
                "torch._C shim has no torch dtype for candle dtype: {}",
                inner.dtype().as_str()
            ))
        })?;
        Ok(Self {
            inner: Repr::Dense(inner),
            tag,
            requires_grad: false,
            backward_hooks: None,
            grad: None,
            from_op: None,
            retains_grad: false,
            strided: None,
            throw_on_mutable_data_ptr: std::sync::atomic::AtomicBool::new(false),
            warn_deprecated_on_mutable_data_ptr: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// A tensor on the `meta` device: shape and dtype, no allocation.
    ///
    /// Unlike `new`/`boolean` there is no dtype narrowing to do -- `meta`
    /// carries the torch tag directly and never has to be storable by candle.
    /// That is a real widening over the dense side: `torch.empty(2,
    /// dtype=torch.complex64, device="meta")` is representable here while its
    /// CPU counterpart is not, which is also true upstream on a build without a
    /// kernel for a dtype. docs/devices/META.md §6.
    pub fn meta(shape: Vec<usize>, tag: TorchDType) -> Self {
        Self::meta_with_storage_id(shape, tag, next_meta_storage_id())
    }

    /// A meta tensor sharing an existing meta storage identity.
    ///
    /// The view half of `meta`. Upstream's `view` shares storage, so its meta
    /// counterpart must answer the same `_cdata` as its input; allocating a
    /// fresh id there would make `meta_utils.py`'s `storage_memo` treat a
    /// tensor and its own view as two unrelated storages, which is the
    /// aliasing information the memo exists to preserve.
    pub fn meta_with_storage_id(shape: Vec<usize>, tag: TorchDType, storage_id: usize) -> Self {
        Self {
            inner: Repr::Meta { shape, storage_id },
            tag,
            requires_grad: false,
            backward_hooks: None,
            grad: None,
            from_op: None,
            retains_grad: false,
            strided: None,
            throw_on_mutable_data_ptr: std::sync::atomic::AtomicBool::new(false),
            warn_deprecated_on_mutable_data_ptr: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// The single entrance for the quantised representation.
    ///
    /// The tag is **the dtype this weight produces**, not a `q*` tag. Upstream
    /// would say `torch.qint8` here, and that was rejected on purpose: a
    /// `qint8` tag sends every reader down upstream's per-tensor-affine
    /// quantised path, which wants `q_scale()`/`q_zero_point()`/`int_repr()`
    /// and a single scale for the whole tensor -- none of which a GGML k-quant
    /// has (Q4K carries eight 6-bit sub-scales and two `f16` super-scales per
    /// 256 elements). It would also be a tag with no meaning for Q4K, there
    /// being no 4-bit torch dtype that is storable here (docs/graph/QUANT.md §2.1).
    ///
    /// So `.dtype` answers what comes out of `_dequantize`/`_quantized_linear`
    /// and `.is_quantized` answers that it is quantised; the *format* is a
    /// separate question with a separate answer, `_quantized_format()`. This
    /// is a narrowing against upstream and is recorded as one in
    /// docs/graph/QUANT2.md §4.
    pub fn quantized(inner: Arc<QTensor>, tag: TorchDType) -> Self {
        Self {
            inner: Repr::Quantized(inner),
            tag,
            requires_grad: false,
            backward_hooks: None,
            grad: None,
            from_op: None,
            retains_grad: false,
            strided: None,
            throw_on_mutable_data_ptr: std::sync::atomic::AtomicBool::new(false),
            warn_deprecated_on_mutable_data_ptr: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// The single entrance for the Vulkan representation.
    ///
    /// The tag is carried the way `quantized` carries one: the buffer is bytes
    /// and knows nothing about torch dtypes, so the tag is the authority. Only
    /// `float32` can get here -- `vulkan::check_dtype` refuses everything else
    /// at the factory, because there is one shader and it is f32.
    pub fn vulkan(inner: crate::vulkan::VkTensor, tag: TorchDType) -> Self {
        Self {
            inner: Repr::Vulkan(inner),
            tag,
            requires_grad: false,
            backward_hooks: None,
            grad: None,
            from_op: None,
            retains_grad: false,
            strided: None,
            throw_on_mutable_data_ptr: std::sync::atomic::AtomicBool::new(false),
            warn_deprecated_on_mutable_data_ptr: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// **The single entrance for the complex representation**, and the only
    /// place the arm's invariants are established.
    ///
    /// Mirrors `boolean()`'s role exactly: there is one way to attach a
    /// complex tag and it is this, so every complex tensor in the process has
    /// been through these four checks. A caller cannot assemble the arm
    /// directly -- `Repr` is public but `PyTensorBase::inner` is not.
    ///
    /// The tag is derived from the component dtype rather than accepted from
    /// the caller (`TorchDType::complex_for_component`), which is what makes
    /// "the tag agrees with the storage" true by construction rather than by
    /// discipline. The mismatched-shape and mismatched-dtype checks are
    /// internal errors in the sense that no op below can produce one -- they
    /// are here because the arm's every consumer reads `re` and `im` as
    /// parallel, and a violated invariant there is silent.
    pub fn complex(re: Tensor, im: Tensor) -> PyResult<Self> {
        if re.dtype() != im.dtype() {
            return Err(not_implemented(format!(
                "torch._C shim: a complex tensor's real and imaginary parts \
                 must have one dtype, got {} and {}",
                re.dtype().as_str(),
                im.dtype().as_str()
            )));
        }
        if re.dims() != im.dims() {
            return Err(not_implemented(format!(
                "torch._C shim: a complex tensor's real and imaginary parts \
                 must have one shape, got {:?} and {:?}",
                re.dims(),
                im.dims()
            )));
        }
        if !re.device().same_device(im.device()) {
            return Err(not_implemented(
                "torch._C shim: a complex tensor's real and imaginary parts \
                 must be on one device",
            ));
        }
        let tag = TorchDType::complex_for_component(re.dtype()).ok_or_else(|| {
            not_implemented(format!(
                "torch._C shim: no complex dtype over {} -- complex tensors \
                 here are pairs of half, float or double (docs/kernels/COMPLEX.md §3)",
                re.dtype().as_str()
            ))
        })?;
        Ok(Self {
            inner: Repr::Complex { re, im },
            tag,
            requires_grad: false,
            backward_hooks: None,
            grad: None,
            from_op: None,
            retains_grad: false,
            strided: None,
            throw_on_mutable_data_ptr: std::sync::atomic::AtomicBool::new(false),
            warn_deprecated_on_mutable_data_ptr: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// The two halves, for the ops `complex_ops` taught this arm by name. The
    /// mirror of `qtensor` and `vk_tensor`: nothing can be handed a real
    /// tensor here and treat it as a complex one.
    #[inline]
    pub fn complex_parts(&self, op: &str) -> PyResult<(&Tensor, &Tensor)> {
        match &self.inner {
            Repr::Complex { re, im } => Ok((re, im)),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: expected a complex tensor, got a {} one",
                self.tag.name()
            ))),
        }
    }

    /// Whether this tensor is held as a pair. Used by the two dispatch sites
    /// in `aten.rs` that have to choose between the real kernel and the
    /// complex one *before* parsing, and by nothing else -- every other site
    /// gets its refusal from `tensor()`.
    #[inline]
    pub fn is_complex_repr(&self) -> bool {
        matches!(self.inner, Repr::Complex { .. })
    }

    /// The single entrance for the `torch.bool` tag (BOOL.md §6.3 item 1).
    /// The caller is asserting the bytes are already normalised to 0/1;
    /// `BRAINWAVE_CHECK_BOOL=1` turns that assertion into a check.
    pub fn boolean(inner: Tensor) -> PyResult<Self> {
        if inner.dtype() != DType::U8 {
            return Err(not_implemented(format!(
                "torch._C shim: a torch.bool tensor stores as U8, got {}",
                inner.dtype().as_str()
            )));
        }
        if std::env::var("BRAINWAVE_CHECK_BOOL").is_ok_and(|v| v != "0") {
            let max = inner
                .flatten_all()
                .and_then(|t| t.max(0))
                .and_then(|t| t.to_scalar::<u8>())
                .map_err(|e| candle_err("bool invariant check", e))?;
            if max > 1 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "torch._C shim: torch.bool invariant violated -- a byte under                      a bool tag is {max}, not 0 or 1 (BOOL.md §6.3)"
                )));
            }
        }
        Ok(Self {
            inner: Repr::Dense(inner),
            tag: TorchDType::Bool,
            requires_grad: false,
            backward_hooks: None,
            grad: None,
            from_op: None,
            retains_grad: false,
            strided: None,
            throw_on_mutable_data_ptr: std::sync::atomic::AtomicBool::new(false),
            warn_deprecated_on_mutable_data_ptr: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// The storage. **Refuses on `meta`**, which has none.
    ///
    /// Every kernel in `aten.rs` reads its inputs through this, so the enum's
    /// one `?` is what makes "no kernel computes on a meta tensor" a property
    /// of the type rather than a rule 96 kernels have to remember. The door's
    /// meta gate (`aten.rs::check_meta`) refuses first and with a better
    /// message; this is the backstop under it, and it is the reason a kernel
    /// added tomorrow without being told about meta is safe.
    #[inline]
    pub fn tensor(&self) -> PyResult<&Tensor> {
        match &self.inner {
            Repr::Dense(tensor) => Ok(tensor),
            Repr::Meta { .. } => Err(no_data()),
            Repr::Quantized(q) => Err(no_dense_storage(crate::quant::format_name(q.dtype()))),
            Repr::Vulkan(_) => Err(no_host_storage()),
            // **The arm that carries the whole representation's safety.**
            // Returning `re` here would compile, would type-check at all ~400
            // call sites, and would make every existing kernel silently
            // compute the real part of a complex expression. See
            // `no_real_storage`.
            Repr::Complex { .. } => Err(no_real_storage(self.tag)),
        }
    }

    /// The Vulkan storage, for the ops `vulkan::dispatch` taught this device by
    /// name. The mirror of `qtensor` and refusing for the same reason: nothing
    /// can be handed a CPU tensor here and treat it as a device buffer.
    #[inline]
    pub fn vk_tensor(&self, op: &str) -> PyResult<&crate::vulkan::VkTensor> {
        match &self.inner {
            Repr::Vulkan(v) => Ok(v),
            _ => Err(not_implemented(format!(
                "{op}: expected a tensor on the vulkan device, got one on {}",
                self.device_label().__str__()
            ))),
        }
    }

    #[inline]
    pub fn repr(&self) -> &Repr {
        &self.inner
    }

    #[inline]
    pub fn is_meta_repr(&self) -> bool {
        matches!(self.inner, Repr::Meta { .. })
    }

    /// The quantised storage, for the ops that were taught this arm by name.
    /// Refuses on the other two, so `quant.rs` cannot be handed a dense tensor
    /// by accident and silently treat it as a weight.
    #[inline]
    pub fn qtensor(&self, op: &str) -> PyResult<&Arc<QTensor>> {
        match &self.inner {
            Repr::Quantized(q) => Ok(q),
            Repr::Dense(_) | Repr::Meta { .. } | Repr::Vulkan(_) | Repr::Complex { .. } => Err(not_implemented(format!(
                "{op}: expected a block-quantised tensor (torch._C._quantize), \
                 got a {} one",
                match &self.inner {
                    Repr::Dense(_) => "dense",
                    Repr::Vulkan(_) => "vulkan",
                    Repr::Complex { .. } => "complex",
                    _ => "meta",
                }
            ))),
        }
    }

    /// The shape, for either representation. This is the half of a tensor meta
    /// still has.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        match &self.inner {
            Repr::Dense(tensor) => tensor.dims(),
            Repr::Meta { shape, .. } => shape,
            Repr::Quantized(q) => q.shape().dims(),
            Repr::Vulkan(v) => &v.shape,
            // `re.dims()` **is** the complex tensor's shape -- that is the
            // whole reason §3.2 chose a pair over interleaving. There is no
            // trailing `2` to hide here, and if there were, this is the site
            // that would have to lie about it.
            Repr::Complex { re, .. } => re.dims(),
        }
    }

    #[inline]
    pub fn elem_count(&self) -> usize {
        match &self.inner {
            Repr::Dense(tensor) => tensor.elem_count(),
            Repr::Meta { shape, .. } => shape.iter().product(),
            Repr::Quantized(q) => q.shape().elem_count(),
            Repr::Vulkan(v) => v.elem_count(),
            // `numel` counts *complex* elements, not floats, which is
            // upstream's answer: `torch.view_as_complex(torch.ones(3,2))
            // .numel()` is 3. The buffers hold twice that many floats, and
            // `element_size` below is 8 rather than 4 for exactly that reason,
            // so `numel * element_size` still sizes the storage correctly.
            Repr::Complex { re, .. } => re.elem_count(),
        }
    }

    /// The device label, for either representation.
    ///
    /// For a dense tensor this is `PyDevice::from_candle`, with the lossy
    /// index reconstruction that documents. For a meta tensor there is no
    /// candle handle to reconstruct from and none is needed: `meta` is a
    /// constant, because upstream normalises every index off it (measured --
    /// `device="meta:7"` reports `device(type='meta')`).
    pub fn device_label(&self) -> PyDevice {
        match &self.inner {
            Repr::Dense(tensor) => PyDevice::from_candle(tensor.device()),
            Repr::Meta { .. } => PyDevice::meta(),
            // A `QTensor` owns a real device, and `device()` returns it by
            // value rather than by reference (its storage enum holds the
            // backend handle, not a `&Device`), so this binds a temporary
            // rather than borrowing like the dense arm.
            Repr::Quantized(q) => PyDevice::from_candle(&q.device()),
            // Not `from_candle`: there is no candle handle to reconstruct
            // from, which is the entire content of this arm. There is one
            // Vulkan device and no index to invent, so the label is a
            // constant, exactly as `meta`'s is.
            Repr::Vulkan(_) => crate::vulkan::label(),
            // Both halves are on one device (checked in `complex`), so either
            // answers. `re` by convention.
            Repr::Complex { re, .. } => PyDevice::from_candle(re.device()),
        }
    }

    pub fn tag(&self) -> TorchDType {
        self.tag
    }

    /// Does this tensor have bytes behind it? `torch._C._has_storage`.
    ///
    /// Upstream this is `unsafeGetTensorImpl()->has_storage()`, and the one
    /// thing it is false for on CPU is a meta tensor -- which is exactly the
    /// distinction `Repr` was made for (docs/devices/META.md §3). A quantised tensor
    /// owns blocks and answers `true`, as upstream's quantised tensors do;
    /// what it cannot do is hand those blocks over as a flat storage, and that
    /// refusal belongs to `storage_snapshot`, one question later.
    pub fn has_storage(&self) -> bool {
        !matches!(self.inner, Repr::Meta { .. })
    }

    /// **The whole candle buffer this tensor's layout addresses**, as
    /// little-endian bytes, with the identity of that buffer.
    ///
    /// Not `to_le_bytes`, and the difference is the entire point. That function
    /// reads *the view* -- row-major, `numel` elements, offset and stride
    /// resolved. This reads *the storage*, which is what upstream's
    /// `untyped_storage()` is: `torch.save` records a tensor as
    /// `(storage, storage_offset, size, stride)` and expects those three
    /// numbers to index into the bytes it was handed. Handing over a
    /// materialised view instead would produce a file whose stride and offset
    /// were lies about its own payload -- readable, silently wrong, which is
    /// the failure shape docs/models/CKPT.md §4 and §5 are both about.
    ///
    /// The second return value is the address of candle's `Storage` inside its
    /// `Arc<RwLock<_>>`. Two tensors that share a buffer -- `x` and `x.t()`,
    /// `x` and `x[1]` -- give the same number, and that is what lets a save
    /// preserve storage sharing across a copy (`storage.rs::origin`). It is an
    /// identity, never dereferenced, and it is only meaningful while the
    /// tensor is alive -- which on the save path it is, since the object being
    /// pickled holds it.
    ///
    /// Half-width floats are read here even though `to_le_bytes` refuses them.
    /// That refusal's stated reason -- "this crate does not depend on `half`"
    /// -- stopped being true when `reduced.rs` took the dependency (Cargo.toml
    /// names `half = "2.7"`), and leaving it in place here would mean no
    /// `bfloat16` weight could be saved. `to_le_bytes` itself is left alone: it
    /// is `from_le_bytes`'s inverse for `aten.view.dtype`, and changing what
    /// that op accepts is a separate decision with its own golden cases.
    pub fn storage_snapshot(&self, op: &str) -> PyResult<(Vec<u8>, usize)> {
        let tensor = self.tensor()?;
        if !tensor.device().is_cpu() {
            return Err(not_implemented(format!(
                "{op}: reading a tensor's storage is implemented for the CPU \
                 backend only in torch._C shim; this tensor is on {}",
                self.device_label().__str__()
            )));
        }
        let (guard, _layout) = tensor.storage_and_layout();
        let storage: &candle_core::Storage = &guard;
        // The identity, taken before the match so that it does not depend on
        // which arm the storage is: the address of the `Storage` inside the
        // `Arc<RwLock<Storage>>` every alias of this tensor shares.
        let identity = storage as *const candle_core::Storage as usize;
        let candle_core::Storage::Cpu(cpu) = storage else {
            return Err(not_implemented(format!(
                "{op}: torch._C shim can read the storage of a CPU tensor only"
            )));
        };
        macro_rules! pour {
            ($v:expr, $ty:ty) => {{
                let v = $v;
                let mut out = Vec::with_capacity(v.len() * std::mem::size_of::<$ty>());
                for x in v.iter() {
                    out.extend_from_slice(&x.to_le_bytes());
                }
                out
            }};
        }
        macro_rules! pour_bits {
            ($v:expr) => {{
                let v = $v;
                let mut out = Vec::with_capacity(v.len() * 2);
                for x in v.iter() {
                    out.extend_from_slice(&x.to_bits().to_le_bytes());
                }
                out
            }};
        }
        let bytes = match cpu {
            CpuStorage::U8(v) => v.clone(),
            CpuStorage::U32(v) => pour!(v, u32),
            CpuStorage::I16(v) => pour!(v, i16),
            CpuStorage::I32(v) => pour!(v, i32),
            CpuStorage::I64(v) => pour!(v, i64),
            CpuStorage::BF16(v) => pour_bits!(v),
            CpuStorage::F16(v) => pour_bits!(v),
            CpuStorage::F32(v) => pour!(v, f32),
            CpuStorage::F64(v) => pour!(v, f64),
            _ => {
                return Err(not_implemented(format!(
                    "{op}: torch._C shim cannot read the raw bytes of a \
                     torch.{} tensor's candle storage -- \
                     tensor.rs::storage_snapshot names the storage kinds it \
                     can read, and this is not one of them",
                    self.tag.name()
                )))
            }
        };
        // The one thing that would make a saved file quietly wrong: a byte
        // count that is not a whole number of the elements `storage_offset`
        // and `stride` are counted in. Both are element counts against
        // `tag.itemsize()`, so a disagreement between candle's element width
        // and the torch tag's would produce a readable, misaligned file.
        let width = self.tag.itemsize();
        if width != 0 && bytes.len() % width != 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: torch._C shim internal error -- {} storage bytes is not \
                 a whole number of torch.{} elements ({width} bytes each)",
                bytes.len(),
                self.tag.name()
            )));
        }
        Ok((bytes, identity))
    }

    /// **Rebinding, not writing.** The wrapper stops pointing at one candle
    /// tensor and starts pointing at another; the buffer it used to point at
    /// is untouched, so an alias taken before the call keeps the old values.
    ///
    /// It takes a whole `PyTensorBase` rather than a bare candle tensor so
    /// that the replacement has already been through `new()` or `boolean()` --
    /// BOOL.md §6.3's rule that only one constructor may attach the `bool`
    /// tag survives a mutating API only if the mutation cannot bypass it.
    ///
    /// **This is no longer the in-place ops' primitive.** They use
    /// `write_into` (below), which writes through the layout into the existing
    /// buffer, so `y = x.detach(); y.fill_(0)` now reaches `x` the way it does
    /// upstream. What is left here is the two callers for which *rebinding is
    /// the operation*, and where upstream rebinds too:
    ///
    ///   * `TensorBase.set_` -- adopts a different storage, a different shape
    ///     and possibly a different dtype; there is no "existing buffer" to
    ///     write into, since the point is to leave it.
    ///   * `tensor.data = other` -- upstream swaps the `TensorImpl`, so a view
    ///     taken before the assignment does not follow it there either
    ///     (docs/devices/DEVICE_ABS.md §4).
    ///
    /// Anything that means "the receiver's values change but the receiver
    /// stays the same tensor" must not come here. docs/kernels/VIEWS.md §6.
    /// Mark this tensor as an `as_strided` result and bar its storage, and the
    /// base's, from in-place writes for as long as it lives.
    ///
    /// Called from exactly one place, `aten.rs::as_strided_default`, and taking
    /// the `Arc` rather than building it here keeps `storage.rs` the only file
    /// that knows how the registry is keyed. docs/kernels/STRIDED.md §2.
    pub fn bar_writes_as_strided_view(
        &mut self,
        barrier: std::sync::Arc<crate::storage::StridedBarrier>,
    ) {
        self.strided = Some(barrier);
    }

    pub fn replace_with(&mut self, replacement: PyTensorBase) {
        self.inner = replacement.inner;
        self.tag = replacement.tag;
    }

    /// **The write primitive for every in-place op**: put `source`'s values
    /// into the buffer this wrapper already points at, through this wrapper's
    /// layout.
    ///
    /// The difference from `replace_with` is the whole of docs/kernels/VIEWS.md §6.
    /// `select.int` and `slice.Tensor` (step 1) return tensors that share
    /// storage with their input -- candle's `narrow`/`squeeze` clone the
    /// storage `Arc` and rebuild only the `Layout` -- so a write that lands in
    /// the *layout* is seen by the base and a write that swaps the wrapper is
    /// not. `x[0] = 3.0` is the visible case, but every alias in the shim
    /// (`detach`, `alias`, `unsqueeze`, `view`) has the same question, and
    /// they now all answer it the same way.
    ///
    /// **Contract, checked rather than assumed** -- an in-place op may change
    /// values and nothing else:
    ///
    ///   * `source` has this tensor's shape exactly (no broadcasting here; the
    ///     kernels broadcast into the receiver's shape before they call);
    ///   * `source` has this tensor's *candle* dtype exactly (the kernels cast
    ///     into it, because in-place cannot widen);
    ///   * the torch tag is this tensor's and does not move -- so the `bool`
    ///     tag cannot be attached or dropped by a write.
    ///
    /// A violation is a defect in the calling kernel, so it raises with the
    /// op's name rather than being coerced into agreement.
    ///
    /// **All three are unreachable today, and that is the measured result
    /// rather than an assumption.** Every in-place kernel already broadcast
    /// into the receiver's shape and cast into its dtype, so candle refuses
    /// first, with its own wording, on every input that would reach them:
    /// `copy_((2,), (2,2))`, `add_((2,1), (2,2))` and
    /// `masked_fill_((4,1), mask (4,2))` all stop at `broadcast_as`. The one
    /// input that *did* reach the tag check -- `bool_tensor.clamp_(0, 5)`,
    /// which produced a `uint8` replacement and would have retagged the
    /// receiver -- now refuses at the door with upstream's own message, so
    /// this is a backstop under a named refusal rather than the refusal
    /// itself. Same layering as `check_meta` over `tensor()`.
    ///
    /// The consequence for testing is stated rather than hidden: **deleting
    /// these checks changes nothing observable** (measured -- 3075/3075 and
    /// 229 smoke tests green with the shape and tag checks removed). They
    /// exist for the kernel that has not been written yet, and a test for
    /// them would have to be a kernel that violates the contract, which the
    /// public API cannot produce.
    ///
    /// **Reads before it writes.** `source` is copied out into an owned
    /// `CpuStorage` first, and only then is the destination's lock taken. That
    /// is not a tidiness choice, it is what makes `x[0:2] = x[1:3]` mean what
    /// it means: the two sides alias one buffer, and a streaming copy would
    /// read values it had already overwritten. It is also what keeps the pair
    /// off candle's `inplace_op2`, whose `self.storage_mut()` and
    /// `rhs.storage()` are a write lock and a read lock on the *same*
    /// `RwLock` when the operands alias -- a deadlock, not an error.
    ///
    /// **No `unsafe`.** candle holds storage in `Arc<RwLock<Storage>>` and
    /// `Tensor::inplace_op1` takes the write lock, so aliasing-XOR-mutability
    /// is enforced at runtime by the lock rather than by a raw pointer. That
    /// is why this takes `&self` and not `&mut self`: the mutation is
    /// candle's interior mutability, and the Python-level `RefCell` borrow the
    /// caller holds stays shared.
    pub fn write_into(
        &self,
        op: &str,
        source: &PyTensorBase,
        overlap: Overlap,
    ) -> PyResult<()> {
        let dest = self.tensor()?;
        let src = source.tensor()?;

        if source.tag != self.tag {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: torch._C shim internal error -- an in-place op tried to \
                 write a torch.{} value into a torch.{} tensor. In-place ops \
                 cast into the receiver's dtype; changing it is `replace_with`'s \
                 job, not this one's (tensor.rs::write_into)",
                source.tag.name(),
                self.tag.name()
            )));
        }
        if dest.dims() != src.dims() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: torch._C shim internal error -- an in-place op computed a \
                 {:?} replacement for a {:?} receiver (tensor.rs::write_into)",
                src.dims(),
                dest.dims()
            )));
        }
        if dest.dtype() != src.dtype() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: torch._C shim internal error -- an in-place op computed a \
                 {} replacement for a {} receiver (tensor.rs::write_into)",
                src.dtype().as_str(),
                dest.dtype().as_str()
            )));
        }
        if !dest.device().is_cpu() {
            return Err(not_implemented(format!(
                "{op}: writing through a view is implemented for the CPU backend \
                 only in torch._C shim; this tensor is on {}",
                self.device_label().__str__()
            )));
        }
        // **The `as_strided` write barrier.** docs/kernels/STRIDED.md §2.
        //
        // This is the single write door -- `aten.rs::write_back` is the only
        // caller of this function and this function is the only thing in the
        // crate that writes through a tensor's layout -- so a check here sees
        // every in-place op there is, on the receiver *and* on every alias of
        // it, because the key is the storage and not the wrapper.
        //
        // It fires when the destination's storage is either an `as_strided`
        // result's or the base one was taken of, which are the two directions
        // upstream's genuine view propagates in and a gather propagates in
        // neither. Refusing is a narrowing of upstream, which allows both; a
        // gather that accepted them would be *wrong* at upstream, silently, in
        // the base's values. §4 records the narrowing and the one case it does
        // not reach.
        //
        // Nullifying this block is what docs/kernels/STRIDED.md 5 measured, and it is
        // the demonstration docs/kernels/COMPLEX2.md set as the standard for calling a
        // guard real: with `if false &&` in front of the call and nothing else
        // changed, `x.as_strided((2,3),(3,1)).fill_(7.)` returns a filled view
        // and leaves `x` at `[0., 1., 2., ...]` where upstream leaves it all
        // 7s -- a **stale base**, with no exception raised. That build fails 2
        // golden cases and 4 tests in `pytests/test_strided.py`, which is the
        // other half of the check: a guard nothing goes red for is a guard
        // nobody is exercising.
        if crate::storage::write_is_barred(&dest) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: this tensor shares storage with the result of \
                 aten.as_strided.default, or is the tensor one was taken \
                 of. torch._C shim materialises as_strided as a gather \
                 (candle 0.11.0 has no public constructor for a Tensor over \
                 an existing storage with arbitrary strides), so it is a \
                 READ-ONLY view: upstream propagates writes between the base \
                 and the view in both directions and this copy propagates \
                 them in neither. Refusing rather than answering, because \
                 the alternative is the right shape and dtype with the wrong \
                 values. Drop the as_strided result to lift this, or \
                 .clone() before writing. docs/kernels/STRIDED.md"
            )));
        }

        if overlap == Overlap::Refuse && has_internal_overlap(dest.layout()) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "unsupported operation: more than one element of the written-to \
                 tensor refers to a single memory location. Please clone() the \
                 tensor before performing the operation.",
            ));
        }

        // Read first, and let every lock go before the write starts. See the
        // aliasing note above -- this line is the reason `x[0:2] = x[1:3]`
        // is correct rather than half-overwritten.
        let payload = flat_storage(op, src)?;
        dest.inplace_op1(&WriteThrough { payload })
            .map_err(|e| candle_err(op, e))
    }
}

/// Whether an in-place op may write into a destination that addresses the
/// same storage element twice -- an *expanded* tensor, whose broadcast axes
/// have stride 0.
///
/// **Upstream is split on this and the split is measured, not guessed**
/// (torch 2.13.0, `x = torch.tensor([1.,2.]).reshape(1,2).expand(3,2)`):
///
/// | op | upstream |
/// |---|---|
/// | `fill_.Scalar`, `zero_` | writes; every position gets the same value |
/// | `masked_fill_`, `index_put_` | writes, with a deprecation warning |
/// | `fill_.Tensor`, `copy_`, `add_`, `relu_`, `clamp_`, `div_`, `uniform_`, `normal_` | **raises** |
///
/// So this is not a property of the destination alone and cannot be decided
/// here; the op decides, in `aten.rs::write_back`, from the same table. What
/// this file owns is the *detection*, which has to be upstream's rule and not
/// a stricter one -- see `has_internal_overlap`.
///
/// Before write-through the question could not arise: every in-place op
/// replaced the wrapper, so writing "through" an expanded tensor wrote through
/// nothing. Silently letting the last write win would be a new divergence in
/// the direction this shim refuses -- upstream raising where this computes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Overlap {
    /// Upstream raises. So does this.
    Refuse,
    /// Upstream writes. Every position the walk visits more than once is
    /// written the same value by these ops, so last-write-wins is not merely
    /// tolerated, it is the answer.
    Allow,
}

/// Does this layout address one storage element more than once?
///
/// **Upstream's own rule, including its deliberate incompleteness**
/// (`c10::has_internal_overlap`): dense and non-overlapping is `No`, a stride
/// of 0 on an axis longer than 1 is `Yes`, and *anything else is `TooHard` and
/// permitted*. Reproducing the conservative half matters as much as the
/// positive half -- a stricter test would refuse strided views that upstream
/// writes into happily, which is a divergence in the other direction.
fn has_internal_overlap(layout: &Layout) -> bool {
    layout
        .stride()
        .iter()
        .zip(layout.dims().iter())
        .any(|(stride, dim)| *stride == 0 && *dim > 1)
}

/// `source`, read out row-major into an owned buffer.
///
/// Owned, not borrowed: the returned value must outlive every lock on
/// `source`'s storage, because the caller takes a write lock on a buffer that
/// may be the same one. See `write_into`.
fn flat_storage(op: &str, source: &Tensor) -> PyResult<CpuStorage> {
    let flat = source
        .contiguous()
        .and_then(|t| t.flatten_all())
        .map_err(|e| candle_err(op, e))?;
    macro_rules! pour {
        ($arm:ident, $ty:ty) => {
            CpuStorage::$arm(flat.to_vec1::<$ty>().map_err(|e| candle_err(op, e))?)
        };
    }
    Ok(match flat.dtype() {
        DType::U8 => pour!(U8, u8),
        DType::U32 => pour!(U32, u32),
        DType::I16 => pour!(I16, i16),
        DType::I32 => pour!(I32, i32),
        DType::I64 => pour!(I64, i64),
        DType::BF16 => pour!(BF16, half::bf16),
        DType::F16 => pour!(F16, half::f16),
        DType::F32 => pour!(F32, f32),
        DType::F64 => pour!(F64, f64),
        // `float8_e4m3fn` joined the list in docs/numerics/FLOAT8C.md §3. `to_vec1`
        // reads the storage slice directly -- it never widens, so it does not
        // touch candle's non-terminating `F8E4M3 -> f64` arm (§1), which is why
        // this arm is a two-line addition rather than a conversion.
        DType::F8E4M3 => pour!(F8E4M3, float8::F8E4M3),
        other => {
            return Err(not_implemented(format!(
                "{op}: torch._C shim cannot write through a view of candle dtype \
                 {other:?} -- tensor.rs::flat_storage names the dtypes it can \
                 read, and this is not one of them"
            )))
        }
    })
}

/// The `InplaceOp1` behind `write_into`.
///
/// candle hands `cpu_fwd` the destination's storage *and its `Layout`*, which
/// is the only public surface in the crate that does. `slice_set` is the other
/// public write path and it cannot serve here: it requires both sides
/// contiguous and it *refuses a pair that shares storage*, which is precisely
/// `x[0:2] = x[1:3]`. docs/kernels/VIEWS.md §6.2 records what was rejected and why.
struct WriteThrough {
    /// Row-major, already read out of the source. Same length and dtype as the
    /// destination view's element count and dtype -- `write_into` checked.
    payload: CpuStorage,
}

impl InplaceOp1 for WriteThrough {
    fn name(&self) -> &'static str {
        "torch._C shim: write_into"
    }

    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> candle_core::Result<()> {
        macro_rules! scatter {
            ($dst:expr, $src:expr) => {
                write_strided($dst, $src, layout)
            };
        }
        match (storage, &self.payload) {
            (CpuStorage::U8(d), CpuStorage::U8(s)) => scatter!(d, s),
            (CpuStorage::U32(d), CpuStorage::U32(s)) => scatter!(d, s),
            (CpuStorage::I16(d), CpuStorage::I16(s)) => scatter!(d, s),
            (CpuStorage::I32(d), CpuStorage::I32(s)) => scatter!(d, s),
            (CpuStorage::I64(d), CpuStorage::I64(s)) => scatter!(d, s),
            (CpuStorage::BF16(d), CpuStorage::BF16(s)) => scatter!(d, s),
            (CpuStorage::F16(d), CpuStorage::F16(s)) => scatter!(d, s),
            (CpuStorage::F32(d), CpuStorage::F32(s)) => scatter!(d, s),
            (CpuStorage::F64(d), CpuStorage::F64(s)) => scatter!(d, s),
            (CpuStorage::F8E4M3(d), CpuStorage::F8E4M3(s)) => scatter!(d, s),
            // `write_into` compared the two candle dtypes before building this,
            // so a mismatch here is a defect in that check rather than a
            // reachable input.
            _ => Err(candle_core::Error::Msg(
                "torch._C shim: write_into reached cpu_fwd with mismatched \
                 storage dtypes"
                    .to_string(),
            )),
        }
    }
}

/// Row-major elements of `src` into the positions `layout` addresses in `dst`.
///
/// The walk is the odometer `gather_strided` reads with, run the other way. It
/// is written out rather than borrowed from candle because `StridedIndex`'s
/// constructors are `pub(crate)`; `Layout`'s dims, strides and offset are not,
/// so the arithmetic is reproducible from outside the crate.
///
/// **Bounds are proved once, before the loop, rather than per element.** The
/// furthest position the walk can reach is `start_offset + sum((dim-1) *
/// stride)`, since every index runs `0..dim` and every stride is
/// non-negative (candle's strides are `usize`). Checking that one number means
/// no write in the loop can be out of range -- and the check is what turns a
/// wrong layout into an error rather than a corrupted neighbouring tensor,
/// which is the failure this whole file exists to prevent.
fn write_strided<T: Copy>(dst: &mut [T], src: &[T], layout: &Layout) -> candle_core::Result<()> {
    let dims = layout.dims();
    let stride = layout.stride();
    let numel: usize = dims.iter().product();
    if src.len() != numel {
        return Err(candle_core::Error::Msg(format!(
            "torch._C shim: write_into has {} values for a {numel}-element view",
            src.len()
        )));
    }
    if numel == 0 {
        return Ok(());
    }
    let start = layout.start_offset();
    let reach = dims
        .iter()
        .zip(stride.iter())
        .map(|(d, s)| (d - 1) * s)
        .sum::<usize>();
    if start + reach >= dst.len() {
        return Err(candle_core::Error::Msg(format!(
            "torch._C shim: write_into's view (offset {start}, dims {dims:?}, \
             stride {stride:?}) reaches element {} of a storage holding {}",
            start + reach,
            dst.len()
        )));
    }

    // The contiguous case is the common one (`x[0]`, `x[1:3]`, and every
    // whole-tensor in-place op) and it is a memcpy. It is not merely an
    // optimisation: the odometer below would give the same answer, so this
    // branch is checked against it by every non-contiguous case in the suite
    // sharing the same expectations.
    if layout.is_contiguous() {
        dst[start..start + numel].copy_from_slice(src);
        return Ok(());
    }

    let rank = dims.len();
    let mut index = vec![0usize; rank];
    let mut offset = start;
    for value in src.iter() {
        dst[offset] = *value;
        for d in (0..rank).rev() {
            index[d] += 1;
            if index[d] < dims[d] {
                offset += stride[d];
                break;
            }
            offset -= (dims[d] - 1) * stride[d];
            index[d] = 0;
        }
    }
    Ok(())
}

/// A tensor from a little-endian payload, under a torch dtype tag.
///
/// The one place raw checkpoint bytes become a tensor. `torch.frombuffer` (the
/// safetensors path) and `TensorBase.set_` (the `torch.load` path) both come
/// here, so the two readers cannot disagree about a dtype.
///
/// Endianness is not checked, and cannot be from here: the caller has the
/// container's byte-order field (safetensors is little-endian by
/// specification; `torch.save` writes a `byteorder` record and
/// `torch/serialization.py` byteswaps before this point). Every target this
/// crate builds for is little-endian.
pub fn from_le_bytes(
    op: &str,
    bytes: &[u8],
    shape: &[usize],
    tag: TorchDType,
) -> PyResult<PyTensorBase> {
    let device = candle_core::Device::Cpu;

    // `torch.bool`: candle stores it as `U8`, and the 0/1 invariant
    // (BOOL.md §6.3) has to hold by construction. Raw checkpoint bytes under a
    // bool tag are not guaranteed normalised, so they are reduced with `!= 0`
    // -- which is exactly what torch guarantees a bool tensor *reads* as
    // (BOOL.md §2.6), so no value changes. Same reduction `_tensor_from_flat`
    // makes, for the same reason.
    if tag == TorchDType::Bool {
        let normalised: Vec<u8> = bytes.iter().map(|b| u8::from(*b != 0)).collect();
        let tensor =
            Tensor::from_vec(normalised, shape.to_vec(), &device).map_err(|e| candle_err(op, e))?;
        return PyTensorBase::boolean(tensor);
    }

    // `storage()` refuses by name for the dtypes candle cannot hold -- `int8`,
    // `uint16`, `uint64`, the complex family, and the fourteen sub-byte integer
    // tags. Upstream accepts several of them (measured: `torch.frombuffer` with
    // `torch.int8` and `torch.uint16` both return tensors), so this is a real
    // narrowing of the surface, and refusing loudly is the point. A checkpoint
    // in one of those dtypes stops here with the dtype in the message instead
    // of being reinterpreted as something else.
    let storage = tag.storage().ok_or_else(|| {
        not_implemented(format!(
            "{op}: dtype not storable by the candle backend in torch._C shim: torch.{}",
            tag.name()
        ))
    })?;
    let tensor =
        Tensor::from_raw_buffer(bytes, storage, shape, &device).map_err(|e| candle_err(op, e))?;
    PyTensorBase::new(tensor)
}

/// The little-endian payload of a tensor, row-major -- `from_le_bytes`'s inverse.
///
/// The pair is used by `aten::view.dtype`, which reinterprets one dtype's bytes
/// as another's. Going out through bytes and back in through `from_le_bytes`
/// rather than converting element-wise is what makes the reinterpretation
/// *exact*: `view` is a bit-level operation, and any route that passed through
/// a numeric type would round.
///
/// Written as the mirror of `from_le_bytes` on purpose, including the dtype
/// list -- the two are only usable together if they agree about which tags this
/// build can hold, and `storage()` is the single place that decides.
///
/// `torch.bool` is emitted as the 0/1 bytes candle stores, which is what
/// upstream's bool storage holds too (BOOL.md §2.6): measured, a `bool` tensor
/// viewed as `uint8` upstream gives exactly 0s and 1s.
///
/// **The half-width floats are refused here, and the asymmetry is deliberate.**
/// Getting at an `f16`/`bf16` tensor's bits through candle means naming the
/// `half` crate's types, and this crate does not depend on `half` -- it arrives
/// only as candle's own dependency, and CANDLE_DEPS.md is about not acquiring
/// dependencies by accident. The refusal costs nothing on the path that
/// exists: a checkpoint arrives as *bytes*, so `bf16` weights go
/// `uint8 -> bf16` through `from_le_bytes`, which reads a raw buffer and never
/// asks this function anything. Only the reverse direction --
/// `bf16_tensor.view(torch.uint8)` -- is closed, and nothing has reached it.
pub fn to_le_bytes(op: &str, tensor: &Tensor) -> PyResult<Vec<u8>> {
    let flat = tensor
        .flatten_all()
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(op, e))?;
    macro_rules! pour {
        ($ty:ty) => {{
            let v = flat.to_vec1::<$ty>().map_err(|e| candle_err(op, e))?;
            let mut out = Vec::with_capacity(v.len() * std::mem::size_of::<$ty>());
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }};
    }
    Ok(match flat.dtype() {
        DType::U8 => flat.to_vec1::<u8>().map_err(|e| candle_err(op, e))?,
        DType::U32 => pour!(u32),
        DType::I16 => pour!(i16),
        DType::I32 => pour!(i32),
        DType::I64 => pour!(i64),
        DType::F32 => pour!(f32),
        DType::F64 => pour!(f64),
        other => {
            return Err(not_implemented(format!(
                "{op}: torch._C shim cannot read the raw bytes of candle dtype \
                 {other:?} -- reaching its bit pattern means naming the `half` \
                 crate's types, which this crate deliberately does not depend on \
                 (see tensor.rs::to_le_bytes). Reading a checkpoint *into* this \
                 dtype goes the other way, through `from_le_bytes`, and works"
            )))
        }
    })
}

/// The contiguous (row-major) stride for a shape, in elements.
fn contiguous_stride(size: &[usize]) -> Vec<i64> {
    let mut stride = vec![1i64; size.len()];
    for i in (0..size.len().saturating_sub(1)).rev() {
        stride[i] = stride[i + 1] * size[i + 1] as i64;
    }
    stride
}

/// Read a `(storage_offset, size, stride)` view out of a byte buffer into a
/// contiguous, row-major copy of it.
///
/// This is what makes `TensorBase.set_` correct for a *view*, and views are not
/// exotic in a checkpoint: `torch.save` records each tensor's stride and offset
/// as it finds them, so a `state_dict` that holds `w.t()`, or a slice of a
/// larger buffer, arrives here non-contiguous. A reader that took the first
/// `numel` elements instead would return the transpose's storage read in the
/// wrong order -- right dtype, right shape, wrong numbers, no error. Measured:
/// a `4x3` tensor saved as `base.t()` of a `3x4`.
///
/// The walk is over element indices rather than bytes so it is dtype-agnostic;
/// `itemsize` only decides how wide each copied element is. The contiguous case
/// is not special-cased into a memcpy because `from_le_bytes` is already the
/// hot path's cost and the gather is a single pass either way -- one code path
/// means the contiguous case cannot pass while the strided case rots.
fn gather_strided(
    op: &str,
    bytes: &[u8],
    storage_offset: usize,
    size: &[usize],
    stride: &[i64],
    itemsize: usize,
    numel: usize,
) -> PyResult<Vec<u8>> {
    let mut out = Vec::with_capacity(numel * itemsize);
    let mut index = vec![0usize; size.len()];

    for _ in 0..numel {
        let mut element = storage_offset;
        for (d, i) in index.iter().enumerate() {
            element += i * stride[d] as usize;
        }
        let start = element * itemsize;
        let end = start + itemsize;
        if end > bytes.len() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: the view (offset {storage_offset}, size {size:?}, stride \
                 {stride:?}) reaches byte {end} of a storage holding {}",
                bytes.len()
            )));
        }
        out.extend_from_slice(&bytes[start..end]);

        // Odometer, last dimension fastest -- row-major, which is the order
        // `from_le_bytes` will read the result back in.
        for d in (0..size.len()).rev() {
            index[d] += 1;
            if index[d] < size[d] {
                break;
            }
            index[d] = 0;
        }
    }
    Ok(out)
}

/// The Python class an op result should wear.
///
/// Upstream, `_C` never hands back a bare `TensorBase`: `THPVariable_Wrap`
/// instantiates `THPVariableClass`, which C++ resolves to `torch._tensor.Tensor`
/// -- the Python subclass. The vendored tree depends on that in ways that have
/// no workaround, and `torch/nn/parameter.py:54` is the sharpest:
///
/// ```python
/// if type(data) is torch.Tensor or type(data) is Parameter:
///     return torch.Tensor._make_subclass(cls, data, requires_grad)
/// # otherwise: the custom-tensor path, which returns something that is not
/// # a Parameter, and `nn.Module.__setattr__` then classifies it as a plain
/// # attribute instead of registering it. A model built that way has no
/// # parameters at all.
/// ```
///
/// So the class is registered here, at the same moment upstream registers it
/// (`_initExtension`, after `from torch._tensor import Tensor` has run), and
/// `promote` puts every op result into it.
///
/// When nothing has registered a class -- which is exactly the standalone
/// `_C` that `tools/golden/loader.py` imports, with no `torch` package around
/// it -- `promote` is the identity and results stay `TensorBase`.
static TENSOR_CLASS: std::sync::OnceLock<Py<PyAny>> = std::sync::OnceLock::new();

#[pyfunction]
#[pyo3(name = "_set_tensor_class")]
pub fn set_tensor_class(cls: Py<PyAny>) {
    // First registration wins; `_initExtension` is idempotent upstream too.
    let _ = TENSOR_CLASS.set(cls);
}

/// `torch.Size`, registered the same way and for the same reason as
/// `TENSOR_CLASS` above: `_C` cannot build the class (it is a Python `tuple`
/// subclass declared in `bootstrap.py`) and must not import `torch` to find it,
/// because the golden loader runs `_C` with no `torch` package present.
static SIZE_CLASS: std::sync::OnceLock<Py<PyAny>> = std::sync::OnceLock::new();

#[pyfunction]
#[pyo3(name = "_set_size_class")]
pub fn set_size_class(cls: Py<PyAny>) {
    let _ = SIZE_CLASS.set(cls);
}

/// Wrap a bare `TensorBase` in the registered Python tensor class.
///
/// Idempotent and narrow on purpose: anything that is already an instance of a
/// *subclass* (a `Tensor`, a `Parameter`) is returned untouched, so an
/// in-place op still hands back the object it mutated.
pub fn promote(py: Python<'_>, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let cls = match TENSOR_CLASS.get() {
        Some(cls) => cls,
        None => return Ok(value),
    };
    let bound = value.bind(py);
    if bound.get_type().is(&py.get_type::<PyTensorBase>()) {
        return Ok(cls.bind(py).call1((bound,))?.unbind());
    }
    Ok(value)
}

/// The name `set_`'s dtype-mismatch message uses -- **a fourth spelling of the
/// same set**, distinct from `TorchDType::name()` (`int64`), `aten.rs`'s
/// `c10_name` (`int64_t`) and its `scalar_type_name` (`Long`).
///
/// It is the plain C++ type name rather than a c10 alias, so it is not
/// derivable from any of the other three, and every row was read off a real
/// `RuntimeError` from torch 2.13.0 by provoking the mismatch across all ten
/// storable dtypes:
///
/// ```text
/// int64 -> "long long"        NOT "int64_t"
/// int16 -> "short"            NOT "int16_t"
/// int8  -> "signed char"      uint8 -> "unsigned char"
/// ```
///
/// The four floating rows happen to agree with `c10_name`; the integral ones
/// do not, which is why this is its own table and not a call into that one.
fn set_type_name(dtype: TorchDType) -> &'static str {
    use TorchDType::*;
    match dtype {
        Float64 => "double",
        Float32 => "float",
        Float16 => "c10::Half",
        BFloat16 => "c10::BFloat16",
        Int64 => "long long",
        Int32 => "int",
        Int16 => "short",
        Int8 => "signed char",
        UInt8 => "unsigned char",
        Bool => "bool",
        other => other.name(),
    }
}

#[pymethods]
impl PyTensorBase {
    /// `TensorBase(other)` -- the constructor `promote` calls.
    ///
    /// PyO3's generated `tp_new` allocates with the *subtype* it was called
    /// with, so `Tensor(base)` produces a `Tensor` and `Parameter(base)` a
    /// `Parameter`, each sharing the candle tensor. That is what makes
    /// `_make_subclass` a three-line Python function in `bootstrap.py` rather
    /// than another piece of native machinery.
    ///
    /// **It now takes two forms, and the split is by argument type.**
    ///
    /// ```text
    /// TensorBase(existing)          re-wrap, sharing the candle tensor
    /// TensorBase()                  a (0,) tensor of the default float
    /// TensorBase(2), TensorBase(2, 3), ...   uninitialised storage of that shape
    /// ```
    ///
    /// The second form is upstream's legacy `torch.Tensor(2, 3)` constructor,
    /// which this refused by name until docs/kernels/KERNELS26.md §4. **The decision
    /// recorded there is that reproducing it is right**, on three grounds, and
    /// the grounds matter more than the conclusion:
    ///
    ///   1. **It is not a new computation path.** A `TorchDispatchMode` trace
    ///      of `torch.Tensor(3)` on 2.13.0 fires exactly one op --
    ///      `aten.empty.memory_format` -- which this shim already implements
    ///      and already golden-compares. So this is a *constructor spelling*
    ///      over an existing kernel, structurally the same as ARCH26.md §3.1's
    ///      `torch.conv2d` over `aten.convolution.default`, and not the kind of
    ///      invention `sqrt`-as-`pow(x, 0.5)` would have been.
    ///   2. **The forms are distinguishable at the type level**, which is
    ///      exactly what the old refusal already did -- it extracted a
    ///      `TensorBase` and refused everything else. Nothing here has to guess.
    ///   3. **It costs a family of architectures, not one.**
    ///      `nn.Parameter(torch.Tensor(config.hidden_size).uniform_())` is the
    ///      `masked_spec_embed` idiom, and it is created *unconditionally* in
    ///      `__init__` whether or not `apply_spec_augment` is set -- so no toy
    ///      config avoids it. `sew_d` is the architecture ARCH26.md §4 found it
    ///      in; `wav2vec2`, `sew`, `hubert`, `unispeech` and `wavlm` share the
    ///      line.
    ///
    /// **The bytes are zeros, and upstream's are uninitialised.** That is a
    /// real divergence and it is the same one `aten.empty.memory_format`
    /// already has (its golden case is `_dtype_shape_only_check`, whose whole
    /// docstring is "there is no correct value to diff"). Reading a
    /// `torch.Tensor(n)` before writing it is undefined upstream, so this is a
    /// narrowing of undefined behaviour rather than a disagreement about a
    /// defined one -- and the real caller writes it immediately, with
    /// `.uniform_()`.
    ///
    /// **The sequence form stays refused.** `torch.Tensor([3, 4])` builds from
    /// data (a `(2,)` tensor of `3.0, 4.0`, *not* a `(3, 4)` empty one), which
    /// is a third function again, is not what any measured caller reaches, and
    /// would need `_tensor_new_from_data` -- a module-level function this type
    /// has no handle to. It refuses by name, and the message says which form it
    /// is refusing rather than being generic.
    ///
    /// PyO3's generated `tp_new` allocates with the *subtype* it was called
    /// with, so `Tensor(base)` produces a `Tensor` and `Parameter(base)` a
    /// `Parameter`, each sharing the candle tensor. That is what makes
    /// `_make_subclass` a three-line Python function in `bootstrap.py` rather
    /// than another piece of native machinery, and it is unaffected by the
    /// second form.
    #[new]
    #[pyo3(signature = (*args))]
    fn py_new(args: &Bound<'_, PyTuple>) -> PyResult<Self> {
        // `TensorBase(existing)` first: it is the form every caller inside
        // this shim uses, and checking it first keeps that path a single
        // `extract`.
        if args.len() == 1 {
            if let Ok(existing) = args.get_item(0)?.extract::<Self>() {
                return Ok(existing);
            }
        }

        // Every remaining argument must be an integer, or this is the
        // sequence form (or something else again) and is refused by name.
        let mut dims: Vec<usize> = Vec::with_capacity(args.len());
        for i in 0..args.len() {
            let item = args.get_item(i)?;
            // `bool` subclasses `int` in Python; upstream's own parser takes
            // `torch.Tensor(True)` as a size of 1, and so does this, because
            // `extract::<i64>` accepts it. Not worth a special case in either
            // direction -- it is the same number.
            let Ok(extent) = item.extract::<i64>() else {
                return Err(not_implemented(format!(
                    "torch._C shim: TensorBase(...) takes either an existing tensor to \
                     re-wrap or a sequence of integer sizes (upstream's legacy \
                     `torch.Tensor(2, 3)` storage constructor); building from data, \
                     `torch.Tensor({})`, is a third form and is not implemented",
                    item.get_type().name().map(|n| n.to_string()).unwrap_or_default()
                )));
            };
            if extent < 0 {
                // Upstream's wording, measured: `torch.Tensor(-1)` raises
                // `Trying to create tensor with negative dimension -1: [-1]`.
                let shown: Vec<i64> = (0..args.len())
                    .map(|j| args.get_item(j).and_then(|v| v.extract::<i64>()).unwrap_or(0))
                    .collect();
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Trying to create tensor with negative dimension {extent}: {shown:?}"
                )));
            }
            dims.push(extent as usize);
        }

        // `TensorBase()` is `(0,)`, not `()` -- measured. The zero-argument
        // form is the one-dimensional empty tensor, not a scalar.
        if dims.is_empty() {
            dims.push(0);
        }

        // The default float, read at call time, so this moves with
        // `set_default_dtype` the way upstream's does (measured:
        // `set_default_dtype(torch.float64)` makes `torch.Tensor(3)` float64).
        let tag = crate::dtype::default_float();
        let storage = PyDtype::new(tag).storage("TensorBase")?;
        let inner = Tensor::zeros(dims, storage, &candle_core::Device::Cpu)
            .map_err(|e| candle_err("TensorBase", e))?;
        Self::new(inner)
    }

    /// torch returns `torch.Size`, itself a tuple subclass, and so does this --
    /// but only when a `torch` package is around to have defined it.
    ///
    /// `bootstrap.py` builds `Size` and hands it back through
    /// `_set_size_class`, the same registration shape `_set_tensor_class` uses
    /// and for the same reason: `tools/golden/loader.py` imports `_C`
    /// standalone with no `torch` package, nothing registers, and `shape` stays
    /// the plain tuple it has always been there. Every case in that harness
    /// compares shapes as sequences, so the two spellings are the same input to
    /// it -- which is why this is safe to make conditional rather than a hard
    /// dependency on a class `_C` cannot build for itself.
    ///
    /// `size()` with no `dim` goes through here, so it gets the same type,
    /// which is what upstream does. `stride()` deliberately does not: upstream
    /// returns a **plain tuple** from `stride()`, measured, and wrapping it
    /// would be a new divergence rather than the removal of one.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dims = PyTuple::new(py, self.dims())?;
        match SIZE_CLASS.get() {
            Some(cls) => cls.bind(py).call1((dims,)),
            None => Ok(dims.into_any()),
        }
    }

    /// Returns *the* module-level `torch.float32` rather than an equal copy.
    /// `PyDtype::new` here made `t.dtype is torch.float32` false for every
    /// tensor, which `==` hides -- and upstream's own `get_higher_dtype` opens
    /// with `if a is b: return a` as its guard against the `ordered_datatypes`
    /// table, so two float32 operands fell through it and promoted to float64
    /// (docs/graph/DECOMP.md §7.2, which had the symptom but not the cause).
    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::dtype::interned(py, self.tag)
    }

    /// `t.grad_dtype` -- **the tensor's own dtype, because that is where this
    /// shim accumulates gradients.**
    ///
    /// Reached by `torch.export` on every parameter of a module that has any
    /// (`nn.Linear` was the first module in this round's sweep to need it), so
    /// it stopped export for anything with weights.
    ///
    /// Upstream's default is the tensor's dtype -- measured across float32,
    /// float64, bfloat16, int64 and bool on 2.13.0, with and without
    /// `requires_grad` -- and it exists to let a low-precision parameter
    /// accumulate its gradient in a wider dtype. **This shim has no such
    /// machinery**: `tape.rs` accumulates into `grad` at the tensor's own
    /// dtype and there is no second dtype anywhere to report.
    ///
    /// So the getter is `self.dtype` and it is a fact, not a default.
    #[getter]
    fn grad_dtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::dtype::interned(py, self.tag)
    }

    /// **The setter accepts the dtype it already reports and refuses the rest.**
    ///
    /// Upstream's is a real setter: `t.grad_dtype = torch.float64` on a float32
    /// tensor takes, and the backward pass then accumulates in float64. Here
    /// there is nowhere to put that intent and nothing that would honour it, so
    /// accepting a different dtype would leave the backward pass accumulating
    /// in the old one while the attribute claimed otherwise -- a setter
    /// invisible to its own effect, which is the shape `_set_conj` refuses for
    /// the same reason (docs/graph/EXPORT5.md §5).
    ///
    /// Assigning the tensor's own dtype is a no-op and is allowed, so that
    /// save/restore round trips do not raise.
    #[setter]
    fn set_grad_dtype(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let mine = crate::dtype::interned(py, self.tag)?;
        if value.eq(mine.bind(py))? {
            return Ok(());
        }
        Err(not_implemented(format!(
            "TensorBase.grad_dtype = {value}: this shim accumulates a gradient in \
             the tensor's own dtype ({}) and has no separate gradient dtype to \
             set. Upstream honours this attribute in its backward pass; here \
             nothing would, so accepting it would leave the attribute claiming a \
             precision the backward pass does not use (tensor.rs, \
             docs/graph/EXPORT5.md §5)",
            self.tag.name()
        )))
    }

    #[getter]
    fn device(&self) -> PyDevice {
        self.device_label()
    }

    /// `tensor.is_meta`. Derived from the device rather than returned as a
    /// constant `False`, so it stays true to whatever `device()` reports if a
    /// meta-like device ever appears; today candle has three device kinds
    /// (`Cpu`, `Cuda`, `Metal`, see `PyDevice::from_candle`) and none of them
    /// is `meta`, so it always answers `False`.
    ///
    /// It is here because `Module.load_state_dict` reads it on every single
    /// parameter -- `torch/nn/modules/module.py:2449`, `if param.is_meta:`,
    /// before the shape check and before the copy. It was the only wall left on
    /// that path once the weights themselves could be read (docs/models/CKPT.md).
    /// A stub property raising by name stopped `load_state_dict` outright,
    /// which is the right behaviour for a hole and the wrong answer for a
    /// question the shim can answer.
    /// `t._has_symbolic_sizes_strides` -- **`False`, as a fact, not a stub.**
    ///
    /// `fake_tensor.py:1293`'s `extract_tensor_metadata` reads it on every
    /// tensor it hashes, so `torch.export` reaches it once per cached dispatch.
    ///
    /// It is `False` for the same reason `is_conj` is (docs/graph/EXPORT.md §2.3):
    /// there is nothing here that could make it `True`. A symbolic size is a
    /// `SymInt` living in a `TensorImpl`'s sizes-and-strides field; this shim's
    /// shapes are `Vec<usize>` on the dense side and `Repr::Meta`'s `shape` on
    /// the meta side, both concrete integers by construction, and there is no
    /// representation for anything else. Answering `True` would promise
    /// upstream a symbolic-shape path that does not exist; answering `False` is
    /// what is true of every tensor this shim can build.
    ///
    /// Note what this does **not** say: `torch.export`'s dynamic-shape support
    /// is a separate question and is untouched by this. This is about the
    /// *tensor*, and no tensor here carries a symbol.
    #[getter]
    fn _has_symbolic_sizes_strides(&self) -> bool {
        false
    }

    #[getter]
    fn is_meta(&self) -> bool {
        self.device_label().kind == "meta"
    }

    /// `tensor.is_cpu` / `tensor.is_cuda`. Same shape as `is_meta` and derived
    /// the same way, so all three agree with `device` by construction rather
    /// than by three separate constants that could drift apart.
    ///
    /// `is_cuda` in particular is read as a *guard* all over the vendored tree
    /// (`if t.is_cuda:` before a CUDA-only path), so answering it is what keeps
    /// those paths from being entered; a stub that raised turned a branch
    /// upstream never takes into a hard stop.
    #[getter]
    fn is_cpu(&self) -> bool {
        self.device_label().kind == "cpu"
    }

    #[getter]
    fn is_cuda(&self) -> bool {
        self.device_label().kind == "cuda"
    }

    /// `tensor.is_mps` / `is_xpu` / `is_maia`, the three remaining device
    /// predicates `torch/_tensor_str.py` reads before it prints anything.
    ///
    /// Same derivation as `is_cpu`/`is_cuda`/`is_meta` above, and here for the
    /// same reason those are: `_tensor_str.py:121-123` branches on all three
    /// (`tensor_totype` picks `float` over `double` on MPS and on Maia, and
    /// asks `xpu.get_device_properties(...).has_fp64` on XPU), and a raising
    /// stub made `print(tensor)` a hard stop on a line upstream skips.
    ///
    /// They answer `False` today because `PyDevice::from_candle` reports only
    /// `cpu`, `cuda` and `metal` -- but that is a fact about the device layer
    /// rather than a constant written here, which is the difference this
    /// spelling preserves. Note `mps` in particular: `torch.device("mps")` is
    /// a *constructible* label in this shim (there is a smoke test for it), so
    /// this predicate is one working backend away from answering `True` on its
    /// own.
    #[getter]
    fn is_mps(&self) -> bool {
        self.device_label().kind == "mps"
    }

    #[getter]
    fn is_xpu(&self) -> bool {
        self.device_label().kind == "xpu"
    }

    #[getter]
    fn is_maia(&self) -> bool {
        self.device_label().kind == "maia"
    }

    /// The five "which representation is this?" predicates, and `layout`.
    ///
    /// `torch/_tensor_str.py` reads every one of them before it prints a
    /// number: `is_nested` picks the `nested_tensor(` prefix, `is_sparse` and
    /// `layout` pick the COO and the compressed-sparse formatters,
    /// `is_quantized` adds the `quantization_scheme=` suffix, `_is_zerotensor`
    /// forces a `clone` and `is_neg` forces a `resolve_neg`.
    ///
    /// **They are an exhaustive `match` over `Repr`, not a `false`.** That is
    /// the whole of the argument for them, and it is deliberately structural.
    /// Writing it as a match means an arm cannot be added to `Repr` without
    /// the compiler asking what these six answer for it -- where a bare
    /// `false` would inherit silently, which is exactly the shape of the
    /// `is_mutable` accident in docs/distributed/DISTRIBUTED.md §8.1.
    ///
    /// **That is no longer a hypothetical: `Repr::Quantized` landed and the
    /// compiler asked.** Five of the six answered `false` again; `is_quantized`
    /// did not, and so this family is now a live predicate with a constructor
    /// behind it rather than a set of constants (docs/graph/QUANT2.md §4).
    ///
    /// The other half of the argument is in `pytests/test_shim.py`
    /// (`test_the_alternative_representations_have_no_constructors`): each of
    /// the representations *still* answering `False` has exactly one way into
    /// existence and every one of those ways refuses by name, so `False` is
    /// derivable from the constructor set rather than asserted. If any of them
    /// ever lands, that test fails and these stop being answerable this way.
    #[getter]
    fn is_nested(&self) -> bool {
        match self.inner {
            Repr::Dense(_) => false,
            Repr::Meta { .. } => false,
            Repr::Quantized(_) => false,
            // The compiler asked, as the docstring above promised it would.
            // A Vulkan tensor is dense and strided; what is different about it
            // is where the bytes are, not how they are addressed.
            Repr::Vulkan(_) => false,
            // It asked again for the complex arm. A pair of dense tensors is
            // not nested, not sparse, not quantised, not a zero tensor and
            // carries no negative-view bit; what is different about it is that
            // there are two buffers, not how either is addressed.
            Repr::Complex { .. } => false,
        }
    }

    #[getter]
    fn is_sparse(&self) -> bool {
        match self.inner {
            Repr::Dense(_) => false,
            Repr::Meta { .. } => false,
            Repr::Quantized(_) => false,
            // The compiler asked, as the docstring above promised it would.
            // A Vulkan tensor is dense and strided; what is different about it
            // is where the bytes are, not how they are addressed.
            Repr::Vulkan(_) => false,
            // It asked again for the complex arm. A pair of dense tensors is
            // not nested, not sparse, not quantised, not a zero tensor and
            // carries no negative-view bit; what is different about it is that
            // there are two buffers, not how either is addressed.
            Repr::Complex { .. } => false,
        }
    }

    /// **The one of the six that is no longer a constant.** `Repr::Quantized`
    /// landed, so this predicate now has something to say -- which is the
    /// point of writing it as a match: the arm asked the question rather than
    /// inheriting a `False`.
    ///
    /// It is upstream's *name* over a representation upstream does not have
    /// (GGML k-quant blocks, not per-tensor-affine `int8`), so agreeing with
    /// the name is a claim about the shape of the storage and not about the
    /// scheme. Anything that reads `True` here and then reaches for
    /// `qscheme()`/`q_scale()` gets a refusal that names the difference, which
    /// is why `qscheme` exists on this class at all.
    #[getter]
    fn is_quantized(&self) -> bool {
        match self.inner {
            Repr::Dense(_) => false,
            Repr::Meta { .. } => false,
            Repr::Quantized(_) => true,
            Repr::Vulkan(_) => false,
            Repr::Complex { .. } => false,
        }
    }

    /// `tensor.qscheme()` -- **a refusal on every arm**, and it is here rather
    /// than absent so that the refusal names the reason.
    ///
    /// `torch/_tensor_str.py` calls this the moment `is_quantized` is `True`,
    /// so without it `print(qweight)` would die with `AttributeError:
    /// 'TensorBase' object has no attribute 'qscheme'` -- a message that tells
    /// the reader nothing about *why*. Upstream raises on a dense tensor too
    /// (`RuntimeError: Could not run 'aten::qscheme' with arguments from the
    /// 'CPU' backend`), so refusing on both arms is not an invention.
    fn qscheme(&self) -> PyResult<()> {
        Err(match &self.inner {
            Repr::Quantized(q) => not_implemented(format!(
                "torch._C shim: a {} tensor has no torch qscheme. GGML block \
                 formats carry per-block scales (and, for the k-quants, per-\
                 sub-block scales and minima) rather than the single scale and \
                 zero point torch.per_tensor_affine names.",
                crate::quant::format_name(q.dtype())
            )),
            _ => pyo3::exceptions::PyRuntimeError::new_err(
                "Could not run 'aten::qscheme' with arguments from the 'CPU' backend.",
            ),
        })
    }

    /// A *method* upstream, not a property -- `_tensor_str.py:336` spells it
    /// `self._is_zerotensor()`. Getting that wrong is not a subtle failure
    /// (`'bool' object is not callable` is what a property gives), but it is
    /// the kind that is only visible by running the tree.
    fn _is_zerotensor(&self) -> bool {
        match self.inner {
            Repr::Dense(_) => false,
            Repr::Meta { .. } => false,
            Repr::Quantized(_) => false,
            // The compiler asked, as the docstring above promised it would.
            // A Vulkan tensor is dense and strided; what is different about it
            // is where the bytes are, not how they are addressed.
            Repr::Vulkan(_) => false,
            // It asked again for the complex arm. A pair of dense tensors is
            // not nested, not sparse, not quantised, not a zero tensor and
            // carries no negative-view bit; what is different about it is that
            // there are two buffers, not how either is addressed.
            Repr::Complex { .. } => false,
        }
    }

    /// Also a method (`_tensor_str.py:341`). The negative bit is set only by
    /// `torch._neg_view`; `neg()` materialises here exactly as it does
    /// upstream, so a negated tensor is a new buffer rather than a view
    /// wearing a flag.
    fn is_neg(&self) -> bool {
        match self.inner {
            Repr::Dense(_) => false,
            Repr::Meta { .. } => false,
            Repr::Quantized(_) => false,
            // The compiler asked, as the docstring above promised it would.
            // A Vulkan tensor is dense and strided; what is different about it
            // is where the bytes are, not how they are addressed.
            Repr::Vulkan(_) => false,
            // It asked again for the complex arm. A pair of dense tensors is
            // not nested, not sparse, not quantised, not a zero tensor and
            // carries no negative-view bit; what is different about it is that
            // there are two buffers, not how either is addressed.
            Repr::Complex { .. } => false,
        }
    }

    /// The name of `tensor.layout`, resolved to the `torch.layout` object in
    /// `bootstrap.py`.
    ///
    /// Split that way on purpose: the *fact* is about the representation and
    /// belongs beside the other five, while the `torch.strided` object is
    /// synthesised in Python (`_install_namespace_types`) and cannot be
    /// constructed here. `bootstrap.py` refuses by name for any string it does
    /// not recognise, so a new arm cannot leak through as `None`.
    fn _layout_name(&self) -> &'static str {
        match self.inner {
            // candle's tensors carry a stride and are dense; there is no
            // sparse or compressed storage anywhere in this build.
            Repr::Dense(_) => "strided",
            // Upstream's meta tensors are strided too -- `torch.zeros(2, 3,
            // device="meta").layout` is `torch.strided`, measured.
            Repr::Meta { .. } => "strided",
            // So are upstream's quantised tensors:
            // `torch.quantize_per_tensor(torch.zeros(4), 0.1, 0,
            // torch.qint8).layout` is `torch.strided`, measured on 2.13.0.
            // `torch.layout` names how the *elements* are addressed, and there
            // is no GGML entry in that enumeration to report even if one
            // wanted to -- the format is reported by `_quantized_format()`.
            Repr::Quantized(_) => "strided",
            // Upstream's `torch.zeros(2, device="vulkan").layout` is
            // `torch.strided` too -- a Vulkan tensor is a flat contiguous
            // buffer, which is what `strided` names.
            Repr::Vulkan(_) => "strided",
            // Upstream's complex tensors are `torch.strided` too --
            // `torch.view_as_complex(torch.ones(3,2)).layout` is
            // `torch.strided`, measured on 2.13.0. `torch.layout` names how
            // the elements are addressed, and each half here is addressed
            // exactly as a dense tensor is.
            Repr::Complex { .. } => "strided",
        }
    }

    // `tensor.get_device()` is *not* here, and the reason is a PyO3 collision
    // rather than a decision: `#[pymethods]` derives the slot name
    // `__pymethod_get_device__` from the `device` getter above and from a
    // method named `get_device` alike, and the crate is built without
    // `multiple-pymethods`, so the two cannot coexist in this block. It is
    // installed from `bootstrap.py` instead, where it is one line over the
    // `device` property this file already exposes.

    #[getter]
    fn ndim(&self) -> usize {
        self.dims().len()
    }

    fn dim(&self) -> usize {
        self.dims().len()
    }

    /// `rwkv`'s wall, and a **spelling, not a kernel** -- checked rather than
    /// assumed. On torch 2.13.0 `Tensor.ndimension` and `Tensor.dim` are two
    /// distinct method objects (`torch.Tensor.ndimension is torch.Tensor.dim`
    /// is `False`) that return the same `int` for every rank measured,
    /// including `0` for a 0-d tensor. There is no `torch.ndimension` free
    /// function and no `aten::ndimension` schema, so nothing is added to
    /// `overloads.json` or to `aten.rs`'s dispatch -- inventing either would
    /// invent a surface upstream does not have.
    ///
    /// It sits here rather than in `bootstrap.py` for the reason `dim` does:
    /// the answer is `dims().len()` and routing it through Python would add a
    /// frame to a call `transformers` makes inside its shape logic.
    fn ndimension(&self) -> usize {
        self.dims().len()
    }

    #[getter]
    fn _backward_hooks(&self) -> Option<&Py<PyAny>> {
        self.backward_hooks.as_ref()
    }

    #[setter]
    fn set__backward_hooks(&mut self, value: Option<Py<PyAny>>) {
        self.backward_hooks = value;
    }

    /// The gradient slot. See the field comment for why there is one now.
    ///
    /// Spelled `_shim_grad` rather than `grad` because `bootstrap.py` owns the
    /// `grad` property: `_install_autograd_shape` puts the type check and the
    /// docstring there, beside `requires_grad_` and `is_leaf`, so the whole
    /// autograd-shaped surface stays readable in one place.
    #[getter]
    fn _shim_grad(&self) -> Option<&Py<PyAny>> {
        self.grad.as_ref()
    }

    #[setter]
    fn set__shim_grad(&mut self, value: Option<Py<PyAny>>) {
        self.grad = value;
    }

    /// `tensor.element_size()` -- bytes per element, from the torch dtype tag
    /// and not from candle's, so `torch.bool` answers 1 rather than borrowing
    /// `uint8`'s answer by accident. (They agree; the point is that the tag is
    /// the authority, per BOOL.md §5-B.)
    ///
    /// **A quantised tensor refuses instead of answering.** Its tag is the
    /// dtype it dequantises to (`float32`), so answering from the tag would
    /// report 4 bytes per element for a Q4K weight that stores 0.5625 -- a
    /// number wrong by 7.1x, in the direction that makes a compression claim
    /// look worse than it is and a memory budget look better. `numel() *
    /// element_size()` is how upstream code sizes a buffer, so this is exactly
    /// the read that must not silently succeed. `_quantized_nbytes()` is the
    /// answerable question.
    fn element_size(&self) -> PyResult<usize> {
        match &self.inner {
            // A Vulkan tensor answers from the tag like a dense one, and
            // unlike a quantised one it is entitled to: the buffer really is
            // `numel * 4` bytes of f32, so `numel() * element_size()` sizes it
            // correctly.
            // A complex tensor answers from the tag as well, and the tag is
            // the *pair* width: `complex64` is 8, not 4. That is upstream's
            // answer and it is the one that makes `numel() * element_size()`
            // size the two buffers together. If this ever answered 4, someone
            // had aliased complex64 onto float32 -- the exact drop this arm
            // exists to prevent.
            Repr::Dense(_) | Repr::Meta { .. } | Repr::Vulkan(_) | Repr::Complex { .. } => {
                Ok(self.tag.itemsize())
            }
            Repr::Quantized(q) => Err(not_implemented(format!(
                "TensorBase.element_size: a {} tensor has no whole number of \
                 bytes per element ({} bytes per {} elements). Use \
                 torch._C._quantized_nbytes(t) for the storage size.",
                crate::quant::format_name(q.dtype()),
                q.dtype().type_size(),
                q.dtype().block_size(),
            ))),
        }
    }

    /// `tensor.untyped_storage()` -- **a snapshot, where upstream lends.**
    ///
    /// `torch/_tensor.py:311 _typed_storage` calls this on every tensor of
    /// every `torch.save`, and `torch/serialization.py`'s `persistent_id`
    /// turns what comes back into one `data/N` record. It is the mirror of
    /// `set_`, which is the *load* side's one non-aliasing point (see its
    /// docstring): upstream hands out the storage a tensor is a view of, and
    /// this hands out a copy of it, because candle owns its buffer and has no
    /// way to lend it as a `StorageBase`.
    ///
    /// Copying is safe *here* in a way it is not on the load side, and the
    /// asymmetry is worth naming since docs/models/CKPT.md §4 spent a section on the
    /// other direction. Saving reads and never writes, so a snapshot taken
    /// while the tensor is alive has exactly the bytes the tensor has. What a
    /// copy cannot carry by itself is *identity* -- and identity is load
    /// bearing here, because `torch.save` decides how many records to write by
    /// comparing storages. That is what `storage.rs::origin` is for: the
    /// snapshot is tagged with the address of the candle buffer it came from,
    /// so `x` and `x.t()` still answer with one storage and the file still says
    /// they were views of one buffer.
    ///
    /// **A meta or quantised tensor refuses**, from `tensor()` and from
    /// `storage_snapshot` respectively: a meta tensor has no bytes at all
    /// (docs/devices/META.md §3, and `torch/_tensor.py:337` takes a different branch
    /// for it before reaching here), and a quantised one has blocks that are
    /// not a flat storage in any dtype torch could name in a record.
    fn untyped_storage(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // **A meta tensor answers with a handle, and that is docs/graph/EXPORT.md
        // §3.3 closed.**
        //
        // The refusal it used to give -- `Cannot copy out of meta tensor; no
        // data!`, out of `tensor()` -- was right about the bytes and wrong
        // about the question, the same mismatch `stride()` had one line
        // earlier (docs/graph/EXPORT4.md §6.5). `meta_utils.py:2071` is not asking
        // for bytes; it is asking for something to key an aliasing memo on,
        // and it never reads through what it gets. So the answerable part of
        // the question is a size and an identity, and both are available: the
        // size from shape and dtype, the identity from `Repr::Meta`'s
        // `storage_id`.
        //
        // What it is NOT is a zero-length CPU storage wearing a meta label.
        // `storage::meta` leaves `filled` false, so `set_` still refuses it,
        // and every byte door on it refuses by name rather than answering over
        // an empty buffer. docs/graph/EXPORT5.md §2 is the table of which of
        // upstream's expectations that meets and which it refuses.
        if let Some(storage_id) = self.meta_storage_id() {
            let nbytes = self.numel() * self.tag.itemsize();
            return crate::storage::meta(py, nbytes, storage_id);
        }
        let (bytes, origin) = self.storage_snapshot("TensorBase.untyped_storage")?;
        crate::storage::snapshot(py, bytes, origin)
    }

    /// `tensor.storage_offset()` -- the first element of this view inside its
    /// storage, in elements.
    ///
    /// Real, not zero. `torch.save` writes it into the record and `torch.load`
    /// feeds it back to `set_`, which walks it (`gather_strided`), so a
    /// constant zero here would save `x[1]` as if it started at the front of
    /// its buffer -- shape and dtype right, values from the wrong row, no
    /// exception anywhere. docs/models/CKPT.md §5 measured that exact failure coming
    /// the other way.
    ///
    /// A meta tensor refuses rather than answering `0`: `Repr::Meta` carries no
    /// layout at all (docs/devices/META.md §6 records the narrowing), so `0` would be
    /// a guess that happens to be right for contiguous meta tensors and wrong
    /// for the transposed ones upstream's meta does model.
    fn storage_offset(&self) -> PyResult<usize> {
        // A meta tensor has no storage, so it cannot be offset into one, and 0
        // is the only representable answer rather than a chosen one. Upstream
        // answers 0 here too for a freshly built meta tensor. See `stride`
        // below for why meta is answered rather than refused at all.
        if let Repr::Meta { .. } = self.inner {
            return Ok(0);
        }
        Ok(self.tensor()?.layout().start_offset())
    }

    /// `tensor.stride()` / `tensor.stride(dim)`, in elements.
    ///
    /// candle's `Layout` has carried a real stride since docs/kernels/VIEWS.md §6 made
    /// narrowing alias, so this reports what the tensor actually is rather than
    /// what a contiguous tensor of its shape would be. It is the fourth of the
    /// four numbers a `torch.save` record is made of, and the same reasoning as
    /// `storage_offset` applies: `w.t()` saved with a contiguous stride is a
    /// file that reads back transposed, silently.
    ///
    /// Upstream returns a `torch.Size`; this returns a plain tuple, the same
    /// narrowing `shape` already documents. `stride(dim)` returns one int, and
    /// negative `dim` counts from the back, as upstream's does.
    ///
    /// **Meta refuses**, for the reason in `Repr::Meta`'s docstring: upstream's
    /// meta tensor does carry stride and this build's does not model it, so
    /// answering would invent one.
    #[pyo3(signature = (dim = None))]
    fn stride<'py>(&self, py: Python<'py>, dim: Option<isize>) -> PyResult<Bound<'py, PyAny>> {
        // **A meta tensor answers its contiguous stride, and that is derived,
        // not guessed.**
        //
        // `Repr::Meta` stores a shape and no stride (docs/devices/META.md §6 records
        // the narrowing), so the obvious reading is that `stride()` cannot be
        // answered and must refuse -- which is what it did, via `tensor()?`,
        // and the refusal was `Cannot copy out of meta tensor; no data!`. That
        // message is about *bytes*, and a stride is not bytes; it was the wrong
        // refusal for the question, and it stopped `torch.export` at
        // `meta_utils.py:2066`.
        //
        // The answer is available because of a fact the type already asserts:
        // `is_contiguous()` returns `true` for every `Repr::Meta`, since no
        // kernel in this tree can produce a non-contiguous one, and
        // `aten.empty_strided.default` refuses by name rather than build one.
        // A contiguous tensor's stride is a function of its shape alone. So
        // this is the same value the dense path would compute, arrived at
        // without a storage to read it off.
        //
        // The invariant this rests on is checked, not assumed:
        // `test_export4.py::test_a_meta_tensor_is_contiguous_so_its_stride_is_derivable`
        // fails if a meta tensor ever becomes non-contiguous, which is exactly
        // the day this answer would start lying.
        if let Repr::Meta { shape, .. } = &self.inner {
            let mut contiguous = vec![0i64; shape.len()];
            let mut acc: i64 = 1;
            for i in (0..shape.len()).rev() {
                contiguous[i] = acc;
                acc *= shape[i] as i64;
            }
            let Some(dim) = dim else {
                return Ok(PyTuple::new(py, &contiguous)?.into_any());
            };
            let rank = contiguous.len() as isize;
            let at = if dim < 0 { dim + rank } else { dim };
            if at < 0 || at >= rank {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "Dimension out of range (expected to be in range of [{}, {}], but got {dim})",
                    -rank,
                    rank - 1
                )));
            }
            return contiguous[at as usize].into_bound_py_any(py);
        }
        let stride = self.tensor()?.layout().stride().to_vec();
        let Some(dim) = dim else {
            return Ok(PyTuple::new(py, &stride)?.into_any());
        };
        let rank = stride.len() as isize;
        let at = if dim < 0 { dim + rank } else { dim };
        if at < 0 || at >= rank {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Dimension out of range (expected to be in range of [{}, {}], but got {dim})",
                -rank,
                rank - 1
            )));
        }
        stride[at as usize].into_bound_py_any(py)
    }

    /// `tensor.data_ptr()` -- the address of this view's first element.
    ///
    /// Storage address plus `storage_offset` in bytes, which is the relation
    /// upstream has. The address is candle's, not a copy's, so two tensors that
    /// share a buffer report addresses that differ by exactly their offsets --
    /// the property `test_which_ops_share_storage_with_their_input_and_which_do_not`
    /// exists to pin and which, before this, had to be measured indirectly.
    ///
    /// Never dereferenced from Python and never handed to a reader. Upstream's
    /// own save path uses it only to tell storages apart
    /// (`torch/serialization.py:1224`), and `torch/_tensor.py:462` compares it
    /// against `0` to detect a storage-less subclass.
    fn data_ptr(&self) -> PyResult<usize> {
        // The bit `torch._C._set_throw_on_mutable_data_ptr` sets, checked
        // before anything reads storage. Upstream's message, from
        // `torch/csrc/autograd/python_variable.cpp`.
        if self
            .throw_on_mutable_data_ptr
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot access data pointer of Tensor that doesn't have storage",
            ));
        }
        // The softer bit: upstream warns and still answers. Emitted before the
        // storage is read so the warning appears even if the read then fails.
        if self
            .warn_deprecated_on_mutable_data_ptr
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            Python::attach(|py| {
                pyo3::PyErr::warn(
                    py,
                    &py.get_type::<pyo3::exceptions::PyUserWarning>(),
                    std::ffi::CString::new(
                        "Accessing the data pointer of FakeTensor is deprecated and will \
                         error in PyTorch 2.5. This is almost definitely a bug in your code \
                         and will cause undefined behavior with subsystems like \
                         torch.compile. Please wrap calls to tensor.data_ptr() in an opaque \
                         custom op; If all else fails, you can guard accesses to \
                         tensor.data_ptr() on isinstance(tensor, FakeTensor).",
                    )
                    .expect("literal has no interior nul")
                    .as_c_str(),
                    1,
                )
            })?;
        }
        let tensor = self.tensor()?;
        let (guard, layout) = tensor.storage_and_layout();
        let storage: &candle_core::Storage = &guard;
        Ok(storage as *const candle_core::Storage as usize
            + layout.start_offset() * self.tag.itemsize())
    }

    /// `tensor.set_(storage, storage_offset, size, stride)` -- **a copy, where
    /// upstream aliases.**
    ///
    /// The only caller that matters is `torch/_utils.py:198 _rebuild_tensor`,
    /// which every `torch.load` of every tensor goes through:
    ///
    /// ```python
    /// t = torch.empty((0,), dtype=storage.dtype, device=...)
    /// return t.set_(storage._untyped_storage, storage_offset, size, stride)
    /// ```
    ///
    /// Upstream this makes `t` a *view* of the storage. Here candle owns its
    /// memory, so the bytes are copied out and the tensor is independent of the
    /// storage afterwards. Two consequences, both refused rather than papered
    /// over, because either one produces silently wrong weights:
    ///
    /// 1. **The storage must already hold its payload.** See storage.rs -- the
    ///    legacy `torch.save` format fills storages *after* `_rebuild_tensor`
    ///    runs, and a copying `set_` there yields a checkpoint of zeros with no
    ///    error anywhere. Measured. So an unfilled storage is refused by name.
    ///
    /// 2. **The result is contiguous even when the view was not.** A saved
    ///    tensor may be a strided view of its storage -- `w.t()`, or a slice of
    ///    a larger buffer -- and `gather_strided` reads it in row-major order
    ///    into a fresh buffer. Upstream would have kept the view; here the copy
    ///    holds the same numbers with contiguous stride. That is visible to
    ///    anything that reads `.stride()`, and it is why `set_` cannot be used
    ///    to build an aliasing view on purpose.
    /// ---
    ///
    /// **`aten.set_.source_Tensor` and the no-argument form now work too**
    /// (docs/kernels/KERNELS26.md §5), and unlike the storage form above they *do*
    /// alias, which is upstream's behaviour rather than a divergence.
    ///
    /// ```text
    /// a.set_(b)   a adopts b's storage, shape and stride, and returns `a`.
    ///             The two share afterwards: writing into `b` is visible in `a`.
    /// a.set_()    a becomes an empty (0,) tensor of its own dtype.
    /// ```
    ///
    /// This is the wall `vits` and `sew_d` both stopped on (ARCH26.md §2),
    /// reached through `torch.nn.utils.parametrizations.weight_norm`:
    /// `register_parametrization` calls `ParametrizationList.__init__`, which
    /// calls `_maybe_set(original, new)`, which is `dest.set_(src)` for two
    /// tensors.
    ///
    /// **The tensor form aliases and the storage form copies, in the same
    /// method, and that asymmetry is not an inconsistency.** The storage form
    /// has to copy because candle owns its memory and a `Storage` is bytes
    /// this shim holds separately; the tensor form does not, because
    /// `Repr::Dense` *is* a candle tensor and a candle clone is an `Arc`
    /// clone of the same storage. So the tensor form gets upstream's semantics
    /// for free, and `test_which_ops_share_storage_with_their_input_and_which_do_not`
    /// is where that is pinned rather than assumed.
    ///
    /// **The dtype must match, and upstream's refusal is reproduced**:
    /// `torch.zeros(2).set_(torch.arange(3))` raises
    /// `Could not set tensor of type long long to a tensor of type float`.
    /// Silently adopting the source's dtype would make `set_` a `to()` with no
    /// conversion, and the parametrize machinery would then be swapping a
    /// float parameter for an integer one without complaint.
    #[pyo3(signature = (source = None, storage_offset = 0, size = None, stride = None))]
    fn set_<'py>(
        slf: &Bound<'py, Self>,
        source: Option<&Bound<'py, PyAny>>,
        storage_offset: usize,
        size: Option<Vec<usize>>,
        stride: Option<Vec<i64>>,
    ) -> PyResult<Bound<'py, Self>> {
        const OP: &str = "TensorBase.set_";

        // `a.set_()` -- `aten.set_.default`. Upstream empties the tensor in
        // place, keeping its dtype: `torch.arange(4.).set_()` is `(0,)` with
        // `numel() == 0`, measured.
        let Some(source) = source else {
            let tag = slf.borrow().tag;
            let storage = PyDtype::new(tag).storage(OP)?;
            let empty = Tensor::zeros(vec![0usize], storage, &candle_core::Device::Cpu)
                .map_err(|e| candle_err(OP, e))?;
            let replacement = Self { tag, ..Self::new(empty)? };
            slf.borrow_mut().replace_with(replacement);
            return Ok(slf.clone());
        };

        // `a.set_(b)` -- `aten.set_.source_Tensor`. Checked before the
        // storage extraction, not after, because a `Parameter` extracts as a
        // `TensorBase` and would otherwise fall into the storage arm's error
        // message -- which is exactly the message ARCH26.md §2 recorded.
        if let Ok(other) = source.extract::<PyTensorBase>() {
            if storage_offset != 0 || size.is_some() || stride.is_some() {
                return Err(not_implemented(format!(
                    "{OP}(tensor, storage_offset, size, stride) is \
                     aten.set_.source_Tensor_storage_offset, a distinct overload that \
                     re-lays-out the source rather than adopting it, and is not \
                     implemented in this shim"
                )));
            }
            let tag = slf.borrow().tag;
            if other.tag != tag {
                // Upstream's wording, with its C++ type names, measured on
                // 2.13.0. No shim prefix: this is torch semantics being
                // reproduced, the same convention `overflow()` follows.
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Could not set tensor of type {} to a tensor of type {}",
                    set_type_name(other.tag),
                    set_type_name(tag),
                )));
            }
            // `replace_with` rather than `write_into`: `set_` re-points the
            // tensor at another one's storage, it does not copy values into
            // the storage this tensor already had. `Repr::Dense` holds a
            // candle tensor whose clone shares its `Arc`, so the two alias
            // afterwards exactly as upstream's do.
            slf.borrow_mut().replace_with(other);
            return Ok(slf.clone());
        }

        let storage: PyRef<'_, crate::storage::PyStorageBase> =
            source.extract().map_err(|_| {
                let got = source
                    .get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_else(|_| "?".to_string());
                not_implemented(format!(
                    "{OP}: expected a torch.UntypedStorage or a tensor, got {got}"
                ))
            })?;

        let size = size.ok_or_else(|| {
            not_implemented(format!(
                "{OP}(storage) without an explicit size is not implemented -- it \
                 would mean adopting the storage's whole extent, which is the \
                 aliasing behaviour this shim does not have"
            ))
        })?;

        let tag = slf.borrow().tag;
        let itemsize = tag.itemsize();
        let numel: usize = size.iter().product();

        if numel > 0 && !storage.is_filled() {
            return Err(not_implemented(format!(
                "{OP}: the storage has never been filled. This shim's set_ copies \
                 out of the storage instead of aliasing it, so a tensor built \
                 from an empty storage would be silently zero. The caller must \
                 deliver the bytes before set_, not after (see storage.rs and \
                 docs/models/CKPT.md §4)."
            )));
        }

        let stride = stride.unwrap_or_else(|| contiguous_stride(&size));
        if stride.len() != size.len() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{OP}: size {size:?} has {} dimensions but stride {stride:?} has {}",
                size.len(),
                stride.len()
            )));
        }
        if stride.iter().any(|s| *s < 0) {
            return Err(not_implemented(format!(
                "{OP}: negative stride in {stride:?}. torch itself does not produce \
                 these, so this is refused rather than guessed at."
            )));
        }

        let bytes = storage.bytes();
        let gathered = gather_strided(
            OP,
            bytes,
            storage_offset,
            &size,
            &stride,
            itemsize,
            numel,
        )?;

        let replacement = from_le_bytes(OP, &gathered, &size, tag)?;
        drop(storage);
        slf.borrow_mut().replace_with(replacement);
        Ok(slf.clone())
    }

    /// `tensor.data = other`, the write half of the `.data` property that
    /// `bootstrap.py` installs.
    ///
    /// **This is the last wall on `nn.Module.to(device)`.** `Module._apply`
    /// converts a parameter, asks `_has_compatible_shallow_copy_type` whether
    /// the result can take the old one's place, and on `True` -- which is what
    /// upstream answers for two dense tensors -- assigns
    /// `param.data = param_applied` (`torch/nn/modules/module.py:995`). With no
    /// setter that assignment is an `AttributeError`, so every `.to()`,
    /// `.cpu()`, `.float()` and `.half()` on a module died there.
    ///
    /// It is `replace_with`, which means it inherits that method's recorded
    /// divergence: the wrapper starts pointing at a different candle tensor
    /// rather than the storage being rewritten, so a view taken *before* the
    /// assignment does not follow it. Upstream's `.data =` swaps the TensorImpl
    /// too, so pre-existing views do not follow there either -- for this
    /// spelling the two agree, and docs/kernels/OPS4.md §8's open aliasing question is
    /// about writes through views, which this is not. docs/devices/DEVICE_ABS.md §4.
    ///
    /// `requires_grad` is deliberately left alone: upstream's `.data =` does
    /// not touch it, and `_apply` relies on that to keep a `Parameter` a
    /// parameter.
    fn _shim_set_data(slf: &Bound<'_, Self>, value: PyTensorBase) {
        slf.borrow_mut().replace_with(value);
    }

    fn numel(&self) -> usize {
        PyTensorBase::elem_count(self)
    }

    #[pyo3(signature = (dim = None))]
    fn size<'py>(&self, py: Python<'py>, dim: Option<isize>) -> PyResult<Bound<'py, PyAny>> {
        match dim {
            None => self.shape(py),
            Some(dim) => {
                let rank = self.dims().len() as isize;
                let index = if dim < 0 { dim + rank } else { dim };
                if index < 0 || index >= rank {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "Dimension out of range (expected to be in range of [{}, {}], but got {dim})",
                        -rank,
                        rank - 1
                    )));
                }
                self.dims()[index as usize].into_bound_py_any(py)
            }
        }
    }

    /// `tensor.is_contiguous()`.
    ///
    /// A meta tensor answers `True` unconditionally, and that is a narrowing
    /// rather than an answer: upstream tracks stride on meta and
    /// `torch.zeros(2,3,device="meta").t().is_contiguous()` is `False`. Here
    /// `Repr::Meta` carries no stride (see its comment) and no meta kernel can
    /// produce a transposed one, so every meta tensor this shim can make *is*
    /// contiguous. It stops being true the day a meta `t`/`permute` kernel
    /// lands, and that kernel is the thing that has to add the stride field.
    /// **`memory_format` is keyword-only and is answered, not ignored.**
    ///
    /// `fake_tensor.py:1295`'s `extract_tensor_metadata` calls
    /// `t.is_contiguous(memory_format=...)` on every tensor it hashes, so
    /// `torch.export` reaches it once per cached dispatch. Measured on 2.13.0,
    /// and the signature is measured too -- upstream refuses a *positional*
    /// memory format (`is_contiguous() takes 0 positional arguments`), so this
    /// is `*, memory_format` rather than an optional first argument:
    ///
    /// | asked | upstream | here |
    /// |---|---|---|
    /// | `contiguous_format` | the ordinary answer | the ordinary answer |
    /// | `preserve_format` | the ordinary answer (measured: `False` on a permuted tensor, not an unconditional `True`) | the ordinary answer |
    /// | `channels_last` / `channels_last_3d` | `True` only for a tensor actually in that layout | **`False`, as a fact** |
    ///
    /// The last row is a fact rather than a stand-in, in the same sense as
    /// `is_mkldnn` (docs/graph/EXPORT.md §2.3): **there is no channels-last
    /// representation in this build at all.** candle carries a `Layout` and no
    /// memory-format tag, no kernel here accepts `memory_format=channels_last`,
    /// and `grep channels_last rust/torch_c/src/*.rs` finds only upsample
    /// *error message* strings. A tensor cannot be in a layout the build cannot
    /// construct, so `False` is true of every tensor this shim can make -- and
    /// `test_export5.py` asserts that non-constructibility rather than trusting
    /// this paragraph, because the day a channels-last kernel lands is the day
    /// this answer starts lying.
    ///
    /// An unrecognised memory format **refuses by name** rather than falling
    /// through to the ordinary answer: silently treating an unknown label as
    /// `contiguous_format` is how a caller gets a `True` it did not ask for.
    #[pyo3(signature = (*, memory_format = None))]
    fn is_contiguous(&self, memory_format: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
        if let Some(mf) = memory_format {
            let label = mf
                .getattr("_shim_name")
                .and_then(|n| n.extract::<String>())
                .unwrap_or_else(|_| mf.str().map(|s| s.to_string()).unwrap_or_default());
            let label = label.rsplit('.').next().unwrap_or(&label).to_string();
            match label.as_str() {
                "contiguous_format" | "preserve_format" => {}
                "channels_last" | "channels_last_3d" => return Ok(false),
                other => {
                    return Err(not_implemented(format!(
                        "TensorBase.is_contiguous(memory_format={other}): this shim knows \
                         torch.contiguous_format, torch.preserve_format, \
                         torch.channels_last and torch.channels_last_3d, and this is none \
                         of them. Refused rather than answered as if it were \
                         contiguous_format (tensor.rs)"
                    )))
                }
            }
        }
        Ok(self.is_contiguous_inner())
    }

    fn is_contiguous_inner(&self) -> bool {
        match &self.inner {
            Repr::Dense(tensor) => tensor.is_contiguous(),
            Repr::Meta { .. } => true,
            // A `QTensor` has no `Layout` and therefore no stride at all: its
            // blocks are laid out in one flat, row-major run, and candle
            // offers no way to build a strided view of one. So every quantised
            // tensor this shim can make is contiguous for the same reason
            // `Meta` is -- there is no operation that could produce a
            // non-contiguous one.
            Repr::Quantized(_) => true,
            // Every Vulkan tensor this build can make is a flat row-major
            // buffer: there is no view, transpose or slice kernel on this
            // device, so there is nothing that could make a non-contiguous
            // one. Same argument as the two arms above, and it stops being
            // true the day a stride-taking kernel lands.
            Repr::Vulkan(_) => true,
            // Both halves are made contiguous at construction (every producer
            // in `complex_ops` calls `.contiguous()`), so this is true by the
            // same argument as the three arms above rather than by inspection
            // -- and it is checked, not assumed: this reads them.
            Repr::Complex { re, im } => re.is_contiguous() && im.is_contiguous(),
        }
    }

    /// The stored flag, **or** the fact that an op produced this tensor from
    /// something that had it.
    ///
    /// Upstream keeps one flag and derives nothing; here the leaf half is
    /// stored (`requires_grad_`, the factory keyword, `nn.Parameter`) and the
    /// non-leaf half is `from_op`, which the door sets. The disjunction is
    /// upstream's invariant `grad_fn is not None => requires_grad` written as
    /// code: a tensor cannot report a `grad_fn` and deny requiring a gradient.
    /// docs/training/BACKWARD4.md §2.
    #[getter]
    fn requires_grad(&self) -> bool {
        self.requires_grad || self.from_op.is_some()
    }

    /// `t.requires_grad = True`, and the one rule the flag has.
    ///
    /// The flag is inert (no graph is built from it), but "inert" says nothing
    /// about which tensors may carry it, and upstream restricts that: only
    /// floating-point and complex tensors may require gradients, because only
    /// those have a derivative to accumulate. `docs/training/BACKWARD2.md` §1.4 measured
    /// this shim accepting `torch.ones(2, dtype=torch.int64).requires_grad_(True)`
    /// where upstream raises -- the single place in the whole autograd chain
    /// where this shim was the permissive one.
    ///
    /// The message is upstream's, transcribed from torch 2.13.0 by running the
    /// failing case, and upstream has **three** wordings for the same rule
    /// depending on which door is used: this one is the attribute setter's.
    /// `bootstrap.py`'s `requires_grad_` and the factory keyword carry theirs.
    /// Reproducing all three is the same practice `_frombuffer` follows for its
    /// `ValueError`s -- a caller who greps for upstream's text finds it.
    ///
    /// Only `True` is checked. Upstream lets `requires_grad = False` through on
    /// any dtype, and `nn.Module._apply` writes exactly that over integer
    /// buffers.
    #[setter]
    fn set_requires_grad(&mut self, value: bool) -> PyResult<()> {
        // Upstream's own refusal, transcribed from 2.13.0 by running the
        // failing case, and it is the divergence docs/training/BACKWARD3.md §1.1 listed
        // and could not close: the flag on a non-leaf is not a flag, it is a
        // consequence of the graph, so changing it is meaningless rather than
        // merely unsupported. Unreachable until `from_op` existed.
        if self.from_op.is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "you can only change requires_grad flags of leaf variables.",
            ));
        }
        if value && !(self.tag.is_floating_point() || self.tag.is_complex()) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "only Tensors of floating point and complex dtype can require gradients",
            ));
        }
        self.requires_grad = value;
        Ok(())
    }

    /// The aten op that produced this tensor, or `None` for a leaf.
    ///
    /// `bootstrap.py` turns it into `grad_fn` and `is_leaf`; it is exposed as a
    /// string rather than as those two properties because the naming table that
    /// makes `type(grad_fn).__name__` agree with upstream is measured data
    /// (docs/training/BACKWARD4.md §3.1) and belongs beside the measurement, in Python,
    /// where the test that checks it against real torch can read it.
    #[getter]
    pub(crate) fn _shim_from_op(&self) -> Option<&str> {
        self.from_op.as_deref()
    }

    /// Called by the door (`aten::mark_autograd`) and by nothing else.
    fn _shim_set_from_op(&mut self, op: &str) {
        self.from_op = Some(op.into());
    }

    /// `t.retains_grad`. `False` for every leaf, and for a non-leaf until
    /// `retain_grad()` is called on it -- which is upstream's rule and, until
    /// `from_op` existed, was a `NotImplementedError` nothing had ever reached.
    #[getter]
    fn _shim_retains_grad(&self) -> bool {
        self.retains_grad
    }

    fn _shim_set_retains_grad(&mut self, value: bool) {
        self.retains_grad = value;
    }

    /// Nested Python lists, as `torch.Tensor.tolist` gives. This is the only
    /// way to read values out at the moment, so tests can compare numbers
    /// against real torch without any further surface being built first.
    ///
    /// On `meta` it raises upstream's own `NotImplementedError: Cannot copy
    /// out of meta tensor; no data!` -- the same refusal, from the same place,
    /// as `.cpu()` and `.to("cpu")`, because it is the same question.
    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let flat = flat_objects(py, self.tensor()?, self.tag)?;
        nest(py, &flat, self.dims())
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorBase(shape={:?}, dtype={}, device={})",
            self.dims(),
            self.tag.name(),
            self.device_label().__str__()
        )
    }
}

/// Flattens to Python scalars. Every float type is read through `f64` and every
/// integer type through `i64`, so a dtype candle can hold but this shim has not
/// taught itself to read fails by name instead of returning garbage.
fn flat_objects(py: Python<'_>, tensor: &Tensor, tag: TorchDType) -> PyResult<Vec<Py<PyAny>>> {
    let flat = tensor.flatten_all().map_err(|e| candle_err("tolist", e))?;
    let dtype = tensor.dtype();
    if tag == TorchDType::Bool {
        // torch's `tolist` on a bool tensor yields Python `bool`s, not 0/1
        // ints. Reading `!= 0` rather than the raw byte is also what torch
        // guarantees (BOOL.md §2.6) -- the tag promises the read, not the byte.
        let values = flat
            .to_vec1::<u8>()
            .map_err(|e| candle_err("tolist", e))?;
        return values
            .into_iter()
            .map(|v| (v != 0).into_py_any(py))
            .collect::<PyResult<Vec<_>>>();
    }
    if dtype.is_float() {
        // `float8_e4m3fn` refused here until docs/numerics/FLOAT8C.md §1: candle's
        // `F8E4M3 -> F64` conversion recurses into itself, so this read hung.
        // `widen_f64` routes it `F8E4M3 -> F32 -> F64`, which is exact for
        // every one of the dtype's 256 bit patterns, so the refusal is gone
        // and `tolist` answers upstream's numbers.
        let values = crate::aten::widen_f64(&flat)
            .and_then(|t| t.to_vec1::<f64>())
            .map_err(|e| candle_err("tolist", e))?;
        values
            .into_iter()
            .map(|v| v.into_py_any(py))
            .collect::<PyResult<Vec<_>>>()
    } else if dtype.is_int() {
        let values = flat
            .to_dtype(DType::I64)
            .and_then(|t| t.to_vec1::<i64>())
            .map_err(|e| candle_err("tolist", e))?;
        values
            .into_iter()
            .map(|v| v.into_py_any(py))
            .collect::<PyResult<Vec<_>>>()
    } else {
        Err(not_implemented(format!(
            "tolist not implemented in torch._C shim for dtype: {}",
            dtype.as_str()
        )))
    }
}

fn nest(py: Python<'_>, flat: &[Py<PyAny>], dims: &[usize]) -> PyResult<Py<PyAny>> {
    match dims.split_first() {
        // 0-d tensor: torch's tolist returns the bare scalar.
        None => Ok(flat[0].clone_ref(py)),
        Some((&outer, rest)) => {
            let stride: usize = rest.iter().product();
            let mut items = Vec::with_capacity(outer);
            for i in 0..outer {
                items.push(nest(py, &flat[i * stride..(i + 1) * stride], rest)?);
            }
            Ok(PyList::new(py, items)?.into_any().unbind())
        }
    }
}

// ---------------------------------------------------------------------------
// `amax` -- a maximum *value* along one dimension, without the argmax.
//
// candle has no such kernel. `Tensor::max`, `max_keepdim`, `min`, `min_keepdim`
// and `max_all` are all public, and all five funnel into
// `Tensor::reduce_impl(.., ReduceOp::Max)`, which is `cpu_backend::ReduceIndex`
// -- the *index-tracking* reduction that `argmax` also uses. Its inner loop is
//
//     for (src_i, &s) in src.iter().enumerate() {
//         if f(val, s) { acc = src_i; val = s }
//     }
//
// a data-dependent compare-and-select with a loop-carried dependency on `val`,
// one element at a time. `ReduceSum` next to it in the same file gets a
// vectorised path; this one does not. docs/numerics/SEQLEN.md §4.3 measured the
// consequence: at `[1, 9, 512, 512]` `float32`, candle's max over the last
// dimension takes 5.69 ms against upstream's 0.099 ms `amax` -- 57x -- and it
// was 24.3% of a `float32` prefill's main thread.
//
// So this is the second of the three routes docs/numerics/SEQLEN.md §5 left open. There
// is **no** public candle API that returns a maximum without the index (route
// one), and forking candle (route three) is not needed: `CustomOp1` +
// `Tensor::apply_op1_no_bwd` are `pub`, they hand a kernel the storage *and*
// the layout, and they are the same mechanism `WriteThrough` above already
// uses for in-place writes (docs/kernels/VIEWS.md §6.2). No `unsafe`, no fork.
//
// **The reduction is not the same function candle computes**, and the one place
// it differs is NaN. candle's predicate is `|x, y| x < y` -- "replace the
// accumulator when it is smaller than the candidate" -- and every comparison
// against a NaN is false, so a NaN that is not the *first* element is silently
// skipped. `max([3, nan, 1])` comes back `3.0` there, where upstream answers
// `nan` (docs/models/E2E_REAL.md; `aten.max.default` already works around it with a
// separate `x != x` pass, and `max.other` had the same fault in its second
// operand, docs/bindings/SPELLINGS.md). This kernel propagates, which is upstream's rule
// and also IEEE-754 `maximum`.
// ---------------------------------------------------------------------------

/// The two scalar predicates `amax_row` needs, over the dtypes `CpuStorage`
/// holds.
///
/// `is_nan` is a constant `false` for the integral arms, so the extra test in
/// the inner loop folds away entirely for them rather than costing a compare.
trait MaxScalar: Copy {
    fn is_nan(self) -> bool;
    /// The NaN of this type -- the answer when `is_nan` held for any element.
    /// The integral arms cannot reach it, since their `is_nan` is a constant
    /// `false`, so they return a value that is never read.
    fn nan() -> Self;
    /// Ordered greater-than: `false` if either side is NaN.
    fn greater(self, other: Self) -> bool;
}

macro_rules! max_scalar_float {
    ($ty:ty, $nan:expr) => {
        impl MaxScalar for $ty {
            #[inline(always)]
            fn is_nan(self) -> bool {
                self != self
            }
            #[inline(always)]
            fn nan() -> Self {
                $nan
            }
            #[inline(always)]
            fn greater(self, other: Self) -> bool {
                self > other
            }
        }
    };
}

macro_rules! max_scalar_int {
    ($ty:ty) => {
        impl MaxScalar for $ty {
            #[inline(always)]
            fn is_nan(self) -> bool {
                false
            }
            #[inline(always)]
            fn nan() -> Self {
                0
            }
            #[inline(always)]
            fn greater(self, other: Self) -> bool {
                self > other
            }
        }
    };
}

max_scalar_float!(f32, f32::NAN);
max_scalar_float!(f64, f64::NAN);
max_scalar_float!(half::f16, half::f16::NAN);
max_scalar_float!(half::bf16, half::bf16::NAN);
max_scalar_int!(u8);
max_scalar_int!(u32);
max_scalar_int!(i16);
max_scalar_int!(i32);
max_scalar_int!(i64);

/// How many independent accumulators the row reduction carries.
///
/// **This is most of the speed-up, and it is not about the argmax.** Dropping
/// the index removes a store; what removes the 20x is that a single accumulator
/// serialises the reduction on the latency of one compare-and-select per
/// element. Sixteen of them break that chain into sixteen independent ones, and
/// a fixed-size inner loop over them is what lets LLVM emit vector compares
/// rather than scalar ones. A maximum is associative and commutative, so
/// splitting the row across lanes cannot change the answer -- see `amax_row`'s
/// note for the one respect in which that is not quite true and why it is
/// unobservable.
///
/// Sixteen and not eight or thirty-two: measured, at the real score shape, 0.26
/// / 0.28 / 0.31 ms for 8 / 16 / 32 lanes on the shape below. docs/numerics/SEQLEN.md §7.3.
const AMAX_LANES: usize = 16;

/// The maximum of one contiguous row, NaN-propagating.
///
/// **The NaN test is a separate accumulator and that is a measurement, not a
/// preference.** The direct spelling of an IEEE-754 `maximum` --
/// `if v > acc || v.is_nan() { acc = v }` -- is correct and **8x slower than
/// this** (2.27 ms against 0.28 at the score shape), because the compound
/// condition stops LLVM recognising the loop as a max reduction. Carrying "did
/// any element fail to be a number" in its own lane array leaves the max itself
/// a plain `if v > acc`, which vectorises, and costs 0.08 ms.
///
/// The lane array is `u32` rather than `bool` for the same reason: `bool` lanes
/// are one byte against the value's four, so the two accumulators have
/// different vector widths and LLVM pays to reconcile them on every iteration
/// -- 0.83 ms with `bool`, 0.28 with `u32`, same arithmetic. All five variants
/// and their timings are in docs/numerics/SEQLEN.md §7.3.
///
/// The one-comparison spelling `!(v <= acc)` is not merely slower, it is
/// **wrong**: once `acc` is a NaN every subsequent `v <= NaN` is false, so the
/// next ordinary value replaces the NaN and it is lost again.
///
/// **Order.** A maximum involves no arithmetic, so there is no rounding to
/// reassociate and the multi-accumulator answer is the single-accumulator
/// answer -- for every input except one: a row containing both `-0.0` and
/// `+0.0`. The rule here, like candle's and like upstream's (measured:
/// `amax([-0., 0.])` is `-0.` and `amax([0., -0.])` is `0.`), keeps the *first*
/// of two equal elements, and splitting a row across lanes can change which
/// equal element is first. That distinguishes `-0.0` from `+0.0` and nothing
/// else, because those two compare equal. docs/numerics/SEQLEN.md §7.2 works through why
/// it cannot reach SDPA's output.
#[inline(always)]
fn amax_row<T: MaxScalar>(row: &[T]) -> T {
    // The caller checked; an empty row has no maximum to return.
    debug_assert!(!row.is_empty());
    let mut acc = [row[0]; AMAX_LANES];
    let mut nan = [0u32; AMAX_LANES];
    let mut chunks = row.chunks_exact(AMAX_LANES);
    for chunk in &mut chunks {
        // `zip`, not `acc[lane]`: indexing a fixed-size array by a loop
        // variable leaves a bounds check the vectoriser will not cross.
        for ((slot, flag), &v) in acc.iter_mut().zip(nan.iter_mut()).zip(chunk.iter()) {
            if v.greater(*slot) {
                *slot = v;
            }
            *flag |= v.is_nan() as u32;
        }
    }
    let mut best = acc[0];
    let mut any_nan = 0u32;
    for lane in 0..AMAX_LANES {
        if acc[lane].greater(best) {
            best = acc[lane];
        }
        any_nan |= nan[lane];
    }
    for &v in chunks.remainder() {
        if v.greater(best) {
            best = v;
        }
        any_nan |= v.is_nan() as u32;
    }
    // The seed is `row[0]`, so a leading NaN is already sitting in every lane
    // and `greater` never displaces it -- but it is also flagged, so the answer
    // comes from here either way.
    if any_nan != 0 {
        T::nan()
    } else {
        best
    }
}

/// The same reduction over a row whose elements are `stride` apart.
///
/// Reached when the reduced dimension is not the innermost one. It gets the
/// lanes but not the contiguity, which is the part that matters least: the
/// dependency chain is what the lanes break.
#[inline(always)]
fn amax_strided<T: MaxScalar>(src: &[T], start: usize, n: usize, stride: usize) -> T {
    debug_assert!(n > 0);
    let mut acc = [src[start]; AMAX_LANES];
    let mut nan = [0u32; AMAX_LANES];
    let full = n - (n % AMAX_LANES);
    let mut i = 0;
    while i < full {
        for lane in 0..AMAX_LANES {
            let v = src[start + (i + lane) * stride];
            if v.greater(acc[lane]) {
                acc[lane] = v;
            }
            nan[lane] |= v.is_nan() as u32;
        }
        i += AMAX_LANES;
    }
    let mut best = acc[0];
    let mut any_nan = 0u32;
    for lane in 0..AMAX_LANES {
        if acc[lane].greater(best) {
            best = acc[lane];
        }
        any_nan |= nan[lane];
    }
    while i < n {
        let v = src[start + i * stride];
        if v.greater(best) {
            best = v;
        }
        any_nan |= v.is_nan() as u32;
        i += 1;
    }
    if any_nan != 0 {
        T::nan()
    } else {
        best
    }
}

/// `amax_row`/`amax_strided` over every slice of a **contiguous** layout.
///
/// Contiguity is the caller's job (`amax_keepdim` makes it so), which is what
/// keeps this to two branches instead of candle's three: there is no strided
/// odometer here, because there is no strided input.
fn amax_reduce<T: MaxScalar>(
    src: &[T],
    layout: &Layout,
    dim: usize,
) -> candle_core::Result<Vec<T>> {
    let dims = layout.dims();
    let n = dims[dim];
    let dst_len: usize = dims.iter().product::<usize>() / n;
    let (o1, o2) = layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(
            "torch._C shim: amax reached its kernel with a non-contiguous layout -- \
             tensor.rs::amax_keepdim is supposed to have made it contiguous first"
                .to_string(),
        )
    })?;
    if src.len() < o2 {
        return Err(candle_core::Error::Msg(format!(
            "torch._C shim: amax's layout addresses {o2} elements of a {}-element buffer",
            src.len()
        )));
    }
    let src = &src[o1..o2];
    let stride = layout.stride()[dim];
    let mut dst = Vec::with_capacity(dst_len);
    if stride == 1 {
        for i in 0..dst_len {
            dst.push(amax_row(&src[i * n..i * n + n]));
        }
    } else {
        // Contiguous, but reducing an outer dimension: slice `i` starts at
        // `(i / stride) * stride * n + (i % stride)`. This is candle's own
        // decomposition of the same index, and it is exact because the layout
        // is contiguous, so `stride` is the product of the extents below `dim`.
        for i in 0..dst_len {
            let start = (i / stride) * stride * n + (i % stride);
            dst.push(amax_strided(src, start, n, stride));
        }
    }
    Ok(dst)
}

/// The `CustomOp1` that carries `amax_reduce` across candle's storage boundary.
struct AMax {
    dim: usize,
}

impl candle_core::CustomOp1 for AMax {
    fn name(&self) -> &'static str {
        "torch._C shim: amax"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, candle_core::Shape)> {
        let dims = layout.dims();
        if self.dim >= dims.len() {
            return Err(candle_core::Error::Msg(format!(
                "torch._C shim: amax over dimension {} of a rank-{} tensor",
                self.dim,
                dims.len()
            )));
        }
        if dims.iter().product::<usize>() == 0 {
            return Err(candle_core::Error::Msg(
                "torch._C shim: amax of an empty tensor".to_string(),
            ));
        }
        let mut out_dims = dims.to_vec();
        out_dims[self.dim] = 1;
        macro_rules! reduce {
            ($arm:ident, $values:expr) => {
                CpuStorage::$arm(amax_reduce($values, layout, self.dim)?)
            };
        }
        let out = match storage {
            CpuStorage::U8(v) => reduce!(U8, v),
            CpuStorage::U32(v) => reduce!(U32, v),
            CpuStorage::I16(v) => reduce!(I16, v),
            CpuStorage::I32(v) => reduce!(I32, v),
            CpuStorage::I64(v) => reduce!(I64, v),
            CpuStorage::BF16(v) => reduce!(BF16, v),
            CpuStorage::F16(v) => reduce!(F16, v),
            CpuStorage::F32(v) => reduce!(F32, v),
            CpuStorage::F64(v) => reduce!(F64, v),
            // The remaining `CpuStorage` arms are candle's sub-byte float
            // types, which this shim has no `TorchDType` for at all -- so a
            // tensor cannot be carrying one when it reaches here.
            _ => {
                return Err(candle_core::Error::Msg(
                    "torch._C shim: amax has no kernel for this candle dtype -- \
                     tensor.rs::AMax names the ones it reduces"
                        .to_string(),
                ))
            }
        };
        Ok((out, candle_core::Shape::from(out_dims)))
    }
}

/// The maximum along `dim`, keeping the reduced dimension as `1`.
///
/// Drop-in for `Tensor::max_keepdim(dim)` apart from the NaN rule above, and
/// the reason to prefer it is docs/numerics/SEQLEN.md §7.
pub(crate) fn amax_keepdim(source: &Tensor, dim: usize) -> candle_core::Result<Tensor> {
    // Free when it already is one -- candle's `contiguous` clones the handle
    // rather than the buffer in that case, which is every call SDPA makes.
    let source = source.contiguous()?;
    source.apply_op1_no_bwd(&AMax { dim })
}

// ---------------------------------------------------------------------------
// Scaling and causal masking, in one pass.
//
// `sdpa_flash_cpu`'s default branch used to spend three full passes over the
// `[batch, head, S, S]` score matrix, plus an `S x S` allocation, getting from
// the raw `q @ kT` product to a masked, scaled one:
//
//     scores.affine(scale, 0.0)                 one pass, read + write
//     build an S x S `Vec<f64>` of 0 / -inf     scalar push loop, then a
//                                               narrowing pass to `acc`
//     scores.broadcast_add(&mask)               another pass, read + write
//
// Measured inside the op at S=1024 (docs/numerics/SEQLEN.md §8.2): 1.450 + 2.100 +
// 2.133 = 5.68 ms of a 21.2 ms call, thirty times a forward. The mask is the
// same mask on all thirty of those calls and on every forward after it.
//
// This does the whole of it in one pass and allocates nothing but the output.
// ---------------------------------------------------------------------------

/// The elements `scale_and_causal_mask` can compute, which are exactly the two
/// `sdpa_flash_cpu` accumulates in -- `f32`, and `f64` when the caller asked
/// for `float64`. Reduced precision never reaches here: SDPA widens `f16` and
/// `bf16` to `f32` before the score matrix exists.
trait ScoreScalar: Copy {
    const ZERO: Self;
    const NEG_INFINITY: Self;
    /// The multiplier candle would have used. `Affine` narrows the `f64`
    /// scale to the tensor's own dtype *before* multiplying
    /// (`T::from_f64(self.0)`), so narrowing here too is not a shortcut --
    /// multiplying in `f64` and narrowing after would round twice and is a
    /// different number.
    fn from_f64(v: f64) -> Self;
    fn mul_add_zero(self, mul: Self) -> Self;
    fn add(self, other: Self) -> Self;
}

macro_rules! score_scalar {
    ($t:ty) => {
        impl ScoreScalar for $t {
            const ZERO: Self = 0.0;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;
            #[inline(always)]
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            #[inline(always)]
            fn mul_add_zero(self, mul: Self) -> Self {
                // `v * mul + add` with `add` zero, spelled out rather than
                // simplified to `v * mul`. The `+ 0.0` is not a no-op: it
                // turns a `-0.0` product into `+0.0`, and candle's `Affine`
                // does it, so dropping it would be a different answer for
                // every score whose product is a negative zero.
                //
                // Two operations and not `mul_add`: Rust does not contract
                // this to an FMA and neither does candle, and an FMA would
                // round once where these round twice.
                self * mul + Self::ZERO
            }
            #[inline(always)]
            fn add(self, other: Self) -> Self {
                self + other
            }
        }
    };
}

score_scalar!(f32);
score_scalar!(f64);

/// One matrix of the batch: `v * scale + 0.0` everywhere, and `+ -inf` on the
/// strictly-upper triangle.
///
/// **`+ -inf` and not `= -inf`.** For a finite product the two agree, but for
/// a `+inf` product `+inf + -inf` is a NaN where an assignment would have
/// written `-inf`, and a NaN product stays a NaN either way. The old two-op
/// path went through `broadcast_add`, so the addition is what has to be
/// reproduced -- not the intent behind it.
#[inline]
fn scale_and_mask_rows<T: ScoreScalar>(
    src: &[T],
    out: &mut Vec<T>,
    rows: usize,
    cols: usize,
    mul: T,
) {
    for r in 0..rows {
        let row = &src[r * cols..(r + 1) * cols];
        // Upper-left aligned, which is the alignment `sdpa_flash_cpu`
        // measured upstream to use: column `c` survives when `c <= r`.
        let keep = (r + 1).min(cols);
        out.extend(row[..keep].iter().map(|&v| v.mul_add_zero(mul)));
        out.extend(
            row[keep..]
                .iter()
                .map(|&v| v.mul_add_zero(mul).add(T::NEG_INFINITY)),
        );
    }
}

/// The `CustomOp1` that carries `scale_and_mask_rows` across the storage
/// boundary, the same mechanism `AMax` above uses and docs/kernels/VIEWS.md §6.2
/// describes.
struct ScaleCausal {
    scale: f64,
}

impl candle_core::CustomOp1 for ScaleCausal {
    fn name(&self) -> &'static str {
        "torch._C shim: scale + causal mask"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, candle_core::Shape)> {
        let dims = layout.dims();
        if dims.len() < 2 {
            return Err(candle_core::Error::Msg(format!(
                "torch._C shim: a causal mask needs a rank-2 or deeper score \
                 matrix, got rank {}",
                dims.len()
            )));
        }
        let (rows, cols) = (dims[dims.len() - 2], dims[dims.len() - 1]);
        // Contiguity is the caller's job, as it is for `AMax`. `q.matmul(&kt)`
        // hands back a fresh contiguous tensor, so this never fires from SDPA
        // -- it is here so that a future caller with a view gets an error
        // rather than a silently transposed answer.
        let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg(
                "torch._C shim: scale + causal mask wants a contiguous score matrix"
                    .to_string(),
            )
        })?;
        let n = end - start;
        macro_rules! run {
            ($arm:ident, $values:expr, $t:ty) => {{
                let src = &$values[start..end];
                let mut out: Vec<$t> = Vec::with_capacity(n);
                let mul = <$t as ScoreScalar>::from_f64(self.scale);
                if rows * cols > 0 {
                    for mat in src.chunks_exact(rows * cols) {
                        scale_and_mask_rows(mat, &mut out, rows, cols, mul);
                    }
                }
                CpuStorage::$arm(out)
            }};
        }
        let out = match storage {
            CpuStorage::F32(v) => run!(F32, v, f32),
            CpuStorage::F64(v) => run!(F64, v, f64),
            _ => {
                return Err(candle_core::Error::Msg(
                    "torch._C shim: scale + causal mask has a kernel for float32 and \
                     float64 only -- SDPA widens the reduced precisions before the \
                     score matrix exists"
                        .to_string(),
                ))
            }
        };
        Ok((out, candle_core::Shape::from(dims.to_vec())))
    }
}

/// `scores.affine(scale, 0.0)` followed by adding an upper-triangular `-inf`
/// mask, in one pass and with no mask allocated.
///
/// Bit-for-bit the two-op spelling it replaces, element by element -- there is
/// no reassociation to argue about because nothing is reduced here.
/// docs/numerics/SEQLEN.md §8.3 has the argument and §8.4 the test that would catch it
/// being wrong.
pub(crate) fn scale_and_causal_mask(source: &Tensor, scale: f64) -> candle_core::Result<Tensor> {
    let source = source.contiguous()?;
    source.apply_op1_no_bwd(&ScaleCausal { scale })
}

// ---------------------------------------------------------------------------
// The transposed copy, blocked.
//
// docs/numerics/SEQLEN.md §8.12 named this as the one clean kernel win left in SDPA:
// `k.transpose(2, 3).contiguous()` moves 2.4 MB at ~3.7 GB/s, against
// upstream's 0.134 ms for the same bytes -- 1.15 ms of a 13.64 ms per-call
// gap at `S=1024`, which is 8% of the SDPA gap and 7% of the model gap.
//
// The reason it is slow is candle's `copy_strided_src`, which walks a
// transposed layout **one element at a time**: for each output element it
// recomputes a multi-dimensional index and reads a source address `head_dim`
// floats away from the last one. Every read is a cache miss once the source is
// bigger than L2.
//
// **This is the only entry in §8.12's table that is bit-identical by
// construction**, and that is why it is the one taken. There is no arithmetic
// here at all -- every output element is a *copy* of exactly one input element,
// so there is no summation order to reassociate and no rounding to move. The
// only thing a blocked traversal changes is the order in which the same
// assignments happen.
//
// Contrast with the trap recorded beside it in §8.5: simply *dropping* the
// `contiguous` is 5% faster and **moves the S=6 digest**, because it lets
// Accelerate take a transposed GEMM with a different accumulation order. That
// was tried and rejected. This makes the copy faster; it does not remove it.
// ---------------------------------------------------------------------------

/// Cache block, in elements per side.
///
/// 32x32 `f32` is 4 KB read plus 4 KB written, so both blocks are live in L1
/// together with room to spare on every target this builds for. The value is
/// not tuned per machine: the win is going from "one cache line per element"
/// to "one cache line per 16 elements", and any block that fits in L1 gets
/// essentially all of it.
const TRANSPOSE_BLOCK: usize = 32;

/// Is this layout "the last two dimensions of a contiguous tensor, swapped"?
///
/// Returns `(batches, src_rows, src_cols, offset)` when it is: the source is
/// `batches` consecutive `src_rows x src_cols` row-major matrices starting at
/// `offset`, and the output is each of them transposed.
///
/// Written as a recogniser rather than assumed, because the caller is one line
/// in `sdpa_flash_cpu` and the guarantee has to hold for whatever that line is
/// handed. Anything it does not recognise falls back to candle's own
/// `contiguous`, so a layout this does not understand is slow rather than
/// wrong.
fn transposed_plan(layout: &Layout) -> Option<(usize, usize, usize, usize)> {
    let dims = layout.dims();
    let strides = layout.stride();
    if dims.len() < 2 {
        return None;
    }
    let last = dims.len() - 1;
    // The swapped pair: the second-to-last dimension is the one packed in
    // storage, and the last one steps by the packed extent.
    if strides[last - 1] != 1 || strides[last] != dims[last - 1] {
        return None;
    }
    // Everything above the pair must be contiguous over the pair's area, or
    // the batches are not consecutive and the offset arithmetic below is
    // wrong.
    let area = dims[last] * dims[last - 1];
    let mut expected = area;
    for i in (0..last - 1).rev() {
        if strides[i] != expected {
            return None;
        }
        expected *= dims[i];
    }
    let batches: usize = dims[..last - 1].iter().product();
    // src is `dims[last] x dims[last-1]` row-major; the output transposes it.
    Some((batches, dims[last], dims[last - 1], layout.start_offset()))
}

/// One batch: `dst[r * rows + c] = src[c * cols + r]`, in cache blocks.
///
/// `rows` and `cols` name the **source** matrix, which is `rows x cols`
/// row-major; the destination is `cols x rows`.
#[inline]
fn transpose_block<T: Copy>(src: &[T], dst: &mut [T], rows: usize, cols: usize) {
    for c0 in (0..rows).step_by(TRANSPOSE_BLOCK) {
        let c_end = (c0 + TRANSPOSE_BLOCK).min(rows);
        for r0 in (0..cols).step_by(TRANSPOSE_BLOCK) {
            let r_end = (r0 + TRANSPOSE_BLOCK).min(cols);
            for c in c0..c_end {
                let row = &src[c * cols..c * cols + cols];
                for r in r0..r_end {
                    dst[r * rows + c] = row[r];
                }
            }
        }
    }
}

/// The `CustomOp1` that carries `transpose_block` across candle's storage
/// boundary, for every dtype `CpuStorage` has an owned `Vec` of.
struct TransposedCopy;

impl candle_core::CustomOp1 for TransposedCopy {
    fn name(&self) -> &'static str {
        "torch._C shim: transposed copy"
    }

    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, candle_core::Shape)> {
        // Re-checked here and not only at the entry point: this is the
        // function that indexes raw storage, so it is the one that must not
        // trust a caller.
        let Some((batches, rows, cols, offset)) = transposed_plan(layout) else {
            return Err(candle_core::Error::Msg(
                "torch._C shim: transposed copy on a layout that is not a \
                 last-two-swapped view -- tensor.rs::transposed_contiguous is \
                 supposed to have routed this to candle's own contiguous"
                    .to_string(),
            ));
        };
        let area = rows * cols;
        macro_rules! run {
            ($arm:ident, $values:expr) => {{
                let src = $values;
                if offset + batches * area > src.len() {
                    return Err(candle_core::Error::Msg(
                        "torch._C shim: transposed copy would read past the storage"
                            .to_string(),
                    ));
                }
                let mut out = vec![Default::default(); batches * area];
                for b in 0..batches {
                    transpose_block(
                        &src[offset + b * area..offset + (b + 1) * area],
                        &mut out[b * area..(b + 1) * area],
                        rows,
                        cols,
                    );
                }
                CpuStorage::$arm(out)
            }};
        }
        let out = match storage {
            CpuStorage::U8(v) => run!(U8, v),
            CpuStorage::U32(v) => run!(U32, v),
            CpuStorage::I16(v) => run!(I16, v),
            CpuStorage::I32(v) => run!(I32, v),
            CpuStorage::I64(v) => run!(I64, v),
            CpuStorage::BF16(v) => run!(BF16, v),
            CpuStorage::F16(v) => run!(F16, v),
            CpuStorage::F32(v) => run!(F32, v),
            CpuStorage::F64(v) => run!(F64, v),
            // candle's sub-byte float types, which this shim has no
            // `TorchDType` for -- unreachable from any tensor it can build.
            _ => {
                return Err(candle_core::Error::Msg(
                    "torch._C shim: transposed copy has no kernel for this candle dtype"
                        .to_string(),
                ))
            }
        };
        Ok((out, candle_core::Shape::from(layout.dims().to_vec())))
    }
}

/// `t.contiguous()`, but blocked when `t` is a transposed view.
///
/// **Drop-in for `Tensor::contiguous`, and identical to it element for
/// element.** The two fast exits are the safety property: an already-contiguous
/// tensor goes to candle (which clones the handle, not the buffer), and any
/// layout `transposed_plan` does not recognise goes to candle as well. So the
/// worst case of a layout this does not understand is candle's own speed, never
/// a wrong answer.
pub(crate) fn transposed_contiguous(t: &Tensor) -> candle_core::Result<Tensor> {
    if t.layout().is_contiguous() {
        return t.contiguous();
    }
    match transposed_plan(t.layout()) {
        Some(_) => t.apply_op1_no_bwd(&TransposedCopy),
        None => t.contiguous(),
    }
}

/// `torch._C._has_storage(tensor)`.
///
/// A module-level function upstream too, not a tensor member. It is the first
/// wall on the `torch.save` path -- `torch/_tensor.py:328`, inside
/// `_reduce_ex_internal`, before anything else about the tensor is looked at
/// (docs/models/SAVE.md §1.1) -- and `torch/_tensor.py:158` asks it again on the
/// deepcopy path.
///
/// The argument is anything, because upstream's takes anything: it is asked
/// about `FakeTensor`s and wrapper subclasses as well as ordinary ones. What is
/// not a `TensorBase` at all gets a refusal naming what it was handed rather
/// than a `False`, because "no storage" and "not a tensor" are different
/// answers and only one of them is this function's to give.
#[pyfunction]
#[pyo3(name = "_has_storage")]
pub fn has_storage(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    match value.extract::<PyRef<'_, PyTensorBase>>() {
        Ok(t) => Ok(t.has_storage()),
        Err(_) => Err(not_implemented(format!(
            "torch._C._has_storage in torch._C shim: expected a tensor, got {}",
            value
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_default()
        ))),
    }
}


// ---------------------------------------------------------------------------
// W5: `grad_fn` as a nullness. docs/training/BACKWARD4.md.
// ---------------------------------------------------------------------------

/// Grad mode, mirrored out of `bootstrap.py`'s `_install_grad_mode` dict.
///
/// The dict stays the source of truth for `torch.is_grad_enabled()` -- moving
/// it here would put a Python-visible flag in two places. What is mirrored is
/// only what the *door* needs, because the door runs once per op and a
/// `PyDict_GetItem` plus a `PyObject_IsTrue` per op is a cost paid by every
/// caller including the ones that never differentiate anything. A relaxed load
/// of an `AtomicBool` is the same shape as `capture::is_active`, which
/// docs/graph/CAPTURE.md §7 already measured at the same door.
///
/// `Ordering::Relaxed` for the same reason capture uses it: there is nothing
/// else for this flag to be ordered *against*. A thread that flips it and then
/// dispatches does both under the GIL.
static GRAD_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// `torch._C._set_throw_on_mutable_data_ptr(tensor)`.
///
/// `fake_tensor.py:943` calls this on every `FakeTensor` as it is constructed,
/// so `torch.export` reaches it once per input. It sets the per-tensor bit that
/// `data_ptr()` refuses on -- see the field's own comment on `PyTensorBase`.
///
/// It takes the tensor by `PyRef` rather than by value because upstream's
/// mutates the object the caller passed; taking a clone would set the bit on a
/// copy and leave the caller's `FakeTensor` answering an address.
///
/// **One-way, and deliberately.** Upstream has no un-setter, and adding one
/// here would let a caller launder a fake tensor into one whose `data_ptr()`
/// answers. There is nothing to restore: the bit is set at construction and the
/// tensor is fake for the whole of its life.
#[pyfunction]
#[pyo3(name = "_set_throw_on_mutable_data_ptr")]
pub fn set_throw_on_mutable_data_ptr(tensor: PyRef<'_, PyTensorBase>) {
    tensor
        .throw_on_mutable_data_ptr
        .store(true, std::sync::atomic::Ordering::Relaxed);
}

/// `torch._C._shim_throws_on_mutable_data_ptr(tensor)` -- readable so a test
/// can assert the bit rather than infer it from a raised exception.
/// `torch._C._set_warn_deprecated_on_mutable_data_ptr(tensor)`.
///
/// The sibling of `_set_throw_on_mutable_data_ptr`, and **not** an alias for
/// it: this one leaves `data_ptr()` answering and adds a `UserWarning`. See the
/// field comment on `PyTensorBase` for why the two are kept apart.
#[pyfunction]
#[pyo3(name = "_set_warn_deprecated_on_mutable_data_ptr")]
pub fn set_warn_deprecated_on_mutable_data_ptr(tensor: PyRef<'_, PyTensorBase>) {
    tensor
        .warn_deprecated_on_mutable_data_ptr
        .store(true, std::sync::atomic::Ordering::Relaxed);
}

/// `torch._C._shim_warns_on_mutable_data_ptr(tensor)` -- readable so a test can
/// assert the bit rather than catch a warning.
#[pyfunction]
#[pyo3(name = "_shim_warns_on_mutable_data_ptr")]
pub fn warns_on_mutable_data_ptr(tensor: PyRef<'_, PyTensorBase>) -> bool {
    tensor
        .warn_deprecated_on_mutable_data_ptr
        .load(std::sync::atomic::Ordering::Relaxed)
}

#[pyfunction]
#[pyo3(name = "_shim_throws_on_mutable_data_ptr")]
pub fn throws_on_mutable_data_ptr(tensor: PyRef<'_, PyTensorBase>) -> bool {
    tensor
        .throw_on_mutable_data_ptr
        .load(std::sync::atomic::Ordering::Relaxed)
}

#[pyfunction]
#[pyo3(name = "_shim_set_grad_enabled_flag")]
pub fn set_grad_enabled_flag(value: bool) {
    GRAD_ENABLED.store(value, std::sync::atomic::Ordering::Relaxed);
}

/// Grad mode off for the duration of a scope, restored on drop.
///
/// Only the *door's* mirror is touched, not `bootstrap.py`'s `state["grad"]`:
/// a caller who asks `torch.is_grad_enabled()` from inside a tape backward is
/// asking about Python's grad mode, which nothing here changed. What is
/// suppressed is graph *marking*, which is upstream's `AutoGradMode` and not
/// upstream's `GradMode` python flag.
pub struct NoGradGuard(bool);

impl NoGradGuard {
    pub fn enter() -> Self {
        Self(GRAD_ENABLED.swap(false, std::sync::atomic::Ordering::Relaxed))
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.store(self.0, std::sync::atomic::Ordering::Relaxed);
    }
}

#[pyfunction]
#[pyo3(name = "_shim_grad_enabled_flag")]
pub fn grad_enabled_flag() -> bool {
    GRAD_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Ops whose output is **not** a differentiable function of their tensor
/// arguments, and which therefore leave a leaf behind even when handed a
/// parameter.
///
/// Every entry was checked against upstream rather than reasoned about
/// (docs/training/BACKWARD4.md §3.2): `torch.ops.aten.<op>(param, ...)` on torch 2.13.0
/// reports `grad_fn is None` for all of them.
///
///   * `detach` is the definition of the boundary -- it is how a caller *asks*
///     for a leaf, and marking its output would make `.detach()` a no-op.
///   * the `*_like` family and `new_ones`/`new_zeros` read a tensor for its
///     shape and dtype and nothing else; the values do not flow.
///   * `lift_fresh` returns a fresh leaf by name.
///   * `view.dtype` reinterprets bytes rather than computing.
///   * `histc`, `multinomial` and `randperm` have no derivative upstream.
///
/// Nothing else in `_aten_implemented()`'s 197 needs an entry: the remaining
/// non-differentiable ops (`argmax`, `sort`'s indices, comparisons, `one_hot`)
/// return integer or boolean tensors and are excluded by the dtype test in
/// `mark_from_op` instead, which is upstream's rule stated where upstream
/// states it -- only floating and complex tensors can carry a gradient.
const NOT_DIFFERENTIABLE: &[&str] = &[
    "aten.detach.default",
    "aten.empty_like.default",
    "aten.full_like.default",
    "aten.ones_like.default",
    "aten.zeros_like.default",
    "aten.new_ones.default",
    "aten.new_zeros.default",
    "aten.lift_fresh.default",
    "aten.view.dtype",
    "aten.histc.default",
    "aten.multinomial.default",
    "aten.randperm.default",
];

/// Walks a dispatcher argument, collecting every `TensorBase` it contains.
///
/// Flat rather than recursive-without-limit: aten arguments nest one level at
/// most (`cat([a, b], 0)`, `where(c, a, b)`), and a bounded walk cannot be made
/// to loop by a caller.
fn collect_tensors<'py>(value: &Bound<'py, PyAny>, out: &mut Vec<Bound<'py, PyAny>>) {
    if value.cast::<PyTensorBase>().is_ok() {
        out.push(value.clone());
        return;
    }
    if let Ok(seq) = value.cast::<PyList>() {
        for item in seq.iter() {
            if item.cast::<PyTensorBase>().is_ok() {
                out.push(item);
            }
        }
        return;
    }
    if let Ok(seq) = value.cast::<PyTuple>() {
        for item in seq.iter() {
            if item.cast::<PyTensorBase>().is_ok() {
                out.push(item);
            }
        }
    }
}

/// The first pass of `mark_from_op`: does any tensor operand require a
/// gradient, or come from an op that did?
///
/// Deliberately allocation-free. It is on the hot path of every dispatch in the
/// process, including the ones with no autograd anywhere near them, and
/// `docs/training/BACKWARD3.md` §4's last row is the reason: a SmolLM2-135M forward at
/// `S=8` is 1862 dispatches, and a `Vec` per dispatch to answer `false` 1862
/// times would be a cost paid by callers who never intend to differentiate.
fn any_operand_requires_grad(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> bool {
    fn wants(value: &Bound<'_, PyAny>) -> bool {
        if let Ok(cell) = value.cast::<PyTensorBase>() {
            return match cell.try_borrow() {
                Ok(t) => t.requires_grad || t.from_op.is_some(),
                Err(_) => false,
            };
        }
        false
    }
    fn wants_nested(value: &Bound<'_, PyAny>) -> bool {
        if wants(value) {
            return true;
        }
        if let Ok(seq) = value.cast::<PyList>() {
            return seq.iter().any(|item| wants(&item));
        }
        if let Ok(seq) = value.cast::<PyTuple>() {
            return seq.iter().any(|item| wants(&item));
        }
        false
    }
    if args.iter().any(|item| wants_nested(&item)) {
        return true;
    }
    match kwargs {
        Some(kwargs) => kwargs.iter().any(|(_, value)| wants_nested(&value)),
        None => false,
    }
}

/// The door's autograd half: decide whether the outputs of `op` are leaves.
///
/// This is **all** of W5. Upstream's condition for `grad_fn is not None` is
/// "an op ran, under grad mode, on an operand that requires a gradient, and the
/// result can carry one", and each clause below is one of those:
///
/// ```text
/// grad mode is on                     GRAD_ENABLED
/// the op is differentiable            NOT_DIFFERENTIABLE
/// some operand requires a gradient    requires_grad || from_op.is_some()
/// the result can carry one            floating or complex
/// the result is a new tensor          not identical to any operand
/// ```
///
/// The last clause is the one that is not upstream's, and it is the round's
/// deliberate stopping point rather than an oversight. An in-place op returns
/// its own receiver, so marking there would rewrite the leafness of a tensor
/// that already exists -- which is `optimizer.step()`'s `add_` turning every
/// parameter in the model into a non-leaf. Upstream can afford to mark it
/// because upstream has version counters and a leaf-mutation refusal
/// (docs/training/BACKWARD2.md §1.5, W10); here the honest answer is to leave leafness
/// alone, and `docs/training/BACKWARD4.md` §4.2 records the divergence that follows:
/// an activation mutated in place stays a leaf here and does not upstream.
///
/// Errors are not propagated: a marking failure must not turn a working
/// dispatch into a raise. A tensor that could not be borrowed (because the
/// caller is holding it mutably) simply stays a leaf, which is the answer the
/// shim gave before this round for every tensor.
/// Returns **whether an output was marked** -- i.e. whether upstream would
/// have built a graph node for this call. `docs/training/BACKWARD7.md` §2: the eager
/// recorder is gated on this `bool` and not on a test of its own, so that a
/// dispatch which differentiates nothing pays one value already in a register
/// rather than a second walk of the argument tuple. It is also the *definition*
/// the recorder wants -- an op is on the eager tape exactly when its result has
/// a `grad_fn`.
#[must_use]
pub fn mark_from_op(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    out: &Py<PyAny>,
) -> bool {
    if !GRAD_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    if NOT_DIFFERENTIABLE.contains(&op) {
        return false;
    }
    // Two passes over the arguments, and the first one allocates nothing.
    //
    // The overwhelmingly common case in this shim is *no* tensor in the call
    // requiring a gradient -- every inference forward, every golden case, every
    // `torch.load`. That case has to leave the door as it found it, so the
    // first pass answers "does anybody want this?" by borrowing one `bool` per
    // tensor argument and returns before a `Vec` exists. Only a call that
    // really is on a gradient path pays for the second pass, which is the one
    // that needs the operand identities for the in-place test below.
    if !any_operand_requires_grad(args, kwargs) {
        return false;
    }
    let mut inputs: Vec<Bound<'_, PyAny>> = Vec::new();
    for item in args.iter() {
        collect_tensors(&item, &mut inputs);
    }
    if let Some(kwargs) = kwargs {
        for (_, value) in kwargs.iter() {
            collect_tensors(&value, &mut inputs);
        }
    }
    let mut outputs: Vec<Bound<'_, PyAny>> = Vec::new();
    collect_tensors(out.bind(py), &mut outputs);
    let mut marked = false;
    for output in outputs {
        if inputs.iter().any(|input| input.is(&output)) {
            continue;
        }
        let cell = match output.cast::<PyTensorBase>() {
            Ok(cell) => cell,
            Err(_) => continue,
        };
        if let Ok(mut borrowed) = cell.try_borrow_mut() {
            if !(borrowed.tag.is_floating_point() || borrowed.tag.is_complex()) {
                continue;
            }
            if borrowed.from_op.is_none() {
                borrowed.from_op = Some(op.into());
            }
            marked = true;
        }
    }
    marked
}


// ---------------------------------------------------------------------------
// `complex_ops` -- the ops taught `Repr::Complex` by name
// ---------------------------------------------------------------------------
//
// docs/kernels/COMPLEX2.md. This module is the counterpart of `quant.rs` and
// `vulkan::dispatch`: the arm refuses everywhere by default (`tensor()`), and
// capability arrives here, one operator at a time, each one having to say what
// it does with *both* halves.
//
// **The set is deliberately small and it is `llama4`'s, not "complex support".**
// `Llama4VisionRotaryEmbedding` / `apply_rotary_emb` is a closed pipeline --
// every complex value is produced by `polar` or `view_as_complex` and consumed
// by `view_as_real` inside one function, never escaping it -- so five ops carry
// the architecture. Anything outside that set refuses rather than approximates.
pub mod complex_ops {
    use super::*;

    /// `torch.view_as_complex(x)` -- the entrance from real data.
    ///
    /// **This is a copy, and upstream's is a view.** Measured on 2.13.0:
    ///
    /// ```text
    /// base = torch.tensor([[1., 2.]]); v = torch.view_as_complex(base)
    /// base[0, 0] = 99.;  v.tolist()   ->  [(99+2j)]
    /// ```
    ///
    /// A pair-of-tensors representation cannot alias an interleaved buffer,
    /// and choosing the pair was the decision that bought the correct `.shape`
    /// (docs/kernels/COMPLEX.md §3.2). So this is a **narrowing**, it is stated here
    /// rather than left to be discovered, it is asserted as a narrowing in
    /// `pytests/test_complex.py::test_view_as_complex_copies_where_upstream_aliases`,
    /// and it is safe for the models measured only because all three of
    /// `llama4`'s call sites feed a freshly computed expression that is never
    /// written to again. docs/kernels/COMPLEX2.md §6.
    pub fn view_as_complex(py: Python<'_>, input: &PyTensorBase) -> PyResult<Py<PyAny>> {
        const OP: &str = "aten.view_as_complex.default";
        // Upstream's own message, verbatim, for the dtype it cannot take.
        // `bfloat16` is refused by upstream too -- there is no
        // `complex(bfloat16)` -- so `complex_for_component` returning `None`
        // and this check are the same rule read from two sides.
        if TorchDType::complex_for_component(
            input.tag.storage().unwrap_or(candle_core::DType::U8),
        )
        .is_none()
        {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "view_as_complex is only supported for half, float and double \
                 tensors, but got a tensor of scalar type: {}",
                crate::aten::scalar_type_name(input.tag)
            )));
        }
        let t = input.tensor()?;
        let dims = t.dims();
        if dims.last().copied() != Some(2) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Tensor must have a last dimension of size 2",
            ));
        }
        let last = dims.len() - 1;
        // **`copy()`, not `contiguous()`, and the difference is measurable.**
        // candle's `contiguous()` returns `self.clone()` when the layout is
        // already contiguous, and a narrow to length 1 on the last axis *is*
        // contiguous -- so `torch.tensor([[1., 2.]])` produced halves that
        // still shared the base's storage, and `base[0,0] = 99.` showed
        // through. That is upstream's own behaviour, but only for the shapes
        // where the narrow happens to stay contiguous; for every other shape
        // the same code copied. **An aliasing rule that holds for some shapes
        // and not others is worse than either answer**, and it was found by
        // the narrowing test below rather than by reading this.
        //
        // `copy()` allocates unconditionally, which makes "a complex tensor in
        // this shim never shares storage with anything" true by construction.
        // Two other things lean on that: the `copy_` narrowing (there is no
        // view that could observe `replace_with`), and the absence of any need
        // to version-stamp the halves in `capture.rs`.
        let re = t
            .narrow(last, 0, 1)
            .and_then(|v| v.squeeze(last))
            .and_then(|v| v.copy())
            .map_err(|e| candle_err(OP, e))?;
        let im = t
            .narrow(last, 1, 1)
            .and_then(|v| v.squeeze(last))
            .and_then(|v| v.copy())
            .map_err(|e| candle_err(OP, e))?;
        wrap(py, PyTensorBase::complex(re, im)?)
    }

    /// `torch.view_as_real(z)` -- the exit, and the op that makes the pipeline
    /// closed.
    ///
    /// `stack([re, im], -1)`, which is the exact inverse of the narrow-and-
    /// squeeze above. Round-tripping is what `pytests/test_complex.py` checks
    /// element-wise against upstream, because losing the imaginary part is the
    /// failure that still returns plausible numbers and this is the one op
    /// that would show it.
    pub fn view_as_real(py: Python<'_>, input: &PyTensorBase) -> PyResult<Py<PyAny>> {
        const OP: &str = "aten.view_as_real.default";
        let (re, im) = match input.repr() {
            Repr::Complex { re, im } => (re, im),
            // Upstream's message, verbatim.
            _ => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "view_as_real is only supported for complex tensors",
                ))
            }
        };
        let out = Tensor::stack(&[re, im], re.dims().len()).map_err(|e| candle_err(OP, e))?;
        let tag = input
            .tag
            .to_real_tag()
            .expect("a Repr::Complex tag always has a real partner");
        finish_real(py, out, tag)
    }

    /// `torch.polar(abs, angle)` -- `llama4_text`'s entrance.
    ///
    /// `re = abs*cos(angle)`, `im = abs*sin(angle)`, computed in the component
    /// dtype. Both arguments must agree on dtype, which is upstream's rule and
    /// upstream's message.
    pub fn polar(
        py: Python<'_>,
        abs: &PyTensorBase,
        angle: &PyTensorBase,
    ) -> PyResult<Py<PyAny>> {
        const OP: &str = "aten.polar.default";
        if abs.tag != angle.tag {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Expected object of scalar type {} but got scalar type {} for \
                 second argument",
                crate::aten::scalar_type_name(abs.tag),
                crate::aten::scalar_type_name(angle.tag)
            )));
        }
        let a = abs.tensor()?;
        let th = angle.tensor()?;
        let re = th
            .cos()
            .and_then(|c| a.broadcast_mul(&c))
            .and_then(|v| v.contiguous())
            .map_err(|e| candle_err(OP, e))?;
        let im = th
            .sin()
            .and_then(|s| a.broadcast_mul(&s))
            .and_then(|v| v.contiguous())
            .map_err(|e| candle_err(OP, e))?;
        wrap(py, PyTensorBase::complex(re, im)?)
    }

    /// `z * w` for the three combinations that have a complex operand.
    ///
    /// `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`, and the mixed complex-by-real case
    /// scales both halves. **The fourth combination, real-by-complex, is not a
    /// separate rule** -- multiplication commutes and both orders route here.
    ///
    /// Everything else refuses: a complex operand against a meta, quantised or
    /// Vulkan one falls to `tensor()`'s refusal, which names the *other*
    /// tensor's representation rather than pretending this op knows what to do
    /// with it.
    pub fn mul(py: Python<'_>, lhs: &PyTensorBase, rhs: &PyTensorBase) -> PyResult<Py<PyAny>> {
        const OP: &str = "aten.mul.Tensor";
        match (lhs.repr(), rhs.repr()) {
            (Repr::Complex { re: a, im: b }, Repr::Complex { re: c, im: d }) => {
                let real = a
                    .broadcast_mul(c)
                    .and_then(|ac| b.broadcast_mul(d).and_then(|bd| ac.broadcast_sub(&bd)))
                    .and_then(|v| v.contiguous())
                    .map_err(|e| candle_err(OP, e))?;
                let imag = a
                    .broadcast_mul(d)
                    .and_then(|ad| b.broadcast_mul(c).and_then(|bc| ad.broadcast_add(&bc)))
                    .and_then(|v| v.contiguous())
                    .map_err(|e| candle_err(OP, e))?;
                wrap(py, PyTensorBase::complex(real, imag)?)
            }
            (Repr::Complex { re, im }, _) => {
                let r = rhs.tensor()?;
                let real = re
                    .broadcast_mul(r)
                    .and_then(|v| v.contiguous())
                    .map_err(|e| candle_err(OP, e))?;
                let imag = im
                    .broadcast_mul(r)
                    .and_then(|v| v.contiguous())
                    .map_err(|e| candle_err(OP, e))?;
                wrap(py, PyTensorBase::complex(real, imag)?)
            }
            (_, Repr::Complex { re, im }) => {
                let l = lhs.tensor()?;
                let real = re
                    .broadcast_mul(l)
                    .and_then(|v| v.contiguous())
                    .map_err(|e| candle_err(OP, e))?;
                let imag = im
                    .broadcast_mul(l)
                    .and_then(|v| v.contiguous())
                    .map_err(|e| candle_err(OP, e))?;
                wrap(py, PyTensorBase::complex(real, imag)?)
            }
            // Not reachable through `aten.rs`'s guard, which only routes here
            // when one side is complex. A `RuntimeError` rather than an
            // `unreachable!`, because a panic across the FFI boundary is worse
            // than a refusal and this is the arm a future caller gets wrong.
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "aten.mul.Tensor: complex kernel reached with two real operands",
            )),
        }
    }

    /// `z.real` and `z.imag`, which is how a test reads the two halves apart
    /// without `view_as_real` in the way. Field access, no arithmetic.
    pub fn part(py: Python<'_>, input: &PyTensorBase, imaginary: bool) -> PyResult<Py<PyAny>> {
        let op = if imaginary { "aten.imag.default" } else { "aten.real.default" };
        let (re, im) = input.complex_parts(op)?;
        let tag = input
            .tag
            .to_real_tag()
            .expect("a Repr::Complex tag always has a real partner");
        finish_real(py, if imaginary { im.clone() } else { re.clone() }, tag)
    }

    /// The ops taught this arm by name -- the complex counterpart of
    /// `torch._C._vulkan_ops()`, and the list `no_real_storage`'s refusal
    /// points a reader at.
    ///
    /// A constant rather than a doc sentence because the refusal names it and
    /// a refusal that names a stale list is worse than one that names none:
    /// `pytests/test_complex.py` checks this against the dispatch table, so
    /// the two cannot drift.
    pub const COMPLEX_OPS: &[&str] = &[
        "aten._to_copy.default",
        "aten._unsafe_view.default",
        "aten.alias.default",
        "aten.clone.default",
        "aten.complex.default",
        "aten.constant_pad_nd.default",
        "aten.contiguous.default",
        "aten.copy_.default",
        "aten.detach.default",
        "aten.imag.default",
        "aten.lift_fresh.default",
        "aten.mul.Scalar",
        "aten.mul.Tensor",
        "aten.polar.default",
        "aten.real.default",
        "aten.slice.Tensor",
        "aten.unsqueeze.default",
        "aten.view.default",
        "aten.view_as_complex.default",
        "aten.view_as_real.default",
    ];

    #[pyfunction]
    #[pyo3(name = "_complex_ops")]
    pub fn complex_ops_list() -> Vec<&'static str> {
        COMPLEX_OPS.to_vec()
    }

    /// `detach` / `alias` / `clone` / `contiguous` / `lift_fresh` on a complex
    /// tensor.
    ///
    /// **This is the op `llama4` needed that the five arithmetic ones did not
    /// cover, and it was found by running the sweep rather than by reasoning.**
    /// `Llama4VisionRotaryEmbedding.__init__` wraps its computed `freqs_ci` in
    /// an `nn.Buffer`, and `nn/parameter.py` opens with
    /// `data.detach().requires_grad_(...)`, so construction reaches `detach`
    /// before anything else can look at the tensor.
    ///
    /// A copy of both halves, which is what this shim's dense `detach`/`alias`
    /// already do -- they copy rather than alias (docs/kernels/OPS4.md §8) -- so the
    /// complex arm is not losing an aliasing property the real one had. The
    /// autograd flags are dropped exactly as `detach` drops them.
    pub fn passthrough(py: Python<'_>, input: &PyTensorBase) -> PyResult<Py<PyAny>> {
        let (re, im) = input.complex_parts("complex pass-through")?;
        wrap(py, PyTensorBase::complex(re.clone(), im.clone())?)
    }

    /// `z * s` for a real Python scalar -- `llama4_text`'s
    /// `freqs_cis * self.attention_scaling`.
    ///
    /// Scaling a complex number by a real one scales both components, so this
    /// is the one complex op with no cross terms. A *complex* scalar is not
    /// accepted: `PyComplex` never reaches the dispatcher here (there is no
    /// complex `Scalar` in this shim's argument forms), and inventing one
    /// would be a second entrance to the representation that
    /// `PyTensorBase::complex` is supposed to be the only one of.
    pub fn mul_scalar(py: Python<'_>, input: &PyTensorBase, value: f64) -> PyResult<Py<PyAny>> {
        const OP: &str = "aten.mul.Scalar";
        let (re, im) = input.complex_parts(OP)?;
        let real = (re * value)
            .and_then(|v| v.contiguous())
            .map_err(|e| candle_err(OP, e))?;
        let imag = (im * value)
            .and_then(|v| v.contiguous())
            .map_err(|e| candle_err(OP, e))?;
        wrap(py, PyTensorBase::complex(real, imag)?)
    }

    /// The replacement value for `dst.copy_(src)` when either side is complex.
    ///
    /// `copy_` in this shim already *replaces* the receiver's representation
    /// rather than writing through its buffer (`write_back`), so the complex
    /// path is the same shape as the dense one and not a new mechanism.
    ///
    /// **Both sides must be complex.** Upstream's `copy_` will cast a real
    /// source into a complex destination (imaginary part zero) and refuses the
    /// other direction; neither is implemented here, because a real->complex
    /// cast is a *constructor* for the representation and there is exactly one
    /// of those by design. Refusing names which side was which, so a caller
    /// can see that the gap is the cast and not `copy_`.
    pub fn copy_replacement(dst: &PyTensorBase, src: &PyTensorBase) -> PyResult<PyTensorBase> {
        const OP: &str = "aten.copy_.default";
        match (dst.repr(), src.repr()) {
            (Repr::Complex { re: dre, .. }, Repr::Complex { re, im }) => {
                let shape = dre.shape().clone();
                let real = re
                    .broadcast_as(shape.clone())
                    .and_then(|t| t.contiguous())
                    .map_err(|e| candle_err(OP, e))?;
                let imag = im
                    .broadcast_as(shape)
                    .and_then(|t| t.contiguous())
                    .map_err(|e| candle_err(OP, e))?;
                PyTensorBase::complex(real, imag)
            }
            (Repr::Complex { .. }, _) => Err(not_implemented(format!(
                "aten.copy_.default: copying a {} tensor into a {} one would                  have to invent an imaginary part. torch._C shim builds                  complex tensors only through torch.view_as_complex and                  torch.polar (torch._C._complex_ops()).",
                src.tag.name(),
                dst.tag.name()
            ))),
            _ => Err(not_implemented(format!(
                "aten.copy_.default: copying a {} tensor into a {} one would                  drop the imaginary part. Use torch.view_as_real(src) to say                  what should be copied.",
                src.tag.name(),
                dst.tag.name()
            ))),
        }
    }

    /// `z.unsqueeze(dim)` -- and, through `__getitem__`, `z[:, :, None, :]`.
    ///
    /// **The last op `llama4_text`'s rope needs, and the only shape op here.**
    /// `apply_rotary_emb` writes `xq_ * freqs_cis[:, :, None, :]`, and
    /// `bootstrap.py`'s `__getitem__` turns a `None` index into exactly one
    /// `aten.unsqueeze.default`; the three full slices are skipped without
    /// dispatching anything.
    ///
    /// `slice`, `select` and `index` are deliberately **not** here even though
    /// the same one-line "do it to both halves" would work for each. Every op
    /// added to this module is surface that has to be compared against
    /// upstream, and no measured caller reaches them on a complex tensor; they
    /// refuse at `tensor()` until one does. docs/kernels/COMPLEX2.md §5.
    pub fn unsqueeze(py: Python<'_>, input: &PyTensorBase, dim: i64) -> PyResult<Py<PyAny>> {
        const OP: &str = "aten.unsqueeze.default";
        let (re, im) = input.complex_parts(OP)?;
        let rank = re.dims().len() as i64;
        // Upstream's range for `unsqueeze` is [-rank-1, rank], one wider than
        // for the other shape ops because the new axis may go after the last.
        let normalised = if dim < 0 { dim + rank + 1 } else { dim };
        if normalised < 0 || normalised > rank {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Dimension out of range (expected to be in range of [{}, {}],                  but got {dim})",
                -rank - 1,
                rank
            )));
        }
        let at = normalised as usize;
        let real = re.unsqueeze(at).map_err(|e| candle_err(OP, e))?;
        let imag = im.unsqueeze(at).map_err(|e| candle_err(OP, e))?;
        wrap(py, PyTensorBase::complex(real, imag)?)
    }

    fn wrap(py: Python<'_>, t: PyTensorBase) -> PyResult<Py<PyAny>> {
        Ok(t.into_pyobject(py)?.into_any().unbind())
    }

    /// Wrap a real result. `PyTensorBase::new` derives the tag from what
    /// candle is storing, so `tag` is not passed through -- it is *checked*
    /// against what came back, which is the only way this can catch a caller
    /// that computed `view_as_real` of a `complex64` and got `float64` halves.
    fn finish_real(py: Python<'_>, t: Tensor, tag: TorchDType) -> PyResult<Py<PyAny>> {
        let wrapped = PyTensorBase::new(t)?;
        if wrapped.tag != tag {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "torch._C shim internal error -- a complex op produced a \
                 torch.{} result where its tag says torch.{} \
                 (tensor.rs::complex_ops)",
                wrapped.tag.name(),
                tag.name()
            )));
        }
        Ok(wrapped.into_pyobject(py)?.into_any().unbind())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensorBase>()?;
    m.add_function(wrap_pyfunction!(set_tensor_class, m)?)?;
    m.add_function(wrap_pyfunction!(set_size_class, m)?)?;
    m.add_function(wrap_pyfunction!(has_storage, m)?)?;
    m.add_function(wrap_pyfunction!(set_grad_enabled_flag, m)?)?;
    m.add_function(wrap_pyfunction!(set_throw_on_mutable_data_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(throws_on_mutable_data_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(set_warn_deprecated_on_mutable_data_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(warns_on_mutable_data_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(grad_enabled_flag, m)?)?;
    m.add_function(wrap_pyfunction!(complex_ops::complex_ops_list, m)?)?;
    Ok(())
}

#[cfg(test)]
mod transpose_tests {
    use super::*;

    fn cpu() -> candle_core::Device {
        candle_core::Device::Cpu
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    /// **The claim this kernel makes is bit-for-bit identity with the spelling
    /// it replaces**, and it is checked against exactly that spelling --
    /// candle's own `contiguous()` -- rather than against a recomputation.
    ///
    /// Sizes straddle the 32-element block in every direction on both axes, so
    /// a remainder bug on either edge, a block that overruns and a transposed
    /// index each have a shape that shows them. `(512, 64)` is the real
    /// SDPA shape at `S=512`.
    #[test]
    fn a_blocked_transpose_is_bit_identical_to_candles_contiguous() {
        for (b, h, s, d) in [
            (1usize, 1usize, 1usize, 1usize),
            (1, 1, 3, 5),
            (1, 1, 32, 32),
            (1, 1, 33, 31),
            (1, 1, 31, 33),
            (1, 1, 64, 96),
            (2, 3, 7, 4),
            (1, 9, 512, 64),
            (2, 2, 129, 65),
        ] {
            let n = b * h * s * d;
            let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect();
            let base = Tensor::from_vec(data, (b, h, s, d), &cpu()).unwrap();
            let view = base.transpose(2, 3).unwrap();
            assert!(!view.is_contiguous() || s == 1 || d == 1);
            let want = view.contiguous().unwrap();
            let got = transposed_contiguous(&view).unwrap();
            assert_eq!(got.dims(), want.dims(), "shape at {b}x{h}x{s}x{d}");
            assert_eq!(bits(&got), bits(&want), "values at {b}x{h}x{s}x{d}");
        }
    }

    /// Rank 2 and rank 3, so the batch loop is exercised at zero and one
    /// leading dimension as well as at two.
    #[test]
    fn every_rank_from_two_upwards_agrees_with_candle() {
        for dims in [vec![5usize, 7], vec![4, 33, 31], vec![2, 3, 8, 9], vec![2, 2, 2, 5, 6]] {
            let n: usize = dims.iter().product();
            let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let base = Tensor::from_vec(data, dims.clone(), &cpu()).unwrap();
            let last = dims.len() - 1;
            let view = base.transpose(last - 1, last).unwrap();
            assert_eq!(
                bits(&transposed_contiguous(&view).unwrap()),
                bits(&view.contiguous().unwrap()),
                "rank {} dims {dims:?}",
                dims.len()
            );
        }
    }

    /// Negative zero, the infinities and NaN survive the copy with their exact
    /// bit patterns. A copy has no excuse to change any of them, and `==`
    /// cannot see the first or the last -- so this compares `to_bits()`.
    #[test]
    fn the_special_values_survive_bit_for_bit() {
        let data = vec![
            -0.0f32,
            0.0,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::MIN_POSITIVE / 3.0, // subnormal
            1.5,
        ];
        let base = Tensor::from_vec(data, (2, 4), &cpu()).unwrap();
        let view = base.transpose(0, 1).unwrap();
        assert_eq!(
            bits(&transposed_contiguous(&view).unwrap()),
            bits(&view.contiguous().unwrap())
        );
    }

    /// A layout the recogniser does not understand must fall through to
    /// candle rather than being read in storage order -- the failure mode that
    /// would be silent, since it produces a tensor of the right shape.
    #[test]
    fn an_unrecognised_layout_falls_through_to_candle() {
        let base = Tensor::from_vec(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            (2, 3, 4),
            &cpu(),
        )
        .unwrap();
        // Swapping the *first* two dimensions is not the pattern.
        let view = base.transpose(0, 1).unwrap();
        assert!(transposed_plan(view.layout()).is_none());
        assert_eq!(
            bits(&transposed_contiguous(&view).unwrap()),
            bits(&view.contiguous().unwrap())
        );
        // A slice of a transposed view: the pair is right but the batches are
        // no longer consecutive.
        let sliced = base.transpose(1, 2).unwrap().narrow(0, 0, 1).unwrap();
        let _ = transposed_plan(sliced.layout());
        assert_eq!(
            bits(&transposed_contiguous(&sliced).unwrap()),
            bits(&sliced.contiguous().unwrap())
        );
        // An already-contiguous tensor is returned by candle's own path.
        assert_eq!(
            bits(&transposed_contiguous(&base).unwrap()),
            bits(&base.contiguous().unwrap())
        );
    }
}

#[cfg(test)]
mod amax_tests {
    use super::*;

    fn cpu() -> candle_core::Device {
        candle_core::Device::Cpu
    }

    /// The row reduction against a plain sequential fold, over lengths that
    /// straddle the lane count in every direction -- so a remainder bug, a
    /// seeding bug and a lane-combining bug each have a length that shows them.
    #[test]
    fn amax_matches_a_sequential_fold_at_every_length() {
        // A cheap deterministic sequence with negatives, duplicates and a run
        // of equal maxima.
        let make = |n: usize, seed: u64| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((s >> 33) as i64 as f32) / 1.0e6 - 4096.0
                })
                .collect()
        };
        for n in 1..=(AMAX_LANES * 4 + 3) {
            for seed in [1u64, 7, 99] {
                let row = make(n, seed);
                let mut want = row[0];
                for &v in &row[1..] {
                    if v > want {
                        want = v;
                    }
                }
                let got = amax_row(&row);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "n={n} seed={seed}: amax_row {got} vs sequential {want}"
                );
            }
        }
        // **The maximum walked through every position of every length.** The
        // random rows above leave it wherever it happens to fall, and that was
        // measured to be insufficient: a deliberate fault dropping exactly one
        // accumulator lane from the combining step survived all three seeds at
        // all 67 lengths, because none of them put the maximum in that lane.
        // This does not depend on luck -- if a lane is skipped, the length and
        // offset that land the maximum there fail here.
        for n in 1..=(AMAX_LANES * 4 + 3) {
            for at in 0..n {
                let mut row: Vec<f32> = (0..n).map(|i| -(i as f32) - 1.0).collect();
                row[at] = 4242.0;
                assert_eq!(
                    amax_row(&row),
                    4242.0,
                    "n={n}: the maximum at position {at} was not found"
                );
            }
        }
        // The same walk through `amax_strided`, which the contiguous rows above
        // never enter -- it is the branch that runs when the reduced dimension
        // is not the innermost one, and it has its own remainder and combine.
        for n in 1..=(AMAX_LANES * 2 + 5) {
            for stride in [1usize, 3, 7] {
                for at in 0..n {
                    let mut src: Vec<f32> = (0..n * stride + 3).map(|i| -(i as f32) - 1.0).collect();
                    src[2 + at * stride] = 4242.0;
                    assert_eq!(
                        amax_strided(&src, 2, n, stride),
                        4242.0,
                        "n={n} stride={stride}: the maximum at position {at} was not found"
                    );
                }
            }
        }
    }

    /// NaN propagates from **any** position, which is the half of the rule
    /// candle gets wrong (docs/models/E2E_REAL.md: `max([3, nan, 1])` is `3.0` there)
    /// and the half a one-comparison `!(v <= acc)` would get wrong is the
    /// other one -- a NaN early in a long row surviving to the end.
    #[test]
    fn nan_propagates_from_every_position() {
        for n in [1usize, 2, 5, AMAX_LANES, AMAX_LANES + 1, AMAX_LANES * 3 + 7] {
            for at in 0..n {
                let mut row: Vec<f32> = (0..n).map(|i| i as f32).collect();
                row[at] = f32::NAN;
                assert!(
                    amax_row(&row).is_nan(),
                    "n={n}: a NaN at position {at} did not reach the result"
                );
            }
        }
    }

    /// An all-`-inf` row -- a fully masked attention row -- answers `-inf`,
    /// not NaN and not the seed of an empty fold.
    #[test]
    fn an_all_negative_infinity_row_is_negative_infinity() {
        for n in [1usize, AMAX_LANES - 1, AMAX_LANES, AMAX_LANES * 2 + 5] {
            let row = vec![f32::NEG_INFINITY; n];
            assert_eq!(amax_row(&row), f32::NEG_INFINITY, "n={n}");
        }
    }

    /// The tensor-level entry, against candle's own `max_keepdim`, on the two
    /// layouts `amax_reduce` branches on -- innermost dimension (contiguous
    /// rows) and an outer one (strided slices) -- plus a non-contiguous input,
    /// which `amax_keepdim` is supposed to make contiguous before the kernel
    /// ever sees it.
    #[test]
    fn amax_keepdim_agrees_with_candle_where_candle_is_right() {
        let n = 3 * 5 * 7;
        let values: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32) - 50.0).collect();
        let t = Tensor::from_vec(values, (3, 5, 7), &cpu()).unwrap();
        for dim in 0..3 {
            let want = t.max_keepdim(dim).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let got = amax_keepdim(&t, dim).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(got, want, "dim={dim}");
            assert_eq!(
                amax_keepdim(&t, dim).unwrap().dims(),
                t.max_keepdim(dim).unwrap().dims(),
                "dim={dim} shape"
            );
        }
        // Non-contiguous: a transpose, reduced along each dimension.
        let tr = t.transpose(0, 2).unwrap();
        assert!(!tr.is_contiguous());
        for dim in 0..3 {
            let want = tr.max_keepdim(dim).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let got = amax_keepdim(&tr, dim).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(got, want, "transposed dim={dim}");
        }
    }

    /// The divergence from candle, stated as a test rather than left in a
    /// comment: on a tensor containing a NaN that is not first, candle's
    /// `max_keepdim` answers a number and this answers NaN. If candle ever
    /// fixes its reduction this fails, which is the notification wanted.
    #[test]
    fn candle_drops_the_nan_this_kernel_keeps() {
        let t = Tensor::from_vec(vec![3.0f32, f32::NAN, 1.0], (1, 3), &cpu()).unwrap();
        let candle = t.max_keepdim(1).unwrap().to_vec2::<f32>().unwrap()[0][0];
        let ours = amax_keepdim(&t, 1).unwrap().to_vec2::<f32>().unwrap()[0][0];
        assert_eq!(candle, 3.0, "candle's ReduceIndex no longer skips NaN");
        assert!(ours.is_nan(), "amax must propagate NaN, upstream's rule");
    }
}

#[cfg(test)]
mod scale_causal_tests {
    use super::*;
    use crate::reduced::FastDType;

    fn cpu() -> candle_core::Device {
        candle_core::Device::Cpu
    }

    /// The two-op path this replaces, transcribed from the `sdpa_flash_cpu`
    /// that was there before -- `affine(scale, 0.0)` and then a
    /// `broadcast_add` of an `f64` upper-triangular mask narrowed to the
    /// tensor's dtype.
    ///
    /// It is written out here rather than referenced so that the test keeps
    /// comparing against the *old* arithmetic even after the call site is
    /// gone. If the two ever have to disagree, this is the thing that says so.
    fn two_op_reference(t: &Tensor, scale: f64) -> Tensor {
        let dims = t.dims();
        let (rows, cols) = (dims[dims.len() - 2], dims[dims.len() - 1]);
        let scaled = t.affine(scale, 0.0).unwrap();
        let mut mask = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                mask.push(if c <= r { 0.0f64 } else { f64::NEG_INFINITY });
            }
        }
        let mask = Tensor::from_vec(mask, (rows, cols), t.device())
            .unwrap()
            .fast_to(t.dtype())
            .unwrap();
        scaled.broadcast_add(&mask).unwrap()
    }

    fn bits32(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    /// The values that make the three easy mistakes visible.
    ///
    /// A kernel written the obvious way -- `= -inf` above the diagonal,
    /// `v * mul` with the `+ 0.0` dropped, the scale multiplied in `f64` --
    /// agrees with the reference on every ordinary finite number. It is only
    /// these that separate them, so a sweep of well-behaved values would be a
    /// test that cannot fail. Each entry is annotated with which mistake it
    /// catches.
    fn awkward() -> Vec<f32> {
        vec![
            0.0,                   // -- ordinary
            -0.0,                  // `+ 0.0` dropped: product stays -0.0
            -1.0,                  // `+ 0.0` dropped, via a negative product
            1.0,
            -2.5,
            3.75,
            f32::INFINITY,         // `= -inf` instead of `+ -inf`: NaN vs -inf
            f32::NEG_INFINITY,
            f32::NAN,              // must survive as a NaN on both sides
            f32::MIN_POSITIVE,     // scale narrowing: underflow to zero
            f32::MIN_POSITIVE / 3.0, // subnormal
            -f32::MIN_POSITIVE / 3.0,
            f32::MAX,              // scale narrowing: overflow to inf
            -f32::MAX,
            1.0 + f32::EPSILON,    // needs a real rounding
            16_777_217.0,          // 2^24 + 1, not representable
        ]
    }

    /// Bit-for-bit against the two-op path, over shapes that straddle the
    /// diagonal in both directions and a batch dimension, with the awkward
    /// values tiled so that each one lands both inside and outside the mask.
    #[test]
    fn scaling_and_masking_in_one_pass_matches_the_two_op_path_bit_for_bit() {
        let vals = awkward();
        // Scales that are exact, that are not, and that push the extremes over
        // the edge in both directions.
        for &scale in &[1.0f64, 0.125, 0.1, 3.0, 1e30, 1e-30, -0.125] {
            for &(rows, cols) in &[
                (1usize, 1usize),
                (1, 5),
                (5, 1),
                (2, 2),
                (3, 4),
                (4, 3),
                (7, 7),
                (17, 16),
                (16, 17),
                (33, 33),
            ] {
                for &batch in &[1usize, 3] {
                    let n = batch * rows * cols;
                    // Rotate the offset with the row length so a value that is
                    // masked in one shape is kept in another.
                    let data: Vec<f32> =
                        (0..n).map(|i| vals[(i * 7 + rows) % vals.len()]).collect();
                    let t = Tensor::from_vec(data, (batch, rows, cols), &cpu()).unwrap();
                    let want = two_op_reference(&t, scale);
                    let got = scale_and_causal_mask(&t, scale).unwrap();
                    assert_eq!(
                        got.dims(),
                        want.dims(),
                        "shape, batch={batch} {rows}x{cols}"
                    );
                    assert_eq!(
                        bits32(&got),
                        bits32(&want),
                        "scale={scale} batch={batch} {rows}x{cols}"
                    );
                }
            }
        }
    }

    /// The rank-4 shape SDPA actually produces, and the only one the call site
    /// ever hands over: `[batch, head, S, S]`.
    #[test]
    fn the_rank_four_score_shape_sdpa_produces_matches_too() {
        let vals = awkward();
        for s in [1usize, 2, 8, 33] {
            let n = 2 * 3 * s * s;
            let data: Vec<f32> = (0..n).map(|i| vals[(i * 11 + s) % vals.len()]).collect();
            let t = Tensor::from_vec(data, (2, 3, s, s), &cpu()).unwrap();
            let want = two_op_reference(&t, 0.125);
            let got = scale_and_causal_mask(&t, 0.125).unwrap();
            assert_eq!(got.dims(), &[2, 3, s, s]);
            assert_eq!(bits32(&got), bits32(&want), "S={s}");
        }
    }

    /// `float64`, which `sdpa_flash_cpu` accumulates in when the caller asked
    /// for it. Same claim, different scalar -- and the `f64` arm is a separate
    /// instantiation, so it needs its own check rather than inheriting one.
    #[test]
    fn the_float64_arm_matches_the_two_op_path_too() {
        let vals: Vec<f64> = vec![
            0.0,
            -0.0,
            -1.0,
            2.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::MIN_POSITIVE,
            f64::MAX,
            1.0 + f64::EPSILON,
        ];
        for &scale in &[1.0f64, 0.125, 0.1, 1e300] {
            for &(rows, cols) in &[(1usize, 1usize), (5, 5), (17, 16), (16, 17)] {
                let n = 2 * rows * cols;
                let data: Vec<f64> = (0..n).map(|i| vals[(i * 3 + cols) % vals.len()]).collect();
                let t = Tensor::from_vec(data, (2, rows, cols), &cpu()).unwrap();
                let want = two_op_reference(&t, scale)
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f64>()
                    .unwrap();
                let got = scale_and_causal_mask(&t, scale)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f64>()
                    .unwrap();
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&got), bits(&want), "f64 scale={scale} {rows}x{cols}");
            }
        }
    }

    /// The diagonal itself, isolated: exactly the elements with `c <= r`
    /// survive. An off-by-one in `keep` moves one element per row, which the
    /// bit comparison above would also catch -- this says *which* element in a
    /// failure message, and it fails for a shape with no awkward values in it
    /// at all.
    #[test]
    fn the_mask_keeps_the_diagonal_and_nothing_to_its_right() {
        let s = 6usize;
        let data: Vec<f32> = (0..s * s).map(|i| (i + 1) as f32).collect();
        let got = scale_and_causal_mask(
            &Tensor::from_vec(data.clone(), (s, s), &cpu()).unwrap(),
            2.0,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
        for r in 0..s {
            for c in 0..s {
                let v = got[r * s + c];
                if c <= r {
                    assert_eq!(
                        v,
                        data[r * s + c] * 2.0,
                        "row {r} column {c} should have survived"
                    );
                } else {
                    assert_eq!(
                        v,
                        f32::NEG_INFINITY,
                        "row {r} column {c} should have been masked"
                    );
                }
            }
        }
    }

    /// A view is refused rather than silently read in storage order. Nothing
    /// in SDPA can hand one over -- `matmul` returns a fresh contiguous tensor
    /// -- so this pins the guard, not a behaviour anyone depends on.
    #[test]
    fn a_non_contiguous_score_matrix_is_refused() {
        let t = Tensor::from_vec((0..24).map(|i| i as f32).collect::<Vec<_>>(), (2, 3, 4), &cpu())
            .unwrap();
        // `scale_and_causal_mask` makes it contiguous first, so the refusal has
        // to be provoked at the kernel itself.
        let tr = t.transpose(0, 2).unwrap();
        assert!(!tr.is_contiguous());
        let err = tr.apply_op1_no_bwd(&ScaleCausal { scale: 1.0 });
        assert!(err.is_err(), "a transposed score matrix must be refused");
        // ...and the public entry point copes with exactly that tensor.
        assert_eq!(
            bits32(&scale_and_causal_mask(&tr, 0.5).unwrap()),
            bits32(&two_op_reference(&tr.contiguous().unwrap(), 0.5))
        );
    }

    /// Rank 1 has no score matrix in it and is told so by name.
    #[test]
    fn a_rank_one_tensor_is_refused_by_name() {
        let t = Tensor::from_vec(vec![1.0f32, 2.0], 2, &cpu()).unwrap();
        let err = scale_and_causal_mask(&t, 1.0).unwrap_err().to_string();
        assert!(err.contains("rank-2"), "unhelpful message: {err}");
    }
}
