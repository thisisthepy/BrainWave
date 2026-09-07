//! `torch._C.StorageBase` -- a flat, untyped byte buffer.
//!
//! The vendored tree's `torch/storage.py:467` opens with
//! `class UntypedStorage(torch._C.StorageBase, _StorageBase)`, so this is the
//! exact name the Python layer subclasses, the same relationship `TensorBase`
//! has with `torch.Tensor`. It is in Rust rather than in `bootstrap.py` for the
//! reason that file's docstring gives: it holds state and enforces an
//! invariant, and bootstrap.py holds neither.
//!
//! # What a storage is here, and what it is not
//!
//! Upstream a storage is the *owner* of a tensor's memory, and a tensor is a
//! view onto it -- `set_` makes the tensor alias the storage, so writing to the
//! storage afterwards changes the tensor. candle owns its own storage and has
//! no way to express that aliasing, so here a storage is a byte buffer that
//! `TensorBase.set_` **copies out of**.
//!
//! That difference is not cosmetic, and docs/CKPT.md §4 has the measurement:
//! `torch.load`'s two container formats fill the storage at different moments
//! relative to `set_`.
//!
//! ```text
//!   zip    (default)  fill storage from the record  ->  _rebuild_tensor -> set_
//!   legacy (v0)       _rebuild_tensor -> set_       ->  fill storage from file
//! ```
//!
//! Under aliasing both orders give the same answer, which is why upstream can
//! use one code path for both. Under copying the second order gives a tensor
//! full of whatever the storage held at `set_` time -- **zeros**. Measured: a
//! copying `set_` loads a legacy checkpoint to a complete state dict in which
//! every single weight is `0.0`, with no error raised anywhere.
//!
//! Silently wrong weights are the worst failure this shim can have, so the
//! ordering is not left to be respected by convention. `filled` records whether
//! any bytes were ever written in, and `set_` refuses on a storage that was
//! never filled. That makes the legacy order impossible to take by accident:
//! anyone who later implements `_set_from_file` gets a refusal naming this
//! invariant instead of a checkpoint of zeros.
//!
//! Note the shape of that invariant, because `from_file` below is the second
//! thing that may set it: `filled` is not "one function may set this", it is
//! **"only something that actually delivered bytes may set this"**. `_shim_fill`
//! takes a buffer, `from_file` reads a file; a plain allocation sets nothing.
//!
//! # `from_file` and why a copy is the right answer
//!
//! `torch.load(mmap=True)` and safetensors' default backend both arrive at
//! `UntypedStorage.from_file(path, shared, nbytes)`, and both pass
//! `shared=False`, because `torch.serialization.get_default_mmap_options()` is
//! `mmap.MAP_PRIVATE`. A private mapping is copy-on-write: measured on upstream
//! 2.13.0, writing through one does not change the file, and a second mapping
//! of the same file does not see the write. Its observable contents are exactly
//! the file's bytes. So reading those bytes into a buffer is not an
//! approximation of `shared=False` -- it is the same object, differing only in
//! residency (eager and whole, rather than lazy and per page) and in
//! `_get_filename()`, which upstream itself answers `None` for `shared=False`.
//!
//! `shared=True` is a different request and is refused by name. It means
//! `MAP_SHARED`: writes must reach the file and other processes. A copy cannot
//! do that, and quietly handing one back would be the same class of failure as
//! the zeros above -- an answer that looks right until someone writes.
//!
//! Slices are real views. `torch/serialization.py:2115` cuts one storage per
//! tensor out of the whole-file storage, so making each slice a copy would
//! double the checkpoint in memory and, worse, would make `data_ptr()` --
//! which upstream's loader uses to tell storages apart -- unrelated to where
//! the bytes actually live. The buffer is behind an `Arc` and a view holds an
//! offset into it, which is what a mapping's slice is.
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule, PySlice, PyType};

use crate::device::PyDevice;
use crate::err::not_implemented;

#[pyclass(name = "StorageBase", module = "torch._C", subclass)]
pub struct PyStorageBase {
    /// The backing bytes, shared with any views taken of this storage.
    buf: Arc<Vec<u8>>,
    /// Where this storage starts inside `buf`. Non-zero only for a view.
    off: usize,
    /// How many bytes of `buf` this storage is.
    len: usize,
    /// **Identity, for a storage that is a snapshot of somebody else's bytes.**
    /// Zero when this storage owns what it holds.
    ///
    /// `TensorBase.untyped_storage()` cannot lend candle's buffer out as a
    /// `StorageBase` -- candle owns it -- so it copies (`snapshot` below, and
    /// docs/SAVE.md §3). A copy alone would lose the one thing `torch.save`
    /// reads a storage's address *for*: `torch/serialization.py:1235` keys the
    /// record table on `storage._cdata`, so two tensors that are views of one
    /// buffer must answer with one value or the file loses that they were
    /// views. Two snapshots of the same candle storage carry the same value
    /// here, so `data_ptr()` and `_cdata` answer "are these the same storage?"
    /// -- which is the only question either path asks of them -- rather than
    /// "where does this particular copy live?", which nothing asks.
    origin: usize,
    /// Whether bytes were ever written in. See the module docstring -- this is
    /// the guard that makes the copy/alias difference loud instead of silent.
    /// Allocation does not set it; only delivering bytes does.
    filled: bool,
    /// `"cpu"` or `"meta"`. Everything else is refused at construction rather
    /// than answered with a CPU buffer.
    ///
    /// A **meta** storage is the one kind here that has a size and no bytes:
    /// `buf` is empty, `len` is what the tensor's bytes *would* occupy, and
    /// `filled` is false and stays false. It exists because
    /// `torch/_subclasses/meta_utils.py:2071` asks a meta tensor for its
    /// storage in order to key an aliasing memo, and a size-and-identity
    /// handle is exactly what that question wants -- see docs/EXPORT5.md §2 for
    /// which of upstream's expectations it meets and which it refuses by name.
    device: String,
}

impl PyStorageBase {
    pub fn bytes(&self) -> &[u8] {
        &self.buf[self.off..self.off + self.len]
    }

    pub fn is_filled(&self) -> bool {
        self.filled
    }

    /// The address this storage answers identity questions with. See `origin`.
    fn base_address(&self) -> usize {
        if self.origin != 0 {
            self.origin
        } else {
            self.buf.as_ptr() as usize
        }
    }

    fn is_meta(&self) -> bool {
        self.device == "meta"
    }

    /// The refusal a meta storage gives to anything that wants its bytes.
    ///
    /// Separate from `snapshot_is_read_only` because it is a different fact
    /// about a different object: a snapshot has bytes and may not be written
    /// through, a meta storage has **no bytes at all**. Sharing the message
    /// would tell a caller to "write to the tensor instead", and for a meta
    /// tensor there is nothing to write to.
    fn meta_has_no_bytes(&self, op: &str) -> PyErr {
        not_implemented(format!(
            "torch._C shim: {op} on a storage of a meta tensor. A meta storage \
             here carries a size ({} bytes) and an identity and no bytes at all, \
             which is what meta_utils.py's aliasing memo asks of it; it is not a \
             buffer of zeros standing in for one. Refused rather than answered \
             (docs/EXPORT5.md §2, storage.rs)",
            self.len
        ))
    }

    /// The refusal every write door gives. Two messages, because the two cases
    /// want different work done: a snapshot could only be made writable by
    /// giving candle a lendable storage, and a plain one has simply never had
    /// a caller.
    fn snapshot_is_read_only(&self, op: &str) -> PyErr {
        if self.origin != 0 {
            not_implemented(format!(
                "torch._C shim: {op} on a storage returned by \
                 TensorBase.untyped_storage(). That storage is a *snapshot* of \
                 the tensor's bytes, not a window onto them -- candle owns its \
                 buffer and cannot lend it out -- so a write here would be \
                 invisible to the tensor it came from. Refused rather than \
                 accepted silently: write to the tensor instead \
                 (docs/SAVE.md §3, storage.rs)"
            ))
        } else {
            not_implemented(format!(
                "torch._C shim: {op} -- this shim's storages are filled once, \
                 by the reader that delivers their bytes, and are read-only \
                 afterwards (storage.rs)"
            ))
        }
    }
}

/// The Python class a storage should wear -- `torch.UntypedStorage`.
///
/// The same relationship, and the same reason, as `tensor::TENSOR_CLASS`:
/// `torch/storage.py:836` refuses a bare `StorageBase` in
/// `TypedStorage(wrap_storage=...)` with an `isinstance` check, and
/// `torch/serialization.py`'s save path wraps every storage that way. Nothing
/// registers it in the standalone `_C` that `tools/golden/loader.py` imports,
/// and there `snapshot` hands back a bare `StorageBase` -- which is the honest
/// answer when no `torch` package exists to name a subclass.
static STORAGE_CLASS: std::sync::OnceLock<Py<PyAny>> = std::sync::OnceLock::new();

#[pyfunction]
#[pyo3(name = "_set_storage_class")]
pub fn set_storage_class(cls: Py<PyAny>) {
    let _ = STORAGE_CLASS.set(cls);
}

/// A storage holding `bytes`, identified as a copy of the buffer at `origin`.
///
/// `TensorBase.untyped_storage()`'s other half. `filled` is set because bytes
/// really did arrive -- the module docstring's invariant is "only something
/// that actually delivered bytes may set this", and a snapshot delivers them.
pub fn snapshot(py: Python<'_>, bytes: Vec<u8>, origin: usize) -> PyResult<Py<PyAny>> {
    let obj = match STORAGE_CLASS.get() {
        Some(cls) => cls.bind(py).call1((0usize,))?,
        None => Bound::new(
            py,
            PyStorageBase {
                buf: Arc::new(Vec::new()),
                off: 0,
                len: 0,
                origin: 0,
                filled: false,
                device: "cpu".to_string(),
            },
        )?
        .into_any(),
    };
    {
        let mut me = obj.cast::<PyStorageBase>()?.borrow_mut();
        me.len = bytes.len();
        me.off = 0;
        me.buf = Arc::new(bytes);
        me.origin = origin;
        me.filled = true;
    }
    Ok(obj.unbind())
}

/// The storage handle of a meta tensor: a size and an identity, and no bytes.
///
/// `TensorBase.untyped_storage()`'s meta half, and the answer to
/// `docs/EXPORT.md` §3.3. `nbytes` is what the tensor's elements *would*
/// occupy, `storage_id` is `Repr::Meta`'s token (see `tensor.rs`), and `buf`
/// stays empty -- there is nothing to put in it.
///
/// `filled` is **false**, and that is load-bearing rather than incidental: the
/// module docstring's invariant is that only something which actually
/// delivered bytes may set it, nothing ever delivers bytes here, and `set_`
/// refuses on an unfilled storage. So a meta storage cannot be laundered into
/// a real tensor's bytes by the one path that would produce silent zeros.
pub fn meta(py: Python<'_>, nbytes: usize, storage_id: usize) -> PyResult<Py<PyAny>> {
    let obj = match STORAGE_CLASS.get() {
        Some(cls) => cls.bind(py).call1((0usize,))?,
        None => Bound::new(
            py,
            PyStorageBase {
                buf: Arc::new(Vec::new()),
                off: 0,
                len: 0,
                origin: 0,
                filled: false,
                device: "cpu".to_string(),
            },
        )?
        .into_any(),
    };
    {
        let mut me = obj.cast::<PyStorageBase>()?.borrow_mut();
        me.buf = Arc::new(Vec::new());
        me.off = 0;
        me.len = nbytes;
        me.origin = storage_id;
        me.filled = false;
        me.device = "meta".to_string();
    }
    Ok(obj.unbind())
}

/// Build an instance of `cls` -- `torch.UntypedStorage`, normally -- holding a
/// view of `parent`'s bytes.
///
/// It goes through `cls(0)` rather than `Py::new` so the result is an instance
/// of the *Python* subclass: `torch/storage.py:836` checks
/// `isinstance(wrap_storage, torch.UntypedStorage)` and a bare `StorageBase`
/// would be rejected there. The Rust fields are then replaced, which is the
/// only way to hand a subclass instance a buffer it did not allocate.
fn view_of<'py>(
    cls: &Bound<'py, PyType>,
    parent: &PyStorageBase,
    off: usize,
    len: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let obj = cls.call1((0usize,))?;
    {
        let mut me = obj.cast::<PyStorageBase>()?.borrow_mut();
        me.buf = Arc::clone(&parent.buf);
        me.off = parent.off + off;
        me.len = len;
        // A view of bytes that arrived is bytes that arrived; a view of an
        // allocation is still an allocation. Inheriting rather than asserting
        // is what keeps `set_`'s guard meaningful one level down.
        me.filled = parent.filled;
        me.origin = parent.origin;
        me.device = parent.device.clone();
    }
    Ok(obj)
}

#[pymethods]
impl PyStorageBase {
    /// `UntypedStorage(nbytes)`. The other upstream spellings -- from a
    /// sequence, wrapping another storage -- are not reachable from the load
    /// path this exists for, and are refused by `_StorageBase`'s own stubs
    /// rather than guessed at here.
    #[new]
    #[pyo3(signature = (size = 0, *, device = None))]
    fn new(size: usize, device: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let device = match device {
            None => "cpu".to_string(),
            Some(d) => {
                let s = d.str()?.to_string();
                if s != "cpu" {
                    return Err(not_implemented(format!(
                        "torch._C shim: UntypedStorage(device={s:?}) -- storages here \
                         are CPU byte buffers only"
                    )));
                }
                s
            }
        };
        Ok(Self {
            buf: Arc::new(vec![0u8; size]),
            off: 0,
            len: size,
            origin: 0,
            filled: false,
            device,
        })
    }

    /// Write the payload in. Not an upstream name: upstream fills a storage
    /// through `_set_from_file` / `_write_file` / the C++ reader, none of which
    /// are implemented here.
    ///
    /// Refuses on a storage that shares its buffer -- that is, on a view, or on
    /// a storage some view was taken of. Upstream a write through either would
    /// be seen by the other; here it would not, and a fill that is invisible to
    /// half its aliases is the aliasing bug this module exists to make loud.
    fn _shim_fill(&mut self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.origin != 0 {
            return Err(self.snapshot_is_read_only("UntypedStorage._shim_fill"));
        }
        let view = pyo3::buffer::PyBuffer::<u8>::get(data)?;
        let bytes = view.to_vec(data.py())?;
        if bytes.len() != self.len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "torch._C shim: UntypedStorage._shim_fill got {} bytes for a \
                 storage of {} bytes",
                bytes.len(),
                self.len
            )));
        }
        let off = self.off;
        let len = self.len;
        let Some(buf) = Arc::get_mut(&mut self.buf) else {
            return Err(not_implemented(
                "torch._C shim: UntypedStorage._shim_fill on a storage whose \
                 bytes are shared with a view -- this shim's views alias, so \
                 the fill would be visible to some holders and not others. \
                 Fill the storage before slicing it (see storage.rs)",
            ));
        };
        buf[off..off + len].copy_from_slice(&bytes);
        self.filled = true;
        Ok(())
    }

    /// `torch.UntypedStorage.from_file(filename, shared=False, nbytes=0)`.
    ///
    /// The entry point `torch.load(mmap=True)` (`serialization.py:1594`) and
    /// safetensors' default `mmap` backend both reach. See the module docstring
    /// for why `shared=False` is answered with a read and `shared=True` is
    /// refused rather than approximated.
    ///
    /// The two errors are upstream's, reproduced with upstream's wording
    /// because a caller that catches on the message should not have to care
    /// which torch it is talking to.
    #[classmethod]
    #[pyo3(signature = (filename, shared = false, nbytes = 0))]
    fn from_file<'py>(
        cls: &Bound<'py, PyType>,
        filename: &str,
        shared: bool,
        nbytes: i64,
    ) -> PyResult<Bound<'py, PyAny>> {
        if shared {
            return Err(not_implemented(format!(
                "torch._C shim: UntypedStorage.from_file({filename:?}, shared=True) \
                 -- shared=True is MAP_SHARED, which requires writes through the \
                 storage to reach the file and other processes. This shim's \
                 storages are owned buffers and cannot do that; shared=False is \
                 MAP_PRIVATE and is supported, which is what \
                 torch.load(mmap=True) and safetensors both ask for"
            )));
        }
        if nbytes < 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "unable to mmap {nbytes} bytes from file <{filename}>: Invalid argument (22)"
            )));
        }
        let nbytes = nbytes as u64;
        let meta = std::fs::metadata(filename).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "unable to open file <{filename}> in read-only mode: {} ({})",
                io_reason(&e),
                e.raw_os_error().unwrap_or(0)
            ))
        })?;
        let size = meta.len();
        if nbytes > size {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "file <{filename}> size <{size}> is smaller than the required \
                 mapping size <{nbytes}>"
            )));
        }
        let want = nbytes as usize;
        let mut data = std::fs::read(filename).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "unable to open file <{filename}> in read-only mode: {} ({})",
                io_reason(&e),
                e.raw_os_error().unwrap_or(0)
            ))
        })?;
        // The file may have grown between the stat and the read; a mapping of
        // `nbytes` sees only the first `nbytes` either way.
        if data.len() < want {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "file <{filename}> size <{}> is smaller than the required \
                 mapping size <{nbytes}>",
                data.len()
            )));
        }
        data.truncate(want);
        let obj = cls.call1((0usize,))?;
        {
            let mut me = obj.cast::<PyStorageBase>()?.borrow_mut();
            me.len = data.len();
            me.off = 0;
            me.buf = Arc::new(data);
            me.origin = 0;
            // Bytes actually arrived, from the file the caller named. This is
            // the second thing in this module allowed to set `filled`, and it
            // qualifies for the same reason `_shim_fill` does.
            me.filled = true;
        }
        Ok(obj)
    }

    /// `storage[i]` and `storage[a:b]`.
    ///
    /// An integer index answers a byte, as upstream's untyped storage does. A
    /// slice answers a storage that *views* this one -- upstream's does too,
    /// and `torch/serialization.py:2115` relies on the offset arithmetic being
    /// real when it cuts one storage per tensor out of a whole-file mapping.
    ///
    /// A step other than 1 is upstream's own refusal, verbatim: a strided
    /// storage has no representation on either side.
    fn __getitem__<'py>(
        slf: &Bound<'py, Self>,
        idx: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let me = slf.borrow();
        let len = me.len;
        if me.is_meta() {
            // Upstream's own message, verbatim: reading or slicing a meta
            // storage raises `NotImplementedError: Not available for 'meta'
            // device type` on 2.13.0. Measured, not transcribed from prose --
            // a slice is refused there too, so this is before the slice arm
            // rather than inside the integer one.
            return Err(not_implemented("Not available for 'meta' device type"));
        }
        if let Ok(slice) = idx.cast::<PySlice>() {
            let step = slice.getattr("step")?;
            if !step.is_none() && step.extract::<i64>()? != 1 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Trying to slice with a step of {}, but only a step of 1 is supported",
                    step.extract::<i64>()?
                )));
            }
            let info = slice.indices(len as isize)?;
            let (start, stop) = (info.start.max(0) as usize, info.stop.max(0) as usize);
            let stop = stop.max(start);
            return view_of(&slf.get_type(), &me, start, stop - start);
        }
        let i = idx.extract::<i64>()?;
        let n = len as i64;
        let at = if i < 0 { i + n } else { i };
        if at < 0 || at >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {i} out of range for storage of size {len}"
            )));
        }
        let byte = me.buf[me.off + at as usize];
        Ok(byte.into_pyobject(slf.py())?.into_any())
    }

    /// Readable from Python so the invariant in the module docstring can be
    /// asserted by a test rather than argued about.
    #[getter]
    fn _shim_filled(&self) -> bool {
        self.filled
    }

    /// Whether this storage is a snapshot of a tensor's bytes. Readable for
    /// the same reason `_shim_filled` is.
    #[getter]
    fn _shim_is_snapshot(&self) -> bool {
        self.origin != 0
    }

    /// **The three write doors, all refusing, and this is what makes
    /// `untyped_storage()` honest.**
    ///
    /// docs/BACKWARD.md §14.3 sized `untyped_storage()` and stopped, on the
    /// grounds that a storage which is a copy would let a caller write through
    /// it and have the write land nowhere -- *"a lie on the public surface to
    /// satisfy the one caller that cannot detect it"*. That objection is right
    /// about the danger and wrong about the remedy: what the danger needs is
    /// not aliasing, it is a **refusal**. A storage that reads correctly and
    /// says so when written to is a narrowing that names itself, which is the
    /// same shape as `filled` on the load side; a `torch.save` that does not
    /// exist is not.
    ///
    /// All three exist upstream and all three are a bare `raise
    /// NotImplementedError` in the vendored `_StorageBase` (`torch/storage.py`
    /// lines 62, 65 and 173) -- the anonymous refusal DESIGN.md §6 forbids and
    /// which cannot be fixed where it lives. `UntypedStorage(torch._C.StorageBase,
    /// _StorageBase)` puts these first in the MRO.
    fn __setitem__(&self, _idx: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.is_meta() {
            return Err(self.meta_has_no_bytes("UntypedStorage.__setitem__"));
        }
        Err(self.snapshot_is_read_only("UntypedStorage.__setitem__"))
    }

    #[pyo3(signature = (*_args, **_kwargs))]
    fn copy_(&self, _args: &Bound<'_, PyAny>, _kwargs: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        if self.is_meta() {
            return Err(self.meta_has_no_bytes("UntypedStorage.copy_"));
        }
        Err(self.snapshot_is_read_only("UntypedStorage.copy_"))
    }

    #[pyo3(signature = (*_args, **_kwargs))]
    fn resize_(
        &self,
        _args: &Bound<'_, PyAny>,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        // Upstream's meta storage *does* resize (measured: 2.13.0 resizes a
        // meta storage to 8 bytes and reports it). This one refuses, and that
        // is a divergence rather than a gap: `len` here is derived from the
        // meta tensor's shape and dtype at the moment the handle was made, so
        // a resize would leave the storage and the tensor disagreeing about a
        // number the tensor is the authority on. docs/EXPORT5.md §2 lists it
        // among the expectations this handle refuses by name.
        if self.is_meta() {
            return Err(not_implemented(format!(
                "torch._C shim: UntypedStorage.resize_ on a storage of a meta \
                 tensor. Upstream resizes one; this handle's size ({} bytes) is \
                 derived from the meta tensor's shape and dtype and is not \
                 independently settable, so resizing would leave the storage and \
                 the tensor disagreeing (docs/EXPORT5.md §2)",
                self.len
            )));
        }
        Err(self.snapshot_is_read_only("UntypedStorage.resize_"))
    }

    fn nbytes(&self) -> usize {
        self.len
    }

    fn size(&self) -> usize {
        self.len
    }

    fn __len__(&self) -> usize {
        self.len
    }

    /// `UntypedStorage.filename` reads this (`torch/storage.py:484`). Upstream
    /// answers the path only for a storage made by `from_file(shared=True)`,
    /// and `None` for everything else -- including `from_file(shared=False)`,
    /// measured on 2.13.0. `shared=True` is refused here, so `None` is not a
    /// stub: it is the same answer upstream gives to every storage this build
    /// can construct.
    fn _get_filename(&self) -> Option<String> {
        None
    }

    /// `False`, and this is one of the places where the shim answers something
    /// upstream does not.
    ///
    /// Measured on 2.13.0: `from_file(p, False, n).is_shared()` is `True` even
    /// though the mapping is `MAP_PRIVATE` -- upstream is reporting "this came
    /// from a file mapping", not "writes are shared" -- while a slice of it and
    /// a plain `UntypedStorage(n)` are both `False`. This build's storage is an
    /// owned buffer with no file behind it, so `False` is what is true of it.
    /// Answering `True` to match the label would claim a relationship to a file
    /// that does not exist.
    ///
    /// Nothing on the load path reads it; it is implemented because
    /// `_StorageBase.is_shared` in the vendored tree is a bare
    /// `raise NotImplementedError` with no message, which is the anonymous
    /// refusal DESIGN.md §6 forbids and which cannot be fixed where it lives.
    fn is_shared(&self) -> bool {
        false
    }

    /// Upstream's untyped storage reports 1: it is a byte buffer, and the
    /// element size belongs to the *typed* storage wrapped around it.
    fn element_size(&self) -> usize {
        1
    }

    /// The address of this storage's first byte. Real, and stable for the life
    /// of the storage, because `torch/serialization.py` and `safetensors` both
    /// use it only to tell two storages apart -- never to read through it.
    ///
    /// A view's address is the parent's plus its offset, which is what makes
    /// two slices of one checkpoint distinguishable; that is the same relation
    /// upstream's mapping has, measured.
    ///
    /// For a *snapshot* -- what `TensorBase.untyped_storage()` returns -- it is
    /// the address of the candle buffer the bytes were copied from, not of the
    /// copy. See `origin`: the save path's two readers of this number
    /// (`torch/serialization.py:1224`'s dtype-conflict table and, one line
    /// later, `_cdata`'s record key) both ask it "same storage?", and a copy's
    /// own address would answer "no" for two views of one buffer.
    fn data_ptr(&self) -> usize {
        // **A meta storage answers 0, and that is upstream's answer, measured.**
        // On torch 2.13.0 every meta storage -- base or view, any size --
        // answers `data_ptr() == 0`, because there is no allocation to point
        // at. Answering `origin` here would hand out a number that looks like
        // an address and is a counter, which is the shape of lie this file
        // exists to refuse. Identity is asked for through `_cdata`, and that is
        // where it is answered.
        if self.is_meta() {
            return 0;
        }
        self.base_address() + self.off
    }

    /// `storage._cdata`. Upstream this is the address of the `THPStorage`
    /// object; `torch/serialization.py:1235` keys `id_map` on it, so it is what
    /// decides how many `data/N` records a save writes, and
    /// `torch/storage.py:239` uses it as a deepcopy memo key.
    ///
    /// Answered with the storage's identity rather than the Python object's,
    /// which is the difference that makes storage sharing survive a save here:
    /// `untyped_storage()` builds a *fresh* Python object on every call
    /// (upstream caches one per storage), so object identity would make every
    /// tensor look like it had a storage of its own.
    ///
    /// **For a meta storage this is the whole of its identity**, since
    /// `data_ptr()` is 0 there and carries none. It is `Repr::Meta`'s
    /// `storage_id` -- distinct per meta storage, shared between a meta tensor
    /// and its views -- which is the relation upstream's `_cdata` has,
    /// measured: `b.untyped_storage()._cdata == b[1:,1:].untyped_storage()._cdata`
    /// and differs from an unrelated meta tensor's.
    #[getter]
    fn _cdata(&self) -> usize {
        self.base_address()
    }

    /// The legacy (non-zip) `torch.save` format's writer, and it stays refused.
    ///
    /// `torch.save(obj, f, _use_new_zipfile_serialization=False)` reaches
    /// `_legacy_save`, which writes each storage with this. It is refused for
    /// the same reason the legacy *reader* is (docs/CKPT.md §4): that format
    /// puts `_rebuild_tensor`'s `set_` *before* the bytes arrive, and this
    /// shim's `set_` copies rather than aliases, so a checkpoint written in it
    /// would have to be read back through a path that produces zeros. Writing a
    /// format this build cannot read is a worse trade than refusing it.
    ///
    /// The refusal is here rather than left to `_StorageBase._write_file` in
    /// the vendored tree because that one is a bare `raise NotImplementedError`
    /// with no message -- the anonymous refusal DESIGN.md §6 forbids -- and
    /// `UntypedStorage(torch._C.StorageBase, _StorageBase)` puts this first in
    /// the MRO, so naming it is possible from here and only from here.
    #[pyo3(signature = (*_args, **_kwargs))]
    fn _write_file(
        &self,
        _args: &Bound<'_, PyAny>,
        _kwargs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        Err(not_implemented(
            "torch._C shim: UntypedStorage._write_file -- this is the legacy \
             (non-zip) torch.save container, reached by \
             _use_new_zipfile_serialization=False. It is refused, not missing: \
             the legacy format fills a storage *after* _rebuild_tensor has \
             called set_, and this shim's set_ copies instead of aliasing, so \
             anything written in that format reads back as zeros here \
             (docs/CKPT.md §4, docs/SAVE.md §6). Save with the default zip \
             container, which this build both writes and reads",
        ))
    }

    #[getter]
    fn device(&self) -> PyDevice {
        if self.is_meta() {
            return PyDevice::meta();
        }
        PyDevice::cpu()
    }

    #[getter]
    fn is_cuda(&self) -> bool {
        false
    }

    #[getter]
    fn is_hpu(&self) -> bool {
        false
    }

    #[getter]
    fn is_sparse(&self) -> bool {
        false
    }

    #[getter]
    fn is_sparse_csr(&self) -> bool {
        false
    }

    /// A storage is never pickled *as an object* by `torch.save`: the pickler
    /// intercepts it through `persistent_id` and writes a `data/N` record
    /// instead (`torch/serialization.py:1204`), so this is not on the save
    /// path. It is what somebody reaches who pickles a storage directly, and
    /// there is no honest reduction for one -- the bytes would have to be
    /// inlined into the pickle, which is the thing the container format exists
    /// to avoid.
    fn __getstate__(&self) -> PyResult<()> {
        Err(not_implemented(
            "torch._C shim: UntypedStorage does not pickle on its own. \
             torch.save does not need it to -- it writes storages as zip \
             records through persistent_id (docs/SAVE.md) -- so this is \
             reached only by pickling a storage directly",
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "<torch._C.StorageBase {} bytes on {}{}{}>",
            self.len,
            self.device,
            if self.off == 0 {
                String::new()
            } else {
                format!(", view at +{}", self.off)
            },
            if self.filled { "" } else { ", unfilled" }
        )
    }

    /// `bytes(storage)`, for tests and for anything that wants the payload back
    /// without going through a tensor.
    fn _shim_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        // A meta storage's `bytes()` is an empty slice, and handing that back
        // would say "this storage is empty" where the truth is "this storage
        // has `len` bytes and none of them exist". Those are different claims
        // and the second one has to be a refusal.
        if self.is_meta() {
            return Err(self.meta_has_no_bytes("UntypedStorage._shim_bytes"));
        }
        Ok(PyBytes::new(py, self.bytes()))
    }
}

/// The text upstream's `from_file` errors carry after "in read-only mode: ".
/// It is `strerror(errno)`, and Rust's `Display` for `io::Error` appends
/// " (os error N)" which upstream does not have, so the message is rebuilt
/// rather than forwarded.
fn io_reason(e: &std::io::Error) -> String {
    let s = e.to_string();
    match s.find(" (os error ") {
        Some(i) => s[..i].to_string(),
        None => s,
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStorageBase>()?;
    m.add_function(wrap_pyfunction!(set_storage_class, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// The `as_strided` write barrier. docs/STRIDED.md.
// ---------------------------------------------------------------------------
//
// `aten.as_strided` promises a **two-way view**: writes through the result are
// visible in the base, and writes to the base are visible through the result.
// candle 0.11.0 cannot produce that -- `Layout::new(shape, stride, offset)` is
// public and `Tensor::from_storage` is public, but the struct that joins them
// (`Tensor_ { storage: Arc<RwLock<Storage>>, layout }`) has no public field, so
// the nine internal sites that build a shallow view cannot be reached from
// here. docs/TAIL3.md §6 and `tools/golden/reach_allow.json` recorded that; it
// is re-verified in `test_strided.py`.
//
// So the result is a **gather**, and a gather is silently wrong for a writer in
// both directions at once. This registry is what makes that wrongness loud
// instead of silent: while an `as_strided` result is alive, the storage of the
// result *and* the storage of the base are barred from every in-place write.
// The single door those writes go through is `tensor::write_into`
// (`aten.rs::write_back` is its only caller), and that is where the barrier is
// read.
//
// **Why an address is a sound key here and is not one in general.** An address
// is reused after the allocation it named is freed, so a permanent poison set
// keyed on one starts refusing writes to unrelated later tensors -- which is
// exactly why docs/TAIL4.md §1.2 rejected this structure. What removes that
// objection is the keep-alive below: a `StridedBarrier` holds a clone of both
// candle tensors, so neither storage can be freed while its key is registered,
// so neither key can be reused while it means something. The barrier is
// unregistered in `Drop`, at which moment the keep-alive is released too --
// the entry and the reservation end in the same statement, which is the
// property that makes reuse unobservable rather than merely unlikely.
//
// Holding the base alive for as long as the view is **not** a leak relative to
// upstream: upstream's `as_strided` result holds the base's storage alive by
// aliasing it. This is the same lifetime, reached a different way.
//
// What this does not close is written down rather than hidden, in
// docs/STRIDED.md §4: an alias of the *result* that outlives the result.

/// `storage address -> how many live barriers name it`.
///
/// A count rather than a set because two `as_strided` calls on one base are
/// ordinary, and the first result to be dropped must not unbar the second.
static STRIDED_BARRIER: std::sync::Mutex<Option<std::collections::HashMap<usize, usize>>> =
    std::sync::Mutex::new(None);

/// The address of the candle `Storage` behind a tensor -- `capture.rs`'s
/// `storage_key` idiom, and the same reason for leaving `storage_offset` out of
/// it: every alias of a buffer must answer the same value.
pub(crate) fn storage_identity(tensor: &candle_core::Tensor) -> usize {
    let (guard, _layout) = tensor.storage_and_layout();
    let storage: &candle_core::Storage = &guard;
    storage as *const candle_core::Storage as usize
}

/// A live `as_strided` result, and the two storages it bars from being written.
///
/// The `keep_alive` field is load-bearing and is the whole of the soundness
/// argument above; it is never read, which is why it is named for what it does.
pub struct StridedBarrier {
    keys: Vec<usize>,
    #[allow(dead_code)]
    keep_alive: Vec<candle_core::Tensor>,
}

impl StridedBarrier {
    /// Bar `base`'s storage and `view`'s storage until the returned handle
    /// (and every clone of the tensor holding it) is dropped.
    pub fn new(base: &candle_core::Tensor, view: &candle_core::Tensor) -> Arc<Self> {
        let mut keys = vec![storage_identity(base), storage_identity(view)];
        keys.sort_unstable();
        keys.dedup();
        if let Ok(mut guard) = STRIDED_BARRIER.lock() {
            let map = guard.get_or_insert_with(std::collections::HashMap::new);
            for key in &keys {
                *map.entry(*key).or_insert(0) += 1;
            }
        }
        Arc::new(Self {
            keys,
            keep_alive: vec![base.clone(), view.clone()],
        })
    }
}

impl Drop for StridedBarrier {
    fn drop(&mut self) {
        if let Ok(mut guard) = STRIDED_BARRIER.lock() {
            if let Some(map) = guard.as_mut() {
                for key in &self.keys {
                    if let Some(count) = map.get_mut(key) {
                        *count -= 1;
                        if *count == 0 {
                            map.remove(key);
                        }
                    }
                }
            }
        }
        // `keep_alive` drops here, after the keys are gone. The order matters:
        // the reservation must not be released while the key still means
        // something, or a reallocation at the same address would inherit it.
    }
}

/// Is this tensor's storage barred from in-place writes?
///
/// Read by exactly one caller, `tensor::write_into`, which is the single write
/// door. A poisoned lock answers `false` -- refusing on a lock failure would
/// turn an unrelated panic into a wall of refusals -- and that is safe here
/// only because the lock is never held across anything that can panic.
pub(crate) fn write_is_barred(tensor: &candle_core::Tensor) -> bool {
    let key = storage_identity(tensor);
    match STRIDED_BARRIER.lock() {
        Ok(guard) => guard.as_ref().is_some_and(|map| map.contains_key(&key)),
        Err(_) => false,
    }
}
