//! `torch._C.dtype`.
//!
//! In real PyTorch `torch.float32` is not a Python-level constant -- it is an
//! instance of a type defined in `_C`, and the vendored Python tree re-exports
//! it (`torch/__init__.py`). So the shim has to own the type, not just a name.
//!
//! It also has to own the *set*. This module used to wrap `candle_core::DType`
//! directly, which meant the shim could only name a dtype candle could store:
//! ten of them. The vendored tree needs thirty-three. `torch/_prims_common`,
//! `torch/_tensor_str.py` and `torch/_refs` all build tables over
//! `torch.bool`, `torch.int8`, `torch.complex64` and the quantised types while
//! `import torch` is still running, so a shim with ten dtypes cannot finish the
//! import at all.
//!
//! BOOL.md settled the shape of the answer for `torch.bool` and it generalises
//! to the rest: **`_C` owns the dtype tag and candle owns the storage.** The
//! two are related by `storage()`, which is `None` for every dtype candle
//! cannot hold. That is the same decision `device` already embodies (a label,
//! resolved on use) and it has the same payoff -- the failure lands where torch
//! puts it, by name, instead of on a near neighbour.
//!
//! Specifically *not* done: aliasing `bool` onto `uint8`. BOOL.md §3 measured
//! what that costs -- `bool + bool` is a logical or while `uint8 + uint8` is an
//! arithmetic sum, `~bool` is a logical not while `~uint8` is a bit flip, and
//! `masked_fill` stops refusing the wrong mask dtype. Six of torch's own
//! guardrails are keyed on the tag; sharing one tag between the two turns all
//! six off.
use candle_core::DType;
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Every dtype the vendored tree names. Derived from the tree's own stub
/// (`torch/_C/__init__.pyi` declares each as a module-level `: dtype`), not
/// from an installed upstream binary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TorchDType {
    Float32,
    Float64,
    Float16,
    BFloat16,
    Complex32,
    Complex64,
    Complex128,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    QUInt4x2,
    QUInt2x4,
    Float8E4M3FN,
    Float8E4M3FNUZ,
    Float8E5M2,
    Float8E5M2FNUZ,
    Float8E8M0FNU,
    Float4E2M1FNX2,
    Bits1x8,
    Bits2x4,
    Bits4x2,
    Bits8,
    Bits16,
    // The sub-byte integer tags. These are *not* in `torch/_C/__init__.pyi`
    // (which is where the rest of this enum came from) -- upstream registers
    // them from C++ without a stub entry, so the generated surface never
    // named them and the shim did not either.
    //
    // They are here because `torch.load` demands them by name.
    // `torch/_weights_only_unpickler.py:194` builds its allowlist with
    //
    //     for t in [getattr(torch, f"uint{x}") for x in range(1, 8)]: ...
    //     for t in [getattr(torch, f"int{x}") for x in range(1, 8)]: ...
    //
    // unconditionally, before a single byte of the checkpoint is read. Missing
    // any one of the fourteen makes *every* `torch.load` raise
    // `AttributeError: module 'torch' has no attribute 'uint1'` -- measured,
    // and it was the second wall on that path (docs/CKPT.md §2).
    //
    // Nothing can be stored under them: `storage()` returns `None` for all
    // fourteen, so a tensor tagged with one refuses by name rather than
    // quietly becoming `uint8`. They are names on the allowlist, and that is
    // the whole of what they are.
    UInt1,
    UInt2,
    UInt3,
    UInt4,
    UInt5,
    UInt6,
    UInt7,
    Int1,
    Int2,
    Int3,
    Int4,
    Int5,
    Int6,
    Int7,
}

use TorchDType::*;

/// The canonical spellings, and the full set. `register` walks this.
pub const ALL: &[TorchDType] = &[
    Float32,
    Float64,
    Float16,
    BFloat16,
    Complex32,
    Complex64,
    Complex128,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    QUInt4x2,
    QUInt2x4,
    Float8E4M3FN,
    Float8E4M3FNUZ,
    Float8E5M2,
    Float8E5M2FNUZ,
    Float8E8M0FNU,
    Float4E2M1FNX2,
    Bits1x8,
    Bits2x4,
    Bits4x2,
    Bits8,
    Bits16,
    UInt1,
    UInt2,
    UInt3,
    UInt4,
    UInt5,
    UInt6,
    UInt7,
    Int1,
    Int2,
    Int3,
    Int4,
    Int5,
    Int6,
    Int7,
];

/// torch's short spellings. `torch.float is torch.float32` is `True` upstream,
/// so these name the same dtype rather than a distinct one.
pub const ALIASES: &[(&str, TorchDType)] = &[
    ("float", Float32),
    ("double", Float64),
    ("half", Float16),
    ("short", Int16),
    ("int", Int32),
    ("long", Int64),
    ("chalf", Complex32),
    ("cfloat", Complex64),
    ("cdouble", Complex128),
];

impl TorchDType {
    /// The name torch's *C++* `ScalarType` prints, which is not the Python one.
    ///
    /// `torch.uint8` is `Byte` in a C++ error message, `torch.float32` is
    /// `Float`, `torch.int64` is `Long`. This exists because `aten::view.dtype`
    /// raises from C++ and its message names both dtypes -- "must be divisible
    /// by 4 to view Byte as Float" -- and a shim that reproduced the sentence
    /// but not the vocabulary would hand callers a message that is upstream's
    /// everywhere except where it identifies the problem.
    ///
    /// Only the dtypes this build can hold are spelled out; the rest fall back
    /// to the Python name, which is what a caller who reaches them will
    /// already have been told by `storage()`'s refusal. All measured on 2.13.0.
    pub fn cpp_name(self) -> &'static str {
        match self {
            Float32 => "Float",
            Float64 => "Double",
            Float16 => "Half",
            BFloat16 => "BFloat16",
            UInt8 => "Byte",
            Int8 => "Char",
            Int16 => "Short",
            Int32 => "Int",
            Int64 => "Long",
            UInt32 => "UInt32",
            Bool => "Bool",
            Float8E4M3FN => "Float8_e4m3fn",
            other => other.name(),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Float32 => "float32",
            Float64 => "float64",
            Float16 => "float16",
            BFloat16 => "bfloat16",
            Complex32 => "complex32",
            Complex64 => "complex64",
            Complex128 => "complex128",
            UInt8 => "uint8",
            UInt16 => "uint16",
            UInt32 => "uint32",
            UInt64 => "uint64",
            Int8 => "int8",
            Int16 => "int16",
            Int32 => "int32",
            Int64 => "int64",
            Bool => "bool",
            QInt8 => "qint8",
            QUInt8 => "quint8",
            QInt32 => "qint32",
            QUInt4x2 => "quint4x2",
            QUInt2x4 => "quint2x4",
            Float8E4M3FN => "float8_e4m3fn",
            Float8E4M3FNUZ => "float8_e4m3fnuz",
            Float8E5M2 => "float8_e5m2",
            Float8E5M2FNUZ => "float8_e5m2fnuz",
            Float8E8M0FNU => "float8_e8m0fnu",
            Float4E2M1FNX2 => "float4_e2m1fn_x2",
            Bits1x8 => "bits1x8",
            Bits2x4 => "bits2x4",
            Bits4x2 => "bits4x2",
            Bits8 => "bits8",
            Bits16 => "bits16",
            UInt1 => "uint1",
            UInt2 => "uint2",
            UInt3 => "uint3",
            UInt4 => "uint4",
            UInt5 => "uint5",
            UInt6 => "uint6",
            UInt7 => "uint7",
            Int1 => "int1",
            Int2 => "int2",
            Int3 => "int3",
            Int4 => "int4",
            Int5 => "int5",
            Int6 => "int6",
            Int7 => "int7",
        }
    }

    /// How candle stores a tensor with this tag. `None` means the shim can
    /// name the dtype but cannot hold a tensor of it -- every op path that
    /// reaches such a dtype must refuse by name rather than substitute.
    ///
    /// `Bool` is the interesting entry: it stores as `U8` but is *not*
    /// `UInt8`. BOOL.md §6.3 states the invariant that buys -- the bytes under
    /// a `bool` tag are 0 or 1 -- and keeping it is the shim's job, which is
    /// why there is one tagging constructor rather than a free-for-all.
    pub fn storage(self) -> Option<DType> {
        Some(match self {
            Float32 => DType::F32,
            Float64 => DType::F64,
            Float16 => DType::F16,
            BFloat16 => DType::BF16,
            UInt8 => DType::U8,
            UInt32 => DType::U32,
            Int16 => DType::I16,
            Int32 => DType::I32,
            Int64 => DType::I64,
            Float8E4M3FN => DType::F8E4M3,
            Bool => DType::U8,
            _ => return None,
        })
    }

    /// The tag a tensor candle handed us carries. Deliberately *not* the
    /// inverse of `storage()`: `U8` maps back to `uint8`, never to `bool`,
    /// because a bool tensor's tag has to be carried alongside the storage
    /// rather than reconstructed from it (BOOL.md §5-B).
    pub fn from_storage(dtype: DType) -> Option<Self> {
        Some(match dtype {
            DType::F32 => Float32,
            DType::F64 => Float64,
            DType::F16 => Float16,
            DType::BF16 => BFloat16,
            DType::U8 => UInt8,
            DType::U32 => UInt32,
            DType::I16 => Int16,
            DType::I32 => Int32,
            DType::I64 => Int64,
            DType::F8E4M3 => Float8E4M3FN,
            // candle's MX formats (`F6E2M3`, `F6E3M2`, `F4`, `F8E8M0`) stay
            // unnamed. torch has `float8_e8m0fnu` and `float4_e2m1fn_x2`, but
            // the latter packs two values per byte, so the correspondence is
            // not established and lending the name would be the silent drift
            // this file exists to avoid.
            _ => return None,
        })
    }

    pub fn is_floating_point(self) -> bool {
        matches!(
            self,
            Float32
                | Float64
                | Float16
                | BFloat16
                | Float8E4M3FN
                | Float8E4M3FNUZ
                | Float8E5M2
                | Float8E5M2FNUZ
                | Float8E8M0FNU
                | Float4E2M1FNX2
        )
    }

    pub fn is_complex(self) -> bool {
        matches!(self, Complex32 | Complex64 | Complex128)
    }

    pub fn is_signed(self) -> bool {
        !matches!(
            self,
            UInt8 | UInt16
                | UInt32
                | UInt64
                | UInt1
                | UInt2
                | UInt3
                | UInt4
                | UInt5
                | UInt6
                | UInt7
                | Bool
                | QUInt8
                | QUInt4x2
                | QUInt2x4
                | Float8E8M0FNU
                | Bits1x8
                | Bits2x4
                | Bits4x2
                | Bits8
                | Bits16
        )
    }

    /// torch's short spelling, used as a key by `torch/utils/_dtype_abbrs.py`
    /// (`{dt: dt.abbr for dt in torch._C._get_all_dtypes()}`, at import). The
    /// mapping is torch's, read off torch 2.13.0 -- it is not derivable from
    /// the name (`bits8` is `b8x1`, and `bool` is `b8`).
    pub fn abbr(self) -> &'static str {
        match self {
            Float32 => "f32",
            Float64 => "f64",
            Float16 => "f16",
            BFloat16 => "bf16",
            Complex32 => "c32",
            Complex64 => "c64",
            Complex128 => "c128",
            UInt8 => "u8",
            UInt16 => "u16",
            UInt32 => "u32",
            UInt64 => "u64",
            Int8 => "i8",
            Int16 => "i16",
            Int32 => "i32",
            Int64 => "i64",
            Bool => "b8",
            QInt8 => "qi8",
            QUInt8 => "qu8",
            QInt32 => "qi32",
            QUInt4x2 => "qu4x2",
            QUInt2x4 => "qu2x4",
            Float8E4M3FN => "f8e4m3fn",
            Float8E4M3FNUZ => "f8e4m3fnuz",
            Float8E5M2 => "f8e5m2",
            Float8E5M2FNUZ => "f8e5m2fnuz",
            Float8E8M0FNU => "f8e8m0fnu",
            Float4E2M1FNX2 => "f4e2m1fnx2",
            Bits1x8 => "b1x8",
            Bits2x4 => "b2x4",
            Bits4x2 => "b4x2",
            Bits8 => "b8x1",
            Bits16 => "b16x1",
            // Upstream has no abbreviation for these: `dtype_abbrs` is built
            // from `_get_all_dtypes()`, which leaves all fourteen out, and
            // `torch.uint1.abbr` reads back as the full name (measured on
            // torch 2.13.0). Spelling them out here says the same thing.
            UInt1 => "uint1",
            UInt2 => "uint2",
            UInt3 => "uint3",
            UInt4 => "uint4",
            UInt5 => "uint5",
            UInt6 => "uint6",
            UInt7 => "uint7",
            Int1 => "int1",
            Int2 => "int2",
            Int3 => "int3",
            Int4 => "int4",
            Int5 => "int5",
            Int6 => "int6",
            Int7 => "int7",
        }
    }

    /// Whether `torch._C._get_all_dtypes()` lists this one. torch leaves the
    /// five quantised dtypes out of that list, and the fourteen sub-byte
    /// integer tags too -- upstream torch 2.13.0 returns exactly 27 entries,
    /// re-measured after `uint1..7`/`int1..7` were added here. That count is
    /// the reason those fourteen are excluded rather than a guess about them:
    /// `torch/utils/_dtype_abbrs.py` builds a dict keyed on exactly this list.
    pub fn in_all_dtypes(self) -> bool {
        !matches!(
            self,
            QInt8
                | QUInt8
                | QInt32
                | QUInt4x2
                | QUInt2x4
                | UInt1
                | UInt2
                | UInt3
                | UInt4
                | UInt5
                | UInt6
                | UInt7
                | Int1
                | Int2
                | Int3
                | Int4
                | Int5
                | Int6
                | Int7
        )
    }

    /// Bytes per element. torch reports 1 for `bool`, the same as `uint8`
    /// (BOOL.md §2.1) -- the two differ in semantics, not in storage width.
    pub fn itemsize(self) -> usize {
        match self {
            Complex128 => 16,
            Float64 | Complex64 | Int64 | UInt64 => 8,
            // `complex32` is **4**, not 2: it is a pair of `float16`, and
            // `itemsize` is the width of the whole complex element. Measured
            // on 2.13.0 (`torch.complex32.itemsize` is 4) -- this row said 2
            // until docs/COMPLEX2.md, which is the silent-wrong-number shape:
            // `numel * element_size` would have sized a complex32 buffer at
            // half its bytes. Nothing checked it, because the only assertion
            // in the tree was about `complex64`.
            Complex32 | Float32 | Int32 | UInt32 | QInt32 => 4,
            Float16 | BFloat16 | Int16 | UInt16 | Bits16 => 2,
            _ => 1,
        }
    }

    /// The complex tag whose *component* dtype candle is storing, for
    /// `Repr::Complex`'s two halves.
    ///
    /// The Rust-level counterpart of the `to_complex()` pymethod below, and
    /// deliberately keyed on `candle_core::DType` rather than on `TorchDType`:
    /// `PyTensorBase::complex` is handed two candle tensors and has to name
    /// the tag from what they actually store, not from what a caller says
    /// they store. `None` is the whole guard on the arm's invariant -- there
    /// is no complex tag over `int64` or `bfloat16` (upstream has no
    /// `complex(bfloat16)` either), so a pair of those cannot be built.
    ///
    /// It is the same table as `to_complex`, read from the storage side, and
    /// `pytests/test_complex.py` asserts the two agree rather than trusting
    /// that they were kept in step by hand.
    pub fn complex_for_component(component: DType) -> Option<Self> {
        Some(match component {
            DType::F16 => Complex32,
            DType::F32 => Complex64,
            DType::F64 => Complex128,
            _ => return None,
        })
    }

    /// The inverse: the component dtype behind a complex tag. `None` for every
    /// tag that is not complex, which is what makes `is_complex_tag` a derived
    /// question rather than a second list to keep in step.
    pub fn complex_component(self) -> Option<DType> {
        Some(match self {
            Complex32 => DType::F16,
            Complex64 => DType::F32,
            Complex128 => DType::F64,
            _ => return None,
        })
    }

    /// Is this one of the three complex tags?
    pub fn is_complex_tag(self) -> bool {
        self.complex_component().is_some()
    }

    /// The real tag a complex tag's halves wear -- the Rust-level `to_real()`.
    /// `None` on every non-complex tag, so `view_as_real` cannot be reached
    /// with a tag that has no real partner.
    pub fn to_real_tag(self) -> Option<Self> {
        Some(match self {
            Complex32 => Float16,
            Complex64 => Float32,
            Complex128 => Float64,
            _ => return None,
        })
    }
}

/// `torch.float32` and friends.
#[pyclass(name = "dtype", module = "torch._C", frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyDtype {
    inner: TorchDType,
}

impl PyDtype {
    pub fn new(inner: TorchDType) -> Self {
        Self { inner }
    }

    pub fn tag(&self) -> TorchDType {
        self.inner
    }

    /// The candle dtype to store this in, or a named refusal. Callers holding
    /// a dtype and needing storage go through here rather than matching on
    /// `storage()` themselves, so the message stays in one place.
    pub fn storage(&self, op: &str) -> PyResult<DType> {
        self.inner.storage().ok_or_else(|| {
            crate::err::not_implemented(format!(
                "{op}: dtype not storable by the candle backend in torch._C shim: torch.{}",
                self.inner.name()
            ))
        })
    }
}

#[pymethods]
impl PyDtype {
    fn __repr__(&self) -> String {
        format!("torch.{}", self.inner.name())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyDtype>() {
            Ok(other) => self.inner == other.inner,
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        self.inner as u64
    }

    /// torch pickles a dtype by its bare name; matching that keeps the
    /// vendored tree's serialisation paths from needing to know about us.
    fn __reduce__(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    fn is_floating_point(&self) -> bool {
        self.inner.is_floating_point()
    }

    #[getter]
    fn is_complex(&self) -> bool {
        self.inner.is_complex()
    }

    #[getter]
    fn is_signed(&self) -> bool {
        self.inner.is_signed()
    }

    #[getter]
    fn itemsize(&self) -> usize {
        self.inner.itemsize()
    }

    #[getter]
    fn abbr(&self) -> &'static str {
        self.inner.abbr()
    }

    /// Not a torch attribute. It answers the one question that is specific to
    /// this shim -- "can candle hold this?" -- so that the vendored tree, the
    /// tests and the golden harness can ask instead of keeping a second copy
    /// of `storage()`'s table.
    #[getter]
    fn _has_storage(&self) -> bool {
        self.inner.storage().is_some()
    }

    fn to_real(&self) -> PyDtype {
        PyDtype::new(match self.inner {
            Complex32 => Float16,
            Complex64 => Float32,
            Complex128 => Float64,
            other => other,
        })
    }

    fn to_complex(&self) -> PyDtype {
        PyDtype::new(match self.inner {
            Float16 => Complex32,
            Float32 => Complex64,
            Float64 => Complex128,
            // **`bfloat16` maps to `complex64`, not to itself.** Measured on
            // 2.13.0: `torch.bfloat16.to_complex()` is `torch.complex64`,
            // because there is no `complex(bfloat16)` and upstream promotes to
            // the next complex type that can hold it rather than returning the
            // input. This row returned `bfloat16` until docs/COMPLEX2.md.
            //
            // The complex tags map to themselves, as upstream.
            //
            // Still divergent, and recorded rather than fixed: upstream
            // *raises* `RuntimeError` for `to_complex()` on an integral, bool
            // or float8 dtype, where this returns the input unchanged. That is
            // a wider change than this round's subject (it turns a total
            // function partial for thirty tags with no measured caller), and
            // it is written down in docs/COMPLEX2.md §7 rather than left
            // silent.
            BFloat16 => Complex64,
            other => other,
        })
    }
}

/// `torch._C._get_all_dtypes()`. `torch/utils/_dtype_abbrs.py:5` calls this
/// during `import torch` and indexes every result by `.abbr`, so both the list
/// and the attribute are import-blocking.
#[pyfunction]
#[pyo3(name = "_get_all_dtypes")]
pub fn get_all_dtypes() -> Vec<PyDtype> {
    ALL.iter()
        .filter(|d| d.in_all_dtypes())
        .map(|d| PyDtype::new(*d))
        .collect()
}

// --- the default floating dtype ---------------------------------------------
//
// `torch.set_default_dtype` / `torch.get_default_dtype`, and the value every
// dtype-inference rule in `aten.rs` and `lib.rs` reads when the caller did not
// name a dtype. This used to be `aten::DEFAULT_FLOAT`, a `const`; docs/
// DISTRIBUTED.md §3.4 refused the setter on the grounds that it would have to
// reach a Rust constant. `transformers` ended that argument --
// `modeling_utils.py:239` calls `torch.set_default_dtype(dtype)` on the way
// into `from_pretrained`, so the const had to become a global.
//
// **Representation: an `AtomicU8` index into `DEFAULT_FLOAT_CHOICES`.** The
// alternatives and why not:
//
//   `RwLock<TorchDType>`   Reads happen on the dispatcher's hottest path -- a
//                          factory call, an integral-to-float promotion -- and
//                          each would pay an atomic read-modify-write plus a
//                          poison check to guard a value that is one byte
//                          wide. A `Relaxed` load of an `AtomicU8` is a plain
//                          byte load.
//   `OnceLock`             Cannot change, which is the entire requirement.
//   `static mut`           Unsound under any concurrent read, and Rust 2024
//                          makes even taking a reference to one an error.
//
// `Relaxed` is the right ordering, not a shortcut: the dtype tag is the whole
// of the payload. Nothing else is being published alongside it, so there is no
// happens-before edge for a stronger ordering to establish. (Upstream's own is
// a plain non-atomic C++ global, so this is if anything stricter.) In practice
// every read and every write here happens under the GIL as well.
//
// The value is an index into `DEFAULT_FLOAT_CHOICES` rather than the enum's
// discriminant so that no `transmute` and no `#[repr(u8)]` is needed, and so
// that the accept set has exactly one spelling: `set_default_dtype` refuses
// anything `position()` cannot find, and `default_float()` can only ever
// return something from that list.
use std::sync::atomic::{AtomicU8, Ordering};

/// The dtypes `torch.set_default_dtype` accepts, measured against upstream
/// 2.13.0 over every `torch.dtype` it exposes. The float8/float4 tags are
/// deliberately absent: they pass upstream's floating-point gate and then fail
/// its storage-class lookup, which is a different refusal and is reproduced as
/// one below.
///
/// Index 0 is the value torch starts at, and `DEFAULT_FLOAT_CHOICE` starts at
/// 0 to match.
const DEFAULT_FLOAT_CHOICES: &[TorchDType] = &[Float32, Float64, Float16, BFloat16];

static DEFAULT_FLOAT_CHOICE: AtomicU8 = AtomicU8::new(0);

/// The current default floating dtype. Every "the caller named no dtype" rule
/// in the shim goes through here; `set_default_dtype` is only load-bearing to
/// the extent that they do.
pub fn default_float() -> TorchDType {
    // Total by construction: the only writer is `set_default_dtype`, and the
    // only value it writes is a `position()` within this same slice.
    DEFAULT_FLOAT_CHOICES[DEFAULT_FLOAT_CHOICE.load(Ordering::Relaxed) as usize]
}

/// `torch._C._set_default_dtype`, which `torch.set_default_dtype` forwards to
/// (`torch/__init__.py:1385`).
///
/// The three refusals are upstream's, reproduced by message rather than
/// invented; the table in `test_shim.py` above
/// `test_set_default_dtype_moves_every_rule_that_reads_the_default` records
/// the measurement that produced them.
#[pyfunction]
#[pyo3(name = "_set_default_dtype")]
pub fn set_default_dtype(dtype: &Bound<'_, PyAny>) -> PyResult<()> {
    let tag = dtype
        .extract::<PyDtype>()
        .map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "invalid dtype object: only floating-point types are supported \
                 as the default type",
            )
        })?
        .tag();
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "only floating-point types are supported as the default type",
        ));
    }
    let choice = DEFAULT_FLOAT_CHOICES
        .iter()
        .position(|d| *d == tag)
        .ok_or_else(|| {
            // Upstream gets here by looking for a `torch.<Name>Storage` class
            // and not finding one, so it names the class rather than the
            // dtype. The spelling is the dtype name with its first character
            // capitalised: `float8_e4m3fn` -> `Float8_e4m3fnStorage`.
            let mut name = tag.name().to_owned();
            name[..1].make_ascii_uppercase();
            pyo3::exceptions::PyTypeError::new_err(format!(
                "couldn't find storage object {name}Storage"
            ))
        })?;
    DEFAULT_FLOAT_CHOICE.store(choice as u8, Ordering::Relaxed);
    Ok(())
}

/// `torch.get_default_dtype()`. Upstream binds `THPModule_getDefaultDtype`
/// straight onto `_C` (`torch/_C/__init__.pyi:1399`) rather than routing it
/// through the operator table; `bootstrap.py` used to install a constant
/// function here, which is what made this a getter with only one answer.
///
/// It returns the *interned* object, so `torch.get_default_dtype() is
/// torch.float32` holds as it does upstream -- dtypes are used as dict keys
/// across the vendored tree.
#[pyfunction]
#[pyo3(name = "get_default_dtype")]
pub fn get_default_dtype(py: Python<'_>) -> PyResult<Py<PyAny>> {
    interned(py, default_float())
}

/// The module-level `torch.float32` and friends, kept so that anything handing
/// a dtype back to Python can hand back *the* object rather than an equal one.
static INTERNED: std::sync::OnceLock<Vec<(TorchDType, Py<PyAny>)>> =
    std::sync::OnceLock::new();

pub(crate) fn interned(py: Python<'_>, tag: TorchDType) -> PyResult<Py<PyAny>> {
    INTERNED
        .get()
        .and_then(|made| made.iter().find(|(d, _)| *d == tag))
        .map(|(_, object)| object.clone_ref(py))
        .ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "torch._C shim: torch.{} was never registered on the module",
                tag.name()
            ))
        })
}

/// Look a dtype up by its torch spelling, aliases included.
pub fn by_name(name: &str) -> Option<TorchDType> {
    ALL.iter()
        .copied()
        .find(|d| d.name() == name)
        .or_else(|| ALIASES.iter().find(|(n, _)| *n == name).map(|(_, d)| *d))
}

/// Registers the type and the module-level dtype instances.
///
/// Upstream puts these on `torch`, not on `torch._C` -- the C side does
/// `PyModule_AddObject` against the `torch` module during `_initExtension`
/// (VENDOR.md wall 7). Registering them on `_C` and letting
/// `from torch._C import *` carry them across lands them in the same place by
/// a different route; `bootstrap.py`'s `_initExtension` fills in whatever the
/// star import did not cover.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDtype>()?;
    m.add_function(wrap_pyfunction!(get_all_dtypes, m)?)?;
    m.add_function(wrap_pyfunction!(set_default_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(get_default_dtype, m)?)?;

    // One Python object per dtype, shared with its aliases. torch guarantees
    // `torch.float is torch.float32`, and the tree uses dtypes as dict keys
    // in enough places that two objects comparing equal but not identical is
    // a difference worth not having.
    let py = m.py();
    let mut made: Vec<(TorchDType, Py<PyAny>)> = Vec::with_capacity(ALL.len());
    for dtype in ALL {
        let object = PyDtype::new(*dtype).into_pyobject(py)?.into_any().unbind();
        m.add(dtype.name(), object.clone_ref(py))?;
        made.push((*dtype, object));
    }
    for (alias, dtype) in ALIASES {
        let object = made
            .iter()
            .find(|(d, _)| d == dtype)
            .map(|(_, o)| o.clone_ref(py))
            .expect("every alias names a dtype in ALL");
        m.add(*alias, object)?;
    }
    // Keep them reachable from Rust, for `get_default_dtype` -- see `interned`.
    // A second call would be a second module init, which this build does not
    // support (no multi-interpreter slot), so losing the race is not a case
    // that can arise; `set` rather than `get_or_init` says so by ignoring it.
    let _ = INTERNED.set(made);
    Ok(())
}
