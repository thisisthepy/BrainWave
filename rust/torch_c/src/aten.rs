//! The aten dispatch choke point.
//!
//! DESIGN.md §6 rejects enumerating the op set up front and makes the shim its
//! own instrument instead: every aten call goes through one entry, and anything
//! missing raises with its own name in the message. Running a model then prints
//! the work queue in frequency order, which is better ordering information than
//! a list written in advance.
//!
//! That only holds if there is exactly one entrance. Hence no arithmetic on
//! `TensorBase` -- a convenience method would be a second door that the
//! instrument cannot see through.
//!
//! Overloads are part of the key (`aten.add.Tensor`, not `aten.add`). torch
//! resolves overloads before it reaches the kernel, and folding them together
//! here would make `add.Tensor` and `add.Scalar` -- different schemas -- look
//! like one implemented op.
use candle_core::{Device, Tensor};
use pyo3::prelude::*;
use pyo3::PyErr;
use pyo3::intern;
use pyo3::types::{PyDict, PyList, PyModule, PyString, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::device::PyDevice;
use crate::dtype::{default_float, PyDtype, TorchDType};
use crate::err::{aten_not_implemented, candle_err, not_implemented};
use crate::reduced::{FastDType, Fused};
use crate::tensor::PyTensorBase;

/// Every op with a real kernel behind it. Kept sorted; `_aten_implemented()`
/// hands it to Python so the vendored layer and the tests can ask rather than
/// keep their own copy of the list.
///
/// The `TensorBase` surface (docs/TENSORBASE.md) is interleaved here rather
/// than kept in a block of its own. `_aten_implemented()` means exactly one
/// thing -- "this op has a kernel *and* `tools/golden/cases.py` compares it
/// against upstream" -- and which Python spelling reaches an op is not part of
/// that meaning.
pub const IMPLEMENTED: &[&str] = &[
    "aten._grouped_mm.default",
    "aten._local_scalar_dense.default",
    "aten._log_softmax.default",
    "aten._safe_softmax.default",
    "aten._scaled_dot_product_flash_attention_for_cpu.default",
    "aten._softmax.default",
    "aten._to_copy.default",
    "aten._unsafe_view.default",
    "aten._weight_norm_interface.default",
    "aten.abs.default",
    "aten.adaptive_avg_pool1d.default",
    "aten.adaptive_avg_pool2d.default",
    "aten.greater.Scalar",
    "aten.greater.Tensor",
    "aten.roll.default",
    "aten.where.ScalarSelf",
    "aten.abs_.default",
    "aten.add.Scalar",
    "aten.add.Tensor",
    "aten.add_.Scalar",
    "aten.add_.Tensor",
    "aten.addmm.default",
    "aten.alias.default",
    "aten.all.default",
    "aten.all.dim",
    "aten.all.dims",
    "aten.amax.default",
    "aten.any.default",
    "aten.any.dim",
    "aten.arange.default",
    "aten.arange.start",
    "aten.arange.start_step",
    "aten.argmax.default",
    "aten.avg_pool2d.default",
    "aten.max_pool2d.default",
    "aten.baddbmm.default",
    "aten.bernoulli_.float",
    "aten.bitwise_and.Scalar",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_not.default",
    "aten.bitwise_or.Scalar",
    "aten.bitwise_or.Tensor",
    "aten.bmm.default",
    "aten.cat.default",
    "aten.ceil.default",
    "aten.ceil_.default",
    "aten.clamp.default",
    "aten.clamp_.default",
    "aten.clamp_min.default",
    "aten.clamp_min_.default",
    "aten.clone.default",
    "aten.constant_pad_nd.default",
    "aten.convolution.default",
    "aten.copy_.default",
    "aten.cos.default",
    "aten.cos_.default",
    "aten.cumsum.default",
    "aten.detach.default",
    "aten.div.Scalar_mode",
    "aten.div.Tensor",
    "aten.div.Tensor_mode",
    "aten.div_.Scalar",
    "aten.div_.Tensor",
    "aten.embedding.default",
    "aten.empty.memory_format",
    "aten.empty_like.default",
    "aten.eq.Scalar",
    "aten.eq.Tensor",
    "aten.erf.default",
    "aten.erf_.default",
    "aten.exp.default",
    "aten.exp_.default",
    "aten.expm1.default",
    "aten.expm1_.default",
    "aten.expand.default",
    "aten.fill_.Scalar",
    "aten.fill_.Tensor",
    "aten.flip.default",
    "aten.floor_divide.default",
    "aten.floor_divide.Scalar",
    "aten.full.default",
    "aten.full_like.default",
    "aten.gather.default",
    "aten.hardtanh.default",
    "aten.ge.Scalar",
    "aten.ge.Tensor",
    "aten.gelu.default",
    "aten.gt.Scalar",
    "aten.gt.Tensor",
    "aten.histc.default",
    "aten.index.Tensor",
    "aten.index_put_.default",
    "aten.is_floating_point.default",
    "aten.isin.Tensor_Tensor",
    "aten.le.Scalar",
    "aten.le.Tensor",
    "aten.leaky_relu.default",
    "aten.lift_fresh.default",
    "aten.linalg_vector_norm.default",
    "aten.linspace.default",
    "aten.log.default",
    "aten.log2.default",
    "aten.log2_.default",
    "aten.log_.default",
    "aten.lt.Scalar",
    "aten.lt.Tensor",
    "aten.masked_fill.Scalar",
    "aten.masked_fill_.Scalar",
    "aten.masked_select.default",
    "aten.matmul.default",
    "aten.max.default",
    "aten.max.dim",
    "aten.max.other",
    "aten.mean.default",
    "aten.mean.dim",
    "aten.min.default",
    "aten.min.dim",
    "aten.min.other",
    "aten.mm.default",
    "aten.mul.Scalar",
    "aten.mul.Tensor",
    "aten.mul_.Scalar",
    "aten.mul_.Tensor",
    "aten.multinomial.default",
    "aten.native_batch_norm.default",
    "aten.native_group_norm.default",
    "aten.native_dropout.default",
    "aten.native_layer_norm.default",
    "aten.ne.Scalar",
    "aten.ne.Tensor",
    "aten.neg.default",
    "aten.neg_.default",
    "aten.new_ones.default",
    "aten.new_zeros.default",
    "aten.nll_loss_forward.default",
    "aten.norm.ScalarOpt_dim",
    "aten.normal_.default",
        "aten.reshape.default",
    "aten.one_hot.default",
    "aten.ones.default",
    "aten.ones_like.default",
    "aten.permute.default",
    "aten.pow.Scalar",
    "aten.pow.Tensor_Scalar",
    "aten.pow.Tensor_Tensor",
    "aten.randint.default",
    "aten.randint.low",
    "aten.randperm.default",
    "aten.reciprocal.default",
    "aten.reciprocal_.default",
    "aten.relu.default",
    "aten.relu_.default",
    "aten.remainder.Scalar",
    "aten.remainder.Tensor",
    "aten.repeat.default",
    "aten.rsqrt.default",
    "aten.rsqrt_.default",
    "aten.rsub.Scalar",
    "aten.scalar_tensor.default",
    "aten.scatter.src",
    "aten.select.int",
    "aten.sigmoid.default",
    "aten.sigmoid_.default",
    "aten.sign.default",
    "aten.silu.default",
    "aten.sin.default",
    "aten.sin_.default",
    "aten.slice.Tensor",
    "aten.softplus.default",
    "aten.sort.default",
    "aten.split.Tensor",
    "aten.split_with_sizes.default",
    "aten.sqrt.default",
    "aten.sqrt_.default",
    "aten.squeeze.default",
    "aten.squeeze.dim",
    "aten.squeeze.dims",
    "aten.stack.default",
    "aten.sub.Scalar",
    "aten.sub.Tensor",
    "aten.sub_.Scalar",
    "aten.sub_.Tensor",
    "aten.sum.default",
    "aten.sum.dim_IntList",
    "aten.t.default",
    "aten.tanh.default",
    "aten.tanh_.default",
    "aten.topk.default",
    "aten.transpose.int",
    "aten.tril.default",
    "aten.triu.default",
    "aten.unbind.int",
    "aten.uniform_.default",
    "aten.unsqueeze.default",
    "aten.upsample_bilinear2d.default",
    "aten.view.default",
    "aten.view.dtype",
    "aten.where.ScalarOther",
    "aten.nonzero.default",
    "aten.where.default",
    "aten.where.self",
    "aten.zero_.default",
    "aten.zeros_like.default",
];

/// Ops with a real kernel that `_aten_implemented()` does **not** advertise.
///
/// This is a reporting workaround, not a capability one, and it is the only
/// thing in this file that is neither implemented nor refused. The golden
/// harness treats "advertised in `_aten_implemented()` but absent from
/// `tools/golden/cases.py::CASE_BUILDERS`" as a hard failure, on purpose, so
/// that an op cannot be added without being compared against upstream. That
/// rule is right; this task simply was not allowed to edit the harness, and
/// the one op it produced that the harness has no builder for is parked here
/// rather than turned off.
///
/// `_aten_dispatch` reaches these exactly like any other op -- so
/// `torch.randint(10, (2,))` works -- but the coverage number stays honest in
/// the conservative direction (it under-reports rather than over-reports).
/// The fix is one case builder and one line move.
///
/// `aten.mul.Scalar` was the first to get that fix (docs/TAIL.md): a
/// re-measurement of `falcon` under `_aten_all_implemented()` found the
/// kernel already dispatching, so the remaining work was exactly the case
/// builder the comment above describes, plus this one line move.
///
/// `aten.max.other` was the second, and it is worth recording that the case
/// builder written for it while it was parked here (docs/SPELLINGS.md §7.3)
/// *found a live defect* -- a NaN in the second operand was dropped -- and
/// held it as a deliberately failing case until the kernel could be fixed.
/// Promotion and fix landed together; a builder written against a parked op is
/// not a formality.
/// `aten.add.Scalar` and `aten.sub.Scalar` were the third and fourth, and
/// they are the ones that say the parking list is not free. docs/SCALAR.md
/// §6 recorded that the *narrowing* half of the scalar family had no golden
/// coverage because of this list -- sabotage F3 was caught by two smoke tests
/// and **zero** golden cases -- and writing the two builders on promotion
/// found a live defect that nothing had ever been in a position to see:
/// neither op had ever been passed a non-unit `alpha` by any case, and
/// `alpha` is narrowed to the tensor's dtype *separately* from `other`
/// upstream. `bfloat16` and `float16` went from 53/150 and 56/150 disagreeing
/// rows to 0. Same shape as `aten.max.other` above: a builder written against
/// a parked op is not a formality.
///
/// `aten.randint.default` was the fifth, and it is the one that says parking
/// can hide a *wrong answer* and not merely an unmeasured one. It sat here
/// while its sibling `aten.randint.low` was advertised against a comparator
/// that checked membership of `[low, high)` and deliberately not the
/// sequence -- so between the two of them, nothing in the harness was ever in
/// a position to notice that both drew from a different generator entirely
/// (docs/RANDINT.md §8). Promotion came with the fix, and the builder that
/// replaced that comparator compares seeded values.
pub const IMPLEMENTED_AWAITING_GOLDEN: &[&str] = &[
    "aten.any.dims",
    "aten.contiguous.default",
    "aten.div.Scalar",
    "aten.masked_fill.Tensor",
    "aten.zeros.default",
];

/// Everything `_aten_dispatch` answers, whether or not it is advertised: the
/// two lists above, unioned. A function rather than a third constant, so that
/// the smoke tests can check the dispatch table against it without anything
/// being able to read the union and report it as golden-compared coverage.
pub fn all_implemented() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = IMPLEMENTED
        .iter()
        .chain(IMPLEMENTED_AWAITING_GOLDEN.iter())
        .copied()
        .collect();
    out.sort_unstable();
    out
}

// `set_default_dtype` arrived, so the default float dtype is no longer a
// constant here -- it is `dtype::default_float()`, a process-global. The
// dtype-inference rules below ("integral arguments give int64, anything else
// gives the default float") call it rather than reading a copy, which is the
// whole of what makes the setter load-bearing rather than decorative.

/// What Python calls. A thin wrapper whose only job is to split `op` off the
/// front of the argument tuple and hand the rest to `aten_dispatch`, which is
/// still the one door and still where everything happens.
///
/// **The signature is exactly `(*args, **kwargs)` and takes no `py`, and both
/// halves of that are load-bearing.** Anything else makes pyo3 take its
/// general argument-extraction path: it iterates every keyword, `to_str()`s
/// each key, compares it against the named parameters, and `set_item`s the
/// survivors into a **freshly allocated `PyDict`** that is item-for-item the
/// dict it was just handed (`DictVarkeywords::handle_varkeyword`,
/// pyo3-0.29.2). That was the single largest shim-side leaf in the profile --
/// 410 of 6031 samples, above every kernel.
///
/// The escape is `pyo3-macros-backend/src/params.rs::is_forwarded_args`,
/// which forwards the caller's tuple and dict untouched:
///
/// ```ignore
/// matches!(signature.arguments.as_slice(),
///          [FnArg::VarArgs(..), FnArg::KwArgs(..),])
/// ```
///
/// **`Python<'_>` is an `FnArg::Py` and therefore an element of that slice**,
/// so a `py` parameter fails the match exactly as a named `op` does. That is
/// not obvious and it is not documented; it cost a build and a profile to
/// find, which is why it is written down here. `py` is recovered from
/// `args.py()` instead -- the tuple is a live Python object, so it can hand
/// back the token for free.
///
/// Nothing about the door changes for Python. `op` is still the first
/// positional argument, still required, and an unimplemented operator is
/// still refused *by name*; the two `TypeError`s pyo3 used to raise for a
/// missing or non-string `op` are raised here instead, in the same words.
///
/// It is a wrapper rather than a rewrite of `aten_dispatch` because
/// `capture.rs` replays a recorded graph by calling `aten_dispatch` from Rust,
/// where the op is already a `String` and there is no tuple to split.
#[pyfunction]
#[pyo3(name = "_aten_dispatch", signature = (*args, **kwargs))]
pub fn aten_dispatch_entry(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    // Not a parameter, for the reason in the doc comment above. `Python` is a
    // zero-sized proof that the GIL is held, and `args` -- a live object we
    // were handed while it is held -- can reproduce it without a C call.
    let py = args.py();
    // BORROWED, not owned. `PyTuple_GetItem` returns a borrowed reference and
    // pyo3's `Borrowed` models exactly that -- no incref is taken here and
    // none is released. Holding it for the rest of the call is sound because
    // `args` is the caller's own argument tuple, which CPython keeps alive
    // across the call, and the borrow's lifetime is tied to `args` by the type
    // system rather than by this comment.
    //
    // That is what makes the `&str` below safe: it points into the
    // interpreter's UTF-8 buffer for that string object and must not outlive
    // it, and tying it to `args` makes that automatic rather than a rule
    // someone has to remember.
    let op_obj = args.get_borrowed_item(0).map_err(|_| {
        // pyo3's own wording for the same mistake, kept verbatim so that
        // anything matching on it keeps matching.
        pyo3::exceptions::PyTypeError::new_err(
            "_aten_dispatch() missing 1 required positional argument: 'op'",
        )
    })?;
    let op: &str = op_obj.extract().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "argument 'op': '{}' object cannot be converted to 'PyString'",
            op_obj
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_default()
        ))
    })?;
    // OWNED. `PyTuple_GetSlice` returns a new reference; this `Bound` owns it
    // and drops it at the end of the call. This is the identical slice pyo3's
    // `TupleVarargs` handler used to build on the way in, so it is not a new
    // cost -- and for the shape `bootstrap.py` actually emits (every argument
    // bound by keyword, so `args` is just `(op,)`) the slice is empty and
    // CPython hands back the interned empty tuple without allocating.
    let rest = args.get_slice(1, args.len());
    aten_dispatch(py, op, &rest, kwargs)
}


// ---------------------------------------------------------------------------
// `float8_e4m3fn`: refusing exactly what upstream refuses (docs/FLOAT8B.md)
// ---------------------------------------------------------------------------

/// The 114 ops upstream 2.13.0 refuses for `float8_e4m3fn`, paired with the
/// **kernel name upstream puts in the message**, transcribed per op.
///
/// docs/FLOAT8B.md §2: the wording is not unified because the difference carries
/// information. `add`, `sub` and `rsub` all report `"add_stub"`; `mean` and `sum`
/// both report `"sum_cpu"`; `relu` reports `"clamp_min_scalar_cpu"` while
/// `hardtanh` reports `"clamp_scalar_cpu"`. Each pair says which kernel upstream
/// would have dispatched to, and a house string would erase it. Same reasoning as
/// docs/PROMOTE.md and docs/BACKWARD2.md, which both transcribed rather than
/// unified.
///
/// Sorted by op: `float8_e4m3fn_kernel` binary-searches it, and a duplicate is
/// visible on sight.
static FLOAT8_E4M3FN_REFUSALS: &[(&str, &str)] = &[
    ("aten._log_softmax.default", "log_softmax_lastdim_kernel_impl"),
    ("aten._safe_softmax.default", "softmax_lastdim_kernel_impl"),
    ("aten._scaled_dot_product_flash_attention_for_cpu.default", "flash_attention"),
    ("aten._softmax.default", "softmax_lastdim_kernel_impl"),
    ("aten._weight_norm_interface.default", "weight_norm_kernel"),
    ("aten.add.Scalar", "add_stub"),
    ("aten.add.Tensor", "add_stub"),
    ("aten.add_.Scalar", "add_stub"),
    ("aten.add_.Tensor", "add_stub"),
    ("aten.amax.default", "max_values_cpu"),
    ("aten.argmax.default", "argmax_cpu"),
    ("aten.avg_pool2d.default", "avg_pool2d"),
    ("aten.baddbmm.default", "baddbmm"),
    ("aten.bernoulli_.float", "bernoulli_scalar_cpu_"),
    ("aten.bmm.default", "bmm"),
    ("aten.ceil.default", "ceil_vml_cpu"),
    ("aten.ceil_.default", "ceil_vml_cpu"),
    ("aten.clamp.default", "clamp_scalar_cpu"),
    ("aten.clamp_.default", "clamp_scalar_cpu"),
    ("aten.clamp_min.default", "clamp_min_scalar_cpu"),
    ("aten.clamp_min_.default", "clamp_min_scalar_cpu"),
    ("aten.convolution.default", "slow_conv2d_cpu"),
    ("aten.cos.default", "cos_vml_cpu"),
    ("aten.cos_.default", "cos_vml_cpu"),
    ("aten.cumsum.default", "cumsum_out_cpu"),
    ("aten.div.Scalar_mode", "div_cpu_reduced_float"),
    ("aten.div.Tensor", "div_cpu"),
    ("aten.div.Tensor_mode", "div_cpu"),
    ("aten.div_.Scalar", "div_cpu_reduced_float"),
    ("aten.div_.Tensor", "div_cpu"),
    ("aten.erf.default", "erf_vml_cpu"),
    ("aten.erf_.default", "erf_vml_cpu"),
    ("aten.exp.default", "exp_vml_cpu"),
    ("aten.exp_.default", "exp_vml_cpu"),
    ("aten.expm1.default", "expm1_vml_cpu"),
    ("aten.expm1_.default", "expm1_vml_cpu"),
    ("aten.flip.default", "flip_cpu"),
    ("aten.floor_divide.Scalar", "div_floor_cpu_reduced_float"),
    ("aten.floor_divide.default", "div_floor_cpu"),
    ("aten.gather.default", "scatter_gather_tensor_cpu"),
    ("aten.ge.Scalar", "ge_cpu"),
    ("aten.ge.Tensor", "ge_cpu"),
    ("aten.gelu.default", "GeluKernelImpl"),
    ("aten.gt.Scalar", "gt_cpu"),
    ("aten.gt.Tensor", "gt_cpu"),
    ("aten.hardtanh.default", "clamp_scalar_cpu"),
    ("aten.histc.default", "aminmax_cpu"),
    ("aten.isin.Tensor_Tensor", "isin_default_cpu"),
    ("aten.le.Scalar", "le_cpu"),
    ("aten.le.Tensor", "le_cpu"),
    ("aten.leaky_relu.default", "leaky_relu_cpu"),
    ("aten.linalg_vector_norm.default", "norm_cpu"),
    ("aten.log.default", "log_vml_cpu"),
    ("aten.log2.default", "log2_vml_cpu"),
    ("aten.log2_.default", "log2_vml_cpu"),
    ("aten.log_.default", "log_vml_cpu"),
    ("aten.lt.Scalar", "lt_cpu"),
    ("aten.lt.Tensor", "lt_cpu"),
    ("aten.masked_fill.Scalar", "masked_fill"),
    ("aten.masked_fill_.Scalar", "masked_fill"),
    ("aten.max.default", "max_all"),
    ("aten.max.dim", "max_cpu"),
    ("aten.max.other", "maximum_cpu"),
    ("aten.max_pool2d.default", "max_pool2d"),
    ("aten.mean.default", "sum_cpu"),
    ("aten.mean.dim", "sum_cpu"),
    ("aten.min.default", "min_all"),
    ("aten.min.dim", "min_cpu"),
    ("aten.min.other", "minimum_cpu"),
    ("aten.mul.Scalar", "mul_cpu_reduced_float"),
    ("aten.multinomial.default", "aminmax_cpu"),
    ("aten.native_batch_norm.default", "batch_norm"),
    ("aten.native_group_norm.default", "GroupNormKernelImpl"),
    ("aten.native_layer_norm.default", "LayerNormKernelImpl"),
    ("aten.neg.default", "neg_cpu"),
    ("aten.neg_.default", "neg_cpu"),
    ("aten.nll_loss_forward.default", "nll_loss_out_frame"),
    ("aten.norm.ScalarOpt_dim", "norm_cpu"),
    ("aten.normal_.default", "normal_kernel_cpu"),
    ("aten.pow.Tensor_Tensor", "pow"),
    ("aten.reciprocal.default", "reciprocal_cpu"),
    ("aten.reciprocal_.default", "reciprocal_cpu"),
    ("aten.relu.default", "clamp_min_scalar_cpu"),
    ("aten.relu_.default", "clamp_min_scalar_cpu"),
    ("aten.remainder.Scalar", "remainder_cpu"),
    ("aten.remainder.Tensor", "remainder_cpu"),
    ("aten.rsqrt.default", "rsqrt_cpu"),
    ("aten.rsqrt_.default", "rsqrt_cpu"),
    ("aten.rsub.Scalar", "add_stub"),
    ("aten.scatter.src", "scatter_gather_tensor_cpu"),
    ("aten.sigmoid.default", "sigmoid_cpu_reduced_float"),
    ("aten.sigmoid_.default", "sigmoid_cpu_reduced_float"),
    ("aten.sign.default", "sign_cpu"),
    ("aten.silu.default", "silu_cpu"),
    ("aten.sin.default", "sin_vml_cpu"),
    ("aten.sin_.default", "sin_vml_cpu"),
    ("aten.softplus.default", "softplus_cpu"),
    ("aten.sort.default", "sorting_kernel_method_name"),
    ("aten.sqrt.default", "sqrt_vml_cpu"),
    ("aten.sqrt_.default", "sqrt_vml_cpu"),
    ("aten.sub.Scalar", "add_stub"),
    ("aten.sub.Tensor", "add_stub"),
    ("aten.sub_.Scalar", "add_stub"),
    ("aten.sub_.Tensor", "add_stub"),
    ("aten.sum.default", "sum_cpu"),
    ("aten.sum.dim_IntList", "sum_cpu"),
    ("aten.tanh.default", "tanh_vml_cpu"),
    ("aten.tanh_.default", "tanh_vml_cpu"),
    ("aten.topk.default", "topk_cpu"),
    ("aten.tril.default", "tril"),
    ("aten.triu.default", "triu"),
    ("aten.uniform_.default", "check_uniform_bounds"),
    ("aten.upsample_bilinear2d.default", "upsample_bilinear2d_channels_last"),
    ("aten.nonzero.default", "nonzero_count_cpu"),
    ("aten.where.default", "nonzero_count_cpu"),
];

/// Upstream's exact refusal for `op`, or `None` if upstream computes it.
fn float8_e4m3fn_kernel(op: &str) -> Option<&'static str> {
    FLOAT8_E4M3FN_REFUSALS
        .binary_search_by(|(name, _)| (*name).cmp(op))
        .ok()
        .map(|i| FLOAT8_E4M3FN_REFUSALS[i].1)
}

/// The ten ops upstream **computes** for this dtype and this build cannot
/// (docs/FLOAT8B.md §4.1): candle 0.11.0's `WithDType for f8e4m3::to_f64`
/// recurses into itself and release-mode LLVM turns that into `.L1: jmp .L1`,
/// so they hang instead of answering.
///
/// They are refused in **this shim's** words, never upstream's. Reporting
/// `"pow" not implemented for 'Float8_e4m3fn'` for `aten.pow.Scalar` would be a
/// lie: upstream implements it.
/// **Empty as of docs/FLOAT8C.md.** All ten ops it held are settled: eight
/// compute here now that `widen_f64` routes around candle's poisoned arm, and
/// the remaining two (`aten.pow.Scalar`, `aten.pow.Tensor_Scalar`) turned out
/// to be ops upstream refuses for all but a degenerate exponent, so they moved
/// to `float8_pow_refusal` and carry upstream's own `"pow"` wording.
///
/// The list is kept rather than deleted because it is the right home for a
/// future op that this shim genuinely cannot answer for this dtype, and its
/// emptiness is the statement that there is no such op today.
static FLOAT8_E4M3FN_SHIM_ONLY: &[&str] = &[];

fn float8_shim_only_refusal(op: &str) -> bool {
    FLOAT8_E4M3FN_SHIM_ONLY.binary_search(&op).is_ok()
}

/// `aten.pow.Scalar` and `aten.pow.Tensor_Scalar` for `float8_e4m3fn`, where
/// upstream's refusal is **value-dependent** rather than op-level -- the same
/// shape as `aten.matmul.default`'s shape-dependence (docs/FLOAT8B.md §2.1),
/// and the reason neither belongs in `FLOAT8_E4M3FN_REFUSALS`.
///
/// Measured on 2.13.0 over `{0, 1, 2, 3, -1, 0.5, 1.5, True, False}`:
///
/// * `pow.Tensor_Scalar(t, e)` computes for `e == 0` (all-ones) and `e == 1`
///   (a copy) and raises `"pow" not implemented for 'Float8_e4m3fn'` for every
///   other exponent. Those two are TensorIterator short-circuits that never
///   reach a kernel.
/// * `pow.Scalar(b, t)` computes only for `b == 1` -- all-ones, independent of
///   the exponent tensor. `b == 0` raises, unlike the `Tensor_Scalar` side.
///
/// Returns `true` when upstream would refuse, i.e. when the shim must too.
fn float8_pow_refuses(op: &str, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> bool {
    let (index, name) = match op {
        "aten.pow.Tensor_Scalar" => (1usize, "exponent"),
        "aten.pow.Scalar" => (0usize, "self"),
        _ => return false,
    };
    let Ok(Some(value)) = scalar_arg(op, args, kwargs, index, name) else {
        // Unreadable operand: leave it to the op, which reports the real
        // argument error rather than a dtype refusal that hides it.
        return false;
    };
    // `Scalar` carries `Int`/`Float`; a Python `bool` arrives as `Int`, which
    // is why `pow(t, True)` and `pow(t, 1)` land on the same answer here and
    // do upstream too (measured).
    let as_f64 = value.as_f64();
    if op == "aten.pow.Scalar" {
        return as_f64 != 1.0;
    }
    as_f64 != 0.0 && as_f64 != 1.0
}

/// Whether the float8 gate applies to this call: at least one tensor operand is
/// `float8_e4m3fn` and no tensor operand is a *different* floating-point dtype.
///
/// The second half is the whole subtlety (docs/FLOAT8B.md §2.1). Upstream refuses
/// a mixed call like `add(float8, float32)` **before** it looks up a kernel, with
/// a `RuntimeError` -- `Promotion for Float8 Types is not supported` -- and
/// answering that with `NotImplementedError` would be a new divergence in
/// exception *type*, which is the failure docs/PROMOTE.md's last section is about.
///
/// Integer and bool operands do not block the gate: `gather`'s int64 index and
/// `masked_fill`'s bool mask do not promote, and upstream does report the kernel
/// message for those calls (measured, docs/FLOAT8B.md §2.1).
fn float8_only_floats(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> bool {
    let mut saw_f8 = false;
    let mut saw_other_float = false;
    fn visit(value: &Bound<'_, PyAny>, saw_f8: &mut bool, saw_other_float: &mut bool) -> bool {
        let Ok(tensor) = value.cast::<PyTensorBase>() else {
            return false;
        };
        let tag = tensor.borrow().tag();
        if tag == TorchDType::Float8E4M3FN {
            *saw_f8 = true;
        } else if tag.is_floating_point() {
            *saw_other_float = true;
        }
        true
    }
    fn scan(value: &Bound<'_, PyAny>, saw_f8: &mut bool, saw_other_float: &mut bool) {
        if visit(value, saw_f8, saw_other_float) {
            return;
        }
        // One level down, exactly as the device gate descends: `cat`/`stack`
        // take `Tensor[]`, and `bootstrap.py` binds it by keyword.
        if let Ok(sequence) = value.cast::<PyList>() {
            for item in sequence.iter() {
                if !visit(&item, saw_f8, saw_other_float) {
                    break;
                }
            }
        } else if let Ok(sequence) = value.cast::<PyTuple>() {
            for item in sequence.iter() {
                if !visit(&item, saw_f8, saw_other_float) {
                    break;
                }
            }
        }
    }
    for value in args.iter() {
        scan(&value, &mut saw_f8, &mut saw_other_float);
    }
    if let Some(kwargs) = kwargs {
        for (_, value) in kwargs.iter() {
            scan(&value, &mut saw_f8, &mut saw_other_float);
        }
    }
    saw_f8 && !saw_other_float
}

/// Every tensor operand is 1-D. Only `aten.matmul.default` asks (see the gate).
fn all_operands_are_1d(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> bool {
    let mut seen = false;
    let mut all_1d = true;
    let mut check = |value: &Bound<'_, PyAny>| {
        if let Ok(tensor) = value.cast::<PyTensorBase>() {
            seen = true;
            if tensor.borrow().dims().len() != 1 {
                all_1d = false;
            }
        }
    };
    for value in args.iter() {
        check(&value);
    }
    if let Some(kwargs) = kwargs {
        for (_, value) in kwargs.iter() {
            check(&value);
        }
    }
    seen && all_1d
}

/// The gate itself, called once at the single door.
///
/// Table first, operands second: the two binary searches miss for every op on
/// neither list, and the operand scan -- the only part that touches Python
/// objects -- runs only after one of them hits. Nothing on the float32 path
/// scans anything new.
///
/// At the door rather than in 114 kernels for the reason the device gate is
/// there: a kernel can forget, and the 27 ops in docs/FLOAT8B.md table B never
/// reach a kernel at all -- they disappear into candle's `F8E4M3 -> f64`
/// tail-call loop first. Only the door runs before that.
fn float8_e4m3fn_gate(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if let Some(kernel) = float8_e4m3fn_kernel(op) {
        if float8_only_floats(args, kwargs) {
            // Upstream's text, character for character, and upstream's exception
            // type. A caller writing `except NotImplementedError` catches this
            // exactly as it catches upstream's.
            return Err(not_implemented(format!(
                "\"{kernel}\" not implemented for 'Float8_e4m3fn'"
            )));
        }
        return Ok(());
    }
    // `aten.matmul.default` is not on the table: upstream's refusal for it is
    // **shape-dependent**, not op-level -- 2-D x 2-D returns a float8 result, and
    // only the 1-D x 1-D form falls through to `dot` (docs/FLOAT8B.md §2.1). So
    // it is gated on the shape upstream actually refuses, and left alone on the
    // shape upstream answers.
    if op == "aten.matmul.default" && float8_only_floats(args, kwargs) && all_operands_are_1d(args, kwargs) {
        return Err(not_implemented("\"dot\" not implemented for 'Float8_e4m3fn'"));
    }
    // `aten.adaptive_avg_pool1d.default` is value-dependent in the same way,
    // and docs/FLOAT8B.md table D has it as an op upstream *computes* only
    // because the generic recipe synthesised `output_size=[0]` -- an empty
    // output, where no kernel is ever dispatched. Re-measured over
    // `[0] [1] [2] [4]` and 2-D and 3-D inputs, upstream refuses every
    // non-empty output size, and with **two different kernel names**:
    // `output_size == [1]` is the global-average path and says `"sum_cpu"`,
    // everything else says `"adaptive_avg_pool2d"`.
    if op == "aten.adaptive_avg_pool1d.default" && float8_only_floats(args, kwargs) {
        if let Ok(output_size) = shape_arg(op, args, kwargs, 1, "output_size") {
            match output_size.first().copied() {
                Some(0) => {}
                Some(1) => {
                    return Err(not_implemented(
                        "\"sum_cpu\" not implemented for 'Float8_e4m3fn'",
                    ))
                }
                Some(_) => {
                    return Err(not_implemented(
                        "\"adaptive_avg_pool2d\" not implemented for 'Float8_e4m3fn'",
                    ))
                }
                None => {}
            }
        }
    }
    if float8_pow_refuses(op, args, kwargs) && float8_only_floats(args, kwargs) {
        return Err(not_implemented("\"pow\" not implemented for 'Float8_e4m3fn'"));
    }
    if float8_shim_only_refusal(op) && float8_only_floats(args, kwargs) {
        return Err(not_implemented(format!(
            "{op}: float8_e4m3fn is not supported by this op in the torch._C shim \
             (candle's F8E4M3 -> f64 conversion does not terminate); upstream \
             computes this -- docs/FLOAT8B.md §4.1"
        )));
    }
    Ok(())
}

/// `t.to_dtype(DType::F64)`, with `F8E4M3` widened through `F32` on the way.
///
/// **This is the whole of docs/FLOAT8C.md §1.** candle 0.11.0's
/// `WithDType for f8e4m3` generates `fn to_f64(self) -> f64 { (|v: f8e4m3|
/// v.to_f64())(self) }`, and the by-value trait method is the exact receiver
/// match, so the inner call resolves to itself; release-mode LLVM turns that
/// tail call into `.L1: jmp .L1`. Only **one** arm of candle's converter is
/// poisoned by it -- `(CpuStorage::F8E4M3, DType::F64)`. Every other conversion
/// out of `F8E4M3` (`U8`/`U32`/`I16`/`I32`/`I64`/`BF16`/`F16`/`F32`) is written
/// `v.to_f32()`, and `WithDType` declares no `to_f32`, so those resolve to the
/// `float8` crate's inherent method and terminate.
///
/// So the poisoned arm can be *routed around* rather than patched: go
/// `F8E4M3 -> F32 -> F64`. That is not a tolerance. `float8_e4m3fn` has 4
/// exponent bits and 3 mantissa bits and no value outside `f32`'s range, so
/// every one of its 256 bit patterns is exactly representable in `f32`, and
/// `f32 -> f64` is exact for all of them; the two-step result is bit-identical
/// to what a non-recursive `to_f64` would return. `tools/golden/compare.py`
/// already reads float8 results by this same widening (docs/FLOAT8B.md §5.1).
///
/// It is a free function rather than a method so that **every** `F64` widening
/// in this crate can be routed through one place: a `to_dtype(DType::F64)` left
/// behind anywhere is a hang, not a wrong answer, and hangs are what this dtype
/// costs the most to find.
pub(crate) fn widen_f64(t: &Tensor) -> candle_core::Result<Tensor> {
    // `reduced::to_dtype` owns the routing -- `fast_to` is the funnel
    // `aten._to_copy.default` already goes through, so putting it there means
    // `x.to(torch.float64)` is fixed by the same line as every internal
    // widening rather than by a second one that can drift from it.
    crate::reduced::to_dtype(t, candle_core::DType::F64)
}

/// The single entrance. `torch.ops.aten.<op>.<overload>(...)` is expected to
/// land here once the Python layer is vendored.
pub fn aten_dispatch(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    // One scan, two answers: that the tensor arguments agree about their
    // device, and *which* device that is. Making meta a second pass would have
    // paid for a scan twice at the hottest line in the crate; merged, meta
    // costs the dispatcher a discriminant test on a value the gate already had
    // in hand.
    // docs/FLOAT8B.md: `float8_e4m3fn` refuses here exactly where upstream
    // refuses, in upstream's own words. Before the device gate, because the
    // hanging ops in table B never come back from the kernel to be refused.
    float8_e4m3fn_gate(op, args, kwargs)?;
    let out = match check_devices_agree(op, args, kwargs)? {
        Some(Where::Meta) => meta_dispatch(py, op, args, kwargs)?,
        _ => aten_dispatch_inner(py, op, args, kwargs)?,
    };
    // One exit as well as one entrance: every tensor leaving the dispatcher
    // wears the registered Python tensor class (`tensor::promote`). Doing it
    // here rather than in each kernel means a kernel can keep returning the
    // native type and cannot forget.
    let out = crate::tensor::promote(py, out)?;
    // The graph capture hook (docs/CAPTURE.md). It is here rather than in
    // `aten_dispatch_inner` for two reasons: the meta path has to be recorded
    // too, and the identity the recorder registers has to be the object
    // *Python will hold*, which is `promote`'s result rather than the kernel's.
    //
    // When capture is off -- which is always, unless something asked for it --
    // this is one relaxed atomic load and a branch that is not taken. That is
    // the entire cost added to the single door, measured in docs/CAPTURE.md §7.
    // W5 (docs/BACKWARD4.md). Beside the capture hook rather than inside it,
    // and for the same three reasons: the meta path has to be answered too, the
    // identity that gets marked has to be the object *Python will hold* --
    // `promote`'s result -- and there is exactly one place where "an op
    // happened" is known.
    //
    // When nothing in the process requires a gradient this is one relaxed
    // atomic load, one table scan of twelve strings, and a walk of the argument
    // tuple that stops at the first tensor and finds a `false`.
    crate::tensor::mark_from_op(py, op, args, kwargs, &out);
    if crate::capture::is_active() {
        crate::capture::record(py, op, args, kwargs, &out);
    }
    Ok(out)
}

/// Where a dispatched tensor argument lives, comparable without allocating.
///
/// Not `PyDevice`: a label costs a `String` per argument and that was measured
/// at +78 ns per dispatch (docs/DEVICE_ABS.md §6). Not `candle_core::Device`
/// either, because **a meta tensor has no candle handle** -- that is the whole
/// content of `Repr::Meta`. So the gate compares this, which is a discriminant
/// test in both arms, and builds a label only on the path that is about to
/// raise.
#[derive(Clone)]
enum Where {
    Dense(Device),
    Meta,
}

impl Where {
    fn of(tensor: &PyTensorBase) -> Self {
        match tensor.repr() {
            crate::tensor::Repr::Dense(inner) => Where::Dense(inner.device().clone()),
            crate::tensor::Repr::Meta { .. } => Where::Meta,
            // A quantised tensor is on a real device, so it compares as one.
            // It will refuse a page later at `tensor()` for any dense kernel,
            // but it must not refuse *here* with a device message -- the
            // reason a Q4K weight cannot go through `aten.mm` is its
            // representation, not where it lives, and the door should not
            // mis-name that.
            crate::tensor::Repr::Quantized(q) => Where::Dense(q.device()),
        }
    }

    fn label(&self) -> PyDevice {
        match self {
            Where::Dense(device) => PyDevice::from_candle(device),
            Where::Meta => PyDevice::meta(),
        }
    }
}

/// Refuse an op whose tensor arguments are not all on one device.
///
/// **Where this lives is the design decision, not whether it exists.**
/// Upstream has no such gate: the check is inside each kernel, which is why the
/// message it gives depends on which kernel you hit -- `a + m` says *"Expected
/// all tensors to be on the same device, but found at least two devices, mps:0
/// and cpu!"*, `torch.mm` says *"Tensor for argument #1 'mat1' is on CPU, but
/// expected it to be on GPU"*, and `torch.cat([cpu, mps])` **segfaults** the
/// process on torch 2.13.0 (measured on this host, docs/DEVICE_ABS.md §6). A
/// per-kernel check is a check every new kernel has to remember; putting it at
/// the single door means no kernel can forget it, and this shim has a single
/// door precisely so that things like this have somewhere to go.
///
/// The message is upstream's most common one, verbatim, so that code matching
/// on it keeps working.
///
/// **The rejecting half of this gate is now reachable, and `meta` is what
/// reached it.** Until there was a second device it could not fire at all --
/// `resolve()` refuses every non-CPU label, so every tensor was on the CPU and
/// the loop always found one device. That was recorded as an untested branch in
/// docs/DEVICE_ABS.md §10. `meta` needs no backend, so `cpu + meta` is an input
/// this build can actually construct, and docs/META.md §5 is the measurement
/// that it raises.
///
/// **The comparison is on candle's handles, not on reconstructed labels, and
/// that is a measured decision.** Building a `PyDevice` per tensor argument
/// costs a `String` allocation each, and it showed: an interleaved A/B of two
/// artefacts differing only in the call above measured **+78 ns per dispatch**
/// for `add.Tensor` (346 -> 424 ns) and **+86 ns** for `cat.default`
/// (392 -> 479 ns) -- +22% on the cheapest op this door has. Comparing `Device`
/// directly is an enum discriminant test; a label is built only on the failing
/// path, where an allocation is free next to raising. docs/DEVICE_ABS.md §6 has
/// the numbers from both versions. `Where` keeps that property while making
/// room for a tensor with no handle at all.
///
/// Returns the device the arguments agreed on, or `None` when there were no
/// tensor arguments at all (every factory call). The dispatcher needs that
/// answer anyway to route meta, so returning it here is one scan instead of two.
fn check_devices_agree(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<Where>> {
    let mut first: Option<Where> = None;

    for value in args.iter() {
        scan_for_device(op, &mut first, &value)?;
    }
    // **The keyword loop descends into sequences too, and it did not used to.**
    // That was a hole rather than an economy, and it was invisible for exactly
    // as long as the gate could not fire: `_torch_level_function` binds every
    // argument by name before dispatching, so a `torch.cat([a, b])` arrives
    // here as `kwargs["tensors"] = [a, b]` and **never touches the positional
    // loop at all**. With one device the two loops could not be told apart.
    // The first `torch.cat([cpu_tensor, meta_tensor])` after meta landed went
    // straight past the gate and died inside the kernel on
    // `Cannot copy out of meta tensor; no data!` -- the right refusal by
    // accident, from the wrong place, and it would have been a wrong *answer*
    // for two backends that could both actually compute. docs/META.md §5.
    if let Some(kwargs) = kwargs {
        for (_, value) in kwargs.iter() {
            scan_for_device(op, &mut first, &value)?;
        }
    }
    Ok(first)
}

/// One dispatched argument, and the sequences one level under it.
///
/// Tensor first, sequence second, and the order is the measurement: almost
/// every argument that is anything is a tensor, so trying the list and tuple
/// casts first paid for two failed type checks on the hot path. `cat`/`stack`
/// take `Tensor[]`, so the sequence branch cannot be dropped -- it is exactly
/// the ops most likely to mix devices that hide their tensors one level down.
///
/// A free function rather than a closure, and that too is a measurement: the
/// first version of this was a closure calling another closure, both capturing
/// `first` mutably, and the inner call did not inline. Taking `first` as a
/// parameter costs nothing and lets both collapse into the caller.
#[inline]
fn scan_for_device(
    op: &str,
    first: &mut Option<Where>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if visit_for_device(op, first, value)? {
        return Ok(());
    }
    if let Ok(sequence) = value.cast::<PyList>() {
        for item in sequence.iter() {
            let is_tensor = visit_for_device(op, first, &item)?;
            if !is_tensor {
                // Not a tensor sequence
                break;
            }
        }
    } else if let Ok(sequence) = value.cast::<PyTuple>() {
        for item in sequence.iter() {
            let is_tensor = visit_for_device(op, first, &item)?;
            if !is_tensor {
                // Not a tensor sequence
                break;
            }
        }
    }
    Ok(())
}

/// One value. Returns whether it *was* a tensor, so the caller can skip the
/// sequence casts for the overwhelmingly common case.
///
/// The agreement test reads the representation in place instead of building a
/// `Where` per argument. Constructing one clones a `candle_core::Device`, and
/// the version that did so on every argument (rather than only on the first)
/// showed up in the A/B -- see docs/META.md §9. `Where` is built once, for the
/// first tensor, and again only on the path that is about to raise.
#[inline]
fn visit_for_device(
    op: &str,
    first: &mut Option<Where>,
    value: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let Ok(tensor) = value.cast::<PyTensorBase>() else {
        return Ok(false);
    };
    let borrowed = tensor.borrow();
    let agrees = match (&*first, borrowed.repr()) {
        (None, _) => {
            *first = Some(Where::of(&borrowed));
            return Ok(true);
        }
        (Some(Where::Dense(seen)), crate::tensor::Repr::Dense(inner)) => {
            seen.same_device(inner.device())
        }
        (Some(Where::Meta), crate::tensor::Repr::Meta { .. }) => true,
        // Same reasoning as `Where::of`: a quantised tensor's device is real,
        // so it agrees with a dense argument on the same device and the op
        // goes on to refuse for the right reason.
        (Some(Where::Dense(seen)), crate::tensor::Repr::Quantized(q)) => seen.same_device(&q.device()),
        _ => false,
    };
    if agrees {
        return Ok(true);
    }
    // `copy_` is the one op upstream lets cross devices, because transferring
    // *is* its definition (docs/DEVICE_ABS.md §3.4, measured: `cpu.copy_(mps)`
    // returns a cpu tensor). Checked only here, on the path that was about to
    // raise, so agreeing arguments pay nothing for it.
    if op == "aten.copy_.default" {
        return Ok(true);
    }
    let seen = first.as_ref().expect("a disagreement needs a first").label();
    Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
        "Expected all tensors to be on the same device, but found at least two \
         devices, {} and {}! ({op} in torch._C shim)",
        seen.__str__(),
        Where::of(&borrowed).label().__str__(),
    )))
}

/// The `meta` half of the dispatcher: ops whose inputs have no storage.
///
/// **Structured as a separate table rather than a branch inside each kernel,
/// because that is what upstream is.** torch has a `Meta` dispatch key with its
/// own registrations (`torch/_meta_registrations.py`); a meta kernel computes
/// shape and dtype and never touches bytes. Mirroring the shape here means the
/// dense kernels stay unaware of meta -- `PyTensorBase::tensor()` refuses them
/// if they ever try -- and it means the answer to "does this op work on meta?"
/// is a list rather than 96 separate readings.
///
/// **The list is short and honest about it.** Shape inference for the other
/// ninety-odd ops is a real body of work (upstream's file is thousands of
/// lines), it has no measured caller here yet, and guessing a shape rule is
/// exactly the kind of silent wrongness this shim refuses. So everything not
/// named below raises with its own name in the message, which is DESIGN.md §6's
/// instrument doing its job: run a model under `with torch.device("meta")` and
/// the work queue prints itself in frequency order.
///
/// `_aten_implemented()` is *not* extended by anything here. That constant
/// means "has a kernel and `tools/golden/cases.py` compares it against
/// upstream", and the golden harness compares values -- which a meta tensor by
/// definition has none of. Meta support is a property of ops already on the
/// list, so op coverage stays 96 and the evidence lives in
/// `pytests/test_shim.py` instead. docs/META.md §7.
fn meta_dispatch(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    match op {
        "aten._to_copy.default" => meta_to_copy(py, args, kwargs),
        "aten.copy_.default" => meta_copy_inplace(py, args, kwargs),
        // Metadata pass-throughs: upstream's kernels for these return a tensor
        // with the same shape and dtype and (for `detach`/`alias`) share
        // storage, which meta has none of. This shim's dense `detach`/`alias`
        // already copy rather than alias (docs/OPS4.md §8), so meta is not
        // losing an aliasing property it otherwise had.
        "aten.detach.default" | "aten.alias.default" | "aten.clone.default"
        | "aten.contiguous.default" | "aten.lift_fresh.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            meta_result(py, input.dims().to_vec(), input.tag())
        }
        // **In-place initialisers: no-ops that return the receiver.** These are
        // not a convenience -- `nn.Linear.reset_parameters` runs
        // `init.kaiming_uniform_(self.weight)` in every `__init__`, so
        // `with torch.device("meta"): nn.Linear(4, 8)` stops on `uniform_`
        // before it can produce a single parameter, which is the exact call
        // `accelerate.init_empty_weights` is built around.
        //
        // "Write nothing and return self" is upstream's meta kernel for them
        // too, and it is the only honest answer: filling a tensor that holds no
        // bytes has no observable effect, and the shape it advertises is
        // unchanged by what would have been written. The refusal that matters
        // -- reading the values back -- is still `tolist`/`item`'s, one layer
        // out, so nothing here lets a caller mistake an uninitialised meta
        // parameter for an initialised one.
        //
        // `add_`/`mul_` are deliberately *not* here even though they are also
        // in-place: they broadcast, so their meta kernel has a shape rule to
        // get right, and a no-op would silently accept a shape upstream
        // rejects.
        "aten.uniform_.default"
        | "aten.normal_.default"
        | "aten.zero_.default"
        | "aten.fill_.Scalar" => {
            let receiver = tensor_receiver(op, args, kwargs)?;
            Ok(receiver.into_any().unbind())
        }
        // `aten::is_floating_point` reads the dtype tag, which meta carries in
        // full. Nothing about it needs storage.
        "aten.is_floating_point.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            Ok(input.tag().is_floating_point().into_bound_py_any(py)?.unbind())
        }
        // `.item()`, `bool(t)`, `float(t)`. Upstream refuses this one with a
        // *different* message from the copy-out family, and the difference is
        // measured, not tidied: `torch.zeros(1, device="meta").item()` is
        // `RuntimeError: Tensor.item() cannot be called on meta tensors` while
        // `.tolist()` on the same tensor is `NotImplementedError: Cannot copy
        // out of meta tensor; no data!`.
        "aten._local_scalar_dense.default" => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Tensor.item() cannot be called on meta tensors",
        )),
        // `new_ones` takes its shape from the argument and its device from the
        // input tensor, so on a meta input it is a meta factory.
        "aten.new_ones.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            let size: Vec<usize> = required(op, args, kwargs, 1, "size")?.extract()?;
            let tag = dtype_arg(args, kwargs, 2, "dtype")?.unwrap_or(input.tag());
            match device_arg_or_label(args, kwargs, 4, "device", &PyDevice::meta())? {
                label if label.is_meta() => meta_result(py, size, tag),
                label => {
                    let device = label.resolve()?;
                    let storage = PyDtype::new(tag).storage(op)?;
                    let out = Tensor::ones(size, storage, &device)
                        .map_err(|e| candle_err(op, e))?;
                    finish(py, out, tag)
                }
            }
        }
        // `aten::empty_like` on a meta input -- the op that carries a model off
        // the meta device.
        //
        // `from_pretrained` builds the whole module tree under
        // `init_empty_weights`, so every parameter and buffer starts on meta.
        // Anything the checkpoint did not supply -- missing keys, and every
        // non-persistent buffer, which is never in a checkpoint by definition
        // -- is brought across by `torch.empty_like(param, device=...)` in
        // `transformers/modeling_utils.py:4763,4771`. Without this the load
        // reads the whole file and then stops one step before the model is
        // usable, which is exactly where docs/CKPT2.md §3 found it.
        //
        // Shape, dtype and device rules are the dense kernel's, restated with
        // the same helpers in the same argument order, for the reason
        // docs/E2E_REAL.md §6.1 gives: a meta kernel that promises a different
        // dtype than the dense one hands the caller an allocation the dense
        // kernel then refuses to compute into.
        //
        // Like the dense kernel, "empty" answers zeros. That is safe *here*
        // for a reason worth stating rather than assuming: every value this
        // produces is overwritten before it is read -- missing keys by
        // `_initialize_missing_keys`, non-persistent buffers by the module's
        // own initialisation -- and if one ever were not, the zeros would
        // reach the forward pass, where `pytests/test_shim.py` compares logits
        // against upstream.
        "aten.empty_like.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            let tag = dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(input.tag());
            reject_unsupported(
                op,
                args,
                kwargs,
                &[(2, "layout"), (4, "pin_memory"), (5, "memory_format")],
            )?;
            let shape = input.dims().to_vec();
            match device_arg_or_label(args, kwargs, 3, "device", &PyDevice::meta())? {
                label if label.is_meta() => meta_result(py, shape, tag),
                label => {
                    let device = label.resolve()?;
                    let storage = PyDtype::new(tag).storage(op)?;
                    let out =
                        Tensor::zeros(shape, storage, &device).map_err(|e| candle_err(op, e))?;
                    finish(py, out, tag)
                }
            }
        }
        // ---------------------------------------------------------------
        // The elementwise family. docs/META.md §7.1.
        //
        // Every arm below is the same two-part answer -- a shape rule and a
        // dtype rule -- and **neither part is restated here.** The shape is
        // `broadcast_shape`, which `masked_select` already needed and which
        // reproduces upstream's wording for a mismatch. The dtype is the
        // dense kernel's own helper, called rather than copied.
        //
        // That is not tidiness. docs/E2E_REAL.md §6.1: a meta kernel that
        // promises a dtype the dense kernel would not produce hands the
        // caller an allocation the dense kernel then refuses to compute into,
        // and the divergence surfaces far from here. Calling the same
        // function makes the two agree by construction, refusals included --
        // so where the dense kernel declines (`same_dtype` on a mixed pair,
        // `neg` on a bool, `pow` on a bool), the meta kernel declines with
        // the identical message rather than advertising a tensor that cannot
        // be built.
        // ---------------------------------------------------------------

        // **The comparisons: `bool` out, whatever went in.** The dtype half
        // is the one that cannot be guessed from the input, and it is why
        // these needed a kernel rather than a pass-through -- upstream's
        // `gt(float32_meta, 1.0)` is `torch.bool`, measured, as are all
        // twelve keys over `{float32, float16, int64, int32, bool}`.
        //
        // This is the family the user's `from_pretrained` report stopped on:
        // `_compute_llama3_parameters` opens with
        // `torch.where(wavelen > low_freq_wavelen, ...)` and `>` on a meta
        // tensor is `gt.Scalar` (transformers/modeling_rope_utils.py:655).
        "aten.eq.Scalar"
        | "aten.ne.Scalar"
        | "aten.lt.Scalar"
        | "aten.le.Scalar"
        | "aten.ge.Scalar"
        | "aten.gt.Scalar" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            // Read and dropped. The value cannot affect a shape or a dtype
            // -- `compare_scalar` uses it only to pick a comparison dtype
            // for the *computation*, which is not happening -- but reading
            // it keeps the missing-argument refusal identical to the dense
            // kernel's, which is the half of the contract a meta kernel can
            // still honour.
            scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
            meta_result(py, input.dims().to_vec(), TorchDType::Bool)
        }
        // The `Tensor` overloads add the broadcast and keep `same_dtype`.
        // Upstream promotes here and this shim does not (docs/BIND.md §9);
        // reproducing the *refusal* is what keeps meta from advertising a
        // comparison the dense kernel declines to run.
        //
        // Measured on 2.13.0, all on meta inputs: `(2,3) vs (2,3)` is
        // `(2,3)`, `(1,3) vs (2,1)` is `(2,3)`, and `(2,3)` against a 0-dim
        // is `(2,3)` -- the 0-dim case being the one a "shape is the left
        // operand's" shortcut gets right by accident and a "shape is the
        // condition's" one gets wrong.
        "aten.eq.Tensor"
        | "aten.ne.Tensor"
        | "aten.lt.Tensor"
        | "aten.le.Tensor"
        | "aten.ge.Tensor"
        | "aten.gt.Tensor" => {
            let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
            let rhs = tensor_arg(op, args, kwargs, 1, "other")?;
            // Promotes, and the result is `bool` whatever it promotes to.
            // The call is still made rather than skipped: it is what refuses
            // the pairs that have no promotion at all, so meta cannot
            // advertise a comparison the dense kernel declines to run.
            promote_operands(op, &lhs, &rhs)?;
            let shape = broadcast_shape(op, lhs.dims(), rhs.dims())?;
            meta_result(py, shape, TorchDType::Bool)
        }
        // **The arithmetic `Tensor` overloads.** `arith_tag` is the whole
        // dtype half and it is not "the input's": `div` on two `int64`
        // tensors is `float32` (true division floats an integral pair), and
        // `mul` is the one member that promotes its operands rather than
        // requiring them equal. Both facts are the dense kernel's, read off
        // the same two lines it uses.
        //
        // `alpha` is parsed and dropped for the same reason the scalar is
        // above: it scales values, and there are none, but a caller that
        // passes a bad one should still hear about it here.
        "aten.add.Tensor" | "aten.sub.Tensor" | "aten.mul.Tensor" | "aten.div.Tensor" => {
            let kind = match op {
                "aten.add.Tensor" => Arith::Add,
                "aten.sub.Tensor" => Arith::Sub,
                "aten.mul.Tensor" => Arith::Mul,
                _ => Arith::Div,
            };
            let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
            let rhs = tensor_arg(op, args, kwargs, 1, "other")?;
            // All four promote, and `sub` refuses a `bool` operand before it
            // promotes. Both halves are `arith_tensor`'s, restated in the
            // same order so meta cannot advertise a pairing the dense kernel
            // declines -- nor decline one it would have answered.
            if kind == Arith::Sub
                && (lhs.tag() == TorchDType::Bool || rhs.tag() == TorchDType::Bool)
            {
                arith_tag(op, kind, TorchDType::Bool, None)?;
            }
            let operand = promote_operands(op, &lhs, &rhs)?;
            let tag = arith_tag(op, kind, operand, None)?;
            alpha_arg(op, args, kwargs)?;
            let shape = broadcast_shape(op, lhs.dims(), rhs.dims())?;
            meta_result(py, shape, tag)
        }
        // `aten::rsub.Scalar` -- `other - alpha * self`. Reversed operands,
        // but the shape is still the tensor's and the dtype is still
        // `arith_tag`'s `Sub` row, exactly as the dense kernel computes it.
        "aten.rsub.Scalar" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            let other =
                scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
            let tag = arith_tag(op, Arith::Sub, input.tag(), Some(!other.is_int()))?;
            alpha_arg(op, args, kwargs)?;
            meta_result(py, input.dims().to_vec(), tag)
        }
        // **`where.self` broadcasts three operands and takes its dtype from
        // the two value operands, not the condition.** Both halves are the
        // ones that are easy to get wrong and both are measured:
        // `where(meta_bool(2,1), meta_f32(1,3), meta_f32(3))` is `(2,3)`
        // `float32` upstream, where a condition-shaped answer would be
        // `(2,1)`, and `where(bool_cond, f32, f32)` is `float32` and not
        // `bool`.
        //
        // The condition's *dtype* check is the dense kernel's, restated in
        // the sense that the same two accepted tags and the same message are
        // used -- `where` on a float condition raises upstream and the meta
        // path has to raise too, since the condition's dtype is one of the
        // few things a meta tensor does carry.
                "aten.where.default" | "aten.nonzero.default" => {
            let input = tensor_arg(op, args, kwargs, 0, if op == "aten.where.default" { "condition" } else { "self" })?;
            let assume = py
                .import("torch.fx.experimental._config")?
                .getattr("meta_nonzero_assume_all_nonzero")?
                .is_truthy()?;
            if !assume {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "The register_meta function for torch.nonzero() raises unimplemented by default, \
                     as a correct data-independent implementation does not exist. This implementation \
                     returns a fake value, assuming all elements of the tensor are non-zero. To enable \
                     this registration, please set 'torch.fx.experimental._config.meta_nonzero_assume_all_nonzero' to True."
                ));
            }
            
            let shape = input.dims();
            let mut numel = 1;
            for &s in shape {
                numel *= s;
            }
            
            let tag = TorchDType::Int64;
            
            if op == "aten.nonzero.default" {
                meta_result(py, vec![numel as usize, shape.len()], tag)
            } else {
                let mut out_tensors = Vec::with_capacity(shape.len());
                for _ in 0..shape.len() {
                    let out_t = meta_result(py, vec![numel as usize], tag.clone())?;
                    out_tensors.push(out_t.into_any());
                }
                let tuple = pyo3::types::PyTuple::new(py, out_tensors)?;
                Ok(tuple.into_any().unbind())
            }
        }
        "aten.where.self" => {
            let condition = tensor_arg(op, args, kwargs, 0, "condition")?;
            let lhs = tensor_arg(op, args, kwargs, 1, "self")?;
            let rhs = tensor_arg(op, args, kwargs, 2, "other")?;
            where_condition_check(&condition)?;
            // Promotes from the two value operands, as the dense kernel does.
            let tag = promote_operands(op, &lhs, &rhs)?;
            let shape = broadcast_shape(op, condition.dims(), lhs.dims())?;
            let shape = broadcast_shape(op, &shape, rhs.dims())?;
            meta_result(py, shape, tag)
        }
        // `where.ScalarOther` is the same three-way join with the third
        // operand a Python scalar, so it broadcasts two and takes its dtype
        // from `where_scalar_tag` -- the wrapped-number rule, where a `bool`
        // scalar leaves the tensor's dtype alone and an `int` one lifts only
        // a boolean tensor. `checked_convert` runs here for the reason §4.2
        // gives about the factories: a meta tensor is a claim about what the
        // real call would have produced, and a claim that skipped the real
        // call's range check is not a claim.
        "aten.where.ScalarSelf" => {
            let condition = tensor_arg(op, args, kwargs, 0, "condition")?;
            let raw = required(op, args, kwargs, 1, "self")?;
            let rhs = tensor_arg(op, args, kwargs, 2, "other")?;
            where_condition_check(&condition)?;
            if raw.is_instance_of::<PyTensorBase>() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "aten::where() Expected a value of type 'number' for argument 'self'                      but instead found type Tensor",
                ));
            }
            let scalar_is_bool = raw.is_instance_of::<pyo3::types::PyBool>();
            let scalar_is_int = scalar_is_bool || raw.is_instance_of::<pyo3::types::PyInt>();
            let tag = where_scalar_tag(rhs.tag(), scalar_is_bool, scalar_is_int);
            checked_convert(&raw, scalar_is_int, tag, 1)?;
            let shape = broadcast_shape(op, condition.dims(), rhs.dims())?;
            meta_result(py, shape, tag)
        }
        "aten.where.ScalarOther" => {
            let condition = tensor_arg(op, args, kwargs, 0, "condition")?;
            let lhs = tensor_arg(op, args, kwargs, 1, "self")?;
            let raw = required(op, args, kwargs, 2, "other")?;
            where_condition_check(&condition)?;
            if raw.is_instance_of::<PyTensorBase>() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "aten::where() Expected a value of type 'number' for argument 'other' \
                     but instead found type Tensor",
                ));
            }
            let scalar_is_bool = raw.is_instance_of::<pyo3::types::PyBool>();
            let scalar_is_int = scalar_is_bool || raw.is_instance_of::<pyo3::types::PyInt>();
            let tag = where_scalar_tag(lhs.tag(), scalar_is_bool, scalar_is_int);
            checked_convert(&raw, scalar_is_int, tag, 1)?;
            let shape = broadcast_shape(op, condition.dims(), lhs.dims())?;
            meta_result(py, shape, tag)
        }
        // **The `unary_float` family, shape-preserving, dtype by promotion.**
        // `unary_float_tag` is the rule and it is the dense family's own
        // function: a floating input keeps its exact dtype (`float16` in,
        // `float16` out -- *not* widened), anything else becomes the default
        // float, which moves with `set_default_dtype`.
        //
        // `rsqrt` is in this list even though its dense kernel is separate,
        // because that kernel now calls the same `unary_float_tag`.
        // `expm1` is here for the same reason.
        //
        // `silu`, `gelu` and `relu` are deliberately **not** here: their
        // dense kernels do not promote (upstream has no integral `silu`
        // kernel at all, measured in `unary_float`'s own comment), so they
        // would need a different rule and nothing has reached them on meta.
        "aten.cos.default"
        | "aten.sin.default"
        | "aten.erf.default"
        | "aten.log2.default"
        | "aten.tanh.default"
        | "aten.exp.default"
        | "aten.log.default"
        | "aten.expm1.default"
        | "aten.rsqrt.default"
        | "aten.sqrt.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            meta_result(py, input.dims().to_vec(), unary_float_tag(input.tag()))
        }
        // `aten::neg` keeps the input dtype rather than promoting, and it has
        // two refusals -- `bool` and the wide unsigned dtypes -- which
        // `neg_result_tag` carries so that both paths decline the same
        // inputs. A meta `neg` that accepted a bool would promise a tensor
        // the dense kernel refuses to compute.
        "aten.neg.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            let tag = neg_result_tag(input.tag())?;
            meta_result(py, input.dims().to_vec(), tag)
        }
        // `aten::bitwise_not` -- shape and dtype both the input's, with
        // upstream's floating-point refusal reproduced. `~mask` in
        // `_compute_llama3_parameters` is this op, and it is bool in, bool
        // out; the refusal is here because a float input raises upstream and
        // the dtype tag is enough to see it.
        "aten.bitwise_not.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            let tag = input.tag();
            if tag.is_floating_point() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "\"bitwise_not_cpu\" not implemented for '{}'",
                    scalar_type_name(tag)
                )));
            }
            meta_result(py, input.dims().to_vec(), tag)
        }
        // `aten::clamp` -- shape is the input's, dtype is the ladder in
        // `clamp_result_tag`, including "both bounds absent is an error".
        // That table is the one the golden cases had to correct once
        // (out-of-place promotes where in-place refuses), which is the
        // argument for calling it rather than writing a second copy.
        "aten.clamp.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            let min = scalar_arg(op, args, kwargs, 1, "min")?;
            let max = scalar_arg(op, args, kwargs, 2, "max")?;
            let tag = clamp_result_tag(op, args, kwargs, input.tag(), min, max)?;
            meta_result(py, input.dims().to_vec(), tag)
        }
        // `aten::clamp_min` shares that ladder -- see `clamp_min_default` for
        // the ten rows that established it is the same one -- and differs only
        // in that `min` is required, so the "both absent" branch is
        // unreachable from here.
        "aten.clamp_min.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            let min = scalar_arg(op, args, kwargs, 1, "min")?.ok_or_else(|| missing(op, "min"))?;
            let tag = clamp_result_tag(op, args, kwargs, input.tag(), Some(min), None)?;
            meta_result(py, input.dims().to_vec(), tag)
        }
        // `aten::pow.Tensor_Scalar` and `.Tensor_Tensor`. The dtype is
        // `pow_result_tag`'s wrapped-number rule -- an integer tensor with an
        // integer exponent stays integral (`pow(int64, 2)` is `int64`,
        // measured on meta), a float on either side floats the result -- and
        // for the two-tensor overload the operands promote first, which is
        // why `pow_tensor_tensor` hands in the *promotion* and not an
        // operand's own dtype.
        "aten.pow.Tensor_Scalar" => {
            let base = tensor_arg(op, args, kwargs, 0, "self")?;
            let exponent = scalar_arg(op, args, kwargs, 1, "exponent")?
                .ok_or_else(|| missing(op, "exponent"))?;
            let tag = pow_result_tag(op, base.tag(), !exponent.is_int())?;
            meta_result(py, base.dims().to_vec(), tag)
        }
        "aten.pow.Tensor_Tensor" => {
            let base = tensor_arg(op, args, kwargs, 0, "self")?;
            let exponent = tensor_arg(op, args, kwargs, 1, "exponent")?;
            let operand = promote_operands(op, &base, &exponent)?;
            let tag = pow_result_tag(op, operand, false)?;
            let shape = broadcast_shape(op, base.dims(), exponent.dims())?;
            meta_result(py, shape, tag)
        }
        // ---------------------------------------------------------------
        // Two shape kernels, and the reason they are the only two.
        //
        // Three shape kernels, and the reason they are the only three.
        //
        // The elementwise block above got `llama` and thirteen others through
        // construction on meta. Sweeping all twenty (docs/META.md §7.2)
        // printed a work queue with exactly two entries behind it --
        // `select.int` for five architectures and `tril.default` for one --
        // and re-running it behind those printed one more, `expand.default`
        // for `bert`. Each of the three is the queue's answer and not a guess
        // at what might be wanted next; ARCH20.md §0.2's "a wall is not one
        // wall" is why the sweep was re-run after each rather than after all.
        // Everything past them is still §7's boundary.
        // ---------------------------------------------------------------

        // `aten::select.int(Tensor self, int dim, SymInt index)` -- the
        // dimension is **removed**, which is what separates it from `slice`.
        //
        // `gemma`, `opt`, `olmo`, `bert` and `cohere` all reach it while
        // *constructing* on meta: their rotary/positional buffers are built
        // with an `x[0]`-shaped indexing step, and `__getitem__` with an
        // integer is this op.
        //
        // Every check the dense kernel runs is run here, and all of them
        // depend only on metadata: the 0-dim refusal, `normalise_dim`'s
        // wrapping and range check, and `normalise_index`'s. Skipping them
        // would make the meta path *accept* an index the dense path raises
        // on, which §4.2's argument about the factories rules out -- a meta
        // tensor is a claim about what the real call would have produced.
        "aten.select.int" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            let dims = input.dims().to_vec();
            if dims.is_empty() {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "invalid index of a 0-dim tensor",
                ));
            }
            let dim = normalise_dim(op, dim_arg(args, kwargs, 1, "dim")?.unwrap_or(0), dims.len())?;
            normalise_index(
                op,
                int_arg(args, kwargs, 2, "index")?.ok_or_else(|| missing(op, "index"))? as isize,
                dims[dim],
            )?;
            let mut shape = dims;
            shape.remove(dim);
            meta_result(py, shape, input.tag())
        }
        // `aten::tril` / `aten::triu` -- shape and dtype both unchanged; the
        // whole op is *which values are zeroed*, and a meta tensor has none.
        // So this is the one place in the table where "same shape, same
        // dtype" is the complete kernel rather than a shortcut, and the only
        // thing left to reproduce is the rank refusal, which reads the shape.
        //
        // `gpt_bigcode` is the caller: its causal-mask buffer is
        // `torch.tril(torch.ones((n, n), dtype=torch.bool))` built in
        // `__init__` (docs/TORCHSCRIPT.md §6), so it runs during
        // construction and lands on meta.
        "aten.tril.default" | "aten.triu.default" => {
            let name = if op == "aten.tril.default" { "tril" } else { "triu" };
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            int_arg(args, kwargs, 1, "diagonal")?;
            let dims = input.dims().to_vec();
            if dims.len() < 2 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "{name}: input tensor must have at least 2 dimensions"
                )));
            }
            meta_result(py, dims, input.tag())
        }
        // `aten::expand(Tensor self, SymInt[] size, *, bool implicit=False)`
        // -- `bert`'s remaining wall, reached while building its position-id
        // buffer during construction.
        //
        // The rank check and the `-1` sentinel are `expand_target`, shared
        // with the dense kernel. The **extent** check is the one thing not
        // shared, and the reason is structural rather than a choice: the
        // dense kernel gets it for free from `broadcast_as`, and there is no
        // candle handle here to hand to it. So it is written out, with
        // upstream's own wording, measured on 2.13.0 for both `cpu` and
        // `meta` (they agree, unlike the three ops in §7.3):
        //
        //     expand(zeros(3), [2, 4])
        //       RuntimeError: The expanded size of the tensor (4) must match
        //       the existing size (3) at non-singleton dimension 1.
        //       Target sizes: [2, 4].  Tensor sizes: [3]
        //
        // A zero extent is *not* singleton for this rule -- `expand(zeros(0,
        // 3), [2, 3])` raises upstream, naming dimension 0 -- which is the
        // case a `!= 1` written as `<= 1` would silently accept.
        "aten.expand.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            let requested = shape_arg(op, args, kwargs, 1, "size")?;
            let dims = input.dims().to_vec();
            let target = expand_target(op, &dims, &requested)?;
            let offset = target.len() - dims.len();
            for (i, &want) in target.iter().enumerate().skip(offset) {
                let have = dims[i - offset];
                if have != want && have != 1 {
                    // `requested` and not `target`: upstream prints the sizes
                    // as they were *asked for*, `-1` sentinels included --
                    // `expand(zeros(2,1,3), [2,4,3,-1])` reports
                    // `Target sizes: [2, 4, 3, -1]`, measured. Printing the
                    // resolved list instead would name a size the caller
                    // never wrote.
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "The expanded size of the tensor ({want}) must match the existing \
                         size ({have}) at non-singleton dimension {i}.  Target sizes: \
                         {requested:?}.  Tensor sizes: {dims:?}"
                    )));
                }
            }
            meta_result(py, target, input.tag())
        }
        // `aten::div.Scalar` and `aten::mul.Scalar` -- shape is the input's,
        // dtype is `arith_tag`'s.
        //
        // Both are reached by `LlamaRotaryEmbedding.__init__`
        // (`transformers/models/llama/modeling_llama.py:108`), which
        // `from_pretrained` runs under `init_empty_weights`: the `/ dim`
        // directly, and the `*` that `torch/_tensor.py:1112` turns the leading
        // `1.0 /` into. There is no broadcasting to get right, a `Scalar`
        // overload having only one tensor, so the whole kernel is the
        // promotion -- and the promotion is `arith_tag`'s rather than a
        // restatement of it, so the dtype this advertises is by construction
        // the dtype the dense kernel would produce, refusals included.
        //
        // `add.Scalar` and `sub.Scalar` join them now that the `Tensor`
        // overloads above are here. Leaving two of the four members of one
        // helper out was defensible while the meta table was four ops wide
        // and is not now: the rule is the same call, and the asymmetry would
        // only show up as `x + 1` refusing on a tensor where `x * 1` works.
        // They carry `alpha`, which `alpha_arg` parses and this drops --
        // see the `Tensor` arm.
        "aten.div.Scalar" | "aten.mul.Scalar" | "aten.add.Scalar" | "aten.sub.Scalar" => {
            let kind = match op {
                "aten.div.Scalar" => Arith::Div,
                "aten.mul.Scalar" => Arith::Mul,
                "aten.add.Scalar" => Arith::Add,
                _ => Arith::Sub,
            };
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            let other =
                scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
            let tag = arith_tag(op, kind, input.tag(), Some(!other.is_int()))?;
            alpha_arg(op, args, kwargs)?;
            meta_result(py, input.dims().to_vec(), tag)
        }
        // `aten::pow.Scalar(Scalar self, Tensor exponent)` -- the next link in
        // the same expression. The scalar is the *base* here, so the shape
        // comes from argument 1, and the dtype is torch's wrapped-number rule
        // (`pow_result_tag`): an integer base leaves an integral exponent
        // integral, a float base floats it.
        "aten.pow.Scalar" => {
            let base = scalar_arg(op, args, kwargs, 0, "self")?.ok_or_else(|| missing(op, "self"))?;
            let exponent = tensor_arg(op, args, kwargs, 1, "exponent")?;
            let tag = pow_result_tag(op, exponent.tag(), !base.is_int())?;
            meta_result(py, exponent.dims().to_vec(), tag)
        }
        // `aten::reciprocal` -- the last link. `torch/_tensor.py:1112` spells
        // `1.0 / t` as `t.reciprocal() * 1.0`, so this is what the rope
        // expression reaches rather than an `rdiv` op.
        //
        // Its dense counterpart is the `unary_float` family, and it now calls
        // that family's `unary_float_tag` rather than restating the rule. The
        // other members are in the block above.
        "aten.reciprocal.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            meta_result(py, input.dims().to_vec(), unary_float_tag(input.tag()))
        }
        other => Err(not_implemented(format!(
            "torch._C shim has no meta kernel for {other}. A meta tensor holds shape and \
             dtype and no storage, so this op would have to infer its output shape without \
             computing -- which is a real kernel (upstream registers one in \
             torch/_meta_registrations.py), not a fallthrough. See docs/META.md §7 for the \
             list that is implemented."
        ))),
    }
}

/// A finished meta tensor: shape and dtype, no allocation.
///
/// The meta counterpart of `finish`, and deliberately not folded into it. The
/// two differ in exactly one interesting way -- `finish` has to ask
/// `PyDtype::storage()` whether candle can hold the dtype, and this does not,
/// because nothing is being held.
fn meta_result(py: Python<'_>, shape: Vec<usize>, tag: TorchDType) -> PyResult<Py<PyAny>> {
    Ok(PyTensorBase::meta(shape, tag)
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

/// `aten::_to_copy` with a meta input: the dtype half works, the device half
/// only outwards-to-nowhere.
///
/// Three cases, all measured on torch 2.13.0:
///
/// | call | upstream |
/// |---|---|
/// | `meta.to(torch.float64)` | meta tensor, float64 |
/// | `meta.to("meta")` | the same object (`t.to("meta") is t`) |
/// | `meta.to("cpu")` / `meta.cpu()` | `NotImplementedError: Cannot copy out of meta tensor; no data!` |
///
/// The last one is the whole point of meta and the reason this is not a
/// fallthrough: a shim that quietly produced a zero-filled CPU tensor here
/// would turn "no weights were loaded" into "the weights are all zero", which
/// is the failure `docs/CKPT.md`'s `filled` guard exists to prevent one layer
/// down.
fn meta_to_copy(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._to_copy.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(input.tag());
    reject_unsupported(OP, args, kwargs, &[(2, "layout"), (4, "pin_memory")])?;
    reject_memory_format(OP, args, kwargs, 6)?;
    // `device=None` keeps the tensor where it is -- on meta. Same contract as
    // the dense path (docs/DEVICE_ABS.md §5.2), and here it is observable:
    // reading the absent argument as "the CPU" would make every `.float()` on
    // a meta parameter try to leave the meta device.
    let label = device_arg_or_label(args, kwargs, 3, "device", &PyDevice::meta())?;
    if !label.is_meta() {
        return Err(crate::tensor::no_data());
    }
    meta_result(py, input.dims().to_vec(), tag)
}

/// `aten::copy_` where one side is meta.
///
/// The gate above lets this op through with disagreeing devices because
/// transferring is its definition. What the two directions mean is not
/// symmetric, and both halves are measured:
///
/// * `meta.copy_(cpu)` **succeeds and does nothing.** The receiver stays meta.
///   `nn.Module.load_state_dict` without `assign=True` lands exactly here and
///   upstream warns about it in as many words -- *"copying from a non-meta
///   parameter in the checkpoint to a meta parameter in the current model,
///   which is a no-op"*.
/// * `cpu.copy_(meta)` **refuses**, with the copy-out message, because there is
///   nothing to read.
///
/// A shim that made the first case allocate would silently turn
/// `load_state_dict` into something that appeared to work; a shim that refused
/// it would stop a path upstream completes with a warning.
fn meta_copy_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.copy_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let source = tensor_arg(OP, args, kwargs, 1, "src")?;
    if !receiver.borrow().is_meta_repr() {
        // Receiver is dense, source is meta: reading the source is the copy.
        return Err(crate::tensor::no_data());
    }
    let _ = source;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

fn aten_dispatch_inner(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    match op {
        "aten.add.Tensor" => add_tensor(py, args, kwargs),
        "aten.addmm.default" => addmm_default(py, args, kwargs),
        "aten.alias.default" => alias_default(py, args, kwargs),
        "aten.arange.default" => arange(py, args, kwargs, ArangeForm::End),
        "aten.arange.start" => arange(py, args, kwargs, ArangeForm::Start),
        "aten.arange.start_step" => arange(py, args, kwargs, ArangeForm::StartStep),
        "aten.amax.default" => amax_default(py, args, kwargs),
        "aten.argmax.default" => argmax_default(py, args, kwargs),
        "aten._grouped_mm.default" => grouped_mm_default(py, args, kwargs),
        "aten.bmm.default" => bmm_default(py, args, kwargs),
        "aten.cat.default" => cat_default(py, args, kwargs),
        "aten.stack.default" => stack_default(py, args, kwargs),
        "aten.scalar_tensor.default" => scalar_tensor_default(py, args, kwargs),
        "aten.embedding.default" => embedding_default(py, args, kwargs),
        "aten.empty.memory_format" => empty_memory_format(py, args, kwargs),
        "aten.full.default" => full_default(py, args, kwargs),
        "aten.full_like.default" => full_like_default(py, args, kwargs),
        "aten.is_floating_point.default" => is_floating_point_default(py, args, kwargs),
        "aten.isin.Tensor_Tensor" => isin_tensor_tensor(py, args, kwargs),
        "aten.lift_fresh.default" => lift_fresh_default(py, args, kwargs),
        "aten.mm.default" => mm_default(py, args, kwargs),
        "aten.ones.default" => ones_default(py, args, kwargs),
        "aten.pow.Scalar" => pow_scalar(py, args, kwargs),
        "aten.pow.Tensor_Scalar" => pow_tensor_scalar(py, args, kwargs),
        "aten.pow.Tensor_Tensor" => pow_tensor_tensor(py, args, kwargs),
        "aten.randint.default" => randint(py, args, kwargs, false),
        "aten.randint.low" => randint(py, args, kwargs, true),
        "aten.randperm.default" => randperm(py, args, kwargs),
        "aten.remainder.Scalar" => {
            remainder_op(py, args, kwargs, "aten.remainder.Scalar", true)
        }
        "aten.remainder.Tensor" => {
            remainder_op(py, args, kwargs, "aten.remainder.Tensor", false)
        }
        "aten.repeat.default" => repeat_default(py, args, kwargs),
        "aten.rsqrt.default" => rsqrt_default(py, args, kwargs),
        // `sqrt` sits with the `unary_float` family rather than beside
        // `rsqrt`'s own kernel: it is one candle call, and sharing the family
        // is what makes `float16` in / `float16` out true for both without
        // restating the rule. docs/KERNELS26.md §1.
        "aten.sqrt.default" => unary_float(py, args, kwargs, "aten.sqrt.default", Unary::Sqrt),
        "aten.rsub.Scalar" => rsub_scalar(py, args, kwargs),

        // -- what upstream's `repr(tensor)` dispatches (docs/E2E_REAL.md) ----
        "aten.abs.default" => abs_default(py, args, kwargs),
        "aten.adaptive_avg_pool1d.default" => adaptive_avg_pool1d_default(py, args, kwargs),
        "aten.greater.Tensor" => compare_tensor(py, args, kwargs, "aten.greater.Tensor", Cmp::Gt),
        "aten.greater.Scalar" => compare_scalar(py, args, kwargs, "aten.greater.Scalar", Cmp::Gt),
        "aten.adaptive_avg_pool2d.default" => adaptive_avg_pool2d_default(py, args, kwargs),
        "aten.roll.default" => roll_default(py, args, kwargs),
        "aten.where.ScalarSelf" => where_scalar_self(py, args, kwargs),
        "aten.ceil.default" => ceil_default(py, args, kwargs),
        "aten.gt.Tensor" => compare_tensor(py, args, kwargs, "aten.gt.Tensor", Cmp::Gt),
        "aten.gt.Scalar" => compare_scalar(py, args, kwargs, "aten.gt.Scalar", Cmp::Gt),
        "aten.masked_select.default" => masked_select_default(py, args, kwargs),
        "aten.unbind.int" => unbind_int(py, args, kwargs),

        // -- attention (docs/OPS8.md) --------------------------------------
        "aten._scaled_dot_product_flash_attention_for_cpu.default" => {
            sdpa_flash_cpu(py, args, kwargs)
        }

        // -- the four docs/GPT2.md measured a 2-layer GPT-2 stopping on -----
        "aten.native_group_norm.default" => native_group_norm_default(py, args, kwargs),
        "aten.upsample_bilinear2d.default" => upsample_bilinear2d_default(py, args, kwargs),
        "aten.avg_pool2d.default" => avg_pool2d_default(py, args, kwargs),
        "aten.max_pool2d.default" => max_pool2d_default(py, args, kwargs),
        "aten.native_layer_norm.default" => native_layer_norm_default(py, args, kwargs),
        "aten.native_batch_norm.default" => native_batch_norm_default(py, args, kwargs),

        // -- the TensorBase surface (docs/TENSORBASE.md) -------------------
        "aten.add.Scalar" => arith_scalar(py, args, kwargs, "aten.add.Scalar", Arith::Add),
        "aten.sub.Tensor" => arith_tensor(py, args, kwargs, "aten.sub.Tensor", Arith::Sub),
        "aten.sub.Scalar" => arith_scalar(py, args, kwargs, "aten.sub.Scalar", Arith::Sub),
        "aten.mul.Tensor" => arith_tensor(py, args, kwargs, "aten.mul.Tensor", Arith::Mul),
        "aten.mul.Scalar" => arith_scalar(py, args, kwargs, "aten.mul.Scalar", Arith::Mul),
        "aten.div.Tensor" => arith_tensor(py, args, kwargs, "aten.div.Tensor", Arith::Div),
        "aten.div.Scalar" => arith_scalar(py, args, kwargs, "aten.div.Scalar", Arith::Div),
        "aten.norm.ScalarOpt_dim" => norm_scalaropt_dim(py, args, kwargs),
        "aten.linalg_vector_norm.default" => linalg_vector_norm_default(py, args, kwargs),
        "aten._weight_norm_interface.default" => weight_norm_interface_default(py, args, kwargs),
        "aten.div.Tensor_mode" => div_mode(py, args, kwargs, "aten.div.Tensor_mode", false),
        "aten.div.Scalar_mode" => div_mode(py, args, kwargs, "aten.div.Scalar_mode", true),
        "aten.matmul.default" => matmul_default(py, args, kwargs),

        "aten.eq.Tensor" => compare_tensor(py, args, kwargs, "aten.eq.Tensor", Cmp::Eq),
        "aten.eq.Scalar" => compare_scalar(py, args, kwargs, "aten.eq.Scalar", Cmp::Eq),
        "aten.ne.Tensor" => compare_tensor(py, args, kwargs, "aten.ne.Tensor", Cmp::Ne),
        "aten.ne.Scalar" => compare_scalar(py, args, kwargs, "aten.ne.Scalar", Cmp::Ne),
        "aten.lt.Tensor" => compare_tensor(py, args, kwargs, "aten.lt.Tensor", Cmp::Lt),
        "aten.lt.Scalar" => compare_scalar(py, args, kwargs, "aten.lt.Scalar", Cmp::Lt),
        "aten.le.Scalar" => compare_scalar(py, args, kwargs, "aten.le.Scalar", Cmp::Le),
        "aten.le.Tensor" => compare_tensor(py, args, kwargs, "aten.le.Tensor", Cmp::Le),

        "aten.bitwise_and.Tensor" => bitwise_binary(py, args, kwargs, "aten.bitwise_and.Tensor", Bitwise::And),
        "aten.bitwise_or.Tensor" => bitwise_binary(py, args, kwargs, "aten.bitwise_or.Tensor", Bitwise::Or),
        "aten.bitwise_and.Scalar" => bitwise_scalar(py, args, kwargs, "aten.bitwise_and.Scalar", Bitwise::And),
        "aten.bitwise_or.Scalar" => bitwise_scalar(py, args, kwargs, "aten.bitwise_or.Scalar", Bitwise::Or),
        "aten.bitwise_not.default" => bitwise_not_default(py, args, kwargs),

        "aten.cos.default" => unary_float(py, args, kwargs, "aten.cos.default", Unary::Cos),
        "aten.sin.default" => unary_float(py, args, kwargs, "aten.sin.default", Unary::Sin),
        "aten.reciprocal.default" => {
            unary_float(py, args, kwargs, "aten.reciprocal.default", Unary::Reciprocal)
        }
        "aten.tanh.default" => unary_float(py, args, kwargs, "aten.tanh.default", Unary::Tanh),
        "aten.neg.default" => neg_default(py, args, kwargs),
        "aten.sigmoid.default" => sigmoid_default(py, args, kwargs),
        "aten.sign.default" => sign_default(py, args, kwargs),
        "aten.silu.default" => silu_default(py, args, kwargs),
        "aten.relu.default" => relu_default(py, args, kwargs),
        "aten.relu_.default" => relu_inplace(py, args, kwargs),

        "aten.sum.default" => sum_or_mean(py, args, kwargs, "aten.sum.default", Reduce::Sum, false),
        "aten.sum.dim_IntList" => {
            sum_or_mean(py, args, kwargs, "aten.sum.dim_IntList", Reduce::Sum, true)
        }
        "aten.mean.default" => sum_or_mean(py, args, kwargs, "aten.mean.default", Reduce::Mean, false),
        "aten.mean.dim" => sum_or_mean(py, args, kwargs, "aten.mean.dim", Reduce::Mean, true),
        "aten.cumsum.default" => cumsum_default(py, args, kwargs),
        "aten.max.default" => extremum_default(py, args, kwargs, Extremum::Max),
        "aten.min.default" => extremum_default(py, args, kwargs, Extremum::Min),
        "aten.max.dim" => extremum_dim(py, args, kwargs, Extremum::Max),
        "aten.min.dim" => extremum_dim(py, args, kwargs, Extremum::Min),
        "aten.max.other" => extremum_other(py, args, kwargs, Extremum::Max),
        "aten.min.other" => extremum_other(py, args, kwargs, Extremum::Min),
        "aten.tril.default" => tril_triu(py, args, kwargs, Triangle::Lower),
        "aten.triu.default" => tril_triu(py, args, kwargs, Triangle::Upper),
        "aten.any.default" => {
            any_or_all_default(py, args, kwargs, "aten.any.default", BoolReduce::Any)
        }
        "aten.any.dim" => {
            any_or_all_dim(py, args, kwargs, "aten.any.dim", false, BoolReduce::Any)
        }
        "aten.any.dims" => {
            any_or_all_dim(py, args, kwargs, "aten.any.dims", true, BoolReduce::Any)
        }
        // `sam3_video`'s wall: `masking_utils.py:330` asks `padding_mask.all()`
        // before it will skip building a bidirectional mask. The three forms
        // are `any`'s three, sharing every line except the reduction and the
        // empty-input identity -- see `BoolReduce`.
        "aten.all.default" => {
            any_or_all_default(py, args, kwargs, "aten.all.default", BoolReduce::All)
        }
        "aten.all.dim" => {
            any_or_all_dim(py, args, kwargs, "aten.all.dim", false, BoolReduce::All)
        }
        "aten.all.dims" => {
            any_or_all_dim(py, args, kwargs, "aten.all.dims", true, BoolReduce::All)
        }

        "aten.masked_fill.Scalar" => masked_fill(py, args, kwargs, "aten.masked_fill.Scalar"),
        "aten.masked_fill.Tensor" => masked_fill(py, args, kwargs, "aten.masked_fill.Tensor"),
        "aten.nonzero.default" => nonzero_default(py, args, kwargs),
        "aten.where.default" => where_default(py, args, kwargs),
        "aten.where.self" => where_self(py, args, kwargs),
        "aten.where.ScalarOther" => where_scalar_other(py, args, kwargs),

        "aten.expand.default" => expand_default(py, args, kwargs),
        "aten.reshape.default" => reshape_like(py, args, kwargs, "aten.reshape.default", "shape"),
        "aten.view.default" => reshape_like(py, args, kwargs, "aten.view.default", "size"),
        "aten.view.dtype" => view_dtype(py, args, kwargs),
        // Upstream's `_unsafe_view` differs from `view` only in what it
        // promises the autograd engine about aliasing -- the value is
        // `view`'s. There is no autograd here, so it is the same kernel, and
        // the key stays distinct because `reshape()`'s non-contiguous path
        // emits this one and not `view`.
        "aten._unsafe_view.default" => {
            reshape_like(py, args, kwargs, "aten._unsafe_view.default", "size")
        }
        "aten.transpose.int" => transpose_int(py, args, kwargs),
        "aten.permute.default" => permute_default(py, args, kwargs),
        "aten.t.default" => t_default(py, args, kwargs),
        "aten.unsqueeze.default" => unsqueeze_default(py, args, kwargs),
        "aten.squeeze.default" => squeeze_default(py, args, kwargs),
        "aten.squeeze.dim" => squeeze_dim(py, args, kwargs),
        "aten.squeeze.dims" => squeeze_dims(py, args, kwargs),
        "aten.split.Tensor" => split_tensor(py, args, kwargs),
        "aten.contiguous.default" => contiguous_default(py, args, kwargs),
        "aten.clone.default" => clone_default(py, args, kwargs),
        "aten.detach.default" => detach_default(py, args, kwargs),
        "aten._to_copy.default" => to_copy_default(py, args, kwargs),
        "aten.new_ones.default" => new_ones_default(py, args, kwargs),
        "aten.new_zeros.default" => new_zeros_default(py, args, kwargs),
        "aten.zeros.default" => zeros_or_ones(py, args, kwargs, "aten.zeros.default", false),
        "aten._local_scalar_dense.default" => local_scalar_dense(py, args, kwargs),

        "aten.select.int" => select_int(py, args, kwargs),
        "aten.slice.Tensor" => slice_tensor(py, args, kwargs),
        "aten.index.Tensor" => index_tensor(py, args, kwargs),

        "aten.fill_.Scalar" => fill_inplace(py, args, kwargs, "aten.fill_.Scalar"),
        "aten.fill_.Tensor" => fill_inplace(py, args, kwargs, "aten.fill_.Tensor"),
        "aten.zero_.default" => zero_inplace(py, args, kwargs),
        "aten.copy_.default" => copy_inplace(py, args, kwargs),
        "aten.uniform_.default" => uniform_inplace(py, args, kwargs),
        "aten.normal_.default" => normal_inplace(py, args, kwargs),

        // -- what widening past the Llama/GPT-2 family asks for (docs/ARCH.md) --
        "aten.gelu.default" => gelu_default(py, args, kwargs),
        "aten.gather.default" => gather_default(py, args, kwargs),
        "aten.hardtanh.default" => hardtanh_default(py, args, kwargs),

        // -- the eight `do_sample=True` stops on (docs/SAMPLING.md) ---------
        "aten._softmax.default" => softmax_default(py, args, kwargs),
        "aten.scatter.src" => scatter_src(py, args, kwargs),
        "aten.sort.default" => sort_default(py, args, kwargs),
        "aten.topk.default" => topk_default(py, args, kwargs),
        "aten.multinomial.default" => multinomial_default(py, args, kwargs),

        // -- the cross-entropy forward (docs/LOSS.md) ----------------------
        "aten._log_softmax.default" => log_softmax_default(py, args, kwargs),
        "aten.nll_loss_forward.default" => nll_loss_forward_default(py, args, kwargs),

        // -- the out-of-place dropout capture can record (docs/LOSS.md §7) --
        "aten.native_dropout.default" => native_dropout_default(py, args, kwargs),

        // -- falcon / bloom / gpt_bigcode (docs/TAIL.md) --------------------
        "aten._safe_softmax.default" => safe_softmax_default(py, args, kwargs),
        "aten.add_.Tensor" => {
            arith_inplace_tensor(py, args, kwargs, "aten.add_.Tensor", Arith::Add)
        }
        "aten.baddbmm.default" => baddbmm_default(py, args, kwargs),
        "aten.split_with_sizes.default" => split_with_sizes(py, args, kwargs),

        // -- mamba / mixtral (docs/OPS4.md) ---------------------------------
        "aten.erf.default" => unary_float(py, args, kwargs, "aten.erf.default", Unary::Erf),
        "aten.exp.default" => unary_float(py, args, kwargs, "aten.exp.default", Unary::Exp),
        // `mamba`'s *construction* wall, not its forward (docs/ARCH20.md §4).
        "aten.log.default" => unary_float(py, args, kwargs, "aten.log.default", Unary::Log),
        "aten.log2.default" => log2_default(py, args, kwargs),
        "aten.leaky_relu.default" => leaky_relu_default(py, args, kwargs),
        "aten.linspace.default" => linspace_default(py, args, kwargs),
        "aten.expm1.default" => expm1_default(py, args, kwargs),
        // `bert`'s wall: `F.pad` on a bias while the model is being built.
        "aten.constant_pad_nd.default" => constant_pad_nd(py, args, kwargs),
        "aten.softplus.default" => softplus_default(py, args, kwargs),
        "aten.convolution.default" => convolution_default(py, args, kwargs),
        "aten.zeros_like.default" => zeros_or_empty_like(py, args, kwargs, "aten.zeros_like.default"),
        "aten.ones_like.default" => zeros_or_empty_like(py, args, kwargs, "aten.ones_like.default"),
        "aten.empty_like.default" => zeros_or_empty_like(py, args, kwargs, "aten.empty_like.default"),
        "aten.ge.Scalar" => compare_scalar(py, args, kwargs, "aten.ge.Scalar", Cmp::Ge),
        // The last of the six comparisons to get its Tensor overload. `le`,
        // `lt` and `gt` all had both halves and `ge` had only `.Scalar`, so
        // `x >= tensor` resolved through `methods.json` and then refused by
        // name (docs/GROUPED_MM.md §6.4). Same kernel as its five siblings.
        "aten.ge.Tensor" => compare_tensor(py, args, kwargs, "aten.ge.Tensor", Cmp::Ge),
        "aten.flip.default" => flip_default(py, args, kwargs),
        "aten.floor_divide.default" => floor_divide_default(py, args, kwargs),
        "aten.floor_divide.Scalar" => floor_divide_scalar(py, args, kwargs),
        "aten.histc.default" => histc_default(py, args, kwargs),
        "aten.clamp_.default" => clamp_inplace_default(py, args, kwargs),
        // `mamba` clamps out of place; only the in-place sibling had a kernel.
        "aten.clamp.default" => clamp_default(py, args, kwargs),
        "aten.clamp_min.default" => clamp_min_default(py, args, kwargs),
        "aten.div_.Tensor" => div_inplace_tensor(py, args, kwargs),
        // `noise.div_(1 - p)`, the scale step of upstream's dropout
        // decomposition (docs/TRAIN.md §1). The out-of-place `div.Scalar` and
        // the in-place `sub_`/`mul_`/`add_` scalar forms were all here already;
        // this one was the hole in the middle of them, and it is the same
        // helper -- `div_.Scalar` differs from `mul_.Scalar` only in
        // `arith_tag`'s true-division promotion, which is what makes
        // `int_tensor.div_(2)` refuse with "result type Float can't be cast to
        // the desired output type Long" rather than floor-dividing.
        "aten.div_.Scalar" => {
            arith_inplace_scalar(py, args, kwargs, "aten.div_.Scalar", Arith::Div)
        }
        "aten.bernoulli_.float" => bernoulli_inplace_float(py, args, kwargs),
        "aten.masked_fill_.Scalar" => masked_fill_inplace(py, args, kwargs, "aten.masked_fill_.Scalar"),
        "aten.index_put_.default" => index_put_inplace(py, args, kwargs),

        // -- the rest of the in-place arithmetic family (docs/ARCH20.md §8) --
        //
        // `add_.Tensor` above was the only one of these with a kernel, and
        // none of the five had a *member*, so `x -= y`, `x *= y`, `x.neg_()`
        // and `x.exp_()` refused at the door. They are the most common
        // in-place operations in the language.
        "aten.add_.Scalar" => {
            arith_inplace_scalar(py, args, kwargs, "aten.add_.Scalar", Arith::Add)
        }
        "aten.sub_.Tensor" => {
            arith_inplace_tensor(py, args, kwargs, "aten.sub_.Tensor", Arith::Sub)
        }
        "aten.sub_.Scalar" => {
            arith_inplace_scalar(py, args, kwargs, "aten.sub_.Scalar", Arith::Sub)
        }
        "aten.mul_.Tensor" => {
            arith_inplace_tensor(py, args, kwargs, "aten.mul_.Tensor", Arith::Mul)
        }
        "aten.mul_.Scalar" => {
            arith_inplace_scalar(py, args, kwargs, "aten.mul_.Scalar", Arith::Mul)
        }
        "aten.neg_.default" => neg_inplace(py, args, kwargs),
        "aten.exp_.default" => exp_inplace(py, args, kwargs),
        "aten.cos_.default" => {
            unary_float_inplace(py, args, kwargs, "aten.cos_.default", Unary::Cos)
        }
        "aten.sin_.default" => {
            unary_float_inplace(py, args, kwargs, "aten.sin_.default", Unary::Sin)
        }
        "aten.erf_.default" => {
            unary_float_inplace(py, args, kwargs, "aten.erf_.default", Unary::Erf)
        }
        "aten.log_.default" => {
            unary_float_inplace(py, args, kwargs, "aten.log_.default", Unary::Log)
        }
        "aten.reciprocal_.default" => {
            unary_float_inplace(py, args, kwargs, "aten.reciprocal_.default", Unary::Reciprocal)
        }
        "aten.tanh_.default" => {
            unary_float_inplace(py, args, kwargs, "aten.tanh_.default", Unary::Tanh)
        }
        "aten.sqrt_.default" => {
            unary_float_inplace(py, args, kwargs, "aten.sqrt_.default", Unary::Sqrt)
        }
        "aten.rsqrt_.default" => rsqrt_inplace(py, args, kwargs),
        "aten.expm1_.default" => expm1_inplace(py, args, kwargs),
        "aten.log2_.default" => log2_inplace(py, args, kwargs),
        "aten.sigmoid_.default" => sigmoid_inplace(py, args, kwargs),
        "aten.abs_.default" => abs_inplace(py, args, kwargs),
        "aten.ceil_.default" => ceil_inplace(py, args, kwargs),
        "aten.clamp_min_.default" => clamp_min_inplace(py, args, kwargs),
        "aten.detach_.default" => detach_inplace_refusal(py, args, kwargs),

        "aten.one_hot.default" => one_hot_default(py, args, kwargs),
        other => Err(aten_not_implemented(other)),
    }
}

#[pyfunction]
#[pyo3(name = "_aten_implemented")]
pub fn aten_implemented() -> Vec<&'static str> {
    IMPLEMENTED.to_vec()
}

/// See `IMPLEMENTED_AWAITING_GOLDEN`. Separate function rather than a flag on
/// `_aten_implemented()` so that nothing can read the union by accident.
#[pyfunction]
#[pyo3(name = "_aten_implemented_awaiting_golden")]
pub fn aten_implemented_awaiting_golden() -> Vec<&'static str> {
    IMPLEMENTED_AWAITING_GOLDEN.to_vec()
}

/// Everything `_aten_dispatch` answers, as one sorted list. Exposed so the
/// smoke tests can check the dispatch table against the three constants rather
/// than keeping a fourth copy of the names.
#[pyfunction]
#[pyo3(name = "_aten_all_implemented")]
pub fn aten_all_implemented() -> Vec<&'static str> {
    all_implemented()
}

// ---------------------------------------------------------------------------
// Implemented ops
//
// Three ops, deliberately of three different *kinds* rather than three of the
// same kind -- a factory, an elementwise binary, and a matmul. Each exercises a
// different part of the floor, so the pattern is shown to generalise. The
// reasoning is written out in docs/TORCH_C.md.
// ---------------------------------------------------------------------------

/// `aten::full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None,
///             Layout? layout=None, Device? device=None, bool? pin_memory=None)`
///
/// The factory. Without one, every tensor has to enter through a back door that
/// the dispatcher cannot see, which would defeat the instrument above.
///
/// `constant_pad_nd` below reuses this function's *fill* half through
/// `filled_block`; the two must agree on how a `Scalar` becomes bytes of a
/// given tag or `F.pad(x, ..., value=v)` and `torch.full(shape, v)` would put
/// different numbers in memory for the same `v`.
fn full_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.full.default";

    let size: Vec<usize> = required(OP, args, kwargs, 0, "size")?.extract()?;
    let fill = required(OP, args, kwargs, 1, "fill_value")?;

    // torch infers int64 from an integer fill value and the default float dtype
    // otherwise. A Python `bool` lands in this branch because `bool` subclasses
    // `int`; torch would give it `torch.bool`, which the shim has no dtype for
    // at all -- recorded in docs/TORCH_C.md rather than papered over.
    // `bool` subclasses `int` in Python, so the bool test has to come first.
    // torch gives `torch.full((2,), True)` dtype `torch.bool`; before the
    // dtype tag existed this branch was unreachable and the shim handed back
    // `int64` (docs/TORCH_C.md §2 recorded it as an open item).
    let fill_is_bool = fill.is_instance_of::<pyo3::types::PyBool>();
    let fill_is_int = fill.is_instance_of::<pyo3::types::PyInt>();
    let dtype = match optional(args, kwargs, 2, "dtype")? {
        Some(value) if !value.is_none() => value.extract::<PyDtype>()?.tag(),
        _ if fill_is_bool => TorchDType::Bool,
        _ if fill_is_int => TorchDType::Int64,
        // `default_float()`, not a literal `Float32`: this arm *is* the "and
        // the default float dtype otherwise" half of the rule the comment
        // above states, and leaving it a constant would have made `full` the
        // one factory that ignores `set_default_dtype`. Measured upstream
        // under a float64 default: `torch.full((2,), 1.0).dtype` is float64.
        _ => default_float(),
    };

    reject_unsupported(OP, args, kwargs, &[(3, "layout"), (5, "pin_memory")])?;
    let label = device_arg_or_label(args, kwargs, 4, "device", &PyDevice::cpu())?;

    // torch refuses a fill value the target dtype cannot hold. candle would
    // wrap (int) or saturate (float) instead, which is the silent divergence
    // the golden harness caught -- see `checked_convert` below.
    //
    // Checked *before* the meta branch on purpose: upstream's meta kernel runs
    // the same conversion check, so `torch.full((3,), 2**31, dtype=torch.int32,
    // device="meta")` raises there too. A meta tensor is a claim about what the
    // real call would have produced, and a claim that skipped the checks the
    // real call makes would be worth less than no claim.
    checked_convert(&fill, fill_is_int, dtype, size.iter().product())?;
    if label.is_meta() {
        return meta_result(py, size, dtype);
    }
    let device = label.resolve()?;
    let storage = PyDtype::new(dtype).storage(OP)?;

    if dtype == TorchDType::Bool {
        // Normalised on the way in, which is what makes the tag's invariant
        // hold by construction rather than by hope (BOOL.md §6.3).
        let truthy = fill.is_truthy()?;
        let tensor = Tensor::full(u8::from(truthy), size, &device)
            .map_err(|e| candle_err(OP, e))?;
        return Ok(PyTensorBase::boolean(tensor)?
            .into_pyobject(py)?
            .into_any()
            .unbind());
    }

    let tensor = if storage.is_int() {
        let value: i64 = fill.extract()?;
        Tensor::full(value, size, &device)
    } else {
        let value: f64 = fill.extract()?;
        Tensor::full(value, size, &device)
    }
    .and_then(|t| t.fast_to(storage))
    .map_err(|e| candle_err(OP, e))?;

    Ok(PyTensorBase::new(tensor)?.into_pyobject(py)?.into_any().unbind())
}

/// A block of `shape`, every element equal to `value`, in `tag`'s storage.
///
/// Split out of `full_default` rather than reimplemented: `constant_pad_nd`'s
/// padding is `torch.full`'s fill with a different shape, and the two rules
/// that make it non-obvious both live here. **Truncation toward zero** for a
/// float value into an integer tag (`constant_pad_nd(int64_t, [1,1], 3.7)`
/// pads with `3`, measured -- not `4`), and **truthiness** for the `bool` tag,
/// which is also what keeps the tag's 0/1 invariant holding by construction
/// (BOOL.md §6.3).
fn filled_block(
    op: &str,
    value: Scalar,
    tag: TorchDType,
    shape: &[usize],
    device: &Device,
) -> PyResult<Tensor> {
    if tag == TorchDType::Bool {
        return Tensor::full(u8::from(value.as_f64() != 0.0), shape, device)
            .map_err(|e| candle_err(op, e));
    }
    let storage = PyDtype::new(tag).storage(op)?;
    if storage.is_int() {
        Tensor::full(value.as_i64(), shape, device)
    } else {
        Tensor::full(value.as_f64(), shape, device)
    }
    .and_then(|t| t.fast_to(storage))
    .map_err(|e| candle_err(op, e))
}

/// `aten::full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None,
///     Layout? layout=None, Device? device=None, bool? pin_memory=None,
///     MemoryFormat? memory_format=None) -> Tensor`
///
/// `t5`'s wall, and `switch_transformers`' -- the *same line* in both, since
/// `T5Attention._relative_position_bucket` is shared verbatim across the whole
/// T5 family (`t5`, `mt5`, `long_t5`, `umt5`, `switch_transformers`,
/// `t5gemma`). The call is
/// `torch.full_like(relative_position_if_large, num_buckets - 1)` on an
/// `int64` tensor with a Python `int` fill (docs/DEMAND.md rank 3).
///
/// A leaf upstream (`_dispatch_has_kernel_for_dispatch_key(...,
/// "CompositeImplicitAutograd")` is `False`), and a `TorchDispatchMode` logger
/// on 2.13.0 records exactly one key for the call above.
///
/// **The dtype rule is the opposite of `full`'s, which is the whole reason
/// this is not a one-line alias for it.** `torch.full` *infers* from the fill
/// value -- an `int` gives `int64`, a `float` gives the default float, a
/// `bool` gives `torch.bool`. `full_like` takes the dtype from the **reference
/// tensor** and ignores the fill value's Python type entirely. Measured on
/// 2.13.0:
///
/// ```text
/// full_like(int64_t, 7)      int64   [7, ...]      full_like(int64_t, 7.5)  int64   [7, ...]
/// full_like(f32_t,   7)      float32 [7.0, ...]    full_like(f32_t,  True)  float32 [1.0, ...]
/// full_like(int64_t, True)   int64   [1, ...]      full_like(bool_t, True)  bool    [True, ...]
/// full_like(int64_t, 3, dtype=torch.float32)       float32 [3.0, ...]   <- override still wins
/// ```
///
/// So `full_like(int64_t, 7.5)` is `7`, where `full([2,3], 7.5)` is `7.5` in
/// the default float. Deriving one from the other by passing the fill through
/// `full`'s inference would give the wrong dtype on the first row and the
/// wrong *value* on the second.
///
/// **The fill itself is not reimplemented.** `filled_block` above is
/// `full_default`'s fill half, already shared with `constant_pad_nd`, and it
/// carries the two rules that would otherwise drift: truncation **toward
/// zero** for a float value into an integral tag (hence `7.5 -> 7`, and
/// `-7.5 -> -7`), and truthiness for the `bool` tag, which is what keeps that
/// tag's 0/1 invariant holding by construction (BOOL.md §6.3). Two
/// implementations of the same fill drift; this is the third caller of one.
///
/// The overflow refusal is `full`'s too, and it fires here: measured,
/// `torch.full_like(int32_t, 2**31)` raises `RuntimeError: value cannot be
/// converted to type int without overflow`. `checked_convert` runs before the
/// meta branch for the reason `full_default`'s comment gives -- upstream's
/// meta kernel makes the same check, and a claim that skipped the checks the
/// real call makes would be worth less than no claim.
///
/// `memory_format` is **accepted and ignored** here rather than rejected,
/// unlike `zeros_like`/`ones_like`/`empty_like` next door, and the difference
/// is measured rather than chosen: `torch.full_like(x, 1.0,
/// memory_format=torch.contiguous_format)` succeeds upstream and returns a
/// contiguous tensor, which is the only layout this shim has. `layout` and
/// `pin_memory` are likewise accepted at their measured defaults and refused
/// otherwise, through `reject_unsupported`.
fn full_like_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.full_like.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let fill = required(OP, args, kwargs, 1, "fill_value")?;
    // The reference tensor's tag, NOT an inference from `fill`. An explicit
    // `dtype=` still beats it, exactly as it does for `new_ones`.
    let tag = dtype_arg(args, kwargs, 2, "dtype")?.unwrap_or(input.tag());
    reject_unsupported(OP, args, kwargs, &[(3, "layout"), (5, "pin_memory")])?;
    let label = device_arg_or_label(args, kwargs, 4, "device", &input.device_label())?;
    let shape = input.dims().to_vec();

    let fill_is_bool = fill.is_instance_of::<pyo3::types::PyBool>();
    let fill_is_int = fill.is_instance_of::<pyo3::types::PyInt>();
    checked_convert(&fill, fill_is_int, tag, shape.iter().product())?;
    if label.is_meta() {
        return meta_result(py, shape, tag);
    }
    let device = label.resolve()?;

    // `Scalar`, then `filled_block` -- the same two steps `constant_pad_nd`
    // takes, so all three callers agree on what a `Scalar` becomes in a tag.
    let value = if fill_is_bool {
        Scalar::Int(i64::from(fill.extract::<bool>()?))
    } else if fill_is_int {
        Scalar::Int(fill.extract()?)
    } else {
        Scalar::Float(fill.extract()?)
    };
    let out = filled_block(OP, value, tag, &shape, &device)?;
    finish(py, out, tag)
}

/// `aten::constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor`
///
/// `bert`'s wall, and the only genuinely new *kernel* the twenty-architecture
/// round needed (docs/ARCH20.md §2). `transformers`'
/// `modeling_utils.py:2701 _adjust_bias` pads the output-embedding bias when
/// the head's vocabulary is wider than the embedding it is tied to, so this
/// runs during `from_config` -- `bert` never reached its own forward.
///
/// `torch.nn.functional.pad(x, pad, "constant", v)` is the only route that
/// reaches it here, and a `TorchDispatchMode` logger confirms upstream lowers
/// that to exactly one record, `aten.constant_pad_nd.default`. The other
/// `mode=` values ("reflect", "replicate", "circular") are different aten ops
/// and are not implemented; `F.pad` picks between them above this layer, so a
/// caller asking for one still gets a refusal naming the op it wanted.
///
/// **`pad` is read back to front and in pairs.** `pad[0..2]` is the last
/// dimension, `pad[2..4]` the one before it, and so on; a shorter list simply
/// leaves the leading dimensions alone. That ordering is the half a plausible
/// implementation gets backwards, so it is pinned by a case whose two
/// dimensions get *different* pads (`[1, 1, 2, 0]` on a `(2, 3)`), which a
/// front-to-back reading cannot pass.
///
/// **Negative entries crop.** Upstream allows them and this reproduces it
/// with `narrow`, including upstream's error when the crop exceeds the axis:
/// "narrow(): length must be non-negative." -- which is upstream's message
/// verbatim, from upstream's own `narrow`, because upstream's implementation
/// takes the same route. The three shapes are measured:
///
/// ```text
/// constant_pad_nd([[0..2],[3..5]], [-1,  0])   [[1,2],[4,5]]
/// constant_pad_nd([[0..2],[3..5]], [-1, -1])   [[1],[4]]
/// constant_pad_nd([[0..2],[3..5]], [-1,  2])   [[1,2,0,0],[4,5,0,0]]
/// ```
///
/// The last one is why cropping and padding cannot be two passes over the
/// whole tensor: they happen on the *same* axis, crop first, and a single
/// axis can do both.
///
/// The two shape refusals are upstream's messages transcribed, spacing
/// included -- "Pad length is 6while the input has 2dimensions." really is
/// missing both spaces upstream, and is reproduced rather than tidied for the
/// reason docs/CKPT2.md §4 gives: a message that differs from upstream's only
/// in wording is useless exactly where it is needed.
fn constant_pad_nd(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.constant_pad_nd.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let pad: Vec<i64> = required(OP, args, kwargs, 1, "pad")?.extract()?;
    let raw_value = optional(args, kwargs, 2, "value")?;
    let value = match raw_value.as_ref() {
        Some(v) if !v.is_none() => scalar_arg(OP, args, kwargs, 2, "value")?
            .ok_or_else(|| missing(OP, "value"))?,
        // The schema default. Written as an integer zero rather than `0.0`
        // because upstream's `Scalar value=0` is an integer too, and the
        // difference is observable through `checked_convert` below.
        _ => Scalar::Int(0),
    };

    if pad.len() % 2 != 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Length of pad must be even but instead it equals {}",
            pad.len()
        )));
    }
    let rank = input.tensor()?.rank();
    if pad.len() / 2 > rank {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Length of pad should be no more than twice the number of \
             dimensions of the input. Pad length is {}while the input has \
             {rank}dimensions.",
            pad.len()
        )));
    }

    let tag = input.tag();
    // Same overflow refusal `full` makes, and for the same reason: candle
    // would wrap an out-of-range integer fill silently. `numel` is the padded
    // element count only in spirit here -- what it selects is upstream's
    // "one element skips the reduced-float check" hole, and a pad block is
    // never the one-element case unless the whole pad is empty, so 2 is the
    // honest argument rather than a computed size.
    if let Some(v) = raw_value.as_ref() {
        if !v.is_none() && !v.is_instance_of::<PyTensorBase>() {
            checked_convert(v, v.is_instance_of::<pyo3::types::PyInt>(), tag, 2)?;
        }
    }

    let device = input.tensor()?.device().clone();
    let mut out = input.tensor()?.contiguous().map_err(|e| candle_err(OP, e))?;
    for (pair, chunk) in pad.chunks(2).enumerate() {
        let (left, right) = (chunk[0], chunk[1]);
        // `pad[0..2]` is the LAST dimension. `rank - 1 - pair` is that rule,
        // and `pad.len() / 2 <= rank` above is what makes it in range.
        let axis = rank - 1 - pair;

        // Crop before padding: a single axis can do both (`[-1, 2]`).
        let drop_front = (-left).max(0) as usize;
        let drop_back = (-right).max(0) as usize;
        if drop_front != 0 || drop_back != 0 {
            let extent = out.dims()[axis];
            let kept = extent as i64 - drop_front as i64 - drop_back as i64;
            if kept < 0 {
                // Upstream's message, from upstream's own `narrow`.
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "narrow(): length must be non-negative.",
                ));
            }
            out = out
                .narrow(axis, drop_front, kept as usize)
                .map_err(|e| candle_err(OP, e))?;
        }

        for (amount, before) in [(left.max(0), true), (right.max(0), false)] {
            if amount == 0 {
                continue;
            }
            let mut shape = out.dims().to_vec();
            shape[axis] = amount as usize;
            let block = filled_block(OP, value, tag, &shape, &device)?;
            let pieces: Vec<&Tensor> = if before {
                vec![&block, &out]
            } else {
                vec![&out, &block]
            };
            out = Tensor::cat(&pieces, axis)
                .and_then(|t| t.contiguous())
                .map_err(|e| candle_err(OP, e))?;
        }
    }
    finish(py, out, tag)
}

/// `aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1)`
///
/// The elementwise binary. This is where torch and candle disagree the most
/// cheaply observable way: torch broadcasts *and* promotes dtypes, candle
/// broadcasts but requires matching dtypes. Promotion is not implemented here;
/// a mismatch raises with both dtypes named rather than guessing, because a
/// wrong promotion is the silent numerical drift DESIGN.md §5 calls A's main
/// risk.
fn add_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.add.Tensor";

    let lhs = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(OP, args, kwargs, 1, "other")?;
    let alpha: f64 = match optional(args, kwargs, 2, "alpha")? {
        Some(value) if !value.is_none() => value.extract()?,
        _ => 1.0,
    };

    // Promotes, over the same lattice as every other elementwise binary op --
    // docs/PROMOTE.md §3, where `add.Tensor`'s 9x9 grid was measured against
    // `torch.promote_types` and agreed in every cell.
    let tag = promote_operands(OP, &lhs, &rhs)?;
    // `bool + bool` is a logical or in torch, not an arithmetic sum
    // (BOOL.md §2.2). candle's `broadcast_add` would give 2 where both are
    // true, which is still truthy and therefore silently wrong downstream --
    // so this refuses rather than approximates.
    //
    // Only `bool + bool` reaches here now: a `bool` against anything else
    // promotes to that other dtype and is ordinary arithmetic upstream
    // (`bool + float32` is `float32` `[2.0]`, measured), which is what makes
    // testing the *promoted* tag the right test rather than either operand's.
    if tag == TorchDType::Bool {
        return Err(not_implemented(format!(
            "{OP}: torch.bool addition is logical or, not arithmetic, and is              not implemented in torch._C shim"
        )));
    }

    // Widened, added, and narrowed once -- `opmath_in`. `alpha` is applied
    // *after* the widening on purpose: upstream converts it to `opmath_t` and
    // multiplies there, so scaling in `bfloat16` first would round the
    // multiply and the add separately where torch rounds only the result.
    let storage = PyDtype::new(tag).storage(OP)?;
    let acc = opmath_in(storage);
    // The whole widen/add/narrow in one pass when the operands allow it. It
    // computes the same function -- `reduced::fused_arith` refuses rather than
    // approximating -- and `alpha != 1` is left to the slow path because
    // `scale_by_alpha` is a rule of its own (§3.1 of docs/BF16.md) and folding
    // it in here would be a second place for that rule to live.
    //
    // Both operands reach the common dtype before either reaches `acc` --
    // `operand_in`, which is where that ordering is justified.
    let lhs_common = operand_in(OP, lhs.tensor()?, storage)?;
    let rhs_common = operand_in(OP, rhs.tensor()?, storage)?;
    if alpha == 1.0 {
        if let Some(out) =
            crate::reduced::fused_arith(Fused::Add, &lhs_common, &rhs_common, storage)
        {
            let out = out.map_err(|e| candle_err(OP, e))?;
            return Ok(PyTensorBase::new(out)?.into_pyobject(py)?.into_any().unbind());
        }
    }
    let lhs = lhs_common.fast_to(acc).map_err(|e| candle_err(OP, e))?;
    let rhs = rhs_common.fast_to(acc).map_err(|e| candle_err(OP, e))?;
    let rhs = scale_by_alpha(OP, &rhs, alpha, storage)?;
    let out = lhs
        .broadcast_add(&rhs)
        .and_then(|t| t.fast_to(storage))
        .map_err(|e| candle_err(OP, e))?;

    Ok(PyTensorBase::new(out)?.into_pyobject(py)?.into_any().unbind())
}

/// The dtype torch accumulates a GEMM of this storage dtype in.
///
/// **`float16` and `bfloat16` GEMMs accumulate in `float32` upstream**
/// (`at::opmath_type<Half> == float`), and this is measured, not inferred:
/// `mm(half a, half b)` is *bitwise* equal to `half(mm(float(a), float(b)))`
/// at k = 4, 64 and 512, and the same holds for `bmm` and `addmm`.
///
/// candle has no such notion -- `Tensor::matmul` accumulates in the storage
/// dtype -- so routing a `float16` matmul straight at it computes a different
/// function, and the difference is not a rounding-order nicety: at k = 512
/// with unit-magnitude inputs, **15 of 64 outputs land outside this harness's
/// `float16` tolerance**, with a maximum absolute error of 0.078 against a
/// tolerance of 5e-3. It grows with the reduction depth (1/64 outputs already
/// wrong at k = 64), so a real model in `float16` drifts layer by layer.
///
/// That went unnoticed because every GEMM case in `tools/golden/cases.py` was
/// small enough for `float16` accumulation to be lossless; docs/GPT2.md §7
/// listed "the error at real layer sizes" as unmeasured, and this was in it.
/// The large-size cases added alongside this function are what found it.
///
/// The same widening incidentally gives `bfloat16` a matmul at all: candle has
/// no BF16 matmul kernel (`unsupported dtype BF16 for op matmul`), which
/// `mm`/`bmm`/`addmm` had been recording as a capability gap. Accumulating in
/// `float32` is not a workaround for that gap -- it is what upstream does --
/// so the gap closes as a side effect rather than being papered over.
///
/// The integral dtypes are deliberately *not* widened. candle having no
/// integral matmul is a real gap, and `float32` cannot hold an `int64` product
/// exactly; standing one in would answer a different question.
fn gemm_accumulate_in(storage: candle_core::DType) -> candle_core::DType {
    // `float8_e4m3fn` is a gemm-only widening and deliberately not part of
    // `opmath_in` (docs/FLOAT8C.md §4). candle has no `F8E4M3` matmul at all --
    // `unsupported dtype F8E4M3 for op matmul` -- while upstream computes
    // `mm`/`addmm`/`bmm`/`matmul` for it and returns a `float8_e4m3fn` result.
    //
    // Widening the operands to `f32`, multiplying, and narrowing back is not a
    // guess about what upstream does: over 700 random cases -- `k` from 1 to
    // 512, three magnitude scales, both `f32` and `f64` accumulation -- the
    // narrowed product was **bit-identical** to upstream's `mm` in every one,
    // and the `f32` and `f64` routes never differed from each other either.
    // Three mantissa bits is coarse enough that the accumulation width cannot
    // show through the final rounding.
    //
    // Scoped to gemm rather than to `opmath_in` because `opmath_in` is the
    // elementwise opmath type, and upstream **refuses** float8 for the
    // elementwise kernels (docs/FLOAT8B.md §2). Widening there would compute
    // where upstream declines, which is the divergence direction this dtype's
    // gate exists to prevent.
    if storage == candle_core::DType::F8E4M3 {
        return candle_core::DType::F32;
    }
    opmath_in(storage)
}

/// Widen a GEMM operand to the accumulation dtype **without flattening the
/// layout it arrived in**.
///
/// `FastDType::fast_to` takes the NEON conversion path only when the tensor is
/// contiguous and hands everything else to candle's per-element `to_dtype`
/// (`reduced.rs`). That gate reads as conservative, and it is the opposite:
/// **every weight in a real forward pass arrives here non-contiguous.**
/// `bootstrap.py::linear` hands the kernel `t(weight)`, a free transpose
/// *view*, so the fast conversion `docs/DTYPE.md` added never once fired on
/// the operand that dominates. It was measured on contiguous tensors, which is
/// the layout a model never produces.
///
/// The fallback costs twice, not once:
///
///   * candle's `to_dtype` is the scalar per-element loop, and
///   * it **materialises a contiguous result**, which throws away the
///     transpose that `gemm_with_layout_fallback` exists to preserve -- so the
///     widened operand also loses its `CblasTrans` and is re-gathered. On
///     SmolLM2-135M's `lm_head` that is 113 MB rewritten per call.
///
/// Measured at the decoding shape (`[1,576] @ [576,49152]`, `bfloat16`):
/// **69.60 ms** through the fallback, against 3.10 ms for our own `float32`
/// and 1.03 ms for upstream. docs/DTYPE_PERF.md §4.
///
/// **Why this cannot move a value.** Conversion is elementwise, so it commutes
/// with a transpose: transposing is a relabelling of *which* element sits
/// where and changes no element, and `fast_to` applies the same function to
/// each. So widening the contiguous base and transposing the result is the
/// same tensor -- bit for bit, not merely within tolerance -- as widening the
/// transposed view. Any layout not recognised here stays on the old path.
fn widen_gemm_operand(t: &Tensor, acc: candle_core::DType) -> candle_core::Result<Tensor> {
    if t.dtype() == acc || t.layout().is_contiguous() || t.rank() < 2 {
        return t.fast_to(acc);
    }
    // The one layout worth recognising is a plain transpose of a contiguous
    // tensor, which is exactly what `linear` produces. `t()` swaps the last
    // two dims, so if the result of that swap is contiguous, the input was
    // that transpose and nothing else.
    if let Ok(base) = t.t() {
        if base.layout().is_contiguous() {
            return base.fast_to(acc)?.t();
        }
    }
    t.fast_to(acc)
}

/// Is this candle error "I cannot consume that layout" rather than anything
/// else?
///
/// `MatMulUnexpectedStriding` is the only error `MatMul` raises for a layout,
/// and it is raised *before* the destination is allocated on both backends, so
/// catching it costs a stride check and not a GEMM. `bt()` wraps the variant in
/// `WithBacktrace` whenever backtraces are enabled, which is why this recurses
/// instead of matching one level.
fn is_matmul_striding_refusal(e: &candle_core::Error) -> bool {
    match e {
        candle_core::Error::MatMulUnexpectedStriding(_) => true,
        candle_core::Error::WithBacktrace { inner, .. } => is_matmul_striding_refusal(inner),
        _ => false,
    }
}

/// Multiply two GEMM operands **in the layout they arrived in**, copying only
/// if candle refuses that layout.
///
/// The copy this replaces was unconditional, and for `F.linear` it was the
/// dominant cost of the whole call: `bootstrap.py::linear` hands the kernel
/// `t(weight)`, a free transpose *view*, and `.contiguous()` on that view is a
/// strided gather of the entire weight. On SmolLM2-135M's `lm_head` that is
/// 113 MB re-written per forward pass (docs/QUANT2.md §6, docs/LINEAR.md).
///
/// **candle does not need it.** Both CPU backends read `lhs_l.stride()` and
/// `rhs_l.stride()` and hand the last two of them to the multiply:
///
/// - Accelerate/MKL (`cpu_backend/mod.rs`, `#[cfg(feature = "accelerate")]`)
///   picks `transa`/`transb` from those strides, so a transposed operand
///   becomes `CblasTrans` -- which is exactly the case `linear` produces. Any
///   *other* non-contiguous layout (a strided slice, say) it refuses with
///   `MatMulUnexpectedStriding`.
/// - The `gemm` backend (every non-Apple target) passes `lhs_cs`/`lhs_rs` and
///   `rhs_cs`/`rhs_rs` straight through and so accepts arbitrary strides on
///   the last two dimensions.
///
/// Both share `MatMul::ab_skip`, which needs the *batch* strides to be one of
/// four recognised shapes and refuses otherwise -- reachable from rank 4 and
/// up, where two batch axes can be swapped.
///
/// So the accepted set differs between backends and between ranks, and this
/// deliberately does not try to restate it. Restating it would mean keeping a
/// copy of candle's predicate in sync with candle, and getting that copy wrong
/// in the permissive direction is a wrong answer rather than a failure. Asking
/// candle and copying only when it says no cannot be wrong in that direction:
/// the layouts that reach the multiply are exactly the ones candle accepts, and
/// nothing is passed through silently.
///
/// `aten.mm.default` has never called `.contiguous()`, so the "candle takes a
/// strided operand" half of this was already shipping and golden-compared; what
/// is new is the fallback and the other four kernels.
fn gemm_with_layout_fallback(
    lhs: &Tensor,
    rhs: &Tensor,
    multiply: impl Fn(&Tensor, &Tensor) -> candle_core::Result<Tensor>,
) -> candle_core::Result<Tensor> {
    match multiply(lhs, rhs) {
        Err(e) if is_matmul_striding_refusal(&e) => {
            multiply(&lhs.contiguous()?, &rhs.contiguous()?)
        }
        other => other,
    }
}

/// `matmul` over operands that may disagree in rank -- **folding the batch
/// into the rows when the right operand has none**, which is what upstream
/// does and is the difference between one GEMM and one copy of the weight.
///
/// candle's `broadcast_matmul` equalises the ranks by broadcasting and then
/// concretising: `rhs.broadcast_as(&r_shape)?.contiguous()?`, with a
/// `// TODO: Avoid concretising the broadcasted matrixes via contiguous.`
/// above it. For `(1, seq, k) @ (k, n)` -- the shape *every* transformer
/// activation has -- that materialises the whole weight on every call, which
/// is the same 113 MB `lm_head` copy `gemm_with_layout_fallback` exists to
/// remove, just performed one level down. Removing our `.contiguous()` alone
/// does nothing here: measured 85.8 ms before, 88.3 ms after, against 4.7 ms
/// upstream (docs/LINEAR.md).
///
/// `at::native::matmul` does not broadcast this case at all. When the right
/// operand is 2-D it *folds*: the left operand's leading dimensions collapse
/// into the row dimension, one 2-D GEMM runs, and the result is unflattened.
/// `at::native::linear` spells the same thing out for the no-bias N-D branch
/// as `t, view, mm, _unsafe_view` -- which `bootstrap.py::_install_nn` already
/// transcribes in a comment above the `linear` it installs.
///
/// The fold is exact rather than an approximation of the batched form: with a
/// 2-D right operand every batch element multiplies the *same* matrix, so
/// stacking the rows computes the identical set of dot products. It is only
/// reachable when the right operand is 2-D and the left has more; everything
/// else still goes to `broadcast_matmul`, so the broadcasting semantics this
/// kernel had are unchanged for every other shape.
///
/// The `reshape` is free for a contiguous left operand and copies for a
/// non-contiguous one -- but that copy is of the *activation*, which is the
/// small operand, and candle's `broadcast_matmul` would have copied the
/// weight instead.
fn batched_matmul(lhs: &Tensor, rhs: &Tensor) -> candle_core::Result<Tensor> {
    if lhs.rank() > 2 && rhs.rank() == 2 {
        let dims = lhs.dims();
        let (lead, k) = dims.split_at(dims.len() - 1);
        let rows: usize = lead.iter().product();
        let folded = lhs.reshape((rows, k[0]))?;
        let product = gemm_with_layout_fallback(&folded, rhs, |a, b| a.matmul(b))?;
        let mut out_shape = lead.to_vec();
        out_shape.push(rhs.dims()[1]);
        return product.reshape(out_shape);
    }
    gemm_with_layout_fallback(lhs, rhs, |a, b| a.broadcast_matmul(b))
}

/// `at::opmath_type` -- the dtype torch *computes* in for a given storage
/// dtype. `float` for both reduced floats, the storage dtype for everything
/// else.
///
/// This is one rule, not a GEMM rule: torch widens `bfloat16` and `float16`
/// for **every** arithmetic kernel, computes there, and narrows back exactly
/// once, with round-to-nearest-even. `gemm_accumulate_in` above is this
/// function under the name the matmuls found it by; the elementwise ops and
/// the reductions need it for the same reason and were missing it.
///
/// **What that cost, measured.** candle's `bfloat16` add narrows by
/// truncating, and only on its vectorised path -- correct below 32 elements,
/// wrong at 32 and above, which is why every case in `tools/golden/cases.py`
/// (all at most 24 elements) passed. Truncation is *biased*: it moves every
/// rounded element toward zero instead of splitting them evenly, so the error
/// accumulates down a residual stream instead of cancelling. On
/// SmolLM2-135M's default `bfloat16` path that reached a maximum logit
/// difference of 11.75 against upstream and changed the generated text; with
/// the widening it is 0.0 and the tokens match. docs/BF16.md.
///
/// Widening is not "more accurate than torch" here -- it is what torch does.
/// `add` on two `bfloat16` values is `float(a) + float(b)` narrowed once
/// upstream too, so this reproduces the operation rather than improving it.
/// The integral dtypes are deliberately untouched: widening an `int64` to
/// `float32` would lose bits rather than keep them.
fn opmath_in(storage: candle_core::DType) -> candle_core::DType {
    match storage {
        candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
        other => other,
    }
}

/// `alpha * operand`, where `operand` is already widened to `opmath_in`.
///
/// `add` and `sub` are the only ops with an `alpha`, and it is **not** simply
/// a factor applied in the widened dtype. Two things about it were measured
/// against torch 2.13.0 over 20000 elements per alpha, because both are the
/// kind of thing that is invisible at `alpha=1` (the only value any caller in
/// this repository passes) and wrong everywhere else:
///
///   * **`alpha` is narrowed to the storage dtype first.** With
///     `alpha=0.3` on `bfloat16`, torch multiplies by `0.30078125`, not by
///     `0.3`. Keeping the `f64` the parser produced disagrees on 312/20000
///     elements.
///   * **On `bfloat16` the product is narrowed too, and on `float16` it is
///     not.** This asymmetry is measured, not derived, and it is not a
///     rounding accident: on `bfloat16`, narrowing the product agrees with
///     upstream on 0/20000 elements for every one of `alpha` in
///     {3, 2, 0.3, -1.5, 1.7}, while leaving it in `float` disagrees on up to
///     1607. On `float16` it is the other way round -- narrowing the product
///     disagrees on up to 1246 where leaving it in `float` disagrees on 0 or
///     1. The two reduced floats reach different kernels upstream.
///
/// The single residual `float16` disagreement (1/20000 at `alpha=1.7`) is a
/// double-rounding edge and is left as measured rather than papered over.
fn scale_by_alpha(
    op: &str,
    operand: &Tensor,
    alpha: f64,
    storage: candle_core::DType,
) -> PyResult<Tensor> {
    if alpha == 1.0 {
        return Ok(operand.clone());
    }
    let acc = opmath_in(storage);
    if acc == storage {
        return operand.affine(alpha, 0.0).map_err(|e| candle_err(op, e));
    }
    let narrowed = Tensor::full(alpha, (), operand.device())
        .and_then(|t| t.fast_to(storage))
        .and_then(|t| widen_f64(&t))
        .and_then(|t| t.to_scalar::<f64>())
        .map_err(|e| candle_err(op, e))?;
    let scaled = operand
        .affine(narrowed, 0.0)
        .map_err(|e| candle_err(op, e))?;
    if storage == candle_core::DType::BF16 {
        return scaled
            .fast_to(storage)
            .and_then(|t| t.fast_to(acc))
            .map_err(|e| candle_err(op, e));
    }
    Ok(scaled)
}

/// `aten::mm(Tensor self, Tensor mat2)`
///
/// The matmul. Chosen over the other elementwise ops because it is the one op
/// that is *hot* -- `nn.Linear`, the only compute-heavy module in the ten
/// live Python modules of IMPORT_WALLS §5, becomes this. It is also the op
/// whose backend choice (naive gemm vs Accelerate vs a fused kernel from §8)
/// will be revisited, so the floor should already route through it.
fn mm_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.mm.default";

    let lhs = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(OP, args, kwargs, 1, "mat2")?;

    // torch's `mm` is strictly 2-D; `matmul`/`bmm` are separate ops with their
    // own overloads. candle's `matmul` accepts batched inputs, so accepting a
    // 3-D argument here would quietly implement a different op.
    if lhs.tensor()?.rank() != 2 || rhs.tensor()?.rank() != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{OP}: both arguments to mm need to be 2D, but they are {}D and {}D",
            lhs.tensor()?.rank(),
            rhs.tensor()?.rank()
        )));
    }
    let tag = require_same_dtype(OP, &lhs, &rhs)?;

    // Accumulate where torch accumulates -- see `gemm_accumulate_in`.
    let storage = PyDtype::new(tag).storage(OP)?;
    let acc = gemm_accumulate_in(storage);
    let rhs_inner = rhs.tensor()?;
    let out = widen_gemm_operand(lhs.tensor()?, acc)
        .and_then(|l| {
            widen_gemm_operand(rhs_inner, acc)
                .and_then(|r| gemm_with_layout_fallback(&l, &r, |a, b| a.matmul(b)))
        })
        .and_then(|p| p.fast_to(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::bmm(Tensor self, Tensor mat2) -> Tensor`
///
/// **Not a one-line route into `matmul_default`.** The kernel underneath is
/// indeed the same candle call, and `matmul_default` already batches -- but it
/// batches by *broadcasting*, and `bmm` does not. Upstream refuses
/// `bmm((1,3,4), (2,4,5))` ("Expected size for first two dimensions of batch2
/// tensor to be: [1, 4] but got: [2, 4]") where `broadcast_matmul` happily
/// expands the batch of 1. Sending `aten.bmm.default` at `matmul_default`
/// would therefore implement a *different* op -- one that computes where
/// torch raises, which is the silent-divergence direction DESIGN.md §5 exists
/// to keep out.
///
/// So the shared part is the multiply and the distinct part is the contract:
/// both operands strictly 3-D, batch extents equal, no broadcasting.
fn bmm_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.bmm.default";

    let lhs = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(OP, args, kwargs, 1, "mat2")?;

    if lhs.tensor()?.rank() != 3 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "batch1 must be a 3D tensor",
        ));
    }
    if rhs.tensor()?.rank() != 3 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "batch2 must be a 3D tensor",
        ));
    }
    let tag = require_same_dtype(OP, &lhs, &rhs)?;

    // torch checks batch2's leading pair against batch1's (batch, k) and says
    // so in exactly these words. Reproduced rather than paraphrased: the
    // message is the work item a caller reads.
    let a = lhs.tensor()?.dims();
    let b = rhs.tensor()?.dims();
    if a[0] != b[0] || a[2] != b[1] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Expected size for first two dimensions of batch2 tensor to be: \
             [{}, {}] but got: [{}, {}].",
            a[0], a[2], b[0], b[1]
        )));
    }

    let storage = PyDtype::new(tag).storage(OP)?;
    let acc = gemm_accumulate_in(storage);
    let rhs_inner = rhs.tensor()?;
    let out = widen_gemm_operand(lhs.tensor()?, acc)
        .and_then(|l| {
            widen_gemm_operand(rhs_inner, acc)
                .and_then(|r| gemm_with_layout_fallback(&l, &r, |a, b| a.matmul(b)))
        })
        .and_then(|p| p.fast_to(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// The dtypes `_grouped_mm`'s CPU kernel accepts, in upstream's own words for
/// the refusal message.
///
/// **Not the meta function's set.** `_meta_grouped_mm_common` in
/// `torch/_meta_registrations.py` checks `mat_a.dtype == torch.bfloat16` and
/// refuses everything else, and it is the only readable implementation in the
/// vendored tree -- so reading it as the specification is the natural mistake.
/// The CPU kernel takes f32, bf16 and f16, measured, and f32 is the dtype
/// Mixtral calls it with. docs/GROUPED_MM.md §1.
fn grouped_mm_dtype_ok(tag: TorchDType) -> bool {
    matches!(
        tag,
        TorchDType::Float32 | TorchDType::BFloat16 | TorchDType::Float16
    )
}

/// Upstream's 16-byte stride rule for a `_grouped_mm` operand.
///
/// Reproduced rather than skipped, and that is a decision rather than an
/// oversight -- see docs/GROUPED_MM.md §2.2. candle would multiply these
/// operands happily; upstream's CPU kernel will not, and `transformers` keeps
/// a whole fallback path (`torch.ops.transformers.grouped_mm_fallback`) for
/// programs that would hit it. Computing where upstream raises is the
/// silent-divergence direction, so the refusal is carried over with upstream's
/// own wording.
///
/// The predicate is `check_valid_strides` from `_meta_registrations.py`,
/// re-derived here by sweeping shapes on 2.13.0 rather than transcribed. Only
/// the last two strides are examined: a 3-D operand's batch stride is not
/// checked, and neither is the data pointer (`transformers`' own
/// `_can_use_grouped_mm` guards the pointer only for torch <= 2.10).
fn grouped_mm_strides_ok(tensor: &Tensor, storage: candle_core::DType) -> bool {
    let dims = tensor.dims();
    let stride = tensor.stride();
    let end = dims.len() - 1;
    // `16 / itemsize`: 4 elements for f32, 8 for the 2-byte floats.
    let alignment = 16 / storage.size_in_bytes().max(1);
    if stride[end - 1] == 1 && stride[end] >= dims[end - 1].max(1) {
        stride[end] % alignment == 0
    } else if stride[end] == 1 && stride[end - 1] >= dims[end].max(1) {
        stride[end - 1] % alignment == 0
    } else {
        false
    }
}

/// Which group owns each index along the partitioned extent, by *simulating
/// upstream's sequential write loop* rather than assuming the offsets behave.
///
/// `offs` is a cumulative end index, so group `g` covers `[offs[g-1], offs[g])`
/// with `offs[-1]` read as `0`. Two measured behaviours make the obvious
/// `cat`-of-blocks implementation wrong, and this returns the information
/// needed to get both right (docs/GROUPED_MM.md §2.3):
///
///  * `offs[-1] < extent` leaves the tail **unwritten**. `transformers` relies
///    on that on purpose for its expert-parallel sentinel rows. Unwritten
///    indices come back as `None` here and are filled with zeros, which is *a*
///    valid answer to an uninitialised question -- so nothing asserts on them.
///  * `offs` is not required to increase. `[9, 5, 24]` writes rows `0..9`, then
///    nothing, then rows `5..24` -- **overwriting** `5..9`, because a later
///    group wins. Simulating the loop reproduces that; concatenating blocks
///    would not.
///
/// Out-of-range offsets are clamped. Upstream reads out of bounds there, which
/// is undefined rather than a behaviour to match.
///
/// The result is compressed into maximal runs so the caller issues one
/// `matmul` per run. For the ordinary monotonic case the runs are exactly the
/// groups, so the simulation costs nothing; for the pathological case it still
/// computes each index once, because every row of a grouped product depends
/// only on its own row of the left operand.
fn grouped_mm_runs(offs: &[i32], extent: usize) -> Vec<(usize, usize, Option<usize>)> {
    let mut owner: Vec<Option<usize>> = vec![None; extent];
    let mut prev = 0usize;
    for (g, &off) in offs.iter().enumerate() {
        let end = if off < 0 {
            0
        } else {
            (off as usize).min(extent)
        };
        for slot in owner.iter_mut().take(end).skip(prev) {
            *slot = Some(g);
        }
        prev = end;
    }

    let mut runs: Vec<(usize, usize, Option<usize>)> = Vec::new();
    for (i, who) in owner.into_iter().enumerate() {
        match runs.last_mut() {
            Some(last) if last.2 == who => last.1 = i + 1,
            _ => runs.push((i, i + 1, who)),
        }
    }
    runs
}

/// One group's product, in the accumulation dtype, tolerating a transposed
/// operand the way the other five GEMM kernels do.
fn grouped_mm_block(
    op: &str,
    lhs: &Tensor,
    rhs: &Tensor,
    acc: candle_core::DType,
) -> PyResult<Tensor> {
    let l = lhs.fast_to(acc).map_err(|e| candle_err(op, e))?;
    let r = rhs.fast_to(acc).map_err(|e| candle_err(op, e))?;
    gemm_with_layout_fallback(&l, &r, |a, b| a.matmul(b)).map_err(|e| candle_err(op, e))
}

/// `aten::_grouped_mm(Tensor self, Tensor mat2, Tensor? offs=None,
///     Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor`
///
/// The mixture-of-experts GEMM, and the last operator standing between this
/// shim and Mixtral (docs/OPS4.md §13.3 left it out of scope by name).
/// Instead of one `(M,K) x (K,N)` it multiplies a stack of *variable-sized*
/// groups described by a cumulative offset vector, which is how an MoE layer
/// routes tokens to experts without materialising a tensor per expert.
///
/// Four layouts, because `self` and `mat2` may each be 2-D or 3-D, and `offs`
/// partitions a different axis in each (docs/GROUPED_MM.md §2):
///
/// ```text
///   (M,K) x (G,K,N)  offs over the rows of self         -> (M,N)
///   (G,M,K) x (K,N)  offs over the columns of mat2      -> (M,N)
///   (M,K) x (K,N)    offs over the contraction K        -> (G,M,N)
///   (G,M,K) x (G,K,N)  offs forbidden -- this is bmm    -> (G,M,N)
/// ```
///
/// `bias` and a non-identity `out_dtype` are in the schema and are **not
/// implemented upstream at all** -- both are refused there, so both are
/// refused here, in upstream's words. There is no capability gap being
/// papered over: a shim that computed a bias here would be answering a
/// question torch does not answer.
///
/// The 2-D x 2-D layout has no contraction check on purpose. `offs` slices
/// both operands with the same range, so the extents outside it are never
/// read and upstream does not compare them -- `_grouped_mm((8,8), (4,4),
/// offs=[2,4])` computes. Measured; adding the check that the other layouts
/// have would refuse a call upstream accepts.
fn grouped_mm_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._grouped_mm.default";

    let mat_a = tensor_arg(OP, args, kwargs, 0, "self")?;
    let mat_b = tensor_arg(OP, args, kwargs, 1, "mat2")?;
    let offs = optional_tensor_arg(OP, args, kwargs, 2, "offs")?;
    let bias = optional_tensor_arg(OP, args, kwargs, 3, "bias")?;
    let out_dtype = dtype_arg(args, kwargs, 4, "out_dtype")?;

    let a = mat_a.tensor()?;
    let b = mat_b.tensor()?;
    if a.rank() != 2 && a.rank() != 3 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "mat_a has to be 2 or 3d",
        ));
    }
    if b.rank() != 2 && b.rank() != 3 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "mat_b has to be 2 or 3d",
        ));
    }
    for (label, tag) in [("mat_a", mat_a.tag()), ("mat_b", mat_b.tag())] {
        if !grouped_mm_dtype_ok(tag) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Expected {label} to be Float32, BFloat16 or Float16 matrix, got {}",
                scalar_type_name(tag)
            )));
        }
    }
    if mat_a.tag() != mat_b.tag() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "expected m1 and m2 to have the same dtype, but got: {} != {}",
            c10_name(mat_a.tag()),
            c10_name(mat_b.tag())
        )));
    }
    if bias.is_some() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Bias not supported yet",
        ));
    }
    if let Some(requested) = out_dtype {
        if requested != mat_a.tag() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Grouped gemm output dtype must match `mat_a` dtype",
            ));
        }
    }

    let a_is_2d = a.rank() == 2;
    let b_is_2d = b.rank() == 2;
    if (a_is_2d || b_is_2d) != offs.is_some() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Have to provide offsets if there is a 2d matrix, or no offset if \
             both matrices are 3d",
        ));
    }

    let storage = PyDtype::new(mat_a.tag()).storage(OP)?;
    for operand in [a, b] {
        if !grouped_mm_strides_ok(operand, storage) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "strides should be multiple of 16 bytes",
            ));
        }
    }
    let acc = gemm_accumulate_in(storage);
    let a_dims = a.dims().to_vec();
    let b_dims = b.dims().to_vec();

    // Both 3-D: no offsets, no groups -- upstream says so in the meta function
    // ("regular bmm") and the CPU kernel agrees. The batch and contraction
    // messages are its own, not `bmm`'s.
    let Some(offs) = offs else {
        if a_dims[0] != b_dims[0] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "batched dimension has to match",
            ));
        }
        if a_dims[2] != b_dims[1] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "contraction dimension of mat_a and mat_b must match",
            ));
        }
        let out = grouped_mm_block(OP, a, b, acc)?
            .fast_to(storage)
            .map_err(|e| candle_err(OP, e))?;
        return finish(py, out, mat_a.tag());
    };

    if offs.tag() != TorchDType::Int32 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Offsets have to be int32",
        ));
    }
    let offs_tensor = offs.tensor()?;
    if offs_tensor.rank() != 1 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "offs has to be 1D",
        ));
    }
    let groups = offs_tensor
        .to_vec1::<i32>()
        .map_err(|e| candle_err(OP, e))?;

    // The contraction check exists for every layout that has a 3-D operand,
    // and for the 2-D x 2-D layout it deliberately does not -- see the doc
    // comment.
    if !a_is_2d || !b_is_2d {
        let a_k = a_dims[a_dims.len() - 1];
        let b_k = b_dims[b_dims.len() - 2];
        if a_k != b_k {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "contraction dimension of mat_a and mat_b must match",
            ));
        }
    }

    let device = a.device().clone();
    let out = if a_is_2d && !b_is_2d {
        // (M,K) x (G,K,N): `offs` walks the rows of `self`, one expert per
        // group. This is Mixtral's shape.
        if groups.len() != b_dims[0] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "matrix batch sizes have to match",
            ));
        }
        let (m, n) = (a_dims[0], b_dims[2]);
        let mut blocks: Vec<Tensor> = Vec::new();
        for (start, end, who) in grouped_mm_runs(&groups, m) {
            let rows = end - start;
            blocks.push(match who {
                Some(g) => {
                    let lhs = a.narrow(0, start, rows).map_err(|e| candle_err(OP, e))?;
                    let rhs = b.narrow(0, g, 1).map_err(|e| candle_err(OP, e))?;
                    let rhs = rhs.squeeze(0).map_err(|e| candle_err(OP, e))?;
                    grouped_mm_block(OP, &lhs, &rhs, acc)?
                }
                None => Tensor::zeros((rows, n), acc, &device).map_err(|e| candle_err(OP, e))?,
            });
        }
        grouped_mm_concat(OP, blocks, (m, n), 0, acc, &device)?
    } else if !a_is_2d && b_is_2d {
        // (G,M,K) x (K,N): `offs` walks the *columns* of `mat2`; the output is
        // still 2-D and each group owns a slab of its columns.
        if groups.len() != a_dims[0] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "matrix batch sizes have to match",
            ));
        }
        let (m, n) = (a_dims[1], b_dims[1]);
        let mut blocks: Vec<Tensor> = Vec::new();
        for (start, end, who) in grouped_mm_runs(&groups, n) {
            let cols = end - start;
            blocks.push(match who {
                Some(g) => {
                    let lhs = a.narrow(0, g, 1).map_err(|e| candle_err(OP, e))?;
                    let lhs = lhs.squeeze(0).map_err(|e| candle_err(OP, e))?;
                    let rhs = b.narrow(1, start, cols).map_err(|e| candle_err(OP, e))?;
                    grouped_mm_block(OP, &lhs, &rhs, acc)?
                }
                None => Tensor::zeros((m, cols), acc, &device).map_err(|e| candle_err(OP, e))?,
            });
        }
        grouped_mm_concat(OP, blocks, (m, n), 1, acc, &device)?
    } else {
        // (M,K) x (K,N): `offs` walks the *contraction*, so the groups do not
        // share an output -- each one is its own matrix and they stack.
        // A group whose slice is empty contributes a zero matrix, which is
        // what a length-zero contraction means and what upstream returns.
        let (m, n) = (a_dims[0], b_dims[1]);
        let extent = a_dims[1].min(b_dims[0]);
        let mut prev = 0usize;
        let mut planes: Vec<Tensor> = Vec::with_capacity(groups.len());
        for &off in &groups {
            let end = if off < 0 {
                0
            } else {
                (off as usize).min(extent)
            };
            planes.push(if end > prev {
                let depth = end - prev;
                let lhs = a.narrow(1, prev, depth).map_err(|e| candle_err(OP, e))?;
                let rhs = b.narrow(0, prev, depth).map_err(|e| candle_err(OP, e))?;
                grouped_mm_block(OP, &lhs, &rhs, acc)?
            } else {
                Tensor::zeros((m, n), acc, &device).map_err(|e| candle_err(OP, e))?
            });
            prev = end;
        }
        if planes.is_empty() {
            Tensor::zeros((0usize, m, n), acc, &device).map_err(|e| candle_err(OP, e))?
        } else {
            Tensor::stack(&planes, 0).map_err(|e| candle_err(OP, e))?
        }
    };

    let out = out.fast_to(storage).map_err(|e| candle_err(OP, e))?;
    finish(py, out, mat_a.tag())
}

/// Join the per-run blocks back into one output, with the degenerate extent
/// spelled out rather than left to `cat`.
///
/// `Tensor::cat` refuses an empty slice, and a zero-extent output produces
/// exactly that -- `offs=[]` against a zero-batch `mat2` is a shape upstream
/// accepts.
fn grouped_mm_concat(
    op: &str,
    blocks: Vec<Tensor>,
    shape: (usize, usize),
    dim: usize,
    acc: candle_core::DType,
    device: &Device,
) -> PyResult<Tensor> {
    if blocks.is_empty() {
        return Tensor::zeros(shape, acc, device).map_err(|e| candle_err(op, e));
    }
    if blocks.len() == 1 {
        return Ok(blocks.into_iter().next().expect("length checked"));
    }
    Tensor::cat(&blocks, dim).map_err(|e| candle_err(op, e))
}

/// `beta`/`alpha` for `addmm`, applied in the *result* dtype.
///
/// The truncation is upstream's, measured: `addmm` on `int64` operands with
/// `alpha=1.9` gives the same answer as `alpha=1`, and with `alpha=-1.9` the
/// same as `alpha=-1`. torch converts the `Scalar` to `scalar_t` before it
/// multiplies, so a fractional factor on an integral matmul is silently
/// truncated toward zero -- including `beta=0.5`, which rounds to `0` and
/// drops `self` entirely.
fn addmm_scale(
    op: &str,
    tensor: &Tensor,
    factor: Scalar,
    storage: candle_core::DType,
) -> PyResult<Tensor> {
    if storage.is_int() {
        let k = factor.as_i64();
        if k == 1 {
            return Ok(tensor.clone());
        }
        let scale = Tensor::full(k, (), tensor.device())
            .and_then(|t| t.fast_to(storage))
            .map_err(|e| candle_err(op, e))?;
        tensor.broadcast_mul(&scale).map_err(|e| candle_err(op, e))
    } else {
        let k = factor.as_f64();
        if k == 1.0 {
            return Ok(tensor.clone());
        }
        tensor.affine(k, 0.0).map_err(|e| candle_err(op, e))
    }
}

/// `aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1,
///     Scalar alpha=1) -> Tensor`
///
/// `beta * self + alpha * (mat1 @ mat2)`, and the reason it exists as its own
/// op rather than as `mm` + `add` is docs/NN_SURFACE.md §5: `at::native::linear`
/// emits `addmm` for every `bias=True` branch, so a shim without it makes
/// `nn.Linear` take a path upstream would not take. `bootstrap.py` already
/// reads `_aten_all_implemented()` to decide, so landing this kernel is what
/// retires that patch -- nothing in the Python layer has to change.
///
/// Everything below is measured against torch 2.13.0, and three of the four
/// surprises would have been got wrong by inference:
///
///   * **`beta == 0` and `alpha == 0` are quick returns, not multiplications.**
///     `addmm(full(nan), m, n, beta=0)` gives a clean product, and
///     `addmm(b, m_with_inf, n, alpha=0)` gives a clean `b` -- so a literal
///     `0.0 * nan` would produce NaN where torch produces a number. Both
///     branches are therefore skipped, not scaled.
///   * **`self` is validated even when `beta == 0`.** A wrongly-shaped `self`
///     raises with `beta=0, alpha=0`, where nothing reads it. The expand check
///     runs unconditionally here for that reason.
///   * **no promotion, and the two checks are asymmetric.** `mat1` is compared
///     to `mat2` first ("mat1 and mat2 must have the same dtype"), then `self`
///     to `mat2` ("self and mat2 must have the same dtype") -- `mat2` is the
///     reference in both messages.
///   * `bool` and the wide unsigned dtypes have no `addmm_impl_cpu_` upstream.
///
/// The dtype gap this inherits is `mm`'s, not a new one: candle's `matmul` has
/// no integral kernel, so `int64`/`int32`/`int16`/`uint8`/`bfloat16` refuse
/// here exactly where `aten.mm.default` already refuses (docs/TORCH_C.md §2),
/// *except* when `alpha == 0` -- then no matmul happens and the answer comes
/// out. That asymmetry is deliberate: it is the quick return above, and
/// refusing it would be inventing a restriction torch does not have.
fn addmm_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.addmm.default";

    let bias = tensor_arg(OP, args, kwargs, 0, "self")?;
    let mat1 = tensor_arg(OP, args, kwargs, 1, "mat1")?;
    let mat2 = tensor_arg(OP, args, kwargs, 2, "mat2")?;

    if mat1.tensor()?.rank() != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "mat1 must be a matrix, got {}-D tensor",
            mat1.tensor()?.rank()
        )));
    }
    if mat2.tensor()?.rank() != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "mat2 must be a matrix, got {}-D tensor",
            mat2.tensor()?.rank()
        )));
    }
    if mat1.tag() != mat2.tag() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "mat1 and mat2 must have the same dtype, but got {} and {}",
            scalar_type_name(mat1.tag()),
            scalar_type_name(mat2.tag())
        )));
    }
    if bias.tag() != mat2.tag() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "self and mat2 must have the same dtype, but got {} and {}",
            scalar_type_name(bias.tag()),
            scalar_type_name(mat2.tag())
        )));
    }

    let tag = mat2.tag();
    if matches!(
        tag,
        TorchDType::Bool | TorchDType::UInt16 | TorchDType::UInt32 | TorchDType::UInt64
    ) {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"addmm_impl_cpu_\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }

    let a = mat1.tensor()?.dims().to_vec();
    let b = mat2.tensor()?.dims().to_vec();
    if a[1] != b[0] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "mat1 and mat2 shapes cannot be multiplied ({}x{} and {}x{})",
            a[0], a[1], b[0], b[1]
        )));
    }
    let target = vec![a[0], b[1]];

    // `self` expanded to the product's shape, with torch's own two messages
    // for the two ways that fails. Done here rather than left to candle so the
    // caller reads the same sentence upstream would have given.
    let self_dims = bias.tensor()?.dims().to_vec();
    if self_dims.len() > target.len() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "expand(torch.{}Tensor{{{:?}}}, size={:?}): the number of sizes provided ({}) \
             must be greater or equal to the number of dimensions in the tensor ({})",
            scalar_type_name(tag),
            self_dims,
            target,
            target.len(),
            self_dims.len()
        )));
    }
    let offset = target.len() - self_dims.len();
    for (i, &extent) in self_dims.iter().enumerate() {
        let wanted = target[i + offset];
        if extent != wanted && extent != 1 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "The expanded size of the tensor ({wanted}) must match the existing size \
                 ({extent}) at non-singleton dimension {}.  Target sizes: {:?}.  \
                 Tensor sizes: {:?}",
                i + offset,
                target,
                self_dims
            )));
        }
    }

    let storage = PyDtype::new(tag).storage(OP)?;
    let beta = scalar_arg(OP, args, kwargs, 3, "beta")?.unwrap_or(Scalar::Int(1));
    let alpha = scalar_arg(OP, args, kwargs, 4, "alpha")?.unwrap_or(Scalar::Int(1));
    // Zero is decided in the *result* dtype, which is why `beta=0.5` on an
    // integral addmm counts as zero: it truncates to 0 before it multiplies.
    let (beta_zero, alpha_zero) = if storage.is_int() {
        (beta.as_i64() == 0, alpha.as_i64() == 0)
    } else {
        (beta.as_f64() == 0.0, alpha.as_f64() == 0.0)
    };

    // The whole body runs in the accumulation dtype and is narrowed once at the
    // end, which is upstream's shape: `addmm(half ...)` is bitwise equal to
    // `half(addmm(float ...))` (measured). Narrowing the product first and then
    // adding the bias in `float16` would round twice where torch rounds once.
    // See `gemm_accumulate_in`.
    let acc_dtype = gemm_accumulate_in(storage);
    let mut acc: Option<Tensor> = None;
    if !alpha_zero {
        let mat2_inner = mat2.tensor()?;
        let product = widen_gemm_operand(mat1.tensor()?, acc_dtype)
            .and_then(|l| {
                widen_gemm_operand(mat2_inner, acc_dtype)
                    .and_then(|r| gemm_with_layout_fallback(&l, &r, |a, b| a.matmul(b)))
            })
            .map_err(|e| candle_err(OP, e))?;
        acc = Some(addmm_scale(OP, &product, alpha, acc_dtype)?);
    }
    if !beta_zero {
        let expanded = bias
            .tensor()?
            .to_dtype(acc_dtype)
            .and_then(|t| t.broadcast_as(target.as_slice()))
            .and_then(|t| t.contiguous())
            .map_err(|e| candle_err(OP, e))?;
        let scaled = addmm_scale(OP, &expanded, beta, acc_dtype)?;
        acc = Some(match acc {
            Some(product) => product.add(&scaled).map_err(|e| candle_err(OP, e))?,
            None => scaled,
        });
    }
    let out = match acc {
        Some(tensor) => tensor.fast_to(storage).map_err(|e| candle_err(OP, e))?,
        // Both factors zero: torch still answers with a correctly shaped,
        // correctly typed tensor of zeros rather than raising.
        None => Tensor::zeros(target.as_slice(), storage, mat1.tensor()?.device())
            .map_err(|e| candle_err(OP, e))?,
    };
    finish(py, out, tag)
}

/// `aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1,
///     Scalar alpha=1) -> Tensor`
///
/// `bmm`'s batching composed with `addmm`'s `beta * self + alpha * (batch1 @
/// batch2)`, needed to open `bloom` (docs/TAIL.md) -- its attention builds the
/// scaled QK^T scores with this one op rather than a separate scale-then-add.
///
/// The `beta=0` quick return is `addmm`'s, reused batched and re-measured to
/// still hold: `baddbmm(self, b1, b2, beta=0)` on a `nan`-filled `self` gives
/// a clean product (no `nan` leaks through the skipped add). `alpha=0` is
/// **not** the mirror-image quick return, despite the kernel used to claim
/// so and despite `addmm` itself skipping the multiply on `alpha=0` --
/// measured against torch 2.13.0 (docs/TAIL.md §2.1):
/// `baddbmm(zeros_self, inf_batch1, batch2, alpha=0)` comes back with `nan`
/// in it, so upstream still runs the real IEEE multiply and only then scales
/// by zero (`0 * inf == nan`, not skipped). This kernel now does the same:
/// the product is always computed and scaled through `addmm_scale`, and only
/// the `self` term is quick-returned on `beta=0`. One consequence: dtypes
/// candle has no matmul kernel for (`_MM_C_ERROR_DTYPES` in
/// `tools/golden/cases.py`) now raise on `alpha=0` too, where the old
/// unconditional skip used to dodge the missing kernel -- that dodge was
/// itself part of the bug, not a feature to preserve. Refusals are `addmm`'s
/// dtype-mismatch pair too (`batch1`/`batch2` compared first, `self`/`batch2`
/// second), plus `bmm`'s rank checks in place of `mm`'s, since the batch
/// dimension is what `bmm` added and `addmm` never had.
///
/// `self` broadcasts to the `(batch, n, p)` output the same way `addmm`'s
/// bias does -- rank up to 3, trailing dimensions either matching or `1` --
/// so a `(n, p)` or scalar `self` is as valid here as a fully-batched one,
/// matching `nn.Linear`-style bias broadcasting into a batched matmul.
fn baddbmm_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.baddbmm.default";

    let bias = tensor_arg(OP, args, kwargs, 0, "self")?;
    let batch1 = tensor_arg(OP, args, kwargs, 1, "batch1")?;
    let batch2 = tensor_arg(OP, args, kwargs, 2, "batch2")?;

    if batch1.tensor()?.rank() != 3 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "batch1 must be a 3D tensor",
        ));
    }
    if batch2.tensor()?.rank() != 3 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "batch2 must be a 3D tensor",
        ));
    }
    if batch1.tag() != batch2.tag() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "batch1 and batch2 must have the same dtype, but got {} and {}",
            scalar_type_name(batch1.tag()),
            scalar_type_name(batch2.tag())
        )));
    }
    if bias.tag() != batch2.tag() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "self and batch2 must have the same dtype, but got {} and {}",
            scalar_type_name(bias.tag()),
            scalar_type_name(batch2.tag())
        )));
    }

    let tag = batch2.tag();
    if matches!(
        tag,
        TorchDType::Bool | TorchDType::UInt16 | TorchDType::UInt32 | TorchDType::UInt64
    ) {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"baddbmm\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }

    let a = batch1.tensor()?.dims().to_vec();
    let b = batch2.tensor()?.dims().to_vec();
    if a[0] != b[0] || a[2] != b[1] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Expected size for first two dimensions of batch2 tensor to be: \
             [{}, {}] but got: [{}, {}].",
            a[0], a[2], b[0], b[1]
        )));
    }
    let target = vec![a[0], a[1], b[2]];

    // Same expand check `addmm_default` runs on its bias, generalised to a
    // 3-D target.
    let self_dims = bias.tensor()?.dims().to_vec();
    if self_dims.len() > target.len() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "expand(torch.{}Tensor{{{:?}}}, size={:?}): the number of sizes provided ({}) \
             must be greater or equal to the number of dimensions in the tensor ({})",
            scalar_type_name(tag),
            self_dims,
            target,
            target.len(),
            self_dims.len()
        )));
    }
    let offset = target.len() - self_dims.len();
    for (i, &extent) in self_dims.iter().enumerate() {
        let wanted = target[i + offset];
        if extent != wanted && extent != 1 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "The expanded size of the tensor ({wanted}) must match the existing size \
                 ({extent}) at non-singleton dimension {}.  Target sizes: {:?}.  \
                 Tensor sizes: {:?}",
                i + offset,
                target,
                self_dims
            )));
        }
    }

    let storage = PyDtype::new(tag).storage(OP)?;
    let beta = scalar_arg(OP, args, kwargs, 3, "beta")?.unwrap_or(Scalar::Int(1));
    let alpha = scalar_arg(OP, args, kwargs, 4, "alpha")?.unwrap_or(Scalar::Int(1));
    // `alpha` has no quick return (see the kernel doc above) so only
    // `beta`'s zero-ness is decided ahead of time.
    let beta_zero = if storage.is_int() {
        beta.as_i64() == 0
    } else {
        beta.as_f64() == 0.0
    };

    // Same accumulation-dtype rule as `mm`/`bmm`/`addmm` -- see
    // `gemm_accumulate_in`.
    let acc_dtype = gemm_accumulate_in(storage);
    // Unlike `addmm`, `alpha == 0` is NOT a quick return here -- measured
    // against torch 2.13.0 (docs/TAIL.md §2.1): the multiply still runs and
    // its NaNs/Infs still leak through, only the *scale* is skipped. So the
    // product is always computed; `addmm_scale` folds the `alpha == 1` case
    // back down to a plain clone, same as it always did.
    //
    // The `?` on each `tensor()` is the meta change: a meta tensor has no bytes
    // to read, and the type says so rather than each kernel remembering to.
    let batch2_inner = batch2.tensor()?;
    let product = widen_gemm_operand(batch1.tensor()?, acc_dtype)
        .and_then(|l| {
            widen_gemm_operand(batch2_inner, acc_dtype)
                .and_then(|r| gemm_with_layout_fallback(&l, &r, |a, b| a.matmul(b)))
        })
        .map_err(|e| candle_err(OP, e))?;
    let mut acc: Option<Tensor> = Some(addmm_scale(OP, &product, alpha, acc_dtype)?);
    if !beta_zero {
        let expanded = bias
            .tensor()?
            .to_dtype(acc_dtype)
            .and_then(|t| t.broadcast_as(target.as_slice()))
            .and_then(|t| t.contiguous())
            .map_err(|e| candle_err(OP, e))?;
        let scaled = addmm_scale(OP, &expanded, beta, acc_dtype)?;
        acc = Some(match acc {
            Some(product) => product.add(&scaled).map_err(|e| candle_err(OP, e))?,
            None => scaled,
        });
    }
    let out = match acc {
        Some(tensor) => tensor.fast_to(storage).map_err(|e| candle_err(OP, e))?,
        None => Tensor::zeros(target.as_slice(), storage, batch1.tensor()?.device())
            .map_err(|e| candle_err(OP, e))?,
    };
    finish(py, out, tag)
}

/// `aten::_scaled_dot_product_flash_attention_for_cpu(Tensor query, Tensor key,
///     Tensor value, float dropout_p=0., bool is_causal=False, *,
///     Tensor? attn_mask=None, float? scale=None) -> (Tensor, Tensor)`
///
/// The one fused op in this file, and the only one that answers with a pair of
/// tensors rather than one. Everything about it below was measured against
/// torch 2.13.0 rather than inferred, because the name says "flash attention"
/// and the observable contract is not what that name suggests:
///
///   * `is_causal` is **upper-left aligned**, not bottom-right: row `t`
///     attends keys `0..=t` even when the key sequence is longer than the
///     query one. Measured on a (q=2, kv=5) pair -- the bottom-right reading
///     disagrees on every element.
///   * `is_causal` and `attn_mask` **compose**. `F.scaled_dot_product_attention`
///     refuses to take both; this aten op accepts both and adds them, measured.
///   * the second result is `logsumexp` over the *masked, scaled* scores, so
///     it has to be computed after both masks land, not from the raw product.
///   * for `float16`/`bfloat16` inputs the output comes back in the input
///     dtype but the logsumexp comes back **`float32`** -- which is upstream
///     telling us the accumulation happens in float. This follows that:
///     reduced-precision inputs are widened to `f32` for the whole body and
///     only the output is narrowed again.
///
/// `dropout_p > 0` is refused here because upstream refuses it too ("Currently
/// do not support dropout > 0"), not because the shim lacks an RNG.
///
/// **Grouped-query attention lives here, not in the wrapper.** See
/// `repeat_kv_heads`.
///
/// Key and value are repeated to the query's head count *before* anything
/// else touches them, so `is_causal`, `attn_mask` and the logsumexp all see
/// the full head count. That order is not cosmetic: an `attn_mask` is indexed
/// by `(batch, head, row, col)` and upstream's has the *query* head count, so
/// repeating after masking would broadcast a per-head mask onto the wrong
/// heads.
///
/// **There are two arithmetics here, and the default is not the exact one.**
/// The body below forks on `crate::flash::reference_enabled()`:
///
///   * **default** -- candle's tensor ops, written out the textbook way: one
///     matmul, one softmax, another matmul. That is the right *mathematics*
///     and it is upstream's *arithmetic* only to within a tolerance, because
///     upstream's kernel is blocked with an online softmax and the order in
///     which it recombines those blocks is observable. The flat formulation
///     disagrees with upstream on 3562 of 4096 elements at `float32`
///     (docs/SDPA.md §3) and on 32% of the attention outputs of a real
///     SmolLM2-135M forward -- all inside the golden harness's tolerance.
///   * **reference** -- `crate::flash`, which reproduces the blocking and
///     brings both of those numbers to 0 for `bfloat16`/`float16`. Not 0 for
///     `float32`/`float64`, where upstream calls a BLAS: docs/SDPA.md §5 has
///     that split. It costs **20x** at T=512 (docs/SDPA.md §12), which is why
///     it is not the default in a library whose point is running on a phone.
///
/// The blocking the reference reproduces is upstream's, not a choice: 32 query
/// rows (64 or 256 for longer queries) by 512 key columns, and the mask's
/// fused multiply-add strides by the *mask* dtype's vector width. Both are the
/// kind of detail that no tolerance can check, so `pytests/test_shim.py`
/// checks them with none -- with the switch on.
///
/// Nothing above the fork differs between the two: the argument checks, the
/// dtype rules, the widening, the GQA repetition and the default `scale` are
/// one copy, so the switch cannot change what this op *accepts*, only how it
/// adds up.
fn sdpa_flash_cpu(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._scaled_dot_product_flash_attention_for_cpu.default";

    let query = tensor_arg(OP, args, kwargs, 0, "query")?;
    let key = tensor_arg(OP, args, kwargs, 1, "key")?;
    let value = tensor_arg(OP, args, kwargs, 2, "value")?;
    let dropout_p = float_arg(args, kwargs, 3, "dropout_p", 0.0)?;
    let is_causal = bool_arg(args, kwargs, 4, "is_causal")?.unwrap_or(false);
    let attn_mask = match optional(args, kwargs, 5, "attn_mask")? {
        Some(value) if !value.is_none() => Some(value.extract::<PyTensorBase>()?),
        _ => None,
    };
    let scale = match optional(args, kwargs, 6, "scale")? {
        Some(value) if !value.is_none() => Some(value.extract::<f64>()?),
        _ => None,
    };

    require_same_dtype(OP, &query, &key)?;
    let tag = require_same_dtype(OP, &query, &value)?;
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "scaled_dot_product_attention_flash_attention: Expected data type in \
             FP32, FP64, BF16, FP16, but got {} instead.",
            scalar_type_name(tag)
        )));
    }
    if dropout_p > 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "scaled_dot_product_attention_flash_attention: Currently do not support dropout > 0",
        ));
    }
    for operand in [&query, &key, &value] {
        if operand.tensor()?.rank() != 4 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "scaled_dot_product_attention_flash_attention: Accept only 4 dims inputs \
                 shape of {B, H, T, K}",
            ));
        }
    }
    if let Some(mask) = attn_mask.as_ref() {
        if mask.tag() != tag {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "scaled_dot_product_attention_flash_attention: Attention mask is the same \
                 data type as query",
            ));
        }
    }

    let storage = PyDtype::new(tag).storage(OP)?;
    // The widening upstream's `float32` logsumexp reports (see the doc comment).
    let acc = match storage {
        candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
        other => other,
    };
    let acc_tag = TorchDType::from_storage(acc).ok_or_else(|| {
        not_implemented(format!("{OP}: no torch dtype for the accumulate type {acc:?}"))
    })?;

    let widen = |t: &Tensor| t.fast_to(acc).and_then(|t| t.contiguous());
    let q = widen(query.tensor()?).map_err(|e| candle_err(OP, e))?;
    let k = repeat_kv_heads(OP, &widen(key.tensor()?).map_err(|e| candle_err(OP, e))?, q.dims()[1])?;
    let v = repeat_kv_heads(
        OP,
        &widen(value.tensor()?).map_err(|e| candle_err(OP, e))?,
        q.dims()[1],
    )?;

    let head_dim = q.dims()[3];
    let scale = scale.unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt());

    // The fork. One relaxed atomic load per call, against a kernel whose
    // cheapest measured shape is tens of microseconds -- see
    // `crate::flash::reference_enabled` for why the default path cannot feel
    // it, and docs/SDPA.md §12 for why the exact kernel is not the default.
    let (out, logsumexp) = if crate::flash::reference_enabled() {
        // ------------------------------------------------------------------
        // Reference: `crate::flash`, upstream's blocked kernel reproduced.
        // Bit-identical to upstream for bfloat16/float16, and 20x slower.
        // ------------------------------------------------------------------
        let (batch, heads, q_len) = {
            let dims = q.dims();
            (dims[0], dims[1], dims[2])
        };
        let kv_len = k.dims()[2];
        // `crate::flash` walks raw slices, so a shape that does not describe
        // one attention would index past a buffer rather than raise. The
        // default branch gets the same rejection from candle's `matmul`, which
        // is why this check is here rather than above the fork: it is not a
        // second opinion, it is the same opinion that branch already has.
        for (name, operand) in [("key", &k), ("value", &v)] {
            let dims = operand.dims();
            if dims[0] != batch || dims[2] != kv_len || dims[3] != head_dim {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "{OP}: query {:?} and {name} {:?} do not describe one attention -- \
                     batch, key length and head dimension all have to agree",
                    q.dims(),
                    dims
                )));
            }
        }

        // The mask is broadcast to the full `[batch, head, q_len, kv_len]`
        // here rather than inside the kernel, because the kernel reads one
        // contiguous row of it per query row -- upstream materialises it the
        // same way. `to_dtype` and not `fast_to`: this is the branch whose
        // bits are asserted with no tolerance, so it keeps the spelling those
        // assertions were measured against.
        let mask = match attn_mask.as_ref() {
            Some(mask) => Some(
                mask.tensor()?
                    .to_dtype(acc)
                    .and_then(|m| m.broadcast_as((batch, heads, q_len, kv_len)))
                    .and_then(|m| m.contiguous())
                    .map_err(|e| candle_err(OP, e))?,
            ),
            None => None,
        };

        // Which storage dtype the probabilities are narrowed to between the
        // two matrix products. `storage`, not `acc`: the narrowing follows the
        // *input* dtype, which is the whole reason reduced precision has a
        // separate path upstream. See `crate::flash` and docs/SDPA.md.
        let narrowing = match storage {
            candle_core::DType::BF16 => crate::flash::Narrowing::BFloat16,
            candle_core::DType::F16 => crate::flash::Narrowing::Float16,
            _ => crate::flash::Narrowing::None,
        };
        let attended = match acc {
            candle_core::DType::F64 => {
                crate::flash::attend::<f64>(&q, &k, &v, mask.as_ref(), scale, is_causal, narrowing)
            }
            _ => {
                crate::flash::attend::<f32>(&q, &k, &v, mask.as_ref(), scale, is_causal, narrowing)
            }
        };
        let (out, logsumexp) = attended.map_err(|e| candle_err(OP, e))?;
        // Exact: the kernel already narrowed every element it wrote.
        let out = out.to_dtype(storage).map_err(|e| candle_err(OP, e))?;
        (out, logsumexp)
    } else {
        // ------------------------------------------------------------------
        // Default: candle's tensor ops. The right mathematics, and an
        // arithmetic that is upstream's only to within a tolerance.
        // ------------------------------------------------------------------
        let raw = k
            .transpose(2, 3)
            // The `contiguous` is not decoration. Dropping it -- letting candle
            // hand the transposed layout straight to Accelerate as a
            // `transa='T'` GEMM, which it does support -- works, is 5% faster
            // at S=512, and **changes the answer at S=6**: the prefill digest
            // moved, measured. A different GEMM blocking is a different
            // summation order. docs/SEQLEN.md §8.5.
            //
            // What changed (docs/KERNELS26.md §7) is *how* the copy is made,
            // not whether it is made. candle's `copy_strided_src` walks a
            // transposed layout one element at a time, recomputing a
            // multi-dimensional index per element and reading `head_dim`
            // floats away from the previous one -- 2.4 MB at ~3.7 GB/s,
            // against upstream's 0.134 ms for the same bytes.
            // `transposed_contiguous` is the same copy in 32x32 cache blocks.
            //
            // **It is bit-identical by construction, not by tolerance**: every
            // output element is a copy of exactly one input element, so there
            // is no summation order to reassociate and no rounding to move.
            // The only thing that changed is the order in which the same
            // assignments happen. That is the whole difference between this
            // and the rejected change above, and it is why the prefill digests
            // hold at every length docs/SEQLEN.md §1.3 pins.
            .and_then(|kt| crate::tensor::transposed_contiguous(&kt))
            .and_then(|kt| q.matmul(&kt))
            .map_err(|e| candle_err(OP, e))?;

        // `affine(scale, 0.0)` and then a `broadcast_add` of an upper-
        // triangular `-inf` mask was three full passes over
        // `[batch, head, S, S]` plus an `S x S` `Vec<f64>` rebuilt on every one
        // of the thirty calls a forward makes -- 5.68 ms of a 21.2 ms call at
        // S=1024, measured inside the op. One pass now, and nothing allocated
        // but the output.
        //
        // The arithmetic per element is the same arithmetic: element-wise work
        // has no summation order to reassociate, and
        // `tensor.rs::scale_and_causal_mask` is checked bit-for-bit against the
        // two-op spelling it replaces -- including the `+ 0.0` that turns a
        // negative-zero product positive and the `+ -inf` that turns a positive
        // infinity into a NaN. docs/SEQLEN.md §8.3.
        let mut scores = if is_causal {
            // Upper-left aligned, per the measurement above.
            crate::tensor::scale_and_causal_mask(&raw, scale).map_err(|e| candle_err(OP, e))?
        } else {
            raw.affine(scale, 0.0).map_err(|e| candle_err(OP, e))?
        };
        if let Some(mask) = attn_mask.as_ref() {
            let mask = mask
                .tensor()?
                .fast_to(acc)
                .map_err(|e| candle_err(OP, e))?;
            scores = scores.broadcast_add(&mask).map_err(|e| candle_err(OP, e))?;
        }

        // Softmax written out: candle-core has no `softmax` (that lives in
        // candle-nn, which DESIGN.md §4 does not pull in). Shifting by the row
        // maximum first is not an optimisation -- without it a masked row's
        // `exp(-inf)` and a large logit's `exp(big)` land on the same NaN.
        //
        // `amax_keepdim`, not candle's `max_keepdim`: this wants a maximum
        // *value* and candle only has the reduction that also computes an
        // argmax, which measured 57x slower than upstream's `amax` at the score
        // shape this very line produces and was 24.3% of a `float32` prefill.
        // docs/SEQLEN.md §7. The two agree bit for bit on every input this line
        // can produce -- §7.2 has the argument, and it is an argument rather
        // than a tolerance.
        let row_max = crate::tensor::amax_keepdim(&scores, 3).map_err(|e| candle_err(OP, e))?;
        let weights = scores
            .broadcast_sub(&row_max)
            .and_then(|s| s.exp())
            .map_err(|e| candle_err(OP, e))?;
        let row_sum = weights.sum_keepdim(3).map_err(|e| candle_err(OP, e))?;
        let out = weights
            .broadcast_div(&row_sum)
            .and_then(|p| p.contiguous())
            .and_then(|p| p.matmul(&v))
            .and_then(|o| o.fast_to(storage))
            .map_err(|e| candle_err(OP, e))?;

        // logsumexp(x) = max(x) + log(sum(exp(x - max(x)))), on the same
        // masked, scaled scores the weights came from.
        let logsumexp = row_sum
            .log()
            .and_then(|l| l.broadcast_add(&row_max))
            .and_then(|l| l.squeeze(3))
            .map_err(|e| candle_err(OP, e))?;
        (out, logsumexp)
    };

    // Promoted element by element: `promote` at the dispatcher's exit does not
    // look inside a tuple, the same reason `max.dim` promotes its own pair.
    let pair = [
        crate::tensor::promote(py, finish(py, out, tag)?)?,
        crate::tensor::promote(py, finish(py, logsumexp, acc_tag)?)?,
    ];
    Ok(PyTuple::new(py, pair)?.into_any().unbind())
}

// ---------------------------------------------------------------------------
// The ops C_SURFACE.md measured a Llama forward + greedy `generate` actually
// calling. That document counted 13 of `_VariableFunctions`' 609 hoisted names
// (2.1%) as *called* rather than merely looked up, and this is that list.
//
// They are here because the overload resolver above now routes `torch.<op>` to
// them; before it existed the only reachable spelling was
// `torch.ops.aten.<op>.<overload>`, which no user-facing code writes.
//
// Every dtype rule below was measured against torch 2.13.0, not inferred. The
// ones that would have been got wrong by inference are called out at the site.
// ---------------------------------------------------------------------------

/// Which of the three `arange` overloads is being served. They differ only in
/// which of `start`/`step` the caller supplied, so one body covers all three
/// -- but the *key* stays distinct, because torch really does send
/// `arange(0, 5)` to `arange.start` and `arange(0, 5, 2)` to
/// `arange.start_step` (measured), and collapsing them would make the work
/// queue report an op that was never called.
#[derive(Clone, Copy)]
enum ArangeForm {
    End,
    Start,
    StartStep,
}

fn arange(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    form: ArangeForm,
) -> PyResult<Py<PyAny>> {
    let (op, start_at, end_at, step_at, options_at) = match form {
        ArangeForm::End => ("aten.arange.default", None, 0usize, None, 1usize),
        ArangeForm::Start => ("aten.arange.start", Some(0usize), 1, None, 2),
        ArangeForm::StartStep => ("aten.arange.start_step", Some(0usize), 1, Some(2usize), 3),
    };

    let start = match start_at {
        Some(index) => scalar_arg(op, args, kwargs, index, "start")?.unwrap_or(Scalar::Int(0)),
        None => Scalar::Int(0),
    };
    let end = scalar_arg(op, args, kwargs, end_at, "end")?
        .ok_or_else(|| missing(op, "end"))?;
    let step = match step_at {
        Some(index) => scalar_arg(op, args, kwargs, index, "step")?.unwrap_or(Scalar::Int(1)),
        None => Scalar::Int(1),
    };

    // torch: an integral start/end/step gives int64, anything else gives the
    // default float dtype. Not "the widest of the three" -- the categories are
    // what matter, so a single float argument floats the whole result.
    let integral = start.is_int() && end.is_int() && step.is_int();
    let dtype = dtype_arg(args, kwargs, options_at, "dtype")?
        .unwrap_or(if integral { TorchDType::Int64 } else { default_float() });
    reject_unsupported(
        op,
        args,
        kwargs,
        &[(options_at + 1, "layout"), (options_at + 3, "pin_memory")],
    )?;
    let label = device_arg_or_label(args, kwargs, options_at + 2, "device", &PyDevice::cpu())?;
    if label.is_meta() {
        // Length before dtype: `arange` is the one factory whose shape is
        // computed rather than given, and `arange_length` is the same
        // arithmetic the dense path below does -- shared, not restated, so the
        // two cannot disagree about how many elements the real call would have
        // produced.
        //
        // `arange_has_cpu_kernel` is deliberately *not* consulted here. It
        // reproduces an upstream CPU gap, and upstream's meta kernel does not
        // have that gap: `torch.arange(5, dtype=torch.uint16, device="meta")`
        // is a tensor while the CPU spelling raises. Measured.
        let n = arange_length(op, &start, &end, &step, integral)?;
        return meta_result(py, vec![n], dtype);
    }
    let device = label.resolve()?;
    let storage = PyDtype::new(dtype).storage(op)?;

    // torch has no `arange_cpu` kernel for these, and the golden harness
    // caught the shim computing an answer where torch refuses. Reproducing an
    // upstream *gap* rather than filling it is the same call docs/IMPORT_TORCH
    // §7 made for `full`'s numel==1 hole: the harness compares against torch,
    // so a shim that is more capable than torch diverges just as loudly as one
    // that is less capable, only in the other direction. Measured, not
    // reasoned: torch 2.13.0 refuses uint16/uint32/uint64, bool and the float8
    // family, and accepts everything else this shim can store.
    if !arange_has_cpu_kernel(dtype) {
        return Err(not_implemented(format!(
            "\"arange_cpu\" not implemented for '{}'",
            scalar_type_name(dtype)
        )));
    }

    // The zero-step and sign refusals live in `arange_length` -- one copy, so
    // the meta path cannot skip a check the dense path makes.
    let (s, d) = (start.as_f64(), step.as_f64());
    let n = arange_length(op, &start, &end, &step, integral)? as i64;
    let tensor = if integral {
        let (s, d) = (start.as_i64(), step.as_i64());
        // `s + i * d`, never an accumulator: candle's own `arange_step` adds
        // repeatedly, which drifts on floats and is the kind of divergence the
        // golden harness exists to catch.
        let values: Vec<i64> = (0..n).map(|i| s + i * d).collect();
        let len = values.len();
        Tensor::from_vec(values, len, &device)
    } else {
        let values: Vec<f64> = (0..n).map(|i| s + (i as f64) * d).collect();
        let len = values.len();
        Tensor::from_vec(values, len, &device)
    }
    .and_then(|t| t.fast_to(storage))
    .map_err(|err| candle_err(op, err))?;

    finish(py, tensor, dtype)
}

/// How many elements `arange(start, end, step)` produces.
///
/// Split out of the kernel because the meta path needs the count without the
/// values, and a second copy of this arithmetic is exactly the kind of thing
/// that drifts: the integral branch is not `ceil((e-s)/d)` in floating point,
/// it is an integer ceiling-divide, and the two disagree at the boundaries the
/// golden harness already pins.
///
/// The sign and zero-step checks stay in the kernel: they are refusals, and a
/// meta call has to make the same ones before it can claim a shape.
fn arange_length(
    op: &str,
    start: &Scalar,
    end: &Scalar,
    step: &Scalar,
    integral: bool,
) -> PyResult<usize> {
    if step.as_f64() == 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("step must be nonzero"));
    }
    let (fs, fe, fd) = (start.as_f64(), end.as_f64(), step.as_f64());
    if (fd > 0.0 && fe < fs) || (fd < 0.0 && fe > fs) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "upper bound and lower bound inconsistent with step sign",
        ));
    }
    let _ = op;
    let n = if integral {
        let (s, e, d) = (start.as_i64(), end.as_i64(), step.as_i64());
        if d > 0 {
            (e - s + d - 1).div_euclid(d).max(0)
        } else {
            (s - e + (-d) - 1).div_euclid(-d).max(0)
        }
    } else {
        (((fe - fs) / fd).ceil()).max(0.0) as i64
    };
    Ok(n as usize)
}

/// `aten::linspace(Scalar start, Scalar end, int steps, *, ScalarType?
///     dtype=None, Layout? layout=None, Device? device=None, bool?
///     pin_memory=None) -> Tensor`
///
/// docs/DEMAND.md §0.1 rank 4 -- `ConvNextModel.__init__`'s stochastic-depth
/// rate schedule, a construction-time wall rather than a forward one. Leaf
/// upstream (`RangeFactoriesKernel.cpp::linspace_kernel`, fetched and read
/// rather than guessed, since three of its details are easy to get wrong by
/// reasoning alone):
///
/// * **`steps=0` is not an error -- it answers the empty tensor.** `steps<0`
///   is upstream's own `"number of steps must be non-negative"`. `steps=1`
///   answers `[start]` (measured: `linspace(3, 999, 1) == [3.0]`, `end`
///   never contributes).
/// * **The default dtype is always the default float dtype**, never
///   `int64` the way `arange`'s all-integral rule would suggest --
///   `linspace(0, 10, 5)` is `float32` even though every argument is an
///   `int`. `dtype=torch.int64` is accepted and **truncates toward zero,
///   not rounds** (`linspace(0, 9, 5, dtype=int64) == [0, 2, 4, 6, 9]`, from
///   the exact schedule `0, 2.25, 4.5, 6.75, 9`).
/// * **The endpoint is exact by construction, not by luck.** Upstream fills
///   the first half of the output forward from `start` (`start +
///   step*i`) and the second half *backward* from `end` (`end -
///   step*(steps-1-i)`), split at `steps/2`, specifically so `end` never
///   accumulates floating error over `steps-1` additions. The `Float`/
///   `Double` arms fold each half's multiply-add into one hardware fused
///   op -- measured: a plain `start + step*i` in matching precision
///   disagrees with upstream on the least-significant bit often enough
///   that a golden case would catch it, and `f32`/`f64::mul_add` (this
///   shim's own FMA) does not. `Half`/`BFloat16` do **not** get this
///   treatment: `at::Half`/`at::BFloat16` arithmetic promotes to `f32` and
///   narrows back on every single operator call rather than fusing, so
///   those two use `float_narrower` after each individual step instead
///   -- measured separately, the two conventions disagree at exactly the
///   dtype boundary you would expect.
fn linspace_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.linspace.default";
    let start = scalar_arg(OP, args, kwargs, 0, "start")?.ok_or_else(|| missing(OP, "start"))?;
    let end = scalar_arg(OP, args, kwargs, 1, "end")?.ok_or_else(|| missing(OP, "end"))?;
    let steps = int_arg(args, kwargs, 2, "steps")?.ok_or_else(|| missing(OP, "steps"))?;
    let dtype = dtype_arg(args, kwargs, 3, "dtype")?.unwrap_or_else(default_float);
    reject_unsupported(OP, args, kwargs, &[(4, "layout"), (6, "pin_memory")])?;
    let label = device_arg_or_label(args, kwargs, 5, "device", &PyDevice::cpu())?;

    if steps < 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "number of steps must be non-negative",
        ));
    }
    if !linspace_has_cpu_kernel(dtype) {
        return Err(not_implemented(format!(
            "\"linspace_cpu\" not implemented for '{}'",
            scalar_type_name(dtype)
        )));
    }
    if label.is_meta() {
        return meta_result(py, vec![steps as usize], dtype);
    }
    let device = label.resolve()?;
    let storage = PyDtype::new(dtype).storage(OP)?;
    let n = steps as usize;

    let tensor = if dtype.is_floating_point() {
        let values = linspace_float_values(dtype, start.as_f64(), end.as_f64(), n);
        Tensor::from_vec(values, n, &device)
    } else {
        let values = linspace_int_values(start.as_i64(), end.as_i64(), n);
        Tensor::from_vec(values, n, &device)
    }
    .and_then(|t| t.fast_to(storage))
    .map_err(|err| candle_err(OP, err))?;

    finish(py, tensor, dtype)
}

/// Does torch have a `linspace_cpu` kernel for this dtype? Upstream's
/// `AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, ...)` leaves
/// `Bool` and the wider unsigned integers this shim can store (`UInt16`,
/// `UInt32`, `UInt64`) unregistered -- measured:
/// `torch.linspace(0, 10, 5, dtype=torch.bool)` raises `"linspace_cpu" not
/// implemented for 'Bool'`, the same wording reproduced here for the whole
/// unregistered set.
fn linspace_has_cpu_kernel(dtype: TorchDType) -> bool {
    use TorchDType::*;
    !matches!(dtype, Bool | UInt16 | UInt32 | UInt64 | Float8E4M3FN)
}

/// The `Float`/`Double`/`Half`/`BFloat16` arms of upstream's
/// `linspace_kernel` -- see the doc comment on `linspace_default` for why
/// the native two are `mul_add` and the reduced two are not. Returns widened
/// `f64` throughout; every value produced is already exact at `dtype`'s own
/// precision, so the final narrowing `linspace_default` does via `fast_to`
/// is a lossless round-trip, not a second rounding.
fn linspace_float_values(dtype: TorchDType, start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![float_narrower(dtype)(start)];
    }
    let halfway = n / 2;
    match dtype {
        TorchDType::Float32 => {
            let (s, e) = (start as f32, end as f32);
            let step = (e - s) / (n as f32 - 1.0);
            (0..n)
                .map(|i| {
                    if i < halfway {
                        step.mul_add(i as f32, s) as f64
                    } else {
                        (-step).mul_add((n - i - 1) as f32, e) as f64
                    }
                })
                .collect()
        }
        TorchDType::Float64 => {
            let (s, e) = (start, end);
            let step = (e - s) / (n as f64 - 1.0);
            (0..n)
                .map(|i| {
                    if i < halfway {
                        step.mul_add(i as f64, s)
                    } else {
                        (-step).mul_add((n - i - 1) as f64, e)
                    }
                })
                .collect()
        }
        // `Half`/`BFloat16`: no fusion, narrowed after every single op --
        // `at::Half`/`at::BFloat16`'s own arithmetic does the same.
        other => {
            let narrow = float_narrower(other);
            let s = narrow(start);
            let e = narrow(end);
            let step = narrow(narrow(e - s) / narrow(n as f64 - 1.0));
            (0..n)
                .map(|i| {
                    if i < halfway {
                        narrow(s + narrow(step * i as f64))
                    } else {
                        narrow(e - narrow(step * (n - i - 1) as f64))
                    }
                })
                .collect()
        }
    }
}

/// The integral arms: `step_t` is always `double` upstream regardless of the
/// output's own width, and the per-element sum is truncated toward zero into
/// the output type -- never rounded (measured: `linspace(0, 9, 5,
/// dtype=int64)` includes `6`, not `7`, from the exact `6.75`). Kept in
/// `i64` throughout and narrowed once by `linspace_default`'s `fast_to`,
/// the same convention `arange`'s integral arm above uses.
fn linspace_int_values(start: i64, end: i64, n: usize) -> Vec<i64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let halfway = n / 2;
    let step = (end as f64 - start as f64) / (n as f64 - 1.0);
    (0..n)
        .map(|i| {
            let v = if i < halfway {
                step.mul_add(i as f64, start as f64)
            } else {
                (-step).mul_add((n - i - 1) as f64, end as f64)
            };
            v as i64
        })
        .collect()
}

/// `aten::ones(SymInt[] size, *, ScalarType? dtype=None, ...)` and
/// `aten::empty.memory_format(SymInt[] size, *, ...)`.
///
/// **`empty` returns zeros, and that is a divergence worth stating.** torch's
/// `empty` hands back whatever was in the allocation, and makes no promise
/// about it, so zeros satisfy the contract -- but they are not the *same*
/// bytes, and any test that reads an uninitialised tensor and compares against
/// torch is comparing noise. The shim is deterministic here where torch is
/// not, which is the safe direction but still a difference.
fn zeros_or_ones(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    one: bool,
) -> PyResult<Py<PyAny>> {
    let size: Vec<usize> = required(op, args, kwargs, 0, "size")?.extract()?;
    let dtype = dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(default_float());
    reject_unsupported(
        op,
        args,
        kwargs,
        &[(2, "layout"), (4, "pin_memory"), (5, "memory_format")],
    )?;
    let label = device_arg_or_label(args, kwargs, 3, "device", &PyDevice::cpu())?;
    // `meta` before `storage()`, and the order is the claim: a meta tensor
    // never has to be storable by candle, so `torch.empty(2,
    // dtype=torch.complex64, device="meta")` is representable here exactly as
    // it is upstream, while its CPU counterpart is refused by name.
    if label.is_meta() {
        return meta_result(py, size, dtype);
    }
    let device = label.resolve()?;
    let storage = PyDtype::new(dtype).storage(op)?;

    let tensor = if one {
        Tensor::ones(size, storage, &device)
    } else {
        Tensor::zeros(size, storage, &device)
    }
    .map_err(|err| candle_err(op, err))?;
    finish(py, tensor, dtype)
}

fn ones_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    zeros_or_ones(py, args, kwargs, "aten.ones.default", true)
}

fn empty_memory_format(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    zeros_or_ones(py, args, kwargs, "aten.empty.memory_format", false)
}

/// `aten::rsqrt(Tensor self) -> Tensor`
///
/// The dtype rule is torch's unary-float promotion: a floating input keeps its
/// own dtype (`float16` in, `float16` out -- *not* widened to float32), and an
/// integral or boolean input becomes the default float. Both halves measured.
fn rsqrt_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.rsqrt.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = unary_float_tag(input.tag());
    let storage = PyDtype::new(tag).storage(OP)?;
    let tensor = input
        .tensor()?
        .fast_to(storage)
        .and_then(|t| t.sqrt())
        .and_then(|t| t.recip())
        .map_err(|err| candle_err(OP, err))?;
    finish(py, tensor, tag)
}

/// `aten::pow.Tensor_Scalar`, `.Tensor_Tensor` and `.Scalar`.
///
/// Not candle's `Tensor::pow`, which is `exp(exponent * log(base))` and so
/// returns NaN for every negative base -- `torch.pow(t, 2)` on a tensor with
/// negative entries is the RMSNorm path, so that would be wrong on the first
/// real model. `powf` (used for the float scalar case) does go through the
/// real `f64::powf` and is fine; the general case is computed here.
///
/// The dtype rule is torch's "wrapped number" promotion, measured: an integer
/// tensor with an integer exponent stays integral (`pow(int64, 2) -> int64`),
/// and a float on either side floats the result. A Python scalar never widens
/// a tensor of the same category.
///
/// **`tensor` is the *result* category, not necessarily an operand's dtype.**
/// `pow_tensor_tensor` hands in the promotion of its two operands
/// (docs/ARCH20.md §6), so a `float32 ** int32` reaches here as `Float32` and
/// the `Bool` arm below only fires when *both* sides are boolean -- which is
/// exactly where upstream raises `NotImplementedError: "pow" not implemented
/// for 'Bool'`, measured. The other two overloads still hand in the tensor
/// operand's own dtype, so a boolean tensor with a scalar exponent is still
/// refused here; see the arm's own comment for why that one is deliberate.
fn pow_result_tag(op: &str, tensor: TorchDType, scalar_is_float: bool) -> PyResult<TorchDType> {
    if tensor == TorchDType::Bool {
        // Two different situations reach this, and both are refusals rather
        // than gaps -- but only one of them is upstream's:
        //
        //   * `pow.Tensor_Tensor(bool, bool)` -- upstream raises
        //     `NotImplementedError: "pow" not implemented for 'Bool'`. Refusing
        //     is agreeing.
        //   * `pow.Tensor_Scalar(bool, <scalar>)` -- upstream *computes*, and
        //     the result category is not a promotion rule but a cascade of
        //     exponent fast paths in `pow_tensor_scalar`: measured on 2.13.0,
        //     `pow(bool_t, 2)` is `int64`, `pow(bool_t, 0)` is `int64` all-ones
        //     (the `exp == 0 -> ones_like` path), `pow(bool_t, True)` is
        //     **bool** (the `exp == 1 -> clone` path, and `True` is the scalar
        //     1), and `pow(bool_t, 2.0)` is `float32`. Reproducing that means
        //     reproducing the fast-path ladder, not a dtype table, and no
        //     measured caller needs it -- `square(bool_t)` is the only reachable
        //     spelling and nothing in the twenty architectures squares a mask.
        //     So it stays refused, now with the measurement recorded rather
        //     than with "has not been measured".
        return Err(not_implemented(format!(
            "{op}: torch.bool operands are not implemented in torch._C shim -- \
             upstream's boolean pow is a ladder of exponent fast paths (exp 0 \
             gives int64 ones, exp 1 clones and stays bool, exp 2 gives int64), \
             not a promotion rule, and reproducing a ladder nothing measured \
             needs is how a wrong answer gets in"
        )));
    }
    Ok(if scalar_is_float && !tensor.is_floating_point() {
        default_float()
    } else {
        tensor
    })
}

/// What upstream does with a *negative integer* exponent, which is not one
/// rule but two, split by overload. Measured on torch 2.13.0:
///
/// ```text
/// pow.Tensor_Scalar(int64([2,1,-1,0]), -1)          RuntimeError
/// pow.Tensor_Tensor(int64([2,1,-1,0]), int64(-1))   [0, 1, -1, 0]
/// pow.Scalar(2, int64([-1, 3]))                     [0, 8]
/// ```
///
/// So only the scalar-exponent overload refuses; the other two compute
/// `c10::powi`'s answer, which is `1` for base 1, `±1` for base -1 by the
/// parity of the exponent, and `0` otherwise. This was one refusal for all
/// three before, which is the safe direction to be wrong in but is still a
/// divergence -- and widening `Tensor_Tensor`'s dtype acceptance (§6) makes
/// many more integer pairs reach it.
#[derive(Clone, Copy, PartialEq)]
enum NegativeIntExponent {
    /// `pow.Tensor_Scalar`: upstream raises.
    Refuse,
    /// `pow.Tensor_Tensor` and `pow.Scalar`: upstream computes `powi`.
    Powi,
}

/// `c10::powi` for a signed base, transcribed. The unsigned dtypes cannot
/// reach the negative arm at all -- a `uint8` exponent is never below zero --
/// so the one function serves both.
fn powi(base: i64, exponent: i64) -> i64 {
    if exponent < 0 {
        return match base {
            1 => 1,
            -1 => {
                // Upstream's `(-b) % 2`: odd gives -1, even gives 1.
                if exponent % 2 == 0 {
                    1
                } else {
                    -1
                }
            }
            _ => 0,
        };
    }
    // Wrapping, like torch's integer kernels: an int64 overflow there
    // wraps rather than raising, and refusing here would diverge in
    // the other direction.
    base.wrapping_pow(exponent.min(u32::MAX as i64) as u32)
}

/// `x ** 2` without leaving the tensor, and without libm.
///
/// `pow_from_pairs` is the general path and it is *very* general: it widens
/// every element to `f64`, copies the vector twice, calls libm `pow` once per
/// element, and narrows back. That is six passes over the data plus a
/// transcendental call that vectorises into nothing.
///
/// It is also, on a real model, almost the only `pow` that runs. `LlamaRMSNorm`
/// -- and every RMSNorm transformers ships -- computes
/// `hidden_states.pow(2).mean(-1, keepdim=True)`, which for SmolLM2-135M is 61
/// calls per forward on a contiguous `[1, S, 576]` `float32` tensor.
/// `docs/SEQLEN.md` §2 measures that one op at **90-173x upstream** and shows
/// that it accounts for the entire *linear-in-S* term of the model-level gap.
/// Upstream is fast for exactly this reason: ATen's `pow_tensor_scalar`
/// special-cases small integral exponents into multiplication.
///
/// ## Why the answer cannot move
///
/// Not a tolerance argument -- an exactness one, for `f32`:
///
/// * `f32 -> f64` is exact, so the old path's input is the same number.
/// * A `f32` significand is 24 bits, so `b * b` needs at most 48 and `f64` has
///   53. **The exact square is representable**, so a correctly-rounded libm
///   `pow(b, 2.0)` returns it with no rounding at all.
/// * `fast_to(F32)` then rounds that exact square to `f32` -- one rounding,
///   from the exact product. IEEE-754 `f32` multiplication is *defined* as the
///   correctly-rounded exact product. Same rounding of the same value.
///
/// There is no double-rounding step to worry about because the intermediate was
/// exact, and no range to worry about either: the smallest `f32` subnormal
/// squared is ~2e-90, comfortably normal in `f64`, and anything that overflows
/// `f32` gives `inf` on both paths.
///
/// `f64` is included on the weaker ground that `b * b` and `pow(b, 2.0)` are
/// both required to be the correctly-rounded product; §3.2's test checks that
/// claim against this platform's libm rather than trusting it.
///
/// **`f16` and `bf16` are deliberately excluded.** For those the `f64`
/// intermediate is still exact but candle's reduced-precision multiply is not
/// obviously a single correctly-rounded step, and `docs/DTYPE_PERF.md` owns the
/// `bfloat16` checksum. Leaving them on the old path means that checksum cannot
/// move as a consequence of this change -- it is not merely expected to hold,
/// it is untouched.
fn pow_square_fast_path(
    op: &str,
    t: &Tensor,
    exponent: Scalar,
    tag: TorchDType,
) -> PyResult<Option<Tensor>> {
    if exponent.as_f64() != 2.0 {
        return Ok(None);
    }
    // `f32`/`f64` only -- see the doc comment.
    if !matches!(
        t.dtype(),
        candle_core::DType::F32 | candle_core::DType::F64
    ) {
        return Ok(None);
    }
    // No promotion in flight: the generic path would have narrowed to the tag's
    // storage at the end, and a multiply cannot do that. Bail if they differ.
    if PyDtype::new(tag).storage(op)? != t.dtype() {
        return Ok(None);
    }
    t.mul(t).map(Some).map_err(|err| candle_err(op, err))
}

fn pow_from_pairs(
    py: Python<'_>,
    op: &str,
    bases: PowSide,
    exponents: PowSide,
    shape: Vec<usize>,
    tag: TorchDType,
    device: &Device,
    negative: NegativeIntExponent,
) -> PyResult<Py<PyAny>> {
    let storage = PyDtype::new(tag).storage(op)?;
    let tensor = if tag.is_floating_point() {
        let (b, e) = (bases.as_f64(), exponents.as_f64());
        let n = b.len().max(e.len());
        let values: Vec<f64> = (0..n)
            .map(|i| b[i % b.len()].powf(e[i % e.len()]))
            .collect();
        Tensor::from_vec(values, shape, device)
    } else {
        let (b, e) = (bases.as_i64(), exponents.as_i64());
        let n = b.len().max(e.len());
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            let exponent = e[i % e.len()];
            if exponent < 0 && negative == NegativeIntExponent::Refuse {
                // torch's message, verbatim.
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Integers to negative integer powers are not allowed.",
                ));
            }
            values.push(powi(b[i % b.len()], exponent));
        }
        Tensor::from_vec(values, shape, device)
    }
    .and_then(|t| t.fast_to(storage))
    .map_err(|err| candle_err(op, err))?;
    finish(py, tensor, tag)
}

/// The values of one side of a `pow`, already flattened to the broadcast shape
/// (or a single element, which the caller cycles).
enum PowSide {
    Floats(Vec<f64>),
    Ints(Vec<i64>),
}

impl PowSide {
    fn as_f64(&self) -> Vec<f64> {
        match self {
            PowSide::Floats(v) => v.clone(),
            PowSide::Ints(v) => v.iter().map(|&x| x as f64).collect(),
        }
    }

    fn as_i64(&self) -> Vec<i64> {
        match self {
            PowSide::Floats(v) => v.iter().map(|&x| x as i64).collect(),
            PowSide::Ints(v) => v.clone(),
        }
    }
}

fn pow_tensor_scalar(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.pow.Tensor_Scalar";
    let base = tensor_arg(OP, args, kwargs, 0, "self")?;
    let exponent = scalar_arg(OP, args, kwargs, 1, "exponent")?
        .ok_or_else(|| missing(OP, "exponent"))?;
    let tag = pow_result_tag(OP, base.tag(), !exponent.is_int())?;
    let shape = base.tensor()?.dims().to_vec();
    // `x ** 2` is the RMSNorm case and it is the whole linear term of the
    // model-level gap (docs/SEQLEN.md §2). Bit-identical -- see the fast path's
    // doc comment for why that is exactness rather than tolerance.
    if let Some(t) = pow_square_fast_path(OP, base.tensor()?, exponent, tag)? {
        return finish(py, t, tag);
    }
    let bases = side_from_tensor(OP, base.tensor()?, tag)?;
    let exponents = side_from_scalar(&exponent, tag);
    pow_from_pairs(
        py,
        OP,
        bases,
        exponents,
        shape,
        tag,
        base.tensor()?.device(),
        // The one overload where upstream refuses a negative integer exponent.
        NegativeIntExponent::Refuse,
    )
}

fn pow_scalar(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.pow.Scalar";
    let base = scalar_arg(OP, args, kwargs, 0, "self")?.ok_or_else(|| missing(OP, "self"))?;
    let exponent = tensor_arg(OP, args, kwargs, 1, "exponent")?;
    let tag = pow_result_tag(OP, exponent.tag(), !base.is_int())?;
    let shape = exponent.tensor()?.dims().to_vec();
    let bases = side_from_scalar(&base, tag);
    let exponents = side_from_tensor(OP, exponent.tensor()?, tag)?;
    pow_from_pairs(
        py,
        OP,
        bases,
        exponents,
        shape,
        tag,
        exponent.tensor()?.device(),
        NegativeIntExponent::Powi,
    )
}

fn pow_tensor_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.pow.Tensor_Tensor";
    let base = tensor_arg(OP, args, kwargs, 0, "self")?;
    let exponent = tensor_arg(OP, args, kwargs, 1, "exponent")?;
    // **Promotes rather than requiring equal dtypes** -- `bloom` is what asked
    // (docs/ARCH20.md §6): `build_alibi_tensor` computes
    // `torch.pow(base, powers)` with a `float32` base and an `int32` exponent,
    // and `same_dtype` refused it by name.
    //
    // The rule is `promote_types`, the same table `mul.Tensor` and
    // `bitwise_and.Tensor` already use, and it was re-measured against
    // `pow.Tensor_Tensor`'s own result dtype over the full 10x10 grid of
    // storable dtypes before being reused: every cell agrees with
    // `torch._prims_common.get_higher_dtype` except `bool ** bool`, where
    // upstream raises and so does this (`pow_result_tag`'s `Bool` arm).
    //
    // Note what `promote_operands` does *not* do: its `lhs.tag() == rhs.tag()`
    // fast path returns the shared dtype before the rank table is consulted,
    // which is the only reason a same-rank pair like `float16 ** float16` does
    // not escape to `float32` the way `float16 ** bfloat16` correctly does
    // (docs/BIND.md §9 -- `get_higher_dtype`'s `if a is b` guards exactly the
    // same table for exactly the same reason).
    let tag = pow_result_tag(OP, promote_operands(OP, &base, &exponent)?, false)?;

    let shape = base
        .tensor()?
        .shape()
        .broadcast_shape_binary_op(exponent.tensor()?.shape(), "pow")
        .map_err(|err| candle_err(OP, err))?;
    let dims = shape.dims().to_vec();
    let broadcast = |t: &Tensor| -> PyResult<Tensor> {
        t.broadcast_as(shape.clone())
            .and_then(|t| t.contiguous())
            .map_err(|err| candle_err(OP, err))
    };
    let bases = side_from_tensor(OP, &broadcast(base.tensor()?)?, tag)?;
    let exponents = side_from_tensor(OP, &broadcast(exponent.tensor()?)?, tag)?;
    pow_from_pairs(
        py,
        OP,
        bases,
        exponents,
        dims,
        tag,
        base.tensor()?.device(),
        NegativeIntExponent::Powi,
    )
}

fn side_from_tensor(op: &str, tensor: &Tensor, tag: TorchDType) -> PyResult<PowSide> {
    let flat = tensor.flatten_all().map_err(|err| candle_err(op, err))?;
    if tag.is_floating_point() {
        Ok(PowSide::Floats(
            widen_f64(&flat)
                .and_then(|t| t.to_vec1::<f64>())
                .map_err(|err| candle_err(op, err))?,
        ))
    } else {
        Ok(PowSide::Ints(
            flat.to_dtype(candle_core::DType::I64)
                .and_then(|t| t.to_vec1::<i64>())
                .map_err(|err| candle_err(op, err))?,
        ))
    }
}

/// The `Scalar` side of a `pow`, **narrowed into the result dtype first**.
///
/// `pow` is on the other side of the split docs/SCALAR.md draws: where
/// `mul_kernel` and `div_*_kernel` read the operand with
/// `original_scalar_value<opmath_t>`, `pow_tensor_scalar_kernel` converts it to
/// the dispatched `scalar_t`, so the exponent really is rounded into the
/// tensor's dtype before `std::pow` sees it. Measured on 2.13.0 over
/// `[3, 5, 7, 11, 13, 96, 2, 0.5]` and five exponents, in every floating dtype:
///
/// ```text
/// float16  [3] ** 0.3    upstream 1.3896484375   narrowing the exponent 1.3896484375
///                                                keeping the f64        1.390625
/// float32  [7] ** 0.3    upstream 1.7927899360   keeping the f64        1.7927900552
/// ```
///
/// **`float32` narrows too**, which is the half that is easy to get wrong:
/// `opmath_type<float>` is `float`, so there is no widening anywhere in this
/// kernel and the `f64` the parser produced is never the value upstream uses.
/// `float64` narrows to itself, so the same line is correct for all four.
///
/// The residual `float32`/`float64` disagreements that remain after this
/// (1 of 7 exponents, ~1 ULP) are `powf` against libm, not the scalar rule --
/// on the pair that separates them, `5.0 ** -1.5`, this shim's `f64` road is
/// the *correctly rounded* `float32` answer and upstream's is one step below.
fn side_from_scalar(value: &Scalar, tag: TorchDType) -> PowSide {
    if tag.is_floating_point() {
        PowSide::Floats(vec![float_narrower(tag)(value.as_f64())])
    } else {
        PowSide::Ints(vec![value.as_i64()])
    }
}

/// `aten::cat(Tensor[] tensors, int dim=0) -> Tensor`
///
/// **A tensor of shape exactly `(0,)` is skipped, not concatenated.** This is
/// torch's "legacy empty" rule, and it is not a corner case anyone can avoid:
/// `transformers`' KV cache starts every layer as `torch.tensor([])` and grows
/// it with `torch.cat([self.keys, key_states], dim=-2)`
/// (`cache_utils.py:144`), so the *first* decoder step of every model is a cat
/// of a 1-D empty against a 4-D tensor. Without the rule this shim raised
/// `IndexError: Dimension out of range` and the forward pass stopped there
/// (docs/E2E_REAL.md).
///
/// The rule is narrow, and measured on 2.13.0 rather than inferred:
///
///   * only shape `(0,)` qualifies. `torch.ones(0, 5)` does **not** -- it
///     raises `Tensors must have same number of dimensions: got 2 and 4`,
///     so "empty" is not the test, `(0,)` is.
///   * a skipped entry takes no part in the rank check, the `dim` check or
///     the extent check -- `cat([tensor([]), ones(1,2,3,4)], dim=-2)` is
///     `(1, 2, 3, 4)`, while a *non*-empty 1-D entry in the same position
///     still raises `IndexError` on `dim=-2`.
///   * when every entry is skipped the result is `(0,)`, whatever `dim` was;
///     `cat([tensor([]), tensor([])], dim=5)` is `(0,)` rather than an
///     `IndexError`.
///
/// It does still take part in **dtype**: upstream promotes, so
/// `cat([int64 (0,), int32 (2,3)])` is `int64` rather than `int32`. A skipped
/// entry is skipped for its *shape* only -- `promote_list` runs over the
/// whole list, before the partition, so a `(0,)` entry still contributes its
/// dtype exactly as upstream's does.
fn cat_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.cat.default";
    let tensors: Vec<PyTensorBase> = required(OP, args, kwargs, 0, "tensors")?.extract()?;
    if tensors.is_empty() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "torch.cat(): expected a non-empty list of Tensors",
        ));
    }
    let tag = promote_list(OP, &tensors)?;
    let storage = PyDtype::new(tag).storage(OP)?;

    // The legacy-empty partition, before anything reads a rank. Each kept
    // entry is brought to the common dtype here -- `Tensor::cat` needs one
    // dtype across the list, and this is the same narrowing upstream applies
    // (`cat([int64([2049]), float16([1.])])` is `[2048., 1.]`, measured, not
    // `[2049., 1.]`).
    let mut kept: Vec<Tensor> = Vec::with_capacity(tensors.len());
    for t in &tensors {
        let inner = t.tensor()?;
        if inner.dims() != [0] {
            kept.push(operand_in(OP, inner, storage)?);
        }
    }
    if kept.is_empty() {
        // Every entry was `(0,)`. Upstream hands back a `(0,)` of the same
        // dtype without ever looking at `dim`.
        let out = Tensor::from_vec(Vec::<f64>::new(), 0usize, tensors[0].tensor()?.device())
            .and_then(|t| t.fast_to(storage))
            .map_err(|err| candle_err(OP, err))?;
        return finish(py, out, tag);
    }

    let rank = kept[0].rank();
    let dim = normalise_dim(OP, dim_arg(args, kwargs, 1, "dim")?.unwrap_or(0), rank)?;
    let tensor = Tensor::cat(&kept, dim).map_err(|err| candle_err(OP, err))?;
    finish(py, tensor, tag)
}

/// `aten::stack(Tensor[] tensors, int dim=0) -> Tensor`
///
/// **Not `cat` with a different name.** `cat` joins along an existing axis and
/// lets the extents differ there; `stack` inserts a *new* axis and therefore
/// requires every entry to have exactly the same size, rank included. Measured:
/// `stack([zeros(2), zeros(2,1)])` raises `stack expects each tensor to be
/// equal size, but got [2] at entry 0 and [2, 1] at entry 1`, where the
/// corresponding `cat` is a legal call on a different axis. Routing this key at
/// `cat_default` would compute a different op.
///
/// Reached by GPT-J's rotary embedding, which is the only architecture of the
/// four this op was added for that calls it: `stack([x1, x2], dim=-1)` on a
/// pair of `(batch, seq, heads, dim/2)` slices, then a flatten. docs/ARCH.md
/// counts three more callers (`cohere`, `helium`, `mamba`).
///
/// Three rules were measured rather than inferred, and two of them differ from
/// `cat`:
///
///   * **`dim` runs to `rank`, not `rank - 1`.** The new axis can go after the
///     last existing one, so the legal range is `[-(rank+1), rank]` -- the same
///     widened range `unsqueeze` has and `cat` does not. `stack([a, b], 1)` on
///     two 1-D tensors is legal; `cat` at `dim=1` on the same pair is not.
///   * **an empty list is refused** (`stack expects a non-empty TensorList`),
///     because there is no shape to invent for the result. `cat` refuses too,
///     with different wording.
///   * **upstream promotes dtypes here** -- `stack([bool, int64])` gives
///     `int64` and `stack([int64, float32])` gives `float32`, both measured.
///     This shim used to refuse; it now promotes through `promote_list`, the
///     same fold `cat_default` uses, `stack`'s own 9x9 grid having been
///     measured against `torch.promote_types` in docs/PROMOTE.md §3 rather
///     than inherited from `cat`'s.
///
/// Entries are made contiguous first. Upstream accepts a non-contiguous entry
/// and answers a contiguous result (measured on a transposed input), and
/// candle's `cat` -- which `Tensor::stack` reaches after unsqueezing -- wants
/// contiguous inputs.
fn stack_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.stack.default";
    let tensors: Vec<PyTensorBase> = required(OP, args, kwargs, 0, "tensors")?.extract()?;
    if tensors.is_empty() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "stack expects a non-empty TensorList",
        ));
    }
    let tag = promote_list(OP, &tensors)?;
    let storage = PyDtype::new(tag).storage(OP)?;
    let first = tensors[0].tensor()?.dims().to_vec();
    for (index, other) in tensors.iter().enumerate().skip(1) {
        if other.tensor()?.dims() != first.as_slice() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "stack expects each tensor to be equal size, but got {:?} at entry 0 \
                 and {:?} at entry {index}",
                first,
                other.tensor()?.dims()
            )));
        }
    }

    // The widened range: `normalise_dim` clamps to `rank.max(1)`, which is the
    // right rule for an *existing* axis and one short for a new one. A 0-D
    // entry stacks at `dim=0` only, and `stack([tensor(1.), tensor(2.)], 1)`
    // is an IndexError upstream -- so the extent is `rank + 1`, uniformly.
    let raw = dim_arg(args, kwargs, 1, "dim")?.unwrap_or(0);
    let extent = first.len() as isize + 1;
    let dim = if raw < 0 { raw + extent } else { raw };
    if dim < 0 || dim >= extent {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "{OP}: Dimension out of range (expected to be in range of [{}, {}], but got {raw})",
            -extent,
            extent - 1
        )));
    }

    // Brought to the common dtype and made contiguous, in that order:
    // `Tensor::stack` needs one dtype across the list, and the narrowing is
    // the one upstream applies to each operand rather than to the result.
    let contiguous: Vec<Tensor> = tensors
        .iter()
        .map(|t| {
            operand_in(OP, t.tensor()?, storage)?
                .contiguous()
                .map_err(|e| candle_err(OP, e))
        })
        .collect::<PyResult<_>>()?;
    let tensor = Tensor::stack(&contiguous, dim as usize).map_err(|e| candle_err(OP, e))?;
    finish(py, tensor, tag)
}

/// `aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None,
///     Device? device=None, bool? pin_memory=None) -> Tensor`
///
/// A 0-D tensor holding one number. It is how `falcon`, `gptj`, `bloom` and
/// `mpt` build their attention mask fill value -- all four call
/// `scalar_tensor(finfo(dtype).min, dtype=..., device=...)` and hand the result
/// straight to `where.self` (measured, docs/OPS4.md §1).
///
/// **The dtype rule is not `full`'s, and inferring it from `full` would be
/// wrong.** `full` reads the fill value's category (`full([], 3)` is `int64`,
/// `full([], True)` is `bool`); `scalar_tensor` ignores it entirely and always
/// answers the default float:
///
/// ```text
/// scalar_tensor(3)      -> float32   (full([], 3)     -> int64)
/// scalar_tensor(True)   -> float32   (full([], True)  -> bool)
/// scalar_tensor(1.5)    -> float32
/// ```
///
/// Measured on torch 2.13.0 over int, bool, float, `nan` and `inf`. A shim that
/// shared `full`'s inference would give `int64` where torch gives `float32`,
/// and the mask value would then be silently truncated.
///
/// The overflow rule *is* `full`'s, including its numel==1 hole, and that too
/// is measured rather than assumed: `scalar_tensor(1e6, float16)` is `inf`
/// while `scalar_tensor(1e300, float32)` raises, which is exactly what
/// `checked_convert` at `numel = 1` reproduces (only the reduced-precision
/// floats take the unchecked fast path). `scalar_tensor(-1, uint8)` is `255`
/// and `scalar_tensor(300, uint8)` raises -- the two's-complement wrap
/// allowance, same as `full`.
///
/// `layout=torch.strided` is accepted rather than refused. `reject_unsupported`
/// would turn it away, and the measured call sites pass it explicitly (`bloom`,
/// `mpt` and `gptj` all send `layout=torch.strided`), so refusing it would
/// block the four architectures this op exists to open on an argument that
/// names the only layout the shim has. Any other layout still refuses.
fn scalar_tensor_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.scalar_tensor.default";
    let raw = required(OP, args, kwargs, 0, "s")?;
    let value = scalar_arg(OP, args, kwargs, 0, "s")?.ok_or_else(|| missing(OP, "s"))?;
    let dtype = dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(default_float());
    reject_layout(OP, args, kwargs, 2)?;
    reject_unsupported(OP, args, kwargs, &[(4, "pin_memory")])?;
    let label = device_arg_or_label(args, kwargs, 3, "device", &PyDevice::cpu())?;

    // A 0-D tensor is one element, so the numel==1 branch of the upstream
    // check is the one that applies -- see the note above.
    if !raw.is_instance_of::<PyTensorBase>() {
        checked_convert(&raw, raw.is_instance_of::<pyo3::types::PyInt>(), dtype, 1)?;
    }
    if label.is_meta() {
        return meta_result(py, Vec::new(), dtype);
    }
    let device = label.resolve()?;

    if dtype == TorchDType::Bool {
        let truthy = u8::from(value.as_f64() != 0.0);
        let tensor = Tensor::full(truthy, (), &device).map_err(|e| candle_err(OP, e))?;
        return finish(py, tensor, dtype);
    }
    let storage = PyDtype::new(dtype).storage(OP)?;
    let tensor = if storage.is_int() {
        // Upstream truncates toward zero rather than rounding:
        // `scalar_tensor(-1.5, dtype=int64)` is `-1`, measured.
        Tensor::full(value.as_i64(), (), &device)
    } else {
        Tensor::full(value.as_f64(), (), &device)
    }
    .and_then(|t| t.fast_to(storage))
    .map_err(|e| candle_err(OP, e))?;
    finish(py, tensor, dtype)
}

/// `aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor`
///
/// `dim=None` flattens first, and `keepdim=True` alongside it gives shape
/// `[1]` rather than `[]` -- measured, and not what "keep the reduced
/// dimension" suggests when there was no named dimension to keep.
///
/// The result is int64. candle's `argmax` yields `u32`, which would be a
/// visible dtype divergence on the very first `generate()` step.
///
/// **A NaN wins, and it is the first NaN that wins.** `argmax([1., nan, 3.])`
/// is `1` upstream (measured) -- there is no ordering under which a real
/// number beats a NaN, so the reduction stops at the first one it meets. This
/// build answered `2`, the same dropped-NaN fault `max.dim`, `max.other` and
/// `max.default` each had in turn, from the same `|x, y| x < y` predicate.
/// Corrected through the shared `nan_along_dim` rather than a fourth private
/// repair; the correction runs only when the input really holds a NaN, so
/// sampling's own `argmax` keeps the exact bits it had.
fn argmax_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.argmax.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let dim = dim_arg(args, kwargs, 1, "dim")?;
    let keepdim = bool_arg(args, kwargs, 2, "keepdim")?.unwrap_or(false);
    let tag = input.tag();

    let tensor = match dim {
        None => {
            // `dim=None` flattens, so the whole tensor is one reduced slice and
            // the answer is a *flat* index -- which is why the correction has
            // to happen against `flat`, not against the original shape.
            let flat = input.tensor()?.flatten_all().map_err(|e| candle_err(OP, e))?;
            let mut reduced = flat
                .argmax_keepdim(0)
                .and_then(|t| t.to_dtype(candle_core::DType::I64))
                .map_err(|e| candle_err(OP, e))?;
            if let Some((any, first)) = nan_along_dim(OP, &flat, 0, tag)? {
                reduced = any
                    .where_cond(&first, &reduced)
                    .map_err(|e| candle_err(OP, e))?;
            }
            if keepdim {
                reduced.reshape(1).map_err(|e| candle_err(OP, e))?
            } else {
                reduced.reshape(()).map_err(|e| candle_err(OP, e))?
            }
        }
        Some(dim) => {
            let dim = normalise_dim(OP, dim, input.tensor()?.rank())?;
            let mut reduced = input
                .tensor()?
                .argmax_keepdim(dim)
                .and_then(|t| t.to_dtype(candle_core::DType::I64))
                .map_err(|e| candle_err(OP, e))?;
            if let Some((any, first)) = nan_along_dim(OP, input.tensor()?, dim, tag)? {
                reduced = any
                    .where_cond(&first, &reduced)
                    .map_err(|e| candle_err(OP, e))?;
            }
            if keepdim {
                reduced
            } else {
                reduced.squeeze(dim).map_err(|e| candle_err(OP, e))?
            }
        }
    };
    finish(py, tensor, TorchDType::Int64)
}

/// `aten::embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1,
///                  bool scale_grad_by_freq=False, bool sparse=False)`
///
/// candle's `Tensor::embedding` demands rank-1 indices; torch takes any shape
/// and appends the embedding dimension, which is the shape `transformers`
/// relies on (`[batch, seq]` in, `[batch, seq, hidden]` out).
///
/// The three trailing arguments are backward-only in torch -- the forward
/// result does not depend on any of them. `padding_idx` is therefore accepted
/// and ignored, which is what upstream's forward does; the other two are
/// refused, because switching on a gradient behaviour behind a shim with no
/// autograd would be claiming something.
fn embedding_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.embedding.default";
    let weight = tensor_arg(OP, args, kwargs, 0, "weight")?;
    let indices = tensor_arg(OP, args, kwargs, 1, "indices")?;
    for (index, name) in [(3, "scale_grad_by_freq"), (4, "sparse")] {
        if bool_arg(args, kwargs, index, name)?.unwrap_or(false) {
            return Err(not_implemented(format!(
                "{OP}: argument '{name}' only affects the backward pass, and there \
                 is no autograd in torch._C shim"
            )));
        }
    }

    if weight.tensor()?.rank() != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{OP}: weight must be 2D, got {}D",
            weight.tensor()?.rank()
        )));
    }
    let mut shape = indices.tensor()?.dims().to_vec();
    shape.push(weight.tensor()?.dims()[1]);

    let flat = indices
        .tensor()?
        .flatten_all()
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(OP, e))?;
    let tensor = weight
        .tensor()?
        .index_select(&flat, 0)
        .and_then(|t| t.reshape(shape))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, tensor, weight.tag())
}

/// `aten::is_floating_point(Tensor self) -> bool`
///
/// The only op here that answers from the tag alone. Upstream's
/// `torch.is_floating_point` does not reach the dispatcher at all (measured
/// with a `TorchDispatchMode` logger: the call produces no aten record), but
/// this shim routes it through the one door anyway, so that the surface has a
/// single entrance rather than one entrance and one shortcut.
fn is_floating_point_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.is_floating_point.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    Ok(input.tag().is_floating_point().into_bound_py_any(py)?.unbind())
}

/// `aten::isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *,
///                           bool assume_unique=False, bool invert=False)`
///
/// Result is `torch.bool` shaped like `elements`. `assume_unique` is a
/// performance hint with no effect on the answer, so it is accepted; `invert`
/// negates.
fn isin_tensor_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.isin.Tensor_Tensor";
    let elements = tensor_arg(OP, args, kwargs, 0, "elements")?;
    let test = tensor_arg(OP, args, kwargs, 1, "test_elements")?;
    
    if elements.tag() == TorchDType::Bool || test.tag() == TorchDType::Bool {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{OP}: upstream refuses bool operands ({} vs {})",
            elements.tag().name(),
            test.tag().name()
        )));
    }
    
    let tag = promote_operands(OP, &elements, &test)?;
    let invert = bool_arg(args, kwargs, 3, "invert")?.unwrap_or(false);

    // Compared as f64 when either side is floating, as i64 otherwise. Equality
    // is exact in both, since the two operands share a dtype -- there is no
    // promotion step that could round one side onto the other.
    let (haystack, needles) = if tag.is_floating_point() {
        (
            side_from_tensor(OP, elements.tensor()?, tag)?.as_f64(),
            side_from_tensor(OP, test.tensor()?, tag)?.as_f64(),
        )
    } else {
        (
            side_from_tensor(OP, elements.tensor()?, tag)?
                .as_i64()
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            side_from_tensor(OP, test.tensor()?, tag)?
                .as_i64()
                .into_iter()
                .map(|v| v as f64)
                .collect(),
        )
    };

    let bytes: Vec<u8> = haystack
        .iter()
        .map(|value| u8::from(needles.iter().any(|n| n == value) != invert))
        .collect();
    let tensor = Tensor::from_vec(bytes, elements.tensor()?.dims().to_vec(), elements.tensor()?.device())
        .map_err(|e| candle_err(OP, e))?;
    finish(py, tensor, TorchDType::Bool)
}

/// `aten::lift_fresh(Tensor(a) self) -> Tensor(a)`
///
/// Identity, and that is the whole op upstream too: it marks a constant as
/// having entered the graph. It is here because it is the *only* aten call
/// `torch.tensor([...])` makes -- measured -- so `torch.tensor` reaching the
/// dispatcher at all depends on it existing.
fn lift_fresh_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.lift_fresh.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    Ok(input.into_pyobject(py)?.into_any().unbind())
}

/// The rounding behaviour `update_from`/`update_to` are written in terms of,
/// for the dtypes `randint` can be asked for.
///
/// `float32`, `float64` and `bfloat16` all have a maximum past `int64_t`'s, so
/// nothing this path can be handed overflows them; `float16` stops at 65504
/// and the overflow is reachable and observable.
fn randint_float_format(tag: TorchDType) -> Option<crate::rng::FloatFormat> {
    use crate::rng::FloatFormat;
    Some(match tag {
        TorchDType::Float64 => FloatFormat { digits: 53, max_finite: u128::MAX },
        TorchDType::Float32 => FloatFormat { digits: 24, max_finite: u128::MAX },
        TorchDType::Float16 => FloatFormat { digits: 11, max_finite: 65504 },
        TorchDType::BFloat16 => FloatFormat { digits: 8, max_finite: u128::MAX },
        _ => return None,
    })
}

/// What `check_from_to_in_range` compares against, and the name it prints.
///
/// The name is `caffe2::TypeMeta`'s C++ spelling, not torch's Python one --
/// upstream's message is `to - 1 is out of bounds for int`, never `int32`.
/// Measured dtype by dtype (docs/RANDINT.md §3.3); guessing `int32` here would
/// produce a refusal that reads right and does not match.
///
/// `None` means the check cannot fire: every `int64_t` is representable, and
/// float32/float64/bfloat16 all have a maximum past `int64_t`'s. Those three
/// still lose precision, which is what `update_to` and the (unimplemented,
/// §7) warning are for -- a lost bit is not an out-of-bounds bound.
fn randint_representable(tag: TorchDType) -> Option<(&'static str, i64, i64)> {
    Some(match tag {
        TorchDType::Int32 => ("int", i32::MIN as i64, i32::MAX as i64),
        TorchDType::Int16 => ("short", i16::MIN as i64, i16::MAX as i64),
        TorchDType::Int8 => ("signed char", i8::MIN as i64, i8::MAX as i64),
        TorchDType::UInt8 => ("unsigned char", 0, u8::MAX as i64),
        TorchDType::UInt16 => ("unsigned short", 0, u16::MAX as i64),
        TorchDType::UInt32 => ("unsigned int", 0, u32::MAX as i64),
        TorchDType::Bool => ("bool", 0, 1),
        TorchDType::Float16 => ("c10::Half", -65504, 65504),
        _ => return None,
    })
}

/// `at::native::templates::check_from_to_in_range`, message included.
fn randint_check_in_range(from: i64, to_inclusive: i64, tag: TorchDType) -> PyResult<()> {
    let Some((name, min, max)) = randint_representable(tag) else {
        return Ok(());
    };
    // `from` first, then `to - 1`: upstream's two `CHECK_OUT_OF_BOUNDS` calls
    // are in that order, and a call that violates both reports the first.
    for (label, value) in [("from", from), ("to - 1", to_inclusive)] {
        if value < min || value > max {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{label} is out of bounds for {name}"
            )));
        }
    }
    Ok(())
}

/// Turn the `int64_t` draws into the requested dtype the way the C++ cast does.
///
/// `static_cast<T>(static_cast<int64_t>(...))`, and for the reduced-precision
/// floats that is *two* roundings, not one: `c10::Half` and `c10::BFloat16`
/// are only constructible from `float`, so an `int64_t` becomes a `float`
/// first and is narrowed again. Handing candle the `int64` tensor and asking
/// for `bf16` would be a single rounding from a wider type, which differs from
/// upstream on exactly the values where it matters.
fn randint_narrow(
    op: &str,
    values: Vec<i64>,
    shape: Vec<usize>,
    storage: candle_core::DType,
    device: &Device,
) -> PyResult<Tensor> {
    use candle_core::DType;
    let built = match storage {
        DType::F64 => {
            Tensor::from_vec(values.into_iter().map(|v| v as f64).collect::<Vec<_>>(), shape, device)
        }
        DType::F32 | DType::F16 | DType::BF16 => {
            let floats = values.into_iter().map(|v| v as f32).collect::<Vec<_>>();
            Tensor::from_vec(floats, shape, device).and_then(|t| t.fast_to(storage))
        }
        _ => Tensor::from_vec(values, shape, device).and_then(|t| t.fast_to(storage)),
    };
    built.map_err(|e| candle_err(op, e))
}

/// `aten::randint.low(SymInt low, SymInt high, SymInt[] size, *,
///                    ScalarType? dtype=4, ...)` and `aten::randint(...)`.
///
/// `dtype=4` in the schema is `ScalarType::Long`, so the default is int64
/// rather than the default float every other factory here uses.
///
/// This used to draw from candle's unseedable generator and say so. It now
/// draws from the ported one (`crate::rng`), so a seeded `randint` reproduces
/// upstream bit for bit -- and, just as importantly, leaves the stream where
/// upstream leaves it. docs/RANDINT.md is the measurement; three things in it
/// are not guessable from the values alone:
///
///   * **One draw per element, never a retry.** `uniform_int_from_to` is a
///     biased modulo (`val % range + base`) and the header says so. A
///     rejection-sampling implementation would agree on the distribution,
///     produce different values, and consume a data-dependent number of words.
///   * **The width, alone, decides whether that draw is 32 or 64 bits**, at
///     2^28 -- see `rng::RANDINT_WIDE_THRESHOLD` for why the dtype-shaped
///     condition in the header is the wrong arm to copy.
///   * **A float dtype moves the bounds before any of that** (`update_from` /
///     `update_to`), which changes both the modulus and which side of the
///     threshold the call falls on.
///
/// The checks below are in upstream's order, and the order is observable:
/// the representable-range check runs *before* the empty-tensor shortcut, so
/// `randint(0, 2**32, (0,), dtype=torch.int32)` raises rather than returning
/// an empty tensor.
fn randint(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    with_low: bool,
) -> PyResult<Py<PyAny>> {
    let (op, low, high_at, size_at) = if with_low {
        (
            "aten.randint.low",
            int_arg(args, kwargs, 0, "low")?.ok_or_else(|| missing("aten.randint.low", "low"))?,
            1usize,
            2usize,
        )
    } else {
        ("aten.randint.default", 0i64, 0usize, 1usize)
    };
    let high = int_arg(args, kwargs, high_at, "high")?.ok_or_else(|| missing(op, "high"))?;
    let size: Vec<usize> = required(op, args, kwargs, size_at, "size")?.extract()?;
    let options_at = size_at + 1;
    let dtype = dtype_arg(args, kwargs, options_at, "dtype")?.unwrap_or(TorchDType::Int64);
    reject_unsupported(
        op,
        args,
        kwargs,
        &[(options_at + 1, "layout"), (options_at + 3, "pin_memory")],
    )?;
    let label = device_arg_or_label(args, kwargs, options_at + 2, "device", &PyDevice::cpu())?;

    if high <= low {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "random_ expects 'from' to be less than 'to', but got from={low} >= to={high}"
        )));
    }
    let (from, to) = match randint_float_format(dtype) {
        Some(format) => {
            let from = crate::rng::update_from(low, format);
            let to = crate::rng::update_to(high, format);
            if to <= from {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "random_ expects 'from' casted to dtype to be less than 'to' casted to \
                     dtype, but got from={from} >= to={to}"
                )));
            }
            (from, to)
        }
        None => (low, high),
    };
    randint_check_in_range(from, to - 1, dtype)?;

    // After the bound checks, before the allocation: upstream's meta kernel
    // raises for `high <= low` too, so the meta answer is not a way around a
    // check the real call makes.
    if label.is_meta() {
        return meta_result(py, size, dtype);
    }
    let device = label.resolve()?;
    let storage = PyDtype::new(dtype).storage(op)?;
    let numel: usize = size.iter().product();
    // `CHECK_EMPTY_AND_RETURN` sits after the checks above and before the
    // kernel, so an empty result draws nothing at all -- measured as zero
    // words consumed, not as "too few to notice".
    let values = if numel == 0 {
        Vec::new()
    } else {
        let mut gen = crate::rng::default_generator();
        crate::rng::randint_from_to_fill(&mut gen, numel, from, to)
    };
    let tensor = randint_narrow(op, values, size, storage, &device)?;
    finish(py, tensor, dtype)
}

/// `aten::randperm(SymInt n, *, ScalarType? dtype=4, Layout? layout=None,
///                 Device? device=None, bool? pin_memory=None) -> Tensor`
///
/// Here because it is the same machinery, not because the surface wanted
/// widening: `randperm_cpu` is Fisher-Yates over the *same* MT19937, one
/// 32-bit `random()` per swap, `n - 1` of them. docs/RANDINT.md §4 reproduced
/// n up to 50000 with that one loop and found no large-`n` alternative path,
/// which is what made adding it cheaper than refusing it by name.
///
/// Two guards that are not the RNG and are still part of matching upstream:
///
///   * `check_supported_max_int_with_precision` refuses when `n - 1` cannot be
///     *exactly* held -- `n cannot be greater than 2049 for Half type.` -- but
///     **only for Half, Float and Double**. `bfloat16` is not in that switch,
///     so upstream happily returns a `randperm(300)` with 279 distinct values
///     (measured). That is upstream's defect and this reproduces it rather
///     than improving on it; a shim that quietly returned a real permutation
///     there would disagree with torch and look like the better answer.
///   * the integral dtypes get the check from the other direction, through
///     `scalar_tensor`: `value cannot be converted to type int16_t without
///     overflow`.
fn randperm(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.randperm.default";
    let n = int_arg(args, kwargs, 0, "n")?.ok_or_else(|| missing(OP, "n"))?;
    let dtype = dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(TorchDType::Int64);
    reject_unsupported(OP, args, kwargs, &[(2, "layout"), (4, "pin_memory")])?;
    let label = device_arg_or_label(args, kwargs, 3, "device", &PyDevice::cpu())?;

    if n < 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Trying to create tensor with negative dimension {n}: [{n}]"
        )));
    }
    if dtype == TorchDType::Bool {
        return Err(not_implemented(
            "\"randperm\" not implemented for 'Bool'".to_string(),
        ));
    }
    // `at::scalar_tensor(n > 0 ? n - 1 : n, options)`: the largest value the
    // result has to hold is `n - 1`, and an integral dtype that cannot hold it
    // refuses here rather than wrapping around.
    if let Some((_, _, max)) = randint_representable(dtype) {
        let largest = if n > 0 { n - 1 } else { n };
        if !dtype.is_floating_point() && largest > max {
            let c_name = match dtype {
                TorchDType::Int32 => "int32_t",
                TorchDType::Int16 => "int16_t",
                TorchDType::Int8 => "int8_t",
                TorchDType::UInt8 => "uint8_t",
                TorchDType::UInt16 => "uint16_t",
                TorchDType::UInt32 => "uint32_t",
                other => other.name(),
            };
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "value cannot be converted to type {c_name} without overflow"
            )));
        }
    }
    // The float switch, exactly as narrow as upstream's -- bfloat16 absent on
    // purpose. The messages are upstream's verbatim, including the one that
    // spells its bound out and the two that leave it as a power.
    let precision_limit = match dtype {
        TorchDType::Float16 => Some(((1i64 << 11) + 1, "n cannot be greater than 2049 for Half type.")),
        TorchDType::Float32 => Some((
            (1i64 << 24) + 1,
            "n cannot be greater than 2^24+1 for Float type.",
        )),
        TorchDType::Float64 => Some((
            (1i64 << 53) + 1,
            "n cannot be greater than 2^53+1 for Double type.",
        )),
        _ => None,
    };
    if let Some((limit, message)) = precision_limit {
        if n > limit {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(message));
        }
    }

    if label.is_meta() {
        return meta_result(py, vec![n as usize], dtype);
    }
    let device = label.resolve()?;
    let storage = PyDtype::new(dtype).storage(OP)?;
    let values = if n == 0 {
        Vec::new()
    } else {
        let mut gen = crate::rng::default_generator();
        crate::rng::randperm_fill(&mut gen, n as usize)
    };
    let tensor = randint_narrow(OP, values, vec![n as usize], storage, &device)?;
    finish(py, tensor, dtype)
}

// ---------------------------------------------------------------------------
// The `TensorBase` surface
//
// docs/C_SURFACE.md §4 measured a small Llama forward plus greedy `generate`
// and found 50 of `TensorBase`'s 694 members actually used. These are the
// kernels behind that list. They are reached from `methods.json` through the
// same resolver `torch.<op>` uses and through the same single `_aten_dispatch`
// door -- there is still no arithmetic on the `TensorBase` type itself.
//
// Two rules carried over from the ops above, because they are what makes this
// shim worth having rather than fast:
//
//   * **No silent dtype promotion between two tensors.** `same_dtype` refuses
//     and names both. torch would promote; a wrong promotion is the silent
//     numerical drift DESIGN.md §5 calls candle's main risk, and a refusal is
//     a work item.
//
//     `mul.Tensor` and `bitwise_and.Tensor` are now the exceptions, and they
//     are *not* a relaxation of that rule -- they are two work items closed.
//     They promote through `promote_types`, whose table was measured cell by
//     cell against upstream and is pinned by the golden harness over every
//     storable pair. `add`/`sub`/`div`/`bitwise_or` still refuse, because no
//     measured caller needs them to promote and inventing the answer would be
//     exactly what the rule forbids. See `promote_operands` for the two
//     callers that made these two measured.
//   * **A Python scalar does not widen a tensor of the same category.** That
//     is torch's "wrapped number" rule, measured for `pow` in
//     docs/OVERLOAD.md §6.3 and re-measured here for the arithmetic ops:
//     `float_t * 2 -> float32`, `int64_t * 2 -> int64`, `int64_t * 2.0 ->
//     float32`. True division is the exception and always floats.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Arith {
    Add,
    Sub,
    Mul,
    Div,
}

impl Arith {
    /// The same operation as seen by the fused reduced-precision kernels.
    ///
    /// It is a total function today and is written to return an `Option`
    /// anyway: adding an arm here that `reduced::Fused` does not have should
    /// fall back to the widening path rather than fail to compile into it.
    fn fused(self) -> Option<Fused> {
        Some(match self {
            Arith::Add => Fused::Add,
            Arith::Sub => Fused::Sub,
            Arith::Mul => Fused::Mul,
            Arith::Div => Fused::Div,
        })
    }
}

/// The result dtype of an arithmetic op, given the tensor's dtype and (for the
/// `Scalar` overloads) whether the Python scalar was a float.
fn arith_tag(
    op: &str,
    kind: Arith,
    tensor: TorchDType,
    scalar_is_float: Option<bool>,
) -> PyResult<TorchDType> {
    // **`torch.bool` does four different things here and the refusal used to
    // say it did one.** The whole matrix was re-measured on 2.13.0 (docs/TAIL.md
    // §2.2 found the first row of it; the rest came out of the same probe):
    //
    // ```text
    //                      add                sub        mul               div
    //   .Tensor            LOGICAL OR, bool   refuses    LOGICAL AND, bool ARITHMETIC, float32
    //   .Tensor in place   logical or, bool   refuses    logical and, bool refuses (cast back)
    //   .Scalar            ARITHMETIC, int64  refuses    ARITHMETIC, int64 ARITHMETIC, float32
    //   .Scalar in place   refuses (cast back) refuses   refuses           refuses
    // ```
    //
    // The message this used to carry -- "torch.bool operands are logical, not
    // arithmetic, in torch (BOOL.md §2.2)" -- is true of exactly two of those
    // twelve cells. `docs/BOOL.md` §2.2's table measured `x + x` for two bool
    // *tensors*, and a later round generalised that finding into a blanket
    // refusal for every overload, message included. docs/AUDIT.md reported it
    // as a live defect: for `.Scalar` upstream reads True/False as 1/0 and
    // promotes exactly like any other integral tensor, which is arithmetic and
    // not logical; and for `-` upstream refuses too, so calling it "logical"
    // describes neither side. `bool / bool` was the third wrong cell -- this
    // comment used to say "`-` and `/` are refused by upstream itself" and
    // `div.Tensor(bool, bool)` computes `[1.0, nan, 1.0]` in `float32`.
    //
    // **The refusals themselves are unchanged and all of them are still
    // right**; only the reason each one gives is. That distinction is the
    // point: a refusal with a wrong reason sends the next reader to the wrong
    // kernel, which is what happened here for two rounds.
    //
    // **`mul.Tensor` is the one exception, and it is an exception because the
    // two operations coincide rather than because it was convenient.** Under
    // the invariant BOOL.md §6.3 attaches to the `bool` tag -- the bytes are 0
    // or 1, enforced by `boolean()` being the only constructor that may attach
    // it -- an arithmetic product *is* the logical and: 1·1=1, 1·0=0, 0·0=0,
    // and no other pair of operands exists. So candle's `broadcast_mul`
    // computes torch's answer exactly, values and dtype both
    // (`tensor([T,F]) * tensor([T,T])` is `tensor([True, False])`,
    // `torch.bool`, measured on 2.13.0). That is not true of `+`, which would
    // give 2 and need clamping.
    //
    // It is here because `torch.isfinite` needs it: upstream's own body is
    // `(self == self) * (self.abs() != inf)`, a multiply of two bool tensors,
    // and that is on the `print(tensor)` path (docs/E2E_REAL.md).
    //
    // `scalar_is_float.is_none()` is exactly "this is the Tensor overload".
    if tensor == TorchDType::Bool && !(kind == Arith::Mul && scalar_is_float.is_none()) {
        // `aten.add_.Scalar` -> the name segment ends in `_`. The in-place
        // forms differ from their out-of-place twins in exactly one way that
        // matters here: the promoted result has nowhere to go.
        let in_place = op.split('.').nth(1).is_some_and(|name| name.ends_with('_'));
        let scalar_overload = scalar_is_float.is_some();
        let detail = match (kind, scalar_overload, in_place) {
            // Upstream refuses too, and not because bool `-` is logical --
            // because it has no meaning at all. `rsub.Scalar` arrives here as
            // `Arith::Sub` and upstream refuses it with the same words.
            (Arith::Sub, _, _) => "upstream refuses a torch.bool subtraction as well \
                 (\"Subtraction, the `-` operator, with a bool tensor is not supported\"), \
                 so both sides refuse; it is not implemented in torch._C shim either",
            // The promoted result cannot be written back into a bool buffer.
            (_, true, true) => "upstream refuses a torch.bool receiver in place -- the \
                 scalar promotes the result to int64 or float32 and \"result type Long \
                 can't be cast to the desired output type Bool\" -- and it is not \
                 implemented in torch._C shim",
            // The two cells where "logical, not arithmetic" is actually true.
            (Arith::Add | Arith::Mul, false, _) => "torch.bool `+` between two tensors is \
                 a logical OR returning bool (measured: [T,F] + [T,T] is [True, True]); \
                 candle would give 2, which is still truthy and so silently wrong \
                 downstream (docs/BOOL.md §6.3). Not implemented in torch._C shim",
            // Everything else: upstream COMPUTES, arithmetically, and the
            // promotion is what is missing here.
            _ => "upstream reads a torch.bool operand as 1/0 here and computes \
                 ARITHMETICALLY, promoting like any other integral tensor (bool*3 is \
                 int64 [3,0,3], bool/3 is float32, bool/bool is float32 [1.0, nan, 1.0]); \
                 that promotion is not implemented in torch._C shim",
        };
        return Err(not_implemented(format!("{op}: {detail}")));
    }
    let mut tag = tensor;
    if scalar_is_float == Some(true) && !tag.is_floating_point() {
        tag = default_float();
    }
    // torch's `/` is true division: it floats an integral pair rather than
    // truncating. `torch.tensor([1]) / torch.tensor([2])` is `0.5`, measured.
    if kind == Arith::Div && !tag.is_floating_point() {
        tag = default_float();
    }
    Ok(tag)
}

fn apply_arith(op: &str, kind: Arith, lhs: &Tensor, rhs: &Tensor) -> PyResult<Tensor> {
    match kind {
        Arith::Add => lhs.broadcast_add(rhs),
        Arith::Sub => lhs.broadcast_sub(rhs),
        Arith::Mul => lhs.broadcast_mul(rhs),
        Arith::Div => lhs.broadcast_div(rhs),
    }
    .map_err(|e| candle_err(op, e))
}

/// `alpha`, which only `add` and `sub` have. It is positional in the `Scalar`
/// schemas and keyword-only in the `Tensor` ones; `optional` covers both.
fn alpha_arg(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<f64> {
    Ok(scalar_arg(op, args, kwargs, 2, "alpha")?
        .map(|s| s.as_f64())
        .unwrap_or(1.0))
}

/// Where a dtype sits in torch's promotion lattice: the category, then the
/// width within it.
///
/// Only the dtypes `TorchDType::storage()` can hold appear; everything else
/// comes back `None` from `promote_types` below, which this shim would have
/// to refuse anyway since it cannot build the operand.
fn promotion_rank(dtype: TorchDType) -> Option<(u8, u8)> {
    use TorchDType::*;
    Some(match dtype {
        Bool => (0, 0),
        UInt8 => (1, 1),
        Int16 => (1, 2),
        Int32 => (1, 3),
        Int64 => (1, 4),
        // The two reduced floats are deliberately the SAME rank. They are not
        // ordered with respect to each other -- neither can hold the other --
        // and the tie is broken by escaping upwards, which is exactly the
        // `float16 x bfloat16 -> float32` cell.
        Float16 | BFloat16 => (2, 1),
        Float32 => (2, 2),
        Float64 => (2, 3),
        _ => return None,
    })
}

/// torch's `promote_types`, over the dtypes this shim can store.
///
/// Measured, not derived. The full 10x10 table over `{bool, uint8, uint32,
/// int16, int32, int64, float16, bfloat16, float32, float64}` was read off
/// `torch.promote_types` on 2.13.0 and separately checked cell by cell
/// against `aten.mul.Tensor`'s own result dtype -- they agree in every cell
/// where both are defined, which is why one function can serve the op.
/// Three cells break a plausible-looking shortcut:
///
/// ```text
/// int64   x float16   -> float16    an integral operand never widens a float
/// float16 x bfloat16  -> float32    two reduced floats promote OUT
/// uint8   x int16     -> int16      unsigned meets signed, no escape needed
/// ```
///
/// `uint32` (and torch's other unsigned types above `uint8`) has no promotion
/// with `bool` or a signed integer *upstream* -- "Promotion for uint16,
/// uint32, uint64 types is not supported" -- so `None` there is reproducing a
/// refusal, not admitting a gap. Against a float it does promote, and that
/// cell is answered.
fn promote_types(lhs: TorchDType, rhs: TorchDType) -> Option<TorchDType> {
    if lhs == rhs {
        return Some(lhs);
    }
    // Handled before the rank lookup, because `promotion_rank` has no entry
    // for these either and the two reasons must not be conflated: one is
    // upstream's refusal, the other would be this shim's.
    let unsigned_wide = |d: TorchDType| {
        matches!(
            d,
            TorchDType::UInt16 | TorchDType::UInt32 | TorchDType::UInt64
        )
    };
    for (a, b) in [(lhs, rhs), (rhs, lhs)] {
        if unsigned_wide(a) {
            return if b.is_floating_point() { Some(b) } else { None };
        }
    }

    let (lhs_cat, lhs_width) = promotion_rank(lhs)?;
    let (rhs_cat, rhs_width) = promotion_rank(rhs)?;
    if lhs_cat != rhs_cat {
        // A higher category wins outright, whatever the widths are:
        // `int64 x float16` is `float16`, not `float32`.
        return Some(if lhs_cat > rhs_cat { lhs } else { rhs });
    }
    match lhs_width.cmp(&rhs_width) {
        std::cmp::Ordering::Greater => Some(lhs),
        std::cmp::Ordering::Less => Some(rhs),
        // Equal rank but different dtypes: only `float16` with `bfloat16`
        // reaches here, and it escapes to `float32` rather than picking one.
        std::cmp::Ordering::Equal => Some(TorchDType::Float32),
    }
}

/// The dtype a promoting binary op computes in, with both operands named when
/// there is no answer.
///
/// **Three ops call this: `mul.Tensor`, `bitwise_and.Tensor` and
/// `pow.Tensor_Tensor`.** Everything else -- `add`, `sub`, `div`,
/// `bitwise_or` -- still goes through `same_dtype` and still refuses. That
/// split is the "no unmeasured implementation" rule (docs/E2E_REAL.md §1.2)
/// rather than an oversight: these are the ops a real forward was measured
/// stopping on, and they were found one at a time, by running it.
///
/// `pow.Tensor_Tensor` is the third and it came from `bloom`
/// (docs/ARCH20.md §6), whose `build_alibi_tensor` raises a `float32` base to
/// an `int32` power. Its cells were re-measured against
/// `pow.Tensor_Tensor`'s own result dtype rather than assumed from `mul`'s:
/// the two agree everywhere except `bool ** bool`, which `pow_result_tag`
/// refuses because upstream raises there.
///
/// `_prepare_attention_mask_for_generation` computes
///
/// ```text
/// attention_mask_from_padding * can_infer_attention_mask
///     + default_attention_mask * ~can_infer_attention_mask
/// ```
///
/// where each `*` has an `int64` left operand (`.long()`) and a 0-D `bool`
/// right one (`.any()`), while the `+` joining them is int64 with int64 --
/// which is why `mul` needed this and `add` did not. Then, once past that,
/// the sampling loop's own stopping condition
///
/// ```text
/// unfinished_sequences = unfinished_sequences & ~stopping_criteria(...)
/// ```
///
/// is `int64 & bool` (`generation/utils.py:2936`). `bitwise_and.Tensor` and
/// `bitwise_or.Tensor` were both re-measured against `torch.promote_types`
/// over the storable dtypes and agree with it in every cell, so the same
/// table serves them; only `bitwise_and` has a measured caller and only
/// `bitwise_and` is wired.
/// The dtype a promoting *n*-ary op computes in: the lattice folded left over
/// the list. `cat` and `stack` are the two.
///
/// **Folding left is upstream's own shape for this** and it is sound because
/// `promote_types` is associative and commutative over the lattice, which is
/// the property that makes the result independent of the list order. The one
/// pair that could have broken it is the reduced-float tie -- `float16` with
/// `bfloat16` escapes upwards to `float32` rather than picking a side -- and
/// it was checked rather than assumed:
///
/// ```text
/// cat([float16, bfloat16, float64])  ->  float64     upstream
/// cat([float64, float16, bfloat16])  ->  float64     upstream
/// cat([float16, bfloat16, float16])  ->  float32     upstream
/// ```
///
/// The error names the accumulated dtype against the entry that could not
/// join it, which is the pair that has no answer -- not the first two entries,
/// which may well have promoted cleanly.
/// Upstream refuses a float8 operand *before* it looks a kernel up, and says so
/// in its own words: `RuntimeError: Promotion for Float8 Types is not
/// supported, attempted to promote Float8_e4m3fn and Float`. That is a
/// different exception type and a different sentence from the
/// "dtype promotion not implemented" this shim raises for pairs it simply has
/// no rule for, and the two must not be merged -- docs/PROMOTE.md's last
/// section exists because a refusal once carried the right words under the
/// wrong type, and `except NotImplementedError` did not catch it.
fn float8_promotion_refusal(a: TorchDType, b: TorchDType) -> Option<PyErr> {
    let f8 = |t: TorchDType| matches!(t, TorchDType::Float8E4M3FN);
    if f8(a) || f8(b) {
        let (first, second) = if f8(a) { (a, b) } else { (b, a) };
        return Some(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Promotion for Float8 Types is not supported, attempted to promote {} and {}",
            scalar_type_name(first),
            scalar_type_name(second)
        )));
    }
    None
}

fn promote_list(op: &str, tensors: &[PyTensorBase]) -> PyResult<TorchDType> {
    let mut tag = tensors[0].tag();
    for other in &tensors[1..] {
        let next = other.tag();
        if next == tag {
            continue;
        }
        if let Some(err) = float8_promotion_refusal(tag, next) {
            return Err(err);
        }
        tag = promote_types(tag, next).ok_or_else(|| {
            not_implemented(format!(
                "{op}: dtype promotion not implemented in torch._C shim: {} vs {}",
                tag.name(),
                next.name()
            ))
        })?;
    }
    Ok(tag)
}

fn promote_operands(op: &str, lhs: &PyTensorBase, rhs: &PyTensorBase) -> PyResult<TorchDType> {
    if lhs.tag() == rhs.tag() {
        return Ok(lhs.tag());
    }
    if let Some(err) = float8_promotion_refusal(lhs.tag(), rhs.tag()) {
        return Err(err);
    }
    promote_types(lhs.tag(), rhs.tag()).ok_or_else(|| {
        not_implemented(format!(
            "{op}: dtype promotion not implemented in torch._C shim: {} vs {}",
            lhs.tag().name(),
            rhs.tag().name()
        ))
    })
}

/// An operand cast **to the common dtype**, which is the step that has to
/// happen before any widening to an accumulator.
///
/// This is one line and it is the whole difference between promoting and
/// *appearing* to promote. Upstream's `TensorIterator` converts every operand
/// to `common_dtype` and the kernel then reads that value into `opmath_t`;
/// the two conversions are not interchangeable with a single conversion
/// straight to `opmath_t`, because the first one can lose bits that the
/// second cannot restore.
///
/// Measured on 2.13.0:
///
/// ```text
/// sub(int64([2049]), float16([1.0]))  ->  float16  2047.0     upstream
/// ```
///
/// `promote_types(int64, float16)` is `float16`, and `float16` cannot hold
/// 2049 -- it rounds to 2048. Upstream narrows the `int64` operand *first*,
/// so it subtracts 2048 - 1 and answers 2047. Casting both operands straight
/// to the `float32` accumulator subtracts 2049 - 1 = 2048 and narrows once at
/// the end, answering **2048**. Both results are labelled `float16` and only
/// one of them is upstream's, which is exactly the failure that is invisible
/// from the outside: the dtype matches, the value does not.
///
/// The same trap is in the comparisons, where it is worse because the result
/// dtype is `bool` either way and carries no trace of where the comparison
/// happened. `eq(int64([16777217]), float32([16777216.0]))` is **True**
/// upstream -- the `int64` narrows to `float32` and the two become the same
/// number -- while comparing in a dtype wide enough to hold both gives False.
fn operand_in(op: &str, tensor: &Tensor, common: candle_core::DType) -> PyResult<Tensor> {
    tensor.fast_to(common).map_err(|e| candle_err(op, e))
}

fn arith_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Arith,
) -> PyResult<Py<PyAny>> {
    let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(op, args, kwargs, 1, "other")?;
    // **All four promote.** They did not always: `mul` was widened for
    // docs/OPS4.md and `div` for `sam3_video`, one measured caller at a time,
    // while `add` and `sub` kept `same_dtype`'s refusal. docs/PROMOTE.md §3
    // closed the split by measuring the whole 9x9 grid for all four against
    // `torch.promote_types` -- they agree in every cell, so there is one rule
    // here and no reason for two of the four to decline it.
    //
    // **`sub` still refuses a `bool` operand, and that check has to run
    // before the promotion rather than after it.** Upstream raises for
    // `bool - float32` even though `promote_types(bool, float32)` is
    // `float32`: the refusal is on the *operand*, not on the promoted type,
    // so promoting first would answer where upstream raises. `arith_tag` is
    // asked for the wording rather than it being repeated here, so the two
    // spellings of the refusal cannot drift apart.
    if kind == Arith::Sub && (lhs.tag() == TorchDType::Bool || rhs.tag() == TorchDType::Bool) {
        arith_tag(op, kind, TorchDType::Bool, None)?;
    }
    let operand = promote_operands(op, &lhs, &rhs)?;
    let tag = arith_tag(op, kind, operand, None)?;
    let storage = PyDtype::new(tag).storage(op)?;

    // Computed in `opmath_in(storage)` and narrowed once at the end -- see
    // that function. `alpha` scales inside the widened dtype for the same
    // reason `add_tensor` does it there.
    let acc = opmath_in(storage);
    let alpha = alpha_arg(op, args, kwargs)?;
    // **To the common dtype first, then to the accumulator** -- `operand_in`,
    // which is where the reason is written. Straight to `acc` is a different
    // answer whenever the common dtype is narrower than the operand.
    let left_common = operand_in(op, lhs.tensor()?, storage)?;
    let right_common = operand_in(op, rhs.tensor()?, storage)?;
    // One pass instead of three, when the operands allow it -- see
    // `add_tensor`, which takes the same fast path for the same reason. It is
    // handed the *narrowed* operands, so a promoting call reaches it too;
    // `fused_arith` still declines anything that is not already `storage`.
    if alpha == 1.0 {
        if let Some(fused) = kind.fused() {
            if let Some(out) =
                crate::reduced::fused_arith(fused, &left_common, &right_common, storage)
            {
                return finish(py, out.map_err(|e| candle_err(op, e))?, tag);
            }
        }
    }
    let left = left_common.fast_to(acc).map_err(|e| candle_err(op, e))?;
    let right = right_common.fast_to(acc).map_err(|e| candle_err(op, e))?;
    let right = scale_by_alpha(op, &right, alpha, storage)?;
    let out = apply_arith(op, kind, &left, &right)?
        .fast_to(storage)
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// Upstream's **reduced-float scalar fast path for true division**, which is
/// not the same arithmetic as the rest of the `Scalar` family.
///
/// `div_true_kernel` (ATen/native/cpu/BinaryOpsKernel.cpp) branches when the
/// second operand is a wrapped scalar and the common dtype is `Half` or
/// `BFloat16`:
///
/// ```text
/// opmath_t inv_b = opmath_t(1) / iter.original_scalar_value<opmath_t>(2);
/// ... return static_cast<opmath_t>(a) * inv_b;
/// ```
///
/// Two departures from what `arith_scalar`'s comment describes, and both are
/// only visible in `float16`:
///
///   * the divisor is the **original** scalar widened to `float`, not the
///     scalar narrowed to the tensor's dtype. `add`/`mul` do narrow it (that
///     is docs/GENERATE.md §3.2's `x + 0.3` adding `0.30078125`); `div` does
///     not, because `original_scalar_value` reads the `Scalar` rather than the
///     promoted operand.
///   * the division is turned into a **reciprocal and a multiply**, once, for
///     the whole tensor.
///
/// Measured on 2.13.0, `float16` ones divided by `0.3`: upstream answers
/// `3.333984375`, which is `f16(1.0f / 0.3f)`. Narrowing first gives
/// `1 / f16(0.3) = 3.33203125`, one representable step below -- and that is
/// what this shim answered until docs/TRAIN.md §4. **`bfloat16` cannot see the
/// difference**: both roads round to `3.328125`, which is why the existing
/// `div.Scalar` kernel passed every bfloat16 case it had and was wrong.
///
/// Returns `None` for anything but `float16`/`bfloat16`, where upstream has no
/// such branch and plain `a / b` is what runs.
fn div_scalar_reduced_float(
    op: &str,
    left: &Tensor,
    scalar: f64,
    storage: candle_core::DType,
) -> PyResult<Option<Tensor>> {
    if !matches!(storage, candle_core::DType::F16 | candle_core::DType::BF16) {
        return Ok(None);
    }
    let inv = 1.0f32 / (scalar as f32);
    let right = Tensor::full(inv, (), left.device()).map_err(|e| candle_err(op, e))?;
    Ok(Some(apply_arith(op, Arith::Mul, left, &right)?))
}

fn arith_scalar(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Arith,
) -> PyResult<Py<PyAny>> {
    let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
    let other =
        scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
    let tag = arith_tag(op, kind, lhs.tag(), Some(!other.is_int()))?;
    let storage = PyDtype::new(tag).storage(op)?;

    let acc = opmath_in(storage);
    let left = lhs.tensor()?.fast_to(acc).map_err(|e| candle_err(op, e))?;
    let alpha = alpha_arg(op, args, kwargs)?;
    // A zero-dim tensor, which is what torch's own `Scalar` overloads become
    // one layer down (`wrapped_scalar_tensor`) -- a `TorchDispatchMode` logger
    // over `f * 2` reports `aten.mul.Tensor`, not `mul.Scalar`, for exactly
    // this reason. The key stays `mul.Scalar` here because that is what the
    // *parser* picked, and the parser is what this shim reproduces.
    //
    // **Where the scalar gets rounded is a property of the kernel, not of the
    // `.Scalar` family**, and the two halves disagree. Both measured on 2.13.0
    // over 420 values per dtype; docs/SCALAR.md has the table.
    //
    //   `mul`, `div`   read the operand at `opmath_t` --
    //                  `original_scalar_value<opmath_t>(2)` in
    //                  `mul_kernel`/`div_true_kernel` reads the `Scalar`
    //                  itself, *before* the iterator's cast to the common
    //                  dtype. `bfloat16([3]) * 0.3` is `0.8984375`, which is
    //                  `bf16(3f * 0.3f)`.
    //   `add`, `sub`   have no such branch, so the operand arrives through the
    //                  iterator's common dtype and really is narrowed:
    //                  `bfloat16 + 0.3` adds `0.30078125`
    //                  (docs/GENERATE.md §3.2), and `alpha` narrows with it
    //                  (`scale_by_alpha`).
    //
    // This shim narrowed for all four until docs/SCALAR.md, which made
    // `bfloat16 * 0.3` answer `0.90234375` where upstream answers `0.8984375`.
    // It was found by a sabotage fault that *failed to fail*: the narrowing
    // made "scale the mask" and "scale the input" bit-identical here where
    // upstream separates them (docs/TRAIN.md §5, S4).
    let widen_scalar = matches!(kind, Arith::Mul | Arith::Div);
    let right = if storage.is_int() {
        Tensor::full(other.as_i64() * (alpha as i64), (), left.device()).and_then(|t| t.fast_to(acc))
    } else if widen_scalar {
        // Built at `acc` and never at `storage`: `fast_to(acc)` on an `f64`
        // fill is exactly `opmath_t(scalar)`, the `float` upstream reads.
        Tensor::full(other.as_f64() * alpha, (), left.device()).and_then(|t| t.fast_to(acc))
    } else {
        // Narrowed to `storage` and widened back, not built at `acc`: torch's
        // promotion makes a python float beside a `bfloat16` tensor a
        // `bfloat16` operand (docs/GENERATE.md §3.2), so `x + 0.3` adds
        // `0.30078125`. Building the scalar at `float` would add `0.3`.
        //
        // **`alpha` is narrowed too, and separately, and the product is
        // narrowed again.** `other * alpha` in `f64` and one narrowing at the
        // end is the obvious spelling and it disagrees: measured over 300
        // random `(other, alpha)` pairs at `bfloat16`, `bf16(bf16(other) *
        // bf16(alpha))` matches upstream 300/300 and the `f64` product 202/300
        // (`float16`: 400/400 against 260/400). `bfloat16([0.0]) + 0.3` with
        // `alpha=0.3` is `0x1.72p-4` upstream and was `0x1.70p-4` here. This
        // could not have been seen before docs/SCALAR.md §6's second bullet
        // was closed: `add.Scalar`/`sub.Scalar` were not in
        // `_aten_implemented()`, so golden had no builder for them and no case
        // had ever passed either op a non-unit `alpha`.
        //
        // `alpha = 1` is unaffected -- `narrow(1.0)` is `1.0` at every dtype
        // and `narrow(narrow(o) * 1.0)` is `narrow(o)` -- which is why no
        // prefill digest moves.
        let narrow = float_narrower(tag);
        let scaled = narrow(narrow(other.as_f64()) * narrow(alpha));
        Tensor::full(scaled, (), left.device())
            .and_then(|t| t.fast_to(storage))
            .and_then(|t| t.fast_to(acc))
    }
    .map_err(|e| candle_err(op, e))?;
    let computed = match kind {
        Arith::Div => div_scalar_reduced_float(op, &left, other.as_f64(), storage)?,
        _ => None,
    };
    let out = match computed {
        Some(t) => t,
        None => apply_arith(op, kind, &left, &right)?,
    }
    .fast_to(storage)
    .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor`
///
/// `other - alpha * self`. The reversed operand order is the whole op: torch
/// reaches it for `scalar - tensor`, which a Llama forward does in mask
/// construction, and it is *not* `sub.Scalar` with the sign flipped -- `alpha`
/// scales `self`, the subtrahend, not the scalar.
///
/// Dtype follows `sub.Scalar`'s rule exactly (`arith_tag` with `Arith::Sub`),
/// including the refusal on `torch.bool`: upstream raises there too
/// ("Subtraction, the `-` operator, with a bool tensor is not supported"),
/// so both sides refuse rather than one of them inventing a number.
fn rsub_scalar(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.rsub.Scalar";
    let lhs = tensor_arg(OP, args, kwargs, 0, "self")?;
    let other = scalar_arg(OP, args, kwargs, 1, "other")?.ok_or_else(|| missing(OP, "other"))?;
    let tag = arith_tag(OP, Arith::Sub, lhs.tag(), Some(!other.is_int()))?;
    let storage = PyDtype::new(tag).storage(OP)?;

    let acc = opmath_in(storage);
    let right = lhs.tensor()?.fast_to(acc).map_err(|e| candle_err(OP, e))?;
    let alpha = alpha_arg(OP, args, kwargs)?;
    let right = scale_by_alpha(OP, &right, alpha, storage)?;
    let left = if storage.is_int() {
        Tensor::full(other.as_i64(), (), right.device()).and_then(|t| t.fast_to(acc))
    } else {
        Tensor::full(other.as_f64(), (), right.device())
            .and_then(|t| t.fast_to(storage))
            .and_then(|t| t.fast_to(acc))
    }
    .map_err(|e| candle_err(OP, e))?;
    let out = apply_arith(OP, Arith::Sub, &left, &right)?
        .fast_to(storage)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::matmul(Tensor self, Tensor other) -> Tensor`
///
/// **Named where the parser names it, not where the dispatcher does.** torch's
/// `matmul` is `CompositeImplicitAutograd`: a `TorchDispatchMode` logger over
/// `a @ b` reports `mm.default` for a 2-D pair and
/// `expand/view/bmm/_unsafe_view` for a batched one, because the decomposition
/// runs below the parser. `THPVariable_matmul` picks `aten::matmul`, and that
/// is the key here. Recorded in docs/TENSORBASE.md as a difference in what the
/// work queue reports, not in what the call returns.
fn matmul_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.matmul.default";
    let lhs = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(OP, args, kwargs, 1, "other")?;
    let tag = require_same_dtype(OP, &lhs, &rhs)?;
    if lhs.tensor()?.rank() < 2 || rhs.tensor()?.rank() < 2 {
        // torch's 1-D rules prepend/append a dimension and remove it again.
        // Not measured as used, and guessing them is what this shim refuses.
        return Err(not_implemented(format!(
            "{OP}: matmul with a 1-D operand ({}D x {}D) is not implemented in \
             torch._C shim -- torch's vector rules were not measured",
            lhs.tensor()?.rank(),
            rhs.tensor()?.rank()
        )));
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let acc = gemm_accumulate_in(storage);
    let rhs_inner = rhs.tensor()?;
    let out = widen_gemm_operand(lhs.tensor()?, acc)
        .and_then(|l| {
            widen_gemm_operand(rhs_inner, acc).and_then(|r| batched_matmul(&l, &r))
        })
        .and_then(|p| p.fast_to(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

#[derive(Clone, Copy)]
enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Ge,
    Gt,
}

/// The comparison ops all answer `torch.bool`, and both operands are read in
/// one common representation so the comparison is exact: `f64` if either side
/// is floating, `i64` otherwise. There is no promotion step that could round
/// one side onto the other.
///
/// `le.Tensor` joined this family for `falcon`/`gptj`/`bloom`/`mpt`, which all
/// build their causal mask as `arange(...) <= arange(...)` on two `int64`
/// tensors (measured, docs/OPS4.md §1). It is a separate key from `le.Scalar`
/// -- different schema, different overload -- but the same kernel, exactly as
/// `lt.Tensor`/`lt.Scalar` already are.
fn compare_common(op: &str, tensor: &Tensor, floating: bool, tag: TorchDType) -> PyResult<Tensor> {
    let _ = tag;
    // `float8_e4m3fn` refused here until docs/FLOAT8C.md §2. `widen_f64` routes
    // it through `F32`, so `eq`/`ne` answer upstream's booleans. `lt`/`le`/
    // `ge`/`gt` are *still* refused for this dtype -- upstream refuses them by
    // kernel name (`lt_cpu` and friends) and the table at the door raises that
    // before any operand reaches here.
    if floating {
        return widen_f64(tensor).map_err(|e| candle_err(op, e));
    }
    tensor
        .to_dtype(candle_core::DType::I64)
        .map_err(|e| candle_err(op, e))
}

fn apply_cmp(op: &str, kind: Cmp, lhs: &Tensor, rhs: &Tensor) -> PyResult<Tensor> {
    match kind {
        Cmp::Eq => lhs.broadcast_eq(rhs),
        Cmp::Ne => lhs.broadcast_ne(rhs),
        Cmp::Lt => lhs.broadcast_lt(rhs),
        Cmp::Le => lhs.broadcast_le(rhs),
        Cmp::Ge => lhs.broadcast_ge(rhs),
        Cmp::Gt => lhs.broadcast_gt(rhs),
    }
    .map_err(|e| candle_err(op, e))
}

fn compare_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Cmp,
) -> PyResult<Py<PyAny>> {
    let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(op, args, kwargs, 1, "other")?;
    // Promotes. The result is `bool` for every pair (docs/PROMOTE.md §3), so
    // the promoted tag is not the *answer's* dtype -- it is the dtype the
    // comparison happens **in**, and that is a load-bearing distinction.
    //
    // `compare_common` then widens to `f64`/`i64` so candle has one kernel to
    // run, which is safe only because both operands have already been brought
    // to the common dtype. Widening straight from the originals compares two
    // numbers that upstream never compares:
    //
    // ```text
    // eq(int64([16777217]), float32([16777216.0]))  ->  True    upstream
    // ```
    //
    // 16777217 is not representable in `float32`; upstream narrows the
    // `int64` operand to the common dtype and the two become the same number.
    // Comparing in `f64` -- which holds both exactly -- answers False. The
    // result is `torch.bool` on both roads and nothing downstream can tell
    // which one ran.
    let tag = promote_operands(op, &lhs, &rhs)?;
    let storage = PyDtype::new(tag).storage(op)?;
    let floating = tag.is_floating_point();
    let left = compare_common(op, &operand_in(op, lhs.tensor()?, storage)?, floating, tag)?;
    let right = compare_common(op, &operand_in(op, rhs.tensor()?, storage)?, floating, tag)?;
    // candle's comparisons yield U8 with 0/1, which is exactly the invariant
    // `boolean()` asserts (BOOL.md §6.3).
    finish(py, apply_cmp(op, kind, &left, &right)?, TorchDType::Bool)
}

fn compare_scalar(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Cmp,
) -> PyResult<Py<PyAny>> {
    let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
    let other =
        scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
    let floating = lhs.tag().is_floating_point() || !other.is_int();
    let left = compare_common(op, lhs.tensor()?, floating, lhs.tag())?;
    let right = if floating {
        Tensor::full(other.as_f64(), (), left.device())
    } else {
        Tensor::full(other.as_i64(), (), left.device())
    }
    .map_err(|e| candle_err(op, e))?;
    finish(py, apply_cmp(op, kind, &left, &right)?, TorchDType::Bool)
}

#[derive(Clone, Copy)]
enum Bitwise {
    And,
    Or,
}

/// `bitwise_and` / `bitwise_or`, which are two different operations wearing one
/// name: logical on `torch.bool`, bit-level on the integer dtypes. That is the
/// distinction BOOL.md §3 measured and refused to collapse -- aliasing `bool`
/// onto `uint8` would make `~mask` a bit flip instead of a negation.
///
/// Computed element by element through `i64`, the same shape of implementation
/// `pow` and `isin` use above. candle has no bitwise kernels, and the ops that
/// reach here in a transformer are mask combinations, not hot arithmetic.
fn bitwise_binary(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Bitwise,
) -> PyResult<Py<PyAny>> {
    let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(op, args, kwargs, 1, "other")?;
    // Both promote, over the same table. `bitwise_or` used to refuse -- not
    // because its rule was unknown (the comment here already recorded that
    // the two had been measured to follow the same table) but because only
    // `and` had a measured caller. docs/PROMOTE.md re-measured `or`'s own 9x9
    // grid rather than inheriting `and`'s, and it agrees with
    // `torch.promote_types` in every cell, so the two share one line.
    let tag = promote_operands(op, &lhs, &rhs)?;
    if tag.is_floating_point() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "\"bitwise_{}_cpu\" not implemented for '{}'",
            match kind {
                Bitwise::And => "and",
                Bitwise::Or => "or",
            },
            scalar_type_name(tag)
        )));
    }

    let shape = lhs
        .tensor()?
        .shape()
        .broadcast_shape_binary_op(rhs.tensor()?.shape(), "bitwise")
        .map_err(|e| candle_err(op, e))?;
    let dims = shape.dims().to_vec();
    let broadcast = |t: &Tensor| -> PyResult<Vec<i64>> {
        t.broadcast_as(shape.clone())
            .and_then(|t| t.contiguous())
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_dtype(candle_core::DType::I64))
            .and_then(|t| t.to_vec1::<i64>())
            .map_err(|e| candle_err(op, e))
    };
    let (a, b) = (broadcast(lhs.tensor()?)?, broadcast(rhs.tensor()?)?);
    let values: Vec<i64> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| match kind {
            Bitwise::And => x & y,
            Bitwise::Or => x | y,
        })
        .collect();

    if tag == TorchDType::Bool {
        let bytes: Vec<u8> = values.into_iter().map(|v| u8::from(v != 0)).collect();
        let out = Tensor::from_vec(bytes, dims, lhs.tensor()?.device())
            .map_err(|e| candle_err(op, e))?;
        return finish(py, out, tag);
    }
    let storage = PyDtype::new(tag).storage(op)?;
    let out = Tensor::from_vec(values, dims, lhs.tensor()?.device())
        .and_then(|t| t.fast_to(storage))
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::bitwise_and.Scalar` / `aten::bitwise_or.Scalar`.
///
/// Unlike the arithmetic dunders, `x & 0b1010` really does keep the Python
/// number as a `Scalar` all the way down -- `TorchDispatchMode` reports
/// `bitwise_and.Scalar`, not `.Tensor` (measured; the same probe reports
/// `mul.Tensor` for `x * 2`). So this is a distinct key with a distinct
/// kernel, and the result keeps the tensor's dtype: a Python int does not
/// widen a `uint8` tensor.
fn bitwise_scalar(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Bitwise,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let other =
        scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
    let tag = input.tag();
    if tag.is_floating_point() || !other.is_int() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "\"bitwise_{}_cpu\" not implemented for '{}'",
            match kind {
                Bitwise::And => "and",
                Bitwise::Or => "or",
            },
            scalar_type_name(tag)
        )));
    }
    let rhs = other.as_i64();
    let dims = input.tensor()?.dims().to_vec();
    let values: Vec<i64> = input
        .tensor()?
        .flatten_all()
        .and_then(|t| t.to_dtype(candle_core::DType::I64))
        .and_then(|t| t.to_vec1::<i64>())
        .map_err(|e| candle_err(op, e))?
        .into_iter()
        .map(|x| match kind {
            Bitwise::And => x & rhs,
            Bitwise::Or => x | rhs,
        })
        .collect();

    if tag == TorchDType::Bool {
        let bytes: Vec<u8> = values.into_iter().map(|v| u8::from(v != 0)).collect();
        let out = Tensor::from_vec(bytes, dims, input.tensor()?.device())
            .map_err(|e| candle_err(op, e))?;
        return finish(py, out, tag);
    }
    let storage = PyDtype::new(tag).storage(op)?;
    let out = Tensor::from_vec(values, dims, input.tensor()?.device())
        .and_then(|t| t.fast_to(storage))
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::bitwise_not(Tensor self) -> Tensor`. Logical negation on `bool`,
/// two's-complement `!x` on the integers.
fn bitwise_not_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.bitwise_not.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    if tag.is_floating_point() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "\"bitwise_not_cpu\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    let dims = input.tensor()?.dims().to_vec();
    let values: Vec<i64> = input
        .tensor()?
        .flatten_all()
        .and_then(|t| t.to_dtype(candle_core::DType::I64))
        .and_then(|t| t.to_vec1::<i64>())
        .map_err(|e| candle_err(OP, e))?;

    if tag == TorchDType::Bool {
        let bytes: Vec<u8> = values.into_iter().map(|v| u8::from(v == 0)).collect();
        let out = Tensor::from_vec(bytes, dims, input.tensor()?.device())
            .map_err(|e| candle_err(OP, e))?;
        return finish(py, out, tag);
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let out = Tensor::from_vec(values.into_iter().map(|v| !v).collect::<Vec<i64>>(), dims,
                               input.tensor()?.device())
        .and_then(|t| t.fast_to(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

#[derive(Clone, Copy)]
enum Unary {
    Cos,
    Sin,
    Reciprocal,
    Tanh,
    Exp,
    Log,
    Sqrt,
    Erf,
}

/// `cos`, `sin`, `reciprocal`, `tanh`, `exp`, `log`, `sqrt` -- torch's unary float promotion, the
/// same rule `rsqrt` above already implements: a floating input keeps its own
/// dtype (`float16` in, `float16` out, *not* widened), and an integral or
/// boolean input becomes the default float.
///
/// `tanh` belongs here rather than beside `silu` even though both are
/// activations, and the difference is measured, not stylistic: `silu` has no
/// integral CPU kernel upstream and raises, while `tanh(int64 tensor)` returns
/// `float32` -- so `tanh` follows the promoting rule and `silu` does not.
/// (`tanh(bool)` promotes too: `[True, False]` gives `[0.7615942, 0.0]`.)
///
/// `exp` joined this family for `mamba` (docs/OPS4.md), which computes
/// `A = -exp(A_log)` (`A_log` a plain `float32` parameter) -- measured
/// `torch.exp` on `int64`/`bool` promotes to `float32` exactly like `tanh`
/// does, and a `float16` input stays `float16`.
///
/// `log` joined for `mamba` (docs/ARCH20.md §4), whose `init_mamba_weights`
/// computes `init.copy_(self.A_log, torch.log(A))` while the model is being
/// *constructed* -- so it is a `from_config` wall, not a forward one. The same
/// promotion holds, re-measured rather than assumed from `exp`: `log(int64)`,
/// `log(int32)`, `log(uint8)` and `log(bool)` all give `float32`, and
/// `float16`/`bfloat16` stay put.
///
/// **`log` has no domain refusal, and that is measured, not an omission.**
/// Upstream returns `-inf` for `log(0.0)` and `nan` for `log(-1.0)` rather
/// than raising, which is IEEE's answer and `f64::ln`'s, so nothing here has
/// to special-case the domain -- but it does have to be *checked*, because
/// "raises on a negative input" is the plausible wrong guess and `mamba`'s
/// `A = arange(1, state_size+1)` never leaves the positive half to reveal it.
///
/// **`sqrt` joined last, for `deberta`/`deberta_v2` (docs/KERNELS26.md §1).**
/// The asymmetry -- `rsqrt` present since RMSNorm, `sqrt` absent -- stopped
/// two architectures before any weight multiplied: `DebertaLayerNorm` computes
/// `(h - mean) / torch.sqrt(var + eps)` by hand instead of calling
/// `nn.LayerNorm`, and `deberta_v2`'s `scaled_size_sqrt` computes an attention
/// temperature through `torch.sqrt` unconditionally, relative-position or not.
///
/// It is candle's own `Tensor::sqrt` rather than a `pow(x, 0.5)` composite,
/// and the difference is measurable rather than stylistic. Three properties,
/// measured against upstream 2.13.0, all of them IEEE's:
///
/// ```text
/// sqrt(-0.0)  ->  -0.0     bit pattern 0x80000000 -- the sign of zero survives
/// sqrt(-inf)  ->   NaN     not -inf
/// sqrt(+inf)  ->  +inf
/// ```
///
/// candle's `Tensor::pow` is `exp(exponent * log(base))` (see `pow_result_tag`
/// below, which exists for exactly this reason) and answers NaN for `+inf`.
/// `powf(0.5)` would get the values right but adds a second rounding on the
/// reduced-float dtypes. The promotion was re-measured rather than inherited
/// from `rsqrt`: `int64`/`int32`/`int16`/`uint8`/`bool` all give `float32`,
/// and each float dtype keeps its own width.
/// The dtype half of the `unary_float` family, on its own so the meta kernels
/// can call it instead of restating it.
///
/// "Floating in, the same floating out; anything else becomes the default
/// float." Named rather than inlined because five call sites now share it --
/// `unary_float`, `rsqrt`, `expm1`, and the two meta arms -- and because it
/// reads `default_float()` at call time, which is what couples the rule to
/// `set_default_dtype` rather than to a constant.
fn unary_float_tag(tag: TorchDType) -> TorchDType {
    if tag.is_floating_point() {
        tag
    } else {
        default_float()
    }
}

fn unary_float(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Unary,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let tag = unary_float_tag(input.tag());
    let storage = PyDtype::new(tag).storage(op)?;
    let out = input
        .tensor()?
        .fast_to(storage)
        .and_then(|t| match kind {
            Unary::Cos => t.cos(),
            Unary::Sin => t.sin(),
            Unary::Reciprocal => t.recip(),
            Unary::Tanh => t.tanh(),
            Unary::Exp => t.exp(),
            Unary::Log => t.log(),
            Unary::Sqrt => t.sqrt(),
            // `sew_d`'s DeBERTa-style GELU spells the error function out
            // (`x * 0.5 * (1 + erf(x / sqrt(2)))`) rather than calling
            // `aten.gelu`, so this op fires on its own. It is a plain member
            // of this family -- the dtype rule is measured to be exactly the
            // family's (`int64`/`int32`/`uint8`/`bool` all give `float32`,
            // each float dtype keeps its own) -- and candle's `erf` is
            // `libm::erf`, which `gelu_default` above already records as
            // landing 4.47e-08 from upstream's own kernel at `float32`, a
            // quarter of an ulp at magnitude 1.
            Unary::Erf => t.erf(),
        })
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::expm1(Tensor self) -> Tensor`
///
/// `exp(x) - 1`, computed as one operation rather than two. `mamba`'s wall
/// (docs/ARCH20.md §4): `init_mamba_weights` inverts softplus with
/// `dt + torch.log(-torch.expm1(-dt))`, again during construction.
///
/// **It is not in the `unary_float` family even though its dtype rule is
/// exactly that family's** (measured: `int64`/`int32`/`uint8`/`bool` all give
/// `float32`, and each float dtype keeps its own), and the reason is the whole
/// point of the op. candle has no `expm1`, and `t.exp()? - 1.0` is *not* it:
/// near zero the subtraction cancels every significant bit that `exp` just
/// produced. Measured against upstream at `1e-8`, float64:
///
/// ```text
/// torch.expm1(1e-8)      1.0000000050000001e-08
/// torch.exp(1e-8) - 1    9.99999993922529e-09      wrong from the 9th digit
/// ```
///
/// So this goes through `f64::exp_m1`, element by element, the same shape of
/// implementation `pow` and `bitwise_binary` use for the same reason -- there
/// is no candle kernel and the callers are not hot loops. `read_flat` widens
/// to `f64` first, which is what makes the `float16`/`bfloat16` cases come out
/// as one correctly-rounded value rather than two roundings of a cancelled
/// subtraction.
fn expm1_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.expm1.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    // `unary_float`'s promotion, restated rather than shared because the
    // *computation* cannot be shared: the family dispatches into candle and
    // this does not.
    let tag = if input.tag().is_floating_point() {
        input.tag()
    } else {
        default_float()
    };
    let source = input.tensor()?;
    // Read at the *result* tag, not the input's: an integral input has to be
    // read as floats, since `expm1(1) = 1.718...` is not an integer.
    let values = match read_flat(OP, source, tag)? {
        Flat::Float(v) => v.into_iter().map(f64::exp_m1).collect::<Vec<f64>>(),
        // Unreachable -- `tag` is floating by construction above -- but a
        // `_ => unreachable!()` would be a panic across the FFI boundary.
        Flat::Int(v) => v.into_iter().map(|x| (x as f64).exp_m1()).collect(),
    };
    let out = write_flat(OP, Flat::Float(values), source.dims().to_vec(), source.device(), tag)?;
    finish(py, out, tag)
}

/// `aten::neg(Tensor self) -> Tensor`
///
/// **Not `unary_float`.** `neg` keeps the input dtype instead of promoting an
/// integral input to the default float -- `int64` in, `int64` out -- so it
/// cannot share that helper.
///
/// The integral path does not go through candle either. `candle_core`'s `neg`
/// is a `unary_op!`, and that macro's integer arms are `todo!()`: calling it on
/// an `i64` tensor **panics** rather than returning an error, which would take
/// the interpreter down instead of raising. So the integers are negated through
/// an `i64` round trip, the same shape `bitwise_not` already uses for the same
/// reason. `to_dtype` back to `u8` truncates, which is torch's answer too --
/// `neg(uint8 [1, 2, 0])` is `[255, 254, 0]`, measured.
///
/// Two refusals, both copied from upstream rather than invented: `bool` (torch
/// points at `~`/`logical_not()` instead) and the wide unsigned dtypes, which
/// have no `neg_cpu` kernel upstream at all.
///
/// The dtype rule *and its two refusals* are `neg_result_tag` below, so that
/// the meta kernel refuses the same two inputs. A meta kernel that accepted
/// `neg(bool_meta)` would advertise a tensor the dense kernel then declines to
/// compute -- the divergence docs/E2E_REAL.md §6.1 names.
fn neg_result_tag(tag: TorchDType) -> PyResult<TorchDType> {
    if tag == TorchDType::Bool {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Negation, the `-` operator, on a bool tensor is not supported. If you are \
             trying to invert a mask, use the `~` or `logical_not()` operator instead.",
        ));
    }
    if matches!(
        tag,
        TorchDType::UInt16 | TorchDType::UInt32 | TorchDType::UInt64
    ) {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"neg_cpu\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    Ok(tag)
}

/// `aten::sign(Tensor self) -> Tensor`
///
/// `sew_d`'s wall after `erf`: `make_log_bucket_position` in
/// `modeling_sew_d.py:160` takes `torch.sign(relative_pos)` on the
/// disentangled-attention bucket table -- measured firing once per forward on
/// a `(19, 19)` tensor.
///
/// **The dtype is the input's, on every dtype including `bool`**, which is
/// what separates it from the `unary_float` family it otherwise looks like:
/// `sign(int64)` is `int64` and `sign(bool)` is `bool` (measured), where
/// `erf(int64)` and `erf(bool)` are both `float32`. Two ops landed in the same
/// section with opposite dtype rules, so each was measured rather than one
/// inferred from the other.
///
/// Three values fix the definition, and all three are places
/// `x > 0 ? 1 : -1` -- the plausible two-way spelling -- gets it wrong:
///
/// ```text
/// sign(0.0)   0.0     not 1 and not -1: there is a zero in the range
/// sign(nan)   0.0     not nan; NaN is neither > 0 nor < 0
/// sign(-0.0)  0.0     POSITIVE zero -- measured with copysign, not with ==
/// sign(-inf) -1.0     the infinities are ordinary
/// ```
///
/// candle's `Sign` is `f32::from(v > 0.) - f32::from(v < 0.)` on the floats
/// and `min(1, v)` on the unsigned types, which gives all four -- so this
/// delegates rather than reproducing them. The `bool` case rides on the
/// unsigned arm, since `bool` and `uint8` share candle's `U8` storage and
/// `min(1, v)` is the identity on `{0, 1}`.
/// `aten::log2(Tensor self) -> Tensor`
///
/// `sam3_video`'s SAM3 detector takes `log2` of a stride ratio when it builds
/// the feature-pyramid level index.
///
/// **The dtype rule is `unary_float`'s and the computation is not**, which is
/// `expm1`'s shape exactly. `int64`, `uint8` and `bool` all give `float32` and
/// each float dtype keeps its own (measured), so the promotion is shared with
/// that family. But candle has no `log2`, and `t.log()? / ln(2)` is *not* it:
/// measured at `float64` it disagrees with `torch.log2` on 2 of 7 probe points,
/// because upstream calls `std::log2` where that divides two separately-rounded
/// values. `f64::log2` reproduces upstream exactly on every `float64` probe
/// (`math.log2` and `torch.log2` agree bit for bit on all of them).
///
/// `float16`/`bfloat16` compute in `f32` and narrow once -- measured
/// bit-identical to upstream over 2000 random points -- so the accumulate type
/// is `f32` for the three narrow floats and `f64` for `float64`, the rule
/// `sigmoid` and `avg_pool2d` also follow.
///
/// The three special values fall out of `log2` itself and are cased rather
/// than guarded: `log2(0)` is `-inf`, `log2(-1)` is NaN, `log2(inf)` is `inf`.
fn log2_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.log2.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = unary_float_tag(input.tag());
    let acc32 = tag != TorchDType::Float64;
    let source = input.tensor()?;
    // Read at the *result* tag, not the input's: an integral input has to be
    // read as floats, since `log2(3)` is not an integer.
    let values = match read_flat(OP, source, tag)? {
        Flat::Float(v) => v
            .into_iter()
            .map(|x| if acc32 { (x as f32).log2() as f64 } else { x.log2() })
            .collect::<Vec<f64>>(),
        // Unreachable -- `tag` is floating by construction above.
        Flat::Int(v) => v.into_iter().map(|x| (x as f64).log2()).collect(),
    };
    let out = write_flat(OP, Flat::Float(values), source.dims().to_vec(), source.device(), tag)?;
    finish(py, out, tag)
}

/// `aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor`
///
/// `vits`' wall after the `IntTensor` constructor: `modeling_vits.py:540`
/// runs `nn.functional.leaky_relu(hidden_states, config.leaky_relu_slope)` on
/// every HiFi-GAN decoder block.
///
/// **It is `silu`'s side of the dtype split, not `relu`'s.** `relu` has an
/// integral CPU kernel upstream and `leaky_relu` does not: measured, `int64`,
/// `uint8` and `bool` all raise `"leaky_relu_cpu" not implemented for
/// '<Type>'`, so this refuses rather than promoting. Two ops that differ by one
/// multiplication and do not share a dtype rule.
///
/// `x < 0 ? x * slope : x`, and three things fall out of writing it that way
/// rather than as `max(x, slope * x)`:
///
/// ```text
/// leaky_relu(-inf, 0.1)   -inf     the max spelling agrees here
/// leaky_relu(-1, -0.5)     0.5     a NEGATIVE slope, where max gives -1
/// leaky_relu(-0.0, 0.1)   -0.0     the sign of zero survives
/// leaky_relu(nan, 0.1)     nan
/// ```
///
/// The negative-slope row is the one that separates them, and it is not
/// hypothetical -- `negative_slope` is a `Scalar` with no sign constraint, and
/// upstream computes it. Cased.
///
/// The default slope is `0.01` and it is the schema's, not this shim's:
/// `F.leaky_relu(x)` on `-1.0` gives `-0.01`.
fn leaky_relu_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.leaky_relu.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let slope = scalar_arg(OP, args, kwargs, 1, "negative_slope")?
        .map(|s| s.as_f64())
        .unwrap_or(0.01);
    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"leaky_relu_cpu\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    // The reduced dtypes scale in `f32` and narrow once, `opmath_in`'s rule --
    // `slope * x` at `bfloat16` would round the product twice.
    let acc = opmath_in(storage);
    let source = input
        .tensor()?
        .fast_to(acc)
        .map_err(|e| candle_err(OP, e))?;
    let scaled = source
        .affine(slope, 0.0)
        .map_err(|e| candle_err(OP, e))?;
    // `x < 0`, not `x <= 0`: at `-0.0` the two differ in the sign of the
    // result, and upstream keeps the `-0.0`. `-0.0 < 0` is false, so the
    // untouched branch is taken and the sign survives; `x <= 0` would take
    // the scaled branch and `0.1 * -0.0` is `-0.0` too -- but `x <= 0` with a
    // negative slope gives `+0.0`, which is where it would show.
    let negative = source
        .lt(0f64)
        .map_err(|e| candle_err(OP, e))?;
    let out = negative
        .where_cond(&scaled, &source)
        .and_then(|t| t.fast_to(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

fn sign_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.sign.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    let out = input.tensor()?.sign().map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

fn neg_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.neg.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = neg_result_tag(input.tag())?;

    let storage = PyDtype::new(tag).storage(OP)?;
    if tag.is_floating_point() {
        let out = input
            .tensor()?
            .fast_to(storage)
            .and_then(|t| t.neg())
            .map_err(|e| candle_err(OP, e))?;
        return finish(py, out, tag);
    }

    let dims = input.tensor()?.dims().to_vec();
    let values: Vec<i64> = input
        .tensor()?
        .contiguous()
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_dtype(candle_core::DType::I64))
        .and_then(|t| t.to_vec1::<i64>())
        .map_err(|e| candle_err(OP, e))?;
    let out = Tensor::from_vec(
        values.into_iter().map(|v| v.wrapping_neg()).collect::<Vec<i64>>(),
        dims,
        input.tensor()?.device(),
    )
    .and_then(|t| t.fast_to(storage))
    .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::abs(Tensor self) -> Tensor`
///
/// The float path is candle's `abs`, which is IEEE `fabs`: `abs(-0.0)` is
/// `0.0`, `abs(-inf)` is `inf`, and `abs(nan)` is `nan` -- all three measured
/// against upstream 2.13.0 and all three agree.
///
/// **The integral path is `wrapping_abs`, not `abs`.** Upstream's answer for
/// the most negative element of a signed type is that element again:
/// `abs(int64 min)` is `int64 min`, measured. Rust's `i64::abs` panics on that
/// input in a debug build, so the round trip uses `wrapping_abs`, the same
/// shape `neg_default` above uses `wrapping_neg` for the same reason. The
/// width matters: an `int32` tensor wraps at `i32::MIN`, not at `i64::MIN`, so
/// the wrap is applied in the *storage* width before widening back.
///
/// It goes through `i64` rather than candle for the same reason `neg` does --
/// candle's `abs` is a `unary_op!` whose integer arms are `todo!()`, which
/// panics and takes the interpreter down instead of raising.
///
/// `uint8`/`uint32` are the identity, which is a fact rather than a special
/// case: no unsigned element is negative. `bool` is refused with upstream's
/// wording (`"abs_cpu" not implemented for 'Bool'`).
fn abs_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.abs.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    if tag == TorchDType::Bool {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "\"abs_cpu\" not implemented for 'Bool'",
        ));
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    if tag.is_floating_point() {
        let out = input.tensor()?.abs().map_err(|e| candle_err(OP, e))?;
        return finish(py, out, tag);
    }
    let dims = input.tensor()?.dims().to_vec();
    let values: Vec<i64> = input
        .tensor()?
        .contiguous()
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_dtype(candle_core::DType::I64))
        .and_then(|t| t.to_vec1::<i64>())
        .map_err(|e| candle_err(OP, e))?;
    let wrapped: Vec<i64> = values
        .into_iter()
        .map(|v| match storage {
            candle_core::DType::I16 => (v as i16).wrapping_abs() as i64,
            candle_core::DType::I32 => (v as i32).wrapping_abs() as i64,
            // Unsigned storages cannot hold a negative, so this is the
            // identity; `i64` is the only remaining signed width.
            _ => v.wrapping_abs(),
        })
        .collect();
    let out = Tensor::from_vec(wrapped, dims, input.tensor()?.device())
        .and_then(|t| t.fast_to(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::ceil(Tensor self) -> Tensor`
///
/// Float: candle's `ceil`. `ceil(-0.5)` is `-0.0` and keeps its sign bit,
/// `ceil(inf)` is `inf`, `ceil(nan)` is `nan` -- measured against upstream,
/// and the `-0.0` one is why `repr` cares: `_tensor_str` compares
/// `value != torch.ceil(value)` to decide whether a float tensor prints in
/// integer mode, and `repr(tensor([-0.5]))` prints `-0.` upstream.
///
/// **Integral dtypes are the identity, not a refusal.** Measured:
/// `torch.arange(3).ceil()` is `tensor([0, 1, 2])`. Only `bool` refuses, with
/// upstream's own kernel name (`"ceil_vml_cpu" not implemented for 'Bool'`) --
/// note it is a different name from `abs`'s, because upstream reaches a
/// different kernel, and copying the wrong one would send a reader to the
/// wrong place.
fn ceil_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.ceil.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    if tag == TorchDType::Bool {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "\"ceil_vml_cpu\" not implemented for 'Bool'",
        ));
    }
    if !tag.is_floating_point() {
        // Already integral: upstream hands the tensor straight back.
        let out = input.tensor()?.clone();
        return finish(py, out, tag);
    }
    let out = input.tensor()?.ceil().map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::masked_select(Tensor self, Tensor mask) -> Tensor`
///
/// Always 1-D, whatever the input rank -- it is "the elements where the mask
/// is true, in row-major order", and there is no shape that could describe
/// that in general.
///
/// Three measured behaviours it would have been easy to get wrong:
///
///   * **The mask must be `torch.bool`.** A `uint8` mask -- which reads like
///     a mask, and which old torch accepted -- raises
///     `masked_select: expected BoolTensor for mask` on 2.13.0, and so does an
///     `int64` one. Accepting them would silently treat `2` as true, which is
///     right for that value and wrong for the caller who meant a count.
///   * **Both sides broadcast**, not just the mask. `tensor([1., 2.])`
///     against a `(2, 2)` mask of `[[T,F],[T,T]]` gives `[1., 1., 2.]` --
///     the *self* tensor is the one that expanded.
///   * **An all-false mask is an empty tensor, not an error**, and it keeps
///     the input's dtype (`tensor([], dtype=torch.int64)`).
///
/// The selection is `index_select` over the flattened, broadcast input rather
/// than a per-element rebuild, so the input dtype is carried by candle instead
/// of being reconstructed -- which is what keeps `float16`/`bfloat16` exact
/// (an `f64` round trip would not be).
fn masked_select_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.masked_select.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let mask = tensor_arg(OP, args, kwargs, 1, "mask")?;
    if mask.tag() != TorchDType::Bool {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "masked_select: expected BoolTensor for mask",
        ));
    }
    let tag = input.tag();
    let shape = broadcast_shape(OP, input.tensor()?.dims(), mask.tensor()?.dims())?;

    let flat_self = input
        .tensor()?
        .broadcast_as(shape.as_slice())
        .and_then(|t| t.contiguous())
        .and_then(|t| t.flatten_all())
        .map_err(|e| candle_err(OP, e))?;
    let flat_mask: Vec<u8> = mask
        .tensor()?
        .broadcast_as(shape.as_slice())
        .and_then(|t| t.contiguous())
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_dtype(candle_core::DType::U8))
        .and_then(|t| t.to_vec1::<u8>())
        .map_err(|e| candle_err(OP, e))?;

    let picked: Vec<u32> = flat_mask
        .iter()
        .enumerate()
        .filter(|(_, m)| **m != 0)
        .map(|(i, _)| i as u32)
        .collect();

    if picked.is_empty() {
        // `narrow` rather than an empty `index_select`: it needs no index
        // tensor at all and cannot depend on how candle handles a zero-length
        // one. The result is `(0,)` in the input's dtype, as upstream's is.
        let out = flat_self.narrow(0, 0, 0).map_err(|e| candle_err(OP, e))?;
        return finish(py, out, tag);
    }
    let count = picked.len();
    let index =
        Tensor::from_vec(picked, count, flat_self.device()).map_err(|e| candle_err(OP, e))?;
    let out = flat_self
        .index_select(&index, 0)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// numpy/torch broadcast shape agreement, right-aligned: each pair of extents
/// must be equal or one of them 1, and the result takes the larger.
///
/// candle has `broadcast_as` but no public "what shape would these two
/// broadcast to?", and `masked_select` needs the answer before it can expand
/// either side. The refusal reproduces upstream's wording, which names both
/// extents and the axis -- a bare "shapes do not match" would make a broadcast
/// mistake much harder to place.
fn broadcast_shape(op: &str, lhs: &[usize], rhs: &[usize]) -> PyResult<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        let a = if i < rank - lhs.len() {
            1
        } else {
            lhs[i - (rank - lhs.len())]
        };
        let b = if i < rank - rhs.len() {
            1
        } else {
            rhs[i - (rank - rhs.len())]
        };
        out[i] = if a == b {
            a
        } else if a == 1 {
            b
        } else if b == 1 {
            a
        } else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: The size of tensor a ({a}) must match the size of tensor b ({b}) at \
                 non-singleton dimension {i}"
            )));
        };
    }
    Ok(out)
}

/// `aten::unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]`
///
/// Every slice along `dim`, with `dim` *removed* -- which is what separates it
/// from `split`, whose chunks keep the dimension with extent 1. `Tensor.
/// __iter__` is this op (`torch/_tensor.py:1215`), so `for row in tensor:` and
/// every `repr` of a tensor with more than one dimension goes through it.
///
/// Returns a `list`, not a tuple: `torch.ops.aten.unbind.int(x, 0)` on 2.13.0
/// hands back a `list` even though `Tensor.unbind` (the method binding, which
/// packs it) gives a tuple. The op-level spelling is what `_tensor_str.py:597`
/// uses, so that is the one reproduced here, matching `split.Tensor` above.
///
/// Two measured refusals rather than invented ones:
///
///   * a **0-d** input raises `IndexError: Dimension specified as 0 but tensor
///     has no dimensions` -- note `normalise_dim` would happily accept `dim=0`
///     for rank 0 (torch treats a scalar as 1-D *for indexing*), so this has
///     to be checked before it, not by it.
///   * an out-of-range `dim` raises `IndexError` naming the valid range, which
///     `normalise_dim` already spells the way upstream does.
///
/// An extent of 0 along `dim` is an **empty list**, not an error.
fn unbind_int(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.unbind.int";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let requested = dim_arg(args, kwargs, 1, "dim")?.unwrap_or(0);
    if rank == 0 {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "Dimension specified as {requested} but tensor has no dimensions"
        )));
    }
    let dim = normalise_dim(OP, requested, rank)?;
    let extent = input.tensor()?.dims()[dim];
    let tag = input.tag();

    let mut slices: Vec<Py<PyAny>> = Vec::with_capacity(extent);
    for i in 0..extent {
        let slice = input
            .tensor()?
            .narrow(dim, i, 1)
            .and_then(|t| t.squeeze(dim))
            .map_err(|e| candle_err(OP, e))?;
        slices.push(crate::tensor::promote(py, finish(py, slice, tag)?)?);
    }
    Ok(PyList::new(py, slices)?.into_any().unbind())
}

/// `aten::relu(Tensor self) -> Tensor`
///
/// Opened `opt`, `nemotron` and `persimmon` (docs/ARCH.md). The whole op is one
/// line of arithmetic and every interesting thing about it is in *which* line.
///
/// **`relu` is not `max(x, 0)`.** Two measured results rule that reading out:
///
/// ```text
/// relu([nan, inf, -inf, -0.0, 0.0]) == [nan, inf, 0.0, -0.0, 0.0]
/// signbit(relu(-0.0)) == True
/// ```
///
/// `nan` survives, so it is not a maximum against zero that would let either
/// operand win by comparison order; and `-0.0` comes back with its sign, so it
/// is not "clamp then normalise". Both fall out of `x < 0 ? 0 : x`, which is
/// what this computes: `-0.0 < 0` is false so the element passes through
/// untouched, and `nan < 0` is false for the same reason. A `max`-shaped
/// implementation would pass every ordinary test and differ on exactly these
/// two inputs -- the golden cases pin both.
///
/// **Unlike `silu`, the integral dtypes are not refused.** `relu` has an
/// integral CPU kernel upstream and `silu` does not, so the refusal that
/// belongs one function down does not belong here; only `bool` is refused, with
/// upstream's wording. On `uint8` it is the identity, which is correct rather
/// than a special case: no unsigned element is negative.
fn relu_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.relu.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    if tag == TorchDType::Bool {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Boolean inputs not supported for relu",
        ));
    }
    let source = input.tensor()?.contiguous().map_err(|e| candle_err(OP, e))?;
    let zeros = source.zeros_like().map_err(|e| candle_err(OP, e))?;
    let out = source
        .lt(&zeros)
        .and_then(|negative| negative.where_cond(&zeros, &source))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::silu(Tensor self) -> Tensor` -- `x * sigmoid(x)`, SwiGLU's activation.
///
/// Float only, and the refusal is upstream's: there is no `silu_cpu` for an
/// integral or boolean input, so an integer tensor raises here rather than
/// being promoted the way `cos`/`sin` promote theirs. (That difference is why
/// this is not another `Unary` variant.)
///
/// `float16`/`bfloat16` are computed in `f32` and narrowed once at the end.
/// candle's `silu` evaluates `v / (1 + exp(-v))` **in the input type**, which
/// rounds three times where upstream's vectorised CPU kernel rounds once, and
/// the shim's job is upstream's answer rather than candle's.
fn silu_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.silu.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"silu_cpu\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let acc = match storage {
        candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
        other => other,
    };
    let out = input
        .tensor()?
        .fast_to(acc)
        .and_then(|t| t.silu())
        .and_then(|t| t.fast_to(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::sigmoid(Tensor self) -> Tensor` -- `1 / (1 + exp(-x))`.
///
/// `sam3_video`'s wall after `all` (§16.4): the SAM3 detector squashes logits
/// with `.sigmoid()`.
///
/// **Its dtype rule is `unary_float`'s and its precision rule is `silu`'s**,
/// which is why it is neither a `Unary` variant nor a copy of `silu`:
///
///   * dtype -- an integral or boolean input promotes to the default float
///     (measured: `int64`, `int32`, `uint8` and `bool` all give `float32`),
///     unlike `silu`, which has no integral CPU kernel upstream and raises.
///     So this promotes rather than refusing.
///   * precision -- `float16`/`bfloat16` are computed in `f32` and narrowed
///     **once** at the end, unlike `tanh`/`exp`/`cos`, which the `unary_float`
///     family evaluates in the input's own dtype.
///
/// The second is measured, not inherited by analogy. Over 20 000 random
/// `randn * 8` inputs, against upstream's own answer:
///
/// ```text
///             f32-then-narrow      evaluated in the reduced dtype
/// float16     0 / 20000 differ     6983 / 20000 differ
/// bfloat16    0 / 20000 differ     5466 / 20000 differ
/// ```
///
/// So the reduced dtypes are **bit-identical** through `f32` and wrong in
/// about a third of elements without it -- a divergence a `float16` tolerance
/// of 1e-3 absorbs completely, which is why it was measured by counting exact
/// mismatches rather than by running cases.
///
/// **`float32` and `float64` are not bit-identical, and the residual is
/// `exp`'s rather than this kernel's.** Upstream computing `1/(1+exp(-x))`
/// with its *own* `exp` reproduces `torch.sigmoid` exactly at both widths
/// (0/20000 differ), so the formula is right; what differs is candle's `exp`,
/// which is already ~1 ULP from upstream's vectorised one on 12 of 80 sampled
/// `f32` points and 16 of 80 in `f64`. The sigmoid mismatches land on exactly
/// those indices -- measured, which is how "inherited" was established rather
/// than asserted. Widening `f32` to `f64` makes it *worse*, not better (20 of
/// 80 differ), so `f32` is computed in `f32`.
///
/// The saturating ends fall out of the formula rather than needing a guard:
/// `+inf` gives `exp(-inf) = 0` and hence `1`, `-inf` gives `exp(inf) = inf`
/// and hence `0`, and NaN propagates -- measured against upstream, which
/// answers `[1., 0., nan, 0., 1.]` for `[inf, -inf, nan, -100, 100]`.
fn sigmoid_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.sigmoid.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = unary_float_tag(input.tag());
    let storage = PyDtype::new(tag).storage(OP)?;
    let acc = match storage {
        candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
        other => other,
    };
    let out = input
        .tensor()?
        .fast_to(acc)
        .and_then(|t| t.neg())
        .and_then(|t| t.exp())
        .and_then(|t| t + 1.0)
        .and_then(|t| t.recip())
        .and_then(|t| t.fast_to(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::flip(Tensor self, int[] dims) -> Tensor`
///
/// `vits`' wall after `clamp_min` (§15.6): `modeling_vits.py:595` reverses the
/// channel order of the residual coupling layer's input on every flow step,
/// `torch.flip(inputs, [1])`.
///
/// **It copies; it is not a view.** Measured:
/// `torch.flip(x, [0]).data_ptr() != x.data_ptr()`. That matters here because
/// a negative-stride view is exactly what candle's `Layout` cannot express, so
/// an op that *had* to alias would have been another docs/VIEWS.md §6.4 entry.
/// It does not, so this is a complete implementation rather than a recorded
/// divergence.
///
/// Four rules, all measured on 2.13.0 with `x = arange(6).reshape(2, 3)`:
///
/// ```text
/// flip(x, [1])      [[2,1,0],[5,4,3]]    reverses WITHIN each row
/// flip(x, [0])      [[3,4,5],[0,1,2]]    reverses the row ORDER
/// flip(x, [-1])     [[2,1,0],[5,4,3]]    negative dims normalise
/// flip(x, [])       [[0,1,2],[3,4,5]]    empty dims is a COPY, not an error
/// flip(x, [0, 0])   RAISES               "dim 0 appears multiple times in the list of dims"
/// ```
///
/// The duplicate refusal is the one a delegating implementation loses:
/// flipping the same axis twice is the identity, so a kernel that just looped
/// would return the input unchanged where upstream raises. `reduce_dims` does
/// the normalisation and this checks the duplicate itself, because "reduce
/// twice over one axis" is harmless for `sum` and is not harmless here.
fn flip_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.flip.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let dims = reduce_dims_named(OP, args, kwargs, 1, "dims", rank)?
        .ok_or_else(|| missing(OP, "dims"))?;

    let mut seen: Vec<usize> = Vec::with_capacity(dims.len());
    for &dim in &dims {
        if seen.contains(&dim) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "dim {dim} appears multiple times in the list of dims"
            )));
        }
        seen.push(dim);
    }

    let tag = input.tag();
    let out = input.tensor()?.flip(&dims).map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::gelu(Tensor self, *, str approximate="none") -> Tensor`
///
/// **The op has two different functions behind one name, and picking the wrong
/// one is not a rounding error.** `approximate="none"` is the exact
/// `x·Φ(x)` written with `erf`; `approximate="tanh"` is Hendrycks' cubic
/// approximation. Measured over `[-3, 3]` in `float32` they differ by up to
/// **4.12e-04** -- four orders of magnitude past this shim's `float32` golden
/// tolerance (`1e-5`), so a shim that silently answered with the other formula
/// would not be caught by "close enough", it would be caught by a wrong token.
///
/// The default is `"none"` (upstream's schema string, re-read from
/// `torch.ops.aten.gelu.default._schema` rather than remembered). Which
/// architectures ask for which was measured, not guessed:
///
/// | architecture | `approximate` |
/// |---|---|
/// | **Gemma / Gemma-2** | **`"tanh"`** (`gelu_pytorch_tanh`) |
/// | BERT · RoBERTa · ELECTRA · DistilBERT · DeBERTa-v2 | `"none"` |
/// | BART · Falcon · GPT-NeoX · GPT-BigCode · Starcoder2 · MPT · ViT | `"none"` |
///
/// GPT-2 is the interesting absentee: it *is* a tanh-gelu model, but HF spells
/// `gelu_new` in Python, so it reaches `aten.tanh.default` and never this op.
/// Both spellings therefore have to agree, which is why the tanh branch below
/// is composed rather than delegated (see next paragraph).
///
/// **Neither branch is candle's own `gelu`/`gelu_erf`, and the reason is
/// measured.** candle's `Tensor::gelu` factors the cubic as
/// `β·v·(1 + κ·v·v)` where upstream ATen writes `β·(v + κ·v³)`; algebraically
/// identical, not identical in `float32`, and the gap is 2.98e-08 on
/// `[-3, 3]`. Writing upstream's association here makes the `float32` and
/// `float64` tanh branch reproduce torch **bit for bit** instead of merely
/// within tolerance. (The exact branch does delegate to `gelu_erf`, which is
/// `libm::erff` and lands 4.47e-08 from torch's own kernel -- there is no
/// composition that closes that one, since the two `erf` implementations
/// differ, and 4.47e-08 is a quarter-ulp at magnitude 1.)
///
/// `float16`/`bfloat16` compute in `float32` and narrow once, which is not a
/// convenience: it is what upstream does (`at::opmath_type<Half> = float`),
/// and it was verified rather than assumed -- `gelu(half x)` is *bitwise*
/// equal to `half(gelu(float(x)))` for both approximations, at every probe
/// point. The same rule `silu_default` above already follows.
///
/// Integral and boolean inputs are refused, with upstream's own message. This
/// is `silu`'s side of the split, not `tanh`'s: there is no `GeluKernelImpl`
/// for `Long`, and a shim that reused the promoting unary helper would compute
/// where torch raises.
fn gelu_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.gelu.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;

    // `approximate` is keyword-only in the schema (`*` before it), and upstream
    // enforces that: `gelu(x, "tanh")` is a TypeError-shaped RuntimeError, not
    // a tanh gelu. Reproduced, because the natural way to write this helper
    // (`optional(args, kwargs, 1, ...)`) would accept the positional form and
    // quietly implement a laxer op.
    if args.len() > 1 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "aten::gelu() takes 1 positional argument(s) but {} was/were given.  \
             Declaration: aten::gelu(Tensor self, *, str approximate=\"none\") -> Tensor",
            args.len()
        )));
    }
    let approximate = match kwargs.and_then(|kw| kw.get_item("approximate").ok().flatten()) {
        Some(value) => {
            if value.is_none() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "aten::gelu() Expected a value of type 'str' for argument \
                     'approximate' but instead found type 'NoneType'.",
                ));
            }
            value.extract::<String>()?
        }
        None => "none".to_string(),
    };

    let tag = input.tag();
    if !tag.is_floating_point() || tag == TorchDType::Float8E4M3FN {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"GeluKernelImpl\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }

    // The string is validated before any arithmetic, so a typo raises on an
    // empty tensor too -- upstream refuses `gelu(zeros(0), approximate="TANH")`
    // and a shim that only checked inside a loop would answer `[]`.
    let use_tanh = match approximate.as_str() {
        "none" => false,
        "tanh" => true,
        _ => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "approximate argument must be either none or tanh.",
            ))
        }
    };

    let storage = PyDtype::new(tag).storage(OP)?;
    let acc = match storage {
        candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
        other => other,
    };
    let x = input.tensor()?.fast_to(acc).map_err(|e| candle_err(OP, e))?;

    let out = if use_tanh {
        // `0.5 · v · (1 + tanh(β·(v + κ·v³)))`, ATen's association exactly.
        // `affine(mul, add)` narrows both constants to the tensor's dtype
        // before it multiplies, so β arrives as the `float32` nearest for a
        // `float32` tensor -- which is what upstream's `opmath_t` constant is.
        const BETA: f64 = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::SQRT_2 * 0.5;
        const KAPPA: f64 = 0.044715;
        let cube = x
            .mul(&x)
            .and_then(|sq| sq.mul(&x))
            .map_err(|e| candle_err(OP, e))?;
        let inner = cube
            .affine(KAPPA, 0.0)
            .and_then(|scaled| x.add(&scaled))
            .and_then(|sum| sum.affine(BETA, 0.0))
            .map_err(|e| candle_err(OP, e))?;
        let half = x.affine(0.5, 0.0).map_err(|e| candle_err(OP, e))?;
        inner
            .tanh()
            .and_then(|t| t.affine(1.0, 1.0))
            .and_then(|t| half.mul(&t))
            .map_err(|e| candle_err(OP, e))?
    } else {
        // `(erf(v/√2) + 1) · 0.5 · v` -- candle's `gelu_erf`, `libm::erf(f)`.
        x.gelu_erf().map_err(|e| candle_err(OP, e))?
    };

    let out = out.fast_to(storage).map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

#[derive(Clone, Copy, PartialEq)]
enum Reduce {
    Sum,
    Mean,
}

/// The dims a reduction runs over, normalised, plus whether the whole tensor
/// is being reduced.
fn reduce_dims(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    rank: usize,
) -> PyResult<Option<Vec<usize>>> {
    reduce_dims_named(op, args, kwargs, index, "dim", rank)
}

/// `reduce_dims` with the keyword name spelled out.
///
/// Every reduction here calls its axis argument `dim`; `aten::flip` calls its
/// `dims`. That is not cosmetic — the resolver binds by the *schema's* name,
/// so a `flip` reaching for `"dim"` sees no keyword at all and refuses with
/// "missing required argument". That is exactly what happened, and it happened
/// on **only the two spelling cases**: `_aten_dispatch(op, t, [1])` passes the
/// list positionally and finds it either way, so the dispatch-key cases were
/// green while `torch.flip(x, [1])` and `x.flip(1)` both refused.
fn reduce_dims_named(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
    rank: usize,
) -> PyResult<Option<Vec<usize>>> {
    let value = match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => value,
        _ => return Ok(None),
    };
    let raw: Vec<isize> = match value.extract::<Vec<isize>>() {
        Ok(list) => list,
        Err(_) => vec![value.extract::<isize>()?],
    };
    if raw.is_empty() {
        // torch: an empty dim list reduces nothing but still runs, which is
        // not the same as `dim=None`.
        return Ok(Some(Vec::new()));
    }
    raw.into_iter()
        .map(|d| normalise_dim(op, d, rank))
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

/// `sum` and `mean`, in their whole-tensor and per-dimension forms.
///
/// The dtype rules are torch's, measured: `sum` promotes every non-floating
/// input to `int64` (`bool_t.sum() -> int64`, `int32_t.sum() -> int64`) while
/// a floating input keeps its dtype, and `mean` refuses a non-floating input
/// outright rather than promoting it.
fn sum_or_mean(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Reduce,
    has_dim: bool,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let (dim_at, keepdim_at, dtype_at) = if has_dim { (1, 2, 3) } else { (99, 99, 1) };
    let dims = if has_dim {
        reduce_dims(op, args, kwargs, dim_at, rank)?
    } else {
        None
    };
    let keepdim = if has_dim {
        bool_arg(args, kwargs, keepdim_at, "keepdim")?.unwrap_or(false)
    } else {
        false
    };

    let natural = match kind {
        Reduce::Sum => {
            if input.tag().is_floating_point() {
                input.tag()
            } else {
                TorchDType::Int64
            }
        }
        Reduce::Mean => {
            if !input.tag().is_floating_point() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "mean(): could not infer output dtype. Input dtype must be either \
                     a floating point or complex dtype. Got: {}",
                    input.tag().name()
                )));
            }
            input.tag()
        }
    };
    let tag = dtype_arg(args, kwargs, dtype_at, "dtype")?.unwrap_or(natural);
    let storage = PyDtype::new(tag).storage(op)?;

    // Reduced in `acc_type<T>` -- `float` for both reduced floats -- and
    // narrowed once, which is what torch's reduction kernels do and what
    // `cumsum_default` below already says in its own doc comment. Reducing in
    // `bfloat16` instead rounds after every partial sum: measured, 168 of 200
    // rows of a 64-wide `bfloat16` sum differed from upstream. `opmath_in`.
    let acc = opmath_in(storage);
    let source = input
        .tensor()?
        .fast_to(acc)
        .map_err(|e| candle_err(op, e))?;
    // torch: an empty `dim` list reduces *every* dimension (it is not the
    // same as reducing none), so it is equivalent to naming every axis.
    let dims = dims.map(|d| if d.is_empty() { (0..rank).collect() } else { d });
    let out = match dims {
        None => match kind {
            Reduce::Sum => source.sum_all(),
            Reduce::Mean => source.mean_all(),
        },
        Some(dims) => match (kind, keepdim) {
            (Reduce::Sum, true) => source.sum_keepdim(dims),
            (Reduce::Sum, false) => source.sum(dims),
            (Reduce::Mean, true) => source.mean_keepdim(dims),
            (Reduce::Mean, false) => source.mean(dims),
        },
    }
    .and_then(|t| t.fast_to(storage))
    .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None)`. Same
/// integral-to-int64 promotion as `sum`.
fn cumsum_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.cumsum.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let dim = normalise_dim(
        OP,
        dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?,
        input.tensor()?.rank(),
    )?;
    let natural = if input.tag().is_floating_point() {
        input.tag()
    } else {
        TorchDType::Int64
    };
    let tag = dtype_arg(args, kwargs, 2, "dtype")?.unwrap_or(natural);
    let storage = PyDtype::new(tag).storage(OP)?;

    // Accumulated here rather than with `candle_core::Tensor::cumsum`, which
    // is a matmul against a triangular matrix and so only exists for the
    // dtypes candle's gemm covers -- the golden harness caught exactly that on
    // `int64` and `bfloat16` (`unsupported dtype I64 for op matmul`).
    //
    // Floating results accumulate in `f64`. torch's CPU kernel accumulates the
    // reduced-precision floats in `float` (`acc_type<BFloat16>`) and narrows
    // once at the end, so this is the same shape of computation with a wider
    // accumulator: it can differ from torch in the last bit of a long
    // `bfloat16` run, in the more-accurate direction. docs/TENSORBASE.md.
    let dims = input.tensor()?.dims().to_vec();
    let n = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    let outer: usize = dims[..dim].iter().product();

    let out = if storage.is_int() {
        let mut flat: Vec<i64> = input
            .tensor()?
            .flatten_all()
            .and_then(|t| t.to_dtype(candle_core::DType::I64))
            .and_then(|t| t.to_vec1::<i64>())
            .map_err(|e| candle_err(OP, e))?;
        for o in 0..outer {
            for k in 0..inner {
                let base = o * n * inner + k;
                for i in 1..n {
                    // Wrapping, like torch's integer kernels.
                    flat[base + i * inner] =
                        flat[base + i * inner].wrapping_add(flat[base + (i - 1) * inner]);
                }
            }
        }
        Tensor::from_vec(flat, dims, input.tensor()?.device())
    } else {
        let mut flat: Vec<f64> = input
            .tensor()?
            .flatten_all()
            .and_then(|t| widen_f64(&t))
            .and_then(|t| t.to_vec1::<f64>())
            .map_err(|e| candle_err(OP, e))?;
        for o in 0..outer {
            for k in 0..inner {
                let base = o * n * inner + k;
                for i in 1..n {
                    flat[base + i * inner] += flat[base + (i - 1) * inner];
                }
            }
        }
        Tensor::from_vec(flat, dims, input.tensor()?.device())
    }
    .and_then(|t| t.fast_to(storage))
    .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

#[derive(Clone, Copy)]
enum Extremum {
    Max,
    Min,
}

/// `aten::max(Tensor self) -> Tensor` and `aten::min(Tensor self) -> Tensor` --
/// the whole-tensor forms, a zero-dim result in the input's own dtype.
///
/// **NaN propagates, and candle does not do that for us.** This started as
/// `max` alone, written as `flatten_all().max(0)`, and that was a wrong answer
/// nobody had asked the right question of: candle's reduction skips NaN, so
/// `max([3, nan, 1])` came back `3.0` where upstream 2.13.0 gives `nan`
/// (measured, both sides, docs/E2E_REAL.md). Torch's rule is the IEEE
/// *maximum*/*minimum* rule rather than `fmax`/`fmin` -- a NaN anywhere in the
/// input is the answer, because there is no ordering that would let a real
/// number beat it.
///
/// It went unnoticed because `max_default_cases` had no NaN case; the
/// `_pair_result_check` in the same harness has explicit NaN handling for
/// `sort`/`topk`, so the harness could always have caught this and simply was
/// never asked. It is caught now, on both ops.
///
/// The NaN test is one extra vectorised pass (`x != x`, summed), not a
/// per-element scan, and it only runs for floating dtypes -- an integer or
/// boolean tensor has no NaN to find, so those take candle's reduction
/// directly.
///
/// `min` is here rather than beside it as a copy because the two differ in one
/// token; upstream's empty-input refusal names the op, so that message is
/// built from the same `match`.
fn extremum_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    which: Extremum,
) -> PyResult<Py<PyAny>> {
    let (op, name) = match which {
        Extremum::Max => ("aten.max.default", "max"),
        Extremum::Min => ("aten.min.default", "min"),
    };
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    if input.tensor()?.elem_count() == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{name}(): Expected reduction dim to be specified for input.numel() == 0."
        )));
    }
    let tag = input.tag();
    let flat = input
        .tensor()?
        .flatten_all()
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(op, e))?;

    if tag.is_floating_point() {
        let nan_count = flat
            .ne(&flat)
            .and_then(|m| m.to_dtype(candle_core::DType::I64))
            .and_then(|m| m.sum_all())
            .and_then(|s| s.to_scalar::<i64>())
            .map_err(|e| candle_err(op, e))?;
        if nan_count > 0 {
            let storage = PyDtype::new(tag).storage(op)?;
            let out = Tensor::full(f64::NAN, (), flat.device())
                .and_then(|t| t.fast_to(storage))
                .map_err(|e| candle_err(op, e))?;
            return finish(py, out, tag);
        }
    }

    let out = match which {
        Extremum::Max => flat.max(0),
        Extremum::Min => flat.min(0),
    }
    .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor`
///
/// The maximum *value* over some dimensions, with no indices -- which is what
/// every softmax wants and what candle does not have. `crate::tensor::
/// amax_keepdim` is the kernel and its header carries the argument for why it
/// exists; this is the op key that makes it reachable and comparable.
///
/// Four behaviours are upstream's, measured on torch 2.13.0 rather than
/// inferred, and three of them differ from the `sum`/`mean` reductions next
/// door:
///
///   * **`dim=[]` reduces *everything*.** For `sum.dim_IntList` an empty list
///     reduces nothing (`reduce_dims`' own comment says so); for `amax` it is
///     the schema default and it answers a scalar. Copying `sum`'s reading here
///     would make the no-argument call a no-op.
///   * **A repeated dimension is refused**, `RuntimeError: dim 1 appears
///     multiple times in the list of dims`. `sum` accepts one.
///   * **An empty input raises two different exceptions**, and which one
///     depends on whether a dim was named: `numel() == 0` with no dim is a
///     `RuntimeError` telling the caller to pass one, and a named dim of extent
///     zero is an `IndexError`. Both messages are transcribed.
///   * **Every dtype is allowed, `torch.bool` included** (`bool_t.amax()` is
///     `torch.bool`), and the result keeps the input's dtype -- no `int64`
///     promotion of the kind `sum` does.
fn amax_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.amax.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let named = reduce_dims(OP, args, kwargs, 1, rank)?;
    let keepdim = bool_arg(args, kwargs, 2, "keepdim")?.unwrap_or(false);

    // `None` (absent) and `Some([])` are the same request here -- the schema's
    // default is `[]` and it means every dimension.
    let all_dims = named.as_ref().map_or(true, |d| d.is_empty());
    let mut dims: Vec<usize> = match named {
        Some(d) if !d.is_empty() => d,
        _ => (0..rank).collect(),
    };

    if let Some(repeated) = first_repeat(&dims) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "dim {repeated} appears multiple times in the list of dims"
        )));
    }

    let extents = input.tensor()?.dims().to_vec();
    if input.tensor()?.elem_count() == 0 {
        if all_dims {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "amax(): Expected reduction dim to be specified for input.numel() == 0. \
                 Specify the reduction dim with the 'dim' argument.",
            ));
        }
        if let Some(&empty) = dims.iter().find(|&&d| extents[d] == 0) {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "amax(): Expected reduction dim {empty} to have non-zero size."
            )));
        }
    }

    // Descending, so the squeezes below run outermost-last and each index is
    // still valid when its turn comes. The maximum itself does not care about
    // the order -- it is associative and commutative.
    dims.sort_unstable();
    dims.dedup();
    let mut out = input.tensor()?.clone();
    for &dim in dims.iter() {
        out = crate::tensor::amax_keepdim(&out, dim).map_err(|e| candle_err(OP, e))?;
    }
    if !keepdim {
        for &dim in dims.iter().rev() {
            out = out.squeeze(dim).map_err(|e| candle_err(OP, e))?;
        }
    }
    finish(py, out, input.tag())
}

/// Where the NaNs are along one dimension: `(any, first)`, both keeping the
/// reduced dimension, or `None` when there is nothing to correct.
///
/// **This is the third repair of one predicate and it is meant to be the
/// last.** candle's reduction and comparison kernels all fold with `|x, y| x <
/// y`, and every comparison against a NaN is false, so a NaN that is not the
/// element the accumulator *started* on is skipped. Three ops had shipped that
/// answer: `max.default`/`min.default` (docs/E2E_REAL.md), `max.other`'s second
/// operand (docs/SPELLINGS.md §7.2), and `max.dim`, which dropped both the
/// value and the index. `tensor::amax_keepdim` was written specifically to
/// avoid it (docs/SEQLEN.md §7.2). Rather than a fourth hand-rolled repair,
/// every reduction in the family now asks this one function.
///
/// **`amax`'s `CustomOp1` is not the mechanism here, and the reason is
/// structural rather than a preference.** That kernel is fast because it drops
/// the index, which lets sixteen accumulator lanes run without a loop-carried
/// compare-and-select. `max.dim`, `min.dim` and `argmax` *need* the index, so
/// `cpu_backend::ReduceIndex` runs whatever this does; routing their values
/// through `amax` as well would add a pass rather than remove one. What
/// transfers is the *rule*, not the kernel -- and the rule costs two
/// vectorised passes (`ne`, then one reduction over a 0/1 mask), the same
/// shape of correction `extremum_default` above already pays.
///
/// Two measured facts hold this together, both read off torch 2.13.0 rather
/// than reasoned about:
///
///   * `max(dim=)` and `min(dim=)` report the index of the **first** NaN in the
///     slice, not of the extremum among the non-NaN elements:
///     `max([1, nan, nan], dim=0)` is `(nan, 1)`.
///   * `argmax`/`argmin` report that same index -- `argmax([1, nan, 3])` is
///     `1`, not `2`.
///
/// So "the first NaN" is the only position any of them needs, and `argmax` over
/// the 0/1 NaN mask *is* that position: candle's own reduction keeps the first
/// of two equal elements, which is the one respect in which its fold is
/// exactly right.
///
/// Returns `None` for an integral or boolean dtype (there is no NaN to find,
/// and the mask passes would be pure cost) **and** for a float tensor that
/// happens to contain none. The second case is not just an optimisation: it
/// keeps a NaN-free reduction bit-for-bit on the path it already took, so the
/// prefill hash cannot move because of a correction that never applies.
fn nan_along_dim(
    op: &str,
    source: &Tensor,
    dim: usize,
    tag: TorchDType,
) -> PyResult<Option<(Tensor, Tensor)>> {
    if !tag.is_floating_point() {
        return Ok(None);
    }
    // `f32` rather than the `u8` that `ne` yields: candle generates `argmax`
    // for the float and wide-integer arms, and `u8` is not one of them.
    let flags = source
        .ne(source)
        .and_then(|m| m.to_dtype(candle_core::DType::F32))
        .map_err(|e| candle_err(op, e))?;
    let total = flags
        .sum_all()
        .and_then(|s| s.to_scalar::<f32>())
        .map_err(|e| candle_err(op, e))?;
    if total == 0.0 {
        return Ok(None);
    }
    let any = flags
        .max_keepdim(dim)
        .and_then(|m| m.ne(0f32))
        .map_err(|e| candle_err(op, e))?;
    let first = flags
        .argmax_keepdim(dim)
        .and_then(|t| t.to_dtype(candle_core::DType::I64))
        .map_err(|e| candle_err(op, e))?;
    Ok(Some((any, first)))
}

/// A NaN of `tag`'s dtype, shaped like `like`. Built through `f64` and
/// `fast_to` for the same reason `extremum_default` does -- `Tensor::full`
/// takes one Rust scalar type, and the tag decides the storage.
fn nan_shaped_like(op: &str, like: &Tensor, tag: TorchDType) -> PyResult<Tensor> {
    let storage = PyDtype::new(tag).storage(op)?;
    Tensor::full(f64::NAN, like.shape(), like.device())
        .and_then(|t| t.fast_to(storage))
        .map_err(|e| candle_err(op, e))
}

/// The first value that occurs twice in `dims`, if any. Written out rather than
/// sorted-and-scanned because the message upstream prints names the *repeated
/// dimension*, not its position, and sorting would lose which one arrived first
/// only if there were more than one -- there is not, because this returns at
/// the first.
fn first_repeat(dims: &[usize]) -> Option<usize> {
    for (i, d) in dims.iter().enumerate() {
        if dims[..i].contains(d) {
            return Some(*d);
        }
    }
    None
}

/// Which side of the diagonal survives.
#[derive(Clone, Copy)]
enum Triangle {
    Lower,
    Upper,
}

/// `aten::tril(Tensor self, SymInt diagonal=0) -> Tensor` and
/// `aten::triu(Tensor self, SymInt diagonal=0) -> Tensor`.
///
/// The last two dimensions are read as a matrix and everything on the wrong
/// side of the `diagonal`-th diagonal is zeroed; leading dimensions are a
/// batch, and the same mask applies to every matrix in it. **The sign
/// convention is read off `native_functions.yaml` and then measured, because it
/// is the one thing here that fails silently if it is backwards:**
///
/// ```text
/// tril  keeps  j - i <= diagonal      triu  keeps  j - i >= diagonal
///
/// tril(ones(3,3), -1)   strictly below      triu(ones(3,3), 1)   strictly above
/// tril(ones(3,3),  1)   one band extra      triu(ones(3,3), -1)  one band extra
/// ```
///
/// A positive `diagonal` moves the boundary *up and right* for both, so it
/// widens `tril` and narrows `triu`. Both are unbounded -- `tril(x, 100)` is
/// `x` and `tril(x, -100)` is all zeros -- so the offset is not range-checked,
/// only compared.
///
/// **The zeroing is a select, not a multiply by a mask.** `nan * 0` is `nan`
/// and `inf * 0` is `nan`, and upstream zeroes those positions like any other
/// (measured: `tril([[1, nan], [inf, -inf]])` is `[[1, 0], [inf, -inf]]`, and
/// `triu` of the same drops the `inf` cleanly). A masked multiply would turn a
/// masked-out `-inf` into a `nan`, which is precisely the kind of
/// plausible-looking wrong answer this repository keeps finding.
///
/// Every dtype passes through unchanged, `torch.bool` included -- which is the
/// call GPT-BigCode actually makes: `torch.tril(torch.ones((n, n),
/// dtype=torch.bool))` as its causal-mask buffer (docs/TORCHSCRIPT.md §6).
///
/// Rank is checked first and refused with upstream's own wording; a 1-D or
/// 0-D input has no diagonal to speak of.
fn tril_triu(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    which: Triangle,
) -> PyResult<Py<PyAny>> {
    let (op, name) = match which {
        Triangle::Lower => ("aten.tril.default", "tril"),
        Triangle::Upper => ("aten.triu.default", "triu"),
    };
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let diagonal = int_arg(args, kwargs, 1, "diagonal")?.unwrap_or(0);
    let tag = input.tag();
    let dims = input.tensor()?.dims().to_vec();
    if dims.len() < 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{name}: input tensor must have at least 2 dimensions"
        )));
    }
    // Contiguous first -- and this is defensive rather than load-bearing,
    // which was measured rather than assumed. Removing it was injected as a
    // deliberate fault (docs/TRIL.md §5, fault 3) and **no test failed**:
    // candle's `WCond` matches on `contiguous_offsets()` and falls back to
    // `strided_index()` for all three operands, so a transposed `on_true` is
    // read by position-in-the-matrix already. `tril(x.t())`,
    // `tril(z.transpose(1, 2))` and `tril(z[:, :, 1:3])` all answer correctly
    // without it.
    //
    // Kept anyway, and the reason is stated so the next reader does not have
    // to re-derive it: the mask below is built row-major and handed to
    // `broadcast_as`, so the kernel's correctness would rest on an internal
    // detail of candle's cpu backend rather than on anything this function
    // establishes. One copy on an input that is almost never non-contiguous
    // (GPT-BigCode's is a fresh `ones`) buys not having that dependency.
    let source = input
        .tensor()?
        .contiguous()
        .map_err(|e| candle_err(op, e))?;
    if source.elem_count() == 0 {
        // `tril(empty(0, 3))` is `empty(0, 3)` -- shape preserved, nothing to
        // zero. Returned before the mask is built because a zero-extent
        // `where_cond` is an edge case with no work in it either way.
        return finish(py, source, tag);
    }

    let cols = dims[dims.len() - 1];
    let rows = dims[dims.len() - 2];
    let mut mask = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let offset = j as i64 - i as i64;
            let keep = match which {
                Triangle::Lower => offset <= diagonal,
                Triangle::Upper => offset >= diagonal,
            };
            mask.push(u8::from(keep));
        }
    }
    let mask = Tensor::from_vec(mask, (rows, cols), source.device())
        .and_then(|m| m.broadcast_as(source.shape()))
        .and_then(|m| m.contiguous())
        .map_err(|e| candle_err(op, e))?;
    let zeros = Tensor::zeros(source.shape(), source.dtype(), source.device())
        .map_err(|e| candle_err(op, e))?;
    let out = mask
        .where_cond(&source, &zeros)
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::max.other(Tensor self, Tensor other)` and
/// `aten::min.other(Tensor self, Tensor other)` -- elementwise, and upstream
/// documents both as aliases of `maximum`/`minimum` (measured: `torch.max(a,
/// b)` dispatches to `aten::maximum`, `torch.min(a, b)` to `aten::minimum`).
///
/// **The NaN rule is IEEE `maximum`/`minimum`, not `fmax`/`fmin`: a NaN on
/// *either* side wins.** candle's `broadcast_maximum` is `|x, y| x > y`
/// elementwise, which propagates a NaN in the first operand (nothing displaces
/// it) and drops one in the second. docs/SPELLINGS.md §7.2 found that
/// asymmetry and pinned it as a failing golden case rather than fixing it;
/// this is the fix. The correction is a mask over the *broadcast* shape, since
/// either operand's NaN has to reach every element it broadcasts to --
/// `max.other([1., 2.], [nan])` is `[nan, nan]` upstream.
///
/// Skipped entirely when neither operand holds a NaN, so a NaN-free call keeps
/// the bits candle's own kernel produced.
fn extremum_other(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    which: Extremum,
) -> PyResult<Py<PyAny>> {
    let op = match which {
        Extremum::Max => "aten.max.other",
        Extremum::Min => "aten.min.other",
    };
    let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(op, args, kwargs, 1, "other")?;
    // Promotes over the lattice (docs/PROMOTE.md §3). Both operands are
    // brought to the common dtype before the comparison, for `operand_in`'s
    // reason -- the elementwise maximum of a narrowed pair is not always the
    // narrowing of the maximum.
    let tag = promote_operands(op, &lhs, &rhs)?;
    let storage = PyDtype::new(tag).storage(op)?;
    let a = &operand_in(op, lhs.tensor()?, storage)?;
    let b = &operand_in(op, rhs.tensor()?, storage)?;
    let out = match which {
        Extremum::Max => a.broadcast_maximum(b),
        Extremum::Min => a.broadcast_minimum(b),
    }
    .map_err(|e| candle_err(op, e))?;

    if !tag.is_floating_point() {
        return finish(py, out, tag);
    }
    // "either side is NaN", broadcast to the result shape. Added rather than
    // or-ed because candle's logical ops are on masks of one shape and
    // `broadcast_add` is the operation that already does the shape join; the
    // sum of two 0/1 masks is non-zero exactly where at least one is set.
    let either = a
        .ne(a)
        .and_then(|m| m.to_dtype(candle_core::DType::F32))
        .and_then(|m| {
            b.ne(b)
                .and_then(|n| n.to_dtype(candle_core::DType::F32))
                .and_then(|n| m.broadcast_add(&n))
        })
        .map_err(|e| candle_err(op, e))?;
    let total = either
        .sum_all()
        .and_then(|s| s.to_scalar::<f32>())
        .map_err(|e| candle_err(op, e))?;
    if total == 0.0 {
        return finish(py, out, tag);
    }
    let nans = nan_shaped_like(op, &out, tag)?;
    let out = either
        .ne(0f32)
        .and_then(|cond| cond.where_cond(&nans, &out))
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// Grouped-query attention: the key/value head dimension repeated up to the
/// query's.
///
/// **This belongs in the aten op, and that was measured rather than
/// assumed.** A `TorchDispatchMode` over
/// `F.scaled_dot_product_attention(q, k, v, enable_gqa=True)` with
/// `q = (2, 9, 4, 8)` and `k = v = (2, 3, 4, 8)` reports exactly one op --
/// `aten._scaled_dot_product_flash_attention_for_cpu.default` -- with the key
/// and value *still* `(2, 3, 4, 8)`. Nothing repeats them on the way in, and
/// the aten op has no `enable_gqa` argument at all: calling it directly with
/// those mismatched shapes answers `(2, 9, 4, 8)` and agrees with the
/// `enable_gqa=True` result to `0.0`. So the flag is a validation switch in
/// the Python wrapper and the broadcast is the op's own behaviour.
///
/// **Which repetition** is the part that fails plausibly rather than loudly.
/// Measured three ways on those shapes:
///
/// ```text
/// repeat_interleave(3, dim=1)               0.0
/// transformers' repeat_kv (expand+reshape)  0.0
/// repeat(1, 3, 1, 1)  ("tile")              2.82
/// ```
///
/// Both correct spellings give query head `i` the key/value head `i / n_rep`;
/// tiling gives it `i % n_rep`. Tiling produces a same-shaped,
/// same-magnitude, entirely wrong answer -- the failure mode docs/ARCH.md
/// §5.1 records for `gelu`, where the logits look reasonable and are not.
/// This uses the `unsqueeze`/`expand`/`reshape` spelling, which is
/// `repeat_interleave` along one axis and is what transformers' own
/// `repeat_kv` does; it moves bytes and computes nothing, so there is no
/// rounding question attached to the choice.
///
/// **Non-divisible head counts are refused by name.** Upstream does not
/// refuse them -- it answers, deterministically, and part of the answer is
/// garbage. Measured per query head against `kv_head = q_head / (h_q /
/// h_kv)`:
///
/// ```text
/// h_q=9 h_kv=4   heads 0..7 agree to 0.0, head 8 differs by 0.93
/// h_q=9 h_kv=2   heads 0..7 agree to 0.0, head 8 differs by 2.28
/// h_q=6 h_kv=4   heads 0..3 agree to 0.0, head 4 by 0.78, head 5 by 2.38e+31
/// ```
///
/// That is this same rule for the first `h_kv * (h_q / h_kv)` heads and an
/// out-of-bounds read for the remainder -- 2.38e+31 is not a value an
/// attention output can take with unit-magnitude inputs. There is nothing
/// there to reproduce, so this refuses and says so. Every caller arriving
/// through `F.scaled_dot_product_attention` is stopped one layer earlier by
/// the divisibility check upstream also has.
fn repeat_kv_heads(op: &str, kv: &Tensor, query_heads: usize) -> PyResult<Tensor> {
    let dims = kv.dims().to_vec();
    let kv_heads = dims[1];
    if kv_heads == query_heads {
        return Ok(kv.clone());
    }
    if kv_heads == 0 || query_heads % kv_heads != 0 {
        return Err(not_implemented(format!(
            "{op}: grouped-query attention with {query_heads} query heads and {kv_heads} \
             key/value heads is not implemented in torch._C shim -- the head counts do not \
             divide. Upstream answers here, but its answer reads past the end of the \
             key/value tensor for the leftover heads (measured), so there is nothing to \
             reproduce"
        )));
    }
    let n_rep = query_heads / kv_heads;
    let (b, s, e) = (dims[0], dims[2], dims[3]);
    kv.unsqueeze(2)
        .and_then(|t| t.expand((b, kv_heads, n_rep, s, e)))
        .and_then(|t| t.reshape((b, query_heads, s, e)))
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(op, e))
}

/// The `(values, indices)` pair `max.dim` and `min.dim` return.
///
/// Upstream's is a *structseq* from `torch.return_types`, built by `_C` and
/// re-exported by `torch/return_types.py`. This shim does not own that
/// machinery, so the pair is a `collections.namedtuple` with the same two
/// field names: index access and `.values`/`.indices` both work, and the type
/// is not `torch.return_types.max`. Recorded in docs/TENSORBASE.md.
///
/// One cache per overload rather than one shared type, because the type's
/// `__name__` is the only thing distinguishing them and `repr()` prints it:
/// upstream shows `torch.return_types.min(values=..., indices=...)` for the
/// minimum, and a shared `max`-named tuple would print the wrong op in every
/// traceback and doctest that touches it.
static MAX_RESULT: std::sync::OnceLock<Py<PyAny>> = std::sync::OnceLock::new();
static MIN_RESULT: std::sync::OnceLock<Py<PyAny>> = std::sync::OnceLock::new();

fn extremum_result_type(py: Python<'_>, which: Extremum) -> PyResult<&'static Py<PyAny>> {
    let (cell, name) = match which {
        Extremum::Max => (&MAX_RESULT, "max"),
        Extremum::Min => (&MIN_RESULT, "min"),
    };
    if let Some(cached) = cell.get() {
        return Ok(cached);
    }
    let namedtuple = py
        .import("collections")?
        .getattr("namedtuple")?
        .call1((name, ("values", "indices")))?
        .unbind();
    let _ = cell.set(namedtuple);
    Ok(cell.get().expect("just set"))
}

/// `aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values,
/// Tensor indices)` and its `min` mirror.
///
/// **Both halves of the pair had the dropped-NaN fault, and dropping the index
/// is not an option here** -- the pair is the whole reason this overload exists
/// rather than `amax`. `max([1., nan, 3.], dim=0)` answered `(3.0, 2)` where
/// upstream answers `(nan, 1)`: candle's `max_keepdim` skipped the NaN it did
/// not start on, and its `argmax_keepdim` skipped it in exactly the same way,
/// so the two were consistently wrong together. `nan_along_dim` above supplies
/// both replacements from one mask -- see its header for why `amax`'s
/// `CustomOp1` is not the mechanism, and for the measurement that says the
/// index upstream reports is the *first* NaN's.
///
/// `min.dim` had no kernel at all until now; docs/SPELLINGS.md §7.2 left it and
/// `min.other` named in `overloads.json`/`methods.json` so they would refuse
/// with the right name and land on this queue. Written as one function with
/// `max.dim` rather than copied, so a fourth version of the NaN rule cannot
/// drift away from the third.
fn extremum_dim(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    which: Extremum,
) -> PyResult<Py<PyAny>> {
    let op = match which {
        Extremum::Max => "aten.max.dim",
        Extremum::Min => "aten.min.dim",
    };
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let dim = normalise_dim(
        op,
        dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(op, "dim"))?,
        rank,
    )?;
    let keepdim = bool_arg(args, kwargs, 2, "keepdim")?.unwrap_or(false);
    let tag = input.tag();
    let source = input.tensor()?;

    // Reduced with the dimension kept whatever the caller asked for, so the
    // NaN correction below has one shape to work in; the squeeze is at the end.
    let (values, indices) = match which {
        Extremum::Max => (source.max_keepdim(dim), source.argmax_keepdim(dim)),
        Extremum::Min => (source.min_keepdim(dim), source.argmin_keepdim(dim)),
    };
    let mut values = values.map_err(|e| candle_err(op, e))?;
    // int64, like `argmax` above: candle yields u32, which would be a visible
    // dtype divergence the first time an index is used.
    let mut indices = indices
        .and_then(|t| t.to_dtype(candle_core::DType::I64))
        .map_err(|e| candle_err(op, e))?;

    if let Some((any, first)) = nan_along_dim(op, source, dim, tag)? {
        let nans = nan_shaped_like(op, &values, tag)?;
        values = any
            .where_cond(&nans, &values)
            .map_err(|e| candle_err(op, e))?;
        indices = any
            .where_cond(&first, &indices)
            .map_err(|e| candle_err(op, e))?;
    }

    if !keepdim {
        values = values.squeeze(dim).map_err(|e| candle_err(op, e))?;
        indices = indices.squeeze(dim).map_err(|e| candle_err(op, e))?;
    }

    // Promoted here, not at the dispatcher's exit: the pair leaves inside a
    // namedtuple, which `promote` (rightly) does not look into.
    let pair = (
        crate::tensor::promote(py, finish(py, values, tag)?)?,
        crate::tensor::promote(py, finish(py, indices, TorchDType::Int64)?)?,
    );
    Ok(extremum_result_type(py, which)?
        .bind(py)
        .call1(pair)?
        .unbind())
}

/// The 0/1 byte mask both `any` and `all` reduce over.
///
/// "is this element non-zero", read through a mask so the result satisfies
/// `boolean()`'s invariant by construction. NaN is non-zero and therefore
/// *true* -- measured, `torch.tensor([nan, 1.]).all()` is `True` -- which is
/// what the `F64` round-trip gives for free (`nan != 0` is true) and what a
/// kernel written as "compare against zero in the input's own dtype" would
/// also give. Recorded because it reads like an accident and is upstream's
/// documented behaviour.
fn any_from(op: &str, source: &Tensor) -> PyResult<Tensor> {
    widen_f64(source)
        .and_then(|t| t.ne(0f64))
        .map_err(|e| candle_err(op, e))
}

/// Which of the two boolean reductions is being run. They differ in exactly
/// two things -- the candle reduction (`max` vs `min` of the 0/1 mask) and the
/// value an *empty* reduction produces -- and both are read off this.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BoolReduce {
    Any,
    All,
}

impl BoolReduce {
    /// The result of reducing **nothing**, which is the half of these two ops
    /// that is not symmetric with the other in a way a reader can eyeball.
    /// Measured on 2.13.0: `torch.tensor([]).any()` is `False` and
    /// `torch.tensor([]).all()` is `True`. A kernel that shares one early
    /// return between the two gets one of them wrong and only on empty input,
    /// which no forward reaches.
    fn identity(self) -> u8 {
        match self {
            BoolReduce::Any => 0,
            BoolReduce::All => 1,
        }
    }
}

/// **The result dtype is `torch.bool` for every input dtype except `uint8`,
/// where it is `uint8`.**
///
/// Not a guess and not symmetry: upstream's own docstring for `torch.all` says
/// so ("matches the behaviour of NumPy in returning output of dtype `bool` for
/// all supported dtypes except `uint8`"), and it is measured on both ops and
/// on all three forms --
///
///     uint8.any()  uint8      int8/int16/int32/int64/bool/float32 .any()  bool
///     uint8.all()  uint8      ...                                 .all()  bool
///     uint8.all(0) uint8
///
/// `any` had this wrong: it returned `torch.bool` unconditionally, and its
/// golden cases probe only `int64`, so the one dtype that separates the rule
/// from "always bool" was never fed to it. Fixed here rather than left,
/// because writing `all` from `any`'s shape would have copied the defect into
/// a second op.
fn bool_reduce_tag(input_tag: TorchDType) -> TorchDType {
    if input_tag == TorchDType::UInt8 {
        TorchDType::UInt8
    } else {
        TorchDType::Bool
    }
}

/// The shape a reduction leaves behind. Shared so that the empty-input path
/// below can name it without running a reduction candle refuses to run.
fn reduced_dims(dims_in: &[usize], reduce: &[usize], keepdim: bool) -> Vec<usize> {
    let mut out = Vec::new();
    for (index, &extent) in dims_in.iter().enumerate() {
        if reduce.contains(&index) {
            if keepdim {
                out.push(1);
            }
        } else {
            out.push(extent);
        }
    }
    out
}

fn any_or_all_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    which: BoolReduce,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let tag = bool_reduce_tag(input.tag());
    if input.tensor()?.elem_count() == 0 {
        let out = Tensor::full(
            which.identity(),
            (),
            input.tensor()?.device(),
        )
        .map_err(|e| candle_err(op, e))?;
        return finish(py, out, tag);
    }
    let mask = any_from(op, input.tensor()?)?
        .flatten_all()
        .map_err(|e| candle_err(op, e))?;
    let out = match which {
        BoolReduce::Any => mask.max(0),
        BoolReduce::All => mask.min(0),
    }
    .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

fn any_or_all_dim(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    list_form: bool,
    which: BoolReduce,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let dims = reduce_dims(op, args, kwargs, 1, rank)?;
    let keepdim = bool_arg(args, kwargs, 2, "keepdim")?.unwrap_or(false);
    let tag = bool_reduce_tag(input.tag());

    let dims = match dims {
        Some(dims) => dims,
        None if list_form => (0..rank).collect(),
        None => return Err(missing(op, "dim")),
    };

    // An empty input is the case candle refuses ("empty tensor for reduce"),
    // and it is not a corner that can be skipped: the answer is the *identity*
    // over the reduced axes, and it is measured. `torch.zeros(0, 3).all(0)` is
    // `[True, True, True]` -- three trues out of nothing -- while
    // `torch.zeros(0, 3).all(1)` is the empty `[]`, because the surviving axis
    // is itself zero-length. Filling `reduced_dims` with the identity gives
    // both, which is why the shape is computed rather than special-cased.
    if input.tensor()?.elem_count() == 0 {
        let shape = reduced_dims(input.dims(), &dims, keepdim);
        let out = Tensor::full(which.identity(), shape, input.tensor()?.device())
            .map_err(|e| candle_err(op, e))?;
        return finish(py, out, tag);
    }

    // "any" over a dimension is the max of the 0/1 mask over it; "all" is the
    // min. Reduced back-to-front so that each `dim` index still refers to the
    // axis it named on the way in.
    let mut out = any_from(op, input.tensor()?)?;
    for dim in dims.into_iter().rev() {
        out = match (which, keepdim) {
            (BoolReduce::Any, true) => out.max_keepdim(dim),
            (BoolReduce::Any, false) => out.max(dim),
            (BoolReduce::All, true) => out.min_keepdim(dim),
            (BoolReduce::All, false) => out.min(dim),
        }
        .map_err(|e| candle_err(op, e))?;
    }
    finish(py, out, tag)
}

/// `aten::masked_fill.Scalar/.Tensor(Tensor self, Tensor mask, X value)`
///
/// The mask has to be `torch.bool`. torch refuses a `uint8` mask (it was
/// deprecated and then removed), and BOOL.md §3 lists that refusal as one of
/// the six guardrails that survive only because the tag is not aliased onto
/// `uint8` -- so this shim can keep it, and does.
fn masked_fill(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let mask = tensor_arg(op, args, kwargs, 1, "mask")?;
    if mask.tag() != TorchDType::Bool {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "masked_fill only supports boolean masks, but got mask with dtype {}",
            mask.tag().name()
        )));
    }
    let value = scalar_arg(op, args, kwargs, 2, "value")?.ok_or_else(|| missing(op, "value"))?;

    let tag = input.tag();
    let storage = PyDtype::new(tag).storage(op)?;
    let shape = input.tensor()?.shape().clone();
    let device = input.tensor()?.device();

    let condition = mask
        .tensor()?
        .broadcast_as(shape.clone())
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(op, e))?;
    let filled = if storage.is_int() {
        Tensor::full(value.as_i64(), shape.clone(), device)
    } else {
        Tensor::full(value.as_f64(), shape.clone(), device)
    }
    .and_then(|t| t.fast_to(storage))
    .map_err(|e| candle_err(op, e))?;
    let source = input
        .tensor()?
        .contiguous()
        .map_err(|e| candle_err(op, e))?;

    let out = condition
        .where_cond(&filled, &source)
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor`
///
/// The three-tensor select. `falcon`, `gptj`, `bloom` and `mpt` all reach it
/// the same way -- a `bool` causal mask and two **0-D** `float32` branches
/// (`scalar_tensor(finfo.min)` and `scalar_tensor(0.0)`) broadcast up to the
/// mask's shape (measured, docs/OPS4.md §1). That is why all three operands
/// broadcast here rather than only two: the branches carry no shape at all.
///
/// **The condition's dtype rule is not `masked_fill`'s.** `masked_fill`
/// refuses a `uint8` mask because upstream refuses one (BOOL.md §3 lists that
/// as a guardrail worth keeping). `where` *accepts* `uint8`, with a deprecation
/// warning, and refuses every other non-`bool` dtype:
///
/// ```text
/// where(bool  cond, ...)  -> computes
/// where(uint8 cond, ...)  -> computes, "where received a uint8 condition
///                            tensor. This behavior is deprecated..."
/// where(int64 cond, ...)  -> RuntimeError: where expected condition to be a
///                            boolean tensor, but got a tensor with dtype Long
/// ```
///
/// All four measured. Carrying `masked_fill`'s refusal over would have refused
/// a call upstream answers, which is the noisier direction but still a
/// divergence; carrying `uint8` acceptance into `masked_fill` would have been
/// the silent one. They are different ops and this shim keeps them different.
///
/// **Upstream promotes the two branches; this shim does not.** The full 9x9
/// table was measured and it is torch's ordinary promotion lattice -- notably
/// `where(float16, bfloat16) -> float32`, and an integral branch never widens a
/// floating one (`where(float16, int64) -> float16`). The shim refuses through
/// `same_dtype` for the reason written there, and the golden cases record the
/// promoting combinations as `c_error` so the gap stays visible. **No measured
/// call site mixes**: all four architectures pass two `float32` branches.
///
/// The unselected branch is never read for its value, only for its shape and
/// dtype -- `where(True, 1.0, nan)` is `1.0`, measured -- and `where_cond`
/// selects rather than blends, so that holds here too.
/// What `where` accepts as a condition, on its own so that all three call
/// sites -- both dense overloads and the meta arms -- refuse the same tensors
/// with the same message.
///
/// `uint8` is accepted beside `bool` because upstream accepts it (as
/// truthiness, not as a bit pattern); everything else raises. This is one of
/// the few dense checks a meta tensor can still run in full, since the dtype
/// tag is carried and the condition's *values* are not consulted.
fn where_condition_check(condition: &PyTensorBase) -> PyResult<()> {
    if condition.tag() != TorchDType::Bool && condition.tag() != TorchDType::UInt8 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "where expected condition to be a boolean tensor, but got a tensor with dtype {}",
            scalar_type_name(condition.tag())
        )));
    }
    Ok(())
}

fn where_self(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.where.self";
    let condition = tensor_arg(OP, args, kwargs, 0, "condition")?;
    let lhs = tensor_arg(OP, args, kwargs, 1, "self")?;
    let rhs = tensor_arg(OP, args, kwargs, 2, "other")?;

    where_condition_check(&condition)?;
    // Promotes over the lattice (docs/PROMOTE.md §3), from the two *value*
    // operands only -- the condition's dtype takes no part, exactly as the
    // meta path below already documented for the same-dtype case.
    //
    // `where_select` casts both branches to the tag's storage itself, so the
    // promoted tag is all it needs; there is no separate `operand_in` call
    // here because that cast is the one `where_select` already performs.
    let tag = promote_operands(OP, &lhs, &rhs)?;
    let out = where_select(OP, &condition, lhs.tensor()?, rhs.tensor()?, tag)?;
    finish(py, out, tag)
}

/// The select `where.self` and `where.ScalarOther` share.
///
/// Both branches arrive already agreed on `tag`; this does the three-way
/// broadcast, the condition's truthiness normalisation, and the selection.
/// Split out when `where.ScalarOther` landed rather than duplicated: the two
/// overloads differ only in where the second branch comes from, and a second
/// copy of the broadcast rule is a second place for it to drift.
fn where_select(
    op: &str,
    condition: &PyTensorBase,
    lhs: &Tensor,
    rhs: &Tensor,
    tag: TorchDType,
) -> PyResult<Tensor> {
    let storage = PyDtype::new(tag).storage(op)?;

    // torch broadcasts all three together, and the result shape is the join of
    // the three -- not the condition's. `where(tensor(True), ones(2,3),
    // zeros(3))` is `(2, 3)`, measured, where a condition-shaped answer would
    // be `()`.
    let rhs_shape = rhs.shape().clone();
    let shape = condition
        .tensor()?
        .shape()
        .broadcast_shape_binary_op(lhs.shape(), "where")
        .and_then(|s| s.broadcast_shape_binary_op(&rhs_shape, "where"))
        .map_err(|e| candle_err(op, e))?;

    let spread = |t: &Tensor| -> PyResult<Tensor> {
        t.broadcast_as(shape.clone())
            .and_then(|t| t.contiguous())
            .map_err(|e| candle_err(op, e))
    };
    // A `uint8` condition is truthiness, not a bit pattern: `where_cond` reads
    // "not zero", which is the same rule, but the tag is normalised to 0/1
    // first so the two dtypes take identical paths.
    let mask = spread(condition.tensor()?)?;
    let mask = if condition.tag() == TorchDType::Bool {
        mask
    } else {
        mask.ne(0u8).map_err(|e| candle_err(op, e))?
    };
    let on_true = spread(&lhs.fast_to(storage).map_err(|e| candle_err(op, e))?)?;
    let on_false = spread(&rhs.fast_to(storage).map_err(|e| candle_err(op, e))?)?;

    mask.where_cond(&on_true, &on_false)
        .map_err(|e| candle_err(op, e))
}

/// The dtype `where.ScalarOther` promotes to, given the tensor branch's dtype
/// and which Python type the scalar arrived as.
///
/// torch's "wrapped number" rule, and every cell measured on 2.13.0 rather
/// than derived from the general promotion lattice -- they are *not* the same
/// function. `promote_types(int64, float32)` is `float32` and so is
/// `where(cond, int64_t, 2.5)`, but that agreement is a coincidence of the
/// default dtype: `promote_types(float16, float32)` is `float32` while
/// `where(cond, float16_t, 2.5)` is `float16`, because a Python float is not
/// a `float32` tensor. A scalar names a *category*, not a width.
///
/// ```text
/// scalar         bool tensor   integral tensor   floating tensor
/// True/False     bool          the tensor's      the tensor's
/// 3              int64         the tensor's      the tensor's
/// 2.5            float32       float32           the tensor's
/// ```
///
/// The `bool` column is the one that needs the Python *type* and not the
/// value: `where(cond, bool_t, True)` is `bool` and `where(cond, bool_t, 1)`
/// is `int64`, measured. `Scalar` in this file folds `bool` into `Int` (it
/// says so, and every other op wants that), so this rule reads the raw
/// object instead of going through `Scalar`.
fn where_scalar_tag(tensor: TorchDType, scalar_is_bool: bool, scalar_is_int: bool) -> TorchDType {
    if scalar_is_bool {
        return tensor;
    }
    if scalar_is_int {
        return if tensor == TorchDType::Bool {
            TorchDType::Int64
        } else {
            tensor
        };
    }
    if tensor.is_floating_point() {
        tensor
    } else {
        default_float()
    }
}

/// `aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor`
///
/// `torch.where(mask, tensor, python_scalar)`. transformers' `eager_mask`
/// reaches it verbatim at `masking_utils.py:603`:
///
/// ```text
/// mask = torch.where(mask, torch.tensor(0.0, device=..., dtype=dtype), min_dtype)
/// ```
///
/// with `min_dtype = torch.finfo(dtype).min`. That single call was the whole
/// of what stood between this shim and a real pretrained model's *eager*
/// forward (docs/CKPT2.md §7.1), and the schema for it was already in
/// `overloads.json` -- only the kernel was missing, so the dispatcher was
/// resolving the call and then refusing it by name.
///
/// **What the overload does was measured, not read off the schema.** A
/// `TorchDispatchMode` over the call above reports two ops:
///
/// ```text
/// aten.scalar_tensor.default(-3.5, dtype=<the PROMOTED dtype>)
/// aten.where.self(cond, self, that)
/// ```
///
/// so upstream itself turns the scalar into a 0-D tensor *at the promoted
/// dtype* and then runs ordinary `where.self`. This does the same two steps,
/// through the same `checked_convert` `scalar_tensor` uses, rather than
/// growing a third select path -- which also means the overflow rule comes
/// out identical for free: `-1` into `uint8` wraps to 255 and is answered,
/// `300` does not fit and is refused, both measured on both sides.
///
/// **A 0-D tensor is not accepted here, and that is upstream's rule rather
/// than a shortcut.** `scalar_arg` takes a 0-D tensor anywhere a `Scalar` is
/// wanted, because torch does; this overload is the exception, measured:
///
/// ```text
/// aten::where() Expected a value of type 'number' for argument 'other'
///   but instead found type Tensor
/// ```
///
/// Answering it would compute where torch raises.
///
/// `where.ScalarSelf` and `where.Scalar` stay unimplemented. They are in the
/// same `overloads.json` entry and would each be a few lines, but no measured
/// caller reaches them -- the rule docs/E2E_REAL.md §1.2 sets, and the reason
/// this kernel exists at all is that a caller *was* measured reaching it.
fn where_scalar_other(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.where.ScalarOther";
    let condition = tensor_arg(OP, args, kwargs, 0, "condition")?;
    let lhs = tensor_arg(OP, args, kwargs, 1, "self")?;
    let raw = required(OP, args, kwargs, 2, "other")?;

    where_condition_check(&condition)?;
    if raw.is_instance_of::<PyTensorBase>() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "aten::where() Expected a value of type 'number' for argument 'other' \
             but instead found type Tensor",
        ));
    }

    // Order matters, as it does in `scalar_arg`: Python's `bool` is a subclass
    // of `int`, and here the two give different answers.
    let scalar_is_bool = raw.is_instance_of::<pyo3::types::PyBool>();
    let scalar_is_int = scalar_is_bool || raw.is_instance_of::<pyo3::types::PyInt>();
    let value = scalar_arg(OP, args, kwargs, 2, "other")?.ok_or_else(|| missing(OP, "other"))?;

    let tag = where_scalar_tag(lhs.tag(), scalar_is_bool, scalar_is_int);
    // The same check `scalar_tensor` runs, at the same numel: upstream builds
    // a one-element tensor here, so the numel==1 hole that check reproduces
    // applies to this call too.
    checked_convert(&raw, scalar_is_int, tag, 1)?;

    let device = lhs.tensor()?.device().clone();
    let other = if tag == TorchDType::Bool {
        Tensor::full(u8::from(value.as_f64() != 0.0), (), &device).map_err(|e| candle_err(OP, e))?
    } else {
        let storage = PyDtype::new(tag).storage(OP)?;
        if storage.is_int() {
            Tensor::full(value.as_i64(), (), &device)
        } else {
            Tensor::full(value.as_f64(), (), &device)
        }
        .and_then(|t| t.fast_to(storage))
        .map_err(|e| candle_err(OP, e))?
    };

    let out = where_select(OP, &condition, lhs.tensor()?, &other, tag)?;
    finish(py, out, tag)
}

/// A shape argument with torch's placeholders resolved: `-1` in `reshape`
/// means "whatever is left", `-1` in `expand` means "keep this dimension".
fn resolve_shape(op: &str, requested: &[isize], numel: usize) -> PyResult<Vec<usize>> {
    let mut known: usize = 1;
    let mut wildcard: Option<usize> = None;
    for (i, &value) in requested.iter().enumerate() {
        if value == -1 {
            if wildcard.is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "only one dimension can be inferred",
                ));
            }
            wildcard = Some(i);
        } else if value < 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: invalid shape dimension {value}"
            )));
        } else {
            known *= value as usize;
        }
    }
    let mut out: Vec<usize> = requested
        .iter()
        .map(|&v| if v == -1 { 0 } else { v as usize })
        .collect();
    if let Some(index) = wildcard {
        if known == 0 || numel % known != 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "shape '{requested:?}' is invalid for input of size {numel}"
            )));
        }
        out[index] = numel / known;
    }
    Ok(out)
}

fn shape_arg(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Vec<isize>> {
    let value = required(op, args, kwargs, index, name)?;
    match value.extract::<Vec<isize>>() {
        Ok(list) => Ok(list),
        Err(_) => Ok(vec![value.extract::<isize>()?]),
    }
}

/// `aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False)`
///
/// torch allows the requested size to have more dimensions than the tensor, in
/// which case the new ones are prepended, and `-1` means "keep whatever is
/// there". candle's `broadcast_as` has the same alignment-from-the-right rule
/// once the `-1`s are resolved.
/// Resolving `expand`'s requested size against the tensor's own: the rank
/// check, the `-1` sentinel, and the negative-size refusal.
///
/// Metadata only, which is why it is a function -- the meta kernel needs
/// exactly this and can share it verbatim. What it deliberately does *not* do
/// is check that each extent is expandable; the dense path gets that from
/// `broadcast_as`, and the meta path, which has no candle handle to hand to
/// `broadcast_as`, does it itself with upstream's wording. That split is
/// recorded in docs/META.md §7.2 rather than hidden.
fn expand_target(op: &str, dims: &[usize], requested: &[isize]) -> PyResult<Vec<usize>> {
    if requested.len() < dims.len() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "expand(torch._C.TensorBase{dims:?}, size={requested:?}): the number of \
             sizes provided ({}) must be greater or equal to the number of \
             dimensions in the tensor ({})",
            requested.len(),
            dims.len()
        )));
    }
    let offset = requested.len() - dims.len();
    let mut target = Vec::with_capacity(requested.len());
    for (i, &value) in requested.iter().enumerate() {
        if value == -1 {
            if i < offset {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "expand: the expanded size of the tensor (-1) isn't allowed in a \
                     leading, non-existing dimension",
                ));
            }
            target.push(dims[i - offset]);
        } else if value < 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: invalid expand size {value}"
            )));
        } else {
            target.push(value as usize);
        }
    }
    Ok(target)
}

/// `remainder`'s correction, on one float pair.
///
/// **`remainder` follows the sign of the divisor; `fmod` follows the sign of
/// the dividend.** Rust's `%` on floats is `fmod`, so this is `fmod` plus
/// upstream's own correction, transcribed rather than reinvented:
///
/// ```text
/// mod = fmod(a, b);
/// if (mod != 0) && ((b < 0) != (mod < 0)) { mod += b }
/// ```
///
/// Three things fall out of that guard rather than being special-cased, and
/// each was checked against upstream instead of reasoned about:
///
///   * **`remainder(-0.0, 3.0)` is `-0.0`.** `fmod(-0.0, 3.0)` is `-0.0`, and
///     `-0.0 != 0.0` is *false*, so the correction does not fire and the
///     negative zero survives. Python's own `-0.0 % 3.0` is `+0.0`, so
///     "spell it as Python's `%`" is wrong here -- and wrong invisibly, since
///     `-0.0 == 0.0`.
///   * **NaN propagates.** `NaN != 0.0` is true, but `NaN < 0.0` is false, so
///     for a positive divisor the guard's two sides agree and NaN is returned
///     unchanged; for a negative one it fires and `NaN + b` is still NaN.
///   * **infinite divisors.** `remainder(5.0, -inf)` is `-inf` and
///     `remainder(-5.0, inf)` is `inf`, both measured, and both are just the
///     correction firing on a finite `fmod`.
fn remainder_f64(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        m + b
    } else {
        m
    }
}

/// The same correction on integers, with upstream's two divergences from C.
///
/// `wrapping_rem` rather than `%`: `i64::MIN % -1` **panics** in Rust (the
/// quotient overflows) where upstream answers `0`, measured. A panic here
/// crosses the FFI boundary as a `PanicException`, which is not a refusal.
///
/// A zero divisor raises rather than producing NaN -- that is the integral
/// path's split from the float one, and upstream's message is the bare string
/// `ZeroDivisionError` inside a `RuntimeError`.
fn remainder_i64(a: i64, b: i64) -> PyResult<i64> {
    if b == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("ZeroDivisionError"));
    }
    let r = a.wrapping_rem(b);
    Ok(if r != 0 && ((r < 0) != (b < 0)) {
        r.wrapping_add(b)
    } else {
        r
    })
}

/// `aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor` and
/// `aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor`.
///
/// `sam3_video`'s wall (docs/ARCH26.md §5): `Sam3ViTRotaryEmbedding.__init__`
/// computes `x_positions = (flattened_indices % end_x) * scale`, so
/// `TensorBase.__mod__` and therefore `remainder.Scalar` -- during
/// *construction*, which is why ARCH26.md's forward-only operator trace never
/// saw it.
///
/// **Dtype.** The `Tensor` overload is `torch.promote_types` exactly; that was
/// checked cell by cell over the eight storable numeric dtypes rather than
/// assumed from `mul`, and there were no disagreements. The `Scalar` overload
/// is the wrapped-number rule: an int scalar never widens a tensor of any
/// category, a float scalar floats an integral one.
///
/// **Bool is refused on both.** For the `Tensor` overload that is *upstream's*
/// refusal, reproduced with its own wording (`"remainder_cpu" not implemented
/// for 'Bool'`). For the `Scalar` overload it is this shim's: upstream
/// computes there (`remainder(bool_t, 2)` is `int64`, `remainder(bool_t, 2.0)`
/// is `float32`, `remainder(bool_t, True)` raises) and this refuses, exactly
/// as `arith_tag` already refuses `bool_tensor * 2` and for the same reason --
/// the rule is a fast-path ladder keyed on the *Python type* of the scalar,
/// and `scalar_arg` has already erased `True` into `Scalar::Int(1)` by the
/// time this is reached. Two golden cases carry it as `expect="c_error"` so
/// the gap is watched rather than filed away.
///
/// **The scalar is narrowed into the result dtype before the arithmetic**, not
/// used as an `i64`, because that is what upstream does and it is observable:
/// `remainder(uint8(200), -3)` is `200`, because `-3` becomes `253` in
/// `uint8` and `200 % 253` is `200`. Building it as a 0-d tensor at `storage`
/// reproduces the narrowing rather than restating it.
///
/// Computed elementwise in `f64`/`i64` rather than through candle, which has
/// no `remainder`. `read_flat`/`write_flat` are the same route `expm1` and
/// `pow` take, for the same reason: no kernel exists and the callers are not
/// hot loops.
fn remainder_op(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    scalar_form: bool,
) -> PyResult<Py<PyAny>> {
    let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
    let left = lhs.tensor()?;

    let (tag, right_dims) = if scalar_form {
        let other =
            scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
        if lhs.tag() == TorchDType::Bool {
            return Err(not_implemented(format!(
                "{op}: a torch.bool tensor with a numeric scalar promotes through a \
                 fast-path ladder keyed on the scalar's Python type (int64 for an int, \
                 float32 for a float, and upstream raises for a bool), which is not \
                 implemented in torch._C shim"
            )));
        }
        let mut tag = lhs.tag();
        if !other.is_int() && !tag.is_floating_point() {
            tag = default_float();
        }
        (tag, left.dims().to_vec())
    } else {
        let rhs = tensor_arg(op, args, kwargs, 1, "other")?;
        if lhs.tag() == TorchDType::Bool || rhs.tag() == TorchDType::Bool {
            // Upstream's own refusal, verbatim, with no shim prefix -- the
            // convention `overflow()` follows, because a caller matching on
            // the message should not have to know which side produced it.
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "\"remainder_cpu\" not implemented for 'Bool'",
            ));
        }
        let tag = promote_operands(op, &lhs, &rhs)?;
        (tag, rhs.tensor()?.dims().to_vec())
    };

    let storage = PyDtype::new(tag).storage(op)?;
    let shape = if scalar_form {
        left.dims().to_vec()
    } else {
        broadcast_shape(op, left.dims(), &right_dims)?
    };

    let a = left
        .fast_to(storage)
        .and_then(|t| t.broadcast_as(shape.clone()))
        .map_err(|e| candle_err(op, e))?;
    let b = if scalar_form {
        let scalar =
            scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
        // Narrowed to `storage` first: see the doc comment's `uint8` case.
        let filled = if storage.is_int() {
            Tensor::full(scalar.as_i64(), (), left.device())
        } else {
            Tensor::full(scalar.as_f64(), (), left.device())
        };
        filled
            .and_then(|t| t.fast_to(storage))
            .and_then(|t| t.broadcast_as(shape.clone()))
            .map_err(|e| candle_err(op, e))?
    } else {
        tensor_arg(op, args, kwargs, 1, "other")?
            .tensor()?
            .fast_to(storage)
            .and_then(|t| t.broadcast_as(shape.clone()))
            .map_err(|e| candle_err(op, e))?
    };

    let values = match (read_flat(op, &a, tag)?, read_flat(op, &b, tag)?) {
        (Flat::Float(x), Flat::Float(y)) => {
            Flat::Float(x.into_iter().zip(y).map(|(p, q)| remainder_f64(p, q)).collect())
        }
        (Flat::Int(x), Flat::Int(y)) => {
            let mut out = Vec::with_capacity(x.len());
            for (p, q) in x.into_iter().zip(y) {
                out.push(remainder_i64(p, q)?);
            }
            Flat::Int(out)
        }
        // Unreachable: both sides are read at the same `tag`, so they are the
        // same `Flat` arm by construction. An `unreachable!()` here would be a
        // panic across the FFI boundary.
        _ => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: operands read at different categories -- internal error"
            )))
        }
    };
    let out = write_flat(op, values, shape, left.device(), tag)?;
    finish(py, out, tag)
}

/// Which of the three functions `rounding_mode` selects.
///
/// **These are three different functions, not one function with a flag.** They
/// differ in dtype (`True` promotes an integral pair to the default float;
/// the other two preserve it), in whether division by zero raises, and -- for
/// `Trunc` vs `Floor` -- in the answer itself. See `div_mode`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RoundMode {
    /// `rounding_mode=None`: true division. Exactly `aten.div.Tensor`.
    True,
    /// `rounding_mode="trunc"`: round the quotient toward zero.
    Trunc,
    /// `rounding_mode="floor"`: round the quotient toward negative infinity.
    Floor,
}

/// Reads the `rounding_mode` slot of `div.{Tensor,Scalar}_mode`.
///
/// The schema is `*, str? rounding_mode` -- keyword-only and, in
/// `native_functions.yaml`, without a default. That absence is what keeps the
/// four `div` overloads apart in `overloads.json`: `torch.div(a, b)` cannot
/// bind `div.Tensor_mode` because the required keyword is missing, so it falls
/// through to `div.Tensor`. Reading index 2 as well as the name costs nothing
/// and matches every other kernel in this file, which is why it is done here
/// even though upstream's own binder refuses `torch.div(a, b, "floor")`
/// positionally (measured: `TypeError: div() received an invalid combination
/// of arguments`).
///
/// An unrecognised string is upstream's `RuntimeError`, reproduced verbatim
/// with no shim prefix -- the convention `remainder`'s `'Bool'` refusal
/// follows, because a caller matching on the message should not have to know
/// which side produced it. Measured on 2.13.0 for `'ceil'`, `''`, `'FLOOR'`
/// and `'Floor'`: the match is exact and case-sensitive.
fn rounding_mode_arg(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<RoundMode> {
    let value = match optional(args, kwargs, 2, "rounding_mode")? {
        Some(value) if !value.is_none() => value,
        _ => return Ok(RoundMode::True),
    };
    let name = value.extract::<String>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{op}: argument 'rounding_mode' must be None, 'trunc' or 'floor', got {}",
            value.get_type().name().map(|n| n.to_string()).unwrap_or_default()
        ))
    })?;
    match name.as_str() {
        "trunc" => Ok(RoundMode::Trunc),
        "floor" => Ok(RoundMode::Floor),
        other => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "div expected rounding_mode to be one of None, 'trunc', or 'floor' \
             but found '{other}'"
        ))),
    }
}

/// Rounds an `f64` to the precision of `tag`, so that the arithmetic below
/// runs at the dtype the *result* is stored in.
///
/// **This is not a detail.** `read_flat` hands every floating dtype over as
/// `f64`, and computing there and narrowing once at the end is the obvious
/// thing to do -- it is also measurably wrong. Upstream computes `div_floor`
/// and `div_trunc` in the tensor's own `scalar_t`, and past each dtype's
/// integer-exactness limit the two disagree: over 42436 `float32` pairs,
/// computing in `f64` and narrowing missed **68** of `floor` and **358** of
/// `trunc` (`16777216.0 / 1.3669793605804443` is `12273204.0` upstream and
/// `12273203.0` that way). `float16` and `bfloat16` were checked the same way
/// and want their own precision too, not `f32`'s: computing in `f32` missed 4
/// and 3 of 154 for `f16`. With per-dtype rounding threaded through every
/// intermediate, all four dtypes matched upstream on every pair.
///
/// So the returned closure is applied after each arithmetic step, not once at
/// the end.
fn float_narrower(tag: TorchDType) -> fn(f64) -> f64 {
    match tag {
        TorchDType::Float64 => |x| x,
        TorchDType::Float32 => |x| x as f32 as f64,
        TorchDType::Float16 => |x| half::f16::from_f64(x).to_f64(),
        TorchDType::BFloat16 => |x| half::bf16::from_f64(x).to_f64(),
        // Every other floating dtype is one `PyDtype::storage()` refuses, so
        // the operands could not have been built. Identity keeps this total.
        _ => |x| x,
    }
}

/// `rounding_mode="floor"` on a floating pair, transcribed from upstream's
/// `div_floor_floating` rather than derived from the name.
///
/// **`floor(a / b)` is the plausible implementation and it is wrong.**
/// Measured on 2.13.0: `inf / 3.0` is **`nan`**, not `inf`; `5.0 / -inf` is
/// **`-1.0`**, not `-0.0`; `-0.5 / 3.0` is `-1.0`; and `-0.0 / 3.0` keeps its
/// sign bit. All four fall out of the algorithm below and none is special-cased
/// here.
///
/// The shape of it, and why each line is there:
///
/// * **`b == 0` returns the IEEE quotient unrounded.** Upstream has this as an
///   explicit early return, and it is the reason `5.0 / 0.0` is `inf` while
///   `inf / 3.0` -- which does *not* take this path -- is `nan`.
/// * `mod = fmod(a, b)`, then `div = (a - mod) / b`: the exact quotient when
///   the division is exact, which is what makes the correction below safe.
///   `fmod(inf, 3.0)` is NaN, and that NaN is what propagates to give upstream's
///   `nan`. Rust's `%` on `f64` is C's `fmod`; Python's `math.fmod` is *not*
///   (it raises on `inf`), which is worth recording because the model used to
///   verify this had to be corrected for exactly that.
/// * the `(b < 0) != (mod < 0)` correction is `remainder`'s, one level up: it
///   turns a truncated quotient into a floored one.
/// * the `div - floordiv > 0.5` nudge recovers the case where `(a - mod) / b`
///   lands just under an integer through rounding.
/// * `copysign(0, a / b)` for a zero quotient is what preserves `-0.0`.
///
/// Verified as an algorithm, not as a table of corners: over 10609 `f64` pairs
/// built from infinities, NaNs, signed zeros, subnormals and randoms, this
/// reproduces upstream **bit for bit, 10609/10609**, compared on the bytes so
/// that `-0.0` and NaN are not read as equal to their opposites.
fn div_floor_float(a: f64, b: f64, narrow: fn(f64) -> f64) -> f64 {
    if b == 0.0 {
        return narrow(a / b);
    }
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    let m = narrow(a % b);
    let mut div = narrow(narrow(a - m) / b);
    if m != 0.0 && (b < 0.0) != (m < 0.0) {
        div = narrow(div - 1.0);
    }
    if div.is_nan() || div.is_infinite() {
        return div;
    }
    if div != 0.0 {
        let mut floordiv = div.floor();
        if div - floordiv > 0.5 {
            floordiv += 1.0;
        }
        narrow(floordiv)
    } else {
        // `-0.0` survives here and only here. `copysign` rather than a sign
        // test, because the sign being carried is the one `==` cannot see.
        narrow(0.0f64.copysign(a / b))
    }
}

/// `rounding_mode="trunc"` on a floating pair: round the quotient toward zero.
///
/// Simpler than `floor` -- there is no correction and no early return, because
/// truncation of `inf`, `-inf` and `nan` is already those values. Verified over
/// the same 10609 pairs, **10609/10609 bit-identical**, including `-0.5 / 3.0`
/// answering `-0.0` (sign bit set) where a naive `as i64 as f64` would give
/// `+0.0`.
fn div_trunc_float(a: f64, b: f64, narrow: fn(f64) -> f64) -> f64 {
    let q = narrow(a / b);
    if q.is_nan() || q.is_infinite() {
        return q;
    }
    narrow(q.trunc())
}

/// `aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode)`
/// and `aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode)`.
///
/// `sam3_video`'s `Sam3ViTRotaryEmbedding.__init__` builds its rotary position
/// grid two lines apart: `flattened_indices % end_x` for the x axis, which is
/// the `remainder` kernel above, and
/// `torch.div(flattened_indices, end_x, rounding_mode="floor")` for the y axis,
/// which is this one (`modeling_sam3.py:428`). It is the wall that architecture
/// landed on the moment `remainder` existed.
///
/// **The three modes are three functions.** Measured on 2.13.0:
///
/// ```text
///          a    b   None    trunc  floor
///          7    3    2.33     2      2
///          7   -3   -2.33    -2     -3     <- trunc and floor DISAGREE
///         -7    3   -2.33    -2     -3     <- trunc and floor DISAGREE
///         -7   -3    2.33     2      2
///         -6    3   -2.0     -2     -2     <- opposite signs, but EXACT: agree
///   dtype(int64)      float32  int64  int64
/// ```
///
/// `trunc` and `floor` differ **exactly when the operands' signs differ and the
/// division is inexact** -- checked rather than asserted, over 210 integer
/// pairs: they disagree on 64 of them, that set has no same-sign member and no
/// exact-division member, and it is precisely the 64 opposite-sign inexact
/// pairs. So a case set built from positive operands passes either
/// implementation, and one built from opposite signs but exact division
/// (`-6 / 3`) passes both too. Both the golden builder and the pytest carry all
/// three kinds.
///
/// **`rounding_mode=None` is delegated, not restated.** It is true division and
/// therefore literally `aten.div.Tensor`/`aten.div.Scalar` -- same promotion to
/// the default float, same answer, same division-by-zero behaviour (`inf`, no
/// raise). Calling those keeps one implementation rather than two that have to
/// be kept in agreement.
///
/// **Dtype for the two rounding modes preserves rather than promotes**, which is
/// the visible difference from `None`: `int64 / int64` stays `int64` under
/// `trunc`/`floor` and becomes `float32` under `None` (measured). The tensor
/// form follows `torch.promote_types` exactly -- checked cell by cell over the
/// seven storable numeric dtypes, 49 pairs, no disagreements. The scalar form
/// follows the wrapped-number rule `remainder.Scalar` already implements: an int
/// scalar never widens the tensor, a float scalar floats an integral one.
///
/// **The scalar is narrowed into the result dtype before the division**, and it
/// is observable in exactly the way `remainder`'s is:
/// `div(uint8(200), -3, rounding_mode="floor")` is **`0`**, because `-3` becomes
/// `253` in `uint8`. Building the scalar as a 0-D tensor at the result storage
/// reproduces that instead of restating it.
///
/// **Division by zero splits by category, and by mode.** An integral divisor of
/// `0` raises `RuntimeError('ZeroDivisionError')` -- upstream's message is that
/// bare string -- but *only* under `trunc`/`floor`; under `None` the same call
/// promotes to float and answers `inf`/`nan` with no raise. A floating divisor
/// never raises under any mode. All six combinations measured.
///
/// **`i64::MIN / -1` answers `i64::MIN`** under both rounding modes (measured),
/// where the quotient overflows. `wrapping_div`/`wrapping_rem` are used for
/// exactly that pair, as `remainder` uses `wrapping_rem` -- a Rust overflow
/// panic here would cross the FFI boundary as a `PanicException`, which is not
/// a refusal.
///
/// **Bool.** `div.Tensor_mode` on a bool pair raises upstream's own
/// `"div_trunc_cpu" not implemented for 'Bool'` (or `div_floor_cpu`),
/// reproduced verbatim. `div.Scalar_mode` on a bool tensor is refused here and
/// upstream computes -- the same deliberate gap `remainder.Scalar` has, for the
/// same reason: the rule is a fast-path ladder keyed on the scalar's *Python*
/// type (`div(bool_t, 2, "floor")` is `int64`, `div(bool_t, 2.0, "floor")` is
/// `float32`, `div(bool_t, True, "floor")` raises), and `scalar_arg` has erased
/// `True` into `Scalar::Int(1)` before this kernel runs. Golden carries it as
/// `c_error` so the gap is watched rather than filed away.
fn div_mode(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    scalar_form: bool,
) -> PyResult<Py<PyAny>> {
    let mode = rounding_mode_arg(op, args, kwargs)?;
    if mode == RoundMode::True {
        // True division, which is `div.Tensor`/`div.Scalar` exactly.
        return if scalar_form {
            arith_scalar(py, args, kwargs, op, Arith::Div)
        } else {
            arith_tensor(py, args, kwargs, op, Arith::Div)
        };
    }
    // Upstream names the kernel, not the op, in its `'Bool'` refusal.
    let kernel = if mode == RoundMode::Trunc {
        "div_trunc_cpu"
    } else {
        "div_floor_cpu"
    };

    let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
    let left = lhs.tensor()?;

    let (tag, right_dims) = if scalar_form {
        let other =
            scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
        if lhs.tag() == TorchDType::Bool {
            return Err(not_implemented(format!(
                "{op}: a torch.bool tensor with a numeric scalar promotes through a \
                 fast-path ladder keyed on the scalar's Python type (int64 for an int, \
                 float32 for a float, and upstream raises for a bool), which is not \
                 implemented in torch._C shim"
            )));
        }
        let mut tag = lhs.tag();
        if !other.is_int() && !tag.is_floating_point() {
            tag = default_float();
        }
        (tag, left.dims().to_vec())
    } else {
        let rhs = tensor_arg(op, args, kwargs, 1, "other")?;
        if lhs.tag() == TorchDType::Bool || rhs.tag() == TorchDType::Bool {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "\"{kernel}\" not implemented for 'Bool'"
            )));
        }
        let tag = promote_operands(op, &lhs, &rhs)?;
        (tag, rhs.tensor()?.dims().to_vec())
    };

    let storage = PyDtype::new(tag).storage(op)?;
    let shape = if scalar_form {
        left.dims().to_vec()
    } else {
        broadcast_shape(op, left.dims(), &right_dims)?
    };

    let a = left
        .fast_to(storage)
        .and_then(|t| t.broadcast_as(shape.clone()))
        .map_err(|e| candle_err(op, e))?;
    // **A reduced-float scalar divisor takes upstream's `opmath_t` road**,
    // exactly as `div.Scalar`'s reciprocal fast path does: `div_floor_kernel`
    // and `div_trunc_kernel` carry the same
    // `iter.is_scalar(2) && isReducedFloatingType(dtype)` branch that
    // `div_true_kernel` does, read the divisor with
    // `original_scalar_value<opmath_t>(2)`, and run the whole of
    // `div_floor_floating` in `float` -- narrowing once on store rather than at
    // every step. docs/SCALAR.md §3.2.
    //
    // Here the difference is not one representable step. A floor turns a
    // fractional error into an integer one: `bfloat16(3) // 0.3` is **10**
    // upstream and was **9** here, because `bf16(0.3)` is `0.30078125` and
    // `3 / 0.30078125` is `9.97`.
    let scalar_at_opmath = scalar_form
        && matches!(storage, candle_core::DType::F16 | candle_core::DType::BF16);
    let b = if scalar_form {
        let scalar =
            scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
        // Narrowed to `storage` first: see the doc comment's `uint8` case. The
        // reduced-float branch above is the exception, and `float` is as narrow
        // as the divisor ever gets there.
        let target = if scalar_at_opmath { candle_core::DType::F32 } else { storage };
        let filled = if storage.is_int() {
            Tensor::full(scalar.as_i64(), (), left.device())
        } else {
            Tensor::full(scalar.as_f64(), (), left.device())
        };
        filled
            .and_then(|t| t.fast_to(target))
            .and_then(|t| t.broadcast_as(shape.clone()))
            .map_err(|e| candle_err(op, e))?
    } else {
        tensor_arg(op, args, kwargs, 1, "other")?
            .tensor()?
            .fast_to(storage)
            .and_then(|t| t.broadcast_as(shape.clone()))
            .map_err(|e| candle_err(op, e))?
    };

    // `read_flat`'s category (float vs int) is the tag's; the *value* is read
    // at `f64` either way, so `b` built at `F32` above arrives un-narrowed.
    let values = match (read_flat(op, &a, tag)?, read_flat(op, &b, tag)?) {
        (Flat::Float(x), Flat::Float(y)) => {
            // The intermediates follow the divisor: `opmath` for the
            // reduced-float scalar branch, the tensor's own dtype otherwise
            // (docs/KERNELS26.md §9.3 measured that per-step narrowing).
            let narrow = float_narrower(if scalar_at_opmath {
                TorchDType::Float32
            } else {
                tag
            });
            let f = if mode == RoundMode::Trunc {
                div_trunc_float
            } else {
                div_floor_float
            };
            Flat::Float(x.into_iter().zip(y).map(|(p, q)| f(p, q, narrow)).collect())
        }
        (Flat::Int(x), Flat::Int(y)) => {
            let mut out = Vec::with_capacity(x.len());
            for (p, q) in x.into_iter().zip(y) {
                if q == 0 {
                    // Upstream's message is this bare string, on an integral
                    // dtype only -- floats fall in the arm above and answer
                    // inf/nan without raising.
                    return Err(pyo3::exceptions::PyRuntimeError::new_err("ZeroDivisionError"));
                }
                // Wrapping for the `i64::MIN / -1` pair, whose quotient
                // overflows; upstream answers `i64::MIN` there.
                let quotient = p.wrapping_div(q);
                out.push(if mode == RoundMode::Trunc {
                    quotient
                } else {
                    let rem = p.wrapping_rem(q);
                    if rem != 0 && (rem < 0) != (q < 0) {
                        quotient.wrapping_sub(1)
                    } else {
                        quotient
                    }
                });
            }
            Flat::Int(out)
        }
        // Unreachable: both sides are read at the same `tag`, so they are the
        // same `Flat` arm by construction. An `unreachable!()` here would be a
        // panic across the FFI boundary.
        _ => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: operands read at different categories -- internal error"
            )))
        }
    };
    let out = write_flat(op, values, shape, left.device(), tag)?;
    finish(py, out, tag)
}

/// `aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim,
///     bool keepdim=False) -> Tensor`
///
/// The kernel docs/KERNELS26.md §5.4 found behind `weight_norm`, and §8.3's
/// correction to ARCH26.md: `weight_norm` costs **three** pieces, not two, and
/// this is the one that was invisible to a traced sweep because
/// `torch.norm_except_dim` is a composite and it is called at *construction*
/// rather than in a forward.
///
/// **`p` is a general real exponent, not a flag.** Measured on 2.13.0 across
/// the whole family rather than only the `p=2` that `norm_except_dim` passes:
///
/// ```text
/// p = None    same as p = 2 (the Frobenius default)
/// p = 0       the COUNT of non-zero elements, not a sum of anything
/// p = +inf    max |x|
/// p = -inf    min |x|
/// p = 1       sum |x|
/// otherwise   (sum |x|^p)^(1/p), including fractional and NEGATIVE p
/// ```
///
/// The negative-`p` rows are the ones that catch a special-cased
/// implementation: `norm([[0, 0], [1, 2]], p=-1, dim=1)` is `[0.0, 0.6666...]`
/// — the zero row gives `|0|^-1 = inf`, a sum of `inf`, and `inf^(-1) = 0`. It
/// falls straight out of the general formula and has to be special-cased *not*
/// to happen.
///
/// **An empty `dim` list reduces every axis** (measured: `norm(x, 2, [])` on a
/// 2x2 is a scalar), which is the opposite of the usual "no dims means no
/// reduction" reading. A repeated dim raises upstream's
/// `dim 0 appears multiple times in the list of dims`, reproduced.
///
/// **Integral and boolean input raise**, with upstream's own wording:
/// `norm(): input dtype should be either floating point or complex. Got Long
/// instead.` — the `scalar_type_name` spelling (`Long`, `Bool`), which is the
/// third of the four namings docs/KERNELS26.md §5.2 tabulates.
///
/// Dtype is preserved, including `float16` and `bfloat16` (measured: a `f16`
/// input gives a `f16` norm), so this does not promote the way a reduction
/// that accumulated in `f32` would.
fn norm_scalaropt_dim(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.norm.ScalarOpt_dim";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "norm(): input dtype should be either floating point or complex. Got {} instead.",
            scalar_type_name(tag)
        )));
    }
    // `Scalar? p` -- absent and `None` are both the Frobenius default.
    let p = scalar_arg(OP, args, kwargs, 1, "p")?
        .map(|s| s.as_f64())
        .unwrap_or(2.0);
    let dims_raw = shape_arg(OP, args, kwargs, 2, "dim")?;
    let keepdim = bool_arg(args, kwargs, 3, "keepdim")?.unwrap_or(false);

    let t = input.tensor()?;
    let rank = t.rank();
    // An empty list means "every axis", measured. `normalise_dim` gives the
    // negative-index convention and upstream's out-of-range message.
    let dims: Vec<usize> = if dims_raw.is_empty() {
        (0..rank.max(1)).collect()
    } else {
        dims_raw
            .iter()
            .map(|&d| normalise_dim(OP, d, rank))
            .collect::<PyResult<Vec<_>>>()?
    };
    refuse_duplicate_dims(&dims)?;
    let dims_set: Vec<bool> = (0..rank).map(|d| dims.contains(&d)).collect();
    let tensor = norm_pow_walk(OP, t, tag, &dims_set, p, keepdim)?;
    finish(py, tensor, tag)
}

/// Upstream's `dim {d} appears multiple times in the list of dims`, on
/// **normalised** indices -- shared by `norm.ScalarOpt_dim`,
/// `linalg_vector_norm.default` and `squeeze.dims`, all three of which raise
/// this exact wording (measured on 2.13.0) for a repeated axis in a `dim`
/// list, independent of whether the repeat is spelled the same way twice
/// (`[0, 0]`) or as a positive/negative pair that normalises to the same axis
/// (`[0, -4]`). Upstream's message does not name the calling op, so this
/// takes none.
fn refuse_duplicate_dims(dims: &[usize]) -> PyResult<()> {
    let mut counted: Vec<usize> = Vec::new();
    for d in dims {
        if counted.contains(d) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "dim {d} appears multiple times in the list of dims"
            )));
        }
        counted.push(*d);
    }
    Ok(())
}

/// The accumulate-in-`opmath` reduction walk `norm.ScalarOpt_dim` and
/// `linalg_vector_norm.default` share (docs/DEMAND1.md §5, docs/DEMAND.md
/// §0.1 rank 3): both compute the same six-arm `ord`/`p` family --
///
/// ```text
///   p = 0      NormZeroOps    acc + (data == 0 ? 0 : 1)      project: acc
///   p = 1      NormOneOps     acc + |data|                   project: acc
///   p = 2      NormTwoOps     acc + data*data                project: sqrt(acc)
///   p = +inf   AbsMaxOps      max(acc, |data|)               project: acc
///   p = -inf   AbsMinOps      min(acc, |data|)               project: acc
///   otherwise  NormOps        acc + pow(|data|, p)           project: pow(acc, 1/p)
/// ```
///
/// `tag` is both the accumulate width (`float` for `Half`/`BFloat16`, the
/// storage dtype otherwise -- upstream's `acc_t`) and the output storage
/// dtype; a caller that promotes via `dtype=` passes the *target* dtype here,
/// not the input's own. `dims_set` is already resolved and de-duplicated by
/// the caller -- this function only runs the walk.
fn norm_pow_walk(
    op: &str,
    t: &Tensor,
    tag: TorchDType,
    dims_set: &[bool],
    p: f64,
    keepdim: bool,
) -> PyResult<Tensor> {
    let rank = dims_set.len();
    // **The reduction is a walk, not a chain of candle reductions, and the
    // reason is `acc_t`.** Upstream's `norm_kernel_tensor_iterator_impl`
    // dispatches `norm_kernel_cpu_impl<scalar_t, acc_t>` with `acc_t = float`
    // for `Half` and `BFloat16` and `acc_t = scalar_t` otherwise, so the
    // running `|x|^p` sum is kept in `float` for the reduced dtypes and
    // narrowed exactly once, at the end. Reducing with candle keeps every
    // partial sum in the storage dtype.
    //
    // docs/SCALAR.md §5 recorded the resulting disagreement -- `bfloat16`
    // 8/10, `float16` 8/10, `float32` 1/10, with `p=2` agreeing exactly
    // everywhere -- and left it as an accumulate-where change with its own
    // digest question. Re-measured before this rewrite over a 3x4 tensor at
    // ten `p` values, three `dim` lists and four dtypes: **29 of 120 rows
    // disagreed.**
    //
    // Each of upstream's six ops is transcribed rather than expressed through
    // another one, because they are not the same arithmetic:
    //
    // ```text
    //   p = 0      NormZeroOps    acc + (data == 0 ? 0 : 1)      project: acc
    //   p = 1      NormOneOps     acc + |data|                   project: acc
    //   p = 2      NormTwoOps     acc + data*data                project: sqrt(acc)
    //   p = +inf   AbsMaxOps      max(acc, |data|)               project: acc
    //   p = -inf   AbsMinOps      min(acc, |data|)               project: acc
    //   otherwise  NormOps        acc + pow(|data|, p)           project: pow(acc, 1/p)
    // ```
    //
    // `p = 2` squares rather than calling `pow(·, 2)`, and takes `sqrt`
    // rather than `pow(·, 0.5)`; that is upstream's own split and it is what
    // keeps the exactly-representable cases exact. The general arm uses
    // `powf` at `acc_t`, not `pow` at `f64` then narrowed -- for the reduced
    // dtypes the difference is invisible after narrowing and at `float32` it
    // is not.
    //
    // The walk is row-major over the *input*, which is the order the
    // partial sums are formed in, and that order is part of the answer:
    // floating addition is not associative.
    let shape = t.dims().to_vec();
    let device = t.device().clone();
    let values = match read_flat(op, t, tag)? {
        Flat::Float(v) => v,
        Flat::Int(_) => unreachable!("the dtype was checked above"),
    };

    let kept: Vec<usize> = shape
        .iter()
        .enumerate()
        .map(|(d, &extent)| if dims_set[d] { 1 } else { extent })
        .collect();
    let out_strides = contiguous_strides(&kept);
    let out_total: usize = kept.iter().product();

    // `acc_t`, and then the storage dtype once at the end.
    let wide = tag == TorchDType::Float64;
    let acc_narrow = |v: f64| if wide { v } else { v as f32 as f64 };
    let store_narrow = float_narrower(tag);

    #[derive(Clone, Copy, PartialEq)]
    enum Norm {
        Count,
        Sum,
        Squares,
        Max,
        Min,
        Power,
    }
    let mode = if p == 0.0 {
        Norm::Count
    } else if p == f64::INFINITY {
        Norm::Max
    } else if p == f64::NEG_INFINITY {
        Norm::Min
    } else if p == 1.0 {
        Norm::Sum
    } else if p == 2.0 {
        Norm::Squares
    } else {
        Norm::Power
    };
    let exponent = acc_narrow(p);

    let mut acc: Vec<Option<f64>> = vec![None; out_total];
    let mut coord = vec![0usize; rank];
    let total: usize = shape.iter().product();
    for src in 0..total {
        let mut dst = 0usize;
        for d in 0..rank {
            if !dims_set[d] {
                dst += coord[d] * out_strides[d];
            }
        }
        let data = values[src];
        let contrib = match mode {
            // `data == 0`, not `|data| == 0`: the same answer, and it is
            // upstream's spelling. NaN counts, because NaN != 0.
            Norm::Count => f64::from(data != 0.0),
            Norm::Sum | Norm::Max | Norm::Min => data.abs(),
            Norm::Squares => {
                let d = acc_narrow(data);
                acc_narrow(d * d)
            }
            Norm::Power => {
                let base = acc_narrow(data.abs());
                if wide {
                    base.powf(exponent)
                } else {
                    acc_narrow(f64::from((base as f32).powf(exponent as f32)))
                }
            }
        };
        acc[dst] = Some(match (acc[dst], mode) {
            (None, _) => contrib,
            (Some(a), Norm::Max) => {
                if contrib > a {
                    contrib
                } else {
                    a
                }
            }
            (Some(a), Norm::Min) => {
                if contrib < a {
                    contrib
                } else {
                    a
                }
            }
            (Some(a), _) => acc_narrow(a + contrib),
        });
        for d in (0..rank).rev() {
            coord[d] += 1;
            if coord[d] < shape[d] {
                break;
            }
            coord[d] = 0;
        }
    }

    let out_values: Vec<f64> = acc
        .into_iter()
        .map(|slot| {
            // An empty reduction contributes nothing; upstream's identity for
            // the summing arms is 0 and this reaches it the same way.
            let a = slot.unwrap_or(0.0);
            let projected = match mode {
                Norm::Squares => {
                    if wide {
                        a.sqrt()
                    } else {
                        acc_narrow(f64::from((a as f32).sqrt()))
                    }
                }
                Norm::Power => {
                    // `acc_t(1) / acc_t(p)`, at `acc_t` -- not the `f64`
                    // reciprocal narrowed afterwards.
                    let inverse = acc_narrow(1.0 / exponent);
                    if wide {
                        a.powf(inverse)
                    } else {
                        acc_narrow(f64::from((a as f32).powf(inverse as f32)))
                    }
                }
                _ => a,
            };
            store_narrow(projected)
        })
        .collect();

    let out_dims: Vec<usize> = if keepdim {
        kept
    } else {
        shape
            .iter()
            .enumerate()
            .filter(|(d, _)| !dims_set[*d])
            .map(|(_, &extent)| extent)
            .collect()
    };
    write_flat(op, Flat::Float(out_values), out_dims, &device, tag)
}

/// `aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None,
///     bool keepdim=False, *, ScalarType? dtype=None) -> Tensor`
///
/// docs/DEMAND.md §0.1 rank 3, measured in full in docs/DEMAND1.md §5. A
/// **distinct leaf** from `aten.norm.ScalarOpt_dim` above -- upstream gives
/// each its own dispatch registration -- but the same `ord`/`p` family, which
/// is why this shares `norm_pow_walk` rather than re-deriving it.
/// `F.normalize(v, p=2, dim=1)` fires this, `clamp_min.default`,
/// `expand.default`, `div.Tensor` in that order; the other three were
/// already implemented, so this one kernel closes `sentence_embed`.
///
/// Three ways it is not just `norm.ScalarOpt_dim` under a different name,
/// each measured on 2.13.0:
///
/// * **`dim` is `int[1]?`, not `int[1]`.** Absent and an explicit `dim=[]`
///   both mean "every axis", the same rule `norm.ScalarOpt_dim` applies when
///   its always-present `dim` is `[]`.
/// * **The dtype-mismatch wording is this op's own**: `"linalg.vector_norm:
///   Expected a floating point or complex tensor as input. Got {name}"` --
///   no trailing "instead." the way `norm()`'s does -- and it fires
///   regardless of whether `dtype=` is also given (an `int64` input with
///   `dtype=torch.float32` still refuses this way rather than promoting
///   first).
/// * **`dtype=` promotes before reducing**, and only *widening* is allowed
///   (`refuse_narrowing_dtype` below).
///
/// **The empty-reduction split** (`ord=2` gives `0.0`, `ord=±inf` raises) is
/// the one behaviour this op adds that the shared walk does not have on its
/// own -- `norm.ScalarOpt_dim` was never measured against an empty tensor
/// and is left as found, so the check lives here. Upstream's message has two
/// shapes, chosen by whether `dim` was given as a *non-empty* list, not by
/// whether the reduction happens to cover every axis:
///
/// ```text
/// dim=None / dim=[]    , whole tensor empty -> "...on an empty tensor because..."
/// dim=[..non-empty..]  , a named axis empty  -> "...on the dimension {raw}because..."
/// ```
///
/// (`{raw}because`, no space -- upstream's own concatenation, transcribed
/// rather than fixed.) The dimension **named is the first one in the
/// *given* list order whose extent is 0, printed exactly as the caller
/// spelled it** -- not normalised and not the smallest index: `dim=[2, 1]`
/// on a `(2, 0, 0)` tensor names `2`; `dim=[-1, -2]` on the same shape names
/// `-1`. An axis that is empty but not in the reduced set never raises --
/// `linalg_vector_norm(zeros(0, 3), dim=[1])` answers the empty `(0,)`
/// tensor with no error, because there are zero rows to compute over.
fn linalg_vector_norm_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.linalg_vector_norm.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let source_tag = input.tag();
    if !source_tag.is_floating_point() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "linalg.vector_norm: Expected a floating point or complex tensor as input. Got {}",
            scalar_type_name(source_tag)
        )));
    }
    let p = scalar_arg(OP, args, kwargs, 1, "ord")?
        .map(|s| s.as_f64())
        .unwrap_or(2.0);
    // Raw, un-normalised: the empty-reduction error prints whatever the
    // caller spelled, not the normalised axis (measured -- see the doc
    // comment above).
    let raw_dim: Option<Vec<isize>> = match optional(args, kwargs, 2, "dim")? {
        Some(value) if !value.is_none() => Some(match value.extract::<Vec<isize>>() {
            Ok(list) => list,
            Err(_) => vec![value.extract::<isize>()?],
        }),
        _ => None,
    };
    let keepdim = bool_arg(args, kwargs, 3, "keepdim")?.unwrap_or(false);
    let dtype = dtype_arg(args, kwargs, 4, "dtype")?;

    let t = input.tensor()?;
    let rank = t.rank();
    let full_reduce = raw_dim.as_ref().map_or(true, |d| d.is_empty());
    let dims: Vec<usize> = if full_reduce {
        (0..rank).collect()
    } else {
        raw_dim
            .as_ref()
            .unwrap()
            .iter()
            .map(|&d| normalise_dim(OP, d, rank))
            .collect::<PyResult<Vec<_>>>()?
    };
    refuse_duplicate_dims(&dims)?;
    let dims_set: Vec<bool> = (0..rank).map(|d| dims.contains(&d)).collect();

    // The identity-less arms: `±inf` has no additive identity, so an empty
    // reduction is refused rather than silently answering 0 the way the
    // summing/counting arms do.
    if p == f64::INFINITY || p == f64::NEG_INFINITY {
        let ord_name = if p == f64::INFINITY { "inf" } else { "-inf" };
        if full_reduce {
            if t.elem_count() == 0 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "linalg.vector_norm cannot compute the {ord_name} norm on an empty \
                     tensor because the operation does not have an identity"
                )));
            }
        } else {
            let shape = t.dims();
            for &raw in raw_dim.as_ref().unwrap() {
                let normalised = normalise_dim(OP, raw, rank)?;
                let extent = if rank == 0 { 1 } else { shape[normalised] };
                if extent == 0 {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "linalg.vector_norm cannot compute the {ord_name} norm on the \
                         dimension {raw}because this dimension is empty and the operation \
                         does not have an identity"
                    )));
                }
            }
        }
    }

    let target_tag = match dtype {
        Some(dtype) if dtype != source_tag => {
            refuse_narrowing_dtype(source_tag, dtype)?;
            dtype
        }
        _ => source_tag,
    };

    let tensor = norm_pow_walk(OP, t, target_tag, &dims_set, p, keepdim)?;
    finish(py, tensor, target_tag)
}

/// Upstream's floating-point "no narrowing" rule for `linalg_vector_norm`'s
/// `dtype=`, measured pairwise across the four floating dtypes this shim
/// stores (`Half`, `BFloat16`, `Float`, `Double`) rather than assumed:
/// `Half` and `BFloat16` are the same width and neither converts to the
/// other without narrowing (both refuse); `Half`/`BFloat16` -> `Float` ->
/// `Double` each widen and are allowed; every pairing that goes down that
/// ladder, or sideways between `Half` and `BFloat16`, is refused with
/// upstream's own wording.
fn refuse_narrowing_dtype(source: TorchDType, target: TorchDType) -> PyResult<()> {
    fn tier(tag: TorchDType) -> u8 {
        match tag {
            TorchDType::Float16 | TorchDType::BFloat16 => 1,
            TorchDType::Float32 => 2,
            TorchDType::Float64 => 3,
            _ => 0,
        }
    }
    if tier(target) > tier(source) {
        return Ok(());
    }
    Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
        "linalg.vector_norm: the dtype of the input ({}) should be convertible without \
         narrowing to the specified dtype ({})",
        scalar_type_name(source),
        scalar_type_name(target)
    )))
}

/// `aten::_weight_norm_interface(Tensor v, Tensor g, int dim=0)
///     -> (Tensor, Tensor)`
///
/// The second of `weight_norm`'s three pieces, and the one ARCH26.md §6 did
/// find -- it fires in the *forward*, where `norm.ScalarOpt_dim` above only
/// fires at construction.
///
/// ```text
/// norms = norm_except_dim(v, 2, dim)        keep `dim`, reduce every other axis
/// out   = v * (g / norms)
/// ```
///
/// Both asserted against upstream rather than taken from the formula:
/// `norms` equals `torch.norm_except_dim(v, 2, dim)` and `out` equals
/// `v * (g / norms)`, on real tensors.
///
/// **`dim` must be `0` or `v.dim() - 1`.** Upstream does not raise a friendly
/// error for anything else -- it trips an `INTERNAL ASSERT FAILED
/// (dim == 0 || dim == v.dim() - 1)`. Both live callers sit inside that: `vits`
/// takes the default `dim=0`, `sew_d` passes `dim=2` on a 3-D `Conv1d` weight,
/// which is `v.dim() - 1`. A middle dim is refused here by name rather than
/// reproducing an internal assertion.
///
/// **The norms come back `float32` for a `float16`/`bfloat16` input** while the
/// output keeps the input's dtype -- measured, and not a rule that follows from
/// anything else in this file, so it is read off rather than derived. `float32`
/// and `float64` inputs give norms of their own dtype.
///
/// A `v`/`g` dtype mismatch raises upstream (`expected scalar type Float but
/// found Double`) rather than promoting, and integral input raises
/// `"weight_norm_kernel" not implemented for 'Long'`. Both reproduced.
///
/// A zero row is not special-cased: upstream answers `nan` for it (the norm is
/// `0` and the division is `g / 0`), and so does this.
fn weight_norm_interface_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._weight_norm_interface.default";
    let v = tensor_arg(OP, args, kwargs, 0, "v")?;
    let g = tensor_arg(OP, args, kwargs, 1, "g")?;
    let dim = dim_arg(args, kwargs, 2, "dim")?.unwrap_or(0);

    let tag = v.tag();
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"weight_norm_kernel\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    if g.tag() != tag {
        // Upstream's wording, which names the C++ type rather than the torch
        // dtype -- the `c10_name` column of §5.2's table, capitalised.
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "expected scalar type {} but found {}",
            scalar_type_name(tag),
            scalar_type_name(g.tag())
        )));
    }

    let vt = v.tensor()?;
    let rank = vt.rank();
    let axis = normalise_dim(OP, dim, rank)?;
    if rank > 0 && axis != 0 && axis != rank - 1 {
        return Err(not_implemented(format!(
            "{OP}: dim must be 0 or v.dim() - 1, got {dim} for a {rank}-D v -- \
             upstream trips an internal assertion here rather than raising, and \
             both measured callers (vits at dim=0, sew_d at dim=v.dim()-1) sit \
             inside the supported range"
        )));
    }

    // **The whole computation runs in `float32` for a reduced-float input**,
    // and that is not a choice made here -- it is *why* upstream's `norms` come
    // back `float32` while `out` keeps the input's dtype.
    //
    // Computing the norm in `float16` and merely casting the result up is a
    // different number, and the golden cases measured it as one: for a
    // `float16` `(2,3)` v, upstream's norm is `2.4494898319244385` -- the
    // `float32` value of `sqrt(6)` -- where narrow-then-widen gives
    // `2.4492188`. So `norm_tag` is the dtype the arithmetic *happens* in, and
    // the `float32` result dtype is a consequence rather than a separate rule.
    let norm_tag = match tag {
        TorchDType::Float16 | TorchDType::BFloat16 => TorchDType::Float32,
        other => other,
    };
    let compute = PyDtype::new(norm_tag).storage(OP)?;
    let x = vt.fast_to(compute).map_err(|e| candle_err(OP, e))?;
    // `norm_except_dim(v, 2, dim)`: keep `axis`, reduce every other axis,
    // keepdim so the result broadcasts back against `v`.
    let mut norms = x.sqr().map_err(|e| candle_err(OP, e))?;
    for d in 0..rank {
        if d != axis {
            norms = norms.sum_keepdim(d).map_err(|e| candle_err(OP, e))?;
        }
    }
    let norms = norms.sqrt().map_err(|e| candle_err(OP, e))?;
    let gt = g.tensor()?.fast_to(compute).map_err(|e| candle_err(OP, e))?;
    // `out` is computed in the widened dtype too and narrowed once, at the end.
    let out_storage = PyDtype::new(tag).storage(OP)?;
    let out = gt
        .broadcast_div(&norms)
        .and_then(|scale| x.broadcast_mul(&scale))
        .and_then(|t| t.to_dtype(out_storage))
        .map_err(|e| candle_err(OP, e))?;

    // Promoted element by element: `promote` at the dispatcher's exit does not
    // look inside a tuple, the same reason `native_layer_norm` promotes its own.
    let pair = [
        crate::tensor::promote(py, finish(py, out, tag)?)?,
        crate::tensor::promote(py, finish(py, norms, norm_tag)?)?,
    ];
    Ok(PyTuple::new(py, pair)?.into_any().unbind())
}

/// `aten::repeat(Tensor self, SymInt[] repeats) -> Tensor`
///
/// **Tiling, not broadcasting.** `expand` above produces a view whose strides
/// are zero; `repeat` materialises a copy, and `[1,2,3].repeat(2, 3)` is
/// `(2, 9)` -- the *last* repeat multiplies the existing dimension and the
/// earlier ones are new leading dimensions. docs/ARCH26.md §8 found this op
/// missing across four of the six architectures (`deberta`, `deberta_v2`,
/// `sew_d`, `sam3_video`), and it is the wall both DeBERTas landed on the
/// moment `sqrt` existed.
///
/// candle has a `Tensor::repeat` and it is **not** called here, for two
/// measured reasons. Its whole loop is
///
/// ```ignore
/// for (idx, &repeat) in repeats.iter().enumerate() {
///     if repeat > 1 { inp = Tensor::cat(&vec![&inp; repeat], idx)? }
/// }
/// ```
///
/// so:
///
///   * **a repeat of `0` is skipped**, which makes it a no-op rather than an
///     empty dimension. Upstream's `[1,2,3].repeat(0)` is `(0,)` and candle's
///     would be `(3,)` -- a wrong *shape*, silently.
///   * **`repeats.len() < rank` is not refused.** candle takes the
///     `self.clone()` branch and then concatenates along axes that no longer
///     line up. Upstream raises, and the raise is the correct answer.
///
/// A third difference is subtler and is the one the aliasing table catches:
/// when every repeat is `1` candle returns `self.clone()`, and a candle clone
/// is an `Arc` clone. `x.repeat(1, 1)` would then *share storage with `x`*,
/// so `x.repeat(1,1).fill_(0)` would zero `x`. Upstream's `repeat` always
/// materialises. That is the `_to_copy` defect docs/VIEWS.md §6 records,
/// wearing a new hat: correct values, corrupted input, and every golden case
/// green because they all read the result.
///
/// So the tiling is written out here: resolve the shape first (which is where
/// both refusals live and where `0` is honoured), then build it with
/// `Tensor::cat`, then guarantee a copy.
///
/// Both refusals carry upstream's wording, measured on 2.13.0:
///
/// ```text
/// m.repeat([2])      Number of dimensions of repeat dims can not be smaller
///                    than number of dimensions of tensor
/// m.repeat([2, -1])  Trying to create tensor with negative dimension -2: [4, -2]
/// ```
///
/// The second reports the *product* (`2 * -1` for a size-2 axis) and the whole
/// computed output shape, not the offending repeat -- transcribed rather than
/// paraphrased, since the number in the message is not the number the caller
/// passed.
fn repeat_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.repeat.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let repeats = shape_arg(OP, args, kwargs, 1, "repeats")?;
    let source = input.tensor()?;
    let dims = source.dims().to_vec();

    if repeats.len() < dims.len() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Number of dimensions of repeat dims can not be smaller than number of \
             dimensions of tensor",
        ));
    }

    // The output shape, and both refusals with it. `offset` is how many
    // leading dimensions the result gains; below it the repeat *is* the
    // extent, at and above it the repeat multiplies the input's extent.
    let offset = repeats.len() - dims.len();
    let target: Vec<isize> = repeats
        .iter()
        .enumerate()
        .map(|(i, &r)| if i < offset { r } else { r * dims[i - offset] as isize })
        .collect();
    if let Some(&bad) = target.iter().find(|&&v| v < 0) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Trying to create tensor with negative dimension {bad}: {target:?}"
        )));
    }

    // Raise the rank first, so every subsequent axis index is the output's.
    // `reshape` on a non-contiguous input copies, which is fine and is not
    // what guarantees the copy below -- an already-rank-matched input skips
    // this entirely.
    let mut out = if offset > 0 {
        let mut raised = vec![1usize; offset];
        raised.extend_from_slice(&dims);
        source.reshape(raised).map_err(|e| candle_err(OP, e))?
    } else {
        source.clone()
    };

    for (axis, &r) in repeats.iter().enumerate() {
        let r = r as usize;
        if r == 1 {
            continue;
        }
        if r == 0 {
            // `Tensor::cat` of nothing is an error, so an empty axis is built
            // rather than concatenated. Everything after this point tiles an
            // empty tensor, which stays empty -- but the *other* axes still
            // have to come out the right size, so the loop is not short-cut.
            let mut zeroed = out.dims().to_vec();
            zeroed[axis] = 0;
            out = Tensor::zeros(zeroed, out.dtype(), out.device())
                .map_err(|e| candle_err(OP, e))?;
            continue;
        }
        let copies = vec![&out; r];
        out = Tensor::cat(&copies, axis).map_err(|e| candle_err(OP, e))?;
    }

    // Upstream's `repeat` always materialises. If nothing above concatenated
    // -- every repeat was 1 -- `out` is still an `Arc` clone of the input's
    // storage, and returning it would make `x.repeat(1,1).fill_(0)` zero `x`.
    let out = out.copy().map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

fn expand_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.expand.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let requested = shape_arg(OP, args, kwargs, 1, "size")?;
    let target = expand_target(OP, input.tensor()?.dims(), &requested)?;
    let out = input
        .tensor()?
        .broadcast_as(target)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// `aten::view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)` --
/// reinterpret the bytes, do not convert the numbers.
///
/// This is how a safetensors checkpoint becomes tensors. safetensors' `mmap`
/// backend hands torch a byte storage, makes a `uint8` tensor of it with
/// `torch.asarray`, and then spells the dtype with `.view(dtype)`; measured
/// with a `TorchDispatchMode` around `safe_open(...).get_tensor(k)`, the ops
/// are `empty` / `set_` / `view.dtype` / `view.default`. So this is not a
/// numeric op at all -- `1.0` viewed as `int32` is `1065353216`.
///
/// **The route is bytes out and bytes in**, `to_le_bytes` then
/// `from_le_bytes`, which is the same function the `torch.load` reader and
/// `torch.frombuffer` use. Anything that went through a numeric conversion
/// instead would round, and a checkpoint reader that rounds is worse than one
/// that refuses.
///
/// The three refusals are upstream's, measured on 2.13.0 including the C++
/// spelling of the dtype names (`dtype.rs::cpp_name`):
///
/// ```text
///   0-dim, different widths  self.dim() cannot be 0 to view Float as Byte ...
///   last dim indivisible     self.size(-1) must be divisible by 4 to view Byte as Float ...
///   last dim not packed      self.stride(-1) must be 1 to view Byte as Float ...
/// ```
///
/// The last one is why this kernel checks a stride even though it then makes a
/// contiguous copy. `stride(-1) == 1` is exactly the condition under which the
/// copy and upstream's genuine view agree: the reinterpretation only merges or
/// splits bytes *inside* the last dimension, so as long as that dimension is
/// packed, reading row-major and re-reading gives the same bytes in the same
/// places. Dropping the check would silently answer a shape upstream rejects.
fn view_dtype(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.view.dtype";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let want = dtype_arg(args, kwargs, 1, "dtype")?.ok_or_else(|| missing(OP, "dtype"))?;
    let have = input.tag();
    let (old, new) = (have.itemsize(), want.itemsize());
    let mut dims = input.dims().to_vec();

    if old != new {
        let (a, b) = (have.cpp_name(), want.cpp_name());
        let Some(&last) = dims.last() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "self.dim() cannot be 0 to view {a} as {b} (different element sizes)"
            )));
        };
        if input.tensor()?.stride().last() != Some(&1) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "self.stride(-1) must be 1 to view {a} as {b} (different element \
                 sizes), but got {}",
                input.tensor()?.stride().last().copied().unwrap_or(0)
            )));
        }
        if (last * old) % new != 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "self.size(-1) must be divisible by {} to view {a} as {b} \
                 (different element sizes), but got {last}",
                new / old
            )));
        }
        *dims.last_mut().expect("checked non-empty above") = last * old / new;
    }

    let bytes = crate::tensor::to_le_bytes(OP, input.tensor()?)?;
    let wrapped = crate::tensor::from_le_bytes(OP, &bytes, &dims, want)?;
    crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())
}

/// `reshape` and `view`. **They are the same kernel here and are not the same
/// op upstream**: `view` requires the existing strides to permit it and raises
/// otherwise, while `reshape` falls back to a copy. This shim copies in both
/// cases, so a `view` that upstream would reject succeeds here. That is a
/// divergence in the safe direction (the values are right either way) and it
/// is recorded in docs/TENSORBASE.md rather than papered over.
fn reshape_like(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    name: &str,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let requested = shape_arg(op, args, kwargs, 1, name)?;
    let target = resolve_shape(op, &requested, input.tensor()?.elem_count())?;
    let out = input
        .tensor()?
        .contiguous()
        .and_then(|t| t.reshape(target))
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, input.tag())
}

fn transpose_int(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.transpose.int";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let dim0 = normalise_dim(
        OP,
        dim_arg(args, kwargs, 1, "dim0")?.ok_or_else(|| missing(OP, "dim0"))?,
        rank,
    )?;
    let dim1 = normalise_dim(
        OP,
        dim_arg(args, kwargs, 2, "dim1")?.ok_or_else(|| missing(OP, "dim1"))?,
        rank,
    )?;
    let out = input
        .tensor()?
        .transpose(dim0, dim1)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// `aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)`
///
/// The most-called op of the four this round opens: `falcon` sends every weight
/// through `permute([1, 0])` and all four send attention through
/// `permute([0, 2, 1, 3])` (measured, docs/OPS4.md §1).
///
/// **Upstream this is an alias, and this shim's is not.** Measured on torch
/// 2.13.0: `permute(x, [1, 0])` shares `x.data_ptr()`, comes back
/// non-contiguous with the strides swapped, and writing through it changes `x`.
/// candle's `permute` also shares storage (it clones the `Arc` and permutes the
/// layout), and **that is now observable**: since docs/VIEWS.md §6 the in-place
/// ops write through the receiver's layout into the buffer it points at, so a
/// write through a permuted result reaches the base exactly as it does
/// upstream. docs/OPS4.md §5 has the original probe and docs/VIEWS.md §6.3 the
/// twenty-eight-row table this op is one line of.
///
/// The refusals were read off torch rather than invented, and the first one
/// would not have been guessed -- it names a layout nobody asked for:
///
/// ```text
/// len(dims) != rank  ->  "permute(sparse_coo): number of dimensions in the
///                         tensor input does not match the length of the
///                         desired ordering of dimensions i.e. input.dim() = 2
///                         is not equal to len(dims) = 3"
/// duplicate entry    ->  "permute(): duplicate dims are not allowed."
/// out of range       ->  IndexError, torch's usual wording
/// ```
///
/// Negative entries are allowed and normalised (`permute(x, [-1, -2])` equals
/// `permute(x, [1, 0])` on a 2-D input, measured), and a 0-D tensor takes
/// `dims=[]` and comes back unchanged -- so the rank check has to be exact
/// rather than `rank.max(1)`, which is why `normalise_dim` is not used for the
/// length test.
fn permute_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.permute.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let requested = shape_arg(OP, args, kwargs, 1, "dims")?;
    let rank = input.tensor()?.rank();
    if requested.len() != rank {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "permute(sparse_coo): number of dimensions in the tensor input does not \
             match the length of the desired ordering of dimensions i.e. input.dim() \
             = {rank} is not equal to len(dims) = {}",
            requested.len()
        )));
    }
    if rank == 0 {
        return finish(py, input.tensor()?.clone(), input.tag());
    }

    let mut order = Vec::with_capacity(rank);
    for &value in &requested {
        let dim = normalise_dim(OP, value, rank)?;
        if order.contains(&dim) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "permute(): duplicate dims are not allowed.",
            ));
        }
        order.push(dim);
    }
    let out = input
        .tensor()?
        .permute(order)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// `aten::t(Tensor(a) self) -> Tensor(a)`
///
/// `nn.Linear` reaches this on every projection (`x @ w.t()`), which is why a
/// Llama forward calls it more than anything else in this file.
///
/// Rank decides the behaviour and torch's rule is not "transpose the last two
/// dims": 0-D and 1-D come back **unchanged**, 2-D swaps, and 3-D or more is a
/// hard error rather than a batched transpose. Measured -- guessing it as
/// `transpose(-2, -1)` would silently compute on a 3-D input where upstream
/// raises.
fn t_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.t.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    if rank > 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "t() expects a tensor with <= 2 dimensions, but self is {rank}D"
        )));
    }
    let out = if rank == 2 {
        input
            .tensor()?
            .transpose(0, 1)
            .map_err(|e| candle_err(OP, e))?
    } else {
        input.tensor()?.clone()
    };
    finish(py, out, input.tag())
}

fn unsqueeze_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.unsqueeze.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    // `unsqueeze` is the one place the legal range is `[-(rank+1), rank]`:
    // the new dimension can go after the last existing one.
    let rank = input.tensor()?.rank();
    let raw = dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?;
    let extent = rank as isize + 1;
    let dim = if raw < 0 { raw + extent } else { raw };
    if dim < 0 || dim >= extent {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "{OP}: Dimension out of range (expected to be in range of [{}, {}], but got {raw})",
            -extent,
            extent - 1
        )));
    }
    let out = input
        .tensor()?
        .unsqueeze(dim as usize)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// The memory format a call asked for, as its name. The instances are built in
/// `bootstrap.py` and carry `_shim_name`; there is no Rust type for them
/// because there is nothing behind `torch.contiguous_format` but a label.
fn memory_format_name(value: &Bound<'_, PyAny>) -> String {
    value
        .getattr("_shim_name")
        .and_then(|v| v.extract::<String>())
        .unwrap_or_else(|_| value.str().map(|s| s.to_string()).unwrap_or_default())
}

fn reject_memory_format(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
) -> PyResult<()> {
    if let Some(value) = optional(args, kwargs, index, "memory_format")? {
        if !value.is_none() {
            let name = memory_format_name(&value);
            // The only two that mean "leave the layout alone", which is all
            // this shim can honour -- it has no strided layouts to rearrange.
            if name != "contiguous_format" && name != "preserve_format" {
                return Err(not_implemented(format!(
                    "{op}: memory_format=torch.{name} is not implemented in torch._C shim"
                )));
            }
        }
    }
    Ok(())
}

/// `layout=`, for the factories that are actually *called* with one.
///
/// `reject_unsupported` refuses any non-`None` layout, which is the right
/// default -- a dropped `layout=torch.sparse_coo` is a wrong answer with no
/// trace. But `torch.strided` names the dense layout this shim already has, and
/// the measured `scalar_tensor` call sites in `bloom`, `mpt` and `gptj` pass it
/// explicitly. Refusing there would block those architectures on an argument
/// that asks for exactly what is being handed back. Everything else still
/// refuses, with the layout named.
///
/// Deliberately *not* applied to `full`/`ones`/`empty`: those already refuse
/// every layout and no measured call site passes one, so widening them here
/// would change behaviour nothing has asked to change.
fn reject_layout(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
) -> PyResult<()> {
    if let Some(value) = optional(args, kwargs, index, "layout")? {
        // `memory_format_name` reads `_shim_name`, which `bootstrap.py` puts on
        // every one of these label objects -- layouts, memory formats and
        // qschemes alike -- so it reads a layout as well as it reads a format.
        // Both spellings are accepted because both arrive: the shim's own label
        // answers `strided` from `_shim_name`, and a *real* `torch.strided`
        // (which the golden harness hands to both sides) has no `_shim_name` and
        // falls back to its `str()`, `torch.strided`.
        let name = memory_format_name(&value);
        if !value.is_none() && name != "strided" && name != "torch.strided" {
            return Err(not_implemented(format!(
                "{op}: argument 'layout' not implemented in torch._C shim (got {value})"
            )));
        }
    }
    Ok(())
}

fn contiguous_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.contiguous.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    reject_memory_format(OP, args, kwargs, 1)?;
    // `transposed_contiguous`, not `contiguous`: it is a drop-in that blocks
    // the copy when the input is a last-two-swapped view and defers to candle
    // for everything else, including the already-contiguous case (where candle
    // clones the handle rather than the buffer, which is what makes
    // `contiguous.default` an aliasing op in the table of docs/VIEWS.md §6 --
    // that has to keep holding, and does, because the fast exit is the first
    // thing it checks).
    //
    // Bit-identical by construction: every output element is a copy of one
    // input element. docs/KERNELS26.md §7.
    let out = crate::tensor::transposed_contiguous(input.tensor()?)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// `aten::clone(Tensor self, *, MemoryFormat? memory_format=None)` -- a real
/// copy. candle's `Tensor::clone` is a refcount bump on shared storage;
/// `Tensor::copy` is the one that allocates, and `clone` has to allocate or
/// the in-place ops below would write through it.
fn clone_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.clone.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    reject_memory_format(OP, args, kwargs, 1)?;
    let out = input.tensor()?.copy().map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// `aten::detach(Tensor(a) self) -> Tensor(a)`
///
/// Upstream returns a *view*: a new tensor sharing storage with autograd
/// history stripped. There is no autograd here, so the history half is a
/// no-op, and **the sharing half now agrees**: `x.detach().fill_(0)` zeroes
/// `x`, as it does upstream. candle's clone was always an `Arc` clone; what
/// was missing was a write that reached storage, and that is
/// `PyTensorBase::write_into` (docs/VIEWS.md §6).
fn detach_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.detach.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    finish(py, input.tensor()?.clone(), input.tag())
}

/// `aten::alias(Tensor(a) self) -> Tensor(a)`
///
/// Upstream's cheapest op: a new tensor object over the same storage, with no
/// autograd stripping and no copy. **Both halves agree now** -- for the same
/// reason `detach` above does, since docs/VIEWS.md §6: an in-place write
/// through either of the two is seen by the other.
///
/// It reaches a Llama forward through GQA's `expand`/`reshape` chain, where the
/// result is read and never written, so the old divergence did not bite there.
/// It would have bitten a KV-cache write, which is the case that now works.
fn alias_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.alias.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    finish(py, input.tensor()?.clone(), input.tag())
}

/// `aten::_to_copy(Tensor self, *, ScalarType? dtype=None, ...)`
///
/// The dtype conversion behind `.to()`, `.float()` and `.long()`. A `bool`
/// destination normalises through `!= 0` and leaves by way of `boolean()`, so
/// the 0/1 invariant holds by construction rather than by hope.
fn to_copy_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._to_copy.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(input.tag());
    reject_unsupported(OP, args, kwargs, &[(2, "layout"), (4, "pin_memory")])?;
    reject_memory_format(OP, args, kwargs, 6)?;
    // `device=None` keeps the input where it is. See `device_arg_or_label`.
    let label = device_arg_or_label(args, kwargs, 3, "device", &input.device_label())?;
    // **cpu -> meta is the only transfer this build can perform, and it is a
    // discard rather than a copy.** Upstream agrees --
    // `torch.zeros(2).to("meta")` is a meta tensor, measured -- and it is the
    // one direction that needs no bytes: going the other way is
    // `Cannot copy out of meta tensor; no data!`, which `meta_to_copy` raises.
    if label.is_meta() {
        return meta_result(py, input.dims().to_vec(), tag);
    }
    let device = label.resolve()?;

    if tag == TorchDType::Bool {
        let out = input
            .tensor()?
            .to_device(&device)
            .and_then(|t| widen_f64(&t))
            .and_then(|t| t.ne(0f64))
            .map_err(|e| candle_err(OP, e))?;
        return finish(py, out, tag);
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let (had_dtype, stayed_put) = {
        let t = input.tensor()?;
        (t.dtype(), t.device().same_device(&device))
    };
    let out = input
        .tensor()?
        .to_device(&device)
        .and_then(|t| t.fast_to(storage))
        .map_err(|e| candle_err(OP, e))?;
    // **The copy this op is named after.** `to_device` and `fast_to` both
    // return `self.clone()` when there is nothing to do, and a candle clone is
    // an `Arc` clone -- so `x.to(torch.float32)` on a float32 tensor used to
    // hand back an *alias*. That was invisible while no in-place op wrote
    // through, and became a divergence the moment one did: measured on torch
    // 2.13.0, `y = x.to(torch.float32); y.fill_(0)` leaves `x` alone upstream,
    // and without this it would zero `x` here. `Tensor::copy` is candle's deep
    // copy; the dtype- and device-changing paths skip it, because those have
    // already allocated. docs/VIEWS.md §6.3.
    let out = if out.dtype() == had_dtype && stayed_put {
        out.copy().map_err(|e| candle_err(OP, e))?
    } else {
        out
    };
    finish(py, out, tag)
}

/// `aten::new_ones(Tensor self, SymInt[] size, *, ScalarType? dtype=None, ...)`
/// -- `ones`, with the dtype and device defaulted from an existing tensor
/// rather than from the global default.
fn new_ones_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    new_ones_or_zeros(py, args, kwargs, "aten.new_ones.default")
}

/// `aten::new_zeros(Tensor self, SymInt[] size, *, ScalarType? dtype=None,
///     Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor`
///
/// `bart`'s wall (docs/DEMAND.md rank 5):
/// `BartForConditionalGeneration.forward` -> `shift_tokens_right`, which
/// allocates the shifted copy with `input_ids.new_zeros(input_ids.shape)`.
/// Generic code duplicated across the whole BART-derived family (`bart`,
/// `marian`, `blenderbot`, `mvp`) -- though not as uniform as it looks, since
/// `mbart` and `pegasus` spell shift-right differently and hit other walls.
///
/// A leaf upstream, and **`new_ones`'s schema character for character** with a
/// different fill: same argument list, same positions, same
/// dtype-and-device-from-`self` defaulting. Measured on 2.13.0 and identical
/// to `new_ones` in every one of them -- `x.new_zeros(())` is 0-d,
/// `x.new_zeros(0)` is empty, a `bool` receiver gives `[False, False]`, an
/// explicit `dtype=` beats the receiver's, `device="meta"` gives a meta
/// tensor.
///
/// So it is the *same function*, switching only `Tensor::zeros` for
/// `Tensor::ones` -- the arrangement `zeros_or_empty_like` already uses for
/// its three `*_like` factories. Writing it out separately would mean two
/// copies of the "defaults come from the reference tensor" rule, which is the
/// half that is easy to get subtly wrong and impossible to see in a forward.
fn new_zeros_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    new_ones_or_zeros(py, args, kwargs, "aten.new_zeros.default")
}

/// The body both `new_ones` and `new_zeros` are. `op` selects the fill and is
/// also the name every refusal below carries, so a caller that spelled one of
/// them is never told about the other.
fn new_ones_or_zeros(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let size: Vec<usize> = required(op, args, kwargs, 1, "size")?.extract()?;
    let tag = dtype_arg(args, kwargs, 2, "dtype")?.unwrap_or(input.tag());
    reject_unsupported(op, args, kwargs, &[(3, "layout"), (5, "pin_memory")])?;
    let label = device_arg_or_label(args, kwargs, 4, "device", &input.device_label())?;
    if label.is_meta() {
        return meta_result(py, size, tag);
    }
    let device = label.resolve()?;
    let storage = PyDtype::new(tag).storage(op)?;
    let out = if op == "aten.new_ones.default" {
        Tensor::ones(size, storage, &device)
    } else {
        Tensor::zeros(size, storage, &device)
    }
    .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::_local_scalar_dense(Tensor self) -> Scalar`
///
/// The op behind `.item()` and `bool(t)`. Upstream reaches it for both -- a
/// `TorchDispatchMode` logger over `t.item()` records exactly this key, and
/// `aten::item` (which also exists) is never named.
fn local_scalar_dense(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._local_scalar_dense.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    if input.tensor()?.elem_count() != 1 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "a Tensor with {} elements cannot be converted to Scalar",
            input.tensor()?.elem_count()
        )));
    }
    let flat = input
        .tensor()?
        .flatten_all()
        .map_err(|e| candle_err(OP, e))?;
    if input.tag() == TorchDType::Bool {
        let value = flat
            .to_vec1::<u8>()
            .map_err(|e| candle_err(OP, e))?[0];
        return Ok((value != 0).into_bound_py_any(py)?.unbind());
    }
    if input.tag().is_floating_point() {
        let value = widen_f64(&flat)
            .and_then(|t| t.to_vec1::<f64>())
            .map_err(|e| candle_err(OP, e))?[0];
        return Ok(value.into_bound_py_any(py)?.unbind());
    }
    let value = flat
        .to_dtype(candle_core::DType::I64)
        .and_then(|t| t.to_vec1::<i64>())
        .map_err(|e| candle_err(OP, e))?[0];
    Ok(value.into_bound_py_any(py)?.unbind())
}

/// torch's negative-index convention for a single position along `dim`.
fn normalise_index(op: &str, index: isize, extent: usize) -> PyResult<usize> {
    let signed = extent as isize;
    let resolved = if index < 0 { index + signed } else { index };
    if resolved < 0 || resolved >= signed {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "{op}: index {index} is out of bounds for dimension with size {extent}"
        )));
    }
    Ok(resolved as usize)
}

fn select_int(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.select.int";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    if rank == 0 {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            "invalid index of a 0-dim tensor",
        ));
    }
    let dim = normalise_dim(
        OP,
        dim_arg(args, kwargs, 1, "dim")?.unwrap_or(0),
        rank,
    )?;
    let index = normalise_index(
        OP,
        int_arg(args, kwargs, 2, "index")?.ok_or_else(|| missing(OP, "index"))? as isize,
        input.tensor()?.dims()[dim],
    )?;
    let out = input
        .tensor()?
        .narrow(dim, index, 1)
        .and_then(|t| t.squeeze(dim))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// `aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None,
///                     SymInt? end=None, SymInt step=1)`
///
/// torch clamps rather than raising: an out-of-range bound gives an empty or
/// truncated result. Reproduced, because a raising slice would break the
/// `x[:seq_len]` idiom that mask construction is written with.
fn slice_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.slice.Tensor";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let dim = normalise_dim(OP, dim_arg(args, kwargs, 1, "dim")?.unwrap_or(0), rank)?;
    let extent = input.tensor()?.dims()[dim] as i64;
    let step = int_arg(args, kwargs, 4, "step")?.unwrap_or(1);
    if step <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "step must be greater than zero, got {step}"
        )));
    }

    let clamp = |value: i64| -> i64 {
        let shifted = if value < 0 { value + extent } else { value };
        shifted.clamp(0, extent)
    };
    let start = clamp(int_arg(args, kwargs, 2, "start")?.unwrap_or(0));
    let end = match int_arg(args, kwargs, 3, "end")? {
        // `sys.maxsize` is how Python spells "to the end" in a slice.
        Some(value) if value >= extent => extent,
        Some(value) => clamp(value),
        None => extent,
    };
    let length = (end - start).max(0) as usize;

    let narrowed = input
        .tensor()?
        .narrow(dim, start as usize, length)
        .map_err(|e| candle_err(OP, e))?;
    let out = if step == 1 {
        narrowed
    } else {
        let picks: Vec<i64> = (0..length as i64).step_by(step as usize).collect();
        let count = picks.len();
        let index = Tensor::from_vec(picks, count, input.tensor()?.device())
            .map_err(|e| candle_err(OP, e))?;
        narrowed
            .contiguous()
            .and_then(|t| t.index_select(&index, dim))
            .map_err(|e| candle_err(OP, e))?
    };
    finish(py, out, input.tag())
}

/// A bool/uint8 mask of rank `k` at `axis`, as the `k` integer index tensors
/// torch replaces it with.
///
/// This is upstream's own move (`at::native::expandTensors`): where the
/// gather happens a mask is not a separate kind of index, it is the
/// coordinates of its true elements. Doing the same here is what lets masks
/// and integer indices mix in one call without a second code path, and the
/// `k` tensors it produces are adjacent and equal-length by construction, so
/// the ordinary placement rule below puts them in the right place.
///
/// Measured: `x[mask3d]` on `(2,3,4,5)` with three true elements gives
/// `(3,5)` -- exactly what three adjacent index tensors of shape `(3,)` at
/// axes 0,1,2 produce.
fn mask_to_indices(
    op: &str,
    mask: &Tensor,
    axis: usize,
    dims: &[usize],
) -> PyResult<Vec<(usize, Vec<i64>, Vec<usize>)>> {
    let mask_dims = mask.dims().to_vec();
    if mask_dims.is_empty() {
        // Upstream does not answer here either: it trips an internal assertion
        // ("ntensor >= 3 INTERNAL ASSERT FAILED", measured), which is a bug
        // rather than a rule. Refusing by name beats reproducing a crash or
        // inventing the semantics it would have had.
        return Err(not_implemented(format!(
            "{op}: a 0-dim boolean mask is not implemented in torch._C shim -- \
             upstream reaches an internal assertion on this input rather than \
             defining it"
        )));
    }
    if axis + mask_dims.len() > dims.len() || dims[axis..axis + mask_dims.len()] != mask_dims[..] {
        // torch names the *first mismatching dimension*, and names it twice --
        // once relative to the mask and once relative to the tensor.
        let at = (0..mask_dims.len())
            .find(|&i| dims.get(axis + i) != Some(&mask_dims[i]))
            .unwrap_or(0);
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "The shape of the mask {mask_dims:?} at index {at} does not match the \
             shape of the indexed tensor {dims:?} at index {}",
            axis + at
        )));
    }
    let bytes = mask
        .flatten_all()
        .and_then(|t| t.to_dtype(candle_core::DType::U8))
        .and_then(|t| t.to_vec1::<u8>())
        .map_err(|e| candle_err(op, e))?;
    let mut coords: Vec<Vec<i64>> = vec![Vec::new(); mask_dims.len()];
    for (flat, &byte) in bytes.iter().enumerate() {
        if byte == 0 {
            continue;
        }
        let mut rest = flat;
        for d in (0..mask_dims.len()).rev() {
            coords[d].push((rest % mask_dims[d]) as i64);
            rest /= mask_dims[d];
        }
    }
    let count = coords[0].len();
    Ok(coords
        .into_iter()
        .enumerate()
        .map(|(d, values)| (axis + d, values, vec![count]))
        .collect())
}

/// `aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor`
///
/// Advanced ("fancy") indexing with any number of index tensors, integer or
/// boolean, in any positions. Every rule below was measured against torch
/// 2.13.0 rather than recalled, and three of them are what make this op easy
/// to get *plausibly* wrong -- right shape, wrong contents:
///
///   * **Where the result axes land.** If the indexed axes are *adjacent*,
///     the broadcast shape is spliced in at the first of them. If any
///     un-indexed axis *separates* them, the broadcast shape moves to the
///     **front** and the un-indexed axes follow in their original order.
///     Measured on `(2,3,4,5)` with index shapes `(2,1)` and `(3,)`:
///     `[i,None,j]` gives `(2,3,3,5)` (fronted) and `[None,i,j)` gives
///     `(2,2,3,5)` (spliced at axis 1). A kernel that always splices, or
///     always fronts, is right for half the calls and silently transposed
///     for the other half.
///   * **A `None` entry separates.** At this layer `None` does not mean "new
///     axis": torch has already unsqueezed `self` by the time the op is
///     called, and the `None` left in the list is an ordinary un-indexed
///     axis that breaks adjacency exactly like a `:`. Measured: the
///     `indices` lists for `x[i,None,j]` and `x[i,:,j]` are identical.
///   * **`uint8` is a mask, not an index.** Upstream accepts `long`, `int`,
///     `byte` and `bool`, and treats `byte` as a deprecated spelling of
///     `bool` -- `x[uint8_tensor]` gathers the *true* positions, not the
///     positions its values name. Reading it as an integer index is wrong
///     with a plausible shape, which is the failure mode this op invites.
///
/// A list shorter than the tensor's rank leaves the trailing axes alone;
/// upstream does not pad it out with `None`, and neither does this.
fn index_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.index.Tensor";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let raw = required(OP, args, kwargs, 1, "indices")?;
    let items: Vec<Bound<'_, PyAny>> = raw.extract()?;

    let dims = input.tensor()?.dims().to_vec();
    let rank = dims.len();
    if items.len() > rank {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "too many indices for tensor of dimension {rank} (got {})",
            items.len()
        )));
    }

    // (axis, values, shape), one entry per *indexed axis* -- masks having
    // already been expanded into one entry per axis they cover.
    let mut picks: Vec<(usize, Vec<i64>, Vec<usize>)> = Vec::new();
    let mut axis = 0usize;
    for item in items.iter() {
        if item.is_none() {
            axis += 1;
            continue;
        }
        let tensor = item.extract::<PyTensorBase>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "{OP}: indices must be tensors or None, got {}",
                item.get_type().name().map(|n| n.to_string()).unwrap_or_default()
            ))
        })?;
        match tensor.tag() {
            TorchDType::Int64 | TorchDType::Int32 => {
                let values = tensor
                    .tensor()?
                    .flatten_all()
                    .and_then(|t| t.to_dtype(candle_core::DType::I64))
                    .and_then(|t| t.to_vec1::<i64>())
                    .map_err(|e| candle_err(OP, e))?;
                picks.push((axis, values, tensor.tensor()?.dims().to_vec()));
                axis += 1;
            }
            TorchDType::Bool | TorchDType::UInt8 => {
                let expanded = mask_to_indices(OP, tensor.tensor()?, axis, &dims)?;
                axis += tensor.tensor()?.rank();
                picks.extend(expanded);
            }
            _ => {
                // Upstream's own wording, and it names four dtypes rather
                // than the one that was passed.
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "tensors used as indices must be long, int, byte or bool tensors",
                ));
            }
        }
    }

    if picks.is_empty() {
        // `x[None]`-only index lists never reach here (bootstrap.py handles
        // `None` with `unsqueeze`), and upstream trips an internal assertion
        // on an all-`None` list, so this stays the identity it always was.
        return finish(py, input.tensor()?.clone(), input.tag());
    }

    // The index tensors broadcast against each other, right-aligned.
    let broadcast_rank = picks.iter().map(|(_, _, shape)| shape.len()).max().unwrap_or(0);
    let mut broadcast: Vec<usize> = vec![1; broadcast_rank];
    for (_, _, shape) in &picks {
        let offset = broadcast_rank - shape.len();
        for (k, &extent) in shape.iter().enumerate() {
            let slot = &mut broadcast[offset + k];
            if *slot == extent || extent == 1 {
                continue;
            }
            if *slot == 1 {
                *slot = extent;
                continue;
            }
            let named = picks
                .iter()
                .map(|(_, _, s)| format!("{s:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "shape mismatch: indexing tensors could not be broadcast together \
                 with shapes {named}"
            )));
        }
    }

    // Each index tensor's stride *through the broadcast shape*: zero where it
    // is being stretched, its own row-major stride where it is not.
    let mut strides: Vec<Vec<usize>> = Vec::with_capacity(picks.len());
    for (_, _, shape) in &picks {
        let mut own = vec![1usize; shape.len()];
        for k in (0..shape.len().saturating_sub(1)).rev() {
            own[k] = own[k + 1] * shape[k + 1];
        }
        let offset = broadcast_rank - shape.len();
        let mut over = vec![0usize; broadcast_rank];
        for k in 0..shape.len() {
            over[offset + k] = if shape[k] == 1 { 0 } else { own[k] };
        }
        strides.push(over);
    }

    // Strides of the gathered subspace, in indexed-axis order.
    let indexed: Vec<usize> = picks.iter().map(|(axis, _, _)| *axis).collect();
    let mut sub = vec![1usize; picks.len()];
    for j in (0..picks.len().saturating_sub(1)).rev() {
        sub[j] = sub[j + 1] * dims[indexed[j + 1]];
    }

    let total: usize = broadcast.iter().product();
    let mut flat: Vec<i64> = Vec::with_capacity(total);
    let mut coords = vec![0usize; broadcast_rank];
    for linear in 0..total {
        let mut rest = linear;
        for k in (0..broadcast_rank).rev() {
            coords[k] = rest % broadcast[k];
            rest /= broadcast[k];
        }
        let mut offset = 0usize;
        for (j, (at, values, _)) in picks.iter().enumerate() {
            let mut position = 0usize;
            for k in 0..broadcast_rank {
                position += coords[k] * strides[j][k];
            }
            let raw = values[position];
            let extent = dims[*at] as i64;
            let resolved = if raw < 0 { raw + extent } else { raw };
            if resolved < 0 || resolved >= extent {
                // Upstream names the dimension being indexed and its size,
                // not the index tensor's own shape.
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "index {raw} is out of bounds for dimension {at} with size {extent}"
                )));
            }
            offset += (resolved as usize) * sub[j];
        }
        flat.push(offset as i64);
    }

    // Indexed axes to the front, gather along them as one flat axis, then put
    // the result axes where the placement rule says they go.
    let untouched: Vec<usize> = (0..rank).filter(|a| !indexed.contains(a)).collect();
    let mut order = indexed.clone();
    order.extend(untouched.iter().copied());
    let block: usize = indexed.iter().map(|&a| dims[a]).product();
    let trailing: Vec<usize> = untouched.iter().map(|&a| dims[a]).collect();
    let span: usize = trailing.iter().product();

    let index = Tensor::from_vec(flat, total, input.tensor()?.device())
        .map_err(|e| candle_err(OP, e))?;
    let mut shape: Vec<usize> = broadcast.clone();
    shape.extend(trailing.iter().copied());
    let mut out = input
        .tensor()?
        .permute(order)
        .and_then(|t| t.contiguous())
        .and_then(|t| t.reshape((block, span)))
        .and_then(|t| t.index_select(&index, 0))
        .and_then(|t| t.reshape(shape))
        .map_err(|e| candle_err(OP, e))?;

    // Adjacent indexed axes splice in place; separated ones stay at the front,
    // which is where `out` already has them.
    let adjacent = indexed.windows(2).all(|pair| pair[1] == pair[0] + 1);
    let at = indexed[0];
    if adjacent && at > 0 {
        let mut permutation: Vec<usize> = (0..at).map(|k| broadcast_rank + k).collect();
        permutation.extend(0..broadcast_rank);
        permutation.extend((at..trailing.len()).map(|k| broadcast_rank + k));
        out = out
            .permute(permutation)
            .and_then(|t| t.contiguous())
            .map_err(|e| candle_err(OP, e))?;
    }
    finish(py, out, input.tag())
}

// ---------------------------------------------------------------------------
// In-place ops
//
// These are the only ops that write. They take the *receiver object* rather
// than a copy of it, compute a whole replacement of the receiver's shape and
// dtype, and hand it to `write_back` below.
// docs/FROM_CONFIG.md §2.1 measured `fill_.Scalar` five times and
// `copy_.default` twice during `AutoModelForCausalLM.from_config`, so a shim
// without them cannot build a model at all.
//
// **They write through the receiver's layout into the buffer it already
// points at** (`PyTensorBase::write_into`). So an alias -- `detach()`,
// `alias()`, `unsqueeze`, `view`, or the `select.int`/`slice.Tensor`
// narrowings behind `x[0] = v` -- sees the write, which is what upstream does
// and what this file did not do before docs/VIEWS.md §6.
//
// The kernels themselves did not change shape to get there. Each of them
// already produced a fresh tensor with the receiver's shape and dtype; what
// changed is where that tensor's values go. `write_into` checks both
// properties on every call, so a kernel that starts producing something else
// raises rather than silently retagging the receiver.
//
// **`replace_with` is not a fallback for anything here.** The two callers left
// on it (`TensorBase.set_` and `tensor.data =`) are the ones where rebinding
// *is* the operation, and upstream rebinds there too.
// ---------------------------------------------------------------------------

/// The receiver of an in-place op, as the live Python object.
fn tensor_receiver<'py>(
    op: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyTensorBase>> {
    let value = required(op, args, kwargs, 0, "self")?;
    value.cast_into::<PyTensorBase>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{op}: argument 'self' must be a torch._C.TensorBase"
        ))
    })
}

/// Put an in-place kernel's computed replacement into the receiver's buffer.
///
/// The one line every in-place op in this file ends with, and the reason it is
/// a named function rather than a method call spelled out twelve times: the
/// choice between "write through the layout" and "swap the wrapper" is the
/// storage model, and a storage model with twelve independent spellings is one
/// that will acquire an exception.
///
/// `borrow()` and not `borrow_mut()` -- the mutation is candle's, behind the
/// `RwLock` in its storage `Arc`, so the Python-level borrow stays shared.
/// That also means a kernel may still hold a read borrow of the receiver when
/// it calls this, which several of them do.
///
/// The one thing it decides is `Overlap`, and that is a table rather than a
/// rule because upstream's answer for an *expanded* receiver is a table --
/// measured on torch 2.13.0 and written out in `tensor.rs::Overlap`. Keying it
/// on the op string here rather than passing a flag from each kernel keeps the
/// measurement in one place, where the next reader can compare all twelve
/// answers at once instead of hunting for the odd one out.
fn write_back(
    op: &str,
    receiver: &Bound<'_, PyTensorBase>,
    replacement: PyTensorBase,
) -> PyResult<()> {
    let overlap = match op {
        // Measured: upstream writes. `fill_.Scalar` and `zero_` write one
        // value everywhere, so a position visited twice is written the same
        // thing twice; `masked_fill_`/`index_put_` upstream warn that the
        // spelling is deprecated but still write.
        //
        // `fill_.Tensor` is deliberately NOT here even though it shares a
        // kernel with `fill_.Scalar`: upstream raises for it and writes for
        // the other, which is the sort of asymmetry that only survives being
        // measured rather than reasoned about.
        "aten.fill_.Scalar"
        | "aten.zero_.default"
        | "aten.masked_fill_.Scalar"
        | "aten.index_put_.default" => crate::tensor::Overlap::Allow,
        _ => crate::tensor::Overlap::Refuse,
    };
    receiver.borrow().write_into(op, &replacement, overlap)
}

/// `aten::fill_.Scalar/.Tensor(Tensor(a!) self, X value) -> Tensor(a!)`
///
/// Shape and dtype are the receiver's and do not change; only the values do.
fn fill_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
) -> PyResult<Py<PyAny>> {
    let receiver = tensor_receiver(op, args, kwargs)?;
    let raw = required(op, args, kwargs, 1, "value")?;
    let value = scalar_arg(op, args, kwargs, 1, "value")?.ok_or_else(|| missing(op, "value"))?;
    let (tag, shape, device, numel) = {
        let borrowed = receiver.borrow();
        (
            borrowed.tag(),
            borrowed.tensor()?.shape().clone(),
            borrowed.tensor()?.device().clone(),
            borrowed.tensor()?.elem_count(),
        )
    };
    // The same `c10::checked_convert` reproduction `full` uses, and the golden
    // harness caught its absence here the same way: `fill_(float16, 1e6)` gave
    // `inf` where torch raises, and `fill_(int32, 2**31)` wrapped to
    // `-2**31`. `fill_` is in fact where upstream's numel==1 hole lives (the
    // CPU fast path this check has to skip), so the rule is not merely
    // borrowed from `full` -- it is the same code path upstream.
    if !raw.is_instance_of::<PyTensorBase>() {
        checked_convert(&raw, raw.is_instance_of::<pyo3::types::PyInt>(), tag, numel)?;
    }

    let replacement = if tag == TorchDType::Bool {
        let truthy = u8::from(value.as_f64() != 0.0);
        PyTensorBase::boolean(
            Tensor::full(truthy, shape, &device).map_err(|e| candle_err(op, e))?,
        )?
    } else {
        let storage = PyDtype::new(tag).storage(op)?;
        let filled = if storage.is_int() {
            Tensor::full(value.as_i64(), shape, &device)
        } else {
            Tensor::full(value.as_f64(), shape, &device)
        }
        .and_then(|t| t.fast_to(storage))
        .map_err(|e| candle_err(op, e))?;
        PyTensorBase::new(filled)?
    };
    write_back(op, &receiver, replacement)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::zero_(Tensor(a!) self) -> Tensor(a!)`
///
/// **Not a spelling of `fill_(0)`, even though it computes the same values.**
/// It is a separate overload upstream, and overloads are part of this shim's
/// key (see the module note), so folding it into `fill_inplace` would make
/// `_aten_implemented()` claim one op where two were asked for.
///
/// It is here because of where it fires, which was measured rather than
/// assumed: **`zero_` never appears in a forward pass.** Recording
/// `TorchDispatchMode` over construction and over inference separately,
/// `nn.LayerNorm(8)`'s constructor calls `empty.memory_format` ×2,
/// `fill_.Scalar` ×1 (the weight, to 1) and `zero_.default` ×1 (the bias),
/// and the forward calls neither. `nn.Linear` does not call it at all --
/// its `reset_parameters` uses `uniform_`. So `zero_` is not on the op-count
/// tail the architecture sweep measures; it is on the path *before* it, and a
/// model that cannot be constructed never reaches the tail at all. That is
/// why docs/GPT2.md saw `nn.LayerNorm` fail twice over: answering
/// `_C._get_cudnn_enabled` only moves the failure to this kernel.
///
/// Zero is representable exactly in every dtype this shim stores, so unlike
/// `fill_` there is no `checked_convert` here -- there is no value to overflow.
/// `bool` zeroes to `False` and a `nan`/`inf` element is overwritten like any
/// other (both measured).
fn zero_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.zero_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let (tag, shape, device) = {
        let borrowed = receiver.borrow();
        (
            borrowed.tag(),
            borrowed.tensor()?.shape().clone(),
            borrowed.tensor()?.device().clone(),
        )
    };
    let storage = PyDtype::new(tag).storage(OP)?;
    let zeros = Tensor::zeros(shape, storage, &device).map_err(|e| candle_err(OP, e))?;
    let replacement = if tag == TorchDType::Bool {
        PyTensorBase::boolean(zeros)?
    } else {
        PyTensorBase::new(zeros)?
    };
    write_back(OP, &receiver, replacement)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False)`
///
/// The destination keeps its own shape and dtype; the source is broadcast and
/// cast into them. That asymmetry is torch's -- `int_t.copy_(float_t)` gives
/// an int tensor, measured -- and it is why this is not just an assignment.
fn copy_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.copy_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let source = tensor_arg(OP, args, kwargs, 1, "src")?;
    let (tag, shape) = {
        let borrowed = receiver.borrow();
        (borrowed.tag(), borrowed.tensor()?.shape().clone())
    };

    let widened = source
        .tensor()?
        .broadcast_as(shape)
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(OP, e))?;
    let replacement = if tag == TorchDType::Bool {
        PyTensorBase::boolean(
            widen_f64(&widened)
                .and_then(|t| t.ne(0f64))
                .map_err(|e| candle_err(OP, e))?,
        )?
    } else {
        let storage = PyDtype::new(tag).storage(OP)?;
        PyTensorBase::new(widened.fast_to(storage).map_err(|e| candle_err(OP, e))?)?
    };
    write_back(OP, &receiver, replacement)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// Upstream's `canCast(result, dest)` guard on an in-place write, with
/// upstream's own message.
///
/// **This is the rule that separates in-place from out-of-place, and it is
/// the one the previous `add_` kernel did not have.** An out-of-place op may
/// promote as far as it likes because it allocates the result; an in-place op
/// has a destination already, so upstream computes the promoted result dtype
/// and then *refuses* if it cannot be cast back:
///
/// ```text
/// float32.add_(int32_tensor)     ok       promote -> float32, fits
/// int32.add_(float32_tensor)     RAISE    "result type Float can't be cast
///                                          to the desired output type Int"
/// int32.mul_(2.5)                RAISE    same, via the wrapped-number rule
/// int64.div_(2)                  RAISE    div always floats
/// int64.exp_()                   RAISE    "... output type Long"
/// ```
///
/// `c10::canCast`, transcribed: a float result may not land in an integral
/// destination, and a non-bool result may not land in a `bool` one (bool is a
/// promotion *category*, which is why `bool_tensor += 5` is disallowed even
/// though every value would fit). Complex has no storage here, so its arm is
/// unreachable and omitted rather than written and never taken.
///
/// The shim used to *compute* in the refusing rows -- `int32.add_(float)` cast
/// the operand down and returned a wrong-by-truncation answer, recorded as a
/// `torch_error` golden case rather than fixed. Computing where upstream
/// raises is the silent-divergence direction, so it is fixed here and the case
/// is promoted to `both_error`.
fn inplace_cast_check(op: &str, result: TorchDType, dest: TorchDType) -> PyResult<()> {
    let refuses = (result.is_floating_point() && !dest.is_floating_point())
        || (result != TorchDType::Bool && dest == TorchDType::Bool);
    if refuses {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "result type {} can't be cast to the desired output type {}",
            scalar_type_name(result),
            scalar_type_name(dest)
        )));
    }
    let _ = op;
    Ok(())
}

/// `aten::add_.Tensor`, `sub_.Tensor` and `mul_.Tensor` -- one kernel, because
/// they differ only in `apply_arith`'s arm.
///
/// `add_` opened `falcon` (docs/TAIL.md): its residual connections write
/// `hidden_states += attn_output` rather than rebinding the name, so the trace
/// calls this overload and not `add.Tensor`. `sub_` and `mul_` joined it in
/// docs/ARCH20.md §8 -- they had no kernel *and* no member, which is why
/// `x -= y` and `x *= y` refused outright.
///
/// **Aliasing is `write_back`'s, the same as every other in-place op in this
/// file**: the result is computed into a fresh tensor of the receiver's shape
/// and dtype and then written through the receiver's *layout*, so an alias or
/// a view taken before this call does observe the update, as upstream's does.
/// docs/VIEWS.md §6.
///
/// Two rules, both upstream's:
///
///   * **`torch.bool` follows `arith_tag`**, so `mul_` accepts it (a bool
///     product is the logical and, exactly, under the tag's 0/1 invariant --
///     and upstream agrees: `tensor([True,False]).mul_(tensor([True,True]))`
///     is `[True, False]`, measured) while `add_` and `sub_` refuse it.
///     Upstream's bool `add_` is a logical *or*, which this shim implements in
///     neither the in-place nor the out-of-place spelling, so `add_` does not
///     acquire a capability `add.Tensor` lacks.
///   * **the cast check**, `inplace_cast_check` above. The result dtype is the
///     one `add.Tensor`/`mul.Tensor` would have produced -- the in-place
///     spelling must not compute a different function from the out-of-place
///     one -- and it is refused rather than truncated when it does not fit.
///
/// One narrower-than-upstream case remains and is recorded rather than hidden:
/// when the promoted result is *wider* than the receiver (`float16.add_(
/// float64_tensor)`), upstream accumulates in the wider type and narrows once,
/// while this accumulates in `opmath_in(receiver)`. Both narrow to the
/// receiver; they can differ in the last bit.
fn arith_inplace_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Arith,
) -> PyResult<Py<PyAny>> {
    let receiver = tensor_receiver(op, args, kwargs)?;
    let other = tensor_arg(op, args, kwargs, 1, "other")?;
    // `mul_.Tensor` has no `alpha` in its schema; `optional` simply finds
    // nothing at index 2 and this is 1.0, so the one kernel still serves it.
    let alpha = alpha_arg(op, args, kwargs)?;

    let (tag, shape) = {
        let borrowed = receiver.borrow();
        (borrowed.tag(), borrowed.tensor()?.shape().clone())
    };
    let operand = if tag == other.tag() {
        tag
    } else {
        promote_types(tag, other.tag()).ok_or_else(|| {
            not_implemented(format!(
                "{op}: dtype promotion not implemented in torch._C shim: {} vs {}",
                tag.name(),
                other.tag().name()
            ))
        })?
    };
    let result = arith_tag(op, kind, operand, None)?;
    inplace_cast_check(op, result, tag)?;

    let storage = PyDtype::new(tag).storage(op)?;
    // Same widening as the out-of-place sibling -- the in-place spelling must
    // not compute a different function from it. See `opmath_in`.
    let acc = opmath_in(storage);
    let lhs = {
        let borrowed = receiver.borrow();
        borrowed
            .tensor()?
            .fast_to(acc)
            .map_err(|e| candle_err(op, e))?
    };
    let rhs = other
        .tensor()?
        .fast_to(acc)
        .and_then(|t| t.broadcast_as(shape))
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(op, e))?;
    let rhs = scale_by_alpha(op, &rhs, alpha, storage)?;
    let out = apply_arith(op, kind, &lhs, &rhs)?
        .fast_to(storage)
        .map_err(|e| candle_err(op, e))?;
    // `tagged` and not `PyTensorBase::new`: `write_into` compares the *torch*
    // tags, and `boolean()` is the only constructor allowed to attach the
    // `bool` one (BOOL.md §6.3). Only `mul_` reaches the bool arm --
    // `arith_tag` refuses it for `add_`/`sub_` -- and it reaches it for the
    // reason `mul.Tensor` accepts bool out of place: the product *is* the
    // logical and. The golden case `mul_(dtype=bool)` is what found this.
    write_back(op, &receiver, tagged(out, tag)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// Wrap a computed tensor with the right torch tag, for the in-place kernels
/// whose result goes to `write_back` rather than to `finish`.
///
/// The same branch `finish` makes, factored out because `write_back` takes a
/// `PyTensorBase` and not a `Py<PyAny>`. Getting it wrong is not silent --
/// `write_into` compares tags and refuses with an "internal error" -- but it
/// is only *not* silent because that check exists; before docs/VIEWS.md §6 it
/// would have retagged the receiver.
fn tagged(tensor: Tensor, tag: TorchDType) -> PyResult<PyTensorBase> {
    if tag == TorchDType::Bool {
        PyTensorBase::boolean(tensor)
    } else {
        PyTensorBase::new(tensor)
    }
}

/// `aten::add_.Scalar`, `sub_.Scalar` and `mul_.Scalar`.
///
/// **Upstream's *dispatcher* never names these from a Python call**, and that
/// is measured: `t.add_(2)` reports `aten.add_.Tensor`, because
/// `add_.Scalar`'s CompositeExplicitAutograd body wraps the number into a 0-d
/// tensor and redispatches. They exist here for the same reason `add.Scalar`
/// and `mul.Scalar` do -- the *parser* is what `methods.json` reproduces, and
/// `x += 2` binds a `Scalar` signature there. Skipping them would make
/// `x += 2` refuse while `x += tensor(2)` worked, which is a difference no
/// caller can see a reason for.
///
/// The dtype rule is `arith_tag`'s wrapped-number promotion, then the same
/// `inplace_cast_check`: `int32.mul_(2.5)` refuses ("result type Float can't
/// be cast to the desired output type Int", upstream's words) rather than
/// truncating to `int32` behind the caller's back.
fn arith_inplace_scalar(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Arith,
) -> PyResult<Py<PyAny>> {
    let receiver = tensor_receiver(op, args, kwargs)?;
    let other = scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
    let alpha = alpha_arg(op, args, kwargs)?;

    let tag = receiver.borrow().tag();
    let result = arith_tag(op, kind, tag, Some(!other.is_int()))?;
    inplace_cast_check(op, result, tag)?;

    let storage = PyDtype::new(tag).storage(op)?;
    let acc = opmath_in(storage);
    let lhs = {
        let borrowed = receiver.borrow();
        borrowed
            .tensor()?
            .fast_to(acc)
            .map_err(|e| candle_err(op, e))?
    };
    // Built exactly as `arith_scalar` builds it, including the narrow-then-
    // widen for the float case: torch's promotion makes a Python float beside
    // a `bfloat16` tensor a `bfloat16` operand, so `x += 0.3` adds
    // `0.30078125` there (docs/GENERATE.md §3.2). Building at `acc` would add
    // `0.3` and the in-place form would disagree with the out-of-place one.
    let rhs = if storage.is_int() {
        Tensor::full(other.as_i64() * (alpha as i64), (), lhs.device())
            .and_then(|t| t.fast_to(acc))
    } else {
        Tensor::full(other.as_f64() * alpha, (), lhs.device())
            .and_then(|t| t.fast_to(storage))
            .and_then(|t| t.fast_to(acc))
    }
    .map_err(|e| candle_err(op, e))?;
    // `div_.Scalar` takes upstream's reduced-float reciprocal path, exactly as
    // the out-of-place `div.Scalar` above does -- see
    // `div_scalar_reduced_float`. The in-place and out-of-place forms have to
    // agree, and a `float16` `x /= 0.3` disagreeing with `x = x / 0.3` by one
    // representable step is the kind of difference nobody would look for.
    let computed = match kind {
        Arith::Div => div_scalar_reduced_float(op, &lhs, other.as_f64(), storage)?,
        _ => None,
    };
    let out = match computed {
        Some(t) => t,
        None => apply_arith(op, kind, &lhs, &rhs)?,
    }
    .fast_to(storage)
    .map_err(|e| candle_err(op, e))?;
    write_back(op, &receiver, tagged(out, tag)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::neg_(Tensor(a!) self) -> Tensor(a!)`
///
/// `neg.default`'s value and refusals, written through the receiver. It keeps
/// the receiver's dtype -- `int64.neg_()` is `int64`, measured -- so there is
/// no cast check to make; the only refusals are `neg`'s own two, and both are
/// upstream's: `bool` (upstream points at `~`/`logical_not()` instead) and the
/// wide unsigned dtypes, which have no `neg_cpu` kernel upstream at all.
///
/// The integral path does not go through candle for the reason `neg_default`
/// gives at length: `candle_core`'s `neg` is a `unary_op!` whose integer arms
/// are `todo!()`, so calling it on an `i64` tensor **panics** rather than
/// returning an error. `0 - x` is the same value with no panic.
fn neg_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.neg_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let tag = receiver.borrow().tag();
    if tag == TorchDType::Bool {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Negation, the `-` operator, on a bool tensor is not supported. If you are \
             trying to invert a mask, use the `~` or `logical_not()` operator instead.",
        ));
    }
    if matches!(
        tag,
        TorchDType::UInt16 | TorchDType::UInt32 | TorchDType::UInt64
    ) {
        return Err(not_implemented(format!(
            "{OP}: torch has no neg_cpu kernel for {}",
            tag.name()
        )));
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let source = {
        let borrowed = receiver.borrow();
        borrowed.tensor()?.contiguous().map_err(|e| candle_err(OP, e))?
    };
    let out = source
        .zeros_like()
        .and_then(|z| z.sub(&source))
        .and_then(|t| t.fast_to(storage))
        .map_err(|e| candle_err(OP, e))?;
    write_back(OP, &receiver, PyTensorBase::new(out)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::exp_(Tensor(a!) self) -> Tensor(a!)`
///
/// `exp.default`'s value, written through the receiver -- with the one
/// difference in-place always makes. `exp` *promotes*: `torch.exp(int64_t)` is
/// `float32`. An in-place `exp_` has nowhere to put that, so upstream refuses
/// rather than truncating: `int64_tensor.exp_()` raises "result type Float
/// can't be cast to the desired output type Long", measured. Every floating
/// dtype keeps its own (`float16` in, `float16` out) exactly as `unary_float`
/// does out of place.
///
/// **The refusal is also load-bearing for safety, which a sabotage run found
/// rather than the reading.** Disabling `inplace_cast_check` did not produce a
/// wrong number here -- it produced
/// `PanicException: not yet implemented: no unary function for i64` and took
/// the golden harness's interpreter down mid-run. candle's `exp` is a
/// `unary_op!` whose integer arms are `todo!()`, the same trap `neg_default`
/// documents from the other side. So the integral path must never reach
/// candle, and the cast check is what guarantees it: this kernel casts to
/// `storage` (the *receiver's* dtype, unchanged) rather than to a float, so
/// without the check an `int64` receiver would hand `i64` bytes to `t.exp()`.
fn exp_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.exp_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let tag = receiver.borrow().tag();
    // `unary_float`'s promotion, then the in-place guard against it.
    let result = if tag.is_floating_point() {
        tag
    } else {
        default_float()
    };
    inplace_cast_check(OP, result, tag)?;
    let storage = PyDtype::new(tag).storage(OP)?;
    let out = {
        let borrowed = receiver.borrow();
        borrowed
            .tensor()?
            .fast_to(storage)
            .and_then(|t| t.exp())
            .map_err(|e| candle_err(OP, e))?
    };
    write_back(OP, &receiver, PyTensorBase::new(out)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::relu_(Tensor(a!) self) -> Tensor(a!)`
///
/// The in-place sibling of `relu.default`, needed because `F.relu(x,
/// inplace=True)` traces to this overload specifically -- `torch.relu_` is a
/// genuinely separate op from `torch.relu`, not an alternate spelling of it
/// (measured: `torch.ops.aten.relu_.default` and `torch.ops.aten.relu.default`
/// are different `OpOverload` objects with different schemas,
/// `Tensor(a!) self` vs plain `Tensor self`), and `aten.rs` had a kernel for
/// neither name before this (docs/SPELLINGS.md §6.6 measured zero).
///
/// **Aliasing is `write_back`'s, the same as `add_inplace`/`copy_inplace`/
/// every other in-place op in this file** (docs/VIEWS.md §6): the result is
/// written through the receiver's layout, so a view or alias taken before this
/// call observes the update. Upstream `relu_` is an alias-preserving in-place
/// write (measured: `y = x.view(-1); x.relu_(); y` shows the update through
/// the view on real torch) and this shim now reproduces that.
///
/// The value and the refusal are `relu.default`'s, reused rather than
/// re-derived: `torch.bool` raises with upstream's exact wording ("Boolean
/// inputs not supported for relu", measured on both the out-of-place and
/// in-place overloads), and every other dtype computes `x < 0 ? 0 : x`
/// element-wise, preserving `nan` and the sign of `-0.0` the same way
/// `relu_default`'s doc comment measured.
fn relu_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.relu_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let tag = receiver.borrow().tag();
    if tag == TorchDType::Bool {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Boolean inputs not supported for relu",
        ));
    }
    let source = {
        let borrowed = receiver.borrow();
        // `?` because `tensor()` now returns a Result -- a meta tensor has no
        // bytes to make contiguous. This kernel arrived on the other branch, so
        // the two changes never textually conflicted and the compiler is what
        // caught it.
        borrowed
            .tensor()?
            .contiguous()
            .map_err(|e| candle_err(OP, e))?
    };
    let zeros = source.zeros_like().map_err(|e| candle_err(OP, e))?;
    let out = source
        .lt(&zeros)
        .and_then(|negative| negative.where_cond(&zeros, &source))
        .map_err(|e| candle_err(OP, e))?;
    write_back(OP, &receiver, PyTensorBase::new(out)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// The in-place sibling of `unary_float`'s family (`cos_`/`sin_`/`erf_`/
/// `log_`/`reciprocal_`/`tanh_`/`sqrt_`), for `docs/SPELLINGS.md` §9's
/// fifteen-name gap. `rsqrt_`/`expm1_`/`log2_`/`sigmoid_` are the same rule
/// but not `Unary` variants (their out-of-place kernels do not go through
/// `unary_float` either), so they get their own functions below rather than
/// an `Unreachable` arm here.
///
/// **In-place cannot promote, and that is the whole dtype story.**
/// `unary_float`'s out-of-place rule sends an integral or boolean input to
/// the default float; an in-place receiver has nowhere to put a wider dtype
/// than the one it already has, so every one of these refuses instead,
/// measured against upstream 2.13.0 for all seven names this function
/// serves plus the four that share the rule without sharing the function:
///
/// ```text
/// int64/int32/int16/uint8/bool . sqrt_/cos_/sin_/erf_/log_/reciprocal_/
///   tanh_/rsqrt_/expm1_/log2_/sigmoid_
///     -> RuntimeError: result type Float can't be cast to the desired
///        output type <Long/Int/Short/Byte/Bool>
/// float16/bfloat16/float32/float64 -> computes, own dtype kept
/// ```
///
/// `inplace_cast_check` is the same door `exp_inplace` already uses for the
/// identical shape of refusal, so the wording is not re-derived here.
fn unary_float_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Unary,
) -> PyResult<Py<PyAny>> {
    let receiver = tensor_receiver(op, args, kwargs)?;
    let tag = receiver.borrow().tag();
    inplace_cast_check(op, unary_float_tag(tag), tag)?;
    let storage = PyDtype::new(tag).storage(op)?;
    let out = {
        let borrowed = receiver.borrow();
        borrowed
            .tensor()?
            .fast_to(storage)
            .and_then(|t| match kind {
                Unary::Cos => t.cos(),
                Unary::Sin => t.sin(),
                Unary::Reciprocal => t.recip(),
                Unary::Tanh => t.tanh(),
                Unary::Log => t.log(),
                Unary::Sqrt => t.sqrt(),
                Unary::Erf => t.erf(),
                // `exp_` has had its own kernel (`exp_inplace`) since before
                // this function existed, so nothing routes `Unary::Exp`
                // here -- but the match has to be exhaustive.
                Unary::Exp => {
                    Err(candle_core::Error::Msg(format!(
                        "{op}: torch._C shim internal error -- exp_ does not \
                         route through unary_float_inplace (aten.rs)"
                    )))
                }
            })
        .map_err(|e| candle_err(op, e))?
    };
    write_back(op, &receiver, PyTensorBase::new(out)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)`
///
/// `rsqrt_default`'s value (`1/sqrt(x)`), written through the receiver.
/// Not `unary_float_inplace` for the same reason `rsqrt_default` is not
/// `unary_float`: it composes `sqrt`+`recip` itself rather than naming a
/// `Unary` variant. The dtype rule is the family's, re-measured rather than
/// assumed: `int64.rsqrt_()` raises "result type Float can't be cast to the
/// desired output type Long", and every floating dtype keeps its own width.
fn rsqrt_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.rsqrt_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let tag = receiver.borrow().tag();
    inplace_cast_check(OP, unary_float_tag(tag), tag)?;
    let storage = PyDtype::new(tag).storage(OP)?;
    let out = {
        let borrowed = receiver.borrow();
        borrowed
            .tensor()?
            .fast_to(storage)
            .and_then(|t| t.sqrt())
            .and_then(|t| t.recip())
            .map_err(|e| candle_err(OP, e))?
    };
    write_back(OP, &receiver, PyTensorBase::new(out)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::expm1_(Tensor(a!) self) -> Tensor(a!)`
///
/// `expm1_default`'s value (`exp(x) - 1`, through `f64::exp_m1` element by
/// element rather than a subtraction that cancels near zero -- see that
/// function's doc comment for the measurement), written through the
/// receiver. The dtype rule is the family's: in-place cannot promote, so
/// every non-floating receiver refuses (measured) rather than the
/// out-of-place kernel's int/bool -> float32 promotion. Because the refusal
/// runs first, `read_flat` is only ever called on an already-floating tag
/// here, so it can never return `Flat::Int` -- the match still has to name
/// that arm, and it does so as an internal error rather than a panic, the
/// same shape `write_into`'s dtype checks use.
fn expm1_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.expm1_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let tag = receiver.borrow().tag();
    inplace_cast_check(OP, unary_float_tag(tag), tag)?;
    let source = receiver.borrow().tensor()?.clone();
    let values = match read_flat(OP, &source, tag)? {
        Flat::Float(v) => v.into_iter().map(f64::exp_m1).collect::<Vec<f64>>(),
        Flat::Int(_) => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{OP}: torch._C shim internal error -- inplace_cast_check above \
                 should have refused every non-floating receiver (aten.rs)"
            )))
        }
    };
    let out = write_flat(OP, Flat::Float(values), source.dims().to_vec(), source.device(), tag)?;
    write_back(OP, &receiver, PyTensorBase::new(out)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::log2_(Tensor(a!) self) -> Tensor(a!)`
///
/// `log2_default`'s value (`f64::log2`, `f32` accumulation for the three
/// narrow floats -- see that function's doc comment for why `t.log()? /
/// ln(2)` is not bit-identical to upstream), written through the receiver.
/// Dtype rule is the family's: refuses rather than promotes.
fn log2_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.log2_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let tag = receiver.borrow().tag();
    inplace_cast_check(OP, unary_float_tag(tag), tag)?;
    let acc32 = tag != TorchDType::Float64;
    let source = receiver.borrow().tensor()?.clone();
    let values = match read_flat(OP, &source, tag)? {
        Flat::Float(v) => v
            .into_iter()
            .map(|x| if acc32 { (x as f32).log2() as f64 } else { x.log2() })
            .collect::<Vec<f64>>(),
        Flat::Int(_) => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{OP}: torch._C shim internal error -- inplace_cast_check above \
                 should have refused every non-floating receiver (aten.rs)"
            )))
        }
    };
    let out = write_flat(OP, Flat::Float(values), source.dims().to_vec(), source.device(), tag)?;
    write_back(OP, &receiver, PyTensorBase::new(out)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)`
///
/// `sigmoid_default`'s value and precision rule (`1/(1+exp(-x))`, `float16`/
/// `bfloat16` computed in `f32` and narrowed once -- see that function's doc
/// comment for the measurement that pins it), written through the receiver.
/// Dtype rule is the family's: refuses rather than promotes, so unlike the
/// out-of-place kernel there is no `unary_float_tag` widening left to do by
/// the time the value is computed -- `tag` and `storage` already are the
/// accumulation dtype's source.
fn sigmoid_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.sigmoid_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let tag = receiver.borrow().tag();
    inplace_cast_check(OP, unary_float_tag(tag), tag)?;
    let storage = PyDtype::new(tag).storage(OP)?;
    let acc = match storage {
        candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
        other => other,
    };
    let out = {
        let borrowed = receiver.borrow();
        borrowed
            .tensor()?
            .fast_to(acc)
            .and_then(|t| t.neg())
            .and_then(|t| t.exp())
            .and_then(|t| t + 1.0)
            .and_then(|t| t.recip())
            .and_then(|t| t.fast_to(storage))
            .map_err(|e| candle_err(OP, e))?
    };
    write_back(OP, &receiver, PyTensorBase::new(out)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::abs_(Tensor(a!) self) -> Tensor(a!)`
///
/// `abs_default`'s value, written through the receiver. **Not** a member of
/// the promoting family above: `abs` does not promote out of place either
/// (`int64.abs()` is `int64`), so the in-place form has nothing to refuse on
/// that account and the two dtype rules are the same, re-measured: every
/// dtype computes except `bool`, which raises upstream's own kernel name
/// ("abs_cpu" not implemented for 'Bool'). The integral path reuses
/// `abs_default`'s `wrapping_abs` round trip through `i64` for the same
/// reason that function gives -- candle's `abs` is a `unary_op!` whose
/// integer arms are `todo!()` and would panic rather than raise.
fn abs_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.abs_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let tag = receiver.borrow().tag();
    if tag == TorchDType::Bool {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "\"abs_cpu\" not implemented for 'Bool'",
        ));
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let out = if tag.is_floating_point() {
        receiver.borrow().tensor()?.abs().map_err(|e| candle_err(OP, e))?
    } else {
        let source = receiver.borrow().tensor()?.clone();
        let dims = source.dims().to_vec();
        let values: Vec<i64> = source
            .contiguous()
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_dtype(candle_core::DType::I64))
            .and_then(|t| t.to_vec1::<i64>())
            .map_err(|e| candle_err(OP, e))?;
        let wrapped: Vec<i64> = values
            .into_iter()
            .map(|v| match storage {
                candle_core::DType::I16 => (v as i16).wrapping_abs() as i64,
                candle_core::DType::I32 => (v as i32).wrapping_abs() as i64,
                _ => v.wrapping_abs(),
            })
            .collect();
        Tensor::from_vec(wrapped, dims, source.device())
            .and_then(|t| t.fast_to(storage))
            .map_err(|e| candle_err(OP, e))?
    };
    write_back(OP, &receiver, PyTensorBase::new(out)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::ceil_(Tensor(a!) self) -> Tensor(a!)`
///
/// `ceil_default`'s value, written through the receiver. Like `abs_` above,
/// **not** a member of the promoting family: `ceil` does not promote out of
/// place (`torch.arange(3).ceil()` is still `int64`), so an integral
/// receiver is the identity rather than a refusal -- re-measured, matching
/// `ceil_default`'s doc comment. Only `bool` refuses, with upstream's own
/// kernel name ("ceil_vml_cpu" not implemented for 'Bool'), a different
/// string from `abs_`'s because upstream reaches a different kernel.
fn ceil_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.ceil_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let tag = receiver.borrow().tag();
    if tag == TorchDType::Bool {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "\"ceil_vml_cpu\" not implemented for 'Bool'",
        ));
    }
    let out = if tag.is_floating_point() {
        receiver.borrow().tensor()?.ceil().map_err(|e| candle_err(OP, e))?
    } else {
        // Already integral: upstream hands the tensor straight back, and
        // writing the unchanged values through the receiver is the in-place
        // shape of that no-op.
        receiver.borrow().tensor()?.clone()
    };
    write_back(OP, &receiver, PyTensorBase::new(out)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// The C++ scalar-type spelling `clamp_min_`'s (and `clamp_`'s) cast-refusal
/// message names for the *destination* side, distinct from all three other
/// dtype-name tables in this file (`TorchDType::name()`, `c10_name`,
/// `scalar_type_name`) and not derivable from any of them -- each string was
/// read off a real `RuntimeError` on torch 2.13.0:
///
/// ```text
/// int32.clamp_min_(2.0)   "result type Float can't be cast to the desired output type int"
/// int64.clamp_min_(2.0)   "... long long"
/// int16.clamp_min_(2.0)   "... short"
/// uint8.clamp_min_(2.0)   "... unsigned char"
/// uint32.clamp_min_(2.0)  "... unsigned int"
/// ```
///
/// Only the dtypes this shim can actually store need an entry (`dtype.rs`'s
/// `storage()`); the rest fall back to `TorchDType::name()` rather than
/// inventing a C++ spelling for a dtype no tensor here can carry.
fn clamp_scalar_cpu_type_name(dtype: TorchDType) -> &'static str {
    use TorchDType::*;
    match dtype {
        Int16 => "short",
        Int32 => "int",
        Int64 => "long long",
        UInt8 => "unsigned char",
        UInt32 => "unsigned int",
        Bool => "bool",
        other => other.name(),
    }
}

/// The dtype refusal `clamp_min_` has, measured separately from `clamp_`'s
/// `clamp_dtype_refusals` because the two diverge on exactly one row and
/// sharing would reproduce the wrong one of them: a **boolean bound on a
/// boolean receiver**. Measured on torch 2.13.0:
///
/// ```text
/// clamp_min_(bool, 0)      "result type Long can't be cast to the desired output type bool"
/// clamp_min_(bool, 0.0)    "result type Float can't be cast to the desired output type bool"
/// clamp_min_(bool, False)  NotImplementedError: "clamp_min_scalar_cpu" not implemented for 'Bool'
/// int32.clamp_min_(2.0)    "result type Float can't be cast to the desired output type int"
/// uint8.clamp_min_(2)      OK, uint8 kept (an int bound never widens an int receiver)
/// ```
///
/// The last row is `clamp_min_`'s half of "in-place cannot promote": unlike
/// `clamp_min.default` (out of place), which promotes an integral receiver
/// when the bound is a float, this refuses instead -- the same asymmetry
/// `clamp_result_tag`'s doc comment already measured between `clamp` and
/// `clamp_`.
///
/// `min` is read from the raw arguments as well as the parsed `Scalar`,
/// for the reason `clamp_result_tag`'s own `bound_is_bool` reads them: `bool`
/// subclasses `int` in Python and `Scalar` collapses the two, but upstream's
/// kernel-name refusal fires only for an actual Python `bool`.
fn clamp_min_inplace_refusal(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    tag: TorchDType,
    min: Scalar,
) -> PyResult<()> {
    if tag == TorchDType::Bool {
        let bound_is_bool = match optional(args, kwargs, 1, "min")? {
            Some(value) if !value.is_none() => value.is_instance_of::<pyo3::types::PyBool>(),
            _ => false,
        };
        if bound_is_bool {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "\"clamp_min_scalar_cpu\" not implemented for 'Bool'",
            ));
        }
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "result type {} can't be cast to the desired output type bool",
            if min.is_int() { "Long" } else { "Float" }
        )));
    }
    if !tag.is_floating_point() && !min.is_int() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "result type Float can't be cast to the desired output type {}",
            clamp_scalar_cpu_type_name(tag)
        )));
    }
    Ok(())
}

/// `aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)`
///
/// `vits`'s wall had the out-of-place form (`clamp_min_default`, `x =
/// torch.clamp_min(...)` reassigned); this is the in-place spelling
/// (`x.clamp_min_(...)`/`torch.clamp_min_(x, ...)`), which had neither a
/// kernel nor a table entry before docs/SPELLINGS.md §9 named it. The value
/// is `clamp_values(..., min, None)`, shared with `clamp_min_default` and
/// `clamp_inplace_default`; the dtype rule is `clamp_min_inplace_refusal`
/// above, kept separate from `clamp_`'s own `clamp_dtype_refusals` because
/// they diverge on the bool-bound-on-bool-receiver row.
fn clamp_min_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.clamp_min_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let min = scalar_arg(OP, args, kwargs, 1, "min")?.ok_or_else(|| missing(OP, "min"))?;
    let tag = receiver.borrow().tag();
    clamp_min_inplace_refusal(args, kwargs, tag, min)?;
    let source = receiver.borrow().tensor()?.clone();
    let out = clamp_values(OP, &source, tag, Some(min), None)?;
    write_back(OP, &receiver, PyTensorBase::new(out)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::detach_(Tensor(a!) self) -> Tensor(a!)` -- refused by name rather
/// than approximated, and this is a judgement call recorded rather than an
/// omission.
///
/// `detach_` is not arithmetic: it rewrites the receiver's autograd metadata
/// (`requires_grad`/`is_leaf`/`grad_fn`), not its storage. Measured against
/// upstream 2.13.0:
///
/// ```text
/// leaf.detach_()          requires_grad -> False, is_leaf stays True,
///                          grad_fn stays None, returns self (identity)
/// non_leaf.detach_()      same triple, and non_leaf becomes indistinguishable
///                          from a leaf that was never part of a graph
/// view.detach_()          RuntimeError: Can't detach views in-place. Use
///                          detach() instead. ...
///                          -- true unconditionally, even for a view whose
///                          requires_grad was already False (measured: a
///                          view of a tensor that never required grad still
///                          refuses)
/// ```
///
/// Two of the three leaf-case fields are **already** upstream's answer for
/// every `TensorBase` this shim has, independent of `detach_`:
/// `is_leaf` is `property(lambda self: True)` and `grad_fn` is
/// `property(lambda self: None)` (`bootstrap.py`, docs/BACKWARD2.md §1.4 W5/
/// W6, neither landed). So the only field `detach_` could actually change
/// here is `requires_grad`, and *that* would be a one-line `set_requires_grad`
/// call -- except for the view refusal, which does not depend on
/// `requires_grad` at all and fires before either flag is touched.
///
/// **This shim cannot tell a view from a non-view.** `PyTensorBase`
/// (tensor.rs) carries shape, dtype and a candle `Tensor`; it has no
/// `_base`/`is_view` slot recording which op produced it, so there is no
/// field a kernel could read to decide which of the two upstream branches
/// applies. Always taking the leaf branch (`requires_grad = false`, return
/// self) would compute a value for every receiver, including the ones
/// upstream refuses -- `y = x.view(-1); y.detach_()` would silently succeed
/// here and raise upstream. That is the silent-divergence direction this
/// file refuses everywhere else (`inplace_cast_check`'s doc comment, the
/// `add_`/`sub_` narrowing fix, `write_into`'s own tag/shape checks), so it
/// is refused here too rather than accepted for the sake of landing a
/// sixteenth kernel. The out-of-place `detach()` this shim already
/// implements is unaffected and is upstream's own suggested replacement.
fn detach_inplace_refusal(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.detach_.default";
    // Argument shape is checked before the deliberate refusal, the same
    // order every other kernel in this file uses -- a caller who passed the
    // wrong thing for `self` should see that error, not this one.
    let _receiver = tensor_receiver(OP, args, kwargs)?;
    let _ = py;
    Err(not_implemented(
        "aten.detach_.default: torch._C shim refuses this op by name rather than \
         guessing at it (docs/INPLACE.md). Upstream sets requires_grad=False on a \
         leaf receiver (is_leaf and grad_fn are already True/None for every tensor \
         this shim has), but raises \"Can't detach views in-place. Use detach() \
         instead.\" unconditionally for a view -- before touching either flag, and \
         regardless of the view's requires_grad. This shim's TensorBase tracks no \
         base/view relationship, so it cannot tell which case a given receiver is \
         in; always taking the leaf branch would compute where upstream sometimes \
         raises. Call the out-of-place detach() instead.",
    ))
}

// ---------------------------------------------------------------------------
// The two RNG ops
//
// docs/RNG.md is the standing decision behind these: candle's CPU backend
// refuses to be seeded at all, so its `rand_uniform`/`rand_normal` cannot be
// used here even in principle, and torch's own CPU generator is ported into
// `rng.rs` instead. What is left for this file is the part that depends on
// the *tensor* rather than on the stream -- which accumulate type the kernel
// runs in, which of `normal_`'s two paths a given size and layout takes, and
// where the narrowing cast happens relative to `uniform_`'s upper-bound clamp.
// Getting any of those wrong reproduces the stream perfectly and still
// produces different numbers.
// ---------------------------------------------------------------------------

/// The floating dtype an RNG op is allowed to fill, with its candle storage.
///
/// `AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, ...)` is the whole
/// permitted set for both kernels; an integer tensor reaches a different op
/// upstream (`random_`), so accepting one here would be implementing something
/// else under this name.
fn rng_float_dtype(op: &str, tag: TorchDType) -> PyResult<candle_core::DType> {
    match tag {
        TorchDType::Float64 => Ok(candle_core::DType::F64),
        TorchDType::Float32 => Ok(candle_core::DType::F32),
        TorchDType::Float16 => Ok(candle_core::DType::F16),
        TorchDType::BFloat16 => Ok(candle_core::DType::BF16),
        other => Err(not_implemented(format!(
            "{op}: not implemented in torch._C shim for torch.{} -- upstream \
             dispatches this op over floating dtypes only, and an integer \
             tensor reaches `random_`, a different op",
            other.name()
        ))),
    }
}

/// The `Generator? generator=None` tail both schemas carry.
///
/// There is exactly one generator here -- the process-wide default that
/// `torch.default_generator` names -- so a *different* generator is refused by
/// name rather than silently served from the default stream, which would make
/// `torch.Generator().manual_seed(0)` look like it worked while sharing state
/// with everything else. `None` is the common case and never even arrives:
/// the overload resolver drops arguments equal to their schema default.
fn generator_arg(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<()> {
    let Some(value) = optional(args, kwargs, index, name)? else {
        return Ok(());
    };
    if value.is_none() {
        return Ok(());
    }
    if value
        .getattr("_shim_is_default_generator")
        .is_ok_and(|flag| flag.is_truthy().unwrap_or(false))
    {
        return Ok(());
    }
    Err(not_implemented(format!(
        "{op}: only torch.default_generator is implemented in torch._C shim; \
         a separate torch.Generator has no state of its own here"
    )))
}

fn float_arg(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
    fallback: f64,
) -> PyResult<f64> {
    match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => value.extract::<f64>(),
        _ => Ok(fallback),
    }
}

/// Shape, device, dtype and layout of an in-place RNG op's receiver.
struct RngTarget {
    tag: TorchDType,
    storage: candle_core::DType,
    shape: candle_core::Shape,
    device: Device,
    numel: usize,
    contiguous: bool,
}

fn rng_target(op: &str, receiver: &Bound<'_, PyTensorBase>) -> PyResult<RngTarget> {
    let borrowed = receiver.borrow();
    let tag = borrowed.tag();
    Ok(RngTarget {
        tag,
        storage: rng_float_dtype(op, tag)?,
        shape: borrowed.tensor()?.shape().clone(),
        device: borrowed.tensor()?.device().clone(),
        numel: borrowed.tensor()?.elem_count(),
        contiguous: borrowed.tensor()?.is_contiguous(),
    })
}

/// Round a value through the storage dtype and back, so it can be compared in
/// the accumulate type. For `float32` this is the identity; for `float16` and
/// `bfloat16` it is the narrowing `static_cast<scalar_t>` that upstream's
/// clamp is written in terms of. candle's `to_dtype` rounds to nearest even,
/// which is what the C++ cast does.
fn narrow_roundtrip_f32(op: &str, value: f32, storage: candle_core::DType, device: &Device) -> PyResult<f32> {
    if storage == candle_core::DType::F32 {
        return Ok(value);
    }
    Tensor::from_vec(vec![value], 1, device)
        .and_then(|t| t.fast_to(storage))
        .and_then(|t| t.to_dtype(candle_core::DType::F32))
        .and_then(|t| t.to_vec1::<f32>())
        .map(|values| values[0])
        .map_err(|e| candle_err(op, e))
}

/// `aten::uniform_(Tensor(a!) self, float from=0., float to=1., *,
///                 Generator? generator=None) -> Tensor(a!)`
///
/// This is the sixth wall on the way to `from_config` (docs/TENSORBASE.md §7):
/// `nn.init.kaiming_uniform_` ends in `tensor.uniform_(-bound, bound)`, so no
/// `nn.Linear` exists until it does.
///
/// Two things here are not the RNG and are still part of the answer.
///
/// *The accumulate type follows `opmath_type<scalar_t>`, not the dtype.* A
/// `float16` tensor draws **one** 32-bit word per element and transforms it in
/// float; only `float64` draws two. Reading the dtype instead would consume
/// the stream at the wrong rate and desynchronise everything after it.
///
/// *The upper bound is enforced after the cast, not before.* Upstream's
/// kernel ends `return value == to_scalar ? from_scalar : value;` -- because
/// narrowing a float that is just under `to` can round it *up to* `to`, and
/// `uniform_` promises a half-open range. On `float16` with `to=1.0` that is
/// roughly one draw in 4096, so a shim without the clamp passes casual
/// inspection and fails the golden harness's range check.
fn uniform_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.uniform_.default";

    let receiver = tensor_receiver(OP, args, kwargs)?;
    let from = float_arg(args, kwargs, 1, "from", 0.0)?;
    let to = float_arg(args, kwargs, 2, "to", 1.0)?;
    generator_arg(OP, args, kwargs, 3, "generator")?;
    let target = rng_target(OP, &receiver)?;

    // torch's own check, message included.
    if !(from <= to) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "uniform_ expects to return a [from, to) range, but found from={from} > to={to}"
        )));
    }

    let mut gen = crate::rng::default_generator();
    let replacement = if target.storage == candle_core::DType::F64 {
        let mut values = crate::rng::uniform_fill_f64(&mut gen, target.numel, from, to);
        for value in values.iter_mut() {
            if *value == to {
                *value = from;
            }
        }
        Tensor::from_vec(values, target.shape, &target.device).map_err(|e| candle_err(OP, e))?
    } else {
        let (from_f32, to_f32) = (from as f32, to as f32);
        let values = crate::rng::uniform_fill_f32(&mut gen, target.numel, from_f32, to_f32);
        // The cast first, then the comparison: `to_scalar` is
        // `static_cast<scalar_t>(to_)`, and both sides of `==` are in
        // `scalar_t`. Round-tripping through the storage dtype is how that
        // comparison is made without hand-rolling half-precision rounding.
        let narrowed = Tensor::from_vec(values, target.numel, &target.device)
            .and_then(|t| t.to_dtype(target.storage))
            .and_then(|t| t.to_dtype(candle_core::DType::F32))
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| candle_err(OP, e))?;
        let to_scalar = narrow_roundtrip_f32(OP, to_f32, target.storage, &target.device)?;
        let clamped: Vec<f32> = narrowed
            .into_iter()
            .map(|v| if v == to_scalar { from_f32 } else { v })
            .collect();
        Tensor::from_vec(clamped, target.shape, &target.device)
            .and_then(|t| t.to_dtype(target.storage))
            .map_err(|e| candle_err(OP, e))?
    };

    drop(gen);
    write_back(OP, &receiver, PyTensorBase::new(replacement)?)?;
    let _ = (py, target.tag);
    Ok(receiver.into_any().unbind())
}

/// `aten::normal_(Tensor(a!) self, float mean=0., float std=1., *,
///                Generator? generator=None) -> Tensor(a!)`
///
/// The op where the *shape of the kernel* is observable output. `normal_kernel`
/// branches on `size >= 16 && self.is_contiguous()`, and the two sides share
/// nothing:
///
///   * **Path B** (small or strided) runs Box-Muller one element at a time in
///     `double` -- for every dtype, `float16` included -- and leaves the
///     unused half of each pair *cached on the generator*, so an odd-sized
///     `normal_` changes what the next one returns.
///   * **Path A** fills the whole buffer with uniforms first and then applies
///     Box-Muller over it in blocks of 16, pairing element `j` with `j+8`.
///     When the size is not a multiple of 16 it steps back to `size - 16` and
///     redraws those sixteen *over values it already wrote*.
///
/// So `n=15` and `n=16` produce entirely different sequences from one seed,
/// and `n=17` differs from `n=16` in its first element too. docs/RNG.md §1.3
/// measured all three; the harness cases below them are the regression.
fn normal_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.normal_.default";

    let receiver = tensor_receiver(OP, args, kwargs)?;
    let mean = float_arg(args, kwargs, 1, "mean", 0.0)?;
    let std = float_arg(args, kwargs, 2, "std", 1.0)?;
    generator_arg(OP, args, kwargs, 3, "generator")?;
    let target = rng_target(OP, &receiver)?;

    if !(std >= 0.0) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "normal expects std >= 0.0, but found std {std}"
        )));
    }

    let mut gen = crate::rng::default_generator();
    let values_f64: Option<Vec<f64>>;
    let values_f32: Option<Vec<f32>>;

    if target.numel >= 16 && target.contiguous {
        match target.storage {
            candle_core::DType::F64 => {
                values_f64 = Some(crate::rng::normal_fill_f64(&mut gen, target.numel, mean, std));
                values_f32 = None;
            }
            candle_core::DType::F32 => {
                values_f64 = None;
                values_f32 = Some(crate::rng::normal_fill_f32(
                    &mut gen,
                    target.numel,
                    mean as f32,
                    std as f32,
                ));
            }
            // float16 / bfloat16 -- the stack-buffer branch.
            _ => {
                values_f64 = None;
                values_f32 = Some(crate::rng::normal_fill_reduced(
                    &mut gen,
                    target.numel,
                    mean as f32,
                    std as f32,
                ));
            }
        }
    } else {
        // Path B is `double` for every dtype; the narrowing happens once, at
        // the end, in `to_dtype`.
        values_f64 = Some(crate::rng::normal_serial(&mut gen, target.numel, mean, std));
        values_f32 = None;
    }
    drop(gen);

    let filled = match (values_f64, values_f32) {
        (Some(values), _) => Tensor::from_vec(values, target.shape, &target.device),
        (_, Some(values)) => Tensor::from_vec(values, target.shape, &target.device),
        _ => unreachable!("one of the two accumulate types is always produced"),
    }
    .and_then(|t| t.to_dtype(target.storage))
    .map_err(|e| candle_err(OP, e))?;

    write_back(OP, &receiver, PyTensorBase::new(filled)?)?;
    let _ = (py, target.tag);
    Ok(receiver.into_any().unbind())
}

/// `aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *,
///                         Generator? generator=None) -> Tensor(a!)`
///
/// The primitive under **training mode** (docs/TRAIN.md). Two callers, and
/// they are not the same caller: `nn.Dropout`'s composite decomposes onto it
/// (docs/TRAIN.md §1), and DeBERTa's `XDropout` reaches for it directly --
/// `transformers/models/sew_d/modeling_sew_d.py:229` is
/// `(1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)`,
/// because it needs the mask itself and not just the masked tensor.
///
/// **The accumulate type is `double` for every dtype, and that is the trap.**
/// `uniform_` follows `opmath_type<scalar_t>`, so a `float16` tensor draws one
/// *32-bit* word per element there (see `uniform_inplace` above). `bernoulli_`
/// does not: its `bernoulli_distribution<double>` holds a
/// `uniform_real_distribution<double>` whose `operator()` takes
/// `generator->random64()` no matter what `scalar_t` is. Reading the dtype
/// here instead would consume the stream at half the rate on the reduced
/// dtypes and desynchronise every draw after it -- while producing values that
/// look perfectly plausible. Measured on torch 2.13.0 for all eight accepted
/// dtypes: `bernoulli_(p)` is elementwise `uniform_(0,1) < p` in `float64`,
/// and leaves the generator in the same state as that `float64` fill.
///
/// **`p == 0` and `p == 1` still draw.** Measured: the `rand(2)` after a
/// `bernoulli_(0.0)` over six elements matches the one after `bernoulli_(1.0)`
/// and `bernoulli_(0.5)`, and none of them matches no draw at all. Upstream
/// has no short-circuit -- the comparison is per element, inside the kernel --
/// so a short-circuit here would be invisible in the values it returns and
/// wrong in everything that comes after it.
///
/// The dtype set is `AT_DISPATCH_ALL_TYPES_AND3(Bool, BFloat16, Half)`, which
/// is narrower than what this shim can store: `uint32` and `float8_e4m3fn`
/// both refuse upstream with `"bernoulli_scalar_cpu_" not implemented for
/// '...'`, so they refuse here with the same message rather than being served
/// because candle happens to have the storage.
fn bernoulli_inplace_float(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.bernoulli_.float";

    let receiver = tensor_receiver(OP, args, kwargs)?;
    let p = float_arg(args, kwargs, 1, "p", 0.5)?;
    generator_arg(OP, args, kwargs, 2, "generator")?;

    // torch's own check, message included. Written as `!(0 <= p <= 1)` rather
    // than `p < 0 || p > 1` so that `nan` is refused -- upstream's
    // `TORCH_CHECK(0 <= p && p <= 1, ...)` refuses it and reports `p=nan`.
    if !(p >= 0.0 && p <= 1.0) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "bernoulli_ expects p to be in [0, 1], but got p={p}"
        )));
    }

    let tag = receiver.borrow().tag();
    match tag {
        TorchDType::Float64
        | TorchDType::Float32
        | TorchDType::Float16
        | TorchDType::BFloat16
        | TorchDType::Int64
        | TorchDType::Int32
        | TorchDType::Int16
        | TorchDType::Int8
        | TorchDType::UInt8
        | TorchDType::Bool => {}
        other => {
            return Err(not_implemented(format!(
                "\"bernoulli_scalar_cpu_\" not implemented for '{}'",
                other.cpp_name()
            )))
        }
    }
    let storage = PyDtype::new(tag).storage(OP)?;

    let (shape, device, numel) = {
        let borrowed = receiver.borrow();
        let tensor = borrowed.tensor()?;
        (
            tensor.shape().clone(),
            tensor.device().clone(),
            tensor.elem_count(),
        )
    };

    let mut gen = crate::rng::default_generator();
    let draws = crate::rng::uniform_fill_f64(&mut gen, numel, 0.0, 1.0);
    drop(gen);

    // `uniform(generator) < p`, in `double`, exactly as
    // `bernoulli_distribution::operator()` writes it -- strictly less-than, so
    // `p == 0` is all zeros (a draw is never negative) and `p == 1` is all ones
    // (a draw is never 1.0, the range is half-open).
    let values: Vec<f64> = draws
        .into_iter()
        .map(|u| if u < p { 1.0 } else { 0.0 })
        .collect();
    let filled = Tensor::from_vec(values, shape, &device)
        .and_then(|t| t.to_dtype(storage))
        .map_err(|e| candle_err(OP, e))?;

    write_back(OP, &receiver, tagged(filled, tag)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

// ---------------------------------------------------------------------------
// The eight ops `do_sample=True` stops on
//
// docs/GAP.md §4 predicted ten; the coordinating session re-measured a real
// transformers Llama against `_aten_implemented()` and found eight still
// missing. docs/SAMPLING.md records what each one turned out to be.
//
// Seven of the eight are ordinary kernels. `multinomial` is not: it is the only
// op in this file that *draws*, and a sampled token is only reproducible if it
// consumes torch's stream in torch's order. That made "read the upstream
// algorithm" a requirement rather than a nicety, and reading it produced two
// facts guessing would have missed -- see `multinomial_default`.
//
// One thing is deliberately *not* reproduced, and it is worth naming here
// rather than only at the call site: `topk`'s `sorted=False` order. Upstream
// leaves it unspecified and its CPU kernel returns a partition artefact
// (measured: `k=3` of an 8-element tensor answers `[7, 6, 0]` where `sorted=True`
// answers `[6, 7, 0]`). This shim always returns the sorted order, which is a
// legal answer to the same question; the harness compares those cases as
// multisets so it cannot accidentally pin an order upstream does not promise.
// ---------------------------------------------------------------------------

/// A tensor's elements read into the one representation that can hold them
/// exactly: `f64` for the floating dtypes (widening f32/f16/bf16 and back is
/// lossless), `i64` for everything else.
///
/// Reading floats as `i64` or `int64` as `f64` would both lose information the
/// ops below depend on -- `sort` on `int64` values beyond 2^53 would start
/// declaring ties that are not ties.
enum Flat {
    Float(Vec<f64>),
    Int(Vec<i64>),
}

impl Flat {
    fn empty_like(&self, n: usize) -> Flat {
        match self {
            Flat::Float(_) => Flat::Float(vec![0.0; n]),
            Flat::Int(_) => Flat::Int(vec![0; n]),
        }
    }
}

fn read_flat(op: &str, tensor: &Tensor, tag: TorchDType) -> PyResult<Flat> {
    let flat = tensor
        .contiguous()
        .and_then(|t| t.flatten_all())
        .map_err(|e| candle_err(op, e))?;
    if tag.is_floating_point() {
        Ok(Flat::Float(
            widen_f64(&flat)
                .and_then(|t| t.to_vec1::<f64>())
                .map_err(|e| candle_err(op, e))?,
        ))
    } else {
        Ok(Flat::Int(
            flat.to_dtype(candle_core::DType::I64)
                .and_then(|t| t.to_vec1::<i64>())
                .map_err(|e| candle_err(op, e))?,
        ))
    }
}

fn write_flat(
    op: &str,
    values: Flat,
    dims: Vec<usize>,
    device: &Device,
    tag: TorchDType,
) -> PyResult<Tensor> {
    let storage = PyDtype::new(tag).storage(op)?;
    match values {
        Flat::Float(v) => Tensor::from_vec(v, dims, device),
        Flat::Int(v) => Tensor::from_vec(v, dims, device),
    }
    .and_then(|t| t.fast_to(storage))
    .map_err(|e| candle_err(op, e))
}

/// torch's ordering of floats, which is not `f64`'s.
///
/// IEEE says every comparison against NaN is false; torch sorts NaN as
/// *greatest*, so ascending puts it last and descending puts it first, and
/// `topk(largest=True)` picks it before `+inf` (all three measured). A
/// `partial_cmp().unwrap()` here would panic on the first NaN instead.
fn cmp_torch_f64(a: f64, b: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// The (values, indices) result `sort` and `topk` share, in the layout
/// upstream produces.
struct Ordered {
    values: Flat,
    indices: Vec<i64>,
    dims: Vec<usize>,
}

/// Reorder every lane along `dim`, keeping `keep` of each.
///
/// **The sort is stable, and for `sort` that is a measurement, not a
/// convenience.** Upstream's CPU `sort` keeps the original index order among
/// equal values in *both* directions: `[3,1,3,1,2,3]` descending answers
/// indices `[0,2,5,4,1,3]`, not the reverse of the ascending run, and an
/// 80-element all-ties tensor comes back as `0..79`. An unstable
/// `sort_unstable_by` here would be a silent behaviour change with no failing
/// test anywhere near it.
///
/// **`topk` is a different story and the difference is recorded rather than
/// hidden.** Upstream's `topk` is a partial selection, not a sort, and its tie
/// order is stable only sometimes: on the same `[3,1,3,1,2,3]`, `k=3` answers
/// `[0,2,5]` (stable, and this shim agrees) but `k=6` answers `[0,2,5,4,3,1]`
/// -- the two 1.0s come back reversed. Reproducing that would mean
/// transcribing the partition, and upstream promises nothing about it. This
/// shim answers `[0,2,5,4,1,3]` there. It matters for nothing measured: the
/// `top_k` warper reads only `values[..., -1]`, and `multinomial`'s
/// no-replacement path feeds `topk` continuous ratios where ties do not occur.
/// docs/SAMPLING.md §4 has the measurement; the golden cases keep `topk`'s
/// index comparison to tie-free inputs and compare the tied ones by value.
fn order_along(
    op: &str,
    input: &PyTensorBase,
    dim: usize,
    descending: bool,
    keep: Option<usize>,
) -> PyResult<Ordered> {
    let dims = input.tensor()?.dims().to_vec();
    // A 0-d tensor is one lane of one element; torch answers `sort`/`topk` on
    // one with a 0-d value and a 0-d index of 0, rather than refusing.
    let (outer, n, inner) = if dims.is_empty() {
        (1usize, 1usize, 1usize)
    } else {
        (
            dims[..dim].iter().product::<usize>(),
            dims[dim],
            dims[dim + 1..].iter().product::<usize>(),
        )
    };
    let keep = keep.unwrap_or(n);

    let source = read_flat(op, input.tensor()?, input.tag())?;
    let mut values = source.empty_like(outer * keep * inner);
    let mut indices = vec![0i64; outer * keep * inner];

    let mut order: Vec<usize> = Vec::with_capacity(n);
    for o in 0..outer {
        for i in 0..inner {
            let at = |j: usize| o * n * inner + j * inner + i;
            order.clear();
            order.extend(0..n);
            match &source {
                Flat::Float(v) => order.sort_by(|&a, &b| {
                    let (x, y) = (v[at(a)], v[at(b)]);
                    if descending {
                        cmp_torch_f64(y, x)
                    } else {
                        cmp_torch_f64(x, y)
                    }
                }),
                Flat::Int(v) => order.sort_by(|&a, &b| {
                    let (x, y) = (v[at(a)], v[at(b)]);
                    if descending {
                        y.cmp(&x)
                    } else {
                        x.cmp(&y)
                    }
                }),
            }
            for (slot, &j) in order.iter().take(keep).enumerate() {
                let dst = o * keep * inner + slot * inner + i;
                indices[dst] = j as i64;
                match (&source, &mut values) {
                    (Flat::Float(src), Flat::Float(out)) => out[dst] = src[at(j)],
                    (Flat::Int(src), Flat::Int(out)) => out[dst] = src[at(j)],
                    _ => unreachable!("values was allocated from source's own variant"),
                }
            }
        }
    }

    let mut out_dims = dims;
    if !out_dims.is_empty() {
        out_dims[dim] = keep;
    }
    Ok(Ordered {
        values,
        indices,
        dims: out_dims,
    })
}

static SORT_RESULT: std::sync::OnceLock<Py<PyAny>> = std::sync::OnceLock::new();
static TOPK_RESULT: std::sync::OnceLock<Py<PyAny>> = std::sync::OnceLock::new();

/// The `(values, indices)` namedtuple `sort` and `topk` answer with, built the
/// same way `max.dim`'s is and cached for the same reason.
fn values_indices_type(
    py: Python<'_>,
    cell: &'static std::sync::OnceLock<Py<PyAny>>,
    name: &str,
) -> PyResult<&'static Py<PyAny>> {
    if let Some(cached) = cell.get() {
        return Ok(cached);
    }
    let namedtuple = py
        .import("collections")?
        .getattr("namedtuple")?
        .call1((name, ("values", "indices")))?
        .unbind();
    let _ = cell.set(namedtuple);
    Ok(cell.get().expect("just set"))
}

fn finish_ordered(
    py: Python<'_>,
    op: &str,
    cell: &'static std::sync::OnceLock<Py<PyAny>>,
    name: &str,
    ordered: Ordered,
    tag: TorchDType,
    device: &Device,
) -> PyResult<Py<PyAny>> {
    let values = write_flat(op, ordered.values, ordered.dims.clone(), device, tag)?;
    let indices = Tensor::from_vec(ordered.indices, ordered.dims, device)
        .map_err(|e| candle_err(op, e))?;
    // Promoted here rather than at the dispatcher's exit, for the reason
    // `max.dim` gives: the pair leaves inside a namedtuple, which `promote`
    // does not look into.
    let pair = (
        crate::tensor::promote(py, finish(py, values, tag)?)?,
        crate::tensor::promote(py, finish(py, indices, TorchDType::Int64)?)?,
    );
    Ok(values_indices_type(py, cell, name)?
        .bind(py)
        .call1(pair)?
        .unbind())
}

/// `aten::sort(Tensor self, int dim=-1, bool descending=False)
///     -> (Tensor values, Tensor indices)`
///
/// Reached by `TopPLogitsWarper`, which sorts the whole vocabulary row before
/// taking a cumulative sum of the softmax over it.
fn sort_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.sort.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let dim = normalise_dim(OP, dim_arg(args, kwargs, 1, "dim")?.unwrap_or(-1), rank)?;
    let descending = bool_arg(args, kwargs, 2, "descending")?.unwrap_or(false);
    let ordered = order_along(OP, &input, dim, descending, None)?;
    let device = input.tensor()?.device().clone();
    finish_ordered(py, OP, &SORT_RESULT, "sort", ordered, input.tag(), &device)
}

/// `aten::topk(Tensor self, SymInt k, int dim=-1, bool largest=True,
///             bool sorted=True) -> (Tensor values, Tensor indices)`
///
/// Reached by `TopKLogitsWarper`, which only ever reads `values[..., -1]` --
/// the k-th largest logit, used as a threshold. The indices still have to be
/// right, because the same op is how `multinomial` picks more than one sample
/// without replacement.
fn topk_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.topk.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let k = int_arg(args, kwargs, 1, "k")?.ok_or_else(|| missing(OP, "k"))?;
    let dim = normalise_dim(OP, dim_arg(args, kwargs, 2, "dim")?.unwrap_or(-1), rank)?;
    let largest = bool_arg(args, kwargs, 3, "largest")?.unwrap_or(true);
    // Read and discarded: see the section note. `sorted=False` licenses an
    // unspecified order and this shim answers with the sorted one, which is
    // within that licence.
    let _sorted = bool_arg(args, kwargs, 4, "sorted")?.unwrap_or(true);

    let extent = if rank == 0 { 1 } else { input.tensor()?.dims()[dim] };
    if k < 0 || k as usize > extent {
        // torch's own wording, and torch's own conflation of "negative" with
        // "too large" -- `topk(x, -1)` gives this same message upstream.
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "selected index k out of range",
        ));
    }
    let ordered = order_along(OP, &input, dim, largest, Some(k as usize))?;
    let device = input.tensor()?.device().clone();
    finish_ordered(py, OP, &TOPK_RESULT, "topk", ordered, input.tag(), &device)
}

/// `aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)`
///
/// The one thing to get right is that a dimension whose size is not 1 is a
/// **no-op, not an error** -- `squeeze(1)` on a `(1,3,1,2)` tensor answers
/// `(1,3,1,2)` unchanged (measured). Refusing there would break the generation
/// loop's `next_tokens.squeeze(1)` the moment a batch had one row.
fn squeeze_dim(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.squeeze.dim";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let dim = normalise_dim(
        OP,
        dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?,
        rank,
    )?;
    // A 0-d tensor has nothing to remove; `normalise_dim` already accepted
    // dim 0 and -1 on it, as torch does.
    if rank == 0 {
        return finish(py, input.tensor()?.clone(), input.tag());
    }
    // candle's `squeeze` carries the same "size != 1 is a no-op" rule and does
    // it without a contiguous copy, so a strided view survives.
    let out = input
        .tensor()?
        .squeeze(dim)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// `aten::squeeze.default(Tensor(a) self) -> Tensor(a)`
///
/// docs/DEMAND.md §0.1 rank 2: `squeeze` is declared in both
/// `overloads.json` and `methods.json` with three overloads --
/// `squeeze()`, `.dim`, `.dims` -- but the dispatch `match` above only had
/// an arm for `.dim`. The no-arg overload looked present in the name tables
/// and was not reachable. `mbart`'s `shift_tokens_right` calls plain
/// `squeeze()`.
///
/// Removes **every** axis of size 1, not just one -- measured:
/// `squeeze(zeros(1,3,1,2)).shape == (3, 2)`. Squeezed from the last axis
/// down so a removal never shifts the index of an axis not yet visited (the
/// same convention `.dims` below uses). A 0-d tensor has nothing to remove
/// and is returned unchanged.
fn squeeze_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.squeeze.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let t = input.tensor()?;
    let dims: Vec<usize> = t.dims().to_vec();
    let mut out = t.clone();
    for dim in (0..dims.len()).rev() {
        if dims[dim] == 1 {
            out = out.squeeze(dim).map_err(|e| candle_err(OP, e))?;
        }
    }
    finish(py, out, input.tag())
}

/// `aten::squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)`
///
/// docs/DEMAND.md §0.1 rank 2 -- the other hole beside `squeeze.default`,
/// same missing dispatch arm, checked while landing it.
///
/// Each named axis of size 1 is removed; a named axis whose size is not 1
/// is a no-op, the same rule `.dim` carries for a single axis (measured:
/// `squeeze(x, dim=(0,1))` on a `(1,3,1,2)` tensor, where axis 1 has size
/// 3, removes only axis 0 and answers `(3,1,2)`). **An empty list is a
/// no-op**, not "every axis" -- unlike `norm.ScalarOpt_dim` and
/// `linalg_vector_norm` above, where an empty `dim` list means the
/// opposite. A repeated axis -- even a positive/negative pair that
/// normalises to the same one, `dim=(0,-4)` on a 4-d tensor -- raises
/// upstream's own `dim {d} appears multiple times in the list of dims`,
/// checked **before** any size is looked at (measured: `squeeze(x,
/// dim=(1,1))` raises even though axis 1 is not size 1, and a 0-d tensor's
/// `dim=(0,0)` raises too). A 0-d tensor otherwise accepts `dim=(0,)` or
/// `dim=(-1,)` (torch treats a 0-d tensor as one-dimensional for indexing)
/// and answers itself unchanged.
fn squeeze_dims(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.squeeze.dims";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let raw = shape_arg(OP, args, kwargs, 1, "dim")?;
    let dims: Vec<usize> = raw
        .iter()
        .map(|&d| normalise_dim(OP, d, rank))
        .collect::<PyResult<Vec<_>>>()?;
    refuse_duplicate_dims(&dims)?;
    if rank == 0 {
        return finish(py, input.tensor()?.clone(), input.tag());
    }
    let t = input.tensor()?;
    let shape = t.dims();
    let mut to_remove: Vec<usize> = dims.into_iter().filter(|&d| shape[d] == 1).collect();
    to_remove.sort_unstable();
    let mut out = t.clone();
    for dim in to_remove.into_iter().rev() {
        out = out.squeeze(dim).map_err(|e| candle_err(OP, e))?;
    }
    finish(py, out, input.tag())
}

/// `aten::split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0)
///     -> Tensor(a)[]`
///
/// The only op in this file that answers with a **list**, which is a fact about
/// the call and not a detail: GPT-2's attention is `c_attn(x).split(n, dim=2)`,
/// so the three-way unpack on the Python side is the op's whole purpose.
/// `promote` at the dispatcher's exit does not look inside a list, so each
/// chunk is promoted here -- the same reason `max.dim` promotes its own pair.
///
/// Measured against torch 2.13.0, including the parts that read like edge cases
/// and are not:
///
///   * the **last chunk is short**, not padded: `split(arange(10), 3)` is four
///     chunks sized 3, 3, 3, 1.
///   * a `split_size` larger than the dimension gives **one** chunk, the whole
///     tensor -- not an error.
///   * `split_size == 0` is an error *unless* the dimension is empty, and the
///     two refusals have different wording ("split_size can only be 0 if
///     dimension size is 0, but got dimension size of 10" vs "split expects
///     split_size be non-negative, but got split_size=-1").
///   * an **empty dimension** gives one empty chunk for any `split_size`,
///     including 0.
///   * a 0-d tensor raises rather than answering with itself.
///
/// **Not reproduced: aliasing.** Upstream's chunks are views -- writing to
/// `split(x, 3)[0][0]` changes `x`. candle's `narrow` gives a view too, but
/// whether a write through this shim's `TensorBase` reaches the source is the
/// same unanswered question `aten.slice.Tensor` already has, and this op does
/// not settle it.
fn split_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.split.Tensor";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    if rank == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "split expects at least a 1-dimensional tensor",
        ));
    }
    let split_size = int_arg(args, kwargs, 1, "split_size")?.ok_or_else(|| missing(OP, "split_size"))?;
    let dim = normalise_dim(OP, dim_arg(args, kwargs, 2, "dim")?.unwrap_or(0), rank)?;
    let extent = input.tensor()?.dims()[dim];

    if split_size < 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "split expects split_size be non-negative, but got split_size={split_size}"
        )));
    }
    let split_size = split_size as usize;
    if split_size == 0 && extent != 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "split_size can only be 0 if dimension size is 0, but got dimension size of {extent}"
        )));
    }

    let mut chunks: Vec<Py<PyAny>> = Vec::new();
    if extent == 0 {
        // One empty chunk, whatever the split size -- measured for both
        // `split_size == 0` and `split_size == 3`.
        chunks.push(crate::tensor::promote(
            py,
            finish(py, input.tensor()?.clone(), input.tag())?,
        )?);
    } else {
        let mut start = 0usize;
        while start < extent {
            let length = split_size.min(extent - start);
            let chunk = input
                .tensor()?
                .narrow(dim, start, length)
                .map_err(|e| candle_err(OP, e))?;
            chunks.push(crate::tensor::promote(py, finish(py, chunk, input.tag())?)?);
            start += length;
        }
    }
    Ok(PyList::new(py, chunks)?.into_any().unbind())
}

/// `aten::split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int
///     dim=0) -> Tensor(a)[]`
///
/// `split.Tensor` with the chunk sizes spelled out individually rather than
/// as one repeated size -- the spelling `gpt_bigcode` (docs/TAIL.md) reaches
/// for `c_attn(x).split((embed_dim, kv_dim, kv_dim), dim=2)`, an *uneven*
/// three-way unpack (query gets the full embedding width, key and value share
/// a narrower one under multi-query attention) that `split.Tensor`'s single
/// repeated size cannot express. `methods.json` already spells this op to a
/// single kernel key (docs/SPELLINGS.md §4); only the kernel was missing.
///
/// Measured against torch 2.13.0:
///
///   * sizes must be **non-negative** and **sum exactly** to the dimension's
///     extent -- unlike `split.Tensor`, there is no "last chunk is short"
///     leniency here, because the caller already spelled out every length.
///     Both refusals are reproduced (`split_with_sizes expects split_sizes
///     have only non-negative entries...` / `...expects split_sizes to sum
///     exactly to N...`).
///   * an **individual size of 0 is fine** even when the dimension itself is
///     not empty, as long as the sizes still sum correctly -- `split(arange(10),
///     [0, 10], 0)` gives an empty first chunk and the whole tensor as the
///     second, both measured.
///   * a **0-d tensor raises**, the same refusal `split.Tensor` gives and the
///     same wording ("split expects at least a 1-dimensional tensor") --
///     upstream's message does not distinguish the two overloads.
fn split_with_sizes(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.split_with_sizes.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    if rank == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "split expects at least a 1-dimensional tensor",
        ));
    }
    let sizes_raw: Vec<i64> = required(OP, args, kwargs, 1, "split_sizes")?.extract()?;
    let dim = normalise_dim(OP, dim_arg(args, kwargs, 2, "dim")?.unwrap_or(0), rank)?;
    let extent = input.tensor()?.dims()[dim];

    if sizes_raw.iter().any(|&s| s < 0) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "split_with_sizes expects split_sizes have only non-negative entries, but got split_sizes={sizes_raw:?}"
        )));
    }
    let total: i64 = sizes_raw.iter().sum();
    if total as usize != extent {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "split_with_sizes expects split_sizes to sum exactly to {extent} (input tensor's \
             size at dimension {dim}), but got split_sizes={sizes_raw:?}"
        )));
    }

    let mut chunks: Vec<Py<PyAny>> = Vec::new();
    let mut start = 0usize;
    for &size in &sizes_raw {
        let length = size as usize;
        let chunk = input
            .tensor()?
            .narrow(dim, start, length)
            .map_err(|e| candle_err(OP, e))?;
        chunks.push(crate::tensor::promote(py, finish(py, chunk, input.tag())?)?);
        start += length;
    }
    Ok(PyList::new(py, chunks)?.into_any().unbind())
}

// ---------------------------------------------------------------------------
// mamba / mixtral -- the last two of the 20 measured architectures
// (docs/OPS4.md) with anything unimplemented. Traced with a real
// `TorchDispatchMode` over `transformers` 5.15.1 + torch 2.13.0 rather than
// read off a doc comment: docs/OPS4.md's own §0 note is that doc comments
// have been wrong about upstream three times before, so every rule below was
// re-measured, not copied from a kernel's docstring.
// ---------------------------------------------------------------------------

/// `aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor`
///
/// `mamba`'s selective-scan `dt` (the discretisation step size) is
/// `softplus(dt_proj(x) + dt_bias)`, always `float32`, always the default
/// `beta`/`threshold` (measured: `softplus.default(float32(1,128,6))`, two
/// positional args absent).
///
/// **Not implemented for integral/boolean input** -- measured on real torch,
/// `softplus_cpu` raises `NotImplementedError` naming the dtype rather than
/// promoting the way `exp`/`tanh` do; softplus is refused, not widened.
///
/// **The formula is upstream's, transcribed, and the "numerically stable"
/// rewrite this kernel used to carry was a divergence rather than an
/// improvement.** Upstream's `softplus_kernel` is one line:
///
/// ```text
/// (a * beta) > threshold ? a : std::log1p(std::exp(a * beta)) / beta
/// ```
///
/// The previous version computed `max(y,0) + log(1 + exp(-|y|))` instead --
/// mathematically the same function, and never equal to it in floating
/// point. docs/SCALAR.md §5 recorded the resulting disagreement as open,
/// with `softplus(-3)` at `float64` reading `0.048587351573742**06**`
/// upstream and `…**196**` here. Two separate causes, both closed here:
///
///   * **`log1p`, not `log(1 + ·)`.** Checked against `math.log1p(math.exp(x))`
///     on ten values in `float64`: upstream agrees with `log1p(exp(x))` on
///     all ten and with the split on six.
///   * **the split is not even the same function at the edges.** With a
///     `threshold` large enough not to fire, upstream's `exp` overflows and
///     the answer is `inf`: `softplus(800.0, 1, 1e9)` is `inf` upstream and
///     was `800.0` here. The split cannot overflow, which is exactly why it
///     was chosen and exactly why it disagrees.
///
/// **The arithmetic runs in `opmath`, not in the storage dtype.** Upstream's
/// `scalar_t` is `c10::BFloat16`/`c10::Half` for the reduced floats, and
/// every arithmetic operator on those promotes to `float` and back, so the
/// whole expression is evaluated in `float32` and narrowed once at the end.
/// This kernel used candle tensor ops, which stay in the storage dtype at
/// every step -- `bfloat16 softplus(-3)` was `0.0458984375` here against
/// upstream's `0.048583984375`, a 6% error that `dtypes.py`'s `bfloat16`
/// tolerance of 6e-2 absorbs completely. That is why the cases for it use
/// `_bit_exact`.
///
/// `beta` and `threshold` are narrowed to the tensor's dtype *first*
/// (`beta_.to<scalar_t>()`). Where that is observable is not where you would
/// look for it: at `bfloat16` and `float16` it is **not** observable at all,
/// because the final narrowing back to 8 or 11 bits absorbs the difference —
/// a 100-point search over `beta` and `x` found no separating pair. It is
/// observable at **`float32`**, where the narrowing of `beta` is a real
/// rounding and the result keeps 24 bits: `softplus(-3.440680608220717,
/// beta=0.1)` is `5.3583855628967285` with `float(0.1)` and
/// `5.358386039733887` with the `double` `0.1`. That gap is 9e-8 relative,
/// inside `dtypes.py`'s `float32` tolerance, so its case is `_bit_exact` too.
///
/// **`float32` cannot be matched bit for bit and that is upstream's
/// property, not this kernel's.** `cpu_kernel_vec` runs a Sleef-vectorised
/// body and a scalar tail, and the two do not agree: `softplus(-3.0)` in
/// `float32` is `0x1.8e070e0p-5` in a tensor of fewer than 8 elements and
/// `0x1.8e07100p-5` in a longer one, measured at n = 1, 2, 3, 4, 7, 8, 16,
/// 17, 32, 64, 100. That is one ULP and it is the same class docs/LOSS.md
/// §5.4 records for `_log_softmax`. The `float32` cases therefore use the
/// ordinary tolerance; `float64`, `float16` and `bfloat16` are all stable
/// across length and are pinned bit-exactly.
///
/// Above `threshold`, upstream skips the formula entirely and returns `x`
/// itself, not an evaluation of it -- measured `softplus(20.1) == 20.1`
/// exactly, which the formula alone would not promise. The comparison is
/// `>`, so `y == threshold` computes. Every measured call in `mamba` stays
/// well inside the default `threshold=20`, so that branch is exercised by
/// the golden cases, not by the model.
///
/// Nothing is special-cased for non-finite input; every one of them falls
/// out of the expression, and all four were measured. `+inf` takes the
/// threshold branch and returns itself; `-inf` gives `log1p(exp(-inf)) =
/// log1p(0) = 0`; `NaN` fails the `>` and propagates through `exp`; `-0.0`
/// gives `log(2)`. `beta = 0` gives `inf` (upstream's `/beta`), and a
/// negative `beta` mirrors the function rather than refusing.
fn softplus_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.softplus.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    if !input.tag().is_floating_point() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"softplus_cpu\" not implemented for '{}'",
            scalar_type_name(input.tag())
        )));
    }
    let beta = scalar_arg(OP, args, kwargs, 1, "beta")?
        .map(Scalar::as_f64)
        .unwrap_or(1.0);
    let threshold = scalar_arg(OP, args, kwargs, 2, "threshold")?
        .map(Scalar::as_f64)
        .unwrap_or(20.0);

    let tag = input.tag();
    // `beta_.to<scalar_t>()` / `threshold_.to<scalar_t>()`, before anything
    // is computed with them.
    let narrow = float_narrower(tag);
    let beta = narrow(beta);
    let threshold = narrow(threshold);

    let dims = input.tensor()?.dims().to_vec();
    let device = input.tensor()?.device().clone();
    let values = match read_flat(OP, input.tensor()?, tag)? {
        Flat::Float(v) => v,
        Flat::Int(_) => unreachable!("the dtype was checked above"),
    };

    // `opmath`: `double` for `double`, `float` for everything else this can
    // hold. The walk carries `f64` because `read_flat` does; the `as f32`
    // round trip is what makes the reduced-float arms compute where upstream
    // computes.
    let out: Vec<f64> = if tag == TorchDType::Float64 {
        values
            .iter()
            .map(|&x| {
                let y = x * beta;
                if y > threshold {
                    x
                } else {
                    y.exp().ln_1p() / beta
                }
            })
            .collect()
    } else {
        let (beta, threshold) = (beta as f32, threshold as f32);
        values
            .iter()
            .map(|&x| {
                let x = x as f32;
                let y = x * beta;
                let value = if y > threshold { x } else { y.exp().ln_1p() / beta };
                narrow(value as f64)
            })
            .collect()
    };

    let tensor = write_flat(OP, Flat::Float(out), dims, &device, tag)?;
    finish(py, tensor, tag)
}

/// `aten::convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[]
///     stride, SymInt[] padding, SymInt[] dilation, bool transposed,
///     SymInt[] output_padding, SymInt groups) -> Tensor`
///
/// `mamba`'s short causal depthwise conv over the SSM's `(x, B, C)` sequence:
/// measured `convolution.default(float32(1,128,6), float32(128,1,4),
/// float32(128,), stride=[1], padding=[3], dilation=[1], transposed=False,
/// output_padding=[0], groups=128)` -- `groups == in_channels == out_channels`
/// (depthwise), `padding == kernel_size - 1` on *both* sides (the caller
/// slices `[..., :seq_len]` afterwards to keep it causal, which is the
/// model's job, not this kernel's).
///
/// Only that shape is implemented: **not transposed**, a zero
/// `output_padding` (irrelevant when not transposed, but checked so an
/// unmeasured combination fails loudly rather than being silently ignored),
/// a 3-D input (`(batch, channels, length)` -- the 1-D conv `nn.Conv1d`
/// lowers to), and floating dtypes only. `candle_core::Tensor::conv1d`
/// already accepts `groups`, and its symmetric `padding` argument is the
/// same convention `aten::convolution` uses for a non-transposed 1-D
/// convolution, so no reshaping trick is needed to reuse it.
///
/// Bias is `(out_channels,)` and is not something `conv1d` applies itself --
/// added afterwards, reshaped to `(1, out_channels, 1)` to broadcast over the
/// batch and length axes.
fn convolution_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.convolution.default";
    let input = tensor_arg(OP, args, kwargs, 0, "input")?;
    let weight = tensor_arg(OP, args, kwargs, 1, "weight")?;
    let bias = optional_tensor_arg(OP, args, kwargs, 2, "bias")?;
    let stride = shape_arg(OP, args, kwargs, 3, "stride")?;
    let padding = shape_arg(OP, args, kwargs, 4, "padding")?;
    let dilation = shape_arg(OP, args, kwargs, 5, "dilation")?;
    let transposed = bool_arg(args, kwargs, 6, "transposed")?.unwrap_or(false);
    let output_padding = shape_arg(OP, args, kwargs, 7, "output_padding")?;
    let groups = required(OP, args, kwargs, 8, "groups")?.extract::<i64>()?;

    if !transposed && output_padding.iter().any(|&v| v != 0) {
        return Err(not_implemented(format!(
            "{OP}: a non-zero output_padding is only meaningful for a transposed \
             convolution, and is not implemented for a forward one in torch._C shim"
        )));
    }
    // 1-D (3-D input) and 2-D (4-D input). `spatial` is how many trailing
    // dimensions are convolved, and it is what every argument below is
    // length-checked against.
    //
    // **2-D was ARCH26.md §3.2's wall and it turned out to be the small piece,
    // not the large one** (docs/KERNELS26.md §7): candle already carries
    // `Tensor::conv2d` with the same `(padding, stride, dilation, groups)`
    // signature `conv1d` has, so this is the same thin wrapper twice rather
    // than a second kernel. What it is *not* is a general 2-D convolution --
    // see the symmetry refusal below.
    let rank = input.tensor()?.rank();
    let spatial = match rank {
        3 => 1usize,
        4 => 2usize,
        _ => {
            return Err(not_implemented(format!(
                "{OP}: only 1-D convolution (3-D input, (batch, channels, length)) and \
                 2-D convolution (4-D input, (batch, channels, height, width)) are \
                 implemented in torch._C shim, got {rank}-D"
            )))
        }
    };
    // **A single value broadcasts to every convolved axis**, which is torch's
    // own `expand_param_if_needed`, measured: `convolution(4-D, ..., padding=[2],
    // ...)` pads both axes by 2 and gives `(1, 3, 7, 7)`. Anything that is
    // neither 1 nor `spatial` long raises, with upstream's wording -- a
    // 2-element stride on a 3-D input is refused upstream too, so this is not
    // a shim restriction.
    let expand = |name: &str, v: Vec<isize>| -> PyResult<Vec<isize>> {
        match v.len() {
            1 => Ok(vec![v[0]; spatial]),
            n if n == spatial => Ok(v),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "expected {name} to be a single integer value or a list of {spatial} \
                 values to match the convolution dimensions, but got {name}={v:?}"
            ))),
        }
    };
    let stride = expand("stride", stride)?;
    let padding = expand("padding", padding)?;
    let dilation = expand("dilation", dilation)?;
    let output_padding = if transposed {
        expand("output_padding", output_padding)?
    } else {
        output_padding
    };
    if stride.iter().any(|&v| v <= 0)
        || padding.iter().any(|&v| v < 0)
        || dilation.iter().any(|&v| v <= 0)
    {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{OP}: stride and dilation must be positive, padding must be non-negative"
        )));
    }
    // **candle's `conv2d` takes scalars, so it is symmetric only.** torch
    // allows `(stride_h, stride_w)` to differ, and an asymmetric call reaching
    // a symmetric kernel would silently convolve with the wrong geometry --
    // the output shape would even be wrong, but only in one axis, which is the
    // kind of thing a single square test case does not show. So it is refused
    // by name, and the message says which axis pair disagreed.
    //
    // Nothing measured needs it: `Dinov2`'s patch embedding, which is what
    // ARCH26.md §3.2 stopped on, is `nn.Conv2d(3, hidden, kernel_size=16,
    // stride=16)` -- square kernel, square stride, no padding.
    if spatial == 2 {
        let mut axed: Vec<(&str, &Vec<isize>)> =
            vec![("stride", &stride), ("padding", &padding), ("dilation", &dilation)];
        if transposed {
            axed.push(("output_padding", &output_padding));
        }
        for (name, v) in axed {
            if v[0] != v[1] {
                return Err(not_implemented(format!(
                    "{OP}: an asymmetric {name} {v:?} is not implemented in torch._C shim \
                     -- candle's conv2d takes one value per argument, not one per axis"
                )));
            }
        }
    }
    // --- transposed convolution: what is implemented, and what is refused ---
    //
    // `zoedepth`'s `ZoeDepthUpsample` is `nn.ConvTranspose2d(channels, channels,
    // kernel_size=factor, stride=factor, padding=0)` -- 2-D, groups=1, square
    // everything. That is what this implements; the rest is refused by name.
    //
    // **candle's `conv_transpose2d` has no `groups` parameter** (its
    // `conv_transpose1d` does, and its `ParamsConvTranspose2D` simply has no
    // field for it), so a grouped transposed convolution cannot be handed to it
    // and is refused rather than silently computed as `groups=1` -- which would
    // produce a tensor of the wrong *shape* here, but only because `c_out` is
    // read off the weight; a decomposition into per-group calls is possible and
    // is left for a round that has a caller for it.
    if transposed {
        if spatial != 1 && spatial != 2 {
            return Err(not_implemented(format!(
                "{OP}: only 1-D and 2-D transposed convolution (3-D or 4-D input) is \
                 implemented in torch._C shim, got {spatial}-D"
            )));
        }
        // **1-D transposed convolution keeps `groups`; 2-D does not.** That is
        // candle's asymmetry, not upstream's: `conv_transpose1d` takes a
        // `groups` argument and `ParamsConvTranspose2D` has no field for one.
        // docs/KERNELS26.md §10.3 refused the 1-D case entirely for the
        // opposite reason -- candle supports it fully and nothing measured
        // reached it. `vits` reaches it now (`modeling_vits.py`'s HiFi-GAN
        // decoder is `nn.ConvTranspose1d(channels, channels//2, kernel,
        // stride=rate, padding=(kernel-rate)//2)` once per upsample rate), so
        // the refusal is lifted for 1-D and stands for grouped 2-D.
        if spatial == 2 && groups != 1 {
            return Err(not_implemented(format!(
                "{OP}: a grouped 2-D transposed convolution (groups={groups}) is not \
                 implemented in torch._C shim -- candle's conv_transpose2d takes no \
                 groups argument, while its conv_transpose1d does"
            )));
        }
        // Upstream's own precondition, measured: `output_padding=1` is accepted
        // with `stride=2, dilation=1` and with `stride=1, dilation=2`, and
        // refused with `stride=1, dilation=1`. So the bound is
        // `max(stride, dilation)`, not `stride`. Message reproduced verbatim,
        // and upstream's 1-D wording names one axis where the 2-D one names
        // two.
        for i in 0..spatial {
            if output_padding[i] >= stride[i].max(dilation[i]) {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(if spatial == 1 {
                    format!(
                        "output padding must be smaller than either stride or dilation, \
                         but got output_padding: {}",
                        output_padding[0]
                    )
                } else {
                    format!(
                        "output padding must be smaller than either stride or dilation, \
                         but got output_padding_height: {} output_padding_width: {}",
                        output_padding[0], output_padding[1]
                    )
                }));
            }
        }
        if output_padding.iter().any(|&v| v < 0) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{OP}: output_padding must be non-negative"
            )));
        }
    }
    if groups <= 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{OP}: groups must be a positive integer"
        )));
    }

    let tag = require_same_dtype(OP, &input, &weight)?;
    if !tag.is_floating_point() {
        return Err(not_implemented(format!(
            "{OP}: only floating-point convolution is implemented in torch._C shim, \
             got {}",
            scalar_type_name(tag)
        )));
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let x = input.tensor()?.fast_to(storage).map_err(|e| candle_err(OP, e))?;
    let w = weight.tensor()?.fast_to(storage).map_err(|e| candle_err(OP, e))?;
    let raw = if transposed {
        // **The weight layout is `(in_channels, out_channels/groups, kH, kW)`,
        // not `(out, in, kH, kW)`** -- the opposite of the forward convolution
        // just above, and the single most dangerous line in this kernel.
        //
        // Established from upstream's behaviour rather than from the docs, three
        // independent ways:
        //
        //   1. shape. `conv_transpose2d(x(2,3,5,7), w(3,5,2,4))` gives
        //      `(2, 5, 6, 10)` -- `out_channels` is `w.shape[1]`. Handing it the
        //      transposed `w(5,3,2,4)` raises `expected input[2,3,5,7] to have
        //      5 channels, but got 3 channels`.
        //   2. `nn.ConvTranspose2d(3, 5, kernel_size=(2,4)).weight.shape` is
        //      `[3, 5, 2, 4]`.
        //   3. a from-scratch scatter-add implementation of the definition
        //      (transposed convolution = the gradient of a convolution with
        //      respect to its input) agrees with upstream on four
        //      configurations including `groups=2`.
        //
        // **The measurement had to be built on a non-square, unequal-channel
        // case, because `zoedepth`'s own call cannot show any of this**: it is
        // `nn.ConvTranspose2d(channels, channels, kernel_size=factor,
        // stride=factor)` -- equal in/out channels and a square kernel, so a
        // transposed weight has exactly the same shape and produces a plausible
        // tensor instead of an error. Measured on `(1,2,3,3)` input with a
        // `(2,2,3,3)` weight: swapping the first two axes changes the sum from
        // 61317 to 54756 with no shape change, and **flipping the kernel
        // spatially leaves the sum at 61317 while changing every element** --
        // so even a checksum does not separate those two. The golden cases
        // compare element by element and carry both wrong layouts as live
        // shapes.
        //
        // candle's `conv_transpose2d` reads its kernel as `(c_in_k, c_out, k_h,
        // k_w)` and bails if `c_in_k` disagrees with the input's channels, so it
        // is the same convention and the weight is passed through unpermuted.
        //
        // Its argument ORDER is not `conv2d`'s: `(kernel, padding,
        // output_padding, stride, dilation)` against `conv2d`'s `(kernel,
        // padding, stride, dilation, groups)`. `output_padding` sits where
        // `stride` sits in the forward call, which is another way to get a
        // plausible tensor of the wrong shape.
        //
        // The 1-D sibling takes the same argument order with `groups` appended
        // -- `(kernel, padding, output_padding, stride, dilation, groups)` --
        // and reads its kernel as `(c_in_k, c_out, k)`, the same
        // input-channels-first convention. So the weight is passed through
        // unpermuted in both ranks.
        if spatial == 1 {
            x.conv_transpose1d(
                &w,
                padding[0] as usize,
                output_padding[0] as usize,
                stride[0] as usize,
                dilation[0] as usize,
                groups as usize,
            )
        } else {
            x.conv_transpose2d(
                &w,
                padding[0] as usize,
                output_padding[0] as usize,
                stride[0] as usize,
                dilation[0] as usize,
            )
        }
    } else if spatial == 1 {
        x.conv1d(
            &w,
            padding[0] as usize,
            stride[0] as usize,
            dilation[0] as usize,
            groups as usize,
        )
    } else {
        x.conv2d(
            &w,
            padding[0] as usize,
            stride[0] as usize,
            dilation[0] as usize,
            groups as usize,
        )
    }
    .map_err(|e| candle_err(OP, e))?;
    let out = match bias {
        Some(b) => {
            if b.tag() != tag {
                return Err(not_implemented(format!(
                    "{OP}: bias dtype must match input/weight dtype in torch._C shim"
                )));
            }
            let c_out = raw.dim(1).map_err(|e| candle_err(OP, e))?;
            // One trailing `1` per convolved axis, so the bias broadcasts
            // along the channel dimension in either rank.
            let mut b_shape = vec![1usize, c_out];
            b_shape.extend(std::iter::repeat(1usize).take(spatial));
            let b_reshaped = b
                .tensor()?
                .fast_to(storage)
                .and_then(|t| t.reshape(b_shape))
                .map_err(|e| candle_err(OP, e))?;
            raw.broadcast_add(&b_reshaped).map_err(|e| candle_err(OP, e))?
        }
        None => raw,
    };
    finish(py, out, tag)
}

/// `aten::zeros_like`/`aten::empty_like`/`aten::ones_like(Tensor self, *,
///     ScalarType? dtype=None, Layout? layout=None, Device? device=None,
///     bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor`
///
/// `mamba`'s selective-scan state is seeded with `zeros_like(...)`, and
/// `mixtral`'s grouped-MoE routing (`transformers`'
/// `integrations/moe.py::grouped_mm_experts_forward`) allocates
/// `torch.empty_like(perm)` purely to be overwritten in full two lines later
/// (`inv_perm[perm] = torch.arange(...)`) -- so, exactly like
/// `empty.memory_format` above, "empty" answers zeros here: the shim is
/// deterministic where upstream is not, and both measured call sites read
/// every element back before using it.
///
/// **`ones_like` joins them as the same function with a different fill.** It
/// is the wall `vits` AND `sam3_video` both landed on once `weight_norm` and
/// `outer` were behind them -- two architectures on one one-line kernel. Unlike
/// its two siblings its value is *defined*, so it is the only one of the three
/// whose golden case can diff real values rather than dtype and shape alone.
///
/// Structured like `new_ones_default`: the reference tensor supplies the
/// defaults (shape always, dtype/device unless overridden) a bare factory
/// would otherwise take from the process-wide default.
fn zeros_or_empty_like(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let tag = dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(input.tag());
    reject_unsupported(op, args, kwargs, &[(2, "layout"), (4, "pin_memory")])?;
    // `memory_format` was in the blanket-refusal list above and is now read the
    // way `full_like` and `clone` next door already read it: the two formats
    // that mean "leave the layout alone" are accepted, everything else still
    // refuses with the format named.
    //
    // **This round is what made it reachable, and that is the reason for the
    // change rather than tidiness.** `torch/autograd/__init__.py` `_make_grads`
    // builds its seed with `torch.ones_like(out, memory_format=preserve_format)`
    // and only reaches that line `if out.requires_grad` -- which was `False` for
    // every intermediate until docs/BACKWARD4.md. With the flag propagating, a
    // blanket refusal here would have **moved `Tensor.backward()`'s refusal**
    // off `_ImperativeEngine.run_backward` and onto a memory-format argument, so
    // the wall a user hits would have stopped naming the thing that is missing.
    reject_memory_format(op, args, kwargs, 5)?;
    let label = device_arg_or_label(args, kwargs, 3, "device", &input.device_label())?;
    let shape = input.tensor()?.dims().to_vec();
    if label.is_meta() {
        return meta_result(py, shape, tag);
    }
    let device = label.resolve()?;
    let storage = PyDtype::new(tag).storage(op)?;
    let out = if op == "aten.ones_like.default" {
        Tensor::ones(shape, storage, &device)
    } else {
        Tensor::zeros(shape, storage, &device)
    }
    .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::floor_divide(Tensor self, Tensor other) -> Tensor`
///
/// `mixtral`'s grouped-MoE routing recovers each selected token's row with
/// `hidden_states[perm // num_top_k]` -- `perm` an `int64` tensor, `//` a
/// **Python `int`**, not a tensor. Measured with a real `TorchDispatchMode`:
/// the dispatched call is `floor_divide.default(int64_tensor, 2)`, a bare
/// Python `int` reaching the `(Tensor, Tensor)` overload's `other` slot
/// rather than the `floor_divide.Scalar` overload the schema alternative
/// would suggest -- torch's own frontend picks `.default` here and leaves the
/// scalar-to-tensor conversion for the kernel, not the dispatcher. Both
/// shapes are accepted below.
///
/// **Floors toward negative infinity, matching Python's `//` -- not C's
/// truncation.** Measured on torch 2.13.0 with mixed-sign input: `[-7, -6,
/// -1, 0, 1, 6, 7] // 2 == [-4, -3, -1, 0, 0, 3, 3]` (not `[-3, -3, 0, 0, 0,
/// 3, 3]`, which is what truncating division would give for the negative
/// entries). Implemented as truncating division with the standard correction
/// (subtract one from the truncated quotient when the remainder is non-zero
/// and its sign disagrees with the divisor's), which reproduces the measured
/// table exactly.
///
/// **Dtype is preserved, not promoted** -- unlike `div.Tensor`'s true
/// division, `int64 // int64` stays `int64` (measured), so this does not
/// route through `arith_tag`.
///
/// **Division by zero on an integral dtype raises**, matching upstream's
/// measured `RuntimeError('ZeroDivisionError')` exactly (message and all) --
/// checked eagerly, before any division happens, so every element of a
/// zero-divisor call fails together the way upstream's does. Division by
/// zero on a floating dtype is not refused (`1.0 // 0.0 == inf`, `-1.0 //
/// 0.0 == -inf`, `0.0 // 0.0 == nan`, all measured) -- ordinary `f64`
/// division already answers this correctly, no special case needed.
fn floor_divide_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    floor_divide_impl(py, "aten.floor_divide.default", args, kwargs)
}

/// `aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor`
///
/// The same arithmetic as `.default`, reached by a different route, and the
/// route is the reason this exists.
///
/// The doc comment above `.default` records that upstream sends
/// `perm // num_top_k` -- a tensor and a bare Python `int` -- to
/// `floor_divide.default`, because torch's own frontend wraps the number into
/// a tensor before it picks an overload. **This shim's resolver does not do
/// that**: `overloads.json` is walked in order and an `int` does not bind a
/// `Tensor` parameter, so `torch.floor_divide(t, 2)` lands on `.Scalar` here
/// where upstream lands on `.default`. The same is already true of
/// `add`/`sub`/`div`, which is why their `.Scalar` overloads are implemented
/// too; this is that pattern, not a new one.
///
/// So the two keys disagree with upstream's *choice* while agreeing on the
/// *answer*, and closing that gap properly means teaching the resolver
/// upstream's "numbers as tensors" rule -- a change in `bootstrap.py`, above
/// this file. Until then, refusing here would stop Mixtral's MoE routing on
/// an op the shim can already compute. docs/GROUPED_MM.md §6.
fn floor_divide_scalar(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.floor_divide.Scalar";
    // A tensor in the `Scalar other` slot is `.default`'s call, not this
    // one's. Refusing keeps the two keys distinguishable in the work queue
    // DESIGN.md §6 is built on, rather than quietly folding them together.
    if required(OP, args, kwargs, 1, "other")?
        .extract::<PyTensorBase>()
        .is_ok()
    {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{OP}: argument 'other' must be a Scalar, not a Tensor -- a tensor \
             divisor is aten.floor_divide.default"
        )));
    }
    floor_divide_impl(py, OP, args, kwargs)
}

fn floor_divide_impl(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
    let raw_other = required(op, args, kwargs, 1, "other")?;
    let n = lhs.tensor()?.elem_count();
    
    let mut scalar_at_opmath = false;
    let tag;

    let other_flat: Flat = if let Ok(other_tensor) = raw_other.extract::<PyTensorBase>() {
        if lhs.tag() == TorchDType::Bool && other_tensor.tag() == TorchDType::Bool {
            // `NotImplementedError`, not `RuntimeError`: upstream raises the
            // first one here, and a caller that writes `except
            // NotImplementedError` around a bool pair would not catch a
            // `RuntimeError` carrying the same words. `isin` a few thousand
            // lines up refuses bool with `RuntimeError` because that is what
            // upstream raises *there* -- the two differ, so they are spelled
            // separately rather than made consistent with each other.
            return Err(not_implemented(
                "\"div_floor_cpu\" not implemented for 'Bool'".to_string()
            ));
        }
        tag = promote_operands(op, &lhs, &other_tensor)?;
        let flat = read_flat(op, other_tensor.tensor()?, tag)?;
        let count = other_tensor.tensor()?.elem_count();
        if count != n && count != 1 {
            return Err(not_implemented(format!(
                "{op}: broadcasting other than a scalar or an exact shape match is not \
                 implemented in torch._C shim"
            )));
        }
        flat
    } else {
        if lhs.tag() == TorchDType::Bool && raw_other.is_instance_of::<pyo3::types::PyBool>() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "\"div_floor_cpu\" not implemented for 'Bool'"
            ));
        }
        let scalar = scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
        
        tag = if !scalar.is_int() {
            if lhs.tag().is_floating_point() { lhs.tag() } else { TorchDType::Float32 }
        } else {
            if lhs.tag() == TorchDType::Bool { TorchDType::Int64 } else { lhs.tag() }
        };

        if tag.is_floating_point() {
            scalar_at_opmath = matches!(tag, TorchDType::Float16 | TorchDType::BFloat16);
            let target = if scalar_at_opmath { TorchDType::Float32 } else { tag };
            Flat::Float(vec![float_narrower(target)(scalar.as_f64())])
        } else {
            Flat::Int(vec![scalar.as_i64()])
        }
    };

    let self_flat = read_flat(op, lhs.tensor()?, tag)?;
    let out_flat = match (self_flat, other_flat) {
        (Flat::Float(a), Flat::Float(b)) => {
            // **`floor(a / b)` is not what upstream computes**, which
            // `div_floor_float`'s own doc comment has said since it was
            // written -- this key was calling the plausible version instead of
            // the transcribed one, and the two disagree wherever the `f64`
            // quotient lands just under an integer. `float64 -3.0 // 0.3` is
            // `-11` upstream (the quotient is -10.000000000000002) and was
            // `-10` here. Sharing the function is what stops the two spellings
            // of floor division from drifting apart again.
            let narrow = float_narrower(if scalar_at_opmath {
                TorchDType::Float32
            } else {
                tag
            });
            let get = |i: usize| if b.len() == 1 { b[0] } else { b[i] };
            Flat::Float(
                a.iter()
                    .enumerate()
                    .map(|(i, &x)| div_floor_float(x, get(i), narrow))
                    .collect(),
            )
        }
        (Flat::Int(a), Flat::Int(b)) => {
            let get = |i: usize| if b.len() == 1 { b[0] } else { b[i] };
            let mut out = Vec::with_capacity(a.len());
            for (i, &x) in a.iter().enumerate() {
                let d = get(i);
                if d == 0 {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err("ZeroDivisionError"));
                }
                let q = x / d;
                let r = x % d;
                out.push(if r != 0 && (r < 0) != (d < 0) { q - 1 } else { q });
            }
            Flat::Int(out)
        }
        _ => unreachable!("self and other share a dtype, checked above"),
    };

    let dims = lhs.tensor()?.dims().to_vec();
    let device = lhs.tensor()?.device().clone();
    let out = write_flat(op, out_flat, dims, &device, tag)?;
    finish(py, out, tag)
}

/// `aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor`
///
/// `mixtral`'s grouped-MoE routing counts tokens per expert with
/// `torch.histc(expert_ids_g.float(), bins=num_experts, min=0,
/// max=num_experts-1)` (`transformers`' `integrations/moe.py`, CPU path --
/// the comment there notes `histc` doesn't support integer dtypes on CPU,
/// which measured confirms: `NotImplementedError("histogram_cpu" not
/// implemented for 'Long')`, so only floating input is implemented here).
///
/// **`min == max` means "ignore both and use the data's own min/max"** --
/// measured on torch 2.13.0 with *non-zero* equal bounds too
/// (`histc(x, min=5, max=5)` on data ranging `[1,3]` bins over `[1,3]`, not
/// `[5,5]`), so the rule is "min equals max", not "both are zero". A
/// genuinely degenerate range (the data itself is constant) is a second,
/// nested case: measured `histc([2,2,2], bins=4, min=0, max=0)` answers
/// `[0,0,3,0]`, which is exactly what falling back to `[value-1, value+1]`
/// produces (bin width `0.5`, `2.0` lands in `[2.0,2.5)`, the third of four
/// bins) -- reproduced here as the same fallback applied twice rather than
/// as a separately-derived formula.
///
/// Range is inclusive on both ends (`x == max` counts in the last bin,
/// measured), and elements strictly outside `[min, max]` are dropped, not
/// clamped into an edge bin (measured: `-1` and `4` are absent from
/// `histc([..., -1, 4], bins=4, min=0, max=3)`'s counts).
///
/// `bins <= 0` and an explicit `min > max` are refused with upstream's exact
/// wording (both measured): `"bins must be > 0, but got {bins} for dimension
/// 0"` and `"torch.histc: max must be larger than min"`.
fn histc_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.histc.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"histogram_cpu\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    let bins = int_arg(args, kwargs, 1, "bins")?.unwrap_or(100);
    if bins <= 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "torch.histogram(): bins must be > 0, but got {bins} for dimension 0"
        )));
    }
    let min_arg = scalar_arg(OP, args, kwargs, 2, "min")?.map(Scalar::as_f64).unwrap_or(0.0);
    let max_arg = scalar_arg(OP, args, kwargs, 3, "max")?.map(Scalar::as_f64).unwrap_or(0.0);

    let values = read_flat(OP, input.tensor()?, tag)?;
    let values = match values {
        Flat::Float(v) => v,
        Flat::Int(_) => unreachable!("floating dtype checked above"),
    };

    let (mut lo, mut hi) = if min_arg == max_arg {
        let mut data_lo = f64::INFINITY;
        let mut data_hi = f64::NEG_INFINITY;
        for &v in &values {
            if v < data_lo {
                data_lo = v;
            }
            if v > data_hi {
                data_hi = v;
            }
        }
        if values.is_empty() {
            (0.0, 0.0)
        } else {
            (data_lo, data_hi)
        }
    } else {
        (min_arg, max_arg)
    };
    if lo > hi {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "torch.histc: max must be larger than min",
        ));
    }
    if lo == hi {
        lo -= 1.0;
        hi += 1.0;
    }

    let mut counts = vec![0.0f64; bins as usize];
    let width = (hi - lo) / bins as f64;
    for &v in &values {
        if v < lo || v > hi || v.is_nan() {
            continue;
        }
        let mut bin = ((v - lo) / width).floor() as i64;
        if bin >= bins {
            bin = bins - 1;
        }
        if bin < 0 {
            bin = 0;
        }
        counts[bin as usize] += 1.0;
    }

    let device = input.tensor()?.device().clone();
    let out = write_flat(OP, Flat::Float(counts), vec![bins as usize], &device, tag)?;
    finish(py, out, tag)
}

/// `aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)`
///
/// `mixtral`'s grouped-MoE routing keeps a per-row expert-bias gather
/// in-bounds with `expert_ids_g.clamp_(max=self.num_experts - 1)` (`min`
/// absent) -- measured `clamp_.default(int64(12,), None, 3)`.
///
/// **The formula is `min(max(x, min_val), max_val)`, applied in that order
/// unconditionally** -- not "refuse if `min > max`". Measured:
/// `x.clamp_(min=8, max=2)` on `[1,5,10,-3]` gives `[2,2,2,2]` (every element
/// hits the floor first, then the ceiling clips it down again), which is
/// exactly what `candle_core::Tensor::clamp` already computes
/// (`maximum(min).minimum(max)`), so this reuses it rather than reproducing
/// it by hand. NaN propagates through both steps for the same reason
/// (Rust's `<`/`>` are false against NaN, so `maximum`/`minimum` return
/// whichever operand *is* NaN) -- measured `[nan,1.,-1.].clamp_(0,2) ==
/// [nan,1.,0.]`.
///
/// **A float bound against an integral receiver is refused outright,
/// regardless of the bound's actual value** -- measured `int32.clamp_(max=2.0)`
/// raises `"result type Float can't be cast to the desired output type
/// Int"` even though `2.0` is exactly representable as an int; torch does
/// not special-case exact values, so this does not either.
///
/// **Both bounds absent is refused, not a no-op** -- measured
/// `tensor.clamp_(None, None)` raises `"torch.clamp: At least one of 'min'
/// or 'max' must not be None"` rather than returning the receiver
/// unchanged, which "nothing to clamp against" would otherwise suggest.
fn clamp_inplace_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.clamp_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let min = scalar_arg(OP, args, kwargs, 1, "min")?;
    let max = scalar_arg(OP, args, kwargs, 2, "max")?;
    if min.is_none() && max.is_none() {
        // Measured, not a guess this shim could have gotten away with
        // skipping: `tensor.clamp_(None, None)` raises on real torch rather
        // than being an accepted no-op.
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "torch.clamp: At least one of 'min' or 'max' must not be None",
        ));
    }
    let tag = receiver.borrow().tag();
    clamp_dtype_refusals(tag, min, max)?;
    let source = receiver.borrow().tensor()?.clone();
    let out = clamp_values(OP, &source, tag, min, max)?;
    write_back(OP, &receiver, PyTensorBase::new(out)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// The dtype refusals `clamp` and `clamp_` share.
///
/// **`torch.bool` refuses whatever the bounds are**, and the message names the
/// *bound's* type rather than the receiver's. Measured on torch 2.13.0:
/// `clamp_(0, 5)` and `clamp_(None, 1)` give "result type Long can't be cast
/// to the desired output type bool", `clamp_(0.0, 1.0)` gives the same with
/// "Float", and `uint8` -- the dtype `bool` shares storage with -- computes
/// normally.
///
/// It is at the door rather than left to `write_into`'s backstop for a reason
/// that is about layering, not politeness: without it, `bool.clamp_(0, 5)`
/// produced a `uint8` replacement and `replace_with` *retagged the receiver*,
/// computing where upstream refuses. `write_into` refuses that now, but with
/// an "internal error" message aimed at whoever wrote the kernel. A
/// user-reachable refusal belongs at the door with upstream's own wording; the
/// tag check underneath stays as the structural backstop, the same shape as
/// `check_meta` sitting over `PyTensorBase::tensor`. docs/VIEWS.md §6.8.
///
/// **A float bound against an integral tensor is refused outright, regardless
/// of the bound's actual value** -- measured `int32.clamp(max=2.0)` raises
/// even though `2.0` is exactly representable as an int; torch does not
/// special-case exact values, so this does not either. Note that the
/// out-of-place `clamp` refuses here *too*, which is the half that does not
/// follow from "in-place cannot widen": upstream's `clamp` does not promote at
/// all, it requires the bound to fit the input.
fn clamp_dtype_refusals(
    tag: TorchDType,
    min: Option<Scalar>,
    max: Option<Scalar>,
) -> PyResult<()> {
    if tag == TorchDType::Bool {
        let saw_float = [min, max]
            .into_iter()
            .flatten()
            .any(|bound| !bound.is_int());
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "result type {} can't be cast to the desired output type bool",
            if saw_float { "Float" } else { "Long" }
        )));
    }
    if !tag.is_floating_point() {
        for bound in [min, max].into_iter().flatten() {
            if !bound.is_int() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "result type Float can't be cast to the desired output type {}",
                    scalar_type_name(tag)
                )));
            }
        }
    }
    Ok(())
}

/// `min(max(x, min_val), max_val)`, the value half of both clamp spellings.
fn clamp_values(
    op: &str,
    source: &Tensor,
    tag: TorchDType,
    min: Option<Scalar>,
    max: Option<Scalar>,
) -> PyResult<Tensor> {
    let mut out = source.clone();
    if let Some(bound) = min {
        out = if tag.is_floating_point() {
            out.maximum(bound.as_f64())
        } else {
            out.maximum(bound.as_i64())
        }
        .map_err(|e| candle_err(op, e))?;
    }
    if let Some(bound) = max {
        out = if tag.is_floating_point() {
            out.minimum(bound.as_f64())
        } else {
            out.minimum(bound.as_i64())
        }
        .map_err(|e| candle_err(op, e))?;
    }
    Ok(out)
}

/// The dtype ladder of `clamp.default`, and the two refusals that sit inside
/// it, extracted so the meta kernel produces the same answer by calling it
/// rather than by restating a table that took golden cases to get right once.
///
/// It reads the *raw* arguments and not only the parsed `Scalar`s, for the
/// reason the table in the body gives: `bool` subclasses `int` in Python and
/// `Scalar` collapses the two, but upstream does not.
fn clamp_result_tag(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    input_tag: TorchDType,
    min: Option<Scalar>,
    max: Option<Scalar>,
) -> PyResult<TorchDType> {
    if min.is_none() && max.is_none() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "torch.clamp: At least one of 'min' or 'max' must not be None",
        ));
    }
    // **The out-of-place form PROMOTES; the in-place one refuses.** That is
    // the whole dtype difference between the two, it is not obvious, and
    // sharing `clamp_dtype_refusals` here (which the first draft did) makes
    // this op refuse four measured rows that upstream computes. Measured on
    // 2.13.0:
    //
    //     clamp(int32,  0,     5)      int32       clamp_ agrees
    //     clamp(int32,  None,  2.0)    float32     clamp_ RAISES
    //     clamp(uint8,  None,  2)      uint8
    //     clamp(uint8,  None,  2.0)    float32
    //     clamp(float16,None,  2.0)    float16     a python float never widens a float tensor
    //     clamp(bool,   0,     5)      int64       clamp_ RAISES
    //     clamp(bool,   0.0,   1.0)    float32
    //     clamp(bool,   False, True)   RAISES      "clamp_scalar_cpu" not implemented for 'Bool'
    //
    // which is `arith_tag`'s wrapped-number rule with one extra: a *boolean*
    // scalar does not lift a boolean tensor out of the bool category, so the
    // result stays `bool` and upstream has no kernel for it. `bool` subclasses
    // `int` in Python and `Scalar` collapses the two, so that last row needs
    // the raw argument rather than the parsed `Scalar` -- which is why the
    // `PyBool` checks below read `optional` and not `scalar_arg`.
    let bound_is_bool = |index: usize, name: &str| -> PyResult<bool> {
        Ok(match optional(args, kwargs, index, name)? {
            Some(value) if !value.is_none() => value.is_instance_of::<pyo3::types::PyBool>(),
            _ => false,
        })
    };
    let saw_float = [min, max].into_iter().flatten().any(|b| !b.is_int());
    let bool_bounds = (min.is_none() || bound_is_bool(1, "min")?)
        && (max.is_none() || bound_is_bool(2, "max")?);

    // The bool-bounds refusal names the *kernel* that upstream failed to find,
    // and `clamp_min` is a different kernel from `clamp` even though every
    // other row of this ladder is shared -- measured on 2.13.0,
    // `torch.clamp_min(bool_t, False)` says `clamp_min_scalar_cpu` while
    // `torch.clamp(bool_t, False)` says `clamp_scalar_cpu`. Derived from `op`
    // rather than passed in, so a third caller cannot forget to say which it
    // is.
    let bool_kernel = if op.starts_with("aten.clamp_min") {
        "clamp_min_scalar_cpu"
    } else {
        "clamp_scalar_cpu"
    };
    Ok(if input_tag == TorchDType::Bool {
        if bool_bounds {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "\"{bool_kernel}\" not implemented for 'Bool'"
            )));
        }
        if saw_float {
            default_float()
        } else {
            TorchDType::Int64
        }
    } else if saw_float && !input_tag.is_floating_point() {
        default_float()
    } else {
        input_tag
    })
}

/// `aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor`
///
/// `mamba`'s wall (docs/ARCH20.md §4): `modeling_mamba.py` clamps `dt` and the
/// discretisation limits out of place, and only the *in-place* sibling had a
/// kernel -- `clamp_.default` has been implemented since docs/OPS8.md while
/// `x.clamp(...)` refused. That asymmetry is the one an in-place-first round
/// leaves behind, and it is the second instance of it in this file after
/// `relu`/`relu_` (docs/SPELLINGS.md §6.6) went the other way.
///
/// The *value* rule is `clamp_`'s, shared through `clamp_values` -- including
/// "both bounds absent is an error, not a no-op", which a fresh out-of-place
/// implementation would plausibly have made a no-op since there is no receiver
/// to leave unchanged. Measured: `x.clamp()` raises "torch.clamp: At least one
/// of 'min' or 'max' must not be None".
///
/// The *dtype* rule is not `clamp_`'s, and assuming it was is the mistake the
/// golden cases caught: see the table in `clamp_result_tag`, which is where
/// the rule now lives so that the meta kernel shares it.
///
/// `clamp.Tensor` (tensor bounds) is a separate overload with a separate
/// kernel and is not implemented; `methods.json` lists it so that
/// `x.clamp(min=some_tensor)` refuses *by the name of the overload it needed*
/// rather than by "no matching signature".
fn clamp_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.clamp.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let min = scalar_arg(OP, args, kwargs, 1, "min")?;
    let max = scalar_arg(OP, args, kwargs, 2, "max")?;
    let tag = clamp_result_tag(OP, args, kwargs, input.tag(), min, max)?;

    let storage = PyDtype::new(tag).storage(OP)?;
    let source = input
        .tensor()?
        .fast_to(storage)
        .map_err(|e| candle_err(OP, e))?;
    let out = clamp_values(OP, &source, tag, min, max)?;
    finish(py, out, tag)
}

/// `aten::clamp_min(Tensor self, Scalar min) -> Tensor`
///
/// `vits`'s wall: `modeling_vits.py:1352` computes
/// `torch.clamp_min(torch.sum(duration, [1, 2]), 1).long()` to keep the
/// predicted waveform length at least one frame -- measured
/// `clamp_min.default(float32(1,), 1)`.
///
/// **It is `clamp(min=..., max=None)` in every respect that this shim can
/// observe, and that was checked rather than assumed.** All ten rows of the
/// measurement below give byte-identical answers from `torch.clamp_min(t, b)`
/// and `torch.clamp(t, min=b)` on 2.13.0 -- the dtype ladder, the NaN rule and
/// the refusals -- so the value half reuses `clamp_values` and the dtype half
/// reuses `clamp_result_tag` rather than restating a table that took golden
/// cases to get right once:
///
///     clamp_min(int32,   0)      int32       clamp_min(bool,  0)     int64
///     clamp_min(int32,   2.0)    float32     clamp_min(bool,  0.0)   float32
///     clamp_min(uint8,   2)      uint8       clamp_min(bool,  False) RAISES
///     clamp_min(uint8,   2.0)    float32     clamp_min(f32,[nan,..]) nan kept
///     clamp_min(float16, 2.0)    float16     clamp_min(int64, -1)    int64
///
/// The one thing that is *not* shared is the wording of the `bool`-bound
/// refusal: upstream names `clamp_min_scalar_cpu`, not `clamp_scalar_cpu`.
/// `clamp_result_tag` derives that from `op`.
///
/// **`min` is required here where `clamp`'s is optional**, which removes
/// `clamp`'s "both bounds absent is an error, not a no-op" branch entirely --
/// there is no spelling of this op that reaches it. Passing `None` cannot bind
/// `Scalar min`, so the overload resolver refuses before a kernel runs.
///
/// `clamp_min.Tensor` (a tensor floor) is a separate overload with a separate
/// kernel and is not implemented -- it is `maximum` with broadcasting and
/// binary promotion, and this shim has no `aten.maximum.default` to delegate
/// to, so writing it is a broadcast kernel and not a one-line alias. Both
/// tables list it so that `torch.clamp_min(x, some_tensor)` refuses *by the
/// name of the overload it needed*, the same shape as `clamp.Tensor`.
fn clamp_min_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.clamp_min.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let min = scalar_arg(OP, args, kwargs, 1, "min")?.ok_or_else(|| missing(OP, "min"))?;
    let tag = clamp_result_tag(OP, args, kwargs, input.tag(), Some(min), None)?;

    let storage = PyDtype::new(tag).storage(OP)?;
    let source = input
        .tensor()?
        .fast_to(storage)
        .map_err(|e| candle_err(OP, e))?;
    let out = clamp_values(OP, &source, tag, Some(min), None)?;
    finish(py, out, tag)
}

/// `aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)`
///
/// `mixtral`'s router normalises top-k weights in place:
/// `top_k_weights.div_(top_k_weights.sum(dim=-1, keepdim=True))`, measured
/// `div_.Tensor(float32(6,2), float32(6,1))` -- `other` broadcasting *into*
/// the receiver's shape, never the other way around, which is in-place's
/// general rule (`add_inplace` above) and is exactly what this call needs
/// (`(6,1)` into `(6,2)`).
///
/// **True division, and the receiver's dtype cannot change to accommodate
/// it** -- unlike `div.Tensor`, which promotes an integral pair to
/// `float32` (`arith_tag`), the in-place form has nowhere to put a wider
/// result: measured `int64_tensor.div_(int64_tensor)` raises `"result type
/// Float can't be cast to the desired output type Long"`. So this refuses
/// non-floating receivers by name rather than silently truncating back to
/// int, which would be a wrong answer with no trace.
fn div_inplace_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.div_.Tensor";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let other = tensor_arg(OP, args, kwargs, 1, "other")?;

    let (tag, shape) = {
        let borrowed = receiver.borrow();
        (borrowed.tag(), borrowed.tensor()?.shape().clone())
    };
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "result type Float can't be cast to the desired output type {}",
            scalar_type_name(tag)
        )));
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let lhs = {
        let borrowed = receiver.borrow();
        borrowed.tensor()?.fast_to(storage).map_err(|e| candle_err(OP, e))?
    };
    let rhs = other
        .tensor()?
        .fast_to(storage)
        .and_then(|t| t.broadcast_as(shape))
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(OP, e))?;
    let out = lhs.broadcast_div(&rhs).map_err(|e| candle_err(OP, e))?;
    write_back(OP, &receiver, PyTensorBase::new(out)?)?;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)`
///
/// `mixtral`'s grouped-MoE routing zeroes sentinel rows twice, once each
/// side of the grouped matmul: `selected_hidden_states_g.masked_fill_(
/// sentinel_mask, 0.0)` and `weighted_out.masked_fill_(sentinel_mask, 0.0)`
/// (`transformers`' `integrations/moe.py`) -- measured
/// `masked_fill_.Scalar(float32(12,64), bool(12,1), 0.0)`, the mask
/// broadcasting from `(12,1)` into the receiver's `(12,64)`.
///
/// Not a new kernel: the value and every refusal are `masked_fill.Scalar`'s
/// (a `torch.bool` mask required, same as that op's doc comment measures),
/// computed once and written into the receiver through its layout by
/// `write_back` -- so an alias or view taken before this call sees the write,
/// as it does upstream. docs/VIEWS.md §6.
///
/// It is one of the four keys `write_back` lets write into an *expanded*
/// destination, because upstream does (with a deprecation warning) where it
/// raises for `copy_`/`add_`/`clamp_`/`div_`. Measured, not derived.
fn masked_fill_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
) -> PyResult<Py<PyAny>> {
    let receiver = tensor_receiver(op, args, kwargs)?;
    let result = masked_fill(py, args, kwargs, op)?;
    let replacement = result.extract::<PyTensorBase>(py)?;
    write_back(op, &receiver, replacement)?;
    Ok(receiver.into_any().unbind())
}

/// `aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool
///     accumulate=False) -> Tensor(a!)`
///
/// `mixtral`'s grouped-MoE routing builds the inverse of a sort permutation
/// with `inv_perm[perm] = torch.arange(perm.size(0))` (`transformers`'
/// `integrations/moe.py`), which lowers to `index_put_` with a single index
/// tensor -- measured `index_put_.default(int64(12,), [int64(12,)],
/// int64(12,), accumulate=False (default, absent))`.
///
/// **One non-`None` index tensor**, either `accumulate`. Within that, `self`
/// may have any rank, the index may have any shape, the index may be a
/// boolean (or `uint8`) mask covering several axes, and `values` broadcasts
/// against the result the way upstream broadcasts it. A second index tensor
/// is still refused by name.
///
/// The first version delegated to `scatter.src` along dimension 0, which is
/// exactly right for the one call Mixtral makes and wrong for everything
/// else: `scatter` wants an int32/int64 index and index/src/self all of the
/// same rank, so it refused a bool mask (`Expected dtype int32 or int64 for
/// index, got bool`) and refused a matrix receiver. Both refusals were
/// recorded as gaps in docs/GROUPED_MM.md §6.4 and both are closed here, by
/// doing the address arithmetic directly instead of borrowing another op's.
///
/// **Everything below was measured against torch 2.13.0, not recalled.**
///
///   * **A mask is not a cast.** `x[bool_mask] = v` selects the positions
///     where the mask is true; the natural lowering is mask -> coordinates,
///     which is upstream's own move (`at::native::expandTensors`) and is
///     already implemented here as `mask_to_indices` for `index.Tensor`. A
///     `k`-dimensional mask consumes `k` axes and contributes a single axis
///     of length `count` to the result, so `x(2,3)[mask(2,3)] = [1,2,3]`
///     writes three elements and `x(2,3)[mask(2,)] = [1,2,3]` writes a whole
///     row. Reading the mask as an integer index instead would be wrong with
///     a plausible shape, which is why this is a different operation rather
///     than a dtype conversion.
///   * **The indexing result shape.** With one index group at axis `a`
///     consuming `m` axes, it is `dims[..a] ++ index_shape ++ dims[a+m..]`.
///     Only one group can exist here, so `index.Tensor`'s fronting-versus-
///     splicing rule (which needs two separated groups to matter) cannot
///     apply and the shape is always the spliced one. Measured:
///     `index_put_(zeros(2,3), [None, tensor([0,2])], v)` wants `v`
///     broadcastable to `(2,2)`.
///   * **`values` broadcasts, right-aligned**, and a value that does not fit
///     raises `shape mismatch: value tensor of shape [...] cannot be
///     broadcast to indexing result of shape [...]`. A 0-d `values` is how
///     `x[idx] = 5` arrives, so this is the path gap 3 needed.
///   * **dtypes must match exactly.** Upstream does not promote here:
///     `Index put requires the source and destination dtypes match, got
///     Float for the destination and Long for the source.` Kept, with
///     upstream's wording.
///   * **Negative indices wrap**, and out-of-range ones raise
///     `index N is out of bounds for dimension A with size E`. The
///     `scatter`-based version could not wrap, because `scatter` has no
///     negative-index rule.
///   * **An empty index or an all-false mask writes nothing** and returns
///     `self` unchanged, after the broadcast check has still been made.
///   * **Repeated positions: last write wins**, in row-major order over the
///     result shape. Same rule the `scatter` version had, re-derived rather
///     than inherited.
///   * **`accumulate=True` adds instead of overwriting**, so a repeated
///     position gets the *sum* of its contributions rather than the last one
///     (`zeros(5)` at `[0, 2, 2, 4]` with `[1, 2, 3, 4]` is
///     `[1, 0, 5, 0, 4]`). Everything else -- the mask lowering, the
///     broadcast, negative indices, the dtype check, the empty-write
///     shortcut -- is the same code and the same rules; only the assignment
///     in the walk differs. Two things about it are *not* the obvious
///     spelling and were measured rather than assumed:
///       * the addition runs at the receiver's precision, not at
///         `read_flat`'s `f64` (see `float_narrower` at the walk); and
///       * `torch.bool` accumulates as a logical or, because upstream's
///         `*dst += *src` on a C++ `bool` promotes and converts back.
///     This is what an embedding's backward wants -- a scatter-add into a
///     zero buffer -- and docs/BACKWARD.md §4.5 records the one-hot
///     composition it currently uses instead, at 200 MB for `S=1024`.
///

/// The write goes back into the receiver through `write_back`, which puts it
/// into the buffer the receiver already points at rather than swapping the
/// wrapper -- so an alias or a view created before the call does see it, as
/// upstream's does. That was §4 of docs/VIEWS.md's open question and §6 is
/// the answer.
///
/// The kernel itself did not change for it, and that is the useful part: it
/// already built a whole `dims`-shaped replacement out of `read_flat`, and a
/// whole replacement of the receiver's shape is exactly what write-through
/// consumes. docs/VIEWS.md §6.1 has the argument that this shape was never
/// the obstacle §4 recorded it as being.
fn index_put_inplace(
    _py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.index_put_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let raw_indices = required(OP, args, kwargs, 1, "indices")?;
    let items: Vec<Bound<'_, PyAny>> = raw_indices.extract()?;
    let values = tensor_arg(OP, args, kwargs, 2, "values")?;
    let accumulate = bool_arg(args, kwargs, 3, "accumulate")?.unwrap_or(false);

    let (tag, dims, device) = {
        let borrowed = receiver.borrow();
        (
            borrowed.tag(),
            borrowed.tensor()?.dims().to_vec(),
            borrowed.tensor()?.device().clone(),
        )
    };
    let rank = dims.len();
    if items.len() > rank {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "too many indices for tensor of dimension {rank} (got {})",
            items.len()
        )));
    }
    if values.tag() != tag {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Index put requires the source and destination dtypes match, got {} for the \
             destination and {} for the source.",
            scalar_type_name(tag),
            scalar_type_name(values.tag())
        )));
    }

    // Row-major strides of `self`, so an index coordinate can be turned into
    // a flat offset once here rather than once per written element.
    let self_strides = contiguous_strides(&dims);

    // The single index group: where it starts, how many axes it eats, the
    // shape it contributes to the result, and one flat within-`self` offset
    // per position it names.
    let mut at_axis = 0usize;
    let mut group: Option<(usize, usize, Vec<usize>, Vec<usize>)> = None;
    for item in &items {
        if item.is_none() {
            at_axis += 1;
            continue;
        }
        let tensor = item.extract::<PyTensorBase>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "{OP}: indices must be tensors or None, got {}",
                item.get_type().name().map(|n| n.to_string()).unwrap_or_default()
            ))
        })?;
        if group.is_some() {
            return Err(not_implemented(format!(
                "{OP}: more than one index tensor is not implemented in torch._C shim"
            )));
        }
        match tensor.tag() {
            TorchDType::Int64 | TorchDType::Int32 => {
                let extent = dims[at_axis] as i64;
                let raw = match read_flat(OP, tensor.tensor()?, tensor.tag())? {
                    Flat::Int(v) => v,
                    Flat::Float(_) => unreachable!("the index dtype was matched above"),
                };
                let mut offsets = Vec::with_capacity(raw.len());
                for value in raw {
                    // torch wraps a negative index here; `scatter` does not,
                    // which is one of the reasons this no longer delegates.
                    let resolved = if value < 0 { value + extent } else { value };
                    if resolved < 0 || resolved >= extent {
                        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                            "index {value} is out of bounds for dimension {at_axis} \
                             with size {extent}"
                        )));
                    }
                    offsets.push(resolved as usize * self_strides[at_axis]);
                }
                group = Some((at_axis, 1, tensor.tensor()?.dims().to_vec(), offsets));
                at_axis += 1;
            }
            TorchDType::Bool | TorchDType::UInt8 => {
                // The mask -> indices lowering, shared with `index.Tensor`.
                // It also owns the shape check, so a mask that does not line
                // up raises upstream's `The shape of the mask ... does not
                // match ...` from one place rather than two.
                let consumed = tensor.tensor()?.rank();
                let expanded = mask_to_indices(OP, tensor.tensor()?, at_axis, &dims)?;
                let count = expanded.first().map_or(0, |(_, values, _)| values.len());
                let mut offsets = vec![0usize; count];
                for (axis, coords, _) in &expanded {
                    for (slot, &coord) in offsets.iter_mut().zip(coords.iter()) {
                        *slot += coord as usize * self_strides[*axis];
                    }
                }
                group = Some((at_axis, consumed, vec![count], offsets));
                at_axis += consumed;
            }
            _ => {
                // Upstream's own wording, and it names four dtypes rather
                // than the one that was passed.
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "tensors used as indices must be long, int, byte or bool tensors",
                ));
            }
        }
    }
    let (at, consumed, index_shape, offsets) = group.ok_or_else(|| {
        not_implemented(format!("{OP}: an all-None index list is not implemented in torch._C shim"))
    })?;

    // `dims[..at] ++ index_shape ++ dims[at+consumed..]` -- what upstream's
    // error message calls "the indexing result".
    let index_rank = index_shape.len();
    let mut result_shape: Vec<usize> = dims[..at].to_vec();
    result_shape.extend(index_shape.iter().copied());
    result_shape.extend(dims[at + consumed..].iter().copied());
    let result_rank = result_shape.len();

    // `values` broadcast onto that shape, right-aligned: its own stride where
    // an axis matches, zero where it is being stretched. Checked before the
    // empty-write shortcut below, because upstream checks it there too.
    let value_dims = values.tensor()?.dims().to_vec();
    let broadcast_failure = || {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "shape mismatch: value tensor of shape {value_dims:?} cannot be broadcast \
             to indexing result of shape {result_shape:?}"
        ))
    };
    if value_dims.len() > result_rank {
        return Err(broadcast_failure());
    }
    let value_contig = contiguous_strides(&value_dims);
    let align = result_rank - value_dims.len();
    let mut value_strides = vec![0usize; result_rank];
    for (k, &extent) in value_dims.iter().enumerate() {
        if extent == result_shape[align + k] {
            value_strides[align + k] = value_contig[k];
        } else if extent != 1 {
            return Err(broadcast_failure());
        }
    }

    let total: usize = result_shape.iter().product();
    if total == 0 {
        // An empty index, an all-false mask, or a zero-extent axis. Upstream
        // writes nothing and hands `self` back; so does this, and doing it
        // here keeps the walk below free of a degenerate case.
        return Ok(receiver.into_any().unbind());
    }

    // Destination stride per *result* axis. The index group's axes are not
    // linear -- they are a lookup into `offsets` -- so they stay at zero and
    // are added separately.
    let mut dest_strides = vec![0usize; result_rank];
    dest_strides[..at].copy_from_slice(&self_strides[..at]);
    for j in 0..(rank - at - consumed) {
        dest_strides[at + index_rank + j] = self_strides[at + consumed + j];
    }
    let group_strides = contiguous_strides(&index_shape);

    let mut out = {
        let borrowed = receiver.borrow();
        read_flat(OP, borrowed.tensor()?, tag)?
    };
    let source = read_flat(OP, values.tensor()?, tag)?;

    // `accumulate=True` adds where the default overwrites, and the addition
    // happens **in the receiver's own dtype**, not in the `f64` `read_flat`
    // hands over. Measured on 2.13.0: 64 accumulations of `bfloat16(0.01)`
    // into one position give `0.65234375`, which is the `bfloat16` running
    // sum; accumulating in `f64` and narrowing once at the end gives
    // `0.640625`. Same argument as `float_narrower`'s own doc comment, and
    // the same function is used for it.
    let narrow = float_narrower(tag);
    // `torch.bool` is the one dtype where `+` is not addition. Upstream's
    // kernel is `*dst += *src` in C++'s `scalar_t`, so for `bool` the sum
    // integer-promotes and converts back -- i.e. a logical or, measured:
    // `zeros(4, bool)` accumulated at `[0, 0, 1]` with `[True, True, True]`
    // is `[True, True, False, False]`, not a 2 anywhere. Writing `o + s`
    // here would put a `2` in a `bool` buffer and break the invariant
    // docs/BOOL.md §6.3 attaches to the tag.
    let is_bool = tag == TorchDType::Bool;

    let mut coord = vec![0usize; result_rank];
    for _ in 0..total {
        let mut dest = 0usize;
        let mut src = 0usize;
        for d in 0..result_rank {
            dest += coord[d] * dest_strides[d];
            src += coord[d] * value_strides[d];
        }
        let mut group_flat = 0usize;
        for j in 0..index_rank {
            group_flat += coord[at + j] * group_strides[j];
        }
        dest += offsets[group_flat];
        match (&source, &mut out) {
            (Flat::Float(s), Flat::Float(o)) => {
                o[dest] = if accumulate { narrow(o[dest] + s[src]) } else { s[src] }
            }
            (Flat::Int(s), Flat::Int(o)) => {
                o[dest] = if !accumulate {
                    s[src]
                } else if is_bool {
                    i64::from(o[dest] != 0 || s[src] != 0)
                } else {
                    // `uint8` overflow wraps upstream (`200 + 100 + 100` is
                    // `144`, measured), and the narrowing to the storage
                    // width happens in `write_flat`; `wrapping_add` keeps the
                    // i64 accumulator from panicking on the way there.
                    o[dest].wrapping_add(s[src])
                }
            }
            _ => unreachable!("self and values share a dtype, checked above"),
        }
        for d in (0..result_rank).rev() {
            coord[d] += 1;
            if coord[d] < result_shape[d] {
                break;
            }
            coord[d] = 0;
        }
    }

    let tensor = write_flat(OP, out, dims, &device, tag)?;
    let replacement = if tag == TorchDType::Bool {
        PyTensorBase::boolean(tensor)?
    } else {
        PyTensorBase::new(tensor)?
    };
    write_back(OP, &receiver, replacement)?;
    Ok(receiver.into_any().unbind())
}

/// `aten::native_layer_norm(Tensor input, SymInt[] normalized_shape,
///     Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)`
///
/// GPT-2's normalisation, and structurally not Llama's: RMSNorm is
/// `mean.dim` + `rsqrt` + `mul` through the ordinary dispatcher, while
/// `LayerNorm` is **one fused op that returns three tensors** -- the output
/// plus the `mean` and `rstd` the backward pass would need. There is no
/// autograd here, but the two extra results are part of the schema, so they are
/// computed and returned rather than filled with zeros.
///
/// Measured against torch 2.13.0. The parts that inference gets wrong:
///
///   * **`mean`/`rstd` are not flat.** They keep the input's rank with the
///     normalised dimensions replaced by 1: `(2,3,4)` with
///     `normalized_shape=[4]` gives `(2,3,1)`, and with `[3,4]` gives
///     `(2,1,1)`.
///   * **the variance is biased** (divided by N, not N-1), and `eps` is added
///     to it *before* the reciprocal square root, not to the standard
///     deviation. A constant row therefore gives `rstd = 1/sqrt(eps)`
///     (`316.2278` at `eps=1e-5`), which is what pins the order.
///   * **`mean`/`rstd` follow the *parameter* dtype, not the input's.** A
///     `float16` input with `float16` (or absent) parameters gives `float16`
///     statistics; the same input with `float32` parameters gives `float32`
///     ones while the output stays `float16`. That is upstream's mixed-dtype
///     autocast path, and it is a *supported* combination, not an error.
///   * a **negative `eps` is not refused** -- it gives NaN, and this follows.
///
/// Refusals copied rather than invented: an integral or boolean input raises
/// `NotImplementedError` naming `LayerNormKernelImpl`; parameters that are
/// neither the input dtype nor the `float32` autocast partner raise
/// `mixed dtype (CPU): ...`, in two different wordings depending on whether the
/// input was `float64`.
///
/// **Not implemented: a zero-extent normalized dimension.** `normalized_shape=[0]`
/// makes upstream answer `mean=0, rstd=nan` -- a mean over no elements that is
/// zero on one side of the pair and NaN on the other. Reproducing an internal
/// inconsistency from one observation is guessing, so it is refused by name.
fn native_layer_norm_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.native_layer_norm.default";
    let input = tensor_arg(OP, args, kwargs, 0, "input")?;
    let normalized = shape_arg(OP, args, kwargs, 1, "normalized_shape")?;
    let weight = optional_tensor_arg(OP, args, kwargs, 2, "weight")?;
    let bias = optional_tensor_arg(OP, args, kwargs, 3, "bias")?;
    let eps = scalar_arg(OP, args, kwargs, 4, "eps")?
        .map(|s| s.as_f64())
        .ok_or_else(|| missing(OP, "eps"))?;

    if normalized.is_empty() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Expected normalized_shape to be at least 1-dimensional, i.e., containing at \
             least one element, but got normalized_shape = []",
        ));
    }

    let dims = input.tensor()?.dims().to_vec();
    let k = normalized.len();
    let suffix_matches = dims.len() >= k
        && dims[dims.len() - k..]
            .iter()
            .zip(normalized.iter())
            .all(|(&extent, &wanted)| wanted >= 0 && extent == wanted as usize);
    if !suffix_matches {
        let star: String = normalized.iter().map(|v| format!(", {v}")).collect();
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Given normalized_shape={normalized:?}, expected input with shape [*{star}], \
             but got input of size{dims:?}"
        )));
    }
    let ns: Vec<usize> = normalized.iter().map(|&v| v as usize).collect();

    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"LayerNormKernelImpl\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }

    for (label, param) in [("weight", &weight), ("bias", &bias)] {
        if let Some(param) = param {
            if param.tensor()?.dims() != ns.as_slice() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Expected {label} to be of same shape as normalized_shape, but got \
                     {label} of shape {:?} and normalized_shape = {ns:?}",
                    param.tensor()?.dims()
                )));
            }
        }
    }

    // Upstream's rule, measured: the parameters agree with each other, and
    // they are either the input dtype or `float32` standing in front of a
    // reduced-precision input.
    let mixed_dtype = || {
        pyo3::exceptions::PyRuntimeError::new_err(if tag == TorchDType::Float64 {
            "mixed dtype (CPU): all inputs must share same datatype."
        } else {
            "mixed dtype (CPU): expect parameter to have scalar type of Float"
        })
    };
    let param_tag = match (&weight, &bias) {
        (Some(w), Some(b)) if w.tag() != b.tag() => return Err(mixed_dtype()),
        (Some(w), _) => Some(w.tag()),
        (None, Some(b)) => Some(b.tag()),
        (None, None) => None,
    };
    let mixed = match param_tag {
        None => false,
        Some(param) if param == tag => false,
        Some(TorchDType::Float32)
            if matches!(tag, TorchDType::Float16 | TorchDType::BFloat16) =>
        {
            true
        }
        Some(_) => return Err(mixed_dtype()),
    };

    let rows: usize = dims[..dims.len() - k].iter().product();
    let cols: usize = ns.iter().product();
    if cols == 0 {
        return Err(not_implemented(format!(
            "{OP}: a zero-extent normalized_shape ({ns:?}) is not implemented in torch._C \
             shim -- upstream answers mean=0 with rstd=nan there, and that pair was not \
             measured well enough to reproduce"
        )));
    }

    let storage = PyDtype::new(tag).storage(OP)?;
    // `opmath_type`: the reduced dtypes accumulate in `f32` and narrow once.
    let acc = match storage {
        candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
        other => other,
    };
    let stat_tag = if mixed { TorchDType::Float32 } else { tag };
    let stat_storage = PyDtype::new(stat_tag).storage(OP)?;
    let stat_dims: Vec<usize> = dims[..dims.len() - k]
        .iter()
        .copied()
        .chain(std::iter::repeat(1).take(k))
        .collect();
    let device = input.tensor()?.device().clone();

    // No rows to reduce over: every result is empty, and candle's reductions
    // have nothing to say about a zero-length axis.
    if rows == 0 {
        let empty = |shape: &[usize], dtype| {
            Tensor::zeros(shape, dtype, &device).map_err(|e| candle_err(OP, e))
        };
        let triple = [
            crate::tensor::promote(py, finish(py, empty(dims.as_slice(), storage)?, tag)?)?,
            crate::tensor::promote(
                py,
                finish(py, empty(stat_dims.as_slice(), stat_storage)?, stat_tag)?,
            )?,
            crate::tensor::promote(
                py,
                finish(py, empty(stat_dims.as_slice(), stat_storage)?, stat_tag)?,
            )?,
        ];
        return Ok(PyTuple::new(py, triple)?.into_any().unbind());
    }

    let flat = input
        .tensor()?
        .contiguous()
        .and_then(|t| t.fast_to(acc))
        .and_then(|t| t.reshape((rows, cols)))
        .map_err(|e| candle_err(OP, e))?;
    let mean = flat.mean_keepdim(1).map_err(|e| candle_err(OP, e))?;
    let centred = flat.broadcast_sub(&mean).map_err(|e| candle_err(OP, e))?;
    let rstd = centred
        .sqr()
        .and_then(|t| t.mean_keepdim(1))
        // `var + eps` first, *then* rsqrt -- see the doc comment.
        .and_then(|t| t.affine(1.0, eps))
        .and_then(|t| t.sqrt())
        .and_then(|t| t.recip())
        .map_err(|e| candle_err(OP, e))?;

    let mut out = centred.broadcast_mul(&rstd).map_err(|e| candle_err(OP, e))?;
    if let Some(weight) = &weight {
        let row = weight
            .tensor()?
            .contiguous()
            .and_then(|t| t.fast_to(acc))
            .and_then(|t| t.reshape((1, cols)))
            .map_err(|e| candle_err(OP, e))?;
        out = out.broadcast_mul(&row).map_err(|e| candle_err(OP, e))?;
    }
    if let Some(bias) = &bias {
        let row = bias
            .tensor()?
            .contiguous()
            .and_then(|t| t.fast_to(acc))
            .and_then(|t| t.reshape((1, cols)))
            .map_err(|e| candle_err(OP, e))?;
        out = out.broadcast_add(&row).map_err(|e| candle_err(OP, e))?;
    }

    let out = out
        .fast_to(storage)
        .and_then(|t| t.reshape(dims.as_slice()))
        .map_err(|e| candle_err(OP, e))?;
    let reshape_stat = |t: Tensor| {
        t.to_dtype(stat_storage)
            .and_then(|t| t.reshape(stat_dims.as_slice()))
            .map_err(|e| candle_err(OP, e))
    };
    let mean = reshape_stat(mean)?;
    let rstd = reshape_stat(rstd)?;

    // Promoted element by element: `promote` at the dispatcher's exit does not
    // look inside a tuple, the same reason `max.dim` promotes its own pair.
    let triple = [
        crate::tensor::promote(py, finish(py, out, tag)?)?,
        crate::tensor::promote(py, finish(py, mean, stat_tag)?)?,
        crate::tensor::promote(py, finish(py, rstd, stat_tag)?)?,
    ];
    Ok(PyTuple::new(py, triple)?.into_any().unbind())
}

/// `aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias,
///     SymInt N, SymInt C, SymInt HxW, int group, float eps)
///     -> (Tensor, Tensor, Tensor)`
///
/// `sew_d`'s wall. `nn.GroupNorm.forward` -> `F.group_norm` ->
/// `torch.group_norm`, which is `CompositeImplicitAutograd` and bottoms out
/// here -- measured with a `TorchDispatchMode` logger on 2.13.0: all three of
/// `torch.group_norm`, `F.group_norm` and an `nn.GroupNorm` forward emit
/// `aten.native_group_norm.default` and nothing else.
///
/// **The three results, and why the second and third are the dangerous ones.**
/// A forward only reads `out`, so `mean` and `rstd` can be the wrong shape,
/// the wrong dtype, or a different *definition* entirely and every model in
/// the sweep still runs. Each of the three was measured on its own:
///
///   * **`mean` and `rstd` are `(N, group)`** -- one statistic per (sample,
///     group), *not* per channel and not keepdim-shaped. This is the one place
///     `native_group_norm` differs in shape from `native_layer_norm` beside
///     it, which keeps the input's rank with 1s.
///   * **the variance is biased** (divided by n, not n-1), and `eps` is added
///     to the variance *before* the reciprocal square root. Both halves are
///     pinned by one measurement: a constant group gives
///     `rstd = 1/sqrt(eps) = 316.2278` at `eps=1e-5`. Adding `eps` to the
///     standard deviation instead would give `1/eps = 100000`, and an
///     unbiased variance over a constant group is still zero -- so the
///     constant case separates the `eps` placement while a random case
///     separates the divisor, and neither one alone does both.
///   * **`rstd` is a reciprocal**, not a standard deviation. `1/sqrt(v+eps)`
///     and `sqrt(v+eps)` have the same shape and the same dtype and differ
///     only in the numbers -- docs/KERNELS26.md's "a wrong answer that has the
///     right shape", in the result no forward reads.
///
/// **The normalisation axes are not the weight axis.** The statistics are
/// taken over `(C/group) * HxW` elements per row -- the tensor read as
/// `(N*group, C/group*HxW)` -- while `weight` and `bias` are per **channel**,
/// shape `(C,)`, applied after the normalised tensor is reshaped back to
/// `(N, C, HxW)`. Folding those two views into one is the plausible error
/// here, and it is invisible in the two configurations a hand-written test
/// reaches for first: with `group == C` (InstanceNorm) or `group == 1`
/// (LayerNorm over C,H,W) the two views coincide. The cases use `C=6,
/// group=3` so that they do not.
///
/// Dtype, measured, and the same mixed-precision rule `native_layer_norm` has:
/// `mean`/`rstd` follow the **parameter** dtype, so a `float16` input with
/// `float32` parameters gives `float32` statistics and a `float16` output,
/// while `float16` parameters give `float16` statistics. A `float32` input
/// with any other parameter dtype raises `mixed dtype (CPU): expect parameter
/// to have scalar type of Float`.
///
/// The refusals are upstream's, transcribed:
///
/// ```text
/// int64 / bool input           "GroupNormKernelImpl" not implemented for 'Long' / 'Bool'
/// C % group != 0               Expected number of channels in input to be divisible by num_groups
/// N * C * HxW != numel         Expected X.numel() == N * C * HxW to be true, but got false. ...
/// weight.shape != [C]          Expected weight to be a vector of size equal to the number of channels
/// group <= 0                   Expected num groups to be greater than 0, got 0
/// ```
///
/// The divisibility check runs **before** the element-count check, which is
/// measured rather than chosen: a wrong `C` that happens to be indivisible
/// reports the divisibility message and not the count one.
///
/// A **negative `eps` is not refused** -- it gives NaN wherever `var + eps` is
/// negative and a finite answer elsewhere, and this follows rather than
/// guarding, exactly as `native_layer_norm` does.
///
/// **Not implemented: `HxW == 0`.** Upstream answers `mean=0` with `rstd=nan`
/// there -- one half of the pair reporting an empty reduction and the other
/// not. That is the same internally-inconsistent corner `native_layer_norm`
/// refuses for a zero-extent `normalized_shape`, and it is refused here for
/// the same reason and by name. `N == 0` *is* implemented: every result is
/// simply empty, with no inconsistency to reproduce.
fn native_group_norm_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.native_group_norm.default";
    let input = tensor_arg(OP, args, kwargs, 0, "input")?;
    let weight = optional_tensor_arg(OP, args, kwargs, 1, "weight")?;
    let bias = optional_tensor_arg(OP, args, kwargs, 2, "bias")?;
    let n = int_arg(args, kwargs, 3, "N")?.ok_or_else(|| missing(OP, "N"))?;
    let c = int_arg(args, kwargs, 4, "C")?.ok_or_else(|| missing(OP, "C"))?;
    let hxw = int_arg(args, kwargs, 5, "HxW")?.ok_or_else(|| missing(OP, "HxW"))?;
    let group = int_arg(args, kwargs, 6, "group")?.ok_or_else(|| missing(OP, "group"))?;
    let eps = scalar_arg(OP, args, kwargs, 7, "eps")?
        .map(|s| s.as_f64())
        .ok_or_else(|| missing(OP, "eps"))?;

    let dims = input.tensor()?.dims().to_vec();
    let count_error = || {
        pyo3::exceptions::PyRuntimeError::new_err(
            "Expected X.numel() == N * C * HxW to be true, but got false.  (Could this \
             error message be improved?  If so, please report an enhancement request to \
             PyTorch.)",
        )
    };
    if group <= 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Expected num groups to be greater than 0, got {group}"
        )));
    }
    if n < 0 || c < 0 || hxw < 0 {
        return Err(count_error());
    }
    // Divisibility before the element count: measured, and it shows in the
    // message a caller gets for a wrong `C`.
    if c % group != 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Expected number of channels in input to be divisible by num_groups, but got \
             input of shape {dims:?} and num_groups={group}"
        )));
    }
    if (n as i128) * (c as i128) * (hxw as i128) != input.tensor()?.elem_count() as i128 {
        return Err(count_error());
    }

    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"GroupNormKernelImpl\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    for (label, param) in [("weight", &weight), ("bias", &bias)] {
        if let Some(param) = param {
            if param.tensor()?.dims() != [c as usize] {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Expected {label} to be a vector of size equal to the number of \
                     channels in input, but got {label} of shape {:?} and input of shape \
                     {dims:?}",
                    param.tensor()?.dims()
                )));
            }
        }
    }

    // `native_layer_norm`'s rule, re-measured here rather than assumed from
    // it: the parameters agree with each other, and they are either the input
    // dtype or `float32` in front of a reduced-precision one.
    let mixed_dtype = || {
        pyo3::exceptions::PyRuntimeError::new_err(
            "mixed dtype (CPU): expect parameter to have scalar type of Float",
        )
    };
    let param_tag = match (&weight, &bias) {
        (Some(w), Some(b)) if w.tag() != b.tag() => return Err(mixed_dtype()),
        (Some(w), _) => Some(w.tag()),
        (None, Some(b)) => Some(b.tag()),
        (None, None) => None,
    };
    let mixed = match param_tag {
        None => false,
        Some(param) if param == tag => false,
        Some(TorchDType::Float32)
            if matches!(tag, TorchDType::Float16 | TorchDType::BFloat16) =>
        {
            true
        }
        Some(_) => return Err(mixed_dtype()),
    };

    let n = n as usize;
    let c = c as usize;
    let hxw = hxw as usize;
    let group = group as usize;
    if hxw == 0 {
        return Err(not_implemented(format!(
            "{OP}: a zero-extent HxW is not implemented in torch._C shim -- upstream \
             answers mean=0 with rstd=nan there, one half of the pair reporting an empty \
             reduction and the other not, and reproducing an internal inconsistency from \
             one observation is guessing. `native_layer_norm` refuses a zero-extent \
             normalized_shape for the same reason"
        )));
    }

    let storage = PyDtype::new(tag).storage(OP)?;
    // `opmath_type`: the reduced dtypes accumulate in `f32` and narrow once.
    let acc = match storage {
        candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
        other => other,
    };
    let stat_tag = if mixed { TorchDType::Float32 } else { tag };
    let stat_storage = PyDtype::new(stat_tag).storage(OP)?;
    let device = input.tensor()?.device().clone();

    if n == 0 {
        let empty = |shape: &[usize], dtype| {
            Tensor::zeros(shape, dtype, &device).map_err(|e| candle_err(OP, e))
        };
        let triple = [
            crate::tensor::promote(py, finish(py, empty(dims.as_slice(), storage)?, tag)?)?,
            crate::tensor::promote(py, finish(py, empty(&[0, group], stat_storage)?, stat_tag)?)?,
            crate::tensor::promote(py, finish(py, empty(&[0, group], stat_storage)?, stat_tag)?)?,
        ];
        return Ok(PyTuple::new(py, triple)?.into_any().unbind());
    }

    // The statistics view: `(N*group, C/group * HxW)`. Deliberately not the
    // view the affine step below uses.
    let cols = (c / group) * hxw;
    let flat = input
        .tensor()?
        .contiguous()
        .and_then(|t| t.fast_to(acc))
        .and_then(|t| t.reshape((n * group, cols)))
        .map_err(|e| candle_err(OP, e))?;
    let mean = flat.mean_keepdim(1).map_err(|e| candle_err(OP, e))?;
    let centred = flat.broadcast_sub(&mean).map_err(|e| candle_err(OP, e))?;
    let rstd = centred
        .sqr()
        // The *biased* variance: candle's `mean` divides by n, which is what
        // upstream does. `var + eps` first, then rsqrt -- the constant-group
        // measurement in the doc comment is what pins that order.
        .and_then(|t| t.mean_keepdim(1))
        .and_then(|t| t.affine(1.0, eps))
        .and_then(|t| t.sqrt())
        .and_then(|t| t.recip())
        .map_err(|e| candle_err(OP, e))?;

    // The affine view: back to `(N, C, HxW)`, because `weight`/`bias` are per
    // channel and a channel is not a group.
    let mut out = centred
        .broadcast_mul(&rstd)
        .and_then(|t| t.reshape((n, c, hxw)))
        .map_err(|e| candle_err(OP, e))?;
    if let Some(weight) = &weight {
        let column = weight
            .tensor()?
            .contiguous()
            .and_then(|t| t.fast_to(acc))
            .and_then(|t| t.reshape((1, c, 1)))
            .map_err(|e| candle_err(OP, e))?;
        out = out.broadcast_mul(&column).map_err(|e| candle_err(OP, e))?;
    }
    if let Some(bias) = &bias {
        let column = bias
            .tensor()?
            .contiguous()
            .and_then(|t| t.fast_to(acc))
            .and_then(|t| t.reshape((1, c, 1)))
            .map_err(|e| candle_err(OP, e))?;
        out = out.broadcast_add(&column).map_err(|e| candle_err(OP, e))?;
    }

    let out = out
        .fast_to(storage)
        .and_then(|t| t.reshape(dims.as_slice()))
        .map_err(|e| candle_err(OP, e))?;
    let stat = |t: Tensor| {
        t.to_dtype(stat_storage)
            .and_then(|t| t.reshape((n, group)))
            .map_err(|e| candle_err(OP, e))
    };
    let mean = stat(mean)?;
    let rstd = stat(rstd)?;

    // Promoted element by element: `promote` at the dispatcher's exit does not
    // look inside a tuple, the same reason `native_layer_norm` promotes its own
    // triple.
    let triple = [
        crate::tensor::promote(py, finish(py, out, tag)?)?,
        crate::tensor::promote(py, finish(py, mean, stat_tag)?)?,
        crate::tensor::promote(py, finish(py, rstd, stat_tag)?)?,
    ];
    Ok(PyTuple::new(py, triple)?.into_any().unbind())
}

/// `optional_tensor_arg` for a slot the kernel is going to **write into**.
///
/// `optional_tensor_arg` extracts the wrapper by value, which is the right
/// thing for every argument that is only read. `native_batch_norm`'s
/// `running_mean`/`running_var` are not: they are mutated in place (see that
/// kernel's doc comment), and a mutation applied to an extracted copy would be
/// invisible to the caller. This keeps the `Bound`, the way `tensor_receiver`
/// does for argument 0 of an `_`-suffixed op, but at an arbitrary index and
/// admitting `None`.
fn optional_tensor_receiver<'py>(
    op: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<Bound<'py, PyTensorBase>>> {
    match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => Ok(Some(value.cast_into::<PyTensorBase>().map_err(
            |_| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "{op}: argument '{name}' must be a torch._C.TensorBase or None"
                ))
            },
        )?)),
        _ => Ok(None),
    }
}

/// `aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias,
///     Tensor? running_mean, Tensor? running_var, bool training,
///     float momentum, float eps) -> (Tensor, Tensor, Tensor)`
///
/// docs/DEMAND.md's **rank 1** -- `resnet` and `mobilenet_v2` both stop here,
/// one layer past a stem `conv2d` that now succeeds, and the same code shape
/// carries EfficientNet, RegNet and DenseNet. The most generic vision-CNN
/// primitive there is. Full measurement round in docs/DEMAND1.md §1; the parts
/// that decide the code are repeated here.
///
/// A leaf (`CompositeImplicitAutograd` is `False`); `aten::batch_norm` is the
/// composite above it, and a `TorchDispatchMode` logger on 2.13.0 shows
/// `torch.batch_norm`, `F.batch_norm` and an `nn.BatchNorm2d` forward all
/// bottom out here with `aten.batch_norm.default` never firing. So the
/// spelling is a `bootstrap.py` composite, exactly as `layer_norm` and
/// `group_norm` are.
///
/// # The train/eval split is two different functions wearing one schema
///
/// ```text
///                       training=True                    training=False
/// statistics for out    the batch's, BIASED variance     running_mean / running_var
/// save_mean             batch mean, shape (C,)           shape (0,)  -- EMPTY
/// save_invstd           1/sqrt(biased_var+eps), (C,)     shape (0,)  -- EMPTY
/// running_mean          WRITTEN IN PLACE                 untouched
/// running_var           WRITTEN IN PLACE                 untouched
/// ```
///
/// The empty `save_*` in eval mode is the detail no forward can see: a
/// `nn.BatchNorm2d` in `.eval()` returns `result[0]` and nothing else, so
/// handing back the running statistics there instead of two empty tensors
/// leaves every model in the sweep green. Measured shape `(0,)`, dtype the
/// parameter dtype.
///
/// # Biased for the output, UNBIASED for the running variance
///
/// The two halves disagree and both were pinned separately (DEMAND1.md §1.4):
///
/// ```text
/// save_invstd = 1 / sqrt(var_biased + eps)                    <- divide by n
/// running_var = (1-momentum)*running_var + momentum*var_unbiased   <- divide by n-1
/// running_mean = (1-momentum)*running_mean + momentum*batch_mean
/// ```
///
/// Pinned at `momentum=1.0`, where the old value drops out entirely and the
/// update is the new statistic exactly: measured `running_var` after is
/// `[0.12244898080825806, ...]`, which is the **unbiased** variance
/// (`0.10714286` is the biased one). Using the biased variance for the update
/// would give `1.3607143` where upstream gives `1.3622448444366455`, and no
/// test that reads only `out` can tell those apart.
///
/// `eps` goes on the **variance**, before the reciprocal square root -- pinned
/// by a constant channel, where `save_invstd` is `1/sqrt(1e-5) = 316.2278` and
/// `1/(sqrt(0)+eps)` would be `100000`. `eps` does **not** enter the
/// running-variance update.
///
/// # It mutates, and its schema does not say so
///
/// Measured: every argument's `alias_info` is `None`, so the schema promises
/// this op writes nothing. It writes. That is upstream's own known wart --
/// `_native_batch_norm_legit` exists beside it carrying the `Tensor(a!)`
/// annotations the functionaliser needs -- and it matters here because
/// `capture.rs` decides what mutates from **torch's name convention**, the
/// trailing `_`. This op has none. `capture.rs` grew an argument-aware guard
/// for it (see `MUTATES_WITHOUT_UNDERSCORE` there) that refuses a capture only
/// when the call would actually mutate: `training=True` with statistics
/// supplied. Eval-mode batch norm is genuinely pure, and it is the mode a
/// captured inference graph is in, so refusing it by name would have made
/// every BatchNorm CNN uncapturable to buy nothing.
///
/// # Axes, dtype and the refusals
///
/// The reduction is per **channel**: over dim 0 and every dim after 1. Checked
/// with exact values at rank 2 (`(N,C)`), rank 3 (`(N,C,L)`) and rank 4 --
/// `save_mean` for `arange(12).reshape(4,3)` is `[4.5, 5.5, 6.5]` and for
/// `arange(12).reshape(2,3,2)` is `[3.5, 5.5, 7.5]`, which the same code path
/// gets right only if `HxW` is derived rather than assumed to be rank-4.
///
/// Dtype, measured, and the same mixed-precision rule the other two norms have
/// with a **different message**: parameters must agree with the input, except
/// that a `float16`/`bfloat16` input may take `float32` parameters, and then
/// `save_mean`/`save_invstd` and the running statistics are `float32` while
/// `out` stays reduced. Anything else is `mixed dtype (CPU): all inputs must
/// share same datatype.` -- note that this is *not* `native_group_norm`'s
/// wording (`expect parameter to have scalar type of Float`), and the two were
/// measured separately rather than shared.
///
/// ```text
/// int64 / bool input          "batch_norm" not implemented for 'Long' / 'Bool'
/// rank < 2                    Dimension out of range (...) -- via normalise_dim on dim 1
/// exactly one of the stats    running_mean and running_var must either both be None or neither be None
/// training and numel == 0     input tensor must have at least one element, but got input_sizes = [...]
/// negative eps                NOT refused -- NaN, followed rather than guarded, as in the other two norms
/// one element per channel     NOT refused at this level; save_invstd is 1/sqrt(eps).
///                             `F.batch_norm`'s `_verify_batch_size` is what raises for a model,
///                             one layer up, in the vendored tree
/// ```
///
/// **Two places this kernel deliberately does not follow upstream**, both
/// because upstream there is not a behaviour but an accident (DEMAND1.md §1.6):
///
///   * `training=False` with both statistics `None` **segfaults** upstream
///     (exit 139, reproduced twice in isolation). Unreachable through any real
///     spelling -- `torch.batch_norm` raises first, and
///     `nn.BatchNorm2d(track_running_stats=False)` passes `training=True` in
///     `.eval()` precisely so this cannot happen. Raised here with the message
///     `torch.batch_norm` uses, because a crash is not an answer to reproduce.
///   * a `weight`/`bias`/`running_*` whose length is not `C` **reads out of
///     bounds** upstream. Measured with `C=3`: `weight` of length 2 and of
///     length 4 both return, and both return bit-identical numbers to the
///     correct length-3 call -- the short one having read a float past the end
///     of its buffer and found `1.5` there. That is uninitialised heap, not a
///     semantic. Refused here by name; `torch.batch_norm` refuses it above
///     with upstream's own message, which is what every real caller meets.
fn native_batch_norm_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.native_batch_norm.default";
    let input = tensor_arg(OP, args, kwargs, 0, "input")?;
    let weight = optional_tensor_arg(OP, args, kwargs, 1, "weight")?;
    let bias = optional_tensor_arg(OP, args, kwargs, 2, "bias")?;
    // Receivers, not extracted copies: these two are written in place.
    let running_mean = optional_tensor_receiver(OP, args, kwargs, 3, "running_mean")?;
    let running_var = optional_tensor_receiver(OP, args, kwargs, 4, "running_var")?;
    let training = bool_arg(args, kwargs, 5, "training")?.ok_or_else(|| missing(OP, "training"))?;
    let momentum = scalar_arg(OP, args, kwargs, 6, "momentum")?
        .map(|s| s.as_f64())
        .ok_or_else(|| missing(OP, "momentum"))?;
    let eps = scalar_arg(OP, args, kwargs, 7, "eps")?
        .map(|s| s.as_f64())
        .ok_or_else(|| missing(OP, "eps"))?;

    if running_mean.is_some() != running_var.is_some() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "running_mean and running_var must either both be None or neither be None",
        ));
    }

    let dims = input.tensor()?.dims().to_vec();
    // The channel axis. `normalise_dim` gives upstream's out-of-range message
    // for a rank-1 or rank-0 input for free, which is measured to be what a
    // direct `torch.ops.aten` call gets there.
    let channel = normalise_dim(OP, 1, dims.len())?;
    let c = dims[channel];
    let n = dims[0];
    let rest: usize = dims[2..].iter().product();

    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"batch_norm\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }

    // The parameter dtype: every supplied parameter and running statistic must
    // agree with the others, and then either match the input or be `float32`
    // in front of a reduced-precision one.
    let mixed_dtype = || {
        pyo3::exceptions::PyRuntimeError::new_err(
            "mixed dtype (CPU): all inputs must share same datatype.",
        )
    };
    let mut param_tag: Option<TorchDType> = None;
    let mut param_tags: Vec<TorchDType> = Vec::new();
    if let Some(w) = &weight {
        param_tags.push(w.tag());
    }
    if let Some(b) = &bias {
        param_tags.push(b.tag());
    }
    if let Some(m) = &running_mean {
        param_tags.push(m.borrow().tag());
    }
    if let Some(v) = &running_var {
        param_tags.push(v.borrow().tag());
    }
    for candidate in param_tags {
        match param_tag {
            None => param_tag = Some(candidate),
            Some(seen) if seen == candidate => {}
            Some(_) => return Err(mixed_dtype()),
        }
    }
    let mixed = match param_tag {
        None => false,
        Some(param) if param == tag => false,
        Some(TorchDType::Float32)
            if matches!(tag, TorchDType::Float16 | TorchDType::BFloat16) =>
        {
            true
        }
        Some(_) => return Err(mixed_dtype()),
    };

    // Length checks. Upstream reads out of bounds here (see the doc comment);
    // `torch.batch_norm` refuses above with upstream's own wording, so nothing
    // real arrives with a mismatch and this names the refusal rather than
    // guessing at heap bytes.
    for (label, extent) in [
        ("weight", weight.as_ref().map(|t| t.tensor().map(|t| t.dims().to_vec()))),
        ("bias", bias.as_ref().map(|t| t.tensor().map(|t| t.dims().to_vec()))),
        (
            "running_mean",
            running_mean.as_ref().map(|t| t.borrow().tensor().map(|t| t.dims().to_vec())),
        ),
        (
            "running_var",
            running_var.as_ref().map(|t| t.borrow().tensor().map(|t| t.dims().to_vec())),
        ),
    ] {
        if let Some(shape) = extent {
            let shape = shape?;
            if shape != [c] {
                return Err(not_implemented(format!(
                    "{OP}: {label} has shape {shape:?} where the input has {c} channel(s). \
                     Upstream does not refuse this -- it reads past the end of the buffer \
                     and returns whatever was there (measured: a length-2 weight against 3 \
                     channels returns bit-identical numbers to the correct length-3 call, \
                     having found the third float in uninitialised heap). \
                     `torch.batch_norm` refuses it above this layer with upstream's own \
                     message, which is what every real caller meets"
                )));
            }
        }
    }

    let numel: usize = dims.iter().product();
    if training && numel == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "input tensor must have at least one element, but got input_sizes = {dims:?}"
        )));
    }
    if !training && running_mean.is_none() {
        // Upstream segfaults here. `torch.batch_norm`'s own message, because a
        // crash is not a behaviour to reproduce and this is the refusal every
        // real caller would have met one layer up anyway.
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "running_mean must be defined in evaluation mode",
        ));
    }

    let stat_tag = if mixed { TorchDType::Float32 } else { tag };
    let storage = PyDtype::new(tag).storage(OP)?;
    let stat_storage = PyDtype::new(stat_tag).storage(OP)?;
    // `opmath_type`: the reduced dtypes accumulate in `f32` and narrow once,
    // the rule `native_group_norm` states next door.
    let acc = match stat_storage {
        candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
        other => other,
    };
    let device = input.tensor()?.device().clone();

    // `(N, C, rest)` -- `rest` derived, so rank 2, 3, 4 and 5 are one path.
    let x3 = input
        .tensor()?
        .contiguous()
        .and_then(|t| t.fast_to(acc))
        .and_then(|t| t.reshape((n, c, rest)))
        .map_err(|e| candle_err(OP, e))?;
    let column = |t: &Tensor| t.reshape((1, c, 1)).map_err(|e| candle_err(OP, e));
    let as_acc = |t: &PyTensorBase| {
        t.tensor()?
            .contiguous()
            .and_then(|t| t.fast_to(acc))
            .map_err(|e| candle_err(OP, e))
    };

    // `(mean, invstd)` as `(1, C, 1)` blocks, plus the flat `(C,)` statistics
    // the first two results are. The training branch computes them; the eval
    // branch reads them.
    let (mean_block, invstd_block, saved) = if training {
        let count = (n * rest) as f64;
        let mean = x3
            .sum_keepdim(0)
            .and_then(|t| t.sum_keepdim(2))
            .and_then(|t| t.affine(1.0 / count, 0.0))
            .map_err(|e| candle_err(OP, e))?;
        let centred = x3.broadcast_sub(&mean).map_err(|e| candle_err(OP, e))?;
        // The BIASED variance -- divided by n. This is the one the output uses.
        let var_biased = centred
            .sqr()
            .and_then(|t| t.sum_keepdim(0))
            .and_then(|t| t.sum_keepdim(2))
            .and_then(|t| t.affine(1.0 / count, 0.0))
            .map_err(|e| candle_err(OP, e))?;
        let invstd = var_biased
            .affine(1.0, eps)
            .and_then(|t| t.sqrt())
            .and_then(|t| t.recip())
            .map_err(|e| candle_err(OP, e))?;

        // The running statistics, written in place. `running_var` takes the
        // UNBIASED variance -- `biased * n/(n-1)` -- while `invstd` above kept
        // the biased one. `eps` is not in this update.
        if let (Some(rm), Some(rv)) = (&running_mean, &running_var) {
            let flat_mean = mean.reshape(c).map_err(|e| candle_err(OP, e))?;
            let var_unbiased = if count > 1.0 {
                var_biased.affine(count / (count - 1.0), 0.0)
            } else {
                // n == 1: upstream's `n/(n-1)` is a division by zero and the
                // result is not finite. Followed rather than guarded, the way
                // a negative `eps` is -- `F.batch_norm`'s `_verify_batch_size`
                // refuses this shape one layer up for every real caller.
                var_biased.affine(f64::INFINITY, 0.0)
            }
            .and_then(|t| t.reshape(c))
            .map_err(|e| candle_err(OP, e))?;
            for (receiver, update) in [(rm, &flat_mean), (rv, &var_unbiased)] {
                let old = as_acc(&receiver.borrow())?;
                let new = old
                    .affine(1.0 - momentum, 0.0)
                    .and_then(|t| t.add(&update.affine(momentum, 0.0)?))
                    .and_then(|t| t.fast_to(stat_storage))
                    .map_err(|e| candle_err(OP, e))?;
                write_back(OP, receiver, PyTensorBase::new(new)?)?;
            }
        }
        let flat = |t: &Tensor| {
            t.reshape(c)
                .and_then(|t| t.fast_to(stat_storage))
                .map_err(|e| candle_err(OP, e))
        };
        (mean.clone(), invstd.clone(), Some((flat(&mean)?, flat(&invstd)?)))
    } else {
        // Eval: the running statistics, unchanged, and two EMPTY results.
        let rm = as_acc(&running_mean.as_ref().expect("checked above").borrow())?;
        let rv = as_acc(&running_var.as_ref().expect("checked above").borrow())?;
        let invstd = rv
            .affine(1.0, eps)
            .and_then(|t| t.sqrt())
            .and_then(|t| t.recip())
            .map_err(|e| candle_err(OP, e))?;
        (column(&rm)?, column(&invstd)?, None)
    };

    // **Upstream's affine is fused, and the difference is measurable.** The
    // obvious spelling -- `(x - mean) * invstd * weight + bias` -- is exact for
    // a constant channel, giving `bias` back unchanged. Upstream does not do
    // that. It precomputes
    //
    //     alpha = invstd * weight            (weight absent => 1)
    //     beta  = bias - mean * alpha        (bias absent   => 0)
    //     out   = x * alpha + beta
    //
    // and the two forms disagree wherever `mean * alpha` is large against the
    // output, because that last line is a subtraction of two nearly equal
    // numbers. Measured on a constant input of 2.0 with `eps=1e-5`, where
    // `invstd = 316.2278`: upstream's `out` for `weight=1, bias=0.1` is
    // **0.0999755859375**, not `0.1`. That is `0.1` rounded to a multiple of
    // `2^-14`, which is exactly one `float32` ULP at magnitude 632 -- the
    // magnitude of the two cancelling terms. The same arithmetic accounts for
    // the other two channels of that measurement to the bit
    // (`-0.1999969482421875` at `weight=2, bias=-0.2`, `0.29998779296875` at
    // `weight=0.5, bias=0.3`).
    //
    // This was caught by the golden harness, not reasoned out in advance: the
    // constant-input case exists to pin where `eps` goes, and it failed on the
    // output instead. Everything with a real variance agrees to well inside
    // tolerance under either form, so no other case could have found it.
    let mut alpha = invstd_block;
    if let Some(weight) = &weight {
        alpha = alpha
            .broadcast_mul(&column(&as_acc(weight)?)?)
            .map_err(|e| candle_err(OP, e))?;
    }
    let mut beta = alpha
        .mul(&mean_block)
        .and_then(|t| t.neg())
        .map_err(|e| candle_err(OP, e))?;
    if let Some(bias) = &bias {
        beta = beta
            .broadcast_add(&column(&as_acc(bias)?)?)
            .map_err(|e| candle_err(OP, e))?;
    }
    let out = x3
        .broadcast_mul(&alpha)
        .and_then(|t| t.broadcast_add(&beta))
        .map_err(|e| candle_err(OP, e))?;

    let out = out
        .fast_to(storage)
        .and_then(|t| t.reshape(dims.as_slice()))
        .map_err(|e| candle_err(OP, e))?;

    let (save_mean, save_invstd) = match saved {
        Some(pair) => pair,
        None => {
            let empty = || Tensor::zeros(0, stat_storage, &device).map_err(|e| candle_err(OP, e));
            (empty()?, empty()?)
        }
    };

    // Promoted element by element: `promote` at the dispatcher's exit does not
    // look inside a tuple, the same reason the other two norms promote their
    // own triple.
    let triple = [
        crate::tensor::promote(py, finish(py, out, tag)?)?,
        crate::tensor::promote(py, finish(py, save_mean, stat_tag)?)?,
        crate::tensor::promote(py, finish(py, save_invstd, stat_tag)?)?,
    ];
    Ok(PyTuple::new(py, triple)?.into_any().unbind())
}

/// `aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[],
///     int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True,
///     int? divisor_override=None) -> Tensor`
///
/// `sew_d`'s wall after `sign`. Its encoder downsamples with
/// `nn.AvgPool1d(kernel_size=2, stride=2)`, and **`aten::avg_pool1d` is
/// `CompositeImplicitAutograd`**: measured with a `TorchDispatchMode` logger
/// on 2.13.0, `torch.avg_pool1d(x, 3, 2)` fires
///
/// ```text
/// aten.unsqueeze.default(-2)  ->  aten.avg_pool2d.default([1,3],[1,2])  ->  aten.squeeze.dim(-2)
/// ```
///
/// so the 1-D name is a `bootstrap.py` composite and this 2-D op is the leaf.
/// sew_d's own call arrives here as `avg_pool2d((1,32,1,39), [1,2], [1,2])`.
///
/// # The window, and the two boundaries that are not the same boundary
///
/// Upstream's rule, per output cell, transcribed:
///
/// ```text
/// start  = out_index * stride - padding
/// end    = min(start + kernel, extent + padding)      <- CLIPPED TO THE PADDED EXTENT
/// count  = (h_end - h_start) * (w_end - w_start)      <- computed BEFORE the next line
/// start  = max(start, 0)
/// end    = min(end, extent)                           <- now clipped to the REAL extent
/// divisor = divisor_override, else count if count_include_pad else the clipped area
/// ```
///
/// The two clips are different clips and the order between them is the whole
/// of `count_include_pad`. Measured on `arange(20).reshape(1,1,4,5)` with
/// `kernel=2, stride=2, padding=1`: the cell at `(0,1)` sums `1+2 = 3` and
/// divides by **4** with `count_include_pad=True` (`0.75`) and by **2**
/// without (`1.5`). Same sum, same window, two answers.
///
/// # `ceil_mode`
///
/// The output extent is `floor` or `ceil` of `(extent + 2*padding - kernel) /
/// stride`, plus one -- and with `ceil` there is a correction upstream applies
/// and a naive implementation does not: **if the last window starts at or past
/// the end of the padded input, drop it.** Measured on a `1x5` input with
/// `kernel=[1,2], stride=[1,2]`: `ceil` gives 3 columns where `floor` gives 2,
/// and the third column is `x[4]` alone divided by 1 -- because `end` is
/// clipped to `extent + padding = 5` while `start + kernel` is 6.
///
/// # Dtype
///
/// `float64`/`float32`/`float16`/`bfloat16` and **`int64`** compute;
/// `int32`, `int16`, `int8`, `uint8` and `bool` all raise
/// `"avg_pool2d" not implemented for '<Type>'` -- measured one dtype at a
/// time, because "integral is supported" would have been the wrong summary:
/// `int64` alone is.
///
/// The integral path **truncates toward zero**, it does not round or floor:
/// measured, a window summing `11` over 4 elements gives `2`, and one summing
/// `-11` gives `-2` (floor would give `-3`).
///
/// `opmath_t`, measured in both directions the way
/// `upsample_bilinear2d_default` above needed:
///
/// ```text
/// float16/bfloat16 accumulated in f32 and narrowed once   max relative 0.0 vs upstream
/// float32          accumulated in f64 and narrowed once   max relative 1.43e-05  -- WORSE
/// ```
///
/// `1.43e-05` is past this repository's `float32` golden tolerance, so `f32`
/// is accumulated in `f32`; the reduced dtypes are bit-identical through `f32`.
///
/// # Refusals, upstream's own wording
///
/// ```text
/// padding > kernel/2   pad should be at most half of effective kernel size, ...
/// stride == 0          stride should not be zero
/// divisor_override 0   divisor must be not zero
/// output extent <= 0   Given input size: (...). Calculated output size: (...). Output size is too small
/// rank not 3 or 4      non-empty 3D or 4D (batch mode) tensor expected for input
/// ```
fn avg_pool2d_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.avg_pool2d.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let kernel = shape_arg(OP, args, kwargs, 1, "kernel_size")?;
    let stride_raw = match optional(args, kwargs, 2, "stride")? {
        Some(value) if !value.is_none() => shape_arg(OP, args, kwargs, 2, "stride")?,
        _ => Vec::new(),
    };
    let padding = match optional(args, kwargs, 3, "padding")? {
        Some(value) if !value.is_none() => shape_arg(OP, args, kwargs, 3, "padding")?,
        _ => vec![0],
    };
    let ceil_mode = bool_arg(args, kwargs, 4, "ceil_mode")?.unwrap_or(false);
    let count_include_pad = bool_arg(args, kwargs, 5, "count_include_pad")?.unwrap_or(true);
    let divisor_override = int_arg(args, kwargs, 6, "divisor_override")?;

    // `int[2]` accepts a single value meaning "both axes", which is how
    // upstream's `padding=0` default is spelled in the schema itself.
    let pair = |values: &[isize], name: &str| -> PyResult<(i64, i64)> {
        match values.len() {
            1 => Ok((values[0] as i64, values[0] as i64)),
            2 => Ok((values[0] as i64, values[1] as i64)),
            n => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "avg_pool2d: {name} must either be a single int, or a tuple of two ints \
                 (got {n})"
            ))),
        }
    };
    let (kh, kw) = pair(&kernel, "kernel_size")?;
    // "stride=[]" means "the kernel size", which is not the same as "1".
    let (sh, sw) = if stride_raw.is_empty() {
        (kh, kw)
    } else {
        pair(&stride_raw, "stride")?
    };
    let (ph, pw) = pair(&padding, "padding")?;

    let dims = input.tensor()?.dims().to_vec();
    if dims.len() != 3 && dims.len() != 4 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "non-empty 3D or 4D (batch mode) tensor expected for input, but got: {dims:?}"
        )));
    }
    if kh <= 0 || kw <= 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "avg_pool2d: kernel_size must be greater than zero",
        ));
    }
    if sh == 0 || sw == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "stride should not be zero",
        ));
    }
    if ph < 0 || pw < 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "avg_pool2d: padding must be non-negative",
        ));
    }
    if 2 * ph > kh || 2 * pw > kw {
        let (pad, size) = if 2 * ph > kh { (ph, kh) } else { (pw, kw) };
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "pad should be at most half of effective kernel size, but got pad={pad}, \
             kernel_size={size} and dilation=1"
        )));
    }
    if divisor_override == Some(0) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "divisor must be not zero",
        ));
    }

    let tag = input.tag();
    let integral = !tag.is_floating_point();
    if integral && tag != TorchDType::Int64 {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"avg_pool2d\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }

    let split = dims.len() - 2;
    let planes: usize = dims[..split].iter().product();
    let ih = dims[split] as i64;
    let iw = dims[split + 1] as i64;

    // `pooling_output_shape`: floor or ceil, with the "the last window must
    // start inside the padded input" correction that only `ceil_mode` can
    // trigger.
    let extent = |input_size: i64, k: i64, pad: i64, stride: i64| -> i64 {
        let numerator = input_size + 2 * pad - k;
        let mut out = if ceil_mode {
            numerator.div_euclid(stride) + if numerator.rem_euclid(stride) != 0 { 1 } else { 0 }
        } else {
            numerator.div_euclid(stride)
        } + 1;
        if ceil_mode && (out - 1) * stride >= input_size + pad {
            out -= 1;
        }
        out
    };
    let oh = extent(ih, kh, ph, sh);
    let ow = extent(iw, kw, pw, sw);
    if oh <= 0 || ow <= 0 {
        // Upstream names the *channel* count on both sides of the message,
        // which is `dims[split - 1]` for a rank-4 `(N,C,H,W)` and for a
        // rank-3 `(C,H,W)` alike -- the batch axis is not in it.
        let channels = dims[split - 1];
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Given input size: ({channels}x{ih}x{iw}). Calculated output size: \
             ({channels}x{oh}x{ow}). Output size is too small"
        )));
    }

    let mut out_dims = dims[..split].to_vec();
    out_dims.push(oh as usize);
    out_dims.push(ow as usize);
    let device = input.tensor()?.device().clone();
    if input.tensor()?.elem_count() == 0 {
        let out = Tensor::zeros(out_dims, PyDtype::new(tag).storage(OP)?, &device)
            .map_err(|e| candle_err(OP, e))?;
        return finish(py, out, tag);
    }

    // `opmath_t`: `f32` for the three narrow floats -- and that is a
    // *narrowing* for `float32`, measured, not only a widening for the
    // reduced two. See the doc comment.
    let acc32 = tag == TorchDType::Float32
        || tag == TorchDType::Float16
        || tag == TorchDType::BFloat16;

    let source = read_flat(OP, input.tensor()?, tag)?;
    let plane = (ih * iw) as usize;
    let total = planes * (oh * ow) as usize;
    let mut out_f = vec![0.0f64; if integral { 0 } else { total }];
    let mut out_i = vec![0i64; if integral { total } else { 0 }];

    for p in 0..planes {
        let base = p * plane;
        for y in 0..oh {
            let h_start_raw = y * sh - ph;
            let h_end_raw = (h_start_raw + kh).min(ih + ph);
            let h_start = h_start_raw.max(0);
            let h_end = h_end_raw.min(ih);
            for x in 0..ow {
                let w_start_raw = x * sw - pw;
                let w_end_raw = (w_start_raw + kw).min(iw + pw);
                // The count is taken from the *unclipped* window: that is
                // what makes `count_include_pad=True` divide by the padded
                // area rather than by the elements it actually summed.
                let padded_count = (h_end_raw - h_start_raw) * (w_end_raw - w_start_raw);
                let w_start = w_start_raw.max(0);
                let w_end = w_end_raw.min(iw);
                let real_count = (h_end - h_start) * (w_end - w_start);
                let divisor = match divisor_override {
                    Some(value) => value,
                    None if count_include_pad => padded_count,
                    None => real_count,
                };
                let at = p * (oh * ow) as usize + (y * ow + x) as usize;
                match &source {
                    Flat::Float(values) => {
                        if acc32 {
                            let mut sum = 0.0f32;
                            for row in h_start..h_end {
                                let offset = base + (row * iw) as usize;
                                for col in w_start..w_end {
                                    sum += values[offset + col as usize] as f32;
                                }
                            }
                            out_f[at] = (sum / divisor as f32) as f64;
                        } else {
                            let mut sum = 0.0f64;
                            for row in h_start..h_end {
                                let offset = base + (row * iw) as usize;
                                for col in w_start..w_end {
                                    sum += values[offset + col as usize];
                                }
                            }
                            out_f[at] = sum / divisor as f64;
                        }
                    }
                    Flat::Int(values) => {
                        let mut sum = 0i64;
                        for row in h_start..h_end {
                            let offset = base + (row * iw) as usize;
                            for col in w_start..w_end {
                                sum += values[offset + col as usize];
                            }
                        }
                        // Truncating division, measured: -11 over 4 is -2.
                        out_i[at] = sum / divisor;
                    }
                }
            }
        }
    }

    let values = if integral {
        Flat::Int(out_i)
    } else {
        Flat::Float(out_f)
    };
    let out = write_flat(OP, values, out_dims, &device, tag)?;
    finish(py, out, tag)
}

/// `aten::upsample_bilinear2d(Tensor self, SymInt[2] output_size,
///     bool align_corners, float? scales_h=None, float? scales_w=None)
///     -> Tensor`
///
/// `zoedepth`'s wall. `F.interpolate(x, scale_factor=2, mode="bilinear",
/// align_corners=...)` -> `torch._C._nn.upsample_bilinear2d`, whose `.vec`
/// signature is `CompositeImplicitAutograd`: measured with a
/// `TorchDispatchMode` logger on 2.13.0, every spelling emits
/// `aten.upsample_bilinear2d.default` with a *concrete* output size and the
/// scale factors passed through, and `.vec` never fires. So `.vec` lives in
/// `bootstrap.py`'s `_install_nn` and this is the leaf.
///
/// # The grid, which is the whole op
///
/// Two conventions, both used in the wild, and they are **different
/// functions** rather than a tolerance apart. Per axis, with `d` the output
/// index:
///
/// ```text
/// align_corners=true    scale = (in-1)/(out-1)   [0 if out == 1]
///                       src   = scale * d
/// align_corners=false   scale = 1/scale_arg  if given and > 0, else in/out
///                       src   = max(scale * (d + 0.5) - 0.5, 0)
/// ```
///
/// The `+0.5 ... -0.5` is the **half-pixel** convention, and dropping it is
/// the classic error this op invites: `scale * d` under `align_corners=false`
/// produces a perfectly plausible, slightly-shifted image rather than an
/// error. Measured on `arange(6).reshape(1,1,2,3)` upsampled to `(4,6)`, the
/// two conventions disagree on 20 of 24 elements:
///
/// ```text
/// align_corners=false   0.00 0.25 0.75 1.25 1.75 2.00 | 0.75 1.00 ...
/// align_corners=true    0.00 0.40 0.80 1.20 1.60 2.00 | 1.00 1.40 ...
/// ```
///
/// They agree at the four corners, which is exactly what `align_corners`
/// means -- so a case set built only from corners cannot separate them, and
/// neither can one built from a symmetric input.
///
/// Three details inside that, each measured on its own:
///
///   * **`scales_h`/`scales_w` are honoured, and they are not `in/out`.**
///     `1/scale` and `in/out` coincide whenever `out == in * scale` exactly,
///     which is every case a `scale_factor=2` test produces. They differ as
///     soon as the product is not integral: with `in=3, out=4,
///     scales_w=1.5`, `1/1.5 = 0.667` against `3/4 = 0.75`, and upstream
///     answers `[0, 0.5, 1.1667, 1.8333]` rather than `[0, 0.625, 1.375, 2]`.
///   * **a non-positive scale is ignored**, falling back to `in/out` --
///     measured with `0.0` and `-1.0`, both of which give the no-scale answer.
///   * **`align_corners=true` ignores the scales entirely** -- measured with
///     `scales_w=9.0`, which changes nothing.
///
/// And the short circuit: **when `out == in` on an axis, the axis is copied**,
/// with no grid at all. That is not the same as "the grid happens to be the
/// identity" -- measured, `out == in` with `scales_w=0.5` still copies, where
/// the grid would have resampled.
///
/// # Precision
///
/// `opmath_t`: `f32` for `float16`/`bfloat16`/`float32`, `f64` for `float64`.
/// Both halves of that are measured and **both directions matter**:
///
/// ```text
/// float16 computed in f32 and narrowed once   0 of 143 differ from upstream
/// float16 computed in f64 and narrowed once   2 of 143 differ
/// float32 computed in f64 and narrowed once   241 of 286 differ
/// ```
///
/// So this is not "compute as wide as possible". `float32` has to be computed
/// in `float32`, which is why the arithmetic below casts through `f32`
/// explicitly rather than staying in the `f64` that `read_flat` hands over.
///
/// # Refusals, in upstream's own order
///
/// ```text
/// output_size.len() != 2   It is expected output_size equals to 2, but got size N
/// input rank != 4          It is expected input_size equals to 4, but got size N
/// any extent <= 0          Input and output sizes should be greater than 0, but got ...
/// a zero non-batch dim     Non-empty 4D data tensor expected but got a tensor with sizes [...]
/// int64 / bool             "upsample_bilinear2d_channels_last" not implemented for 'Long'
/// ```
///
/// The order is measured, not chosen: a rank-3 input with a length-1
/// `output_size` reports the *output_size* message, and an `int64` input with
/// a zero output extent reports the *size* message. `N == 0` is accepted (the
/// non-empty check looks at the product of the dims *after* the batch), and
/// `C == 0` is not.
///
/// **Not implemented: `uint8`.** Upstream computes it -- and not by rounding a
/// bilinear result. Over 60 random shapes (5584 elements), `round-half-away-
/// from-zero` applied to the `float32` answer disagrees with upstream's
/// `uint8` answer on **355** of them, so upstream is running a different
/// (fixed-point) kernel there and reproducing it is its own measurement round.
/// Refused by name, with a `c_error` case watching it, rather than shipping
/// the 94%-correct rule.
fn upsample_bilinear2d_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.upsample_bilinear2d.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let output_size = shape_arg(OP, args, kwargs, 1, "output_size")?;
    let align_corners =
        bool_arg(args, kwargs, 2, "align_corners")?.ok_or_else(|| missing(OP, "align_corners"))?;
    let scales_h = scalar_arg(OP, args, kwargs, 3, "scales_h")?.map(|s| s.as_f64());
    let scales_w = scalar_arg(OP, args, kwargs, 4, "scales_w")?.map(|s| s.as_f64());

    if output_size.len() != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "It is expected output_size equals to 2, but got size {}",
            output_size.len()
        )));
    }
    let dims = input.tensor()?.dims().to_vec();
    if dims.len() != 4 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "It is expected input_size equals to 4, but got size {}",
            dims.len()
        )));
    }
    let (in_h, in_w) = (dims[2] as i64, dims[3] as i64);
    let (out_h, out_w) = (output_size[0] as i64, output_size[1] as i64);
    if in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Input and output sizes should be greater than 0, but got input (H: {in_h}, \
             W: {in_w}) output (H: {out_h}, W: {out_w})"
        )));
    }
    // Upstream's own guard: a zero *batch* is fine, a zero anywhere else is
    // not. `N == 0` gives an empty answer; `C == 0` raises. Measured both ways.
    if dims[1..].iter().product::<usize>() == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Non-empty 4D data tensor expected but got a tensor with sizes {dims:?}"
        )));
    }

    let tag = input.tag();
    if tag == TorchDType::UInt8 {
        return Err(not_implemented(format!(
            "{OP}: a uint8 input is not implemented in torch._C shim -- upstream computes \
             it with a separate fixed-point kernel, not by rounding the float answer \
             (measured: round-half-away-from-zero on the float32 result disagrees with \
             upstream on 355 of 5584 elements over 60 random shapes), and reproducing that \
             kernel is its own measurement round"
        )));
    }
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"upsample_bilinear2d_channels_last\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }

    let out_dims = vec![dims[0], dims[1], out_h as usize, out_w as usize];
    let device = input.tensor()?.device().clone();
    if dims[0] == 0 {
        let out = Tensor::zeros(out_dims, PyDtype::new(tag).storage(OP)?, &device)
            .map_err(|e| candle_err(OP, e))?;
        return finish(py, out, tag);
    }

    // `opmath_t`, and it is a *narrowing* for `float32` as much as a widening
    // for `float16` -- see the measurement in the doc comment.
    let acc32 = tag != TorchDType::Float64;

    // `area_pixel_compute_scale`, per axis.
    let scale_of = |in_size: i64, out_size: i64, given: Option<f64>| -> f64 {
        if align_corners {
            if out_size > 1 {
                if acc32 {
                    ((in_size - 1) as f32 / (out_size - 1) as f32) as f64
                } else {
                    (in_size - 1) as f64 / (out_size - 1) as f64
                }
            } else {
                0.0
            }
        } else {
            match given {
                Some(scale) if scale > 0.0 => {
                    if acc32 {
                        (1.0 / scale) as f32 as f64
                    } else {
                        1.0 / scale
                    }
                }
                _ => {
                    if acc32 {
                        (in_size as f32 / out_size as f32) as f64
                    } else {
                        in_size as f64 / out_size as f64
                    }
                }
            }
        }
    };
    // `compute_source_index_and_lambda`, per axis, precomputed once per output
    // row/column rather than per element.
    let grid = |in_size: i64, out_size: i64, scale: f64| -> Vec<(usize, usize, f64, f64)> {
        (0..out_size)
            .map(|index| {
                if out_size == in_size {
                    return (index as usize, index as usize, 1.0, 0.0);
                }
                let real = if align_corners {
                    if acc32 {
                        (scale as f32 * index as f32) as f64
                    } else {
                        scale * index as f64
                    }
                } else if acc32 {
                    let value = (scale as f32 * (index as f32 + 0.5) - 0.5) as f64;
                    if value < 0.0 {
                        0.0
                    } else {
                        value
                    }
                } else {
                    let value = scale * (index as f64 + 0.5) - 0.5;
                    if value < 0.0 {
                        0.0
                    } else {
                        value
                    }
                };
                let i0 = real as i64;
                let i1 = i0 + if i0 < in_size - 1 { 1 } else { 0 };
                let l1 = if acc32 {
                    (real as f32 - i0 as f32) as f64
                } else {
                    real - i0 as f64
                };
                let l0 = if acc32 { (1.0f32 - l1 as f32) as f64 } else { 1.0 - l1 };
                (i0 as usize, i1 as usize, l0, l1)
            })
            .collect()
    };

    let h_grid = grid(in_h, out_h, scale_of(in_h, out_h, scales_h));
    let w_grid = grid(in_w, out_w, scale_of(in_w, out_w, scales_w));

    let source = match read_flat(OP, input.tensor()?, tag)? {
        Flat::Float(values) => values,
        // Unreachable: every non-floating tag is refused above.
        Flat::Int(values) => values.into_iter().map(|v| v as f64).collect(),
    };
    let plane = (in_h * in_w) as usize;
    let planes = dims[0] * dims[1];
    let mut out = vec![0.0f64; planes * (out_h * out_w) as usize];
    let mut at = 0usize;
    for p in 0..planes {
        let base = p * plane;
        for &(h0, h1, hl0, hl1) in &h_grid {
            let row0 = base + h0 * in_w as usize;
            let row1 = base + h1 * in_w as usize;
            for &(w0, w1, wl0, wl1) in &w_grid {
                let v00 = source[row0 + w0];
                let v01 = source[row0 + w1];
                let v10 = source[row1 + w0];
                let v11 = source[row1 + w1];
                out[at] = if acc32 {
                    let top = wl0 as f32 * v00 as f32 + wl1 as f32 * v01 as f32;
                    let bottom = wl0 as f32 * v10 as f32 + wl1 as f32 * v11 as f32;
                    (hl0 as f32 * top + hl1 as f32 * bottom) as f64
                } else {
                    hl0 * (wl0 * v00 + wl1 * v01) + hl1 * (wl0 * v10 + wl1 * v11)
                };
                at += 1;
            }
        }
    }

    let out = write_flat(OP, Flat::Float(out), out_dims, &device, tag)?;
    finish(py, out, tag)
}

/// `aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor`
///
/// Two refusals are reproduced rather than papered over, both measured:
///
///   * `half_to_float=True` is **not supported on CPU at all**, for any dtype
///     -- `float16`, `bfloat16` and `float32` inputs all raise. It is a CUDA-only
///     fusion. A shim that quietly honoured it would return `float32` where
///     upstream raises.
///   * an integral input raises `NotImplementedError`, naming the kernel, not
///     a `RuntimeError`.
///
/// The accumulate type follows `opmath_type<scalar_t>` -- `float` for
/// `float16`/`bfloat16`/`float32`, `double` for `float64` -- so the reduced
/// dtypes compute in float and narrow once at the end, which is why they agree
/// with upstream to better than their own epsilon.
///
/// The max is subtracted before the exponential for the reason docs/OPS8.md §3
/// gives for the attention kernel: without it a masked `-inf` and a large logit
/// both come out NaN. With it, `exp(-inf - max)` is a clean zero. A row that is
/// *entirely* `-inf` still gives NaN on both sides, and a case pins that.
fn softmax_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._softmax.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let dim_raw = dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?;
    let half_to_float = bool_arg(args, kwargs, 2, "half_to_float")?
        .ok_or_else(|| missing(OP, "half_to_float"))?;
    if half_to_float {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "softmax with half to float conversion is not supported on CPU",
        ));
    }
    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(not_implemented(format!(
            "\"softmax_lastdim_kernel_impl\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    let rank = input.tensor()?.rank();
    let dim = normalise_dim(OP, dim_raw, rank)?;

    let dims = input.tensor()?.dims().to_vec();
    let (outer, n, inner) = if dims.is_empty() {
        (1usize, 1usize, 1usize)
    } else {
        (
            dims[..dim].iter().product::<usize>(),
            dims[dim],
            dims[dim + 1..].iter().product::<usize>(),
        )
    };

    let source = match read_flat(OP, input.tensor()?, tag)? {
        Flat::Float(v) => v,
        Flat::Int(_) => unreachable!("the integral dtypes were refused above"),
    };
    let storage = PyDtype::new(tag).storage(OP)?;
    let double_acc = storage == candle_core::DType::F64;
    let out = softmax_body(&source, outer, n, inner, double_acc, false);

    let device = input.tensor()?.device().clone();
    let tensor = write_flat(OP, Flat::Float(out), dims, &device, tag)?;
    finish(py, tensor, tag)
}

/// The reduction shared by `_softmax.default` and `_safe_softmax.default`:
/// max-subtract, exponentiate, normalise, over the `(outer, n, inner)` view of
/// a dim-`dim` softmax (`outer`/`inner` are the product of the extents on
/// either side of `dim`; `n` is `dim`'s own extent).
///
/// `safe` is the only difference between the two ops, and it is applied where
/// the divergence is measured to live: a row whose max is `-inf` (every
/// element `-inf`, since `-inf` is softmax's own floor) computes `0` for
/// every element instead of running the usual exponential, which on that row
/// is a `NaN` produced by `-inf - (-inf)`. `safe=false` skips the check
/// entirely, so `_softmax.default`'s behaviour (a `NaN` row, matching
/// upstream, per this file's docs above) is untouched by this refactor.
fn softmax_body(source: &[f64], outer: usize, n: usize, inner: usize, double_acc: bool, safe: bool) -> Vec<f64> {
    let mut out = vec![0.0f64; source.len()];

    for o in 0..outer {
        for i in 0..inner {
            let at = |j: usize| o * n * inner + j * inner + i;
            if double_acc {
                let mut max = f64::NEG_INFINITY;
                for j in 0..n {
                    let v = source[at(j)];
                    if !(v <= max) {
                        max = v;
                    }
                }
                if safe && max == f64::NEG_INFINITY {
                    for j in 0..n {
                        out[at(j)] = 0.0;
                    }
                    continue;
                }
                let mut sum = 0.0f64;
                for j in 0..n {
                    let e = (source[at(j)] - max).exp();
                    out[at(j)] = e;
                    sum += e;
                }
                for j in 0..n {
                    out[at(j)] /= sum;
                }
            } else {
                let mut max = f32::NEG_INFINITY;
                for j in 0..n {
                    let v = source[at(j)] as f32;
                    if !(v <= max) {
                        max = v;
                    }
                }
                if safe && max == f32::NEG_INFINITY {
                    for j in 0..n {
                        out[at(j)] = 0.0;
                    }
                    continue;
                }
                let mut sum = 0.0f32;
                for j in 0..n {
                    let e = ((source[at(j)] as f32) - max).exp();
                    out[at(j)] = e as f64;
                    sum += e;
                }
                for j in 0..n {
                    out[at(j)] = ((out[at(j)] as f32) / sum) as f64;
                }
            }
        }
    }
    out
}

/// `aten::_safe_softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor`
///
/// torch's own decomposition is the spec this reproduces
/// (`torch/_decomp/decompositions.py::safe_softmax`, `register_decomposition(aten._safe_softmax)`):
///
/// ```text
/// out = torch.softmax(self, dim=dim, dtype=dtype)
/// masked_rows = torch.all(self.eq(-inf), dim=dim, keepdim=True)
/// return torch.where(masked_rows, zeros, out)
/// ```
///
/// So the one place this disagrees with `_softmax.default` is a row that is
/// *entirely* `-inf`: measured on torch 2.13.0, plain `_softmax` answers `nan`
/// there (see that op's docs above) and `_safe_softmax` answers `0` for every
/// element of the row instead. That is exactly the shape of a fully-masked
/// attention row -- every key excluded by the causal mask plus padding -- and
/// `nan` there would poison every downstream matmul rather than staying
/// contained the way a `0` attention weight does.
///
/// `dtype`, when given, casts `self` **before** the integral/boolean refusal
/// below runs, not after -- measured: `_safe_softmax(int64_tensor, 0,
/// torch.float32)` succeeds. Composite `torch.softmax(self, dim, dtype)`
/// upcasts first and then calls the kernel in that dtype, and the refusal is
/// the kernel's, so a shim that checked the *original* dtype would reject a
/// call upstream accepts.
fn safe_softmax_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._safe_softmax.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let dim_raw = dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?;
    let dtype = dtype_arg(args, kwargs, 2, "dtype")?;

    let (tensor, tag) = match dtype {
        Some(want) => {
            let storage = PyDtype::new(want).storage(OP)?;
            let cast = input
                .tensor()?
                .fast_to(storage)
                .map_err(|e| candle_err(OP, e))?;
            (cast, want)
        }
        None => (input.tensor()?.clone(), input.tag()),
    };
    if !tag.is_floating_point() {
        return Err(not_implemented(format!(
            "\"softmax_lastdim_kernel_impl\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    let rank = tensor.rank();
    let dim = normalise_dim(OP, dim_raw, rank)?;

    let dims = tensor.dims().to_vec();
    let (outer, n, inner) = if dims.is_empty() {
        (1usize, 1usize, 1usize)
    } else {
        (
            dims[..dim].iter().product::<usize>(),
            dims[dim],
            dims[dim + 1..].iter().product::<usize>(),
        )
    };

    let source = match read_flat(OP, &tensor, tag)? {
        Flat::Float(v) => v,
        Flat::Int(_) => unreachable!("the integral dtypes were refused above"),
    };
    let storage = PyDtype::new(tag).storage(OP)?;
    let double_acc = storage == candle_core::DType::F64;
    let out = softmax_body(&source, outer, n, inner, double_acc, true);

    let device = tensor.device().clone();
    let result = write_flat(OP, Flat::Float(out), dims, &device, tag)?;
    finish(py, result, tag)
}

/// `aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor`
///
/// The first half of a cross-entropy forward, and the reason `docs/TRAIN.md`'s
/// 26 of 26 are *lossless* forwards: without it there is no scalar to call
/// `.backward()` on. `docs/LOSS.md` is the round that landed it.
///
/// It shares `_softmax`'s two refusals verbatim -- `half_to_float=True` is a
/// CUDA-only fusion and raises on CPU for every dtype, and an integral input
/// raises `NotImplementedError` naming the kernel -- but **it does not share
/// the kernel name in that message**, and the difference is measurable:
///
/// ```text
/// _log_softmax(int64 (4,),   dim 0)  "log_softmax_lastdim_kernel_impl" not implemented for 'Long'
/// _log_softmax(int64 (2,3),  dim 1)  "log_softmax_lastdim_kernel_impl" ...
/// _log_softmax(int64 (2,3),  dim 0)  "log_softmax_kernel_impl"         ...
/// ```
///
/// upstream picks between two kernels on whether `dim` is the trailing axis,
/// and each names itself. (`_softmax` above answers `softmax_lastdim_kernel_impl`
/// for both, which is a pre-existing near-miss in *that* op, not this one.)
///
/// ## The split that is not guessable
///
/// That same fork decides the **arithmetic**, and the two halves do not agree
/// on where the sum of exponentials is rounded. Measured against upstream on
/// `bfloat16` and `float16`, over seven shape/dim combinations each -- the
/// column is the number of elements that differ from upstream:
///
/// ```text
///                                    sum kept in f32   sum narrowed to dtype
///   bfloat16  (3,5,9) dim -1              26                    0
///   bfloat16  (3,5,9) dim  1               0                   27
///   float16   (2,3,4,5) dim -1            19                    0
///   float16   (4,7) dim 0                  0                    8
/// ```
///
/// It is exactly upstream's own source. `serial_vec_log_softmax_lastdim_range`
/// (`ATen/native/cpu/LogSoftmaxKernelImpl.h`) accumulates the sum in `float`
/// but **stores it into a `scalar_t[]` buffer**, takes the log of that narrowed
/// value, and stores the log back into the same `scalar_t` buffer -- so on the
/// last-dim path a `bfloat16` row loses the sum to 8 significand bits *twice*
/// before it is subtracted. `serial_vec_logsoftmax_range`, the strided path,
/// holds both in `float[]` and never narrows. For `float32` and `float64` the
/// narrowing is the identity and the two paths coincide, which is why a
/// float-only test cannot see any of this.
///
/// The separating case is small and is in `tools/golden/cases.py`:
/// `bfloat16 [0, ln(0.002)]` sums to `1.00203`, which rounds to exactly `1.0`
/// in `bfloat16`, so the last-dim path answers `log(1) = 0` and the strided
/// path answers `log(1.00203) = 0.00198`. The first output element is `0.0`
/// one way and `-0.00198` the other -- a **relative** difference of 1.0, which
/// is the only reason golden can see it at all: one `bfloat16` ULP is 0.4%
/// relative and this file's tolerance for that dtype is 6%.
///
/// ## Order of operations
///
/// `x - max - log(sum)`, left to right, never `x - (max + log(sum))`.
/// Upstream's comment says why, citing pytorch#11752: with large logits and a
/// small spread, forming `max + tmp_sum` first loses the difference the whole
/// computation is about.
fn log_softmax_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._log_softmax.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let dim_raw = dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?;
    let half_to_float = bool_arg(args, kwargs, 2, "half_to_float")?
        .ok_or_else(|| missing(OP, "half_to_float"))?;
    if half_to_float {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "softmax with half to float conversion is not supported on CPU",
        ));
    }
    let tag = input.tag();
    let rank = input.tensor()?.rank();
    let dim = normalise_dim(OP, dim_raw, rank)?;
    let dims = input.tensor()?.dims().to_vec();
    // Upstream views a 0-dim input as `(1,)` before choosing, so its only axis
    // is the trailing one.
    let lastdim = dims.is_empty() || dim + 1 == rank;
    if !tag.is_floating_point() {
        return Err(not_implemented(format!(
            "\"{}\" not implemented for '{}'",
            if lastdim {
                "log_softmax_lastdim_kernel_impl"
            } else {
                "log_softmax_kernel_impl"
            },
            scalar_type_name(tag)
        )));
    }

    let (outer, n, inner) = if dims.is_empty() {
        (1usize, 1usize, 1usize)
    } else {
        (
            dims[..dim].iter().product::<usize>(),
            dims[dim],
            dims[dim + 1..].iter().product::<usize>(),
        )
    };

    let source = match read_flat(OP, input.tensor()?, tag)? {
        Flat::Float(v) => v,
        Flat::Int(_) => unreachable!("the integral dtypes were refused above"),
    };
    let storage = PyDtype::new(tag).storage(OP)?;
    let double_acc = storage == candle_core::DType::F64;
    let narrow = if lastdim { Some(float_narrower(tag)) } else { None };
    let out = log_softmax_body(&source, outer, n, inner, double_acc, narrow);

    let device = input.tensor()?.device().clone();
    let tensor = write_flat(OP, Flat::Float(out), dims, &device, tag)?;
    finish(py, tensor, tag)
}

/// The reduction behind `_log_softmax.default`, over the same `(outer, n,
/// inner)` view `softmax_body` uses.
///
/// `narrow`, when present, is the tensor dtype's rounding, applied to the sum
/// and again to its logarithm -- upstream's last-dim path, which round-trips
/// both through a `scalar_t` buffer. `None` is the strided path, which holds
/// them in `float`. See `log_softmax_default`'s docs for the measurement that
/// decided this; for `float32`/`float64` the two are the same function.
fn log_softmax_body(
    source: &[f64],
    outer: usize,
    n: usize,
    inner: usize,
    double_acc: bool,
    narrow: Option<fn(f64) -> f64>,
) -> Vec<f64> {
    let mut out = vec![0.0f64; source.len()];

    for o in 0..outer {
        for i in 0..inner {
            let at = |j: usize| o * n * inner + j * inner + i;
            if double_acc {
                let mut max = f64::NEG_INFINITY;
                for j in 0..n {
                    let v = source[at(j)];
                    if !(v <= max) {
                        max = v;
                    }
                }
                let mut sum = 0.0f64;
                for j in 0..n {
                    sum += (source[at(j)] - max).exp();
                }
                // `float64`'s narrower is the identity, so this branch is the
                // same either way; it is written out so the two paths read
                // alike.
                let logsum = match narrow {
                    Some(f) => f(f(sum).ln()),
                    None => sum.ln(),
                };
                for j in 0..n {
                    out[at(j)] = source[at(j)] - max - logsum;
                }
            } else {
                let mut max = f32::NEG_INFINITY;
                for j in 0..n {
                    let v = source[at(j)] as f32;
                    if !(v <= max) {
                        max = v;
                    }
                }
                let mut sum = 0.0f32;
                for j in 0..n {
                    sum += ((source[at(j)] as f32) - max).exp();
                }
                let logsum = match narrow {
                    Some(f) => {
                        let narrowed = f(sum as f64) as f32;
                        f(narrowed.ln() as f64) as f32
                    }
                    None => sum.ln(),
                };
                for j in 0..n {
                    out[at(j)] = (((source[at(j)] as f32) - max) - logsum) as f64;
                }
            }
        }
    }
    out
}

/// `aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight,
///     int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)`
///
/// The second half of a cross-entropy forward (docs/LOSS.md), and the op whose
/// **second return value** is the reason a forward-only test is not enough:
/// `total_weight` is what `nll_loss_backward` divides by, and every caller in
/// `transformers` throws it away. Its rules are not derivable from the loss:
///
/// ```text
/// reduction=none, 2-D input   total_weight = 0      <- always, even weighted
/// reduction=none, 1-D input   total_weight = 1      (1-D takes the reduce path)
/// reduction=mean/sum, no w    total_weight = batch_size - num_ignored
/// reduction=mean/sum, w       total_weight = sum of the weights not ignored
/// empty target                total_weight = 0
/// ```
///
/// The first line is upstream writing `*total_weight_data = 0` at the top of
/// `nll_loss_out_frame` and then returning before it is ever updated. A shim
/// that computed the "obvious" total weight there would be wrong in a way no
/// loss value can show.
///
/// ## The summation is a cascade, and transcribing it as a loop is wrong
///
/// Upstream accumulates into **eight partial sums** with a carry at every
/// `2^4` elements, all in `scalar_t`. Measured against a plain left-to-right
/// sum, upstream and naive disagree from `n=8` in `bfloat16` and from `n=300`
/// in `float32`:
///
/// ```text
///   n=300  bfloat16   upstream -225        naive -226        f64 -226.61255
///   n=300  float32    upstream 373.92365   naive 373.92377   f64 373.92358
///   n=4096 bfloat16   upstream -1528       naive -1384       f64 -1545.9946
/// ```
///
/// Three details in it are load-bearing and each was found by a mismatch:
///
/// * **An ignored target `continue`s past the carry loop.** It does not just
///   skip the addition -- the whole cascade advance is skipped for that `b`,
///   so `ignore_index` changes *where* the carries land, not only what is
///   summed. Running the carry anyway matched upstream on unweighted runs and
///   drifted on `ignore_index=3` ones.
/// * **`float32` and `float64` contract `sum -= data * weight` into an FMA;
///   the reduced dtypes cannot.** `c10::BFloat16 operator*` returns a
///   `BFloat16`, so the product is rounded before the subtraction and no
///   contraction is possible; native `float`/`double` are contracted by the
///   compiler. Using an FMA everywhere, or nowhere, both mismatch -- and in
///   opposite dtypes, which is how the split was found.
/// * **`total_weight` for the unweighted case is a *count*, not a sum**:
///   `static_cast<scalar_t>(batch_size - num_ignored)`, so it never
///   accumulates rounding and never goes through the cascade at all.
///
/// With all three, a transcription of `nll_loss_out_frame` matched upstream
/// **1200 of 1200** combinations bit for bit: 25 batch sizes from 1 to 5000 x
/// 4 dtypes x {sum, mean} x 3 `ignore_index` values x {weighted, unweighted}.
///
/// ## The checks, in upstream's order
///
/// `ignore_index` is tested **before** the bounds check, so a target equal to
/// an out-of-range `ignore_index` is accepted -- measured:
/// `nll_loss_forward(x, [0, 77], None, mean, 77)` succeeds where the same call
/// with `ignore_index=-100` raises `IndexError: Target 77 is out of bounds.`
///
/// `reduction` is **not validated**: upstream accepts `3` and treats it as
/// anything-but-Mean, i.e. a sum (measured). This reproduces that rather than
/// adding a refusal upstream does not have.
fn nll_loss_forward_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.nll_loss_forward.default";
    const MEAN: i64 = 1;
    const NONE: i64 = 0;

    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let target = tensor_arg(OP, args, kwargs, 1, "target")?;
    let weight = optional_tensor_arg(OP, args, kwargs, 2, "weight")?;
    let reduction = int_arg(args, kwargs, 3, "reduction")?.ok_or_else(|| missing(OP, "reduction"))?;
    let ignore_index =
        int_arg(args, kwargs, 4, "ignore_index")?.ok_or_else(|| missing(OP, "ignore_index"))?;

    let in_dims = input.tensor()?.dims().to_vec();
    let tgt_dims = target.tensor()?.dims().to_vec();

    // -- the meta function's checks, in its order ------------------------
    if in_dims.is_empty() || in_dims.len() > 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "input tensor should be 1D or 2D",
        ));
    }
    if tgt_dims.len() > 1 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "0D or 1D target tensor expected, multi-target not supported",
        ));
    }
    if !matches!(target.tag(), TorchDType::Int64 | TorchDType::UInt8) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "expected target dtype to be Long or Byte, but got {}",
            scalar_type_name(target.tag())
        )));
    }
    if in_dims.len() == 1 && tgt_dims.len() == 1 && tgt_dims[0] != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "For 1D input, 1D target must have size 1, but got target size: {}",
            tgt_dims[0]
        )));
    }
    if in_dims.len() != 1 && (tgt_dims.is_empty() || in_dims[0] != tgt_dims[0]) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "size mismatch (got input: {:?}, target: {:?})",
            in_dims, tgt_dims
        )));
    }
    let n_classes = in_dims[in_dims.len() - 1];
    if let Some(w) = &weight {
        let wd = w.tensor()?.dims().to_vec();
        if wd.len() != 1 || wd[0] != n_classes {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "weight tensor should be defined either for all {n_classes} classes or no \
                 classes but got weight tensor of shape: {wd:?}"
            )));
        }
    }

    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(not_implemented(format!(
            "\"nll_loss_out_frame\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    // `data_ptr<scalar_t>()` on the weight is what raises this upstream, so it
    // is an exact dtype match rather than a promotion.
    if let Some(w) = &weight {
        if w.tag() != tag {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "expected scalar type {} but found {}",
                scalar_type_name(tag),
                scalar_type_name(w.tag())
            )));
        }
    }

    let device = input.tensor()?.device().clone();
    let source = match read_flat(OP, input.tensor()?, tag)? {
        Flat::Float(v) => v,
        Flat::Int(_) => unreachable!("the integral dtypes were refused above"),
    };
    let targets = match read_flat(OP, target.tensor()?, target.tag())? {
        Flat::Int(v) => v,
        Flat::Float(_) => unreachable!("only Long and Byte reach here"),
    };
    let weights = match &weight {
        Some(w) => match read_flat(OP, w.tensor()?, tag)? {
            Flat::Float(v) => Some(v),
            Flat::Int(_) => unreachable!("the weight shares the input's floating dtype"),
        },
        None => None,
    };
    let narrow = float_narrower(tag);
    let bounds = |t: i64| -> PyResult<usize> {
        if t < 0 || t as usize >= n_classes {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Target {t} is out of bounds."
            )));
        }
        Ok(t as usize)
    };

    // -- reduction=None over a 2-D input: elementwise, and total_weight
    //    stays at the zero it was initialised to.
    if reduction == NONE && in_dims.len() == 2 {
        let batch = in_dims[0];
        let mut out = vec![0.0f64; batch];
        for i in 0..batch {
            let t = targets[i];
            if t == ignore_index {
                continue; // upstream writes an explicit 0 here
            }
            let c = bounds(t)?;
            let w = weights.as_ref().map(|w| w[c]).unwrap_or(1.0);
            out[i] = narrow(narrow(-source[i * n_classes + c]) * w);
        }
        let output = write_flat(OP, Flat::Float(out), vec![batch], &device, tag)?;
        let total = write_flat(OP, Flat::Float(vec![0.0]), vec![], &device, tag)?;
        let pair = [
            crate::tensor::promote(py, finish(py, output, tag)?)?,
            crate::tensor::promote(py, finish(py, total, tag)?)?,
        ];
        return Ok(PyTuple::new(py, pair)?.into_any().unbind());
    }

    // -- the reduce path, which a 1-D input always takes ------------------
    let scalar = |v: f64| -> PyResult<Tensor> {
        write_flat(OP, Flat::Float(vec![v]), vec![], &device, tag)
    };
    if targets.is_empty() {
        // Mean over nothing is NaN, by upstream's own choice (pytorch#64572).
        let out = if reduction == MEAN { f64::NAN } else { 0.0 };
        let pair = [
            crate::tensor::promote(py, finish(py, scalar(out)?, tag)?)?,
            crate::tensor::promote(py, finish(py, scalar(0.0)?, tag)?)?,
        ];
        return Ok(PyTuple::new(py, pair)?.into_any().unbind());
    }

    let batch = if in_dims.len() == 1 { 1 } else { in_dims[0] };
    let (loss, total_weight) = nll_cascade(
        &source, &targets, weights.as_deref(), n_classes, batch, ignore_index, tag, narrow,
        &bounds,
    )?;
    let out = if reduction == MEAN {
        narrow(loss / total_weight)
    } else {
        loss
    };
    let pair = [
        crate::tensor::promote(py, finish(py, scalar(out)?, tag)?)?,
        crate::tensor::promote(py, finish(py, scalar(total_weight)?, tag)?)?,
    ];
    Ok(PyTuple::new(py, pair)?.into_any().unbind())
}

/// `nll_loss_out_frame`'s eight-level cascade sum, and the `total_weight` that
/// comes out of the same loop. See `nll_loss_forward_default`'s docs for the
/// three details that are not guessable and for the 1200-case agreement.
#[allow(clippy::too_many_arguments)]
fn nll_cascade(
    source: &[f64],
    targets: &[i64],
    weights: Option<&[f64]>,
    n_classes: usize,
    batch: usize,
    ignore_index: i64,
    tag: TorchDType,
    narrow: fn(f64) -> f64,
    bounds: &dyn Fn(i64) -> PyResult<usize>,
) -> PyResult<(f64, f64)> {
    const LEVELS: usize = 8;
    // `std::max(4, CeilLog2(batch_size) / 8)`. In practice 4 for every batch
    // below 2^40; written out because it is upstream's and cost nothing.
    let ceil_log2 = |x: usize| -> u32 {
        if x <= 2 {
            1
        } else {
            (x - 1).ilog2() + 1
        }
    };
    let level_power = std::cmp::max(4u32, ceil_log2(batch) / LEVELS as u32);
    let level_mask: u64 = (1u64 << level_power) - 1;

    // The contraction split: native `float`/`double` fold `sum -= a * b` into
    // one FMA, `c10::BFloat16`/`c10::Half` cannot because their `operator*`
    // rounds. Measured -- using either everywhere mismatches, in opposite
    // dtypes.
    let fused = matches!(tag, TorchDType::Float32 | TorchDType::Float64);
    let double = tag == TorchDType::Float64;

    let mut wp = [0.0f64; LEVELS];
    let mut lp = [0.0f64; LEVELS];
    let mut num_ignored = 0usize;

    for b in 0..batch {
        let t = targets[b];
        if t == ignore_index {
            num_ignored += 1;
            continue; // and past the carry loop, which is the point
        }
        let c = bounds(t)?;
        let data = source[b * n_classes + c];
        match weights {
            Some(w) => {
                let wv = w[c];
                lp[0] = if fused {
                    if double {
                        (-data).mul_add(wv, lp[0])
                    } else {
                        ((-data as f32).mul_add(wv as f32, lp[0] as f32)) as f64
                    }
                } else {
                    narrow(lp[0] - narrow(data * wv))
                };
                wp[0] = narrow(wp[0] + wv);
            }
            None => lp[0] = narrow(lp[0] - data),
        }
        for j in 0..LEVELS - 1 {
            if (b as u64) & (level_mask << (j as u32 * level_power)) != 0 {
                break;
            }
            wp[j + 1] = narrow(wp[j + 1] + wp[j]);
            lp[j + 1] = narrow(lp[j + 1] + lp[j]);
            wp[j] = 0.0;
            lp[j] = 0.0;
        }
    }

    let total_weight = match weights {
        // A count, cast once -- not a sum, so it carries no rounding.
        None => narrow((batch - num_ignored) as f64),
        Some(_) => wp.iter().fold(0.0f64, |a, v| narrow(a + v)),
    };
    let loss = lp.iter().fold(0.0f64, |a, v| narrow(a + v));
    Ok((loss, total_weight))
}

/// `aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)`
///
/// The **out-of-place** dropout, and the reason it is here is not arithmetic:
/// `capture` refuses mutation so that a trace stays single-assignment
/// (docs/CAPTURE.md), and the eager composite `torch.dropout` decomposes onto
/// `bernoulli_`, which writes in place. So a `.train()` forward with real
/// dropout could not be captured at all — `gpt2`, `bert`, `opt` and
/// `gpt_bigcode`, `docs/TRAIN.md`'s own four. This op is upstream's own answer
/// to the same problem: it is the spelling functionalisation rewrites to, and
/// it returns the mask rather than hiding it inside an in-place fill.
///
/// **It is one kernel and not a decomposition, and that is the requirement.**
/// A `bootstrap.py` decomposition would emit its steps through the one door,
/// capture would record each of them, and `bernoulli_` would be among them —
/// which is the thing being fixed. One node in, one node out.
///
/// ## It is NOT the composite with the mutation removed
///
/// `at::native::native_dropout_cpu` and `_dropout_impl` differ in three
/// measurable ways, and two of them would be invisible in `float32`:
///
/// ```text
///                        _dropout_impl (torch.dropout)   native_dropout_cpu
///   the mask's dtype     the input's                     bool
///   where the scale goes on the MASK: noise.div_(1-p)    on the OUTPUT:
///                        then input * noise              input.mul(mask).mul_(scale)
///   p out of [0,1]       TORCH_CHECK naming p            no check of its own
/// ```
///
/// The second is the same distinction `docs/TRAIN.md` §5's S4 fault is about,
/// with the sides swapped — and it is a real difference in `bfloat16`/`float16`,
/// where `x * (1/(1-p))` and `x / (1-p)` disagree by an ULP on some survivors.
/// Following upstream means each spelling keeps its own answer.
///
/// The third is measured rather than read: `native_dropout(x, 1.5, True)`
/// raises **`bernoulli_ expects p to be in [0, 1], but got p=-0.5`** — the
/// message names `1 - p`, not `p`, because the only check on the road is
/// `bernoulli_`'s and it sees the survival probability. `torch.dropout(x, 1.5,
/// True)` raises `dropout probability has to be between 0 and 1, but got 1.5`.
/// A shared range check would answer the wrong one for whichever caller it was
/// not written for.
///
/// ## Two edges upstream chose and one of them looks like a bug
///
/// * **`numel == 0` returns the input itself and a mask of the INPUT's dtype**,
///   not `bool`: `return std::make_tuple(input, at::empty_like(input,
///   input.options()))`, taken before the branch that would have made it bool.
///   Measured — `native_dropout(torch.zeros(0,3), 0.5, True)[1].dtype` is
///   `torch.float32`. Reproduced rather than tidied; a caller that switches on
///   the mask's dtype sees what upstream shows it.
/// * **`train=False` copies.** `output = input.clone()`, so the result is not
///   the same object — unlike `_dropout_impl`, whose `p == 0 || !train` branch
///   returns `input` itself (docs/TRAIN.md §1 pins that identity). Two dropout
///   spellings, opposite answers to `out is x`.
///
/// `train=None` means `True`: upstream tests `!train.has_value() || *train`.
fn native_dropout_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.native_dropout.default";
    let input = tensor_arg(OP, args, kwargs, 0, "input")?;
    let p = float_arg(args, kwargs, 1, "p", 0.5)?;
    // `bool? train` -- absent and `None` both mean "not given", which upstream
    // treats as `true`.
    let train = bool_arg(args, kwargs, 2, "train")?.unwrap_or(true);

    let tag = input.tag();
    let dims = input.tensor()?.dims().to_vec();
    let device = input.tensor()?.device().clone();
    let numel: usize = dims.iter().product();

    // The zero-element early return, before anything decides the mask is bool.
    //
    // `return std::make_tuple(input, at::empty_like(input, input.options()))`
    // -- the input **itself**, so `out is input` is `True` upstream (measured),
    // and a mask carrying the *input's* dtype rather than `bool` because this
    // return is taken above the branch that would have made it bool. Both are
    // reproduced rather than tidied; a caller switching on the mask's dtype
    // should see what upstream shows it.
    if numel == 0 {
        let mask = write_flat(OP, Flat::Float(Vec::new()), dims.clone(), &device, tag)?;
        let same = required(OP, args, kwargs, 0, "input")?;
        let pair = [
            crate::tensor::promote(py, same.unbind())?,
            crate::tensor::promote(py, finish(py, mask, tag)?)?,
        ];
        return Ok(PyTuple::new(py, pair)?.into_any().unbind());
    }

    if !train {
        let mask = Tensor::from_vec(vec![1u8; numel], dims.clone(), &device)
            .map_err(|e| candle_err(OP, e))?;
        let out = input.tensor()?.copy().map_err(|e| candle_err(OP, e))?;
        let pair = [
            crate::tensor::promote(py, finish(py, out, tag)?)?,
            crate::tensor::promote(py, finish(py, mask, TorchDType::Bool)?)?,
        ];
        return Ok(PyTuple::new(py, pair)?.into_any().unbind());
    }

    let p1m = 1.0 - p;
    // Upstream's own comment: guard the reciprocal so `p == 1` gives 0 rather
    // than `inf`, which would turn every masked-out element into NaN.
    let scale = if p1m == 0.0 { 0.0 } else { 1.0 / p1m };

    // `mask.bernoulli_(p1m)`, and the range check that fires is `bernoulli_`'s,
    // reporting `p1m`. Same message text as `bernoulli_inplace_float`, which is
    // the function upstream actually reaches.
    if !(p1m >= 0.0 && p1m <= 1.0) {
        // `{}` on an `f64` NaN prints `NaN` in Rust and `nan` in C++'s
        // `operator<<`, and this message is compared as text.
        let shown = if p1m.is_nan() { "nan".to_string() } else { p1m.to_string() };
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "bernoulli_ expects p to be in [0, 1], but got p={shown}"
        )));
    }
    if !tag.is_floating_point() {
        // `output = input.mul(mask).mul_(scale)` is what raises: the scalar is
        // `double`, and an integral receiver cannot take it in place.
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "result type Float can't be cast to the desired output type {}",
            scalar_type_name(tag)
        )));
    }

    // One `random64()` per element, in `double`, exactly as `bernoulli_` does
    // for every dtype -- the asymmetry with `uniform_` that
    // `bernoulli_inplace_float`'s docs record. The mask is `bool`, but the
    // draws are not `bool`-shaped: they are the same stream a `float32` mask
    // would consume, which is why a seeded `native_dropout` and a seeded
    // `torch.dropout` produce the same mask (measured, all four dtypes).
    let mut gen = crate::rng::default_generator();
    let draws = crate::rng::uniform_fill_f64(&mut gen, numel, 0.0, 1.0);
    drop(gen);

    let source = match read_flat(OP, input.tensor()?, tag)? {
        Flat::Float(v) => v,
        Flat::Int(_) => unreachable!("the integral dtypes were refused above"),
    };
    let narrow = float_narrower(tag);
    // **The scale is narrowed to the input's dtype before it multiplies**, and
    // that is measured rather than read off `native_dropout_cpu`'s
    // `output.mul_(scale)`. A standalone `x.mul_(1/0.3)` on a `bfloat16` tensor
    // does NOT narrow -- `mul_kernel`'s reduced-float branch takes
    // `original_scalar_value<opmath_t>`, which is `float` (docs/TRAIN.md §5,
    // docs/SCALAR.md) -- and the two answers differ:
    //
    //     bfloat16, x = -9.875, p = 0.7
    //       x.mul(mask).mul_(scale)  step by step from Python  ->  -33.0
    //       native_dropout(x, 0.7, True) on the same survivor  ->  -32.75
    //       bfloat16(scale) = 3.328125, and -9.875 * 3.328125  ->  -32.75
    //
    // The narrowed route reproduces upstream on **1280 of 1280** combinations
    // (4 dtypes x 5 values of p x 64 elements); the un-narrowed one misses 41
    // of 377 in the harness. This is the same family docs/SCALAR.md closed by
    // recording that it *has no rule to infer* -- `hardshrink` narrows and
    // `softshrink` widens -- so it is measured per op, and this op narrows.
    let scale = narrow(scale);
    let mut mask = vec![0u8; numel];
    let mut out = vec![0.0f64; numel];
    for i in 0..numel {
        let keep = draws[i] < p1m;
        mask[i] = keep as u8;
        // `input.mul(mask)` promotes the bool to the input's dtype, so a
        // survivor is `x * 1` and a casualty is `x * 0` -- which keeps a signed
        // zero signed and turns an infinity into NaN, both measured. Then
        // `.mul_(scale)`, on the OUTPUT and not on the mask.
        let masked = narrow(source[i] * if keep { 1.0 } else { 0.0 });
        out[i] = narrow(masked * scale);
    }

    let out_t = write_flat(OP, Flat::Float(out), dims.clone(), &device, tag)?;
    let mask_t = Tensor::from_vec(mask, dims, &device).map_err(|e| candle_err(OP, e))?;
    let pair = [
        crate::tensor::promote(py, finish(py, out_t, tag)?)?,
        crate::tensor::promote(py, finish(py, mask_t, TorchDType::Bool)?)?,
    ];
    Ok(PyTuple::new(py, pair)?.into_any().unbind())
}

/// Row-major strides for a contiguous shape, as element counts.
fn contiguous_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// `aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor`
///
/// Written out by hand rather than routed to `candle`'s `scatter`, because the
/// two do not agree on what a valid call is: candle requires
/// `index.dims() == src.dims()` and `self.dims() == src.dims()` off the scatter
/// axis, while torch requires only that `index` be **no larger than** either.
/// The generation loop uses exactly the shape candle rejects -- `scatter` of a
/// `(batch, k)` index into a `(batch, vocab)` row of logits -- so borrowing
/// candle's checks would refuse the call that matters.
///
/// Duplicate indices resolve to the *last* write in iteration order, matching
/// upstream (measured: scattering `[1,2,3]` at index `[0,0,0]` leaves 3).
fn scatter_src(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.scatter.src";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let dim_raw = dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?;
    let index = tensor_arg(OP, args, kwargs, 2, "index")?;
    let src = tensor_arg(OP, args, kwargs, 3, "src")?;

    let tag = input.tag();
    if src.tag() != tag {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "scatter(): Expected self.dtype to be equal to src.dtype",
        ));
    }
    // torch accepts both int32 and int64 here (measured -- an `int32` index is
    // *not* refused, unlike a `uint8` mask in `masked_fill`).
    if !matches!(index.tag(), TorchDType::Int64 | TorchDType::Int32) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "scatter(): Expected dtype int32 or int64 for index, got {}",
            index.tag().name()
        )));
    }

    let rank = input.tensor()?.rank();
    if rank == 0 {
        return Err(not_implemented(
            "aten.scatter.src: 0-dim self not implemented in torch._C shim",
        ));
    }
    let dim = normalise_dim(OP, dim_raw, rank)?;
    let self_dims = input.tensor()?.dims().to_vec();
    let idx_dims = index.tensor()?.dims().to_vec();
    let src_dims = src.tensor()?.dims().to_vec();
    if idx_dims.len() != rank {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Index tensor must have the same number of dimensions as self tensor",
        ));
    }
    if src_dims.len() != rank {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Index tensor must have the same number of dimensions as src tensor",
        ));
    }
    for d in 0..rank {
        if idx_dims[d] > src_dims[d] || (d != dim && idx_dims[d] > self_dims[d]) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Expected index {idx_dims:?} to be no larger than self {self_dims:?} apart \
                 from dimension {dim} and to be no larger size than src {src_dims:?}"
            )));
        }
    }

    let mut out = read_flat(OP, input.tensor()?, tag)?;
    let source = read_flat(OP, src.tensor()?, tag)?;
    let positions = match read_flat(OP, index.tensor()?, index.tag())? {
        Flat::Int(v) => v,
        Flat::Float(_) => unreachable!("the index dtype was checked above"),
    };

    let self_strides = contiguous_strides(&self_dims);
    let idx_strides = contiguous_strides(&idx_dims);
    let src_strides = contiguous_strides(&src_dims);
    let count: usize = idx_dims.iter().product();

    let mut coord = vec![0usize; rank];
    for _ in 0..count {
        let idx_off: usize = coord.iter().zip(&idx_strides).map(|(c, s)| c * s).sum();
        let target = positions[idx_off];
        if target < 0 || target as usize >= self_dims[dim] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "index {target} is out of bounds for dimension {dim} with size {}",
                self_dims[dim]
            )));
        }
        let src_off: usize = coord.iter().zip(&src_strides).map(|(c, s)| c * s).sum();
        let self_off: usize = coord
            .iter()
            .enumerate()
            .map(|(d, c)| if d == dim { target as usize } else { *c } * self_strides[d])
            .sum();
        match (&source, &mut out) {
            (Flat::Float(s), Flat::Float(o)) => o[self_off] = s[src_off],
            (Flat::Int(s), Flat::Int(o)) => o[self_off] = s[src_off],
            _ => unreachable!("self and src share a dtype, checked above"),
        }
        for d in (0..rank).rev() {
            coord[d] += 1;
            if coord[d] < idx_dims[d] {
                break;
            }
            coord[d] = 0;
        }
    }

    let device = input.tensor()?.device().clone();
    let tensor = write_flat(OP, out, self_dims, &device, tag)?;
    finish(py, tensor, tag)
}

/// `aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor`
///
/// `scatter.src` read backwards, and written the same way for the same reason:
/// candle's `Tensor::gather` exists but does not agree with torch about what a
/// valid call is, so borrowing it would move the disagreement out of sight
/// rather than remove it. Three measured differences, each of which would be a
/// silent divergence:
///
/// 1. **Index dtype.** candle accepts `u8`/`u32`/`i64`; torch accepts exactly
///    `int32` and `int64` and refuses the rest by name. A `uint8` index is a
///    *mask* everywhere else in this shim, and candle would silently read it as
///    positions.
/// 2. **Out of range.** candle's `i64` path reaches `as_usize()` on a negative
///    index, so `-1` becomes a huge `usize` and the error names that number
///    instead of `-1`. torch says `index -1 is out of bounds for dimension 1
///    with size 3`, and that text is what a caller debugs against. There is no
///    negative-index convention here: unlike `select`/`slice`, `gather` does
///    **not** wrap (measured).
/// 3. **Contiguity.** candle refuses a non-contiguous `self` or `index`
///    outright (`RequiresContiguous`); torch gathers from a transposed tensor
///    without comment (measured on `arange(12).reshape(3,4).t()`), and BERT's
///    path arrives that way.
///
/// The rank rule is upstream's `ensure_nonempty_dim`, i.e. `max(rank, 1)` on
/// both sides, which is why a 0-d `self` accepts a 1-d index (`gather(tensor(7.),
/// 0, tensor([0,0]))` -> `[7., 7.]`) and a 1-d `self` accepts a 0-d index
/// (-> 0-d), but a 0-d `self` with a 2-d index is refused. Guessing "ranks must
/// be equal" would have refused two calls torch answers.
///
/// The output has the **index's** shape, not `self`'s: off-axis the index may
/// be *smaller* than `self` (and the extra rows are simply not read), and along
/// `dim` it may be *longer* (values repeat). Both measured.
///
/// `sparse_grad` is accepted and ignored -- it selects an autograd
/// representation, and there is no autograd here. Upstream's forward answer is
/// the same for either value (measured).
fn gather_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.gather.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let dim_raw = dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?;
    let index = tensor_arg(OP, args, kwargs, 2, "index")?;
    let _sparse_grad = bool_arg(args, kwargs, 3, "sparse_grad")?;

    if !matches!(index.tag(), TorchDType::Int64 | TorchDType::Int32) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "gather(): Expected dtype int32/int64 for index",
        ));
    }

    let tag = input.tag();
    // `ensure_nonempty_dim`/`ensure_nonempty_size`: a 0-d tensor counts as one
    // dimension of extent 1 for the purposes of both the rank check and the
    // bounds check. The *output* still gets the index's real shape.
    let self_dims: Vec<usize> = if input.tensor()?.rank() == 0 {
        vec![1]
    } else {
        input.tensor()?.dims().to_vec()
    };
    let idx_shape = index.tensor()?.dims().to_vec();
    let idx_dims: Vec<usize> = if idx_shape.is_empty() {
        vec![1]
    } else {
        idx_shape.clone()
    };
    if idx_dims.len() != self_dims.len() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Index tensor must have the same number of dimensions as input tensor",
        ));
    }
    let dim = normalise_dim(OP, dim_raw, input.tensor()?.rank())?;
    for d in 0..self_dims.len() {
        if d != dim && idx_dims[d] > self_dims[d] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Size does not match at dimension {d} expected index {idx_dims:?} \
                 to be no larger than self {self_dims:?} apart from dimension {dim}"
            )));
        }
    }

    let source = read_flat(OP, input.tensor()?, tag)?;
    let positions = match read_flat(OP, index.tensor()?, index.tag())? {
        Flat::Int(v) => v,
        Flat::Float(_) => unreachable!("the index dtype was checked above"),
    };

    let self_strides = contiguous_strides(&self_dims);
    let idx_strides = contiguous_strides(&idx_dims);
    let count: usize = idx_dims.iter().product();
    let rank = self_dims.len();

    let mut out = match &source {
        Flat::Float(_) => Flat::Float(vec![0.0f64; count]),
        Flat::Int(_) => Flat::Int(vec![0i64; count]),
    };
    let mut coord = vec![0usize; rank];
    for _ in 0..count {
        let idx_off: usize = coord.iter().zip(&idx_strides).map(|(c, s)| c * s).sum();
        let target = positions[idx_off];
        if target < 0 || target as usize >= self_dims[dim] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "index {target} is out of bounds for dimension {dim} with size {}",
                self_dims[dim]
            )));
        }
        let self_off: usize = coord
            .iter()
            .enumerate()
            .map(|(d, c)| if d == dim { target as usize } else { *c } * self_strides[d])
            .sum();
        match (&source, &mut out) {
            (Flat::Float(s), Flat::Float(o)) => o[idx_off] = s[self_off],
            (Flat::Int(s), Flat::Int(o)) => o[idx_off] = s[self_off],
            _ => unreachable!("out was built from source's own variant"),
        }
        for d in (0..rank).rev() {
            coord[d] += 1;
            if coord[d] < idx_dims[d] {
                break;
            }
            coord[d] = 0;
        }
    }

    let device = input.tensor()?.device().clone();
    let tensor = write_flat(OP, out, idx_shape, &device, tag)?;
    finish(py, tensor, tag)
}

/// Round a vector through the storage dtype and back, in one candle pass.
///
/// `multinomial`'s fast path needs this twice, because upstream writes both
/// intermediates into tensors of the input's dtype: `q = empty_like(self)`
/// narrows the exponential draws, and `at::div_out(q, self, q)` narrows the
/// ratio. Doing the arithmetic in `f64` throughout and narrowing once at the
/// end would give a *different argmax* wherever two categories are within an
/// ulp of each other, which is precisely where a sampler's choice is decided.
fn narrow_through(
    op: &str,
    values: Vec<f64>,
    storage: candle_core::DType,
    device: &Device,
) -> PyResult<Vec<f64>> {
    if storage == candle_core::DType::F64 {
        return Ok(values);
    }
    let n = values.len();
    if n == 0 {
        return Ok(values);
    }
    Tensor::from_vec(values, n, device)
        .and_then(|t| t.fast_to(storage))
        .and_then(|t| widen_f64(&t))
        .and_then(|t| t.to_vec1::<f64>())
        .map_err(|e| candle_err(op, e))
}

/// `aten::multinomial(Tensor self, SymInt num_samples, bool replacement=False,
///                    *, Generator? generator=None) -> Tensor`
///
/// The only op here that draws, and therefore the only one whose *answer*
/// depends on consuming torch's stream in torch's order. Two facts decide it,
/// and both were measured rather than assumed (docs/SAMPLING.md §2):
///
/// **1. There are two algorithms, and the branch is not the one the argument
/// name suggests.** `multinomial_out` takes a Gumbel-style fast path when
/// `!replacement` **or `num_samples == 1`** -- so the call
/// `GenerationMixin._sample` makes, `multinomial(probs, num_samples=1)`, takes
/// it even though `replacement` defaults to False and the docs describe the
/// cumulative-sum kernel. The fast path is
///
///     q = empty_like(self).exponential_(1);  q = self / q;  argmax(q, -1, keepdim)
///
/// (`topk` instead of `argmax` when more than one sample is wanted), which
/// consumes `2 * numel` 32-bit words regardless of `num_samples`. The
/// with-replacement kernel consumes `2 * n_dist * num_samples`. Counting words
/// on real torch is what distinguished them: `multinomial(ones(100), 1)`
/// advances the generator by 200 words, `multinomial(ones(100), 3,
/// replacement=True)` by 6.
///
/// **2. Every intermediate is narrowed to the input dtype.** See
/// `narrow_through`.
///
/// The with-replacement kernel's cumulative sum accumulates *in the input
/// dtype* (`scalar_t sum = 0; sum += val;`), sets the last bucket to exactly 1
/// before searching, and compares `cum_prob < uniform_sample` with the
/// left-biased binary search transcribed below. Both paths were checked against
/// upstream on a `(3, 11)` distribution over six seeds: 12/12 identical index
/// lists, no tolerance involved.
fn multinomial_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.multinomial.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let num_samples =
        int_arg(args, kwargs, 1, "num_samples")?.ok_or_else(|| missing(OP, "num_samples"))?;
    let replacement = bool_arg(args, kwargs, 2, "replacement")?.unwrap_or(false);
    generator_arg(OP, args, kwargs, 3, "generator")?;

    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "multinomial only supports floating-point dtypes for input, got: {}",
            scalar_type_name(tag)
        )));
    }
    let rank = input.tensor()?.rank();
    if rank != 1 && rank != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "prob_dist must be 1 or 2 dim",
        ));
    }
    let dims = input.tensor()?.dims().to_vec();
    let n_categories = dims[rank - 1];
    let n_dist = if rank == 2 { dims[0] } else { 1 };
    if num_samples <= 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "cannot sample n_sample <= 0 samples",
        ));
    }
    let n_sample = num_samples as usize;
    if !replacement && n_sample > n_categories {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "cannot sample n_sample > prob_dist.size(-1) samples without replacement",
        ));
    }

    let storage = PyDtype::new(tag).storage(OP)?;
    let device = input.tensor()?.device().clone();
    let probs = match read_flat(OP, input.tensor()?, tag)? {
        Flat::Float(v) => v,
        Flat::Int(_) => unreachable!("the integral dtypes were refused above"),
    };
    let out_dims: Vec<usize> = if rank == 2 {
        vec![n_dist, n_sample]
    } else {
        vec![n_sample]
    };

    let picks: Vec<i64> = if !replacement || n_sample == 1 {
        // Upstream's sanity checks, in upstream's order. `(max < INFINITY) &
        // (min >= 0)` is a single expression there; a NaN anywhere fails it
        // because every comparison against NaN is false.
        if probs.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "probability tensor contains either `inf`, `nan` or element < 0",
            ));
        }
        for row in probs.chunks(n_categories.max(1)) {
            if row.iter().sum::<f64>() == 0.0 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "invalid multinomial distribution (sum of probabilities <= 0)",
                ));
            }
        }

        let q = {
            let mut gen = crate::rng::default_generator();
            crate::rng::exponential_serial(&mut gen, probs.len(), 1.0)
        };
        let q = narrow_through(OP, q, storage, &device)?;
        // `at::div_out(q, self, q)` runs in `opmath_type<scalar_t>` -- float
        // for everything but float64 -- and stores back into `scalar_t`.
        let ratio: Vec<f64> = if storage == candle_core::DType::F64 {
            probs.iter().zip(&q).map(|(p, d)| p / d).collect()
        } else {
            probs
                .iter()
                .zip(&q)
                .map(|(p, d)| ((*p as f32) / (*d as f32)) as f64)
                .collect()
        };
        let ratio = narrow_through(OP, ratio, storage, &device)?;

        let mut picks = Vec::with_capacity(n_dist * n_sample);
        for i in 0..n_dist {
            let row = &ratio[i * n_categories..(i + 1) * n_categories];
            if n_sample == 1 {
                // `at::argmax_out` -- the *first* maximum wins.
                let mut best = 0usize;
                for j in 1..n_categories {
                    if cmp_torch_f64(row[j], row[best]) == std::cmp::Ordering::Greater {
                        best = j;
                    }
                }
                picks.push(best as i64);
            } else {
                // `at::topk_out(vals, result, q, n_sample)` -- largest first,
                // sorted, and stable among ties for the reason `order_along`
                // records.
                let mut order: Vec<usize> = (0..n_categories).collect();
                order.sort_by(|&a, &b| cmp_torch_f64(row[b], row[a]));
                picks.extend(order.into_iter().take(n_sample).map(|j| j as i64));
            }
        }
        picks
    } else {
        // `multinomial_with_replacement_apply`.
        //
        // **The cumulative distribution is *not* kept in the input's dtype**,
        // and that is the one thing here that had to be measured rather than
        // read off the shape of the rest of this file. A `bfloat16` input runs
        // the accumulation and the normalising division in `float`, with no
        // narrowing anywhere; carrying `scalar_t` through instead -- which is
        // what the surrounding code makes the natural guess -- put this shim on
        // a different bucket from upstream on 2 of 140 measured bf16 draws.
        //
        // The measurement is in docs/SAMPLING.md §5: 20,000 draws from a known
        // MT stream bracket every one of the eleven bucket boundaries, and the
        // brackets are ~2e-4 wide, which is a fifth of `bfloat16`'s spacing
        // there. A `bfloat16` cumulative distribution lands outside six of the
        // eleven; a `float` one lands inside all eleven, for `float16` too.
        //
        // So the accumulate type is `at::acc_type<scalar_t, false>` -- float
        // for float16/bfloat16/float32, double for float64 -- exactly as the
        // *other* kernels here use `opmath_type`, and the fast path above is
        // the odd one out precisely because it materialises real tensors of
        // `self`'s dtype.
        let double_acc = storage == candle_core::DType::F64;
        let add_acc = |a: f64, b: f64| -> f64 {
            if double_acc {
                a + b
            } else {
                ((a as f32) + (b as f32)) as f64
            }
        };
        let div_acc = |a: f64, b: f64| -> f64 {
            if double_acc {
                a / b
            } else {
                ((a as f32) / (b as f32)) as f64
            }
        };
        let mut picks = Vec::with_capacity(n_dist * n_sample);
        let mut gen = crate::rng::default_generator();
        for i in 0..n_dist {
            let row = &probs[i * n_categories..(i + 1) * n_categories];
            let mut cum = vec![0.0f64; n_categories];
            let mut sum = 0.0f64;
            for (j, &val) in row.iter().enumerate() {
                if val < 0.0 {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "invalid multinomial distribution (encountering probability entry < 0)",
                    ));
                }
                if !val.is_finite() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "invalid multinomial distribution (encountering probability entry = infinity or NaN)",
                    ));
                }
                sum = add_acc(sum, val);
                cum[j] = sum;
            }
            if !(sum > 0.0) {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "invalid multinomial distribution (sum of probabilities <= 0)",
                ));
            }
            for slot in cum.iter_mut() {
                *slot = div_acc(*slot, sum);
            }
            // "Make sure the last cumulative distribution bucket sums to 1",
            // upstream's comment and upstream's assignment -- it is written
            // inside the sample loop there, which is the same thing done once.
            cum[n_categories - 1] = 1.0;
            for _ in 0..n_sample {
                let sample = crate::rng::uniform_sample_f64(&mut gen, 0.0, 1.0);
                let (mut left, mut right) = (0usize, n_categories);
                while right - left > 0 {
                    let mid = left + (right - left) / 2;
                    if cum[mid] < sample {
                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }
                picks.push(left as i64);
            }
        }
        drop(gen);
        picks
    };

    let out = Tensor::from_vec(picks, out_dims, &device).map_err(|e| candle_err(OP, e))?;
    finish(py, out, TorchDType::Int64)
}

// ---------------------------------------------------------------------------
// Argument plumbing
//
// aten schemas allow positional or keyword for everything before the `*`, and
// the vendored Python layer will use both spellings, so each argument is looked
// up by index and by name.
// ---------------------------------------------------------------------------

/// An aten `Scalar`, kept in the category it arrived as.
///
/// The category is not cosmetic: torch's dtype inference asks "was this an
/// integer?" in several of the ops above (`arange` picks int64 or the default
/// float from it, `pow` decides whether to float the result), and collapsing
/// everything to `f64` on the way in would throw away the only information
/// those rules read. Python `bool` lands in `Int` -- it subclasses `int` and
/// torch treats it as an integral scalar in these positions.
#[derive(Clone, Copy)]
enum Scalar {
    Int(i64),
    Float(f64),
}

impl Scalar {
    fn is_int(self) -> bool {
        matches!(self, Scalar::Int(_))
    }

    fn as_f64(self) -> f64 {
        match self {
            Scalar::Int(v) => v as f64,
            Scalar::Float(v) => v,
        }
    }

    fn as_i64(self) -> i64 {
        match self {
            Scalar::Int(v) => v,
            Scalar::Float(v) => v as i64,
        }
    }
}

fn missing(op: &str, name: &str) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!("{op}: missing required argument '{name}'"))
}

fn scalar_arg(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<Scalar>> {
    let value = match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => value,
        _ => return Ok(None),
    };
    // Order matters: `bool` is a subclass of `int` in Python.
    if value.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(Some(Scalar::Int(i64::from(value.extract::<bool>()?))));
    }
    if value.is_instance_of::<pyo3::types::PyInt>() {
        return Ok(Some(Scalar::Int(value.extract()?)));
    }
    if value.is_instance_of::<pyo3::types::PyFloat>() {
        return Ok(Some(Scalar::Float(value.extract()?)));
    }
    // torch accepts a zero-dim tensor anywhere a `Scalar` is taken, and the
    // overload resolver in bootstrap.py binds one here for the same reason, so
    // refusing it at the kernel would make the two disagree.
    if let Ok(tensor) = value.extract::<PyTensorBase>() {
        if tensor.tensor()?.rank() != 0 {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{op}: argument '{name}' as a tensor must be zero-dim, got {}D",
                tensor.tensor()?.rank()
            )));
        }
        let as_f64 = widen_f64(tensor.tensor()?)
            .and_then(|t| t.to_scalar::<f64>())
            .map_err(|err| candle_err(op, err))?;
        return Ok(Some(if tensor.tag().is_floating_point() {
            Scalar::Float(as_f64)
        } else {
            Scalar::Int(as_f64 as i64)
        }));
    }
    // A number that is not one of Python's own -- a `numpy` scalar, which is
    // what `np.prod([4, 4])` returns. `vits` reaches this: `modeling_vits.py:1379`
    // is `predicted_lengths * np.prod(self.config.upsample_rates)`, and upstream
    // takes it (measured: `torch.tensor([1,2]) * np.int64(16)` fires
    // `aten.mul.Tensor` and keeps `int64`).
    //
    // Read through `__index__` and `__float__` rather than by importing numpy,
    // for two reasons: the shim has no numpy dependency, and the protocol is
    // what upstream's own `Scalar` parser uses -- anything that can present
    // itself as a number is one. `__index__` is tried first because
    // `np.int64` has BOTH, and taking the float would turn an integer into
    // `Scalar::Float` and, through `arith_tag`'s wrapped-number rule, an
    // `int64` tensor into a `float32` one.
    //
    // `PyTensorBase` was already handled above, so nothing that is a tensor
    // reaches here.
    if value.hasattr("__index__").unwrap_or(false) {
        if let Ok(as_int) = value.call_method0("__index__").and_then(|v| v.extract::<i64>()) {
            return Ok(Some(Scalar::Int(as_int)));
        }
    }
    if value.hasattr("__float__").unwrap_or(false) {
        if let Ok(as_float) = value.call_method0("__float__").and_then(|v| v.extract::<f64>()) {
            return Ok(Some(Scalar::Float(as_float)));
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{op}: argument '{name}' must be a number, got {}",
        value.get_type().name().map(|n| n.to_string()).unwrap_or_default()
    )))
}

fn dtype_arg(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<TorchDType>> {
    match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract::<PyDtype>()?.tag())),
        _ => Ok(None),
    }
}

fn int_arg(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<i64>> {
    match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract()?)),
        _ => Ok(None),
    }
}

fn dim_arg(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<isize>> {
    match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract()?)),
        _ => Ok(None),
    }
}

fn bool_arg(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<bool>> {
    match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract()?)),
        _ => Ok(None),
    }
}

/// torch's negative-dimension convention, with torch's error message shape.
fn normalise_dim(op: &str, dim: isize, rank: usize) -> PyResult<usize> {
    // torch treats a zero-dim tensor as one-dimensional for indexing purposes.
    let extent = rank.max(1) as isize;
    let index = if dim < 0 { dim + extent } else { dim };
    if index < 0 || index >= extent {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "{op}: Dimension out of range (expected to be in range of [{}, {}], but got {dim})",
            -extent,
            extent - 1
        )));
    }
    Ok(index as usize)
}

/// Wraps a finished tensor, routing the `torch.bool` tag through the one
/// constructor that is allowed to attach it (BOOL.md §6.3 item 1).
fn finish(py: Python<'_>, tensor: Tensor, tag: TorchDType) -> PyResult<Py<PyAny>> {
    let wrapped = if tag == TorchDType::Bool {
        PyTensorBase::boolean(tensor)?
    } else {
        PyTensorBase::new(tensor)?
    };
    Ok(wrapped.into_pyobject(py)?.into_any().unbind())
}

/// Every argument name this file reads a keyword by, as an interned Python
/// string that is built once per process instead of once per argument per
/// call.
///
/// **This is a lookup table, not a source of truth.** A name that is missing
/// falls through to `None` and the caller builds the string the old way, so
/// the table can never make a call answer differently -- only more slowly.
/// That is why it is safe to write it by hand and safe to leave it
/// incomplete; adding an argument to a kernel without adding it here costs
/// that kernel some nanoseconds and nothing else.
///
/// Why it is worth having: `bootstrap.py` binds every argument by keyword, so
/// `optional` reaches the dict on essentially every read, and pyo3's
/// `get_item(&str)` has to **allocate a fresh `PyString` and hash it from
/// scratch** each time (a new string has no cached hash). In the profile that
/// was `unicode_decode_utf8` 146 + `pysiphash` 139 of 6043 samples, plus the
/// allocate/free traffic underneath it -- `dim_arg` -> `PyString::new` ->
/// `unicode_dealloc` -> `_PyObject_Free` was a visible chain into CPython's
/// thread-local allocator. An interned string is allocated once, carries its
/// hash, and matches on pointer identity inside the dict probe.
///
/// The names were extracted mechanically from the helper call sites rather
/// than typed out, so a name here cannot disagree with the name the kernel
/// asks for. docs/DISPATCH.md §4.
fn interned_name<'py>(py: Python<'py>, name: &str) -> Option<&'py Bound<'py, PyString>> {
    // BORROWED, and borrowed from a process-lifetime cache. `intern!` stores
    // the object in a `PyOnceLock` that is never cleared, so the reference is
    // valid for as long as the interpreter is; nothing here increfs it and
    // nothing may decref it.
    Some(match name {
        "self" => intern!(py, "self"),
        "other" => intern!(py, "other"),
        "dim" => intern!(py, "dim"),
        "dtype" => intern!(py, "dtype"),
        "size" => intern!(py, "size"),
        "value" => intern!(py, "value"),
        "keepdim" => intern!(py, "keepdim"),
        "exponent" => intern!(py, "exponent"),
        "alpha" => intern!(py, "alpha"),
        "weight" => intern!(py, "weight"),
        "src" => intern!(py, "src"),
        "mat2" => intern!(py, "mat2"),
        "indices" => intern!(py, "indices"),
        "index" => intern!(py, "index"),
        "generator" => intern!(py, "generator"),
        "beta" => intern!(py, "beta"),
        "tensors" => intern!(py, "tensors"),
        "step" => intern!(py, "step"),
        "start" => intern!(py, "start"),
        "s" => intern!(py, "s"),
        "min" => intern!(py, "min"),
        "max" => intern!(py, "max"),
        "mask" => intern!(py, "mask"),
        "input" => intern!(py, "input"),
        "end" => intern!(py, "end"),
        "condition" => intern!(py, "condition"),
        "bias" => intern!(py, "bias"),
        "dim0" => intern!(py, "dim0"),
        "dim1" => intern!(py, "dim1"),
        "dims" => intern!(py, "dims"),
        "device" => intern!(py, "device"),
        "layout" => intern!(py, "layout"),
        "memory_format" => intern!(py, "memory_format"),
        "fill_value" => intern!(py, "fill_value"),
        "normalized_shape" => intern!(py, "normalized_shape"),
        "eps" => intern!(py, "eps"),
        "half_to_float" => intern!(py, "half_to_float"),
        "accumulate" => intern!(py, "accumulate"),
        "attn_mask" => intern!(py, "attn_mask"),
        "batch1" => intern!(py, "batch1"),
        "batch2" => intern!(py, "batch2"),
        "bins" => intern!(py, "bins"),
        "descending" => intern!(py, "descending"),
        "dilation" => intern!(py, "dilation"),
        "dropout_p" => intern!(py, "dropout_p"),
        "elements" => intern!(py, "elements"),
        "from" => intern!(py, "from"),
        "groups" => intern!(py, "groups"),
        "high" => intern!(py, "high"),
        "invert" => intern!(py, "invert"),
        "is_causal" => intern!(py, "is_causal"),
        "k" => intern!(py, "k"),
        "key" => intern!(py, "key"),
        "largest" => intern!(py, "largest"),
        "low" => intern!(py, "low"),
        "mat1" => intern!(py, "mat1"),
        "mean" => intern!(py, "mean"),
        "num_samples" => intern!(py, "num_samples"),
        "output_padding" => intern!(py, "output_padding"),
        "padding" => intern!(py, "padding"),
        "query" => intern!(py, "query"),
        "replacement" => intern!(py, "replacement"),
        "scale" => intern!(py, "scale"),
        "sorted" => intern!(py, "sorted"),
        "sparse_grad" => intern!(py, "sparse_grad"),
        "split_size" => intern!(py, "split_size"),
        "split_sizes" => intern!(py, "split_sizes"),
        "std" => intern!(py, "std"),
        "stride" => intern!(py, "stride"),
        "test_elements" => intern!(py, "test_elements"),
        "threshold" => intern!(py, "threshold"),
        "to" => intern!(py, "to"),
        "transposed" => intern!(py, "transposed"),
        "values" => intern!(py, "values"),
        _ => return None,
    })
}

fn optional<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if index < args.len() {
        return Ok(Some(args.get_item(index)?));
    }
    match kwargs {
        // OWNED either way: `get_item` is `PyDict_GetItemRef`, which returns a
        // *new* reference, and the `Bound` the caller receives owns it. The
        // key is borrowed in the interned arm and temporary in the fallback
        // arm, and neither is affected by what happens to the value.
        Some(kwargs) => match interned_name(kwargs.py(), name) {
            Some(key) => kwargs.get_item(key),
            None => kwargs.get_item(name),
        },
        None => Ok(None),
    }
}

fn required<'py>(
    op: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    optional(args, kwargs, index, name)?.ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err(format!("{op}: missing required argument '{name}'"))
    })
}

fn tensor_arg(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<PyTensorBase> {
    let value = required(op, args, kwargs, index, name)?;
    // The wrapper, not the bare candle tensor: the torch dtype tag lives on
    // the wrapper and `torch.bool` is invisible in the candle dtype.
    value.extract::<PyTensorBase>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{op}: argument '{name}' must be a torch._C.TensorBase, got {}",
            value.get_type().name().map(|n| n.to_string()).unwrap_or_default()
        ))
    })
}

/// `tensor_arg` for a `Tensor?` slot. `None` and an absent argument are the
/// same answer -- `native_layer_norm(x, [4], None, None, eps)` is how a
/// `nn.LayerNorm(elementwise_affine=False)` arrives.
fn optional_tensor_arg(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<PyTensorBase>> {
    match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract::<PyTensorBase>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "{op}: argument '{name}' must be a torch._C.TensorBase or None, got {}",
                value.get_type().name().map(|n| n.to_string()).unwrap_or_default()
            ))
        })?)),
        _ => Ok(None),
    }
}

/// A factory op's `device=` slot, as a **label**.
///
/// The label rather than a resolved handle, because `meta` has no handle to
/// resolve to: a factory has to see `meta` and build a storage-less tensor
/// instead of allocating one. `device_arg` below is this plus `resolve()`, for
/// the kernels that have no meta path and should fail loudly if handed one.
///
/// Absent means `fallback`. For most factories that is the CPU -- the
/// process-wide default device lives above this layer, in the torch-function
/// mode stack that `bootstrap.py` consults before a call ever reaches the
/// dispatcher (docs/META.md §8), which is where upstream puts it too.
fn device_arg_or_label(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
    fallback: &PyDevice,
) -> PyResult<PyDevice> {
    match optional(args, kwargs, index, name)? {
        // torch accepts a plain string, another `device`, or an integer
        // wherever a device is taken; `coerce` is the one place that knows
        // which of those are legal and what each means.
        Some(value) if !value.is_none() => PyDevice::coerce(&value),
        _ => Ok(fallback.clone()),
    }
}

/// Arguments the shim has no answer for. Ignoring them silently would make the
/// call look supported; a `layout=torch.sparse_coo` that is dropped on the floor
/// produces a wrong answer with no trace.
fn reject_unsupported(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    fields: &[(usize, &str)],
) -> PyResult<()> {
    for (index, name) in fields {
        if let Some(value) = optional(args, kwargs, *index, name)? {
            if !value.is_none() {
                return Err(not_implemented(format!(
                    "{op}: argument '{name}' not implemented in torch._C shim (got {value})"
                )));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar -> dtype conversion, as torch does it
//
// `c10::checked_convert` refuses a scalar the destination dtype cannot hold.
// candle has no such check: an out-of-range integer wraps (two's complement)
// and an out-of-range float saturates to infinity, both without a word. The
// golden harness caught exactly that pair -- `full([3], 1e6, float16)` gave
// `inf` where torch raises, and `full([3], 2**31, int32)` gave `-2**31` where
// torch raises -- so the check is transcribed here rather than approximated.
//
// The rules below are measured against torch 2.13.0, not read off the C++.
// Two of them are counter-intuitive enough that guessing would have got them
// wrong, and both are load-bearing for cases the harness already passes:
//
//   * A negative value converted to an *unsigned* dtype is allowed to wrap,
//     as long as its magnitude fits. `full(-1, uint8) == 255` is legal in
//     torch; `full(-300, uint8)` is not. (c10 spells this "allow for negative
//     numbers to wrap using two's complement arithmetic".)
//   * Half / BFloat16 / Float8 skip the check entirely when the tensor has
//     exactly one element, and saturate silently instead. That is an upstream
//     inconsistency -- `fill_` takes a CPU numel==1 fast path whose conversion
//     is unchecked -- but `full([], 1e6, float16) == inf` while
//     `full([3], 1e6, float16)` raises, on real torch, so a shim that always
//     refuses would diverge from torch in the other direction.
// ---------------------------------------------------------------------------

/// The name torch puts in the message. These are C++ type spellings, so they
/// are not derivable from the torch dtype name -- each one was read off a real
/// `RuntimeError` from torch 2.13.0.
fn c10_name(dtype: TorchDType) -> &'static str {
    use TorchDType::*;
    match dtype {
        Float64 => "double",
        Float32 => "float",
        Float16 => "c10::Half",
        BFloat16 => "c10::BFloat16",
        Int64 => "int64_t",
        Int32 => "int",
        Int16 => "int16_t",
        Int8 => "int8_t",
        UInt8 => "uint8_t",
        UInt16 => "uint16_t",
        UInt32 => "uint32_t",
        UInt64 => "uint64_t",
        Bool => "bool",
        Float8E4M3FN => "c10::Float8_e4m3fn",
        Float8E5M2 => "c10::Float8_e5m2",
        other => other.name(),
    }
}

/// `c10::toString(ScalarType)` -- the spelling torch puts in a "not
/// implemented for '<X>'" message. A *third* naming of the same set, distinct
/// from both `TorchDType::name()` (`uint32`) and `c10_name` (`uint32_t`), and
/// like those it is not derivable: only the entries this shim can actually
/// reach are listed, each read off a real torch error.
fn scalar_type_name(dtype: TorchDType) -> &'static str {
    use TorchDType::*;
    match dtype {
        Float32 => "Float",
        Float64 => "Double",
        Float16 => "Half",
        BFloat16 => "BFloat16",
        UInt8 => "Byte",
        UInt16 => "UInt16",
        UInt32 => "UInt32",
        UInt64 => "UInt64",
        Int8 => "Char",
        Int16 => "Short",
        Int32 => "Int",
        Int64 => "Long",
        Bool => "Bool",
        Float8E4M3FN => "Float8_e4m3fn",
        other => other.name(),
    }
}

/// Does torch have an `arange_cpu` kernel for this dtype? Measured against
/// torch 2.13.0 over every dtype this shim can store.
fn arange_has_cpu_kernel(dtype: TorchDType) -> bool {
    use TorchDType::*;
    !matches!(
        dtype,
        UInt16
            | UInt32
            | UInt64
            | Bool
            | Float8E4M3FN
            | Float8E4M3FNUZ
            | Float8E5M2
            | Float8E5M2FNUZ
            | Float8E8M0FNU
            | Float4E2M1FNX2
    )
}

fn overflow(dtype: TorchDType) -> PyErr {
    // torch's own wording, with no shim prefix: this is torch semantics being
    // reproduced, not a shim limitation, and a caller matching on the message
    // should not have to know which of the two produced it.
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "value cannot be converted to type {} without overflow",
        c10_name(dtype)
    ))
}

/// Largest finite magnitude of a floating dtype; `None` for the others, which
/// are range-checked exactly rather than through `f64`.
fn float_max(dtype: TorchDType) -> Option<f64> {
    use TorchDType::*;
    Some(match dtype {
        Float64 => f64::MAX,
        Float32 => f32::MAX as f64,
        Float16 => 65504.0,
        BFloat16 => 3.3895313892515355e38,
        Float8E4M3FN | Float8E4M3FNUZ => 448.0,
        Float8E5M2 | Float8E5M2FNUZ => 57344.0,
        _ => return None,
    })
}

/// Inclusive integer range, as `(min, max)`. `min` is negative for the signed
/// types and zero for the unsigned ones -- the wrap allowance is applied by
/// the caller, not folded in here, so the two rules stay separable.
fn int_range(dtype: TorchDType) -> Option<(i64, i64)> {
    use TorchDType::*;
    Some(match dtype {
        Int64 => (i64::MIN, i64::MAX),
        Int32 => (i32::MIN as i64, i32::MAX as i64),
        Int16 => (i16::MIN as i64, i16::MAX as i64),
        Int8 => (i8::MIN as i64, i8::MAX as i64),
        UInt8 => (0, u8::MAX as i64),
        UInt16 => (0, u16::MAX as i64),
        UInt32 => (0, u32::MAX as i64),
        UInt64 => (0, i64::MAX),
        _ => return None,
    })
}

fn checked_convert(
    fill: &Bound<'_, PyAny>,
    fill_is_int: bool,
    dtype: TorchDType,
    numel: usize,
) -> PyResult<()> {
    // Every value is a valid `bool`; torch converts by truthiness and never
    // reports an overflow for it.
    if dtype == TorchDType::Bool {
        return Ok(());
    }

    // The upstream numel==1 hole, reproduced deliberately. Restricted to the
    // reduced-precision float types because torch's own fast path checks
    // float/double/int even at one element -- measured, not assumed.
    let unchecked_at_one = matches!(
        dtype,
        TorchDType::Float16
            | TorchDType::BFloat16
            | TorchDType::Float8E4M3FN
            | TorchDType::Float8E4M3FNUZ
            | TorchDType::Float8E5M2
            | TorchDType::Float8E5M2FNUZ
    );
    if numel == 1 && unchecked_at_one {
        return Ok(());
    }

    if fill_is_int {
        // A Python int too large for `i64` raises OverflowError here, which is
        // what torch does too ("int too big to convert") -- from the same
        // place, its own Python-to-Scalar conversion.
        let value: i64 = fill.extract()?;
        if let Some((min, max)) = int_range(dtype) {
            let fits = if value < 0 && min == 0 {
                // Two's-complement wrap, allowed for magnitude <= max.
                value.checked_neg().map(|m| m <= max).unwrap_or(false)
            } else {
                value >= min && value <= max
            };
            if !fits {
                return Err(overflow(dtype));
            }
        } else if let Some(max) = float_max(dtype) {
            if (value as f64).abs() > max {
                return Err(overflow(dtype));
            }
        }
    } else {
        let value: f64 = fill.extract()?;
        if let Some((min, max)) = int_range(dtype) {
            // Integer dtypes have neither infinity nor NaN, so both are
            // refused; a finite value must land inside the range.
            if value.is_nan() || value < min as f64 || value > max as f64 {
                return Err(overflow(dtype));
            }
        } else if let Some(max) = float_max(dtype) {
            // Infinity converts to infinity; only finite-but-too-large is an
            // overflow. NaN is fine -- the float types all have a quiet NaN.
            if value.is_finite() && value.abs() > max {
                return Err(overflow(dtype));
            }
        }
    }
    Ok(())
}

/// **Upstream requires these operands to have equal dtypes and raises
/// otherwise**, so refusing here is reproducing upstream rather than
/// admitting a gap.
///
/// This used to say "torch would promote here; the shim does not", and for
/// five of its six callers that was simply false. docs/PROMOTE.md §1.3
/// measured the full 9x9 grid for each and found the diagonal is the *only*
/// non-raising cell: the matmul family and the two structured kernels do not
/// consult `promote_types` at all.
///
/// ```text
/// mm / matmul     expected m1 and m2 to have the same dtype, but got: float != double
/// bmm / conv      expected scalar type Float but found Double
/// sdpa flash      expected scalar type Float but found Double
/// ```
///
/// The distinction matters because a refusal that calls itself unimplemented
/// puts the op on the work queue DESIGN.md §6 is built on, and these five do
/// not belong there -- implementing "promotion" for them would be a departure
/// from upstream, not a convergence with it.
///
/// Compares the *torch* dtype, so `bool` and `uint8` are not accidentally the
/// same operand type just because candle stores both as `U8`.
fn require_same_dtype(op: &str, lhs: &PyTensorBase, rhs: &PyTensorBase) -> PyResult<TorchDType> {
    if lhs.tag() != rhs.tag() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{op}: expected both operands to have the same dtype, but got: {} != {} \
             -- upstream requires equal dtypes here too and does not promote",
            lhs.tag().name(),
            rhs.tag().name()
        )));
    }
    Ok(lhs.tag())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aten_dispatch_entry, m)?)?;
    m.add_function(wrap_pyfunction!(aten_implemented, m)?)?;
    m.add_function(wrap_pyfunction!(aten_implemented_awaiting_golden, m)?)?;
    m.add_function(wrap_pyfunction!(aten_all_implemented, m)?)?;
    Ok(())
}

#[cfg(test)]
mod widen_tests {
    use super::widen_gemm_operand;
    use candle_core::{DType, Device, Tensor};

    /// Widening a **transposed** operand is bit-for-bit what candle's own
    /// `to_dtype` produces on that same view.
    ///
    /// This is the equality the fast path rests on, and it is the one a wrong
    /// transpose would break: taking `t()` of the base and forgetting to put it
    /// back would still return a tensor of the right *shape* whenever the
    /// operand is square, so a square-only check could not fail. The shapes
    /// here are deliberately non-square, and the values deliberately need real
    /// rounding (`/7.0` is not representable in either reduced format).
    #[test]
    fn widening_a_transposed_operand_matches_candle_bit_for_bit() {
        let dev = Device::Cpu;
        let (rows, cols) = (37usize, 91usize); // non-square, not a multiple of 8
        let src: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32 - 900.0) / 7.0)
            .collect();
        let base = Tensor::from_slice(&src, (rows, cols), &dev).unwrap();

        for reduced in [DType::F16, DType::BF16] {
            let stored = base.to_dtype(reduced).unwrap();
            let view = stored.t().unwrap(); // what `linear` hands the kernel
            assert!(
                !view.layout().is_contiguous(),
                "the case under test must be non-contiguous"
            );

            let mine = widen_gemm_operand(&view, DType::F32).unwrap();
            let theirs = view.to_dtype(DType::F32).unwrap();

            assert_eq!(mine.dims(), theirs.dims());
            assert_eq!(
                mine.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                theirs.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                "{reduced:?} transposed widening disagrees with candle"
            );
        }
    }

    /// The layout survives. If the widened operand came back contiguous the
    /// values would still be right and the whole point would be gone -- the
    /// GEMM would lose `CblasTrans` and re-gather the weight, which is the
    /// 69.60 ms that docs/DTYPE_PERF.md §4 measured.
    #[test]
    fn widening_a_transposed_operand_keeps_it_a_transposed_view() {
        let dev = Device::Cpu;
        let base = Tensor::from_slice(
            &(0..(8 * 5)).map(|i| i as f32).collect::<Vec<_>>(),
            (8usize, 5usize),
            &dev,
        )
        .unwrap();
        let view = base.to_dtype(DType::BF16).unwrap().t().unwrap();
        let widened = widen_gemm_operand(&view, DType::F32).unwrap();
        assert_eq!(widened.dtype(), DType::F32);
        assert!(
            !widened.layout().is_contiguous(),
            "the transpose was flattened -- the operand lost its CblasTrans"
        );
    }

    /// A layout that is neither contiguous nor a plain transpose still gets the
    /// right values. It falls through to candle, and that arm has to stay
    /// correct rather than merely unreached.
    #[test]
    fn an_unrecognised_layout_still_widens_correctly() {
        let dev = Device::Cpu;
        let src: Vec<f32> = (0..60).map(|i| i as f32 / 3.0).collect();
        let base = Tensor::from_slice(&src, (10usize, 6usize), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let strided = base.narrow(0, 2, 5).unwrap().narrow(1, 1, 4).unwrap();
        let mine = widen_gemm_operand(&strided, DType::F32).unwrap();
        let theirs = strided.to_dtype(DType::F32).unwrap();
        assert_eq!(
            mine.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            theirs.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }
}

#[cfg(test)]
mod pow_square_tests {
    use candle_core::{DType, Device, Tensor};

    /// The old path, transcribed: widen to `f64`, libm `pow`, narrow back.
    /// `pow_from_pairs` does exactly this plus two vector copies, and the
    /// copies cannot change a value.
    fn old_path_f32(src: &[f32]) -> Vec<f32> {
        let dev = Device::Cpu;
        let widened: Vec<f64> = Tensor::from_slice(src, src.len(), &dev)
            .unwrap()
            .to_dtype(DType::F64)
            .unwrap()
            .to_vec1::<f64>()
            .unwrap();
        let powed: Vec<f64> = widened.iter().map(|b| b.powf(2.0)).collect();
        Tensor::from_vec(powed, src.len(), &dev)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// Squaring by multiplication is **bit-identical** to the libm round-trip
    /// it replaces, across the whole `f32` range.
    ///
    /// This is the equality `pow_square_fast_path` rests on. It can fail: if
    /// this platform's libm returned anything other than the correctly-rounded
    /// square for exponent 2, or if the exact-square argument in that function's
    /// doc comment were wrong at the extremes, the vectors would differ. The
    /// inputs are chosen to press exactly there -- subnormals (which square to
    /// zero), values whose square overflows `f32` (which must give `inf` on both
    /// sides), negatives, signed zeros, and a sweep of ordinary values that need
    /// real rounding.
    #[test]
    fn squaring_matches_the_libm_round_trip_bit_for_bit() {
        let mut src: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,          // smallest normal
            -f32::MIN_POSITIVE,
            f32::from_bits(1),          // smallest subnormal -- squares to 0
            f32::from_bits(0x007f_ffff), // largest subnormal
            f32::MAX,                   // squares to inf
            -f32::MAX,
            f32::MIN,
            1e-30,
            1e30,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        // Ordinary values that need rounding: /7.0 is not representable.
        src.extend((0..4096).map(|i| (i as f32 - 2048.0) / 7.0));
        // A geometric sweep across the exponent range.
        src.extend((0..200).map(|i| 2.0f32.powi(i - 100) * 1.234_567_9));

        let dev = Device::Cpu;
        let fast = Tensor::from_slice(&src, src.len(), &dev)
            .unwrap()
            .mul(&Tensor::from_slice(&src, src.len(), &dev).unwrap())
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let slow = old_path_f32(&src);

        assert_eq!(fast.len(), slow.len());
        for (i, (a, b)) in fast.iter().zip(slow.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "input {} ({:e}) squared: fast {:e} vs libm round-trip {:e}",
                i,
                src[i],
                a,
                b
            );
        }
    }

    /// The same claim for `f64`, which rests on libm rather than on
    /// representability -- so it is the one that could actually surprise us.
    #[test]
    fn squaring_matches_libm_for_f64_too() {
        let mut src: Vec<f64> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::MAX,
            -f64::MAX,
            f64::INFINITY,
        ];
        src.extend((0..4096).map(|i| (i as f64 - 2048.0) / 7.0));
        src.extend((0..600).map(|i| 2.0f64.powi(i - 300) * 1.234_567_891_234));

        for (i, &b) in src.iter().enumerate() {
            assert_eq!(
                (b * b).to_bits(),
                b.powf(2.0).to_bits(),
                "f64 input {} ({:e}): b*b {:e} vs powf {:e}",
                i,
                b,
                b * b,
                b.powf(2.0)
            );
        }
    }

    /// A **`NaN`** input squares to a `NaN` on both paths. Bit equality is the
    /// wrong assertion here -- `NaN` payloads are not contractual -- so this
    /// asserts the property upstream actually guarantees.
    #[test]
    fn nan_squares_to_nan() {
        assert!((f32::NAN * f32::NAN).is_nan());
        assert!((f64::from(f32::NAN)).powf(2.0).is_nan());
    }
}

fn max_pool2d_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.max_pool2d.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let kernel = shape_arg(OP, args, kwargs, 1, "kernel_size")?;
    let stride_raw = match optional(args, kwargs, 2, "stride")? {
        Some(value) if !value.is_none() => shape_arg(OP, args, kwargs, 2, "stride")?,
        _ => Vec::new(),
    };
    let padding = match optional(args, kwargs, 3, "padding")? {
        Some(value) if !value.is_none() => shape_arg(OP, args, kwargs, 3, "padding")?,
        _ => vec![0],
    };
    let dilation = match optional(args, kwargs, 4, "dilation")? {
        Some(value) if !value.is_none() => shape_arg(OP, args, kwargs, 4, "dilation")?,
        _ => vec![1],
    };
    let ceil_mode = bool_arg(args, kwargs, 5, "ceil_mode")?.unwrap_or(false);

    let normalize = |name: &str, s: &[isize], default: &[isize]| -> PyResult<(i64, i64)> {
        let v = if s.is_empty() { default } else { s };
        if v.len() == 1 {
            Ok((v[0] as i64, v[0] as i64))
        } else if v.len() == 2 {
            Ok((v[0] as i64, v[1] as i64))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "max_pool2d: {} must either be a single int, or a tuple of two ints",
                name
            )))
        }
    };

    let (kh, kw) = normalize("kernel_size", &kernel, &[])?;
    let (sh, sw) = normalize("stride", &stride_raw, &kernel)?;
    let (ph, pw) = normalize("padding", &padding, &[])?;
    let (dh, dw) = normalize("dilation", &dilation, &[])?;

    if kh <= 0 || kw <= 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "max_pool2d: kernel_size must be greater than zero",
        ));
    }
    if sh <= 0 || sw <= 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "max_pool2d: stride must be greater than zero",
        ));
    }
    if dh <= 0 || dw <= 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "max_pool2d: dilation must be greater than zero",
        ));
    }

    let t = input.tensor()?;
    let tag = input.tag();
    let dims = t.dims();
    if dims.len() != 3 && dims.len() != 4 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "max_pool2d: non-empty 3D or 4D (batch mode) tensor expected for input",
        ));
    }

    let split = dims.len() - 2;
    let planes: usize = dims[..split].iter().product();
    let ih = dims[split] as i64;
    let iw = dims[split + 1] as i64;

    let extent = |input_size: i64, k: i64, pad: i64, stride: i64, dilation: i64| -> i64 {
        let numerator = input_size + 2 * pad - dilation * (k - 1) - 1;
        let mut out = if ceil_mode {
            numerator.div_euclid(stride) + if numerator.rem_euclid(stride) != 0 { 1 } else { 0 }
        } else {
            numerator.div_euclid(stride)
        } + 1;
        if ceil_mode && (out - 1) * stride >= input_size + pad {
            out -= 1;
        }
        out
    };
    let oh = extent(ih, kh, ph, sh, dh);
    let ow = extent(iw, kw, pw, sw, dw);
    if oh <= 0 || ow <= 0 {
        let c = if split == 1 { dims[0] } else { dims[1] };
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "max_pool2d: Given input size: ({}x{}x{}). Calculated output size: ({}x{}x{}). \
             Output size is too small",
            c, ih, iw, c, oh, ow
        )));
    }

    let out_dims = if split == 1 {
        vec![dims[0], oh as usize, ow as usize]
    } else {
        vec![dims[0], dims[1], oh as usize, ow as usize]
    };
    
    let source = read_flat(OP, t, tag)?;
    
    let acc32 = match tag {
        TorchDType::Float32 => true,
        TorchDType::Float64 => false,
        _ => return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "\"max_pool2d\" not implemented for '{}'",
            scalar_type_name(tag)
        ))),
    };

    let mut out_f = vec![0.0f64; planes * (oh * ow) as usize];
    for p in 0..planes {
        let base = p * (ih * iw) as usize;
        for y in 0..oh {
            let h_start_raw = y * sh - ph;
            let h_end_raw = (h_start_raw + dh * kh).min(ih + ph);
            let h_start = h_start_raw.max(0);
            let h_end = h_end_raw.min(ih);
            for x in 0..ow {
                let w_start_raw = x * sw - pw;
                let w_end_raw = (w_start_raw + dw * kw).min(iw + pw);
                let w_start = w_start_raw.max(0);
                let w_end = w_end_raw.min(iw);
                let at = p * (oh * ow) as usize + (y * ow + x) as usize;
                
                match &source {
                    Flat::Float(values) => {
                        if acc32 {
                            let mut max_val = f32::NEG_INFINITY;
                            for row_idx in 0..kh {
                                let row = h_start_raw + row_idx * dh;
                                if row >= 0 && row < ih {
                                    let offset = base + (row * iw) as usize;
                                    for col_idx in 0..kw {
                                        let col = w_start_raw + col_idx * dw;
                                        if col >= 0 && col < iw {
                                            let val = values[offset + col as usize] as f32;
                                            if val > max_val || max_val.is_nan() {
                                                max_val = val;
                                            }
                                        }
                                    }
                                }
                            }
                            out_f[at] = max_val as f64;
                        } else {
                            let mut max_val = f64::NEG_INFINITY;
                            for row_idx in 0..kh {
                                let row = h_start_raw + row_idx * dh;
                                if row >= 0 && row < ih {
                                    let offset = base + (row * iw) as usize;
                                    for col_idx in 0..kw {
                                        let col = w_start_raw + col_idx * dw;
                                        if col >= 0 && col < iw {
                                            let val = values[offset + col as usize];
                                            if val > max_val || max_val.is_nan() {
                                                max_val = val;
                                            }
                                        }
                                    }
                                }
                            }
                            out_f[at] = max_val;
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    let out_t = if acc32 {
        let floats: Vec<f32> = out_f.into_iter().map(|v| v as f32).collect();
        Tensor::from_vec(floats, out_dims, t.device()).map_err(|e| candle_err(OP, e))?
    } else {
        Tensor::from_vec(out_f, out_dims, t.device()).map_err(|e| candle_err(OP, e))?
    };
    
    let py_t = pyo3::Py::new(py, PyTensorBase::new(out_t)?)?;
    Ok(py_t.into_any())
}


fn nonzero_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.nonzero.default";
    let condition = tensor_arg(OP, args, kwargs, 0, "self")?;
    let t = condition.tensor()?;

    let zeros = t.zeros_like().map_err(|e| candle_err(OP, e))?;
    let mask = t.ne(&zeros).map_err(|e| candle_err(OP, e))?;
    let mask_u8 = mask.flatten_all().map_err(|e| candle_err(OP, e))?.to_vec1::<u8>().map_err(|e| candle_err(OP, e))?;

    let shape = t.dims();
    let num_dims = shape.len();

    let mut z = 0;
    for &val in mask_u8.iter() {
        if val != 0 {
            z += 1;
        }
    }

    if num_dims == 0 {
        let out_t = Tensor::from_vec(Vec::<i64>::new(), (z, 0), t.device()).map_err(|e| candle_err(OP, e))?;
        let out = pyo3::Py::new(py, PyTensorBase::new(out_t)?)?;
        return Ok(out.into_any());
    }

    let mut indices: Vec<i64> = Vec::with_capacity(z * num_dims);
    let mut strides = vec![0i64; num_dims];
    let mut current_stride = 1i64;
    for i in (0..num_dims).rev() {
        strides[i] = current_stride;
        current_stride *= shape[i] as i64;
    }

    for (flat_idx, &val) in mask_u8.iter().enumerate() {
        if val != 0 {
            let mut rem = flat_idx as i64;
            for i in 0..num_dims {
                indices.push(rem / strides[i]);
                rem %= strides[i];
            }
        }
    }

    let out_t = Tensor::from_vec(indices, (z, num_dims), t.device()).map_err(|e| candle_err(OP, e))?;
    let out = pyo3::Py::new(py, PyTensorBase::new(out_t)?)?;
    Ok(out.into_any())
}

fn where_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.where.default";
    let condition = tensor_arg(OP, args, kwargs, 0, "condition")?;
    let t = condition.tensor()?;
    
    let zeros = t.zeros_like().map_err(|e| candle_err(OP, e))?;
    let mask = t.ne(&zeros).map_err(|e| candle_err(OP, e))?;
    let mask_u8 = mask.flatten_all().map_err(|e| candle_err(OP, e))?.to_vec1::<u8>().map_err(|e| candle_err(OP, e))?;

    let shape = t.dims();
    let num_dims = shape.len();

    if num_dims == 0 {
        let mut vals = Vec::new();
        if mask_u8[0] != 0 {
            vals.push(0i64);
        }
        let out_t = Tensor::from_vec(vals.clone(), vals.len(), t.device()).map_err(|e| candle_err(OP, e))?;
        let out = pyo3::Py::new(py, PyTensorBase::new(out_t)?)?;
        let tuple = PyTuple::new(py, vec![out.into_any()])?;
        return Ok(tuple.into_any().unbind());
    }

    let mut indices: Vec<Vec<i64>> = vec![Vec::new(); num_dims];
    let mut strides = vec![0i64; num_dims];
    let mut current_stride = 1i64;
    for i in (0..num_dims).rev() {
        strides[i] = current_stride;
        current_stride *= shape[i] as i64;
    }

    for (flat_idx, &val) in mask_u8.iter().enumerate() {
        if val != 0 {
            let mut rem = flat_idx as i64;
            for i in 0..num_dims {
                indices[i].push(rem / strides[i]);
                rem %= strides[i];
            }
        }
    }

    let mut out_tensors = Vec::with_capacity(num_dims);
    let count = indices[0].len();
    for i in 0..num_dims {
        let out_t = Tensor::from_vec(indices[i].clone(), count, t.device()).map_err(|e| candle_err(OP, e))?;
        let out = pyo3::Py::new(py, PyTensorBase::new(out_t)?)?;
        out_tensors.push(out.into_any());
    }

    let tuple = PyTuple::new(py, out_tensors)?;
    Ok(tuple.into_any().unbind())
}

fn hardtanh_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.hardtanh.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let min = scalar_arg(OP, args, kwargs, 1, "min_val")?;
    let max = scalar_arg(OP, args, kwargs, 2, "max_val")?;
    
    let min_val = min.unwrap_or(Scalar::Float(-1.0));
    let max_val = max.unwrap_or(Scalar::Float(1.0));

    let tag = clamp_result_tag(OP, args, kwargs, input.tag(), Some(min_val), Some(max_val))?;

    let storage = PyDtype::new(tag).storage(OP)?;
    let source = input
        .tensor()?
        .fast_to(storage)
        .map_err(|e| candle_err(OP, e))?;
    let out = clamp_values(OP, &source, tag, Some(min_val), Some(max_val))?;
    finish(py, out, tag)
}

fn one_hot_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.one_hot.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let num_classes_opt = int_arg(args, kwargs, 1, "num_classes")?;
    let num_classes = num_classes_opt.unwrap_or(-1);
    
    let t = input.tensor()?;
    if input.tag() != TorchDType::Int64 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "one_hot is only applicable to index tensor of type LongTensor.",
        ));
    }
    
    let mut actual_num_classes = num_classes as i64;
    let flat = t.flatten_all().map_err(|e| candle_err(OP, e))?;
    let data = flat.to_vec1::<i64>().map_err(|e| candle_err(OP, e))?;
    
    if actual_num_classes == -1 {
        if data.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Can not infer total number of classes from empty tensor.",
            ));
        }
        let max_val = *data.iter().max().unwrap();
        actual_num_classes = max_val + 1;
    }
    
    for &val in &data {
        if val < 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Class values must be non-negative.",
            ));
        }
        if val >= actual_num_classes {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Class values must be smaller than num_classes.",
            ));
        }
    }
    
    let mut out_data = vec![0i64; data.len() * actual_num_classes as usize];
    for (i, &val) in data.iter().enumerate() {
        out_data[i * actual_num_classes as usize + val as usize] = 1;
    }
    
    let mut out_dims = t.dims().to_vec();
    out_dims.push(actual_num_classes as usize);
    
    let out_t = Tensor::from_vec(out_data, out_dims, t.device()).map_err(|e| candle_err(OP, e))?;
    let py_t = pyo3::Py::new(py, PyTensorBase::new(out_t)?)?;
    Ok(py_t.into_any())
}

fn adaptive_avg_pool1d_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.adaptive_avg_pool1d.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let output_size = shape_arg(OP, args, kwargs, 1, "output_size")?;
    
    let t = input.tensor()?;
    let tag = input.tag();
    let dims = t.dims();
    
    if dims.len() != 2 && dims.len() != 3 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "adaptive_avg_pool1d: non-empty 2D or 3D (batch mode) tensor expected for input",
        ));
    }
    
    if output_size.len() != 1 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "adaptive_avg_pool1d: output_size must be a single int or a tuple of one int",
        ));
    }
    
    let osize = output_size[0] as i64;
    let isize = dims[dims.len() - 1] as i64;
    
    let split = dims.len() - 1;
    let planes: usize = dims[..split].iter().product();
    
    let mut out_dims = dims.to_vec();
    out_dims[split] = osize as usize;
    
    // **An empty output runs no kernel, for any dtype.** Measured on 2.13.0
    // over float32/float64/float16/bfloat16/float8_e4m3fn/int64/bool:
    // `output_size=[0]` returns a `(..., 0)` tensor of the input's own dtype in
    // every case, including the dtypes whose non-empty call raises
    // `"adaptive_avg_pool2d" not implemented for '<Type>'`.
    //
    // It is also the row that made docs/FLOAT8B.md table D claim upstream
    // *computes* `adaptive_avg_pool1d` for `float8_e4m3fn`: the generic recipe
    // had synthesised exactly this output size (docs/FLOAT8C.md §2). Returning
    // here, before the dtype gate below, is what makes that claim true for the
    // one input it was ever true for.
    if osize == 0 {
        let storage = PyDtype::new(tag).storage(OP)?;
        let empty = Tensor::zeros(out_dims, storage, t.device()).map_err(|e| candle_err(OP, e))?;
        return finish(py, empty, tag);
    }

    let source = read_flat(OP, t, tag)?;
    
    let acc32 = match tag {
        TorchDType::Float32 => true,
        TorchDType::Float64 => false,
        _ => return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "\"adaptive_avg_pool1d\" not implemented for '{}'",
            scalar_type_name(tag)
        ))),
    };

    let mut out_f = vec![0.0f64; planes * osize as usize];
    
    match &source {
        Flat::Float(values) => {
            if acc32 {
                for p in 0..planes {
                    let base = p * isize as usize;
                    let out_base = p * osize as usize;
                    for i in 0..osize {
                        let start = (i * isize) / osize;
                        let end = ((i + 1) * isize + osize - 1) / osize;
                        let mut sum = 0.0f32;
                        for j in start..end {
                            sum += values[base + j as usize] as f32;
                        }
                        let count = end - start;
                        out_f[out_base + i as usize] = (sum / count as f32) as f64;
                    }
                }
            } else {
                for p in 0..planes {
                    let base = p * isize as usize;
                    let out_base = p * osize as usize;
                    for i in 0..osize {
                        let start = (i * isize) / osize;
                        let end = ((i + 1) * isize + osize - 1) / osize;
                        let mut sum = 0.0f64;
                        for j in start..end {
                            sum += values[base + j as usize];
                        }
                        let count = end - start;
                        out_f[out_base + i as usize] = sum / count as f64;
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    
    let out_t = if acc32 {
        let floats: Vec<f32> = out_f.into_iter().map(|v| v as f32).collect();
        Tensor::from_vec(floats, out_dims, t.device()).map_err(|e| candle_err(OP, e))?
    } else {
        Tensor::from_vec(out_f, out_dims, t.device()).map_err(|e| candle_err(OP, e))?
    };
    
    let py_t = pyo3::Py::new(py, PyTensorBase::new(out_t)?)?;
    Ok(py_t.into_any())
}

fn where_scalar_self(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.where.ScalarSelf";
    let condition = tensor_arg(OP, args, kwargs, 0, "condition")?;
    let raw = required(OP, args, kwargs, 1, "self")?;
    let rhs = tensor_arg(OP, args, kwargs, 2, "other")?;

    where_condition_check(&condition)?;
    if raw.is_instance_of::<PyTensorBase>() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "aten::where() Expected a value of type 'number' for argument 'self'              but instead found type Tensor",
        ));
    }

    let scalar_is_bool = raw.is_instance_of::<pyo3::types::PyBool>();
    let scalar_is_int = scalar_is_bool || raw.is_instance_of::<pyo3::types::PyInt>();
    let value = scalar_arg(OP, args, kwargs, 1, "self")?.ok_or_else(|| missing(OP, "self"))?;

    let tag = where_scalar_tag(rhs.tag(), scalar_is_bool, scalar_is_int);
    checked_convert(&raw, scalar_is_int, tag, 1)?;

    let device = rhs.tensor()?.device().clone();
    let lhs = if tag == TorchDType::Bool {
        Tensor::full(u8::from(value.as_f64() != 0.0), (), &device).map_err(|e| candle_err(OP, e))?
    } else {
        let storage = PyDtype::new(tag).storage(OP)?;
        if storage.is_int() {
            Tensor::full(value.as_i64(), (), &device)
        } else {
            Tensor::full(value.as_f64(), (), &device)
        }
        .and_then(|t| t.fast_to(storage))
        .map_err(|e| candle_err(OP, e))?
    };

    let out = where_select(OP, &condition, &lhs, rhs.tensor()?, tag)?;
    finish(py, out, tag)
}

fn roll_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.roll.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let shifts_raw = shape_arg(OP, args, kwargs, 1, "shifts")?;

    let dims_raw = if let Some(v) = optional(args, kwargs, 2, "dims")? {
        if v.is_none() {
            vec![]
        } else {
            shape_arg(OP, args, kwargs, 2, "dims")?
        }
    } else {
        vec![]
    };

    if !dims_raw.is_empty() && shifts_raw.len() != dims_raw.len() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            format!("shifts and dimensions must align. shifts: {}, dims:{}", shifts_raw.len(), dims_raw.len()),
        ));
    }

    let t = input.tensor()?;
    let tag = input.tag();
    let mut current = t.clone();

    if dims_raw.is_empty() {
        if shifts_raw.len() != 1 { 
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("shifts and dimensions must align. shifts: {}, dims:0", shifts_raw.len())
            )); 
        }
        let total_shift = shifts_raw[0];
        let shape = current.dims().to_vec();
        current = current.flatten_all().map_err(|e| candle_err(OP, e))?;
        let len = current.dims()[0] as isize;
        if len > 0 {
            let mut s = total_shift % len;
            if s < 0 {
                s += len;
            }
            if s > 0 {
                let split_idx = (len - s) as usize;
                let front = current.narrow(0, split_idx, s as usize).map_err(|e| candle_err(OP, e))?;
                let back = current.narrow(0, 0, split_idx).map_err(|e| candle_err(OP, e))?;
                current = candle_core::Tensor::cat(&[&front, &back], 0).map_err(|e| candle_err(OP, e))?;
            }
        }
        current = current.reshape(shape).map_err(|e| candle_err(OP, e))?;
    } else {
        let rank = current.rank();
        for (shift, dim_raw) in shifts_raw.into_iter().zip(dims_raw.into_iter()) {
            let dim = normalise_dim(OP, dim_raw, rank)?;
            let len = current.dims()[dim] as isize;
            if len > 0 {
                let mut s = shift % len;
                if s < 0 {
                    s += len;
                }
                if s > 0 {
                    let split_idx = (len - s) as usize;
                    let front = current.narrow(dim, split_idx, s as usize).map_err(|e| candle_err(OP, e))?;
                    let back = current.narrow(dim, 0, split_idx).map_err(|e| candle_err(OP, e))?;
                    current = candle_core::Tensor::cat(&[&front, &back], dim).map_err(|e| candle_err(OP, e))?;
                }
            }
        }
    }

    finish(py, current, tag)
}

fn adaptive_avg_pool2d_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.adaptive_avg_pool2d.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let output_size = shape_arg(OP, args, kwargs, 1, "output_size")?;

    let t = input.tensor()?;
    let tag = input.tag();
    let dims = t.dims();

    if dims.len() != 3 && dims.len() != 4 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "adaptive_avg_pool2d: expected 3D or 4D input",
        ));
    }

    if output_size.len() != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "adaptive_avg_pool2d: output_size must be 2",
        ));
    }

    let osize_h = output_size[0] as i64;
    let osize_w = output_size[1] as i64;
    let isize_h = dims[dims.len() - 2] as i64;
    let isize_w = dims[dims.len() - 1] as i64;

    let split = dims.len() - 2;
    let planes: usize = dims[..split].iter().product();

    let mut out_dims = dims.to_vec();
    out_dims[split] = osize_h as usize;
    out_dims[split + 1] = osize_w as usize;

    let source = read_flat(OP, t, tag)?;

    let acc32 = match tag {
        TorchDType::Float32 => true,
        TorchDType::Float64 => false,
        _ => return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "\"adaptive_avg_pool2d\" not implemented for '{}'",
            scalar_type_name(tag)
        ))),
    };

    let mut out_f = vec![0.0f64; planes * (osize_h * osize_w) as usize];

    match &source {
        Flat::Float(values) => {
            if acc32 {
                for p in 0..planes {
                    let base = p * (isize_h * isize_w) as usize;
                    let out_base = p * (osize_h * osize_w) as usize;
                    for ih in 0..osize_h {
                        let start_h = (ih * isize_h) / osize_h;
                        let end_h = ((ih + 1) * isize_h + osize_h - 1) / osize_h;
                        for iw in 0..osize_w {
                            let start_w = (iw * isize_w) / osize_w;
                            let end_w = ((iw + 1) * isize_w + osize_w - 1) / osize_w;

                            let mut sum = 0.0f32;
                            for h in start_h..end_h {
                                for w in start_w..end_w {
                                    sum += values[base + (h * isize_w + w) as usize] as f32;
                                }
                            }
                            let count = (end_h - start_h) * (end_w - start_w);
                            out_f[out_base + (ih * osize_w + iw) as usize] = (sum / count as f32) as f64;
                        }
                    }
                }
            } else {
                for p in 0..planes {
                    let base = p * (isize_h * isize_w) as usize;
                    let out_base = p * (osize_h * osize_w) as usize;
                    for ih in 0..osize_h {
                        let start_h = (ih * isize_h) / osize_h;
                        let end_h = ((ih + 1) * isize_h + osize_h - 1) / osize_h;
                        for iw in 0..osize_w {
                            let start_w = (iw * isize_w) / osize_w;
                            let end_w = ((iw + 1) * isize_w + osize_w - 1) / osize_w;

                            let mut sum = 0.0f64;
                            for h in start_h..end_h {
                                for w in start_w..end_w {
                                    sum += values[base + (h * isize_w + w) as usize];
                                }
                            }
                            let count = (end_h - start_h) * (end_w - start_w);
                            out_f[out_base + (ih * osize_w + iw) as usize] = sum / count as f64;
                        }
                    }
                }
            }
        }
        _ => unreachable!(),
    }

    let out_t = if acc32 {
        let f32_vals: Vec<f32> = out_f.into_iter().map(|v| v as f32).collect();
        Tensor::from_vec(f32_vals, out_dims.as_slice(), t.device())
    } else {
        Tensor::from_vec(out_f, out_dims.as_slice(), t.device())
    }
    .map_err(|e| candle_err(OP, e))?;

    finish(py, out_t, tag)
}
