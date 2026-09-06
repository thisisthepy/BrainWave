//! Reverse mode over a captured trace.
//!
//! `docs/AUTOGRAD.md` §6 chose this shape over a `VariableType` equivalent and
//! gave the reason: upstream's dispatcher has many doors, so recording
//! gradients there needs a generated wrapper per op; this shim has one door and
//! the recorder is already one line at the end of it. §6.2 then showed the
//! input already exists -- a `.train()` decoder layer captures to 57 nodes and
//! replays at max abs diff 0.0. What was missing was the walk.
//!
//! This is that walk, and it is deliberately small:
//!
//! | | |
//! |---|---|
//! | replay | the forward, keeping every intermediate (`PyCaptureTrace::run`) |
//! | seed | the declared outputs' gradients |
//! | walk | the node list **backwards**, once, accumulating into a map keyed on the trace's own `Ref` |
//!
//! Three properties come from capture rather than from anything here, and they
//! are the ones a reverse mode is usually built to establish:
//!
//! * **Single assignment.** Capture refuses in-place ops (docs/CAPTURE.md §4),
//!   so no recorded value is ever overwritten between its use and this walk.
//!   Upstream spends `ADInplaceOrView` and a version counter per tensor on
//!   exactly this.
//! * **Straight line.** Capture refuses host reads, so there is no branch
//!   inside the region and the reverse order is just the forward order
//!   reversed -- no ready queue, no dependency counts, no `GraphTask`.
//! * **A closed op set.** The trace names every op it used, so
//!   `differentiable()` can say what would stop a backward *before* one is run.
//!
//! **The rules are compositions, not kernels.** A derivative here is spelled in
//! ops the shim already has and dispatched through the same door. That is what
//! makes the bill in `docs/AUTOGRAD.md` §4 -- "25 need their own kernel" --
//! not the bill for this file: the backward runs *outside* a capture region, so
//! it may use ops capture would refuse to record, and it may recompute rather
//! than read a saved value. `_scaled_dot_product_flash_attention_for_cpu` is
//! the case where that matters most and `docs/BACKWARD.md` §5 measures it.

use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::capture::{Arg, Env, Node, PyCaptureTrace, Ref, Slot};
use crate::tensor::PyTensorBase;

type Obj<'py> = Bound<'py, PyAny>;

// ---------------------------------------------------------------------------
// Talking to the door
// ---------------------------------------------------------------------------

/// Every op this file issues goes through `aten_dispatch`, the same entrance
/// the forward used. A derivative computed some other way would not be
/// comparable to the forward it differentiates.
fn call<'py>(py: Python<'py>, op: &str, args: Vec<Obj<'py>>) -> PyResult<Obj<'py>> {
    let tuple = PyTuple::new(py, args)?;
    Ok(crate::aten::aten_dispatch(py, op, &tuple, None)?.into_bound(py))
}

fn call_kw<'py>(
    py: Python<'py>,
    op: &str,
    args: Vec<Obj<'py>>,
    kw: Vec<(&str, Obj<'py>)>,
) -> PyResult<Obj<'py>> {
    let tuple = PyTuple::new(py, args)?;
    let dict = PyDict::new(py);
    for (name, value) in kw {
        dict.set_item(name, value)?;
    }
    Ok(crate::aten::aten_dispatch(py, op, &tuple, Some(&dict))?.into_bound(py))
}

fn dims(value: &Obj<'_>) -> PyResult<Vec<usize>> {
    let tensor = value.cast::<PyTensorBase>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "torch._C tape: a value the tape needed the shape of is not a tensor",
        )
    })?;
    let shape = tensor.borrow().dims().to_vec();
    Ok(shape)
}

fn scalar<'py>(py: Python<'py>, value: f64) -> PyResult<Obj<'py>> {
    value.into_bound_py_any(py)
}

fn ints<'py>(py: Python<'py>, values: &[i64]) -> PyResult<Obj<'py>> {
    Ok(PyList::new(py, values)?.into_any())
}

fn shape_ints(shape: &[usize]) -> Vec<i64> {
    shape.iter().map(|d| *d as i64).collect()
}

fn add<'py>(py: Python<'py>, a: Obj<'py>, b: Obj<'py>) -> PyResult<Obj<'py>> {
    call(py, "aten.add.Tensor", vec![a, b])
}

fn mul<'py>(py: Python<'py>, a: Obj<'py>, b: Obj<'py>) -> PyResult<Obj<'py>> {
    call(py, "aten.mul.Tensor", vec![a, b])
}

fn muls<'py>(py: Python<'py>, a: Obj<'py>, s: f64) -> PyResult<Obj<'py>> {
    let s = scalar(py, s)?;
    call(py, "aten.mul.Scalar", vec![a, s])
}

fn neg<'py>(py: Python<'py>, a: Obj<'py>) -> PyResult<Obj<'py>> {
    call(py, "aten.neg.default", vec![a])
}

fn sub<'py>(py: Python<'py>, a: Obj<'py>, b: Obj<'py>) -> PyResult<Obj<'py>> {
    call(py, "aten.sub.Tensor", vec![a, b])
}

fn div<'py>(py: Python<'py>, a: Obj<'py>, b: Obj<'py>) -> PyResult<Obj<'py>> {
    call(py, "aten.div.Tensor", vec![a, b])
}

fn reshape<'py>(py: Python<'py>, a: Obj<'py>, shape: &[usize]) -> PyResult<Obj<'py>> {
    if dims(&a)? == shape {
        return Ok(a);
    }
    let want = ints(py, &shape_ints(shape))?;
    call(py, "aten.reshape.default", vec![a, want])
}

fn sum_dims<'py>(
    py: Python<'py>,
    a: Obj<'py>,
    dim: &[i64],
    keepdim: bool,
) -> PyResult<Obj<'py>> {
    let d = ints(py, dim)?;
    let k = keepdim.into_bound_py_any(py)?;
    call(py, "aten.sum.dim_IntList", vec![a, d, k])
}

fn transpose<'py>(py: Python<'py>, a: Obj<'py>, d0: i64, d1: i64) -> PyResult<Obj<'py>> {
    let x = d0.into_bound_py_any(py)?;
    let y = d1.into_bound_py_any(py)?;
    call(py, "aten.transpose.int", vec![a, x, y])
}

/// The last two axes swapped -- what every matmul derivative needs and what
/// `t()` cannot do above rank 2.
fn mt<'py>(py: Python<'py>, a: Obj<'py>) -> PyResult<Obj<'py>> {
    transpose(py, a, -2, -1)
}

fn matmul<'py>(py: Python<'py>, a: Obj<'py>, b: Obj<'py>) -> PyResult<Obj<'py>> {
    call(py, "aten.matmul.default", vec![a, b])
}

fn zeros_like<'py>(py: Python<'py>, a: &Obj<'py>) -> PyResult<Obj<'py>> {
    call(py, "aten.zeros_like.default", vec![a.clone()])
}

/// Undo broadcasting: bring `g` back to `want` by summing the axes broadcasting
/// expanded.
///
/// This is the one piece of arithmetic every elementwise rule needs and the
/// easiest to leave out, because it is invisible whenever the two operands
/// already have the same shape -- which is every case anyone writes first. An
/// RMSNorm is where it shows: `weight * hidden` has a `[576]` on one side and a
/// `[1, 8, 576]` on the other, and a rule without this returns the *right shape
/// for the wrong reason* only because a later reshape would have hidden it.
fn reduce_to<'py>(py: Python<'py>, g: Obj<'py>, want: &[usize]) -> PyResult<Obj<'py>> {
    let have = dims(&g)?;
    if have == want {
        return Ok(g);
    }
    let mut g = g;
    if have.len() > want.len() {
        let leading: Vec<i64> = (0..(have.len() - want.len()) as i64).collect();
        g = sum_dims(py, g, &leading, false)?;
    }
    let have = dims(&g)?;
    let squashed: Vec<i64> = have
        .iter()
        .zip(want.iter())
        .enumerate()
        .filter(|(_, (h, w))| **w == 1 && **h != 1)
        .map(|(i, _)| i as i64)
        .collect();
    if !squashed.is_empty() {
        g = sum_dims(py, g, &squashed, true)?;
    }
    reshape(py, g, want)
}

/// `g` broadcast up to `want`, which is `reduce_to` run the other way.
fn expand_to<'py>(py: Python<'py>, g: Obj<'py>, want: &[usize]) -> PyResult<Obj<'py>> {
    if dims(&g)? == want {
        return Ok(g);
    }
    let size = ints(py, &shape_ints(want))?;
    call(py, "aten.expand.default", vec![g, size])
}

fn normalise_dim(dim: i64, rank: usize) -> usize {
    if dim < 0 {
        (dim + rank as i64) as usize
    } else {
        dim as usize
    }
}

// ---------------------------------------------------------------------------
// Reading a node's operands
// ---------------------------------------------------------------------------

/// One argument of a node as the tape needs it: the live value, and -- when the
/// argument was a traced value rather than a literal -- where a gradient for it
/// has to be accumulated.
struct Operand<'py> {
    target: Option<Ref>,
    value: Obj<'py>,
}

impl<'py> Operand<'py> {
    fn shape(&self) -> PyResult<Vec<usize>> {
        dims(&self.value)
    }
}

/// Bind a node's positional and keyword arguments to a schema's parameter
/// names.
///
/// Both spellings have to be handled and neither is hypothetical: the vendored
/// tree's `_torch_level_function` binds every argument by name before
/// dispatching, so a trace holds `aten.sum.default() {'self': %1}` for one op
/// and `aten.addmm.default(%c1, %in0, %0)` for the next (docs/CAPTURE.md §2).
/// A rule that read `args[0]` would be right half the time.
fn bind<'py>(
    py: Python<'py>,
    node: &Node,
    env: &Env,
    names: &[&str],
) -> PyResult<Vec<Option<Operand<'py>>>> {
    let mut out = Vec::with_capacity(names.len());
    for (index, name) in names.iter().enumerate() {
        let arg = if index < node.args.len() {
            Some(&node.args[index])
        } else {
            node.kwargs
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, arg)| arg)
        };
        out.push(match arg {
            None => None,
            Some(Arg::Value(reference)) => Some(Operand {
                target: Some(*reference),
                value: env.get(py, *reference)?,
            }),
            Some(other) => Some(Operand { target: None, value: other.materialise(py, env)? }),
        });
    }
    Ok(out)
}

/// The `tensors` argument of `cat`/`stack`: a list of traced values rather than
/// one.
fn bind_list<'py>(
    py: Python<'py>,
    node: &Node,
    env: &Env,
    index: usize,
    name: &str,
) -> PyResult<Vec<Operand<'py>>> {
    let arg = if index < node.args.len() {
        Some(&node.args[index])
    } else {
        node.kwargs.iter().find(|(key, _)| key == name).map(|(_, a)| a)
    };
    let items = match arg {
        Some(Arg::List(items)) | Some(Arg::Tuple(items)) => items,
        _ => {
            return Err(missing(&node.op, name));
        }
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.push(match item {
            Arg::Value(reference) => {
                Operand { target: Some(*reference), value: env.get(py, *reference)? }
            }
            other => Operand { target: None, value: other.materialise(py, env)? },
        });
    }
    Ok(out)
}

fn missing(op: &str, name: &str) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "torch._C tape: {op} was recorded without its {name} argument, so its derivative \
         cannot be formed"
    ))
}

fn required<'py, 'a>(
    op: &str,
    operands: &'a [Option<Operand<'py>>],
    index: usize,
    name: &str,
) -> PyResult<&'a Operand<'py>> {
    operands
        .get(index)
        .and_then(|slot| slot.as_ref())
        .ok_or_else(|| missing(op, name))
}

fn opt_i64(operands: &[Option<Operand<'_>>], index: usize, fallback: i64) -> PyResult<i64> {
    match operands.get(index).and_then(|s| s.as_ref()) {
        None => Ok(fallback),
        Some(operand) if operand.value.is_none() => Ok(fallback),
        Some(operand) => operand.value.extract::<i64>(),
    }
}

fn opt_f64(operands: &[Option<Operand<'_>>], index: usize, fallback: f64) -> PyResult<f64> {
    match operands.get(index).and_then(|s| s.as_ref()) {
        None => Ok(fallback),
        Some(operand) if operand.value.is_none() => Ok(fallback),
        Some(operand) => operand.value.extract::<f64>(),
    }
}

fn opt_bool(operands: &[Option<Operand<'_>>], index: usize, fallback: bool) -> PyResult<bool> {
    match operands.get(index).and_then(|s| s.as_ref()) {
        None => Ok(fallback),
        Some(operand) if operand.value.is_none() => Ok(fallback),
        Some(operand) => operand.value.extract::<bool>(),
    }
}

fn i64_list(operands: &[Option<Operand<'_>>], index: usize, name: &str, op: &str) -> PyResult<Vec<i64>> {
    let operand = operands
        .get(index)
        .and_then(|s| s.as_ref())
        .ok_or_else(|| missing(op, name))?;
    if let Ok(one) = operand.value.extract::<i64>() {
        return Ok(vec![one]);
    }
    operand.value.extract::<Vec<i64>>()
}

// ---------------------------------------------------------------------------
// The rule table
// ---------------------------------------------------------------------------

/// Every op the tape can differentiate.
///
/// This list is what `differentiable()` reports against, and it is checked
/// against the *implementation* by a test rather than by reading: every name
/// here has a gradient case in `pytests/test_shim.py` compared against
/// upstream, and that test asserts its own case list equals this one. A name
/// added here without a case makes that test fail, which is the only way a
/// second list of op names stays honest in this repository (docs/AUDIT.md).
pub const RULE_OPS: &[&str] = &[
    "aten._log_softmax.default",
    "aten._safe_softmax.default",
    "aten._scaled_dot_product_flash_attention_for_cpu.default",
    "aten._softmax.default",
    "aten._to_copy.default",
    "aten._unsafe_view.default",
    "aten.add.Scalar",
    "aten.add.Tensor",
    "aten.addmm.default",
    "aten.alias.default",
    "aten.bmm.default",
    "aten.cat.default",
    "aten.clone.default",
    "aten.constant_pad_nd.default",
    "aten.contiguous.default",
    "aten.cos.default",
    "aten.detach.default",
    "aten.div.Scalar",
    "aten.div.Tensor",
    "aten.embedding.default",
    "aten.exp.default",
    "aten.expand.default",
    "aten.gelu.default",
    "aten.lift_fresh.default",
    "aten.log.default",
    "aten.masked_fill.Scalar",
    "aten.matmul.default",
    "aten.mean.default",
    "aten.mean.dim",
    "aten.mm.default",
    "aten.mul.Scalar",
    "aten.mul.Tensor",
    "aten.native_dropout.default",
    "aten.native_batch_norm.default",
    "aten.native_layer_norm.default",
    "aten.neg.default",
    "aten.nll_loss_forward.default",
    "aten.permute.default",
    "aten.pow.Tensor_Scalar",
    "aten.relu.default",
    "aten.reshape.default",
    "aten.rsqrt.default",
    "aten.rsub.Scalar",
    "aten.select.int",
    "aten.sigmoid.default",
    "aten.silu.default",
    "aten.sin.default",
    "aten.slice.Tensor",
    "aten.split.Tensor",
    "aten.sqrt.default",
    "aten.squeeze.dim",
    "aten.sub.Scalar",
    "aten.sub.Tensor",
    "aten.sum.default",
    "aten.sum.dim_IntList",
    "aten.t.default",
    "aten.tanh.default",
    "aten.transpose.int",
    "aten.unsqueeze.default",
    "aten.view.default",
    "aten.where.self",
];

pub fn has_rule(op: &str) -> bool {
    RULE_OPS.contains(&op)
}

fn no_rule(op: &str) -> PyErr {
    crate::err::not_implemented(format!(
        "torch._C tape: no derivative rule for {op} -- a gradient reached it, and the tape \
         refuses to guess. Add a rule in tape.rs and a gradient case in \
         pytests/test_shim.py; trace.differentiable() lists every op in a trace that would \
         need one"
    ))
}

/// The gradient of one node with respect to each of its operands.
///
/// Returns the bound operands beside the gradients so that the caller, and not
/// each of the fifty-odd arms below, decides where a gradient goes.
type Rule<'py> = (Vec<Option<Operand<'py>>>, Vec<Option<Obj<'py>>>);

fn derivative<'py>(
    py: Python<'py>,
    node: &Node,
    env: &Env,
    gouts: &[Option<Obj<'py>>],
    outs: &[Obj<'py>],
) -> PyResult<Rule<'py>> {
    let op = node.op.as_str();
    // Every op below has exactly one differentiable result, except the two
    // that are checked for it explicitly.
    let g = || -> PyResult<Obj<'py>> {
        gouts
            .first()
            .and_then(|slot| slot.clone())
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "torch._C tape: {op} was reached with no gradient on its first result"
                ))
            })
    };

    match op {
        // -------------------------------------------------------------- shape
        "aten.t.default" => {
            let ops = bind(py, node, env, &["self"])?;
            let gi = call(py, "aten.t.default", vec![g()?])?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.transpose.int" => {
            let ops = bind(py, node, env, &["self", "dim0", "dim1"])?;
            let d0 = opt_i64(&ops, 1, 0)?;
            let d1 = opt_i64(&ops, 2, 1)?;
            let gi = transpose(py, g()?, d0, d1)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.permute.default" => {
            let ops = bind(py, node, env, &["self", "dims"])?;
            let perm = i64_list(&ops, 1, "dims", op)?;
            let rank = perm.len();
            let mut inverse = vec![0i64; rank];
            for (position, axis) in perm.iter().enumerate() {
                inverse[normalise_dim(*axis, rank)] = position as i64;
            }
            let dims_arg = ints(py, &inverse)?;
            let gi = call(py, "aten.permute.default", vec![g()?, dims_arg])?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.view.default" | "aten.reshape.default" | "aten._unsafe_view.default"
        | "aten.unsqueeze.default" | "aten.squeeze.dim" => {
            let ops = bind(py, node, env, &["self", "size"])?;
            let want = required(op, &ops, 0, "self")?.shape()?;
            let gi = reshape(py, g()?, &want)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.expand.default" => {
            let ops = bind(py, node, env, &["self", "size", "implicit"])?;
            let want = required(op, &ops, 0, "self")?.shape()?;
            let gi = reduce_to(py, g()?, &want)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.contiguous.default" | "aten.clone.default" => {
            let ops = bind(py, node, env, &["self", "memory_format"])?;
            Ok((ops, vec![Some(g()?)]))
        }
        "aten.alias.default" | "aten.lift_fresh.default" => {
            let ops = bind(py, node, env, &["self"])?;
            Ok((ops, vec![Some(g()?)]))
        }
        // `detach` exists to leave the graph, so its rule is that no gradient
        // leaves it. Returning `None` rather than refusing is the whole point
        // of the op.
        "aten.detach.default" => {
            let ops = bind(py, node, env, &["self"])?;
            Ok((ops, vec![None]))
        }
        "aten._to_copy.default" => {
            let ops = bind(py, node, env, &["self"])?;
            let source = required(op, &ops, 0, "self")?;
            let dtype = source.value.getattr("dtype")?;
            let gi = call_kw(py, "aten._to_copy.default", vec![g()?], vec![("dtype", dtype)])?;
            Ok((ops, vec![Some(gi)]))
        }

        // -------------------------------------------------------- elementwise
        "aten.add.Tensor" => {
            let ops = bind(py, node, env, &["self", "other", "alpha"])?;
            let alpha = opt_f64(&ops, 2, 1.0)?;
            let gs = reduce_to(py, g()?, &required(op, &ops, 0, "self")?.shape()?)?;
            let go = match ops.get(1).and_then(|s| s.as_ref()) {
                Some(other) if other.target.is_some() => {
                    let scaled = if alpha == 1.0 { g()? } else { muls(py, g()?, alpha)? };
                    Some(reduce_to(py, scaled, &other.shape()?)?)
                }
                _ => None,
            };
            Ok((ops, vec![Some(gs), go]))
        }
        "aten.sub.Tensor" => {
            let ops = bind(py, node, env, &["self", "other", "alpha"])?;
            let alpha = opt_f64(&ops, 2, 1.0)?;
            let gs = reduce_to(py, g()?, &required(op, &ops, 0, "self")?.shape()?)?;
            let go = match ops.get(1).and_then(|s| s.as_ref()) {
                Some(other) if other.target.is_some() => {
                    let scaled = muls(py, g()?, -alpha)?;
                    Some(reduce_to(py, scaled, &other.shape()?)?)
                }
                _ => None,
            };
            Ok((ops, vec![Some(gs), go]))
        }
        "aten.add.Scalar" | "aten.sub.Scalar" => {
            let ops = bind(py, node, env, &["self", "other", "alpha"])?;
            Ok((ops, vec![Some(g()?)]))
        }
        "aten.rsub.Scalar" => {
            let ops = bind(py, node, env, &["self", "other", "alpha"])?;
            let alpha = opt_f64(&ops, 2, 1.0)?;
            let gi = muls(py, g()?, -alpha)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.neg.default" => {
            let ops = bind(py, node, env, &["self"])?;
            let gi = neg(py, g()?)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.mul.Tensor" => {
            let ops = bind(py, node, env, &["self", "other"])?;
            let lhs = required(op, &ops, 0, "self")?;
            let rhs = required(op, &ops, 1, "other")?;
            let gs = if lhs.target.is_some() {
                Some(reduce_to(py, mul(py, g()?, rhs.value.clone())?, &lhs.shape()?)?)
            } else {
                None
            };
            let go = if rhs.target.is_some() {
                Some(reduce_to(py, mul(py, g()?, lhs.value.clone())?, &rhs.shape()?)?)
            } else {
                None
            };
            Ok((ops, vec![gs, go]))
        }
        "aten.mul.Scalar" => {
            let ops = bind(py, node, env, &["self", "other"])?;
            let factor = opt_f64(&ops, 1, 1.0)?;
            let gi = muls(py, g()?, factor)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.div.Tensor" => {
            let ops = bind(py, node, env, &["self", "other"])?;
            let lhs = required(op, &ops, 0, "self")?;
            let rhs = required(op, &ops, 1, "other")?;
            let gs = if lhs.target.is_some() {
                Some(reduce_to(py, div(py, g()?, rhs.value.clone())?, &lhs.shape()?)?)
            } else {
                None
            };
            let go = if rhs.target.is_some() {
                // -g * self / other^2, spelled as (-g * out) / other so that the
                // square is never formed -- `out` is already `self / other`.
                let term = neg(py, mul(py, g()?, outs[0].clone())?)?;
                Some(reduce_to(py, div(py, term, rhs.value.clone())?, &rhs.shape()?)?)
            } else {
                None
            };
            Ok((ops, vec![gs, go]))
        }
        "aten.div.Scalar" => {
            let ops = bind(py, node, env, &["self", "other"])?;
            let divisor = opt_f64(&ops, 1, 1.0)?;
            let gi = muls(py, g()?, 1.0 / divisor)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.pow.Tensor_Scalar" => {
            let ops = bind(py, node, env, &["self", "exponent"])?;
            let exponent = opt_f64(&ops, 1, 1.0)?;
            let base = required(op, &ops, 0, "self")?.value.clone();
            let e = scalar(py, exponent - 1.0)?;
            let lowered = call(py, "aten.pow.Tensor_Scalar", vec![base, e])?;
            let gi = muls(py, mul(py, g()?, lowered)?, exponent)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.exp.default" => {
            let ops = bind(py, node, env, &["self"])?;
            let gi = mul(py, g()?, outs[0].clone())?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.log.default" => {
            let ops = bind(py, node, env, &["self"])?;
            let x = required(op, &ops, 0, "self")?.value.clone();
            let gi = div(py, g()?, x)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.sqrt.default" => {
            let ops = bind(py, node, env, &["self"])?;
            let gi = div(py, muls(py, g()?, 0.5)?, outs[0].clone())?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.rsqrt.default" => {
            let ops = bind(py, node, env, &["self"])?;
            let cube = call(py, "aten.pow.Tensor_Scalar", vec![outs[0].clone(), scalar(py, 3.0)?])?;
            let gi = muls(py, mul(py, g()?, cube)?, -0.5)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.sin.default" => {
            let ops = bind(py, node, env, &["self"])?;
            let x = required(op, &ops, 0, "self")?.value.clone();
            let gi = mul(py, g()?, call(py, "aten.cos.default", vec![x])?)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.cos.default" => {
            let ops = bind(py, node, env, &["self"])?;
            let x = required(op, &ops, 0, "self")?.value.clone();
            let gi = neg(py, mul(py, g()?, call(py, "aten.sin.default", vec![x])?)?)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.tanh.default" => {
            let ops = bind(py, node, env, &["self"])?;
            let squared = mul(py, outs[0].clone(), outs[0].clone())?;
            let one_minus = call(py, "aten.rsub.Scalar", vec![squared, scalar(py, 1.0)?])?;
            let gi = mul(py, g()?, one_minus)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.sigmoid.default" => {
            let ops = bind(py, node, env, &["self"])?;
            let one_minus = call(py, "aten.rsub.Scalar", vec![outs[0].clone(), scalar(py, 1.0)?])?;
            let gi = mul(py, g()?, mul(py, outs[0].clone(), one_minus)?)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.relu.default" => {
            let ops = bind(py, node, env, &["self"])?;
            let positive = call(py, "aten.gt.Scalar", vec![outs[0].clone(), scalar(py, 0.0)?])?;
            let gi = call(
                py,
                "aten.where.ScalarOther",
                vec![positive, g()?, scalar(py, 0.0)?],
            )?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.silu.default" => {
            // x * sigmoid(x), so d/dx = s * (1 + x * (1 - s)). Written with the
            // sigmoid recomputed rather than divided out of the output: at
            // x = 0 the output is 0 and `out / x` is a division by zero, which
            // is the shape of bug a test at ordinary inputs never sees.
            let ops = bind(py, node, env, &["self"])?;
            let x = required(op, &ops, 0, "self")?.value.clone();
            let s = call(py, "aten.sigmoid.default", vec![x.clone()])?;
            let one_minus = call(py, "aten.rsub.Scalar", vec![s.clone(), scalar(py, 1.0)?])?;
            let inner = call(
                py,
                "aten.add.Scalar",
                vec![mul(py, x, one_minus)?, scalar(py, 1.0)?],
            )?;
            let gi = mul(py, g()?, mul(py, s, inner)?)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.gelu.default" => {
            let ops = bind(py, node, env, &["self", "approximate"])?;
            let approximate = match ops.get(1).and_then(|s| s.as_ref()) {
                Some(operand) if !operand.value.is_none() => {
                    operand.value.extract::<String>().unwrap_or_else(|_| "none".into())
                }
                _ => "none".to_string(),
            };
            if approximate != "none" {
                return Err(crate::err::not_implemented(format!(
                    "torch._C tape: gelu(approximate={approximate:?}) has a different \
                     derivative from the exact one, and only the exact one is written here"
                )));
            }
            let x = required(op, &ops, 0, "self")?.value.clone();
            // 0.5 * (1 + erf(x/sqrt2)) + x * exp(-x^2/2) / sqrt(2*pi)
            let scaled = muls(py, x.clone(), std::f64::consts::FRAC_1_SQRT_2)?;
            let erfed = call(py, "aten.erf.default", vec![scaled])?;
            let cdf = muls(py, call(py, "aten.add.Scalar", vec![erfed, scalar(py, 1.0)?])?, 0.5)?;
            let squared = mul(py, x.clone(), x.clone())?;
            let bell = call(py, "aten.exp.default", vec![muls(py, squared, -0.5)?])?;
            let pdf = muls(py, mul(py, x, bell)?, 1.0 / (2.0 * std::f64::consts::PI).sqrt())?;
            let gi = mul(py, g()?, add(py, cdf, pdf)?)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.masked_fill.Scalar" => {
            let ops = bind(py, node, env, &["self", "mask", "value"])?;
            let mask = required(op, &ops, 1, "mask")?.value.clone();
            let gi = call(
                py,
                "aten.masked_fill.Scalar",
                vec![g()?, mask, scalar(py, 0.0)?],
            )?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.where.self" => {
            let ops = bind(py, node, env, &["condition", "self", "other"])?;
            let condition = required(op, &ops, 0, "condition")?.value.clone();
            let zero = zeros_like(py, &g()?)?;
            let lhs = required(op, &ops, 1, "self")?;
            let rhs = required(op, &ops, 2, "other")?;
            let gs = if lhs.target.is_some() {
                let picked =
                    call(py, "aten.where.self", vec![condition.clone(), g()?, zero.clone()])?;
                Some(reduce_to(py, picked, &lhs.shape()?)?)
            } else {
                None
            };
            let go = if rhs.target.is_some() {
                let picked = call(py, "aten.where.self", vec![condition, zero, g()?])?;
                Some(reduce_to(py, picked, &rhs.shape()?)?)
            } else {
                None
            };
            Ok((ops, vec![None, gs, go]))
        }

        // ---------------------------------------------------------- reduction
        "aten.sum.default" => {
            let ops = bind(py, node, env, &["self", "dtype"])?;
            let want = required(op, &ops, 0, "self")?.shape()?;
            let gi = expand_to(py, reshape(py, g()?, &vec![1usize; want.len()])?, &want)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.mean.default" => {
            let ops = bind(py, node, env, &["self", "dtype"])?;
            let want = required(op, &ops, 0, "self")?.shape()?;
            let count: usize = want.iter().product();
            let scaled = muls(py, g()?, 1.0 / count as f64)?;
            let gi = expand_to(py, reshape(py, scaled, &vec![1usize; want.len()])?, &want)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.sum.dim_IntList" | "aten.mean.dim" => {
            let ops = bind(py, node, env, &["self", "dim", "keepdim", "dtype"])?;
            let want = required(op, &ops, 0, "self")?.shape()?;
            let rank = want.len();
            let axes: Vec<usize> = i64_list(&ops, 1, "dim", op)?
                .into_iter()
                .map(|d| normalise_dim(d, rank))
                .collect();
            let keepdim = opt_bool(&ops, 2, false)?;
            // Put the reduced axes back as extent 1, then broadcast. Doing it
            // by reshape rather than by `unsqueeze` per axis keeps the order of
            // several reduced axes from mattering.
            let mut kept: Vec<usize> = want.clone();
            for axis in &axes {
                kept[*axis] = 1;
            }
            let mut gi = g()?;
            if !keepdim {
                gi = reshape(py, gi, &kept)?;
            }
            if op == "aten.mean.dim" {
                let count: usize = axes.iter().map(|a| want[*a]).product();
                gi = muls(py, gi, 1.0 / count as f64)?;
            }
            let gi = expand_to(py, gi, &want)?;
            Ok((ops, vec![Some(gi)]))
        }

        // ------------------------------------------------------------- matmul
        "aten.mm.default" | "aten.matmul.default" | "aten.bmm.default" => {
            let second = if op == "aten.mm.default" || op == "aten.bmm.default" {
                "mat2"
            } else {
                "other"
            };
            let ops = bind(py, node, env, &["self", second])?;
            let lhs = required(op, &ops, 0, "self")?;
            let rhs = required(op, &ops, 1, second)?;
            let lshape = lhs.shape()?;
            let rshape = rhs.shape()?;
            if lshape.len() < 2 || rshape.len() < 2 {
                return Err(crate::err::not_implemented(format!(
                    "torch._C tape: {op} with a 1-D operand contracts a dimension away, and \
                     that case is not written here -- shapes were {lshape:?} and {rshape:?}"
                )));
            }
            let gs = if lhs.target.is_some() {
                let product = matmul(py, g()?, mt(py, rhs.value.clone())?)?;
                Some(reduce_to(py, product, &lshape)?)
            } else {
                None
            };
            let go = if rhs.target.is_some() {
                let product = matmul(py, mt(py, lhs.value.clone())?, g()?)?;
                Some(reduce_to(py, product, &rshape)?)
            } else {
                None
            };
            Ok((ops, vec![gs, go]))
        }
        "aten.addmm.default" => {
            let ops = bind(py, node, env, &["self", "mat1", "mat2", "beta", "alpha"])?;
            let beta = opt_f64(&ops, 3, 1.0)?;
            let alpha = opt_f64(&ops, 4, 1.0)?;
            let bias = required(op, &ops, 0, "self")?;
            let mat1 = required(op, &ops, 1, "mat1")?;
            let mat2 = required(op, &ops, 2, "mat2")?;
            let gb = if bias.target.is_some() {
                let scaled = if beta == 1.0 { g()? } else { muls(py, g()?, beta)? };
                Some(reduce_to(py, scaled, &bias.shape()?)?)
            } else {
                None
            };
            let g1 = if mat1.target.is_some() {
                let product = call(py, "aten.mm.default", vec![g()?, mt(py, mat2.value.clone())?])?;
                Some(if alpha == 1.0 { product } else { muls(py, product, alpha)? })
            } else {
                None
            };
            let g2 = if mat2.target.is_some() {
                let product = call(py, "aten.mm.default", vec![mt(py, mat1.value.clone())?, g()?])?;
                Some(if alpha == 1.0 { product } else { muls(py, product, alpha)? })
            } else {
                None
            };
            Ok((ops, vec![gb, g1, g2]))
        }

        // ------------------------------------------------------- slice & join
        "aten.cat.default" => {
            let items = bind_list(py, node, env, 0, "tensors")?;
            let ops = bind(py, node, env, &["tensors", "dim"])?;
            let rank = dims(&outs[0])?.len();
            let axis = normalise_dim(opt_i64(&ops, 1, 0)?, rank) as i64;
            let mut grads = Vec::with_capacity(items.len());
            let mut offset: i64 = 0;
            for item in &items {
                let shape = item.shape()?;
                // `cat` has a legacy exemption for the *empty* tensor: a
                // 1-D size-0 operand is skipped rather than checked against
                // the concatenation axis. `DynamicCache` relies on it -- an
                // empty past key is spelled `[0]` and concatenated with a
                // `[1, 3, 8, 64]` present one -- so a rule that assumed every
                // operand had the output's rank crashed on the first real
                // model it saw, at layer 0.
                if shape.len() != rank {
                    if shape.iter().product::<usize>() != 0 {
                        return Err(crate::err::not_implemented(format!(
                            "torch._C tape: cat was given a {shape:?} operand and a rank-{rank} \
                             result, and only the empty-tensor exemption explains that"
                        )));
                    }
                    grads.push(Some(zeros_like(py, &item.value)?));
                    continue;
                }
                let extent = shape[axis as usize] as i64;
                let piece = call(
                    py,
                    "aten.slice.Tensor",
                    vec![
                        g()?,
                        axis.into_bound_py_any(py)?,
                        offset.into_bound_py_any(py)?,
                        (offset + extent).into_bound_py_any(py)?,
                        1i64.into_bound_py_any(py)?,
                    ],
                )?;
                offset += extent;
                grads.push(Some(piece));
            }
            Ok((items.into_iter().map(Some).collect(), grads))
        }
        "aten.slice.Tensor" => {
            let ops = bind(py, node, env, &["self", "dim", "start", "end", "step"])?;
            let source = required(op, &ops, 0, "self")?;
            let shape = source.shape()?;
            let rank = shape.len();
            let axis = normalise_dim(opt_i64(&ops, 1, 0)?, rank);
            let step = opt_i64(&ops, 4, 1)?;
            if step != 1 {
                return Err(crate::err::not_implemented(format!(
                    "torch._C tape: slice with step {step} scatters its gradient into every \
                     {step}th row, and only the contiguous case is written here"
                )));
            }
            let extent = shape[axis] as i64;
            let raw_start = opt_i64(&ops, 2, 0)?;
            let start = if raw_start < 0 { (raw_start + extent).max(0) } else { raw_start.min(extent) };
            let taken = dims(&outs[0])?[axis] as i64;
            let after = extent - start - taken;
            // `constant_pad_nd` counts pairs from the *last* dimension
            // backwards, so an axis near the front needs the axes after it
            // spelled out as zero pairs.
            let mut pad = vec![0i64; 2 * (rank - axis)];
            let slot = 2 * (rank - 1 - axis);
            pad[slot] = start;
            pad[slot + 1] = after;
            let gi = call(
                py,
                "aten.constant_pad_nd.default",
                vec![g()?, ints(py, &pad)?, scalar(py, 0.0)?],
            )?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten.constant_pad_nd.default" => {
            // The exact inverse of the `slice` rule, and it earns its place:
            // `transformers` builds its shifted labels with `F.pad`, so this is
            // on the path of every `labels=` forward. That it is reached at all
            // by *ids* rather than by an activation is what §3 of
            // docs/BACKWARD.md is about.
            let ops = bind(py, node, env, &["self", "pad", "value"])?;
            let source = required(op, &ops, 0, "self")?;
            let shape = source.shape()?;
            let rank = shape.len();
            let pad = i64_list(&ops, 1, "pad", op)?;
            if pad.iter().any(|p| *p < 0) {
                return Err(crate::err::not_implemented(format!(
                    "torch._C tape: constant_pad_nd with a negative pad {pad:?} crops rather \
                     than pads, and its gradient would have to pad back; only the padding \
                     case is written here"
                )));
            }
            let mut gi = g()?;
            for (pair, chunk) in pad.chunks(2).enumerate() {
                if chunk[0] == 0 && chunk.get(1).copied().unwrap_or(0) == 0 {
                    continue;
                }
                let axis = (rank - 1 - pair) as i64;
                let start = chunk[0];
                let end = start + shape[axis as usize] as i64;
                gi = call(
                    py,
                    "aten.slice.Tensor",
                    vec![
                        gi,
                        axis.into_bound_py_any(py)?,
                        start.into_bound_py_any(py)?,
                        end.into_bound_py_any(py)?,
                        1i64.into_bound_py_any(py)?,
                    ],
                )?;
            }
            Ok((ops, vec![Some(gi)]))
        }
        "aten.select.int" => {
            let ops = bind(py, node, env, &["self", "dim", "index"])?;
            let source = required(op, &ops, 0, "self")?;
            let shape = source.shape()?;
            let rank = shape.len();
            let axis = normalise_dim(opt_i64(&ops, 1, 0)?, rank);
            let extent = shape[axis] as i64;
            let raw = opt_i64(&ops, 2, 0)?;
            let index = if raw < 0 { raw + extent } else { raw };
            let lifted = call(
                py,
                "aten.unsqueeze.default",
                vec![g()?, (axis as i64).into_bound_py_any(py)?],
            )?;
            let mut pad = vec![0i64; 2 * (rank - axis)];
            let slot = 2 * (rank - 1 - axis);
            pad[slot] = index;
            pad[slot + 1] = extent - index - 1;
            let gi = call(
                py,
                "aten.constant_pad_nd.default",
                vec![lifted, ints(py, &pad)?, scalar(py, 0.0)?],
            )?;
            Ok((ops, vec![Some(gi)]))
        }

        // ------------------------------------------------------------ softmax
        "aten._log_softmax.default" => {
            let ops = bind(py, node, env, &["self", "dim", "half_to_float"])?;
            let rank = dims(&outs[0])?.len();
            let axis = opt_i64(&ops, 1, -1)?;
            let total = sum_dims(py, g()?, &[normalise_dim(axis, rank) as i64], true)?;
            let softmaxed = call(py, "aten.exp.default", vec![outs[0].clone()])?;
            let gi = sub(py, g()?, mul(py, softmaxed, total)?)?;
            Ok((ops, vec![Some(gi)]))
        }
        "aten._softmax.default" => {
            let ops = bind(py, node, env, &["self", "dim", "half_to_float"])?;
            let rank = dims(&outs[0])?.len();
            let axis = normalise_dim(opt_i64(&ops, 1, -1)?, rank) as i64;
            let weighted = sum_dims(py, mul(py, g()?, outs[0].clone())?, &[axis], true)?;
            let gi = mul(py, sub(py, g()?, weighted)?, outs[0].clone())?;
            Ok((ops, vec![Some(gi)]))
        }
        // `_safe_softmax` -- softmax's own rule, and the whole of the "safe"
        // part is that it is applied to the **saved output**.
        //
        // The third argument is `dtype`, not `half_to_float`, so this cannot
        // share `_softmax`'s arm even though the expression is identical: the
        // vendored tree binds arguments by *name* before dispatching
        // (docs/CAPTURE.md §2), so a trace can hold `{'dtype': ...}` and a
        // rule reading `half_to_float` would find nothing there.
        //
        // **A fully-masked row is where the two softmaxes differ, and it is a
        // real shape** -- every key excluded by a causal mask plus padding.
        // Measured on upstream 2.13.0, on a row that is entirely `-inf`:
        //
        // ```text
        //   _softmax        forward nan   backward nan
        //   _safe_softmax   forward 0     backward 0     (SafeSoftmaxBackward0)
        // ```
        //
        // and upstream's `0` there is **not** a special case in the backward.
        // `(g - rowsum(g*p)) * p` with `p = 0` is `0` by arithmetic, so the
        // forward's zero is what makes the gradient finite. That is only true
        // of a rule that *reads* `outs[0]`: one that recomputed a softmax of
        // the recorded input -- which the tape is allowed to do, it runs
        // outside a capture region -- would exponentiate `-inf - (-inf)` and
        // answer `nan` on exactly the rows this op exists for, while agreeing
        // everywhere a finite-difference case can look. §15's V2 is that
        // fault and the test that catches it is a structural one.
        "aten._safe_softmax.default" => {
            let ops = bind(py, node, env, &["self", "dim", "dtype"])?;
            let rank = dims(&outs[0])?.len();
            let axis = normalise_dim(opt_i64(&ops, 1, -1)?, rank) as i64;
            let weighted = sum_dims(py, mul(py, g()?, outs[0].clone())?, &[axis], true)?;
            let gi = mul(py, sub(py, g()?, weighted)?, outs[0].clone())?;
            // `dtype`, when given, upcasts the forward. Upstream differentiates
            // in that dtype and narrows back to the input's -- measured: a
            // `float32` self with `dtype=float64` gives a `float32` gradient
            // equal to the `float64` rule rounded once.
            let gi = cast_like(py, gi, &required(op, &ops, 0, "self")?.value)?;
            Ok((ops, vec![Some(gi)]))
        }

        // ------------------------------------------------------ normalisation
        "aten.native_layer_norm.default" => {
            layer_norm_backward(py, node, env, gouts, outs)
        }
        "aten.native_batch_norm.default" => {
            batch_norm_backward(py, node, env, gouts, outs)
        }

        // ------------------------------------------------------------ dropout
        "aten.native_dropout.default" => {
            native_dropout_backward(py, node, env, gouts, outs)
        }

        // --------------------------------------------------------------- loss
        "aten.nll_loss_forward.default" => {
            nll_loss_backward(py, node, env, gouts, outs)
        }

        // ------------------------------------------------------------- split
        //
        // The one op here whose forward answers with a **list**, so `gouts` has
        // one slot per chunk rather than per tuple position. The derivative is
        // the concatenation of the chunk gradients along the split axis, and
        // the part that is not optional is the *zero*: a chunk no gradient
        // reached still occupies its width in the input, so it has to be
        // present in the `cat` at full size. GPT-2 splits a fused qkv
        // projection three ways and then uses all three, so the model that
        // needs this rule is exactly the model that cannot exercise that zero.
        "aten.split.Tensor" => {
            let ops = bind(py, node, env, &["self", "split_size", "dim"])?;
            let input = required(op, &ops, 0, "self")?;
            let rank = input.shape()?.len();
            let axis = normalise_dim(opt_i64(&ops, 2, 0)?, rank) as i64;
            if outs.is_empty() {
                return Err(crate::err::not_implemented(
                    "torch._C tape: split's gradient needs the chunks the forward produced \
                     and this node recorded none",
                ));
            }
            let mut pieces = Vec::with_capacity(outs.len());
            for (slot, chunk) in outs.iter().enumerate() {
                pieces.push(match gouts.get(slot).and_then(|g| g.clone()) {
                    Some(g) => g,
                    None => zeros_like(py, chunk)?,
                });
            }
            let list = PyList::new(py, pieces)?.into_any();
            let gi = call(py, "aten.cat.default", vec![list, axis.into_bound_py_any(py)?])?;
            Ok((ops, vec![Some(gi)]))
        }

        // ---------------------------------------------------------- embedding
        //
        // Upstream calls this `embedding_dense_backward` and it is one of the
        // five ops docs/AUTOGRAD.md §5.1 listed as missing. It is a scatter-add
        // into a zero buffer -- `index_put_(accumulate=True)`, which
        // docs/VIEWS.md §7 landed.
        //
        // **It was a one-hot matrix and a matmul until that flag existed**, and
        // the switch is not only about the `[vocab, tokens]` intermediate the
        // one-hot cost (1.5 MB at SmolLM2's 8 tokens, 200 MB at 1024). The two
        // compositions are not the same function at reduced precision:
        // upstream's kernel is `*dst += *src` in the receiver's dtype, so the
        // running sum is rounded at every step, while a matmul accumulates in
        // `float32` (docs/ARCH.md's GEMM accumulate-dtype rule) and rounds
        // once. docs/VIEWS.md §7.4 measured both against upstream and this one
        // is the one that agrees; §4.5 of docs/BACKWARD.md measures the
        // difference on the real table. At `float32` they are identical, so the
        // switch cannot move a `float32` gradient -- which is the check that
        // says nothing else came with it.
        "aten.embedding.default" => {
            let ops = bind(
                py,
                node,
                env,
                &["weight", "indices", "padding_idx", "scale_grad_by_freq", "sparse"],
            )?;
            let weight = required(op, &ops, 0, "weight")?;
            let indices = required(op, &ops, 1, "indices")?.value.clone();
            let wshape = weight.shape()?;
            if wshape.len() != 2 {
                return Err(crate::err::not_implemented(format!(
                    "torch._C tape: embedding's gradient is written for a 2-D table and this \
                     one is {wshape:?}"
                )));
            }
            let (rows, width) = (wshape[0], wshape[1]);
            let count: usize = dims(&indices)?.iter().product();
            let flat_rows = reshape(py, indices.clone(), &[count])?;
            let mut flat_grad = reshape(py, g()?, &[count, width])?;
            let padding = match ops.get(2).and_then(|s| s.as_ref()) {
                Some(operand) if !operand.value.is_none() => operand.value.extract::<i64>().ok(),
                _ => None,
            };
            if let Some(row) = padding {
                // A padding row gets no gradient. The one-hot spelling zeroed a
                // *column of the one-hot*; with no one-hot to zero, the same
                // two ops are applied to the contributions instead -- every
                // token whose id is the padding one contributes nothing, which
                // is the same statement one step later. The mask is `[count, 1]`
                // and broadcasts across the width.
                let kept = call(
                    py,
                    "aten.ne.Scalar",
                    vec![flat_rows.clone(), row.into_bound_py_any(py)?],
                )?;
                let mask = reshape(py, kept, &[count, 1])?;
                flat_grad = call(
                    py,
                    "aten.where.ScalarOther",
                    vec![mask, flat_grad, scalar(py, 0.0)?],
                )?;
            }
            let dtype = weight.value.getattr("dtype")?;
            let size = ints(py, &[rows as i64, width as i64])?;
            let table = call_kw(py, "aten.zeros.default", vec![size], vec![("dtype", dtype)])?;
            let index_list = PyList::new(py, [flat_rows])?.into_any();
            let gi = call(
                py,
                "aten.index_put_.default",
                vec![table, index_list, flat_grad, true.into_bound_py_any(py)?],
            )?;
            Ok((ops, vec![Some(gi), None]))
        }

        // ---------------------------------------------------------------- sdpa
        "aten._scaled_dot_product_flash_attention_for_cpu.default" => {
            sdpa_backward(py, node, env, gouts)
        }

        _ => Err(no_rule(op)),
    }
}

/// `nll_loss_forward` -> the gradient of its first result.
///
/// The three things this has to get right are the three `docs/LOSS.md` §3
/// found in the forward, read from the other side:
///
/// * the divisor for `reduction=Mean` is `total_weight`, the op's **second**
///   result, and not the number of rows;
/// * an ignored target contributes nothing, and its index cannot be used to
///   scatter with because it is routinely out of range (`-100`);
/// * `reduction=None` hands back a gradient per row rather than a scalar.
fn nll_loss_backward<'py>(
    py: Python<'py>,
    node: &Node,
    env: &Env,
    gouts: &[Option<Obj<'py>>],
    outs: &[Obj<'py>],
) -> PyResult<Rule<'py>> {
    const OP: &str = "aten.nll_loss_forward.default";
    let ops = bind(py, node, env, &["self", "target", "weight", "reduction", "ignore_index"])?;
    if gouts.len() > 1 && gouts[1].is_some() {
        return Err(crate::err::not_implemented(
            "torch._C tape: a gradient reached nll_loss_forward's total_weight, which is a \
             count rather than a differentiable value",
        ));
    }
    let g = gouts
        .first()
        .and_then(|slot| slot.clone())
        .ok_or_else(|| missing(OP, "grad"))?;

    let logits = required(OP, &ops, 0, "self")?;
    let target = required(OP, &ops, 1, "target")?.value.clone();
    let reduction = opt_i64(&ops, 3, 1)?;
    let ignore_index = opt_i64(&ops, 4, -100)?;
    let shape = logits.shape()?;
    if shape.len() != 2 {
        return Err(crate::err::not_implemented(format!(
            "torch._C tape: nll_loss_forward's gradient is written for a 2-D input and this \
             one is {shape:?}"
        )));
    }

    // Mean divides by `total_weight`; Sum and None do not. Reading it off the
    // recorded second result rather than recomputing it is what keeps the
    // weighted and the ignored cases right at once.
    let mut scaled = match reduction {
        1 => div(py, g, outs[1].clone())?,
        _ => g,
    };
    if let Some(weight) = ops.get(2).and_then(|s| s.as_ref()) {
        if !weight.value.is_none() {
            let safe = call(
                py,
                "aten.clamp.default",
                vec![target.clone(), 0i64.into_bound_py_any(py)?, py.None().into_bound(py)],
            )?;
            let index_list = PyList::new(py, [safe])?.into_any();
            let per_row = call(py, "aten.index.Tensor", vec![weight.value.clone(), index_list])?;
            scaled = mul(py, scaled, per_row)?;
        }
    }
    let kept = call(
        py,
        "aten.ne.Scalar",
        vec![target.clone(), ignore_index.into_bound_py_any(py)?],
    )?;
    let rows = shape[0];
    let contribution = neg(py, expand_to(py, reshape(py, scaled, &[1])?, &[rows])?)?;
    let zeroed = call(
        py,
        "aten.where.ScalarOther",
        vec![kept.clone(), contribution, scalar(py, 0.0)?],
    )?;
    // An ignored row's target is not a column, so it cannot be scattered to.
    // Sending it to column 0 is safe precisely because its value is already 0.
    let column = call(
        py,
        "aten.where.ScalarOther",
        vec![kept, target, 0i64.into_bound_py_any(py)?],
    )?;
    let index = reshape(py, column, &[rows, 1])?;
    let source = reshape(py, zeroed, &[rows, 1])?;
    let buffer = zeros_like(py, &logits.value)?;
    let gi = call(
        py,
        "aten.scatter.src",
        vec![buffer, 1i64.into_bound_py_any(py)?, index, source],
    )?;
    Ok((ops, vec![Some(gi), None]))
}

/// `g` in `like`'s dtype, and `g` itself when they already agree.
///
/// Only mixed precision makes this do anything, and `native_layer_norm` is
/// where mixed precision is *supported* rather than an error: `aten.rs`
/// measured that `float32` parameters in front of a `bfloat16` input give
/// `float32` statistics while the output stays `bfloat16`. A rule that read
/// those statistics without narrowing again would hand back a `float32`
/// gradient for a `bfloat16` input, which is a widening no forward asked for.
fn cast_like<'py>(py: Python<'py>, g: Obj<'py>, like: &Obj<'py>) -> PyResult<Obj<'py>> {
    let want = like.getattr("dtype")?;
    if g.getattr("dtype")?.eq(&want)? {
        return Ok(g);
    }
    call_kw(py, "aten._to_copy.default", vec![g], vec![("dtype", want)])
}

/// `native_layer_norm` -> gradients for input, weight and bias.
///
/// Three gradients, and only one of them is difficult. With `N` the number of
/// elements the statistics were taken over, `y = (x - mean) * rstd` and
/// `gh = g * weight`:
///
/// ```text
///   grad_bias   = sum over the outer axes of  g
///   grad_weight = sum over the outer axes of  g * y
///   grad_input  = rstd * ( gh - mean(gh) - y * mean(gh * y) )
/// ```
///
/// The two reductions are `reduce_to` under another name and would be right by
/// accident. **`grad_input`'s last two terms are the ones a plausible
/// implementation omits**: they are the correction for the fact that `mean` and
/// `rstd` are themselves functions of every element of the row, so a rule that
/// stops at `rstd * gh` -- which has the right shape, the right dtype and the
/// right order of magnitude -- is wrong everywhere and looks right. `docs/BACKWARD.md`
/// §7's T3 is the same shape of omission one layer down.
///
/// **`mean` and `rstd` are read off the forward's second and third results**
/// rather than recomputed. That is what those results are *for* -- the same
/// reading `docs/LOSS.md` §3.1 made of `nll_loss_forward`'s `total_weight`, and
/// it is the reading that matters under mixed precision, where `aten.rs`
/// measured the statistics following the *parameter* dtype and not the input's.
/// Recomputing them would silently substitute the input's precision there.
/// **W11** (docs/BACKWARD8.md §2): the derivative of `native_batch_norm`.
///
/// The reason it is here is a measurement, not a checklist. `docs/BACKWARD7.md`
/// §10 row 3 predicted that a model writing a buffer mid-forward -- a KV cache,
/// a batch-norm running statistic -- would be refused where upstream answers,
/// and had not checked it against a real model. Checked: a real
/// `nn.Sequential(Conv2d, BatchNorm2d, ReLU)` in `train()` mode was refused,
/// and behind the storage-version defect that `Recorder::forgive_own_write`
/// closes stood a second wall -- there was no rule here at all, so the fixed
/// guard would have handed back *"no derivative rule"* instead. Fixing one of
/// two walls buys nothing, so this is the other one.
///
/// Two modes, and they are different functions:
///
/// * **`training=true`.** The statistics are the batch's own, so `x` reaches
///   the output through the mean and the variance as well as directly, and the
///   two extra terms are the whole content of the rule. It reads `save_mean`
///   and `save_invstd`, which the forward *returns* precisely so a backward
///   need not recompute them -- and, importantly, it does **not** read
///   `running_mean`/`running_var`. That is what makes the storage-version
///   forgiveness in `capture.rs` safe: the buffers this op writes are an output
///   of the forward and an input to nothing.
/// * **`training=false`.** The statistics are constants, `x` reaches the output
///   only directly, and the rule is one multiply. Here `running_mean` and
///   `running_var` *are* read -- and in this mode the op writes nothing, so
///   they are held to the ordinary freshness rule with no forgiveness.
///
/// The channel axis is 1 by definition of this op and the reduction is every
/// other axis, which is the one structural difference from
/// `layer_norm_backward` above: layer norm reduces a *suffix* and keeps the
/// statistics at the input's rank, batch norm reduces the complement of one
/// axis and keeps its statistics at rank 1. The reshape to `[1, C, 1, ...]` is
/// therefore explicit here and absent there.
fn batch_norm_backward<'py>(
    py: Python<'py>,
    node: &Node,
    env: &Env,
    gouts: &[Option<Obj<'py>>],
    outs: &[Obj<'py>],
) -> PyResult<Rule<'py>> {
    const OP: &str = "aten.native_batch_norm.default";
    let ops = bind(
        py,
        node,
        env,
        &[
            "input",
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "training",
            "momentum",
            "eps",
        ],
    )?;
    // The same refusal `layer_norm_backward` makes, for the same reason: these
    // two results exist so that a backward need not recompute them, and
    // nothing here differentiates a saved statistic.
    for (slot, name) in [(1usize, "save_mean"), (2usize, "save_invstd")] {
        if gouts.len() > slot && gouts[slot].is_some() {
            return Err(crate::err::not_implemented(format!(
                "torch._C tape: a gradient reached native_batch_norm's {name}, which this \
                 op returns so that a backward need not recompute it; nothing here \
                 differentiates a saved statistic"
            )));
        }
    }
    let g = gouts
        .first()
        .and_then(|slot| slot.clone())
        .ok_or_else(|| missing(OP, "grad"))?;

    let input = required(OP, &ops, 0, "input")?;
    let xshape = input.shape()?;
    let rank = xshape.len();
    if rank < 2 {
        return Err(crate::err::not_implemented(format!(
            "torch._C tape: native_batch_norm's gradient was given a rank-{rank} input, and \
             this op's channel axis is 1 by definition"
        )));
    }
    let channels = xshape[1];
    // Every axis but the channel one. `[N, C, *spatial]` reduces over `N` and
    // the spatial axes, which is what makes `M` the count *per channel* rather
    // than per row.
    let reduce: Vec<i64> = (0..rank as i64).filter(|d| *d != 1).collect();
    let count: f64 = reduce
        .iter()
        .map(|d| xshape[*d as usize])
        .product::<usize>() as f64;
    // The shape a `[C]` statistic has to take to broadcast against `x`.
    let mut broadcast = vec![1usize; rank];
    broadcast[1] = channels;

    let training = opt_bool(&ops, 5, true)?;
    let eps = opt_f64(&ops, 7, 1e-5)?;

    let weight = ops
        .get(1)
        .and_then(|slot| slot.as_ref())
        .filter(|operand| !operand.value.is_none());
    let bias = ops
        .get(2)
        .and_then(|slot| slot.as_ref())
        .filter(|operand| !operand.value.is_none());

    // Where the interior is computed. Under mixed precision the statistics are
    // the parameters' dtype and the input is not, and `layer_norm_backward`
    // above records what happens to a rule that ignores that: `x - mean` with
    // a `bfloat16` x against a `float32` mean is a promotion this shim declines
    // by name, so the rule would not lose precision, it would refuse.
    let (mean, invstd) = if training {
        if outs.len() < 3 {
            return Err(crate::err::not_implemented(format!(
                "torch._C tape: native_batch_norm's training gradient reads the save_mean \
                 and save_invstd this op returns, and this node recorded {} result(s)",
                outs.len()
            )));
        }
        (
            reshape(py, outs[1].clone(), &broadcast)?,
            reshape(py, outs[2].clone(), &broadcast)?,
        )
    } else {
        // Eval. `save_mean`/`save_invstd` come back empty in this mode
        // (measured on 2.13.0: both are shape `[0]`), so the statistics are the
        // running buffers and the reciprocal square root has to be formed here.
        let rm = required(OP, &ops, 3, "running_mean")?;
        let rv = required(OP, &ops, 4, "running_var")?;
        if rm.value.is_none() || rv.value.is_none() {
            return Err(crate::err::not_implemented(
                "torch._C tape: native_batch_norm with training=False and no running \
                 statistics normalises by nothing this rule can name -- upstream's kernel \
                 rejects that combination too"
                    .to_string(),
            ));
        }
        let shifted = call(py, "aten.add.Scalar", vec![rv.value.clone(), scalar(py, eps)?])?;
        let rstd = call(py, "aten.rsqrt.default", vec![shifted])?;
        (
            reshape(py, rm.value.clone(), &broadcast)?,
            reshape(py, rstd, &broadcast)?,
        )
    };

    let xa = cast_like(py, input.value.clone(), &mean)?;
    let ga = cast_like(py, g.clone(), &mean)?;
    let xhat = mul(py, sub(py, xa, mean)?, invstd.clone())?;

    // `g * weight`, which is `dL/dy_normalised`.
    let gh = match weight {
        Some(w) => {
            let wb = reshape(py, cast_like(py, w.value.clone(), &ga)?, &broadcast)?;
            mul(py, ga.clone(), wb)?
        }
        None => ga.clone(),
    };

    let gi = if training {
        // The batch's own statistics depend on every element, so `x` reaches
        // the loss three ways. Dropping either mean term gives a gradient that
        // is right to a few percent on a large batch and wrong on a small one,
        // which is why the test for this rule is element-wise against upstream
        // on N=2 rather than a tolerance on N=64.
        let per = |py: Python<'py>, t: Obj<'py>| -> PyResult<Obj<'py>> {
            let summed = sum_dims(py, t, &reduce, true)?;
            call(py, "aten.div.Scalar", vec![summed, scalar(py, count)?])
        };
        let mean_gh = per(py, gh.clone())?;
        let mean_gh_xhat = per(py, mul(py, gh.clone(), xhat.clone())?)?;
        let centred = sub(
            py,
            sub(py, gh, mean_gh)?,
            mul(py, xhat.clone(), mean_gh_xhat)?,
        )?;
        mul(py, centred, invstd)?
    } else {
        // Eval: the statistics are constants, so this is the chain rule and
        // nothing else.
        mul(py, gh, invstd)?
    };
    let gi = cast_like(py, gi, &input.value)?;

    let gw = match weight {
        None => None,
        Some(w) => {
            let summed = sum_dims(py, mul(py, ga.clone(), xhat)?, &reduce, false)?;
            Some(cast_like(py, reshape(py, summed, &[channels])?, &w.value)?)
        }
    };
    let gb = match bias {
        None => None,
        Some(b) => {
            let summed = sum_dims(py, ga, &reduce, false)?;
            Some(cast_like(py, reshape(py, summed, &[channels])?, &b.value)?)
        }
    };

    // `running_mean` and `running_var` get no gradient: they are buffers, they
    // are `requires_grad=False` upstream, and in training mode this rule does
    // not even read them.
    Ok((ops, vec![Some(gi), gw, gb, None, None]))
}

fn layer_norm_backward<'py>(
    py: Python<'py>,
    node: &Node,
    env: &Env,
    gouts: &[Option<Obj<'py>>],
    outs: &[Obj<'py>],
) -> PyResult<Rule<'py>> {
    const OP: &str = "aten.native_layer_norm.default";
    let ops = bind(py, node, env, &["input", "normalized_shape", "weight", "bias", "eps"])?;
    for (slot, name) in [(1usize, "mean"), (2usize, "rstd")] {
        if gouts.len() > slot && gouts[slot].is_some() {
            return Err(crate::err::not_implemented(format!(
                "torch._C tape: a gradient reached native_layer_norm's {name}, which this op \
                 returns so that a backward need not recompute it; nothing here \
                 differentiates a saved statistic"
            )));
        }
    }
    let g = gouts
        .first()
        .and_then(|slot| slot.clone())
        .ok_or_else(|| missing(OP, "grad"))?;

    let input = required(OP, &ops, 0, "input")?;
    let xshape = input.shape()?;
    let rank = xshape.len();
    let normalized = i64_list(&ops, 1, "normalized_shape", OP)?;
    let k = normalized.len();
    if k == 0 || k > rank {
        return Err(crate::err::not_implemented(format!(
            "torch._C tape: native_layer_norm's gradient was given normalized_shape \
             {normalized:?} against a rank-{rank} input"
        )));
    }
    if outs.len() < 3 {
        return Err(crate::err::not_implemented(format!(
            "torch._C tape: native_layer_norm's gradient reads the mean and rstd this op \
             returns, and this node recorded {} result(s)",
            outs.len()
        )));
    }
    let outer = rank - k;
    let inner: Vec<i64> = (outer..rank).map(|d| d as i64).collect();
    let want: Vec<usize> = xshape[outer..].to_vec();
    let count = want.iter().product::<usize>() as f64;

    // The op's own second and third results, kept at the input's rank with the
    // normalised axes replaced by 1 -- so they broadcast against `x` with no
    // reshaping, which is the shape `aten.rs` measured and the reason they are
    // usable here at all.
    let mean = outs[1].clone();
    let rstd = outs[2].clone();
    // The interior is computed in the **statistics'** dtype, which under mixed
    // precision is the parameters' rather than the input's. Upstream's backward
    // does the same, and without it this rule would not merely lose precision --
    // it would refuse: `x - mean` with a `bfloat16` x against a `float32` mean
    // is a promotion this shim declines by name. Each result is narrowed back
    // on the way out, to the dtype of the thing it is a gradient *for*.
    let xa = cast_like(py, input.value.clone(), &mean)?;
    let ga = cast_like(py, g.clone(), &mean)?;
    let y = mul(py, sub(py, xa, mean.clone())?, rstd.clone())?;

    let weight = ops
        .get(2)
        .and_then(|slot| slot.as_ref())
        .filter(|operand| !operand.value.is_none());
    let bias = ops
        .get(3)
        .and_then(|slot| slot.as_ref())
        .filter(|operand| !operand.value.is_none());

    let gh = match weight {
        Some(w) => mul(py, ga.clone(), cast_like(py, w.value.clone(), &mean)?)?,
        None => ga.clone(),
    };
    let per = |py: Python<'py>, t: Obj<'py>| -> PyResult<Obj<'py>> {
        let summed = sum_dims(py, t, &inner, true)?;
        call(py, "aten.div.Scalar", vec![summed, scalar(py, count)?])
    };
    let mean_gh = per(py, gh.clone())?;
    let mean_ghy = per(py, mul(py, gh.clone(), y.clone())?)?;
    let centred = sub(py, sub(py, gh, mean_gh)?, mul(py, y.clone(), mean_ghy)?)?;
    let gi = cast_like(py, mul(py, centred, rstd)?, &input.value)?;

    let gw = match weight {
        None => None,
        Some(w) => {
            let summed = reduce_to(py, mul(py, ga.clone(), y)?, &want)?;
            Some(cast_like(py, summed, &w.value)?)
        }
    };
    let gb = match bias {
        None => None,
        Some(b) => {
            let summed = reduce_to(py, ga, &want)?;
            Some(cast_like(py, summed, &b.value)?)
        }
    };
    Ok((ops, vec![Some(gi), None, gw, gb, None]))
}

/// `native_dropout` -> the gradient of its first result.
///
/// ```text
///   grad_input = (g * mask) * scale
/// ```
///
/// **The mask is read off the forward's second result, and that is the whole
/// reason the op returns one.** `docs/LOSS.md` §3.1's reading of
/// `nll_loss_forward`'s `total_weight` met for the third time, and here it is
/// not an optimisation: a dropout mask *cannot* be recomputed. There is no
/// function of the input that says which elements were dropped, so a rule that
/// did not read it would have to draw again -- a different mask, and therefore
/// the gradient of a different function than the one the forward computed.
/// **A forward-only test cannot see this**, which is why §15's D2 is the fault
/// it is.
///
/// Three details are upstream's and two of them would be invisible in
/// `float32`:
///
/// * **The scale is `1 / (1 - p)` and it is guarded exactly as the forward
///   guards it** -- `1` when `train` is false or absent, `0` when `p == 1`
///   (`derivatives.yaml`: `!train.has_value() || !train.value() ? 1 : (p == 1
///   ? 0.0 : 1.0 / (1.0 - p))`). Without the `p == 1` guard the reciprocal is
///   `inf` and every dropped element's gradient becomes `nan` rather than `0`.
/// * **The association is `(g * mask) * scale` and not `g * (mask * scale)`.**
///   Upstream has both spellings, for two different callers:
///   `native_dropout_backward` is the first and
///   `infinitely_differentiable_native_dropout_backward` (taken only when
///   `GradMode` is on, i.e. `create_graph=True`) is the second. Measured on
///   2.13.0 at `p = 0.7`, they are *different numbers* in `bfloat16` and
///   `float16` and identical in `float32` and `float64`; a plain
///   `loss.backward()` takes the first, so this rule does.
/// * **The scalar is not narrowed to the tensor's dtype.** `docs/SCALAR.md`
///   fixed `mul.Scalar` to follow upstream's un-narrowed
///   `original_scalar_value<opmath_t>`, and this rule inherits that by
///   spelling the multiply as `mul.Scalar`. A `bfloat16(1/0.3)` here would
///   disagree with upstream on the survivors. Note this is the *opposite* of
///   what the forward does -- `native_dropout`'s own kernel narrows before it
///   multiplies (`aten.rs`, measured on 1280 combinations) -- so the two
///   halves of one op answer the question differently and neither is a
///   guess.
///
/// A gradient reaching the **mask** is refused by name rather than dropped:
/// upstream's mask is `bool` and carries no `grad_fn`, so anything that got
/// there is a bug in the walk.
fn native_dropout_backward<'py>(
    py: Python<'py>,
    node: &Node,
    env: &Env,
    gouts: &[Option<Obj<'py>>],
    outs: &[Obj<'py>],
) -> PyResult<Rule<'py>> {
    const OP: &str = "aten.native_dropout.default";
    let ops = bind(py, node, env, &["input", "p", "train"])?;
    if gouts.len() > 1 && gouts[1].is_some() {
        return Err(crate::err::not_implemented(
            "torch._C tape: a gradient reached native_dropout's mask, which is a bool \
             record of which elements survived rather than a differentiable value",
        ));
    }
    let g = gouts
        .first()
        .and_then(|slot| slot.clone())
        .ok_or_else(|| missing(OP, "grad"))?;
    if outs.len() < 2 {
        return Err(crate::err::not_implemented(format!(
            "torch._C tape: native_dropout's gradient reads the mask this op returns, and \
             this node recorded {} result(s)",
            outs.len()
        )));
    }
    let p = opt_f64(&ops, 1, 0.5)?;
    // `bool? train` -- absent and `None` both mean `true`, which is what
    // `opt_bool`'s fallback says and what upstream's `!train.has_value() ||
    // *train` says.
    let train = opt_bool(&ops, 2, true)?;
    let scale = if !train {
        1.0
    } else if p == 1.0 {
        0.0
    } else {
        1.0 / (1.0 - p)
    };
    let masked = mul(py, g, outs[1].clone())?;
    let gi = muls(py, masked, scale)?;
    Ok((ops, vec![Some(gi), None, None]))
}

/// `_scaled_dot_product_flash_attention_for_cpu` -> gradients for q, k and v.
///
/// `docs/AUTOGRAD.md` §5.1 named this the one op on SmolLM2's backward path
/// that is a real CPU kernel with **no** Core ATen decomposition, and therefore
/// the one thing that would have to be hand-written. That is true of a
/// *kernel*; it is not true of a *rule*. The tape's backward runs outside a
/// capture region, so it may recompute the attention probabilities from the
/// saved q, k and v and differentiate the textbook formulation:
///
/// ```text
///   P  = softmax(scale * q k^T + mask)        out = P v
///   dv = P^T dout                             dP  = dout v^T
///   dS = P * (dP - rowsum(dP * P))
///   dq = scale * dS k                         dk  = scale * dS^T q
/// ```
///
/// Every op in that is one the shim already has. What it costs is a second
/// forward attention and the memory for one `[B, H, T, S]` probability matrix
/// per layer -- upstream's fused kernel exists to avoid exactly that, so this
/// is a correctness-for-memory trade and `docs/BACKWARD.md` §5 measures both
/// sides of it.
fn sdpa_backward<'py>(
    py: Python<'py>,
    node: &Node,
    env: &Env,
    gouts: &[Option<Obj<'py>>],
) -> PyResult<Rule<'py>> {
    const OP: &str = "aten._scaled_dot_product_flash_attention_for_cpu.default";
    let ops = bind(
        py,
        node,
        env,
        &["query", "key", "value", "dropout_p", "is_causal", "attn_mask", "scale"],
    )?;
    if gouts.len() > 1 && gouts[1].is_some() {
        return Err(crate::err::not_implemented(
            "torch._C tape: a gradient reached this op's logsumexp, which no forward in this \
             repository consumes and which has no rule here",
        ));
    }
    let g = gouts
        .first()
        .and_then(|slot| slot.clone())
        .ok_or_else(|| missing(OP, "grad"))?;

    let query = required(OP, &ops, 0, "query")?;
    let key = required(OP, &ops, 1, "key")?;
    let value = required(OP, &ops, 2, "value")?;
    let dropout = opt_f64(&ops, 3, 0.0)?;
    if dropout != 0.0 {
        return Err(crate::err::not_implemented(format!(
            "torch._C tape: this op's forward refuses dropout_p > 0 and its gradient would \
             need the draw that was never made (dropout_p={dropout})"
        )));
    }
    let is_causal = opt_bool(&ops, 4, false)?;
    let qshape = query.shape()?;
    let kshape = key.shape()?;
    if qshape.len() != 4 || kshape.len() != 4 {
        return Err(crate::err::not_implemented(format!(
            "torch._C tape: this op's gradient is written for rank-4 operands and query was \
             {qshape:?}"
        )));
    }
    let head_size = qshape[3];
    let scale = opt_f64(&ops, 6, 1.0 / (head_size as f64).sqrt())?;

    // Grouped-query attention: the kernel repeats key and value up to the
    // query's head count before anything else touches them, so the gradient
    // has to be summed back down over each group. Doing it here rather than
    // pretending the head counts match is the difference between a `[3, ...]`
    // gradient and a wrong `[9, ...]` one.
    let groups = qshape[1] / kshape[1].max(1);
    let repeat = |py: Python<'py>, t: Obj<'py>| -> PyResult<Obj<'py>> {
        if groups <= 1 {
            return Ok(t);
        }
        let shape = dims(&t)?;
        let lifted = reshape(py, t, &[shape[0], shape[1], 1, shape[2], shape[3]])?;
        let grown = expand_to(py, lifted, &[shape[0], shape[1], groups, shape[2], shape[3]])?;
        reshape(py, grown, &[shape[0], shape[1] * groups, shape[2], shape[3]])
    };
    let fold = |py: Python<'py>, t: Obj<'py>, want: &[usize]| -> PyResult<Obj<'py>> {
        if groups <= 1 {
            return Ok(t);
        }
        let shape = dims(&t)?;
        let split = reshape(py, t, &[shape[0], want[1], groups, shape[2], shape[3]])?;
        let summed = sum_dims(py, split, &[2], false)?;
        reshape(py, summed, want)
    };

    let k_full = repeat(py, key.value.clone())?;
    let v_full = repeat(py, value.value.clone())?;

    let mut scores = muls(py, matmul(py, query.value.clone(), mt(py, k_full.clone())?)?, scale)?;
    if let Some(mask) = ops.get(5).and_then(|s| s.as_ref()) {
        if !mask.value.is_none() {
            scores = add(py, scores, mask.value.clone())?;
        }
    }
    if is_causal {
        // Upper-left aligned, which is what the forward measured (aten.rs) --
        // row t attends keys 0..=t even when the key sequence is longer.
        let rows = qshape[2] as i64;
        let cols = dims(&scores)?[3] as i64;
        let ones = call(
            py,
            "aten.ones.default",
            vec![ints(py, &[rows, cols])?],
        )?;
        let keep = call(py, "aten.tril.default", vec![ones, 0i64.into_bound_py_any(py)?])?;
        let blocked = call(py, "aten.eq.Scalar", vec![keep, scalar(py, 0.0)?])?;
        scores = call(
            py,
            "aten.masked_fill.Scalar",
            vec![scores, blocked, f64::NEG_INFINITY.into_bound_py_any(py)?],
        )?;
    }
    let probs = call(
        py,
        "aten._safe_softmax.default",
        vec![scores, (-1i64).into_bound_py_any(py)?],
    )?;

    let gv = if value.target.is_some() {
        let product = matmul(py, mt(py, probs.clone())?, g.clone())?;
        Some(fold(py, product, &value.shape()?)?)
    } else {
        None
    };
    let dp = matmul(py, g, mt(py, v_full)?)?;
    let weighted = sum_dims(py, mul(py, dp.clone(), probs.clone())?, &[-1], true)?;
    let ds = mul(py, probs, sub(py, dp, weighted)?)?;
    let gq = if query.target.is_some() {
        Some(muls(py, matmul(py, ds.clone(), k_full)?, scale)?)
    } else {
        None
    };
    let gk = if key.target.is_some() {
        let product = matmul(py, mt(py, ds)?, query.value.clone())?;
        Some(fold(py, muls(py, product, scale)?, &kshape)?)
    } else {
        None
    };
    Ok((ops, vec![gq, gk, gv]))
}

// ---------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------

/// Which constants a gradient is wanted for.
///
/// An integer constant is dropped whichever way it was named. A gradient of an
/// integer is not a thing upstream has either -- `requires_grad_` refuses on a
/// non-floating tensor there -- and admitting one here would put token ids on
/// the gradient path, which is how the first run of this found itself needing a
/// derivative for the `F.pad` that builds `transformers`' shifted labels.
fn wrt_set(trace: &PyCaptureTrace, wrt: Option<&Bound<'_, PyAny>>) -> PyResult<HashSet<usize>> {
    let differentiable = |index: &usize| trace.consts[*index].dtype.is_floating_point();
    match wrt.filter(|value| !value.is_none()) {
        None => Ok((0..trace.consts.len()).filter(differentiable).collect()),
        Some(value) => {
            let indices: Vec<usize> = value.extract()?;
            for index in &indices {
                if *index >= trace.consts.len() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "torch._C tape: wrt_constants names constant {index} and this trace \
                         has {}",
                        trace.consts.len()
                    )));
                }
            }
            Ok(indices.into_iter().filter(differentiable).collect())
        }
    }
}

fn walk_refs(arg: &Arg, sink: &mut impl FnMut(Ref)) {
    match arg {
        Arg::Value(reference) => sink(*reference),
        Arg::List(items) | Arg::Tuple(items) => {
            for item in items {
                walk_refs(item, sink);
            }
        }
        Arg::Literal(_) => {}
    }
}

/// Which nodes lie on a path from something a gradient is wanted for.
///
/// Without this the walk would try to differentiate the rotary embedding's
/// `arange`, because a tape has no `requires_grad` of its own and every value
/// looks alike from inside. Upstream gets the same pruning from the flag; here
/// it comes from the declaration of what the caller wants gradients *for*,
/// which is the same information one step earlier.
fn reachable(trace: &PyCaptureTrace, wrt: &HashSet<usize>) -> Vec<bool> {
    let mut needed = vec![false; trace.nodes.len()];
    for (index, node) in trace.nodes.iter().enumerate() {
        let mut hit = false;
        {
            let mut sink = |reference: Ref| {
                hit = hit
                    || match reference {
                        Ref::Input(input) => trace.inputs[input].dtype.is_floating_point(),
                        Ref::Const(constant) => wrt.contains(&constant),
                        Ref::Node { node, .. } => needed[node],
                    };
            };
            for arg in &node.args {
                walk_refs(arg, &mut sink);
            }
            for (_, arg) in &node.kwargs {
                walk_refs(arg, &mut sink);
            }
        }
        needed[index] = hit;
    }
    needed
}

fn wanted(trace: &PyCaptureTrace, reference: Ref, wrt: &HashSet<usize>, needed: &[bool]) -> bool {
    match reference {
        Ref::Input(index) => trace.inputs[index].dtype.is_floating_point(),
        Ref::Const(index) => wrt.contains(&index),
        Ref::Node { node, .. } => needed[node],
    }
}

pub fn backward<'py>(
    py: Python<'py>,
    trace: &PyCaptureTrace,
    inputs: &Bound<'py, PyAny>,
    grad_outputs: Option<&Bound<'py, PyAny>>,
    wrt_constants: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    if trace.outputs.is_empty() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "torch._C tape: this trace declared no outputs, so there is nothing to \
             differentiate",
        ));
    }
    let env = trace.run(py, inputs)?;
    backward_in(py, trace, &env, grad_outputs, wrt_constants)
}

/// The reverse walk, over a record and an `Env` that **already holds the
/// values**.
///
/// Split out of `backward` for W8 (`docs/BACKWARD7.md`). `backward` is this
/// plus `trace.run()` -- a *replay* that reconstructs the forward, which a
/// `CaptureTrace` needs because it drops its keepalives at `_capture_end`. An
/// eager tape does not need it: it never stopped holding the values, so its
/// `Env` is the objects the program actually computed, and handing them
/// straight to this function is the whole of the reuse `docs/BACKWARD3.md` §4
/// and `docs/BACKWARD5.md` §4 argued for. Not one derivative rule below knows
/// which of the two called it, and `derivative()`, `wrt_set()`, `reachable()`
/// and `wanted()` are shared verbatim.
pub(crate) fn backward_in<'py>(
    py: Python<'py>,
    trace: &PyCaptureTrace,
    env: &Env,
    grad_outputs: Option<&Bound<'py, PyAny>>,
    wrt_constants: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let wrt = wrt_set(trace, wrt_constants)?;
    let needed = reachable(trace, &wrt);

    let mut grads: HashMap<Ref, Py<PyAny>> = HashMap::new();
    let given_seeds = grad_outputs.filter(|value| !value.is_none());
    let seeds: Vec<Obj<'py>> = match given_seeds {
        None => {
            let mut built = Vec::with_capacity(trace.outputs.len());
            for reference in &trace.outputs {
                let value = env.get(py, *reference)?;
                if !dims(&value)?.is_empty() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "torch._C tape: backward() without grad_outputs seeds a one, which \
                         only means anything for a scalar output; this trace's output is not \
                         a scalar",
                    ));
                }
                built.push(call(py, "aten.ones_like.default", vec![value])?);
            }
            built
        }
        Some(given) => {
            let items: Vec<Obj<'py>> = match given.extract::<Vec<Obj<'py>>>() {
                Ok(list) => list,
                Err(_) => vec![given.clone()],
            };
            if items.len() != trace.outputs.len() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "torch._C tape: this trace has {} output(s) and backward() was given {} \
                     gradient(s)",
                    trace.outputs.len(),
                    items.len()
                )));
            }
            items
        }
    };
    for (reference, seed) in trace.outputs.iter().zip(seeds.into_iter()) {
        accumulate(py, &mut grads, *reference, seed)?;
    }

    for index in (0..trace.nodes.len()).rev() {
        if !needed[index] {
            continue;
        }
        let node = &trace.nodes[index];
        let mut gouts = Vec::with_capacity(node.outputs.len());
        let mut any = false;
        for slot in 0..node.outputs.len() {
            match grads.get(&Ref::Node { node: index, output: slot }) {
                Some(found) => {
                    any = true;
                    gouts.push(Some(found.bind(py).clone()));
                }
                None => gouts.push(None),
            }
        }
        if !any {
            continue;
        }
        // A gradient arriving at a non-tensor result is a bug in the walk
        // rather than a missing rule, and saying so with the op's name is the
        // difference between a five-minute and an afternoon diagnosis.
        for (slot, kind) in node.outputs.iter().enumerate() {
            if matches!(kind, Slot::Other) && gouts[slot].is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "torch._C tape: a gradient reached result {slot} of {}, which the record \
                     says is not a tensor",
                    node.op
                )));
            }
        }
        let outs: Vec<Obj<'py>> = env
            .nodes
            .get(index)
            .map(|slots| slots.iter().map(|s| s.bind(py).clone()).collect())
            .unwrap_or_default();

        let (operands, contributions) = derivative(py, node, env, &gouts, &outs)?;
        for (operand, contribution) in operands.iter().zip(contributions.into_iter()) {
            let (Some(operand), Some(contribution)) = (operand, contribution) else {
                continue;
            };
            let Some(target) = operand.target else {
                continue;
            };
            if !wanted(trace, target, &wrt, &needed) {
                continue;
            }
            accumulate(py, &mut grads, target, contribution)?;
        }
    }

    let out = PyDict::new(py);
    let mut input_grads: Vec<Obj<'py>> = Vec::with_capacity(trace.inputs.len());
    for index in 0..trace.inputs.len() {
        input_grads.push(match grads.get(&Ref::Input(index)) {
            Some(found) => found.bind(py).clone(),
            None => py.None().into_bound(py),
        });
    }
    let mut const_grads: Vec<Obj<'py>> = Vec::with_capacity(trace.consts.len());
    for index in 0..trace.consts.len() {
        const_grads.push(match grads.get(&Ref::Const(index)) {
            Some(found) => found.bind(py).clone(),
            None => py.None().into_bound(py),
        });
    }
    out.set_item("inputs", PyList::new(py, input_grads)?)?;
    out.set_item("constants", PyList::new(py, const_grads)?)?;
    Ok(out)
}

fn accumulate<'py>(
    py: Python<'py>,
    grads: &mut HashMap<Ref, Py<PyAny>>,
    reference: Ref,
    contribution: Obj<'py>,
) -> PyResult<()> {
    match grads.remove(&reference) {
        None => {
            grads.insert(reference, contribution.unbind());
        }
        Some(previous) => {
            let summed = add(py, previous.into_bound(py), contribution)?;
            grads.insert(reference, summed.unbind());
        }
    }
    Ok(())
}

pub fn differentiable<'py>(
    py: Python<'py>,
    trace: &PyCaptureTrace,
    wrt_constants: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let wrt = wrt_set(trace, wrt_constants)?;
    let needed = reachable(trace, &wrt);
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (index, node) in trace.nodes.iter().enumerate() {
        if needed[index] {
            *counts.entry(node.op.as_str()).or_insert(0) += 1;
        }
    }
    let mut covered: Vec<(&str, usize)> = Vec::new();
    let mut absent: Vec<(&str, usize)> = Vec::new();
    for (op, count) in counts {
        if has_rule(op) {
            covered.push((op, count));
        } else {
            absent.push((op, count));
        }
    }
    covered.sort();
    absent.sort();

    let out = PyDict::new(py);
    let to_dict = |pairs: Vec<(&str, usize)>| -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (op, count) in pairs {
            dict.set_item(op, count)?;
        }
        Ok(dict)
    };
    out.set_item("nodes", trace.nodes.len())?;
    out.set_item("nodes_on_a_gradient_path", needed.iter().filter(|n| **n).count())?;
    out.set_item("covered", to_dict(covered)?)?;
    out.set_item("missing", to_dict(absent)?)?;
    Ok(out)
}

/// The rule table, readable from Python.
///
/// `pytests/test_shim.py` asserts that its own gradient-case list equals this,
/// which is what keeps the table and the cases from drifting -- the failure
/// mode docs/AUDIT.md found six times is a second list nobody re-reads.
#[pyfunction]
#[pyo3(name = "_tape_rules")]
pub fn tape_rules(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    PyList::new(py, RULE_OPS)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tape_rules, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rule_table_is_sorted_and_has_no_duplicates() {
        // Sorted so that a reader can find a name, and unique so that
        // `differentiable()`'s counts cannot double.
        let mut sorted = RULE_OPS.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.as_slice(), RULE_OPS, "RULE_OPS must be sorted and unique");
    }

    #[test]
    fn every_rule_names_an_aten_overload() {
        for op in RULE_OPS {
            assert!(op.starts_with("aten."), "{op}");
            assert_eq!(op.split('.').count(), 3, "{op} should be aten.<name>.<overload>");
        }
    }
}
