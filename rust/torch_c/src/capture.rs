//! Graph capture at the single door.
//!
//! DESIGN.md §11.1 named the problem this is the first piece of: an NPU is not
//! an eager device. ANE, NNAPI and QNN do not take one op at a time -- they
//! take a whole graph, compile it ahead of execution, and run it. Porting
//! kernels one at a time never reaches them, however many kernels are ported.
//! What reaches them is handing a *region* of the program over at once.
//!
//! The same paragraph named why this shim is in an unusually good position to
//! do that: **every op goes through `_aten_dispatch`.** Upstream has to trace
//! with `__torch_dispatch__` modes, `torch.export`'s fake-tensor machinery and
//! a dynamo frame evaluator because upstream's dispatcher has many doors.
//! Here there is one, so the recorder is one branch at one line, and no kernel
//! can escape it by being written later.
//!
//! What this module is:
//!
//! | | |
//! |---|---|
//! | record | ops in order, with the shape/dtype/device of every operand and result |
//! | guard | the conditions under which the record may be replayed at all |
//! | replay | run the record with new inputs and get eager's answer |
//!
//! What it is deliberately not, and refuses by name rather than approximating
//! (docs/CAPTURE.md §4): control flow, in-place mutation, dynamic shapes, and
//! randomness. §6 of DESIGN.md is the rule being followed -- a refusal that
//! says its own name is worth more than a silent wrong answer, and a capture
//! layer is a place where silent wrong answers are *cheap to produce*, because
//! the replayed graph looks exactly as plausible as a correct one.
//!
//! **Recording is an observation, and an observation must not change what it
//! observes.** So an unsupported op does not raise where it happens: it
//! poisons the recording with a reason and lets the program run to completion
//! on the eager path it was already on. The refusal arrives at
//! `_capture_end`, which is where the *claim* of a capture is made.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PySet, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::dtype::TorchDType;
use crate::tensor::PyTensorBase;

// ---------------------------------------------------------------------------
// The off switch
// ---------------------------------------------------------------------------

/// Whether *any* thread is recording.
///
/// The whole cost of this module on the ordinary path is one relaxed load of
/// this flag and a branch that is never taken. That is not an aesthetic
/// preference: docs/DEVICE_ABS.md §6 measured a `String` allocation per tensor
/// argument at +78 ns on `add.Tensor`, a door that costs 346 ns in total, so
/// anything added here is measured against a small number. docs/CAPTURE.md §7
/// has the A/B for this flag.
///
/// Global rather than thread-local because reading a `thread_local!` costs a
/// TLS lookup, and the recorder itself is thread-local anyway -- a second
/// thread that takes the branch finds no recorder and falls straight back out.
static CAPTURING: AtomicBool = AtomicBool::new(false);

thread_local! {
    static RECORDER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

#[inline(always)]
pub fn is_active() -> bool {
    CAPTURING.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------

/// Where a value in the trace came from.
///
/// The three cases are FX's three: a `placeholder`, a lifted `get_attr`
/// constant, and the result of an earlier `call_function`. Nothing else can
/// appear as a tensor operand in a straight-line segment, which is what makes
/// the segment straight.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Ref {
    Input(usize),
    Const(usize),
    Node { node: usize, output: usize },
}

/// One argument of one recorded op.
///
/// `Literal` holds the Python object itself rather than a re-encoded copy.
/// Re-encoding would mean writing a second converter for dtypes, devices,
/// memory formats and `None`, which is a second place for the two to disagree;
/// holding the object means replay hands the kernel the identical value it saw.
/// These are scalars, dtype singletons and small integer lists -- immutable in
/// practice, and the ones that are not are exactly the ones a trace must not
/// be a function of.
pub(crate) enum Arg {
    Value(Ref),
    Literal(Py<PyAny>),
    List(Vec<Arg>),
    Tuple(Vec<Arg>),
}

/// Shape, dtype and device -- everything a trace knows about a tensor, and
/// everything a guard is allowed to check.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct TensorMeta {
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: TorchDType,
    pub(crate) device: String,
}

impl TensorMeta {
    fn of(tensor: &Bound<'_, PyTensorBase>) -> Self {
        let borrowed = tensor.borrow();
        Self {
            shape: borrowed.dims().to_vec(),
            dtype: borrowed.tag(),
            device: borrowed.device_label().__str__(),
        }
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", self.shape.clone())?;
        d.set_item("dtype", format!("torch.{}", self.dtype.name()))?;
        d.set_item("device", &self.device)?;
        Ok(d)
    }

    fn to_slot_dict<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("index", index)?;
        d.set_item("shape", self.shape.clone())?;
        d.set_item("dtype", format!("torch.{}", self.dtype.name()))?;
        d.set_item("device", &self.device)?;
        Ok(d)
    }
}

/// One result slot of one recorded op.
///
/// `Other` is a result that is not a tensor. It is recorded -- the op was
/// called and the record has to say so -- but no `Ref` points at it, so it can
/// never become an operand. See `refusal_for`: the only such ops that survive
/// recording are the ones whose answer is a function of *metadata*, and
/// metadata is what the guards pin.
pub(crate) enum Slot {
    Tensor(TensorMeta),
    Other,
}

pub(crate) struct Node {
    pub(crate) op: String,
    pub(crate) args: Vec<Arg>,
    pub(crate) kwargs: Vec<(String, Arg)>,
    pub(crate) outputs: Vec<Slot>,
    /// Whether the op returned a sequence. Replay has to index the result the
    /// same way the recorded program did, and "one tensor" and "a list of one
    /// tensor" are different returns.
    pub(crate) sequence: bool,
}

struct Recorder {
    nodes: Vec<Node>,
    inputs: Vec<TensorMeta>,
    consts: Vec<TensorMeta>,
    /// Object address -> where that value came from. Every address in here
    /// belongs to an object `keepalive` (or `const_objects`) holds a strong
    /// reference to, so an address can never be reused under us while the
    /// recording is open. That is the price of identity: a trace of a long
    /// model holds its activations until `_capture_end`. docs/CAPTURE.md §6.
    known: HashMap<usize, Ref>,
    keepalive: Vec<Py<PyAny>>,
    const_objects: Vec<Py<PyAny>>,
    poisoned: Option<String>,
}

impl Recorder {
    fn poison(&mut self, reason: String) {
        if self.poisoned.is_none() {
            self.poisoned = Some(reason);
        }
    }
}

// ---------------------------------------------------------------------------
// What cannot be captured, and the name it is refused under
// ---------------------------------------------------------------------------

/// `aten.<name>_.<overload>` -- torch's own spelling for "this writes into its
/// receiver".
///
/// Refusing all of them is what buys the aliasing exemption. In-place aliasing
/// is out of scope, and rather than model storage sharing, capture removes the
/// only way sharing is observable: with no mutation in the segment, whether
/// two recorded values share bytes cannot change any answer. The trace is
/// single-assignment by construction rather than by assumption.
fn is_mutating(op: &str) -> bool {
    op.rsplit_once('.').is_some_and(|(head, _)| head.ends_with('_'))
}

/// Ops that mutate an argument and whose **name does not say so**.
///
/// `is_mutating` above is torch's own convention, and it is right for every op
/// that follows it -- which is why nothing here needed a list until now.
/// `aten::native_batch_norm` breaks it: its schema declares no alias on any
/// argument (measured on 2.13.0, every `alias_info` is `None`) and it writes
/// `running_mean`/`running_var` in place anyway when `training=True`. Upstream
/// knows -- `_native_batch_norm_legit` exists beside it carrying the
/// `Tensor(a!)` annotations the functionaliser needs, which is the only reason
/// the discrepancy is survivable there.
///
/// It would not have been survivable here. Without this, a training-mode
/// BatchNorm would have been recorded as a pure node and replaying the trace
/// would advance the running statistics a second time: the trace would not be
/// single-assignment, which is the one property this module exists to keep.
///
/// **The judgement is per call, not per name**, and that is why it lives here
/// rather than in a widened `is_mutating`. Measured (docs/DEMAND1.md §1.3):
/// with `training=False` this op touches nothing -- the running statistics come
/// back byte-identical and `save_mean`/`save_invstd` are empty. Eval is the
/// mode a captured inference graph is in and the mode every BatchNorm CNN
/// backbone runs in here, so refusing the name outright would have made all of
/// them uncapturable to buy nothing. A call with no running statistics at all
/// (`nn.BatchNorm2d(track_running_stats=False)`) has nothing to mutate in
/// either mode and is likewise recordable.
const MUTATES_WITHOUT_UNDERSCORE: &[&str] = &["aten.native_batch_norm.default"];

/// Whether this *particular* call of a `MUTATES_WITHOUT_UNDERSCORE` op writes.
///
/// Reads the same argument positions the kernel does, positionally or by
/// keyword, because `bootstrap.py`'s `batch_norm` composite passes them
/// positionally and a direct `torch.ops.aten` call need not.
fn mutates_this_call(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> bool {
    if op != "aten.native_batch_norm.default" {
        // The list has one member; a second one arriving without its own arm
        // here should be refused outright rather than waved through.
        return true;
    }
    let arg = |index: usize, name: &str| -> Option<Bound<'_, PyAny>> {
        if let Ok(value) = args.get_item(index) {
            return Some(value);
        }
        kwargs.and_then(|kw| kw.get_item(name).ok().flatten())
    };
    let training = arg(5, "training").is_some_and(|v| v.is_truthy().unwrap_or(true));
    let has_stats = arg(3, "running_mean").is_some_and(|v| !v.is_none());
    training && has_stats
}

/// Ops that consume the generator. Recorded traces are checked by replaying
/// them and comparing against eager, and an op that legitimately differs on
/// every call makes that check unable to fail -- which is worse than not
/// having it. There is also no story yet for handing a seed to a delegate.
const RANDOM: &[&str] = &[
    "aten.multinomial.default",
    "aten.randint.default",
    "aten.randint.low",
    "aten.randperm.default",
];

/// Ops that leave the tensor world entirely, taking a value the guards say
/// nothing about with them.
///
/// This is the runtime half of the rule DESIGN.md §6's static scan states:
/// branching on a tensor value is untraceable. `t.item()`, `bool(t)`,
/// `float(t)` and `int(t)` are all spelled `aten._local_scalar_dense.default`
/// in `bootstrap.py`, so one name covers all four -- and a Python `if` taken
/// on the result is a decision that is *not in the record*. The recorded
/// straight line would be one arm of a branch, replayed unconditionally.
const HOST_READS: &[&str] = &["aten._local_scalar_dense.default"];

/// Ops whose non-tensor result is a function of metadata alone.
///
/// The line this module draws is metadata versus data. `is_floating_point`
/// reads the dtype, and the dtype is pinned by a guard, so its answer is the
/// same for every input the trace admits and burning it in is sound.
/// `_local_scalar_dense` reads the bytes, which no guard constrains. Keeping
/// the allowlist explicit is what stops it from growing by accident.
const METADATA_ONLY: &[&str] = &["aten.is_floating_point.default"];

const DATA_DEPENDENT_SHAPE: &[&str] = &[
    "aten.nonzero.default",
    "aten.where.default",
];

fn refusal_for(op: &str) -> Option<String> {
    if DATA_DEPENDENT_SHAPE.contains(&op) {
        return Some(format!(
            "{op} produces an output shape that depends on tensor values; a trace whose node output shape is not a function of its inputs cannot be recorded, because a replay with different inputs might produce a different shape."
        ));
    }

    if HOST_READS.contains(&op) {
        return Some(format!(
            "{op} reads a tensor value onto the host; a Python branch taken on \
             that value is not in the record, so the trace would be one arm of \
             a branch replayed unconditionally"
        ));
    }
    if is_mutating(op) {
        return Some(format!(
            "{op} writes in place; capture refuses mutation so that aliasing \
             cannot be observed, which is what keeps a trace single-assignment"
        ));
    }
    if RANDOM.contains(&op) {
        return Some(format!(
            "{op} draws random numbers; a replay that legitimately differs from \
             eager would make the eager comparison unable to fail, and there is \
             no way yet to hand a delegate a seed"
        ));
    }
    None
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

/// The recorder's view of one dispatch. Called from `aten_dispatch` **after**
/// `promote`, so the identity registered is the object Python will actually
/// hold and pass to the next op.
///
/// Errors are not returned: a failure to record is a failure of the capture,
/// not of the program. Everything that goes wrong lands in `poisoned` and
/// surfaces at `_capture_end`.
pub fn record(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    out: &Py<PyAny>,
) {
    RECORDER.with(|cell| {
        let Ok(mut slot) = cell.try_borrow_mut() else {
            return;
        };
        let Some(rec) = slot.as_mut() else {
            return;
        };
        if rec.poisoned.is_some() {
            return;
        }
        if let Some(reason) = refusal_for(op) {
            rec.poison(reason);
            return;
        }
        // The arg-aware half of the same question, for the one op whose name
        // does not carry torch's mutation convention. See
        // `MUTATES_WITHOUT_UNDERSCORE`.
        if MUTATES_WITHOUT_UNDERSCORE.contains(&op) && mutates_this_call(op, args, kwargs) {
            rec.poison(format!(
                "{op} writes running_mean/running_var in place on this call \
                 (training=True with running statistics supplied), even though its \
                 schema declares no alias on any argument; capture refuses mutation so \
                 that aliasing cannot be observed, which is what keeps a trace \
                 single-assignment. The same op in eval mode touches nothing and IS \
                 capturable"
            ));
            return;
        }

        let mut positional = Vec::with_capacity(args.len());
        for value in args.iter() {
            match rec.arg_of(&value) {
                Ok(arg) => positional.push(arg),
                Err(reason) => return rec.poison(format!("{op}: {reason}")),
            }
        }
        let mut named = Vec::new();
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.iter() {
                let Ok(name) = key.extract::<String>() else {
                    return rec.poison(format!("{op}: keyword argument name is not a string"));
                };
                match rec.arg_of(&value) {
                    Ok(arg) => named.push((name, arg)),
                    Err(reason) => return rec.poison(format!("{op}: {reason}")),
                }
            }
        }

        let node_index = rec.nodes.len();
        let bound = out.bind(py);
        let (slots, sequence) = match sequence_items(bound) {
            Some(items) => (items, true),
            None => (vec![bound.clone()], false),
        };

        let mut outputs = Vec::with_capacity(slots.len());
        for (position, item) in slots.iter().enumerate() {
            match item.cast::<PyTensorBase>() {
                Ok(tensor) => {
                    outputs.push(Slot::Tensor(TensorMeta::of(tensor)));
                    rec.known
                        .insert(item.as_ptr() as usize, Ref::Node { node: node_index, output: position });
                    rec.keepalive.push(item.clone().unbind());
                }
                Err(_) => {
                    if !METADATA_ONLY.contains(&op) {
                        return rec.poison(format!(
                            "{op} returned a value that is not a tensor, and it is not on \
                             the metadata-only allowlist; capture cannot tell whether that \
                             value depends on tensor *data*, which no guard constrains"
                        ));
                    }
                    outputs.push(Slot::Other);
                }
            }
        }

        rec.nodes.push(Node {
            op: op.to_string(),
            args: positional,
            kwargs: named,
            outputs,
            sequence,
        });
    });
}

fn sequence_items<'py>(value: &Bound<'py, PyAny>) -> Option<Vec<Bound<'py, PyAny>>> {
    if let Ok(list) = value.cast::<PyList>() {
        return Some(list.iter().collect());
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return Some(tuple.iter().collect());
    }
    None
}

impl Recorder {
    /// One dispatched argument, turned into something the record can hold.
    ///
    /// A tensor that has not been seen before is a **constant**: a weight, a
    /// buffer, a mask built before the region began. It is held by reference
    /// and burned into the graph, which is the same split
    /// `ExportedProgram.graph_signature` makes between user inputs and lifted
    /// parameters.
    fn arg_of(&mut self, value: &Bound<'_, PyAny>) -> Result<Arg, String> {
        if let Ok(tensor) = value.cast::<PyTensorBase>() {
            let address = value.as_ptr() as usize;
            if let Some(known) = self.known.get(&address) {
                return Ok(Arg::Value(*known));
            }
            let index = self.consts.len();
            self.consts.push(TensorMeta::of(tensor));
            self.const_objects.push(value.clone().unbind());
            self.known.insert(address, Ref::Const(index));
            return Ok(Arg::Value(Ref::Const(index)));
        }
        if let Ok(list) = value.cast::<PyList>() {
            let mut items = Vec::with_capacity(list.len());
            for item in list.iter() {
                items.push(self.arg_of(&item)?);
            }
            return Ok(Arg::List(items));
        }
        if let Ok(tuple) = value.cast::<PyTuple>() {
            let mut items = Vec::with_capacity(tuple.len());
            for item in tuple.iter() {
                items.push(self.arg_of(&item)?);
            }
            return Ok(Arg::Tuple(items));
        }
        // A container capture cannot walk is a container that may be hiding a
        // tensor. No aten schema takes one today, so refusing costs nothing and
        // keeps the "every tensor operand is a `Ref`" invariant true rather
        // than probably true.
        if value.cast::<PyDict>().is_ok() || value.cast::<PySet>().is_ok() {
            return Err(format!(
                "argument is a {}, which capture does not walk; a tensor inside it \
                 would be missed from the graph",
                value
                    .get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_default()
            ));
        }
        Ok(Arg::Literal(value.clone().unbind()))
    }
}

// ---------------------------------------------------------------------------
// The value reference, as Python sees it
// ---------------------------------------------------------------------------

/// A reference to a value in a trace: `%in0`, `%c1`, `%3`, `%3#1`.
///
/// Spelled as a type rather than a tuple so that a reference can never be
/// mistaken for a literal argument -- a graph in which `("node", 0)` might be
/// either a reference or a genuine tuple argument is one no reader can lower.
// `from_py_object` is opted into explicitly rather than inherited from `Clone`:
// pyo3 0.29 deprecates the implicit version, and the same opt-in is what
// `PyTensorBase` carries.
#[pyclass(name = "CaptureValue", module = "torch._C", frozen, eq, hash, from_py_object)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PyCaptureValue {
    kind: String,
    index: usize,
    output: usize,
}

#[pymethods]
impl PyCaptureValue {
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    #[getter]
    fn index(&self) -> usize {
        self.index
    }

    #[getter]
    fn output(&self) -> usize {
        self.output
    }

    fn __repr__(&self) -> String {
        match self.kind.as_str() {
            "input" => format!("%in{}", self.index),
            "const" => format!("%c{}", self.index),
            _ if self.output == 0 => format!("%{}", self.index),
            _ => format!("%{}#{}", self.index, self.output),
        }
    }
}

impl PyCaptureValue {
    fn of(reference: Ref) -> Self {
        match reference {
            Ref::Input(index) => Self { kind: "input".into(), index, output: 0 },
            Ref::Const(index) => Self { kind: "const".into(), index, output: 0 },
            Ref::Node { node, output } => Self { kind: "node".into(), index: node, output },
        }
    }
}

/// Build one by hand. Exists so tests can state the graph they expect instead
/// of reading it back out of the object under test.
#[pyfunction]
#[pyo3(name = "_capture_value", signature = (kind, index, output = 0))]
pub fn capture_value(kind: &str, index: usize, output: usize) -> PyResult<PyCaptureValue> {
    if !matches!(kind, "input" | "const" | "node") {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "capture value kind must be 'input', 'const' or 'node', got {kind:?}"
        )));
    }
    Ok(PyCaptureValue { kind: kind.to_string(), index, output })
}

// ---------------------------------------------------------------------------
// The trace
// ---------------------------------------------------------------------------

/// A recorded straight-line segment, and the conditions it is valid under.
#[pyclass(name = "CaptureTrace", module = "torch._C", frozen)]
pub struct PyCaptureTrace {
    pub(crate) nodes: Vec<Node>,
    pub(crate) inputs: Vec<TensorMeta>,
    pub(crate) consts: Vec<TensorMeta>,
    pub(crate) const_objects: Vec<Py<PyAny>>,
    pub(crate) outputs: Vec<Ref>,
}

impl Arg {
    pub(crate) fn to_python<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            Arg::Value(reference) => PyCaptureValue::of(*reference).into_bound_py_any(py)?,
            Arg::Literal(object) => object.bind(py).clone(),
            Arg::List(items) => {
                let built: PyResult<Vec<_>> = items.iter().map(|a| a.to_python(py)).collect();
                PyList::new(py, built?)?.into_any()
            }
            Arg::Tuple(items) => {
                let built: PyResult<Vec<_>> = items.iter().map(|a| a.to_python(py)).collect();
                PyTuple::new(py, built?)?.into_any()
            }
        })
    }

    /// The same argument with every `Ref` replaced by the live value replay has
    /// computed for it.
    pub(crate) fn materialise<'py>(&self, py: Python<'py>, env: &Env) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            Arg::Value(reference) => env.get(py, *reference)?,
            Arg::Literal(object) => object.bind(py).clone(),
            Arg::List(items) => {
                let built: PyResult<Vec<_>> = items.iter().map(|a| a.materialise(py, env)).collect();
                PyList::new(py, built?)?.into_any()
            }
            Arg::Tuple(items) => {
                let built: PyResult<Vec<_>> = items.iter().map(|a| a.materialise(py, env)).collect();
                PyTuple::new(py, built?)?.into_any()
            }
        })
    }
}

/// Live values during a replay, indexed the way `Ref` indexes them.
pub(crate) struct Env {
    pub(crate) inputs: Vec<Py<PyAny>>,
    pub(crate) consts: Vec<Py<PyAny>>,
    pub(crate) nodes: Vec<Vec<Py<PyAny>>>,
}

impl Env {
    pub(crate) fn get<'py>(&self, py: Python<'py>, reference: Ref) -> PyResult<Bound<'py, PyAny>> {
        let missing = || {
            pyo3::exceptions::PyRuntimeError::new_err(
                "torch._C capture: replay reached a value that had not been produced yet",
            )
        };
        Ok(match reference {
            Ref::Input(index) => self.inputs.get(index).ok_or_else(missing)?.bind(py).clone(),
            Ref::Const(index) => self.consts.get(index).ok_or_else(missing)?.bind(py).clone(),
            Ref::Node { node, output } => self
                .nodes
                .get(node)
                .and_then(|slots| slots.get(output))
                .ok_or_else(missing)?
                .bind(py)
                .clone(),
        })
    }
}

#[pymethods]
impl PyCaptureTrace {
    fn __len__(&self) -> usize {
        self.nodes.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "<CaptureTrace {} nodes, {} inputs, {} constants, {} outputs>",
            self.nodes.len(),
            self.inputs.len(),
            self.consts.len(),
            self.outputs.len()
        )
    }

    /// What every input must look like for this trace to mean anything.
    ///
    /// Exact shape, exact dtype, exact device -- no ranges and no symbols.
    /// Every intermediate shape in the record was *computed from* these, so a
    /// guard that admitted a different one would make each recorded output
    /// shape a false statement. Dynamic shapes are the named gap
    /// (docs/CAPTURE.md §4), and this is what naming it has to mean.
    #[getter]
    fn guards<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut out = Vec::with_capacity(self.inputs.len());
        for (index, meta) in self.inputs.iter().enumerate() {
            out.push(meta.to_slot_dict(py, index)?);
        }
        PyList::new(py, out)
    }

    /// Tensors the region read but did not receive: weights, buffers, masks.
    /// Held by reference and burned in -- `ExportedProgram`'s lifted
    /// parameters, and the same limitation, that swapping them means capturing
    /// again.
    #[getter]
    fn constants<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut out = Vec::with_capacity(self.consts.len());
        for (index, meta) in self.consts.iter().enumerate() {
            out.push(meta.to_slot_dict(py, index)?);
        }
        PyList::new(py, out)
    }

    /// The constant tensors themselves, in the order `constants` describes
    /// them.
    ///
    /// `constants` is metadata, which is all a *reader* of the trace needs.
    /// Anything that rewrites the trace needs the objects: a pass that lowers
    /// this record to another dialect has to carry the burned-in weights
    /// across, and a Python-side pass cannot get at them through the metadata.
    /// So this getter exists for exactly one caller shape -- a rewrite that
    /// produces a new trace -- and hands out the same references replay uses,
    /// not copies, because a copy would silently decouple the two records from
    /// the weights and from each other.
    #[getter]
    fn constant_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.const_objects.iter().map(|c| c.clone_ref(py)))
    }

    #[getter]
    fn nodes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut out = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let dict = PyDict::new(py);
            dict.set_item("op", &node.op)?;
            let args: PyResult<Vec<_>> = node.args.iter().map(|a| a.to_python(py)).collect();
            dict.set_item("args", PyList::new(py, args?)?)?;
            let kwargs = PyDict::new(py);
            for (name, arg) in &node.kwargs {
                kwargs.set_item(name, arg.to_python(py)?)?;
            }
            dict.set_item("kwargs", kwargs)?;
            let mut outputs: Vec<Bound<'py, PyAny>> = Vec::with_capacity(node.outputs.len());
            for slot in &node.outputs {
                outputs.push(match slot {
                    Slot::Tensor(meta) => meta.to_dict(py)?.into_any(),
                    Slot::Other => py.None().into_bound(py),
                });
            }
            dict.set_item("outputs", PyList::new(py, outputs)?)?;
            dict.set_item("sequence", node.sequence)?;
            out.push(dict);
        }
        PyList::new(py, out)
    }

    #[getter]
    fn outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.outputs.iter().map(|r| PyCaptureValue::of(*r)))
    }

    /// The whole record in one dict, in the shape docs/CAPTURE.md §5 argues is
    /// the one an `ExportedProgram` is built from.
    fn graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("placeholders", self.guards(py)?)?;
        out.set_item("constants", self.constants(py)?)?;
        out.set_item("nodes", self.nodes(py)?)?;
        out.set_item("outputs", self.outputs(py)?)?;
        Ok(out)
    }

    /// Run the record with new inputs.
    ///
    /// **Replay goes back through `_aten_dispatch`.** It is not a second
    /// interpreter: the same op names, the same non-tensor arguments and the
    /// same door, which is precisely why agreement with eager is evidence
    /// about the *record* rather than about two implementations happening to
    /// match. When a delegate arrives it replaces this loop, and the eager
    /// comparison is then a comparison of backends -- the same test, one layer
    /// down. docs/CAPTURE.md §3.
    fn replay<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let env = self.run(py, inputs)?;
        let mut out = Vec::with_capacity(self.outputs.len());
        for reference in &self.outputs {
            out.push(env.get(py, *reference)?);
        }
        PyTuple::new(py, out)
    }

    /// Reverse-mode over this trace. docs/BACKWARD.md.
    ///
    /// The forward is replayed first, because a `CaptureTrace` keeps only the
    /// *shape* of every intermediate and not the intermediate itself -- capture
    /// drops its keepalives at `_capture_end` (docs/CAPTURE.md §6), and holding
    /// every activation for the life of every trace would be the wrong default
    /// for the traces that are never differentiated. The cost is one extra
    /// forward per backward, and it is named here rather than hidden.
    #[pyo3(signature = (inputs, grad_outputs = None, wrt_constants = None))]
    fn backward<'py>(
        &self,
        py: Python<'py>,
        inputs: &Bound<'py, PyAny>,
        grad_outputs: Option<&Bound<'py, PyAny>>,
        wrt_constants: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        // Under `no_grad`, and for upstream's own reason rather than a
        // convenience. A backward runs *outside* the graph: upstream's engine
        // executes with grad mode off unless `create_graph=True`, so the
        // gradients it produces are leaves. Since docs/BACKWARD4.md the door
        // marks non-leaves, and the tape's backward is a composition of
        // ordinary `aten_dispatch` calls on tensors that require gradients --
        // so without this the returned gradients would come back with a
        // `grad_fn`, and `torch/optim/optimizer.py:1064` would then call
        // `p.grad.detach_()`, which this shim refuses by name. A gradient is a
        // value, not a node.
        let _guard = crate::tensor::NoGradGuard::enter();
        crate::tape::backward(py, self, inputs, grad_outputs, wrt_constants)
    }

    /// Which ops a gradient would have to flow through in this trace, and
    /// whether the tape has a rule for each.
    ///
    /// Exists so that "what stops this model" is answerable **without** running
    /// a backward and reading the exception -- docs/BACKWARD.md's wall table is
    /// generated from this, not hand-kept, because hand-kept op lists in this
    /// repository have gone stale five times.
    #[pyo3(signature = (wrt_constants = None))]
    fn differentiable<'py>(
        &self,
        py: Python<'py>,
        wrt_constants: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        crate::tape::differentiable(py, self, wrt_constants)
    }
}

impl PyCaptureTrace {
    /// Replay the record and keep **every** value it produced.
    ///
    /// `replay` is this plus a projection onto the declared outputs; the tape
    /// is this plus a reverse walk. Extracting it means the two cannot drift
    /// apart -- a backward that materialised its activations its own way would
    /// be differentiating a different forward from the one `replay` proves
    /// equal to eager.
    pub(crate) fn run<'py>(&self, py: Python<'py>, inputs: &Bound<'py, PyAny>) -> PyResult<Env> {
        if is_active() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "torch._C capture: cannot replay a trace while recording -- the replay's \
                 ops would be recorded as if the program had issued them",
            ));
        }
        let given = sequence_items(inputs).ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err(
                "torch._C capture: replay() takes a list or tuple of tensors",
            )
        })?;
        if given.len() != self.inputs.len() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "torch._C capture: this trace was recorded with {} input{}, and replay was \
                 given {}",
                self.inputs.len(),
                if self.inputs.len() == 1 { "" } else { "s" },
                given.len()
            )));
        }

        let mut env = Env {
            inputs: Vec::with_capacity(given.len()),
            consts: self.const_objects.iter().map(|c| c.clone_ref(py)).collect(),
            nodes: Vec::with_capacity(self.nodes.len()),
        };
        for (index, value) in given.iter().enumerate() {
            let tensor = value.cast::<PyTensorBase>().map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "torch._C capture: input {index} of replay is a {}, and the trace \
                     recorded a tensor there",
                    value.get_type().name().map(|n| n.to_string()).unwrap_or_default()
                ))
            })?;
            let seen = TensorMeta::of(tensor);
            self.check_guard(index, &seen)?;
            env.inputs.push(value.clone().unbind());
        }

        for node in &self.nodes {
            let args: PyResult<Vec<_>> = node.args.iter().map(|a| a.materialise(py, &env)).collect();
            let args = PyTuple::new(py, args?)?;
            let kwargs = PyDict::new(py);
            for (name, arg) in &node.kwargs {
                kwargs.set_item(name, arg.materialise(py, &env)?)?;
            }
            let produced =
                crate::aten::aten_dispatch(py, &node.op, &args, Some(&kwargs))?.into_bound(py);
            let slots = match sequence_items(&produced) {
                Some(items) if node.sequence => items,
                None if !node.sequence => vec![produced],
                // A trace that recorded three results and got two back has met
                // a shape it was not recorded under. Guarding the arity is the
                // cheapest place dynamic shape shows itself -- `split` is the
                // op where it will.
                _ => {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "torch._C capture: replaying {} returned a differently shaped result \
                         than the recording did",
                        node.op
                    )))
                }
            };
            if slots.len() != node.outputs.len() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "torch._C capture: replaying {} returned {} results and the recording \
                     had {}",
                    node.op,
                    slots.len(),
                    node.outputs.len()
                )));
            }
            env.nodes.push(slots.into_iter().map(|s| s.unbind()).collect());
        }

        Ok(env)
    }

    fn check_guard(&self, index: usize, seen: &TensorMeta) -> PyResult<()> {
        let want = &self.inputs[index];
        if want.shape != seen.shape {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "torch._C capture: input {index} has shape {:?} and this trace is only valid \
                 for shape {:?}; capture records concrete shapes and does not generalise over \
                 them",
                seen.shape, want.shape
            )));
        }
        if want.dtype != seen.dtype {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "torch._C capture: input {index} has dtype torch.{} and this trace is only \
                 valid for dtype torch.{}",
                seen.dtype.name(),
                want.dtype.name()
            )));
        }
        if want.device != seen.device {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "torch._C capture: input {index} is on device {} and this trace is only valid \
                 for device {}",
                seen.device, want.device
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The module-level switches
// ---------------------------------------------------------------------------

fn not_recording() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("torch._C capture: not recording")
}

#[pyfunction]
#[pyo3(name = "_capture_active")]
pub fn capture_active() -> bool {
    RECORDER.with(|cell| cell.borrow().is_some())
}

/// Why the recording in progress has given up, if it has.
///
/// Readable without ending the recording so that a caller who wants to *fall
/// back* rather than fail has a way to ask. That is the shape a delegate needs:
/// try to capture a region, and run it eagerly when capture says it cannot.
#[pyfunction]
#[pyo3(name = "_capture_reason")]
pub fn capture_reason() -> Option<String> {
    RECORDER.with(|cell| cell.borrow().as_ref().and_then(|r| r.poisoned.clone()))
}

/// Begin recording. `inputs` are the tensors the trace is a function *of*;
/// every other tensor it touches becomes a burned-in constant.
#[pyfunction]
#[pyo3(name = "_capture_begin")]
pub fn capture_begin(py: Python<'_>, inputs: &Bound<'_, PyAny>) -> PyResult<()> {
    if capture_active() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "torch._C capture: already recording -- nested capture would have to decide \
             which trace an op belongs to, and there is no answer to that yet",
        ));
    }
    let items = sequence_items(inputs).ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err(
            "torch._C capture: _capture_begin() takes a list or tuple of tensors",
        )
    })?;

    let mut rec = Recorder {
        nodes: Vec::new(),
        inputs: Vec::new(),
        consts: Vec::new(),
        known: HashMap::new(),
        keepalive: Vec::new(),
        const_objects: Vec::new(),
        poisoned: None,
    };
    for (index, value) in items.iter().enumerate() {
        let tensor = value.cast::<PyTensorBase>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "torch._C capture: input {index} is a {}, and only tensors can be trace inputs",
                value.get_type().name().map(|n| n.to_string()).unwrap_or_default()
            ))
        })?;
        let address = value.as_ptr() as usize;
        if rec.known.contains_key(&address) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "torch._C capture: input {index} is the same object as an earlier input; \
                 a trace cannot tell two names for one tensor apart"
            )));
        }
        rec.inputs.push(TensorMeta::of(tensor));
        rec.known.insert(address, Ref::Input(index));
        rec.keepalive.push(value.clone().unbind());
    }
    let _ = py;

    RECORDER.with(|cell| *cell.borrow_mut() = Some(rec));
    CAPTURING.store(true, Ordering::Relaxed);
    Ok(())
}

/// Throw the recording away. The program's answers are unaffected -- nothing
/// was ever routed through the record.
#[pyfunction]
#[pyo3(name = "_capture_abandon")]
pub fn capture_abandon() -> PyResult<()> {
    take_recorder()?;
    Ok(())
}

fn take_recorder() -> PyResult<Recorder> {
    let taken = RECORDER.with(|cell| cell.borrow_mut().take());
    CAPTURING.store(false, Ordering::Relaxed);
    taken.ok_or_else(not_recording)
}

/// Stop recording and make the claim.
///
/// `outputs` is a tensor, a sequence of tensors, or `None` for a trace with no
/// declared results. Everything that went wrong during the recording surfaces
/// here, because this is the line where "it was captured" is asserted.
#[pyfunction]
#[pyo3(name = "_capture_end")]
pub fn capture_end(py: Python<'_>, outputs: &Bound<'_, PyAny>) -> PyResult<PyCaptureTrace> {
    let rec = take_recorder()?;
    if let Some(reason) = rec.poisoned {
        return Err(crate::err::not_implemented(format!(
            "torch._C capture: cannot capture this region -- {reason}"
        )));
    }

    let declared = if outputs.is_none() {
        Vec::new()
    } else {
        sequence_items(outputs).unwrap_or_else(|| vec![outputs.clone()])
    };
    let mut refs = Vec::with_capacity(declared.len());
    for (index, value) in declared.iter().enumerate() {
        let address = value.as_ptr() as usize;
        let found = rec.known.get(&address).copied().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "torch._C capture: output {index} was not produced inside the recorded \
                 region, so the trace has no way to compute it"
            ))
        })?;
        refs.push(found);
    }
    let _ = py;

    Ok(PyCaptureTrace {
        nodes: rec.nodes,
        inputs: rec.inputs,
        consts: rec.consts,
        const_objects: rec.const_objects,
        outputs: refs,
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCaptureValue>()?;
    m.add_class::<PyCaptureTrace>()?;
    m.add_function(wrap_pyfunction!(capture_active, m)?)?;
    m.add_function(wrap_pyfunction!(capture_reason, m)?)?;
    m.add_function(wrap_pyfunction!(capture_begin, m)?)?;
    m.add_function(wrap_pyfunction!(capture_abandon, m)?)?;
    m.add_function(wrap_pyfunction!(capture_end, m)?)?;
    m.add_function(wrap_pyfunction!(capture_value, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutating_ops_are_recognised_by_torchs_own_spelling() {
        for op in [
            "aten.add_.Tensor",
            "aten.relu_.default",
            "aten.fill_.Scalar",
            "aten.zero_.default",
            "aten.copy_.default",
            "aten.uniform_.default",
            "aten.normal_.default",
        ] {
            assert!(is_mutating(op), "{op}");
        }
        // The trailing underscore is on the *op name*, not anywhere in it.
        for op in [
            "aten.add.Tensor",
            "aten._to_copy.default",
            "aten._softmax.default",
            "aten.lift_fresh.default",
            "aten._local_scalar_dense.default",
            "aten.split_with_sizes.default",
            "aten._unsafe_view.default",
        ] {
            assert!(!is_mutating(op), "{op}");
        }
    }

    #[test]
    fn every_refusal_names_the_op_it_refuses() {
        for op in ["aten.add_.Tensor", "aten._local_scalar_dense.default", "aten.multinomial.default"]
        {
            let reason = refusal_for(op).expect(op);
            assert!(reason.contains(op), "{reason}");
        }
        assert!(refusal_for("aten.add.Tensor").is_none());
    }
}
