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
//! (docs/graph/CAPTURE.md §4): control flow, in-place mutation, dynamic shapes, and
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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
/// preference: docs/devices/DEVICE_ABS.md §6 measured a `String` allocation per tensor
/// argument at +78 ns on `add.Tensor`, a door that costs 346 ns in total, so
/// anything added here is measured against a small number. docs/graph/CAPTURE.md §7
/// has the A/B for this flag.
///
/// Global rather than thread-local because reading a `thread_local!` costs a
/// TLS lookup, and the recorder itself is thread-local anyway -- a second
/// thread that takes the branch finds no recorder and falls straight back out.
static CAPTURING: AtomicBool = AtomicBool::new(false);

thread_local! {
    static RECORDER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
    /// **W8** (docs/training/BACKWARD7.md): the eager tape. Same `Recorder`, no region.
    ///
    /// There is no `_eager_begin`. It exists from the first dispatch that
    /// produces a `grad_fn` and lasts until `_eager_backward` frees it, which
    /// is `docs/training/BACKWARD5.md` §4's "`CAPTURING` on outside a capture region"
    /// with the gating moved to where it costs nothing -- see `EAGER_ON`.
    static EAGER: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

#[inline(always)]
pub fn is_active() -> bool {
    CAPTURING.load(Ordering::Relaxed)
}

/// Whether the eager tape records at all.
///
/// On by default, because a `.backward()` that only works after the caller
/// asked for it is not `.backward()`. It costs nothing when nothing
/// differentiates: the door consults it **only after `mark_from_op` has said
/// it marked an output**, which is upstream's exact condition for a node
/// existing at all, and which is already computed. An inference forward pays
/// one `bool` returned in a register.
static EAGER_ON: AtomicBool = AtomicBool::new(true);

#[inline(always)]
pub fn eager_enabled() -> bool {
    EAGER_ON.load(Ordering::Relaxed)
}

/// Off switch, so that the recorder's own cost can be measured against a
/// control in the same binary rather than against a second build, and so that
/// a caller who knows they will never differentiate can decline the retention
/// W9 is about.
#[pyfunction]
#[pyo3(name = "_eager_set_enabled")]
pub fn eager_set_enabled(value: bool) {
    EAGER_ON.store(value, Ordering::Relaxed);
    if !value {
        eager_free();
    }
}

#[pyfunction]
#[pyo3(name = "_eager_enabled")]
pub fn eager_enabled_py() -> bool {
    eager_enabled()
}

/// **W11** (docs/training/BACKWARD8.md): the largest number of nodes the eager tape
/// will hold before it refuses.
///
/// `docs/training/BACKWARD7.md` §10 row 2 recorded that the tape *"cannot grow without
/// bound"* was not established, and it was not, because it could not: the tape
/// is freed by `backward()` and by nothing else, so a program that runs
/// forwards under grad mode and never differentiates retains every
/// intermediate for the life of the interpreter. Upstream is bounded here for
/// free -- its graph hangs off the output tensors and refcounting collects it
/// when the caller drops them -- and this is not, because the tape is a
/// thread-local the outputs do not own.
///
/// **Measured, docs/training/BACKWARD8.md §4**, a hand-written greedy decode loop on
/// real SmolLM2-135M with `use_cache=True`, parameters left exactly as
/// `from_pretrained` hands them over (every one `requires_grad=True`, which is
/// why the tape records at all), sizes read with `_eager_tape_bytes`:
///
/// | | nodes | tape |
/// |---|---:|---:|
/// | prefill, `S=6` | 1 721 | 13.19 MiB |
/// | + 8 decode steps | 15 489 | 33.93 MiB |
///
/// 1 721 nodes and about 2.6 MiB per decode step, growing without limit, and
/// the same loop under `torch.no_grad()` records **0 nodes and 0 bytes**. RSS
/// is not the instrument here for the reason `_eager_tape_bytes` documents;
/// what RSS could be got to say -- about 3.7 MiB a step before the machine
/// corrupted the reading -- agrees in order with the tape's own count.
///
/// Upstream in this same program is *not* flat, and that correction matters
/// more than the number: `DynamicCache` holds every step's key and value, each
/// with a `grad_fn`, so upstream's graph is retained too. The difference is
/// **ownership, not bookkeeping.** Upstream's graph hangs off tensors, so
/// dropping the cache frees it and `no_grad` never builds it; this tape is a
/// thread-local that outlives every output, and `_eager_reset()` was the only
/// thing that could free it. **A silent unbounded leak is the wrong default in
/// a library that exists for on-device inference**, so the choice here is a
/// bound with a named refusal rather than a documented requirement: a program
/// that trips it is told what it did and what to do about it, at a point where
/// the process is still alive to be told.
///
/// The default is `100_000` nodes, chosen against the measurement above rather
/// than picked. The largest single forward this project runs -- a full
/// SmolLM2-135M prefill -- is 1 721 nodes, so the bound is **58x** the biggest
/// graph a `backward()` here has ever had to hold, and no
/// forward-and-backward can reach it by accident. What it does reach is the
/// runaway: about 58 decode steps and 170 MiB, which is where a leak stops
/// being a rounding error on a phone. Reaching it at all means the caller
/// meant `torch.no_grad()`, because a loop that intended to differentiate
/// would have called `backward()` and freed the tape.
///
/// Tripping it is a **refusal, and a release.** A tape over the bound can
/// never answer again, so it drops every value it holds rather than keeping
/// them alive until `_eager_reset()`: the memory the bound exists to protect
/// is given back at the moment the bound is hit, not at the moment the caller
/// notices. That is safe because `eager_backward` tests `poisoned` before it
/// reads `known`, and `record_into` returns at `poisoned` before it reads
/// anything -- so no path can observe the emptied tables.
static EAGER_MAX_NODES: AtomicUsize = AtomicUsize::new(100_000);

pub fn eager_max_nodes() -> usize {
    EAGER_MAX_NODES.load(Ordering::Relaxed)
}

/// Read the bound. Exists so that a test can assert the default rather than
/// restate it, and so that the refusal can be provoked without recording
/// 100 000 nodes to do it.
#[pyfunction]
#[pyo3(name = "_eager_max_nodes")]
pub fn eager_max_nodes_py() -> usize {
    eager_max_nodes()
}

/// Set the bound. `0` means unbounded, which restores exactly the behaviour
/// `docs/training/BACKWARD7.md` §10 row 2 described -- kept so that the bound can be
/// **nullified** and the tests that assert it seen to go red (`CLAUDE.md`
/// §5.5), not because unbounded is an option anyone should choose.
#[pyfunction]
#[pyo3(name = "_eager_set_max_nodes")]
pub fn eager_set_max_nodes(value: usize) {
    EAGER_MAX_NODES.store(value, Ordering::Relaxed);
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
#[derive(Clone)]
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

impl Arg {
    /// A second reference to the same argument, for `retain_graph=True`.
    ///
    /// Hand-written rather than derived: `Py<PyAny>` is deliberately not
    /// `Clone` in this pyo3 build (the `py-clone` feature is off, because an
    /// implicit refcount bump without a `Python` token is what that feature
    /// removes). `clone_ref` takes the token, so the duplication is explicit
    /// about being a refcount and not a copy of the value.
    fn duplicate(&self, py: Python<'_>) -> Arg {
        match self {
            Arg::Value(reference) => Arg::Value(*reference),
            Arg::Literal(object) => Arg::Literal(object.clone_ref(py)),
            Arg::List(items) => Arg::List(items.iter().map(|a| a.duplicate(py)).collect()),
            Arg::Tuple(items) => Arg::Tuple(items.iter().map(|a| a.duplicate(py)).collect()),
        }
    }
}

impl Node {
    fn duplicate(&self, py: Python<'_>) -> Node {
        Node {
            op: self.op.clone(),
            args: self.args.iter().map(|a| a.duplicate(py)).collect(),
            kwargs: self
                .kwargs
                .iter()
                .map(|(name, arg)| (name.clone(), arg.duplicate(py)))
                .collect(),
            outputs: self.outputs.clone(),
            sequence: self.sequence,
        }
    }
}

struct Recorder {
    nodes: Vec<Node>,
    inputs: Vec<TensorMeta>,
    consts: Vec<TensorMeta>,
    /// Object address -> where that value came from. Every address in here
    /// belongs to an object `input_objects`, `node_objects` or `const_objects`
    /// holds a strong reference to, so an address can never be reused under us
    /// while the recording is open. That is the price of identity: a trace of a
    /// long model holds its activations until `_capture_end`.
    /// docs/graph/CAPTURE.md §6.
    known: HashMap<usize, Ref>,
    /// The declared inputs, held for the identity reason above. A region has
    /// them; an eager tape has none, because everything it reads from outside
    /// itself is a constant.
    input_objects: Vec<Py<PyAny>>,
    /// **W9** (docs/training/BACKWARD7.md). Every tensor result the recording produced,
    /// indexed the way `Ref::Node` indexes it -- `node_objects[n][o]` is the
    /// object of output `o` of node `n`, with `None` in the slots the record
    /// calls `Slot::Other`.
    ///
    /// This was a flat `keepalive: Vec<Py<PyAny>>` before this round, kept
    /// only so that an address could not be reused mid-recording, and dropped
    /// at `_capture_end`. Reshaped, **it is an `Env`**: `docs/training/BACKWARD5.md` §4
    /// found that the thing `docs/training/BACKWARD2.md` §1.5 asked W8 to invent -- what
    /// keeps the intermediates alive -- already existed and was being thrown
    /// away. A capture region still throws it away (`_capture_end` drops the
    /// whole `Recorder`, and `CaptureTrace.backward()` replays); an eager tape
    /// keeps it until `backward()`, which is the lifetime W9 is about.
    node_objects: Vec<Vec<Option<Py<PyAny>>>>,
    const_objects: Vec<Py<PyAny>>,
    /// The stamp each constant carried when it was **first seen**, in
    /// `const_objects` order. For a region this is re-taken at `_capture_end`;
    /// for an eager tape there is no `_capture_end`, so first sight is the
    /// point the tape differentiates at and this is the snapshot W10a compares.
    const_stamps: Vec<Stamp>,
    /// The storage addresses of every value this recording holds -- constants
    /// and node results alike.
    ///
    /// Exists so that `note_mutation` can ask *"is this write landing on a
    /// value my tape depends on?"* in one lookup, and poison rather than
    /// answer. That question is why the eager tape does not need W10b: see
    /// `poison_on_write_to_recorded_storage`.
    storages: HashMap<usize, ()>,
    /// Whether this is the always-on eager tape rather than a `_capture_begin`
    /// region. The two differ in what they refuse (`refusal_for` is a *replay*
    /// guard and an eager tape never replays) and in what they do with
    /// `node_objects`.
    eager: bool,
    poisoned: Option<String>,
}

impl Recorder {
    fn poison(&mut self, reason: String) {
        if self.poisoned.is_none() {
            self.poisoned = Some(reason);
        }
    }

    /// Drop every value this recording holds, keeping only the reason it can
    /// never answer again.
    ///
    /// **W11.** A poisoned tape is unusable by construction -- `record_into`
    /// returns at `poisoned` before it reads any table, and `eager_backward`
    /// tests `poisoned` before it reads `known` -- so the objects it is still
    /// holding are pure retention. For the bound in `EAGER_MAX_NODES` that
    /// retention *is* the thing being bounded, so the refusal releases rather
    /// than waiting for `_eager_reset()`.
    ///
    /// Only called after `poison`. Calling it on a live recording would
    /// destroy the graph without saying so, which is why it is not public and
    /// why `poisoned` is asserted rather than assumed.
    fn release_values(&mut self) {
        debug_assert!(self.poisoned.is_some(), "released a tape that can still answer");
        self.nodes = Vec::new();
        self.inputs = Vec::new();
        self.consts = Vec::new();
        self.known = HashMap::new();
        self.input_objects = Vec::new();
        self.node_objects = Vec::new();
        self.const_objects = Vec::new();
        self.const_stamps = Vec::new();
        self.storages = HashMap::new();
    }

    /// Advance the recorded stamp of `running_mean`/`running_var` by one, to
    /// account for the write the op being recorded has already made and that
    /// `note_mutation` is about to count.
    ///
    /// **W11**, and see the call site for why this is not a loosening. Two
    /// details matter here:
    ///
    /// * It looks the arguments up in `known` rather than re-stamping from the
    ///   tensor, so it works on the second and later calls too -- by then
    ///   `arg_of` has returned the existing `Ref::Const` without touching
    ///   `const_stamps`, and re-stamping would have swallowed every past write
    ///   rather than this one.
    /// * It adds one rather than reading the storage's current version, for
    ///   the same reason: reading would forgive whatever else had happened.
    fn forgive_own_write(&mut self, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) {
        for (index, name) in [(3usize, "running_mean"), (4, "running_var")] {
            let value = match args.get_item(index) {
                Ok(value) => Some(value),
                Err(_) => kwargs.and_then(|kw| kw.get_item(name).ok().flatten()),
            };
            let Some(value) = value else { continue };
            if value.is_none() {
                continue;
            }
            let Some(Ref::Const(slot)) = self.known.get(&(value.as_ptr() as usize)).copied() else {
                continue;
            };
            if let Some(Some((_, version))) = self.const_stamps.get_mut(slot) {
                *version += 1;
            }
        }
    }

    /// A second `Recorder` over the same values, for `retain_graph=True`.
    ///
    /// **W12** (`docs/training/BACKWARD9.md` §3). `eager_backward` *takes* the tape,
    /// which is upstream's `retain_graph=False` default and the whole of W9's
    /// lifetime rule. `retain_graph=True` is the caller saying they will
    /// differentiate the same forward again, so the tape has to survive its
    /// own backward -- and the backward moves every field into a
    /// `PyCaptureTrace` and an `Env`.
    ///
    /// Duplicating rather than borrowing, for a reason the shorter version
    /// would have got wrong: the backward runs Python, Python runs kernels,
    /// and a kernel that reaches `record_into` would try to borrow the same
    /// `RefCell` this call is holding. Duplicating releases the borrow before
    /// any of that. It is not a copy of the tensors -- every `Py` here is a
    /// reference count -- so what it costs is the node list, which is the
    /// honest price of asking for the graph twice.
    ///
    /// `const_stamps` is duplicated **as it stands**, not re-read: a retained
    /// backward is the same freshness question asked twice, so a write between
    /// the two must still be refused by the second.
    fn duplicate_for_retained_backward(&self, py: Python<'_>) -> Self {
        Self {
            nodes: self.nodes.iter().map(|n| n.duplicate(py)).collect(),
            inputs: self.inputs.clone(),
            consts: self.consts.clone(),
            known: self.known.clone(),
            input_objects: self.input_objects.iter().map(|o| o.clone_ref(py)).collect(),
            node_objects: self
                .node_objects
                .iter()
                .map(|slots| {
                    slots
                        .iter()
                        .map(|slot| slot.as_ref().map(|o| o.clone_ref(py)))
                        .collect()
                })
                .collect(),
            const_objects: self.const_objects.iter().map(|o| o.clone_ref(py)).collect(),
            const_stamps: self.const_stamps.clone(),
            storages: self.storages.clone(),
            eager: self.eager,
            poisoned: self.poisoned.clone(),
        }
    }

    fn empty(eager: bool) -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            consts: Vec::new(),
            known: HashMap::new(),
            input_objects: Vec::new(),
            node_objects: Vec::new(),
            const_objects: Vec::new(),
            const_stamps: Vec::new(),
            storages: HashMap::new(),
            eager,
            poisoned: None,
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
/// rather than in a widened `is_mutating`. Measured (docs/architectures/DEMAND1.md §1.3):
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

// ---------------------------------------------------------------------------
// W10a: constant freshness (docs/training/BACKWARD6.md)
// ---------------------------------------------------------------------------

/// One monotonic `u64` per **storage**, bumped when an op writes into it.
///
/// The defect this closes is `docs/training/BACKWARD5.md` §1.3: `PyCaptureTrace` holds
/// strong references to the caller's live tensors (`const_objects`) and
/// `run()` copies those references into the replay `Env`, so a trace
/// differentiates whatever its constants hold **at `backward()` time** -- which
/// in a training loop is whatever `optimizer.step()` last wrote. There is no
/// in-place op in the record; the mutation happens after `_capture_end`, and
/// before this the tape said nothing about it.
///
/// This is the *snapshot* half of upstream's `c10::VariableVersion` applied to
/// the 333 values a trace holds rather than to every saved variable
/// (`docs/training/BACKWARD5.md` §2). It is deliberately **not** the general version
/// counter: there is no `ADInplaceOrView` key, no alias set, no view metadata
/// and no rebasing.
///
/// **Keyed on the storage address, not on the Python object**, and that is
/// what makes it see a write through a view: in-place ops here go through
/// `tensor::write_into`, which writes into the buffer the wrapper already
/// points at (`docs/kernels/VIEWS.md` §6), so a base and its views share one candle
/// `Storage` and therefore one entry. `replace_with` -- `set_` and
/// `tensor.data = ...` -- rebinds instead, and that shows up as a *different*
/// key, which the stamp comparison catches as well because a stamp is the pair
/// `(key, version)` and not the version alone.
///
/// A `BTreeMap` rather than a `HashMap` only because `BTreeMap::new` is `const`
/// and this needs no lazy initialisation. Entries are created **only** by a
/// mutating op, so an inference forward -- which issues none -- never touches
/// the lock and never allocates. A storage address can be reused after its
/// tensor is dropped, which would leave a stale count attached to an unrelated
/// buffer; that cannot produce a false refusal for a *trace constant*, because
/// a trace holds a strong reference to every constant and its storage
/// therefore cannot be freed while the stamp it is compared against exists.
static STORAGE_VERSIONS: std::sync::Mutex<std::collections::BTreeMap<usize, u64>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// The identity of a tensor's bytes, as far as freshness is concerned: which
/// storage, and how many writes had gone into it when this was taken.
///
/// `None` for a tensor with no storage to stamp -- `Repr::Meta` has no buffer
/// at all and `Repr::Quantized` lives in candle's separate type system. Both
/// compare equal to themselves by being absent, which is the right answer:
/// nothing here can write into either of them in place.
pub(crate) type Stamp = Option<(usize, u64)>;

/// The address of the candle `Storage` behind a tensor, or `None` if it has
/// none. Two tensors that share a buffer answer the same value; the
/// `storage_offset` is deliberately *not* added, unlike `TensorBase.data_ptr`,
/// because a view at a non-zero offset must share its base's version.
fn storage_key(tensor: &Bound<'_, PyTensorBase>) -> Option<usize> {
    let borrowed = tensor.try_borrow().ok()?;
    match borrowed.repr() {
        crate::tensor::Repr::Dense(inner) => {
            let (guard, _layout) = inner.storage_and_layout();
            let storage: &candle_core::Storage = &guard;
            Some(storage as *const candle_core::Storage as usize)
        }
        _ => None,
    }
}

/// Read a tensor's stamp. Never raises: a tensor that cannot be borrowed (a
/// kernel is holding it mutably) stamps as `None`, which is the answer that
/// refuses nothing.
pub(crate) fn stamp_of(tensor: &Bound<'_, PyTensorBase>) -> Stamp {
    let key = storage_key(tensor)?;
    let version = STORAGE_VERSIONS.lock().ok()?.get(&key).copied().unwrap_or(0);
    Some((key, version))
}

/// The receiver of an in-place op, found the way `aten.rs`'s `tensor_receiver`
/// finds it -- positional 0, or the keyword `self`, because `bootstrap.py`
/// binds every argument by keyword and a direct `torch.ops.aten` call need not.
///
/// Only the receiver is bumped. Bumping every tensor argument would have been
/// one line shorter and would fire on `w.add_(x)` for `x` as well, which is a
/// check that refuses a trace whose constant was merely *read* -- the exact
/// shape of failure this round is not allowed to introduce.
fn inplace_receiver<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> Option<Bound<'py, PyTensorBase>> {
    let value = match args.get_item(0) {
        Ok(value) => value,
        Err(_) => kwargs?.get_item("self").ok().flatten()?,
    };
    value.cast_into::<PyTensorBase>().ok()
}

/// **The bump site**, called from the door for every dispatch.
///
/// It is at the door and not inside `tensor::write_into` for the reason the
/// capture hook is at the door: there is exactly one place where "an op
/// happened" is known, and a kernel cannot forget to call it by being written
/// later. It is beside `mark_from_op` rather than inside it because
/// `mark_from_op` returns early when grad mode is off, and **`optimizer.step()`
/// runs under `no_grad`** -- putting the bump behind that branch would have
/// missed the one call this whole round exists for. `docs/training/BACKWARD5.md` §6
/// sized it as "the door's existing grad-mode branch"; that sizing is wrong by
/// one branch, and `docs/training/BACKWARD6.md` §3 records why.
///
/// The cost on the ordinary path is `is_mutating` -- one `rsplit_once` on a
/// `&str` already in a register and an `ends_with` on one byte -- and a branch
/// that is not taken. No lock is taken and nothing is allocated unless the op
/// writes.
pub fn note_mutation(op: &str, args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) {
    if !is_mutating(op) {
        // The ops that mutate without saying so in their name. The list has
        // one member and the judgement is per call, which is why this reuses
        // `mutates_this_call` rather than the name alone.
        if !MUTATES_WITHOUT_UNDERSCORE.contains(&op) || !mutates_this_call(op, args, kwargs) {
            return;
        }
        // `native_batch_norm` writes its running statistics, which are
        // arguments 3 and 4, and not its receiver.
        for (index, name) in [(3usize, "running_mean"), (4, "running_var")] {
            let value = match args.get_item(index) {
                Ok(value) => Some(value),
                Err(_) => kwargs.and_then(|kw| kw.get_item(name).ok().flatten()),
            };
            if let Some(tensor) = value.and_then(|v| v.cast_into::<PyTensorBase>().ok()) {
                if let Some(key) = storage_key(&tensor) {
                    bump(key);
                }
            }
        }
        return;
    }
    if let Some(receiver) = inplace_receiver(args, kwargs) {
        if let Some(key) = storage_key(&receiver) {
            bump(key);
            poison_on_write_to_recorded_storage(key);
        }
    }
}

/// **W8's answer to `docs/training/BACKWARD5.md` §1's 100×-wrong gradient, and the
/// reason this round does not need W10b.**
///
/// `docs/training/BACKWARD5.md` §1.1 measured what a naive eager recorder does with
/// `a = x*1; v = a.view(3); a.mul_(10); (v*v).sum()`: the recorder keys values
/// on object identity, the view is a *second object over one storage*, the
/// write goes to the base and the tape differentiates a program the machine
/// never ran -- `[0.6, 1.4, 2.2]` where upstream says `[60.0, 140.0, 220.0]`,
/// **silently**. §6 called the fix "upstream's whole aliasing layer" and
/// deferred it on the grounds that it had no consumer until a recorder
/// existed.
///
/// A recorder now exists, and the fix it needs is not that layer. **A tape
/// that refuses does not have to know what aliases what** -- it has to know
/// that a write landed on bytes it depends on, and `docs/training/BACKWARD6.md` §4
/// already made that one lookup by keying versions on the candle `Storage`
/// rather than on the Python object. A base and its views answer the same key,
/// so the write above is seen without any view metadata, any alias set or any
/// `ADInplaceOrView` key existing.
///
/// What is bought is a **refusal**, not a gradient: upstream *differentiates*
/// A5/A6 and this refuses them, which is strictly less. That is the trade
/// `docs/training/AUTOGRAD.md` §6 chose, and the difference between it and W10b is the
/// difference between "no wrong answer" and "the right answer".
///
/// Cost: on the ordinary path, nothing -- `note_mutation` has already returned
/// for any op that is not in-place. On an in-place op it is one hash lookup,
/// and only if an eager tape exists at all.
fn poison_on_write_to_recorded_storage(key: usize) {
    if !eager_enabled() {
        return;
    }
    EAGER.with(|cell| {
        let Ok(mut slot) = cell.try_borrow_mut() else {
            return;
        };
        let Some(rec) = slot.as_mut() else {
            return;
        };
        if !rec.storages.contains_key(&key) {
            return;
        }
        rec.poison(
            "an in-place operation wrote into a tensor the eager graph holds. The graph \
             records the *mathematics* of each op and reads the values back at backward() \
             time, so a write that landed after the op ran -- including one made through a \
             view of the same storage -- would make it differentiate a program that never \
             ran. Upstream refuses the same shape with its version counter; this refuses it \
             by storage. Compute the value again after the write, or do the write under \
             torch.no_grad() on a tensor no graph depends on (docs/training/BACKWARD7.md)"
                .to_string(),
        );
    });
}

fn bump(key: usize) {
    if let Ok(mut table) = STORAGE_VERSIONS.lock() {
        *table.entry(key).or_insert(0) += 1;
    }
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

/// Ops whose output *shape* is a function of tensor **values** rather than of
/// its inputs' shapes and dtypes.
///
/// A recorded node's shape has to be implied by the guards, or a replay on a
/// different input silently produces a differently shaped answer through a
/// graph built for the first one. `nonzero` and `where.default` count their
/// true elements; `repeat_interleave.Tensor` sums its `repeats` (docs/kernels/REPEAT.md
/// §3) -- the same property, arrived at from the other side, and it joined this
/// list in the same change that gave it a kernel rather than after somebody
/// noticed a wrong replay.
const DATA_DEPENDENT_SHAPE: &[&str] = &[
    "aten.nonzero.default",
    "aten.repeat_interleave.Tensor",
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
        record_into(py, rec, op, args, kwargs, out);
    });
}

/// **W8**: the eager tape's half of the door.
///
/// Called only when `tensor::mark_from_op` has just given an output a
/// `grad_fn`, which is why this function does not repeat any of that test.
/// The consequences of that gating are the whole design and are worth stating:
///
/// * An op recorded here is exactly an op upstream would have built a node
///   for -- grad mode on, a differentiable op, an operand that requires a
///   gradient, a floating result, and a result that is a *new* tensor.
/// * Therefore **no in-place op is ever recorded**, because an in-place op
///   returns its receiver and fails that last clause. That is not a silent
///   omission: `poison_on_write_to_recorded_storage` refuses the tape by name
///   when a write lands on a value it holds.
/// * A region wins. While `_capture_begin` is open the ops belong to that
///   trace and this does not run, so nesting never has to be decided.
pub fn eager_record(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    out: &Py<PyAny>,
) {
    if is_active() {
        return;
    }
    EAGER.with(|cell| {
        let Ok(mut slot) = cell.try_borrow_mut() else {
            return;
        };
        let rec = slot.get_or_insert_with(|| Recorder::empty(true));
        record_into(py, rec, op, args, kwargs, out);
    });
}

fn record_into(
    py: Python<'_>,
    rec: &mut Recorder,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    out: &Py<PyAny>,
) {
    {
        if rec.poisoned.is_some() {
            return;
        }
        // **W11** (docs/training/BACKWARD8.md §4): the bound. Tested before the node is
        // built rather than after, so the tape never exceeds the number it
        // reports, and only for the eager tape -- a capture region is bounded
        // by its own `_capture_end` and by the caller who wrote it.
        //
        // One relaxed load and a comparison, on a path that has already
        // decided to allocate a `Node`. `0` disables it, which is what
        // nullifying the bound means.
        let limit = eager_max_nodes();
        if rec.eager && limit != 0 && rec.nodes.len() >= limit {
            rec.poison(format!(
                "the eager graph grew past {limit} nodes without ever being differentiated. \
                 The tape is freed by backward() and by nothing else, so a loop that runs \
                 forwards under grad mode and never calls it retains every intermediate -- \
                 measured at 1721 nodes and ~2.6 MiB per decode step of SmolLM2-135M, with \
                 no limit, where upstream's graph hangs off the output tensors and this \
                 one does not. The values held so far have been released. Run the loop \
                 under torch.no_grad(), which records nothing at all, or call \
                 torch._C._eager_reset() each iteration, or raise the bound with \
                 torch._C._eager_set_max_nodes(n) (docs/training/BACKWARD8.md §4)"
            ));
            rec.release_values();
            return;
        }
        // `refusal_for` and the `native_batch_norm` test below are **replay**
        // guards, and an eager tape never replays: it differentiates the
        // values the program actually computed, in `node_objects`. So the
        // three things they refuse mean different things on the two paths.
        // Randomness is the clearest: docs/graph/CAPTURE.md §9-1 records a gradient
        // taken at a *different dropout draw* than the one reported, which is
        // a replay defect, and an eager tape cannot have it because it holds
        // the draw. Mutation is not exempted -- it is refused somewhere else,
        // and more precisely, by `poison_on_write_to_recorded_storage`.
        if !rec.eager {
            if let Some(reason) = refusal_for(op) {
            rec.poison(reason);
            return;
            }
        }
        // The arg-aware half of the same question, for the one op whose name
        // does not carry torch's mutation convention. See
        // `MUTATES_WITHOUT_UNDERSCORE`.
        if !rec.eager && MUTATES_WITHOUT_UNDERSCORE.contains(&op) && mutates_this_call(op, args, kwargs) {
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

        // **W11** (docs/training/BACKWARD8.md §2): forgive this op its *own* write.
        //
        // Measured on a real `nn.Sequential(Conv2d, BatchNorm2d, ReLU)` in
        // `train()` mode: before this, the tape refused its own forward. The
        // mechanism is an ordering, not a policy. `aten.rs` calls
        // `eager_record` -- which stamps a constant the first time it sees it
        // -- and then calls `note_mutation`, which bumps
        // `running_mean`/`running_var` for the write the kernel had *already*
        // made before either ran. So the stamp is taken at version N over a
        // buffer that is at version N, and one line later the same, single,
        // already-completed write moves it to N+1. `backward()` then finds a
        // constant one version stale and refuses, on a program upstream
        // answers, with nothing in between having written anything.
        //
        // Anticipating the bump here is exactly as narrow as the defect: it
        // moves the stamp by one, for the two arguments `note_mutation` is
        // about to bump, only when `mutates_this_call` says this call writes
        // them. A *second* write -- a real `optimizer.step()`, a
        // `running_mean.zero_()`, a second BatchNorm call reading the same
        // buffer -- lands at N+2 against an expected N+1 and is still refused
        // by name. The guard is not loosened; it stops counting the op's own
        // write twice.
        //
        // Safe because the value is not read back. The training-mode
        // derivative of `native_batch_norm` is a function of the input, the
        // weight and `save_mean`/`save_invstd`, which the op *returns*; the
        // running statistics are an output of the forward and an input to
        // nothing. Eval mode does read them and writes nothing, so
        // `mutates_this_call` is `false` there and this does not run.
        if rec.eager
            && MUTATES_WITHOUT_UNDERSCORE.contains(&op)
            && mutates_this_call(op, args, kwargs)
        {
            rec.forgive_own_write(args, kwargs);
        }

        let node_index = rec.nodes.len();
        let bound = out.bind(py);
        let (slots, sequence) = match sequence_items(bound) {
            Some(items) => (items, true),
            None => (vec![bound.clone()], false),
        };

        let mut outputs = Vec::with_capacity(slots.len());
        let mut held: Vec<Option<Py<PyAny>>> = Vec::with_capacity(slots.len());
        for (position, item) in slots.iter().enumerate() {
            match item.cast::<PyTensorBase>() {
                Ok(tensor) => {
                    outputs.push(Slot::Tensor(TensorMeta::of(tensor)));
                    let address = item.as_ptr() as usize;
                    rec.known
                        .insert(address, Ref::Node { node: node_index, output: position });
                    if let Some(key) = storage_key(tensor) {
                        rec.storages.insert(key, ());
                    }
                    held.push(Some(item.clone().unbind()));
                }
                Err(_) => {
                    if !rec.eager && !METADATA_ONLY.contains(&op) {
                        return rec.poison(format!(
                            "{op} returned a value that is not a tensor, and it is not on \
                             the metadata-only allowlist; capture cannot tell whether that \
                             value depends on tensor *data*, which no guard constrains"
                        ));
                    }
                    outputs.push(Slot::Other);
                    held.push(None);
                }
            }
        }
        rec.node_objects.push(held);

        rec.nodes.push(Node {
            op: op.to_string(),
            args: positional,
            kwargs: named,
            outputs,
            sequence,
        });
    }
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
            // W10a, at first sight rather than at `_capture_end`. A region
            // re-takes these when it ends (`capture_end`), because that is the
            // line where "it was captured" is asserted; an eager tape has no
            // such line, so the point it differentiates at is the point it
            // first read the value, and that is what has to be stamped.
            self.const_stamps.push(stamp_of(tensor));
            if let Some(key) = storage_key(tensor) {
                self.storages.insert(key, ());
            }
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
    /// The stamp each constant carried at `_capture_end`, in `const_objects`
    /// order. Compared in `run()`; see `STORAGE_VERSIONS`.
    pub(crate) const_stamps: Vec<Stamp>,
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
    /// (docs/graph/CAPTURE.md §4), and this is what naming it has to mean.
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

    /// The version each constant's storage was at when the trace was captured
    /// (`None` for a constant with no storage to stamp). Exists so that a test
    /// can say *which* stamp moved rather than inferring it from a refusal,
    /// and so that "the counter is not moving at all" is distinguishable from
    /// "the counter moved and the check let it through".
    #[getter]
    fn constant_versions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.const_stamps
                .iter()
                .map(|stamp| stamp.map(|(_, version)| version)),
        )
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

    /// The whole record in one dict, in the shape docs/graph/CAPTURE.md §5 argues is
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
    /// down. docs/graph/CAPTURE.md §3.
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

    /// Reverse-mode over this trace. docs/training/BACKWARD.md.
    ///
    /// The forward is replayed first, because a `CaptureTrace` keeps only the
    /// *shape* of every intermediate and not the intermediate itself -- capture
    /// drops its keepalives at `_capture_end` (docs/graph/CAPTURE.md §6), and holding
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
        // gradients it produces are leaves. Since docs/training/BACKWARD4.md the door
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
    /// a backward and reading the exception -- docs/training/BACKWARD.md's wall table is
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
    /// **W10a**: refuse a replay whose burned-in constants have moved since
    /// the trace was captured (`docs/training/BACKWARD5.md` §1.3, `docs/training/BACKWARD6.md`).
    ///
    /// A `CaptureTrace` holds its constants *by reference*, so without this a
    /// `backward()` called after `optimizer.step()` differentiates at the new
    /// weights and returns a plausible number with no message -- which is the
    /// failure `docs/graph/CAPTURE.md` §9-1 already records one other instance of,
    /// and the reason a refusal by name is worth more here than a recomputed
    /// answer would be. Recomputing is not on the table anyway: the trace has
    /// no record of how its constants were produced, because they were made
    /// before the region began.
    ///
    /// The wording keeps upstream's `is at version N; expected version M`
    /// clause, because upstream refuses this same program by that phrase and
    /// anything matching on it should keep matching.
    fn check_constants_are_fresh(&self, py: Python<'_>) -> PyResult<()> {
        for (index, object) in self.const_objects.iter().enumerate() {
            let expected = match self.const_stamps.get(index) {
                Some(Some(stamp)) => *stamp,
                // Not stamped at capture (meta, quantised, or unborrowable):
                // there is nothing this could compare, so it refuses nothing.
                _ => continue,
            };
            let tensor = match object.bind(py).cast::<PyTensorBase>() {
                Ok(tensor) => tensor,
                Err(_) => continue,
            };
            let seen = match stamp_of(tensor) {
                Some(stamp) => stamp,
                None => continue,
            };
            if seen == expected {
                continue;
            }
            let meta = self.consts.get(index);
            let described = meta
                .map(|m| {
                    format!(
                        "torch.{}{:?} on {}",
                        m.dtype.name(),
                        m.shape.as_slice(),
                        m.device
                    )
                })
                .unwrap_or_else(|| "a tensor".to_string());
            let versions = if seen.0 == expected.0 {
                format!("it is at version {}; expected version {}", seen.1, expected.1)
            } else {
                // `set_` or `tensor.data = ...`: the wrapper stopped pointing
                // at the storage that was stamped, so there is no version to
                // compare and the identity of the buffer is the difference.
                "it points at a different storage than the one the trace was captured over"
                    .to_string()
            };
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "torch._C capture: constant {index} of this trace ({described}) has been \
                 modified by an in-place operation since the trace was captured -- \
                 {versions}. This trace was captured *before* that tensor moved and holds \
                 it by reference, so replaying or differentiating it now would silently \
                 answer at the new value instead of the one the region ran on. Capture the \
                 region again after the update -- e.g. call trace.backward() before \
                 optimizer.step(), not after (docs/training/BACKWARD6.md)"
            )));
        }
        Ok(())
    }

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

        self.check_constants_are_fresh(py)?;

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

    let mut rec = Recorder::empty(false);
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
        rec.input_objects.push(value.clone().unbind());
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
    // W10a: the freshness snapshot, taken here because this is the line where
    // "it was captured" is asserted, and the point the tape differentiates at
    // is the point its constants were at when that claim was made.
    let const_stamps: Vec<Stamp> = rec
        .const_objects
        .iter()
        .map(|object| match object.bind(py).cast::<PyTensorBase>() {
            Ok(tensor) => stamp_of(tensor),
            Err(_) => None,
        })
        .collect();

    Ok(PyCaptureTrace {
        nodes: rec.nodes,
        inputs: rec.inputs,
        consts: rec.consts,
        const_objects: rec.const_objects,
        const_stamps,
        outputs: refs,
    })
}

// ---------------------------------------------------------------------------
// W8 / W9: the eager tape, its lifetime, and its backward
// ---------------------------------------------------------------------------

/// Throw the eager tape away, releasing every intermediate it held.
///
/// **This is W9.** `docs/training/BACKWARD5.md` §3 measured what is being released:
/// 18.9 MiB at `S=8`, 75.7 at 32, 302.6 at 128 for SmolLM2-135M, distinct
/// storages, held for one iteration -- which §3 also established is what
/// upstream already pays for the same program. The lifetime rule is upstream's
/// `retain_graph=False` **default** and not a lesser refusal: a graph is freed
/// by the backward that consumes it, and a second backward over the same graph
/// raises.
fn eager_free() {
    EAGER.with(|cell| {
        let Ok(mut slot) = cell.try_borrow_mut() else {
            return;
        };
        drop(slot.take());
    });
}

#[pyfunction]
#[pyo3(name = "_eager_reset")]
pub fn eager_reset() {
    eager_free();
}

/// How many nodes the eager tape is currently holding.
///
/// Exists so that a test can assert the *retention* W9 is about -- that the
/// tape grows during a forward and is empty after a backward -- without
/// inferring it from a gradient.
#[pyfunction]
#[pyo3(name = "_eager_tape_size")]
pub fn eager_tape_size() -> usize {
    EAGER.with(|cell| {
        cell.try_borrow()
            .ok()
            .and_then(|slot| slot.as_ref().map(|rec| rec.nodes.len()))
            .unwrap_or(0)
    })
}

/// **W11** (docs/training/BACKWARD8.md §4): how many bytes of tensor the eager tape is
/// keeping alive that nothing else would.
///
/// Exists because RSS is not an instrument on this machine. The decode-loop
/// measurement in §4 was first taken with `ru_maxrss`, which is a *peak* and
/// therefore reported `+0.0 MiB` for a tape that had grown by fourteen
/// thousand nodes; retaken with `ps -o rss=` it reported a growth that then
/// went **negative by 599 MiB** between two consecutive steps, because eight
/// other agents were on the machine and pages were reclaimed underneath it.
/// `docs/training/BACKWARD7.md` §8 already recorded this machine corrupting a
/// measurement. A number the tape computes about itself cannot be corrupted
/// that way, and it is the number the bound in `EAGER_MAX_NODES` is about.
///
/// **Two corrections make it a size and not a fiction**, and both were found
/// by measuring without them. Summing `shape.product() * itemsize` over every
/// recorded result reported **529 MiB for a single SmolLM2-135M prefill** and
/// 516 MiB per decode step, against an RSS that moved by about 4 MiB a step.
/// The excess is entirely aliasing and ownership:
///
/// * **Deduplicated by storage.** Every `Linear` records a `t()` of its
///   weight, and that result is a *view*: a distinct tensor over storage the
///   tape is already counting. The `lm_head` transpose alone is a `[49152,
///   576]` view, 108 MiB counted for a buffer that costs nothing. Each
///   storage is therefore counted once, at the largest view of it the tape
///   holds.
/// * **Constant storages excluded.** That transpose does not merely duplicate
///   a count, it counts *the model's parameters* as tape. The tape holds them
///   by reference and `from_pretrained` owns them; freeing the tape returns
///   none of it. So a storage that any constant also points at contributes
///   nothing.
///
/// What is left is an **upper bound on what dropping the tape would return**:
/// a result the caller is still holding anyway is counted here too, which
/// errs high, which is the direction a bound wants. It is not free -- it binds
/// and borrows every held tensor -- so it is a diagnostic to be called between
/// steps, not on the dispatch path.
#[pyfunction]
#[pyo3(name = "_eager_tape_bytes")]
pub fn eager_tape_bytes(py: Python<'_>) -> usize {
    EAGER.with(|cell| {
        let Ok(slot) = cell.try_borrow() else {
            return 0;
        };
        let Some(rec) = slot.as_ref() else {
            return 0;
        };
        let mut owned_by_the_model: std::collections::HashSet<usize> = Default::default();
        for object in &rec.const_objects {
            if let Ok(tensor) = object.bind(py).cast::<PyTensorBase>() {
                if let Some(key) = storage_key(tensor) {
                    owned_by_the_model.insert(key);
                }
            }
        }
        let mut largest: HashMap<usize, usize> = HashMap::new();
        for held in &rec.node_objects {
            for item in held.iter().flatten() {
                let Ok(tensor) = item.bind(py).cast::<PyTensorBase>() else {
                    continue;
                };
                let Some(key) = storage_key(tensor) else {
                    continue;
                };
                if owned_by_the_model.contains(&key) {
                    continue;
                }
                let Ok(borrowed) = tensor.try_borrow() else {
                    continue;
                };
                let bytes = borrowed
                    .dims()
                    .iter()
                    .product::<usize>()
                    .saturating_mul(borrowed.tag().itemsize());
                let entry = largest.entry(key).or_insert(0);
                if bytes > *entry {
                    *entry = bytes;
                }
            }
        }
        largest.values().sum()
    })
}

/// Why the eager tape has given up, if it has. The same shape as
/// `_capture_reason`, and readable without consuming the tape.
#[pyfunction]
#[pyo3(name = "_eager_reason")]
pub fn eager_reason() -> Option<String> {
    EAGER.with(|cell| {
        cell.try_borrow()
            .ok()
            .and_then(|slot| slot.as_ref().and_then(|rec| rec.poisoned.clone()))
    })
}

/// Reverse-mode over the eager tape, from `output`.
///
/// **The reuse is the point.** This function builds no derivative rules. It
/// projects the `Recorder` onto a `PyCaptureTrace` -- which is legal because
/// `docs/training/BACKWARD5.md` §4 found `Recorder` already holds four of the five
/// things `tape::backward` reads -- pairs it with the fifth, an `Env` made out
/// of `node_objects` **without replaying anything**, and calls
/// `tape::backward_in`. All 60 rules, `derivative()`, `wrt_set()`,
/// `reachable()` and `wanted()` run unchanged and cannot tell which producer
/// called them.
///
/// `reachable()` in particular was expected to need replacing
/// (`docs/training/BACKWARD5.md` §4, last paragraph). It did not. With no trace inputs
/// -- an eager tape has none, everything it reads from outside is a constant --
/// its rule "a node is needed if it reads a wanted constant or a needed node"
/// **is** the propagated `requires_grad` flag `docs/training/BACKWARD4.md` installed,
/// computed from the same information one step earlier.
///
/// **Wired into `Tensor.backward()` since `docs/training/BACKWARD9.md`.** It kept its
/// name: `_ImperativeEngine.run_backward` in `bootstrap.py` is a translation
/// onto this function and holds no derivative rule, no traversal and no
/// lifetime rule of its own. What it adds is upstream's shape -- `.grad`
/// accumulation, `allow_unused`, `inputs=`, and the two flags that tell
/// `Tensor.backward()` from `torch.autograd.grad()`. `docs/training/BACKWARD7.md` §6
/// was the list of what stood between the two.
#[pyfunction]
#[pyo3(name = "_eager_backward")]
#[pyo3(signature = (output, grad_output = None, wrt = None, retain_graph = false))]
pub fn eager_backward<'py>(
    py: Python<'py>,
    output: &Bound<'py, PyAny>,
    grad_output: Option<&Bound<'py, PyAny>>,
    wrt: Option<&Bound<'py, PyAny>>,
    retain_graph: bool,
) -> PyResult<Bound<'py, PyDict>> {
    if is_active() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "torch._C eager: cannot differentiate the eager graph while a capture region is \
             recording -- the backward's own ops would be recorded into that region",
        ));
    }
    let address = output.as_ptr() as usize;
    // `retain_graph=True` duplicates instead of taking, so the tape is still
    // there for the second backward (`Recorder::duplicate_for_retained_backward`).
    let rec = if retain_graph {
        EAGER.with(|cell| {
            cell.borrow()
                .as_ref()
                .map(|rec| rec.duplicate_for_retained_backward(py))
        })
    } else {
        EAGER.with(|cell| cell.borrow_mut().take())
    };
    let Some(rec) = rec else {
        return Err(eager_missing(py, output, "there is no eager graph"));
    };
    // Taken, not borrowed: W9's free happens whether this succeeds or raises,
    // which is `retain_graph=False`. Putting it back on the error paths would
    // make a failed backward keep 302 MiB alive for the length of a traceback.
    let restore = |rec: Recorder| drop(rec);
    if let Some(reason) = &rec.poisoned {
        let message = format!("torch._C eager: cannot differentiate this graph -- {reason}");
        restore(rec);
        return Err(crate::err::not_implemented(message));
    }
    let Some(reference) = rec.known.get(&address).copied() else {
        restore(rec);
        return Err(eager_missing(py, output, "this tensor is not in the eager graph"));
    };
    if !matches!(reference, Ref::Node { .. }) {
        restore(rec);
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "torch._C eager: this tensor is a leaf of the eager graph -- nothing recorded \
             here produced it, so there is nothing to differentiate through. Upstream says \
             \"element 0 of tensors does not require grad and does not have a grad_fn\"",
        ));
    }

    // The projection. Every field is moved, not copied: the trace lives only
    // for this call and the `Recorder` is already gone.
    let trace = PyCaptureTrace {
        nodes: rec.nodes,
        inputs: Vec::new(),
        consts: rec.consts,
        const_objects: rec.const_objects,
        const_stamps: rec.const_stamps,
        outputs: vec![reference],
    };
    // W10a, unchanged and now load-bearing for a second caller: a leaf that
    // moved between the op that read it and this backward is refused by name
    // rather than differentiated at its new value.
    //
    // **W10a's freshness check, and what a real model did to it**
    // (docs/training/BACKWARD8.md §2). `docs/training/BACKWARD7.md` §10 row 3 predicted that a
    // mid-forward buffer write would be refused here where upstream answers,
    // and named two: a KV cache and a batch-norm running statistic. Measured,
    // both, on real models:
    //
    // * **The KV cache does not reach this at all.** transformers'
    //   `DynamicCache` grows by `torch.cat`, which allocates; there is no
    //   in-place write for a storage guard to see. SmolLM2-135M with
    //   `use_cache=True`, prefilled and decoded and differentiated, answers,
    //   and the gradient agrees with 2.13.0.
    // * **The batch-norm statistic did reach it, and it was this guard's own
    //   fault.** `aten.rs` calls `eager_record` -- which stamps the constant
    //   -- before `note_mutation`, which then bumped the same storage for the
    //   write the kernel had already made, so the tape refused *its own
    //   forward*. `Recorder::forgive_own_write` counts that write once, and a
    //   training-mode BatchNorm now differentiates.
    //
    // What is left refusing here is a genuinely second write: an
    // `optimizer.step()` before the `backward()`, a `zero_()` on a recorded
    // buffer. The wording is widened rather than replaced, because the message
    // is `PyCaptureTrace`'s and a capture region still raises it verbatim. On
    // the eager path its advice -- *"capture the region again"* -- names a call
    // the caller never made, and a refusal that tells you to do something
    // impossible is a worse refusal than one that admits it.
    if let Err(stale) = trace.check_constants_are_fresh(py) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{}\n\nThere is no region to capture again: this is the eager tape, and the \
             write happened between the op that read this tensor and backward() \
             (docs/training/BACKWARD8.md §2). Take the gradient before the write -- backward() \
             before optimizer.step(), not after -- or run the write under torch.no_grad() \
             on a tensor no graph depends on",
            stale.value(py).str().map(|s| s.to_string()).unwrap_or_default()
        )));
    }

    // Which constants a gradient is wanted for: upstream's rule, read off the
    // flag `docs/training/BACKWARD4.md` landed. `wrt_set` does the dtype half.
    let selected: Vec<usize> = match wrt.filter(|value| !value.is_none()) {
        Some(value) => {
            let wanted: Vec<Bound<'py, PyAny>> = value.extract()?;
            let mut out = Vec::new();
            for tensor in wanted {
                let at = tensor.as_ptr() as usize;
                match trace
                    .const_objects
                    .iter()
                    .position(|c| c.bind(py).as_ptr() as usize == at)
                {
                    Some(index) => out.push(index),
                    None => {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err(
                            "torch._C eager: a tensor named in wrt is not a leaf of this \
                             graph -- nothing recorded here read it, so no gradient flows \
                             to it. Upstream returns None for the same case under \
                             allow_unused=True",
                        ))
                    }
                }
            }
            out
        }
        None => {
            let mut out = Vec::new();
            for (index, object) in trace.const_objects.iter().enumerate() {
                if let Ok(tensor) = object.bind(py).cast::<PyTensorBase>() {
                    let wants = tensor
                        .getattr("requires_grad")
                        .and_then(|v| v.extract::<bool>())
                        .unwrap_or(false);
                    if wants {
                        out.push(index);
                    }
                }
            }
            out
        }
    };

    // The `Env`, and this is W9's payoff: no replay. `run()` re-executes the
    // whole forward because a `CaptureTrace` kept only shapes; here the values
    // are the ones the program computed, so the gradient is taken at the point
    // the forward actually ran -- which is also why a random draw is safe on
    // this path and refused on the other (docs/graph/CAPTURE.md §9-1).
    let env = Env {
        inputs: Vec::new(),
        consts: trace.const_objects.iter().map(|c| c.clone_ref(py)).collect(),
        nodes: rec
            .node_objects
            .into_iter()
            .map(|slots| {
                slots
                    .into_iter()
                    .map(|slot| slot.unwrap_or_else(|| py.None()))
                    .collect()
            })
            .collect(),
    };

    let seeds = grad_output.filter(|value| !value.is_none());
    let grads = {
        let _guard = crate::tensor::NoGradGuard::enter();
        let indices = PyList::new(py, &selected)?;
        crate::tape::backward_in(py, &trace, &env, seeds, Some(indices.as_any()))?
    };

    // Handed back beside the tensors they belong to, because the caller has no
    // other way to name a constant: the tape assigned those indices, not them.
    let constants = grads.get_item("constants")?.ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("torch._C eager: tape returned no constants")
    })?;
    let out = PyDict::new(py);
    out.set_item("tensors", PyList::new(py, trace.const_objects.iter().map(|c| c.clone_ref(py)))?)?;
    out.set_item("grads", constants)?;
    out.set_item("wrt", PyList::new(py, &selected)?)?;
    out.set_item("nodes", trace.nodes.len())?;
    Ok(out)
}

/// The two ways a tensor can fail to be differentiable here, told apart.
///
/// "The graph was freed" and "this never had a graph" are the same absence in
/// `known`, and upstream distinguishes them by message -- `retain_graph` for
/// the first, *"does not have a grad_fn"* for the second. Getting it wrong
/// sends a reader to the wrong half of the problem.
///
/// The question is asked of the **tensor** and not of a table of remembered
/// addresses. An earlier version of this kept the freed tape's addresses in a
/// side set, and the first program that exercised both branches got the wrong
/// message: a freed address had been reused by an unrelated tensor. The
/// tensor's own `grad_fn` cannot be wrong that way -- it is set by the door at
/// the moment the op ran, it is exactly upstream's condition, and a tensor
/// carrying one whose graph is gone is precisely the `retain_graph` case.
fn eager_missing(py: Python<'_>, output: &Bound<'_, PyAny>, generic: &str) -> PyErr {
    let had_a_node = output
        .cast::<PyTensorBase>()
        .ok()
        .and_then(|cell| cell.try_borrow().ok().map(|t| t._shim_from_op().is_some()))
        .unwrap_or(false);
    let _ = py;
    // `eager_enabled()` as well as the `grad_fn`: with the recorder switched
    // off a tensor still gets a `grad_fn` (that is `mark_from_op`, which is
    // W5 and not W8), and calling its graph "freed" would send the reader
    // looking for a `retain_graph` that was never the problem.
    if had_a_node && eager_enabled() {
        return pyo3::exceptions::PyRuntimeError::new_err(
            "torch._C eager: Trying to backward through the graph a second time -- the \
             saved intermediate values have already been freed. This is upstream's \
             retain_graph=False default, and it is the whole of the lifetime rule: a \
             backward consumes the graph it walks (docs/training/BACKWARD5.md §3 measured what is \
             being released -- 302.6 MiB for SmolLM2-135M at S=128). Run the forward again, \
             or keep a reference to what you need, or pass retain_graph=True",
        );
    }
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "torch._C eager: {generic}. A tensor is in the eager graph only if an op produced \
         it under grad mode from an operand that requires one -- which is upstream's \
         condition for grad_fn being non-None, and `.grad_fn` reports it. Upstream says \
         \"element 0 of tensors does not require grad and does not have a grad_fn\""
    ))
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
    m.add_function(wrap_pyfunction!(eager_backward, m)?)?;
    m.add_function(wrap_pyfunction!(eager_reset, m)?)?;
    m.add_function(wrap_pyfunction!(eager_tape_size, m)?)?;
    m.add_function(wrap_pyfunction!(eager_reason, m)?)?;
    m.add_function(wrap_pyfunction!(eager_set_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(eager_enabled_py, m)?)?;
    m.add_function(wrap_pyfunction!(eager_tape_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(eager_max_nodes_py, m)?)?;
    m.add_function(wrap_pyfunction!(eager_set_max_nodes, m)?)?;
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
