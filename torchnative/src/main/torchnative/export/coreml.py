"""Build and **run** a CoreML model from a lowered trace.

## Why this does not go through `torch/backends/_coreml`

docs/graph/DECOMP.md §12.6 recorded CoreML as *not measured*, for two reasons, and
this round changes one of them and not the other.

`torch/backends/_coreml/preprocess.py` is a packaging wrapper: it calls
`coremltools.convert` on a `torch.jit` object and stores the blob. That door is
shut here for the same reason `nnapi.py`'s is -- this shim has no TorchScript
compiler, `torch.jit.trace` returns its argument unchanged. It is shut harder,
in fact: NNAPI's serialiser could be driven behind a duck-typed graph because
every access to the graph was one of thirteen Python methods, whereas
coremltools' torch frontend walks a real `torch.jit` IR with type refinement
and its own `InternalTorchIRGraph`.

But CoreML has a second, documented entry that NNAPI has no equivalent of:
**MIL**, coremltools' own intermediate language, with a Python builder
(`coremltools.converters.mil.Builder`). `ct.convert` on a MIL program is a
supported, first-class path -- the torch frontend's whole job is to *produce*
one. So the analogue of `nnapi.to_jit_module` here is `to_mil_program`: our
recorded graph emitted directly as MIL, skipping the frontend that needs a
`jit` trace rather than faking one.

## And this one is executed

`libcoremlpython` is present on macOS, so a model built here is compiled by
the OS and run. `verify()` runs it and compares against
`DecomposedTrace.replay` -- our own graph through `_aten_dispatch` -- on the
same inputs. That is an executed artefact, and it is the only thing in this
round that is: the NNAPI blob is *structurally validated* and nothing more,
because there is no NNAPI runtime on a Mac. docs/graph/NPU.md keeps the two apart.

## `coreml_ops` and §12.6

`target.coreml_ops()` still refuses, and correctly: the claim it makes is that
there is no CoreML operator set *in this tree*, and there still is not.
`coreml_ops()` here is the successor §12.6 asked for -- it reads coremltools'
own `@register_torch_op` registry, which is authoritative in exactly the sense
`ADDER_MAP` is for NNAPI, and it is a *read*, not a transcription. It reports
what the torch frontend accepts; what this module emits is `supported_ops()`,
which is smaller and separately named because conflating "coremltools could
convert this from a jit graph" with "this module has a MIL lowering for it"
would be the §12.6 mistake wearing the other hat.
"""

from __future__ import annotations

from typing import Any

from .nnapi import JitFacadeRefused, _positional, fold_constants

__all__ = [
    "CoreMLRefused",
    "CoreMLUnsupported",
    "PRECISIONS",
    "compute_plan",
    "computes",
    "coreml_ops",
    "coremltools_available",
    "compile_model",
    "plan_lowering",
    "supported_ops",
    "to_mil_program",
    "verify",
]


class CoreMLRefused(RuntimeError):
    """The trace cannot be expressed in MIL by this module, and why."""


def coremltools_available() -> bool:
    try:
        import coremltools  # noqa: F401
    except Exception:
        return False
    return True


def coreml_ops() -> frozenset[str]:
    """The torch op names coremltools' own frontend registry accepts.

    Read out of `coremltools.converters.mil.frontend.torch.ops`, which is
    populated by `@register_torch_op`. This is the authority docs/graph/DECOMP.md
    §12.6 said to go and read once coremltools existed, and it is read rather
    than copied for the same reason `target.nnapi_ops()` parses `ADDER_MAP`.

    Note what it is *not*: these names are what the frontend can translate from
    a TorchScript graph, which this build cannot produce. So this is a measure
    of CoreML's operator coverage, not of what is reachable from here. What is
    reachable from here is `supported_ops()`.
    """
    from coremltools.converters.mil.frontend.torch.ops import _TORCH_OPS_REGISTRY

    mapping = getattr(_TORCH_OPS_REGISTRY, "name_to_func_mapping", None)
    if mapping is None:  # pragma: no cover -- coremltools changed shape
        raise CoreMLRefused(
            "torchnative coreml: coremltools' TorchOpsRegistry no longer "
            "exposes name_to_func_mapping; re-read it rather than guessing"
        )
    return frozenset(mapping)


# ---------------------------------------------------------------------------
# Trace -> MIL
# ---------------------------------------------------------------------------


#: torch dtype -> numpy dtype, for the tensors that cross into MIL.
_NUMPY_DTYPES = {
    "torch.float32": "float32",
    "torch.float64": "float64",
    "torch.int64": "int64",
    "torch.int32": "int32",
    "torch.bool": "bool",
}


def _np(value):
    """A shim tensor as a numpy array.

    Via `tolist()`, not `numpy()`: `TensorBase.numpy` is not implemented in
    this shim and `np.asarray` goes through it, so the buffer route fails with
    a message about a method nobody called. `tolist()` is exact for every dtype
    here -- it goes through Python ints and floats, and float32 round-trips
    through a Python float without loss -- so this is slow, not lossy, and the
    numerical claims in `verify()` are unaffected by it.
    """
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        dtype = _NUMPY_DTYPES.get(str(value.dtype))
        if dtype is None:
            raise CoreMLRefused(
                f"torchnative coreml: no numpy dtype mapped for {value.dtype}"
            )
        return np.array(value.detach().tolist(), dtype=dtype).reshape(
            tuple(value.shape)
        )
    return value


def _conv(mb, x, args):
    weight, bias, stride, padding, dilation, transposed, output_padding, groups = args
    if transposed:
        raise CoreMLRefused(
            "torchnative coreml: transposed convolution is a different MIL op "
            "(conv_transpose) with its own padding convention; unmapped rather "
            "than approximated"
        )
    if any(p != 0 for p in output_padding):
        raise CoreMLRefused("torchnative coreml: output_padding is only meaningful "
                            "for transposed convolution")
    rank = len(padding)
    pad = []
    for p in padding:
        pad.extend([p, p])
    kwargs = dict(x=x, weight=_np(weight), strides=list(stride),
                  pad_type="custom", pad=pad, dilations=list(dilation),
                  groups=int(groups))
    if bias is not None:
        kwargs["bias"] = _np(bias)
    return mb.conv(**kwargs), rank


def _adaptive_avg_pool2d(mb, x, output_size):
    if list(output_size) != [1, 1]:
        raise CoreMLRefused(
            f"torchnative coreml: adaptive_avg_pool2d to {list(output_size)} is "
            f"not a plain spatial mean; only (1, 1) is mapped, because any "
            f"other output size is a pooling window computation and writing it "
            f"here would be inventing one"
        )
    return mb.reduce_mean(x=x, axes=[2, 3], keep_dims=True)


#: `captured op -> builder`. Each builder receives `(mb, inputs, consts, node)`
#: where `inputs` are MIL vars for the tensor positions and `consts` are the
#: recorded Python values, both already flattened onto schema order by
#: `nnapi._positional`.
#:
#: Absence is a refusal. The refusals above (`conv_transpose`, non-(1,1)
#: adaptive pooling) are deliberate and say what they would have had to invent.
_BUILDERS: dict[str, Any] = {}


def _builder(name):
    def register(fn):
        _BUILDERS[name] = fn
        return fn
    return register


def supported_ops() -> frozenset[str]:
    """Captured op names this module has a MIL lowering for.

    Registers first. The table is filled inside a function so that importing
    this module does not import coremltools, and an earlier version returned an
    empty set to any caller who asked before the first compile -- which reads
    as "nothing is supported" rather than as "not loaded yet".
    """
    _register_all()
    return frozenset(_BUILDERS)


def _register_all():
    """Populated in a function so `mb` is imported only when CoreML is wanted."""
    from coremltools.converters.mil import Builder as mb

    if _BUILDERS:
        return mb

    _BUILDERS.update({
        "aten.relu.default": lambda a: mb.relu(x=a[0]),
        "aten.sigmoid.default": lambda a: mb.sigmoid(x=a[0]),
        "aten.tanh.default": lambda a: mb.tanh(x=a[0]),
        "aten.erf.default": lambda a: mb.erf(x=a[0]),
        "aten.exp.default": lambda a: mb.exp(x=a[0]),
        "aten.sqrt.default": lambda a: mb.sqrt(x=a[0]),
        "aten.rsqrt.default": lambda a: mb.rsqrt(x=a[0]),
        "aten.neg.default": lambda a: mb.mul(x=a[0], y=-1.0),
        "aten.sin.default": lambda a: mb.sin(x=a[0]),
        "aten.cos.default": lambda a: mb.cos(x=a[0]),
        "aten.detach.default": lambda a: mb.identity(x=a[0]),
        "aten.clone.default": lambda a: mb.identity(x=a[0]),
        "aten.contiguous.default": lambda a: mb.identity(x=a[0]),
        "aten.add.Tensor": lambda a: mb.add(x=a[0], y=a[1]),
        "aten.sub.Tensor": lambda a: mb.sub(x=a[0], y=a[1]),
        "aten.mul.Tensor": lambda a: mb.mul(x=a[0], y=a[1]),
        "aten.div.Tensor": lambda a: mb.real_div(x=a[0], y=a[1]),
        "aten.add.Scalar": lambda a: mb.add(x=a[0], y=float(a[1])),
        "aten.mul.Scalar": lambda a: mb.mul(x=a[0], y=float(a[1])),
        "aten.pow.Tensor_Scalar": lambda a: mb.pow(x=a[0], y=float(a[1])),
        "aten.mm.default": lambda a: mb.matmul(x=a[0], y=a[1]),
        "aten.matmul.default": lambda a: mb.matmul(x=a[0], y=a[1]),
        "aten.bmm.default": lambda a: mb.matmul(x=a[0], y=a[1]),
        "aten.t.default": lambda a: mb.transpose(x=a[0], perm=[1, 0]),
        "aten.permute.default": lambda a: mb.transpose(x=a[0], perm=list(a[1])),
        "aten.view.default": lambda a: mb.reshape(x=a[0], shape=list(a[1])),
        "aten.reshape.default": lambda a: mb.reshape(x=a[0], shape=list(a[1])),
        "aten._softmax.default": lambda a: mb.softmax(x=a[0], axis=int(a[1])),
        "aten.softmax.int": lambda a: mb.softmax(x=a[0], axis=int(a[1])),
        "aten.cat.default": lambda a: mb.concat(
            values=list(a[0]), axis=int(a[1]) if len(a) > 1 else 0
        ),
        "aten.mean.dim": lambda a: mb.reduce_mean(
            x=a[0], axes=list(a[1]), keep_dims=bool(a[2]) if len(a) > 2 else False
        ),
        "aten.sum.dim_IntList": lambda a: mb.reduce_sum(
            x=a[0], axes=list(a[1]), keep_dims=bool(a[2]) if len(a) > 2 else False
        ),
        "aten.hardtanh.default": lambda a: mb.clip(
            x=a[0],
            alpha=float(a[1]) if len(a) > 1 else -1.0,
            beta=float(a[2]) if len(a) > 2 else 1.0,
        ),
        "aten.unsqueeze.default": lambda a: mb.expand_dims(x=a[0], axes=[int(a[1])]),
        "aten.adaptive_avg_pool2d.default": lambda a: _adaptive_avg_pool2d(mb, a[0], a[1]),
        "aten._adaptive_avg_pool2d.default": lambda a: _adaptive_avg_pool2d(mb, a[0], a[1]),
        "aten.linear.default": lambda a: mb.linear(
            x=a[0], weight=_np(a[1]),
            **({"bias": _np(a[2])} if len(a) > 2 and a[2] is not None else {})
        ),
        "aten.addmm.default": lambda a: mb.add(
            x=mb.matmul(x=a[1], y=a[2]), y=a[0]
        ),
        "aten.convolution.default": lambda a: _conv(mb, a[0], a[1:9])[0],
        "aten.gelu.default": lambda a: mb.gelu(
            x=a[0],
            mode="TANH_APPROXIMATION" if len(a) > 1 and a[1] == "tanh"
            else "EXACT",
        ),
        "aten.silu.default": lambda a: mb.silu(x=a[0]),
    })
    return mb


def to_mil_program(trace, *, fold: bool = True):
    """Emit `trace` as a MIL program. Returns `(program, input_names, trace)`.

    The returned trace is the one actually emitted -- folded, if folding ran --
    so a caller that wants to compare against `replay` compares against the
    same graph and not the one before the pass.
    """
    import numpy as np
    import torch

    mb = _register_all()

    if fold:
        trace, _ = fold_constants(trace)

    specs, names = [], []
    for index, guard in enumerate(trace.guards):
        if guard["dtype"] != "torch.float32":
            raise CoreMLRefused(
                f"torchnative coreml: input {index} is {guard['dtype']}; MIL "
                f"input specs here are float32 only, and widening the claim "
                f"without a test for each dtype is how a wrong one ships"
            )
        names.append(f"x{index}")
        specs.append(mb.TensorSpec(shape=tuple(guard["shape"])))

    constant_values = list(trace.constant_values)
    nodes = list(trace.nodes)
    outputs = list(trace.outputs)

    def body(inputs):
        env: dict[tuple, Any] = {
            ("input", index, 0): var for index, var in enumerate(inputs)
        }
        for index, value in enumerate(constant_values):
            env[("const", index, 0)] = value

        def resolve(arg):
            if isinstance(arg, torch._C.CaptureValue):
                key = (arg.kind, arg.index, arg.output)
                if key not in env:
                    raise CoreMLRefused(
                        f"torchnative coreml: reference {key} used before it "
                        f"is produced"
                    )
                value = env[key]
                return _np(value) if isinstance(value, torch.Tensor) else value
            if isinstance(arg, (list, tuple)):
                return [resolve(item) for item in arg]
            return arg

        for position, node in enumerate(nodes):
            op = node["op"]
            if op not in _BUILDERS:
                raise CoreMLRefused(
                    f"torchnative coreml: no MIL lowering for {op}. "
                    f"coreml_ops() reports whether coremltools' *frontend* "
                    f"knows it; this module's supported_ops() is what has a "
                    f"lowering here, and the two are different questions"
                )
            args = [resolve(a) for a in _positional(op, node["args"], node["kwargs"])]
            produced = _BUILDERS[op](args)
            slots = list(produced) if node["sequence"] else [produced]
            if len(slots) != len(node["outputs"]):
                raise CoreMLRefused(
                    f"torchnative coreml: {op} produced {len(slots)} MIL "
                    f"result(s) and the trace recorded {len(node['outputs'])}"
                )
            for slot, var in enumerate(slots):
                env[("node", position, slot)] = var

        return [resolve(ref) for ref in outputs]

    # `mb.program` reads the *parameter names* of the decorated function to
    # name the model's inputs, so a `*args` function gets zero inputs and the
    # error blames the spec list. The function therefore has to be built with
    # real parameters, and `exec` is the only way to make named parameters from
    # a list computed at runtime.
    namespace = {"body": body}
    exec(  # noqa: S102 -- source is `x0, x1, ...`, built here, not from input
        f"def program({', '.join(names)}):\n"
        f"    return body([{', '.join(names)}])\n",
        namespace,
    )
    program = mb.program(input_specs=specs)(namespace["program"])
    return program, names, trace


def compile_model(trace, *, fold: bool = True, float32: bool = True,
                  compute_units=None):
    """Convert `trace` to a real `MLModel`. Returns `(model, input_names, trace)`.

    `ct.convert` on a MIL program runs coremltools' full backend pipeline and
    the result is an `.mlpackage` the OS compiles -- the same artefact
    `torch/backends/_coreml` would have produced from a jit trace.

    `float32=True` is not a detail. coremltools defaults `mlprogram` to
    **float16** compute precision, and the first run of `verify()` here showed
    exactly that: a two-layer MLP disagreed with replay by 1.4e-4 and a
    `cat` of `x` with `2x` by 8.2e-4 -- both far outside any float32
    tolerance, both entirely explained by half precision, and neither a defect
    in the lowering. Taking that as a lowering error would have sent the
    search to the wrong place; taking it as "close enough" would have hidden a
    real one behind the same number. So precision is pinned and the default is
    the one under which a numerical claim means what it says.

    `compute_units` is passed to `ct.convert` unchanged and defaults to
    coremltools' own default. It is named here because a Neural Engine claim
    needs `ct.ComputeUnit.CPU_AND_NE` -- with `ALL` the GPU remains an option
    and "it agreed" would not say which of the two produced the numbers. That
    is docs/graph/NPU2.md §2's construction, and it is the reason `verify` takes
    the argument too rather than the tests reaching around it.
    """
    import coremltools as ct

    program, names, emitted = to_mil_program(trace, fold=fold)
    extra = {} if compute_units is None else {"compute_units": compute_units}
    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=(
            ct.precision.FLOAT32 if float32 else ct.precision.FLOAT16
        ),
        **extra,
    )
    return model, names, emitted


def verify(trace, inputs, *, tolerance: float = 2e-5, fold: bool = True,
           float32: bool = True, compute_units=None) -> dict:
    """Compile, **run**, and compare against `DecomposedTrace.replay`.

    This is the executed claim. Both sides are computed on the same inputs:
    the CoreML side by the operating system's runtime over the compiled
    `.mlpackage`, the reference side by our own graph through
    `torch._C._aten_dispatch`. Agreement is evidence about the MIL lowering,
    not about two libraries happening to implement an op the same way, for the
    reason docs/graph/CAPTURE.md §3 gives about replaying through the door capture
    recorded at.

    **The tolerance is an argument and the default is the float32 one.** 2e-5 is
    where docs/graph/NPU.md set it and it is the bar the word *agrees* means in
    this project. A float16 model does not meet it -- docs/graph/NPU.md §6
    measured 2.3e-04 and docs/graph/NPU2.md §2 measured 2.0e-04 -- so a caller
    verifying the Neural Engine path passes its own, larger tolerance and gets a
    result that says which one it used. Widening this default so that one number
    covered both precisions would erase the distinction the whole of
    docs/graph/NPU2.md is about.
    """
    import numpy as np

    model, names, emitted = compile_model(
        trace, fold=fold, float32=float32, compute_units=compute_units)
    feed = {
        name: _np(value).astype(np.float32)
        for name, value in zip(names, inputs)
    }
    produced = model.predict(feed)
    reference = emitted.replay(inputs)

    ordered = list(produced.values())
    if len(ordered) != len(reference):
        raise CoreMLRefused(
            f"torchnative coreml: the model returned {len(ordered)} output(s) "
            f"and the trace has {len(reference)}"
        )
    worst = 0.0
    for got, want in zip(ordered, reference):
        got = np.asarray(got, dtype=np.float64)
        want = _np(want).astype(np.float64)
        if got.shape != want.shape:
            raise CoreMLRefused(
                f"torchnative coreml: CoreML returned shape {got.shape} and "
                f"the trace computes {want.shape}"
            )
        worst = max(worst, float(np.max(np.abs(got - want))))
    return {
        "executed": True,
        "outputs": len(ordered),
        "max_abs_diff": worst,
        "within_tolerance": worst <= tolerance,
        "tolerance": tolerance,
    }


# ---------------------------------------------------------------------------
# The lowering arm: `nn.Module.to(torchnative.device.npu)` on a CoreML host
# ---------------------------------------------------------------------------
#
# ## Why there are two precisions here and not one
#
# docs/graph/NPU2.md §1.1 measured the tension this section exists inside.
# `compile_model(float32=True)` is pinned because coremltools' float16 default
# disagreed with `DecomposedTrace.replay` by 2.3e-04 against 3.0e-08 -- so the
# pin is what makes a numerical claim mean what it says. It is also what puts
# the Neural Engine out of reach: for a float32 program CoreML does not list
# the unit in the *supported* column at all, so no `compute_units` setting
# reaches it.
#
# **There is no third road, and this was checked rather than assumed.**
# coremltools' `compute_precision` accepts exactly three things
# (`converters/_converters_entry.py`, the `compute_precision` docstring and
# the `compute_precision not in [precision.FLOAT32, precision.FLOAT16]`
# validation): `precision.FLOAT32` (no transform), `precision.FLOAT16` (cast
# everything), and `transform.FP16ComputePrecision(op_selector=...)` -- which
# is a *subset selector for the float16 cast*, not a float32 route to the
# unit. Nothing in the API asks for float32 on the Neural Engine, because the
# Neural Engine is float16 hardware. Measured here on an `ios16.linear` at
# (1, 1024), (1, 4096) and (128, 1024): float32's supported set is
# `CPU, GPU` at every size, float16's contains `NeuralEngine` at every size.
#
# So a float16 path that reaches the unit and a float32 path that agrees are
# two different products, and they are **two spellings**:
#
#     model.to(device.npu)                        # float16, reaches the unit
#     model.to(device.npu, precision="float32")   # agrees, CPU/GPU only
#
# and never one spelling with a silent mode. `precision` is on the report, the
# per-operation `MLComputePlan` rows are on the report, and the case that
# cannot reach the unit the caller named warns. What is not offered is a
# default that quietly picks for you and a report that does not say which.
#
# ## What the grades are
#
# `float32`  -- **agrees**, at `verify`'s own 2e-05, where docs/graph/NPU.md set
#               it. Runs on CPU or GPU through CoreML. Does not reach the ANE.
# `float16`  -- **agrees at float16**, which is around 1e-03 for a
#               `Linear(1024, 1024)` and is not this project's usual bar. It is
#               named differently for that reason rather than covered by a
#               widened tolerance.


class CoreMLUnsupported(NotImplementedError):
    """Something is refused by name: a leaf, a precision, or an empty lowering."""


#: The two spellings, and the whole set of them. A precision outside this is a
#: refusal and not a fallback -- an argument accepted and dropped is how a mode
#: goes silent, which is the failure docs/graph/NPU2.md is about.
PRECISIONS = ("float16", "float32")


def _torch():
    import torch

    return torch


def _check_precision(precision: str) -> str:
    if precision not in PRECISIONS:
        raise CoreMLUnsupported(
            f"torchnative coreml: precision={precision!r} is not one of "
            f"{list(PRECISIONS)}. coremltools' `compute_precision` accepts "
            f"float32, float16 and FP16ComputePrecision(op_selector=...) -- and "
            f"the third is a subset selector for the float16 cast, not a third "
            f"precision, so there is nothing here for a name outside these two "
            f"to mean. float16 reaches the Neural Engine and agrees to about "
            f"1e-03; float32 agrees at 2e-05 and runs on CPU/GPU only. Pick "
            f"one by name; this will not pick for you."
        )
    return precision


def compute_plan(model, *, compute_units=None) -> list[dict]:
    """CoreML's own answer to which unit runs each operation of `model`.

    `model` is an `MLModel` as `compile_model` returns it. Returns one row per
    *computing* operation -- `const` has no compute device and is dropped --
    with `op`, `preferred` and the sorted `supported` set.

    This is the evidence, not a decoration. docs/graph/NPU2.md §1 is the record
    of a model that was compiled by macOS, run, and agreed with replay at
    3e-08 while every operation in it ran on the **CPU**, and the only thing
    that revealed it was this call. "It produced the right answer" and "it ran
    on the NPU" are different sentences and nothing but `MLComputePlan`
    separates them.

    The double compile is not tidiness. `MLComputePlan.load_from_path` wants a
    compiled `.mlmodelc`; handed the `.mlpackage` that `MLModel.save` writes it
    aborts the process with a C++ exception rather than raising.
    """
    import os
    import tempfile

    import coremltools as ct
    import coremltools.models.utils as ct_utils
    from coremltools.models.compute_device import (
        MLCPUComputeDevice, MLGPUComputeDevice, MLNeuralEngineComputeDevice)
    from coremltools.models.compute_plan import MLComputePlan

    if compute_units is None:
        compute_units = ct.ComputeUnit.ALL

    def name_of(device):
        if isinstance(device, MLNeuralEngineComputeDevice):
            return "NeuralEngine"
        if isinstance(device, MLGPUComputeDevice):
            return "GPU"
        if isinstance(device, MLCPUComputeDevice):
            return "CPU"
        return type(device).__name__

    directory = tempfile.mkdtemp(prefix="torchnative-coreml-")
    package = os.path.join(directory, "m.mlpackage")
    model.save(package)
    plan = MLComputePlan.load_from_path(
        ct_utils.compile_model(package), compute_units=compute_units)
    function = plan.model_structure.program.functions["main"]
    rows = []
    for operation in function.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if usage is None:
            continue
        rows.append({
            "op": operation.operator_name,
            "preferred": name_of(usage.preferred_compute_device),
            "supported": sorted(
                name_of(d) for d in usage.supported_compute_devices),
        })
    return rows


def computes(rows) -> list[dict]:
    """`rows` without the boundary `cast`s, which are not the model's work.

    A float16 program has an `ios16.cast` at each end converting the float32
    interface tensors, and those stay on the CPU by construction. Counting them
    against the offload would make every float16 program look partial, so the
    question "which unit ran this" is asked of the computing operations.
    """
    return [row for row in rows if not row["op"].endswith("cast")]


def _eligible_linear(module) -> None:
    """Raise `CoreMLUnsupported` if this `Linear` has no MIL lowering here.

    **Pure, and deliberately so.** No coremltools, no compile, no MLModel --
    which is what lets `plan_lowering` answer "what would happen" on a machine
    that cannot run any of it, and what keeps the plan and the real lowering
    from drifting: `_CoreMLLinear.__init__` calls this same function.
    """
    weight = module.weight
    if weight.dim() != 2:
        raise CoreMLUnsupported(
            f"torchnative coreml: a Linear's weight must be 2-D for "
            f"mb.linear; got shape {tuple(weight.shape)}"
        )
    if not weight.dtype.is_floating_point:
        raise CoreMLUnsupported(
            f"torchnative coreml: will not lower a {weight.dtype} weight. "
            f"This stage emits float weights into MIL; an integer weight is "
            f"the quantized path and there is no MIL lowering for it here"
        )


def plan_lowering(model, predicate=None) -> dict:
    """What `_compile_model` would lower and what it would leave. No CoreML.

    The same function, with the same keys, that `intelnpu.plan_lowering`
    provides for the Intel arm, and for the same reason: the lowering needs
    coremltools and a Mac, and **the selection does not**. A granularity defect
    lives in the selection, so the selection has to be inspectable on a machine
    that cannot run the thing.

    It is **not** evidence that anything ran on the Neural Engine. `compute_plan`
    is the function that answers that, and it needs a compiled model.

    Returns `eligible`, `skipped` (`(name, reason)`), `left_on_cpu`,
    `fully_offloaded`, `parameters_moved`, `parameters_total`, `fraction_moved`.
    """
    torch = _torch()
    eligible, skipped, left = [], [], {}
    moved = 0

    def walk(parent, prefix):
        nonlocal moved
        for name, child in list(parent.named_children()):
            path = f"{prefix}{name}"
            if isinstance(child, torch.nn.Linear):
                if predicate is not None and not predicate(path, child):
                    skipped.append((path, "excluded by predicate"))
                    continue
                try:
                    _eligible_linear(child)
                except CoreMLUnsupported as exc:
                    skipped.append((
                        path,
                        f"Linear(out_features={child.out_features}, "
                        f"in_features={child.in_features}) stays on the CPU: "
                        f"{str(exc).split(chr(10))[0]}",
                    ))
                    continue
                eligible.append(path)
                moved += child.weight.numel() + (
                    child.bias.numel()
                    if getattr(child, "bias", None) is not None else 0
                )
                continue
            grandchildren = list(child.named_children())
            if not grandchildren:
                left[type(child).__name__] = left.get(type(child).__name__, 0) + 1
            else:
                walk(child, f"{path}.")

    walk(model, "")
    total = sum(p.numel() for p in model.parameters())
    return {
        "eligible": eligible,
        "skipped": skipped,
        "left_on_cpu": dict(sorted(left.items())),
        "fully_offloaded": not left and not skipped,
        "parameters_moved": moved,
        "parameters_total": total,
        "fraction_moved": moved / total if total else 0.0,
    }


class _CoreMLLinear:
    """A `torch.nn.Linear` replacement whose forward runs through CoreML.

    The same mechanism as `intelnpu._NPULinear`, and the shape is copied
    deliberately: `named_children()` + `add_module()`, a leaf that holds the
    original weight, and a model that is still an `nn.Module` afterwards so
    `state_dict()` and `generate()` do not learn anything about the hardware.

    **The weight is kept at float32.** `_NPULinear` casts its parameter to
    float16 because OpenVINO's IR here is f16; CoreML takes precision as a
    *conversion* option, so casting the parameter as well would change what
    `state_dict()` returns for no gain and would make `precision="float32"`
    a lie about the weights.

    **Compiled per shape, lazily.** MIL input specs here are static, so a
    different batch is a different program. That is stated rather than hidden
    and it is why the compute plan is recorded per shape: docs/graph/NPU2.md
    §2.1 measured that CoreML prefers the CPU for a small program and the
    Neural Engine for a large one *with nothing changed but size*, so "which
    unit" is not answerable until there is a real shape.

    **The compute plan is read at every compile and kept.** Not optionally.
    The cost is one extra OS compile per shape, once; the alternative is a
    model that ran and nobody knows where, which is the outcome
    docs/graph/NPU2.md exists to prevent.
    """

    def __new__(cls, *args, **kwargs):
        # Built as a subclass of the *shim's* `nn.Module` at first use rather
        # than at import, so this module can be imported -- and its refusals
        # tested -- without torch being importable at all. `_NPULinear` does
        # the same and for the same reason.
        torch = _torch()
        if not issubclass(cls, torch.nn.Module):
            cls = type("_CoreMLLinear", (_CoreMLLinear, torch.nn.Module), {})
            return torch.nn.Module.__new__(cls)
        return super().__new__(cls)

    def __init__(self, weight, bias=None, *, precision: str = "float16",
                 compute_units=None, report=None):
        torch = _torch()
        torch.nn.Module.__init__(self)
        self.precision = _check_precision(precision)
        self.out_features, self.in_features = (
            int(weight.shape[0]), int(weight.shape[1])
        ) if weight.dim() == 2 else (0, 0)
        self.weight = torch.nn.Parameter(weight.detach())
        self.bias = (
            torch.nn.Parameter(bias.detach()) if bias is not None else None
        )
        _eligible_linear(self)
        self._compute_units = compute_units
        self._compiled = {}
        #: The shared report dict `_compile_model` attaches to the model. Every
        #: leaf appends its plans to the same object, so a caller reading
        #: `model.torchnative_offload` after a forward sees what actually ran
        #: rather than what was intended at `to()` time.
        self._report = report if report is not None else {"plans": [], "_said": []}
        self._weight_np = None
        self._bias_np = None

    @classmethod
    def from_torch(cls, layer, *, precision="float16", compute_units=None,
                   report=None):
        return cls(layer.weight, getattr(layer, "bias", None),
                   precision=precision, compute_units=compute_units,
                   report=report)

    # -- the CoreML layer -------------------------------------------------
    def _units(self):
        import coremltools as ct

        return ct.ComputeUnit.ALL if self._compute_units is None \
            else self._compute_units

    def _arrays(self):
        if self._weight_np is None:
            self._weight_np = _np(self.weight)
            self._bias_np = _np(self.bias) if self.bias is not None else None
        return self._weight_np, self._bias_np

    def _compile_for(self, batch: int, *, probe: bool = False):
        if batch in self._compiled:
            return self._compiled[batch]

        import coremltools as ct
        from coremltools.converters.mil import Builder as mb

        weight, bias = self._arrays()
        kwargs = {"weight": weight}
        if bias is not None:
            kwargs["bias"] = bias

        @mb.program(input_specs=[mb.TensorSpec(shape=(batch, self.in_features))])
        def program(x):
            return mb.linear(x=x, **kwargs)

        units = self._units()
        model = ct.convert(
            program,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=(ct.precision.FLOAT32
                               if self.precision == "float32"
                               else ct.precision.FLOAT16),
            compute_units=units,
        )
        rows = compute_plan(model, compute_units=units)
        self._report.setdefault("plans", []).append({
            "batch": batch, "probe": bool(probe), "rows": rows,
        })
        self._compiled[batch] = model
        if not probe:
            self._say_what_ran(batch, rows)
        return model

    def _say_what_ran(self, batch, rows):
        """Warn when CoreML did not put this shape on the Neural Engine.

        Only for a shape the **caller** asked for. `_compile_model`'s eager
        compile uses batch 1 because that is the shape it can know without a
        prompt, and warning that CoreML preferred the CPU for a shape nobody
        requested is noise -- docs/graph/NPU2.md §2.1 measured that a small
        program legitimately goes to the CPU at float16. That probe's plan is
        still recorded; it is only the warning that waits for a real shape.

        The unreachable case -- `precision="float32"`, where the unit is not in
        the supported column at all -- is warned at `to()` time instead, by
        `_compile_model`, because it is decided by the precision and not by the
        shape.
        """
        import warnings

        compute = computes(rows)
        if not compute:
            return
        preferred = sorted({row["preferred"] for row in compute})
        if preferred == ["NeuralEngine"]:
            return
        if not any("NeuralEngine" in row["supported"] for row in compute):
            # Already said once, at `to()`, and it does not change with shape.
            return
        key = ("preferred", batch, tuple(preferred))
        said = self._report.setdefault("_said", [])
        if key in said:
            return
        said.append(key)
        warnings.warn(
            f"torchnative coreml: at batch {batch} the Neural Engine is in "
            f"CoreML's supported set for this program and is NOT what it "
            f"preferred -- MLComputePlan says {preferred}. Nothing is wrong "
            f"with the lowering; CoreML weighs dispatch cost against work and "
            f"below some amount of work the CPU wins (docs/graph/NPU2.md "
            f"§2.1), so reaching the unit is a property of the model's size. "
            f"This is said rather than left silent because "
            f"`to(torchnative.device.npu)` succeeded and a caller would "
            f"otherwise believe the Neural Engine ran it. The per-operation "
            f"plan is on the model as `.torchnative_offload['plans']`.",
            UserWarning,
            stacklevel=3,
        )

    def forward(self, x):
        import numpy as np

        torch = _torch()
        shape = tuple(int(d) for d in x.shape)
        if shape[-1] != self.in_features:
            raise CoreMLUnsupported(
                f"torchnative coreml: input last dimension {shape[-1]} does "
                f"not match in_features={self.in_features}."
            )
        batch = 1
        for dim in shape[:-1]:
            batch *= dim
        model = self._compile_for(batch)
        feed = np.asarray(x.detach().tolist(), dtype=np.float32).reshape(
            batch, self.in_features)
        produced = list(model.predict(
            {model.get_spec().description.input[0].name: feed}).values())[0]
        result = torch.tensor(
            np.asarray(produced, dtype=np.float32).reshape(
                *shape[:-1], self.out_features).tolist())
        return result.to(x.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, precision={self.precision!r}"
        )


def _compile_model(model, *, precision: str = "float16", compute_units=None,
                   predicate=None, eager: bool = True, progress=None):
    """Swap every `torch.nn.Linear` in `model` for a `_CoreMLLinear`. In place.

    Returns `(model, report)`, the same contract `intelnpu._compile_model` has,
    and the report is delivered onwards by `device._module_to` in the same two
    ways: as `model.torchnative_offload` and as a `UserWarning` when the
    offload is partial.

    **Linear only, as on the Intel arm.** Conv2d has a MIL lowering here
    (`supported_ops()` lists `aten.convolution.default`) and is still left on
    the CPU, because a conv leaf's program cannot be built without its input's
    spatial dimensions and those are not knowable at `to()` time. A Linear's
    is: batch is the only free dimension, which is why `_NPULinear` can compile
    at batch 1 too. Widening this to conv needs a shape source, not a bigger
    table, and it is left named rather than half-done.

    **Zero leaves lowered raises.** Returning an untouched model with a success
    message is the silent CPU fallback this path exists to prevent.

    **`eager=True` compiles the batch-1 program for every leaf before
    returning.** As on the Intel arm, that is a change of placement rather than
    a speed-up: it makes "coremltools cannot build this" a failure of *this
    call* instead of a surprise several layers into a `generate()` loop. It is
    marked `probe: True` in the report and does not warn about which unit
    CoreML preferred, because batch 1 is a shape this function chose.
    """
    torch = _torch()
    import warnings

    _check_precision(precision)
    report = {
        "backend": "coreml",
        "precision": precision,
        "plans": [],
        "_said": [],
    }
    swapped, left, skipped = [], {}, []
    moved_parameters = 0

    def walk(parent, prefix):
        nonlocal moved_parameters
        for name, child in list(parent.named_children()):
            path = f"{prefix}{name}"
            if isinstance(child, torch.nn.Linear):
                if predicate is not None and not predicate(path, child):
                    skipped.append((path, "excluded by predicate"))
                    continue
                numel = child.weight.numel() + (
                    child.bias.numel() if child.bias is not None else 0
                )
                try:
                    lowered = _CoreMLLinear.from_torch(
                        child, precision=precision,
                        compute_units=compute_units, report=report)
                except CoreMLUnsupported as exc:
                    skipped.append((
                        path,
                        f"{type(child).__name__}"
                        f"(out_features={child.out_features}, "
                        f"in_features={child.in_features}) stays on the CPU: "
                        f"{str(exc).split(chr(10))[0]}",
                    ))
                    continue
                parent.add_module(name, lowered)
                swapped.append(path)
                moved_parameters += numel
                continue
            grandchildren = list(child.named_children())
            if not grandchildren:
                left[type(child).__name__] = left.get(type(child).__name__, 0) + 1
            else:
                walk(child, f"{path}.")

    walk(model, "")
    if not swapped:
        raise CoreMLUnsupported(
            f"torchnative coreml: nothing was lowered, so nothing runs on the "
            f"Neural Engine. Leaf module types found: "
            f"{sorted(left) or ['<none>']}. {len(skipped)} Linear(s) were "
            f"skipped: {skipped[:4]}. Returning the model unchanged with a "
            f"success message would be the silent CPU fallback this path "
            f"exists to prevent -- docs/graph/NPU2.md §1 is that exact failure, "
            f"found only by reading MLComputePlan. torch.nn.Linear is the only "
            f"leaf lowered at this stage; see `_compile_model` for why conv is "
            f"named rather than half-done."
        )

    def leaf_at(path):
        node = model
        for part in path.split("."):
            node = node[int(part)] if part.isdigit() else getattr(node, part)
        return node

    leaves = [leaf_at(path) for path in swapped]

    # The first eager compile is NOT caught by name, unlike every one after it.
    # It is different in kind: it is the assertion that coremltools on this host
    # can build and compile what this module emits at all. Absorbing it into a
    # report would turn "CoreML is not usable here" into a model the caller
    # believes is offloaded.
    eager_failed = []
    if eager:
        leaves[0]._compile_for(1, probe=True)
        if progress is not None:
            progress(1, len(leaves), swapped[0])
        for index, (path, leaf) in enumerate(zip(swapped[1:], leaves[1:]), 2):
            try:
                leaf._compile_for(1, probe=True)
            except Exception as exc:  # noqa: BLE001
                eager_failed.append((path, f"{type(exc).__name__}: {exc}"))
            if progress is not None:
                progress(index, len(leaves), path)

    total = sum(p.numel() for p in model.parameters())
    report.update({
        "swapped": swapped,
        "skipped": skipped,
        "left_on_cpu": dict(sorted(left.items())),
        "eager_failed": eager_failed,
        "fully_offloaded": not left and not skipped and not eager_failed,
        "parameters_moved": moved_parameters,
        "parameters_total": total,
        "fraction_moved": moved_parameters / total if total else 0.0,
    })

    # The unreachable case, said once, at the moment the caller asked. This one
    # does not depend on the shape: at float32 the Neural Engine is not in
    # CoreML's supported column for any of these programs at any size, so no
    # `compute_units` setting can reach it. A caller who wrote
    # `to(device.npu, precision="float32")` and got a model that silently runs
    # on the CPU is exactly docs/graph/NPU2.md §1.
    probe_rows = [row for plan in report["plans"] for row in computes(plan["rows"])]
    if probe_rows and not any(
            "NeuralEngine" in row["supported"] for row in probe_rows):
        warnings.warn(
            f"torchnative coreml: lowered at precision={precision!r}, and at "
            f"that precision the Neural Engine is NOT in CoreML's supported "
            f"column for this program -- MLComputePlan offers "
            f"{sorted({d for row in probe_rows for d in row['supported']})}, "
            f"so no compute_units setting can reach the unit. This model runs "
            f"through CoreML on the CPU or GPU. That is the trade this "
            f"precision buys: it agrees with DecomposedTrace.replay at 2e-05, "
            f"where float16 agrees at about 1e-03. Use "
            f"`to(torchnative.device.npu)` (float16) to reach the unit, and "
            f"see docs/graph/NPU2.md §1.1.",
            UserWarning,
            stacklevel=4,
        )
    return model, report
