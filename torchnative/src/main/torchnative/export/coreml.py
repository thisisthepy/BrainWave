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
    "coreml_ops",
    "coremltools_available",
    "compile_model",
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


def compile_model(trace, *, fold: bool = True, float32: bool = True):
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
    """
    import coremltools as ct

    program, names, emitted = to_mil_program(trace, fold=fold)
    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=(
            ct.precision.FLOAT32 if float32 else ct.precision.FLOAT16
        ),
    )
    return model, names, emitted


def verify(trace, inputs, *, tolerance: float = 2e-5, fold: bool = True,
           float32: bool = True) -> dict:
    """Compile, **run**, and compare against `DecomposedTrace.replay`.

    This is the executed claim. Both sides are computed on the same inputs:
    the CoreML side by the operating system's runtime over the compiled
    `.mlpackage`, the reference side by our own graph through
    `torch._C._aten_dispatch`. Agreement is evidence about the MIL lowering,
    not about two libraries happening to implement an op the same way, for the
    reason docs/graph/CAPTURE.md §3 gives about replaying through the door capture
    recorded at.
    """
    import numpy as np

    model, names, emitted = compile_model(trace, fold=fold, float32=float32)
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
