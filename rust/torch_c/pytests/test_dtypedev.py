"""The (dtype x device) matrix, and the two names that told the world Metal
was absent while it was computing on Metal.

docs/numerics/DTYPEDEV.md is the document; this file is what holds it down.

**The defect this round started from.** On this machine, with this shim:

    >>> a = torch.randn(64, 64).to("mps"); b = torch.randn(64, 64).to("mps")
    >>> (a @ b).device
    device(type='mps', index=0)
    >>> torch.backends.mps.is_available()
    False
    >>> torch.backends.mps.is_built()
    False

Both were constants. `torch.backends.mps.is_available()` is the gate
transformers and accelerate branch on, so nothing on top of this shim would
ever *select* the device it was already using. `_install_mps_backend` in
`bootstrap.py` is the fix and carries the reasoning; the tests here are what
makes it impossible to put the constant back quietly.

**Three grades, and this file says which it is establishing.** The project's
standard is *agrees*, not *runs*:

  builds    the cell can be constructed
  reaches   an operator on it returns something rather than refusing
  agrees    what it returns is upstream's answer

A dtype that produces numbers on a device and disagrees with upstream is worse
than one that refuses, because it is silent. So the matrix test below grades
at *agrees* -- every value the shim computes, on either device, is compared
against upstream's value for the same computation -- and the refusals are
frozen as a set, so a cell that starts refusing where it used to compute is as
red as one that starts computing where it used to refuse.

**What this file structurally cannot see.** It runs on an arm64 Mac with a
Metal GPU and no CUDA, no NPU and no Android device. The `cuda`, `vulkan` and
`npu` columns of the matrix are not measured here and are not asserted here;
docs/numerics/DTYPEDEV.md section 5 says per row what hardware would settle
them. Every `mps` test skips by name, saying why, on a machine with no Metal.

Nullifications this file is meant to catch, each verified by making the break
and watching it go red -- see docs/numerics/DTYPEDEV.md section 6 for what was
seen:

* `_mps_is_available` back to a constant  -> test_mps_is_available_is_a_probe_and_not_a_constant
* `_has_mps` back to a constant `False`   -> test_mps_is_built_is_the_artefacts_own_answer
* `metal_dtype_gate` returning `Ok(())`   -> test_float64_refuses_by_name_on_every_road_onto_metal
* a wrong value on either device          -> test_the_dtype_device_matrix_agrees_with_upstream
"""

import json
import os
import subprocess
import sys

from test_shim import _C

try:
    import torch as _upstream_torch
except ImportError:  # pragma: no cover
    _upstream_torch = None

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


# ---------------------------------------------------------------------------
# Reaching the shim as `torch`, which is the only way `torch.backends.mps` exists
# ---------------------------------------------------------------------------
#
# `_C` above is the standalone artefact; `torch.backends.mps.is_available()` is
# a *vendored-tree* road -- `torch/backends/mps/__init__.py:30` reads
# `torch._C._mps_is_available()`, and there is no `torch` package around the
# standalone `_C`. So the backends half of this file runs in a subprocess with
# `torchnative/src/main` on PYTHONPATH, the same shape `test_shim.py`'s
# checkpoint and device-road sections use and for the same reason. One launch
# answers every question below; it returns one JSON object on stdout.


def _vendor_available():
    return os.path.isfile(_VENDOR_SHIM)


_BACKENDS_SCRIPT = r"""
import json, sys
import torch

out = {}
out["probe"] = torch._C._mps_probe()
out["is_built"] = torch.backends.mps.is_built()
out["is_available"] = torch.backends.mps.is_available()
out["has_mps"] = torch._C._has_mps
out["mps_is_available_is_callable"] = callable(torch._C._mps_is_available)
out["device_count"] = torch.mps.device_count()

# The independent measurement: allocate on Metal and look at where it landed.
# Deliberately not `torch._C._mps_is_available()` in another spelling -- a
# probe checked against itself is not checked.
try:
    t = torch.empty(2, 2, device="mps")
    out["really"] = (t.device.type == "mps")
    out["really_error"] = None
except BaseException as e:
    out["really"] = False
    out["really_error"] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0])

# The coupling: `torch/mps/__init__.py:67` guards `_mps_get_default_generator`
# behind `_has_mps`, and `torch.manual_seed` reaches it unconditionally. While
# `_has_mps` was False that guard was the only thing holding it up.
try:
    torch.manual_seed(0)
    out["manual_seed"] = None
except BaseException as e:
    out["manual_seed"] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0])

# And that the generator behind it is the one stream this build has.
try:
    out["mps_generator_is_default"] = (
        torch._C._mps_get_default_generator() is torch.default_generator
    )
except BaseException as e:
    out["mps_generator_is_default"] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0])

# It really computed on Metal: the opening line of the defect, as an assertion.
try:
    a = torch.randn(8, 8).to("mps")
    out["matmul_device"] = str((a @ a).device)
except BaseException as e:
    out["matmul_device"] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0])

# float64 on Metal, by each of the three roads onto the device.
out["f64"] = {}
for name, thunk in (
    ("cast_then_move", lambda: torch.zeros(2).double().to("mps")),
    ("move_then_cast", lambda: torch.zeros(2).to("mps").to(torch.float64)),
    ("factory", lambda: torch.zeros(2, dtype=torch.float64, device="mps")),
):
    try:
        out["f64"][name] = "made a %s tensor on %s" % (thunk().dtype, "mps")
    except BaseException as e:
        out["f64"][name] = "%s: %s" % (type(e).__name__, str(e).splitlines()[0])

json.dump(out, sys.stdout)
"""

_backends_cache = {}


def _backends():
    if "r" not in _backends_cache:
        env = dict(os.environ)
        env["PYTHONPATH"] = _VENDOR_DIR
        # Not a workaround -- upstream ships this switch for builds without
        # libtorch_global_deps, which is exactly this one (VENDOR.md:181).
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", _BACKENDS_SCRIPT],
            capture_output=True, text=True, env=env, timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "mps backends subprocess exited %d\n--- stdout ---\n%s\n--- stderr ---\n%s"
                % (proc.returncode, proc.stdout, proc.stderr)
            )
        _backends_cache["r"] = json.loads(proc.stdout)
    return _backends_cache["r"]


# The fact `is_built()` is supposed to be reporting, derived here from the
# interpreter rather than from the artefact, so the assertion has an
# independent left-hand side. `_shim_target()` is what the artefact was built
# for; `sys.platform` is what this interpreter is running on. They agree on
# every machine that can run this file at all, and the point of reading it
# this way is that a nullified `_has_mps` cannot move it.
def _this_is_an_apple_build():
    return sys.platform == "darwin"


# ---------------------------------------------------------------------------
# is_built() and is_available() -- two questions, two answers
# ---------------------------------------------------------------------------


def test_mps_probe_answers_built_and_available_separately():
    """They are different claims and upstream says so.

    Upstream's own docstring for `is_built()` reads "Note that this doesn't
    necessarily mean MPS is available; just that if this PyTorch binary were
    run a machine with working MPS drivers and devices, we would be able to use
    it." So a probe that answered one number for both would be wrong even on
    this host, where they happen to agree -- the case that separates them is a
    Mac VM with no GPU passthrough, which this machine is not and so cannot
    demonstrate. What is checkable here is that the two come from different
    places: `built` from `cfg!(target_vendor = "apple")` and `available` from
    an actual `resolve()`.
    """
    if not _vendor_available():
        print("   (skipped: no vendored tree at %s -- run vendor/vendor_torch.sh "
              "then vendor/install_shim.sh)" % _VENDOR_SHIM)
        return
    r = _backends()
    p = r["probe"]
    assert set(p) == {"built", "available", "reason", "error"}, p
    assert isinstance(p["built"], bool), p
    assert isinstance(p["available"], bool), p
    # Necessary and not sufficient, in that direction only.
    if p["available"]:
        assert p["built"], p
    if not p["built"]:
        assert p["reason"] == "not_built", p


def test_mps_is_built_is_the_artefacts_own_answer():
    """`torch.backends.mps.is_built()` reads `torch._C._has_mps`, which was a
    hardcoded `False` in `_BUILD_FLAGS` justified by a comment saying candle's
    `metal` feature was off in `Cargo.toml`. It is not off -- `Cargo.toml`
    enables it for `target_vendor = "apple"`, and `PyDevice::resolve` has
    carried an `mps` arm under that same `#[cfg]` for several rounds.

    The left-hand side here is `sys.platform`, not another shim name, so
    putting the constant back cannot make this pass.
    """
    if not _vendor_available():
        print("   (skipped: no vendored tree at %s)" % _VENDOR_SHIM)
        return
    r = _backends()
    assert r["is_built"] == r["has_mps"], r
    assert r["is_built"] is _this_is_an_apple_build(), (
        "torch.backends.mps.is_built() is %r on sys.platform=%r. is_built() is "
        "the compile-time question -- was Metal built in -- and on an Apple "
        "target the answer is yes. A False here is the constant coming back."
        % (r["is_built"], sys.platform)
    )


def test_mps_is_available_is_a_probe_and_not_a_constant():
    """The load-bearing one, and the whole defect.

    `is_available()` is checked against an **independent allocation**, not
    against `torch._C._mps_is_available()` in another spelling: a probe
    compared with itself is not compared with anything. If the shim can put a
    tensor on Metal, `is_available()` has to say so; if it cannot, it has to
    say that instead. Either direction of disagreement fails here.
    """
    if not _vendor_available():
        print("   (skipped: no vendored tree at %s)" % _VENDOR_SHIM)
        return
    r = _backends()
    assert r["mps_is_available_is_callable"], r
    assert r["is_available"] == r["really"], (
        "torch.backends.mps.is_available() is %r but torch.empty(2, 2, "
        "device='mps') %s. is_available() is the gate transformers and "
        "accelerate branch on, so a False here on a machine that computes on "
        "Metal means nothing will ever select the device it is already using."
        % (r["is_available"],
           "succeeded" if r["really"] else "failed: %s" % r["really_error"])
    )
    # `device_count()` is `int(_has_mps and _mps_is_available())` upstream, so
    # it is the third name that has to move with the other two.
    assert r["device_count"] == int(r["is_built"] and r["is_available"]), r


def test_the_shim_really_computes_on_metal_when_it_says_it_can():
    """The opening line of the defect report, as an assertion. Without this,
    `is_available()` could be made to agree with the allocation by weakening
    both.
    """
    if not _vendor_available():
        print("   (skipped: no vendored tree at %s)" % _VENDOR_SHIM)
        return
    r = _backends()
    if not r["is_available"]:
        print("   (skipped: no mps device on this machine -- %s)" % r["really_error"])
        return
    assert r["matmul_device"].startswith("mps"), r["matmul_device"]


def test_manual_seed_survives_has_mps_being_true():
    """The cost of the honest flag, paid in the same commit.

    `torch/mps/__init__.py:67` is `if not torch._C._has_mps: return` followed
    by `_get_default_mps_generator().manual_seed(seed)`, and
    `torch.manual_seed` reaches it unconditionally through `torch/random.py`.
    While `_has_mps` was `False` that guard was the only thing keeping every
    model script's `torch.manual_seed(0)` off `_mps_get_default_generator`,
    which was an `_Unimplemented`. Measured before the fix: flipping the flag
    alone turns `torch.manual_seed(0)` into `NotImplementedError: not
    implemented in torch._C shim: torch._C._mps_get_default_generator`.
    """
    if not _vendor_available():
        print("   (skipped: no vendored tree at %s)" % _VENDOR_SHIM)
        return
    r = _backends()
    assert r["manual_seed"] is None, (
        "torch.manual_seed(0) raised %s. On a build where torch._C._has_mps is "
        "True this call reaches torch._C._mps_get_default_generator, so the "
        "flag and the generator have to move together." % r["manual_seed"]
    )
    if r["is_built"]:
        assert r["mps_generator_is_default"] is True, (
            "torch._C._mps_get_default_generator() is not torch.default_generator "
            "(%r). This build has one RNG stream (rng.rs) and every mps tensor "
            "with random contents is filled from it, so a second Generator "
            "object here would own a state nothing advances."
            % (r["mps_generator_is_default"],)
        )


# ---------------------------------------------------------------------------
# float64 on Metal -- a cell that must refuse by name
# ---------------------------------------------------------------------------


_F64_REFUSAL = (
    "Cannot convert a MPS Tensor to float64 dtype as the MPS framework "
    "doesn't support float64. Please use float32 instead."
)


def test_float64_refuses_by_name_on_every_road_onto_metal():
    """Metal has no `double`. That is a property of the API -- MSL has no
    64-bit floating type -- not of candle and not of this build, and upstream
    refuses at the boundary with the message above.

    This build did not. `x.double().to("mps")` succeeded, because candle will
    allocate an `F64` Metal buffer, and what came back was a tensor that could
    be cloned and nothing else: `add` died with `Error while loading function:
    badd_f64`, `sum` with `Metal contiguous reduce op Sum F64 not implemented`,
    and even `.to(torch.float32)` -- the documented escape hatch -- with `Metal
    contiguous to_dtype F64 F32 not implemented`. So the object could be made
    and could not be converted back, and every message named an internal candle
    symbol rather than the fact.

    All three roads are checked because the gate has two halves and each covers
    a different one: `metal_dtype_gate` in `PyTensorBase::new` is the one
    nothing can get round, and the call in `aten._to_copy.default` is what
    makes the *message* right for `move_then_cast`, which candle would
    otherwise refuse first with `Metal contiguous to_dtype F32 F64 not
    implemented`.
    """
    if not _vendor_available():
        print("   (skipped: no vendored tree at %s)" % _VENDOR_SHIM)
        return
    r = _backends()
    if not r["is_available"]:
        print("   (skipped: no mps device on this machine -- %s)" % r["really_error"])
        return
    for road in ("cast_then_move", "move_then_cast", "factory"):
        got = r["f64"][road]
        assert got.startswith("TypeError: "), (
            "%s produced %r. float64 on Metal has to refuse by name -- silently "
            "making a tensor the device cannot compute with is the failure mode "
            "docs/graph/NPU2.md is about, in its quieter form: a capability "
            "claim made by construction succeeding." % (road, got)
        )
        assert _F64_REFUSAL in got, (
            "%s refused with %r, which is not upstream's sentence. The refusal "
            "has to name the dtype and the framework, not a candle symbol."
            % (road, got)
        )


def test_no_float64_tensor_can_exist_on_a_metal_device():
    """The invariant behind the three roads, asked of `_C` directly rather than
    through the vendored tree: every dense tensor goes through
    `PyTensorBase::new`, so if that refuses, there is no spelling that produces
    one. Nullifying `metal_dtype_gate` to `Ok(())` makes this red.
    """
    mps = _mps_or_skip("the float64 invariant")
    if mps is None:
        return
    cpu64 = _C._tensor_from_flat([1.0, 2.0], [2], _C.float64)
    try:
        got = _C._aten_dispatch("aten._to_copy.default", cpu64, device=mps)
    except (TypeError, NotImplementedError, RuntimeError) as e:
        assert _F64_REFUSAL in str(e), str(e)
        return
    raise AssertionError(
        "a float64 tensor landed on %s. Metal has no double; every arithmetic "
        "kernel refuses it, so this object can only be cloned." % (got.device,)
    )


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------


def _mps_or_skip(what):
    """A live `mps` device, or None having said why not.

    Same shape as `test_mpsfwd._mps_or_skip` and deliberately not imported from
    it: the skip line is the thing this file promises to keep truthful
    (docs/devices/VULKAN3.md §6.1 is what a lying skip line cost), so it says
    which file skipped.
    """
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except (NotImplementedError, RuntimeError) as e:
        print("   (skipped %s: no mps device on this machine -- %s)"
              % (what, str(e).splitlines()[0]))
        return None
    return _C.device("mps")


# The dtypes this build can put in candle storage, plus the ones it cannot --
# both halves, because a matrix that listed only what works cannot tell a
# deliberate refusal from a dtype somebody forgot.
#
# The spellings are torch's, and every one of them exists on both `torch` and
# `_C` (tools/golden/dtypes.py's rule). `complex64` and friends are absent on
# purpose: this shim holds a complex tensor as a pair of real tensors and
# reading its values refuses by name, which is docs/numerics/ COMPLEX ground
# and a different axis from this one.
_STORABLE = (
    "float64", "float32", "float16", "bfloat16",
    "int64", "int32", "int16", "uint8", "uint32", "bool",
    "float8_e4m3fn",
)
_NOT_STORABLE = (
    "int8", "uint16", "uint64",
    "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu",
    "float4_e2m1fn_x2",
    "qint8", "quint8", "qint32",
)

# A 2x2 of 1..4, which every dtype here can hold exactly -- including
# `float8_e4m3fn` (whose grid has 1, 2, 3 and 4 on it) and `bool` (where all
# four are True). An input a dtype cannot represent would make a disagreement
# about the *input* look like a disagreement about the operator.
_VALUES = [1.0, 2.0, 3.0, 4.0]
_SHAPE = [2, 2]

_OPS = {
    "add": lambda t, m: t + t,
    "sub": lambda t, m: t - t,
    "mul": lambda t, m: t * t,
    "div": lambda t, m: t / t,
    "matmul": lambda t, m: t @ t,
    "sum": lambda t, m: t.sum(),
    "mean": lambda t, m: t.mean(),
    "max": lambda t, m: t.max(),
    "neg": lambda t, m: -t,
    "abs": lambda t, m: t.abs(),
    "exp": lambda t, m: t.exp(),
    "sqrt": lambda t, m: t.sqrt(),
    "pow2": lambda t, m: t.pow(2),
    "softmax": lambda t, m: t.softmax(-1),
    "cumsum": lambda t, m: t.cumsum(0),
    "argmax": lambda t, m: t.argmax(),
    "eq": lambda t, m: t == t,
    "where": lambda t, m: m.where(t == t, t, t),
    "cat": lambda t, m: m.cat([t, t], 0),
    "clone": lambda t, m: t.clone(),
    "index0": lambda t, m: t[0],
    "to_f32": lambda t, m: t.to(m.float32),
    "transpose_contig": lambda t, m: t.t().contiguous(),
}


def _build(mod, name, device):
    dt = getattr(mod, name)
    if mod is _C:
        t = _C._tensor_from_flat(_VALUES, _SHAPE, _C.float32).to(dt)
    else:
        t = mod.tensor(_VALUES).reshape(_SHAPE).to(dt)
    return t if device == "cpu" else t.to(device)


def _read(mod, t):
    """Values, as strings, in a form both sides spell the same way.

    Everything is read back on the host and widened to float64 before
    `tolist()`, so `1` and `1.0` do not look like a disagreement, and so an
    integer dtype's values survive exactly (every value here is under 2**53).
    """
    h = t.cpu() if str(t.device) != "cpu" else t
    return [float(v) for v in h.to(mod.float64).flatten().tolist()]


def _cell(mod, name, device, op):
    """`("value", dtype, [floats])` or `("refuse", ExceptionName)`."""
    try:
        t = _build(mod, name, device)
        out = _OPS[op](t, mod)
        return ("value", str(out.dtype).replace("torch.", ""), _read(mod, out))
    except BaseException as e:  # noqa: BLE001
        return ("refuse", type(e).__name__)


def _close(a, b):
    # One ULP at magnitude ~1 for float32, which is the widest dtype any cell
    # here computes in on Metal, applied relatively. tools/golden/dtypes.py
    # owns the per-dtype table for the CPU golden sweep; this is the one number
    # that covers the whole of this much smaller matrix, chosen loose enough
    # that a rounding-direction difference passes and tight enough that a wrong
    # operator does not.
    return abs(a - b) <= 1e-4 + 1e-4 * max(abs(a), abs(b))


def test_the_dtype_device_matrix_agrees_with_upstream():
    """Grade: **agrees**. Every value the shim computes, on either device, is
    upstream's value for the same computation.

    Upstream is asked on the **cpu** for every cell, including the `mps` ones.
    That is on purpose: upstream refuses several of these on Metal that it
    computes on the CPU (`float64` outright, `uint32` arithmetic with `Failed
    to create function state object for: add_dense_uint_uint`), and the
    question here is whether the shim's number is *right*, not whether upstream
    would have produced it on that device. A cell where the shim computes and
    upstream refuses everywhere is reported by the frozen set below instead.

    This is the test that would catch a silently wrong answer, which is the
    worst outcome available in this matrix -- worse than a refusal, because a
    refusal is loud.
    """
    if _upstream_torch is None:
        print("   (skipped: no upstream torch in this interpreter -- "
              "the oracle half of this test cannot run)")
        return
    mps = _mps_or_skip("the mps half of the matrix")
    devices = ["cpu"] + (["mps"] if mps is not None else [])
    wrong = []
    graded = 0
    for name in _STORABLE:
        for device in devices:
            for op in sorted(_OPS):
                got = _cell(_C, name, device, op)
                if got[0] != "value":
                    continue
                want = _cell(_upstream_torch, name, "cpu", op)
                if want[0] != "value":
                    continue
                graded += 1
                if got[1] != want[1] or len(got[2]) != len(want[2]) or not all(
                    _close(a, b) for a, b in zip(got[2], want[2])
                ):
                    wrong.append((name, device, op, got, want))
    assert graded > 200, graded
    assert not wrong, (
        "%d (dtype, device, operator) cells compute a value that is not "
        "upstream's:\n%s" % (len(wrong), "\n".join(
            "  %s on %s: %s -> %s, upstream %s" % (n, d, o, g, w)
            for n, d, o, g, w in wrong))
    )


# The frozen shape of the matrix: which cells produce a value at all.
#
# Recorded rather than derived, because both directions of change are a
# regression worth a red test and a derived expectation catches neither. A cell
# that stops computing is a lost capability; a cell that starts computing where
# it used to refuse is a capability that arrived without anybody checking it
# against upstream -- and it will be checked, by the test above, as soon as
# this list is updated to admit it.
#
# The counts, per docs/numerics/DTYPEDEV.md section 3. `cpu` is where this
# build lives and it shows: every storable dtype reaches most of the operator
# set there, while on `mps` the answer depends almost entirely on whether
# candle ships a Metal kernel for that candle dtype.
_REACHES = {}


def _reaches(device):
    key = ("reaches", device)
    if key not in _REACHES:
        _REACHES[key] = {
            "%s|%s" % (name, op)
            for name in _STORABLE
            for op in sorted(_OPS)
            if _cell(_C, name, device, op)[0] == "value"
        }
    return _REACHES[key]


def test_every_dtype_this_build_cannot_store_refuses_by_name():
    """Grade: **refuses by name**. Eleven of the dtypes `torch` publishes have
    no candle storage here, and the requirement for those is not that they work
    -- it is that asking says so, naming the dtype, rather than silently
    downcasting to one that does.

    `int8` is the one on this list most likely to surprise: upstream computes
    with it on both cpu and mps, and this build cannot hold it at all.
    docs/numerics/INT8.md and INT8B.md are that ground; what is checked here is
    only that the refusal names the dtype.
    """
    for name in _NOT_STORABLE:
        dt = getattr(_C, name, None)
        assert dt is not None, (
            "_C has no %s. tools/golden/dtypes.py's rule is that every dtype "
            "name exists on both torch and _C with the same spelling." % name)
        try:
            _build(_C, name, "cpu")
        except (NotImplementedError, RuntimeError, TypeError) as e:
            assert name in str(e), (
                "%s refused with %r, which does not name the dtype. A refusal "
                "that does not say what was refused sends the reader to the "
                "wrong place." % (name, str(e).splitlines()[0]))
            continue
        raise AssertionError(
            "%s built a tensor on the cpu. It is on the not-storable list, so "
            "either candle grew the dtype (move it to _STORABLE and let the "
            "agreement test grade it) or it is being silently stored as "
            "something else." % name)


# Recorded 2026-09-12 on an arm64 Mac (Metal, no CUDA, no NPU), against the
# artefact built from this tree. docs/numerics/DTYPEDEV.md section 3 is the
# reading of it.
_CPU_REACHES = {
    "bfloat16|abs",
    "bfloat16|add",
    "bfloat16|argmax",
    "bfloat16|clone",
    "bfloat16|cumsum",
    "bfloat16|div",
    "bfloat16|eq",
    "bfloat16|exp",
    "bfloat16|index0",
    "bfloat16|matmul",
    "bfloat16|max",
    "bfloat16|mean",
    "bfloat16|mul",
    "bfloat16|neg",
    "bfloat16|pow2",
    "bfloat16|softmax",
    "bfloat16|sqrt",
    "bfloat16|sub",
    "bfloat16|sum",
    "bfloat16|to_f32",
    "bfloat16|transpose_contig",
    "bool|argmax",
    "bool|clone",
    "bool|cumsum",
    "bool|eq",
    "bool|exp",
    "bool|index0",
    "bool|max",
    "bool|mul",
    "bool|sqrt",
    "bool|sum",
    "bool|to_f32",
    "bool|transpose_contig",
    "float16|abs",
    "float16|add",
    "float16|argmax",
    "float16|clone",
    "float16|cumsum",
    "float16|div",
    "float16|eq",
    "float16|exp",
    "float16|index0",
    "float16|matmul",
    "float16|max",
    "float16|mean",
    "float16|mul",
    "float16|neg",
    "float16|pow2",
    "float16|softmax",
    "float16|sqrt",
    "float16|sub",
    "float16|sum",
    "float16|to_f32",
    "float16|transpose_contig",
    "float32|abs",
    "float32|add",
    "float32|argmax",
    "float32|clone",
    "float32|cumsum",
    "float32|div",
    "float32|eq",
    "float32|exp",
    "float32|index0",
    "float32|matmul",
    "float32|max",
    "float32|mean",
    "float32|mul",
    "float32|neg",
    "float32|pow2",
    "float32|softmax",
    "float32|sqrt",
    "float32|sub",
    "float32|sum",
    "float32|to_f32",
    "float32|transpose_contig",
    "float64|abs",
    "float64|add",
    "float64|argmax",
    "float64|clone",
    "float64|cumsum",
    "float64|div",
    "float64|eq",
    "float64|exp",
    "float64|index0",
    "float64|matmul",
    "float64|max",
    "float64|mean",
    "float64|mul",
    "float64|neg",
    "float64|pow2",
    "float64|softmax",
    "float64|sqrt",
    "float64|sub",
    "float64|sum",
    "float64|to_f32",
    "float64|transpose_contig",
    "float8_e4m3fn|abs",
    "float8_e4m3fn|clone",
    "float8_e4m3fn|eq",
    "float8_e4m3fn|index0",
    "float8_e4m3fn|matmul",
    "float8_e4m3fn|mul",
    "float8_e4m3fn|to_f32",
    "int16|abs",
    "int16|add",
    "int16|argmax",
    "int16|clone",
    "int16|cumsum",
    "int16|div",
    "int16|eq",
    "int16|exp",
    "int16|index0",
    "int16|max",
    "int16|mul",
    "int16|neg",
    "int16|pow2",
    "int16|sqrt",
    "int16|sub",
    "int16|sum",
    "int16|to_f32",
    "int16|transpose_contig",
    "int32|abs",
    "int32|add",
    "int32|argmax",
    "int32|clone",
    "int32|cumsum",
    "int32|div",
    "int32|eq",
    "int32|exp",
    "int32|index0",
    "int32|max",
    "int32|mul",
    "int32|neg",
    "int32|pow2",
    "int32|sqrt",
    "int32|sub",
    "int32|sum",
    "int32|to_f32",
    "int32|transpose_contig",
    "int64|abs",
    "int64|add",
    "int64|argmax",
    "int64|clone",
    "int64|cumsum",
    "int64|div",
    "int64|eq",
    "int64|exp",
    "int64|index0",
    "int64|max",
    "int64|mul",
    "int64|neg",
    "int64|pow2",
    "int64|sqrt",
    "int64|sub",
    "int64|sum",
    "int64|to_f32",
    "int64|transpose_contig",
    "uint32|abs",
    "uint32|add",
    "uint32|argmax",
    "uint32|clone",
    "uint32|cumsum",
    "uint32|div",
    "uint32|eq",
    "uint32|exp",
    "uint32|index0",
    "uint32|max",
    "uint32|mul",
    "uint32|pow2",
    "uint32|sqrt",
    "uint32|sub",
    "uint32|sum",
    "uint32|to_f32",
    "uint32|transpose_contig",
    "uint8|abs",
    "uint8|add",
    "uint8|argmax",
    "uint8|clone",
    "uint8|cumsum",
    "uint8|div",
    "uint8|eq",
    "uint8|exp",
    "uint8|index0",
    "uint8|max",
    "uint8|mul",
    "uint8|neg",
    "uint8|pow2",
    "uint8|sqrt",
    "uint8|sub",
    "uint8|sum",
    "uint8|to_f32",
    "uint8|transpose_contig",
}

# The Metal column. Two readings worth carrying: `float64` appears **nowhere**
# -- it refuses at the boundary now, by name, which is the cell this round
# closed -- and `int16`/`int32` reach only four operators each where `int64`
# reaches eighteen, because candle ships Metal kernels for I64 and U32 and
# almost none for I16/I32. docs/numerics/DTYPEDEV.md section 4 lists what
# closing that would take.
_MPS_REACHES = {
    "bfloat16|add",
    "bfloat16|clone",
    "bfloat16|div",
    "bfloat16|exp",
    "bfloat16|index0",
    "bfloat16|matmul",
    "bfloat16|mean",
    "bfloat16|mul",
    "bfloat16|neg",
    "bfloat16|softmax",
    "bfloat16|sqrt",
    "bfloat16|sub",
    "bfloat16|sum",
    "bfloat16|to_f32",
    "bfloat16|transpose_contig",
    "bool|clone",
    "bool|cumsum",
    "bool|eq",
    "bool|exp",
    "bool|index0",
    "bool|mul",
    "bool|sqrt",
    "bool|sum",
    "bool|to_f32",
    "bool|transpose_contig",
    "float16|add",
    "float16|clone",
    "float16|div",
    "float16|exp",
    "float16|index0",
    "float16|matmul",
    "float16|mean",
    "float16|mul",
    "float16|neg",
    "float16|softmax",
    "float16|sqrt",
    "float16|sub",
    "float16|sum",
    "float16|to_f32",
    "float16|transpose_contig",
    "float32|add",
    "float32|clone",
    "float32|div",
    "float32|exp",
    "float32|index0",
    "float32|matmul",
    "float32|mean",
    "float32|mul",
    "float32|neg",
    "float32|pow2",
    "float32|softmax",
    "float32|sqrt",
    "float32|sub",
    "float32|sum",
    "float32|to_f32",
    "float32|transpose_contig",
    "float8_e4m3fn|clone",
    "float8_e4m3fn|index0",
    "int16|clone",
    "int16|index0",
    "int32|clone",
    "int32|index0",
    "int64|add",
    "int64|clone",
    "int64|cumsum",
    "int64|div",
    "int64|eq",
    "int64|exp",
    "int64|index0",
    "int64|mul",
    "int64|neg",
    "int64|pow2",
    "int64|sqrt",
    "int64|sub",
    "int64|sum",
    "int64|to_f32",
    "int64|transpose_contig",
    "uint32|add",
    "uint32|clone",
    "uint32|cumsum",
    "uint32|div",
    "uint32|eq",
    "uint32|exp",
    "uint32|index0",
    "uint32|mul",
    "uint32|pow2",
    "uint32|sqrt",
    "uint32|sub",
    "uint32|sum",
    "uint32|to_f32",
    "uint32|transpose_contig",
    "uint8|add",
    "uint8|clone",
    "uint8|cumsum",
    "uint8|div",
    "uint8|eq",
    "uint8|exp",
    "uint8|index0",
    "uint8|mul",
    "uint8|neg",
    "uint8|pow2",
    "uint8|sqrt",
    "uint8|sub",
    "uint8|sum",
    "uint8|to_f32",
    "uint8|transpose_contig",
}


def test_the_cpu_column_reaches_what_it_is_recorded_as_reaching():
    """Grade: **reaches**, frozen. See `_CPU_REACHES` for why both directions
    of change are red.
    """
    got = _reaches("cpu")
    assert got == _CPU_REACHES, _diff("cpu", got, _CPU_REACHES)


def test_the_mps_column_reaches_what_it_is_recorded_as_reaching():
    """Grade: **reaches**, frozen -- and the column where the recorded set is
    doing the most work, because what a dtype can do on Metal here is decided
    by whether candle ships a kernel for it, which moves when candle moves.
    """
    if _mps_or_skip("the recorded mps column") is None:
        return
    got = _reaches("mps")
    assert got == _MPS_REACHES, _diff("mps", got, _MPS_REACHES)


def _diff(device, got, want):
    gained = sorted(got - want)
    lost = sorted(want - got)
    return (
        "the %s column is not what docs/numerics/DTYPEDEV.md records.\n"
        "  now computes and did not (%d): %s\n"
        "  no longer computes (%d): %s\n"
        "A gain is not automatically good: it is a cell nothing has yet "
        "compared against upstream. Add it here and "
        "test_the_dtype_device_matrix_agrees_with_upstream will grade it."
        % (device, len(gained), gained, len(lost), lost)
    )


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
