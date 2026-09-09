"""One OpenVINO Core for a whole model, and the compiles moved off the hot path.

The user's report: `Qwen3-4B.to(device.npu)` lowers 252 `nn.Linear` leaves, and
`generate()` then stalls mid-flight because each leaf compiles a static-shape IR
lazily, on its first forward, for each of the two shapes `generate()` uses (the
prompt length, then 1 per token with a KV cache). 504 driver compiles land
*inside* the first two tokens.

Two separate defects hide behind that one symptom, and this file separates them
because the fixes are not the same kind of thing.

**Defect one, and it is not a micro-optimisation.** `_NPULinear._compile_for`
did `self._ov = OpenVINO(self.library)` per leaf. A 252-leaf model therefore
constructed **252 `ov::Core` objects**, each one dlopening and initialising the
whole plugin set, each running `ov_core_get_available_devices` (which enumerates
and initialises every plugin present, NPU and GPU included), and each resolving
and `makedirs`-ing the compile cache directory. That is 252 copies of the most
expensive object in OpenVINO's API, and none of them shares the in-process cache
guard with any other -- so the model-cache serialisation that OpenVINO provides
*within one Core* was not even reaching across the model. `test_a_252_leaf_model
_builds_exactly_one_openvino_core` is the whole of the fix, measured by counting
constructions.

**Defect two is a question, and this round answers it "no".** Whether those
compiles can run concurrently is not settled by wanting them to. Four things
have to hold, and `docs/devices/NPUPAR.md` records what was found for each with
its source. Two are unverified for lack of any vendor statement, one bounds the
speedup, and one is safe only within a process. So **no threads ship**, and
`test_the_compile_path_still_spawns_no_threads` is the standing check that none
appeared later without the document being revisited.

**What ships instead is eager compilation**, which addresses the actual
complaint -- the stall's *placement*, not its total cost. With one shared Core
the decode-shape (batch=1) compile for every leaf happens inside
`to(device.npu)`, where the user asked for it and can watch it, instead of
inside the first generated token where it looks like a hang. It is reportable
(`progress=`), and a leaf that will not compile is **named in the report and
left on the lazy path** rather than taking the model down -- the same epistemics
`_compile_model` already applies to an oversized leaf.

**What this file is evidence about.** Construction counts, call ordering,
report contents and the absence of threads -- all of it above the OpenVINO
boundary, all of it against a fake. There is no Intel NPU and no OpenVINO on
this host (`library_candidates` refuses on darwin by design), so **nothing here
is evidence that a compile got faster, or that anything ran on an NPU.** The
one number this file does establish about the hardware path is the count of
`ov::Core` constructions, and that count is real because it is the real
`_compile_model` walk driving the real `_NPULinear` -- only the runtime is
faked.
"""

import os
import sys
import threading

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")
sys.path.insert(0, _VENDOR_DIR)

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from torchnative.export import intelnpu  # noqa: E402
from torchnative.export.intelnpu import (  # noqa: E402
    IntelNPUUnavailable,
    IntelNPUUnsupported,
    _compile_model,
    _NPULinear,
)

_SOURCE_PATH = os.path.join(
    _ROOT, "torchnative", "src", "main", "torchnative", "export", "intelnpu.py"
)
_DOC_PATH = os.path.join(_ROOT, "docs", "devices", "NPUPAR.md")


# --------------------------------------------------------------------------
# The fake. One boundary: `intelnpu.OpenVINO`. Everything above it is real.
# --------------------------------------------------------------------------


class _CountingOpenVINO:
    """Stands in for `intelnpu.OpenVINO` and counts what it was asked to do.

    Deliberately **not** a stub that returns a constant: `compile_ir` records
    the batch dimension it read out of the IR the real `linear_ir` emitted, so
    a wiring mistake that compiled the wrong shape shows up as a wrong number
    rather than as a green test. `infer` is not needed by anything here and is
    absent, so a test that accidentally reached forward() would raise rather
    than pass quietly.

    `constructed` is a class-level list because the thing under test is *how
    many were built for one model*, which no instance can answer.
    """

    constructed = []

    #: leaf names whose compile must raise, keyed by the (in, out) shape.
    refuse_shapes = set()

    def __init__(self, path=None, cache_dir=None):
        self.path = path
        self.device_calls = 0
        self.compiled = []
        _CountingOpenVINO.constructed.append(self)

    @classmethod
    def reset(cls):
        cls.constructed = []
        cls.refuse_shapes = set()

    def devices(self):
        self.device_calls += 1
        return ("CPU", "NPU")

    def compile_ir(self, xml, device, weights):
        import re

        dims = [int(d) for d in re.findall(r"<dim>(\d+)</dim>", xml)]
        batch, in_f, out_f = dims[0], dims[1], dims[-1]
        if (in_f, out_f) in _CountingOpenVINO.refuse_shapes:
            raise IntelNPUUnavailable(
                f"torchnative intelnpu: fake refuses ({in_f}, {out_f})"
            )
        handle = {"device": device, "batch": batch, "in": in_f, "out": out_f}
        self.compiled.append(handle)
        return handle

    def execution_devices(self, compiled):
        return (compiled["device"],)


class _with_fake_openvino:
    def __enter__(self):
        _CountingOpenVINO.reset()
        self._real = intelnpu.OpenVINO
        intelnpu.OpenVINO = _CountingOpenVINO
        return _CountingOpenVINO

    def __exit__(self, *exc):
        intelnpu.OpenVINO = self._real
        return False


def _tree(n_leaves, width=4):
    """`n_leaves` `nn.Linear`s under a `ModuleList`, small enough to be quick."""
    return nn.Sequential(*[nn.Linear(width, width) for _ in range(n_leaves)])


# --------------------------------------------------------------------------
# Defect one: the shared core.
# --------------------------------------------------------------------------


def test_a_252_leaf_model_builds_exactly_one_openvino_core():
    """252 is the user's number: Qwen3-4B's `nn.Linear` count under this walk.

    Before this round the count was 252 -- one `ov::Core` per leaf, each
    dlopening and initialising the plugin set. The assertion is `== 1` rather
    than a bound, because "one Core per model" is the whole claim and anything
    else is the defect coming back in a smaller size.
    """
    with _with_fake_openvino() as fake:
        model = _tree(252)
        model, report = _compile_model(model, device="NPU", eager=False)
        assert len(fake.constructed) == 1, (
            f"{len(fake.constructed)} OpenVINO Cores built for a 252-leaf model; "
            f"one is the claim"
        )
        core = fake.constructed[0]
        leaves = [m for m in model.modules() if isinstance(m, _NPULinear)]
        assert len(leaves) == 252, len(leaves)
        assert all(leaf._ov is core for leaf in leaves), (
            "a lowered leaf is holding a Core that is not the shared one"
        )
        assert len(report["swapped"]) == 252, len(report["swapped"])
    print(
        "ok   ovpar: a 252-leaf model builds exactly 1 OpenVINO Core (was 252, "
        "one per leaf), and every lowered leaf holds that same one"
    )


def test_the_device_enumeration_runs_once_for_the_model_not_once_per_leaf():
    """`ov_core_get_available_devices` initialises every plugin it finds.

    Doing it per leaf is 252 plugin-set enumerations, and it is a *separate*
    cost from constructing the Core -- a design that shared the Core but left
    the check in `_compile_for` would keep most of the waste. So it is asserted
    on its own rather than assumed to follow.

    **What nullifies this is not a memo.** There is no per-core "already
    verified" set; the check runs exactly once because it lives in `open_core`
    and `open_core` runs exactly once per model. A first version of this did
    memoise on the core, and removing that memo did **not** turn this test red
    -- the memo was unreachable, because a leaf that was handed a core never
    calls `open_core` at all. It was deleted rather than kept as a guarantee
    nothing could check.
    """
    with _with_fake_openvino() as fake:
        _compile_model(_tree(40), device="NPU", eager=False)
        core = fake.constructed[0]
        assert core.device_calls == 1, (
            f"devices() called {core.device_calls} times for a 40-leaf model"
        )
    print(
        "ok   ovpar: the NPU-presence check enumerates OpenVINO's devices once "
        "per model, not once per lowered leaf"
    )


def test_a_standalone_npulinear_still_builds_its_own_core():
    """`from_torch` must keep working with no one to hand it a Core.

    The shared Core comes from `_compile_model`, which is not the only caller:
    `_NPULinear.from_torch` is public-ish inside this module and is used to
    lower a single layer. If it depended on being given a Core, sharing would
    have been bought by breaking the one-layer path.
    """
    with _with_fake_openvino() as fake:
        leaf = _NPULinear.from_torch(nn.Linear(4, 4), device="NPU")
        assert leaf._ov is None, "no Core should exist before the first compile"
        leaf._compile_for(1)
        assert len(fake.constructed) == 1, len(fake.constructed)
        assert leaf._ov is fake.constructed[0]
    print(
        "ok   ovpar: an _NPULinear built on its own by from_torch still makes "
        "its own Core on first compile -- sharing did not make a Core mandatory"
    )


def test_a_compiled_model_cannot_outlive_the_core_that_made_it():
    """Lifetime, guaranteed by a reference rather than by a convention.

    `OpenVINO.close()` calls `ov_core_free`, and a compiled model whose Core has
    been freed is a use-after-free -- which on this path would present as wrong
    numbers, not as a crash. The guarantee here is the cheapest one that cannot
    be forgotten: `compile_ir` anchors the owning `OpenVINO` object on the
    handle it returns, so no compiled model can be reached while the Python
    object that owns its Core is collectable.

    This is *not* a claim that an explicit `close()` is safe. `probe()` closes
    its Core deliberately at the end of a `with` block, after the compiled
    models inside it are gone; that stays the caller's call. What is ruled out
    is the accidental version -- the Core being collected because nobody kept a
    name for it, which is exactly what a shared Core created inside
    `_compile_model` and never stored anywhere else would have been.
    """
    import ctypes

    real_loader = intelnpu.load_openvino_c

    class _Fn:
        argtypes = None
        restype = None

        def __init__(self, impl):
            self._impl = impl

        def __call__(self, *a):
            return self._impl(*a)

    class _Lib:
        def __init__(self):
            self.freed = 0
            self.ov_core_create = _Fn(self._out(0x1000))
            self.ov_core_free = _Fn(self._free)
            self.ov_core_read_model_from_memory_buffer = _Fn(
                lambda core, blob, n, w, out: self._set(out, 0x2000)
            )
            self.ov_model_free = _Fn(lambda m: None)
            self.ov_core_compile_model = _Fn(
                lambda core, m, dev, cnt, out, *v: self._set(out, 0x4000)
            )
            self.ov_tensor_create_from_host_ptr = _Fn(
                lambda e, s, d, out: self._set(out, 0x3000)
            )
            self.ov_tensor_free = _Fn(lambda t: None)

        def __getitem__(self, name):
            raise KeyError(name)

        def _set(self, out, value):
            out._obj.value = value
            return 0

        def _out(self, value):
            return lambda out: self._set(out, value)

        def _free(self, core):
            self.freed += 1
            return 0

    lib = _Lib()
    intelnpu.load_openvino_c = lambda path=None: lib
    try:
        ov = intelnpu.OpenVINO(cache_dir=None)
        compiled = ov.compile_ir("<net/>", "NPU", None)
    finally:
        intelnpu.load_openvino_c = real_loader

    anchor = getattr(compiled, "_torchnative_core", None)
    assert anchor is ov, (
        "the compiled handle does not hold a reference to the OpenVINO object "
        "that made it, so nothing stops the Core being collected under it"
    )
    assert isinstance(compiled, ctypes.c_void_p), type(compiled)
    assert lib.freed == 0, "nothing should have freed the core yet"
    print(
        "ok   ovpar: a compiled model anchors the OpenVINO object that made it, "
        "so the ov::Core cannot be collected while the compiled model is reachable"
    )


# --------------------------------------------------------------------------
# Eager compilation.
# --------------------------------------------------------------------------


def test_to_npu_compiles_the_decode_shape_for_every_leaf_before_returning():
    """The stall moves to where the user asked for it.

    batch=1 specifically, because that is the shape `generate()` uses for every
    token after the prompt -- it is the one that would otherwise be paid inside
    the *second* token, several seconds into what looks like a working
    generation. The prompt-length shape is not compiled here because it is not
    known until the prompt is.
    """
    with _with_fake_openvino() as fake:
        model, report = _compile_model(_tree(12), device="NPU")
        core = fake.constructed[0]
        leaves = [m for m in model.modules() if isinstance(m, _NPULinear)]
        assert all(1 in leaf._compiled for leaf in leaves), (
            "some leaf still has no batch=1 compile after to(device.npu)"
        )
        assert report["eager_compiled"] == 12, report["eager_compiled"]
        assert report["eager_failed"] == [], report["eager_failed"]
        batches = [h["batch"] for h in core.compiled]
        assert batches == [1] * 12, batches
    print(
        "ok   ovpar: to(device.npu) compiles the batch=1 decode shape for every "
        "leaf up front, so generate() does not stall on it mid-token"
    )


def test_eager_compilation_is_reportable_while_it_happens():
    """Minutes of silence inside `to()` is indistinguishable from a hang.

    The callback gets `(done, total, name)`; `name` is the leaf's dotted path,
    which is the same identifier the report and the oversized-leaf refusal use,
    so a user watching progress and a user reading the report are looking at the
    same names.
    """
    seen = []
    with _with_fake_openvino():
        _compile_model(
            _tree(5), device="NPU", progress=lambda done, total, name: seen.append(
                (done, total, name)
            )
        )
    assert [d for d, _, _ in seen] == [1, 2, 3, 4, 5], seen
    assert {t for _, t, _ in seen} == {5}, seen
    assert [n for _, _, n in seen] == ["0", "1", "2", "3", "4"], seen
    print(
        "ok   ovpar: eager compilation reports (done, total, name) per leaf, so "
        "a to() that takes minutes says what it is doing"
    )


def test_a_leaf_that_will_not_compile_is_named_and_left_on_the_lazy_path():
    """Eager compilation must not turn a working lazy model into a hard failure.

    A leaf OpenVINO refuses at `to()` time is reported by name -- not swallowed,
    and not raised -- and its lazy path is left intact, so the model still runs
    if the refusal was transient or shape-specific. The alternative, raising,
    would mean a single unlucky leaf makes a model that used to work
    unreachable, which is the exact failure `_compile_model` already refuses to
    commit for an oversized leaf.
    """
    with _with_fake_openvino() as fake:
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(6, 8), nn.Linear(4, 4))
        fake.refuse_shapes = {(6, 8)}
        model, report = _compile_model(model, device="NPU")
        assert report["eager_compiled"] == 2, report["eager_compiled"]
        names = [n for n, _ in report["eager_failed"]]
        assert names == ["1"], report["eager_failed"]
        assert "fake refuses" in report["eager_failed"][0][1], report["eager_failed"]
        assert report["fully_offloaded"] is False, (
            "a model with a leaf that would not compile must not report as whole"
        )
        leaf = model[1]
        assert leaf._compiled == {}, leaf._compiled
        # The lazy path is untouched: once the refusal goes away, it compiles.
        fake.refuse_shapes = set()
        assert leaf._compile_for(1)["out"] == 8
    print(
        "ok   ovpar: a leaf that refuses to compile eagerly is named in the "
        "report, drops fully_offloaded, and keeps its working lazy path"
    )


def test_the_first_leaf_is_still_the_hard_no_npu_failure():
    """Eager compilation must not soften the assertion that the device is real.

    `_compile_model` compiles the first swapped leaf and lets that failure out,
    because that is what makes `device="NPU"` on a machine with no NPU a failure
    of *this call* rather than a model that looks offloaded. If eager
    compilation caught everything by name, a machine with no NPU would get 252
    named failures and a model it believes is on the NPU -- the silent fallback
    with a report attached.
    """
    with _with_fake_openvino() as fake:
        fake.refuse_shapes = {(4, 4)}
        try:
            _compile_model(_tree(3), device="NPU")
        except IntelNPUUnavailable as exc:
            assert "fake refuses" in str(exc), exc
        else:
            raise AssertionError(
                "a first leaf that cannot compile was absorbed into the report"
            )
    print(
        "ok   ovpar: a first leaf that cannot compile still raises out of "
        "_compile_model -- 'no NPU here' did not become 252 named warnings"
    )


def test_eager_compilation_can_be_declined():
    """The lazy behaviour is still reachable, and it is still the old behaviour.

    Not for symmetry: eager compilation of 252 leaves is minutes of wall time,
    and a caller who wants to lower a model and inspect it without paying that
    (the tests above, among others) must be able to say so.
    """
    with _with_fake_openvino() as fake:
        model, report = _compile_model(_tree(6), device="NPU", eager=False)
        leaves = [m for m in model.modules() if isinstance(m, _NPULinear)]
        compiled = [leaf for leaf in leaves if leaf._compiled]
        assert len(compiled) == 1, (
            f"{len(compiled)} leaves compiled with eager=False; only the first, "
            f"which is the device assertion, should be"
        )
        assert report["eager_compiled"] == 1, report["eager_compiled"]
        assert len(fake.constructed) == 1, len(fake.constructed)
    print(
        "ok   ovpar: eager=False keeps the lazy path and still shares one Core, "
        "compiling only the first leaf as the device assertion"
    )


# --------------------------------------------------------------------------
# Defect two: the refusal, kept honest.
# --------------------------------------------------------------------------


def test_the_compile_path_still_spawns_no_threads():
    """No thread pool ships this round, and this is the check that says so.

    `docs/devices/NPUPAR.md` records why, question by question, with sources:
    the GIL is held for the IR text and for `torch._C._shim_f16_bytes` (which
    takes `Python<'py>` and never calls `allow_threads`), OpenVINO publishes no
    thread-safety guarantee for concurrent `ov::Core::compile_model`, the NPU's
    compiler-in-driver is closed source and its concurrency behaviour is
    unknown, and the compile cache's write path is guarded per-hash only
    *within one process*.

    This test is the tripwire. If a later round adds threads without revisiting
    that document, this goes red and names it.
    """
    before = threading.active_count()
    with _with_fake_openvino():
        _compile_model(_tree(8), device="NPU")
    assert threading.active_count() == before, (
        f"the compile path left threads behind: {before} -> "
        f"{threading.active_count()}"
    )
    source = open(_SOURCE_PATH, encoding="utf-8").read()
    live = [
        line
        for line in source.splitlines()
        if ("ThreadPoolExecutor" in line or "threading.Thread(" in line)
        and not line.strip().startswith("#")
    ]
    assert live == [], (
        f"intelnpu.py now starts threads: {live}. Read docs/devices/NPUPAR.md "
        f"before shipping this -- four questions there are unresolved, and two "
        f"of them fail as corruption rather than as slowness."
    )
    print(
        "ok   ovpar: the compile path spawns no threads, and intelnpu.py starts "
        "none -- the parallel-compile refusal is still in force"
    )


def test_the_refusal_is_written_down_with_its_four_questions():
    """A refusal with no reasons is indistinguishable from not having looked.

    The document has to name what would settle each question, because "we did
    not parallelise" is only useful to the next person if it comes with what
    they would have to find out.
    """
    assert os.path.exists(_DOC_PATH), f"no {_DOC_PATH}"
    text = open(_DOC_PATH, encoding="utf-8").read()
    for needle in (
        "UNVERIFIED",
        "_shim_f16_bytes",
        "allow_threads",
        "CacheGuard",
        "write_cache_entry",
        "compiler-in-driver",
    ):
        assert needle in text, f"NPUPAR.md does not mention {needle!r}"
    assert "ctypes.CDLL" in text and "PyDLL" in text, (
        "NPUPAR.md does not settle which ctypes loader releases the GIL"
    )
    print(
        "ok   ovpar: NPUPAR.md records the four parallel-compile questions, "
        "their sources, and which two are UNVERIFIED for lack of hardware"
    )


def test_the_withdrawn_public_names_did_not_come_back():
    """This round touches the same module the withdrawal did.

    `compile_model`, `NPULinear`, `quantize_`, `dynamo_backend` and
    `compile_module` were withdrawn as public names. A round that adds keyword
    arguments to `_compile_model` is exactly the kind that reintroduces one by
    reflex.
    """
    withdrawn = intelnpu.IntelNPUWithdrawn
    for name in ("compile_model", "NPULinear", "compile_module", "dynamo_backend"):
        try:
            getattr(intelnpu, name)
        except withdrawn:
            continue
        except AttributeError:
            continue
        raise AssertionError(f"intelnpu.{name} resolves again")
    assert "compile_model" not in intelnpu.__all__, intelnpu.__all__
    assert "NPULinear" not in intelnpu.__all__, intelnpu.__all__
    print(
        "ok   ovpar: the withdrawn public names (compile_model, NPULinear, "
        "compile_module, dynamo_backend) are still withdrawn"
    )


def test_the_oversized_leaf_refusal_still_names_the_leaf_and_the_limit():
    """`docs/graph/NPU2.md`'s invariant, re-asserted from this round's side.

    Eager compilation walks the swapped list; an oversized leaf is never on it,
    because it was refused at `from_torch`. If eager compilation had been built
    by walking the module tree instead, an oversized leaf would have been
    reached and reported as an *eager* failure -- which reads as "OpenVINO would
    not compile it" rather than "this exceeds MAX_DIM", losing the limit.
    """
    with _with_fake_openvino():
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(intelnpu.MAX_DIM + 1, 4))
        model, report = _compile_model(model, device="NPU")
        assert report["eager_failed"] == [], report["eager_failed"]
        names = [n for n, _ in report["skipped"]]
        assert names == ["1"], report["skipped"]
        reason = report["skipped"][0][1]
        assert str(intelnpu.MAX_DIM) in reason, reason
        assert str(intelnpu.MAX_DIM + 1) in reason, reason
    print(
        "ok   ovpar: an oversized leaf is still refused by name with its shape "
        "and MAX_DIM, and does not arrive as an eager-compile failure instead"
    )


def test_unsupported_is_still_raised_for_a_bad_device():
    """The shared Core is created inside `_compile_model`; the device check that
    refuses AUTO/HETERO/MULTI must still come first, before anything is built."""
    with _with_fake_openvino() as fake:
        try:
            _compile_model(_tree(2), device="AUTO")
        except IntelNPUUnsupported as exc:
            assert "AUTO" in str(exc), exc
        else:
            raise AssertionError("AUTO was accepted")
        assert fake.constructed == [], (
            "a Core was built before the device name was checked"
        )
    print(
        "ok   ovpar: an unsupported device is refused before any ov::Core is "
        "constructed, so the refusal costs nothing"
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
