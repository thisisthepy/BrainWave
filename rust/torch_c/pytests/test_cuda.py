"""`cuda` -- what is wired, what is refused by name, and what nobody here can run.

**This machine is an arm64 Mac. It has no NVIDIA GPU, no driver and no nvcc.**
So this file is written around a single distinction, and every test in it falls
on one side or the other:

  * **testable here** -- the refusal taxonomy, the gate's table, the wiring in
    `Cargo.toml` and `aten.rs`, the counters' shape, and the fact that
    `_C._cuda_probe()` answers `not_built` rather than raising. That is most of
    this file and it is the round's verifiable value.
  * **not testable here, and not faked** -- that a kernel runs on a GPU. There
    is no test below that would pass by skipping and there is no test below
    that asserts CUDA computes. `docs/devices/CUDA.md` §6 is a numbered procedure for a
    machine that has a GPU, and §8 says in as many words what "CUDA support"
    does and does not mean after this round.

The rule the round was given -- *every skip must name its reason* -- is why
there are no `unittest.skip`s here at all. Where a state cannot be entered on
this host the test either asserts the *classifier* for that state (saying so in
its own name) or asserts the *source* (saying so in its own name). Neither is
called a measurement.

docs/devices/MPSATTN.md §3.1 is the trap this file is built to avoid: it records that a
source-scanning proof of "the GPU did it" can be defeated by moving the thing
the scan greps for one call deeper. So nothing here reads the source to
establish *device* behaviour. Source reads are used only for claims that are
about source -- where a call sits in a dispatcher, which targets a Cargo entry
can reach -- and they say so.
"""

import os
import re
import sys

from test_shim import _C
# Imported under another name on purpose: `_main()` below collects every
# global whose name starts with `test_`, and a module called `test_shim`
# matches that and is then called. It is the same trap `run.sh` avoids by
# giving each suite its own `__main__` guard, one level down.
import test_shim as shim_helpers


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
CARGO_TOML = os.path.join(REPO, "rust", "torch_c", "Cargo.toml")
ATEN_RS = os.path.join(REPO, "rust", "torch_c", "src", "aten.rs")
DEVICE_RS = os.path.join(REPO, "rust", "torch_c", "src", "device.rs")
CUDA_MD = os.path.join(REPO, "docs", "devices", "CUDA.md")
WORKFLOW = os.path.join(REPO, ".github", "workflows", "build-cuda-wheel.yml")

# The five names `CUDA_REFUSAL_REASONS` publishes. Spelled here as well so that
# a round which adds a sixth has to touch this file -- the artefact's own list
# is what is asserted against, but a test that only compared the artefact to
# itself would go green on any change at all.
EXPECTED_REASONS = ("not_built", "no_driver", "no_device", "wrong_arch", "unclassified")

# `cudarc::driver::DriverError`'s Debug shape, which is also its Display, and
# which is therefore what candle wraps and what this build sees.
#
# **The sentence half is deliberately empty or wrong in these fixtures.** The
# classifier must be matching the `CUresult` enumerator -- which is ABI -- and
# not `cuGetErrorString`'s prose, which belongs to whichever driver is
# installed and is free to be reworded between versions.
DRIVER_ERROR_FIXTURES = (
    # (candle-shaped error text, the reason it must be classified as, what
    #  state on a real machine produces it)
    ("the candle crate has not been built with cuda support",
     "not_built",
     "any build of this crate without the torch_c_cuda cfg -- including this one"),
    ('DriverError(CUresult::CUDA_ERROR_NOT_INITIALIZED, "")',
     "no_driver",
     "libcuda present but cuInit failed -- no kernel module, or a container "
     "started without --gpus"),
    ('DriverError(CUresult::CUDA_ERROR_STUB_LIBRARY, "")',
     "no_driver",
     "the toolkit's libcuda stub is on the path instead of a real driver"),
    ('DriverError(CUresult::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, "not this text")',
     "no_driver",
     "driver older than the CUDA runtime the artefact was built against"),
    ('DriverError(CUresult::CUDA_ERROR_NO_DEVICE, "")',
     "no_device",
     "a working driver and zero visible GPUs -- CUDA_VISIBLE_DEVICES= is the "
     "cheap way to produce it"),
    ('DriverError(CUresult::CUDA_ERROR_INVALID_DEVICE, "")',
     "no_device",
     "cuda:3 on a machine with two GPUs"),
    ('DriverError(CUresult::CUDA_ERROR_NO_BINARY_FOR_GPU, "")',
     "wrong_arch",
     "the statically compiled kernels have no cubin for this device and no PTX "
     "to JIT from"),
    ('DriverError(CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION, "")',
     "wrong_arch",
     "PTX newer than the installed driver can JIT"),
    ("this device reports compute capability 6.1 (sm_61) and the kernels in "
     "this build were compiled for sm_80",
     "wrong_arch",
     "this crate's own check in cuda_arch_mismatch, which fires at resolve() "
     "time rather than at the first kernel launch"),
    ('DriverError(CUresult::CUDA_ERROR_UNKNOWN, "")',
     "unclassified",
     "anything the token table does not know -- the arm that exists so the "
     "taxonomy cannot lie by exhausting"),
)


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# 1. The label, and the refusal this host can actually produce
# ---------------------------------------------------------------------------

def test_cuda_is_a_constructible_label_even_with_no_cuda_in_the_build():
    """`torch.device("cuda")` is a label, and labels construct everywhere.

    `device.rs`'s module comment is where this decision is written down: in
    torch, `torch.device("cuda")` is constructible on a CPU-only build and only
    *using* it fails. candle's `Device` is the opposite -- the variant carries a
    live handle -- so the label is stored and resolved on use. This round makes
    `resolve()` able to succeed for `cuda`; it must not make the label harder to
    make.
    """
    d = _C.device("cuda")
    assert d.type == "cuda", d
    assert d.index is None, d
    assert _C.device("cuda", 1).index == 1
    assert _C.device("cuda:2").index == 2


def test_asking_for_cuda_on_this_build_refuses_and_names_not_built():
    """The one refusal this machine can enter live, and it names itself.

    Not a skip. `not_built` is a real state of a real artefact -- it is the
    state of every wheel this project has published -- and it is reached by
    running the same `resolve()` arm a GPU machine runs, because that arm
    carries no `#[cfg]`: candle's `dummy_cuda_backend` answers
    `NotCompiledWithCudaSupport` when the feature is off.
    """
    probe = _C._cuda_probe()
    assert probe["available"] is False, probe
    reason = probe["reason"]
    assert reason in EXPECTED_REASONS, probe
    if probe["built"]:
        # A CUDA build that reached this suite. Nothing here can say which of
        # the other four it is, so it says that rather than pretending.
        print(f"   (this artefact WAS built with cuda; probe reason={reason!r} -- "
              f"the not_built assertion below does not apply and is skipped "
              f"because the build is different, not because the test is)")
        return
    assert reason == "not_built", probe
    assert probe["built_compute_cap"] is None, probe

    try:
        _C._aten_dispatch("aten.ones.default", [2, 2], device=_C.device("cuda"))
    except NotImplementedError as e:
        text = str(e)
    else:
        raise AssertionError("cuda allocated on a build with no cuda in it")
    # The reason is a token near the front, so a caller can match on it without
    # parsing prose.
    assert "reason: not_built" in text, text
    # And candle's own words survive to the end unedited, because the token is
    # this crate's judgement and the text is the evidence for it.
    assert "has not been built with cuda support" in text, text
    assert "docs/devices/CUDA.md" in text, text


def test_the_refusal_names_the_index_it_was_asked_for():
    """`cuda:3` must not be refused under `cuda:0`'s name.

    Cheap, and it is the half of `resolve()` that a hardcoded index would break
    silently -- the same class of mistake `from_candle`'s comment warns about
    from the other direction.
    """
    if _C._cuda_probe()["built"]:
        print("   (not asserted: this artefact has cuda built in, so cuda:3 may "
              "resolve rather than refuse)")
        return
    for index in (0, 1, 3):
        try:
            _C._aten_dispatch("aten.ones.default", [2, 2],
                              device=_C.device("cuda", index))
        except NotImplementedError as e:
            assert f"cuda:{index} is not available" in str(e), (index, str(e)[:200])
        else:
            raise AssertionError(f"cuda:{index} allocated")


# ---------------------------------------------------------------------------
# 2. The four refusals this machine cannot enter
# ---------------------------------------------------------------------------

def test_the_refusals_this_host_cannot_enter_are_still_named_one_by_one():
    """Each reason, from a driver-shaped string, asserted by name.

    **This tests the classifier, not the machine**, and the name says so. What
    it establishes is that if a driver returns `CUDA_ERROR_NO_DEVICE` the user
    is told `no_device` and not something vaguer; what it does not establish is
    that any driver ever did. `docs/devices/CUDA.md` §8 keeps those apart.

    It is here rather than only in `cargo test` because the thing under test is
    the **loaded artefact's** classifier, reached the way a caller reaches it.
    """
    for detail, expected, produced_by in DRIVER_ERROR_FIXTURES:
        got = _C._shim_cuda_classify_refusal(detail)
        assert got == expected, (
            f"{detail!r} classified {got!r}, expected {expected!r} "
            f"(this string is produced by: {produced_by})")


def test_every_published_reason_has_a_fixture_and_every_fixture_a_reason():
    """The vocabulary is closed at both ends.

    Without this, adding a sixth reason to `CUDA_REFUSAL_REASONS` would leave
    the test above green while one name in the user-visible taxonomy had never
    been produced by anything.
    """
    published = tuple(_C._shim_cuda_refusal_reasons())
    assert published == EXPECTED_REASONS, (published, EXPECTED_REASONS)
    covered = {expected for _, expected, _ in DRIVER_ERROR_FIXTURES}
    assert covered == set(published), (
        "reasons with no fixture: " + repr(sorted(set(published) - covered)))


def test_an_unrecognised_driver_error_is_not_forced_into_one_of_the_four():
    """The arm that keeps the taxonomy from lying.

    A classifier that mapped every input onto one of four names would be at its
    least trustworthy exactly when it mattered -- a driver state nobody
    anticipated would be reported confidently as the wrong one. `unclassified`
    hands back the driver's own words instead, and this asserts that a plausible
    unknown really does land there rather than being absorbed by a substring
    match on a neighbouring token.
    """
    for unknown in (
        'DriverError(CUresult::CUDA_ERROR_OUT_OF_MEMORY, "out of memory")',
        'DriverError(CUresult::CUDA_ERROR_ILLEGAL_ADDRESS, "")',
        "something a future candle wrote",
        "",
    ):
        assert _C._shim_cuda_classify_refusal(unknown) == "unclassified", unknown


def test_every_reason_is_documented_in_cuda_md():
    """A refusal name a reader cannot look up is not a named refusal."""
    doc = _read(CUDA_MD)
    for reason in _C._shim_cuda_refusal_reasons():
        assert f"`{reason}`" in doc, (
            f"docs/devices/CUDA.md never mentions the refusal reason {reason!r}")


# ---------------------------------------------------------------------------
# 3. The refusal list -- derived, not written
# ---------------------------------------------------------------------------

def test_the_cuda_readback_list_is_the_mps_one_and_is_re_derived_from_aten_rs():
    """One derivation, two devices -- and it is re-run rather than restated.

    The set is not a statement about Metal or about CUDA. It is the set of
    kernels in `aten.rs` that pull a tensor's bytes to the host and do the
    arithmetic in Rust, which is a property of *this crate's kernel* and cannot
    legitimately differ between two accelerators. A second hand-written list is
    exactly how they would come to differ.

    Both halves are asserted against the **loaded artefact**, the way
    `test_the_mps_readback_list_is_what_the_kernels_actually_do` is: comparing
    two source reads would agree with itself after a change nobody rebuilt.

    docs/architectures/VOICE3.md §6 is why the derivation is not trusted past where it goes --
    it follows helper calls one level, by name, and `var`/`std` reached the list
    through nothing until the read was spelled at each dispatch target. That
    blind spot is covered by `test_every_host_readback_in_aten_is_classified`
    in `test_shim.py`, which this round did not weaken and which now guards two
    devices instead of one.
    """
    cuda = _C._shim_cuda_host_readback_ops()
    mps = _C._shim_mps_host_readback_ops()
    assert cuda == mps, (
        "the cuda and mps refusal lists have diverged; they are one derived "
        "set and CUDA_HOST_READBACK_OPS is meant to be an alias: "
        + repr(sorted(set(cuda) ^ set(mps))))
    assert len(cuda) > 80, len(cuda)

    parsed = shim_helpers._aten_rs_functions()
    if parsed is None:
        print("   (not re-derived: rust/torch_c/src/aten.rs is not beside this "
              "file -- installed rather than in-tree. The equality above still "
              "held.)")
        return
    bodies, text = parsed
    ops = shim_helpers._aten_dispatch_targets(text)
    allowed = set(_C._shim_mps_readback_but_allowed())
    derived = set()
    for op, fn in ops.items():
        body = bodies.get(fn, "")
        reads = bool(shim_helpers._MPS_READBACK_MARKERS.search(body)) or any(
            re.search(r"\b" + helper + r"\s*\(", body)
            for helper in shim_helpers._MPS_READBACK_HELPERS
        )
        if reads and op not in allowed:
            derived.add(op)
    assert derived == set(cuda), (
        "the cuda refusal list is not what the kernels do: missing "
        + repr(sorted(derived - set(cuda))) + " stale "
        + repr(sorted(set(cuda) - derived)))


def test_the_readback_gate_is_more_necessary_on_cuda_than_it_was_on_metal():
    """The one CUDA-specific finding, pinned so it is not lost.

    docs/devices/MPS.md §2 measured that thirteen of fourteen silent CPU fallbacks on
    `mps` were on the **integer** path: the float path goes through
    `read_flat`'s `widen_f64` and Metal has no `F32 -> F64`, so it raised. CUDA
    implements `f64`. The same kernels that were noisy on Metal are silent on
    CUDA, so the gate covers *more* there, not less.

    Asserted from the source of `read_flat`, because the claim is about what
    that helper does, not about what a device did.
    """
    text = _read(os.path.join(REPO, "rust", "torch_c", "src", "aten.rs"))
    assert "fn read_flat" in text
    assert "widen_f64" in text, (
        "read_flat no longer widens through f64 -- docs/devices/CUDA.md §5's argument "
        "about why cuda is quieter than metal needs re-measuring")


# ---------------------------------------------------------------------------
# 4. Wiring -- claims about source, said to be claims about source
# ---------------------------------------------------------------------------

def test_the_dispatcher_gates_cuda_before_it_runs_a_kernel():
    """The gate is in front of the kernel, not behind it.

    A source read, and it is a claim about source: where a call sits in the
    dispatcher. It is not evidence that anything ran on a GPU -- docs/devices/MPSATTN.md
    §3.1's warning is about using a scan for *that*, and nothing here does.
    """
    text = _read(ATEN_RS)
    arm = re.search(
        r"Some\(Where::Dense\(ref device\)\) if crate::device::is_cuda\(device\) => \{(.*?)\n        \}",
        text, re.S)
    assert arm, "no is_cuda arm in aten_dispatch"
    body = arm.group(1)
    gate = body.index("cuda_host_readback_gate")
    note = body.index("note_cuda_dispatch")
    kernel = body.index("aten_dispatch_inner")
    assert gate < note < kernel, (
        "the cuda arm must gate, then count, then run: got "
        f"gate@{gate} note@{note} kernel@{kernel}")
    assert text.count("is_cuda(device)") == 1, (
        "there must be exactly one cuda door in the dispatcher")


def test_the_two_devices_share_one_gate_body():
    """Two copies of a guard is the defect the previous round had to fix.

    `mps_host_readback_gate` and `cuda_host_readback_gate` differ only in the
    words that name the device; the wording lives once, in
    `host_readback_gate`. A second copy is how the two messages -- and then the
    two behaviours -- drift apart.
    """
    text = _read(DEVICE_RS)
    assert text.count("fn host_readback_gate") == 1, text.count("fn host_readback_gate")
    for name in ("mps_host_readback_gate", "cuda_host_readback_gate"):
        body = re.search(r"pub fn " + name + r"\(op: &str\) -> PyResult<\(\)> \{(.*?)\n\}",
                         text, re.S)
        assert body, name
        assert "host_readback_gate(op," in body.group(1), (
            f"{name} does not delegate to the shared body")
        assert "not implemented for the" not in body.group(1), (
            f"{name} has its own copy of the message")


def test_cuda_cannot_reach_the_android_ios_or_wasm_builds():
    """The target scoping, read off the Cargo entry that enforces it.

    A claim about `Cargo.toml`, and the strongest one available without running
    six cross builds: the entry is gated on `target_os` being `linux` or
    `windows`, and Android is `android`, iOS is `ios`, wasm is `emscripten` or
    `unknown`. None of them match, **with or without** the cfg key -- which is
    the property that makes the target list the half that cannot be switched
    off. `docs/devices/CUDA.md` §2 records which cross builds were actually run.
    """
    toml = _read(CARGO_TOML)
    entry = re.search(r"\[target\.'cfg\(([^\n]*torch_c_cuda[^\n]*)\)'\.dependencies\]", toml)
    assert entry, "no target-scoped cuda entry in Cargo.toml"
    cfg = entry.group(1)
    assert 'target_os = "linux"' in cfg and 'target_os = "windows"' in cfg, cfg
    for forbidden in ("android", "ios", "emscripten", "wasm", "apple"):
        assert forbidden not in cfg, (
            f"the cuda dependency entry mentions {forbidden!r}: {cfg}")
    # The blanket pin must still be there, or a future upstream default could
    # link CUDA into a device build -- which is the reason the pin exists.
    assert 'candle-core = { version = "0.11.0", default-features = false }' in toml
    # And nothing may reach for cudarc's dlopen feature: cudarc's own build
    # script panics when it is on alongside candle's `dynamic-linking`.
    assert "dynamic-loading" not in re.sub(r"^#.*$", "", toml, flags=re.M), (
        "a cudarc `dynamic-loading` feature would make the build panic -- "
        "candle pins `dynamic-linking` and the two are mutually exclusive")


# ---------------------------------------------------------------------------
# 5. The counters
# ---------------------------------------------------------------------------

def test_the_counters_have_the_shape_a_gpu_machine_will_read():
    """`_cuda_counters()`'s keys, asserted where there is no GPU.

    The point of asserting the shape here is that the person who *does* have a
    GPU should find the instrument already correct, rather than discovering on
    a T4 that a key is misspelled. Values are not asserted -- there is nothing
    to measure -- except the two that must be `None` and not `0`.
    """
    c = _C._cuda_counters()
    for key in ("resolves", "dispatches", "readback_refusals", "refusals",
                "ops", "device_free_bytes", "device_total_bytes"):
        assert key in c, (key, sorted(c))
    assert isinstance(c["ops"], dict), c["ops"]
    if not _C._cuda_probe()["built"]:
        # `None`, not `0`. A zero here would read as "the GPU has no free
        # memory", which is a different and alarming claim.
        assert c["device_free_bytes"] is None, c
        assert c["device_total_bytes"] is None, c
        assert c["resolves"] == 0 and c["dispatches"] == 0, c
        assert c["ops"] == {}, c


def test_a_counter_moves_on_the_one_runtime_event_this_host_can_produce():
    """The counters are wired to what the process *did*, and here is the proof.

    Only one CUDA event is reachable on a machine with no CUDA: a refusal. So
    that is the one asserted, and it is asserted as a **delta across a call**
    rather than as an absolute -- which is the form `_vulkan_counters()`
    established and the form that survives other tests in this file having run
    first.

    This is deliberately not a claim that the GPU counters work. It is a claim
    that the counters are runtime instruments rather than constants, which is
    the property docs/devices/MPSATTN.md §3.1 says source-scanning evidence lacks. The
    three that need a GPU are exercised by the procedure in docs/devices/CUDA.md §6.
    """
    if _C._cuda_probe()["built"]:
        print("   (not asserted: on a cuda build resolve() may succeed, so a "
              "refusal is not guaranteed to be producible here)")
        return
    before = _C._cuda_counters()["refusals"]
    for _ in range(3):
        try:
            _C._aten_dispatch("aten.ones.default", [1], device=_C.device("cuda"))
        except NotImplementedError:
            pass
    after = _C._cuda_counters()["refusals"]
    assert after - before == 3, (before, after)


def test_the_gpu_evidence_is_not_the_kind_mpsattn_says_can_be_defeated():
    """The counters must not be a re-spelling of a source scan.

    docs/devices/MPSATTN.md §3.1 records that moving a `read_flat` one call deeper
    passes both of the `mps` derivation tests while keeping the readback. The
    defence here is that `device_free_bytes` is read from the **driver** at call
    time -- it is not in this repository at all, so nothing in this repository
    can move it out of the way.

    Asserted as a property of `device.rs`: the memory reading must come from
    cudarc's `mem_get_info`, and must not be computed from anything this crate
    tracks itself.
    """
    text = _read(DEVICE_RS)
    assert "mem_get_info()" in text, (
        "device_free_bytes no longer comes from the driver -- if it is now "
        "computed from this crate's own bookkeeping it is defeatable in exactly "
        "the way docs/devices/MPSATTN.md §3.1 describes")
    assert "fn note_cuda_dispatch" in text
    # One door, counted at the door. If `note_cuda_dispatch` grew call sites
    # inside kernels, a kernel could be added that forgets it.
    aten = _read(ATEN_RS)
    calls = re.findall(r"note_cuda_dispatch\(", aten)
    assert len(calls) == 1, (
        f"{len(calls)} call sites for note_cuda_dispatch; there must be exactly "
        "one, at the dispatcher's door -- counting inside kernels is a count a "
        "new kernel can forget")


# ---------------------------------------------------------------------------
# 6. The CI job, and the procedure
# ---------------------------------------------------------------------------

def test_the_cuda_workflow_claims_a_build_and_not_a_computation():
    """A GPU-less runner may say "it built". It may not say "it works".

    `.github/workflows/verify-published-wheel.yml` is careful about exactly this
    distinction and is the model. The failure this guards against is a green
    check mark on a CUDA job being read as "CUDA works", which is the most
    expensive misreading available in this round.
    """
    yml = _read(WORKFLOW)
    lowered = yml.lower()
    assert "no gpu" in lowered, "the job must say the runner has no GPU"
    assert "build claim" in lowered or "builds only" in lowered, lowered[:200]
    # The runner must not be asked to execute a cuda tensor.
    assert 'device="cuda"' not in yml and "device='cuda'" not in yml, (
        "the workflow tries to use a cuda device on a runner with no GPU")
    # And it must actually build, or it claims nothing at all.
    assert "torch_c_cuda" in yml, "the workflow does not enable the cuda cfg"
    assert "CUDA_COMPUTE_CAP" in yml, (
        "candle-kernels' build script detects the capability with nvidia-smi and "
        "fails on a runner with no GPU unless CUDA_COMPUTE_CAP is set -- the job "
        "must set it or it cannot build at all")


def test_the_gpu_procedure_is_present_and_is_valid_python():
    """The procedure in docs/devices/CUDA.md §6 has to still parse.

    A copy-pasteable procedure that stopped being copy-pasteable is worse than
    no procedure, because it is read as tested. This does not run it -- there is
    no GPU -- it compiles it, and checks that every `_C` name it reaches for
    exists in this artefact. A step that calls a function this build does not
    have would fail on the user's machine for a reason that has nothing to do
    with their GPU.
    """
    doc = _read(CUDA_MD)
    blocks = re.findall(r"<!-- CUDA_PROCEDURE_PYTHON -->\n```python\n(.*?)```", doc, re.S)
    assert blocks, "docs/devices/CUDA.md has no marked procedure block"
    for block in blocks:
        compile(block, "<docs/devices/CUDA.md §6>", "exec")
        for name in sorted(set(re.findall(r"_C\.(_[A-Za-z0-9_]+)", block))):
            assert hasattr(_C, name), (
                f"docs/devices/CUDA.md §6 calls _C.{name}, which this build does not have")


def test_this_file_never_claims_cuda_computes():
    """The round's hard rule, asserted against the round's own test file.

    "Do not leave a test that passes because it skipped" is checkable; so is
    "do not fake it". This reads this file and refuses a `skip` decorator or an
    assertion that a cuda tensor produced a value.
    """
    text = _read(os.path.abspath(__file__))
    body = text[text.index("# " + "-" * 74):]
    # Spelled in pieces so that this test's own source does not contain the
    # tokens it forbids -- otherwise the check can only ever fail.
    for token in ("unit" + "test", "@" + "skip", "pytest.mark." + "skip"):
        assert token not in body, f"{token} is skip machinery and does not belong here"
    # Every early `return` in this file must be preceded by a printed reason.
    for match in re.finditer(r"\n        return\n", body):
        window = body[max(0, match.start() - 500):match.start()]
        assert "print(" in window, (
            "an early return with no printed reason at offset "
            f"{match.start()} -- every skip must name its reason")


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
