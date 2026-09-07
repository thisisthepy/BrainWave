"""Tests for docs/QNNCI.md -- the CI job that produces the QNN artefact.

**Why a new file rather than more of `test_qnn.py`.** `test_qnn.py` is about
`torchnative.export.qnn`'s behaviour, and its three fixtures all skip by name
when their environment is absent -- in particular `_qnn_et_fixture` skips
unless `$TORCHNATIVE_QNN_PYTHON` points at an interpreter with executorch. That
is right for what it tests and wrong for what is tested here. The decisions in
`tools/ci/qnn_lower.py` are exactly the ones a green CI run never exercises,
and putting them behind an executorch-shaped skip would mean that on this
project's own machine -- the only machine that runs the gate -- they never run
at all. A test that always skips is the `CLAUDE.md` §5.5 shape wearing a new
hat, and §5.5 is the most-cited rule in this repository.

So **nothing in this file imports executorch, torch, or torchnative.** It
imports `tools/ci/qnn_lower.py` (which is pure Python until `main()`), feeds it
dicts, and reads `.github/workflows/qnn-lower.yml` as text and as YAML.

Two halves:

* **the verdict** -- does `verify_facts` refuse the artefacts it must refuse?
  Every branch, including the ones a successful run never takes. The dangerous
  artefact here is not a corrupt file; it is a perfectly good `.pte` that
  silently fell back to portable CPU kernels, loads, runs, and returns the
  right answer (docs/QNN.md §6.2).
* **the workflow's honesty** -- does the job actually upload something, does
  the verification run before the upload, and is any assertion disabled? This
  repository has shipped a workflow whose default pointed at an unpublished
  version for three releases (`test_release.py`'s docstring), so "the YAML says
  what it means" is not assumed here either.
"""

import importlib.util
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github/workflows/qnn-lower.yml"
SCRIPT = REPO / "tools/ci/qnn_lower.py"
QNN_MODULE = REPO / "torchnative/src/main/torchnative/export/qnn.py"
DOC = REPO / "docs/QNNCI.md"


def _skip(reason):
    print(f"     skip -- {reason}")


def _load_script():
    spec = importlib.util.spec_from_file_location("_qnn_lower_ci", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


QL = _load_script()


# ---------------------------------------------------------------------------
# Fact dicts. `_facts()` is a *passing* artefact; every test below breaks
# exactly one thing about it, so each failure message names one cause.
# ---------------------------------------------------------------------------

def _facts(**over):
    facts = {
        "path": "layer0_mlp.pte",
        "soc_model": "SM8550",
        "backend_ids": ("QnnBackend",),
        "is_qnn": True,
        "match_ok": True,
        "match_why": "HTP v73, backend_type kHtpBackend, precision kHtpFp16",
        "delegation": {
            "subgraphs": 1,
            "delegated_nodes": 19,
            "non_delegated_nodes": 1,
            "total_nodes": 20,
            "delegated_fraction": 0.95,
        },
    }
    delegation = over.pop("delegation_over", None)
    facts.update(over)
    if delegation:
        facts["delegation"] = dict(facts["delegation"], **delegation)
    return facts


# ---------------------------------------------------------------------------
# Half 1 -- the verdict
# ---------------------------------------------------------------------------

def test_a_clean_qnn_artefact_clears_both_gates():
    verdict = QL.verify_facts(_facts(), "SM8550")
    assert verdict.ok, verdict.failures
    assert verdict.integrity_ok
    assert verdict.integrity == [] and verdict.coverage == []


def test_a_program_with_no_qnn_delegate_is_never_uploaded():
    """The silent fallback, in its purest form.

    docs/QNN.md §6.2 row 1: whatever `QnnPartitioner` declined stays in the
    program as portable CPU kernels. A `.pte` with zero QNN segments still
    loads, still runs and still returns the right answer -- so nothing about
    running it says it is not what this job would be labelling it. It has to
    be caught here or not at all.
    """
    verdict = QL.verify_facts(
        _facts(is_qnn=False, backend_ids=("XnnpackBackend",)), "SM8550"
    )
    assert not verdict.integrity_ok, "a non-QNN program passed the upload gate"
    assert not verdict.ok
    blob = " ".join(verdict.integrity)
    assert "is_qnn" in blob and "XnnpackBackend" in blob, blob


def test_the_backend_id_is_checked_independently_of_is_qnn():
    """Two reads of the same file, deliberately not collapsed into one.

    `is_qnn` is derived by `torchnative.export.qnn`; `backend_ids` is what the
    artefact carries. If they ever disagree, that disagreement is the finding,
    and a check that trusted the derived one could not see it. So an artefact
    claiming `is_qnn` while carrying no `QnnBackend` delegate is refused.
    """
    verdict = QL.verify_facts(
        _facts(is_qnn=True, backend_ids=("CoreMLBackend",)), "SM8550"
    )
    assert not verdict.integrity_ok, verdict
    assert any("QnnBackend" in f for f in verdict.integrity), verdict.integrity


def test_an_artefact_for_the_wrong_silicon_is_refused_before_it_is_pushed():
    """docs/QNN.md §5 step 7's `assert ok, why`.

    v75 built, v73 device. QNN catches this itself -- at backend init, on the
    phone, by aborting the method load (§6.3) -- so the cost of missing it here
    is a push, a run and a stack trace rather than a wrong answer. It is still
    worth an upload gate: an artefact uploaded under a `SM8550` name that only
    a v75 part can load is mislabelled.
    """
    verdict = QL.verify_facts(
        _facts(
            match_ok=False,
            match_why="artefact was built for HTP v75 (soc_model 57) but "
                      "SM8550 is HTP v73.",
        ),
        "SM8550",
    )
    assert not verdict.integrity_ok
    assert any("match_device" in f and "v75" in f for f in verdict.integrity), \
        verdict.integrity


def test_poor_coverage_fails_the_job_and_keeps_the_artefact():
    """The docs/QNN.md §11 case, and the reason integrity and coverage split.

    §11: `delegated_fraction` for a real `LlamaMLP` under `QnnPartitioner` is
    UNKNOWN and may be poor, citing docs/REFOLD.md §1.1 -- three occasions of
    assuming coverage and being wrong. If it IS poor, the artefact is still a
    genuine QNN program and it is still the answer to the question §11 asked.
    The job goes red; the evidence survives.
    """
    verdict = QL.verify_facts(
        _facts(delegation_over={
            "delegated_nodes": 7, "non_delegated_nodes": 13,
            "total_nodes": 20, "delegated_fraction": 0.35,
        }),
        "SM8550",
    )
    assert not verdict.ok, "poor coverage passed the job"
    assert verdict.integrity_ok, (
        "poor coverage was classified as an integrity failure, which would "
        "suppress the upload and destroy the measurement docs/QNN.md §11 asked "
        "for"
    )
    assert len(verdict.coverage) == 1, verdict.coverage
    assert "0.3500" in verdict.coverage[0], verdict.coverage[0]
    assert "do NOT lower MIN_DELEGATED_FRACTION" in verdict.coverage[0]


def test_an_empty_graph_is_an_integrity_failure_not_a_coverage_one():
    """`delegation_report` returns `delegated_fraction = 0.0` when total is 0.

    Without a separate branch, "nothing was exported" would be reported as
    "the partitioner claimed 0%" -- a true number attached to a false cause,
    sending the reader to QNN's op coverage when the problem is that there was
    no graph. A check whose failure message points at the wrong thing is half
    a check.
    """
    verdict = QL.verify_facts(
        _facts(delegation_over={
            "delegated_nodes": 0, "non_delegated_nodes": 0,
            "total_nodes": 0, "delegated_fraction": 0.0,
        }),
        "SM8550",
    )
    assert not verdict.integrity_ok, verdict
    assert any("empty graph" in f for f in verdict.integrity), verdict.integrity
    assert verdict.coverage == [], (
        "an empty graph was ALSO reported as poor coverage, which is the "
        "double-report this branch exists to prevent"
    )


def test_a_missing_delegated_fraction_is_refused_rather_than_defaulted():
    verdict = QL.verify_facts(
        _facts(delegation_over={"delegated_fraction": None}), "SM8550"
    )
    assert not verdict.integrity_ok
    assert any("delegated_fraction" in f for f in verdict.integrity)


def test_the_threshold_is_strictly_greater_than():
    """docs/QNN.md §5 step 6 is `> 0.9`, not `>= 0.9`. Kept strict."""
    at = QL.verify_facts(
        _facts(delegation_over={"delegated_fraction": 0.9}), "SM8550"
    )
    assert at.coverage, "delegated_fraction == 0.9 cleared a `> 0.9` threshold"
    just_over = QL.verify_facts(
        _facts(delegation_over={"delegated_fraction": 0.9001}), "SM8550"
    )
    assert just_over.ok, just_over.failures


def test_the_threshold_cannot_be_lowered_from_the_command_line():
    """docs/QNN.md §11 warns against tuning this number until it passes.

    A `--min-delegated-fraction` flag would make that a one-word edit to a
    workflow input, invisible in the diff of any file that records a
    measurement. There is no such flag, and this is what keeps there from
    being one.
    """
    parser = QL.build_parser()
    options = {a for action in parser._actions for a in action.option_strings}
    offenders = {o for o in options if "fraction" in o or "threshold" in o
                 or "min" in o.lower()}
    assert not offenders, (
        f"the lowering script grew a way to relax its own threshold: "
        f"{sorted(offenders)}. docs/QNN.md §11: if the number is poor, that is "
        f"the finding -- record it, do not dial the assertion down to it."
    )
    assert QL.MIN_DELEGATED_FRACTION == 0.9, (
        f"MIN_DELEGATED_FRACTION is {QL.MIN_DELEGATED_FRACTION}, but "
        f"docs/QNN.md §5 step 6 asserts > 0.9. If this was lowered to make a "
        f"run green, that is the thing §11 says not to do."
    )


def test_the_backend_id_here_matches_the_one_the_qnn_module_uses():
    """`tools/ci/qnn_lower.py` restates `QnnBackend` instead of importing it.

    That is deliberate -- the CI logic has to be testable on a host where
    `torchnative.export.qnn` cannot be imported. The cost of restating is that
    the two can drift, and this is what pays it. Read as text, so it holds on a
    machine that cannot import the module either.
    """
    source = QNN_MODULE.read_text()
    m = re.search(r'^QNN_BACKEND_ID\s*=\s*"([^"]+)"', source, re.M)
    assert m, "qnn.py no longer defines QNN_BACKEND_ID as a string literal"
    assert QL.QNN_BACKEND_ID == m.group(1), (
        f"tools/ci/qnn_lower.py says {QL.QNN_BACKEND_ID!r} and "
        f"torchnative.export.qnn says {m.group(1)!r}. The CI gate would be "
        f"checking for a delegate name nothing produces."
    )


def test_the_default_soc_is_the_device_this_project_actually_has():
    """CLAUDE.md §7 and docs/QNN.md §5.1: SM8550, read off the device.

    The doc's example command line says SM8650, which is HTP v75 against this
    device's v73. Defaulting to the doc's example rather than the measurement
    would produce an artefact that this project's only Snapdragon refuses at
    backend init -- an entire CI round spent to build something unloadable.
    """
    assert QL.DEFAULT_SOC_MODEL == "SM8550", QL.DEFAULT_SOC_MODEL


def test_the_fraction_is_reported_on_every_path_including_the_red_ones():
    """docs/QNN.md §11 asked for the number, not for a pass/fail.

    The paths where somebody wants it most are the failing ones, so it is
    checked on all three: clean, poor coverage, and integrity-refused.
    """
    for label, facts in (
        ("clean", _facts()),
        ("poor", _facts(delegation_over={
            "delegated_nodes": 2, "non_delegated_nodes": 898,
            "total_nodes": 900, "delegated_fraction": 0.0022})),
        ("refused", _facts(is_qnn=False, backend_ids=())),
    ):
        summary = QL.render_summary(QL.verify_facts(facts, "SM8550"), "SM8550")
        assert "delegated_fraction" in summary, label
        assert "delegation_report" in summary, label
        fraction = facts["delegation"]["delegated_fraction"]
        assert f"{fraction:.4f}" in summary or str(fraction) in summary, \
            (label, summary)


def test_the_summary_says_what_a_green_run_does_not_prove():
    summary = QL.render_summary(QL.verify_facts(_facts(), "SM8550"), "SM8550")
    assert "NOTHING about HTP execution" in summary, summary


class _FakeArtefact:
    def __init__(self, backend_ids, is_qnn):
        self.backend_ids = backend_ids
        self.is_qnn = is_qnn

    def htp_plan(self):
        return {"backend_type": "kHtpBackend", "soc_model": 30, "htp_arch": 73}


class _FakeQnn:
    """A stand-in for `torchnative.export.qnn` with no executorch behind it."""

    def __init__(self, artefact, match):
        self._artefact = artefact
        self._match = match
        self.match_calls = []

    def read_artefact(self, path):
        return self._artefact

    def match_device(self, artefact, soc):
        self.match_calls.append(soc)
        return self._match


def test_collect_facts_asks_all_four_questions_step_7_asks():
    """And asks `match_device` about the SoC it was told, not a hardcoded one."""
    qnn = _FakeQnn(_FakeArtefact(("QnnBackend",), True), (True, "HTP v73"))
    facts = QL.collect_facts(qnn, "a.pte", "SM8550", {
        "total_nodes": 20, "delegated_nodes": 19, "non_delegated_nodes": 1,
        "subgraphs": 1, "delegated_fraction": 0.95,
    })
    assert facts["backend_ids"] == ("QnnBackend",)
    assert facts["is_qnn"] is True
    assert facts["match_ok"] is True
    assert facts["htp_plan"]["htp_arch"] == 73
    assert qnn.match_calls == ["SM8550"], (
        f"match_device was asked about {qnn.match_calls}, not the requested "
        f"SoC. A hardcoded SoC here would make the --soc-model input a lie."
    )
    assert QL.verify_facts(facts, "SM8550").ok


def test_collect_facts_carries_a_refusal_through_rather_than_swallowing_it():
    qnn = _FakeQnn(
        _FakeArtefact(("XnnpackBackend",), False),
        (False, "built for HTP v75"),
    )
    facts = QL.collect_facts(qnn, "a.pte", "SM8550", {
        "total_nodes": 20, "delegated_nodes": 0, "non_delegated_nodes": 20,
        "subgraphs": 0, "delegated_fraction": 0.0,
    })
    verdict = QL.verify_facts(facts, "SM8550")
    assert not verdict.integrity_ok
    assert len(verdict.integrity) == 3, (
        f"expected is_qnn, backend_ids and match_device all to fire; got "
        f"{verdict.integrity}"
    )


# ---------------------------------------------------------------------------
# Half 2 -- is the workflow honest?
# ---------------------------------------------------------------------------

def _yaml():
    try:
        import yaml
    except ImportError:
        return None
    return yaml.safe_load(WORKFLOW.read_text())


def _lower_job(doc):
    # PyYAML resolves the bare key `on` to the boolean True (the "Norway
    # problem"), so `doc["on"]` is not how to reach the triggers. Not relevant
    # for `jobs`, but worth saying once here rather than debugging it twice.
    return doc["jobs"]["lower"]


def test_the_workflow_exists_and_parses():
    assert WORKFLOW.is_file(), f"no workflow at {WORKFLOW}"
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    assert "lower" in doc["jobs"], sorted(doc["jobs"])


def test_the_job_uploads_an_artefact():
    """A job that verifies and uploads nothing has produced nothing.

    docs/QNN.md §5 step 8 wants three files copied to the host holding the
    device. The `.pte` is the one this repository cannot make anywhere else.
    """
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    uploads = [s for s in _lower_job(doc)["steps"]
               if "upload-artifact" in str(s.get("uses", ""))]
    assert uploads, "the lowering job uploads nothing"
    paths = " ".join(str(s.get("with", {}).get("path", "")) for s in uploads)
    assert ".pte" in paths, f"no .pte among the uploaded paths: {paths!r}"


def test_the_verification_runs_before_every_upload():
    """Ordering, read off the step list rather than assumed from reading it.

    An upload placed above the verification would upload whatever the lowering
    wrote, unverified -- and would still look correct in review, because both
    steps are present and both are spelled right.
    """
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    steps = _lower_job(doc)["steps"]
    verify_at = [i for i, s in enumerate(steps) if s.get("id") == "lower"]
    assert len(verify_at) == 1, (
        f"expected exactly one step with id `lower`; found {verify_at}"
    )
    upload_at = [i for i, s in enumerate(steps)
                 if "upload-artifact" in str(s.get("uses", ""))]
    assert upload_at, "no upload steps"
    assert min(upload_at) > verify_at[0], (
        f"an upload step at index {min(upload_at)} runs before the "
        f"lower-and-verify step at {verify_at[0]}"
    )


def test_the_pte_upload_is_gated_on_the_verification_result():
    """`if:` must consult the verifier, not merely run after it.

    `always()` on an upload without the `integrity_ok` conjunct would publish
    a non-QNN `.pte` from a red run under a name saying it is a QNN artefact.
    """
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    pte = [s for s in _lower_job(doc)["steps"]
           if ".pte" in str(s.get("with", {}).get("path", ""))]
    assert len(pte) == 1, f"expected one .pte upload step, found {len(pte)}"
    condition = str(pte[0].get("if", ""))
    assert "steps.lower.outputs.integrity_ok" in condition, (
        f"the .pte upload is not gated on the verifier's own output: "
        f"if: {condition!r}"
    )
    assert "'true'" in condition, condition


def test_nothing_in_the_lowering_job_disables_its_own_assertions():
    """`|| true`, `continue-on-error`, and `set +e`, in the job that decides.

    CLAUDE.md §5.5. The `android-runtime` job IS `continue-on-error` and says
    at length why; this asserts that the exemption did not spread to the job
    that carries the claim.
    """
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    job = _lower_job(doc)
    assert not job.get("continue-on-error"), (
        "the lowering job is continue-on-error, so every check in it is "
        "advisory"
    )
    for step in job["steps"]:
        name = step.get("name") or step.get("uses")
        assert not step.get("continue-on-error"), \
            f"step {name!r} is continue-on-error"
        script = step.get("run") or ""
        for line in script.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "|| true" not in stripped, \
                f"`|| true` in step {name!r}: {stripped!r}"
            assert not re.search(r"\bset\s+\+e\b", stripped), \
                f"`set +e` in step {name!r}: {stripped!r}"


def test_no_assertion_in_the_extracted_script_is_commented_out():
    """The other half of the same question, in the file that does the work."""
    suspicious = []
    for number, line in enumerate(SCRIPT.read_text().splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        body = stripped.lstrip("#").strip()
        if re.match(r"^(assert\b|integrity\.append|coverage\.append)", body):
            suspicious.append((number, stripped))
    assert not suspicious, (
        f"commented-out checks in {SCRIPT.name}: {suspicious}"
    )


def test_the_job_runs_the_extracted_script_rather_than_inline_logic():
    """The reason `tools/ci/` exists (CLAUDE.md §5.5).

    If the verification migrated back into a `run: |` block, everything in the
    first half of this file would still pass while testing nothing the job
    does.
    """
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    steps = _lower_job(doc)["steps"]
    verifier = [s for s in steps if s.get("id") == "lower"]
    assert verifier, "no step with id `lower`"
    script = verifier[0].get("run") or ""
    assert "tools/ci/qnn_lower.py" in script, (
        f"the verifying step no longer invokes the extracted script: {script!r}"
    )
    for marker in ("is_qnn", "match_device", "delegated_fraction"):
        assert marker not in script, (
            f"{marker!r} is being checked inline in the workflow. That copy is "
            f"untestable from here -- put it in tools/ci/qnn_lower.py."
        )


def test_the_runner_is_pinned_and_not_latest():
    """docs/QNN.md's numbers are keyed to hosts; `ubuntu-latest` moves."""
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    for name, job in doc["jobs"].items():
        runs_on = str(job["runs-on"])
        assert "ubuntu-latest" not in runs_on, (
            f"job {name!r} runs on `ubuntu-latest`. Pin the image: the QNN SDK "
            f"host requirement in docs/QNN.md §1.1 is version-specific and a "
            f"rolling image changes it without a commit."
        )
        assert re.search(r"ubuntu-\d\d\.\d\d", runs_on), runs_on


def test_the_soc_default_agrees_across_the_workflow_and_the_script():
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    triggers = doc.get("on") or doc.get(True)
    default = triggers["workflow_dispatch"]["inputs"]["soc_model"]["default"]
    assert default == QL.DEFAULT_SOC_MODEL, (
        f"the workflow defaults to {default!r} and the script to "
        f"{QL.DEFAULT_SOC_MODEL!r}. A dispatch with no input and a `push` run "
        f"would target different silicon."
    )
    # The `||` fallbacks inside the run: block are a second copy of the same
    # default (the `inputs` context is empty on a `push` trigger), so they have
    # to agree too.
    text = WORKFLOW.read_text()
    fallbacks = set(re.findall(r"inputs\.soc_model \|\| '([^']+)'", text))
    assert fallbacks <= {QL.DEFAULT_SOC_MODEL}, (
        f"the workflow's push-trigger fallbacks name {sorted(fallbacks)}, not "
        f"{QL.DEFAULT_SOC_MODEL!r}"
    )


def test_the_workflow_header_says_what_it_does_not_prove():
    """`verify-published-wheel.yml` is the model and this is the habit copied.

    docs/QNN.md §6.4 is a paragraph that has to stay true. A job that produces
    a QNN artefact is the most tempting moment in this whole round to let
    "lowered for the HTP" slide into "ran on the HTP".
    """
    text = WORKFLOW.read_text()
    header = text.split("name: qnn lower")[0]
    assert "PROVES NOTHING" in header, "the header makes no negative claim"
    for cite in ("§6.1", "§6.4", "HTP execution", "physical device"):
        assert cite in header or cite in header.replace("the physical device",
                                                        "physical device"), cite


def test_every_version_the_job_pins_is_pinned_in_one_place():
    """Four numbers from docs/QNN.md §1.1, each declared once.

    `test_release.py`'s docstring records what a second copy of a version costs
    here: CI's default stayed three releases behind and the green runs the
    README cited were measuring an older wheel.
    """
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    env = doc["env"]
    assert env["EXECUTORCH_VERSION"] == "1.4.1", env
    assert env["EXPECTED_QNN_SDK_VERSION"] == "2.37.0.250724", env
    assert env["ANDROID_NDK_VERSION"].startswith("26."), env
    assert env["PYTHON_VERSION"] == "3.13", env
    text = WORKFLOW.read_text()
    # The pins must be *used* through the env, not re-typed. A literal second
    # occurrence outside the `env:` block is the drift this guards.
    for key in ("EXECUTORCH_VERSION", "EXPECTED_QNN_SDK_VERSION",
                "ANDROID_NDK_VERSION", "PYTHON_VERSION"):
        literal = env[key]
        occurrences = [
            line for line in text.splitlines()
            if f'"{literal}"' in line and line.strip().startswith(key)
        ]
        assert len(occurrences) == 1, (
            f"{key} = {literal!r} is declared {len(occurrences)} times"
        )


def test_the_sdk_version_is_asserted_against_executorchs_own_answer():
    """docs/QNN.md §1.1: the SDK version is read, never transcribed.

    The workflow pins the expected value, but what it compares against must be
    `qnn.qnn_sdk_version()` -- ExecuTorch's own `download_qnn_sdk.py` -- rather
    than a second literal. Otherwise the check is a string equalling itself.

    Searched in the `run:` scripts and NOT in the whole file. The first version
    of this test grepped the raw text, and a nullification that replaced the
    call with a literal tuple went **uncaught**: the workflow also names
    `qnn.qnn_sdk_version()` in a comment two lines above, and the comment
    satisfied the grep. That is `CLAUDE.md` §5.5's shape exactly -- a check
    that a prose mention can satisfy is a check on the prose.
    """
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    scripts = "\n".join(
        step.get("run") or ""
        for job in doc["jobs"].values() for step in job["steps"]
    )
    code = "\n".join(
        line for line in scripts.splitlines() if not line.strip().startswith("#")
    )
    assert "qnn.qnn_sdk_version()" in code, (
        "no `run:` step calls qnn.qnn_sdk_version(). The workflow's SDK "
        "version check is comparing a literal against a literal, so it cannot "
        "fail -- docs/QNN.md §1.1 is explicit that the number is read, not "
        "transcribed."
    )
    assert 'os.environ["EXPECTED_QNN_SDK_VERSION"]' in code, (
        "the SDK check no longer reads the pin from the workflow env"
    )
    # The pinned literal must appear ONLY in the `env:` block -- a second copy
    # inside the comparison is the tautology this guards.
    assert "2.37.0.250724" not in code, (
        "the expected SDK version is written as a literal inside a run: "
        "script. It is declared once, in `env:`, and read from there."
    )


def test_the_doc_exists_and_names_the_same_versions():
    if not DOC.is_file():
        return _skip(f"no {DOC.relative_to(REPO)}")
    text = DOC.read_text()
    for pin in ("1.4.1", "2.37.0.250724", "SM8550", "ubuntu-24.04"):
        assert pin in text, f"docs/QNNCI.md does not mention {pin}"


def test_the_doc_does_not_claim_the_job_has_run():
    """It never has. docs/QNNCI.md is written before the first dispatch."""
    if not DOC.is_file():
        return _skip(f"no {DOC.relative_to(REPO)}")
    text = DOC.read_text()
    assert "has never run" in text or "never been run" in text, (
        "docs/QNNCI.md must say the job has not run. CLAUDE.md §4: built, "
        "reached and agreed are three different claims, and this job is not "
        "yet even the first."
    )


def test_the_doc_states_this_files_own_test_count_correctly():
    """CLAUDE.md §5.3: report "N of M", and do not let N drift.

    docs/QNNCI.md §0 and §5.2 both carry the number of tests in this file. A
    stale count is a small lie, but it is the same species as the one
    test_release.py exists for -- four files that have to agree and are edited
    by different hands.
    """
    if not DOC.is_file():
        return _skip(f"no {DOC.relative_to(REPO)}")
    actual = len(re.findall(r"^def test_", pathlib.Path(__file__).read_text(), re.M))
    claimed = set(re.findall(r"\*\*(\d+)\*\* (?:in `rust/torch_c/pytests/test_qnnci\.py`|tests)",
                             DOC.read_text()))
    assert claimed, "docs/QNNCI.md no longer states a test count for this file"
    assert claimed == {str(actual)}, (
        f"docs/QNNCI.md claims {sorted(claimed)} tests in this file; there are "
        f"{actual}."
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
