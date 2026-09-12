"""Tests for `.github/workflows/publish-pypi.yml` and `tools/ci/check_tag_version.py`.

The workflow this file guards replaces a manual release: nine wheels built by
hand on one Mac, then `twine upload`. A workflow file whose only check is
running it in production is the thing that round existed to replace, so
everything here runs without a runner -- the YAML is read as text and as YAML,
and the tag/version script is imported and called.

**Nothing here imports torch, torchnative or executorch**, for the reason
`test_qnnci.py`'s docstring gives: a test that skips on this project's own
machine never runs at all, and the decisions being checked are exactly the ones
a green run never exercises.

Four things are checked, and they are not the same claim:

* **the credentials** -- `environment: pypi`, `id-token: write`, and no token.
  PyPI's Trusted Publisher for thisisthepy/torchnative is registered against
  this *filename* plus that environment name; both are load-bearing strings and
  a rename produces a 403 that does not say which one moved. A password or an
  `api-token` input appearing anywhere is a separate failure -- there is no
  PyPI token in this repository and adding one would silently take the release
  off OIDC.
* **the refusals happen before the work.** The tag/version comparison, and the
  nine-wheels check, must both be able to stop the upload. `needs:` is read off
  the graph rather than inferred from job order in the file, because YAML
  mappings have no order that GitHub honours.
* **the nine are named, not counted.** `build.py`'s `EXPECTED_TARGET_KEYS`
  exists because a dict comprehension silently collapsed two entries into one
  and a count could not say which; the matrix here is checked against that same
  list, so adding a target to the registry and forgetting the workflow is a red
  test rather than an eight-wheel release.
* **nothing disables its own assertions** -- `|| true`, `continue-on-error`,
  `set +e`. CLAUDE.md §5.5.

The `check_tag_version` half is the part that can be run and falsified locally:
it is given agreeing and disagreeing pairs, including the one a reviewer waves
through (`v0.1.0-beta3` against `0.1.0b3`, which normalise to the same release
and are not the same text).
"""

import importlib.util
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[3]
WORKFLOW = REPO / ".github/workflows/publish-pypi.yml"
SCRIPT = REPO / "tools/ci/check_tag_version.py"
BUILD_PY = REPO / "tools/wheel/build.py"
DOC = REPO / "docs/platform/PUBLISH_CI.md"

#: The nine wheel filenames a release consists of, spelled out. Duplicated from
#: the workflow on purpose -- this is the `test_publish.py` pattern, where two
#: independent copies of a list are asserted equal so neither can be edited
#: alone. A test that read the list out of the file it is testing would agree
#: with any edit, including the one that drops a platform.
NINE_PLATFORMS = (
    "android_21_arm64_v8a",
    "ios_12_0_arm64_iphoneos",
    "ios_14_0_arm64_iphonesimulator",
    "macosx_11_0_arm64",
    "manylinux_2_17_aarch64",
    "manylinux_2_17_x86_64",
    "pyemscripten_2026_0_wasm32",
    "win_amd64",
    "win_arm64",
)

#: The `--target` keys the matrix must carry. `host` is `build.py`'s spelling
#: for "no --target at all", which is why it is not in `EXPECTED_TARGET_KEYS`;
#: `android-x86_64` is in that list and is not here, because it refuses by name
#: (build.py `_ANDROID_X86_64_REFUSAL`: no x86-64 Android CPython exists to
#: derive a tag from, and no Apple Silicon host can run the result).
NINE_TARGETS = (
    "android-arm64-v8a",
    "host",
    "ios-arm64",
    "ios-arm64-sim",
    "linux-aarch64",
    "linux-x86_64",
    "wasm32-emscripten",
    "windows-arm64",
    "windows-x86_64",
)


def _skip(reason):
    print(f"     skip -- {reason}")


def _load_script():
    spec = importlib.util.spec_from_file_location("_check_tag_version", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CTV = _load_script()


def _yaml():
    try:
        import yaml
    except ImportError:
        return None
    return yaml.safe_load(WORKFLOW.read_text())


def _text():
    return WORKFLOW.read_text()


# ---------------------------------------------------------------------------
# check_tag_version -- the half that can be falsified without a runner
# ---------------------------------------------------------------------------

def test_an_agreeing_tag_and_version_produce_no_problems():
    assert CTV.compare("v0.1.0b3", "0.1.0b3") == []


def test_a_disagreeing_version_is_reported_and_names_both():
    problems = CTV.compare("v0.1.0b4", "0.1.0b3")
    assert problems, "v0.1.0b4 against 0.1.0b3 was accepted"
    joined = " ".join(problems)
    assert "0.1.0b4" in joined and "0.1.0b3" in joined, joined


def test_a_tag_that_normalises_equal_but_reads_differently_is_refused():
    """`v0.1.0-beta3` and `0.1.0b3` are the same PEP 440 release.

    Refused anyway, and this is the test that says why: the tag text is what
    appears on the GitHub release page and what a reader copies into
    `pip install torchnative==...`. A checker that compared only the parsed
    forms would accept a tag naming something PyPI has never heard of.
    """
    problems = CTV.compare("v0.1.0-beta3", "0.1.0b3")
    assert problems, "v0.1.0-beta3 was accepted against 0.1.0b3"
    assert "normalise" in " ".join(problems), problems


def test_a_tag_without_the_v_prefix_is_refused():
    problems = CTV.compare("0.1.0b3", "0.1.0b3")
    assert problems, "a bare 0.1.0b3 tag was accepted"
    assert "does not start with" in " ".join(problems), problems


def test_a_tag_that_is_not_a_version_at_all_is_refused():
    problems = CTV.compare("vlatest", "latest")
    assert problems, "vlatest/latest was accepted"
    assert "PEP 440" in " ".join(problems), problems


def test_an_empty_tag_is_refused_rather_than_treated_as_a_match():
    assert CTV.compare("", "0.1.0b3"), "an empty tag was accepted"


def test_both_ref_spellings_reduce_to_the_same_tag():
    """`github.ref` is long, `github.ref_name` is short, both get typed."""
    assert CTV.normalise_ref("refs/tags/v1.2.3") == "v1.2.3"
    assert CTV.normalise_ref("v1.2.3") == "v1.2.3"
    assert CTV.normalise_ref("  v1.2.3\n") == "v1.2.3"


def test_the_checker_agrees_with_this_repositorys_own_pyproject():
    """Not a tautology: it proves `read_version` reaches the right table.

    A `read_version` that picked up some other `version` key would still
    return a string, and every test above would still pass on the pairs it is
    handed.
    """
    version = CTV.read_version(REPO / "pyproject.toml")
    assert re.fullmatch(r"\d+\.\d+\.\d+[a-z0-9.]*", version), version
    assert CTV.compare(f"v{version}", version) == []


# ---------------------------------------------------------------------------
# the workflow -- credentials
# ---------------------------------------------------------------------------

def test_the_workflow_is_at_the_registered_filename():
    """The path is part of the Trusted Publisher registration, not a choice."""
    assert WORKFLOW.is_file(), (
        f"no workflow at {WORKFLOW}. PyPI Trusted Publishing for "
        "thisisthepy/torchnative names publish-pypi.yml; a differently named "
        "file cannot authenticate no matter what is in it.")


def test_the_workflow_parses():
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    assert "publish" in doc["jobs"], sorted(doc["jobs"])


def test_the_publish_job_runs_in_the_pypi_environment():
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    env = doc["jobs"]["publish"]["environment"]
    name = env["name"] if isinstance(env, dict) else env
    assert name == "pypi", (
        f"the publish job's environment is {name!r}; the Trusted Publisher is "
        "registered against `pypi` and will reject any other")


def test_the_publish_job_asks_for_an_oidc_token():
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    perms = doc["jobs"]["publish"].get("permissions") or {}
    assert perms.get("id-token") == "write", (
        f"publish permissions are {perms!r}; without id-token: write there is "
        "no OIDC token to exchange and, since there is no API token either, "
        "no way to authenticate at all")


def test_no_other_job_can_mint_an_oidc_token():
    """The token is the credential. Only the job that uploads should hold it."""
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    for name, job in doc["jobs"].items():
        if name == "publish":
            continue
        perms = job.get("permissions") or {}
        assert perms.get("id-token") != "write", (
            f"job {name!r} also requests id-token: write")


def test_the_workflow_carries_no_pypi_token_of_any_kind():
    """A token would silently take the release off Trusted Publishing.

    Searched as text rather than through the parsed document because the ways
    to smuggle one in -- an `env:` value, a `with: password:`, a shell
    `$PYPI_TOKEN` -- do not share a structural shape.
    """
    text = _text()
    for needle in ("PYPI_API_TOKEN", "PYPI_TOKEN", "TWINE_PASSWORD",
                   "TWINE_USERNAME", "__token__", "password:"):
        assert needle not in text, (
            f"{needle!r} appears in the workflow. There is no PyPI token in "
            "this repository and adding one replaces OIDC without saying so.")


def test_the_upload_goes_through_the_pypa_action():
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    uses = [str(s.get("uses", "")) for s in doc["jobs"]["publish"]["steps"]]
    assert any("pypa/gh-action-pypi-publish" in u for u in uses), uses


def test_nothing_is_ever_uploaded_to_testpypi_by_accident():
    """`repository-url` unset means pypi.org. A half-set one is the hazard."""
    text = _text()
    assert "test.pypi.org" not in text, (
        "the workflow mentions test.pypi.org; the registered Trusted Publisher "
        "is for pypi.org and a TestPyPI upload needs its own registration")


# ---------------------------------------------------------------------------
# the workflow -- the refusals, and their order
# ---------------------------------------------------------------------------

def test_the_tag_version_check_actually_runs():
    """Present in the file is not the same as invoked."""
    text = _text()
    assert "tools/ci/check_tag_version.py" in text, (
        "nothing in the workflow runs check_tag_version.py, so `git tag "
        "v0.1.0b4` would publish 0.1.0b3 under that name")


def test_the_tag_version_check_runs_before_anything_is_built():
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    jobs = doc["jobs"]
    checked_in = [n for n, j in jobs.items()
                  if "check_tag_version.py" in str(j.get("steps", ""))]
    assert checked_in == ["preflight"], (
        f"check_tag_version runs in {checked_in}; it must be in preflight "
        "alone, which everything else depends on")
    for name in ("vendor", "build", "gate"):
        needs = jobs[name].get("needs") or []
        needs = [needs] if isinstance(needs, str) else needs
        assert "preflight" in needs, (
            f"job {name!r} does not need preflight, so it would compile "
            f"before the tag was compared to the version: needs={needs}")


def test_the_publish_job_cannot_start_without_the_nine_wheel_check():
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    needs = doc["jobs"]["publish"].get("needs") or []
    needs = [needs] if isinstance(needs, str) else needs
    assert "collect" in needs, (
        f"publish needs {needs}; without `collect` a partial matrix uploads "
        "whatever arrived")
    collect_needs = doc["jobs"]["collect"].get("needs") or []
    assert "build" in collect_needs and "gate" in collect_needs, collect_needs


def test_the_gate_is_run_rather_than_assumed():
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    steps = str(doc["jobs"]["gate"]["steps"])
    assert "pytests/run.sh" in steps, (
        "the gate job does not run rust/torch_c/pytests/run.sh")
    assert "PYTHON=" in steps, (
        "run.sh refuses to start without PYTHON set; the gate job does not "
        "set it, so the job would fail for the wrong reason")


def test_the_matrix_builds_exactly_the_nine_targets():
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    entries = doc["jobs"]["build"]["strategy"]["matrix"]["include"]
    got = tuple(sorted(e["target"] for e in entries))
    assert got == NINE_TARGETS, (
        f"the matrix builds {got}\n  expected {NINE_TARGETS}")


def test_the_matrix_matches_build_pys_own_registry():
    """The registry is the API; the matrix must follow it, not shadow it.

    `EXPECTED_TARGET_KEYS` in build.py is that file's second, independent copy
    of its target list. Adding a target there and not here would ship eight
    wheels out of a nine-target registry with nothing saying so.
    """
    text = BUILD_PY.read_text()
    block = re.search(r"EXPECTED_TARGET_KEYS = \((.*?)\)", text, re.S)
    assert block, "build.py no longer declares EXPECTED_TARGET_KEYS"
    keys = set(re.findall(r'"([a-z0-9_\-]+)"', block.group(1)))
    # `host` is build.py's name for "no --target"; `android-x86_64` refuses.
    expected = (keys - {"android-x86_64"}) | {"host"}
    assert expected == set(NINE_TARGETS), (
        f"build.py's registry implies {sorted(expected)}; this file expects "
        f"{sorted(NINE_TARGETS)}")


def test_the_nine_wheel_filenames_are_named_in_the_workflow():
    """Named, not counted -- a count cannot say which one did not arrive."""
    text = _text()
    missing = [p for p in NINE_PLATFORMS if p not in text]
    assert not missing, (
        f"the workflow's wheel list does not name {missing}")


def test_the_nine_platforms_are_not_secretly_eight():
    assert len(set(NINE_PLATFORMS)) == 9, sorted(NINE_PLATFORMS)
    assert len(set(NINE_TARGETS)) == 9, sorted(NINE_TARGETS)


def test_every_wheel_is_twine_checked_before_the_upload_job():
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    collect = str(doc["jobs"]["collect"]["steps"])
    assert "twine check" in collect, (
        "no `twine check` in the collect job; a wheel twine refuses would be "
        "discovered half-way through an upload, with the earlier files already "
        "live and unreplaceable")
    assert "--strict" in collect, (
        "twine check without --strict passes on metadata warnings")


def test_the_dry_run_input_exists_and_defaults_to_not_uploading():
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    # PyYAML resolves the bare key `on` to the boolean True -- the "Norway
    # problem" -- so the triggers are not at doc["on"].
    triggers = doc[True]
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert "dry_run" in inputs, sorted(inputs)
    assert inputs["dry_run"]["default"] is True, (
        "dry_run defaults to false, so the safe answer is not the default one")


def test_a_dry_run_cannot_reach_the_upload():
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    condition = str(doc["jobs"]["publish"].get("if", ""))
    assert "dry_run" in condition, (
        f"the publish job's `if:` does not consult dry_run: {condition!r}")


def test_the_tag_trigger_exists_and_is_scoped():
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    tags = doc[True]["push"]["tags"]
    assert tags == ["v*"], (
        f"on.push.tags is {tags!r}; anything broader makes every tag a release")


def test_nothing_in_the_workflow_disables_its_own_assertions():
    """CLAUDE.md §5.5, applied to the file that decides what ships."""
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    for name, job in doc["jobs"].items():
        assert not job.get("continue-on-error"), (
            f"job {name!r} is continue-on-error, so every check in it is "
            "advisory")
        for step in job.get("steps", []):
            label = step.get("name") or step.get("uses")
            assert not step.get("continue-on-error"), \
                f"{name}: step {label!r} is continue-on-error"
            for line in (step.get("run") or "").splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert not stripped.endswith("|| true"), \
                    f"{name}: step {label!r} ends a command with `|| true`"
                assert "set +e" not in stripped, \
                    f"{name}: step {label!r} turns off errexit"


def test_the_workflow_does_not_weaken_build_pys_guards():
    """The guards are what caught the two real incidents; CI must not route
    around them.

    `require_current` is the freshness guard that stopped eight 44-hour-stale
    artefacts shipping, and `check_build_cache` is why `.DS_Store` cannot be
    re-copied into every wheel. Both are asserted still present in build.py,
    and the workflow is asserted not to set an environment variable that would
    bypass them.
    """
    text = BUILD_PY.read_text()
    assert "def require_current(" in text, \
        "build.py lost require_current -- the artefact freshness guard"
    assert "def check_build_cache(" in text, \
        "build.py lost check_build_cache"
    wf = _text()
    for escape in ("SKIP_FRESHNESS", "--no-verify", "--force", "TORCHNATIVE_SKIP"):
        assert escape not in wf, \
            f"the workflow passes {escape!r}, which would route around a guard"


def test_the_vendoring_source_is_pinned_to_a_platform():
    """The upstream dist-info is copied verbatim into all nine wheels.

    `torch-2.13.0.dist-info/WHEEL` in the released wheels says
    `Tag: cp313-cp313-macosx_14_0_arm64`. Vendoring from a Linux torch wheel
    changes that member and METADATA in every one of the nine, which is a
    content change nobody asked for. So the platform is pinned, not defaulted.
    """
    text = _text()
    assert "TORCH_WHEEL_PLATFORM" in text, \
        "the workflow does not pin the platform of the upstream torch wheel"
    assert "macosx_14_0_arm64" in text, \
        "the pinned upstream torch platform is not the one the released wheels " \
        "were vendored from"
    assert "--platform" in text and "--only-binary" in text, \
        "the torch wheel is not downloaded for a fixed platform"


def test_the_vendored_tree_is_built_once_and_shared():
    """One tree for nine wheels, so `--reference` can be a real check.

    Today all nine agree because one machine built them in one afternoon.
    That is a property of the afternoon, not of the process.
    """
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    assert "vendor" in doc["jobs"], sorted(doc["jobs"])
    build_needs = doc["jobs"]["build"].get("needs") or []
    assert "vendor" in build_needs, (
        f"the build matrix does not consume the vendor job: {build_needs}")
    build_steps = str(doc["jobs"]["build"]["steps"])
    assert "vendor_torch.sh" not in build_steps, (
        "a build job re-vendors, so the nine wheels can disagree about the "
        "tree they carry")


def test_the_reference_comparison_runs_with_a_real_reference():
    """`verify_cross.py --reference` is the member-list comparison.

    It must be pointed at the macOS wheel of the same version -- which is what
    `_default_reference` picks locally -- and not at a placeholder. A
    `--reference /dev/null` would report on nothing and pass.
    """
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    collect = str(doc["jobs"]["collect"]["steps"])
    assert "verify_cross.py" in collect, \
        "nothing compares the cross wheels against the macOS one"
    assert "/dev/null" not in collect, \
        "verify_cross.py is given /dev/null as its reference"
    assert "macosx_11_0_arm64" in collect, \
        "the reference wheel is not the macOS one"


def test_the_unobtainable_target_pythons_are_refused_early_and_by_name():
    """docs/platform/TARGET_PYTHON.md §5: the Android CPython is not reproducible.

    A runner cannot build it and cannot download it. The workflow must say so
    in preflight rather than discovering it forty minutes into a matrix job --
    and must not quietly substitute a different distribution, which would
    change the wheel's platform tag, because build.py reads ANDROID_API_LEVEL
    out of whatever it is given.
    """
    doc = _yaml()
    if doc is None:
        return _skip("pyyaml is not installed in this interpreter")
    pre = str(doc["jobs"]["preflight"]["steps"])
    assert "TARGET_PYTHON_BUNDLE_URL" in pre, (
        "preflight does not check that the target-python bundle is reachable")
    build = str(doc["jobs"]["build"]["steps"])
    assert "BUNDLE_SHA256" in build, (
        "the bundle is fetched without a checksum -- it is the one input to "
        "the release that nothing else can check")


def test_the_downloadable_target_pythons_are_pinned_to_a_release():
    text = _text()
    assert "python-build-standalone" in text, text[:0]
    assert "20260825" in text, (
        "the python-build-standalone release is not pinned; TARGET_PYTHON.md "
        "§3 and §4 checksum that release specifically")


# ---------------------------------------------------------------------------
# the documentation
# ---------------------------------------------------------------------------

def test_the_doc_exists():
    assert DOC.is_file(), f"no {DOC.relative_to(REPO)}"


def test_the_doc_says_the_workflow_has_never_run():
    """CLAUDE.md §4: written, reached and agreed are three different claims.

    This workflow has been validated without a runner and has not executed.
    Delete this assertion when it has -- and record the run, not the intent.
    """
    if not DOC.is_file():
        return _skip(f"no {DOC.relative_to(REPO)}")
    text = DOC.read_text()
    assert "has never run" in text or "never been run" in text, (
        "docs/platform/PUBLISH_CI.md must state that this workflow has not "
        "been executed")


def test_the_doc_states_this_files_own_test_count_correctly():
    """CLAUDE.md §5.3: a stale count is the small end of a real species."""
    if not DOC.is_file():
        return _skip(f"no {DOC.relative_to(REPO)}")
    actual = len(re.findall(r"^def test_", pathlib.Path(__file__).read_text(), re.M))
    claimed = set(re.findall(
        r"\*\*(\d+)\*\* (?:in `rust/torch_c/pytests/test_cipub\.py`|tests)",
        DOC.read_text()))
    assert claimed, "PUBLISH_CI.md no longer states a test count for this file"
    assert claimed == {str(actual)}, (
        f"PUBLISH_CI.md claims {sorted(claimed)} tests in this file; there are "
        f"{actual}.")


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
