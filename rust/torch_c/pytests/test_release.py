"""Is the release consistent with itself, and does it claim only what shipped?

A release is four files that have to agree and are edited by four different
hands: `pyproject.toml` carries the version, `README.md` says what is on PyPI,
`.github/workflows/verify-published-wheel.yml` decides which published wheel CI
installs, and `tools/ci/verify_published.py` decides what is asked of it. Every
one of those has been wrong at least once in this repository, and none of the
mistakes was visible from inside any single file:

  * CI's default stayed at `0.0.9a0` through three releases, so the green runs
    that the README cites as "Linux and Windows compute" were computing an
    older wheel than the one the README's table says is published.
  * A count marker was written `eq` where a later round could legitimately
    raise the number, which turns a correct improvement into a red suite. That
    has now happened three times, most recently in `docs/devices/MPS.md`.

None of this checks whether the release *works* -- the rest of the suite does
that. It checks that the four files tell one story, and that the story is not
ahead of PyPI.
"""

import pathlib
import re
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
PYPROJECT = REPO / "pyproject.toml"
README = REPO / "README.md"
WORKFLOW = REPO / ".github/workflows/verify-published-wheel.yml"
VERIFY = REPO / "tools/ci/verify_published.py"


def _version_tuple(s):
    """(0, 1, 0, 1, 0) from '0.1.0b0'. Enough to ORDER this project's own
    versions; deliberately not a PEP 440 parser, because a dependency on
    `packaging` here would make the suite refuse on a host that has none.

    The pre-release letter is part of the ordering, not decoration: `a` sorts
    before `b`, so 0.1.0a9 < 0.1.0b0 < 0.1.0b1. This accepted `a` alone until
    0.1.0b0, and the assertion below fired on the first beta -- which is the
    right failure, but it is worth knowing that it is a NUMBERING limit here
    and not a claim about the version."""
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)([ab])(\d+)", s)
    assert m, f"not an X.Y.Z[a|b]N version: {s!r}"
    major, minor, patch, letter, serial = m.groups()
    return (int(major), int(minor), int(patch),
            {"a": 0, "b": 1}[letter], int(serial))


def _project_version():
    for line in PYPROJECT.read_text().splitlines():
        if line.startswith("version = "):
            return line.split('"')[1]
    raise AssertionError("no `version = ` line in pyproject.toml")


def test_the_version_is_a_prerelease_and_parses():
    """A pre-release, which is `a` or `b` -- not `a` alone.

    This asserted `"a" in v` until 0.1.0b0 and fired on the first beta. The
    intent was never "must be alpha"; it was "must be a PRE-release, because
    the README tells users to pass `--pre` and a final version would make that
    instruction wrong". Widened to what it meant.
    """
    v = _project_version()
    _version_tuple(v)
    assert re.search(r"\d[ab]\d", v), (
        f"{v} is not a pre-release. Every release of this project so far has "
        f"been one, and the README tells users to pass --pre; a final version "
        f"would make that instruction wrong."
    )


def test_pyproject_is_the_only_place_a_version_is_declared():
    """`setup.py` exists and deliberately declares no version; `Cargo.toml`
    holds `0.0.0` because the crate is not what is published. If either grows
    a version they will disagree with pyproject the first time one is bumped."""
    setup = (REPO / "setup.py").read_text()
    assert not re.search(r"^\s*version\s*=", setup, re.M), (
        "setup.py declares a version now. pyproject.toml is the authority; two "
        "declarations means one of them goes stale at the next release."
    )
    cargo = (REPO / "rust/torch_c/Cargo.toml").read_text()
    assert re.search(r'^version = "0\.0\.0"', cargo, re.M), (
        "rust/torch_c/Cargo.toml no longer carries the placeholder 0.0.0 -- if "
        "the crate version is now meaningful it has to be kept in step here."
    )


def test_the_readme_does_not_claim_an_unpublished_version_is_on_pypi():
    """The README's platform table has a row naming the version on PyPI. It may
    lag the version being prepared -- that is the normal state between the bump
    and the upload -- but it may never lead it, which would be a claim about a
    wheel nobody can install."""
    claimed = re.findall(r"\*\*on PyPI `([^`]+)`\*\*", README.read_text())
    assert claimed, "the README no longer names the version that is on PyPI"
    project = _version_tuple(_project_version())
    for c in claimed:
        assert _version_tuple(c) <= project, (
            f"README says {c} is on PyPI, but the version being built is "
            f"{_project_version()}. The README is ahead of the release."
        )


def test_ci_never_points_at_a_version_that_is_not_published_yet():
    """`verify-published-wheel.yml` installs *from PyPI*. Defaulting it to the
    version in pyproject.toml before the upload makes every push a red run that
    says nothing about any platform -- and a red CI that is expected to be red
    is a check nobody reads."""
    text = WORKFLOW.read_text()
    defaults = set(re.findall(r"inputs\.version \|\| '([^']+)'", text))
    defaults |= set(re.findall(r'^\s*default: "([^"]+)"', text, re.M))
    assert defaults, "the workflow no longer has a default version"
    project = _version_tuple(_project_version())
    for d in defaults:
        assert _version_tuple(d) <= project, (
            f"CI defaults to {d}, which is newer than the {_project_version()} "
            f"being prepared here."
        )
    assert len(defaults) == 1, (
        f"the workflow names more than one default version {sorted(defaults)} -- "
        f"the download step and the install step would test different wheels"
    )


def test_every_check_section_is_actually_called():
    """A section added to `verify_published.py` and never wired into `main()`
    is invisible: the file still exits 0 and the platform still reports ALL
    PASS, having asked nothing."""
    text = VERIFY.read_text()
    sections = set(re.findall(r"^def (\w+)\(torch\):", text, re.M))
    body = text[text.index("def main():"):]
    called = set(re.findall(r"^\s+(\w+)\(torch\)", body, re.M))
    missing = sections - called
    assert not missing, f"defined in verify_published.py but never called: {sorted(missing)}"


def test_the_newer_checks_can_skip_on_an_older_published_wheel():
    """This script runs against **published** wheels, so a check written for
    the release being prepared will outrun PyPI until the upload. Reporting
    that as a platform failure would be a lie about the platform, so each new
    section skips by name and says so."""
    text = VERIFY.read_text()
    assert text.count("Not a platform result.") >= 3, (
        "fewer skip-by-name paths than sections that need one"
    )
    assert "def training(torch):" in text, (
        "the loss.backward() check is gone -- it is the headline of 0.0.13a0"
    )
    assert "_predates(" in text, "the version-based skip helper is gone"


def test_count_markers_use_ge_wherever_another_round_could_raise_them():
    """`eq` on a number that a later correct round can raise turns growth into
    a red suite. Three separate rounds in this repository have made that
    mistake. The legitimate `eq`s are the two counts that are zero and must
    stay zero -- `golden_pending` and `golden_cases_failed`. For those a
    change in either direction is news rather than progress, and `ge` would
    be worse than useless: `golden_cases_failed ge 0` holds for every number.

    `golden_cases_failed` was added after the absence of it let a failing
    golden case ride three commits with the gate green. The only two markers
    were `ge` floors on *passed* and on *total*, and a pair of floors cannot
    see `passed < total`."""
    zero_and_must_stay_zero = {"golden_pending", "golden_cases_failed"}
    offenders = []
    for path in sorted(REPO.glob("docs/*.md")) + [README]:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in re.finditer(r"DOCWATCH: count (\w+) (\w+) (\d+)", line):
                name, op, _ = m.groups()
                if op == "eq" and name not in zero_and_must_stay_zero:
                    offenders.append(f"{path.relative_to(REPO)}:{i} count {name} eq")
    assert not offenders, (
        "count markers pinned with `eq` on a number that can legitimately "
        f"rise: {offenders}"
    )


def test_the_release_notes_exist_and_split_the_work_four_ways():
    """CLAUDE.md 5.3: a release note that reports one number has merged
    implementation, defect fixes, measurement and documentation into a single
    figure that looks like progress. The four headings are the whole point."""
    notes = REPO / f"docs/platform/RELEASE_{_project_version().replace('.', '_')}.md"
    assert notes.exists(), f"no release notes at {notes.relative_to(REPO)}"
    text = notes.read_text()
    for heading in ("Features added", "Defects fixed", "Measured but not implemented", "Documentation corrected"):
        assert heading in text, f"release notes have no `{heading}` section"


def test_the_release_notes_state_the_gaps_and_not_only_the_additions():
    """An alpha note that lists what landed and omits that 82 architectures do
    not forward, and that `torch.compile` is recommended for permanent refusal,
    is selling something."""
    text = (REPO / f"docs/platform/RELEASE_{_project_version().replace('.', '_')}.md").read_text()
    assert "82" in text and "297" in text, (
        "the notes do not carry the architecture coverage denominator"
    )
    assert "torch.compile" in text, "the notes do not mention the torch.compile gap"
    assert "ARCH100" in text and "COMPILE" in text, (
        "the notes do not point at the documents that measured the gaps"
    )


def test_the_abi3_wheel_loads_on_later_cpythons():
    """The abi3 promise, exercised rather than asserted.

    `setup.py` passes `py_limited_api = "cp313"` so one wheel serves 3.13 AND
    every later CPython, and `pyproject.toml` now advertises 3.14 and 3.15 on
    the strength of that. Until this test there was nothing behind either
    claim: no gate ran a single line on an interpreter newer than the one it
    runs on, so "abi3 works" was a sentence in a docstring.

    This loads the BUILT extension under whichever later interpreters exist on
    the machine and makes it compute. It skips BY NAME when none are present,
    because the alternative -- passing silently on a host with only 3.13 --
    is a check that cannot fail (CLAUDE.md §5.5).
    """
    artefact = REPO / "torchnative/src/main/torch/_C.abi3.so"
    if not artefact.exists():
        print("   (skipped: no built _C.abi3.so; run vendor/install_shim.sh)")
        return

    floor = (3, 13)
    later = []
    for minor in range(floor[1] + 1, floor[1] + 8):
        exe = shutil.which(f"python3.{minor}")
        if exe:
            later.append((minor, exe))
    if not later:
        print(
            f"   (skipped: no CPython newer than 3.{floor[1]} on PATH, so the "
            "abi3 floor could not be exercised above itself)"
        )
        return

    probe = (
        "import importlib.util, sys, pathlib;"
        "p = pathlib.Path(sys.argv[1]);"
        "spec = importlib.util.spec_from_file_location('_C', p);"
        "m = importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(m);"
        "ops = m._aten_implemented();"
        "a = m._aten_dispatch('aten.empty.memory_format', [2, 2], m.float32);"
        "print(sys.version_info[:2], len(ops), tuple(a.shape))"
    )
    for minor, exe in later:
        done = subprocess.run(
            [exe, "-c", probe, str(artefact)],
            capture_output=True, text=True, timeout=120,
        )
        assert done.returncode == 0, (
            f"the cp313-abi3 extension failed to load on python3.{minor} "
            f"({exe}). That is the abi3 promise breaking, and pyproject.toml "
            f"advertises this interpreter.\n{done.stderr[-800:]}"
        )
        head = done.stdout.strip().splitlines()[-1]
        assert f"(3, {minor})" in head, (done.stdout, done.stderr)
        count = int(head.split()[2].rstrip(","))
        assert count > 100, f"python3.{minor} loaded it but sees {count} ops"
        print(f"   abi3 on python3.{minor}: {head}")


def test_every_suite_prints_its_passes_in_the_one_format():
    """`ok` + three spaces, everywhere, so a pass can be COUNTED.

    The gate's only defence against a test file that dies at import is
    comparing the `ok` count against a baseline -- a crashed file reports no
    FAIL at all, so `FAIL=0` on its own means nothing. That comparison is
    arithmetic on a grep, and a suite that prints `ok name:` with one space
    is invisible to it.

    This is not hypothetical and it is not once: `test_intelnpu.py` hid 26
    passes this way, and `test_devicens.py` and `test_tntransformers.py` hid
    39 more the same day. Both times the count looked like tests that had
    silently stopped running.
    """
    # Matches the STRING LITERAL, not `print(...)`. The first version of this
    # check keyed on a line starting `print("ok `, and eleven passes escaped it
    # by being written as
    #     print(
    #         f"ok tntransformers: ..."
    # -- the literal on its own line. A guard against an uncountable pass that
    # cannot see half the ways one is written is the same defect one level up.
    offenders = []
    for path in sorted((REPO / "rust/torch_c/pytests").glob("test_*.py")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                # Prose, not a pass. Without this the check flags its own
                # comment explaining the bad form -- which it did, twice.
                continue
            for m in re.finditer(r'["\']ok (?! )', line):
                offenders.append(f"{path.name}:{i}")
    assert not offenders, (
        "these print a pass in a format the gate's count cannot see -- use "
        f"`ok` followed by THREE spaces: {offenders}"
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
