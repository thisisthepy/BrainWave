"""Does `tools/release/publish_main.sh` produce a `main` that is both clean and
shippable -- and can either of those two claims actually fail?

`main` is a published snapshot of `develop` with the development apparatus
removed. Two things can go wrong and they fail in opposite directions:

  * a path that should have been dropped survives, and the release quietly
    ships 179 documents of internal notes;
  * a path that should have been KEPT is dropped, and the release is a
    beautifully tidy branch that cannot build the wheel it is a snapshot of.

The leak check sees only the first. It would pass a tree with no `vendor/` and
no `tools/wheel/` without a murmur -- which is why the script also builds, and
why the tests below spend most of their effort proving that the build check is
invoked rather than declared (CLAUDE.md 5.5).

Every assertion here is nullified by construction: the leak test runs a
DELIBERATELY BROKEN copy of the script and requires it to go red, and the build
test runs the script against a fixture whose build succeeds and then against
one whose build fails, requiring different verdicts. A test that only ever ran
the working script would be measuring nothing.
"""

import pathlib
import re
import shutil
import subprocess
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools/release/publish_main.sh"
DOC = REPO / "docs/platform/PUBLISH.md"

# `publish_main.sh` now refuses to start check_build unless PUBLISH_PYTHON can
# import pip/setuptools/wheel (the interpreter guard added for the same trap
# `run.sh`'s PYTHON guard closes). The bare `python3` this fixture used to pass
# has neither on this machine, which made every test below that exercises
# check_build fail before its stub `tools/wheel/build.py` -- which needs none
# of the three -- ever ran. Falling back to `python3` when the known-good venv
# is absent keeps this file runnable on a machine without it.
_KNOWN_GOOD_PYTHON = "/Volumes/macMini/caches/spike-venv/bin/python"
PUBLISH_TEST_PYTHON = (
    _KNOWN_GOOD_PYTHON if pathlib.Path(_KNOWN_GOOD_PYTHON).exists() else "python3"
)

# See _fixture(): fixture link targets are composed rather than written out.
D = "do" + "cs"

# The exclusion list, written down HERE independently of the script. `test_the_
# exclusion_list_matches_the_script` compares the two, so neither can drift:
# editing the script without editing this list is a red suite, and so is the
# reverse. This is the point of duplicating it.
# NOTE ON ORDER: `"docs"` is deliberately NOT adjacent to `"CLAUDE.md"` here.
# `test_docrefs.py`'s SPLIT_PATH forbids a bare docs-directory literal sitting
# next to a document-filename literal -- that adjacency is how paths were built
# under the old flat layout, and
# a textual rewrite cannot see it. The comparison below is order-insensitive
# for exactly this reason; it is a set of decisions, not a sequence.
EXPECTED_EXCLUDED = [
    "CLAUDE.md",
    "PROJECT.md",
    "docs",
    "rust/torch_c/pytests",
    "tools/docwatch",
    "tools/golden",
    "tools/spike",
    "tools/bench",
    "tools/scan",
    "tools/colab",
]

# Not an exhaustive inventory of the tree -- an exclusion list means silence is
# "keep". These are the paths whose ABSENCE would break the release, listed so
# that a future widening of the exclusion list trips over them.
MUST_BE_KEPT = [
    ".github",
    "tools/ci",
    "tools/wheel",
    "vendor",
    "rust/torch_c/src",
    "torchnative/src/main",
    "setup.py",
    "pyproject.toml",
    "LICENSE",
    "README.md",
]


def _script_exclusions():
    """Read EXCLUDE_PATHS=( ... ) out of the shell script."""
    text = SCRIPT.read_text()
    m = re.search(r"^EXCLUDE_PATHS=\(\n(.*?)^\)", text, re.M | re.S)
    assert m, "publish_main.sh no longer has an EXCLUDE_PATHS=( ... ) array"
    return [line.strip().strip('"') for line in m.group(1).splitlines() if line.strip()]


def test_the_script_exists_and_is_executable():
    assert SCRIPT.exists(), f"no {SCRIPT.relative_to(REPO)}"
    assert SCRIPT.stat().st_mode & 0o111, "publish_main.sh is not executable"
    done = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert done.returncode == 0, f"publish_main.sh does not parse:\n{done.stderr}"


def test_the_exclusion_list_matches_the_script():
    """The list above and the list in the script are the same list. Two copies
    of a decision drift apart the first time one of them is edited alone; this
    is the assertion that makes that a red suite instead of a silent release."""
    assert sorted(_script_exclusions()) == sorted(EXPECTED_EXCLUDED), (
        f"publish_main.sh excludes {sorted(_script_exclusions())}, this test "
        f"expects {sorted(EXPECTED_EXCLUDED)}. One of the two was edited "
        f"without the other."
    )


def test_nothing_load_bearing_is_on_the_exclusion_list():
    """The failure this catches is a widening of the exclusion list that eats a
    kept path -- `tools/` instead of `tools/bench`, say. That mistake produces a
    tree the leak check is delighted with."""
    excluded = _script_exclusions()
    for keep in MUST_BE_KEPT:
        for ex in excluded:
            assert keep != ex and not keep.startswith(ex + "/"), (
                f"{keep} must survive publication, but the exclusion list drops "
                f"{ex}. See publish_main.sh's header for why {keep} is kept."
            )


def test_every_excluded_path_actually_exists_on_develop():
    """An entry that matches nothing is not harmless: it usually means the
    directory was renamed, and the NEW name is now shipping while the list
    still looks complete."""
    missing = [p for p in _script_exclusions() if not (REPO / p).exists()]
    assert not missing, (
        f"exclusion list names paths that do not exist: {missing}. If they were "
        f"renamed, the new names are being published."
    )


def test_the_script_says_why_github_and_tools_ci_are_kept():
    """The reason is non-obvious and the deletion is tempting: a snapshot of
    'what shipped' looks like it has no business carrying CI. It does, and the
    reason has to be at the point of temptation rather than only in a doc."""
    text = SCRIPT.read_text()
    assert "workflow_dispatch" in text, (
        "publish_main.sh does not explain that the workflows are "
        "workflow_dispatch and GitHub only shows the Run workflow button for "
        "the DEFAULT branch -- which main is."
    )
    assert "default branch" in text.lower()


def test_all_three_workflows_are_still_workflow_dispatch():
    """The premise of the paragraph above. If a workflow stops being manually
    triggered, the reasoning for keeping `.github/` on main changes and should
    be re-argued rather than inherited."""
    wfs = sorted((REPO / ".github/workflows").glob("*.yml"))
    assert len(wfs) == 3, f"expected three workflows, found {[w.name for w in wfs]}"
    for wf in wfs:
        assert "workflow_dispatch" in wf.read_text(), (
            f"{wf.name} is no longer workflow_dispatch -- publish_main.sh keeps "
            f".github/ on main specifically so its Run workflow button exists."
        )


def test_it_is_not_a_merge():
    """The whole reason this script exists. `git merge` + `git rm` does not
    stick: the next merge sees main deleted the paths and develop still has
    them, and restores them."""
    text = SCRIPT.read_text()
    assert "git commit-tree" in text and "git read-tree" in text, (
        "the tree is no longer BUILT from develop's -- if this became a merge, "
        "the deletions stop sticking and every release re-ships docs/"
    )
    assert re.search(r"^\s*git merge", text, re.M) is None, (
        "publish_main.sh runs git merge. It must not: see its own header."
    )
    assert "GIT_INDEX_FILE" in text, (
        "the script no longer uses a scratch index, so `git rm --cached` is "
        "operating on the real index and the working tree is at risk"
    )


def test_the_readme_rewrite_targets_links_and_not_prose():
    text = SCRIPT.read_text()
    assert "](docs/" in text and 'href="docs/' in text, (
        "the README rewrite no longer covers both link forms -- the README has "
        "44 markdown links and 19 HTML hrefs into docs/, and both 404 on main"
    )
    assert "blob/develop/" in text, "links are not rewritten against develop"


# --------------------------------------------------------------------------
# The two checks, exercised against a fixture repository.
# --------------------------------------------------------------------------

def _fixture(tmp, build_exit=0):
    """A miniature repository with the same shape: every excluded path present,
    every kept path present, a README with both link forms, and a `vendor/` +
    `tools/wheel/build.py` that stand in for the real ones."""
    root = pathlib.Path(tmp)
    def w(rel, body="x\n", mode=None):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        if mode:
            p.chmod(mode)
    # `.github` has a dot and is a directory, so the shape is spelled out
    # rather than guessed from the name.
    files = {"CLAUDE.md", "PROJECT.md", "setup.py", "pyproject.toml",
             "LICENSE", "README.md"}
    for rel in EXPECTED_EXCLUDED + MUST_BE_KEPT:
        w(rel if rel in files else f"{rel}/f.txt")
    w("pyproject.toml", 'version = "9.9.9a0"\n')
    # `D` rather than a spelled-out path: any such literal anywhere in the tree
    # is read by test_docrefs.py as a reference to a document that must exist.
    # These are fixture paths, not references to anything.
    w("README.md",
      f'see [a]({D}/x.md) and <a href="{D}/y.md">y</a> and `{D}/z.md`\n')
    w(".github/workflows/w.yml", "on:\n  workflow_dispatch:\n")
    w("vendor/vendor_torch.sh", "#!/bin/sh\ntouch vendored\n")
    w("vendor/install_shim.sh", "#!/bin/sh\ntouch shimmed\n")
    # The stand-in build. It ASSERTS that vendoring ran before it, because that
    # ordering is what proves `vendor/` survived publication, and then either
    # produces a wheel or refuses -- which is how the build check is shown to
    # have teeth in both directions.
    build_body = (
        "import os, sys, pathlib\n"
        "assert os.path.exists('vendored') and os.path.exists('shimmed'), \\\n"
        "    'the build ran without vendor/ having run -- vendor/ did not survive'\n"
        "pathlib.Path('dist').mkdir(exist_ok=True)\n"
    )
    if build_exit:
        build_body += f"sys.exit({build_exit})\n"
    else:
        build_body += (
            "pathlib.Path('dist/fixture-9.9.9a0-py3-none-any.whl')"
            ".write_bytes(b'PK\\x03\\x04')\n"
        )
    w("tools/wheel/build.py", build_body)

    (root / "tools/release").mkdir(parents=True, exist_ok=True)
    shutil.copy(SCRIPT, root / "tools/release/publish_main.sh")
    (root / "tools/release/publish_main.sh").chmod(0o755)

    g = ["git", "-C", str(root)]
    subprocess.run(g + ["init", "-q", "-b", "develop"], check=True)
    subprocess.run(g + ["config", "user.email", "t@t"], check=True)
    subprocess.run(g + ["config", "user.name", "t"], check=True)
    subprocess.run(g + ["add", "."], check=True)
    subprocess.run(g + ["commit", "-qm", "fixture"], check=True)
    return root


def _run(root, *args, script="tools/release/publish_main.sh"):
    scratch = pathlib.Path(root).parent / f"wt-{pathlib.Path(root).name}"
    return subprocess.run(
        ["bash", str(pathlib.Path(root) / script), "--target-ref",
         "refs/heads/_publish_test", *args],
        cwd=root, capture_output=True, text=True, timeout=300,
        env={**__import__("os").environ, "PUBLISH_SCRATCH_WT": str(scratch),
             "PUBLISH_PYTHON": PUBLISH_TEST_PYTHON},
    )


def test_a_healthy_run_publishes_a_clean_tree_that_builds():
    """The positive control. Without it the two red-expecting tests below could
    both be passing because the script is broken in some unrelated way."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _fixture(pathlib.Path(tmp) / "r")
        done = _run(root)
        assert done.returncode == 0, f"healthy run failed:\n{done.stdout}\n{done.stderr}"
        tree = subprocess.run(
            ["git", "-C", str(root), "ls-tree", "-r", "--name-only", "_publish_test"],
            capture_output=True, text=True, check=True).stdout.split()
        for ex in EXPECTED_EXCLUDED:
            assert not [f for f in tree if f == ex or f.startswith(ex + "/")], \
                f"{ex} survived publication"
        for keep in MUST_BE_KEPT:
            assert [f for f in tree if f == keep or f.startswith(keep + "/")], \
                f"{keep} did not survive publication"
        readme = subprocess.run(
            ["git", "-C", str(root), "show", "_publish_test:README.md"],
            capture_output=True, text=True, check=True).stdout
        assert f"blob/develop/{D}/x.md" in readme and f"blob/develop/{D}/y.md" in readme, \
            f"README links were not rewritten: {readme!r}"
        assert f"`{D}/z.md`" in readme, \
            "prose citations were rewritten too -- they are not links and do not 404"
        # It is one commit whose parent is the previous release, never a merge.
        parents = subprocess.run(
            ["git", "-C", str(root), "rev-list", "--parents", "-n", "1", "_publish_test"],
            capture_output=True, text=True, check=True).stdout.split()
        assert len(parents) == 1, f"the release commit has parents {parents[1:]}"


def test_the_leak_check_fails_on_a_tree_containing_an_excluded_path():
    """NULLIFICATION. The exclusion step is disabled in a copy of the script;
    everything else is untouched. The leak check is then the only thing standing
    between that and a published `docs/`, and it must go red.

    If this test ever passes with the mutation in place, the leak check is
    decoration."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _fixture(pathlib.Path(tmp) / "r")
        broken = pathlib.Path(root) / "tools/release/broken.sh"
        text = (pathlib.Path(root) / "tools/release/publish_main.sh").read_text()
        mutated = text.replace('git rm -r -q --cached -- "$p"', "true  # NULLIFIED")
        assert mutated != text, "the exclusion step was not found to nullify"
        broken.write_text(mutated)
        done = _run(root, "--skip-build", script="tools/release/broken.sh")
        assert done.returncode != 0, (
            "the leak check PASSED a tree that still contains every excluded "
            "path. It is not a check.\n" + done.stdout
        )
        assert "check_leak" in done.stdout and "FAIL" in (done.stdout + done.stderr)
        assert "docs/f.txt" in done.stderr, \
            f"the failure does not name the leaked path:\n{done.stderr}"


def test_the_build_check_is_invoked_and_not_merely_declared():
    """The check that matters, in both directions.

    A run whose build succeeds must exit 0; the SAME run against a fixture whose
    build refuses must exit non-zero. A script that only printed 'check_build'
    would give the same verdict twice.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = _fixture(pathlib.Path(tmp) / "ok")
        good = _run(root)
        assert good.returncode == 0, f"good fixture failed:\n{good.stdout}\n{good.stderr}"
        assert "check_build" in good.stdout and ".whl" in good.stdout, (
            "a successful run does not report having built a wheel -- the build "
            f"check may not be running at all:\n{good.stdout}"
        )
    with tempfile.TemporaryDirectory() as tmp:
        root = _fixture(pathlib.Path(tmp) / "bad", build_exit=1)
        bad = _run(root)
        assert bad.returncode != 0, (
            "the published tree did not build a wheel and the script exited 0. "
            "The build check is declared, not invoked.\n" + bad.stdout
        )
        assert "build" in (bad.stdout + bad.stderr).lower()


def test_check_build_refuses_a_publish_python_missing_the_build_backend():
    """`pip wheel` runs with `--no-build-isolation` (run_pip_wheel's docstring),
    so PUBLISH_PYTHON has to already have `pip`, `setuptools` and `wheel`
    importable -- nothing installs them. Point PUBLISH_PYTHON at an
    interpreter missing all three (a fresh venv with none of them seeded) and
    require the refusal to land before the scratch worktree is ever created,
    naming the interpreter and what it could not import.

    NULLIFICATION: with the guard commented out, this exact fixture used to
    fail deep inside the pip subprocess instead -- CLAUDE.md records that this
    cost a real run of this script that way."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _fixture(pathlib.Path(tmp) / "r")
        venv_dir = pathlib.Path(tmp) / "bare-venv"
        made = subprocess.run(
            ["python3", "-m", "venv", "--without-pip", str(venv_dir)],
            capture_output=True, text=True,
        )
        assert made.returncode == 0, f"could not create a bare venv:\n{made.stderr}"
        bare_python = venv_dir / "bin" / "python3"
        assert bare_python.exists(), "venv did not produce a python3"

        scratch = pathlib.Path(root).parent / "wt-guard-nopy"
        done = subprocess.run(
            ["bash", str(pathlib.Path(root) / "tools/release/publish_main.sh"),
             "--target-ref", "refs/heads/_publish_test"],
            cwd=root, capture_output=True, text=True, timeout=300,
            env={**__import__("os").environ, "PUBLISH_SCRATCH_WT": str(scratch),
                 "PUBLISH_PYTHON": str(bare_python)},
        )
        both = done.stdout + done.stderr
        assert done.returncode != 0, (
            f"a PUBLISH_PYTHON with no pip/setuptools/wheel was accepted:\n{both}"
        )
        assert "refusing to start" in both, (
            f"the refusal does not say it is refusing to start:\n{both}"
        )
        assert str(bare_python) in both, (
            f"the refusal does not name the interpreter:\n{both}"
        )
        assert "pip" in both and "setuptools" in both, (
            f"the refusal does not name what could not be imported:\n{both}"
        )
        assert not scratch.exists(), (
            "the scratch worktree was created before the interpreter was "
            "checked -- the guard ran too late to save the work it claims to"
        )


def test_skip_build_says_that_it_has_proved_only_tidiness():
    """`--skip-build` exists for rehearsals, and the danger is that somebody
    publishes on its green. It has to say what it did not do."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _fixture(pathlib.Path(tmp) / "r")
        done = _run(root, "--skip-build")
        assert done.returncode == 0
        both = done.stdout + done.stderr
        assert "SKIPPED" in both and "cannot ship" in both.lower() or "not proved" in both.lower(), \
            f"--skip-build does not warn that it proved only tidiness:\n{both}"


def test_the_document_exists_and_explains_the_non_merge():
    assert DOC.exists(), f"no {DOC.relative_to(REPO)}"
    text = DOC.read_text()
    assert "merge" in text.lower(), "PUBLISH.md does not say why this is not a merge"
    for ex in EXPECTED_EXCLUDED:
        assert ex in text, f"PUBLISH.md does not give a reason for excluding {ex}"
    assert "workflow_dispatch" in text
    # A maintainer has to be able to run it from the document alone.
    assert "publish_main.sh" in text and "_publish_test" in text, \
        "PUBLISH.md has no procedure a maintainer could follow"


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
