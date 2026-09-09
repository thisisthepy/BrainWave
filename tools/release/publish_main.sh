#!/usr/bin/env bash
# Build `main` as a PUBLISHED SNAPSHOT of `develop`, not a merge of it.
#
# ---------------------------------------------------------------------------
# WHAT `main` IS
# ---------------------------------------------------------------------------
# `main` is what a user who has never read this repository sees: the code that
# is on PyPI, the workflows that verify it, and nothing else. It is not a
# branch anybody develops on and it is never a merge target. `develop` is the
# repository; `main` is one commit per release whose tree is DERIVED from
# develop's.
#
# ---------------------------------------------------------------------------
# WHY THIS IS NOT `git merge` PLUS `git rm`
# ---------------------------------------------------------------------------
# Deleting files in a merge commit does not stick. If release N merges develop
# into main and then deletes `docs/`, release N+1's `git merge develop` sees
# that main deleted those paths and develop still has them -- and three-way
# merge resolves "deleted on one side, unchanged on the other" as DELETE only
# when the other side is unchanged *since the merge base*. The merge base moves
# forward each release, so every doc develop has touched comes back, silently,
# and every doc it has not touched has to be re-deleted by hand every single
# release. Forgetting once ships the entire documentation tree.
#
# So main's tree is BUILT, not merged. We read develop's tree into a scratch
# index, drop the excluded paths from that index, write the tree, and commit it
# with main's previous tip as its only parent. Because it is not a merge, there
# is no "the other side still has it" to restore anything: each release is a
# fresh derivation from develop and main's history stays linear.
#
# The working tree is NEVER touched. Everything below runs through
# GIT_INDEX_FILE pointed at a temporary file, so `git status` in the checkout
# you ran this from is the same before and after.
#
# ---------------------------------------------------------------------------
# WHY `.github/` AND `tools/ci/` ARE KEPT
# ---------------------------------------------------------------------------
# This is the entry most likely to be "cleaned up" by a later hand, so the
# reason is here rather than in a document:
#
#   1. All three workflows (`verify-published-wheel.yml`, `build-cuda-wheel.yml`,
#      `qnn-lower.yml`) are `workflow_dispatch`. GitHub only renders the
#      "Run workflow" button for a workflow file present on the DEFAULT branch
#      -- and `main` is the default branch. Stripping `.github/` would not just
#      hide the workflows from main's tree; it would REMOVE THE MANUAL TRIGGER
#      for all three, repository-wide, while leaving the files on develop
#      looking perfectly healthy.
#   2. `.github/` and `tools/ci/` are a pair. The workflows invoke
#      `tools/ci/verify_published.py` and `tools/ci/qnn_lower.py` by path.
#      Keeping one without the other leaves a green button that fails at
#      "No such file".
#   3. `verify_published.py`'s subject is *what is on PyPI*. That is precisely
#      what main represents, so it belongs on the published snapshot more than
#      it belongs on develop.
#
# ---------------------------------------------------------------------------
# THE TWO CHECKS, AND WHICH ONE MATTERS
# ---------------------------------------------------------------------------
#   check_leak    no excluded path survives in the published tree.
#   check_build   the published tree still BUILDS A WHEEL.
#
# The second is the real one. A wrong exclusion that removes `tools/wheel/` or
# `vendor/` produces a branch that is spotlessly clean and completely useless,
# and the leak check would pass it without a murmur. CLAUDE.md 5.5: a check
# that cannot fail is not a check, and the leak check cannot fail on the one
# mistake that would actually hurt a user.
#
#   ** THE LEAK CHECK ALONE GIVES YOU A TIDY BRANCH THAT CANNOT SHIP. **
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   tools/release/publish_main.sh --target-ref refs/heads/_publish_test   # rehearse
#   tools/release/publish_main.sh                                          # publish
#
#   --source <rev>       what to publish from            (default: develop)
#   --target-ref <ref>   which ref to move               (default: refs/heads/main)
#   --version <v>        message version   (default: read from pyproject.toml)
#   --skip-build         skip check_build. Says loudly that the round has
#                        only proved tidiness. Not for a real publish.
#   --keep-scratch       leave the build worktree for inspection.
#
# Publishing (and the `git push` that follows it) is the coordinating
# session's action with the user's approval -- exactly as the PyPI upload is.
# This script moves a local ref and pushes nothing.

set -euo pipefail

REPO=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd -- "$REPO"

# --- the exclusion list. -----------------------------------------------------
# `rust/torch_c/pytests/test_publish.py` asserts this exact list, reading it
# out of this file, so the two cannot drift apart: change one without the other
# and the gate goes red.
#
# EXCLUDED, with the reason per entry:
#   docs/                   179 documents of round-by-round measurement notes.
#                           They are the record of how the work was done, not
#                           part of what ships, and they are the bulk of the
#                           tree a user would clone.
#   CLAUDE.md, PROJECT.md   instructions to agents working ON the repository.
#                           Meaningless to somebody consuming the wheel, and
#                           actively confusing on the front page. README.md is
#                           kept -- it is the front page.
#   rust/torch_c/pytests/   the development gate: 50-odd suites that need a
#                           vendored tree, a spike venv, emulators and a Vulkan
#                           ICD. None of it runs from an installed wheel.
#   tools/docwatch/         checks markers in `docs/`, which is excluded.
#   tools/golden/           upstream value-comparison harness; needs an
#                           upstream torch installed alongside, which a user of
#                           this distribution by definition does not have.
#   tools/spike/            scratch probes against the local spike venv.
#   tools/bench/            measurement harnesses; their numbers only mean
#                           anything on this project's own machine.
#   tools/scan/             an audit tool over upstream's source, in the same
#                           class as docwatch and golden above.
#   tools/colab/            a notebook for verifying on a Colab box; developer
#                           tooling, not part of the distribution.
#
# KEPT, and why the non-obvious ones:
#   .github/  tools/ci/     see the block above -- workflow_dispatch needs them
#                           on the default branch, and they are a pair.
#   tools/wheel/            builds and verifies the wheel. Without it main
#                           cannot produce the thing it is a snapshot of.
#   vendor/                 lays down the upstream tree and installs `_C`.
#                           `torchnative/src/main/torch/` is gitignored, so
#                           this is the only way any checkout gets one.
#   rust/torch_c/src/       the extension itself.
#   torchnative/src/main/   the Python package.
#   setup.py  pyproject.toml  LICENSE  README.md
#
# `scripts/` is KEPT on purpose rather than by silence: `tools/wheel/build.py`
# names `scripts/device_android.sh build` as the FIX in its own refusal message
# when the Android artefact is stale, so a main without it would print a
# recovery command that main cannot run.
#
# Paths present in the tree and named in NEITHER list (`rust/vk_probe/`,
# `rust/wasm_probe/`, `rust/torch_c/Cargo.toml`, ...) are KEPT. The list is an
# exclusion list: silence means keep, so a path added to the tree later ships
# by default and has to be excluded deliberately.
EXCLUDE_PATHS=(
    "docs"
    "CLAUDE.md"
    "PROJECT.md"
    "rust/torch_c/pytests"
    "tools/docwatch"
    "tools/golden"
    "tools/spike"
    "tools/bench"
    "tools/scan"
    "tools/colab"
)

# `https://github.com/thisisthepy/torchnative/blob/develop/` -- the README on
# main carries ~63 links into `docs/`, every one of which 404s once `docs/` is
# gone. They are rewritten to absolute URLs against develop IN THE PUBLISHED
# TREE ONLY; develop's own README keeps its relative links, which are correct
# there. Both markdown `](docs/...)` and HTML `href="docs/..."` forms.
DOCS_BASE="https://github.com/thisisthepy/torchnative/blob/develop/"

SOURCE=develop
TARGET_REF=refs/heads/main
VERSION=""
SKIP_BUILD=0
KEEP_SCRATCH=0

while [ $# -gt 0 ]; do
    case "$1" in
        --source)      SOURCE=$2; shift 2 ;;
        --target-ref)  TARGET_REF=$2; shift 2 ;;
        --version)     VERSION=$2; shift 2 ;;
        --skip-build)  SKIP_BUILD=1; shift ;;
        --keep-scratch) KEEP_SCRATCH=1; shift ;;
        -h|--help)     sed -n '1,90p' "$0"; exit 0 ;;
        *) echo "publish_main.sh: unknown argument $1" >&2; exit 2 ;;
    esac
done

if [ -z "$VERSION" ]; then
    VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)
fi
[ -n "$VERSION" ] || { echo "publish_main.sh: no version in pyproject.toml" >&2; exit 1; }

SRC=$(git rev-parse --verify "$SOURCE^{commit}")

echo "publish_main.sh"
echo "  source     $SOURCE ($SRC)"
echo "  target     $TARGET_REF"
echo "  version    $VERSION"
echo

# The parent of the new release is whatever $TARGET_REF points at LOCALLY. In
# this checkout that ref was 112 commits behind origin/main and its only reflog
# entry was the original clone -- publishing on top of it would have produced a
# history that forks off the published one. Warn rather than refuse, because a
# scratch target ref legitimately has no upstream.
upstream=$(git rev-parse --verify --quiet "${TARGET_REF#refs/heads/}@{upstream}" 2>/dev/null || true)
if [ -n "$upstream" ] && [ "$upstream" != "$(git rev-parse "$TARGET_REF")" ]; then
    cat >&2 <<EOF
publish_main.sh: WARNING -- $TARGET_REF is not at its upstream.
  local    $(git rev-parse "$TARGET_REF")
  upstream $upstream
The new release commit's parent is the LOCAL tip. Fetch and fast-forward first,
or the published history forks off the one on the remote.
EOF
fi

# --- build the tree ----------------------------------------------------------
# A temporary index, so the real one -- and the working tree -- are untouched.
# `git rm --cached` against $GIT_INDEX_FILE removes entries from THAT index and
# nothing on disk; without this it would delete the coordinating session's
# files.
SCRATCH_INDEX=$(mktemp "${TMPDIR:-/tmp}/publish-main-index.XXXXXX")
cleanup_index() { rm -f "$SCRATCH_INDEX"; }
trap cleanup_index EXIT
export GIT_INDEX_FILE="$SCRATCH_INDEX"

git read-tree "$SRC"

for p in "${EXCLUDE_PATHS[@]}"; do
    if git ls-files --cached --error-unmatch -- "$p" >/dev/null 2>&1; then
        git rm -r -q --cached -- "$p"
        echo "  excluded   $p"
    else
        # Not an error: an entry may legitimately vanish from develop. But say
        # so, because the other reading is that the path was renamed and the
        # new name is now shipping.
        echo "  ABSENT     $p (nothing to exclude -- renamed? then this list is stale)"
    fi
done

# --- rewrite the README's docs links, in the published tree only -------------
# Comments inside kept files (the workflows, `tools/wheel/*.py`, `vendor/*.sh`)
# also cite `docs/...`. Those are COMMENTS: nobody clicks them, they do not
# 404, and rewriting them would enlarge the release diff every round for no
# reader-visible gain. They are deliberately left alone.
readme_blob=$(git rev-parse "$SRC:README.md")
new_readme=$(
    git cat-file blob "$readme_blob" \
    | sed -e "s|](docs/|](${DOCS_BASE}docs/|g" \
          -e "s|href=\"docs/|href=\"${DOCS_BASE}docs/|g" \
    | git hash-object -w --stdin
)
git update-index --cacheinfo "100644,$new_readme,README.md"
rewritten=$(git cat-file blob "$new_readme" | grep -c "$DOCS_BASE" || true)
echo "  README     $rewritten lines rewritten to absolute docs URLs"

TREE=$(git write-tree)
echo "  tree       $TREE"

# --- commit ------------------------------------------------------------------
# One parent: the previous release. Linear history, and NOT a merge -- see the
# header. `commit-tree` rather than `commit` because the latter would want the
# working tree.
if git rev-parse --verify --quiet "$TARGET_REF^{commit}" >/dev/null; then
    PARENT=$(git rev-parse "$TARGET_REF^{commit}")
    NEW=$(git commit-tree "$TREE" -p "$PARENT" -m "Release $VERSION")
    echo "  parent     $PARENT"
else
    NEW=$(git commit-tree "$TREE" -m "Release $VERSION")
    echo "  parent     (none -- first release on $TARGET_REF)"
fi
git update-ref "$TARGET_REF" "$NEW"
echo "  commit     $NEW -> $TARGET_REF"
echo

unset GIT_INDEX_FILE

# --- check 1: leak -----------------------------------------------------------
# Without this a missed path ships silently: the tree is bigger than intended
# and nothing says so.
echo "check_leak: no excluded path may exist in $TARGET_REF"
leaks=$(
    git ls-tree -r --name-only "$NEW" | while IFS= read -r f; do
        for p in "${EXCLUDE_PATHS[@]}"; do
            if [ "$f" = "$p" ] || [ "${f#$p/}" != "$f" ]; then
                echo "$f"
            fi
        done
    done
)
if [ -n "$leaks" ]; then
    echo "FAIL: excluded paths present in the published tree:" >&2
    printf '  %s\n' $leaks >&2
    exit 1
fi
echo "  ok ($(git ls-tree -r --name-only "$NEW" | wc -l | tr -d ' ') files, none excluded)"
echo

# --- check 2: the one that matters -------------------------------------------
# A tidy branch that cannot build a wheel is a worse outcome than a branch with
# a stray document in it, and check_leak passes it happily. So: check the
# published tree out somewhere scratch and actually run the build.
if [ "$SKIP_BUILD" -eq 1 ]; then
    cat >&2 <<'EOF'
check_build: SKIPPED by --skip-build.

  This round has proved that the published tree is TIDY. It has not proved
  that it can ship. An exclusion that removed tools/wheel/ or vendor/ would
  look exactly like this run does. Do not publish on this result.
EOF
    exit 0
fi

# Fail closed on PUBLISH_PYTHON before touching the scratch worktree.
#
# `tools/wheel/build.py` drives the build through `pip wheel --no-build-
# isolation`, which means PUBLISH_PYTHON needs `pip` and the PEP 517 backend
# (`setuptools`, per pyproject.toml's `build-backend = "setuptools.build_meta"`,
# plus `wheel` since isolation is off and nothing will fetch it) already
# importable -- isolation off is deliberate (see run_pip_wheel's docstring),
# so nothing here will install them. This is the same trap as run.sh's:
# a `PUBLISH_PYTHON` without `setuptools` has previously cost a run of this
# exact script, and the failure landed deep inside the pip subprocess instead
# of here, up front.
_publish_py=${PUBLISH_PYTHON:-python3}
_publish_py_missing=$("$_publish_py" - <<'PYEOF' 2>&1 || true
import importlib
missing = []
for mod in ("pip", "setuptools", "wheel"):
    try:
        importlib.import_module(mod)
    except ImportError as exc:
        missing.append(mod + " (" + str(exc) + ")")
if missing:
    print("\n".join(missing))
PYEOF
)
if [ -n "$_publish_py_missing" ]; then
    cat >&2 <<EOF
check_build: refusing to start -- $_publish_py cannot import what the build needs.

Interpreter: $_publish_py
Could not import:
$_publish_py_missing

pip wheel runs with --no-build-isolation (see run_pip_wheel's docstring), so
nothing will install these for you. Nothing has been built yet.

Fix: set PUBLISH_PYTHON to this repo's known-good interpreter and re-run:
    PUBLISH_PYTHON=/Volumes/macMini/caches/spike-venv/bin/python $0
EOF
    exit 1
fi

echo "check_build: $TARGET_REF must still build a wheel"
SCRATCH_WT=${PUBLISH_SCRATCH_WT:-/Volumes/macMini/worktrees/publish-buildcheck-$$}
cleanup_all() {
    cleanup_index
    if [ "$KEEP_SCRATCH" -eq 0 ] && [ -d "$SCRATCH_WT" ]; then
        git worktree remove --force "$SCRATCH_WT" >/dev/null 2>&1 || rm -rf "$SCRATCH_WT"
    fi
}
trap cleanup_all EXIT

git worktree add --detach "$SCRATCH_WT" "$NEW" >/dev/null
echo "  worktree   $SCRATCH_WT"

# `torchnative/src/main/torch/` is gitignored and so is absent from EVERY
# checkout, main's included. This is the step that proves `vendor/` was kept:
# if it had been excluded, this line is where the round dies.
(
    cd "$SCRATCH_WT"
    export PATH="$HOME/.cargo/bin:$PATH"
    sh vendor/vendor_torch.sh
    bash vendor/install_shim.sh
    "${PUBLISH_PYTHON:-python3}" tools/wheel/build.py
) > "${TMPDIR:-/tmp}/publish-main-build.log" 2>&1 && build_status=0 || build_status=$?
if [ "$build_status" -ne 0 ]; then
    echo "FAIL: the published tree does not build a wheel (exit $build_status)" >&2
    echo "  log: ${TMPDIR:-/tmp}/publish-main-build.log" >&2
    tail -40 "${TMPDIR:-/tmp}/publish-main-build.log" >&2
    exit 1
fi

wheel=$(ls -1 "$SCRATCH_WT"/dist/*.whl 2>/dev/null | head -1 || true)
if [ -z "$wheel" ]; then
    echo "FAIL: the build reported success but produced no wheel in dist/" >&2
    exit 1
fi
echo "  ok         $(basename "$wheel") ($(wc -c < "$wheel" | tr -d ' ') bytes)"
echo
echo "published $TARGET_REF = $NEW"
