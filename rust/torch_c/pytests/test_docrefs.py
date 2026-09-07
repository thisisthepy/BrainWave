"""Do the documentation paths this tree writes down still point at documents?

`docs/` was a flat list of 183 files, and roughly six thousand strings across
Markdown prose, Rust comments, Python docstrings, DOCWATCH markers and workflow
files name those files by path. Sorting them into subfolders is only safe if
every one of those strings moves with its target, and a missed one is invisible:
nothing dereferences a path written in a comment.

The specific failure this file was written against is sharper than a broken
link, and it is the shape CLAUDE.md §5.5 records repeatedly -- a check that
cannot fail. `run.sh` fed the documentation checker with a shell glob:

    "$repo_root"/docs/*.md "$repo_root/README.md"

A glob does not recurse. The moment a document moved into `docs/<folder>/` it
would drop out of DOCWATCH silently: the gate stays green, the marker count gets
*smaller*, and nothing says a word. `check_docs.py`'s own default was the same
glob (`DOCS_DIR.glob("*.md")`), which is the more dangerous of the two because
it under-reports for anyone who invokes the checker directly rather than through
the gate. Both are now recursive, and `test_docwatch_enumerates_docs_recursively`
below fails if either reverts.
What this cannot see, stated rather than implied:

  * **Interpolated paths.** `f"docs/platform/RELEASE_{version}.md"` has no literal
    filename, so neither the reference scan nor the split-literal scan can check
    it. Two of these were in `test_release.py` and the gate found them by
    FileNotFoundError, one run apart. If you build a documentation path from a
    variable, the suite that reads it is the only thing that will notice.
  * **References inside the vendored tree** (`torchnative/src/main/torch/`),
    which is upstream's and regenerated.
"""

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[3]
DOCS = REPO / "docs"
RUN_SH = REPO / "rust/torch_c/pytests/run.sh"
CHECK_DOCS = REPO / "tools/docwatch/check_docs.py"

# A `docs/NAME.md` string anywhere in a text file. The leading group catches a
# path or URL prefix so that references to *other* projects' docs are not
# mistaken for ours -- `pypackpack/docs/SPEC.md` and torch-mlir's
# `.../blob/main/docs/architecture.md` both appear in this repository's prose.
# The folder group is what makes this match the post-move spelling. Without it
# the pattern could only match `docs/NAME.md`, so once every reference became
# `docs/<folder>/NAME.md` the check would have matched nothing and passed
# vacuously -- it did exactly that when first written, and the deliberate-fault
# run below is what caught it. A test that cannot fail is not a test.
DOCS_REF = re.compile(
    r"(?P<prefix>[A-Za-z0-9_.:-]*/?)docs/(?:(?P<folder>[a-z_]+)/)?(?P<name>[A-Za-z0-9_.-]+\.md)"
)

# Relative Markdown links inside docs/, e.g. `[`docs/SCALAR.md`](SCALAR.md)`.
# These carry no `docs/` prefix, so the substitution that moved every
# `docs/NAME.md` string did not touch them -- and they only break once the
# linking and linked documents land in *different* folders, which is exactly
# what sorting into subfolders does. Resolved from the linking file's directory.
# The target may contain `/` -- after the sort these links are `../folder/NAME.md`.
# The first version of this pattern excluded `/`, so it matched nothing once the
# links were rewritten and passed vacuously; the deliberate-fault run caught it.
REL_LINK = re.compile(r"\]\((?!https?:|#)(?P<target>[A-Za-z0-9_./-]+\.md)\)")

# Documents named in prose that do not exist, with the reason each is allowed.
# Named, not pattern-matched: a new dangling reference must fail this test, so
# the exception list is the only thing that may grow, and only deliberately.
KNOWN_MISSING = {
    "docs/TAIL2.md": (
        "referenced by docs/kernels/TAIL3.md's opening sentence, but the document "
        "was never committed -- rust/torch_c/pytests/test_tail2.py exists, so the "
        "round happened and only its write-up is missing. Pre-existing; found "
        "while sorting docs/ into subfolders and deliberately not invented here. "
        "Delete this entry when TAIL2.md lands."
    ),
}

# Prefixes that mean the reference belongs to another project, not this one.
FOREIGN_PREFIXES = ("pypackpack/", "torch-mlir/", "https://", "http://")


def _text_files():
    """Every tracked text file, cheaply: walk the tree and skip the places that
    hold binaries or generated trees. `torchnative/src/main/torch/` is upstream's
    vendored tree -- gitignored, regenerated, and not ours to reason about."""
    skip_dirs = {".git", "target", "torch", "__pycache__", "node_modules", ".venv"}
    for p in REPO.rglob("*"):
        if not p.is_file():
            continue
        if any(part in skip_dirs for part in p.relative_to(REPO).parts):
            continue
        if p.suffix.lower() in {".png", ".jpg", ".so", ".dylib", ".a", ".pt", ".bin", ".safetensors", ".whl", ".zip"}:
            continue
        try:
            yield p, p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def test_no_docs_reference_in_the_tree_dangles():
    """Every `docs/[<folder>/]NAME.md` string names a file that exists."""
    dangling = {}
    for path, text in _text_files():
        if path == pathlib.Path(__file__).resolve():
            continue  # this file quotes example paths while explaining itself
        for m in DOCS_REF.finditer(text):
            prefix, folder, name = m.group("prefix"), m.group("folder"), m.group("name")
            # A URL or another project's checkout is not ours to resolve. Judge
            # by the whole line, not the few characters before the match:
            # `https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md`
            # leaves only `main/` in front of `docs/`, which looks local.
            line = text[text.rfind("\n", 0, m.start()) + 1: text.find("\n", m.end())]
            if any(f in line for f in FOREIGN_PREFIXES):
                continue
            ref = f"docs/{folder}/{name}" if folder else f"docs/{name}"
            if ref in KNOWN_MISSING:
                continue
            if folder:
                # Fully-qualified: it must exist exactly where it says.
                exists = (DOCS / folder / name).is_file()
            else:
                # Unqualified (a historical `git show <rev>:docs/NAME.md` path,
                # or a reference nobody re-pointed): accept it if the document
                # exists anywhere under docs/.
                exists = (DOCS / name).is_file() or bool(list(DOCS.glob(f"*/{name}")))
            if not exists:
                dangling.setdefault(ref, []).append(str(path.relative_to(REPO)))
    assert not dangling, (
        "documentation references that point at no file:\n"
        + "\n".join(f"  {r}  <- {', '.join(sorted(set(w))[:4])}" for r, w in sorted(dangling.items()))
    )


def test_known_missing_documents_are_still_missing():
    """The exception list may not outlive its reason. If TAIL2.md is written,
    this fails and the entry must be deleted -- an allow-list that silently
    covers a file that now exists is how an exception becomes permanent."""
    for ref, why in KNOWN_MISSING.items():
        name = ref.split("/")[-1]
        found = list(DOCS.glob(f"*/{name}")) + ([DOCS / name] if (DOCS / name).exists() else [])
        assert not found, (
            f"{ref} now exists ({found[0].relative_to(REPO) if found else ''}) but is still "
            f"listed as KNOWN_MISSING. Remove the entry. Reason it was listed: {why}"
        )


def test_relative_markdown_links_inside_docs_resolve():
    """`[`docs/SCALAR.md`](SCALAR.md)` resolves against the linking file's own
    directory, so it survives a flat tree and breaks the moment the two files
    land in different folders. The `docs/NAME.md` rewrite does not see these."""
    broken = []
    for path in DOCS.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for m in REL_LINK.finditer(text):
            target = (path.parent / m.group("target")).resolve()
            if not target.exists():
                broken.append(f"{path.relative_to(REPO)} -> {m.group('target')}")
    assert not broken, "relative links that do not resolve:\n  " + "\n  ".join(broken)


def test_every_document_lives_in_a_subfolder():
    """The habit this sorting exists to break: a round drops its document at the
    top of `docs/` because there is no visible place for it. `docs/README.md` is
    the index and is the one Markdown file allowed at this level."""
    loose = sorted(p.name for p in DOCS.glob("*.md") if p.name != "README.md")
    assert not loose, (
        f"{len(loose)} document(s) at the top level of docs/: {', '.join(loose)}.\n"
        "Put each in the folder matching its subject -- docs/README.md lists them "
        "and says what belongs in each."
    )


def test_docs_readme_names_every_folder_and_only_real_ones():
    """The index and the tree agree in both directions. A folder missing from the
    index is a folder the next round will not find; a folder in the index that
    does not exist sends them somewhere that is not there."""
    readme = (DOCS / "README.md").read_text(encoding="utf-8")
    on_disk = {p.name for p in DOCS.iterdir() if p.is_dir() and not p.name.startswith(".")}
    # Only the index table's first column, so prose mentioning `docs/`
    # is not mistaken for a folder name.
    named = set(re.findall(r"^\| `([a-z_]+)/` \|", readme, re.M))
    assert on_disk <= named, f"folders not described in docs/README.md: {sorted(on_disk - named)}"
    assert named <= on_disk, f"docs/README.md describes folders that do not exist: {sorted(named - on_disk)}"


def test_docwatch_enumerates_docs_recursively():
    """The nullification guard. Both enumerations must recurse; if either reverts
    to a non-recursive glob, every document in a subfolder leaves DOCWATCH
    silently and the marker count drops while the gate stays green."""
    run_sh = RUN_SH.read_text(encoding="utf-8")
    # Comment lines are stripped first: the fix's own comment quotes the glob it
    # replaced, and a check that cannot tell a comment from code would fail on
    # the explanation of the thing it is checking.
    code = "\n".join(l for l in run_sh.splitlines() if not l.lstrip().startswith("#"))
    invocation = code[code.index("check_docs.py"):]
    assert '"$repo_root"/docs/*.md' not in invocation, (
        "run.sh feeds check_docs.py with a non-recursive shell glob again. A glob "
        "does not descend into docs/<folder>/, so every document there would drop "
        "out of DOCWATCH with no error and a smaller marker count."
    )
    assert 'find "$repo_root/docs" -name' in code, (
        "run.sh no longer enumerates docs/ recursively before calling check_docs.py"
    )
    assert "$doc_files" in invocation, (
        "run.sh computes a recursive doc list but does not pass it to check_docs.py"
    )

    check = CHECK_DOCS.read_text(encoding="utf-8")
    assert 'DOCS_DIR.glob("*.md")' not in check, (
        "check_docs.py's default scan is a non-recursive glob again. This is the "
        "more dangerous of the two: it under-reports for anyone invoking the "
        "checker directly, not just through the gate."
    )
    assert 'DOCS_DIR.rglob("*.md")' in check, (
        "check_docs.py's default scan should be rglob so it finds documents in "
        "docs/<folder>/"
    )


def test_the_documents_are_all_still_here():
    """A move is not a deletion. The count is the one that was measured before
    sorting began; `ge` rather than `eq` because later rounds legitimately add
    documents, and a *drop* is the news."""
    n = len(list(DOCS.rglob("*.md"))) - 1  # minus docs/README.md, added by the sort
    assert n >= 183, (
        f"{n} documents under docs/, but 183 were there before they were sorted "
        f"into subfolders. A move must not lose one."
    )


# A path built from separate string literals -- `os.path.join(REPO, "docs",
# "CUDA.md")` or `root / "docs" / "SETITEM.md"` -- is invisible to any check
# that greps for `docs/NAME.md`, because that string never appears in the source.
# Four of these survived the textual rewrite and the gate caught them by
# FileNotFoundError, one suite at a time. This is the cheaper way to find them.
SPLIT_PATH = re.compile(r'"docs"\s*[,/]\s*"(?P<name>[A-Za-z0-9_.-]+\.md)"')


def test_no_path_is_built_from_split_docs_literals():
    """`"docs", "NAME.md"` does not resolve now that documents live in folders,
    and no textual rewrite of `docs/NAME.md` can see it."""
    bad = []
    for path, text in _text_files():
        if path == pathlib.Path(__file__).resolve():
            continue
        if path.suffix not in {".py", ".rs", ".sh", ".c"}:
            continue  # a path is built in code; prose quoting one is not a path
        for m in SPLIT_PATH.finditer(text):
            bad.append(f"{path.relative_to(REPO)}: \"docs\", \"{m.group('name')}\"")
    assert not bad, (
        "documentation paths built from split string literals, which point at "
        "the old flat layout and which a `docs/NAME.md` rewrite cannot see:\n  "
        + "\n  ".join(bad)
        + "\nWrite the folder in: os.path.join(REPO, \"docs\", \"<folder>\", \"NAME.md\")."
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
