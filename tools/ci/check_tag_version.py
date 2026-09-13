#!/usr/bin/env python3
"""Refuse a release whose git tag does not name the version being built.

    python tools/ci/check_tag_version.py refs/tags/v0.1.0b3
    python tools/ci/check_tag_version.py v0.1.0b3 --pyproject pyproject.toml

Nothing in this repository checked this before. `tools/wheel/build.py` reads the
version out of the built metadata and never sees the tag; the workflow reads the
tag and never sees `pyproject.toml`. So `git tag v0.1.0b4 && git push --tags`
would have built, verified and published **0.1.0b3** under a release called
v0.1.0b4 -- nine wheels, all internally consistent, all the wrong version, and
the second attempt at the real 0.1.0b4 would then hit PyPI's "file already
exists" for a file nobody meant to upload. PyPI does not allow re-uploading a
filename even after deletion, so that mistake is not undoable.

The comparison is deliberately **not** `tag.lstrip("v") == version`:

* `packaging.version.Version` normalises, so `v0.1.0-beta3`, `v0.1.0b3` and
  `v0.1.0.b3` all parse to the same release. Comparing the parsed objects would
  accept a tag whose *text* is not the version -- and the tag text is what
  appears on the GitHub release page and in every `pip install
  torchnative==...` a reader copies from it. So the parsed forms must agree
  AND the literal strings must agree.
* An unparseable tag is a refusal, not a fallback to string equality. A tag of
  `vlatest` matching a `version = "latest"` would otherwise pass.

Exit 0 and print the version on agreement; exit 1 with the reason otherwise.
The version is also written to `$GITHUB_OUTPUT` as `version=` when that
variable is set, so the workflow has exactly one place the number comes from.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tomllib

#: Stripped from the front of the tag, and from nowhere else. `v` is the only
#: prefix this project has ever used (`git tag -l` is v0.0.2a0 ... v0.1.0b3), and
#: accepting more of them is how a tag stops being readable as the version.
TAG_PREFIX = "v"


def read_version(pyproject: pathlib.Path) -> str:
    """`[project] version`, as written, not as normalised.

    `tomllib` rather than a regex: `version` also appears under
    `[build-system]`-adjacent tables in other projects and a regex over the
    whole file cannot tell those apart from the one that matters.
    """
    if not pyproject.is_file():
        raise SystemExit(f"check_tag_version: no such file: {pyproject}")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    try:
        version = data["project"]["version"]
    except KeyError:
        raise SystemExit(
            f"check_tag_version: {pyproject} has no [project] version. A "
            "dynamic version would have to be resolved here instead, and this "
            "check would have to be told how."
        ) from None
    if not isinstance(version, str):
        raise SystemExit(
            f"check_tag_version: [project] version is {version!r}, not a string")
    return version


def normalise_ref(ref: str) -> str:
    """`refs/tags/v1.2.3` and `v1.2.3` both reduce to `v1.2.3`.

    GitHub hands the workflow `github.ref` in the long form and
    `github.ref_name` in the short one, and both spellings get typed into a
    `workflow_dispatch` box. Accepting one and silently mis-reading the other
    would make the check pass for the wrong reason.
    """
    ref = ref.strip()
    if ref.startswith("refs/tags/"):
        ref = ref[len("refs/tags/"):]
    return ref


def compare(tag: str, version: str) -> list[str]:
    """Every disagreement between the tag and the version, as messages.

    A list rather than the first failure: "the tag has no v" and "the versions
    differ" are different repairs, and a run that reports one at a time costs a
    push per problem.
    """
    problems: list[str] = []
    if not tag:
        return ["the tag is empty -- nothing to compare the version against"]
    if not tag.startswith(TAG_PREFIX):
        problems.append(
            f"the tag {tag!r} does not start with {TAG_PREFIX!r}; every tag in "
            "this repository does, and the workflow's `on.push.tags` filter "
            "only matches those")
        body = tag
    else:
        body = tag[len(TAG_PREFIX):]

    if body != version:
        problems.append(
            f"the tag names {body!r} and pyproject.toml [project] version is "
            f"{version!r}")

    # The parsed comparison is additional, not a substitute: it catches the
    # case where the two strings differ only in normalisation, which is the
    # one a human reviewer is most likely to wave through.
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        problems.append(
            "packaging is not importable, so the normalised comparison could "
            "not be made -- install packaging rather than relying on the "
            "string comparison alone")
        return problems

    try:
        parsed_tag = Version(body)
    except InvalidVersion:
        problems.append(
            f"the tag body {body!r} is not a PEP 440 version, so it cannot be "
            "the version of anything PyPI will accept")
        return problems
    try:
        parsed_version = Version(version)
    except InvalidVersion:
        problems.append(
            f"pyproject.toml [project] version {version!r} is not a PEP 440 "
            "version")
        return problems
    if parsed_tag != parsed_version and body == version:
        # Unreachable by construction (equal strings parse equally); kept as a
        # refusal rather than an assert so a future `packaging` that disagrees
        # with itself is reported instead of trusted.
        problems.append(
            f"{body!r} and {version!r} are the same string but parse to "
            f"{parsed_tag} and {parsed_version}")
    if parsed_tag == parsed_version and body != version:
        problems[-1] += (
            f" -- they normalise to the same release ({parsed_tag}), which is "
            "why this is refused on the literal text: the tag is what a reader "
            "copies into `pip install`")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", help="a tag, or a refs/tags/... ref")
    ap.add_argument("--pyproject", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[2]
                    / "pyproject.toml")
    args = ap.parse_args(argv)

    version = read_version(args.pyproject)
    tag = normalise_ref(args.ref)
    problems = compare(tag, version)
    if problems:
        print(f"check_tag_version: the tag and {args.pyproject} disagree.",
              file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print("  Nothing was built. Either move the tag or edit "
              "pyproject.toml; a wheel built from this pair would be "
              "published under a name that is not its version, and PyPI "
              "never lets a filename be reused.", file=sys.stderr)
        return 1

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"version={version}\n")
            fh.write(f"tag={tag}\n")
    print(f"check_tag_version: {tag} == [project] version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
