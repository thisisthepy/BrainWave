#!/usr/bin/env python3
"""Standing check for claims `docs/*.md` makes about this tree.

docs/AUDIT.md audited eleven load-bearing documents by hand and found six of
eleven carrying a false claim -- not concentrated, but sharing one mechanism
almost every time: a later, unrelated commit closed a gap the document had
named, and nobody came back to update the document. AUDIT.md's own
conclusion is that this argues for a standing check over another manual
pass. This is that check. docs/DOCWATCH.md is the design record (what was
tried, what was rejected, why).

What it checks
---------------
Free-text prose cannot be parsed reliably -- there is no way to tell "X is
missing" the claim from "X is missing" the quoted example inside a sentence
about something else. So this does not scan prose. It reads a small,
explicit marker instead, an HTML comment placed next to the sentence it
backs (invisible when the document renders):

    `torch.gelu` still has no bare spelling upstream.
    <!-- DOCWATCH: hasattr gelu false -->

Six marker kinds, each backed by exactly one ground-truth source (never a
number baked into this script):

    op-implemented <op>                 _C._aten_implemented(), live
    op-not-implemented <op>             same
    hasattr <attr> true|false           hasattr(upstream torch, attr), live
    json-key <path> <key> present|absent   the JSON file itself
    symbol-in-file <path> <name>        a regex read of the file
    count <name> <eq|ge|le> <value>     one of a fixed registry of live
                                         harness runs -- see COUNT_SOURCES

Usage
-----
    python3 tools/docwatch/check_docs.py [FILES...]

With no FILES, scans every docs/*.md. Needs a Python environment with real
upstream torch installed and TORCH_C_ARTEFACT pointing at a built shim (the
same environment tools/golden/compare.py needs) for anything other than
`symbol-in-file`/`json-key`:

    export TORCH_C_ARTEFACT=/path/to/lib_C.dylib
    /path/to/venv/bin/python3 tools/docwatch/check_docs.py

`--no-live` runs only the static checks (`symbol-in-file`, `json-key`) and
reports every other marker as SKIPPED rather than silently passing --
silently passing a check nobody ran is exactly the false-confidence failure
mode this tool exists to avoid.

`--list` prints every marker found without evaluating any of them, for
measuring how much of a document's claims this tool actually covers.

Exit code is 0 iff every evaluated marker is PASS (SKIPPED does not count
against it; a marker this run could not evaluate is not the same claim as
one it evaluated and found false). Non-zero otherwise. Never grep stdout for
a success marker -- read the exit code, and read it without a pipe in the
way (CLAUDE.md's own rule, and this tool's acceptance criteria follow it).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
SHIM_PROBE = Path(__file__).resolve().parent / "_shim_probe.py"

MARKER_RE = re.compile(r"<!--\s*DOCWATCH:\s*(.+?)\s*-->")

# --------------------------------------------------------------------------
# Claim model
# --------------------------------------------------------------------------


@dataclass
class Claim:
    doc: Path
    line: int
    kind: str
    args: list[str]
    context: str  # nearest preceding non-blank line, for reporting

    def label(self) -> str:
        try:
            rel = self.doc.relative_to(REPO_ROOT)
        except ValueError:
            rel = self.doc
        return f"{rel}:{self.line}"


@dataclass
class Result:
    claim: Claim
    status: str  # PASS | FAIL | ERROR | SKIP
    detail: str

    def format(self) -> str:
        return f"{self.claim.label()}: {self.status:5} {self.claim.kind} {' '.join(self.claim.args)}" + (
            f"  -- {self.detail}" if self.detail else ""
        )


_FENCE_RE = re.compile(r"^\s*```")


def parse_markers(paths: list[Path]) -> list[Claim]:
    """A marker inside a fenced code block is documentation ABOUT the
    marker syntax (docs/DOCWATCH.md is full of these), not a live claim --
    skip fenced regions so this tool's own design doc does not get parsed
    as a claim about the tree."""
    claims: list[Claim] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        in_fence = False
        for i, line in enumerate(lines):
            if _FENCE_RE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            matches = list(MARKER_RE.finditer(line))
            if not matches:
                continue
            context = ""
            for j in range(i - 1, -1, -1):
                if lines[j].strip():
                    context = lines[j].strip().lstrip("> ").lstrip("- ")
                    break
            # A line can carry more than one marker (e.g. a table cell
            # packing several corrected counts side by side) -- iterate
            # every match, not just the first.
            for m in matches:
                # shlex, not str.split(): a `symbol-in-file` name can need a
                # literal space (e.g. the exact text of a signature typo,
                # `"*args, *kwargs"`), and shlex's quoting lets a marker say
                # so without inventing a second delimiter convention.
                tokens = shlex.split(m.group(1))
                if not tokens:
                    continue
                kind, *args = tokens
                claims.append(Claim(doc=path, line=i + 1, kind=kind, args=args, context=context[:140]))
    return claims


# --------------------------------------------------------------------------
# Live ground truth, memoized per run
# --------------------------------------------------------------------------


class LiveFactsError(RuntimeError):
    """Ground truth could not be produced -- distinct from a claim being
    false. A marker whose source raised this is reported ERROR, not FAIL:
    CLAUDE.md's own house style (run.sh's cmp-exit-code guard, build.py's
    three-way verdict) is explicit that collapsing "wrong" and "could not
    tell" into one outcome is the bug, not a simplification."""


class LiveFacts:
    def __init__(self, python_exe: str, env: dict[str, str]):
        self.python_exe = python_exe
        self.env = env
        self._shim_cache: dict | None = None
        self._golden_cache: dict | None = None
        self._schema_cache: dict | None = None
        self._smoke_cache: int | None = None
        self._decomp_cache: dict | None = None
        self._requested_attrs: set[str] = set()

    def preload_attrs(self, attrs: set[str]) -> None:
        self._requested_attrs |= attrs

    # -- shim facts: op-implemented / hasattr -----------------------------
    def shim(self) -> dict:
        if self._shim_cache is None:
            request = json.dumps({"attrs": sorted(self._requested_attrs)})
            proc = subprocess.run(
                [self.python_exe, str(SHIM_PROBE)],
                input=request,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env=self.env,
            )
            if proc.returncode != 0:
                raise LiveFactsError(f"_shim_probe.py exited {proc.returncode}: {proc.stderr.strip()[-800:]}")
            try:
                self._shim_cache = json.loads(proc.stdout.strip().splitlines()[-1])
            except (json.JSONDecodeError, IndexError) as e:
                raise LiveFactsError(f"_shim_probe.py produced no parseable JSON: {e}; stderr={proc.stderr[-400:]}") from e
        return self._shim_cache

    def implemented_ops(self) -> set[str]:
        facts = self.shim()
        if "implemented_error" in facts:
            raise LiveFactsError(f"could not load _C shim: {facts['implemented_error']}")
        return set(facts["implemented"])

    def hasattr_upstream(self, attr: str) -> bool:
        facts = self.shim()
        if attr not in facts.get("hasattr", {}):
            raise LiveFactsError(f"hasattr probe was not asked about {attr!r} (preload bug)")
        return facts["hasattr"][attr]

    # -- tools/golden/compare.py SUMMARY line ------------------------------
    def golden(self) -> dict:
        if self._golden_cache is None:
            proc = subprocess.run(
                [self.python_exe, str(REPO_ROOT / "tools" / "golden" / "compare.py")],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env=self.env,
            )
            m = re.search(
                r"SUMMARY:\s*(\d+)/(\d+) cases passed, (\d+) failed, "
                r"ops covered=(\d+), pending case builders=(\d+)",
                proc.stdout,
            )
            if not m:
                raise LiveFactsError(
                    f"compare.py exit={proc.returncode}, no SUMMARY line found; "
                    f"stderr={proc.stderr[-400:]}"
                )
            passed, total, failed, ops, pending = (int(x) for x in m.groups())
            self._golden_cache = {
                "golden_cases_passed": passed,
                "golden_cases_total": total,
                "golden_cases_failed": failed,
                "golden_ops_covered": ops,
                "golden_pending": pending,
            }
        return self._golden_cache

    # -- rust/torch_c/pytests/verify_schemas.py SUMMARY line ---------------
    def schema(self) -> dict:
        if self._schema_cache is None:
            proc = subprocess.run(
                [self.python_exe, str(REPO_ROOT / "rust" / "torch_c" / "pytests" / "verify_schemas.py")],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env=self.env,
            )
            m = re.search(
                r"SUMMARY:\s*(\d+)/(\d+) table entries matched upstream, (\d+) failed",
                proc.stdout,
            )
            if not m:
                raise LiveFactsError(
                    f"verify_schemas.py exit={proc.returncode}, no SUMMARY line found; "
                    f"stderr={proc.stderr[-400:]}"
                )
            matched, total, failed = (int(x) for x in m.groups())
            self._schema_cache = {
                "schema_entries_matched": matched,
                "schema_entries_total": total,
                "schema_entries_failed": failed,
            }
        return self._schema_cache

    # -- rust/torch_c/pytests/test_shim.py, staged without a cargo build --
    def smoke_ok(self) -> int:
        if self._smoke_cache is None:
            artefact = self.env.get("TORCH_C_ARTEFACT")
            if not artefact or not os.path.isfile(artefact):
                raise LiveFactsError(
                    f"TORCH_C_ARTEFACT={artefact!r} is not a file; smoke_ok needs a built artefact "
                    "(this check does not build one -- run vendor/install_shim.sh or cargo build first)"
                )
            with tempfile.TemporaryDirectory(prefix="docwatch-stage-") as stage:
                so_path = os.path.join(stage, "_C.abi3.so")
                import shutil

                shutil.copy(artefact, so_path)
                env = dict(self.env)
                env["PYTHONPATH"] = stage
                proc = subprocess.run(
                    [self.python_exe, str(REPO_ROOT / "rust" / "torch_c" / "pytests" / "test_shim.py")],
                    capture_output=True,
                    text=True,
                    cwd=REPO_ROOT,
                    env=env,
                )
                ok_count = sum(1 for line in proc.stdout.splitlines() if line.startswith("ok "))
                if proc.returncode != 0 and ok_count == 0:
                    raise LiveFactsError(
                        f"test_shim.py exit={proc.returncode}, produced no 'ok' lines; "
                        f"stderr={proc.stderr[-400:]}"
                    )
                self._smoke_cache = ok_count
        return self._smoke_cache

    # -- rust/torch_c/pytests/decomp_sweep.py, vendored-tree import --------
    def decomp(self) -> dict:
        if self._decomp_cache is None:
            env = dict(self.env)
            env["PYTHONPATH"] = str(REPO_ROOT / "torchnative" / "src" / "main")
            env["TORCH_USE_RTLD_GLOBAL"] = "1"
            proc = subprocess.run(
                [self.python_exe, str(REPO_ROOT / "rust" / "torch_c" / "pytests" / "decomp_sweep.py")],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env=env,
            )
            head = re.search(
                r"implemented\s+(\d+)\s+core\s+(\d+)\s+non-core\s+(\d+)", proc.stdout
            )
            pop = re.search(r"population:\s*(\d+)", proc.stdout)
            if proc.returncode != 0 or not head or not pop:
                raise LiveFactsError(
                    f"decomp_sweep.py exit={proc.returncode}, could not parse summary; "
                    f"stderr={proc.stderr[-400:]}"
                )
            tally: dict[str, int] = {}
            for tm in re.finditer(r"^\s*(\d+)\s+([A-Z_]+)\s*$", proc.stdout, re.MULTILINE):
                tally[tm.group(2)] = int(tm.group(1))
            self._decomp_cache = {
                "decomp_implemented": int(head.group(1)),
                "decomp_core": int(head.group(2)),
                "decomp_non_core": int(head.group(3)),
                "decomp_population": int(pop.group(1)),
                "decomp_lowered": tally.get("LOWERED", 0),
                "decomp_refused": tally.get("REFUSED", 0),
                "decomp_no_case": tally.get("NO_CASE", 0),
                "decomp_capture_raised": tally.get("CAPTURE_RAISED", 0),
            }
        return self._decomp_cache


COUNT_SOURCES = {
    "smoke_ok": lambda lf: lf.smoke_ok(),
    "golden_cases_passed": lambda lf: lf.golden()["golden_cases_passed"],
    "golden_cases_failed": lambda lf: lf.golden()["golden_cases_failed"],
    "golden_cases_total": lambda lf: lf.golden()["golden_cases_total"],
    "golden_ops_covered": lambda lf: lf.golden()["golden_ops_covered"],
    "golden_pending": lambda lf: lf.golden()["golden_pending"],
    "schema_entries_matched": lambda lf: lf.schema()["schema_entries_matched"],
    "schema_entries_total": lambda lf: lf.schema()["schema_entries_total"],
    "decomp_implemented": lambda lf: lf.decomp()["decomp_implemented"],
    "decomp_population": lambda lf: lf.decomp()["decomp_population"],
    "decomp_lowered": lambda lf: lf.decomp()["decomp_lowered"],
}

COUNT_OPS = {
    "eq": lambda a, b: a == b,
    "ge": lambda a, b: a >= b,
    "le": lambda a, b: a <= b,
}


# --------------------------------------------------------------------------
# Static checks (no live shim needed)
# --------------------------------------------------------------------------

_SYMBOL_PATTERNS = [
    r"\bfn\s+{name}\b",  # Rust
    r"\bdef\s+{name}\b",  # Python
    r"\bclass\s+{name}\b",
    r'"{name}"\s*:',  # JSON key
    r"\b{name}\b",  # fallback: bare identifier occurrence
]


def check_symbol_in_file(path_arg: str, name: str, presence: str) -> tuple[bool, str]:
    path = REPO_ROOT / path_arg
    if not path.is_file():
        return False, f"{path_arg} does not exist"
    text = path.read_text(encoding="utf-8", errors="replace")
    found_line = None
    for pat in _SYMBOL_PATTERNS:
        rx = re.compile(pat.format(name=re.escape(name)))
        m = rx.search(text)
        if m:
            found_line = text.count("\n", 0, m.start()) + 1
            break
    else:
        # None of the identifier-shaped patterns matched -- try `name` as a
        # literal substring with no `\b` at all. `\b` is a *word*-boundary:
        # it silently fails to bracket a name containing non-word characters
        # (e.g. `*kwargs`, from a signature typo this tool checks the
        # absence of), which would make this check pass even when the text
        # it is supposed to find is sitting right there.
        idx = text.find(name)
        if idx != -1:
            found_line = text.count("\n", 0, idx) + 1
    exists = found_line is not None
    want_present = presence == "present"
    ok = exists == want_present
    where = f"found at {path_arg}:{found_line}" if exists else "not found"
    return ok, where


def check_json_key(path_arg: str, key: str, presence: str) -> tuple[bool, str]:
    path = REPO_ROOT / path_arg
    if not path.is_file():
        return False, f"{path_arg} does not exist"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return False, f"{path_arg} is not valid JSON: {e}"
    exists = isinstance(data, dict) and key in data
    want_present = presence == "present"
    ok = exists == want_present
    return ok, f"key {key!r} is {'present' if exists else 'absent'} in {path_arg}"


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate(claims: list[Claim], live: LiveFacts | None) -> list[Result]:
    results: list[Result] = []
    for c in claims:
        try:
            if c.kind == "symbol-in-file":
                path_arg, name, presence = c.args[0], c.args[1], c.args[2]
                ok, detail = check_symbol_in_file(path_arg, name, presence)
                results.append(Result(c, "PASS" if ok else "FAIL", detail))

            elif c.kind == "json-key":
                path_arg, key, presence = c.args[0], c.args[1], c.args[2]
                ok, detail = check_json_key(path_arg, key, presence)
                results.append(Result(c, "PASS" if ok else "FAIL", detail))

            elif c.kind == "op-implemented":
                if live is None:
                    results.append(Result(c, "SKIP", "--no-live"))
                    continue
                op = c.args[0]
                implemented = live.implemented_ops()
                ok = op in implemented
                results.append(Result(c, "PASS" if ok else "FAIL", f"{op} {'is' if ok else 'is NOT'} in _aten_implemented()"))

            elif c.kind == "op-not-implemented":
                if live is None:
                    results.append(Result(c, "SKIP", "--no-live"))
                    continue
                op = c.args[0]
                implemented = live.implemented_ops()
                ok = op not in implemented
                results.append(Result(c, "PASS" if ok else "FAIL", f"{op} {'is NOT' if ok else 'IS'} in _aten_implemented()"))

            elif c.kind == "hasattr":
                if live is None:
                    results.append(Result(c, "SKIP", "--no-live"))
                    continue
                attr, expected = c.args[0], c.args[1]
                actual = live.hasattr_upstream(attr)
                want = expected == "true"
                ok = actual == want
                results.append(Result(c, "PASS" if ok else "FAIL", f"hasattr(upstream torch, {attr!r}) = {actual}"))

            elif c.kind == "count":
                if live is None:
                    results.append(Result(c, "SKIP", "--no-live"))
                    continue
                name, op, value_s = c.args[0], c.args[1], c.args[2]
                if name not in COUNT_SOURCES:
                    results.append(Result(c, "ERROR", f"unknown count source {name!r}; known: {sorted(COUNT_SOURCES)}"))
                    continue
                if op not in COUNT_OPS:
                    results.append(Result(c, "ERROR", f"unknown count op {op!r}; known: {sorted(COUNT_OPS)}"))
                    continue
                actual = COUNT_SOURCES[name](live)
                expected = int(value_s)
                ok = COUNT_OPS[op](actual, expected)
                results.append(Result(c, "PASS" if ok else "FAIL", f"{name} = {actual} (claim: {op} {expected})"))

            else:
                results.append(Result(c, "ERROR", f"unknown marker kind {c.kind!r}"))

        except LiveFactsError as e:
            results.append(Result(c, "ERROR", str(e)))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="doc files to check (default: every docs/*.md)")
    ap.add_argument("--no-live", action="store_true", help="skip anything needing the shim/harnesses")
    ap.add_argument("--list", action="store_true", help="list discovered markers and exit, without checking")
    ap.add_argument("--python", default=os.environ.get("DOCWATCH_PYTHON", sys.executable),
                     help="interpreter used for live subprocess checks (needs real upstream torch)")
    args = ap.parse_args()

    if args.files:
        paths = [Path(f).resolve() for f in args.files]
    else:
        paths = sorted(DOCS_DIR.glob("*.md"))

    claims = parse_markers(paths)

    if args.list:
        by_kind: dict[str, int] = {}
        for c in claims:
            by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
            print(f"{c.label()}: {c.kind} {' '.join(c.args)}  ({c.context!r})")
        print(f"\n{len(claims)} marker(s) across {len(paths)} file(s): " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
        return 0

    live = None
    if not args.no_live:
        env = os.environ.copy()
        live = LiveFacts(args.python, env)
        attrs = {c.args[0] for c in claims if c.kind == "hasattr"}
        live.preload_attrs(attrs)

    results = evaluate(claims, live)

    for r in results:
        print(r.format())

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    total = len(results)
    print(
        f"\n{total} marker(s) checked: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )

    bad = counts.get("FAIL", 0) + counts.get("ERROR", 0)
    if bad:
        print(f"\nDOCWATCH: FAIL -- {bad} marker(s) did not hold (see FAIL/ERROR lines above)", file=sys.stderr)
        return 1
    evaluated = total - counts.get("SKIP", 0)
    print(f"\nDOCWATCH: PASS -- {counts.get('PASS', 0)}/{evaluated} evaluated marker(s) hold" + (
        f", {counts['SKIP']} skipped" if counts.get("SKIP") else ""
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
