"""Reachability audit — is every declared name wired to a kernel, and every
kernel wired to a name?

`tools/golden/compare.py` is the correctness number and it is structurally
blind to this question, by construction: it calls `_C._aten_dispatch(key, ...)`
with a dispatch key it took from its own case table. It therefore cannot see

  1. a name in `overloads.json` / `methods.json` whose keys have no arm in
     `aten_dispatch_inner`'s `match op` — the table entry exists, the resolver
     picks it, and the call dies with "aten op not implemented"
     (`squeeze.default`, `squeeze.dims`, `where.ScalarSelf` arrived this way),
  2. a kernel that no Python spelling reaches — `_aten_implemented()` lists it,
     golden compares it, and nothing in `torch.*` or `Tensor.*` can call it
     (twenty-two names in docs/SPELLINGS.md, six more in its §9),
  3. a spelling that exists and dispatches but that no test ever *spells* —
     golden proves the kernel, not the door to it (docs/DEMAND5.md's
     `torch.roll`).

Each of those was found by a human noticing or a model tripping over it, four
times. This file is the structural version. docs/REACH.md is the inventory it
produced and the reasoning behind the allowlist.

Cost: no builds, no upstream import in the default path, one zero-argument
`_aten_dispatch` per declared key (a few hundred `TypeError`s), and one regex
pass over `pytests/*.py`. It is meant to be cheap enough that nobody switches
it off.

Run it standalone:

    TORCH_C_ARTEFACT=... python3 tools/golden/reach.py
    TORCH_C_ARTEFACT=... python3 tools/golden/reach.py --verify-upstream

or from the suite, through `reach.check(_C, repo_root)`.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ALLOW_PATH = pathlib.Path(__file__).resolve().parent / "reach_allow.json"

# The string `aten_not_implemented` produces. Matching on this exact wording is
# the whole discrimination between shape 1 ("no arm") and a live arm refusing
# the empty argument tuple ("missing required argument"), so it is spelled once
# here and asserted against a known-absent op in `self_test`.
_NO_ARM = "aten op not implemented in torch._C shim"


# --- the three questions ----------------------------------------------------


def declared(module) -> dict:
    """`{(table, name): [dispatch key, ...]}` for both resolution tables.

    Read from `_shim_overloads`/`_shim_methods` — the parsed tables the running
    artefact will actually resolve against — rather than from the JSON files,
    so that a table the loader rejected or rewrote cannot pass this audit while
    a different table runs.
    """
    out = {}
    for table, attr in (("overloads", "_shim_overloads"), ("methods", "_shim_methods")):
        for name, keys in getattr(module, attr).items():
            out[(table, name)] = list(keys)
    return out


def has_dispatch_arm(module, key: str) -> bool:
    """Does `aten_dispatch_inner` have an arm for this key?

    Probed by calling it with no arguments. An arm that exists rejects the
    empty tuple by naming a missing argument (or, for an op whose arguments are
    all optional, runs); an arm that does not exist raises
    `NotImplementedError` with `_NO_ARM`. Nothing here needs the op to be
    callable, only to be *reached*, which is what distinguishes this from the
    golden harness's job.
    """
    try:
        module._aten_dispatch(key)
    except NotImplementedError as exc:
        return _NO_ARM not in str(exc)
    except BaseException:
        return True
    return True


def code_string_constants(text: str) -> set:
    """Every string literal in `text` that is *code* — not a docstring, and not
    a comment (comments are not literals at all).

    Shape 2 asks whether a kernel is reachable through a Python spelling, and
    `bootstrap.py` reaches several kernels from composites rather than from the
    tables (`softmax`, `index_put_`, the sdpa path, `__getitem__`'s slicing).
    Those are found by looking for the dispatch key as a string literal in that
    file — but a key *named in prose* is not a spelling, and that file is more
    prose than code. Deleting a composite and leaving its docstring behind would
    otherwise stay green, which is the exact shape of the failures this check
    exists for.

    Via `ast` rather than a token scan: the first cut of this used "a string
    whose previous token ends a line" as the docstring test, and that reads every
    element of a multi-line list literal as a docstring — it dropped
    `aten.slice.Tensor`, which `__getitem__` very much does reach, and reported
    it as unspelled. A wrong gap in a gap report is the failure mode of the
    whole idea.
    """
    tree = ast.parse(text)
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        # `body` is a statement list on Module/Class/Function/If/..., but a bare
        # expression on `IfExp` and `Lambda`.
        if not isinstance(body, list):
            continue
        for child in body:
            if (isinstance(child, ast.Expr)
                    and isinstance(child.value, ast.Constant)
                    and isinstance(child.value.value, str)):
                docstrings.add(id(child.value))
    return {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in docstrings
    }


_KEY_LITERAL = re.compile(r"aten\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+")


def composite_keys(repo_root: pathlib.Path) -> set:
    src = (repo_root / "rust/torch_c/src/bootstrap.py").read_text()
    return {s for s in code_string_constants(src) if _KEY_LITERAL.fullmatch(s)}


def _blank_line_comments(text: str) -> str:
    """Blank `#` to end of line, per line, respecting quotes on that line.

    Line-based rather than `tokenize`-based, and that is the whole point: a
    third of this suite's coverage lives inside *road scripts* — Python source
    held as a triple-quoted string and run in a vendored-tree subprocess
    (docs/SPELLINGS.md §9). To the tokenizer those are one STRING token, so a
    `#` comment inside one is not a comment at all, and the paragraph inside a
    road script that says `torch.roll(...)` was read as a call to it. The
    scanner is deliberately naive about multi-line strings for the same reason:
    it is being pointed *into* them on purpose.
    """
    out = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n")
        quote = None
        cut = None
        i = 0
        while i < len(body):
            ch = body[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
            elif ch in "'\"":
                quote = ch
            elif ch == "#":
                cut = i
                break
            i += 1
        if cut is not None:
            body = body[:cut] + " " * (len(body) - cut)
        out.append(body + line[len(line.rstrip("\n")):])
    return "".join(out)


def executable_text(text: str) -> str:
    """`text` with comments and docstrings blanked, positions preserved.

    A comment that *mentions* a call is not a test that makes it. The first cut
    of shape 3 counted one: deleting all three `torch.roll(...)` calls from the
    suite left the check green, because the paragraph above them explaining why
    they exist says `torch.roll(...)` too. A gap check that a comment can
    satisfy is not a gap check.

    Blanking spans rather than deleting them keeps line and column numbers
    intact, so a future caller can report a hit's location.
    """
    lines = text.splitlines(keepends=True)
    spans = []
    tree = ast.parse(text)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for child in body:
            if (isinstance(child, ast.Expr)
                    and isinstance(child.value, ast.Constant)
                    and isinstance(child.value.value, str)):
                v = child.value
                spans.append((v.lineno, v.col_offset, v.end_lineno, v.end_col_offset))
    for lineno, col, end_lineno, end_col in spans:
        for row in range(lineno, end_lineno + 1):
            line = lines[row - 1]
            start = col if row == lineno else 0
            stop = end_col if row == end_lineno else len(line.rstrip("\n"))
            keep_nl = line[len(line.rstrip("\n")):]
            body_text = line.rstrip("\n")
            lines[row - 1] = (
                body_text[:start] + " " * (stop - start) + body_text[stop:] + keep_nl
            )
    return _blank_line_comments("".join(lines))


def test_corpus(repo_root: pathlib.Path) -> str:
    """Everything the smoke suite can execute, as text.

    `pytests/*.py` only. The golden harness is deliberately excluded: it
    dispatches by key and never spells a name, which is the blindness this file
    exists to cover, so counting it as coverage would defeat the check.
    """
    paths = sorted((repo_root / "rust/torch_c/pytests").glob("*.py"))
    return "\n".join(executable_text(p.read_text()) for p in paths)


def spelling_exercised(corpus: str, name: str) -> bool:
    """Does any test call `torch.<name>(...)` or `<something>.<name>(...)`?

    Loose on purpose — `.name(` matches a method call on anything, so this
    over-reports coverage rather than under-reporting it. A false "covered"
    costs one name in the allowlist; a false "uncovered" would put the whole
    check into the category of gates people switch off.
    """
    esc = re.escape(name)
    return bool(
        re.search(r"\btorch\.%s\s*\(" % esc, corpus)
        or re.search(r"_VariableFunctions\.%s\b" % esc, corpus)
        or re.search(r"\.%s\s*\(" % esc, corpus)
    )


# --- the audit --------------------------------------------------------------


def audit(module, repo_root=REPO_ROOT) -> dict:
    repo_root = pathlib.Path(repo_root)
    decl = declared(module)
    arm = {}
    for keys in decl.values():
        for key in keys:
            if key not in arm:
                arm[key] = has_dispatch_arm(module, key)

    dead_names = []       # shape 1: no key of this name has an arm
    dead_keys = []        # shape 1, weaker: some overload of this name is dead
    for (table, name), keys in sorted(decl.items()):
        missing = [k for k in keys if not arm[k]]
        if missing and len(missing) == len(keys):
            dead_names.append({"table": table, "name": name, "keys": keys})
        else:
            dead_keys.extend(
                {"table": table, "name": name, "key": k} for k in missing
            )

    implemented = set(module._aten_implemented())
    implemented |= set(module._aten_implemented_awaiting_golden())
    spelled = {k for keys in decl.values() for k in keys}
    spelled |= composite_keys(repo_root)
    unspelled = sorted(implemented - spelled)   # shape 2

    corpus = test_corpus(repo_root)
    names = sorted({name for _, name in decl})
    unexercised = sorted(n for n in names if not spelling_exercised(corpus, n))  # shape 3

    return {
        "declared_names": len(names),
        "declared_keys": len(arm),
        "implemented_keys": len(implemented),
        "shape1_dead_names": dead_names,
        "shape1_dead_keys": dead_keys,
        "shape2_unspelled_kernels": unspelled,
        "shape3_unexercised_names": unexercised,
    }


def load_allow(path=ALLOW_PATH) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def check(module, repo_root=REPO_ROOT, allow_path=ALLOW_PATH) -> list:
    """Answer a list of failure strings. Empty means every connection holds.

    The allowlist is matched **exactly**, not as an upper bound, in both
    directions: an unallowed gap fails, and an allowlisted name that is no
    longer a gap fails too. The second half is what keeps it from decaying into
    a blanket pass — closing a gap forces the entry (and its stated reason) out
    of the file in the same change, so the file always says what is missing
    *now* rather than what was missing once.
    """
    report = audit(module, repo_root)
    allow = load_allow(allow_path)
    fails = []

    for entry in report["shape1_dead_names"]:
        key = "%s:%s" % (entry["table"], entry["name"])
        if key not in allow["shape1_no_dispatch_arm"]:
            fails.append(
                "shape 1: %s is declared in %s.json but none of its keys %s has "
                "a dispatch arm -- calling it raises 'aten op not implemented'"
                % (entry["name"], entry["table"], entry["keys"])
            )
    for key in allow["shape1_no_dispatch_arm"]:
        live = {"%s:%s" % (e["table"], e["name"]) for e in report["shape1_dead_names"]}
        if key not in live:
            fails.append(
                "shape 1: %s is allowlisted as having no dispatch arm, but it now "
                "has one. Remove the entry." % key
            )

    ceiling = allow["shape1_dead_overload_keys_ceiling"]
    if len(report["shape1_dead_keys"]) > ceiling:
        fails.append(
            "shape 1 (partial): %d declared overload keys have no dispatch arm, "
            "above the recorded ceiling of %d. A name still resolves through its "
            "other overloads, so this is a ratchet rather than a hard zero -- but "
            "it may not grow. See docs/REACH.md §2."
            % (len(report["shape1_dead_keys"]), ceiling)
        )

    allowed2 = allow["shape2_kernel_without_spelling"]
    for key in report["shape2_unspelled_kernels"]:
        if key not in allowed2:
            fails.append(
                "shape 2: %s has a kernel and is compared by golden, but no "
                "`torch.<name>` / `Tensor.<name>` spelling and no composite in "
                "bootstrap.py reaches it -- it is invisible from Python." % key
            )
    for key in allowed2:
        if key not in report["shape2_unspelled_kernels"]:
            fails.append(
                "shape 2: %s is allowlisted as unspelled, but a spelling now "
                "reaches it (or its kernel is gone). Remove the entry." % key
            )

    allowed3 = allow["shape3_unexercised_spelling"]
    for name in report["shape3_unexercised_names"]:
        if name not in allowed3:
            fails.append(
                "shape 3: nothing in pytests/ calls `torch.%s(...)` or "
                "`.%s(...)`. The kernel may be golden-compared, but the spelling "
                "that reaches it is not." % (name, name)
            )
    for name in allowed3:
        if name not in report["shape3_unexercised_names"]:
            fails.append(
                "shape 3: %r is allowlisted as unexercised, but a test now spells "
                "it. Remove the entry." % name
            )

    return fails


# --- the reasons, checked ---------------------------------------------------


def upstream_claims(allow=None) -> dict:
    """`{attribute path: expected bool}` gathered from the allowlist's reasons.

    An allowlist whose reasons are prose is an allowlist nobody can audit. Every
    entry that says "upstream has no such spelling" carries the attribute path
    it is claiming that about, and this collects them so they can be put to
    upstream directly.
    """
    allow = allow or load_allow()
    claims = {}
    for section in ("shape1_no_dispatch_arm", "shape2_kernel_without_spelling",
                    "shape3_unexercised_spelling"):
        for entry in allow[section].values():
            for path in entry.get("upstream_absent", ()):
                claims[path] = False
            for path in entry.get("upstream_present", ()):
                claims[path] = True
    return claims


_UPSTREAM_PROBE = r"""
import json, sys, torch
if hasattr(torch._C, "_aten_implemented"):
    print(json.dumps({"error": "not upstream torch -- the shim is on the path"}))
    sys.exit(0)
out = {}
for path in json.loads(sys.argv[1]):
    obj = torch
    for part in path.split(".")[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            break
    out[path] = obj is not None
print(json.dumps(out))
"""


def verify_upstream(allow=None, python=None):
    """Put the allowlist's `hasattr` claims to a real upstream torch.

    In a subprocess with `PYTHONPATH` stripped, because this process has the
    shim's `_C` loaded and importing upstream torch beside it is not a thing to
    do inside a test. Answers `(ok, detail)`; `ok is None` means upstream torch
    was not importable here, which is reported rather than passed silently.
    """
    claims = upstream_claims(allow)
    if not claims:
        return True, "no upstream claims in the allowlist"
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "TORCH_USE_RTLD_GLOBAL")}
    proc = subprocess.run(
        [python or sys.executable, "-c", _UPSTREAM_PROBE, json.dumps(sorted(claims))],
        capture_output=True, text=True, env=env,
    )
    if proc.returncode != 0:
        return None, "upstream torch not importable: %s" % proc.stderr.strip()[-200:]
    try:
        actual = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None, "unreadable probe output: %r" % proc.stdout[-200:]
    if "error" in actual:
        return None, actual["error"]
    wrong = {p: (claims[p], actual.get(p)) for p in claims if actual.get(p) != claims[p]}
    if wrong:
        return False, "; ".join(
            "%s: allowlist says %s, upstream says %s" % (p, w, g)
            for p, (w, g) in sorted(wrong.items())
        )
    return True, "%d upstream claims hold" % len(claims)


def self_test(module) -> list:
    """The probe's own discrimination, checked rather than assumed."""
    fails = []
    if has_dispatch_arm(module, "aten.definitely_not_an_op.default"):
        fails.append("probe reports an arm for an op that cannot have one")
    if not has_dispatch_arm(module, "aten.add.Tensor"):
        fails.append("probe reports no arm for aten.add.Tensor")
    src = (
        "x = ['aten.in_list.default',\n     'aten.also_in_list.default']\n"
        "def f():\n"
        "    '''aten.in_prose.default'''\n"
        "    # aten.in_comment.default\n"
        "    return 1\n"
    )
    blanked = executable_text(
        "def f():\n    '''calls torch.demo(x)'''\n    # torch.demo(x) again\n    return 1\n"
    )
    if "torch.demo(" in blanked:
        fails.append("comments/docstrings still count as a call: %r" % blanked)
    seen = {s for s in code_string_constants(src) if _KEY_LITERAL.fullmatch(s)}
    if seen != {"aten.in_list.default", "aten.also_in_list.default"}:
        fails.append("docstring stripping is wrong: %r" % sorted(seen))
    return fails


def main(argv) -> int:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import loader

    module = loader.load_shim(os.environ.get("TORCH_C_ARTEFACT"))
    report = audit(module)
    print("declared names: %d, declared keys: %d, implemented keys: %d"
          % (report["declared_names"], report["declared_keys"], report["implemented_keys"]))
    print("shape 1 (declared, no dispatch arm at all): %d"
          % len(report["shape1_dead_names"]))
    for e in report["shape1_dead_names"]:
        print("    %s.%s %s" % (e["table"], e["name"], e["keys"]))
    print("shape 1 partial (one overload of a live name is dead): %d keys"
          % len(report["shape1_dead_keys"]))
    print("shape 2 (kernel, no spelling): %d" % len(report["shape2_unspelled_kernels"]))
    for k in report["shape2_unspelled_kernels"]:
        print("    %s" % k)
    print("shape 3 (spelling never exercised by a test): %d"
          % len(report["shape3_unexercised_names"]))
    print("    %s" % ", ".join(report["shape3_unexercised_names"]))
    if "--json" in argv:
        print(json.dumps(report, indent=2))

    problems = self_test(module) + check(module)
    if "--verify-upstream" in argv:
        ok, detail = verify_upstream()
        print("upstream reasons: %s -- %s"
              % ({True: "PASS", False: "FAIL", None: "SKIP"}[ok], detail))
        if ok is False:
            problems.append("upstream reasons: %s" % detail)
    for p in problems:
        print("FAIL: %s" % p)
    print("REACH: %s" % ("PASS" if not problems else "FAIL (%d)" % len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
