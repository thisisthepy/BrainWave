#!/usr/bin/env python3
"""Static early-warning scan for torch usage that is known to break on-device (iOS) targets.

This is NOT an op-set census. Python torch APIs don't map 1:1 to aten ops
(``nn.Linear`` alone becomes ``addmm``), so a static scan cannot size the
shim's implementation work. What it *can* do, because it needs no execution,
is catch a short list of patterns that have actually broken this project
before -- including in code that has never been run (stubs, unfinished
architectures). See docs/design/DESIGN.md §6 ("정적 스캔은 계측이 아니라 조기
경보로 쓴다") and §9 (the incidents this tool encodes) for the rationale.

Patterns detected, with severity:

  CRITICAL  torch.compile / torch.jit.*      -- no runtime codegen on iOS.
  CRITICAL  tensor-value python branching    -- .item()/.any()/.all() feeding
                                                 an `if`; host sync + blocks
                                                 export/tracing permanently.
  WARN      torch.cuda.* / torch.backends.cudnn.*  -- needs an availability
                                                 guard; these repos call it
                                                 unconditionally.
  INFO      torch._* private API             -- shim must provide it.
  INFO      torch.einsum                     -- expensive to shim (general
                                                 string-parsed contraction).

Every finding also reports whether it sits at *import time* (module scope,
or class-body scope -- i.e. anywhere not nested inside a function/lambda).
Import-time hits are the sharpest ones: merely importing the file executes
them, no call required.

Usage:
    python3 scan_torch_risks.py <dir> [<dir> ...] [--json] [--fail-on LEVEL]

Exit code is non-zero when findings at or above --fail-on (default: info,
i.e. any finding at all) are present, so this is CI-gateable directly.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import re
import sys
from pathlib import Path

SEVERITY_ORDER = {"info": 0, "warn": 1, "critical": 2}

DEFAULT_SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules",
    "build", "dist", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "site-packages",
}

TENSOR_VALUE_METHODS = {"item", "any", "all"}


@dataclasses.dataclass
class Finding:
    path: str
    line: int
    col: int
    severity: str  # "critical" | "warn" | "info"
    kind: str
    message: str
    import_time: bool

    def format(self) -> str:
        scope = "import-time" if self.import_time else "runtime-guarded-by-call"
        return (
            f"{self.path}:{self.line}:{self.col}: [{self.severity.upper()}] "
            f"{self.kind} ({scope}) - {self.message}"
        )


# --------------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------------

def build_parent_map(tree: ast.AST) -> dict:
    parent_map = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent_map[child] = node
    return parent_map


def resolve_dotted(node: ast.AST, symtab: dict) -> str | None:
    """Resolve a Name/Attribute chain to a dotted string, honoring import aliases."""
    if isinstance(node, ast.Name):
        return symtab.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = resolve_dotted(node.value, symtab)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def build_import_symtab(tree: ast.Module) -> dict:
    """Map local names to fully-qualified dotted paths, from every Import/ImportFrom
    in the file (not just top-level, in case of guarded/deferred imports)."""
    symtab: dict[str, str] = {"torch": "torch"}  # safe default even if not explicitly seen
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    # `import a.b.c as x` binds x -> a.b.c
                    symtab[alias.asname] = alias.name
                else:
                    # `import a.b.c` binds name "a" -> "a"; a.b.c.X is then
                    # reached via ordinary attribute-chain resolution off "a".
                    local = alias.name.split(".")[0]
                    symtab[local] = local
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                symtab[local] = f"{node.module}.{alias.name}"
    return symtab


def is_tensor_value_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in TENSOR_VALUE_METHODS
    )


def contains_tensor_value_call(node: ast.AST) -> bool:
    return any(is_tensor_value_call(n) for n in ast.walk(node))


def enclosing_function(node: ast.AST, parent_map: dict):
    p = parent_map.get(node)
    while p is not None:
        if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return p
        p = parent_map.get(p)
    return None  # nothing but Module/ClassDef above -> import-time scope


def scope_key(node: ast.AST, parent_map: dict):
    fn = enclosing_function(node, parent_map)
    return id(fn) if fn is not None else "<module>"


AVAILABILITY_TEST_RE = re.compile(r"is_available|hasattr|available|cuda", re.IGNORECASE)


def _test_src(test: ast.expr) -> str:
    try:
        return ast.unparse(test)
    except Exception:
        return ""


def _looks_like_availability_guard(test: ast.expr) -> bool:
    """True if the *literal* source of an `if`/ternary test reads like an
    availability check: `torch.cuda.is_available()`, `hasattr(torch, "compile")`,
    a helper named `..._available(...)`, or anything mentioning `cuda`."""
    return bool(AVAILABILITY_TEST_RE.search(_test_src(test)))


def collect_guard_taint(tree: ast.Module, parent_map: dict) -> list[tuple]:
    """Track variables assigned from an availability-guard-like call, e.g.
    `use_cudnn = _cudnn_available()`, so that a later `if use_cudnn:` is recognized
    as a guard even though the check happened through a helper function rather than
    a literal `is_available()`/`hasattr()` call sitting directly in the `if` test.

    This mirrors the tensor-value taint tracking in `scan_tensor_value_branching`
    (Pass A there tags variables assigned from a *risky* call so later `if`s that
    branch on them are flagged; here we tag variables assigned from a *guard* call
    so later `if`s that branch on them are recognized as safe) -- same mechanism,
    opposite direction, per KNOWN_FALSE_POSITIVES.md #2.
    """
    tainted: list[tuple] = []  # (scope, name, lineno)
    for node in ast.walk(tree):
        targets = None
        value = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        elif isinstance(node, ast.AugAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if not any(
            isinstance(n, ast.Call) and _looks_like_availability_guard(n.func)
            for n in ast.walk(value)
        ):
            continue
        sk = scope_key(node, parent_map)
        for t in targets:
            if isinstance(t, (ast.Name, ast.Attribute)):
                try:
                    name = ast.unparse(t)
                except Exception:
                    continue
                tainted.append((sk, name, node.lineno))
    return tainted


def _test_tainted(test: ast.expr, sk, guard_taint: list) -> bool:
    test_src = _test_src(test)
    test_lineno = getattr(test, "lineno", None)
    for (tsk, name, taint_line) in guard_taint:
        if tsk != sk:
            continue
        if test_lineno is not None and taint_line >= test_lineno:
            continue
        if name_appears(test_src, name):
            return True
    return False


def is_guarded_by_availability_check(node: ast.AST, parent_map: dict, guard_taint: list) -> bool:
    """True if `node` sits in the guarded branch of an enclosing `if <guard>:`
    (statement) or `<x> if <guard> else <y>` (ternary), within the same function
    scope (doesn't cross function/lambda boundaries). A "guard" is either a literal
    availability check in the test's source text, or a bare/boolean-combined
    reference to a variable that was itself assigned from such a check -- see
    `collect_guard_taint`."""
    sk = scope_key(node, parent_map)
    child = node
    while True:
        parent = parent_map.get(child)
        if parent is None:
            return False
        if isinstance(parent, ast.If):
            if any(child is s or child in ast.walk(s) for s in parent.body):
                if _looks_like_availability_guard(parent.test) or _test_tainted(parent.test, sk, guard_taint):
                    return True
        if isinstance(parent, ast.IfExp) and child is parent.body:
            if _looks_like_availability_guard(parent.test) or _test_tainted(parent.test, sk, guard_taint):
                return True
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Module)):
            return False
        child = parent


NAME_BOUNDARY = re.compile(r"[A-Za-z0-9_.]")


def name_appears(haystack: str, needle: str) -> bool:
    """Substring match for `needle` (a simple or dotted identifier) in `haystack`,
    with boundary checks so 'use_bias' doesn't match inside 'use_bias2' or
    'other.use_bias'."""
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return False
        before_ok = idx == 0 or not NAME_BOUNDARY.match(haystack[idx - 1])
        end = idx + len(needle)
        after_ok = end >= len(haystack) or not NAME_BOUNDARY.match(haystack[end])
        if before_ok and after_ok:
            return True
        start = idx + 1


# --------------------------------------------------------------------------
# Scanner
# --------------------------------------------------------------------------

class Scanner:
    def __init__(self, path: str, source: str, tree: ast.Module):
        self.path = path
        self.source = source
        self.tree = tree
        self.symtab = build_import_symtab(tree)
        self.parent_map = build_parent_map(tree)
        self.guard_taint = collect_guard_taint(tree, self.parent_map)
        self.findings: list[Finding] = []

    # -- category: torch.compile / torch.jit.* , torch.cuda.* / torch.backends.cudnn.* ,
    #    torch._* private API, torch.einsum -- all driven off dotted-attribute resolution.
    def scan_attribute_namespaces(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Attribute):
                continue
            dotted = resolve_dotted(node, self.symtab)
            if dotted is None:
                continue

            matched = self._classify(dotted)
            if matched is None:
                continue
            kind, severity, guard_exempt = matched
            if guard_exempt:
                continue

            import_time = enclosing_function(node, self.parent_map) is None

            if kind == "CUDA_UNGUARDED" and is_guarded_by_availability_check(node, self.parent_map, self.guard_taint):
                continue
            if kind == "TORCH_COMPILE" and not import_time and is_guarded_by_availability_check(
                node, self.parent_map, self.guard_taint
            ):
                # Deferred + availability-guarded: the risky call moved out of
                # import time and won't run where torch.compile is unavailable.
                # See KNOWN_FALSE_POSITIVES.md #1 -- guarded/lazy torch.compile
                # is not the same hazard as a module-scope one.
                continue

            # Only report the maximal (outermost) chain: if our ast-tree parent is
            # itself an Attribute that resolves to a dotted path in the *same*
            # category, let the parent report instead (it's a longer, more
            # specific chain covering this one).
            parent = self.parent_map.get(node)
            if isinstance(parent, ast.Attribute):
                parent_dotted = resolve_dotted(parent, self.symtab)
                if parent_dotted is not None and self._classify(parent_dotted) is not None:
                    pk, ps, pg = self._classify(parent_dotted)
                    if pk == kind and not pg:
                        continue

            invoked = isinstance(parent, ast.Call) and parent.func is node

            usage = "called" if invoked else "referenced"
            message = self._message_for(kind, dotted, usage)
            self.findings.append(Finding(
                path=self.path, line=node.lineno, col=node.col_offset,
                severity=severity, kind=kind, message=message,
                import_time=import_time,
            ))

    def _classify(self, dotted: str):
        """Return (kind, severity, guard_exempt) or None."""
        if dotted == "torch.compile" or dotted.startswith("torch.compile."):
            return ("TORCH_COMPILE", "critical", False)
        if dotted == "torch.jit" or dotted.startswith("torch.jit."):
            if dotted == "torch.jit":
                return None  # bare module reference, not a risky symbol by itself
            return ("TORCH_JIT", "critical", False)
        if dotted == "torch.cuda" or dotted.startswith("torch.cuda."):
            if dotted == "torch.cuda":
                return None
            leaf = dotted.rsplit(".", 1)[-1]
            if leaf in ("is_available", "device_count"):
                return ("CUDA_GUARDED_CHECK", "info", True)
            return ("CUDA_UNGUARDED", "warn", False)
        if dotted == "torch.backends.cudnn" or dotted.startswith("torch.backends.cudnn."):
            if dotted == "torch.backends.cudnn":
                return None
            leaf = dotted.rsplit(".", 1)[-1]
            if leaf == "is_available":
                return ("CUDA_GUARDED_CHECK", "info", True)
            return ("CUDA_UNGUARDED", "warn", False)
        if dotted == "torch.einsum":
            return ("EINSUM", "info", False)
        # torch._<private>[...]: second path segment starts with underscore.
        parts = dotted.split(".")
        if len(parts) >= 2 and parts[0] == "torch" and parts[1].startswith("_") and parts[1] != "_":
            return ("PRIVATE_API", "info", False)
        return None

    @staticmethod
    def _message_for(kind: str, dotted: str, usage: str) -> str:
        if kind == "TORCH_COMPILE":
            return f"{dotted} {usage} -- no runtime code generation on iOS (torch.compile/Inductor)"
        if kind == "TORCH_JIT":
            return f"{dotted} {usage} -- torch.jit is unavailable/unsupported on iOS targets"
        if kind == "CUDA_UNGUARDED":
            return f"{dotted} {usage} without a guard -- CUDA is not present on-device"
        if kind == "PRIVATE_API":
            return f"{dotted} {usage} -- private torch API; the shim must provide it explicitly"
        if kind == "EINSUM":
            return f"{dotted} {usage} -- general string-parsed contraction is expensive to shim"
        return f"{dotted} {usage}"

    # -- category: tensor-value python branching (.item()/.any()/.all() feeding `if`)
    def scan_tensor_value_branching(self):
        # Pass A: record assignments whose RHS contains a tensor-value call.
        tainted: list[tuple] = []  # (scope, name, lineno, value_src)
        for node in ast.walk(self.tree):
            targets = None
            value = None
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            elif isinstance(node, ast.AugAssign):
                targets, value = [node.target], node.value
            else:
                continue
            if not contains_tensor_value_call(value):
                continue
            sk = scope_key(node, self.parent_map)
            try:
                value_src = ast.unparse(value)
            except Exception:
                value_src = "<unprintable>"
            for t in targets:
                if isinstance(t, (ast.Name, ast.Attribute)):
                    try:
                        name = ast.unparse(t)
                    except Exception:
                        continue
                    tainted.append((sk, name, node.lineno, value_src))

        # Pass B: for every `if`, check direct or tainted-variable use in the test.
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.If):
                continue
            direct = contains_tensor_value_call(node.test)
            if direct:
                try:
                    test_src = ast.unparse(node.test)
                except Exception:
                    test_src = "<unprintable>"
                import_time = enclosing_function(node, self.parent_map) is None
                self.findings.append(Finding(
                    path=self.path, line=node.lineno, col=node.col_offset,
                    severity="critical", kind="TENSOR_VALUE_BRANCH",
                    message=(
                        f"if condition directly evaluates a tensor value ({test_src}) "
                        f"-- host sync per call, blocks tracing/export permanently"
                    ),
                    import_time=import_time,
                ))
                continue

            sk = scope_key(node, self.parent_map)
            try:
                test_src = ast.unparse(node.test)
            except Exception:
                test_src = None
            if test_src is None:
                continue
            for (tsk, name, taint_line, value_src) in tainted:
                if tsk != sk or taint_line >= node.lineno:
                    continue
                if name_appears(test_src, name):
                    import_time = enclosing_function(node, self.parent_map) is None
                    self.findings.append(Finding(
                        path=self.path, line=node.lineno, col=node.col_offset,
                        severity="critical", kind="TENSOR_VALUE_BRANCH",
                        message=(
                            f"if branches on `{name}` (assigned at line {taint_line} "
                            f"from a tensor-value call: `{value_src}`) -- host sync, "
                            f"blocks tracing/export permanently"
                        ),
                        import_time=import_time,
                    ))
                    break

    def run(self) -> list[Finding]:
        self.scan_attribute_namespaces()
        self.scan_tensor_value_branching()
        return self.findings


def scan_file(path: Path) -> list[Finding]:
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        print(f"warning: skipping {path}: {exc}", file=sys.stderr)
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        print(f"warning: skipping {path}: SyntaxError: {exc}", file=sys.stderr)
        return []
    return Scanner(str(path), source, tree).run()


def iter_python_files(root: Path, skip_dirs: set):
    if root.is_file():
        if root.suffix == ".py":
            yield root
        return
    for p in sorted(root.rglob("*.py")):
        if any(part in skip_dirs for part in p.parts):
            continue
        yield p


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Directories or files to scan")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    parser.add_argument(
        "--fail-on", choices=["info", "warn", "critical", "none"], default="info",
        help="Minimum severity that causes a non-zero exit (default: info -- any finding fails)",
    )
    parser.add_argument(
        "--skip-dir", action="append", default=[],
        help="Additional directory name to exclude (repeatable)",
    )
    args = parser.parse_args(argv)

    skip_dirs = set(DEFAULT_SKIP_DIRS) | set(args.skip_dir)

    all_findings: list[Finding] = []
    for raw_path in args.paths:
        root = Path(raw_path)
        if not root.exists():
            print(f"error: path does not exist: {root}", file=sys.stderr)
            return 2
        for py_file in iter_python_files(root, skip_dirs):
            all_findings.extend(scan_file(py_file))

    all_findings.sort(key=lambda f: (f.path, f.line, f.col))

    counts = {"critical": 0, "warn": 0, "info": 0}
    for f in all_findings:
        counts[f.severity] += 1

    if args.json:
        print(json.dumps({
            "findings": [dataclasses.asdict(f) for f in all_findings],
            "counts": counts,
        }, indent=2))
    else:
        for f in all_findings:
            print(f.format())
        print(
            f"\n{len(all_findings)} finding(s): "
            f"{counts['critical']} critical, {counts['warn']} warn, {counts['info']} info"
        )

    if args.fail_on == "none":
        return 0
    threshold = SEVERITY_ORDER[args.fail_on]
    if any(SEVERITY_ORDER[f.severity] >= threshold for f in all_findings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
