#!/usr/bin/env python3
"""Extract the `torch._C` name surface from the *vendored tree's own* stubs.

Why this exists
---------------
VENDOR.md measured the hole: upstream `torch._C` is 989 names, 32 C
submodules, a 694-member `TensorBase`, and a 985-member
`_VariableFunctions`. `import torch` demands most of that surface before it
finishes, so `rust/torch_c` has to present it.

Where the names come from matters. `vendor/probe.py --dump-surface` reads
them off an *installed* upstream `_C.so`, which is right for an instrument
(it is measuring the hole) and wrong for the shim (it would be borrowing
from the binary we are replacing, and it would tie the build to having real
torch installed).

This script reads `torchnative/src/main/torch/_C/*.pyi` instead. Those files are part of
the vendored BSD Python tree -- upstream ships them precisely so that other
tools can know `_C`'s interface without loading it. They are also the
tree's *own* statement of what it expects, which is the thing the shim has
to satisfy.

They are parsed with `ast`, not with regexes: `.pyi` files are valid Python
and the two distinctions that matter -- `def name(...)` (a method) versus
`name: T` (a getset descriptor, which `torch/_prims_common/__init__.py:90`
reaches for as a *descriptor object*, VENDOR.md wall 10) -- are exactly the
distinction between `FunctionDef` and `AnnAssign`.

Output is a single JSON blob compiled into the crate (`include_str!`), so
the built `_C.so` needs neither the stubs nor an interpreter with torch in
it at runtime.

    vendor/gen_surface.py                 # writes rust/torch_c/src/surface.json
    vendor/gen_surface.py --print-summary
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The enum-ish types whose module-level instances upstream injects into the
# `torch` namespace from C during `_initExtension` (VENDOR.md wall 7). The
# stubs declare them as annotated module-level names, which is how they are
# found here.
NAMESPACE_TYPES = ("dtype", "layout", "memory_format", "qscheme")

# Module-level names the stubs annotate `_bool`. They are carried through as
# their own kind rather than flattened into "value" because a placeholder is
# *not* an acceptable answer for one, and the shim can only enforce that if it
# can tell them apart.
#
# The reason is measured, not stylistic. `_Unimplemented` has no `__bool__`, so
# it is truthy, and `torch/backends/cudnn/__init__.py:231` -- a class body, so
# it runs during `import torch` -- reads `_has_cudnn` through `is_available()`
# and took the "cuDNN is present" branch. Every one of these names is read that
# way somewhere; a build flag has two answers and both change behaviour.
# `bootstrap.install` refuses to build a surface that leaves one unanswered.
BOOL_VALUE_TYPE = "_bool"


def _is_ellipsis(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is Ellipsis


def _class_members(node: ast.ClassDef) -> tuple[list[str], list[str]]:
    """(methods, attributes) of a stubbed class.

    Attributes become `property` objects in the shim rather than plain class
    values, because the tree pulls the descriptor itself out of the class
    (`torch.Tensor.is_sparse.__get__`).
    """
    methods: list[str] = []
    attrs: list[str] = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name not in methods:
                methods.append(item.name)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if item.target.id not in attrs:
                attrs.append(item.target.id)
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id not in attrs:
                    attrs.append(target.id)
    return methods, attrs


def _annotation_name(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def parse_stub(path: str) -> dict:
    """One `.pyi` -> {functions, types, values, namespace}."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)

    functions: list[str] = []
    types: dict[str, dict] = {}
    values: dict[str, str] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in functions:
                functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            methods, attrs = _class_members(node)
            metaclass = None
            for kw in node.keywords:
                if kw.arg == "metaclass":
                    metaclass = _annotation_name(kw.value)
            bases = [b for b in (_annotation_name(b) for b in node.bases) if b]
            types[node.name] = {
                "metaclass": metaclass,
                "bases": bases,
                "methods": methods,
                "attrs": attrs,
            }
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            values[name] = _annotation_name(node.annotation) or "Any"
        elif isinstance(node, ast.Assign):
            # `X: TypeAlias = ...` and plain `X = ...`; the latter is how the
            # stub spells a handful of module-level constants.
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_ellipsis(node.value):
                    values.setdefault(target.id, "Any")

    return {"functions": functions, "types": types, "values": values}


def submodule_stubs(c_dir: str) -> dict[str, str]:
    """`torch._C.<name>` -> stub path, for both file and package stubs."""
    out: dict[str, str] = {}
    for entry in sorted(os.listdir(c_dir)):
        full = os.path.join(c_dir, entry)
        if entry.endswith(".pyi") and entry != "__init__.pyi":
            out[entry[: -len(".pyi")]] = full
        elif os.path.isdir(entry_path := full) and os.path.isfile(
            init := os.path.join(entry_path, "__init__.pyi")
        ):
            out[entry] = init
    return out


def _submodule_entry(parsed: dict, nested: dict[str, dict] | None = None) -> dict:
    entry = {
        "functions": parsed["functions"],
        "types": {
            k: {"methods": v["methods"], "attrs": v["attrs"], "bases": v["bases"]}
            for k, v in parsed["types"].items()
        },
        "values": sorted(parsed["values"]),
    }
    # Only stamped on when there is something to nest, so directory stubs
    # with no sibling `.pyi` files (`_acc/`, today) stay byte-identical to
    # the pre-recursion output -- the point is to add names, not to touch
    # entries that were already complete.
    if nested:
        entry["submodules"] = nested
    return entry


def parse_package_stub(dir_path: str) -> dict:
    """Recursively parse a directory-shaped `.pyi` package.

    `submodule_stubs()` registers a directory stub (`_dynamo/`, `_export/`)
    by its `__init__.pyi` alone, which is what made `_dynamo.guards` and
    `_dynamo.eval_frame` invisible to the surface even though the vendored
    tree declares them in full (docs/graph/DYNAMO.md §5: 8 of 137 names known,
    6%). This walks the rest of the directory -- sibling `.pyi` files and,
    in case a package ever nests a package, sibling subdirectories with
    their own `__init__.pyi` -- and files each one under `submodules`,
    keyed by its own name. `_dynamo.guards.<name>` is then reachable as
    `surface["submodules"]["_dynamo"]["submodules"]["guards"]`.
    """
    parsed = parse_stub(os.path.join(dir_path, "__init__.pyi"))
    nested: dict[str, dict] = {}
    for entry in sorted(os.listdir(dir_path)):
        if entry == "__init__.pyi":
            continue
        full = os.path.join(dir_path, entry)
        if entry.endswith(".pyi"):
            nested[entry[: -len(".pyi")]] = _submodule_entry(parse_stub(full))
        elif os.path.isdir(full) and os.path.isfile(os.path.join(full, "__init__.pyi")):
            nested[entry] = parse_package_stub(full)
    return _submodule_entry(parsed, nested)


# `torch._C.<name>` / `_C.<name>` attribute reads, and `from torch._C import
# <name>`. Both spellings occur; `torch/nn/functional.py:12` uses the second.
_ATTR_RE = re.compile(r"(?:torch\._C|(?<![\w.])_C)\.([A-Za-z_]\w*)")
_FROM_RE = re.compile(r"from\s+torch\._C\s+import\s+\(([^)]*)\)|from\s+torch\._C\s+import\s+([^\n(]+)")
# A name asked about rather than used. VENDOR.md wall 11: the tree switches
# whole subsystems off by asking whether `_C` has a name, so a name that only
# ever appears inside such a question is a *switch*, and the shim must not
# define it. Deriving this from the tree beats hand-listing it -- the hand list
# was already missing entries when this replaced it.
_PROBE_RE = re.compile(
    r"(?:hasattr|getattr)\(\s*(?:torch\._C|_C)\s*,\s*[\"']([A-Za-z_]\w*)[\"']"
)


# `TensorBase` members the stub does not list. `torch/_tensor_docs.py` is 547
# `add_docstr_all("<method>", ...)` calls and every one of them does
# `getattr(torch._C.TensorBase, method)` at import (`_tensor_docs.py:10`), so a
# missing member is an `AttributeError` during `import torch`, not a lazy
# failure. The stub has 627 methods; upstream's `TensorBase` has 694.
_DOCSTR_ALL_RE = re.compile(r"add_docstr_all\(\s*[\"']([A-Za-z_]\w*)[\"']")
_TENSORBASE_RE = re.compile(r"TensorBase\.([A-Za-z_]\w*)")


def scan_tensorbase(torch_dir: str) -> set:
    found: set = set()
    for name in ("_tensor_docs.py", "_torch_docs.py", "_tensor.py", "_C/_TensorBase.pyi"):
        path = os.path.join(torch_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        found.update(_DOCSTR_ALL_RE.findall(text))
        found.update(_TENSORBASE_RE.findall(text))
    return found


def scan_tree(torch_dir: str) -> tuple[set, set]:
    """(names the tree uses, names the tree only probes for).

    Reads every vendored `.py` as text. Not `ast`: `torch._C.foo` inside a
    string, a comment or a docstring is harmless to over-collect (the shim
    gains a name that raises if used), whereas *under*-collecting stops the
    import dead, which is what happened before this function existed -- the
    stubs do not declare `_ScalingType`, but `torch/nn/functional.py` imports
    it.
    """
    used: set = set()
    probed: set = set()
    for root, dirs, files in os.walk(torch_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            try:
                with open(os.path.join(root, fname), encoding="utf-8") as fh:
                    text = fh.read()
            except (OSError, UnicodeDecodeError):
                continue
            if "_C" not in text:
                continue
            used.update(_ATTR_RE.findall(text))
            probed.update(_PROBE_RE.findall(text))
            for group in _FROM_RE.findall(text):
                for chunk in group:
                    # Strip trailing comments *per line* before splitting on
                    # commas. Without this a `# pyrefly: ignore [...]` after
                    # one name swallows the next one, which is exactly how
                    # `_SwizzleType` went missing while `_ScalingType`, one
                    # line above it, came through.
                    lines = [line.split("#", 1)[0] for line in chunk.splitlines()]
                    for item in " ".join(lines).split(","):
                        item = item.strip().split(" as ")[0].strip()
                        if item and item.replace("_", "a").isalnum():
                            used.add(item)
    return used, probed


def build(vendor_dir: str) -> dict:
    c_dir = os.path.join(vendor_dir, "torch", "_C")
    if not os.path.isdir(c_dir):
        raise SystemExit(
            f"no vendored stubs at {c_dir} -- run vendor/vendor_torch.sh first"
        )

    main = parse_stub(os.path.join(c_dir, "__init__.pyi"))

    # `_VariableFunctions` is a stub *module* upstream, but the tree reaches it
    # as an object attribute of `_C` (`torch/__init__.py:2212`), so its
    # functions are lifted out here rather than registered as a submodule.
    varfns_path = os.path.join(c_dir, "_VariableFunctions.pyi")
    varfns = parse_stub(varfns_path)["functions"] if os.path.isfile(varfns_path) else []

    submodules: dict[str, dict] = {}
    for name, path in submodule_stubs(c_dir).items():
        if name == "_VariableFunctions":
            continue
        # `bases` is carried, not dropped: `torch._C._functorch`'s
        # `TransformType` is an `Enum`, and `torch/_ops.py:139` asserts on
        # `isinstance`, so losing the base turns its members into
        # properties and the assertion fires with `got <class 'property'>`.
        # (See `_submodule_entry` for where `bases` actually gets kept.)
        entry_dir = os.path.join(c_dir, name)
        if os.path.isdir(entry_dir):
            submodules[name] = parse_package_stub(entry_dir)
        else:
            submodules[name] = _submodule_entry(parse_stub(path))

    tensorbase = main["types"].pop("TensorBase", {"methods": [], "attrs": []})
    known = set(tensorbase["methods"]) | set(tensorbase["attrs"])
    tensorbase["methods"] = tensorbase["methods"] + sorted(
        scan_tensorbase(os.path.join(vendor_dir, "torch")) - known
    )

    namespace = {
        name: kind
        for name, kind in main["values"].items()
        if kind in NAMESPACE_TYPES
    }

    used, probed = scan_tree(os.path.join(vendor_dir, "torch"))

    module: dict[str, str] = {}
    for name in main["functions"]:
        module[name] = "function"
    for name in main["types"]:
        module[name] = "type"
    for name, kind in main["values"].items():
        module.setdefault(name, "bool" if kind == BOOL_VALUE_TYPE else "value")
    for name in submodules:
        module[name] = "module"
    # Names the tree reaches for that the stubs never declare. They land as
    # plain placeholders -- present, and loud when used.
    for name in sorted(used):
        module.setdefault(name, "value")

    return {
        "probes": sorted(probed),
        "module": module,
        "types": main["types"],
        "tensorbase": {"methods": tensorbase["methods"], "attrs": tensorbase["attrs"]},
        "varfns": varfns,
        "submodules": submodules,
        "namespace": namespace,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    # The tree lands in the package rather than beside it, so that one build
    # sees both `torch` and `torchnative` as top-level packages. The scripts
    # that assemble it stay here in vendor/; only the destination moved. Same
    # override the shell scripts take.
    ap.add_argument(
        "--vendor-dir",
        default=os.environ.get(
            "TORCHNATIVE_VENDOR_DIR",
            os.path.join(REPO, "torchnative", "src", "main"),
        ),
    )
    ap.add_argument(
        "--out", default=os.path.join(REPO, "rust", "torch_c", "src", "surface.json")
    )
    ap.add_argument("--print-summary", action="store_true")
    args = ap.parse_args()

    surface = build(args.vendor_dir)

    # Compact separators, sorted keys: the file is compiled into the artefact
    # and diffed by humans, so it should be stable across runs and not waste
    # space on indentation.
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(surface, fh, separators=(",", ":"), sort_keys=True)
        fh.write("\n")

    print(f"wrote {args.out} ({os.path.getsize(args.out)} B)")
    print(f"  module names   {len(surface['module'])}")
    print(f"  types          {len(surface['types'])}")
    print(f"  TensorBase     {len(surface['tensorbase']['methods'])} methods "
          f"+ {len(surface['tensorbase']['attrs'])} attrs")
    print(f"  _VariableFunctions {len(surface['varfns'])}")
    print(f"  submodules     {len(surface['submodules'])}")
    print(f"  torch namespace {len(surface['namespace'])}")
    print(f"  probed (off-switches) {len(surface['probes'])}")
    print(f"  build flags (_bool)   "
          f"{sum(1 for k in surface['module'].values() if k == 'bool')}")
    if args.print_summary:
        for name, sub in sorted(surface["submodules"].items()):
            print(f"    {name}: {len(sub['functions'])} fn, "
                  f"{len(sub['types'])} types, {len(sub['values'])} values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
