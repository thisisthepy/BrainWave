#!/usr/bin/env python3
"""Check `src/overloads.json` and `src/methods.json` against upstream torch,
schema string by schema string.

Why this exists
---------------
`surface.json` is generated from the vendored tree's own `.pyi` stubs, and
docs/models/IMPORT_TORCH.md §1 gives the reason: borrowing names from the upstream
`_C.so` would make the build depend on the binary we are replacing. The
overload table cannot be produced the same way. The tree carries the aten
*overload names* (`aten.arange.start_step` and 980 more appear literally in
`_decomp`/`_meta_registrations`/`_refs`) and it carries Python-level
signatures (`torch/_C/_VariableFunctions.pyi`), but nothing in it joins the
two -- a `.pyi` overload does not say which aten overload it lowers to. See
docs/bindings/OVERLOAD.md §2.

So the table is transcribed, and a transcription needs a check. This script
is that check: it re-derives every schema from an installed upstream torch
and diffs. It is *not* part of the build -- the table is compiled into the
artefact and `cargo build` needs no torch -- it is the same kind of tool as
`tools/golden/compare.py`: run it when upstream moves.

Usage
-----
    /Volumes/macMini/caches/spike-venv/bin/python rust/torch_c/pytests/verify_schemas.py

Exit code is 0 iff every entry matched. Read the exit code; do not grep.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, os.pardir, "src")
TABLES = (
    ("overloads.json", os.path.join(SRC, "overloads.json")),
    ("methods.json", os.path.join(SRC, "methods.json")),
)


def _aten_name(schema: str) -> tuple[str, str]:
    """`aten::mul.Tensor(...) -> ...` -> `("mul", "Tensor")`.

    Derived from the schema string rather than from the table *key*, which is
    the only thing that works for both tables: in `overloads.json` the key is
    the op name, but in `methods.json` it is the Python method name, and the
    two are not always the same -- `item` is `_local_scalar_dense`, `__mul__`
    is `mul`, `__invert__` is `bitwise_not`.
    """
    head = schema.split("(", 1)[0].strip()
    _, _, rest = head.rpartition("::")
    name, _, overload = rest.partition(".")
    return name, overload or "default"


def check(label: str, path: str, torch) -> tuple[int, int]:
    with open(path, encoding="utf-8") as fh:
        table = json.load(fh)

    failures = 0
    checked = 0
    for name, schemas in sorted(table.items()):
        if name.startswith("_README"):
            continue  # the embedded README
        for schema in schemas:
            checked += 1
            op, overload = _aten_name(schema)
            try:
                packet = getattr(torch.ops.aten, op)
                upstream = {
                    str(getattr(packet, ov)._schema) for ov in packet.overloads()
                }
            except Exception as exc:
                print(f"FAIL {label} {name}: no upstream op packet for aten::{op}: {exc}")
                failures += 1
                continue
            if schema not in upstream:
                failures += 1
                print(f"FAIL {label} {name} ({op}.{overload}): "
                      f"table entry is not an upstream schema:")
                print(f"     table:    {schema}")
                # Show the closest upstream sibling by overload name, which is
                # what a drift looks like in practice (an argument gained a
                # default, or `int` became `SymInt`).
                head = schema.split("(", 1)[0]
                for candidate in sorted(upstream):
                    if candidate.split("(", 1)[0] == head:
                        print(f"     upstream: {candidate}")
    return checked, failures


# The `_c10d_functional` family, which is not a JSON table but a tuple literal
# in `bootstrap.py`. Same problem, same answer: the text was transcribed from
# upstream's own registry (docs/distributed/DISTRIBUTED.md), so it needs the same check.
#
# It is read out of the source with `ast` rather than imported, because
# importing `bootstrap.py` means importing the shim, and this script has to run
# with *upstream* torch on the path -- putting both in one interpreter is the
# thing the golden harness goes to a second process to avoid.
BOOTSTRAP = os.path.join(SRC, "bootstrap.py")
NON_ATEN_NAMESPACES = ("_c10d_functional", "_c10d_functional_autograd", "_dtensor")


def _literal_table(name: str) -> list:
    with open(BOOTSTRAP, encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if name in targets:
            return list(ast.literal_eval(node.value))
    raise RuntimeError(f"{name} not found in {BOOTSTRAP}")


def _non_aten_schema_table() -> list:
    return _literal_table("_NON_ATEN_SCHEMA_TEXT")


def check_generated_aten(torch) -> tuple[int, int]:
    """`_GENERATED_ATEN_SCHEMA_TEXT`, the other half of the same problem.

    `native_functions.yaml` declares 2584 aten schemas and upstream's registry
    has 3754; the difference is what `torchgen/native_function_generation.py`
    synthesises at build time. That generator is vendored and cannot be run
    here (it parses the YAML, and `pyyaml` is not a dependency of this
    distribution), so the handful of generated schemas the tree asks questions
    about are transcribed. Transcribed means checked -- same as the `c10d`
    table above.

    Both directions again, but the reverse direction is a different claim:
    every entry has to be an operator upstream *and* has to be one
    `native_functions.yaml` does not carry. An entry the file does carry is
    dead weight that would silently shadow the file's own text.
    """
    table = _literal_table("_GENERATED_ATEN_SCHEMA_TEXT")
    declared = set()
    yaml_path = os.path.join(
        HERE, os.pardir, os.pardir, os.pardir, "torchnative", "src", "main",
        "torchgen", "packaged", "ATen", "native", "native_functions.yaml")
    if os.path.isfile(yaml_path):
        with open(yaml_path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("- func:"):
                    declared.add(line[len("- func:"):].strip().split("(", 1)[0])

    failures = 0
    for text in table:
        head = text.split("(", 1)[0].strip()
        _, _, rest = head.rpartition("::")
        op, overload = _aten_name(text)
        try:
            upstream = str(getattr(getattr(torch.ops.aten, op), overload)._schema)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL generated aten {rest}: upstream has no such operator: {exc}")
            continue
        if _normalise(upstream) != _normalise(text):
            failures += 1
            print(f"FAIL generated aten {rest}:")
            print(f"     table:    {text}")
            print(f"     upstream: {upstream}")
        if declared and rest in declared:
            failures += 1
            print(f"FAIL generated aten {rest}: native_functions.yaml declares "
                  "this, so the entry shadows the file rather than filling a gap")
    return len(table), failures


def check_non_aten(torch) -> tuple[int, int]:
    """Both directions, unlike the aten tables.

    A *missing* entry matters here in a way it does not for aten: the aten
    tables are deliberately a subset of what the dispatcher knows, but this
    table's whole job is to be the schemas the tree will ask for, and the tree
    asks for all of them. So an op upstream registers in one of these
    namespaces and this table lacks is reported.
    """
    # Importing this is what registers the namespaces; without it upstream's
    # own registry does not have them either.
    import torch.distributed._functional_collectives  # noqa: F401

    upstream = {}
    for schema in torch._C._jit_get_all_schemas():
        namespace = schema.name.split("::")[0]
        if namespace in NON_ATEN_NAMESPACES:
            upstream[str(schema).split("(")[0]] = str(schema)

    table = _non_aten_schema_table()
    failures = 0
    seen = set()
    for text in table:
        head = text.split("(")[0]
        seen.add(head)
        candidate = upstream.get(head)
        if candidate is None:
            failures += 1
            print(f"FAIL c10d schemas {head}: upstream has no such operator")
        elif _normalise(candidate) != _normalise(text):
            failures += 1
            print(f"FAIL c10d schemas {head}:")
            print(f"     table:    {text}")
            print(f"     upstream: {candidate}")
    for head in sorted(set(upstream) - seen):
        failures += 1
        print(f"FAIL c10d schemas {head}: upstream registers it and the table "
              "does not carry it")
    return len(table), failures


def _normalise(schema: str) -> str:
    return re.sub(r"\s+", " ", schema).strip()


# ---------------------------------------------------------------------------
# What the shim actually answers (docs/bindings/SCHEMA.md)
# ---------------------------------------------------------------------------
#
# The three checks above compare *tables* against upstream. A table can be
# right and the answer still wrong: docs/distributed/DISTRIBUTED.md §8.1 is exactly that
# case, `overloads.json` matching upstream 255/255 while
# `torch.ops.aten.add_.Tensor._schema.is_mutable` was False, because the
# schema the tree reads never came from a table at all. So this asks the shim
# itself, in a subprocess with the vendored tree on the path, and diffs.
#
# It must be a subprocess: this script needs *upstream* torch importable, and
# the shim needs the vendored tree, and one interpreter cannot have both --
# the same reason `tools/golden/compare.py` goes to a second process.

REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
VENDOR_DIR = os.path.join(REPO_ROOT, "torchnative", "src", "main")
VENDOR_SHIM = os.path.join(VENDOR_DIR, "torch", "_C.abi3.so")

_SHIM_REPORT_SCRIPT = r"""
import json, os, sys
import torch

# `import torch` alone reaches no placeholder predicate. The reads happen in
# the modules that *probe* the registry -- `torch/distributed/tensor/_ops/
# autogen.py` synthesises `<base>_` and `<base>_functional` per op and asks each
# one `is_mutable` -- so the check below is vacuous unless they are imported.
# This is the same trap as `check_non_aten` importing
# `torch.distributed._functional_collectives` to make upstream's own registry
# have the namespace at all.
out = {"probed": []}
for name in ("torch._refs", "torch._decomp", "torch._prims",
             "torch._meta_registrations", "torch._subclasses.functional_tensor",
             "torch.distributed.tensor", "torch.distributed.tensor._ops.autogen",
             "torch._functorch.partitioners", "torch._inductor.ir"):
    try:
        __import__(name)
    except Exception as error:  # noqa: BLE001
        out["probed"].append(f"{name}: {type(error).__name__}: {error}")
    else:
        out["probed"].append(name)

out["source"] = torch._C._shim_schema_source()
out["ops"] = {}

# Every `- func:` the file declares, as this shim re-prints it. 2584 entries,
# so this is the only check that reaches all five normalisation rules -- the
# implemented 117 exercise three of them and the transcribed tables exercise
# the same three.
declared = {}
if os.path.isabs(out["source"]):
    with open(out["source"], encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("- func:"):
                continue
            head = line[len("- func:"):].strip().split("(", 1)[0]
            name, _, overload = head.partition(".")
            key = f"aten::{name}|{overload}"
            declared[key] = {
                "text": str(torch._C._get_schema(f"aten::{name}", overload)),
                "from": torch._C._shim_schema_provenance(f"aten::{name}", overload),
            }
out["declared"] = declared
for key in torch._C._aten_implemented():
    namespace, _, rest = key.partition(".")
    name, _, overload = rest.rpartition(".")
    schema = getattr(getattr(torch.ops, namespace), name)
    schema = getattr(schema, overload)._schema
    out["ops"][key] = {
        "text": str(schema),
        "is_mutable": schema.is_mutable,
        "placeholder": schema.is_placeholder,
    }
out["unanswered"] = torch._C._shim_unanswered_predicates()

# docs/graph/DECOMP.md §3. `_jit_get_operation` used to report `["default"]` for
# every packet, so `@register_decomposition(aten.transpose)` landed on
# `aten.transpose.default` -- an overload no torch has. The overload list is
# read out of `native_functions.yaml` now, and upstream is what it is diffed
# against. Restricted to the packets of implemented ops: the file declares
# 1554 aten names and walking all of them here would trade a measurement for
# a wait.
_packets = sorted({key.rsplit(".", 1)[0] for key in torch._C._aten_implemented()})
out["overloads"] = {}
for _key in _packets:
    _namespace, _, _name = _key.partition(".")
    out["overloads"][_key] = getattr(
        getattr(torch.ops, _namespace), _name).overloads()

# The other half of §3: `core_aten_decompositions()` raised here because
# `CustomDecompTable.__init__` enumerates CompositeImplicitAutograd through
# this query. It is answered from the same file now.
out["cia"] = torch._C._dispatch_get_registrations_for_dispatch_key(
    "CompositeImplicitAutograd")

# docs/graph/DECOMP.md §2: `OpOverload.tags` was `[]` for every op.
out["tags"] = {}
for _key in torch._C._aten_implemented():
    _ns, _name, _ov = _key.split(".")
    out["tags"][_key] = sorted(
        tag.name for tag in
        getattr(getattr(torch.ops, _ns), _name).__getattr__(_ov).tags)
out["unknown_tags"] = torch._C._shim_unknown_tags()
try:
    torch._C._dispatch_get_registrations_for_dispatch_key("CPU")
except NotImplementedError as _error:
    out["cia_backend_refusal"] = str(_error)
else:
    out["cia_backend_refusal"] = None

json.dump(out, sys.stdout)
"""


@functools.lru_cache(maxsize=1)
def _shim_report():
    env = dict(os.environ)
    env["PYTHONPATH"] = VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    proc = subprocess.run(
        [sys.executable, "-c", _SHIM_REPORT_SCRIPT],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"shim subprocess exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout)


def check_shim_schemas(torch) -> tuple[int, int]:
    """Every implemented op's `_schema`, character for character, plus
    `is_mutable`.

    The judgement docs/distributed/DISTRIBUTED.md §8.1 set. `is_mutable` is checked
    separately from the text rather than being taken as implied by it, because
    the two failures this predicate has had were both invisible in the text:
    it was a bound method (always truthy), then a property over an empty
    argument list (always False).
    """
    report = _shim_report()
    if not os.path.isabs(report["source"]):
        print(f"FAIL shim schemas: no schema source -- {report['source']}")
        return 1, 1
    failures = 0
    for key, entry in sorted(report["ops"].items()):
        # The namespace off the key, not a literal `aten`. The implemented set
        # is two namespaces since docs/kernels/PRIMS.md, and `torch.ops.aten.clone`
        # exists -- so hardcoding `aten` would silently check
        # `prims.clone.default` against a *different operator's* schema and
        # call it a match, which is worse than the AttributeError
        # `broadcast_in_dim` happened to raise.
        namespace, _, rest = key.partition(".")
        op, _, overload = rest.rpartition(".")
        upstream = getattr(getattr(torch.ops, namespace), op)
        upstream = getattr(upstream, overload)._schema
        if entry["placeholder"]:
            failures += 1
            print(f"FAIL shim schemas {key}: still a placeholder")
            continue
        if _normalise(entry["text"]) != _normalise(str(upstream)):
            failures += 1
            print(f"FAIL shim schemas {key}:")
            print(f"     shim:     {entry['text']}")
            print(f"     upstream: {upstream}")
        if entry["is_mutable"] != upstream.is_mutable:
            failures += 1
            print(f"FAIL shim schemas {key}: is_mutable is "
                  f"{entry['is_mutable']}, upstream says {upstream.is_mutable}")
    # The predicate must be able to take both values over this set. A run where
    # every op agrees with upstream *and* every op answers the same thing would
    # pass the loop above while reproducing the defect, if upstream's answer
    # were ever uniform.
    answers = {entry["is_mutable"] for entry in report["ops"].values()}
    if answers != {True, False}:
        failures += 1
        print(f"FAIL shim schemas: is_mutable took only {answers} over "
              f"{len(report['ops'])} ops -- a constant predicate")
    return len(report["ops"]), failures


def check_declared_schemas(torch) -> tuple[int, int]:
    """All 2584 `- func:` entries, re-printed and diffed against upstream.

    This is the check the re-printer needs and the other two do not provide.
    `check_shim_schemas` covers the implemented 117 and the in-repo round-trip
    test covers the 173 in the transcribed tables; between them they exercise
    three of the five normalisation rules (float defaults, string quoting,
    enum-valued defaults). The list-broadcast rule -- `SymInt[2] stride=1` ->
    `[1, 1]`, and the `int[N>1]` exception that keeps `int[2] padding=0` a
    scalar -- appears on 101 arguments, none of them in either of those sets.
    Deleting it would leave both green.

    Residual was 165/2584 before the re-printer and is 0 after.
    """
    report = _shim_report()
    declared = report.get("declared") or {}
    if not declared:
        print("FAIL declared schemas: the shim reported no native_functions.yaml")
        return 1, 1
    failures = 0
    for key, entry in sorted(declared.items()):
        qualname, _, overload = key.partition("|")
        op = qualname[len("aten::"):]
        try:
            upstream = str(getattr(getattr(torch.ops.aten, op),
                                   overload or "default")._schema)
        except Exception:  # noqa: BLE001 -- declared here, not registered
            # upstream. `_foreach_*` entries behind a build flag land here.
            continue
        if _normalise(entry["text"]) != _normalise(upstream):
            failures += 1
            print(f"FAIL declared schemas {op}.{overload or 'default'}:")
            print(f"     shim:     {entry['text']}")
            print(f"     upstream: {upstream}")
        elif entry["from"] != "native_functions.yaml":
            failures += 1
            print(f"FAIL declared schemas {op}.{overload or 'default'}: the file "
                  f"declares this and `{entry['from']}` answered instead, so the "
                  "re-printer was not exercised")
    return len(declared), failures


def check_unanswered(torch) -> tuple[int, int]:
    """No operator upstream has may be answered from an empty schema.

    A placeholder answers `is_mutable` with False and records the fact
    (`_C._shim_unanswered_predicates()`). That is tolerable exactly while the
    ops involved do not exist -- upstream raises AttributeError at the packet
    and the caller's guard takes the same branch. The moment one of them is a
    real operator, `False` is a claim about it, which is how
    `aten::native_dropout_backward.out` (mutable upstream) was being answered.

    So the check is: for every recorded pair, upstream must have no such
    operator. This is the direction the in-repo test cannot take, because it is
    the one that needs a real torch.
    """
    report = _shim_report()
    failures = 0
    pairs = report["unanswered"]
    # Which probing imports actually landed. Not decoration: some of these fail
    # in this build (`torch._inductor.ir` reaches an `_Unimplemented`), and if
    # they all failed the check below would be vacuously green over an empty
    # set. Printed rather than asserted, because which of them import is a
    # property of the shim's coverage and moves on its own.
    landed = [name for name in report["probed"] if ":" not in name]
    print(f"    probes that imported: {len(landed)}/{len(report['probed'])}")
    for entry in report["probed"]:
        if ":" in entry:
            print(f"      (not probed) {entry.split(':')[0]}")
    if not landed:
        print("FAIL unanswered: no probing import landed, so the set is empty "
              "for the wrong reason")
        return 0, 1
    for spelling, predicate in pairs:
        qualname, _, overload = spelling.partition(".")
        namespace, _, name = qualname.partition("::")
        try:
            packet = getattr(getattr(torch.ops, namespace), name)
            upstream = getattr(packet, overload or "default")._schema
        except Exception:  # noqa: BLE001 -- no such operator, which is the pass
            continue
        failures += 1
        print(f"FAIL unanswered {spelling}: upstream has this operator, and "
              f"`{predicate}` was answered here from an empty schema")
        print(f"     upstream: {upstream}")
        print(f"     add it to _GENERATED_ATEN_SCHEMA_TEXT in bootstrap.py")
    return len(pairs), failures


def check_overload_names(torch) -> tuple[int, int]:
    """Every implemented op's packet must list the overloads upstream lists.

    docs/graph/DECOMP.md §3 is what this exists for. `_jit_get_operation` answered
    `["default"]` for every packet, and nothing in the repository could see
    that: the schema checks above ask about one `(name, overload)` at a time
    and are right whatever the *list* says, and in-tree tests have no upstream
    to compare a list to. The visible symptom was two numbers in an unrelated
    place -- the decomposition registry holding 592 entries against upstream's
    1097, 525 of ours ending in `.default` against upstream's 456.

    **The two directions are not the same claim, and only one of them is a
    failure.**

    An overload this lists that upstream does not have is the defect itself,
    one shape smaller: a packet-level rule would land on a key no dispatcher
    knows, which is exactly what `.default` was. That fails.

    An overload upstream has and this does not is expected and mostly
    *desirable*. `native_functions.yaml` declares 2584 entries and upstream's
    registry has 3754, the difference being torchgen's generated `.out`
    variants and TorchScript's numeric builtins -- `aten::sub.float_int`,
    `aten::sort.bool`. Upstream throws that second group away itself:
    `torch/_decomp/__init__.py:88` filters `op_overloads()` through
    `_dispatch_has_kernel` with the comment "TorchScript dumps a bunch of extra
    nonsense overloads which don't have corresponding dispatcher entries". So
    the missing direction is failed on one criterion only, and it is the one
    that costs something: **upstream registers a decomposition for it, and it
    is not an `.out` variant.**

    The `.out` carve-out is not a convenience. Every missing overload that
    carries a rule is one of torchgen's generated `.out` variants
    (docs/bindings/SCHEMA.md §12's 1148, which need `pyyaml` to generate), and no
    `.out` aten key is implemented in this build -- asserted below rather than
    said -- so no recording can contain one and no rule for one can be
    wanted. If a missing overload with a rule ever has another shape, that is
    a real loss and this fails with its name.
    """
    report = _shim_report()
    import torch._decomp as decomp

    registered = {str(key) for key in decomp.decomposition_table}
    failures = 0
    missing_total = 0
    unreachable: list = []
    for packet, overloads in sorted(report["overloads"].items()):
        namespace, _, name = packet.partition(".")
        try:
            upstream = getattr(getattr(torch.ops, namespace), name).overloads()
        except AttributeError:
            failures += 1
            print(f"FAIL overloads {packet}: upstream has no such packet")
            continue
        invented = sorted(set(overloads) - set(upstream))
        if invented:
            failures += 1
            print(f"FAIL overloads {packet}: not overloads of the upstream op: "
                  f"{invented}")
            print(f"     upstream: {sorted(upstream)}")
        missing = sorted(set(upstream) - set(overloads))
        missing_total += len(missing)
        for overload in missing:
            if f"{packet}.{overload}" not in registered:
                continue
            if overload == "out" or overload.endswith("_out"):
                unreachable.append(f"{packet}.{overload}")
                continue
            failures += 1
            print(f"FAIL overloads {packet}: upstream registers a "
                  f"decomposition for {overload!r} and this packet does not "
                  f"list it, and it is not an out-variant")

    # The carve-out's premise, checked rather than asserted in prose: if an
    # `.out` key were implemented, a trace could contain one and the rules
    # above would stop being unwanted.
    implemented_out = sorted(
        key for key in report["ops"]
        if key.rsplit(".", 1)[1] == "out" or key.endswith("_out")
    )
    if implemented_out:
        failures += 1
        print(f"FAIL overloads: this build implements out-variants "
              f"({implemented_out[:5]}), so the {len(unreachable)} "
              f"decompositions skipped above are no longer unreachable")

    print(f"    overloads upstream has and the file does not declare: "
          f"{missing_total}; {len(unreachable)} of them carry a decomposition "
          f"and all are out-variants no recording can contain")
    return len(report["overloads"]), failures


def check_tags(torch) -> tuple[int, int]:
    """`OpOverload.tags`, per implemented op, against upstream's.

    It was `[]` for everything. docs/graph/DECOMP.md §2 measured one consequence
    (`torch.Tag.core in op.tags` False for every op) and the CIA collection
    above is the other (`maybe_aliasing_or_mutating` unreadable, so
    `aten.dropout.default` was collected where upstream excludes it). Both are
    "an empty answer read as a negative", the same shape as
    docs/distributed/DISTRIBUTED.md §8.1 -- and the same reason this is diffed against a
    real torch rather than asserted in-repo.

    A tag name dropped for want of a `_C.Tag` member is a failure here even
    though the shim only records it: `Tag`'s members come from the vendored
    `.pyi` and the tags come from the vendored yaml, so a name in one and not
    the other means the two halves of one release disagree.
    """
    report = _shim_report()
    failures = 0
    for key, ours in sorted(report["tags"].items()):
        namespace, name, overload = key.split(".")
        try:
            upstream = sorted(
                tag.name for tag in
                getattr(getattr(torch.ops, namespace), name).__getattr__(
                    overload).tags)
        except AttributeError:
            failures += 1
            print(f"FAIL tags {key}: upstream has no such overload")
            continue
        if ours != upstream:
            failures += 1
            print(f"FAIL tags {key}: {ours} here, {upstream} upstream")
    if report["unknown_tags"]:
        failures += 1
        print(f"FAIL tags: the file used tag names `_C.Tag` has no member for: "
              f"{report['unknown_tags']}")
    return len(report["tags"]), failures


def check_cia_registrations(torch) -> tuple[int, int]:
    """The CompositeImplicitAutograd list, against the real dispatcher.

    `_dispatch_get_registrations_for_dispatch_key` is answered here from
    `native_functions.yaml` -- explicit `CompositeImplicitAutograd:` entries
    plus `torchgen/model.py:872`'s default, which gives the key to every entry
    with no `dispatch:` block that is neither structured nor a structured
    delegate. That second rule is most of the list, so a scan that only looked
    for the literal word would be quietly short.

    Both directions fail, on different criteria.

    A name here that upstream does not register would materialise an operator
    that is not CIA and hand `CustomDecompTable` a decomposition for it. That
    fails outright.

    A name upstream registers and this does not is allowed **only if the file
    does not declare that operator at all** -- the file is not a complete list
    of aten names (docs/bindings/SCHEMA.md §8.2), and 2.13.0's one absentee here is
    `aten::get_gradients`, a TorchScript builtin. A missing name the file
    *does* declare means the scan lost it, which is what dropping
    `torchgen/model.py:872`'s implicit default would do to two thirds of the
    list -- and a check that only looked for invented names would stay green
    through exactly that.
    """
    report = _shim_report()
    ours = set(report["cia"])
    upstream = {
        name for name in torch._C._dispatch_get_registrations_for_dispatch_key(
            "CompositeImplicitAutograd")
        if name.startswith("aten::")
    }
    failures = 0
    invented = sorted(ours - upstream)
    if invented:
        failures += len(invented)
        for name in invented[:10]:
            print(f"FAIL cia {name}: upstream does not register this under "
                  f"CompositeImplicitAutograd")
    # The file's own `- func:` list, spelled the way the dispatcher spells a
    # registration (`aten::max.other`, `aten::broadcast_tensors`).
    declared = set()
    for key in (report.get("declared") or {}):
        qualname, _, overload = key.partition("|")
        declared.add(f"{qualname}.{overload}" if overload else qualname)
    missing = sorted(upstream - ours)
    lost = [name for name in missing if name in declared]
    if lost:
        failures += len(lost)
        for name in lost[:10]:
            print(f"FAIL cia {name}: upstream registers this and the file "
                  f"declares it, so the scan lost it")
    if missing:
        print(f"    (upstream has {len(missing)} the file does not declare: "
              f"{', '.join(missing[:5])})")
    if report["cia_backend_refusal"] is None:
        failures += 1
        print("FAIL cia: a backend key was answered from the same file, which "
              "claims kernels this build does not have")
    # The union rather than either side: both directions are failures here, so
    # the population being diffed is every name either has.
    return len(ours | upstream), failures


def main() -> int:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment problem
        print(f"FATAL: needs an environment with upstream torch installed: {exc}",
              file=sys.stderr)
        return 2

    print(f"torch {torch.__version__}")
    total = 0
    failed = 0
    for label, path in TABLES:
        checked, failures = check(label, path, torch)
        print(f"  {label}: {checked - failures}/{checked} matched")
        total += checked
        failed += failures

    checked, failures = check_non_aten(torch)
    print(f"  bootstrap.py _NON_ATEN_SCHEMA_TEXT: {checked - failures}/{checked} matched")
    total += checked
    failed += failures

    checked, failures = check_generated_aten(torch)
    print(f"  bootstrap.py _GENERATED_ATEN_SCHEMA_TEXT: "
          f"{checked - failures}/{checked} matched")
    total += checked
    failed += failures

    # The tables are not the answer; what the shim hands the tree is. Needs a
    # built vendored shim, which is a separate build step (`install_shim.sh`),
    # so its absence is reported rather than treated as a pass.
    if os.path.isfile(VENDOR_SHIM):
        checked, failures = check_shim_schemas(torch)
        print(f"  shim _schema text and is_mutable: "
              f"{checked - failures}/{checked} matched upstream")
        total += checked
        failed += failures

        checked, failures = check_declared_schemas(torch)
        print(f"  native_functions.yaml re-printed: "
              f"{checked - failures}/{checked} matched upstream")
        total += checked
        failed += failures

        checked, failures = check_unanswered(torch)
        print(f"  predicates answered without text: {checked}, "
              f"{checked - failures}/{checked} about ops upstream does not have")
        total += checked
        failed += failures

        checked, failures = check_overload_names(torch)
        print(f"  packet overload lists: {checked - failures}/{checked} "
              f"matched upstream")
        total += checked
        failed += failures

        checked, failures = check_tags(torch)
        print(f"  OpOverload.tags: {checked - failures}/{checked} matched upstream")
        total += checked
        failed += failures

        checked, failures = check_cia_registrations(torch)
        print(f"  CompositeImplicitAutograd registrations: "
              f"{checked - failures}/{checked} are upstream's")
        total += checked
        failed += failures
    else:
        print(f"  shim _schema text: SKIPPED -- no {VENDOR_SHIM}")
        print("    (run vendor/install_shim.sh; this is the check that would "
              "have caught docs/distributed/DISTRIBUTED.md §8.1)")

    # The other direction is deliberately *not* an error. The tables list the
    # overloads torch's Python bindings expose, which is a subset: `aten::pow`
    # has fifteen overloads and `torch.pow` reaches three of them.
    print()
    print(f"SUMMARY: {total - failed}/{total} table entries matched upstream, "
          f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
