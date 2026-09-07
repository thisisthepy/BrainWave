"""The QNN ahead-of-time lowering, as a script a CI job runs and a suite imports.

`docs/QNN.md` §1.3 measured why the lowering cannot run on this project's
machine: the `executorch` macOS-arm64 wheel has no
`executorch/backends/qualcomm/python/`, all 112 op builders import
`PyQnnManagerAdaptor` at module scope, the QNN SDK's host libraries exist only
under `lib/x86_64-linux-clang/`, and ExecuTorch's own installer gates on
`is_linux_x86()`. A GitHub-hosted `ubuntu-*` runner is a native Linux x86-64
host, which is the one thing that was missing. `.github/workflows/qnn-lower.yml`
is the job; this file is everything that job decides.

**Why this is a file and not inline YAML.** `CLAUDE.md` §5.5 -- a verification
that cannot fail is not a verification. Logic that lives only inside a
`run: |` block cannot be imported, cannot be nullified, and cannot be tested
anywhere except by pushing to a branch and waiting. `tools/ci/verify_published.py`
is this repository's existing answer to that and this file follows it.
`rust/torch_c/pytests/test_qnnci.py` is what exercises the decisions below,
on a machine that has no executorch and never will.

**The decision this file exists to make.** `docs/QNN.md` §6.2: QNN's dangerous
fallback is at *compile* time, not run time. `QnnPartitioner` silently declines
whatever it cannot handle, those nodes stay in the program as portable CPU
kernels, and the resulting `.pte` loads, runs and returns the right answer. So
an artefact can be uploaded, pushed to a phone and run, and be a CPU program
throughout, with nothing in its output saying so. §5 step 7 is the check, and
this file runs it *before* the upload rather than after.

Two classes of failure, kept apart, because they call for different handling:

* **integrity** -- the file is not what the job would be claiming it is. Not a
  QNN program at all, or built for the wrong silicon. Uploading it would put a
  mislabelled artefact in front of the next person, so the job must **not**
  upload. This is item 4's list: `is_qnn`, `backend_ids`, `match_device`.

* **coverage** -- the file *is* a genuine QNN program, and the partitioner
  claimed too little of it. `delegated_fraction` is the number, §5 step 6
  asserts `> 0.9`, and `docs/QNN.md` §11 says plainly that its value for a real
  `LlamaMLP` under `QnnPartitioner` is **UNKNOWN and may be poor** -- citing
  `docs/REFOLD.md` §1.1, where this project assumed op-table coverage and was
  wrong three times. So this fails the job loudly, and the artefact is still
  uploaded, because a genuine QNN program with poor coverage is *the finding*
  and throwing it away would destroy the evidence for it.

**The threshold is a constant, not an input.** It would be trivial to expose
`--min-delegated-fraction` on the workflow and let a maintainer dial it down
until the run went green. That is the specific thing `docs/QNN.md` §11 warns
against and the reason it is `MIN_DELEGATED_FRACTION` below, cited to the line
of the doc that chose it. If the real number is 0.34, the job goes red and the
right response is to write 0.34 into `docs/QNN.md` §11 as a measurement --
not to edit this constant so the red goes away.
"""

import argparse
import json
import os
import sys


# docs/QNN.md §5 step 6: `assert report["delegated_fraction"] > 0.9, report`.
# Copied from the procedure rather than chosen here, and deliberately strict
# (`>`, not `>=`) for the same reason.
MIN_DELEGATED_FRACTION = 0.9

# docs/QNN.md §4.3 / `torchnative.export.qnn.QNN_BACKEND_ID`. Restated rather
# than imported so that this module's decisions can be tested on a host where
# `torchnative.export.qnn` cannot be imported at all -- and
# `test_qnnci.py::test_the_backend_id_here_matches_the_one_the_module_uses`
# is what keeps the two from drifting apart.
QNN_BACKEND_ID = "QnnBackend"

# The device that will run this. `CLAUDE.md` §7: Galaxy Tab S9 Ultra,
# `ro.soc.model = SM8550`, Snapdragon 8 Gen 2, HTP v73, arm64-v8a.
# `docs/QNN.md` §5.1 read those off the physical device rather than assuming
# them. The doc's *example* says SM8650, which is a different HTP generation
# (v75) and would produce an artefact this device refuses at backend init --
# so the default here is the measured value and the workflow takes it as an
# input.
DEFAULT_SOC_MODEL = "SM8550"

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_SUBMODULE = "model.layers.0.mlp"


class Verdict:
    """What the checks concluded, split by what the job should do about it.

    `integrity` gates the upload. `coverage` does not -- it fails the job while
    letting the artefact through, because a genuine QNN program that the
    partitioner barely claimed is exactly the measurement `docs/QNN.md` §11
    says is missing, and deleting it would be deleting the answer.
    """

    def __init__(self, facts, integrity, coverage):
        self.facts = facts
        self.integrity = list(integrity)
        self.coverage = list(coverage)

    @property
    def integrity_ok(self):
        return not self.integrity

    @property
    def ok(self):
        return not self.integrity and not self.coverage

    @property
    def failures(self):
        return self.integrity + self.coverage

    def __repr__(self):
        return (
            f"<Verdict integrity_ok={self.integrity_ok} ok={self.ok} "
            f"integrity={self.integrity!r} coverage={self.coverage!r}>"
        )


def verify_facts(facts, soc_model, min_fraction=MIN_DELEGATED_FRACTION):
    """`docs/QNN.md` §5 step 7 plus step 6's assertion, as a pure function.

    `facts` is what `collect_facts` reads off a real artefact; taking it as a
    plain dict rather than reaching for `torchnative.export.qnn` here is what
    makes every branch below reachable from a test on a host with no
    executorch. The point of that is not convenience -- it is that the failure
    branches are the ones that matter, and they are exactly the branches a
    green CI run never takes.
    """
    integrity = []
    coverage = []

    backend_ids = tuple(facts.get("backend_ids") or ())

    # (1) §5 step 7: `assert a.is_qnn, a.backend_ids`.
    #
    # `is_qnn` is "at least one QNN segment", not "all" (§4.3) -- a real model
    # legitimately mixes QNN segments with portable kernels for what the
    # partitioner declined. What it must never be is True for a program with
    # zero QNN segments, which is the entire silent-fallback shape.
    if not facts.get("is_qnn"):
        integrity.append(
            f"read_artefact(...).is_qnn is False. The delegates in this "
            f"program are {backend_ids or '()'}. This is not a QNN artefact: "
            f"either the lowering silently produced a portable-kernel program, "
            f"or it fell through to a different partitioner. docs/QNN.md §6.2 "
            f"row 1 -- it would still load, still run, and still return the "
            f"right answer on the CPU."
        )

    # (2) The backend id by name. Not redundant with `is_qnn`: `is_qnn` is a
    # property computed by `torchnative.export.qnn`, and this reads the ids
    # the artefact actually carries. If the two ever disagree, that
    # disagreement is itself the finding -- so both are checked rather than
    # trusting the derived one. (docs/QNN.md §4.3 on why there is no second
    # parser here: both of these come from upstream's own deserialiser.)
    if QNN_BACKEND_ID not in backend_ids:
        integrity.append(
            f"no delegate in this program is named {QNN_BACKEND_ID!r}; "
            f"backend_ids = {backend_ids or '()'}. Whatever this file is, it "
            f"is not the thing the job would be uploading it as."
        )

    # (3) §5 step 7: `ok, why = qnn.match_device(a, SOC); assert ok, why`.
    #
    # This compares HTP *architecture*, not part number (§4.3). Catching it
    # here costs nothing; catching it on the device costs a push, a run and an
    # abort -- QNN validates the arch at backend init and fails the method
    # load with `Arch NN ... is different from arch associated with SoC NN`
    # (§6.3). It does not fall back.
    if not facts.get("match_ok"):
        integrity.append(
            f"match_device(artefact, {soc_model!r}) refused: "
            f"{facts.get('match_why') or 'no reason given'}"
        )

    delegation = facts.get("delegation") or {}
    total = delegation.get("total_nodes")
    fraction = delegation.get("delegated_fraction")

    # (4a) An empty graph is an integrity failure, not a coverage one.
    #
    # `delegation_report` returns `delegated_fraction = 0.0` when `total` is 0,
    # so without this the "nothing was lowered at all" case would be reported
    # as "poor coverage" and read as a partitioner result. It is not a
    # partitioner result; it means there was no graph. Naming it separately is
    # the §5.5 habit -- a check whose failure message sends the reader to the
    # wrong place is only half a check.
    if not total:
        integrity.append(
            f"the delegation report describes an empty graph "
            f"(total_nodes={total!r}). Nothing was exported, so nothing was "
            f"partitioned, and `delegated_fraction` below is not a "
            f"partitioner result. docs/QNN.md §6.1(a)."
        )
    elif fraction is None:
        integrity.append(
            "the delegation report carries no `delegated_fraction`. "
            "delegation_report() changed shape, or this is not one."
        )
    # (4b) §5 step 6: `assert report["delegated_fraction"] > 0.9, report`.
    #
    # docs/QNN.md §11: this number is UNKNOWN for a real `LlamaMLP` under
    # `QnnPartitioner` and **may be poor**. Do not tune it to make the run
    # green -- §11 cites docs/REFOLD.md §1.1, three occasions of assuming
    # coverage and being wrong. If this fires, the number is the deliverable.
    elif not fraction > min_fraction:
        coverage.append(
            f"delegated_fraction is {fraction:.4f} "
            f"({delegation.get('delegated_nodes')} of {total} nodes in "
            f"{delegation.get('subgraphs')} subgraph(s)), which does not clear "
            f"the {min_fraction} that docs/QNN.md §5 step 6 asserts. The "
            f"artefact IS a QNN program -- the integrity checks passed -- so "
            f"the {delegation.get('non_delegated_nodes')} node(s) "
            f"`QnnPartitioner` declined stay in it as portable CPU kernels and "
            f"will run on the application processor. docs/QNN.md §11 says this "
            f"value was unknown and might be poor. It is now known. Record it "
            f"in §11 as a measurement; do NOT lower MIN_DELEGATED_FRACTION."
        )

    return Verdict(facts, integrity, coverage)


def collect_facts(qnn, path, soc_model, report):
    """Read `docs/QNN.md` §5 step 7's four answers off a real `.pte`.

    `qnn` is passed in rather than imported so this stays callable with a
    double in the local suite. Every read goes through
    `torchnative.export.qnn`, which reads through **upstream's own**
    deserialiser (§4.3) -- there is no second parser here that could agree
    with the first by sharing a mistake.
    """
    artefact = qnn.read_artefact(path)
    match_ok, match_why = qnn.match_device(artefact, soc_model)
    facts = {
        "path": path,
        "soc_model": soc_model,
        "backend_ids": tuple(artefact.backend_ids),
        "is_qnn": bool(artefact.is_qnn),
        "match_ok": bool(match_ok),
        "match_why": match_why,
        "delegation": dict(report or {}),
    }
    try:
        facts["htp_plan"] = {
            k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
            for k, v in artefact.htp_plan().items()
        }
    except Exception as exc:  # noqa: BLE001
        # Not fatal on its own: `match_device` already read the plan and had
        # to succeed for `match_ok`. This is for the log.
        facts["htp_plan"] = f"unreadable: {type(exc).__name__}: {exc}"
    return facts


def render_summary(verdict, soc_model):
    """Markdown for `$GITHUB_STEP_SUMMARY`.

    `docs/QNN.md` §11 item 1 asks for `delegated_fraction` to be *reported*,
    not merely asserted on, so it is printed on every path -- including the
    paths where the job is about to go red, which are the ones where somebody
    will actually want the number.
    """
    facts = verdict.facts
    delegation = facts.get("delegation") or {}
    lines = []
    lines.append("## QNN ahead-of-time lowering")
    lines.append("")
    lines.append(f"* artefact: `{facts.get('path')}`")
    lines.append(f"* target SoC: `{soc_model}`")
    lines.append(f"* backend ids: `{facts.get('backend_ids')}`")
    lines.append(f"* `is_qnn`: `{facts.get('is_qnn')}`")
    lines.append(
        f"* `match_device`: `{facts.get('match_ok')}` "
        f"-- {facts.get('match_why')}"
    )
    lines.append(f"* `htp_plan`: `{facts.get('htp_plan')}`")
    lines.append("")
    lines.append("### delegation_report")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    for key in ("subgraphs", "delegated_nodes", "non_delegated_nodes",
                "total_nodes", "delegated_fraction"):
        lines.append(f"| `{key}` | `{delegation.get(key)}` |")
    lines.append("")
    fraction = delegation.get("delegated_fraction")
    if isinstance(fraction, (int, float)):
        lines.append(
            f"`delegated_fraction` is **{fraction:.4f}** against a threshold of "
            f"{MIN_DELEGATED_FRACTION} (docs/QNN.md §5 step 6). §11 recorded "
            f"this value as UNKNOWN for a real `LlamaMLP` under "
            f"`QnnPartitioner`; this run is the measurement."
        )
        lines.append("")

    if verdict.ok:
        lines.append(
            "**Lowering succeeded and the artefact is a QNN program built for "
            "this silicon.** This proves NOTHING about HTP execution -- see "
            "the workflow header and docs/QNN.md §6.4."
        )
    else:
        if verdict.integrity:
            lines.append("### INTEGRITY FAILURES -- artefact NOT uploaded")
            lines.append("")
            for item in verdict.integrity:
                lines.append(f"* {item}")
            lines.append("")
        if verdict.coverage:
            lines.append("### COVERAGE FAILURE -- artefact uploaded anyway")
            lines.append("")
            for item in verdict.coverage:
                lines.append(f"* {item}")
            lines.append("")
    return "\n".join(lines) + "\n"


def _emit(text, stream=None):
    (stream or sys.stdout).write(text)


def _write_github_output(pairs, path=None):
    """Append `name=value` lines to `$GITHUB_OUTPUT`, if it exists.

    The workflow gates the upload step on `integrity_ok`, so this is load
    bearing rather than cosmetic: if it silently does nothing, the upload is
    skipped and the job's failure is confusing rather than wrong -- which is
    the safe direction, and is why there is no fallback to "assume ok".
    """
    path = path or os.environ.get("GITHUB_OUTPUT")
    if not path:
        return False
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in pairs.items():
            handle.write(f"{key}={value}\n")
    return True


def _write_step_summary(text, path=None):
    path = path or os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return False
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)
    return True


def build_parser():
    parser = argparse.ArgumentParser(
        description="Lower one submodule to a QNN-delegated .pte and verify it."
    )
    parser.add_argument("--soc-model", default=DEFAULT_SOC_MODEL)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--submodule", default=DEFAULT_SUBMODULE)
    parser.add_argument("--out", default="layer0_mlp.pte")
    parser.add_argument("--facts-json", default="delegation_report.json")
    parser.add_argument(
        "--seq-len", type=int, default=5,
        help="example-input sequence length; docs/QNN.md §5 step 6 uses 5",
    )
    # Deliberately NOT a --min-delegated-fraction. See the module docstring:
    # an adjustable threshold is a threshold that gets adjusted until it
    # passes, and docs/QNN.md §11 is specifically about that failure.
    return parser


def _resolve_submodule(model, path):
    """`model.layers.0.mlp` -> the module, or a named refusal.

    Refuses by name (`CLAUDE.md` §6) rather than raising a bare
    `AttributeError` from somewhere three frames down, because the most likely
    cause of a miss here is a `transformers` version whose Llama attribute
    names moved, and that reader needs to be told which component was absent.
    """
    current = model
    walked = []
    for part in path.split("."):
        walked.append(part)
        if part.isdigit() and hasattr(current, "__getitem__"):
            current = current[int(part)]
            continue
        if not hasattr(current, part):
            raise SystemExit(
                f"torchnative qnn-ci: no submodule {path!r} in this model -- "
                f"{'.'.join(walked)} is absent. Available at that level: "
                f"{sorted(n for n, _ in getattr(current, 'named_children', lambda: [])())}"
            )
        current = getattr(current, part)
    return current


def main(argv=None):
    args = build_parser().parse_args(argv)

    # `torchnative.export.qnn` is a thin driver (docs/QNN.md §2): the exporter
    # and the partitioner are UPSTREAM's. So this process must be running
    # upstream torch, NOT this project's shim -- which is the opposite of the
    # `assert hasattr(torch._C, "_aten_implemented")` that CLAUDE.md §3 asks
    # of every *probe*. That rule exists so a probe of our shim cannot
    # silently measure upstream; here upstream is the thing under test, and
    # the shim would be the contamination. Asserted in that direction:
    import torch

    if hasattr(torch._C, "_aten_implemented"):
        raise SystemExit(
            "torchnative qnn-ci: this interpreter has the torchnative shim "
            "installed. The QNN lowering is upstream torch + upstream "
            "executorch (docs/QNN.md §2); running it under the shim would "
            "measure this project's torch.export, which docs/EXPORT5.md §0 "
            "measures at 0 of 40 architectures. Refusing."
        )

    sys.path.insert(0, os.path.join("torchnative", "src", "main"))
    from torchnative.export import qnn

    # The wall, named, before anything else happens. On a host where the QNN
    # AOT half is missing this is the entire output of the job, and it names
    # the module rather than surfacing as an ImportError from inside a
    # partitioner pass. docs/QNN.md §1.3 is the measurement it encodes.
    # `probe()` rather than calling `executorch_version()` / `qnn_sdk_version()`
    # directly: those two raise `QnnRefused` when executorch is absent, and on
    # the host this job is most likely to fail on, executorch being absent is
    # precisely the thing we want *reported* rather than raised over. `probe()`
    # returns the whole picture and says which half is missing.
    report_probe = qnn.probe()
    print(json.dumps(
        {k: (list(v) if isinstance(v, tuple) else v)
         for k, v in report_probe.items()},
        indent=2, sort_keys=True,
    ))
    refusal = report_probe["qnn_aot_refusal"]
    if refusal is not None:
        raise SystemExit(
            f"torchnative qnn-ci: the QNN ahead-of-time half is not available "
            f"on this runner -- {refusal}\n\n"
            f"This job exists because docs/QNN.md §1.3 measured that it is "
            f"unavailable on this project's own machine and a Linux x86-64 "
            f"host is what it needs. If this fires HERE, then a hosted "
            f"ubuntu runner is not sufficient either, and that is a finding "
            f"for docs/QNNCI.md -- not something to work around."
        )

    if args.soc_model not in qnn.soc_targets():
        raise SystemExit(
            f"torchnative qnn-ci: {args.soc_model!r} is not in ExecuTorch's "
            f"own QcomChipset table. Known: {sorted(qnn.soc_targets())}"
        )

    from transformers import AutoModelForCausalLM

    print(f"loading {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(args.model_id).eval()
    submodule = _resolve_submodule(model, args.submodule)
    example = torch.randn(1, args.seq_len, model.config.hidden_size)
    print(f"lowering {args.submodule}: {type(submodule).__name__}")

    # fp16=True: docs/QNN.md §7/§7.2. `kHtpQuantized` is QNN's *default* and
    # needs a calibrated quantizer, which §11 names as a different round.
    path, report = qnn.lower(
        submodule, (example,), args.soc_model, args.out, fp16=True
    )
    print(f"delegation: {json.dumps(report, sort_keys=True)}")

    facts = collect_facts(qnn, path, args.soc_model, report)
    verdict = verify_facts(facts, args.soc_model)

    with open(args.facts_json, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "facts": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in facts.items()},
                "integrity_failures": verdict.integrity,
                "coverage_failures": verdict.coverage,
                "min_delegated_fraction": MIN_DELEGATED_FRACTION,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    summary = render_summary(verdict, args.soc_model)
    _emit(summary)
    _write_step_summary(summary)
    _write_github_output({
        "integrity_ok": "true" if verdict.integrity_ok else "false",
        "delegated_fraction": str(
            (facts.get("delegation") or {}).get("delegated_fraction")
        ),
    })

    for item in verdict.failures:
        print(f"FAIL {item}", file=sys.stderr)
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
