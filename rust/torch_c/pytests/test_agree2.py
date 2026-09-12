"""docs/numerics/AGREE2.md -- what holds the re-measured agreement counts down.

`agree_sweep.py` is not wired into `run.sh` and must not be: it needs both
interpreters and hours. So the counts AGREE2.md publishes cannot be re-taken by
the gate, and the previous round showed exactly what that costs -- its artefacts
were in `/tmp`, they were gone by the next session, and the state of the
project's strongest claim was not knowable without re-running the whole sweep.

The fix is to store the *per-architecture measurements* rather than the counts,
and re-derive the counts here by calling `agree_sweep.verdict` itself. That
draws the line where it belongs:

  * A change to the RULE that LOWERS agreement -- a deleted refusal branch,
    architectures dropped from the recording -- moves these counts and fails
    this file. A change that WIDENS it does not, and cannot: the markers are
    `ge`/`le` because a later round may legitimately raise the count, so
    ORACLE_FACTOR 4 -> 40 leaves every count assertion green (measured). That
    direction is held by `test_agree.py`'s
    `test_the_oracle_factor_is_stated_and_is_not_a_free_parameter`, and the two
    files are only jointly sufficient -- which is the honest statement, since
    widening a threshold until a flag disappears is exactly what CLAUDE.md
    warns about.
  * A change to the TREE -- an operator landing, a kernel regressing -- cannot
    be caught here, and nothing in this file pretends otherwise. Only re-running
    the sweep settles that. `test_the_recording_says_when_it_was_taken` is the
    one honest guard: the measurement carries its own date.

The no-regression invariant is the other thing pinned here, and it is stated
structurally rather than as a list of names: every architecture that diverges
must either be the one `docs/numerics/AGREE.md` already flagged, or one of the
seven that document's §7 listed as unmeasured. Anything else means something
that agreed on 2026-08-24 stopped agreeing, which is the most important thing
this sweep could ever report.

Nothing here needs torch, the network, a checkpoint, or `transformers`.
"""

import json
import os
import re
import sys

import agree_sweep as A

_PYTESTS = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PYTESTS, "..", "..", ".."))
_SCORES = os.path.join(_PYTESTS, "agree2_scores.json")
_DOC = os.path.join(_REPO_ROOT, "docs", "numerics", "AGREE2.md")
_CHECK_DOCS = os.path.join(_REPO_ROOT, "tools", "docwatch", "check_docs.py")

# `docs/numerics/AGREE.md` §7, 2026-08-24: the seven that forwarded upstream and
# not under the shim, so the round that measured 284/285 never scored them.
# AGREE2 scores all seven, which is why its `diverge` column can legitimately
# grow without anything having regressed.
_NEWLY_REPLAYING = {
    "fastspeech2_conformer", "led", "longformer", "nystromformer",
    "sam3_lite_text_text_model", "univnet", "vilt",
}
# The single flag AGREE.md §3.1 recorded, kept because a rule that is edited
# whenever it fires is not a rule.
_ALREADY_FLAGGED = {"chinese_clip"}


def _rec():
    with open(_SCORES) as fh:
        return json.load(fh)


def _tally():
    rec = _rec()
    out = {}
    for s in rec["scores"]:
        v = A.verdict(s["rel"], s["scale"], s["oracle_rel"], rec["tol"],
                      s["self_repeat_rel"])
        out.setdefault(v, []).append(s["model_type"])
    return rec, out


def test_the_recording_exists_and_carries_every_field_a_verdict_needs():
    """A record missing one field would silently change the verdict rather than
    fail: `verdict` treats a `None` oracle as "no ratio rule available", which
    turns an `agree_within_float32` into a `diverge`."""
    rec = _rec()
    assert rec["scores"], "no per-architecture scores recorded"
    for s in rec["scores"]:
        for k in ("model_type", "rel", "scale", "oracle_rel", "self_repeat_rel",
                  "load", "oracle", "key"):
            assert k in s, f"{s.get('model_type')} is missing {k}"


def test_the_recording_says_when_it_was_taken():
    """The only guard against the tree moving under a measurement nobody
    retook. AGREE2.md §7 lists that as the limit it cannot close."""
    rec = _rec()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", rec["taken"]), rec.get("taken")
    assert rec["commit"] and rec["upstream_python"].endswith("python")


def test_the_scored_population_is_exactly_what_both_sides_produced():
    """297 produced upstream, 297 replayed, 297 scored. A silently dropped
    architecture would raise the agreement percentage for free."""
    rec = _rec()
    assert rec["produced_upstream"] == rec["replayed_shim"] == len(rec["scores"]), (
        rec["produced_upstream"], rec["replayed_shim"], len(rec["scores"]))
    assert len(rec["scores"]) >= 297
    assert len({s["model_type"] for s in rec["scores"]}) == len(rec["scores"])


def test_the_headline_counts_are_what_the_verdict_rule_produces_today():
    """AGREE2.md §1's numbers, re-derived. Editing the rule moves these."""
    rec, t = _tally()
    agree = len(t.get("exact", [])) + len(t.get("agree", [])) + \
        len(t.get("agree_within_float32", []))
    unjudgeable = len(t.get("degenerate", [])) + len(t.get("nondeterministic", []))
    assert agree >= 288, agree
    assert len(t.get("diverge", [])) <= 2, t.get("diverge")
    assert len(rec["scores"]) - unjudgeable >= 290
    assert unjudgeable <= 7, t


def test_the_two_refusal_outcomes_cannot_be_deleted_into_passes():
    """The counts in the test above are `ge`/`le`, so deleting a refusal branch
    RAISES the agreement count and nothing fires -- measured: removing the
    `degenerate` check from `verdict` left every count assertion green. So the
    seven refusals are pinned by name and by the branch that produces them.
    AGREE2.md §5 re-examines each; this only refuses to let them be
    reclassified silently."""
    rec, t = _tally()
    refused = set(t.get("degenerate", [])) | set(t.get("nondeterministic", []))
    for m in ("efficientnet", "sam_vision_model", "sam_hq_vision_model"):
        assert m in t.get("degenerate", []), f"{m} is no longer refused as degenerate"
    for m in ("vit_mae", "vits", "univnet", "vilt"):
        assert m in t.get("nondeterministic", []), f"{m} is no longer refused"
    assert not (refused & set(t.get("agree", []) + t.get("agree_within_float32", []))), \
        "an architecture is being counted both ways"


def test_no_architecture_that_previously_agreed_now_diverges():
    """The whole point of re-measuring. Structural, not a name list: a diverge
    is admissible only if the previous round never scored it."""
    _, t = _tally()
    for m in t.get("diverge", []):
        assert m in _ALREADY_FLAGGED or m in _NEWLY_REPLAYING, (
            f"{m} diverges and was scored as agreeing on 2026-08-24 -- "
            "that is a regression, not a larger population")


def test_all_seven_formerly_blocked_architectures_are_scored_now():
    rec = _rec()
    have = {s["model_type"] for s in rec["scores"]}
    assert _NEWLY_REPLAYING <= have, sorted(_NEWLY_REPLAYING - have)


def test_the_state_dict_reached_the_shim_intact_for_every_architecture():
    """Identical weights are the whole experiment. A missing key means the two
    sides ran different numbers and every figure about them is about RNG."""
    rec = _rec()
    bad = [s["model_type"] for s in rec["scores"]
           if s["load"].get("missing") or s["load"].get("unexpected")]
    assert not bad, bad


def test_the_tolerance_is_derived_and_never_below_its_floor():
    """`8 * F32_EPS` is the floor AGREE.md §2 argues for. A tolerance below it
    would be measuring LayerNorm's last ulp, not the shim."""
    rec = _rec()
    assert rec["tol"] >= 8 * A.F32_EPS, rec["tol"]
    assert rec["tol"] <= 1e-05, "a tolerance this loose is not p90 of anything"


def test_the_architectures_without_an_oracle_are_held_to_the_fixed_tolerance():
    """They cannot reach the ratio rule, so for them `diverge` means only
    "above the fixed tolerance". AGREE2.md §3 says so about the one that is."""
    rec = _rec()
    nof64 = {s["model_type"] for s in rec["scores"] if s["oracle"] != "ok"}
    assert len(nof64) <= 23, sorted(nof64)
    assert "fastspeech2_conformer" in nof64
    for s in rec["scores"]:
        if s["model_type"] in nof64:
            assert s["oracle_rel"] is None, s["model_type"]


def test_docwatch_exposes_the_agreement_counts_as_live_sources():
    """Without these, a count in AGREE2.md is unpinnable prose -- which is how
    ARCH100's 82 survived three superseding rounds."""
    with open(_CHECK_DOCS) as fh:
        text = fh.read()
    for name in ("agree_agree", "agree_diverge", "agree_judgeable",
                 "agree_unjudgeable", "agree_no_oracle", "agree_scored",
                 "agree_state_dict_clean"):
        assert re.search(r'"%s":\s*lambda' % name, text), f"{name} not in COUNT_SOURCES"
    assert "agree_sweep" in text, "the rule must be re-derived, not stored"


def test_the_document_pins_its_counts_and_uses_ge_where_a_round_could_raise_them():
    """CLAUDE.md is explicit: `ge` for anything a later round could legitimately
    raise, and the exceptions are the ones that must stay at zero -- here,
    divergence and unjudgeability, which are `le`."""
    with open(_DOC) as fh:
        doc = fh.read()
    markers = dict(re.findall(r"DOCWATCH: count (agree_\w+) (eq|ge|le)", doc))
    assert markers, "AGREE2.md pins no count at all"
    for name in ("agree_agree", "agree_judgeable", "agree_scored",
                 "agree_produced_upstream", "agree_replayed_shim",
                 "agree_state_dict_clean"):
        assert markers.get(name) == "ge", (name, markers.get(name))
    for name in ("agree_diverge", "agree_unjudgeable", "agree_no_oracle"):
        assert markers.get(name) == "le", (name, markers.get(name))
    assert "eq" not in markers.values(), markers


def test_the_document_does_not_restate_a_count_it_did_not_measure():
    """AGREE.md's 775-leaf / 8.4-ulp figure was not re-taken this round, so
    AGREE2.md must cite it with its date rather than assert it."""
    with open(_DOC) as fh:
        doc = fh.read()
    if "8.4 ulp" in doc:
        assert "2026-08-24" in doc and "did not re-run" in doc, (
            "an inherited figure must name the round it came from")


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
