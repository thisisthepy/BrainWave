# A standing check for docs/verification/AUDIT.md's findings

`docs/verification/AUDIT.md` (eleven documents, six of eleven carrying a false claim, one
mechanism nearly every time: a later unrelated commit closed a gap and nobody
came back to the document that had named it) ends by asking whether this
needs a standing check rather than another manual pass, and answers yes. This
document is that check's design record, written incrementally as the design
was decided rather than reconstructed afterward.

## The core design decision, made first

Free-text prose cannot be parsed reliably into a checkable claim.
`docs/verification/AUDIT.md` itself is the proof: its findings quote things like "37개 중
9개" and "여전히 한 번도 실행되지 않았습니다" — no regex distinguishes those
from the surrounding narrative sentences that are *not* claims (design
rationale, historical color, a sentence explaining why a number is what it
is). A pattern match broad enough to catch prose like that is also broad
enough to fire on sentences that only *look* like the pattern, and CLAUDE.md
already has the relevant judgment on this repository's other checkers
(`run.sh`'s `cmp` exit-code handling, `build.py`'s three-way verdict): a check
that cannot distinguish "wrong" from "could not tell" gets ignored, and an
ignored check is worse than no check because it still looks like coverage.

Three designs were on the table:

1. **A machine-readable marker the documents opt into.** An HTML comment
   (invisible when the file renders) naming exactly what to check and against
   what ground truth. Zero false positives by construction — nothing fires
   unless someone deliberately wrote a marker — at the cost of coverage: only
   claims someone annotated are checked.
2. **Conservative pattern matching over the prose**, with a manual allow-list
   for everything else. Rejected: the allow-list *is* the coverage, so this
   degenerates to design 1 with worse ergonomics (a side-list instead of an
   inline marker) and a harder-to-audit diff (the pattern and the text it
   matches are not adjacent).
3. **A narrower check covering only counts and symbol references**, skipping
   "X is missing" prose entirely. Rejected as the *only* design: counts and
   symbols are two of AUDIT.md's four claim categories, and the highest-
   damage category in AUDIT.md's own account (DESIGN.md's `from_pretrained`
   claim, ARCH20.md's three invented-spelling names) is exactly the "X is
   missing" prose that this would skip.

**Chosen: design 1, markers**, with the primitive set narrowed to what
AUDIT.md actually verified this round (op-implemented, symbol-exists,
json-key, hasattr, count) rather than inventing new claim shapes. A marker
sits immediately next to the sentence it backs, in an HTML comment:

```
어떤 이름이 여전히 거부됩니다.
<!-- DOCWATCH: op-not-implemented aten.clamp.Tensor -->
```

Design 3's narrowing is still real inside design 1: op-existence, symbol-
existence, json-key and count checks are the only primitives this tool
offers. There is no free-form "python-eval this arbitrary claim" primitive
— every primitive is a fixed, reviewable operation with one ground-truth
source, so a marker is auditable by reading it, the same way `run.sh`'s
guard is auditable by reading its `cmp` exit code branches. What this trades
away, honestly: a marker has to be added by a human (or an audit round) who
read the claim and decided it was checkable. Nothing here retroactively
scans 74 documents' prose. See "What this cannot see" at the end.

## Where it runs

`rust/torch_c/pytests/run.sh` and `tools/golden/compare.py --self-test` set
the shape this follows: read ground truth from the live tree (never from a
hardcoded expectation baked into the checker), report `PASS`/`FAIL` per
category with counts, exit non-zero on any failure, and — the specific
lesson from `run.sh`'s own incident report in its own comments — distinguish
"this is wrong" from "I could not check this" rather than collapsing both to
a single failure.

`tools/docwatch/check_docs.py` is the entry point:

```
python3 tools/docwatch/check_docs.py [FILES...]
```

With no arguments it scans every `docs/*.md`. It needs the same environment
`rust/torch_c/pytests/decomp_sweep.py` documents needing — a built shim on
`PYTHONPATH`:

```
PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
    python3 tools/docwatch/check_docs.py
```

Static-only markers (`symbol-in-file`, `json-key`) work without that
environment too (`--no-live` skips anything that needs the shim or the
harness scripts, and reports what it skipped rather than silently passing).

## Primitives (final set — see "rejected primitives" below for what did not make it)

| Marker | Ground truth | What it catches |
|---|---|---|
| `op-implemented <op>` | `torch._C._aten_implemented()`, live | "X is implemented" / "X now works" |
| `op-not-implemented <op>` | same | "X refuses" / "X is missing" |
| `hasattr <module> <attr> true\|false` | live `hasattr()` | "no bare `torch.<name>` upstream" claims (ARCH20.md §9's exact mechanism) |
| `json-key <path> <key> present\|absent` | the JSON file itself | "`<name>` has/has no overloads.json entry" |
| `symbol-in-file <path> <name>` | regex over the file (`fn NAME`, `def NAME`, `class NAME`, `"NAME"` for JSON, else a plain substring) | "X exists in `path`" (BIND.md's `interned()` / `_install_tensor_dtype_identity` claims) |
| `count <name> <op> <value>` | one of a fixed registry of live harness runs (below) | "N of M", "K tests", "P/Q cases" |

`count`'s registry (deliberately small and named, not a free-form shell
command — a marker cannot ask the checker to run anything other than these):

| name | source |
|---|---|
| `smoke_ok` | `test_shim.py`'s own count of `ok ` lines |
| `golden_cases_passed`, `golden_cases_total`, `golden_ops_covered`, `golden_pending` | `tools/golden/compare.py`'s `SUMMARY:` line |
| `schema_entries_matched`, `schema_entries_total` | `verify_schemas.py`'s `SUMMARY:` line |
| `decomp_implemented`, `decomp_population`, `decomp_lowered` | `decomp_sweep.py`'s summary line + verdict tally |

`op` is one of `eq`, `ge`, `le` — `ge`/`le` exist because AUDIT.md's own
findings distinguish "this exact number" from "at least this many", and a
count marker for "148 ops implemented, and growth is expected" should use
`ge` so a *later* correct growth does not itself trip the checker; a marker
for "these are the walls left, N of them" should use `eq` because a change
in either direction is news.

## Rejected primitives

- **A general `path:line` exact-line check.** AUDIT.md's own method used
  `path:line` in its findings (`test_shim.py:7625`), but a line number is
  invalidated by any unrelated edit above it in the same file — it would cry
  wolf constantly, which is exactly the failure mode this document opens by
  rejecting. `symbol-in-file` checks the same underlying claim (the named
  thing still exists in that file) without depending on which line it is on.
  The exact line is reported when found, as a courtesy, but never asserted.
- **A free-form python-eval marker.** Considered for flexibility (arbitrary
  expressions like AUDIT.md's live one-off checks). Rejected: every primitive
  above is independently readable from its marker text alone, the same way
  reading `run.sh` tells you exactly what it checks; an eval'd string does
  not have that property, and it would make the checker as hard to audit as
  the prose it replaces.
- **Auto-detecting "blocked on" / "waiting for" language.** These turned out
  not to need a fifth primitive: every "blocked on" claim AUDIT.md found
  reduces to an op-implemented or symbol-in-file question once you ask what,
  specifically, would have to become true to unblock it (DESIGN.md's
  `from_pretrained` block was `hasattr`-shaped once traced to
  `torch._C._is_autocast_available`).

## What this run demonstrates

Recorded as each step actually ran, not reconstructed after.

### Finding on the current tree, before any marker was added

The very first live-count check this tool ran was against `docs/verification/AUDIT.md`'s
own baseline paragraph — the numbers that paragraph says were established
"before touching any file" for the audit round that produced it (268 `ok`
lines, 5634/5634 golden cases, ops covered=148, 4392/4392 schema entries).
Re-running the exact same commands today (before I had added a single
marker) gave 274 / 6374/6374 ops=161 / 4458/4458. The gap is
`cb6780d`/`f596426` ("Feat/Docs: Twenty-six of twenty-six" — the KERNELS26
round), which landed *after* `docs/verification/AUDIT.md`'s last commit and moved every
gate count the audit used as its ground truth. **The document whose whole
thesis is "a later commit closes a gap and nobody updates the document that
named it" was already, itself, an instance of exactly that** — this is not
a hypothetical the design needed to manufacture; it was sitting in the tree
before this task started, one commit later than the audit that should have
caught it. That is the finding the task brief predicted ("if it finds
nothing, that is suspicious") — it did not find nothing. A `>
**Correction (docs/verification/DOCWATCH.md, 2026-09):**` block was added to
`docs/verification/AUDIT.md` at the same spot, with `ge`-comparison markers on the
corrected numbers (so the *next* growth round does not repeat the same
failure against this paragraph).

### Coverage after instrumenting the audited files

39 markers landed across 9 of the 11 audited files (`ARCH20.md`,
`AUDIT.md`, `BIND.md`, `CAPTURE.md`, `DECOMP.md`, `DESIGN.md`, `SEQLEN.md`,
`SPELLINGS.md`, `TORCHSCRIPT.md`, `VIEWS.md`), all evaluating `PASS` against
today's tree:

```
43 marker(s) checked: PASS=43
DOCWATCH: PASS -- 43/43 evaluated marker(s) hold
```

(43, not 39 — four more were added afterward for `DESIGN.md`'s two
remaining §9 import-wall claims, once `symbol-in-file` proved able to check
them too.) `--no-live` alone (no shim, no torch) still evaluates the 14
static (`symbol-in-file`/`json-key`) markers and correctly skips the rest
rather than silently reporting them green:

```
43 marker(s) checked: PASS=14, SKIP=29
DOCWATCH: PASS -- 14/14 evaluated marker(s) hold, 29 skipped
```

By kind: `count`=11, `hasattr`=7, `json-key`=6, `op-implemented`=10,
`op-not-implemented`=1, `symbol-in-file`=8.

Two of DECOMP.md's and CAPTURE.md's markers found a **second** instance of
the same staleness mechanism while wiring them up, independent of the
AUDIT.md baseline finding above: `docs/verification/AUDIT.md`'s own correction to
DECOMP.md (45 population / 11 lowered, replacing the original 37/9) was
*also* already stale by the time this task started, for the identical
reason — the same KERNELS26 commit. Today's number is 50/12. A nested
`> >`-style re-correction was added (matching `docs/design/DESIGN.md`'s own
nesting convention for a second correction on top of a first), with `ge`
markers this time specifically so a third round of kernel growth does not
require a third manual correction — the check catches it the moment growth
happens, run.sh moving is enough to make the next `check_docs.py` run go
red.

### Deliberate-fault demonstration

`docs/bindings/SPELLINGS.md` was backed up with `cp` (not staged in git), and two
false markers were inserted next to its real, already-verified claims: a
"not implemented" claim for an op this tree does implement
(`aten.clamp.default`), and a golden-ops-covered count off by one (160
instead of 161):

```
docs/bindings/SPELLINGS.md:780: FAIL  op-not-implemented aten.clamp.default  -- aten.clamp.default IS in _aten_implemented()
docs/bindings/SPELLINGS.md:781: FAIL  count golden_ops_covered eq 160  -- golden_ops_covered = 161 (claim: eq 160)

11 marker(s) checked: FAIL=2, PASS=9
```

Exit code 1. Both caught by file, line, and the exact mismatch. The file
was then restored from the `cp` backup and `diff`-verified identical to the
pre-fault version.

### Historical-staleness demonstration

`git show 36d3a2d:docs/DECOMP.md` (the commit immediately before
`docs/verification/AUDIT.md`'s correction landed, `23c7097`) has, unhedged and in the
present tense: "**37 개 중 9 개**" (37 population, 9 lowered) — no
round-scoping language, the exact shape of claim `docs/verification/AUDIT.md` later
found false. That historical text was extracted into a scratch file with
two markers instantiating exactly that claim (`count decomp_population eq
37`, `count decomp_lowered eq 9`) and run against **today's** tree (no
special historical build needed — the claim is about a count, and the
checker's job is exactly "does this count still hold"):

```
count decomp_population eq 37  -- decomp_population = 50 (claim: eq 37)
count decomp_lowered eq 9  -- decomp_lowered = 12 (claim: eq 9)
```

Both fail. Had this marker existed in `docs/graph/DECOMP.md` on 2026-08-30, the
first `decomp_sweep.py`-moving commit after it (kernel work landing on top,
unrelated to DECOMP.md itself) would have turned this check red immediately
— not days or weeks later when a manual audit happened to read that
paragraph again. This is the mechanism `docs/verification/AUDIT.md` names as the one
that recurred nearly every time; this is that same mechanism caught at the
moment it would have first fired, not after the fact.

### What this cannot see (fraction, honestly)

`docs/verification/AUDIT.md` records roughly 30 distinct findings (claim + status +
how-checked) across its eleven files. This tool's markers now stand in for
about 15 of them directly — the ones that reduce to an op's presence in
`_aten_implemented()`, an upstream `hasattr`, a JSON table key, a symbol's
existence in a source file, or a harness's own summary count. That is
roughly half, and it is not a representative half — it is specifically the
half AUDIT.md's own account says did the most damage in aggregate count
(the six-of-eleven staleness pattern is dominated by exactly these
existence/count claims), but *not* the single highest-severity one
(DESIGN.md's `from_pretrained` claim was the worst finding in the whole
pass, and it is structurally outside what any of these six primitives can
express).

What is structurally invisible to this design, by category:

- **End-to-end behavioral claims** ("a real Hub checkpoint loads and
  generates tokens", "`torch.compile` reaches wall X not wall Y"). These
  need to actually run a multi-step pipeline and inspect what it produced,
  not check that a symbol or op exists. `DESIGN.md`'s `from_pretrained`
  finding and `DYNAMO.md`'s wall-identity finding are both this shape.
  Expressible in principle with a seventh primitive (run a named smoke
  scenario, assert on its outcome), deliberately not added here: unlike the
  six primitives above, "did this scenario produce the right answer" is not
  one ground-truth source, it is a bespoke script per claim, and a checker
  whose primitives are one-script-per-marker is not meaningfully narrower
  than not having markers at all.
- **Numeric/behavioral divergences from upstream** (`VIEWS.md`'s
  write-through-view semantics, the partial-overlap `copy_` divergence,
  the `slice.Tensor` step>1 write-loss). These are "does the computed
  *value* match a specific expectation", not "does this thing exist" — the
  existence half of these claims (`aten.ge.Tensor`/`aten.index_put_.default`
  now have kernels) is covered; the behavioral half is not.
- **Prose reasoning and design arguments** (the A-vs-B tensor engine
  decision, the TTL/TTA/TTT taxonomy) — `docs/verification/AUDIT.md` itself declined to
  check these, on the same grounds this document opens with: they are not
  checkable facts.
- **Cross-document narrative claims** ("a wall in document A is superseded
  by a fix in document B") where the fix is a *default flipping*
  (`DYNAMO.md`/`TORCHSCRIPT.md`'s `PYTORCH_JIT` default) rather than a
  symbol or count changing. `TORCHSCRIPT.md`'s own `tril`-landing claim
  *is* covered (that half is an op-existence claim); the "which wall does
  `torch.compile` hit now" half is not.
- **`docs/platform/WHEEL.md`'s self-test counts** (`build.py --self-test` and
  siblings). These fit the `count` primitive's shape exactly, but the
  scripts that produce them live under `tools/wheel/`, outside this task's
  territory — extending the registry to them is a natural, low-risk
  follow-up, deliberately not done here.
- **74 documents total, 11 audited.** The other 63 were not read this round
  (by `docs/verification/AUDIT.md`'s own account) and carry zero markers by construction
  — this tool only checks what someone read and tagged, and nobody has
  read those 63 yet.

The honest framing, in the terms the task brief itself offered: this is
thirty-ish claims checked correctly, not three hundred checked
approximately. The four that fired for real (the AUDIT.md baseline, the
DECOMP.md/CAPTURE.md re-staleness, the deliberate fault, the historical
replay) are not synthetic — three of the four were sitting in the tree
before this task touched anything.
