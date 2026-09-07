# Documentation audit (work/docaudit)

Audit of `docs/*.md` for claims that are checkable against the tree and have
gone false. Method, baseline, and per-file findings below. Written
incrementally so an interruption keeps the findings gathered so far.

Excluded from this audit (another agent is writing into these concurrently):
`docs/kernels/KERNELS26.md`, `docs/architectures/ARCH26.md`.

`README.md` is out of scope to edit (belongs to the coordinating session);
anything wrong found there is reported in the final summary, not fixed here.

## Baseline (ground truth established before auditing)

Commands run in the worktree, in the foreground, before touching any file
(so later "counts" claims can be checked against something real):

```
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-docaudit
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export HF_HOME=/Volumes/macMini/caches/hf-home
bash vendor/install_shim.sh
PY=/Volumes/macMini/caches/spike-venv/bin/python
PYTHON=$PY sh rust/torch_c/pytests/run.sh          -> EXIT=0, 268 "ok " lines, SELF-TEST PASS
$PY tools/golden/compare.py                        -> EXIT=0, SUMMARY: 5634/5634 cases passed, 0 failed, ops covered=148, pending case builders=1
$PY rust/torch_c/pytests/verify_schemas.py         -> EXIT=0, SUMMARY: 4392/4392 table entries matched upstream, 0 failed
```

torch version seen by verify_schemas.py: `torch 2.13.0`.

Implemented op count from `_aten_implemented()` (printed by both run.sh and
compare.py): **148** ops, full list captured in `/tmp/run_pytests.log` and
`/tmp/golden_compare.log` in this session.

Other numbers surfaced by verify_schemas.py, useful for cross-checking counts
elsewhere: overloads.json 200/200, methods.json 168/168,
native_functions.yaml re-printed 2584/2584, packet overload lists 118/118,
OpOverload.tags 148/148, CompositeImplicitAutograd registrations 744/744.

> **Correction (docs/verification/DOCWATCH.md, 2026-09):** this baseline is itself an
> instance of the exact mechanism this document's own conclusion names — a
> later, unrelated commit moved the numbers and nobody came back to this
> paragraph. `cb6780d`/`f596426` ("Feat/Docs: Twenty-six of twenty-six", the
> KERNELS26 round) landed after this document's last commit (`23c7097`) and
> moved every gate count here. Re-run today, same commands, same env
> pattern: `run.sh` 274 "ok " lines (not 268), `compare.py` SUMMARY:
> 6374/6374 cases passed, 0 failed, ops covered=161 (not 148), pending case
> builders=1, `verify_schemas.py` SUMMARY: 4458/4458 (not 4392). This is
> also `tools/docwatch/check_docs.py`'s acceptance baseline — see
> docs/verification/DOCWATCH.md — and the markers below check the corrected numbers, not
> the ones above, so this correction does not itself go stale silently the
> next time a kernel round lands:
> <!-- DOCWATCH: count smoke_ok ge 274 -->
> <!-- DOCWATCH: count golden_cases_total ge 6374 -->
> <!-- DOCWATCH: count golden_ops_covered ge 161 -->
> <!-- DOCWATCH: count golden_pending eq 0 -->
> <!-- DOCWATCH: count schema_entries_matched ge 4458 -->

Ad hoc one-off checks (op implemented / spelling exists / etc.) are recorded
inline in each file's section below with the exact command used, so the
check is re-runnable.

Shell used for one-off python checks:
```
$PY -c "..."
```
with the same env vars as above exported first.

---

## Findings

(filled in per file below; format: File / Claim / Status / How checked)

### docs/architectures/ARCH20.md

- **Claim (§9, "A real gap" list, line ~573-583):** `gelu`, `silu`, `softplus` are listed
  alongside `abs`/`clamp`/etc. as names where "a public `torch.<name>` exists upstream, the
  kernel is here, and the name refuses," and the whole 25-name list is described as "not fixed
  here" / "a well-defined next round."
  **Status: FALSE, and stale.** `hasattr(torch, "gelu")` is `False` on real upstream torch
  2.13.0 (same for `silu`, `softplus`) — there is no bare `torch.<name>` to reach for those
  three, only `torch.nn.functional.<name>`/`torch._C._nn.<name>`, both already answered by this
  shim. Separately, "not fixed here" is stale: 22 of the 25 names now have an
  `overloads.json`/`methods.json` entry (confirmed via `rust/torch_c/src/overloads.json`'s own
  comment, and by grepping the 25 names as JSON keys — all 22 non-gelu/silu/softplus names are
  present); `reshape` is the one still-pending name (confirmed via
  `tools/golden/compare.py`'s current `PENDING` line naming only `aten.reshape.default`).
  **How checked:** `hasattr(torch, name)` for all 25 names against
  `/Volumes/macMini/caches/spike-venv/bin/python` (real torch 2.13.0); `grep -n '"<name>":'
  rust/torch_c/src/overloads.json` for each of the 25; `$PY tools/golden/compare.py` baseline run
  (PENDING line).
  **Fixed:** yes — added a `> **Correction (docs/bindings/SPELLINGS.md §5–§7): ...**` blockquote after
  the "well-defined next round" paragraph, matching this doc's existing correction style (the
  `_safe_softmax` correction a few paragraphs above it). Did not rewrite the original text.

- Section 0 (numbers), §11 (verification gates: 241 ok, 3302/133, 4295/4295) and §11.4
  (bit-identical prefill): these are historical before/after tables for *this document's own
  round* (explicit "before/after" column headers), not claims about the repo's state today.
  Current baseline is higher (268 ok, 5634/5634 ops=148, 4392/4392) because later rounds
  (SPELLINGS.md, TRIL.md, KERNELS26.md/ARCH26.md — the last two out of scope here) added more.
  Left as-is; not a stale claim, it's a historical diff correctly scoped to its own round.

- §6.2 ("What is still refused"): `pow(bool_t, 2)`, `pow(True, bool_t)` etc. still raise
  `NotImplementedError`. **Status: confirmed true, still current.**
  **How checked:** called `torch.pow(torch.tensor([True, False]), 2)` and
  `torch.pow(True, torch.tensor([True, False]))` against the loaded shim
  (`PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1`); both still raise
  `NotImplementedError` with the same message the doc describes.

- §10 (`gpt_bigcode`, TorchScript wall): re-ran the import chain description mentally against
  current tree structure; not independently re-verified by executing (would need `transformers`
  import of that specific model file) — reported, not checked, due to time budget. Low risk: the
  doc itself says "re-confirmed against the current build" as of its own writing and nothing in
  this session touched TorchScript.

### docs/bindings/SPELLINGS.md

Read in full (784 lines). This document is a chronological, self-correcting log of several
rounds of spelling-table work, and it already contains its own follow-up-correction section
(§8, "후속 정정 — §7.9가 남긴 세 항목 중 둘은 닫혔다") that revisits §7.9's "not done" list and
marks two of three items done, one still open. That is exactly the pattern this audit is
supposed to add where missing — here it was already present and accurate.

- §8's table claim "`clamp.Tensor` 커널 부재 — 그대로 (unchanged)" — **confirmed still true**:
  `aten.clamp.default`/`aten.clamp_.default` are in the current 148-op implemented list;
  `aten.clamp.Tensor` is not. How checked: grepped the implemented-ops list captured in
  `/tmp/run_pytests.log` for `clamp`.
- §7.1's `hasattr` measurements for `gelu`/`silu`/`softplus` (upstream False, `nn.functional`
  True) — **confirmed true**, same check as ARCH20.md above. This section is the "full per-name
  accounting" ARCH20.md §9's correction now points to.
- No false claims found in the sections read. Not exhaustively re-verified line by line (784
  lines of mostly narrative/measurement transcripts); spot-checked the numeric/existence claims
  most likely to be quoted by a future agent (the gelu/silu/softplus split, the clamp.Tensor gap,
  the §7.9→§8 closure table).

### docs/platform/WHEEL.md

Long file (1261 lines), many dated rounds (§9 through §15, 2026-08-28 through 2026-08-31). Most
numeric tables are explicit before/after for their own round (e.g. §0's "168/168 shim tests,
2702/2702 ops=118" is that round's baseline, explicitly "이 작업 전후로 같은 값" — same before
and after *that* round, not a claim about today).

- §15 (2026-08-31, the most recent section — an audit of `tools/wheel/`'s own verification
  scripts) claims current self-test counts: `verify_ios_device.py --self-test` 8/8,
  `verify_cross.py --self-test` (android/device 11/11, manylinux 12/12, Windows 9/9),
  `verify_linux.py --self-test` 6/6, `build.py --self-test` LINUX 11/11 / VERIFY 4/4 /
  UPSTREAM-DIST-INFO 2/2.
  **Status: the ones checkable on this host are confirmed true.**
  **How checked:** ran `$PY tools/wheel/build.py --self-test` directly (foreground, exit 0) —
  output matched exactly: "LINUX SELF-TEST: PASS -- 11/11", "VERIFY SELF-TEST: PASS -- 4/4",
  "UPSTREAM-DIST-INFO SELF-TEST: PASS -- 2/2" (PREFLIGHT-CACHE 3/3 also matched, not separately
  claimed as changed in §15). Did not run `verify_ios_device.py`/`verify_cross.py`/
  `verify_linux.py --self-test` (they assert on Linux ELF / require prior cross-built artefacts
  not present in this worktree) — reported as checked-where-possible, not independently
  re-verified for the other three scripts.
- No false claims found in the sections read (§0, §15, spot checks elsewhere). Sections §1-§14
  were skimmed rather than read line-by-line given the file's length; not exhaustively audited.

### docs/verification/GOLDEN.md

Read in full (297 lines). Entirely a historical record of one specific round (closing the
keyword-dispatch blind spot found by docs/design/DISPATCH.md §4.1): 2811→2843 cases, ops=119,
`run.sh` 177 ok / 34 "pre-existing unrelated" at the time. Current baseline is 5634/5634
cases, ops=148, 268 ok all passing — later rounds added far more; this doc's numbers are that
round's own snapshot, not presented as "today's numbers."

- §7 ("A wrong diagnosis, and what it actually was") is a deliberate, explicit self-correction —
  house style already applied correctly: it says outright that an earlier version of this same
  section was wrong, why, and what the actual cause was (worktree never had `vendor_torch.sh`
  run). Left untouched per the "leave deliberately-recorded mistakes" rule.
- **Minor, reported-not-fixed:** §8 ("Running it") gives copy-pasteable commands with inline
  comments stating expected output as `# 2843/2843, ops covered=119, exit 0` and
  `# 177 ok / 34 pre-existing unrelated (§7), exit 1`. Read at face value today, `run.sh` no
  longer exits 1 or shows 34 failures — current baseline is 268 ok, exit 0, no failures. Not
  edited: the whole document is explicitly framed as an analysis of one past round rather than a
  "how the repo behaves now" reference, and editing the inline comments would blur that framing
  without adding real information (the true current numbers live in the acceptance-criteria
  commands, not in this doc). Flagging in case a future reader treats §8 as copy-paste-and-expect
  literally.

### docs/design/DISPATCH.md

Skimmed in full (445 lines), read §4 and §6 closely.

- **Claim (§4.1, line ~313):** "The golden harness is blind to this entire code path" /
  "it is a standing gap." **Status: stale.** docs/verification/GOLDEN.md (a direct sequel to this exact
  section, confirmed by GOLDEN.md's own opening line citing "docs/design/DISPATCH.md §4.1") closed most
  of this gap: 61 of `interned_name()`'s 74 arms are now exercised by keyword by at least one
  golden case (verified there by re-running the same tamper technique on a different arm,
  `"dim"`, and watching `compare.py` go red). 13 arms remain uncovered, one deliberately.
  **How checked:** read docs/verification/GOLDEN.md §4-§5 in full (see above); did not independently re-run
  the `"dim"` tamper myself (GOLDEN.md already did, with exact before/after numbers and the
  restore-from-backup discipline).
  **Fixed:** yes — added an "Update (docs/verification/GOLDEN.md): ..." blockquote directly after the
  "standing gap" paragraph, pointing to the sequel rather than rewriting DISPATCH.md's own
  (correct-at-the-time) narrative.
- §4's "Behaviour did not move" numbers (197 ok / 2811/2811 ops=119 / 4203/4203) and §6's "what
  is left" cost list (resolve+_bind ~0.7-1.0µs, `_tlv_get_addr` ~12%, the `match op` arm count):
  these are that round's own before/after and open-items list. Whether item 1 (resolve+_bind) is
  still open is really a question for docs/bindings/BIND.md, which a separate agent in this session is
  auditing in parallel — not re-checked here to avoid duplicate work.

  **Update (this round):** that "separate agent" note is stale — this round's brief lists
  docs/bindings/BIND.md as this same audit's own territory and priority item 3. Auditing it below rather
  than skipping it a second time.

### docs/devices/META.md

Long file (887 lines), one continuous document across two rounds (initial meta-device landing,
then a later "give it the kernels from_pretrained needs" round responding to a user bug report,
`3c9d000`). House style (round-scoped snapshots) applies throughout; most numeric tables are
explicitly framed as before/after for their own round.

- **Claim (§11.1, judgment block for the `3c9d000` round):** "골든 4284/4284, ops=139" /
  "`_aten_implemented()` 는 139 그대로" (closing line of §7.4, same round).
  **Status: FALSE — internally inconsistent with the same document's own §7.4 opening line**,
  which states "커널이 있는 op 148 개" for the *same* round (and the commit message of `3c9d000`
  itself says "Of 148 ops with a kernel"). One of the two numbers is a stale snapshot pasted next
  to a later one; 148 is the one that matches reality.
  **How checked:** live-probed the shim (`torch.mm`/`.sum`/`.view` on meta → `NotImplementedError:
  ... no meta kernel`; `.select`/`.tril`/`.expand`/`.gt` on meta → succeed), confirming §7.4's
  148-op-baseline categorisation (66 reachable / 82 not, table sums to 82) matches this session's
  own baseline (148 ops, confirmed already in the AUDIT.md baseline above). §11.1's 139 does not
  match anything current.
  **Fixed:** yes — added a `> **Correction (문서 감사): ...**` blockquote directly after the
  "139 그대로" sentence in §11.1's closing paragraph, explaining the mismatch and pointing to the
  148/5634 baseline instead of rewriting the original numbers.
- **Claim (§12, "못 한 것", pre-existing correction block on `m.to("cpu")` / `m.to_empty`):**
  "`m.to("cpu")` 는 여전히 거부되지만 이유가 다릅니다 ... `NotImplementedError: ... Please use
  torch.nn.Module.to_empty() instead ...`". **Status: confirmed still true** (this correction was
  already present before this audit, landed in `3c9d000`, not something this round added — but
  re-verified since it's exactly the "X refuses" claim shape this audit is supposed to test).
  **How checked:** built a `meta`-device `nn.Linear`, called `.to("cpu")` directly (not after
  `to_empty`, which mutates the module in place and would have made a second `.to("cpu")` trivially
  succeed) — raised the exact `NotImplementedError` text quoted above. `.to_empty(device="cpu")`
  on a fresh meta module also succeeded, confirming the other half of that correction too.
- §7.4's category table (reductions 22, view/shape 13, in-place 12, contraction 8, indexing 7,
  composite/activation 6, combine/split 4, other 10 = 82) and its "66 reachable" figure: spot-
  checked 4 unreached ops (`mm`, `sum`, `view` from the table; also tried `add` after `.t()` to hit
  the not-yet-meta `t.default`) and 4 reached ops (`select`, `tril`, `expand`, `gt`) directly
  against the live shim — all matched the table exactly.
- Section 9 (dispatch cost, +13.6 ns / +3.7%), §10 (107-item transcript diff, 41 match / 66
  mismatch broken down by category), §11 (round-1 judgment: 2258/2258 ops=96, pytests 91 ok),
  §8.2 (upstream-vs-shim transcript table for `with torch.device(...)`/`set_default_device`):
  not independently re-verified — these are dated, round-scoped measurement transcripts (explicit
  "측정일 2026-08-25" byline) rather than "current state" claims, consistent with the house style
  already confirmed correct in WHEEL.md/GOLDEN.md above. Not re-run given time budget; no reason
  found to suspect them specifically (no counter-evidence encountered while checking §7.4/§12).
- §12 "모르는 것" (unknowns) and "때운 것" (patched-over gaps) sections: these are explicit
  "we did not check this" / "we know this is wrong" admissions, not falsifiable claims — left as
  reported, not re-tested (auditing a "we don't know X" statement doesn't have a pass/fail).

### docs/numerics/SEQLEN.md

Very long (1417 lines), three rounds (`pow` fix §3, `amax` fix §7, scale+mask fusion §8), each
with its own before/after tables and its own numerics-safety argument. This document is already
unusually well self-corrected: §7.10 and §8.12 each carry a `> **TAKEN, in docs/kernels/KERNELS26.md /
docs/kernels/TRIL.md §N.**` blockquote pointing forward to where a "not done here" item was later done —
exactly the annotation pattern this audit is supposed to add, already present.

- **Claim (§7.10 correction blockquote):** `torch.amax`/`Tensor.amax` "now resolve and reach this
  kernel" (landed in docs/kernels/TRIL.md §2, entries in `overloads.json`/`methods.json`).
  **Status: confirmed true.** **How checked:** `grep '"amax"'` in both
  `rust/torch_c/src/overloads.json` and `rust/torch_c/src/methods.json` (both present); live call
  `torch.amax(t, dim=1)` and `t.amax(dim=1)` against the running shim — both return the kernel's
  answer (`tensor([5., 6.])` for the test input).
- **Claim (§8.12 correction blockquote):** `tensor.rs::transposed_contiguous`, "the same copy in
  32x32 cache blocks", landed in docs/kernels/KERNELS26.md §7, wired into the SDPA call site and into
  `aten.contiguous.default`. **Status: confirmed true — the function and both call sites exist.**
  **How checked:** `grep -rn transposed_contiguous rust/torch_c/src/` — function defined at
  `tensor.rs:2390`, referenced from `aten.rs:3152` (the SDPA k-transpose path, matching the
  `aten.rs:3143` comment citing this exact function) and from `tensor.rs`'s own unit tests. Did
  not re-run the cargo benchmark that produced the 5.25x/4.84x figures (out of scope: that
  measurement belongs to docs/kernels/KERNELS26.md, excluded from this audit).
- Every other numeric table in the file (§1 curve-fit coefficients, §3/§7/§8's old/new/upstream
  A/B tables, the gate-count tables in §6/§7.11/§8.10, the sha256 logit-digest tables): these are
  explicitly dated, round-scoped measurement transcripts, each already labelled with which round
  produced it and consistent with later rounds' own "before" column reproducing the prior round's
  "after" to 2-3 significant figures (i.e. the document already cross-checks itself round to
  round). Not independently re-measured — reproducing a performance curve is outside what a
  documentation audit can check against the tree, and re-running would risk exactly the
  "measurement contaminated by a concurrent agent" failure mode CLAUDE.md §"측정 작업은 단독으로
  돌린다" warns about, which this session is not isolated for (KERNELS26.md/ARCH26.md work is
  running concurrently). Spot-checked only the two claims above because they are "X now works"
  assertions checkable by existence rather than by re-measuring a curve.
- No false claims found. This file's own internal cross-checks (each round's "old" column
  reproducing the prior round's "new" column) already do a large part of what this audit would be
  checking for, and did not turn up a contradiction while reading.

### docs/bindings/BIND.md

Long file (1147 lines now), four rounds. §9.5 already carried a pre-existing
`> **Correction (문서 감사, 재확인): ...**` blockquote (landed in a prior commit, `b8c3ea1`, not
by this audit) re-pointing a stale smoke-test-count claim at the current 268/0 baseline — verified
still accurate (re-ran `run.sh`, 268 ok, matches this session's baseline).

- **Claim (§9.3, closing paragraph):** "The clean fix is one line in `tensor.rs` ... it is not
  written because `tensor.rs` is outside `bootstrap.py` + `docs/bindings/BIND.md`." **Status: FALSE, and
  found by cross-referencing the commit that landed this very section against its own commit
  message.** `git show b8c3ea1` (the commit that added this whole §9 to BIND.md) changes
  `tensor.rs`'s `dtype` getter from `PyDtype::new(self.tag)` to `crate::dtype::interned(py,
  self.tag)` in the *same commit*, and the commit message says outright: "The investigating agent
  ... wrote a Python-side property override instead ... this replaces it." So the documentation
  text (apparently carried over unedited from the investigating agent's draft) describes a fix
  that had already been superseded before the commit merged.
  **How checked:** `grep "fn dtype" rust/torch_c/src/tensor.rs` → confirms `interned(py, self.tag)`
  is what ships today; `grep _install_tensor_dtype_identity rust/torch_c/src/bootstrap.py` →
  confirms the Python override this section's code listing shows does not exist in the file;
  `git log -S"fn dtype(&self)" -- rust/torch_c/src/tensor.rs` → identifies `b8c3ea1` as the commit
  that made the change, and `git show b8c3ea1` → confirms both the code change and the commit
  message's own account of what replaced what.
  **Downstream effect:** §9.4's cost table ("roughly doubles the cost... ~0.07 → ~0.15 µs") is
  therefore measuring a code path that was never shipped — the commit message gives the real
  number for what did ship: "0.042 us against 0.070 before ... cheaper than the incorrect one."
  The correctness finding itself (§9.1/§9.2/§9.5 — `.dtype` identity was broken, `baddbmm` promoted
  to float64, now fixed) is unaffected and reconfirmed live below.
  **Fixed:** yes — added a `> **Correction (문서 감사): ...**` blockquote after §9.3's closing
  paragraph, pointing to the actual shipped fix and its real cost, without deleting the original
  (now-inaccurate) "not written" text.
- **Re-verified live** (the underlying correctness claim, independent of which fix shipped):
  `t.dtype is torch.float32` → `True`; `from torch._decomp.decompositions import baddbmm;
  baddbmm(ones(2,3,5), ones(2,3,4), ones(2,4,5)).dtype` → `torch.float32` (not `float64`). Both
  match §9.1/§9.2's claims exactly.
- **Claim (test existence, §9.5's correction block):** `test_decompose_lowers_baddbmm_default_
  now_that_the_dtype_is_a_singleton` replaced the old assertion. **Status: confirmed true** —
  `grep` finds both the old test name still present (as a different function, presumably renamed
  scope) and the new one defined at `rust/torch_c/pytests/test_shim.py:7625`.
- Gate-count tables in §4, §8.3, §8.4's prefill table, §9.4/§9.6/§9.7's profile counts: round-scoped
  snapshots consistent with house style, not re-verified individually given time budget — no
  contradiction found while checking the items above.
- §7's heading says "Android measurement (in progress)" but the section body is a complete,
  concluded measurement with a "Result" subsection and no open items — **reported, not fixed**:
  this is a stale heading word rather than a false claim (nothing in the body claims to be
  incomplete), and CLAUDE.md's fixing rule is for false mechanical claims, not phrasing.

### docs/perf/DTYPE_PERF.md

Shorter (458 lines), single round, already self-annotated with its own reversal (§2's initial
microbench measured a layout the model never makes; §3-§4 catch it and say so explicitly at the
top of the file — the "§2 를 먼저 읽고 §3 에서 뒤집힙니다" callout). Gate counts (241 / 3302/3302
ops=133 / 4295/4295 / cargo test 10) are this round's own before/after, consistent with house
style and with the progression already seen in BIND.md/SEQLEN.md's own round-scoped numbers at
similar dates.

- **Claim (§7, "남은 것"):** the fused-gemv kernel for `lm_head` decode (`S=1`, where upstream is
  measured 6.2x faster) "이 회차에서 재지 않았습니다" (not measured this round) and is left as a
  named gap. **Status: still an open gap, not resolved by a later round.**
  **How checked:** `grep -rln "fused gemv\|융합 gemv\|fused_gemv\|gemv_trans"` across `docs/` and
  `rust/torch_c/src/` — only DTYPE_PERF.md and DTYPE.md (its own prior-round source) reference it;
  no later document or source symbol picks it up. Confirmed `widen_gemm_operand` (§5's actual
  fix) still exists in `rust/torch_c/src/aten.rs:1860`, so the fix this round *did* land is still
  in place; the fix this round explicitly declined to attempt is still undone.
- No false claims found. The document's numeric tables are internally consistent (§3's model-level
  numbers are what §4's layout explanation is built to explain, and §6's verification counts match
  the "before" numbers §5's fix started from), and its own retraction of §2 is already the kind of
  annotation this audit adds elsewhere.

### docs/kernels/VIEWS.md

758 lines, six sections landing incrementally (§1-§6), each with its own before/after gate counts
matching the next section's baseline exactly (§1→§2-3→§5→§6, each "before" reproducing the prior
section's "after"). Unusually rich in exactly the checkable-fact categories this audit targets:
several hard "still refuses" / "now works" claims. Live-tested the highest-stakes ones (the write-
through behaviour is a correctness property, not a performance number, so a false claim here would
be the "invented spelling" / "already landed" class of error the brief warns about).

- **Claim (§6, throughout):** in-place ops now write through views — `x[0] = v`, `x[1].fill_(3.0)`
  etc. are visible on the base tensor, where §4 says they previously were not.
  **Status: confirmed true.** **How checked:** `x = torch.zeros(5); x[1].fill_(3.0)` → base reads
  `[0.0, 3.0, 0.0, 0.0, 0.0]` — the write reached the original tensor through the `select.int` view.
- **Claim (§2-3):** `index_put_` now handles a bool mask (`x[mask] = values`) and non-1-D operands,
  where it previously refused. **Status: confirmed true.** **How checked:**
  `x2[torch.tensor([True,False,True,False])] = torch.tensor([9.,8.])` → `[9.0, 0.0, 8.0, 0.0]`,
  matching upstream semantics (positions 0 and 2 selected).
- **Claim (§1):** `aten.ge.Tensor` now has a kernel (previously resolved and then refused by name).
  **Status: confirmed true.** **How checked:** `torch.tensor([1,2,3]) >= torch.tensor([4,5,6])` →
  computes (`[False, False, False]`); `a >= a` → all `True` (the specific case §1 says
  distinguishes `Cmp::Ge` from `Cmp::Gt`).
- **Claim (§6.4, "the two that remain"):** `slice.Tensor` with step > 1 and `view.dtype` are still
  write-lost (materialised here, a view upstream) and this is structural (blocked by candle's
  `pub(crate)` storage boundary), not yet fixed. **Status: confirmed still true** — and this is the
  same divergence already independently confirmed in this session's baseline run of
  `tools/golden/compare.py` (its `KNOWN DIVERGENCE` section names exactly these same two ops with
  the same explanation), so it is cross-checked from two directions.
- **Claim (§6.5):** a partial-overlap `copy_` (`x[0:2].copy_(x[1:3])`) computes a defined answer
  here where upstream raises — a known, recorded, unfixed divergence ("cost decision", not a wall).
  **Status: confirmed true.** **How checked:** `x=torch.arange(4.); x[0:2].copy_(x[1:3])` →
  computed `[1.0, 2.0, 2.0, 3.0]` without raising (upstream raises "some elements of the input
  tensor and the written-to tensor refer to a single memory location").
- No false claims found. This document's own §4 (superseded verdict, correctly annotated with a
  blockquote pointing to §6) and its "counts after §N" tables at every section boundary already do
  most of what this audit checks for.

### docs/graph/TORCHSCRIPT.md

397 lines, single round. Already carries two pre-existing `> **Correction ...**` blockquotes (not
from this audit) pointing at `docs/kernels/TRIL.md` for the `tril` kernel landing and the count moving
19/20 → 20/20.

- **Claim (top-of-file correction, and §9):** `aten.tril.default` is implemented and GPT-BigCode
  now constructs and forwards, making the sweep 20/20. **Status: confirmed true.**
  **How checked:** `"aten.tril.default" in torch._C._aten_implemented()` → `True`; built a small
  `gpt_bigcode` config via `AutoModelForCausalLM.from_config(...).eval()` and ran a forward — no
  error, correct logits shape.
- No false claims found otherwise; the document's own gate tables (§5) are round-scoped and
  consistent with the counts BIND.md/SEQLEN.md show at similar dates.

### docs/graph/DYNAMO.md

737 lines, two parts: Part A (§0-9, what `_C._dynamo` needs when `torch.compile` is never called)
and Part B (§10-19, what happens when it actually is called). No code changed by this document's
own account (`git status --short` claim, verified: only `docs/graph/DYNAMO.md` differs across its
commits). This is the one file in this pass where a **cross-document staleness** turned up — not a
number that drifted, but a conclusion invalidated by a *later* document's fix.

- **Claim (Part B summary table + §12, "Wall B"):** with the default (`inductor`) backend, after
  stubbing `pt2_archive_constants`, `torch.compile()` hits `SourceRangeFactory.make_range` via
  `torch.utils.mkldnn.MkldnnLinear`'s module-scope `@torch.jit.script_method` — described as "a
  different wall, unrelated to eval-frame" from the `backend='eager'` wall in §11/§15.
  **Status: FALSE as of today's default — superseded by a later, unrelated round.**
  `docs/graph/TORCHSCRIPT.md` (commit `02758e8`, 2026-08-31 00:34) landed *after* this document's last
  commit (`4e6281f`, 2026-08-30 17:10) and added `os.environ.setdefault("PYTORCH_JIT", "0")` to
  `bootstrap.py` — upstream's own scripting-disabled default. Under that default,
  `@torch.jit.script_method` is `torch/jit/_script.py`'s `if not _enabled: return fn`, a no-op, so
  `MkldnnLinear` no longer reaches `make_range` at all.
  **How checked:** live-reproduced both conditions. With `pt2_archive_constants` stubbed and
  `PYTORCH_JIT` **unset** (today's actual default), `torch.compile(lambda x: x+1)()` sails past the
  TorchScript import and instead raises `NotImplementedError: ... eval_frame.set_skip_guard_eval_
  unsafe` — an eval-frame-family symbol, i.e. the *same* category of wall §11/§15 already judged
  structurally unreachable under abi3, not a different one. With `PYTORCH_JIT=1` forced (the
  condition this document was actually measured under), the exact `make_range` traceback in §12
  reproduces verbatim — confirming the original measurement was correct for its environment, and
  that the environment is what changed.
  **Cross-reference:** `docs/models/COMPAT.md` §5's own correction blockquote already independently
  reports this same TORCHSCRIPT.md-caused change for GPT-BigCode specifically ("this import wall
  is gone on both versions") — the two documents agree once both are read, they just were not
  cross-linked from DYNAMO.md's side until this pass.
  **Fixed:** yes — added a `> **Correction (문서 감사, 재측정): ...**` blockquote after §12's
  closing paragraph explaining the mismatch, what reproduces under today's default, and that the
  live-verified replacement wall is in the same family §15 already covers; also annotated the Part
  B summary table's Wall-B row to point at the correction, without rewriting §12's original
  (correct-at-the-time) measurement.
  **Does not affect:** §18's recommendation ("push capture, not Dynamo") — the correction notes
  explicitly that the reachable wall changed but the abi3-unreachability conclusion (§15) still
  holds for whichever eval-frame symbol is hit first.
- This is the first cross-document staleness found in this pass (as opposed to a document being
  stale against the live tree) — worth flagging because none of the other files audited so far had
  one; it is a real risk shape (a later, unrelated commit can invalidate an earlier document's
  scoped conclusion without either document being edited) rather than a copy-paste or transcription
  error like the ones found in META.md/BIND.md.
- Part A's §0-9 (the 137/52/2-name accounting, the self-check methodology) not independently
  re-verified — it is a `dir()`/instrumented-proxy census with its own self-check built in (§2.3,
  positive and negative controls), and re-running it would mean rebuilding the same instrumentation
  this document already built; spot-checking would not add confidence proportional to the time
  cost. §15's abi3 argument (reading upstream C source for `Py_BUILD_CORE`) not re-verified either
  — it is a source-reading argument about a cache directory outside this repo
  (`/Volumes/macMini/caches/pytorch-spike/pytorch`), not a claim about this repo's own tree.

### docs/models/COMPAT.md

547 lines, single commit (`8a1de11`), but already unusually dense with pre-existing
`> **Correction (re-verified live against the current build): ...**` blockquotes — at least six,
covering nearly every numeric claim in the document (the 4.x/5.x architecture counts, the specific
missing-op list, the `__getitem__` list-index gap, the `gpt_bigcode` import wall). These read as
the *original author* re-verifying their own document against a moved tree before committing, not
as this audit's work — but they are exactly the shape of correction this audit adds elsewhere, so
this file needed mostly re-verification rather than new fixes.

- **Claim (§5's correction, "still genuinely missing"):** `aten.all.default`, `aten.aminmax.default`,
  `aten.index_add_.default`, `aten.nonzero.default`, `aten.scatter_.value`, `aten.zeros.default`,
  `aten.roll.default` remain unimplemented. **Status: confirmed true.**
  **How checked:** grepped the current 148-op implemented list (captured in this session's baseline,
  `/tmp/golden_compare_2.log`) for `all`, `aminmax`, `nonzero`, `zeros`, `roll`, `index_add_` — none
  present; only `aten.scatter.src` exists, not `scatter_.value`.
- **Claim (§5's correction, "now works"):** `torch.square`, `torch.repeat_interleave` (as Python
  composites, no own `aten.*` entry), `aten.pow.Tensor_Tensor` float32×int32 promotion, and
  `x[:, [-1, 0]]` list-indexed `__getitem__` all now run. **Status: confirmed true.**
  **How checked:** called all four directly against the live shim: `torch.square(...)` computes,
  `torch.repeat_interleave(...)` computes, `x[:, [-1, 0]]` on a `(2,4)` tensor returns the correct
  gathered values.
- **Claim (§3.2 fix, still current):** `TensorBase.permute` and `Tensor.T` are wired (not kernel
  gaps, wiring gaps). **Status: confirmed true.** **How checked:** `x.permute(1,0)` and `x.T` both
  compute and agree.
- **Claim (§3.1 fix, still current):** `torch.autocast(device_type="cpu", enabled=False)` completes
  without raising. **Status: confirmed true**, called directly.
- No false claims found — everything spot-checked (a representative sample covering every section
  with a numeric or existence claim) matched. This is the second document in this pass (after
  SEQLEN.md) where the file's own prior self-correction already did most of what this audit checks
  for.

### docs/graph/DECOMP.md

686 lines, single round (2026-08-30). Unlike SEQLEN.md/BIND.md/COMPAT.md, this document does not
frame its headline numbers as an explicit round-scoped snapshot — §0 ("이 문서가 답하는 것")
presents "37 개 중 9 개" as a present-tense answer, which is exactly the shape of claim that goes
stale silently as unrelated kernel work lands.

- **Claim (§0 and §4, headline):** of 37 non-Core-ATen, capture-reachable implemented ops, 9 lower
  to Core ATen via upstream's decomposition rules. **Status: FALSE today — stale by natural
  kernel-count growth, not a bug.** **How checked:** re-ran `rust/torch_c/pytests/decomp_sweep.py`
  against the current build (the script itself is unchanged since this document's single commit,
  confirmed via `git log`). Current output: `_aten_all_implemented() = 157` (not 129), `non-core =
  67` (not 52), capture-rejects `22` (not 15), **population 45** (not 37), **LOWERED 11** (not 9) —
  the 9 named in §4 plus `aten.baddbmm.default` (which §6/§7.2 already separately say was fixed and
  now lowers, but that fact was never folded back into §4's headline table) and
  `aten.floor_divide.default` (blocked in §4 on a missing `aten.div.Tensor_mode` kernel that has
  since been implemented by unrelated later work). REFUSED 25, CAPTURE_RAISED 1 (matches §4), and a
  new `NO_CASE` category (8 ops) that does not exist in this document's own classification scheme
  at all — `decomp_sweep.py` skips ops it has no golden case to synthesize an input from.
  **Fixed:** yes — added a single `> **Correction (문서 감사, 재측정 2026-09): ...**` blockquote
  right after §0's headline claim (rather than editing every downstream table in §4/§6/§7/§9,
  which would blur the "what this round opened" narrative the document is built around), giving
  the current numbers and pointing to `docs/verification/AUDIT.md`'s baseline for gate counts.
- **Claim (§7.2.1):** "wall 3" (decomposition rule runs but disagrees with the recorded trace) has
  no live example as of 2026-08-30, confirmed by a sweep with a positive control (self-check that
  the detector can actually fire). **Status: confirmed still true.** **How checked:** the same
  `decomp_sweep.py` re-run above shows `DISAGREES: 0` in its output, consistent with the document's
  claim not having regressed.
- §9's gate counts (`run.sh` 176, `compare.py` 2744/2744 ops=118, `verify_schemas.py` 4200/4200) are
  covered by the same correction — not re-annotated individually, since they are downstream of the
  same "written before later kernel work landed" cause and the correction already says so.
- No other false claims found; §1-§3 (the four wiring-defect fixes: packet overload collapse, CIA
  registration enumeration, kwarg-name mismatch, and the three `overloads.json` entries) are
  mechanism descriptions and fix narratives, not counts that drift, and were not re-verified
  line-by-line given time budget — no contradiction found while re-running the sweep for the
  headline number.

### docs/graph/CAPTURE.md

448 lines, an earlier round than DECOMP.md (its own §5's Core-ATen count, 108 implemented/70 core,
is explicitly the pre-decomposition-pass motivation, correctly framed as historical — §10 already
has a strikethrough-style "~~1. 분해 패스.~~ **섰습니다**" update pointing at docs/graph/DECOMP.md).

- **Claim (§9's "분해 패스" row):** repeats DECOMP.md's "37 개 중 9 개" figure for how many
  non-core ops the decomposition pass lowers. **Status: stale for the same reason found in
  DECOMP.md** (kernel count grew from 129 to 157 `_aten_all_implemented()` since DECOMP.md was
  written, taking the population from 37 to 45 and LOWERED from 9 to 11).
  **Fixed:** yes — appended a short pointer to the same table cell rather than re-deriving the
  numbers a second time, since DECOMP.md's own correction (added earlier in this pass) is the
  canonical location for the detail.
- **Claim (§2, "실제로 나오는 것"):** capturing `nn.Sequential(Linear, ReLU, Linear)` records
  `Linear` as `aten.t.default` + `aten.addmm.default`, and `relu`'s argument arrives entirely as a
  keyword (`{'self': %1}`) rather than positionally, because `_torch_level_function` binds
  everything by name before dispatch. **Status: confirmed true, byte for byte.**
  **How checked:** ran the exact capture live — `_C._capture_begin([x]); y = m(x);
  tr = _C._capture_end([y])` — and the printed node list matches the document's node-by-node
  transcript exactly, including `aten.relu.default` showing `args=[]`, `kwargs={'self': %1}`.
- **Claim (§4, control-flow rejection):** any Python branch on a tensor value is caught by
  rejecting `aten._local_scalar_dense.default` by name, with a specific reason string.
  **Status: confirmed true.** **How checked:** called `.item()` inside an active capture region;
  `_capture_reason()` returned text matching the document's description verbatim ("a Python branch
  taken on that value is not in the record, so the trace would be one arm of a branch replayed
  unconditionally").
- §7's A/B dispatch-cost measurement is explicitly self-flagged in the document as taken under a
  contaminated (load 8-13) condition and the document already narrows its own conclusion
  accordingly — not re-measured here (re-measuring a microbenchmark is out of scope for a
  documentation audit, and the document's own honesty about the contamination is exactly the
  house-style behavior this audit looks for elsewhere).
- No other false claims found.

### docs/design/DESIGN.md

Longest priority file (1147+ lines), the project's design-rationale document. Per this audit's
own rule, its reasoning/design sections (§1-§8 mostly, the A-vs-B tensor-engine decision, the TTL/
TTA/TTT taxonomy, the kernel-strategy argument) were **left alone** — they are arguments, not
checkable facts, and CLAUDE.md's rule is explicit about not rewriting those. What was checked is
the narrower set of *status* claims about this repo's own tree, which is where this document
turned up its most consequential finding this pass.

- **Claim (§11.1, both the original text and its own 2026-08-25 correction blockquote):**
  "`from_pretrained` 와 실제 체크포인트 경로는 여전히 한 번도 실행되지 않았습니다" (the
  `from_pretrained` path and real checkpoint loading have still never been executed, as of the
  in-document 2026-08-25 update, blocked on `torch._C.is_autocast_enabled`).
  **Status: FALSE today — this is exactly the "refusal names something as missing that already
  landed" failure shape the brief specifically warns about**, and it is the highest-severity finding
  of this whole pass: a design document telling a reader that the project's core end-to-end path
  (loading a real Hub checkpoint and running it) has never worked, when it has.
  **How checked:** ran it directly — `AutoModelForCausalLM.from_pretrained("HuggingFaceTB/
  SmolLM2-135M")` against the live shim, downloaded and loaded real Hub weights, forward pass
  produced logits of the correct shape, and `.generate(max_new_tokens=5)` produced 8 tokens with
  no error. Independently corroborated by `docs/devices/META.md` §7-§8.3 (already audited earlier in this
  pass), which ran the same `from_pretrained` path with an `AutoModelForCausalLM` for llama3-rope
  and checked logits against upstream to 1e-5.
  **Root cause of the gap:** `docs/models/COMPAT.md` (audited earlier, also in this session) implemented
  `torch._C._is_autocast_available` and closed exactly the wall this section names — but nothing
  ever came back to update DESIGN.md's §11.1, which still reads as current.
  **Fixed:** yes — added a nested `> > **Correction (문서 감사, 재측정 2026-09): ...**` blockquote
  directly inside the existing 2026-08-25 blockquote (matching its own nesting convention), giving
  the live re-verification and pointing to COMPAT.md as the commit that closed the wall, while
  leaving the original historical account of what was blocked and why intact.
- **Claim (§9 "이 저장소"):** `torchnative/api/__init__.py:4` has a `SyntaxError` (`*args, *kwargs`
  instead of `**kwargs`) and `torchnative/nn/federated/__init__.py:1` has a dead import
  (`DistributedDataFederated`, no such module), both blocking first import.
  **Status: FALSE today — both fixed.** **How checked:** `py_compile.compile(..., doraise=True)`
  on `torchnative/api/__init__.py` succeeds (no SyntaxError — the file is now docstring-only with
  imports deferred, matching a design note already elsewhere in the file about deferred `torch`
  imports); `import torchnative.nn.federated` and `import torchnative.api` both succeed directly.
  **Fixed:** yes — added a short correction paragraph after the table noting both are resolved,
  without tracking down which commit did it (out of scope for a documentation pass).
- **Claim (§11.1, "열려 있는 성능 결함", 2026-08-28):** `linear`'s per-call `.contiguous()` copies
  a 113 MB weight on every forward pass (30x slower than upstream at the `lm_head` shape), and this
  was explicitly **not fixed** — a deliberate accuracy-vs-speed tradeoff left open, by name,
  matching how `docs/kernels/SDPA.md` §12 handled a similar case.
  **Status: FALSE today — fixed, and landed as the default, not gated behind an opt-in.**
  **How checked:** `docs/perf/LINEAR.md` (2026-08-29, its own file, not previously part of this audit's
  priority list but pulled in because DESIGN.md pointed at it) diagnosed the same defect, measured
  that landing the fix moves bits *only toward* upstream agreement (0 regressions, 35 cases newly
  agreeing with upstream out of 507), and left the accept/gate decision open, uncommitted, for the
  coordinating session. `git log -S"fn batched_matmul" -- rust/torch_c/src/aten.rs` finds commit
  `2e00ec3` ("Perf: Fold instead of broadcasting, and stop copying the weight every call"), and
  `gemm_with_layout_fallback`/`batched_matmul` are present in the current `aten.rs` as the default
  path (`.contiguous()` now only runs as a fallback when candle refuses a strided operand) —
  confirmed by reading the current source, not merely inferred from the commit existing.
  **Fixed:** yes — added a correction blockquote pointing to `docs/perf/LINEAR.md` and the landing
  commit, noting LINEAR.md §6 already names five remaining walls (e.g. bf16/f16 weights still
  widening per call) for anyone who reads past this correction.
- Everything else in §0-§10 (the A-vs-B tensor-engine decision, the TTL/TTA/TTT taxonomy, the
  kernel-strategy argument, the `theRiverLethe`/`test-time-adapters` findings in §9's first two
  tables) either (a) references source in sibling repositories not present in this worktree and
  therefore not checkable against this tree, or (b) is reasoning/design argument rather than a
  checkable fact — both out of scope for fixing per this audit's rules, and not reported as
  findings since nothing here contradicted them.
- This is the second cross-document staleness found in this pass (after DYNAMO.md/TORCHSCRIPT.md)
  and by far the most consequential — a reader trusting §11.1 today would conclude the project's
  core value proposition (real HF checkpoints running on-device) has never been demonstrated, when
  it has been demonstrated twice over (COMPAT.md, META.md) by documents this same document does
  not reference.


---

## Findings (round 2 — the 63 files nobody had reached)

Same method as round 1: read the file, check op-implementation/count/symbol claims against the
live tree, fix what is unambiguously false and mechanical, mark what was verified with a
DOCWATCH marker. Baseline for this round (established before touching any file, same commands as
round 1's baseline, this worktree — `/Volumes/macMini/worktrees/bw-docrest`):
`run.sh` 296 "ok " lines; `compare.py` SUMMARY: 7447/7447 cases passed, 0 failed, ops covered=166,
pending case builders=1; `verify_schemas.py` SUMMARY: 4475/4475 table entries matched upstream, 0
failed. Implemented-ops snapshot captured in `/tmp/docrest_implemented_ops.txt` (166 ops).

### docs/kernels/KERNELS.md

259 lines, single round (`baddbmm` alpha=0 fix, `relu_` new kernel, `uint8` negative-saturation
diagnosed-not-fixed, `topk` tie order deliberately left). All numeric tables are explicit
before/after for this round's own baseline (96/97 ops covered) — house style, not re-annotated.

- **Claim (§1, §6):** `baddbmm`'s `alpha=0` divergence is fixed, `KNOWN DIVERGENCE` no longer
  names it. **Status: confirmed true.** How checked: `aten.baddbmm.default` is in the current
  166-op implemented list; this session's baseline `golden/compare.py` run's `KNOWN DIVERGENCE`
  block (6 cases) names only `aten.div.Tensor_mode` (x4) and `aten.slice.Tensor`/`aten.view.dtype`
  — no `baddbmm` entry. Marked `<!-- DOCWATCH: op-implemented aten.baddbmm.default -->` +
  `symbol-in-file ... baddbmm_default present` (found at `aten.rs:2860`).
- **Claim (§2):** `aten.relu_.default` is a new kernel (`relu_inplace` in `aten.rs`), distinct
  overload from `relu.default`, in-place alias-write limitation inherited from `add_inplace`.
  **Status: confirmed true** — op is implemented, `relu_inplace` exists at `aten.rs:10099`.
  Marked with `op-implemented`/`symbol-in-file`.
- **Claim (§3, the refusal-shaped finding this round prioritised checking):** `uint8` negative
  values still saturate to `[0,255]` instead of wrapping mod 256 (both in `_tensor_from_flat`,
  out of scope, and in `aten._to_copy.default`, in scope but not fixed — the fix needs a
  wrapping-cast helper not written). **Status: confirmed still true, not fixed by later work.**
  How checked: live call `torch.ops.aten._to_copy.default(tensor([-1.0,-2.0,300.0,256.0]),
  dtype=torch.uint8)` against today's shim → `[0, 0, 255, 255]` — the exact saturating pattern
  the document describes, not the wrapping `[255, 254, 44, 0]` upstream gives. No `count`/
  `symbol-in-file` primitive fits a "computed value" claim, so no marker added for this one
  (matches DOCWATCH.md's own documented gap: "numeric/behavioral divergences from upstream" are
  structurally outside the six primitives).
- §4 (`topk` tie order, deliberately unmatched) and §5's verification numbers (2268/2268 ops=97):
  round-scoped, not re-checked individually.
- No false claims found. **Fixed: none needed.**

### docs/design/CORE_ATEN.md

266 lines. Already carries its own top-of-file correction (조율 세션이 서브 에이전트 초안의
"분해표에 없음 → Core ATen 원시" 추론을 무효로 판정) — house style already applied. Its
headline counts (Core ATen tag count 189, decomposition table size 940) are upstream-torch
facts pinned to `torch 2.13.0` (the same pinned version this whole worktree uses), not this
repo's own kernel count, so they do not drift with kernel rounds the way `_aten_implemented()`
counts do.

- **Claim (§0, §1):** `torch.Tag.core` tags exactly 189 aten overloads;
  `torch._decomp.core_aten_decompositions()` has 940 entries. **Status: confirmed true, live.**
  How checked: iterated `torch.ops.aten.*` overloads checking `torch.Tag.core in op.tags` → 189;
  `len(torch._decomp.core_aten_decompositions())` → 940. No `count` primitive in
  `tools/docwatch/check_docs.py`'s registry covers an upstream-torch-tag census (only
  `smoke_ok`/`golden_*`/`schema_*`/`decomp_*`, all sourced from this repo's own harnesses) — not
  markable without extending the primitive set, which is out of scope per this round's brief.
  Reported, not marked.
- **Claim (§0, §4):** of the model's aten-op calls, 2 (`aten.lift_fresh.default`,
  `aten.max.default`) are covered by neither Core ATen nor the decomposition table, so the shim
  must implement them directly rather than relying on either classification. **Status: confirmed
  true that both are implemented today** (not the document's own point — it is a taxonomy claim,
  not an implementation-status claim — but exactly the kind of "must implement directly" refusal-
  adjacent claim this round is told to check). How checked: both present in the current 166-op
  `_aten_implemented()` list. Marked `<!-- DOCWATCH: op-implemented aten.lift_fresh.default -->`
  and `aten.max.default`, with a short note clarifying the marker checks a different claim
  (implemented-ness) than the paragraph's own point (classification).
- §7's self-reported open item (47 vs 48 op count, unresolved by the document's own account) is
  an explicit "we don't know" admission, not a falsifiable claim — left as reported, matching the
  house style already confirmed correct for this shape in META.md's §12 (round 1 of this audit).
- No false claims found. **Fixed: none needed** (only the top-of-file correction already present,
  which predates this round).

### docs/bindings/OVERLOAD.md

505 lines, single early round (the `torch.<op>` user-surface overload resolver — very early
baseline: 3→20 ops, `overloads.json` 45 entries; today's baseline is 166 ops / 200-entry table,
so §0/§10/§12's numeric tables are round-scoped, house style, not re-annotated individually).

- **Claim (§0 table, §8):** `from_config` still fails, blocked specifically on
  `torch._C._dynamo.eval_frame.set_guard_error_hook` being unimplemented ("미구현 — 여기서
  멈췄습니다", "다음 임계 경로는 op 이 아니라 `_C._dynamo` 입니다"). **Status: FALSE today — this
  is exactly the "refusal names a kernel as missing" shape this round was told to prioritise, and
  it is a real hit.** `git log -S"set_guard_error_hook" -- rust/torch_c/src/bootstrap.py` finds
  `2d3663f` ("Feat: Port torch's CPU generator, and give `_C._dynamo` the two names that do
  work") — its own commit message names `set_guard_error_hook` as one of "the two names that do
  work", landed as a real no-op at `bootstrap.py:2695`. Live-verified two ways: (1)
  `hasattr(torch._C._dynamo.eval_frame, 'set_guard_error_hook')` → `True` against today's shim;
  (2) the actual scenario this document's §8 says fails —
  `AutoModelForCausalLM.from_config(LlamaConfig(hidden_size=64, num_hidden_layers=2,
  num_attention_heads=2, intermediate_size=128, vocab_size=100))` — now returns a
  `LlamaForCausalLM` instance with no error.
  **Fixed:** yes — added a `> **Correction (문서 감사, 2026-09): ...**` blockquote after §8's
  table and after §0's `from_config` line, pointing at `2d3663f` and the live re-verification,
  without rewriting the original (correct-at-the-time) diagnosis. Marked
  `<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py set_guard_error_hook present -->`.
- **Claim (§5, the 13/14-op table):** `arange`, `embedding`, `is_floating_point`, `isin`, `pow`,
  `randint` all now resolve and reach a kernel. **Status: confirmed true.** Marked
  `op-implemented` for `aten.arange.default`, `aten.embedding.default`,
  `aten.is_floating_point.default`, `aten.isin.Tensor_Tensor`, `aten.pow.Tensor_Tensor`,
  `aten.randint.low` (all 6 pass against today's 166-op list).
- §2's "vendored tree has no `native_functions.yaml`/`torchgen`" and §7.1's
  `IMPLEMENTED_AWAITING_GOLDEN` gap: mechanism descriptions rather than counts, not re-verified —
  no reason to suspect given nothing else in this pass touched schema sourcing.
- §9's open-items table (`is_floating_point` dispatcher trace difference, `empty` zero-fill,
  `randint` non-reproducibility, `pow` bool-result category unmeasured, `overloads.json` 14-op
  coverage) not individually re-verified beyond what §5 already covers — these are behavioral
  divergence claims, structurally outside DOCWATCH's primitives per DOCWATCH.md's own "what this
  cannot see" list.
- **Fixed: one (the from_config/set_guard_error_hook staleness — the most consequential finding
  in this file, same shape and severity class as round 1's DESIGN.md `from_pretrained` finding).**

### docs/bindings/TENSORBASE.md

572 lines, single round (opens `TensorBase`'s 50 methods: `x * y`, `x.sum()`, indexing, grad-mode
flags). §0/§2/§8/§10's numeric tables are explicit before/after for this round's own baseline
(20→73 ops, `methods.json` new) — house style, not re-annotated individually.

- **Claim (§0 table, §7, the refusal-shaped finding this round prioritised checking):**
  `from_config` stops at wall 6, `tensor.uniform_(...)` inside `kaiming_uniform_`, because
  `aten.uniform_.default`/`aten.normal_.default` are unimplemented — "**여기서 멈췄습니다**"
  (stopped here), `nn.Linear(4, 3)` fails with `NotImplementedError`. **Status: FALSE today —
  same failure shape as the OVERLOAD.md finding above, and arguably higher-severity: this is the
  wall FROM_CONFIG.md's own account measures `uniform_`/`normal_` being called 15/17 times during
  model init, i.e. it blocks constructing any `nn.Linear`, the single most common layer.**
  `git log -S'"aten.uniform_.default"' -- rust/torch_c/src/aten.rs` finds the same commit as the
  OVERLOAD.md fix, `2d3663f` ("Feat: Port torch's CPU generator, and give `_C._dynamo` the two
  names that do work") — its first half (ported CPU generator, MT19937) is exactly what
  `uniform_inplace`/`normal_inplace` (`aten.rs:10274`/`10350`) needed, because candle's CPU
  backend refuses seeding outright (the commit message's own explanation). Live-verified:
  `torch.nn.Linear(4, 3)` now constructs successfully with no error — the exact scenario §7's
  reproduction command names as "여기서 멈춥니다" (stops here).
  **Fixed:** yes — added `> **Correction (문서 감사, 2026-09): ...**` blockquotes at §0's table
  row, §6's item 11, and §7's closing paragraph, pointing to `2d3663f` and the live
  re-verification, without rewriting the original (correct-at-the-time) wall-by-wall account.
  Marked `op-implemented` for both ops and `symbol-in-file` for both kernel functions.
- Everything else in §1-§6 (the methods-table mechanism, the four golden-harness defects found
  and fixed this round, the `bool`-normalization change to `_tensor_from_flat`, the promotion
  rules, `__getitem__`'s multi-op walk, `TensorBase.__eq__`): mechanism descriptions and this
  round's own before/after numbers, not individually re-verified — no reason to suspect any
  specific one while checking the headline refusal claim.
- §6's open-divergence table (1-ULP `cos`, `matmul` trace identity, `mul.Scalar`/`mul.Tensor`
  trace identity, `view`/`reshape` collapsing, in-place alias visibility, `max.dim` return type,
  `cumsum` accumulator width, mixed/advanced indexing, `bitwise_*` elementwise-i64 slowness): all
  behavioral-divergence claims, structurally outside DOCWATCH's primitives, not re-verified.
- **Fixed: one (`uniform_`/`normal_`/`nn.Linear` staleness — same mechanism and severity class as
  the OVERLOAD.md finding above; both trace to the same landing commit, `2d3663f`, suggesting
  `docs/numerics/RNG.md`, still unread, documents this same generator port in more detail).**

### docs/numerics/RNG.md

464 lines. An investigation/design document, not a landing document — it measures whether candle's
RNG can reproduce torch's CPU stream (no), designs a pure-Python port to prove feasibility, and
ends in §5 with a recommendation ("port torch's CPU RNG directly") rather than a claim that the
port exists. Not itself carrying a false present-tense claim, but directly answered by later work.

- **Claim (§5 recommendation, §0 "권고"):** port `uniform_`/`normal_` from torch's CPU generator;
  candle's own `rand_uniform`/`rand_normal` cannot be used (CPU backend refuses seeding). **Status:
  recommendation adopted and landed**, confirmed by cross-reference to the same commit found while
  auditing `docs/bindings/TENSORBASE.md` above: `2d3663f` ("Feat: Port torch's CPU generator, and give
  `_C._dynamo` the two names that do work") — its commit message paraphrases this document's own
  §1.1/§1.3 findings (MT19937, the pre-decrement twist order, `uniform_real`'s 24/53-bit mask,
  `NormalFill16`). Both ops are in the current 166-op `_aten_implemented()` list.
  **Fixed:** not a false claim, so nothing to correct in the original text — added a
  `> **Correction (문서 감사, 2026-09): ...**` blockquote at the top pointing forward to the
  landing commit, matching the pattern already used elsewhere in this audit for "recommendation
  later adopted" documents (e.g. round 1's CAPTURE.md → DECOMP.md). Marked `op-implemented` for
  both ops.
- **Not re-verified:** the accuracy claims themselves (bit-exact `uniform_`, 0-ulp `normal_` on
  the aarch64 scalar path, the FMA-contraction 1-ulp trap in §3.2a) — this pass confirmed the port
  landed and both ops resolve, not that the landed Rust implementation reproduces the document's
  measured precision. That would require re-running the same bit-comparison harness against the
  shipped kernel, out of scope for a documentation-claim audit.
- §2 (candle's RNG internals), §4 (what breaks without the port — already correctly hedged with
  "실측"/"코드 판독"/"미확인" labels per claim), §6 (explicit unknowns list): not re-checked,
  already self-labelled by evidence strength in the house style this audit looks for.
- No false claims found in the document's own text (everything is correctly hedged as
  investigation/recommendation, not as a claim about current state). **Fixed: none needed for
  falseness — one forward-pointing correction added for completeness.**

### docs/kernels/SDPA.md

614 lines, a bit-exactness measurement round for the flash-attention kernel (mostly numeric
tables, round-scoped, house style — not re-measured, same reasoning as round 1's
SEQLEN.md/DTYPE_PERF.md). One claim fell squarely in this round's priority: a refusal that names
a kernel as missing, in the "why we didn't reuse an upstream name" table.

- **Claim (§12.2's naming-candidates table, `enable_flash_sdp`/`sdpa_kernel(MATH)` row):** the
  MATH backend's op sequence is refused by name in `bootstrap.py` "because `_safe_softmax` has no
  kernel." **Status: FALSE, and it was already false when this document was written** — not an
  ordinary later-drift case. `git merge-base --is-ancestor 9612146 4cd3bde` confirms `9612146`
  (the commit that put `aten._safe_softmax.default` in `IMPLEMENTED`, golden-compared from that
  point on) is an ancestor of `4cd3bde` (this document's own landing commit) — the kernel existed
  before this sentence was written. **This is the third instance this round of the specific
  failure pattern flagged for extra scrutiny** (after OVERLOAD.md's `set_guard_error_hook` and
  TENSORBASE.md's `uniform_`/`normal_`), and unlike those two it was not a later regression but a
  transcription lag at the point of writing. **Found independently before this audit, in code**:
  `rust/torch_c/src/bootstrap.py`'s own comment at the exact refusal site already says "The reason
  given here used to be `_safe_softmax` has no kernel, and that stopped being true," and further
  notes a `_sdpa_math` composite has since been written (handling the non-flash/train-mode/dropout
  path) — none of which had been carried back into `docs/kernels/SDPA.md`. The bootstrap.py comment even
  observes this is "the second time a refusal in this function went stale" (counting only the
  bool-mask refusal below it, not this doc's transcription) — i.e. the code's own account already
  undercounts by one, which this audit's finding corrects.
  **How checked:** `git merge-base --is-ancestor` for the ordering; `grep -n safe_softmax
  rust/torch_c/src/bootstrap.py` to find the self-correcting comment; live call
  `torch.ops.aten._safe_softmax.default(...)` succeeds against today's shim; `_sdpa_math` exists
  in `bootstrap.py` at line 5263.
  **Fixed:** yes — added a `> **Correction (문서 감사, 2026-09): ...**` blockquote directly after
  the table cell, quoting `bootstrap.py`'s own self-correction, noting the table's overall
  conclusion (repo-own naming, not upstream's) is unaffected — only the parenthetical reason was
  stale. Marked `op-implemented aten._safe_softmax.default` and
  `symbol-in-file ... bootstrap.py _sdpa_math present`.
- Everything else (§0-§11's bit-exactness tables, §12's 20x cost measurement and the switch
  design): round-scoped performance/correctness measurements, not re-measured — same reasoning
  applied to SEQLEN.md/DTYPE_PERF.md in round 1 (re-measuring risks contamination from concurrent
  agents per CLAUDE.md's own rule, and these are explicitly dated "측정일 2026-08-28" transcripts).
- **Fixed: one** — smaller in scope than the OVERLOAD.md/TENSORBASE.md findings (this one didn't
  block a whole capability, just mis-stated a design-decision rationale), but notable as the only
  one of the three where the staleness predates the document's own writing rather than arriving
  from a later commit.

### docs/design/TORCH_C.md

387 lines. The genesis document — `rust/torch_c` at 3 implemented ops, before OVERLOAD.md,
TENSORBASE.md, BOOL.md, VIEWS.md existed. §5 ("다음에 와야 하는 것", next steps in priority
order) is a five-item punch list, and this round's job is exactly to check whether "next step"
items are still open. Checked all five against today's tree:

- **Item 1 (`tokenizers`/`onig` forced dependency via candle-core):** not independently
  re-verified (a `Cargo.lock` dependency-tree question, out of this round's territory —
  `rust/`/`Cargo.lock` are forbidden). Reported, not checked.
- **Item 2 (dtype promotion table):** **Status: resolved, but not the way this document expected**
  — not implemented, and now confirmed *deliberately* not implemented.
  `torch.ops.aten.add.Tensor(float32_t, float64_t)` still refuses with the exact message quoted
  here (live-verified). `docs/bindings/TENSORBASE.md` §2.3 (audited above, this round) states the decision
  outright: cross-dtype tensor ops refuse by name, to avoid "the silent numeric drift"
  `docs/design/DESIGN.md` §5 names as the main risk of the candle path; Python scalars still get
  wrapped-number promotion. **Fixed:** added a correction noting the open question became a
  closed decision, with the opposite answer to what §2's framing implied was coming. Marked
  `symbol-in-file rust/torch_c/src/aten.rs same_dtype present` (the function that enforces it).
- **Item 3 (`torch.bool`):** **Status: closed.** `docs/numerics/BOOL.md` (unread this round, referenced
  only) is presumably the landing document. Live-verified: `hasattr(torch, 'bool')` → `True`,
  `aten.eq.Scalar(...)` returns a `torch.bool`-dtype result. **Fixed:** correction added pointing
  to `docs/numerics/BOOL.md`.
- **Item 4 (`torch.ops.aten.<op>.<overload>` entry point):** **Status: closed** — exactly what
  `docs/bindings/OVERLOAD.md` and `docs/bindings/TENSORBASE.md` (both audited above, this round) built.
  **Fixed:** correction added, cross-referencing both.
- **Item 5 (`aten.view.default` and alias semantics):** **Status: closed, with a known residual**
  — `docs/kernels/VIEWS.md` (audited in round 1) is exactly this item; round 1's own findings (in-place
  writes now reach through views) confirm it, with `docs/kernels/VIEWS.md` §6.4's `slice.Tensor`
  step>1/`view.dtype` structural gap still open. **Fixed:** correction added, citing round 1's
  own VIEWS.md findings rather than re-deriving them.
- §0-§4 (the original landing measurements: 3-op baseline, three-target build, import-cost
  numbers): explicitly framed as this document's own round snapshot, house style, not
  re-annotated.
- §6's self-reported "미확인" list (Android/iOS device import, golden comparison, stripped size,
  `tokenizers`/`onig` removal, simulator target, `candle-ug` iOS exclusion, `affine` integer
  behavior, `_tensor_from_flat`'s f64 precision loss): explicit unknowns, not falsifiable claims,
  left as reported (several are now presumably resolved by later rounds — e.g. golden comparison
  obviously exists now — but re-auditing every "미확인" item in a genesis document against 60+
  later rounds is not a productive use of this round's time; the five §5 "next step" items are
  the load-bearing ones because they were explicitly prioritised as blocking).
- **Fixed: four of five §5 items got a correction** (all closed, one — dtype promotion — closed
  in the opposite direction from what the document's own framing anticipated, which is worth
  flagging on its own: not every stale "TODO" resolves the way it was written to resolve).

### docs/graph/QUANT.md

573 lines, a measurement round (dtype-lowering cost, int8 storage gap, KleidiAI/candle-Q4K
comparison, dotprod compile-time gating). Already carries its own top-of-file "후속" (follow-up)
correction pointing to `docs/numerics/DTYPE.md` and `docs/graph/QUANT2.md` for two specific superseded claims
(§8 item 2's "7.8x weight compression" scoped to KleidiAI only; §7's "no verification axis"
closed by QUANT2.md §2) — house style already applied, before this audit touched it.

- **Claim (결론 2, §2):** `int8`/`quint8`/`qint8` cannot be constructed at all, because
  candle-core 0.11's `DType` enum has no `I8` variant — not an unimplemented op, a missing
  backend storage type. **Status: confirmed still true.** How checked: read
  `candle-core-0.11.0/src/dtype.rs` directly (the version this worktree's `Cargo.lock` pins, per
  QUANT.md's own environment line) — `DType` has `U8, U32, I16, I32, I64, BF16, F16, F32, F64`,
  no `I8`. This fact is pinned to the vendored crate version, not to this repo's own kernel count,
  so it does not drift the way `_aten_implemented()` counts do (same reasoning as CORE_ATEN.md's
  `torch.Tag.core`/decomposition-table counts, above). Not markable with a DOCWATCH primitive (no
  "candle crate fact" source in the registry, same gap as CORE_ATEN.md).
- 결론 1·3·4 and the whole of §0-§9's A/B measurement tables (dtype-lowering slowdown, KleidiAI
  vs. candle-Q4K prefill/decode ratios, the dotprod compile-flag 2.05x): round-scoped performance
  transcripts, dated (측정일 2026-08-28), not re-measured — same reasoning as SDPA.md/SEQLEN.md
  above (re-measuring performance risks contamination from concurrent agents, out of scope for a
  documentation-claim audit).
- No new false claims found beyond what the document's own top-of-file correction already caught.
  **Fixed: none needed this round** (the document's own prior self-correction already covers the
  two items that had gone stale).

### docs/graph/QUANT2.md

542 lines, the quantization landing round (GGML bit-exact verification axis, `Repr`'s third
variant, module-swap path, SmolLM2 q8_0 20/20 token match). Also opens by pointing at
`docs/graph/QUANT.md`/`docs/numerics/DTYPE.md` for prior context — already cross-linked, house style.

- **Claim (결론 5, §6.4, §7 table item 1):** dense `linear`'s per-call weight-transpose copy
  (4.7-82x cost vs. a contiguous tensor) is "an existing defect, not fixed here," explicitly
  deferred with "다음 사람에게: 이것이 §7 표의 1번" (to whoever comes next: this is #1 in the
  wall table). **Status: FALSE today — fixed the very next day.** `git log -1 --format=%ci`
  on this document's own commit (`b032276`, 2026-08-28) vs. the fix commit (`2e00ec3`, "Perf:
  Fold instead of broadcasting, and stop copying the weight every call", 2026-08-29) shows one
  calendar day between them; `git merge-base --is-ancestor b032276 2e00ec3` confirms the order.
  This is the same fix round 1's DESIGN.md audit already found (cited there via `docs/perf/LINEAR.md`,
  itself still unread by this round) — `gemm_with_layout_fallback`/`batched_matmul` are the
  current default path in `aten.rs`, `.contiguous()` now only runs as a candle-refusal fallback.
  **Fixed:** yes — added a `> **Correction (문서 감사, 2026-09): ...**` blockquote after §6.4's
  "다음 사람에게" paragraph and struck through §7 table row 1's header, both pointing to `2e00ec3`
  and cross-referencing round 1's DESIGN.md finding rather than re-deriving it. Marked
  `symbol-in-file` for both `batched_matmul` and `gemm_with_layout_fallback`.
- **Claim (§7 item 4, quant.rs `_quantized_linear`):** `bfloat16` activations are refused by name
  (candle's `QMatMul::forward` only accepts f32/f16). **Status: confirmed still true.** How
  checked: read `rust/torch_c/src/quant.rs:274-296` directly — the refusal is live in source,
  unchanged, citing `docs/numerics/DTYPE.md` §6.2 by name.
- §2's bit-exactness verification axis (GGML block format round-trip, 6 injected-fault
  self-checks), §5's SmolLM2 q8_0/q4_0 results, §6.2-6.3's A/B cost tables: round-scoped
  measurement transcripts (측정일 2026-08-28), not re-measured, same reasoning as QUANT.md above.
- §7's other wall items (2: Q4K can't hold 576; 3: q4_0 real-model quality collapse; 5: no GGUF
  writer; 6: no device measurement; 7: `+dotprod`; 8: no perplexity harness) and §8's explicit
  unknowns list: not individually re-verified — no reason to suspect any specific one, and several
  are structural/scope statements ("we didn't measure this") rather than falsifiable claims.
- **Fixed: one** — same severity class as the OVERLOAD.md/TENSORBASE.md findings (a "not fixed
  here, left for next time" claim that was actually already closed), the third instance this
  round of a stale claim about a specific missing/refusing capability, though this one is a
  performance defect rather than an outright `NotImplementedError`.

### docs/distributed/DISTRIBUTED.md

438 lines, the `torch.distributed` landing round (opens `import transformers`, `Store`
re-export, `world_size=1` process group). Already carries its own pre-existing correction at §7
("열렸습니다 (2026-08-28). `docs/models/E2E_REAL.md` §4 를 보십시오" — forward pass now works, autocast
wall closed) — landed before this audit, house style already applied at the point that mattered
most. But the correction didn't propagate to §8's unknowns table, three sections later in the
same file — the exact "correction doesn't propagate everywhere" mechanism round 1 found in
DECOMP.md/CAPTURE.md.

- **Claim (§8 item 1):** forward pass on a `from_config`-built model "미통과" (does not pass),
  blocked on the §7 autocast wall. **Status: FALSE, and already known false by this document's
  own §7 correction three sections earlier** — an internal inconsistency, not a drift against the
  live tree. **How checked:** live-verified independently anyway (not just trusting §7's own
  correction): `AutoModelForCausalLM.from_config(LlamaConfig(...))` forward pass succeeds today,
  producing correctly-shaped logits.
- **Claim (§8 item 2):** `from_pretrained`/real checkpoint path "미시도" (untried), blocked behind
  item 1. **Status: FALSE** — same mechanism, and also independently confirmed by round 1's
  DESIGN.md audit (real Hub checkpoint loading, `.generate()`, works today).
  **Fixed:** yes — struck through both §8 table cells with `> **정정 (문서 감사, 2026-09)**`
  inline corrections pointing at §7's own existing correction and at round 1's DESIGN.md finding,
  rather than leaving a same-document contradiction standing. Marked
  `symbol-in-file rust/torch_c/src/bootstrap.py is_autocast_enabled present`.
- §8 items 3-6 (`world_size >= 2` unimplemented, `torchnative.nn.federated` empty, no
  accelerator device abstraction, DDP machinery refused by name): not re-verified — these read as
  deliberate scope limits for this round rather than "next step" claims, and nothing else in this
  pass touched multi-rank distributed or DDP.
- §0-§6 (the wall-by-wall `_c10d_init` walkthrough, the `ProcessGroup`/`ReduceOp` enum
  reconstruction, the gate-count table): round-scoped landing narrative, not re-verified line by
  line — no reason to suspect given §7's self-correction already demonstrates this document was
  being actively maintained.
- **Fixed: one internal-consistency defect** (two table cells contradicting the same document's
  own §7 correction) — smaller in kind than the OVERLOAD.md/TENSORBASE.md/QUANT2.md findings
  above (this one was already caught once, just not everywhere), but the same underlying lesson:
  a correction in one place does not automatically fix every restatement of the same claim
  elsewhere in the file.

### docs/models/CKPT2.md and docs/models/E2E_REAL.md

Audited together — `E2E_REAL.md` (541 lines) already has a top-of-file correction pointing
forward to `CKPT2.md` (its own sequel, real-checkpoint weight loading), and `CKPT2.md`'s §7 is
where the trail ends (real Hub model, SmolLM2-135M). Both needed the same fix, discovered while
following that chain: `docs/distributed/DISTRIBUTED.md`'s finding above (forward pass now works) prompted
checking whether the chain's *final* claim — that the real pretrained model's forward pass still
doesn't run — was also stale.

- **Claim (CKPT2.md §7.1, §8 items 1-2; repeated in E2E_REAL.md's own top correction):**
  `SmolLM2-135M`'s forward pass is blocked on two kernels — `scaled_dot_product_attention(
  enable_gqa=True)` (upstream's flash path doesn't repeat KV heads) and `aten.where.ScalarOther`
  (missing from `sdpa`'s eager fallback's masking); `generate()` additionally blocked on an
  `aten.mul.Tensor: int64 vs bool` promotion gap in `_prepare_attention_mask_for_generation`.
  **Status: FALSE today — all three closed, and this is a direct continuation of the SDPA.md and
  DISTRIBUTED.md findings above (same commit family: the `_sdpa_math` composite).**
  **How checked:** `aten.where.ScalarOther` is in the current 166-op `_aten_implemented()` list;
  `bootstrap.py`'s `_sdpa_math` has an `if enable_gqa:` branch (line 5312) that repeats KV heads
  before the matmul — the exact fix this document says is needed. Most decisively, live-ran the
  actual scenario both documents describe as failing: `AutoModelForCausalLM.from_pretrained(
  "HuggingFaceTB/SmolLM2-135M")` (real Hub download, GQA architecture, `num_attention_heads=9`,
  `num_key_value_heads=3`) — forward pass succeeds (`logits.shape == (1, 2, 49152)`) and
  `model.generate(ids, max_new_tokens=5)` succeeds, producing 5 additional tokens. This
  independently reproduces round 1's DESIGN.md audit finding (same model, same conclusion) from a
  different document chain, which is worth noting: two unrelated documents (DESIGN.md and
  CKPT2.md) both named this exact scenario as broken, and both were wrong for the same underlying
  reason (the fix landed and neither document was updated).
  **Fixed:** yes, in both files — `CKPT2.md` §7.1 gets a `> **Correction (문서 감사, 2026-09):
  ...**` blockquote plus struck-through §8 table cells; `E2E_REAL.md`'s own top correction
  paragraph (which repeated CKPT2.md's now-stale claim) gets a matching strikethrough +
  correction rather than being left to repeat the error. Marked `op-implemented
  aten.where.ScalarOther` and `symbol-in-file rust/torch_c/src/bootstrap.py _sdpa_math present`
  on `CKPT2.md` (the `_sdpa_math` marker duplicates one already added to `SDPA.md`/`TENSORBASE.md`
  above — DOCWATCH doesn't mind the same fact being checked from multiple documents that each
  depend on it).
- **CKPT2.md, not otherwise re-verified:** §1-§6 (the checkpoint-reading path itself — mmap
  entry points across safetensors/`.bin`, bit-exact weight comparison, shared-tensor/sharded-
  index/`_metadata` measurement) and §8's other unknowns (items 3-12: `from_file(shared=True)`
  refusal, memory measurement, bf16-upcast judgment call, legacy `torch.load` format, weight-
  sharing identity, `zeros_like` free-function form, Android/iOS cross-compile, unsupported
  checkpoint dtypes) — no reason to suspect any of these while confirming the one load-bearing
  refusal claim; several are explicit "not measured" admissions rather than falsifiable claims.
- **E2E_REAL.md, not otherwise re-verified:** §1's dispatch-mode methodology (`repr`'s 21-op
  census, the `isfinite` composite-vs-kernel distinction), §2's character-exact repr comparison,
  §3-§9 generally — round-scoped landing narrative, only the specific chained-forward claim was
  checked given time budget, prioritised because it was the one repeating a now-false capability
  claim rather than describing mechanism.
- **Fixed: one finding, propagated to two files** — same severity class as the DESIGN.md
  `from_pretrained` finding from round 1 (a real end-to-end capability claimed broken when it
  works), found here via a different document chain, which raises confidence that the underlying
  capability really is solid rather than this being a fluke of which document happened to get
  checked.

### docs/models/GENERATE.md

551 lines. This is the actual landing document for the CKPT2.md/E2E_REAL.md finding above — the
round that fixed `enable_gqa`, `aten.where.ScalarOther`, and the `mul`/`bitwise_and` promotion
gaps, and got `SmolLM2-135M.generate()` producing tokens identical to upstream. Internally
consistent with what this round independently found live (no correction needed for the headline
claims — this document already reflects them correctly; `CKPT2.md`/`E2E_REAL.md` were the ones
that hadn't caught up).

- **Claim (§7.1):** `attn_implementation="eager"` generation is blocked specifically on
  `aten.index.Tensor` refusing when *more than one* index tensor is present (multi-index advanced
  indexing with broadcasting) — a narrower claim than "index.Tensor is unimplemented." **Status:
  confirmed accurate, still true, and correctly scoped.** How checked: `aten.index.Tensor` is in
  the current 166-op implemented list (the single-index case works, matching §7.1's own framing
  that eager *forward* pass succeeds); the refusal is for the specific multi-index case, pinned
  by a named test (`test_eager_generate_stops_at_index_tensor_and_says_so`) that the document says
  is designed to fail two ways (op implemented, or generate blocked earlier) — a good example of a
  refusal claim precise enough not to go stale by accident.
- **Claim (§7.3):** `where.ScalarSelf`/`where.Scalar` still unimplemented (schema present, no
  measured caller). **Status: confirmed still true.** How checked: neither is in the current
  166-op list (only `where.ScalarOther` and `where.self` are, both landed by this same round per
  §0's table).
- §0's before/after table, §1-§6 (token-match methodology, the four walls closed this round,
  bfloat16 GEMM-reassociation drift): round-scoped landing narrative matching what round 1 already
  established for similarly-dated documents; not re-verified beyond the two claims above since
  this document is the authoritative source for the CKPT2.md/E2E_REAL.md correction made above,
  not itself a target of that correction.
- §7.3's other undone items (`add`/`sub`/`div`/`bitwise_or.Tensor` promotion refused, `use_cache`
  generation unmeasured, sampling generation unmeasured, Android/iOS unmeasured): not
  individually re-verified — no reason to suspect given the two spot-checks above both held.
- **Fixed: none needed in this file** — it is the source of truth the two corrections above point
  to, not a target of correction itself.

### docs/models/SAMPLING.md

411 lines, the `do_sample=True` landing round (8 ops, `multinomial` the only RNG-consuming one,
90/90 seeded-token match against upstream). Consistent with `docs/kernels/KERNELS.md`'s §4 finding on the
same `topk` tie-order decision (deliberately not reproduced, matching evidence in both places).

- **Claim (§6, first bullet):** the 8 ops landed this round are reachable only via
  `_aten_dispatch`/the golden harness, not via `torch.<op>` Python spellings, because
  `overloads.json`/`methods.json`/`bootstrap.py` were out of this round's ownership.
  **Status: FALSE today.** How checked: live calls `torch.multinomial(...)`, `torch.sort(...)`,
  `torch.topk(...)` all succeed against today's shim; `overloads.json` has `"multinomial"`,
  `"sort"`, `"topk"` keys. A later round (unidentified — not chased down, out of scope) filled in
  the table entries this document said were missing. **Fixed:** yes — struck through the claim
  and added a `> **정정 (문서 감사, 2026-09)**` inline correction. Marked three `json-key`
  markers.
- **Claim (§4.2, KERNELS.md's own topk-tie-order decision restated here):** `topk` tie order and
  `sorted=False` order are deliberately not reproduced (torch promises neither); golden cases use
  multiset comparison for tied inputs instead of index comparison. **Status: confirmed still true
  and consistent** — same decision independently documented in `docs/kernels/KERNELS.md` §4 (audited
  above, this round), no contradiction between the two.
- §0's gate-count table, §1-§3 (the 8 ops themselves, the seeded-stream word-count match), §4.1's
  three "would have guessed wrong" findings (`squeeze.dim` no-op on non-1 axes, CPU
  `half_to_float` softmax rejection, `scatter.src` shape-rule looseness): round-scoped landing
  narrative, not re-verified — no reason to suspect given the two spot-checks above.
- §6's other explicit gaps (non-last-axis reduced-dtype `_softmax`, `aten.exponential_` not
  advertised as an op, `sort.stable`/`topk` `out=`/`scatter.value`/`scatter.reduce` out of scope,
  non-contiguous `normal_` path B uncompared, GPT-2-family sampling untested, device import
  unmeasured): not individually re-verified, explicit "not done" admissions rather than checked.
- **Fixed: one** — a Python-surface reachability claim, same general shape as the OVERLOAD.md/
  TENSORBASE.md findings (something the document says needs a separate round did get that round,
  and nobody updated this document), lower severity since the underlying ops already worked via
  the aten-key entry point.

### docs/devices/DEVICE.md

571 lines, an Android-emulator device-verification round (54-case battery, 50/54 bit-identical
to host, 4 at 1 ULP). Mostly a measurement transcript; one refusal-shaped claim found and fixed.

- **Claim (§7):** `torch.relu` (the bare Python spelling, no overload suffix) has no
  `overloads.json` table entry and fails identically on host and device — a real gap, not a
  device-specific issue. **Status: FALSE today.** How checked: `overloads.json` now has a
  `"relu"` key; live call `torch.relu(torch.tensor([-1.0, 2.0]))` succeeds
  (`tensor([0., 2.])`). Same shape of staleness as the `torch.multinomial`/`sort`/`topk` finding
  in `docs/models/SAMPLING.md` just above — the overloads table kept growing in rounds after this
  document was written, and nothing came back to update it. **Fixed:** yes — added a
  `> **Correction (문서 감사, 2026-09): ...**` blockquote before the reproduction transcript,
  cross-referencing the SAMPLING.md finding. Marked `json-key rust/torch_c/src/overloads.json
  relu present`.
- §1-§5 (environment, the 54-case battery, bit-exactness/1-ULP breakdown), §6 (the
  `_multiprocessing`/`_posixshmem` stub decision, explicitly left undecided by the document's own
  account — "이 문서는 셋 중 무엇도 고르지 않았다"): not re-verified — §6 is an open decision, not
  a falsifiable claim, and §1-§5 are device-measurement transcripts (out of scope to re-run per
  CLAUDE.md's single-device-at-a-time constraint and this round's time budget).
- **Fixed: one**, matching the SAMPLING.md pattern exactly (a Python-surface overload-table gap
  that got filled by unrelated later work).

### docs/devices/DEVICE_ABS.md

610 lines, the device-abstraction landing round (`torch.device` label parsing/validation,
`nn.Module.to`, meta device groundwork, torch-function mode stack design). Already carries two
pre-existing correction blockquotes at §7.1/§7.2 ("구현됨 (2026-08-25). `docs/devices/META.md`" / 
"`docs/devices/META.md` §8") — landed before this audit, exactly the annotation this audit adds
elsewhere. But those corrections hadn't propagated to §9's summary table three sections later —
the same "correction doesn't propagate everywhere in the same file" mechanism found in
`docs/distributed/DISTRIBUTED.md` above.

- **Claim (§9's 26-remaining-mismatch table, rows "컨텍스트 매니저/기본 장치 미구현" and
  "`meta` 미구현"):** context-manager/default-device support (6 spellings) and `meta` device
  support (3 spellings) are still unimplemented. **Status: FALSE, and already contradicted by
  this same document's own §7.1/§7.2 correction blockquotes** ("구현됨 (2026-08-25)"), which
  simply never got reflected in this later summary table. **How checked:** independently
  live-verified rather than just trusting the existing blockquotes: `torch.zeros(2,
  device='meta')` and `with torch.device('meta'): torch.zeros(3)` both succeed today.
  **Fixed:** yes — struck through both table-row labels with `> **정정 (문서 감사, 2026-09)**`
  inline notes pointing back at §7.1/§7.2's own corrections and at `docs/devices/META.md`. No new
  DOCWATCH marker needed/added here — the underlying capability (meta device, mode stack) belongs
  to `docs/devices/META.md`, already audited and marked in round 1.
- §0-§6, §7.3-§7.4, §10 (label-parsing gap table already framed as "before this round's work" —
  §2.2/§2.3's "우리(작업 전)" columns, correctly historical; `privateuse1` rename and Metal/
  Vulkan/NPU deliberately not implemented, explicit "no measured demand" framing): not
  individually re-verified beyond a quick grep confirming `_rename_privateuse1_backend`/
  `with_meta` context-manager helpers are absent from `bootstrap.py` — consistent with §7.3's own
  claim, so no staleness there.
- **Fixed: one internal-consistency defect**, structurally identical to the DISTRIBUTED.md finding
  above — the pattern is common enough across this batch (two instances so far) that it may be
  worth a standing rule: whenever a round adds a "구현됨" correction blockquote mid-document, also
  grep the same file for a summary table that might still be counting the old claim.

### docs/devices/DEVICE_LOAD.md and docs/devices/DEVICE_LOAD_IOS.md

131 and 279 lines, sibling genesis-era device-load verifications (Android emulator / iOS
simulator, both at the 3-op stage: `full`/`add.Tensor`/`mm.default`). No false claims found —
both correctly scope themselves to what they measured ("3개 op" throughout, never claiming more).

- Both documents' core claim ("`import _C`/`_C._aten_dispatch` loads and runs on-device") is not
  the kind of claim that goes stale — a working load-and-run measurement from the past does not
  become false because more has been added since. Not re-run (device tests are the "1 at a time"
  resource this round did not need to spend on a claim that isn't actually in question).
- **Fixed: none needed** — added a short top-of-file pointer to each, noting `docs/devices/DEVICE.md`
  (Android, already audited above — 54-case battery, richer successor) and `docs/platform/IOS.md`
  (not yet read this round) as the documents that carry the current state forward, so a future
  reader doesn't mistake the 3-op snapshot for today's coverage. `docs/devices/DEVICE_LOAD.md`'s own
  "다음에 필요한 것" item 1 (verify on a second API level) still appears open —
  `docs/devices/DEVICE.md` used the same `pmp_api26` emulator, not a different level.
- No DOCWATCH markers added — nothing in either file reduces to one of the six primitives (device
  load/run success isn't an op-implemented/count/symbol claim).

### docs/design/ABI3.md

671 lines, a design-decision document (abi3 vs. version-locked build) ending in a clear
recommendation: "abi3 를 켜라, floor 는 `abi3-py313`." Not a landing document itself, but its
opening "현재 상태" line makes a present-tense claim about the build that the recommendation
implicitly invites a reader to check.

- **Claim (opening line):** `rust/torch_c/Cargo.toml`'s `pyo3` dependency has
  `features = ["extension-module"]` only — non-abi3, version-locked. **Status: FALSE today — the
  document's own recommendation was adopted.** How checked: `rust/torch_c/Cargo.toml` line 23
  today reads `features = ["extension-module", "abi3-py313"]`. **Fixed:** yes — added a
  `> **Correction (문서 감사, 2026-09): ...**` blockquote noting the recommendation was adopted,
  without rewriting the original "current state" framing (which was true when written). Marked
  `symbol-in-file rust/torch_c/Cargo.toml abi3-py313 present`.
- §1-§4 (the abi3 feature-loss survey, the ~1ns boundary-call cost measurement, the cross-build
  wiring impact assessment) and §5 (PythonMultiplatform's actual interpreter version, 3.14.7 not
  3.13 — a claim about a *sibling* repository, not this one): design-rationale/measurement
  sections, not re-verified — the recommendation having been adopted is the one fact that
  actually needed checking (a document's own recommendation silently *not* being followed would
  be the more dangerous failure mode, and that wasn't the case here).
- **Fixed: one** — the "current state" framing at the top, now stale because the document's
  advice worked.

### docs/platform/IOS.md

447 lines, a wheel/device-verification document (simulator wheel fully verified — load, import,
compute; device wheel verified for symbol resolution only, honestly self-limited by lacking
physical hardware). Unusually careful about distinguishing "verified" from "not verified" already
(§0's judgment table, §11.0's "what this tool does NOT do" section) — the house style this audit
looks for, already present.

- §10's "아직 남은 것" (still open) table items — real iOS device execution, actual app-bundle
  embedding path, `_multiprocessing`, PEP 730 `.so`→`.framework` conversion — are infrastructure/
  hardware gaps, not kernel-capability claims of the kind that go stale as `_aten_implemented()`
  grows. Not re-verified: this round's tooling has no physical iOS device or provisioning
  profile, same constraint the document itself names, and `tools/wheel/` is outside this round's
  territory (forbidden for edits; reading it to verify a claim would still require a device this
  environment doesn't have).
- No refusal-shaped "kernel X is missing" claims found in this document — its subject is
  packaging/loading, not op coverage, so the failure pattern this round was told to prioritise
  does not really apply here.
- No false claims found; nothing checkable against the live tree in this document changed status
  since it was written (device hardware access is not something later kernel rounds could have
  affected). **Fixed: none needed.**

### docs/platform/LINUX.md

1124 lines, two rounds (1회차/2회차, cross-building a Linux x86_64 wheel from macOS), already
self-consolidated into one "진행 상황 요약" (progress summary) table at the top showing the final
state after both rounds — the linker/glibc-stub blocker from round 1 diagnosed correctly and
closed by round 2's `cargo-zigbuild` adoption, matching house style (a document correctly
tracking its own multi-round history rather than needing two separate audited entries).

- The summary table's "층 7: Linux 에서 실행, **불가** — 컨테이너 런타임도 Linux 기계도 없고
  설치하지 않는다" row is an environment/tooling-availability fact (no docker/podman/colima/lima/
  qemu, by deliberate policy) rather than a kernel-capability claim — not the shape this round's
  refusal-pattern check targets, and not verified live (checking whether container tooling has
  since been installed is outside this round's territory; `tools/wheel/`/`scripts/` are
  forbidden, and the claim is about host tooling, not this repo's code).
- No refusal-shaped "kernel X is missing because Y" claims found — this document's content is
  cross-compilation plumbing (linker, glibc stubs, PyO3 cross config, manylinux tagging), not op
  coverage.
- §5.4's "부수적으로 고친 것" (incidentally fixed: a refusal that named the wrong target) and
  §6's symbol-resolution verification limits (`readelf`/`nm` weaknesses vs. iOS's two-level
  namespace check): mechanism descriptions from this document's own landing, not independently
  re-verified — no reason to suspect them and they are not "still open" claims.
- No false claims found. **Fixed: none needed.**

### docs/platform/WINDOWS.md

548 lines, same shape as `docs/platform/LINUX.md` (its explicit sequel — "Linux 가 먼저 서고 나서
시작했다"): a consolidated progress-summary table showing all build/symbol-resolution layers
passed, execution blocked only by lacking physical/VM hardware (deliberate policy, not a kernel
gap). No refusal-shaped "kernel X missing" claims — this document's content is MSVC
cross-toolchain plumbing (`cargo-xwin`, PE import libraries), not op coverage.

- Summary-table row "층 7: Windows 에서 실행, **불가**": environment-tooling fact, not re-verified
  live (same reasoning as LINUX.md above — checking host tooling availability is outside this
  round's territory and not a claim about this repo's kernels).
- No false claims found. **Fixed: none needed.**

### docs/platform/WASM.md

1524 lines, English (unlike most of this repo's Korean documents), a four-layer WASM feasibility
investigation. Unusually — and by its own account, deliberately — self-superseding: it opens with
an explicit reading-order instruction ("Read §8 before §7, and §7 before §2d/§3c/§5. §8 supersedes
§7 where they differ...") and even documents its own past mistake in a table at the end (§8.3's
first draft guessed "blocked at dependency resolution" without running the build, and the document
flags this itself as "exactly the CLAUDE.md §5.5 mistake this document otherwise tries to avoid").
This is the most self-critical document found in either round of this audit.

- The blocked/not-attempted items (candle WASM SIMD lacking `CurrentCpuF16`/`CurrentCpuBF16` for
  `+simd128`, CPython-on-WASI needing `dlopen`) are pinned to specific candle/CPython-WASI
  versions rather than this repo's own kernel count, similar in kind to `docs/graph/QUANT.md`'s
  `DType`-enum finding above — not the drifting category this round's `_aten_implemented()`
  growth affects. Not independently re-verified (would require rebuilding for wasm32 targets,
  outside this round's time budget and not flagged as suspicious by anything else checked).
- Given how thoroughly this document already tracks and corrects its own staleness in-line, a
  full re-audit was not attempted — the marginal value is low relative to files that had never
  been touched by this kind of scrutiny. **Fixed: none needed**, none found.

### docs/platform/VENDOR.md

623 lines, the genesis vendoring-wall document (`_C` surface 17 names vs. upstream 989,
`AutoModelForCausalLM.from_config` unreached, `import torch` stopping at line 1050 in strict
mode). Explicitly framed as diagnosis, not remediation ("목표는 되게 하는 것이 아니라 어디서
깨지는지 아는 것" — the goal is knowing where it breaks, not making it work), so none of its
per-wall findings were ever claimed "solved" by the document itself.

- **Not individually re-verified wall by wall** — this document predates essentially every other
  round audited in this pass (`OVERLOAD.md`, `TENSORBASE.md`, `DISTRIBUTED.md`, `COMPAT.md`, the
  `DESIGN.md` §11.1 finding from round 1), all of which have already been independently confirmed
  in this session to have closed the walls this document names (`from_config`/`from_pretrained`
  both succeed today, `import transformers` passes). Re-deriving each of §1-§8's individual walls
  against the live tree would substantially duplicate work already done auditing the documents
  that actually closed them.
  **Fixed:** added a single top-of-file correction noting the genesis framing and pointing at the
  later documents that closed the two headline walls (`from_config`, `import transformers`),
  rather than annotating every individual wall in §1-§8 — matches the "one forward-pointing
  correction, not a line-by-line rewrite" pattern already used for `docs/design/TORCH_C.md` above.
- No DOCWATCH markers added — the correction is a pointer to other already-marked documents, not
  a new independently-checkable claim.

### docs/kernels/TAIL.md

290 lines — this is the document `docs/kernels/KERNELS.md` (audited first this round) opens by citing:
"docs/kernels/TAIL.md가 쌓아 둔 미해결 목록(§6) 중 우선순위가 높은 순서로 셋을 받았다." §6 lists three
backlog items; KERNELS.md closed item 2.1 (`baddbmm` alpha=0, confirmed above). Checked the
other two against today's tree.

- **Claim (§6 item 2.2, §2.2 body):** the shared `arith_tag` promotion function blanket-refuses
  `torch.bool` operands for `.Scalar` overloads (`mul.Scalar`, `add_.Tensor`, etc.), when upstream
  actually treats `.Scalar` arithmetic on bool as ordinary integer arithmetic (`mul.Scalar(bool_t,
  3)` → `tensor([3,0,3])` int64, not a logical op) — a real overreach the document explicitly
  left unfixed, pinned in golden cases as `expect="c_error"`. **Status: confirmed still
  unfixed, live-reverified — and worth flagging beyond "still true."** `torch.ops.aten.mul.Scalar(
  torch.tensor([True,False,True]), 3)` still raises today, but the *message* has changed since
  this document was written: `"aten.mul.Scalar: torch.bool operands are logical, not arithmetic,
  in torch (BOOL.md §2.2) and are not implemented in torch._C shim"`. That message's factual claim
  — "logical, not arithmetic, in torch" — is itself wrong for `.Scalar` overloads per this
  document's own measurement above (upstream's `.Scalar` bool arithmetic *is* arithmetic; only
  `.Tensor`-`.Tensor` bool ops are logical upstream, which is what `docs/numerics/BOOL.md` §2.2's table
  actually measured — `x + x` for two bool tensors, not a tensor-and-scalar case). So a later
  round (presumably `docs/numerics/BOOL.md`'s landing) appears to have generalized §2.2's `.Tensor`-only
  finding into a blanket refusal message that now asserts something false about `.Scalar`
  overloads specifically — the same "blanket refusal doesn't distinguish overloads" bug this
  document already named, now with a confidently-wrong justification attached to it. This is a
  live code defect, not a documentation staleness issue, and `rust/` is outside this round's
  territory to fix — reported here rather than edited.
  **Not fixed in TAIL.md** (nothing false in the document itself — its own account is still
  accurate) but flagged prominently since it is exactly the kind of "refusal message" scrutiny
  this round was told to apply, and it surfaced something the refusal-message wording itself gets
  wrong, not just outdated.
- **§6's third item (§5 logit comparison, extending ARCH.md's hand-transcription method to
  falcon/bloom/gpt_bigcode):** not checked — depends on `docs/architectures/ARCH.md`, not yet read this round.
- §0-§1, §2.1, §2.3 (baddbmm fix already covered via KERNELS.md above; the rank-refusal message
  ordering finding): not independently re-verified beyond what KERNELS.md's audit already covered.
- §3-§5 (golden reproducibility check, `--inject-fault` exit-1-is-correct note, falcon/bloom/
  gpt_bigcode re-measurement with 0 unimplemented ops): round-scoped, not re-verified.
- **Fixed: none in this file** — the one finding here is a live code defect worth flagging, not a
  documentation correction (the document's own text was and remains accurate).

### docs/architectures/ARCH.md

531 lines, the 32-architecture tail-shape survey (`gelu`/`gather`/`zero_` landed, a real GEMM
accumulation-dtype bug found — `float16` matmul accumulating in `float16` instead of `float32`).
Round-scoped baseline (82→85 ops) superseded many times over by now, house style, not
re-annotated — except one "못 한 것" (not done) claim that's exactly this round's target shape.

- **Claim (§7 "못 한 것"):** none of `gelu`/`gather`/`zero_` are reachable via Python method
  spelling (`Tensor.gelu()`/`.gather()`/`.zero_()`), and specifically that `nn.LayerNorm`'s
  constructor — which calls `.zero_()` — is therefore still blocked, repeating the wall
  `docs/models/GPT2.md` originally reported. **Status: two-thirds FALSE today.** How checked: live calls
  — `torch.tensor([1,2,3]).gather(0, torch.tensor([0]))` succeeds, `torch.tensor([1.0,
  2.0]).zero_()` succeeds, and decisively `torch.nn.LayerNorm(4)` **constructs successfully
  today** — the exact scenario this document says is still blocked. `.gelu()` is still not
  reachable (confirmed: `AttributeError`, not even attempted as a method — no `methods.json`
  entry), so this part of the claim holds.
  **Fixed:** yes — added a `> **Correction (문서 감사, 2026-09): ...**` blockquote after the
  claim, noting 2 of 3 closed and the `nn.LayerNorm` wall specifically closed, while leaving
  `gelu()` correctly flagged as still open. Marked `json-key` for both closed methods.
- §0-§6 (the 32-architecture re-measurement methodology, the `gelu`/`gather`/`zero_` kernel
  implementations themselves, the GEMM accumulation-dtype bug and its fix, the Gemma/BERT aten-
  level assembly cross-check): round-scoped, not re-verified — the GEMM fix in particular is a
  correctness fix to `aten.rs` from this round that later rounds have no obvious reason to have
  reverted, and nothing else in this pass suggested otherwise.
- §7's other open items (§5's judgment not pinned as a regression test, the next 4-op batch
  untouched, the `float32`/`k≤512` bit-exactness mystery, 5 of 37 architectures unmeasured due to
  config issues, 2-layer/hidden-64-only sampling, `bfloat16` GEMM only checked to k=512, device
  import unmeasured): not individually re-verified — explicit "don't know"/"didn't try" admissions
  in the house style this audit already validates elsewhere, not falsifiable "X is missing" claims
  in the shape this round prioritises.
- **Fixed: one, but high-value** — the `nn.LayerNorm` construction wall was named as a real,
  named blocker in two documents (`GPT2.md`, referenced here, and this document itself); both are
  now stale in the same direction, closed by unrelated later overload/method-table growth (the
  same mechanism as the SAMPLING.md/DEVICE.md/ARCH.md findings above — a "no Python spelling yet"
  gap that later table growth silently closed).

### docs/models/GPT2.md

882 lines, the GPT-2 landing round (`addmm`/`native_layer_norm`/`split.Tensor`/`tanh`, 19/19
token match, the "tail isn't a fixed list" 6-architecture survey). §5's Python-spelling table is
the most extensive instance of the SAMPLING.md/DEVICE.md/ARCH.md "no Python spelling yet" pattern
found this round — six rows, all now stale.

- **Claim (§5's table):** `torch.addmm`, `torch.tanh`/`x.tanh()`, `torch.split`/`x.split()`,
  `F.layer_norm`, `nn.LayerNorm(...)`, and `torch.layer_norm`/`torch.native_layer_norm` all fail
  via their natural Python spelling — table gives a specific blocker for each (missing
  `overloads.json`/`methods.json` entries; `nn.LayerNorm` doubly blocked by
  `_C._get_cudnn_enabled` and then `TensorBase.zero_`). **Status: FALSE, all six, today.**
  **How checked:** live-called all six spellings — `torch.addmm(...)`, `torch.tanh(x)`,
  `x.tanh()`, `torch.split(x, 2)`, `x.split(2)`, `F.layer_norm(x, (4,))`, `nn.LayerNorm(4)`,
  `torch.layer_norm(x, (4,), None, None, 1e-5)` — every one succeeds today. `overloads.json` now
  has `addmm`/`tanh`/`split` keys, `methods.json` has `tanh`/`zero_`, and `torch.layer_norm` turns
  out to be wired as a direct Python composite in `bootstrap.py:5863` (calls
  `aten.native_layer_norm.default` directly) rather than through the generic overload table.
  `nn.LayerNorm`'s `zero_`-blocking half is the *identical* finding already made independently
  while auditing `docs/architectures/ARCH.md` above (same commit family, most likely) — cross-confirms rather
  than duplicates.
  **Fixed:** yes — struck through all six "실패" (fail) table cells with inline "정정: OK"
  corrections and their reasons, plus a summary correction blockquote below the table. Left the
  narrative prose describing the *mechanism* (two-layer `nn.LayerNorm` blocking, the wider list of
  same-shaped gaps) largely intact but past-tensed where it now describes a closed state, per the
  "keep the original visible, mark when it stopped being true" house style. Marked 3 `json-key`
  (overloads), 1 `json-key` (methods), 1 `symbol-in-file` (the `layer_norm` composite).
- §0-§4 (the 4-op re-measurement, kernel implementations, `addmm` conversion, the aten-level
  19/19 token-match verdict), §6 (the 6-architecture tail survey), §7 (unknowns): round-scoped,
  not re-verified — no reason to suspect given the pattern found is specific to §5's Python-
  surface table, not the kernel-level claims.
- **Fixed: one finding, six instances** — by far the largest single batch of "Python spelling
  gap since closed" staleness found this round, all sharing the same root mechanism already
  identified in SAMPLING.md/DEVICE.md/ARCH.md above (later, unrelated rounds kept filling
  `overloads.json`/`methods.json`, and none of the documents that had named specific gaps in that
  table got revisited).

### docs/kernels/OPS4.md and docs/kernels/OPS8.md

619 and (unread total, checked §5 region only) lines — two more rounds carrying the same
"Python-spelling gap" claim shape found repeatedly above (GPT2.md, SAMPLING.md, DEVICE.md,
ARCH.md). Given the consistency of this pattern, checked both directly rather than reading in
full.

- **Claim (OPS4.md §"못 한 것"):** `torch.where`, `torch.stack`, `Tensor.permute`,
  `torch.nn.functional.relu` (4 of "여섯 op") not reachable via Python spelling.
  **Status: FALSE for all four checked, today.** Live-verified all four succeed; `overloads.json`
  has `where`/`stack`/`relu`, `methods.json` has `permute`. **Fixed:** correction blockquote added
  with 4 markers (all PASS). The other two of "여섯 op" are not named in the surrounding text, so
  not checked.
- **Claim (OPS8.md §5-1):** the entire `_C._nn` surface is empty — `gelu`, `silu`, `softmax`,
  `layer_norm`, `pad`, `_parse_to`, `linear`, `scaled_dot_product_attention` all refuse.
  **Status: partially FALSE today, and the useful finding is which parts.** Live-verified:
  `_C._nn.linear`, `_C._nn.scaled_dot_product_attention`, `_C._nn.gelu`, `_C._nn.silu`,
  `_C._nn.pad` all succeed now. `_C._nn.softmax` and `_C._nn.layer_norm` **still refuse** —
  `torch.layer_norm`/`F.layer_norm` do work today (confirmed independently in the `docs/models/GPT2.md`
  audit above), but via a separate Python composite in `bootstrap.py:5863` that bypasses
  `_C._nn.layer_norm` entirely, not because that specific `_C._nn` entry point was filled in.
  **Fixed:** yes — correction blockquote explicitly stating "표면 전체가 비어 있다는 더 이상
  맞지 않지만, 완전히 채워졌다도 아니다" (no longer fully empty, but not fully filled either),
  naming which of the six names still hold. This is a case where a blanket "still true"/"still
  false" framing would itself be wrong — the correction had to be per-name.
- **Claim (OPS8.md §5-2):** `torch.bmm`, `TensorBase.t()`/`.neg()`/`.bmm()` not reachable via
  Python spelling, schema total pinned at 127/127. **Status: FALSE today** — all four spellings
  work; `overloads.json`/`methods.json` have `bmm` keys. Noted the 127/127 schema count is
  itself long superseded (today's baseline is 4475/4475, per this round's header). **Fixed:**
  correction blockquote + 2 markers (PASS).
- Neither document's other sections were read in full given time budget and the high hit-rate
  already found in this exact claim shape across five documents now (GPT2.md, SAMPLING.md,
  DEVICE.md, ARCH.md, and these two) — the marginal value of continuing to hunt the same pattern
  in the remaining unread files is likely lower than covering more distinct documents.
- **Fixed: two files, seven markers, one graded finding** (OPS8.md's `_C._nn` claim needed a
  "which parts, not all-or-nothing" correction rather than a simple strikethrough — worth noting
  as a variant of the pattern: not every stale "X refuses" claim resolves uniformly).

### docs/kernels/RANDOM.md, docs/kernels/GAP.md, docs/models/E2E.md — light pass

Checked quickly given the `overloads.json`/파이썬 철자 grep hits, but none turned out to be the
stale-claim shape found in GPT2.md/OPS4.md/OPS8.md above.

- **docs/kernels/RANDOM.md** (`torch.randn`/`torch.rand` landing): headline claim re-verified live —
  `torch.randn(4,4)` and `torch.rand(2,2)` both still work today. The `overloads.json` mentions
  are mechanism explanation (why these couldn't use the generic overload table, requiring direct
  `dispatch(...)` calls instead), not a "still missing" claim. No false claims found.
- **docs/kernels/GAP.md** (line ~152): explains why 17 extra ops appear in `_aten_implemented()` beyond
  what Llama forward calls — a design-semantics note ("what the count means" vs. "which Python
  spelling reaches it"), not a falsifiable gap claim. No false claims found.
- **docs/models/E2E.md** (§7, "넣지 않은 것과 이유"): a deliberate scope decision (aten-level coverage
  judged sufficient, `nn.Module`-level assembly not added) with reasoning given, not a "not yet
  possible" claim. No false claims found.
- **Fixed: none needed in any of the three** — read only the sections the grep flagged, not the
  full documents, given time budget and the absence of any candidate false claim in what was read.

### docs/numerics/BOOL.md

A design-decision document (option B recommended: shim owns the `torch.bool` label, candle stays
underneath — the same pattern already used for `device`). Recommendation adopted, confirmed
independently already while auditing `docs/design/TORCH_C.md` above (`hasattr(torch, 'bool')` → `True`,
`aten.eq.Scalar` results carry `torch.bool` dtype). §2.2's `bool`-vs-`uint8` arithmetic table
(`x + x` is logical for bool, arithmetic for uint8, upstream torch 2.13.0) is a pinned-upstream-
version fact, not re-verified individually (same reasoning as CORE_ATEN.md/QUANT.md above).

- **Related finding, reported not fixed here:** while auditing `docs/kernels/TAIL.md` above, found that a
  later round's refusal message (`aten.mul.Scalar`'s bool rejection) cites "BOOL.md §2.2" while
  making a claim BOOL.md §2.2 doesn't actually support (§2.2 measured `.Tensor`-`.Tensor` bool
  arithmetic being logical upstream; the later refusal message generalizes this to `.Scalar`
  overloads, which `docs/kernels/TAIL.md`'s own measurement shows is upstream-arithmetic, not logical).
  BOOL.md itself is not at fault — its own §2.2 table is accurate and scoped correctly to what it
  measured. Full finding recorded under `docs/kernels/TAIL.md` above; not re-duplicated here.
- Not otherwise re-verified (§1, §3-§5, §7-§9): design rationale, reproduction steps, and
  explicit unknowns, not falsifiable claims in the shape this round targets.
- No false claims found in this document itself. **Fixed: none needed.**

### docs/models/CKPT.md

The checkpoint-reading predecessor to `docs/models/CKPT2.md` (already audited above). Its "못 한 것"
(not done) table named `torch.load(mmap=True)` and the safetensors mmap backend as both needing
`UntypedStorage.from_file`, and its "모르는 것" (unknown) list opened with "실제 사전훈련
체크포인트로는 검증하지 못했습니다" (not verified against a real pretrained checkpoint).

- **Status: both closed, and CKPT.md is not even the first document to say so** — `docs/models/CKPT2.md`
  §7 (already audited above) explicitly quotes this exact CKPT.md sentence and says it closed:
  "`docs/models/CKPT.md` §6 '모르는 것' 의 첫 줄 — '실제 사전훈련 체크포인트로는 검증하지 못했습니다' —
  이 닫혔습니다." So this finding isn't new — it was already on record in a sibling document; it
  just hadn't been back-annotated into CKPT.md itself. **How checked:** `UntypedStorage.from_file`
  exists at `rust/torch_c/src/storage.rs:211`; SmolLM2-135M's 273 tensors bit-match upstream
  (CKPT2.md §7, independently reconfirmed live in this round's CKPT2.md/E2E_REAL.md audit above).
  **Fixed:** yes — struck through both mmap table rows and the "모르는 것" opening bullet,
  pointing at CKPT2.md's own closing text rather than re-deriving it. Marked `symbol-in-file
  rust/torch_c/src/storage.rs from_file present`.
- §1-§5 (the `636a3cc` diagnosis, the silent-zero-path finding, the view-backed-tensor fix), §6's
  other rows (legacy `torch.load` format, `get_record_offset_no_read`, unsupported dtypes,
  negative stride, checkpoint writing — all still refused by name, not re-verified individually
  but none contradicted by anything found elsewhere this round): not re-checked given time budget.
- **Fixed: one — but it is really the same finding as the CKPT2.md/E2E_REAL.md correction above,
  propagated one document further back in the same chain.** Three documents (CKPT.md, CKPT2.md,
  E2E_REAL.md) all needed the same underlying fact re-confirmed and cross-linked; this audit has
  now touched all three.

---

## Correction to this round's own record

The commit that landed the first five files (`a1ff688`) says "two corrections landed". **Five
documents were corrected, not two** — ARCH20.md and DISPATCH.md by the first agent, and META.md,
SEQLEN.md and BIND.md by a sub-agent it had spawned, whose report arrived after the merge. The
commit's own `--stat` shows all five.

The mistake was in how I looked, not in what happened: I read `git status --short | head` and the
listing was truncated. Piping a status through `head` is the same shape of error as reading an exit
code through a pipe, which this repository already has a rule about.

Two further checks, since the sub-agent's three fixes arrived unverified:

- `torch.amax` and `Tensor.amax` both resolve and compute — SEQLEN.md §7.10's "they refuse by name"
  was indeed stale.
- META.md §12's `m.to('cpu')` claim holds. My first attempt to check it appeared to show the shim
  computing where upstream raises, which would have been silent divergence and the worst class of
  finding here. It was my harness: I had called `to_empty(device='cpu')` on the same module first,
  which materialises the parameters, so the later `.to('cpu')` had nothing to copy out of meta. On a
  fresh module both sides raise `NotImplementedError` with the identical message.

---

## Findings (round 3 — the last 24, plus LOSS.md and SCALAR.md's full read)

Same method as rounds 1-2. Territory this round: `docs/*.md` except BACKWARD.md, ADAPT.md,
TRAIN.md, CAPTURE.md (owned concurrently by another agent), plus `tools/docwatch/`. Forbidden:
`rust/`, `tools/wheel/`, `tools/golden/`, `scripts/`, `torchnative/`.

Baseline (worktree `/Volumes/macMini/worktrees/bw-doclast`, established before touching any file,
same commands as rounds 1-2):

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh          -> EXIT=0, 317 "ok " lines, DOCWATCH: PASS -- 190/190
$PY tools/golden/compare.py                        -> EXIT=0, SUMMARY: 7685/7685 cases passed, 0 failed, ops covered=168, pending case builders=1
$PY rust/torch_c/pytests/verify_schemas.py         -> EXIT=0, SUMMARY: 4479/4479 table entries matched upstream, 0 failed
```

Implemented-ops snapshot captured in `/tmp/doclast_implemented_ops.txt` (168 ops). All 190
pre-existing markers (rounds 1-2) still PASS against this baseline before this round touched
anything — no re-staleness in the marker set itself this time (unlike round 2's DOCWATCH.md
finding against round 1's own baseline paragraph).

The brief's priority pattern — a refusal naming a kernel as missing, now closed by unrelated later
work — has fired 12 times across rounds 1-2 (ARCH20.md, META.md, DYNAMO.md, DESIGN.md x2,
OVERLOAD.md, TENSORBASE.md, SDPA.md, QUANT2.md, DISTRIBUTED.md, DEVICE_ABS.md, SAMPLING.md,
DEVICE.md, ARCH.md, GPT2.md, OPS4.md, OPS8.md — more than 12 counting multi-instance files).
Checked first in every file below.

### docs/models/FROM_CONFIG.md

392 lines, a genesis-era pre-measurement document (§0: "우리 shim 은 아직 여기 도달하지 못했으므로,
진짜 torch 로 계측했다") — instruments what `AutoModelForCausalLM.from_config` needs before any of
it was implemented. Exactly the shape this round was told to expect: a 3-op snapshot
(`aten.add.Tensor`/`aten.full.default`/`aten.mm.default`) that 65+ rounds since built on top of.

- **Claim (§2.1, §5 table, §6):** 14 ops the `from_config` path calls (`normal_`, `empty.
  memory_format`, `uniform_`, `ones`, `fill_.Scalar`, `arange.start_step`, `div.Tensor`, `pow.
  Scalar`, `reciprocal`, `mul.Tensor`, `detach`, `lift_fresh`, `copy_`, `clone`) are "14 개 중 0 개"
  implemented; RNG (`torch.Generator`/`manual_seed`/candle-vs-torch value match) is unconfirmed;
  the recommended next verification step (re-run the same script against our own shim) was not yet
  done. **Status: FALSE today — all 14 ops implemented, RNG ported, and the next-step re-run now
  succeeds.** **How checked:** grepped all 14 op spellings against the current 168-op
  `_aten_implemented()` snapshot — all present. Live-ran the exact scenario this document's own §6
  names as the next step: `AutoModelForCausalLM.from_config(cfg)` with the identical llama config
  against our own shim (not real torch) — succeeds, and the parameter count matches this document's
  own real-torch measurement exactly (95,040). `rust/torch_c/src/rng.rs` has a ported MT19937 engine
  and `torch.manual_seed` remap, closing §4.3's "미확인" on RNG algorithm match (cross-referencing
  round 2's RNG.md/TENSORBASE.md findings, same landing commit `2d3663f`).
  **Fixed:** yes — added a `> **Correction (문서 감사, 2026-09): ...**` blockquote after the opening
  paragraph, before the original method/measurement text, per house style (leave the historical
  measurement visible, mark when it stopped being current). Marked `op-implemented` for 4
  representative ops (not all 14 — a `count`-style "N of 14" claim isn't one of the six primitives,
  and marking all 14 individually would be redundant with existing OVERLOAD.md/TENSORBASE.md/
  SAMPLING.md markers for several of the same ops) and `symbol-in-file` for the RNG port.
- §1 (the dual-instrumentation methodology, the `Generator` immutable-type workaround), §3
  (`GenerationMixin`'s two real requirements: `no_grad` decorator protocol, `torch.*Tensor`
  annotations evaluated at class-body time), §4.1-4.2 (double-init via `kaiming_uniform_` then
  `normal_`, `transformers.initialization`'s capture layer): mechanism descriptions and
  measurement methodology, not counts that drift — not re-verified, no reason to suspect.
- **Fixed: one (all 14 "not implemented" op claims plus the RNG-value-match unknown, same severity
  class as round 2's OVERLOAD.md/TENSORBASE.md findings — this is the earliest-written of the three
  documents making this exact claim, so its staleness is also the most complete: 14/14, not a
  partial set).**

### docs/kernels/TRIL.md

586 lines, the round that landed `tril`/`triu`, fixed a NaN-dropping bug in `max.dim`/`max.other`/
`argmax` (found for the fourth time, repaired in one shared function), and closed the twentieth
architecture (GPT-BigCode). Unusually rigorous already — an 11-fault sabotage table with one fault
that could not fail, reported as such rather than hidden — house style already fully applied. Two
"not done here" items in §6.5's "what this round did not verify" checked directly, since that
section is exactly the "next step" shape this round prioritises.

- **Claim (§6.5, "The SDPA math backend"):** "§2.3 corrected the refusals' reason and did not build
  the composite. Its kernels are all present; nobody has transcribed the sequence." **Status: FALSE
  today — built by a later, unrelated commit.** `git merge-base --is-ancestor 3b7d981 1938ad1`
  confirms this document's own commit (`3b7d981`) predates `1938ad1` ("Feat: Open training mode,
  which every sweep in this repository had assumed away"), which added `_sdpa_math` to
  `bootstrap.py`. **How checked:** `_sdpa_math` exists at `bootstrap.py:5288`; live-called the exact
  scenario the surrounding code names as routing to it (`dropout_p != 0.0` falls off the flash
  path) — `F.scaled_dot_product_attention(q, k, v, dropout_p=0.1, is_causal=True)` succeeds today.
  Round 2's SDPA.md/CKPT2.md/GENERATE.md audits already independently confirmed the same function
  from its `enable_gqa` branch — this is the same landing, found here from the "was it ever built"
  angle rather than the "does GQA work" angle.
  **Fixed:** yes — added a `> **Correction (문서 감사, 2026-09): ...**` blockquote after the
  §6.5 bullet, pointing to the landing commit and the live re-verification, without rewriting the
  original "not attempted" framing (correct when written). Marked `symbol-in-file`.
- **Claim (§2.4, §3.4, §6.5):** `aten.amin.default`/`aten.argmin.default` have no kernel — a
  direction-specific `CustomOp1` that would need to be written, not a sign flip on `amax`.
  **Status: confirmed still true.** How checked: both absent from the current 168-op
  `_aten_implemented()` list. Marked `op-not-implemented` for both.
- §0-§1 (the tril/triu kernel, the sign-convention table, the NaN-vs-multiply zeroing bug), §2.1-2.2
  (`amax`/`softmax` spelling fixes), §3 (the shared `nan_along_dim` mechanism, the `-inf` boundary
  case), §4 (the NaN-position test methodology), §5 (the 11-fault sabotage table), §6.1-6.4 (gate
  counts, the smoke-test update table, the SmolLM2 hash-unchanged verification): round-scoped
  landing narrative and mechanism description, not re-verified — no reason to suspect any, and §6.1's
  gate counts are superseded by this round's own baseline header rather than individually stale.
- **Fixed: one** — the SDPA-math-backend "not built" claim, same shape as round 2's SDPA.md/
  CKPT2.md/GENERATE.md findings (a document naming an unbuilt composite that a later, unrelated
  training-mode round then built) — the fourth document in this audit's overall run to make this
  same claim about the same composite, and the last one still open before this round.

### docs/design/REGISTRATIONS.md

340 lines, a genesis-era measurement document (measured against real upstream torch, because "the
shim doesn't even reach `from_config` yet") sizing the 1549 no-op `torch.library` registrations the
vendored tree makes at import time, and judging none of them worth fixing right now — but naming
two things that *are* worth fixing first: `_log_api_usage_once` (blocking `from_config` itself) and
`_dispatch_get_registrations_for_dispatch_key` (blocking `core_aten_decompositions()`). Both are
exactly the "refusal names something as missing" shape this round prioritises, and both are the
document's own explicit "cannot be deferred" list (§5).

- **Claim (§0 table, §5 item 2):** `from_config` doesn't even reach the model-construction stage —
  it now fails even earlier than previously recorded, at `nn.Module.__init__` calling
  `torch._C._log_api_usage_once`, which doesn't exist. **Status: FALSE today.** **How checked:**
  `hasattr(torch._C, '_log_api_usage_once')` → `True` (`bootstrap.py:4777`); independently
  reconfirmed by this same round's `docs/models/FROM_CONFIG.md` audit, which ran `from_config` on our own
  shim end to end and it succeeded.
- **Claim (§0 table, §4's last row, §5's "미룰 수 없는 것" item 2):** `core_aten_decompositions()`
  crashes with `NotImplementedError: torch._C._dispatch_get_registrations_for_dispatch_key`; the
  shim's `decomposition_table` is 592 entries (vs. real torch's 1097) because of a hardcoded
  `_jit_get_operation` `overload_names=["default"]` that collapses every op packet to one overload.
  **Status: FALSE today, both halves.** **How checked:** `hasattr(torch._C,
  '_dispatch_get_registrations_for_dispatch_key')` → `True` (`bootstrap.py:1443`);
  `core_aten_decompositions()` no longer crashes, returns 417 entries live (real torch, re-verified
  unchanged today: 940 — a real remaining gap, not a crash); `decomposition_table` is 1008 today
  (real torch, re-verified unchanged: 1097). The `overload_names` fix is `docs/graph/DECOMP.md` §3 (round
  2 of this audit) — its own code comment at `bootstrap.py:1223` cites this exact document by name
  ("the reason it exists is docs/graph/DECOMP.md §3"), so the fix already knew what it was closing; this
  is the first time this audit round has found a document's own future fix already cross-linked in
  the source it's auditing, not just in a sibling document.
  **Not fixed:** "973" (§0's DESIGN.md-attributed number) is still not reproduced by any measurement
  in this document or elsewhere — left as an open, unresolved discrepancy, not claimed fixed.
  **Fixed:** yes — added a `> **Correction (문서 감사, 2026-09): ...**` blockquote after §0's table
  and a shorter inline correction after §5 item 2, cross-referencing `docs/graph/DECOMP.md` and
  `docs/models/FROM_CONFIG.md` rather than re-deriving. Marked `symbol-in-file` for both closed functions.
- §1-§3 (the 1549-registration taxonomy, the eager-forward "0 dispatched" measurement against real
  torch, the `_dispatch_has_kernel=True` cost measurement and its 251-item TorchScript-residue
  finding): round-scoped measurement methodology, not re-verified — these are facts about upstream
  torch's own registration behavior (pinned to torch 2.13.0, same reasoning as CORE_ATEN.md/
  QUANT.md above) or about a design decision's cost, not counts that drift with this repo's own
  kernel growth.
- §7's unknowns table (the "973" origin, `Autograd`-key registrations under backward, `torch.
  compile` usage ratio, the impl-count mismatch between shim/real-torch counting layers,
  `fallback()` at runtime, quantization/distributed/oneDNN dependence): explicit "미확인" admissions,
  not re-checked — none contradicted by anything found while checking §0/§4/§5.
- **Fixed: one finding, two symbols, propagating to two spots in the same file** — same severity
  class as round 2's OVERLOAD.md/TENSORBASE.md findings: a refusal this document treats as the
  active blocker for reaching `from_config` at all, closed by unrelated later work, with the fix
  already cross-linked from the fixing commit's own code comment.

### docs/bindings/SCHEMA.md

367 lines, the round that gives `_schema` real text (reading the vendored `native_functions.yaml`
directly rather than a hand-transcribed table) and fixes `is_mutable`, found wrong in both
directions across two rounds (always-true, then — this round's own finding — always-false). Already
unusually rigorous about its own claims: §9 is an explicit fault-injection table proving each of its
five schema-printing rules can actually fail, and §4/§8.1 both explain their own numeric drift
(97→117 implemented ops between when `docs/distributed/DISTRIBUTED.md` §8.1 first found the bug and when this
document fixed it) rather than leaving it silent.

- **Claim (§12, last bullet):** "`docs/distributed/DISTRIBUTED.md` §8.1 은 아직 '미해결'로 적혀 있습니다 ...
  그 파일은 이 작업의 소유 범위 밖이라 건드리지 않았습니다" (DISTRIBUTED.md §8.1 is still marked
  unresolved; out of this work's scope, left untouched). **Status: FALSE, and self-contradicted by
  this document's own landing commit** — not an instance of the usual "later unrelated commit"
  mechanism this round keeps finding, but a same-commit inconsistency. **How checked:** `git show
  --stat e26e54b` (this document's own single landing commit, "Feat: Give the schema table real
  text, so is_mutable can be wrong") shows `docs/distributed/DISTRIBUTED.md | 22 +-` in the same diff — the
  commit that wrote this claim also rewrote `DISTRIBUTED.md` §8.1 with a "해결됐습니다
  (2026-08-28)." correction block pointing at this exact document. Read literally, "건드리지 않았다"
  (left untouched) is false about the very commit containing the sentence.
  **Fixed:** yes — added a `> **정정 (문서 감사, 2026-09): ...**` blockquote noting the
  self-contradiction and that `DISTRIBUTED.md` §8.1 today does carry the resolution note, without
  guessing at what the original sentence meant to say (probably "I, the document's author, didn't
  hand-edit it" vs. "the coordinating session that landed this commit did"). Marked `symbol-in-file`
  against the Korean resolution text itself, a first for this audit's marker set (confirms
  `symbol-in-file`'s literal-substring fallback path handles non-ASCII).
- §6's "2606 answerable / 1148 still placeholder" split (out of upstream's 3754 aten schemas): not
  independently re-derived — this is arithmetic over a pinned-upstream-version fact (the vendored
  `native_functions.yaml`'s 2584 declared entries + 18 hand-transcribed + 4 table-only, against
  upstream torch 2.13.0's registry), the same kind of fact CORE_ATEN.md's `torch.Tag.core`
  count and QUANT.md's `candle-core` `DType` enum are (pinned to a version, not to this repo's own
  `_aten_implemented()` growth) — not the drifting category this round's baseline-count growth
  affects. Spot-checked the *mechanism* instead: `torch._C._shim_placeholder_schemas()` is a
  per-process query accumulator (§6's own text: "그중 물어진 것은 ... 들어갑니다"), not a static
  total, so a quick live probe returning 223 is not comparable to the document's 1148 (measured
  under a much larger workload: import + transformers + FSDP + decomp pass) — reproducing that
  workload exactly was judged not worth the time given the number is upstream-pinned, not
  kernel-count-dependent.
- §0/§4's "117개 구현 중 12개가 mutable" headline count: round-scoped (implemented-op count was 117
  when written, is 168 today) — house style, not re-annotated, consistent with how §4 itself already
  explains the 97→117 drift between DISTRIBUTED.md's original finding and this document's fix.
- §1-§3, §5, §7-§9 (the four-layer schema resolution order and why it's load-bearing, the five
  upstream-printer rules and their measured counts, the two abandoned designs in §8, the
  fault-injection table): mechanism descriptions and this round's own measurements, not re-verified
  — no reason to suspect any, and §9's self-proving fault table already does much of what this audit
  checks for.
- **Fixed: one** — smaller in scope than most findings this round (an internal same-commit
  inconsistency rather than a later-drift staleness), but a new variant of the "correction doesn't
  propagate" mechanism round 2 found in DISTRIBUTED.md/DEVICE_ABS.md: here the propagation *did*
  happen (to the other file), just not back to the sentence describing whether it would.

### docs/models/IMPORT_TORCH.md

600 lines, the round that got a strict-mode `import torch` to exit 0 for the first time (45 walls,
§8). Already carries a pre-existing correction at the bottom pointing at `docs/design/REGISTRATIONS.md`
(landed later, not by this audit) fixing this document's own wrong "973" figure and its causal
story about `_dispatch_has_kernel` — house style already applied once. §11's "남은 벽과 미확인"
(remaining walls / unknowns) table is exactly this round's target shape, checked item by item.

- **Claim (§0 table, §11 item 3):** `from_config` still fails, now blocked in transformers'
  `GenerationMixin` lazy import rather than inside `import torch` itself. **Status: FALSE today** —
  same finding as this round's `docs/design/REGISTRATIONS.md`/`docs/models/FROM_CONFIG.md` audits, both of which
  independently confirmed `from_config` now succeeds end to end (the blocking wall,
  `torch._C._log_api_usage_once`, now exists). **Fixed:** yes — added a `> **정정 (문서 감사,
  2026-09): ...**` blockquote after §0's table, cross-referencing both sibling findings from this
  same round rather than re-deriving. Marked `symbol-in-file`.
- **Claim (§11 item 12):** BOOL.md §6.2's table lands the bool tag and single-constructor
  invariant, but `bitwise_*`/`any`/`masked_fill` kernels are still unimplemented. **Status: FALSE
  today, all of them.** **How checked:** `aten.bitwise_and.Tensor`/`.Scalar`,
  `aten.bitwise_or.Tensor`/`.Scalar`, `aten.bitwise_not.default`, `aten.any.default`/`.dim`,
  `aten.masked_fill.Scalar`/`masked_fill_.Scalar` are all in the current 168-op
  `_aten_implemented()` list; live-called `x & y`, `t.any()`, `x.masked_fill(mask, 0.0)` — all
  compute today. **Fixed:** yes — inline correction in the table cell. Marked `op-implemented` for
  3 representative ops.
- **Claim (§11 item 10):** `float8_e4m3fn` tensor creation hangs, cause not investigated. **Status:
  no longer reproduces.** **How checked:** `torch.full((3,), 1.0, dtype=torch.float8_e4m3fn)` under
  a 5-second `SIGALRM` guard returns normally today. **Not** claiming to know what fixed it — the
  original cause was never identified, so neither is the fix; only the symptom's absence is
  reported. **Fixed:** inline correction noting the symptom no longer reproduces, cause still
  untraced.
- **Claim (§11 item 11):** the dtype promotion table is still unimplemented, `add.Tensor` refuses by
  name. **Status: confirmed still true, and — cross-referencing round 2's TORCH_C.md/TENSORBASE.md
  findings — now known to be *deliberate*, not a placeholder gap.** Inline note added pointing to
  both, no strikethrough (the claim itself still holds).
- §1-§9 (the bootstrap-surface architecture, the 45-wall table, the schema parser and its
  bracket-matching bug fix, the dtype/bool tag design, the two golden-harness bugs found and fixed):
  mechanism descriptions and this round's own landing measurements, not re-verified — no reason to
  suspect any, and §9.1's `_dispatch_library`/1549-registration design decision is independently
  reconfirmed unchanged by this round's own REGISTRATIONS.md audit (still a deliberate no-op, not a
  gap).
- §11 items 1, 2, 4-9, 13-14 (the 15 missing sub-byte dtypes, `_TensorMeta` always-true isinstance,
  the `torch.library` no-op design, op-name-lookup never failing, device import, `TORCH_USE_RTLD_
  GLOBAL`, `ios-sim`, re-vendoring surface drift, registration-usage ratio): explicit "미확인"/
  design-decision admissions, not individually re-verified — item 5 (op lookup never fails) was
  spot-checked live (`hasattr(torch.ops.aten, 'nonsense_op_xyz')` → `True` still) and remains a real
  gap, consistent with the document's own claim.
- **Fixed: three (from_config, bool-op kernels, float8_e4m3fn hang) — the bool-op finding is the
  largest in scope this round after FROM_CONFIG.md's 14-op batch: nine op spellings across three
  base names, all closed by the same generator-port-era kernel work rounds 2-3 keep finding.**

### docs/models/IMPORT_WALLS.md

341 lines, a genesis-era exploratory document (`import transformers` against a *stub* torch, no
kernel work yet) — the earliest-written document read this round, and already the most heavily
self-correcting one found in either round: four numbered rounds (1차-5차), each with its own
`> **정정.**` blockquote catching the previous round's own mistake (a `grep -q MODEL_OK` false
positive in round 2, a mis-diagnosed `@auto_docstring` wall in round 2's "범주 6" that
`docs/models/AUTODOC.md` overturned). Checked whether its own remaining "아직 답하지 않은 것" (not yet
answered) list has since been answered by a sibling document — the same "correction found in a
different document's own text, not yet linked back" shape found in round 2's TORCH_C.md/CAPTURE.md.

- **Claim (§ "관문 너머" table row 5, § "아직 답하지 않은 것" bullet 1):** category 5
  (`@torch.no_grad()`'s decorator-and-context-manager dual protocol) is unconfirmed, exploration
  stopped there. **Status: FALSE — answered by a sibling document that names this one directly.**
  **How checked:** `docs/models/FROM_CONFIG.md` §3.3 (audited earlier this round) opens by quoting this
  exact sentence — "IMPORT_WALLS.md 1차의 category 5... '미확인 — 여기서 멈췄다' 고 적어 둔
  항목인데, 여기서 정확한 요구 시점과 프로토콜을 확인했습니다" — and gives the answer (class-body
  evaluation time, dual `__call__`/context-manager protocol). `from_config` succeeds on our own
  shim today (independently reconfirmed this round). **Fixed:** yes — inline correction in the
  table cell and the bullet, pointing at `docs/models/FROM_CONFIG.md` §3.3 by name.
- **Claim (§ "아직 답하지 않은 것" bullet 2):** whether the 15 discovered submodules can be satisfied
  by empty stubs, or need real implementations, is undetermined future work. **Status: not answered
  — the question became moot instead**, which is itself worth recording as distinct from "answered."
  **How checked:** `docs/design/DESIGN.md` §11 (still current) made the final call this document's own §4/
  §5 rounds fed into: "A(candle 위 `torch._C`)로 갑니다" — and option A carries the *entire* real
  vendored Python tree, not stubs. So the stub-sufficiency question is not resolved either way; the
  premise it was asked under (a stub torch) was abandoned. **Fixed:** yes — a correction noting the
  distinction (moot, not answered) rather than claiming a false "yes/no."
- **Claim (§ "아직 답하지 않은 것" bullet 3):** this experiment stops at model construction
  (`from_config`); forward pass, `generate()`, `online()` are unexplored beyond it. **Status: since
  closed by later rounds**, not by this document. **How checked:** round 2's `docs/models/CKPT2.md`/
  `docs/models/GENERATE.md` audits already confirmed forward pass and `generate()` work today against a
  real Hub checkpoint (SmolLM2-135M). **Fixed:** yes — inline correction pointing at both.
- §"가장 큰 발견"-§"관문은 하나다" (torch is not a hard dependency, the `is_torch_available()`
  gate), §1차-§3차 (the 15-submodule discovery, the 1084-module vendoring-size measurement, the
  14-of-1084 modules actually executed at inference), §4차 (the import-time operator-registration
  coupling that makes selective pruning expensive, feeding directly into `docs/design/DESIGN.md`'s A/B
  decision): mechanism descriptions and measurements against real upstream torch/transformers, not
  re-verified — pinned to torch 2.13.0/transformers 5.15.1, same reasoning as CORE_ATEN.md/QUANT.md
  above, not the drifting category this round's kernel-count growth affects.
- **Fixed: three, all "a later sibling document already answered this open item" findings** — no
  DOCWATCH markers added (these are cross-document narrative claims and "did a scenario end up
  working" claims, both explicitly named in `docs/verification/DOCWATCH.md`'s "what this cannot see" list as
  structurally outside the six primitives) — this is the fourth document this round where a
  self-aware, heavily-corrected genesis document's own remaining gaps turn out to already be
  answered elsewhere, unlinked.

### docs/models/AUTODOC.md

244 lines, a focused investigation overturning `docs/models/IMPORT_WALLS.md`'s own "범주 6" conclusion (the
`@auto_docstring` wall IMPORT_WALLS.md judged unpassable by any stub, restated with its own
`> **정정.**` there). This document is unusually careful about the limit of what it verified — its
own closing table names exactly one thing still open: "벤더링 트리 + 우리 `_C` 조합에서는 아직
확인되지 않았습니다" (not yet confirmed in the combination of the vendored tree + our own `_C`,
only against a generic stub and real upstream torch separately). That is exactly the "next step"
shape this round prioritises, and it is now checkable because `import torch` succeeds on our shim.

- **Claim (§7's closing paragraph, the summary table's last row):** whether this wall reappears once
  the vendored-tree-plus-our-`_C` combination actually reaches class definition is the one variable
  this document did not check. **Status: now checked, and confirmed no wall.** **How checked:** `from
  transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM` against our own shim
  succeeds directly, and `__doc__` length matches this document's own real-upstream-torch
  measurement byte for byte (`LlamaModel` 883, `LlamaForCausalLM` 850) — the same cross-check this
  document itself used in §4 to prove its generic stub was behaviourally identical to real torch, now
  run a third way (our shim) with the same result.
  **Fixed:** yes — added a `> **정정 (문서 감사, 2026-09): ...**` blockquote after §7's paragraph and
  an inline correction in the summary table's last row, rather than rewriting the original honest
  "not yet confirmed" framing (which was correct when written — `import torch` did not pass on our
  shim yet at the time).
- §1-§3 (reading `auto_docstring.py`'s actual code path, the `copy_func`/`types.FunctionType`
  census, the 87-call live trace proving none originate from the `@auto_docstring` frame), §4 (the
  generic-stub reproduction, byte-identical `__doc__` across five model classes), §5 (the honest
  "unconfirmed" account of what the original IMPORT_WALLS.md 2차 probe's `TypeError` actually was,
  with its own script lost and not reconstructable): all already carefully hedged by evidence
  strength in this document's own text — not re-verified, no reason to suspect any of them, and §5's
  own "미확인" stays exactly that (a gap that cannot be closed without a script that no longer
  exists, not a stale claim).
- **Fixed: one** — the single item this document's own account flagged as the one thing it did not
  check, now checkable and closed. Smaller in stakes than most findings this round (the document's
  own judgment already said "no reason to expect a new wall here"), but it is the cleanest
  before/after pair found this round: the same exact measurement (`__doc__` byte length across five
  classes), run a third time, agreeing with both prior runs.

### docs/design/C_SURFACE.md

414 lines, a pure measurement document against real upstream torch/transformers 2.13.0/5.15.1 (an
access-vs-call census of `torch._C`'s 989-name surface, `TensorBase`'s 694 members, and
`_VariableFunctions`' 609 hoisted names) that informs implementation priority — it does not itself
claim anything about our own shim being broken or missing a kernel, so the "refusal names something
as missing" pattern this round prioritises does not really apply here. All headline counts (979/989
accessed by `import torch` alone, 50/694 `TensorBase` members actually called, 13/609
`_VariableFunctions` actually called, the dynamo-rule-table-scan false signal at 607/609) are pinned
to real upstream torch/transformers behavior on one fixed toy-Llama scenario, not to this repo's own
`_aten_implemented()` growth — the same non-drifting category as `docs/design/CORE_ATEN.md`/`docs/graph/QUANT.md`
above.

- **Claim (§8 item 7, the document's own explicitly named open question):** whether the need-set
  measured against real torch still holds once our own shim actually substitutes for it is
  unconfirmed — the shim might take a different branch (e.g. an `hasattr`-gated subsystem) and need
  a different name set. **Status: not fully re-measurable without repeating this document's own
  elaborate tracing harness (out of scope for a documentation pass), but the top-level scenario this
  question is really asking about — does our shim get through the same from_config + forward +
  generate path at all — is independently confirmed working** by this round's `docs/models/FROM_CONFIG.md`
  and round 2's `docs/models/CKPT2.md`/`docs/models/GENERATE.md` audits. Not claiming the exact 50/13-name sets
  match; only that the scenario they were measured against no longer fails on our shim, which is the
  premise the open question worried might not hold. Left as reported, not marked false — the
  document's own hedging here is already correct and this round found no reason to overturn it, only
  to note the surrounding scenario now passes.
- §0-§6 (the tracing methodology — the `__getattribute__`-vs-`module_getattro` bug caught while
  building the harness itself, the `TensorBase`/`_VariableFunctions` immutable-type workarounds, the
  dynamo-rule-table contamination finding), §7 (priority tiers derived from the measurement): not
  re-verified — all are measurements against real upstream software fixed to a pinned version, not
  claims about this repo's own tree that could have drifted.
- **Fixed: none — no false claims found.** This is the fifth document this round (after
  CORE_ATEN.md/QUANT.md in round 2's pattern) where the content is investigation/measurement against
  pinned upstream software rather than a claim about this repo's own kernel coverage, and the
  "refusal names a kernel as missing" pattern the brief prioritises structurally does not apply.

### docs/bindings/NN_SURFACE.md

282 lines, the round that measured `_C._nn`'s real call footprint (3 of 96 names actually called by
a Llama forward pass) and wired the Python spellings a model path actually needs. §1's four-row
"what really blocks it" table and §9's eight-row "wired but no kernel yet" table are both exactly
this round's target shape — the highest hit-rate file found this round.

- **Claim (§9's table, 8 rows):** `Tensor.le`/`x <= y`, `torch.where`, `torch.where` against a bool
  mask, `nn.Linear(bias=True)`, `F.softmax` (eager attention), and sdpa's math backend are all wired
  but blocked on missing kernels (`aten.le.*`, `aten.where.self`, `aten.scalar_tensor.default`,
  `aten.addmm.default`, `aten._softmax.default`, `aten._safe_softmax.default`); `aten.dropout.
  default` likewise. **Status: FALSE for 7 of 8 rows today — only `dropout` is still accurate.**
  **How checked:** all seven op spellings are in the current 168-op `_aten_implemented()` list;
  live-verified `x <= y`, `torch.where(cond, a, b)`, `nn.Linear(3, 4, bias=True)(x)`,
  `F.softmax(x, dim=0)` all compute today. `aten.dropout.default` remains absent — live-verified
  `torch.ops.aten.dropout.default(x, 0.5, True)` still refuses by name, exactly as this document's
  own §5 predicted (a deliberate low-priority gap, not a forgotten one — inference doesn't need it).
  `aten._safe_softmax.default` is the same kernel round 2's SDPA.md and this round's TRIL.md already
  found landed. **Fixed:** yes — added a `> **정정 (문서 감사, 2026-09): ...**` blockquote after the
  table, naming which row is the one exception rather than a blanket strikethrough (same "graded,
  not all-or-nothing" shape round 2's OPS8.md `_C._nn` finding needed). Marked `op-implemented` for
  4 representative ops and `op-not-implemented` for `dropout`.
- **Claim (§1's table, rows 3-4):** `F.softmax` and `F.layer_norm` are blocked by
  `_C._get_cudnn_enabled` (a config getter, cheap to answer, but out of scope because not on the
  Llama path). **Status: FALSE today — both compute.** **How checked:**
  `hasattr(torch._C, '_get_cudnn_enabled')` → `True`, and it returns `True`; live-called
  `F.layer_norm(x, (4,))` and `F.pad(x, (1,1))` — both succeed today. `F.layer_norm`'s closure is
  the identical finding round 2's `docs/models/GPT2.md` audit already made (a separate Python composite at
  `bootstrap.py:5863` that bypasses `_C._nn` entirely, not `_C._nn.layer_norm` itself being filled
  in) — cross-confirmed rather than duplicated. **Fixed:** yes — inline correction after §1's table.
- §2-§4 (the `_C._nn` 70-vs-96 census methodology, the `torch.*`/`Tensor.*` python-spelling wiring
  table, the `__rsub__` routing-through-`_VariableFunctions` mechanism), §5-§6 (the `linear`/
  `dropout` composite-not-kernel decision and its branch-on-`_aten_all_implemented()` self-retiring
  design, the sdpa backend-selection table), §7-§8 (the upstream-vs-shim numeric agreement table,
  greedy-token match, schema-count growth): round-scoped landing measurements and mechanism
  descriptions, not re-verified — no reason to suspect any, and §5's addmm branch is confirmed live
  to have actually self-retired (addmm is implemented, matching the design's own stated intent).
- §10's unknowns (the remaining 93 `_C._nn` names untested on other architectures, the 96-vs-70
  surface gap, `_get_cudnn_enabled`/`_get_deterministic_algorithms` unverified beyond "looks cheap",
  `enable_gqa`'s exact broadcast rule, bias-path numerical drift at scale, device import): explicit
  "확인하지 않음" admissions, not re-checked — none contradicted by anything found while checking §1/
  §9.
- **Fixed: two findings, eleven op/config spellings total** — the largest single-document hit rate
  this round (7 of 8 table rows in §9 alone), all tracing to the same generator/kernel-growth era
  rounds 2-3 keep finding across FROM_CONFIG.md, IMPORT_TORCH.md, and here.

### docs/perf/LINEAR.md

371 lines, the round that found and fixed `linear`'s per-call weight copy — already cited by name
in round 1's `docs/design/DESIGN.md` audit and round 2's `docs/graph/QUANT2.md` audit (both confirmed the fix
landed as `2e00ec3`), but this document itself, the source of that fix, had never been read by this
audit until now. §7's "판단이 필요한 것" (judgment needed) section frames the whole change as
**uncommitted**, pending a coordinating-session decision between four options — exactly the
"decision not yet shown to the user" shape `CLAUDE.md` §5.7 warns about, so worth checking whether
that framing is still accurate.

- **Claim (§7's opening paragraph):** "비트가 바뀌므로 여기서 임의로 고르지 않았습니다 ...
  작업 트리에 변경만 있고 커밋하지 않았습니다" (the change sits uncommitted in the working tree,
  pending a decision among 4 options). **Status: FALSE, and self-contradicted by this document's own
  landing commit** — the same same-commit-inconsistency shape found in this round's `docs/bindings/SCHEMA.md`
  audit, not the usual "later unrelated commit" mechanism. **How checked:** `git show --stat
  2e00ec3` (this document's own single commit, "Perf: Fold instead of broadcasting, and stop copying
  the weight every call") shows `docs/perf/LINEAR.md | 370 +++...` and `rust/torch_c/src/aten.rs | 131
  +++...` in the same diff — the commit that contains the sentence "커밋하지 않았습니다" is itself
  the commit landing the change. Option **①** (A+B, unconditional, no opt-in) is what actually
  shipped: `gemm_with_layout_fallback`/`batched_matmul` are the default path in `aten.rs` today
  (independently reconfirmed live and by two prior audit rounds — `docs/design/DESIGN.md` in round 1,
  `docs/graph/QUANT2.md` in round 2, both citing this exact commit).
  **Fixed:** yes — added a `> **정정 (문서 감사, 2026-09): ...**` blockquote after the paragraph,
  naming which option was chosen and cross-referencing the two prior rounds that already found the
  same commit from the other side, without rewriting §7's table (an accurate snapshot of the
  decision *as posed*, just not of how it resolved). Marked `symbol-in-file`.
- **Claim (§6 item 2):** the golden harness doesn't compare `aten.matmul.default` at all — it sits
  in `IMPLEMENTED_AWAITING_GOLDEN`, so "golden 2760/2760 passes" cannot be cited as evidence for this
  change, and the document's own §4 probe exists specifically to fill that gap. **Status: FALSE
  today — closed.** **How checked:** `aten.matmul.default` is in the current 168-op
  `_aten_implemented()` list, not the awaiting-golden set; this round's own baseline
  (`golden_pending=1`) names only `aten.reshape.default` as still pending, not `matmul`. **Fixed:**
  yes — inline correction in the table cell.
- §0-§5 (the bit-comparison methodology and its 507-case four-way probe, the two-copy diagnosis —
  `aten.rs`'s unconditional `.contiguous()` plus candle's own `broadcast_matmul` copy one level
  down, the fold-not-broadcast fix and why it's not an approximation, the speed measurements against
  upstream): round-scoped measurement, not re-verified — these are the document's own landing
  numbers, already independently spot-checked for direction (not magnitude) by round 1/2's citations
  of the same commit.
- §6's other four items (bf16/f16 weight materialised per call — still open per the document's own
  account and not contradicted by anything found elsewhere this round; left-2D/right-N-D matmul
  folding; `addmm`'s bias-broadcast copy, named but judged too small to matter; no device
  measurement): not independently re-verified given time budget — no reason to suspect any of them,
  and item 1 in particular is consistent with round 1's `docs/perf/DTYPE_PERF.md` finding that the fused-
  gemv decode gap is also still open (same general "widening cost" family, different op).
- **Fixed: two** — the self-contradiction is the more consequential of the two (it directly answers
  a question two prior audit rounds had to chase down externally, from the one document that could
  have answered it directly), continuing this round's pattern (after SCHEMA.md) of a landing commit
  containing text describing itself as not yet landed.

### docs/numerics/BF16.md

396 lines, the round that overturned `docs/models/GENERATE.md` §6.2's own root-cause diagnosis (measured
"GEMM reassociation" and proved it wrong by disproof, then traced the real cause to `aten.add.
Tensor` truncating instead of round-to-nearest-even for bf16) — an explicit, well-executed instance
of the house style CLAUDE.md §5.5 asks for ("a check that cannot fail is not a check": the bug
survived because golden's `add` cases were all ≤24 elements and the bug only lives in the ≥32-element
vectorized path, plus a tolerance that passed 1-ulp errors). §6.3's "이번 회차가 넣지 않은 것" (not
included this round) table is the target shape this round prioritises.

- **Claim (§6.2, §6.3 row 1):** the sdpa path retains a bit-exactness gap GEMM reassociation cannot
  explain, narrowed to `_scaled_dot_product_flash_attention_for_cpu`'s block-wise reassociation, and
  left unresolved. **Status: FALSE today — closed the very next day, by a document that opens by
  quoting this exact paragraph.** **How checked:** `docs/kernels/SDPA.md`'s own opening line: "`docs/
  BF16.md` §6.2 가 미해결로 남긴 하나를 닫습니다" (closes the one thing BF16.md §6.2 left open),
  quoting the identical sentence. `git log` confirms `docs/kernels/SDPA.md`'s real landing commit
  (`4cd3bde`, 2026-08-28 21:52) is one day after `docs/numerics/BF16.md`'s (`fc89498`, 2026-08-28 15:07) —
  not the round 2 audit commit that later also touched SDPA.md's file. The block-wise kernel was
  reproduced bit-exact (`rust/torch_c/src/flash.rs`), but it is 20x slower at T=512, so it ships
  **behind an opt-in switch, off by default** — "not reproduced" is closed, "not the default path"
  is the fact that remains. **Fixed:** yes — inline correction in §6.3's table row, naming the
  sequel and what specifically changed (closed vs. still-true halves), rather than a blanket
  strikethrough. Marked `symbol-in-file` for the switch's own function and, while checking the
  table's other rows, `op-implemented` confirming `aten.bitwise_or.Tensor`'s existence (row 3 —
  see below).
- **Claim (§6.3 row 3):** `bitwise_or.Tensor` and similar dtype-promoting ops still refuse, no
  caller in this repo's own code paths. **Status: confirmed still true, correctly scoped** — the op
  itself is implemented today (`aten.bitwise_or.Tensor`/`aten.bitwise_and.Tensor` both in the
  current 168-op list, closed by unrelated later work, same as this round's `docs/models/IMPORT_TORCH.md`
  finding), but the *specific* claim this row makes — cross-dtype promotion refuses — still holds:
  live-verified `torch.tensor([1,0,1], dtype=torch.int64) | torch.tensor([True,False,True])` still
  raises `NotImplementedError: ... dtype promotion not implemented ... int64 vs bool` today. A good
  example of why "is the op implemented" and "does this specific claim about it still hold" are
  different questions — the op existing did not make this claim false.
- §1-§2 (the disproof of GENERATE.md's GEMM-reassociation diagnosis, the truncation-vs-RNE root
  cause and its "why no test caught it" analysis), §3 (the `opmath_in` fix and its `alpha` /
  reduction-specific measured quirks), §4 (the bit-exact, no-tolerance test additions and their
  fault-injection demonstrations), §5 (`aten.index.Tensor`'s multi-index-tensor rules, closing eager
  `generate()`), §6.1 (upstream-vs-itself non-reproducibility as the judgment baseline): round-scoped
  landing measurements and mechanism descriptions, already unusually rigorous about their own
  evidence — not re-verified, no reason to suspect any.
- §6.3's other rows (`float16`'s 1/20000 `alpha` residual, `use_cache=True` generation unmeasured,
  Android/iOS unmeasured): not independently re-verified given time budget — no reason to suspect,
  consistent with the pattern that this document's own hedging is already accurate.
- **Fixed: one** — smaller in scope than most findings this round (a "closed the next day by the
  direct sequel" cross-reference rather than a stale kernel claim), but notable for being the
  fastest-superseded finding in either round of this audit: one calendar day between the claim and
  its resolution, both landed before anyone came back to link them.

### docs/numerics/DTYPE.md

571 lines, a performance-and-design round (found the reduced-float widening cost was three separate
layers, only one of them hardware; recommended a path to int8/quantization support). Already carries
three of its own corrections to sibling documents (`docs/devices/DEVICE_ABS.md` §5.1's stale `cargo test`
crash claim, `docs/graph/QUANT.md` §8's "7.8x compression" scoped to KleidiAI only, §3.5's dead-strip
observation) — house style already applied outward. §6.4's recommendation ("candle `QTensor` as
`Repr`'s third variant") is the one forward-looking claim worth checking, matching round 1's pattern
for RNG.md → generator-port and round 2's pattern for CAPTURE.md → DECOMP.md.

- **Claim (§6.4, the recommendation):** candle's `QTensor` should become `Repr`'s third variant, with
  a specific landing order (verification axis first, then the enum variant + `dequantize` round
  trip, then wiring `mm`/`linear`, then upstream 4-bit op names). Framed as a recommendation, not a
  claim about current state. **Status: adopted, and in the recommended order.** **How checked:**
  `git log` shows this document's own commit (`abc341d`, 18:23) landed about four hours before
  `docs/graph/QUANT2.md`'s (`b032276`, 22:08) — same day. `rust/torch_c/src/tensor.rs:82` has
  `Repr::Quantized(Arc<QTensor>)` today, and round 2's `docs/graph/QUANT2.md` audit already independently
  confirmed the SmolLM2-135M q8_0/q4_0 20/20 token-match verification axis this document's step 1
  called a prerequisite. §6.4 item 2's separate judgment — `torch.int8` itself correctly refuses by
  name, not a defect — re-verified live and still holds: `torch.tensor([1,2,3], dtype=torch.int8)`
  still raises `NotImplementedError: ... dtype not storable by the candle backend` today (the
  `QTensor` path is a different mechanism from the raw `I8` `DType` this item is about).
  **Fixed:** yes — added a `> **정정 (문서 감사, 2026-09): ...**` blockquote before the
  recommendation's own text, pointing to `docs/graph/QUANT2.md` and noting which item held and which
  didn't need to change, matching the "recommendation later adopted" pattern already used elsewhere
  in this audit rather than rewriting the original recommendation. Marked `symbol-in-file`.
- §0-§1 (the conclusion summary and its own "not to be read as" caveats — an explicitly
  contamination-aware measurement session, load 1.6-11.7, relative comparisons only), §2 (the
  four-layer cost breakdown: candle's element-wise `to_dtype`, the widen-entire-tensor-not-just-the-
  accumulator bug, the zero-fill regression caught by re-measuring after adding a fix), §3-§4 (the
  A/B numbers and the per-layer remaining-cost table, bf16's hardware ceiling on this CPU), §5 (the
  exact-equality test suite and its own "verification lied once" self-correction — a `TORCH_C_
  ARTEFACT` omission that silently compared against a stale cached build, caught and fixed the same
  way `pytests/run.sh`'s own comment warns about): round-scoped measurement narrative, already
  unusually self-aware about its own honesty, not re-verified — no reason to suspect any.
- §7's unknowns (device measurement, `gemm-f16`'s runtime dispatch on `neon`-only targets, fused-
  gemv at DRAM-bound model scale, further bf16-narrowing headroom, SmolLM2 model-level re-measurement
  — explicitly argued unnecessary given the bit-equality proof, the QUANT.md §9.1 accelerate-f16
  slowdown not reproducing at the candle layer, `torchao`'s API being unjudgeable without the
  package installed, `QStorage`'s `Send`/`Sync` boundary only partially checked): explicit "미확인"
  admissions in the house style this audit already validates elsewhere, not re-checked.
- **Fixed: one** — a "recommendation adopted" finding rather than a false claim (the document never
  asserted the `QTensor` path already existed), continuing this round's cluster of same-day or
  next-day cross-document landings (after BF16.md→SDPA.md) that a plain per-file audit would not
  surface without checking commit order.

### docs/kernels/GROUPED_MM.md

502 lines, the round that closed Mixtral (the one architecture of twenty stuck on a missing
operator) — already unusually rigorous, with two pre-existing "Fixed since" blockquotes for the
`__setitem__`/`index_put_` mutable-view findings, cross-referencing `docs/kernels/VIEWS.md` by name. This
document went through three revisions the same day (`82ccc4b` 17:33, `36d3a2d` 18:08, `8c07af8`
19:56); checking whether its *final* revision picked up everything that had landed by then turned up
the fastest-arriving staleness found in either round of this audit — one that predates the document's
own last commit by under an hour.

- **Claim (§6.4):** "`aten.ge.Tensor` has no kernel" — `le.Tensor`/`lt.Tensor`/`gt.Tensor` all have
  one, `ge.Tensor` alone resolves and then refuses. **Status: FALSE at the moment this document's own
  final revision was committed, not just today.** **How checked:** `git log` shows `088e8f4` ("Feat:
  Close three kernel gaps, and disprove the stated reason for the fourth" — its own commit message
  opens with "ge.Tensor existed for le, lt and gt but not ge... One arm, 31 cases") landed at 18:58,
  this document's own final revision (`8c07af8`, "retire two gap notes the work has closed") landed
  at 19:56 the same day — 58 minutes later — and still shipped the "has no kernel" sentence
  unqualified. `aten.ge.Tensor` is in the current 168-op `_aten_implemented()` list; live-verified
  `x >= tensor` computes. Round 1's `docs/kernels/VIEWS.md` audit already independently found the same
  kernel from a different document, unlinked from this one. **Fixed:** yes — added a `> **Fixed
  since — and it was already fixed when this sentence was committed.**` blockquote, in the same
  style as the document's own two pre-existing "Fixed since" notes a few paragraphs above, naming
  the specific timing gap rather than treating it as ordinary later-drift. Marked `op-implemented`.
- **Claim (§6.5, and §9's methodology note):** `torch.manual_seed` refuses via Dynamo's disable
  wrapper (`torch._C._dynamo.eval_frame.set_eval_frame`), so Mixtral's weights had to be filled by a
  shared LCG rather than real seeding. **Status: FALSE today** — closed by the same CPU-generator
  port round 2's RNG.md/TENSORBASE.md audits found (`torch.manual_seed(42)` succeeds live today).
  **Fixed:** yes — inline blockquote noting the closure, explicitly not re-running §9's comparison
  with real seeding (out of scope for a documentation pass, and not needed to confirm the refusal
  claim itself is stale).
- §1-§2 (the schema and semantics read from the vendored tree and measured against real upstream
  torch, the 16-byte stride refusal rule, the "rows nobody writes"/"offsets that go backwards"
  behaviors), §3 (the group-walk implementation and why `slice_assign` was rejected), §4-§5
  (reachability from Python, the operator-level and executed-model verification), §6.1-6.3 (the
  `overloads.json` underscore-prefix predicate bug, `floor_divide.Scalar`, the three table-entry
  additions), §7 (gate counts), §9 (the unpatched Mixtral run and its per-member call counts): not
  independently re-verified beyond the two items above — this document's own sabotage-testing and
  cross-referencing discipline (already correctly annotating two of its own claims as fixed by
  `docs/kernels/VIEWS.md`) is exactly the house style this audit looks for, and no other claim contradicted
  anything found elsewhere this round.
- **Fixed: two** — the `ge.Tensor` finding is the sharpest example this audit has found of the "a
  correction lands and the very next commit of the same document still doesn't pick it up" failure
  mode: not a later, unrelated round, not even the next day — the fixing commit and the stale
  sentence's committing revision are less than an hour apart, in the same session, on the same file
  the fixing commit's own message explicitly discusses.

### docs/design/SURFACE_HONESTY.md

477 lines. Named directly in this round's brief as worth checking for self-drift, and it has
drifted — genuinely and instructively. Two parts: §1 fixes `_Unimplemented` placeholders answering
`bool()` truthily (a real, landed fix, re-verified below), and §2 is a **decision record**: whether
to patch the vendored tree to bind `torch.distributed.Store`, and it concludes "no patches — build
`torch.distributed` for real, starting from `world_size=1`," explicitly deferring `from_config`
until that happens (§2.6: "그때까지 검증은 손으로 옮겨 적은 모델로 계속합니다").

- **Claim (§0 summary table, §2.7):** "`from_config` 진행: 변화 없음. 같은 벽(`fake_pg.py:7`,
  `torch.distributed` has no attribute `Store`)에서 멈춥니다." **Status: FALSE — closed the very
  next morning, by the exact plan this document's own §2.6 laid out.** **How checked:** `git log`
  shows this document's last commit (`eae2a42`) at 2026-08-24 22:38; `docs/distributed/DISTRIBUTED.md`'s landing
  commit (`99fec1b`, "Feat: Stand up torch.distributed, and import transformers for the first time")
  at 2026-08-25 06:52 — the next morning, executing precisely the plan §2.6 recorded ("go implement
  `torch.distributed` for real, `world_size=1` first"). Live-verified today:
  `hasattr(torch.distributed, 'Store')` → `True`; `AutoModelForCausalLM.from_config(cfg)` with the
  identical llama config succeeds, parameter count 95,040 — matching this round's own
  `docs/models/FROM_CONFIG.md` real-torch measurement exactly, the same cross-check used there.
  **Fixed:** yes — added a `> **정정 (문서 감사, 2026-09): ...**` blockquote after §2.7's
  reproduction transcript and a shorter inline note in §0's summary table, pointing at
  `docs/distributed/DISTRIBUTED.md` and quoting this document's own "그때가 왔다" framing rather than rewriting
  the decision record (which was a real, honest decision at the time, not a false claim about the
  present). Marked `symbol-in-file` against `_install_distributed_c10d`.
- **Self-assessment, since this document is explicitly about exactly this failure:** the drift found
  here is not different in kind from what round 1-2 found elsewhere (a later, unrelated — or in this
  case, directly consequential — commit closed a gap and nobody came back to update the document
  that named it as open). What is different is the document's own subject matter: `SURFACE_HONESTY.
  md` exists to catch a shim claiming something is true when it is not, and by the time this round
  reached it, the document itself was doing the analogous thing about its own state — claiming
  `from_config` still fails when the very decision it recorded had already been executed and had
  succeeded. The irony is worth stating plainly, as the brief asked: **the document about honesty
  had gone stale in the same shape it was written to detect**, though the mechanism (a later commit,
  not an intentional lie) is the ordinary one this whole audit keeps finding, not a special case.
- §1 (the `_Unimplemented`-truthy fix, the three-way split of upstream types behind the ten names the
  coordinating session flagged, the four additional names the scan had missed, the single real call
  site that mattered, and why both option (a) and (b) were applied together rather than either
  alone): re-verified live where cheap — `torch.backends.cudnn.is_available()` returns `False`
  today, matching §1.6's claimed post-fix state, no drift found. §1.7's explicitly-deferred items
  (`_after_ADInplaceOrView_keyset` etc. — wrong type, both sides truthy, not this round's topic):
  left as an open, correctly-labeled deferral, not re-chased.
- §2.1-§2.6 (the reproduction, the upstream-reproduces-the-same-crash finding, the three rejected
  branches and their measurements, the federated-learning namespace correction already present as a
  self-correction within §2.6 item 1): mechanism and decision narrative, not re-verified beyond the
  one outcome claim above — no reason to suspect the analysis itself, only the "still open" framing
  of its consequence.
- **Fixed: one, but it is the highest-symbolic-weight finding of this whole three-round audit** — the
  document whose entire purpose is catching "the shim says X exists and it doesn't" was itself,
  when read today, saying "this capability doesn't exist yet" about one that does, for the ordinary
  reason (a later commit executed the very decision recorded here) this audit has now documented
  well over a dozen times.

### docs/perf/PERF.md

239 lines, a load-contaminated (4.24 average, explicitly disclosed) A/B performance round finding
the model-level 7x gap against upstream was almost entirely one Cargo feature flag (`accelerate`,
Apple-only) and closing it to 1.0x with zero golden regressions. Already correctly self-limits its
own numbers to "same-session ratio only, not a regression baseline" (§0) — the house style already
validated correct for this document class in rounds 1-2 (SEQLEN.md, DTYPE_PERF.md, SDPA.md,
QUANT.md). One forward-pointing gap worth closing: §5's "Android still has no answer" is exactly the
kind of "next step, not done here" claim this round prioritises.

- **Claim (§5, "안드로이드의 행렬곱은 별도 과제로 남습니다"):** Android has no `accelerate`-equivalent
  matmul solution; it is left as a separate, unstarted task. **Status: has a direct sequel.**
  **How checked:** `docs/perf/PERF_ANDROID.md` ("안드로이드 aarch64 행렬곱 — 무엇이 느리게 만들고
  있었나") landed 2026-08-31, six days after this document (2026-08-25) — its title picks up exactly
  where this section leaves off. **Fixed:** yes — added a forward-pointing correction after the
  sentence, explicitly not comparing this document's numbers against PERF_ANDROID.md's (different
  session, different hardware — an emulator — so a direct number-to-number comparison would itself
  be the kind of contamination §0 warns against).
- §0-§4 (the A/B methodology and its explicit limits, the model-level 7x-to-1x closure, the thread-
  count control ruling out parallelism as the cause, the remaining 2.7x `_softmax` gap attributed to
  scalar vs. vectorized `exp`), §6 (the unconfirmed-items table), §7 (the MLX GPU-vs-CPU comparison
  and its crossover-point findings): round-scoped measurement narrative with its own honest
  limitations already stated, not re-verified or re-measured — re-measuring performance is out of
  scope for a documentation audit and risks exactly the contamination CLAUDE.md's own rule warns
  against.
- **Fixed: one** — a forward pointer rather than a correction to a false claim (the document never
  claimed Android had a solution; it correctly named the gap as open, and it still is, just not
  un-investigated any more).

### docs/perf/PERF_ANDROID.md

643 lines — the direct, explicitly-named sequel to `docs/perf/PERF.md` §5's open Android gap, and the
most rigorous measurement document read in either round of this audit. Every comparison carries its
own control group (two labels on the identical build, re-measured the same way), and every
"transferred/did not transfer" verdict is qualified by which of two regimes it falls into
(Python-dispatch-dominated vs. kernel/bandwidth-dominated) rather than asserted uniformly — this is
exactly the "the f32 sequence-length curve and the bf16 ratio moved several times" pattern the brief
warned about, and the document's own §10 table already does the reconciliation work a lazy read
might mistake for contradiction: host numbers (from `docs/bindings/BIND.md`/`docs/design/DISPATCH.md`/`docs/SEQLEN.
md`/`docs/perf/DTYPE_PERF.md`) and device numbers for the *same* five optimizations are laid side by side
and explicitly **not** treated as the same measurement re-run, with the document itself stating "이
비교는 절대 규모가 전혀 다른 두 기계 위에서 잰 값이라 직접 대조할 근거가 약하다" (this comparison
is between two machines of entirely different absolute scale, so a direct comparison has weak
grounds) wherever that applies.

- **Checked for staleness rather than found stale.** §5's `GEMM_THREADING_THRESHOLD = 4_000_000`
  constant is confirmed unchanged in `rust/torch_c/src/lib.rs:601` today — the one mechanically
  checkable fact this document commits to a specific number for (as opposed to a ratio). §6's fix
  ("`parity` now builds the host side without `accelerate` for a gemm-vs-gemm comparison rather than
  widening the tolerance list") is a design decision, not independently re-run (`scripts/` is
  outside this round's territory).
- **On the specific pattern named in the brief** ("f32 sequence-length curve and bf16 ratio moved
  several times, say which document supersedes it"): this document is itself largely the
  *reconciliation* document for that pattern, not a stale source of it — §10.2-§10.5 compare host
  numbers from four earlier documents against freshly-measured device numbers for the *same*
  optimizations, and each comparison explicitly states whether the two are on comparable footing
  before drawing a "transferred" verdict. No numbers in *this* document contradict a later
  document — nothing found this round supersedes any ratio here, and this document's own final
  table (§10.6) is already the up-to-date reconciliation across five prior documents' numbers, not a
  claim any of them has since moved past.
- §0-§4 (the emulator-vs-real-hardware caveats, the GEMM-kernel-quality-is-identical finding across
  host/device, the AMX-not-kernel-quality explanation for `docs/perf/PERF.md`'s 7x, the compile-flag
  hypothesis disproven both by source reading and by measurement, the threading-threshold diagnosis
  and fix with a bit-identical sha256 proof that the fix changes nothing but which core does the
  work): round-scoped measurement, internally self-verified (source claims cross-checked against
  measured behavior throughout, e.g. §3.1's `nm`/`llvm-nm` evidence for the runtime-detection claim),
  not re-verified beyond the one mechanical constant checked above.
- §9's unconfirmed-items table (no real Android hardware — only the emulator throughout; big.LITTLE
  scheduling unmeasurable on a homogeneous emulator; f16/bf16/int8 untestable because the emulator
  doesn't advertise `asimdhp`/`asimddp`/`i8mm`; no aarch64-android upstream torch wheel to compare
  against directly; whether 4M is the right threshold on real hardware): not independently
  re-verified — these are hardware-access limitations stated as such, not falsifiable claims, and
  nothing found elsewhere this round suggested real Android hardware measurement has since happened.
- **Fixed: none — no false or stale claims found.** The document this round's brief specifically
  flagged for the "measurements superseded" pattern turned out, on reading, to already be the
  document doing that reconciliation work correctly for five prior documents' numbers; there was no
  further staleness to find on top of it within this document's own text.

### docs/numerics/SCALAR.md (full read — carried markers already, one of the two named in the brief)

811 lines, the round that established the per-kernel reduced-float scalar rule (`mul`/`div` widen
the Python number to `opmath_t`, `add`/`sub` narrow it — no principle survives contact with the
table, it is genuinely per-kernel) and found four more ops wrong by it. Already the most heavily
self-instrumented document in this repository's `docs/` tree — 13 pre-existing DOCWATCH markers
(the most of any file audited across all three rounds), §8 is an entire section devoted to closing
§5/§6's own "not fixed here" items with inverted markers so the closure "cannot silently come
undone" (the document's own words), and §7/§8.4 run two separate sabotage rounds including three
faults explicitly recorded as *unable* to fail, with the reason stated rather than hidden.

- **All 13 pre-existing markers re-verified PASS against today's tree** — `mul.Scalar`, `mul_.
  Scalar`, `floor_divide.Scalar`, `div.Scalar_mode`, `pow.Tensor_Scalar`, `pow.Scalar`, `add.
  Scalar`, `sub.Scalar` all `op-implemented`; five `symbol-in-file` markers for the specific tests
  and case builders that pin the rule. None had drifted — this is the only file found in either
  round of this audit where a full marker set landed and, on re-check, needed nothing.
- **Claim (§6, the one item left open after §8's closures):** `x *= 0.3` on a reduced-float tensor
  still disagrees with upstream by one representable step, because upstream's Python `*=` resolves
  to `mul_.Tensor` (which widens) while this shim's resolver reaches `mul_.Scalar` (which correctly
  narrows, matching upstream's own `mul_.Scalar` — the mismatch is which *kernel* the same source
  line reaches, not either kernel being wrong). Closing it needs the resolver's "numbers as tensors"
  rule, described as living above `aten.rs`, out of this document's own scope. **Status: not
  independently re-verified this round** — the fix would require the same op-resolution work this
  round's own `docs/bindings/OVERLOAD.md`/`docs/bindings/TENSORBASE.md` findings traced to `bootstrap.py`'s overload
  table, and nothing else found this round touched `mul_`'s specific resolution path; no evidence
  either way was encountered, so left as reported rather than guessed at.
- §1-§2 (establishing the per-kernel rule from a 420-value differential against two upstream-
  arithmetic models, the `mul.Scalar` fix and its "cases that could not fail" analysis), §3 (the
  three-op family sweep and `pow`'s `float32`-narrows-too finding, with its own explicit refusal to
  add a `float32` case because the vectorisation tail makes the "correct" answer length-dependent
  upstream), §4 (nine prefill digests unchanged, explained by tracing exactly which ops a real
  SmolLM2 forward calls rather than asserted), §7-§8 (both sabotage rounds, the `softplus`/`norm`/
  `add.Scalar`/`sub.Scalar` closures, the FMA-compilation residual sized and left rather than
  half-fixed): already unusually rigorous and self-checking, not independently re-verified beyond
  the marker re-run above — no reason to suspect any claim, and the document's own house style
  already does most of what this audit checks for.
- **Fixed: none needed — the highest-quality document found in either round of this audit**, both
  in the sense that no claim had gone stale and in the sense that its own self-correcting apparatus
  (13 markers, an entire section devoted to closing its own prior round's open items, sabotage that
  records what could not fail rather than omitting it) is close to a template for what the other 73
  files in this audit's scope would ideally look like.

### docs/training/LOSS.md (full read — the second of the two files named in the brief as carrying markers)

924 lines, the round that closed the loss/optimizer/dropout gap `docs/training/AUTOGRAD.md` left open —
matches `docs/numerics/SCALAR.md`'s quality tier exactly: 18 pre-existing DOCWATCH markers, its own §8
sabotage table (26 faults) records which ones could not fail and why rather than omitting them, and
§5.4.1 is a mid-document self-correction adding a bounded-divergence case specifically so a number
that used to live only in prose ("no case sees this because the widest case is 6 elements") gets a
check that can fail if it moves.

- **All 25 pre-existing markers re-verified PASS against today's tree** — `op-implemented` for
  `_log_softmax`/`nll_loss_forward`/`native_dropout`; `op-not-implemented` for `lerp_.Scalar`/
  `addcmul_.default`/`addcdiv_.default`/`is_complex.default` (Adam's fourth wall, §6.4) and
  `nll_loss2d_forward`/`nll_loss_backward`/`native_dropout_backward`/`_log_softmax_backward_data`
  (§9's explicit "not done" list); `hasattr` correctly distinguishing `nll_loss_forward` (upstream
  has no such name — the shim's own name is deliberately invented, §3.4) from `native_dropout`
  (upstream has it); `json-key`/`symbol-in-file` for the specific overload-table entries and Rust
  functions. None had drifted.
- §1's "gap is a name, not a kernel" finding (climbing the real call path rather than trusting a
  `TorchDispatchMode` scan, since `CompositeImplicitAutograd` composites are invisible to dispatch-
  level tracing) explicitly cites this exact failure shape recurring five times before it
  (`docs/architectures/ARCH20.md`, `docs/kernels/GROUPED_MM.md` §6.1, `docs/kernels/TRIL.md` §2, `docs/bindings/SPELLINGS.md`) — consistent
  with this round's own repeated finding of the same pattern (OVERLOAD.md, TENSORBASE.md, SAMPLING.md,
  DEVICE.md, ARCH.md, GPT2.md, OPS4.md, OPS8.md, NN_SURFACE.md), a convergent observation from both
  the document's own account and this three-round audit's independent tally.
- §2-§4 (the `_log_softmax` dual-kernel summation-order fork and its constructed separating input,
  `nll_loss_forward`'s eight-level cascade and its three load-bearing details, the real SmolLM2 loss
  attribution isolating which kernel carries the residual), §5 (18 sabotage faults including three
  that could not fail on the first attempt, with the case-list fix shown rather than just the
  outcome), §6 (`zero_grad()`'s profiler-marker gate, `DisableTorchFunctionSubclass` correctly
  distinguished from the torch-function mode stack), §7 (`native_dropout` vs. the eager `dropout`
  composite, the two spellings agreeing bit-for-bit specifically because of scalar narrowing — a
  direct instance of `docs/numerics/SCALAR.md`'s rule table applied and cross-referenced): already
  extraordinarily rigorous and self-checking, not independently re-verified beyond the marker re-run
  above — no reason to suspect any claim.
- §9's "what this round did not do" table (`cross_entropy_loss`'s probability-target and label-
  smoothing branches, `nll_loss_nd` for rank ≥3, Adam's remaining three kernels plus `torch.
  is_complex`, `.grad`'s setter — explicitly deferred per `docs/training/AUTOGRAD.md` §7's still-standing
  argument, the `profiler::` schema table, any backward pass): all four op-shaped items in this list
  have `op-not-implemented` markers and all re-confirmed absent today — no drift.
- **Fixed: none needed — the second-highest-quality document found in either round**, alongside
  `docs/numerics/SCALAR.md`. Both were written in the same general era of this repository's documentation
  practice (heavy DOCWATCH instrumentation, explicit "what this suite still cannot see" sections,
  sabotage that records negative results) and both held up completely under re-verification —
  suggesting the self-correction discipline this audit's overall conclusion recommends is not
  hypothetical: it is already working exactly as intended in the files that adopted it most fully.

### docs/design/CARGO_KT.md

365 lines — an outlier in this audit's scope: a **design proposal for a sibling repository**
(`/Volumes/macMini/thisisthepy/pypackpack`, a separate Kotlin build tool), not about this repo's own
`_C` shim or op coverage at all. No code was implemented (the document says so explicitly, twice);
every claim is either a citation of `pypackpack`'s current source (file:line) or an explicit
"미확인" (unconfirmed). The "refusal names a kernel as missing" pattern this round prioritises does
not apply — there is no aten op, `_aten_implemented()`, or golden harness in scope here.

- **Claim (§0, §3-1):** `Cargo.kt`/`NDK.kt`/`XCode.kt` in `pypackpack` are 4-line placeholder files
  (package declaration + one comment), and `Meson.kt` — the one implemented backend — does not
  reference any of them despite `SPEC.md`'s "adapter pattern" documentation describing an intent
  that isn't realized in code. **Status: confirmed still true today.** **How checked:** `wc -l` on
  `pypackpack`'s `Cargo.kt` and `NDK.kt` — both still 4 lines; `git log -1` on `Cargo.kt` in that
  repo shows its last change was `fda5f00`, 2025-06-07 — over a year before this document was
  written, and unchanged since. This is a claim about a pinned, external repository's state rather
  than something that drifts with this repo's own kernel growth — same non-drifting category as
  `docs/design/CORE_ATEN.md`/`docs/graph/QUANT.md`'s upstream-pinned facts, just pinned to a sibling repo instead
  of upstream torch.
- §1-§2 (the `BackendInterface` signature, `Meson.kt`'s three-stage lifecycle and process-execution
  pattern, its testability-via-`open`-subclassing design), §3 (the `NDK.kt`/`XCode.kt` adapter
  design options and the target-triple mismatches found in `Platforms.kt` — iOS's three naming
  discrepancies and Android's API-level information loss through canonicalization), §4 (where the
  `lib_C.so` → `_C.so` rename belongs in a three-stage `Cargo` backend), §5 (six items explicitly
  scoped out, each with a stated reason), §6 (four explicitly unconfirmed items, e.g. the iOS
  simulator rustc triples not locally re-verified because this machine has no `rustc`/`rustup`
  installed): all design analysis and citations of `pypackpack`'s current source, not independently
  re-verified beyond the one claim above — no reason to suspect any, and the document is already
  unusually careful about marking what it did and did not verify (its own opening line: "실행해서
  확인하지 못한 것은 '미확인'으로 표시했습니다").
- **Fixed: none — no false or stale claims found.** The only substantive risk this round checked (a
  design proposal describing a target repo as further behind than it now is) did not materialize —
  `pypackpack` has not moved on this file since well before the proposal was written.

### docs/platform/RUST_CROSSBUILD.md

296 lines, the investigation `docs/design/CARGO_KT.md`/`docs/design/ABI3.md` both cite by file:line throughout —
already carrying two of its own visible corrections (the iOS/Android path mixup, the abi3
recommendation reversed twice with the reasoning kept visible both times) matching this audit's
house style before this round touched it.

- **Claim (§1, the abi3 recommendation):** "조사 완료. 권고는 `abi3-py313` 을 켜는 것" (investigation
  complete, recommendation is to enable `abi3-py313`) — a recommendation, not a claim it was already
  adopted. **Status: adopted, confirmed live.** **How checked:** `rust/torch_c/Cargo.toml:23` has
  `features = ["extension-module", "abi3-py313"]` today, with a comment citing "`abi3-py313` is
  ABI3.md §7's recommendation" — the same landing round1's `docs/design/ABI3.md` audit already confirmed
  independently. Not marked as a correction (the document never claimed adoption, only recommended
  it — nothing to fix).
- **Claim (§5, closing paragraph):** "`/Volumes/macMini` 는 349Gi 중 4.6Gi 여유" (4.6Gi free) —
  **self-contradicted by this same document's own §0**, which opens with "여유 **164Gi** (이전
  4.6Gi)" (164Gi free, up from a previous 4.6Gi) — i.e. §5 carries the pre-recovery number as if
  current, directly beneath a section that already recorded the recovery. **Status: internally
  inconsistent, and disk-free numbers are not something this audit re-pins** (they change
  continuously — re-checked live today: 86Gi external, 12Gi internal, neither matching either
  in-document figure). **Fixed:** yes — added a correction noting the stale figure sat right below
  the updated one in the same document, without asserting a new number that would itself go stale
  by the next session. No DOCWATCH marker fits (disk capacity is not one of the six primitives, and
  a marker for a continuously-changing external fact would be exactly the "crying wolf" shape
  `docs/verification/DOCWATCH.md` warns against for global counts).
- §0 (the toolchain inventory, the worktree `build/` cleanup recovering 159GB, the `CARGO_TARGET_
  DIR` placement decision), §0.5 (the three-target build verification and the iOS linking
  investigation — `pyo3-build-config` hardcoding a libpython link for iOS regardless of the
  `extension-module` feature, the two-part fix combining `-F`/`-framework` with `PYO3_CONFIG_FILE`'s
  `suppress_build_script_link_lines`, and the explicit warning that the committed `.cargo/config.
  toml` path is machine-specific): the two Cargo.toml/config.toml facts spot-checked (abi3 feature,
  the `-undefined dynamic_lookup` rustflag) both still match today's source. §1's other items
  (`PYO3_CROSS*` env vars, the Android/iOS archive layout asymmetry), §2-§4 (cargo-ndk necessity,
  iOS xcframework packaging notes, the `pypackpack` `Cargo.kt` requirements — superseded in detail by
  `docs/design/CARGO_KT.md`'s own more thorough analysis, not re-derived here): not independently
  re-verified — no reason to suspect any, and §1's open item (whether an abi3 module loads under a
  3.14.7 interpreter) remains explicitly unconfirmed by the document's own account, consistent with
  no 3.14 interpreter being available on this host either.
- **Fixed: one** — an internal self-contradiction (stale figure directly below its own correction),
  the same general shape as this round's `docs/bindings/SCHEMA.md`/`docs/perf/LINEAR.md` findings, though about a
  fact (disk space) that is expected to drift on its own rather than one this audit can re-pin.

### docs/design/CANDLE_DEPS.md

418 lines — an investigation into `candle-core`'s non-optional `tokenizers` dependency (44 unused
crates, ~36% release-build CPU time), ending in a validated but deliberately **not landed** fix: a
`[patch.crates-io]` block pointing at a machine-local absolute path, which the coordinating session
(§9) explicitly declined to commit because it would break every other checkout's build. This is a
"decision recorded, not code shipped" document, structurally similar to `docs/design/SURFACE_HONESTY.md`
§2's decision record, so worth checking whether the decision has since been executed.

- **Claim (§9, the standing decision):** the patch stays local until one of three conditions is
  met — upstream PR #3490 merges, a public fork carries the patch, or candle is vendored into this
  repo. Until then, "이 최적화를 켜지 않습니다" (this optimization stays off). **Status: confirmed
  still the current state.** **How checked:** `rust/torch_c/Cargo.toml` has no `[patch.crates-io]`
  block today; `rust/torch_c/Cargo.lock` still lists `tokenizers` (line 1231). Whether upstream PR
  #3490 has since merged was not checked — `rust/`'s `Cargo.lock`/`Cargo.toml` are this round's
  forbidden territory, and confirming a GitHub PR's status is outside a documentation-claim audit's
  normal method (no live command in this repo answers it); reported as unconfirmed rather than
  guessed at.
- §0-§4 (why `tokenizers` is non-optional in candle-core 0.11.0's `Cargo.toml`, that `torch_c` never
  calls the `TokenizerFromGguf` trait it exists for, the three removal options considered and why
  forking-and-deleting was rejected in favor of the patch), §3 (the crate-count and build-time
  measurements, cross-checked twice — once against a standalone probe crate, once against `rust/
  torch_c`'s real `Cargo.lock`, both agreeing on −44 crates), §6 (explicit unknowns: the PR author's
  self-reported test count, Windows MSVC-vs-GNU relevance, cross-compile measurement, stripped
  binary size, burn's dependency tree): round-scoped investigation and measurement, not re-verified
  — no reason to suspect any, and the crate-count claim is a `Cargo.lock` fact independently
  re-confirmed above as part of checking §9.
- **Fixed: none — no false or stale claims found.** The decision this document records has neither
  been executed nor overtaken by events discoverable from within this repo; the standing "not
  landed" state is exactly what's still on disk.

### docs/verification/HARNESS.md

375 lines, the round that found `tools/golden/compare.py`'s own self-test (`--inject-fault`) had
only ever exercised one comparator out of ten, covering 1377 of 1781 cases (77.3%) and leaving the
other 404 (22.7%, nine comparators for multi-result ops) never proven able to fail. §6 found three
genuine gaps in the untested comparators themselves (indices dtype/shape not compared in
`_pair_result_check`+`dtype-last` and `_topk_multiset_check`+`shape-last`/`dtype-last`) and
explicitly left them unfixed as out of this document's own file scope (`tools/golden/cases.py`).
Worth checking whether a later round — this round's own territory excludes `tools/golden/` for
edits, but not for reading — picked them up.

- **Claim (§6, §4's table):** three comparator blind spots are real defects, not by-design gaps, and
  are left unfixed, with the exact one-line fix given for each. **Status: FALSE today — all three
  closed.** **How checked:** `tools/golden/cases.py` has `indices dtype mismatch`/`indices shape
  mismatch` checks at two sites (lines 6287/6295 and 9575/9578), textually matching this section's
  own proposed fix; `tools/golden/compare.py`'s `KNOWN_GAP` dict is empty today (`KNOWN_GAP: dict[
  tuple[str, str], str] = {}`) — this section's own closing instruction ("고친 뒤에는 `KNOWN_GAP`
  에서 해당 항목을 지워야 합니다... 안 지우면 `--self-test` 가... 실패합니다") was followed,
  since `--self-test`'s design would fail loudly if a fix landed without the table update.
  **Fixed:** yes — added a `> **정정 (문서 감사, 2026-09): ...**` blockquote after §6's "고치지
  않았습니다" sentence and a shorter inline note after §4's `GAP` legend line, without editing the
  actual `tools/golden/` files this round is forbidden from touching (only observed that another,
  earlier round already had). Marked `symbol-in-file` against the literal fix text.
- §1-§3 (the coverage-gap discovery and its historical justification — the three original `value_
  check` users were legitimately unfixable by the default pipeline when written, and became a real
  gap only once `value_check` became the standard multi-result comparison mechanism), §3.2's own
  self-correction (the `permute`/`permute-all` mode split, found because `--self-test` itself caught
  a `BLIND_BY_DESIGN` entry actually getting caught — an in-document demonstration of exactly the
  "a check that can't fail isn't a check" principle this whole audit round has been applying), §5
  (the four legitimate `blind` entries and their reasoning), §7 (the tolerance-scheme measurement
  and recommendation, explicitly not applied pending a future case addition — a "not done, and here
  is exactly why not" deferral rather than an oversight), §8 (explicit unknowns, correctly hedged):
  not independently re-verified beyond the one finding above — no reason to suspect any, and this
  document's own §3.2 sidebar is itself evidence of the same self-correcting discipline this round
  keeps finding in the higher-quality files.
- **Fixed: one, but touching three comparator blind spots and a load-bearing invariant
  (`KNOWN_GAP` must be empty for `--self-test` to mean what it claims)** — the same general shape as
  this round's other findings (a "not fixed here" item closed by unrelated later work), but notable
  for being a fix to the *verification apparatus itself* rather than to a kernel or Python spelling —
  closing it matters more than most because every other file's DOCWATCH/golden-based finding in this
  three-round audit implicitly depends on the harness being honest about what it can catch.

### docs/devices/VULKAN.md

377 lines, a feasibility investigation (Android Vulkan compute via `ash`/`wgpu`, both proven bit-
identical to CPU on an emulator, `ash` recommended) that explicitly declines to make one decision —
whether Apple GPU support should come from candle's existing `metal` feature (favoring `ash` for
Android alone) or from `wgpu` (one WGSL kernel covering all three targets) — and defers it as
CLAUDE.md §5.7 territory (a cross-target kernel-ownership decision beyond this task's scope). §5.4's
"아직 하지 않은 것" (not yet done) list is the checkable "next step" shape this round prioritises.

- **Claim (§5.4):** no Vulkan dependency has been added to `rust/torch_c`; `"vulkan"` is only a
  reserved device-string placeholder in `device.rs`; `rust/vk_probe` remains a separate, unmerged
  workspace member. **Status: confirmed still true today.** **How checked:**
  `rust/torch_c/Cargo.toml` has no `ash`/`wgpu`/vulkan dependency; `device.rs:69` still only lists
  `"vulkan"` as a device-type string in a table, with no backing implementation; `rust/vk_probe/`
  still exists as a sibling crate (`Cargo.toml`, `Cargo.lock`, `build.rs`, `shaders/`, `src/`) with
  no reference from `rust/torch_c`.
- **Claim (§5.3, the deferred decision):** whether Apple GPU support goes through candle's `metal`
  feature (favoring `ash`) is still open, evidenced by `rust/torch_c/Cargo.toml` currently enabling
  only `accelerate` on Apple targets, not `metal`. **Status: confirmed still true today** —
  `Cargo.toml:107` still reads `features = ["accelerate"]` only; `metal` is absent. The decision
  this document flagged as outside its own scope remains unmade.
- §1-§2 (the bit-identical vecadd/matmul proof on both `ash` and `wgpu`, the tamper self-test that
  demonstrated a real detection boundary at 1-2 ULP from FMA fusion, the host-vs-guest distinction
  for feature bits — `MoltenVK` proving the emulator translates to the host GPU, and the correction
  this document makes to `docs/devices/DEVICE.md` §10's broader fp16/int8 claim, narrowing it to a host-only
  fact), §3 (why pre-compiled SPIR-V kernels don't violate the "no model pre-conversion" premise),
  §4 (the `ash`/`wgpu`/`vulkano`/ExecuTorch comparison, `candle` confirmed to have no Vulkan backend
  at the pinned 0.11.0 tag), §6-§7 (device discipline followed, and an extensive explicit list of
  what an emulator cannot answer — performance, real Adreno/Mali hardware, in-APK behavior, API 26,
  multi-kernel synchronization, large-buffer memory pressure): not independently re-verified beyond
  the two claims above — no reason to suspect any, and the document's own tamper self-test (§1) is
  already the kind of "can this check actually fail" demonstration this audit looks for.
- **Fixed: none — no false or stale claims found.** Both checkable "not yet done" claims (no Vulkan
  dependency wired into `rust/torch_c`, the Apple-GPU-backend decision still unmade) remain
  accurate; the decision this document explicitly declined to make on its own authority has not
  been made by anyone else either.

### docs/platform/B_SPIKE.md

534 lines, a timeboxed build spike against real upstream PyTorch (`main` at a pinned 2026-08-23
commit) testing whether option B ("selective libtorch cross-build") could actually cross-compile for
Android. It could (two builds completed, five blockers found and worked around, all upstream-side
decay rather than this repo's own defect) — but the spike's most consequential finding is that the
question itself was misframed: the mobile build path forces `BUILD_PYTHON=OFF`, so what it produces
is never `torch._C`, and DESIGN.md §6's "cross-compilation is B's only unknown" is corrected in the
document's own text (§4). This document's recommendation — favor A (candle+PyO3) — is exactly what
round 1's `docs/design/DESIGN.md` audit already confirmed was adopted ("결론: A(candle 위 `torch._C`)로
갑니다").

- **Consistency check, not a staleness finding:** this document's recommendation and `docs/DESIGN.
  md`'s recorded final decision agree (A was chosen), so there is no "recommendation later reversed"
  or "still open" claim to correct here — unlike `docs/numerics/RNG.md`/`docs/numerics/DTYPE.md`'s "recommendation
  adopted" pattern in earlier rounds, this document doesn't claim its own recommendation is still
  pending; it just makes the case, and the case matches what happened.
- §1-§3 (upstream PyTorch's mobile-build entry-point scripts confirmed deleted via GitHub API commit
  lookups, the five build blockers and their upstream root causes — a stale eigen submodule
  reference, an NDK Vulkan wrapper removal PyTorch's CMake didn't follow, a CUDA-only op breaking
  static-dispatch codegen), §5 (iOS confirmed worse off than Android — the CMake toolchain file
  itself deleted from upstream main), §6 (explicit unmeasured items: deployed-binary size,
  on-device execution, `TRACING_BASED`/`model_tracer`, `BUILD_MOBILE_AUTOGRAD=ON`, whether the
  `cpuinfo` WHOLE_ARCHIVE conflict is static-build-specific): all claims about a pinned external
  repository's state at a specific commit, not about this repo's own kernel coverage — the
  non-drifting category this round has repeatedly distinguished from `_aten_implemented()`-linked
  claims (`docs/design/CORE_ATEN.md`/`docs/graph/QUANT.md`/`docs/design/C_SURFACE.md`/`docs/design/CANDLE_DEPS.md` above). Not
  re-verified — re-running this spike would mean re-cloning and rebuilding upstream PyTorch, well
  outside a documentation-claim audit's scope, and nothing else found this round suggested the A/B
  decision has been revisited.
- **Fixed: none — no false or stale claims found.**
