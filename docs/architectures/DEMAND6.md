# Demand Round 6

Re-run of the model sweep from docs/architectures/DEMAND.md / docs/architectures/DEMAND3.md / docs/architectures/DEMAND4.md, from
measurement, at `develop`+worktree commit `017a8b6`. Same scope as every prior round: self-built
toy `AutoConfig`s (small hidden size, 1-2 layers, few heads, tiny vocab), hand-built tensors, no
Hub checkpoints, no real image/audio preprocessing — kept identical on purpose so the tables stay
comparable. Two harness bugs in my own scripts were found and fixed mid-round (not shim bugs):
`Wav2Vec2ForCTC` needs `.eval()` or its dropout consumes RNG differently between the two
subprocesses and produces a spurious 0.45 output diff; `WhisperConfig`'s default `pad/bos/eos`
token ids (50256/50257) exceed a 99-token toy vocab and its default 30s mel length (3000 frames →
this test's 64) is below Whisper's hard-coded 128-frame-multiple requirement. Both are recorded
here so a future round does not re-discover them as "regressions."

Every script printed `print("shim" if hasattr(torch._C, "_aten_implemented") else "upstream")` as
line one, both sides, every run, and every dump/log below has it transcribed at the top — no run
printed the wrong label. Environment: `PYTHONPATH=torchnative/src/main
TORCH_USE_RTLD_GLOBAL=1` for the shim, `env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL` for upstream,
never both `torch` variants in one interpreter. `pip` was not run this round, so there is no
`transformers`-replaces-`torch` risk to check provenance against — noted per the brief's warning,
not because it fired. `torch` version used: the repo's pinned upstream (`2.13.0`) via
`/Volumes/macMini/caches/spike-venv`. Every construction+forward wrapped in a 120s `SIGALRM`;
**no hang observed** — every run (17 forward attempts, upstream and shim) returned in well under a
second. Nothing was left running; no background probe processes were started, all runs were
foreground per the instructions.

Build: `cargo build --release` in `rust/torch_c` with `CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-sweep2`,
`TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib` exported before `bash vendor/install_shim.sh`.
Scratch scripts under `/tmp/sweep2/` (not committed): `check_common.py` (19-target pass/fail sweep,
shared by both sides via `PYTHONPATH`), `dump_model.py` + `compare_dumps.py` (numeric comparison
for the 14 that forward on both sides).

## 1. Model table

| model (task) | loads | forwards | matches upstream | refusal (exact) |
|---|---|---|---|---|
| `bert` | yes | yes | **yes** — 44/44 weights identical, max weight diff 0.0, output max abs diff 8.9e-08 | — |
| `roberta` | yes | yes | **yes** — 44/44 identical, max abs diff 7.5e-08 | — |
| `albert` | yes | yes | **yes** — 30/30 identical, max abs diff 5.2e-08 | — |
| `vit` | yes | yes | **yes** — 40/40 identical, max abs diff 2.2e-08 | — |
| `clip` | yes | yes | **yes** — 78/78 identical, `logits_per_image` max abs diff 2.4e-07 | — |
| `wav2vec2` | yes | yes | **yes** (with `.eval()` — see harness note above) — 53/53 identical, weight max diff 3.0e-08, output max abs diff 3.7e-07 | — |
| `qwen2_moe` | yes | yes | **yes** — 35/35 identical, max abs diff 6.0e-08 | — |
| `whisper` (forward) | yes | yes | **yes** (with corrected config/mel-length — see harness note) — 90/90 identical, weight max diff 4.8e-07, output max abs diff 8.9e-08 | — |
| `whisper` (`.generate()`) | yes | forward yes, **generate no** | n/a | `NotImplementedError: aten op not implemented in torch._C shim: aten.where.ScalarSelf` — same wall as DEMAND4 §2, unmoved |
| `t5` | yes | **yes — newly passing** | **yes** — 50/50 identical, max abs diff 7.2e-07 | — (was blocked on `full_like`, closed since DEMAND) |
| `switch_transformers` | yes | **no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch.greater(...) -- overload resolution has no table entry for this op` — same wall as DEMAND4 §2, unmoved |
| `bart` | yes | **yes — newly passing** | **yes** — 95/95 identical, max abs diff 6.0e-08 | — (was blocked on `new_zeros`, closed since DEMAND) |
| `mbart` | yes | **yes — newly passing** | **yes** — 99/99 identical, max abs diff 6.0e-08 | — (was blocked on `squeeze.default`, closed since DEMAND) |
| `pegasus` | yes | **yes — newly passing** | **yes** — 95/95 identical, max abs diff 8.9e-08 | — (was blocked on `Tensor(ndarray)`, closed since DEMAND) |
| `resnet` | yes | **no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch._C._nn.adaptive_avg_pool2d` — same wall as DEMAND4 §2, unmoved |
| `mobilenet_v2` | yes | **no** | n/a | same as `resnet`: `torch._C._nn.adaptive_avg_pool2d` — same wall as DEMAND4 §2, unmoved |
| `convnext` | yes | **yes — newly passing** | **yes** — 30/30 identical, max abs diff 2.2e-08 | — (was blocked on `linspace`, closed since DEMAND) |
| `swin` | yes | **no** | n/a | `NotImplementedError: not implemented in torch._C shim: torch.roll(...) -- overload resolution has no table entry for this op` — DEMAND said `swin` failed at **construction** on `meshgrid`; that closed, and `swin` now constructs and reaches forward, stopping at a **new**, later wall (`roll`, inside the cyclic-shift window attention) — the same wall DEMAND4 already recorded, unmoved by this round |
| `sentence_embed` | yes | **yes — newly passing** | **yes** — 39/39 identical, max abs diff 6.0e-08 | — (was blocked on `linalg_vector_norm`, closed since DEMAND) |

**6 models newly pass end to end since DEMAND.md: `t5`, `bart`, `mbart`, `pegasus`, `convnext`,
`sentence_embed`.** `whisper`'s plain forward also now matches (it already "forwarded" in DEMAND —
this round additionally confirms the numeric match, which DEMAND already had). `swin` moved from
failing at construction to failing at forward — real progress, not yet a pass. Of 19 targets:
**13 forward and match upstream** (`bert`, `roberta`, `albert`, `vit`, `clip`, `wav2vec2`,
`qwen2_moe`, `whisper` forward, `t5`, `bart`, `mbart`, `pegasus`, `convnext`, `sentence_embed` —
that's 14, correcting the count: 14 of 19). **1 forwards but `.generate()` does not** (`whisper`).
**4 do not forward** (`switch_transformers`, `resnet`, `mobilenet_v2`, `swin`), each on exactly the
wall DEMAND4 already named for it. No new, previously-unseen wall was found this round — every
refusal above is one DEMAND4 already recorded.

## 2. Ranked list of what is missing

**Note on "in flight":** another agent is closing `adaptive_avg_pool2d`, `where.ScalarSelf`,
`roll`, and `greater` in a separate worktree right now. This build does not have them — confirmed
directly (`overloads.json` has no `roll` or `greater` key; `where.ScalarSelf` is declared but not
in `aten.rs`'s dispatch `match`; `adaptive_avg_pool2d` is absent from both `overloads.json` and the
`torch._C._nn` bootstrap table). Their absence here is not news; it is the same DEMAND4 state,
confirmed. The ranking below is built to be read **after** they land — i.e. it ranks what would be
newly exposed once they close, not what is open today in some other order.

| rank | gap | models wanting it | kind | why this position, and what happens once it closes |
|---|---|---|---|---|
| 1 | `torch._C._nn.adaptive_avg_pool2d` | `resnet`, `mobilenet_v2` | missing kernel — absent from `overloads.json` entirely (not a wiring gap like `squeeze` was), a new leaf op the way `adaptive_avg_pool1d` was in DEMAND1 | Highest model count (2) of anything open. Once closed, both models' *next* wall is unmeasured by this round — this sweep does not know what lies past the pooler for either, because both currently stop exactly there. That is a real gap in this document: closing rank 1 does not guarantee either model reaches "matches upstream," only that it reaches a new refusal or a pass, and only a re-run will say which. |
| 2 | `aten.where.ScalarSelf` | `whisper` (`.generate()`) | unbound member — declared in `overloads.json` (`aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor` is right there next to `.default`/`.self`/`.ScalarOther`, all three of which already work) but no arm in `aten.rs`'s dispatch `match`. Structurally identical to the `squeeze.default` gap DEMAND3/DEMAND4 both ranked #1 — declared-but-unreachable is this project's single most recurring gap shape. | Once closed, `whisper.generate()` is the only target with a *further* unknown — HF's generation loop calls many ops per step; DEMAND4 only got as far as this one wall. A pass here does not mean `.generate()` matches upstream token-for-token; only `t5`-style forward-only targets have had that verified this round. |
| 3 | `torch.roll` | `swin` | missing spelling — absent from `overloads.json` outright (same absence-shape as `greater`, not the same shape as `where.ScalarSelf`) | Reach is currently exactly 1 model, but `swin` is the only Transformer-style vision model in this set with windowed attention, so this is the last wall standing between `swin` and either a pass or a genuinely new refusal — worth watching closely next round since `swin` has moved twice already (construction → forward) in two rounds. |
| 4 | `torch.greater` | `switch_transformers` | missing spelling — same absence-shape as `roll`, used in MoE expert-choice routing | Narrowest reach (1 model, and only inside routing logic that fires once per forward). Lowest priority of the four by reach, unchanged from DEMAND4's own ranking. |

**Kind taxonomy used above**, per the task's request to distinguish missing kernel / missing
spelling / unbound member / promotion rule / meta kernel / structural: none of the four open items
this round are a promotion rule, meta kernel, or structural gap (the one structural gap this
project has seen, `Tensor(ndarray)`, closed between DEMAND and DEMAND4 and is now in the "closed
already" list below). Two are **missing spelling** (`roll`, `greater` — the op or function does
not appear in the dispatch tables at all yet, the same shape `meshgrid` and `as_tensor` were before
they closed). One is an **unbound member** (`where.ScalarSelf` — declared, not wired; `squeeze`'s
former shape). One is a **missing kernel** (`adaptive_avg_pool2d` — no leaf implementation
anywhere, the same shape `adaptive_avg_pool1d` was before DEMAND1 closed it).

**docs/verification/GOLDEN.md's blind spot, and why it does not invalidate this table:** GOLDEN.md documents
that `tools/golden/compare.py`'s 2811 (now 8469) cases called `_aten_dispatch` **positionally**,
never exercising `bootstrap.py`'s keyword-argument path through `interned_name()` — a wrong
interned string there would pass every golden case and still break real calls. This sweep's
targets all go through real `transformers` forward code, which calls through `bootstrap.py`'s
`resolve()`/`dispatch(key, **bound)` — i.e. this measurement **does** exercise the path GOLDEN.md
found blind, and is not itself blind to that particular hole. What this sweep's own ad-hoc probing
*does* miss, per docs/architectures/DEMAND.md's own running note: an op that is a **spelling** (composite
decomposing entirely into already-implemented ops, e.g. `as_tensor`, `meshgrid`) never moves the
`ops covered` counter and is invisible in that number — the only way to see it close is a model
that previously refused now forwarding, exactly the method this document uses. This round's "6
newly passing" list is therefore also the only trustworthy record of what `meshgrid`/`as_tensor`-
shaped closures bought — `ops covered` alone (168 → 189 → 197 across the referenced rounds) would
undercount them.

## 3. DEMAND4 rows already closed when it was written — the concurrency finding

Measured directly against this build: of DEMAND4's 8-row open list, **4 were already closed**
before or as it was written — `squeeze()` (`.default`), `torch.Tensor(ndarray)`, `torch.linspace`,
and `aten.linalg_vector_norm.default`. All four are now confirmed working end to end above
(`mbart`, `pegasus`, `convnext`, `sentence_embed` respectively forward and match upstream on
exactly the operations those rows named).

**Why the document said otherwise is not "staleness" in the usual sense** — the usual failure mode
this project's own CLAUDE.md names is "a fix lands and the document is never revisited." That is
not what happened here. DEMAND4 was itself a round that *closed five gaps and re-measured the
walls behind them* (`max_pool2d`, `where.default`, `hardtanh`, `adaptive_avg_pool1d`, `one_hot`) —
it was actively being written from a live build at the time. The four rows it listed as open
(`squeeze`, `Tensor(ndarray)`, `linspace`, `linalg_vector_norm`) were being closed by **other
rounds running concurrently with it**, against different worktrees, and DEMAND4's author measured
its own worktree's state — correctly, for that worktree, at that moment — and wrote it down. By
the time DEMAND4 existed as a file, at least one sibling round had already merged a fix for
something DEMAND4 had just recorded as open. The document was accurate to its own inputs and wrong
about the shared state the moment concurrent work landed.

This means the drift mechanism this document set was built to catch has a second variant beyond
"nobody revisits it": **the list is a snapshot of one worktree's build, and this project runs
several worktrees closing different gaps from the same list at the same time, so a snapshot is
stale by the time it is read even if it was accurate the instant it was taken.** No single round's
process was at fault — DEMAND4 measured correctly against what it had. The fix is not "measure
more carefully" (DEMAND4 did); it is that any list ranked "by current relevance" needs either a
build-identifying stamp (which commit/worktree it was measured against) or must be treated as
provisional until the next round's from-scratch re-measurement — which is what this document did
by re-deriving everything from a fresh build rather than editing DEMAND4's table in place. The same
caution applies going forward: **this document's own ranking in §2 is a snapshot of `017a8b6`
plus the stated absences, and the agent named in the task brief is closing exactly the four rows
ranked above as this is being written** — the "why this position" column for each of the four
already says what is unmeasured once it lands, for exactly this reason.

## 4. What was not run, and why

- No Hub checkpoint, real tokenizer, or real image/audio file was used anywhere in this round —
  out of scope per docs/architectures/DEMAND.md §3/docs/architectures/DEMAND3.md §1, kept identical on purpose.
- `whisper.generate()`'s numeric match to upstream was not attempted — it does not run far enough
  under the shim to produce output to diff (stops on `where.ScalarSelf`), so there is nothing to
  compare yet, per this project's own "n/a where forward did not succeed" convention.
- `resnet`, `mobilenet_v2`, `switch_transformers`, `swin` likewise have no numeric comparison —
  none of the four forwards.
- The `TorchnativeConfig("q8_0")` quantised-load path from docs/architectures/DEMAND.md was not re-run this
  round; nothing in the four open/closed gaps touches that code path and DEMAND.md already
  established it bit-identical against `quantize_`, so re-running it would not have told the
  ranking anything new. Flagging the omission rather than silently dropping the row.
- `float8_e4m3fn` / non-default-dtype construction was not exercised — no target in this set
  requests a non-default dtype, matching every prior round's note on this.
- No performance/overhead measurement (`@HighOverheadNativeCall` sites, FFI-per-call cost) was run
  this round — this task is a correctness/coverage sweep, not the performance-measurement class of
  task CLAUDE.md §3 asks for separately, and mixing the two would have violated the "measurement
  work runs alone" rule by adding unrelated model-loading load during a timing run (moot here since
  no timing run happened, but noted for the next round that might combine them).

## 5. Gates (unchanged — no source was touched)

```
380 ok
DOCWATCH: PASS -- 323/323
8469/8469 cases passed, 0 failed, ops covered=197
```

All three ran once, at the end, against the same build used for the model sweep above (no rebuild
in between). Numbers are unchanged from docs/architectures/DEMAND4.md §4's own gate line, as expected — this
round changed no source under `rust/torch_c/src/` or elsewhere; only `/tmp/sweep2/*.py` scratch
scripts and this document were written.
