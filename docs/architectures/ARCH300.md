# ARCH300 — the sweep taken a third time: 290 of 297 forward, 7 blocked

Worktree `work/arch300` on develop `b33e2ee` (vendored tree assembled fresh). torch 2.13.0,
`transformers` 5.15.1, upstream (`/Volumes/macMini/caches/spike-venv/bin/python`) — same venv
ARCH100 and ARCH200 used, unmodified since 2026-08-23. No Rust, `bootstrap.py`, `aten.rs`, or
`tools/` was changed in this round. Golden stays at **11336/11336, ops=299**, exactly unmoved.

**This is a snapshot of this checkout, not of develop tonight.** Three other agents were landing
operator work in parallel worktrees while this document was written. The suite gate below (868
ok, DOCWATCH 777/777, EXIT=0) was measured on this worktree's own freshly built shim; a later
merge can only raise the forward count, never lower it, which is why every marker this document
adds is `ge`.

---

## 0. The control: reused, not re-run — and why

The 290/297 figure the coordinating session handed off was built on a shim-side sweep
(`/tmp/shim-sweep3.json`, 528 entries, taken 04:16) compared against an **upstream control
reused** from `/tmp/upstream-sweep2.json` (528 entries, taken 01:42 the same day), on the grounds
that upstream torch 2.13.0 and `transformers` had not changed between the two runs. ARCH200 §6 was
explicit that reuse is not automatically safe — a nine-month-old upstream run would have graded
this round's shim against a `transformers` version the venv no longer matches — so that reuse was
checked here rather than trusted:

* `spike-venv`'s `python` binary carries an mtime of 2026-08-23, six weeks before either sweep;
  nothing installed or upgraded the venv between the control's run and this document.
* `torch.__version__` and `transformers.__version__` read `2.13.0` / `5.15.1` right now, matching
  what both prior sweeps recorded.
* Five architectures were sampled at random from `upstream-sweep2.json`'s 528 entries
  (`ernie4_5_vl_moe`, `biogpt`, `ministral3`, `llava_next`, `kimi_k25_vision` — two failing
  constructions/forwards, three passing) and re-run individually against the live upstream venv
  with `arch_sweep.py --one`. All five reproduced the stored status and stage exactly, including
  the two failure tracebacks matching to the line.

**Accepted the reused control.** The venv is unmodified and the spot-check reproduces it exactly;
re-running the full 528-architecture upstream sweep here would spend the better part of an hour
re-measuring a population that provably has not moved, for a document whose territory excludes
`arch_sweep.py` and the shim it measures. The denominator (297) and exclusion count (231) below
are therefore the same as ARCH100 and ARCH200's, not re-derived — as before, that identity is
itself a check that the excluded population did not silently change between rounds, not an
assumption.

## 1. The two headline numbers

```text
architectures swept                       528   (unchanged: every model type AutoModel can build)
  forward on UPSTREAM torch (baseline)    297   <- the denominator, unchanged from ARCH100/ARCH200
  fail on upstream too (not our gap)      231   <- unchanged, same exclusion set

of the 297 upstream-clean architectures:
  FORWARD UNDER THE SHIM TODAY            290   (98%)   <- was 270 (91%) in ARCH200, 215 (72%) in ARCH100
  blocked by the shim                       7   <- was 27, was 82

DISTINCT MISSING OPERATORS                  6
```

**290 of 297 forward; the remaining 7 need operators or argument forms across 6 distinct walls**
(one wall — `Tensor.new_zeros` with a tuple size — blocks two architectures; every other wall
blocks exactly one). The same two qualifications ARCH100 §1 and ARCH200 §1 carry forward,
restated rather than assumed:

* **6 is a first-wall count.** Closing the top entry reveals whatever stands behind it — §2
  documents this happening twice in this very round (`led`/`longformer` moved off
  `aten.as_strided.default` onto `Tensor.new_zeros`; `fastspeech2_conformer` moved off
  `torch.zeros(...)`'s tuple form onto `torch.repeat_interleave`).
* **A forward is not a match.** This sweep records whether the model runs, not whether its output
  agrees with upstream. A concurrent round is measuring numerical agreement right now; that
  number is not pre-empted here — see §5.

## 2. The seven still blocked

```text
fastspeech2_conformer   torch.repeat_interleave(Tensor repeats)     missing_shim_name
led                     Tensor.new_zeros((tuple))                  argument form
longformer              Tensor.new_zeros((tuple))                  argument form
nystromformer           aten.convolution.default, per-axis-different padding   argument form
sam3_lite_text_text_model   torch.embedding, an argument form the shim refuses argument form
univnet                 TensorBase.unfold                          missing_shim_name
vilt                    torch.multinomial(Tensor, Tensor)          argument form
```

**`led` and `longformer` share one wall exactly** — both stop inside `Tensor.new_zeros` called
with a tuple size argument, the identical refusal text
(`aten::new_zeros(Tensor self, SymInt[] size, ...)`). Together with `vilt`'s
`torch.multinomial(Tensor, Tensor)`, three of the seven are argument-form gaps of the same
shape — an accessor called with a tensor or tuple where the shim's argument-form table has no
matching row, not a missing kernel. `sam3_lite_text_text_model`'s `torch.embedding` gap is a
fourth argument-form refusal, but a different call shape from those three. This is not seven
independent problems: closing `Tensor.new_zeros`'s tuple form clears two architectures in one
patch, and the argument-form class as a whole (four of the seven) is cheaper than a kernel to
close — no new operator, no candle call, just a wider dispatch table entry.

Verified directly against this worktree's freshly built shim
(`PYTHONPATH=.../torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY arch_sweep.py --one <name>`),
not read from a stale JSON — every one of the seven above reproduced its listed wall when run
individually, after `cargo build --release` and `vendor/install_shim.sh` in this worktree.

## 3. The delta against ARCH200 — newly forwarding / moved wall / unchanged

ARCH200 recorded 27 blocked architectures. This round's 20 + 5 + 2 = 27 reproduces that count
exactly, split the same three ways ARCH200 split ARCH100's 82 in its own §3 — a consistency check
that no architecture was silently dropped between rounds.

### 3.1 Newly forwarding — 20 of ARCH200's 27

```text
_vmap_increment_nesting (4->0)   nemotron3_5_asr, nemotron_asr_streaming,
                                  nemotron_asr_streaming_encoder, t5gemma2
TensorBase.index_copy_ (2->0)    aria, aria_text
lstm (2->0)                      parakeet_rnnt, parakeet_tdt
torch._C._nn.upsample_linear1d (2->0)   sam_hq_vision_model, sam_vision_model
torch.conv1d (2->0)              lasr_ctc, lasr_encoder
diag (1->0)                      rwkv
logsumexp (1->0)                 granite_swa
std (1->0)                       gemma3n_text
torch._C._fft.fft_fftn (1->0)    fnet
torch._C._nn.avg_pool2d (1->0)   efficientnet
torch._C._nn.im2col (1->0)       convbert
torch.einsum (1->0)              longt5
backend: aten.embedding.default.unsupported (1->0)   cpmant
```

That is 20 names — 15 of them (everything above except `fnet`, `lasr_ctc`, `lasr_encoder`,
`sam_vision_model`, `sam_hq_vision_model`) were already forwarding in the full 528-architecture
shim sweep taken at 04:16 today (`/tmp/shim-sweep3.json`, 285/297). The remaining five were
individually re-checked against this worktree's shim after the last round of merges and cleared
between then and now — `fnet`, `lasr_ctc`, `lasr_encoder`, `sam_vision_model`,
`sam_hq_vision_model` — bringing 285 to **290**. All five were re-verified directly with
`arch_sweep.py --one` in this session, not assumed from the coordinating session's report.

### 3.2 Moved to a different wall — 5

Still blocked, but the operator standing in front of the architecture changed since ARCH200 —
each represents a real fix landing that ARCH200's headline number cannot show:

```text
led, longformer          aten.as_strided.default -> Tensor.new_zeros((tuple))
vilt                      TensorBase._unique2 -> torch.multinomial(Tensor, Tensor)
fastspeech2_conformer     round -> torch.zeros((tuple),...) -> torch.repeat_interleave(Tensor repeats)
univnet                   torch._C._nn.pad -> TensorBase.unfold
```

`fastspeech2_conformer` moved twice inside this window alone: `/tmp/shim-sweep3.json` (04:16)
already showed it past `round` and stopped at `torch.zeros` called with a tuple size; by the time
this document's own individual re-check ran, that argument form had cleared too and the wall was
`torch.repeat_interleave(Tensor repeats)`. `led` and `longformer` moved the same way, but only
between `shim-sweep3.json` and this document's own re-check — both were still at
`aten.as_strided.default` at 04:16.

### 3.3 Unchanged, same wall — 2

```text
nystromformer                aten.convolution.default, asymmetric per-axis padding — unchanged
sam3_lite_text_text_model    torch.embedding, an argument form — unchanged
```

`sam3_lite_text_text_model` had already moved once before ARCH200 was written (ARCH200 §3.2:
one `unsupported_arg_form` of `torch.embedding` closed, immediately exposing a different one of
the same call) and stayed at that second argument form through this round — no fix has landed for
it since ARCH200.

`20 + 5 + 2 = 27` — ARCH200's exact blocked count, split three ways. No architecture appeared in
the upstream-clean set that ARCH200's 27 did not already account for.

## 4. What did not change, and should not have — the honesty items ARCH100 and ARCH200 carry

* **A first-wall count is a lower bound.** Closing the top entry can reveal another wall behind
  it, as §2 and §3.2 both show happening inside this very round.
* **Only `AutoModel` bodies are swept — no task heads, no generation.** A model that builds and
  forwards its base body says nothing about a `...ForCausalLM` head or a `.generate()` loop built
  on top of it.
* **Random weights may not reach every data-dependent branch.** A forward that runs to completion
  on randomly-initialized weights can still take a code path that real checkpoint weights would
  not, or skip one they would hit.
* **A forward is not a match.** This document measures reachability — does the model run at all —
  not whether its output numerically agrees with upstream. A separate, concurrent round is
  measuring that agreement right now; **its result is not pre-empted here**. As of ARCH200, 26
  architectures had been checked for numerical agreement; that count is left to the round doing
  the measuring rather than restated or guessed at in this document.
* **Golden: 11336/11336 cases, ops=299 — exactly unmoved.** No Rust, `bootstrap.py`, or `aten.rs`
  was touched in this worktree; `git status --short` before writing this document showed changes
  confined to this document's own territory (`README.md`, `docs/architectures/ARCH300.md`, a pointer added to
  `docs/architectures/ARCH200.md`'s head, `docs/platform/RELEASE_0_0_13a0.md`).
* **Suite gate: 868 ok, `DOCWATCH: PASS` 777/777, `EXIT=0`**, measured in this worktree on a
  freshly built shim (`cargo build --release` then `vendor/install_shim.sh`,
  `CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-arch300`), with `PYTHON=$PY` set for
  `run.sh` — its default `python3` lacks numpy in this environment.

## 5. How this round was taken, and how to take it again

Same method as ARCH100 §7 / ARCH200 §6 — `arch_sweep.py` needed no changes this round either:

```text
PY=/Volumes/macMini/caches/spike-venv/bin/python
cd rust/torch_c/pytests

PYTHONPATH=$REPO/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY arch_sweep.py --out /tmp/shim.json
env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL          $PY arch_sweep.py --out /tmp/upstream.json
$PY arch_sweep.py --compare /tmp/shim.json /tmp/upstream.json

# to re-check one architecture against the live shim without a full 528-way sweep:
PYTHONPATH=$REPO/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY arch_sweep.py --one <name>
```

This round did not re-run the full upstream sweep — §0 records exactly why the reuse was
considered safe to accept rather than simply assumed. The shim side was re-swept in full at 04:16
today, and the seven architectures that decide the headline number here were each re-checked
individually against this worktree's own freshly built shim after the last merges, rather than
read from that earlier JSON.

## 6. Gates

```text
rust/torch_c/pytests/run.sh    868 ok, 0 FAIL, exit 0
DOCWATCH                       PASS -- 777/777 evaluated marker(s) hold
tools/golden/compare.py        11336/11336 cases passed, 0 failed, ops covered=299, pending=0
```

All three measured on the freshly built `lib_C.dylib` in this worktree
(`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-arch300`, `TORCH_C_ARTEFACT` set
explicitly per ARCH100 §8's trap — an unset `TORCH_C_ARTEFACT` falls back to the shared cache
binary and reports a plausible but wrong number).

Golden is exactly unmoved from this worktree's starting point (11336/11336, ops=299), which is the
correct result: this round changed no Rust, `bootstrap.py`, `aten.rs`, or `tools/`. The seven names
in §2 are data for the next batch, the same way ARCH100's 31 and ARCH200's 16 were — this round
measured and implemented nothing. **These numbers are a snapshot**: three other agents were
landing operator work in parallel worktrees while this document was written, so every marker below
is `ge`, not `eq` — a later round raising the forward count is progress, not a contradiction of
this one.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/arch_sweep.py classify present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/arch_sweep.py verify_random_weights present -->
<!-- DOCWATCH: count smoke_ok ge 480 -->
<!-- DOCWATCH: count golden_cases_passed ge 11336 -->
<!-- DOCWATCH: count golden_ops_covered ge 299 -->
<!-- DOCWATCH: count golden_pending eq 0 -->
