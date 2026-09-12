# AGREE2 — the agreement sweep taken again: 288 of 290 judgeable architectures match, and all 297 now replay

Worktree `work/agree2` on develop `44f4dec`, vendored tree assembled fresh (`vendor/vendor_torch.sh`
then `vendor/install_shim.sh`) so every number below is this checkout's own `lib_C.dylib`. torch
2.13.0 upstream (`/Volumes/macMini/caches/spike-venv/bin/python`) is the oracle throughout, same
venv `docs/architectures/ARCH100.md`, `ARCH200.md`, `ARCH300.md` and `docs/numerics/AGREE.md` used.
No Rust, `bootstrap.py` or `aten.rs` was changed: this round measures, and adds
`rust/torch_c/pytests/agree2_scores.json`, `rust/torch_c/pytests/test_agree2.py`, the
`agree_*` count sources in `tools/docwatch/check_docs.py`, and this document.

**Measured 2026-09-12.** Every count here was taken this round unless it is explicitly dated to an
earlier one.

---

## 0. Why this round exists, and the failure that prompted it

The coordinating session tried to pick the next piece of work out of the documents and could not,
for two reasons that are both about how numbers are stored rather than about the shim:

* It read *"82 architectures blocked"* out of `docs/architectures/ARCH100.md` and proposed acting
  on it. That document is a deliberately frozen baseline and says so in its own banner — but the
  banner pointed only as far as ARCH200 (270/297), and ARCH300 (290/297) carried no banner at all.
  Three superseded headlines, and the most recent one was itself stale. §6 fixes that chain.
* It then tried to re-measure instead and hit `FileNotFoundError: '/tmp/agree/_produce.json'`.
  The previous round's artefacts lived in `/tmp`, **no DOCWATCH marker pinned any agreement
  count**, and `grep`ing the docs for `count .*agree.*` returned nothing. So the state of the
  strongest claim in `docs/platform/RELEASE_0_1_0b0.md` was not knowable without re-running a
  multi-hour sweep. §5 is the fix.

## 1. The headline

```text
architectures attempted                       528   (every model type AutoModel can build)
  forward on UPSTREAM torch (the denominator) 297   <- unchanged from ARCH100/200/300 and AGREE
  replayed under the SHIM on those weights    297   <- 100%, was 290 in AGREE (2026-08-24)

of the 297, after excluding what cannot be judged:
  exact  (bit-identical)                        0
  agree  (within 1.186e-06 relative)          262
  agree_within_float32                         26   <- outside it, but no further from the
                                                       float64 truth than upstream itself is
  DIVERGE                                       2   <- chinese_clip, fastspeech2_conformer
  degenerate    (output underflowed)            3   } not counted either way
  nondeterministic (upstream re-samples)        4   }

  288 of 290 judgeable architectures AGREE  (99.3%)
```

<!-- DOCWATCH: count agree_produced_upstream ge 297 -->
<!-- DOCWATCH: count agree_replayed_shim ge 297 -->
<!-- DOCWATCH: count agree_scored ge 297 -->
<!-- DOCWATCH: count agree_agree ge 288 -->
<!-- DOCWATCH: count agree_judgeable ge 290 -->
<!-- DOCWATCH: count agree_diverge le 2 -->
<!-- DOCWATCH: count agree_unjudgeable le 7 -->
<!-- DOCWATCH: count agree_state_dict_clean ge 297 -->

The markers are `ge` on everything a later round could legitimately raise and `le` on the two that
must not grow — divergence and unjudgeability. That is CLAUDE.md's rule, and it is the right way
round here: an operator landing can only add architectures to the population, never remove one.

**Nothing that agreed before disagrees now.** The one change in the `diverge` column is an
architecture that previously could not be measured at all (§3), not a regression.

## 2. What moved, and in which direction

`docs/numerics/AGREE.md` (2026-08-24, develop `b33e2ee`) is the baseline. Every row below names
the document the old number came from.

| | AGREE.md, 2026-08-24 | this round, 2026-09-12 | direction |
|---|---|---|---|
| forward upstream | 297 | 297 | control — unchanged, as it must be |
| replayed under the shim | 290 | **297** | +7, the whole of AGREE.md §7's blocked list |
| scored | 290 | **297** | +7 |
| judgeable (denominator) | 285 | **290** | +5 |
| agree | 284 | **288** | +4 |
| diverge | 1 | **2** | +1, and it is a newly-judgeable one |
| unjudgeable | 5 | **7** | +2, both newly-replaying |
| no float64 oracle | 22 | **23** | +1, same cause |
| architectures with a working oracle | 263 | **267** | +4 |
| tolerance (p90 of upstream's own f32 error, floored at 8 ulp) | 1.186e-06 = 10 ulp | **1.186e-06 = 10 ulp** | unmoved |
| `state_dict` transferred cleanly | 290/290 | **297/297** | still no missing or unexpected key anywhere |

The seven that `docs/numerics/AGREE.md` §7 listed as *"forward upstream and not under the shim …
unchanged and unmeasured here"* all replay now, and this is the first round to score them:

| architecture | what blocked it (AGREE.md §7) | verdict this round |
|---|---|---|
| `led` | `new_zeros` | `agree`, rel 1.950e-07 |
| `longformer` | `new_zeros` | `agree`, rel 1.274e-07 |
| `nystromformer` | asymmetric convolution padding | `agree`, rel 1.770e-07 |
| `sam3_lite_text_text_model` | asymmetric convolution padding | `agree`, rel 1.476e-07 |
| `univnet` | `TensorBase.unfold` | **`nondeterministic`** — upstream self-repeat 8.11e-03 |
| `vilt` | `multinomial` | **`nondeterministic`** — upstream self-repeat 1.60 |
| `fastspeech2_conformer` | `repeat_interleave` with a tensor `repeats` | **`diverge`** — §3 |

Four agree at one to two ulp. Two turn out to be architectures upstream cannot reproduce against
itself, so they join `vit_mae` and `vits` in the refusal column rather than the pass column. One
diverges.

The tolerance being *identical* to the previous round's, on a population four architectures
larger, is worth one sentence: it is read off the p90 of upstream's own float32-vs-float64 error
and that distribution did not move (median 2.73e-07, p90 1.19e-06, p99 6.07e-06, max 6.08e-04 over
267 architectures, against 2.74e-07 / 1.19e-06 / 6.07e-06 / 6.08e-04 over 263). Upstream was not
touched, so this is the control behaving.

## 3. The second divergence: `fastspeech2_conformer`, and it is not an operator

`fastspeech2_conformer` [`pitch_outputs`] is `diverge` at rel **7.934e-06**, absolute **9.969e-06**
on a tensor of scale 1.256 — 66 ulp end to end, on an 8-element auxiliary head. Its other outputs
are much closer: `spectrogram` 2.757e-06, `encoder_last_hidden_state` 1.634e-06, `energy_outputs`
3.219e-06, `duration_outputs` **bit-identical**. The architecture's score is the worst key, which is
`pitch_outputs`.

It is `diverge` rather than `agree_within_float32` **because it has no float64 oracle** (§4), so
the ratio rule that covers the other 26 above-tolerance architectures cannot be applied to it. That
is the rule working as designed — the weaker test is the stricter verdict — but it means this flag
carries less information than a flag on an architecture that has an oracle.

Attribution, by AGREE.md §4's technique (upstream records every leaf module's input and output; the
shim re-runs each module in isolation on upstream's recorded input, so nothing accumulates):

```text
architecture            leaves  worst single module                                  rel        ulp
fastspeech2_conformer      250  Conv1d  decoder.conformer_layers.1...pointwise_conv1  3.279e-07  2.75
                                LayerNorm encoder.conformer_layers.2.ff_layer_norm    2.934e-07  2.46
                                Conv1d  decoder.conformer_layers.0...pointwise_conv1  2.363e-07  1.98
```

**250 leaf modules, worst 2.75 ulp, nothing above 3.3e-07.** The same negative result AGREE.md §4
reached for its seven: the end-to-end number is float32 accumulation over a 4+4-layer conformer
stack feeding a length regulator, not a wrong kernel. `Conv1d` and `LayerNorm` fill the table, which
are the ops `docs/architectures/DEMAND8.md` §1.3 and `docs/devices/MPSFWD.md` already measured at
this scale.

`chinese_clip` is **unchanged and still flagged**: rel 2.873e-06, absolute 1.401e-06 on scale 0.488,
upstream's own oracle error 6.00e-07, ratio 4.79 — the same four figures `docs/numerics/AGREE.md`
§3.1 recorded, reproduced to every printed digit from a fresh vendored tree. It stays flagged for
the reason it was flagged then: a rule that is edited whenever it fires is not a rule.

## 4. The 23 with no float64 oracle — and 21 of them can have one

`docs/platform/RELEASE_0_1_0b0.md` §0.1 says *"22 MoE models have no float64 oracle at all"*. Two
things in that sentence are wrong, and the second one matters.

**It is 23, and they are not all MoE.** Grouped by the exact exception upstream raises:

| n | upstream's own error when the model is run in float64 | models |
|---|---|---|
| 20 | `Expected mat_a to be Float32, BFloat16 or Float16 matrix, got Double` | `afmoe`, `axk2`, `ernie4_5_moe`, `exaone_moe`, `glm4_moe`, `glm4_moe_lite`, `hy_v3`, `laguna`, `mellum`, `mistral4`, `nemotron_h`, `qwen2_moe`, `qwen3_5_moe`, `qwen3_5_moe_text`, `qwen3_moe`, `qwen3_next`, `qwen3_vl_moe`, `qwen3_vl_moe_text`, `solar_open`, `zaya` |
| 1 | `value cannot be converted to type float without overflow` | `xglm` |
| 1 | `mat1 and mat2 must have the same dtype, but got Float and Double` | `mra` |
| 1 | `Input type (float) and bias type (double) should be the same` | `fastspeech2_conformer` |

**And an oracle is constructible for 21 of the 23.** This was measured this round, not reasoned
about:

* **The 20 grouped-`mm` models are blocked by a kernel dtype restriction, not by MoE.** `transformers`
  already exposes the alternative — `model.set_experts_implementation("eager")`
  (`modeling_utils.py:1965 get_correct_experts_implementation`, whose own fallback path does exactly
  this when `_grouped_mm_can_dispatch()` refuses). With the experts on the eager path, **all 20 run
  in float64 and yield an oracle**, with upstream's own f32-vs-f64 error between **1.19e-07 and
  1.79e-06** — the same distribution as the rest of the population, all but three of them inside the
  10-ulp tolerance.
* **Switching to the eager experts barely moves the float32 answer either**: `rel` between the
  default grouped path and the eager path is **0.0 for five of the 20 and at most 2.42e-07 (2 ulp)**
  for the rest. So the oracle is an oracle *for the same computation*, not for a different model.
* **`xglm` yields an oracle under `torch.set_default_dtype(torch.float64)`** (its blocker is
  `torch.full((), torch.finfo(attn_weights.dtype).min)` at `modeling_xglm.py:209` building the mask
  sentinel at the *default* dtype, then overflowing float32). Upstream error 1.30e-07.
* **`mra` and `fastspeech2_conformer` cannot, without editing upstream's source.** Both hard-code
  float32 in the module body, so no dtype setting reaches them:
  * `modeling_mra.py:590-593` calls its approximation kernel as
    `mra2_attention(query_layer.float(), key_layer.float(), value_layer.float(), attention_mask.float(), …)`
    — an unconditional cast, and the `Float`/`Double` mismatch then surfaces one layer later at
    `MraSelfOutput.dense`.
  * `modeling_fastspeech2_conformer.py:115-118`'s length regulator allocates its output buffer as
    `torch.zeros(…, dtype=torch.float)` — literal, not `default_dtype` — so the decoder receives
    float32 activations while its own `Conv1d` is double.

**So the honest statement is not "MoE has no oracle".** It is: *twenty-one of the twenty-three have
one available and this harness does not yet take it, and two are prevented by a hard-coded
`float32` in upstream's own module source.* That is a harness gap, listed in §7 as the first thing
the next round should close — and it matters for exactly one verdict, because
`fastspeech2_conformer` is one of the two that genuinely cannot have an oracle and is also the new
`diverge`.

**On the routing question**, which was the interesting hypothesis and did not survive: MoE routing
*is* discrete, and a float64 run *could* in principle select a different expert from a float32 one,
which would make any oracle meaningless. It does not happen here. If it did, the float32-eager
versus float32-grouped comparison and the oracle numbers would be order 1; they are at most 2 ulp
and 1.8e-06 respectively. At these shrunk configs with random weights the top-k decision is stable
across the two paths. This is **not** a claim that it is stable in general — a router whose top two
logits are within float32 noise will flip, and nothing here measured a trained checkpoint or a long
sequence.

<!-- DOCWATCH: count agree_no_oracle le 23 -->

## 5. The 7 that cannot be judged — each one re-examined

*"Excluded from the denominator"* is the honest move only while an architecture is genuinely
unjudgeable, so each was re-checked rather than carried forward.

| architecture | verdict | the measurement | still unjudgeable? |
|---|---|---|---|
| `efficientnet` | `degenerate` | output scale **8.03e-29** | yes — both sides underflow; rel 1.16e-06 is a comparison of denormals |
| `sam_vision_model` | `degenerate` | scale **7.98e-21** | yes |
| `sam_hq_vision_model` | `degenerate` | scale **7.98e-21** | yes |
| `vit_mae` | `nondeterministic` | upstream self-repeat **1.36** | yes — re-draws its patch mask every forward |
| `vits` | `nondeterministic` | self-repeat **inf** | yes — samples a duration; the two runs are different lengths |
| `univnet` | `nondeterministic` | self-repeat **8.11e-03** | **new this round** |
| `vilt` | `nondeterministic` | self-repeat **1.60** | **new this round** |

Two findings sit in that table.

**No previously-unjudgeable architecture has become judgeable, so the denominator is not
flattering.** The three `degenerate` ones are degenerate for the reason `docs/architectures/DEMAND8.md`
§1.1 identified and the calibration pass of `agree_sweep.py` cannot reach: BatchNorm calibration
fixes uncalibrated *running statistics*, and these three underflow through depth with the statistics
already calibrated. Their upstream-vs-float64 errors (2.14e-06, 1.64e-06, 1.64e-06) confirm upstream
is equally in the noise there — it is not the shim's scale.

**`univnet` is the weakest of the seven refusals and should be read as such.** Its self-repeat is
8.11e-03, not order 1 like the other three: upstream mostly reproduces itself and differs in the
third digit. That is above the 1e-06 threshold the report uses, so the mechanical rule refuses it,
and the rule is not being edited. But *"upstream does not reproduce itself at all"* (`vits`, `inf`)
and *"upstream reproduces itself to three digits"* (`univnet`) are not the same statement, and the
shim's own number against upstream, 7.411e-03, is the same size as upstream's disagreement with
itself — which is what you would expect from an architecture that agrees, measured through a
sampler. **Unresolved**, and listed in §7: the way to settle it is to seed the sampler and re-run,
which the harness does not currently do.

<!-- DOCWATCH: count agree_degenerate le 3 -->
<!-- DOCWATCH: count agree_nondeterministic le 4 -->

## 6. Stale numbers corrected, each with where it lived

| document | what it said | status |
|---|---|---|
| `docs/architectures/ARCH100.md` §1 | "215 of 297 forward; the remaining **82** blocked" | **not stale — correctly frozen.** Its banner already disclaims the headline. The banner itself was stale (it named ARCH200 as the successor); a line pointing on to ARCH300 and here has been added |
| `docs/architectures/ARCH200.md` | "270 of 297 forward, 16 operators missing" | correctly banner-disclaimed, banner points to ARCH300. A line pointing here added |
| `docs/architectures/ARCH300.md` | "**290 of 297 forward, 7 blocked**" | **stale, and carried no banner at all** — this was the one a reader would have trusted. All 297 forward and replay now. Banner added |
| `docs/numerics/AGREE.md` §1, §7 | 290 replayed, 284/285 agree, 5 unjudgeable, the 7 "unchanged and unmeasured" | superseded, not wrong when taken. Banner added pointing here; the file is left unedited as the baseline §2 compares against |
| `docs/platform/RELEASE_0_1_0b0.md` §0.1 | "**22 MoE models** have no float64 oracle **at all**" | **wrong in two ways, and was wrong when written**: it is 23 now and was 22 then, but three of those are not MoE (`xglm`, `mra`, and in this round `fastspeech2_conformer`) — AGREE.md §2 said so and the release note flattened it — and "at all" is false for 21 of the 23 (§4). Corrected in place with a footnote, since it is a factual error rather than a superseded measurement |
| `docs/platform/RELEASE_0_1_0b0.md` §0 | "284 of 285 judgeable architectures agree", "5 could not be judged" | superseded by 288/290 and 7. Dated forward-reference added |

**Not re-measured this round, and therefore cited with its date rather than restated:** *"worst
single-operator error across 775 replayed leaf modules: 8.4 ulp (a `Conv2d` in `hgnet_v2`)"*
(`docs/numerics/AGREE.md` §4, 2026-08-24). This round added 250 more replayed leaves
(`fastspeech2_conformer`, worst 2.75 ulp) but did not re-run the other seven captures, so the
aggregate figure belongs to that round and is left attributed to it.

## 7. Limits, and what is UNVERIFIED

Everything `docs/numerics/AGREE.md` §7 lists still applies in full — `AutoModel` bodies only, no
task head, no generation loop, no KV-cache reuse; shrunk configs and random weights, which reach the
same operators but not data-dependent branches; one checkout at one moment; the oracle is upstream's
float64 rather than exact arithmetic; an architecture's score is the worst key in its output.

New to this round, and each is a thing this round could NOT settle:

* **UNVERIFIED — the 20 grouped-`mm` models still have no oracle in the sweep itself.** §4 shows one
  is constructible and measures it for all 20, but `agree_sweep.py`'s `produce_one` was not changed
  to take it, so those 20 are still scored against the fixed tolerance in §1's table. *What would
  settle it:* fall back to `set_experts_implementation("eager")` for the float64 run only, record
  in the per-architecture record that the oracle came from the eager path, and re-run `--produce`.
  All 20 currently land in `agree` well inside the tolerance, so this cannot change a verdict from
  pass to fail — it can only strengthen 20 weak passes and, for `fastspeech2_conformer`, it
  provably cannot help at all.
* **UNVERIFIED — whether `univnet` agrees.** §5. *What would settle it:* seed its sampler (or run it
  under `torch.use_deterministic_algorithms`) on both sides and re-score.
* **UNVERIFIED — the aggregate worst-operator figure.** §6's last paragraph. *What would settle it:*
  re-run `--capture`/`--bisect` for the seven of AGREE.md §4 on this tree.
* **UNVERIFIED — whether MoE routing is stable at trained weights or longer sequences.** §4 measures
  stability only at these shrunk configs with random weights.
* **The counts are pinned to a recorded measurement, not to a live one,** and only in one
  direction. §1's markers re-derive the verdicts from `rust/torch_c/pytests/agree2_scores.json` by
  calling `agree_sweep.verdict`, so deleting a refusal branch or dropping architectures from the
  recording fires them (both nullified and confirmed red). **Widening the rule does not fire them
  and cannot** — `ge 288` and `le 2` accept anything that moves architectures *into* `agree`, which
  is the price of using `ge` so a later round can legitimately raise the count. That direction is
  held by `test_agree.py::test_the_oracle_factor_is_stated_and_is_not_a_free_parameter` instead;
  the two are only jointly sufficient. And neither can see the tree moving under a measurement
  nobody retook — the date in the artefact is the only guard against that. `agree_sweep.py` needs both interpreters and hours; it is not
  wired into `run.sh` and must not be.

## 8. Where the artefacts are, and how to take this round again

The previous round's artefacts were in `/tmp` and were gone when the next session wanted them.
These are not:

```text
/Volumes/macMini/caches/agree-sweep/       69 MB  inputs, up32, up64, shim32 and the per-module
                                                  capture, plus _produce/_replay/_report and the
                                                  per-architecture records both drivers resume from
rust/torch_c/pytests/agree2_scores.json   128 KB  the per-architecture scores, in the repo
```

The sweep produced **6.2 GB**; the `*.weights.npz` state-dict bundles were **all of it** and have
been deleted, leaving 69 MB. What that costs is exact and worth stating rather than burying:
`--report` still reproduces from what remains (re-run after the prune: same 288 of 290), but
`--replay` and `--bisect` both load the weights, so re-running either now needs `--produce` again
(~26 minutes of upstream compute at these deadlines). That is the trade this round chose against
84 GB of free disk and 146 worktrees. `agree2_scores.json` is committed and is what the counts are
re-derived from, so nothing a reader needs depends on the external directory at all.

To retake the whole thing:

```text
PY=/Volumes/macMini/caches/spike-venv/bin/python
D=/Volumes/macMini/caches/agree-sweep          # NOT /tmp -- 6.2 GB, and /tmp is internal
REPO=$(git rev-parse --show-toplevel)
vendor/vendor_torch.sh && vendor/install_shim.sh     # or the shim side reports "not the shim"
cd rust/torch_c/pytests

env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL $PY agree_sweep.py --produce --dir $D --deadline 520
PYTHONPATH=$REPO/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
                                           $PY agree_sweep.py --replay  --dir $D --deadline 500
$PY agree_sweep.py --report --dir $D
```

Both drivers are resumable, and `--deadline` exists so the sweep can be issued repeatedly **in the
foreground** rather than backgrounded and lost. This round took three `--produce` passes and two
`--replay` passes at those deadlines. Do not filter the population to go faster: CLAUDE.md records a
round invalidated by exactly that.

## 9. Gates

```text
rust/torch_c/pytests/run.sh    1517 ok, 0 FAIL, exit 0      (1504 on develop; +13 from test_agree2.py)
DOCWATCH                       PASS -- 1141/1141 evaluated marker(s) hold   (1126/1126 on develop;
                                                                            +15 markers, 8 of them
                                                                            the counts in §1)
cargo test --release           33 passed, 0 failed
```

Golden was not re-run by this round directly, but `run.sh` runs it: **11478/11478 at ops=304**,
`golden_pending` 0. That is *not* the 11336/299 `docs/numerics/AGREE.md` §9 recorded on 2026-08-24 —
the tree moved between the two rounds, which is the same reason this document exists. This round
changed no Rust, so it did not move it.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/agree_sweep.py verdict present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/agree_sweep.py ORACLE_FACTOR present -->
<!-- DOCWATCH: symbol-in-file tools/docwatch/check_docs.py agree_no_oracle present -->
<!-- DOCWATCH: json-key rust/torch_c/pytests/agree2_scores.json scores present -->
