# ARCH200 — the sweep taken again: 270 of 297 forward, 16 operators missing

> **Re-swept in `docs/ARCH300.md`.** Operator work landed between this round and that one —
> `_vmap_increment_nesting`, `index_copy_`, `lstm`, `upsample_linear1d`, `conv1d`, `diag`,
> `logsumexp`, `std`, `fft_fftn`, `avg_pool2d`, `im2col`, `einsum`, and the `embedding` backend
> refusal on `cpmant`, among others. The headline below (270/297, 16 operators) no longer describes
> the current tree: ARCH300 measures **290/297 (98%), 7 blocked**. This document is left in place
> as the record of what was true at `2498122` — the comparison ARCH300 exists to make depends on
> this baseline staying unedited.

Worktree `work/sweep2` on develop `2498122` (vendored tree assembled fresh). torch 2.13.0
upstream (`/Volumes/macMini/caches/spike-venv/bin/python`). No Rust, `bootstrap.py`, or `aten.rs`
was changed in this round — it re-runs `rust/torch_c/pytests/arch_sweep.py` exactly as
`docs/ARCH100.md` §7 describes and reports the delta. Golden stays at **10039/10039, ops=270**,
exactly unmoved.

**This is a snapshot of this checkout, not of develop tonight.** Four other agents were landing
operator work in parallel worktrees while this sweep ran (`__setitem__`, the scatter family,
`index_select`, `glu`, `linalg_qr`, `linalg_norm`, complex tensors, reflect/replicate padding,
rank-5 `matmul`, `eye`, `erfinv`, `scatter_reduce`, `index_add`, `view_as`, `bitwise_xor`, and
more — the ATen count moved 222 → 270 across those rounds). None of that work is visible here
until it merges; this document measures the tree as checked out, at one point in time.

---

## 1. The two headline numbers

```text
architectures swept                       528   (unchanged: every model type AutoModel can build)
  forward on UPSTREAM torch (baseline)    297   <- the denominator, unchanged from ARCH100
  fail on upstream too (not our gap)      231   <- unchanged, same exclusion set

of the 297 upstream-clean architectures:
  FORWARD UNDER THE SHIM TODAY            270   (91%)   <- was 215 (72%)
  blocked by the shim                      27   <- was 82

DISTINCT MISSING OPERATORS                 16   <- was 31
```

**270 of 297 forward; the remaining 27 need 16 distinct operators.** The denominator held exactly
at 297 (upstream's own set of buildable, forwardable architectures did not change — it cannot,
since upstream was not touched), which is itself a check that this round's exclusion set is the
same 231 architectures ARCH100 excluded, not a different population.

The same two qualifications from ARCH100 §1 still apply and are restated rather than assumed:

* **16 is a first-wall count.** Closing the top entry reveals whatever stands behind it — see §3,
  where seventeen architectures did exactly that this round.
* **A forward is not a match.** This sweep records whether the model runs, not whether its output
  agrees with upstream.

```text
argument forms the shim refuses (the OP exists)     2 distinct,  2 architectures   (was 6/6)
backend refusals (candle layout, not a missing op)  1 distinct,  1 architecture    (was 1/3)
unclassified -- NOT guessed at                                   0 architectures   (was 2)
```

The unclassified bucket that ARCH100 §6 left open (`dinov3_convnext`, `efficientnet`,
`adaptive_avg_pool2d`) is empty now — both moved to classifiable walls (§3), not because the
classifier changed. `arch_sweep.py`'s `_RULES` and `classify()` were not touched this round
(CLAUDE.md's territory for this round excludes `arch_sweep.py` rewrites, and none were needed —
the existing rules matched every refusal text encountered).

## 2. The ranked list — the deliverable

Each row is a distinct missing operator and the number of the 297 upstream-clean architectures
whose forward (or construction) stops there **first**, from this round's actual
`arch_sweep.py --compare` output.

```text
   4  torch._C._functorch._vmap_increment_nesting  missing_shim_name      cum=4
      e.g. nemotron3_5_asr, nemotron_asr_streaming, nemotron_asr_streaming_encoder, t5gemma2
   2  TensorBase.index_copy_                       missing_shim_name      cum=6
      e.g. aria, aria_text
   2  aten.as_strided.default                      missing_aten_op        cum=8
      e.g. led, longformer
   2  lstm                                         missing_aten_op        cum=10
      e.g. parakeet_rnnt, parakeet_tdt
   2  torch._C._nn.upsample_linear1d               missing_shim_name      cum=12
      e.g. sam_hq_vision_model, sam_vision_model
   2  torch.conv1d                                 missing_shim_name      cum=14
      e.g. lasr_ctc, lasr_encoder
   1  _unique2                                     missing_aten_op        cum=15   vilt
   1  diag                                         missing_aten_op        cum=16   rwkv
   1  logsumexp                                    missing_aten_op        cum=17   granite_swa
   1  round                                        missing_aten_op        cum=18   fastspeech2_conformer
   1  std                                          missing_aten_op        cum=19   gemma3n_text
   1  torch._C._fft.fft_fftn                       missing_shim_name      cum=20   fnet
   1  torch._C._nn.avg_pool2d                      missing_shim_name      cum=21   efficientnet
   1  torch._C._nn.im2col                          missing_shim_name      cum=22   convbert
   1  torch._C._nn.pad                             missing_shim_name      cum=23   univnet
   1  torch.einsum                                 missing_shim_name      cum=24   longt5
```

```text
argument forms the shim refuses (the OP exists -- cheaper than a kernel):
   1  aten.convolution.default                     nystromformer
   1  torch.embedding.(Parameter, NoneType, int, bool, bool)   sam3_lite_text_text_model

backend refusals (candle layout, not a missing op):
   1  aten.embedding.default.unsupported            cpmant

unclassified: 0
```

Twenty of ARCH100's thirty-one blocked exactly one architecture each. Now, of the sixteen: **eleven
of sixteen block exactly one architecture each.** The shape held — the tail is still a tail, just
shorter. The head moved: `_vmap_increment_nesting` (4 architectures, all ASR streaming encoders and
`t5gemma2`) is now the single largest wall, where ARCH100's head was `__setitem__` at 13. No
operator blocks more than 4 architectures this round, against ARCH100's 13/7/7/5 top four — the
distribution flattened as well as shrank.

## 3. The delta against ARCH100 — this is the part a headline hides

ARCH100 recorded 82 blocked architectures across its ranked list (71), its two cheaper classes
(6 + 3 = wait, 6 argform + 3 backend = 9, one of which — `hiera`/`olmo_hybrid`/`qwen3_next` under
`backend_limitation` — is 3 architectures at 1 op), and 2 unclassified: 71 + 6 + 3 + 2 = 82,
matching exactly. This round's 55 + 17 + 10 = 82 reproduces that same total, which is itself a
consistency check that no architecture was silently dropped from the population between rounds.

### 3.1 Newly forwarding — 55 of ARCH100's 82

Every one of these was blocked in ARCH100 and forwards under the shim today. Grouped by what
ARCH100 recorded as their wall:

```text
TensorBase.__setitem__ (13->0)   cosmos3_omni, minicpmv4_6, qwen3_5, qwen3_5_moe, qwen3_5_moe_text,
                                  qwen3_5_text, qwen3_vl, qwen3_vl_moe, qwen3_vl_moe_text,
                                  qwen3_vl_text, seamless_m4t, wav2vec2-conformer
                                  (fastspeech2_conformer moved wall instead -- see 3.2)
TensorBase.scatter_ (7->0)       exaone_moe, glm4_moe, glm4_moe_lite, groupvit, mistral4,
                                  nemotron_h, solar_open
torch._C._nn.glu (5 of 7 ->0)    cohere_asr, parakeet_ctc, parakeet_encoder
                                  (lasr_ctc/lasr_encoder and parakeet_rnnt/parakeet_tdt moved
                                  wall instead -- see 3.2)
TensorBase.index_select (5->0)   kosmos-2.5, m2m_100, nllb-moe, seamless_m4t_v2, xglm
aten.scatter.value (4->0)        axk2, deepseek_v32, glm_moe_dsa, jetmoe
argsort (1 of 3 ->0)             vit_mae               (aria/aria_text moved wall -- see 3.2)
aten.where.Scalar (1 of 2 ->0)   git                   (cpmant moved wall -- see 3.2)
bucketize (2->0)                 idefics3_vision, smolvlm_vision
torch._C._linalg.linalg_norm(2)  owlv2, owlvit
TensorBase._is_all_true (1)      vits
TensorBase.masked_scatter (1)    higgs_audio_v2
TensorBase.reshape_as (1)        roformer
TensorBase.unflatten (1)         siglip_vision_model
acos (1)                         yoso
broadcast_tensors (1)            gemma3n_text
chunk (1)                        diffllama
diff (1)                         voxtral_realtime_encoder
max_pool1d (1)                   canine
multiply (1)                     convbert            (reopened at a different op -- see note)
polar (1)                        llama4_text
prod (1)                         tapas
view_as_complex (1)              llama4
argument-form: torch.ones()      gpt_neo
argument-form: Tensor.mean()     ibert
argument-form: torch.mean()      imagegpt
backend: candle matmul striding  hiera, olmo_hybrid, qwen3_next
unclassified: adaptive_avg_pool2d  dinov3_convnext
```

That is 55 architectures across every failure class ARCH100 recorded — construction and forward,
`missing_shim_name` and `missing_aten_op`, both cheaper classes and one of the two unclassified.
`convbert` needs a note: it forwards past `multiply` now but hits a **new** wall
(`torch._C._nn.im2col`, §3.2) on the second attempt inside this same sweep run's classification —
i.e. it is not simply closed, it is double-counted between "newly forwarding on the old wall" and
"moved wall" in the raw data; it is listed once here (newly forwarding past `multiply`, since that
is what ARCH100 recorded) and again in §3.2 for its new wall, because both are true: the operator
ARCH100 named no longer blocks it, and a different one now does.

### 3.2 Moved to a different wall — 17

The group a single headline number hides completely: these did not go from blocked to forwarding,
but the operator standing in front of them changed. Each of these represents at least one real
fix that landed between rounds, even though the architecture is still in the "blocked" column.

```text
aria, aria_text            argsort -> TensorBase.index_copy_       (missing_aten_op -> missing_shim_name)
cpmant                     aten.where.Scalar -> aten.embedding.default.unsupported (-> backend_limitation)
convbert                   multiply -> torch._C._nn.im2col          (missing_aten_op -> missing_shim_name)
efficientnet               adaptive_avg_pool2d (unclassified) -> torch._C._nn.avg_pool2d
fastspeech2_conformer      TensorBase.__setitem__ -> round          (construction wall fell; new forward wall)
gemma3n_text                broadcast_tensors -> std                (both missing_aten_op)
lasr_ctc, lasr_encoder     torch._C._nn.glu -> torch.conv1d         (both missing_shim_name)
led                        TensorBase.new_full -> aten.as_strided.default
longformer                 argform torch.div() -> aten.as_strided.default
longt5                     logical_and -> torch.einsum              (missing_aten_op -> missing_shim_name)
parakeet_rnnt, parakeet_tdt torch._C._nn.glu -> lstm                (glu closed; rnn kernel is next)
rwkv                       torch._C._linalg.linalg_qr -> diag       (still construction -- see note)
sam3_lite_text_text_model  argform torch.embedding(Parameter,...) -> a DIFFERENT argform of torch.embedding
vilt                       torch._C._nn.upsample_nearest2d -> _unique2
```

Two of these are worth reading closely because CLAUDE.md flagged them by name:

* **`rwkv`** moved through a wall again, as it has every round docs/PRIMS.md §6 and
  docs/TAIL1.md's head-note track: `ndimension` -> `new_empty` -> `torch.maximum` (pre-ARCH100),
  `linalg_qr` (ARCH100), now `diag`. Still stopped at **construction** (`_init_weights`'s
  `nn.init.orthogonal_`), not forward — the split in §4 records this explicitly rather than
  letting a construction wall masquerade as a forward one.
* **`sam3_lite_text_text_model`** moved from one `unsupported_arg_form` of `torch.embedding` to a
  different one of the same call — an argument-form fix landed and immediately exposed the next
  argument-form gap on the same op. This is the argform-class version of the same pattern
  `__setitem__` showed at the operator level: closing a wall reveals what is behind it, at every
  granularity this sweep can see.

### 3.3 Still blocked, same wall — 10

No fix landed here between rounds; these are still standing exactly where ARCH100 found them:

```text
fnet, granite_swa, nemotron3_5_asr, nemotron_asr_streaming, nemotron_asr_streaming_encoder,
nystromformer, sam_hq_vision_model, sam_vision_model, t5gemma2, univnet
```

`55 + 17 + 10 = 82` — ARCH100's exact blocked count, split three ways. No architecture appeared in
the upstream-clean set that was not already accounted for in ARCH100's 82 (§3's "newly blocked"
check against `arch_sweep.py`'s own output returned zero).

## 4. Construction versus forward

```text
blocked at CONSTRUCTION     1   (was 6)
blocked in the FORWARD     26   (was 76)
```

The one remaining construction failure is `rwkv`, discussed in §3.2 — still inside
`nn.init.orthogonal_`, now naming `diag` rather than `linalg_qr`. Every other construction wall
ARCH100 recorded (`fastspeech2_conformer`, `seamless_m4t`, `wav2vec2-conformer` on
`__setitem__`; `llama4` on `view_as_complex`; `gpt_neo` on `torch.ones(dtype=bool)`) is gone —
four moved to forward-only architectures (they now build and either forward or stop later in the
graph) and `gpt_neo` forwards outright.

## 5. What did not change, and should not have

* **The denominator (297) and exclusion count (231) are identical to ARCH100.** Upstream was not
  touched between rounds — it cannot report a different set of buildable/forwardable
  architectures unless `transformers` itself changed underneath the venv, which it did not
  (same `spike-venv`). This is the control this round re-ran, not assumed: both sides were
  re-swept in full (§7 has the exact commands), not just the shim side.
* **Golden: 10039/10039 cases, ops=270 — exactly unmoved.** No Rust, `bootstrap.py`, or `aten.rs`
  was touched in this worktree; `git status --short` before writing this document showed changes
  confined to `rust/torch_c/pytests/arch_sweep.py`'s own artifacts (none — the script needed no
  edits), `docs/ARCH100.md` (pointer only), `docs/ARCH200.md` (new), and `README.md`.
* **Suite gate: 668 ok, `DOCWATCH: PASS` 616/616, `EXIT=0`**, measured before the sweep ran, on
  the freshly built shim, from `rust/torch_c/pytests/run.sh` with `PYTHON=$PY` set (its default
  `python3` lacks numpy in this environment, per CLAUDE.md).

## 6. How this round was taken, and how to take it again

Identical method to ARCH100 §7 — the script needed no changes, so the commands are unchanged:

```text
PY=/Volumes/macMini/caches/spike-venv/bin/python
cd rust/torch_c/pytests

PYTHONPATH=$REPO/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY arch_sweep.py --out /tmp/shim.json
env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL          $PY arch_sweep.py --out /tmp/upstream.json
$PY arch_sweep.py --compare /tmp/shim.json /tmp/upstream.json
```

Both sides were re-run in full (528 architectures each, one subprocess per architecture, per
`arch_sweep.py`'s own design) rather than reusing ARCH100's upstream JSON — CLAUDE.md is explicit
that the control is not optional, and reusing a nine-month-old upstream run would have measured
this round's shim against a `transformers` version that no longer matches the venv. The re-run
upstream count landing exactly on 297 is the evidence that reuse was unnecessary in outcome, not
a reason to have skipped re-running it.

`HF_HOME=/tmp/hf-sweep2` was set for the run and removed afterward — no checkpoints were
downloaded (this sweep uses random weights per ARCH100 §5.1; `HF_HOME` only matters for
`--verify-random-weights`, which was not re-run this round since ARCH100 §5.1's claim is about the
harness, not about any one round's operator set, and nothing in this round's territory could have
invalidated it).

## 7. Gates

```text
rust/torch_c/pytests/run.sh    668 ok, exit 0
DOCWATCH                       PASS -- 616/616 evaluated marker(s) hold
tools/golden/compare.py        10039/10039 cases passed, 0 failed, ops covered=270, pending=0
```

All three measured on the freshly built `lib_C.dylib` in this worktree
(`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-sweep2`,
`TORCH_C_ARTEFACT` set explicitly per ARCH100 §8's trap — an unset `TORCH_C_ARTEFACT` falls back
to the shared cache binary and reports a plausible but wrong number).

Golden is exactly unmoved from this worktree's starting point (10039/10039, ops=270), which is the
correct result: this round changed no Rust. The 16 names in §2 are data for the next batch, the
same way ARCH100's 31 were — this round measured and implemented nothing.

<!-- DOCWATCH: op-implemented aten.scatter.value -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/arch_sweep.py classify present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/arch_sweep.py verify_random_weights present -->
<!-- DOCWATCH: count smoke_ok ge 480 -->
<!-- DOCWATCH: count golden_cases_passed ge 10039 -->
<!-- DOCWATCH: count golden_ops_covered ge 270 -->
<!-- DOCWATCH: count golden_pending eq 0 -->
