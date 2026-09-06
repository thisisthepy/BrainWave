# ARCH100 — every `transformers` architecture, swept: 215 of 297 forward, 31 operators missing

> **Re-swept in `docs/ARCH200.md`.** Two batches of operator work landed between this round and
> that one — `__setitem__`, the scatter family, `index_select`, `glu`, `linalg_qr`,
> `linalg_norm`, complex tensors, reflect/replicate padding, rank-5 `matmul`, `eye`, `erfinv`,
> `scatter_reduce`, `index_add`, `view_as`, `bitwise_xor`, and more. The headline below (215/297,
> 31 operators) no longer describes the current tree: ARCH200 measures **270/297 (91%), 16
> operators**. This document is left in place as the record of what was true at `6d016f0` — the
> comparison ARCH200 exists to make depends on this baseline staying unedited.

Worktree `work/arch` on develop `6d016f0`. torch 2.13.0 upstream
(`/Volumes/macMini/caches/spike-venv/bin/python`), `transformers` 5.15.1.
No Rust was changed in this round: golden stays at **8921/8921, ops=222**, exactly unmoved.

The project has verified roughly twelve architectures deeply (docs/DEMAND*.md). `transformers`
ships hundreds. The question that decides whether a non-alpha release is weeks away or a different
project entirely — **how many distinct operators does this shim still lack across the whole
field** — had never been measured. This document is that measurement, and
`rust/torch_c/pytests/arch_sweep.py` is the script that takes it, so it can be taken again in a
month rather than re-derived.

---

## 1. The two numbers

```text
architectures swept                       528   (every model type `AutoModel` can build)
  forward on UPSTREAM torch (baseline)    297   <- the denominator
  fail on upstream too (not our gap)      231   <- excluded, see §3

of the 297 upstream-clean architectures:
  FORWARD UNDER THE SHIM TODAY            215   (72%)
  blocked by the shim                      82

DISTINCT MISSING OPERATORS                 31
```

**215 of 297 forward; the remaining 82 need 31 distinct operators.** The answer to the question
this round exists to settle is therefore "weeks, not a different project" — the tail is 31 names,
not 300, and §2's ranking says the first four of them account for 32 of the 82.

Two qualifications sit on that number and both are in the honest direction:

* **31 is a first-wall count.** Closing the top entry reveals whatever stands behind it. `rwkv`
  is the recorded precedent (docs/DEMAND8.md §2.6, docs/PRIMS.md §6): `ndimension` fell and
  `new_empty` appeared; `new_empty` fell and `torch.maximum` appeared. So 31 is a lower bound on
  the work and an exact count of the *currently visible* walls. It is not an estimate of the
  total, and this document does not offer one.
* **A forward is not a match.** This sweep records whether the model runs, not whether its output
  agrees with upstream. The deep rounds do that, and every model they checked matched to float32
  accumulation (docs/DEMAND8.md §1). "215 forward" is a reachability number.

Beyond the 31, two smaller and cheaper classes were separated out rather than folded in:

```text
argument forms the shim refuses (the OP exists)     6 distinct,  6 architectures
backend refusals (candle layout, not a missing op)  1 distinct,  3 architectures
unclassified -- NOT guessed at                                   2 architectures
```

## 2. The ranked list — the deliverable

Each row is a distinct missing operator and the number of the 297 upstream-clean architectures
whose forward (or construction) stops there **first**.

```text
  13  TensorBase.__setitem__                       missing_shim_name      cum=13
       cosmos3_omni, fastspeech2_conformer, minicpmv4_6, qwen3_5, qwen3_5_moe,
       qwen3_5_moe_text, qwen3_5_text, qwen3_vl, qwen3_vl_moe, qwen3_vl_moe_text,
       qwen3_vl_text, seamless_m4t, wav2vec2-conformer
   7  TensorBase.scatter_                          missing_shim_name      cum=20
       exaone_moe, glm4_moe, glm4_moe_lite, groupvit, mistral4, nemotron_h, solar_open
   7  torch._C._nn.glu                             missing_shim_name      cum=27
       cohere_asr, lasr_ctc, lasr_encoder, parakeet_ctc, parakeet_encoder, parakeet_rnnt, parakeet_tdt
   5  TensorBase.index_select                      missing_shim_name      cum=32
       kosmos-2.5, m2m_100, nllb-moe, seamless_m4t_v2, xglm
   4  aten.scatter.value                           missing_aten_op        cum=36
       axk2, deepseek_v32, glm_moe_dsa, jetmoe
   4  torch._C._functorch._vmap_increment_nesting  missing_shim_name      cum=40
       nemotron3_5_asr, nemotron_asr_streaming, nemotron_asr_streaming_encoder, t5gemma2
   3  argsort                                      missing_aten_op        cum=43   aria, aria_text, vit_mae
   2  aten.where.Scalar                            missing_aten_op        cum=45   cpmant, git
   2  bucketize                                    missing_aten_op        cum=47   idefics3_vision, smolvlm_vision
   2  torch._C._linalg.linalg_norm                 missing_shim_name      cum=49   owlv2, owlvit
   2  torch._C._nn.upsample_linear1d               missing_shim_name      cum=51   sam_hq_vision_model, sam_vision_model
   1  TensorBase._is_all_true                      missing_shim_name      cum=52   vits
   1  TensorBase.masked_scatter                    missing_shim_name      cum=53   higgs_audio_v2
   1  TensorBase.new_full                          missing_shim_name      cum=54   led
   1  TensorBase.reshape_as                        missing_shim_name      cum=55   roformer
   1  TensorBase.unflatten                         missing_shim_name      cum=56   siglip_vision_model
   1  acos                                         missing_aten_op        cum=57   yoso
   1  broadcast_tensors                            missing_aten_op        cum=58   gemma3n_text
   1  chunk                                        missing_aten_op        cum=59   diffllama
   1  diff                                         missing_aten_op        cum=60   voxtral_realtime_encoder
   1  logical_and                                  missing_aten_op        cum=61   longt5
   1  logsumexp                                    missing_aten_op        cum=62   granite_swa
   1  max_pool1d                                   missing_aten_op        cum=63   canine
   1  multiply                                     missing_aten_op        cum=64   convbert
   1  polar                                        missing_aten_op        cum=65   llama4_text
   1  prod                                         missing_aten_op        cum=66   tapas
   1  torch._C._fft.fft_fftn                       missing_shim_name      cum=67   fnet
   1  torch._C._linalg.linalg_qr                   missing_shim_name      cum=68   rwkv
   1  torch._C._nn.pad                             missing_shim_name      cum=69   univnet
   1  torch._C._nn.upsample_nearest2d              missing_shim_name      cum=70   vilt
   1  view_as_complex                              missing_aten_op        cum=71   llama4
```

Read as a roadmap, the shape is unusually favourable:

* **The head is four names for 32 architectures.** `__setitem__`, `scatter_`, `_nn.glu` and
  `index_select` are 32 of the 82 blocked, and none of them is exotic — `glu` is a gated linear
  unit, `index_select` is a gather. `__setitem__` alone is 13, and its 13 are dominated by one
  family (`qwen3_5*` / `qwen3_vl*`, six rows), which is worth knowing before it is ranked as
  thirteen independent wins.
* **`missing_shim_name` outnumbers `missing_aten_op` 49 to 22.** A *spelling* gap is much cheaper
  than a kernel: docs/DEMAND8.md §2.1 landed `ndimension` as one method on `PyTensorBase` with no
  kernel, no `overloads.json` entry and no golden case. Not all 49 are that cheap — `linalg_qr`
  and `fft_fftn` need real numerics — but the counting split says most of the remaining work is
  binding surface rather than arithmetic.
* **The tail is genuinely a tail.** 20 of the 31 block exactly one architecture each.

### 2.1 The two cheaper classes, kept separate on purpose

Folding these into "missing operators" would have inflated the number this round exists to
produce. In each the operator already exists; only an argument form does not.

```text
torch.ones(): no matching overload for (tuple, dtype=type)                gpt_neo
Tensor.mean(): no matching overload for (axis=int, keepdim=bool)          ibert
torch.mean():  no matching overload for (Tensor, axis=int, keepdim=bool)  imagegpt
torch.div():   no matching overload for (int, int, rounding_mode=str)     longformer
torch.embedding(): (Parameter, NoneType, int, bool, bool)                 sam3_lite_text_text_model
aten.convolution.default: an asymmetric padding [0, 5]                    nystromformer
```

`axis=` is the same gap twice — it is numpy's spelling of `dim`, which upstream accepts as an
alias — and `torch.ones(..., dtype=bool)` is a `dtype` the overload resolver does not take from
a Python `type`. Four of these six are keyword-alias work.

The backend row is different in kind and should not be read as a missing op:

```text
aten.matmul.default: candle: MatMulUnexpectedStriding      hiera, olmo_hybrid, qwen3_next
```

## 3. What was excluded, and why the control is not optional

**231 of 528 fail on upstream torch too**, and every one of them is excluded. Without that control
the headline would have read "215 of 528, 41%" and would have been wrong: the 231 are
`transformers` refusing our synthetic configs, not this shim refusing anything.

The exclusions are dominated by this script's own config-shrinking meeting architectures that
will not shrink — `RuntimeError: selected index k out of range`, `IndexError: tuple index out of
range`, `Given normalized_shape=[N], expected ...`, 18 architectures wanting `pip install timm`,
one wanting `scipy`. **These are not operator gaps and folding them in was the specific way this
measurement could have produced a wrong number.** They are also not evidence that those 231
architectures work: they are simply *unmeasured*, and a later round that shrinks configs better
would move them into the denominator, in either direction.

Two of them are worth naming because they look like ours and are not: `detr` and 17 other
vision architectures need `timm`, and `mimi`'s config has a read-only `num_codebooks` property.

## 4. Construction versus forward

```text
blocked at CONSTRUCTION    6
blocked in the FORWARD    76
```

**The headline is a forward-time number.** This was checked rather than assumed, because
docs/PRIMS.md §6 records `rwkv` as a case where the last wall moved *into* `_init_weights` — and
`rwkv` is one of the six here, still stopped at `torch.linalg.qr` inside `nn.init.orthogonal_`.
If the split had come out the other way the headline would have meant something quite different:
"cannot even be built" is a far weaker result than "builds and stops at one operator".

The six construction failures are `fastspeech2_conformer`, `seamless_m4t` and `wav2vec2-conformer`
(`TensorBase.__setitem__`), `llama4` (`view_as_complex`), `rwkv` (`linalg_qr`) and `gpt_neo`
(`torch.ones(dtype=bool)`). Every one is an ordinary operator gap that happens to be reached from
`__init__` rather than `forward`, so none of them changes the ranking in §2.

## 5. Which 528, and how they were chosen

**Every model type in `transformers.models.auto.modeling_auto.MODEL_MAPPING_NAMES`, in
alphabetical order, with no selection at all.** The full list is reproducible with
`arch_sweep.py --list`.

Exhaustive rather than sampled, and alphabetical rather than by popularity, because **the point of
this round is the tail.** A hundred popular models would have been a hundred decoder-only LLMs
sharing one operator set, and it would have answered a much easier question with a much better
number. The operators that are actually missing live in the corners: `_nn.glu` is seven ASR
encoders (`parakeet`, `lasr`, `cohere_asr`), `upsample_linear1d` is SAM's vision tower,
`fft_fftn` is `fnet`, `linalg_norm` is OWL-ViT. Not one of those is in any popularity ranking, and
a popularity-weighted sweep would have reported a smaller and less useful number.

Modality breadth is a consequence of taking all of them: the 297-architecture baseline contains
text encoders and decoders, encoder-decoders (`t5`, `m2m_100`, `led`, `longt5`), vision (`vit`,
`swin`, `dinov3`, `hiera`, `vit_mae`), audio (`wav2vec2`, `whisper`, `parakeet`, `univnet`,
`vits`), multimodal (`clip`, `blip`, `llava`, `owlvit`, `idefics3`, `qwen3_vl`, `kosmos-2.5`) and
the odd ones (`fnet`, `canine`, `tapas`, `rwkv`, `yoso`).

### 5.1 No checkpoints — and the claim that makes that legitimate, measured

Downloading hundreds of checkpoints is not viable here (disk was at 86%), so every model is built
from a config with random weights: `AutoModel.from_config` on a shrunk config. The whole sweep
rests on one claim — **a tiny random-weight instance reaches the same operators as the real
checkpoint** — and the claim is about the graph, which a config decides and weights do not.

That is an argument, not a measurement, so it was measured, on three architectures, tracing the
shim's own `torch._C._aten_dispatch` so that what is compared is exactly the set of keys the shim
must supply (`arch_sweep.py --verify-random-weights`):

```text
google/bert_uncased_L-2_H-128_A-2  (bert)
  1 real weights, real shape       10 aten keys
  2 random weights, real shape     10   equal to (1): True
  3 random weights, tiny config    10   equal to (1): True      symmetric difference: none

WinKawaks/vit-tiny-patch16-224     (vit)
  1 real weights, real shape        8 aten keys
  2 random weights, real shape      8   equal to (1): True
  3 random weights, tiny config     8   equal to (1): True      symmetric difference: none
```

Row (1)→(2) isolates the weights; (2)→(3) isolates the shape. Both are exact, in both modalities.

The third is the more informative direction, because a *success* pair can agree by having too
little to disagree about. The real 200M-parameter `nvidia/groupvit-gcc-yfcc` checkpoint was loaded
under the shim and run:

```text
nvidia/groupvit-gcc-yfcc  ->  wall: ('missing_shim_name', 'TensorBase.scatter_')
tiny random groupvit      ->  wall: ('missing_shim_name', 'TensorBase.scatter_')
```

**The same wall, from a real checkpoint and from a 32-hidden-size random one.** That is the claim
holding on a failure, which is the case the sweep actually depends on.

**What this could not have caught**, since CLAUDE.md §5.4 asks for it: an operator that only a
*trained* weight distribution reaches — a `torch.where` on a threshold that random weights never
cross, or a data-dependent branch. Nothing in these three architectures had one, and three
architectures are not a proof that none does. The failure mode is one-directional, though: it
makes this sweep **under**-report missing operators, never over-report.

### 5.2 Where the harness itself distorts

Stated so the number can be argued with:

* **Configs are shrunk** (2 layers, hidden 32, vocab 99+). Shrinking is what makes 528 models
  viable and it is also what produced most of §3's 231 exclusions. Two guards were needed and are
  in the script: `vocab_size` is floored above the largest special token id the config names
  (otherwise Whisper fails in `nn.Embedding`), and an MoE's expert count is floored at its
  `num_experts_per_tok` (otherwise `topk` raises "selected index k out of range", which was 22
  architectures before the fix).
* **Inputs are minimal and chosen from `main_input_name`.** Multimodal models that refuse a
  single-modality forward get a second attempt carrying every modality their signature accepts.
  The refusal reported is the **first classifiable** one across attempts, not the last — the
  fallback attempt fails for input reasons, and an earlier run of this sweep buried real refusals
  under `'NoneType' object has no attribute 'device'` because it kept the last.
* **Only `AutoModel`** — the base body. Task heads (`...ForCausalLM`, `...ForObjectDetection`) add
  operators this sweep never reaches, and generation adds more still. **This is a lower bound on
  the gap for that reason too.**
* **Every architecture runs in its own subprocess.** A shim gap can be a segfault as easily as an
  exception, and one crash must not take the sweep with it. (None crashed: all 528 returned JSON.)

## 6. Unclassified — 2, and not guessed at

```text
dinov3_convnext   forward   RuntimeError: adaptive_avg_pool2d: output_size must be 2
efficientnet      forward   RuntimeError: adaptive_avg_pool2d: output_size must be 2
```

The refusal names an operator that **is** implemented (`aten.adaptive_avg_pool2d.default` is in
`_aten_implemented()`), and the message matches no rule the classifier has, so it is reported as
unclassified rather than filed under a kind. It is most likely a single-element `output_size`
argument form, which would put it in §2.1 rather than in the 31 — but "most likely" is exactly
what CLAUDE.md §5.4 says not to write down, so it is counted here instead, and the ranking in §2
does not include it either way.

An earlier run of this sweep had **17** unclassified. Fifteen of them became classifiable once the
first-classifiable-attempt rule (§5.2) and the `no matching overload` / `candle:` rules were added.
The bucket is reported rather than hidden precisely because it moved so much: a wrong
classification here becomes a wrong roadmap.

## 7. How to take the number again

```text
PY=/Volumes/macMini/caches/spike-venv/bin/python
cd rust/torch_c/pytests

PYTHONPATH=$REPO/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY arch_sweep.py --out /tmp/shim.json
env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL          $PY arch_sweep.py --out /tmp/upstream.json
$PY arch_sweep.py --compare /tmp/shim.json /tmp/upstream.json
```

It is **not** wired into `rust/torch_c/pytests/run.sh` and must not be: it constructs 528 models,
takes minutes per side, needs the network for `--verify-random-weights`, and its result is a
measurement rather than an invariant. A measurement that becomes a gate becomes a number people
edit.

What *is* in the suite is one test —
`test_the_arch_sweep_classifier_maps_real_refusals_to_the_operator_they_name` — because §2's
ranking is produced by parsing refusal text and is only as good as that parser. It checks nine
refusal strings **transcribed from this round's real runs**, including the two the classifier must
*not* get wrong: `no matching overload` is not a missing operator, and `No module named 'timm'` is
not our gap at all. It also asserts that a message with no rule stays `unclassified` rather than
being guessed at.

## 8. Gates

```text
rust/torch_c/pytests/run.sh    457 ok, exit 0     (456 -> 457: +1 = the classifier test)
DOCWATCH                       PASS -- 442/442 evaluated marker(s) hold
tools/golden/compare.py        8921/8921 cases passed, 0 failed,
                               ops covered=222, pending case builders=0
```

Golden is **exactly unmoved**, which is the correct result: this round changed no Rust, no
`bootstrap.py`, no `tools/`. It measured, and it implemented nothing — the 31 names in §2 are
data for the rounds that follow, and deliberately not closed here.

One trap worth recording for the next person, since it produced a false number for ten minutes:
running `compare.py` without `TORCH_C_ARTEFACT` set makes `tools/golden/loader.py` fall back to
the **shared** `/Volumes/macMini/caches/cargo-target/release/lib_C.dylib`, and it reported
`8509/8509, ops=203` — a plausible-looking green summary measured against another checkout's
binary. It warns on stderr when it does this. Set the variable.

<!-- DOCWATCH: op-implemented aten.adaptive_avg_pool2d.default -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/arch_sweep.py classify present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/arch_sweep.py verify_random_weights present -->
<!-- DOCWATCH: count smoke_ok ge 457 -->
<!-- DOCWATCH: count golden_cases_passed ge 8921 -->
