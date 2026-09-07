# 0.0.13a0 — release notes

`0.0.12a0` has been the published release while a great deal landed behind it.
This is what changed, split four ways rather than summed into one number:
**features added**, **defects fixed**, **measured but not implemented**, and
**documentation corrected**. CLAUDE.md §5.3 asks for that split because the
four are not the same kind of thing and a single figure that mixes them reads
as progress whichever of the four it was.

Read §5 first if you are deciding whether to upgrade: what this release does
*not* do is the part a version number cannot tell you.

---

## 0. The claim this release can make that no previous one could

Every coverage number this project has published — `docs/ARCH100.md`'s 215, `ARCH200.md`'s
270, `ARCH300.md`'s 290 — measured **reachability**. Each of those documents ends on the same
sentence and none of them acted on it: *a forward is not a match.* "It imports and runs" and
"it computes upstream's numbers" are different claims, and only the second supports the word
**drop-in**, which is what this package's own description calls it.

`docs/AGREE.md` is the first measurement of the second claim, and it is this release's
strongest result:

```text
of the architectures that forward, after excluding what cannot be judged:
  284 of 285 judgeable architectures AGREE with upstream numerically   (99.6%)
  nine of them are CLOSER to the float64 answer than upstream is
  worst single-operator error across 775 replayed leaf modules:  8.4 ulp
```

**The tolerance is derived, not chosen.** Every architecture was additionally run upstream in
`float64` and both float32 answers scored against it; the threshold is the **p90 of upstream's
own float32-vs-float64 error distribution** over the 263 architectures that have a working
oracle, floored at 8 ulp — 1.186e-06. Its defence is that anything tighter fails *upstream*
against float64 on a tenth of the same set, and a threshold that would have to call upstream
wrong is not measuring the shim. Above it, a per-architecture rule rather than a bigger
constant: a difference is not a defect if it is within 4× upstream's own distance from the
float64 truth for that same output. That factor is pinned by
`test_agree.py::test_the_oracle_factor_is_stated_and_is_not_a_free_parameter`, so widening it
to make a number look better is a failing test.

**And the result is negative about operators, which is the part worth stating plainly.**
Replaying every leaf module in isolation on upstream's own recorded input — so that nothing
accumulates — the worst single kernel invocation anywhere in 775 leaf modules is 8.4 ulp
(a `Conv2d` in `hgnet_v2`); the median top row is 2 ulp. The large end-to-end differences are
float32 accumulation over depth. §3 of `AGREE.md` is a ranking of depth, not of defects.

### 0.1 The caveats, which are part of the claim and not footnotes to it

* **5 architectures could not be judged, and are excluded from the denominator rather than
  counted as passes.** Three underflowed (`efficientnet` at 8.0e-29, `sam_vision_model` and
  `sam_hq_vision_model` at 8.0e-21) — both sides agreeing on noise is not agreement. Two are
  cases where **upstream does not reproduce itself**: `vit_mae` re-draws its patch mask and
  `vits` samples a duration, so there is no fixed answer to match and any number about them
  measures a sampler. 288 of 290 were bit-identical to themselves across two upstream runs.
* **22 MoE models have no float64 oracle at all**, so only the fixed tolerance applies to them
  and the ratio rule cannot. The reason is upstream's own and not ours: `qwen3_moe`,
  `qwen2_moe`, `glm4_moe`, `nemotron_h`, `zaya` and the rest **refuse `Double` at their grouped
  matmul**. All 22 land in `agree` and none sits near the tolerance, but they are held to a
  weaker test than the other 263 and that has to be said.
* **`chinese_clip` is left flagged**, as the one `DIVERGE`, even though the round that measured
  it believed the flag spurious — its absolute difference is 1.4e-06 on a tensor of scale 0.49
  (twelve ulp), no operator in the whole model exceeds 2.5 ulp, and it crossed the 4× rule only
  because upstream's own oracle error on that output is unusually *small*. It stays flagged
  because **a rule that is edited whenever it fires is not a rule**, and tuning one until a flag
  disappears is not a result.

## 1. Features added

| | |
|---|---|
| **`loss.backward()` and a real training loop** | The eager autograd engine reaches upstream's own path — `torch/_tensor.py` → `_engine_run_backward` → `_ImperativeEngine.run_backward` — so `loss.backward()`, `optimizer.step()` and `zero_grad()` are the ordinary PyTorch code and not a shim-specific call. A six-step SGD loop over an `nn.Sequential`, driven by the real `torch.optim.SGD`, matches upstream to **2.98e-08** — one float32 ulp — across the loss trajectory, the gradients and the final parameters. `retain_graph=True`, `backward(inputs=...)` and `torch.autograd.grad`'s `allow_unused` semantics land with it (`docs/BACKWARD9.md`) |
| **Metal computes on the real GPU** | `mps` is candle's Metal backend, on. An `mps` tensor is an ordinary candle tensor, so no kernel had to be taught it (`docs/VULKAN3.md`) |
| **Vulkan computes on the real GPU** | A fourth arm of `tensor::Repr`, outside candle entirely, with a real `VkBuffer` round-trip on this host. Four ops by name; everything else refuses naming itself, which is what makes a silent CPU fallback structurally unrepresentable rather than merely avoided (`docs/VULKAN3.md`) |
| **A CoreML model that executes** | A captured graph serialises to CoreML MIL, macOS compiles the `.mlpackage`, and `MLModel.predict` runs it — agreeing with the replayed trace to **2–3e-08** at float32. Float32 had to be forced: `coremltools` defaults `mlprogram` to float16, which is four orders of magnitude looser (`docs/NPU.md`) |
| **NNAPI lowering, end to end for one model** | Prims folded back to aten and BatchNorm fused into the preceding convolution, so `mobilenet_v2` lowers with nothing left outside NNAPI's op set (`docs/NPU.md`) |
| **`torch.distributed` at `world_size >= 3`** | `ProcessGroupLocal` over real loopback TCP in a star, hub at rank 0, folding contributions in ascending rank order so the answer is a property of one process and not of who arrived when. Proper-subset cohorts, a survivor set after a dropout, and `on_missing='average_arrived'` with `min_participants=k` all run (`docs/FEDERATED4.md`) |
| **Federated aggregation strategies** | FedAvgM, FedProx and rank dropout, each verified against a central oracle |
| **The eager recorder** | Always-on tape recording, reusing the 60 existing derivative rules verbatim rather than growing a second implementation, with a bounded tape (`EAGER_MAX_NODES = 100,000`) that refuses by name and releases what it held (`docs/BACKWARD7.md`, `docs/BACKWARD8.md`) |
| **`rwkv` forwards** | Its wall was `torch.maximum` |
| **All thirteen missing `prims.*` ops** | Missing prims 13 → 0 |
| **New operators and spellings** | `nonzero` — the first op here whose output *shape* depends on the values — plus `ndimension`, `upsample_bicubic2d`, `torch.fmod`, `torch.maximum`, `Tensor.shape` returning a real `torch.Size`, and the `rsub.Scalar`/`pow.Scalar` autograd-key spellings. **ATen operators 203 → 255**, golden cases 8,509 → 9,691 |
| **float8 (E4M3) computes what upstream computes** | Across the 23 op rows that previously hung or refused, without forking candle |
| **A verifiable WASM/Pyodide wheel** | `PyEmscriptenTarget` and a checker for it; the ABI trap is closed structurally rather than by convention |
| **`from_pretrained(dtype=torch.int8)`** | Beside a `TorchnativeConfig`, widens to float32 and *discloses* rather than raising |
| **Landed after this note was first drafted** | `torch.fft.fftn` and **complex tensors** (`complex64` arithmetic, `.real`/`.imag`); `as_strided` as a read-only view and `TensorBase.unfold` beside it; `lstm`; per-axis convolution padding; `torch.multinomial(Tensor, Tensor)`'s argument form. Each is checked against upstream on this host by `tools/ci/verify_published.py`'s new `signal_and_complex` section, where all four agree **exactly** — `fftn` of an impulse, `[1,2,3,4]`'s transform real and imaginary, `(2i)^2 = -4`, a non-descending `as_strided`, and a two-step `LSTM` — so those are identities on a new platform rather than tolerances (`docs/FFT.md`, `docs/STRIDED.md`, `docs/RNN.md`, `docs/COMPLEX.md`, `docs/LAST7.md`, `docs/BIND5.md`) |

Counted at the tip of this branch on the day of the build rather than on the
day the note was drafted, which is why they are higher than the paragraph they
replace (255 operators, 9,691 cases, 587 tests): **300 ATen operators**,
**11,385 / 11,385** golden comparison cases with **none pending**, **929** tests
passing across 30 files (480 of them in `test_shim.py`, which is what `smoke_ok`
counts) and **830 / 830** DOCWATCH markers holding, at `EXIT=0`.

**Three other agents were landing operator work while this was written, so every
number here is a snapshot rather than a ceiling** — which is why every marker
below is `ge`. A later round that raises one of these is progress, and `eq`
would turn it into a red suite
(`test_release.py::test_count_markers_use_ge_wherever_another_round_could_raise_them`).

<!-- DOCWATCH: count golden_ops_covered ge 300 -->
<!-- DOCWATCH: count golden_cases_total ge 11385 -->
<!-- DOCWATCH: count golden_cases_passed ge 11385 -->
<!-- DOCWATCH: count golden_pending eq 0 -->
<!-- DOCWATCH: count smoke_ok ge 480 -->

## 2. Defects fixed

These are the most useful part of this note if you are already on `0.0.12a0`.
Several were found while implementing something else, which is to say they were
live in the published wheel and silent.

| | |
|---|---|
| **A gradient taken after the weights moved answered at the new weights** | The tape held caller parameters by reference and read them at `backward()` time, so a `backward()` after `optimizer.step()` returned wrong gradients **with no error**. Fixed with per-storage version stamping, matching upstream's "expected version" refusal. Found while measuring, not while testing |
| **A tape returning one tensor for two operands** | `.grad` accumulation shared one object across two leaves of a single `add`, and `sum()` produced a non-writable expanded-stride gradient. Both silently wrong, both now match upstream |
| **Training-mode BatchNorm invalidated its own tape** | The kernel recorded, then noted its own already-completed mutation, so the guard refused a legitimate write. Fixed with `forgive_own_write` — and the missing `native_batch_norm` backward rule was added at the same time; the tape had none |
| **`fmod` read back to the host on `mps`** | Its kernel moved device bytes to the host, and the newly-landed `mps` readback gate did not know the op existed. Caught at a merge between two branches that were each green alone |
| **54 `mps` ops were silently computing on the CPU** | Under an `mps:0` label. An earlier round recorded two; enumerating found 54, `aten._softmax.default` among them. They refuse by name now |
| **`torch.compile` fell back to eager silently** | Empty frame counter, no error. It refuses loudly by name, before install |
| **Seven arithmetic and table defects** | Missing `methods.json` rows for `Tensor.floor_divide` and `Tensor.histc`; a `uint8` scalar-wrapping conversion bug (`uint8 // -3`, `uint8 < -3`); a missing `uint8 ** 300` overflow refusal; `x ** 0.3` off by one ulp at float32, from narrowing the exponent before the `pow` rather than after |
| **Three spellings raised where the function form worked** | Found by calling 52 of 55 spellings nothing had ever called |
| **`__getitem__` mishandled `None` indices** | Hit by whisper's decode loop |
| **A "was this freed?" check lied on a reused address** | It asked an address; it asks the tensor's own `grad_fn` now |
| **The golden harness graded the wrong artefact** | With `TORCH_C_ARTEFACT` unset it silently fell back to a shared cache path — so a suite could grade another checkout's build. Separately, a case pair sharing tensor variable names closed over them, so `both_error` cases ran the *next* case's tensors |
| **CI staged nothing and said nothing** | `stage_dependencies` skipped every wheel requirement on a runner because `SPIKE_SITE` defaulted to this project's own machine, surfacing as a bare `ModuleNotFoundError`. Missing entries fail loudly now |

## 3. Measured but not implemented

Rounds that produced a number and deliberately closed nothing. They are listed
because the number is the deliverable, and because a reader counting commits
would otherwise count these as features.

- **`docs/ARCH100.md`** — every `transformers` architecture swept: **215 of 297
  forward**, the remaining **82** blocked behind **31 distinct operator names**.
  Nothing was implemented in that round.
- **`docs/ARCH200.md`** — the same sweep taken again: **270 of 297 forward**, 27
  blocked behind 16 operator names. A re-measurement, not an implementation.
- **`docs/ARCH300.md`** — taken a third time, after two batches of operator
  work landed: **290 of 297 forward**, the remaining **7** behind 6 walls,
  three of them sharing one argument-form gap. **These are a snapshot, not a
  ceiling** — three other agents were landing operator work in parallel
  worktrees while this round was measured, and the upstream control was
  reused from a same-day run rather than re-swept in full, checked against a
  venv-mtime and a five-architecture spot re-run rather than assumed.
- **`docs/COMPILE.md`** — re-diagnoses what blocks `torch.compile` (abi3 against
  PEP 523 frame evaluation, not Dynamo generally) and **recommends refusing it
  by name permanently**, spending the effort on `torch.export` instead.
- **int8 in candle** — a patch priced at 135 of 203 ops unlocked, and explicitly
  not landed: it forks a pinned dependency.
- **Three `torch.save` paths** — legacy container, `skip_data`,
  `write_record(compress=True)` — left unimplemented with evidence that no
  caller needs them.
- **The NNAPI blocker sized** at thirteen prims ops rather than sixty
  decompositions (implemented in a later round, listed in §1).
- **Vulkan priced before it was wired**, and the demand model set widened,
  recording walls (`torch.floor`, `upsample_bicubic2d`, `index_add_`,
  `ndimension`) without closing them.

- **`docs/AGREE.md`** — the reachability sweep's own closing sentence, acted on for the first
  time: of the architectures that forward, **284 of 285 judgeable ones agree with upstream
  numerically**, at a tolerance derived from upstream's own float32-vs-float64 error
  distribution rather than chosen. §0 above is the summary and §0.1 the caveats. It changed
  **no Rust, no `bootstrap.py` and no `aten.rs`** — golden stayed exactly unmoved, which is the
  correct result for a round that changed no kernel — and it is listed here because *the number
  is the deliverable*. It also produced a **negative** result about operators, and the technique
  that would have found a positive one is the same technique, so its silence is informative.

<!-- DOCWATCH: symbol-in-file docs/ARCH300.md 290 present -->
<!-- DOCWATCH: symbol-in-file docs/ARCH300.md 7 present -->
<!-- DOCWATCH: symbol-in-file docs/AGREE.md 284 present -->
<!-- DOCWATCH: symbol-in-file docs/AGREE.md 285 present -->

## 4. Documentation corrected

- 17 DOCWATCH markers added, and two already-false numeric claims corrected
  (schema entries 4,479 → 4,641; a stale "the rest of the table stands").
- `docs/VULKAN.md` said this machine had no Vulkan loader. It has one; the
  loader was not *found*, which is a different sentence.
- The claim that no WASM wheel existed — one did.
- The provenance of the five target CPython builds CI uses, recorded.
- The suite was split from one `test_shim.py` into `test_*.py`, because
  reconstructing one conflict hunk had twice silently dropped tests.
- **This release's README.** The Roadmap table said `torch.distributed` was
  coming "from `world_size = 1` upward", that NPU "needs a capture layer", and
  that Metal was "disabled here" — after all three had landed. A roadmap is a
  progress record, not a specification (CLAUDE.md §5.1), and a stale one
  misleads in the direction a reader cannot check.

---

## 5. What this release does not do

- **3 of 297 architectures do not forward.** `docs/ARCH300.md` recorded 7 of 297; at release
  time each of those seven was re-run individually against this build and **four of them now
  forward** (`univnet`, `nystromformer`, `vilt`, `sam3_lite_text_text_model`), taking coverage
  to **294 of 297** — from 82 blocked in `docs/ARCH100.md` and 27 in `docs/ARCH200.md`.
  **What that number does and does not rest on:** ARCH300's 290 plus four individual runs. The
  other 290 were **not** re-swept here, so this is a spot re-measurement and not a fresh full
  sweep, and `ARCH300.md`'s own §2 gap list is left standing verbatim rather than edited to
  match — the round that measured it is the round that owns it.
  The denominator is not 528: of the 528 model types `AutoModel` can build, 231 fail on
  *upstream* torch under the same shrunk-config sweep and are excluded as not this project's
  gap. *294 of 528* would be a different and wrong claim.
  The three still blocked are `fastspeech2_conformer` (`torch.repeat_interleave` with a tensor
  `repeats`) and `led`/`longformer` — and those two have **moved wall**, from
  `Tensor.new_zeros` with a tuple size to `TensorBase.where`. That is ARCH300 §1's own
  *first-wall* caveat firing for the third time in three rounds, not a new problem, and it is
  the reason 294 should not be read as "three operators from 297".
- **`torch.compile` is not coming.** `docs/COMPILE.md` recommends refusing it
  by name, permanently. Nothing here has ever implemented any part of it.
  `torch.export` is the direction and it is not implemented either.
- **A transformer now does train through `loss.backward()`** — this bullet said the opposite
  when it was drafted, and it was checked rather than restated. A BERT encoder built from a
  shrunk config runs three `zero_grad` / `backward` / `step` iterations under the real
  `torch.optim.SGD` and its loss trajectory matches upstream's to float32 (5.12 → 0.0 → −5.12
  on both sides), with a gradient on 21 of its 23 parameters — the two without one being the
  unused pooler, which upstream also leaves ungradiented. **Two qualifications that belong to
  that sentence:** it was measured with dropout disabled, because with dropout on the two sides
  diverge after the first step where the RNG streams differ (a sampler difference, not a
  gradient defect); and it is one encoder at a shrunk config, not a training run anyone would
  call converged. **Convolution backward landed after this bullet was
  drafted** — the sentence here said it was missing, which was true when written and was
  re-checked rather than assumed; it is now false. A small CNN with strided, depthwise and
  pointwise convolutions, three batch-norms in training mode and a linear head trains
  end to end through five SGD steps, agreeing with upstream to 2.98e-08
  (`docs/TRAIN2.md`). `create_graph=True`, double backward,
  multiple root tensors, `GradientEdge` inputs, `torch.autograd.Function`,
  hooks and `retain_grad` on non-leaves refuse by name. Mutation through a view
  is refused rather than differentiated — deliberately less than upstream.
- **A transformer does forward on `mps`, and both halves of this bullet were
  wrong in opposite directions** — it said the opposite when it was drafted and
  was re-measured rather than restated (`docs/MPSATTN.md`). "No transformer
  forwards on `mps`" was **already false when it was written**: `docs/MPSFWD.md`
  ran SmolLM2-135M there and this bullet was not updated. "Every attention block
  passes through `aten._softmax.default`" was **true for an eager attention
  block and false for the SDPA path** — and `docs/MPSFWD.md`, correcting the
  first half, generalised its own one-model observation into "no attention block
  reaches it", which is a different sentence and was not measured. A BERT built
  with `attn_implementation="eager"` reaches it twice a layer, which is exactly
  where it stopped. `_softmax` and `_safe_softmax` have been rewritten out of a
  host readback and onto the device (`MPS_HOST_READBACK_OPS` 87 → 85), and three
  further walls that are **not** refusals were fixed: candle's Metal spelling of
  a matmul striding refusal was not recognised by the retry that exists for it,
  and two constants were being materialised by a device that has no `f64`. A
  shrunk BERT encoder now forwards on `mps` and lands **5.257e-07** from
  upstream's own `float64` answer against upstream's own `float32` error of
  **4.320e-07** — 1.22×, inside `docs/AGREE.md`'s derived 4× rule — and differs
  from the shim's own `cpu` answer by at most **0.72 float32 ulp**.
  **Three qualifications that belong to that sentence:** it is one model, one
  input, one dtype, `no_grad`, at a shrunk config, which is the same narrowness
  that produced the error this bullet is correcting; `token_type_ids` must be
  supplied, because `BertEmbeddings` otherwise builds them with `torch.gather`,
  which is still refused on `mps` — that op is in the *embeddings*, not the
  attention block; and **GPT-2 still does not forward**, stopping on Metal's
  inability to allocate a zero-byte buffer, the same wall `docs/MPSFWD.md` §6
  already recorded and left alone.
- **Vulkan is four ops.** Correctness is testable on this host; performance
  needs a phone and has not been measured.
- **One hardware accelerator has been reached; the graphs credited above were not on it.**
  Both halves now execute (`docs/NPU2.md`). The CoreML models this
  document credited above ran on the **CPU**: `MLComputePlan` reports the
  Neural Engine as not even *supported* for a float32 program, so the
  `float32=True` that makes the 2–3e-08 claim meaningful is the same flag that
  puts the NPU out of reach. A wider graph at float16 does execute on the
  Neural Engine, agreeing with replay at 2.0e-04. The NNAPI blob executes on an
  Android emulator through `ANeuralNetworksModel` — the whole 1,156-byte model,
  8/8 operations, ~3e-08 against replay — but every driver there is software
  (`nnapi-reference` and the image's sample drivers). A vendor NPU driver needs
  a physical device.
- **`world_size >= 3` runs eleven collectives over loopback on one machine —
  and the sentence that used to stand here was wrong in both directions.** It
  said `allreduce(op=SUM)` only, with everything else refusing by name. When it
  was re-measured (`docs/COLLECT2.md` §1), `all_gather`,
  `all_gather_into_tensor` and `barrier` **already worked** and had done since
  `FEDERATED4`; and `reduce_scatter`, `scatter`, `all_to_all` and
  `all_to_all_single` **did not refuse at all** — they returned the calling
  rank's own input, unreduced and untransposed, with no error, because their
  bodies were the `world_size = 1` identity and nothing checked the size. A
  promised refusal is worse than a missing one when the truth is a wrong
  answer, because the reader has been told there is nothing there to check.
  All four are fixed and all eleven are now compared element-wise against
  upstream's own gloo backend at world 3 and world 4, at a tolerance derived
  from upstream's float32-vs-float64 error the way `docs/AGREE.md` §2 derives
  its own — which for the collectives that move numbers rather than combining
  them comes out at zero, so those are held to bit equality.
  `allreduce` now folds `MIN`, `MAX`, `PRODUCT` and `AVG` beside `SUM`.
  **Still refusing by name**: genuine `async_op` (these collectives complete
  inside the call, so the returned `Work` is already done where upstream's is
  not — `docs/COLLECT2.md` §7), uneven-split `all_to_all`, the bitwise reduce
  ops, the `_coalesced` spellings, `send`/`recv`, `new_group`, secure
  aggregation and differential privacy.

## 6. Platform status for this release

Do not read a platform as verified unless it is listed here. And a wheel
that *built* is not a platform that works: all seven wheels for this
release build and pass `tools/wheel/verify_cross.py`, which is a claim about
tags, binaries and symbol resolution. `computes` is a separate claim and is
made below only where something ran.

| | |
|---|---|
| macOS arm64 | the machine everything above was measured on. The `macosx_11_0_arm64` wheel installs into a clean venv and its torch computes (`tools/wheel/verify.py` PASS) |
| Linux x86_64 · Windows amd64 | verified by CI installing the **published** wheel and computing — but the green runs installed the version the workflow defaults to, which is `0.0.12a0`. `tools/ci/verify_published.py` carries a `loss.backward()` training step, three operator checks, and now a `signal_and_complex` section for `torch.fft.fftn`, complex tensors, `as_strided` and `lstm` — each skipping **by name** on an older wheel. **None of those have run green on Linux or Windows**, because the wheel they check is this one and it is not uploaded; they skip themselves, and a skip is not a platform result. The `manylinux_2_17_x86_64` and `win_amd64` wheels for this release build and pass `verify_cross.py` — glibc floor 2.17 read off the artefact's own `.gnu.version_r`, `DT_NEEDED` inside the PEP 599 policy list, 123 `python3.dll` imports on the Windows side — which is a **symbol-level** claim, as `verify_cross.py` says of itself, and not a run |
| iOS simulator arm64 | **computes on this machine, and CI is now green.** An earlier `0.0.13a0` simulator wheel was unpacked into an iOS CPython inside a booted simulator here and its torch computed (`verify_ios_sim.py` PASS, 1,282 `_C` names, 896 aten ops); that has **not** been re-run against the wheel rebuilt for §7 below. The CI leg was the separate question and it is answered: the `setuptools<81` pin landed, was pushed, and **run 34038982934 is green on all three legs** — `linux-x86_64` 43s, `windows-amd64` 1m38s, `ios-simulator-arm64` 5m5s. That run installs the **published `0.0.12a0`** wheel, which is what the workflow's default says and what it should say until an upload happens; it is a green result for the iOS staging harness and for the checks that predate 0.0.13a0, and not for the ones that skip themselves by name |
| iOS device | never executed, on any release. The `ios_12_0_arm64_iphoneos` wheel builds and passes `verify_cross.py`; nothing has imported it |
| Android arm64 | emulator and device runs exist for earlier releases; **not re-run for this one**. The `android_21_arm64_v8a` wheel builds and passes `verify_cross.py` |
| WASM | `build.py --target wasm32-emscripten` now produces one — `PyEmscriptenTarget` landed this batch, and the sentence that `build.py` cannot is no longer true. The `pyemscripten_2026_0_wasm32` wheel builds and passes `verify_cross.py`'s wasm reader: `PyInit__C` exported as a **function**, 133 exports, both binaries wasm32 side modules, and **96 `Py*` imports all resolved against `pyodide.asm.wasm`**. Two things that check does not cover and says so: the non-`Py*` `env` imports, and the abi3 binding, which has no wasm spelling. Nothing has imported *this* wheel under Pyodide — the computing claim still rests on the earlier hand-built one |

---

## 7. What is left for the release itself

This document and the version bump are prepared; **nothing has been published.**

### 7.1 The seven wheels, rebuilt at this head

All seven were built again from this checkout after `cargo build --release` and
`vendor/install_shim.sh`, one target at a time with `rm -rf build` between them —
`build.py` refuses while a `build/` cache from a previous target is present. The Linux leg
needs `/Volumes/macMini/caches/zig-venv/bin` on `PATH` (`cargo zigbuild`), the Windows leg
`/Volumes/macMini/caches/msvc-shims` (`cargo xwin`), and the WASM leg an `EM_CACHE` pointed
**away from** the shared emsdk, which must not be written to.

| platform | tag | size | check | pass |
|---|---|---|---|---|
| macOS arm64 | `cp313-abi3-macosx_11_0_arm64` | 14,644,677 B | `verify.py` — **installs into a clean venv and computes** | ✅ |
| Android arm64 | `cp313-abi3-android_21_arm64_v8a` | 14,740,236 B | `verify_cross.py` — symbol level | ✅ |
| iOS device | `cp313-abi3-ios_12_0_arm64_iphoneos` | 14,714,150 B | `verify_cross.py` — symbol level | ✅ |
| iOS simulator | `cp313-abi3-ios_14_0_arm64_iphonesimulator` | 14,618,730 B | `verify_cross.py` — symbol level | ✅ |
| Linux x86_64 | `cp313-abi3-manylinux_2_17_x86_64` | 14,760,075 B | `verify_cross.py` — symbol level | ✅ |
| Windows amd64 | `cp313-abi3-win_amd64` | 14,792,216 B | `verify_cross.py` — symbol level | ✅ |
| WASM | `cp313-abi3-pyemscripten_2026_0_wasm32` | 13,679,733 B | `verify_cross.py` — symbol level | ✅ |

**Six of those seven rows are not compute claims and must not be read as ones.**
`verify_cross.py` says so itself in its own PASS line: *"NOT established here: that it loads,
imports, or computes."* It establishes that the archive is tagged for a platform an installer
will match and holds binaries for that platform. The only compute claim in the table is the
macOS row, where `verify.py` installs the wheel into a clean venv and runs it (1,283 `_C`
names, 896 aten ops, `aten.mm.default` and `x + x` both correct). The iOS simulator wheel
**was not re-run** through `verify_ios_sim.py` after this rebuild, so §6's simulator compute
claim belongs to the earlier build of that wheel and not to this one.

Disk was at **67% with 116 GB free** afterwards, on the external volume, with three other
agents building concurrently.

### 7.2 In order, from here

1. ~~Build the wheels~~ **done** — §7.1.
2. Upload. The token is not in this worktree and was not read here.
3. **Then** bump two things that must not lead the upload, and which two tests in
   `rust/torch_c/pytests/test_release.py` hold to that rule:
   `.github/workflows/verify-published-wheel.yml`'s default version, and the README platform
   table's **on PyPI `…`** row, both to `0.0.13a0`. **Both still read `0.0.12a0`, and that is
   correct** — CI installs *from PyPI*, so defaulting it to an unpublished version makes every
   push a red run that says nothing about any platform, and a red CI that is expected to be red
   is a check nobody reads. Bumping it early would be the mistake, not the fix.
4. Re-run the workflow. The Linux and Windows legs will then exercise the `loss.backward()`
   training step, the three operator checks, and the new `signal_and_complex` section
   (`torch.fft.fftn`, complex tensors, `as_strided`, `lstm`) for the first time — until step 3
   they skip themselves by name against `0.0.12a0`, which is correct and **is not a green
   result for those checks**.
5. ~~The iOS leg's staging fix has never run on a runner~~ **it has.** `setuptools<81` is in
   the workflow on `develop`, and run **34038982934** is green on all three legs. What remains
   untested there is everything gated behind step 3.
