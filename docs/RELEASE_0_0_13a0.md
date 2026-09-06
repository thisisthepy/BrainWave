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
| **New operators and spellings** | `nonzero` — the first op here whose output *shape* depends on the values — plus `ndimension`, `upsample_bicubic2d`, `torch.fmod`, `torch.maximum`, `Tensor.shape` returning a real `torch.Size`, and the `rsub.Scalar`/`pow.Scalar` autograd-key spellings. **ATen operators 203 → 224**, golden cases 8,509 → 9,137 |
| **float8 (E4M3) computes what upstream computes** | Across the 23 op rows that previously hung or refused, without forking candle |
| **A verifiable WASM/Pyodide wheel** | `PyEmscriptenTarget` and a checker for it; the ABI trap is closed structurally rather than by convention |
| **`from_pretrained(dtype=torch.int8)`** | Beside a `TorchnativeConfig`, widens to float32 and *discloses* rather than raising |

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

- **82 of 297 architectures do not forward.** The denominator is not 528: of
  the 528 model types `AutoModel` can build, 231 fail on *upstream* torch under
  the same shrunk-config sweep and are excluded as not this project's gap.
  *215 of 528* would be a different and wrong claim. And a forward is not a
  match — only 26 architectures have been checked for numerical agreement
  against upstream (`docs/ARCH100.md`).
- **`torch.compile` is not coming.** `docs/COMPILE.md` recommends refusing it
  by name, permanently. Nothing here has ever implemented any part of it.
  `torch.export` is the direction and it is not implemented either.
- **No transformer trains through `loss.backward()` yet.** What is verified is
  a small `nn.Sequential` with a real optimizer. There is no convolution
  backward rule, so vision models stop; `create_graph=True`, double backward,
  multiple root tensors, `GradientEdge` inputs, `torch.autograd.Function`,
  hooks and `retain_grad` on non-leaves refuse by name. Mutation through a view
  is refused rather than differentiated — deliberately less than upstream.
- **No transformer forwards on `mps`.** `aten._softmax.default` is in the
  set of ops refused there, and every attention block passes through it.
- **Vulkan is four ops.** Correctness is testable on this host; performance
  needs a phone and has not been measured.
- **Nothing has run on an NPU.** CoreML executes on macOS; the NNAPI blob is
  structurally validated and has never met an NNAPI runtime.
- **`world_size >= 3` is `allreduce(op=SUM)` only**, over loopback on one
  machine. Other collectives, other reduce ops, secure aggregation and
  differential privacy refuse by name.

## 6. Platform status for this release

Do not read a platform as verified unless it is listed here.

| | |
|---|---|
| macOS arm64 | the machine everything above was measured on |
| Linux x86_64 · Windows amd64 | verified by CI installing the **published** wheel and computing — but the green runs installed the version the workflow defaults to. `tools/ci/verify_published.py` gained a `loss.backward()` training step and three operator checks for this release, each skipping by name on an older wheel; **those have not run green on Linux or Windows yet**, because the wheel they check is this one |
| iOS simulator arm64 | **red.** The leg builds an iOS-simulator CPython and reaches the wheel, then fails staging the pure-Python requirements: setuptools 84 dropped `pkg_resources`, which the wheel's METADATA requires. Pinned to `setuptools<81` in this release's workflow, **not yet verified** |
| iOS device | never executed, on any release |
| Android arm64 | emulator and device runs exist for earlier releases; not re-run for this one |
| WASM | a hand-built wheel imports and computes under Pyodide 3.14; `build.py` does not produce one |

---

## 7. What is left for the release itself

This document and the version bump are prepared; **nothing has been published.**
In order:

1. Build the wheels (`tools/wheel/build.py`). Disk was at 88% when this was
   written — check before starting.
2. Upload. The token is not in this worktree and was not read here.
3. **Then** bump two things that must not lead the upload, and which two tests
   in `rust/torch_c/pytests/test_release.py` hold to that rule:
   `.github/workflows/verify-published-wheel.yml`'s default version, and the
   README platform table's **on PyPI `…`** row, both to `0.0.13a0`.
4. Re-run the workflow. The Linux and Windows legs will then exercise the
   `loss.backward()` training step and the three new operator checks for the
   first time — until step 3 they skip themselves by name against `0.0.12a0`,
   which is correct and is not a green result for those checks.
5. The iOS leg is red for a staging reason, fixed but unverified (§6).
