# REMEASURE2 — re-measuring `README.md` on 2026-09-07

The seventh time a claim in this repository has been found stale on re-measurement, and the
second time the `README.md` Roadmap table specifically has been. The table carries its own
warning that it is a progress record; this round is that warning being acted on.

**Method.** Every row of the Roadmap table, and the Status, Devices and Platforms sections, was
checked by **running something** — not by reading the code that implements it, and not by reading
the `docs/` file that the row cites. Negative claims ("X refuses", "X has never happened") were
**provoked**, because those are the claims a reader cannot check: they say there is nothing to try.
`docs/distributed/COLLECT2.md` is why that rule exists — four collectives were documented as
refusing by name while they silently returned each rank's own input.

Every probe asserted `hasattr(torch._C, "_aten_implemented")` first, so that nothing here is a
measurement of upstream torch wearing this project's name.

**No performance numbers were produced.** `docs/perf/PERF.md` is dated 2026-08-25 against 96
operators and the tree is at 302; this machine cannot be made idle. §4 below says what a perf
round would have to measure.

---

## 1. Claim by claim

| Claim, as `README.md` had it | Still true? | How it was checked | What it says now |
|---|---|---|---|
| **Roadmap · Eager training** — "no transformer has been trained through it, and there is no convolution backward rule" | **FALSE — both halves** | Ran both. A multi-head attention block (`LayerNorm` + FFN) took 3 `zero_grad`/`backward`/`step` iterations, loss falling, gradient on **16 of 16** parameters. A CNN with strided, depthwise and pointwise convolutions and three training-mode `BatchNorm`s trained end to end, gradient on **12 of 12** | A transformer trains; convolution backward exists and agrees with upstream to **2.98e-08**. `max_pool2d` backward, transposed-convolution gradient, `ceil_mode`, double-backward, `autograd.Function` and hooks are still refused by name |
| **Roadmap · `torch.distributed`** — "Only `allreduce(op=SUM)` is implemented; every other collective and every other reduce op refuses by name" | **FALSE** | Spawned real multi-process worlds at **world 3 and world 4** over loopback TCP and called every collective. **29 of 39 probes returned a value**; 10 refused | Eleven collectives run and agree with upstream gloo. **Four of the supposed refusals were never refusing** — `reduce_scatter`, `scatter`, `all_to_all`, `all_to_all_single` returned each rank's own input. Still refusing by name: `BAND`/`BOR`/`BXOR`, `PREMUL_SUM`, `send`/`recv` |
| **Roadmap · NPU** — "the NNAPI blob is structurally validated and has never met an NPU" | **HALF FALSE — and the other half is true** | Ran `test_npu2.py` on this Mac (CoreML) **and against a physical Galaxy Tab S9 Ultra, SM8550 / Snapdragon 8 Gen 2** over wireless ADB. 9/9 passed on real silicon | "Structurally validated" is **understated**: the blob executes through `ANeuralNetworksModel` on emulator *and* on the physical device, agreeing with the replay to **1.2e-07**. "Has never met an NPU" is **still true for NNAPI** — see §2 |
| **Roadmap · NPU** — "**CoreML executes**" | **True, and now more so** | `test_a_graph_executes_on_the_neural_engine_and_agrees_with_replay` passes | A float16 graph runs on the **Neural Engine** with every compute op placed there. Separately: the *earlier* CoreML models this row called executed had **run on the CPU**, because the float32 accuracy pin excludes the NE |
| **Roadmap · Metal** — "`aten._softmax.default` [is refused], so a transformer does not forward on `mps`" | **FALSE** | Asked the runtime for the refused set and searched it; then forwarded an encoder layer on `mps` | `_softmax` is **not** in the set — it was rebuilt from candle ops that stay on the device. A transformer **does** forward on `mps` |
| **Roadmap · Device abstraction** — "**Done.**" | **True, but under-specified** | Constructed every device label against the runtime: `cpu`, `meta`, `mps`, `vulkan`, `cuda`, `xpu` construct; `npu` and an invented label are refused against a closed vocabulary | Kept as Done, now scoped: it is about **`torch._C`'s `torch.device`**, not about `torchnative.device`, which is a separate user-facing abstraction on a parallel branch and **is not measured here** |
| **Roadmap · Vulkan** — "Eighteen ops by name… a transformer does not" | **Checked, unchanged** | `test_vulkan4.py` in the gate, with the loader supplied so the tests do not skip by name | Unchanged |
| **Roadmap · CUDA** — "Wired… nothing has been compiled with CUDA on and nothing has run on a GPU" | **Checked, unchanged** | `test_cuda.py` in the gate; this machine has no `nvcc` | Unchanged |
| **Roadmap · `torch.compile`** — "refuse it by name, permanently" | **Checked, unchanged** | Provoked: `torch.compile(f)(x)` raises `NotImplementedError` naming `torch._C._dynamo.eval_frame.set_skip_guard_eval_unsafe`. `torch.export` also refuses | Unchanged |
| **Status · Accelerators** — "A transformer does not run on `mps` yet" | **FALSE** | As Metal above | Corrected |
| **Status · NPU** — "nothing here has run on an NPU" | **FALSE** | Neural Engine execution test passes | Corrected; the NNAPI half is kept and sharpened |
| **Status · `torch.distributed`** — "Everything else… refuses by name" | **FALSE** | As above | Corrected |
| **Status · Test-time adaptation** — "`nn.LayerNorm` models are refused: `aten.native_layer_norm.default` has no derivative rule" | **FALSE** | Provoked: `nn.LayerNorm(8)` backward runs and produces a gradient. `test_train.py:703` already asserts the rule is present — the sentence had simply not been revisited | Corrected. **`docs/models/ADAPT.md` still carries the old sentence** and is left standing and marked (§3) |
| **Status · ATen operators — 302** | **Checked, unchanged** | `len(torch._C._aten_implemented())` = **302**; DOCWATCH `golden_ops_covered ge 302` holds | Unchanged |
| **Status · golden 11,420 / 11,420, `failed` 0, `pending` 0** | **Checked, unchanged** | Gate: `golden_cases_total = 11420`, `passed = 11420`, `failed = 0`, `pending = 0` | Unchanged |
| **Status · 297 of 297 architectures forward** | **Not re-checked — left standing** | No fresh sweep was run (§3) | Unchanged, with its existing caveat that 297 rests on ARCH300's 290 plus seven individual re-runs |
| **"Not working yet" · "7 of 297 architectures do not forward"** | **FALSE, and self-contradictory** | No run needed: the Status table in the *same file* records all 297 forwarding | Marked struck-through and explained, rather than deleted |
| **"Not working yet" · "no transformer forwards there [`mps`]"** | **FALSE** | As Metal above | Corrected; the Vulkan half of the bullet is true and is now stated separately |
| **Devices table · `mps` "54 of them"** | **FALSE — a number** | `len(_C._shim_mps_host_readback_ops())` = **85** | **54 → 85.** It had been 87 before MPSATTN.md removed two. Nothing compared the published number to the runtime; a test now does (§5) |
| **Devices table · NNAPI · CoreML "needs the graph path, blocked at decomposition"** | **FALSE** | Both execute; see the NPU rows | Corrected, with the acceleration-vs-execution distinction kept |
| **dtypes · "the other 35 … complex … refuse by name"** | **Misleading rather than false** | Provoked each: `torch.zeros(dtype=complex64)` refuses, but `torch.fft.fftn` **returns a `complex64` tensor**, and `view`, `slice`, `mul` and `pad` compute on it while `add`, `abs`, `reshape` and `ifftn` refuse by name | Left standing and marked (§3): complex is not a storable dtype, but complex tensors exist as a `Repr` arm and five ops were taught them |
| **dtypes · `int8`/`qint8`/`quint8` unstorable** | **Checked, unchanged** | Provoked: `torch.zeros(2, dtype=torch.int8)` refuses, naming the candle backend. `uint8` constructs, as the table says | Unchanged |
| **Status · Speed vs upstream** (prefill 0.97x/1.13x/1.52x/2.03x, decode 46.6 tok/s) | **Stale, not re-measured** | Deliberately not measured — see §4 | Row marked ⚠️ stale with the numbers **kept as the last reading**, not deleted |
| **Platforms table** — CI, iOS device, Windows arm64, WASM, Android x86_64 cells | **Not re-checked — left standing** | Needs CI runs, an iPhone, an ARM64 Windows machine and a Pyodide run; none reachable from this round (§3) | Unchanged |

---

## 2. The one the brief got wrong, and why that matters

The brief for this round asserted that the NPU row was stale because `docs/graph/NPU2.md`
"executed it on a device through `ANeuralNetworksModel`, 8/8 operations". That is true, and it is
**not** the same claim as the one the row makes.

The row says the NNAPI blob **"has never met an NPU."** So the blob was run on a physical
Galaxy Tab S9 Ultra — SM8550, Snapdragon 8 Gen 2, which has a Hexagon NPU — and the runtime was
asked which drivers it offers:

```
NNAPI devices reported by the physical device:
    {'name': 'nnapi-reference', 'type': '2', 'version': 'X910XXS6EZH3', 'feature': '1000008'}
```

**One driver, and it is the CPU reference implementation.** NNAPI is deprecated as of Android 15
and vendors have moved to LiteRT delegates, so the Hexagon NPU is not reachable through this API
on this image. The blob executed and agreed to 1.2e-07 — on a CPU.

So "has never met an NPU" **stands**, and had this round taken the brief's list on trust it would
have replaced a true sentence with a false one, in a round whose entire purpose is to stop that.
This is CLAUDE.md §5.4: the criterion I am handed decides the answer, and an agreeing result is
the moment to look harder rather than less hard. The row now separates the two claims — the blob
**executes** (understated before) and NNAPI **has not reached an NPU** (true before, true now) —
because they were fused into one sentence and one of them moved.

The Neural Engine is a different matter: CoreML **does** put a float16 graph on it, so the
Status row's broader "nothing here has run on an NPU" was false and is corrected.

---

## 3. What could not be checked, and is left standing

Left in place and marked rather than deleted, per this round's rules:

- **The full 297-architecture sweep.** Not re-run. The README's own caveat already says 297 rests
  on ARCH300's 290 plus seven individual re-runs rather than a fresh sweep; that caveat is the
  honest statement and is kept.
- **The Platforms table's CI, iOS-device, Windows-arm64 and WASM cells.** These need a hosted
  runner, an iPhone, an ARM64 Windows machine and a Pyodide run. None was reachable here.
- **Android x86_64.** Cannot be checked on Apple Silicon at all — the emulator ships only
  `qemu/darwin-aarch64`. The refusal is by name and stays.
- **`docs/models/ADAPT.md`'s LayerNorm sentence.** The README's copy is corrected; the `docs/`
  file still carries "`nn.LayerNorm` models are refused". A `docs/` file is a record of what one
  round measured, so it is **not** rewritten here — but a reader arriving at ADAPT.md alone will
  still be told a false thing, and that is recorded rather than silently left.
- **The complex dtype row.** Left standing: it is accurate about *storability* and misleading
  about *capability*. Correcting it properly means describing the `Repr::Complex` arm in a table
  organised by dtype, which is a restructure rather than a re-measurement.

---

## 4. Performance — deliberately not measured

`docs/perf/PERF.md` is dated **2026-08-25** at **96** operators; the tree is at **302**. This
machine cannot be made idle: the app displaying the session holds roughly 2.2 of 8 cores and two
other agents were running throughout. A loaded machine has already made one commit here read
between **672 and 1076 ns**, which destroyed a bisect.

A perf round would have to measure, on an idle machine and **without filtering the suite** —
filtering changes the measured region, and cost one bisect round when both ends rose to
1300–1400 ns:

- prefill at 6, 128, 512 and 1024 tokens against upstream, in `float32` **and** `bfloat16`;
- `generate()` decode tok/s with a KV cache (the last reading was 46.6 against upstream's 44.4);
- the Android NEON-vs-AMX split, dispatch-bound against kernel-bound, on a physical device.

The existing numbers are kept in the README as the last reading, marked stale, not deleted.

---

## 5. Tests added, and why only these

Most corrections in §1 already had something holding them, which is the reason not to add a test:
`test_collect2.py` for the collectives, `test_train.py` for both the convolution backward rule and
the `native_layer_norm` derivative (`test_train.py:703` already asserted the rule that the README
said was missing), `test_mpsattn.py` for the softmax gate, `test_npu2.py` for the Neural Engine and
the NNAPI replay, `test_vulkan4.py` and `test_cuda.py` for the rows that did not move.

**One correction had nothing behind it**: the host-readback count. The README published 54, the
runtime reported 85, and nothing anywhere compared the two — the same shape as the golden `ge`
floors that could not see `passed < total`. `rust/torch_c/pytests/test_remeasure2.py` adds four
guards, and reads the number **out of the README prose** rather than restating it, so it checks the
document instead of copying it.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_remeasure2.py test_the_readme_host_readback_count_is_the_number_the_runtime_reports present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_remeasure2.py test_softmax_is_not_in_the_host_readback_set present -->
<!-- DOCWATCH: count golden_ops_covered ge 302 -->
<!-- DOCWATCH: count golden_cases_failed eq 0 -->

**Nullification.** The count guard was shown to fail: setting the README back to 54 turned it red
with `README.md says 54 host-readback ops; the runtime reports 85`, and the other three stayed
green. A guard that cannot fail is not a guard (CLAUDE.md §5.5).

---

## 6. Tally

Counted as CLAUDE.md §5.3 asks — by kind, not by test count.

- **Claims evaluated:** 24 rows and bullets.
- **Stale:** **9** — eager training, `torch.distributed` (Roadmap *and* Status), NPU (Status), Metal
  (Roadmap), Accelerators (Status), Test-time adaptation, the `mps` readback count, the
  NNAPI·CoreML devices row, and the "7 of 297" bullet.
- **Direction:** **all nine understate what works.** Not one row overclaimed. That is the same
  direction as every previous round, and it is the direction a reader cannot check, because a
  claim that something does not work tells them not to try it.
- **Checked, unchanged:** 8 — Vulkan, CUDA, `torch.compile`, operator count, golden totals, `int8`
  unstorability, `max_pool2d` backward's refusal, and the closed device vocabulary.
- **Left standing and marked:** 5 — the 297 sweep, the Platforms cells, Android x86_64, ADAPT.md's
  LayerNorm sentence, and the complex dtype row.
- **Corrected in the brief itself:** 1 — "NNAPI has never met an NPU" is **true** (§2).
- **Documentation corrections:** 15 edits to `README.md`. **Feature changes: none.** Nothing in
  this round implemented anything; the tree is unchanged apart from one new test file.
