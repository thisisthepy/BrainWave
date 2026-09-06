<div align="center">

# torchnative

**Run the real PyTorch ecosystem on device, not reimplementing it.**

[![PyPI](https://img.shields.io/pypi/v/torchnative?color=blue)](https://pypi.org/project/torchnative/)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Android%20%7C%20iOS%20%7C%20Linux%20%7C%20Windows-lightgrey)](#platform-support)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange)](#status)

</div>

---

`torchnative` replaces PyTorch's compiled core — `torch._C` — with a native extension, so the
genuine `torch` and `transformers` packages run on a phone the way they run on a workstation.

Models are not ported, converted, or re-expressed. They are imported.

```python
from transformers import AutoModelForCausalLM     # the real one
model = AutoModelForCausalLM.from_pretrained("...")
model.generate(...)                                # on the device
```

> [!WARNING]
> **Pre-alpha.** The operator layer matches upstream PyTorch numerically, 19 of 20 tested
> architectures reach zero missing operators, real checkpoints load, `transformers` imports and
> generates, and an Android device runs the built artefact — but there is no accelerator backend,
> `torch.compile` does not work, and three of six platforms have been executed rather than
> merely built. See [Status](#status) and [Platform support](#platform-support) before depending
> on this.

---

## Why not a reimplementation

Every other route to on-device inference re-expresses the model somewhere else.

| | approach | cost |
|---|---|---|
| llama.cpp | architectures rewritten in C++ | each new architecture is a porting task |
| ExecuTorch · CoreML | ahead-of-time compiled graph | export step, and what runs is not what you wrote |
| MLC | lowered to its own runtime | same |
| **torchnative** | **the real Python package** | **the substrate is hard; architectures are free** |

The reason nobody runs the real thing is that `torch._C` cannot be built for mobile. PyTorch's own
build sets `INTERN_BUILD_MOBILE` for any Android or iOS toolchain, and that path forces
`BUILD_PYTHON` off — so the mobile build is structurally incapable of producing the Python
extension module the Python package needs.

`torchnative` supplies that module instead. Everything above it is upstream source, unmodified.

---

## What it does

### 1 · LLM inference

Run `transformers` models directly. No conversion step, no per-architecture port — if
`transformers` supports it and the operators are covered, it runs.

Tokens arrive one at a time, the way an app wants them:

```python
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

name  = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
tok   = AutoTokenizer.from_pretrained(name)

streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
inputs   = tok("On-device inference is", return_tensors="pt")

Thread(target=model.generate, kwargs=dict(**inputs, max_new_tokens=32, streamer=streamer)).start()

for piece in streamer:          # yields as the model decodes
    print(piece, end="", flush=True)
```

**Today:** this is the output of that exact script, not a sketch of it —

```
 a very powerful technique for learning from data. It is a powerful technique for
 learning from data because it is a very fast and efficient way to learn from data.
```

first token in **34 ms**, then **47 tokens/second** on an M-series desktop. `generate` runs on a
background thread and the main thread consumes the iterator, so the shim is holding up under two
threads and the GIL, not only under a single-threaded loop.

The weights come from the Hub through `from_pretrained` — 273 tensors, bit-identical to upstream —
and in `float32` the tokens are the same ones upstream emits.

Loaded in the checkpoint's native `bfloat16`, which is what you get if you pass no `dtype`, the
tokens diverge. That is not a defect to fix: upstream disagrees with *itself* on one prompt in
three under a mathematically equivalent change of accumulation order, so bitwise agreement in
bf16 is not a bar any independent implementation can clear. See [Status](#status).

### 2 · Federated learning

Devices train locally and share updates, not data. Federated averaging *is* collective
communication, so this is built on `torch.distributed` rather than beside it — broadcast the
model, gather the updates, weighted all-reduce.

```python
from torchnative.nn import federated

engine = federated.Engine(model, method=adapt.Tent(), aggregator=federated.FedAvg())
report = engine.participate(batches, weight=n_local_samples)   # local epochs, then a delta
```

**Today:** **one round runs, between two operating-system processes that share no memory.** Each
rank adapts locally with `adapt.wrap(model, method=Tent())`, contributes the delta that produces,
and comes back holding the group's weighted average — and the acceptance check is not that the
distributed path returned something. It is that

```
aggregate_across_the_two_ranks  ==  (3·d0 + 7·d1) / 10
```

element for element, with the right-hand side computed **centrally** in a third process on upstream
torch from the two deltas the ranks dumped. `torch.equal`, not a tolerance: every operation on both
sides is a correctly-rounded IEEE `float32` multiply, add or divide, so one ulp apart would be a
real disagreement. Both ranks land on the same bits, and after the round they hold the same model
([`docs/FEDERATED.md`](docs/FEDERATED.md)).

**The trap was that `FedAvg` at `world_size = 1` is the identity function** — it returns the delta
it was handed, so a test at that size passes whether the weights are honoured, ignored, or never
read. So a world of one is *refused* at all four doors rather than served, two threads are not
enough either, and the controls are part of the claim: the weighted average differs from the
unweighted one by 0.041 and 0.165 on the two covered parameters, so an aggregator that dropped its
weights fails. Five injected defects were counted, and the two that removed an agreement check made
the mismatch **complete silently** — different parameter sets summed into a number with no
exception.

The three things it was waiting on are all closed: `torch.save` works and upstream reads what it
writes bit-for-bit across eight dtypes ([`docs/SAVE.md`](docs/SAVE.md)); `world_size = 2` works over
a real socket, through the ordinary `init_process_group(backend="local", init_method="tcp://…")` and
not a private door ([`docs/TRANSPORT.md`](docs/TRANSPORT.md)); and `Delta.publish`, the seam, now
sends.

**What it does not cover, by name:** more than one round, participant selection, dropout handling,
secure aggregation, and any aggregator that is not `FedAvg` — each refusing with what it would take
rather than approximating. A rank that does not arrive makes the round raise; it never produces a
partial average. And a delta above ~2 MB on the wire refuses, because the transport under it sends
before it receives and deadlocks — a limit of `ProcessGroupLocal`, named there rather than worked
around here.

### 3 · Test-time adaptation & training

A model that ships to a device meets data the training set never had. TTA, TTT and the wider
test-time learning family let it adapt in place — and every method reduces to the same thing: a
weight delta over base weights, differing only in lifetime and destination.

```python
from torchnative import adapt

model = adapt.wrap(model, method=adapt.Tent(), lr=1e-3)
model.online()                                   # adapt as it serves
model.revert()                                   # the base weights are back, byte for byte
```

**Today:** **Tent runs on a real checkpoint.** On `SmolLM2-135M`, ten steps of entropy
minimisation over the 61 normalisation weights take prediction entropy on unlabelled text from
**4.1604 to 2.9828**, and a *held-out* sentence the loop never adapted on falls **3.7237 to
2.9439** — so the model adapted rather than memorising one batch. The adapted weights agree with
upstream's own autograd running the same step to a median relative **1.5e-06** with 100% sign
agreement over 35,136 numbers. The controls are part of the claim: the same code with the
objective's sign flipped sends entropy *up* to 7.4062, `lr=0` holds it identical to the last
printed digit, and an objective on a detached tensor is refused by name rather than running
vacuously ([`docs/ADAPT.md`](docs/ADAPT.md)).

The delta abstraction of [`docs/DESIGN.md`](docs/DESIGN.md) §3 is what carries it:
`torchnative.delta.Delta` owns the base, the offset, and the three lifetime questions, so
`Tent` holds no state and is 40 lines. A revert restores the base **bit-identically** — all 272
parameters, not only the 61 covered — and the base copy costs 137 KiB against the model's
513 MiB. Lifetime is driven by system events rather than by the domain boundaries a benchmark
hands you, and the two lifetimes that do not exist yet (surviving a restart, leaving the device)
refuse with the check that would prove them stale.

**What it does not cover:** any `nn.LayerNorm` model. `aten.native_layer_norm.default` has no
derivative rule, so every RMSNorm architecture adapts and `gpt2`/`bert` are refused before the
backward, by name.

---

## How it works

```
your code · transformers · torch/*.py        upstream Python, unmodified
──────────────────────────────────────────
torch._C                                     ← replaced
  ├── _aten_dispatch                         the single door every operator passes
  ├── Python spellings                       torch.mm, x.softmax(), F.linear, ...
  └── kernels                                Rust, backed by candle
──────────────────────────────────────────
CPU today · Metal, Vulkan, NPU planned
```

**One door.** Every operator reaches its kernel through `_aten_dispatch`, and nothing bypasses
it. That makes the surface measurable — an unimplemented operator names itself rather than
failing downstream — and it gives graph capture, which NPU backends will need, exactly one place
to attach.

**Demand-driven.** Nothing is implemented because it might be needed. The shim refuses by name,
the refusal names the next thing to build, and that list comes from running real models.

**Stable ABI.** Built against CPython's limited API (`abi3-py313`), so one binary per platform
loads on 3.13, 3.14 and later without a rebuild.

---

## Status

<table>
<tr><th align="left">Working</th><th align="left"></th></tr>
<tr><td>ATen operators</td><td><b>203</b>, each compared against upstream</td></tr>
<!-- DOCWATCH: count golden_ops_covered ge 203 -->
<tr><td>Golden comparison cases</td><td><b>8,509 / 8,509</b> — values, shapes, dtypes, positional <i>and</i> keyword, through the door <i>and</i> through the member</td></tr>
<!-- DOCWATCH: count golden_cases_total ge 8509 -->
<!-- DOCWATCH: count golden_cases_passed ge 8509 -->
<!-- DOCWATCH: count golden_pending eq 0 -->
<tr><td>Smoke tests</td><td><b>389</b></td></tr>
<!-- DOCWATCH: count smoke_ok ge 389 -->
<tr><td><code>from_pretrained</code></td><td>works for models whose init computes on the <b>meta</b> device — the Llama-3.2 <code>rope_scaling</code> path needed 30-odd meta kernels that were absent (<a href="docs/META.md">META.md</a>)</td></tr>
<tr><td>Signature and schema tables</td><td><b>4,641</b> entries checked against upstream</td></tr>
<!-- DOCWATCH: count schema_entries_matched ge 4641 -->
<!-- DOCWATCH: count schema_entries_total ge 4641 -->
<tr><td>Architectures — operator coverage</td><td><b>26 of 26</b> reach zero missing operators in the traced sweep</td></tr>
<tr><td>Architectures — <b>actually forward</b></td><td><b>26 of 26</b>, matching upstream. Agreement is module-by-module through forward hooks, because two of the toy outputs are degenerate enough that their argmax is a tie — reported as a tie rather than as a match (<a href="docs/KERNELS26.md">KERNELS26.md</a>)</td></tr>
<tr><td>Checkpoints</td><td><code>torch.load</code> and safetensors, round-tripped against upstream</td></tr>
<tr><td>Build targets</td><td>macOS · Android · iOS · Linux · Windows — <b>five of six build a wheel</b>. WASM builds the extension and computes under Node, and a wheel has been built by hand but not by <code>build.py</code> (<a href="#platform-support">table</a>)</td></tr>
<tr><td>Training mode</td><td><b>26 of 26</b> forward in <code>.train()</code> as well as <code>.eval()</code>, agreeing with upstream draw for draw — <code>bernoulli_</code> draws in <code>float64</code> for every dtype, so a seeded dropout is comparable. Test-time adaptation runs on real checkpoints — <code>adapt.wrap(model, method=adapt.Tent())</code> drops GPT-2's prediction entropy 39% and transfers to held-out text — in <code>.train()</code> as well as <code>.eval()</code>, with dropout active. A training step moves all 272 SmolLM2 parameters the way upstream moves them — gradients compared element-wise over all 134,515,008 values, sign agreement 99.9987%. It is a <b>tape over a captured region</b>, not <code>Tensor.backward()</code>, which still refuses — though <code>requires_grad=True</code> is now carried rather than refused, and the refusal has moved to the engine itself and names what does work. Ten walls stand between here and an eager <code>.backward()</code> and <b>seven of them do not raise</b>, so a traceback finds one at a time (<a href="docs/BACKWARD2.md">BACKWARD2.md</a>, <a href="docs/BACKWARD3.md">BACKWARD3.md</a>) Unlike <code>torch.compile</code>, autograd <b>is reachable under abi3</b> — <code>torch/csrc/autograd</code> defines <code>Py_BUILD_CORE</code> in 0 of 129 files — and a SmolLM2 backward needs 24 ops of which 16 exist and one is a real missing kernel (<a href="docs/AUTOGRAD.md">AUTOGRAD.md</a>)</td></tr>
<tr><td>Test-time adaptation</td><td><b>Tent runs on SmolLM2-135M.</b> Ten steps of entropy minimisation over the 61 normalisation weights: entropy <b>4.1604 → 2.9828</b> on unlabelled text, <b>3.7237 → 2.9439</b> on a held-out sentence never adapted on, adapted weights within a median relative <b>1.5e-06</b> of upstream's own autograd at 100% sign agreement. Reverting restores the base <b>bit-identically</b> across all 272 parameters, for a 137 KiB base copy against 513 MiB of model. The wrong sign sends entropy <i>up</i>, <code>lr=0</code> holds it to the last digit, and a detached objective is refused by name — because a loop that silently does nothing passes every test that only checks it completed. <code>nn.LayerNorm</code> models are refused: <code>aten.native_layer_norm.default</code> has no derivative rule (<a href="docs/ADAPT.md">ADAPT.md</a>)</td></tr>
<tr><td>Devices run</td><td>Android arm64 — <code>import torch</code>, 119 ops, <code>nn</code> forward. <b>WASM runs under Pyodide</b> — a hand-built wheel installs, imports and computes, on CPython 3.14</td></tr>
<tr><td>Speed vs upstream</td><td>desktop CPU, SmolLM2-135M prefill: <b>0.97x at 6 tokens, 1.13x at 128, 1.52x at 512, 2.03x at 1024</b> in <code>float32</code> — the gap grows with sequence length and what is left is attention (<a href="docs/SEQLEN.md">SEQLEN.md</a>). In <code>bfloat16</code> it is <b>2.3x faster than upstream</b> (<a href="docs/DTYPE_PERF.md">DTYPE_PERF.md</a>). Decode is the other half and it was never measured until now: <code>generate()</code> with a KV cache — the default, and what the example above runs — is <b>0.95x</b>, <b>46.6 tok/s against upstream's 44.4</b> on SmolLM2-135M <code>float32</code>, with character-identical output. The long-sequence gap is attention, and <b>not because we materialise the score matrix</b>: two independent blocked kernels were built to stop materialising it and both were slower — upstream's own, reproduced exactly, by 20x (<a href="docs/FLASH.md">FLASH.md</a>)</td></tr>
</table>

**All twenty forward**: Llama · GPT-2 · Qwen2 · Mistral · Gemma · GPT-NeoX · OPT · MPT ·
StarCoder2 · StableLM · OLMo · Phi · Mixtral · BERT · BLOOM · Cohere · Falcon · Mamba ·
Persimmon · GPT-BigCode

The two rows measure different things, and conflating them is a mistake this README made. The
coverage sweep traces a forward pass **on upstream torch** and asks whether every operator it
dispatches is implemented here — so it cannot see anything that is not an operator: an unbound
tensor member, a missing `torch.<name>` spelling, a dtype-promotion rule.

Closing the six took 11 new kernels, 12 spellings, 16 tensor members, 13 `_C` surface names and 3
rule changes — and **none of the six stopped on only one wall.** Each had one to five more behind
it, of a different kind each time: Cohere needed three spellings and no kernel at all, BERT went
surface then spelling then kernel. "One operator away" was never true of any of them
([`docs/ARCH20.md`](docs/ARCH20.md)).

Measured against **`transformers` 5.x**, which is what a fresh `pip install transformers`
resolves today. 4.x costs four more architectures and needs a disjoint set of operators from
Mixtral ([`docs/COMPAT.md`](docs/COMPAT.md)).

`uniform_` and `normal_` are **bit-identical** to upstream, and `multinomial` consumes the same
generator stream — a seeded run reproduces exactly. `randn`, `rand`, their `_like` forms and
`torch.normal` are composed from those, and agree with upstream value for value under a seed.

**Not working yet**

- `torch.compile` does not work, and the reason is structural rather than a missing piece.
  Dynamo's frame-evaluation hook needs CPython internals — all six C files under
  `torch/csrc/dynamo` define `Py_BUILD_CORE`, and `set_eval_frame` reaches
  `_PyInterpreterState_SetEvalFrameFunc` on a `_PyInterpreterFrame` — which cannot coexist with
  the limited API in one extension. **`torch.compile` and abi3 are mutually exclusive**, and abi3
  is what lets one binary per platform serve 3.13 and every later CPython. Eager is the supported
  path, and graph capture through the single door — already bit-exact against eager — is the
  route being pursued instead ([`docs/DYNAMO.md`](docs/DYNAMO.md)).
- CPU only. No GPU or NPU backend.
- The Android run is an emulator, not a phone. No number here describes real silicon.
- Apple is much faster than Android at `f32` matmul, and that is the hardware. Accelerate
  reaches the AMX coprocessor; ARMv8.2-A NEON has no equivalent. Our Android throughput equals
  our own throughput on the same core under the same backend, at 88% of that core's NEON peak —
  so the kernels are not the gap. Upstream PyTorch has no Android wheel, so how we compare to it
  *there* is unmeasured. See [`docs/PERF_ANDROID.md`](docs/PERF_ANDROID.md).
- **Speed work transfers to Android, but not uniformly.** Dispatch-bound wins arrive slightly
  larger on device than on the host; kernel- and bandwidth-bound ones arrive *smaller* — the
  attention copy is 3.6x there against 5.25x here. Measured by swapping one `.so` between
  published wheels, which land the optimisations one at a time
  ([`docs/PERF_ANDROID.md`](docs/PERF_ANDROID.md) §10).

Tracked with the measurements behind them in [`docs/DESIGN.md`](docs/DESIGN.md) §11.1.

---

## Platform support

Three axes, and they are not independent: a dtype only means something on a device, and a device
only exists on a platform. Every ✅ has a run behind it.

**Legend** — ✅ measured working · ❌ measured refusing · ⚠️ built, never executed ·
🔲 not built · — not applicable to that platform

### Platforms

| | macOS<br>arm64 | Android<br>arm64 | iOS sim<br>arm64 | iOS device<br>arm64 | Linux<br>x86_64 | Windows<br>x86_64 | WASM |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| in the target matrix | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — *deliberately* |
| rust target installed | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| target CPython | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ *Pyodide 3.14* |
| candle builds | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| candle **computes** | ✅ | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | ✅ *under Node* |
| extension builds | ✅ | ✅ | ✅ | ✅ | ✅ *`cargo-zigbuild`* | ✅ *`cargo-xwin`* | ✅ *emscripten* |
| wheel builds | ✅ | ✅ | ✅ | ✅ | ✅ *`manylinux_2_17`* | ✅ *`win_amd64`* | ⚠️ *by hand, not by `build.py`* |
| symbols resolve | ✅ | ✅ | ✅ | ⚠️ *weaker: ELF names only versioned imports* | ✅ *PE names every one* | ✅ *stub behaviour proven against the real host* |
| `dlopen` + `PyInit_` runs | — | — | — | — | — | — | ✅ |
| installs | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ *mounted, no wheel* |
| `import torch` | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| computes | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **on PyPI `0.0.12a0`** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| can be run *here* | ✅ | emulator | simulator | ❌ | CI | CI | ✅ *Node* |

**Linux and Windows now compute, and it is a run rather than an argument.** A hosted runner is the
machine this project does not have, so `.github/workflows/verify-published-wheel.yml` installs the
**published** wheel from PyPI on `ubuntu-latest` and `windows-latest` at Python 3.13 and asks it to
work. Both answered `RESULT: ALL PASS` — `mm`, `nn.Linear`, and the mixed-dtype promotion where the
value and not just the label is at stake (`int64(2049) - float16(1.0)` is `2047.0`) — and then both
ran SmolLM2-135M through real `transformers` and produced text **character-identical to macOS
arm64**. Every expected value is hardcoded from an arm64 run of the same source, so a disagreement
would have localised to the platform rather than to the check.

Windows had moved once before on a user's report rather than our own run, and that report could not
carry `computes`: they installed `0.0.5a0` with `uv`, `import torch` succeeded, and `transformers`
carried them 655 lines into `modeling_rope_utils.py` before a missing **meta** kernel — since fixed
— but everything on that path ran on the meta device, which by construction computes nothing.

The same report settled something else the table has no row for. Their interpreter was **Python
3.14** and the wheel is `cp313-abi3`. One binary per platform loading on 3.13 and every later
CPython is the property five published wheels rest on, and until then it had only been argued.

The last row is why the columns differ, and the iOS column had to become two. **The simulator
computes**: `verify_ios_sim.py` boots one, unpacks the published wheel into an iOS CPython's
`site-packages`, and gets `aten.mm`, `x + x` and an `nn.Linear` forward, with `platform.system()`
answering `iOS`. That runs on every release build and in CI. It was previously reported as ⚠️
along with the device, which understated it.

**The device is the one platform with no way to run, and CI cannot close it.** A hosted macOS
runner has a simulator, not an iPhone — the same rung, on somebody else's machine — and the
simulator runs on the *host* kernel, which its own output says: `uname().version` is this Mac's.
Same instruction set, different Mach-O platform, separate artefact. So the device column stays at
*symbols resolve* until a physical device runs it.

WASM is the exception, and it has now been executed. A complete emsdk with `emcc` and a bundled
Node 24 sits in this machine's cache — `command -v node` finds nothing only because it is not on
`PATH`, which an earlier draft of this line published as "no node on this machine". Under a real
Pyodide the extension loads, `import torch` returns 2.13.0 from the vendored tree, and `a @ b` and
an `nn.Linear` forward match a host build. Two things keep it short of the others: Pyodide ships
CPython **3.14**, not 3.13, so the module is tied to one interpreter rather than to an abi3 floor
— and `torch/__init__.py` imports `torch.multiprocessing`,
which a browser sandbox cannot supply, so that import is stubbed by the harness rather than solved.
A WASM wheel has been built by hand, installed into Pyodide 314.0.6 and imported; `build.py`
does not build one, because `verify_cross.py` reads ELF and Mach-O symbol tables that a wasm
module does not have, and a target this repo cannot check would ship unchecked
([`docs/WASM.md`](docs/WASM.md) §9).

### Devices

| device | macOS | Android | iOS | Linux | Windows | WASM | what it is |
|---|:--:|:--:|:--:|:--:|:--:|:--:|---|
| `cpu` | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | ✅ | the only device that holds a tensor |
| `meta` | ✅ | ✅ | ⚠️ | ⚠️ | ⚠️ | 🔲 | shape and dtype, no storage |
| `mps` | ✅ | — | 🔲 | — | — | — | candle's Metal backend, on. An op whose kernel would compute on the **CPU under an `mps` label** is refused at the door, naming the op — 54 of them, listed by `_C._shim_mps_host_readback_ops()` ([`docs/MPS.md`](docs/MPS.md)) |
| `vulkan` | ✅ | ❌ | — | 🔲 | 🔲 | — | four ops by name through real `VkBuffer`s; every other op refuses naming itself |
| NNAPI · CoreML | — | ❌ | ❌ | — | — | — | needs the graph path, blocked at decomposition |
| `cuda` | ❌ | ❌ | ❌ | 🔲 | 🔲 | — | constructible as a label, refuses to allocate |
| WebGPU | — | — | — | — | — | 🔲 | the only accelerator a browser offers |

### dtypes on `cpu`

11 of 46 storable, and the same 11 on both platforms measured — Android was probed on the device
rather than inferred from the host.

| dtype | macOS | Android | iOS | Linux · Windows · WASM | arithmetic path |
|---|:--:|:--:|:--:|:--:|---|
| `float32` | ✅ | ✅ | ⚠️ | ⚠️ · ⚠️ · 🔲 | **macOS: AMX** via Accelerate · **Android: NEON `gemm`**, 88% of core peak |
| `float64` | ✅ | ✅ | ⚠️ | ⚠️ · ⚠️ · 🔲 | `gemm`. **Mixes with the other float and integer dtypes**, on `add`, `sub`, the six comparisons, `max`/`min`, `bitwise_or`, `where`, `cat` and `stack` — result dtype *and* value bit-identical to upstream over a 9×9 grid, which is not the same thing: upstream casts each operand to the common dtype before the accumulator, so `int64(2049) - float16(1.0)` is `2047.0` and not `2048.0` ([`docs/PROMOTE.md`](docs/PROMOTE.md)). `mm`, `matmul`, `bmm`, `convolution` and SDPA still refuse a mixed pair, and so does upstream |
| `bfloat16` · `float16` | ✅ | ✅ | ⚠️ | ⚠️ · ⚠️ · 🔲 | widened to `f32` in registers, accumulated, narrowed once — upstream's rule. Prefill is 1.19x `float32` here and **2.3x faster than upstream's own `bfloat16`**; decode still materialises the widened weight ([`docs/DTYPE_PERF.md`](docs/DTYPE_PERF.md)) |
| `bool` `uint8` `uint32`<br>`int16` `int32` `int64` | ✅ | ✅ | ⚠️ | ⚠️ · ⚠️ · 🔲 | integer kernels |
| `float8_e4m3fn` | ⚠️ | ⚠️ | ⚠️ | ⚠️ · ⚠️ · 🔲 | **it no longer hangs** — the hang was infinite recursion in candle's own `with_dtype!` for this type, which release-mode tail-call optimisation collapses into a bare `jmp` to itself, so it span the CPU without ever overflowing the stack. Comparison, `tolist`, `item` and `matmul` refuse by name instead. It diverges the other way too: upstream ships `mul` and `abs` for this dtype and refuses `add`/`sub`/`div`/`neg`/`exp`/`sum`/`mean`, **and this build **now refuses the same set, in upstream's own kernel wording**. A first probe found seven; enumerating all 197 ops found **114 divergent** — 48 computed and 27 *hung* where upstream refuses ([`docs/FLOAT8B.md`](docs/FLOAT8B.md)). **In** the golden suite now — the exclusion reason was that construction hung on both sides, which stopped being true |
| `int8` `qint8` `quint8` | ❌ | ❌ | ❌ | ❌ | candle's `DType` has no `I8`: the tensor cannot be created. Adding it would buy the storage type and not the speed — candle has no int8 matmul either, and its quantisation is `QTensor`/`GgmlDType`, a **separate system** that `Tensor`/`DType` never sees, so teaching `aten.mm` a new element type does not reach the fast kernel. **int8 inference is here, as module replacement rather than as a dtype.** It is reached at load time through the slot transformers provides for it — `from_pretrained(name, quantization_config=TorchnativeConfig("q8_0"))`, a registered `HfQuantizer` — and the leaves are swapped **before the weights land**, so the dense model is never assembled: peak RSS for SmolLM2-135M is <b>924 MB against 1231 dense and 1337 quantising afterwards</b>. `dtype=torch.int8` in that same call is closed by transformers itself, before any of this runs ([`docs/HFQUANT.md`](docs/HFQUANT.md)) |
| the other 35 | ❌ | ❌ | ❌ | ❌ | complex, other float8, 4-bit — refuse by name |

The last two rows are ❌ everywhere rather than 🔲, because the cause is in candle's type system
and does not vary by platform.

### Quantisation — beside the dtype system, not inside it

candle keeps quantisation in a separate `QTensor` type, which is why `int8` being unstorable does
not block it. Reached through `torchnative.quant`, which swaps `nn.Linear`.

| format | macOS | Android | iOS | Linux · Windows · WASM | note |
|---|:--:|:--:|:--:|:--:|---|
| Q8_0 | ✅ | ✅ | ⚠️ | ⚠️ · ⚠️ · 🔲 | lossless on integer operands — bit-identical to a dense `linear` |
| Q4_0 | ✅ | ✅ | ⚠️ | ⚠️ · ⚠️ · 🔲 | 29.5% logit RMS on SmolLM2; degrades generation |
| Q4K | ✅ | ✅ | ⚠️ | ⚠️ · ⚠️ · 🔲 | a k-quant, needing `k % 256` — **a model constraint, not a platform one** |

All three were measured on macOS and on the Android device with the same probe. SmolLM2 cannot use
the k-quants because its layers are 576 wide and 576 is not a multiple of 256 — that is about the
model, and an earlier draft of this table wrongly put it in the platform column.

Android Q4K is **1.60× f32 at prefill** as shipped, and 3.29× with `+dotprod` — which cannot be
turned on, candle having no runtime dispatch and ARMv8.0 devices no `sdot`
([`docs/QUANT.md`](docs/QUANT.md)).

### Linux, Windows and WASM

**Linux x86_64** crosses four of six layers ([`docs/LINUX.md`](docs/LINUX.md)). One thing blocks
it, and it is not the linker — `rust-lld` ships with rustup and links ELF fine. It is that
`x86_64-unknown-linux-gnu` is the one target rustup ships no glibc stubs for, and that
`candle → tokenizers → onig → onig_sys` is a C crate, so the build stops at
`failed to find tool "x86_64-linux-gnu-gcc"` before linking is even reached. `cargo-zigbuild`
supplies all of it and is not installed; that is a decision, not an oversight.

**Windows x86_64** has its CPython distribution and nothing else yet.

**WASM** runs. Under Emscripten and the Node in this machine's emsdk, candle computes a
quantised matmul to `511.96875` — bit-identical to the host, the same quantisation error rather
than a round number agreeing — and `dlopen` loads our own `cdylib`, whose `PyInit_` executes and
returns a module definition ([`docs/WASM.md`](docs/WASM.md) §7). The `onig` subtree drops out
there, so the dependency count falls 129 → 80.

**What it costs is `abi3`.** Pyodide pins CPython 3.13, 3.14 and 3.15 to Emscripten 4.0.9, 5.0.3
and 6.0.5 — three releases, three compilers — so WASM would be one binary per CPython feature
release rather than one per platform. That is a different distribution model from the other five,
not a variation on it. WASI is separately blocked: no `dlopen`, so `torch._C` cannot be a wheel
there at all — but WASI was never the route, and Pyodide, which is, has dynamic linking. And PEP 783 forbids `-pthread`, so the honest line is scalar and single-threaded —
`simd128` is off because candle's own WASM SIMD backend does not compile.

It is absent from the matrix on purpose: that table is a `kernels` *backend* matrix, and `kernels`
has no wasm backend — the same gap it already records for `vulkan`.

---

## Verification

Correctness here means *agreeing with upstream PyTorch*, so the strategy is comparison rather
than assertion.

| | |
|---|---|
| **Golden comparison** | Every operator runs on both upstream torch and this shim, compared on value, shape and dtype. It has caught a `float16` GEMM accumulating in `float16` where torch accumulates in `float32`, `cumsum` routed through the wrong kernel, and integer overflow where torch refuses. |
| **The harness tests itself** | `--self-test` injects a fault shaped like a plausible misimplementation at each comparator and fails if the comparator accepts it — 11 comparators × 11 fault modes, with any comparator never exercised reported as failure. It found that the previous fault injection reached exactly one case out of 1781. |
| **Tokens are not enough** | A wrong `gelu` approximation produced *identical tokens* while logits differed by 5.9e-04. End-to-end tests compare logits too, with a tolerance measured to sit between normal float32 noise and that failure. |

```sh
sh rust/torch_c/pytests/run.sh                  # smoke tests + harness self-test
python tools/golden/compare.py                  # golden comparison against upstream
python rust/torch_c/pytests/verify_schemas.py   # signature tables vs upstream
```

---

## Roadmap

The next milestone is the device abstraction, because everything waits on it — a distributed rank
needs a device to point at, and every accelerator attaches there.

```
torchnative.nn.federated    rounds · client selection · aggregation · dropout
  └ torch.distributed       ProcessGroup · collectives (transport)
      └ backends            ours, via register_backend
          └ devices         CPU · Metal · Vulkan · NPU
```

| | |
|---|---|
| **Device abstraction** | `torch.device`, per-device dispatch. Everything else waits on it. |
| **Metal** | On, Apple targets only. One arm of `resolve()` -- an `mps` tensor is an ordinary candle tensor, so no kernel here had to be taught it ([`docs/VULKAN3.md`](docs/VULKAN3.md)). That is also why it needs a gate the Vulkan representation does not: the ops whose kernels read the tensor back to the host are refused by name ([`docs/MPS.md`](docs/MPS.md)). |
| **`torch.distributed`** | From `world_size = 1` upward. Unblocks `transformers` as a side effect. |
| **Vulkan** | No candle backend, so it is a fourth arm of `tensor::Repr` outside candle entirely — which is what makes a silent CPU fallback unrepresentable rather than merely avoided ([`docs/VULKAN3.md`](docs/VULKAN3.md)). Wiring and correctness are testable on this host; only the performance question needs a phone. |
| **NPU** | NNAPI, CoreML and QNN compile at runtime, so no export step is added — but they take a whole subgraph, not one operator. That needs a capture layer, and the single door is where it attaches. |

---

## Install

```sh
pip install torchnative
```

Every published version is a pre-release, so if your resolver is configured to skip those, ask for
one by name: `pip install --pre torchnative`.

`0.0.12a0` ships five platform wheels, all `cp313-abi3` — one binary per platform, loadable by
CPython 3.13 and every later release. Each carries the `_C` extension and the vendored upstream
tree, so `import torch` resolves to *this* build.

**They are not all verified to the same depth, and the table says which is which.**

| wheel | built | installed | `import torch` | computes |
|---|:--:|:--:|:--:|:--:|
| `macosx_11_0_arm64` | ✅ | ✅ | ✅ | ✅ |
| `android_21_arm64_v8a` | ✅ | ✅ | ✅ | ✅ |
| `ios_12_0_arm64_iphoneos` | ✅ | — | — | — |
| `manylinux_2_17_x86_64` | ✅ | — | — | — |
| `win_amd64` | ✅ | — | — | — |

The iOS simulator wheel **is** published, and this sentence used to say the opposite — that it was
"deliberately not published" because a resolver reaching it would be trapped. PyPI has carried it
since `0.0.2a0`; the two iOS wheels differ by platform tag (`iphonesimulator` against `iphoneos`)
and pip selects on that, so the trap does not exist. Corrected rather than left, because it is the
kind of claim nobody re-reads.

Linux and Windows were in that position and are not any more: CI installs the published wheel on
`ubuntu-latest` and `windows-latest` and both compute, matching macOS arm64 character for character
on a real SmolLM2 generation. What follows is the artefact-level check that used to be all there
was, and it still runs — it catches a broken wheel before anything is uploaded. Every
import in the Linux wheel resolves, and every import in the Windows one is attributed to a
naming DLL, which is the stronger of the two checks because PE records a DLL per import where ELF
records only versioned ones ([`docs/LINUX.md`](docs/LINUX.md),
[`docs/WINDOWS.md`](docs/WINDOWS.md)).

macOS is checked in a clean virtualenv and Android on a device, unpacked into its CPython's
`site-packages` — in both, `torch.__file__` lands inside the install, `aten.mm` returns the right
answer and an `nn.Linear` forward runs ([`docs/WHEEL.md`](docs/WHEEL.md) §7).

> [!IMPORTANT]
> **The iOS wheel has never been executed.** What is verified is everything short of running it:
> its 222 undefined symbols all resolve against the device `Python.framework` and the iOS SDK,
> checked through the two-level namespace bindings dyld itself uses, and every file in it outside
> the extension is byte-identical to the simulator wheel, which does import and compute. What is
> not verified is the load itself, `@rpath` resolution inside a real app bundle, and code signing
> — none of which can be answered without a device ([`docs/IOS.md`](docs/IOS.md)).
>
> If you run it on a phone, we would like to hear either way.

> [!NOTE]
> `0.0.1a0` is still on PyPI and does **not** work — it is `py3-none-any` and carries the
> `torchnative` skeleton alone, no `_C` and no `torch`, so it installs cleanly and then fails to
> import. Ask for `0.0.12a0` or later.
>
> There is no source distribution. Building needs a Rust toolchain and a vendoring step that
> `pip` cannot drive, so an sdist would install and then fail; the recipe is below instead.

### Building from source

Requires a Rust toolchain and CPython 3.13+.

```sh
bash vendor/vendor_torch.sh     # assemble the vendored torch tree
bash vendor/install_shim.sh     # build the extension and install it
```

### Building a wheel

Additionally requires `pip`, `setuptools` and `wheel` in the building interpreter, and a C
compiler for the empty `libtorch_global_deps` (see [`docs/WHEEL.md`](docs/WHEEL.md) §3.2).

```sh
bash vendor/vendor_torch.sh
bash vendor/install_shim.sh
python tools/wheel/build.py                            # -> dist/*.whl
python tools/wheel/verify.py dist/torchnative-*.whl    # clean venv, real import
```

`verify.py` is the part that matters: it installs into a throwaway virtualenv and asserts that
`torch.__file__` resolves *inside* it. A check that lets the development tree answer proves
nothing about the wheel.

Cross-compilation is documented in [`docs/RUST_CROSSBUILD.md`](docs/RUST_CROSSBUILD.md),
including the PyO3 configuration iOS needs in order not to link `libpython`.

---

## Repository layout

```
torchnative/     the Python library
rust/torch_c/    the torch._C replacement (Rust · PyO3 · candle)
tools/golden/    the upstream comparison harness
tools/wheel/     build a platform wheel, and prove it installs (docs/WHEEL.md)
vendor/          scripts that assemble the vendored torch tree (not checked in)
docs/            design, measurements, and the reasoning behind open decisions
```

`docs/` is written to be read. It records what was measured, what was assumed, and where an
earlier conclusion turned out to be wrong — corrections are left visible rather than edited away.
Start with [`DESIGN.md`](docs/DESIGN.md); [`SURFACE_HONESTY.md`](docs/SURFACE_HONESTY.md) and
[`HARNESS.md`](docs/HARNESS.md) show the standard the rest aims for.

---

## Related

- [PythonMultiplatform](https://github.com/thisisthepy/PythonMultiplatform) — embeds CPython 3.13
  into Kotlin Multiplatform; the deployment target for this library
- [pypackpack](https://github.com/thisisthepy/pypackpack) — the build and bundling tool
- [Hugging Face `kernels`](https://github.com/huggingface/kernels) — the fused-kernel contract
  this adopts, with resolution moved from runtime download to build time, since downloading
  executable code is not permitted on every target platform

---

## License

MIT — see [LICENSE](LICENSE).

PyTorch is vendored under its own BSD-3-Clause license. The vendored tree is assembled at build
time and is not redistributed in this repository.
