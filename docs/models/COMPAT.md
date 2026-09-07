# `transformers` 4.x — what it needs that this shim did not provide, and how much it costs

Every architecture measurement in this repository so far — "20 of 20", the golden model runs, the
Mixtral verification in `docs/kernels/GROUPED_MM.md` — was made against `transformers` **5.15.1**, because
that is what `spike-venv` happens to have. `pip install torchnative` (the published 0.0.4a0 wheel)
does not depend on `transformers` at all, so a user picks that version themselves, and with
`transformers==4.*` Mixtral's rotary embedding died on `torch.autocast(device_type=..., enabled=False)`
wanting `torch._C._is_autocast_available`, which this shim did not implement. This document measures
how much of 4.x is missing, closes the three names that turned out to be small and safe, and answers
which version this project should claim.

**One line first.** At the architecture level, with the fixes below applied, 4.x costs this shim
**4 of the 20 "complete" architectures** (`gpt2`, `opt`, `bert`, `mixtral`) — not the "one op" the
README's Mixtral note implied, because under 4.x Mixtral's MoE block does not go anywhere near
`_grouped_mm`. Before the fixes, the cost looked much larger — **13 of 20 architectures refused
outright** on the single missing `_is_autocast_available` predicate, which was hiding the real,
smaller shape of the gap.

> **Correction (re-verified live against the current build): the 4.x cost has since shrunk.**
> Later rounds (out of this document's territory — `aten.rs` kernel work, plus `docs/graph/TORCHSCRIPT.md`'s
> `PYTORCH_JIT` default) closed `gpt2`'s and `mixtral`'s gaps named here. Re-running this document's
> own 20-toy-config sweep against the current shim: **5.x is 20/20** (was 14/20 — `persimmon`,
> `cohere`, `falcon`, `bloom`, `gpt_bigcode`, `mamba` all now construct and forward), and **4.x is
> 16/20** (was 10/20 — only `opt`/`bert` (`aten.all.default`), `mixtral`'s MoE loop (`_nn.one_hot`'s
> decomposition), and `mamba` (`TensorBase.roll`) still refuse). See the corrections at §4 and §5
> below for the per-name detail.

---

## 0. Method

Two measurements, because they catch different things and the difference is itself a finding
(§4).

1. **Op-coverage sweep** — the method `docs/architectures/ARCH.md` §0 and `docs/kernels/OPS4.md` §3 already used for the
   other 19 architectures: trace a small (2-layer, hidden 64) model's forward pass with
   `torch.utils._python_dispatch.TorchDispatchMode` on **upstream** torch (no shim involved), and
   diff the traced op set against `_C._aten_implemented()` read from the **built** artefact. This
   catches missing aten kernels. It does **not** catch anything that happens at `nn.Module.__init__`
   time, because the capture context in this and the prior work only wraps the forward call — §4.1
   found a case this misses.
2. **Full shim run** — actually construct and forward each of the 20 toy models through the
   **built shim itself** (`TORCH_USE_RTLD_GLOBAL=1`, `PYTHONPATH` at the vendored tree), the same
   thing `docs/kernels/GROUPED_MM.md` §5.2 did for Mixtral alone. This is what the op-coverage sweep is not:
   an execution, not just a coverage measurement, and it is the only one of the two that can see a
   missing `torch._C.*` surface symbol (autocast, in this case) rather than a missing aten kernel.
   **For each failure, the model construction/forward is re-attempted after either fixing the
   named gap in this shim's territory or, when the gap was out of territory, leaving it and moving
   to the next architecture** — never patched in place with a throwaway monkeypatch, so every number
   below is the shim's real, current behavior, not a stubbed one. The one exception is explicitly
   marked in §3.3.

Two venvs, both pointed at `python 3.13`:

| venv | transformers | torch | role |
|---|---|---|---|
| `/Volumes/macMini/caches/spike-venv` | 5.15.1 | 2.13.0 (pip) | existing, **not modified**. Reference for every 5.x number below. |
| `/Volumes/macMini/caches/compat-tf4-venv` | **4.57.6** (latest stable 4.x, measured against PyPI 2026-08-30 — see §6) | 2.13.0 (pip) | new, created for this investigation |

Both venvs run the shim by setting `TORCH_USE_RTLD_GLOBAL=1` and `PYTHONPATH=<repo>/torchnative/src/main`,
and run against upstream by leaving those unset — the same dual-purpose pattern `spike-venv` already
used for every prior measurement in this repository (it has both a real pip `torch` and is the
vehicle for running the shim).

20 toy configs (2 layers, hidden 64, 2 heads, vocab 64) covering exactly the architectures the
README lists as complete: Llama, GPT-2, Qwen2, Mistral, Gemma, GPT-NeoX, OPT, MPT, StarCoder2,
Persimmon, Cohere, StableLM, OLMo, Phi, BERT, Falcon, BLOOM, GPT-BigCode, Mixtral, Mamba — built with
`transformers.AutoConfig.for_model(model_type, **kwargs)` rather than hand-written config classes, so
the actual `transformers` code decides the architecture's shape.

**Reproduction.** The harness scripts live outside the repo (`/tmp/compat_trace.py`,
`/tmp/compat_shim_run.py`), the same read-only-repo convention `docs/kernels/GAP.md` §0 already used for its
measurement scripts:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-compat
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export HF_HOME=/Volumes/macMini/caches/hf-home
bash vendor/install_shim.sh

# op-coverage sweep, either venv, unset TORCH_USE_RTLD_GLOBAL/PYTHONPATH
/Volumes/macMini/caches/compat-tf4-venv/bin/python /tmp/compat_trace.py > /tmp/trace_4x.json
/Volumes/macMini/caches/spike-venv/bin/python /tmp/compat_trace.py > /tmp/trace_5x.json

# full shim run, either venv, WITH TORCH_USE_RTLD_GLOBAL=1 and PYTHONPATH set
export TORCH_USE_RTLD_GLOBAL=1
export PYTHONPATH=$(pwd)/torchnative/src/main
/Volumes/macMini/caches/compat-tf4-venv/bin/python /tmp/compat_shim_run.py > /tmp/shimrun_4x.json
/Volumes/macMini/caches/spike-venv/bin/python /tmp/compat_shim_run.py > /tmp/shimrun_5x.json
```

---

## 1. Op-coverage sweep — 4.x vs 5.x, against the 122-op implemented set

```
              4.x ops  4.x missing   5.x ops  5.x missing
llama              26            0        26            0
gpt2                20            0        20            0
qwen2               27            0        27            0
mistral             26            0        26            0
gemma               26            0        26            0
gpt_neox            27            0        27            0
opt                 24            1        23            0    <- aten.all.default (4.x only)
mpt                 32            0        31            0
starcoder2          25            0        25            0
persimmon           28            0        28            0
cohere              29            0        29            0
stablelm            25            0        25            0
olmo                24            0        24            0
phi                 26            0        26            0
bert                21            1        14            0    <- aten.all.default (4.x only)
falcon              27            0        30            0
bloom               36            2        37            0    <- mul_.Tensor, triu.default (4.x only)
gpt_bigcode         17            0        18            0
mixtral             41            5        42            0    <- aminmax, index_add_, nonzero,
                                                                   scatter_.value, zeros.default
mamba               35            4        31            1    <- clamp, constant_pad_nd, roll,
                                                                   zeros.default (+ zeros.default on 5.x too)
```

At the pure operator-coverage level 4.x is close to 5.x — 5 of 20 architectures need something extra,
and for 3 of those 5 it is one or two ops. **Mixtral is the outlier and the important one**: its
5-op gap under 4.x has **zero overlap** with the 5-op gap 5.x's own `_grouped_mm` story left behind
(§3.3).

This sweep has a blind spot, and §1's method note already named it: **it only sees ops dispatched
during the traced forward call, not during `nn.Module.__init__`.** GPT-2's `tril` causal-mask buffer
(§3.2) is built in `__init__`, so it is invisible here on *both* versions even though it blocks 4.x
outright once the model is actually constructed through the shim. That is exactly why §2 exists.

---

## 2. Full shim run — before the fixes in this session

Running each toy model through the **actual built shim**, unmodified, before this session's changes:

```
llama       FAIL  torch._C._is_autocast_available
qwen2       FAIL  torch._C._is_autocast_available
mistral     FAIL  torch._C._is_autocast_available
gemma       FAIL  torch._C._is_autocast_available
gpt_neox    FAIL  torch._C._is_autocast_available
starcoder2  FAIL  torch._C._is_autocast_available
persimmon   FAIL  torch._C._is_autocast_available
cohere      FAIL  torch._C._is_autocast_available
stablelm    FAIL  torch._C._is_autocast_available
olmo        FAIL  torch._C._is_autocast_available
phi         FAIL  torch._C._is_autocast_available
falcon      FAIL  torch._C._is_autocast_available
mixtral     FAIL  torch._C._is_autocast_available
mpt         FAIL  TensorBase.permute
gpt2        FAIL  torch.tril(...) -- no overload table entry
opt         FAIL  torch.all(...) -- no overload table entry
bert        FAIL  TensorBase.__getitem__ with an index of type list
bloom       FAIL  aten.pow.Tensor_Tensor: dtype promotion float32 vs int32
gpt_bigcode FAIL  ModuleNotFoundError (torch.jit.script at import time)
mamba       FAIL  torch.log(...) -- no overload table entry

OK: 0 / 20
```

**Thirteen of twenty architectures were blocked by one missing predicate.** That is the number that
motivated this investigation, and it overstates the real 4.x-specific gap by a wide margin, because
`_is_autocast_available` is not a 4.x-specific feature — it is a general shim gap that 4.x's
attention modules happen to trip over (5.x's equivalent code already went through `maybe_autocast`,
which was implemented) and 5.x's did not.

---

## 3. What was fixed, and what it revealed underneath

### 3.1 `_is_autocast_available` — the decision

**Verdict: `True` for the device types upstream registers an Autocast dispatch key for, `False` for
the rest — not `False` for everything, which is what the investigation prompt guessed and which
turns out to be actively wrong.**

The guess was: this build is CPU-only, has no autocast, so the honest answer is `False`, and that
would make `torch.autocast(device_type=..., enabled=False)` a no-op. That was checked against the
vendored `torch/amp/autocast_mode.py` (2.13.0) rather than assumed, and it does not hold:

```python
# autocast.__init__, torch/amp/autocast_mode.py:222
...
self.device = device_type
if not is_autocast_available(self.device):
    raise RuntimeError(
        f"User specified an unsupported autocast device_type '{self.device}'"
    )
```

This check runs **before `enabled` is inspected anywhere** — `enabled=False` does not skip it. So a
`False` return does not make `autocast(enabled=False)` a no-op; it makes `__init__` itself raise,
on every device type, whether autocast was ever going to be enabled or not. Verified directly by
monkeypatching real upstream torch (`spike-venv`, torch 2.13.0):

```python
>>> torch.autocast(device_type='cpu', enabled=False).__enter__()  # unpatched
# succeeds
>>> torch._C._is_autocast_available = lambda dt: False
>>> torch.autocast(device_type='cpu', enabled=False).__enter__()
RuntimeError: User specified an unsupported autocast device_type 'cpu'
```

What the predicate actually answers, read from upstream's own values (measured on this machine,
which has no CUDA hardware at all):

```
cpu True   cuda True   xpu True   hpu True   mtia True   maia True   ipu True   xla True   mps True   privateuseone True
mkldnn False   opengl False   opencl False   ideep False   hip False   ve False   fpga False   lazy False   vulkan False   meta False
```

`_is_autocast_available("cuda")` is `True` on a CPU-only host. It is a **build-time registration
fact** ("was an Autocast dispatch key registered for this backend"), not a runtime capability check
("does this device physically exist" or "would casting actually happen"). Nothing about it says
casting occurs — that question is `is_autocast_enabled`, which this shim already pins permanently to
`False` and refuses to raise (existing code, unchanged). Answering `_is_autocast_available` honestly
does not reopen that door: `is_autocast_enabled`/`set_autocast_enabled(..., True)` still refuse
exactly as before. This was implemented as the fixed 10/10 split above (`_AUTOCAST_DEVICE_TYPES` from
the existing code, `_AUTOCAST_AVAILABLE_DEVICE_TYPES` new).

Getting `autocast(..., enabled=False)` to actually complete needed four more names in the same
family, all bookkeeping around a cache this shim never populates (`__enter__`/`__exit__` in
`torch/amp/autocast_mode.py:308-352`, none of it gated by `enabled` either):

- `is_autocast_cache_enabled` / `set_autocast_cache_enabled` — a real flag, default `True`
  (measured), and unlike the enabled-flag it is safe to let a caller set it either way: there is no
  cache to mismanage.
- `autocast_increment_nesting` / `autocast_decrement_nesting` — an honest nesting counter (matches
  upstream's measured return values: 1, 2, ... on increment; ..., 0, -1, ... on decrement).
- `clear_autocast_cache` — no-op, returns `None` (measured); there is nothing to clear.
- `set_autocast_dtype` — `get_autocast_dtype` already existed as two hardcoded per-device constants
  (`cpu`→`bfloat16`, `cuda`→`float16`); `autocast.__init__` reads it *unconditionally* into
  `self.fast_dtype` and `__enter__`/`__exit__` write-then-restore it around the region even when
  `enabled=False`. Turned the two constants into a mutable dict so the round trip has somewhere to
  write; unknown device types still refuse on read exactly as before, until a `set` gives them a
  value.

All five: `rust/torch_c/src/bootstrap.py`, `_install_autocast`.

### 3.2 `TensorBase.permute` and `Tensor.T` — binding gaps, not kernel gaps

`mpt`'s failure (`TensorBase.permute`) and one of `falcon`'s two failures (`TensorBase.T`) were not
missing kernels: `aten.permute.default` was already in the 122-op implemented set and already
reachable as `torch.permute(x, dims)` (an entry existed in `overloads.json`), and a
`TorchDispatchMode` trace on upstream confirms `.T` is upstream's own Python-level alias for
`self.permute(*range(ndim - 1, -1, -1))` — no `aten::T` exists. Both were pure wiring gaps:

- `methods.json` had `transpose`, `view`, `unsqueeze`, `squeeze`, ... but not `permute` — added the
  same schema `overloads.json` already carries.
- `TensorBase.T` had never been overridden past the generic "not implemented" placeholder every
  `tensorbase` member gets by default — added as a computed `property` calling the now-wired
  `permute` member.

Both: `rust/torch_c/src/methods.json` (`permute`), `rust/torch_c/src/bootstrap.py`
(`_install_tensor_T`).

### 3.3 What those fixes revealed once applied

Rebuilt and re-ran the full shim sweep. `mpt` passed outright. The 13 architectures blocked on
`_is_autocast_available` mostly passed too — **10 of the 13**, once the autocast fix above was
complete — and the other 3 hit *different*, unrelated gaps once the autocast wall was gone:

```
llama OK   qwen2 OK   mistral OK   gemma OK   gpt_neox OK   mpt OK   starcoder2 OK
stablelm OK   olmo OK   phi OK
persimmon   FAIL  torch.square(...) -- no overload table entry     (present on 5.x too, §4)
cohere      FAIL  torch.repeat_interleave(...) -- no overload table entry  (present on 5.x too, §4)
falcon      FAIL  TensorBase.__getitem__ with an index of type list        (present on 5.x too, §4)
mixtral     FAIL  torch._C._nn.one_hot                                     (4.x's own MoE path, not `_grouped_mm`)
gpt2        FAIL  torch.tril(...) -- no overload table entry               (the `__init__`-time gap §1 predicted)
opt         FAIL  torch.all(...) -- no overload table entry
bert        FAIL  TensorBase.__getitem__ with an index of type list
bloom       FAIL  aten.pow.Tensor_Tensor: dtype promotion float32 vs int32 (present on 5.x too, §4)
gpt_bigcode FAIL  ModuleNotFoundError (torch.jit.script at import time)    (present on 5.x too, §4)
mamba       FAIL  torch.log(...) -- no overload table entry                (present on 5.x too, §4)

OK: 10 / 20   (up from 0 / 20)
```

One exception to "never stubbed in place," flagged as the method promised: `torch._C._nn.one_hot`
was traced one level further **without patching the shim**, by tracing `F.one_hot` on upstream torch
directly — it decomposes into `aminmax`, `_local_scalar_dense`, `zeros`, `unsqueeze`,
`scatter_.value` (CompositeImplicitAutograd, confirmed by schema lookup: it never appears as its own
dispatcher record). This matches, op for op, what the coverage sweep in §1 already found missing for
`mixtral` under 4.x — so implementing `one_hot` itself would not have finished the job; four of its
five component ops (`aminmax` is the exception) are still missing kernels regardless.

None of `mixtral`'s five (§1: `aminmax`, `index_add_`, `nonzero`, `scatter_.value`, `zeros`), `gpt2`'s
`tril`, or `opt`'s `all` are reachable from `bootstrap.py`/`overloads.json`/`methods.json` alone —
each is a genuinely missing aten kernel (`rust/torch_c/src/aten.rs`, out of territory this session,
owned by another agent). `bert`'s and `falcon`'s `__getitem__` failure is the indexing region another
agent is rewriting right now, also out of territory by name. None were implemented.

---

## 4. Full shim run, 5.x — for comparison, same shim, same fixes in place

```
llama OK    gpt2 OK     qwen2 OK    mistral OK   gemma OK    gpt_neox OK  opt OK
mpt OK      starcoder2 OK  stablelm OK  olmo OK   phi OK      bert OK     mixtral OK
persimmon   FAIL  torch.square(...) -- no overload table entry
cohere      FAIL  torch.repeat_interleave(...) -- no overload table entry
falcon      FAIL  TensorBase.__getitem__ with an index of type list
bloom       FAIL  aten.pow.Tensor_Tensor: dtype promotion float32 vs int32
gpt_bigcode FAIL  ModuleNotFoundError (torch.jit.script at import time)
mamba       FAIL  torch.log(...) -- no overload table entry

OK: 14 / 20
```

> **Correction (re-verified live against the current build, transformers 5.15.1/spike-venv): all
> six below now pass — 5.x is 20/20, not 14/20.** `torch.square`/`torch.repeat_interleave` now
> resolve (as Python-level composites decomposing into already-implemented kernels — neither has
> its own `aten.*` entry in `_aten_implemented()`, confirmed), `TensorBase.__getitem__` with a list
> index now works, `aten.pow.Tensor_Tensor`'s dtype promotion (`float32` base, `int32` exponent)
> now runs, `aten.log.default` is implemented, and GPT-BigCode's `torch.jit.script` import wall is
> gone (`docs/graph/TORCHSCRIPT.md`'s `PYTORCH_JIT=0` default; its own construct-time `tril` gap is also
> closed, `docs/kernels/TRIL.md`). All six were re-run end to end
> (`AutoModelForCausalLM.from_config(...).eval()` forward) against the current shim and none raised.

Six architectures fail on **both** versions, identically — `persimmon`, `cohere`, `falcon`, `bloom`,
`gpt_bigcode`, `mamba`. These are pre-existing shim gaps, not part of the 4.x story; they would need
closing regardless of which `transformers` version this project targets, and closing them is out of
this session's territory (all six are `aten.rs` kernel gaps or the forbidden indexing region, except
`gpt_bigcode` — see §5.3).

> This paragraph's premise is superseded too — none of the six still fail (correction above).

**The honest 4.x-specific cost, with the fixes in this session applied, is 4 architectures**:
`gpt2`, `opt`, `bert`, `mixtral` pass on 5.x and fail on 4.x. All four fail for the same underlying
reason — 4.x's older attention-masking and MoE code calls different primitives than 5.x's:

> **Correction (re-verified live): `gpt2` is no longer in this set — `aten.tril.default` was
> implemented (`docs/kernels/TRIL.md`), and 4.x's `GPT2Attention.__init__` now succeeds. `mamba` has taken
> its place**, still failing on 4.x specifically (`TensorBase.roll`, per §1's op-coverage row for
> `mamba` — `clamp`/`constant_pad_nd` from that same row are now implemented, `roll` is not). The
> count is still 4, but the membership changed: **`opt`, `bert`, `mixtral`, `mamba`** — re-run
> end to end against the current shim on both venvs (`compat-tf4-venv` 4.57.6, `spike-venv` 5.15.1).
> `gpt2`'s row below is kept for the record (it correctly explains what *used to* block it) rather
> than deleted.

| architecture | 5.x path | 4.x path |
|---|---|---|
| `gpt2` | builds its causal mask through `masking_utils`, no `tril` at init | `GPT2Attention.__init__` still does `self.register_buffer("bias", torch.tril(...))` |
| `opt` | `masking_utils`-based mask, no `torch.all` | `AttentionMaskConverter._ignore_causal_mask_sdpa` reads `torch.all(attention_mask == 1)` |
| `bert` | no fancy indexing in the padding-warning check | `warn_if_padding_and_no_attention_mask` does `input_ids[:, [-1, 0]]` |
| `mixtral` | `integrations/moe.py`'s grouped-GEMM MoE, needs only `_grouped_mm` (already implemented, `docs/kernels/GROUPED_MM.md`) | `MixtralSparseMoeBlock.forward` (`modeling_mixtral.py:102`) loops over experts with a `one_hot` mask, `nonzero`, `index_add_` |

---

## 5. All missing names, classified

**(a) a symbol we could implement** — small, in this session's territory, done:

- `torch._C._is_autocast_available` (+ `is_autocast_cache_enabled`, `set_autocast_cache_enabled`,
  `autocast_increment_nesting`, `autocast_decrement_nesting`, `clear_autocast_cache`,
  `set_autocast_dtype`) — `bootstrap.py`.
- `TensorBase.permute` — `methods.json`.
- `Tensor.T` — `bootstrap.py`.

**(b) a kernel we do not have** (`rust/torch_c/src/aten.rs`, out of territory this session, owned by
another agent this pass):

| op | needed by | 4.x-specific? |
|---|---|---|
| `aten.tril.default` | `gpt2` (`__init__`) | yes |
| `aten.all.default` | `opt`, `bert` (op-coverage, §1) | yes |
| `aten.aminmax.default`, `aten.index_add_.default`, `aten.nonzero.default`, `aten.scatter_.value`, `aten.zeros.default` | `mixtral` 4.x's MoE loop (§3.3) | yes — disjoint from 5.x's `_grouped_mm` need |
| `aten.square.default` | `persimmon` | no — fails on 5.x too |
| `aten.repeat_interleave.*` | `cohere` | no — fails on 5.x too |
| `aten.pow.Tensor_Tensor` dtype promotion, `float32` × `int32` | `bloom` (`build_alibi_tensor`) | no — fails on 5.x too |
| `aten.log.default` | `mamba` | no — fails on 5.x too |
| `aten.clamp.default`, `aten.constant_pad_nd.default`, `aten.roll.default` | `mamba` (op-coverage, §1; not reached by the shim run before `log` already blocks it) | mixed — `zeros.default` is common to both, the rest only showed up in the 4.x trace |

> **Correction (re-verified live against the current build): six of the ten rows above are stale.**
> `aten.tril.default` (`docs/kernels/TRIL.md`), `aten.log.default`, `aten.clamp.default`, and
> `aten.constant_pad_nd.default` are all now `in torch._C._aten_implemented()` and were confirmed by
> calling them directly (`torch.tril`, `torch.log`, `torch.clamp`, `F.pad`). `aten.pow.Tensor_Tensor`
> dtype promotion (`float32` base, `int32` exponent) also now completes — confirmed with the exact
> shapes `bloom`'s `build_alibi_tensor` uses. `torch.square` and `torch.repeat_interleave` also now
> run end to end, though **neither has grown its own `aten.*` table entry** — both resolve as
> Python-level composites that decompose into kernels this shim already had, which is why they do
> not appear in `_aten_implemented()` even though the call succeeds. Still genuinely missing,
> confirmed absent from `_aten_implemented()`: `aten.all.default`, `aten.aminmax.default`,
> `aten.index_add_.default`, `aten.nonzero.default`, `aten.scatter_.value`, `aten.zeros.default`,
> `aten.roll.default`.

`TensorBase.__getitem__` with a list index (`bert`, `falcon`) is a kernel-shaped gap too, but it
sits in the `__setitem__`/`__getitem__` region explicitly out of bounds for this session (another
agent is rewriting it now) — not classified further here, and not fixed.

> **Correction (re-verified live against the current build): this is fixed.** `x[:, [-1, 0]]` on a
> 2-D float tensor now returns the correct values through the shim — confirmed by calling it
> directly, not inferred from an architecture passing. Neither `bert` nor `falcon` hits this gap
> anymore (§4's correction above already covers both by name).

**(c) blocked under abi3 the way `torch.compile` is** — one candidate, and it turns out **not** to
belong here:

`gpt_bigcode` fails at import time (`modeling_gpt_bigcode.py:54`, `@torch.jit.script` on a module-level
helper) on `SourceRangeFactory.make_range`. This is the exact wall `docs/graph/DYNAMO.md` §12 already
found and already distinguished from the eval-frame wall: `torch._C._jit_tree_views`'s
`SourceRangeFactory` is TorchScript's own Python-level AST-to-IR frontend
(`torch/csrc/jit/python/python_tree_views.cpp`), and DYNAMO.md §12 confirmed by reading that source
that it carries **no** `Py_BUILD_CORE`/internal headers — unlike `set_eval_frame`, nothing about it
is structurally blocked by the limited API. It is (a)-shaped, not (c)-shaped: a real, sizeable,
buildable-in-principle Python-level IR parser (`Ident`, `Def`, `Return`, `Tuple`, `Call`, `Attribute`
node types and onward, per DYNAMO.md §12's own probe), not a small one, and not attempted this
session. Reused DYNAMO.md's own evidence rather than re-deriving it, per instruction. This gap is
identical on 4.x and 5.x — `gpt_bigcode` still carries the same decorator upstream.

> **Correction (docs/graph/TORCHSCRIPT.md; re-verified live): this import wall is gone on both versions.**
> A later round made this shim default to upstream's own scripting-disabled mode
> (`os.environ.setdefault("PYTORCH_JIT", "0")`, `docs/graph/TORCHSCRIPT.md`), which is exactly the kind of
> off-switch this section's own (c)-vs-(a) analysis was checking for and did not find at the time.
> `from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeForCausalLM` now
> succeeds, and (with `aten.tril.default` also since implemented, `docs/kernels/TRIL.md`) the model
> constructs and forwards on both `transformers` 4.57.6 and 5.15.1. This does not overturn the
> (a)-not-(c) classification argument itself — the reasoning for why `make_range` is buildable
> rather than abi3-blocked was never wrong — it changes only whether GPT-BigCode still needs it,
> which it no longer does.

---

## 6. Which `transformers` version should this project target

**Recommendation: keep claiming 5.x, and say so explicitly rather than implicitly (the README
already started doing this after the finding that opened this investigation).**

Reasons, in order of weight:

1. **`pip install torchnative` resolves nothing for `transformers` — the user's own `pip install
   transformers` does, and today (2026-08-30) that is 5.16.1.** `torchnative`'s own `requires_dist`
   (checked against the published `0.0.4a0` metadata on PyPI) does not mention `transformers` at
   all. A new user typing the README's own `pip install transformers` gets 5.x, not 4.x, unless they
   pin otherwise. The install-time default already matches the claim.
2. **Latest stable 4.x (`4.57.6`) and latest stable 5.x (`5.16.1`) checked against PyPI just now**
   (`pip index versions transformers`) — 4.x is not moving forward; 5.x is where new releases land.
3. **The measured gap is smaller on 5.x and getting smaller in the right direction.** §4: 14/20 vs
   10/20 with identical fixes applied. The 4.x-specific losses (§4's table) are all *older* code
   paths that 5.x already replaced with masking-utils/grouped-GEMM equivalents this shim already
   supports — closing 4.x's gap would mean re-supporting patterns the ecosystem itself is moving away
   from, for architectures that already work under the version going forward.
4. **Against that: a large fraction of deployed model code still pins `transformers<5` or `==4.*`,**
   because 5.0 is a recent major bump. This is where the claim genuinely costs something — a caller
   who has not moved to 5.x yet gets 10/20 architectures today, and needs stub-worthy new kernels
   (`tril`, `all`, the mixtral MoE set) to reach parity, none of which are small.

Net: the ecosystem's own default (`pip install transformers`) already points at 5.x, the newer
version needs less from this shim, and the gap that remains is upstream's own kernel surface, not
this shim's autocast surface — so 5.x is both the honest target and the cheaper one to keep honest.

> **Correction (re-verified live against the current build): the counts in points 3 and 4 are
> stale — 5.x is 20/20 (not 14/20) and 4.x is 16/20 (not 10/20), and `tril` is no longer part of
> the remaining 4.x gap (§4's and §5(b)'s corrections above).** The recommendation itself (target
> 5.x) is a judgment call this correction does not attempt to re-decide — points 1 and 2, about
> what `pip install transformers` resolves to and where new releases land, do not depend on these
> counts and are unaffected. Whoever revisits this recommendation should re-check the counts first,
> since the gap this section describes as motivation has narrowed on both sides.

---

## 7. Tests

Not added directly — `rust/torch_c/pytests/test_shim.py` is out of this session's territory (another
agent's). Two snippets, in that file's own style (plain asserts, `import _C` for the door,
`_upstream_torch` for cross-checking where the fixture already does), to be placed near
`test_autocast_is_off_and_cannot_be_turned_on`:

```python
def test_is_autocast_available_matches_upstream_per_device_and_still_permits_the_noop_path():
    """`torch._C._is_autocast_available` -- docs/models/COMPAT.md §3.1.

    Answering `False` here for every device type was the plausible guess this
    shim's autocast surface never implemented; it is wrong; `True` for the
    device types upstream actually registers an Autocast key for is the
    honest answer, and the test that would have caught the wrong one is the
    second half below: `enabled=False` must complete without raising, on a
    device type this predicate calls available.
    """
    import torch

    for name in ("cpu", "cuda", "xpu", "hpu", "mtia", "maia", "ipu", "xla",
                 "mps", "privateuseone"):
        assert _C._is_autocast_available(name) is True
    for name in ("mkldnn", "opengl", "opencl", "ideep", "hip", "ve", "fpga",
                 "lazy", "vulkan", "meta"):
        assert _C._is_autocast_available(name) is False
    import pytest
    with pytest.raises(RuntimeError):
        _C._is_autocast_available("bogus")
    with pytest.raises(TypeError):
        _C._is_autocast_available(None)

    # the door a real model actually calls through
    with torch.autocast(device_type="cpu", enabled=False):
        pass  # must not raise -- this is the path Mixtral's rotary embedding needs


def test_permute_member_matches_the_door_and_T_is_permute_reversed():
    """`TensorBase.permute`/`Tensor.T` -- docs/models/COMPAT.md §3.2.

    Both were reachable through `torch.permute(x, dims)` before this fix and
    not through `x.permute(dims)` or `x.T` -- the distinction the door and the
    member catch separately, per CLAUDE.md's own note about that class of
    bug. Tested both ways here rather than only the member, so a future
    regression on the door does not slip past this file.
    """
    import torch

    x = torch.arange(24).reshape(2, 3, 4).to(torch.float32)
    via_door = torch.permute(x, (2, 0, 1))
    via_member = x.permute(2, 0, 1)
    assert torch.equal(via_door, via_member)
    assert via_member.shape == (4, 2, 3)

    y = torch.arange(12).reshape(3, 4).to(torch.float32)
    assert torch.equal(y.T, y.permute(1, 0))

    z = torch.arange(5).to(torch.float32)
    assert torch.equal(z.T, z)  # 1-D: permute(0), values unchanged
```

---

## 8. What this session did and did not do

- **Implemented** (all in territory): `torch._C._is_autocast_available` and five companion autocast
  bookkeeping names (`bootstrap.py`), `TensorBase.permute` (`methods.json`), `Tensor.T`
  (`bootstrap.py`).
- **Investigated and classified but did not implement**: 8 missing aten kernels (`tril`, `all`,
  `aminmax`, `index_add_`, `nonzero`, `scatter_.value`, `zeros`, `square`, `repeat_interleave`,
  `log`, `clamp`, `constant_pad_nd`, `roll` — `aten.rs`, out of territory), the `__getitem__`
  list-index gap (forbidden region, another agent's), and `torch.jit.script`'s TorchScript frontend
  (large, not attempted — DYNAMO.md's own evidence says it is not the abi3 wall, just a different,
  bigger one).
- **Measured, not decided**: the four fixes above were verified against the four acceptance
  commands (all still exit 0 — `run.sh` 225/225, `compare.py` 3037/3037 ops=122,
  `compare.py --self-test` exit 0, `verify_schemas.py` 4234/4234, up from 4233 because `permute`
  added one methods-table row) and against the full 20-architecture shim-run sweep before and after.
- **Left for the doc, not the code**: which `transformers` version to claim (§6) — a project-level
  claim, not a code change, and the README already changed once on the finding that opened this
  investigation; a second edit is for whoever reads this to make deliberately.
