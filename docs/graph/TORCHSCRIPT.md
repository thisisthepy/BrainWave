# `torch.jit.script` at module scope — what GPT-BigCode actually needs

**Verdict up front, because it changes the shape of everything below.** GPT-BigCode does not
need a TorchScript frontend. It needs upstream's own scripting-disabled mode, which this shim now
defaults to, plus one missing kernel (`aten.tril`) that is outside this file's territory. The
import wall (docs/graph/DYNAMO.md §12's `SourceRangeFactory.make_range`) is gone. The architecture still
does not forward — it stops one step later, at a small, named, ordinary kernel gap — so the count
stays 19/20, not 20/20. Reporting anything else would be reporting the count on the strength of an
import succeeding, which the brief for this round explicitly rules out.

> **Correction (docs/kernels/TRIL.md; re-verified live): `aten.tril.default` has since been implemented,
> and the count is 20/20, not 19/20.** `docs/kernels/TRIL.md` §0 records the kernel landing (`tril.default`
> added to `aten.rs`, wired in `overloads.json`/`methods.json`) — out of this document's territory,
> as §6 below already anticipated. Confirmed directly against the current build:
> `"aten.tril.default" in torch._C._aten_implemented()` is `True`, and
> `AutoModelForCausalLM.from_config(AutoConfig.for_model("gpt_bigcode", ...)).eval()` constructs
> and forwards without error. (A separate, previously-unnoted gap — `aten.dropout.default` — blocks
> the *training*-mode forward, i.e. with the model not put in `.eval()` first; that is a new,
> unaudited finding, not a claim this document ever made, and is left for whoever picks it up.)
>
> Standing check (docs/verification/DOCWATCH.md):
> <!-- DOCWATCH: op-implemented aten.tril.default -->

---

## 1. Baseline — reproduced, unchanged from the brief

```
$ TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=torchnative/src/main HF_HOME=... \
    python3 /tmp/arch7/sweep.py /tmp/out
...
gpt_bigcode    FAIL ModuleNotFoundError: Could not import module 'GPTBigCodeForCausalLM'. ...
TOTAL 19/20
```

Full traceback (`AutoModelForCausalLM.from_config` → transformers' lazy-module `__getattr__` →
`ModuleNotFoundError` with `__cause__` set):

```
transformers/models/gpt_bigcode/modeling_gpt_bigcode.py:54, in <module>
    @torch.jit.script
torch/jit/_script.py:1498, in script
    ret = _script_impl(...)
torch/jit/_script.py:1255, in _script_impl
    ast = get_jit_def(obj, obj.__name__)
torch/jit/frontend.py:380, in get_jit_def
    return build_def(...)
torch/jit/frontend.py:428, in build_def
    r = ctx.make_range(py_def.lineno, py_def.col_offset, py_def.col_offset + len("def"))
torch_c_bootstrap.py:229, in fn
NotImplementedError: not implemented in torch._C shim: SourceRangeFactory.make_range

The above exception was the direct cause of the following exception:
...
ModuleNotFoundError: Could not import module 'GPTBigCodeForCausalLM'. Are this object's
requirements defined correctly?
```

This is the same wall docs/graph/DYNAMO.md §12 already characterised from a different caller
(`torch.utils.mkldnn.MkldnnLinear`'s `@torch.jit.script_method`, reached through
`torch.compile`'s default `inductor` backend): **not the abi3 wall** (`python_tree_views.cpp` is
ordinary pybind11, no `Py_BUILD_CORE`), just an unattempted TorchScript source frontend. §7 below
extends that document's symbol-level accounting with a probe run directly against GPT-BigCode's
own functions rather than `MkldnnLinear`'s.

---

## 2. The question that decides the shape of the answer

`torch.jit.script` at module scope runs at import time, as a decorator. Upstream ships an off
switch for exactly this — the question was whether it degrades to something usable or to nothing.

**Read directly from the vendored tree** (`torchnative/src/main/torch/jit/_state.py`, byte-identical
to `/Volumes/macMini/caches/spike-venv/.../torch/jit/_state.py`, `diff` exit 0 — this file is
unmodified upstream, torch 2.13.0):

```python
class EnabledProxy:
    def __init__(self) -> None:
        self.enabled = self.parse_env(
            "PYTORCH_JIT", True, "> Using PyTorch JIT", "> PyTorch JIT DISABLED"
        )
    ...
_enabled = EnabledProxy()

def disable() -> None:
    _enabled.enabled = False
```

And `torch/jit/_script.py` (also unmodified, `diff` exit 0):

```python
def script(obj, ...):
    ...
    if not _enabled:
        return obj              # <-- line 1492-1493
    try:
        ...
        ret = _script_impl(...)
        ...

def script_method(fn):
    ...
    if not _enabled:
        return fn                # <-- line 369, before any frame introspection
    _rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=2)
    ast = get_jit_def(fn, fn.__name__, self_name="ScriptModule")
    return ScriptMethodStub(_rcb, ast, fn)
```

**Confirmed by running it, not by reading it**: with `PYTORCH_JIT=0` set before `import torch`,

```
>>> import torch.jit._state as st
>>> bool(st._enabled)
False
>>> def f(x): return x + 1
>>> torch.jit.script(f) is f
True
>>> torch.jit.script(f)(3)
4
```

`torch.jit.script(f) is f` — not a copy, not a wrapper, the exact same function object. This is
upstream's own documented fallback, publicly named (`PYTORCH_JIT` is the env var upstream's own
docs describe as the scripting kill switch), evaluated once at `torch/jit/_state.py` import and
read from every `script`/`script_method` call thereafter. **A shim that reports scripting as
unavailable through this path is not inventing behaviour — it is upstream's own disabled mode,
verbatim, in files this repository does not modify.**

`_enabled` is also read by roughly a dozen other `if _enabled:` blocks in `_script.py` that gate
class/method definitions (`ScriptModule`, the magic-method list, etc.) at **import** time of
`torch.jit` itself — separate from the `script()`/`script_method()` calls above, which gate at
**use** time. Those import-time blocks already run today with `_enabled=True` (the pre-existing
default) for all 19 currently-passing architectures, since none of them touches `torch.jit`
directly; they are unaffected either way and not re-verified here beyond the full test suite
passing.

---

## 3. What it costs — checked, not assumed

The brief's condition for taking this path was: if GPT-BigCode's forward pass genuinely depends on
scripted behaviour, degrading to eager must reproduce the same numbers, checked rather than
asserted. That check turned out to be unnecessary for a stronger reason — **the three functions
GPT-BigCode decorates are dead code**:

```
transformers/models/gpt_bigcode/modeling_gpt_bigcode.py:54  @torch.jit.script def upcast_masked_softmax(...)
transformers/models/gpt_bigcode/modeling_gpt_bigcode.py:65  @torch.jit.script def upcast_softmax(...)
transformers/models/gpt_bigcode/modeling_gpt_bigcode.py:73  @torch.jit.script def masked_softmax(...)
```

`grep -c` for each name elsewhere in the file: **zero** call sites past their own `def`. Current
`GPTBigCodeAttention.forward` (transformers 5.15.1) routes through
`ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)` —
the modern attention-interface registry, added since these three functions were written for an
older "upcast" attention path that this version of the class no longer calls. `git blame` on the
transformers repo was not needed to establish this: the file in front of the shim simply never
calls them, which is what determines behaviour here.

So for GPT-BigCode specifically, scripted-vs-eager is moot: **the decorator's only observable
effect at runtime is at decoration time**, and after `PYTORCH_JIT=0` that effect is "return the
function unchanged," which then sits unused exactly as it would if scripting had succeeded and
also gone unused. There is no eager-vs-scripted numeric comparison to make for these three
functions because neither version ever executes.

(§8 below covers the other transformers modeling files that share this decorator, where — unlike
here — some of the functions *are* on the forward path.)

---

## 4. The fix — one `setdefault`, in territory, both directions tested

`rust/torch_c/src/bootstrap.py`, executed once per `_C` import (before `torch/__init__.py`
reaches `import torch.jit`, since `_C`'s own import is what runs this file):

```python
os.environ.setdefault("PYTORCH_JIT", "0")
```

`setdefault`, not an unconditional set: a caller who exports `PYTORCH_JIT=1` explicitly still
reaches the real (`NotImplementedError`-naming) path. Verified in both directions —
`rust/torch_c/pytests/test_shim.py`:

* `test_torch_jit_script_defaults_to_returning_the_original_function` — subprocess with
  `PYTORCH_JIT` unset, vendored `torch` on `PYTHONPATH`: `torch.jit.script(f) is f`,
  `torch.jit.script_method` likewise, both through the **Python-facing call**, not
  `_C._aten_dispatch` or a dispatch key. (The brief's own warning — "golden compares by dispatch
  key and is structurally blind to a missing spelling" — applies exactly as much to this fix: a
  passing dispatch-key case proves nothing about what a real `@torch.jit.script` decorator does.)
* `test_an_explicit_pytorch_jit_1_is_not_clobbered_by_the_default` — subprocess with
  `PYTORCH_JIT=1` forced: `@torch.jit.script` still raises, and the message still names
  `SourceRangeFactory.make_range` — the refusal-by-name half of the correctness bar, confirmed
  live, not just left alone and assumed unaffected.
* `test_gpt_bigcode_imports_now_that_scripting_defaults_to_unavailable` — subprocess, default env:
  `from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeForCausalLM`
  succeeds. (Import only — §6 covers why this test does not also construct the model.)

All three added at the end of `rust/torch_c/pytests/test_shim.py`; `_main()` picks up every
`test_*` in `globals()`, so no registration step was needed. 245 tests, 0 FAIL (was 242).

---

## 5. Verification — all four gates, before and after

| gate | before this change | after |
|---|---|---|
| `vendor/install_shim.sh` | exit 0 | exit 0 |
| `pytests/run.sh` | 242 ok, 0 FAIL | **245 ok**, 0 FAIL |
| `tools/golden/compare.py` | 3302/3302, ops=133 | 3302/3302, ops=133 (unchanged — no kernel touched) |
| `tools/golden/compare.py --self-test` | 13×11, 0 problems | unchanged |
| `verify_schemas.py` | 4331/4331 | unchanged |
| 20-arch sweep | 19/20, gpt_bigcode fails at **import** | 19/20, gpt_bigcode fails at **construction**, one named kernel |

The golden/schema numbers are unchanged on purpose — nothing in `aten.rs`, `overloads.json`'s
kernel entries, or the dispatch tables moved. `overloads.json` and `methods.json` (both in this
file's territory) were read but not edited: `tril` has no entry in either, and adding an entry
without a kernel behind it would just move the refusal from `bootstrap.py`'s generic
"no table entry" message to a different generic message, which is not a fix and not what
`overloads.json` is for (its job, per `install()`'s own comment, is naming *which overload* a
resolved op reached — there is nothing behind `tril` to resolve to).

Sweep reproduction:

```
export TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=<repo>/torchnative/src/main HF_HOME=...
python3 /tmp/arch7/sweep.py <outdir>
```

---

## 6. What's left for GPT-BigCode — one kernel, measured to be the *only* one

With the frontend wall gone, GPT-BigCode's `__init__` reaches this instead:

```
transformers/models/gpt_bigcode/modeling_gpt_bigcode.py:386, in GPTBigCodeModel.__init__
    self.register_buffer(
        "bias", torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)),
        persistent=False,
    )
torch_c_bootstrap.py:3044, in fn
NotImplementedError: not implemented in torch._C shim: torch.tril(...) -- overload resolution has
no table entry for this op (rust/torch_c/src/overloads.json); call torch.ops.aten.tril.<overload>,
which carries the overload and reaches the same dispatcher
```

Checked, not guessed: `tril` is in `_C._VariableFunctions`' name surface (so `torch.tril` exists
as a *name*) but `"tril" in torch._C._aten_implemented()` is `False`, and neither
`overloads.json` nor `aten.rs`/`tensor.rs` has any occurrence of the string `tril` at all. This is
a genuine missing kernel, not a wiring gap like `torch._grouped_mm`'s (docs/architectures/ARCH20.md's account of
that one: the entry sat in the table and the predicate excluding it was wrong). There is nothing
to wire here — the kernel does not exist.

**This is a finding, not a fix landed here.** `aten.rs`/`tensor.rs` are explicitly out of this
file's territory this round (another agent owns them). Sizing it for whoever picks it up: `tril`
is a shape-preserving elementwise-by-position op (zero everything above the k-th diagonal), the
same size class as `constant_pad_nd` in docs/architectures/ARCH20.md's own classification table (a "kernel
(small)"), not a promotion or dispatch-table change. It is used here with `diagonal=0` (the
default) on a `bool` tensor.

**Confirmed to be the *only* remaining wall for this specific model.** A Python-level-only stand-in
for `torch.tril` (exact integer/boolean lower-triangular masking, computed via `.tolist()` and
plain Python — not a claim about what the real kernel should look like, just enough to keep the
investigation moving past this one call) was applied *outside* the shim, in a throwaway test
harness, purely to see what comes next:

```python
torch.tril = _fake_tril   # investigation-only, not landed anywhere in this repo
```

With that in place, `GPTBigCodeForCausalLM` **both constructs and forwards successfully** — no
further walls. §6.1 has the numeric comparison against upstream.

### 6.1 Numeric comparison against upstream, with `tril` stood in

Two-process comparison (the two `torch`s cannot coexist in one interpreter): upstream
(`spike-venv`'s real torch 2.13.0, no `PYTHONPATH` override) builds a 2-layer GPT-BigCode
(`hidden_size=32, heads=4, kv_heads=2, vocab=100, intermediate=64`), fills weights from a fixed
`manual_seed(1234)`, forwards `[[3, 17, 42, 8, 91, 5]]`, and dumps `state_dict()` (as nested lists)
plus logits/argmax to JSON. The shim side loads that exact state dict via `load_state_dict(...,
strict=True)` (no missing/unexpected keys — the `tril`-built `bias` buffer is `persistent=False`
and never crosses the JSON boundary) and forwards the same ids.

```
upstream argmax:  [3, 17, 42, 8, 67, 5]
shim     argmax:  [3, 17, 42, 8, 67, 5]      -- identical at every position
max abs logit diff:  8.94e-08   (600 logits compared)
max rel logit diff:  3.24e-05
```

Both scripts are in `/tmp/tscript_gen_upstream.py` and `/tmp/tscript_gen_shim.py` (not committed,
same status as this project's other `/tmp` probes). The magnitude (~1e-7 absolute on float32
matmul chains) is the same order the existing llama end-to-end golden test in `test_shim.py`
measures for kernels already landed (`_REAL_LLAMA_ATOL = 5e-7`), not a looser bound invented for
this case.

**This does not mean GPT-BigCode is ready.** The `tril` stand-in above is not `aten.rs` code and
is not proposed as any — it exists only to answer "is `tril` really the only wall," and the answer
is yes, for this configuration and this forward shape. Whoever implements the real kernel should
re-run this comparison against the real implementation, not treat this result as already covering
it.

---

## 7. Sizing the frontend that was *not* built

Not needed for GPT-BigCode (§2-§6 above), but the brief asked for the size regardless, so someone
choosing whether to build it later has a number instead of a feeling. This extends
docs/graph/DYNAMO.md §12, which reached the same first wall from a different caller
(`MkldnnLinear.forward`, via `torch.compile`'s default backend) and stopped at
`ctx.source.encode("utf-8")` failing because the shim's `SourceContext`/`SourceRangeFactory` never
implements `.source` as real string content. Reproduced independently here against GPT-BigCode's
actual `upcast_masked_softmax` (not a trivial stand-in), stubbing one symbol at a time
(`/tmp/tscript_probe_stub.py`, `_probe2.py`, `_probe3.py` — investigation only, nothing landed):

| step | stub in place | next wall | how reached |
|---|---|---|---|
| 0 | none | `SourceRangeFactory.make_range` | same as docs/graph/DYNAMO.md §12, confirmed live against this function |
| 1 | `make_range` → dummy range object | `AttributeError: 'function' object has no attribute 'encode'` on `ctx.source` | **earlier than docs/graph/DYNAMO.md's path**: reached from `build_param` → `build_expr(py_arg.annotation)` on the `x: torch.Tensor` parameter annotation, before the function body is ever visited. `.source` is a pybind11 **property** upstream (`torch/_C/_jit_tree_views.pyi:23`); this shim's catch-all does not distinguish a data-returning property from an unimplemented method, so it hands back a placeholder callable, and `frontend.py:900`'s `ctx.source.encode(...)` fails on the callable rather than a string |
| 2 | + `source` → `property(lambda self: "")` | `NotImplementedError: TreeView.range` | `build_Attribute`'s recursive `build_expr(ctx, expr.value)` on the `torch` half of `torch.Tensor` produces an `Ident`/`Var`, and `base.range()` (a `TreeView` base-class method every one of the 45 tree-view types inherits) is asked for immediately after |

Stopped at step 2 deliberately, same judgment docs/graph/DYNAMO.md §12 already made: continuing does not
change the answer to "how big," it only produces more of the same answer. What's confirmed by
running rather than inferred: two full method families are both needed just to parse a single
type-annotated one-line function signature, before a single statement of the body is visited —
`SourceRangeFactory`'s three behavioural members (`make_range`, `make_raw_range`, `source`, of
which only `source` is a property and the shim does not currently tell that apart from a method),
and `TreeView.range()`, inherited by all 45 tree-view types the surface already stubs the
constructors for (`_C._jit_tree_views`'s `types` table: `Apply`, `Assign`, `Attribute`, `BinOp`,
`Def`, `Ident`, `Return`, `Tuple` via `TupleLiteral`, `Call` via `Apply`, and 36 more — every one
currently has `__init__` only, no query methods).

None of those 45 types' actual behaviour is implemented — `install()`'s type-building policy in
`bootstrap.py` deliberately gives every `_jit_tree_views` type a constructible-but-inert stand-in
(comment: "the stubs are incomplete for them... nothing here computes"). Building real parsing
means giving each of them real range/name/child accessors, then building the statement/expression
dispatch tables `frontend.py` already has Python-side logic for calling into, then (past parsing)
the actual TorchScript type system, `CompilationUnit`, and IR execution needed to make a scripted
function *run* rather than merely parse. docs/graph/DYNAMO.md §12 already measured the scale that all of
this sits inside: upstream's `torch/csrc/jit/` is **213,000 lines of C++** (`wc -l`, that document's
own measurement). Nothing found here changes that number — GPT-BigCode's own function needed
*more* surface reached before body-parsing even starts (parameter annotations), not less, so if
anything this confirms the estimate rather than narrowing it.

**Conclusion, matching the brief's instruction:** do not build this. It would be a second compiler
project, not a kernel or a wiring fix, and §2-§6 show it is not needed for this architecture.

---

## 8. Other architectures and libraries silently affected

Grepped `transformers` 5.15.1's installed tree (the version this repo's sweep uses) for
module-scope `@torch.jit.script`, beyond `gpt_bigcode`:

| model file | scripted functions | called elsewhere in the file? |
|---|---|---|
| `sew_d/modeling_sew_d.py` | `c2p_dynamic_expand`, `p2c_dynamic_expand`, `pos_dynamic_expand` | **no** — dead, same shape as gpt_bigcode |
| `sam3_video/modeling_sam3_video.py` | `fast_diag_box_iou` | **no** — dead |
| `vits/modeling_vits.py` | `fused_add_tanh_sigmoid_multiply` | yes — 1 call site |
| `zoedepth/modeling_zoedepth.py` | `inv_attractor` | yes — 4 call sites |
| `deberta/modeling_deberta.py` | `build_relative_position`, `c2p_dynamic_expand`, `p2c_dynamic_expand`, `pos_dynamic_expand`, `scaled_size_sqrt`, `build_rpos`, `compute_attention_span`, `uneven_size_corrected` | yes — all called (`build_relative_position` ×4) |
| `deberta_v2/modeling_deberta_v2.py` | `make_log_bucket_position`, `c2p_dynamic_expand`, `p2c_dynamic_expand`, `pos_dynamic_expand`, `scaled_size_sqrt`, `build_rpos` | yes — all called |

None of `sew_d`, `sam3_video`, `vits`, `deberta`, `deberta_v2`, `zoedepth` are in the 20-architecture
sweep, so this wall was invisible outside `gpt_bigcode` before this investigation — the sweep's
20 happened not to include any of them. `deberta`/`deberta_v2` are the ones worth flagging loudest:
unlike GPT-BigCode's dead functions, their scripted helpers sit directly on the forward path
(relative-position bucketing inside attention), so importing them was previously impossible at all,
and now runs **eagerly instead of scripted** — correct per §2's upstream-disabled-mode argument
(same Python function, same semantics, no operator fusion), but not numerically re-verified here;
that was explicitly out of scope for this round (the brief names GPT-BigCode as "the one left" of
twenty, and none of these six are in that twenty).

Also affected, inside the vendored tree itself rather than `transformers`: eight files under
`torchnative/src/main/torch/distributed/optim/` (`functional_sgd.py`, `functional_adamw.py`,
`functional_adam.py`, `functional_adagrad.py`, `functional_adadelta.py`, `functional_adamax.py`,
`functional_rmsprop.py`, `functional_rprop.py`) each have a module-scope `@torch.jit.script` on
their optimizer step function, plus `torch/distributed/optim/optimizer.py:104`. `torch.distributed`
is one of the subsystems this shim answers "yes" to (`ANSWERED_PROBES = {"_c10d_init"}` in
`bootstrap.py`), so `import torch.distributed.optim` would have hit this same wall; it now imports
under the default. Not exercised by any test in this repository either way — noted for completeness,
not verified further. The remaining matches are all under `torch/testing/_internal/distributed/rpc/`
— test-only code, not a production import path.

---

## 9. What was and was not done

**Done, in territory:**
* `rust/torch_c/src/bootstrap.py` — `os.environ.setdefault("PYTORCH_JIT", "0")`, one line, with the
  reasoning inline.
* `rust/torch_c/pytests/test_shim.py` — three new tests, all through the Python-facing
  `torch.jit.script`/`import` path, covering the default, the explicit-override, and GPT-BigCode's
  import specifically.
* This document.

**Not done, and why:**
* `aten.tril` kernel — §6, a genuine missing kernel, `aten.rs`/`tensor.rs` are forbidden territory
  this round. Reported, not implemented.
* Numeric re-verification of `deberta`/`deberta_v2`/`vits`/`zoedepth` under the new default — §8,
  out of this round's scope (GPT-BigCode is the named target), flagged for whoever picks up those
  architectures.
* A TorchScript frontend — §7, deliberately not built; sized instead.

**Count:** still 19/20 *as of this round*. GPT-BigCode moved from failing at *import* (a wall with
no name a caller could act on beyond "the frontend doesn't exist") to failing at *construction* on
one named, ordinary, appropriately-sized kernel gap. That is real progress and it is not twenty of
twenty — this document does not claim the second number.

> **Correction: it is now 20/20** — see the correction after the top verdict. `tril` landed in a
> later round (`docs/kernels/TRIL.md`) and closed exactly the gap named above.
