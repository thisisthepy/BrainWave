# `tril`, the names that had kernels, and the third dropped NaN

Three jobs. They arrived as three and stayed three, but two of them turned out to be the same
question asked from opposite ends — *what is reachable from Python* — and the third is the same
predicate this repository has now repaired four times.

---

## 0. The numbers

| gate | before | after |
|---|---:|---:|
| architectures forwarding (of 20) | 19 | **20** |
| `pytests/run.sh` | 249 ok | **253 ok** |
| `tools/golden/compare.py` | 3422/3422, ops=134 | **4284/4284, ops=139** |
| `compare.py` pending case builders | 2 | **1** |
| `verify_schemas.py` | 4334/4334 | **4353/4353** |
| `cargo test --release` | 18 | 18 |
| SmolLM2-135M prefill, `f32` and `bf16` | — | **unchanged, §6.4** |

Split by kind, because the kinds are not interchangeable — a promotion, a defect fix and a doc
correction are not the same thing arriving in the same column:

| | |
|---|---|
| new aten kernels | **4** — `tril.default`, `triu.default`, `min.dim`, `min.other` |
| defects fixed in existing kernels | **3** — `max.dim` (values *and* indices), `max.other`, `argmax` |
| names wired to kernels that already existed | **4** — `torch.amax`, `Tensor.amax`, `torch.softmax`, `torch._safe_softmax` |
| names wired to kernels added here | **4** — `torch.tril`/`Tensor.tril`, `torch.triu`/`Tensor.triu` |
| ops promoted out of `IMPLEMENTED_AWAITING_GOLDEN` | **1** — `max.other` |
| stale refusals / notes corrected | **4** — two SDPA refusals in `bootstrap.py`, the `min` notes in both spelling tables |
| tests updated because they detected this work | **6** — §6.2 |
| new smoke tests | **4** |
| golden cases added | **+862** |
| deletions | **0** |

Two of the three jobs turned out to be the same question — *what is reachable from a real `import
torch`* — and the audit §2.3 asked for found a fourth instance of it plus a refusal that had been
naming a present kernel as missing. The third job is one predicate in candle, repaired for the
fourth time and this time in one shared function rather than a fourth private copy.

---

## 1. `torch.tril` — the twentieth architecture

### 1.1 The schema, read rather than remembered

The brief said to read upstream's schema instead of recalling it, and the vendored tree has it:
`torchnative/src/main/torchgen/packaged/ATen/native/native_functions.yaml:8722`.

```yaml
- func: tril(Tensor self, SymInt diagonal=0) -> Tensor
  structured_delegate: tril.out
- func: triu(Tensor self, SymInt diagonal=0) -> Tensor
  structured_delegate: triu.out
```

Both are `structured_delegate`s onto an `.out` form whose CPU kernel is `tril_cpu`/`triu_cpu`;
neither is `CompositeImplicitAutograd`, so unlike `softmax` in §2 the dispatch key really is the
name. Confirmed from the other side with a `TorchDispatchMode` logger on torch 2.13.0 —
`torch.tril(x)` and `x.tril()` both fire exactly one record, `aten.tril.default`.

**The sign convention was measured, not derived**, because it is the one thing here that fails
silently when it is backwards — a transposed answer is the same shape, the same dtype and the
same magnitude:

```
              diagonal:  -2      -1       0       1       2
tril keeps j - i <=       ▝▖      ▝▖▖     ▝▖▖▖    ▝▖▖▖▖   all
triu keeps j - i >=      all     ▘▘▘▘     ▘▘▘     ▘▘      ▘
```

Read as a table, on `[[1,2,3],[4,5,6],[7,8,9]]`:

| `diagonal` | `tril` | `triu` |
|---:|---|---|
| `-1` | `[[0,0,0],[4,0,0],[7,8,0]]` | `[[1,2,3],[4,5,6],[0,8,9]]` |
| `0` | `[[1,0,0],[4,5,0],[7,8,9]]` | `[[1,2,3],[0,5,6],[0,0,9]]` |
| `1` | `[[1,2,0],[4,5,6],[7,8,9]]` | `[[0,2,3],[0,0,6],[0,0,0]]` |

A positive `diagonal` moves the boundary up and to the right for *both*, so it widens `tril` and
narrows `triu`. The offset is unbounded in both directions — `tril(x, 100)` is `x`, `tril(x, -100)`
is all zeros — so it is compared, never range-checked.

### 1.2 Zeroing is a select, not a multiply by a mask

The obvious implementation is a 0/1 mask of the input's dtype and a broadcast multiply. It is
wrong, and wrong in the direction that survives every test written with small integers:

```
nan * 0  ==  nan          inf * 0  ==  nan          -inf * 0  ==  nan
```

Upstream zeroes those positions like any other. Measured:

```
tril([[1., nan], [inf, -inf]])   ->  [[1., 0.], [inf, -inf]]
triu([[1., nan], [inf, -inf]])   ->  [[1., nan], [0., -inf]]
```

so the masked-out `nan` becomes `0.`, not `nan`. The kernel uses `where_cond` — a select — and the
golden cases carry the `nan`/`inf` matrix specifically so a multiply cannot pass them.

### 1.3 What the kernel is

`aten.rs::tril_triu`, one function for both, parameterised by a two-arm `Triangle` enum for the
same reason `extremum_default` takes an `Extremum`: the two differ in one comparison operator, and
a copy is a second place for the convention to drift.

* Rank is checked first, with upstream's own wording — `tril: input tensor must have at least 2
  dimensions`, which upstream raises for both 1-D and 0-D input.
* Leading dimensions are a batch: the mask is built once at `(rows, cols)` and `broadcast_as` the
  full shape, so a `(2, 3, 4)` input gets the same `(3, 4)` mask on both matrices.
* The input is made contiguous first — **and that is defensive, not load-bearing.** Removing it
  was injected as a fault and *no test failed* (§5, fault 3): candle's `WCond` matches on
  `contiguous_offsets()` and falls back to `strided_index()` for all three operands, so a
  transposed `on_true` is already read by position-in-the-matrix. Kept anyway, because the mask is
  built row-major and handed to `broadcast_as`, and without the normalisation the kernel's
  correctness would rest on an internal detail of candle's cpu backend rather than on anything
  this function establishes. The claim in the code says exactly this now; the first draft claimed
  the normalisation was doing work, which the sabotage disproved.
* Every dtype passes through unchanged, `torch.bool` included, which is the call GPT-BigCode
  actually makes.
* A zero-extent input returns its own shape before the mask is built.

`triu` was not added on the grounds that it is free. It was added after checking that
`hasattr(torch, 'triu')` and `hasattr(torch.Tensor, 'triu')` are both `True` — the test that kept
`gelu`/`silu`/`softplus` out of `overloads.json` in docs/bindings/SPELLINGS.md §7.1. Nothing in the twenty
architectures calls it; it is listed here as a name upstream has and this shim now answers, not as
a requirement anything raised.


### 1.4 GPT-BigCode, with the real kernel

`GPTBigCodeModel.__init__` registers its causal-mask buffer as
`torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool))`, and that call was
the model's last wall once docs/graph/TORCHSCRIPT.md §4 removed the `@torch.jit.script` import wall by
taking upstream's own `PYTORCH_JIT=0` path.

**The twenty-architecture sweep is 20/20.** Same method as docs/architectures/ARCH20.md §1 — 2-layer, hidden 32,
4 heads, 2 kv heads, vocab 100, toy `AutoConfig.for_model`, a 4-token forward — re-run against the
built artefact:

```
llama gpt2 qwen2 mistral gemma gpt_neox opt mpt starcoder2 stablelm
olmo phi mixtral bert bloom cohere falcon mamba persimmon      PASS
gpt_bigcode                                                    PASS  <- was the 20th
TOTAL 20/20
```

docs/graph/TORCHSCRIPT.md §6.1 established the numeric result with a Python-level `tril` stand-in
monkeypatched over the refusal, and explicitly said the real kernel had to be re-measured rather
than inheriting that result. Re-measured, two processes (the two `torch`s cannot share an
interpreter), upstream generating the state dict and the shim loading it with `strict=True`:

```
upstream argmax:  [3, 17, 42, 8, 67, 5]
shim     argmax:  [3, 17, 42, 8, 67, 5]     -- identical at every position
max abs logit diff:  8.94e-08   (600 logits)
max rel logit diff:  3.24e-05
```

**Identical to the stand-in's numbers, to every digit reported.** That is the check that matters:
the stand-in was exact integer/boolean masking done in Python, so a real kernel that agrees with it
bit-for-bit is doing the same masking and no more. 8.94e-08 is the ordinary `float32` matmul-chain
residual — the existing llama end-to-end test in `test_shim.py` uses `_REAL_LLAMA_ATOL = 5e-7`, so
this is inside a bound that predates it rather than one invented for it.

The shim-side script asserts, before it builds anything, that `torch.tril` is this shim's own
(`torch.jit.script(f) is f` through `bootstrap.py`'s `setdefault`, and `tril(ones(3,3,bool))`
against a hand-written lower triangle). The stand-in cannot come back silently.

---

## 2. Spellings: three kernels that had no name, and one that had a wrong reason

### 2.1 `torch.amax` / `Tensor.amax` — the notification worked

The kernel landed in docs/numerics/SEQLEN.md §7 and is why the attention path got 35% faster. Neither Python
name resolved. The golden harness had been comparing `aten.amax.default` against upstream with 120
cases the entire time — **it dispatches by key and is structurally blind to a missing name.**

What caught it was a test the same round wrote *for that purpose*:
`test_amax_has_no_python_spelling_yet_and_says_so_by_name`, asserting the refusal and saying in its
own docstring "when the table entry lands, this test fails -- which is the notification wanted, not
a nuisance." It failed on the first run after the entries went in. It is now
`test_amax_now_has_both_python_spellings_and_they_reach_the_kernel`, and it checks three routes to
one kernel plus the arguments a wrong signature would drop (`keepdim`, negative `dim`, the `dim=[]`
default, both keyword spellings).

```
overloads.json   "amax": ["aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor"]
methods.json     same
```

### 2.2 `torch.softmax` — a table entry would have been the wrong fix

`torch.softmax(x, dim=1)` refused with "no table entry". The tempting fix is an `overloads.json`
entry, and it is wrong in a way that would have *validated*: the parser-level key is
`aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None)`, a real schema that
`verify_schemas.py` would have accepted — but it is `CompositeImplicitAutograd` and never reaches a
kernel. Re-measured with a `TorchDispatchMode` logger on 2.13.0 rather than taken from the earlier
round's note:

```
torch.softmax(x, dim=1)                      -> aten._softmax.default
torch.softmax(x, dim=1, dtype=torch.float64) -> aten._to_copy.default, then aten._softmax.default
x.softmax(1)                                 -> aten._softmax.default
F.softmax(x, dim=1)                          -> aten._softmax.default
```

`aten.softmax.int` never fires, for any of the four. So the fix is a Python-level composite in
`bootstrap.py::_install_composites`, **bound to `TensorBase.softmax`** — one function, so the free
spelling and the member cannot disagree about `dtype=` handling or `half_to_float`. This is the
same treatment `flatten` gets six lines above it, and the reason `methods.json`'s README already
keeps `softmax` out of that table.

### 2.3 The audit found one more, and a stale refusal behind it

The brief asked for an audit of the same class. It was run against the **built artefact**
(`torch._C._aten_implemented()`, 139 keys) rather than against the tables, and every base name was
probed by *calling* it in a live shim session, distinguishing three outcomes: computes, refuses
naming the shim, or raises `TypeError` because the probe's arguments were wrong. That third bucket
matters — a bad argument guess reported as a missing spelling is a false finding, and there were a
dozen of them.

One genuine gap:

| name | in shim | upstream | verdict |
|---|---|---|---|
| `torch._safe_softmax` | refused, "no table entry" | `hasattr` is **True**, fires `aten._safe_softmax.default` | **real gap, now closed** |

**The leading underscore is why it was missed, and the miss had a cost.** docs/architectures/ARCH20.md §9's
inventory filed `_safe_softmax` under "no such public function upstream, or reached another way",
and both halves are false: upstream has the name, and it is a *leaf* op — not a composite — so
unlike `softmax` above the table entry is exactly the right fix. This shim has had the kernel,
golden-compared, since docs/kernels/SDPA.md.

Meanwhile **two refusals in `bootstrap.py::scaled_dot_product_attention` named
`aten._safe_softmax.default` as a kernel that did not exist.** Verbatim, before this change:

```python
f"...on a {query.dim()}-D query -- upstream drops to the math backend for anything
 but 4-D {B, H, T, K}, which needs aten._safe_softmax.default; it has no kernel"
```

It has one. This is the third stale refusal this repository has found in the same function — the
bool-mask branch twenty lines below already carries a note about being the first — and the brief
named the pattern in advance ("a refusal message went stale for weeks: it named two kernels it was
waiting on, both landed, and an architecture stayed blocked on a wall that was gone").

Both messages are corrected to the reason that is actually true. **The refusals stay**: every
kernel the math backend needs (`_safe_softmax`, `mul.Scalar`, `expand`, `view`, `bmm`) is
implemented, and what is missing is the *composite* that sequences them, which nobody has
transcribed. That is a real reason to refuse and a different one. Building the math backend was not
attempted and is not proposed here.

`test_the_two_stale_sdpa_refusals_no_longer_claim_a_missing_kernel` asserts the **claim**, not the
wording: every kernel a refusal names as present must be in `_aten_implemented()`, every one it
names as absent must not be. It fails the day either drifts.

### 2.4 What was left, and why

| name | left because |
|---|---|
| `torch.amin` / `Tensor.amin` | **no kernel.** `aten.amin.default` is not implemented; an entry would resolve and then refuse. `amax`'s `CustomOp1` is direction-specific (`MaxScalar::greater`), so this is a real kernel task, not a sign flip in a table. Named here so it lands on the next queue. |
| `torch.gelu` / `silu` / `softplus` | `hasattr(torch, 'gelu')` is **False** upstream. Adding these would invent a spelling — docs/bindings/SPELLINGS.md §7.1's finding, re-verified. |
| `clamp.Tensor` | listed in both tables with no kernel, deliberately, so it refuses by the right name. Unchanged. |
| `_local_scalar_dense`, `_to_copy`, `_unsafe_view`, `alias`, `lift_fresh`, `slice` | no `torch.<name>` upstream either; all reachable through the member or operator that upstream itself routes through (`.item()`, `.float()`, `x[0:1]`, …). |
| `_scaled_dot_product_flash_attention_for_cpu` | the flat name exists upstream and refuses here, but the kernel is reachable through the correct public spelling `F.scaled_dot_product_attention` for the 4-D `dropout_p == 0` path. Not a functional gap; ARCH20 §9's justification for it was imprecise, its verdict was right. |

**What that audit cannot find:** it checks whether a call *resolves and returns*, not whether the
returned tensor is right. A `max.other`-style silent miscompute is invisible to it — that needs the
golden harness. Which is the mirror image of the harness's own blindness, and the reason both had
to run.

---

## 3. One predicate, four repairs — and the mechanism question, answered

### 3.1 What was wrong

candle folds every reduction and every elementwise comparison with `|x, y| x < y`. Every
comparison against a NaN is false, so a NaN the accumulator does not *start* on is never selected.
One fact, and this repository has now found four separate wrong answers from it:

| found in | op | what was dropped |
|---|---|---|
| docs/models/E2E_REAL.md | `max.default` / `min.default` | the value |
| docs/bindings/SPELLINGS.md §7.2 | `max.other` | a NaN in the **second** operand only |
| docs/numerics/SEQLEN.md §7.2 | `amax` | avoided by construction — never shipped wrong |
| **here** | `max.dim` | **the value *and* the index** |
| **here** | `argmax` | the index |

Measured against the artefact before the fix:

```
max.dim([[1., nan, 3.]], dim=1)   here: (3.0, 2)      upstream: (nan, 1)
argmax([[1., nan, 3.]], dim=1)    here: 2             upstream: 1
max.other([1,nan,3], [5,2,nan])   here: [5, nan, 3]   upstream: [5, nan, nan]
min.dim, min.other                no kernel at all
```

**`max.dim`'s two halves were wrong *consistently*.** `values == input[indices]` held — 3.0 really
is at index 2 — so any self-consistency check passes. Only comparison against upstream catches it,
which is the whole argument for the golden harness and the reason a "does the pair agree with
itself" test would have been worthless.

### 3.2 Can they route through `amax`'s `CustomOp1`? No, and the reason is structural

The brief asked this directly. The answer is no for every member of the family, and it is not a
preference:

* **`max.dim` / `min.dim` / `argmax` need the index.** `amax` is fast precisely *because* it drops
  the index — that is what lets sixteen accumulator lanes run without a loop-carried
  compare-and-select (docs/numerics/SEQLEN.md §7.3). These ops must run `cpu_backend::ReduceIndex` whatever
  else they do, so routing their *values* through `amax` as well would **add** a pass rather than
  remove one. The performance argument runs backwards here.
* **`max.other` / `min.other` are binary.** `CustomOp1` is unary; `CustomOp2` would apply, but
  candle's `broadcast_maximum` is already a vectorised elementwise op with no reduction in it, so
  there is nothing to win.

**What transfers is the rule, not the kernel.** The rule costs two vectorised passes — `ne`, then
one reduction over a 0/1 mask — which is the same shape of correction `extremum_default` has been
paying since docs/models/E2E_REAL.md. So instead of a fifth hand-rolled repair, every reduction in the
family now asks one function:

```rust
fn nan_along_dim(op, source, dim, tag) -> PyResult<Option<(Tensor, Tensor)>>
```

returning `(any, first)` with the dimension kept, or `None` when there is nothing to correct.
`max.dim`, `min.dim` and `argmax` all consume it; `extremum_other` uses the same `ne`-and-count
shape over the broadcast join.

Two measured facts hold it together, read off torch 2.13.0 rather than reasoned about:

* `max(dim=)` and `min(dim=)` report the index of the **first NaN in the slice**, not of the
  extremum among the non-NaN elements: `max([1., nan, nan], dim=0)` is `(nan, 1)`.
* `argmax`/`argmin` report that same index.

So "the first NaN" is the only position any of them needs — and `argmax` over the 0/1 NaN mask *is*
that position, because candle's own reduction keeps the first of two equal elements. That is the
one respect in which its fold is exactly right, and it does the work.

### 3.3 `None` when there is no NaN, and why that is not just an optimisation

`nan_along_dim` returns `None` for a non-floating dtype *and* for a float tensor that happens to
hold no NaN (one `sum_all` over the mask decides). The second case keeps a NaN-free reduction
bit-for-bit on the path it already took, so **the prefill hash cannot move because of a correction
that never applies** — §6.4 confirms it did not.

### 3.4 What was fixed, and what could not be

| op | before | after |
|---|---|---|
| `max.dim` | value and index both dropped the NaN | fixed, shares `extremum_dim` |
| `min.dim` | **no kernel** | new, same function |
| `max.other` | second operand's NaN dropped | fixed, shares `extremum_other` |
| `min.other` | **no kernel** | new, same function |
| `argmax` | index dropped the NaN | fixed, same `nan_along_dim` |
| `max.default` / `min.default` | already correct (docs/models/E2E_REAL.md) | NaN-position cases added |
| `amax` | already correct by construction | unchanged |
| `amin` | **no kernel, and none added** | see §2.4 — a direction-specific `CustomOp1`, a real task |
| `argmin` | no kernel | not in `_aten_implemented()`; nothing to fix |

`min.dim` and `min.other` were not speculative additions. docs/bindings/SPELLINGS.md §7.2 put both in
`overloads.json` and `methods.json` **with no kernel behind either**, deliberately, so
`torch.min(x, dim=0)` would refuse by the right name and land on the next owner's queue as a
precise work item. This is that owner. Written as one function with the `max` side rather than
copied, so a fifth version of the NaN rule cannot drift away from the fourth.

One detail that a shared implementation nearly lost: `min.dim`'s result must print as `min(...)`,
not `max(...)`. Upstream's is a `torch.return_types.min` structseq; this shim's is a
`collections.namedtuple`, and sharing one cached type between the overloads — the obvious economy,
since the field names are identical — would make every `min` result claim to be a `max` in every
`repr` and traceback. Two `OnceLock`s, and a test.

### 3.5 The `-inf` boundary

The tempting shortcut is to key the correction on "not finite" rather than on `x != x`. It passes
every NaN case and breaks a fully masked attention row, whose maximum is `-inf` and must stay
`-inf`. Fault 7 in §5 is exactly that mistake; twelve `inf`-bearing cases catch it.

---

## 4. Where the NaN cases went, and the position rule

The brief's instruction — "where you fix one, add the NaN-position cases: first, middle, last,
because a case with the NaN in the first position passes under this bug" — is the single most
load-bearing sentence in this round, and §5 proves it numerically.

**A NaN in element 0 seeds candle's accumulator and nothing displaces it.** So a kernel with *no
NaN handling whatsoever* answers `at=0` correctly. A suite built only from `at=0` cases cannot
fail. docs/numerics/SEQLEN.md §7.12 recorded the same hole in `amax`'s first test; this round found it
again, in the numbers rather than in the reasoning.

Added, per op that returns something a NaN can reach:

| builder | positions | extras |
|---|---|---|
| `argmax_cases` | first / middle / last × 4 float dtypes × 3 `(dim, keepdim)` forms | two NaNs (tie-break), NaN in one row of two |
| `max_dim_cases` / `min_dim_cases` (shared) | first / middle / last × 4 dtypes × 2 keepdims | two NaNs, one-row-only, strided slice, three `inf` cases |
| `max_other_cases` / `min_other_cases` (shared) | NaN in each operand at each position, separately | 0-d broadcast both ways, all-NaN operand, `inf` with no NaN |
| `max_default_cases` / `min_default_cases` | first / middle / last × 4 dtypes | (was one middle-position case each) |

and one smoke test, `test_the_whole_max_min_family_agrees_on_one_nan_rule`, that walks all six ops
through all three positions in one table — written as one test rather than six because what keeps
going wrong is not any single kernel, it is that a *new* member of the family gets written against
candle's primitive and inherits the fault silently. A seventh op added to that table with no NaN
handling fails there.

---

## 5. Sabotage — eleven faults, and one that could not fail

Every fault was injected into the source, rebuilt, installed, and run through the real gates.
`cp` backups, never `git checkout`. The restored tree was verified by md5 against the backups and
the final artefact was checked with `strings ... | grep -c SABOTAGE` → **0**.

| # | fault | smoke | golden |
|---|---|---:|---:|
| 1 | `tril`/`triu` comparison operators swapped | **2 FAIL** | **496 FAIL** |
| 2 | zero by multiplying a 0/1 mask instead of selecting | **1 FAIL** | **12 FAIL** |
| 3 | drop the input's `.contiguous()` | 0 | 0 |
| 4 | `nan_along_dim` always returns `None` (the pre-fix state) | **2 FAIL** | **68 FAIL** |
| 5 | correct the values, leave the indices uncorrected (a half-fix) | **2 FAIL** | **40 FAIL** |
| 6 | `max`/`min.other` mask by the left operand only (the original bug) | **1 FAIL** | **48 FAIL** |
| 7 | key the correction on `!is_finite` instead of `x != x` | **2 FAIL** | **80 FAIL** |
| 8 | delete `amax` from both spelling tables | **3 FAIL** | **0** |
| 9 | do not install the `torch.softmax` composite | **1 FAIL** | not run |
| 10 | delete `tril` from both spelling tables | **2 FAIL** | not run |
| 11 | restore the stale `_safe_softmax` refusal text | **1 FAIL** | not run |

### 5.1 Fault 3 could not fail, and that is the finding

**Deleting the kernel's `.contiguous()` broke nothing.** All 253 smoke tests and all 4284 golden
cases stayed green.

Chased rather than shrugged at. candle 0.11.0's `WCond`
(`candle-core/src/cpu_backend/mod.rs:85`) matches on `contiguous_offsets()` for all three operands
and falls through to `strided_index()` otherwise — so a transposed `on_true` is already read by
position-in-the-matrix. Confirmed live under the faulted build: `tril(x.t())`,
`tril(z.transpose(1, 2))` and `tril(z[:, :, 1:3])` all answered correctly with the normalisation
gone.

Two things changed as a result, and neither is "add a test":

* **The code's claim was wrong and is corrected.** It said a transposed view "would otherwise be
  masked by position-in-memory". It would not. It now says the normalisation is defensive, says
  what the sabotage showed, and gives the actual reason for keeping it (not resting the kernel's
  correctness on an internal detail of candle's backend).
* **The smoke assertion that covers the shape is labelled as unable to fail**, at the assertion,
  so a future reader does not mistake a green run for evidence that the normalisation is doing
  work.

Recording it as coverage-without-a-check is the honest outcome. Writing a test that "covers" a
line no fault can perturb would have been the dishonest one.

### 5.2 Fault 4 is the position rule, in numbers

68 golden failures, broken down by the position of the NaN:

```
in the last position    28
in the middle position  28
in the first position    0        <-- every at=0 case PASSED under the fault
```

**Zero.** Every single `at=0` case passes a kernel with the NaN handling entirely removed. If the
suite had been written the obvious way — one NaN case per op, NaN first — it would have been 4284
green against a completely broken correction.

### 5.3 Fault 8 is the harness's blindness, in numbers

Deleting `amax` from both spelling tables: **golden stayed 4284/4284, zero failures**, while three
smoke tests failed. The kernel is untouched and golden dispatches by key, so it cannot see that
`torch.amax` has stopped existing. That is why §2's fixes are pinned by tests that go through the
Python name and the tensor member, not by cases.

### 5.4 Fault 2 is the narrowest catch

12 of 4284 cases — the `nan`/`inf` matrix, 2 ops × 2 dtypes × 3 diagonals — plus one smoke test.
Nothing else in the suite distinguishes a select from a multiply, because every other case is built
from small integers where `x * 0` and `select(false)` agree. Those twelve cases exist only because
§1.2 asked what a plausible wrong implementation would be *before* the cases were written.

---

## 6. Verification

### 6.1 The gates, all exit 0

```
bash vendor/install_shim.sh                     exit 0
PYTHON=$PY sh rust/torch_c/pytests/run.sh       253 ok, 0 FAIL            exit 0
$PY tools/golden/compare.py                     4284/4284, ops=139        exit 0
$PY tools/golden/compare.py --self-test         13 x 11, 0 problems       exit 0
$PY rust/torch_c/pytests/verify_schemas.py      4353/4353                 exit 0
cargo test --release                            18 passed                 exit 0
```

| gate | before | after |
|---|---:|---:|
| architectures forwarding | 19/20 | **20/20** |
| `run.sh` | 249 ok | **253 ok** (+4) |
| `compare.py` cases | 3422 | **4284** (+862) |
| `compare.py` ops covered | 134 | **139** (+5) |
| `compare.py` pending builders | 2 | **1** |
| `verify_schemas.py` | 4334 | **4353** (+19) |
| `cargo test` | 18 | 18 |

**`pending` went 2 → 1**, which the brief expected to stay at 2. `aten.max.other` was promoted out
of `IMPLEMENTED_AWAITING_GOLDEN` into `_aten_implemented()`: its case builder existed
(docs/bindings/SPELLINGS.md §7.3) and was holding a deliberately failing NaN case until the kernel could be
fixed. Fixing it and promoting it in the same change is what closes that loop.
`aten.reshape.default` is the one still pending, unchanged.

+5 ops covered: `tril.default`, `triu.default`, `min.dim`, `min.other` (new kernels) and
`max.other` (promoted). +19 schema entries: `amax`/`tril`/`triu` in both tables and `_safe_softmax`
in one, at ~3 checks each.

### 6.2 The four smoke tests that failed on purpose, and were updated

Not counted as regressions; each was a notification some earlier round installed deliberately:

| test | what it detected |
|---|---|
| `test_amax_has_no_python_spelling_yet_and_says_so_by_name` | the entry landed — renamed and inverted (§2.1) |
| `test_amax_propagates_nan_where_candles_own_reduction_drops_it` | its `max.dim` assertion asked "is amax's NaN pass now redundant?" — answered **no**, in the test (§6.3) |
| `test_spelling_road_through_the_vendored_tree` | `min.other`/`min.dim` stopped refusing |
| `test_grouped_mm_resolves_from_the_torch_level_name` | "if a second underscore-prefixed op is added later, that assertion is where it announces itself" — `_safe_softmax` was |

Plus two count assertions, both re-derived rather than bumped: `tag_core_count` 83 → **84** (only
`min.dim` of the five new keys is core-tagged upstream — read off each op's own `.tags`, not
inferred; `max.dim`, implemented by the same function, is *not* core), and distinct schema
identities 217 → **221** (+3 for `amax`/`tril`/`triu`, which are in both tables and so are one
identity each not two, +1 for `_safe_softmax`, which upstream has no member for).

### 6.3 Is `amax`'s own NaN pass now redundant?

The test that failed asked this, so it is answered rather than assumed. **No.**
`max.dim`'s correction lives in `aten.rs`, above the aten dispatch boundary. `sdpa_flash_cpu` calls
`crate::tensor::amax_keepdim` *directly* in Rust and never crosses that boundary, so an aten-level
correction does not reach it. candle's own reduction is unchanged and still drops the NaN — that
statement now lives where it can be made against candle directly rather than through an aten op
that has stopped exhibiting it: `tensor.rs::candle_drops_the_nan_this_kernel_keeps`, a `cargo test`.

### 6.4 SmolLM2-135M prefill — unchanged, both dtypes

`float32`, against the values docs/numerics/SEQLEN.md §1.3 recorded before any of this:

| S | recorded | measured here |
|---:|---|---|
| 6 | `b9fc5553ee1bf6a2…` | `b9fc5553ee1bf6a2…` ✅ |
| 32 | `331668f36da02f21…` | `331668f36da02f21…` ✅ |
| 128 | `00159a9dbd308eda…` | `00159a9dbd308eda…` ✅ |
| 512 | `07c2797dabc4552e…` | `07c2797dabc4552e…` ✅ |

`bfloat16` was measured too, because `aten.rs`'s shared helpers were touched. `S=128` matches
docs/numerics/SEQLEN.md's recorded `7ff8e9334449b147…`; the other two had not been recorded before and are
written down here as the new baseline:

| S | `bfloat16` |
|---:|---|
| 6 | `8ef1550ea33c4f3d…` |
| 32 | `b81325c83a0a3d15…` |
| 128 | `7ff8e9334449b147…` ✅ matches §7 of docs/numerics/SEQLEN.md |

This is the guarantee §3.3's `None` fast path exists for: a NaN-free reduction never enters the
correction, so it keeps the exact bits it had.

### 6.5 What this round did not verify

* **Android / iOS.** Host artefact only. No new `#[cfg]` and no new FFI surface, so the same scope
  limit docs/bindings/SPELLINGS.md §7.9 and docs/architectures/ARCH20.md §11.5 already record.
* **Performance.** No timing claim is made. The NaN corrections add one `ne` + one `sum_all` per
  floating-point call to five ops; none is on the prefill path (the sweep's wall times moved within
  the noise of a machine that was not quiet, and are not reported as a measurement). `amax` and
  SDPA are untouched.
* **The SDPA math backend.** §2.3 corrected the refusals' *reason* and did not build the composite.
  Its kernels are all present; nobody has transcribed the sequence. That is the largest item this
  round names and leaves.
  > **Correction (문서 감사, 2026-09):** built since — `_sdpa_math` exists in
  > `rust/torch_c/src/bootstrap.py` (landed in `1938ad1`, "Feat: Open training mode, which every
  > sweep in this repository had assumed away", which post-dates this document's own commit
  > `3b7d981`; `git merge-base --is-ancestor 3b7d981 1938ad1` confirms the order). Live-verified:
  > `F.scaled_dot_product_attention(q, k, v, dropout_p=0.1, is_causal=True)` — the exact scenario
  > that falls off the flash path this document's own `bootstrap.py` comment names
  > (`dropout_p != 0.0` routes to `_sdpa_math`) — succeeds today. Round 2's `docs/kernels/SDPA.md`/
  > `docs/models/CKPT2.md`/`docs/models/GENERATE.md` audits already independently confirm the same function
  > (its `enable_gqa` branch, specifically).
  > <!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _sdpa_math present -->
* **`aten.amin.default`.** Named in §2.4, not written. It is a direction-specific `CustomOp1`, not
  a sign flip. **Status: confirmed still true** — `aten.amin.default` and `aten.argmin.default` are
  both still absent from the current 168-op `_aten_implemented()` list.
  <!-- DOCWATCH: op-not-implemented aten.amin.default -->
  <!-- DOCWATCH: op-not-implemented aten.argmin.default -->
