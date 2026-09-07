# METAEMB — the meta wall real models actually stop on, re-measured and closed

Worktree `work/metaemb` on develop `c4b8b6a`. Upstream torch 2.13.0
(`/Volumes/macMini/caches/spike-venv/bin/python`) is the oracle throughout, in its own
process with `PYTHONPATH` stripped — the same method docs/METAFAM.md §4 and docs/VOICE4.md
§6 use.

---

## 0. Answers first

| question | answer | where |
|---|---|---|
| What did METAFAM.md leave undone, by name? | `aten.embedding.default` — measured as the first wall for 5 of 7 architectures and skipped because it belonged to a different family — plus the four multi-output ops `max.dim`, `argmax`, `topk`, `sort`. | §1 |
| Was METAFAM's measured list still true? | **No.** Two of its three entries were closed by METAFAM itself, so bert and roberta had already advanced. Re-measured here: `embedding` ×5, `gather` ×2, `convolution` ×1. **A measured priority list in this repository goes stale in one round.** | §2.1 |
| **How many of the eight architectures construct end to end under `torch.device("meta")`?** | **0 of 8 before. 7 of 8 after.** | §2.4 |
| What is the new top wall? | **There is no meta-kernel wall left for these eight.** The one architecture that does not finish (`bart`) stops at `Tensor.item() cannot be called on meta tensors` — **and upstream torch fails it identically**, so 7 of 8 is upstream parity, not a shortfall. | §2.5 |
| What is the INDEX-DTYPE rule? | The index half of a multi-output reduction is `int64` **unconditionally**; the value half keeps the input's own dtype with no widening. Derived by measurement across 9 dtypes × 2 devices × 4 ops, not assumed. | §3.1 |
| How many meta kernels? | 22 op names, 17 new `match` arms. **Zero new operators** — every one already had a dense kernel and golden cases. | §3 |
| Did nullification catch everything? | **12 of 12 caught.** Nothing passed silently. | §7 |
| What refuses BY NAME, with a reason? | `masked_select`, `_unique2`, `repeat_interleave.Tensor`, and `index.Tensor` **with a boolean mask** — output shape is a function of the input's VALUES. | §5 |
| Gate | §8 |

---

## 1. What was left open, and why it was left

docs/METAFAM.md closed the reduction and view families and reported, in its own §2.2, that
the measured top wall was somewhere else entirely:

> `aten.embedding.default` — the contraction family, first wall for 5 of 7 real
> architectures' forward pass under meta, ranked ABOVE the reduction/view families in
> measured priority. Out of this round's scope.

and in §6, the four ops it did not attempt:

> `max.dim`, `argmax.default`, `topk.default`, `sort.default` — all return an INDEX tensor
> alongside (or instead of) a value tensor, so their meta kernel has to invent an `int64`
> shape rule for the index half too.

This round is those two sentences.

---

## 2. The measurement — re-run, not inherited

### 2.1 The inherited list had gone stale in one round

The instruction was to re-run METAFAM's §2.2 measurement rather than trust it, because
"this repository has had a measured priority list go stale between rounds". It had.

```
METAFAM §2.2 (before this round)      re-measured here (same probe, current develop)
gpt2        -> embedding              gpt2        -> embedding
bert        -> slice.Tensor           bert        -> gather.default      <-- moved
mistral     -> embedding              mistral     -> embedding
qwen2       -> embedding              qwen2       -> embedding
bart        -> embedding              bart        -> embedding
distilbert  -> embedding              distilbert  -> embedding
roberta     -> cumsum.default         roberta     -> gather.default      <-- moved
(not probed)                          whisper     -> convolution.default <-- added
```

**Both entries that moved moved because METAFAM closed them.** `slice.Tensor` and
`cumsum.default` were two of the kernels that round landed, so the wall behind each became
visible the moment they were fixed — docs/ARCH20.md §0.2's *"one wall is not one wall"*,
observed again. A list of walls is only true of the build it was measured on.

**Measured ranking, this round, before any change:**

```
aten.embedding.default    x5   (gpt2, mistral, qwen2, bart, distilbert)
aten.gather.default       x2   (bert, roberta)
aten.convolution.default  x1   (whisper)
```

`embedding` is confirmed as the top of measured demand, which is what METAFAM predicted;
the rest of its list was not.

### 2.2 The probe

Eight architectures from the local HF cache — `gpt2`, `bert-base-uncased`,
`mistralai/Mistral-7B-v0.1`, `Qwen/Qwen2-7B`, `facebook/bart-base`, `openai/whisper-tiny`,
`distilbert-base-uncased`, `roberta-base` — built with `AutoModel.from_config` (no weights
downloaded, nothing installed into the spike venv) under `with torch.device("meta")`, then
a full `forward` with meta inputs. The first exception per architecture is the wall.

The probe **asserts `hasattr(torch._C, "_aten_implemented")`** before doing anything. A
previous round lost hours to an unbuilt vendored tree making every probe silently import
upstream torch and report a clean run; the assertion is there so that failure is loud.

Each probe stops at the FIRST wall per architecture, so a second-ranked op is invisible
until the first is closed. That is a floor on the measurement and the reason the loop below
exists rather than a single list.

### 2.3 Re-measured after every landing, which is the method

docs/META.md §7.2's own approach: do not implement a list, implement one wall and measure
again.

| round | landed | what the queue printed next | construct |
|---|---|---|---|
| 0 | — | `embedding` ×5, `gather` ×2, `convolution` ×1 | **0 / 8** |
| 1 | `embedding`, `gather` | `native_layer_norm` ×5, `matmul` ×2, `convolution` ×1 | 0 / 8 |
| 2 | contraction family + `native_layer_norm` | SDPA ×4, `cat` ×2, `split` ×1, `convolution` ×1 | 0 / 8 |
| 3 | `cat`, `split`, `convolution`, SDPA | `gelu` ×5, `silu` ×2 | **1 / 8** (gpt2) |
| 4 | `gelu`, `silu`, `relu` | `repeat` ×1 (whisper), `.item()` ×1 (bart) | **6 / 8** |
| 5 | `repeat` | `index.Tensor` ×1 (whisper) | 6 / 8 |
| 6 | `index.Tensor` (integer half) | — | **7 / 8** |

**Rounds 0–2 added ten kernels and moved the count from 0 to 0.** That is the whole of
CLAUDE.md §5.3's point in one table: a kernel count is not progress, and this round would
have reported "twelve kernels added" as a success at the end of round 2 while nothing a
user could do had changed.

### 2.4 The answer: 0 of 8 before, 7 of 8 after

```
BEFORE                                     AFTER
gpt2        embedding                      gpt2        OK  (1, 8, 768)
bert        gather.default                 bert        OK  (1, 8, 768)
mistral     embedding                      mistral     OK  (1, 8, 4096)
qwen2       embedding                      qwen2       OK  (1, 8, 3584)
bart        embedding                      bart        .item() on a meta tensor
whisper     convolution.default            whisper     OK  (1, 4, 384)
distilbert  embedding                      distilbert  OK  (1, 8, 768)
roberta     embedding                      roberta     OK  (1, 8, 768)

0 of 8 construct                           7 of 8 construct
```

The seven output shapes are identical to upstream's, which was checked rather than assumed
(§2.5 runs the same probe on upstream).

### 2.5 The eighth is upstream's wall, not this shim's

`facebook/bart-base` stops at

```
RuntimeError: Tensor.item() cannot be called on meta tensors
```

which is not a missing meta kernel — it is bart's own code calling `.item()` on a meta
tensor, and reading a value out of a meta tensor is the one thing meta is *defined* not to
support (docs/META.md §2.2). **The identical probe run against upstream torch 2.13.0 fails
bart at the identical message**, and passes the other seven. So:

```
upstream torch 2.13.0   7 of 8
this shim (after)       7 of 8
```

The control run is why this is reported as parity rather than as a remaining gap. Without
it, "7 of 8" would have looked like one architecture still owed.

---

## 3. The kernels

22 op names, 17 new `match` arms in `meta_dispatch` (`rust/torch_c/src/aten.rs`).

| arm | ops | shape | dtype |
|---|---|---|---|
| embedding | `embedding.default` | index shape + `weight[1]` | **weight's**, never the index's |
| gather | `gather.default` | the INDEX's | the INPUT's |
| extremum | `max.dim`, `min.dim` | axis removed / kept | value: input's; index: `int64` |
| argmax | `argmax.default` | axis removed / kept / flattened | **`int64` only** |
| topk | `topk.default` | `dim` set to `k` | value: input's; index: `int64` |
| sort | `sort.default` | unchanged | value: input's; index: `int64` |
| mm | `mm.default` | `[m, n]` | lhs's |
| bmm | `bmm.default` | `[b, m, n]` | lhs's |
| addmm | `addmm.default` | `mm`'s | `mat1`'s |
| matmul | `matmul.default` | batch broadcast + matrix product | lhs's |
| layer norm | `native_layer_norm.default` | out: input's; mean/rstd: normalised axes **collapsed to 1** | stats `float32` only under mixed dtype |
| cat | `cat.default` | concat axis summed | `promote_list` (the dense path's own) |
| split | `split.Tensor`, `split_with_sizes.default` | chunked along `dim` | input's, every chunk |
| convolution | `convolution.default` | the arithmetic, both directions | input's |
| SDPA | `_scaled_dot_product_flash_attention_for_cpu.default` | out: query's; logsumexp: `(B, H, Lq)` | out: query's; **logsumexp `float32` always** |
| activations | `gelu.default`, `silu.default`, `relu.default` | unchanged | unchanged |
| repeat | `repeat.default` | right-aligned tile | input's |
| index | `index.Tensor` (integer half) | broadcast index block, placed by adjacency | input's |

### 3.1 The INDEX-DTYPE rule, derived rather than assumed

The instruction was to state the rule explicitly and derive it from upstream. Derived by
transcript, across **9 input dtypes × 2 devices (`cpu`, `meta`) × 4 ops**, plus `keepdim`,
`largest`, `sorted`, `descending`, `stable`, empty results and 0-dim inputs:

> **The index half of a multi-output reduction is `int64`, unconditionally.** It does not
> depend on the input dtype, on `keepdim`, on `largest`/`sorted`/`descending`/`stable`, on
> whether the result is empty, or on the device.
>
> **The value half carries the input's own dtype, unchanged** — no widening, no promotion.
>
> `argmax` is the degenerate member: it returns only the index half, so its **entire**
> output is `int64` whatever went in.

Two things make this worth stating rather than assuming.

**The value half is where a `sum`-shaped habit goes wrong.** `sum` widens an integral input
to `int64` (`sum_natural_tag`, VOICE4.md §4.2). `max.dim`, `topk` and `sort` do **not**. A
kernel that reused `sum_natural_tag` here would pass every floating-point case and fail
every integral one — and floating-point is what a model exercises, so the model would not
catch it.

**The index half is where the failure is silent.** A meta kernel returning the input's dtype
for the index produces a pair with the right shapes and a plausible-looking value half.
Nothing fails at the call. It fails the first time somebody *uses* the index — an
`index_select`, a `gather`, a `torch.take` — which is arbitrarily far from the kernel that
caused it. That is why the rule is written once, in one helper, and not four times:

```rust
fn meta_values_indices(py, result_type, shape, values_tag) -> PyResult<Py<PyAny>> {
    let pair = (
        promote(py, meta_result(py, shape.clone(), values_tag)?)?,
        promote(py, meta_result(py, shape, TorchDType::Int64)?)?,   // <-- here, and only here
    );
    ...
}
```

Four call sites cannot drift to four different index dtypes if none of them names one.
Nullification 4 (§7) breaks that single line and five tests go red.

**The same shape of rule, twice more.** `native_layer_norm`'s `mean`/`rstd` and SDPA's
`logsumexp` are both second outputs with a different shape *and* a different dtype from the
first. SDPA's `logsumexp` is `float32` even for a `float16` query — measured on both
upstream devices. `native_layer_norm`'s statistics are collapsed-to-1, not axis-removed;
answering `reduced_dims(..., keepdim=false)` there gives a plausible tensor of the wrong
rank.

### 3.2 What this shim follows when upstream disagrees with itself

docs/META.md §7.3 already catalogues three places where upstream's own `cpu` kernel and its
own `meta` kernel answer differently for the same call. **This round found thirteen more.**
The standing rule applies unchanged — this shim has one door, so it follows its dense
kernel, which follows cpu:

| # | call | upstream `meta` | upstream `cpu` | this shim |
|---|---|---|---|---|
| 1 | `max.dim` reducing a zero-extent dim | answers a shape | `IndexError` | refuses |
| 2 | `argmax(empty)` with no `dim` | `RuntimeError` | `IndexError` | `IndexError` |
| 3 | `sort(x, dim=9)` on a 2-D input | **silently answers** | `IndexError` | refuses |
| 4 | `embedding` with a 1-D or 3-D weight | `AssertionError` | `RuntimeError` | `RuntimeError` |
| 5 | `native_layer_norm` on an integral input | `RuntimeError` | `NotImplementedError` | `NotImplementedError` |
| 6 | `native_layer_norm` statistics for `f16`/`bf16` | `float32` | the input's dtype | the input's dtype |
| 7 | SDPA with unequal Q/K/V head sizes | answers the query's | `RuntimeError` | refuses |
| 8 | `gelu`/`silu` on a non-floating input | answers | `NotImplementedError` | refuses |
| 9 | `relu(bool)` | answers | `RuntimeError` | refuses |
| 10 | `index.Tensor` with too many indices | `RuntimeError` | `IndexError` | `IndexError` |

Row 3 is the one worth pausing on: **upstream's meta kernel for `sort` accepts an
out-of-range `dim` and answers a shape.** `sort(zeros(3, 4, device="meta"), dim=9)` returns
`(3, 4)`. The `normalise_dim` call in this shim's `sort` meta arm looks dead — its result is
discarded — and it is not: it is the entire refusal, and nullification 7 (§7) confirms it.

Every one of these is in `_KNOWN_META_VS_CPU_DIVERGENCE` in the test file, excluded from the
strict diff and then **asserted by name for the chosen answer**, so "excluded" never means
"unchecked". The test additionally asserts that upstream still disagrees — if a future torch
release makes upstream self-consistent, the exclusion is reported as stale rather than
silently protecting nothing.

### 3.3 Pre-existing DENSE defects this round found and did not fix

Writing a meta kernel means writing down what the dense kernel's rule *is*, and that is an
audit of the dense kernel. Five disagreements turned out to be the dense side being wrong,
not the meta side. **None is fixed here** — a dense kernel change is a golden-case change
and out of this round's scope — and all five are recorded so the next round has them:

| # | call | this shim's dense | upstream (BOTH devices agree) |
|---|---|---|---|
| 1 | `argmax(x, keepdim=True)` with `dim=None` | `(1,)` | `(1, 1)` — a shape of `rank` ones |
| 2 | `max.dim` on a 0-dim tensor | `RuntimeError: candle: max: dimension index 0 out of range for shape []` | answers `()` |
| 3 | `argmax` on an empty tensor along a NON-reduced dim | `RuntimeError: candle: empty tensor for reduce` | answers `(0,)` |
| 4 | `convolution` with a kernel bigger than the padded input | answers `(1, 8, 0)` | `RuntimeError: ... Kernel size can't be greater than actual input size` |
| 5 | `torch.cat([], 0)` | `RuntimeError` | `ValueError` |
| 6 | `mm`/`matmul` with mismatched inner dims | leaks candle's message | `mat1 and mat2 shapes cannot be multiplied (3x4 and 5x6)` |

Defects 2 and 3 are candle error messages leaking through a kernel that should have refused
in torch's own words, and 4 is worse than a leak: **it answers**. The meta arms here follow
upstream in all six rather than transcribing a leak into a second place, and each is noted
at its call site in `aten.rs`.

Defect 5 has a further consequence recorded in the tests: `torch.cat([], 0)` carries no
tensor, so `check_devices_agree` finds no meta device to route on and the call lands in the
**dense** kernel even when the caller meant meta. The meta arm's own `ValueError` for an
empty list is therefore unreachable today. It is kept, and said to be unreachable, rather
than deleted — deleting it would make the arm wrong the day a meta device can be named
without a tensor.

---

## 4. What a meta `gather` cannot promise

The dense `gather_default` makes four checks. Three are shape checks and are reproduced
exactly. The fourth is not, and cannot be:

```rust
if target < 0 || target as usize >= self_dims[dim] {
    return Err(... "index {target} is out of bounds for dimension {dim} ...");
}
```

That reads index **values**, and a meta tensor holds none. Upstream's own meta kernel omits
the same check for the same reason (measured: an out-of-bounds index answers a shape on
meta and raises on cpu). So a `gather` that would raise on a real tensor answers cleanly on
meta. **That is the one thing a caller loses by running `gather` under meta**, and it is
stated here rather than left to be discovered.

The same boundary, drawn explicitly rather than discovered, is what §5 is about.

---

## 5. Refused BY NAME, with the reason

Four things in this area must never get a meta kernel, and before this round three of them
fell through to `meta_dispatch`'s generic message:

> ...this op would have to infer its output shape without computing — which is a real
> kernel (upstream registers one in `torch/_meta_registrations.py`), not a fallthrough.

That sentence is true of everything else behind that fallthrough and **false of these**, and
a false-but-encouraging refusal is worse than none: it sends the next round to write a
kernel that cannot exist.

| refused | why no meta kernel can exist | upstream |
|---|---|---|
| `aten.masked_select.default` | output length = number of TRUE entries in the mask | **no meta registration at all** |
| `aten._unique2.default` | output length = number of distinct elements | **no meta registration at all** |
| `aten.repeat_interleave.Tensor` | output length = sum of the repeat tensor | refuses by name: *"cannot repeat_interleave a meta tensor without output_size"* |
| `aten.index.Tensor` **with a bool mask** | output extent = number of TRUE entries | routes into `torch.nonzero`'s meta registration, which refuses |

All four now name themselves and give the reason ("its output SHAPE is a function of the
input's VALUES ... This is a refusal, not a gap — upstream has no meta kernel for these
either"). `test_the_three_data_dependent_ops_refuse_by_name_with_the_reason` asserts both
the exception class and the message content.

**`index.Tensor` is therefore deliberately PARTIAL, and the split is the point.** With
integer indices the output shape is a function of the shapes alone and is answered; with a
boolean mask it is not and is refused. That is exactly where upstream draws the same line.

`aten.nonzero.default` is **not** in this list although it belongs to the same category: it
already has an arm, gated behind upstream's own
`torch.fx.experimental._config.meta_nonzero_assume_all_nonzero` flag, which is upstream's way
of letting a caller opt in to an upper-bound answer. Where upstream offers that switch, this
shim mirrors it; where it does not, neither does this.

---

## 6. Tests — `rust/torch_c/pytests/test_metaemb.py`

20 test functions over a single probe of **~290 recorded cases**, run in two subprocesses
(shim / upstream) and diffed key by key.

**Shape, dtype AND stride** are recorded for every case, per the instruction. Strides are
comparable because this shim's meta tensors do report `.stride()` — checked before the
tests were written rather than assumed.

Coverage per the required edge argument forms:

- **0-dim** — `max.dim`, `argmax`, `topk`, `sort`, `gather`, `embedding` (0-dim index),
  `repeat`, `index.Tensor` (0-dim index)
- **empty** — zero-extent reduction dims, zero-extent non-reduced dims, `topk(k=0)`,
  `sort(empty)`, empty index tensors, `split` of an empty tensor, `cat` skipping a 1-D empty
- **keepdim** — `max.dim`, `argmax` (including the `dim=None, keepdim=True` form)
- **negative dims** — every op that takes a `dim`
- **dtype promotion** — all nine dtypes through `embedding`, `gather`, `max.dim`, `argmax`,
  `topk`, `sort`, `matmul`; `cat`'s promotion including `int64` × `bool`; `native_layer_norm`
  and SDPA's reduced-precision cases
- **refusals** — 30-odd, each compared against upstream's exception class

Three tests do not diff against upstream and assert the shim's own answer literally, so that
a change making both sides wrong in the same way still fails:
`test_the_index_half_is_int64_whatever_the_input_dtype_was` (36 assertions),
`test_the_value_half_keeps_the_inputs_own_dtype_and_does_not_widen`, and
`test_the_thirteen_upstream_self_disagreements_are_followed_to_the_dense_side`.

### 6.1 The exclusion lists are bounded and checked

Three named sets exclude cases from the strict diff. Each is asserted to have exactly its
stated membership, and each excluded case is checked separately:

- `_KNOWN_META_VS_CPU_DIVERGENCE` (13) — §3.2, asserted by name for the chosen answer
- `_KNOWN_SHIM_WIDE_REFUSAL` (3) — `embedding`'s two no-autograd refusals and `matmul`'s
  1-D refusal, all three inherited from the dense kernel and nothing to do with meta
- `_UNREACHABLE_FROM_META` (2) — cases that never reach `meta_dispatch` at all (§3.3)
- `_STRIDE_DIVERGENCE` (7) — **stride only**; shape and dtype are still compared, and that
  is asserted rather than assumed

### 6.2 The stride finding

Two of the seven stride divergences are degenerate (a zero-length axis). **Five are not**,
and they are this round's contribution to docs/META.md §12's standing gap:

```
split(zeros(6, 4, device="meta"), 3, dim=1)
    upstream   chunk strides (4, 1) and (4, 1)   <- VIEWS, keeping the parent's stride
    this shim  chunk strides (3, 1) and (1, 1)   <- contiguous, computed from each shape

SDPA logsumexp, shape (2, 4, 8)
    upstream   (32, 1, 4)                        <- laid out transposed
    this shim  (32, 8, 1)                        <- contiguous
```

META.md §12 records that this shim's meta tensors carry no stride field and that `expand`
was the first case where "everything meta makes is contiguous" stopped being self-evident.
**These are the second and third, and the first where the divergence is not confined to an
empty tensor.** Closing them means adding a stride field to `PyTensorBase`'s `Repr::Meta`,
which is a representation change and out of this round's scope — §9.

### 6.3 The existing boundary test moved rather than being deleted

`test_ops_without_a_meta_kernel_name_themselves` in `test_shim.py` named `mm`, `bmm`, `cat`,
`max.dim`, `argmax`, `topk`, `sort` and `index.Tensor` as ops that must refuse. All eight now
answer, so the test would have failed as a **regression detector** without being wrong about
anything real. The refusing list was replaced with twelve ops **confirmed still refusing by
direct dispatch before the edit** (METAFAM.md's discipline, and CLAUDE.md §5.5's), and the
answering half — the half that stops this test passing on an empty meta table — grew by 17
names plus a new multi-output section that reads the index dtype.

---

## 7. Nullification — 12 of 12 caught

CLAUDE.md §5.5. Each row is a deliberate ONE-LINE break of the landed code, a full rebuild
and reinstall of the vendored shim, a re-run of `test_metaemb.py`, and a restore from a
known-good copy. Automated end to end so that no round was skipped or hand-waved.

| # | nullification | tests that went red | verdict |
|---|---|---:|---|
| 1 | `embedding` takes its dtype from the INDEX instead of the weight | 1 | **red** |
| 2 | `embedding` drops the index-dtype check entirely | 1 | **red** |
| 3 | `gather` answers the INPUT's shape instead of the index's | 2 | **red** |
| 4 | **the index half is the value dtype instead of `int64`** (the one shared line) | **5** | **red** |
| 5 | `argmax` answers the input dtype instead of `int64` | 2 | **red** |
| 6 | `topk` keeps the full extent instead of `k` | 2 | **red** |
| 7 | `sort` drops the `normalise_dim` refusal | 1 | **red** |
| 8 | `native_layer_norm` removes the normalised axes instead of collapsing them to 1 | 1 | **red** |
| 9 | `matmul` takes the lhs's batch dims instead of broadcasting | 1 | **red** |
| 10 | `convolution` drops the dilation term | 1 | **red** |
| 11 | `index.Tensor` accepts a bool mask (drops the refusal) | 2 | **red** |
| 12 | SDPA's `logsumexp` takes the query's dtype instead of `float32` | 2 | **red** |

**Nothing passed silently. There is no "not caught" row in this table**, which is the row
VOICE4.md §6.1 had and the row this format exists to make visible.

Two observations worth keeping:

**Nullification 4 is the shape of the whole round.** One line, five tests. Three of the five
are the per-op upstream diffs (`max.dim`, `topk`, `sort`) and one is the direct assertion
that the dtype is literally `int64` — so the rule is caught both by comparison and by
statement, and would still be caught if upstream changed.

**`test_the_stride_exclusion_covers_exactly_one_named_case` fired on four of the twelve**
as a secondary detector, because it asserts that shape and dtype still agree for the
stride-excluded keys. An exclusion list that also checks the thing it is not excluding turns
out to be a load-bearing test rather than bookkeeping.

**What was NOT nullified**, and why: `mm`/`bmm`/`addmm` (same shape rule as `matmul`, whose
broadcast is nullified in 9), `cat`, `split`, `repeat`, the three activations, and the
`min.dim` half of the extremum arm. Each is the same *kind* of rule as one that was
nullified — axis arithmetic, dtype pass-through, or a transcribed refusal — and the twelve
chosen cover every distinct kind in the set: dtype-from-the-wrong-operand,
shape-from-the-wrong-operand, index-dtype, dropped-refusal, collapse-vs-remove,
broadcast-vs-take-one-side, dropped-arithmetic-term, and a partial kernel's boundary.
Extending the count would extend coverage of the same shapes, not find a new one.

---

## 8. Gate

```
PATH="$HOME/.cargo/bin:$PATH" PYTHON=/Volumes/macMini/caches/spike-venv/bin/python \
    bash rust/torch_c/pytests/run.sh
```

```
pytests         1055 ok / 0 FAIL   (baseline 1035 -> 1055, +20, all in test_metaemb.py)
cargo test      30 passed / 0 failed
golden          11420 / 11420 passed, failed 0, pending 0, ops = 302
golden self-test PASS -- 25 comparators x 11 fault modes, 0 problems
schema          5024 / 5037 matched
DOCWATCH        968 / 968 evaluated marker(s) hold  (baseline 939 -> 968, +29)
run.sh exit     0
```

The baseline run on this worktree, before any change, reported **1034 ok / 1 FAIL** -- the
one failure was `test_all_five_reduce_ops_agree_with_upstream_and_the_bitwise_three_refuse`
timing out at 600s in its multiprocess (`gloo32-3`) leg while a sibling worktree's gate ran
concurrently on the same eight cores. 1034 + 1 = 1035, the documented baseline. It passes in
the landing run above, so it was contention and not this round's changes -- stated rather
than assumed innocent, per CLAUDE.md §5.5. That failing baseline run also exited before
`run.sh` reached the golden self-test and DOCWATCH, which is why the baseline for those two
is taken from the documented figures rather than from that run.

`_aten_implemented()` and the golden case count are **unchanged by design**: every op here
already had a dense kernel and golden cases, a meta kernel adds no values to compare, and no
new spelling was created. `test_every_op_this_round_gave_a_meta_kernel_is_already_a_dense_op`
asserts all 22 names are already in `_aten_implemented()`, so the claim is checked rather
than written down — and so this round cannot trip the `golden_cases_failed eq 0` marker by
adding an op without a case.

---

## 9. What is not closed

- **Strides on meta tensors** (§6.2). Five non-degenerate divergences now measured. Needs a
  stride field on `Repr::Meta` — a representation change, docs/META.md §12's standing item.
- **`aten.sort.stable`** — a separate overload from `sort.default`, with `stable` in
  argument position 1. No meta kernel; found while writing the tests and dropped from the
  probe rather than silently passed.
- **`bart` under meta**, which needs `.item()` on a meta tensor. **Not closable** — upstream
  fails identically (§2.5).
- **The six dense defects in §3.3.** Found here, fixed nowhere. Each is a golden-case change.
- **Everything else in docs/META.md §7.4's table** that these eight architectures do not
  reach: `stack`, `unbind`, `squeeze.dims`, `narrow`, `flip`, `scatter`, `_softmax`,
  `constant_pad_nd`, `zeros_like`, `abs`, `ceil`, `floor_divide`, `bitwise_and`/`bitwise_or`.
  Confirmed still refusing by direct dispatch (§6.3) rather than assumed.
- **Architectures beyond these eight.** Eight is eight. docs/META.md §7.2's twenty-architecture
  *construction* sweep is a different measurement from this *forward* one and was not re-run.
- **GraalVM native image.** Not attempted; out of this round's scope.

<!-- DOCWATCH: op-implemented aten.embedding.default -->
<!-- DOCWATCH: op-implemented aten.gather.default -->
<!-- DOCWATCH: op-implemented aten.max.dim -->
<!-- DOCWATCH: op-implemented aten.min.dim -->
<!-- DOCWATCH: op-implemented aten.argmax.default -->
<!-- DOCWATCH: op-implemented aten.topk.default -->
<!-- DOCWATCH: op-implemented aten.sort.default -->
<!-- DOCWATCH: op-implemented aten.mm.default -->
<!-- DOCWATCH: op-implemented aten.bmm.default -->
<!-- DOCWATCH: op-implemented aten.addmm.default -->
<!-- DOCWATCH: op-implemented aten.matmul.default -->
<!-- DOCWATCH: op-implemented aten.native_layer_norm.default -->
<!-- DOCWATCH: op-implemented aten.cat.default -->
<!-- DOCWATCH: op-implemented aten.split.Tensor -->
<!-- DOCWATCH: op-implemented aten.split_with_sizes.default -->
<!-- DOCWATCH: op-implemented aten.convolution.default -->
<!-- DOCWATCH: op-implemented aten.gelu.default -->
<!-- DOCWATCH: op-implemented aten.silu.default -->
<!-- DOCWATCH: op-implemented aten.relu.default -->
<!-- DOCWATCH: op-implemented aten.repeat.default -->
<!-- DOCWATCH: op-implemented aten.index.Tensor -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs meta_values_indices present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs mm_shape_refusal present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metaemb.py test_the_index_half_is_int64_whatever_the_input_dtype_was present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metaemb.py test_the_value_half_keeps_the_inputs_own_dtype_and_does_not_widen present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metaemb.py test_the_three_data_dependent_ops_refuse_by_name_with_the_reason present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metaemb.py test_the_thirteen_upstream_self_disagreements_are_followed_to_the_dense_side present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_metaemb.py test_every_op_this_round_gave_a_meta_kernel_is_already_a_dense_op present -->
<!-- DOCWATCH: count golden_ops_covered ge 302 -->
