# ARGFORM — the argument-form gaps, and the two owed bindings picked up alongside them

Worktree `work/argform` on develop `eb84708`. Territory: `rust/torch_c/src/bootstrap.py`,
`rust/torch_c/pytests/test_argform.py`. `test_shim.py`, `aten.rs`, `capture.rs`, `tape.rs`,
`tensor.rs`, `device.rs`, `methods.json`, `overloads.json`, `tools/`, `torchnative/` were not
touched, per this round's territory split.

Every alias or coercion below was **measured against real torch 2.13.0 in a separate process**
before being added — CLAUDE.md's own warning about this round: upstream accepts a numpy spelling
on some ops and refuses it on others, so accepting it everywhere would make this shim more
permissive than the thing it replaces, which is a silent divergence nobody tests for.

## 1. The table

| form | upstream accepts? | what was done |
|---|---|---|
| `torch.ones(dtype=bool)` | yes, maps to `torch.bool` | fixed |
| `torch.ones(dtype=int)` | yes, maps to `torch.int64` | fixed (not one of the six, but the same rule; measured alongside) |
| `torch.ones(dtype=float)` | yes, maps to `torch.float64` | fixed (ditto) |
| `torch.ones(dtype=complex)` | yes, maps to `torch.complex128` | **not fixed** — complex has no candle storage behind it here (`_NUMPY_DTYPE_TO_TORCH`'s own note), so mapping to a dtype this shim cannot back would trade one refusal for a worse one |
| `Tensor.mean(axis=)` | yes, aliases `dim=` | fixed |
| `torch.mean(axis=)` | yes, aliases `dim=` | fixed |
| `Tensor.mean(keepdims=)` | yes, aliases `keepdim=` | fixed (measured alongside `axis=` on the same call) |
| `Tensor.sum(axis=)` | not measured this round | **not installed** — see §2's trap |
| `torch.div(int, int, rounding_mode=)` | yes — upstream wraps bare Python numbers into 0-dim tensors wherever a schema wants `Tensor` and gets a `Scalar` | fixed, for `div` only |
| `torch.add(int, int)` | yes (measured, not fixed) | **not generalised** — see §2 |
| `torch.mean(int)` | **no** — `argument 'input' (position 1) must be Tensor, not int` | confirms the rule is per-op; nothing changed for `mean` here |
| `F.adaptive_avg_pool2d(x, 2)` (bare int) | yes, upstream's `torch._C._nn.adaptive_avg_pool2d` binding (not the raw aten op) normalises it | fixed |
| `torch.ops.aten.adaptive_avg_pool2d.default(x, 2)` | **no**, raw aten op refuses | left refusing — confirmed still refuses after the fix (docs/kernels/FIXES.md §3's own trap) |
| `aten.convolution.default` with per-axis padding `[0, 5]` | **yes** — this is not asymmetric padding, it is two axes with different (each individually symmetric) amounts, and upstream computes it fine | **not fixable here** — this shim's candle backend refuses any per-axis-differing stride/padding/dilation by name (`aten.rs`, out of territory); a genuine backend limitation, not an argument-form gap, matching ARCH100 §2.1's `MatMulUnexpectedStriding` row in kind |
| `torch.embedding(weight, None, padding_idx, ...)` | **no** — `TypeError: embedding(): argument 'indices' ... must be Tensor, not NoneType` | **not a gap** — upstream refuses the exact same call. See §3 |

## 2. The trap, and where the line was drawn

`axis=`/`keepdims=` and the bare-number-wraps-to-Tensor rule are both accepted on some ops and
refused on others upstream — measured directly rather than assumed:

* `x.sum(axis=1)`, `x.std(axis=1)`, `x.var(axis=1)`, `x.argmax(axis=1)` all accept `axis=` too
  (measured), but **none of them was one of the six architectures**, so none of them got the
  alias. `_NUMPY_KEYWORD_ALIASES` in `bootstrap.py` is keyed by op name and holds exactly one
  entry, `mean`, with exactly what was measured for it. `test_axis_alias_is_not_installed_for_
  an_op_that_was_not_measured` pins this — `Tensor.sum(axis=1)` must still refuse, proving the
  table has not quietly grown into "reductions in general".
* `torch.add(2, 3)` also wraps a bare number into a 0-dim tensor for `self` upstream (measured),
  but the wrapping added here is written **only for `div`** (`varfns.div`, overriding the
  table-driven entry after it is built) rather than folded into the generic `_TypeChecker`'s
  `Tensor` predicate. Folding it in would make every `Tensor self` position in every overload
  table accept a bare number, regardless of whether upstream does — `torch.mean(5)` is the
  measured counter-example: upstream refuses it by name (`argument 'input' ... must be Tensor,
  not int`), so a blanket rule would be wrong for `mean` while being right for `div`.

Both alias tables (`_NUMPY_KEYWORD_ALIASES` for keyword renaming, the hand-written `div` override
for the Tensor-position wrap) are scoped to exactly what was measured. Extending either to another
op needs its own measurement, not an inference from this one.

### 2.1 The collision case

`kwargs[canonical] = kwargs.pop(alias)` would silently pick the alias's value over the caller's own
if both `dim=` and `axis=` were given — a plain dict assignment does not raise on a key that already
exists. Measured upstream: `x.mean(dim=1, axis=1)` raises `TypeError: mean() received multiple
values for argument 'dim'`. The rewrite in `_strip_python_only_kwargs` checks for the collision and
raises before overwriting, rather than silently choosing a value — `test_giving_both_dim_and_axis_
does_not_silently_pick_one` is the regression test for this, added after it was caught red the first
time this file was written (see the commit history of this round for the discovery).

## 3. `torch.embedding` — measured to not be an argument-form gap at all

ARCH100.md's transcription was `torch.embedding(): (Parameter, NoneType, int, bool, bool)` for
`sam3_lite_text_text_model`. Read as types, position 2 (`indices`) is `NoneType`. Measured directly:

```text
w = nn.Parameter(randn(5, 3))
torch.embedding(w, None, 3, False, False)     # upstream, real torch 2.13.0
  -> TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not NoneType
```

Upstream refuses the identical call with the identical shape of error. There is no argument form to
add here — the model, under this sweep's shrunk/random-weight config, is constructing an `indices`
that evaluates to `None` before it ever reaches `embedding`, which is a config/harness question
(ARCH100.md §5.2's own caveat about multimodal fallback inputs) and not an operator gap this file can
close. Left alone.

## 4. `torch._C._nn.adaptive_avg_pool2d` — the fix and its exact boundary

docs/kernels/FIXES.md §3 already measured and recorded where this belongs: `F.adaptive_avg_pool2d`'s Python
wrapper (`torch/nn/functional.py`) hands a bare int straight through `_list_with_default`
unchanged, to `torch._C._nn.adaptive_avg_pool2d` — a *different* binding from
`torch.ops.aten.adaptive_avg_pool2d.default`, and the one whose own argument parser
(`BroadcastingList2[int]`) does the int-to-pair expansion. The raw aten op's schema takes
`SymInt[2] output_size` with no such expansion and refuses a bare int, matching this shim's
already-existing (and untouched) `aten.rs` refusal.

The fix is exactly the one door docs/kernels/FIXES.md named: `bootstrap.py`'s `adaptive_avg_pool2d`
composite (`_install_nn`) now normalises `output_size` before dispatching, and the raw aten call
downstream is unchanged. Verified by construction, not just by reading the doc:
`test_adaptive_avg_pool2d_bare_int_normalises_to_a_pair` calls
`_C._aten_dispatch("aten.adaptive_avg_pool2d.default", x, 2)` directly and asserts it still raises —
the exact SILENT DIVERGENCE docs/kernels/FIXES.md §3 measured and reverted, pinned so it cannot come back
through this door.

`None` entries in the output-size tuple (`F.adaptive_avg_pool2d(x, (None, 4))`, "keep this axis")
are handled too — the same `_list_with_default` rule, measured alongside the bare-int case rather
than assumed to follow from it.

ARCH100.md §6 named `dinov3_convnext` and `efficientnet` as *unclassified* failures with this exact
`adaptive_avg_pool2d: output_size must be 2` message, suspecting but not asserting (per CLAUDE.md
§5.4) that it was this same gap. Confirmed after the fix: `dinov3_convnext` clears its wall entirely
(forward runs). `efficientnet` clears this wall and reaches a new one (§5).

## 5. Which architectures clear their wall — measured with `arch_sweep.py --one`, both artefacts rebuilt

```text
                              before                                    after
gpt_neo          torch.ones(dtype=bool)      construction    torch.bitwise_xor -- overload resolution
                                                               has no table entry                          (new wall, still construction)
ibert            Tensor.mean(axis=)          forward         FORWARD RUNS
imagegpt         torch.mean(axis=)           forward         FORWARD RUNS
longformer       torch.div(rounding_mode=)   forward         TensorBase.as_strided                        (new wall)
nystromformer    asymmetric per-axis padding forward         SAME WALL -- backend limitation, see §1
sam3_lite_text_  torch.embedding(...)        forward         SAME WALL -- not an argform gap, see §3
  text_model
dinov3_convnext  adaptive_avg_pool2d (uncl.) forward         FORWARD RUNS
efficientnet     adaptive_avg_pool2d (uncl.) forward         torch._C._nn.avg_pool2d -- missing binding   (new wall)
```

**3 of 8 now run a complete forward** (`ibert`, `imagegpt`, `dinov3_convnext` — the last of these
was one of the two *unclassified* cases, now resolved and folded into the closed set rather than
left unclassified). **3 more clear the specific wall they were named for** and immediately hit a
different, unrelated operator gap (`gpt_neo` -> `bitwise_xor`, `longformer` -> `as_strided`,
`efficientnet` -> `_nn.avg_pool2d`) — a result, not a failure, per this round's brief. **2 stay
exactly where they were**, for reasons measured and recorded above rather than left as "not yet
gotten to": `nystromformer` is a backend limitation in `aten.rs` (out of territory), and
`sam3_lite_text_text_model`'s call is one upstream refuses too.

Read against ARCH100 §2.1's own framing ("four of six are keyword-alias work"): the four were
`ones(dtype=bool)`, `mean(axis=)` x2, `div(rounding_mode=)`, and all four are now fixed and
measured to advance their architecture. The fifth (`adaptive_avg_pool2d`) was not one of the six but
was named in this round's brief and closes an *unclassified* pair on top. The sixth (asymmetric
conv padding) was measured and found to be mis-named in ARCH100 — it is a backend limitation, not
an argument form — and the seventh (`embedding`) was found to not be a gap here at all.

## 6. Two bindings picked up mid-round, in the same file

Two more items landed in `bootstrap.py` this round because their owning rounds could not reach this
file (nine other agents editing `aten.rs`-adjacent territory concurrently) and this round already
had write access:

### 6.1 `torch._C._nn.glu`

The kernel (`aten.glu.default`) and its golden cases were landed by a concurrent round
(docs/kernels/GLU.md) with the binding deliberately left out, exactly the same shape as this round's own
`adaptive_avg_pool2d` finding: a kernel with no spelling. The three-line binding named in that
document was added verbatim (`dim=-1` default, matching upstream's own default measured there — a
bare `F.glu(x)` halves the *last* axis).

**This worktree's own `aten.rs` does not have the kernel** (it lands on a different branch), so
`_C._nn.glu(x)` in this worktree still raises `NotImplementedError: aten op not implemented in
torch._C shim: aten.glu.default` — which is the *correct* behaviour for the binding alone (it
reaches the dispatcher under the right key, with `dim` read as a real argument rather than
hardcoded) and is exactly what `test_glu_is_advertised_and_reaches_the_kernel_key_with_the_right_
default_dim` asserts. The numeric behaviour was verified separately, in a disposable scratch
worktree checked out at local `develop` (which already carries the merged kernel) with this
round's `bootstrap.py` diff applied on top: `_C._nn.glu` computes and the seven ASR encoders'
shared wall (docs/architectures/ARCH100.md) is gone at the binding layer. That scratch worktree was discarded
before finishing — nothing from it was committed, and this worktree's own `aten.rs` is untouched.

### 6.2 `TensorBase.__setitem__` — the stepped-slice write

docs/bindings/SETITEM.md §2's patch (measured, applied, and reverted by a concurrent round for the same
territory reason as §6.1) was applied **verbatim** at its named anchor: a step != 1 slice on the
*write* side now lowers to `aten.index_put_.default` instead of refusing, with the dtype cast and
negative-bound handling the document calls out by name. No `aten.rs` change was needed — the
document's own §5 established that `index_put_` already answered for both architecture shapes
before this round began.

Verified directly in this worktree (where `index_put_` already has a kernel, unlike `glu`):

```text
g = arange(1, 13).reshape(3, 4)
g[0:3:2] = 0.0
  upstream   [[0,0,0,0], [5,6,7,8], [0,0,0,0]]
  patched    [[0,0,0,0], [5,6,7,8], [0,0,0,0]]   -- exact match, separate-process comparison
```

`tools/golden/compare.py --self-test` aside, running the full comparison shows exactly the one
promotion docs/bindings/SETITEM.md §2.1 named already flipping from `expect="c_error"` to a live pass:

```text
FAIL aten.copy_.default :: member x[0:4:2] = 0.0 [refused -- ...] -- gap appears CLOSED: both
     sides now succeed, promote this case to expect=match and diff real values
```

**This is `tools/golden/cases.py`, out of this round's territory**, and docs/bindings/SETITEM.md already
names the one-line promotion needed for all three flipped cases (§2.1) — so this is the expected,
documented next step, not a new finding.

**A new finding this round did surface, though**: `rust/torch_c/pytests/test_shim.py`'s existing
`test_setitem_writes_the_basic_index_through_to_the_base` pins the *old* refusal —

```python
    g = grid()
    try:
        g[0:3:2] = 0.0
    except NotImplementedError as error:
        assert "step-2" in str(error), str(error)
    else:
        raise AssertionError("a step-2 slice write was silently accepted")
```

— and with the patch applied this assertion now fails, because the write is (correctly) no longer
refused. `git diff eb84708 develop -- rust/torch_c/pytests/test_shim.py` is empty, so whoever lands
this patch for real needs to update this one assertion too (delete the `try`/`except` and assert
the written values instead, matching the `d`/`e` cases just above it in the same test) — a step
docs/bindings/SETITEM.md's own verification (§7, "481 ok, 0 FAIL") did not appear to hit, most likely because
their round's local `test_shim.py` copy differed from what ultimately merged. Recorded here rather
than fixed, because `test_shim.py` is this round's forbidden file, not because it is unimportant —
whoever applies this patch to the tree for real should not be surprised by a suite that goes from
green to one red the moment the patch takes effect.

## 7. Verification run (this round's own bootstrap.py + test_argform.py only, no glu kernel, no setitem promotion)

```text
suite (rust/torch_c/pytests/run.sh)    519 ok, 0 FAIL, DOCWATCH PASS -- 480/480
golden (tools/golden/compare.py)       9137/9137, ops=224 -- exactly unmoved
```

With the setitem patch additionally applied (§6.2, verified separately since it is not part of this
round's committed diff pending the two out-of-territory follow-ups above):

```text
golden      9136/9137 -- the one promotable case (§6.2), everything else unmoved
suite       1 known FAIL in test_shim.py (§6.2), everything else unchanged
```
