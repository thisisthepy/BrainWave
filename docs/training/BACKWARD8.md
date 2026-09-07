# W11: the guard, on a real model — and the tape's lifetime made finite

`docs/training/BACKWARD7.md` §10 is a list of seven things the eager-recorder round did not establish. Three
of them were about whether the thing works rather than whether it was built, and this document is
those three, run.

The headline is row 3, and it did not come out the way §10 predicted on either half.

Environment: worktree `work/eager2` on `develop` `479b3cf`, torch 2.13.0, transformers 5.15.1,
`/Volumes/macMini/caches/spike-venv/bin/python`,
`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-eager2`. Every shim reading printed `shim`;
every upstream reading was taken with `env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`. Nine agents were
on the machine, so **no time is reported anywhere below** — §4.1 is about what happened when one
was.

---

## 0. Answers, before the evidence

| question | answer |
|---|---|
| **Does the guard refuse SmolLM2-135M with `use_cache=True`?** | **No, and §10's reason for expecting it to was wrong about the mechanism.** transformers' `DynamicCache` grows by `torch.cat`, which *allocates*. There is no in-place write for a storage guard to see, so "a KV cache update" is not the mutation shape §10 assumed. Prefill and decode record 1 720 nodes each, `_eager_reason()` stays `None`, and the gradient of the tied embedding agrees with 2.13.0. §2.1 |
| **Does it refuse a BatchNorm in `train()` mode?** | **It did, and the refusal was the guard's own fault.** Not `poison_on_write_to_recorded_storage` — that never fired, `_eager_reason()` was `None` — but W10a's constant freshness, **one version late, on the op's own write.** `aten.rs` stamps the constant in `eager_record` and bumps it in `note_mutation` one line later, for a write the kernel had already made. A training-mode BatchNorm invalidated its own tape on its own call. §2.2 |
| **Fixed?** | **Yes, twice, because there were two walls.** `Recorder::forgive_own_write` counts that write once; behind it stood the fact that `tape.rs` had no `native_batch_norm` rule at all, so the fixed guard would have answered *"no derivative rule"*. Both are closed: a real `nn.Sequential(Linear, BatchNorm1d, Tanh)` in `train()` mode now differentiates, 94 gradient elements over two modes, max absolute difference `1.5e-08`. §2.3, §2.5 |
| **Is randomness on the eager path actually safe?** | **Yes, and the test is not vacuous.** The eager backward multiplies by the very mask tensor the forward returned, so the gradient has a closed form in that mask and the comparison is exact. The nullification is real machinery, not a hypothesis: the **capture** tape differentiates the same program by replaying it, redraws, and the same comparator rejects it on almost every element. §3 |
| **What bounds the tape?** | A **bound with a named refusal, and a release**: 100 000 nodes, derived from a measured 1 720-node prefill, after which the tape refuses by name *and drops every value it holds*. Refusing while still holding the bytes would protect nothing. Demonstrated firing on real SmolLM2 mid-decode, releasing 15.6 MiB, and leaving the process able to keep running. §4 |
| **Did the forward move?** | **No.** The three `docs/training/BACKWARD7.md` §7 digests reproduced from the recipe in the tree. §5 |

---

## 1. What was run, and against what

Nothing below is read off `capture.rs`. Every claim about the guard firing or not firing comes from
a forward on a model that a `from_pretrained` or an `nn.Module` constructor built.

| program | why this one |
|---|---|
| `HuggingFaceTB/SmolLM2-135M`, float32, `use_cache=True`, prefill `S=6` then one decode step, `loss = logits²·sum()`, gradient of the tied embedding | §10 row 3's first named case. Parameters left exactly as `from_pretrained` hands them over — every one `requires_grad=True`, which is why the tape records at all |
| `nn.Sequential(Linear(5,4), BatchNorm1d(4), Tanh())`, `train()` and `eval()`, `loss = out³·sum()` | §10 row 3's second. `Conv2d + BatchNorm2d` was tried first and reached a *third* wall — there is no `aten.convolution.default` derivative rule either — so the model was reduced to the smallest real `nn.Module` stack that isolates batch norm |
| the same decode loop, run 8 steps and never differentiated | §10 row 2 |
| `native_dropout` on 256 elements, differentiated, three independent draws | §10 row 5 |

The oracle is upstream torch 2.13.0 in the same interpreter with the shim off, element-wise.

## 2. The guard on a real model

### 2.1 The KV cache does not reach the guard at all

```
shim                                        upstream
cache type: DynamicCache                    DynamicCache
after prefill: tape 1720 reason None
after decode:  tape 3440 reason None
BACKWARD OK  grad l2 = 71039736.0           grad l2 = 71040328.0
             grad sum = 216355648.0         grad sum = 216359504.0
```

Relative difference `8.3e-06` on the norm over 28.3M accumulated elements, which is float32
summation and not a rule.

**§10 row 3 named a KV cache as a mid-forward buffer write, and it is not one here.**
`DynamicCache` extends itself with `torch.cat`, which returns a new tensor; the old key and value
buffers are never written. A guard keyed on storage versions has nothing to see. That is worth
stating as a property of transformers rather than a property of this shim — a cache implementation
that preallocated and wrote slices *would* be the shape §10 feared, and §2.4 is what would then
happen.

### 2.2 The BatchNorm refusal, and why it was not the guard §5 described

The first reading, before any change:

```
running_mean moved: True
tape size: 5
eager_reason: None                                    <-- the §5 guard did not fire
BACKWARD REFUSED: RuntimeError  ... constant 5 of this trace (torch.float32[4] on cpu) has been
  modified by an in-place operation since the trace was captured -- it is at version 2;
  expected version 1.
```

Upstream answers this program. So it is a **false refusal**, and three things about it matter:

1. **The guard `docs/training/BACKWARD7.md` §5 is about never fired.** `note_mutation`'s
   `MUTATES_WITHOUT_UNDERSCORE` arm bumps `running_mean`/`running_var` but does not call
   `poison_on_write_to_recorded_storage`, so `_eager_reason()` is `None` on exactly the program §10
   row 3 was written about. What refused is W10a's constant freshness, a round earlier and a
   different mechanism.
2. **It is stale by exactly one version, and that version is the op's own write.** `aten.rs` calls
   `eager_record` — which stamps a constant the first time `arg_of` sees it — and then calls
   `note_mutation`, which bumps `running_mean` for the write the kernel had *already completed
   before either ran*. Nothing else touched the buffer. A training-mode BatchNorm invalidated its
   own tape on its own call.
3. **It is not a policy that needs loosening.** The training-mode derivative of
   `native_batch_norm` is a function of the input, the weight and the `save_mean`/`save_invstd` the
   op *returns*. The running statistics are an output of the forward and an input to nothing, so
   forgiving that one write forgives nothing the backward reads.

### 2.3 The fix, and what it deliberately does not forgive

`Recorder::forgive_own_write` advances the recorded stamp by **one**, for the two arguments
`note_mutation` is about to bump, only when `mutates_this_call` says this call writes them.

Two decisions inside those few lines are load-bearing, and both have a test whose only job is to
tell them from the shorter alternative:

* **Add one, rather than re-read the storage's current version.** Re-reading would have been a line
  shorter and would have forgiven *everything that had ever happened to the buffer*.
  `test_the_eager_guard_still_refuses_a_second_write_to_a_batch_norm_buffer` is that difference: the
  same program with one extra `zero_()` between the forward and the backward is still refused. It is
  refused by `poison_on_write_to_recorded_storage` — the §5 guard, firing on the case it was written
  for, which is the other half of why the fix did not belong there.
* **Look the argument up in `known` rather than re-stamping from the tensor.** On the second and
  later calls `arg_of` returns the existing `Ref::Const` without touching `const_stamps`, so
  re-stamping would have swallowed every past write rather than this one.

**Nullified** (`CLAUDE.md` §5.5), by disabling the call and rebuilding:

```
breaks: test_the_eager_graph_differentiates_a_training_mode_batch_norm_that_wrote_its_buffers
        RuntimeError: ... has been modified by an in-place operation since the trace was
        captured -- it is at version 1; expected version 0
PASSES: test_the_eager_guard_still_refuses_a_second_write_to_a_batch_norm_buffer
```

The control passing under the nullification is the point: the second-write refusal does not depend
on the fix, so the two tests are not one test written twice.

### 2.4 What still refuses, and it is the right set

A genuinely second write to a storage a live tape holds — `optimizer.step()` before `backward()`, a
`zero_()` on a recorded buffer, an in-place cache written through a view. That is upstream's own
`retain_graph`/version-counter shape, and the refusal message is now the eager one:

> There is no region to capture again: this is the eager tape, and the write happened between the op
> that read this tensor and backward(). Take the gradient before the write — backward() before
> optimizer.step(), not after — or run the write under torch.no_grad() on a tensor no graph depends
> on

The previous wording came from `PyCaptureTrace` and told the caller to *"capture the region again"*,
which names a call an eager caller never made.

**The whole-tape coarseness §10 row 3 also complains of is unchanged and is still a real gap.** A
write to a storage no gradient would have reached still refuses the whole tape. What this round
establishes is that the coarseness was not what was refusing real models — the off-by-one was — and
narrowing it would not have fixed batch norm, because `running_mean` *is* an operand of a node the
backward needs. It is left standing rather than widened on a guess.

### 2.5 The second wall: there was no `native_batch_norm` derivative rule

With freshness fixed, the same program answered `no derivative rule for
aten.native_batch_norm.default`. Fixing one of two walls buys nothing, so `tape.rs` gained the rule.
Two modes, and they are different functions:

* **`training=true`** — the statistics are the batch's own, so `x` reaches the output through the
  mean and the variance as well as directly. `dL/dx = invstd·(gh − mean(gh) − x̂·mean(gh·x̂))`,
  `gh = g·w`. It reads `save_mean`/`save_invstd` and **not** the running buffers, which is what makes
  §2.3's forgiveness safe.
* **`training=false`** — the statistics are constants and the rule is one multiply. Here the running
  buffers *are* read, and in this mode the op writes nothing, so they are held to the ordinary
  freshness rule with no forgiveness at all.

Against upstream, `nn.Sequential(Linear(5,4), BatchNorm1d(4), Tanh())`, `loss = (out³).sum()`,
gradients for the input and all four parameter tensors:

| mode | quantity | elements | max abs difference | `max\|upstream\|` |
|---|---|---:|---|---:|
| `train` | loss | 1 | `9.313e-10` | 0.0056 |
| `train` | `d/d input` | 15 | `9.168e-10` | 0.0007 |
| `train` | `d/d linear.weight` | 20 | `9.779e-09` | 0.0003 |
| `train` | `d/d linear.bias` | 4 | `1.467e-08` | ~0 |
| `train` | `d/d bn.weight` | 4 | `1.304e-08` | 0.0520 |
| `train` | `d/d bn.bias` | 4 | `1.118e-08` | 0.0958 |
| `eval` | loss | 1 | `4.657e-10` | 0.0046 |
| `eval` | `d/d input` | 15 | `8.731e-11` | 0.0005 |
| `eval` | `d/d linear.weight` | 20 | `1.164e-10` | 0.0009 |
| `eval` | `d/d linear.bias` | 4 | `2.328e-10` | 0.0027 |
| `eval` | `d/d bn.weight` | 4 | `1.164e-10` | 0.0199 |
| `eval` | `d/d bn.bias` | 4 | `1.118e-08` | 0.0909 |

**94 elements over two modes, every one at float32 rounding**, and `running_mean` moved in `train`
and did not in `eval` in both interpreters — so this is still the program §10 row 3 asked about and
not one that quietly stopped writing.

### 2.6 The oracle that could not see this, and the one that could

`test_every_tape_rule_agrees_with_central_differences_in_float64` requires a gradient case per rule,
so this rule got one. **It can only reach the eval arm**: capture refuses a training-mode
`native_batch_norm` by name (`MUTATES_WITHOUT_UNDERSCORE`, because that call writes and a trace must
stay single-assignment), and the finite-difference oracle runs on the capture tape.

That is not a footnote. Nullifying the training rule's two mean terms — the entire content of the
training arm — gives:

```
breaks: test_the_eager_graph_differentiates_a_training_mode_batch_norm_that_wrote_its_buffers
PASSES: test_the_batch_norm_rule_agrees_with_the_eval_mode_closed_form_too
PASSES: test_every_tape_rule_agrees_with_central_differences_in_float64
```

**The oracle this repository trusts most for derivative rules cannot see a training-mode batch-norm
defect at all.** The closed-form eager test is what catches it, and it catches it because `g = 1`
makes both mean terms cancel the direct term exactly, so a rule missing either is `O(1)` wrong on a
quantity that should be zero. Where a rule has a mode the capture tape refuses, the eager tape is
now the only place it can be checked — that is new, and it is an argument for the eager recorder
that `docs/training/BACKWARD7.md` did not have.

## 3. Randomness: the argument, tested

`docs/training/BACKWARD7.md` §10 row 5: the eager gating means a dropout mask is *held*, not redrawn, so
`docs/graph/CAPTURE.md` §9-1's failure — a gradient taken at a different draw than the one reported —
structurally cannot occur; and that argument was not tested, because the two RNG streams make a
direct upstream comparison impossible.

The streams do not have to agree. `native_dropout` returns the mask it drew, so the gradient has a
closed form *in that mask*:

    y = x·mask/(1−p);  loss = sum(y²);  dloss/dx = 2·x·mask²/(1−p)²

and the comparison is **exact** — the backward multiplies by the very tensor the forward returned,
so there is no rounding to allow for. Three independent draws, 256 elements each, all exact; the
draws differ from each other on well over an eighth of their elements, so no iteration was vacuous.

**The nullification is the part that makes this a test.** A comparator that accepted any mask would
have accepted a redraw, so something in the tree has to actually produce one and be rejected.
Something does: the **capture** tape differentiates by replaying the forward, `native_dropout` draws
a second time (`test_the_tape_replays_a_dropout_forward_and_therefore_redraws_its_mask`;
`docs/models/ADAPT.md` §14.3 measures it as the dominant error term on a real gpt2 Tent step, by two orders
of magnitude). Running the same program through capture and applying the same comparator disagrees
on almost every element. `docs/graph/CAPTURE.md` §9-1's failure is live in this repository on one path and
absent on the other, and the test now says so with both.

## 4. The tape's lifetime: a bound, a refusal, and a release

`docs/training/BACKWARD7.md` §10 row 2: the tape is freed by `backward()` and by nothing else, so a forward
that is never differentiated retains every intermediate, and `_eager_reset()` was the only escape.

Measured, real SmolLM2-135M, `use_cache=True`, greedy decode, bound nullified so the loop runs:

| | nodes | tape |
|---|---:|---:|
| prefill, `S=6` | 1 721 | 13.19 MiB |
| + 1 decode step | 3 442 | 15.62 MiB |
| + 8 decode steps | 15 489 | 33.93 MiB |
| the same prefill under `torch.no_grad()` | **0** | **0.00 MiB** |

1 721 nodes and about 2.6 MiB a step, without limit.

**One correction to §10 row 2, and it changes the argument rather than the conclusion.** §10 says
upstream is bounded here for free. In *this* program it is not: `DynamicCache` holds every step's
key and value, each carrying a `grad_fn`, so upstream retains its graph too — measured at about
3.7 MiB a step before the machine took the reading away (§4.1). The difference is **ownership, not
bookkeeping.** Upstream's graph hangs off tensors, so dropping the cache frees it and `no_grad`
never builds it. This tape is a thread-local that outlives every output. Saying "upstream is flat"
would have been a false contrast; the true one is that upstream's leak has an owner and this one
had none.

**The contract chosen is a bound with a named refusal and a release**, not a documented requirement:
a silent unbounded leak in a library that exists for on-device inference is the wrong default, and a
requirement that is only written down is a silent leak with a paper trail. Not a bare refusal
either: a tape that refuses while still holding the bytes protects nothing, so tripping the bound
drops every value it holds at that moment rather than at the moment the caller notices.

`EAGER_MAX_NODES` defaults to **100 000**, derived rather than picked: the largest single forward
this project runs is a 1 721-node SmolLM2-135M prefill, so the bound is **58×** the biggest graph a
`backward()` here has ever had to hold, and no forward-and-backward can reach it by accident. What
it does reach is the runaway — about 58 decode steps and 170 MiB. Reaching it means the caller meant
`torch.no_grad()`, because a loop that intended to differentiate would have called `backward()` and
freed the tape.

On the real model, with the bound lowered to 5 000 so the demonstration is affordable:

```
prefill nodes 1721  tape 13.19 MiB  reason None
step 0  nodes 3442  tape 15.62 MiB  refused=False
step 1  nodes 0     tape 0.00 MiB   refused=True
  MESSAGE: the eager graph grew past 5000 nodes without ever being differentiated. ...
           The values held so far have been released. Run the loop under torch.no_grad(),
           which records nothing at all, or call torch._C._eager_reset() each iteration,
           or raise the bound with torch._C._eager_set_max_nodes(n)
after the refusal, a no_grad forward still runs: (1, 6, 49152)  nodes 0
```

The release is visible in the same line as the refusal, and the process is still able to work
afterwards — which is the whole reason to refuse in the running process rather than let the phone
decide.

Nullified two ways, and the failures name different things:

```
release_values() removed         breaks: ... AssertionError: the bound refused but kept the
                                         values it refused over, which is the leak it exists to stop
default changed to 0 (unbounded) breaks: ... AssertionError: the default bound moved -- 100000 is
                                         derived from a measured 1720-node prefill
```

### 4.1 Why the memory figure is the tape's own and not RSS

RSS is not an instrument on this machine, and the first two attempts to use it are worth recording
because each failed differently and each *looked* like an answer.

* **`ru_maxrss` reported `+0.0 MiB` for all eight decode steps**, on a tape that grew by fourteen
  thousand nodes. It is a peak, and the peak had already been set while loading the weights.
* **`ps -o rss=` reported `+3.4, +7.2, +11.0` MiB and then `−598.9 MiB`** between two consecutive
  steps. Eight other agents were running; the pages were reclaimed underneath the reading.
  `docs/training/BACKWARD7.md` §8 already records this machine corrupting a measurement, which is why the
  reaction was to change the instrument rather than to re-run until the number looked nice.

`_eager_tape_bytes()` is what replaced it, and **the naive form of it is a fiction**: summing
`numel × itemsize` over every recorded result reported **529 MiB for a single prefill** and 516 MiB
per decode step. Two corrections, both found by measuring without them:

* **Deduplicate by storage.** Every `Linear` records a `t()` of its weight, and that result is a
  *view*: a distinct tensor over storage already counted.
* **Exclude constant storages.** That transpose does not merely double-count, it counts *the model's
  parameters* — the `lm_head` transpose alone is a `[49152, 576]` view, 108 MiB of buffer that
  `from_pretrained` owns and that freeing the tape would not return.

What is left is an upper bound on what dropping the tape would give back, and it agrees in order
with what RSS could be got to say before it stopped making sense.
`test_the_tape_byte_count_excludes_parameters_and_counts_each_storage_once` pins both corrections in
miniature — a transpose of a leaf contributes 0, a view of a recorded result contributes nothing
beyond what it aliases — and removing the constant exclusion turns it red at the same 256 bytes the
`lm_head` view scales up to 108 MiB.

## 5. The forward did not move

This round edits `record_into`, so the measurement is the evidence rather than an argument about
compile inputs. `docs/training/BACKWARD7.md` §7's recipe is in the tree — sha256 over
`struct.pack("<%df" % len(row), *row)` for each row of `logits.tolist()[0]`, real
`HuggingFaceTB/SmolLM2-135M`, float32, greedy prefill of `torch.arange(S).unsqueeze(0)` — and the
script was written from it rather than copied.

| S | sha256, this build | |
|---:|---|:--:|
| 6 | `606e00d1be05fccce84b23b02dae8f473d490ad0f706b9b612036dd55cee531b` | ✅ |
| 32 | `62d923807cce47cee35a5e7b6091c9544e180c6c9ff771bb4cad3f478f4cbb79` | ✅ |
| 128 | `4750428e94f7383c42853b0d547192a7946fdde9e086d3b16cebdeb2ca68e37d` | ✅ |

All three equal `docs/training/BACKWARD7.md` §9.1's, and through it `docs/training/BACKWARD6.md` §7's,
`docs/training/BACKWARD5.md` §6.5.1's and `docs/training/BACKWARD4.md` §6.2's — a five-round chain, one link of which
was taken against a deliberately reverted build.

The reason the forward *can* be unmoved: everything added here is behind `on_graph &&
eager_enabled()`, and a prefill under `torch.no_grad()` never reaches the first operand.

## 6. Gates

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh
    422 ok, 0 FAIL
    DOCWATCH: PASS -- 417/417 evaluated marker(s) hold
    EXIT=0

TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib $PY tools/golden/compare.py
    SUMMARY: 8681/8681 cases passed, 0 failed, ops covered=207, pending case builders=0
    EXIT=0
```

`ops=207` is unchanged on purpose: `native_batch_norm` already had a kernel, and what landed here is
its *derivative*, which the golden harness does not count.

## 7. Reported by kind, not by count

`CLAUDE.md` §5.3: test counts are not progress, so this is split.

| kind | what |
|---|---|
| **defect fixed** | `Recorder::forgive_own_write` — a training-mode BatchNorm refused its own forward, one storage version late, on its own write |
| **feature added** | `native_batch_norm` derivative rule (`tape.rs`), both modes |
| **feature added** | `EAGER_MAX_NODES` — a bound on the eager tape, with a named refusal and a release; `_eager_max_nodes` / `_eager_set_max_nodes` |
| **feature added** | `_eager_tape_bytes()` — the tape's own size, deduplicated by storage and excluding parameters |
| **wording fixed** | the eager freshness refusal no longer tells the caller to capture a region they never opened |
| **tests added** | 5: training batch norm, second-write control, eval closed form, byte count, dropout nullification (extends an existing test) |
| **documentation corrected** | §10 row 2's *"upstream is bounded here for free"* — in a `use_cache` decode loop upstream retains its graph too, and the difference is ownership |

## 8. What this document still does not establish

| # | not established | why |
|---|---|---|
| 1 | **That the whole-tape coarseness of the storage guard has no false refusals left.** | §2.4. It still refuses the whole tape for a write to a value no gradient would have reached. This round shows that was not what refused real models, not that it never will |
| 2 | **A convolution derivative.** | Reaching batch norm through a real `Conv2d` needs one, and `nn.Sequential(Conv2d, BatchNorm2d, ReLU)` is refused today by `no derivative rule for aten.convolution.default`. The batch-norm model here is `Linear`-based for that reason |
| 3 | **That 100 000 is the right bound rather than a defensible one.** | §4 derives it from one model's prefill. A training step on longer sequences or a larger model would move the numerator; nothing here measures where |
| 4 | **Anything about `run_backward`, `.grad`, hooks, `create_graph` or `torch.autograd.Function`.** | `docs/training/BACKWARD7.md` §6 is still the list, and `Tensor.backward()` was explicitly out of scope this round |
| 5 | **Any per-dispatch cost.** | Nine agents were on the machine. §10 row 1 stands unchanged, and this round deliberately reports no time at all |
| 6 | **A preallocated, slice-written KV cache.** | §2.1 measures the cache transformers actually ships. A cache that wrote in place would be §2.4's shape, and no such implementation was run |

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs forgive_own_write present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs EAGER_MAX_NODES present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs release_values present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs eager_tape_bytes present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs batch_norm_backward present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_eager_graph_differentiates_a_training_mode_batch_norm_that_wrote_its_buffers present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_eager_guard_still_refuses_a_second_write_to_a_batch_norm_buffer present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_batch_norm_rule_agrees_with_the_eval_mode_closed_form_too present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_tape_byte_count_excludes_parameters_and_counts_each_storage_once present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_eager_tape_refuses_and_releases_when_it_grows_past_its_bound present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_eager_backward_uses_the_dropout_draw_the_forward_made present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_eager_graph_survives_a_kv_cache_update_because_the_cache_is_concatenated present -->
<!-- DOCWATCH: op-implemented aten.native_batch_norm.default -->
<!-- DOCWATCH: count smoke_ok ge 415 -->
<!-- DOCWATCH: count golden_cases_passed ge 8681 -->
