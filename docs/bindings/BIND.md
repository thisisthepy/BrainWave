# BIND — the Python argument-binding layer

What was hot in `rust/torch_c/src/bootstrap.py`'s overload resolution, what was
precomputed, why that is safe, and what is still slower than upstream.

**This is `rust/torch_c/src/bootstrap.py`, and Android loads the same file** —
it is embedded in the artefact `vendor/install_shim.sh` installs, and the
Android build embeds that same source. Nothing here is host-specific: it is
plain Python doing dict and attribute work, so the win applies on device too,
where the interpreter is slower and therefore the share it occupies is larger.
It was measured on the host only, because that is where it can be measured
against upstream on the same machine.

---

## 1. The problem

The kernels are already faster than upstream at the shapes SmolLM2-135M
actually uses, yet the model is slower. Measured by the coordinating session on
a quiet machine (load 1.4, no other agents):

| | upstream | ours | ratio |
|---|---|---|---|
| SmolLM2-135M float32 prefill | 35.3 ms | 39.8 ms | **1.13x slower** |

Per kernel, at the model's real shapes (µs/call):

| kernel | upstream | ours | ratio |
|---|---|---|---|
| `_softmax` (attention shape) | 16.25 | 2.39 | 0.15x |
| `native_layer_norm` | 12.25 | 6.13 | 0.50x |
| `silu` | 11.23 | 6.19 | 0.55x |
| `sdpa` | 26.77 | 22.24 | 0.83x |
| `linear` (MLP) | 90.49 | 89.50 | 0.99x |

Weighted by call count that is **−1.17 ms**; the model showed **+4.5 ms**.
Dispatch itself is also ours-faster (0.86 vs 1.11 µs per small-tensor call).
The loss was in the Python surface:

| | upstream | ours | ratio |
|---|---|---|---|
| `.view()` | 0.76 µs | 4.97 µs | 6.5x |
| `.transpose()` | 0.82 µs | 4.02 µs | 4.9x |
| attribute reads (`.shape`, `.dtype`, `.dim()`) | — | — | comparable |

`cProfile` over 5 forward passes, before:

```
ncalls   tottime  function
  9275    0.166   torch._C._aten_dispatch
  8255    0.025   bootstrap.py:1889 _bind
 35030    0.024   bootstrap.py:1677 _decompose_type
105090    0.009   str.endswith          <- 21018 per forward pass
 49890    0.004   str.strip
 15610    0.010   bootstrap.py:1742 check
 14535    0.006   bootstrap.py:1760 coerce
```

A microbenchmark of `t.view(1, 6, 576)` and `t.transpose(1, 2)` at the model's
shape isolates it further. Of the 0.800 s that 40 000 such calls cost,
`_aten_dispatch` — everything the shim actually computes — was **0.030 s, under
4%**. The other 96% was the Python layer deciding which overload to call.

---

## 2. Method

Backends alternate inside one script run, so a drift in machine state moves
both, and only the **ratio to upstream** is reported. `uptime` was recorded
before and after every measurement and no measurement was taken above load 2.4.

```sh
export HF_HOME=/Volumes/macMini/caches/hf-home
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-bind
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
# ours:     PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1
# upstream: no PYTHONPATH
```

* **prefill** — SmolLM2-135M, `dtype=torch.float32`, prompt of 6 tokens; the
  minimum of 5 timed passes after 2 warmups; 4 alternating rounds.
* **microbench** — minimum of 5 blocks of 20 000 calls after 200 warmups, on a
  `(1, 6, 9, 64)` float32 tensor.

**The "before" row was re-measured, not reused.** The first baseline reading
was taken at load 3.4 and gave 1.149; the pre-change artefact was rebuilt from
`git show HEAD` and re-measured under the same conditions as the "after" row.
That is the number in the table.

---

## 3. What changed

Nothing in the resolution *algorithm*. Every step below moves a computation
from call time to table-parse time, or removes one whose answer was already
known.

### 3.1 `_decompose_type` was the wrong thing to cache

The obvious move — `functools.lru_cache` on `_decompose_type` — is in the diff,
and it is **not where the win came from**. Measured after the rest of the work:

```
after import:          hits=1863  misses=28  distinct=28
after 4000 op calls:   hits=0     misses=0
```

Twenty-eight distinct type spellings exist in the whole of `overloads.json` and
`methods.json`. Caching turns 35 030 parses per forward pass into 35 030 dict
lookups; **hoisting the call out of the loop turns them into zero.** The memo is
kept because it makes `import torch` do 1863 fewer parses and because
`_SchemaType.isSubtypeOf` and `containedTypes` — the fake-tensor and prims path,
which this benchmark does not reach — still call it per question. But it is an
import-time and elsewhere win, not the one being reported.

It is safe to memoise: the answer depends on nothing but the characters of the
string, there is no context in which `"int[1]?"` means two different things, and
the returned tuple is immutable so no caller can corrupt the next one's entry.
The bound (`maxsize=4096`) is there because `parse_schema` accepts text from
outside the tables.

### 3.2 `_SchemaPlan` / `_ArgPlan` — everything fixed, computed once

`_bind` re-derived all of this on every call, from data that is settled the
moment `_Schema.parse` returns and that nothing ever writes to again:

| was | now |
|---|---|
| `[a for a in schema.arguments if not a.kwarg_only]` | `plan.positional` |
| `{a.name: a for a in schema.arguments}` | `plan.by_name` |
| `_decompose_type(str(arg.type))` in `check`, once per argument per call | `_ArgPlan.base` / `.is_list` / `.optional` |
| `_decompose_type(str(arg.type))` again in `coerce` | `_ArgPlan.sized_int_list` |
| `_decompose_type(str(positional[skip].type))` for the varargs rule | `plan.varargs_intlist` |
| `1 if self.self_bound else 0` | `_Overloads._skip` |
| `arg.has_default_value()` | `_ArgPlan.has_default` |
| `zip(self.schemas, self.keys)` per `resolve` | `_Overloads._candidates` |

`check`'s "a sized int list also accepts a bare int" precondition and `coerce`'s
"widen it to a one-tuple" precondition are the *same* predicate on the type, so
they collapse into one precomputed flag.

### 3.3 The twelve-way type chain, compiled per argument

`_TypeChecker._base` walks up to twelve string comparisons to decide what test
to run, every call. `predicate_for` returns a closure that has already made that
decision. `check`, `coerce` and `_base` are kept **unchanged** as the readable
statement of the rules — including the three that would be got wrong by
intuition (`bool` does not satisfy `int`; `int` does satisfy `float`; a 0-dim
tensor satisfies `Scalar`).

The predicates are built lazily, on the first bind against that schema, for the
same reason `_TypeChecker` itself is: `layout` and `memory_format` do not exist
when the tables are parsed. Once built they are fixed, which was already true of
the attributes `check` reads — `_TypeChecker.__init__` snapshots them.

### 3.4 Four passes that could not fail, or ran when they had nothing to do

* `all(base_ok(item) for item in value)` → an explicit loop. The generator costs
  a frame per element plus one to stop, and shape lists are the hottest thing
  the checker sees. It short-circuits where `all` does.
* The "was every required argument bound?" walk is skipped when
  `len(bound) == plan.n_arguments`: `bound`'s keys are always argument names, so
  equal counts means equal sets. (A schema with a repeated argument name has
  fewer distinct names than arguments, so the counts cannot match and the walk
  still happens — the skip fails safe.)
* The "drop arguments equal to their own default" pass built a second dict to
  hold exactly what the first held, for every schema with no defaults —
  `view(Tensor self, SymInt[] size)` and every other pure-shape schema.
  `plan.any_defaults` returns `bound` directly in that case.
* The "given twice" check moved out of the positional loop and behind
  `if kwargs:`, so a call with no keywords does not pay a dict lookup per
  argument. Both spellings answer `None` for the same calls and neither has a
  side effect, so which of the two reasons is found first is not observable.
* `_strip_python_only_kwargs({})` is `{}`, and `**kwargs` already handed the
  caller a fresh dict, so the call is skipped when there is nothing to strip.
* `type(value) is int` is tried before `isinstance(value, int) and not
  isinstance(value, bool)`. The first implies the second and is true of nearly
  every value that reaches there; the full test still decides everything else.

---

## 4. Why this is safe

Two arguments, one by construction and one by measurement.

**By construction.** Every value moved out of the call path is a pure function
of the parsed schema, and the parsed schema is written once. `_Schema.parse`
fills `arguments` and nothing in the file writes to an `_Argument` afterwards;
`_Overloads.__init__` is the only constructor of `_SchemaPlan`. There is no
context in which the same argument decomposes two ways, which is the failure
mode a cache has to be checked against. The one genuinely late-bound thing —
the type predicate, which needs `layout` and `memory_format` — is still built
late, on first call.

**By measurement.** The pre-change `_bind` was extracted verbatim and run beside
the new one, driven off the same parsed schemas, over **every installed entry**
(124 of them, recovered from the closures the install actually left behind) and
6613 call shapes — positional tuples up to length 3 drawn from 18 values that
exercise every rule the checker has, plus six keyword shapes. Any difference in
refusal, in chosen key, in bound keys, or in a bound value is a behaviour
change.

```
{"entries": 124, "comparisons": 1679702, "mismatches": 0}
```

That harness can fail. Tampering with each precomputed field in turn, after the
plans were built:

| tamper | mismatches |
|---|---|
| `plan.any_defaults = False` | 271 |
| `plan.varargs_intlist = False` | many (first at `torch.view` with `size=(-1,-1)`) |
| `plan.required = ()` | many (first at `torch.masked_select` with no arguments) |
| `_ArgPlan.sized_int_list = False` | 84 |
| `plan.n_arguments = 1` | 788 |
| `plan.n_arguments = -1` | **0** |

The last one is not a hole: `-1` makes `len(bound) != n_arguments` always true,
so the required walk always runs. That is the conservative direction — correct,
just slower — which is what the skip is designed to fail into.

The repository's own gates, run on the final artefact:

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh   -> 197 ok, exit 0
$PY tools/golden/compare.py                 -> 2811/2811, ops covered=119, exit 0
$PY rust/torch_c/pytests/verify_schemas.py  -> 4203/4203, exit 0
```

Unchanged from before the work, which is the point.

---

## 5. Result

Load 1.8–2.4 throughout; the pre-change artefact was rebuilt and re-measured
under the same conditions rather than reusing the earlier, noisier reading.

### Prefill, ratio to upstream

| | upstream (ms) | ours (ms) | ratio |
|---|---|---|---|
| before | 35.912 | 41.067 | **1.1435** |
| after | 35.933 | 38.096 | **1.0602** |

Pairwise across the four alternating rounds: before 1.1340 / 1.1444 / 1.1486 /
1.1507; after 1.0572 / 1.0587 / 1.0597 / 1.0634. The two bands do not overlap.

**2.99 ms of the 5.16 ms gap is gone — 58% of it.** (Gap before
41.067 − 35.912 = 5.155 ms; after 38.096 − 35.933 = 2.163 ms.)

#### What the residual is, and is not, resolved to

Three independent runs put the *after* ratio at 1.019, 1.033 (median 1.021) and
1.060. The improvement is not in doubt — the before and after bands above do not
overlap, and every reading lands far below 1.1435. **The residual itself is not
resolved to a percent, and this document should not be read as claiming 1.06
exactly.**

The reason is the noise floor rather than a disagreement about method. A
four-round re-measurement taken at load 3.2 scattered pairwise 0.915 / 1.012 /
1.039 / 1.045 — one round came out *faster* than upstream. A spread that
straddles 1.00 cannot resolve a 2–6% residual, so the correct summary is a
range: **within a few percent of upstream on desktop CPU, from 14% behind.**

This machine cannot be made quiet on demand. The load is a windowing server, a
user application, and two Android emulators that are shared with other projects
and must not be killed. Pinning the residual would need either a quiet machine
or enough rounds to average the interference out; neither was done, so the range
stands. The Android numbers in §7 are less affected because the effect there is
1.8–2.75x — an order of magnitude above the same noise.

### Microbench, µs/call

| | upstream | before | after | still |
|---|---|---|---|---|
| `.view(1, 6, 576)` | 0.796 | 5.212 | **1.790** | 2.25x |
| `.transpose(1, 2)` | 0.821 | 4.039 | **1.460** | 1.78x |
| `.view((1, 6, 576))` | 1.034 | 6.363 | **1.944** | 1.88x |
| `t + t` | 1.031 | 3.744 | **1.943** | 1.88x |
| `.shape` | 0.120 | 0.078 | 0.079 | 0.66x |
| `.dim()` | 0.061 | 0.046 | 0.045 | 0.74x |

Function calls for those 40 000 method calls: **3 300 002 → 500 002**.
Profiled time 0.800 s → 0.145 s. `_aten_dispatch`'s share of it went from
under 4% to 19%.

### The forward pass, profiled

| | before | after |
|---|---|---|
| total function calls (5 passes) | 486 795 | 146 000 |
| `bootstrap.py` `tottime`, summed | 0.084 s | 0.024 s |
| `_aten_dispatch` `tottime` | 0.170 s | 0.170 s |
| `_decompose_type` calls | 35 030 | **0** |
| `str.endswith` calls | 105 090 | 0 |

`_aten_dispatch` did not move, which is the check that this touched only the
Python layer.

---

## 6. What is still slower, and why

**`.view()` is still 2.2x upstream.** The gap that remains is structural, not
redundant work:

1. **Three Python frames per call** — `method` → `resolve` → `_bind` — where
   upstream has none. Merging them would save perhaps 0.1 µs per call
   (~0.15 ms per prefill) at the cost of a second copy of the resolution loop.
   Not taken: a duplicated loop that can drift out of step with the original is
   a worse trade than 0.15 ms.
2. **`dispatch(key, **bound)`** — the bound arguments are built as a dict and
   then unpacked into keyword arguments for the C entry point, which re-parses
   them. Removing that round trip means changing `_aten_dispatch`'s signature in
   `aten.rs`, which is outside this work's area.
3. **`view` costs two binds.** `view.dtype` is tried and refused before
   `view.default` binds. The refusal is now cheap (a length compare) but it is
   still a Python call. The order is upstream's and is part of the answer, so it
   cannot be reordered.

**Roughly 2.2 ms of prefill gap remains.** Attributing it: about 1855
dispatches per prefill at the ~0.7 µs per-call Python overhead the microbench
still shows is ~1.3 ms, which leaves ~0.9 ms not explained by this layer. That
residue was not chased — it is on the C side, and this work was scoped to
`bootstrap.py`.

**Two things were tried and did not help.**

* **`functools.lru_cache` on `_decompose_type`** — the brief's first candidate.
  It is a real win at import (1863 parses avoided) and for the prims path, but
  **zero** on the call path, because the right fix was to stop calling it. Kept
  for the other two reasons, not counted toward the result. §3.1.
* **Fusing `check` and `coerce` into one value-or-sentinel closure.** One fewer
  call and one fewer attribute load per bound argument, and it measured *within
  noise* — view 1.84 vs 1.80 µs, transpose 1.52 vs 1.51, identical function-call
  counts (500 002 either way). It needed a second copy of the twelve `_base`
  rules with a different return shape; two spellings of "a zero-dim tensor
  satisfies `Scalar`" is a real hazard bought with no measurable time, so it was
  reverted. The bookkeeping it *did* pay for — hoisting the "given twice" check
  out of the positional loop — was kept.

**Not measured on device.** The reasoning that this transfers to Android is that
the same `bootstrap.py` is embedded there and the change is pure interpreter
work with no host-specific path, so the share it occupies is if anything larger
on a slower CPU. That is an argument, not a measurement.

---

## 7. Android measurement (in progress)

Turning §6's closing claim into a number. Device: `emulator-5556`, API 36,
arm64-v8a. There is no upstream torch for Android (no wheel is published,
which is a large part of why this project exists), so this section reports
**old/new ratios on the same device**, not a ratio to upstream.

### 7.1 `bootstrap.py` is baked into the artefact, not loaded from disk

`rust/torch_c/src/lib.rs:568` does
`std::ffi::CString::new(include_str!("bootstrap.py"))` — the source text is
compiled into `lib_C.so` at Rust build time. Swapping the `.py` file on the
device without rebuilding does nothing; the interpreter never reads a
`bootstrap.py` file at all on either platform. **Both sides were rebuilt** for
`aarch64-linux-android` via `scripts/device_android.sh build`:

* new (HEAD, `972dfe4`): `rust/torch_c/src/bootstrap.py` unchanged, built as-is.
* old (`972dfe4^`): `git show 972dfe4^:rust/torch_c/src/bootstrap.py` copied
  over `rust/torch_c/src/bootstrap.py`, built, then the working tree file was
  immediately restored from a `cp` backup (`git status --short` confirmed a
  clean diff afterward).

Both `.so` artefacts were saved to `/tmp/bw_bind_android/lib_C.{old,new}.so`
(distinct md5s) before restoring the tree, so rounds alternate by swapping the
staged file rather than rebuilding per round: `_C.abi3.so` is the only file
that differs between old and new, so once the rest of the tree (CPython
runtime + vendored `torch` + deps) is staged once via
`scripts/device_android.sh stage`, alternation is a single `adb push` of the
5.4 MB `.so` to `/data/local/tmp/bw_device/site/torch/_C.abi3.so` — no re-stage
of the ~440 MB tree per round.

### 7.2 Method

Same shapes as the host (§2): a `(1, 6, 9, 64)` float32 tensor, 200 warmups,
minimum of 5 blocks of 20 000 calls per round. Three microbenchmarks:
`.view(1, 6, 576)`, `.transpose(1, 2)`, and `t + t` (the dispatch-bound loop —
smallest possible op, so the Python binding layer dominates the per-call cost).
Rounds alternate old, new, old, new, ... and the minimum per side across
rounds is reported, per the host methodology. A control round swaps the *same*
artefact in for both labels to check the harness itself reads ~1.00x.

Load stayed 2.27–2.78 for every measured round (`uptime`, checked immediately
before each push and each run); no round was taken above that. The one 3.55
reading during an earlier import smoke-test (not a timed round) was discarded
and re-checked before measurement began.

### 7.3 Result

Minimum of 3 alternating rounds per side, µs/call:

| | old (`972dfe4^`) | new (`972dfe4`) | ratio old/new |
|---|---|---|---|
| `.view(1, 6, 576)` | 9.172 | 3.331 | **2.75x** |
| `.transpose(1, 2)` | 6.891 | 2.742 | **2.51x** |
| `t + t` (dispatch-bound) | 6.164 | 3.418 | **1.80x** |

All three rounds per side agreed within ~1.3% of their own minimum (e.g. view:
9.172 / 9.204 / 9.249 old; 3.331 / 3.412 / 3.421 new) — tight enough that
old and new do not overlap on any metric.

**Control** (same `new` artefact pushed and measured twice, under labels A and
B, through the identical harness): view 1.038x, transpose 1.033x, add 1.033x.
All three land at 1.03–1.04x, not exactly 1.00 but an order of magnitude below
the 1.80–2.75x the real comparison shows, and it sets the noise floor the
above spreads (~1.3%) are already comfortably inside of. Harness bias is not
what is producing the win.

**The win transfers, and transfers more strongly than the host measurement.**
Host (§5) showed view 2.25x-still-slow-but-improved (5.212 → 1.790 µs, a 2.91x
speedup) and transpose 4.039 → 1.460 µs (2.77x); device shows 2.75x and 2.51x
on the same two calls, plus 1.80x on the dispatch-bound `t + t` the host table
also carries (3.744 → 1.943 µs there, 1.93x — comparable). The device ratios
land in the same range as the host's, not a different regime, and if anything
the interpreter-bound share is if anything larger here: absolute per-call
times are 2–3x the host's on both sides (e.g. new-view 3.33 µs on device vs.
1.79 µs on host), consistent with §7's opening argument that a slower
interpreter makes the Python layer's share larger rather than smaller. **This
device measurement supports, rather than merely assumes, the transfer.**

### 7.4 Method notes

* **Rebuild required, `.py` swap alone does not work** (§7.1) — both artefacts
  were built via `scripts/device_android.sh build` for `aarch64-linux-android`,
  saved to `/tmp/bw_bind_android/lib_C.{old,new}.so` (verified distinct md5),
  and `rust/torch_c/src/bootstrap.py` was restored to HEAD (`cp` backup, not
  `git checkout`) immediately after the old build — `git status --short` was
  clean on that file before device rounds began.
* Only `/data/local/tmp/bw_device/site/torch/_C.abi3.so` was swapped between
  rounds (direct `adb push`, not a full re-stage) — the CPython runtime,
  vendored `torch` tree and dependencies were staged once via
  `scripts/device_android.sh stage` and are identical across all rounds; the
  `.so` is the only variable.
* `_multiprocessing`/`_posixshmem` stubs from `scripts/device_parity.py`
  (`_install_android_stubs`, gated on `BW_STUB_MULTIPROCESSING=1`) were copied
  into the microbenchmark script — `torch/multiprocessing/__init__.py` imports
  `multiprocessing.resource_tracker` unconditionally and Android's CPython
  ships neither extension; without the stub `import torch` fails before any
  timing runs.
* Raw per-round JSON and build/stage logs are under `/tmp/bw_bind_android/`
  (`out_{old,new,controlA,controlB}_r*.json`, `build_{old,new}.log`,
  `stage_new.log`) for anyone who wants to re-check the arithmetic above.

---

## 8. Round 3 — merging the per-candidate parse into `resolve`

Picks up from docs/design/DISPATCH.md §6, which named `resolve` + `_bind` as the
largest item left and sized it at **~1.5 ms per forward pass, "five times
everything round 2 removed"**. That figure was derived before round 2 landed,
by multiplying a per-call cost by the dispatch count. It is roughly **twice**
what a direct measurement finds, and §8.1 is why.

### 8.1 The profile, as actually found

Counted rather than timed, so the machine's load cannot move it — one
SmolLM2-135M float32 prefill, probing `_Overloads._bind` and `.resolve` (both
are looked up on `self` per call, so unlike `_aten_dispatch` they really can be
wrapped; DISPATCH.md's spy warning applies to the door, not to these):

| per forward pass | count |
|---|---|
| `_C._aten_dispatch` calls | 1855 |
| …of which reach `resolve` | **1188 (64%)** |
| `_bind` calls (candidates tried) | 1651 |
| …refusals | 463 (28% of attempts) |
| `resolve` calls passing any keyword | 183 (15%) |
| arguments bound | 2375 |

**36% of dispatches never touch the overload machine at all.** They are the
hand-written paths — `to`, `__getitem__`, the scalar, softmax and indexing
installers — which call the door directly. DISPATCH.md §6's estimate assumed
1855 × ~0.8 µs; the population is 1188.

Per-call, decomposed on the current build (minimum of 5 blocks of 20 000 after
200 warmups, `(1, 6, 9, 64)` float32, load 4.45 — high, so read the shares
rather than the absolutes):

| µs/call | total | `resolve` | of which `_bind` | `dispatch(key, **bound)` |
|---|---|---|---|---|
| `.view(1, 6, 576)` | 1.683 | 0.900 | 0.717 | 0.613 |
| `.transpose(1, 2)` | 1.277 | 0.612 | 0.522 | 0.543 |
| `t + t` | 1.797 | 0.759 | 0.670 | 0.880 |
| `.unsqueeze(0)` | 1.235 | 0.533 | 0.444 | 0.564 |

So `resolve` is **43–53% of an operator call**, and at 1188 calls × ~0.65 µs it
is **~0.8 ms per prefill, not ~1.5 ms**. The brief's target is real and is
still the largest single item; it is half the size it was advertised at.

Two further readings from the same run:

* `dispatch_pos` minus `dispatch_kw` is now **0.02–0.03 µs** (e.g. view 0.592
  vs 0.613). Round 2 did what it said: the keyword convention is spent, and
  DISPATCH.md §3.1's refusal to pass `bound` positionally costs almost nothing
  now.
* Under `cProfile`, `_aten_dispatch` is 0.168 s of 0.220 s across five passes —
  76%. The profiler inflates Python frames, so it sizes nothing here; it is
  reported only because it is the same instrument §1 used.

#### Why the 463 refusals are hard to prefilter

Cross-tabulating each refusal by the reason that decided it:

| op | refusals / pass | reason | keywords? |
|---|---|---|---|
| `cat` | 121 | required argument missing | yes |
| `view` | 90 | positional type check | no |
| `__add__` | 62 | positional type check | no |
| `pow` | 61 | positional type check | no |
| `mean` | 61 | arity | yes |
| `rsqrt` | 61 | required argument missing | no |

Arity, "given twice", unknown-keyword and required-missing are all pure
functions of (argument count, keyword names) — no value is consulted — so they
*can* be answered from a precomputed table. Positional type checks cannot: they
are what decides `add.Tensor` against `add.Scalar`. That splits 248 / 215, and
only 62 of the 248 are in calls with no keywords, where the table key would be
a bare integer. A prefilter was therefore **not** built: it addresses 3.8% of
candidate attempts for a per-call key construction on the other 96%.

### 8.2 What changed

One change. `_bind` — the per-candidate parse — is folded into `resolve`'s
loop, and the keyword half is split into `_bind_keywords`.

**This is a move, not a copy, and that is the whole argument for doing it.**
BIND.md §6 item 1 priced merging the frames at "~0.15 ms per prefill at the
cost of a second copy of the resolution loop", and declined. That price is real
for `fn` and `_tensor_method`'s `method`, which are **two** call sites into
`resolve`. It is not real one level down: `_bind` had **exactly one caller**,
`resolve`, in the whole repository. Folding it in removes 1651 Python frames
per forward pass and leaves every rule stated exactly once — the positional
half in `resolve`, the keyword half in `_bind_keywords`.

Three smaller things ride along, each a consequence of the merge:

* **Arming moved from per-plan to per-entry.** `_SchemaPlan` objects are
  constructed in `_Overloads.__init__` and nowhere else, and are reachable only
  through their entry, so arming them together is the same work — one flag test
  per *call* rather than one per *candidate*, and one `_TypeChecker` built
  rather than one per plan.
* **`tuple(args[:skip]) + (tuple(args[skip:]),)` → `args[:skip] + (args[skip:],)`.**
  Slicing a tuple yields a tuple, so both `tuple()` calls were copying
  something already of the right type. `args` is always a tuple: `fn` and
  `method` are the only callers of `resolve` and both build one.
* **"given twice" now runs after the positional type checks** rather than
  before, because it moved into the keyword half. Both orders refuse the same
  calls and neither has a side effect, so which reason is found first is not
  observable — the same argument §3.4 made when this check was first hoisted.

#### `given twice` turns out to be unreachable, and is kept anyway

Two of the tampers in §8.3 disable the "given twice" walk outright and produce
**zero** mismatches. That is not a blind harness; the check is genuinely
redundant in the present structure. After the arity gate, `given <=
n_positional`, and the positional loop zips `call` (length `given`) against
`positional`, so on reaching the keyword half `bound` holds exactly the names
of `positional[:given]` — the very slice "given twice" walks. `name in bound`
therefore answers every call it would. Driven over all 251 plans, of 5570 cases
where the walk fires, 3120 are caught by `name in bound` and the remaining 2450
had already been refused by a positional type check.

It is left in place: it is torch's stated rule, it costs a walk only on the 15%
of calls that pass keywords, and its redundancy is a property of the current
loop rather than a guarantee. This is recorded so that the next person does not
read the zero as coverage.

### 8.3 Why this is safe

**A differential over the whole front door.** Round 1 compared `_bind` against a
verbatim copy of its predecessor. This round compares **`resolve`**, which is a
superset — it includes candidate ordering, the refusal `TypeError`, and the
keyword half. The pre-merge `resolve` *and* `_bind` are extracted verbatim from
`git show HEAD:rust/torch_c/src/bootstrap.py` and exec'd against the live
module's globals; one substitution is applied and asserted to occur exactly once
(`self._bind(` → `_old_bind(self, `, since the new class has no `_bind`). The
new side is likewise loaded from the source file rather than off the class, so
what is compared is proved to be what is on disk.

Over every installed entry and 8930 call shapes — positional tuples up to
length 3 over 19 values that exercise every rule the checker has, crossed with
10 keyword shapes:

```
{"entries": 123, "call_shapes": 8930, "comparisons": 1098390, "mismatches": 0}
```

Refusal, chosen key, bound keys in order, and every bound value are compared.

**A method note that cost a false negative, and would have hidden one.** Round
1's tampers corrupted the precomputed `_SchemaPlan` fields. That cannot work
here: the old side is HEAD, which is *after* round 1, so it reads **the same
plan objects** — corrupting one corrupts both sides equally and the harness
reports a serene zero. The tampers below rewrite the source of the function
under test instead, and each asserts its anchor occurs the expected number of
times before firing (three early attempts were rejected on exactly that check,
having been written against the wrong indentation).

| tamper (new side only) | mismatches |
|---|---|
| positional type check disabled | **80 648** |
| unknown keyword accepted | **12 590** |
| arity gate disabled | **3 601** |
| required-argument walk skipped | **1 537** |
| varargs int-list rule disabled | **381** |
| varargs widening removed (`call[:skip] + (call[skip:],)`) | **381** |
| default-valued arguments never dropped | **154** |
| positional coercion disabled | **84** |
| keyword coercion disabled | **16** |
| keyword type check disabled | **8** |
| "given twice" disabled (in `resolve`) | 0 — §8.2, unreachable |
| "given twice" disabled (in `_bind_keywords`) | 0 — §8.2, unreachable |
| arming skipped | 0 — see below |

**The arming tamper reads zero for a harness reason, and it was chased rather
than accepted.** Evaluation order is new-side-first precisely so the old side
cannot arm the plans on the new side's behalf — but the *first* call shape is
the empty tuple, which refuses before any predicate is needed, and the old side
arms everything on it. Checked separately and directly: 123 of 123 entries are
cold at start, and calling the new `resolve` with arming suppressed raises a
non-refusal `TypeError`. The arming path is load-bearing; this tamper simply
cannot express it.

The repository's gates, on the final artefact:

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh   -> 211 ok,                        exit 0
$PY tools/golden/compare.py                 -> 2843/2843, ops covered=119,    exit 0
$PY tools/golden/compare.py --self-test     -> PASS, 12 x 11 fault modes,     exit 0
$PY rust/torch_c/pytests/verify_schemas.py  -> 4203/4203,                     exit 0
```

Golden is a real guard on this path now — it carries 32 keyword cases, which is
the hole DISPATCH.md §4.1 recorded — and it was not weakened to get here.

**And the model agrees bit for bit.** Every prefill round in §8.4 checksums all
294 912 logits; old and new produced the identical pair across four rounds:

| | Σ logits | max abs |
|---|---|---|
| old (`1f23ac4`) | 4193738.350776 | 29.98118 |
| **new** | **4193738.350776** | **29.98118** |
| upstream torch | 4193739.325235 | 29.981203 |

The upstream difference is the shim's pre-existing float divergence, unchanged.

### 8.4 It is faster, on the shape ops, and prefill cannot see it

Method is §2's and DISPATCH.md §5.1's, unchanged so the numbers compose:
minimum of 5 blocks of 20 000 calls after 200 warmups on a `(1, 6, 9, 64)`
float32 tensor; **upstream, old and new alternate inside every round**; 4
rounds. Only `_C.abi3.so` differs between old and new — `bootstrap.py` is
`include_str!`'d into the artefact, so both sides were built and saved
(distinct md5s; `strings` confirms `_bind_keywords` present in one and absent
in the other) and rounds alternate by file swap. Load 2.0–3.3.

µs/call, minimum of 4 rounds, and **two independent runs** of the whole thing:

| | upstream | old | new | old/new run 1 | run 2 | control |
|---|---|---|---|---|---|---|
| `.view(1, 6, 576)` | 0.776 | 1.622 | **1.513** | **1.072** | 1.065 | 0.994 |
| `.view((1, 6, 576))` | 1.034 | 1.787 | **1.699** | **1.052** | 1.065 | 1.007 |
| `.unsqueeze(0)` | 0.740 | 1.202 | **1.144** | **1.050** | 1.032 | 1.007 |
| `.transpose(1, 2)` | 0.803 | 1.261 | **1.217** | **1.036** | 1.030 | 1.001 |
| `t + t` | 0.987 | 1.797 | 1.786 | 1.006 | 1.023 | 0.980 |
| `.rsqrt()` | 1.475 | 2.457 | 2.445 | 1.005 | 1.013 | 1.000 |
| `.mean(-1, keepdim=True)` | 2.594 | 2.823 | 2.810 | 1.005 | 1.007 | 1.003 |

Ratio to upstream, old → new: view **2.09 → 1.95**, view-tuple 1.73 → 1.64,
unsqueeze 1.62 → 1.55, transpose 1.57 → 1.52.

**Control** — the same artefact under both labels through the identical
harness, 3 rounds: 0.980 / 0.994 / 1.000 / 1.001 / 1.003 / 1.007 / 1.007.
Range **0.980–1.007**, i.e. this harness reads 1.00x to within 2.0% when there
is nothing to find.

**What that resolves and what it does not.** The top four cases beat the
control's worst deviation and their per-round bands do not overlap in either
run (view: old 1.622/1.625/1.630/1.641, new 1.513/1.515/1.540/1.540). The
bottom three — `add`, `rsqrt`, `mean` — sit at 1.005–1.023, **inside the
control's spread**, and their bands overlap. They are reported as **unresolved,
not as small wins.** `mean` is the honest reason why: at 1.09x upstream it has
almost no Python share left to remove.

Per-call saving on the four that resolve is **0.04–0.11 µs**, which is one
Python frame's worth — exactly the size of the thing removed, and about half of
what round 2 delivered (7.3–15.3%).

#### Prefill, which does not resolve it

SmolLM2-135M float32, 6-token prompt, minimum of 5 timed passes after 2
warmups; upstream, old and new alternate inside every round; 4 rounds, load
2.5–2.7.

| | min (ms) | ratio to upstream |
|---|---|---|
| upstream | 35.232 | 1.0000 |
| old (`1f23ac4`) | 37.422 | **1.0622** |
| **new** | 37.467 | **1.0634** |

Pairwise per round: old 1.0515 / 1.0561 / 1.0618 / 1.0698; new 1.0463 / 1.0655
/ 1.0631 / 1.0684. **Fully overlapping**, and predicted to be: 1651 candidate
attempts × ~0.045 µs is ~0.07 ms against a 37 ms measurement whose
round-to-round spread is ~2%. This is the third round to hit the same wall for
the same reason (§5, DISPATCH.md §5.2); the prefill run is reported as a
did-not-regress check and as the source of the §8.3 checksum, not as evidence.

### 8.5 Tried and did not pay

**Building the bound mapping with `dict(zip(names, call))`.** The per-argument
loop stores into `bound` one key at a time, after three attribute loads on the
`_ArgPlan`. Replacing it with a predicate-only loop plus a single C-level
`dict(zip(plan.names, call))` — with the coercion rule lifted into a separate
pass, justified because only **6 of 251 plans** have a coercible positional
argument and no schema in either table names two arguments the same (checked:
0 of 251) — looked like a clear win and was **measurably worse**:

| old/new | view | transpose | add | unsqueeze | view-tuple |
|---|---|---|---|---|---|
| merge only | 1.072 | 1.036 | 1.006 | 1.050 | 1.052 |
| merge + `dict(zip(...))` | **0.974** | **0.941** | **0.931** | **0.928** | 1.006 |

5–7% *slower* than the merge alone, with the per-round bands separated in the
wrong direction on every case. Building a second `zip` iterator and handing it
to `dict` costs more than the two or three dict stores and the attribute loads
it replaces, at the argument counts operators actually have. Reverted; the loop
is as §8.2 leaves it. Recorded because the C-level spelling is the obvious move
and it is a trap at this size.

**A candidate prefilter keyed on arity.** Not built — §8.1 measured its reach
at 3.8% of candidate attempts, against a key construction on every call.

### 8.6 What is left

1. **The per-argument predicate call, ~0.09–0.26 µs per bind.** The largest
   item still inside `resolve`. Removing it means either code generation or a
   second statement of the twelve `_base` rules; §3.3 and §6 have both refused
   the latter twice, once on measurement and once on hazard. A sound middle
   path exists — annotate each rule at its point of definition with whether it
   is decided by the value's *type* alone (all of them except `Scalar`, which
   accepts a 0-dim tensor, and the list rules), then test a cached type
   identity before calling — but it puts mutable state on the hot path and
   buys perhaps 0.07 µs. Not taken.
2. **The `method` → `resolve` frame**, ~1188 per pass. This is the merge BIND.md
   §6 actually priced, and its objection stands: `fn` and `method` are two call
   sites, so folding `resolve` into them duplicates the loop.
3. **`(self,) + args`** in `method` — a tuple concatenation on every method
   call, avoidable only by changing `resolve`'s signature.

None of these is reachable from prefill on this machine. §8.4's control puts
the floor at 2%; item 1 is ~4% of a `view` and the rest are smaller.

### 8.7 Method notes

* `bootstrap.py` is compiled into the artefact by `include_str!`, so **swapping
  the `.py` file changes nothing without a rebuild** (§7.1). Both sides were
  built and saved to `/tmp/bind2_art/lib_C.{old,new}.dylib`, md5s verified
  distinct, and the presence/absence of the `_bind_keywords` symbol in
  `strings` was used to confirm which is which before believing any reading.
  The final rebuild from the reverted source reproduced `lib_C.new.dylib`
  byte for byte, which is why §8.4's first run did not need repeating.
* `TORCH_C_ARTEFACT` was set explicitly for every gate run, per DISPATCH.md's
  note that `tools/golden/loader.py` otherwise ignores a custom
  `CARGO_TARGET_DIR` and can compare a stale binary.
* `uptime` was recorded before every round. No timing round was taken above
  load 3.3 except the control, which ran at 3.7–4.1 — the direction that makes
  a control *worse*, so its 0.980–1.007 is an upper bound on harness bias.
* The working tree was restored from `cp` backups, never `git checkout`.
* Harnesses and raw per-round output are under `/tmp/bind2/`: `differential.py`
  (old-vs-new plus the tamper table), `census.py` (§8.1's counts), `inert.py`
  (the two zero-tamper explanations), `layers.py`, `ablate.py`,
  `bind2_ab.sh` + `bind2_micro.py` + `bind2_report.py` (§8.4), and the
  `*.jsonl` per-round records. They are not committed; `differential.py` is the
  one worth promoting into the repository if this area is touched again.

---

## 9. Round 4 — `Tensor.dtype` was never interned, and it was a correctness bug

The brief named this as one lead to check first, "a correctness question
wearing a performance hat." It is a correctness question, the answer is a
real bug, and the bug already had a name in this repository before this round
started -- `docs/graph/DECOMP.md` §7.2 recorded its symptom and said "not yet
narrowed down which op it's from." It is this.

### 9.1 The identity check, confirmed

`tensor.rs:614`:

```rust
#[getter]
fn dtype(&self) -> PyDtype {
    PyDtype::new(self.tag)
}
```

A fresh Rust struct -- and therefore a fresh Python object -- on every read.
`PyDtype::__eq__` compares the tag, so `t.dtype == torch.float32` was never
wrong. `t.dtype is torch.float32` was `False` unconditionally, for every
tensor, of every dtype, every time:

```
>>> t = torch.zeros(3, dtype=torch.float32)
>>> t.dtype is torch.float32
False
>>> t.dtype == torch.float32
True
>>> t.dtype is t.dtype
False
```

Round 1's "attribute reads are comparable to upstream" table (§1) measured
`.shape` and `.dim()`. It never measured `.dtype`, which is the one of the
three with a singleton-identity contract upstream actually relies on --
`torch.float32` is a module-level constant precisely so code can compare
dtypes by `is` (`dtype.rs`'s own `register()` interns exactly one Python
object per dtype and the module-level names all point at it -- `torch.float32
is torch.float32` was always `True`; it is `Tensor.dtype`'s *return value*
that was never one of those objects).

### 9.2 This is not a hypothetical hazard -- it produces a wrong dtype today

The vendored tree checks dtype by identity in reachable code.
`_meta_registrations.py` gates several quantised-linear registrations
(`torch._weight_int8pack_mm` and neighbours) on `w.dtype is torch.uint8` /
`b.dtype is torch.int8` / `w.dtype is torch.int32`; against the native getter
those are `False` unconditionally, so the guard takes the "wrong dtype"
branch regardless of the tensor's actual dtype. SmolLM2-135M float32 does not
reach that code, so it was not caught by the model-checksum check §8.3 relies
on.

Something SmolLM2's own capture/decompose path *does* reach was already
failing, silently, and already had a name: `docs/graph/DECOMP.md` §7.2, "**아직**"
(not yet):

```
                       input                    decomposed dtype
upstream  baddbmm(f32, f32, f32)                 float32
here      baddbmm(f32, f32, f32)                 float64
```

recorded as "a divergence in the scalar promotion rule... not yet narrowed
down which op it's from." It is this op. `torch/_prims_common/__init__.py`'s
`get_higher_dtype` (used by `elementwise_dtypes`, which every
`@pw_cast_for_opmath`-decorated decomposition calls, `baddbmm` among them):

```python
    a, b = _extract_dtype(a), _extract_dtype(b)

    if a is b:
        return a
    ...
    ordered_datatypes = (
        (torch.bool,),
        (torch.uint8, torch.int8),
        ...
        (torch.float32,),
        (torch.float64,),
        ...
    )
    for idx, dtypes in enumerate(ordered_datatypes):
        if a in dtypes and b in dtypes:
            return ordered_datatypes[idx + 1][0]
```

`if a is b: return a` is upstream's fast path for "these are literally the
same dtype," and it is load-bearing, not an optimisation: the table below it
is grouped for dtypes that are *different but equal-ranked*
(`torch.uint8`/`torch.int8`, `torch.float16`/`torch.bfloat16`) and its rule
for two same-group members is "promote to the next group" -- correct when `a`
and `b` are genuinely different dtypes sharing a rank, wrong when they are the
same dtype read twice. Two `float32` tensors reach `get_higher_dtype(t1.dtype,
t2.dtype)`; against the native getter `a is b` is always `False`, so both fall
into the table, both are found in `(torch.float32,)`, and the function returns
`ordered_datatypes[idx + 1][0]` -- `torch.float64`. That is the exact
divergence DECOMP.md §7.2 recorded, reproduced directly:

```
>>> from torch._decomp.decompositions import baddbmm
>>> c, a, b = torch.ones(2,3,5), torch.ones(2,3,4), torch.ones(2,4,5)
>>> baddbmm(c, a, b).dtype     # pre-fix artefact
torch.float64
>>> baddbmm(c, a, b).dtype     # post-fix artefact, same inputs
torch.float32
```

Confirmed against both artefacts directly (§9.4's `old`/`new` builds, md5s
distinct), not inferred: rebuilding with `_install_tensor_dtype_identity`
reverted reproduces `float64`; rebuilding with it in place reproduces
`float32`, matching upstream's `f32,f32,f32 -> f32`. **This was not a
performance lead that happened to also matter for correctness -- it was
already a filed, unexplained correctness bug, and the performance-shaped
question the brief asked is what found its cause.**

### 9.3 The fix -- in `bootstrap.py`, without touching `tensor.rs`

`tensor.rs` is not this round's territory, so the fix is a Python-level
override of `TensorBase.dtype`, installed from `_install_tensor_methods`
(new function `_install_tensor_dtype_identity`, called before
`_install_tensor_conversions`). It works because `torch._C._get_all_dtypes()`
already carries the fact needed: it enumerates the dtypes by *name*
(`str(d)`), and `getattr(module, name)` for each name is the interned
singleton `dtype.rs::register` put on the module -- the same object
`torch.float32` names, *not* what `_get_all_dtypes()` itself returns
(`get_all_dtypes()` is `PyDtype::new` over the table too, so building the
intern table from its return value directly would reproduce the exact bug
this exists to fix -- tried first, and it does: see the inline comment in
`_install_tensor_dtype_identity`). `PyDtype.__eq__`/`__hash__` are defined on
the tag, so a dict keyed on a freshly-read (uninterned) dtype and valued on
the module attribute resolves any `PyDtype` to its singleton by tag equality:

```python
def _install_tensor_dtype_identity(module, tensorbase) -> None:
    raw_dtype = tensorbase.dtype
    intern_table = {}
    for raw in module._get_all_dtypes():
        name = str(raw)[len("torch."):]
        intern_table[raw] = getattr(module, name)

    def dtype(self):
        raw = raw_dtype.__get__(self)
        return intern_table.get(raw, raw)

    setattr(tensorbase, "dtype", property(dtype))
```

Verified for every one of the ten dtypes this build can actually store
(`float32`, `float64`, `float16`, `bfloat16`, `uint8`, `uint32`, `int16`,
`int32`, `int64`, `bool` -- `PyDtype::storage()` in `dtype.rs`):

```
float32 is: True eq: True   float64 is: True eq: True
float16 is: True eq: True   bfloat16 is: True eq: True
uint8 is: True eq: True     uint32 is: True eq: True
int16 is: True eq: True     int32 is: True eq: True
int64 is: True eq: True     bool is: True eq: True
```

`_to_copy`'s `!=` (not `is not`) is kept: `self.dtype` is now interned, but a
caller-supplied `dtype` value is not guaranteed to be one, because
`PyDtype.to_real()`/`.to_complex()` (`dtype.rs`) build a fresh, uninterned
`PyDtype` unconditionally. `!=` is correct either way; the comment at that
call site is updated to say why `is not` is still not used.

**The clean fix is one line in `tensor.rs`**, changing the getter to call
`dtype::interned(py, self.tag)` (already `pub(crate)`, already what
`get_default_dtype` uses) instead of `PyDtype::new`. That needs no dict, no
extra Python frame, and no per-read allocation -- strictly better than what
is installed here, and it is not written because `tensor.rs` is outside
`bootstrap.py` + `docs/bindings/BIND.md`. Recorded so whoever next touches `tensor.rs`
does not have to re-derive it.

> **Correction (문서 감사):** it was written, in the same commit that landed this
> section (`b8c3ea1`) -- the commit message says so directly ("The investigating
> agent could not touch tensor.rs and wrote a Python-side property override
> instead ... this replaces it"), and `tensor.rs`'s `dtype` getter is
> `crate::dtype::interned(py, self.tag)` today, not `PyDtype::new`. The
> `_install_tensor_dtype_identity` Python override this section describes does
> not exist in `bootstrap.py` (re-verified, absent). So §9.4's cost table below
> measures a code path that was never shipped -- the coordinating session
> replaced it with the `tensor.rs` one-liner before merging, per the same commit
> message: **0.042 µs against the original 0.070 µs, i.e. cheaper than the bug it
> fixes, not 2.1x more expensive.** The correctness finding (§9.1-§9.2, §9.5) and
> the live behaviour (`t.dtype is torch.float32` → `True`, `baddbmm` → `float32`)
> are unaffected and reconfirmed live in this audit.
>
> Standing check (docs/verification/DOCWATCH.md):
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs interned present -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _install_tensor_dtype_identity absent -->

### 9.4 What it costs -- measured, and it is not the story

Cost of `t.dtype` itself, minimum of 5 blocks of 20 000 after 200 warmups on
a `(1, 6, 9, 64)` float32 tensor, artefacts alternating old/new every round in
fresh subprocesses (md5s distinct: `old` f4d91c5..., `new` 6f8860e...), 4
rounds, load 3.26-3.29:

| | old | new | ratio |
|---|---|---|---|
| round 1 | 0.0696 | 0.1518 | 2.18x |
| round 2 | 0.0697 | 0.1530 | 2.19x |
| round 3 | 0.0727 | 0.1518 | 2.09x |
| round 4 | 0.0701 | 0.1530 | 2.18x |

µs/call. The bands do not overlap: the dict lookup plus the extra Python
frame roughly **doubles** the cost of a single `.dtype` read, from ~0.07 µs to
~0.15 µs -- the honest price of the fix in §9.3, and the reason the clean
version belongs in `tensor.rs` instead.

**It does not matter at the model level, because `.dtype` is read rarely.**
Counted directly (a counting wrapper installed over the already-fixed
property, one SmolLM2-135M float32 prefill, after 2 warmup passes so import-
time reads do not inflate it):

```
dtype reads per prefill: 220
```

against ~9275 `_aten_dispatch` calls (§9.5's profile). At +0.082 µs per read,
220 reads is **+0.018 ms per prefill**, against a ~37-38 ms baseline (§5,
§DISPATCH.md §5.2) -- **about 0.05%**, an order of magnitude below what this
machine's round-to-round spread on prefill can resolve (§5 could not resolve
2-6%). This is reported as a did-not-regress number, the same status BIND.md
and DISPATCH.md give their own unresolved prefill readings, not as a measured
loss.

### 9.5 The one thing this changes, and it is not touched

`baddbmm`'s decomposition no longer wrongly promotes to `float64` (§9.2), so
the smoke test written *against* that documented bug now disagrees with it:

```
FAIL test_decompose_refuses_by_name_what_it_cannot_lower: AssertionError
```

`rust/torch_c/pytests/test_shim.py`'s
`test_decompose_refuses_by_name_what_it_cannot_lower` asserts, as its third
of three refusal cases, that lowering `aten.baddbmm.default` produces a
result the capture *disagrees* with (`"aten.baddbmm.default" in
r["refuse_disagrees"]`, `"torch.float64"` and `"torch.float32"` both in the
message) -- i.e. it pins DECOMP.md §7.2's bug as expected behaviour. With this
fix, the decomposition agrees with the recording (both `float32`), so
`_lower_node` no longer raises, `r["refuse_disagrees"]` reads `"ACCEPTED"`,
and the assertion fails. Confirmed directly: calling `_decomp_road_fixture()`
after the fix returns `refuse_disagrees: "ACCEPTED"` where it used to return
the `DecompositionRefused` message the test checks the wording of.

**This is not touched.** `rust/torch_c/pytests/test_shim.py` and
`docs/graph/DECOMP.md` are outside this round's territory (`bootstrap.py` +
`docs/bindings/BIND.md`), and the assertion encodes a bug this round's fix genuinely
removes -- updating it is a real, small change (the test's case 3 needs a
different op that still disagrees, or the assertion needs to become "now
lowers cleanly," matching how `test_decompose_lowers_sum_default_now_that_
the_kernel_agrees` was written the last time a case in this test stopped
reproducing its bug) but it is a decision about what that test should now
assert, not a decision this round's territory covers. **This is why the
acceptance bar's "smoke exactly 211" does not hold as of this fix**: it reads
210 ok / 1 failed, and the one failure is this. Golden (2843/2843), the
self-test (PASS), and schemas (4203/4203) are unaffected and still hold
exactly, all re-verified against the rebuilt artefact after the earlier false
alarm in §9.6 was found and corrected.

`docs/graph/DECOMP.md` §7.2 can also be updated to close out "아직" (not yet) --
§9.2 above is the missing "which op it's from" -- but that file is likewise
outside this round's territory.

> **Correction (문서 감사, 재확인):** `test_decompose_refuses_by_name_what_it_cannot_lower`
> no longer fails. It was updated in a later round to drop `aten.baddbmm.default` from wall
> 3's expected disagreements (a new test,
> `test_decompose_lowers_baddbmm_default_now_that_the_dtype_is_a_singleton`, pins the fixed
> behaviour instead), exactly along the lines this section anticipated. The "smoke exactly
> 211" / "210 ok, 1 failed" state described above is this round's own landing number, not the
> current one -- the full suite passes today (268 ok, 0 fail, confirmed via `run.sh`).

### 9.6 A measurement-hygiene miss, caught before it was reported as real

The first pass at golden/schemas after this fix read 2784/2843 and a hard
crash in `verify_schemas.py` (`NotImplementedError:
torch._C._jit_get_all_schemas`). That looked like a second, worse regression.
It was not: the same two commands against the **unmodified baseline artefact**
produced the identical 2784/2843 and the identical crash. The cause was this
session's own environment -- `PYTHONPATH`/`TORCH_USE_RTLD_GLOBAL` had been
left set (needed for the direct `.dtype` micro-benchmarks) when invoking
`tools/golden/compare.py` and `verify_schemas.py`, which the brief's own
VERIFY block does not set for those two commands. Unset, both baseline and
fixed builds read exactly 2843/2843 and 4203/4203. Recorded per the brief's
own instruction to suspect the harness before the code when a baseline
disagrees with the number in the brief -- this is the twelfth instance of that
class in this file's and DISPATCH.md's combined history, not the first.

A second near-miss in the same pass: the first "final" golden/schemas run
after popping a `git stash` (used to compare old vs. new source) was not
preceded by a rebuild, so it silently re-tested the *previous* artefact.
Caught by `strings _C.abi3.so | grep -c _install_tensor_dtype_identity`
reading `0` where the patched build reads `3` -- the same class of trap
DISPATCH.md §7.1 and this file's §8.7 both name (`bootstrap.py` is
`include_str!`'d in at Rust build time, so a source-only diff proves nothing
about which artefact is loaded).

### 9.7 The profile, as it stands after three prior rounds

Counted the same way §8.1 was -- `cProfile` over 5 SmolLM2-135M float32
forward passes, after 2 warmups, on the artefact with this round's fix
applied:

```
137081 function calls (132631 primitive calls) in 0.221 seconds

ncalls   tottime  function
  9275    0.170    {built-in method torch._C._aten_dispatch}
  5940    0.010    bootstrap.py:2190(resolve)
   615    0.003    bootstrap.py:3437(__getitem__)
 33775    0.003    {built-in method builtins.isinstance}
  5020    0.003    bootstrap.py:3064(method)
  1540    0.001    bootstrap.py:2150(_bind_keywords)
   625    0.001    bootstrap.py:3207(to)
```

`resolve` is now **4.5% of total tottime** (0.010 / 0.221), down from the
53-63 % of an *individual operator call* §8.1 measured before round 3's
fold -- consistent with that fold, not a new finding. `_aten_dispatch` itself
is 77%, and DISPATCH.md already closed the removable third of that
(the keyword convention); the rest is kernel work upstream pays for too. Two
things follow from this profile, both against the brief's instruction not to
assume `resolve` is still the top item:

1. **`resolve`/`_bind`/`method`/`fn` are not where the round-5 win is.** Three
   rounds have already measured and either landed or explicitly declined
   every change to this path that showed a measurable effect (§3, §8) or
   priced the remaining ones below this machine's noise floor (§8.6: item 1
   ~0.07 µs, items 2-3 smaller still, against a control that reads 1.00x to
   within ~1-2%). Re-attempting them without a new idea would reproduce §8.5's
   "measured 5-7% slower" or §3.4's "measured within noise, reverted" outcome
   -- exactly the dead ends the brief named.
2. **Nothing else in `bootstrap.py` stands out.** `__getitem__` (615 calls,
   0.003 s), `to` (625 calls, 0.001 s) and `_bind_keywords` (1540 calls,
   0.001 s) are each under 1.5% of total tottime on this model; none has the
   call-count-times-per-call-cost shape that made `resolve` worth three
   rounds. The dominant remaining cost is `_aten_dispatch` itself, which is
   forbidden territory this round (`aten.rs`) and is mostly real kernel work
   rather than convention overhead (DISPATCH.md §1.1's decomposition already
   separated the two).

No change is proposed from this profile. It is reported because the brief
asked for it "as found," and because "nothing left to fold" is itself the
answer to "is `resolve` still the top item" after three rounds -- it explains
why this round's finding is a correctness fix rather than a fourth `resolve`
optimisation.

### 9.8 Summary

- **`.dtype` question: correctness, not performance.** `Tensor.dtype` returned
  a fresh, uninterned object on every read; `is torch.<dtype>` was `False`
  unconditionally. Fixed in `bootstrap.py` (§9.3); the clean fix is a one-line
  change in `tensor.rs`, described but not written (out of territory).
- **It was not latent.** It was the unexplained cause of a bug already on
  record, `docs/graph/DECOMP.md` §7.2's `baddbmm` float32→float64 promotion,
  traced to `get_higher_dtype`'s `if a is b: return a` fast path in
  `torch/_prims_common/__init__.py:1360` failing open into a table lookup
  that promotes same-rank dtypes to the next rank.
- **The fix costs ~0.018 ms per SmolLM2-135M prefill** (220 `.dtype` reads ×
  ~0.082 µs added each), against a ~37-38 ms baseline -- not measurable
  against this machine's noise floor, reported as did-not-regress.
- **One smoke test now fails**, `test_decompose_refuses_by_name_what_it_cannot
  _lower`, because it pinned the bug this fix removes as expected behaviour.
  Not fixed here -- out of territory (`rust/torch_c/pytests/test_shim.py`).
  Golden (2843/2843), the self-test (PASS) and schemas (4203/4203) all hold
  exactly, unaffected.
- **The profile shows nothing else to fold in `bootstrap.py`.** `resolve` is
  now 4.5% of prefill tottime; the dominant cost is `_aten_dispatch` itself,
  outside this round's territory.
