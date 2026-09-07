# DISPATCH — the C side of one operator call

What `_C._aten_dispatch` costs per call, measured rather than subtracted, and
which part of it is removable without changing what the door does.

This picks up from docs/bindings/BIND.md, which closed most of the Python-side gap and
left ~2.2 ms of SmolLM2-135M prefill unexplained. BIND.md attributed roughly
1.3 ms of that to the three Python frames and roughly **0.9 ms to "the C
side"** — by subtraction, because the agent that wrote it could not edit
`aten.rs`. §1 below is the direct measurement it could not take, and it does
not agree with the subtraction.

---

## 1. The brief's premise, checked

BIND.md §6 item 2 names the largest remaining item:

> **`dispatch(key, **bound)`** — the bound arguments are built as a dict and
> then unpacked into keyword arguments for the C entry point, which re-parses
> them.

Two claims are folded together there, and they are not both true.

**The Python-side unpack is free.** `dispatch(key, **bound)` and
`dispatch(key, self=t, size=s)` cost the *same*, to within noise:

| | view | transpose | add |
|---|---|---|---|
| `D(key, self=…, size=…)` — keywords at the call site | 0.782 | 0.771 | 1.102 |
| `D(key, **bound)` — what `bootstrap.py` emits | 0.775 | 0.764 | 1.094 |
| difference | **−0.007** | **−0.007** | **−0.008** |

µs/call, minimum of 5 blocks of 20 000 after 200 warmups, on a
`(1, 6, 9, 64)` float32 tensor. CPython 3.13 hands the merged dict straight to
a `METH_VARARGS | METH_KEYWORDS` C function; there is no extra copy for the
`**`. **Removing the `**` from the call site buys nothing.** That half of the
brief is answered: there is no Python-side round trip to remove.

**The keyword *convention* is not free.** The same call made positionally:

| | view | transpose | add |
|---|---|---|---|
| `D(key, *tup)` — positional | 0.628 | 0.527 | 0.894 |
| `D(key, **bound)` — keyword | 0.775 | 0.764 | 1.094 |
| **cost of the keyword convention** | **0.147** | **0.237** | **0.200** |

and for scale, the pyo3 entry itself (`_shim_sdpa_reference(None)`, one
argument, no work) is **0.051 µs**. So the keyword convention costs 3–5x the
entire function-call mechanism.

### 1.1 The corrected attribution

Decomposing `t.view(1, 6, 576)` and friends all the way down (same run, same
tensor, load 4.5–7.0):

| layer | view | transpose | add |
|---|---|---|---|
| `resolve` + `_bind` (bootstrap.py) | 1.063 | 0.747 | 0.907 |
| keyword convention (C side, removable) | 0.147 | 0.237 | 0.200 |
| C work — device scan, op match, kernel, promote | 0.576 | 0.476 | 0.843 |
| pyo3 entry | 0.051 | 0.051 | 0.051 |
| **total** | **1.839** | **1.511** | **2.001** |

Sanity: the totals reproduce BIND.md §5's "after" row (view 1.790, transpose
1.460, add 1.943) to within 3%, on an independently written harness. The two
measurements agree about the thing they both measured.

At ~1855 dispatches per prefill:

| | BIND.md's estimate | measured here |
|---|---|---|
| Python binding layer | ~1.3 ms | ~1.5–1.7 ms |
| C side, **removable** | — | **~0.28–0.44 ms** |
| C side, real work | — | ~1.1 ms (upstream pays its own) |
| “the C side”, as one number | ~0.9 ms | — |

**BIND.md's 0.9 ms is not one removable item.** About a third of it is the
keyword convention and can go; the rest is the kernel, the device gate and
tensor construction, which upstream also pays for and which removing would
mean not computing the answer. The honest ceiling for this work is therefore
**~0.3–0.4 ms of the 2.2 ms residual, not 0.9 ms** — and that is below what
this machine's noise floor can resolve on prefill (BIND.md §5 could not
resolve 2–6%; this is 1.5%). It is comfortably resolvable on the microbench,
which is where it will be reported.

---

## 2. Where the keyword convention's cost actually goes

`sample(1)` at 1 ms over an 8-second loop of
`D("aten.transpose.int", **{"self": t, "dim0": 1, "dim1": 2})`, 6031 samples,
leaves only. Compared against the same loop calling positionally (6076
samples, equal wall time — note the positional loop completes ~45% more
iterations in that time, so its counts are *per iteration* lower still than
they look):

| leaf | keyword | positional |
|---|---|---|
| `pyo3 … extract_arguments_tuple_dict<TupleVarargs, DictVarkeywords>` | **410** | — |
| `unicode_hash` | 148 | — |
| `dict_merge` | 139 | — |
| `PyDict_GetItemRef` | 124 | — |
| `PyDict_Next` | 121 | — |
| `pysiphash` | 122 | — |
| `unicode_decode_utf8` | 122 | — |
| `PyDict_SetItem` | 96 | — |
| `PyDict_Size` | 95 | — |
| `PyUnicode_AsUTF8AndSize` | 94 | — |
| `insertdict` | 60 | — |
| `_tlv_get_addr` (pyo3 thread-local) | 524 | 815 |
| `PyTuple_GetItem` | — | 74 |

**The key strings are walked three times per call.**

1. CPython's `**bound` merges into a fresh dict (`dict_merge`, `insertdict`).
2. **pyo3 rebuilds that dict a second time.** `#[pyo3(signature = (op, *args,
   **kwargs))]` has one *named* parameter, so pyo3 takes the general path: it
   iterates every key, `to_str()`s it, compares it against `"op"`, and
   `set_item`s the survivors into a **brand new `PyDict`**
   (`DictVarkeywords::handle_varkeyword` in
   `pyo3-0.29.2/src/impl_/extract_argument.rs:986`). That is the 410-sample
   leaf, and it is pure ceremony — the dict it builds is item-for-item the
   dict it was handed.
3. Each kernel then reads its arguments back out by name. `optional()` calls
   `kwargs.get_item(name)` with a Rust `&str`, and pyo3 has to **build a
   Python string for every argument of every call** (`unicode_decode_utf8`),
   then hash it from scratch (`pysiphash`, because a fresh string has no
   cached hash), then probe the dict (`PyDict_GetItemRef`).

None of the three passes learns anything the previous one did not know.

### 2.1 pyo3 has a fast path, and two things in the signature disable it

`pyo3-macros-backend-0.29.2/src/params.rs:40`:

```rust
/// Return true if the argument list is simply (*args, **kwds).
pub fn is_forwarded_args(signature: &FunctionSignature<'_>) -> bool {
    matches!(signature.arguments.as_slice(),
             [FnArg::VarArgs(..), FnArg::KwArgs(..),])
}
```

and at line 71, when that holds:

```rust
// In the varargs convention, we can just pass though if the signature
// is (*args, **kwds).
```

— the tuple and the dict are handed to the function **as they arrived**, with
no `FunctionDescription`, no per-key `to_str()`, and no rebuilt dict.

The door fails that match for **two** reasons, and only the first is visible in
the source above. It names `op`; and it takes `py: Python<'_>`, which is an
`FnArg::Py` and therefore *also* an element of the slice being matched. Both
have to go. §3.3(a) records what it cost to learn the second one.

(The fast path is guarded by `!fastcall`. This build takes the non-fastcall
path — the profile shows `extract_arguments_tuple_dict`, which is the
`METH_VARARGS | METH_KEYWORDS` entry — so the fast path is available to it.)

### 2.2 §1.1's estimate held up

Worth stating because it could easily not have: the ceiling §1.1 derived
before any code changed — 0.147 / 0.237 / 0.200 µs for view / transpose / add
— is what the finished work actually removed, 0.164 / 0.194 / 0.130 µs
measured end to end in §5.1. The two disagree by less than the round-to-round
spread, in both directions, so the estimate was not flattered by the thing it
predicted.

What is left of the convention after both changes is 0.042 / 0.051 / 0.006 µs
(§5.1), i.e. most of it is gone and the remainder is at or under the noise on
two of the three.

---

## 3. What is being changed, and what is deliberately not

Two changes, both entirely inside `aten.rs`. **Neither touches the call site
in `bootstrap.py`, and neither changes the door's signature as Python sees
it.**

### 3.1 Not taken: passing the bound arguments positionally

The obvious reading of the brief — have `bootstrap.py` emit
`dispatch(key, *bound)` — is **rejected on safety, not on cost.** `_bind`
drops any argument equal to its own default (BIND.md §3.4), so `bound` can
have *holes*: a call that supplies `step` but not `start` on
`slice.Tensor(self, dim, start, end, step)` binds `{self, dim, step}`, and
positionally that puts `step` where the kernel reads `start`. Making it safe
means having `_bind` emit a dense tuple with every skipped argument filled
from the schema default — which is a second, independent statement of every
default in the table, in a different place from the one the resolver already
uses, and any disagreement between the two is a silent wrong answer rather
than a refusal. That is a bad trade for ~0.2 µs, and it is work in `_bind`
rather than at the call site the brief scoped.

### 3.2 Not taken: `_tlv_get_addr`

The single largest leaf in *both* profiles (524 keyword, 815 positional —
~10–12% of wall time) is macOS's dynamic thread-local accessor. It is present
in equal measure on the positional path, so **it is not part of the keyword
convention's cost** and nothing in §5 comes from it — which is the reason it
is set aside here rather than chased. It is not nothing, though, and §6 item 2
records what its two callers are and what removing each would cost, because
the biggest number on the page deserves better than being waved at.

### 3.3 Taken: the two changes

Both are in `rust/torch_c/src/aten.rs`. **`bootstrap.py` is not touched at
all** — the call site turned out not to be where the cost was (§1), so the
brief's `dispatch(key, **bound)` stays exactly as `972dfe4` left it.

**(a) The door stops naming its arguments.** `_aten_dispatch` was
`signature = (op, *args, **kwargs)`; it is now `signature = (*args, **kwargs)`
with `op` split off the front of the tuple by hand. That is what qualifies it
for pyo3's forwarding fast path (§2.1) and deletes the rebuilt dict.

The trap, which cost a build and a profile: **`Python<'_>` is itself an
`FnArg::Py` in the signature list pyo3 pattern-matches on**, so a `py`
parameter disqualifies the fast path exactly as a named `op` does. The first
attempt removed `op` but kept `py`, and measured *nothing* — 0.775 → 0.760 µs
on view, inside the noise — because `extract_arguments_tuple_dict` was still
there at 488 samples. `py` is now recovered from `args.py()`. The check that
this is real is that the symbol is gone from the binary:

```
$ nm -a lib_C.dylib | grep -c 'extract_arguments_tuple_dict.*DictVarkeywords'
0      # was 1
```

`aten_dispatch` keeps its Rust signature and its name; the pyfunction is a
thin wrapper around it. That is not cosmetic — `capture.rs` replays a recorded
graph by calling `aten_dispatch` from Rust, where the op is already a `String`
and there is no tuple to split.

**(b) The argument names are interned.** `optional()` — the single funnel every
kernel reads an argument through — called `kwargs.get_item(name)` with a Rust
`&str`, which makes pyo3 **allocate a fresh `PyString` and hash it from
scratch on every argument of every call**. `interned_name()` maps the 74 names
this file actually uses to `intern!`ed statics, built once per process.

It is a lookup table and not a source of truth: an unknown name returns `None`
and the caller takes the old path. **The table cannot make a call answer
differently, only more slowly**, which is what makes it safe to write by hand
and safe to leave incomplete. The names were extracted mechanically from the
helper call sites rather than retyped, so none of them can disagree with the
name a kernel asks for.

---

## 4. Behaviour did not move

The three gates, on the final artefact, unchanged from before the work — which
is the point:

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh   -> 197 ok,                      exit 0
$PY tools/golden/compare.py                 -> 2811/2811, ops covered=119,  exit 0
$PY rust/torch_c/pytests/verify_schemas.py  -> 4203/4203,                   exit 0
```

**And a real model agrees bit for bit.** Every prefill round in §5.2 dumps a
checksum of all 294 912 logits. Across four rounds, old and new produced the
identical triple:

| | n | Σ logits | max abs |
|---|---|---|---|
| old (`061e3d8`) | 294 912 | 4193738.350776 | 29.98118 |
| **new** | 294 912 | **4193738.350776** | **29.98118** |
| upstream torch 2.13.0 | 294 912 | 4193739.325235 | 29.981203 |

The old/new agreement is exact. The upstream difference is the pre-existing
float divergence this shim already had; it is unchanged by this work, which is
the thing worth checking.

The two `TypeError`s pyo3 used to raise are now raised by hand, in pyo3's own
wording, so the error surface is the same shape:

* no arguments at all → `_aten_dispatch() missing 1 required positional
  argument: 'op'`
* a non-string first argument → `argument 'op': '<type>' object cannot be
  converted to 'PyString'`

An unimplemented operator still refuses **by name** through the same
`aten_not_implemented(other)` arm — `test_unimplemented_op_names_itself` and
`test_every_advertised_op_is_actually_dispatchable` both pass, and the latter
calls the door 129 times with no arguments at all, which is precisely the path
the hand-rolled `op` extraction changed.

The crate also still cross-compiles for the device target, which matters
because `aten.rs` is shared:

```
sh scripts/device_android.sh build   -> aarch64-linux-android, exit 0
```

### 4.1 Which gate actually guards change (b) — and it is not the golden one

The gates above only mean something if they can go red. `interned_name` was
tampered with — one arm changed from `"self" => intern!(py, "self")` to
`intern!(py, "self_TAMPER")`, so every kernel looking up its receiver by
keyword misses — and the three gates were re-run:

| gate | tampered result |
|---|---|
| `tools/golden/compare.py` | **2811/2811, exit 0 — did not notice** |
| `pytests/run.sh` | 149 ok, **exit 1** (48 tests red) |
| SmolLM2 prefill | **dies immediately**, `aten.normal_.default: missing required argument 'self'` |

**The golden harness is blind to this entire code path**, and the reason is
structural rather than accidental: every builder in `tools/golden/cases.py`
calls the door **positionally** — `c_module._aten_dispatch("aten.add.Tensor",
a_c, b_c)` — so `optional()` takes the `index < args.len()` branch and the
keyword lookup never runs. All 2811 cases share that shape. The headline
correctness number for this repository does not cover how `bootstrap.py`
actually calls the shim.

That is worth writing down for two reasons. It is why §4 leans on the smoke
suite and the real-model checksum rather than on the golden count; and it is a
standing gap — a future change to argument *binding*, as opposed to argument
*arithmetic*, can be green on 2811 cases and still be broken. The cheap fix
would be a handful of case builders that pass their arguments by keyword.

> **Update (docs/verification/GOLDEN.md): the cheap fix above was taken, and the gap is
> mostly closed, not standing.** docs/verification/GOLDEN.md added 32 keyword-argument
> cases; 61 of `interned_name()`'s 74 arms are now exercised by keyword by at
> least one golden case, verified by re-running this same tamper (a different
> arm, `"dim"`) and watching `compare.py` turn red. 13 arms remain uncovered
> (docs/verification/GOLDEN.md §4) — 12 because no implemented op reads them yet, one
> (`generator`) left uncovered on purpose because no case can observe that
> particular tamper. Read docs/verification/GOLDEN.md before treating "the golden harness
> is blind to this code path" as still true of the whole path — it is no
> longer true of most of it.

(Restored from a `cp` backup, not `git checkout`; the source md5 matched the
pre-tamper file and all three gates returned to 197 / 2811 / 4203 afterwards.)

---

## 5. It is faster

### 5.1 Microbench — the measurement that resolves

Method is BIND.md §2's: minimum of 5 blocks of 20 000 calls after 200 warmups,
on a `(1, 6, 9, 64)` float32 tensor. Artefacts **alternate** old, new, old,
new… in fresh subprocesses, 4 rounds; only `_C.abi3.so` differs between them.
Load 2.48–4.44 throughout, recorded before every single round.

µs/call, minimum of 4 rounds, and the **ratio to upstream** either side:

| | upstream | old (`061e3d8`) | new | **old/new** | ratio to upstream, old → new |
|---|---|---|---|---|---|
| `.view(1, 6, 576)` | 0.784 | 1.802 | **1.638** | **1.100** | 2.30 → **2.09** |
| `.transpose(1, 2)` | 0.814 | 1.467 | **1.273** | **1.153** | 1.80 → **1.56** |
| `t + t` | 1.134 | 1.904 | **1.774** | **1.073** | 1.68 → **1.57** |
| `.unsqueeze(0)` | 0.755 | 1.351 | **1.209** | **1.118** | 1.79 → **1.60** |
| `.view((1, 6, 576))` | 1.048 | 1.947 | **1.801** | **1.081** | 1.86 → **1.72** |

**The bands do not overlap on any metric** — the *worst* new round beats the
*best* old round in all five cases (e.g. view: old 1.802/1.804/1.806/1.822,
new 1.638/1.642/1.643/1.666).

**Control** — the same final artefact run against itself under labels A and B,
through the identical harness, 3 rounds, ratio A/B per case:

```
view 0.9995   transpose 1.0138   add 1.0125   unsqueeze 1.0108   view-tuple 0.9966
```

Range **0.997–1.014**. The harness reads 1.00x to within 1.4% when there is
nothing to find, against the 7.3–15.3% the real comparison shows — the
smallest real effect (`add`, 1.073) is five times the largest control
deviation. A second control, run against the intermediate build earlier in the
session, read **0.996–1.008** on the same five cases.

Per-call saving is **0.130–0.194 µs**, average ~0.155 µs. The layer table of
§1.1, re-measured on the final build, shows where it came from:

| keyword-convention cost, µs/call | old | after (a) | after (a)+(b) |
|---|---|---|---|
| view | 0.147 | 0.114 | **0.042** |
| transpose | 0.237 | 0.137 | **0.051** |
| add | 0.200 | 0.130 | **0.006** |

Calling the door with keywords now costs very nearly what calling it
positionally costs, which was the whole object.

### 5.2 Prefill — the measurement that does not resolve, as predicted

SmolLM2-135M, float32, 6-token prompt; minimum of 5 timed passes after 2
warmups. **upstream, old and new alternate inside every round**, 4 rounds,
load 2.28–2.75.

| | min (ms) | ratio to upstream |
|---|---|---|
| upstream | 35.642 | 1.0000 |
| old (`061e3d8`) | 37.626 | **1.0557** |
| **new** | 37.498 | **1.0521** |

Pairwise within each round, excluding round 1 (upstream's first process reads
39.89 ms cold against 35.64/35.68/35.70 for the other three — a startup
outlier, not a datum):

```
old   1.0603  1.0604  1.0624
new   1.0521  1.0576  1.0610
```

**Those bands overlap, so prefill does not resolve this improvement**, and
§1.1 said in advance that it would not: ~0.29 ms predicted against a
round-to-round spread of ~0.3 ms. The prefill numbers are reported as a
did-not-regress check and as the source of the §4 checksum, not as evidence of
the speedup. The evidence is §5.1, where the effect is 5–15x the control.

This is the same wall BIND.md §5 hit, for the same reason, and it has not
moved: the machine cannot be made quiet, and an effect of half a percent on a
37 ms measurement is not reachable from here.

---

## 6. What is left, and what it would cost

**The keyword convention is now nearly free**, so the item BIND.md named is
closed. What remains in the ~2.0 ms residual, in size order:

1. **`resolve` + `_bind`, ~0.7–1.0 µs per call, ~1.5 ms per prefill.** Still
   the largest single item by a factor of five, and still in `bootstrap.py`
   rather than here. BIND.md §6 already costed the obvious move (merging the
   three frames) at ~0.15 ms against a duplicated resolution loop.

2. **`_tlv_get_addr`, ~12% of the dispatch loop's wall time.** macOS's dynamic
   thread-local accessor, and the largest single leaf in every profile taken
   here. It has two callers and they want different fixes:

   * pyo3's `AttachGuard::drop`, on every `extract::<PyTensorBase>()`. Reachable
     only by patching pyo3 or by the `disable-reference-pool` feature —
     **which makes dropping a `Py<T>` without the GIL abort the process.**
     That is a real safety change at exactly the boundary this work was told to
     be careful at, and it was not taken without an audit of every `Py<T>` drop
     in the crate.
   * CPython's own allocator (`_PyObject_Malloc` / `_PyObject_Free`), driven by
     object churn. Change (b) already removed one string allocation per
     argument per call from this, which is part of why it pays more than the
     leaf counts alone suggested.

3. **The `match op` in `aten_dispatch_inner`** — ~130 string arms. Not measured
   separately; rustc buckets by length and then compares, so it is unlikely to
   be near the items above, but it has not been ruled out.

**Not measured on device.** The change is platform-independent Rust with no
`cfg` in it, and the crate builds for `aarch64-linux-android` (§4), but the
host is the only place it has been timed. BIND.md §7 is the precedent for
turning that into a number; the same harness would do it.
