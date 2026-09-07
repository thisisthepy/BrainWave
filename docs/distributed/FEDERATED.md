# FEDERATED — one round, across two operating-system processes

`README.md` §2 named three layers. Two closed on 2026-09-02: serialisation
(`docs/models/SAVE.md` — `torch.save` works and upstream reads what it writes
bit-for-bit) and transport (`docs/distributed/TRANSPORT.md` — `world_size = 2` over a real
socket). This document is the third, `torchnative.nn.federated`, and what it
records is one round of `FedAvg` between two processes that share no memory,
what that round was checked against, and what refuses by name instead of being
approximated.

Measured 2026-09-02, host `darwin/arm64`, CPython 3.13, upstream torch 2.13.0
(`/Volumes/macMini/caches/spike-venv`). The vendored tree was not touched.

---

## 1. The trap is the whole round: `FedAvg` at `world_size = 1` is the identity

```
sum(w_k * d_k) / sum(w_k)      k ∈ {0}      =   d_0
```

An aggregator that ignores its weights returns `d_0`. One that drops its peer
returns `d_0`. One whose body is `return table` returns `d_0`. **At one rank
all three are correct**, so a green test at that size is evidence of nothing —
and `docs/design/DESIGN.md` §11.1 puts aggregation a layer *above* the process group,
which is exactly where an in-process aggregator would look plausible.

Two threads would be no better. They share the tensors, so a "collective" that
touched only local memory would also pass.

So the rule this round was built under, and the rule the tests enforce:

> **Every test of aggregation involves two `subprocess.Popen`s that share
> nothing but a TCP socket.**

and the acceptance check is not that the distributed path returned something.
It is that the average computed *across the ranks* equals the same average
computed **centrally**, in a third process, from the two deltas those ranks
dumped.

`world_size = 1` is not served. Four doors refuse it, with the reason and not
only the fact:

```
FedAvg.aggregate      NotImplementedError: ... a world of one, reached through FedAvg.aggregate.
Delta.publish         Averaging a single client's delta with itself is the identity function,
federated.agree       so this would return what it was handed and prove nothing about the
Engine.participate    aggregation -- a test at this size passes whether the weights are
                      honoured, ignored, or never read. ...
                      Check: torch.distributed.get_world_size() == 2, ...
```

`Engine.participate` refuses **before** the local epochs rather than after.
Refusing afterwards would leave the model adapted and the round
uncontributable, which is a worse place to stop than not starting.

---

## 2. The acceptance measurement

Two OS processes, rendezvous through `TCPStore` over `tcp://127.0.0.1:<port>`,
`init_process_group(backend="local", ...)` — the ordinary front door, not a
private one. Each rank builds the same `TinyLM` from a deterministic generator
(so the bases match), adapts it with `adapt.wrap(model, method=adapt.Tent())`
for three steps on **different local token ids**, and carries a different
weight.

```
rank 0   ids [[3, 7, 1, 19, 5]]     weight 3      entropy 3.0572 → 3.0069 → 2.9306
rank 1   ids [[11, 2, 23, 0, 14]]   weight 7      entropy 3.0882 → 3.0288 → 2.9127
```

The deltas that come out are the ones the existing `adapt`/`delta` machinery
produces — `Delta.value`, recorded by `Adapted.step`. Nothing here builds a
delta by hand.

### 2.1 Distributed against central, element for element

`(3·d0 + 7·d1) / 10`, computed in the parent test process on **upstream** torch
from the two ranks' JSON, against what each rank's `Delta.publish` returned:

```
norm1.weight
  d0        [-0.2921586,  0.1427829, -0.0670061, ...]
  d1        [-0.3230541,  0.3030677, -0.1410002, ...]
  rank 0    [-0.3137855,  0.2549823, -0.1188020, ...]
  rank 1    [-0.3137855,  0.2549823, -0.1188020, ...]
  central   [-0.3137855,  0.2549823, -0.1188020, ...]

norm2.weight
  d0        [ 0.6016294,  0.6393167,  0.7781166, ...]
  d1        [ 0.5883340, -0.0168402, -0.0461791, ...]
  rank 0    [ 0.5923226,  0.1800068,  0.2011096, ...]
  rank 1    [ 0.5923226,  0.1800068,  0.2011096, ...]
  central   [ 0.5923226,  0.1800068,  0.2011096, ...]

torch.equal(rank0, central)   True        torch.equal(rank1, central)   True
torch.equal(rank0, rank1)     True
```

**`torch.equal`, not a tolerance.** Every operation on both sides is a
correctly-rounded IEEE `float32` multiply, add or divide over the same inputs,
so one ulp of difference would be a real disagreement about the arithmetic and
not noise. Two things make that reachable:

* the scale and the divisor are applied as **0-dim tensors of the table's own
  dtype** rather than as Python floats, so there is one dtype and one rounding
  rather than a promotion rule deciding the width of the intermediate;
* `tolist()` on a `float32` tensor yields Python floats holding the exact
  `float32` value, `json` round-trips a float through `repr`, and rebuilding a
  `float32` tensor narrows back to the identical bits — so the numbers compared
  are the ranks' own and not approximations of them.

The two ranks land on the same bits as each other for a third reason: the
transport's `all_reduce` adds in the opposite order on the two sides, and
float addition is commutative.

### 2.2 The controls, because "some average" is the failure mode

Every way `FedAvg` can go quietly wrong produces *an* average. So:

| control | measured |
|---|---|
| the two ranks' deltas differ | asserted — identical deltas make every average equal to either one |
| the two ranks share a base | asserted — otherwise the offsets are incomparable (§3) |
| **weighted ≠ unweighted** | `max abs` gap **0.0409** on `norm1.weight`, **0.1649** on `norm2.weight` |
| the aggregate is neither operand | gap to `d0` **0.577**, to `d1` **0.247** on `norm2.weight` |
| both ranks end with the same aggregate | `torch.equal`, and the `Engine`'s parameter tables compare equal as JSON |

The third row is the one that makes the acceptance test able to fail for the
right reason. Without a *measured* separation between 3-and-7 and 1-and-1, an
aggregator that dropped its weights would satisfy §2.1's comparison as long as
the central side dropped them too. The unweighted figure is itself produced by
the distributed path (`FedAvg(weighted=False)` over the same two deltas), so
`weighted=False` is pinned as a different computation rather than a synonym.

### 2.3 The `Engine` road

```python
engine = federated.Engine(model, method=adapt.Tent(), lr=4.0,
                          aggregator=federated.FedAvg())
report = engine.participate([{"input_ids": ids}] * 3, weight=n_local_samples)
```

```
rank 0   <Round rank 0/2, 3 step(s) over 2 parameters, weight 3 of 10, |local|=1.557 |aggregate|=1.631>
rank 1   <Round rank 1/2, 3 step(s) over 2 parameters, weight 7 of 10, |local|=1.817 |aggregate|=1.631>
```

The local norms differ (the ranks trained on different data) and the aggregate
norms do not — `1.6311656276032023` on both. After the round the two ranks hold
**the same weights**, which is what makes it federated learning rather than two
devices training separately, and the parameters the `Engine` left behind equal
`base + aggregate` computed on the low-level road in the same process. So
`Engine` is a *use* of `Delta.publish` rather than a second implementation that
could drift from it.

### 2.4 Sabotage — each test was made to fail

Five defects were introduced one at a time, the two-process round re-run, and
the reds counted. `cp` backups; `git checkout` was not used.

| defect | went red |
|---|---|
| `FedAvg` ignores its weights (`w = 1.0`) | **2** — acceptance, and the weights control (`[0.0, 0.0]`: the gap it measures collapsed) |
| `FedAvg.aggregate` returns `table` (the `world_size = 1` identity) | **5** — acceptance, weights, both refusal tests, and the Engine |
| the schema `agree` call removed | **1** — the mismatch test, and the probe reported **ACCEPTED**: rank 0's `{"norm1.weight": ...}` and rank 1's `{"norm2.weight": ...}` were summed into a number with no exception |
| the base `agree` call in `Delta.publish` removed | **1** — same shape: two different bases aggregated silently |
| `Engine` installs the local delta instead of the aggregate | **1** — the two ranks ended the round with different models |

The two `ACCEPTED` rows are the point of §3: both defects produce a *number*,
and nothing downstream of them raises.

---

## 3. Two premises that fail silently, and one collective that checks them

`all_reduce` sums element-wise whatever it is handed. That makes two things
un-noticeable unless they are checked:

**Which parameters this round covers.** Two ranks whose deltas cover different
names, in a different order, or at different shapes still produce a result.

**The base the deltas are offsets from.** FedAvg averages
`w_k^local − w_global`; that only means anything if every rank subtracted the
same `w_global`. Two ranks that started from different models produce an
average of incomparable quantities and report success.

Both are answered by one `int64` `all_reduce` of a 56-bit `sha256` digest, and
the test is `sum == value * world_size`:

```
h0 + h1 == 2·h0   ⟺   h0 == h1
```

Three details, each of which was a way to get this wrong:

* **`hashlib`, not `hash()`.** CPython salts `hash` of a string per
  interpreter, so the built-in would disagree between two ranks holding
  identical tables — a false alarm indistinguishable from the real one.
* **56 bits**, so `value * world_size` stays exact in the `int64` tensor it is
  reduced in.
* **The equality test is only valid for a world of two.** At three it accepts
  `(h−1, h, h+1)`. So `agree` refuses a larger world rather than quietly
  weakening, and names `all_gather` as what would settle it for any world —
  which refuses above `world_size 1` (`docs/distributed/TRANSPORT.md` §3).

The digest over values uses `Delta._bytes`, the same little-endian encoding
`Delta.persist` writes. So two ranks agree here exactly when a byte comparison
of what they would persist agrees.

**Both ranks reduce before either compares**, which is why the group survives a
refusal: the test re-runs a full aggregation afterwards and gets the same
answer as the first one.

---

## 4. What was scoped out, by name

The full stack is aggregation + the round loop + participant selection +
failure handling. This round is the first of those. The rest refuse, and each
refusal says what it would take.

| | why it refuses rather than approximates |
|---|---|
| `Engine(rounds=n)` for `n != 1` | Round two is not a loop around round one. The delta each rank contributes has to be opened over the **aggregated** weights, and `Adapted.online()` deliberately never re-snapshots its base — so every later round would measure from the original model and contribute the accumulated offset again. What already exists is the part that makes a second round *provable*: after round one both ranks hold bit-identical weights (§2.3), and `Delta.publish`'s base check would confirm it. What is missing is re-opening the delta per round and a test that the ranks stay identical across the boundary |
| `Engine(select=...)` | Participant selection needs a world larger than the two this transport carries, and a sub-group to run the collective over. `torch.distributed.new_group` is what would build it and the backend refuses above `world_size 2`. At two ranks every selection rule that is not "both" leaves one, and `FedAvg` refuses a world of one |
| `Engine(allow_missing=...)` | Averaging over whichever ranks arrived makes the divisor a number nobody chose, and the round reports success. **As built, a rank that does not arrive makes the collective raise** — the socket times out after 30 s — and never produces a partial average. What is missing is a *policy*, not a mechanism to notice |
| `FedAvg()` with no `weight` | There is no default sample count and none is inferred from a batch. Weighting two clients equally when one holds ten times the data is a different model, silently. `FedAvg(weighted=False)` is how a caller says they meant it — and it refuses a `weight` if one is passed, because it would be accepted and ignored |
| `weight <= 0`, `nan`, `inf` | A rank weighted 0 contributes nothing while still counting as having participated |
| a table holding integers | A delta holds floating parameters; averaging an integer table rounds every intermediate |
| an empty table | A round that contributes no parameters would complete and change nothing |
| an unrecorded `Delta` | It is the zero offset. Publishing it contributes nothing and is counted at full weight in the average anyway |
| `participate` twice on one `Engine` | It is one round; a second call opens no new delta and contributes the first round's offset again |
| secure aggregation, differential privacy, gradient compression | **Not offered at all.** No surface here takes a key, an epsilon or a codec |
| aggregators other than `FedAvg` | FedProx, FedAdam, SCAFFOLD: none. `Engine` takes any object with `.aggregate`, so a fourth is a class and not a change here |


> **Correction (2026-09-06, `docs/distributed/FEDERATED3.md`).** Three rows of the table above are no longer true, and are left in place rather than edited away for the reason §6 gives. `Engine(select=...)` no longer refuses at construction: `federated.cohort` agrees the participant set across the ranks and only a *proper subset* refuses. `Engine(allow_missing=...)` has become `on_missing=`, whose `'refuse'` policy is implemented — a lost peer raises `federated.RankDropped` and the round is undone. And there are now two aggregators besides `FedAvg`: `FedAvgM` and `FedProx`. Secure aggregation and differential privacy are still not offered, and now refuse by name.

**Also not done, and not refused because there is nothing to refuse:** no
device other than CPU (`docs/design/DESIGN.md` §11.1's fourth layer), no measurement
of what a round costs on a real model, and no run on a phone. Every number here
is from a `TinyLM` with 24 vocabulary entries.

---

## 5. A transport limit found from above, and left where it belongs

`ProcessGroupLocal.allreduce` serialises a tensor to JSON and **both ranks
send before either receives**. A payload larger than the kernel's socket buffer
therefore blocks both until the 30 s socket timeout fires. Measured cold — the
first collective of a fresh pair of processes:

```
   2,836,968 B     completes in 0.17 s
   4,053,011 B     BrokenPipeError on one rank, TimeoutError on the other,
                   after 30 s
```

**The wall is not a constant, and that is the finding.** The same 8,106,264 B
payload that deadlocks as a pair's *first* collective goes through in 0.23 s if
smaller ones ran first — loopback socket buffers auto-tune upward, so the
threshold depends on the history of the connection rather than on the size
alone:

```
cold, one collective only        400,000 f32   8.1 MB   deadlock
after a 60k→320k ramp            400,000 f32   8.1 MB   0.229 s
```

Two things follow, and the second is a decision.

1. **A size-based bound stated as a property of federation would be a guess
   dressed as a measurement.** So `federated` refuses above **2,000,000 B**,
   below the cold measurement, and the message names the transport, gives both
   numbers, and says the fix is to interleave the send with the receive or to
   chunk inside the collective — *"nothing at this layer does"*.

2. **Chunking in the aggregator was considered and not done.** It would be
   exact, and it would work — and it would hide a nondeterministic hang one
   layer below by making the layer above avoid it. The refusal is what names
   the next thing to build (`docs/design/DESIGN.md` §6); a workaround is what stops it
   being named.

The check costs nothing in the common case: `numel * 26 + 2` is an upper bound
on the JSON length (a `float64` holding a `float32` value has a `repr` of at
most 24 characters), and the exact length is measured only when that bound is
passed. A Tent-adapted SmolLM2 delta is 35,136 floats — 709,229 B on the wire,
well under.

---

## 6. Two things this round found in code it did not write

**`bootstrap.py` defines `class ProcessGroupLocal` twice, back to back.** The
first body is a docstring and nothing else, and is immediately rebound by the
second. It is harmless — Python keeps the last — but it is dead, and a reader
looking for the `world_size == 1` branch finds a class that has none.

**`docs/models/ADAPT.md` says `Delta.publish` still refuses**, in three places (§1's
prose, the lifetime table at line 166, and the gap table at line 621). Those
sentences were true when written and this round made them false, which is
exactly the mechanism `docs/verification/AUDIT.md` found six times. Corrections are added in
place rather than edited away.

---

## 7. Regression

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh     354 ok   (before 348, +6)
                                              DOCWATCH: PASS -- 274/274
$PY tools/golden/compare.py                   8126/8126, ops=185
```

**No aten op was added**, so `tools/golden/cases.py` gains nothing — this round
is Python above the dispatcher, and the only Rust it touched is none.

`354` is this round's snapshot and is asserted as a lower bound only, the way
every other document here treats `smoke_ok`.

<!-- DOCWATCH: count smoke_ok ge 354 -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/nn/federated/__init__.py FedAvg present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/nn/federated/__init__.py Engine present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/nn/federated/__init__.py agree present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/delta/__init__.py publish present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_fedavg_over_two_processes_equals_the_same_average_computed_centrally present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_federated_refuses_a_world_of_one_by_name_at_every_door present -->

---

## 8. What is not known

* **No round has been run on a real model.** Everything here is a `TinyLM` of
  24 vocabulary entries and 12 hidden units. A SmolLM2 Tent delta is 709 KB on
  the wire, which §5's guard admits — but nobody has sent one.
* **Nothing was measured across machines.** Both ranks are on `127.0.0.1`. The
  transport is a real socket, but it has never crossed a network interface, so
  latency, MTU and partial reads are untested at their real sizes.
* **Nothing was measured on a device.** `docs/perf/PERF_ANDROID.md`'s lesson — that
  wins arrive at different sizes on a phone — has no analogue here yet because
  there is no cost measurement at all.
* **A rank that dies mid-round was not tested.** What is argued is that the
  collective raises rather than averaging partially, and §4 records that the
  socket timeout is the mechanism; the argument was not run as an experiment,
  because with no timeout knob on `TCPStore.wait` the experiment is a 30-second
  hang inside the suite.
* **`float32` only.** A `bfloat16` delta would go through `Delta._bytes`'s
  reduced-precision path on the digest and through the transport's
  `tolist()`/`torch.tensor` round trip on the values; neither was exercised.
* **The digest is not a security boundary.** It detects an accident. Two ranks
  that wanted to disagree could collide 56 bits, and nothing here is trying to
  stop them — that is §4's "secure aggregation", which is absent.
