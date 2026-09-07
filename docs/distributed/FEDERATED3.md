# FEDERATED3 — the four things FEDERATED2 scoped out

`docs/distributed/FEDERATED.md` landed one round of `FedAvg` between two OS processes;
`docs/distributed/FEDERATED2.md` landed N rounds of it over `Delta.re_snapshot`. Both
closed with the same four names in the "scoped out" table: **participant
selection**, **dropout policy**, **aggregators other than FedAvg**, and
**secure aggregation / differential privacy**. This document is what happened
to those four. Two are built, one is built by half and the half that is
missing is missing for a reason that is measured rather than asserted, and the
fourth is still not offered — with the chain of things it needs written down
instead of a shrug.

Measured 2026-09-06, host `darwin/arm64`, CPython 3.13, upstream torch 2.13.0
(`/Volumes/macMini/caches/spike-venv`). The vendored tree was not touched, and
no aten op was added.

| | |
|---|---|
| aggregators other than FedAvg | **built** — `FedAvgM` (server momentum) and `FedProx` (a proximal term on the *local* objective) |
| dropout policy | **built** — `on_missing='refuse'`, `federated.RankDropped`, and the round is *undone* rather than left half-applied |
| participant selection | **half built** — the cohort is agreed across the ranks; a proper subset refuses and names `world_size N` as the next thing |
| secure aggregation, differential privacy | **not offered**, and now refusing by name from `Engine(secure_aggregation=…)` / `Engine(differential_privacy=…)` with the dependency chain |

---

## 1. The trap, twice over, and what it forces

`FedAvg` at `world_size = 1` is the identity, and **everything added here
inherits that**. `FedAvgM`'s velocity over a world of one is a running sum of
the rank's own deltas. `FedProx`'s server step *is* `FedAvg`, so at any world
size a class that only overrode `aggregate` would be `FedAvg` wearing a name.
A cohort of one is the identity by definition. And the failure a dropout
policy exists to prevent — dividing by whoever arrived — **produces the
identity at two ranks**, because two minus one is one.

So every test here is two `subprocess.Popen`s that share nothing but a TCP
socket, through `import torchnative.distributed` and
`init_process_group(backend="local", init_method="tcp://127.0.0.1:<port>")`,
and every acceptance check is against the same computation done **centrally**,
in the parent process, on upstream torch, from the JSON those ranks dumped.
Every subprocess prints `shim`/`upstream` on stderr and asserts the shim.

Wall time: the aggregator/cohort round is **2.1 s** for both ranks, the
dropout round **2.0 s**.

---

## 2. `FedAvgM` — the first aggregator whose answer depends on the past

```
v_k  <-  beta * v_(k-1)  +  mean_k          mean_k = sum(w_j d_j) / sum(w_j)
round installs  eta * v_k
```

Hsu, Qi and Brown (2019). The client half is untouched: every rank still
contributes `w^local - w_global`, and the group still forms the weighted mean
of those with `all_reduce(SUM)`.

**Why this is a second aggregator and not a knob.** One round cannot
distinguish it from `FedAvg` — at `k = 0` the velocity is the mean. So the
acceptance test does not try: it runs **three rounds** and checks each against
the recursion computed centrally, `torch.equal`.

```
round 0  norm1.weight  mean[:3] [-0.313785,  0.254982, -0.118802]
                          v[:3] [-0.313785,  0.254982, -0.118802]   equal ✓
round 1  norm1.weight  mean[:3] [-0.007018,  0.005147, -0.095361]
                          v[:3] [-0.289425,  0.234631, -0.202283]   equal ✓   |v-mean| 0.2824
round 2  norm1.weight  mean[:3] [ 0.442526, -0.186574, -0.029901]
                          v[:3] [ 0.182044,  0.024594, -0.211956]   equal ✓   |v-mean| 0.2665
round 1  norm2.weight                                               equal ✓   |v-mean| 1.1697
round 2  norm2.weight                                               equal ✓   |v-mean| 2.3477
```

`torch.equal`, not a tolerance, for `docs/distributed/FEDERATED.md` §2.1's reason: the
scale, the divisor and `beta` are all applied as **0-dim tensors of the
table's own dtype**, so there is one rounding rather than a promotion rule.

### 2.1 The controls

| control | measured |
|---|---|
| `FedAvgM(momentum=0)` is `FedAvg` | **bit for bit** — the same table through both classes, equal as JSON |
| momentum actually carries | from round 1 the aggregate is **not** the round's mean: `‖v − mean‖∞` is 0.2824 / 1.1697 (round 1) and 0.2665 / 2.3477 (round 2), asserted `> 1e-3` |
| the aggregate is neither operand | asserted at every round |
| the two ranks' deltas differ | asserted at every round |
| **both ranks hold the same velocity** | asserted as JSON equality at every round |

The last one is worth its own line. **Nothing reduces the velocity.** It stays
in step because the mean is already identical on both ranks and the update
applied to it is deterministic — but "it cannot drift" is an argument, and a
server state that drifted would still return a table of the right names and
shapes and still report success. So it is checked.

A schema change between rounds refuses: a momentum buffer silently restarted
at zero for a renamed parameter is the `FedAvg` answer reported as the
`FedAvgM` one.

---

## 3. `FedProx` — the difference is in the local objective, so the aggregator refuses without one

FedProx (Li et al. 2020) adds `mu/2 ‖w − w_global‖²` to what each client
minimises, which on the gradient is `mu (w − w_global)`. **Its server step is
FedAvg's, unchanged.** That is the whole trap: a `FedProx` class that only
overrode `aggregate` would compute FedAvg's weighted mean, at every world
size, for ever, and nothing downstream could tell.

So the term goes where it belongs — on the local step — through a new hook:

```python
torchnative.adapt.Adapted.add_grad_hook(hook)   # hook(wrapper, params, names)
```

called after the gradients are on the parameters and **before**
`optimizer.step()`, which is the only moment at which "what this step will
apply" exists as an object. `Engine` arms the aggregator if it has an `arm`;
`FedProx.aggregate` **refuses outright if it never armed**, because the
low-level `Delta.publish` road would otherwise hand it the table and take back
FedAvg's answer under FedProx's name.

### 3.1 The shape of the term, not the fact of a difference

`mu (w − w_global)` is exactly zero at the **first** local step, because
`w == w_global` there. So:

```
                       step 1 max|prox − plain|      step 3 max|prox − plain|
rank 0   norm1/norm2       0.0        0.0                0.0548     0.1395
rank 1   norm1/norm2       0.0        0.0                0.0535     0.2335

                       ‖delta‖ prox      ‖delta‖ plain
rank 0                    1.28786           1.55681
rank 1                    1.53059           1.81749
```

The first row is what a constant nudge, a flipped sign, or a term computed
against the live weights instead of the base would all break — and it is exact
zero, not a tolerance. The last two columns are the second half: the term pulls
back toward the weights the round started from, so the contributed delta is
*smaller*. `mu = 0.05` with `lr = 4.0` — a shrink factor of 0.2 per step, which
is why this damps rather than overshooting past the base.

The aggregate over the two ranks is still checked against the central weighted
mean of the two prox deltas, `torch.equal`: FedProx changed what was
contributed, not how it was combined.

`FedProx(mu=0)` refuses — at `mu = 0` it *is* FedAvg, and a caller who means
FedAvg should say FedAvg rather than reach it through a tuning knob.

---

## 4. Dropout — the divisor nobody chose

The failure this guards is an aggregator that **silently divides by however
many ranks arrived**. It returns a table of the right names and shapes and
every caller downstream sees success.

`docs/distributed/FEDERATED.md` §8 listed "a rank that dies mid-round was not tested",
because with no timeout knob on `TCPStore.wait` the experiment looked like a
30-second hang inside the suite. **It is not.** Rank 1 exits with `os._exit(0)`
— no atexit, no flush, a device that lost power rather than one that said
goodbye — and rank 0 notices when `_recv_all` reads zero bytes from a closed
socket. The whole two-process test takes **2.0 s**, not 30.

```
live_sum == 3.0                     the group was real before it was broken
rank 0 raises   RankDropped: ... rank 1 did not report during
                federated.agree(the base these deltas are offsets from ...),
                so this round has no aggregate. The collective over 2 rank(s)
                did not complete, and no partial average was produced ...
e.missing == (1,)                   which rank, as data and not only as prose
engine.on_missing == "refuse"       the default, so this is policy not accident
before == after                     bit-identical: the round was undone
```

Three parts, and the third is the policy:

1. **Mechanism.** `_collective` translates the three faces of a lost peer —
   `RuntimeError('connection closed')`, `BrokenPipeError`, a 30 s
   `socket.timeout` — into `federated.RankDropped`, which names the rank, the
   collective, and the fact that no partial average exists. All three mean the
   same thing at this layer and none of them says so; they name the socket.
2. **Policy.** `Engine(on_missing='refuse')` is the default and is
   implemented. `on_missing='average_arrived'` refuses.
   `allow_missing=True` now points at `on_missing`.
3. **The round is undone.** The local epochs had already moved the model.
   Leaving them would keep an update no other rank has, and **the two ranks
   would silently stop holding the same weights** — the one property that makes
   this federated learning rather than two devices training alone. So
   `participate` reverts to the delta's base, which is the last aggregate every
   rank agreed on, and marks the Engine used.

### 4.1 What the silent divisor would have returned, measured

```
true aggregate    (3 d0 + 7 d1) / 10
partial           (3 d0) / 3           = d0, to 6e-8

                       max|partial − d0|      max|partial − true|
norm1.weight              0.0                      0.1432
norm2.weight              6.0e-8                   0.5770
```

So the "partial average" is rank 0's own delta: the **identity** — the same
degenerate answer `world_size = 1` gives, arrived at by a socket close instead
of by a decision. That is why `on_missing='average_arrived'` cannot be honestly
served here even as an experiment: at two ranks its output is the operand, and a
test of it would pass with no aggregation at all.

One thing fell out of writing that control. `x * 3 / 3` **is not `x`** in
float32 — 6.0e-8 on `norm2.weight`. The first draft asserted `torch.equal` and
went red for a reason with nothing to do with aggregation.

---

## 5. Participant selection — the half that does not need a bigger world

A round picking a subset of the available ranks needs two things, and only one
of them needs a transport this project does not have.

**The half that is built: the ranks have to agree.** Two ranks holding
different cohorts complete every collective and produce a weighted mean, over a
cohort neither of them chose — the same silent shape as the schema and base
disagreements `docs/distributed/FEDERATED.md` §3 found. And a rule as ordinary as "sample
half the clients at random" disagrees whenever the ranks seed differently,
which is the default. So `federated.cohort(select, group)` evaluates the rule
and reduces its digest over the same `agree` collective:

```
ValueError: the ranks disagree about which ranks this round selected
            (rank 0 proposed [0, 1]). Rank 0 holds digest 20542060814578571;
            the 2 ranks sum to 90942765663793498, and would sum to
            41084121629157142 if they agreed.
```

`Engine(select=…)` runs it **before the local epochs**, for the same reason the
world-size check does: a cohort the ranks disagree about should refuse before
the model has moved. `Round.cohort` records what the round actually aggregated
over.

**The half that refuses.** A proper subset needs the collective to run on a
sub-group — `torch.distributed.new_group` — and `ProcessGroupLocal` refuses any
world but 1 and 2 (`docs/distributed/TRANSPORT.md` §3). At two ranks every subset that is
not both leaves one, and `FedAvg` over a world of one is the identity. So there
is no version of this that can be *tested* here, and a version that cannot be
tested is the thing this whole layer refuses to ship. The message names
`ProcessGroupLocal at world_size N, then new_group over a subset of it` as the
next thing to build.

---

## 6. Secure aggregation and differential privacy — a round of its own, and why

Both were "not offered at all". They still are, but they now refuse from a
named surface rather than by absence, and each names the chain it needs.

**Secure aggregation** (Bonawitz et al. 2017) is pairwise masks between
clients. That needs **point-to-point `send`/`recv`, which the transport refuses
by name**, plus a key agreement, threshold secret sharing so that a dropout does
not destroy the sum — the same problem §4 is about — and an unmasking round.
It is not one flag's worth of work, and it is *downstream of the dropout
policy*, not parallel to it.

**Differential privacy** is a **number**: `(epsilon, delta)` at a stated
granularity. Producing one needs per-example gradient clipping inside the local
step, noise calibrated to the clipping norm, and an accountant over the rounds.
Two of those are missing at the bottom: this stack's backward produces one
gradient per parameter **over the batch**, not per example, and the sampling
rate an accountant integrates over is set by participant selection — which §5
refuses. Adding a noise term without the rest produces a model that is worse
and a guarantee that is absent, and reports both as success.

So yes: **a round of its own**, and it is ordered after `world_size N`.

The digest in `federated.agree` is not a security boundary and never was — 56
bits, detecting an accident, with nothing trying to stop two ranks that want to
collide it.

---

## 7. Sabotage — each test was made to fail

Five defects, introduced one at a time into the aggregation layer, the
two-process tests re-run, and the reds counted. `cp` backups; `git checkout`
was not used.

| defect | went red |
|---|---|
| `FedAvgM` ignores its velocity (`v = mean`) | **2** — the momentum acceptance test, and the `momentum=0` control (whose second half measures that momentum *does* something) |
| `cohort` does not agree the set across the ranks | **1** — the disagreement arm accepted two different cohorts |
| a lost peer is not translated into `RankDropped` | **1** — the dropout test |
| the `FedProx` proximal term is a no-op | **1** — the prox test (step 3 stopped differing from the unregularised run) |
| the dropped round is left half-applied | **1** — the dropout test, on `before == after` |

The fourth and fifth are the ones worth noting: both leave a path that
*completes and returns a plausible answer*. A no-op proximal term is FedProx
reported as working; a half-applied dropped round is two ranks holding
different weights with nothing raised.

---

## 8. What was scoped out, by name

| | why it refuses rather than approximates |
|---|---|
| a cohort that is a **proper subset** | needs `new_group` over a world larger than 2; at two ranks every subset is a world of one, where FedAvg is the identity (§5) |
| `on_missing='average_arrived'` | the divisor is chosen by a socket timeout; and at two ranks the survivor set is one, so its output is the operand (§4.1) |
| secure aggregation | needs point-to-point send/recv (refused by the transport), key agreement, threshold sharing, an unmasking round (§6) |
| differential privacy | needs per-example gradients (this backward is per-batch), a clipping norm, an accountant over a sampling rate selection defines (§6) |
| `FedAdam`, `FedYogi`, `SCAFFOLD` | **not built, and not refused by name** — `Engine` takes any object with `.aggregate`, so each is a class and not a change here. The adaptive three would also need `sqrt` on the server path, where `torch.equal` against a central computation is a claim about one more kernel; nobody has checked that |
| gradient compression | no surface takes a codec |
| optimiser reset between rounds | unchanged from `docs/distributed/FEDERATED2.md` §4 — the state carries forward and nobody has measured the difference |
| a cross-machine, non-CPU or on-device round | unchanged. Every number here is a `TinyLM` of 24 vocabulary entries on `127.0.0.1` |

**Not done and not refused, because there is nothing to refuse:** no
measurement of what any of this costs on a real model, and no run on a phone.

---

## 9. What was changed

| file | change |
|---|---|
| `torchnative/src/main/torchnative/adapt/__init__.py` | `+Adapted.add_grad_hook` / `.grad_hooks`; the step calls them after the gradients are assigned and before `optimizer.step()` |
| `torchnative/src/main/torchnative/nn/federated/__init__.py` | `+FedAvgM`, `+FedProx`, `+cohort`, `+RankDropped`, `+_collective`; `Engine` gains `select=`, `on_missing=`, `secure_aggregation=`, `differential_privacy=`, unwinds a dropped round, and `Round` gains `.cohort` |
| `rust/torch_c/pytests/test_shim.py` | 8 new tests, 2 new two-process fixtures |

---

## 10. Regression

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh     397 ok   (before 389, +8)
                                              DOCWATCH: PASS -- 350/350
$PY tools/golden/compare.py                   8509/8509, ops=203
```

**No aten op was added** — this round is Python above the dispatcher, and the
only Rust it touched is none. `397` is a lower bound only, the way every other
document here treats `smoke_ok`.

<!-- DOCWATCH: count smoke_ok ge 397 -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/nn/federated/__init__.py FedAvgM present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/nn/federated/__init__.py FedProx present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/nn/federated/__init__.py RankDropped present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/nn/federated/__init__.py cohort present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/adapt/__init__.py add_grad_hook present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_fedavgm_over_two_processes_equals_the_server_momentum_computed_centrally present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_fedprox_leaves_its_first_local_step_untouched_and_moves_the_rest present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_participant_selection_is_agreed_across_the_ranks_and_a_subset_refuses present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_a_dropped_rank_refuses_by_name_and_leaves_the_round_uncontributed present -->

---

## 11. What is not known

* **Nothing was run at a world larger than two**, so every claim about
  `world_size N` — that cohort agreement generalises, that a survivor set of
  two or more makes `average_arrived` meaningful — is a design statement and
  not a measurement.
* **`FedAvgM`'s velocity was never reduced**, only shown to agree. Two ranks
  that fell out of step for some other reason (a different `momentum`, a
  restart) would not be caught by anything here; the digest covers the delta,
  not the server state.
* **`FedProx` was measured at one `mu` on one model.** That the prox delta is
  smaller is a measurement at `mu = 0.05, lr = 4.0`; at `mu * lr > 2` the term
  overshoots the base and the sign of that comparison flips. Nothing refuses
  that combination, because what makes it wrong is the product and the
  optimiser, and no measurement here bounds it.
* **The dropout was one shape of failure.** A rank that hangs rather than
  exiting still waits 30 s for the socket timeout, and that path was reasoned
  about, not run. A rank that dies *between* two collectives of the same round
  was not tried either.
* **`float32` only, CPU only, loopback only** — unchanged from
  `docs/distributed/FEDERATED.md` §8.
