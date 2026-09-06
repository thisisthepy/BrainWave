# FEDERATED4 — the transport at N ranks, and the three things that were waiting on it

`docs/FEDERATED3.md` named the transport as the single blocker for everything
left in the federated layer: `ProcessGroupLocal` implemented `world_size` 1 and
2 and refused above it, and three things refused *behind* that refusal. This
round extends the transport to `world_size >= 1` and closes all three.

Measured on `darwin/arm64`, CPython 3.13, upstream torch 2.13.0
(`/Volumes/macMini/caches/spike-venv`). No aten op was added; the vendored tree
was not touched.

| | |
|---|---|
| `ProcessGroupLocal` at `world_size >= 3` | **built** — a star of loopback sockets, hub at rank 0 |
| a proper subset cohort | **built** — served by attendance and abstention, not by `new_group` |
| a survivor set of ≥ 2 after a dropout | **built** — three ranks, one leaves, two aggregate |
| `on_missing='average_arrived'` | **built** — with `min_participants=k`, and refusing below a world of three |
| secure aggregation, differential privacy | **not offered**, refusing by name with the chain they need (unchanged from FEDERATED3 §7) |

---

## 1. The topology, and why it is a star

Every rank other than 0 opens one socket to rank 0 and talks to nobody else.
Rank 0 binds an ephemeral port, publishes it on the single `Store` key
`pg_local_port`, and accepts until `size - 1` peers have arrived.

A ring would halve the wire volume of an allreduce and is what a real backend
does. This is not that, for one reason: **a star makes the reduction order a
property of one process rather than an emergent property of N.** In a ring the
order in which contributions are folded is a consequence of who was ready
when; here it is a line of code on the hub, and that line is the contract §2
tests.

Each leaf writes its own rank as the first four bytes it sends, because
`accept` returns connections in whatever order the kernel finished the
handshakes. Without that greeting the hub would have to infer rank from arrival
order — which would be right most of the time, and that is the shape of defect
this layer exists to refuse.

The hub reads from **every** leaf before it writes to any of them. That is what
makes it deadlock-free at any size: `sendall` returns when the kernel took the
bytes, not when the peer read them, so a leaf blocked mid-send is only ever
waiting for a hub that is already draining. This is the same argument the
two-rank even/odd ordering made, generalised.

`_WIRE_SAFE_BYTES` was **not** widened. N ranks did not force it: the hub reads
one leaf at a time and the bound is per-collective payload, not per-world, so
the number that was measured at two ranks is the number that still applies.
The refusal in `_check_wire` is unchanged.

## 2. The ordering contract

> **The hub folds contributions in ascending rank order,
> `((x_0 + x_1) + x_2) + …`, and sends one result back to every rank. Nobody
> adds anything locally.**

Float addition is not associative, so "sum the contributions" is not a
specification. An implementation that summed in arrival order would give a
different answer on a different day, and — the part that matters — **a test
that tolerated that difference would also tolerate a lost rank.** "Close
enough" is exactly how a dropped contribution hides.

So the contract is bit-exact against the same expression written centrally, in
rank order, on upstream torch. The probe is chosen so the two orders are not
merely different but visibly so in float32:

    rank 0: +1.0      rank 1: +1e8      rank 2: -1e8

    ascending rank order   ((1.0 + 1e8) + -1e8)  ==  0.0
    any order that folds the large terms first    ==  1.0

`test_the_transport_reduces_three_ranks_in_rank_order_and_says_so` asserts
`0.0` and asserts that `1.0` — the arrival-order answer — is *not* what came
back. It also asserts the premise, that the two orders really do differ, so the
test cannot pass by coincidence.

Since one fold happens on the hub and one result is sent back, every rank ends
holding the same bytes. That is asserted directly: the three ranks' tables are
compared element-wise, not within a tolerance.

## 3. `agree` stopped being a sum

`agree` was one `all_reduce(SUM)` and the test `total == value * world`. That is
an equality test **only at two ranks**: `h0 + h1 == 2*h0` iff `h0 == h1`. At
three it accepts `(h-1, h, h+1)` — three ranks that disagree about the schema,
the base or the cohort, summing to exactly what agreement would have summed to.
FEDERATED3 §5 named this and refused the larger world rather than weaken the
check.

It is now one `int64` `all_gather` and an element-by-element comparison. The
refusal is gone and the check is *stronger* rather than wider.
`test_agree_is_exact_at_three_ranks_where_the_old_sum_check_was_not` feeds
exactly `(h-1, h, h+1)` and requires **every** rank to refuse, not only the two
whose own product misses.

The holders are pre-filled with `-1`, a value no digest can take, so a slot the
transport never wrote is visible as one rather than read as a zero that agrees.

## 4. The sabotage

Three faults were injected into `_star_exchange`/`_sum_fold`, rebuilt, and the
N-rank tests re-run. Each was reverted afterwards; `bootstrap.py` is
`include_str!`'d, so each round is a real rebuild and reinstall.

| sabotage | result |
|---|---|
| **A.** hub folds `present[:-1]` — the last rank's contribution is dropped from *both* the sum and the gather, and `missing` still reports `[]` | **red, 5/5.** Rank 0 dies in `allgather`: the absent rank's slot is `None` and `torch.tensor` refuses it. Structural, not arithmetic. |
| **A′.** `_sum_fold` iterates `payloads[1:-1]` — the **sum** silently loses one term, the gather is intact, nothing structural notices | **red, 2/5** at first. The value assertions caught it: the ordering probe returned `1e8` instead of `0.0`, and the survivor average returned a different model. |
| **B.** hub folds in **descending** rank order — every contribution present, only the order of the float additions changed | **red, 1/5** — and it is the ordering test alone. Nothing else can see it, which is the point of having it. |

A′ is where this round found something. Two of the five stayed green, and one
of them was `test_a_proper_subset_cohort_aggregates_over_the_subset_and_not_the_world`
— **for a real reason**: its cohort was `[0, 1]`, so the rank whose
contribution A′ drops is rank 2, and rank 2 is abstaining. An abstention
contributes a zero, so dropping it is arithmetically invisible.

That is honest but it is a blind spot in a test that looks like it should see
this. So a second cohort was added, `[1, 2]` — one that **excludes the hub**.
Now rank 0 is the one contributing zero and rank 2 carries weight 2.0, and A′
turns that test red as well (3/5). It is also a stronger structural claim on
its own: the hub is where the fold happens, not a privileged member of every
cohort.

`test_agree_is_exact_at_three_ranks` stays green under A′, correctly — `agree`
travels on the gather, not the sum.

## 5. A proper subset, without `new_group`

Every rank attends every collective. The ranks outside the cohort contribute
`federated.ABSTAIN`: a zero table at weight zero. The divisor is then the
cohort's weight and the result is the cohort's weighted mean, exactly.

`ABSTAIN` is a distinct value rather than `0.0`, which every aggregator here
refuses — a rank weighted zero contributes nothing while still counting as
having participated, and that is a caller mistake worth refusing. This is the
same arithmetic said on purpose.

Acceptance: the subset's aggregate must equal `(3·d0 + 7·d1)/10` **and differ
from** `(3·d0 + 7·d1 + 2·d2)/12`, both computed centrally on upstream torch
from the ranks' JSON. A cohort that was recorded but not honoured passes the
first and fails the second. The unselected rank reports `participated=False`,
`steps=0`, `weight=0.0` — it does not train, which is what selection *is* — and
all three end the round holding the same model.

**What `new_group` would still add** is that the unselected ranks are not
*reached* at all. That is a property of the wire, not of the arithmetic, and it
is what a real deployment wants, because a client that is asleep cannot attend
a barrier. It is not built here.

**A refusal was removed because it could not fire.** `cohort()` had two: a
cohort of one, and "a proper subset needs a world of at least three". In a world
of two the only proper subsets have one member, so the first had already refused
every input that could reach the second. A refusal that cannot fire is not a
refusal; its reason was folded into the one that does, and
`test_participant_selection_is_agreed_across_the_ranks_and_a_subset_refuses`
now asserts the merged message — including that it no longer names
`ProcessGroupLocal at world_size N` as the next thing to build, since that
thing now exists.

## 6. `on_missing='average_arrived'`

FEDERATED3 §4.1 refused this for a measured reason: at two ranks the partial
average is the survivor's own delta to **6e-8**, so it would hand back the
survivor's weights as if aggregation had happened. At three it is real.

It is built on two shim-only spellings, `allreduce_partial` and
`allgather_partial`, which return `(Work, missing_ranks)`. There is no upstream
spelling because upstream's allreduce either completes over the world or is an
error. The survivor set is the **hub's** verdict and travels in the same
envelope as the data, so the survivors divide by the same number rather than by
whatever each of them timed out on.

Four doors still refuse, and each says something different:

| | |
|---|---|
| `on_missing='average_arrived'` with no `min_participants` | `TypeError` — "however many arrived" is the divisor nobody chose |
| `min_participants=1` | `ValueError` — a survivor set of one is a world of one |
| `min_participants=` under `on_missing='refuse'` | `TypeError` — accepted and ignored |
| the policy in a world of 2 | `NotImplementedError`, carrying the 6e-8 measurement |

Two further guards are in the mechanism rather than the policy:

- **The survivor set may not change inside one round.** A divisor that changed
  halfway is not the divisor of any average, so `_Arrival.record` compares
  every collective's verdict against the first and raises `RankDropped` if they
  differ.
- **A rank that contributed and then died before the result reached it is
  refused, not patched over.** The ranks it was folded with have already been
  sent a result that counts it, so there is no verdict the hub can report that
  every survivor also holds.

Acceptance: three ranks, rank 2 leaves with `os._exit` after a collective has
demonstrably completed (`live_sum == 6.0`), and the two survivors' aggregate
must equal `(3·d0 + 7·d1)/10` centrally — **and must not equal either
survivor's own delta**, which is exactly what it would have been at two ranks.
Below the floor (`min_participants=3`) the same round refuses and is undone.

## 7. What still refuses, by name

- **Point-to-point `send`/`recv`/`recv_anysource`.** The topology is a star, so
  a message between two leaves would have to be relayed by the hub — and a
  relay the hub can read is not the primitive secure aggregation needs.
  Refused rather than faked.
- **Secure aggregation.** Needs pairwise secrets between clients (so:
  point-to-point, above), a key agreement, threshold secret sharing so a
  dropout does not destroy the sum, and an unmasking round. The digest in
  `agree()` detects an accident, not an adversary: 56 bits, and nothing here is
  trying to stop two ranks that want to collide it.
- **Differential privacy.** A DP guarantee is a *number* — (ε, δ) at a stated
  granularity. Producing one needs per-example gradient clipping inside the
  local step, noise calibrated to the clipping norm, and an accountant over the
  rounds. This stack's backward produces one gradient per parameter over the
  batch, not per example. Adding a noise term without that would produce a
  model that is worse and a guarantee that is absent, and report both as
  success.
- **A cohort of one, and a world of one, at every door.** FedAvg over one delta
  is that delta.
- **`new_group`** — see §5. The subset is served arithmetically; not reaching
  the unselected ranks at all is not.
- **Reductions other than `SUM`** above `world_size` 1. `AVG` would be `SUM`
  over a divisor, which is the one number a federated aggregator must choose
  for itself.
