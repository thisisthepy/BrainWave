# COLLECT2 — the collectives above one rank, and the sentence that described them wrongly in both directions

`docs/platform/RELEASE_0_0_13a0.md` §5 said:

> `world_size >= 3` is `allreduce(op=SUM)` only, over loopback on one machine.
> Other collectives, other reduce ops, secure aggregation and differential
> privacy refuse by name.

This round was asked to widen that. The first thing it did was **run it**, and
two thirds of it was already false — in opposite directions, which is why
neither half had been noticed. Three ranks were doing more than the sentence
claimed, and four collectives were doing something much worse than refusing.

Measured on `darwin/arm64`, CPython 3.13, upstream torch 2.13.0
(`/Volumes/macMini/caches/spike-venv`), against upstream's own `torch.distributed`
over the **gloo** backend on loopback at the same world sizes with the same
inputs. No aten op was added; the vendored tree was not edited by hand.

---

## 1. What §5 actually described

The probe ran every collective `torch.distributed` exposes, at `world_size = 3`,
in three real processes on the shim, and then the same probe on upstream gloo.

| | §5 said | measured |
|---|---|---|
| `all_gather`, `all_gather_into_tensor` | refuses | **already worked**, rank-ordered and exact |
| `barrier` | refuses | **already worked** — a real trip through the star |
| `allreduce(SUM)` on `int64` | — | already exact |
| `allreduce` with MIN/MAX/PRODUCT/AVG | refuses by name | **refused by name**, correctly |
| `broadcast`, `reduce`, `gather` | refuses by name | **refused by name**, correctly |
| `reduce_scatter` | refuses by name | **returned a wrong answer, silently** |
| `scatter` | refuses by name | **returned a wrong answer, silently** |
| `all_to_all`, `all_to_all_single` | refuses by name | **returned a wrong answer, silently** |
| `reduce_scatter_tensor` | refuses by name | raised `cannot broadcast [6] to [2]` from inside `copy_` |
| secure aggregation, differential privacy | refuses by name | refuses by name (unchanged, `docs/distributed/FEDERATED4.md` §7) |

The four silent rows are the finding. They did not check `self._size` at all.
Their bodies were the `world_size = 1` identity — `output.copy_(sources[0])` —
and that identity is *correct* at one rank, which is why they had a passing
test. At three ranks each one returned the calling rank's own input:

```text
reduce_scatter, three ranks, chunks 10*r+0..5

  upstream gloo   [30, 33]   [36, 39]   [42, 45]
  this backend    [ 0,  1]   [10, 11]   [20, 21]      <- its own chunk 0, unreduced

all_to_all, three ranks

  upstream gloo   [[0],[10],[20]]  [[1],[11],[21]]  [[2],[12],[22]]
  this backend    [[0],[ 1],[ 2]]  [[10],[11],[12]] [[20],[21],[22]]   <- its own input
```

**A promised refusal is worse than a missing one when the truth is a wrong
answer**, because the reader of §5 has been told there is nothing there to
check. The refusal was in the release note; it was never in the code.

`reduce_scatter_tensor` is a fifth case with its own shape: it *did* stop, but
from the wrong layer, with a message about broadcasting a `[6]` into a `[2]`.
A caller reading that looks at their tensor and never at their world size,
which is where the mistake is. A refusal by accident is not a refusal.

## 2. The reduce ops, and the refusal that was two refusals

`SUM`, `PRODUCT`, `MIN`, `MAX` and `AVG` now fold above one rank. `BAND`, `BOR`
and `BXOR` refuse by name; `PREMUL_SUM` and `UNUSED` refuse as they always did.

The `AVG` case is the one worth stating, because widening it looks like
weakening a refusal and is not. The old message read:

> Only SUM is implemented above world_size 1. AVG would be SUM over a divisor,
> which is the one number a federated aggregator must choose for itself.

That sentence is true of `federated.FedAvg`, whose divisor is a **weight total
the caller supplies** and which must not guess it. It is not true of `c10d`'s
`ReduceOp.AVG`, whose divisor is the **world size** and is nobody's choice.
One refusal was standing in for two different arguments, and only one of them
was about this layer. The federated aggregator's refusal is untouched and
`test_shim.py` still holds it; what moved is the `c10d` one.

`test_avg_is_the_sum_over_the_world_and_not_over_whoever_arrived` checks the
divisor against the world rather than only against gloo, so a divisor that
happened to equal three at three ranks and not at four fails.

**`MIN` and `MAX` are folded elementwise on the JSON payloads rather than
through a tensor**, and this is exact rather than a shortcut: an extremum
*selects* a contribution and never combines two, so no rounding happens at any
width. It is also the one place where the ascending-rank contract is vacuous —
min and max are associative and commutative, so every bracketing agrees.
(`aten.minimum` is not implemented in this shim and `aten.maximum` is; doing
both the same way keeps them symmetric rather than making `MIN` the odd one.)

`SUM` and `PRODUCT` accumulate **in the tensor's own dtype**, which keeps
`docs/distributed/FEDERATED4.md` §2's ordering contract observable — see §5.

## 3. What was built

| | |
|---|---|
| `allreduce` with `MIN`, `MAX`, `PRODUCT`, `AVG` | **built**, matching gloo at world 3 and 4 |
| `broadcast` from **any** root | **built** — `src=1` and `src=world-1`, not only the hub |
| `reduce` to any root | **built** — root's value specified, non-roots untouched (§4) |
| `gather` to any root | **built** |
| `scatter` from any root | **built** — was silently wrong |
| `reduce_scatter`, `reduce_scatter_tensor` | **built** — was silently wrong |
| `all_to_all`, `all_to_all_single` | **built**, equal splits — was silently wrong |
| `all_gather`, `all_gather_into_tensor`, `barrier` | **already worked**; now tested against gloo |
| `all_to_all_single` with uneven splits | refuses by name |
| `BAND`/`BOR`/`BXOR`, `PREMUL_SUM`, `UNUSED` | refuse by name |
| `reduce_scatter_*_coalesced` | refuse by name above one rank — were silently copying |
| `send`/`recv`/`recv_anysource`, secure aggregation, DP, `new_group` | refuse by name, unchanged |

Three of these are **new capability**, four are **defect fixes to code that was
returning wrong numbers**, three were **already working and only documented**,
and the rest are **refusals that replaced silence**. `CLAUDE.md` §5.3 asks for
that split rather than a count of eight.

### 3.1 How they are built, and what it costs

The topology is unchanged: a star, hub at rank 0, one frame in and one frame
out per collective (`docs/distributed/FEDERATED4.md` §1). Everything here is expressed as a
fold on the hub, so the deadlock-freedom argument is the existing one and not a
new one per collective.

That has a price, and it is stated rather than hidden:

- **`broadcast` and `scatter` send from every rank**, then discard all but the
  root's. The star delivers *one* frame to every rank, so the root cannot hand
  each leaf a different chunk in a single exchange; `scatter` broadcasts the
  whole list and each rank selects its own index.
- **`all_to_all` sends the full `size x size` matrix to everybody**, and each
  rank reads its own column. Every rank therefore sees every other rank's
  payload, which a real all-to-all does not do. This is why `send`/`recv` stay
  refused rather than being built on top of it: the capability is there, the
  *privacy* secure aggregation needs is not, and building `send` here would
  make that look solved.
- **`reduce_scatter` does `size` folds on the hub** where a real backend does
  one per rank in parallel.

On loopback these are memcpys. On a wire they would not be, and a ring or tree
is what a real backend uses. That is not built here.

## 4. `reduce` promises only the root, and this backend says which nothing

Upstream gloo leaves the non-root buffers holding whatever partial its tree
produced. Measured at three ranks on this host, where the total was `6.0`:

```text
rank 0 (root)  6.0        rank 1  5.0        rank 2  3.0
```

Those are not answers. They are the absence of a promise, and reading one as a
result is the same mistake as everything in §1. This backend leaves the
non-root buffers **untouched** — inside the same freedom, and a different
choice from upstream's.

`test_reduce_leaves_the_non_root_buffers_exactly_as_it_found_them` asserts it,
and also asserts that upstream *disagrees*, so the test cannot quietly become a
test of nothing if gloo's tree ever changes. It exists because of §6: nullifying
the root check was the one sabotage the first draft of the suite did not catch.

## 5. The ordering contract, extended to `PRODUCT`

`docs/distributed/FEDERATED4.md` §2 pinned the ascending-rank fold for `SUM` with
`1.0 + 1e8 - 1e8`, which is `0.0` in rank order and `1.0` in any order that
folds the large terms first. Float multiplication is not associative either,
and the probe for it is sharper:

```text
    rank 0: 2**-120      rank 1: 2**100      rank 2: 2**100

    ascending rank order   ((2**-120 * 2**100) * 2**100)  ==  2**80    finite
    any order that multiplies the two large factors first  ==  inf     (float32 overflow)
```

A tolerance cannot paper over the difference between `2**80` and `inf`.
`test_the_product_fold_is_in_ascending_rank_order_and_says_so` asserts `2**80`
and asserts the premise — that `2**100 * 2**100` really does overflow float32 —
so it cannot pass by the two orders happening to agree.

**Upstream gloo is deliberately not the oracle for this one.** Its fold order is
a consequence of its tree and is not a promised property, so the two are allowed
to differ. What is pinned is that this backend's order is the one its own
docstring claims.

## 6. The tolerance is derived, and mostly comes out at zero

`docs/numerics/AGREE.md` §2's method: a threshold picked by eye is not a result, so each
collective's is read off **upstream's own float32-vs-float64 error on that same
collective with those same inputs**, floored at 8 float32 ulp.

Applied here, the interesting outcome is that **most of the derived tolerances
are zero**. `broadcast`, `all_gather`, `gather`, `scatter`, `all_to_all`, and
`MIN`/`MAX` select and move numbers; they never combine two. Upstream's float32
and float64 runs of them are bit-identical, the derived width is zero, and
spending the 8-ulp floor anyway would be choosing a number after all. So those
are held to **bit equality** — which is a reading of the measurement, not a
decision. Integer dtypes are exact everywhere, in every op.

Only `SUM`, `AVG`, `PRODUCT` and `reduce_scatter` have a real width, and there
the derived threshold applies.

The inputs are built by a **pure-Python LCG seeded per rank**, not by
`torch.manual_seed`. The shim and upstream are two different RNG
implementations, and "same seed" only means "same numbers" if the generator is
the same object. Values are quantised to a multiple of `2**-12` so every one is
exactly representable in both float32 and float64 — the input is then not itself
a source of difference, and the float64 run measures the collective's own
arithmetic rather than a rounding that happened before it started.

`test_the_tolerance_is_derived_from_upstream_and_is_not_a_free_parameter` pins
the floor at 8 ulp and pins the claim that the exact-by-measurement collectives
really do have an upstream error of exactly zero. If a future change makes
`broadcast` lossy, that goes red before any value test does.

## 7. The async surface diverges, and the divergence is pinned rather than hidden

> **Superseded by `docs/distributed/ASYNCWORK.md` (later round).** The divergence measured
> below is **closed**: the collectives are genuinely asynchronous now,
> `is_completed()` is False before `wait()` on both sides, and the buffer
> before `wait()` holds the caller's own pre-collective input rather than the
> answer. The table is kept as the record of what was true when it was taken
> — it is not a description of the current backend. The test named here was
> **inverted**, not deleted, and is now
> `test_async_op_is_genuinely_async_on_both_sides_and_neither_publishes_early`.


A collective that is secretly synchronous passes every value test in §6 and is
still not upstream's contract. So it was measured:

| | upstream gloo | this backend |
|---|---|---|
| `async_op=True` returns | a `Work` | a `Work` |
| `is_completed()` before `wait()` | **False** | **True** |
| buffer valid before `wait()` | no | **yes** — `before_wait == after_wait` |
| `wait()` | blocks, returns True | no-op, returns True |

**These collectives run to completion inside the call.** That is not a defect in
the values — a correct caller cannot observe an invalid buffer, because there
is no window in which one exists. It is a real difference in the *contract*: a
caller who overlaps a collective with local compute gets no overlap, and a
caller who polls `is_completed()` gets `True` on the first poll.

That test asserted **both sides** — that ours was complete on return and
that upstream's was not — specifically so that a later round making these
genuinely asynchronous would turn it red. That is what happened, and it is the
claim that changed, not the test that was wrong. It was updated in place, still
asserting both sides; `docs/distributed/ASYNCWORK.md` §6 lists which assertions were
inverted and why each is still a real check.

The paragraph that stood here said genuine asynchrony was **not built**, and
that it would need a `Work` that owns the buffer until `wait()` — "the
buffer-ownership half is the part with teeth, because a caller who reads early
must get something the implementation chose to give them rather than a
half-written tensor". That was the right diagnosis and it is how it was built:
`AsyncWork` runs the collective against a private staging clone and publishes
onto the caller's tensor only at a synchronisation point, so an early read is
deterministically the input rather than a race it happens to win. What it does
*not* buy is wire parallelism — the star still carries one collective at a
time, so two async collectives overlap with the caller's compute and not with
each other. `docs/distributed/ASYNCWORK.md` §4.

## 8. The sabotage

Every collective landed here was nullified, rebuilt — `bootstrap.py` is
`include_str!`'d, so each round is a real rebuild and reinstall — and the suite
re-run. **Fifteen nullifications, fifteen caught**, but not on the first pass.

| nullification | caught by |
|---|---|
| `MIN` folds as `MAX` | reduce-ops, sweep |
| `AVG` drops its division | reduce-ops, avg-divisor, sweep |
| `PRODUCT` folds in descending rank order | product-ordering **alone** |
| `broadcast` keeps the hub's payload, not the root's | broadcast-root, sweep, regression |
| `reduce_scatter` keeps slot 0 instead of `self._rank` | sweep, regression |
| `reduce_scatter_tensor` skips its size check | misshapen-refusal |
| `scatter` keeps slot 0 instead of `self._rank` | sweep, regression |
| `all_to_all` transposes the wrong way | sweep, regression |
| `gather` fills on every rank, not only the root | sweep |
| `barrier` returns `Work()` immediately | barrier-blocks **alone** |
| uneven-split refusal removed | uneven-refusal |
| bitwise refusal removed | reduce-ops |
| `reduce` copies onto every rank, not only the root | **nothing, at first** |

The ordering and barrier rows each have exactly one test that can see them,
which is the point of having them: nothing else in the suite distinguishes a
correct fold order from an incorrect one, or a real barrier from a reported one.

**The last row is what the sabotage round was for.** `reduce` writing its answer
onto every rank is a real behaviour change and the first draft of the suite was
blind to it, because the probe returned `None` off-root — honest, since upstream
leaves those buffers undefined, but "undefined" is not "unobserved". This
backend makes a specific choice inside that freedom and §4 documents it, and a
documented choice with no test is how this repository's documents have drifted
before. `test_reduce_leaves_the_non_root_buffers_exactly_as_it_found_them` was
added and the nullification re-run against it.

## 9. What is still not built, refusing by name

- ~~**Genuine `async_op`.** §7.~~ **Built** in a later round —
  `docs/distributed/ASYNCWORK.md`. What remains unbuilt from that round is wire
  parallelism (two async collectives still serialise on the star),
  `get_future()`, and cancelling an exchange already in flight.
- **`all_to_all` with uneven splits.** Not a harder transpose — a different
  collective. Each rank must know every other rank's split vector before it can
  place its own chunk, and that is an exchange this backend does not do.
  Silently treating them as equal is what the old body did to the whole
  operation.
- **`BAND`, `BOR`, `BXOR`.** Defined only on integral dtypes — upstream gloo
  raises `Cannot use ReduceOp.BAND with non-integral dtype` — and the float
  tables this layer's federated callers reduce are exactly the ones upstream
  would refuse. There was no caller to build it for.
- **The `_coalesced` spellings** of `allreduce`, `allgather` and
  `reduce_scatter`. Coalescing is a batching of the single form and there is no
  caller for it in this tree. `reduce_scatter_single_coalesced` and
  `reduce_scatter_tensor_coalesced` were **silently copying** above one rank and
  now refuse.
- **`send` / `recv` / `recv_anysource`.** Unchanged, and deliberately not built
  on the new `all_to_all` — see §3.1.
- **Secure aggregation, differential privacy, `new_group`, a cohort of one.**
  Unchanged; `docs/distributed/FEDERATED4.md` §7 is still the reason for each.
- **A ring or tree topology.** §3.1 states what the star costs.

## 10. Reproducing

```sh
PATH="$HOME/.cargo/bin:$PATH" PYTHON=/Volumes/macMini/caches/spike-venv/bin/python \
    bash rust/torch_c/pytests/run.sh
```

`rust/torch_c/pytests/test_collect2.py` runs three process groups at each of
world 3 and world 4 — the shim, upstream gloo in float32, and upstream gloo in
float64 — on an **ephemeral port bound and released by the parent**, never a
fixed one. Several rounds of this repository share this machine, and a fixed
port is the same shared-mutable-state defect as the fixed log path that has
already cost a round here. Every spawn kills its children on every path,
including the raising ones.

The probe bodies are **one source string shared by both sides** rather than two
copies, because the whole claim is that the two were given the same inputs and
two copies are two chances to drift. Only the four lines that choose a backend
differ. The shim workers assert `hasattr(torch._C, "_aten_implemented")` on the
way in and the gloo workers assert its absence, and
`test_every_rank_really_ran_the_shim_and_the_oracle_really_ran_upstream` checks
the same fact from outside.

Some probes are shim-only, and not for tidiness: **gloo does not survive them.**
Its `reduce_scatter_tensor` shape check is a C++ `Check failed:` that calls
`abort()`, so a misshapen input takes the rank down with SIGABRT rather than
raising anything Python can catch, and `send` with no matching `recv` blocks
until the harness times out.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _reduce_fold present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _reduce_kind present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _reduce_scatter_fold present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _pick_fold present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _check_root_n present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _extremum present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _require_sum absent -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_collect2.py test_every_collective_matches_upstream_gloo_at_world_three_and_four present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_collect2.py test_the_tolerance_is_derived_from_upstream_and_is_not_a_free_parameter present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_collect2.py test_the_product_fold_is_in_ascending_rank_order_and_says_so present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_collect2.py test_barrier_actually_blocks_rather_than_reporting_that_it_did present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_collect2.py test_async_op_is_genuinely_async_on_both_sides_and_neither_publishes_early present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_collect2.py test_reduce_leaves_the_non_root_buffers_exactly_as_it_found_them present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_collect2.py test_the_collectives_that_used_to_return_their_own_input_no_longer_do present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_collect2.py test_all_to_all_single_refuses_uneven_splits_by_name present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_collect2.py test_point_to_point_still_refuses_by_name_and_was_not_weakened present -->
<!-- DOCWATCH: count golden_ops_covered ge 302 -->
