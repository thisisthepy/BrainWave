# ASYNCWORK — `async_op=True` stops being a lie, and the half that needed designing was ownership

`docs/distributed/COLLECT2.md` §7 measured this backend's async surface against upstream
gloo and pinned a divergence rather than hiding it:

| | upstream gloo | this backend, before |
|---|---|---|
| `is_completed()` before `wait()` | False | **True** |
| buffer valid before `wait()` | no | **yes** — `before_wait == after_wait` |
| `wait()` | blocks, returns True | no-op, returns True |

Every collective ran to completion **inside the call**. No value was wrong —
a correct caller could not observe an invalid buffer, because there was no
window in which one existed — but the contract was not upstream's. A caller
who overlapped a collective with local compute got no overlap, and a caller
who polled `is_completed()` got `True` on the first poll. §9 listed genuine
asynchrony as the first thing not built.

It is built. Measured on `darwin/arm64`, CPython 3.13, upstream torch 2.13.0
(`/Volumes/macMini/caches/spike-venv`), against upstream's own
`torch.distributed` over the **gloo** backend on loopback at world sizes 3 and
4. No aten op was added; the vendored tree was not edited by hand.

---

## 1. The thread was the easy half

The obvious reading of "make it async" is "run it on a thread", and that
reading is what this repository keeps recording as a failure. A worker thread
that finishes quickly makes every test pass. "The buffer is invalid before
`wait()`" then holds only on a slow day, and the suite is measuring the
scheduler rather than the contract. `CLAUDE.md` §5.5: a check that cannot fail
is not a check.

So the design question was not *when* the collective runs. It was **who owns
the output buffer while it runs**, and the answer had to make an early read
detectably wrong rather than accidentally right.

## 2. The design: the handle owns a staging clone, and publication is explicit

`AsyncWork` never lets the collective touch the caller's tensor.

    caller                          AsyncWork                     worker thread
    ------                          ---------                     -------------
    all_reduce(t, async_op=True)
                                    staged = t.clone()   <-- the handle owns this
                                    queue.put(body, staged)
    <- returns immediately                                        body runs on `staged`
    read t  ................................................ t is untouched: the input
    work.wait()
                                    t.copy_(staged)      <-- publication, and the
                                                             only write to `t` there is

Two consequences, both deliberate, both tested:

- **Reading the output before `wait()` gives the pre-collective contents**, on
  every run, at every speed. This is not a race that a fast worker wins: there
  is no code path on which the caller's buffer changes without the caller
  asking. That is what makes a test of it able to fail.
- **`is_completed()` is a synchronisation point too.** It reports whether the
  exchange finished, and if it has, it *publishes before returning True*. So a
  caller who polls to True may then read the buffer and find the answer, which
  is gloo's contract; a caller who polls to False still holds their own bytes.

Publication is guarded and idempotent. Waiting twice, or polling after
waiting, copies once. That is not tidiness — a second publication would
silently overwrite whatever the caller had put in their own tensor since, which
is a data-loss bug no return value would reveal (§6, `wait_twice`).

## 3. The collectives were not rewritten, and that was the point

Every collective body in `bootstrap.py` is still exactly the synchronous body
that `test_collect2.py` proved equal to gloo at world 3 and 4. Asynchrony is
added **around** them: each public method is renamed `_sync_<name>` and
replaced by a wrapper that either calls it inline — today's behaviour,
unchanged, for `asyncOp=False` and for `world_size == 1` — or hands it to the
worker with its **output argument swapped for a staging clone**.

Rewriting twelve bodies to be re-entrant would have put every one of
COLLECT2's measured results back in question to buy a property none of them is
about. The table of what is wrapped, and the positional index of each one's
output argument, is `_ASYNC_COLLECTIVES` in `bootstrap.py`.

`test_the_async_answer_is_the_synchronous_answer_and_both_are_upstreams`
asserts the two paths are **bit-identical**, which is the check that this
restructuring changed no number. Bit equality rather than a tolerance, because
there is no arithmetic difference between them to spend one on: it is the same
fold in a different thread.

## 4. One collective on the star at a time — and one place that enforces it

The topology is unchanged (`docs/distributed/FEDERATED4.md` §1): a star, hub at rank 0,
**one socket per peer carrying one length-prefixed frame per collective**. Two
exchanges in flight on that socket at once would interleave their frames and
each would read the other's bytes.

So asynchrony here buys **caller overlap, not wire parallelism**, and that is
stated rather than implied. The queue is served by exactly one worker thread,
which keeps the wire order equal to the issue order that c10d already requires
of every rank. And a *synchronous* collective issued from the calling thread
while async work is still queued would jump that line, so `_star_exchange`
drains the queue first.

That drain is in **one** place, and it did not start there — §8.

## 5. Where this is deliberately stricter than gloo

Gloo's output buffer becomes valid when the work does, whether or not anybody
asked. Here it becomes valid when somebody asks. A caller who issues a
collective, never waits, and reads the buffer gets their input on this backend
and gets gloo's answer-or-partial on gloo.

This is inside upstream's freedom — reading an un-waited buffer is undefined
in c10d — and it is a *different choice* within that freedom, which is exactly
the shape of `docs/distributed/COLLECT2.md` §4's `reduce` decision. It is written down
here for the same reason, and tested for the reason §8 records: **"undefined"
is not "unobserved"**, and a documented choice with no test is how this
repository's documents have drifted before.

The tests assert this about *ours* and only **record** upstream's early read.
Asserting that gloo leaves the input there would be inventing a contract
upstream does not offer.

## 6. The tests, and what each one can fail on

`rust/torch_c/pytests/test_asyncwork.py`, at world 3 and 4, against three real
process groups per size — the shim, gloo in float32, gloo in float64. The
tolerance is `docs/numerics/AGREE.md` §2's method, read off upstream's own
float32-vs-float64 error on the same collective, exact for integer dtypes; the
comparison helpers and the ephemeral-port spawn are **imported** from
`test_collect2` rather than copied, because two copies of a harness are two
chances for the two sides to stop being given the same thing.

| test | what makes it able to fail |
|---|---|
| `is_completed` False before `wait()` | Every rank but 0 sleeps 0.5 s, so at the moment rank 0 polls there is provably a peer that has not sent a byte. "Not finished" is a fact about the collective, not the scheduler. Also asserts the call *returned* in under 0.25 s, which separates an honest handle from one that blocked anyway. |
| `wait()` is required | Two reads: one immediately, one **a full second later**. The first alone would be a race this design happens to win; the second cannot be won by luck. Both must equal the rank's own input, and `after_wait` must differ from it so the test cannot pass on a backend that does nothing. |
| two overlapping collectives | Different lengths (4 and 2) and different reduce ops (SUM and PRODUCT), so a crossed publication cannot coincidentally look right. Waited on in **reverse** order: waiting on B must publish B and leave A untouched, which is what says publication follows the caller and not the queue. |
| `wait()` twice is a no-op | The return values are the easy half. The half with teeth scribbles on the buffer after completion, then calls `is_completed()` and `wait()` again — the scribble must survive. |
| async == sync == upstream | The only assertion here that involves no timing at all. |
| integers exact | No tolerance to spend, and the expected sum is computed by the test itself as well as read off gloo — so it does not go through upstream at all. |
| abandoned handle | The hub arrives 0.3 s late, four times over, holding every leaf's async exchange open across its own synchronous call. Reaching the assertion is half the test: a dropped handle that wedged its peers would time out instead. |
| `all_gather` async | The output is a list *of lists*. A staging clone that shallow-copied the outer list would hand the worker the caller's own tensors — which no `allreduce` test could reach. |

The pinning test in `test_collect2.py` was **inverted, not deleted** — §7.

## 7. What stays synchronous, and why

- **`allreduce_partial` and `allgather_partial`.** They return
  `(Work, missing_ranks)`. The survivor set is the hub's verdict about who was
  in *this* collective, and it is not a thing a caller can be handed later —
  `torchnative.nn.federated`'s `Engine` reads it to choose a divisor before it
  reads any value. A deferred verdict would be a divisor chosen after the
  average.
- **Anything whose options struct has no `asyncOp`.** The wrapper reads
  `getattr(opts, "asyncOp", False)`, so a caller who does not ask gets exactly
  today's behaviour. `federated` builds a bare options object and is therefore
  untouched by this round; that it is untouched is checked by the whole of
  `test_shim.py` staying green.
- **`world_size == 1`.** There is no peer to wait for, and a thread would be
  a thread to make the answer no later than it already is.
- **An output passed by keyword.** The wrapper finds the output positionally.
  Nothing in this tree passes it any other way, and guessing which keyword it
  was would make ownership unprovable — so that call runs synchronously rather
  than saying something false about itself.

## 8. The sabotage — nine nullifications, and two that escaped

Each nullification was applied to `bootstrap.py`, **rebuilt** (it is
`include_str!`'d at Rust build time, so a round that skipped the rebuild would
be testing the previous binary), and the two suites re-run.

| nullification | caught by |
|---|---|
| run every collective inline — remove the worker thread entirely | 5 tests in `test_asyncwork`, and the inverted `test_collect2` pin |
| publish from the worker instead of at a synchronisation point | `wait_is_required` (**the one-second read only**), `overlapping` |
| publication no longer idempotent | `wait_twice`, via the scribble |
| staging clone made shallow for nested lists | `wait_is_required`, `overlapping` |
| remove the queue drain from **`_star_exchange` only** | **nothing** |
| remove the queue drain from **the wrapper only** | **nothing** |
| remove **both** drains | `abandoned_handle`, `async_all_gather` |
| remove the single drain, after the two were collapsed into one | `abandoned_handle`, `async_all_gather` |
| a second worker thread on the same sockets | **deadlock** — no rank finishes. Caught, but as a hang rather than an assertion: the suite's 600 s per-spawn timeout turns it into a `RuntimeError` and a red test. Measured with a bounded world-3 probe rather than the full suite, because nine tests each re-spawning into a deadlock is ninety minutes of waiting for the same answer. |

The last row is the argument for §4's single worker thread, and it is worth
being precise about what "caught" means there: nothing asserts, everything
*stops*. Two threads pulling from the same queue put two exchanges on the same
socket and each blocks reading a reply the other consumed. A hang is a red
suite here because every spawn has a timeout — but a suite whose only defence
against a design error is a timeout is a suite that reports it slowly, and that
is why the invariant is stated in `_drain_async`'s docstring and in §4 rather
than left for the timeout to discover.

**Two escapes, and they are different in kind.**

The first is the one worth reading. The drain existed in *two* places — the
wrapper's synchronous branch and `_star_exchange` — and each silently covered
the other, so **neither could be nullified on its own**. The suite could see
the property and neither of its two implementations. A reviewer running either
single nullification would have concluded the guard was untested and the
answer would have been half right: the *property* was tested, but no test
distinguished the guard that was doing the work. The two were collapsed into
one choke point at `_star_exchange`, which every collective body reaches, and
the single nullification now goes red. Defence in depth and testability were in
tension here and testability won, which is the choice `CLAUDE.md` §5.5 asks for.

The second escape was in the test rather than the code. The first version of
`abandoned_then_synchronous` issued the async and the synchronous collective
back to back with every rank in lockstep, so the async one had always finished
before the synchronous one began. The hazard window was **never open**, and
removing both drains left the whole suite green. The probe now makes the hub
arrive late, four times over, which holds the window open on every leaf. This
is the same shape as COLLECT2 §8's `reduce` escape: the behaviour was
understood and written down, and the test could not reach the state that
would have contradicted it.

## 9. What is still not built

- **Wire parallelism.** §4. One collective at a time on the star, so a caller
  who issues two async collectives gets them serialised on the wire — they
  overlap with the *caller's* compute, not with each other. A ring or tree
  backend overlaps them for real, and that is not built here (COLLECT2 §3.1).
- **`get_future()`.** Still refuses by name. A `torch.futures.Future` is a
  composition surface — `then`, chaining, the DDP comm hooks — and there is no
  caller for it in this tree; building it on top of this queue would make it
  look like the comm-hook path works.
- **Cancellation and per-work timeouts against the wire.** `wait(timeout)`
  bounds how long the *caller* waits, not how long the exchange runs. There is
  no way to abort an exchange that a peer never answers; the peer-loss path is
  still COLLECT2's, reported at `wait()` rather than at issue.
- **Everything COLLECT2 §9 lists that is not asynchrony**: uneven `all_to_all`
  splits, the bitwise reduce ops, the `_coalesced` spellings, `send`/`recv`,
  secure aggregation, differential privacy, `new_group`.

## 10. Reproducing

```sh
PATH="$HOME/.cargo/bin:$PATH" PYTHON=/Volumes/macMini/caches/spike-venv/bin/python \
    bash rust/torch_c/pytests/run.sh
```

Every group runs in real `subprocess.Popen`s on an **ephemeral port bound and
released by the parent**, never a fixed one — several rounds of this repository
share this machine. Every spawn kills its children on every path, including the
raising ones. The shim workers assert `hasattr(torch._C, "_aten_implemented")`
on the way in and the gloo workers assert its absence, and
`test_every_rank_really_ran_the_shim_and_the_oracle_really_ran_upstream` checks
the same fact from outside.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py AsyncWork present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _tree_clone present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _tree_copy present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _drain_async present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _async_queue present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _async_shutdown present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _ASYNC_COLLECTIVES present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _make_async_collective present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_asyncwork.py test_is_completed_is_false_before_wait_for_a_collective_that_has_not_finished present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_asyncwork.py test_wait_is_required_because_the_buffer_before_it_holds_the_input present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_asyncwork.py test_two_overlapping_collectives_do_not_corrupt_each_other present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_asyncwork.py test_wait_on_an_already_completed_handle_is_a_no_op_and_not_an_error present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_asyncwork.py test_the_async_answer_is_the_synchronous_answer_and_both_are_upstreams present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_asyncwork.py test_an_abandoned_handle_does_not_wedge_or_corrupt_the_next_collective present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_asyncwork.py test_asynchrony_is_not_a_property_of_allreduce_alone present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_asyncwork.py test_integer_collectives_are_exact_through_the_async_path present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_collect2.py test_async_op_is_genuinely_async_on_both_sides_and_neither_publishes_early present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _require_sum absent -->
<!-- DOCWATCH: count golden_ops_covered ge 302 -->
