"""`async_op=True` is genuinely asynchronous here, and this is what holds it.

`docs/COLLECT2.md` §7 measured the previous state and pinned it rather than
hiding it: `async_op=True` returned a handle that was **already complete**,
because every collective ran to completion inside the call. Upstream gloo's
handle is not complete until `wait()`. That was a real difference in contract
even though no value differed, and §9 listed genuine asynchrony as the first
thing not built.

It is built now, and `docs/ASYNCWORK.md` is the design. The half that needed
designing was never the thread. It was **ownership**, and the failure mode this
file exists to make impossible is the one this repository keeps recording: a
worker thread that finishes fast enough that every test passes, so that
"the buffer is invalid before `wait()`" is true only on a slow day and the
suite is measuring the scheduler instead of the contract.

So the ownership is structural rather than temporal. `AsyncWork` runs the
collective against a **private staging clone it owns**, and the caller's buffer
is written only by *publication*, which happens at `wait()` or at an
`is_completed()` that observes the work is done -- and nowhere else. There is
no code path on which the caller's buffer changes without the caller asking.
That makes "reading before `wait()` gives the pre-collective contents" a
deterministic claim, provable on every run at every speed, rather than a race
the tests would win by luck.

Every value here is compared against **upstream's own `torch.distributed` over
gloo** at world 3 and 4, on the same inputs, at a tolerance derived from
upstream's own float32-vs-float64 error on the same collective
(`docs/AGREE.md` §2's method), and exactly for integer dtypes. The comparison
helpers, the ephemeral-port spawn and the child-killing teardown are imported
from `test_collect2` rather than copied: two copies of a harness are two
chances for the two sides to stop being given the same thing.
"""

import functools

from test_collect2 import (
    C2_WORLDS,
    _c2_spawn,
    c2_assert_matches,
    c2_ok,
)


# ---------------------------------------------------------------------------
# The probe source, shared verbatim by the shim and the gloo oracle.
# ---------------------------------------------------------------------------

_AW_PROBE_SRC = r'''
AW_SEED = 20260907

def aw_values(rank, n, tag):
    """`n` reproducible floats in [-2, 2), quantised to a multiple of 2**-12.

    Same LCG as `test_collect2`, and for the same reason: `torch.manual_seed`
    would be two different generators on the two sides, so "same seed" would
    not mean "same numbers". Quantised so every value is exact in float32 and
    float64 alike, which keeps the float64 oracle measuring the collective's
    arithmetic rather than a rounding that happened before it started.
    """
    state = (AW_SEED + rank * 7919 + sum(ord(c) * 31 for c in tag)) % 2147483647
    out = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out.append(((state >> 8) % 16384 - 8192) / 4096.0)
    return out


def aw_run(torch, dist, rank, world, DT, shim):
    out = {}

    def ft(values):
        return torch.tensor(values, dtype=DT)

    def probe(key, fn):
        try:
            out[key] = {"ok": fn()}
        except BaseException as exc:
            out[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}

    # -- 1. is_completed() is False before wait() ---------------------------
    #
    # Made a question with one answer rather than a race. Every rank but 0
    # sleeps before entering, so when rank 0 polls there is provably a peer
    # that has not sent a byte and the collective cannot have finished. The
    # half second is orders of magnitude above the skew the rendezvous leaves
    # (the ranks leave `init_process_group` within milliseconds) and far below
    # the harness timeout.
    #
    # The later ranks are recorded and not asserted on: the last rank into a
    # collective may legitimately find it done, on any backend, and asserting
    # otherwise would be asserting a scheduling accident.
    def incomplete_before_wait():
        t = ft(aw_values(rank, 4, "incomplete"))
        source = t.tolist()
        if rank != 0:
            time.sleep(0.5)
        started = time.time()
        work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        issued = time.time() - started
        polled = bool(work.is_completed())
        buffer_at_poll = t.tolist()
        work.wait()
        return {"completed_before_wait": polled,
                "issue_seconds": issued,
                "source": source,
                "buffer_at_poll": buffer_at_poll,
                "after_wait": t.tolist()}
    probe("incomplete_before_wait", incomplete_before_wait)

    # -- 2. wait() is required, and the proof is a value ---------------------
    #
    # Nothing here asserts in a comment that the buffer is invalid early. It
    # reports what was actually in it, and the test outside compares that to
    # the rank's own untouched input and to the answer. On this backend the
    # early read is the input, deterministically. On gloo it is whatever gloo
    # left there, which is upstream's undefined space -- so the test outside
    # *records* upstream's and asserts only on ours, and says so.
    def wait_is_required():
        t = ft(aw_values(rank, 4, "required"))
        source = t.tolist()
        work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        early = t.tolist()
        # A second early read, after enough wall time that any plausible
        # worker thread has finished. If the buffer were published by the
        # worker rather than by the caller, this is where it would show -- the
        # first read might win a race, this one cannot.
        time.sleep(1.0)
        early_late = t.tolist()
        work.wait()
        return {"source": source, "early": early, "early_after_a_second":
                early_late, "after_wait": t.tolist()}
    probe("wait_is_required", wait_is_required)

    # -- 3. two overlapping collectives ------------------------------------
    #
    # Issued back to back without waiting on the first, then waited on in the
    # *reverse* order. Two things could go wrong and both are checked from the
    # outside: the two could corrupt each other (they share one socket per
    # peer, and two frames interleaved on it would make each read the other's
    # bytes), and the second could be published into the first's buffer.
    #
    # They are deliberately different collectives with different shapes, so a
    # crossed publication cannot look like a correct one.
    def overlapping():
        a = ft(aw_values(rank, 4, "overlap-a"))
        b = ft(aw_values(rank, 2, "overlap-b"))
        source_a, source_b = a.tolist(), b.tolist()
        work_a = dist.all_reduce(a, op=dist.ReduceOp.SUM, async_op=True)
        work_b = dist.all_reduce(b, op=dist.ReduceOp.PRODUCT, async_op=True)
        both_before = (a.tolist(), b.tolist())
        # Reverse order on purpose: the issue order is what the wire must
        # keep, and the wait order is the caller's business. If publication
        # were tied to wire order this would hand back the wrong tensor.
        work_b.wait()
        b_after_b_wait = b.tolist()
        a_after_b_wait = a.tolist()
        work_a.wait()
        return {"source_a": source_a, "source_b": source_b,
                "both_before": [both_before[0], both_before[1]],
                "b_after_b_wait": b_after_b_wait,
                "a_after_b_wait": a_after_b_wait,
                "a_final": a.tolist(), "b_final": b.tolist()}
    probe("overlapping", overlapping)

    # -- 4. wait() on a completed handle is a no-op -------------------------
    def wait_twice():
        t = ft(aw_values(rank, 4, "twice"))
        work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        first = bool(work.wait())
        after_first = t.tolist()
        second = bool(work.wait())
        third = bool(work.wait())
        after_third = t.tolist()
        completed = bool(work.is_completed())
        # is_completed() is a publication point, so calling it after the
        # buffer is already published must also be a no-op rather than a
        # second copy onto a buffer the caller may since have written.
        t.add_(1.0)
        scribbled = t.tolist()
        work.is_completed()
        work.wait()
        return {"first": first, "second": second, "third": third,
                "after_first": after_first, "after_third": after_third,
                "completed_after": completed,
                "scribbled": scribbled, "after_scribble": t.tolist()}
    probe("wait_twice", wait_twice)

    # -- 5. the same collective synchronously, as the value oracle ----------
    #
    # `async_op=False` must be bit-identical to `async_op=True` after wait().
    # If asynchrony had changed a single number this is where it shows, and it
    # is the one comparison in this file that does not depend on any timing.
    def synchronous():
        t = ft(aw_values(rank, 4, "required"))
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return {"result": t.tolist()}
    probe("synchronous", synchronous)

    # -- 6. integers, where there is no tolerance to spend -------------------
    def integral_async():
        t = torch.tensor([rank * 10 + i for i in range(4)], dtype=torch.int64)
        source = t.tolist()
        work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        early = t.tolist()
        work.wait()
        return {"source": source, "early": early, "after_wait": t.tolist()}
    probe("integral_async", integral_async)

    # -- 7. an async collective the caller never waits on --------------------
    #
    # A handle dropped on the floor must not wedge the next collective, and
    # the next collective must be correct. This is the case where the worker
    # thread still has an exchange in flight when a *synchronous* call arrives
    # on the calling thread: two collectives would then be on the same socket
    # at once, and the two threads would race to read each other's reply.
    #
    # The stagger and the repetition are both load-bearing, and the first
    # version of this probe had neither -- it issued the two back to back with
    # every rank in lockstep, the async one always finished first, and
    # deleting the guard that prevents the overlap changed nothing it could
    # see. That escape is recorded in `docs/ASYNCWORK.md` §8.
    #
    # Now **rank 0 -- the hub -- arrives late**, so every leaf has an async
    # exchange genuinely stuck waiting for it at the moment the leaf's calling
    # thread enters the synchronous collective. That is the hazard, held open
    # for a third of a second, four times over. The two collectives use
    # different inputs, so a reply delivered to the wrong thread is a wrong
    # *value* and not merely a wrong order.
    def abandoned_then_synchronous():
        results = []
        for _ in range(4):
            t = ft(aw_values(rank, 4, "abandoned"))
            if rank == 0:
                time.sleep(0.3)
            dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)  # dropped
            u = ft(aw_values(rank, 4, "required"))
            dist.all_reduce(u, op=dist.ReduceOp.SUM)
            results.append(u.tolist())
        return {"results": results, "result": results[-1]}
    probe("abandoned_then_synchronous", abandoned_then_synchronous)

    # -- 8. async all_gather, so this is not a claim about allreduce only ----
    def async_all_gather():
        holders = [ft([0.0] * 4) for _ in range(world)]
        marker = [holders[i].tolist() for i in range(world)]
        t = ft(aw_values(rank, 4, "gatherer"))
        work = dist.all_gather(holders, t, async_op=True)
        early = [h.tolist() for h in holders]
        work.wait()
        return {"marker": marker, "early": early,
                "after_wait": [h.tolist() for h in holders]}
    probe("async_all_gather", async_all_gather)

    return out
'''


_AW_SHIM_SRC = (
    "import json, sys, time\n"
    "import torch\n"
    'assert hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded upstream torch, not the shim -- every value '
    'below would be upstream measuring itself")\n'
    "import torch.distributed as dist\n"
    "import torchnative.distributed  # noqa: F401 -- registers backend='local'\n"
    + _AW_PROBE_SRC
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
DT = getattr(torch, sys.argv[5])
dist.init_process_group(backend="local",
                        init_method="tcp://127.0.0.1:%d" % port,
                        rank=rank, world_size=world)
out = aw_run(torch, dist, rank, world, DT, True)
out["_meta"] = {"shim": True, "rank": dist.get_rank(),
                "world": dist.get_world_size(), "backend": "local"}
with open(dest, "w") as handle:
    json.dump(out, handle)
'''
)


_AW_GLOO_SRC = (
    "import json, os, sys, time\n"
    "import torch\n"
    'assert not hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded the shim, not upstream torch -- the oracle '
    'would be the thing under test")\n'
    "import torch.distributed as dist\n"
    + _AW_PROBE_SRC
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
DT = getattr(torch, sys.argv[5])
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(port)
dist.init_process_group(backend="gloo", rank=rank, world_size=world)
out = aw_run(torch, dist, rank, world, DT, False)
out["_meta"] = {"shim": False, "rank": dist.get_rank(),
                "world": dist.get_world_size(), "backend": "gloo"}
with open(dest, "w") as handle:
    json.dump(out, handle)
dist.destroy_process_group()
'''
)


@functools.cache
def aw_shim(world):
    """The shim's answers at `world` ranks, float32."""
    return _c2_spawn(_AW_SHIM_SRC, world, "float32", True, "aw-shim%d" % world)


@functools.cache
def aw_gloo32(world):
    """Upstream gloo's answers at `world` ranks, float32. The oracle."""
    return _c2_spawn(_AW_GLOO_SRC, world, "float32", False,
                     "aw-gloo32-%d" % world)


@functools.cache
def aw_gloo64(world):
    """Upstream gloo in float64 -- where the tolerance comes from, not decoration."""
    return _c2_spawn(_AW_GLOO_SRC, world, "float64", False,
                     "aw-gloo64-%d" % world)


# ---------------------------------------------------------------------------
# The async contract
# ---------------------------------------------------------------------------

def test_is_completed_is_false_before_wait_for_a_collective_that_has_not_finished():
    """The first half of `docs/COLLECT2.md` §7's divergence, now closed.

    This asserts on **rank 0 only**, and the restriction is the thing that
    makes it an assertion rather than a coin toss. The probe holds every other
    rank back half a second, so at the moment rank 0 polls there is a peer that
    has not sent a byte; "not finished" is then a fact about the collective and
    not about the scheduler. For a rank that arrives last the honest answer is
    that completion is unspecified, and this test does not pretend otherwise --
    §6 of `docs/ASYNCWORK.md`.

    Both sides are asserted. Upstream gloo is False here too, and if it ever
    stopped being, the claim that this backend now matches upstream's contract
    would be comparing against nothing.

    Nullifying the worker thread -- running the body inline in the calling
    thread as it was before this round -- turns this red at every world size,
    because the call itself would then block for the half second and return a
    finished handle. Confirmed, `docs/ASYNCWORK.md` §8.
    """
    for world in C2_WORLDS:
        ours = c2_ok(aw_shim(world)[0], "incomplete_before_wait")
        theirs = c2_ok(aw_gloo32(world)[0], "incomplete_before_wait")

        assert theirs["completed_before_wait"] is False, (
            "upstream gloo reported a completed work at world %d rank 0 while "
            "two peers were still asleep. The oracle for this contract is "
            "gone, so passing it would mean nothing" % world)
        assert ours["completed_before_wait"] is False, (
            "this backend reported a completed work at world %d rank 0 while "
            "%d peers were still asleep -- the collective ran inside the call, "
            "so async_op is a lie again" % (world, world - 1))

        # And it returned *promptly*, which is the difference between a handle
        # that is honest about being incomplete and one that blocked anyway.
        # The peers sleep 0.5s; issuing must cost a small fraction of that.
        assert ours["issue_seconds"] < 0.25, (
            "world %d rank 0 spent %.3fs inside an async_op=True call whose "
            "peers were asleep for 0.5s. It waited for them, so the handle's "
            "incompleteness is bookkeeping rather than asynchrony"
            % (world, ours["issue_seconds"]))


def test_wait_is_required_because_the_buffer_before_it_holds_the_input():
    """`wait()` is not advisory, and the proof is a measured value.

    This is the ownership half, and it is the assertion in this file with the
    sharpest teeth, because it does not depend on timing at all. `AsyncWork`
    runs the collective against a staging clone; the caller's tensor is written
    only by publication, and publication only happens when the caller asks. So
    an early read returns the rank's own pre-collective input **on every run**,
    and it still does a full second later, when any worker thread has long
    finished. A build that published from the worker would pass the first read
    on a lucky day and fail the second one always.

    The two reads are both here on purpose. The first alone would be a race
    that this design happens to win; the second alone would not catch a
    publication that happened at some later synchronisation point. Together
    they say the buffer is the caller's until the caller asks.

    Upstream is **recorded and not asserted on**, and that is the one place
    this file declines to make a claim. Gloo's output buffer before `wait()` is
    upstream's undefined space, and `docs/COLLECT2.md` §8 is the reason to be
    careful here rather than casual: "undefined" is not "unobserved", and a
    test that asserted gloo leaves the input there would be inventing a
    contract upstream does not offer. What is asserted about upstream is only
    that it eventually agrees, which it promises.
    """
    for world in C2_WORLDS:
        for rank in range(world):
            ours = c2_ok(aw_shim(world)[rank], "wait_is_required")

            assert ours["early"] == ours["source"], (
                "world %d rank %d: reading an un-waited buffer gave %r, which "
                "is neither the input nor an error. The handle does not own "
                "its output" % (world, rank, ours["early"]))
            assert ours["early_after_a_second"] == ours["source"], (
                "world %d rank %d: the buffer was the input immediately after "
                "the call and %r a second later. Something published it "
                "without being asked, and the immediate read only passed "
                "because it won a race"
                % (world, rank, ours["early_after_a_second"]))
            assert ours["after_wait"] != ours["source"], (
                "world %d rank %d: wait() did not change the buffer, so every "
                "assertion above would also hold for a backend that does "
                "nothing at all" % (world, rank))

            # The value it finally publishes is upstream's.
            theirs = c2_ok(aw_gloo32(world)[rank], "wait_is_required")
            c2_assert_matches(
                "async.wait_is_required", world, rank,
                ours["after_wait"], theirs["after_wait"],
                c2_ok(aw_gloo64(world)[rank], "wait_is_required")["after_wait"])


def test_two_overlapping_collectives_do_not_corrupt_each_other():
    """Two in flight at once, waited on in the reverse order.

    Three distinct failures are reachable here and each has an assertion:

    * **Crossed publication.** The two handles could publish into each other's
      buffers. They are given different lengths and different reduce ops so a
      crossed result cannot coincidentally look right -- a 4-element sum
      landing in a 2-element product buffer is not a near miss.
    * **Interleaved frames.** The star is one socket per peer carrying one
      length-prefixed frame per collective. Two exchanges running at once on it
      would each read the other's bytes; the values would be garbage or the
      ranks would hang. The single worker thread is what prevents this, and
      this test is what would notice it being removed.
    * **Publication tied to wire order.** `work_b` is waited on **first**,
      although `work_a` was issued first and must go on the wire first.
      Waiting on B must publish B and leave A alone; asserting that A is still
      its own input at that moment is what says publication follows the caller
      and not the queue.
    """
    for world in C2_WORLDS:
        for rank in range(world):
            ours = c2_ok(aw_shim(world)[rank], "overlapping")
            theirs = c2_ok(aw_gloo32(world)[rank], "overlapping")
            up64 = c2_ok(aw_gloo64(world)[rank], "overlapping")

            assert ours["both_before"] == [ours["source_a"], ours["source_b"]], (
                "world %d rank %d: with two collectives in flight, one of the "
                "buffers had already changed" % (world, rank))

            assert ours["b_after_b_wait"] != ours["source_b"], (
                "world %d rank %d: waiting on B did not publish B" % (world, rank))
            assert ours["a_after_b_wait"] == ours["source_a"], (
                "world %d rank %d: waiting on B published A as well. "
                "Publication is following the queue rather than the caller, "
                "and a caller who waits on one handle is being handed another "
                "handle's timing" % (world, rank))

            assert len(ours["a_final"]) == 4 and len(ours["b_final"]) == 2, ours

            c2_assert_matches("overlapping.a", world, rank, ours["a_final"],
                              theirs["a_final"], up64["a_final"])
            c2_assert_matches("overlapping.b", world, rank, ours["b_final"],
                              theirs["b_final"], up64["b_final"])


def test_wait_on_an_already_completed_handle_is_a_no_op_and_not_an_error():
    """Three `wait()`s, and the buffer is written exactly once.

    "No-op" is checked as two separate things, because only one of them is
    obvious. The easy one is that the second and third calls return True rather
    than raising. The one with teeth is that they do not *write* -- the probe
    scribbles on the buffer after the handle is complete and then calls
    `is_completed()` and `wait()` again, and the scribble must survive. A
    publication that ran a second time would silently undo whatever the caller
    had done with their own tensor since, which is a data-loss bug that no
    return value would reveal.

    Both sides are asserted for the return values, since repeated `wait()` is
    something upstream does promise.
    """
    for world in C2_WORLDS:
        for rank in range(world):
            ours = c2_ok(aw_shim(world)[rank], "wait_twice")
            theirs = c2_ok(aw_gloo32(world)[rank], "wait_twice")

            for side, report in (("shim", ours), ("gloo", theirs)):
                assert report["first"] is True, (side, world, rank, report)
                assert report["second"] is True, (
                    "%s: the second wait() on a completed handle at world %d "
                    "rank %d did not return True" % (side, world, rank))
                assert report["third"] is True, (side, world, rank, report)
                assert report["after_third"] == report["after_first"], (
                    "%s: waiting again changed the answer at world %d rank %d"
                    % (side, world, rank))

            assert ours["completed_after"] is True, (
                "world %d rank %d: the handle is not complete after wait() "
                "returned" % (world, rank))
            assert ours["after_scribble"] == ours["scribbled"], (
                "world %d rank %d: is_completed()/wait() on an already "
                "published handle wrote the buffer a second time and threw "
                "away what the caller had put there" % (world, rank))


def test_the_async_answer_is_the_synchronous_answer_and_both_are_upstreams():
    """Asynchrony changed no numbers -- the check that does not involve timing.

    Every other test here depends on when something happened. This one does
    not: the same collective on the same inputs, once with `async_op=True` and
    once without, must be **bit-identical**, and both must be upstream's answer
    at the derived tolerance. If the staging clone lost a dtype, or publication
    copied the wrong tensor, or the worker thread folded in arrival order
    rather than rank order, this is where it lands.

    Bit equality between the two paths rather than a tolerance, because there
    is no arithmetic difference between them to spend one on: it is the same
    fold, run in a different thread.
    """
    for world in C2_WORLDS:
        for rank in range(world):
            shim = aw_shim(world)[rank]
            asynchronous = c2_ok(shim, "wait_is_required")["after_wait"]
            synchronous = c2_ok(shim, "synchronous")["result"]
            assert asynchronous == synchronous, (
                "world %d rank %d: async_op=True gave %r and async_op=False "
                "gave %r for the same collective on the same inputs"
                % (world, rank, asynchronous, synchronous))

            c2_assert_matches(
                "synchronous", world, rank, synchronous,
                c2_ok(aw_gloo32(world)[rank], "synchronous")["result"],
                c2_ok(aw_gloo64(world)[rank], "synchronous")["result"])


def test_integer_collectives_are_exact_through_the_async_path():
    """No tolerance, and the early read is the input here too.

    Integer reductions are exact or they are wrong. Running them through the
    staging clone is where a dtype could be lost -- a clone that came back
    float32 would round `int64` sums above 2**24 and a float tolerance would
    hide it, so this is held to equality against upstream and against the
    arithmetic the test can do itself.
    """
    for world in C2_WORLDS:
        expected = [sum(r * 10 + i for r in range(world)) for i in range(4)]
        for rank in range(world):
            ours = c2_ok(aw_shim(world)[rank], "integral_async")
            theirs = c2_ok(aw_gloo32(world)[rank], "integral_async")

            assert ours["early"] == ours["source"], (
                "world %d rank %d: the integer buffer was published before "
                "wait()" % (world, rank))
            assert ours["after_wait"] == theirs["after_wait"], (
                "world %d rank %d: %r against upstream gloo's %r; an integer "
                "reduction has no tolerance to spend"
                % (world, rank, ours["after_wait"], theirs["after_wait"]))
            assert ours["after_wait"] == expected, (
                "world %d rank %d: %r, but the sum of the inputs is %r. This "
                "is the check that does not go through upstream at all"
                % (world, rank, ours["after_wait"], expected))


def test_an_abandoned_handle_does_not_wedge_or_corrupt_the_next_collective():
    """A caller who drops a handle, and then does something synchronous.

    Two things are being held, and reaching the assertions at all is the first
    of them: the dropped collective must still complete, or the peers that are
    in it block forever and the harness times out.

    The second is the one with teeth. The star carries **one** collective at a
    time -- one socket per peer, one length-prefixed frame per collective -- so
    a synchronous call issued from the calling thread while the worker still
    has an exchange in flight would put two on the same socket, and the two
    threads would race to read each other's reply. `_drain_async` is what
    prevents it.

    **The first version of this test could not see that guard removed**, and
    that is worth stating rather than quietly fixing: the probe issued the two
    collectives back to back with every rank in lockstep, so the async one had
    always finished before the synchronous one began and the hazard window was
    never open. Deleting `_drain_async` left the whole suite green. The probe
    now makes rank 0 -- the hub -- arrive a third of a second late, four times
    over, which holds every leaf's async exchange open across its own
    synchronous call. `docs/ASYNCWORK.md` §8 records the escape and the fix.

    Every one of the four iterations is compared, not just the last, and
    against upstream's answer for those same inputs -- so a reply delivered to
    the wrong thread shows up as a wrong number rather than as a wrong order.
    """
    for world in C2_WORLDS:
        for rank in range(world):
            ours = c2_ok(aw_shim(world)[rank], "abandoned_then_synchronous")
            up32 = c2_ok(aw_gloo32(world)[rank], "synchronous")["result"]
            up64 = c2_ok(aw_gloo64(world)[rank], "synchronous")["result"]
            assert len(ours["results"]) == 4, ours
            for index, got in enumerate(ours["results"]):
                c2_assert_matches(
                    "abandoned_then_synchronous[%d]" % index,
                    world, rank, got, up32, up64)


def test_asynchrony_is_not_a_property_of_allreduce_alone():
    """`all_gather` through the same path, including its nested output list.

    `allreduce`'s output is a flat list of tensors; `all_gather`'s is a list of
    lists, and the staging clone has to walk that. A clone that shallow-copied
    the outer list would hand the worker the caller's own tensors and publish
    early -- which the early-read assertion below catches, and which no
    `allreduce` test could ever reach.
    """
    for world in C2_WORLDS:
        for rank in range(world):
            ours = c2_ok(aw_shim(world)[rank], "async_all_gather")
            theirs = c2_ok(aw_gloo32(world)[rank], "async_all_gather")

            assert ours["early"] == ours["marker"], (
                "world %d rank %d: the gather's output list was written "
                "before wait(); the staging clone did not reach the inner "
                "tensors" % (world, rank))
            assert ours["after_wait"] == theirs["after_wait"], (
                "world %d rank %d: %r against gloo's %r"
                % (world, rank, ours["after_wait"], theirs["after_wait"]))
            assert ours["after_wait"] != ours["marker"], (world, rank)


def test_every_rank_really_ran_the_shim_and_the_oracle_really_ran_upstream():
    """The premise of every number above, checked from outside the workers.

    The workers assert it on the way in too, but a worker that skipped its own
    assert would still be caught here. A previous round of this repository lost
    hours to probes that silently imported upstream torch and then reported
    that the shim agreed with it perfectly.
    """
    for world in C2_WORLDS:
        for rank in range(world):
            meta = aw_shim(world)[rank]["_meta"]
            assert meta == {"shim": True, "rank": rank, "world": world,
                            "backend": "local"}, meta
            for oracle in (aw_gloo32(world), aw_gloo64(world)):
                other = oracle[rank]["_meta"]
                assert other["shim"] is False, other
                assert (other["rank"], other["world"]) == (rank, world), other
                assert other["backend"] == "gloo", other


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
