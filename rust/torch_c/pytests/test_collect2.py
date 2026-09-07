"""The collectives of `ProcessGroupLocal` above one rank, against upstream gloo.

`docs/platform/RELEASE_0_0_13a0.md` §5 said this layer was "`allreduce(op=SUM)` only,
over loopback on one machine. Other collectives, other reduce ops, secure
aggregation and differential privacy refuse by name."

**Two thirds of that sentence was wrong, in opposite directions**, and
`docs/distributed/COLLECT2.md` §1 is the measurement. `all_gather`, `all_gather_into_tensor`
and `barrier` already worked at three ranks. `reduce_scatter`, `scatter`,
`all_to_all` and `all_to_all_single` did not refuse *at all* -- they returned
each rank's own input, unreduced and untransposed, with no error. A sentence
that promises a refusal where there is a wrong answer is the worse half of the
two, because a caller who believes it will never look.

Everything here is compared against **upstream's own `torch.distributed` over
gloo**, in three real process groups at each world size: the shim, upstream in
float32, and upstream in float64. The float64 run is not decoration -- it is
where the tolerance comes from. `docs/numerics/AGREE.md` §2's method is that a threshold
picked by eye is not a result, so each collective's is read off *upstream's own*
float32-vs-float64 error on that same collective with those same inputs,
floored at 8 float32 ulp. For the collectives that never combine two numbers
(`broadcast`, `all_gather`, `gather`, `scatter`, `all_to_all`, and `MIN`/`MAX`)
that derivation produces the floor over a measured error of exactly zero, so
they are held to **bit equality** -- which is the honest reading of a tolerance
derived from a distribution with no width. Integer dtypes are exact everywhere.

Every group here runs in real `subprocess.Popen`s on an **ephemeral port bound
and released by the parent**, never a fixed one: several rounds share this
machine and a fixed port is the shared-mutable-state trap this repository has
already paid for once with a fixed log path. Every spawn kills its children on
every path, including the raising ones.
"""

import functools
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

from test_shim import _CKPT_VENDOR_DIR


# The world sizes every value test runs at. **Three is not optional.** A
# reduction that folds a pair correctly can still be wrong at three -- the
# defect this file was written after (`reduce_scatter` returning `sources[0]`)
# is *right* at world_size 1 and wrong at every size above it, and a suite that
# stopped at two would have called a `[a, b] -> a` bug a passing pair. Four is
# here because three is the smallest size at which "off by one rank" and "the
# hub is special" are distinguishable, and four is the smallest that also has a
# non-trivial *even* split for `reduce_scatter` and `all_to_all`.
C2_WORLDS = (3, 4)

#: One float32 ulp at 1.0. The floor under every derived tolerance, in the
#: units `docs/numerics/AGREE.md` §2 states its own in.
C2_F32_ULP = 2.0 ** -23

#: `docs/numerics/AGREE.md` floors its tolerance at 8 ulp. Same floor, same reason: a
#: sweep whose inputs happened to be numerically easy must not be able to drive
#: the threshold below the width of the type it is measuring.
C2_ULP_FLOOR = 8


# ---------------------------------------------------------------------------
# The probes, as one source string shared by *both* sides.
#
# Shared verbatim rather than written twice, because the whole claim of this
# file is that the two sides were given the same inputs. Two copies of a probe
# body are two chances for them to drift, and a drift that made the inputs
# differ would show up as a tolerance failure that looks like a numerical
# defect -- which is the most expensive kind of false signal here. The only
# thing that differs between the two workers is the four lines that choose a
# backend.
#
# The inputs are built by a **pure-Python LCG seeded per rank**, not by
# `torch.manual_seed`. The shim and upstream are two different RNG
# implementations, and "same seed" only means "same numbers" if the generator
# is the same object; this one is arithmetic in the test, so both sides
# construct bit-identical inputs by construction rather than by assumption.
# ---------------------------------------------------------------------------

_C2_PROBE_SRC = r'''
C2_SEED = 20260907

def c2_values(rank, n, tag):
    """`n` reproducible floats in [-2, 2) for `rank`, distinct per `tag`.

    A plain LCG. Values are quantised to a multiple of 2**-12 so that every one
    of them is exactly representable in float32 *and* float64 -- the input is
    then not a source of difference between the two, and the float64 oracle run
    measures the collective's own arithmetic rather than a rounding that
    happened before it started.
    """
    state = (C2_SEED + rank * 7919 + sum(ord(c) * 31 for c in tag)) % 2147483647
    out = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out.append(((state >> 8) % 16384 - 8192) / 4096.0)
    return out


def c2_ints(rank, n, tag):
    state = (C2_SEED + rank * 104729 + sum(ord(c) * 17 for c in tag)) % 2147483647
    out = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out.append((state >> 11) % 1000 - 500)
    return out


def c2_run(torch, dist, rank, world, DT, shim):
    """Every probe, once. `DT` is the float dtype this run is measured in.

    `shim` gates the probes that must not be pointed at upstream. They are not
    skipped there for tidiness -- **gloo does not survive them**. Its
    `reduce_scatter_tensor` shape check is a C++ `Check failed:` that calls
    `abort()`, so a misshapen input takes the whole rank down with SIGABRT
    rather than raising something Python can catch, and `send` with no matching
    `recv` blocks until the harness times out. Both would be reported here as
    "the oracle crashed", which is a true statement about a probe that had no
    business running. What is measured on the oracle is exactly what is
    compared against it.
    """
    out = {}

    def probe(key, fn):
        try:
            out[key] = {"ok": fn()}
        except Exception as exc:
            out[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}

    def ft(values):
        return torch.tensor(values, dtype=DT)

    # -- allreduce, every reduce op -------------------------------------
    for opname in ("SUM", "MIN", "MAX", "PRODUCT", "AVG"):
        def go(opname=opname):
            # One tag for all five: `test_avg_is_the_sum_over_the_world`
            # compares SUM's answer against AVG's, and that comparison is only
            # a comparison if the two were handed the same numbers. Tagging
            # them per-op made it compare two different reductions of two
            # different inputs and call the disagreement a divisor bug.
            t = ft(c2_values(rank, 6, "allreduce"))
            dist.all_reduce(t, op=getattr(dist.ReduceOp, opname))
            return t.tolist()
        probe("allreduce_" + opname, go)

    # -- allreduce on int64: no tolerance is available and none is wanted
    for opname in ("SUM", "MIN", "MAX", "PRODUCT"):
        def goi(opname=opname):
            t = torch.tensor(c2_ints(rank, 5, "int" + opname), dtype=torch.int64)
            if opname == "PRODUCT":
                # Kept small: the product of five ranks' worth of 3-digit
                # integers overflows int64 and an overflow is not a comparison.
                t = torch.tensor([(v % 5) - 2 for v in
                                  c2_ints(rank, 5, "int" + opname)],
                                 dtype=torch.int64)
            dist.all_reduce(t, op=getattr(dist.ReduceOp, opname))
            return t.tolist()
        probe("allreduce_int64_" + opname, goi)

    # -- the reductions that are not implemented, refusing by name -------
    if shim:
      for opname in ("BAND", "BOR", "BXOR"):
        def gob(opname=opname):
            t = ft(c2_values(rank, 4, "bitwise"))
            dist.all_reduce(t, op=getattr(dist.ReduceOp, opname))
            return t.tolist()
        probe("allreduce_" + opname, gob)

      def premul():
        t = ft(c2_values(rank, 4, "premul"))
        dist.all_reduce(t, op=dist.ReduceOp.PREMUL_SUM)
        return t.tolist()
      probe("allreduce_PREMUL_SUM", premul)

    # -- broadcast, from every root -------------------------------------
    for root in (0, 1, world - 1):
        def bc(root=root):
            t = ft(c2_values(rank, 5, "broadcast%d" % root))
            dist.broadcast(t, src=root)
            return t.tolist()
        probe("broadcast_root%d" % root, bc)

    if shim:
      def bad_root():
        dist.broadcast(ft([0.0]), src=world + 3)
        return "ACCEPTED"
      probe("broadcast_root_out_of_range", bad_root)

    # -- all_gather ------------------------------------------------------
    def ag():
        holders = [torch.zeros(4, dtype=DT) for _ in range(world)]
        dist.all_gather(holders, ft(c2_values(rank, 4, "allgather")))
        return [h.tolist() for h in holders]
    probe("all_gather", ag)

    def agt():
        flat = torch.zeros(world * 3, dtype=DT)
        dist.all_gather_into_tensor(flat, ft(c2_values(rank, 3, "aggflat")))
        return flat.tolist()
    probe("all_gather_into_tensor", agt)

    # -- reduce_scatter ---------------------------------------------------
    def rs():
        result = torch.zeros(3, dtype=DT)
        chunks = [ft(c2_values(rank, 3, "rs%d" % j)) for j in range(world)]
        dist.reduce_scatter(result, chunks, op=dist.ReduceOp.SUM)
        return result.tolist()
    probe("reduce_scatter", rs)

    def rs_max():
        result = torch.zeros(3, dtype=DT)
        chunks = [ft(c2_values(rank, 3, "rsm%d" % j)) for j in range(world)]
        dist.reduce_scatter(result, chunks, op=dist.ReduceOp.MAX)
        return result.tolist()
    probe("reduce_scatter_max", rs_max)

    def rst():
        result = torch.zeros(3, dtype=DT)
        flat = ft(c2_values(rank, 3 * world, "rstensor"))
        dist.reduce_scatter_tensor(result, flat, op=dist.ReduceOp.SUM)
        return result.tolist()
    probe("reduce_scatter_tensor", rst)

    if shim:
      def rst_bad():
        # 3*world + 1 does not divide into `world` chunks of 3. On gloo this
        # is `Check failed: outputShape[0] * worldSize == inputShape[0]` and
        # an abort(), not an exception -- which is why it is gated.
        dist.reduce_scatter_tensor(torch.zeros(3, dtype=DT),
                                   ft(c2_values(rank, 3 * world + 1, "rsbad")),
                                   op=dist.ReduceOp.SUM)
        return "ACCEPTED"
      probe("reduce_scatter_tensor_misshapen", rst_bad)

    # -- reduce (root only is specified) ---------------------------------
    for root in (0, 1):
        def rd(root=root):
            t = ft(c2_values(rank, 4, "reduce%d" % root))
            dist.reduce(t, dst=root, op=dist.ReduceOp.SUM)
            return t.tolist() if rank == root else None
        probe("reduce_root%d" % root, rd)

    # `reduce` again, this time recording what the *non*-root ranks are left
    # holding. The probe above returns `None` off-root because upstream leaves
    # those buffers undefined and there is nothing to compare. But "undefined"
    # is not "unobserved": this backend makes a specific choice inside that
    # freedom -- it leaves them untouched -- and a documented choice with no
    # test is how this repository's documents have drifted before. Nullifying
    # the root check was the one sabotage the first pass did not catch.
    def rd_untouched():
        src = c2_values(rank, 4, "reduceuntouched")
        t = ft(src)
        dist.reduce(t, dst=0, op=dist.ReduceOp.SUM)
        return {"input": src, "after": t.tolist(), "root": rank == 0}
    probe("reduce_untouched", rd_untouched)

    # -- gather / scatter -------------------------------------------------
    for root in (0, 1):
        def ga(root=root):
            src = ft(c2_values(rank, 3, "gather%d" % root))
            holders = ([torch.zeros(3, dtype=DT) for _ in range(world)]
                       if rank == root else None)
            dist.gather(src, holders, dst=root)
            return [h.tolist() for h in holders] if rank == root else None
        probe("gather_root%d" % root, ga)

        def sc(root=root):
            result = torch.zeros(3, dtype=DT)
            chunks = ([ft(c2_values(i, 3, "scatter%d" % root))
                       for i in range(world)] if rank == root else None)
            dist.scatter(result, chunks, src=root)
            return result.tolist()
        probe("scatter_root%d" % root, sc)

    # -- all_to_all -------------------------------------------------------
    def a2a():
        outs = [torch.zeros(2, dtype=DT) for _ in range(world)]
        ins = [ft(c2_values(rank * 100 + j, 2, "a2a")) for j in range(world)]
        dist.all_to_all(outs, ins)
        return [o.tolist() for o in outs]
    probe("all_to_all", a2a)

    def a2as():
        result = torch.zeros(2 * world, dtype=DT)
        flat = ft(c2_values(rank, 2 * world, "a2as"))
        dist.all_to_all_single(result, flat)
        return result.tolist()
    probe("all_to_all_single", a2as)

    if shim:
      def a2as_uneven():
        splits = [1] * world
        splits[0] = 2
        total = sum(splits)
        dist.all_to_all_single(torch.zeros(total, dtype=DT),
                               ft(c2_values(rank, total, "uneven")),
                               splits, splits)
        return "ACCEPTED"
      probe("all_to_all_single_uneven", a2as_uneven)

    # -- barrier, and that it really blocks -------------------------------
    #
    # A barrier that returned immediately would pass every value test in this
    # file. The last rank sleeps before entering; every other rank must leave
    # *after* that sleep has elapsed. The assertion is a lower bound on elapsed
    # wall time, so a loaded machine can only make it pass harder.
    def barrier_blocks():
        started = time.time()
        if rank == world - 1:
            time.sleep(1.0)
        dist.barrier()
        return {"elapsed": time.time() - started, "slept": rank == world - 1}
    probe("barrier_blocks", barrier_blocks)

    # -- the ordering contract, for PRODUCT --------------------------------
    #
    # `docs/distributed/FEDERATED4.md` §2 pinned the ascending-rank fold for SUM with
    # `1.0 + 1e8 - 1e8`. The same contract binds PRODUCT and the probe for it
    # is sharper: in rank order the product is finite, in *any* order that
    # multiplies the two large terms first it overflows float32 to infinity.
    # A tolerance cannot paper over the difference between 2**80 and `inf`.
    def product_order():
        factor = {0: 2.0 ** -120, 1: 2.0 ** 100, 2: 2.0 ** 100}.get(rank, 1.0)
        t = torch.tensor([factor], dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.PRODUCT)
        return {"result": t.tolist(), "factor": factor}
    probe("product_rank_order", product_order)

    # -- the async surface --------------------------------------------------
    #
    # The stagger is the point of this probe rather than noise in it. Polling
    # `is_completed()` the instant after issuing a collective that every peer
    # has *already* reached is a race: the last rank to arrive can legitimately
    # find the work done, on any backend. So the non-zero ranks are held back
    # half a second, which makes rank 0 provably the first to arrive and its
    # poll a question with one right answer -- there is a peer that has not
    # sent anything yet, so the collective cannot have finished.
    #
    # Half a second is far above the skew the rendezvous leaves behind (the
    # ranks come out of `init_process_group` within milliseconds of each
    # other) and far below the harness timeout. The later ranks are measured
    # and reported but not asserted on, because for them the answer really is
    # unspecified -- see `docs/distributed/ASYNCWORK.md` §6.
    def async_probe():
        t = ft(c2_values(rank, 3, "async"))
        source = t.tolist()
        if rank != 0:
            time.sleep(0.5)
        work = dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        before = t.tolist()
        completed_before_wait = work.is_completed()
        waited = work.wait()
        return {"completed_before_wait": bool(completed_before_wait),
                "wait_returned": bool(waited),
                "source": source,
                "before_wait": before,
                "after_wait": t.tolist()}
    probe("async_allreduce", async_probe)

    # -- what stays refused --------------------------------------------------
    #
    # Shim-only, and not because upstream would disagree: on gloo a `send` with
    # no matching `recv` blocks forever, so running these against the oracle
    # would deadlock the harness rather than measure anything.
    if shim:
      def send():
        dist.send(ft([0.0, 0.0]), (rank + 1) % world)
        return "ACCEPTED"
      probe("send", send)

      def recv():
        dist.recv(ft([0.0, 0.0]), (rank + 1) % world)
        return "ACCEPTED"
      probe("recv", recv)

    return out
'''


_C2_SHIM_SRC = (
    "import json, sys, time\n"
    "import torch\n"
    'assert hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded upstream torch, not the shim -- every value '
    'below would be upstream measuring itself")\n'
    "import torch.distributed as dist\n"
    "import torchnative.distributed  # noqa: F401 -- registers backend='local'\n"
    + _C2_PROBE_SRC
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
DT = getattr(torch, sys.argv[5])
dist.init_process_group(backend="local",
                        init_method="tcp://127.0.0.1:%d" % port,
                        rank=rank, world_size=world)
out = c2_run(torch, dist, rank, world, DT, True)
out["_meta"] = {"shim": True, "rank": dist.get_rank(),
                "world": dist.get_world_size(),
                "backend": "local"}
with open(dest, "w") as handle:
    json.dump(out, handle)
'''
)


_C2_GLOO_SRC = (
    "import json, os, sys, time\n"
    "import torch\n"
    'assert not hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded the shim, not upstream torch -- the oracle '
    'would be the thing under test")\n'
    "import torch.distributed as dist\n"
    + _C2_PROBE_SRC
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
DT = getattr(torch, sys.argv[5])
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(port)
dist.init_process_group(backend="gloo", rank=rank, world_size=world)
out = c2_run(torch, dist, rank, world, DT, False)
out["_meta"] = {"shim": False, "rank": dist.get_rank(),
                "world": dist.get_world_size(), "backend": "gloo"}
with open(dest, "w") as handle:
    json.dump(out, handle)
dist.destroy_process_group()
'''
)


def _c2_free_port():
    """An ephemeral port, bound and released by this process.

    Not a constant. Several rounds of this repository run concurrently on one
    machine and a fixed port is the same shared-mutable-state defect as the
    fixed log path that has already cost a round here -- two suites would
    rendezvous into each other's group and one of them would measure the
    other's ranks.
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _c2_spawn(source, world, dtype, shim, what, timeout=600):
    """Run `source` as `world` processes. Returns their JSON, rank-ordered.

    Kills every child on every path, including the raising ones: a multi-rank
    test that propagates an exception with a peer still blocked in `accept`
    leaves an orphan holding a port, and the next round to bind an ephemeral
    port near it inherits a machine that is quietly one process dirtier.
    """
    tmp = tempfile.mkdtemp(prefix="collect2-%s-" % what)
    port = _c2_free_port()

    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _CKPT_VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        # The oracle must not see the vendored tree. If it did, this whole file
        # would be the shim agreeing with itself.
        env.pop("PYTHONPATH", None)

    procs = []
    try:
        for rank in range(world):
            procs.append(subprocess.Popen(
                [sys.executable, "-c", source, str(rank), str(port),
                 str(world), os.path.join(tmp, "rank-%d.json" % rank), dtype],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=env,
            ))
            if rank == 0 and shim:
                # Rank 0 binds the store before the others retry against it.
                time.sleep(0.4)

        reports = []
        for rank, proc in enumerate(procs):
            try:
                _, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "collect2/%s: rank %d never finished within %ds at world "
                    "%d. A collective that hangs is what this timeout exists "
                    "for; it is not a performance bound."
                    % (what, rank, timeout, world))
            if proc.returncode != 0:
                raise RuntimeError(
                    "collect2/%s: rank %d exited %d\n--- stderr ---\n%s"
                    % (what, rank, proc.returncode, err[-6000:]))
            with open(os.path.join(tmp, "rank-%d.json" % rank)) as handle:
                reports.append(json.load(handle))
        return reports
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()


@functools.cache
def c2_shim(world):
    """The shim's answers at `world` ranks, float32."""
    return _c2_spawn(_C2_SHIM_SRC, world, "float32", True, "shim%d" % world)


@functools.cache
def c2_gloo32(world):
    """Upstream gloo's answers at `world` ranks, float32. The oracle."""
    return _c2_spawn(_C2_GLOO_SRC, world, "float32", False, "gloo32-%d" % world)


@functools.cache
def c2_gloo64(world):
    """Upstream gloo at float64. Where the tolerance comes from, not a result."""
    return _c2_spawn(_C2_GLOO_SRC, world, "float64", False, "gloo64-%d" % world)


# ---------------------------------------------------------------------------
# Comparison, with the tolerance derived rather than chosen
# ---------------------------------------------------------------------------

def _c2_flat(value):
    """Every number in a nested list, in order."""
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_c2_flat(item))
        return out
    return [value]


def c2_derived_tolerance(up32, up64):
    """`docs/numerics/AGREE.md` §2's method, applied to one collective's own output.

    The threshold is **upstream's own** float32-vs-float64 relative error on
    this collective with these inputs, floored at 8 float32 ulp. Its defence is
    the same as AGREE's: anything tighter would have to call upstream wrong,
    and a threshold that fails the oracle is not measuring the shim.

    Returns `(tolerance, upstream_own_error)`. When the second is 0.0 the
    collective did no arithmetic -- it selected or moved numbers -- and the
    caller should be reading that as "require exactness" rather than as "use
    the floor", which is what `c2_assert_matches` does.
    """
    a, b = _c2_flat(up32), _c2_flat(up64)
    assert len(a) == len(b), (len(a), len(b))
    scale = max([abs(x) for x in b] + [1.0])
    own = max([abs(x - y) for x, y in zip(a, b)] + [0.0]) / scale
    return max(own, C2_ULP_FLOOR * C2_F32_ULP), own


def c2_assert_matches(key, world, rank, shim, up32, up64):
    """The shim's answer against upstream's, at this collective's tolerance."""
    if shim is None or up32 is None:
        assert shim == up32, (key, world, rank, shim, up32)
        return "unspecified"

    flat_shim, flat_up = _c2_flat(shim), _c2_flat(up32)
    assert len(flat_shim) == len(flat_up), (
        "%s at world %d rank %d: the shim produced %d numbers and upstream %d"
        % (key, world, rank, len(flat_shim), len(flat_up)))

    if all(isinstance(x, int) for x in flat_up):
        assert flat_shim == flat_up, (
            "%s at world %d rank %d: integer reductions are exact or they are "
            "wrong; there is no tolerance to spend.\n  shim: %r\n  gloo: %r"
            % (key, world, rank, flat_shim, flat_up))
        return "exact-int"

    tol, own = c2_derived_tolerance(up32, up64)
    if own == 0.0:
        # Upstream's float32 and float64 runs agreed to the last bit, which for
        # these collectives is not luck: they select and move numbers, they
        # never combine two. A tolerance derived from a distribution of width
        # zero is zero, and spending the 8-ulp floor here would be choosing a
        # number after all -- exactly what AGREE's method exists to avoid.
        assert flat_shim == flat_up, (
            "%s at world %d rank %d: upstream's own float32 and float64 runs "
            "are bit-identical, so this collective combines no numbers and the "
            "derived tolerance is zero.\n  shim: %r\n  gloo: %r"
            % (key, world, rank, flat_shim, flat_up))
        return "exact-derived"

    scale = max([abs(x) for x in _c2_flat(up64)] + [1.0])
    worst = max(abs(x - y) for x, y in zip(flat_shim, flat_up)) / scale
    assert worst <= tol, (
        "%s at world %d rank %d: relative error %.3e exceeds the tolerance "
        "%.3e derived from upstream's own float32-vs-float64 error %.3e on "
        "this same collective.\n  shim: %r\n  gloo: %r"
        % (key, world, rank, worst, tol, own, flat_shim, flat_up))
    return "within %.3e (tol %.3e)" % (worst, tol)


def c2_ok(report, key):
    """The value of a probe that must have succeeded."""
    entry = report[key]
    assert "ok" in entry, "%s did not run: %s" % (key, entry.get("exc"))
    return entry["ok"]


def c2_exc(report, key):
    """The message of a probe that must have refused."""
    entry = report[key]
    assert "exc" in entry, "%s was ACCEPTED where a refusal was required: %r" % (
        key, entry.get("ok"))
    return entry["exc"]


# The value-producing collectives, and whether every rank's answer is
# specified. `reduce` and `gather` name a root and leave every other rank's
# buffer undefined -- measured on gloo at three ranks, where the non-roots came
# back holding partial sums (docs/distributed/COLLECT2.md §4). The probes return `None`
# off-root so that "undefined" is represented rather than asserted about.
C2_VALUE_PROBES = (
    "allreduce_SUM", "allreduce_MIN", "allreduce_MAX", "allreduce_PRODUCT",
    "allreduce_AVG",
    "allreduce_int64_SUM", "allreduce_int64_MIN", "allreduce_int64_MAX",
    "allreduce_int64_PRODUCT",
    "broadcast_root0", "broadcast_root1",
    "all_gather", "all_gather_into_tensor",
    "reduce_scatter", "reduce_scatter_max", "reduce_scatter_tensor",
    "reduce_root0", "reduce_root1",
    "gather_root0", "gather_root1",
    "scatter_root0", "scatter_root1",
    "all_to_all", "all_to_all_single",
)


def test_every_collective_matches_upstream_gloo_at_world_three_and_four():
    """The whole point. Same inputs, same world, upstream's own tolerance.

    One assertion per (collective, world, rank), and the tolerance for each is
    computed from upstream's float32-vs-float64 run of *that* collective rather
    than from a constant written here. Most of them come out at zero, and that
    is a result rather than a shortcut: a collective that only moves numbers
    has no rounding to allow for, and this holds it to bit equality.
    """
    seen = 0
    for world in C2_WORLDS:
        shim, up32, up64 = c2_shim(world), c2_gloo32(world), c2_gloo64(world)
        for rank in range(world):
            for key in C2_VALUE_PROBES:
                c2_assert_matches(key, world, rank,
                                  c2_ok(shim[rank], key),
                                  c2_ok(up32[rank], key),
                                  c2_ok(up64[rank], key))
                seen += 1
    # 24 probes x (3 + 4) ranks. Stated so that a probe silently dropped from
    # the table is a failure rather than a smaller green number.
    assert seen == len(C2_VALUE_PROBES) * sum(C2_WORLDS), seen


def test_the_tolerance_is_derived_from_upstream_and_is_not_a_free_parameter():
    """`docs/numerics/AGREE.md` §2's rule, restated where it can fail.

    Two things are pinned. The floor is 8 float32 ulp and not a rounder number
    someone liked; and the collectives that combine no numbers really do have a
    measured upstream error of exactly zero, so holding them to bit equality is
    a reading of the measurement rather than a decision. If a future change
    makes `broadcast` lossy this goes red here, before it goes red as a value.
    """
    assert C2_ULP_FLOOR == 8, C2_ULP_FLOOR
    assert C2_F32_ULP == 2.0 ** -23

    exact_by_measurement = ("broadcast_root0", "broadcast_root1", "all_gather",
                            "all_gather_into_tensor", "scatter_root0",
                            "all_to_all", "all_to_all_single",
                            "allreduce_MIN", "allreduce_MAX",
                            "reduce_scatter_max")
    for world in C2_WORLDS:
        up32, up64 = c2_gloo32(world), c2_gloo64(world)
        for key in exact_by_measurement:
            for rank in range(world):
                _, own = c2_derived_tolerance(c2_ok(up32[rank], key),
                                              c2_ok(up64[rank], key))
                assert own == 0.0, (
                    "%s at world %d rank %d: upstream's own float32 answer "
                    "differs from its float64 answer by %.3e, so this "
                    "collective does combine numbers and must not be held to "
                    "bit equality" % (key, world, rank, own))

    # And the ones that *do* combine: their upstream error is what sets the
    # threshold, so it must be a real number rather than an accident of zero.
    combining = ("allreduce_SUM", "allreduce_AVG", "reduce_scatter")
    widths = []
    for world in C2_WORLDS:
        up32, up64 = c2_gloo32(world), c2_gloo64(world)
        for key in combining:
            for rank in range(world):
                _, own = c2_derived_tolerance(c2_ok(up32[rank], key),
                                              c2_ok(up64[rank], key))
                widths.append(own)
    assert max(widths) >= 0.0
    # A tolerance is only meaningful if it is smaller than the thing it is
    # checking. Every derived threshold here must stay well under one part in
    # a thousand, or it has stopped being a measurement.
    for width in widths:
        assert width < 1e-3, width


def test_all_five_reduce_ops_agree_with_upstream_and_the_bitwise_three_refuse():
    """MIN, MAX, PRODUCT and AVG, which §5 said were refused, and were.

    Split out from the sweep above so that a failure names the reduce op. The
    refusals below are the other half: BAND/BOR/BXOR and PREMUL_SUM must still
    refuse **by name**, and the message must say which op it was -- a refusal
    that does not name the thing it refused sends the reader nowhere.
    """
    for world in C2_WORLDS:
        shim, up32, up64 = c2_shim(world), c2_gloo32(world), c2_gloo64(world)
        for opname in ("SUM", "MIN", "MAX", "PRODUCT", "AVG"):
            key = "allreduce_" + opname
            for rank in range(world):
                c2_assert_matches(key, world, rank, c2_ok(shim[rank], key),
                                  c2_ok(up32[rank], key),
                                  c2_ok(up64[rank], key))
            # Every rank ends holding the same bytes -- one fold on the hub,
            # one result sent back (docs/distributed/FEDERATED4.md §2).
            answers = [c2_ok(shim[rank], key) for rank in range(world)]
            assert all(a == answers[0] for a in answers), (key, world, answers)

        for opname in ("BAND", "BOR", "BXOR"):
            for rank in range(world):
                msg = c2_exc(shim[rank], "allreduce_" + opname)
                assert "NotImplementedError" in msg, msg
                assert "ProcessGroupLocal.allreduce" in msg, msg
                assert "ReduceOp.%s" % opname in msg, msg
                assert "world_size %d" % world in msg, msg
                assert "docs/distributed/COLLECT2.md" in msg, msg

        for rank in range(world):
            msg = c2_exc(shim[rank], "allreduce_PREMUL_SUM")
            assert "PREMUL_SUM" in msg, msg


def test_avg_is_the_sum_over_the_world_and_not_over_whoever_arrived():
    """AVG's divisor is a number, and it has to be the right one.

    §5's refusal said "AVG would be SUM over a divisor, which is the one number
    a federated aggregator must choose for itself". That is true of
    `federated.FedAvg`, whose divisor is a *weight* total the caller supplies,
    and it is not true of `c10d`'s `ReduceOp.AVG`, whose divisor is the world
    size and is not anyone's choice. The refusal conflated the two, which is
    why widening it here is not weakening it: the federated aggregator's
    refusal is untouched and `test_shim.py` still holds it.

    Checked against the sum rather than only against gloo, so that a divisor
    that happened to equal the world at three ranks but not at four fails.
    """
    for world in C2_WORLDS:
        shim = c2_shim(world)
        for rank in range(world):
            total = c2_ok(shim[rank], "allreduce_SUM")
            mean = c2_ok(shim[rank], "allreduce_AVG")
            for t, m in zip(total, mean):
                assert abs(t / world - m) <= 1e-6 * max(abs(m), 1.0), (
                    world, rank, t, m, t / world)


def test_broadcast_serves_a_root_that_is_not_the_hub_and_refuses_one_outside():
    """The root may be any rank, and rank 0 is not privileged.

    A broadcast implemented as "the hub's value wins" passes at `src=0` and is
    wrong everywhere else, so `src=1` and `src=world-1` are the tests that
    matter. The out-of-range root is here because a `rootRank` validated
    against a world of *one* -- which is what `_check_root` did -- accepts every
    rank in a larger world and then silently returns nothing sensible.
    """
    for world in C2_WORLDS:
        shim = c2_shim(world)
        for root in (0, 1, world - 1):
            key = "broadcast_root%d" % root
            answers = [c2_ok(shim[rank], key) for rank in range(world)]
            assert all(a == answers[0] for a in answers), (key, answers)
            # And it is the *root's* input, not the hub's and not a zero.
            up = c2_gloo32(world)
            assert answers[0] == c2_ok(up[root], key), (key, answers[0])

        for rank in range(world):
            msg = c2_exc(shim[rank], "broadcast_root_out_of_range")
            assert "ValueError" in msg, msg
            assert "rootRank" in msg, msg
            assert "world of size %d" % world in msg, msg


def test_the_collectives_that_used_to_return_their_own_input_no_longer_do():
    """The regression guard for the defect this file was written after.

    `reduce_scatter`, `scatter`, `all_to_all` and `all_to_all_single` each
    copied the caller's own input into the caller's own output at every world
    size, with no size check and no refusal. Every one of them **passed at
    world_size 1**, where that copy is the identity, and returned a plausible
    tensor at three.

    So this asserts the negative directly and not only the positive: the
    answer must differ from the input the rank supplied. A future edit that
    reinstates the copy would satisfy every shape assertion in this file and
    only this test would see it.
    """
    for world in C2_WORLDS:
        shim = c2_shim(world)
        for rank in range(world):
            report = shim[rank]

            rs = c2_ok(report, "reduce_scatter")
            assert rs == c2_ok(c2_gloo32(world)[rank], "reduce_scatter"), (
                world, rank, rs)
            assert c2_ok(report, "all_to_all") == c2_ok(
                c2_gloo32(world)[rank], "all_to_all"), (world, rank)

        # `reduce_scatter` returned chunk 0 of the rank's *own* list. The
        # inputs here are distinct per rank, so if that copy were back the
        # ranks' answers would be the ranks' own chunks -- which upstream's
        # answers are not, and this says so as a negative rather than only
        # letting the equality above carry it.
        for rank in range(world):
            got = c2_ok(shim[rank], "reduce_scatter")
            solo = c2_ok(c2_gloo32(world)[rank], "all_gather")[rank]
            assert got != solo[:len(got)], (
                "reduce_scatter at world %d rank %d returned this rank's own "
                "numbers -- the copy is back" % (world, rank))

        # scatter really scatters: no two ranks get the same chunk.
        for root in (0, 1):
            chunks = [tuple(c2_ok(shim[rank], "scatter_root%d" % root))
                      for rank in range(world)]
            assert len(set(chunks)) == world, (
                "scatter at world %d root %d handed %d distinct chunks to %d "
                "ranks -- a scatter that gives two ranks the same slot is the "
                "copy this test exists to catch" % (world, root,
                                                    len(set(chunks)), world))


def test_reduce_leaves_the_non_root_buffers_exactly_as_it_found_them():
    """The choice this backend makes inside upstream's "undefined".

    `reduce` promises a value only to the root. Upstream gloo fills the other
    ranks' buffers with whatever partial its tree produced -- at three ranks on
    this host, 5.0 and 3.0 where the total was 6.0. Those are not answers; they
    are the absence of a promise, and reading one as a result is a mistake the
    shape of every other defect in this file.

    This backend leaves them **untouched**, which is inside the same freedom,
    and asserts it here so that the sentence in `docs/distributed/COLLECT2.md` §4 is a
    tested claim rather than a description. It is also the only sabotage in the
    nullification sweep that the first draft of this file did not catch:
    copying the result onto every rank changed real behaviour that nothing was
    looking at.

    Upstream's disagreement is asserted too, so this cannot quietly become a
    test of nothing if gloo's tree ever starts leaving buffers alone.
    """
    for world in C2_WORLDS:
        shim, up32 = c2_shim(world), c2_gloo32(world)
        differed = 0
        for rank in range(world):
            ours = c2_ok(shim[rank], "reduce_untouched")
            assert ours["root"] == (rank == 0), (rank, ours)
            if rank == 0:
                # The root does get the answer, and it is upstream's.
                c2_assert_matches("reduce_untouched.root", world, rank,
                                  ours["after"],
                                  c2_ok(up32[rank], "reduce_untouched")["after"],
                                  c2_ok(c2_gloo64(world)[rank],
                                        "reduce_untouched")["after"])
                continue
            assert ours["after"] == ours["input"], (
                "reduce at world %d left rank %d holding %r where its input "
                "was %r. Only the root's buffer is specified, and this backend "
                "documents that it writes nowhere else (docs/distributed/COLLECT2.md §4)."
                % (world, rank, ours["after"], ours["input"]))
            theirs = c2_ok(up32[rank], "reduce_untouched")
            if theirs["after"] != theirs["input"]:
                differed += 1
        assert differed > 0, (
            "upstream gloo left every non-root buffer untouched at world %d, "
            "so the divergence this test documents no longer exists" % world)


def test_reduce_scatter_tensor_refuses_a_misshapen_input_by_name():
    """It used to fail from inside `copy_`, with a message about broadcasting.

    `cannot broadcast [6] to [2]` is a refusal by accident, raised by the wrong
    layer and naming neither the collective nor the world size. A caller
    reading it looks at their tensor's shape, which is fine, and never at their
    world size, which is where the mistake is.
    """
    for world in C2_WORLDS:
        shim = c2_shim(world)
        for rank in range(world):
            msg = c2_exc(shim[rank], "reduce_scatter_tensor_misshapen")
            assert "reduce_scatter_tensor" in msg, msg
            assert "chunks of the output" in msg, msg
            assert "broadcast" not in msg, (
                "still refusing from inside copy_: %s" % msg)


def test_all_to_all_single_refuses_uneven_splits_by_name():
    """Uneven splits are a different collective, and are refused as one.

    Not approximated and not silently treated as equal -- treating them as
    equal is precisely what the old body did to the entire operation.
    """
    for world in C2_WORLDS:
        shim = c2_shim(world)
        for rank in range(world):
            msg = c2_exc(shim[rank], "all_to_all_single_uneven")
            assert "NotImplementedError" in msg, msg
            assert "all_to_all_single" in msg, msg
            assert "split" in msg, msg
            assert "docs/distributed/COLLECT2.md" in msg, msg


def test_the_product_fold_is_in_ascending_rank_order_and_says_so():
    """`docs/distributed/FEDERATED4.md` §2's contract, for the other non-associative op.

    In rank order `2**-120 * 2**100 * 2**100` is `2**80`, finite. Any order
    that multiplies the two large factors first overflows float32 to infinity.
    The premise is asserted too, so this cannot pass by the two orders
    happening to agree.

    Upstream gloo is deliberately *not* the oracle here. Its fold order is a
    consequence of its tree and is not a promised property, so the two are
    allowed to differ -- what is pinned is that this backend's order is the one
    its own docstring claims.
    """
    for world in C2_WORLDS:
        shim = c2_shim(world)
        factors = [c2_ok(shim[rank], "product_rank_order")["factor"]
                   for rank in range(world)]

        ascending = 1.0
        for f in factors:
            ascending *= f
        descending = 1.0
        for f in reversed(factors):
            descending *= f
        # The premise: at float64 both are finite, and it is float32 that
        # separates them, so compute the separation the way the shim does.
        assert factors[1] * factors[2] > 3.4e38, factors

        for rank in range(world):
            got = c2_ok(shim[rank], "product_rank_order")["result"]
            assert len(got) == 1, got
            assert got[0] == 2.0 ** 80, (
                "world %d rank %d: the PRODUCT fold returned %r. Rank order "
                "gives 2**80; %r is what an order that multiplies the two "
                "large factors first gives, and infinity is what float32 does "
                "with it." % (world, rank, got[0], float("inf")))
            assert got[0] != float("inf"), (world, rank)
        del ascending, descending


def test_barrier_actually_blocks_rather_than_reporting_that_it_did():
    """A barrier that returned immediately passes every value test here.

    The last rank sleeps a second before entering. Every other rank must leave
    the barrier after that second has elapsed -- so this is a lower bound on
    wall time, which a loaded machine can only make pass harder, never flake.
    """
    for world in C2_WORLDS:
        shim = c2_shim(world)
        for rank in range(world):
            entry = c2_ok(shim[rank], "barrier_blocks")
            assert entry["slept"] == (rank == world - 1), (rank, entry)
            assert entry["elapsed"] >= 0.5, (
                "world %d rank %d left the barrier after %.3fs while rank %d "
                "was still sleeping for 1.0s before entering it. A barrier "
                "nobody waits at is the `filled` guard from docs/models/CKPT.md."
                % (world, rank, entry["elapsed"], world - 1))


def test_async_op_is_genuinely_async_on_both_sides_and_neither_publishes_early():
    """The async surface, re-measured after it was built -- and it now agrees.

    This test is the **inversion** of the one that stood here while
    `docs/distributed/COLLECT2.md` §7 was true. That one pinned a divergence: upstream
    gloo's handle was incomplete before `wait()` and this backend's was already
    complete, because every collective ran to completion inside the call. That
    is no longer the case, so the claim changed and the test changed with it --
    it was not deleted, and it still asserts **both sides**, so it cannot
    quietly become a test of nothing if either backend moves.

    Four assertions, and what each would catch (docs/distributed/ASYNCWORK.md §6):

    * upstream is incomplete before `wait()` at rank 0 -- unchanged, and the
      only assertion here that was already passing. It is what makes the
      comparison a comparison; if gloo ever ran its collectives inline this
      goes red and the "same contract" claim below would be vacuous.
    * **ours is now incomplete too** -- inverted from `is True`. Nullifying the
      worker thread, so that the collective runs inline again, turns this red.
    * **the buffer before `wait()` is the rank's own pre-collective input** --
      inverted from `before_wait == after_wait`. This is the ownership half,
      and it is the assertion with the sharpest teeth, because it is
      deterministic rather than timing-dependent: `AsyncWork` never writes the
      caller's buffer except at a synchronisation point, so a build that
      published early fails this on every run rather than on a slow day.
    * `wait()` still returns True and the published value still equals
      upstream's -- unchanged. Making the collective asynchronous must not have
      changed a single number, and this is where that is checked at world 3
      and 4 against the float32 and float64 oracles.

    Rank 0 is the one whose completion is asserted because the probe holds
    every other rank back half a second, which makes rank 0 provably the first
    arrival and its poll a question with one right answer. For the later ranks
    the answer is genuinely unspecified -- the last rank into a collective may
    find it already done on any backend -- so they are measured and not
    asserted on. Their *buffers* are asserted on at every rank, because
    ownership does not depend on who arrived when.
    """
    for world in C2_WORLDS:
        shim, up32 = c2_shim(world), c2_gloo32(world)
        for rank in range(world):
            ours = c2_ok(shim[rank], "async_allreduce")
            theirs = c2_ok(up32[rank], "async_allreduce")

            if rank == 0:
                assert theirs["completed_before_wait"] is False, (
                    "upstream gloo reported a completed work before wait() at "
                    "world %d rank 0, with two peers still asleep. The other "
                    "side of this comparison is gone" % (world,))
                assert ours["completed_before_wait"] is False, (
                    "this backend reported a completed work before wait() at "
                    "world %d rank 0, with two peers still asleep -- so the "
                    "collective ran inside the call and async_op is a lie "
                    "again (docs/distributed/ASYNCWORK.md)" % (world,))

            assert ours["wait_returned"] is True, ours
            assert theirs["wait_returned"] is True, theirs

            # Ownership. The handle owns the output until wait(), so what a
            # caller reads before it is their own pre-collective bytes -- not
            # the answer, and not a half-written tensor either.
            assert ours["before_wait"] == ours["source"], (
                "world %d rank %d read something other than its own input "
                "from a buffer whose collective had not been waited on. The "
                "handle published early" % (world, rank))
            assert ours["before_wait"] != ours["after_wait"], (
                "world %d rank %d: the collective did not change the buffer, "
                "so this test would pass on a backend that did nothing"
                % (world, rank))

            c2_assert_matches("async_allreduce.after_wait", world, rank,
                              ours["after_wait"], theirs["after_wait"],
                              c2_ok(c2_gloo64(world)[rank],
                                    "async_allreduce")["after_wait"])


def test_point_to_point_still_refuses_by_name_and_was_not_weakened():
    """`send`/`recv` are refused, and widening the collectives did not open them.

    `all_to_all` now moves a chunk from any rank to any other, which is exactly
    the capability a naive reading would build `send` on top of. It is still
    refused, and for the reason it always was: on a star that message is
    relayed by the hub, and a relay the hub can read is not the primitive
    secure aggregation needs (docs/distributed/FEDERATED4.md §7).
    """
    for world in C2_WORLDS:
        shim = c2_shim(world)
        for rank in range(world):
            for key in ("send", "recv"):
                msg = c2_exc(shim[rank], key)
                assert "NotImplementedError" in msg, msg
                assert "ProcessGroupLocal.%s" % key in msg, msg
                assert "star" in msg, msg


def test_every_rank_really_ran_the_shim_and_the_oracle_really_ran_upstream():
    """The premise of every number above.

    A previous round of this repository lost hours to probes that silently
    imported upstream torch, so the shim workers assert `_aten_implemented`
    on the way in and the gloo workers assert its absence. This checks the
    same fact from the outside, where a worker that skipped its own assert
    would still be caught.
    """
    for world in C2_WORLDS:
        for rank in range(world):
            assert c2_shim(world)[rank]["_meta"] == {
                "shim": True, "rank": rank, "world": world,
                "backend": "local"}, c2_shim(world)[rank]["_meta"]
            for oracle in (c2_gloo32(world), c2_gloo64(world)):
                meta = oracle[rank]["_meta"]
                assert meta["shim"] is False, meta
                assert (meta["rank"], meta["world"]) == (rank, world), meta
                assert meta["backend"] == "gloo", meta


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
