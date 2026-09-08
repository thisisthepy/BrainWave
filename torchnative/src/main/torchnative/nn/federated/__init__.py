"""Federated learning.

A layer above the single-device adaptation methods, not a sibling of them: the
local step is the same mechanism, with aggregation, transport and privacy on
top. Its dependencies stay behind the ``federated`` extra so that using
adaptation alone does not pull in the aggregation stack. See DESIGN.md §3.

    from torchnative.nn import federated

    engine = federated.Engine(model, method=adapt.Tent(), lr=1e-3,
                              aggregator=federated.FedAvg())
    report = engine.participate(batches, weight=n_local_samples)

**What this is built on, and why it is one round.** DESIGN.md §11.1 stacks
rounds/selection/aggregation above ``torch.distributed``, and aggregation *is*
collective communication -- so ``FedAvg`` here is a weighted ``all_reduce`` and
not a private channel. The transport under it (docs/distributed/TRANSPORT.md) implements
``world_size`` 2 with ``ReduceOp.SUM``; everything past that refuses by name,
and so does everything here that would need it.

**The refusal that matters most is at one rank.** ``FedAvg`` over a world of
one is the identity function: it returns the delta it was handed, and no test
at that size can distinguish a correct weighted average from an aggregator that
ignores its weights, drops a peer, or does nothing at all. So a world of one is
refused rather than served -- see :meth:`FedAvg.aggregate`. This is the one
place in this package where "it works at world_size 1" would be a lie told by a
green test rather than by a sentence.

**Two premises are checked rather than assumed**, because both fail silently:

``the ranks cover the same parameters``
    An ``all_reduce`` sums element-wise whatever it is handed. Two ranks whose
    deltas cover different names, in a different order, or at different shapes
    would produce a number rather than an error. :func:`agree` refuses first.

``the ranks started from the same weights``
    FedAvg averages *offsets from a common base*. If the bases differ the
    average is over incomparable quantities, and again nothing raises.
    :meth:`torchnative.delta.Delta.publish` refuses first.

Both are answered with one ``int64`` ``all_reduce`` of a digest, which costs a
collective of one element and is exact -- ``h0 + h1 == 2 * h0`` if and only if
``h1 == h0``, which is why the check is written for a world of exactly two and
says so.

**What is not here**, named rather than approximated:

======================================  ====================================
more than one round                     built -- ``Engine(rounds=n)``, over
                                        ``Delta.re_snapshot``
                                        (docs/distributed/FEDERATED2.md)
participant selection                   *half* built. :func:`cohort` agrees
                                        the participant set across the ranks,
                                        which nothing downstream would notice
                                        going wrong; a **proper subset**
                                        refuses, because it needs a sub-group
                                        and a world larger than two
a rank that does not arrive             ``Engine(on_missing='refuse')`` is
                                        implemented and is the default: the
                                        collective raises
                                        :class:`RankDropped`, naming the rank,
                                        and the round is *undone* rather than
                                        left half-applied.
                                        ``on_missing='average_arrived'``
                                        refuses
aggregators other than FedAvg           :class:`FedAvgM` (server momentum) and
                                        :class:`FedProx` (a proximal term on
                                        the *local* objective) are built.
                                        FedAdam, SCAFFOLD: none
secure aggregation, differential
privacy, compression                    not offered at all -- no surface here
                                        takes a key, an epsilon or a codec,
                                        and ``Engine(secure_aggregation=...)``
                                        / ``Engine(differential_privacy=...)``
                                        refuse by name with the chain of
                                        things each would need first
======================================  ====================================

docs/distributed/FEDERATED.md records the measurements and what each refusal was weighed
against.
"""

from __future__ import annotations

import contextlib
import hashlib
import threading

import torch

__all__ = ["FedAvg", "FedAvgM", "FedProx", "Engine", "Round", "digest",
           "agree", "cohort", "RankDropped", "ABSTAIN", "tolerate_missing"]


# The wire is JSON over a socket (docs/distributed/TRANSPORT.md §2). This guard existed
# because both ranks used to `sendall` before either received, so a payload
# larger than the kernel's socket buffer blocked both of them until the 30 s
# timeout: 2,836,968 B completed in 0.17 s and 4,053,011 B failed after 30 s,
# and the wall was not even a constant -- the same 8.1 MB went through in
# 0.23 s if smaller collectives had run first and grown the buffers.
#
# **That transport defect is fixed**: `ProcessGroupLocal.allreduce` now orders
# the exchange by rank, so one side is always draining. 3,000,000 floats
# complete in 0.60 s where 4 MB used to hang. The guard is kept, far above the
# old wall, as a *refusal* rather than a limit -- there is no longer a known
# size that fails, and a number here that nothing measured would be a claim
# this file cannot support. It is the size past which nobody has run one.
# 15 MB: what 3,000,000 `ones` floats actually encode to, which is the
# largest payload measured through the fixed transport (0.60 s). Not a
# rounded-up guess -- raise it with another measurement.
_WIRE_SAFE_BYTES = 15_000_000
_WIRE_COLD_OK_BYTES = 2_836_968
_WIRE_COLD_DEADLOCK_BYTES = 4_053_011

# An upper bound on the JSON length of one float element including its ", "
# separator: `repr` of a float64 holding a float32 value is at most 24
# characters ("-1.1754943508222875e-38"). Used so the common case costs no
# serialisation -- the exact length is measured only when this bound is passed.
_MAX_ELEMENT_BYTES = 26


def _dist():
    """``torch.distributed``, with the ``local`` backend registered.

    Imported here rather than at module import so that reading
    ``torchnative.nn.federated`` does not register a backend as a side effect.
    ``torchnative.distributed`` is what registers it, and its ``register()`` is
    idempotent by construction.
    """
    import torch.distributed as dist

    import torchnative.distributed  # noqa: F401  -- registers backend="local"

    return dist


def _group(group):
    """``(world_size, rank)`` for ``group``, refusing an uninitialised world.

    A caller who never called ``init_process_group`` gets the check that would
    have made this work, rather than ``Default process group has not been
    initialized`` from four frames down.
    """
    dist = _dist()
    if not dist.is_initialized():
        raise RuntimeError(
            "torchnative.nn.federated: there is no process group, so there is "
            "nobody to aggregate with.\n"
            "Check: import torchnative.distributed; "
            "torch.distributed.init_process_group(backend='local', "
            "init_method='tcp://127.0.0.1:<port>', rank=<0 or 1>, "
            "world_size=2)  -- docs/distributed/TRANSPORT.md"
        )
    return dist.get_world_size(group), dist.get_rank(group)


def _require_world(group, who, minimum=2):
    """``(world_size, rank)``, refusing a world too small to prove anything.

    One place, because the *interesting* refusal -- a world of one -- has to
    read the same wherever a caller hits it. It was written twice first, and
    the second copy (in ``agree``) shadowed the first: at ``world_size = 1``
    ``Delta.publish`` reached the base-agreement check before the aggregator
    and reported "written for a world of exactly two", which is true and is
    not the point.

    ``minimum`` is 2 for aggregation itself and larger for the things that
    two ranks cannot show: a **proper subset** cohort and a **survivor set**
    of two or more both need three (docs/distributed/FEDERATED4.md), because at two ranks
    each of them is a world of one and FedAvg over one delta is that delta.
    """
    world, rank = _group(group)
    if world == 1:
        raise NotImplementedError(
            "torchnative.nn.federated: a world of one, reached through %s. "
            "Averaging a single client's delta with itself is the identity "
            "function, so this would return what it was handed and prove "
            "nothing about the aggregation -- a test at this size passes "
            "whether the weights are honoured, ignored, or never read. "
            "FedAvg is not defined here as 'the average of one'; the "
            "degenerate case is named instead of served.\n"
            "Check: torch.distributed.get_world_size() >= 2, reached through "
            "init_process_group(backend='local', init_method='tcp://...', "
            "world_size=2) -- docs/distributed/TRANSPORT.md" % (who,)
        )
    if world < minimum:
        raise NotImplementedError(
            "torchnative.nn.federated: world_size %d, reached through %s, "
            "which needs at least %d. The transport carries any world "
            "(docs/distributed/FEDERATED4.md), but %d ranks are not enough to show what "
            "this does: whatever it selects or whatever survives is a world "
            "of one, and FedAvg over one delta is that delta -- so a test of "
            "it would pass with no aggregation at all.\n"
            "Check: init_process_group(..., world_size=%d)"
            % (world, who, minimum, world, minimum)
        )
    return world, rank


#: The spelling ``Delta.publish`` calls. It no longer means "exactly two" --
#: it means "at least two", which is what it always checked for.
_require_two = _require_world


class _Arrival:
    """Who reported, across the collectives of one round.

    ``Engine(on_missing='average_arrived')`` divides by the weights of the
    ranks that arrived, and that is only a number if **every** collective in
    the round agreed about which ranks those were. The survivor set is the
    hub's verdict and travels with the data
    (``ProcessGroupLocal.allreduce_partial``), so the survivors hold the same
    one -- but a rank that leaves *between* two collectives of the same round
    changes it, and a divisor that changed halfway is not the divisor of any
    average. So it is recorded once and compared after that.
    """

    def __init__(self, world):
        self.world = world
        self.missing = None

    @property
    def survivors(self):
        if self.missing is None:
            return tuple(range(self.world))
        return tuple(r for r in range(self.world) if r not in self.missing)

    def record(self, missing, what):
        missing = tuple(sorted(int(r) for r in missing))
        if self.missing is None:
            self.missing = missing
            return
        if missing != self.missing:
            raise RankDropped(
                "torchnative.nn.federated: the survivor set changed inside "
                "one round -- %s were missing, and %s are missing during %s. "
                "A weighted mean whose divisor changed halfway is not the "
                "mean of anything, so this refuses rather than reporting the "
                "last one."
                % (list(self.missing), list(missing), what),
                missing=missing, during=what,
            )


_ARRIVAL = threading.local()


@contextlib.contextmanager
def tolerate_missing(group=None):
    """Let the collectives inside complete over whoever arrived.

    Yields an :class:`_Arrival`. Outside this block every collective in this
    module is all-or-nothing and a lost rank raises :class:`RankDropped`;
    inside it, the collectives use the shim's ``*_partial`` spellings, which
    return the hub's survivor set alongside the data.

    This is a *mechanism*. The policy -- and the minimum number of survivors
    the caller is willing to divide by -- is ``Engine(on_missing=...,
    min_participants=...)``, because "however many arrived" is exactly the
    divisor nobody chose that :class:`RankDropped` exists to refuse.
    """
    world, _ = _group(group)
    previous = getattr(_ARRIVAL, "current", None)
    arrival = _Arrival(world)
    _ARRIVAL.current = arrival
    try:
        yield arrival
    finally:
        _ARRIVAL.current = previous


def _arrival():
    return getattr(_ARRIVAL, "current", None)


def _backend(group, what):
    """The shim's backend object for ``group``, or a refusal naming why.

    ``allreduce_partial`` has no upstream spelling: upstream's allreduce either
    completes over the world or is an error. So this reaches past
    ``torch.distributed``'s API to the backend the shim registered, and says so
    when that backend is not there.
    """
    dist = _dist()
    pg = group if group is not None else dist.distributed_c10d._get_default_group()
    sole = getattr(pg, "_sole_backend", None)
    if sole is None:
        raise NotImplementedError(
            "torchnative.nn.federated: %r is not this shim's process group, "
            "and %s has no upstream spelling -- a collective that completes "
            "over whoever arrived is not an allreduce, so no backend but this "
            "one offers it (docs/distributed/FEDERATED4.md)" % (pg, what)
        )
    return sole(what)


def _sum_opts():
    dist = _dist()

    class _Opts:
        pass

    opts = _Opts()
    opts.reduceOp = dist.ReduceOp.SUM
    return opts


def _all_reduce(tensor, group, what):
    """``all_reduce(SUM)``, tolerant or not depending on the context."""
    dist = _dist()
    arrival = _arrival()
    if arrival is None:
        return _collective(
            lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group),
            group, what)

    def run():
        work, missing = _backend(group, "allreduce_partial").allreduce_partial(
            [tensor], _sum_opts())
        arrival.record(missing, what)
        return work

    return _collective(run, group, what)


def _all_gather(holders, tensor, group, what):
    """``all_gather``, tolerant or not depending on the context."""
    dist = _dist()
    arrival = _arrival()
    if arrival is None:
        return _collective(
            lambda: dist.all_gather(holders, tensor, group=group), group, what)

    def run():
        work, missing = _backend(group, "allgather_partial").allgather_partial(
            [holders], [tensor])
        arrival.record(missing, what)
        return work

    return _collective(run, group, what)


class RankDropped(RuntimeError):
    """A rank that was selected for this round did not report.

    Raised *instead of* an aggregate, never alongside one.  The failure this
    names is the one docs/design/DESIGN.md §6 puts below a refusal: an aggregator that
    divides by however many ranks arrived returns a number, and the round
    reports success while the model it produced is a weighted mean over a
    cohort nobody chose.  ``sum(w_k d_k) / sum_{arrived}(w_k)`` is not the
    quantity FedAvg names, and nothing downstream can tell the difference.

    ``rank`` is this process's, ``missing`` the peer(s) that did not report,
    and ``during`` the collective that noticed.
    """

    def __init__(self, message, rank=None, missing=(), during=None):
        super().__init__(message)
        self.rank = rank
        self.missing = tuple(missing)
        self.during = during


def _collective(fn, group, what):
    """Run one collective, turning a lost peer into :class:`RankDropped`.

    The transport is a socket (docs/distributed/TRANSPORT.md §2), so a rank that exits
    mid-round surfaces as ``RuntimeError('connection closed')`` from
    ``_recv_all``, as a ``socket.timeout`` after 30 s, or as ``BrokenPipeError``
    on the send.  All three mean the same thing to this layer and none of them
    says so: they name the socket rather than the round.

    Translating them is the *mechanism*; the policy is in
    :class:`Engine`'s ``on_missing=``.  What is guaranteed here is that no
    partial average exists to be returned -- the exception replaces the value
    rather than accompanying it, because ``all_reduce`` either completes over
    every rank or does not complete.
    """
    import struct

    dist = _dist()
    try:
        return fn()
    except RankDropped:
        raise
    except (OSError, EOFError, struct.error) as exc:
        pass
    except RuntimeError as exc:
        if "connection closed" not in str(exc):
            raise
    try:
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        missing = tuple(r for r in range(world) if r != rank)
    except Exception:  # the group itself is gone
        world, rank, missing = -1, -1, ()
    raise RankDropped(
        "torchnative.nn.federated: rank %s did not report during %s, so this "
        "round has no aggregate. The collective over %s rank(s) did not "
        "complete, and no partial average was produced -- averaging over "
        "whichever ranks arrived would divide by a number nobody chose and "
        "would report success (docs/design/DESIGN.md §6).\n"
        "Policy: Engine(on_missing='refuse') is the default and is this. "
        "Engine(on_missing='average_arrived', min_participants=k) divides by "
        "the survivors instead, and needs k >= 2 and a world of at least "
        "three -- at two ranks the survivor set is one and FedAvg over one "
        "delta is that delta (docs/distributed/FEDERATED4.md)."
        % (",".join(str(r) for r in missing) or "?", what, world),
        rank=rank, missing=missing, during=what,
    )


def digest(table, values=True):
    """A deterministic integer over a ``{name: tensor}`` table.

    Over the *schema* -- names, shapes, dtypes, in sorted name order -- and,
    with ``values=True``, over the bytes as well.

    ``hashlib`` rather than ``hash()``: this number is compared across
    processes, and CPython salts ``hash`` of a string per interpreter, so the
    built-in would disagree between two ranks holding identical tables. That
    failure looks exactly like the disagreement this exists to detect.

    Truncated to 56 bits so ``value * world_size`` stays exact in the ``int64``
    tensor :func:`agree` reduces it in.

    The bytes come from ``Delta._bytes``, the same little-endian encoding
    ``Delta.persist`` writes -- so two ranks agree here exactly when a byte
    comparison of what they would persist agrees.
    """
    from torchnative.delta import Delta

    h = hashlib.sha256()
    for name in sorted(table):
        t = table[name]
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(repr(tuple(t.shape)).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(t.dtype).encode("utf-8"))
        h.update(b"\x00")
        if values:
            h.update(Delta._bytes(t)[1])
        h.update(b"\x01")
    return int.from_bytes(h.digest()[:7], "big")


def agree(value, group=None, what="this value"):
    """Refuse unless every rank passed the same integer. Returns the sum.

    One ``int64`` ``all_gather``, and the comparison is **element by element
    against every rank's digest**, not against a total.

    It used to be one ``all_reduce(SUM)`` and the test ``total == value *
    world``, which is an equality test only because the world had exactly two
    members: ``h0 + h1 == 2 * h0`` iff ``h0 == h1``.  At three it accepts
    ``(h - 1, h, h + 1)`` -- three ranks that disagree about the schema, the
    base or the cohort, summing to exactly what agreement would have summed
    to.  docs/distributed/FEDERATED3.md §5 named that and refused the larger world rather
    than weaken here; ``ProcessGroupLocal.allgather`` at ``world_size N``
    (docs/distributed/FEDERATED4.md) is what settles it, so the refusal is gone and the
    check is stronger rather than wider.

    Inside :func:`tolerate_missing` the gather is the partial spelling and the
    ranks that did not report are not compared -- there is nothing of theirs
    to compare.
    """
    world, rank = _require_world(group, "federated.agree")
    probe = torch.tensor([int(value)], dtype=torch.int64)
    # Filled with a value no digest can take, so a slot the transport never
    # wrote is visible as one rather than read as a zero that agrees.
    holders = [torch.full((1,), -1, dtype=torch.int64) for _ in range(world)]
    _all_gather(holders, probe, group,
                "federated.agree(%s)" % (what,))
    arrival = _arrival()
    missing = set(arrival.missing or ()) if arrival is not None else set()
    seen = {r: int(holders[r][0].item())
            for r in range(world) if r not in missing}
    seen[rank] = int(value)
    odd = sorted(r for r, v in seen.items() if v != int(value))
    if odd:
        raise ValueError(
            "torchnative.nn.federated: the ranks disagree about %s. Rank %d "
            "holds digest %d; the %d ranks that reported hold %s, and rank(s) "
            "%s differ from this one.\n"
            "Gathered rank by rank rather than summed: a sum-and-compare is "
            "an equality test only at two ranks, and at three it accepts "
            "(h-1, h, h+1). An all_reduce would have averaged them anyway and "
            "returned a number, which is why this is checked instead of "
            "assumed."
            % (what, rank, int(value), len(seen),
               [seen[r] for r in sorted(seen)], odd)
        )
    return int(value) * len(seen)


def _check_wire(tensor, name):
    """Refuse a tensor too large for the transport to carry, by name.

    See the ``_WIRE_*`` constants: this is a limit of ``ProcessGroupLocal``'s
    send-before-receive, not of aggregation, and the message says so because
    the fix belongs there.
    """
    if tensor.numel() * _MAX_ELEMENT_BYTES + 2 <= _WIRE_SAFE_BYTES:
        return
    import json

    size = len(json.dumps(tensor.tolist()).encode("utf-8"))
    if size <= _WIRE_SAFE_BYTES:
        return
    raise NotImplementedError(
        "torchnative.nn.federated: %r is %d elements and %d bytes on the wire, "
        "and nobody has run a round that large. The bound is %d.\n"
        "This is a refusal, not a measured wall. There used to be one: both "
        "ranks called sendall before either received, so a payload past the "
        "socket buffer blocked both until the 30 s timeout -- %d B completed "
        "and %d B did not, and the wall moved with how much traffic came "
        "before it. ProcessGroupLocal.allreduce now orders the exchange by "
        "rank, and 3,000,000 floats complete in 0.60 s. Raise this bound with "
        "a measurement rather than a guess."
        % (name, tensor.numel(), size, _WIRE_SAFE_BYTES,
           _WIRE_COLD_OK_BYTES, _WIRE_COLD_DEADLOCK_BYTES)
    )


def _total_weight(weight, group=None):
    """``sum(weight)`` over the group, as a float. One 1-element collective."""
    dist = _dist()
    total = torch.tensor([float(weight)], dtype=torch.float64)
    _all_reduce(total, group, "the sum of the ranks' weights")
    return float(total[0].item())


class _Abstain:
    """The weight of a rank that is in the world but not in this round's cohort.

    Not ``0.0``, which every aggregator here refuses: a rank weighted zero
    contributes nothing while still counting as having participated, and that
    is a caller mistake worth refusing. This is the same arithmetic said on
    purpose -- ``ABSTAIN`` contributes a zero table at weight zero, so the
    divisor is the cohort's weight and the aggregate is the cohort's mean.

    It exists because a proper subset of the ranks still has to *attend* every
    collective: the transport is a star and a rank that skipped one would hang
    the ones that did not. Attending without contributing is what selection
    means here, and this is the value that says so.
    """

    __slots__ = ()

    def __repr__(self):
        return "federated.ABSTAIN"

    def __bool__(self):
        return False


#: See :class:`_Abstain`. Passed as ``weight=`` by :class:`Engine` for a rank
#: outside the round's cohort.
ABSTAIN = _Abstain()


class FedAvg:
    """Federated averaging: the weighted mean of the ranks' deltas.

    ``sum(w_k * d_k) / sum(w_k)`` over the ranks of a process group, element by
    element, computed with ``all_reduce(SUM)`` -- which is the identity McMahan
    et al. (2017) write for FedAvg and also, literally, what the collective
    does. Every rank ends holding the same table.

    **Weighting is explicit or refused.** ``FedAvg()`` requires a ``weight`` at
    every call. ``FedAvg(weighted=False)`` takes none and gives every rank 1.0.
    There is no default sample count and none is inferred from a batch: an
    aggregator that silently weighted every client equally when the caller
    meant to weight by data volume produces a different model and reports
    success, which docs/design/DESIGN.md §6 puts below a refusal.
    """

    def __init__(self, weighted=True):
        self.weighted = bool(weighted)

    def __repr__(self):
        return "FedAvg(weighted=%r)" % (self.weighted,)

    def resolve_weight(self, weight):
        """The weight this aggregator will use, or a refusal. Not a collective.

        Split out so that :class:`Engine` reports the same number the average
        was computed with, rather than a second guess at it.
        """
        if weight is ABSTAIN:
            # A rank outside the cohort. It attends the collectives -- it has
            # to, the others are waiting on it -- and contributes nothing.
            return 0.0
        if self.weighted:
            if weight is None:
                raise TypeError(
                    "torchnative.nn.federated.FedAvg.aggregate: weight= is "
                    "required. FedAvg weights each client by how much data it "
                    "trained on, and there is no safe guess -- weighting two "
                    "clients equally when one holds ten times the data is a "
                    "different model, silently.\n"
                    "Pass weight=<local sample count>, or say so with "
                    "FedAvg(weighted=False)."
                )
            w = float(weight)
            if w != w or w in (float("inf"), float("-inf")) or not w > 0.0:
                raise ValueError(
                    "torchnative.nn.federated.FedAvg: weight=%r. A weight has "
                    "to be finite and positive; a rank weighted 0 contributes "
                    "nothing while still counting as having participated"
                    % (weight,)
                )
            return w
        if weight is not None:
            raise TypeError(
                "torchnative.nn.federated.FedAvg(weighted=False) was given "
                "weight=%r. Unweighted means every rank counts 1.0, so a "
                "weight here would be accepted and ignored" % (weight,)
            )
        return 1.0

    def aggregate(self, table, weight=None, group=None):
        """The weighted average of ``table`` across the group. Returns a table.

        ``table`` is ``{name: tensor}`` -- what ``Delta.value`` holds and what
        ``Delta.load`` returns. Not a ``Delta``: which model an averaged offset
        belongs on is the caller's knowledge, the reasoning ``Delta.load``'s
        docstring gives.

        **A world of one is refused.** At ``world_size = 1`` this function *is*
        ``table``; returning it would make every test of this class pass no
        matter what the arithmetic below said, including no arithmetic at all.
        A federated aggregator that cannot fail is not evidence of anything, so
        the degenerate case is named instead of served.
        """
        world, rank = _require_world(group, "FedAvg.aggregate")
        if not table:
            raise ValueError(
                "torchnative.nn.federated.FedAvg: an empty table. A round that "
                "contributes no parameters would complete and change nothing"
            )
        w = self.resolve_weight(weight)

        for name, t in table.items():
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    "torchnative.nn.federated.FedAvg: %r is %r, not a Tensor"
                    % (name, type(t).__name__)
                )
            if not t.dtype.is_floating_point:
                raise NotImplementedError(
                    "torchnative.nn.federated.FedAvg: %r is %s. A delta holds "
                    "floating parameters; averaging an integer table would "
                    "round every intermediate, and no caller here means that"
                    % (name, t.dtype)
                )

        # The two ranks must be averaging the same thing. `all_reduce` sums
        # element-wise whatever it is handed, so a disagreement about which
        # parameters are covered -- or their order, shape or dtype -- comes
        # back as a number rather than as an error.
        agree(digest(table, values=False), group,
              "which parameters this round covers")

        total = _total_weight(w, group)
        if not total > 0.0:
            raise ValueError(
                "torchnative.nn.federated.FedAvg: the ranks' weights sum to "
                "%r, so there is no average. Every rank that reported either "
                "abstained (federated.ABSTAIN) or weighed nothing -- a round "
                "over an empty cohort would otherwise divide by zero and "
                "return a table of the right names full of nan" % (total,)
            )

        out = {}
        for name in sorted(table):
            t = table[name]
            # Scaled by a 0-dim tensor of the table's own dtype rather than by
            # a Python float: tensor-tensor arithmetic has one dtype and one
            # rounding, where a Python scalar leaves the width of the
            # intermediate to a promotion rule. What is compared against the
            # centrally computed average has to be reproducible by anyone
            # writing the same expression, and this is the spelling that is.
            scaled = (t.detach() * torch.tensor(w, dtype=t.dtype)).clone()
            _check_wire(scaled, name)
            _all_reduce(scaled, group,
                        "the sum of the ranks' deltas for %r" % (name,))
            out[name] = scaled / torch.tensor(total, dtype=t.dtype)
        return out


class FedAvgM:
    """FedAvg with server momentum: ``v <- beta v + avg``, and the round applies ``eta v``.

    Hsu, Qi and Brown (2019).  The client half is unchanged -- every rank still
    contributes ``w_k^local - w_global`` and the group still forms the weighted
    mean of those.  What differs is what the *server* does with that mean: it
    accumulates it into a velocity that carries across rounds, so a direction
    every round agrees on is amplified and one that reverses is damped.

    **Why this is a real second aggregator and not FedAvg with a knob.** At
    ``momentum=0`` it is FedAvg, bit for bit, and that is asserted rather than
    claimed.  Above zero its output at round *k* depends on rounds *1..k-1*,
    which no FedAvg does -- so a single round cannot distinguish them and the
    test does not try: it runs three and checks each against
    ``v_k = beta v_(k-1) + mean_k`` computed centrally.

    **Every rank holds the same velocity without communicating it.** The mean
    is already identical on both ranks (``all_reduce`` leaves the same bits
    everywhere, docs/distributed/FEDERATED.md §2.1), and the update applied to it is
    deterministic, so the states cannot drift.  This is checked rather than
    argued: the test asserts the two ranks' outputs are equal at every round.
    Nothing here reduces the velocity itself -- a server state that needed a
    collective to stay in step would be a different design and would have to
    say so.

    The state is keyed on parameter name and refuses a schema change between
    rounds: a momentum buffer silently restarted at zero for a parameter that
    was renamed is the FedAvg answer reported as the FedAvgM one.
    """

    def __init__(self, momentum=0.9, server_lr=1.0, weighted=True):
        m = float(momentum)
        if m != m or m in (float("inf"), float("-inf")) or not 0.0 <= m < 1.0:
            raise ValueError(
                "torchnative.nn.federated.FedAvgM: momentum=%r. Has to be "
                "finite and in [0, 1). At 1 the velocity never decays and the "
                "server diverges; below 0 it alternates sign every round, and "
                "neither is a thing a caller means" % (momentum,)
            )
        lr = float(server_lr)
        if lr != lr or lr in (float("inf"), float("-inf")) or not lr > 0.0:
            raise ValueError(
                "torchnative.nn.federated.FedAvgM: server_lr=%r. Has to be "
                "finite and positive; at 0 every round installs the zero "
                "update and the model never moves while the round reports "
                "success" % (server_lr,)
            )
        self.momentum = m
        self.server_lr = lr
        self._avg = FedAvg(weighted=weighted)
        self.velocity = {}
        self.rounds_seen = 0

    @property
    def weighted(self):
        return self._avg.weighted

    def __repr__(self):
        return "FedAvgM(momentum=%r, server_lr=%r, weighted=%r)" % (
            self.momentum, self.server_lr, self.weighted)

    def resolve_weight(self, weight):
        """Delegated, so the Engine reports the number the mean was formed with."""
        return self._avg.resolve_weight(weight)

    def aggregate(self, table, weight=None, group=None):
        """The weighted mean, accumulated into the velocity. Returns a table.

        Every refusal :meth:`FedAvg.aggregate` makes is made first, because
        the mean is formed by that method and not by a copy of it.
        """
        mean = self._avg.aggregate(table, weight=weight, group=group)
        if self.velocity and set(self.velocity) != set(mean):
            raise ValueError(
                "torchnative.nn.federated.FedAvgM: this round covers %s and "
                "the velocity was built over %s. A momentum buffer that "
                "restarted at zero for a renamed parameter would return the "
                "FedAvg answer and call it FedAvgM"
                % (sorted(mean), sorted(self.velocity))
            )
        out = {}
        for name in sorted(mean):
            m = mean[name]
            beta = torch.tensor(self.momentum, dtype=m.dtype)
            eta = torch.tensor(self.server_lr, dtype=m.dtype)
            previous = self.velocity.get(name)
            v = m.clone() if previous is None else (previous * beta + m)
            self.velocity[name] = v
            out[name] = v * eta
        self.rounds_seen += 1
        return out


class FedProx:
    """FedAvg's server step with FedProx's *local* objective.

    Li et al. (2020).  The proximal term is ``mu/2 * ||w - w_global||^2`` added
    to what each client minimises, which on the gradient is
    ``mu * (w - w_global)`` -- and that is the whole of the difference.  **The
    aggregation is FedAvg's, unchanged**, which is the fact that makes a
    "FedProx aggregator" a trap: a class that only overrode ``aggregate``
    would compute FedAvg and be indistinguishable from it, at every world size,
    for ever.

    So this refuses to aggregate unless :meth:`arm` has installed the term on a
    real local loop -- :meth:`torchnative.adapt.Adapted.add_grad_hook`.
    ``Engine`` arms it; the low-level ``Delta.publish`` road does not, and
    there the refusal is the honest answer rather than a silent FedAvg.

    **What the term does, in a shape a test can catch.** At the first local
    step ``w == w_global``, so the term is exactly zero and the step is
    FedAvg's, bit for bit.  From the second step on it pulls back toward the
    round's starting weights, so the delta is *smaller* than the unregularised
    one -- both halves are asserted, and the first one is what shows the term
    is the stated function of ``w - w_global`` rather than a constant nudge.
    """

    def __init__(self, mu=0.01, weighted=True):
        m = float(mu)
        if m != m or m in (float("inf"), float("-inf")) or not m > 0.0:
            raise ValueError(
                "torchnative.nn.federated.FedProx: mu=%r. Has to be finite and "
                "positive -- FedProx at mu=0 *is* FedAvg, and a caller who "
                "means FedAvg should say FedAvg rather than reach it through a "
                "parameter that reads like a tuning knob" % (mu,)
            )
        self.mu = m
        self._avg = FedAvg(weighted=weighted)
        self._armed = None
        self._remove = None

    @property
    def weighted(self):
        return self._avg.weighted

    def __repr__(self):
        return "FedProx(mu=%r, weighted=%r)" % (self.mu, self.weighted)

    def resolve_weight(self, weight):
        return self._avg.resolve_weight(weight)

    def arm(self, adapted):
        """Install the proximal term on ``adapted``'s local step. Idempotent.

        The base is read at *call* time from the live delta, not captured here,
        so a round that re-snapshots (docs/distributed/FEDERATED2.md §1.1) gets the new
        global weights without re-arming: ``w_global`` is by definition the
        weights the round started from, which is exactly what the delta's base
        holds.
        """
        if self._armed is adapted:
            return self
        if self._armed is not None:
            raise RuntimeError(
                "torchnative.nn.federated.FedProx: this aggregator is already "
                "armed on another model. The proximal term is defined against "
                "one round's w_global; sharing an instance across two local "
                "loops would pull each toward the other's base"
            )
        mu = self.mu

        def proximal(wrapper, params, names):
            delta = wrapper.adapted
            base = delta.base
            for name in names:
                if name not in base:
                    continue
                p = params[name]
                if p.grad is None:
                    continue
                scale = torch.tensor(mu, dtype=p.grad.dtype)
                p.grad = p.grad + (p.detach() - base[name]) * scale

        self._remove = adapted.add_grad_hook(proximal)
        self._armed = adapted
        return self

    def disarm(self):
        """Remove the term again. Here so that arming is reversible in a test."""
        if self._remove is not None:
            self._remove()
        self._remove = None
        self._armed = None
        return self

    def aggregate(self, table, weight=None, group=None):
        """FedAvg's weighted mean -- but only if the local term was installed.

        A FedProx that never armed is FedAvg wearing a different name, and it
        would report success at every world size. docs/design/DESIGN.md §6.
        """
        if self._armed is None:
            raise RuntimeError(
                "torchnative.nn.federated.FedProx: the proximal term was never "
                "installed, so this would compute FedAvg's weighted mean and "
                "report it as FedProx. The difference between the two is "
                "entirely in the local objective -- mu*(w - w_global) on the "
                "gradient -- and the server step is identical, so nothing "
                "downstream could tell them apart.\n"
                "Check: federated.Engine(model, method=..., "
                "aggregator=FedProx(mu=...)) arms it, or call "
                "FedProx.arm(adapt.wrap(...)) before the local steps."
            )
        return self._avg.aggregate(table, weight=weight, group=group)


def cohort(select, group=None, who="federated.cohort"):
    """The ranks this round runs over, agreed across the group. Returns a tuple.

    ``select`` is a sequence of ranks or a callable taking the world size and
    returning one.  Every rank evaluates it and the results are compared over
    the same digest collective the schema and base use, because **selection
    fails silently in exactly the way they do**: two ranks that disagree about
    who is participating still complete every collective and still produce a
    weighted mean, over a cohort neither of them chose.  A rule as ordinary as
    "sample half the clients at random" disagrees whenever the ranks seed
    differently, which is the default.

    **A proper subset is now served, and not with ``new_group``.**  Every rank
    still attends every collective -- the transport is a star and a rank that
    walked away would hang the ones that did not -- and the ranks outside the
    cohort contribute ``ABSTAIN``: a zero table at weight zero.  The divisor is
    then the cohort's weight and the result is the cohort's weighted mean,
    exactly.  What ``torch.distributed.new_group`` would add on top is that the
    unselected ranks are not *reached* at all, which is a property of the wire
    and not of the arithmetic; it is named in docs/distributed/FEDERATED4.md as still
    missing, because a client that is asleep cannot attend a barrier.

    A cohort of **one** still refuses, at any world size, and so does any
    proper subset in a world of two: FedAvg over one delta is that delta, and
    a test of it would pass with no aggregation at all.
    """
    world, rank = _require_world(group, who)
    proposal = select(world) if callable(select) else select
    try:
        ranks = tuple(sorted({int(r) for r in proposal}))
    except (TypeError, ValueError):
        raise TypeError(
            "torchnative.nn.federated.cohort: select= produced %r, which is "
            "not a sequence of rank numbers" % (proposal,)
        )
    if not ranks:
        raise ValueError(
            "torchnative.nn.federated.cohort: the empty cohort. A round with "
            "no participants would complete, aggregate nothing and report "
            "success"
        )
    bad = [r for r in ranks if not 0 <= r < world]
    if bad:
        raise ValueError(
            "torchnative.nn.federated.cohort: rank(s) %s are not in a world of "
            "%d" % (bad, world)
        )

    # Agreed before it is acted on, and before the local epochs: two ranks
    # holding different cohorts is not detectable downstream.
    agree(digest({"cohort": torch.tensor(ranks, dtype=torch.float64)}),
          group,
          "which ranks this round selected (rank %d proposed %s)"
          % (rank, list(ranks)))

    # One refusal, not two. There was a second here -- "a proper subset needs
    # a world of at least three" -- and it was **unreachable**: in a world of
    # two the only proper subsets have one member, so the check above had
    # already refused every input that could reach it. A refusal that cannot
    # fire is not a refusal, so its reason was folded into the one that does
    # (docs/distributed/FEDERATED4.md §5).
    if len(ranks) < 2:
        raise NotImplementedError(
            "torchnative.nn.federated.cohort: %s of a world of %d is a cohort "
            "of %d. FedAvg over one delta is that delta -- the identity this "
            "whole package refuses to serve -- so a round over it would "
            "return the selected rank's own weights and report success, and a "
            "test of it would pass with no aggregation at all.\n"
            "In a world of two this is the only shape a proper subset can "
            "have, which is why a proper subset needs a world of at least "
            "three. The transport carries one now (docs/distributed/FEDERATED4.md), so "
            "what refuses here is the arithmetic and no longer the wire: a "
            "subset of a world of three is served, and aggregates over the "
            "subset.\n"
            "Check: len(select(world)) >= 2, in a world of at least three if "
            "the cohort is not the whole world."
            % (list(ranks), world, len(ranks))
        )
    # A rank outside the cohort is neither an error nor excused from the
    # collectives: it attends and abstains. See `_Abstain`.
    return ranks


class Round:
    """What one round did, as an object rather than as a printed line.

    Held so a caller can assert on the round instead of on the model that came
    out of it: ``weight`` is this rank's, ``total_weight`` the group's, and
    ``share`` the fraction of the aggregate this rank contributed -- the number
    that is exactly 0.5 when the round was unweighted.
    """

    def __init__(self, rank, world, weight, total_weight, steps, history,
                 covers, local_norm, aggregate_norm, cohort=None, missing=(),
                 participated=True):
        #: The ranks that did not report. Empty under the default
        #: `on_missing='refuse'`, where a drop raises instead of being
        #: reported; non-empty only under `'average_arrived'`, where it is the
        #: record of who the divisor left out.
        self.missing = tuple(missing)
        #: Whether *this* rank contributed. False for a rank the cohort left
        #: out, which still attended every collective at weight zero.
        self.participated = bool(participated)
        #: The ranks this round ran over. Equal to every rank of the world --
        #: `cohort()` refuses a proper subset -- but recorded rather than
        #: assumed, so a report says what it aggregated and not what it hoped.
        self.cohort = tuple(range(world)) if cohort is None else tuple(cohort)
        self.rank = rank
        self.world = world
        self.weight = weight
        self.total_weight = total_weight
        self.steps = steps
        self.history = list(history)
        self.covers = tuple(covers)
        self.local_norm = local_norm
        self.aggregate_norm = aggregate_norm

    @property
    def share(self):
        return self.weight / self.total_weight

    def __repr__(self):
        return ("<Round rank %d/%d, %d step(s) over %d parameters, "
                "weight %g of %g, |local|=%.4g |aggregate|=%.4g>"
                % (self.rank, self.world, self.steps, len(self.covers),
                   self.weight, self.total_weight,
                   self.local_norm, self.aggregate_norm))


class Engine:
    """Federated averaging engine: N rounds of adapt, aggregate, re-snapshot.

        engine = federated.Engine(model, method=adapt.Tent(), lr=1e-3,
                                  aggregator=federated.FedAvg(), rounds=3)
        reports = engine.participate(batches, weight=len(local_dataset))

    The local half is ``torchnative.adapt`` unchanged -- ``wrap`` a method
    round the model, go ``online``, take a step per batch -- so the delta this
    contributes is produced by the same machinery a device uses when it is not
    federated at all. That is DESIGN.md §3's claim made operational: the
    federated destination is a *destination* for a delta, not a second kind of
    delta.

    **``participate`` takes the local data.** README §2 sketches it with no
    arguments; it does not have that shape here, because a round with no
    arguments would have to invent either the batches or the sample count, and
    both change the result. The rest of the sketch stands.

    **Rounds > 1.** Each round adapts locally, publishes the delta, and
    installs ``base + aggregate`` into the model. Before the next round, the
    delta's base is **re-snapshotted** to the current (aggregated) weights, so
    that round *k+1*'s delta measures only the new local training, not
    cumulative movement from round 1. Without this, ``online()`` leaves the
    base at the value captured at construction, and every later round would
    re-send round 1's offset -- the model diverges by compounding.

    **Optimiser state.** The optimiser is **not** reset between rounds.
    Momentum and variance from the previous round carry into the next.
    Whether that matters depends on the optimiser and the task; it is named
    here rather than silently decided (DESIGN.md §6).
    """

    #: The dropout policies this Engine knows. Only the first is implemented.
    ON_MISSING = ("refuse", "average_arrived")

    def __init__(self, model, method=None, aggregator=None, rounds=1,
                 group=None, lr=1e-3, optimizer=None, select=None,
                 allow_missing=False, on_missing="refuse",
                 min_participants=None,
                 secure_aggregation=False, differential_privacy=None,
                 **optimizer_kwargs):
        if method is None:
            raise TypeError(
                "torchnative.nn.federated.Engine: method= is required. The "
                "local half of a round is an adaptation method, and "
                "torchnative.adapt.wrap refuses a default for the same reason: "
                "the choice decides which parameters move and what is minimised"
            )
        # `select=` is evaluated in `participate`, not here: it needs the
        # world size, and the cohort has to be *agreed across the ranks*, which
        # is a collective. What it cannot do is a proper subset -- see
        # `cohort()`, which refuses that by name and says what it would take.
        self.select = select
        if allow_missing:
            raise NotImplementedError(
                "torchnative.nn.federated.Engine: allow_missing= is dropout "
                "handling by another name. Averaging over whichever ranks "
                "arrived makes the divisor a number nobody chose, and the "
                "round reports success -- docs/design/DESIGN.md §6 puts that below a "
                "refusal.\n"
                "The policy surface is on_missing=. 'refuse' is the default "
                "and is implemented: a rank that does not report raises "
                "federated.RankDropped, naming which rank, and the round is "
                "undone rather than left half-applied -- the collective "
                "raises and it never produces a partial average. "
                "on_missing='average_arrived' is what allow_missing=True "
                "meant, and it refuses."
            )
        if on_missing not in self.ON_MISSING:
            raise ValueError(
                "torchnative.nn.federated.Engine: on_missing=%r. One of %s"
                % (on_missing, list(self.ON_MISSING))
            )
        # `min_participants` is what turns "however many arrived" into a
        # number the caller chose. Without it this policy is exactly the
        # defect `RankDropped` exists to refuse -- a weighted mean over a
        # cohort decided by a socket close, reported as success.
        if on_missing == "average_arrived":
            if min_participants is None:
                raise TypeError(
                    "torchnative.nn.federated.Engine: "
                    "on_missing='average_arrived' requires "
                    "min_participants=k. Averaging over whoever arrived, with "
                    "no floor, makes the divisor a number nobody chose: the "
                    "round returns a weighted mean over a cohort decided by a "
                    "socket close and reports success (docs/design/DESIGN.md §6). "
                    "The floor is the caller saying how few ranks an aggregate "
                    "may still be built from.\n"
                    "Check: min_participants=2 or more."
                )
            if not isinstance(min_participants, int) or min_participants < 2:
                raise ValueError(
                    "torchnative.nn.federated.Engine: min_participants=%r. It "
                    "has to be an integer of at least 2 -- a survivor set of "
                    "one is a world of one, where FedAvg returns the "
                    "survivor's own delta and a round over it would report "
                    "success having aggregated nothing (docs/distributed/FEDERATED3.md "
                    "§4.1 measured that identity to 6e-8)."
                    % (min_participants,)
                )
        elif min_participants is not None:
            raise TypeError(
                "torchnative.nn.federated.Engine: min_participants=%r with "
                "on_missing=%r. A floor on the survivor set only means "
                "something under 'average_arrived'; with 'refuse' the round "
                "needs every rank and this would be accepted and ignored"
                % (min_participants, on_missing)
            )
        self.min_participants = min_participants
        if secure_aggregation:
            raise NotImplementedError(
                "torchnative.nn.federated.Engine: secure_aggregation= is not "
                "implemented, and it is not one flag's worth of work. Masked "
                "aggregation (Bonawitz et al. 2017) needs pairwise secrets "
                "between clients, which needs point-to-point send/recv -- "
                "ProcessGroupLocal refuses both by name (docs/distributed/TRANSPORT.md §3) "
                "-- plus a key agreement, a threshold secret-sharing scheme so "
                "a dropout does not destroy the sum, and an unmasking round. "
                "The dropout half is the same problem on_missing= names.\n"
                "The digest in federated.agree() detects an accident, not an "
                "adversary: 56 bits, and nothing here is trying to stop two "
                "ranks that want to collide it."
            )
        if differential_privacy is not None:
            raise NotImplementedError(
                "torchnative.nn.federated.Engine: differential_privacy= is not "
                "implemented. A DP guarantee is a *number* -- (epsilon, delta) "
                "at a stated granularity -- and producing one needs per-example "
                "gradient clipping inside the local step, noise calibrated to "
                "the clipping norm, and an accountant over the rounds. Two of "
                "those do not exist here: this stack's backward produces one "
                "gradient per parameter over the batch, not per example, and "
                "the sampling rate an accountant integrates over is set by "
                "participant selection, which cohort() refuses above.\n"
                "Adding a noise term without those would produce a model that "
                "is worse and a guarantee that is absent, and report both as "
                "success. It is a round of its own, after world_size N."
            )
        if not isinstance(rounds, int) or rounds < 1:
            raise ValueError(
                "torchnative.nn.federated.Engine: rounds=%r. Must be a "
                "positive integer." % (rounds,)
            )
        from torchnative import adapt

        self.model = model
        self.method = method
        self.aggregator = aggregator if aggregator is not None else FedAvg()
        self.rounds = rounds
        self.group = group
        self.on_missing = on_missing
        self.adapted = adapt.wrap(model, method=method, optimizer=optimizer,
                                  lr=lr, **optimizer_kwargs)
        # An aggregator whose difference from FedAvg lives in the *local*
        # objective gets to install it here. FedProx is the one; it refuses to
        # aggregate at all if this never happened, so the wiring cannot be
        # forgotten silently.
        if hasattr(self.aggregator, "arm"):
            self.aggregator.arm(self.adapted)
        self._participated = False

    def __repr__(self):
        return "<Engine method=%s aggregator=%r rounds=%d>" % (
            type(self.method).__name__, self.aggregator, self.rounds)

    def participate(self, batches, weight=None, epochs=1):
        """Run ``self.rounds`` federated rounds over the same local data.

        Each round: adapt locally (``epochs`` passes over ``batches``),
        publish the delta, install ``base + aggregate``, re-snapshot.  The
        model ends holding the weights from the *last* round's aggregate.

        ``batches`` is an iterable of what the model is called with: a mapping
        goes as keyword arguments, a tuple as positional, anything else as one
        positional argument. One adaptation step per batch per epoch.

        Returns a **list** of :class:`Round` objects, one per round.  For
        backwards compatibility with code that expected a single ``Round``,
        ``rounds=1`` still returns a list of length 1 -- callers that used
        ``report = engine.participate(...)`` should use
        ``[report] = engine.participate(...)`` or ``reports[-1]``.
        """
        if self._participated:
            raise RuntimeError(
                "torchnative.nn.federated.Engine.participate: called twice. "
                "A second call would open no new delta and would contribute "
                "this rank's previous offset again"
            )
        batches = list(batches)
        if not batches:
            raise ValueError(
                "torchnative.nn.federated.Engine.participate: no batches. A "
                "round with no local data contributes the zero delta and would "
                "still be counted, at full weight, in the average"
            )
        if epochs < 1:
            raise ValueError(
                "torchnative.nn.federated.Engine.participate: epochs=%r"
                % (epochs,)
            )

        # Both checks run before the local epochs rather than after: a world of
        # one, or a missing weight, should refuse before the model has been
        # moved -- not after a round of training that then cannot be
        # contributed and cannot be undone without a revert the caller did not
        # ask for.
        world, rank = _require_world(self.group, "Engine.participate")
        if self.on_missing == "average_arrived":
            # Checked here rather than in `__init__`: the world is not known
            # until there is a group, and this is the first moment there is.
            if world < 3:
                raise NotImplementedError(
                    "torchnative.nn.federated.Engine: "
                    "on_missing='average_arrived' in a world of %d. One rank "
                    "leaving leaves %d, and FedAvg over one delta is that "
                    "delta -- the partial average would be the survivor's own "
                    "weights (docs/distributed/FEDERATED3.md §4.1 measured that identity "
                    "to 6e-8), so this policy cannot be shown to do anything "
                    "here.\n"
                    "Check: init_process_group(..., world_size=3) or more."
                    % (world, world - 1))
            if self.min_participants > world:
                raise ValueError(
                    "torchnative.nn.federated.Engine: min_participants=%d in "
                    "a world of %d, so no round could ever meet it"
                    % (self.min_participants, world))
        # Agreed before the local epochs, for the same reason the world size
        # is: a cohort the ranks disagree about should refuse before the model
        # has moved, not after a round that cannot be contributed.
        ranks = (tuple(range(world)) if self.select is None
                 else cohort(self.select, self.group, "Engine.participate"))
        # Resolved before the local epochs, not after: a missing or nonsensical
        # weight should refuse before the model has been moved, not after a
        # round of training that then cannot be contributed.
        # A rank the cohort left out still attends every collective and
        # contributes nothing -- see `_Abstain`. Its weight is resolved from
        # ABSTAIN rather than from what the caller passed, so a caller who
        # passed a sample count does not have it counted in a round it was not
        # selected for.
        participating = rank in ranks
        contributed = weight if participating else ABSTAIN
        resolved = (self.aggregator.resolve_weight(contributed)
                    if hasattr(self.aggregator, "resolve_weight")
                    else (1.0 if contributed is None else 0.0
                          if contributed is ABSTAIN else float(contributed)))

        reports = []
        try:
            self._rounds(batches, contributed, epochs, resolved, world, rank,
                         ranks, reports, participating)
        except RankDropped:
            # Policy `on_missing='refuse'`: the round is *undone*. The local
            # epochs have already moved the model, and leaving it there would
            # keep an update no other rank has -- the two ranks would silently
            # stop holding the same weights, which is the one property that
            # makes this federated learning rather than two devices training
            # alone. `revert` restores the base byte for byte, and the base is
            # the last aggregate every rank agreed on.
            self.adapted.revert()
            self._participated = True
            raise
        self._participated = True
        return reports

    def _rounds(self, batches, weight, epochs, resolved, world, rank, ranks,
                reports, participating=True):
        """The round loop. Split out so `participate` owns the drop policy."""
        for round_idx in range(self.rounds):
            self.adapted.online()
            steps = 0
            if participating:
                for _ in range(epochs):
                    for batch in batches:
                        if isinstance(batch, dict):
                            self.adapted.step(**batch)
                        elif isinstance(batch, tuple):
                            self.adapted.step(*batch)
                        else:
                            self.adapted.step(batch)
                        steps += 1
            else:
                # Not selected: no local epochs at all. This is what
                # participant selection *is* -- a client that was not picked
                # does not train. The delta is recorded anyway, against a model
                # that has not moved, so it is the zero offset and it goes on
                # the wire at weight zero. It has to go: the ranks that were
                # picked are inside the same collectives.
                self.adapted.adapted.record(self.model)

            delta = self.adapted.adapted
            local_norm = delta.norm()
            missing = ()
            if self.on_missing == "average_arrived":
                with tolerate_missing(self.group) as arrival:
                    table = delta.publish(group=self.group, weight=weight,
                                          aggregator=self.aggregator)
                    total = _total_weight(resolved, self.group)
                    missing = tuple(arrival.missing or ())
                    survivors = len(arrival.survivors)
                if survivors < self.min_participants:
                    raise RankDropped(
                        "torchnative.nn.federated: rank(s) %s did not report, "
                        "leaving %d of %d, and this Engine was built with "
                        "min_participants=%d. The aggregate is refused rather "
                        "than divided by whoever was left: below the floor the "
                        "caller chose, the 'partial average' stops being an "
                        "average of anything they asked for."
                        % (list(missing), survivors, world,
                           self.min_participants),
                        rank=rank, missing=missing, during="the round's "
                        "collectives")
            else:
                table = delta.publish(group=self.group, weight=weight,
                                      aggregator=self.aggregator)
                total = _total_weight(resolved, self.group)

            # Install the aggregate: base + aggregated_delta.
            delta.value = dict(table)
            delta.apply(self.model)

            reports.append(Round(
                rank=rank, world=world, weight=resolved, total_weight=total,
                steps=steps, history=self.adapted.history, covers=delta.covers,
                local_norm=local_norm, aggregate_norm=delta.norm(),
                cohort=ranks, missing=missing, participated=participating,
            ))

            # Re-snapshot: the next round's delta must be measured against the
            # aggregated weights, not against the original base. Without this,
            # round k+1 re-sends round 1's movement and the model diverges.
            if round_idx < self.rounds - 1:
                delta.re_snapshot(self.model)
