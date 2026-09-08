"""Weight deltas over base weights.

Every adaptation method produces one of these; they differ only in lifetime
and destination. The boundary is ``model(x)`` -- state a model manages inside
its own forward (fast weights, caches) is the model's, not ours.

    from torchnative.delta import Delta

    d = Delta.over(model, names)     # snapshot the base of exactly those names
    ...                              # something writes into the parameters
    d.record(model)                  # the offset from base is now an object
    d.revert(model)                  # the base is back, bit for bit
    d.apply(model)                   # the offset is back on top of it

**Why the base is held per delta rather than per model.** DESIGN.md §9 item 5
records that ``ttadapters`` keeps a full weight copy in ``base_state`` so that
``AdaptationEngine.reset()`` has something to restore. A delta narrows that: it
holds the base of the parameters *it covers*, which for a method that adapts
normalisation affine parameters is four orders of magnitude smaller than the
model. docs/models/ADAPT.md §4 has both numbers on a real checkpoint.

**Why a base copy at all, rather than subtracting the offset back off.**
Because ``(w + d) - d`` is not ``w`` in floating point, and "reverted" has to
mean the base weights are the bytes they were. Subtraction is offered as
:meth:`Delta.revert_by_subtraction` precisely so the two can be compared --
docs/models/ADAPT.md §5 measures how far apart they land on a real checkpoint. If they
were equal the copy would be waste; they are not.

**Lifetime is the axis DESIGN.md §3 deliberately left unnamed.** It withdrew
one invented set of names and one borrowed set, and said the names get chosen
once a real integration shows the usage. What a lifetime policy has to answer
is three questions, and this type answers them by *running* rather than by
carrying a label:

======================================================  ========================
can this delta be discarded, and at what cost           :meth:`Delta.revert`,
                                                        :attr:`Delta.nbytes`
does it survive a process restart                       :meth:`Delta.persist`,
                                                        :meth:`Delta.load`
can it leave the device                                 :meth:`Delta.publish`
======================================================  ========================

**All three are answered now.** ``persist``/``load`` write and read a delta as
safetensors and the round trip is bit-identical; ``publish`` sends it to the
other ranks of a process group and returns the aggregate.

``publish`` refused until 2026-09-02, and it refused with a check the reader
could run rather than with a claim about the world -- ``init_process_group(
backend='local', ..., world_size=2)``. That check started returning
(docs/distributed/TRANSPORT.md) and this method was written on top of it (docs/distributed/FEDERATED.md).
The two refusals that remain are narrower and are about *this* call rather than
about the world: an unrecorded delta has nothing to send, and a process group
that was never initialised has nobody to send to.

``persist`` deliberately does **not** go through ``torch.save``, and the
reason is no longer the one BACKWARD.md §14 gave. That section said the blocker
was a storage object aliasing its tensor, which this stack could not honestly
fake; docs/models/SAVE.md §2 found the danger was real and the remedy wrong -- what has
to refuse is the *write*, not the aliasing -- and ``torch.save`` now works.

What remains is a choice of format rather than a wall: safetensors is a flat,
mmap-able, pickle-free container, and a delta on the wire is exactly the payload
where not executing arbitrary pickle on arrival is worth something.
"""

from __future__ import annotations

__all__ = ["Delta"]


class Delta:
    """The offset of a set of named parameters from the base they were taken over.

    A delta owns two dictionaries keyed on parameter name: ``base``, the values
    the covered parameters held when the delta was opened, and ``value``, the
    offset recorded on top of them. ``value`` is ``None`` until
    :meth:`record` -- an unrecorded delta is the zero delta and costs one copy
    of the covered parameters rather than two.

    It does *not* hold the model. A delta that needed the object it came from
    could not be the thing federated averaging sends, and DESIGN.md §3's whole
    claim is that those two are the same object with different destinations.
    """

    def __init__(self, base, value=None):
        self.base = dict(base)
        self.value = None if value is None else dict(value)

    # -- construction ------------------------------------------------------

    @classmethod
    def over(cls, model, names):
        """Open a delta over the named parameters of ``model``.

        ``names`` is checked against the model rather than trusted: a name that
        does not resolve is a typo that would otherwise produce a delta silently
        covering less than it claims, which is the failure mode this whole file
        is arranged against.
        """
        params = dict(model.named_parameters())
        names = list(names)
        missing = [n for n in names if n not in params]
        if missing:
            raise KeyError(
                "torchnative.delta: %d of %d names are not parameters of this "
                "model, first is %r" % (len(missing), len(names), missing[0])
            )
        if not names:
            raise ValueError(
                "torchnative.delta: a delta over no parameters. A method that "
                "selects nothing adapts nothing, and would report every step as "
                "having run"
            )
        return cls({n: params[n].detach().clone() for n in names})

    # -- what it covers ----------------------------------------------------

    @property
    def covers(self):
        """The parameter names this delta is over, in the order it was opened."""
        return tuple(self.base)

    def __len__(self):
        return len(self.base)

    @property
    def nbytes(self):
        """``(base_bytes, value_bytes)`` -- what holding this delta costs.

        Counted through ``element_size()`` rather than assumed from the dtype,
        for the reason ``torchnative.quant.storage_bytes`` gives.
        """
        base = sum(t.numel() * t.element_size() for t in self.base.values())
        val = 0 if self.value is None else sum(
            t.numel() * t.element_size() for t in self.value.values()
        )
        return base, val

    def norm(self):
        """The L2 norm of the whole delta, as one number.

        Zero for a delta that has not been recorded, and zero is the honest
        answer there: an unrecorded delta is the zero offset.
        """
        if self.value is None:
            return 0.0
        total = 0.0
        for t in self.value.values():
            total += float((t * t).sum().item())
        return total ** 0.5

    def max_abs(self):
        """The largest single component of the delta."""
        if self.value is None:
            return 0.0
        return max(float(t.abs().max().item()) for t in self.value.values())

    # -- the three operations ---------------------------------------------

    def record(self, model):
        """Read the covered parameters and store their offset from base.

        Called after something has written into the parameters -- an optimiser
        step, a hand-applied update -- to turn "the model is different now" into
        an object that can be kept, measured, reverted and (one day) sent.
        """
        params = dict(model.named_parameters())
        self.value = {}
        for name, base in self.base.items():
            self.value[name] = (params[name].detach() - base).clone()
        return self

    def revert(self, model):
        """Put the base weights back, byte for byte.

        A copy, not a subtraction. See the module docstring, and
        :meth:`revert_by_subtraction` for the comparison that makes the choice
        a measurement rather than an assertion.

        The delta itself is *kept*. Reverting is not discarding: the whole point
        of holding the offset separately is that it can go back on with
        :meth:`apply`.
        """
        params = dict(model.named_parameters())
        for name, base in self.base.items():
            params[name].data.copy_(base)
        return self

    def apply(self, model):
        """Put ``base + value`` into the covered parameters.

        A no-op for an unrecorded delta, because an unrecorded delta is zero.

        This is *not* guaranteed to reproduce the weights :meth:`record` read:
        ``base + (w - base)`` is not ``w`` in floating point. docs/models/ADAPT.md §5
        measures the gap; it is what "keeping a delta" costs against "keeping
        the weights".
        """
        if self.value is None:
            return self
        params = dict(model.named_parameters())
        for name, base in self.base.items():
            params[name].data.copy_(base + self.value[name])
        return self

    def revert_by_subtraction(self, model):
        """Undo the delta by subtracting it instead of restoring the base.

        Here so that "the base copy is necessary" is a number rather than an
        argument. It is not what :meth:`revert` does.
        """
        if self.value is None:
            return self
        params = dict(model.named_parameters())
        for name in self.base:
            p = params[name]
            p.data.copy_(p.detach() - self.value[name])
        return self

    def zero(self):
        """Discard the offset, keeping the base. The delta is the zero delta again."""
        self.value = None
        return self

    def re_snapshot(self, model):
        """Replace the base with a copy of the current live weights and zero the value.

        After this call, ``delta()`` measures movement from the new base.  This
        is what multi-round federated learning needs: after round *k* installs
        the aggregate, round *k+1*'s delta must reflect only the new local
        training, not cumulative movement from round 1.

        Without this, ``online()`` puts the wrapper back into training mode but
        does **not** take a new snapshot -- the base remains the one captured at
        construction -- so every later round would re-send round 1's movement
        and the model would diverge by compounding.

        The optimiser state is **not** reset.  Momentum / variance from the
        previous round carry into the next.  Whether that matters depends on the
        optimiser and the task; it is named here rather than silently decided
        either way (DESIGN.md §6).
        """
        params = dict(model.named_parameters())
        self.base = {n: params[n].detach().clone() for n in self.base}
        self.value = None
        return self

    # -- persistence -------------------------------------------------------
    #
    # `torch.save` is **not** what this uses, and that is a finding rather than
    # a convenience. docs/training/BACKWARD.md §14 sizes it: two real items and five
    # one-liners, and the substantial one -- `Tensor.untyped_storage()` -- is a
    # semantic problem and not a kernel. Upstream a storage *aliases* its
    # tensor; here a storage is a byte buffer that `set_` copies out of
    # (`storage.rs`'s module docstring), so `untyped_storage()` on this stack
    # would hand back a copy that silently does not write through.
    # `torch.save` would not notice, because it only reads -- and every other
    # caller would.
    #
    # What a delta on the wire actually needs is "tensor -> little-endian
    # bytes" and a container, and safetensors is a container that is a JSON
    # header and a concatenated blob: no pickle, no zip, no storage object. The
    # reading side already works on this stack (docs/models/CKPT.md §1), so a delta
    # written here is read back by the same library upstream reads it with.

    #: safetensors dtype names. Only the dtypes a delta can hold -- a covered
    #: parameter is floating point by construction, because `wrt_set` in
    #: tape.rs drops non-floating constants from the gradient path.
    _ST_NAME = {"torch.float32": "F32", "torch.float64": "F64",
                "torch.float16": "F16", "torch.bfloat16": "BF16"}
    #: struct codes for the dtypes that have one. The reduced-precision two do
    #: not and are written through their bit pattern below.
    _ST_PACK = {"F32": "f", "F64": "d"}

    @staticmethod
    def _flat(values):
        out = []

        def walk(v):
            if isinstance(v, list):
                for item in v:
                    walk(item)
            else:
                out.append(v)

        walk(values)
        return out

    @classmethod
    def _bytes(cls, tensor):
        """One tensor as little-endian bytes.

        Through ``tolist()``, which is the only way out of this shim that is
        public -- ``numpy``, ``data_ptr`` and ``untyped_storage`` all refuse --
        and is what docs/numerics/SEQLEN.md's logits sha256 has always used. It is a
        conversion rather than a memcpy, and that is stated rather than hidden:
        it costs a Python float per element, so this is a road for a *delta*
        (137 KiB on a Tent-adapted SmolLM2, docs/models/ADAPT.md §5.2) and not for a
        checkpoint.
        """
        import struct

        key = str(tensor.dtype)
        name = cls._ST_NAME.get(key)
        if name is None:
            raise NotImplementedError(
                "torchnative.delta: no safetensors dtype for %s; a delta holds "
                "floating parameters and this is not one of %s"
                % (key, sorted(cls._ST_NAME))
            )
        flat = cls._flat(tensor.tolist())
        code = cls._ST_PACK.get(name)
        if code is not None:
            return name, struct.pack("<%d%s" % (len(flat), code), *flat)
        # `float16`/`bfloat16`: the value came out of a tensor that already
        # holds it at that precision, so `tolist()` is exact and the only
        # question left is the bit layout. `bfloat16` is the top 16 bits of the
        # `float32` pattern; `float16` has a struct code of its own.
        if name == "BF16":
            return name, b"".join(struct.pack("<f", v)[2:] for v in flat)
        return name, struct.pack("<%de" % len(flat), *flat)

    def persist(self, path, what="value"):
        """Write this delta where it survives the process, and return the path.

        ``what`` selects ``"value"`` -- the offset, which is what a send would
        carry -- or ``"base"``, which is what a revert would need. The format
        is safetensors, for the reason the comment above gives; :meth:`load`
        reads it back and docs/training/BACKWARD.md §14.4 measures that round trip as
        bit-identical on a real adapted checkpoint.
        """
        import json
        import struct

        if what not in ("value", "base"):
            raise ValueError(
                "torchnative.delta: persist(what=%r) -- 'value' (the offset) "
                "or 'base' (what a revert needs)" % (what,)
            )
        table = getattr(self, what)
        if table is None:
            raise ValueError(
                "torchnative.delta: this delta has no offset to write; call "
                "record(model) first, or persist(what='base').\n"
                "Check: delta.value is None."
            )
        header, blob, offset = {}, bytearray(), 0
        for name in sorted(table):
            dtype, raw = self._bytes(table[name])
            header[name] = {"dtype": dtype,
                            "shape": list(table[name].shape),
                            "data_offsets": [offset, offset + len(raw)]}
            blob += raw
            offset += len(raw)
        # safetensors wants the header padded to an 8-byte boundary.
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * ((8 - len(encoded) % 8) % 8)
        with open(path, "wb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            handle.write(bytes(blob))
        return path

    @staticmethod
    def load(path):
        """``{name: tensor}`` from a file :meth:`persist` wrote.

        Deliberately **not** a ``Delta``: what comes off the wire is a table of
        tensors, and which model it belongs on -- and whether it is a base or
        an offset -- is the caller's knowledge and not the file's. Federated
        aggregation wants the table, not a reconstructed object.
        """
        from safetensors.torch import load_file

        return load_file(path, device="cpu")

    def publish(self, group=None, weight=None, aggregator=None):
        """Send this delta to the other ranks and return the aggregate.

        This is the federated destination -- DESIGN.md §3's second axis, the one
        place where adaptation state leaves the device. It returns a
        ``{name: tensor}`` table, not a ``Delta``, for the reason :meth:`load`
        gives: what comes back off the wire is the *group's* offset, and which
        model it belongs on is the caller's knowledge.

        It refused until 2026-09-02, and what it was waiting for was a second
        rank rather than a serialiser -- docs/models/SAVE.md §7 sized the three walls
        and this was the last of them. docs/distributed/TRANSPORT.md opened it.

        **What this checks before it sends.** FedAvg averages *offsets from a
        common base*: ``sum(w_k (w_k^local - w_global)) / sum(w_k)`` only means
        anything if every rank subtracted the same ``w_global``. Nothing
        downstream notices if they did not -- ``all_reduce`` sums whatever it
        is handed -- so the bases are compared here, by digest, over the same
        collective the aggregation runs on. It costs one ``int64`` element.

        Pass ``over_common_base=False``... there is no such argument, and that
        is deliberate: an opt-out would exist to be used by the first caller
        whose bases did not match.
        """
        if self.value is None:
            raise ValueError(
                "torchnative.delta: this delta has nothing to publish -- it is "
                "the zero offset, so the round would contribute nothing and "
                "still be counted at full weight in the average.\n"
                "Check: delta.value is None. Call record(model) after the "
                "local step."
            )
        from torchnative.nn import federated

        if aggregator is None:
            aggregator = federated.FedAvg()
        # Named here rather than inherited from `agree` below, so that a world
        # of one is reported against the call the caller made.
        federated._require_two(group, "Delta.publish")
        federated.agree(
            federated.digest(self.base),
            group,
            "the base these deltas are offsets from (FedAvg averages offsets "
            "from a common model, and averaging offsets from two different "
            "models is arithmetic on incomparable quantities)",
        )
        return aggregator.aggregate(self.value, weight=weight, group=group)

    def __repr__(self):
        base, val = self.nbytes
        state = "zero" if self.value is None else "|d|=%.4g" % self.norm()
        return "<Delta over %d parameters, %s, base %d B, value %d B>" % (
            len(self.base), state, base, val,
        )
