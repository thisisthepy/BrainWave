"""The `local` backend -- this project's `torch.distributed` transport.

DESIGN.md §11.1 puts four layers in a stack::

    torchnative.nn.federated   rounds, client selection, aggregation
      └ torch.distributed      ProcessGroup, collectives
          └ backends           ours, registered with register_backend
              └ devices        CPU, Metal, Vulkan, NPU

This module is the third layer's registration. The collectives themselves live
in ``torch._C._distributed_c10d.ProcessGroupLocal`` -- upstream builds its
backends in C++ and this project replaces that half, so that is where they
belong. What has to happen in Python is the *registration*, because
``Backend.register_backend`` is an API of ``torch.distributed.distributed_c10d``
and ``_C`` is imported before that file exists.

Why here rather than in the vendored tree: DESIGN.md §1 forbids a facade, and
IMPORT_TORCH.md records that the vendored tree is not edited. Registering from
``torchnative`` uses the extension point upstream published for exactly this.

**What this backend is, and is not.** This paragraph used to say it was "the
honest world_size-1 case, not a simulation of a larger one", and that it has
not been for some time --- it is corrected here rather than deleted, because a
module docstring understating its own module is the direction a reader cannot
check. ``ProcessGroupLocal`` now runs a **real world of three or more** over
loopback TCP in a star with the hub at rank 0, folding contributions in
ascending rank order so the answer is a property of the ranks and not of who
arrived first. **Eleven collectives** run and agree with upstream gloo at world
3 and 4 --- ``broadcast``, ``all_gather``, ``all_gather_into_tensor``,
``gather``, ``scatter``, ``reduce``, ``reduce_scatter``,
``reduce_scatter_tensor``, ``all_to_all``, ``all_to_all_single``, ``barrier``
--- with ``SUM``/``MIN``/``MAX``/``PRODUCT``/``AVG``
(``docs/distributed/COLLECT2.md``, ``docs/distributed/TRANSPORT.md``). The
world of one still behaves as described --- reductions are the identity,
``broadcast`` and ``barrier`` are no-ops that are *true* rather than
convenient --- it is simply no longer the only world on offer.

What still refuses by name: the bitwise reduce ops ``BAND``/``BOR``/``BXOR``,
``PREMUL_SUM``, and ``send``/``recv``. The reason for the last has changed with
the transport and is worth stating correctly: it is not that "no amount of
local work makes them mean anything", but that the star has no route between
two non-hub ranks (``docs/distributed/TRANSPORT.md`` section 3).

It is deliberately not ``fake``: upstream's own ``FakeProcessGroup`` docstring
says it "would produce wrong results for every collective", and a wrong result
is worse than a refusal.
"""

from __future__ import annotations

import torch.distributed as dist


#: The name to pass as ``backend=`` to ``torch.distributed.init_process_group``.
BACKEND_NAME = "local"


def _create_local_backend(dist_backend_opts, backend_options=None):
    """``creator_fn`` for ``Backend.register_backend``.

    Called with ``extended_api=True``, so the first argument is a
    ``_DistributedBackendOptions`` carrying the group's rank and size rather
    than the four positional arguments the narrow API passes.
    """
    from torch._C._distributed_c10d import ProcessGroupLocal

    rank = getattr(dist_backend_opts, "group_rank", 0)
    size = getattr(dist_backend_opts, "group_size", 1)
    store = getattr(dist_backend_opts, "store", None)
    return ProcessGroupLocal(rank, size, store)


def register() -> str:
    """Register the ``local`` backend, once. Returns its name.

    Idempotent because importing this module calls it and a caller may call it
    again; ``register_backend`` raises on a duplicate name.
    """
    if BACKEND_NAME.upper() not in dist.Backend._plugins:
        dist.Backend.register_backend(
            BACKEND_NAME,
            _create_local_backend,
            extended_api=True,
            # CPU only. The comment here used to justify that by saying
            # DESIGN.md 11.1's fourth layer -- Metal, Vulkan, NPU -- "is not
            # built", and two of those three now are (see the README's
            # Accelerators row). The restriction stands on its own ground
            # instead: no collective in ProcessGroupLocal has been run on a
            # non-CPU tensor, and claiming a device here would route one onto
            # a device nothing has exercised.
            devices=["cpu"],
        )
    return BACKEND_NAME


register()
