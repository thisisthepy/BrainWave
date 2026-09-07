"""The `local` backend -- `torch.distributed` with a world of one.

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

**What this backend is, and is not.** It is the honest world_size-1 case, not a
simulation of a larger one. ``docs/distributed/DISTRIBUTED.md`` has the table; the short
version is that reductions are the identity, ``broadcast`` and ``barrier`` are
no-ops that are *true* rather than convenient, the gather/scatter family
copies, and ``send``/``recv`` refuse by name because no amount of local work
makes them mean anything. It is deliberately not ``fake``: upstream's own
``FakeProcessGroup`` docstring says it "would produce wrong results for every
collective", and a wrong result is worse than a refusal.
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
            # CPU only. DESIGN.md §11.1's fourth layer -- Metal, Vulkan, NPU --
            # is not built, and claiming a device here would route a collective
            # onto a device the tensors cannot be on.
            devices=["cpu"],
        )
    return BACKEND_NAME


register()
