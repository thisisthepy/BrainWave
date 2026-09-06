# Distributed Transport over TCP

This document records the exact mechanism and findings during the implementation of the first distributed transport for `torchnative`.

## 1. Store Rendezvous Order

When initializing a process group over a real network with `init_process_group` at `world_size = 2` (tested with upstream gloo backend), the participants must find each other before any collective executes. The order of operations on the `Store` is:

**Rank 0:**
1. `set('0//cpu//0/rank_0', value)`
2. `set('0//cpu//0/0', value)`
3. `wait(['0//cpu//0/1'])`
4. `get('0//cpu//0/1')`

**Rank 1:**
1. `set('0//cpu//0/rank_1', value)`
2. `get('0//cpu//0/rank_0')` (blocks/retries until Rank 0 sets it)
3. `set('0//cpu//0/1', value)`
4. `wait(['0//cpu//0/0'])`
5. `get('0//cpu//0/0')`

The `Store` interface essentially requires `.set`, `.get`, and `.wait` to complete rendezvous. 

## 2. Implemented Components

The `delta` aggregation demands a process group with more than one rank. To unblock this, we built a minimal but completely honest distributed stack over real OS sockets, entirely in Python inside `rust/torch_c/src/bootstrap.py`:

- **`TCPStore`:** Implements `world_size = 2` rendezvous using Python's `socket` module. The master rank launches a daemon thread serving a JSON-over-TCP protocol that implements the blocking `set`, `get`, and `wait` primitives. 
- **`ProcessGroupLocal` at `world_size = 2`:** No longer refuses. We repurposed `backend="local"` to support `world_size = 2` using an actual TCP transport. During initialization, rank 0 binds to a free port and sets `pg_local_port` in the store; rank 1 waits for it and connects.
- **`allreduce(..., op=SUM)`:** We chose `SUM` because weighted average calculation in `FedAvg` requires it. Tensors are serialized to a JSON list (`t.tolist()`), sent over the wire length-prefixed, received by the peer, parsed back to a `torch.tensor`, and accumulated using `t.add_(peer_tensor)`.

This ensures that the two ranks are physically separated by a socket and do not share memory, fulfilling the requirements for true distribution.

## 3. Scope Boundaries & Named Refusals

Everything else remains refused by name:
- Any `reduceOp` other than `SUM` (e.g. `PREMUL_SUM`) refuses.
- Collectives like `broadcast`, `allgather`, `reduce`, `gather`, and point-to-point ops (`send`, `recv`) refuse if `world_size != 1`.
- Any `world_size` other than 1 and 2 refuses.

> **Superseded on the last point.** `docs/FEDERATED4.md` extended the transport
> to `world_size >= 1` — a star of loopback sockets with the hub at rank 0 — so
> a world larger than two no longer refuses. Everything else in this list still
> holds: `allreduce(op=SUM)` remains the only collective. Left in place rather
> than rewritten, because this document is the record of what two ranks cost.

This enforces the policy: a collective must be honest end-to-end, or it refuses by name. A test at `world_size = 2` ensures we are no longer running an identity function disguised as a distributed operation.
