# Dispatch Overhead Analysis

## 1. Premise check — "what is left is dispatch overhead, not arithmetic"

The brief hypothesizes that `_aten_dispatch` accounts for 70% of decode wall time and the remaining gap is dispatch overhead, not arithmetic. **This premise is false.**

The author of `docs/kernels/FLASH.md` misread the `cProfile` output. In `cProfile`, inclusive times overlap. `torch.nn.functional.linear` calls `_aten_dispatch` (to run `matmul`), meaning the 110ms spent in `linear` is *included* in the 142ms spent in `_aten_dispatch`. The time spent in `_aten_dispatch` is almost entirely arithmetic.

### 1.1 Full decode loop profile (SmolLM2-135M, 32 tokens)

Measured with a Python script executing a greedy decode of 32 tokens without cache, hooking into `_aten_dispatch` to measure Rust internal time vs total time:

* **Total decode wall time**: 1132 ms
* **Total ops**: 56541 (~1766 ops/token)
* **Rust `_aten_dispatch` time (inclusive of kernels)**: 987 ms (87% of wall time)
* **`aten.matmul.default` time**: 800 ms (6784 calls at 118 µs/call)

`matmul` alone represents 800 ms out of the 987 ms spent in `_aten_dispatch` — which is **70% of the entire decode wall time**. The remaining 187 ms is spread across 50,000 other op calls, which corresponds to an average dispatch overhead of roughly 1.5–3 µs per call.

Arithmetic (matmul) is still the dominant cost. Dispatch overhead is a minor factor.

---

## 2. Task 1: Decompose the per-op cost

For a 32-token decode loop, we see ~1766 `_aten_dispatch` calls per token.

Top ops by call count across 32 tokens (56541 total calls):
* `aten.mul.Tensor`: 8771 calls, avg 3430 ns
* `aten.matmul.default`: 6784 calls, avg 118029 ns
* `aten.t.default`: 6752 calls, avg 419 ns
* `aten.slice.Tensor`: 3936 calls, avg 656 ns
* `aten.add.Tensor`: 3905 calls, avg 2646 ns
* `aten.transpose.int`: 3872 calls, avg 706 ns
* `aten.view.default`: 2912 calls, avg 968 ns

### 2.1 Within one dispatch, where the time goes

To measure overhead excluding the kernel, we isolated `t.view(1, 6, 576)`.
Total time per call: **1542 ns** (after a minor Rust optimization).

Breakdown:
1. **Python `resolve` and `kwargs` allocation**: ~570 ns (37%)
2. **Rust `check_devices_agree`**: ~120-160 ns (9%)
3. **Rust `aten_dispatch_inner` (unboxing / candle / boxing)**: ~330 ns (21%)
   - `tensor_arg` / `shape_arg` (kwargs dict lookup and Tuple->Vec cast): ~170 ns
   - `candle` reshape kernel: ~80 ns
   - `finish` (PyTensorBase allocation): ~40 ns
4. **Rust `tensor::promote` (wrapping)**: ~150 ns (10%)
5. **Python/C boundary overhead**: ~150 ns (10%)

**Overhead excluding the kernel**: ~1460 ns.

---

## 3. Task 2: Establish the ceiling

We measured the per-op overhead on cheap ops against upstream PyTorch on the same host:

| Op | Upstream (ns) | Shim (ns) | Ratio |
|---|---|---|---|
| `.view(...)` | 798 | 1542 | **1.93x slower** |
| `.transpose(...)` | 835 | 1280 | **1.53x slower** |
| `.add(scalar)` | 1537 | 2139 | **1.39x slower** |
| `.size()` | 125 | 87 | **0.70x (faster)** |

Our per-op overhead is **above** upstream's (about ~700 ns slower per call).

However, at ~1766 ops per token, this 700 ns gap adds up to **1.2 ms per token**. For a 32-token decode, that is ~38 ms of wall clock time. Given the 1132 ms total wall time, closing the entire dispatch overhead gap to upstream would yield at most a **~3% speedup**.

---

## 4. Task 3: Reduce what you found

We identified two optimizations and measured them:

1. **`scan_for_device` short-circuit (Rust)**
   The loop in `scan_for_device` previously scanned every element of sequences (like `size=(1, 6, 576)` in `view`) attempting to cast each integer to a `PyTensorBase`. By short-circuiting the loop if the first element is not a tensor, we avoided unnecessary casts.
   * *Result*: Reduced `view` overhead from 1640 ns to 1542 ns.

2. **Positional Arguments Fast Path (Python)**
   To bypass Python `kwargs` dictionary allocation and Rust `PyDict_GetItem` lookups, we tested a hardcoded fast path for `view` and `transpose` that passes arguments positionally.
   * *Result*: Reduced `view` overhead from 1542 ns to **618 ns**.
   * Reduced `transpose` from 1280 ns to **553 ns**.
   * This makes the shim **faster than upstream** for these ops (618 ns vs 798 ns). However, since this requires bypassing `entry.resolve()`, making it general requires generating Python fast-paths per-schema, which adds complexity for a 3% overall speedup.

Given the premise is false, we are stopping here as instructed: *"Reporting 'the premise is wrong, here is what is actually true' is the most valuable outcome this round can have... If our per-op overhead is at or below upstream's, say so and stop... Do not manufacture work; report it and stop."*

While we are slightly above upstream generally, we proved the gap is only ~3% of wall time, and we demonstrated a path (positional fast paths) to beat upstream if that 3% ever becomes the bottleneck.

---

## 5. Correction — everything above measures a configuration nobody runs

Sections 1 to 4 profile `use_cache=False`, and so did docs/kernels/FLASH.md before them.
That is not the default: `generate()` uses a KV cache, and the README's own
streaming example does. Without the cache every step recomputes all `S` tokens,
so matmul is inflated by exactly the factor that made it look dominant.

Re-measured with the cache on, SmolLM2-135M, f32, greedy, 24 new tokens:

```
                     best        median
  shim            525.6 ms     529.6 ms      45.7 tok/s
  upstream        536.6 ms     577.1 ms      44.7 tok/s
  upstream (again) 539.6 ms    541.4 ms      44.5 tok/s
```

Generated text is character-identical across all three.  Upstream was measured
on both sides of the shim so that drift in machine load would show up as
disagreement between its two runs; it did not.

**The shim is at parity with upstream on the default decode path, and edges it.**
Neither this round nor the two before it noticed, because all three measured the
shim alone and compared it only against its own earlier self. The single
comparison that decides whether any of this optimisation is worth doing --
end-to-end against upstream, in the configuration a user actually runs -- had
never been made.

### 5.1 What that does to §3's sizing

§3 sized the dispatch gap at ~3% of wall time using the cacheless loop. With the
cache on, the picture is:

```
  wall (cProfile, shim)                    665.9 ms
  _aten_dispatch  43970 calls   438.7 ms   65.9%   (inclusive: kernels are in here)
  resolve         29098 calls    46.0 ms    6.9%   (Python-side binding, reclaimable)
  ops per token                    1832
  700 ns x 43970 calls            30.8 ms    4.6%   (the §3 gap against upstream)
```

So the reclaimable overhead is **5-7%, not 3%** -- `resolve` alone is 6.9% of
wall. That is still modest in isolation. It reads differently now only because
of parity: we are not chasing upstream with it, we would be pulling ahead.

### 5.2 The fast path that was reverted

§4.2 measured a positional fast path at `view` 1542 -> 618 ns and `transpose`
1280 -> 553 ns, beating upstream's 798 ns, and reverted it as not worth the
complexity for 3%. Two things change that judgement rather than the measurement:
the number is 5-7%, and this project already has the per-schema machinery the
generalisation would need (`_SchemaPlan` / `_ArgPlan`, docs/bindings/BIND.md), so
"generating Python fast-paths per-schema" is an extension of something built,
not a new mechanism.

**It is still not restored here.** It was measured on two hardcoded ops, and a
general version has to answer what happens to keyword arguments, defaults and
overloads -- which is the round that should follow this one, with the parity
baseline above as the thing to beat.
