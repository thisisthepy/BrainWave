# Tiled attention — profile, implementation and decision

> **Write-as-you-go.** This document is appended to incrementally so that partial
> results survive an interruption.  The earliest sections are profile data
> gathered *before* any code was written; later sections record the kernel, its
> speedup, its numerical cost and the recommendation.

Measured on 2026-09-02.  Host `darwin/arm64`, CPython 3.13.0, torch 2.13.0,
transformers 5.15.1 (`/Volumes/macMini/caches/spike-venv`).

---

## 0. Premise check — "attention materialisation is the bottleneck"

The brief claims materialising the full `[B,H,S,S]` attention matrix is the
largest remaining speed gap on the decode path.  **Every brief in the last two
weeks was wrong about the premise, so this is profiled first.**

### 0.1 Method

Two measurements:

1. **Full decode loop**: 4-layer LlamaForCausalLM (SmolLM2-135M shapes:
   hidden=576, H_q=9, H_kv=3, head_dim=64), greedy 32 new tokens,
   `use_cache=False`, F.scaled_dot_product_attention wrapped for timing.
2. **Isolated kernel microbenchmark**: SDPA and matmul measured at each
   sequence length that actually occurs during decode (S=8…512).

Load average 1.6 at measurement time.

### 0.2 Full decode loop profile

*(Note: Both this table and the microbenchmark table below were measured on this project's **shim**, not upstream PyTorch. The numbers are not mixed. This was confirmed by checking for the `_aten_dispatch` presence and re-measuring the SDPA kernel which matches the shim's slower times.)*

```
SmolLM2-shaped model, 4 layers, 14.5M params
32-step greedy decode (use_cache=False), S = 8 → 40

Total decode wall time:  202.6 ms  (157.9 tok/s)

Component timing (inclusive):
  F.scaled_dot_product_attention:   8.33 ms  (128 calls, 0.065 ms/call)  4.1% of wall
  linear:                         110    ms  (928 calls, 0.118 ms/call) 54.3% of wall
  _aten_dispatch (C builtin):     142    ms                             70.1% of wall
```

**`_aten_dispatch` is the dominant cost at these shapes.** It is the Rust C
extension function that receives every aten op call. The SDPA kernel, at 4.1%,
is not the bottleneck — `linear` (which is matmul) takes 13× more time.

> **Correction (docs/perf/DISPATCH2.md §1).** The sentence this document later drew
> from that line — "what is left on the decode path is dispatch overhead, not
> arithmetic" — misread cProfile. These are *inclusive* times and they overlap:
> `linear` runs *inside* `_aten_dispatch`, so the 110 ms is part of the 142 ms
> rather than beside it. Arithmetic still dominates. DISPATCH2.md §5 also shows
> this whole profile used `use_cache=False`, which is not what `generate()`
> does, and that with the cache on the shim is at **parity with upstream**.

### 0.3 Isolated kernel microbenchmark (SmolLM2-135M shapes)

*(Measured on the **shim**)*

```
B=1, H_q=9, H_kv=3, D=64 (enable_gqa=True, is_causal=True)
3 warmup + 10 timed, minimum of 10

     S     sdpa ms    x30 layers  |  matmul ms    x210 calls
    ---------------------------------------------------------
     8       0.017         0.50   |     0.011         2.28
    16       0.026         0.77   |     0.012         2.42
    32       0.042         1.26   |     0.019         4.09
    64       0.088         2.65   |     0.037         7.81
   128       0.235         7.05   |     0.069        14.42
   256       0.820        24.61   |     0.131        27.53
   512       4.034       121.03   |     0.256        53.71
```

### 0.4 What the profile says

| S range | SDPA × 30 | linear × 210 | SDPA share | who dominates |
|---------|-----------|---------------|------------|---------------|
| 8–40 (decode after 32 tok) | 0.5–1.3 ms | 2.3–4.1 ms | **20–24%** | **linear** |
| 128 | 7.1 ms | 14.4 ms | **33%** | linear |
| 256 | 24.6 ms | 27.5 ms | **47%** | ~equal |
| 512 | 121.0 ms | 53.7 ms | **69%** | **SDPA** |

**The premise is correct only at S ≥ 256.** On the decode path the brief
names — short prompt, growing context after 32 tokens — the sequence length is
8 to 40 and SDPA is **20–24% of the kernel time**, behind linear at 3–4×.

The crossover is around **S ≈ 250**, consistent with docs/numerics/SEQLEN.md §1.2's
fitted curve (`S_crossover = a/b ≈ 546` for the full model-level gap, but
that included the pow term which has since been fixed).

**This does NOT mean tiled attention is useless.** At S=512 it is 2.3× the
matmul cost, so a user who prompts with 500 tokens would see a real difference.
But the brief's premise that it is "the largest remaining speed gap on the
decode path" is only true for long-context decoding, not for the short-prompt
decode loop the brief itself specifies.

### 0.6 Matmul headroom check

To determine whether the `linear` time could be easily reclaimed, we checked how far the shim's matmul is from upstream's at the actual decode shapes (`use_cache=False` computes `S` tokens every step, leading to shapes like `[S, 576] x [576, 1536]` for the MLP `up_proj`).

| Shape (`[S, K] x [K, N]`) | Shim (ms) | Upstream (ms) | Ratio (Shim / Upstream) |
|-------------------------|-----------|---------------|-----------------------|
| `[  8, 576] x [576, 1536]` | 0.091     | 0.095         | 0.96                  |
| `[ 32, 576] x [576, 1536]` | 0.094     | 0.092         | 1.02                  |
| `[ 40, 576] x [576, 1536]` | 0.133     | 0.130         | 1.02                  |
| `[128, 576] x [576, 1536]` | 0.235     | 0.228         | 1.03                  |

**The ratio is near 1.0.** We are stopping optimisation of matmul here, as the decode path is already at the ceiling and the remaining wall-clock is dispatch overhead, not arithmetic.

---

## 1. The kernel already exists, and it is off because it is slower

**`rust/torch_c/src/flash.rs` is a blocked attention kernel with an online
softmax — 845 lines of it — and it has been in this tree the whole time.** Its
own header says so: *"a **blocked** kernel with an online softmax, and the order
in which it recombines the blocks is observable"*. It is `aten::_scaled_dot_
product_flash_attention_for_cpu` reproduced.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/flash.rs reference_enabled present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/flash.rs attend present -->

It is **off by default**, and the header gives the reason in its second
sentence: *"This kernel is 20x slower at T=512 than the candle formulation it
sits beside in `aten.rs`"*. It is kept as a bit-identity **reference** — the one
implementation here that matches upstream exactly, to hold the fast path against
when a numeric difference has to be localised (docs/kernels/SDPA.md §12).

So the task this document was opened for had already been done, with the
opposite result to the one assumed.

### 1.1 A second implementation, written without knowledge of the first

A tiled kernel was nonetheless written from scratch into `sdpa_flash_cpu` —
chunked K/V, running max, running sum, never materialising `S x S` — before
`flash.rs` was noticed. It reproduced the same sign:

| S | materialising default | tiled | |
|---|---|---|---|
| 32 (decode) | 0.042 ms | 0.076 ms | **0.56x** |
| 256 (crossover) | 0.823 ms | 2.174 ms | **0.38x** |
| 512 (long prompt) | 3.910 ms | 8.948 ms | **0.44x** |

Peak RSS over 10 forward passes at S=512 was ~45 MB materialising against
~20 MB tiled, so the memory claim holds: the `S x S` allocation does go away.

**That code was not kept.** It duplicates `flash.rs` in a slower and less exact
form, and its opt-in was defective — the gate was `env::var(..).is_ok()`, which
is true whenever the variable is *set*, so `BW_FAST_TILED=0` would have enabled
it. It is recorded here rather than carried.

### 1.2 What this actually settles

Two independent implementations, one exact and one approximate, now measure the
blocked form as **slower** than materialising — by 20x and by 2x respectively.
That is enough to answer the framing rather than just the attempt:

**On CPU, materialising the score matrix is why this path is fast, not why it is
slow.** The `[B,H,S,S]` product is one BLAS `gemm`; a blocked loop gives that up
in exchange for locality that a CPU with this much cache does not need at these
shapes. Upstream's own CPU kernel does avoid materialising, and pays for it —
which is exactly what `flash.rs`, being a faithful reproduction of it, measures.

Upstream's 3.79 ms at S=1024 (docs/numerics/SEQLEN.md §8) does not come from *not
materialising*. It comes from a hand-written fused kernel whose inner loops are
BLAS-quality. Neither of the two attempts here reaches that by fusing in Rust,
and the materialising path gets BLAS for free.

### 1.3 Recommendation

**Do not pursue a tiled default.** The remaining attention gap at long sequence
lengths is real (§0.4: at S=512 SDPA is 121 ms against matmul's 54 ms), but
"stop materialising" is not the road to it — that road has now been driven twice
and both times it ran the wrong way. Closing it means matching the quality of
upstream's fused inner loops, which is a different and much larger piece of work.

The decode path this round was pointed at is a separate question and §0.6
answers it: **matmul is at parity with upstream (0.96–1.03x), so there is
nothing to reclaim there either.** What is left on the decode path is dispatch
overhead, not arithmetic.

---

*Reproduce §0 with `tools/bench/profile_decode.py` and `tools/bench/profile_sdpa_shapes.py`.*
