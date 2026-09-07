"""Profile the SDPA op in isolation at realistic shapes.

This measures the SDPA kernel alone (not the full model), at the shapes a
SmolLM2-135M actually produces during decode steps, to establish what fraction
of a forward the SDPA kernel accounts for at various sequence lengths.

SmolLM2-135M: B=1, H=9 query, H_kv=3, head_dim=64, S=varying

Usage:
    PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
        /Volumes/macMini/caches/spike-venv/bin/python tools/bench/profile_sdpa_shapes.py
"""

import sys
import time
import torch
import torch.nn.functional as F

# SmolLM2-135M attention shapes
B = 1
H_Q = 9    # query heads
H_KV = 3   # key/value heads
D = 64     # head_dim
N_LAYERS = 30  # SmolLM2-135M has 30 layers

print(f"SDPA microbenchmark at SmolLM2-135M shapes (B={B}, H_q={H_Q}, H_kv={H_KV}, D={D})")
print(f"Each measurement: 3 warmup + 10 timed passes, minimum of 10")
print(f"{'S':>6s}  {'sdpa ms':>10s}  {'ms/call':>10s}  {'x30 layers':>12s}  notes")
print("-" * 70)

for S in [8, 16, 32, 64, 128, 256, 512]:
    q = torch.randn(B, H_Q, S, D)
    k = torch.randn(B, H_KV, S, D)
    v = torch.randn(B, H_KV, S, D)

    # Warmup
    for _ in range(3):
        F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

    # Timed
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        times.append(time.perf_counter() - t0)

    best = min(times)
    x30 = best * N_LAYERS
    note = f"SxS = {S}x{S} = {S*S} elements"
    print(f"{S:>6d}  {best*1000:>10.3f}  {best*1000:>10.3f}  {x30*1000:>12.2f}  {note}")

# Also measure matmul at the shapes that linear produces
print(f"\n\nmatmul microbenchmark (linear's inner op)")
print(f"{'shape':>24s}  {'ms':>10s}  notes")
print("-" * 60)
for S in [8, 16, 32, 64, 128, 256, 512]:
    # Linear: input [B, S, 576], weight [576, 576] -> matmul
    x = torch.randn(B, S, 576)
    w = torch.randn(576, 576)

    for _ in range(3):
        torch.matmul(x, w)

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        torch.matmul(x, w)
        times.append(time.perf_counter() - t0)

    best = min(times)
    calls_per_fwd = 7 * N_LAYERS  # q,k,v,o proj + gate,up,down in MLP = 7 per layer
    print(f"{'['+str(B)+','+str(S)+',576]x[576,576]':>24s}  {best*1000:>10.3f}  x{calls_per_fwd} = {best*calls_per_fwd*1000:.2f} ms")
