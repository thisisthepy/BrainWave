"""Profile the decode loop by op, to find where time actually goes.

Uses a bigger model config than the test suite's toy config, to make the
attention shapes large enough that the SDPA kernel cost is measurable.
The model is from_config (random weights, no download needed).

Usage:
    PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
        /Volumes/macMini/caches/spike-venv/bin/python tools/bench/profile_decode.py
"""

import cProfile
import io
import pstats
import sys
import time
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig

# ---- model setup ---------------------------------------------------------
# SmolLM2-135M-like config, but smaller so it runs fast.
# SmolLM2-135M: hidden=576, intermediate=1536, num_hidden_layers=30,
#   num_attention_heads=9, num_key_value_heads=3, vocab=49152
# We use a 4-layer version to keep it fast while making SDPA measurable.
CFG = dict(
    vocab_size=256, hidden_size=576, intermediate_size=1536, num_hidden_layers=4,
    num_attention_heads=9, num_key_value_heads=3, max_position_embeddings=2048,
    tie_word_embeddings=False,
)

PROMPT_LEN = 8  # short prompt, decode path is what we care about
NEW_TOKENS = 32  # enough to see growing context


def _fill(model):
    sd = model.state_dict()
    new = {}
    for i, key in enumerate(sorted(sd)):
        ref = sd[key]
        n = 1
        for d in ref.shape:
            n *= int(d)
        state = (i + 1) * 7919
        vals = []
        for _ in range(n):
            state = (state * 1103515245 + 12345) % 2147483648
            vals.append(round(((state / 2147483648.0) * 2.0 - 1.0) * 0.2, 6))
        t = torch.tensor(vals, dtype=torch.float32)
        new[key] = t.reshape(list(int(d) for d in ref.shape)) if len(ref.shape) != 1 else t
    model.load_state_dict(new)
    return model


print("Building model...", flush=True)
model = AutoModelForCausalLM.from_config(
    LlamaConfig(**CFG), attn_implementation="sdpa"
)
model.eval()
_fill(model)
print(f"  {sum(p.numel() for p in model.parameters())} parameters", flush=True)

# ---- wrapping ---------------------------------------------------------------
_orig_sdpa = F.scaled_dot_product_attention

sdpa_time = 0.0
sdpa_count = 0
sdpa_shapes = []


def _wrap_sdpa(*args, **kwargs):
    global sdpa_time, sdpa_count
    t0 = time.perf_counter()
    r = _orig_sdpa(*args, **kwargs)
    sdpa_time += time.perf_counter() - t0
    sdpa_count += 1
    if len(sdpa_shapes) < 10:
        sdpa_shapes.append(tuple(
            tuple(int(d) for d in a.shape) for a in args if hasattr(a, 'shape')
        ))
    return r


F.scaled_dot_product_attention = _wrap_sdpa

# ---- decode loop ---------------------------------------------------------
# Use deterministic token ids so EOS isn't hit early
ids = torch.tensor([[(i * 7919 + 13) % (CFG["vocab_size"] - 1) + 1 for i in range(PROMPT_LEN)]])
print(f"Prompt: {ids[0].tolist()}", flush=True)

# Warm up
print(f"Warming up (1 forward)...", flush=True)
with torch.no_grad():
    dummy = model(ids)
print(f"  done.", flush=True)

# Reset SDPA counters
sdpa_time = 0.0
sdpa_count = 0
sdpa_shapes.clear()

# Run the decode loop manually (to avoid EOS stopping early)
print(f"Profiling {NEW_TOKENS}-step greedy decode (use_cache=False, no EOS)...", flush=True)
pr = cProfile.Profile()
generated = list(ids[0].tolist())
t_total_start = time.perf_counter()
with torch.no_grad():
    pr.enable()
    for step in range(NEW_TOKENS):
        input_ids = torch.tensor([generated])
        logits = model(input_ids).logits
        next_token = int(logits[0, -1].argmax().item())
        generated.append(next_token)
    pr.disable()
t_total = time.perf_counter() - t_total_start

# Restore
F.scaled_dot_product_attention = _orig_sdpa

n_steps = NEW_TOKENS
final_seq_len = len(generated)
print(f"\nTotal decode wall time: {t_total*1000:.1f} ms")
print(f"Tokens/sec: {n_steps / t_total:.1f}")
print(f"Sequence length grew from {PROMPT_LEN} to {final_seq_len}")

print(f"\n=== SDPA timing ===")
if sdpa_count > 0:
    print(f"  Total: {sdpa_time*1000:.2f} ms ({sdpa_count} calls, {sdpa_time/sdpa_count*1000:.3f} ms/call)")
    print(f"  Share of total wall: {100*sdpa_time/t_total:.1f}%")
    print(f"\n  Shapes (first few calls):")
    for i, sh in enumerate(sdpa_shapes[:5]):
        print(f"    call {i}: Q={sh[0]}, K={sh[1]}, V={sh[2]}")
    if len(sdpa_shapes) > 5:
        print(f"    ...")
        for i, sh in enumerate(sdpa_shapes[-2:]):
            print(f"    call {sdpa_count - 2 + i}: Q={sh[0]}, K={sh[1]}, V={sh[2]}")
else:
    print("  ** No SDPA calls recorded!")

# cProfile breakdown
print(f"\n\n=== cProfile top 30 (tottime) ===")
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats(30)
print(s.getvalue())

print(f"\n=== cProfile top 30 (cumulative) ===")
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(30)
print(s.getvalue())
