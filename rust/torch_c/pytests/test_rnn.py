"""`aten.lstm.input` and `aten.upsample_linear1d.default` (docs/RNN.md).

Three operator names were this round's target and only two of them were
operators. `torch.conv1d` is a **name**, bound to `aten.convolution.default`
by `bootstrap.py` since docs/ARCH20.md; what `lasr_ctc`/`lasr_encoder`
actually stop on is that composite's refusal of `padding="same"` with an odd
`dilation * (kernel - 1)`, and the last three tests in this file pin
upstream's lowering for it so the fix is a transcription rather than a
re-derivation. See docs/RNN.md §1.

Every number here is measured against real torch 2.13.0 in **its own
process** (`env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`), the shape
`test_voice3.py` uses. Nothing needs numpy or a network.

What is actually at risk, and therefore what is pinned:

  * **`lstm` on a multi-step sequence with a non-trivial `(h_0, c_0)`.** A
    single timestep from a zero hidden state cannot separate a correct
    recurrence from one that drops `h`, folds the two biases, or -- the bug
    this file caught in its own kernel -- writes `h[unit]` in place so that
    unit *k+1*'s gates read a half-stepped hidden row. Every one of those
    returns the right shape.
  * **Gate order `i, f, g, o`,** pinned by a fixture whose four gate blocks
    are deliberately unequal, so a transposed order is a wrong number and
    not a symmetry.
  * **`upsample_linear1d`'s source index is one FUSED multiply-add.**
    Compared as raw BITS, not within a tolerance: the unfused form differs
    by ~3 ULP, which the golden float32 tolerance (1e-5) would pass.
"""

import json
import math
import os
import struct
import subprocess
import sys

from test_shim import _C

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


_PROBE_SCRIPT = r"""
import json, struct, sys
import torch

out = {"_marker": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}


def rec(name, fn):
    try:
        r = fn()
    except Exception as e:
        out[name] = {"raised": type(e).__name__, "msg": str(e)}
        return
    if isinstance(r, torch.Tensor):
        out[name] = {"ok": [float(v) for v in r.reshape(-1).double()],
                     "shape": list(r.shape), "dtype": str(r.dtype)}
    elif isinstance(r, (tuple, list)) and r and isinstance(r[0], torch.Tensor):
        out[name] = {"tuple": [
            {"ok": [float(v) for v in t.reshape(-1).double()],
             "shape": list(t.shape), "dtype": str(t.dtype)} for t in r]}
    else:
        out[name] = {"ok": r}


def recbits(name, fn):
    # Raw float32 bit patterns. A 3-ULP disagreement in the source-index
    # arithmetic is invisible to any tolerance a golden case would use.
    try:
        t = fn()
    except Exception as e:
        out[name] = {"raised": type(e).__name__, "msg": str(e)}
        return
    out[name] = {"bits": [struct.unpack("<I", struct.pack("<f", float(v)))[0]
                          for v in t.reshape(-1).tolist()],
                 "shape": list(t.shape)}


aten = torch.ops.aten

# ---------------------------------------------------------------------------
# Deterministic fixtures. A linear congruential walk rather than `randn`: the
# shim has no `Generator.manual_seed`, so the two sides could not be handed
# the same random draw, and a fixture that differs between them measures
# nothing.
# ---------------------------------------------------------------------------
def vals(n, seed):
    s = seed * 7919 + 13
    v = []
    for _ in range(n):
        s = (s * 1103515245 + 12345) % 2147483648
        v.append(round((s / 2147483648.0 - 0.5) * 2.4, 6))
    return v


def mk(shape, seed, dtype=torch.float32):
    n = 1
    for d in shape:
        n *= d
    return torch.tensor(vals(n, seed), dtype=dtype).reshape(list(shape))


def lstm_call(num_layers, bidirectional, has_biases, batch_first, dtype,
              seq=4, batch=3, in_size=5, hidden=4, dropout=0.0, train=False,
              zero_state=False):
    dirs = 2 if bidirectional else 1
    params = []
    seed = 100
    feat = in_size
    for _ in range(num_layers):
        for _ in range(dirs):
            params.append(mk((4 * hidden, feat), seed, dtype)); seed += 1
            params.append(mk((4 * hidden, hidden), seed, dtype)); seed += 1
            if has_biases:
                params.append(mk((4 * hidden,), seed, dtype)); seed += 1
                params.append(mk((4 * hidden,), seed, dtype)); seed += 1
        feat = hidden * dirs
    x = mk((batch, seq, in_size) if batch_first else (seq, batch, in_size), 1, dtype)
    if zero_state:
        h0 = torch.zeros(num_layers * dirs, batch, hidden, dtype=dtype)
        c0 = torch.zeros(num_layers * dirs, batch, hidden, dtype=dtype)
    else:
        h0 = mk((num_layers * dirs, batch, hidden), 2, dtype)
        c0 = mk((num_layers * dirs, batch, hidden), 3, dtype)
    return aten.lstm.input(x, [h0, c0], params, has_biases, num_layers,
                           dropout, train, bidirectional, batch_first)


_DTYPES = (("f32", torch.float32), ("f64", torch.float64),
           ("f16", torch.float16), ("bf16", torch.bfloat16))

for dname, dt in _DTYPES:
    for nl in (1, 2, 3):
        for bd in (False, True):
            for bs in (True, False):
                for bf in (True, False):
                    rec("lstm_%s_%d_%s_%s_%s" % (dname, nl, bd, bs, bf),
                        lambda nl=nl, bd=bd, bs=bs, bf=bf, dt=dt:
                            lstm_call(nl, bd, bs, bf, dt))

# A zero initial state as well as a non-trivial one: the difference between
# the two is the whole content of "the recurrence carries h_0".
rec("lstm_zero_state", lambda: lstm_call(2, False, True, True, torch.float32, zero_state=True))
rec("lstm_nonzero_state", lambda: lstm_call(2, False, True, True, torch.float32))
# One timestep -- the case that CANNOT tell a correct recurrence from a wrong
# one, recorded so the claim is checkable rather than asserted in prose.
rec("lstm_seq1", lambda: lstm_call(1, False, True, False, torch.float32, seq=1))
rec("lstm_seq12", lambda: lstm_call(1, False, True, False, torch.float32, seq=12))
# batch=1 and hidden=1: the degenerate shapes where a transposed axis is
# invisible, kept so a later change cannot pass by only being tested there.
rec("lstm_batch1", lambda: lstm_call(2, True, True, True, torch.float32, batch=1))
rec("lstm_hidden1", lambda: lstm_call(1, False, True, False, torch.float32, hidden=1))
rec("lstm_wide", lambda: lstm_call(2, True, True, True, torch.float32,
                                   seq=6, batch=2, in_size=3, hidden=5))

# `nn.LSTM` end to end -- the route `parakeet_rnnt`/`parakeet_tdt` take
# (`_VF.lstm(...)`, nine positional arguments, i.e. the `.input` overload).
def nn_lstm():
    m = torch.nn.LSTM(input_size=4, hidden_size=3, num_layers=2, batch_first=True)
    m.eval()
    with torch.no_grad():
        i = 0
        for p in m.parameters():
            flat = vals(p.numel(), 200 + i)
            p.copy_(torch.tensor(flat, dtype=p.dtype).reshape(p.shape))
            i += 1
    x = mk((2, 5, 4), 1)
    y, (h, c) = m(x)
    return (y, h, c)


rec("nn_lstm_module", nn_lstm)

# The bare `torch.lstm(...)` spelling -- the `_VariableFunctions` member
# `_VF.lstm` resolves to, exercised directly so the overload TABLE is checked
# and not only the dispatch key. `overloads.json` lists `lstm.data` before
# `lstm.input` (the vendored `_VariableFunctions.pyi` declaration order), and
# this call is what proves the earlier row does not swallow it.
def torch_lstm_spelling():
    hidden, batch, in_size, seq = 4, 3, 5, 4
    params = [mk((4 * hidden, in_size), 100), mk((4 * hidden, hidden), 101),
              mk((4 * hidden,), 102), mk((4 * hidden,), 103)]
    return torch.lstm(mk((seq, batch, in_size), 1),
                      [mk((1, batch, hidden), 2), mk((1, batch, hidden), 3)],
                      params, True, 1, 0.0, False, False, False)


rec("lstm_via_torch_lstm", torch_lstm_spelling)

# Refusals. `.data` is the OTHER overload and must be refused BY NAME rather
# than answered with `.input`'s kernel.
rec("lstm_packed_data", lambda: aten.lstm.data(
    mk((6, 5), 1), torch.tensor([3, 2, 1]),
    [mk((1, 3, 4), 2), mk((1, 3, 4), 3)],
    [mk((16, 5), 100), mk((16, 4), 101), mk((16,), 102), mk((16,), 103)],
    True, 1, 0.0, False, False))
rec("lstm_dropout_train", lambda: lstm_call(2, False, True, True, torch.float32,
                                            dropout=0.5, train=True))
rec("lstm_dropout_eval", lambda: lstm_call(2, False, True, True, torch.float32,
                                           dropout=0.5, train=False))
rec("lstm_int64", lambda: lstm_call(1, False, True, False, torch.int64))
rec("lstm_2d_input", lambda: aten.lstm.input(
    mk((4, 5), 1), [mk((1, 3, 4), 2), mk((1, 3, 4), 3)],
    [mk((16, 5), 100), mk((16, 4), 101), mk((16,), 102), mk((16,), 103)],
    True, 1, 0.0, False, False, False))
rec("lstm_one_hidden", lambda: aten.lstm.input(
    mk((4, 3, 5), 1), [mk((1, 3, 4), 2)],
    [mk((16, 5), 100), mk((16, 4), 101), mk((16,), 102), mk((16,), 103)],
    True, 1, 0.0, False, False, False))
rec("lstm_wrong_param_count", lambda: aten.lstm.input(
    mk((4, 3, 5), 1), [mk((1, 3, 4), 2), mk((1, 3, 4), 3)],
    [mk((16, 5), 100), mk((16, 4), 101)],
    True, 1, 0.0, False, False, False))

# ---------------------------------------------------------------------------
# upsample_linear1d
# ---------------------------------------------------------------------------
line = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
short = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)

for out_w in (1, 2, 3, 4, 5, 7, 8, 9, 16):
    for ac in (False, True):
        rec("ul1d_%d_%s" % (out_w, ac),
            lambda w=out_w, ac=ac: aten.upsample_linear1d(line, [w], ac, None))
        recbits("ul1d_bits_%d_%s" % (out_w, ac),
                lambda w=out_w, ac=ac: aten.upsample_linear1d(line, [w], ac, None))
# `scales` given explicitly, including one small enough to push the source
# index past the end of the input (4 -> 8 at scales=0.5 reaches 14.5).
for scale, w in ((2.0, 8), (0.5, 8), (1.5, 6), (3.0, 12), (0.25, 5)):
    rec("ul1d_scale_%s_%d" % (scale, w),
        lambda s=scale, w=w: aten.upsample_linear1d(line, [w], False, s))
for dname, dt in (("f64", torch.float64), ("f16", torch.float16),
                  ("bf16", torch.bfloat16)):
    for ac in (False, True):
        rec("ul1d_dtype_%s_%s" % (dname, ac),
            lambda d=dt, ac=ac: aten.upsample_linear1d(line.to(d), [7], ac, None))
rec("ul1d_zero_batch", lambda: aten.upsample_linear1d(
    torch.zeros(0, 2, 4), [7], False, None))
rec("ul1d_uint8", lambda: aten.upsample_linear1d(line.to(torch.uint8), [7], False, None))
rec("ul1d_int64", lambda: aten.upsample_linear1d(line.long(), [7], False, None))
rec("ul1d_bool", lambda: aten.upsample_linear1d(line.bool(), [7], False, None))
rec("ul1d_4d", lambda: aten.upsample_linear1d(
    torch.zeros(1, 2, 3, 4), [7], False, None))
rec("ul1d_1d", lambda: aten.upsample_linear1d(torch.zeros(4), [7], False, None))
rec("ul1d_two_sizes", lambda: aten.upsample_linear1d(line, [7, 7], False, None))
rec("ul1d_zero_channels", lambda: aten.upsample_linear1d(
    torch.zeros(1, 0, 4), [7], False, None))
rec("ul1d_zero_out", lambda: aten.upsample_linear1d(line, [0], False, None))
# The clamp at 0 that cubic does NOT have: the unclamped index here is -0.25.
recbits("ul1d_clamp_at_zero", lambda: aten.upsample_linear1d(short, [8], False, 2.0))
# The `out == in` case, where bicubic has no short circuit.
rec("ul1d_identity_false", lambda: aten.upsample_linear1d(line, [4], False, None))
rec("ul1d_identity_true", lambda: aten.upsample_linear1d(line, [4], True, None))

# ---------------------------------------------------------------------------
# conv1d: a NAME, not a kernel. What upstream's composite lowers to for the
# `padding="same"` case with an ODD `dilation * (kernel - 1)`, recorded from
# both sides so docs/RNN.md §1's four-line fix is a transcription.
# ---------------------------------------------------------------------------
cx = torch.arange(2 * 3 * 9, dtype=torch.float32).reshape(2, 3, 9) / 7
cw4 = torch.arange(3 * 1 * 4, dtype=torch.float32).reshape(3, 1, 4) / 5
cw3 = torch.arange(3 * 1 * 3, dtype=torch.float32).reshape(3, 1, 3) / 5
cb = torch.tensor([0.5, -0.25, 2.0])
rec("conv1d_same_even_kernel", lambda: torch.conv1d(cx, cw3, cb, 1, "same", 1, 3))
rec("conv1d_same_odd_total", lambda: torch.conv1d(cx, cw4, cb, 1, "same", 1, 3))
rec("conv1d_valid", lambda: torch.conv1d(cx, cw4, cb, 1, "valid", 1, 3))
rec("conv1d_int_padding", lambda: torch.conv1d(cx, cw4, cb, 1, 2, 1, 3))
# The composition, spelled with kernels this shim already has. Equal to
# `conv1d_same_odd_total` on upstream; that equality is the fix's proof.
rec("conv1d_same_odd_by_hand", lambda: aten.convolution.default(
    aten.constant_pad_nd.default(cx, [0, 1], 0.0), cw4, cb,
    [1], [1], [1], False, [0], 3))
# ...and the WRONG half of the split, so "either half would do" is refuted
# rather than assumed.
rec("conv1d_same_odd_by_hand_wrong_side", lambda: aten.convolution.default(
    aten.constant_pad_nd.default(cx, [1, 0], 0.0), cw4, cb,
    [1], [1], [1], False, [0], 3))

json.dump(out, sys.stdout)
"""

_cache = {}


def _run(side):
    """`side` is `"shim"` or `"upstream"`. `None` on the shim side when the
    vendored shim is not installed -- `test_voice3.py`'s silent skip, for the
    same reason: `run.sh` builds the standalone `_C`, and only
    `vendor/install_shim.sh` writes the one the vendored tree loads."""
    if side in _cache:
        return _cache[side]
    env = dict(os.environ)
    if side == "shim":
        if not os.path.isfile(_VENDOR_SHIM):
            _cache[side] = None
            return None
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT],
        capture_output=True, text=True, env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"the {side} probe failed to run at all:\n{proc.stderr[-3000:]}"
    )
    data = json.loads(proc.stdout)
    assert data["_marker"] == side, (
        f"the {side} probe imported the wrong torch ({data['_marker']}) -- every "
        "assertion below would have been about the wrong library"
    )
    _cache[side] = data
    return data


def _both(name):
    s = _run("shim")
    if s is None:
        return "skip"
    up = _run("upstream")
    assert name in up, f"{name} is not in the upstream probe output"
    assert name in s, f"{name} is not in the shim probe output"
    return s[name], up[name]


def _compare_entry(name, got, want, atol, rtol):
    assert got.get("shape") == want.get("shape"), f"{name}: shape {got.get('shape')} != {want.get('shape')}"
    assert got.get("dtype") == want.get("dtype"), f"{name}: dtype {got.get('dtype')} != {want.get('dtype')}"
    a, b = got["ok"], want["ok"]
    assert len(a) == len(b), f"{name}: length {len(a)} != {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        if isinstance(x, float) and isinstance(y, float):
            if math.isnan(x) and math.isnan(y):
                continue
        if x == y:
            continue
        assert math.isclose(x, y, rel_tol=rtol, abs_tol=atol), (
            f"{name}[{i}]: {x!r} != {y!r}"
        )


def _agree(name, atol=1e-9, rtol=1e-9):
    """Element-wise agreement with upstream, dtype and shape included."""
    pair = _both(name)
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, f"{name}: upstream itself refused: {want}"
    assert "raised" not in got, f"{name}: the shim refused where upstream did not: {got}"
    if "tuple" in want:
        assert "tuple" in got, f"{name}: expected a tuple, got {got}"
        assert len(got["tuple"]) == len(want["tuple"]), f"{name}: arity"
        for i, (g, w) in enumerate(zip(got["tuple"], want["tuple"])):
            _compare_entry(f"{name}[{i}]", g, w, atol, rtol)
        return
    _compare_entry(name, got, want, atol, rtol)


def _agree_bitwise(name):
    """Raw float32 bit patterns. Nothing is within a tolerance here -- this is
    the check the golden harness structurally cannot make, and the one the
    fused multiply-add finding needs."""
    pair = _both(name)
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want and "raised" not in got, f"{name}: {got} {want}"
    assert got["shape"] == want["shape"], f"{name}: shape"
    assert got["bits"] == want["bits"], (
        f"{name}: float32 bit patterns differ from upstream. This is the "
        f"fused-multiply-add finding in docs/RNN.md §3 -- an unfused "
        f"`scale * (i + 0.5) - 0.5` differs by ~3 ULP and would pass every "
        f"tolerance in the golden harness.\nshim={got['bits']}\nup ={want['bits']}"
    )


def _both_refuse(name):
    pair = _both(name)
    if pair == "skip":
        return
    got, want = pair
    assert "raised" in want, f"{name}: upstream did NOT refuse: {want}"
    assert "raised" in got, (
        f"{name}: upstream refuses and the shim computed something instead: {got}"
    )


def _entry(side, name):
    data = _run(side)
    return None if data is None else data[name]


# --- the op is advertised ---------------------------------------------------


def test_both_ops_are_advertised():
    implemented = set(_C._aten_implemented())
    for op in ("aten.lstm.input", "aten.upsample_linear1d.default"):
        assert op in implemented, op


def test_lstm_data_is_not_advertised():
    """The packed-sequence overload is refused by name, so it must not appear
    in either list -- an entry there would say a kernel exists."""
    both = set(_C._aten_implemented()) | set(_C._aten_implemented_awaiting_golden())
    assert "aten.lstm.data" not in both


# --- lstm: the arithmetic ---------------------------------------------------


_LSTM_TOL = {"f32": (1e-5, 1e-5), "f64": (1e-9, 1e-9),
             "f16": (5e-3, 5e-3), "bf16": (6e-2, 6e-2)}


def test_lstm_matches_upstream_across_every_option_combination():
    """96 combinations: four dtypes x {1,2,3} layers x bidirectional x
    has_biases x batch_first. Each is a multi-step (seq=4) sequence from a
    NON-ZERO `(h_0, c_0)`, which is what makes it a check of the recurrence
    rather than of one cell."""
    checked = 0
    for dname in ("f32", "f64", "f16", "bf16"):
        atol, rtol = _LSTM_TOL[dname]
        for nl in (1, 2, 3):
            for bd in (False, True):
                for bs in (True, False):
                    for bf in (True, False):
                        _agree("lstm_%s_%d_%s_%s_%s" % (dname, nl, bd, bs, bf),
                               atol=atol, rtol=rtol)
                        checked += 1
    assert checked == 96, checked


def test_lstm_carries_the_initial_hidden_state():
    """The check a single timestep from a zero state cannot make. Both runs
    use identical weights and input and differ ONLY in `(h_0, c_0)`; a kernel
    that drops the initial state returns the same numbers for both, and this
    asserts upstream itself does not."""
    zero, nonzero = _entry("upstream", "lstm_zero_state"), _entry("upstream", "lstm_nonzero_state")
    if zero is None:
        return
    assert zero["tuple"][0]["ok"] != nonzero["tuple"][0]["ok"], (
        "upstream gives the same output for a zero and a non-zero h_0 with "
        "this fixture -- the fixture, not the kernel, is what would be wrong"
    )
    _agree("lstm_zero_state", atol=1e-5, rtol=1e-5)
    _agree("lstm_nonzero_state", atol=1e-5, rtol=1e-5)


def test_lstm_multi_step_is_not_a_repeated_single_step():
    """`seq=1` and `seq=12` on the same weights. If `seq=12`'s last timestep
    equalled its first, the recurrence would not be advancing -- asserted
    against upstream's own numbers so this cannot pass by both sides being
    wrong the same way."""
    one, twelve = _entry("upstream", "lstm_seq1"), _entry("upstream", "lstm_seq12")
    if one is None:
        return
    hidden = twelve["tuple"][0]["shape"][-1]
    batch = twelve["tuple"][0]["shape"][1]
    row = hidden * batch
    first, last = twelve["tuple"][0]["ok"][:row], twelve["tuple"][0]["ok"][-row:]
    assert first != last, "the fixture's sequence does not move the hidden state"
    # h_n is the LAST timestep's hidden state, not the first -- the check a
    # kernel that returns h_0 unchanged, or that returns step 0's h, fails.
    assert twelve["tuple"][1]["ok"] == last, (
        "upstream's h_n is not the last timestep of the output -- the "
        "convention this test pins has changed"
    )
    assert one["tuple"][1]["ok"] != twelve["tuple"][1]["ok"], (
        "seq=1 and seq=12 give the same final hidden state, so this pair "
        "does not discriminate a recurrence that advances from one that "
        "does not"
    )
    _agree("lstm_seq1", atol=1e-5, rtol=1e-5)
    _agree("lstm_seq12", atol=1e-5, rtol=1e-5)


def test_lstm_degenerate_and_wide_shapes():
    for name in ("lstm_batch1", "lstm_hidden1", "lstm_wide"):
        _agree(name, atol=1e-5, rtol=1e-5)


def test_lstm_bidirectional_concatenates_per_timestep():
    """The trap: the backward direction's output for step `t` goes into the
    SECOND half of the feature axis at step `t`, not into a reversed output
    sequence. Both are `(seq, batch, 2H)` and only one is upstream's."""
    pair = _both("lstm_f32_1_True_True_False")
    if pair == "skip":
        return
    got, want = pair
    seq, batch, width = want["tuple"][0]["shape"]
    assert width % 2 == 0
    hidden = width // 2
    up = want["tuple"][0]["ok"]

    def block(vals_, t, b, half):
        base = (t * batch + b) * width + half * hidden
        return vals_[base:base + hidden]

    # The backward half at the LAST timestep is the direction's *first* step,
    # so it is the one closest to h_0 -- and it must equal h_n[1], which is
    # the backward direction's final hidden state.
    hn = want["tuple"][1]["ok"]
    for b in range(batch):
        assert block(up, 0, b, 1) == hn[batch * hidden + b * hidden:
                                        batch * hidden + (b + 1) * hidden], (
            "upstream's backward h_n is not the backward output at t=0 -- the "
            "concatenation convention this test pins has changed"
        )
    _agree("lstm_f32_1_True_True_False", atol=1e-5, rtol=1e-5)


def test_torch_lstm_spelling_resolves_to_the_input_overload():
    """`torch.lstm(...)` with nine positional arguments must reach
    `aten::lstm.input`. `overloads.json` lists `lstm.data` FIRST (the vendored
    `.pyi` declaration order), and both overloads take nine positional
    arguments -- so if the resolver bound on arity alone this call would land
    on the packed-sequence row and refuse. It does not, and this is what says
    so."""
    _agree("lstm_via_torch_lstm", atol=1e-5, rtol=1e-5)


def test_nn_lstm_module_end_to_end():
    """`nn.LSTM` with `batch_first=True`, num_layers=2 -- `parakeet_rnnt`'s
    and `parakeet_tdt`'s own spelling, through overload resolution
    (`_VF.lstm`) rather than through `_aten_dispatch`."""
    _agree("nn_lstm_module", atol=1e-5, rtol=1e-5)


# --- lstm: the refusals -----------------------------------------------------


def test_lstm_packed_data_overload_is_refused_by_name():
    """`aten::lstm.data` is a DIFFERENT iteration over the same weights.
    Upstream computes it; this shim must refuse it rather than answer with
    `.input`'s kernel, which would return a right-shaped wrong tensor."""
    pair = _both("lstm_packed_data")
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, "upstream refuses lstm.data -- the fixture is wrong"
    assert got.get("raised") == "NotImplementedError", got
    assert "packed" in got["msg"], got["msg"]


def test_lstm_refuses_dropout_only_while_training():
    """`dropout=0.5, train=False` is the eval path every checkpoint takes and
    must compute; `train=True` draws from the RNG and is refused."""
    _agree("lstm_dropout_eval", atol=1e-5, rtol=1e-5)
    pair = _both("lstm_dropout_train")
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, "upstream refuses train-mode dropout -- fixture wrong"
    assert got.get("raised") == "NotImplementedError", got


def test_lstm_argument_refusals_match_upstream():
    for name in ("lstm_int64", "lstm_one_hidden", "lstm_wrong_param_count"):
        _both_refuse(name)


def test_lstm_2d_input_is_a_shim_restriction_not_an_upstream_one():
    """Measured, and recorded as a divergence rather than filed as a match:
    `aten::lstm.input` does NOT rank-check upstream, and a 2-D `(4, 5)` input
    returns a `(4, 3, 4)` result there. This shim refuses it. `nn.LSTM`
    unsqueezes an unbatched input before it ever reaches this op, so nothing
    in a real model takes this route -- but "upstream also refuses" would be
    false, and a `_both_refuse` here would have asserted it."""
    pair = _both("lstm_2d_input")
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, (
        "upstream now refuses a 2-D lstm input; this shim's refusal is no "
        "longer a divergence and this test should become _both_refuse"
    )
    assert got.get("raised") == "RuntimeError", got
    assert "3D tensor" in got["msg"], got["msg"]


# --- upsample_linear1d ------------------------------------------------------


def test_upsample_linear1d_matches_upstream():
    for out_w in (1, 2, 3, 4, 5, 7, 8, 9, 16):
        for ac in (False, True):
            _agree("ul1d_%d_%s" % (out_w, ac))


def test_upsample_linear1d_is_bit_exact_not_merely_close():
    """The fused multiply-add. `docs/GLU.md` §2 called `upsample_linear1d` a
    likely sibling of `upsample_bilinear2d`; transcribing that kernel's
    unfused `scale * index - 0.5` here gives lambdas `(0.5, 0.5)` where
    upstream gives `(0.49999988, 0.50000012)`, ~3 ULP out, on the 4 -> 7
    float32 resample below. That is inside every tolerance the golden harness
    uses, which is why this test compares bits."""
    for out_w in (1, 2, 3, 4, 5, 7, 8, 9, 16):
        for ac in (False, True):
            _agree_bitwise("ul1d_bits_%d_%s" % (out_w, ac))


def test_upsample_linear1d_explicit_scales():
    """Including `scales=0.5` on a 4 -> 8 resample, whose source index reaches
    14.5 -- past the end of a 4-wide input. `upsample_bilinear2d` clamps only
    the upper tap and would read out of bounds here."""
    for scale, w in ((2.0, 8), (0.5, 8), (1.5, 6), (3.0, 12), (0.25, 5)):
        _agree("ul1d_scale_%s_%d" % (scale, w))


def test_upsample_linear1d_clamps_the_source_index_at_zero():
    """`align_corners=False` clamps at 0 -- bilinear's rule, NOT bicubic's
    (docs/DEMAND8.md §2.4 found cubic does not clamp). The unclamped index
    for the first output here is -0.25, so an unclamped kernel extrapolates
    below the first input value instead of returning it."""
    _agree_bitwise("ul1d_clamp_at_zero")
    up = _entry("upstream", "ul1d_clamp_at_zero")
    if up is None:
        return
    assert up["bits"][0] == 0, (
        "upstream's first output is no longer exactly 0.0 -- the clamp this "
        "test pins has changed, and the shim should be re-measured, not "
        "adjusted to match"
    )


def test_upsample_linear1d_out_equals_in():
    """bicubic has no `out == in` short circuit (docs/DEMAND8.md). Whether
    linear's is observable is a separate question from whether it is there:
    with `align_corners=False` and scale 1 the index arithmetic is exact
    anyway, so this pins the VALUES (an identity) and does not claim to have
    observed the branch."""
    _agree("ul1d_identity_false")
    _agree("ul1d_identity_true")
    up = _entry("upstream", "ul1d_identity_false")
    if up is None:
        return
    assert up["ok"] == list(float(v) for v in range(24)), up["ok"]


def test_upsample_linear1d_dtypes():
    for dname in ("f64", "f16", "bf16"):
        for ac in (False, True):
            tol = 1e-9 if dname == "f64" else (5e-3 if dname == "f16" else 6e-2)
            _agree("ul1d_dtype_%s_%s" % (dname, ac), atol=tol, rtol=tol)
    _agree("ul1d_zero_batch")


def test_upsample_linear1d_refusals_match_upstream():
    """`uint8` is refused HERE where `upsample_nearest1d` computes it and
    `upsample_bilinear2d` has a separate fixed-point kernel -- measured, not
    carried over."""
    for name in ("ul1d_uint8", "ul1d_int64", "ul1d_bool", "ul1d_4d", "ul1d_1d",
                 "ul1d_two_sizes", "ul1d_zero_channels", "ul1d_zero_out"):
        _both_refuse(name)


def test_upsample_linear1d_refusal_names_the_right_kernel():
    """`"compute_indices_weights_linear"`, not bilinear's
    `"upsample_bilinear2d_channels_last"`. A refusal that names the wrong
    kernel says the wrong thing about which kernel a caller failed to
    reach."""
    pair = _both("ul1d_int64")
    if pair == "skip":
        return
    got, want = pair
    assert "compute_indices_weights_linear" in want["msg"], want["msg"]
    assert "compute_indices_weights_linear" in got["msg"], got["msg"]


# --- conv1d: the name that was not a kernel ---------------------------------


def test_conv1d_is_a_name_not_a_kernel():
    """There is no `aten.conv1d.*` in this shim and there does not need to be.
    `aten::conv1d` is `CompositeImplicitAutograd` upstream and `bootstrap.py`
    binds `torch.conv1d` to `aten.convolution.default`, which has been here
    since docs/OPS4.md."""
    both = set(_C._aten_implemented()) | set(_C._aten_implemented_awaiting_golden())
    assert not [k for k in both if k.startswith("aten.conv1d.")], sorted(both)
    assert "aten.convolution.default" in both


def test_conv1d_forms_that_already_work():
    for name in ("conv1d_same_even_kernel", "conv1d_valid", "conv1d_int_padding"):
        _agree(name, atol=1e-5, rtol=1e-5)


def test_conv1d_same_with_odd_total_padding_is_still_refused():
    """`lasr_ctc` and `lasr_encoder`'s ACTUAL wall (docs/RNN.md §1): not the
    name `torch.conv1d`, which exists, but this one branch of its composite.
    `LasrEncoderConvolutionModule` uses `padding="same"` with
    `conv_kernel_size=32` (an EVEN kernel, whose comment says it should be
    odd), so `dilation * (kernel - 1)` is odd and upstream pads
    asymmetrically.

    Left as a refusal, not weakened: `bootstrap.py` is where the fix goes and
    is not this round's territory. When it lands, this test flips to
    `_agree` -- and the next two tests are what makes that a transcription
    rather than a re-derivation."""
    pair = _both("conv1d_same_odd_total")
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, "upstream refuses -- the fixture is wrong"
    assert got.get("raised") == "NotImplementedError", (
        "the shim now computes conv1d(padding='same') with an odd total. If "
        "bootstrap.py gained the lowering, change this test to _agree(...) "
        "rather than deleting it -- docs/RNN.md §1"
    )


def test_conv1d_same_odd_lowers_to_pad_right_then_convolve():
    """Upstream's own lowering, read off a `TorchDispatchMode` logger and
    reproduced here with kernels this shim ALREADY has:

        constant_pad_nd(x, [0, total % 2]) ; convolution(..., [total // 2], ...)

    Both sides compute this, so the equality below is checked on the shim as
    well as upstream -- which is the whole claim that `lasr`'s wall needs no
    new kernel."""
    _agree("conv1d_same_odd_by_hand", atol=1e-5, rtol=1e-5)
    up_same = _entry("upstream", "conv1d_same_odd_total")
    up_hand = _entry("upstream", "conv1d_same_odd_by_hand")
    if up_same is None:
        return
    assert up_same["ok"] == up_hand["ok"], (
        "upstream's conv1d(padding='same') no longer equals "
        "constant_pad_nd(x,[0,1]) + convolution(...,[1],...) -- docs/RNN.md "
        "§1's proposed four-line fix is no longer the right lowering"
    )


def test_conv1d_same_odd_the_other_half_is_wrong():
    """"Either half of the split would do" -- refuted, not assumed. Padding
    the LEFT instead of the right shifts the output by one sample, which is
    exactly what `bootstrap.py`'s refusal message says and is the reason the
    branch was refused rather than rounded."""
    up_same = _entry("upstream", "conv1d_same_odd_total")
    up_wrong = _entry("upstream", "conv1d_same_odd_by_hand_wrong_side")
    if up_same is None:
        return
    assert up_same["ok"] != up_wrong["ok"], (
        "the two padding splits now agree, so this fixture no longer "
        "discriminates -- pick a kernel/dilation where they differ"
    )
    _agree("conv1d_same_odd_by_hand_wrong_side", atol=1e-5, rtol=1e-5)


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
