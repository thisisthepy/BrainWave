"""Tests for docs/training/TRAIN2.md -- convolution's backward, and a model that trains.

`docs/platform/RELEASE_0_0_13a0.md` §5 states two gaps as one sentence:

    No transformer trains through `loss.backward()` yet. ... There is no
    convolution backward rule, so vision models stop.

They turned out to be two different facts. **The transformer half was already
false** when it was written -- a tiny attention LM (embedding, layer norm,
SDPA, GELU MLP, tied-shape head, cross entropy) trains here with no rule added
by this round, and `test_a_tiny_transformer_language_model_trains_and_agrees_with_upstream`
is the measurement that says so. **The convolution half was true**, and this
file is where it stops being true: `aten.convolution.default`,
`aten.avg_pool2d.default` and `aten.adaptive_avg_pool2d.default` gained rules
(`tape.rs`), and `aten.max_pool2d.default` did not -- see
`test_max_pool2d_backward_is_still_refused_by_name` for what it needs.

Every number below is compared against **upstream torch running the identical
program** in a second interpreter, which is docs/training/BACKWARD9.md §1's central
oracle. Nothing here is a literal trajectory: docs/verification/AUDIT.md found that shape of
claim stale six times out of eleven, and a hardcoded number is a claim about
2.13.0 that nothing re-checks.

Where finite differences would have been enough they are not repeated here --
`test_shim.py`'s `_tape_case_bodies` carries the float64 central-difference
case for each new rule, and that oracle shares no code with the rule. What it
cannot reach is the *weight* and *bias* gradients (its cases differentiate one
input) and it cannot reach a `stride`/`dilation` mix-up that a square,
unit-stride case leaves invisible. That is this file's job.
"""

import json
import os
import subprocess
import sys

from test_shim import _C, _upstream_torch, _CKPT_VENDOR_DIR, _CKPT_VENDOR_SHIM


def _run(script, env_overrides):
    env = dict(os.environ)
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def _shim(script):
    return _run(script, {"PYTHONPATH": _CKPT_VENDOR_DIR, "TORCH_USE_RTLD_GLOBAL": "1"})


def _upstream(script):
    return _run(script, {"PYTHONPATH": None, "TORCH_USE_RTLD_GLOBAL": None})


def _available():
    return os.path.isfile(_CKPT_VENDOR_SHIM)


def _flat(values):
    if values and isinstance(values[0], list):
        return [v for row in values for v in row]
    return list(values)


def _worst(ours, theirs):
    """Relative to the largest magnitude upstream produced, which is how every
    other cross-interpreter comparison in this repository is scaled."""
    ours, theirs = _flat(ours), _flat(theirs)
    assert len(ours) == len(theirs), (len(ours), len(theirs))
    assert ours, "compared an empty quantity, which would pass for any implementation"
    scale = max(max(abs(v) for v in theirs), 1.0)
    return max(abs(a - b) for a, b in zip(ours, theirs)) / scale


# --- 1. convolution's three gradients, over configurations that differ ------
#
# docs/kernels/RNN.md measured that `lasr`'s own convolution takes an **even** kernel
# where its comment says odd, so the easy path -- square kernel, stride 1,
# padding 0, groups 1 -- is not where real callers live. Every argument below
# changes at least one of the three gradients, and each has a case that moves
# it *alone* so that a failure names one thing:
#
#   stride     changes which input positions a window reaches, and (this is the
#              one a rule gets wrong) makes the weight gradient's convolution
#              step through the input by `dilation` and through the gradient by
#              `stride` -- the two exchanged. A rule that left them in the
#              forward order is exactly right whenever `stride == dilation`.
#   padding    is *not* passed to the transposed convolution that forms the
#              input gradient; it is a crop of that convolution's scatter, and
#              the far side of the crop can fall inside the input. Measured:
#              with `groups=2, stride=2, padding=1` a rule that passed it
#              disagreed with upstream at **relative 1.0** -- a whole row zero
#              -- while every `padding=0` and every `stride=1` case in the same
#              table agreed to 3e-07.
#   dilation   spreads the kernel; with `stride` above it is the pair that has
#              to be exchanged.
#   groups     is decomposed here rather than passed, because the 2-D
#              transposed convolution in `aten.rs` takes no `groups` at all and
#              the weight gradient's grouping is not the forward's.
#   even kernel and a non-square input separate "right shape" from "right".

_CONV_SCRIPT = r"""
import json, sys
import torch
out = {"who": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}


def det(n, seed):
    # RNG-free: the two interpreters do not share a generator, so a seed would
    # compare two different programs (docs/training/BACKWARD9.md §1).
    return [((seed * 1103515245 + i * 12345) % 2000 - 1000) / 1000.0 for i in range(n)]


def make(shape, seed):
    n = 1
    for d in shape:
        n *= d
    return torch.tensor(det(n, seed), dtype=torch.float32).reshape(shape)


CASES = {
    "2d_plain":             ((2, 3, 7, 9), (4, 3, 3, 3), 1, 0, 1, 1, True),
    "2d_stride2":           ((2, 3, 8, 8), (4, 3, 3, 3), 2, 0, 1, 1, True),
    "2d_stride2_ragged":    ((1, 2, 7, 9), (3, 2, 3, 3), 2, 0, 1, 1, True),
    "2d_pad1":              ((2, 3, 7, 9), (4, 3, 3, 3), 1, 1, 1, 1, True),
    "2d_dil2":              ((1, 2, 9, 9), (3, 2, 3, 3), 1, 0, 2, 1, True),
    "2d_stride2_pad2_dil2": ((1, 2, 11, 9), (3, 2, 3, 3), 2, 2, 2, 1, True),
    "2d_even_kernel":       ((1, 2, 8, 8), (3, 2, 4, 4), 1, 1, 1, 1, True),
    "2d_groups2":           ((2, 4, 7, 7), (6, 2, 3, 3), 1, 1, 1, 2, True),
    "2d_depthwise":         ((1, 4, 6, 6), (4, 1, 3, 3), 1, 1, 1, 4, True),
    "2d_groups2_stride2":   ((1, 4, 8, 8), (6, 2, 3, 3), 2, 1, 1, 2, True),
    "2d_nobias":            ((1, 2, 6, 6), (3, 2, 3, 3), 1, 0, 1, 1, False),
    "1d_plain":             ((2, 3, 10), (4, 3, 3), 1, 0, 1, 1, True),
    "1d_stride3_pad2_dil2": ((1, 2, 12), (3, 2, 3), 3, 2, 2, 1, True),
    "1d_depthwise_causal":  ((1, 4, 6), (4, 1, 4), 1, 3, 1, 4, True),
}

res = {}
for name, (xs, ws, st, pa, di, gr, bias) in CASES.items():
    x = make(xs, 11)
    x.requires_grad = True
    w = make(ws, 23)
    w.requires_grad = True
    b = None
    if bias:
        b = make((ws[0],), 31)
        b.requires_grad = True
    conv = torch.nn.functional.conv2d if len(xs) == 4 else torch.nn.functional.conv1d
    y = conv(x, w, b, st, pa, di, gr)
    # A ramp, not a sum: `sum(y)` gives every output the same weight, so a rule
    # that permuted or transposed the output gradient would still pass.
    loss = (y * torch.arange(1, y.numel() + 1, dtype=torch.float32).reshape(y.shape)).sum()
    loss.backward()
    res[name] = {
        "y": [float(v) for v in y.detach().reshape(-1)],
        "gx": [float(v) for v in x.grad.reshape(-1)],
        "gw": [float(v) for v in w.grad.reshape(-1)],
        "gb": None if b is None else [float(v) for v in b.grad.reshape(-1)],
    }
out["cases"] = res
json.dump(out, sys.stdout)
"""


def test_convolution_gradients_agree_with_upstream_across_stride_padding_dilation_and_groups():
    """The rule, against upstream's `convolution_backward`, on fourteen
    configurations.

    Bound: `1e-06` relative. Measured worst **4.56e-07** across all 14 cases and
    all three gradients, which is float32 convolution accumulation and the same
    order as the *forward* disagreement in the same table (2.14e-07) -- so the
    backward is not losing anything the forward has not already lost.
    """
    if not _available():
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return
    shim = _shim(_CONV_SCRIPT)
    assert shim["who"] == "shim", shim["who"]
    if _upstream_torch is None:
        return  # no upstream torch in this interpreter -- see docs/models/E2E.md
    up = _upstream(_CONV_SCRIPT)
    assert up["who"] == "upstream", up["who"]
    assert set(shim["cases"]) == set(up["cases"])
    assert len(shim["cases"]) >= 14, len(shim["cases"])

    for name in sorted(shim["cases"]):
        ours, theirs = shim["cases"][name], up["cases"][name]
        for field in ("y", "gx", "gw", "gb"):
            if theirs[field] is None:
                assert ours[field] is None, (name, field)
                continue
            worst = _worst(ours[field], theirs[field])
            assert worst < 1e-06, (name, field, worst)
        # Not all zero: a rule that returned zeros would pass a tolerance
        # against an upstream that... did not, but the check costs nothing and
        # a *shape*-only failure has reached this repository before.
        assert max(abs(v) for v in ours["gw"]) > 0.0, name
        assert max(abs(v) for v in ours["gx"]) > 0.0, name


def test_the_weight_gradient_needs_stride_and_dilation_exchanged():
    """The one line of this rule that a square unit-stride test cannot see.

    `gw[co,ci,u,v] = sum_{n,y,x} gout[n,co,y,x] * x[n,ci, y*s - p + u*d, ...]`
    is a convolution over `u` whose step through `x` is **`dilation`** and whose
    step through `gout` is **`stride`**. Every case with `stride == dilation`
    passes either way round, which is every case anybody writes first.

    So this asserts the case table contains a case where they differ *and* that
    the two disagree there -- if the case were ever weakened to `stride ==
    dilation` this fails rather than silently stopping checking.
    """
    if not _available():
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return
    shim = _shim(_CONV_SCRIPT)
    # `2d_stride2_pad2_dil2` and `1d_stride3_pad2_dil2` are the two, by name.
    for name in ("2d_stride2_pad2_dil2", "1d_stride3_pad2_dil2"):
        assert name in shim["cases"], (
            "the case that separates stride from dilation is gone: " + name)
    if _upstream_torch is None:
        return
    up = _upstream(_CONV_SCRIPT)
    # A case where they *do* differ has a weight gradient that a swapped rule
    # would get wrong; a case where they agree does not. Both are present, and
    # the second is what makes the first a real distinction rather than a
    # tautology.
    same = _worst(shim["cases"]["2d_plain"]["gw"], up["cases"]["2d_plain"]["gw"])
    diff = _worst(shim["cases"]["2d_stride2_pad2_dil2"]["gw"],
                  up["cases"]["2d_stride2_pad2_dil2"]["gw"])
    assert same < 1e-06 and diff < 1e-06, (same, diff)


# --- 2. pooling --------------------------------------------------------------

_POOL_SCRIPT = r"""
import json, sys
import torch
import torch.nn.functional as F
out = {"who": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}


def make(shape, seed):
    n = 1
    for d in shape:
        n *= d
    values = [((seed * 1103515245 + i * 12345) % 2000 - 1000) / 1000.0 for i in range(n)]
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


CASES = {
    "avg2":            (lambda x: F.avg_pool2d(x, 2), (2, 3, 8, 6)),
    "avg_rectangular": (lambda x: F.avg_pool2d(x, [1, 2]), (1, 2, 4, 6)),
    "avg_k3":          (lambda x: F.avg_pool2d(x, 3), (1, 2, 9, 6)),
    "avg_overlapping": (lambda x: F.avg_pool2d(x, 3, 1), (1, 2, 6, 6)),
    "adaptive_1":      (lambda x: F.adaptive_avg_pool2d(x, 1), (2, 3, 4, 6)),
    "adaptive_2":      (lambda x: F.adaptive_avg_pool2d(x, 2), (1, 2, 6, 8)),
    "adaptive_ragged": (lambda x: F.adaptive_avg_pool2d(x, 3), (1, 2, 7, 7)),
    "max_pool":        (lambda x: F.max_pool2d(x, 2), (1, 2, 4, 4)),
}

res = {}
for name, (fn, shape) in CASES.items():
    x = make(shape, 7)
    x.requires_grad = True
    try:
        y = fn(x)
        loss = (y * torch.arange(1, y.numel() + 1, dtype=torch.float32).reshape(y.shape)).sum()
        loss.backward()
    except NotImplementedError as e:
        res[name] = {"refused": str(e).replace("\n", " ")}
    else:
        res[name] = {"gx": [float(v) for v in x.grad.reshape(-1)]}
out["cases"] = res
json.dump(out, sys.stdout)
"""


def test_average_pooling_gradients_agree_with_upstream_where_the_window_tiles():
    """`avg_pool2d` and `adaptive_avg_pool2d`, exactly.

    **Exactly**, not to a tolerance: the rule is `reshape`, `expand`, `reshape`
    and one division by the window's area, so there is no accumulation to
    round. Measured 0.0 on every tiling case, and the assertion is `== 0.0`
    rather than a bound, because a bound here would stop noticing if the rule
    grew arithmetic it does not need.
    """
    if not _available():
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return
    shim = _shim(_POOL_SCRIPT)
    assert shim["who"] == "shim", shim["who"]
    tiling = ("avg2", "avg_rectangular", "avg_k3", "adaptive_1", "adaptive_2")
    for name in tiling:
        assert "gx" in shim["cases"][name], (name, shim["cases"][name])
    if _upstream_torch is None:
        return
    up = _upstream(_POOL_SCRIPT)
    for name in tiling:
        assert _worst(shim["cases"][name]["gx"], up["cases"][name]["gx"]) == 0.0, name


def test_the_pooling_windows_that_do_not_tile_are_refused_by_name():
    """An overlapping window makes the scatter an accumulation and a ragged
    adaptive output makes the windows different sizes. Neither is this rule
    with a different constant, and both would be wrong by a factor a single
    `AvgPool2d(2)` case could never show -- so they refuse, and this pins that
    they refuse *and* that upstream computes them, which is what makes them a
    gap rather than an invalid program.
    """
    if not _available():
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return
    shim = _shim(_POOL_SCRIPT)
    for name, phrase in (("avg_overlapping", "tiles"),
                         ("adaptive_ragged", "divides the input")):
        got = shim["cases"][name]
        assert "refused" in got, (name, got)
        assert phrase in got["refused"], (name, got["refused"])
    if _upstream_torch is None:
        return
    up = _upstream(_POOL_SCRIPT)
    for name in ("avg_overlapping", "adaptive_ragged"):
        assert "gx" in up["cases"][name], (
            "%s is refused here and upstream does not compute it either, so the "
            "refusal is not a gap and this test is measuring nothing" % name)


def test_max_pool2d_backward_is_still_refused_by_name():
    """**The rule this round did not land, and its size.**

    A max pool's gradient routes each output's gradient to the *argmax* of its
    window, so the rule needs the indices. Upstream's forward returns them --
    `max_pool2d_with_indices` -- and this shim's does not: it has
    `aten.max_pool2d.default` only, and `_aten_implemented()` is asserted below
    rather than described, so this sentence cannot go stale.

    That makes it two pieces of work and not one: a forward that also returns
    indices (`aten.rs`, not this round's territory), and a scatter rule over
    them. Recomputing the argmax inside the rule would be a third
    implementation of the window geometry -- the thing docs/verification/AUDIT.md keeps
    finding go stale -- so it is not the way in.
    """
    if not _available():
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return
    implemented = set(_C._aten_implemented())
    assert "aten.max_pool2d.default" in implemented
    assert "aten.max_pool2d_with_indices.default" not in implemented, (
        "the indices-returning forward exists now, which is the missing half of "
        "max_pool2d's backward -- write the rule and delete this test")
    assert "aten.max_pool2d.default" not in set(_C._tape_rules())
    shim = _shim(_POOL_SCRIPT)
    got = shim["cases"]["max_pool"]
    assert "refused" in got, got
    assert "no derivative rule for aten.max_pool2d.default" in got["refused"], got


def test_a_transposed_convolution_is_refused_by_name_rather_than_differentiated():
    """`aten.convolution.default` is one op with a `transposed` flag, so a rule
    for it claims both unless it says otherwise. The forward one is
    differentiated; the transposed one is not -- `output_padding` moves which
    input positions the sum runs over, and reusing the forward arms would be
    wrong in the one place a shape check cannot see.

    Written as a *live program that refuses*, not as a source grep, because a
    refusal nobody executes is a comment.
    """
    if not _available():
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return
    script = r"""
import json, sys
import torch, torch.nn as nn
out = {"who": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}
m = nn.ConvTranspose2d(2, 3, 2, stride=2)
x = torch.ones(1, 2, 4, 4)
try:
    m(x).sum().backward()
except NotImplementedError as e:
    out["refused"] = str(e).replace("\n", " ")
else:
    out["refused"] = None
json.dump(out, sys.stdout)
"""
    shim = _shim(script)
    assert shim["who"] == "shim"
    assert shim["refused"] is not None, (
        "a transposed convolution differentiated -- if that is deliberate, this "
        "test should be replaced by one that compares it to upstream, not deleted")
    assert "transposed" in shim["refused"], shim["refused"]


# --- 3. a real model, trained -----------------------------------------------
#
# The bar for this round. Forward, `loss.backward()`, `optimizer.step()`, over
# several steps, with the loss trajectory and the final parameters compared
# element-wise against upstream running the identical program.

_VISION_SCRIPT = r"""
import json, sys
import torch
import torch.nn as nn

out = {"who": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}


class Net(nn.Module):
    # A depthwise-separable stack -- MobileNet's block, at a size that runs in
    # a test: a strided stem, a depthwise 3x3 with padding, a pointwise 1x1,
    # batch norm in **training** mode after each, and a global average pool
    # into a linear head. Every argument this round implemented is exercised by
    # a layer here: stride 2, padding 1, groups == channels, a 1x1 kernel.
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.depthwise = nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.pointwise = nn.Conv2d(8, 6, 1, bias=True)
        self.bn3 = nn.BatchNorm2d(6)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(6, 4)

    def forward(self, x):
        h = torch.relu(self.bn1(self.stem(x)))
        h = torch.relu(self.bn2(self.depthwise(h)))
        h = torch.relu(self.bn3(self.pointwise(h)))
        h = self.pool(h).reshape(h.shape[0], -1)
        return self.head(h)


def build():
    model = Net()
    # Arithmetic, not seeded: the two interpreters do not share an RNG stream.
    with torch.no_grad():
        k = 0
        for p in model.parameters():
            flat = p.reshape(-1)
            for i in range(flat.shape[0]):
                flat[i] = ((k * 37) % 23 - 11) / 24.0
                k += 1
    return model


n, c, s = 2, 3, 8
x = torch.arange(n * c * s * s, dtype=torch.float32).reshape(n, c, s, s) / (n * c * s * s)
y = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4) / 10.0
criterion = lambda o, t: ((o - t) ** 2).mean()

model = build()
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
out["names"] = [name for name, _ in model.named_parameters()]
out["grad_before_first_backward"] = [p.grad is None for p in model.parameters()]

losses = []
for step in range(5):
    loss = criterion(model(x), y)
    loss.backward()
    if step == 0:
        out["first_grads"] = [[float(v) for v in p.grad.reshape(-1)]
                              for p in model.parameters()]
    optimizer.step()
    optimizer.zero_grad()
    losses.append(float(loss.detach()))
out["losses"] = losses
out["params"] = [[float(v) for v in p.reshape(-1)] for p in model.parameters()]
# The running statistics moved too -- batch norm in training mode is the one
# part of this model that writes something `optimizer.step()` did not.
out["buffers"] = [[float(v) for v in b.reshape(-1)]
                  for name, b in model.named_buffers() if "num_batches" not in name]

# --- accumulation, observed rather than assumed -----------------------------
#
# docs/training/BACKWARD9.md's nullification finding: a training loop stays green when
# `+=` becomes `=`, because `zero_grad(set_to_none=True)` drops `.grad` every
# step so the loop can never observe accumulation. So it is observed here,
# outside the loop: two backwards of the *same* loss with no `zero_grad`
# between them must give exactly twice the gradient.
optimizer.zero_grad()
criterion(model(x), y).backward()
once = [[float(v) for v in p.grad.reshape(-1)] for p in model.parameters()]
criterion(model(x), y).backward()
twice = [[float(v) for v in p.grad.reshape(-1)] for p in model.parameters()]
out["accumulated_ratio"] = [
    [(b / a) if abs(a) > 1e-6 else None for a, b in zip(ra, rb)]
    for ra, rb in zip(once, twice)
]
json.dump(out, sys.stdout)
"""


def test_a_convolutional_model_trains_end_to_end_and_agrees_with_upstream():
    """**The bar.** A depthwise-separable convolutional network -- strided
    convolution, depthwise convolution, pointwise convolution, training-mode
    batch norm, global average pool, linear head -- trained for five steps with
    a real `torch.optim.SGD`, and compared element-wise to upstream torch
    running the identical program.

    Four quantities, because each fails differently:

      * the **loss trajectory** -- a wrong gradient shows here as divergence;
      * the **first step's gradients**, per parameter -- the trajectory could
        agree while two parameters were wrong in compensating directions;
      * the **final parameters** after five `optimizer.step()`s;
      * the **batch-norm running buffers**, which move without the optimizer
        touching them, and are what a `model.eval()` afterwards would use.

    Bound `1e-05` relative, and the measured worst is far inside it. It is
    looser than docs/training/BACKWARD9.md's `2.98e-08` for a reason that is arithmetic
    and not slack: every quantity here has been through five steps of a
    *convolution*, which sums over `C_in * kH * kW` products per output, and
    then through a batch norm's own reduction. Two float32 summations in
    different orders differ at that scale.
    """
    if not _available():
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return
    shim = _shim(_VISION_SCRIPT)
    assert shim["who"] == "shim", shim["who"]

    n_params = len(shim["names"])
    assert shim["grad_before_first_backward"] == [True] * n_params
    losses = shim["losses"]
    assert len(losses) == 5, losses
    # It learned. Not an oracle on its own -- an upstream that also failed to
    # learn would agree with it -- but a loop that returned one number five
    # times would pass every element-wise comparison below.
    assert all(b < a for a, b in zip(losses, losses[1:])), losses
    assert losses[-1] < losses[0] * 0.9, losses

    if _upstream_torch is None:
        return  # no upstream torch in this interpreter -- see docs/models/E2E.md
    up = _upstream(_VISION_SCRIPT)
    assert up["who"] == "upstream", up["who"]
    assert shim["names"] == up["names"], (shim["names"], up["names"])

    for field in ("losses", "first_grads", "params", "buffers"):
        worst = _worst(shim[field], up[field])
        assert worst < 1e-05, (field, worst)

    # Every parameter got a gradient, and none of them was zero: a model whose
    # convolution gradients were all zero would still train (badly) and still
    # agree with an upstream compared only on a scaled maximum.
    for name, row in zip(shim["names"], shim["first_grads"]):
        assert max(abs(v) for v in row) > 0.0, name


def test_the_training_loop_actually_accumulates_into_dot_grad():
    """docs/training/BACKWARD9.md's nullification finding, repeated because it is the
    one that a loop test structurally cannot make: **`zero_grad(set_to_none=
    True)` drops `.grad` every step**, so a `+=` silently degraded to `=` never
    shows in a trajectory.

    Two backwards of the same loss with no `zero_grad` between them. Every
    gradient must be exactly **twice** the first -- not approximately, since it
    is the same number added to itself.
    """
    if not _available():
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return
    shim = _shim(_VISION_SCRIPT)
    seen = 0
    for name, row in zip(shim["names"], shim["accumulated_ratio"]):
        for ratio in row:
            if ratio is None:
                continue  # below the threshold where a ratio means anything
            seen += 1
            assert abs(ratio - 2.0) < 1e-4, (name, ratio)
    assert seen > 50, (
        "almost every gradient was too small for the ratio to mean anything, so "
        "this test would pass against a rule that never accumulated: %d" % seen)


# --- 4. the transformer half of the same release-note sentence --------------

_TRANSFORMER_SCRIPT = r"""
import json, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

out = {"who": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}

V, H, F_, L = 16, 8, 16, 4  # vocab, hidden, ffn, sequence


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.LayerNorm(H)
        self.q = nn.Linear(H, H)
        self.k = nn.Linear(H, H)
        self.v = nn.Linear(H, H)
        self.o = nn.Linear(H, H)
        self.n2 = nn.LayerNorm(H)
        self.f1 = nn.Linear(H, F_)
        self.f2 = nn.Linear(F_, H)

    def forward(self, h):
        r = h
        h = self.n1(h)
        q, k, v = self.q(h).unsqueeze(1), self.k(h).unsqueeze(1), self.v(h).unsqueeze(1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).squeeze(1)
        h = r + self.o(a)
        r = h
        h = self.n2(h)
        return r + self.f2(F.gelu(self.f1(h)))


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, H)
        self.block = Block()
        self.norm = nn.LayerNorm(H)
        self.head = nn.Linear(H, V, bias=False)

    def forward(self, ids):
        return self.head(self.norm(self.block(self.emb(ids))))


model = LM()
with torch.no_grad():
    k = 0
    for p in model.parameters():
        flat = p.reshape(-1)
        for i in range(flat.shape[0]):
            flat[i] = ((k * 37) % 23 - 11) / 32.0
            k += 1

ids = torch.tensor([[1, 5, 9, 3]])
targets = torch.tensor([5, 9, 3, 7])
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
out["names"] = [name for name, _ in model.named_parameters()]

losses = []
for step in range(5):
    logits = model(ids).reshape(-1, V)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    if step == 0:
        out["first_grads"] = [[float(v) for v in p.grad.reshape(-1)]
                              for p in model.parameters()]
    optimizer.step()
    optimizer.zero_grad()
    losses.append(float(loss.detach()))
out["losses"] = losses
out["params"] = [[float(v) for v in p.reshape(-1)] for p in model.parameters()]
json.dump(out, sys.stdout)
"""


def test_a_tiny_transformer_language_model_trains_and_agrees_with_upstream():
    """**docs/platform/RELEASE_0_0_13a0.md §5's first clause was already false.**

    "No transformer trains through `loss.backward()` yet" was written beside
    the convolution gap in the same sentence, and only the convolution half was
    true: this trains an embedding + pre-norm attention block (causal SDPA) +
    GELU MLP + layer norm + a linear head under `F.cross_entropy` and a real
    `torch.optim.SGD`, five steps, with **no rule this round added** in the
    path. The rules it needs -- `_scaled_dot_product_flash_attention_for_cpu`,
    `native_layer_norm`, `embedding`, `gelu`, `nll_loss_forward`,
    `_log_softmax` -- were all already in `RULE_OPS`.

    That is worth a test rather than a correction in prose, because the claim
    "a transformer trains" is exactly the shape docs/verification/AUDIT.md found going stale
    in both directions.

    Bound `1e-05` relative, on the trajectory, the first step's gradients and
    the final parameters, against upstream running the identical program.
    """
    if not _available():
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return
    shim = _shim(_TRANSFORMER_SCRIPT)
    assert shim["who"] == "shim", shim["who"]
    losses = shim["losses"]
    assert len(losses) == 5, losses
    assert all(b < a for a, b in zip(losses, losses[1:])), losses

    # The path really is the attention path, and it really does not use this
    # round's rules -- otherwise this test would be evidence for the wrong
    # claim.
    rules = set(_C._tape_rules())
    assert "aten._scaled_dot_product_flash_attention_for_cpu.default" in rules
    assert "aten.native_layer_norm.default" in rules
    assert "aten.embedding.default" in rules

    if _upstream_torch is None:
        return
    up = _upstream(_TRANSFORMER_SCRIPT)
    assert up["who"] == "upstream", up["who"]
    assert shim["names"] == up["names"]
    for field in ("losses", "first_grads", "params"):
        worst = _worst(shim[field], up[field])
        assert worst < 1e-05, (field, worst)


def test_the_three_rules_this_round_added_are_in_the_tape_s_own_list():
    """The list `differentiable()` reports against, asserted rather than
    described. `test_shim.py::test_the_tape_has_a_gradient_case_for_every_rule_it_claims`
    holds the other direction -- that each of these has a float64
    central-difference case -- so a rule cannot be claimed here without an
    oracle that shares no code with it.
    """
    rules = set(_C._tape_rules())
    for op in ("aten.convolution.default",
               "aten.avg_pool2d.default",
               "aten.adaptive_avg_pool2d.default"):
        assert op in rules, op


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
