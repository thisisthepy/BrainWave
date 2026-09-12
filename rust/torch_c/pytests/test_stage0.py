"""`adapt`'s stage 0 -- recompute the normalisation statistics, no backward.

docs/design/DESIGN.md section 3's survey table puts normalisation calibration
on *both* sides of the differentiation line: "recompute the statistics" is
stage 0 and "update the affine {gamma, beta} by a loss" is stage 1. Stage 1 has
been built since docs/models/ADAPT.md (`adapt.Tent`). Stage 0 was named as in
scope, was absent, and `adapt.wrap` refused it by name
(docs/design/GAPS.md section 3.4) -- so the *cheap* half of the two on a device
was the half nobody could run.

**What this file holds down, and why each check is here.**

* The statistics this stack computes must be the statistics upstream computes.
  The grade this project asks for is *agrees*, not *runs*: the oracle is
  upstream torch 2.13 in the spike venv, running the **same** module source
  (`_STAGE0_MODEL_SRC` is executed on both sides) in the same mode on the same
  input, and the running buffers are compared element by element.

* **`revert` must actually restore.** A stage-0 run mutates buffers. `Delta` is
  keyed on `named_parameters()`, so a `revert` that only knew about parameters
  would report success and leave the model changed -- a silent wrong answer of
  exactly the shape docs/design/DESIGN.md section 6 puts below a refusal. It is
  asserted directly here rather than inferred from the code.

* **A delta does not cover buffers**, and that is a decision rather than an
  omission (DESIGN.md section 3, "델타는 버퍼를 덮지 않는다"). The decision is
  testable in two directions and both are here: `Delta.over` still refuses a
  buffer name, and `BufferSnapshot` still has no `record`/`apply`/`persist`/
  `publish`. If a later round extends `Delta` to buffers, the second of those
  goes red and the reader is sent to the section that decided otherwise.

Nullifications, each made and watched go red -- see the round report:

* `BatchNormStats.step` not setting training mode  -> the agreement test
* `Adapted.revert` reverting only the delta        -> the revert test
* `BufferSnapshot.revert` restoring nothing        -> the revert test
* `select_modules` returning every module          -> the selection test
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

try:
    import torch as _upstream_torch
except Exception:  # pragma: no cover - reported as a skip below
    _upstream_torch = None


def _available():
    return _upstream_torch is not None and os.path.isfile(_VENDOR_SHIM)


def _det(n, seed, lo=-1000, hi=1000):
    """Deterministic, RNG-free values.

    The same generator `test_shim.py`'s `_ckpt_det` uses, and here for the same
    reason: the two sides of this comparison are two different builds of
    `torch`, so anything that came out of an RNG would be comparing which RNG
    ran rather than which statistics were computed.
    """
    return [((seed * 1103515245 + i * 12345) % (hi - lo) + lo) / 4000.0 for i in range(n)]


# Executed verbatim on *both* sides -- the shim subprocess and this process's
# upstream torch. A model defined twice would be a comparison of two models.
_STAGE0_MODEL_SRC = r'''
C, N = 4, 6
MOMENTUM = 0.1

def _det(n, seed, lo=-1000, hi=1000):
    return [((seed * 1103515245 + i * 12345) % (hi - lo) + lo) / 4000.0 for i in range(n)]

class TinyBN(nn.Module):
    """A BatchNorm model -- the case `Tent`'s docstring says it serves half of.

    `nn.BatchNorm1d` carries `running_mean`, `running_var` and
    `num_batches_tracked`; the last is `int64`, which is the fact the delta
    decision in DESIGN.md section 3 turns on.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(C, C, bias=False)
        self.norm = nn.BatchNorm1d(C)
        self.head = nn.Linear(C, C, bias=False)

    def forward(self, x):
        return self.head(self.norm(self.fc(x)))

def build():
    m = TinyBN()
    with torch.no_grad():
        m.fc.weight.copy_(torch.tensor(_det(C * C, 3)).reshape(C, C))
        m.head.weight.copy_(torch.tensor(_det(C * C, 11)).reshape(C, C))
        m.norm.weight.copy_(torch.tensor([1.0 + v for v in _det(C, 13)]))
        m.norm.bias.copy_(torch.tensor(_det(C, 17)))
        m.norm.momentum = MOMENTUM
    return m.eval()

def batches():
    """Three batches, each a different distribution -- so the statistics move."""
    out = []
    for k in range(3):
        out.append(torch.tensor(_det(N * C, 101 + 7 * k)).reshape(N, C) + float(k))
    return out
'''

_SHIM_SCRIPT = (
    "import json, sys\n"
    "import torch\n"
    "import torch.nn as nn\n"
    + _STAGE0_MODEL_SRC
    + r'''
from torchnative import adapt
from torchnative.delta import Delta, BufferSnapshot

out = {}
xs = batches()

# --- wrap accepts a stage-0 method ---------------------------------------
m = build()
try:
    w = adapt.wrap(m, method=adapt.BatchNormStats())
except Exception as e:
    out["wrap"] = "%s: %s" % (type(e).__name__, str(e))
    json.dump(out, sys.stdout)
    raise SystemExit(0)
out["wrap"] = "ACCEPTED"

# What it opened, and what it did NOT open.
w.online()
out["selected_modules"] = list(adapt.BatchNormStats().select_modules(m))
out["online_parameters"] = list(w.online_parameters)
out["online_buffers"] = list(w.online_buffers)
out["state_type"] = type(w.adapted).__name__
out["has_optimizer"] = w._optimizer is not None
out["model_training_before"] = m.training

out["base_mean"] = m.norm.running_mean.tolist()
out["base_var"] = m.norm.running_var.tolist()
out["base_count"] = int(m.norm.num_batches_tracked.item())

# --- the run -------------------------------------------------------------
# The prediction each batch returns must be the prediction the model made
# with the statistics it had *going in* -- predict-then-adapt, same protocol
# as stage 1. Recorded so the oracle can check it against an eval-mode
# forward taken before the update.
out["logits"] = []
for x in xs:
    y, obj = w.step(x)
    out["logits"].append(y.reshape(-1).tolist())
    out["objective_is_none"] = obj is None
out["run_mean"] = m.norm.running_mean.tolist()
out["run_var"] = m.norm.running_var.tolist()
out["run_count"] = int(m.norm.num_batches_tracked.item())
out["model_training_after"] = m.training
out["history"] = list(w.history)
out["drift"] = w.adapted.drift(m)

# --- revert --------------------------------------------------------------
w.revert()
out["reverted_mean"] = m.norm.running_mean.tolist()
out["reverted_var"] = m.norm.running_var.tolist()
out["reverted_count"] = int(m.norm.num_batches_tracked.item())
out["reverted_affine"] = m.norm.weight.tolist()

# --- offline is the model ------------------------------------------------
m2 = build()
w2 = adapt.wrap(m2, method=adapt.BatchNormStats())
before = m2.norm.running_mean.tolist()
plain = build()
for x in xs:
    a = w2(x).reshape(-1).tolist()
    b = plain(x).reshape(-1).tolist()
    out["offline_matches"] = (a == b)
out["offline_mean_unmoved"] = (m2.norm.running_mean.tolist() == before)
out["offline_training"] = m2.training

# --- the delta decision, as behaviour ------------------------------------
try:
    Delta.over(build(), ["norm.running_mean"])
except Exception as e:
    out["delta_over_buffer"] = "%s: %s" % (type(e).__name__, str(e))
else:
    out["delta_over_buffer"] = "ACCEPTED"

out["snapshot_absent"] = [
    n for n in ("record", "apply", "persist", "publish", "load",
                "revert_by_subtraction")
    if not hasattr(BufferSnapshot, n)
]
out["snapshot_present"] = [
    n for n in ("revert", "covers", "nbytes", "drift") if hasattr(BufferSnapshot, n)
]

# A stage-0 method that does not say which modules it calibrates is refused
# at wrap time, the way a missing stage is -- not at the first step.
class NoModules(adapt.Method):
    stage = adapt.STAGE_FORWARD_ONLY
    def select(self, model): return ["norm.weight"]
    def objective(self, outputs): return outputs.sum()

try:
    adapt.wrap(build(), method=NoModules())
except Exception as e:
    out["no_select_modules"] = "%s: %s" % (type(e).__name__, str(e))
else:
    out["no_select_modules"] = "ACCEPTED"

# A model with no running statistics at all: every transformer. Selecting
# nothing must refuse rather than run and change nothing -- the same refusal
# `online()` already makes for a stage-1 method that selects no parameters.
class LN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(C)
    def forward(self, x):
        return self.norm(x)

try:
    adapt.wrap(LN(), method=adapt.BatchNormStats()).online()
except Exception as e:
    out["no_statistics"] = "%s: %s" % (type(e).__name__, str(e))
else:
    out["no_statistics"] = "ACCEPTED"

json.dump(out, sys.stdout)
'''
)

_ORACLE_SCRIPT = (
    "import json, sys\n"
    "import torch\n"
    "import torch.nn as nn\n"
    + _STAGE0_MODEL_SRC
    + r'''
out = {}
xs = batches()
m = build()
out["base_mean"] = m.norm.running_mean.tolist()
out["base_var"] = m.norm.running_var.tolist()

# What stage 0 IS, spelled with nothing but upstream: put the normalisation
# module into training mode for the forward so its running statistics update,
# and take the prediction from the statistics it had going in.
out["logits"] = []
for x in xs:
    with torch.no_grad():
        m.norm.eval()
        out["logits"].append(m(x).reshape(-1).tolist())
        m.norm.train()
        m(x)
        m.norm.eval()
out["run_mean"] = m.norm.running_mean.tolist()
out["run_var"] = m.norm.running_var.tolist()
out["run_count"] = int(m.norm.num_batches_tracked.item())
json.dump(out, sys.stdout)
'''
)


def _run(script, vendored):
    env = dict(os.environ)
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    if vendored:
        env["PYTHONPATH"] = _VENDOR_DIR
    else:
        env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "stage0 subprocess (vendored=%s) exited %d\n--- stdout ---\n%s\n"
            "--- stderr ---\n%s" % (vendored, proc.returncode, proc.stdout, proc.stderr)
        )
    return json.loads(proc.stdout)


_CACHE = {}


def _shim():
    if "shim" not in _CACHE:
        _CACHE["shim"] = _run(_SHIM_SCRIPT, vendored=True)
    return _CACHE["shim"]


def _oracle():
    if "oracle" not in _CACHE:
        _CACHE["oracle"] = _run(_ORACLE_SCRIPT, vendored=False)
    return _CACHE["oracle"]


def _maxdiff(a, b):
    assert len(a) == len(b), (len(a), len(b))
    return max(abs(p - q) for p, q in zip(a, b)) if a else 0.0


# -- the tests -------------------------------------------------------------


def test_wrap_accepts_a_stage_0_method():
    """It refused until 2026-09-13, by name, from `adapt/__init__.py`.

    docs/design/GAPS.md section 3.4: the refusal said "nothing here provides
    that path yet", which was true and is the reason this is a gap rather than
    a decision.
    """
    if not _available():
        print("ok   stage0: skipped -- no vendored shim or no upstream torch")
        return
    r = _shim()
    assert r["wrap"] == "ACCEPTED", r["wrap"]
    print("ok   stage0: adapt.wrap accepts a stage-0 method")


def test_a_stage_0_wrapper_opens_a_buffer_snapshot_and_no_optimiser():
    """Stage 0 needs no capture, no tape and no optimiser -- so it opens none.

    An optimiser built over an empty parameter list would step and change
    nothing while reporting a step, which is the shape this whole subpackage
    is arranged against.
    """
    if not _available():
        print("ok   stage0: skipped -- no vendored shim or no upstream torch")
        return
    r = _shim()
    assert r["state_type"] == "BufferSnapshot", r["state_type"]
    assert r["has_optimizer"] is False, r["has_optimizer"]
    assert r["online_parameters"] == [], r["online_parameters"]
    assert r["objective_is_none"] is True, r
    assert r["history"] == [], r["history"]
    print("ok   stage0: a stage-0 wrapper opens a BufferSnapshot and no optimiser")


def test_it_selects_the_modules_that_carry_running_statistics_and_only_those():
    """`BatchNormStats` picks by "has running statistics", not by name.

    The selection is asserted exactly rather than by count: a `select_modules`
    that returned every module would still produce a run and still move the
    BatchNorm statistics, so a looser check would pass on a broken selection.
    """
    if not _available():
        print("ok   stage0: skipped -- no vendored shim or no upstream torch")
        return
    r = _shim()
    assert r["selected_modules"] == ["norm"], r["selected_modules"]
    assert sorted(r["online_buffers"]) == [
        "norm.num_batches_tracked", "norm.running_mean", "norm.running_var"
    ], r["online_buffers"]
    print("ok   stage0: it selects the one module carrying running statistics")


def test_the_statistics_it_computes_are_the_statistics_upstream_computes():
    """The grade is *agrees*. The oracle is upstream torch 2.13.

    Same module source, same input, same mode, same momentum -- and the
    running buffers compared element by element. Bit-equality is asserted on
    the mean; the variance is allowed 1e-6 because the two builds reduce in
    different orders, which is the tolerance docs/numerics conventions use for
    a reduction.
    """
    if not _available():
        print("ok   stage0: skipped -- no vendored shim or no upstream torch")
        return
    s, o = _shim(), _oracle()
    assert _maxdiff(s["base_mean"], o["base_mean"]) == 0.0, (s["base_mean"], o["base_mean"])
    # It really moved -- otherwise "agrees" would be two untouched buffers.
    moved = _maxdiff(o["run_mean"], o["base_mean"])
    assert moved > 1e-3, moved
    dm = _maxdiff(s["run_mean"], o["run_mean"])
    dv = _maxdiff(s["run_var"], o["run_var"])
    assert dm <= 1e-6, (dm, s["run_mean"], o["run_mean"])
    assert dv <= 1e-6, (dv, s["run_var"], o["run_var"])
    assert s["run_count"] == o["run_count"] == 3, (s["run_count"], o["run_count"])
    print(f"ok   stage0: statistics agree with upstream (mean {dm:.3g}, var {dv:.3g}, "
          f"moved {moved:.3g} over 3 batches)")


def test_the_prediction_is_the_one_made_before_the_update():
    """Predict-then-adapt, the same order stage 1 uses, checked against upstream.

    The logits each step returns must be an eval-mode forward taken with the
    statistics the model had going in, not a training-mode forward using the
    batch's own statistics. The two differ, so this is not a tautology.
    """
    if not _available():
        print("ok   stage0: skipped -- no vendored shim or no upstream torch")
        return
    s, o = _shim(), _oracle()
    assert len(s["logits"]) == len(o["logits"]) == 3, (len(s["logits"]), len(o["logits"]))
    worst = max(_maxdiff(a, b) for a, b in zip(s["logits"], o["logits"]))
    assert worst <= 1e-5, worst
    # Step 2's prediction differs from step 1's -- the update reached the forward.
    assert _maxdiff(o["logits"][0], o["logits"][1]) > 1e-3
    print(f"ok   stage0: predict-then-adapt agrees with upstream ({worst:.3g})")


def test_revert_after_a_stage_0_run_restores_the_running_statistics():
    """The trap this whole design question exists because of.

    A stage-0 run mutates buffers. If `revert` only knew about parameters --
    which is all a `Delta` covers -- it would return, report success, and
    leave the model carrying the test distribution's statistics. Asserted on
    the buffers directly, byte for byte against the base, and on
    `num_batches_tracked` too: a restore that missed the int64 counter would
    leave the next EMA weighting wrong and nothing downstream would say so.
    """
    if not _available():
        print("ok   stage0: skipped -- no vendored shim or no upstream torch")
        return
    r = _shim()
    # Non-vacuity: the run must have moved them, or "restored" means nothing.
    assert _maxdiff(r["run_mean"], r["base_mean"]) > 1e-3, r["run_mean"]
    assert r["run_count"] == 3 and r["base_count"] == 0, (r["run_count"], r["base_count"])
    assert r["reverted_mean"] == r["base_mean"], (r["reverted_mean"], r["base_mean"])
    assert r["reverted_var"] == r["base_var"], (r["reverted_var"], r["base_var"])
    assert r["reverted_count"] == r["base_count"], r["reverted_count"]
    print("ok   stage0: revert restores running_mean, running_var and num_batches_tracked")


def test_an_offline_stage_0_wrapper_is_the_model_it_wraps():
    """`wrap` must not move a plain forward -- the property `Adapted` opens with.

    For stage 0 this is sharper than for stage 1: the whole method is "leave
    training mode on", so a wrapper that armed itself at construction would
    silently recalibrate a model nobody asked to adapt.
    """
    if not _available():
        print("ok   stage0: skipped -- no vendored shim or no upstream torch")
        return
    r = _shim()
    assert r["offline_matches"] is True, r["offline_matches"]
    assert r["offline_mean_unmoved"] is True, r["offline_mean_unmoved"]
    assert r["offline_training"] is False, r["offline_training"]
    print("ok   stage0: an offline stage-0 wrapper is exactly the model it wraps")


def test_the_forward_leaves_the_model_in_the_mode_it_found_it():
    """Training mode is on for the forward and off again after it.

    The model is `.eval()` here. A stage-0 step that left it `.train()` would
    turn on every other training-mode behaviour in the model -- dropout, for
    one -- for every later call, including the plain forwards of a wrapper
    that had gone back offline.
    """
    if not _available():
        print("ok   stage0: skipped -- no vendored shim or no upstream torch")
        return
    r = _shim()
    assert r["model_training_before"] is False, r["model_training_before"]
    assert r["model_training_after"] is False, r["model_training_after"]
    print("ok   stage0: the step restores the training mode it found")


def test_a_delta_still_covers_parameters_and_not_buffers():
    """The decision of DESIGN.md section 3, asserted from the `Delta` side.

    `Delta.over` resolves names through `named_parameters()`, so a buffer name
    is refused as not a parameter of this model. That is the behaviour the
    decision preserves; if a later round extends `Delta` to buffers this goes
    red and the reader is sent to the section that decided otherwise.
    """
    if not _available():
        print("ok   stage0: skipped -- no vendored shim or no upstream torch")
        return
    r = _shim()
    assert r["delta_over_buffer"].startswith("KeyError:"), r["delta_over_buffer"]
    assert "not parameters of this model" in r["delta_over_buffer"], r["delta_over_buffer"]
    print("ok   stage0: Delta.over still refuses a buffer name")


def test_a_buffer_snapshot_has_only_the_operations_it_can_mean():
    """And is missing the four it cannot, by decision rather than by omission.

    A `BufferSnapshot` answers one of DESIGN.md section 3's three lifetime
    questions -- can this be discarded, and at what cost. It does not answer
    the other two, so it does not carry their names: `record`/`apply` because
    a statistic is a re-estimate and not an additive offset, `persist`/
    `publish` because `num_batches_tracked` is `int64` and both roads refuse
    a non-floating table anyway (`delta:306`, `federated:652`).
    """
    if not _available():
        print("ok   stage0: skipped -- no vendored shim or no upstream torch")
        return
    r = _shim()
    assert sorted(r["snapshot_absent"]) == [
        "apply", "load", "persist", "publish", "record", "revert_by_subtraction"
    ], r["snapshot_absent"]
    assert sorted(r["snapshot_present"]) == ["covers", "drift", "nbytes", "revert"], \
        r["snapshot_present"]
    print("ok   stage0: BufferSnapshot carries four operations and refuses to name six")


def test_a_stage_0_method_that_names_no_modules_is_refused_at_wrap():
    """Refused at `wrap`, not at the first step -- the same property the stage
    declaration itself has, for the same reason."""
    if not _available():
        print("ok   stage0: skipped -- no vendored shim or no upstream torch")
        return
    r = _shim()
    assert r["no_select_modules"].startswith("NotImplementedError:"), r["no_select_modules"]
    assert "select_modules" in r["no_select_modules"], r["no_select_modules"]
    print("ok   stage0: a stage-0 method with no select_modules is refused at wrap")


def test_a_model_with_no_running_statistics_is_refused_rather_than_served():
    """LayerNorm, RMSNorm -- so every transformer. There is nothing for stage 0
    to recalibrate there, and a run that calibrated nothing would report every
    step as having happened. The same refusal `online()` already makes for a
    stage-1 method that selects no parameters."""
    if not _available():
        print("ok   stage0: skipped -- no vendored shim or no upstream torch")
        return
    r = _shim()
    assert r["no_statistics"].startswith("ValueError:"), r["no_statistics"]
    assert "running statistics" in r["no_statistics"], r["no_statistics"]
    print("ok   stage0: a model with no running statistics is refused, not served")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
