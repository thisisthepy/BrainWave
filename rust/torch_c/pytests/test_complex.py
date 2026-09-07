"""Complex tensors: the imaginary part survives, and everything else refuses.

`docs/kernels/COMPLEX.md` sized this and left it unstarted because `tensor.rs` was
another round's file; `docs/kernels/COMPLEX2.md` is the round that built it. What
landed is `Repr::Complex { re, im }` -- a *pair* of real candle tensors, not
interleaved storage -- plus twelve ops taught that arm by name.

**The failure this file exists to catch is not a crash.** A complex tensor that
silently loses its imaginary part still returns plausible numbers: right shape,
right dtype, right order of magnitude. It survives a smoke test. So every
positive assertion below is **element-wise against upstream torch**, run in a
separate process, on *both* components -- and the tests that matter most are the
ones that would still pass if `im` were quietly dropped, which is why none of
them is a shape check.

Two things are asserted that are not "does it compute":

1.  **The refusals.** ~400 sites read a tensor's storage through
    `PyTensorBase::tensor()`, which refuses on the complex arm. That is what
    makes dropping the imaginary part unrepresentable rather than merely
    avoided (docs/devices/VULKAN2.md's standard). `test_every_untaught_op_refuses`
    walks a sample of them; nullifying that arm in `tensor.rs` (returning `re`)
    turns it red, which was checked rather than assumed.
2.  **The narrowing.** Upstream's `view_as_complex` genuinely aliases its base;
    a pair copies. `llama4` does not depend on it, but it is written down and
    tested *as a narrowing* rather than left for someone to discover.

Nothing here needs numpy or a network. It does need the vendored shim
(`vendor/install_shim.sh`); without it every probe skips, the same way
`test_tail2.py` skips.
"""

import json
import math
import os
import subprocess
import sys

from test_shim import _C

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

# float32 arithmetic on two sides that do not share a libm. Measured: candle's
# `cos` gives 0.99999994 for cos(0.0) where upstream gives exactly 1.0, so an
# exact comparison would fail on `polar` for a reason that has nothing to do
# with the representation. 1e-6 is ~8 ulp at unit magnitude and is far tighter
# than any imaginary part being dropped (which moves a value by its whole
# magnitude, not by an ulp).
_TOL = 1e-6


# ---------------------------------------------------------------------------
# The oracle: one script, run on both libraries, compared
# ---------------------------------------------------------------------------
#
# Every complex value is reported through `torch.view_as_real(...).tolist()`,
# which exists on both sides and is the *only* spelling that shows both halves.
# Reporting `.real` alone is precisely the read that cannot see the bug.
_PROBE = r"""
import json, sys
import torch

MARKER = "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"

def pair(z):
    return torch.view_as_real(z).tolist()

out = {"_marker": MARKER}

# 1. round trip, with an imaginary part that is nowhere near the real one
base = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
z = torch.view_as_complex(base)
out["rt_shape"] = list(z.shape)
out["rt_dtype"] = str(z.dtype)
out["rt_element_size"] = z.element_size()
out["rt_numel"] = z.numel()
out["rt_pair"] = pair(z)
out["rt_real"] = torch.real(z).tolist()
out["rt_imag"] = torch.imag(z).tolist()
out["rt_back"] = torch.view_as_real(z).tolist()

# 2. polar
angle = torch.tensor([0.0, 1.0, 2.0, -0.5])
mag = torch.tensor([1.0, 2.0, 0.5, 3.0])
p = torch.polar(mag, angle)
out["polar_pair"] = pair(p)
out["polar_dtype"] = str(p.dtype)

# 3. complex * complex -- the op where a dropped imaginary part is not merely
#    a missing column but a *wrong real part*, since re' = ac - bd.
q = torch.view_as_complex(torch.tensor([[0.5, -1.5], [2.0, 0.25], [-3.0, 1.0],
                                        [1.0, 1.0]]))
out["mul_pair"] = pair(p * q)
out["mul_scalar_pair"] = pair(q * 2.5)

# 4. the `llama4` rope pipeline end to end, at its real shapes
torch.manual_seed(0)
B, S, H, D = 2, 3, 4, 8
xq = torch.arange(B * S * H * D, dtype=torch.float32).reshape(B, S, H, D) / 97.0
freqs = torch.arange(B * S * D // 2, dtype=torch.float32).reshape(B, S, D // 2) / 11.0
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
out["rope_freqs_shape"] = list(freqs_cis.shape)
out["rope_xq_shape"] = list(xq_.shape)
indexed = freqs_cis[:, :, None, :]
out["rope_indexed_shape"] = list(indexed.shape)
xq_out = torch.view_as_real(xq_ * indexed).flatten(3)
out["rope_out_shape"] = list(xq_out.shape)
out["rope_out"] = xq_out.flatten().tolist()
out["rope_sum"] = float(xq_out.sum())
out["rope_abs_sum"] = float(xq_out.abs().sum())

# 5. copy_ and detach, the two the sweep found
dst = torch.polar(torch.zeros(4), torch.zeros(4))
dst.copy_(p)
out["copy_pair"] = pair(dst)
out["detach_pair"] = pair(p.detach())

# 6. the refusals upstream itself makes, transcribed rather than invented
def refusal(fn):
    try:
        fn()
    except Exception as e:
        return {"raised": type(e).__name__, "msg": str(e)}
    return {"ok": True}

out["ref_odd"] = refusal(lambda: torch.view_as_complex(torch.ones(3, 3)))
out["ref_int"] = refusal(
    lambda: torch.view_as_complex(torch.ones(3, 2, dtype=torch.long)))
out["ref_var_real"] = refusal(lambda: torch.view_as_real(torch.ones(3)))
out["ref_polar_mismatch"] = refusal(
    lambda: torch.polar(torch.ones(3), torch.ones(3, dtype=torch.float64)))

# 7. the narrowing: does `view_as_complex` alias its base?
b2 = torch.tensor([[1., 2.]])
v2 = torch.view_as_complex(b2)
b2[0, 0] = 99.
out["aliases_its_base"] = pair(v2)[0][0] == 99.0

# 8. every name `_complex_ops()` advertises, asked whether it actually answers
#    for a complex tensor. Derived rather than restated: the point is to catch
#    the list drifting in either direction (docs/kernels/COMPLEX3.md §8.1 drifted short
#    and nothing failed, because "sorted, unique, reachable" is true of a list
#    that is simply incomplete).
if hasattr(torch._C, "_complex_ops"):
    z8 = torch.view_as_complex(torch.tensor([[1., -2.], [3., -4.], [5., -6.]]))
    probes = {
        "aten.alias.default":            lambda: torch.ops.aten.alias(z8),
        "aten.clone.default":            lambda: z8.clone(),
        "aten.contiguous.default":       lambda: z8.contiguous(),
        "aten.copy_.default":            lambda: z8.clone().copy_(z8),
        "aten.detach.default":           lambda: z8.detach(),
        "aten.imag.default":             lambda: torch._C._aten_dispatch("aten.imag.default", z8),
        "aten.real.default":             lambda: torch._C._aten_dispatch("aten.real.default", z8),
        "aten.lift_fresh.default":       lambda: torch.ops.aten.lift_fresh(z8),
        "aten.mul.Scalar":               lambda: z8 * 2,
        "aten.mul.Tensor":               lambda: z8 * z8,
        "aten.polar.default":            lambda: torch.polar(torch.ones(3), torch.ones(3)),
        "aten.unsqueeze.default":        lambda: z8.unsqueeze(0),
        "aten.view_as_complex.default":  lambda: torch.view_as_complex(torch.ones(2, 2)),
        "aten.view_as_real.default":     lambda: torch.view_as_real(z8),
        "aten._to_copy.default":         lambda: torch.ones(3).to(torch.complex64),
        "aten.slice.Tensor":             lambda: z8[0:2],
        "aten.constant_pad_nd.default":  lambda: torch.nn.functional.pad(z8, (0, 1)),
        "aten.view.default":             lambda: z8.view(3, 1),
        "aten._unsafe_view.default":     lambda: torch._C._aten_dispatch("aten._unsafe_view.default", z8, [3, 1]),
        "aten.complex.default":          lambda: torch.complex(torch.ones(3), torch.ones(3)),
    }
    # Three of these go through `_aten_dispatch` rather than a Python
    # spelling, and the first version of this probe did not: it called
    # `z8.imag`, `z8.real` and `z8.reshape(...)` and reported all three as
    # refusing. They do -- but the *bindings* refuse, not the ops, and
    # `_complex_ops()` is a list of **ops**. Probing the spelling would have
    # made this check demand three names be removed from a list they belong on.
    # (`Tensor.real` and `Tensor.imag` really are missing; docs/kernels/COMPLEX3.md
    # §6.2 routes them, and `fnet` waits on `real`.)
    answers = {}
    for name in torch._C._complex_ops():
        fn = probes.get(name)
        if fn is None:
            answers[name] = None          # no probe written; neither claim made
            continue
        try:
            fn()
            answers[name] = True
        except Exception:
            answers[name] = False
    out["complex_ops_answer"] = answers

json.dump(out, sys.stdout)
"""

_cache = {}


def _run(side):
    """`_PROBE`'s result from `side` ("shim" or "upstream"), or `None` to skip.

    Two processes, never one: this interpreter has real torch installed (the
    suite's venv does, for exactly this), so `import torch` here would import
    upstream and every "shim" assertion below would be made about the wrong
    library. The marker is checked on both sides so a mis-wired environment
    fails loudly instead of comparing something against itself.
    """
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
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True, env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"the {side} probe failed to run at all:\n{proc.stderr[-3000:]}"
    )
    data = json.loads(proc.stdout)
    assert data["_marker"] == side, (
        f"the {side} probe imported the other library ({data['_marker']}) -- "
        f"every comparison below would have been about the wrong torch"
    )
    _cache[side] = data
    return data


def _both():
    s = _run("shim")
    if s is None:
        return None, None
    return s, _run("upstream")


def _flat(x):
    if isinstance(x, list):
        for i in x:
            yield from _flat(i)
    else:
        yield x


def _close(a, b, key):
    fa, fb = list(_flat(a)), list(_flat(b))
    assert len(fa) == len(fb), f"{key}: {len(fa)} values vs {len(fb)}"
    assert fa, f"{key}: nothing was compared, which is not a passing test"
    for i, (x, y) in enumerate(zip(fa, fb)):
        assert math.isclose(x, y, rel_tol=_TOL, abs_tol=_TOL), (
            f"{key}[{i}]: shim {x!r} vs upstream {y!r}"
        )


# ---------------------------------------------------------------------------
# 1. The bar: the imaginary part survives a round trip
# ---------------------------------------------------------------------------


def test_view_as_complex_round_trip_keeps_the_imaginary_part():
    """`view_as_complex` -> `view_as_real`, element-wise against upstream.

    **This is the test the whole round is for.** A `Repr::Complex` arm that
    dropped `im` -- or a `tensor()` that returned `re` for it -- would give a
    round trip of `[[1,0],[3,0],[5,0]]`: same shape, same dtype, plausible
    values. So the imaginary column is asserted by value, separately, in
    addition to the whole-tensor comparison.
    """
    shim, up = _both()
    if shim is None:
        return
    _close(shim["rt_back"], up["rt_back"], "view_as_real(view_as_complex(x))")
    _close(shim["rt_pair"], up["rt_pair"], "pair")
    _close(shim["rt_imag"], up["rt_imag"], "imag")
    _close(shim["rt_real"], up["rt_real"], "real")
    # Said again as a value rather than a comparison, so that a probe which
    # somehow compared upstream against itself still fails here.
    assert shim["rt_imag"] == [2.0, 4.0, 6.0], shim["rt_imag"]
    assert any(v != 0.0 for v in shim["rt_imag"]), (
        "every imaginary component is zero -- that is what a dropped "
        "imaginary part looks like, and it is why this test does not stop at "
        "comparing shapes"
    )


def test_the_complex_tensors_metadata_matches_upstream():
    """Shape, dtype, `numel` and `element_size`.

    `element_size` is the one worth spelling out: `complex64` is **8** bytes,
    not 4. If it is ever 4, someone aliased the tag onto `float32` -- and
    `numel * element_size` is how upstream code sizes a buffer, so that read
    would be wrong by half rather than absent.
    """
    shim, up = _both()
    if shim is None:
        return
    for key in ("rt_shape", "rt_dtype", "rt_numel", "rt_element_size", "polar_dtype"):
        assert shim[key] == up[key], f"{key}: {shim[key]!r} vs {up[key]!r}"
    assert shim["rt_element_size"] == 8
    # `numel` counts complex elements, not floats: 3, not 6.
    assert shim["rt_numel"] == 3


def test_polar_and_complex_multiplication_agree_with_upstream():
    """`polar`, `z * w` and `z * s`, both components, element-wise.

    Multiplication is the sharpest of the three: `(a+bi)(c+di)` has real part
    `ac - bd`, so an implementation that dropped the imaginary part would get
    the **real** part wrong too. A comparison that only looked at `.real` would
    still catch this one, which is why it is here and not only the round trip.
    """
    shim, up = _both()
    if shim is None:
        return
    _close(shim["polar_pair"], up["polar_pair"], "polar")
    _close(shim["mul_pair"], up["mul_pair"], "z * w")
    _close(shim["mul_scalar_pair"], up["mul_scalar_pair"], "z * 2.5")
    # The cross term is real: if `bd` were dropped, these would be equal.
    a, b = shim["mul_pair"][0]
    assert abs(b) > _TOL, "the product's imaginary part is zero; check the fixture"


def test_the_llama4_rope_pipeline_matches_upstream_element_wise():
    """`polar` -> `view_as_complex` -> `[:, :, None, :]` -> `*` ->
    `view_as_real` -> `flatten(3)`, at `llama4`'s own shapes.

    This is the closed pipeline `docs/kernels/COMPLEX.md` §3 scoped the round to, run
    end to end and compared **every element**, not a checksum. The two sums are
    kept as well, because a checksum that agrees while the elements do not is a
    different defect (a permutation) and one worth being able to tell apart.
    """
    shim, up = _both()
    if shim is None:
        return
    for key in ("rope_freqs_shape", "rope_xq_shape", "rope_indexed_shape",
                "rope_out_shape"):
        assert shim[key] == up[key], f"{key}: {shim[key]} vs {up[key]}"
    assert shim["rope_out_shape"] == [2, 3, 4, 8], shim["rope_out_shape"]
    _close(shim["rope_out"], up["rope_out"], "rope elementwise")
    assert math.isclose(shim["rope_sum"], up["rope_sum"], rel_tol=1e-5, abs_tol=1e-4)
    assert math.isclose(
        shim["rope_abs_sum"], up["rope_abs_sum"], rel_tol=1e-5, abs_tol=1e-4)
    # `abs_sum` >= |sum| always; if the imaginary half were zeroed the two
    # would move together in a way this catches only because both are kept.
    assert shim["rope_abs_sum"] >= abs(shim["rope_sum"])


def test_copy_and_detach_preserve_both_halves():
    """The two ops `arch_sweep` found, which no amount of reading the model
    source had predicted: `nn.Buffer` calls `data.detach()`, and
    `_init_weights` refills the buffer with `init.copy_`. Both are compared
    against upstream rather than merely not raising.
    """
    shim, up = _both()
    if shim is None:
        return
    _close(shim["copy_pair"], up["copy_pair"], "dst.copy_(src)")
    _close(shim["detach_pair"], up["detach_pair"], "z.detach()")


# ---------------------------------------------------------------------------
# 2. The narrowing, stated rather than discovered
# ---------------------------------------------------------------------------


def test_view_as_complex_copies_where_upstream_aliases():
    """**Upstream's `view_as_complex` is a view; this one is a copy.**

    Measured on 2.13.0: `base[0,0] = 99.` shows through the complex tensor.
    A pair-of-tensors representation cannot do that, and choosing the pair is
    what bought the correct `.shape` (docs/kernels/COMPLEX.md §3.2).

    Asserted as a *divergence* -- the two sides are required to disagree --
    rather than skipped, so that it fails if either side changes. If the shim
    ever starts aliasing, this test is where the narrowing gets deleted; if
    upstream ever stops, it is where the note stops being true.
    """
    shim, up = _both()
    if shim is None:
        return
    assert up["aliases_its_base"] is True, (
        "upstream's view_as_complex stopped aliasing its base -- COMPLEX.md "
        "§3.3's measurement is stale"
    )
    assert shim["aliases_its_base"] is False, (
        "the shim's view_as_complex now aliases its base. That is upstream's "
        "behaviour and an improvement -- but docs/kernels/COMPLEX2.md and "
        "docs/kernels/VIEWS.md record the copy as a narrowing, so remove it there "
        "before removing it here."
    )


# ---------------------------------------------------------------------------
# 3. The refusals -- most of the work, and the part that must not rot
# ---------------------------------------------------------------------------


def test_upstream_refusals_are_reproduced_verbatim():
    """The four error messages upstream raises, byte for byte.

    Transcribed from upstream in `docs/kernels/COMPLEX.md` §3.3 item 3 and now checked
    against a live upstream process rather than against the transcription, so
    they cannot drift.
    """
    shim, up = _both()
    if shim is None:
        return
    for key in ("ref_odd", "ref_int", "ref_var_real", "ref_polar_mismatch"):
        assert "raised" in up[key], f"upstream no longer refuses {key}: {up[key]}"
        assert "raised" in shim[key], f"the shim computed {key}: {shim[key]}"
        assert shim[key]["msg"] == up[key]["msg"], (
            f"{key}:\n  shim     {shim[key]['msg']!r}\n"
            f"  upstream {up[key]['msg']!r}"
        )


_UNTAUGHT = """
import json, sys
import torch
assert hasattr(torch._C, "_aten_implemented")
z = torch.view_as_complex(torch.tensor([[1., 2.], [3., 4.]]))
PROBES = {
    "sum": lambda: z.sum(),
    "add": lambda: z + z,
    "abs": lambda: z.abs(),
    "tolist": lambda: z.tolist(),
    "matmul": lambda: z @ z,
    "reshape": lambda: z.reshape(2, 1),
    "to_float32": lambda: z.to(torch.float32),
    "select": lambda: z[0],
    "cat": lambda: torch.cat([z, z]),
}
out = {}
for name, fn in PROBES.items():
    try:
        r = fn()
    except Exception as e:
        out[name] = {"raised": type(e).__name__, "msg": str(e)}
    else:
        out[name] = {"ok": repr(r)[:200]}
json.dump(out, sys.stdout)
"""


def test_every_untaught_op_refuses_rather_than_dropping_the_imaginary_part():
    """**The property the whole representation rests on**, sampled.

    `Repr::Complex` is refused by `PyTensorBase::tensor()`, which is how ~400
    kernels read their inputs. So an op that was never taught the pair cannot
    quietly receive the real half: it gets a `PyResult` whose only content is a
    refusal. That is `docs/devices/VULKAN2.md`'s standard -- the wrong answer is
    unrepresentable, not merely avoided.

    Nine ops are sampled here, chosen because each one would return a
    *plausible* answer from `re` alone: `sum`, `abs` and `matmul` would be
    numerically wrong with no shape to give it away, and `reshape` would be
    right in shape and silently half the data.

    **`slice` was the tenth and has been inverted out of this list**, not
    deleted from the suite. docs/kernels/COMPLEX3.md taught `aten.slice.Tensor` the
    complex arm because `docs/bindings/BIND3.md` §6 measured it as one of the two ops
    that put `fft_fftn`'s `s=` argument out of reach, and it is now proven
    element-wise against upstream on *both* components -- including two strided
    forms -- in `pytests/test_cplx2.py::test_slice_keeps_the_imaginary_part_on_every_form`.
    `reshape` and `select` stay here deliberately: they are the untaught
    neighbours of the two ops that round taught (`view` and `slice`), so they
    are what would go red if a guard had been written at the wrong level.

    **Nullified and confirmed red**: changing `tensor()`'s complex arm to
    `Ok(re)` in `tensor.rs` makes the sampled ops return values instead of
    raising, and this test reports every one. It is not a formality.
    """
    if not os.path.isfile(_VENDOR_SHIM):
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _UNTAUGHT],
        capture_output=True, text=True, env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    data = json.loads(proc.stdout)
    computed = sorted(k for k, v in data.items() if "ok" in v)
    assert not computed, (
        f"these ops computed something on a complex tensor without being "
        f"taught the representation: {computed}. Whatever they returned was "
        f"derived from the real part alone -- see this module's docstring. If "
        f"one of them was genuinely implemented, move it out of this list and "
        f"give it an element-wise comparison against upstream above."
    )
    for name, r in sorted(data.items()):
        assert r["raised"] == "NotImplementedError", f"{name}: {r}"
        assert "complex64" in r["msg"], (
            f"{name} refused without naming the dtype, so the reader cannot "
            f"tell whether the gap is the dtype or the operator: {r['msg']}"
        )


def test_the_refusal_points_at_a_list_that_is_not_stale():
    """`no_real_storage`'s message says "the ops taught the complex
    representation by name (torch._C._complex_ops())".

    A refusal that names a stale list is worse than one that names none --
    this repository has been bitten by exactly that (`overloads.json`'s note on
    `min.other`). So the list is a constant in `tensor.rs`, exported, and
    checked here against the dispatch table it claims to describe.
    """
    ops = list(_C._complex_ops())
    assert ops == sorted(ops), "COMPLEX_OPS is not sorted"
    assert len(set(ops)) == len(ops), "COMPLEX_OPS has a duplicate"
    reachable = set(_C._aten_implemented()) | set(_C._aten_implemented_awaiting_golden())
    missing = [op for op in ops if op not in reachable]
    assert not missing, (
        f"_complex_ops() names ops the dispatcher does not answer at all: "
        f"{missing}"
    )
    # The five that only exist for complex are parked, not advertised: golden
    # compares by reading both sides as real tensors and there is nothing to
    # read on a complex one. See aten.rs's note on IMPLEMENTED_AWAITING_GOLDEN.
    parked = set(_C._aten_implemented_awaiting_golden())
    for op in ("aten.view_as_complex.default", "aten.view_as_real.default",
               "aten.polar.default", "aten.real.default", "aten.imag.default"):
        assert op in parked, f"{op} is advertised to golden but has no builder"
        assert op not in set(_C._aten_implemented()), op
    # ...and the guarded ones stay advertised, because their dense path is
    # unchanged and still golden-compared.
    for op in ("aten.mul.Tensor", "aten.detach.default", "aten.copy_.default"):
        assert op in set(_C._aten_implemented()), (
            f"{op} left _aten_implemented(). Its dense path is unchanged by "
            f"the complex guard; if it moved, golden stopped comparing it."
        )


# ---------------------------------------------------------------------------
# 4. The tag half, from the storage side
# ---------------------------------------------------------------------------


def test_complex_ops_is_derived_from_behaviour_and_not_only_from_its_own_shape():
    """The staleness check above reads the list's *shape*; this reads its truth.

    `docs/kernels/COMPLEX3.md` §8.1 taught five more ops and the list did not follow.
    Nothing failed, because sorted-unique-reachable is true of a list that is
    simply short -- and a refusal naming a list shorter than the truth tells a
    user an op is unavailable when it works, which is the same class of harm as
    naming one that is too long.

    The `strided` round recommended this and declined to write it, because the
    complex round was editing this file at the time and a collision there would
    have cost more than it saved. It also refused to append the six names in
    its own worktree, where the ops genuinely did refuse -- appending there
    would have produced the staleness *inverted*, a list naming ops that raise.
    Both calls were right, and this is the check that makes the next such
    append unnecessary.

    So: every name on the list must actually answer for a complex tensor, and
    the ops the untaught sweep asserts must refuse must not be on it. Drift in
    **either** direction is red.
    """
    r = _run("shim")
    if r is None:
        return
    taught = set(_C._complex_ops())

    answers = r["complex_ops_answer"]

    # Every name that has a probe must answer. A `None` means no probe was
    # written for it -- recorded, and neither claim is made about it, because
    # asserting on an op nobody exercised would be the shape of check this
    # repository keeps removing.
    refusing = sorted(op for op, ok in answers.items() if ok is False)
    assert not refusing, (
        f"_complex_ops() names ops that refuse on a complex tensor: {refusing}. "
        "The refusal points a reader at this list, so a name here is a promise."
    )
    unprobed = sorted(op for op, ok in answers.items() if ok is None)
    assert not unprobed, (
        f"no probe exists for {unprobed}, so this check says nothing about "
        "them. Add one to _PROBE section 8 rather than leaving the list "
        "partly unchecked."
    )
    assert set(answers) == taught, (sorted(set(answers) ^ taught))

    # The *other* direction -- an op that answers and is missing from the list,
    # which is how docs/kernels/COMPLEX3.md §8.1 drifted -- is
    # `test_every_untaught_op_refuses_rather_than_dropping_the_imaginary_part`
    # above: it asserts each op *not* on the list refuses. Between the two,
    # drift is red either way. Asserting it here would be vacuous, since these
    # answers are gathered by iterating the list itself.


def test_the_complex_tags_still_report_no_candle_storage():
    """**Landing the representation did not make `_has_storage` true**, and it
    must not.

    That flag means "candle can store this dtype", and candle still cannot --
    the pair lives outside candle's `DType` entirely, exactly as
    `Repr::Quantized` does. `test_tail2.py` asserts the same thing from the
    other direction; it is restated here because this is the round that would
    have broken it, and because a reader of *this* file needs to know that
    `torch.zeros(2, dtype=torch.complex64)` still refuses.
    """
    for dt in (_C.complex32, _C.complex64, _C.complex128):
        assert dt._has_storage is False, (
            f"{dt} claims candle storage. Repr::Complex is a pair of real "
            f"tensors outside candle's DType; if this became True, some "
            f"constructor is about to allocate a single real buffer and call "
            f"it complex."
        )
    assert _C.float32._has_storage is True


def test_the_component_mapping_is_the_inverse_of_to_complex():
    """`TorchDType::complex_for_component` (Rust) against `to_complex()`/
    `to_real()` (the Python surface).

    Two tables over the same three pairs, added in different rounds. Checked
    against each other rather than each against a transcription, so neither can
    drift alone.
    """
    for real, cplx in ((_C.float16, _C.complex32),
                       (_C.float32, _C.complex64),
                       (_C.float64, _C.complex128)):
        assert real.to_complex() == cplx
        assert cplx.to_real() == real
        assert cplx.itemsize == 2 * real.itemsize
    # `bfloat16` has no complex partner of its own width, and upstream does
    # not answer the identity for it -- it promotes: `torch.bfloat16
    # .to_complex()` is `torch.complex64`, measured. Pinned here because this
    # table said `bfloat16` until docs/kernels/COMPLEX2.md.
    assert _C.bfloat16.to_complex() == _C.complex64
    # ...and the complex tags map to themselves.
    for cplx in (_C.complex32, _C.complex64, _C.complex128):
        assert cplx.to_complex() == cplx


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
