"""Vulkan, widened from four ops to eighteen -- and what the four actually were.

docs/platform/RELEASE_0_1_0b0.md §5 says:

    Vulkan is four ops. Correctness is testable on this host; performance
    needs a phone and has not been measured.

**That sentence was re-measured before it was widened** (docs/devices/VULKAN4.md §1),
because four §5 gap statements in this repository have turned out to be wrong
when re-checked. This one is not wrong. It is *true and misleading*, in a way
that only shows up when you ask each of the four what it does:

  * `aten.add.Tensor`        -- a real SPIR-V compute shader on the GPU.
  * `aten._to_copy.default`  -- a memory copy. No shader, no arithmetic.
  * `aten.detach.default`    -- an `Arc` clone of a shape. No GPU work at all.
  * `aten.alias.default`     -- the same.

So "four ops" counted *reachability* and was read as *computation*, and one of
the four was the whole of the arithmetic. Worse, and not visible from the op
list at all: **there was no way to put arbitrary data on the device.** The only
routes in were `ones`/`zeros`/`empty`, so every Vulkan kernel that had ever
been tested had been tested on constants -- and a matmul of all-ones agrees
with any implementation that sums the right number of ones.

This file holds the widening down. Every assertion here is one of three kinds,
and the kinds are kept apart on purpose:

  * **value** -- element-wise against upstream torch, at a tolerance *derived*
    from upstream's own float32-vs-float64 error the way docs/numerics/AGREE.md §2
    derives its own. For every exactly-rounded op that derivation comes out at
    zero, so those are held to **bit equality**.
  * **device** -- that the GPU did the work, asserted from `_vulkan_counters()`
    at runtime. Not inferred from the answer being right, and deliberately not
    a source scan: docs/devices/MPSATTN.md §3.1 records a way to defeat exactly that
    shape of evidence, and §5 of docs/devices/VULKAN4.md explains why a counter placed
    after `vkWaitForFences` cannot be defeated the same way.
  * **refusal** -- that everything not taught still refuses naming itself, and
    that the narrowings (broadcast, alpha, non-f32, rank > 2) refuse rather
    than reach for the CPU implementation half a metre away.

Everything that needs a Vulkan loader skips **by name**, printing the loader's
own words. docs/devices/VULKAN3.md §6.1 is the reason that matters: macOS strips
`DYLD_*` when `/bin/sh` execs, so running this through `run.sh` skips these
even when the loader was pointed at correctly. A skip that says "no vulkan"
when a loader was supplied is the closest thing to a false green this device
has produced, and it was caught by the skip line quoting the loader.
"""

import json
import math
import os
import struct
import subprocess
import sys

from test_shim import _C


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
VENDOR_DIR = os.path.join(REPO, "torchnative", "src", "main")

# The four ops docs/devices/VULKAN3.md landed, so this file can state the widening as a
# difference rather than a number, and so a later round that removes one is a
# failure here rather than a silently smaller list.
VULKAN3_OPS = (
    "aten._to_copy.default",
    "aten.add.Tensor",
    "aten.alias.default",
    "aten.detach.default",
)

# Of those four, the only one that ever ran a compute shader (docs/devices/VULKAN4.md §1).
VULKAN3_OPS_THAT_COMPUTED = ("aten.add.Tensor",)

FLOAT32_EPS = 2.0 ** -23


# ---------------------------------------------------------------------------
# Skipping by name
# ---------------------------------------------------------------------------

def _vulkan_or_skip(what):
    """A live Vulkan device, or None having said why not -- in the loader's words.

    Deliberately not shared with `test_shim.py`'s helper, for docs/devices/VULKAN3.md
    §6.1's reason: the skip line is the thing this file promises to keep
    truthful, so it names the file that skipped and quotes the probe's own
    `error` rather than paraphrasing it.
    """
    probe = _C._vulkan_probe()
    if not probe["available"]:
        print(f"   (skipped {what}: no vulkan loader here -- {probe['error']})")
        return None
    return probe


def _counters():
    return _C._vulkan_counters()


def _delta(before, after):
    return {k: after[k] - before[k] for k in after}


# ---------------------------------------------------------------------------
# Moving data
# ---------------------------------------------------------------------------

def _cpu(values, shape):
    return _C._tensor_from_flat([float(v) for v in values], list(shape),
                                dtype=_C.float32)


def _to_vulkan(t):
    return _C._aten_dispatch("aten._to_copy.default", t,
                             device=_C.device("vulkan"))


def _to_cpu(t):
    return _C._aten_dispatch("aten._to_copy.default", t, device=_C.device("cpu"))


def _flat(t):
    out, stack = [], [t.tolist()]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def _bits(x):
    """A float32's bit pattern, so 'equal' means equal and not 'close'."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


# ---------------------------------------------------------------------------
# The upstream oracle
# ---------------------------------------------------------------------------

def _upstream():
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"   (skipped: no upstream torch here -- {type(e).__name__})")
        return None
    # The oracle must not be the thing under test. If `torch` on this path is
    # the shim, every agreement number below would be the shim agreeing with
    # itself -- which is the failure a previous round in this repository spent
    # hours inside before noticing.
    assert not hasattr(torch._C, "_aten_implemented"), (
        "upstream torch expected here, got the shim: the oracle would be "
        "comparing the shim against itself")
    return torch


def _rand(torch, *shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1. What the list is, and what the list means
# ---------------------------------------------------------------------------

def test_the_taught_list_grew_and_kept_everything_it_had():
    """Eighteen, containing the four -- stated as a difference, not a number.

    `ge` rather than `eq` on the length, so a later round that teaches a
    nineteenth op does not have to edit this file; but every one of
    docs/devices/VULKAN3.md's four is asserted by name, so a round that *drops* one
    fails here. A count alone could not tell those two apart.
    """
    ops = _C._vulkan_ops()
    assert len(ops) == len(set(ops)), f"duplicate entries in _vulkan_ops(): {ops}"
    assert ops == sorted(ops), "_vulkan_ops() is meant to be a sorted, closed list"
    for op in VULKAN3_OPS:
        assert op in ops, f"docs/devices/VULKAN3.md taught {op} and it is no longer taught"
    assert len(ops) >= 18, f"expected at least 18 taught ops, got {len(ops)}: {ops}"


def test_an_op_that_is_not_taught_refuses_and_names_itself():
    """The property that makes the list mean something.

    The candidate is chosen *because* it is absent from `_vulkan_ops()`, so a
    later round that teaches `aten.gelu.default` makes this test pick a
    different op by itself rather than starting to lie.
    """
    if _vulkan_or_skip("the refusal-names-the-op check") is None:
        return
    ops = set(_C._vulkan_ops())
    candidates = [op for op in ("aten.gelu.default", "aten._softmax.default",
                                "aten.native_layer_norm.default",
                                "aten.bmm.default", "aten.exp.default")
                  if op not in ops]
    assert candidates, "every candidate is taught now -- widen this list"
    op = candidates[0]
    x = _to_vulkan(_cpu([1.0, 2.0], [2]))
    try:
        _C._aten_dispatch(op, x)
    except NotImplementedError as e:
        message = str(e)
    else:
        raise AssertionError(f"{op} is not in _vulkan_ops() but did not refuse")
    assert op in message, f"the refusal does not name the op: {message}"
    assert "vulkan" in message, message
    # The refusal must point somewhere, or the reader is stuck.
    assert ".cpu()" in message, message


def test_the_four_ops_of_the_previous_round_were_one_shader_and_three_that_were_not():
    """docs/devices/VULKAN4.md §1's re-measurement of §5, as an assertion.

    This is the finding that made "Vulkan is four ops" misleading rather than
    wrong, and it is checked here so that it cannot quietly stop being true:
    of the four, `add` dispatches a compute shader and `detach`/`alias`
    dispatch nothing at all.
    """
    if _vulkan_or_skip("the four-ops re-measurement") is None:
        return
    a = _to_vulkan(_cpu([1.0, 2.0, 3.0], [3]))
    b = _to_vulkan(_cpu([4.0, 5.0, 6.0], [3]))

    before = _counters()
    _C._aten_dispatch("aten.add.Tensor", a, b)
    add = _delta(before, _counters())
    assert add["shader_dispatches"] == 1, add
    assert add["host_downloads"] == 0, add

    for op in ("aten.detach.default", "aten.alias.default"):
        before = _counters()
        _C._aten_dispatch(op, a)
        d = _delta(before, _counters())
        assert d["shader_dispatches"] == 0, (op, d)
        assert d["host_uploads"] == 0 and d["host_downloads"] == 0, (op, d)


# ---------------------------------------------------------------------------
# 2. Arbitrary data can reach the device -- the thing that did not exist
# ---------------------------------------------------------------------------

def test_arbitrary_data_reaches_the_device_and_returns_unchanged():
    """`x.to("vulkan")`, which raised "device not available" before this round.

    Until it existed the only tensors on this device were `ones` and `zeros`,
    so no numerical claim about a Vulkan kernel could have been stronger than
    "it handles constants" (docs/devices/VULKAN4.md §1). The values here are chosen to
    have nothing in common with each other or with 1.0, and the comparison is
    on **bits**: an upload followed by a download changes no arithmetic, so
    anything less than bit equality would be a defect and not a tolerance.
    """
    if _vulkan_or_skip("the host-to-device round trip") is None:
        return
    values = [0.1, -2.5, 3.75, 1e-8, -1e8, 0.0, -0.0, 6.02e23]
    src = _cpu(values, [2, 4])
    before = _counters()
    dev = _to_vulkan(src)
    up = _delta(before, _counters())
    assert str(dev.device) == "vulkan", dev.device
    assert up["host_uploads"] == 1, up
    assert up["shader_dispatches"] == 0, "an upload is a copy, not a kernel"

    back = _to_cpu(dev)
    got, want = _flat(back), _flat(src)
    assert [_bits(v) for v in got] == [_bits(v) for v in want], (got, want)


def test_a_dtype_that_has_no_shader_refuses_on_the_way_in():
    """f32 only, and the refusal says so rather than widening silently."""
    if _vulkan_or_skip("the dtype refusal") is None:
        return
    src = _C._tensor_from_flat([1.0, 2.0], [2], dtype=_C.float64)
    try:
        _to_vulkan(src)
    except NotImplementedError as e:
        assert "float" in str(e), str(e)
    else:
        raise AssertionError("a float64 tensor reached the vulkan device")


# ---------------------------------------------------------------------------
# 3. Values -- with the tolerance derived, not chosen
# ---------------------------------------------------------------------------

# The exactly-rounded ops. Each is one IEEE-754 single operation per element
# (or none at all), and IEEE-754 specifies those to the last bit, so upstream's
# own float32-vs-float64 error *is* the shim's: the derivation in
# docs/numerics/AGREE.md §2, applied to this population, produces a tolerance of zero
# and these are held to bit equality. Anything looser here would be a
# tolerance hiding a defect, not measuring a precision.
EXACT_OPS = ("add", "sub", "mul", "div", "neg", "relu", "clone",
             "contiguous", "view", "t", "transpose")


def _shim_apply(op, a, b=None):
    d = _C._aten_dispatch
    if op == "add":
        return d("aten.add.Tensor", a, b)
    if op == "sub":
        return d("aten.sub.Tensor", a, b)
    if op == "mul":
        return d("aten.mul.Tensor", a, b)
    if op == "div":
        return d("aten.div.Tensor", a, b)
    if op == "neg":
        return d("aten.neg.default", a)
    if op == "relu":
        return d("aten.relu.default", a)
    if op == "clone":
        return d("aten.clone.default", a)
    if op == "contiguous":
        return d("aten.contiguous.default", a)
    if op == "view":
        return d("aten.view.default", a, [-1])
    if op == "t":
        return d("aten.t.default", a)
    if op == "transpose":
        return d("aten.transpose.int", a, 0, 1)
    raise AssertionError(op)


def _upstream_apply(torch, op, a, b=None):
    return {
        "add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b,
        "div": lambda: a / b, "neg": lambda: -a, "relu": lambda: torch.relu(a),
        "clone": lambda: a.clone(), "contiguous": lambda: a.contiguous(),
        "view": lambda: a.reshape(-1), "t": lambda: a.t(),
        "transpose": lambda: a.transpose(0, 1),
    }[op]()


def test_the_exactly_rounded_ops_are_bit_identical_to_upstream():
    """Eleven ops, on real data, compared as bit patterns.

    The data is `randn`, not constants -- which only became possible this round
    (docs/devices/VULKAN4.md §1). `div`'s divisor is pushed away from zero so the case
    measures division rather than the representation of infinity.
    """
    if _vulkan_or_skip("the bit-equality sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    checked = 0
    for op in EXACT_OPS:
        for seed, shape in ((1, (4, 5)), (2, (7, 3)), (3, (2, 6))):
            a = _rand(torch, *shape, seed=seed)
            b = None
            if op in ("add", "sub", "mul", "div"):
                b = _rand(torch, *shape, seed=seed + 100)
                if op == "div":
                    b = b + torch.where(b.abs() < 0.5, torch.sign(b) + (b == 0),
                                        torch.zeros_like(b))
            want = _upstream_apply(torch, op, a, b)

            va = _to_vulkan(_cpu(a.reshape(-1).tolist(), shape))
            vb = None if b is None else _to_vulkan(_cpu(b.reshape(-1).tolist(), shape))
            got = _to_cpu(_shim_apply(op, va, vb))

            assert list(got.shape) == list(want.shape), (op, shape, got.shape, want.shape)
            gb = [_bits(v) for v in _flat(got)]
            wb = [_bits(v) for v in want.reshape(-1).tolist()]
            assert gb == wb, (
                f"{op}{list(shape)} is not bit-identical to upstream: "
                f"{sum(x != y for x, y in zip(gb, wb))} of {len(gb)} elements differ")
            checked += 1
    assert checked == len(EXACT_OPS) * 3, checked
    print(f"   {checked} exactly-rounded cases, all bit-identical to upstream")


def _derived_tolerance(upstream_rel_errors):
    """docs/numerics/AGREE.md §2's rule, recomputed here from this round's population.

    The p90 of upstream's own float32-vs-float64 relative error, floored at
    8 float32 ulp. The floor is AGREE's, and its reason is AGREE's: a
    population that happened to be numerically easy must not be able to drive
    the tolerance down to where float32 differs for reasons nobody claims are
    defects. Recomputed rather than pasted, so a change in the population
    changes the number instead of silently failing against a stale constant.
    """
    pop = sorted(upstream_rel_errors)
    p90 = pop[int(0.9 * (len(pop) - 1))]
    return max(p90, 8 * FLOAT32_EPS), p90


MATMUL_SHAPES = ((2, 3, 4), (8, 16, 8), (5, 64, 7), (32, 128, 16), (1, 512, 1),
                 (3, 257, 5))


def test_the_matmuls_agree_with_upstream_at_a_derived_tolerance():
    """`mm` and `addmm` -- the first kernels here whose answer is not forced.

    Everything in `EXACT_OPS` is one exactly-rounded IEEE operation, so a
    difference could only be plumbing. A dot product is a *sum*, and a sum has
    an order. So this is the one place a tolerance is needed, and it is read
    off upstream's own float32-vs-float64 error rather than chosen -- and then
    the *next* test proves the residue is that order and not a defect.
    """
    if _vulkan_or_skip("the matmul agreement sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    rows, up_errs = [], []
    for i, (m, k, n) in enumerate(MATMUL_SHAPES):
        a = _rand(torch, m, k, seed=200 + i)
        b = _rand(torch, k, n, seed=300 + i)
        c = _rand(torch, n, seed=400 + i)
        va = _to_vulkan(_cpu(a.reshape(-1).tolist(), (m, k)))
        vb = _to_vulkan(_cpu(b.reshape(-1).tolist(), (k, n)))
        vc = _to_vulkan(_cpu(c.reshape(-1).tolist(), (n,)))
        for name, want32, want64, got in (
            ("mm",
             torch.mm(a, b), torch.mm(a.double(), b.double()),
             _to_cpu(_C._aten_dispatch("aten.mm.default", va, vb))),
            ("addmm",
             torch.addmm(c, a, b),
             torch.addmm(c.double(), a.double(), b.double()),
             _to_cpu(_C._aten_dispatch("aten.addmm.default", vc, va, vb))),
        ):
            truth = want64.reshape(-1).tolist()
            scale = max(abs(v) for v in truth)
            up_rel = max(abs(x - y) for x, y in
                         zip(want32.double().reshape(-1).tolist(), truth)) / scale
            shim_rel = max(abs(x - y) for x, y in zip(_flat(got), truth)) / scale
            up_errs.append(up_rel)
            rows.append((f"{name}[{m},{k},{n}]", shim_rel, up_rel))

    tol, p90 = _derived_tolerance(up_errs)
    worst = max(r[1] for r in rows)
    print(f"   derived tolerance = max(p90 {p90:.3e}, 8 ulp {8*FLOAT32_EPS:.3e}) "
          f"= {tol:.3e}; worst shim relative error {worst:.3e}")
    for name, shim_rel, up_rel in rows:
        assert shim_rel <= tol, (
            f"{name}: shim {shim_rel:.3e} exceeds the derived tolerance {tol:.3e} "
            f"(upstream's own error on the same output was {up_rel:.3e})")


def _f32(x):
    """Round a Python double to float32, the way the hardware would."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _host_model_of_the_matmul_kernel(a, b, m, k, n):
    """What `shaders/matmul_f32.comp` says it does, executed on the host.

    One invocation per output element, accumulating over `k` in declaration
    order, in float32, **with the multiply-add contracted** -- the product of
    two float32 values is exact in a Python double, so rounding
    `acc + a*b` once is precisely a fused multiply-add.

    This is not a re-implementation for its own sake. It is the instrument that
    settles whether the matmul's disagreement with upstream is precision or a
    defect (docs/devices/VULKAN4.md §4.2): a kernel that had transposed an index or
    lost a term would not be reproduced by a model of the arithmetic it claims
    to do, at every shape, to the bit.
    """
    out = []
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for kk in range(k):
                acc = _f32(acc + a[i * k + kk] * b[kk * n + j])
            out.append(acc)
    return out


def test_the_matmul_residue_is_fma_contraction_and_not_a_defect():
    """The proof, rather than a tolerance that happens to hold.

    `mm` differs from upstream in the last bits at longer `k` -- 2 of 15
    elements at k=257, 1 of 1 at k=512. That is either an accumulation-order
    difference or a broken kernel, and a tolerance cannot tell those apart:
    both look like "small". So the question is answered by construction
    instead. The GPU's answer is reproduced **bit for bit, at every shape** by
    a host model of sequential float32 accumulation with FMA, which is a legal
    contraction of `acc += a*b` and the one the driver applied.

    A kernel that read the wrong element would still be "small" and would not
    survive this.
    """
    if _vulkan_or_skip("the FMA-contraction proof") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    differed_from_upstream = 0
    for i, (m, k, n) in enumerate(MATMUL_SHAPES):
        a = _rand(torch, m, k, seed=200 + i)
        b = _rand(torch, k, n, seed=300 + i)
        af, bf = a.reshape(-1).tolist(), b.reshape(-1).tolist()
        got = _flat(_to_cpu(_C._aten_dispatch(
            "aten.mm.default",
            _to_vulkan(_cpu(af, (m, k))), _to_vulkan(_cpu(bf, (k, n))))))
        model = _host_model_of_the_matmul_kernel(af, bf, m, k, n)
        assert [_bits(v) for v in got] == [_bits(v) for v in model], (
            f"mm[{m},{k},{n}] is not what shaders/matmul_f32.comp describes: "
            f"{sum(_bits(x) != _bits(y) for x, y in zip(got, model))} of "
            f"{m * n} elements differ from the host model of the kernel")
        upstream = torch.mm(a, b).reshape(-1).tolist()
        differed_from_upstream += sum(
            _bits(x) != _bits(y) for x, y in zip(got, upstream))
    # If nothing ever differed from upstream, this test would be asserting
    # something true for a trivial reason and the sweep above would be enough.
    # It does differ, which is what makes the model the interesting evidence.
    assert differed_from_upstream > 0, (
        "no element differed from upstream anywhere -- this population no "
        "longer exercises the accumulation-order question")
    print(f"   the host FMA model reproduces every shape bit-for-bit; "
          f"{differed_from_upstream} elements differ from upstream's blocked gemm")


# ---------------------------------------------------------------------------
# 4. Did the GPU do it -- asserted from the runtime, not from the answer
# ---------------------------------------------------------------------------

# op -> (how many compute shaders it must run, arity)
#
# Zero is as much a claim as one. `view` and `contiguous` are shape-only on a
# device where every tensor is contiguous by construction, and saying they
# dispatch nothing is more honest than a number that would imply GPU work.
EXPECTED_DISPATCHES = {
    "aten.add.Tensor": (1, 2),
    "aten.sub.Tensor": (1, 2),
    "aten.mul.Tensor": (1, 2),
    "aten.div.Tensor": (1, 2),
    "aten.neg.default": (1, 1),
    "aten.relu.default": (1, 1),
    "aten.clone.default": (1, 1),
    "aten.t.default": (1, 1),
    "aten.transpose.int": (1, 1),
    "aten.mm.default": (1, 2),
    "aten.addmm.default": (2, 3),
    "aten.detach.default": (0, 1),
    "aten.alias.default": (0, 1),
    "aten.contiguous.default": (0, 1),
    "aten.view.default": (0, 1),
    "aten._unsafe_view.default": (0, 1),
    "aten.reshape.default": (0, 1),
}


def test_every_taught_op_ran_on_the_gpu_or_says_it_did_not():
    """The device assertion, from `_vulkan_counters()` at runtime.

    **Why not a source scan.** docs/devices/MPSATTN.md §3.1 records, against its own
    round, that an op could have been taken off the `mps` refusal list while
    keeping its host readback and *both* of that device's derivation tests
    would still have passed: the per-op scan looks for six helper names in a
    kernel body, the classification test looks for three markers, and moving
    the readback one call deeper into a differently-named helper is invisible
    to both. Every check of that shape can be defeated by moving the thing it
    greps for.

    This one cannot, because it is not a description of the source.
    `shader_dispatches` is incremented inside `dispatch_kernel` *after*
    `vkWaitForFences` returns success, and `host_downloads` inside `download`,
    which is the module's only map-for-reading. An op that computed on the host
    would have to read its operands to do so, and reading them goes through
    that one function however many helpers deep it is buried. So the assertion
    below is about what the process did:

        the expected number of compute shaders ran, and nothing was read back.
    """
    if _vulkan_or_skip("the per-op GPU assertion") is None:
        return
    a = _to_vulkan(_cpu([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]))
    b = _to_vulkan(_cpu([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [2, 3]))
    sq = _to_vulkan(_cpu([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]))
    bias = _to_vulkan(_cpu([0.5, 1.5], [2]))

    taught = set(_C._vulkan_ops())
    assert taught == set(EXPECTED_DISPATCHES) | {"aten._to_copy.default"}, (
        "an op was taught or dropped without saying how many shaders it runs; "
        f"missing from this table: {sorted(taught - set(EXPECTED_DISPATCHES) - {'aten._to_copy.default'})}")

    for op, (expected, arity) in sorted(EXPECTED_DISPATCHES.items()):
        if op == "aten.mm.default":
            args = (a, sq)
        elif op == "aten.addmm.default":
            args = (bias, a, sq)
        elif op == "aten.transpose.int":
            args = (a, 0, 1)
        elif op in ("aten.view.default", "aten._unsafe_view.default",
                    "aten.reshape.default"):
            args = (a, [6])
        elif arity == 2:
            args = (a, b)
        else:
            args = (a,)
        before = _counters()
        out = _C._aten_dispatch(op, *args)
        d = _delta(before, _counters())
        assert str(out.device) == "vulkan", (op, out.device)
        assert d["shader_dispatches"] == expected, (
            f"{op} ran {d['shader_dispatches']} compute shaders, expected "
            f"{expected}")
        assert d["host_downloads"] == 0, (
            f"{op} read {d['host_downloads']} buffer(s) back to the host -- it "
            f"is computing on the CPU under a vulkan label")
        assert d["host_uploads"] == 0, (
            f"{op} uploaded {d['host_uploads']} buffer(s); no taught op builds "
            f"an operand on the host")


def test_the_readback_counter_moves_when_something_is_actually_read_back():
    """The positive control for the instrument above.

    An assertion of the form "this counter did not move" is worthless if the
    counter never moves. `.cpu()` genuinely reads the buffer back, so it must
    move it -- and if a future change made `download` stop counting, the test
    above would go quietly green on an op that had started computing on the
    host. This is the test that fails first in that case.
    """
    if _vulkan_or_skip("the readback-counter control") is None:
        return
    x = _to_vulkan(_cpu([1.0, 2.0], [2]))
    before = _counters()
    _to_cpu(x)
    d = _delta(before, _counters())
    assert d["host_downloads"] == 1, (
        f"a .cpu() did not register as a host readback ({d}) -- the counter "
        f"that the per-op assertions rely on is not live")


# ---------------------------------------------------------------------------
# 5. A whole module -- the honest answer to "does Vulkan work?"
# ---------------------------------------------------------------------------

_MLP_SHIM_SCRIPT = r"""
import json, sys
import torch, torch.nn as nn

# The probe a previous round in this repository paid for by measuring upstream
# torch for hours: this subprocess must be the shim, not the oracle.
assert hasattr(torch._C, "_aten_implemented"), "this subprocess got upstream torch"

cfg = json.load(sys.stdin)
out = {"probe": torch._C._vulkan_probe()}
if not out["probe"]["available"]:
    json.dump(out, sys.stdout); raise SystemExit

m = nn.Sequential(nn.Linear(cfg["i"], cfg["h"]), nn.ReLU(),
                  nn.Linear(cfg["h"], cfg["o"]))
m.load_state_dict({k: torch.as_tensor(v, dtype=torch.float32)
                   for k, v in cfg["sd"].items()})
m.eval()
x = torch.as_tensor(cfg["x"], dtype=torch.float32).reshape(cfg["sx"])
with torch.no_grad():
    out["cpu"] = m(x).reshape(-1).tolist()
m.to("vulkan")
before = torch._C._vulkan_counters()
with torch.no_grad():
    r = m(x.to("vulkan"))
after = torch._C._vulkan_counters()
out["device"] = str(r.device)
out["counters"] = {k: after[k] - before[k] for k in after}
out["vulkan"] = r.cpu().reshape(-1).tolist()
json.dump(out, sys.stdout)
"""


def test_a_whole_module_forwards_on_the_gpu_and_agrees_with_upstream():
    """`nn.Sequential(Linear, ReLU, Linear)`, forward, on the Vulkan device.

    **This is the claim docs/devices/VULKAN4.md §5 makes and the one it stops at.** A
    module, not an op: `nn.Linear` dispatches `aten.t.default` and then
    `aten.addmm.default`, so the forward really does go through the transpose
    and the matmul rather than around them, and `m.to("vulkan")` really does
    move the parameters.

    It is **not** a transformer. `native_layer_norm`, `_softmax`, `gelu`,
    `embedding` and `bmm` are all still refused, and every one of them is on
    the measured trace of a BERT forward (docs/devices/VULKAN4.md §2), so a transformer
    stops at the first of them. Saying "an MLP forwards" is the honest size of
    this result.

    Three assertions, and the third is the one that is not about the answer:

      1. the output really is a vulkan tensor;
      2. it agrees with upstream inside docs/numerics/AGREE.md's derived rule, and with
         the shim's own cpu answer to about one float32 ulp;
      3. **seven compute shaders ran and nothing was read back** -- 2x(t,
         matmul, bias) + 1 relu, which is what the two Linears and the ReLU
         must cost if the GPU is doing them.
    """
    if _vulkan_or_skip("the module forward") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    try:
        import torch.nn as nn
    except Exception:  # noqa: BLE001
        print("   (skipped the module forward: no torch.nn upstream)")
        return

    i, h, o, batch = 12, 32, 6, 5
    g = torch.Generator().manual_seed(11)
    model = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o)).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32))
    x = torch.randn(batch, i, generator=g, dtype=torch.float32)

    cfg = {"i": i, "h": h, "o": o, "sx": [batch, i],
           "x": x.reshape(-1).tolist(),
           "sd": {k: v.tolist() for k, v in model.state_dict().items()}}
    env = dict(os.environ)
    env["PYTHONPATH"] = VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", _MLP_SHIM_SCRIPT],
                          input=json.dumps(cfg), capture_output=True,
                          text=True, env=env, timeout=300)
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-3000:])
    got = json.loads(proc.stdout)
    if not got["probe"]["available"]:
        print(f"   (skipped the module forward: the vendored-tree subprocess "
              f"has no loader -- {got['probe']['error']})")
        return

    assert got["device"].startswith("vulkan"), got["device"]

    with torch.no_grad():
        f32 = model(x).reshape(-1).tolist()
        wide = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o)).eval()
        wide.load_state_dict(model.state_dict())
        truth = wide.double()(x.double()).reshape(-1).tolist()

    scale = max(abs(v) for v in truth)
    up_err = max(abs(a - b) for a, b in zip(f32, truth))
    vk_err = max(abs(a - b) for a, b in zip(got["vulkan"], truth))
    cpu_err = max(abs(a - b) for a, b in zip(got["cpu"], truth))
    backends = max(abs(a - b) for a, b in zip(got["vulkan"], got["cpu"]))
    ulp = scale * FLOAT32_EPS

    # docs/numerics/AGREE.md §2's second rule: a difference is not a defect if it is
    # within 4x upstream's own distance from the float64 truth on the same
    # output. Measured at 0.90x (docs/devices/VULKAN4.md §4.3).
    assert vk_err <= 4 * up_err, (
        f"vulkan is {vk_err:.3e} from the float64 truth against upstream's own "
        f"{up_err:.3e}; docs/numerics/AGREE.md's rule allows 4x")
    assert cpu_err <= 4 * up_err, (cpu_err, up_err)
    assert backends <= 2 * ulp, (
        f"vulkan and the shim's own cpu differ by {backends:.3e}, more than "
        f"two float32 ulp ({2 * ulp:.3e}) at this magnitude")

    # The third assertion, and the one that is about the device rather than
    # the answer. Two Linears = 2 x (transpose + matmul + bias) = 6, plus the
    # ReLU = 7. A forward that had fallen back to the host would still get the
    # numbers right and would fail here.
    counters = got["counters"]
    assert counters["shader_dispatches"] == 7, (
        f"the module forward ran {counters['shader_dispatches']} compute "
        f"shaders, expected 7 (2 Linears x 3 + 1 ReLU): {counters}")
    assert counters["host_downloads"] == 0, (
        f"the module forward read {counters['host_downloads']} buffer(s) back "
        f"to the host: {counters}")
    print(f"   MLP on vulkan: upstream f32 err {up_err:.3e}, shim cpu "
          f"{cpu_err:.3e}, shim vulkan {vk_err:.3e} ({vk_err / up_err:.2f}x), "
          f"vulkan-vs-cpu {backends / ulp:.2f} ulp, "
          f"{counters['shader_dispatches']} shaders, "
          f"{counters['host_downloads']} readbacks")


def test_a_transformer_still_does_not_forward_and_the_wall_is_named():
    """The other half of the honest answer.

    docs/devices/VULKAN4.md §2 measured what a BERT forward dispatches. Five of those
    ops are not taught this device, and this asserts they still refuse -- so
    that "an MLP forwards, a transformer does not" cannot quietly become
    stale in either direction. If a later round teaches one, this test says so
    by failing, and whoever teaches it gets to update the sentence.
    """
    if _vulkan_or_skip("the named transformer wall") is None:
        return
    taught = set(_C._vulkan_ops())
    walls = ("aten.native_layer_norm.default", "aten._softmax.default",
             "aten.gelu.default", "aten.embedding.default", "aten.bmm.default")
    still = [op for op in walls if op not in taught]
    assert still == list(walls), (
        f"these are taught now and docs/devices/VULKAN4.md §5 needs updating: "
        f"{sorted(set(walls) - set(still))}")


# ---------------------------------------------------------------------------
# 6. The narrowings refuse rather than reaching for the CPU
# ---------------------------------------------------------------------------

def test_every_narrowing_refuses_by_name_rather_than_being_emulated():
    """The property the whole `Repr::Vulkan` design exists to protect.

    Each of these has a perfectly good CPU implementation a few lines away, and
    reaching for one would be the silent fallback docs/devices/VULKAN.md §5 calls the
    worst available outcome. They refuse, and the message says which narrowing
    was hit -- a generic "not implemented" would leave the reader unable to
    tell a missing broadcast from a missing dtype.
    """
    if _vulkan_or_skip("the narrowing refusals") is None:
        return
    a23 = _to_vulkan(_cpu([1.0] * 6, [2, 3]))
    a32 = _to_vulkan(_cpu([1.0] * 6, [3, 2]))
    a3d = _to_vulkan(_cpu([1.0] * 8, [2, 2, 2]))
    bias = _to_vulkan(_cpu([1.0, 1.0], [2]))

    cases = (
        ("broadcast", lambda: _C._aten_dispatch("aten.add.Tensor", a23, a32),
         "broadcast"),
        ("alpha", lambda: _C._aten_dispatch("aten.add.Tensor", a23, a23, 2.0),
         "alpha"),
        ("3-D transpose",
         lambda: _C._aten_dispatch("aten.transpose.int", a3d, 0, 1), "2-D"),
        ("batched matmul",
         lambda: _C._aten_dispatch("aten.mm.default", a3d, a3d), "2-D"),
        ("addmm beta",
         lambda: _C._aten_dispatch("aten.addmm.default", bias, a23, a32,
                                   2.0), "beta"),
        ("bad reshape",
         lambda: _C._aten_dispatch("aten.view.default", a23, [4, 4]),
         "invalid"),
    )
    for what, call, needle in cases:
        try:
            call()
        except (NotImplementedError, RuntimeError) as e:
            assert needle in str(e), (
                f"the {what} refusal does not say which narrowing was hit "
                f"(looked for {needle!r}): {e}")
        else:
            raise AssertionError(f"{what} was emulated instead of refused")


def test_a_vulkan_tensor_still_has_no_cpu_storage_to_read():
    """The structural property, re-asserted after eighteen ops were added.

    `PyTensorBase::tensor()` refuses on every non-`Dense` arm, and that refusal
    is what makes a silent CPU fallback unrepresentable rather than merely
    avoided. Widening the device is exactly the change that could have
    weakened it -- so it is checked again here rather than assumed to have
    survived.
    """
    if _vulkan_or_skip("the no-cpu-storage property") is None:
        return
    x = _to_vulkan(_cpu([1.0, 2.0], [2]))
    try:
        x.tolist()
    except (NotImplementedError, RuntimeError) as e:
        assert "cpu" in str(e).lower(), str(e)
    else:
        raise AssertionError(
            "a vulkan tensor handed out CPU storage through tolist()")


# ---------------------------------------------------------------------------
# 7. The checked-in SPIR-V is what the GLSL says
# ---------------------------------------------------------------------------

def test_the_checked_in_spirv_is_not_stale_for_any_shader():
    """docs/devices/VULKAN3.md's guard, extended from one shader to all of them.

    The `.spv` words are checked in and `include_bytes!`d rather than built by
    a `build.rs`, so an edit to a `.comp` does nothing until
    `shaders/compile.sh` is run -- and the old kernel ships silently. That
    guard existed for `add_f32` alone; this round added nine more files, and a
    guard that covers one of ten is a guard someone will trust.
    """
    shaders = os.path.join(REPO, "rust", "torch_c", "shaders")
    comps = sorted(f for f in os.listdir(shaders) if f.endswith(".comp"))
    assert len(comps) >= 10, f"expected at least ten shaders, found {comps}"
    for comp in comps:
        spv = os.path.join(shaders, comp[:-len(".comp")] + ".spv")
        assert os.path.exists(spv), (
            f"{comp} has no compiled {os.path.basename(spv)} -- run "
            f"shaders/compile.sh")
        assert os.path.getmtime(spv) >= os.path.getmtime(os.path.join(shaders, comp)), (
            f"{os.path.basename(spv)} is older than {comp}: the checked-in "
            f"kernel is not the one the GLSL describes. Run shaders/compile.sh")
        with open(spv, "rb") as fh:
            words = fh.read()
        assert len(words) % 4 == 0 and len(words) > 20, (comp, len(words))
        # SPIR-V's magic number, little-endian. A truncated or text file that
        # happened to be newer would otherwise pass the two checks above.
        assert words[:4] == b"\x03\x02\x23\x07", (
            f"{os.path.basename(spv)} does not begin with the SPIR-V magic "
            f"number -- it is not a compiled shader")


def test_every_shader_on_disk_is_reachable_from_the_dispatcher():
    """A shader nobody dispatches is dead weight that still looks like coverage."""
    shaders = os.path.join(REPO, "rust", "torch_c", "shaders")
    src = os.path.join(REPO, "rust", "torch_c", "src", "vulkan.rs")
    with open(src) as fh:
        text = fh.read()
    for comp in sorted(f for f in os.listdir(shaders) if f.endswith(".comp")):
        stem = comp[:-len(".comp")]
        assert f'"{stem}"' in text, (
            f"shaders/{comp} is compiled and checked in but no call in "
            f"vulkan.rs names {stem!r}")


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
