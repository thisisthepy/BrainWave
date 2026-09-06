"""docs/AGREE.md -- what holds down the agreement sweep's own machinery.

`agree_sweep.py` is a measurement, not an invariant, and it is deliberately not
wired into `run.sh`: it needs both interpreters and a long time. But three
things inside it are load-bearing enough that a silent change to any of them
would produce a *plausible wrong number* rather than a failure, and those are
what this file pins.

  * **The comparison arithmetic.** `diff_stats` measures a difference against
    the reference tensor's own scale rather than element-wise, because dividing
    by a 1e-30 element manufactures a divergence out of a denormal.
    `verdict` has two "I cannot tell" outcomes and their ORDER is the claim: an
    architecture upstream cannot reproduce has no answer to match, and two
    sides that both underflow to zero are equal without having agreed.

  * **The calibration batch.** Its one requirement is that the rows DIFFER, and
    getting it wrong is silent both ways: identical float rows give a batch
    variance of zero and turn every BatchNorm into a 1/sqrt(eps) amplifier,
    identical integer rows collapse a text branch's BatchNorm to its bias.
    `groupvit` was measured at 5.8e-01 end-to-end under the second of those and
    at 3.1e-06 once fixed -- a factor of 185,000 of pure harness artefact.

  * **The claim that both sides share an RNG.** The sweep does not rely on it
    (weights travel as bytes through `.npz`), but the claim is load-bearing in
    `arch_sweep.py` and in this project's prose, so upstream's seeded values are
    frozen here. If the shim's generator drifts, this fails rather than quietly
    turning some other round's experiment into a comparison of two different
    random initialisations.

Nothing here needs the network, a checkpoint, or `transformers`.
"""

import json
import math
import os
import subprocess
import sys

import agree_sweep as A

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PYTESTS = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


# --------------------------------------------------------------------------
# upstream's seeded values, recorded rather than trusted
# --------------------------------------------------------------------------
#
# Produced by `agree_sweep.py --rng-check` under `env -u PYTHONPATH -u
# TORCH_USE_RTLD_GLOBAL <spike-venv python>`, torch 2.13.0, seed 0. Transcribed
# in full so a reader can re-derive them with one command.
_UPSTREAM_SEED0 = {
    "randn": [1.5409960746765137, -0.293428897857666, -2.1787893772125244,
              0.5684312582015991, -1.0845223665237427, -1.3985954523086548,
              0.40334683656692505],
    "rand": [0.49625658988952637, 0.7682217955589294, 0.08847743272781372,
             0.13203048706054688, 0.30742281675338745, 0.6340786814689636,
             0.4900934100151062],
    "randint": [44.0, 39.0, 33.0, 60.0, 63.0, 79.0, 27.0],
    "randperm": [4.0, 0.0, 5.0, 3.0, 2.0, 6.0, 1.0],
    "uniform_": [-0.007486820220947266, 0.5364435911178589, -0.8230451345443726,
                 -0.7359390258789062, -0.3851543664932251, 0.26815736293792725,
                 -0.019813179969787598],
}


def _shim_json(args):
    """Run `agree_sweep.py <args>` against the VENDORED shim and parse its JSON.

    Returns None when the vendored shim is not installed, which is the same
    silent skip `test_tail3.py` and `test_tail4.py` take -- `run.sh` refuses to
    run at all when that file is stale, so a present-but-old shim is not a case
    this has to defend against.
    """
    if not os.path.isfile(_VENDOR_SHIM):
        return None
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR + os.pathsep + _PYTESTS
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, os.path.join(_PYTESTS, "agree_sweep.py")] + args,
        capture_output=True, text=True, env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, f"the shim probe failed:\n{proc.stderr[-3000:]}"
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "shim", (
        f"the probe imported the wrong torch ({lines[0]!r}) -- every assertion "
        "below would have been about upstream")
    return json.loads("\n".join(lines[1:]))


# --------------------------------------------------------------------------
# the comparison arithmetic
# --------------------------------------------------------------------------


def test_diff_stats_is_zero_for_identical_arrays_and_reports_the_scale():
    a = [1.0, -2.0, 3.5]
    st = A.diff_stats(a, a)
    assert st["abs"] == 0.0 and st["rel"] == 0.0, st
    assert st["scale"] == 3.5, st
    assert st["n"] == 3, st


def test_diff_stats_measures_against_the_tensors_scale_not_element_wise():
    """A denormal element must not manufacture a divergence.

    `[1.0, 1e-30]` against `[1.0, 2e-30]` is a 100% element-wise error on the
    second element and no error at all in any sense that matters: there is
    nothing in that element to be right about. Element-wise relative error would
    have put a wall of such rows at the top of the ranking and buried the real
    ones -- which is the failure mode this whole exercise exists to avoid.
    """
    st = A.diff_stats([1.0, 1e-30], [1.0, 2e-30])
    assert st["rel"] < 1e-29, st
    # and a difference that IS at the tensor's scale is reported in full
    st2 = A.diff_stats([1.0, 0.0], [2.0, 0.0])
    assert st2["rel"] == 0.5, st2


def test_diff_stats_refuses_a_shape_mismatch_rather_than_broadcasting():
    st = A.diff_stats([1.0, 2.0], [1.0, 2.0, 3.0])
    assert st["shape_mismatch"] is True and math.isinf(st["rel"]), st


def test_diff_stats_counts_a_nan_against_a_number_instead_of_dropping_it():
    """A NaN where upstream has a number is the loudest possible disagreement.

    Masking non-finite entries out of the difference is right for the magnitude
    (inf - inf is not a measurement) and wrong as a verdict, so the count is
    carried separately rather than folded into `rel`.
    """
    st = A.diff_stats([float("nan"), 1.0], [0.0, 1.0])
    assert st["nonfinite_mismatch"] == 1, st
    same = A.diff_stats([float("inf"), 1.0], [float("inf"), 1.0])
    assert same["nonfinite_mismatch"] == 0 and same["rel"] == 0.0, same


# --------------------------------------------------------------------------
# the verdict, and the order of its two refusals
# --------------------------------------------------------------------------


def test_verdict_refuses_first_when_upstream_cannot_reproduce_itself():
    """`vit_mae` re-draws its patch mask every forward and `vits` samples a
    duration. Neither has a fixed answer, so neither can be scored -- and that
    outcome has to win over every other, including `exact`."""
    assert A.verdict(0.0, 1.0, 1e-7, 1e-6, self_repeat_rel=1.36) == "nondeterministic"
    assert A.verdict(1e-9, 1.0, 1e-7, 1e-6, self_repeat_rel=0.0) == "agree"


def test_verdict_calls_an_underflowed_output_degenerate_rather_than_exact():
    """Two sides that both underflow to zero are equal without having agreed.
    Measured, not hypothetical: `efficientnet` lands at 8.0e-29 even after
    BatchNorm calibration, and `sam_vision_model` at 8.0e-21."""
    assert A.verdict(0.0, 8.0e-29, None, 1e-6) == "degenerate"
    assert A.verdict(0.0, 1.0, None, 1e-6) == "exact"


def test_verdict_accepts_a_difference_upstream_itself_incurs():
    """docs/DEMAND8.md §1.4's standard, as code.

    `mobilenet_v2` differs from upstream by 1.59e-04 relative while upstream's
    own float32 answer is 9.85e-05 from the float64 truth. That is not a defect;
    it is two independent float32 truncation paths of the same size. The rule
    admits it and refuses an order of magnitude.
    """
    assert A.verdict(1.587e-04, 6.0, 9.854e-05, 1.186e-06) == "agree_within_float32"
    assert A.verdict(1.0e-03, 6.0, 9.854e-05, 1.186e-06) == "diverge"


def test_the_oracle_factor_is_stated_and_is_not_a_free_parameter():
    """4x, and the reason is on the constant. If someone widens it to make a
    number look better, this fails and says where the 4 came from."""
    assert A.ORACLE_FACTOR == 4.0
    assert 1.28e-04 / 1.06e-04 < A.ORACLE_FACTOR, (
        "DEMAND8 §1.4's measured ratio of shim-vs-upstream to upstream-vs-truth "
        "must sit inside the factor, or the factor is not derived from anything")
    assert abs(A.F32_EPS - 1.1920929e-07) < 1e-14


# --------------------------------------------------------------------------
# output traversal
# --------------------------------------------------------------------------


class _FakeTensor:
    def __init__(self, tag):
        self.shape, self.dtype, self.tag = (1,), "float32", tag

    def reshape(self, *a):
        return self


def test_flatten_outputs_finds_tensors_in_dicts_tuples_and_drops_the_rest():
    t1, t2, t3 = _FakeTensor("a"), _FakeTensor("b"), _FakeTensor("c")
    got = A.flatten_outputs({"last_hidden_state": t1,
                             "hidden": (t2, "not a tensor", None),
                             "cfg": object(),
                             "nested": {"x": t3}})
    assert {k: v.tag for k, v in got.items()} == {
        "last_hidden_state": "a", "hidden.0": "b", "nested.x": "c"}, got


def test_flatten_outputs_names_a_bare_tensor_rather_than_returning_it_unkeyed():
    got = A.flatten_outputs(_FakeTensor("solo"))
    assert list(got) == ["__out__"], got


# --------------------------------------------------------------------------
# the calibration batch -- the artefact that cost a factor of 185,000
# --------------------------------------------------------------------------


def test_the_calibration_batch_rows_differ_on_both_float_and_integer_inputs():
    """The regression for `groupvit`.

    Runs under the shim, because that is where the sweep's own inputs are
    rebuilt and because it doubles as a check that `randn_like` and `randint`
    are there. Both branches are asserted: a float row must not equal the
    original (zero variance turns BatchNorm into a 1/sqrt(eps) amplifier) and
    an integer row must not either (zero variance on a text branch collapses
    BatchNorm to its bias -- `groupvit` read 5.8e-01 that way and 3.1e-06 once
    the rows differed).
    """
    data = _shim_json(["--self-test-calibration"])
    if data is None:
        print("   (skipped: no vendored shim installed)")
        return
    assert data["float_rows_distinct"] is True, data
    assert data["int_rows_distinct"] is True, data
    assert data["float_batch_variance"] > 1e-3, data
    assert data["int_batch_variance"] > 0.0, data
    # a boolean mask is repeated verbatim: a random mask is a different
    # sequence length, not more variance
    assert data["bool_rows_identical"] is True, data
    assert data["batch"] == 4, data


# --------------------------------------------------------------------------
# transport, and the RNG claim
# --------------------------------------------------------------------------


def test_the_weight_transport_round_trips_every_dtype_it_claims_to_carry():
    """`Tensor.numpy` and `torch.from_numpy` are both absent on the shim, so the
    sweep moves weights as `tolist()` out and `as_tensor()` in. That path is the
    experiment: if it loses a dtype or a shape, the two sides are running
    different models and every number afterwards is about the transport."""
    data = _shim_json(["--self-test-transport"])
    if data is None:
        print("   (skipped: no vendored shim installed)")
        return
    unavailable = {d: r["unavailable"] for d, r in data.items() if "unavailable" in r}
    # `int8` is not storable by the candle backend at all, so it cannot be
    # transported and is not a transport defect. It is asserted as the ONLY such
    # dtype, so the day another one joins it this says so rather than shrinking
    # the check silently. No architecture in the sweep carried one: the
    # `state_dict` round-tripped for 290 of 290.
    assert set(unavailable) == {"torch.int8"}, unavailable
    for dtype, row in data.items():
        if "unavailable" in row:
            continue
        assert row["dtype_preserved"], (dtype, row)
        assert row["shape_preserved"], (dtype, row)
        assert row["values_exact"], (dtype, row)
    # the shapes `tolist()` cannot round-trip on its own are the point
    assert "torch.float32:scalar" in data and "torch.float32:empty" in data, sorted(data)


def test_the_shim_reproduces_upstreams_seeded_random_numbers():
    """Frozen from upstream, seed 0. The sweep does not depend on this -- weights
    travel as bytes -- but `arch_sweep.py` and this project's prose do, and a
    drift here silently turns a same-weights experiment somewhere else into a
    comparison of two different initialisations."""
    data = _shim_json(["--rng-check"])
    if data is None:
        print("   (skipped: no vendored shim installed)")
        return
    for name, want in _UPSTREAM_SEED0.items():
        got = data[name]
        assert isinstance(got, list), (name, got)
        assert len(got) == len(want), (name, got)
        for g, w in zip(got, want):
            assert g == w, (name, g, w)


def test_the_rng_check_covers_more_than_one_generator():
    """`randn` alone would pass on a shim that only wired the normal path. The
    frozen set spans normal, uniform, integer and permutation."""
    assert {"randn", "rand", "randint", "randperm", "uniform_"} <= set(_UPSTREAM_SEED0)
    assert {n for n, _ in A._RNG_SPEC} >= set(_UPSTREAM_SEED0)


# --------------------------------------------------------------------------
# the harness's own honesty
# --------------------------------------------------------------------------


def test_the_module_capture_clones_rather_than_referencing():
    """The bisection replays each leaf on upstream's recorded input, and the
    bundle is written after the forward has finished -- so a residual `+=` or an
    in-place activation rewrites a referenced tensor before it is serialised.
    Measured: without the clone, `groupvit`'s `downsample.assign.proj`, a plain
    `nn.Linear`, reported a relative error of 1.0. There is no value assertion
    that can catch this from outside, so the clone is pinned at the source."""
    import inspect
    src = inspect.getsource(A.capture_one)
    assert "detach().clone()" in src, (
        "capture_one stopped cloning; every bisection result is now about "
        "whatever ran after the hook, not about the shim")


def test_the_report_cannot_count_an_unjudgeable_architecture_as_agreeing():
    """The headline's denominator excludes both refusal outcomes. Asserted on
    the verdict function rather than on prose, because the denominator is the
    number a reader will quote."""
    for v in ("degenerate", "nondeterministic"):
        assert v not in ("exact", "agree", "agree_within_float32")
    assert A.verdict(0.0, 1e-30, None, 1e-6, 0.0) == "degenerate"
    assert A.verdict(0.0, 1.0, None, 1e-6, 1.0) == "nondeterministic"


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
