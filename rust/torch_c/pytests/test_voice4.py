"""docs/architectures/VOICE4.md -- the first voicestudio model run end to end, and the wall
that a coverage list structurally could not see.

Seven rounds of `docs/VOICE*.md` closed operators that speech models *tripped
over*, one wall at a time, and none of them ever ran a voicestudio model
through. This round picked one -- **BigVGAN v2 24 kHz 100-band 256x**, the
vocoder voicestudio's F5-TTS and CosyVoice hold as a submodel -- loaded the real
NVIDIA checkpoint (112,414,512 parameters), and vocoded one second of real
speech. It agrees with upstream to **5.69e-05 relative** on the waveform, inside
the 2.71e-04 that upstream's own float32-vs-float64 error allows, and it is
**closer to the float64 truth than upstream's own float32 answer is** (ratio
0.61).

The finding this file exists to hold down is that the round's first answer was
wrong in a way its own method guaranteed:

  * **Every operator BigVGAN dispatches was already in `_aten_implemented()`.**
    14 distinct ops in the forward, 15 in `__init__`, 0 missing -- so the round
    reported no wall. Then the model was actually run, and it stopped twice, on
    `aten.sum.default` and `aten.view.default`: both on that list, both with
    dense kernels and golden cases for months, and **neither with a meta
    kernel**. `from_pretrained` builds on meta, so the list was answering a
    different question than the one being asked.
    `test_the_two_meta_kernels_are_the_meta_half_of_ops_already_implemented`
    asserts that shape from both sides so it cannot be forgotten.

  * **A coverage claim read off a manifest is vacuous if the manifest is
    empty.** "Every op in the list is implemented" passes trivially for an empty
    list, and that is precisely the nullification a later edit would not notice.
    `test_the_bigvgan_op_manifest_cannot_pass_by_being_empty` asserts the
    manifest's size *and* names the four operators that are load bearing for
    this model specifically (`replication_pad1d`, `kaiser_window.beta`, `sinc`,
    `convolution`), so truncating it fails rather than passes.

  * **The tolerance is not a choice.** docs/numerics/AGREE.md derives its number from
    upstream's own float32-vs-float64 error, and so does this one: upstream
    BigVGAN disagrees with *itself* across those two dtypes by **6.78e-05
    relative** on the waveform, which is ~57x the 1.186e-06 that AGREE.md's
    297-architecture sweep used. A 112M-parameter convolutional vocoder
    accumulates far more than the median architecture in that sweep, and a
    round that used AGREE.md's constant here would be calling upstream wrong.
    `test_the_tolerance_is_read_off_upstreams_own_error_not_chosen` recomputes
    it from the recorded measurement so it cannot be quietly widened.

`voice4_bigvgan_ops.json` is the measurement, captured on upstream torch 2.13.0
under `TorchDispatchMode`; `voice4_capture.py` regenerates it.

The replay under the shim needs the vendored tree *and* a converted checkpoint,
neither of which is hermetic, so it is behind `TORCH_C_VOICE4_ASSETS` and skips
loudly rather than silently passing when they are absent. docs/architectures/VOICE4.md §5 is
what it measured when it was run by hand.
"""

import json
import os
import subprocess
import sys

from test_shim import _C

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# The vendored tree is the one `vendor/vendor_torch.sh` and
# `vendor/install_shim.sh` build inside this checkout, and nothing may point
# it elsewhere. An override existed here while the tree was unbuilt; it is
# removed because `run.sh` refuses to run when the in-checkout `_C.abi3.so`
# does not match what was just compiled, and a suite reading a tree outside
# the checkout is exactly the case that guard cannot see.
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")
_MANIFEST_PATH = os.path.join(_HERE, "voice4_bigvgan_ops.json")

# AGREE.md §2's fixed floor, for the case where a model's own oracle error is
# smaller than what float32 costs anywhere. It is not the number used below --
# BigVGAN's own error is two orders of magnitude larger -- but the rule is
# `max(floor, factor * oracle)` and the floor has to be in it to be that rule.
_AGREE_FLOOR = 1.186e-06
# AGREE.md §2's oracle factor: "a difference is still not a defect if it is
# within 4x upstream's own distance from the float64 truth for that same
# output". Carried over rather than re-derived, and named so that widening it
# is a visible edit to a constant with a citation on it.
_ORACLE_FACTOR = 4.0

# The four operators that are load bearing for BigVGAN specifically, as opposed
# to the arithmetic every model in the repository already needed. Each is here
# for a reason the manifest alone would not say:
#
#   replication_pad1d   the anti-aliased activation pads by replication before
#                       every upsample and every downsample -- 218 calls. It was
#                       docs/architectures/VOICE.md rank 13, open at that round.
#   kaiser_window.beta  the resampling filters are built in __init__ from a
#                       Kaiser window; docs/architectures/VOICE3.md §2.1 landed it and this is
#                       the model that asked for it.
#   sinc                the other half of that filter (VOICE.md rank 6).
#   convolution         the generator IS a convolution stack; 334 calls, both
#                       ordinary and transposed, and 1-D throughout.
_LOAD_BEARING = (
    "aten.replication_pad1d.default",
    "aten.kaiser_window.beta",
    "aten.sinc.default",
    "aten.convolution.default",
)


def _manifest():
    with open(_MANIFEST_PATH) as fh:
        return json.load(fh)


def _tolerance(manifest):
    """The rule, computed rather than typed. AGREE.md §2: the fixed floor, or
    `_ORACLE_FACTOR` times upstream's own float32-vs-float64 error on this very
    output, whichever is larger."""
    return max(_AGREE_FLOOR, _ORACLE_FACTOR * manifest["upstream_f32_vs_f64_rel"])


def test_bigvgan_forward_needs_no_operator_this_shim_lacks():
    manifest = _manifest()
    implemented = set(_C._aten_implemented())
    missing = sorted(op for op in manifest["forward_ops"] if op not in implemented)
    assert not missing, (
        "BigVGAN's forward dispatches operators this build does not implement: "
        f"{missing}. VOICE4.md's whole claim is that the count is zero."
    )


def test_bigvgan_construction_needs_no_operator_this_shim_lacks():
    manifest = _manifest()
    implemented = set(_C._aten_implemented())
    missing = sorted(op for op in manifest["construction_ops"] if op not in implemented)
    assert not missing, (
        "BigVGAN's __init__ dispatches operators this build does not implement: "
        f"{missing}. Construction blocked three of docs/architectures/VOICE.md's five models, "
        "so this half is not a formality."
    )


def test_the_bigvgan_op_manifest_cannot_pass_by_being_empty():
    """The two tests above are `all(op in implemented)`, which an empty manifest
    satisfies. This is the test that makes emptying or truncating the manifest a
    failure rather than a pass."""
    manifest = _manifest()
    forward = manifest["forward_ops"]
    construction = manifest["construction_ops"]
    assert len(forward) == 14, f"forward op set was measured at 14 distinct ops, got {len(forward)}"
    assert len(construction) == 15, (
        f"construction op set was measured at 15 distinct ops, got {len(construction)}"
    )

    both = dict(forward)
    for op, n in construction.items():
        both[op] = both.get(op, 0) + n
    for op in _LOAD_BEARING:
        assert op in both, (
            f"{op} is not in the manifest -- it is one of the four operators that "
            "are load bearing for this model rather than for arithmetic in general, "
            "so a manifest without it is not a capture of BigVGAN"
        )

    # A capture with zero calls in it is a capture of nothing.
    assert sum(forward.values()) > 1000, "the forward capture records implausibly few calls"
    assert forward["aten.convolution.default"] == 334, (
        "the generator's convolution count is the coarsest witness that the whole "
        "stack ran and not merely its first layer"
    )
    assert forward["aten.replication_pad1d.default"] == 218, (
        "218 = two pads per anti-aliased activation, 109 of them; if the forward "
        "had stopped early this would be smaller"
    )


def test_the_tolerance_is_read_off_upstreams_own_error_not_chosen():
    manifest = _manifest()
    oracle = manifest["upstream_f32_vs_f64_rel"]

    # Upstream's own float32 answer is this far from its own float64 answer on
    # the very output being compared. Recorded to a loose band rather than
    # exactly, because it is a measurement and re-measuring may move the last
    # digits -- but it is nowhere near AGREE.md's population tolerance, and that
    # is the finding.
    assert 1e-05 < oracle < 1e-04, (
        f"upstream's own f32-vs-f64 error on the BigVGAN waveform was measured at "
        f"6.78e-05; got {oracle:.3e}"
    )
    assert oracle > 40 * _AGREE_FLOOR, (
        "the point of measuring this per-model is that a 112M-parameter vocoder "
        "accumulates far more than AGREE.md's 297-architecture p90; if that stops "
        "being true the derivation should be revisited, not the constant"
    )

    tol = _tolerance(manifest)
    assert tol == _ORACLE_FACTOR * oracle, (
        "the tolerance must be the oracle rule's output, not a typed constant"
    )
    assert tol > _AGREE_FLOOR, "the floor should not be what binds for this model"

    # The line above is a TAUTOLOGY on its own -- `tol` is *defined* as
    # `_ORACLE_FACTOR * oracle`, so it holds for any factor whatsoever. That was
    # found by nullification: setting `_ORACLE_FACTOR = 400.0` left this whole
    # file green, the end-to-end replay included, because a 100x wider tolerance
    # still accepts a correct answer. docs/architectures/VOICE4.md §6 records it as the one
    # nullification the round's first draft did not catch.
    #
    # So the factor is pinned to the value docs/numerics/AGREE.md §2 derived and cited.
    # Widening it is now an edit to this assertion, which is a visible act,
    # rather than an edit to a constant nobody checks.
    assert _ORACLE_FACTOR == 4.0, (
        "AGREE.md §2 derived this factor and pinned it there with "
        "`test_the_oracle_factor_is_stated_and_is_not_a_free_parameter`; this "
        "round carries the same number and may not quietly widen it"
    )


def test_the_tolerance_would_actually_reject_a_wrong_waveform():
    """The other half of the same finding. A tolerance is only a test if
    something can fail it, and every number this round measured *passed* -- so
    nothing here had ever shown the comparison is able to say no.

    This drives the comparison the replay makes with a deliberately wrong
    waveform at ten times the tolerance and asserts it is rejected, then with
    one at a tenth and asserts it is accepted. Needs no assets: what is under
    test is the comparison, not the model."""
    manifest = _manifest()
    tol = _tolerance(manifest)
    scale = manifest["audio_absmax_f32"]
    want = [0.0, scale, -scale, scale / 2]

    bad = list(want)
    bad[1] = want[1] + 10 * tol * scale
    rel_bad = max(abs(a - b) for a, b in zip(bad, want)) / scale
    assert rel_bad > tol, (
        f"the comparison must reject a waveform {rel_bad:.3e} away when the "
        f"tolerance is {tol:.3e} -- if this does not hold, the replay test is "
        "incapable of failing and is not a measurement"
    )

    ok = list(want)
    ok[1] = want[1] + 0.1 * tol * scale
    rel_ok = max(abs(a - b) for a, b in zip(ok, want)) / scale
    assert rel_ok <= tol, f"{rel_ok:.3e} should have been accepted at {tol:.3e}"


def test_the_manifest_says_which_checkpoint_and_which_upstream_it_came_from():
    """A measurement whose provenance is not recorded cannot be reproduced or
    contradicted. docs/numerics/AGREE.md's own opening caveat is that its numbers are a
    snapshot of one checkout; this manifest carries the equivalent."""
    manifest = _manifest()
    assert manifest["checkpoint"] == "nvidia/bigvgan_v2_24khz_100band_256x"
    assert manifest["num_parameters"] == 112414512
    assert manifest["upstream_torch"] == "2.13.0"
    assert manifest["model"] == "voicestudio.models.bigvgan.BigVGANModel"
    # Real weights, not from_config: docs/architectures/VOICE.md §2.1 is explicit that the two
    # are different claims, and four of its five models could only make the
    # weaker one.
    assert manifest["mel_shape"] == [1, 100, 93]
    assert manifest["audio_shape"] == [1, 23808]
    # 93 mel frames at a hop of 256 upsampled 256x is 23808 samples: the model
    # emits one waveform sample per input frame per hop, and a stack that had
    # silently dropped an upsample stage would not land on this number.
    assert manifest["audio_shape"][-1] == manifest["mel_shape"][-1] * 256


def test_the_generated_waveform_is_speech_scaled_and_not_silence_or_clipping():
    """The vocoder's output is bounded by a tanh, so `absmax` near 1.0 would mean
    it is clipping and near 0.0 would mean it produced silence -- either of which
    a relative-error comparison against an equally degenerate reference would
    happily call agreement. docs/numerics/AGREE.md excludes `degenerate` outputs from its
    denominator for exactly this reason; this asserts the case does not need
    excluding."""
    manifest = _manifest()
    for key in ("audio_absmax_f32", "audio_absmax_f64"):
        absmax = manifest[key]
        assert 0.05 < absmax < 0.95, (
            f"{key} is {absmax}, which is silence or clipping rather than speech"
        )


def test_bigvgan_replays_under_the_shim_and_agrees_with_upstream():
    """The end-to-end replay. Needs the vendored tree (`install_shim.sh`) and a
    converted checkpoint plus the captured mel, which are not hermetic -- so it
    is opt-in through `TORCH_C_VOICE4_ASSETS` and says why it skipped instead of
    passing quietly. docs/architectures/VOICE4.md §5 records the run."""
    assets = os.environ.get("TORCH_C_VOICE4_ASSETS")
    if not assets:
        print("   (skipped: TORCH_C_VOICE4_ASSETS is not set -- see docs/architectures/VOICE4.md §5)")
        return
    if not os.path.isfile(_VENDOR_SHIM):
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return

    manifest = _manifest()
    tol = _tolerance(manifest)
    # The mel arrives as JSON rather than through `numpy.load`, because
    # `torch.from_numpy` is itself not implemented in the shim -- the first wall
    # this replay hit, and one that belongs to the harness rather than to
    # BigVGAN (docs/architectures/VOICE4.md §4). Loading it the other way keeps the thing
    # under test the model.
    script = r"""
import json, os, sys
import torch
assets = os.environ["TORCH_C_VOICE4_ASSETS"]
sys.path.insert(0, os.path.join(assets, "pkg"))
sys.path.insert(0, os.path.join(assets, "stubs"))
from vsbig import BigVGANModel
marker = "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"
with open(os.path.join(assets, "ref", "mel_input.json")) as fh:
    mel = torch.tensor(json.load(fh), dtype=torch.float32)
model = BigVGANModel.from_pretrained(os.path.join(assets, "bigvgan-converted"),
                                     dtype=torch.float32).eval()
with torch.no_grad():
    audio = model(input_features=mel).audio_values
json.dump({"_marker": marker, "audio": audio.reshape(-1).tolist()}, sys.stdout)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, f"the shim replay failed to run:\n{proc.stderr[-4000:]}"
    got = json.loads(proc.stdout)
    assert got["_marker"] == "shim", (
        f"the replay imported the wrong torch ({got['_marker']}) -- every number "
        "below would have been about upstream comparing against itself"
    )

    with open(os.path.join(assets, "ref", "audio_f32.json")) as fh:
        want = json.load(fh)
    shim = got["audio"]
    assert len(shim) == len(want), f"waveform length {len(shim)} != {len(want)}"
    scale = max(abs(v) for v in want)
    rel = max(abs(a - b) for a, b in zip(shim, want)) / scale
    print(f"   (bigvgan waveform rel={rel:.3e} tol={tol:.3e} scale={scale:.4f})")
    assert rel <= tol, (
        f"the shim's waveform is {rel:.3e} from upstream's, past the {tol:.3e} that "
        "upstream's own float32-vs-float64 error allows"
    )


# --------------------------------------------------------------------------
# The two meta kernels this round landed, each against upstream's own meta
# answer. docs/architectures/VOICE4.md §4.
#
# These are NOT new operators: `aten.sum.default` and `aten.view.default` have
# had dense kernels and golden cases for a long time. What was missing was
# their *meta* half, and `_aten_implemented()` cannot see the difference --
# which is the finding the whole round turns on. So the check here cannot be a
# golden case (a meta tensor has no values to compare, as
# `test_ops_without_a_meta_kernel_name_themselves` says) and is instead a
# shape-and-dtype comparison against real torch in its own process.

_META_PROBE = r"""
import json, sys
import torch

out = {"_marker": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}
DTYPES = {
    "f32": torch.float32, "f64": torch.float64, "f16": torch.float16,
    "bf16": torch.bfloat16, "i64": torch.int64, "i32": torch.int32,
    "i16": torch.int16, "u8": torch.uint8, "bool": torch.bool,
}

def rec(name, fn):
    try:
        t = fn()
    except Exception as e:
        out[name] = {"error": type(e).__name__}
    else:
        out[name] = {"shape": list(t.shape), "dtype": str(t.dtype), "is_meta": bool(t.is_meta)}

for tag, dt in DTYPES.items():
    rec(f"sum_{tag}", lambda dt=dt: torch.empty(2, 3, dtype=dt, device="meta").sum())
# The explicit `dtype=` wins over the natural rule, in both directions.
rec("sum_f32_to_f64", lambda: torch.empty(2, 3, dtype=torch.float32, device="meta").sum(dtype=torch.float64))
rec("sum_i64_to_f32", lambda: torch.empty(2, 3, dtype=torch.int64, device="meta").sum(dtype=torch.float32))
# Rank 0 in, rank 0 out -- a full reduction of a scalar is still a scalar.
rec("sum_rank0", lambda: torch.empty((), dtype=torch.float32, device="meta").sum())
rec("sum_rank3", lambda: torch.empty(2, 3, 4, dtype=torch.float32, device="meta").sum())
# An empty tensor sums to a scalar too; the shape rule does not consult numel.
rec("sum_empty", lambda: torch.empty(0, dtype=torch.float32, device="meta").sum())

for tag, dt in DTYPES.items():
    rec(f"view_{tag}", lambda dt=dt: torch.empty(2, 3, dtype=dt, device="meta").view(3, 2))
rec("view_wildcard", lambda: torch.empty(2, 3, dtype=torch.float32, device="meta").view(-1))
rec("view_wildcard_mid", lambda: torch.empty(2, 3, 4, dtype=torch.float32, device="meta").view(2, -1))
rec("view_add_axes", lambda: torch.empty(6, dtype=torch.float32, device="meta").view(1, 1, 6))
rec("view_to_rank0", lambda: torch.empty(1, dtype=torch.float32, device="meta").view(()))
rec("view_bad_numel", lambda: torch.empty(2, 3, dtype=torch.float32, device="meta").view(4, 2))
rec("view_two_wildcards", lambda: torch.empty(2, 3, dtype=torch.float32, device="meta").view(-1, -1))
rec("view_empty", lambda: torch.empty(0, dtype=torch.float32, device="meta").view(0, 3))
# BigVGAN's own line, at the exact kernel size the 218 filters use.
rec("view_bigvgan", lambda: torch.empty(12, dtype=torch.float32, device="meta").view(1, 1, 12))

json.dump(out, sys.stdout)
"""

_meta_cache = {}


def _meta_run(side):
    if side in _meta_cache:
        return _meta_cache[side]
    env = dict(os.environ)
    if side == "shim":
        if not os.path.isfile(_VENDOR_SHIM):
            _meta_cache[side] = None
            return None
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _META_PROBE], capture_output=True, text=True,
        env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, f"the {side} meta probe failed:\n{proc.stderr[-3000:]}"
    data = json.loads(proc.stdout)
    assert data["_marker"] == side, (
        f"the {side} meta probe imported the wrong torch ({data['_marker']})"
    )
    _meta_cache[side] = data
    return data


def _meta_compare(prefix):
    shim = _meta_run("shim")
    if shim is None:
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return 0
    upstream = _meta_run("upstream")
    keys = sorted(k for k in upstream if k.startswith(prefix))
    assert keys, f"the probe recorded no {prefix}* cases"
    for key in keys:
        assert shim.get(key) == upstream[key], (
            f"{key}: shim {shim.get(key)} != upstream {upstream[key]}"
        )
    return len(keys)


def test_the_sum_meta_kernel_answers_what_upstream_answers():
    """Shape is always rank 0; the dtype is the interesting half. Every integral
    input -- `bool` and `uint8` included -- widens to `int64`, and only the
    floating ones keep their own. Restating that rule instead of calling the
    dense kernel's `sum_natural_tag` is how a meta kernel comes to promise a
    dtype the dense kernel would not produce."""
    n = _meta_compare("sum_")
    if n:
        assert n >= 13, f"only {n} sum cases were compared"


def test_the_view_meta_kernel_answers_what_upstream_answers():
    """Including the two refusals. `view(4, 2)` on a 6-element tensor and
    `view(-1, -1)` both have to fail: with no candle call behind it, a meta
    `view` that only resolved the wildcard would answer a shape holding a
    different number of elements than it was given."""
    n = _meta_compare("view_")
    if n:
        assert n >= 15, f"only {n} view cases were compared"


def test_the_meta_view_refusals_are_refusals_and_not_answers():
    """`_meta_compare` would pass if BOTH sides raised the same exception class
    for every case, which is what a meta kernel that refused everything looks
    like. This names which cases must raise and which must not, so that
    direction cannot be lost."""
    upstream = _meta_run("upstream")
    for key in ("view_bad_numel", "view_two_wildcards"):
        assert "error" in upstream[key], f"{key} was expected to refuse upstream"
    for key in ("view_bigvgan", "view_wildcard", "sum_f32", "sum_bool"):
        assert "error" not in upstream[key], (
            f"{key} was expected to ANSWER upstream, got {upstream[key]}"
        )
        assert upstream[key]["is_meta"] is True, key


def test_the_two_meta_kernels_are_the_meta_half_of_ops_already_implemented():
    """The round's actual finding, asserted rather than only written down: both
    ops were in `_aten_implemented()` the whole time. A round that had read that
    list as "BigVGAN is covered" -- which this one did, at first -- would have
    reported no wall and been wrong."""
    implemented = set(_C._aten_implemented())
    for op in ("aten.sum.default", "aten.view.default"):
        assert op in implemented, (
            f"{op} should be in _aten_implemented() -- it has had a dense kernel "
            "and golden cases since long before this round"
        )
    manifest = _manifest()
    both = dict(manifest["forward_ops"])
    for op, n in manifest["construction_ops"].items():
        both[op] = both.get(op, 0) + n
    assert "aten.sum.default" in both and "aten.view.default" in both, (
        "both walls must be in the captured manifest, or the capture is not what "
        "hit them"
    )


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
