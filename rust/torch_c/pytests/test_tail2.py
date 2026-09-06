"""The complex-number wall, asserted from both sides.

`docs/COMPLEX.md` is the round; this file is the part of it that fails when
something changes. Four names in the `docs/ARCH100.md` tail --
`view_as_complex` (`llama4`, at construction), `polar` (`llama4_text`),
`fft_fftn` (`fnet`) and the `torch.stft` a concurrent speech round wants -- are
one question: **can this shim hold a complex tensor at all?**

The measured answer is no, and the reason is one level below this repository:
`candle_core::DType` has no complex variant, and unlike `torch.int8`
(docs/INT8.md) it cannot get one without relaxing `WithDType`'s `PartialOrd`
bound, which every comparison and reduction kernel in candle is generic over.
COMPLEX.md §2 sizes that.

So there are two things worth locking down, and they pull in opposite
directions:

1.  **The refusal is correct today, and must stay correct.** A complex tensor
    that silently loses its imaginary part still returns plausible numbers --
    it is the failure mode that survives a smoke test, and `docs/VULKAN2.md`
    set the standard that the wrong-answer path be *unrepresentable* rather
    than merely unused. `TorchDType` already carries `Complex32/64/128` tags
    whose `storage()` is `None`, so construction refuses by name. The tests
    below assert that, because it is the property an eventual implementation
    is most likely to break by accident.

2.  **The refusals are work items, and must stop being true when the work is
    done.** These assert the *absence* of each operator by name. That is the
    notification pattern `methods.json`'s README describes for `amax`: a round
    that implements `view_as_complex` will find this file red, and the diff
    that turns it green is where the real numerics go. If you are that round,
    do not delete these -- invert them against the upstream values recorded in
    `_LLAMA4_ROPE_SPEC` and `_VIEW_AS_COMPLEX_SPEC`, which were measured from
    upstream torch 2.13.0 in a separate process and are the oracle.

Nothing here needs numpy or a network.
"""

import json
import os
import subprocess
import sys

from test_shim import _C

# `torch` is deliberately NOT imported at module level. This process has plain
# upstream torch installed (the suite's venv has it, for the oracle
# comparisons `test_shim.py` makes), so `import torch` here would import
# *upstream* and every refusal asserted below would be asserted against the
# wrong library -- passing or failing for reasons that have nothing to do with
# this shim. The same two-interpreter recipe `test_shim.py`'s checkpoint
# section uses is used here instead: a subprocess with `torchnative/src/main`
# on PYTHONPATH gets the vendored, shim-backed `torch`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


# --------------------------------------------------------------------------
# The oracle, transcribed
# --------------------------------------------------------------------------
#
# Measured with `env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL python`, i.e. the
# real torch 2.13.0 in its own process, not this shim. Recorded here so that
# an implementing round has the expected values already in the tree rather
# than having to re-derive them, and so COMPLEX.md's claims about upstream
# semantics have a machine-readable copy.

# `Llama4VisionRotaryEmbedding` / `apply_rotary_emb`, on B,S,H,D = 2,3,4,8 with
# `torch.manual_seed(0)`. This is the whole complex surface `llama4` uses, and
# it is a *closed* pipeline: every complex value is produced by `polar` or
# `view_as_complex` and consumed by `view_as_real` inside one function.
_LLAMA4_ROPE_SPEC = {
    "polar_out_shape": (2, 3, 4),
    "polar_out_dtype": "torch.complex64",
    "view_as_complex_out_shape": (2, 3, 4, 4),
    "indexed_shape": (2, 3, 1, 4),
    "mul_out_shape": (2, 3, 4, 4),
    "view_as_real_flatten3_shape": (2, 3, 4, 8),
    "view_as_real_flatten3_dtype": "torch.float32",
    "checksum_sum": -8.959174156188965,
    "checksum_abs_sum": 142.30633544921875,
}

# `torch.view_as_complex(torch.tensor([[1.,2.],[3.,4.],[5.,6.]]))`
_VIEW_AS_COMPLEX_SPEC = {
    "shape": (3,),
    "dtype": "torch.complex64",
    "real": [1.0, 3.0, 5.0],
    "imag": [2.0, 4.0, 6.0],
    "view_as_real_roundtrip": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    "element_size": 8,
    "abs": [2.2360680103302, 5.0, 7.8102498054504395],
    # `view_as_complex` is a *view*: mutating the real base is visible through
    # it. A pair-of-tensors representation (COMPLEX.md §3) copies and loses
    # this. `llama4` does not depend on it -- all three of its call sites feed
    # a freshly computed expression that is never written to again -- but the
    # narrowing has to be stated rather than discovered.
    "aliases_its_base": True,
}

# Upstream's own refusals, which an implementation should reproduce verbatim
# rather than inventing. Measured the same way.
_UPSTREAM_REFUSALS = {
    "odd_last_dim": "Tensor must have a last dimension of size 2",
    "int_input": (
        "view_as_complex is only supported for half, float and double "
        "tensors, but got a tensor of scalar type: Long"
    ),
    "view_as_real_on_real": "view_as_real is only supported for complex tensors",
    "polar_dtype_mismatch": (
        "Expected object of scalar type Float but got scalar type Double "
        "for second argument"
    ),
}


def _refuses(fn):
    """Return the exception a call raises, or `None` if it returned."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return e
    return None


# One subprocess buys every torch-level probe below. Each entry is evaluated
# in the vendored tree and reported as either `{"ok": <repr>}` or
# `{"raised": <type>, "msg": <text>}`, so the assertions here read a result
# rather than catching an exception across a process boundary.
_PROBE_SCRIPT = r"""
import json, sys
import torch

assert hasattr(torch._C, "_aten_implemented"), "subprocess did not get the shim"

PROBES = {
    "view_as_complex": lambda: torch.view_as_complex(torch.ones(2, 2)),
    "view_as_real": lambda: torch.view_as_real(torch.ones(2)),
    "polar": lambda: torch.polar(torch.ones(2), torch.zeros(2)),
    "zeros_complex64": lambda: torch.zeros(2, dtype=torch.complex64),
    "empty_complex64": lambda: torch.empty(2, dtype=torch.complex64),
    "fft_fftn": lambda: torch._C._fft.fft_fftn(torch.ones(4)),
    "stft": lambda: torch.stft(torch.arange(64).float(), n_fft=16,
                               return_complex=True),
    "stft_real": lambda: torch.stft(torch.arange(64).float(), n_fft=16,
                                    return_complex=False),
    "linalg_norm": lambda: torch._C._linalg.linalg_norm(torch.ones(2, 2)),
    "linalg_norm_fro": lambda: torch._C._linalg.linalg_norm(
        torch.ones(2, 2), "fro"),
    "vmap_increment_nesting":
        lambda: torch._C._functorch._vmap_increment_nesting(2, "error"),
    "add_batch_dim":
        lambda: torch._C._functorch._add_batch_dim(torch.ones(2, 2), 0, 1),
}

out = {"_marker": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}
for name, fn in PROBES.items():
    try:
        r = fn()
    except Exception as e:
        out[name] = {"raised": type(e).__name__, "msg": str(e)}
    else:
        out[name] = {"ok": repr(r)[:200]}
json.dump(out, sys.stdout)
"""

_probe_cache = {}


def _probe():
    """Run the probe script in the vendored tree, once per process.

    Returns `None` when the vendored shim is not installed -- the same silent
    skip `test_shim.py`'s checkpoint section uses, and for the same reason
    (docs/E2E.md): `pytests/run.sh` builds the *standalone* `_C`, while this
    needs the one `vendor/install_shim.sh` writes into the tree.
    """
    if "r" in _probe_cache:
        return _probe_cache["r"]
    if not os.path.isfile(_VENDOR_SHIM):
        _probe_cache["r"] = None
        return None
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT],
        capture_output=True, text=True, env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"the vendored-tree probe failed to run at all:\n{proc.stderr[-2000:]}"
    )
    data = json.loads(proc.stdout)
    assert data["_marker"] == "shim", (
        "the probe subprocess imported upstream torch, not the shim -- every "
        "assertion below would have been about the wrong library"
    )
    _probe_cache["r"] = data
    return data


def _raised(name):
    """`(exc_type, message)` for a probe, or `None` if the probe returned a
    value, or the string `"skip"` when the vendored shim is absent."""
    p = _probe()
    if p is None:
        return "skip"
    r = p[name]
    if "ok" in r:
        return None
    return r["raised"], r["msg"]


# --------------------------------------------------------------------------
# 1. The tag exists and refuses -- the half that must never regress
# --------------------------------------------------------------------------


def test_the_three_complex_tags_are_named():
    """`TorchDType` carries them even though candle cannot store them.

    This is BOOL.md's split (`_C` owns the tag, candle owns the storage) doing
    the job it was built for. The names are import-blocking: `_prims_common`
    and `_dtype_abbrs` build tables over `torch.complex64` while `import
    torch` is still running.
    """
    for name in ("complex32", "complex64", "complex128", "chalf", "cfloat", "cdouble"):
        assert hasattr(_C, name), f"torch.{name} is missing"
    assert repr(_C.complex64) == "torch.complex64"
    assert isinstance(_C.complex64, _C.dtype)
    # The three aliases are the same object as the three primary names, as
    # upstream: `torch.cfloat is torch.complex64`.
    assert _C.chalf == _C.complex32
    assert _C.cfloat == _C.complex64
    assert _C.cdouble == _C.complex128


def test_complex_tags_report_no_storage():
    """The single fact the whole refusal rests on.

    If this ever becomes True without a real complex representation landing
    underneath it, every constructor below starts allocating *something* --
    and whatever it allocates will not have an imaginary part. That is the
    silent-wrong-number path, so it is asserted directly rather than through
    any operator that happens to consult it.
    """
    for dt in (_C.complex32, _C.complex64, _C.complex128):
        assert dt._has_storage is False, (
            f"{dt} claims candle storage. Either a complex representation "
            f"landed (in which case invert this test against "
            f"_VIEW_AS_COMPLEX_SPEC) or a complex tag was aliased onto a real "
            f"dtype, which loses the imaginary part silently."
        )
    # The neighbours, to show this test can fail in the other direction too:
    # a change that made `_has_storage` uniformly False would pass the loop
    # above and be caught here.
    assert _C.float32._has_storage is True
    assert _C.bool._has_storage is True


def test_to_complex_and_to_real_pair_up():
    """`dtype.to_complex()` / `.to_real()` are exact inverses on the three
    real widths that have a complex partner. These are pure tag arithmetic --
    no storage is involved -- which is why they work today and are worth
    pinning: an implementation will build on them.
    """
    for real, cplx in (
        (_C.float16, _C.complex32),
        (_C.float32, _C.complex64),
        (_C.float64, _C.complex128),
    ):
        assert real.to_complex() == cplx, f"{real}.to_complex()"
        assert cplx.to_real() == real, f"{cplx}.to_real()"


def test_constructing_a_complex_tensor_refuses_by_name():
    """...and names the dtype, not a near neighbour.

    The message has to say `complex64`, because the next thing the reader does
    is decide whether the gap is their dtype or their operator.
    """
    for probe in ("zeros_complex64", "empty_complex64"):
        r = _raised(probe)
        if r == "skip":
            return
        assert r is not None, (
            f"{probe} returned a tensor. Whatever it returned has no "
            f"imaginary part -- see this module's docstring."
        )
        exc, msg = r
        assert exc == "NotImplementedError", f"{probe}: {exc}: {msg}"
        assert "complex64" in msg, (
            f"{probe} refused without naming the dtype, so the reader cannot "
            f"tell whether the gap is the dtype or the operator: {msg}"
        )


# --------------------------------------------------------------------------
# 2. The four operators, asserted absent -- the half that should go red
# --------------------------------------------------------------------------


def test_view_as_complex_and_polar_are_not_implemented():
    """`llama4` and `llama4_text`'s walls, in `docs/ARCH100.md`'s tail.

    `llama4` is the one blocked at *construction*
    (`Llama4VisionRotaryEmbedding.__init__` builds a complex buffer), which is
    why it cannot be tested at all until this moves.
    """
    for label in ("view_as_complex", "view_as_real", "polar"):
        r = _raised(label)
        if r == "skip":
            return
        assert r is not None, (
            f"torch.{label} computed something. If a complex representation "
            f"landed, this file's specs are the oracle to check it against."
        )
        exc, msg = r
        assert exc == "NotImplementedError", f"{label}: {exc}: {msg}"


def test_fft_and_stft_are_not_implemented():
    """`fnet` (`fft_fftn`) and the `torch.stft` a speech round wants.

    Worth separating: `stft`'s *first* wall is not complex at all. Measured,
    it is `torch._C._nn.pad(mode='reflect')`, which `stft` calls before any
    transform happens. So `return_complex=False` does not route around this
    round -- COMPLEX.md §5.
    """
    r = _raised("fft_fftn")
    if r == "skip":
        return
    assert r is not None, "fft_fftn computed something"
    exc, msg = r
    assert exc == "NotImplementedError", f"{exc}: {msg}"
    assert "fft_fftn" in msg, msg

    # Both spellings of stft, because `return_complex=False` is the obvious
    # thing to try next and it does not route around anything: the wall is
    # reached before the transform.
    for label in ("stft", "stft_real"):
        exc, msg = _raised(label)
        assert exc == "NotImplementedError", f"{label}: {exc}: {msg}"
        assert "pad" in msg, (
            f"{label}'s first wall moved. It was `_C._nn.pad(mode='reflect')`, "
            f"which is reached before any complex value exists; if it is now "
            f"something else, COMPLEX.md §5's advice to the speech round is "
            f"stale. Got: {msg}"
        )


def test_linalg_norm_binding_landed_on_the_kernel_that_was_already_there():
    """OWL-ViT's wall (`owlv2`, `owlvit`). This test used to pin the *gap* --
    the kernel present, the name unreachable, COMPLEX.md §6 naming the install
    site -- and said in its own message that when the binding landed it should
    become an element-wise comparison instead. It has landed (docs/BINDINGS.md),
    so this is the other half: the kernel is still what the binding is built
    on, and the binding computes rather than raising.

    The element-wise agreement with upstream lives in `test_bindings.py`, which
    runs upstream in a second subprocess. What is kept here is the *pairing* --
    that the name and the kernel it depends on do not come apart.
    """
    assert "aten.linalg_vector_norm.default" in set(_C._aten_implemented()), (
        "the kernel linalg_norm is built on is gone -- the binding in "
        "bootstrap.py now dispatches to nothing"
    )
    r = _raised("linalg_norm")
    if r == "skip":
        return
    assert r is None, (
        f"torch._C._linalg.linalg_norm raised: {r}. It is installed in "
        "bootstrap.py beside linalg_vector_norm; `linalg_norm(ones(2,2))` is "
        "the flattened 2-norm and should answer 2.0"
    )
    # `ones(2, 2)` flattens to four ones, so the 2-norm is exactly 2.
    assert "2." in _probe()["linalg_norm"]["ok"], _probe()["linalg_norm"]

    # And the half that is deliberately still shut: `ord` selects between
    # different computations, and the matrix ones are `linalg_matrix_norm`
    # upstream. Answering the flattened vector norm there would have the right
    # shape and the wrong number, so it refuses by name.
    fro = _raised("linalg_norm_fro")
    assert fro is not None and fro != "skip", (
        "linalg_norm(ord='fro') computed something. That is a matrix norm; if "
        "a matrix-norm kernel landed, compare it against upstream here"
    )
    exc, msg = fro
    assert exc == "NotImplementedError", f"{exc}: {msg}"
    assert "linalg_norm" in msg, msg


def test_vmap_increment_nesting_refuses_rather_than_counting():
    """The four ASR/T5 architectures' wall, and the one that must *not* be
    stubbed cheaply.

    Measured (COMPLEX.md §7): all four reach it through
    `transformers/masking_utils.py:348`'s real `torch.vmap` of a mask closure,
    and inside that closure `_add_batch_dim`/`_remove_batch_dim` do the actual
    work. A `_vmap_increment_nesting` that returns a level and does nothing
    would let the call proceed and then produce a *wrongly shaped or wrongly
    valued mask* -- the same silent-plausible-number shape as dropping an
    imaginary part.

    So this asserts the refusal deliberately. Returning an int here is a
    regression unless `_add_batch_dim` landed with it.
    """
    r = _raised("vmap_increment_nesting")
    if r == "skip":
        return
    assert r is not None, (
        "_vmap_increment_nesting returned. If batching landed, this test "
        "should be replaced by one that vmaps a closure and compares the "
        "mask element-wise against upstream -- not merely deleted."
    )
    exc, msg = r
    assert exc == "NotImplementedError", f"{exc}: {msg}"

    # The other half. A counter without batched tensors is the cheap stub this
    # test exists to forbid, so if `_add_batch_dim` ever starts working while
    # the counter refuses (or the reverse), that is the half-built state.
    assert _raised("add_batch_dim") is not None, (
        "_add_batch_dim returned while _vmap_increment_nesting refuses; the "
        "two halves of vmap have to land together or the nesting counter and "
        "the batched tensors disagree"
    )


# --------------------------------------------------------------------------
# 3. The specs are internally consistent
# --------------------------------------------------------------------------


def test_the_recorded_upstream_spec_is_self_consistent():
    """A transcription guard on the oracle above.

    These numbers came from a separate upstream process and nothing in this
    tree can currently reproduce them, so the only check available is that
    they agree with each other. Cheap, and it catches the copy-paste slip
    that would otherwise be discovered by an implementer trusting them.
    """
    s = _VIEW_AS_COMPLEX_SPEC
    assert len(s["real"]) == len(s["imag"]) == s["shape"][0]
    assert s["view_as_real_roundtrip"] == [
        [re, im] for re, im in zip(s["real"], s["imag"])
    ], "the round-trip does not rebuild the interleaved pairs it came from"
    for re, im, mag in zip(s["real"], s["imag"], s["abs"]):
        assert abs((re * re + im * im) ** 0.5 - mag) < 1e-6, (re, im, mag)
    # complex64 is a pair of float32, so 8 bytes. If this is ever 4, someone
    # aliased complex64 onto float32 -- exactly the drop this file guards.
    assert s["element_size"] == 2 * _C.float32.itemsize == _C.complex64.itemsize

    r = _LLAMA4_ROPE_SPEC
    b, sq, h, d = 2, 3, 4, 8
    assert r["polar_out_shape"] == (b, sq, d // 2)
    assert r["view_as_complex_out_shape"] == (b, sq, h, d // 2)
    assert r["mul_out_shape"] == r["view_as_complex_out_shape"]
    # view_as_real adds a trailing 2, flatten(3) folds it back into the head
    # dimension -- so the output is the input shape, which is the property
    # that makes the pipeline closed.
    assert r["view_as_real_flatten3_shape"] == (b, sq, h, d)
    assert r["checksum_abs_sum"] >= abs(r["checksum_sum"])


def test_complex_is_absent_from_the_implemented_op_list():
    """Nothing named complex has quietly appeared in `_aten_implemented()`.

    The op list is the tree's own answer to "what computes here", and
    `docs/ARCH100.md`'s tail was derived from it. This is the cheap sweep that
    would catch a complex op landing without this file being revisited.
    """
    implemented = set(_C._aten_implemented())
    leaked = sorted(
        op
        for op in implemented
        if any(
            k in op
            for k in ("view_as_complex", "view_as_real", "polar", "fft_", "_vmap_")
        )
    )
    assert not leaked, (
        f"these landed without this file being updated: {leaked}. Each needs "
        f"an element-wise comparison against upstream, not just a table entry."
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
