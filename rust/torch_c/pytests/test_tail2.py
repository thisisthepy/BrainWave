"""The complex-number wall, asserted from both sides.

`docs/kernels/COMPLEX.md` is the round; this file is the part of it that fails when
something changes. Four names in the `docs/architectures/ARCH100.md` tail --
`view_as_complex` (`llama4`, at construction), `polar` (`llama4_text`),
`fft_fftn` (`fnet`) and the `torch.stft` a concurrent speech round wants -- are
one question: **can this shim hold a complex tensor at all?**

The measured answer is no, and the reason is one level below this repository:
`candle_core::DType` has no complex variant, and unlike `torch.int8`
(docs/numerics/INT8.md) it cannot get one without relaxing `WithDType`'s `PartialOrd`
bound, which every comparison and reduction kernel in candle is generic over.
COMPLEX.md §2 sizes that.

So there are two things worth locking down, and they pull in opposite
directions:

1.  **The refusal is correct today, and must stay correct.** A complex tensor
    that silently loses its imaginary part still returns plausible numbers --
    it is the failure mode that survives a smoke test, and `docs/devices/VULKAN2.md`
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
    # Reports the DC term, not just a shape: docs/kernels/FFT.md's `normalization` is a
    # code rather than a `norm=` string and the same codes serve both
    # directions, so a wrong reading scales every bin uniformly -- which a
    # shape check cannot see. `arange(8)` sums to 28.
    "fft_fftn": lambda: {
        "shape": list(torch.fft.fftn(torch.arange(8.).reshape(2, 4), dim=(0, 1)).shape),
        "dtype": str(torch.fft.fftn(torch.arange(8.).reshape(2, 4), dim=(0, 1)).dtype),
        "dc_real": float(torch.fft.fftn(torch.arange(8.).reshape(2, 4), dim=(0, 1)).real.flatten()[0]),
        "dc_imag": float(torch.fft.fftn(torch.arange(8.).reshape(2, 4), dim=(0, 1)).imag.flatten()[0]),
    },
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
        # NOT `repr(r)`. Once `view_as_complex` started returning a tensor
        # (docs/kernels/COMPLEX2.md), `repr` of it reached `torch/_tensor_str.py`,
        # which calls `self.resolve_conj()` for every complex tensor -- and
        # that is not implemented, so formatting the *success* of one probe
        # crashed the whole script and reported every other probe as a
        # failure to run. A probe that cannot survive its own subject
        # succeeding is not a probe. Recorded as a live gap in
        # docs/kernels/COMPLEX2.md §7: `print(z)` on a complex tensor still refuses.
        # A dict result is a probe that reports values on purpose; anything
        # else gets the type-and-shape string. `repr` is deliberately not used:
        # once `view_as_complex` began returning a tensor it reached
        # `torch/_tensor_str.py` and refused (docs/kernels/COMPLEX2.md).
        out[name] = {"ok": r if isinstance(r, dict)
                     else f"{type(r).__name__}{tuple(getattr(r, 'shape', ()))}"}
json.dump(out, sys.stdout)
"""

_probe_cache = {}


def _probe():
    """Run the probe script in the vendored tree, once per process.

    Returns `None` when the vendored shim is not installed -- the same silent
    skip `test_shim.py`'s checkpoint section uses, and for the same reason
    (docs/models/E2E.md): `pytests/run.sh` builds the *standalone* `_C`, while this
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


def _eval_in_vendored_tree(body):
    """Run `body` in a subprocess with the vendored shim on PYTHONPATH and
    return its `OUT` dict. The marker check is the same one `_probe` makes and
    for the same reason: without it the assertions could be about upstream."""
    script = (
        body
        + "\nimport json, sys\n"
        "assert hasattr(torch._C, '_aten_implemented'), 'not the shim'\n"
        "json.dump(OUT, sys.stdout)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


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


def test_view_as_complex_and_polar_compute_against_the_recorded_spec():
    """`llama4` and `llama4_text`'s walls, **now closed** (docs/kernels/COMPLEX2.md).

    This test asserted the *absence* of these three operators, exactly as this
    module's docstring says such tests should: a work item that goes red when
    the work is done. It went red. What replaces it is the inversion that
    docstring asks for -- the same probes, compared against
    `_VIEW_AS_COMPLEX_SPEC`, which was transcribed from upstream 2.13.0 in a
    separate process *before* any of this existed.

    Deliberately checked against the recorded spec rather than against a live
    upstream run: `pytests/test_complex.py` does the live element-wise
    comparison, and having one of the two be a transcription made before the
    implementation is what makes them independent. If the implementation and a
    live oracle ever agreed on something both got wrong, this is the copy that
    would not move.
    """
    p = _probe()
    if p is None:
        return
    s = _VIEW_AS_COMPLEX_SPEC
    for label in ("view_as_complex", "polar"):
        r = p[label]
        assert "ok" in r, (
            f"torch.{label} refused after docs/kernels/COMPLEX2.md landed it: {r}"
        )
    # `view_as_real` is the third op and its probe passes a *real* tensor, so
    # it still raises -- and must, with upstream's own sentence. That is a
    # different assertion from the two above, not a weaker one: it is the
    # refusal transcribed in `_UPSTREAM_REFUSALS`.
    assert p["view_as_real"].get("raised") == "RuntimeError", p["view_as_real"]
    assert p["view_as_real"]["msg"] == _UPSTREAM_REFUSALS["view_as_real_on_real"]

    # The values, not merely the absence of an exception. `_probe`'s script
    # only reports a repr, so the numbers come from a second subprocess that
    # evaluates the spec's own fixture.
    got = _eval_in_vendored_tree(
        "import torch\n"
        "z = torch.view_as_complex(torch.tensor([[1.,2.],[3.,4.],[5.,6.]]))\n"
        "OUT = {'shape': list(z.shape), 'dtype': str(z.dtype),\n"
        "       'element_size': z.element_size(),\n"
        "       'real': torch.real(z).tolist(),\n"
        "       'imag': torch.imag(z).tolist(),\n"
        "       'roundtrip': torch.view_as_real(z).tolist()}\n"
    )
    assert got["shape"] == list(s["shape"]), got["shape"]
    assert got["dtype"] == s["dtype"], got["dtype"]
    assert got["element_size"] == s["element_size"], got["element_size"]
    assert got["real"] == s["real"], got["real"]
    # **The one that matters.** A representation that dropped the imaginary
    # part would satisfy every line above and fail only here.
    assert got["imag"] == s["imag"], (
        f"the imaginary part is {got['imag']}, upstream's is {s['imag']}. "
        f"Losing it is the failure that still returns plausible numbers."
    )
    assert got["roundtrip"] == s["view_as_real_roundtrip"], got["roundtrip"]


def test_view_as_complex_copies_rather_than_aliasing():
    """The narrowing `_VIEW_AS_COMPLEX_SPEC["aliases_its_base"]` recorded.

    Upstream's `view_as_complex` is a view; `Repr::Complex` is a pair of real
    tensors and cannot alias an interleaved base. The spec above says upstream
    aliases; this asserts the shim does not, so the divergence is pinned in the
    file that recorded it rather than only in `test_complex.py`.
    """
    p = _probe()
    if p is None:
        return
    assert _VIEW_AS_COMPLEX_SPEC["aliases_its_base"] is True
    got = _eval_in_vendored_tree(
        "import torch\n"
        "b = torch.tensor([[1., 2.]])\n"
        "v = torch.view_as_complex(b)\n"
        "b[0, 0] = 99.\n"
        "OUT = {'through': torch.view_as_real(v).tolist()[0][0]}\n"
    )
    assert got["through"] == 1.0, (
        f"the shim's view_as_complex now aliases its base (saw "
        f"{got['through']}). That matches upstream and is an improvement, but "
        f"docs/kernels/COMPLEX2.md records the copy as a narrowing -- remove it there "
        f"first."
    )


def test_stft_and_fft_fftn_both_compute_now():
    """Inverted a fourth time, by the merge of two rounds that each did half.

    The sequence is the record and it is why this test was never deleted:

      1. `stft` refused at `_nn.pad(mode='reflect')`, before any complex value
         existed -- `docs/kernels/COMPLEX.md`'s reason for ordering reflect pad first.
      2. `docs/bindings/BIND2.md` gave the pad its kernel; the wall moved one line, to
         `stft`'s own missing table row.
      3. `docs/kernels/FFT.md` implemented the transform; `stft` computed, and this
         test kept asserting `fft_fftn`'s absence, which was still true.
      4. `docs/bindings/BIND4.md` bound `_fft.fft_fftn` and `docs/kernels/COMPLEX3.md` taught
         `_to_copy(complex64)`. **Neither alone was enough** -- BIND4 measured
         `fnet` stopping *inside* `fft_fftn` at the `_to_copy` gate, and said
         it should clear once the two met. It did, on this merge.

    So what is asserted is the values, and specifically the DC term, because a
    transform that returns the right shape and the wrong normalisation is the
    plausible failure `docs/kernels/FFT.md` warned about: `normalization` is a code,
    not a `norm=` string, and the same three codes serve both directions, so a
    shim reading it as a string is right forward and wrong by `n` inverse.
    `fftn` of `arange(8)` has DC equal to the sum, 28.
    """
    r = _probe()
    if r.get("skip"):
        return

    for label in ("stft", "stft_real"):
        assert "ok" in r[label], f"{label} refuses again: {r[label]}"

    got = r["fft_fftn"]
    assert "ok" in got, f"fft_fftn refuses again: {got}"
    assert got["ok"]["shape"] == [2, 4], got
    assert got["ok"]["dtype"] == "torch.complex64", got
    # DC = sum of arange(8) = 28, and the imaginary part of DC is exactly 0.
    assert abs(got["ok"]["dc_real"] - 28.0) < 1e-4, got
    assert abs(got["ok"]["dc_imag"]) < 1e-6, got

def _linalg_norm_value():
    """`float(linalg_norm(ones(2, 2)))` from the vendored tree, as a number.

    Separate from `_probe` because `_probe` deliberately does not stringify
    results any more, and a norm's whole content is its value.
    """
    import json
    import subprocess
    import sys

    script = (
        "import json, torch\n"
        "assert hasattr(torch._C, '_aten_implemented'), 'not the shim'\n"
        "print(json.dumps(float(torch._C._linalg.linalg_norm(torch.ones(2, 2)))))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, env=env, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_linalg_norm_binding_landed_on_the_kernel_that_was_already_there():
    """OWL-ViT's wall (`owlv2`, `owlvit`). This test used to pin the *gap* --
    the kernel present, the name unreachable, COMPLEX.md §6 naming the install
    site -- and said in its own message that when the binding landed it should
    become an element-wise comparison instead. It has landed (docs/bindings/BINDINGS.md),
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
    # The probe records a type-and-shape string, not `repr(r)` -- once
    # `view_as_complex` began returning a tensor, `repr` reached
    # `torch/_tensor_str.py` and refused (docs/kernels/COMPLEX2.md). So the *value*
    # is asserted here directly rather than looked for in that string, which
    # is the stronger check anyway: `linalg_norm(ones(2, 2))` is the Frobenius
    # norm of four ones, and 2.0 is the one number a correct binding gives.
    # `Tensor()` -- a 0-dim scalar, which is the shape a norm has.
    assert _probe()["linalg_norm"]["ok"] == "Tensor()", _probe()["linalg_norm"]
    assert _linalg_norm_value() == 2.0, _linalg_norm_value()

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


def test_only_the_expected_complex_ops_landed():
    """The cheap sweep, re-pointed rather than deleted.

    It asserted that nothing named complex had appeared in the op list. Five
    such ops now exist -- but in `_aten_implemented_awaiting_golden()`, not in
    `_aten_implemented()`, because golden compares by reading both sides as
    real tensors and a complex tensor has nothing to read (aten.rs's note on
    that list). So the sweep still holds for the advertised list, and the
    parked list is pinned to exactly the five.

    `fft_` and `_vmap_` are unchanged: neither landed, and if either appears
    this file is the thing that has to be revisited.
    """
    implemented = set(_C._aten_implemented())
    parked = set(_C._aten_implemented_awaiting_golden())

    leaked = sorted(
        op for op in implemented
        if any(k in op for k in ("view_as_complex", "view_as_real", "polar",
                                 "fft_", "_vmap_"))
    )
    assert not leaked, (
        f"these are advertised to golden but golden has no way to compare a "
        f"complex result: {leaked}"
    )
    assert sorted(op for op in parked if any(
        k in op for k in ("view_as_complex", "view_as_real", "polar", "real",
                          "imag"))) == [
        "aten.imag.default",
        "aten.polar.default",
        "aten.real.default",
        "aten.view_as_complex.default",
        "aten.view_as_real.default",
    ]
    # **Inverted by docs/kernels/FFT.md**, which is what this file's docstring asks an
    # implementing round to do rather than delete the assertion. Three
    # `_fft_*` keys landed, all three parked for the same reason the five
    # above are: two of them return a complex tensor and one takes one, so a
    # golden comparison has nothing dense to read on at least one side. The
    # list is pinned, so a fourth appearing is still a failure here.
    assert sorted(op for op in parked if "fft_" in op) == [
        "aten._fft_c2c.default",
        "aten._fft_c2r.default",
        "aten._fft_r2c.default",
    ]
    assert not [op for op in implemented if "fft_" in op], (
        "an _fft_* op is advertised to golden, which cannot compare a complex "
        "result -- see aten.rs's note on IMPLEMENTED_AWAITING_GOLDEN"
    )
    # `aten.stft.*` IS advertised, and deliberately: its `return_complex=False`
    # form is real on both sides (docs/kernels/FFT.md §3).
    assert {"aten.stft.default", "aten.stft.center"} <= implemented
    # Still nothing: docs/kernels/COMPLEX.md §3.3 step 5 (`fft_fftn`, for `fnet`) is a
    # separate decision -- `_fft_r2c` takes a multi-axis `dim` here, which is
    # the arithmetic `fft_fftn` needs, but no `aten.fft_fftn.default` kernel
    # or spelling was added. §7 (vmap) is untouched.
    assert not [op for op in implemented | parked
                if "fft_fftn" in op or "_vmap_" in op]


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
