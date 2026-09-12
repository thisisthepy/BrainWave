"""The marshalling fix for `coreml._np` and the tempfile leak in `compute_plan`.

docs/graph/NPU2.md §7.3 recorded a reproducible **segfault at interpreter
shutdown** after marshalling a 1024x4096 tensor through `coreml._np`, which
builds four million Python floats via `tolist()`. The same defect family hit
`intelnpu.py` as a `MemoryError` (docs/devices/INTELNPU.md, `The weights path`)
and was fixed there with `torch._C._shim_f16_bytes`.

The CoreML fix is the general-dtype counterpart: `torch._C._shim_tensor_bytes`
reads the candle storage directly and `np.frombuffer` wraps the result, so no
Python scalar objects are built. This is bit-identical for every dtype in
`_NUMPY_DTYPES` (float32, float64, int32, int64, bool).

What is asserted here:

1. **The route**: `_np` uses `_shim_tensor_bytes`, not `tolist()`. Asserted by
   checking that no `PyFloat` allocation happens (peak Python heap), not by an
   absence of a crash -- a crash test only passes on the right OS/machine
   combination.
2. **Bit-identity**: the bytes route produces the same numpy arrays as the old
   `tolist()` route, for every dtype.
3. **Tempfile cleanup**: `compute_plan` no longer leaks its `mkdtemp`.
4. **`_shim_tensor_bytes` itself**: it exists, handles every mapped dtype, and
   refuses non-tensors.

Non-vacuity nullifications performed and recorded in docstrings.
"""

import os
import sys
import tracemalloc

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")


def _shim_torch():
    os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    import torch

    assert hasattr(torch._C, "_aten_implemented"), (
        "not the torchnative shim -- torch._C has no _aten_implemented"
    )
    return torch


def _coreml():
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    from torchnative.export import coreml

    return coreml


def _peak_python_heap(fn):
    """Run `fn()`, return `(result, peak_bytes)` of Python-managed heap."""
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


# ---------------------------------------------------------------------------
# 1. _shim_tensor_bytes exists and handles every mapped dtype
# ---------------------------------------------------------------------------


def test_shim_tensor_bytes_exists():
    """The Rust route is available on this build."""
    torch = _shim_torch()
    assert hasattr(torch._C, "_shim_tensor_bytes"), (
        "torch._C._shim_tensor_bytes is missing -- rebuild with install_shim.sh"
    )


def test_shim_tensor_bytes_handles_every_dtype_coreml_needs():
    """Every dtype in `_NUMPY_DTYPES` round-trips through the byte route.

    Nullified: make `shim_tensor_bytes` return an empty `PyBytes` and every
    dtype below fails with a shape mismatch or a zero-length buffer.
    """
    torch = _shim_torch()
    import numpy as np

    cases = [
        ("float32", "float32", torch.randn(10, 20)),
        ("float64", "float64", torch.randn(10, 20).to(torch.float64)),
        ("int32", "int32", torch.randint(0, 100, (10, 20)).to(torch.int32)),
        ("int64", "int64", torch.randint(0, 100, (10, 20))),
        ("bool", "bool", torch.tensor([[True, False], [False, True]])),
    ]
    for label, np_dtype, t in cases:
        raw = torch._C._shim_tensor_bytes(t)
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(tuple(t.shape))
        ref = np.array(t.tolist(), dtype=np_dtype).reshape(tuple(t.shape))
        assert np.array_equal(arr, ref), (
            f"{label}: bytes route differs from tolist route"
        )


def test_shim_tensor_bytes_refuses_non_tensors():
    """A non-tensor is refused by name, not silently converted."""
    torch = _shim_torch()
    try:
        torch._C._shim_tensor_bytes(42)
        assert False, "should have raised"
    except (NotImplementedError, TypeError):
        pass


# ---------------------------------------------------------------------------
# 2. _np uses the byte route, not tolist
# ---------------------------------------------------------------------------


def test_np_uses_byte_route_and_does_not_build_python_scalars():
    """The fix: `_np` goes through `_shim_tensor_bytes`, not `tolist()`.

    Asserted by the peak Python heap: 100x200 = 20,000 float32 elements would
    be about 640 KB of `PyFloat` objects via `tolist()`. The byte route
    allocates ~80 KB of bytes and nothing else, so the peak stays well under
    200 KB.

    Nullified: replace the `reader is not None` branch in `_np` with
    `if False:` and this test goes red (peak jumps to ~700 KB).
    """
    torch = _shim_torch()
    C = _coreml()
    import numpy as np

    t = torch.randn(100, 200)
    blob_bytes = 100 * 200 * 4  # 80 KB

    arr, peak = _peak_python_heap(lambda: C._np(t))
    assert arr.shape == (100, 200), arr.shape
    assert arr.dtype == np.float32, arr.dtype
    # The byte route: raw bytes + frombuffer. Well under 3x the blob.
    assert peak < 3 * blob_bytes, (
        f"peak Python heap {peak} exceeds 3x blob size {blob_bytes} -- "
        f"_np is probably still going through tolist()"
    )


def test_np_is_bit_identical_to_tolist_for_every_dtype():
    """The numerical claim: the byte route and the tolist route produce the
    same numpy arrays, for every dtype `_NUMPY_DTYPES` maps.

    This is what makes verify()'s claims unaffected.
    """
    torch = _shim_torch()
    C = _coreml()
    import numpy as np

    cases = [
        ("float32", torch.randn(50, 60)),
        ("float64", torch.randn(50, 60).to(torch.float64)),
        ("int32", torch.randint(0, 100, (50, 60)).to(torch.int32)),
        ("int64", torch.randint(0, 100, (50, 60))),
        ("bool", torch.tensor([[True, False, True], [False, True, False]])),
    ]
    for label, t in cases:
        arr_bytes = C._np(t)
        dtype = C._NUMPY_DTYPES[str(t.dtype)]
        arr_tolist = np.array(t.detach().tolist(), dtype=dtype).reshape(
            tuple(t.shape)
        )
        assert np.array_equal(arr_bytes, arr_tolist), (
            f"{label}: byte route differs from tolist route"
        )


def test_np_leaves_non_tensors_unchanged():
    """Non-tensor values pass through `_np` unchanged."""
    C = _coreml()
    assert C._np(42) == 42
    assert C._np(None) is None
    assert C._np([1, 2, 3]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# 3. Tempfile leak: compute_plan cleans up
# ---------------------------------------------------------------------------


def test_compute_plan_cleans_up_its_tempdir():
    """The leak: `compute_plan` called `mkdtemp` and never cleaned up.

    277 directories were found left behind after a round. After the fix,
    the directory is removed in a `finally` block.

    Nullified: remove the `finally: shutil.rmtree(...)` and this test
    goes red (a torchnative-coreml-* directory is found in /tmp).
    """
    torch = _shim_torch()
    import glob
    import tempfile

    try:
        import coremltools  # noqa: F401
    except ImportError:
        print("   (skipped: coremltools not installed)")
        return

    C = _coreml()

    # Count existing torchnative-coreml- dirs
    pattern = os.path.join(tempfile.gettempdir(), "torchnative-coreml-*")
    before = set(glob.glob(pattern))

    # Build a trivial model and get its compute plan
    from torchnative.export.decompose import DecomposedTrace
    from coremltools.converters.mil import Builder as mb

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4))])
    def prog(x):
        return mb.relu(x=x)

    import coremltools as ct
    model = ct.convert(
        prog, convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT32,
    )
    _ = C.compute_plan(model)

    # Check no new dirs were left behind
    after = set(glob.glob(pattern))
    leaked = after - before
    assert not leaked, (
        f"compute_plan leaked {len(leaked)} tempdir(s): {leaked}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
