"""The weights and activation blobs, and the route they take out of a tensor.

A user ran `Qwen/Qwen3-4B-Instruct-2507` on a Windows Intel NPU. `to(device.npu)`
lowered 252 Linears and OpenVINO reported `EXECUTION_DEVICES=['NPU']`. Then
`generate()` died:

    File "torchnative/export/intelnpu.py", line 1152, in _weights_blob
        blob = pack_f16(self.weight.detach().flatten().tolist())
    MemoryError:

    thread '<unnamed>' panicked at pyo3-0.29.2/src/instance.rs:347:60:
    PyObject pointer is null
    pyo3_runtime.PanicException: PyObject pointer is null

Two defects, and this file separates them.

**Defect 1 -- the route.** `tolist()` materialises one `PyFloat` per element:
~24 bytes of object plus an 8-byte list slot. Qwen3's `down_proj` is
9728 x 2560 = 24_903_680 elements, so building a 49 MB f16 blob asked CPython
for roughly **800 MB** of heap first. The same round trip ran on every
activation of every forward, in both directions, which is why the device was
idle between matmuls. `intelnpu.f16_bytes` / `f16_tensor` replace it with
`torch._C._shim_f16_bytes` and `torch.frombuffer`.

**Defect 2 -- the panic.** `tolist` built those scalars with pyo3's
`IntoPyObject` conversion, which for `f64` is
`PyFloat::new -> ffi::PyFloat_FromDouble(val).assume_owned(py)`
(`pyo3-0.29.2/src/types/float.rs:59-64`), and `assume_owned` is "same as
`assume_owned_or_err`, but **panics** on NULL" (`ffi_ptr_ext.rs:17-18`,
panicking at `instance.rs:347`). `PyFloat_FromDouble` returns NULL for exactly
one reason: CPython could not allocate. So an out-of-memory condition crossed
the FFI boundary as a Rust panic where Python code was waiting for a
`MemoryError`. `tensor.rs::py_float` / `py_int` now build those objects through
`Bound::from_owned_ptr_or_err`, which is the fallible sibling pyo3 already has.

**What is asserted here, and what is not.** There is no Intel NPU and no
OpenVINO on this machine, so nothing here runs the real device path. What is
checkable without hardware is (a) that the new route's bytes are *identical* to
the route it replaces, for every dtype `_NPULinear` can receive, and (b) that
the new route does not build Python objects -- asserted as a **peak Python heap
measurement**, not as an absence of `MemoryError`, so it is decisive on a small
machine at a size that runs in a fraction of a second. And the allocation-failure
branch of `py_float` itself is **not** exercised: `setrlimit(RLIMIT_AS)` and
`RLIMIT_DATA` are both refused by this darwin kernel ("current limit exceeds
maximum limit"), so `PyFloat_FromDouble` cannot be made to return NULL here.
That branch is asserted structurally instead, by the route the source takes.

Non-vacuity, each guarantee nullified and observed red:

* restore `_weights_blob` to `pack_f16(...tolist())` and
  `test_f16_bytes_does_not_build_one_python_object_per_element` and
  `test_the_weights_blob_no_longer_travels_through_python_floats` go red;
* make `f16_bytes` fall back to `.tolist()` when `_shim_f16_bytes` is missing and
  `test_f16_bytes_refuses_rather_than_falling_back_to_the_route_that_crashed`
  goes red;
* replace `shim_f16_bytes`'s `flatten_all().contiguous()` with a storage read
  and `test_f16_bytes_reads_the_view_and_not_the_whole_storage` goes red -- but
  **only after that test was widened to float16**. Its first version used
  float32 views alone and stayed green through that nullification, because
  `f16_bytes`'s own `.to(torch.float16)` had already materialised the view. That
  is recorded in the test body, because it is the exact shape of a check that
  cannot fail;
* put `into_py_any` back in `flat_objects` and
  `test_tolist_builds_its_scalars_through_the_fallible_pyo3_spelling` goes red.
"""

import os
import re
import sys
import tracemalloc

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")


def _shim_torch():
    """`import torch` from the vendored tree, past VENDOR.md wall 1.

    `TORCH_USE_RTLD_GLOBAL` before the import, not after: the vendored tree's
    `_load_global_deps()` looks for a `libtorch_global_deps` this build does not
    ship, and upstream's own escape hatch is that variable. Set here rather than
    assumed from the shell, because a suite that only passes in a shell where it
    is already set fails under run.sh.
    """
    os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    import torch

    assert hasattr(torch._C, "_aten_implemented"), (
        "not the torchnative shim -- torch._C has no _aten_implemented"
    )
    return torch


def _intelnpu():
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    from torchnative.export import intelnpu

    return intelnpu


def _code_only(text, comment):
    """`text` with its comments and docstrings removed.

    A source assertion that reads comments is not a source assertion: the
    docstrings below say the word `tolist` on purpose, to record what the code
    stopped doing, and an assertion that tripped over that would be testing the
    prose.
    """
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith(comment)
    )


def _intelnpu_source():
    return open(
        os.path.join(_VENDOR_DIR, "torchnative", "export", "intelnpu.py"), encoding="utf-8"
    ).read()


def _tensor_rs_source():
    return open(
        os.path.join(_ROOT, "rust", "torch_c", "src", "tensor.rs"), encoding="utf-8"
    ).read()


# The dtypes `_NPULinear` can receive: its `__init__` refuses anything that is
# not floating point by name, and `.to(torch.float16)` is what it does to the
# rest. These are the four the shim can hold.
_FLOAT_DTYPES = ("float32", "float64", "float16", "bfloat16")


# ------------------------------------------------------- byte identity


def test_f16_bytes_is_byte_identical_to_pack_f16_of_tolist_for_every_dtype():
    """The new route must produce the *same bytes*, not merely similar ones.

    `pack_f16` is `struct.pack("<{n}e", ...)`; `_shim_f16_bytes` is
    `half::f16::to_bits().to_le_bytes()`. Both are IEEE-754 binary16
    little-endian, and this is the measurement that says so rather than the
    claim. The values straddle what a half encoding can get wrong: a negative,
    a signed zero, a subnormal, an exact power of two, and 65504 -- the largest
    finite half, where a rounding error becomes an infinity.
    """
    torch = _shim_torch()
    intelnpu = _intelnpu()
    values = [1.5, -2.25, 0.0, -0.0, 6.0e-8, 1024.0, 65504.0, -0.125]
    for name in _FLOAT_DTYPES:
        dtype = getattr(torch, name)
        t = torch.tensor(values, dtype=dtype).reshape(2, 4)
        old = intelnpu.pack_f16(t.detach().to(torch.float16).flatten().tolist())
        new = intelnpu.f16_bytes(t)
        assert new == old, (name, new.hex(), old.hex())
        assert len(new) == 2 * t.numel(), (name, len(new))
    print(f"ok   npublob: f16_bytes is byte-identical to pack_f16(tolist()) for {_FLOAT_DTYPES}")


def test_the_weights_blob_no_longer_travels_through_python_floats():
    """`_NPULinear._weights_blob` is where the user's `MemoryError` was raised.

    Both halves -- weight then bias, in that order, because that is the order
    the two `Constant` layers of `linear_ir` read the payload in.
    """
    torch = _shim_torch()
    intelnpu = _intelnpu()
    torch.manual_seed(0)
    layer = torch.nn.Linear(6, 4)
    lowered = intelnpu._NPULinear.from_torch(layer, "NPU", None)

    expected = intelnpu.pack_f16(
        layer.weight.detach().to(torch.float16).flatten().tolist()
    ) + intelnpu.pack_f16(layer.bias.detach().to(torch.float16).flatten().tolist())
    assert lowered._weights_blob() == expected, "the blob changed value, not only route"
    assert len(lowered._weights_blob()) == 2 * (6 * 4 + 4)

    body = _code_only(_intelnpu_source(), "#").split("def _weights_blob(self)", 1)[1]
    body = body.split("\n    def ", 1)[0]
    assert "tolist" not in body, body
    assert "f16_bytes" in body, body
    print("ok   npublob: _weights_blob keeps its bytes and drops its Python floats")


def test_a_bias_free_linear_blobs_only_its_weight():
    """`bias=None` must not append a zero block: `linear_ir` emits no bias layer."""
    torch = _shim_torch()
    intelnpu = _intelnpu()
    layer = torch.nn.Linear(5, 3, bias=False)
    lowered = intelnpu._NPULinear.from_torch(layer, "NPU", None)
    assert lowered.bias is None
    assert len(lowered._weights_blob()) == 2 * 5 * 3
    print("ok   npublob: a bias-free Linear blobs exactly out*in halves")


# ------------------------------------------------------- the route itself


def _peak_python_heap(fn):
    """Peak *Python* heap during `fn()`, in bytes.

    `tracemalloc` traces CPython's own allocator, which is where `PyFloat` and
    list storage come from -- and is *not* where the Rust side's `Vec` comes
    from. That asymmetry is the point: a route that builds no Python objects
    shows a peak of about the size of the one `bytes` it returns, while a route
    that builds `numel` scalars shows a peak an order of magnitude above it.
    """
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def test_f16_bytes_does_not_build_one_python_object_per_element():
    """The measurement that makes this a route claim and not a hope.

    A million elements, so this runs in a fraction of a second and still
    separates the two routes by more than an order of magnitude: the `tolist`
    route has to hold 1_048_576 `PyFloat`s (~24 B each) plus a list of pointers
    (8 B each) -- about 32 MB -- while the byte route's Python-side peak is the
    2 MB `bytes` object it returns. The threshold is written as a multiple of
    the blob, not as a megabyte count, so it does not encode this machine.

    Deliberately **not** an assertion that `MemoryError` is absent: a test that
    only fails on a small machine is not a test.
    """
    torch = _shim_torch()
    intelnpu = _intelnpu()
    n = 1 << 20
    t = torch.arange(n, dtype=torch.float32).reshape(1024, 1024)
    blob_bytes = 2 * n

    new_blob, new_peak = _peak_python_heap(lambda: intelnpu.f16_bytes(t))
    old_blob, old_peak = _peak_python_heap(
        lambda: intelnpu.pack_f16(t.detach().to(torch.float16).flatten().tolist())
    )
    assert new_blob == old_blob, "the two routes disagree about the bytes"

    # The new route's Python-side peak is the blob and a little slack. The old
    # route's is many times it. Both halves are asserted so that a change which
    # made the *old* route cheap would be noticed rather than silently making
    # this test vacuous.
    assert new_peak < 3 * blob_bytes, (new_peak, blob_bytes)
    assert old_peak > 8 * blob_bytes, (old_peak, blob_bytes)
    assert old_peak > 8 * new_peak, (old_peak, new_peak)
    print(
        f"ok   npublob: {n} elements cost {new_peak / blob_bytes:.2f}x the blob on the "
        f"byte route and {old_peak / blob_bytes:.1f}x on the tolist route"
    )


def test_f16_tensor_does_not_build_one_python_object_per_element_either():
    """The way back is the same defect pointing the other way.

    `torch.tensor(unpack_f16(blob))` built `numel` `PyFloat`s on *every*
    forward's return. `torch.frombuffer` reads the bytes.
    """
    torch = _shim_torch()
    intelnpu = _intelnpu()
    n = 1 << 20
    blob = bytes(2 * n)

    new_t, new_peak = _peak_python_heap(lambda: intelnpu.f16_tensor(blob, (1024, 1024)))
    _, old_peak = _peak_python_heap(
        lambda: torch.tensor(intelnpu.unpack_f16(blob), dtype=torch.float16)
    )
    assert tuple(new_t.shape) == (1024, 1024), new_t.shape
    assert new_t.dtype == torch.float16, new_t.dtype
    assert new_peak < 3 * len(blob), (new_peak, len(blob))
    assert old_peak > 8 * new_peak, (old_peak, new_peak)
    print(
        f"ok   npublob: f16_tensor returns {n} elements at {new_peak / len(blob):.2f}x "
        f"the blob where the list route cost {old_peak / len(blob):.1f}x"
    )


def test_the_forward_path_no_longer_names_tolist_or_torch_tensor_of_a_list():
    """`_NPULinear.forward` ran the round trip on every call, both ways.

    Asserted on the source because the real forward needs OpenVINO and an NPU,
    neither of which exists here. It is the weakest test in this file and it is
    here only to stop the two ends drifting back; the byte-identity and heap
    measurements above are what actually hold the route.
    """
    body = _code_only(_intelnpu_source(), "#").split("    def forward(self, x):", 1)[1]
    body = body.split("\n    def ", 1)[0]
    assert "tolist" not in body, body
    assert "unpack_f16" not in body, body
    assert "f16_bytes(x)" in body, body
    assert "f16_tensor(" in body, body
    print("ok   npublob: forward crosses in bytes in both directions")


def test_f16_bytes_reads_the_view_and_not_the_whole_storage():
    """A sliced or transposed tensor must give the elements `tolist` gives.

    This is the trap in the obvious alternative: `untyped_storage()._shim_bytes()`
    hands back the *whole buffer* a view happens to sit inside, so a slice would
    blob its neighbours and a transpose would blob the wrong order -- right byte
    count for the storage, wrong function, no exception anywhere.
    """
    torch = _shim_torch()
    intelnpu = _intelnpu()
    # **float16 is the decisive dtype here and float32 is not.** `f16_bytes`
    # spells `.to(torch.float16)` before the call, and on the shim that is a
    # materialising copy -- so on a float32 view the conversion flattens the
    # view before `_shim_f16_bytes` ever sees it, and the test would pass with
    # the Rust side reading the whole storage. Measured: with `flatten_all() +
    # contiguous()` replaced by a storage read, the float32 cases stayed green
    # and only these float16 ones went red. An already-f16 weight or activation
    # is exactly what `_NPULinear` holds.
    for dtype in (torch.float16, torch.float32):
        base = torch.arange(24, dtype=dtype).reshape(4, 6)
        for label, view in (
            ("column slice", base[:, 2:5]),
            ("row slice", base[1:3]),
            ("transpose", base.t()),
        ):
            where = f"{dtype} {label}"
            expected = intelnpu.pack_f16(view.detach().to(torch.float16).flatten().tolist())
            assert intelnpu.f16_bytes(view) == expected, where
            assert len(intelnpu.f16_bytes(view)) == 2 * view.numel(), where
            assert len(torch._C._shim_f16_bytes(view)) == 2 * view.numel(), where
    # And the trap itself, named: the storage really is bigger than the view.
    half = torch.arange(24, dtype=torch.float16).reshape(4, 6)
    assert len(half[:, 2:5].untyped_storage()._shim_bytes()) > 2 * half[:, 2:5].numel()
    print("ok   npublob: f16_bytes reads the view, where untyped_storage reads the buffer")


def test_f16_bytes_refuses_rather_than_falling_back_to_the_route_that_crashed():
    """No silent `tolist` fallback: that would put the `MemoryError` back.

    A torch without `torch._C._shim_f16_bytes` and without a working
    `Tensor.numpy()` is exactly the shape this module used to run on, and the
    honest answer there is a named refusal rather than the route that died.
    """
    torch = _shim_torch()
    intelnpu = _intelnpu()

    class _NoByteDoor:
        """Stands in for `torch`, with the byte doors shut and `.to` intact."""

        def __init__(self, real):
            self._real = real

            class _C:
                pass

            self._C = _C()

        def __getattr__(self, name):
            return getattr(self._real, name)

    class _NoNumpy:
        def __init__(self, real):
            self._real = real

        def detach(self):
            return self

        def to(self, *a, **k):
            return self

    fake_torch = _NoByteDoor(torch)
    real_torch = intelnpu._torch
    intelnpu._torch = lambda: fake_torch
    try:
        intelnpu.f16_bytes(_NoNumpy(torch.tensor([1.0])))
    except intelnpu.IntelNPUUnsupported as exc:
        text = str(exc)
        assert "_shim_f16_bytes" in text, text
        assert "tolist" in text, text
        assert "MemoryError" in text, text
    else:
        raise AssertionError("f16_bytes fell back instead of refusing")
    finally:
        intelnpu._torch = real_torch
    print("ok   npublob: with no byte door, f16_bytes refuses by name instead of using tolist")


def test_shim_f16_bytes_refuses_a_non_float_tensor_by_name():
    """An int weight reinterpreted as halves loads fine and computes nonsense.

    `_NPULinear.__init__` already refuses an integer weight; this is the second
    wall, at the encoder, for anything that reaches `torch._C` directly.
    """
    torch = _shim_torch()
    try:
        torch._C._shim_f16_bytes(torch.tensor([1, 2, 3]))
    except NotImplementedError as exc:
        text = str(exc)
        assert "_shim_f16_bytes" in text, text
        assert "int64" in text, text
    else:
        raise AssertionError("_shim_f16_bytes reinterpreted an integer tensor")
    try:
        torch._C._shim_f16_bytes("not a tensor")
    except NotImplementedError as exc:
        assert "expected a tensor" in str(exc), exc
    else:
        raise AssertionError("_shim_f16_bytes accepted a non-tensor")
    print("ok   npublob: _shim_f16_bytes refuses an integer tensor and a non-tensor by name")


def test_f16_tensor_refuses_an_odd_byte_count_and_a_shape_it_cannot_fill():
    """The refusals `unpack_f16` carried must survive the route that skips it."""
    _shim_torch()
    intelnpu = _intelnpu()
    for blob, shape, needle in (
        (b"\x00\x00\x00", (1,), "not a whole number of 2-byte halves"),
        (b"\x00\x00" * 6, (4, 4), "expected 16 output elements"),
    ):
        try:
            intelnpu.f16_tensor(blob, shape)
        except intelnpu.IntelNPUExecutionError as exc:
            assert needle in str(exc), (needle, str(exc))
        else:
            raise AssertionError(f"f16_tensor accepted {blob!r} for {shape}")
    print("ok   npublob: f16_tensor keeps the odd-byte and wrong-count refusals")


# ------------------------------------------------- defect 2, the panic


def test_tolist_builds_its_scalars_through_the_fallible_pyo3_spelling():
    """The panic is pyo3's, and it is removable *here* but not in the trait.

    `f64: IntoPyObject` goes `PyFloat::new -> PyFloat_FromDouble(...).assume_owned(py)`
    and `assume_owned` panics on NULL. pyo3 0.29 offers no fallible `PyFloat::new`,
    so the fix is to stop using the conversion trait in the one function that
    builds millions of objects and call `Bound::from_owned_ptr_or_err` directly.

    Asserted on the source, and said plainly: the NULL branch itself cannot be
    reached on this machine. `resource.setrlimit` refuses both `RLIMIT_AS` and
    `RLIMIT_DATA` here with "current limit exceeds maximum limit", so
    `PyFloat_FromDouble` cannot be made to fail. What *is* checkable is that the
    panicking spelling is gone from the two arms that build per-element objects,
    and that `tolist` still answers the same numbers.
    """
    source = _code_only(_tensor_rs_source(), "//")
    body = source.split("fn flat_objects(", 1)[1].split("\nfn py_float(", 1)[0]
    # The bool arm is left alone on purpose: `True`/`False` are immortal
    # singletons, so that conversion allocates nothing and cannot return NULL.
    scalar_arms = body.split("if dtype.is_float()", 1)[1]
    assert "into_py_any" not in scalar_arms, scalar_arms
    assert "py_float(py, v)" in scalar_arms, scalar_arms
    assert "py_int(py, v)" in scalar_arms, scalar_arms
    for fn in ("fn py_float(", "fn py_int("):
        block = source.split(fn, 1)[1].split("\n}\n", 1)[0]
        assert "from_owned_ptr_or_err" in block, (fn, block)
        assert "assume_owned" not in block, (fn, block)
    print("ok   npublob: tolist's scalars are built with a NULL-checked allocation")


def test_tolist_still_answers_the_same_numbers_it_did():
    """A regression guard on the allocation change: values and nesting unchanged."""
    torch = _shim_torch()
    f = torch.tensor([[1.5, -2.25], [0.0, 3.75]], dtype=torch.float32)
    assert f.tolist() == [[1.5, -2.25], [0.0, 3.75]], f.tolist()
    i = torch.tensor([[1, -2], [0, 3]], dtype=torch.int64)
    assert i.tolist() == [[1, -2], [0, 3]], i.tolist()
    assert [type(v) for v in f.flatten().tolist()] == [float] * 4
    assert [type(v) for v in i.flatten().tolist()] == [int] * 4
    b = torch.tensor([True, False])
    assert b.tolist() == [True, False] and [type(v) for v in b.tolist()] == [bool] * 2
    s = torch.tensor(2.5)
    assert s.tolist() == 2.5 and isinstance(s.tolist(), float)
    print("ok   npublob: tolist's values, types and nesting are unchanged")


def test_the_remaining_panic_point_is_named_in_the_docs_rather_than_left_silent():
    """`nest`'s `PyList::new` is still pyo3's panicking spelling, and is recorded.

    `PyList::new` reaches `ffi::PyList_New(len).assume_owned(py)`
    (`pyo3-0.29.2/src/types/list.rs:98`), so a failed *list* allocation still
    panics. That one is not worth hand-rolling -- it is one allocation per
    dimension slice against `numel` per element -- but an unrecorded trap is
    worse than a recorded one.
    """
    doc = open(os.path.join(_ROOT, "docs", "devices", "INTELNPU.md"), encoding="utf-8").read()
    assert "PyList::new" in doc, "the surviving panic point is not recorded"
    assert "assume_owned" in doc, doc[:0]
    assert re.search(r"instance\.rs:347", doc), "the panic site is not named"
    print("ok   npublob: the panic point that survives is named in INTELNPU.md")


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
