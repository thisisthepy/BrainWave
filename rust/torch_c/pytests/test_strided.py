"""docs/kernels/STRIDED.md -- `as_strided`, and the barrier that makes a gather honest.

`tools/golden/cases.py` compares every value this op answers against upstream
element-wise, including the five refusal messages, and this file deliberately
does not repeat that. What is here is the part a value comparison structurally
cannot hold down: **the aliasing that upstream has and this shim does not.**

`as_strided` upstream is a two-way view. Writes through the result reach the
base and writes to the base show through the result. candle 0.11.0 cannot build
one -- `Layout::new(shape, stride, offset)` is public and
`Tensor::from_storage` is public, but `from_storage` takes an *owned* `Storage`
and allocates a fresh `Arc`, and the nine sites inside candle that do build a
shallow view write `Tensor_ { storage: self.storage.clone(), layout }` directly
against a struct with no public field. Both halves of the constructor are
public and the struct that joins them is not. That was `docs/kernels/TAIL3.md` §6's
refusal and `docs/kernels/TAIL4.md` §1's, and neither is overturned here.

What is new is that the gather's wrongness is made **loud rather than silent**.
`storage.rs::StridedBarrier` bars in-place writes to the result's storage and
to the base's for as long as the result is alive, and the bar is read at
`tensor::write_into`, which is the single write door. So every case where
upstream would have propagated a write and this copy would not is a named
refusal. That is a narrowing of upstream, and this file's job is to prove it is
a narrowing rather than a widening -- in both directions, against upstream
measured in a separate process, and including the two aliasing shapes
`docs/kernels/TAIL4.md` §1.2 named as the ones that defeat every cheaper marker:

  * `z = y.view(-1)` -- a shallow view of the *result*, which gets a fresh
    candle `TensorId` and would escape a `TensorId`-keyed guard;
  * `v = x.view(...)` taken **before** `x.as_strided(...)` was ever called --
    an alias of the *base* that predates the mark, which no field propagated
    forward through the view ops could ever reach.

Both are refused, because the key is the storage and not the wrapper.

Nothing here needs numpy or a network.
"""

import json
import os
import re
import subprocess
import sys

from test_shim import _C

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")
_SRC = os.path.join(_REPO_ROOT, "rust", "torch_c", "src")
_ATEN_RS = os.path.join(_SRC, "aten.rs")
_TENSOR_RS = os.path.join(_SRC, "tensor.rs")
_STORAGE_RS = os.path.join(_SRC, "storage.rs")


# --------------------------------------------------------------------------
# The probe. Run once per side, in its own process, with PYTHONPATH stripped
# for the upstream side so that every upstream number below is measured rather
# than transcribed.
# --------------------------------------------------------------------------

_PROBE_SCRIPT = r"""
import json
import torch

out = {"_marker": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}


def rec(name, fn):
    try:
        r = fn()
    except Exception as e:
        out[name] = {"raised": type(e).__name__, "msg": str(e)}
        return
    if isinstance(r, torch.Tensor):
        out[name] = {"ok": [float(v) for v in r.reshape(-1).double()],
                     "shape": list(r.shape), "dtype": str(r.dtype)}
    else:
        out[name] = {"ok": r}


FLAT = [float(i) for i in range(12)]


def base():
    return torch.tensor(FLAT)


# -- values. The overlapping case is longformer's `_chunk` shape. -----------
rec("plain", lambda: base().as_strided((2, 3), (3, 1)))
rec("overlapping", lambda: base().as_strided((3, 3), (2, 1)))
rec("zero_stride", lambda: base().as_strided((3, 3), (0, 1)))
rec("explicit_offset", lambda: base().as_strided((2,), (1,), 4))


# -- the two-way view, one direction each ----------------------------------
def write_through_view():
    x = base()
    y = x.as_strided((2, 3), (3, 1))
    y.fill_(7.0)
    return x


def write_to_base():
    x = base()
    y = x.as_strided((2, 3), (3, 1))
    x.fill_(7.0)
    return y


rec("write_through_view_then_read_base", write_through_view)
rec("write_to_base_then_read_view", write_to_base)


# -- the alias that gets a fresh TensorId ----------------------------------
def write_through_alias_of_view():
    x = base()
    y = x.as_strided((2, 3), (3, 1))
    z = y.view(-1)
    z.fill_(7.0)
    return x


rec("write_through_alias_of_view_then_read_base", write_through_alias_of_view)


# -- the alias of the base that PREDATES the as_strided call ---------------
def write_through_older_alias_of_base():
    x = base()
    v = x.view(3, 4)          # taken BEFORE as_strided is ever called
    y = x.as_strided((2, 3), (3, 1))
    v.fill_(7.0)
    return y


rec("write_through_older_alias_of_base_then_read_view",
    write_through_older_alias_of_base)


# -- a non-contiguous receiver reads the STORAGE, not the logical order ----
# `x.reshape(3,4).t()` is `[[0,4,8],[1,5,9],...]` logically and `0,1,2,...` in
# storage. Upstream reads the storage. A gather over the logical order would
# answer [[0,4],[4,8]] where upstream answers [[0,1],[1,2]] -- same shape, same
# dtype, different numbers.
rec("noncontiguous_receiver",
    lambda: base().reshape(3, 4).t().as_strided((2, 2), (1, 1)))


# -- the barrier must not be permanent -------------------------------------
def base_is_writable_again_after_the_view_dies():
    x = base()
    y = x.as_strided((2, 3), (3, 1))
    del y
    x.fill_(7.0)
    return x


rec("base_after_view_dropped", base_is_writable_again_after_the_view_dies)


# -- an unrelated tensor is never barred -----------------------------------
def unrelated_tensor_still_writable():
    x = base()
    y = x.as_strided((2, 3), (3, 1))
    other = torch.zeros(4)
    other.fill_(1.0)
    return other + y.reshape(-1)[:4] * 0.0


rec("unrelated_while_barred", unrelated_tensor_still_writable)


# -- does the alias really alias? the barrier test is vacuous otherwise -----
def alias_of_view_shares_storage():
    x = base()
    y = x.as_strided((2, 3), (3, 1))
    z = y.view(-1)
    return z.data_ptr() == y.data_ptr()


rec("alias_of_view_shares_storage", alias_of_view_shares_storage)

print(json.dumps(out))
"""

_cache = {}


def _run(side):
    """`side` is `"shim"` or `"upstream"`; `None` when the vendored shim is not
    installed, which is the same silent skip `test_tail4.py` takes."""
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
        [sys.executable, "-c", _PROBE_SCRIPT],
        capture_output=True, text=True, env=env, cwd=_REPO_ROOT,
    )
    assert proc.returncode == 0, (
        f"the {side} probe failed to run at all:\n{proc.stderr[-3000:]}"
    )
    data = json.loads(proc.stdout)
    assert data["_marker"] == side, (
        f"the {side} probe imported the wrong torch ({data['_marker']}) -- every "
        "assertion below would have been about the wrong library"
    )
    _cache[side] = data
    return data


def _both(name):
    s = _run("shim")
    if s is None:
        return "skip"
    return s[name], _run("upstream")[name]


def _source(path):
    if not os.path.isfile(path):
        return None
    return open(path, encoding="utf-8").read()


# --------------------------------------------------------------------------
# 1. The op exists, and its values are upstream's
# --------------------------------------------------------------------------


def test_as_strided_is_implemented_and_its_values_are_upstreams():
    """Including the overlapping stride, which is the whole reason it was wanted.

    `longformer`'s `_chunk` halves the second stride so that consecutive chunks
    share half their elements, and an implementation that quietly refused
    overlap -- or that produced disjoint chunks -- would answer a plausible
    shape with the wrong attention windows. So the overlap case is asserted
    element-wise against upstream, not just for shape.
    """
    assert "aten.as_strided.default" in _C._aten_implemented()
    for name in ("plain", "overlapping", "zero_stride", "explicit_offset"):
        pair = _both(name)
        if pair == "skip":
            return
        got, want = pair
        assert "raised" not in want, (name, want)
        assert "raised" not in got, (name, got)
        assert got["ok"] == want["ok"], (name, got, want)
        assert got["shape"] == want["shape"], (name, got, want)
    # The overlap is real and not an accident of the numbers: element [1][0] of
    # the overlapping view is element [0][2] of it, which is what "sliding
    # window" means.
    over = _run("upstream")["overlapping"]["ok"]
    assert over[3] == over[2], over
    assert over[6] == over[5], over


def test_the_default_storage_offset_is_the_receivers_own_and_not_zero():
    """`x[4:].as_strided((2,), (1,))` is `[4., 5.]` upstream, not `[0., 1.]`.

    Reading the default as zero is the plausible mistake and it is invisible on
    any receiver that starts at the front of its buffer, which is every
    receiver a test writes by hand. Asserted through the explicit-offset case
    and against upstream's own answer for the same numbers.
    """
    pair = _both("explicit_offset")
    if pair == "skip":
        return
    got, want = pair
    assert want["ok"] == [4.0, 5.0], want
    assert got["ok"] == want["ok"], (got, want)


def test_a_noncontiguous_receiver_is_refused_because_upstream_reads_storage_order():
    """The check that no shape or dtype comparison could have found.

    Upstream's `as_strided` addresses the **storage** and ignores the
    receiver's layout entirely. This shim gathers out of the receiver's
    elements, which are in storage order only when the receiver is contiguous.
    On a transposed receiver the two orders differ, so gathering would answer
    upstream's shape and upstream's dtype with elements read from the wrong
    places -- the exact failure this repository's refusals exist to prevent.

    The assertion is in two halves so that it cannot pass vacuously: upstream
    must *succeed* here (otherwise there is nothing being narrowed), and it
    must succeed with the **storage-order** answer (otherwise the refusal is
    unnecessary).
    """
    pair = _both("noncontiguous_receiver")
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, f"upstream refused, so there is no narrowing: {want}"
    # storage order: [[0,1],[1,2]]. Logical order would be [[0,4],[4,8]].
    assert want["ok"] == [0.0, 1.0, 1.0, 2.0], want
    assert got["raised"] == "NotImplementedError", got
    assert "aten.as_strided.default" in got["msg"], got["msg"]
    assert "contiguous" in got["msg"], got["msg"]


# --------------------------------------------------------------------------
# 2. The barrier: upstream's two-way view, narrowed to read-only
# --------------------------------------------------------------------------


def test_upstream_propagates_a_write_THROUGH_the_view_and_this_shim_refuses_it():
    """Direction one, and the one a gather gets wrong most obviously.

    Upstream: `y = x.as_strided(...); y.fill_(7.)` leaves `x` beginning
    `[7, 7, 7, 7, 7, 7, 6, ...]`. A gather leaves `x` untouched and answers
    nothing at all about it, which is a wrong answer with no symptom.

    Here it raises, and the message names `as_strided`. Note what is asserted
    about upstream: that it *did* propagate. If upstream ever stopped, this
    refusal would be gratuitous and the test says so instead of passing.
    """
    pair = _both("write_through_view_then_read_base")
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, want
    assert want["ok"][:6] == [7.0] * 6, want
    assert want["ok"][6] == 6.0, want
    assert got["raised"] == "RuntimeError", got
    assert "as_strided" in got["msg"], got["msg"]


def test_upstream_propagates_a_write_TO_THE_BASE_and_this_shim_refuses_that_too():
    """Direction two -- the one `docs/kernels/TAIL4.md` §1.2 said no write guard could
    address at all.

    Its words: *"There is a third fact that no guard on writes can address:
    upstream's `as_strided` is a view in both directions... Poisoning the base
    as well would close it, and it needs the same unreachable marker."*

    The marker turned out to be reachable, and the reason it is worth writing
    down is that it is not a per-tensor field: it is the **storage address**,
    with a keep-alive holding it reserved. A field on `PyTensorBase` genuinely
    could not have closed this direction -- see the next test.
    """
    pair = _both("write_to_base_then_read_view")
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, want
    assert want["ok"] == [7.0] * 6, want
    assert got["raised"] == "RuntimeError", got
    assert "as_strided" in got["msg"], got["msg"]


def test_the_barrier_holds_through_an_alias_that_gets_a_fresh_candle_TensorId():
    """`z = y.view(-1); z.fill_(0)` -- the escape that defeats a `TensorId` set.

    candle's `reshape` builds a new `Tensor_` with `id: TensorId::new()` over
    `storage: self.storage.clone()`, so `z` is a different tensor identity on
    the same buffer. `docs/kernels/TAIL4.md` §1.2 named this as the reason a
    `HashSet<TensorId>` populated at `as_strided` time cannot work: it would be
    defeated by the most ordinary thing a caller does next.

    Keyed on the storage it is not defeated. The test first checks that `z`
    really does alias `y` -- if the shim's `view` had copied, the refusal would
    be about a tensor that shares nothing and this test would be checking
    nothing.
    """
    shim = _run("shim")
    if shim is None:
        return
    assert shim["alias_of_view_shares_storage"]["ok"] is True, (
        "y.view(-1) no longer aliases y in this shim, so the barrier is not "
        "being asked the question this test exists to ask"
    )
    pair = _both("write_through_alias_of_view_then_read_base")
    got, want = pair
    assert "raised" not in want, want
    assert want["ok"][:6] == [7.0] * 6, want
    assert got["raised"] == "RuntimeError", got
    assert "as_strided" in got["msg"], got["msg"]


def test_the_barrier_reaches_an_alias_of_the_base_that_PREDATES_the_call():
    """The one that decides between a field and a storage key, and it is why
    `docs/kernels/TAIL4.md` §1.3's sizing was not sufficient.

    That sizing was *"one `bool`/`Option<...>` field on `PyTensorBase` that
    survives aliasing, plus propagation through the view-producing ops"*.
    Propagation runs **forward** from the moment the mark is attached. The
    exposure runs **backward**:

    ```python
    x = torch.tensor(...)
    v = x.view(3, 4)              # alias of the base, taken FIRST
    y = x.as_strided((2, 3), (3, 1))
    v.fill_(7.0)                  # upstream: y sees 7s. A gather: y sees nothing.
    ```

    `v` existed before `as_strided` was called. No field attached to `y`, and
    no propagation forward from `y`, can ever reach it. A key on the storage
    reaches it because `v`, `x` and the base of `y` are one buffer.

    And this is not a corner: `longformer`'s `_chunk` calls `as_strided` on a
    `hidden_states` that is *itself* the result of a `.view(...)` one line
    earlier, so the base is already an alias before the op is reached.
    """
    pair = _both("write_through_older_alias_of_base_then_read_view")
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, want
    assert want["ok"] == [7.0] * 6, want
    assert got["raised"] == "RuntimeError", got
    assert "as_strided" in got["msg"], got["msg"]


def test_the_barrier_lifts_when_the_view_dies_which_is_what_keeps_it_from_poisoning():
    """A permanent poison set keyed on an address is the structure
    `docs/kernels/TAIL4.md` §1.2 rejected, and it rejected it correctly.

    Its objection: an address is reused after the allocation it named is freed,
    so a permanent entry starts refusing writes to unrelated tensors allocated
    later at the same address. What removes the objection is
    `StridedBarrier`'s keep-alive: it holds a clone of both candle tensors, so
    neither storage can be freed while its key is registered, and `Drop`
    removes the keys *before* releasing the clones. The entry and the address
    reservation end together, so a reused address can never inherit an entry.

    Two halves, and both are needed. If only the first held, the barrier would
    be a leak; if only the second, it would be permanent.
    """
    pair = _both("base_after_view_dropped")
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, want
    assert "raised" not in got, (
        "the base is still barred after the as_strided result was dropped -- "
        f"the barrier is permanent, which is the failure mode docs/kernels/TAIL4.md "
        f"§1.2 named: {got}"
    )
    assert got["ok"] == want["ok"] == [7.0] * 12, (got, want)

    unrelated = _both("unrelated_while_barred")
    ug, uw = unrelated
    assert "raised" not in ug, (
        f"an unrelated tensor was refused while a barrier was live: {ug}"
    )
    assert ug["ok"] == uw["ok"], (ug, uw)


# --------------------------------------------------------------------------
# 3. The guard is where it says it is, and it is the only door
# --------------------------------------------------------------------------


def test_the_write_door_is_still_single_which_is_what_makes_the_barrier_total():
    """Inherited from `docs/kernels/TAIL4.md` §1.1, and now load-bearing rather than a
    precondition for a sizing.

    The barrier is read in exactly one place. That is sufficient **only** while
    `write_into` is the only thing that writes through a tensor's layout and
    `write_back` is its only caller. Counted rather than read:

    ```text
    aten.rs   .write_into(   1 occurrence   (inside fn write_back)
    tensor.rs .write_into(   0 occurrences
    ```

    The day either number moves, an in-place op can reach a barred storage
    without passing the barrier, and this fails at that moment rather than at
    the moment somebody notices a wrong answer.
    """
    aten = _source(_ATEN_RS)
    tensor = _source(_TENSOR_RS)
    if aten is None or tensor is None:
        print("   (skipped: the crate sources are not beside this file)")
        return
    calls = re.findall(r"\.write_into\s*\(", aten)
    assert len(calls) == 1, (
        f"aten.rs now calls write_into {len(calls)} times. The as_strided "
        "barrier is read inside write_into and was sized against there being "
        "exactly one caller -- docs/kernels/STRIDED.md §2."
    )
    assert "fn write_back(" in aten
    assert "pub fn write_into(" in tensor
    assert len(re.findall(r"\.write_into\s*\(", tensor)) == 0


def test_the_barrier_is_read_at_that_door_and_nullifying_it_is_what_STRIDED_md_measured():
    """The demonstration `docs/kernels/COMPLEX2.md` set as the standard for calling a
    guard real, pinned as source structure.

    `docs/kernels/COMPLEX2.md` did not argue that `tensor()`'s non-`Dense` refusal
    mattered; it **deleted the refusal and measured what computed anyway** --
    6 of 10 sampled ops answered silently. `docs/kernels/STRIDED.md` §5 does the same
    thing for this barrier: with the `write_is_barred` block commented out of
    `write_into` and nothing else changed, the crate builds and
    `x.as_strided(...).fill_(7.)` returns a filled view while leaving `x` at
    `[0., 1., 2., ...]` -- upstream's `x` is all 7s, and nothing raises. The
    wrong thing that becomes representable is a **stale base**, and its name in
    the caller that wanted this op is `hidden_states`.

    It is also *caught*, which is the other half of the check: that build
    fails 2 golden cases and 4 of the tests in this file. A guard that can be
    nullified without anything going red is a guard nobody is exercising.

    A test cannot re-run that (it needs a rebuild), so what it pins is the two
    halves that a deletion would have to get past: the call is present at the
    door, and it is the *only* reader, so there is no second copy to keep the
    behaviour alive while the real one is removed.
    """
    tensor = _source(_TENSOR_RS)
    storage = _source(_STORAGE_RS)
    if tensor is None or storage is None:
        print("   (skipped: the crate sources are not beside this file)")
        return
    door = tensor[tensor.index("pub fn write_into("):]
    assert "crate::storage::write_is_barred(&dest)" in door, (
        "the as_strided write barrier is no longer read at the write door. If "
        "it moved, move this assertion; if it was deleted, docs/kernels/STRIDED.md §5 "
        "says what that puts back."
    )
    readers = re.findall(r"write_is_barred\s*\(", tensor)
    assert len(readers) == 1, (tensor.count("write_is_barred"), readers)
    # And the registry it reads is keyed on the storage with a keep-alive.
    # Without `keep_alive` the address is free to be reused while the entry
    # lives, which is precisely docs/kernels/TAIL4.md §1.2's objection.
    assert "pub struct StridedBarrier" in storage
    assert "keep_alive" in storage
    assert "impl Drop for StridedBarrier" in storage


def test_as_strided_and_unfold_are_the_two_ops_that_take_the_barrier():
    """A closed producer list, so "which ops can bar a storage" is answerable by
    reading two lines rather than by trusting a convention.

    **This test was `test_as_strided_is_the_only_op_that_takes_the_barrier` and
    asserted `len(calls) == 1`.** Its own docstring said what to do when that
    stopped being true: *"If a second op ever needs it, that is a decision worth
    making explicitly -- an op that bars its input's storage is an op that can
    make an unrelated later write fail."* `aten.unfold.default` is that second
    op (docs/kernels/LAST7.md §3): upstream's `Tensor.unfold` is a two-way view for the
    same reason `as_strided` is, candle cannot build it for the same reason, and
    a gather without the barrier would lose a write in both directions
    silently. So the test is inverted rather than deleted, and it still names
    every producer -- a third one fails here until somebody writes it down.
    """
    aten = _source(_ATEN_RS)
    tensor = _source(_TENSOR_RS)
    if aten is None or tensor is None:
        print("   (skipped: the crate sources are not beside this file)")
        return
    calls = re.findall(r"bar_writes_as_strided_view\s*\(", aten)
    assert len(calls) == 2, calls
    assert "fn as_strided_default(" in aten
    assert "fn unfold_default(" in aten
    assert "pub fn bar_writes_as_strided_view(" in tensor
    # The handle is carried by `Clone`, not dropped -- a clone of the wrapper
    # points at the same storage, so dropping it would launder a barred tensor
    # into a writable one.
    clone = tensor[tensor.index("impl Clone for PyTensorBase"):]
    clone = clone[:clone.index("\n}\n")]
    assert "strided: self.strided.clone()" in clone, clone


# --------------------------------------------------------------------------
# 4. The narrowing is declared where a reader will meet it
# --------------------------------------------------------------------------


def test_the_read_only_narrowing_is_declared_in_the_golden_harness():
    """Not only in prose.

    `aten.slice.Tensor` (step > 1) and `aten.view.dtype` are already in
    `tools/golden/cases.py` as `expect="diverge"`: they alias upstream,
    materialise here, and a write through them is *silently* lost.
    `as_strided` is in the same file as `expect="c_error"`, which is the
    stronger register -- the divergence is a refusal rather than a number --
    and `compare.py` reports it every run, so the narrowing cannot rot into a
    document nobody opens.
    """
    path = os.path.join(_REPO_ROOT, "tools", "golden", "cases.py")
    if not os.path.isfile(path):
        print("   (skipped: cases.py is not in this tree)")
        return
    text = open(path, encoding="utf-8").read()
    assert "def as_strided_cases(" in text
    assert '"aten.as_strided.default": as_strided_cases,' in text
    block = text[text.index("def as_strided_cases("):]
    block = block[:block.index("\n# --- aten.expand.default")]
    # Past the builder's own docstring, which quotes both spellings.
    body = block[block.index('"""', block.index('"""') + 3) + 3:]
    assert body.count('expect="c_error"') == 2, (
        "both directions of the narrowing must be registered separately, so "
        "that closing one does not hide the other -- found "
        f"{body.count(chr(34) + chr(34))}"
    )
    # The five upstream refusals share one `expect="both_error"` inside a loop
    # over five (size, stride, offset) tuples, so the message count is what is
    # asserted rather than the literal.
    assert body.count('expect="both_error"') == 1, body.count('expect="both_error"')
    for message in ["mismatch in length of strides and shape",
                    "Negative strides are not supported",
                    "Storage size calculation overflowed",
                    "invalid storage offset",
                    "out of bounds for storage"]:
        assert message in body, message


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
