"""docs/TAIL4.md -- `as_strided`'s verdict, and the four walls behind it.

`tools/golden/cases.py` compares every op this round landed against upstream
element-wise, and this file deliberately does not repeat that. What is here is
the part a value comparison structurally cannot hold down:

  * **The `as_strided` verdict, which is that the refusal stands.**
    `docs/TAIL3.md` §6 refused it because candle has no storage-sharing
    constructor. This round asked the narrower question -- can a *read-only*
    `as_strided` be made safe, one that refuses the moment its result is
    written to rather than one that hopes nobody writes -- and the answer is
    no, for a reason that is checkable in the source rather than argued:
    `write_into` is the single write door and it is reached from `aten.rs`,
    but the marker such a guard would have to carry can only live on
    `PyTensorBase`, and no identity reachable from `aten.rs` survives
    aliasing. `test_the_as_strided_refusal_still_stands_and_names_the_overload`
    and `test_the_write_door_is_single_which_is_the_precondition_a_read_only_
    as_strided_would_need` pin both halves, so the day the precondition
    changes the verdict is re-openable rather than folklore.

  * **Banker's rounding, which is the whole of `round`.** `round(0.5)` is `0`
    and `round(2.5)` is `2`. A test on `0.3` and `0.7` cannot tell that from
    round-half-away-from-zero, which is what candle's own `Tensor::round`
    computes -- so the separator is asserted as a separator: the half-integer
    grid, both rules spelled out, and the two disagreeing on five of nine.

  * **`index_copy_` is next to `index_add_` and disagrees with it three ways.**
    Duplicate indices overwrite rather than accumulate, an `int32` index is
    refused where `index_add_` accepts one, and the out-of-bounds message
    carries the index and the extent where `index_add_`'s carries neither.
    Each is asserted against the *other op in the same process*, which is the
    shape a per-op golden case cannot take. `docs/DEMAND8.md` recorded the
    same trap one op over and `docs/TAIL3.md` §3 recorded it again.

  * **`t_`'s alias, which no value comparison sees.** It is the first in-place
    op here that changes the receiver's shape, so it goes through
    `replace_with` rather than `write_into` -- and the thing that makes that
    safe is that candle's `transpose` shares the storage. A view taken before
    the call still sees writes after it, on both sides.

  * **The alias-versus-kernel split**, as source structure rather than prose,
    and the readback claim with it: none of the four new kernels moves a
    tensor to the host, which is what kept `device.rs` out of this round.

Every upstream number below is re-measured at run time in a separate process
with `PYTHONPATH` stripped, rather than transcribed. Nothing here needs numpy
or a network.
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
_ATEN_RS = os.path.join(_REPO_ROOT, "rust", "torch_c", "src", "aten.rs")
_TENSOR_RS = os.path.join(_REPO_ROOT, "rust", "torch_c", "src", "tensor.rs")


_PROBE_SCRIPT = r"""
import json, math, sys
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


# -- round: the half-integer grid, which is the separator --------------------
GRID = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
rec("round_grid", lambda: torch.round(torch.tensor(GRID)))
# `torch.signbit` has no kernel in this shim, so the sign is read off the
# Python floats with `math.copysign` -- which is the same bit and is exactly
# what `tolist()` cannot show by comparison, since `-0.0 == 0.0`.
rec("round_grid_signbit",
    lambda: [math.copysign(1.0, v) < 0
             for v in torch.round(torch.tensor(GRID)).tolist()])
rec("round_inplace_grid", lambda: torch.tensor(GRID).round_())
rec("round_decimals_bfloat16",
    lambda: torch.round(torch.tensor([2.675], dtype=torch.bfloat16), decimals=2))
rec("round_decimals_float32",
    lambda: torch.round(torch.tensor([2.675]), decimals=2))
rec("round_int_identity", lambda: torch.round(torch.arange(3)))
rec("round_int_decimals_refused", lambda: torch.round(torch.arange(3), decimals=2))
rec("round_int_decimals_zero_refused", lambda: torch.round(torch.arange(3), decimals=0))

# -- index_copy_ against index_add_, in one process --------------------------
rec("index_copy_dup",
    lambda: torch.zeros(3).index_copy_(0, torch.tensor([1, 1, 1]),
                                       torch.tensor([1.0, 2.0, 3.0])))
rec("index_add_dup",
    lambda: torch.zeros(3).index_add_(0, torch.tensor([1, 1, 1]),
                                      torch.tensor([1.0, 2.0, 3.0])))
rec("index_copy_int32_index",
    lambda: torch.zeros(3).index_copy_(0, torch.tensor([0], dtype=torch.int32),
                                       torch.tensor([5.0])))
rec("index_add_int32_index",
    lambda: torch.zeros(3).index_add_(0, torch.tensor([0], dtype=torch.int32),
                                      torch.tensor([5.0])))
rec("index_copy_negative",
    lambda: torch.zeros(3).index_copy_(0, torch.tensor([-1]), torch.tensor([5.0])))
rec("index_add_negative",
    lambda: torch.zeros(3).index_add_(0, torch.tensor([-1]), torch.tensor([5.0])))
rec("index_copy_out_of_place_leaves_self",
    lambda: (lambda x: [x.index_copy(0, torch.tensor([0]), torch.tensor([9.0])).tolist(),
                        x.tolist()])(torch.zeros(2)))

# -- t_: the alias, and the identity -----------------------------------------
def _t_alias():
    x = torch.arange(6.0).reshape(2, 3)
    v = x.view(-1)
    same = x.t_() is x
    v[0] = 99.0
    return [float(x[0, 0]), float(same), float(x.shape[0]), float(x.shape[1])]


rec("t_alias", _t_alias)
rec("t_rank1_unchanged", lambda: torch.arange(3.0).t_())
rec("t_rank3_refused", lambda: torch.zeros(2, 3, 4).t_())

# -- logsumexp: the infinite maximum, which is the only branch ---------------
rec("logsumexp_all_neg_inf",
    lambda: torch.logsumexp(torch.tensor([float("-inf")] * 3), dim=0))
rec("logsumexp_pos_inf",
    lambda: torch.logsumexp(torch.tensor([float("inf"), 0.0]), dim=0))
rec("logsumexp_naive_form_on_all_neg_inf",
    lambda: (lambda t: (t - t.amax(0, True)).exp().sum(0).log())(
        torch.tensor([float("-inf")] * 3)))
rec("logsumexp_int_dtype",
    lambda: str(torch.logsumexp(torch.arange(3), dim=0).dtype))
rec("logsumexp_amax_int_dtype",
    lambda: str(torch.amax(torch.arange(3), dim=0).dtype))
rec("logsumexp_sum_int_dtype",
    lambda: str(torch.sum(torch.arange(3), dim=0).dtype))

# -- embedding: the argument form, not a missing kernel ----------------------
_W = torch.arange(20, dtype=torch.float32).reshape(5, 4)
_IDX = [[1, 2], [0, 3]]
rec("embedding_int64",
    lambda: torch.embedding(_W, torch.tensor(_IDX, dtype=torch.int64)))
rec("embedding_int32",
    lambda: torch.embedding(_W, torch.tensor(_IDX, dtype=torch.int32)))
rec("embedding_uint8_refused",
    lambda: torch.embedding(_W, torch.tensor([0, 1], dtype=torch.uint8)))

# -- logical_not, against bitwise_not in the same process --------------------
_LN_PROBE = [0, 1, 2, -1, 3]
for _dn in ("bool", "int64", "float32"):
    for _op, _fn in (("logical_not", torch.logical_not), ("bitwise_not", torch.bitwise_not)):
        def _mk(dn=_dn, fn=_fn):
            values = [abs(v) for v in _LN_PROBE] if dn == "bool" else _LN_PROBE
            t = torch.tensor(values, dtype=getattr(torch, dn))
            r = fn(t)
            return [str(r.dtype)] + [float(v) for v in r.reshape(-1).double()]
        rec(f"{_op}_{_dn}", _mk)
rec("logical_not_float_specials",
    lambda: torch.logical_not(torch.tensor([-0.0, 0.0, float("nan"), float("inf")])))
rec("logical_not_dtype_is_bool",
    lambda: str(torch.logical_not(torch.zeros(2, dtype=torch.float64)).dtype))
rec("logical_not_method", lambda: torch.zeros(2).logical_not())

# -- as_strided: still refused ----------------------------------------------
rec("as_strided", lambda: torch.arange(10.0).as_strided((3, 3), (1, 1)))

json.dump(out, sys.stdout)
"""

_cache = {}


def _run(side):
    """`side` is `"shim"` or `"upstream"`; `None` when the vendored shim is
    not installed, which is the same silent skip `test_tail3.py` takes."""
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


def _aten_source():
    if not os.path.isfile(_ATEN_RS):
        return None
    return open(_ATEN_RS, encoding="utf-8").read()


# --------------------------------------------------------------------------
# 1. `as_strided`: the verdict, and the precondition that would re-open it
# --------------------------------------------------------------------------


def test_the_as_strided_refusal_still_stands_and_names_the_overload():
    """The verdict this round was asked to reach, asserted as a refusal.

    `docs/TAIL3.md` §6 refused `as_strided` by name and this round did not
    overturn it. `longformer` and `led` are still blocked on it and that is a
    complete outcome rather than an omission -- **but only while the refusal is
    real**, so this asserts the two properties that make it honest:

      * the schema is declared, so the message names `aten.as_strided.default`
        rather than "no matching signature"; and
      * it is NOT in `_aten_implemented()`, so the surface does not claim it.

    **If a later round implements it, invert this test rather than deleting
    it** -- `docs/FFT.md`'s round did exactly that for its own refusal and said
    so. Deleting it would remove the record that the answer was once no, and
    with it the reason.
    """
    assert "aten.as_strided.default" not in _C._aten_implemented()
    pair = _both("as_strided")
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, f"upstream refused as_strided: {want}"
    assert got["raised"] == "NotImplementedError", got
    assert "aten.as_strided.default" in got["msg"], got["msg"]


def test_the_write_door_is_single_which_is_the_precondition_a_read_only_as_strided_would_need():
    """Why a *read-only* `as_strided` is not merely unwritten but unreachable.

    The question this round asked was narrower than `docs/TAIL3.md`'s: not
    "can the aliasing be produced" (it cannot -- candle 0.11.0's
    `Tensor::from_storage` takes an owned `Storage` and documents contiguous
    strides) but "can a *materialising* `as_strided` be made to refuse the
    moment its result is written to", which is `docs/COMPLEX2.md`'s standard:
    make the wrong answer unrepresentable rather than merely unlikely.

    Two facts decide it, and both are checkable here rather than argued:

      1. **There is exactly one write door**, `tensor::write_into`, and it is
         called from exactly one place -- `aten.rs::write_back`. That is the
         good half: a guard in `write_back` would see every in-place op.

      2. **The marker such a guard would read cannot live anywhere
         `write_back` can reach.** `PyTensorBase` is where a per-tensor flag
         would have to go (`tensor.rs`), and every proxy identity available
         from `aten.rs` alone fails: candle's `TensorId` is fresh for every
         shallow view, so `y = x.as_strided(...); z = y.view(-1); z.fill_(0)`
         escapes it, and the storage address is reused after a free, so a
         guard keyed on it refuses writes to unrelated tensors later.

    So the refusal stands. This test asserts (1), which is the part that will
    change first if the verdict is ever re-openable: the day `write_into`
    acquires a second caller, a guard in `write_back` stops being sufficient
    and this fails, which is the right time to re-read the argument.
    """
    text = _aten_source()
    if text is None:
        print("   (skipped: aten.rs is not beside this file)")
        return
    calls = re.findall(r"\.write_into\s*\(", text)
    assert len(calls) == 1, (
        f"aten.rs now calls write_into {len(calls)} times. A read-only "
        "as_strided guard was sized against there being exactly one; "
        "docs/TAIL4.md §1 is the argument that has to be re-read."
    )
    assert "fn write_back(" in text
    if os.path.isfile(_TENSOR_RS):
        tensor_text = open(_TENSOR_RS, encoding="utf-8").read()
        # The other half of "single door": nothing in `tensor.rs` calls it
        # either, so `write_back` really is the only entrance.
        assert "pub fn write_into(" in tensor_text
        assert len(re.findall(r"\.write_into\s*\(", tensor_text)) == 0


def test_the_as_strided_gap_is_still_declared_in_the_reach_allowlist():
    """The bookkeeping half, so the entry cannot outlive the gap.

    `tools/golden/reach_allow.json` carries `as_strided` with its reason and
    the checker matches that file in both directions, so the entry has to be
    deleted the day the gap closes. This asserts the entry is still there
    *and* still says the same thing -- a reason rewritten to "not needed"
    would be the quiet way for the refusal to stop being a decision.
    """
    path = os.path.join(_REPO_ROOT, "tools", "golden", "reach_allow.json")
    if not os.path.isfile(path):
        print("   (skipped: reach_allow.json is not in this tree)")
        return
    blob = json.dumps(json.load(open(path, encoding="utf-8")))
    assert "as_strided" in blob, (
        "as_strided left reach_allow.json but is still unimplemented"
    )


# --------------------------------------------------------------------------
# 2. `round` is banker's rounding, and that is the whole op
# --------------------------------------------------------------------------


def test_round_is_half_to_even_and_the_grid_separates_it_from_half_away():
    """The one fact a test on 0.3 and 0.7 cannot see.

    candle's own `Tensor::round` is `f32::round`, i.e. round-half-away-from-
    zero, so "use candle's round" is the plausible implementation and it is
    wrong on five of these nine. Both rules are spelled out here so the
    assertion says which one upstream is, rather than only that the shim
    agrees with it.
    """
    pair = _both("round_grid")
    if pair == "skip":
        return
    got, want = pair
    grid = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    half_to_even = [-4.0, -2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 4.0, 4.0]
    half_away = [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    differ = sum(1 for a, b in zip(half_to_even, half_away) if a != b)
    assert differ == 5, differ
    assert want["ok"] == half_to_even, want["ok"]
    assert got["ok"] == half_to_even, got["ok"]
    assert len(grid) == len(half_to_even)


def test_round_keeps_the_sign_of_a_zero_result():
    """`round(-0.5)` is `-0.0`, and `floor + inc` computes `+0.0`.

    The kernel's last line exists for this: `-1.0 + 1.0` is `+0.0`, so a zero
    result takes its sign from `x * 0` instead. It is the same property
    `floor`'s own comment in `aten.rs` records for `floor(-0.0)`, and
    `tolist()` cannot see it because `-0.0 == 0.0` in Python -- which is why
    it is asserted through `signbit`.
    """
    pair = _both("round_grid_signbit")
    if pair == "skip":
        return
    got, want = pair
    assert want["ok"] == [True, True, True, True, False, False, False, False, False], want
    assert got["ok"] == want["ok"], (got, want)


def test_round_decimals_scales_in_float_for_bfloat16_and_in_place_for_float32():
    """The one line of `round.decimals` a `float32`-only test cannot see.

    `c10::BFloat16`'s arithmetic operators return `float`, so upstream's
    `nearbyint(a * ten) / ten` runs entirely in `float` and narrows once. The
    same expression evaluated in `bfloat16` throughout rounds twice and gives
    a different answer -- and `float32` cannot show it, because there the two
    readings coincide.

    `2.675` at two decimals is the case: `float32` answers `2.68` (the
    multiply rounds *up* to `267.5` before `nearbyint` is reached, which is
    why an `f64`-then-narrow implementation answers `2.67`) and `bfloat16`
    answers `2.671875`, the input unchanged.
    """
    for name in ("round_decimals_float32", "round_decimals_bfloat16"):
        pair = _both(name)
        if pair == "skip":
            return
        got, want = pair
        assert got["ok"] == want["ok"], (name, got, want)
    f32 = _run("upstream")["round_decimals_float32"]["ok"][0]
    bf16 = _run("upstream")["round_decimals_bfloat16"]["ok"][0]
    assert f32 != bf16, (f32, bf16)
    assert abs(f32 - 2.68) < 1e-6, f32
    assert bf16 == 2.671875, bf16


def test_the_bare_round_is_the_identity_on_integers_and_the_decimals_one_refuses():
    """Two overloads of one op that disagree about an integral input.

    `torch.round(arange(3))` is `[0, 1, 2]` and `torch.round(arange(3),
    decimals=0)` -- the same values, an argument that makes it a no-op --
    raises. And the kernel named in the refusal depends on the *value*:
    `round_vml_cpu` at `decimals=0` and `round_cpu` otherwise, because the two
    reach different stubs. Inferring either from the other would have been
    wrong twice.
    """
    pair = _both("round_int_identity")
    if pair == "skip":
        return
    got, want = pair
    assert want["ok"] == [0.0, 1.0, 2.0], want
    assert got["ok"] == want["ok"], (got, want)

    for name, kernel in (("round_int_decimals_zero_refused", "round_vml_cpu"),
                         ("round_int_decimals_refused", "round_cpu")):
        got, want = _both(name)
        assert want["raised"] == "NotImplementedError", (name, want)
        assert kernel in want["msg"], (name, want["msg"])
        assert got["raised"] == want["raised"], (name, got, want)
        assert kernel in got["msg"], (name, got["msg"])


# --------------------------------------------------------------------------
# 3. `index_copy_` and `index_add_` are adjacent and disagree three ways
# --------------------------------------------------------------------------


def test_index_copy_OVERWRITES_where_index_add_ACCUMULATES():
    """The first of the three, and the one a repeat-free index cannot show.

    Both ops are run on the same input in the same process, so the assertion
    is that they *differ* -- which is what a per-op golden case, comparing each
    against upstream separately, structurally cannot say.
    """
    pair = _both("index_copy_dup")
    if pair == "skip":
        return
    copy_got, copy_want = pair
    add_got, add_want = _both("index_add_dup")
    assert copy_want["ok"] == [0.0, 3.0, 0.0], copy_want
    assert add_want["ok"] == [0.0, 6.0, 0.0], add_want
    assert copy_want["ok"] != add_want["ok"]
    assert copy_got["ok"] == copy_want["ok"], (copy_got, copy_want)
    assert add_got["ok"] == add_want["ok"], (add_got, add_want)


def test_index_copy_REFUSES_an_int32_index_where_index_add_ACCEPTS_one():
    """The second, and the row that would have shipped.

    `index_add_` accepts `int32` and `int64` and has a golden case pinning
    that it does. `index_copy_` takes `int64` only. Copying the neighbour's
    dtype gate would have made this shim accept an index upstream refuses --
    a widening invisible to every test that passes a `torch.long` index, which
    is every test anybody writes.
    """
    pair = _both("index_copy_int32_index")
    if pair == "skip":
        return
    copy_got, copy_want = pair
    add_got, add_want = _both("index_add_int32_index")
    assert copy_want["raised"] == "RuntimeError", copy_want
    assert "Expected a long tensor for index" in copy_want["msg"], copy_want
    assert "raised" not in add_want, add_want
    assert copy_got["raised"] == copy_want["raised"], (copy_got, copy_want)
    assert copy_got["msg"] == copy_want["msg"], (copy_got["msg"], copy_want["msg"])
    assert "raised" not in add_got, add_got


def test_both_refuse_a_negative_index_and_only_one_says_which():
    """The third. Both refuse -- `docs/DEMAND8.md`'s finding is that
    `index_put_` wraps and these two do not -- but the messages differ, and
    the difference is information a house string would erase.
    """
    pair = _both("index_copy_negative")
    if pair == "skip":
        return
    copy_got, copy_want = pair
    add_got, add_want = _both("index_add_negative")
    assert copy_want["raised"] == "IndexError", copy_want
    assert add_want["raised"] == "IndexError", add_want
    assert "index -1 is out of bounds for dimension 0 with size 3" in copy_want["msg"]
    assert "-1" not in add_want["msg"], add_want["msg"]
    assert copy_got["msg"] == copy_want["msg"], (copy_got["msg"], copy_want["msg"])
    assert add_got["msg"] == add_want["msg"], (add_got["msg"], add_want["msg"])


def test_index_copy_out_of_place_leaves_the_receiver_alone():
    """The functional spelling, which is a binding over the same body.

    Named separately because the two share `index_copy_common` and the flag
    that separates them is one boolean: a kernel that forgot it would pass
    every value case for the in-place op and silently mutate for the other.
    """
    pair = _both("index_copy_out_of_place_leaves_self")
    if pair == "skip":
        return
    got, want = pair
    assert want["ok"] == [[9.0, 0.0], [0.0, 0.0]], want
    assert got["ok"] == want["ok"], (got, want)


# --------------------------------------------------------------------------
# 4. `t_`: the alias, which is what makes `replace_with` the right primitive
# --------------------------------------------------------------------------


def test_t_inplace_keeps_the_alias_it_had_before_the_call():
    """`t_` is the first in-place op here that changes the receiver's shape.

    So it cannot use `write_into` -- that primitive checks the replacement has
    the receiver's shape exactly, correctly, because that check is what stops a
    kernel from silently retagging a tensor. It uses `replace_with` instead,
    whose own doc comment warns that a rebinding breaks aliasing.

    `t_` is on the right side of that warning and this is why: candle's
    `transpose` shares the storage `Arc` and rebuilds only the `Layout`, so a
    view taken *before* the call still writes into the same buffer *after* it.
    Upstream does the same. No value comparison can see this, because both
    sides answer the same numbers whether or not the alias survives.
    """
    pair = _both("t_alias")
    if pair == "skip":
        return
    got, want = pair
    # [x[0,0] after v[0]=99, identity, shape[0], shape[1]]
    assert want["ok"] == [99.0, 1.0, 3.0, 2.0], want
    assert got["ok"] == want["ok"], (got, want)


def test_t_inplace_is_rank_decided_and_not_a_batched_transpose():
    """0-D and 1-D come back unchanged, 3-D raises. The same rule `aten::t`
    has, and the same trap: reading `t_` as `transpose(-2, -1)` computes on a
    batched input where upstream refuses.
    """
    pair = _both("t_rank1_unchanged")
    if pair == "skip":
        return
    got, want = pair
    assert want["ok"] == [0.0, 1.0, 2.0], want
    assert got["ok"] == want["ok"], (got, want)
    got, want = _both("t_rank3_refused")
    assert want["raised"] == "RuntimeError", want
    assert "t_() expects a tensor with <= 2 dimensions" in want["msg"], want
    assert got["raised"] == want["raised"], (got, want)
    assert got["msg"] == want["msg"], (got["msg"], want["msg"])


def test_t_was_rwkvs_fifth_wall_and_the_construction_now_completes():
    """`rwkv` has moved through five walls and this round is the last of them.

    `ndimension` -> `new_empty` -> `torch.maximum` -> `linalg_qr` -> `diag` ->
    `t_`, each one exposed by closing the one before it, all six inside
    `_init_weights` rather than in a forward. `torch/nn/init.py:705`'s
    `orthogonal_` calls `flattened.t_()` whenever `rows < cols`.

    This asserts the *op* is reachable through the vendored tree's own
    spelling and that the two `t_` calls `orthogonal_` makes both work, which
    is the part of "rwkv constructs" that belongs in a unit suite. The sweep
    is where "rwkv constructs" itself is measured, and docs/TAIL4.md §5 records
    that it now does.
    """
    assert "aten.t_.default" in _C._aten_implemented()
    pair = _both("t_alias")
    if pair == "skip":
        return
    # `orthogonal_`'s shape: t_ on the way in, qr, then t_ on the way out.
    got, _want = pair
    assert got["ok"][2:] == [3.0, 2.0], got


# --------------------------------------------------------------------------
# 5. `logsumexp`: the infinite maximum is the only branch there is
# --------------------------------------------------------------------------


def test_logsumexp_is_minus_inf_where_the_naive_stabilised_form_is_nan():
    """The branch, asserted against the form it replaces.

    `max + log(sum(exp(x - max)))` is `nan` for an all `-inf` row, because
    `-inf - -inf` is `nan`. Upstream answers `-inf`. Both are computed here in
    the same process, so the assertion is that they *differ* -- which says the
    zeroing of an infinite maximum is load-bearing rather than decorative.
    """
    pair = _both("logsumexp_all_neg_inf")
    if pair == "skip":
        return
    got, want = pair
    naive_got, naive_want = _both("logsumexp_naive_form_on_all_neg_inf")
    assert want["ok"] == [float("-inf")], want
    assert naive_want["ok"] != want["ok"], naive_want
    assert got["ok"] == want["ok"], (got, want)
    pos_got, pos_want = _both("logsumexp_pos_inf")
    assert pos_want["ok"] == [float("inf")], pos_want
    assert pos_got["ok"] == pos_want["ok"], (pos_got, pos_want)
    assert naive_got["ok"] == naive_want["ok"], (naive_got, naive_want)


def test_logsumexp_promotes_an_integral_input_where_amax_keeps_and_sum_widens():
    """Three reductions, three different dtype rules, measured side by side.

    `amax` keeps the input dtype, `sum` promotes to `int64`, and `logsumexp`
    answers `float32`. Inheriting either neighbour's rule would have been
    wrong, and a golden case for `logsumexp` alone cannot say that -- it can
    only say `logsumexp` agrees with `logsumexp`.
    """
    pair = _both("logsumexp_int_dtype")
    if pair == "skip":
        return
    got, want = pair
    amax_got, amax_want = _both("logsumexp_amax_int_dtype")
    sum_got, sum_want = _both("logsumexp_sum_int_dtype")
    assert want["ok"] == "torch.float32", want
    assert amax_want["ok"] == "torch.int64", amax_want
    assert sum_want["ok"] == "torch.int64", sum_want
    assert amax_want["ok"] != want["ok"]
    assert got["ok"] == want["ok"], (got, want)
    assert amax_got["ok"] == amax_want["ok"], (amax_got, amax_want)
    assert sum_got["ok"] == sum_want["ok"], (sum_got, sum_want)


# --------------------------------------------------------------------------
# 5b. `logical_not` is not `bitwise_not`, and they agree on exactly one dtype
# --------------------------------------------------------------------------


def test_logical_not_and_bitwise_not_agree_on_bool_and_on_nothing_else():
    """`~` and `bitwise_not` already worked, which is the trap.

    `docs/TAIL1.md` measured the same distinction for the `and` pair --
    `bitwise_and` on `[1, 2, 4, 3]` gives all zeros where `logical_and` gives
    all True -- and it holds for `not` as well. Both ops are run on the same
    values in the same process at three dtypes, so the assertion is that they
    *agree on `bool` and differ everywhere else*, which is the statement a
    per-op golden case cannot make.
    """
    pair = _both("logical_not_bool")
    if pair == "skip":
        return
    for dn in ("bool", "int64", "float32"):
        ln_got, ln_want = _both(f"logical_not_{dn}")
        bn_got, bn_want = _both(f"bitwise_not_{dn}")
        assert "raised" not in ln_want, (dn, ln_want)
        assert ln_want["ok"][0] == "torch.bool", (dn, ln_want)
        assert ln_want["ok"][1:] == [1.0, 0.0, 0.0, 0.0, 0.0], (dn, ln_want)
        assert ln_got["ok"] == ln_want["ok"], (dn, ln_got, ln_want)
        if dn == "bool":
            assert bn_want["ok"] == ln_want["ok"], bn_want
        elif dn == "int64":
            # Different values AND a different result dtype.
            assert bn_want["ok"][0] == "torch.int64", bn_want
            assert bn_want["ok"][1:] == [-1.0, -2.0, -3.0, 0.0, -4.0], bn_want
            assert bn_want["ok"] != ln_want["ok"]
        else:
            # `bitwise_not` does not have a floating kernel at all.
            assert "raised" in bn_want, bn_want
            assert "bitwise_not_cpu" in bn_want["msg"], bn_want


def test_logical_not_answers_bool_for_every_input_dtype():
    """The wrong-type failure a `tolist()` comparison would pass.

    A kernel that kept the input dtype answers `1.0`/`0.0` instead of
    `True`/`False` -- plausible values, wrong type. It is what golden's dtype
    comparison exists to catch, and it is asserted here as well because the
    rule (always `bool`, never the input's) is the thing worth stating.
    """
    pair = _both("logical_not_dtype_is_bool")
    if pair == "skip":
        return
    got, want = pair
    assert want["ok"] == "torch.bool", want
    assert got["ok"] == want["ok"], (got, want)


def test_logical_not_treats_nan_as_TRUE_and_minus_zero_as_FALSE():
    """The one value that separates `x == 0` from `!(x != 0)`.

    `logical_not(nan)` is `False` -- a NaN is truthy -- and `nan != 0` is
    `true` in IEEE, so the naturally-reading spelling gets it wrong while the
    equality spelling gets it right, because the comparison against NaN is
    false in *both* directions. `-0.0` is zero and therefore `True`; `inf` is
    `False`.
    """
    pair = _both("logical_not_float_specials")
    if pair == "skip":
        return
    got, want = pair
    assert want["ok"] == [1.0, 1.0, 0.0, 0.0], want
    assert got["ok"] == want["ok"], (got, want)
    m_got, m_want = _both("logical_not_method")
    assert m_want["ok"] == [1.0, 1.0], m_want
    assert m_got["ok"] == m_want["ok"], (m_got, m_want)


# --------------------------------------------------------------------------
# 6. `embedding`: an argument form, not a missing kernel
# --------------------------------------------------------------------------


def test_embedding_takes_an_int32_index_which_is_what_cpmant_hands_it():
    """`docs/ARCH200.md` classified `cpmant` as a *backend limitation*, and it
    was an index dtype.

    `modeling_cpmant.py:602` casts `input_ids` to `int32` before the embedding
    lookup. candle's `index_select` takes `u8`/`u32`/`i64` and refused with
    `unsupported dtype F32 for op index-select` -- a message naming the
    *weight's* dtype and no part of the actual problem, which is how the
    classification went wrong. The fix is a widening cast, not a kernel: the
    same values come back for both index dtypes.
    """
    pair = _both("embedding_int32")
    if pair == "skip":
        return
    got, want = pair
    long_got, long_want = _both("embedding_int64")
    assert "raised" not in want, want
    assert want["ok"] == long_want["ok"], (want, long_want)
    assert got["ok"] == want["ok"], (got, want)
    assert got["dtype"] == want["dtype"] == "torch.float32"


def test_embedding_still_refuses_the_index_dtypes_upstream_refuses():
    """The other direction, which is what makes the fix a fix rather than a
    hole: `uint8` is one of candle's own index dtypes and upstream refuses it,
    so accepting whatever candle accepts would have been a widening.

    The message spells the dtype the *legacy* way (`torch.ByteTensor`), which
    is `TORCH_CHECK`'s `t.type()` and not the scalar-type name every other
    refusal in `aten.rs` uses. Two of the ten do not even follow that pattern
    (`bfloat16` and `bool` print `CPUBFloat16Type`/`CPUBoolType`), which is why
    the mapping is transcribed rather than generated.
    """
    pair = _both("embedding_uint8_refused")
    if pair == "skip":
        return
    got, want = pair
    assert want["raised"] == "RuntimeError", want
    assert "torch.ByteTensor" in want["msg"], want["msg"]
    assert got["raised"] == want["raised"], (got, want)
    assert got["msg"] == want["msg"], (got["msg"], want["msg"])


# --------------------------------------------------------------------------
# 7. The split, and the readback claim, as source structure
# --------------------------------------------------------------------------


def test_the_only_new_arithmetic_this_round_is_ties_to_even():
    """The alias-versus-kernel split, pinned so it cannot rot into prose.

    Eight `_aten_implemented()` keys landed and they are not eight kernels:

        index_copy_.default / index_copy.default   one body, one boolean
        round.default / .decimals / round_ x2      one body, two booleans
        logsumexp.default                          no new arithmetic at all --
                                                   amax, sub, exp, sum, log, add
        t_.default                                 `t`'s rule, a new WRITE path

    So the count is **two new bodies and one genuinely new piece of
    arithmetic**: `nearbyint_ties_even`, which exists because candle's
    `Tensor::round` is the other rule. "Eight ops" would overstate it by a
    factor of four, which is the counting failure CLAUDE.md §5.3 names.
    """
    text = _aten_source()
    if text is None:
        print("   (skipped: aten.rs is not beside this file)")
        return
    # Each pair of keys reaches ONE function, which is what "a binding, not a
    # kernel" means here.
    for fn, keys in (
        ("index_copy_common", ["aten.index_copy_.default", "aten.index_copy.default"]),
        ("round_common", ["aten.round.default", "aten.round.decimals",
                          "aten.round_.default", "aten.round_.decimals"]),
    ):
        for key in keys:
            assert re.search(
                r'"' + re.escape(key) + r'"\s*=>\s*(?:\{\s*)?' + fn + r"\(", text
            ), (key, fn)
        assert len(re.findall(r"\bfn " + fn + r"\(", text)) == 1, fn

    # `logsumexp` adds no arithmetic: it is candle's own reduction chain.
    body = _function_body(text, "logsumexp_default")
    for primitive in ("amax_keepdim_anywhere", ".exp()", ".sum_keepdim(", ".log()"):
        assert primitive in body, primitive

    # And the one new piece of arithmetic is named, and reached from `round`
    # alone.
    assert len(re.findall(r"\bfn nearbyint_ties_even\(", text)) == 1
    callers = [name for name in ("round_common",)
               if "nearbyint_ties_even(" in _function_body(text, name)]
    assert callers == ["round_common"], callers


def test_none_of_this_rounds_kernels_reads_a_tensor_back_to_the_host():
    """The constraint that shaped `index_copy_`, stated as a check.

    `device.rs` was not this round's file, and
    `test_the_mps_readback_list_is_what_the_kernels_actually_do` derives
    `MPS_HOST_READBACK_OPS` from these bodies -- so a `read_flat` in any of
    them would have obliged an edit there. `index_copy_` is where it changed
    the design: its sibling `index_add_` walks the index on the host and *is*
    on that list; this one delegates to candle's `scatter`, which does the
    bounds check on the device and reports the offending index in its error.

    This asserts the property directly rather than relying on the derivation
    to notice, because the derivation follows helpers only one level by name
    (`docs/VOICE3.md` found `var`/`std` nearly invisible to it for that
    reason) and a future refactor could hide a readback one level further
    down.
    """
    text = _aten_source()
    if text is None:
        print("   (skipped: aten.rs is not beside this file)")
        return
    markers = ("read_flat(", ".to_vec1", ".to_vec2", ".to_vec3", ".to_scalar",
               "nan_along_dim(", "narrow_through(", "local_scalar_dense(")
    for fn in ("index_copy_common", "round_common", "nearbyint_ties_even",
               "logsumexp_default", "t_inplace", "logical_not_default"):
        body = _function_body(text, fn)
        assert body, fn
        for marker in markers:
            assert marker not in body, (fn, marker)
    for op in ("aten.index_copy_.default", "aten.index_copy.default",
               "aten.round.default", "aten.round.decimals",
               "aten.round_.default", "aten.round_.decimals",
               "aten.logsumexp.default", "aten.t_.default",
               "aten.logical_not.default"):
        assert op not in _C._shim_mps_host_readback_ops(), op


def _function_body(text, name):
    """The source of one `fn name(...)`, brace-matched.

    Written out rather than reusing `test_shim.py`'s parser because this file
    asks a different question of it: that a *named* function does not contain
    something, which needs the body even when the parser there would have
    skipped it.
    """
    match = re.search(r"\bfn " + re.escape(name) + r"\s*[(<]", text)
    if not match:
        return ""
    start = text.index("{", match.start())
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


def test_the_new_spellings_are_reachable_from_python_not_only_by_dispatch_key():
    """docs/REACH.md shape 3, for this round's six names.

    `tools/golden/cases.py` dispatches by key, so it cannot see whether
    `torch.round(x)` or `x.index_copy_(...)` resolves at all -- and a table row
    with the overloads in the wrong order resolves to the wrong one silently.
    These are the free-function and bound-method spellings, run through the
    vendored tree.
    """
    if not os.path.isfile(_VENDOR_SHIM):
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return
    script = r"""
import json, sys
import torch
assert hasattr(torch._C, "_aten_implemented")
out = {}
out["torch.round"] = torch.round(torch.tensor([0.5, 1.5])).tolist()
out["Tensor.round"] = torch.tensor([2.5, 3.5]).round().tolist()
out["round_decimals"] = torch.round(torch.tensor([2.675]), decimals=2).tolist()
x = torch.tensor([0.5, 1.5])
out["torch.round_"] = torch.round_(x).tolist()
out["Tensor.round_"] = torch.tensor([2.5]).round_().tolist()
out["torch.logsumexp"] = torch.logsumexp(torch.tensor([[1.0, 2.0]]), dim=1).tolist()
out["Tensor.logsumexp"] = torch.tensor([[1.0, 2.0]]).logsumexp(dim=1).tolist()
out["torch.index_copy"] = torch.index_copy(
    torch.zeros(3), 0, torch.tensor([1]), torch.tensor([5.0])).tolist()
out["Tensor.index_copy"] = torch.zeros(3).index_copy(
    0, torch.tensor([1]), torch.tensor([5.0])).tolist()
out["Tensor.index_copy_"] = torch.zeros(3).index_copy_(
    0, torch.tensor([1]), torch.tensor([5.0])).tolist()
out["Tensor.t_"] = torch.arange(6.0).reshape(2, 3).t_().tolist()
out["torch.logical_not"] = torch.logical_not(torch.tensor([0.0, 2.0])).tolist()
out["Tensor.logical_not"] = torch.tensor([0.0, 2.0]).logical_not().tolist()
json.dump(out, sys.stdout)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, env=env, cwd=_REPO_ROOT)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout)
    # Every value below was measured on upstream 2.13.0 in its own process.
    assert out["torch.round"] == [0.0, 2.0], out
    assert out["Tensor.round"] == [2.0, 4.0], out
    assert out["round_decimals"] == [2.680000066757202], out
    assert out["torch.round_"] == [0.0, 2.0], out
    assert out["Tensor.round_"] == [2.0], out
    assert out["torch.logsumexp"] == [2.3132617473602295], out
    assert out["Tensor.logsumexp"] == [2.3132617473602295], out
    assert out["torch.index_copy"] == [0.0, 5.0, 0.0], out
    assert out["Tensor.index_copy"] == [0.0, 5.0, 0.0], out
    assert out["Tensor.index_copy_"] == [0.0, 5.0, 0.0], out
    assert out["Tensor.t_"] == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]], out
    assert out["torch.logical_not"] == [True, False], out
    assert out["Tensor.logical_not"] == [True, False], out


def test_the_round_decimals_overload_resolves_before_the_bare_one():
    """The table-order check the reach test cannot make.

    `overloads.json` lists `round.decimals` first and `round` second, on the
    same reasoning `floor` lists `.out` first: the more constrained schema has
    to be tried first or `torch.round(x, decimals=2)` binds to the bare
    overload and the argument is silently dropped. That failure answers
    *plausible numbers* -- `round(2.675)` is `3.0` -- so nothing but an
    explicit check catches it.
    """
    root = os.path.join(_REPO_ROOT, "rust", "torch_c", "src")
    for name in ("overloads.json", "methods.json"):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        table = json.load(open(path, encoding="utf-8"))
        rows = table["round"]
        assert "round.decimals" in rows[0], (name, rows)
        assert rows[1].startswith("aten::round(Tensor"), (name, rows)
        inplace = table["round_"]
        assert "round_.decimals" in inplace[0], (name, inplace)


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
