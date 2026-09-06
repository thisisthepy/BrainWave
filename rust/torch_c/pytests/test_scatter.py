"""The scatter family, `bucketize` and `prod`, through the spellings the
architectures actually use (docs/SCATTER.md).

`tools/golden/cases.py` compares these kernels against upstream at the
`_aten_dispatch` level, which is the right place for dtype and boundary
coverage and the wrong place for two questions this file exists to answer:

  * **Does the user-level spelling reach the kernel at all?** docs/ARCH100.md's
    finding was that missing *bindings* outnumber missing kernels 49 to 22, and
    `aten.scatter.value` is that finding in one op -- the schema was already in
    both transcribed tables and only the dispatch arm was missing, so a kernel
    test would have passed while `zeros.scatter(1, idx, 1)` still raised. Every
    assertion below goes through `torch.<name>` or `Tensor.<name>`.
  * **Does the in-place form write through a view?** `scatter_` returns `self`
    whichever way it is implemented, so a test that reads the return value
    passes against a kernel that computed into a fresh buffer and handed it
    back -- docs/VIEWS.md §6's failure exactly. The cases here read the
    **base**.

Both sides are measured. Every expectation is computed by running the same
program on upstream torch in a subprocess with `PYTHONPATH` stripped, so
nothing here is a transcribed constant that could drift.
"""

import json
import os
import subprocess
import sys

from test_shim import _C, _CKPT_VENDOR_DIR, _CKPT_VENDOR_SHIM


# The programs. Each prints one JSON object and must print `who` so that a
# run against the wrong interpreter is visible rather than silently green.
_PROBE = r'''
import json
import torch

out = {"who": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}


def record(name, fn):
    try:
        out[name] = {"ok": fn()}
    except Exception as exc:                                  # noqa: BLE001
        out[name] = {"err": type(exc).__name__}


# --- the seven `TensorBase.scatter_` architectures ---------------------------
# `exaone_moe`, `glm4_moe`, `glm4_moe_lite`, `mistral4`, `nemotron_h` and
# `solar_open` all reach the identical line in their MoE group router:
#     group_mask = torch.zeros_like(group_scores)
#     group_mask.scatter_(1, group_idx, 1)
def group_router():
    scores = torch.tensor([[0.1, 0.9, 0.4, 0.7], [0.6, 0.2, 0.8, 0.3]])
    group_idx = torch.topk(scores, k=2, dim=-1, sorted=False)[1]
    mask = torch.zeros_like(scores)
    returned = mask.scatter_(1, group_idx, 1)
    return {
        "mask": mask.tolist(),
        "same_object": returned is mask,
        "dtype": str(mask.dtype),
    }


record("group_router", group_router)


# `groupvit`'s straight-through estimator, which spells the same op with a
# float value and a keepdim'd argmax index.
def hard_softmax():
    logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 0.5, 4.0]])
    soft = logits.softmax(-1)
    index = soft.max(-1, keepdim=True)[1]
    hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
    return hard.tolist()


record("hard_softmax", hard_softmax)


# --- the four `aten.scatter.value` architectures -----------------------------
# `axk2`, `deepseek_v32` and `glm_moe_dsa`: a BOOL receiver and a bool value.
def bool_index_mask():
    topk = torch.tensor([[[0, 2], [1, 3]]])
    mask = topk.new_ones((1, 2, 4), dtype=torch.bool).scatter(-1, topk, False)
    return {"mask": mask.tolist(), "dtype": str(mask.dtype)}


record("bool_index_mask", bool_index_mask)


# `jetmoe`: an int value into a float receiver, out of place.
def jetmoe_gates():
    logits = torch.tensor([[0.1, 0.9, 0.4], [0.6, 0.2, 0.8]])
    _, top_k_indices = logits.topk(2, dim=1)
    zeros = torch.zeros([2, 3], dtype=logits.dtype)
    gates = zeros.scatter(1, top_k_indices, 1)
    return {"gates": gates.tolist(), "untouched": zeros.tolist()}


record("jetmoe_gates", jetmoe_gates)


# --- write-through, the property a return-value check cannot see -------------
def writes_through_a_view():
    base = torch.zeros(3, 4)
    column = base[:, 1]
    column.scatter_(0, torch.tensor([0, 2]), 5.0)
    return base.tolist()


record("writes_through_a_view", writes_through_a_view)


def refuses_an_overlapping_receiver():
    torch.zeros(1, 3).expand(2, 3).scatter_(1, torch.tensor([[0]]), 5.0)
    return "wrote"


record("refuses_an_overlapping_receiver", refuses_an_overlapping_receiver)


# --- `higgs_audio_v2`'s masked_scatter --------------------------------------
def masked_scatter_rows():
    hidden = torch.zeros(2, 3, 4)
    token_mask = torch.tensor([[True, False, True], [False, True, False]])
    replacement = torch.arange(12.0).reshape(3, 4)
    return hidden.masked_scatter(token_mask.unsqueeze(-1), replacement).tolist()


record("masked_scatter_rows", masked_scatter_rows)


def masked_scatter_reads_source_in_logical_order():
    # A transposed source: storage order and logical order differ, and only
    # one of them is upstream's answer.
    return torch.zeros(2, 3).masked_scatter(
        torch.tensor([[True, False, True], [False, True, False]]),
        torch.arange(6.0).reshape(2, 3).t(),
    ).tolist()


record("masked_scatter_reads_source_in_logical_order",
       masked_scatter_reads_source_in_logical_order)


def masked_scatter_too_few():
    return torch.zeros(2, 3).masked_scatter(
        torch.tensor([[True, False, True], [False, True, False]]),
        torch.tensor([1.0, 2.0]),
    ).tolist()


record("masked_scatter_too_few", masked_scatter_too_few)


# --- `idefics3_vision` / `smolvlm_vision`'s bucketize ------------------------
def bucketize_on_the_boundaries():
    boundaries = torch.tensor([1.0, 3.0, 5.0, 7.0])
    values = torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 8.0])
    return {
        "left": torch.bucketize(values, boundaries).tolist(),
        "right": torch.bucketize(values, boundaries, right=True).tolist(),
        "dtype": str(torch.bucketize(values, boundaries).dtype),
        "int32": str(
            torch.bucketize(values, boundaries, out_int32=True).dtype
        ),
    }


record("bucketize_on_the_boundaries", bucketize_on_the_boundaries)


def bucketize_duplicate_boundaries():
    edges = torch.tensor([1.0, 3.0, 3.0, 5.0])
    v = torch.tensor([3.0])
    return [
        torch.bucketize(v, edges).tolist(),
        torch.bucketize(v, edges, right=True).tolist(),
    ]


record("bucketize_duplicate_boundaries", bucketize_duplicate_boundaries)


def bucketize_nonfinite():
    edges = torch.tensor([1.0, 3.0, 5.0, 7.0])
    v = torch.tensor([float("nan"), float("inf"), float("-inf")])
    return [
        torch.bucketize(v, edges).tolist(),
        torch.bucketize(v, edges, right=True).tolist(),
    ]


record("bucketize_nonfinite", bucketize_nonfinite)


def bucketize_idefics3():
    # transformers/models/idefics3/modeling_idefics3.py:162 verbatim in shape.
    boundaries = torch.arange(1, 4).to(torch.float32) / 4
    coords = torch.clamp(
        torch.arange(4, dtype=torch.float32)[None, :] * 0.25, max=1.0 - 1e-6
    )
    return torch.bucketize(coords, boundaries, right=True).tolist()


record("bucketize_idefics3", bucketize_idefics3)


def bucketize_rejects_a_matrix_of_boundaries():
    return torch.bucketize(
        torch.tensor([2.0]), torch.tensor([[1.0, 3.0]])
    ).tolist()


record("bucketize_rejects_a_matrix_of_boundaries",
       bucketize_rejects_a_matrix_of_boundaries)


# --- `tapas`' prod ----------------------------------------------------------
def prod_tapas():
    r = torch.prod(torch.tensor([4, 4]))
    return {"value": r.tolist(), "dtype": str(r.dtype)}


record("prod_tapas", prod_tapas)


def prod_empty_is_one():
    return {
        "float": torch.prod(torch.tensor([])).tolist(),
        "int": torch.prod(torch.tensor([], dtype=torch.int64)).tolist(),
        "along_dim": torch.prod(torch.zeros(2, 0), 1).tolist(),
    }


record("prod_empty_is_one", prod_empty_is_one)


def prod_widens_every_integral_input():
    return {
        name: str(torch.prod(torch.tensor([2, 3], dtype=dtype)).dtype)
        for name, dtype in [
            ("int64", torch.int64), ("int32", torch.int32),
            ("int16", torch.int16), ("uint8", torch.uint8),
            ("bool", torch.bool), ("float32", torch.float32),
            ("float16", torch.float16), ("bfloat16", torch.bfloat16),
            ("float64", torch.float64),
        ]
    }


record("prod_widens_every_integral_input", prod_widens_every_integral_input)


def prod_wraps_rather_than_saturating():
    return torch.prod(torch.tensor([2 ** 32, 2 ** 32])).tolist()


record("prod_wraps_rather_than_saturating", prod_wraps_rather_than_saturating)


def prod_dtype_casts_the_input_not_the_answer():
    # 2.5 * 3.0 is 7.5; rounding it gives 8 and truncating it gives 7.
    # Casting first gives 2 * 3 = 6, which is upstream's answer.
    return torch.prod(torch.tensor([2.5, 3.0]), dtype=torch.int64).tolist()


record("prod_dtype_casts_the_input_not_the_answer",
       prod_dtype_casts_the_input_not_the_answer)


def prod_accumulates_in_the_output_dtype():
    # bfloat16 discriminates: stepwise rounding gives 2.15625, and
    # accumulating in f64 and narrowing once gives 2.171875.
    return torch.tensor([1.1] * 8, dtype=torch.bfloat16).prod().item()


record("prod_accumulates_in_the_output_dtype",
       prod_accumulates_in_the_output_dtype)


def prod_along_a_dim():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    return {
        "dim1": torch.prod(x, 1).tolist(),
        "dim0_keepdim": torch.prod(x, 0, True).tolist(),
        "negative": torch.prod(x, -1).tolist(),
    }


record("prod_along_a_dim", prod_along_a_dim)


print(json.dumps(out))
'''


def _run(vendored):
    env = dict(os.environ)
    if vendored:
        env["PYTHONPATH"] = _CKPT_VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True, env=env, timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{'shim' if vendored else 'upstream'} probe exited "
            f"{proc.returncode}\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


_CACHE = {}


def _both():
    """Both sides of the comparison, run once and reused.

    Returns `None` when there is no vendored tree or no upstream torch, so the
    file skips rather than passing vacuously.
    """
    if "pair" not in _CACHE:
        if not os.path.isfile(_CKPT_VENDOR_SHIM):
            _CACHE["pair"] = None
        else:
            try:
                import torch  # noqa: F401
            except ImportError:
                _CACHE["pair"] = None
            else:
                shim = _run(vendored=True)
                upstream = _run(vendored=False)
                assert shim["who"] == "shim", shim["who"]
                assert upstream["who"] == "upstream", upstream["who"]
                _CACHE["pair"] = (shim, upstream)
    return _CACHE["pair"]


def _agree(name):
    pair = _both()
    if pair is None:
        print(f"   (skipped {name}: no vendored tree or no upstream torch)")
        return None
    shim, upstream = pair
    assert name in upstream, f"{name} is not in the upstream probe's output"
    assert shim[name] == upstream[name], (
        f"{name}: shim {shim[name]!r} != upstream {upstream[name]!r}"
    )
    return upstream[name]


def test_the_group_router_seven_architectures_share_one_scatter_line():
    """`group_mask.scatter_(1, group_idx, 1)` -- the identical line in
    `exaone_moe`, `glm4_moe`, `glm4_moe_lite`, `mistral4`, `nemotron_h` and
    `solar_open` (measured in transformers 5.15.1, docs/SCATTER.md §2).

    Three things at once, because the router needs all three: the values, that
    the receiver itself changed (not a copy), and that the return is the same
    object. `same_object` is the weakest of the three and is asserted anyway --
    the router discards the return value and reads `group_mask`, so a kernel
    that rebound the wrapper would leave the model reading zeros.
    """
    got = _agree("group_router")
    if got is None:
        return
    assert got["ok"]["same_object"] is True
    # An int `1` into a float receiver stays float -- the scalar is converted
    # to the receiver's dtype, it does not promote it.
    assert got["ok"]["dtype"] == "torch.float32"
    assert sum(sum(row) for row in got["ok"]["mask"]) == 4.0


def test_groupvits_straight_through_estimator():
    """The same op with a float value and a `keepdim=True` index, which is a
    `(rows, 1)` index into a `(rows, cols)` receiver -- the shape candle's own
    `scatter` refuses, and the reason `scatter_src` beside it is hand-written.
    """
    got = _agree("hard_softmax")
    if got is None:
        return
    # Exactly one 1.0 per row, or it is not a one-hot.
    assert [sum(row) for row in got["ok"]] == [1.0, 1.0]


def test_a_bool_receiver_takes_a_bool_value():
    """`axk2`/`deepseek_v32`/`glm_moe_dsa`:
    `new_ones(..., dtype=bool).scatter(-1, topk_indices.long(), False)`.

    The dtype is asserted as well as the values: a kernel that widened the
    receiver to `int64` on the way through would produce the same truth table
    and then break `masked_fill` one line later, which is where these three
    models take it.
    """
    got = _agree("bool_index_mask")
    if got is None:
        return
    assert got["ok"]["dtype"] == "torch.bool"


def test_jetmoes_gates_do_not_disturb_the_zeros_they_came_from():
    """`zeros.scatter(1, top_k_indices, 1)` is out of place, and `jetmoe`
    reuses `zeros` afterwards. Asserting the receiver is untouched is what
    separates `scatter` from `scatter_` here; the values alone would not.
    """
    got = _agree("jetmoe_gates")
    if got is None:
        return
    assert got["ok"]["untouched"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert [sum(row) for row in got["ok"]["gates"]] == [2.0, 2.0]


def test_scatter_writes_through_a_strided_view_and_not_into_a_copy():
    """docs/VIEWS.md §6's failure shape, asked of the new op.

    `base[:, 1]` is stride-4 and offset-1, so a kernel that wrote a contiguous
    `dst[..numel]` would corrupt the neighbouring columns rather than write
    nothing -- and one that rebound the wrapper would leave the base all zeros.
    The base is what is read; the return value is discarded.
    """
    got = _agree("writes_through_a_view")
    if got is None:
        return
    assert got["ok"] == [
        [0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
    ]


def test_scatter_refuses_a_receiver_whose_elements_share_memory():
    """Measured, and it is **not** what `masked_fill_` beside it does.

    Upstream lets `masked_fill_` write an expanded tensor and refuses
    `scatter_`, so `write_back`'s `Overlap` table gains no entry for this op.
    A shim that shared one answer for every in-place op would pass one of the
    two and fail this.
    """
    got = _agree("refuses_an_overlapping_receiver")
    if got is None:
        return
    assert got == {"err": "RuntimeError"}, got


def test_masked_scatter_fills_the_rows_higgs_audio_selects():
    """`hidden_states.masked_scatter(mask.unsqueeze(-1), replacement)`.

    The `(B, S, 1)` mask broadcasts across the hidden axis, so each true token
    consumes a whole **row** of `replacement` -- four consecutive source
    elements, not one. A kernel that expanded the source instead of the mask
    would put `replacement[0]` in every column of the first selected row.
    """
    got = _agree("masked_scatter_rows")
    if got is None:
        return
    assert got["ok"][0][0] == [0.0, 1.0, 2.0, 3.0]
    assert got["ok"][0][1] == [0.0, 0.0, 0.0, 0.0]


def test_masked_scatter_consumes_the_source_in_its_logical_order():
    """A transposed source distinguishes the two plausible readings, and only
    this one is upstream's: the elements arrive in the source's own row-major
    order (`0, 3, 1`), not in its storage order (`0, 1, 2`).
    """
    got = _agree("masked_scatter_reads_source_in_logical_order")
    if got is None:
        return
    assert got["ok"] == [[0.0, 0.0, 3.0], [0.0, 1.0, 0.0]]


def test_masked_scatter_refuses_a_source_shorter_than_the_mask():
    got = _agree("masked_scatter_too_few")
    if got is None:
        return
    assert got == {"err": "RuntimeError"}, got


def test_bucketize_is_measured_on_the_boundaries_and_not_between_them():
    """docs/SCATTER.md §5.

    `right=False` and `right=True` agree on every value that is **not** a
    boundary, so a sweep of interior points passes against either. These seven
    values include all four boundaries, which is the only place the two
    columns differ -- the same shape docs/FIXES.md's `-300..300` sweep had.
    """
    got = _agree("bucketize_on_the_boundaries")
    if got is None:
        return
    left, right = got["ok"]["left"], got["ok"]["right"]
    assert left != right, "the two flags gave the same answer on every value"
    assert got["ok"]["dtype"] == "torch.int64"
    assert got["ok"]["int32"] == "torch.int32"


def test_bucketize_lands_on_the_ends_of_a_run_of_equal_boundaries():
    """The second witness that this is `lower_bound`/`upper_bound` and not a
    scan: on `[1, 3, 3, 5]` the answers are the two **ends** of the run of
    threes, so any implementation that stopped at the first match it found for
    both flags would give the same number twice.
    """
    got = _agree("bucketize_duplicate_boundaries")
    if got is None:
        return
    assert got["ok"] == [[1], [3]], got["ok"]


def test_bucketize_puts_nan_at_the_top_the_way_upstream_does():
    """IEEE makes every comparison against NaN false, so the *negated*
    predicate upstream uses runs off the end and NaN lands past the last
    boundary. Writing the comparison the obvious way round gives 0 instead,
    and `-inf` is in the same case to prove the search is not simply saturating.
    """
    got = _agree("bucketize_nonfinite")
    if got is None:
        return
    left, right = got["ok"]
    assert left[0] == 4 and right[0] == 4, got["ok"]
    assert left[2] == 0, got["ok"]


def test_bucketize_answers_idefics3s_own_call():
    _agree("bucketize_idefics3")


def test_bucketize_requires_one_dimensional_boundaries():
    got = _agree("bucketize_rejects_a_matrix_of_boundaries")
    if got is None:
        return
    assert got == {"err": "RuntimeError"}, got


def test_prod_answers_tapas_and_widens_to_int64():
    got = _agree("prod_tapas")
    if got is None:
        return
    assert got["ok"] == {"value": 16, "dtype": "torch.int64"}


def test_the_empty_product_is_one_and_not_zero():
    """The identity that makes this not "sum with a different operator".
    Checked in three shapes, because an implementation that special-cased the
    full reduction could still return 0 along an empty axis.
    """
    got = _agree("prod_empty_is_one")
    if got is None:
        return
    assert got["ok"] == {"float": 1.0, "int": 1, "along_dim": [1.0, 1.0]}


def test_prod_widens_integrals_and_leaves_floats_at_their_own_width():
    got = _agree("prod_widens_every_integral_input")
    if got is None:
        return
    assert got["ok"]["bool"] == "torch.int64"
    assert got["ok"]["uint8"] == "torch.int64"
    assert got["ok"]["float16"] == "torch.float16"
    assert got["ok"]["bfloat16"] == "torch.bfloat16"


def test_prod_wraps_on_integer_overflow():
    got = _agree("prod_wraps_rather_than_saturating")
    if got is None:
        return
    assert got["ok"] == 0, got["ok"]


def test_the_dtype_argument_casts_the_input_before_reducing():
    got = _agree("prod_dtype_casts_the_input_not_the_answer")
    if got is None:
        return
    assert got["ok"] == 6, got["ok"]


def test_prod_accumulates_step_by_step_in_the_output_dtype():
    """`bfloat16` is the only dtype here that can tell the two apart, and it
    took a discriminating input to find one: eight copies of `1.1`. Stepwise
    `bfloat16` gives `2.15625`; accumulating in `f64` and narrowing once at the
    end gives `2.171875`, which is what the obvious implementation does.
    """
    got = _agree("prod_accumulates_in_the_output_dtype")
    if got is None:
        return
    assert got["ok"] != 2.171875, "prod narrowed once instead of per step"


def test_prod_along_a_dim_with_keepdim_and_a_negative_axis():
    _agree("prod_along_a_dim")


def test_every_op_this_round_added_is_advertised_and_dispatchable():
    """The list and the dispatch table, asked about this round's keys by name.

    `test_every_advertised_op_is_actually_dispatchable` already sweeps the
    whole list; this names the eight, so that a later change that drops one
    fails with the name of what it dropped rather than with a sweep.
    """
    implemented = set(_C._aten_implemented())
    for op in [
        "aten.scatter.value",
        "aten.scatter_.src",
        "aten.scatter_.value",
        "aten.masked_scatter.default",
        "aten.bucketize.Tensor",
        "aten.bucketize.Scalar",
        "aten.prod.default",
        "aten.prod.dim_int",
    ]:
        assert op in implemented, op


def test_reduce_is_absent_by_name_rather_than_by_omission():
    """docs/SCATTER.md §2: **none** of the eleven architectures passes
    `reduce=`, so no kernel was written for it -- and this asserts the refusal
    is the *unimplemented-op* one, which is a precise work item, rather than an
    overload-resolution failure, which is a vague one.

    If a later round implements `scatter.reduce`, this test fails and has to be
    deleted deliberately. That is the intent: it marks a decision, not a bug.
    """
    try:
        _C._aten_dispatch("aten.scatter.reduce")
    except NotImplementedError as exc:
        assert "aten.scatter.reduce" in str(exc), str(exc)
    else:
        raise AssertionError(
            "aten.scatter.reduce answers -- docs/SCATTER.md §2 says it should "
            "not exist yet; if it now does, this test has served its purpose "
            "and the doc needs updating with it"
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
