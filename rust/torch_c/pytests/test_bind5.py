"""docs/bindings/BIND5.md -- one argument form across three architectures, and
`torch.export`'s step two.

  1. `Tensor.new_zeros((..., 0-dim int Tensor, ...))` -- `led` and
     `longformer`'s shared wall (docs/kernels/STRIDED.md §6). The same argument form
     docs/bindings/BIND4.md §3 landed for `torch.zeros`, arriving at a *method* this
     time. `chunks_count` in the borrowed
     `_sliding_chunks_query_key_matmul` is a 0-dim int64 Tensor, confirmed by
     instrumenting the real call rather than read off the source.

  1b. **The refusals, which is the half BIND4 shipped wrong.** `int(item)`
     applied to every Tensor element made this build accept a FLOAT 0-dim
     Tensor (`torch.zeros((2, tensor(3.5)))` gave a (2, 3) tensor) and a BOOL
     one, both of which upstream refuses -- docs/bindings/ARGFORM.md's "more permissive
     than the thing it replaces". Nothing could see it because nothing
     asserted it. Every one of upstream's three lines is asserted below,
     against a live upstream, **including the exception type**, because
     upstream reaches the `bool` refusal one layer further in than the others
     and answers `RuntimeError` where the rest answer `TypeError`.

  2. `torch._C._NodeBase` / `_NodeIter` / `_fx_map_arg` / `_fx_map_aggregate`
     -- docs/graph/EXPORT.md §6 item **2**, taken only because item 1 (the
     dispatcher entrance) landed first in `aten.rs` (docs/design/DISPATCH3.md).
     `torch.fx.Graph()` now constructs. The `_sort_key` arithmetic is the
     part a plausible implementation gets wrong and is separated here by
     inserting nodes in the MIDDLE of a graph, where a monotonic counter --
     which reproduces the right list order -- gives the wrong key.

  3. `fastspeech2_conformer` and `sam3_lite_text_text_model` were checked and
     found NOT actionable from this file, each for a measured reason rather
     than an assumed one (§6, §7 of the doc).

Every case goes through the spelling a user writes and is compared against a
**live upstream torch in a separate process** (`env -u PYTHONPATH -u
TORCH_USE_RTLD_GLOBAL`), element-wise, shape and dtype included. Golden
compares at the dispatch key and is structurally blind to whether any Python
spelling reaches it.

Nothing here needs numpy or a network.
"""

import json
import os
import re
import subprocess
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")
_BOOTSTRAP = os.path.join(_REPO_ROOT, "rust", "torch_c", "src", "bootstrap.py")


_PROBE_SCRIPT = r"""
import json, sys, collections
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


# ===========================================================================
# 1. Tensor.new_zeros with a 0-dim int Tensor in the size tuple
# ===========================================================================
base = torch.randn(2, 3)
n3 = torch.tensor(3)

# `led`/`longformer`'s literal spelling: three plain ints and `chunks_count`,
# a 0-dim int64 Tensor, in position 2 of a four-element tuple.
batch_x_heads, chunks_count, window = 2, torch.tensor(3), 4
rec("new_zeros_led_spelling", lambda: base.new_zeros(
    (batch_x_heads, chunks_count + 1, window, window * 2 + 1)))

rec("new_zeros_tensor_in_tuple", lambda: base.new_zeros((2, n3, 4)))
rec("new_zeros_tensor_in_list", lambda: base.new_zeros([2, n3, 4]))
rec("new_zeros_tensor_only_element", lambda: base.new_zeros((n3,)))
rec("new_zeros_every_element_a_tensor",
    lambda: base.new_zeros((torch.tensor(2), n3)))
rec("new_zeros_tensor_with_kwargs", lambda: base.new_zeros(
    (2, n3), dtype=torch.int64, device="cpu"))
rec("new_zeros_zero_sized_tensor", lambda: base.new_zeros((2, torch.tensor(0))))

# forms that already worked, held so the wrapper is transparent to them
rec("new_zeros_plain_tuple", lambda: base.new_zeros((2, 3)))
rec("new_zeros_plain_list", lambda: base.new_zeros([2, 3]))
rec("new_zeros_varargs", lambda: base.new_zeros(2, 3))
rec("new_zeros_torch_size", lambda: base.new_zeros(torch.Size([2, 3])))
rec("new_zeros_empty_tuple", lambda: base.new_zeros(()))
rec("new_zeros_dtype_kwarg", lambda: base.new_zeros((2, 3), dtype=torch.int64))

# every integral dtype upstream takes here.  `int8` is absent on purpose: this
# shim cannot CONSTRUCT an int8 tensor at all (candle storage), so the probe
# would be measuring `torch.tensor`, not the size-list rule.
for d in ("uint8", "int16", "int32", "int64"):
    rec("new_zeros_dtype_%s" % d, lambda d=d: base.new_zeros(
        (2, torch.tensor(3, dtype=getattr(torch, d)))))
# one element, but not 0-dim -- upstream keys on numel, not ndim
rec("new_zeros_one_element_1d", lambda: base.new_zeros((2, torch.tensor([3]))))
rec("new_zeros_one_element_2d", lambda: base.new_zeros((2, torch.tensor([[3]]))))

# --- the refusals -----------------------------------------------------------
rec("new_zeros_float_tensor_refused",
    lambda: base.new_zeros((2, torch.tensor(3.5))))
rec("new_zeros_whole_float_tensor_refused",
    lambda: base.new_zeros((2, torch.tensor(3.0))))
rec("new_zeros_bool_tensor_refused",
    lambda: base.new_zeros((2, torch.tensor(True))))
rec("new_zeros_multi_element_tensor_refused",
    lambda: base.new_zeros((2, torch.tensor([3, 4]))))
rec("zeros_float_tensor_refused", lambda: torch.zeros((2, torch.tensor(3.5))))
rec("zeros_whole_float_tensor_refused",
    lambda: torch.zeros((2, torch.tensor(3.0))))
rec("zeros_bool_tensor_refused", lambda: torch.zeros((2, torch.tensor(True))))
rec("zeros_multi_element_tensor_refused",
    lambda: torch.zeros((2, torch.tensor([3, 4]))))
# negative: both refuse, and the exception TYPE differs (the kernel's, not the
# parser's).  Held as "both refuse" only -- see the docstring on the test.
rec("zeros_negative_tensor_refused", lambda: torch.zeros((2, torch.tensor(-1))))

# `torch.zeros`'s own accepted forms, unaffected by the tightening
rec("zeros_tensor_in_tuple", lambda: torch.zeros((2, n3, 5)))
rec("zeros_plain_tuple", lambda: torch.zeros((2, 3, 5), dtype=torch.float32))
rec("zeros_varargs", lambda: torch.zeros(2, 3))

# --- the device-context question, for the METHOD ---------------------------
# docs/bindings/BIND4.md §3.1: a wrapper that skips to the inner table-driven closure
# detaches `DeviceContext`, which matches by object identity.  A tensor method
# is not among `_device_constructors()`'s 36 names, so the answer here should
# be the RECEIVER's device either way -- asserted rather than assumed.
def _meta_method():
    with torch.device("meta"):
        return str(base.new_zeros((2, n3)).device)
rec("new_zeros_inside_device_context", _meta_method)

def _meta_factory():
    with torch.device("meta"):
        return str(torch.zeros(2).device)
rec("zeros_inside_device_context", _meta_factory)


# ===========================================================================
# 2. torch.fx.Graph / _NodeBase
# ===========================================================================
import torch.fx

def _graph_facts():
    g = torch.fx.Graph()
    root = g._root
    facts = {
        "root_op": root.op, "root_name": root.name, "root_target": root.target,
        "root_sort_key": list(root._sort_key), "root_self_linked":
            root._next is root and root._prev is root,
        "root_erased": root._erased, "root_meta": root.meta,
        "root_args": list(root._args), "root_type": root.type,
    }
    a = g.placeholder("a")
    b = g.placeholder("b")
    n = g.call_function(torch.ops.aten.mul.Tensor, (a, 2))
    g.output(n)
    facts["names"] = [x.name for x in g.nodes]
    facts["ops"] = [x.op for x in g.nodes]
    facts["sort_keys"] = [list(x._sort_key) for x in g.nodes]
    facts["users_of_a"] = [x.name for x in a.users]
    facts["input_nodes_of_n"] = [x.name for x in n._input_nodes]
    facts["n_graph_is_g"] = n.graph is g
    facts["printout"] = str(g).strip()
    facts["len"] = len(g.nodes)

    # insertion in the MIDDLE -- where a monotonic counter separates from the
    # real rule, in both directions and at both ends
    with g.inserting_before(b):
        mid = g.call_function(torch.relu, (a,))
    with g.inserting_before(mid):
        g.call_function(torch.relu, (a,))
    with g.inserting_before(a):
        # a CONSTANT argument, not `a` -- this node lands before `a` is
        # defined, and `g.lint()` below rejects a use that precedes its def
        g.call_function(torch.relu, (1,))
    with g.inserting_after(n):
        g.call_function(torch.relu, (a,))
    facts["sort_keys_after_inserts"] = [list(x._sort_key) for x in g.nodes]
    facts["sorted_by_key_matches_list"] = (
        [x.name for x in sorted(g.nodes, key=lambda x: x._sort_key)]
        == [x.name for x in g.nodes])
    facts["lt_holds_along_the_list"] = all(
        p < q for p, q in zip(list(g.nodes), list(g.nodes)[1:]))
    facts["reversed"] = [x.name for x in reversed(g.nodes)]

    n.replace_all_uses_with(a)
    facts["args_after_replace"] = [
        [getattr(v, "name", v) for v in x.args] for x in g.nodes]
    g.erase_node(n)
    facts["names_after_erase"] = [x.name for x in g.nodes]
    g.lint()
    facts["lint_ok"] = True
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    facts["code"] = gm.code.strip()
    return facts

rec("fx_graph", _graph_facts)


def _node_round_trip():
    # A Node added to a graph, read back through every member.
    g = torch.fx.Graph()
    a = g.placeholder("a")
    n = g.call_function(torch.relu, ([a, {"k": a}],), {"z": (a, 1)})
    facts = {
        "args_type": type(n.args).__name__,
        "nested_list_type": type(n.args[0]).__name__,
        "nested_dict_type": type(n.args[0][1]).__name__,
        "kwargs_type": type(n._kwargs).__name__,
        "input_nodes": [x.name for x in n._input_nodes],
        "users_of_a": [x.name for x in a.users],
        "name": n.name, "op": n.op,
        # `str(target)` is a function repr and differs by build, not by
        # behaviour -- identity against the same name is what is meant
        "target_is_torch_relu": n.target is torch.relu,
        "type": n.type, "repr_fn": n._repr_fn, "erased": n._erased,
        "meta": n.meta, "next": n._next.name, "prev": n._prev.name,
        "all_input_nodes": [x.name for x in n.all_input_nodes],
    }
    # rewiring drops the old user, which is the part a naive setter loses
    b = g.placeholder("b")
    n.args = (b,)
    facts["after_rewire_input_nodes"] = [x.name for x in n._input_nodes]
    facts["after_rewire_users_of_a"] = [x.name for x in a.users]
    facts["after_rewire_users_of_b"] = [x.name for x in b.users]
    n._replace_input_with(b, a)
    facts["after_replace_input_args"] = [x.name for x in n.args]
    facts["after_replace_users_of_b"] = [x.name for x in b.users]
    return facts

rec("fx_node_round_trip", _node_round_trip)


def _map_facts():
    g = torch.fx.Graph()
    a = g.placeholder("a")
    NT = collections.namedtuple("NT", "x y")
    return {
        "map_arg_nested": str(torch._C._fx_map_arg(
            [a, {"k": a}, (a, 1)], lambda n: n.name)),
        "map_arg_slice": str(torch._C._fx_map_arg(
            slice(a, None, a), lambda n: n.name)),
        "map_arg_set_is_a_leaf": str(torch._C._fx_map_arg({a}, lambda n: n.name)),
        "map_agg_leaves": str(torch._C._fx_map_aggregate(
            [1, (2, "s")], lambda x: str(x))),
        "map_agg_namedtuple": repr(torch._C._fx_map_aggregate(
            NT(1, 2), lambda x: x * 2)),
        "map_agg_str_is_a_leaf": repr(torch._C._fx_map_aggregate(
            "str", lambda x: x + "!")),
        "map_agg_list_type": type(torch._C._fx_map_aggregate(
            [1], lambda x: x)).__name__,
        "map_agg_dict_type": type(torch._C._fx_map_aggregate(
            {"a": 1}, lambda x: x)).__name__,
        "map_agg_tuple_type": type(torch._C._fx_map_aggregate(
            (1,), lambda x: x)).__name__,
    }

rec("fx_map", _map_facts)


def _iter_facts():
    g = torch.fx.Graph()
    a = g.placeholder("a")
    b = g.placeholder("b")
    c = g.placeholder("c")
    before = [x.name for x in g.nodes]
    # STILL LINKED, only flagged -- graph.py:1619's "iterators may retain
    # handles to erased nodes"
    b._erased = True
    after = [x.name for x in g.nodes]
    b._erased = False
    # _remove_from_list does not self-link: the removed node still points at
    # its former neighbours
    c._remove_from_list()
    return {
        "before": before, "with_one_flagged_erased": after,
        "after_remove_names": [x.name for x in g.nodes],
        "removed_prev": c._prev.name, "removed_next_is_root": c._next is g._root,
        "iter_forward": [x.name for x in torch._C._NodeIter(g._root, False)],
        "iter_reversed": [x.name for x in torch._C._NodeIter(g._root, True)],
    }

rec("fx_iter", _iter_facts)


# ===========================================================================
# 3. a single-element integral Tensor in a SCALAR int/SymInt position
#    -- `vilt`'s wall, torch.multinomial(Tensor, Tensor)
# ===========================================================================
# A one-hot weight, so `multinomial` is deterministic and the VALUES can be
# compared and not only the shape.
hot = torch.tensor([0.0, 0.0, 1.0, 0.0])
rec("multinomial_int", lambda: torch.multinomial(hot, 2, replacement=True))
rec("multinomial_tensor", lambda: torch.multinomial(hot, torch.tensor(2),
                                                    replacement=True))
rec("multinomial_tensor_kwarg",
    lambda: torch.multinomial(hot, num_samples=torch.tensor(2),
                              replacement=True))
rec("multinomial_tensor_1d",
    lambda: torch.multinomial(hot, torch.tensor([2]), replacement=True))
rec("multinomial_tensor_int32",
    lambda: torch.multinomial(hot, torch.tensor(2, dtype=torch.int32),
                              replacement=True))
rec("multinomial_float_tensor_refused",
    lambda: torch.multinomial(hot, torch.tensor(2.0), replacement=True))
rec("multinomial_bool_tensor_refused",
    lambda: torch.multinomial(hot, torch.tensor(True), replacement=True))
rec("multinomial_multi_element_refused",
    lambda: torch.multinomial(hot, torch.tensor([2, 3]), replacement=True))

# the same rule on ops that are not multinomial -- it is upstream's PARSER,
# not one binding
grid = torch.arange(12, dtype=torch.float32).reshape(3, 4)
rec("select_int", lambda: grid.select(0, 1))
rec("select_tensor", lambda: grid.select(0, torch.tensor(1)))
rec("select_tensor_2d_one_element",
    lambda: grid.select(0, torch.tensor([[1]])))
rec("select_tensor_negative", lambda: grid.select(0, torch.tensor(-1)))
rec("select_tensor_uint8",
    lambda: grid.select(0, torch.tensor(1, dtype=torch.uint8)))
rec("transpose_tensor", lambda: grid.transpose(torch.tensor(0), 1))
rec("unsqueeze_tensor", lambda: grid.unsqueeze(torch.tensor(0)))
rec("select_float_tensor_refused", lambda: grid.select(0, torch.tensor(1.0)))
rec("select_bool_tensor_refused", lambda: grid.select(0, torch.tensor(True)))
rec("select_multi_element_refused",
    lambda: grid.select(0, torch.tensor([1, 2])))
rec("select_empty_tensor_refused",
    lambda: grid.select(0, torch.tensor([], dtype=torch.int64)))

# NOT opened: a Tensor as an ELEMENT of a sized int LIST (`int[1]? dim`).
# Upstream accepts this; this build refuses it.  See the test.
rec("sum_dim_tensor_in_a_list_position", lambda: grid.sum(dim=torch.tensor(0)))


json.dump(out, sys.stdout)
"""


_cache = {}


def _run(side):
    """`side` is `"shim"` or `"upstream"`. `None` on the shim side when the
    vendored shim is not installed, matching `test_bind4.py`: `run.sh` builds
    the standalone `_C`, and only `vendor/install_shim.sh` writes the one the
    vendored tree loads."""
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
    up = _run("upstream")
    assert name in up, f"{name} is not in the upstream probe output"
    return s[name], up[name]


def _agree(name, atol=1e-6, rtol=1e-6):
    pair = _both(name)
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, f"{name}: upstream itself refused: {want}"
    assert "raised" not in got, (
        f"{name}: the shim refused where upstream computed: {got}"
    )
    assert got.get("shape") == want.get("shape"), f"{name}: shape {got} != {want}"
    assert got.get("dtype") == want.get("dtype"), f"{name}: dtype {got} != {want}"
    gv, wv = got["ok"], want["ok"]
    if isinstance(gv, (str, bool)) or isinstance(wv, (str, bool)):
        assert gv == wv, f"{name}: {gv!r} != {wv!r}"
        return
    assert len(gv) == len(wv), f"{name}: length {len(gv)} != {len(wv)}"
    for a, b in zip(gv, wv):
        assert abs(a - b) <= atol + rtol * abs(b), f"{name}: {a} != {b}"


def _both_refuse(name, same_type=True):
    """Both sides refuse. `same_type=True` also demands the same exception
    CLASS, which is what separates "we happen to fail here too" from "we
    reproduce upstream's rule"."""
    pair = _both(name)
    if pair == "skip":
        return
    got, want = pair
    assert "raised" in want, f"{name}: upstream did NOT refuse: {want}"
    assert "raised" in got, (
        f"{name}: upstream refuses and the shim computed something instead: {got}"
    )
    if same_type:
        assert got["raised"] == want["raised"], (
            f"{name}: upstream raises {want['raised']} ({want['msg'][:120]!r}) "
            f"and this shim raises {got['raised']} ({got['msg'][:120]!r})"
        )
    return got, want


def _facts(name):
    pair = _both(name)
    if pair == "skip":
        return None, None
    got, want = pair
    assert "raised" not in want, f"{name}: upstream itself refused: {want}"
    assert "raised" not in got, f"{name}: the shim refused: {got}"
    return got["ok"], want["ok"]


def _same_facts(name, keys):
    got, want = _facts(name)
    if got is None:
        return
    for k in keys:
        assert k in want, f"{name}: upstream probe has no fact {k!r}"
        assert got[k] == want[k], (
            f"{name}.{k}: shim {got[k]!r} != upstream {want[k]!r}"
        )


# ===========================================================================
# 1. Tensor.new_zeros -- the form led and longformer stop on
# ===========================================================================


def test_new_zeros_led_and_longformer_spelling_now_computes():
    """`_sliding_chunks_query_key_matmul`'s literal call, borrowed verbatim by
    `modeling_led.py:440` and `modeling_longformer.py:796`: four elements, one
    of them a 0-dim int64 Tensor.  Confirmed to be a Tensor by instrumenting
    the real model, not read off the source (docs/bindings/BIND5.md §1)."""
    _agree("new_zeros_led_spelling")


def test_new_zeros_tensor_in_size_agrees_in_every_container():
    _agree("new_zeros_tensor_in_tuple")
    _agree("new_zeros_tensor_in_list")
    _agree("new_zeros_tensor_only_element")
    _agree("new_zeros_every_element_a_tensor")
    _agree("new_zeros_zero_sized_tensor")


def test_new_zeros_tensor_in_size_carries_dtype_and_device():
    """The keyword arguments must survive the wrapper -- it forwards `*args`
    and `**kwargs`, and a version that ate them would still pass every
    shape-only assertion above."""
    _agree("new_zeros_tensor_with_kwargs")


def test_new_zeros_plain_forms_are_unaffected():
    """Five spellings that worked before this round.  The wrapper sits in
    front of all of them."""
    for n in ("new_zeros_plain_tuple", "new_zeros_plain_list",
              "new_zeros_varargs", "new_zeros_torch_size",
              "new_zeros_empty_tuple", "new_zeros_dtype_kwarg"):
        _agree(n)


def test_new_zeros_accepts_every_integral_dtype_upstream_does():
    """The rule is dtype-keyed, and `int64` is the only dtype the two
    architectures actually produce -- so the others are the ones that would
    silently not be covered."""
    for d in ("uint8", "int16", "int32", "int64"):
        _agree("new_zeros_dtype_%s" % d)


def test_new_zeros_keys_on_element_count_not_on_ndim():
    """`tensor([3])` and `tensor([[3]])` are accepted upstream: the rule is
    "one element", not "0-dim".  A `dim() == 0` check passes every other test
    in this file."""
    _agree("new_zeros_one_element_1d")
    _agree("new_zeros_one_element_2d")


# --- the refusals -----------------------------------------------------------


def test_a_float_tensor_in_a_size_list_is_refused_as_upstream_refuses_it():
    """**The defect docs/bindings/BIND4.md §3 shipped.** `int(item)` took `tensor(3.5)`
    to 3 and built a (2, 3) tensor; upstream raises.  Asserted for a
    whole-valued float too, because "it had no fractional part" is the
    plausible excuse for accepting it and upstream does not make it."""
    _both_refuse("new_zeros_float_tensor_refused")
    _both_refuse("new_zeros_whole_float_tensor_refused")
    _both_refuse("zeros_float_tensor_refused")
    _both_refuse("zeros_whole_float_tensor_refused")


def test_a_bool_tensor_is_refused_with_upstreams_own_exception_type():
    """`tensor(True)` used to become 1.  Upstream refuses it, and refuses it
    with a `RuntimeError` where the float and multi-element cases give a
    `TypeError` -- it gets past the unpack and dies at the scalar conversion.
    `_both_refuse` compares the class, so a single blanket `TypeError` here
    would fail."""
    _both_refuse("new_zeros_bool_tensor_refused")
    _both_refuse("zeros_bool_tensor_refused")


def test_a_multi_element_tensor_is_refused():
    _both_refuse("new_zeros_multi_element_tensor_refused")
    _both_refuse("zeros_multi_element_tensor_refused")


def test_a_negative_size_tensor_refuses_but_not_with_upstreams_class():
    """Both refuse; the classes differ, and that is recorded rather than
    papered over.  Upstream says `RuntimeError: zeros: Dimension size must be
    non-negative`; here the value reaches the kernel and the unsigned
    conversion says `OverflowError`.  Inventing a matching message in the
    argument parser would be a second surface claiming to own a rule the
    kernel already enforces, so the assertion is the weaker one and this
    docstring is why."""
    got, want = _both_refuse("zeros_negative_tensor_refused", same_type=False) \
        or (None, None)
    if got is None:
        return
    assert want["raised"] == "RuntimeError", want
    assert got["raised"] == "OverflowError", got


def test_zeros_own_accepted_forms_survive_the_tightening():
    """The three `torch.zeros` spellings docs/bindings/BIND4.md §3 landed still work --
    the refusals above are a narrowing of that change, not a revert of it."""
    _agree("zeros_tensor_in_tuple")
    _agree("zeros_plain_tuple")
    _agree("zeros_varargs")


def test_the_wrappers_are_transparent_to_a_device_context():
    """docs/bindings/BIND4.md §3.1's trap, asked of both wrappers.

    `DeviceContext.__torch_function__` matches its constructors by OBJECT
    IDENTITY off the `torch` module, so a wrapper that skips to the inner
    table-driven closure silently detaches it.  `torch.zeros` reproduces the
    `_MODE_STACK` guard for that reason; `Tensor.new_zeros` does not need to,
    because a bound method is not among `_device_constructors()`'s names --
    and this asserts that difference against upstream rather than reasoning
    about it, in both directions."""
    _agree("zeros_inside_device_context")
    _agree("new_zeros_inside_device_context")


def test_the_size_list_rule_is_installed_at_exactly_two_call_sites():
    """docs/bindings/ARGFORM.md §2's scoping, as source structure.

    Upstream takes a Tensor in EVERY `SymInt[]` position -- `new_ones`,
    `new_empty`, `new_full`, `ones`, `empty`, `view` were all measured to
    accept it (docs/bindings/BIND5.md §1).  This build installs it for the two
    spellings three architectures actually write, and folding it into
    `_TypeChecker` would open all of them at once with no matching
    measurement for each.  Pinned here so that generalising it is a
    deliberate edit to this count and not a quiet one."""
    src = open(_BOOTSTRAP).read()
    calls = re.findall(r"_coerce_symint_size_tensors\(module, \"([a-z_]+)\"", src)
    assert sorted(calls) == ["new_zeros", "zeros"], calls


# ===========================================================================
# 2. torch.fx.Graph and torch._C._NodeBase
# ===========================================================================


def test_fx_graph_constructs_and_its_root_node_matches_upstreams():
    """docs/graph/EXPORT.md §4.1 in four lines: an empty `Graph` builds a sentinel
    root `Node`, so the failure was at construction and not at the first
    operator.  Every member of that root is compared, because it is the one
    node no user ever writes and therefore the one whose defaults nothing
    else would check."""
    _same_facts("fx_graph", [
        "root_op", "root_name", "root_target", "root_sort_key",
        "root_self_linked", "root_erased", "root_meta", "root_args",
        "root_type",
    ])


def test_a_graph_of_nodes_round_trips_exactly_as_upstream_builds_it():
    """Names, opcodes, use lists, and the printed graph -- the last one being
    the check that reads every member at once."""
    _same_facts("fx_graph", [
        "names", "ops", "sort_keys", "users_of_a", "input_nodes_of_n",
        "n_graph_is_g", "printout", "len",
    ])


def test_the_sort_key_survives_insertion_in_the_middle():
    """**The separator.**  A monotonically increasing counter reproduces the
    right list order and the right `_sort_key` for an append-only graph, and
    gives the wrong key the moment anything is inserted between two nodes.
    Four insertions here -- before a middle node, before that one, before the
    first node and after the last -- exercise all three branches of upstream's
    rule in both directions, and the keys are compared to upstream's
    element-wise rather than only checked for monotonicity."""
    _same_facts("fx_graph", [
        "sort_keys_after_inserts", "sorted_by_key_matches_list",
        "lt_holds_along_the_list", "reversed",
    ])


def test_erase_and_replace_all_uses_leave_the_graph_upstream_leaves():
    _same_facts("fx_graph", [
        "args_after_replace", "names_after_erase", "lint_ok", "code",
    ])


def test_a_node_added_to_a_graph_round_trips_through_every_member():
    """Including the container conversions, which are asymmetric and measured
    rather than chosen: a top-level `args` tuple stays a `tuple`, a nested
    list becomes `immutable_list`, and the top-level `kwargs` dict becomes
    `immutable_dict`."""
    _same_facts("fx_node_round_trip", [
        "args_type", "nested_list_type", "nested_dict_type", "kwargs_type",
        "input_nodes", "users_of_a", "name", "op", "target_is_torch_relu",
        "type",
        "repr_fn", "erased", "meta", "next", "prev", "all_input_nodes",
    ])


def test_rewiring_a_node_drops_it_from_the_old_inputs_user_list():
    """The half a naive `_update_args_kwargs` loses.  Setting `args` to a
    different node must remove this node from the OLD input's `users`, or
    every dead-code pass in `fx` reads a stale use count and keeps nodes
    nothing uses."""
    _same_facts("fx_node_round_trip", [
        "after_rewire_input_nodes", "after_rewire_users_of_a",
        "after_rewire_users_of_b", "after_replace_input_args",
        "after_replace_users_of_b",
    ])


def test_fx_map_arg_and_map_aggregate_agree_with_upstream_on_every_shape():
    """Nested aggregates, `slice`, namedtuples (rebuilt positionally, not from
    an iterable), a `set` which is a LEAF rather than an aggregate, and a
    `str` which is a leaf too -- each one a place a reimplementation diverges
    quietly."""
    _same_facts("fx_map", [
        "map_arg_nested", "map_arg_slice", "map_arg_set_is_a_leaf",
        "map_agg_leaves", "map_agg_namedtuple", "map_agg_str_is_a_leaf",
        "map_agg_list_type", "map_agg_dict_type", "map_agg_tuple_type",
    ])


def test_the_node_iterator_skips_erased_nodes_without_unlinking_them():
    """`graph.py:1619` sets `_erased` AFTER `_remove_from_list()` and says why
    ("iterators may retain handles to erased nodes").  The flag alone must be
    enough to hide a node that is still linked, and `_remove_from_list` must
    NOT self-link -- a removed node still points at its former neighbours,
    which is what makes a retained handle able to walk on."""
    _same_facts("fx_iter", [
        "before", "with_one_flagged_erased", "after_remove_names",
        "removed_prev", "removed_next_is_root", "iter_forward",
        "iter_reversed",
    ])


# ===========================================================================
# 3. a single-element integral Tensor in a scalar int/SymInt position
# ===========================================================================


def test_multinomial_takes_a_tensor_num_samples_which_is_vilts_wall():
    """`vilt` stops at `torch.multinomial(probs, num_samples)` with a Tensor
    `num_samples`.  Not a missing kernel and not a schema gap --
    `multinomial.default` exists and its schema matches upstream's -- but
    upstream's ARGUMENT PARSER, which here is `_TypeChecker`.

    The weights are one-hot so the RESULT is deterministic and this compares
    values rather than only a shape."""
    _agree("multinomial_int")
    _agree("multinomial_tensor")
    _agree("multinomial_tensor_1d")
    _agree("multinomial_tensor_int32")


def test_the_positional_and_keyword_spellings_of_it_agree():
    """The two coercion sites, held against each other through the two
    spellings that reach them.  `resolve` has one for positionals and one for
    keywords, and the generated fast path is a THIRD -- which reproduced only
    the sized-int-list coercion and handed the raw Tensor to the dispatcher
    for this one (docs/bindings/BIND5.md §7.2).  The Rust side unpacked it anyway, so
    the divergence came back as a plausible answer rather than as an error."""
    _agree("multinomial_tensor")
    _agree("multinomial_tensor_kwarg")


def test_the_rule_is_upstreams_parser_and_not_one_binding():
    """Measured on five more ops before being written table-wide: `select`,
    `transpose` and `unsqueeze` here, plus `narrow` and `repeat_interleave`
    upstream (both blocked in this shim for unrelated reasons).  A
    multinomial-only wrapper would have passed every assertion above."""
    _agree("select_int")
    _agree("select_tensor")
    _agree("select_tensor_2d_one_element")
    _agree("select_tensor_negative")
    _agree("select_tensor_uint8")
    _agree("transpose_tensor")
    _agree("unsqueeze_tensor")


def test_a_float_or_multi_element_tensor_in_a_scalar_int_position_is_refused():
    """The asymmetry that makes this an argument-form fix rather than a
    widening: upstream refuses both, so this build must too."""
    _both_refuse("multinomial_float_tensor_refused")
    _both_refuse("multinomial_multi_element_refused")
    _both_refuse("select_float_tensor_refused")
    _both_refuse("select_multi_element_refused")
    _both_refuse("select_empty_tensor_refused")


def test_a_bool_tensor_in_a_scalar_int_position_raises_upstreams_class():
    """`RuntimeError`, not `TypeError`, and the difference is structural:
    upstream's `ParameterType::INT64` check accepts an integral tensor
    *including* bool, and the unpack then asserts `scalar.isIntegral(false)`.
    So the predicate has to say yes and the coercion has to say no, in that
    order -- which is how it is written here.  A predicate that simply
    excluded `bool` would give `TypeError` and fail this."""
    _both_refuse("multinomial_bool_tensor_refused")
    _both_refuse("select_bool_tensor_refused")


def test_a_tensor_inside_a_sized_int_LIST_is_still_refused_here():
    """**An absence asserted on purpose, to be inverted when it is closed.**

    `grid.sum(dim=tensor(0))` works upstream: a `int[1]? dim` position takes
    the bare value through torch's "if a size is specified we also allow
    passing a single int" rule, and that rule sees the same parser check.
    This round opened SCALAR `int`/`SymInt` positions only, because that is
    what was measured and what `vilt` needs; the list positions are
    `_coerce_symint_size_tensors`'s two measured spellings and nothing more
    (docs/bindings/BIND5.md §1.2, §7.3).

    So this build is NARROWER than upstream here, deliberately, and the test
    says so in both directions -- if a later round opens it, this fails and
    gets inverted rather than quietly passing."""
    pair = _both("sum_dim_tensor_in_a_list_position")
    if pair == "skip":
        return
    got, want = pair
    assert "raised" not in want, (
        "upstream stopped accepting a Tensor in a sized int-list position; "
        f"this test's premise is gone: {want}"
    )
    assert "raised" in got, (
        "a Tensor in a sized int-list position now works here.  That is a "
        "widening this round scoped out; invert this test and record the "
        "measurement rather than deleting it"
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
