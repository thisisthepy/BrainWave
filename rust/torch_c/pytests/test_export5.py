"""`torch.export` — it works, and here is exactly how far that goes.

`docs/EXPORT4.md` §3 listed four claims and answered "not reached" to three of
them.  This round answers all four **yes on four hand-written modules** and
**no on every one of the ten real architectures upstream can export**.  Both
halves are the result; `docs/EXPORT5.md` is the measurement.

What these tests are shaped against
-----------------------------------
`docs/EXPORT.md` §4.2 predicted, in prose, an `ExportedProgram` that "would
print, would serialise" and contain **no operators**.  Halfway through this
round that is precisely what `torch.export.export()` returned, for a nameable
reason (§6 of the doc: the proxy mode lives on a pre-dispatch stack this shim's
door never read).  It did not look wrong.  So:

* nothing here asserts "did it export" without also counting `call_function`
  nodes.  `test_the_exported_graph_is_not_empty_which_is_the_failure_that_looks_right`
  is that check standing on its own, because it is the one a future round is
  most likely to lose;
* the three verdicts `exported` / `replayed` / `agreed` are three separate
  tests, never collapsed, and the agreement tolerance is **derived from
  upstream's own float32-vs-float64 error** (`docs/AGREE.md`'s method) rather
  than chosen;
* the meta storage handle is tested for what it **refuses** as much as for what
  it answers, because a handle that answered plausibly to everything would have
  satisfied `meta_utils.py` and lied to everyone else.

Every probe asserts `is_shim` before it asserts anything else — `docs/EXPORT4.md`
§1 lost hours to a worktree whose vendored tree had no `torch/__init__.py`, so
every measurement silently came from upstream torch 2.13.0.

`DOCWATCH` markers in `docs/EXPORT5.md` use `ge`, never `eq` on a shared global
count.
"""

import json
import math
import os
import subprocess
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

_EPS32 = 1.1920929e-07


def _available():
    return os.path.isfile(_VENDOR_SHIM)


# ---------------------------------------------------------------------------
# the probe -- run on BOTH sides, so every claim has upstream's answer beside it
# ---------------------------------------------------------------------------

_PROBE = r"""
import json, math, traceback
import torch
from torch import nn

out = {}
out["is_shim"] = hasattr(torch._C, "_aten_implemented")
# NOTE: no `torchnative.export.upstream.install()` anywhere in this probe.
# That is the point of docs/EXPORT5.md §7 -- before the hand-off, every export
# measurement in this repository was taken under a runtime monkey-patch, and
# without it export stopped at census name #0.


def attempt(fn):
    try:
        return {"status": "ok", "value": fn()}
    except BaseException as e:
        return {"status": "raise", "type": type(e).__name__, "message": str(e)[:400]}


# -- inputs built arithmetically, so both sides get identical bytes without
#    depending on either one's RNG --------------------------------------------
a64 = torch.tensor([math.sin(i * 0.7) * 3.0 for i in range(64)], dtype=torch.float64)
b64 = torch.tensor([math.cos(i * 0.31) * 2.0 for i in range(64)], dtype=torch.float64)


class Relu(nn.Module):
    def forward(self, x):
        return x.relu()


class AddTwo(nn.Module):
    def forward(self, x, y):
        return (x + y).relu()


class ScalarChain(nn.Module):
    def forward(self, x):
        return (x * 2 + 1).relu()


class Chain(nn.Module):
    def forward(self, x):
        return ((x * 3.5 - 0.25).relu() * x).sum()


CASES = [
    ("relu", Relu(), (a64,)),
    ("add_two", AddTwo(), (a64, b64)),
    ("scalar_chain", ScalarChain(), (a64,)),
    ("chain", Chain(), (a64,)),
]

cases = {}
for name, module, inputs64 in CASES:
    rec = {}
    inputs32 = tuple(t.to(torch.float32) for t in inputs64)
    rec["eager32"] = module(*inputs32).reshape(-1).tolist()
    rec["eager64"] = module(*inputs64).to(torch.float64).reshape(-1).tolist()
    got = attempt(lambda: torch.export.export(module, inputs32))
    if got["status"] != "ok":
        rec["exported"] = False
        rec["export_error"] = f"{got['type']}: {got['message']}"
    else:
        ep = got["value"]
        rec["exported"] = True
        rec["ops"] = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
        rec["n_call_function"] = len(rec["ops"])
        rec["n_placeholder"] = len([n for n in ep.graph.nodes if n.op == "placeholder"])
        # A lifted constant holding a fake tensor is how an empty graph
        # disguises itself -- docs/EXPORT5.md §6.
        rec["constants"] = sorted(getattr(ep, "constants", {}) or {})
        played = attempt(lambda: ep.module()(*inputs32))
        if played["status"] != "ok":
            rec["replayed"] = False
            rec["replay_error"] = f"{played['type']}: {played['message']}"
        else:
            rec["replayed"] = True
            rec["replay32"] = played["value"].reshape(-1).tolist()
    cases[name] = rec
out["cases"] = cases

if out["is_shim"]:
    # ---- the meta storage handle, docs/EXPORT5.md §2 -----------------------
    t = torch.empty((3, 4), dtype=torch.float32, device="meta")
    s = attempt(lambda: t.untyped_storage())
    if s["status"] == "ok":
        st = s["value"]
        out["meta_storage"] = {
            "nbytes": st.nbytes(),
            "size": st.size(),
            "len": len(st),
            "device": str(st.device),
            "data_ptr": st.data_ptr(),
            "cdata": st._cdata,
            "element_size": st.element_size(),
            "filled": st._shim_filled,
            "repr": repr(st),
        }
        out["meta_storage_doors"] = {
            "getitem": attempt(lambda: st[0]),
            "setitem": attempt(lambda: st.__setitem__(0, 3)),
            "copy_": attempt(lambda: st.copy_(st)),
            "resize_": attempt(lambda: st.resize_(8)),
            "bytes": attempt(lambda: st._shim_bytes()),
        }
        # identity: a view shares it, an unrelated meta tensor does not
        v = t.view(12)
        other = torch.empty((3, 4), dtype=torch.float32, device="meta")
        out["meta_identity"] = {
            "self_twice": t.untyped_storage()._cdata == t.untyped_storage()._cdata,
            "view_shares": v.untyped_storage()._cdata == t.untyped_storage()._cdata,
            "other_differs": other.untyped_storage()._cdata != t.untyped_storage()._cdata,
            "view_data_ptr": v.untyped_storage().data_ptr(),
        }
        # a dense tensor's storage is untouched by any of this
        d = torch.ones(3)
        ds = d.untyped_storage()
        out["dense_storage"] = {
            "device": str(ds.device),
            "filled": ds._shim_filled,
            "data_ptr_nonzero": ds.data_ptr() != 0,
            "nbytes": ds.nbytes(),
        }
    else:
        out["meta_storage_error"] = s

    # ---- the invariant the channels-last answer rests on -------------------
    out["is_contiguous"] = {
        "default": torch.ones(2, 3).is_contiguous(),
        "contiguous_format": torch.ones(2, 3).is_contiguous(
            memory_format=torch.contiguous_format),
        "preserve_format": torch.ones(2, 3).is_contiguous(
            memory_format=torch.preserve_format),
        "channels_last": torch.ones(2, 3, 4, 5).is_contiguous(
            memory_format=torch.channels_last),
    }
    out["is_contiguous_positional"] = attempt(
        lambda: torch.ones(3).is_contiguous(torch.contiguous_format))
    # Can this build make a channels-last tensor at all? The `False` above is a
    # fact only while it cannot. docs/EXPORT5.md §3.
    out["channels_last_constructible"] = attempt(
        lambda: torch.ones(2, 3, 4, 5).to(memory_format=torch.channels_last))

    # ---- the setters that refuse rather than lie to their own getters ------
    x = torch.ones(3)
    out["setters"] = {
        "conj_false": attempt(lambda: torch._C._set_conj(x, False)),
        "conj_true": attempt(lambda: torch._C._set_conj(x, True)),
        "neg_true": attempt(lambda: torch._C._set_neg(x, True)),
        "grad_dtype_same": attempt(lambda: setattr(x, "grad_dtype", torch.float32)),
        "grad_dtype_other": attempt(lambda: setattr(x, "grad_dtype", torch.float64)),
        "mkldnn_off": attempt(lambda: torch._C._set_mkldnn_enabled(False)),
        "mkldnn_on": attempt(lambda: torch._C._set_mkldnn_enabled(True)),
        "fp32_none": attempt(
            lambda: torch._C._set_fp32_precision_setter("mkldnn", "all", "none")),
        "fp32_tf32": attempt(
            lambda: torch._C._set_fp32_precision_setter("mkldnn", "all", "tf32")),
    }
    out["getters"] = {
        "grad_dtype": str(x.grad_dtype),
        "has_symbolic": x._has_symbolic_sizes_strides,
        "mkldnn": torch._C._get_mkldnn_enabled(),
        "onednn_tf32": torch._C._get_onednn_allow_tf32(),
        "fp32_precision": torch._C._get_fp32_precision_getter("mkldnn", "all"),
        "dispatch_key_set_cpu": torch._C._dispatch_key_set(torch.ones(3)),
        "dispatch_key_set_meta": torch._C._dispatch_key_set(
            torch.ones(3, device="meta")),
    }

    # ---- the guard that had stopped restoring, docs/EXPORT5.md §5 ----------
    seq = []
    torch._C._set_meta_in_tls_dispatch_include(False)
    seq.append(torch._C._meta_in_tls_dispatch_include())
    with torch._C._PreserveDispatchKeyGuard():
        torch._C._set_meta_in_tls_dispatch_include(True)
        seq.append(torch._C._meta_in_tls_dispatch_include())
    seq.append(torch._C._meta_in_tls_dispatch_include())
    out["preserve_guard_sequence"] = seq

    # nesting: the guard must restore to what it saw, not to a constant
    nested = []
    torch._C._set_meta_in_tls_dispatch_include(True)
    with torch._C._PreserveDispatchKeyGuard():
        nested.append(torch._C._meta_in_tls_dispatch_include())
        with torch._C._PreserveDispatchKeyGuard():
            torch._C._set_meta_in_tls_dispatch_include(False)
            nested.append(torch._C._meta_in_tls_dispatch_include())
        nested.append(torch._C._meta_in_tls_dispatch_include())
    nested.append(torch._C._meta_in_tls_dispatch_include())
    torch._C._set_meta_in_tls_dispatch_include(False)
    out["preserve_guard_nested"] = nested

    # ---- the hand-off, docs/EXPORT5.md §7 ----------------------------------
    from torchnative.export import upstream
    report = upstream.install()
    out["handoff"] = {
        "replaced": sorted(report.replaced),
        "rebound": report.rebound,
        "still_stubbed": sorted(
            n for n in upstream.installed_names()
            if upstream._is_stub(getattr(torch._C, n, None))
        ),
    }

# `default=repr` because `attempt()` keeps the live return value and some of
# those are tensors -- the point of recording them is only ever "did this
# answer or refuse, and with what message".
# `_functionality_to_backend_keys`, asked on BOTH sides. It is outside the
# `is_shim` block on purpose: the whole value of this record is the comparison,
# and a shim-only answer could only be checked against a transcription.
f2b = {}
for key_name in ("Dense", "Quantized", "Sparse", "SparseCsr", "NestedTensor",
                 "AutogradFunctionality", "CPU", "Meta"):
    key = getattr(torch._C.DispatchKey, key_name, None)
    if key is None:
        continue
    got = attempt(lambda key=key: [k.name for k in
                                   torch._C._functionality_to_backend_keys(key)])
    f2b[key_name] = got["value"] if got["status"] == "ok" else {"error": got}
out["functionality_to_backend_keys"] = f2b

# `default=repr` because `attempt()` keeps the live return value and some of
# those are tensors -- the point of recording them is only ever "did this
# answer or refuse, and with what message".
print(json.dumps(out, default=repr))
"""


def _run(script, shim):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _shim(_cache={}):
    if "v" not in _cache:
        _cache["v"] = _run(_PROBE, shim=True)
    assert _cache["v"]["is_shim"], "subprocess did not get the shim-backed torch"
    return _cache["v"]


def _upstream(_cache={}):
    if "v" not in _cache:
        _cache["v"] = _run(_PROBE, shim=False)
    assert not _cache["v"]["is_shim"], "the upstream probe got the shim"
    return _cache["v"]


# ---------------------------------------------------------------------------
# the tolerance -- derived from upstream, never chosen
# ---------------------------------------------------------------------------

def _rel(x, y):
    """`max|x-y| / max|y|` -- the scale-relative error docs/AGREE.md uses."""
    num = max(abs(p - q) for p, q in zip(x, y))
    den = max(abs(q) for q in y) or 1.0
    return num / den


def _derived_tolerance(up):
    """`docs/AGREE.md`'s method, re-run here rather than transcribed.

    For each case, upstream's **own** float32 answer is scored against its own
    float64 answer.  The tolerance is the p90 of that distribution, floored at
    8 ulp of float32.

    The floor is doing the work on this population and that is worth saying
    plainly: these four modules are numerically easy (p90 lands near 0.4 ulp),
    and without a floor the tolerance would be tighter than float32 arithmetic
    can be relied on to be.  `docs/AGREE.md` puts the floor there for exactly
    that case — "so that a population which happened to be numerically easy
    could not drive the tolerance below a few ulp".
    """
    errors = sorted(
        _rel(c["eager32"], c["eager64"]) for c in up["cases"].values()
    )
    p90 = errors[max(0, int(round(0.9 * (len(errors) - 1))))]
    return max(p90, 8 * _EPS32), p90, errors


def test_the_tolerance_is_derived_from_upstreams_own_error_and_not_chosen():
    """The number has to come from a measurement, or the agreement claim is empty.

    This is `docs/AGREE.md`'s rule and `docs/EXPORT4.md` §9's reason for keeping
    `agreed` as its own verdict.  Asserted here rather than trusted: the
    tolerance must equal the p90-floored-at-8-ulp of upstream's own
    float32-vs-float64 error, and every one of those errors must be a real
    positive number rather than a zero that would make any tolerance pass.
    """
    if not _available():
        return
    up = _upstream()
    tol, p90, errors = _derived_tolerance(up)
    assert len(errors) == 4, errors
    assert all(e > 0 for e in errors), (
        "upstream's float32 answer was bit-identical to its float64 answer on "
        f"some case, so it cannot serve as an oracle for the tolerance: {errors}"
    )
    assert tol == max(p90, 8 * _EPS32)
    assert 8 * _EPS32 <= tol <= 1e-5, tol


# ---------------------------------------------------------------------------
# the three verdicts, kept apart
# ---------------------------------------------------------------------------

def test_exported_torch_export_returns_an_exported_program():
    """Verdict 1 of 3, and **the weakest of the three.**

    It is separated from the other two because it is the one that can be true
    while the others are false without looking wrong at all — `docs/EXPORT.md`
    §4.2, and this round met that failure in the flesh (§6 of the doc).
    """
    if not _available():
        return
    shim = _shim()
    failed = {n: c.get("export_error") for n, c in shim["cases"].items()
              if not c.get("exported")}
    assert not failed, failed


def test_the_exported_graph_is_not_empty_which_is_the_failure_that_looks_right():
    """Verdict 1.5: an `ExportedProgram` holding no operators is not an export.

    **This test exists because that graph was really produced.** Before
    `docs/EXPORT5.md` §6, `torch.export.export()` on `x.relu()` returned:

        graph():
            %c_lifted_tensor_0 : [num_users=1] = placeholder[...]
            %x : [num_users=0] = placeholder[target=x]
            return (c_lifted_tensor_0,)

    One `call_function` node short of a computation, with the input unused and
    the answer folded into a lifted constant. It exported, it printed, it had a
    graph signature. `export_sweep.py` fails that as its own `export_empty`
    stage for the same reason.

    The constants check is the second half: a fake tensor in the constants list
    is how the empty graph announced itself, and `torch/export/_trace.py:2330`
    raises on it only when `error_on_lifted_constant_tensors` is left on.
    """
    if not _available():
        return
    shim = _shim()
    for name, c in shim["cases"].items():
        assert c.get("exported"), (name, c.get("export_error"))
        assert c["n_call_function"] > 0, (
            f"{name}: torch.export.export() returned an ExportedProgram whose "
            f"graph holds NO operators. docs/EXPORT.md §4.2 is this exact "
            f"failure; do not relax this test to 'did it export'."
        )
        assert not c["constants"], (
            f"{name}: the exported program lifted constants {c['constants']}. "
            f"That is how an untraced computation hides -- the answer is "
            f"folded into a constant and the input goes unused "
            f"(docs/EXPORT5.md §6)."
        )


def test_replayed_the_graph_the_program_holds_actually_runs():
    """Verdict 2 of 3. Separate from 1 because a graph can build and not run."""
    if not _available():
        return
    shim = _shim()
    failed = {n: c.get("replay_error") for n, c in shim["cases"].items()
              if not c.get("replayed")}
    assert not failed, failed


def test_agreed_the_replay_matches_upstream_element_wise_at_a_derived_tolerance():
    """Verdict 3 of 3, and **the headline.**

    The comparison is the shim's *replayed* output against **upstream's eager**
    output, element for element — not against the shim's own eager output,
    which would only prove that export and eager agree with each other and
    would pass on a shim where both were wrong the same way.

    Measured: all four are **bit-identical**, which is a stronger result than
    "within tolerance" and is asserted as such below, so that a future round
    which drifts into merely-inside-tolerance is noticed rather than absorbed.
    """
    if not _available():
        return
    shim, up = _shim(), _upstream()
    tol, _, _ = _derived_tolerance(up)
    for name, c in shim["cases"].items():
        assert c.get("replayed"), (name, c.get("replay_error"))
        theirs = up["cases"][name]["eager32"]
        got = _rel(c["replay32"], theirs)
        assert got <= tol, (
            f"{name}: shim's exported-and-replayed output differs from "
            f"upstream's eager output by {got:.3e}, over the derived tolerance "
            f"{tol:.3e}"
        )


def test_the_agreement_is_bit_identical_and_not_merely_inside_the_tolerance():
    """Recorded separately, because the two are different results.

    `docs/EXPORT4.md` §9 makes this point about `export_sweep.py` recording the
    worst deviation rather than a boolean: a boolean cannot tell bit-identical
    from just-inside-tolerance, and that difference is the whole question once
    decompositions are involved.  Here there are no decompositions yet and the
    answer is exact, so the exactness is what is pinned.
    """
    if not _available():
        return
    shim, up = _shim(), _upstream()
    for name, c in shim["cases"].items():
        assert c["replay32"] == up["cases"][name]["eager32"], (
            f"{name}: agreement is no longer bit-identical. That may be fine -- "
            f"check it against the derived tolerance -- but it is a change and "
            f"docs/EXPORT5.md §8 says it was exact."
        )


def test_the_shim_and_upstream_agree_on_which_operators_the_graph_holds():
    """Same operators, and the **overload** disagreement named rather than hidden.

    `docs/EXPORT.md` §5 found `capture.rs` recording `aten.mul.Scalar` where
    upstream dispatches `aten.mul.Tensor`; `docs/EXPORT4.md` §5 found the same
    difference reaching a `TorchDispatchMode`.  It now reaches the **exported
    graph**, which is the front end that matters most, because Core ATen and
    ExecuTorch's Edge dialect are defined per overload.

    So this asserts the operators agree and records that the overloads do not.
    It is deliberately not an `assert ops == ops`: that would fail for a known,
    documented reason and tempt someone to delete it.
    """
    if not _available():
        return
    shim, up = _shim(), _upstream()
    disagreements = {}
    for name, c in shim["cases"].items():
        theirs = up["cases"][name].get("ops")
        if theirs is None:
            continue
        mine = c["ops"]
        assert [o.rsplit(".", 1)[0] for o in mine] == \
               [o.rsplit(".", 1)[0] for o in theirs], (
            f"{name}: the shim's graph holds different OPERATORS from "
            f"upstream's, which is a real divergence and not the known "
            f"overload one: {mine} vs {theirs}"
        )
        if mine != theirs:
            disagreements[name] = (mine, theirs)
    # The known one, pinned so that closing it is noticed.
    assert "scalar_chain" in disagreements, (
        "the .Scalar/.Tensor overload disagreement (docs/EXPORT.md §5) no "
        "longer appears in the exported graph. If it was closed, say so in a "
        "doc and delete this assertion deliberately."
    )
    mine, theirs = disagreements["scalar_chain"]
    assert "aten.mul.Scalar" in mine and "aten.mul.Tensor" in theirs, (mine, theirs)


# ---------------------------------------------------------------------------
# §2 -- the meta storage handle
# ---------------------------------------------------------------------------

def test_a_meta_tensor_answers_a_storage_handle_with_a_size_and_no_bytes():
    """`docs/EXPORT.md` §3.3, closed.

    The three numbers that make it a handle rather than a buffer: the size is
    the tensor's real footprint, the device says meta, and `filled` is false and
    stays false -- which is what keeps `set_`'s guard (docs/CKPT.md §4) able to
    refuse it.
    """
    if not _available():
        return
    s = _shim()["meta_storage"]
    assert s["nbytes"] == 3 * 4 * 4, s          # 12 float32 elements
    assert s["size"] == s["nbytes"] and s["len"] == s["nbytes"], s
    assert s["device"] == "meta", s
    assert s["element_size"] == 1, s
    assert s["filled"] is False, "a meta storage claimed bytes had arrived"
    assert "unfilled" in s["repr"] and "meta" in s["repr"], s["repr"]


def test_the_meta_storage_data_ptr_is_zero_which_is_upstreams_own_answer():
    """`data_ptr()` carries no identity on a meta storage, upstream or here.

    Measured on torch 2.13.0: **every** meta storage answers `0` -- base, view,
    any size.  Answering the identity token here instead would have handed out
    a number that looks like an address and is a counter, and `torch/
    serialization.py` reads `data_ptr()` to tell storages apart.  Identity is
    asked for through `_cdata` and that is where it is answered.
    """
    if not _available():
        return
    s = _shim()["meta_storage"]
    assert s["data_ptr"] == 0, s["data_ptr"]
    assert s["cdata"] != 0, "a meta storage has no identity at all"
    assert _shim()["meta_identity"]["view_data_ptr"] == 0


def test_the_meta_storage_identity_is_shared_by_a_view_and_not_by_a_stranger():
    """What `meta_utils.py`'s aliasing memo actually asks of the handle.

    Upstream, `b.untyped_storage()._cdata == b[1:,1:].untyped_storage()._cdata`
    and differs from an unrelated meta tensor's -- measured.  Here the token
    lives on `Repr::Meta` and the meta `view` kernel propagates it, which is why
    `t` and `t.view(12)` answer the same and `torch.empty_like`-style fresh
    construction does not.

    **A fresh id in `view` would have been invisible**: the memo would simply
    have held two entries, export would still have run, and the aliasing it
    exists to preserve would have been silently lost.
    """
    if not _available():
        return
    ident = _shim()["meta_identity"]
    assert ident["self_twice"], "the same meta tensor gave two identities"
    assert ident["view_shares"], (
        "a meta tensor and its own view answered different storage identities, "
        "so meta_utils.py's storage_memo would treat them as unrelated storages"
    )
    assert ident["other_differs"], (
        "two unrelated meta tensors answered the SAME storage identity, which "
        "would make the memo alias tensors that share nothing"
    )


def test_every_door_that_would_need_bytes_refuses_on_a_meta_storage():
    """The handle is honest because of what it will not do.

    A storage that answered `bytes()` with `b""` would be saying "this storage
    is empty"; the truth is "this storage has 48 bytes and none of them exist".
    Those are different claims and the second one has to refuse.

    `resize_` is in the list and is a **divergence**: upstream's meta storage
    really does resize (measured).  It refuses here because `len` is derived
    from the meta tensor's shape and dtype, so a resize would leave the storage
    and the tensor disagreeing about a number the tensor owns.
    """
    if not _available():
        return
    doors = _shim()["meta_storage_doors"]
    for name in ("getitem", "setitem", "copy_", "resize_", "bytes"):
        d = doors[name]
        assert d["status"] == "raise", f"{name} answered on a meta storage: {d}"
    # upstream's own wording for the read, ours for the rest
    assert "meta" in doors["getitem"]["message"], doors["getitem"]
    for name in ("setitem", "copy_", "bytes"):
        assert "no bytes at all" in doors[name]["message"], (name, doors[name])
    assert "not independently settable" in doors["resize_"]["message"], doors["resize_"]


def test_the_dense_storage_path_is_untouched_by_the_meta_handle():
    """The narrowing has to be a *new arm*, not a change to the old one.

    A dense tensor's storage still reports cpu, still says its bytes arrived,
    and still answers a non-zero address.  Without this, making the meta branch
    work by loosening `snapshot` would pass every test above.
    """
    if not _available():
        return
    d = _shim()["dense_storage"]
    assert d["device"] == "cpu", d
    assert d["filled"] is True, d
    assert d["data_ptr_nonzero"], d
    assert d["nbytes"] == 12, d


# ---------------------------------------------------------------------------
# §3 -- is_contiguous(memory_format=...)
# ---------------------------------------------------------------------------

def test_is_contiguous_reads_memory_format_and_only_as_a_keyword():
    """Upstream refuses a positional memory format; so does this.

    The signature is measured, not guessed: `is_contiguous() takes 0 positional
    arguments but 1 was given` on 2.13.0.
    """
    if not _available():
        return
    c = _shim()["is_contiguous"]
    assert c["default"] is True
    assert c["contiguous_format"] is True
    assert c["preserve_format"] is True
    pos = _shim()["is_contiguous_positional"]
    assert pos["status"] == "raise", pos


def test_channels_last_is_false_as_a_fact_because_the_build_cannot_make_one():
    """The invariant under the answer, asserted rather than argued.

    `is_contiguous(memory_format=channels_last)` answers `False` here for every
    tensor, and that is only *true* while no tensor in this build can be in that
    layout.  So the test checks the premise: constructing a channels-last tensor
    must fail.  The day it succeeds, this test fails -- and that is the day the
    `False` above starts lying, which is exactly `docs/EXPORT4.md` §6.5's shape
    of argument for the meta `stride()`.
    """
    if not _available():
        return
    assert _shim()["is_contiguous"]["channels_last"] is False
    made = _shim()["channels_last_constructible"]
    assert made["status"] == "raise", (
        "this build constructed a channels-last tensor, so answering False to "
        "is_contiguous(memory_format=channels_last) is no longer a fact about "
        "the build -- it is now a wrong answer (docs/EXPORT5.md §3)"
    )


# ---------------------------------------------------------------------------
# §5 -- setters that refuse rather than lie to their own getters
# ---------------------------------------------------------------------------

def test_a_setter_that_cannot_take_effect_refuses_instead_of_accepting():
    """The shape this repository keeps meeting, met four more times.

    `_len_torch_dispatch_stack` answering a constant `0` (docs/EXPORT.md §2.2)
    and `no_dispatch()` suppressing nothing (docs/EXPORT4.md §5) are both "a
    call that entered, reported itself absent, and changed nothing".  A setter
    whose effect its own getter cannot see is the same failure.

    Each of these accepts the value the getter already reports -- so
    save/restore round trips do not raise -- and refuses every other value.
    """
    if not _available():
        return
    s = _shim()["setters"]
    for accepted in ("conj_false", "grad_dtype_same", "mkldnn_off", "fp32_none"):
        assert s[accepted]["status"] == "ok", (accepted, s[accepted])
    for refused in ("conj_true", "neg_true", "grad_dtype_other",
                    "mkldnn_on", "fp32_tf32"):
        assert s[refused]["status"] == "raise", (
            f"{refused} was accepted. Its getter cannot report the change, so "
            f"accepting it makes the setter invisible to its own effect "
            f"(docs/EXPORT5.md §5): {s[refused]}"
        )
        assert s[refused]["type"] == "NotImplementedError", s[refused]


def test_the_getters_behind_those_setters_answer_facts_about_this_build():
    """And two of them disagree with upstream **on purpose**.

    `_get_mkldnn_enabled()` is `True` on upstream 2.13.0 on this machine, where
    `torch.backends.mkldnn.is_available()` is `False` -- upstream's flag is a
    user preference that outlives the backend being absent.  Here it is `False`,
    because `_BUILD_FLAGS`' `_has_mkldnn` is `False` and there is nothing to
    prefer.  Transcribing upstream's answer would have claimed a backend.
    """
    if not _available():
        return
    g = _shim()["getters"]
    assert g["mkldnn"] is False, g
    assert g["onednn_tf32"] is None, (
        "upstream answers None -- not applicable -- on a build without oneDNN, "
        "and this build is in that position", g)
    assert g["fp32_precision"] == "none", g
    assert g["grad_dtype"] == "torch.float32", g
    assert g["has_symbolic"] is False, g


def test_dispatch_key_set_is_device_only_and_matches_upstreams_spelling():
    """Measured, not reasoned: dtype, rank, requires_grad and view-ness change nothing.

    Its only consumer compares two of these for equality
    (`fake_tensor.py:2083`), so a version keyed on `requires_grad` would have
    looked more careful and failed as a silent **cache miss** rather than an
    error.  Note that meta carries no Autocast key while CPU does -- the
    asymmetry a hand-written table would most likely have smoothed over.
    """
    if not _available():
        return
    g = _shim()["getters"]
    assert g["dispatch_key_set_cpu"] == (
        "DispatchKeySet(CPU, ADInplaceOrView, AutogradCPU, AutocastCPU)"), g
    assert g["dispatch_key_set_meta"] == (
        "DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)"), g
    assert g["dispatch_key_set_cpu"] != g["dispatch_key_set_meta"]


def test_functionality_to_backend_keys_matches_upstream_key_for_key():
    """**This test exists because its nullification went uncaught.**

    `docs/EXPORT5.md` §11 records it: replacing the whole body with
    `return []` broke nothing.  Every suite stayed green and
    `torch.export.export()` still worked, because the only caller is
    `torch/utils/_python_dispatch.py::_push_mode`, which uses the result to
    *uncache* per-dispatch-key handlers:

        ks = torch._C._functionality_to_backend_keys(k)
        for op in get_cached_ops():
            for key in ks:
                op._uncache_dispatch(key)

    With `[]` nothing is uncached.  Today `get_cached_ops()` is empty in this
    shim so the loop has no body either way — which is precisely why it was
    invisible.  It is a **cache invalidation that quietly invalidates nothing**,
    the same "entered and changed nothing" shape as
    `_len_torch_dispatch_stack`'s constant `0` (docs/EXPORT.md §2.2), and it
    would surface as a stale handler rather than as an error.

    So the answer is checked against **upstream's**, key for key, rather than
    against the list this shim computes from its own enum.  A transcription
    would have gone stale silently; deriving it on both sides and diffing means
    a torch upgrade that renumbers or renames a backend fails here.
    """
    if not _available():
        return
    shim, up = _shim(), _upstream()
    mine, theirs = shim["functionality_to_backend_keys"], up["functionality_to_backend_keys"]
    # The shim's `DispatchKey` enum is narrower than upstream's -- `Quantized`
    # is absent, measured. That is a pre-existing narrowing and not this
    # round's, so it is *recorded* rather than asserted away: the comparison
    # runs over the keys both sides have, and the difference is pinned so it
    # cannot grow without someone noticing.
    absent = sorted(set(theirs) - set(mine))
    assert absent == ["Quantized"], (
        "the set of DispatchKey members this shim lacks changed. That is a "
        f"narrowing of the dispatch-key enum and wants a decision: {absent}"
    )
    # **A second, narrower gap that this comparison found**, and it is in the
    # enum rather than in the function: the shim's `DispatchKey` has the bare
    # `MAIA` member but none of its per-functionality spellings
    # (`AutogradMAIA`, `SparseMAIA`, `QuantizedMAIA`, `NestedTensorMAIA`), while
    # `MTIA` has all of them. So every list below is upstream's minus its MAIA
    # entry. That is pre-existing -- `_functionality_to_backend_keys` is derived
    # from the enum and reports exactly what the enum holds -- and it is pinned
    # here rather than smoothed over, because the *difference* is the thing that
    # must not grow. docs/EXPORT5.md §11.
    #
    # A transcribed table would have listed `AutogradMAIA` and answered a key
    # the enum does not have; deriving from the enum and diffing against
    # upstream is what turned that into a visible, bounded gap.
    for key in sorted(set(mine) & set(theirs)):
        expected = [k for k in theirs[key] if "MAIA" not in k or k == "MAIA"]
        assert mine[key] == expected, (
            f"{key}: shim answers {mine[key]!r}, upstream (minus the MAIA "
            f"spellings this shim's enum lacks) is {expected!r}"
        )
    missing_maia = sorted(
        set(theirs["AutogradFunctionality"]) - set(mine["AutogradFunctionality"]))
    assert missing_maia == ["AutogradMAIA"], (
        "the set of per-functionality DispatchKey members this shim's enum "
        f"lacks changed: {missing_maia}"
    )
    # And the shape that `return []` would satisfy, named so it cannot come back:
    assert len(theirs["Dense"]) == 16, theirs["Dense"]
    assert mine["Dense"], (
        "the functionality keys answered an empty list, which is the "
        "nullification docs/EXPORT5.md §11 found uncaught"
    )


# ---------------------------------------------------------------------------
# §5 -- the guard that had stopped restoring
# ---------------------------------------------------------------------------

def test_preserve_dispatch_key_guard_actually_restores_what_it_saved():
    """The defect, as a test. `docs/EXPORT5.md` §5.

    `_PreserveDispatchKeyGuard`'s one-line description has always read "saves
    and restores the whole TLS key state", and until this round it saved and
    restored **nothing** -- it was a counter, like the rest of the family.

    That was invisible until export went deep enough, because upstream
    *delegates* a restore to it: `fake_tensor.py::in_kernel_invocation_manager`
    sets `_set_meta_in_tls_dispatch_include(True)` inside the guard and leaves
    the matching reset **commented out**, naming the guard as the thing that
    undoes it.  With the guard inert the flag latched `True` and the next entry
    died on that function's own `assert meta_in_tls == prev_in_kernel`.

    The sequence is asserted, not the final value: a guard that reset the flag
    to a constant `False` on exit would pass a final-value check and would be
    wrong the moment it was nested.
    """
    if not _available():
        return
    assert _shim()["preserve_guard_sequence"] == [False, True, False], \
        _shim()["preserve_guard_sequence"]


def test_the_guard_restores_to_what_it_saw_rather_than_to_a_constant():
    """Nesting is what separates a real save/restore from a reset-to-False.

    Outer sees `True`, inner sets `False`, inner exit must give back `True`,
    outer exit must give back `True` again.  A guard that cleared the flag on
    exit passes the un-nested test above and fails this one.
    """
    if not _available():
        return
    assert _shim()["preserve_guard_nested"] == [True, False, True, True], \
        _shim()["preserve_guard_nested"]


# ---------------------------------------------------------------------------
# §7 -- the hand-off
# ---------------------------------------------------------------------------

def test_the_census_names_are_present_with_no_monkey_patch_at_all():
    """`docs/EXPORT.md` §8's hand-off, paid -- and this is how it is checked.

    Every measurement in the probe above runs **without** calling
    `torchnative.export.upstream.install()`.  Before this round that was not
    possible: `export_sweep.py` against the shim stopped at census name #0 on
    all 40 architectures for exactly that reason, and `docs/EXPORT4.md` §1.1
    recorded that every export number in the repository was taken under the
    patch.

    The assertion is on `replaced` being **empty**, which is the inverse of the
    test this replaced in `test_export.py`.  Empty means every name was already
    an implementation when `torch` finished importing.
    """
    if not _available():
        return
    h = _shim()["handoff"]
    assert h["still_stubbed"] == [], (
        "these census names are still placeholders after import, so the "
        f"bootstrap hand-off does not cover them: {h['still_stubbed']}"
    )
    assert h["replaced"] == [], (
        "install() found placeholders to replace, so the names are NOT coming "
        f"from bootstrap.py and the hand-off has regressed: {h['replaced']}"
    )
    assert h["rebound"] == 0, (
        "the rebind pass repointed bindings, which means something was patched "
        "after `import torch` -- the exact cost docs/EXPORT.md §8 moved this "
        f"code to remove: {h['rebound']}"
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
