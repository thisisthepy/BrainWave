"""Run the decomposition pass over every implemented non-core aten op.

This is where docs/graph/DECOMP.md §4's table comes from. It is a measurement script
and not a test: it prints a verdict per op and exits 0 whatever they are,
because "how many lower today" is a number that moves and a test that pinned it
would go red on progress. `test_shim.py` pins the individual ops that matter.

Run it inside the vendored tree, which is where the shim `_C` lives:

    PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
        python rust/torch_c/pytests/decomp_sweep.py

`--json` dumps the whole record, including each refusal's text, which is what
makes the "why is this one still shut" column of §4 derivable rather than
recalled.

## What the population is

`_aten_all_implemented()` minus Core ATen minus what capture refuses by name.
The last subtraction is not a convenience: docs/graph/CAPTURE.md §4 refuses mutation
and randomness with the op named, so those cannot appear in any trace. They are
not a decomposition problem, they are not a problem at all.

## Why the arguments are written out below

A decomposition has to be *run*, so each op needs operands that bind to its
schema. There is no way to derive them -- `aten.split.Tensor` needs a split
size that divides its input and `aten.isin.Tensor_Tensor` needs two tensors of
different lengths -- so they are written down, one case per op. An op with no
case is reported as `NO_CASE` rather than skipped, so that adding a kernel
without adding a case here is visible instead of quietly shrinking the
denominator.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback


#: docs/graph/CAPTURE.md §4. `is_mutating` (any op whose name ends in `_`) is derived
#: rather than listed; these are the ones named individually there. Mirrors
#: `capture.rs`'s `RANDOM` -- an op missing here would be swept as if it were
#: reproducible and would fail replay on every run.
RANDOM = frozenset({
    "aten.multinomial.default",
    "aten.randint.default",
    "aten.randint.low",
    "aten.randperm.default",
})


def _cases(torch):
    """`op -> (inputs, thunk)`. The thunk runs inside the recording."""
    d = torch._C._aten_dispatch

    def ones(*shape, scale=1.0):
        return torch.ones(*shape) * scale

    def mask():
        return d("aten.gt.Scalar", ones(3, 4, scale=0.5), 0.0)

    t34 = ones(3, 4)
    t44 = ones(4, 4)
    built: dict = {}

    def case(op, inputs, thunk):
        built[op] = (inputs, thunk)

    case("aten._safe_softmax.default", [t34], lambda: d("aten._safe_softmax.default", t34, -1))
    q, k, v = ones(1, 2, 3, 4), ones(1, 2, 3, 4), ones(1, 2, 3, 4)
    case("aten._scaled_dot_product_flash_attention_for_cpu.default", [q, k, v],
         lambda: d("aten._scaled_dot_product_flash_attention_for_cpu.default",
                   q, k, v, 0.0, False))
    case("aten._unsafe_view.default", [t34], lambda: d("aten._unsafe_view.default", t34, [4, 3]))
    case("aten.arange.default", [], lambda: d("aten.arange.default", 5))
    case("aten.arange.start", [], lambda: d("aten.arange.start", 1, 5))
    bb_c, bb_a, bb_b = ones(2, 3, 5), ones(2, 3, 4), ones(2, 4, 5)
    case("aten.baddbmm.default", [bb_c, bb_a, bb_b],
         lambda: d("aten.baddbmm.default", bb_c, bb_a, bb_b))
    case("aten.contiguous.default", [t34], lambda: d("aten.contiguous.default", t34))
    case("aten.detach.default", [t34], lambda: d("aten.detach.default", t34))
    case("aten.empty_like.default", [t34], lambda: d("aten.empty_like.default", t34))
    fd_a, fd_b = ones(3, 4, scale=7.0), ones(3, 4, scale=2.0)
    case("aten.floor_divide.default", [fd_a, fd_b],
         lambda: d("aten.floor_divide.default", fd_a, fd_b))
    case("aten.histc.default", [t34], lambda: d("aten.histc.default", t34))
    case("aten.is_floating_point.default", [t34],
         lambda: d("aten.is_floating_point.default", t34))
    isin_a, isin_b = ones(3, 4), ones(2)
    case("aten.isin.Tensor_Tensor", [isin_a, isin_b],
         lambda: d("aten.isin.Tensor_Tensor", isin_a, isin_b))
    case("aten.lift_fresh.default", [t34], lambda: d("aten.lift_fresh.default", t34))
    mf_mask = mask()
    case("aten.masked_fill.Scalar", [t34, mf_mask],
         lambda: d("aten.masked_fill.Scalar", t34, mf_mask, 0.0))
    mft_mask, mft_value = mask(), torch.ones(())
    case("aten.masked_fill.Tensor", [t34, mft_mask, mft_value],
         lambda: d("aten.masked_fill.Tensor", t34, mft_mask, mft_value))
    ms_mask = mask()
    case("aten.masked_select.default", [t34, ms_mask],
         lambda: d("aten.masked_select.default", t34, ms_mask))
    mm_a, mm_b = ones(3, 4), ones(4, 5)
    case("aten.matmul.default", [mm_a, mm_b], lambda: d("aten.matmul.default", mm_a, mm_b))
    case("aten.max.default", [t34], lambda: d("aten.max.default", t34))
    mo_a, mo_b = ones(3, 4), ones(3, 4, scale=2.0)
    case("aten.max.other", [mo_a, mo_b], lambda: d("aten.max.other", mo_a, mo_b))
    case("aten.min.default", [t34], lambda: d("aten.min.default", t34))
    case("aten.new_ones.default", [t34], lambda: d("aten.new_ones.default", t34, [2, 2]))
    case("aten.ones.default", [], lambda: d("aten.ones.default", [2, 3]))
    case("aten.reshape.default", [t34], lambda: d("aten.reshape.default", t34, [4, 3]))
    case("aten.rsub.Scalar", [t34], lambda: d("aten.rsub.Scalar", t34, 1.0))
    case("aten.silu.default", [t34], lambda: d("aten.silu.default", t34))
    case("aten.softplus.default", [t34], lambda: d("aten.softplus.default", t34))
    case("aten.split.Tensor", [t44], lambda: d("aten.split.Tensor", t44, 2))
    st_a, st_b = ones(3, 4), ones(3, 4, scale=2.0)
    case("aten.stack.default", [st_a, st_b], lambda: d("aten.stack.default", [st_a, st_b]))
    case("aten.sum.default", [t34], lambda: d("aten.sum.default", t34))
    case("aten.t.default", [t34], lambda: d("aten.t.default", t34))
    case("aten.transpose.int", [t34], lambda: d("aten.transpose.int", t34, 0, 1))
    case("aten.unbind.int", [t34], lambda: d("aten.unbind.int", t34, 0))
    case("aten.view.dtype", [t34], lambda: d("aten.view.dtype", t34, torch.int32))
    w_mask = mask()
    case("aten.where.ScalarOther", [w_mask, t34],
         lambda: d("aten.where.ScalarOther", w_mask, t34, 0.0))
    case("aten.zeros.default", [], lambda: d("aten.zeros.default", [2, 3]))
    case("aten.zeros_like.default", [t34], lambda: d("aten.zeros_like.default", t34))
    return built


def sweep() -> dict:
    import torch
    import torch._decomp as decomp

    from torchnative.export import (
        DecompositionRefused,
        decompose,
        decomposition_table,
        decomposition_table_source,
        is_core,
    )

    implemented = sorted(torch._C._aten_all_implemented())
    non_core = [op for op in implemented if not is_core(op)]
    mutating = [op for op in non_core if op.rsplit(".", 1)[0].endswith("_")]
    random = [op for op in non_core if op in RANDOM]
    population = [op for op in non_core if op not in mutating and op not in random]

    table = decomposition_table()
    cases = _cases(torch)
    results: dict = {}
    for op in population:
        built = cases.get(op)
        if built is None:
            results[op] = {"verdict": "NO_CASE"}
            continue
        inputs, thunk = built
        entry: dict = {"has_rule": op in table}
        try:
            torch._C._capture_begin(inputs)
            produced = thunk()
            reason = torch._C._capture_reason()
            if reason is not None:
                torch._C._capture_abandon()
                results[op] = dict(entry, verdict="CAPTURE_REFUSED", detail=reason)
                continue
            trace = torch._C._capture_end(produced)
        except BaseException as error:  # noqa: BLE001 -- reported, not raised
            try:
                torch._C._capture_abandon()
            except BaseException:  # noqa: BLE001
                pass
            results[op] = dict(entry, verdict="CAPTURE_RAISED",
                               detail=f"{type(error).__name__}: {error}")
            continue
        entry["recorded_ops"] = [node["op"] for node in trace.nodes]
        try:
            lowered = decompose(trace)
        except DecompositionRefused as error:
            entry.update(verdict="REFUSED", detail=str(error))
        except BaseException as error:  # noqa: BLE001
            entry.update(verdict="RAISED", detail="".join(
                traceback.format_exception_only(type(error), error)).strip())
        else:
            entry.update(verdict="LOWERED", ops_after=lowered.ops)
        results[op] = entry

    registry = decomp.global_decomposition_table["post_autograd"]
    return {
        "implemented": len(implemented),
        "core": len(implemented) - len(non_core),
        "non_core": len(non_core),
        "mutating": mutating,
        "random": random,
        "population": population,
        "table_source": list(decomposition_table_source()),
        "table_size": len(table),
        "registry": len(registry),
        "registry_default": sum(
            1 for key in registry if str(key).endswith(".default")),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="dump the whole record, refusal texts included")
    args = parser.parse_args()
    report = sweep()
    if args.json:
        json.dump(report, sys.stdout, indent=1)
        print()
        return 0

    print(f"implemented {report['implemented']}  core {report['core']}  "
          f"non-core {report['non_core']}")
    print(f"  capture refuses by name: {len(report['mutating'])} mutating, "
          f"{len(report['random'])} random")
    print(f"  population: {len(report['population'])}")
    source, reason = report["table_source"]
    print(f"table: {source} ({report['table_size']} rules)"
          + (f" -- fell back because {reason}" if reason else ""))
    print(f"registry: {report['registry']} entries, "
          f"{report['registry_default']} of them `.default`")
    print()
    counts: dict = {}
    for op, entry in sorted(report["results"].items()):
        counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1
        line = f"{entry['verdict']:16} {op}"
        if entry["verdict"] == "LOWERED":
            line += "  ->  " + ", ".join(entry["ops_after"])
        print(line)
        if "detail" in entry:
            detail = " ".join(entry["detail"].split())
            print(f"                 {detail[:200]}")
    print()
    for verdict, count in sorted(counts.items()):
        print(f"{count:3}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
