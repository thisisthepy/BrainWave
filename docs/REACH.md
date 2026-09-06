# REACH — the check that fails when a name and a kernel are not connected

`tools/golden/compare.py` is this repository's correctness number and it is
structurally blind to one question: **is the thing it just proved correct
reachable from Python at all?** It dispatches by key — `_aten_dispatch("aten.
roll.default", ...)` — using keys taken from its own case table, so a kernel it
compares 8509 times can still be unreachable, and a name in `overloads.json`
can still resolve to nothing. docs/GOLDEN.md names this class; this document is
the inventory of it on the current tree, and `tools/golden/reach.py` is the
check that keeps it at that inventory.

It has bitten four times, in three distinct shapes:

| shape | what it looks like from Python | how it was found before |
|---|---|---|
| 1. declared, no dispatch arm | `RuntimeError: aten op not implemented in torch._C shim` | `squeeze.default`, `squeeze.dims`, `where.ScalarSelf` — a human noticing (docs/DEMAND5.md) |
| 2. kernel, no spelling | nothing; the op is invisible from `torch.*` and `Tensor.*` | 22 names in docs/SPELLINGS.md, 6 more in its §9 — a sweep, months later |
| 3. spelling never exercised | works, or does not; no test would know | `torch.roll` (docs/DEMAND5.md) — golden compares by key and cannot see the door |

Each is a *different* failure and the check reports them separately, because
the fix for each is different: shape 1 is a missing arm in `aten.rs`, shape 2 is
a missing table entry or composite, shape 3 is a missing test.

---

## 1. Inventory, on develop `9f4557e` + this round

Measured by `tools/golden/reach.py` against this checkout's own artefact
(`TORCH_C_ARTEFACT` set — the warning `compare.py` prints when it is not is
there because the fallback path is shared by a dozen checkouts, and an
inventory taken from another agent's build is not an inventory).

```text
declared names: 166, declared keys: 287, implemented keys: 208

shape 1 (declared, no dispatch arm at all):            0
shape 1 partial (one overload of a live name is dead): 109 keys
shape 2 (kernel, no spelling):                         1     aten.alias.default
shape 3 (spelling never exercised by a test):          75
```

**Shape 1 is zero, and that is this round's result.** Every name in
`overloads.json` and `methods.json` now has at least one dispatch arm behind
it; the sweeps that closed `squeeze.default`, `squeeze.dims` and
`where.ScalarSelf` closed the shape entirely. The check is not confirming a
belief here — it was written expecting to find survivors of that class and
found none, which is why the ceiling below exists to keep it at zero rather
than a list to work through.

**Shape 2 is one**, and it is a correct gap: upstream exposes neither
`torch.alias` nor `Tensor.alias` (`aten::alias` is a dispatcher-internal view
op), so inventing a spelling for it would be the mistake docs/SPELLINGS.md
refuses on purpose. The allowlist entry stakes that reason on two `hasattr`
claims and the check puts them to a real upstream torch.

**Shape 3 is 75**, of which 20 are dunders — `__add__`, `__matmul__`,
`__invert__` and so on, exercised as syntax (`a + b`) that no textual search
can attribute back to the method name. The remaining 55 are a real backlog:
names whose kernels golden compares by dispatch key, with nothing anywhere
calling `torch.<name>(...)` or `t.<name>(...)`.

Four of those 55 were found by this check rather than by inspection, and they
are the reason it strips comments and docstrings before searching: `cat`,
`where`, `erf` and `sigmoid` each appear in `pytests/test_shim.py` **only inside
a comment or a docstring** —

    #  ... so that `torch.sigmoid(x, out=y)` refuses by the right name --
    step and then `torch.cat([self.keys, key_states], dim=-2)`

— and a first cut of the search counted those as coverage. It also counted a
comment *inside a road script* (docs/SPELLINGS.md §9's pattern: Python source
held in a triple-quoted string and run in a vendored-tree subprocess), which is
why the comment stripper is line-based rather than `tokenize`-based; to the
tokenizer a whole road script is one string. Deleting all three `torch.roll`
calls left the suite green until that was fixed, because the paragraph above
them explaining why they exist says `torch.roll(...)` too.

### 1.1 What this round did **not** do

The 55 backlog names are recorded, not fixed. `aten.rs` was another agent's
territory this round, and a check whose first act is to close every gap it
finds cannot be shown to fail. The deliverable is the failing check.

## 2. The 109 partially-dead overload keys

A name is *live* if any of its overloads has an arm; 109 declared keys belong
to a live name but have no arm of their own. Overwhelmingly these are `.out`
variants (`aten.sqrt.out`, `aten.zeros.out`, ...) plus a handful of unimplemented
forms (`aten.scatter.value_reduce`, `aten.sort.stable`, `aten.where.Scalar`).

These are **not** treated as failures, and the reason is in `overloads.json`'s
own README: order is the algorithm, `.out` variants bind only when the caller
passed `out=`, and their presence in the table is what makes `torch.sqrt(x)`
skip past them to the arm that exists. Declaring them is how the resolver
reproduces upstream's refusal wording rather than a wrong match.

They are still a slope, so the check holds them under a ceiling
(`shape1_dead_overload_keys_ceiling`, currently 109): the number may fall, and
implementing one of these arms is expected to lower it, but it may not grow.
A new declaration with nothing behind it fails here even when the name it is
attached to happens to work.

## 3. The check

`tools/golden/reach.py`, run three ways:

    python3 tools/golden/reach.py                  # inventory + verdict
    python3 tools/golden/reach.py --verify-upstream # + put the reasons to upstream
    # and in the suite, three tests in pytests/test_shim.py:
    #   test_reach_probe_tells_a_missing_arm_from_a_refused_call
    #   test_reach_every_declared_name_reaches_a_kernel_and_every_kernel_a_name
    #   test_reach_allowlist_reasons_are_answerable_by_upstream

**Ground truth is the running artefact, not the source.** Shape 1 is answered
by calling `_aten_dispatch(key)` with *no arguments* and reading which refusal
comes back: an arm that exists names a missing argument, an arm that does not
exist says `aten op not implemented in torch._C shim`. That distinction is the
whole check, so `self_test` asserts it in both directions against a known-live
op and a fabricated one — if the wording ever changes, the suite fails there
instead of turning into a silent all-clear. Declared names come from
`_shim_overloads`/`_shim_methods` (the parsed tables the artefact resolves
against) rather than from the JSON, so a table that failed to load cannot pass
an audit of a table that did.

**Cost:** no build, no upstream import on the default path, one zero-argument
dispatch per declared key (287 of them, all landing in an exception), and one
regex pass over `pytests/*.py`. This is deliberate: the check is only worth
anything if nobody has a reason to switch it off.

### 3.1 The allowlist, and why every entry can be questioned

`tools/golden/reach_allow.json`. Some gaps are correct — upstream has no
`torch.new_zeros` and no `Tensor.native_group_norm`, and docs/SPELLINGS.md
records three names deliberately left unspelled because inventing a door
upstream lacks is worse than the gap. So the check needs an allowlist; the
danger is that an allowlist is a blanket pass wearing a reason.

Two things keep it from becoming one.

* **Matched exactly, in both directions.** An unlisted gap fails, and an entry
  whose gap has since closed fails too. Adding a test that spells `torch.sort`
  turns the suite red until the `sort` entry is deleted. The file therefore
  describes what is missing *now*, and coverage arriving cannot quietly leave a
  stale excuse behind.
* **Reasons carry a question, not just prose.** An entry claiming "upstream has
  no such spelling" also carries the attribute paths it is staking that on
  (`upstream_absent: ["torch.alias", "torch.Tensor.alias"]`), and
  `--verify-upstream` puts them to a real upstream torch in a subprocess with
  `PYTHONPATH` stripped. If upstream grows the attribute, the reason stops being
  true and the check says so. When upstream torch is not importable the result
  is reported as a skip, never as a pass.

## 4. The demonstration — removing one connection turns it red

Each shape was broken on purpose, one at a time, and restored.

**Shape 3** — deleted the three `torch.roll(...)` / `r6.roll(2)` calls from
`pytests/test_shim.py` and changed nothing else:

```text
FAIL: shape 3: nothing in pytests/ calls `torch.roll(...)` or `.roll(...)`.
REACH: FAIL (5)
```

(five, because the same run is what first exposed `cat`, `erf`, `sigmoid` and
`where` as comment-only.)

**Shapes 1 and 2** — deleted `"roll"` from both `overloads.json` and
`methods.json`, added a `"reachdemo"` entry naming a schema no kernel
implements, and rebuilt:

```text
shape 1 (declared, no dispatch arm at all): 1
shape 2 (kernel, no spelling): 2
FAIL: shape 1: reachdemo is declared in overloads.json but none of its keys
      ['aten.reachdemo.default'] has a dispatch arm -- calling it raises
      'aten op not implemented'
FAIL: shape 2: aten.roll.default has a kernel and is compared by golden, but no
      `torch.<name>` / `Tensor.<name>` spelling and no composite in bootstrap.py
      reaches it -- it is invisible from Python.
REACH: FAIL (3)
```

That is exactly docs/DEMAND5.md's `torch.roll` gap and exactly the
`squeeze.default` gap, reproduced and caught by the suite rather than by a
person. Both tampers were reverted and the artefact rebuilt from the restored
tables before the gates below were taken.

## 5. What this check still cannot see

Written down because §5.5 of the house rules is that a verification which
cannot fail is not a verification, and the same applies to one whose limits are
not stated.

* **Spelling coverage is textual.** `.name(` matches a method call on anything,
  so a call to `list.sort()` would count as coverage for `sort`. It
  over-reports rather than under-reports on purpose: a false "covered" costs one
  allowlist entry, a false "uncovered" costs the check its welcome.
* **Dunders are invisible.** No search attributes `a + b` to `__add__`. Twenty
  names are allowlisted for this and the check cannot tell a covered operator
  from an uncovered one.
* **It proves reachability, not correctness.** Shape 1 answers "an arm exists",
  not "the arm is right" — that is `compare.py`'s job, and the two together are
  the point.
* **Composite reach is by name, not by call graph.** A kernel counts as spelled
  if its key appears as a string literal in `bootstrap.py`'s *code* (docstrings
  and comments excluded, via `ast`). A key sitting in dead code would count.
* **It sees `pytests/` only.** Coverage that lives anywhere else — a sample app,
  a device script — does not count, which is the conservative direction.

## 6. Gates

```text
392 ok                                   (389 before this round + 3 reach tests)
DOCWATCH: PASS -- 361/361 evaluated marker(s) hold  (350 before, + 11 in this file)
golden: 8509/8509 table entries matched upstream, ops covered 203
REACH: PASS
```

<!-- DOCWATCH: symbol-in-file tools/golden/reach.py has_dispatch_arm present -->
<!-- DOCWATCH: symbol-in-file tools/golden/reach.py executable_text present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_reach_every_declared_name_reaches_a_kernel_and_every_kernel_a_name present -->
<!-- DOCWATCH: hasattr alias false -->
<!-- DOCWATCH: op-implemented aten.alias.default -->
<!-- DOCWATCH: op-implemented aten.roll.default -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json roll present -->
<!-- DOCWATCH: json-key rust/torch_c/src/methods.json roll present -->
<!-- DOCWATCH: count smoke_ok ge 389 -->
<!-- DOCWATCH: count golden_cases_passed ge 8509 -->
<!-- DOCWATCH: count golden_ops_covered ge 203 -->
