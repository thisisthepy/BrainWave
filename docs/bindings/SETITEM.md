# `__setitem__` is one form, not thirteen — the stepped write

docs/architectures/ARCH100.md §2 ranks `TensorBase.__setitem__` first among the operators that block the 82
architectures its 297-model sweep could not run: **13 architectures stop there, more than any other
single name.** `x[i] = v` is upstream's most overloaded surface — int, slice, ellipsis, `None`,
bool mask, integer tensor, and tuples mixing all of them — so the round was scoped as "find out
which forms the 13 actually use, implement those, refuse the rest by name."

The answer turned out to be much narrower than the scoping allowed for, and that is the result.

Environment: worktree `work/setitem` on develop `eb84708`, torch 2.13.0 upstream
(`/Volumes/macMini/caches/spike-venv/bin/python`). Every number below was measured — the shim in a
vendored-tree process, upstream in a separate process with `PYTHONPATH` and
`TORCH_USE_RTLD_GLOBAL` unset.

---

## 1. Which index forms the 13 architectures use

**One.** All 13 stop on a slice with `step != 1` on the *left* of an assignment, and none of them
reaches any other `__setitem__` refusal. This was read off `arch_sweep.py --only <the 13>`, whose
tracebacks name the call site; it was not inferred from the models' source.

```text
form                                   step  arch  stage         where
x[:, 0::2]        = <matrix>              2     3  CONSTRUCTION  *ConformerRelPositionalEmbedding
x[..., k::3]      = <rank-3 tensor>       3    10  forward       apply_interleaved_mrope
```

The three construction failures — `fastspeech2_conformer`, `seamless_m4t`, `wav2vec2-conformer` —
are the same line in three copies of the conformer relative-positional-encoding code:

```python
pe_positive[:, 0::2] = torch.sin(position * div_term)     # modeling_seamless_m4t.py:305
                                                          # modeling_wav2vec2_conformer.py:186
                                                          # modeling_fastspeech2_conformer.py:741
```

The other ten — `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_moe_text`, `qwen3_vl`,
`qwen3_vl_text`, `qwen3_vl_moe`, `qwen3_vl_moe_text`, `cosmos3_omni`, `minicpmv4_6` — are one line
in `apply_interleaved_mrope`, reached through six model files but written once:

```python
freqs_t[..., idx] = freqs[dim, ..., idx]      # idx = slice(offset, None, 3)
```

So docs/architectures/ARCH100.md's own caution is stronger than it put it. It noted that the 13 are "dominated by
one family, which is worth knowing before it is ranked as thirteen independent wins"; the sweep says
the 13 are **two source lines**, and that ten of them share one.

**Nothing else about `__setitem__` was implemented, because nothing else was demanded.** Bool masks,
integer-tensor indices, `None`, ellipsis, negative integer indices, and the `copy_`/`fill_` shape
rule were already there (docs/kernels/VIEWS.md §6); mixed basic-and-advanced indexing is still refused by
name and no swept architecture asks for it.

### 1.1 Why it was a refusal and not a wrong answer

`aten.slice.Tensor` aliases its input at step 1 and **materialises above it** — measured in
docs/kernels/VIEWS.md §4 with candle's `same_storage` as an oracle: `slice.Tensor(x, 0, 5, 2)` holds an
independent buffer, because candle has no public constructor for a stepped view and the kernel
reaches step > 1 through `index_select`. The `__getitem__` walk read backwards would therefore
narrow to a tensor that shares nothing with the receiver, and the write would return normally and
change nothing. `__setitem__` refused by name rather than doing that, which is why the sweep could
classify all 13 in the first place.

That refusal was correct and it is the reason the fix below does not go through the walk.

---

## 2. The patch

**The translation is Python-level and lives in `bootstrap.py`, which was another round's file.** So
it is written here to be applied rather than applied here. It was applied, built, and measured in
this worktree (§3, §4) and then reverted; nothing in `rust/torch_c/src/` changed this round.

The lowering: **a stepped slice is a set of positions, and `aten.index_put_.default` already writes
a set of positions through the receiver's own storage.** It needed no kernel change — §5 is the
evidence that it already answers for both shapes.

Insert into `TensorBase.__setitem__`, immediately after

```python
        index = _index_tuple(index)
        index = _expand_ellipsis(self, index)
```

and before the `if any(isinstance(item, tensorbase) ...)` advanced-index arm:

```python
        # A step != 1 slice on the *write* side, lowered to `index_put_`.
        #
        # `aten.slice.Tensor` reaches a step above 1 through `index_select`,
        # which materialises (docs/kernels/VIEWS.md §6.4), so the basic walk below
        # would narrow to a tensor that does not share storage with the
        # receiver and the write would be silently lost. The positions the
        # slice names are handed to `index_put_` as an integer index instead,
        # and `index_put_` writes through the receiver's own buffer.
        #
        # Restricted, on purpose, to **one** stepped slice with every other
        # item a full slice: `index_put_` implements a single index group, and
        # a second one has to be refused rather than approximated. That covers
        # every call site the 297-architecture sweep found (docs/bindings/SETITEM.md
        # §1) -- `pe[:, 0::2] = v` and `freqs_t[..., k::3] = v`.
        #
        # The dtype cast is not incidental. Upstream reaches this through
        # `copy_`, which casts (measured: `int64 x; x[0::2] = [1.7, 2.7, 3.7]`
        # gives `[1, 0, 2, 0, 3, 0]`), while `index_put_` requires the dtypes
        # to match exactly. Without the cast this path would refuse a write
        # upstream performs.
        _stepped = [
            k
            for k, item in enumerate(index)
            if isinstance(item, slice) and item.step is not None and item.step != 1
        ]
        if _stepped:
            if len(_stepped) > 1 or any(
                not (
                    _is_full_slice(item)
                    or (isinstance(item, slice) and item.step not in (None, 1))
                )
                for item in index
            ):
                raise NotImplementedError(
                    "not implemented in torch._C shim: TensorBase.__setitem__ with a "
                    "stepped slice alongside another non-trivial index -- the write "
                    "lowers to aten.index_put_, which implements a single index "
                    "group, and a second group is refused rather than approximated "
                    "(docs/bindings/SETITEM.md §3)"
                )
            axis = _stepped[0]
            item = index[axis]
            if item.step <= 0:
                # torch's own wording, from aten.slice.Tensor.
                raise ValueError(f"step must be greater than zero, got {item.step}")
            positions = list(range(*item.indices(self.shape[axis])))
            source = _lift(value, self)
            if source.dtype != self.dtype:
                source = dispatch("aten._to_copy.default", source, dtype=self.dtype)
            if not positions:
                # Upstream writes nothing here but still checks the broadcast,
                # and raises with `copy_`'s wording rather than
                # `index_put_`'s. Not reproduced: an empty stepped write is
                # not a form any swept architecture uses (docs/bindings/SETITEM.md §4).
                return
            dispatch(
                "aten.index_put_.default",
                self,
                [None] * axis + [_lift_sequence_index(positions)],
                source,
                False,
            )
            return

```

Also delete the now-unreachable `step != 1` refusal further down the same method — the arm

```python
                step = 1 if item.step is None else item.step
                if step != 1:
                    raise NotImplementedError(
                        "not implemented in torch._C shim: TensorBase.__setitem__ with "
                        f"a step-{step} slice -- aten.slice.Tensor reaches a step above "
                        ...
```

may be left in place (nothing reaches it) or reduced to `step = 1 if item.step is None else
item.step`. It was left in place for the measurements below, so that a stepped index that somehow
escaped the new arm would still refuse rather than write into a copy.

### 2.1 What to run after applying it

`bootstrap.py` is compiled into the artefact, so the change does not reach anything until both the
crate and the vendored shim are rebuilt:

```sh
export CARGO_TARGET_DIR=...; export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
cd rust/torch_c && cargo build --release
cd - && bash vendor/install_shim.sh
```

Then **three golden cases flip from `expect="c_error"` to `expect="match"`** and `compare.py` says
so by name (`gap appears CLOSED: both sides now succeed, promote this case to expect=match`):

```text
member x[0:4:2] = 0.0 [refused -- a step > 1 slice is not a view here]
member x[:, 0::2] = matrix [the conformer positional-encoding write]
member x[..., 1::3] = 3-D src [the interleaved-mrope write]
```

The first was already there; the other two were added this round, in `_setitem_member_cases`, with
their upstream values in the `note` so the promotion is a one-line edit per case. They are three
rather than one because **a lowering that handled a 1-D stepped write and not a stepped write at a
later axis would close the first and leave the 13 exactly where they are** — which is the failure
the two new cases exist to catch.

`rust/torch_c/pytests/test_setitem.py` does not need editing. Every one of its stepped-write tests
is written two-sided: upstream's exact values if the write happens, a refusal that names itself and
mutates nothing if it does not. §6 says why that shape and not a pinned refusal.

---

## 3. What the 13 do once the wall falls

Re-run of `arch_sweep.py --only <the 13>` with the patch applied and both artefacts rebuilt:

```text
                          before                        after
cosmos3_omni              __setitem__  (forward)        FORWARD RUNS
qwen3_vl                  __setitem__  (forward)        FORWARD RUNS
qwen3_vl_moe              __setitem__  (forward)        FORWARD RUNS
qwen3_vl_moe_text         __setitem__  (forward)        FORWARD RUNS
qwen3_vl_text             __setitem__  (forward)        FORWARD RUNS

minicpmv4_6               __setitem__  (forward)        aten.matmul.default  MatMulUnexpectedStriding
qwen3_5                   __setitem__  (forward)        aten.matmul.default  MatMulUnexpectedStriding
qwen3_5_text              __setitem__  (forward)        aten.matmul.default  MatMulUnexpectedStriding
qwen3_5_moe               __setitem__  (forward)        aten.matmul.default  MatMulUnexpectedStriding
qwen3_5_moe_text          __setitem__  (forward)        aten.matmul.default  MatMulUnexpectedStriding

fastspeech2_conformer     __setitem__  (CONSTRUCTION)   TensorBase.view_as        (now forward)
wav2vec2-conformer        __setitem__  (CONSTRUCTION)   TensorBase.view_as        (now forward)
seamless_m4t              __setitem__  (CONSTRUCTION)   TensorBase.index_select   (now forward)
```

**All 13 clear this wall. Five run a complete forward. The three that could not previously be
built now build**, which is the part docs/architectures/ARCH100.md §4 says is worth separating: "cannot even be
built" is a far weaker result than "builds and stops at one operator", and three architectures moved
across that line.

The three second walls, and what they are worth:

* **`TensorBase.view_as`** (2 arch) is a spelling gap of the docs/architectures/DEMAND8.md §2.1 kind — one method
  on `PyTensorBase`, no kernel, since `view` is already there. It is not in docs/architectures/ARCH100.md's
  ranked list at all, because nothing reached it before.
* **`TensorBase.index_select`** (1 arch here) is already **rank 4** in docs/architectures/ARCH100.md, blocking 5
  other architectures. `seamless_m4t` joins that group rather than forming a new one, so
  `index_select` is now worth 6.
* **`aten.matmul.default` / `MatMulUnexpectedStriding`** (5 arch) is not a missing operator. It is
  the `backend_limitation` class docs/architectures/ARCH100.md §2.1 kept separate — candle refusing a
  non-contiguous lhs inside `torch_chunk_gated_delta_rule`'s
  `k_beta @ key.transpose(-1, -2)`. A `contiguous()` before the matmul is a plausible fix and it is
  a different kind of work from binding a name.

**None of the three is caused by the lowering.** Each is reached from a line that has nothing to do
with subscript assignment, and each names its own call site.

Read against docs/architectures/ARCH100.md's roadmap: the head of the ranked list is worth less than 13 and more
than it looks. Less, because ten of the thirteen were one line and five of them land on one shared
backend limitation immediately; more, because closing it made `view_as` visible, moved
`index_select` from 5 to 6, and turned three construction failures into forward failures.

---

## 4. Semantics, measured rather than reasoned

Each row was run on upstream in its own process, then on the patched shim, and compared element for
element. They are the tests in `pytests/test_setitem.py`.

| question | upstream 2.13.0 | patched shim |
|---|---|---|
| `float32 x; x[0::2] = tensor` | `[1,0,2,0,3,0]` | same |
| **dtype**: `int64 x; x[0::2] = [1.7,2.7,3.7]` | `[1,0,2,0,3,0]` — **casts, truncating** | same |
| `float32 x; x[0::2] = [1,2,3]` (int64 src) | `[1.0,0,2.0,0,3.0,0]` | same |
| **scalar rhs**: `x[1::2] = 5.0` | `[0,5,0,5,0,5]` | same |
| **broadcast**: `(3,6) x; x[:, 0::2] = (1,3)` | the row, three times | same |
| **negative bound**: `x[-6::2] = [1,2,3]` | `[1,0,2,0,3,0]` | same |
| **negative step**: `x[::-1] = v` | `ValueError: step must be greater than zero` | same class, same text |
| view taken before the write sees it | yes | yes |

Two findings are worth stating on their own, because they are the ones a reasoned implementation
gets wrong:

**The dtype rule is the destination's, and it casts.** `index_put_` refuses a dtype mismatch with
upstream's own wording — and that refusal is *right* for the calls that really are `index_put_`
(`x[t] = v`, measured in docs/kernels/VIEWS.md §2). But a stepped write is not one of those upstream; it is
`slice` + `copy_`, and `copy_` casts. Routing the stepped write to `index_put_` without casting
first would refuse a write upstream performs, which is the direction that stops an architecture.
The `_to_copy` in the patch is that bridge and nothing else.

**Negative slice bounds resolve; they are not the index-wrap question.** docs/architectures/DEMAND8.md warns by
name that `index_add_`'s indices do not wrap while `index_put_`'s do, and that the neighbouring op
is not evidence. Here the negative number is a slice *bound*, resolved by
`slice.indices(extent)` before any index exists, so it is a third rule again — measured
(`x[-6::2]` writes positions 0, 2, 4) rather than borrowed from either neighbour.

### 4.1 The one divergence, stated

**An empty stepped range with a non-broadcastable value does not raise.**

```text
zeros(6)[4:2:2] = tensor([1.0, 2.0])
  upstream   RuntimeError: The expanded size of the tensor (0) must match the existing size (2)
  patched    writes nothing, returns
```

Upstream writes nothing here either, so the *values* agree; what differs is that upstream still runs
the broadcast check and this does not. It is not reproduced because reproducing it means running
`index_put_`'s broadcast check and then discarding its (differently worded) error, and no swept
architecture performs an empty stepped write at all. Recorded rather than hidden.

### 4.2 In-place, autograd, and the eager recorder

`__setitem__` is an in-place write, so docs/training/BACKWARD8.md's storage-version guard and the eager
recorder both apply. Measured, all three `__setitem__` routes on the same build:

```text
                       upstream            shim (patched and unpatched alike)
x[idx]   = w           grad [1,1,1]        RuntimeError: there is no eager graph
x[0:3]   = w           grad [1,1,1]        RuntimeError: there is no eager graph
x[0::2]  = w           grad [1,1,1]        RuntimeError: there is no eager graph
```

**The stepped write introduces no new divergence.** Differentiating through any `__setitem__` is a
pre-existing gap, it is uniform across the existing forms, and it **raises** rather than returning a
silently wrong gradient — which is the convention docs/kernels/INPLACE.md records. The new form joins that
rule rather than making an exception to it. Closing it is a recorder question and not a
`__setitem__` one, so it was not attempted here.

Write-through is the property that makes the lowering equivalent to upstream's `slice` + `copy_` at
all, and it holds because `index_put_` goes through `write_back` (docs/kernels/VIEWS.md §6): a view taken
before the call sees the write. `test_index_put_writes_through_a_view_taken_before_the_call` is that
question asked directly, and it is the one a value comparison structurally cannot answer — a
rebinding kernel reproduces every number through the receiver and loses them through every alias.

---

## 5. What did NOT change, and why that is the interesting part

**No kernel.** `rust/torch_c/src/aten.rs`, `methods.json` and `overloads.json` are untouched.
`aten.index_put_.default` already answered for both architecture shapes before this round began,
which was checked by driving it directly rather than assumed:

```text
index_put_(zeros(3,4), [None, [0,2]],       (3,2) values)   -> [[1,0,2,0],[3,0,4,0],[5,0,6,0]]
index_put_(zeros(2,1,6), [None, None, [1,4]], (2,1,2) values) -> [[[0,1,0,0,2,0]],[[0,3,0,0,4,0]]]
index_put_(zeros(3,4), [None, [0,2]],       (1,2) values)   -> the row, three times
```

That is the whole reason the top-ranked blocker cost no arithmetic: docs/architectures/ARCH100.md's split of
`missing_shim_name` (49) against `missing_aten_op` (22) predicted that most of the remaining work is
binding surface rather than kernels, and the head of its own list turned out to be neither — it was a
**routing** gap, an op that existed being reached by a walk that could not get to it.

Four golden cases were added anyway, in `_setitem_member_cases`, because the three above had no
coverage: every existing `[None, index]` case has exactly one leading `None` and the mrope shape
needs two, and no case pinned a `(1,k)` value broadcasting onto an `(n,k)` result. They pass today.
If one of them regressed, the two `c_error` member cases would keep passing — a refusal is a refusal
however it is reached — and the reason the lowering is possible would have quietly gone away.

---

## 6. Why the tests are shaped the way they are

`pytests/test_setitem.py` had to be written in a tree where the fix could not land, which is a
constraint worth naming because it produced a better test than the unconstrained version would have.

A test that pinned the refusal would have to be **deleted** when the patch lands. A test that pinned
the values would be **red** until then. Neither is a check across the transition, and the transition
is exactly where a mistake would enter. So every stepped-write test goes through one helper that
accepts exactly two outcomes:

* the write happened and produced **upstream's values, element for element**; or
* it refused, **by name**, naming the step, having mutated nothing.

Everything else fails — including the outcome this area exists to prevent, a call that returns
normally and leaves the receiver untouched, which lands on the first branch with the wrong values.
That test does not change when the patch is applied; it changes which branch it takes.

One further test reads `docs/bindings/SETITEM.md` and checks that the anchor §2's patch keys on still appears
twice in `bootstrap.py` (`__getitem__` and `__setitem__`). A document is not a check; that much of it
can be made into one, so that the patch fails loudly if a later edit moves the anchor rather than
rotting until somebody tries to apply it.

---

## 7. Verification

```text
suite                  481 ok, 0 FAIL   (+11 in test_setitem.py -> 492 ok)
DOCWATCH               PASS -- 480/480 evaluated marker(s) hold
compare.py             9142/9142, ops=224   (+5 cases: 2 c_error, 3 match)
compare.py --self-test 19 comparators x 11 fault modes, 0 problems
arch_sweep --only <13> 0/13 forward before, 5/13 after, 13/13 past __setitem__
```

`TORCH_C_ARTEFACT` was set for every `compare.py` run; without it the loader reads a fixed cache path
that may hold another checkout's binary and reports a plausible green.

The measurements in §3 and §4 were taken with the patch applied and both the crate and the vendored
shim rebuilt; `bootstrap.py` was then restored, rebuilt, and re-verified, so what is committed here
is the pristine file plus this document, the golden cases, and the tests.
