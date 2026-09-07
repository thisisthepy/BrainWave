# GOLDEN — closing the keyword-dispatch blind spot

`tools/golden/compare.py` is this repository's headline correctness number:
every case it runs calls both upstream torch and this shim's `_C._aten_dispatch`
on the same inputs and diffs value, shape and dtype. docs/design/DISPATCH.md §4.1
found that this number was structurally blind to one entire code path — this
records what closed it, and what is still open.

---

## 1. The gap, as found

Every one of the 2811 cases `tools/golden/cases.py` built (before this
change) called `_aten_dispatch` **positionally**:
`c_module._aten_dispatch("aten.add.Tensor", a_c, b_c)`. Production code never
calls that way — `bootstrap.py`'s `resolve()`/`_bind()` binds a call into a
dict and dispatches with `dispatch(key, **bound)`, always by keyword. The
keyword lookup goes through `optional()` in `rust/torch_c/src/aten.rs`, which
for an argument found in `kwargs` (not in the positional tuple) consults
`interned_name()` — a hand-written table mapping ~74 argument names to
pre-interned `PyString`s, added for the performance work docs/design/DISPATCH.md §3
describes. That table is a lookup table, not a source of truth: a name not
in it just takes the slower path. But an entry that names the **wrong**
string is different — `kwargs.get_item(wrong_key)` returns "not found," which
`optional()` returns as `Ok(None)`, indistinguishable from "the caller never
passed this argument." A kernel reading an optional argument this way falls
back to its default silently; a kernel reading a *required* one raises
`missing required argument`. Either way the call does not do what the caller
asked, and no case built positionally could ever see it, because none of
them ever exercised `interned_name()` at all.

docs/design/DISPATCH.md §4.1 proved this by hand: changing `"self" =>
intern!(py, "self")` to `intern!(py, "self_TAMPER")` left `compare.py` at
2811/2811 while the pytest smoke suite (which does route real calls through
`bootstrap.py`, and therefore through `**bound`) dropped 48 tests.

## 2. What "closing the gap" means here, and what it does not

The fix is **not** a second, keyword-flavoured copy of every case — most
operators here take `self` and little else, and duplicating the whole
2811-case matrix would roughly double runtime for coverage that repeats the
same `interned_name()` code path 2811 times over instead of once. The table
is one flat `match name { ... }` shared by every kernel; a tamper of one arm
breaks *that name*, everywhere it is used, identically regardless of which
op reads it. So the target is: **for every argument name the currently
implemented ops actually read through `optional()`/`required()`/`tensor_arg()`
(directly, or through one of the small `*_arg` helpers built on top of it),
have at least one golden case that supplies it by keyword** — enough that
tampering any one `interned_name()` arm turns at least one case red, without
inflating the suite by more than the number of names that needed it.

Which names that is was read out of `rust/torch_c/src/aten.rs` mechanically
(grepping `aten_dispatch_inner`'s `match op` block for the op → function
mapping, then each function's `optional`/`required`/`tensor_arg`/
`{dim,bool,int,float,scalar,dtype}_arg`/`device_arg_or_label` call sites for
`(index, name)`), not guessed from the schema tables — the schema tables
(`rust/torch_c/pytests/verify_schemas.py`'s 4203 entries) say what upstream
accepts, but what matters for this specific gap is what **this shim's
`optional()` helper is actually asked to look up**, which can be a subset
(not every schema argument is implemented) with different names in a few
places (`convolution`/`native_layer_norm` read `input`, not `self`).

Each addition was verified two ways before being written down as a case:
against **upstream** — `torch.ops.aten.<op>.<overload>(name=value, ...)` — to
confirm upstream itself accepts the call by keyword (a few arguments are
keyword-only after the schema's `*`, e.g. `add.Tensor`'s `alpha`; none of the
names this change touches turned out to be positional-only through
`torch.ops.aten.*`, which is a different, more permissive doorway than the
friendly `torch.add()`-style wrappers), and against the **untampered shim**,
to confirm the same call succeeds and agrees with upstream before the case
is trusted to mean anything.

## 3. What was added

32 new cases (2811 → 2843), one or a small handful of arguments per op,
appended to the existing per-op builder in `tools/golden/cases.py` (search
`Keyword-argument coverage` — every addition carries that marker and cites
this doc). No existing case was changed; nothing was removed.

| op | arguments added by keyword |
|---|---|
| `aten.add.Tensor` | `self`, `other`, `alpha` |
| `aten.where.self` | `condition`, `self`, `other` |
| `aten.argmax.default` | `self`, `dim`, `keepdim` |
| `aten.transpose.int` | `self`, `dim0`, `dim1` |
| `aten.permute.default` | `self`, `dims` |
| `aten.addmm.default` | `self`, `mat1`, `mat2`, `beta`, `alpha` |
| `aten.baddbmm.default` | `self`, `batch1`, `batch2`, `beta`, `alpha` |
| `aten.multinomial.default` | `self`, `num_samples`, `replacement` |
| `aten.topk.default` | `self`, `k`, `dim`, `largest`, `sorted` |
| `aten.sort.default` | `self`, `dim`, `descending` |
| `aten.histc.default` | `self`, `bins`, `min`, `max` |
| `aten.isin.Tensor_Tensor` | `elements`, `test_elements` |
| `aten.full.default` | `size`, `fill_value`, `dtype` |
| `aten.scalar_tensor.default` | `s`, `dtype` |
| `aten.embedding.default` | `weight`, `indices` |
| `aten.index_put_.default` | `self`, `indices`, `values`, `accumulate` |
| `aten.masked_fill.Scalar` | `self`, `mask`, `value` |
| `aten.gather.default` | `self`, `dim`, `index` (`sparse_grad` was already keyword-exercised, see §4) |
| `aten.scatter.src` | `self`, `dim`, `index`, `src` |
| `aten.split.Tensor` | `self`, `split_size`, `dim` |
| `aten.split_with_sizes.default` | `self`, `split_sizes`, `dim` |
| `aten._softmax.default` | `self`, `dim`, `half_to_float` |
| `aten.slice.Tensor` | `self`, `dim`, `start`, `end`, `step` |
| `aten.pow.Tensor_Scalar` | `self`, `exponent` |
| `aten.convolution.default` | every argument (`input`, `weight`, `bias`, `stride`, `padding`, `dilation`, `transposed`, `output_padding`, `groups`) |
| `aten.native_layer_norm.default` | `input`, `normalized_shape`, `weight`, `bias`, `eps` |
| `aten.cat.default` | `tensors`, `dim` |
| `aten.randint.low` | `low`, `high`, `size`, `dtype` |
| `aten._scaled_dot_product_flash_attention_for_cpu.default` | `query`, `key`, `dropout_p`, `is_causal`, `scale` |
| `aten.normal_.default` | `self`, `mean`, `std` |
| `aten.uniform_.default` | `self`, `from`, `to` (`from` is a Python keyword, spelled `**{"from": ...}`) |
| `aten.clone.default` | `self`, `memory_format` — see §4 on why this one is a refusal case, not a `match` |

Picks favoured operators that cover several interned names in one case
(`convolution` alone accounts for six: `groups`, `dilation`, `padding`,
`output_padding`, `stride`, `transposed`) over one case per name, which is
why 32 cases reach roughly five dozen names rather than needing one case
each.

## 4. Coverage against `interned_name()`'s table, and what stayed out

`interned_name()` currently has 74 arms. Counting every place `cases.py`
supplies an argument by keyword — literal `name=value` at an `_aten_dispatch`
call site, **and** the handful of pre-existing helpers that forward a
`kwargs` dict (`_gather_case`'s `kwargs={"sparse_grad": True}`,
`_scalar_tensor_case`'s `kwargs={"device": "cpu"}`, `_unary_case`'s
`kwargs={"beta": ..., "threshold": ...}` in `softplus_cases` — none of these
show up in a literal-keyword grep, which is why the first pass at this count
undercounted) — **61 of the 74 names** are now exercised by keyword by at
least one case.

**13 remain uncovered.** Twelve of those are inert as far as this table goes
— they are reserved for ops `_aten_implemented()` does not advertise yet
(`query`/`key`/`scale`/`is_causal`/`dropout_p` were in this set before §3's
SDPA case; `mean`/`std`/`from`/`to` before the `normal_`/`uniform_` cases;
`memory_format` before the `clone` case) or genuinely unused by any
implemented op today. One is left uncovered **on purpose**:

**`generator`.** `generator_arg()` in `aten.rs` does not read a generator's
*value* — it checks only whether the argument is absent/`None`, or is the
object `bootstrap.py` marks with `_shim_is_default_generator = True`, or
anything else (which it refuses by name — "only torch.default_generator is
implemented"). `normal_`/`uniform_`/`multinomial`'s actual random stream
never depends on which branch was taken; it always reads the one process-
global generator. That means a case built the obvious way —
`generator=None` — cannot distinguish a working lookup from a tampered one:
under `interned_name("generator")` returning the wrong key, `kwargs.get_item`
misses, `optional()` returns `Ok(None)`, and `generator_arg` takes the exact
same "absent" branch it would have taken for a genuine `None`. The only
call shape that *would* observe the tamper is a **non-default** generator —
upstream computes fine with its own generator instance; this shim, reading
its argument correctly, refuses it by name (`c_error`); a tampered lookup
would make the shim silently treat it as absent and compute anyway,
flipping `expect="c_error"` into an unexpected success. No case anywhere in
this file constructs such a generator (there is no golden fixture for a
non-default `torch.Generator()`), so this was left as a genuine, named gap
rather than writing a `generator=None` case that would report "covered"
while proving nothing. If this needs closing, the shape to build is: a
non-default generator by keyword, `expect="c_error"`, and — the same
building block `clone`'s `memory_format` case above uses — a value the shim
is documented to *refuse*, so the tamper is observable as "refusal stopped
firing" rather than "no-op stayed a no-op."

`clone.default`'s `memory_format` is the same shape for the same reason:
`reject_memory_format` never uses the value it reads except to decide
whether to error, and `contiguous_format`/`preserve_format` are silently
accepted no-ops — indistinguishable from "argument absent" to a black-box
comparison. `memory_format=torch.channels_last`, which the shim is
documented to refuse and upstream is not, is the value that makes a
tampered lookup observable, so that is the case that was added
(`expect="c_error"`).

## 5. The tamper, both ways

Acceptance test, run exactly as docs/design/DISPATCH.md §4.1 ran it originally —
`"dim" => intern!(py, "dim")` changed to `intern!(py, "TAMPERED_dim")` in
`rust/torch_c/src/aten.rs`, restored from a `cp` backup afterward (md5
verified identical before and after; never `git checkout -- <path>`).

**With `aten.rs` unmodified** (this change's new baseline):

```
tools/golden/compare.py     -> 2843/2843 cases passed, ops covered=119, exit 0
tools/golden/compare.py --self-test -> PASS, 12 comparators x 11 fault modes, exit 0
rust/torch_c/pytests/run.sh -> 177 ok / 34 pre-existing, unrelated failures (see §6), exit 1
```

**With the tamper applied:**

```
tools/golden/compare.py -> 2838/2843 cases passed, 5 FAILED, exit 1
rust/torch_c/pytests/run.sh -> 176 ok / 35 failures (one new), exit 1
```

Golden turns red. The five cases that catch it are exactly the ones that
supply `dim` by keyword — `argmax`, `_softmax`, `gather`, `scatter.src`,
`slice.Tensor` — each reported either a shape mismatch (`argmax`: `(2,1)` vs
`(1,)` because the shim silently reduced over the whole flattened tensor
instead of `dim=1`; `slice.Tensor`: `(2,2)` vs `(1,4)`) or a
`SILENT DIVERGENCE` (`_softmax`/`gather`/`scatter.src`: the shim raised
`missing required argument 'dim'` — these three have no default for `dim`,
so the miss surfaces as a hard refusal instead of a wrong answer — while
torch computed a value). No case outside those five moved: the other 27 new
cases test names other than `dim`, and 2811 old cases were never touching
the interned lookup at all, so they could not have moved either way.

The smoke suite drops by exactly one test,
`test_a_bare_int_binds_a_sized_int_list` — a real, separate, additional
failure caused by the tamper, isolated by diffing the FAIL set before and
after (the other 34 fails present both before and after are pre-existing
and unrelated, see §6). This is far fewer than the 48 the original `"self"`
tamper caused in docs/design/DISPATCH.md §4.1, which is expected: `self`/`other`
are read by nearly every op bootstrap.py's real call sites reach, while
`dim` is read by a narrower set, and the smoke suite's coverage of that set
by keyword was already partial before this change.

**Honest framing, per the brief's own instruction:** this is not "one or
two cases resting on one operator" — it is five *different* operators
(`argmax`, `_softmax`, `gather`, `scatter.src`, `slice.Tensor`), each
independently built and independently verified against upstream before the
tamper, all failing for the same underlying reason. That breadth is the
signal that the fix generalises rather than accidentally covering one
lucky case.

## 6. A verification pitfall found along the way

`tools/golden/loader.py`'s `load_shim()` defaults to
`/Volumes/macMini/caches/cargo-target/release/lib_C.dylib` (no `TORCH_C_ARTEFACT`
env var, no `--artefact` flag) when neither is given. Building with a
different `CARGO_TARGET_DIR` — as this task's own instructions require, to
avoid colliding with other agents' concurrent builds — writes the artefact
somewhere `compare.py` never looks unless told to. The first tamper run
here (`compare.py` with no `--artefact`) reported **2839/2839, exit 0** —
looked like the tamper had no effect — because it was silently comparing a
stale, unrelated binary that happened to already exist at the default path
(built at some earlier, unrelated time, `strings`-verified to contain no
`TAMPERED` marker). Passing `--artefact` (or exporting
`TORCH_C_ARTEFACT`) explicitly to point at the actual build fixed this; the
numbers in §5 are all from runs that did. **Any verification of this
harness that sets a non-default `CARGO_TARGET_DIR` must also set
`TORCH_C_ARTEFACT`, or it is not measuring what it thinks it is measuring.**

```
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
```

## 7. A wrong diagnosis, and what it actually was

This section previously reported 34 of 211 smoke tests and all of
`verify_schemas.py` failing, attributed them to `spike-venv`'s real `torch`
winning Python's package resolution over the vendored namespace portion, and
concluded that the 4203-entry baseline "could not be reproduced in this
sandbox". It was argued in detail, with a minimal two-directory reproduction,
and it was wrong.

The cause was that **this worktree had never had `vendor/vendor_torch.sh`
run in it.** `torchnative/src/main/torch/` held four files instead of the
upstream tree, so every subprocess that puts that directory on `PYTHONPATH`
found nothing to shadow with and fell through to the real `torch` — which is
exactly the symptom described, arrived at by a different route. The vendored
tree is gitignored and is per-worktree setup, so a fresh `git worktree add`
does not have it and nothing in the suite says so.

Running it takes both numbers straight back:

```
bash vendor/vendor_torch.sh && bash vendor/install_shim.sh
sh rust/torch_c/pytests/run.sh          211 ok, exit 0
python tools/golden/compare.py          2843/2843, ops=119, exit 0
python rust/torch_c/pytests/verify_schemas.py   4203/4203, exit 0
python tools/golden/compare.py --self-test      exit 0
```

The mechanism the old text described is real Python behaviour; it just was not
what was happening here, and the reasoning ran downhill from a wrong premise
to a confident conclusion. The tell was available and not taken: a baseline
that disagrees with the number the task hands you is more likely to be your
environment than a stale figure in the brief — and `ls torchnative/src/main/torch`
answers it in one command.

It is kept rather than deleted because §6 records a false-green from the same
session and this is its opposite, a false-red. Both came from trusting a
result without checking what produced it.


## 8. Running it

```
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-golden
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib   # see §6
bash vendor/install_shim.sh
PY=/Volumes/macMini/caches/spike-venv/bin/python
$PY tools/golden/compare.py                 # 2843/2843, ops covered=119, exit 0
$PY tools/golden/compare.py --self-test     # PASS, exit 0
PYTHON=$PY sh rust/torch_c/pytests/run.sh   # 177 ok / 34 pre-existing unrelated (§7), exit 1
```
