# REFOLD — folding `prims.*` back to `aten`, and BatchNorm into the convolution

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/refold.py refold present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/refold.py REFOLDABLE present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/refold.py UNREFOLDABLE_PRIMS present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/refold.py lower_and_refold present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/fuse.py fold_conv_batch_norm present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/fuse.py batch_norm_affine present -->

## 1. The table this round is judged by

Ops outside NNAPI, per model, before and after. Two columns are counted
because there are two honest answers to "outside NNAPI" and they differ:

| | counts |
|---|---|
| **ADDER\_MAP** | `target.NNAPI` — base names parsed out of `torch/backends/_nnapi/serializer.py`. An **upper bound** on coverage (docs/DECOMP.md §12.2): it has no overloads in it |
| **serialisable** | `nnapi.supported_ops()` — captured overloads that have a calling convention and can actually be handed to an adder. This is the one that decides whether a blob comes out |

### 1.1 The refold, against the ADDER\_MAP count

| model | raw | core table | core + refold | union table | **union + refold** |
|---|---|---|---|---|---|
| `smollm2_llama` | 17 | 16 | 16 | 15 | **15** |
| `vit` | 11 | 10 | 10 | **11** | **10** |
| `mobilenet_v2` | 3 | 3 | 3 | **6** | **6** |
| `mlp_gelu` (`Linear→GELU→Linear`) | 2 | 2 | 2 | 2 | **2** |
| `resnet` | capture refuses first (in-place residual add, docs/CAPTURE.md §4) |

**One model improves and two do not, and the two that do not are the point of
this table.** docs/PRIMS.md §3 reported `vit` going 10 → 11 when the thirteen
prims kernels landed — lowering ran to completion and the graph *ended* on
`prims.erf` and `prims.transpose` instead of stopping at `aten.erf` and
`aten.permute`. The refold takes it back to 10. It does not take it below 10,
and nothing here claims it does.

`smollm2_llama` and `mobilenet_v2` both had prims folded out of them (seven
distinct prims keys and three, respectively) and neither moved. Their residue
is outside NNAPI for reasons prims have nothing to do with: SDPA, `embedding`,
`arange`, `index.Tensor`, `constant_pad_nd`. Counting the fold as progress on
those two would be CLAUDE.md §5.3's "test count is not progress" wearing a
different hat.

**§12.5's warning held a third time.** `mobilenet_v2` under the union table is
still 3 → 6 ops outside and 203 → 1191 nodes. The refold does not rescue that;
it re-spells `prims.clone`/`sqrt`/`reciprocal` as their aten names and they are
all still outside NNAPI. **For `mobilenet_v2` the correct table is the core
one, and the correct pass is §3's fold, not lowering at all.**

### 1.2 The BatchNorm fold, against the serialisable count

| model | serialisable-outside before | after fold | pairs folded | nodes |
|---|---|---|---|---|
| `mobilenet_v2` | **2** — `native_batch_norm`, `constant_pad_nd` | **1** — `constant_pad_nd` | 52 | 203 → **151** |
| `conv_bn_relu_conv_relu6_pool_linear_softmax` | **1** — `native_batch_norm` | **0** | 1 | 10 → 9 → 8 after constant folding |
| `vit`, `smollm2_llama`, `mlp_gelu` | 9 / 18 / 2 | unchanged | 0 | unchanged — no convolution+BatchNorm pair to fold |

docs/NPU.md §7 said `mobilenet_v2` was **two ops away**. It is now one.

## 2. The refold — `torchnative/export/refold.py`

### 2.1 The direction is forced, not chosen

`prims.transpose` is **stricter** than `aten.permute`, and that decides which
way the rewrite may go. Measured, not recalled — the test calls both:

```
prims.transpose(randn(3,4), [-1, 0])  ->  ValueError: Received an invalid permutation, [-1, 0]!
aten.permute  (randn(3,4), [-1, 0])  ->  shape [4, 3]
```

So every argument `prims.transpose` accepts, `aten.permute` accepts and
computes identically: **prims → aten is total**. The converse is partial, and
`[-1, 0]` is a permutation real graphs produce, so a "canonicalise to prims"
pass would turn working graphs into raises. `prims.split_dim` gives the same
asymmetry a second time — it rejects a negative `dim` every aten op in this
shim accepts.

### 2.2 The table, and the one that is absent

| prims | aten | why it is the same function |
|---|---|---|
| `cos` `sin` `erf` `tanh` `sqrt` `rsqrt` `reciprocal` `neg` | same name | upstream's own `impl_aten` for `prims.cos` **is** `torch.cos` (docs/PRIMS.md §1) — the same kernel under another key |
| `clone` | `aten.clone.default` | same, `memory_format` carried through rather than dropped |
| `view_of(a)` | `aten.alias.default` | docs/PRIMS.md §1 states the identity |
| `transpose(a, perm)` | `aten.permute.default` | full permutation, not `aten.transpose.int`'s two-axis swap |
| `split_dim(a, dim, n)` | `aten.view.default(a, shape)` | splits one axis into two adjacent ones, so it never reorders and never crosses a discontiguity; shape read from the meta capture recorded for the node |
| **`broadcast_in_dim`** | **none — refused by name** | see below |

`prims.broadcast_in_dim(a, shape, broadcast_dimensions)` is XLA's broadcast:
the caller names which result axis each input axis becomes, so it can insert an
axis in the middle and broadcast a size-3 axis against a size-2 one.
`broadcast_in_dim(ones(3), [3, 2], [0])` **has no `expand` spelling at all** —
and that is kept as a live control rather than a recollection: the test calls
`aten.expand(ones(3), [3, 2])` and requires it to raise. If expand ever
computes it, the test goes red and the reason for the refusal gets re-examined.

A `view` + `expand` composite does compute it. It is deliberately not emitted.
Refolding is a *re-spelling* — one node in, one node out, nothing to prove
beyond the identity. A two-node composite is a decomposition, and this project
does not write decompositions; it runs upstream's (docs/DECOMP.md §12.1).

### 2.3 It is terminal, and it has to be

`lower_and_refold` lowers first and refolds once, and does not feed the result
back. It cannot: the two passes are inverses on the ops they share — the union
table lowers `aten.tanh` to `prims.tanh` and this pass folds it back — so a
loop over both does not terminate. Lowering's fixed point is prims; the refold
is the single step off it.

### 2.4 The numerical proof

A re-spelling has no tolerance to spend, so **`0.0` is the only passing
answer** and it is measured two ways:

* **per table entry.** Each of the twelve prims is dispatched, refolded, and
  both traces replayed. All twelve: `max_abs_diff == 0.0`.
* **per model graph.** Each lowered graph is replayed against its refolded self
  on inputs the trace has not seen. `smollm2_llama`, `vit`, `mobilenet_v2`,
  `mlp_gelu`: all `0.0`.

## 3. The BatchNorm fold — `torchnative/export/fuse.py`

### 3.1 Why it is not a decomposition

A convolution is linear, so for an inference batch norm

    BN(Conv(x, W, B)) = Conv(x, W, B) * alpha + beta
                      = Conv(x, W * alpha, B * alpha + beta)

and two nodes become one. docs/DECOMP.md §12.7 called this the
*target-dependent* category, and §12.5 showed why the target-independent
treatment is wrong here: decomposing `native_batch_norm` produces `sqrt`,
`reciprocal` and `new_zeros`, and NNAPI has none of the three.

### 3.2 The arithmetic is quoted, because deriving it gives the wrong answer

docs/DEMAND1.md §5 records a defect this project already had. Upstream applies
a **fused** affine whose last line cancels two large nearly-equal numbers:

    alpha = rsqrt(running_var + eps) * weight
    beta  = bias - mean * alpha
    out   = x * alpha + beta

The algebraically identical `(x - mean) * invstd * weight + bias` **is exact
where upstream is not, and is therefore wrong.** Both are measured against
`aten.native_batch_norm.default` on the same numbers, and the constant-channel
case is measured beside them:

| | random channel, max abs diff | constant channel (`mean = x = 632`, `bias = 0.1`) |
|---|---|---|
| upstream | — | **0.0999755859375** |
| fused (this fold) | **0.0** | **0.0999755859375** |
| the obvious unfused form | 4.77e-07 | 0.10000000149011612 |

The test requires the fused form to match **and the obvious one not to**. If
they ever agree the check has stopped distinguishing them, and it says so by
name rather than passing.

### 3.3 What it refuses

Not folding is always safe; folding wrongly is not. Each of these must fold
**zero** pairs, and the test asserts zero rather than "fewer":

| case | why |
|---|---|
| training-mode batch norm | statistics depend on `x`, so there is no constant `alpha`. (Capture refuses it first — it mutates running statistics, docs/CAPTURE.md §4 — which is also a refusal) |
| the convolution's result read more than once | its unfused value is still needed |
| transposed convolution | its weight carries output channels on **axis 1**, so scaling axis 0 would scale the input channels |
| non-constant BN parameters, or `save_mean`/`save_invstd` read | the affine is not known at fold time, or the fused convolution does not produce what is being read |

### 3.4 The numerical claim, and what it is *not*

The fold moves the scale **inside the convolution's accumulation**, so it is
not bit-exact against the unfused graph and this document does not say it is.
§3.2's bit-exactness is a claim about the *affine*, not about the fold.

| graph | max abs diff over 3 unseen inputs | output magnitude |
|---|---|---|
| `conv_bn_relu_conv_relu6_pool_linear_softmax` | **1.49e-08** | O(1) — softmax output |
| `mobilenet_v2` (52 pairs) | 2.1e-32 | **2.3e-26** |

`mobilenet_v2`'s absolute number is meaningless on its own: at
`depth_multiplier=0.25` with random init this toy graph's output magnitude is
~1e-26, so *any* answer would clear an absolute threshold. The test therefore
bounds it **relatively** (ratio ~1e-6), and the absolute proof of the fold is
the first row, whose output is O(1). Writing the 2.1e-32 down as the headline
would be CLAUDE.md §5.5's check that cannot fail.

## 4. The deliverable — a whole model inside NNAPI's set

`Conv → BatchNorm → ReLU → Conv → ReLU6 → AdaptiveAvgPool → Linear → Softmax`,
the network docs/NPU.md §7 measured at one unmapped op before lowering and ten
after.

| step | ops outside `nnapi.supported_ops()` |
|---|---|
| captured | 1 — `aten.native_batch_norm.default`; `N.serialize` **refuses by name** |
| after `fold_conv_batch_norm` | 1 — `aten.t.default`, over a constant weight |
| after `fold_constants` (docs/NPU.md §5) | **0** |

Upstream's serialiser then writes the blob, and it decodes:

```
operations  CONV_2D(3) RELU(19) CONV_2D(3) RELU6(21)
            AVERAGE_POOL_2D(1) RESHAPE(22) FULLY_CONNECTED(9) SOFTMAX(25)
bytes       1156   (a multiple of 4, tables consumed exactly)
shapes      8 checked, 0 mismatches   (serialiser's propagation vs capture's record)
max diff    1.49e-08  fused graph vs unfused, 3 unseen inputs
```

The opcodes are asserted **by value**. A blob whose operation table is off by
one still decodes and still passes a shape check.

**The negative control is the half that makes this a result.** The unfused
graph is required to be *refused*, by name. If it ever serialises, the fold is
not what made this model reachable and the test says so instead of passing.

> **Still structurally validated, not executed.** There is no NNAPI runtime on
> a Mac; docs/NPU.md §2 draws that line and nothing here moves it.

## 5. What is still outside, and why each one is not a refold

### `mobilenet_v2` — `constant_pad_nd`, and it is not asymmetry-free

The obvious next pass is folding a zero `constant_pad_nd` into the following
convolution's `padding`, which is exact when the pad is symmetric. **It does
not close this model**, and the reason is worth writing down rather than
discovering:

    52 pads in the captured graph
    34 [0, 0, 0, 0]   -- no-ops; removable outright
    13 [1, 1, 1, 1]   -- symmetric; foldable into the conv's padding, exactly
     5 [0, 1, 0, 1]   -- asymmetric; not foldable

The five are the TF-style "same" padding in front of the stride-2 depthwise
convolutions. NNAPI's explicit-padding `CONV_2D` does take four separate
padding values, but upstream's `add_conv_underscore` reads torch's **symmetric**
`padding` argument, so asymmetric padding is unreachable *through this
serialiser* regardless. So such a pass would remove 47 nodes and leave the
`constant_pad_nd` count at 1 — the headline number would not move. It was
measured and not built.

### `vit` — `native_layer_norm`, SDPA, `select.int`, `contiguous`

None is a prims problem. `native_layer_norm` is the analogue of §3's job for a
normalisation with no preceding linear op to absorb it, and SDPA is a composite
NNAPI has no adder for at all.

### `smollm2_llama` — `embedding`, `arange`, `index.Tensor`, SDPA

Same. `prims.broadcast_in_dim` is also in its residue, and §2.2 says why that
one stays.

## 6. Reproducing

    export PATH="$HOME/.cargo/bin:$PATH"
    export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-refold
    export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
    export TORCH_C_STAGE=/tmp/stage-refold
    PY=/Volumes/macMini/caches/spike-venv/bin/python

    cd rust/torch_c && cargo build --release && cd -
    bash vendor/install_shim.sh
    PYTHON=$PY sh rust/torch_c/pytests/run.sh
    PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
        $PY rust/torch_c/pytests/nnapi_sizing.py

The nine tests this document is about are in `rust/torch_c/pytests/test_shim.py`
and share one subprocess fixture (`_REFOLD_SCRIPT`):

    test_the_refold_goes_prims_to_aten_because_the_other_direction_is_partial
    test_every_refold_table_entry_computes_the_same_value_as_the_prim
    test_a_prim_with_no_aten_spelling_is_refused_by_name
    test_the_refold_recovers_vits_regression_and_claims_nothing_more
    test_the_refold_is_a_respelling_and_replays_bit_for_bit
    test_the_batch_norm_affine_this_fold_uses_is_upstreams
    test_the_batch_norm_fold_refuses_where_the_algebra_does_not_hold
    test_folding_batch_norm_into_conv_takes_mobilenet_to_one_op_outside
    test_a_whole_model_now_lowers_with_nothing_outside_nnapis_set

No Rust changed this round. The golden harness is **8921/8921, ops covered 222**
— unmoved, and that is the check that it did not.
