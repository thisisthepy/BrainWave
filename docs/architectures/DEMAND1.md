# DEMAND1 — closing the top of the demand list

Working round for `docs/architectures/DEMAND.md`'s ranked queue. Rank 2 (the legacy `torch.Tensor(...)`
constructor) is **out of scope** and untouched: it is `tensor.rs`'s `#[new]` slot and structural.

Everything below marked "measured" was produced by running upstream torch 2.13.0 in its own
interpreter (`env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`), with
`print("shim" if hasattr(torch._C, "_aten_implemented") else "upstream")` as the first line of
every script. Every one printed `upstream`. Scratch scripts under `/tmp/d1/`.

---

## 1. `aten.native_batch_norm.default` — upstream's measured behaviour

Written **before** any Rust, because this is the op where a wrong answer is most likely to look
right: two of its three results are read by nobody, and its most important side effect is not a
result at all.

### 1.1 The schema, and the lie in it

```text
aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias,
                        Tensor? running_mean, Tensor? running_var,
                        bool training, float momentum, float eps)
                     -> (Tensor, Tensor, Tensor)
```

**No argument carries an alias annotation.** Measured: `[(a.name, a.alias_info) for a in
schema.arguments]` is `[('input', None), ('weight', None), ('bias', None), ('running_mean',
None), ('running_var', None), ('training', None), ('momentum', None), ('eps', None)]` — every
one `None`. The schema says this op mutates nothing.

It mutates. `running_mean` and `running_var` are written in place in training mode (§1.3). The
schema is wrong on purpose upstream, which is why `_native_batch_norm_legit` exists beside it
carrying the annotations the functionaliser needs:

```text
aten::_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias,
                               Tensor(a!) running_mean, Tensor(b!) running_var,
                               bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
aten::_native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias,
                               bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
```

This matters here for one reason: **`capture.rs` decides what mutates by looking for a trailing
underscore in the op name** (`is_mutating`, capture.rs:203). `native_batch_norm` has none, so
capture would have recorded a training-mode BatchNorm as a pure node and replayed a trace that is
not single-assignment. §1.7 is what was done about it.

### 1.2 It is a leaf; `batch_norm` is the composite over it

```text
_dispatch_has_kernel_for_dispatch_key("aten::native_batch_norm", "CompositeImplicitAutograd")  False   <- leaf
_dispatch_has_kernel_for_dispatch_key("aten::batch_norm",        "CompositeImplicitAutograd")  True    <- composite
```

`TorchDispatchMode` logger, torch 2.13.0:

```text
torch.batch_norm(x, w, b, rm, rv, training=True,  0.1, 1e-5, False)  ->  empty.memory_format, native_batch_norm.default
torch.batch_norm(x, w, b, rm, rv, training=False, 0.1, 1e-5, False)  ->  empty.memory_format, native_batch_norm.default
nn.BatchNorm2d(3)(x)          [train]                                ->  add_.Tensor, empty.memory_format, native_batch_norm.default
nn.BatchNorm2d(3).eval()(x)                                          ->  empty.memory_format, native_batch_norm.default
```

`aten.batch_norm.default` never fires. So `torch.batch_norm` is a `bootstrap.py` composite, not an
`overloads.json` entry — the same call `layer_norm` and `group_norm` already make, for the same
measured reason. The `add_.Tensor` in the training row is the module's `num_batches_tracked += 1`,
above this layer.

### 1.3 The train/eval split — the whole of it, measured

Fixture used throughout: `x = arange(24, float32).reshape(1,3,2,4) / 7`, `w = [2.0, 0.5, 1.5]`,
`b = [0.1, -0.2, 0.3]`, `running_mean = [0.25, 0.5, 0.75]`, `running_var = [1.5, 2.0, 0.5]`,
`momentum = 0.1`, `eps = 1e-5`.

**training=True**

```text
out[:8]     -2.9549078941345215  -2.0820770263671875  -1.2092461585998535  -0.33641552925109863
             0.5364155769348145   1.4092464447021484   2.282076835632324    3.154907703399658
save_mean    (3,) float32   [0.5, 1.6428570747375488, 2.7857143878936768]
save_invstd  (3,) float32   [3.05490779876709, 3.054908037185669, 3.054908037185669]
running_mean AFTER          [0.2750000059604645, 0.6142857074737549, 0.9535713791847229]      <- MUTATED
running_var  AFTER          [1.3622448444366455, 1.8122448921203613, 0.4622448980808258]      <- MUTATED
```

**training=False**

```text
out[:8]     -0.3082469403743744  -0.07496297359466553  0.15832099318504333  0.3916049301624298
             0.6248888969421387   0.8581728935241699   1.0914567708969116   1.3247407674789429
save_mean    (0,) float32   []          <- EMPTY, not the running mean, not the batch mean
save_invstd  (0,) float32   []          <- EMPTY
running_mean AFTER          [0.25, 0.5, 0.75]     <- UNTOUCHED
running_var  AFTER          [1.5, 2.0, 0.5]       <- UNTOUCHED
```

So, itemised:

| | training=True | training=False |
|---|---|---|
| statistics used for `out` | the **batch's**, biased variance | `running_mean` / `running_var` |
| `save_mean` | batch mean, shape `(C,)` | **shape `(0,)`** |
| `save_invstd` | `1/sqrt(biased_var + eps)`, shape `(C,)` | **shape `(0,)`** |
| `running_mean` | **written in place** | untouched |
| `running_var` | **written in place** | untouched |

### 1.4 Biased for the output, unbiased for the running variance

The single most falsifiable detail here, and the two halves disagree:

```text
batch mean      [0.5,        1.6428571939468384, 2.7857143878936768]
biased   var    [0.1071428582072258, 0.1071428433060646, 0.1071428507566452]   (divide by n)
unbiased var    [0.12244898080825806, 0.12244895845651627, 0.12244897335767746] (divide by n-1)
1/sqrt(biased+eps) = [3.05490779876709, ...]  ==  measured save_invstd   <- OUTPUT uses BIASED
```

and the running-variance update:

```text
running_var[0] after = 0.9 * 1.5 + 0.1 * 0.12244898 = 1.3622449     <- matches measured 1.3622448444366455
                       (with the BIASED 0.10714286 it would be 1.3607143, which is NOT what upstream gives)
```

Pinned independently with `momentum=1.0`, where the old value drops out entirely:

```text
momentum=1.0  ->  running_mean after [0.5, 1.6428570747375488, 2.7857143878936768]      == batch mean exactly
                  running_var  after [0.12244898080825806, 0.12244896590709686, ...]    == UNBIASED var exactly
momentum=0.0  ->  running_mean after [0.25, 0.5, 0.75]   running_var after [1.5, 2.0, 0.5]   (unchanged)
```

**Momentum convention** (the one torch inverts relative to most literature):

```text
running = (1 - momentum) * running + momentum * batch_statistic
```

`eps` is added to the **variance**, before the reciprocal square root — not to the standard
deviation. Pinned by a constant channel: `native_batch_norm(ones(1,3,1,1), ...)` in training mode
gives `save_invstd = [316.2277526855469] * 3`, and `1/sqrt(1e-5) = 316.2277660168379`.
`1/(sqrt(0)+eps)` would be `100000`.

`eps` is not applied to the running-variance update: `running_var` after `momentum=1` is exactly
the unbiased variance with no `eps` in it.

### 1.5 Reduction axes, dtype, and the rest of the measured surface

**Axes.** The statistics are per **channel** — over dim 0 and every dim after 1. Confirmed on
three ranks with exact values:

```text
(N,C)   x=arange(12).reshape(4,3)   save_mean [4.5, 5.5, 6.5]     save_invstd [0.2981422543525696]*3
(N,C,L) x=arange(12).reshape(2,3,2) save_mean [3.5, 5.5, 7.5]     save_invstd [0.3287977874279022]*3
(N,C,H,W)                            as §1.3
(N,C,D,H,W) rank 5 accepted, same shapes
rank 1   IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
rank 0   ValueError: SmallVector unable to grow. Requested capacity (18446744073709551615) ...
```

**Dtype.** The `native_layer_norm` / `native_group_norm` rule, re-measured here rather than
assumed from them:

```text
input       parameters    out          save_mean / save_invstd
float32     float32       float32      float32
float64     float64       float64      float64
float64     float32       RuntimeError: mixed dtype (CPU): all inputs must share same datatype.
float16     float32       float16      float32        <- mixed precision allowed
float16     float16       float16      float16
bfloat16    float32       bfloat16     float32
bfloat16    bfloat16      bfloat16     bfloat16
int64       any           NotImplementedError: "batch_norm" not implemented for 'Long'
bool        any           NotImplementedError: "batch_norm" not implemented for 'Bool'
```

Note the mixed-dtype message differs from `native_group_norm`'s (`expect parameter to have scalar
type of Float`); this one is `all inputs must share same datatype.` — trailing full stop included.

Running statistics follow the parameter dtype and are updated in it (measured: float16 input with
float32 stats updates them in float32, `[0.274993896484375, ...]`; with float16 stats,
`[0.27490234375, ...]` — different numbers, so the accumulation dtype is observable).

**Other measured corners**

```text
weight=None, bias=None                 accepted; treated as 1 and 0
running stats both None, training=True accepted; batch statistics, no update, save_* populated
running stats both None, training=False *** SEGFAULT (exit 139) ***  see §1.6
exactly one of the two None            ValueError: running_mean and running_var must either both be None or neither be None
N=0, training=True                     RuntimeError: input tensor must have at least one element, but got input_sizes = [0, 3, 2, 2]
N=0, training=False                    accepted; out (0,3,2,2), save_* shape (0,)
one element per channel, training      accepted (no "more than 1 value" check at this level); save_invstd = 1/sqrt(eps)
eps = -1                               NOT refused; out and save_invstd are all NaN
non-contiguous input (permuted)        accepted; out is contiguous, values are the logical ones
out aliases input?                     No — `out is x` False, `data_ptr` differs, in both modes
weight of the wrong length             NOT refused, and the answer is garbage: see §1.6
```

### 1.6 Two places upstream is not a usable oracle

**(a) `training=False` with `running_mean=None`, `running_var=None` segfaults.** Reproduced twice,
in isolation, exit 139 with no Python traceback. It is unreachable through any real spelling —
`torch.batch_norm` raises `RuntimeError: running_mean must be defined in evaluation mode` first,
and `nn.BatchNorm2d(track_running_stats=False)` passes `training=True` in `.eval()` precisely so
that this cannot happen. **The shim raises upstream's `batch_norm`-level message here rather than
crashing.** A crash is not a behaviour to reproduce, and there is nothing else to compare against.

**(b) A `weight` / `bias` / `running_mean` whose length is not `C` reads out of bounds.** Measured
with `C=3`: `weight=[2.0, 0.5]` (too short) and `weight=[2.0, 0.5, 1.5, 9.0]` (too long) both
return, and both return **bit-identical numbers to the correct `weight=[2.0, 0.5, 1.5]`** — i.e.
the two-element case read a third float past the end of its buffer and happened to find `1.5`
there. That is uninitialised heap, not a semantic. `torch.batch_norm` refuses all of these before
the leaf (§1.8), so again nothing real reaches it. **The shim refuses a length mismatch at the
leaf, by name.**

### 1.7 What was done about capture

`capture.rs`'s `is_mutating` is a name rule — trailing `_` on the op name — and it is correct for
every op that has one. `native_batch_norm` mutates and has no underscore, so the rule does not see
it, and a training-mode capture would have recorded a node whose replay silently advances the
running statistics a second time.

Handled deliberately, not by widening the name rule: a small argument-aware guard beside it that
refuses `aten.native_batch_norm.default` **only when the call would actually mutate** — that is,
`training=True` with running statistics supplied. Eval-mode batch norm is genuinely pure (§1.3
measured it: statistics untouched, `save_*` empty), and that is the mode a captured inference
graph is in, so refusing it by name would have made every BatchNorm CNN uncapturable to buy
nothing.

### 1.8 `torch.batch_norm`, the composite — its own checks, in the measured order

These are the composite's, not the leaf's. Measured one at a time, and the order matters because
a call with two wrong arguments reports only the first:

```text
num_features = input.size(1) if input.dim() >= 2 else 0
                      (rank 0 and rank 1 both report num_features 0 — measured, both give the
                       running_mean message below with "0 elements")

1. running_mean defined?  -> length must be num_features:  "running_mean should contain 3 elements not 4"
   not defined and not training                          :  "running_mean must be defined in evaluation mode"
2. running_var  same                                     :  "running_var should contain 3 elements not 4"
3. weight  defined? length                               :  "weight should contain 3 elements not 4"
4. bias    defined? length                               :  "bias should contain 3 elements not 4"
```

Confirmed order: a call with `weight=[1]*4` **and** `running_mean=[1]*5` reports the
*running_mean* one.

`input.numel() == 0` short-circuits **before** all four checks (measured: `N=0` with a wrong-length
`weight` succeeds). The short-circuit is `out = input.clone(); if weight: out = out * weight[0];
if bias: out = out + bias[0]` — pinned by `C=0`, which raises `IndexError: select(): index 0 out
of range for tensor of size [0] at dimension 0` from the `weight[0]` inside it. Running statistics
are not touched on this path.

`F.batch_norm` in the vendored tree is upstream's own Python and already carries the two checks
above it — `_verify_batch_size` (`ValueError: Expected more than 1 value per channel when
training, got input size torch.Size([1, 3, 1, 1])`) and the `eps` sign checks. Nothing to add
there; it is not this repository's file.

---

## 2. `aten.full_like.default`

Leaf upstream (`CompositeImplicitAutograd` is `False`); `torch.full_like(x, v)` fires exactly one
record, `aten.full_like.default`.

```text
aten::full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None,
                Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
```

**The dtype rule is the opposite of `full`'s.** `torch.full` infers from the fill value;
`full_like` takes it from the reference tensor and ignores the fill value's Python type entirely:

```text
full_like(int64 x, 7)      int64 [7,...]      full_like(int64 x, 7.5)  int64 [7,...]   <- truncates toward zero
full_like(f32   x, 7)      float32 [7.0,...]  full_like(f32 x, True)   float32 [1.0]
full_like(int64 x, True)   int64 [1,...]      full_like(bool x, True)  bool [True,...]
full_like(int64 x, 3, dtype=float32)          float32 [3.0,...]        <- override wins
full_like(int32 x, 2**31)  RuntimeError: value cannot be converted to type int without overflow
full_like(zeros(0,3), 4.0) (0,3) float32 []   full_like(tensor(1.0), 4.0)  () float32 [4.0]
full_like(x, 1.0, device="meta")              meta tensor, shape/dtype preserved
memory_format=contiguous_format / layout= / pin_memory=  all accepted
```

Implemented by handing `full_default`'s **existing** fill half, `filled_block`, the reference
tensor's shape and tag — the same function `constant_pad_nd` already shares, so truncation and
bool truthiness cannot drift between `full`, `constant_pad_nd` and `full_like`.

`t5`/`switch_transformers`' actual call is `torch.full_like(relative_position_if_large,
num_buckets - 1)` on an `int64` tensor with an `int` fill — the first row above.

## 3. `aten.new_zeros.default`

Leaf upstream. `x.new_zeros(...)` fires exactly one record, `aten.new_zeros.default`.

```text
aten::new_zeros(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None,
                Device? device=None, bool? pin_memory=None) -> Tensor
```

Character-for-character `new_ones`'s schema with a different fill, and measured to behave that
way: dtype and device inherited from `self` unless overridden, `()` gives a 0-d tensor, `(0,)`
gives an empty one, a bool receiver gives `[False, False]`, `device="meta"` gives a meta tensor.
Implemented as one function shared with `new_ones_default`, switching only `Tensor::zeros` for
`Tensor::ones`, exactly as `zeros_or_empty_like` already shares three `*_like` factories.

`bart`'s call is `input_ids.new_zeros(input_ids.shape)` in `shift_tokens_right`.

## 4. `torch.as_tensor` — a spelling, confirmed

Not a kernel. Measured dispatch:

```text
torch.as_tensor([1,2,3])                 ->  aten.lift_fresh.default        (one record)
torch.as_tensor(t, dtype=float64)        ->  aten._to_copy.default          (one record)
torch.as_tensor(t)                       ->  NO record; returns the SAME OBJECT (`is` is True)
torch.as_tensor(t, dtype=t.dtype)        ->  NO record; same object
```

Both primitives are already implemented and golden-compared. Value/dtype table measured:

```text
as_tensor([1,2,3])        (3,)  int64      as_tensor([1.0,2.0])   (2,)  float32
as_tensor([[1,2],[3,4]])  (2,2) int64      as_tensor(5)           ()    int64
as_tensor(5.0)            ()    float32    as_tensor(True)        ()    bool
as_tensor([1,2,3], dtype=float32)  float32
as_tensor(ndarray f64)    float64, and SHARES MEMORY with the array (mutating the array shows through)
```

The zero-copy ndarray path is upstream's, and is not reproducible here — this shim's tensors do
not wrap foreign buffers. `as_tensor` is spelled over `lift_fresh` like `torch.tensor` already is,
which **copies**. Recorded rather than papered over.

## 5. `aten.linalg_vector_norm.default` — a kernel, and it was worth checking

```text
aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *,
                         ScalarType? dtype=None) -> Tensor
```

Leaf (`CompositeImplicitAutograd` False), distinct from the already-implemented
`aten.norm.ScalarOpt_dim`. `F.normalize(v, p=2, dim=1)` fires
`linalg_vector_norm.default, clamp_min.default, expand.default, div.Tensor` — the other three are
implemented, so this one kernel closes `sentence_embed`.

Measured on `v = [[3,4,0],[1,2,2]]`:

```text
ord=2   dim=None            5.830951690673828   (0-d)
ord=2   dim=1 keepdim=True  [[5.0],[3.0]]
ord=1   dim=1               [7.0, 5.0]
ord=inf dim=1               [4.0, 2.0]          max |x|
ord=-inf dim=1              [0.0, 1.0]          min |x|
ord=0   dim=1               [2.0, 3.0]          count of non-zeros, as a float
ord=3   dim=1               [4.497941493988037, 2.571281671524048]
ord=-1  / ord=-2            finite; the general sum(|x|^p)^(1/p) formula
dim=[]  == dim=None (reduces everything);  dim=[0,1] likewise;  dim=-1 negative accepted
dtype=float64 promotes before reducing
int64 input   RuntimeError: linalg.vector_norm: Expected a floating point or complex tensor as input. Got Long
empty reduction, ord=2      0.0
empty reduction, ord=inf    RuntimeError: linalg.vector_norm cannot compute the inf norm on the dimension 1because
                            this dimension is empty and the operation does not have an identity
```

(That missing space in upstream's message is upstream's, transcribed.)

## 6. `torch.meshgrid` — a spelling over `view` + `expand`

`CompositeImplicitAutograd`. Measured dispatch for two 1-D inputs, **both** indexing modes:

```text
view.default, expand.default, view.default, expand.default
```

and nothing else — no transpose, no `permute`. Both ops are implemented.

```text
meshgrid(arange(3), arange(4), indexing="ij")  ->  two (3,4)
meshgrid(arange(3), arange(4), indexing="xy")  ->  two (4,3)
meshgrid(arange(3))                            ->  one (3,)
meshgrid(a, b, c, indexing="ij")               ->  three (3,4,2)
meshgrid(tensor(1), arange(4), indexing="ij")  ->  two (1,4)   0-d inputs count as extent 1
meshgrid(a, b)  with no indexing=              ->  "ij", with a UserWarning from C++
meshgrid([a, b], indexing="ij")                ->  a single list argument is accepted
```

`"xy"` is `"ij"` with the **first two inputs swapped and the first two outputs swapped back** —
which is exactly what produces `(4,3)` from `view`+`expand` alone with no transpose in the trace.
`swin`'s `_create_relative_position_index` is the caller.

---

---

## 7. What landed

Split by kind, because "five items closed" hides which of them were work and which were a
table entry:

**New kernels (3)** — these are the whole of the `ops covered` movement, 168 → 171:

| op | file | shares with |
|---|---|---|
| `aten.native_batch_norm.default` | `aten.rs::native_batch_norm_default` | — (new) |
| `aten.full_like.default` | `aten.rs::full_like_default` | `full`'s `filled_block` |
| `aten.new_zeros.default` | `aten.rs::new_ones_or_zeros` | `new_ones`, same function |

**New spellings (4)** — no kernel, no movement in `ops covered`:

| spelling | route | where |
|---|---|---|
| `torch.batch_norm` | composite → `native_batch_norm` | `bootstrap.py::_install_composites` |
| `torch.as_tensor` | composite → `lift_fresh` / `_to_copy` / identity | same |
| `torch.meshgrid` | composite → `view` + `expand` | same |
| `torch.full_like`, `Tensor.new_zeros` | table entries | `overloads.json`, `methods.json` |

**Defect found by this round's own tests (1).** The golden harness's constant-input case —
written to pin where `eps` goes — failed on the *output* instead, and that is how upstream's
**fused** affine was found: `alpha = invstd*weight`, `beta = bias - mean*alpha`,
`out = x*alpha + beta`, where the last line is a cancellation of two large nearly-equal
numbers. Upstream's answer for a constant channel with `bias=0.1` is `0.0999755859375`, one
`float32` ULP grid at magnitude 632, not `0.1`. The obvious `(x-mean)*invstd*w + b` is exact
there and therefore wrong. Every case with a real variance agrees under either form, so
nothing else in the suite could have found it.

**Safety work (1).** `capture.rs` grew `MUTATES_WITHOUT_UNDERSCORE` and `mutates_this_call`
— see §1.7. Refuses a *training-mode* `native_batch_norm` with statistics; records eval mode
and stats-free training mode.

**Left undone, deliberately (1).** `aten.linalg_vector_norm.default` (`sentence_embed`) is
**not** implemented. It is a genuine kernel, not a spelling, and doing it properly means
sharing `norm.ScalarOpt_dim`'s existing six-arm accumulate-in-`opmath` walk rather than
writing a second one — the two compute the same `ord` family and a second copy would drift,
which is the same argument that made `full_like` reuse `filled_block` and `new_zeros` reuse
`new_ones`. That refactor is larger than the remaining budget for this round, so what landed
instead is §5: the full upstream measurement, so the next round starts with the oracle done
rather than re-deriving it. `docs/architectures/DEMAND.md` §0.1 promotes it to rank 3.

Rank 2 of the original list (the legacy `torch.Tensor(...)` constructor) was out of scope for
this round and is untouched.

## 8. The re-ranking of docs/architectures/DEMAND.md

DOCWATCH failed on three of that file's five `op-not-implemented` markers, which is the
mechanism working: the file asserts each gap is open precisely so that closing one forces a
re-rank rather than leaving a stale queue. `docs/architectures/DEMAND.md` §0 now carries §0.1 (what is
still open, renumbered) and §0.2 (what closed and by what), with the original ranking
preserved rather than edited in place — "which wall did each model hit" is that file's
measurement and rewriting it would lose the evidence. The three closed markers are flipped to
`op-implemented`, so a regression fails there too; the two spellings are pinned by `hasattr`,
having no kernel of their own to assert.

<!-- DOCWATCH: op-implemented aten.native_batch_norm.default -->
<!-- DOCWATCH: op-implemented aten.full_like.default -->
<!-- DOCWATCH: op-implemented aten.new_zeros.default -->
<!-- DOCWATCH: hasattr as_tensor true -->
<!-- DOCWATCH: hasattr meshgrid true -->

Measured in §5 above so the round that closed it did not have to re-measure it, and closed by
that round — docs/architectures/DEMAND.md §0.2 rank 3, kernel shared with `norm.ScalarOpt_dim`. Asserted
present now for the same reason the three markers above are: a regression (the kernel removed,
or removed from `_aten_implemented()`) fails here too.

<!-- DOCWATCH: op-implemented aten.linalg_vector_norm.default -->
