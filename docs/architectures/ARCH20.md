# The seven blocked architectures, and the in-place members with kernels but no names

Two jobs in one round, and they turned out to be the same job seen from two sides.

**Job one.** Twenty architectures, `transformers` 5.15.1, toy configs, a 4-token forward.
Upstream forwards 20/20; this shim forwarded **13/20**. Six of the seven blocked ones now
forward; the seventh (`gpt_bigcode`) is left, deliberately, and §10 says why.

**Job two.** `add_`, `sub_`, `mul_`, `neg_`, `exp_` were unbound tensor members. `x += y`
raised `NotImplementedError: TensorBase.__iadd__` — on a kernel (`aten.add_.Tensor`) that had
been implemented and golden-compared since docs/kernels/TAIL.md. §8.

The two jobs met in the middle: `falcon`'s third wall *was* `__iadd__`.

---

## 0. The numbers

| | before | after |
|---|---|---|
| architectures forwarding (of 20) | **13** | **19** |
| `pytests/run.sh` smoke tests | 229 | **241** |
| `tools/golden/compare.py` cases / ops | 3075 / 122 | **3302 / 133** |
| `verify_schemas.py` | 4234 / 4234 | **4295 / 4295** |
| SmolLM2-135M float32 prefill | — | **bit-identical (§11.4)** |

Split by kind rather than reported as one number, because the kinds are not interchangeable:

| | |
|---|---|
| new aten kernels | **11** — `log`, `expm1`, `constant_pad_nd`, `clamp`, and 7 in-place arithmetic overloads |
| rules changed inside existing kernels | **3** — `pow` promotion, `powi`, `inplace_cast_check` |
| defects fixed | **2** — in-place unsafe casts computed where upstream raises (§8.3); a stale SDPA refusal (§7.2) |
| names wired to kernels that already existed | **7** — 5 spellings, 2 members |
| other new spellings / members / surface | **34** |
| golden cases added for an *existing* kernel with no coverage | **14** (§11.3) |
| deletions | **0** |

Both sides of the architecture sweep were re-measured for this round rather than taken from the
brief; the script is `/tmp/arch7/sweep.py` (not committed — it is four lines of `AutoConfig`
around a `try`). Upstream reproduced 20/20 and the shim reproduced 13/20 with the same seven
names and the same seven messages, so the brief's measurement stands as given.

The shim side needs `TORCH_USE_RTLD_GLOBAL=1` and `PYTHONPATH=torchnative/src/main`, the same
two the checkpoint tests in `pytests/test_shim.py` already set (VENDOR.md wall 1).

### 0.1 The twenty, before and after

| model | before | after |
|---|---|---|
| llama, gpt2, qwen2, mistral, gemma | PASS | PASS |
| gpt_neox, opt, mpt, starcoder2, stablelm | PASS | PASS |
| olmo, phi, mixtral | PASS | PASS |
| **bert** | `torch._C._get_deterministic_algorithms` | **PASS** |
| **bloom** | `aten.pow.Tensor_Tensor: dtype promotion` | **PASS** |
| **cohere** | `torch.repeat_interleave` no table entry | **PASS** |
| **falcon** | `__getitem__` with a list index | **PASS** |
| **mamba** | `torch.log` no table entry | **PASS** |
| **persimmon** | `torch.square` no table entry | **PASS** |
| gpt_bigcode | `SourceRangeFactory.make_range` (TorchScript) | unchanged — §10 |

### 0.2 What kind of work each of the seven was

The brief asked for this classification before planning, and the answer is why the round was
cheap. **Only two of the seven needed a new kernel at all** (`bert`, `mamba`), and neither
needed one for the wall it stopped on.

| model | first wall | classification | later walls |
|---|---|---|---|
| bert | `_get_deterministic_algorithms` | **surface** (a state cell) | one **kernel** (`constant_pad_nd`), one spelling (`_nn.pad`) |
| bloom | `pow.Tensor_Tensor` | **promotion** | two **surface** (`_are_functorch_transforms_active`, `_FunctionBase.apply`) |
| cohere | `torch.repeat_interleave` | **spelling** (a composite) | two **spellings** (`torch.stack`, `Tensor.flatten`) |
| falcon | list `__getitem__` | **member** | one **spelling** (bool SDPA mask), one **member** (`__iadd__`) |
| mamba | `torch.log` | **kernel** (small) | `clamp`, `expm1` **kernels**; `conv1d`, `softplus`, `zeros_like` **spellings** |
| persimmon | `torch.square` | **spelling** (a composite) | none — one line opened it |
| gpt_bigcode | TorchScript frontend | **out of scope** | §10 |

**A wall is not one wall.** Every one of the six had more behind it, and the shape of what was
behind it was not the shape of the first. `persimmon` took a single composite; `mamba` took
five more items across three categories. There is no way to have known that from the first
error message, which is the argument for measuring after every fix rather than planning the
whole round from the first sweep.

### 0.3 Why the coverage sweep could not see any of this

The existing coverage sweep traces a forward **on upstream torch** and asks whether every
dispatched operator is implemented here. That question has an answer only for the last row of
this table:

| what was added | count | could the sweep see it? |
|---|---|---|
| new aten kernels (`aten.rs` dispatch keys) | 11 | **yes** |
| dtype / cast rules changed inside an existing kernel | 3 | **no** — upstream computes, so nothing is *missing* to notice |
| `torch.<name>` / `_C._nn.<name>` spellings | 12 | **no** — the sweep sees the aten op, which was implemented |
| `TensorBase` members | 16 | **no** — same reason |
| `_C` surface names that are not aten at all | 13 | **no** — never dispatched |
| behaviour fixes at the Python layer (list index, bool SDPA mask) | 2 | **no** |

Five of the twelve spellings and two of the sixteen members named kernels that were already
implemented **and golden-compared**: `torch.stack`, `torch.exp`, `torch.zeros_like`,
`torch.conv1d`, `torch._C._nn.softplus`, `Tensor.add_` and `Tensor.relu_`. The sweep reported
full coverage of all seven the whole time. §9 has the remaining inventory of that class.

---

## 1. Method

Reproduce, fix the first wall, rebuild, re-measure, repeat. Seven rounds of the sweep in all.
Two things about the loop are worth keeping:

* **The rebuild is not optional and not obvious.** `bootstrap.py` is `include_str!`-ed into the
  artefact, so editing it and re-running the sweep tests the old binary with no sign that
  anything is stale. Every round here ended with
  `strings torchnative/src/main/torch/_C.abi3.so | grep -c <a marker from the edit>`.
* **The sweep runs one architecture per line and writes the full traceback per failure.**
  Reading only the last line would have hidden that `bert`'s wall moved from `_C` to `_nn` to a
  kernel — three different problems that all print as `NotImplementedError`.

---

## 2. `bert` — a getter, then a real kernel

**Wall 1, `torch._C._get_deterministic_algorithms`.** `F.pad` reads
`torch.are_deterministic_algorithms_enabled()` on **every call**, before it does anything else
(`torch/nn/functional.py:5806`). `bert`'s `tie_weights` pads the output-embedding bias when the
head's vocabulary is wider than the embedding it is tied to, so this fires during
`from_config`: `bert` never reached its own forward.

Nine names, installed as one state cell in `_install_behaviour` beside `_get_cudnn_enabled`,
which is the same shape of thing. Defaults read off torch 2.13.0 rather than guessed:

```
_get_deterministic_algorithms()                 False
_get_deterministic_algorithms_warn_only()       False
_get_deterministic_fill_uninitialized_memory()  True     <- the one that is not False
_get_cudnn_deterministic()                      False
_get_mkldnn_deterministic()                     False
```

`fill_uninitialized_memory` defaulting *True* is the cell a blanket "determinism starts off"
would have got wrong. It changes nothing here — it is only consulted when determinism is on —
but it is the value `torch.utils.deterministic` reads back.

The setters are state cells and nothing more, which is stated at the definition: turning
determinism on makes this shim *report* determinism without any kernel changing behaviour.
Refusing the setter would be worse — `torch/__init__.py:1585` calls it unconditionally from
`set_deterministic_debug_mode` — and there is nothing to refuse *for*, since every kernel here
is a single implementation with no nondeterministic sibling to pick instead.

**Wall 2, `torch._C._nn.pad`.** `F.pad` picks a mode and hands off. Only `mode="constant"` is
wired; the other three are genuinely different aten ops (`reflection_pad*`, `replication_pad*`,
and a `cat`-based circular path) and are refused by name.

**Wall 3, `aten.constant_pad_nd.default` — a new kernel.** One of the three the round needed.
The parts that are not obvious, each measured on 2.13.0:

* **`pad` is read back to front, in pairs.** `pad[0:2]` is the *last* dimension. Pinned by a
  case whose two dimensions get different pads (`[1, 1, 2, 0]` on a `(2, 3)` gives `(4, 5)`),
  which a front-to-back reading cannot pass.
* **Negative entries crop**, and crop-then-pad happens on the *same* axis:
  `constant_pad_nd(x, [-1, 2])` removes one from the front and adds two zeros at the back.
* Cropping past the axis raises `narrow(): length must be non-negative.` — upstream's message,
  from upstream's own `narrow`, because upstream's implementation takes the same route.
* The fill follows `torch.full`'s conversion rules exactly (shared through `filled_block`), so
  `constant_pad_nd(int64_t, [1,1], 3.7)` pads with `3` and an out-of-range value raises
  `value cannot be converted to type int without overflow`.
* Two shape refusals are transcribed with upstream's missing spaces intact:
  `Pad length is 6while the input has 2dimensions.`

---

## 3. `persimmon` — one composite, and the exponent is an integer

`torch.square(relu_applied)`, `transformers/activations.py:213`
(`ReLUSquaredActivation`), once per MLP per layer.

A `TorchDispatchMode` logger shows `torch.square(x)` firing exactly one record —
`aten.pow.Tensor_Scalar` — for every dtype tried. `aten::square` is `CompositeImplicitAutograd`
and its body is `self.pow(2)`. So this is *not* an `overloads.json` entry: an entry would name
`aten.square.default`, a key no dispatcher ever sees, which is the complaint `layer_norm`'s own
note in that table already makes.

**The exponent is the integer `2`, not `2.0`, and that is load-bearing.** `pow`'s wrapped-number
rule keeps an integral tensor integral under an integer exponent, so `square(int64([2,3]))` is
`int64([4,9])` — measured — while a `2.0` here would have returned `float32`. `square(float16)`
stays `float16` for the same reason.

`square(bool_t)` is `int64` upstream and refuses here; §6.2 has the measurement and why the
ladder behind it is not reproduced.

---

## 4. `mamba` — five items, and the one that is about precision

Every one of `mamba`'s walls except `clamp` is in `init_mamba_weights`, i.e. during
construction. It is the architecture in this set that does the most arithmetic before it has a
forward to run.

| wall | what it needed |
|---|---|
| `torch.log(A)` | **kernel** `aten.log.default` (joins the `unary_float` family) + spelling + member |
| `Tensor.clamp` | **kernel** `aten.clamp.default` + member |
| `torch.expm1(-dt)` | **kernel** `aten.expm1.default` (not the family — see below) + spelling + member |
| `torch.conv1d` | **spelling** — `aten.convolution.default` already had a kernel |
| `F.softplus` | **spelling** — `aten.softplus.default` already had a kernel |
| `torch.zeros_like` | **spelling** — `aten.zeros_like.default` already had a kernel |

**`log`** is `unary_float`'s rule exactly (`int64`/`int32`/`uint8`/`bool` → `float32`, each
float dtype keeps its own), re-measured rather than assumed from `exp`. It has **no domain
refusal**: upstream returns `-inf` for `log(0.0)` and `nan` for `log(-1.0)`. That had to be
checked rather than assumed, because "raises on a negative input" is the plausible wrong guess
and `mamba`'s own `A = arange(1, state+1)` never leaves the positive half to reveal it.

**`clamp`** is the second instance in `aten.rs` of an in-place op landing before its
out-of-place sibling — `clamp_.default` has been implemented since docs/kernels/OPS8.md while
`x.clamp(...)` refused. (`relu`/`relu_` went the other way, docs/bindings/SPELLINGS.md §6.6.) Every rule
is `clamp_`'s, shared through two extracted helpers rather than restated; in particular "both
bounds absent is an error, not a no-op", which a fresh out-of-place implementation would
plausibly have made a no-op since there is no receiver to leave unchanged.

**`expm1` is the interesting one, and it is not in the `unary_float` family even though its
dtype rule is that family's.** candle has no `expm1`, and `t.exp()? - 1.0` is not it — near zero
the subtraction cancels every bit `exp` just produced:

```
torch.expm1(1e-8)      1.0000000050000001e-08
torch.exp(1e-8) - 1    9.99999993922529e-09     wrong from the 9th significant digit
```

`mamba` computes `dt + torch.log(-torch.expm1(-dt))`, the softplus inverse, whose whole point is
the small-argument regime. So the kernel goes through `f64::exp_m1` element by element — the
same shape `pow` and `bitwise_binary` use, for the same reason: no candle kernel, and the
callers are not hot loops. Reading at `f64` first is also what makes the `float16`/`bfloat16`
cases one correctly-rounded value instead of two roundings of a cancelled subtraction.

**`conv1d`** is a composite: measured, `conv1d(x, w, b, 1, 2, 1, 4)` fires exactly
`convolution(x, w, b, [1], [2], [1], False, [0], 4)`. The whole of the composite is filling in
the two arguments `conv1d` does not have. Scalars widen to one-element lists because that is
what the trace shows upstream passing. `padding="same"` is implemented for the symmetric case
and **refused by name when `dilation*(kernel-1)` is odd** — upstream pads the input
asymmetrically there, and rounding either way would shift the output by one sample.

---

## 5. `cohere` — three spellings, no kernels

| wall | classification |
|---|---|
| `torch.repeat_interleave(freqs, 2, dim=-1)` | **spelling** — a composite of four ops that all existed |
| `torch.stack` | **spelling** — `aten.stack.default` implemented and golden-compared since docs/architectures/ARCH.md |
| `Tensor.flatten` | **spelling** — `CompositeImplicitAutograd`, lowers to `reshape` |

`modeling_cohere.py:115`'s own comment says why the first one is needed: *"diff from Llama: we
interleave() instead of cat()"*.

**`repeat_interleave` has two overloads and only one is here.** Measured:

```
repeat_interleave(x, 2, dim=-1)   unsqueeze, expand, clone, view
repeat_interleave(x, 2)           view, unsqueeze, expand, clone, view
repeat_interleave(x, tensor, ...) repeat_interleave.Tensor, index_select
```

The integer-`repeats` overload is `CompositeImplicitAutograd` and emits no record of its own —
it is that four-op expansion, every one of which this shim already had. The tensor-`repeats`
overload is a real kernel plus `index_select`, neither of which exists here, so it is refused by
name rather than approximated.

The expansion is transcribed from the trace, and it is checked against upstream's *answers*, not
just its op sequence: `dim=1` on `[[0,1,2],[3,4,5]]` gives `[[0,0,1,1,2,2],[3,3,4,4,5,5]]` and
`dim=0` gives `[[0,1,2],[0,1,2],[3,4,5],[3,4,5]]`. The two differ, so a wrong unsqueeze axis
cannot pass both. Three refusals are copied from upstream's messages (negative `repeats`,
out-of-range `dim`, disagreeing `output_size`); `repeats=0` is **not** an error — it produces a
zero-length axis.

**`flatten`** is Python-level for the same reason `softmax` and `chunk` are: a `methods.json`
entry would name `aten.flatten.using_ints`, and the logger shows every form of the call firing
`aten.view.default` and nothing else. The body is `at::native::flatten` transcribed, including
the 0-d arm — `torch.tensor(5.).flatten()` has shape `(1,)`, not `()`, which "flatten a scalar
does nothing" would have got wrong. It calls `reshape` and not `view`, because that is what
upstream's body calls; spelling `view` would refuse where upstream copies.

---

## 6. `bloom` — a promotion rule, then two autograd surfaces

### 6.1 `pow.Tensor_Tensor` promotion

`build_alibi_tensor` computes `torch.pow(base, powers)` with a `float32` base and an `int32`
exponent. `pow_tensor_tensor` went through `same_dtype`, which refuses any mismatch by name.

The rule is `promote_types` — the table `mul.Tensor` and `bitwise_and.Tensor` already use — and
it was **re-measured against `pow.Tensor_Tensor`'s own result dtype over the full 10×10 grid of
storable dtypes** before being reused, not assumed from `mul`'s. Every cell agrees with
`torch._prims_common.get_higher_dtype` **except one**: `bool ** bool`, where upstream raises
`NotImplementedError: "pow" not implemented for 'Bool'`. So the shim refuses exactly there, and
computes everywhere else.

The subtlety docs/bindings/BIND.md §9 already paid for applies unchanged and is why the fast path is not
an optimisation: `promote_operands`' `lhs.tag() == rhs.tag()` early return is the only reason a
same-rank pair like `float16 ** float16` does not escape to `float32`, the way
`float16 ** bfloat16` correctly does. `get_higher_dtype`'s `if a is b` guards the same table for
the same reason.

**A second divergence turned up while measuring, and it is fixed in the same place: negative
integer exponents.** Upstream splits by overload, and the shim refused for all three:

```
pow.Tensor_Scalar(int64([2,1,-1,0]), -1)          RuntimeError   <- only this one refuses
pow.Tensor_Tensor(int64([2,1,-1,0]), int64(-1))   [0, 1, -1, 0]
pow.Scalar(2, int64([-1, 3]))                     [0, 8]
```

The computing rows are `c10::powi`: `1` for base 1, `±1` for base −1 by the parity of the
exponent, `0` otherwise. Refusing where upstream computes is the *safe* direction to be wrong
in, but it is still a divergence, and widening `Tensor_Tensor`'s dtype acceptance makes many
more integer pairs reach it. `powi` is transcribed; the `Refuse`/`Powi` choice is a parameter
of `pow_from_pairs` so that the split is visible at all three call sites.

### 6.2 What is still refused, now with the measurement instead of "not measured"

`pow_result_tag`'s `Bool` arm used to say torch's boolean-pow result category "has not been
measured". It has been now, and it is not a promotion rule — it is a ladder of exponent fast
paths in `pow_tensor_scalar`:

```
pow(bool_t, 2)      int64      the exp==2 -> self*self path, promoted
pow(bool_t, 0)      int64      the exp==0 -> ones_like path
pow(bool_t, True)   bool       the exp==1 -> clone path (True is the scalar 1)
pow(bool_t, 2.0)    float32
pow(True, bool_t)   RAISES     the Scalar overload has no such ladder
```

Reproducing a ladder no measured caller needs is how a wrong answer gets in, so it stays
refused — with the ladder written down, so the next reader does not have to re-measure it. The
one reachable spelling is `square(bool_t)`, and nothing in the twenty architectures squares a
mask.

### 6.3 `_are_functorch_transforms_active` and `_FunctionBase.apply`

`bloom`'s `BloomGelu` calls `GeLUFunction.apply(x)` in every MLP. So an inference-only shim
reaches `torch.autograd.Function` on a **forward**, through nothing gradient-shaped at all.

`torch/autograd/function.py:622` is the first line of `Function.apply` and reads
`_are_functorch_transforms_active()`. `False` is upstream's answer outside a transform and the
only one this shim can give truthfully — functorch's transforms are `vmap`/`grad`/`jvp`, none of
which exists here, so nothing is ever pushed onto the stack the predicate reports on. It is also
upstream's *ordinary* branch. `unwrap_if_dead`, three lines later, is derived from the same
empty dynamic-layer stack the four existing `_functorch` predicates read, rather than written
down as "return the argument".

`super().apply(...)` then lands on `_C._FunctionBase.apply`, which the shim's placeholder
surface did not have — surfacing as `AttributeError: 'super' object has no attribute 'apply'`
rather than as this shim's own refusal, because a `super()` lookup does not go through
`_ShimMeta.__getattr__`.

**What upstream's `THPFunction_apply` does that this does not, stated rather than skipped:** it
allocates a graph node, records input metadata, marks the outputs' `grad_fn`, and handles
dirty/non-differentiable marking. All of that is autograd bookkeeping (DESIGN.md §3 stage 0),
and none of it changes the *value* `forward` returns — the only thing a forward-only shim can
observe. So this constructs the ctx the metaclass already made (`cls._backward_cls`) and calls
`forward`.

Both of upstream's two `forward` shapes are honoured, because a model may use either:

```
combined:  forward(ctx, *args)                       <- bloom's shape
separate:  forward(*args) + setup_context(ctx, inputs, output)
```

The test is the tree's own `_is_setup_context_defined`, read out of `sys.modules` at call time
rather than reimplemented — the same late-binding shape as `_set_generator_metaclass`.

One detail that cost a round: `needs_input_grad` is a **read-only getset** upstream
(`type(torch._C._FunctionBase.needs_input_grad)` is `getset_descriptor`), and the shim's
placeholder surface reproduces that as a `property`, so assigning it raises "property ... has no
setter". It is now a property reading a private slot, which keeps it read-only from the model's
side as upstream's is.

---

## 7. `falcon` — a list index, a stale refusal, and `+=`

### 7.1 The list index

`fused_qkv[..., [-2], :]` (`modeling_falcon.py:283`, `_split_heads`). Upstream lifts the list
into an index tensor and takes the advanced-indexing path — measured:

```
x[..., [-2], :]     lift_fresh.default, index.Tensor
x[..., (0, 1), :]   lift_fresh.default, index.Tensor      (a tuple item too)
x[[0, 1]]           lift_fresh.default, index.Tensor
x[0, [1, 2]]        select.int, lift_fresh.default, index.Tensor
```

and `torch.equal(x[..., [-2], :], x.index_select(1, tensor([1])))` is True, so the lifted tensor
is an ordinary index tensor with ordinary negative-index wrapping. `aten.index.Tensor` here
already handled `None` placeholders and negative values, so **the fix is entirely in
`bootstrap.py`** — no kernel changed.

The half that is not obvious is the *top-level* list, and it is upstream's `treatSequenceAsTuple`:

```
x[[slice(None)]]   alias.default                 tuple arm
x[[[0, 1]]]        lift_fresh, index.Tensor      tuple arm, inner list lifts
x[[0, 1]]          lift_fresh, index.Tensor      tensor arm
```

A short list containing a slice / `Ellipsis` / `None` / tensor / sequence is read as a *tuple of
indices*; anything else (or any list of 32 or more items) is one index tensor. Transcribed.
Upstream also emits a deprecation `UserWarning` on the tuple arm; that is not reproduced, and
the reason is written at the function — it is a notice about Python-level spelling that carries
no information this shim can act on. The *behaviour* it describes is reproduced exactly.

`__setitem__` takes the same treatment through the same two helpers, because the two walks have
to agree on what an index *is* or `x[i] = x[i]` would take two different routes.

### 7.2 A refusal that had gone stale

Wall 2 was `scaled_dot_product_attention(attn_mask=<bool tensor>)`, and its refusal text named
the two kernels it was waiting on: `aten.scalar_tensor.default` and `aten.where.self`. **Both
have been in `IMPLEMENTED`, and golden-compared, since docs/architectures/ARCH.md.** Nothing re-read the
refusal when they landed, so an architecture stayed blocked on a wall that had already been
removed.

A refusal that names its dependencies is only better than one that does not if somebody
re-checks them. Worth a standing habit: when a kernel lands, grep the refusals for its name.

The conversion is upstream's, reproduced op for op and in order:

```
scalar_tensor(-inf)               the masked-out fill
scalar_tensor(0.0)                the attend fill
where.self(mask, zero, neg_inf)   the additive mask
_scaled_dot_product_flash_attention_for_cpu(q, k, v, mask)
```

Argument order in the `where` is the half a plausible reading gets backwards, so it was read off
the *values*: `where(tensor([[True, False]]), 0.0, -inf)` is `[[0.0, -inf]]`. A `True` in a
boolean attention mask means *attend*, so it selects the zero. Both fills carry the query's
dtype, not the default float, which is what keeps a float16 forward in float16.

### 7.3 `+=`

Wall 3 was `TensorBase.__iadd__` — job two, §8.

---

## 8. The in-place family

### 8.1 The inventory, measured

`aten.rs` before this round, against `methods.json` before this round:

| member | kernel before | member before | now |
|---|---|---|---|
| `add_` | `add_.Tensor` | **none** | bound, `+ add_.Scalar` kernel |
| `sub_` | **none** | none | **kernel** (Tensor + Scalar) + bound |
| `mul_` | **none** | none | **kernel** (Tensor + Scalar) + bound |
| `neg_` | **none** | none | **kernel** + bound |
| `exp_` | **none** | none | **kernel** + bound |
| `relu_` | `relu_.default` | **none** | bound |
| `__iadd__` / `__isub__` / `__imul__` | (as above) | **none** | bound |
| `div_` | `div_.Tensor` | bound | unchanged |
| `clamp_` | `clamp_.default` | bound | unchanged |
| `masked_fill_` | `masked_fill_.Scalar` | bound | unchanged |
| `fill_`, `copy_`, `zero_`, `uniform_`, `normal_` | yes | bound | unchanged |
| `index_put_` | `index_put_.default` | via `__setitem__` | unchanged |

**Two kernels existed with no way in.** `aten.add_.Tensor` since docs/kernels/TAIL.md and
`aten.relu_.default` since docs/kernels/KERNELS.md — both golden-compared the whole time, both
unreachable from Python. `x += y` raised `NotImplementedError: TensorBase.__iadd__`.

### 8.2 What still has no kernel, by name

Bound and refusing on their own overloads (unchanged, and each names the overload it needed):

* `div_.Tensor_mode`, `div_.Scalar_mode`, `div_.Scalar`
* `clamp_.Tensor` (tensor bounds)
* `masked_fill_.Tensor`
* `clamp.Tensor` (the out-of-place sibling, added this round with the same gap)

Not bound and not implemented, listed so the next round does not have to re-derive it:
`log_`, `sqrt_`, `pow_`, `clamp_min_`, `clamp_max_`, `addcmul_`, `addcdiv_`, `erf_`, `sigmoid_`,
`tanh_`, `abs_`, `reciprocal_`, `rsqrt_`. None is called by any of the twenty architectures.
`log_` and `tanh_`/`rsqrt_` would be one line each on top of the out-of-place kernels that
already exist; the rest need kernels that do not.

### 8.3 The rule that separates in-place from out-of-place, which the old `add_` did not have

An out-of-place op may promote as far as it likes because it allocates the result. An in-place
op has a destination already, so upstream computes the promoted result dtype and then **refuses**
if it cannot be cast back. `c10::canCast`, transcribed into `inplace_cast_check`:

```
float32.add_(int32_tensor)     ok       promote -> float32, fits
int32.add_(float32_tensor)     RAISE    "result type Float can't be cast to
                                          the desired output type Int"
int32.mul_(2.5)                RAISE    same, via the wrapped-number rule
int64.div_(2)                  RAISE    div always floats
int64.exp_()                   RAISE    "... output type Long"
int64.neg_()                   ok       neg does not promote
bool.mul_(bool)                ok       the product IS the logical and
```

**This closed an existing divergence in the wrong direction.** `int32.add_(float_tensor)` used
to *compute* here — casting the operand down and returning a truncated answer — and was recorded
as a `torch_error` golden case rather than fixed. Computing where upstream raises is silent
divergence; that case is now `both_error`.

Bool follows `arith_tag`, so `mul_` accepts it (a bool product is exactly the logical and under
the tag's 0/1 invariant, and upstream agrees: `[True,False].mul_([True,True])` is
`[True, False]`) while `add_` and `sub_` refuse it. `add_` does not acquire a capability
`add.Tensor` lacks.

One narrower-than-upstream case remains and is recorded at the function: when the promoted
result is *wider* than the receiver (`float16.add_(float64_tensor)`), upstream accumulates in
the wider type and narrows once while this accumulates in `opmath_in(receiver)`. Both narrow to
the receiver; they can differ in the last bit.

### 8.4 The `Scalar` overloads, and why they exist here

Upstream's *dispatcher* never names them: measured, `t.add_(2)` reports `aten.add_.Tensor`,
because `add_.Scalar`'s `CompositeExplicitAutograd` body wraps the number and redispatches. They
are here for the reason `add.Scalar` and `mul.Scalar` are — `methods.json` reproduces the
**parser**, and `x += 2` binds a `Scalar` signature there. Without them `x += 2` would refuse
while `x += tensor(2)` worked, a difference no caller can see a reason for.

### 8.5 Write-through

Every in-place kernel here ends at `write_back`, so the result is written **through the
receiver's layout** and a view or alias taken before the call observes it (docs/kernels/VIEWS.md §6).
That is not asserted, it is tested: §11.2 lists the cases that read the **base** afterwards, and
§11.3 the sabotage run that shows they can fail.

---

## 9. Kernels that are still unreachable by name

The class §0.3 names is not closed. This is the full remaining inventory — every op with a
kernel in `aten.rs` and no `overloads.json` entry, split by whether that is a gap or correct:

**Correct to have no `torch.<name>`** (no such public function upstream, or reached another
way): `_local_scalar_dense`, ~~`_safe_softmax`~~, `_scaled_dot_product_flash_attention_for_cpu`,
`_softmax`, `_to_copy`, `_unsafe_view`, `alias`, `lift_fresh`, `index`, `index_put_`, `slice`,
`copy_`, `fill_`, `masked_fill`, `masked_fill_`, `new_ones`, `normal_`, `uniform_`, `view`,
`contiguous`, `expand`, `detach`, `add_`, `sub_`, `mul_`, `div_`, `neg_`, `exp_`, `clamp_`,
`relu_` — all of these are members, `_nn` entries, or private.

> **Correction (docs/kernels/TRIL.md §2.3): `_safe_softmax` was in the wrong bucket, and both halves of
> the justification were false.** `hasattr(torch, '_safe_softmax')` is `True` on 2.13.0 —
> `torch._safe_softmax(x, 1)` is a real, working public function — and it fires
> `aten._safe_softmax.default`, a *leaf* op, so it is not "reached another way" either. The
> leading underscore is what put it here: the name reads as private, so nobody called it and
> nobody checked. It has an `overloads.json` entry now.
>
> That mis-bucketing cost more than one missing name. Two refusals in
> `bootstrap.py::scaled_dot_product_attention` went on naming `aten._safe_softmax.default` as a
> kernel that did not exist, long after it did — because **a name nothing calls cannot correct
> the text that says it is missing**. Both are fixed, and a test now asserts the *claim* (every
> kernel a refusal names as present is in `_aten_implemented()`, every one it names as absent is
> not) rather than the wording.
>
> `_scaled_dot_product_flash_attention_for_cpu` stays in this bucket, but half of its reason is
> also imprecise: the flat name *does* exist upstream. The verdict is unchanged because the
> kernel is genuinely reached another way — `F.scaled_dot_product_attention` on the 4-D
> `dropout_p == 0` path — which is the half that holds.
>
> `_softmax` likewise stays, and for a reason that got stronger: `torch.softmax` and
> `Tensor.softmax` now both reach it as Python-level composites (docs/kernels/TRIL.md §2.2), which is
> what "reached another way" was supposed to mean here.

**A real gap** — a public `torch.<name>` exists upstream, the kernel is here, and the name
refuses: `abs`, `clamp`, `clone`, `cos`, `sin`, `reciprocal`, `eq`, `ne`, `lt`, `le`, `gt`,
`ge`, `max`, `min`, `mul`, `reshape`, `unbind`, `bitwise_and`, `bitwise_or`, `bitwise_not`,
`scalar_tensor`, `convolution`, `gelu`, `silu`, `softplus`.

That last group is **not** fixed here, deliberately. Each is one table entry, but each also
needs a golden case that goes through the spelling — the discipline docs/kernels/GROUPED_MM.md §6.4
established after kernel-level cases passed for weeks while the members raised. Adding
twenty-five entries without twenty-five cases would raise a count without adding a checked
capability, which is the failure mode the last round of this repository was told to stop
repeating. It is a well-defined next round: the list above is the whole of it.

> **Correction (docs/bindings/SPELLINGS.md §5–§7): this "next round" already happened, and `gelu`/`silu`/
> `softplus` were never in the gap it closed.** 22 of the 25 names above now have an
> `overloads.json`/`methods.json` entry with a golden case (`rust/torch_c/src/overloads.json`'s
> own comment: *"docs/architectures/ARCH20.md §9's 25-name inventory ... 22 got a `torch.<name>` entry here, 3
> did not"*). `reshape` is the one exception still pending promotion in `aten.rs` itself
> (`tools/golden/compare.py` reports it as the sole `PENDING` case builder, not yet in
> `_aten_implemented()` — docs/bindings/SPELLINGS.md §7.3), not a missing table entry.
>
> The three left out — `gelu`, `silu`, `softplus` — were never "a real gap" by this section's own
> definition, and listing them in the same bucket as `abs`/`clamp`/etc. is what made them look
> like one: measured against torch 2.13.0, `hasattr(torch, "gelu")` is `False` (same for `silu`,
> `softplus`), so there is no bare `torch.<name>` upstream to reach — only
> `torch.nn.functional.<name>`/`torch._C._nn.<name>`, both of which this shim already answers via
> `bootstrap.py`'s `_install_nn`. Adding a `torch.gelu` entry would invent a spelling upstream
> itself does not have. See docs/bindings/SPELLINGS.md §7.1 for the per-name measurement.
>
> Standing check (docs/verification/DOCWATCH.md), so this correction cannot itself go stale the same way the
> paragraph above it did:
> <!-- DOCWATCH: hasattr gelu false -->
> <!-- DOCWATCH: hasattr silu false -->
> <!-- DOCWATCH: hasattr softplus false -->
> <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json triu present -->
> <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json gelu absent -->
> <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json silu absent -->
> <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json softplus absent -->
> <!-- DOCWATCH: count golden_pending eq 0 -->

---

## 10. `gpt_bigcode` — confirmed out of scope, and left

`transformers/models/gpt_bigcode/modeling_gpt_bigcode.py:54` is a bare `@torch.jit.script` at
**module scope**, so importing the model file at all runs the TorchScript frontend. Re-confirmed
against the current build:

```
modeling_gpt_bigcode.py:54   @torch.jit.script
torch/jit/_script.py:1255    ast = get_jit_def(obj, obj.__name__)
torch/jit/frontend.py:428    r = ctx.make_range(py_def.lineno, ...)
torch_c_bootstrap.py:229     NotImplementedError: SourceRangeFactory.make_range
```

Unchanged from the brief's measurement, and unchanged by any of this round's work — it is not
downstream of anything that moved. docs/graph/DYNAMO.md §12 already established that this is **not**
the abi3 wall (no `Py_BUILD_CORE` is involved), just a large un-attempted frontend:
`SourceRangeFactory` is the first of a chain that continues through `CompilationUnit`,
`ScriptFunction`, the IR builder and the type system.

**Left, on purpose.** Six of seven forwarding is a result; a half-built TorchScript frontend is
not. Note also that it is not on the critical path for inference — the decorated function is a
fused-mask helper, and the model's forward would run without TorchScript if the decorator were
not evaluated at import.

---

## 11. Verification

### 11.1 The gates, all exit 0

```
bash vendor/install_shim.sh                       exit 0
PYTHON=$PY sh rust/torch_c/pytests/run.sh         241 ok, 0 FAIL          exit 0
$PY tools/golden/compare.py                       3302/3302, ops=133      exit 0
$PY tools/golden/compare.py --self-test           13 x 11, 0 problems     exit 0
$PY rust/torch_c/pytests/verify_schemas.py        4295/4295               exit 0
```

| | before | after |
|---|---|---|
| `run.sh` | 229 ok | **241 ok** |
| `compare.py` cases | 3075 | **3302** |
| `compare.py` ops covered | 122 | **133** |
| `verify_schemas.py` | 4234 | **4295** |
| architectures forwarding | 13/20 | **19/20** |

The twelve new smoke tests are all *reachability* checks, and that is the point: the golden
harness already compared `aten.add_.Tensor` against upstream while `x += y` raised
`NotImplementedError`. A kernel case structurally cannot see a missing name.

### 11.2 The base-reading cases

docs/kernels/VIEWS.md §6 landed write-through, so every in-place member added here has cases that read
the **base** after mutating a **view** — not the return value, which every in-place op supplies
as `self` and which therefore passes against a kernel that computed into a fresh buffer.

Nine new entries in `_view_write_cases` (`sub_.Tensor`, `mul_.Tensor`, `add_.Scalar`,
`sub_.Scalar`, `mul_.Scalar`, `neg_.default` ×2, `exp_.default`), deliberately spread across
three view shapes so that a write-through handling only the contiguous case cannot pass all of
them:

* `select.int(base, 1, k)` — strided (the odometer branch of `tensor.rs::write_strided`)
* `select.int(base, 0, k)` — contiguous at a non-zero start offset
* `t.default(base)` — non-contiguous in both axes (`mul_.Scalar` uses this one)

Plus `_inplace_member_cases`, which does the same through the **member** rather than the
dispatch key, for `add_`/`__iadd__`, `sub_`/`-=`, `mul_`/`*=`, `neg_`, `relu_` — and a smoke
test (`test_the_in_place_members_write_through_to_the_base`) that does it once more without the
golden harness at all.

The out-of-place `clamp` gets the mirror-image case: mutate nothing, read the **receiver**, and
assert it is unchanged. A kernel that wrote through like `clamp_` passes a return-value case and
fails that one.

### 11.3 Sabotage

Nineteen deliberate misimplementations, each rebuilt and run through both gates. **Every one was
caught.**

| golden fails | smoke fails | sabotage |
|---:|---:|---|
| 22 | 0 | `log` computes `exp` |
| 0 | 1 | `expm1` computes `exp(x) - 1` |
| 16 | 1 | `constant_pad_nd` reads `pad` front-to-back |
| 4 | 1 | `constant_pad_nd` ignores negative (cropping) entries |
| 3 | 1 | `clamp` drops the promotion |
| **9** | 1 | `pow.Tensor_Tensor` reverts to `same_dtype` |
| **2** | 1 | `powi` drops the `base == ±1` arms |
| crash | 1 | `inplace_cast_check` accepts every cast |
| 9 | 3 | in-place arith returns a fresh tensor instead of writing through |
| 11 | 2 | `neg_` is the identity |
| 6 | 1 | `exp_` is the identity |
| 0 | 1 | `__getitem__` refuses a list index again |
| 0 | 1 | `square` uses a float exponent `2.0` |
| 0 | 1 | `repeat_interleave` unsqueezes at `axis` instead of `axis + 1` |
| 0 | 1 | `flatten`'s 0-d arm becomes a no-op |
| 0 | 1 | `_nn.pad` accepts every mode as constant |
| 0 | 1 | determinism `fill_uninitialized_memory` defaults False |
| 0 | 1 | `_FunctionBase.apply` drops `needs_input_grad` |
| 0 | 1 | `conv1d` passes `transposed=True` |

The `0` rows in the golden column are correct rather than alarming: those nine are *spellings*,
*members* and *surface* items, which the golden harness compares by dispatch key and cannot see
at all. §0.3 is the same fact from the other direction. They are exactly why the twelve smoke
tests exist.

**Two rows were `0` in both columns on the first run, and that is the finding.**
`pow.Tensor_Tensor` reverting to `same_dtype` and `powi` losing its `base == ±1` arms broke
`bloom` and left the entire golden suite green — because **every existing `pow.Tensor_Tensor`
case used a same-dtype pair**, and no case anywhere used a negative integer exponent on the two
overloads that compute rather than refuse. Those were kernel changes with zero coverage: the
"case that cannot fail" the brief said to assume was there. Fourteen cases were added (the
mixed-dtype promotion grid, the `bool ** bool` refusal, three negative-exponent columns, and the
`pow.Scalar` sibling), and the two sabotages now fail 9 and 2.

**One sabotage did not produce a wrong answer — it produced a crash, and that changed a
comment.** Disabling `inplace_cast_check` made `exp_(int64)` reach candle's `exp`, whose integer
arms are `todo!()`:

```
pyo3_runtime.PanicException: not yet implemented: no unary function for i64
```

which took the harness's interpreter down mid-run rather than failing a case. So that refusal is
not only fidelity to upstream: it is what keeps an `i64` tensor from reaching a `todo!()` across
the FFI boundary. `exp_inplace`'s doc comment says so now, with the panic text.

### 11.4 SmolLM2-135M float32 prefill — bit-identical

A promotion rule that changes a model result is a bug, not a feature. The baseline artefact was
rebuilt from `8c07af8` in a throwaway worktree and swapped into the vendored tree (only
`_C.abi3.so` differs between the two runs; `strings` confirmed which was installed each time),
then a 14-token prefill was checksummed over all 688 128 float32 logits.

| | Σ logits | max | sha256 |
|---|---|---|---|
| before (`8c07af8`) | 10034413.728565 | 31.687803 | `9f1d6e8c…d5f7` |
| **after** | **10034413.728565** | **31.687803** | **`9f1d6e8c…d5f7`** |
| upstream torch 2.13.0 | 10034415.318427 | 31.687809 | `9ee44c44…43fd` |

Identical to the bit. The upstream difference is the shim's pre-existing float divergence
(docs/bindings/BIND.md §8.3), unchanged by this round — which is the other half of what the check is for:
if the divergence had *shrunk*, something in here would have been changing model numerics.

### 11.5 What this round did NOT verify

* **`gpt_bigcode`.** §10 — left, not attempted.
* **The twenty-five remaining unreachable spellings.** §9 — inventoried, not fixed.
* **Android and iOS.** Only the host artefact was built and run. Nothing added here is
  platform-specific (no new `#[cfg]`, no new FFI surface), but "should be fine" is not a
  measurement and this section is where it would go.
* **`pow.Tensor_Scalar` on a boolean tensor.** §6.2 — measured, still refused.
