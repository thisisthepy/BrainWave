# `float8_e4m3fn`, round 2: refusing exactly what upstream refuses

> **Superseded in part by `docs/FLOAT8C.md` (round 3).** Tables **D** and **E**
> below — the ten this build hung on and the thirteen it refused — are all
> closed, so their "after" columns no longer describe this build. §4.1's
> conclusion that fixing the ten needs a `candle-core` fork is **wrong**: the
> recursion reaches exactly one arm of candle's converter, and routing
> `F8E4M3 -> F32 -> F64` around it is exact and needs no patch. Two of table
> D's rows were also artefacts of the generic recipe rather than ops upstream
> computes; `docs/FLOAT8C.md` §2 has the re-measurement. Everything else here —
> §2's 114 transcribed refusals above all — still holds.

`docs/FLOAT8.md` turned three hangs into named refusals and then recorded a wider
finding from an eleven-op probe: seven ops answered where upstream declined. This
document is that finding enumerated properly, and closed.

**The eleven-op probe was a sample and it undercounted by an order of magnitude.**
Enumerating all 197 ops in `torch._C._aten_implemented()` finds **114** where
upstream 2.13.0 refuses `float8_e4m3fn` by name and this build did not refuse in
upstream's words: 48 computed an answer nothing can check, 27 hung, and 39 refused
for a reason of their own.

## 1. Method

Every op in `torch._C._aten_implemented()`, both sides, one call each.

1. **Recipe.** Arguments are synthesised from the op's own `_schema`, filling only
   the arguments that have no default and letting the op supply the rest. The
   shape is chosen by trying `(2,)`, `(2,3)`, `(1,3,4,4)`, `(1,)` against
   **upstream in `float32`** and keeping the first that returns. 158 of 197 ops
   get a recipe this way; 39 do not (table G), and for 32 of those a
   hand-written call was added (`aten.mm`, `aten.convolution`, `aten.gather`, ...).
2. **Same recipe, `float8_e4m3fn` operands, both sides.** Upstream is the oracle.
3. **Every call is under a timeout.** Each probe runs in its own subprocess; the
   driver kills the process group when a run stops producing output for 20s and
   restarts from the next op, recording the stalled op as `HANG`. That is not
   optional here — `SIGALRM` does not help, because the hang is a tail-call loop
   inside candle with the GIL held, so no Python-level handler ever runs. 27 ops
   in table B and 10 in table D were found this way.

## 2. What upstream's refusal actually is

One shape, transcribed rather than unified:

    NotImplementedError: "<kernel>" not implemented for 'Float8_e4m3fn'

The exception type is `NotImplementedError` in every one of the 114 — checked, not
assumed, for the reason `docs/PROMOTE.md`'s last section exists. The **kernel name
differs per op and carries information**, so it is transcribed per op rather than
folded into a house string:

| distinct kernel name | ops |
|---|---|
| `"add_stub"` | 9 — `aten.add.Scalar`, `aten.add.Tensor`, `aten.add_.Scalar`, `aten.add_.Tensor`, `aten.rsub.Scalar`, `aten.sub.Scalar`, `aten.sub.Tensor`, `aten.sub_.Scalar`, `aten.sub_.Tensor` |
| `"clamp_min_scalar_cpu"` | 4 — `aten.clamp_min.default`, `aten.clamp_min_.default`, `aten.relu.default`, `aten.relu_.default` |
| `"sum_cpu"` | 4 — `aten.mean.default`, `aten.mean.dim`, `aten.sum.default`, `aten.sum.dim_IntList` |
| `"clamp_scalar_cpu"` | 3 — `aten.clamp.default`, `aten.clamp_.default`, `aten.hardtanh.default` |
| `"div_cpu"` | 3 — `aten.div.Tensor`, `aten.div.Tensor_mode`, `aten.div_.Tensor` |
| `"aminmax_cpu"` | 2 — `aten.histc.default`, `aten.multinomial.default` |
| `"ceil_vml_cpu"` | 2 — `aten.ceil.default`, `aten.ceil_.default` |
| `"cos_vml_cpu"` | 2 — `aten.cos.default`, `aten.cos_.default` |
| `"div_cpu_reduced_float"` | 2 — `aten.div.Scalar_mode`, `aten.div_.Scalar` |
| `"erf_vml_cpu"` | 2 — `aten.erf.default`, `aten.erf_.default` |
| `"exp_vml_cpu"` | 2 — `aten.exp.default`, `aten.exp_.default` |
| `"expm1_vml_cpu"` | 2 — `aten.expm1.default`, `aten.expm1_.default` |
| `"ge_cpu"` | 2 — `aten.ge.Scalar`, `aten.ge.Tensor` |
| `"gt_cpu"` | 2 — `aten.gt.Scalar`, `aten.gt.Tensor` |
| `"le_cpu"` | 2 — `aten.le.Scalar`, `aten.le.Tensor` |
| `"log2_vml_cpu"` | 2 — `aten.log2.default`, `aten.log2_.default` |
| `"log_vml_cpu"` | 2 — `aten.log.default`, `aten.log_.default` |
| `"lt_cpu"` | 2 — `aten.lt.Scalar`, `aten.lt.Tensor` |
| `"masked_fill"` | 2 — `aten.masked_fill.Scalar`, `aten.masked_fill_.Scalar` |
| `"neg_cpu"` | 2 — `aten.neg.default`, `aten.neg_.default` |
| `"norm_cpu"` | 2 — `aten.linalg_vector_norm.default`, `aten.norm.ScalarOpt_dim` |
| `"reciprocal_cpu"` | 2 — `aten.reciprocal.default`, `aten.reciprocal_.default` |
| `"remainder_cpu"` | 2 — `aten.remainder.Scalar`, `aten.remainder.Tensor` |
| `"rsqrt_cpu"` | 2 — `aten.rsqrt.default`, `aten.rsqrt_.default` |
| `"scatter_gather_tensor_cpu"` | 2 — `aten.gather.default`, `aten.scatter.src` |
| `"sigmoid_cpu_reduced_float"` | 2 — `aten.sigmoid.default`, `aten.sigmoid_.default` |
| `"sin_vml_cpu"` | 2 — `aten.sin.default`, `aten.sin_.default` |
| `"softmax_lastdim_kernel_impl"` | 2 — `aten._safe_softmax.default`, `aten._softmax.default` |
| `"sqrt_vml_cpu"` | 2 — `aten.sqrt.default`, `aten.sqrt_.default` |
| `"tanh_vml_cpu"` | 2 — `aten.tanh.default`, `aten.tanh_.default` |
| `"GeluKernelImpl"` | 1 — `aten.gelu.default` |
| `"GroupNormKernelImpl"` | 1 — `aten.native_group_norm.default` |
| `"LayerNormKernelImpl"` | 1 — `aten.native_layer_norm.default` |
| `"argmax_cpu"` | 1 — `aten.argmax.default` |
| `"avg_pool2d"` | 1 — `aten.avg_pool2d.default` |
| `"baddbmm"` | 1 — `aten.baddbmm.default` |
| `"batch_norm"` | 1 — `aten.native_batch_norm.default` |
| `"bernoulli_scalar_cpu_"` | 1 — `aten.bernoulli_.float` |
| `"bmm"` | 1 — `aten.bmm.default` |
| `"check_uniform_bounds"` | 1 — `aten.uniform_.default` |
| `"cumsum_out_cpu"` | 1 — `aten.cumsum.default` |
| `"div_floor_cpu"` | 1 — `aten.floor_divide.default` |
| `"div_floor_cpu_reduced_float"` | 1 — `aten.floor_divide.Scalar` |
| `"flash_attention"` | 1 — `aten._scaled_dot_product_flash_attention_for_cpu.default` |
| `"flip_cpu"` | 1 — `aten.flip.default` |
| `"isin_default_cpu"` | 1 — `aten.isin.Tensor_Tensor` |
| `"leaky_relu_cpu"` | 1 — `aten.leaky_relu.default` |
| `"log_softmax_lastdim_kernel_impl"` | 1 — `aten._log_softmax.default` |
| `"max_all"` | 1 — `aten.max.default` |
| `"max_cpu"` | 1 — `aten.max.dim` |
| `"max_pool2d"` | 1 — `aten.max_pool2d.default` |
| `"max_values_cpu"` | 1 — `aten.amax.default` |
| `"maximum_cpu"` | 1 — `aten.max.other` |
| `"min_all"` | 1 — `aten.min.default` |
| `"min_cpu"` | 1 — `aten.min.dim` |
| `"minimum_cpu"` | 1 — `aten.min.other` |
| `"mul_cpu_reduced_float"` | 1 — `aten.mul.Scalar` |
| `"nll_loss_out_frame"` | 1 — `aten.nll_loss_forward.default` |
| `"nonzero_count_cpu"` | 1 — `aten.where.default` |
| `"normal_kernel_cpu"` | 1 — `aten.normal_.default` |
| `"pow"` | 1 — `aten.pow.Tensor_Tensor` |
| `"sign_cpu"` | 1 — `aten.sign.default` |
| `"silu_cpu"` | 1 — `aten.silu.default` |
| `"slow_conv2d_cpu"` | 1 — `aten.convolution.default` |
| `"softplus_cpu"` | 1 — `aten.softplus.default` |
| `"sorting_kernel_method_name"` | 1 — `aten.sort.default` |
| `"topk_cpu"` | 1 — `aten.topk.default` |
| `"tril"` | 1 — `aten.tril.default` |
| `"triu"` | 1 — `aten.triu.default` |
| `"upsample_bilinear2d_channels_last"` | 1 — `aten.upsample_bilinear2d.default` |
| `"weight_norm_kernel"` | 1 — `aten._weight_norm_interface.default` |

71 distinct kernel names across 114 ops. `add`, `sub` and `rsub` all say
`"add_stub"`; `mean` and `sum` both say `"sum_cpu"`; `relu` says
`"clamp_min_scalar_cpu"` and `hardtanh` says `"clamp_scalar_cpu"`. None of that
would survive a unified message, and each of those pairs is a real statement about
which kernel upstream would have dispatched to.

### 2.1 Three refusals that are *not* this shape, and are therefore not in the table

| op | upstream | why it is excluded |
|---|---|---|
| `aten.matmul.default` | `NotImplementedError: "dot" not implemented for 'Float8_e4m3fn'` for 1-D x 1-D, **but computes for 2-D x 2-D** | The refusal is shape-dependent, not op-level. `aten.mm.default` on two 2x2 float8 tensors returns a float8 result upstream. A gate keyed on the op would refuse a call upstream answers |
| `aten.view.dtype` | `RuntimeError: self.size(-1) must be divisible by 4 to view Float8_e4m3fn as Float ...` | Different exception type and a size rule, not a missing kernel. This build already reproduces it verbatim |
| mixed-dtype calls (`add(float8, float32)`) | `RuntimeError: Promotion for Float8 Types is not supported, attempted to promote Float8_e4m3fn and Float` | A **promotion** rule that fires *before* the kernel lookup, and a `RuntimeError`. Emitting the `NotImplementedError` kernel message for a mixed call would be a new divergence in exception type — exactly what `docs/PROMOTE.md` §last warns about |

The gate implemented in §4 is therefore scoped: it fires only when at least one
tensor operand is `float8_e4m3fn` **and no tensor operand is a different
floating-point dtype**. Integer and bool operands (`gather`'s index,
`masked_fill`'s mask) do not block it, because upstream refuses those calls with
the kernel message too — measured, not assumed.

## 3. The enumerated table

### A. upstream refuses, this build computed — the divergence this round closes (48)

| op | probe | upstream 2.13.0 | upstream exception / message | this build (before) | this build's message (before) | after |
|---|---|---|---|---|---|---|
| `aten._weight_norm_interface.default` | generic | REFUSE | NotImplementedError `"weight_norm_kernel" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.add.Scalar` | generic | REFUSE | NotImplementedError `"add_stub" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.add.Tensor` | generic | REFUSE | NotImplementedError `"add_stub" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.argmax.default` | generic | REFUSE | NotImplementedError `"argmax_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.ceil.default` | generic | REFUSE | NotImplementedError `"ceil_vml_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.clamp_min.default` | generic | REFUSE | NotImplementedError `"clamp_min_scalar_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.cos.default` | generic | REFUSE | NotImplementedError `"cos_vml_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.div.Scalar_mode` | generic | REFUSE | NotImplementedError `"div_cpu_reduced_float" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.div.Tensor` | generic | REFUSE | NotImplementedError `"div_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.div.Tensor_mode` | generic | REFUSE | NotImplementedError `"div_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.erf.default` | generic | REFUSE | NotImplementedError `"erf_vml_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.exp.default` | generic | REFUSE | NotImplementedError `"exp_vml_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.flip.default` | generic | REFUSE | NotImplementedError `"flip_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.hardtanh.default` | generic | REFUSE | NotImplementedError `"clamp_scalar_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.leaky_relu.default` | generic | REFUSE | NotImplementedError `"leaky_relu_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.log.default` | generic | REFUSE | NotImplementedError `"log_vml_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.max.default` | generic | REFUSE | NotImplementedError `"max_all" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.max.dim` | generic | REFUSE | NotImplementedError `"max_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.max.other` | generic | REFUSE | NotImplementedError `"maximum_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.mean.default` | generic | REFUSE | NotImplementedError `"sum_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.mean.dim` | generic | REFUSE | NotImplementedError `"sum_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.min.default` | generic | REFUSE | NotImplementedError `"min_all" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.min.dim` | generic | REFUSE | NotImplementedError `"min_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.min.other` | generic | REFUSE | NotImplementedError `"minimum_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.mul.Scalar` | generic | REFUSE | NotImplementedError `"mul_cpu_reduced_float" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.native_group_norm.default` | generic | REFUSE | NotImplementedError `"GroupNormKernelImpl" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.neg.default` | generic | REFUSE | NotImplementedError `"neg_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.reciprocal.default` | generic | REFUSE | NotImplementedError `"reciprocal_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.relu.default` | generic | REFUSE | NotImplementedError `"clamp_min_scalar_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.rsqrt.default` | generic | REFUSE | NotImplementedError `"rsqrt_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.rsub.Scalar` | generic | REFUSE | NotImplementedError `"add_stub" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.sigmoid.default` | generic | REFUSE | NotImplementedError `"sigmoid_cpu_reduced_float" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.sign.default` | generic | REFUSE | NotImplementedError `"sign_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.silu.default` | generic | REFUSE | NotImplementedError `"silu_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.sin.default` | generic | REFUSE | NotImplementedError `"sin_vml_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.sqrt.default` | generic | REFUSE | NotImplementedError `"sqrt_vml_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.sub.Scalar` | generic | REFUSE | NotImplementedError `"add_stub" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.sub.Tensor` | generic | REFUSE | NotImplementedError `"add_stub" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.sum.default` | generic | REFUSE | NotImplementedError `"sum_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.sum.dim_IntList` | generic | REFUSE | NotImplementedError `"sum_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.tanh.default` | generic | REFUSE | NotImplementedError `"tanh_vml_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.tril.default` | generic | REFUSE | NotImplementedError `"tril" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.triu.default` | generic | REFUSE | NotImplementedError `"triu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.where.default` | generic | REFUSE | NotImplementedError `"nonzero_count_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.clamp.default` | hand | REFUSE | NotImplementedError `"clamp_scalar_cpu" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.masked_fill.Scalar` | hand | REFUSE | NotImplementedError `"masked_fill" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.native_batch_norm.default` | hand | REFUSE | NotImplementedError `"batch_norm" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |
| `aten.native_layer_norm.default` | hand | REFUSE | NotImplementedError `"LayerNormKernelImpl" not implemented for 'Float8_e4m3fn'` | WORK |   | refuses (this round) |

### B. upstream refuses, this build hung (27)

| op | probe | upstream 2.13.0 | upstream exception / message | this build (before) | this build's message (before) | after |
|---|---|---|---|---|---|---|
| `aten._log_softmax.default` | generic | REFUSE | NotImplementedError `"log_softmax_lastdim_kernel_impl" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten._safe_softmax.default` | generic | REFUSE | NotImplementedError `"softmax_lastdim_kernel_impl" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten._softmax.default` | generic | REFUSE | NotImplementedError `"softmax_lastdim_kernel_impl" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.cumsum.default` | generic | REFUSE | NotImplementedError `"cumsum_out_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.expm1.default` | generic | REFUSE | NotImplementedError `"expm1_vml_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.expm1_.default` | generic | REFUSE | NotImplementedError `"expm1_vml_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.floor_divide.Scalar` | generic | REFUSE | NotImplementedError `"div_floor_cpu_reduced_float" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.floor_divide.default` | generic | REFUSE | NotImplementedError `"div_floor_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.histc.default` | generic | REFUSE | NotImplementedError `"aminmax_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.isin.Tensor_Tensor` | generic | REFUSE | NotImplementedError `"isin_default_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.linalg_vector_norm.default` | generic | REFUSE | NotImplementedError `"norm_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.log2.default` | generic | REFUSE | NotImplementedError `"log2_vml_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.log2_.default` | generic | REFUSE | NotImplementedError `"log2_vml_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.multinomial.default` | generic | REFUSE | NotImplementedError `"aminmax_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.norm.ScalarOpt_dim` | generic | REFUSE | NotImplementedError `"norm_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.pow.Tensor_Tensor` | generic | REFUSE | NotImplementedError `"pow" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.remainder.Scalar` | generic | REFUSE | NotImplementedError `"remainder_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.remainder.Tensor` | generic | REFUSE | NotImplementedError `"remainder_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.softplus.default` | generic | REFUSE | NotImplementedError `"softplus_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.sort.default` | generic | REFUSE | NotImplementedError `"sorting_kernel_method_name" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.topk.default` | generic | REFUSE | NotImplementedError `"topk_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.avg_pool2d.default` | hand | REFUSE | NotImplementedError `"avg_pool2d" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.gather.default` | hand | REFUSE | NotImplementedError `"scatter_gather_tensor_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.max_pool2d.default` | hand | REFUSE | NotImplementedError `"max_pool2d" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.nll_loss_forward.default` | hand | REFUSE | NotImplementedError `"nll_loss_out_frame" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.scatter.src` | hand | REFUSE | NotImplementedError `"scatter_gather_tensor_cpu" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |
| `aten.upsample_bilinear2d.default` | hand | REFUSE | NotImplementedError `"upsample_bilinear2d_channels_last" not implemented for 'Float8_e4m3fn'` | HANG |   | refuses (this round) |

### C. upstream refuses, this build refused — but often in its own words (42)

| op | probe | upstream 2.13.0 | upstream exception / message | this build (before) | this build's message (before) | after |
|---|---|---|---|---|---|---|
| `aten._scaled_dot_product_flash_attention_for_cpu.default` | generic | REFUSE | NotImplementedError `"flash_attention" not implemented for 'Float8_e4m3fn'` | REFUSE | RuntimeError `aten._scaled_dot_product_flash_attention_for_cpu.default: candle: torch._C shim: transposed copy has no kernel for this candle dtype` | refuses (this round) |
| `aten.add_.Scalar` | generic | REFUSE | NotImplementedError `"add_stub" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.add_.Scalar: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.add_.Tensor` | generic | REFUSE | NotImplementedError `"add_stub" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.add_.Tensor: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.amax.default` | generic | REFUSE | NotImplementedError `"max_values_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | RuntimeError `aten.amax.default: candle: torch._C shim: amax has no kernel for this candle dtype -- tensor.rs::AMax names the ones it reduces` | refuses (this round) |
| `aten.bernoulli_.float` | generic | REFUSE | NotImplementedError `"bernoulli_scalar_cpu_" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `"bernoulli_scalar_cpu_" not implemented for 'Float8_e4m3fn'` | refuses (this round) |
| `aten.ceil_.default` | generic | REFUSE | NotImplementedError `"ceil_vml_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.ceil_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.clamp_min_.default` | generic | REFUSE | NotImplementedError `"clamp_min_scalar_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.clamp_min_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.cos_.default` | generic | REFUSE | NotImplementedError `"cos_vml_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.cos_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.div_.Scalar` | generic | REFUSE | NotImplementedError `"div_cpu_reduced_float" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.div_.Scalar: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.div_.Tensor` | generic | REFUSE | NotImplementedError `"div_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.div_.Tensor: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.erf_.default` | generic | REFUSE | NotImplementedError `"erf_vml_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.erf_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.exp_.default` | generic | REFUSE | NotImplementedError `"exp_vml_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.exp_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.ge.Scalar` | generic | REFUSE | NotImplementedError `"ge_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.ge.Scalar: float8_e4m3fn` | refuses (this round) |
| `aten.ge.Tensor` | generic | REFUSE | NotImplementedError `"ge_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.ge.Tensor: float8_e4m3fn` | refuses (this round) |
| `aten.gelu.default` | generic | REFUSE | NotImplementedError `"GeluKernelImpl" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `"GeluKernelImpl" not implemented for 'Float8_e4m3fn'` | refuses (this round) |
| `aten.gt.Scalar` | generic | REFUSE | NotImplementedError `"gt_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.gt.Scalar: float8_e4m3fn` | refuses (this round) |
| `aten.gt.Tensor` | generic | REFUSE | NotImplementedError `"gt_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.gt.Tensor: float8_e4m3fn` | refuses (this round) |
| `aten.le.Scalar` | generic | REFUSE | NotImplementedError `"le_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.le.Scalar: float8_e4m3fn` | refuses (this round) |
| `aten.le.Tensor` | generic | REFUSE | NotImplementedError `"le_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.le.Tensor: float8_e4m3fn` | refuses (this round) |
| `aten.log_.default` | generic | REFUSE | NotImplementedError `"log_vml_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.log_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.lt.Scalar` | generic | REFUSE | NotImplementedError `"lt_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.lt.Scalar: float8_e4m3fn` | refuses (this round) |
| `aten.lt.Tensor` | generic | REFUSE | NotImplementedError `"lt_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.lt.Tensor: float8_e4m3fn` | refuses (this round) |
| `aten.matmul.default` | generic | REFUSE | NotImplementedError `"dot" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.matmul.default: matmul with a 1-D operand (1D x 1D) is not implemented in torch._C shim -- torch's vector rules were not measured` | unchanged |
| `aten.neg_.default` | generic | REFUSE | NotImplementedError `"neg_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.neg_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.normal_.default` | generic | REFUSE | NotImplementedError `"normal_kernel_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.normal_.default: not implemented in torch._C shim for torch.float8_e4m3fn -- upstream dispatches this op over floating dtypes only, and an integer tensor reaches `random_`, a different op` | refuses (this round) |
| `aten.reciprocal_.default` | generic | REFUSE | NotImplementedError `"reciprocal_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.reciprocal_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.relu_.default` | generic | REFUSE | NotImplementedError `"clamp_min_scalar_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.relu_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.rsqrt_.default` | generic | REFUSE | NotImplementedError `"rsqrt_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.rsqrt_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.sigmoid_.default` | generic | REFUSE | NotImplementedError `"sigmoid_cpu_reduced_float" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.sigmoid_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.sin_.default` | generic | REFUSE | NotImplementedError `"sin_vml_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.sin_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.sqrt_.default` | generic | REFUSE | NotImplementedError `"sqrt_vml_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.sqrt_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.sub_.Scalar` | generic | REFUSE | NotImplementedError `"add_stub" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.sub_.Scalar: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.sub_.Tensor` | generic | REFUSE | NotImplementedError `"add_stub" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.sub_.Tensor: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.tanh_.default` | generic | REFUSE | NotImplementedError `"tanh_vml_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.tanh_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.uniform_.default` | generic | REFUSE | NotImplementedError `"check_uniform_bounds" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.uniform_.default: not implemented in torch._C shim for torch.float8_e4m3fn -- upstream dispatches this op over floating dtypes only, and an integer tensor reaches `random_`, a different op` | refuses (this round) |
| `aten.view.dtype` | generic | REFUSE | RuntimeError `self.size(-1) must be divisible by 4 to view Float8_e4m3fn as Float (different element sizes), but got 2` | REFUSE | RuntimeError `self.size(-1) must be divisible by 4 to view Float8_e4m3fn as Float (different element sizes), but got 2` | unchanged |
| `aten._grouped_mm.default` | hand | REFUSE | RuntimeError `Expected mat_a to be Float32, BFloat16 or Float16 matrix, got Float8_e4m3fn` | REFUSE | RuntimeError `Expected mat_a to be Float32, BFloat16 or Float16 matrix, got Float8_e4m3fn` | unchanged |
| `aten.baddbmm.default` | hand | REFUSE | NotImplementedError `"baddbmm" not implemented for 'Float8_e4m3fn'` | REFUSE | RuntimeError `aten.baddbmm.default: candle: unsupported dtype F8E4M3 for op matmul` | refuses (this round) |
| `aten.bmm.default` | hand | REFUSE | NotImplementedError `"bmm" not implemented for 'Float8_e4m3fn'` | REFUSE | RuntimeError `aten.bmm.default: candle: unsupported dtype F8E4M3 for op matmul` | refuses (this round) |
| `aten.clamp_.default` | hand | REFUSE | NotImplementedError `"clamp_scalar_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.clamp_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |
| `aten.convolution.default` | hand | REFUSE | NotImplementedError `"slow_conv2d_cpu" not implemented for 'Float8_e4m3fn'` | REFUSE | RuntimeError `aten.convolution.default: candle: unsupported dtype F8E4M3 for op matmul` | refuses (this round) |
| `aten.masked_fill_.Scalar` | hand | REFUSE | NotImplementedError `"masked_fill" not implemented for 'Float8_e4m3fn'` | REFUSE | NotImplementedError `aten.masked_fill_.Scalar: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | refuses (this round) |

### D. upstream computes, this build hangs — no upstream refusal to transcribe (10)

| op | probe | upstream 2.13.0 | upstream exception / message | this build (before) | this build's message (before) | after |
|---|---|---|---|---|---|---|
| `aten.adaptive_avg_pool1d.default` | generic | WORK |   | HANG |   | refuses (this round, shim wording) |
| `aten.all.default` | generic | WORK |   | HANG |   | refuses (this round, shim wording) |
| `aten.all.dim` | generic | WORK |   | HANG |   | refuses (this round, shim wording) |
| `aten.all.dims` | generic | WORK |   | HANG |   | refuses (this round, shim wording) |
| `aten.any.default` | generic | WORK |   | HANG |   | refuses (this round, shim wording) |
| `aten.any.dim` | generic | WORK |   | HANG |   | refuses (this round, shim wording) |
| `aten.pow.Scalar` | generic | WORK |   | HANG |   | refuses (this round, shim wording) |
| `aten.pow.Tensor_Scalar` | generic | WORK |   | HANG |   | refuses (this round, shim wording) |
| `aten.fill_.Tensor` | hand | WORK |   | HANG |   | refuses (this round, shim wording) |
| `aten.index_put_.default` | hand | WORK |   | HANG |   | refuses (this round, shim wording) |

### E. upstream computes, this build refuses — pre-existing gaps, unchanged (13)

| op | probe | upstream 2.13.0 | upstream exception / message | this build (before) | this build's message (before) | after |
|---|---|---|---|---|---|---|
| `aten._local_scalar_dense.default` | generic | WORK |   | REFUSE | RuntimeError `a Tensor with 2 elements cannot be converted to Scalar` | unchanged |
| `aten.abs_.default` | generic | WORK |   | REFUSE | NotImplementedError `aten.abs_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | unchanged |
| `aten.copy_.default` | generic | WORK |   | REFUSE | NotImplementedError `aten.copy_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | unchanged |
| `aten.eq.Scalar` | generic | WORK |   | REFUSE | NotImplementedError `aten.eq.Scalar: float8_e4m3fn` | unchanged |
| `aten.eq.Tensor` | generic | WORK |   | REFUSE | NotImplementedError `aten.eq.Tensor: float8_e4m3fn` | unchanged |
| `aten.fill_.Scalar` | generic | WORK |   | REFUSE | NotImplementedError `aten.fill_.Scalar: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | unchanged |
| `aten.mul_.Scalar` | generic | WORK |   | REFUSE | NotImplementedError `aten.mul_.Scalar: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | unchanged |
| `aten.mul_.Tensor` | generic | WORK |   | REFUSE | NotImplementedError `aten.mul_.Tensor: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | unchanged |
| `aten.ne.Scalar` | generic | WORK |   | REFUSE | NotImplementedError `aten.ne.Scalar: float8_e4m3fn` | unchanged |
| `aten.ne.Tensor` | generic | WORK |   | REFUSE | NotImplementedError `aten.ne.Tensor: float8_e4m3fn` | unchanged |
| `aten.zero_.default` | generic | WORK |   | REFUSE | NotImplementedError `aten.zero_.default: torch._C shim cannot write through a view of candle dtype F8E4M3 -- tensor.rs::flat_storage names the dtypes it can read, and this is not one of them` | unchanged |
| `aten.addmm.default` | hand | WORK |   | REFUSE | RuntimeError `aten.addmm.default: candle: unsupported dtype F8E4M3 for op matmul` | unchanged |
| `aten.mm.default` | hand | WORK |   | REFUSE | RuntimeError `aten.mm.default: candle: unsupported dtype F8E4M3 for op matmul` | unchanged |

### F. upstream computes, this build computes (50)

| op | probe | upstream 2.13.0 | upstream exception / message | this build (before) | this build's message (before) | after |
|---|---|---|---|---|---|---|
| `aten._to_copy.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.abs.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.alias.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.arange.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.arange.start` | generic | WORK |   | WORK |   | unchanged |
| `aten.arange.start_step` | generic | WORK |   | WORK |   | unchanged |
| `aten.cat.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.clone.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.detach.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.empty.memory_format` | generic | WORK |   | WORK |   | unchanged |
| `aten.empty_like.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.expand.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.full.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.full_like.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.is_floating_point.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.lift_fresh.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.linspace.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.mul.Tensor` | generic | WORK |   | WORK |   | unchanged |
| `aten.native_dropout.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.new_ones.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.new_zeros.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.ones.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.ones_like.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.permute.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.randint.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.randperm.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.repeat.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.scalar_tensor.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.select.int` | generic | WORK |   | WORK |   | unchanged |
| `aten.slice.Tensor` | generic | WORK |   | WORK |   | unchanged |
| `aten.squeeze.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.squeeze.dim` | generic | WORK |   | WORK |   | unchanged |
| `aten.squeeze.dims` | generic | WORK |   | WORK |   | unchanged |
| `aten.stack.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.t.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.transpose.int` | generic | WORK |   | WORK |   | unchanged |
| `aten.unsqueeze.default` | generic | WORK |   | WORK |   | unchanged |
| `aten.zeros_like.default` | generic | WORK |   | WORK |   | unchanged |
| `aten._unsafe_view.default` | hand | WORK |   | WORK |   | unchanged |
| `aten.constant_pad_nd.default` | hand | WORK |   | WORK |   | unchanged |
| `aten.embedding.default` | hand | WORK |   | WORK |   | unchanged |
| `aten.index.Tensor` | hand | WORK |   | WORK |   | unchanged |
| `aten.masked_select.default` | hand | WORK |   | WORK |   | unchanged |
| `aten.reshape.default` | hand | WORK |   | WORK |   | unchanged |
| `aten.split.Tensor` | hand | WORK |   | WORK |   | unchanged |
| `aten.split_with_sizes.default` | hand | WORK |   | WORK |   | unchanged |
| `aten.unbind.int` | hand | WORK |   | WORK |   | unchanged |
| `aten.view.default` | hand | WORK |   | WORK |   | unchanged |
| `aten.where.ScalarOther` | hand | WORK |   | WORK |   | unchanged |
| `aten.where.self` | hand | WORK |   | WORK |   | unchanged |

### G. not probed (7 of 197)

| op | why the probe could not build a call |
|---|---|
| `aten.bitwise_and.Scalar` | "bitwise_and_cpu" not implemented for 'Float' |
| `aten.bitwise_and.Tensor` | "bitwise_and_cpu" not implemented for 'Float' |
| `aten.bitwise_not.default` | "bitwise_not_cpu" not implemented for 'Float' |
| `aten.bitwise_or.Scalar` | "bitwise_or_cpu" not implemented for 'Float' |
| `aten.bitwise_or.Tensor` | "bitwise_or_cpu" not implemented for 'Float' |
| `aten.one_hot.default` | one_hot is only applicable to index tensor of type LongTensor. |
| `aten.randint.low` | random_ expects 'from' to be less than 'to', but got from=1 >= to=1 |

## 4. What this round changes

A single gate at the single door (`aten_dispatch`, `rust/torch_c/src/aten.rs`),
before `aten_dispatch_inner`, keyed on a static table of 114 `(op, kernel-name)`
pairs. On a hit it raises

    NotImplementedError: "<kernel>" not implemented for 'Float8_e4m3fn'

which is upstream's text and upstream's exception type, character for character.

The gate is at the door rather than in 114 kernels for the reason the device gate
is: a kernel can forget, and the 27 ops in table B never reach a kernel at all —
they disappear into candle's `F8E4M3 -> f64` tail-call loop first. Only the door
runs before that.

**Cost.** The table lookup is a `match` on `&str` that misses for every dtype but
this one, and the operand scan runs only after the table hits. Nothing on the
float32 path scans anything new.

### 4.1 The ten this build cannot settle against an oracle

Table D: upstream **computes** these and this build **hangs**. There is no upstream
refusal to transcribe, so refusing them is a divergence in the conservative
direction — the same trade `docs/FLOAT8.md` made for `tolist`, `item` and the
comparisons, and made for the same underlying reason: candle 0.11.0's
`WithDType for f8e4m3::to_f64` recurses into itself, and release-mode LLVM turns
that into `.L1: jmp .L1`.

    aten.adaptive_avg_pool1d.default  aten.all.default   aten.all.dim
    aten.all.dims                     aten.any.default   aten.any.dim
    aten.pow.Scalar                   aten.pow.Tensor_Scalar
    aten.fill_.Tensor                 aten.index_put_.default

They refuse with a message that says so, and does **not** borrow upstream's
wording — claiming `"pow" not implemented for 'Float8_e4m3fn'` for
`aten.pow.Scalar` would be a lie, because upstream implements it:

    NotImplementedError: aten.pow.Scalar: float8_e4m3fn is not supported by this
    op in the torch._C shim (candle's F8E4M3 -> f64 conversion does not
    terminate); upstream computes this -- docs/FLOAT8B.md 4.1

Fixing them properly means patching `candle-core`, which is a vendored-dependency
change and a different round. Recorded and refused, the way `docs/DEMAND1.md` did
for the two `native_batch_norm` cases where upstream is not an oracle.

### 4.2 Table E: upstream computes, this build refuses — left alone

Thirteen ops refuse here while upstream answers: `eq`/`ne` (`.Scalar` and
`.Tensor`), `_local_scalar_dense`, the in-place writers
`abs_`/`copy_`/`fill_.Scalar`/`mul_`/`zero_`, and `mm`/`addmm` (candle has no
`F8E4M3` matmul). These are `docs/FLOAT8.md`'s own refusals plus two pre-existing
families -- "cannot write through a view of candle dtype F8E4M3" and "unsupported
dtype F8E4M3 for op matmul". They are gaps in the safe direction, they are not
what this round was asked to close, and closing them needs write-through and a
matmul kernel, not a message change.

Two of them are now **held in place by the golden suite** rather than only
described here: `fill_.Scalar` and `zero_` are `expect="c_error"` cases, which
fail if the gap silently closes as well as if it silently widens.

## 5. Is the dtype now includable in the golden suite?

`tools/golden/dtypes.py` excludes `float8_e4m3fn` with this reason:

> `torch.tensor(..., dtype=torch.float8_e4m3fn)` and
> `_C._tensor_from_flat(..., dtype=_C.float8_e4m3fn)` both hang indefinitely when
> probed independently on this host (observed 2026-08-24).

**That reason is stale and was already stale when `docs/FLOAT8.md` was written** —
its path table records construction as WORK on both sides. The hang was never in
construction; it was in `to_dtype(F64)`, which construction does not call.

### 5.1 What actually blocked it, once the reason above was gone

Including the dtype surfaced a second blocker the stale reason had been hiding:
**the harness reads every result with `.tolist()`, and this shim refuses
`.tolist()` for `float8_e4m3fn`** (docs/FLOAT8.md, and still true -- the
underlying candle bug is not fixed). Every case where both sides computed
correctly would have died reading the answer.

`compare.py` now reads float8 results through a lossless widening to `float32`
(`_as_list`). `float8_e4m3fn` has 4 exponent and 3 mantissa bits, so every finite
value of it is exactly representable in `float32`: the widening is not a
tolerance, and it is applied to **both** sides by the same route, so it cannot
hide a disagreement between them. The `float32` constant is looked up by dtype
*type* from the owning module -- `torch.float32` for one side, `_C.float32` for
the other -- because docs/TORCH_C.md §1's point is that those two are not
interchangeable.

**Answer: yes, and this round includes it.** The harness has the vocabulary
already — `expect="both_error"` for a refusal both sides agree on, and a
`SILENT DIVERGENCE` failure for the case where one side refuses and the other
computes, which is precisely the shape this document is about. What made the dtype
uncheckable was that 48 ops computed unverifiable answers and 37 hung; after §4
neither is true, so every float8 case either matches upstream's value or matches
upstream's refusal.

## 6. What this document does not establish

| # | not established | why |
|---|---|---|
| 1 | That 114 is the whole refusal set | 7 of 197 ops still have no probe at all (table G, minus the 32 hand-written): `_grouped_mm`, the five `bitwise_*`, and `randint.low`. The `bitwise_*` family refuses `'Float'` upstream too, so `float8` never reaches a dtype check there; `randint.low` takes no float operand |
| 2 | That the transcribed kernel name is right for every shape | `aten.upsample_bilinear2d.default` reports `"upsample_bilinear2d_channels_last"`, which names a memory format. A channels-first input may well produce a different kernel name, and that was not probed. Every other name in the table came from a call whose only unusual property was the dtype |
| 3 | That the gate is right for mixed float8/float16 operands | §2.1 measured float8 against `float32` only. `float16` and `bfloat16` were assumed to take the same promotion path |
| 4 | Anything about a device other than CPU | One host, one device, macOS |

## 7. Verification

### 7.1 The gate is load-bearing

`docs/DEMAND1.md`'s rule -- a verification that cannot fail is not a verification
-- applied to the new path. The gate call at the door was replaced with
`let _ = float8_e4m3fn_gate(...)`, the crate rebuilt, and the six new tests run
one per subprocess under a 90-second wall clock:

| test | gate on | gate disabled |
|---|---|---|
| `..._transcribes_upstreams_wording_per_op...` | PASS | **FAIL** `('aten.add.Tensor', None, 'NO-RAISE')` |
| `..._refuses_every_op_on_the_table...` | PASS | **HANG** (killed at 90s) |
| `..._shim_only_refusals_do_not_borrow...` | PASS | **HANG** (killed at 90s) |
| `..._matmul_refuses_the_shape_upstream_refuses...` | PASS | **FAIL** (falls back to the shim's own matmul message) |
| `..._still_computes_everything_upstream_computes` | PASS | PASS — correctly: it asserts the gate does *not* touch `mul`/`abs`/`clone`/`cat` |
| `..._gate_does_not_fire_on_a_mixed_dtype_call` | PASS | PASS — correctly: it asserts the gate does *not* fire |

The two that still pass are the two whose subject is the gate's *restraint*, so
passing without the gate is the right answer for them and not a blind spot.
The tree was restored from a `cp` backup and rebuilt.

### 7.2 Gates

    rust/torch_c/pytests/run.sh   EXIT=0, 379 ok, DOCWATCH: PASS -- 316/316
    tools/golden/compare.py       EXIT=0, SUMMARY: 8465/8465 cases passed,
                                  0 failed, ops covered=197
    tools/golden/compare.py --self-test
                                  SELF-TEST: PASS -- 19 comparators x 11 fault
                                  modes, 0 problem(s)

Six of those `ok` lines are new, and they nearly were not. `test_shim.py` ended
with its `if __name__ == "__main__": raise SystemExit(_main())` block, and
`_main()` collects tests by walking `globals()` -- so **anything appended after
that block is defined only after the collection has already run**. The first
suite run with the six new tests present reported 373, exactly as before, and
green. The block is now last in the file. That is `docs/DEMAND1.md`'s rule again
in a new place: a test that cannot fail is not a test, and one that is never
collected cannot fail.

`ops covered` is unchanged at 197, as it should be: refusing a dtype inside an op
does not un-implement the op. Case count rose 8436 -> 8465 (+29), which is what
including a tenth dtype in the builders that loop over `DEFAULT_DTYPES` adds.

### 7.3 `docs/FLOAT8.md`'s path table still holds

Re-run against this build, both sides. Every row is as `docs/FLOAT8.md` recorded
it, and nothing hangs:

| path | shim | upstream |
|---|---|---|
| construction | WORK | WORK |
| `.to(float8_e4m3fn)` | WORK | WORK |
| `t + t` | `NotImplementedError: "add_stub" not implemented for 'Float8_e4m3fn'` | **identical** |
| `t1 == t2` | REFUSE | WORK |
| `print`/`repr` | WORK | WORK |
| `.tolist()` | REFUSE | WORK |
| `.item()` | REFUSE | WORK |
| `t[0]` | WORK | WORK |
| `cat`/`stack` | WORK | WORK |
| `matmul` (1-D) | `NotImplementedError: "dot" not implemented for 'Float8_e4m3fn'` | **identical** |
| `mul` `abs` `clone` | WORK | WORK |

Two rows moved *toward* upstream: `t + t` and `matmul` now carry upstream's text
and type rather than merely being refusals of some kind.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs FLOAT8_E4M3FN_REFUSALS present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs float8_shim_only_refusal present -->
<!-- DOCWATCH: symbol-in-file tools/golden/dtypes.py float8_e4m3fn present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py _FLOAT8_TRANSCRIBED present -->

---

## 6. Mixed dtype: two of three closed on review, one named

`promote_operands` and `promote_list` now raise upstream's own refusal when a
float8 operand meets a different dtype:

    RuntimeError: Promotion for Float8 Types is not supported,
                  attempted to promote Float8_e4m3fn and Float

Type and text, character for character, and it fires *before* kernel lookup the
way upstream's does. Before this it was `NotImplementedError: dtype promotion
not implemented in torch._C shim` -- the right idea under the wrong exception,
which is precisely the shape docs/PROMOTE.md's last section exists to prevent.

| pair | upstream | here |
|---|---|---|
| `f8 + float32` | `RuntimeError: Promotion for Float8 Types…` | same |
| `f8 + float64` | `RuntimeError: Promotion for Float8 Types…` | same |
| **`f8 + int64`** | `RuntimeError: Promotion for Float8 Types…` | `NotImplementedError: "add_stub" not implemented…` |

### 6.1 Why the third one is not a one-line fix

`float8_only_floats` blocks the kernel gate on a second *floating* operand and
not on an integer one, and that is deliberate and right for most of the table:
`gather`'s index and `masked_fill`'s mask are integer and bool, take no part in
promotion, and upstream does give the kernel message for those.

The distinction is not "is the other operand floating" but **"does the other
operand participate in this op's type promotion"** -- true for `add`'s second
argument, false for `gather`'s index. Getting it right means threading the set of
promoting ops into the gate, which the gate deliberately does not know: it is a
table lookup at the door, ahead of 114 kernels, and giving it per-op promotion
semantics makes it a second dispatcher.

So the pair refuses either way and the message is wrong for the integer case.
Recorded rather than half-fixed: a predicate that blocked on *any* second dtype
would close this row and reopen `gather` and `masked_fill`, which is a worse
trade than the one it fixes.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs float8_promotion_refusal present -->
