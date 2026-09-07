# A backward pass: what it needs, how big it is, and whether abi3 allows it

> **§6.6's recommendation was taken and steps 3, 4 and 5 have landed —
> [`docs/training/BACKWARD.md`](../training/BACKWARD.md).** A tape over a captured trace runs a full SmolLM2-135M
> backward and one SGD step; all 134,515,008 gradient elements were compared against upstream.
> Four things below are now wrong or narrower than they read, and each is corrected in place:
> §4/§5's kernel bill (a *rule* is not a kernel, so **zero** new kernels were needed — §5.1),
> SDPA's backward as "the one genuinely new kernel" (it stopped nothing — BACKWARD.md §3.4),
> §7's argument for a read-only `.grad` (its antecedent is gone — BACKWARD.md §10), and §8 item 1
> ("that a tape-based backward is numerically correct" — it is, to a median relative L2 of
> 8.5e-07 in `float64`).

`docs/training/TRAIN.md` closes by naming this as "the next wall and much larger than this one". Training
mode landed — all 26 architectures forward in `.train()` and agree with upstream draw for draw —
but every one of those runs inside `torch.no_grad()`, which isolates the mode axis and leaves the
harder half untouched. README §2 and §3 sell federated learning and test-time adaptation. A
federated round is *forward, backward, optimiser step, aggregate*. We have the first.

**This is an investigation, not an implementation.** No kernel was written, `rust/torch_c/src/` is
untouched, and the vendored tree was not modified. The whole diff is this document plus **one
test** that pins what §1 measured (§10). Everything below is either a command that was run, or is
labelled as coming from a stub.

Environment: `/Volumes/macMini/caches/spike-venv/bin/python`, torch 2.13.0, transformers 5.15.1,
worktree at `develop` `f83f94c`. Upstream C++ source read from
`/Volumes/macMini/caches/pytorch-spike/pytorch` (the same cache `docs/graph/DYNAMO.md` §15 used).

The gates, before this work and after it:

| gate | before | after |
|---|---|---|
| `pytests/run.sh` | 285 ok, 0 FAIL | **286 ok, 0 FAIL** (§10's test) |
| `run.sh` DOCWATCH | 43/43 | **59/59** (16 new markers, all in this document) |
| `tools/golden/compare.py` | 6587/6587, ops=163, pending 1 | unchanged |
| `verify_schemas.py` | 4465/4465 | unchanged |

`ops=163` is unchanged on purpose: no kernel landed, so nothing here could have moved it.

---

## 0. The answer, before the evidence

| question | answer |
|---|---|
| **Is autograd reachable under abi3?** | **Yes.** `torch/csrc/autograd/` contains **zero** `Py_BUILD_CORE` and **zero** `internal/pycore_*` includes, and `engine.cpp` contains **zero** occurrences of `Py`/`PyObject` of any kind. This is the opposite of the Dynamo verdict, measured with the same command (§3). |
| Where does it stop today? | Two walls, both explicit refusals: `torch.randn(requires_grad=True)` refuses at the factory, and `Tensor.backward()` refuses at `_ImperativeEngine.run_backward`. Between them, nothing: `(x*x).sum().requires_grad` is `False`, so no graph is ever built (§1). |
| How many derivative formulas? | **122** of the shim's 163 ops have one upstream. **66 trivial, 31 composition, 25 need their own kernel** — and those 25 reach **24 distinct backward ops, of which 14 are real CPU kernels and 10 are composites that decompose** (§4). |
| Cheapest useful subset? | A real SmolLM2-135M training step's backward touches **24 aten ops, 16 of which the shim already has**. The 8 missing ones are mostly cheap; **exactly one** (`_scaled_dot_product_flash_attention_for_cpu_backward`) is a real kernel with no decomposition. **LoRA removes exactly one op from that list** — it saves optimiser state, not kernels (§5). |
| Tape over capture, or `VariableType`? | **Tape over capture**, decisively. The capture path already works under abi3, already decomposes to Core ATen, and turns "163 ops × a graph-construction wrapper" into "one recorder" (§6). |
| Does anything already work? | The flag plumbing does, and further than the comments claim — but it is inert by construction, and §1.3 demonstrates exactly where the inertness begins. |

The single most valuable sentence: **autograd is not blocked by abi3 the way `torch.compile` is.**
`docs/graph/DYNAMO.md` §15 found Dynamo needs `_PyInterpreterFrame` through `Py_BUILD_CORE`, a struct
whose layout changes every CPython minor version, which is precisely what a single abi3 wheel
cannot carry. Autograd has no equivalent. It is a graph scheduler over C++ objects; the only
CPython it touches is the binding layer, which this shim already writes in pyo3.

That makes this a **finite engineering problem with a known shape**, in the same category as the
kernel work already done — not a "do we abandon abi3" decision. §4 and §5 give the size.

---

## 1. Where it stops today

### 1.1 The literal command from the brief

```
$ PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 python -c \
    "import torch; x = torch.randn(4, requires_grad=True); (x*x).sum().backward()"
```

```
File "torch_c_bootstrap.py", line 2368, in _strip_python_only_kwargs
NotImplementedError: not implemented in torch._C shim: torch.empty(requires_grad=True)
  -- there is no autograd behind this shim, and returning a tensor that quietly records
  nothing would be worse than refusing
```

**Wall 1**, and it is a deliberate one. `bootstrap.py:2356` (`_strip_python_only_kwargs`) refuses
`requires_grad=True` by name rather than dropping it, on the stated ground that handing back a
tensor that silently records nothing is worse than refusing. That judgement is correct and this
document does not propose changing it.

### 1.2 Stubbing past wall 1

`requires_grad` is *settable*; only the factory keyword refuses. So `torch.randn(4).requires_grad_(True)`
is the same tensor without the refusal, and needs no stub at all:

```
$ ... python -c "import torch
x = torch.randn(4).requires_grad_(True)
print(x.requires_grad, x.is_leaf, x.grad_fn)
y = (x*x).sum(); print(y.requires_grad, y.grad_fn)
y.backward()"

True True None
False None
...
File "torch/autograd/graph.py", line 976, in _engine_run_backward
    torch._C._stash_obj_in_tls("context", contextvars.copy_context())
NotImplementedError: not implemented in torch._C shim: torch._C._stash_obj_in_tls
```

**Wall 2**, and it is incidental — `_stash_obj_in_tls` stores a `contextvars.Context` in a C++
thread-local so device threads can see the compiler config. For a single-threaded CPU backward a
dict is the entire observable contract. Stubbed in `/tmp/ag/stubs.py` (**this result is from a
stub**):

```
File "torch/autograd/graph.py", line 979, in _engine_run_backward
    return Variable._execution_engine.run_backward(...)
NotImplementedError: not implemented in torch._C shim: _ImperativeEngine.run_backward
```

**Wall 3**, and this one is the whole thing. `torch._C._ImperativeEngine` exists in the shim's
surface with exactly three methods, all refusing:

```
$ ... python -c "import torch; print(sorted(a for a in dir(torch._C._ImperativeEngine())
                                            if not a.startswith('__')))"
['is_checkpoint_valid', 'queue_callback', 'run_backward']
```

The climb stops here, and stubbing further would be dishonest rather than informative — see §1.3.

### 1.3 The finding that matters more than the walls

Look again at the second line of §1.2's output:

```
y.requires_grad = False        y.grad_fn = None
```

**The flag does not propagate through a single operation.** `x` is a leaf with `requires_grad=True`;
`x*x` comes back with `requires_grad=False`. So there is no graph for wall 3's engine to traverse
even if the engine existed. `bootstrap.py:4087` (`_install_autograd_shape`) says this in its own
docstring and is accurate:

> `requires_grad` stores and reports what was set. **Nothing reads it.**

This is why stubbing past wall 3 was not attempted. A stub for `run_backward` would have to invent
the graph as well as the engine — at which point the "measurement" would be measuring the stub. The
two requirements are cleanly separable and both are missing:

| | what it is | present? |
|---|---|---|
| **graph construction** | every op, on the way out, records a node and links it to its inputs | **no** — `requires_grad` is inert |
| **graph execution** | walk that graph in reverse-topological order, accumulate into leaves | **no** — `run_backward` refuses |

`is_leaf` is hardcoded `True` and `grad_fn`/`grad` are hardcoded `None`
(`bootstrap.py:4165-4167`), which is the honest report of that state: no node was ever created and
no gradient was ever accumulated.

### 1.4 The higher rungs

The brief asks for `nn.Linear` and then a transformer block. Both reach **the same wall 3 in the
same place**, because wall 3 is upstream of anything model-shaped — `nn.Linear(8,4)(x).sum()` and a
`LlamaDecoderLayer`'s output both arrive at `_engine_run_backward` with `requires_grad=False`
and no `grad_fn`. Climbing them on the shim therefore yields no new information.

So the requirement for those rungs was measured **on upstream instead**, where the graph really is
built, and then differenced against the shim's op set. That is §4 and §5, and it gives the whole
requirement rather than the first item — which is what the brief actually asked for.

<!-- DOCWATCH: op-implemented aten.mul.Tensor -->
<!-- DOCWATCH: op-implemented aten.expand.default -->
<!-- DOCWATCH: op-implemented aten.ones_like.default -->

---

## 2. Which layer autograd actually lives in

Upstream's autograd is four things, and the brief is right that they are not one. Measured on
upstream, not recalled:

### 2.1 Graph construction is a generated C++ kernel per op, at the Autograd dispatch key

```
$ python -c "import torch
print(torch._C._dispatch_key_set(torch.randn(4, requires_grad=True)))
for l in torch._C._dispatch_dump('aten::mul.Tensor').splitlines():
    if 'utograd' in l: print(l.strip())"

DispatchKeySet(CPU, ADInplaceOrView, AutogradCPU, AutocastCPU)
Autograd[alias]: registered at .../torch/csrc/autograd/generated/VariableType_0.cpp:10502
CompositeExplicitAutogradNonFunctional[alias]: registered at .../RegisterCompositeExplicitAutogradNonFunctional_0.cpp:7305
```

`native_layer_norm` reports the same shape at `VariableType_6.cpp:8660`. So for **every**
differentiable op there is a *generated* wrapper registered above the CPU kernel, whose job is to
allocate a `Node`, save what the formula needs, run the real kernel, and attach the node to the
output. This is the "wrap every op" cost, and it is the part `VariableType` names.

### 2.2 The nodes are C++ classes generated from `derivatives.yaml`, one per formula

```
$ python /tmp/ag/graph_walk.py llama_block
=== llama_block  (95 nodes, 111 edges)
    grad_fn type: <class 'SumBackward0'>   module=builtins
    base classes: ['SumBackward0', 'object']
       13 ViewBackward0    11 UnsafeViewBackward0   9 AccumulateGrad   9 MulBackward0
        7 MmBackward0       7 TBackward0            6 ExpandBackward0  5 AddBackward0
        5 TransposeBackward0 4 SliceBackward0       3 CloneBackward0   2 BmmBackward0
        2 UnsqueezeBackward0 2 ReshapeAliasBackward0 2 CatBackward0    2 NegBackward0
        1 SumBackward0  1 RsqrtBackward0  1 MeanBackward1  1 PowBackward0
        1 SoftmaxBackward0  1 SiluBackward0
```

`module=builtins` and a two-entry MRO: these are C++ types, not Python classes. They live in
`torch/csrc/autograd/generated/Functions.cpp`, generated from `derivatives.yaml`. One transformer
block builds **95 nodes across 22 distinct classes**.

### 2.3 The engine is Python-free

```
$ grep -c "Py_\|PyObject\|PyGILState" torch/csrc/autograd/engine.cpp
0
```

`engine.cpp` (1862 lines) includes `<torch/csrc/autograd/*.h>`, `<ATen/*>` and `<c10/*>` and **no
Python header at all**. It is a work-queue scheduler: ready queue, dependency counts,
`InputBuffer` accumulation, reentrancy depth. Python enters only through `python_engine.cpp`
(501 lines), which is the binding.

### 2.4 `torch.autograd.Function` is not on this path

None of the 95 nodes in §2.2 is a `PyNode`. The Python-side `Function` mechanism
(`torch/csrc/autograd/python_function.cpp`) is for user-defined ops and is bypassed entirely by a
backward over built-in ops. It is vendored already (`torch/autograd/function.py`) and needs
nothing new.

### 2.5 So: which of the four would this shim have to provide?

| layer | upstream | this shim would need |
|---|---|---|
| `VariableType` generated kernels | `torch/csrc/autograd/generated/VariableType_*.cpp` | **yes — or an equivalent.** §6 argues the capture path is that equivalent and is far cheaper |
| `AutogradMeta` on every tensor | `c10::TensorImpl::autograd_meta_`, `autograd_meta.cpp` (319 lines) | **yes** — a `grad_fn`/`grad`/`output_nr` triple on `TensorBase`; the slot already exists inertly |
| the C++ engine | `engine.cpp` (1862 lines), `input_buffer.cpp` (399), `graph_task.h` (231) | **yes**, but a single-threaded CPU version is a small fraction of that — most of those lines are device streams, reentrancy, and multi-threaded queues |
| Python `autograd.Function` | `python_function.cpp` | **no** — not traversed (§2.4) |
| derivative formulas | `derivatives.yaml` (687 entries) + `FunctionsManual.cpp` (8765 lines) | **partly** — §4 counts exactly how much |

The vendored tree contains **no** `torch/csrc/autograd/` at all (`torchnative/src/main/torch/csrc/`
holds only `inductor`), so none of this arrives for free the way the Python-level `torch/autograd/`
package does.

---

## 3. The abi3 question — the same method as DYNAMO.md §15, opposite result

`docs/graph/DYNAMO.md` §15 judged `torch.compile` unreachable under the limited API and that finding
changed the roadmap. The method there was: grep the upstream C sources for `Py_BUILD_CORE` and
internal headers, because `Py_LIMITED_API` and `Py_BUILD_CORE` cannot coexist in one extension.
Applied unchanged to autograd:

```
$ P=.../pytorch/torch/csrc
$ grep -rl "Py_BUILD_CORE" $P/dynamo/
dynamo/cpython_includes.h   dynamo/framelocals_mapping.cpp   dynamo/eval_frame.c
dynamo/cpython_defs.c       dynamo/stackref_bridge.c         dynamo/guards.cpp
$ grep -rl "Py_BUILD_CORE" $P/autograd/
                                                          <-- nothing

$ grep -rl "internal/pycore" $P/dynamo/
dynamo/cpython_includes.h   dynamo/eval_frame.c   dynamo/framelocals_mapping.cpp
dynamo/eval_frame_cpp.cpp   dynamo/cpython_defs.c dynamo/stackref_bridge.c  dynamo/guards.cpp
$ grep -rl "internal/pycore" $P/autograd/
                                                          <-- nothing
```

**Six files versus zero, and seven versus zero.** The grep covers `torch/csrc/autograd/` including
`generated/`, 129 files and 284579 lines.

### 3.1 What CPython surface autograd does use

```
$ grep -rh '#include *[<"]\(Python\.h\|structmember\.h\|frameobject\.h\|...\)' $P/autograd/ \
    | sort | uniq -c | sort -rn
  17 #include <Python.h>
   2 #include <structmember.h>
   1 #include <frameobject.h>
```

Three things to say about the two non-`Python.h` entries, and both turn out to be clear:

* `frameobject.h` appears **once**, in `profiler_python.cpp` — the Python-stack profiler, which is
  an optional observability feature and is not on any backward path. Dropping it costs a profiler,
  not a gradient.
* `structmember.h` appears twice (`python_variable.cpp`, `python_function.cpp`). It is CPython's own
  deprecated alias header; its own comment says *"New definitions are in descrobject.h"*. And
  `PyMemberDef` in `descrobject.h` is declared at line 41, **above** that header's
  `#ifndef Py_LIMITED_API` at line 91 — i.e. it is *inside* the limited API. It is also moot here:
  pyo3 expresses members as getters, and the shim's `TensorBase` already does exactly that.

### 3.2 The verdict, stated plainly

**Autograd is reachable under abi3.** There is no equivalent of `_PyInterpreterFrame` — no CPython
struct whose layout the engine must know, and therefore nothing that makes one binary-per-minor-
version necessary. The engine is C++ over C++ objects and touches no interpreter internals at all
(§2.3, zero `Py` tokens in `engine.cpp`). Everything Python-facing is ordinary reference-counting
and attribute access, which is what pyo3 already expresses across the rest of this shim.

The contrast is worth stating precisely, because these two walls have been spoken of in the same
breath and they are not the same kind of wall:

| | Dynamo | autograd |
|---|---|---|
| what it needs from CPython | `_PyInterpreterState_SetEvalFrameFunc`, `_PyInterpreterFrame` layout | nothing |
| guarded by `Py_BUILD_CORE`? | 6 files | 0 files |
| changes shape per CPython minor? | yes — that is the abi3 killer | n/a |
| verdict | **out of reach without abandoning one-binary abi3** | **in reach; it is an amount of work, not a impossibility** |

So the correct sentence for the roadmap is: *autograd is expensive, not blocked.* What follows
sizes the expense.

---

## 4. The size, in derivative formulas

Ground truth is the vendored `derivatives.yaml` — this is torch 2.13.0's own file, in the tree:

```
$ grep -c "^- name:" torchnative/src/main/torchgen/packaged/autograd/derivatives.yaml
687
```

687 entries is upstream's whole differentiable surface. This shim implements 163 ops
(`torch._C._aten_implemented()`, which is the same 163 `tools/golden/compare.py` reports). The
question is how many formulas *those* need.

```
$ python /tmp/ag/classify.py
shim ops                     : 163
  have a derivative formula  : 122
  have none                  : 41

  trivial  : 66
  composed : 31
  kernel   : 25
```

### 4.1 How the count was made, and what it cannot see

`/tmp/ag/classify.py` maps each `aten.NAME.OVERLOAD` to a `derivatives.yaml` `name:` key (bare
names become `.default`), then reads only the **reverse-mode** lines of each entry. The `result:` /
`result0:` lines are forward-mode (jvp) and a `backward()` never evaluates them — counting them
would have inflated this number substantially. Each body's call sites are then resolved against two
ground truths, both files on disk:

* `torchgen/packaged/ATen/native/native_functions.yaml` — an identifier here is a **dispatched aten
  op**, i.e. something that needs a kernel;
* `torch/csrc/autograd/FunctionsManual.h` — an identifier here is **C++ composition** over ordinary
  tensor ops.

giving three buckets:

| bucket | n | meaning |
|---|---|---|
| **trivial** | 66 | grad arithmetic only. `add.Tensor` is the archetype: `self: grad`, `other: maybe_multiply(grad, alpha)` — pass it through |
| **composed** | 31 | reaches `FunctionsManual` helpers that are themselves expressions over ops the shim already has (`mm_mat1_backward` is `grad.mm(mat2.t())`) |
| **kernel** | 25 | reaches a dispatched aten op the shim does not have |
| *(no formula)* | 41 | see §4.3 |

**What this method cannot see**, stated because a criterion I wrote decides the answer: it reads
the formula *text*, so a body that is textually an expression but numerically delicate
(`pow_backward`'s zero-exponent branch, `div`'s complex conjugation) is counted trivial. The
buckets measure *how many new kernels*, not *how much care*. And a `FunctionsManual` helper is
counted "composed" without checking that every op *it* uses is in the shim's 163 — so "composed"
is a lower bound on work, not a promise of zero work.

### 4.2 The 25, and the number that actually decides weeks-versus-months

Those 25 formulas reach **24 distinct backward ops**. Those 24 are not equal, and the split is
measurable — a backward op registered `CompositeExplicitAutograd`/`CompositeImplicitAutograd`
decomposes into ordinary ops, while one registered `CPU` is a hand-written kernel:

```
$ python -c "... torch._C._dispatch_dump('aten::' + n) ... for each of the 24"
has a real CPU kernel : 14
composite only        : 10
```

| | ops |
|---|---|
| **real CPU kernel (14)** | `native_layer_norm_backward` · `_softmax_backward_data` · `gelu_backward` · `silu_backward` · `tanh_backward` · `sigmoid_backward` · `threshold_backward` · `softplus_backward` · `leaky_relu_backward` · `native_group_norm_backward` · `avg_pool2d_backward` · `upsample_bilinear2d_backward` · `_weight_norm_interface_backward` · `_scaled_dot_product_flash_attention_for_cpu_backward` |
| **composite, decomposes (10)** | `select_backward` · `embedding_backward` · `gather_backward` · `matmul_backward` · `masked_select_backward` · `convolution_backward` · `value_selecting_reduction_backward` · `_weight_norm_differentiable_backward` · `_nested_select_backward` · `_nested_sum_backward` |

And of the 14, **10 have a Core ATen decomposition available** and 4 do not
(`torch._decomp.core_aten_decompositions()`, counted in code rather than by eye — the first draft of
this table said 12/2 from reading the column, and was wrong):

```
HAVE a decomposition (10): _softmax_backward_data gelu_backward leaky_relu_backward
                           native_group_norm_backward native_layer_norm_backward sigmoid_backward
                           silu_backward softplus_backward tanh_backward threshold_backward
have NONE (4):             _scaled_dot_product_flash_attention_for_cpu_backward
                           _weight_norm_interface_backward avg_pool2d_backward
                           upsample_bilinear2d_backward
```

That matters because `docs/graph/DECOMP.md` records a
working Core ATen decomposition path in this shim already; a decomposition is a route to a
gradient without writing the kernel, at the usual cost in speed and memory.

So the honest headline is not "24 kernels". It is:

> **122 formulas, of which 66 are one-liners, 31 are composition, and 25 need a backward op.
> Those 25 reach 24 distinct ops; 10 of those are composites that decompose already, 10 more have a
> Core ATen decomposition to fall back on, and 4 — SDPA's backward, `_weight_norm_interface_backward`,
> `avg_pool2d_backward`, `upsample_bilinear2d_backward` — have neither and must be hand-written.**

Of those 4, only SDPA's backward is on any transformer's path; the other three belong to vision
models (`docs/kernels/KERNELS26.md`'s `zoedepth`/`sam3_video` end of the sweep).

That ratio says **weeks, not months**, for the formulas — which means the formulas are not the
expensive part. The engine and the graph-construction wrapper are (§2.5, §6).

<!-- DOCWATCH: op-implemented aten.native_layer_norm.default -->
<!-- DOCWATCH: op-not-implemented aten.native_layer_norm_backward.default -->
<!-- DOCWATCH: op-not-implemented aten.gelu_backward.default -->
<!-- DOCWATCH: op-not-implemented aten.silu_backward.default -->
<!-- DOCWATCH: op-not-implemented aten._softmax_backward_data.default -->

### 4.3 The 41 with no formula are not a gap

They are ops that are not differentiable and upstream does not pretend otherwise: factories
(`arange`, `empty`, `ones`, `full`, `zeros_like`, `scalar_tensor`, `randint.low`), in-place
mutations (`add_`, `mul_`, `div_`, `clamp_`, `relu_`, `neg_`, `exp_`, `masked_fill_`,
`index_put_`), integer and predicate ops (`bitwise_*`, `floor_divide`, `argmax`, `isin`,
`is_floating_point`), and `detach`/`_local_scalar_dense`, which exist precisely to leave the graph.
None of them needs a derivative. The one to watch is the in-place set: upstream handles those
through `ADInplaceOrView` and version counters, which is a separate mechanism from
`derivatives.yaml` and is **not** counted anywhere in §4.

---

## 5. The cheapest useful subset

A federated or test-time-adaptation step does not need all of autograd. It needs the ops in **one**
model's backward. So this was measured on the model this repository already uses as its yardstick
(`docs/numerics/SEQLEN.md`, `docs/perf/DTYPE_PERF.md`): real SmolLM2-135M weights from the HF cache, `float32`,
`.train()`, the deterministic ids `(i*7919+13) % 49152`, `labels=ids` so there is a real loss, and a
`TorchDispatchMode` recording forward and backward separately.

```
$ HF_HOME=... python /tmp/ag/smol_bwd.py
=== SmolLM2-135M, full fine-tune
    trainable params: 134,515,008 of 134,515,008 (100.000%)
    forward  : 29 uniq / 2320 calls
    backward : 24 uniq / 4264 calls

=== SmolLM2-135M, LoRA r=8 on q_proj/v_proj
    trainable params: 460,800 of 134,515,008 (0.343%)
    forward  : 29 uniq / 2859 calls
    backward : 23 uniq / 4052 calls
```

### 5.1 The subset, for full fine-tuning

**24 distinct aten ops in the whole backward. 16 of them the shim already has.** The 8 missing,
with call counts for one step at sequence length 8, and what each would actually cost:

| missing op | calls | what it is |
|---|---:|---|
| `aten.div.Scalar` | 61 | `CompositeExplicitAutograd`. The shim has `div.Tensor` and `div_.Scalar`; this is a **spelling gap, not a kernel** |
| `aten.zeros.default` | 60 | `CompositeExplicitAutograd`. The shim has `ones.default` and `full.default`. Trivial |
| `aten.slice_backward.default` | 120 | `CompositeExplicitAutograd` — zeros + `slice_scatter`. Decomposes; Core ATen has an entry |
| `aten.silu_backward.default` | 30 | `CompositeImplicitAutograd`, Core ATen decomposition available |
| `aten._log_softmax_backward_data.default` | 1 | real CPU kernel, but Core ATen decomposition available |
| `aten.nll_loss_backward.default` | 1 | real CPU kernel, Core ATen decomposition available |
| `aten.embedding_dense_backward.default` | 1 | real CPU kernel (`index_add_` into a zero buffer), Core ATen decomposition available |
| `aten._scaled_dot_product_flash_attention_for_cpu_backward.default` | 30 | **real CPU kernel, no Core ATen decomposition.** The only genuinely new kernel on this list |

Measured, not asserted — the composite/CPU/decomposition columns come from
`torch._C._dispatch_dump` and `torch._decomp.core_aten_decompositions()`.

So the kernel bill for one federated step on SmolLM2-135M is:

> **two spelling gaps, five ops with a Core ATen decomposition to fall back on, and one real
> kernel to write (SDPA's backward).**

> **The bill was paid at zero.** `docs/training/BACKWARD.md` §1.1: the tape's backward runs *outside* a
> capture region, so a derivative may use ops capture would refuse, may mutate, and may recompute
> instead of reading a saved value. Every entry above is therefore a composition rather than a
> kernel, including the last one — `_scaled_dot_product_flash_attention_for_cpu_backward` is still
> absent and stopped nothing (BACKWARD.md §3.4). `ops=166` is unchanged across that whole round,
> which is the check that says so. Two of the eight rows are also simply out of date: `div.Scalar`
> and `zeros.default` have had kernels since `docs/training/LOSS.md`, and are absent from
> `_aten_implemented()` only because that list means "has a kernel **and** golden compares it".

This is much smaller than §4's 24 and far smaller than the 687 in `derivatives.yaml`, and the
reason is that a decoder-only transformer is a narrow slice of ATen. That is the same reason
`docs/kernels/KERNELS26.md` could reach 26 architectures on 163 ops.

**The kernels are not the expensive part of this.** The engine and graph construction are, and
they are fixed cost paid once, not per-op (§2.5, §6).

### 5.2 What LoRA actually saves, which is not what one would guess

The LoRA run freezes all 134.5M parameters and adds rank-8 `A`/`B` pairs on every `q_proj` and
`v_proj` (60 adapters, 460,800 parameters, **0.343%** trainable), spliced in with forward hooks.

```
full-only ops : ['aten.embedding_dense_backward.default']
lora-only ops : []
shared        : 23 of 24
```

**LoRA removes exactly one op from the requirement.** `embedding_dense_backward` goes, because the
embedding table is frozen and its gradient is never formed. Everything else is identical, and the
call counts barely move (4264 → 4052).

This is obvious in hindsight and worth stating loudly because it is easy to get backwards: **the
adapter is small, but the backward still has to traverse the entire network to reach it.** Gradient
has to flow through all 30 blocks — through SDPA, through the SwiGLU MLP, through every RMSNorm —
to get to a rank-8 matrix in layer 0. Freezing a parameter removes its `AccumulateGrad` leaf; it
does not remove the path.

So, for an on-device adaptation library, the honest statement is:

| what LoRA saves | what it does not save |
|---|---|
| optimiser state — 460,800 params instead of 134.5M, i.e. ~3.5 MB of Adam moments instead of ~1 GB | the kernel set: **23 of 24 ops, unchanged** |
| the gradient buffers for frozen weights | the engine, graph construction, `AutogradMeta` — all fixed cost |
| what has to be transmitted in a federated round | activation memory for the backward, which dominates on a phone |

**LoRA is not a cheaper route to a backward. It is a cheaper thing to do once you have one.** If
the plan was "ship LoRA first because it needs less autograd", that plan does not survive this
measurement — it needs 23/24ths of the same autograd.

<!-- DOCWATCH: op-implemented aten._scaled_dot_product_flash_attention_for_cpu.default -->
<!-- DOCWATCH: op-implemented aten.silu.default -->
<!-- DOCWATCH: op-implemented aten.div.Tensor -->
<!-- DOCWATCH: op-not-implemented aten.div.Scalar -->
<!-- DOCWATCH: op-not-implemented aten.slice_backward.default -->
<!-- DOCWATCH: op-not-implemented aten.embedding_dense_backward.default -->

### 5.3 A prerequisite that is not autograd at all

The measurement above needed `labels=ids` to produce a loss, and that exposed something the
`.train()` sweep could not: **the shim cannot compute a cross-entropy loss forward yet.**

```
full: forward uniq 29, MISSING from shim: 2 -> ['aten._log_softmax.default',
                                                'aten.nll_loss_forward.default']
```

`docs/training/TRAIN.md`'s 26/26 are `.train()` forwards **without a loss** — the sweep feeds ids and reads
logits. A training step needs the loss, and those two ops precede any backward. They are cheap
(`_log_softmax` is a max-subtract-exp-sum-log; `nll_loss_forward` is a gather and a mean) and
neither needs autograd, so they are the smallest genuinely useful next commit in this direction
and could land before any of §4 or §6 is decided.

> **Landed, in [`docs/training/LOSS.md`](../training/LOSS.md) — and the count above was low, for a reason worth having.**
> Both kernels are in and both agree with upstream. The sentence "those two ops" is where this
> section was wrong: **a `TorchDispatchMode` sits below the composite layer**, so the scan that
> produced it records the leaves a call lands on and never the names a caller uses to get there.
> The real path needed the two kernels *and four more names* — `Tensor.log_softmax`,
> `torch.log_softmax`, `torch._C._nn.nll_loss_nd` and `torch._C._nn.cross_entropy_loss` — every one
> `CompositeImplicitAutograd` and therefore invisible to the measurement above. `F.cross_entropy`
> is what `transformers` calls, and with both kernels present it still refused, on the last of them.
>
> "Cheap" was generous too. `_log_softmax` has **two** upstream CPU kernels that round the sum in
> different places, and `nll_loss_forward` sums through an **eight-level cascade** whose carry an
> ignored target skips. Neither falls out of transcribing the formula, and both are invisible in
> `float32`. LOSS.md §2 and §3.

<!-- DOCWATCH: op-implemented aten._log_softmax.default -->
<!-- DOCWATCH: op-implemented aten.nll_loss_forward.default -->

---

## 6. Tape-over-capture versus a `VariableType` equivalent

**Judgement: the tape wins, and not narrowly.** The reasons are structural rather than
preferential, and three of them are demonstrable on today's build.

### 6.1 The shim's single door is exactly the advantage upstream does not have

`docs/graph/CAPTURE.md` §1 states the asymmetry that decides this. Upstream's dispatcher has many doors,
so recording a graph there requires `__torch_dispatch__` modes, fake tensors, and a frame
evaluator — and recording *gradients* requires a generated wrapper for every op, which is what
`VariableType_*.cpp` is (§2.1). **This shim has one door**, `aten_dispatch`, and the recorder is
already one line at the end of it.

Building a `VariableType` equivalent here would mean re-introducing per-op work that the single
door removed. It would be **163 wrappers** to get a property **one line** already provides.

### 6.2 It is not a proposal — it runs today, on a training-mode transformer block

Measured on this build, through the vendored tree:

```
$ ... python -c "capture a .train() LlamaDecoderLayer forward and replay it"
<CaptureTrace 57 nodes, 1 inputs, 11 constants, 1 outputs>
distinct ops: 18
    9 aten.mul.Tensor  7 aten.t.default  7 aten.matmul.default  4 aten.transpose.int
    4 aten.slice.Tensor  4 aten.add.Tensor  3 aten.view.default  2 aten.pow.Tensor_Scalar
    2 aten.mean.dim  2 aten.add.Scalar  2 aten.rsqrt.default  2 aten.unsqueeze.default
    2 aten.neg.default  2 aten.cat.default  2 aten.contiguous.default
    1 aten._scaled_dot_product_flash_attention_for_cpu.default  1 aten.reshape.default
    1 aten.silu.default

replay max abs diff vs eager: 0.0
```

**The forward half of a backward pass already exists.** A tape with the ops, the arguments, the
value identities and the output shapes, replayable bit-for-bit. A reverse-mode backward over this
is a walk of that list backwards with a `grad` map — the thing §2.3's `engine.cpp` spends 1862
lines on is, for a straight-line single-threaded tape, on the order of a hundred.

Note the second number: **18 distinct ops for a whole transformer block**, against 163 implemented
and 687 in `derivatives.yaml`. The tape narrows the formula requirement to what the model actually
executed, which is the same narrowing §5 found from the other end.

### 6.3 Capture already enforces the invariant a tape needs

```
torch._C capture: cannot capture this region -- aten.bernoulli_.float writes in place;
capture refuses mutation so that aliasing cannot be observed, which is what keeps a trace
single-assignment
```

Upstream needs the `ADInplaceOrView` dispatch key and per-tensor version counters to make
reverse-mode safe in the presence of mutation. **Capture gets the same guarantee for free by
refusing mutation at record time.** Single-assignment is precisely the property a tape needs, and
it is already enforced and already tested.

### 6.4 And the decomposition pass shrinks the formula bill

`docs/graph/DECOMP.md` records `torchnative.export.decompose` lowering a captured trace to Core ATen by
*running upstream's own `torch/_decomp` rules*. Derivatives written against Core ATen therefore
cover every spelling that decomposes to them — which is how §4's "24 backward ops" collapses to
§5's "one real kernel" for an actual model.

### 6.5 What the tape does not give, named rather than glossed

This is the honest side of the recommendation.

| limitation | severity for README §2/§3 |
|---|---|
| **Straight-line only.** A tape records one execution. Data-dependent control flow between ops is not captured; a different branch needs a different tape | **low** — a training step on a fixed model is straight-line, and the guards already exist to detect when a tape no longer applies |
| **`.train()` with real dropout is refused.** `bernoulli_` is in-place, so capture rejects it — measured above | **medium, and specific.** SmolLM2-135M has dropout 0.0 so it captures fine (§5), but `gpt2`/`bert`/`opt`/`gpt_bigcode` — TRAIN.md's own four — do not. `docs/training/TRAIN.md` §8 already names `native_dropout`, the functional spelling, as absent; **that op is the fix for this**, and this is a second, independent reason to want it |
| **`backward()` anywhere, on anything.** A tape only differentiates what was recorded inside a capture region; `VariableType` differentiates arbitrary user code | **low** for a federated/TTA library, which owns its own training loop. **High** if the goal is "be torch" |
| **Double backward / `create_graph=True`.** | low — no federated or TTA algorithm in README needs it. A tape can in principle record its own backward, which is arguably *cheaper* here than upstream's approach |
| **`.grad` on leaves** still needs a per-tensor slot, i.e. a small piece of what `AutogradMeta` is | unavoidable either way; it is a leaf-side map, not a per-op node |

### 6.6 The recommendation

> **Build the tape.** Reverse-walk a captured, Core-ATen-lowered trace, with a `grad` map keyed on
> the trace's existing value identities, and derivative rules written against Core ATen only. Do
> **not** build a `VariableType` equivalent — it re-creates per-op work that this shim's single-door
> design specifically removed, and it buys eager-anywhere semantics that a federated/TTA library
> does not need.

The order this implies:

1. `aten._log_softmax.default` + `aten.nll_loss_forward.default` — the loss forward (§5.3). No
   autograd involved; unblocks measuring a real training step at all.
2. `aten.native_dropout` — the functional dropout spelling. Fixes capture in `.train()` for the
   four architectures it currently refuses, and closes `docs/training/TRAIN.md` §8's third item.
3. ~~The tape walker and a `grad` map~~ — **landed, docs/training/BACKWARD.md.** 56 rules, not 18: a
   whole `labels=` forward of SmolLM2-135M reaches 20 distinct ops and the rest are the small
   cases' coverage.
4. ~~`_scaled_dot_product_flash_attention_for_cpu_backward`~~ — **not needed.** The kernel is
   still absent; the tape recomputes the attention and differentiates the textbook formulation
   (BACKWARD.md §3.4). Removing SDPA from the model entirely moves the gradient residual by 1%
   (§4.4 there), so it was never the wall this list expected.
5. ~~An optimiser step.~~ **Landed for SGD** — 272 of 272 parameters, weights moved, 99.9976% of
   the step components pointing where upstream's point. Adam still needs the four items below. This was measured rather than assumed, because the first draft of this list
   guessed it and guessed wrong:

```
$ python /tmp/ag/optim_ops.py       (steady-state step, after state allocation)
SGD            ops= 3  missing=2  [profiler._record_function_enter_new, profiler._record_function_exit]
SGD+momentum   ops= 4  missing=2  [ the same two ]
Adam           ops=10  missing=5  [aten.addcdiv_.default, aten.addcmul_.default,
                                   aten.lerp_.Scalar, + the same two profiler markers]
AdamW          ops=10  missing=5  [ same as Adam ]
```

   **SGD needs zero new aten kernels.** Adam and AdamW need three, all elementwise in-place
   (`addcdiv_`, `addcmul_`, `lerp_.Scalar`) and all in the same class as `add_`/`mul_`, which the
   shim already has. The two `profiler::_record_function_*` entries are not kernels — they are the
   `record_function` markers `torch.optim` wraps every step in, and they currently refuse:

```
$ ... python -c "import torch; torch.profiler.record_function('x').__enter__()"
NotImplementedError: aten op not implemented in torch._C shim:
    profiler._record_function_enter_new.default
```

   They are a pair of no-ops for a build with no profiler, and they block **every** optimiser in
   `torch.optim`, SGD included.

Steps 1 and 2 are ordinary kernel work of the kind this repository has done 163 times. Step 3 is
the genuinely new thing, and §6.2 is the evidence that its input already exists. Step 5's profiler
markers are the smallest item on the whole list and gate the last stage of a federated round.

---

## 7. What already works, which is more than the comments claim

The brief asks whether the shim already threads `requires_grad` further than anyone has checked.
It does — **the entire parameter-selection half of a federated or LoRA setup runs today**, on this
build, through the vendored tree:

```
nn.Parameter carries requires_grad                   OK   True
module.requires_grad_(False) propagates              OK   True
selective unfreeze (LoRA shape)                      OK   32
named_parameters filtering                           OK   ['2.weight']
torch.optim.SGD constructs                           OK   SGD
torch.optim.AdamW constructs                         OK   AdamW
p.grad reads as None                                 OK   None
torch.no_grad() round-trips                          OK   (True, True)
nn.Parameter is a Parameter                          OK   True
detach()                                             OK   False
torch.autograd.Function subclass defines             OK   F
--
optimiser.zero_grad()                                FAIL profiler._record_function_enter_new.default
p.grad is assignable                                 FAIL property '<lambda>' of 'Parameter' has no setter
```

This is worth stating precisely, because it changes what "we have none of it" means. **A LoRA
harness can already build its model, freeze 99.657% of it, select the adapters, and construct an
AdamW over them.** Everything up to the point where a gradient would have to exist is in place.
`bootstrap.py:4087`'s docstring calls this "the papered-over part", which is fair about the
semantics and understates the reach — `module.requires_grad_(False)` recursing correctly over a
module tree and `named_parameters()` filtering on the flag are not nothing, and they are what
§5's LoRA measurement needed.

Two named gaps in that list, both small and both on the critical path:

* **`optimizer.zero_grad()` fails**, and not for a gradient reason — it fails on the
  `record_function` marker (§6.6 step 5). Every `torch.optim` optimiser is affected.
* **`.grad` has no setter.** It is `property(lambda self: None)` at `bootstrap.py:4166`. A
  tape-based backward has to write leaf gradients somewhere, and that is the slot. Deliberately
  not changed here: making `.grad` writable while nothing writes to it would move the shim from
  "honestly reports no gradient" to "has a slot that is always empty", which is the direction
  `_install_autograd_shape` explicitly argues against.

  > **Closed, and the argument above is what decided when.** `docs/training/BACKWARD.md` §10: the tape
  > writes gradients, so the slot is no longer always empty and the antecedent of that sentence is
  > gone. What would be dishonest now is a `torch.optim` step that silently skipped every parameter
  > because the slot it reads cannot be filled. Nothing fills it implicitly — `backward()` returns
  > gradients and the caller assigns them.

---

## 8. What this document does not establish

Kept explicit, in the shape `docs/graph/DYNAMO.md` §19 uses, because several of these are places where a
reader could over-read what is above.

| # | not established | why |
|---|---|---|
| 1 | ~~**That a tape-based backward is numerically correct.**~~ | **Established — docs/training/BACKWARD.md.** An `nn.Linear`'s gradient is bit-identical to upstream; a whole SmolLM2-135M's 134,515,008 gradient elements agree at a median relative L2 of 8.8e-05 in `float32` and **8.5e-07 in `float64`**, which is what says the residual is the forward's arithmetic rather than the rules |
| 2 | **Effort in time.** | §4 and §5 count formulas and kernels, which is what the brief asked for. Converting counts to weeks would be the estimate-without-measurement this repository refuses |
| 3 | **Memory.** | Not measured at all, and on a phone it may dominate everything here. A backward keeps every intermediate activation alive; SmolLM2-135M at S=8 is tiny, and nothing was measured at a realistic sequence length. `docs/numerics/SEQLEN.md`'s quadratic term is a forward-only measurement |
| 4 | **That the 4 kernels with no decomposition are hard.** | They were classified by dispatch registration, not read. `avg_pool2d_backward` is probably easy; SDPA's backward is probably not. Neither was opened |
| 5 | **Anything on device.** | Desktop macOS only. `docs/design/DESIGN.md` §5's iOS W^X constraint does not obviously apply — a tape walker generates no code — but that is reasoning, not a measurement |
| 6 | **Double backward and forward-mode.** | §4 explicitly discards the `result:` lines of every formula. If forward-mode AD or `create_graph=True` is ever wanted, §4's counts are **not** the right size for it and the count must be redone |
| 7 | **That "composed" formulas cost nothing.** | §4.1 says this: a `FunctionsManual` helper was counted as composition without checking that every op *it* reaches is among the 163. "composed: 31" is a lower bound |
| 8 | **Whether capture's guards are sufficient for training.** | A training loop replays the same shape every step, which is the favourable case, but `docs/graph/CAPTURE.md` §4's list of what capture refuses was not re-examined against a *backward* tape, only a forward one |

### 8.1 One thing that would change the recommendation

If the goal turns out to be "arbitrary user code calls `.backward()`" rather than "this library
owns its training loop", §6's judgement inverts — a tape cannot serve the first and
`VariableType` equivalents become unavoidable, at 163 wrappers plus version counters. README §2
and §3 describe the second, so this document recommends for the second. **That reading of README
is the one assumption here that is not a measurement**, and it is the one to check first.

---

## 9. Every command in this document

```sh
export PATH="$HOME/.cargo/bin:$PATH" CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-autograd
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
export HF_HOME=/Volumes/macMini/caches/hf-home
bash vendor/install_shim.sh
PY=/Volumes/macMini/caches/spike-venv/bin/python
SHIM="PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 $PY"     # VENDOR.md wall 1

# §1  where it stops
$SHIM -c "import torch; torch.randn(4, requires_grad=True)"            # wall 1
$SHIM /tmp/ag/rung1.py 0                                               # wall 2
$SHIM /tmp/ag/rung1.py 1                                               # wall 3 (TLS stubbed)

# §2  which layer it lives in
$PY /tmp/ag/graph_walk.py elementwise linear layernorm llama_block
$PY -c "import torch; print(torch._C._dispatch_dump('aten::mul.Tensor'))"

# §3  the abi3 verdict          (P = .../pytorch-spike/pytorch/torch/csrc)
grep -rl "Py_BUILD_CORE"  $P/dynamo/   ;  grep -rl "Py_BUILD_CORE"  $P/autograd/
grep -rl "internal/pycore" $P/dynamo/  ;  grep -rl "internal/pycore" $P/autograd/
grep -c "Py_\|PyObject\|PyGILState" $P/autograd/engine.cpp             # -> 0

# §4  the formula count
grep -c "^- name:" torchnative/src/main/torchgen/packaged/autograd/derivatives.yaml   # -> 687
$SHIM -c "import torch,json; json.dump(sorted(torch._C._aten_implemented()),
                                       open('/tmp/ag/shim_ops.json','w'))"            # -> 163
$PY /tmp/ag/classify.py                                  # 122 = 66 trivial + 31 composed + 25 kernel

# §5  the cheapest useful subset
$PY /tmp/ag/upstream_bwd.py   &&  $PY /tmp/ag/diff_ops.py
$PY /tmp/ag/smol_bwd.py                                  # full vs LoRA, real SmolLM2-135M weights
$PY /tmp/ag/optim_ops.py                                 # SGD 0 kernels, Adam 3

# §6  capture is the tape
$SHIM -c "capture a .train() LlamaDecoderLayer forward and replay it"  # 57 nodes, diff 0.0
```

The scratch scripts live under `/tmp/ag/` and are reproduced nowhere else; they are throwaway
harnesses, and every number they produce is quoted above with the command that made it. Nothing in
`rust/`, `tools/`, `scripts/` or the vendored tree was modified by this investigation.

---

## 10. The one thing this round implemented

Nothing in `rust/torch_c/src/` changed. One test was added, because §1's boundary was written down
in a document and **nothing checked it** — which is precisely the mechanism `docs/verification/AUDIT.md` found
behind six of eleven false claims, and `docs/verification/DOCWATCH.md` exists to stop.

`test_the_autograd_boundary_is_where_autograd_md_says_it_is` in `rust/torch_c/pytests/test_shim.py`
pins the three facts §1 measured, against `_C` alone (no vendored-tree subprocess needed):

| assertion | what it catches |
|---|---|
| `requires_grad` round-trips both spellings, `is_leaf`/`grad_fn`/`grad` report no graph | the flag plumbing regressing under an unrelated change |
| `mul(x, x).requires_grad is False` | **graph construction appearing** — the one that would silently invalidate §1.3 and §6 |
| `_stash_obj_in_tls` and `run_backward` both raise `NotImplementedError` | a refusal being replaced by a zero, which is the failure `_install_autograd_shape` argues against |

It is written to **fail when autograd lands**, and says so in its own message. That follows
`test_the_two_stale_sdpa_refusals_no_longer_claim_a_missing_kernel`: when the boundary moves,
invert the test rather than deleting it, and revisit this document in the same commit.

### 10.1 Proof that it can fail

`CLAUDE.md`'s rule — a check that cannot fail is not a check — applied before claiming it as a
gate. Three faults, each shaped like the real change that would make the claim wrong, injected
into a scratch copy under `/tmp/ag/` (never into the tree):

```
fault: graph-construction-appeared   FAIL  an op propagated requires_grad -- graph construction
                                           has appeared; see docs/training/AUTOGRAD.md §1.3 and §6 ...
fault: engine-stopped-refusing       FAIL  run_backward no longer refuses -- if autograd landed,
                                           invert this test and update docs/training/AUTOGRAD.md §1 ...
fault: flag-stopped-round-tripping   FAIL  AssertionError
unmodified                           ok    test_the_autograd_boundary_is_where_autograd_md_says_it_is
```

All three are caught, and each names which of the three claims broke.
