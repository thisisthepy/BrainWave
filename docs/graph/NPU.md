# NPU — serialising a captured graph for NNAPI and CoreML

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/nnapi.py to_jit_module present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/nnapi.py parse_model present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/nnapi.py verify_shapes present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/nnapi.py fold_constants present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/coreml.py to_mil_program present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/coreml.py compile_model present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/export/coreml.py coreml_ops present -->

## 1. The question this round had to answer first — and the answer is *both*

> **Can upstream's NNAPI serialiser be driven from our lowered graph, or does it
> need a `torch.jit` trace we cannot produce?**

**Its entry point needs a trace we cannot produce. Everything behind that entry
point can be driven, and is.**

`_NnapiSerializer.serialize_model(model, inputs)` begins at
`next(model.graph.inputs())`. There is no TorchScript compiler in this shim, and
this is measured rather than assumed
(`test_upstreams_nnapi_serialiser_needs_a_jit_graph_this_build_cannot_make`):

| | |
|---|---|
| `torch.jit.trace(m, x) is m` | **True** — it hands the module straight back |
| `traced.graph` | absent |
| `torch._C.Graph.__doc__` | `torch._C shim placeholder for torch._C.Graph` |

So any plan of the form "trace the module, then call upstream's converter" is
dead before it starts, for NNAPI and for `torch/backends/_coreml` alike.

But the entry point is the *only* part that needs a real graph. `serialize_model`
and its 47 `add_*` methods reach their argument through a surface of thirteen
methods, every one duck-typed, none downcasting or calling into C++:

    Graph   inputs  nodes  return_node
    Node    kind  inputsSize  outputsSize  inputsAt  outputsAt  inputs  s
    Value   type  toIValue
    Type    kind  getElementType

Values are used as plain dict keys, hashed by identity. So the round is neither
"wire up upstream's serialiser" nor "write one": it is **write the graph
façade**. `torchnative/export/nnapi.py` presents our recorded trace as those
thirteen methods, and upstream's `ADDER_MAP`, operand table, immediate cache and
blob layout then run unmodified.

`test_the_serialisers_graph_surface_is_three_methods_and_stays_three` re-reads
the vendored file on every run, so a vendored update that reaches for a fourth
graph method turns red rather than leaving the façade silently incomplete.

## 2. What was executed and what was structurally validated

These are different claims and this document does not merge them.

| Artefact | Claim | How |
|---|---|---|
| NNAPI model blob | **structurally validated** | `parse_model` decodes it back through the layout `serialize_model` wrote; `verify_shapes` compares every operand shape against capture |
| CoreML `.mlpackage` | **executed** | compiled by macOS and run through `MLModel.predict`, compared numerically against `DecomposedTrace.replay` |

There is no NNAPI runtime on a Mac. Nothing here claims a blob ran on an NPU.

**Both rows were revisited in docs/graph/NPU2.md, and both moved.** The CoreML row's
"executed" turned out to mean *executed on the CPU* — `MLComputePlan` does not
even list the Neural Engine as supported for a float32 program, so §6's
`float32=True` and NPU execution exclude each other. And the NNAPI row is no
longer structural: the blob runs on an Android emulator's NNAPI runtime through
`ANeuralNetworksModel`. Read this table as the state of things at the time it
was written.

`parse_model` is a decoder, not a length check: it walks the operand, value,
operation and flat-argument tables in the exact order and packing upstream
emits them, requires every table length to match the header, every operand
reference to be in range, and the bytes to be consumed exactly. It has to be
able to say no, so
`test_the_blob_decoder_rejects_blobs_that_do_not_decode` injects two faults
shaped like real serialiser bugs — a header claiming one more operand than the
tables carry, and four trailing bytes — and requires each to be refused by name.

## 3. What serialises today

Capture → NNAPI blob, end to end, through upstream's serialiser:

| Graph | Captured ops | NNAPI operations |
|---|---|---|
| `Conv2d → ReLU` | `convolution`, `relu` | `CONV_2D`, `RELU` |
| `Conv2d → ReLU6` | `convolution`, `hardtanh` | `CONV_2D`, `RELU6` |
| `Linear → softmax` | `t`, `addmm`, `_softmax` | `FULLY_CONNECTED`, `SOFTMAX` |
| `AdaptiveAvgPool2d` | `adaptive_avg_pool2d` | `AVERAGE_POOL_2D` |
| `Sigmoid` / `add` / `mul` / `cat` / `unsqueeze` | — | `LOGISTIC` / `ADD` / `MUL` / `CONCATENATION` / `EXPAND_DIMS` |

CoreML, compiled and **run**, agreeing with replay to float32 precision:

| Graph | max abs diff vs `replay` |
|---|---|
| `Linear → GELU → Linear → softmax` | 3.0e-08 |
| `Conv2d → AdaptiveAvgPool2d → ReLU` | 2.0e-08 |
| `Sigmoid` | 3.0e-08 |
| `a - a*2` | 0.0 (bit for bit) |

## 4. `_SIGNATURES`: where docs/graph/DECOMP.md §12.2's caveat becomes an obstacle

§12.2 noted that `ADDER_MAP` is keyed by TorchScript node kind (`aten::add`),
which carries no overload, and called the 29-op count an upper bound. Driving
the serialiser turns that from a caveat into work, because every adder asserts
an **exact** `inputsSize()`:

* `aten::add` wants three inputs; capture records `aten.add.Tensor` with two,
  `alpha` having been defaulted away.
* `aten::_convolution` wants thirteen; capture records
  `aten.convolution.default` with nine, the last four being cuDNN switches the
  adder unpacks into `_`.
* `aten::hardtanh` wants three; `nn.ReLU6` records one.

`_SIGNATURES` is that map, and it is a **calling convention** table, not a
semantics table: no entry changes what an op computes, and an op that cannot be
mapped exactly is *absent* rather than approximated. Absence refuses by name
(`test_an_op_with_no_calling_convention_is_refused_by_name`).

A second mismatch sits underneath it: the dispatcher may pass an argument by
keyword, so `aten.relu.default` arrives with zero positional arguments and
`self` in `kwargs` while `aten.convolution.default` arrives with nine
positional. `_positional()` flattens both onto schema order by reading
`torch._C._get_schema` — the same registry `verify_schemas.py` checks against
upstream — rather than by guessing which name goes where.

**`supported_ops()` is strictly smaller than `target.nnapi_ops()`**, and
`test_what_serialises_is_smaller_than_what_nnapi_nominally_accepts` pins that.
The gap is the distance between "NNAPI has an adder for this name" and "a
captured overload of it can be handed to that adder", which is exactly what
§12.2 said the 29 was an upper bound *on*.

Two ops NNAPI nominally accepts are deliberately unmapped:
`max_pool2d_with_indices` (capture records two outputs, `aten::max_pool2d` has
one — dropping the indices is a graph rewrite, not a calling convention) and
`size` (its adder emits nothing and only feeds flexible-shape bookkeeping this
façade does not produce).

### The check that can fail

A wrong `_SIGNATURES` position is silent: put `output_padding` where `groups`
belongs and the blob decodes perfectly and computes something else. So
`verify_shapes` compares the shape the serialiser assigned to each node output
against **the shape capture recorded for that same node** — two derivations of
the same number, one from upstream's shape propagation over NNAPI operands, one
from a real CPU execution.

The test injects the fault and requires it to be caught: it swaps `stride` and
`padding` in the convolution plan. The first version of that injection used the
`padding=1, stride=1` convolution, where the swap is the **identity** — it
"passed" while checking nothing, which is CLAUDE.md §5.5 exactly. It now runs on
a `stride=2, padding=1` convolution, where the swap moves the output shape —
the serialiser then reports `(1, 4, 10, 10)` for a node capture recorded as
`(1, 4, 4, 4)`, and `verify_shapes` returns that disagreement instead of an
empty list.

## 5. Constant folding is a prerequisite, not an optimisation

`nn.Linear` captures as

    %0 = aten.t.default(%weight)
    %1 = aten.addmm.default(%bias, %x, %0)

and `aten::t` is not in `ADDER_MAP` — NNAPI has no transpose. It does not need
one: `add_addmm` requires `mat2` to be a **constant** weight and transposes it
itself (`weight_tensor.t().contiguous()`), because `FULLY_CONNECTED` wants
`[out, in]`. The recorded graph is unserialisable while the graph it *denotes*
is entirely serialisable, and the difference is one node over a value known
before the model runs.

`fold_constants` evaluates any node whose inputs are all constants by
**executing** it through `torch._C._aten_dispatch` — the door capture recorded
at, and the one `DecomposedTrace.replay` goes back through — so the folded
value is upstream's own answer rather than a second implementation of transpose.

An early version resolved already-remapped references a second time, and the
visible symptom was `_softmax` folding away even though its input comes from the
model input. That fold produces a graph which serialises, runs, and returns the
same answer for every input. The test now asserts by name that `_softmax`
survives.

## 6. CoreML: MIL, not the packaging wrapper

`torch/backends/_coreml/preprocess.py` calls `coremltools.convert` on a
`torch.jit` object, so §1 shuts that door — harder than NNAPI's, in fact, since
coremltools' torch frontend walks a real jit IR with type refinement and its own
`InternalTorchIRGraph`, not thirteen duck-typed methods.

CoreML has a second, first-class entry NNAPI has no equivalent of: **MIL**,
coremltools' own intermediate language with a Python builder. Producing a MIL
program is the torch frontend's entire job, so `to_mil_program` emits our
recorded graph as MIL directly rather than faking a trace, and `ct.convert` then
runs the full backend pipeline.

`libcoremlpython` is present on macOS, so the result is compiled by the OS and
run. That is the executed claim in §2.

### coremltools was installed, and it did not bring a torch

Checked before installing, because a dependency that replaced our `torch` would
silently invalidate every measurement in this repository:

    coremltools 9.0 (cp313, macosx_11_0_arm64, 2.8 MB)
    new packages: attrs, cattrs, pyaml   — 4 wheels, ~3 MB total
    torch: NOT pulled in

Disk went 85% → 86% on the external volume. coremltools warns that torch 2.13.0
is newer than it has tested against; nothing in this path uses its torch
frontend, so that warning does not bear on anything here.

### §12.6's successor item, and what did *not* change

docs/graph/DECOMP.md §12.6 recorded the CoreML operator set as **not measured** and
told the next person to install coremltools and read
`coremltools.converters.mil.frontend.torch.ops` rather than transcribe a list.
`coreml.coreml_ops()` does that read: **447** ops in the `@register_torch_op`
registry.

`target.coreml_ops()` still refuses, and that is correct rather than an
oversight. Its claim is that there is no CoreML operator set **in this tree**,
and installing a package into a venv does not put one there. Changing it so a
test that asserts the refusal goes green would be adjusting the test's subject
to fit a new fact, and the two functions answer different questions anyway:

| | answers |
|---|---|
| `target.coreml_ops()` | is there an op list in the vendored tree? — no, and here is where the real one lives |
| `coreml.coreml_ops()` | what does coremltools' frontend accept? — 447 names, read from its registry |
| `coreml.supported_ops()` | what has a MIL lowering *here*? — far fewer, and an op outside it refuses by name |

Conflating the second with the third would report CoreML coverage this project
does not have — §12.6's mistake from the other side.

### float16 is the default and it changes what a number means

coremltools defaults `mlprogram` to **float16** compute precision. The first run
of `verify()` disagreed with replay by **1.4e-4** on a two-layer MLP and
**8.2e-4** on a `cat(x, 2x)` graph — far outside any float32 tolerance, and
entirely explained by half precision. The suite re-measures it every run on one
model: **2.3e-04 at float16 against 3.0e-08 at float32**, four orders of
magnitude apart. Reading those as a lowering error would have sent
the search to the wrong place; waving them through as "close enough" would have
hidden a real one behind the same number. So `compile_model(float32=True)` is
the default, and
`test_coremls_default_precision_is_float16_and_that_changes_the_claim` measures
both on the same model and requires the float16 build to be visibly worse — if
they ever agree, the flag stopped doing anything and the float32 claim is no
longer the claim being made.

## 7. What is still outside, and why it is not a decomposition problem

Against the models docs/graph/DECOMP.md §12.3 measured, after folding:

| Model | Ops with no calling convention |
|---|---|
| `mobilenet_v2` | **2** — `native_batch_norm`, `constant_pad_nd` |
| `vit` | 7 — incl. `native_layer_norm`, `transpose.int`, `select.int`, SDPA |
| `smollm2_llama` | 15 — incl. `embedding`, `matmul`, `silu`, SDPA, RoPE's `sin`/`cos` |
| `resnet` | capture refuses first (in-place residual add, docs/graph/CAPTURE.md §4) |

`mobilenet_v2` is two ops away, and §12.5 already showed that **lowering makes
it worse**: running the union table drives it from 3 ops outside NNAPI to 6 and
from 203 nodes to 1191, because `native_batch_norm` decomposes into `sqrt`,
`reciprocal` and `new_zeros`, none of which NNAPI has. Re-measured here on a
`Conv → BatchNorm → ReLU → Conv → ReLU6 → pool → Linear → softmax` network, the
same shape appears: one unmapped op before lowering, ten after.

The right treatment for inference batch-norm on an NPU is not a decomposition at
all — it is **folding the normalisation into the preceding convolution's
weights**, which is §12.7's *target-dependent* category. That pass is not
written here. It is the highest-value next step for NNAPI and it is a semantic
rewrite, so it wants its own round with its own numerical proof rather than
being appended to this one.

## 8. Reproducing

    export PATH="$HOME/.cargo/bin:$PATH"          # install_shim.sh needs cargo
    export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-npu
    export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
    export TORCH_C_STAGE=/tmp/stage-npu
    PY=/Volumes/macMini/caches/spike-venv/bin/python

    cd rust/torch_c && cargo build --release && cd -
    bash vendor/install_shim.sh
    PYTHON=$PY sh rust/torch_c/pytests/run.sh

The eleven tests this document is about are in `rust/torch_c/pytests/test_shim.py`
and all begin `test_upstreams_nnapi_`, `test_the_serialiser`, `test_a_conv_relu`,
`test_serialised_shapes`, `test_constant_folding`, `test_an_op_with_no_`,
`test_the_blob_decoder`, `test_what_serialises`, `test_coreml`. They skip rather
than fail where coremltools is absent, so the NNAPI half stands alone.
