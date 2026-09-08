# QNNOPS — ground truth for what QNN/Hexagon HTP actually supports, per op

`torchnative.export.qnn_ops` (`torchnative/src/main/torchnative/export/qnn_ops.py`).
Tests: `rust/torch_c/pytests/test_qnn_ops.py`.

This is **not** the AOT-lowering round. `docs/devices/QNN.md` is that round: it
built `torchnative/export/qnn.py` and `torchnative/export/qnn_device.py`, and
found that ExecuTorch's QNN Python bindings are Linux-only and no artefact could
be produced on this arm64 Mac (`QNN.md` §1.3). This document is narrower: **for
one ATen op, does QNN's partitioner take it at all**, sourced from the
partitioner's own input rather than from documentation, so a sibling round can
write a per-leaf lowering judgement against something checked rather than
recalled.

## Why sourcing is the whole point here, not a formality

QNN's failure mode is silent and at *compile* time. The partitioner declines a
node it cannot take and the graph falls back to CPU; the program still runs and
still produces the right numbers. So "it ran and the answer was correct" proves
nothing about whether Hexagon executed anything — the opposite of NNAPI/CoreML,
which fail closed at *run* time (`docs/devices/MPS.md`, `docs/graph/NPU2.md`
record the same asymmetry for other backends). A table entry here that is
optimistic rather than sourced would manufacture exactly that false confidence,
in writing, for the next round to build on. Hence: every accepted entry is
traced to a specific line in a specific installed package, and everything that
could not be traced is named below in **UNVERIFIED**, not folded into the table.

## 1. What was searched for on this machine

| Source (best → worst) | Found here? |
|---|---|
| QNN SDK headers (`QnnTypes.h`, `QnnOpDef.h`, HTP op-package headers) | **No.** `find / -iname "*.h" -path "*QNN*"`, `find / -iname "QnnTypes.h" -o -iname "QnnOpDef.h"`, and `find / -iname "*htp*constraint*" -o -iname "*HtpOpPackage*"` all returned empty on this box |
| ExecuTorch's QNN backend partitioner source, local checkout | **Yes.** A real installed wheel: `/Volumes/macMini/caches/qnn-venv/lib/python3.13/site-packages/executorch-1.4.1.dist-info` (`executorch==1.4.1`, `macosx_14_0_arm64`) |
| Official Qualcomm documentation | Not consulted for this table — the local partitioner source is strictly better (it is literally the code the partitioner runs, not a description of it) and was sufficient for every entry below |

Everything in the sourced table traces to the second row. Nothing here is a
paraphrase of a doc page or of memory.

## 2. How the supported table was built

`executorch/backends/qualcomm/builders/op_*.py` — 115 files — are QNN's **node
visitors**. Each registers itself with `@register_node_visitor` against a
`target = [...]` list of ATen/edge op-overload strings
(`node_visitor_manager.register_node_visitor`), and
`executorch/backends/qualcomm/partition/qnn_partitioner.py` accepts a graph
node into the QNN partition **iff a visitor is registered for its target**.
`target` is therefore not documentation of the partitioner's behaviour, it is
the partitioner's input — the same list the real object consults.

```sh
cd /Volumes/macMini/caches/qnn-venv/lib/python3.13/site-packages/executorch/backends/qualcomm/builders
grep -oE '^\s*target\s*=\s*\[[^]]*\]' op_*.py   # one line per builder file
```

extracted **127 unique targets across 115 files**. `qnn_ops.supported_ops()`
returns exactly that set, and
`test_qnn_ops.test_supported_ops_matches_a_fresh_extraction_from_the_installed_wheel`
re-derives the same extraction at test time and diffs it against the module —
so a stale table fails a test rather than sitting undetected.

## 3. The explicit refusal lists

`executorch/backends/qualcomm/partition/common_defs.py`, same package, three
lists the real `qnn_partitioner.py` reads to *remove* nodes even when no
visitor error would otherwise stop them:

| List | Source | Meaning |
|---|---|---|
| `not_supported_operator` (3 entries) | `common_defs.py` | Never partitioned. Each has a source-code comment reason, reproduced verbatim in `qnn_ops.not_supported_ops()`: `aten._embedding_bag.default` ("output size is data dependent on the slice index"), `dim_order_ops._clone_dim_order.default` ("for graph sharding purpose..."), `quantized_decomposed.embedding_4bit.dtype` ("QNN does not support 4-bit embedding") |
| `to_be_implemented_operator` (6 entries) | `common_defs.py` | ExecuTorch's own partitioner names these as not yet done for QNN: `adaptive_max_pool3d`, `max_pool3d_with_indices`, `median.default`, `median.dim`, `round.decimals`, `le.Scalar` |
| `constant_operator` | `common_defs.py` | Folded as constants rather than run as ops (`arange.start_step`, `full.default`, `full_like.default`, `scalar_tensor.default`) — not a refusal, noted for completeness |

A test (`test_the_refusal_lists_are_disjoint_from_the_supported_set`) asserts
none of these overlap the supported set, since overlap would mean the table
contradicts itself.

## 4. Dtypes

Source: `executorch/backends/qualcomm/builders/node_visitor.py`,
`QNN_TENSOR_TYPE_MAP` (unquantized I/O, ~line 59) and `QNN_QUANT_TYPE_MAP`
(quantized tensors, ~line 48).

| | Unquantized | Quantized |
|---|---|---|
| Accepted | bool, float16, float32, float64\*, int8, int16, int32, int64, uint8, uint16, uint32 | int8, int16, int32, uint8, uint16 |
| Notably absent | — | int64 ("there is no int64 tensor data type in Qnn" — source comment), uint32, bool, float |

\* float64 has no native QNN type either; `QNN_TENSOR_TYPE_MAP` maps it to
`QNN_DATATYPE_FLOAT_32` — a silent downcast, which `qnn_ops.check_leaf` names
explicitly rather than reporting as a plain accept.

## 5. UNVERIFIED — what this document could not settle, and why

* **Max tensor rank / max dimension size.** `intelnpu.py`'s `MAX_DIM` has a
  direct analogue nowhere in `executorch/backends/qualcomm/builders/` or
  `partition/` at version 1.4.1, and no QNN SDK header was present on this
  machine to check against directly (§1). QNN/HTP is widely known to have
  practical tensor-size limits, but "widely known" is exactly the kind of claim
  this document exists to keep out of the sourced table. Would be settled by:
  a QNN SDK install (`QnnTypes.h`/`QnnOpDef.h` or the HTP op-package headers),
  or by running the real partitioner against a deliberately oversized tensor on
  a Linux host and reading its refusal message.
* **Static-shape requirement.** HTP graphs are widely understood to require
  static shapes, but no assertion enforcing that was found in the builder or
  partition source at this version either. Same settling path as above.
* **Per-op numeric constraints beyond dtype** (e.g. `conv2d`'s allowed kernel
  sizes/strides, `layer_norm`'s axis restrictions, `topk`'s `k` limits). The
  builder files (`op_conv.py`, `op_layer_norm.py`, `op_topk.py`, ...) contain
  QNN-parameter *construction* code but this audit did not walk all 115 files
  line-by-line for embedded shape assertions — only the `target =` extraction
  (§2) and the two partition-level refusal files (§3) were read in full.
  A sibling round doing a per-op deep dive should treat this as open, not
  closed, for any op it plans to rely on heavily.
* **Whether a *quantized/calibrated* graph is actually accepted end-to-end.**
  §4 says which ops have a quantized dtype mapping; it does not exercise
  `executorch.backends.qualcomm.quantizer` calibration, which `QNN.md` §11
  already names as unbuilt for the AOT-lowering round.
* **Real Hexagon execution of any of this.** Nothing in this document or in
  `qnn_ops.py` claims a node ran on Hexagon silicon. `QNN.md` §5 is the one
  round that touched real silicon (SM8550, HTP v73) and even that round could
  not push an artefact to it.

## 6. Entry count

127 sourced supported-op entries, 3 sourced explicit-refusal entries, 6 sourced
to-be-implemented entries, 2 sourced dtype tables (11 unquantized / 5
quantized types) — **145 total sourced facts**, all traced to
`executorch==1.4.1` installed at `/Volumes/macMini/caches/qnn-venv`. Zero
entries were put in the table unsourced; the numeric-constraint gaps are listed
in §5 instead of being guessed into `constraints()`.
