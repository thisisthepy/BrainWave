"""Passes that run over a captured trace, between capture and a delegate.

`torch._C._capture_end` produces a record in the **ATen** dialect: whatever the
dispatcher was actually asked for. ExecuTorch's Edge dialect is defined over
**Core ATen**, a named subset. docs/CAPTURE.md §5 measured the gap and found it
is not hypothetical -- the smallest example in that document, an
`nn.Sequential` of two `Linear` layers, records `aten.t.default`, which is not
Core ATen.

Serialising is the step after that, and it has its own module per device --
`torchnative.export.nnapi` and `torchnative.export.coreml`. Neither is imported
here: `coreml` needs coremltools installed and `nnapi` reaches into the
vendored `torch.backends`, and making a bare `import torchnative.export` depend
on either would turn a missing optional package into an import error for the
lowering passes that do not need it. docs/NPU.md says which of the two produces
an artefact that has been *executed* and which one has only been structurally
validated; they are different claims.

So a pass has to stand between the two, and `decompose` is it. The rules it
applies are upstream's, read out of the vendored tree rather than restated
here: see `torchnative.export.decompose` for which table, and for the list of
what that table does not reach.
"""

from torchnative.export.decompose import (
    DecomposedTrace,
    DecompositionRefused,
    core_ops,
    decompose,
    decomposition_table,
    decomposition_table_source,
    is_core,
    non_core_ops,
)


__all__ = [
    "DecomposedTrace",
    "DecompositionRefused",
    "core_ops",
    "decompose",
    "decomposition_table",
    "decomposition_table_source",
    "is_core",
    "non_core_ops",
]
