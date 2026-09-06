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
lowering passes that do not need it.

`torchnative.export.nnapi_device` is a third, and is not imported here for the
same reason twice over: it needs `adb`, an `ANDROID_SERIAL`, and an NDK to
build `nnapi_runner.c` with. It is what makes the NNAPI blob *executed* rather
than merely decoded -- docs/NPU2.md §3, which also records that the CoreML
side's executed claim was a **CPU** claim until §2 of that document put a graph
on the Neural Engine. Executed and structurally validated are different claims
and every one of these modules is careful about which it is making.

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
