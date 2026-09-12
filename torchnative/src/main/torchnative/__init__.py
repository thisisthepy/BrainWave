"""torchnative — on-device test-time learning and federated learning.

See docs/design/DESIGN.md for the design and its reasoning.

**Every subpackage** is resolved lazily, through PEP 562, for the reason
`torchnative.quant` already gives: importing this package must stay cheap and
must not drag in `torch` (let alone `transformers`, which is not a hard
dependency) for someone who imported it for something else.
`import torchnative; torchnative.device.npu` works because the attribute lookup
lands here and imports the submodule then, and so does
`torchnative.adapt.wrap` -- which it did not until 2026-09-12, because only
`device` and `transformers` were listed below.

That laziness is also what makes the `nn.Module.to` patch safe to install at
`torchnative.device` import time: see `device/_module_to.py`.
"""

# Every subpackage, not a chosen two. `__getattr__` below refuses anything not
# named here, so a subpackage left off this list is unreachable through the
# package object -- `torchnative.adapt` raised `AttributeError` while
# `adapt/__init__.py` sat on disk with `wrap` in it, and the same for the other
# six. `from torchnative import adapt` always worked, which is why nothing
# noticed: the import system binds the submodule onto the parent as a side
# effect, and every example in this tree is spelled that way.
#
# Written out rather than scanned off `__path__`: a directory listing at import
# time is a stat call per entry on a road that exists to be cheap, and it would
# also make `__all__` depend on what happens to be installed. Added here is
# added on purpose. `rust/torch_c/pytests/test_tnnamespace.py` compares this
# list against the disk and fails if a new subpackage is not added.
__all__ = [
    "adapt",
    "api",
    "delta",
    "device",
    "distributed",
    "export",
    "kernels",
    "nn",
    "quant",
    "transformers",
]


def __getattr__(name):
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
