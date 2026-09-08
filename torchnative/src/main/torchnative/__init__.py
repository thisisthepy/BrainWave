"""torchnative — on-device test-time learning and federated learning.

See docs/design/DESIGN.md for the design and its reasoning.

`torchnative.device` and `torchnative.transformers` are resolved lazily,
through PEP 562, for the reason `torchnative.quant` already gives: importing
this package must stay cheap and must not drag in `torch` (let alone
`transformers`, which is not a hard dependency) for someone who imported it
for something else. `import torchnative; torchnative.device.npu` works because
the attribute lookup lands here and imports the submodule then.

That laziness is also what makes the `nn.Module.to` patch safe to install at
`torchnative.device` import time: see `device/_module_to.py`.
"""

__all__ = ["device", "transformers"]


def __getattr__(name):
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
