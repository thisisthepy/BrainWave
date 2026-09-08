def test_import_torchnative():
    import torchnative  # noqa: F401


def test_import_submodules():
    from torchnative import adapt, delta, kernels  # noqa: F401


def test_import_export_does_not_need_torch():
    """`torchnative.export` must import without a `torch` on the interpreter.

    Same rule `torchnative.api` follows, for the reason its docstring gives:
    this distribution *provides* torch rather than depending on one, so a
    module that reaches for it at import time fails on a correctly-installed
    wheel. Every `import torch` in the decomposition pass is inside a function,
    and this is what says so.
    """
    from torchnative import export

    assert callable(export.decompose)
    assert callable(export.core_ops)
