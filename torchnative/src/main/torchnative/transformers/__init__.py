"""`torchnative.transformers` --- the `Auto*` family, keeping transformers' names.

    - from transformers import AutoModelForCausalLM
    + from torchnative.transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it")
    model(**batch, labels=labels).loss.backward()   # a real nn.Module
    model.to(torchnative.device.npu)                # recompiles for the accelerator

**The class keeps transformers' own name.** The module path already
disambiguates, so a user's diff is the import line and nothing else. `optimum`
prefixes (`OVModelForCausalLM`) because `optimum.intel` hosts several backends
in one namespace; that pressure does not exist here.

**What is inherited, and why it is the whole point.** `AutoModelForCausalLM`
and its siblings are **factories, not `nn.Module`s** ---
`_BaseAutoModelClass.__init__` raises `OSError` telling you to use
`from_pretrained`. So subclassing does not subclass a model; it inherits the
config-to-architecture dispatch (`_model_mapping`, `from_config`,
`from_pretrained`'s resolution), which is the entire value of `Auto*`. The
object that comes back is upstream's real model class, unmodified and
unwrapped.

**`from_pretrained` returns the real model.** Not a wrapper. `optimum` returns
an inference object and therefore cannot backprop; it has no choice, because it
runs on somebody else's `torch`. This project ships its own, so the model that
trains and the model that runs on an accelerator are one `nn.Module`, and
`loss.backward()` works on it. See `torchnative/device/_module_to.py`.

**How the family is enumerated.** Not by hand. `_auto_classes()` walks
`transformers.models.auto.modeling_auto` and takes every public name bound to a
`_BaseAutoModelClass` subclass --- 49 of them in the pinned transformers ---
so a transformers release that adds one is covered without an edit here. A
hand-written list is the CLAUDE.md section 5.4 trap: it answers "which did I
think of", not "which exist".

**The shadowing hazard.** Because the class name is identical to
transformers', a module that imports both silently keeps whichever came last:

    from transformers import AutoModelForCausalLM
    from torchnative.transformers import AutoModelForCausalLM   # rebinds

`__module__` tells them apart in a traceback but not while reading the source.
This module **detects the case it can detect**: attribute access here inspects
the importing frame's globals, and if the name is already bound to the
transformers counterpart it emits a `ShadowedAutoClassWarning` naming both.
That catches `transformers` first, `torchnative` second.

It does **not** catch the reverse order --- ours first, transformers second ---
because at that point nothing of ours runs; CPython's import machinery rebinds
the global directly. There is no cheap hook for it (it would need an audit hook
or a module-globals proxy), and it is stated here rather than left undiscussed.
`docs/api/TRANSFORMERS.md` section 4 records both directions and which one is
covered.
"""

import sys
import warnings

_CACHE = {}
_MODELING_AUTO = "transformers.models.auto.modeling_auto"


class ShadowedAutoClassWarning(UserWarning):
    """Both this module's and transformers' class of the same name are in scope."""


class UnsupportedArgument(NotImplementedError):
    """An argument this build cannot honour, refused by name rather than dropped."""


def _modeling_auto():
    import importlib

    return importlib.import_module(_MODELING_AUTO)


def _base():
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    return _BaseAutoModelClass


def _auto_classes():
    """`{name: transformers class}` for every `Auto*` factory, enumerated."""
    module = _modeling_auto()
    base = _base()
    out = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, base) and obj is not base:
            out[name] = obj
    return out


# --------------------------------------------------------------------------
# The refusals
# --------------------------------------------------------------------------


def _refuse_unsupported(cls_name, kwargs):
    """Refuse `export=` and `load_in_4bit=` by name, before anything else.

    Both appear in this project's README example. Neither is implementable in
    this build, and CLAUDE.md section 6 is explicit that a promised refusal
    which does not happen is worse than no refusal: an argument accepted and
    silently dropped leaves the caller believing something that is not true.
    So they are checked here, ahead of the delegation, and named.
    """
    if "export" in kwargs:
        raise UnsupportedArgument(
            f"torchnative.transformers.{cls_name}.from_pretrained: `export=` is "
            f"not implemented in this build and is refused rather than ignored.\n"
            f"\n"
            f"`export=True` would mean 'lower this checkpoint to an accelerator "
            f"graph while loading it'. The capture layer it needs exists "
            f"(torchnative.export.decompose / refold), and the step that turns a "
            f"captured graph into a module leaf does not -- the same wall "
            f"`model.to(torchnative.device.npu)` reports. See "
            f"docs/api/TRANSFORMERS.md section 5."
        )
    if "load_in_4bit" in kwargs:
        raise UnsupportedArgument(
            f"torchnative.transformers.{cls_name}.from_pretrained: "
            f"`load_in_4bit=` is not implemented in this build and is refused "
            f"rather than ignored.\n"
            f"\n"
            f"There is no 4-bit path here: torchnative's quantisation is "
            f"module replacement, not a dtype (docs/graph/QUANT.md section 2.1 "
            f"-- candle-core 0.11's DType has no I8, so the storage does not "
            f"exist). What does work today is 8-bit, after loading:\n"
            f"    import torchnative.quant as q; q.quantize_(model, format='q8_0')\n"
            f"or transformers' own quantizer hook, "
            f"torchnative.quant.TorchnativeConfig. See "
            f"docs/api/TRANSFORMERS.md section 5."
        )


# --------------------------------------------------------------------------
# The subclasses
# --------------------------------------------------------------------------


def _build(name):
    upstream = _auto_classes().get(name)
    if upstream is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def from_pretrained(cls, *args, **kwargs):
        _refuse_unsupported(name, kwargs)
        return super(subclass, cls).from_pretrained(*args, **kwargs)

    def from_config(cls, *args, **kwargs):
        _refuse_unsupported(name, kwargs)
        return super(subclass, cls).from_config(*args, **kwargs)

    subclass = type(
        name,
        (upstream,),
        {
            "__module__": __name__,
            "__qualname__": name,
            "__doc__": (
                f"torchnative's {name}.\n\n"
                f"Subclasses `{upstream.__module__}.{name}`, so the "
                f"config-to-architecture dispatch is upstream's. "
                f"`from_pretrained` returns the real model --- a genuine "
                f"nn.Module on which loss.backward() works --- and refuses "
                f"`export=` and `load_in_4bit=` by name.\n\n"
                f"Note the name is identical to transformers'; importing both "
                f"in one module silently keeps one. See "
                f"docs/api/TRANSFORMERS.md section 4."
            ),
            "_torchnative_upstream": upstream,
            "from_pretrained": classmethod(from_pretrained),
            "from_config": classmethod(from_config),
        },
    )
    return subclass


def _warn_if_shadowing(name):
    """Warn when the importing module already has transformers' class bound.

    Catches `transformers` first, `torchnative` second. Cannot catch the
    reverse; see this module's docstring.
    """
    try:
        frame = sys._getframe(2)
    except ValueError:
        return
    existing = frame.f_globals.get(name)
    if existing is None or not isinstance(existing, type):
        return
    upstream = _auto_classes().get(name)
    if upstream is not None and existing is upstream:
        warnings.warn(
            f"{frame.f_globals.get('__name__', '<unknown>')} already binds "
            f"{name!r} to {upstream.__module__}.{name}; importing "
            f"{__name__}.{name} rebinds it. Both classes have the same name, "
            f"so nothing downstream will look wrong. Import the module "
            f"instead -- `import torchnative.transformers as tnt` and use "
            f"`tnt.{name}` -- or alias one of them.",
            ShadowedAutoClassWarning,
            stacklevel=3,
        )


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name not in _CACHE:
        _CACHE[name] = _build(name)
    _warn_if_shadowing(name)
    return _CACHE[name]


def __dir__():
    try:
        return sorted(set(list(globals()) + list(_auto_classes())))
    except Exception:  # noqa: BLE001
        return sorted(globals())


def covered():
    """The `Auto*` names this module provides, as enumerated from transformers."""
    return tuple(sorted(_auto_classes()))
