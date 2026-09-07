"""`torchnative.transformers`: the `Auto*` family. See docs/api/TRANSFORMERS.md.

**The claim under test is "the real model, not a wrapper".** That is what
separates this from `optimum`, which returns an inference object and therefore
cannot backprop. So the load-bearing test is not that `from_pretrained` runs ---
it is `test_from_config_returns_a_real_nn_module_that_backprops`, which checks
the returned object's class comes from `transformers.models.*` (not from
`torchnative`), and then runs `loss.backward()` and counts populated grads.
A wrapper would fail both halves.

**No checkpoint is downloaded.** `from_config` exercises exactly the inherited
machinery `from_pretrained` uses to pick an architecture --- the
config-to-architecture dispatch is the whole value of `Auto*` and is what is
being subclassed --- without the network. `test_the_refusals_come_before_any_
resolution` covers the `from_pretrained` entry point with a model id that would
fail if it were ever reached.

Nullifications this file catches:

* the family hand-listed instead of enumerated -> `test_the_family_is_enumerated_not_hand_listed`
* a wrapper returned instead of the model      -> `test_from_config_returns_a_real_nn_module_that_backprops`
* `export=`/`load_in_4bit=` accepted and dropped -> `test_export_refuses_by_name`, `test_load_in_4bit_refuses_by_name`
* a refusal placed after the download           -> `test_the_refusals_come_before_any_resolution`
"""

import os
import sys
import warnings

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")
sys.path.insert(0, _VENDOR_DIR)

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def _shim():
    assert hasattr(torch._C, "_aten_implemented"), (
        "not the torchnative shim -- torch._C has no _aten_implemented"
    )


def _tnt():
    import torchnative.transformers as tnt

    return tnt


def _upstream():
    import transformers

    return transformers


# --------------------------------------------------------------------------
# The family
# --------------------------------------------------------------------------


def test_the_family_is_enumerated_not_hand_listed():
    """Every `_BaseAutoModelClass` in `modeling_auto` must be reachable.

    Enumeration is the point: a hand-written list answers "which did the author
    think of" rather than "which exist" (CLAUDE.md section 5.4). This computes
    the population independently and requires the module to cover all of it.
    """
    _shim()
    tnt = _tnt()
    from transformers.models.auto import modeling_auto
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    population = {
        n
        for n in dir(modeling_auto)
        if not n.startswith("_")
        and isinstance(getattr(modeling_auto, n), type)
        and issubclass(getattr(modeling_auto, n), _BaseAutoModelClass)
        and getattr(modeling_auto, n) is not _BaseAutoModelClass
    }
    covered = set(tnt.covered())
    assert covered == population, (
        f"not covered: {sorted(population - covered)}; "
        f"extra: {sorted(covered - population)}"
    )
    assert len(covered) >= 40, len(covered)
    print(f"ok   tntransformers: {len(covered)} of {len(population)} Auto* classes covered")


def test_the_four_named_in_the_readme_are_present_and_subclass_upstream():
    _shim()
    tnt = _tnt()
    up = _upstream()
    named = [
        "AutoModel",
        "AutoModelForCausalLM",
        "AutoModelForSeq2SeqLM",
        "AutoModelForSequenceClassification",
    ]
    for name in named:
        ours = getattr(tnt, name)
        theirs = getattr(up, name)
        assert ours is not theirs, f"{name}: got transformers' class, not ours"
        assert issubclass(ours, theirs), f"{name} does not subclass transformers'"
        assert ours.__name__ == theirs.__name__ == name
        assert ours.__module__ == "torchnative.transformers", ours.__module__
        assert ours._torchnative_upstream is theirs
    print(f"ok   tntransformers: {named} keep their names and subclass transformers'")


def test_what_is_inherited_is_the_config_to_architecture_dispatch():
    """`Auto*` are factories, not modules; the mapping is the value."""
    _shim()
    tnt = _tnt()
    up = _upstream()
    ours = tnt.AutoModelForCausalLM
    assert ours._model_mapping is up.AutoModelForCausalLM._model_mapping, (
        "the subclass must inherit upstream's mapping, not shadow it"
    )
    try:
        ours()
    except OSError as exc:
        assert "from_pretrained" in str(exc), exc
    else:
        raise AssertionError("an Auto* class must refuse direct instantiation")
    print("ok   tntransformers: the mapping is inherited and direct construction refuses")


# --------------------------------------------------------------------------
# The real model
# --------------------------------------------------------------------------


def _tiny_config():
    from transformers import AutoConfig

    return AutoConfig.for_model(
        "gpt2", n_layer=1, n_head=2, n_embd=16, vocab_size=64, n_positions=16
    )


def test_from_config_returns_a_real_nn_module_that_backprops():
    """The claim that separates this from `optimum`."""
    _shim()
    tnt = _tnt()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = tnt.AutoModelForCausalLM.from_config(_tiny_config())

    assert isinstance(model, nn.Module), type(model)
    assert type(model).__module__.startswith("transformers.models."), (
        f"from_config returned {type(model).__module__}.{type(model).__name__} -- "
        f"if that lives under torchnative it is a wrapper, which is the one thing "
        f"this API exists not to be"
    )
    assert not isinstance(model, tnt.AutoModelForCausalLM), (
        "the factory must not return an instance of itself"
    )

    ids = torch.randint(0, 64, (1, 8))
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    params = list(model.parameters())
    with_grad = [p for p in params if p.grad is not None]
    assert len(params) > 0
    assert len(with_grad) == len(params), (
        f"only {len(with_grad)} of {len(params)} parameters got a gradient"
    )
    print(
        f"ok   tntransformers: from_config returned a real "
        f"{type(model).__name__}; loss.backward() populated "
        f"{len(with_grad)}/{len(params)} parameter grads"
    )


def test_the_returned_model_is_one_object_that_to_can_move():
    """Training model and accelerator model must stay the same `nn.Module`."""
    _shim()
    tnt = _tnt()
    import torchnative.device as D

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = tnt.AutoModelForCausalLM.from_config(_tiny_config())
    same = model.to(D.cpu)
    assert same is model, "to() must return the same object, never a wrapper"
    print("ok   tntransformers: to(device.cpu) returns the same nn.Module object")


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_export_refuses_by_name():
    _shim()
    tnt = _tnt()
    try:
        tnt.AutoModelForCausalLM.from_pretrained("gpt2", export=True)
    except NotImplementedError as exc:
        assert "export=" in str(exc), exc
        assert "AutoModelForCausalLM" in str(exc), exc
    else:
        raise AssertionError("export= must refuse, not be ignored")
    print("ok   tntransformers: export= refuses by name")


def test_load_in_4bit_refuses_by_name():
    _shim()
    tnt = _tnt()
    try:
        tnt.AutoModelForCausalLM.from_pretrained("gpt2", load_in_4bit=True)
    except NotImplementedError as exc:
        assert "load_in_4bit=" in str(exc), exc
        assert "q8_0" in str(exc), "the refusal should name what does work"
    else:
        raise AssertionError("load_in_4bit= must refuse, not be ignored")
    print("ok   tntransformers: load_in_4bit= refuses by name and names the 8-bit path")


def test_from_config_refuses_the_unsupported_arguments_too():
    """`from_config` is the other inherited entry point, and it must refuse too.

    Found by nullification: removing the check from `from_config` left every
    test green, because they all went through `from_pretrained`. A refusal on
    one of two doors is a door left open.
    """
    _shim()
    tnt = _tnt()
    cfg = _tiny_config()
    for kw in ({"export": True}, {"load_in_4bit": True}):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tnt.AutoModelForCausalLM.from_config(cfg, **kw)
        except NotImplementedError as exc:
            assert "AutoModelForCausalLM" in str(exc), exc
        else:
            raise AssertionError(f"from_config accepted {kw}")
    print("ok   tntransformers: from_config refuses export= and load_in_4bit= as well")


def test_the_refusals_come_before_any_resolution():
    """A refusal after the download is a refusal the user pays for.

    The model id here does not exist. If the refusal were placed after
    `from_pretrained` began resolving it, the error would be an HTTP or
    repository error rather than ours, and this test would go red.
    """
    _shim()
    tnt = _tnt()
    bogus = "torchnative-does-not-exist/nope-" + "x" * 24
    for kw in ({"export": True}, {"load_in_4bit": True}):
        try:
            tnt.AutoModelForCausalLM.from_pretrained(bogus, **kw)
        except NotImplementedError as exc:
            assert "torchnative.transformers" in str(exc), exc
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"{kw}: got {type(exc).__name__} instead of our refusal -- the "
                f"check is happening after resolution began: {str(exc)[:120]}"
            ) from None
        else:
            raise AssertionError(f"{kw} was accepted")
    print("ok   tntransformers: both refusals fire before the checkpoint is resolved")


def test_every_covered_class_refuses_the_unsupported_arguments():
    """Support the family, not one class: the refusals must be family-wide."""
    _shim()
    tnt = _tnt()
    checked = 0
    for name in tnt.covered():
        cls = getattr(tnt, name)
        for kw in ({"export": True}, {"load_in_4bit": True}):
            try:
                cls.from_pretrained("x", **kw)
            except NotImplementedError as exc:
                assert name in str(exc), (name, str(exc)[:80])
            else:
                raise AssertionError(f"{name}: {kw} accepted")
        checked += 1
    print(f"ok   tntransformers: all {checked} classes refuse export= and load_in_4bit=")


# --------------------------------------------------------------------------
# Shadowing
# --------------------------------------------------------------------------


def test_shadowing_is_detected_when_transformers_was_imported_first():
    _shim()
    tnt = _tnt()
    up = _upstream()
    ns = {"__name__": "shadow_demo", "AutoModelForCausalLM": up.AutoModelForCausalLM}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec("from torchnative.transformers import AutoModelForCausalLM", ns)
    kinds = [type(w.message) for w in caught]
    assert tnt.ShadowedAutoClassWarning in kinds, kinds
    assert ns["AutoModelForCausalLM"] is tnt.AutoModelForCausalLM
    print("ok   tntransformers: shadowing is warned when transformers came first")


def test_no_warning_when_there_is_nothing_to_shadow():
    """The warning must not fire on an ordinary import, or it is noise."""
    _shim()
    tnt = _tnt()
    ns = {"__name__": "clean_demo"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec("from torchnative.transformers import AutoModelForSeq2SeqLM", ns)
    kinds = [type(w.message) for w in caught]
    assert tnt.ShadowedAutoClassWarning not in kinds, kinds
    print("ok   tntransformers: no shadow warning when the name was free")


def test_the_reverse_order_is_not_detected_and_this_measures_that():
    """The known hole, measured rather than described.

    ours first, transformers second: CPython rebinds the global directly and
    nothing of ours runs, so no warning is possible without an audit hook. This
    asserts the hazard *exists* --- if a later round closes it, this test goes
    red and that round updates docs/api/TRANSFORMERS.md section 4 rather than
    the hole quietly outliving its documentation.
    """
    _shim()
    tnt = _tnt()
    up = _upstream()
    ns = {"__name__": "reverse_demo"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec("from torchnative.transformers import AutoModelForCausalLM", ns)
        exec("from transformers import AutoModelForCausalLM", ns)
    assert ns["AutoModelForCausalLM"] is up.AutoModelForCausalLM
    assert tnt.ShadowedAutoClassWarning not in [type(w.message) for w in caught], (
        "the reverse order is now detected -- update docs/api/TRANSFORMERS.md "
        "section 4, which says it is not"
    )
    print(
        "ok   tntransformers: reverse-order shadowing is silent (measured hole, "
        "recorded in docs/api/TRANSFORMERS.md section 4)"
    )


def test_module_import_is_the_unambiguous_spelling():
    _shim()
    tnt = _tnt()
    up = _upstream()
    assert tnt.AutoModelForCausalLM.__module__ != up.AutoModelForCausalLM.__module__
    print(
        "ok   tntransformers: __module__ distinguishes them "
        f"({tnt.AutoModelForCausalLM.__module__} vs "
        f"{up.AutoModelForCausalLM.__module__})"
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
