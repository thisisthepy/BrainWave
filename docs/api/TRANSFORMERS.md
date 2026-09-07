# `torchnative.transformers` — the `Auto*` family, keeping transformers' names

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/transformers/__init__.py covered present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/transformers/__init__.py ShadowedAutoClassWarning present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/transformers/__init__.py UnsupportedArgument present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/transformers/__init__.py _refuse_unsupported present -->
<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/transformers/__init__.py _auto_classes present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tntransformers.py test_the_family_is_enumerated_not_hand_listed present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tntransformers.py test_from_config_returns_a_real_nn_module_that_backprops present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tntransformers.py test_export_refuses_by_name present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tntransformers.py test_load_in_4bit_refuses_by_name present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tntransformers.py test_the_refusals_come_before_any_resolution present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_tntransformers.py test_the_reverse_order_is_not_detected_and_this_measures_that present -->

## 0. The diff this API exists to be

    - from transformers import AutoModelForCausalLM
    + from torchnative.transformers import AutoModelForCausalLM
      import torchnative

      model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it")
      model(**batch, labels=labels).loss.backward()   # a real nn.Module
      model.to(torchnative.device.npu)                # recompiles for the accelerator

One line. The class keeps transformers' own name and the module path
disambiguates. `optimum` prefixes its classes (`OVModelForCausalLM`) because
`optimum.intel` hosts several backends in one namespace; that pressure does not
exist here. §4 is the cost of that choice, stated plainly.

## 1. What is subclassed, and why that is the whole value

`AutoModelForCausalLM` and its siblings are **factories, not `nn.Module`s**.
`_BaseAutoModelClass.__init__` raises:

    OSError: AutoModelForCausalLM is designed to be instantiated using the
    `AutoModelForCausalLM.from_pretrained(...)` or `.from_config(config)` methods.

So subclassing does not subclass a model. What is inherited is the
**config-to-architecture dispatch** — `_model_mapping`, `from_config`, and
`from_pretrained`'s checkpoint resolution — which is the entire value of
`Auto*`. `test_what_is_inherited_is_the_config_to_architecture_dispatch`
asserts `_model_mapping` is upstream's object, not a copy.

## 2. The family is enumerated, not hand-listed

`_auto_classes()` walks `transformers.models.auto.modeling_auto` and takes every
public name bound to a `_BaseAutoModelClass` subclass. On the pinned
transformers (5.15.1) that is **49 classes**, and `covered()` returns all 49 —
`AutoModel`, `AutoModelForCausalLM`, `AutoModelForSeq2SeqLM`,
`AutoModelForSequenceClassification`, `AutoBackbone`, and the other 44.

A hand-written list is the CLAUDE.md §5.4 trap: it answers "which did the author
think of", not "which exist". `test_the_family_is_enumerated_not_hand_listed`
computes the population independently in the test and requires equality, so a
transformers release that adds a class is covered without an edit here — and a
regression to a hand-list fails (§6, N10).

Access is through PEP 562 `__getattr__` with a cache, which is also what makes
the shadow check in §4 possible.

## 3. `from_pretrained` returns the real model

Not a wrapper. `optimum` wraps the model in an inference object, which is why it
cannot backprop; it has no choice, because it runs on somebody else's `torch`.
**This project ships its own `torch`**, so the model that trains and the model
that runs on an accelerator stay one `nn.Module`.

Measured, on the shim, through `from_config` (no download; it exercises the same
inherited dispatch):

| | |
|---|---|
| returned class | `transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel` |
| `isinstance(model, nn.Module)` | `True` |
| class's `__module__` | `transformers.models.…` — **not** `torchnative.…` |
| `loss.backward()` | **16 of 16** parameters received a gradient |
| `model.to(torchnative.device.cpu) is model` | `True` |

`test_from_config_returns_a_real_nn_module_that_backprops` checks both halves:
that the class comes from `transformers.models.*` (a wrapper would not) and that
backward populates every gradient (a wrapper could not).

## 4. The shadowing hazard — one direction is caught, one is not

Because the class name is identical to transformers', a module importing both
silently keeps whichever came last. `__module__` distinguishes them in a
traceback but **not while reading the source**.

**Caught — transformers first, torchnative second:**

    from transformers import AutoModelForCausalLM
    from torchnative.transformers import AutoModelForCausalLM   # ShadowedAutoClassWarning

Attribute access goes through this module's `__getattr__`, which inspects the
importing frame's globals; if the name is already bound to the transformers
counterpart it warns, naming both and suggesting `import torchnative.transformers
as tnt`. `test_shadowing_is_detected_when_transformers_was_imported_first`
holds it; `test_no_warning_when_there_is_nothing_to_shadow` keeps it from
becoming noise.

**Not caught — torchnative first, transformers second:**

    from torchnative.transformers import AutoModelForCausalLM
    from transformers import AutoModelForCausalLM   # silent

Nothing of ours runs: CPython's import machinery rebinds the global directly.
Closing it would need an audit hook or a module-globals proxy, neither of which
is cheap, and neither was done.

This hole is **measured rather than described**.
`test_the_reverse_order_is_not_detected_and_this_measures_that` asserts the name
ends up bound to transformers' class and that no warning fires — so if a later
round closes the hole, that test goes red and the round updates this section,
rather than the hole quietly outliving its documentation.

**The unambiguous spelling, either way, is the module import:**

    import torchnative.transformers as tnt
    model = tnt.AutoModelForCausalLM.from_pretrained(...)

## 5. `export=` and `load_in_4bit=` refuse by name

Both appear in the README example. **Neither is implemented, and both refuse.**
CLAUDE.md §6: a promised refusal that does not happen is worse than no refusal,
and an argument accepted and dropped is the worst outcome available — the caller
would believe something untrue and have nothing to check.

`_refuse_unsupported` runs **before** the delegation, on `from_pretrained` and
`from_config`, for **all 49 classes**:

| argument | refusal |
|---|---|
| `export=` | Not implemented. It would mean "lower this checkpoint to an accelerator graph while loading it". The capture layer it needs exists (`torchnative.export.decompose` / `refold`); the step that turns a captured graph into a module leaf does not — the same wall `model.to(torchnative.device.npu)` reports ([`../devices/DEVICE_NS.md`](../devices/DEVICE_NS.md) §5.5). |
| `load_in_4bit=` | Not implemented. There is no 4-bit path: candle-core 0.11's `DType` has no `I8`, so the storage does not exist ([`../graph/QUANT.md`](../graph/QUANT.md) §2.1). The refusal names what *does* work — `torchnative.quant.quantize_(model, format="q8_0")`, or `torchnative.quant.TorchnativeConfig`. |

`test_the_refusals_come_before_any_resolution` passes a model id that does not
exist: if the check were placed after resolution began, the error would be an
HTTP or repository error instead of ours, and the test would go red.
`test_every_covered_class_refuses_the_unsupported_arguments` runs both arguments
against all 49.

## 6. Nullification

| # | nullification | caught by |
|---|---|---|
| N8 | `export=` accepted and silently dropped | `test_export_refuses_by_name`, `test_the_refusals_come_before_any_resolution` |
| N9 | `load_in_4bit=` accepted and silently dropped | `test_load_in_4bit_refuses_by_name`, and the family-wide test |
| N10 | the family hand-listed to four classes | `test_the_family_is_enumerated_not_hand_listed` |
| N12 | the shadow warning never fires | `test_shadowing_is_detected_when_transformers_was_imported_first` |
| N15 | `__getattr__` hands back transformers' class instead of our subclass | `test_the_four_named_in_the_readme_are_present_and_subclass_upstream` |
| N17 | `from_config` skips the refusal check | **initially NOT caught** — every refusal test went through `from_pretrained`, so a refusal on one of the two inherited entry points was enough to pass. Closed by `test_from_config_refuses_the_unsupported_arguments_too`. |

## 7. What this round did not verify

* **No checkpoint was downloaded.** Everything model-shaped went through
  `from_config` with a tiny GPT-2 config. `from_pretrained` is exercised only on
  its refusal path. The inherited dispatch is the same code, but "a real
  checkpoint loads and generates" is **not** claimed here.
* **`to(torchnative.device.npu)` on one of these models refuses**, so no
  transformer has been compiled for or executed on any NPU through this API.
* **Only the pinned transformers 5.15.1 was enumerated.** The enumeration is
  computed rather than fixed, so a different version is covered by construction,
  but no other version was run.
