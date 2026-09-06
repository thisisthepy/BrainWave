# `from_pretrained(dtype=torch.int8)` — where the request dies, and what is left of it

Round date 2026-09-06. Branch `work/scalar3`, on develop `479b3cf`. Host Apple M1,
CPython 3.13, transformers 5.15.1, upstream torch 2.13.0, `candle-core` 0.11.0.
**No Rust changed this round.** `transformers` was read, never edited.

The ask, in the user's words:

> `quant.quantize` 형태가 아니라 그냥 모델 로드할 때 dtype을 torch.int8 이렇게 넣어주면
> 자동으로 그렇게 처리하면 되잖아. `AutoModel.from_pretrained(dtype=torch.bf16)` 이 자리에서의
> 역할을 말하는거잖아

That is: `AutoModelForCausalLM.from_pretrained(name, dtype=torch.int8)`, in the same
slot where `dtype=torch.bfloat16` works.

> **Answer first.** The spelling splits in two and only one half is reachable from this
> repository.
>
> | spelling | today | after this round |
> |---|---|---|
> | `from_pretrained(name, dtype=torch.int8)` | `ValueError` from transformers | **unchanged — `ValueError` from transformers** |
> | `from_pretrained(name, dtype=torch.int8, quantization_config=TorchnativeConfig("q8_0"))` | `ValueError` from **us**, saying "cannot run int8 activations" | **loads**, `q8_0` weights, `float32` activations, bit-identical to the same config at `float32` |
>
> The bare form is closed by `transformers.modeling_utils.local_torch_dtype`, and the
> only hook that could have redirected it (`get_hf_quantizer`) runs **earlier** and
> reads **only** `quantization_config`. **Nothing in transformers is keyed on `dtype`,**
> so there is no public seam a bare `dtype=torch.int8` could reach. Opening it means
> editing `transformers`, which is the one thing `docs/DESIGN.md` §1 rules out. §1 is
> the measurement; §5 sizes what opening it would cost.

---

## 1. What `transformers` does with `dtype=` before it reaches us

This is the question that decides whether the feature is buildable at all, so it is
first, and it is read out of `transformers 5.15.1` rather than remembered.

### 1.1 The order inside `from_pretrained`

`modeling_utils.py` `PreTrainedModel.from_pretrained`, in source order:

```
  4225   hf_quantizer, config, device_map = get_hf_quantizer(
             config, quantization_config, device_map, weights_only, user_agent)
  4263   config, dtype = _get_dtype(
             dtype, checkpoint_files, config, sharded_metadata, state_dict,
             weights_only, hf_quantizer)
  ...
  3773   init_contexts = [local_torch_dtype(dtype, cls.__name__), ...]
```

**The quantizer is chosen before the dtype is looked at.** `get_hf_quantizer`
(`quantizers/auto.py:330`) takes its decision from exactly two places:

```python
quantization_params_from_config = getattr(config, "quantization_config", None) or ...
if pre_quantized or quantization_config is not None:
    ...
else:
    hf_quantizer = None
```

`dtype` is not a parameter of that function. So with `dtype=torch.int8` and no
`quantization_config=`, `hf_quantizer` is `None` — decided, irrevocably, before
anything about `int8` has been read.

### 1.2 `_get_dtype` does not refuse — it delegates

`_get_dtype` (`modeling_utils.py:816`) resolves `"auto"`, rejects a `dtype` that is
neither `str`, `dict` nor `torch.dtype`, and then does this:

```python
if hf_quantizer is not None:
    dtype = hf_quantizer.update_dtype(dtype)
```

**`torch.int8` passes through `_get_dtype` untouched.** It is a `torch.dtype`, so the
type check accepts it, and there is no float check anywhere in the function.

> `docs/HFQUANT.md` §1 named `_get_dtype` as the refusal site. **That is wrong and this
> document corrects it.** The correction is not cosmetic: `_get_dtype` is *before*
> `update_dtype` in the sense HFQUANT meant, but the refusal is *after* it, and that
> gap is the entire feature below. HFQUANT read the message and inferred the site.

### 1.3 Where it actually dies

`modeling_utils.py:222`, entered from the `init_contexts` list at line 3773:

```python
@contextmanager
def local_torch_dtype(dtype, model_class_name=None):
    if not dtype.is_floating_point:
        raise ValueError(
            f"{model_class_name} cannot be instantiated under `dtype={dtype}` "
            "as it's not a floating-point dtype")
    torch.set_default_dtype(dtype)
```

The guard exists because the next line is `torch.set_default_dtype`, which cannot take
an integer dtype in upstream torch either. On the shim, `torch.int8.is_floating_point`
is `False` and `is_signed` is `True` and `itemsize` is `1` — all correct, and nothing
here should change that.

Measured, on the shim, transformers 5.15.1, `HuggingFaceTB/SmolLM2-135M`:

```
AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.int8)
ValueError: LlamaForCausalLM cannot be instantiated under `dtype=torch.int8`
            as it's not a floating-point dtype
```

The regression test asserts the traceback contains a frame named `local_torch_dtype`,
that some frame is inside `transformers`, and that **no** frame is inside
`torchnative` — so a later round cannot mistake this for something this repository
broke, and cannot "fix" it in the wrong file.

### 1.4 So: the honest answer is not one of the three offered

The framing offered three answers — map it to q8_0, refuse by name, or lie. **On the
bare spelling none of them is available**, because no code in this repository runs.
Refusing "by name" would already be an improvement over what happens, and we cannot
even do that: the message the user sees is transformers' own, and it names
`torch.int8` but not `q8_0`, not `TorchnativeConfig`, and not candle.

What *is* available is the second spelling, and §2 takes it.

---

## 2. What was built: `dtype=torch.int8` next to the config

`update_dtype` is a public `HfQuantizer` hook and it is ours. §1.2 shows `_get_dtype`
calls it **before** `local_torch_dtype` is reached, so a quantizer that widens an
integer dtype makes the guard a non-event.

Before this round, that hook refused:

```
ValueError: torchnative quantisation cannot run torch.int8 activations: candle's
QMatMul accepts float32 and float16 only. Pass dtype=torch.float32 to from_pretrained.
```

**That refusal is technically true and reads as a category error on the user's part,
which it is not.** They did not ask for int8 *activations*. They asked for an int8
*model*, and the config in their own call already delivers the nearest thing this stack
has. Now it is accepted, widened, and disclosed:

```python
_WEIGHT_DTYPES = (torch.int8,)
```

<!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/quant/hf.py _WEIGHT_DTYPES present -->

```
dtype=torch.int8 was read as a request about *weights*, not activations. The weights
are being quantised to q8_0 by the quantization_config, which is the nearest thing this
stack has to it -- there is no torch.int8 tensor here at all, because candle-core 0.11
has no I8 dtype (docs/INT8.md §1). Activations are loaded in float32, so model.dtype
will report torch.float32 and not the dtype you passed; model.torchnative_quantization
records what actually happened.
```

**Only `torch.int8`.** `torch.uint8` is not widened — `q8_0` is a *signed* 8-bit block
format, so `uint8` is a different request and keeps the old refusal. `torch.float64`
keeps it too. Both are asserted.

---

## 3. What the user gets, on SmolLM2-135M

`HuggingFaceTB/SmolLM2-135M`, three loads, same prompt, greedy 20 tokens.

| | `dtype=torch.int8` + `TorchnativeConfig("q8_0")` | `dtype=torch.float32` + same config | `dtype=torch.float32`, no config |
|---|---|---|---|
| loads | yes | yes | yes |
| `QuantizedLinear` / `Linear` | **210 / 1** | 210 / 1 | 0 / 211 |
| `model.dtype` | `torch.float32` | `torch.float32` | `torch.float32` |
| `p.dtype` over all parameters | `{torch.float32}` | `{torch.float32}` | `{torch.float32}` |
| logits sum | `853834.62500000` | `853834.62500000` | `603999.12500000` |
| logits `[0,-1,:3]` | `3.036725, -12.787579, -12.650931` | identical | `2.598259, -13.394009, -13.257526` |

**`dtype=torch.int8` and the explicit `quantization_config` produce the same model.**
Not "close" — the same. They are one path: `update_dtype` returns `float32` and
everything downstream is the load `docs/HFQUANT.md` already measured. The dense column
is the negative control; it differs, so the identity above is evidence of something.

Generation, all three:

```
'The capital of France is the capital of the country.\n\nThe capital of France is the
 capital of the country.\n'
```

(SmolLM2-135M at greedy decoding is repetitive on this prompt in every configuration
including dense float32. It is not a quantisation artefact, and it is not a quality
claim either way — one prompt, per `docs/QUANT2.md` §5.3.)

### 3.1 Can the user tell what they loaded?

This is the part that decides whether the mapping is honest, and `model.dtype` alone
fails it: it says `torch.float32` for a user who typed `torch.int8`. Three other
surfaces carry the truth, and one of them is unprompted.

| surface | what it says |
|---|---|
| a warning at load, unprompted | the text in §2 — names `q8_0`, names candle's missing `I8`, and says in advance that `model.dtype` will read `float32` |
| `str(model)` | `QuantizedLinear(in_features=576, out_features=576, bias=False, format=q8_0)` × 210, and one plain `Linear` for `lm_head` |
| `model.torchnative_quantization` | the load report, now with a line naming the dtype that was asked for |

```
format=q8_0 converted=210 left dense=1
  modules_to_not_convert=['lm_head', 'model.embed_tokens'] -- transformers' default
    (get_keys_to_not_convert: output embedding, last parameter, tied weights)
  every replaced leaf was swapped before its weight had storage
  caller asked for dtype=torch.int8; there is no such tensor on this stack, so the
    weights are q8_0 and the activations are torch.float32 (model.dtype says float32)
  left dense (1): modules_to_not_convert
    e.g. lm_head
```

**What is still not visible from `model.dtype`.** A caller who writes
`dtype=torch.int8`, suppresses warnings and then reads `model.dtype` gets `float32`
with no indication. That is transformers' contract, not a choice made here —
`model.dtype` reads `config.dtype`, which is the *activation* dtype, and every
quantised model in transformers behaves this way (bitsandbytes 8-bit reports `float16`).
It is recorded rather than fixed.

---

## 4. What is guarded

`test_dtype_int8_is_refused_by_transformers_alone_and_works_with_the_config` in
`rust/torch_c/pytests/test_shim.py`, running on the local 2-layer tied fixture in a
subprocess (the existing `_hfquant_fixture` harness, one new mode).

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_dtype_int8_is_refused_by_transformers_alone_and_works_with_the_config present -->

| assertion | what it would catch |
|---|---|
| bare `int8` raises, message says "not a floating-point dtype" | someone quietly making `int8` loadable as float32 — the dishonest option |
| the traceback has a `local_torch_dtype` frame, has a `transformers` frame, has **no** `torchnative` frame | this repository taking the blame, or the credit, for a refusal that is not its |
| `int8` + config logits **bit-identical** to `float32` + config | two paths where there should be one |
| a **dense** load's logits differ | the line above comparing two values that could not have differed |
| `n_quantized == 14`, `model_dtype == "torch.float32"`, `param_dtypes == ["torch.float32"]` | a claim of quantisation with nothing quantised |
| the report string contains `torch.int8` **and** `q8_0` | the disclosure being dropped while the widening stays |
| `uint8` and `float64` still refuse, with candle's message | the widening spreading to dtypes it does not describe |

---

## 5. Sizing the half that is not ours

For the bare `from_pretrained(name, dtype=torch.int8)` to work, `transformers` has to
grow a seam. There is no smaller version: §1.1 shows the quantizer is picked before the
dtype is read, and §1.3 shows the guard sits in front of `torch.set_default_dtype`.

| approach | size | why it is not done here |
|---|---|---|
| Runtime monkeypatch of `modeling_utils` from `torchnative.quant` | ~30 lines | An import that changes what an unrelated `from_pretrained` does. `docs/HFQUANT.md` §7 went out of its way to guarantee the opposite, and there is a regression test asserting it (`test_the_quantizer_registers_a_name_and_changes_nothing_else`). |
| Upstream PR: a dtype→quantizer registry in `get_hf_quantizer` | small in transformers, unknown in review | Not this repository's to land, and it would be a new public concept in transformers (`dtype` selecting a quantiser) rather than a bug fix. |
| A real `torch.int8` storage, so the guard is the only obstacle left | **+252 lines in a forked `candle-core`, +7 here** (`docs/INT8.md` §2, §3) | Still would not make the bare spelling work — `local_torch_dtype` refuses on `is_floating_point`, not on whether the dtype exists. **This is the important one:** landing the candle patch does **not** unlock the user's spelling. |

The last row is the finding worth carrying forward. `docs/INT8.md` sized the storage
half carefully and left it unlanded; **this round establishes that landing it would not
have delivered this request anyway.** The blocker is transformers' float check, not
candle's `DType`.

---

## 6. What this round did not establish

- **No timings.** Six other agents were running. Nothing here is a performance claim.
- **No memory measurement.** `docs/HFQUANT.md` §2's peak-RSS numbers apply unchanged —
  §3 shows the `int8` spelling produces a bit-identical model to the one measured there
  — but they were not re-taken.
- **No perplexity.** One prompt, as in `docs/QUANT2.md` §5.3. Accuracy for this model
  is `docs/HFQUANT.md` §3's table and is unmoved.
- **Llama only, host only.** No device, no other architecture.
- **`dtype="int8"` as a string** was not probed. `_get_dtype` resolves a string through
  `getattr(torch, dtype)`, so it should reach the same place; not measured.
- **Whether upstream transformers would take a dtype→quantizer seam.** Not asked.

---

## 7. Report classification

`CLAUDE.md` §5.3.

| kind | what |
|---|---|
| **feature added** | `TorchnativeHfQuantizer.update_dtype` accepts `torch.int8` beside a `TorchnativeConfig`, widening activations to `float32` with a disclosure; `_LoadReport` carries the requested dtype |
| **defect fixed** | none. **Golden 8681/8681, ops=207, unmoved — no Rust changed** |
| **test added** | 1 (415 → 416), plus one new mode in the existing subprocess fixture |
| **documentation corrected** | `docs/HFQUANT.md` §1 and `torchnative/quant/hf.py`'s module docstring both named `_get_dtype` as the refusal site. It is `local_torch_dtype`, and the difference is what made §2 possible |
| **deleted** | none |
| **not built** | the bare `from_pretrained(dtype=torch.int8)`. Not reachable from this repository (§1), sized in §5 |

<!-- DOCWATCH: count smoke_ok ge 416 -->
