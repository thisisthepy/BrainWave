"""`from_pretrained(..., quantization_config=TorchnativeConfig("q8_0"))`.

    from torchnative.quant import TorchnativeConfig
    m = AutoModelForCausalLM.from_pretrained(name, quantization_config=TorchnativeConfig("q8_0"))

**`dtype=torch.int8` alone, and `dtype=torch.int8` with this config.** They are
not the same question, and only the second one is ours.

*Alone* it is closed, and not by us::

    AutoModelForCausalLM.from_pretrained(name, dtype=torch.int8)
    ValueError: LlamaForCausalLM cannot be instantiated under `dtype=torch.int8`
                as it's not a floating-point dtype

That is raised by `modeling_utils.local_torch_dtype`, which guards
`torch.set_default_dtype` and is entered from `from_pretrained` *after*
`get_hf_quantizer` has already decided there is no quantizer. `get_hf_quantizer`
reads only `quantization_config` and `config.quantization_config` -- nothing in
transformers is keyed on `dtype`, so there is no public hook that a bare
`dtype=torch.int8` could reach. Opening it means editing transformers, which is
the one thing this project does not do (`docs/DESIGN.md` §1: a facade defeats
the reason an embedded CPython is worth having). See `docs/INT8B.md` §1.

*With this config* it is ours, because `_get_dtype` calls
`hf_quantizer.update_dtype(dtype)` before that guard is reached, so a quantizer
can widen it. `update_dtype` below accepts `torch.int8` there, reads it as a
statement about weights rather than activations, and says so. It is not honoured
literally: `docs/QUANT.md` §2.1 and `docs/INT8.md` §1 close that door
independently -- candle-core 0.11's `DType` has no `I8`, at any version
including `main`, so there is no `torch.int8` *tensor* on this stack to load
into. What the caller gets is `q8_0` weights and `float32` activations, and
`model.torchnative_quantization` records that they asked for something else.

The slot transformers does provide is `quantization_config`, and it is a public
plugin API -- `transformers.quantizers.auto.register_quantizer` and
`register_quantization_config`. That is what this module fills.

**What it buys over `quantize_`.** `quantize_(model, ...)` runs after
`from_pretrained` returns, so the dense model exists in full before anything
shrinks; peak memory is the dense model no matter what format you pick. That is
survivable at 135M and self-defeating at 7B, and on-device is the premise of
this repository. `HfQuantizer._process_model_before_weight_loading` runs while
the model is still a meta-device skeleton, so the leaves can be swapped *before*
the checkpoint is read, and each weight is quantised as it comes off disk. The
dense tensor's only reference is the conversion op's local, so it is freed
immediately and the dense model is never assembled. `docs/HFQUANT.md` has the
measurement.

**What it does not change.** Everything in `torchnative/quant/__init__.py`'s
docstring still holds: this is module replacement, not a dtype; a replaced leaf
is all it covers; activations stay dense; there is no calibration and no
importance matrix. This module adds a door, not a method.
"""

import torch

from . import FORMATS, QuantizedLinear

# transformers is imported at module scope on purpose. `torchnative.quant`
# reaches this module only through its PEP 562 `__getattr__`, so a machine
# without transformers never gets here and `import torchnative.quant` keeps
# working there (`__init__.py` `__getattr__` says why).
from transformers.core_model_loading import ConversionOps
from transformers.quantizers.auto import register_quantization_config, register_quantizer
from transformers.quantizers.base import HfQuantizer
from transformers.quantizers.quantizers_utils import get_module_from_name, should_convert_module
from transformers.utils import logging
from transformers.utils.quantization_config import QuantizationConfigMixin

logger = logging.get_logger(__name__)

__all__ = ["TorchnativeConfig", "TorchnativeHfQuantizer", "QUANT_METHOD"]

QUANT_METHOD = "torchnative"

_MISSING_SHIM = (
    "torch._C has no `_quantize`. torchnative's quantisation lives in this "
    "repository's `torch._C` shim, not in upstream torch -- the import that is "
    "live is {module}. Put `torchnative/src/main` on PYTHONPATH "
    "(docs/VENDOR.md), or use a format-free load."
)


_TIED = (
    "{name}.weight is tied to another parameter, so the checkpoint does not contain "
    "it -- no tensor arrives for this layer, and there is nothing to quantise before "
    "the load. Quantising it afterwards would also break the tie and *cost* memory: "
    "the dense copy stays alive inside the embedding and the quantised head is an "
    "additional one (2.10x all-in against 2.38x leaving it dense, docs/QUANT2.md §5.4)."
)

_TIED_WAY_OUT = (
    "Leave it in modules_to_not_convert -- that is what the default does -- or call "
    "torchnative.quant.quantize_ on the loaded model and take both costs knowingly."
)

# Integer dtypes a caller may reasonably put in `dtype=` meaning "quantise the
# weights". Only `int8` is here: `q8_0` is a *signed* 8-bit block format, so
# `torch.uint8` would not be the same request and is left to the refusal below.
# There is no `torch.int8` tensor on this stack either way (docs/INT8.md §1),
# which is why this widens rather than honouring the spelling literally.
_WEIGHT_DTYPES = (torch.int8,)


_SHAPE_WAY_OUT = (
    "Either pick a format whose block size divides this width, or name these layers "
    "in TorchnativeConfig(modules_to_not_convert=[...]) to load them dense on purpose."
)


def _require_shim():
    if not hasattr(torch._C, "_quantize"):
        raise RuntimeError(_MISSING_SHIM.format(module=getattr(torch, "__file__", "?")))


def _probe_shape(fmt, in_features, _cache={}):
    """Ask the real gate whether `fmt` can hold a weight this wide.

    The block-size rule lives in `quant.rs` (`{fmt} stores {n} elements per
    block, so it must have their last dim divisible by block size`) and this
    does not restate it. It quantises a one-row tensor of the candidate width
    and lets the same refusal come back, so the check cannot drift from the
    thing it is checking -- and when a future format arrives with a block size
    nobody here has heard of, this keeps working with no edit.

    The wall is not exotic. SmolLM2-135M is 576 wide and `576 % 256 == 64`, so
    every 256-element k-quant (`q2_k` `q3_k` `q4_k` `q5_k` `q6_k`) is
    unavailable for it -- docs/QUANT2.md §5.2.

    `device="cpu"` is not decoration: this runs inside `from_pretrained`'s
    `torch.device("meta")` context, and a meta probe would have no storage to
    quantise.
    """
    key = (fmt, in_features)
    if key not in _cache:
        try:
            torch._C._quantize(torch.zeros(1, in_features, device="cpu"), fmt)
            _cache[key] = None
        except NotImplementedError as exc:
            _cache[key] = str(exc)
    return _cache[key]


class TorchnativeConfig(QuantizationConfigMixin):
    """`quantization_config=` for torchnative's block-quantised linear layers.

    Args:
        format: a GGML block format, in GGUF's spelling. `FORMATS()` lists what
            the loaded build accepts. Defaults to `"q8_0"`, which is the only
            format measured to leave SmolLM2-135M's greedy generation unchanged
            (docs/QUANT2.md §5.3 -- `q4_0` collapses at 29.5% relative RMS on
            the same model).
        modules_to_not_convert: layer names to leave dense, transformers'
            spelling of the choice `quantize_` leaves to its `predicate`.

            **`None` is not "convert everything".** It means transformers' own
            convention, `HfQuantizer.get_modules_to_not_convert` ->
            `get_keys_to_not_convert`, which excludes the output embedding, the
            last parameter, and every tied weight. On SmolLM2-135M that is
            `lm_head`, and the choice is not arbitrary in either direction:
            `lm_head` is 63% of the parameters and the layer whose error lands
            straight on the logits, and because the model ties it to the
            embedding, replacing it *breaks the tie and costs memory* -- 2.10x
            all-in against 2.38x with it left dense (docs/QUANT2.md §5.4).
            Pass `[]` to convert everything including it, or a list of names to
            choose exactly. The resolved list is recorded on the quantizer and
            reported by `model.torchnative_quantization`, so which behaviour
            ran is inspectable rather than assumed.

    Refuses an unknown format at construction rather than at load: a typo would
    otherwise surface a few gigabytes of download later, and `quant.rs` already
    refuses by name rather than picking a default.
    """

    def __init__(self, format="q8_0", modules_to_not_convert=None, **kwargs):
        self.quant_method = QUANT_METHOD
        self.format = format
        self.modules_to_not_convert = modules_to_not_convert
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.post_init()

    def post_init(self):
        if not isinstance(self.format, str):
            raise ValueError(f"format must be a string, got {type(self.format).__name__}")
        _require_shim()
        known = FORMATS()
        if self.format not in known:
            raise ValueError(
                f"unknown quantisation format {self.format!r}. This build has: "
                + ", ".join(known)
            )
        if self.modules_to_not_convert is not None:
            if isinstance(self.modules_to_not_convert, str):
                raise ValueError(
                    "modules_to_not_convert is a list of names, not a single string -- "
                    f"pass [{self.modules_to_not_convert!r}]."
                )
            self.modules_to_not_convert = list(self.modules_to_not_convert)


class _QuantizeOnLoad(ConversionOps):
    """Quantise one weight as it arrives, and hand nothing back to the loader.

    Returning `{}` is the load-bearing part. transformers' default is to route
    the op's result through `set_param_for_module`, which wraps a plain tensor
    in `torch.nn.Parameter` -- and a quantised tensor must not become one
    (`QuantizedLinear`'s docstring: `parameters()` feeds optimisers and
    `_apply`/`.to()` walks buffers calling dense ops, which
    `Repr::Quantized` refuses by name). So this assigns into the module itself
    and returns an empty mapping, discarding the key from `missing_keys` so the
    loader does not then report the layer as unfilled.
    """

    def __init__(self, quantizer):
        self.quantizer = quantizer

    def convert(self, input_dict, model=None, missing_keys=None, **kwargs):
        for param_name, value in input_dict.items():
            value = value[0] if isinstance(value, list) else value
            module, _ = get_module_from_name(model, param_name)
            module.adopt(value)
            if missing_keys is not None:
                missing_keys.discard(param_name)
            self.quantizer.adopted.add(param_name)
        return {}


class TorchnativeHfQuantizer(HfQuantizer):
    """Swaps `nn.Linear` for `QuantizedLinear` before the weights land.

    The three hooks that matter:

    | hook | when | what |
    |---|---|---|
    | `_process_model_before_weight_loading` | model is a meta skeleton | replace the leaves, refuse the combinations that cannot work |
    | `get_quantize_ops` / `param_needs_quantization` | per weight, as it is read | quantise it and drop the dense tensor |
    | `_process_model_after_weight_loading` | model is loaded | check every placeholder was filled, publish the report |

    `requires_calibration` stays `False` -- every scale comes from the block it
    belongs to, so there is no calibration pass to have skipped.
    """

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.modules_to_not_convert = []
        self.converted = []
        self.skipped = []
        self.adopted = set()
        # What the caller spelled in `dtype=`, kept only when it was an integer
        # dtype we widened (see `update_dtype`). `None` means the caller asked
        # for a float dtype and got it, so there is nothing to disclose.
        self.widened_from = None
        # Recorded rather than asserted, and published on the report. The
        # claim this plugin makes is that the swap happens *before* the
        # weights exist, and a machine too noisy to measure peak RSS can still
        # check that: every weight replaced below had no storage. A post-hoc
        # swap wearing this plugin's clothes cannot make this True.
        self.swapped_before_weights = True

    # -- environment ------------------------------------------------------

    def validate_environment(self, *args, **kwargs):
        _require_shim()
        if self.pre_quantized:
            raise ValueError(
                "this checkpoint declares quant_method='torchnative', but a "
                "torchnative-quantised model cannot be serialised in the first "
                "place: `QuantizedLinear.qweight` is deliberately absent from "
                "`state_dict()` (docs/QUANT2.md §4), and writing one out means "
                "writing GGUF, which this repository has no container for (§7). "
                "So there is no such checkpoint, and this config only applies to "
                "a dense one."
            )
        device_map = kwargs.get("device_map")
        if isinstance(device_map, dict):
            devices = {str(d) for d in device_map.values()}
            if devices - {"cpu"}:
                raise ValueError(
                    "torchnative quantisation is CPU-only -- candle's `QMatMul` is "
                    f"reached through the host build. Got device_map={device_map}."
                )

    def update_dtype(self, dtype):
        """Deal with `bfloat16` here rather than at the first forward.

        candle's `QMatMul::forward` takes `f32` and `f16` activations and
        nothing else (docs/QUANT2.md §7, wall 4). `bfloat16` is not an exotic
        request -- it is what SmolLM2-135M's own `config.json` asks for, so it
        is the default path and not a corner. Left alone it would load fine and
        fail inside the first matmul with candle's phrasing, which names
        neither the activation dtype nor this decision.

        So it is widened to `float32` with a warning, which is the same move
        `Bnb8BitHfQuantizer.update_dtype` makes for the same reason. **The
        warning is not decoration**: widening doubles the dense remainder (on a
        tied model that is the embedding, and it is the largest thing left),
        and a reader comparing memory against a `bfloat16` dense load has to
        know the two are not the same activation dtype. Anything that is
        neither `float32`, `float16` nor `bfloat16` is refused by name instead
        of quietly widened -- a `float64` request means the caller believes
        something about this model that is not true.
        """
        if isinstance(dtype, dict) or dtype is None:
            return dtype
        if dtype == torch.bfloat16:
            logger.warning_once(
                "torchnative quantisation cannot run bfloat16 activations (candle's "
                "QMatMul accepts float32 and float16 only, docs/QUANT2.md §7 wall 4), "
                "so the model is being loaded in float32 instead. Every tensor that is "
                "not replaced -- the embedding above all -- is therefore twice the size "
                "it would have been. Pass dtype=torch.float16 to avoid that."
            )
            return torch.float32
        if dtype in _WEIGHT_DTYPES:
            self.widened_from = dtype
            logger.warning_once(
                f"dtype={dtype} was read as a request about *weights*, not activations. "
                f"The weights are being quantised to "
                f"{self.quantization_config.format} by the quantization_config, which is "
                "the nearest thing this stack has to it -- there is no torch.int8 tensor "
                "here at all, because candle-core 0.11 has no I8 dtype (docs/INT8.md §1). "
                "Activations are loaded in float32, so model.dtype will report "
                "torch.float32 and not the dtype you passed; model.torchnative_quantization "
                "records what actually happened."
            )
            return torch.float32
        if dtype not in (torch.float32, torch.float16):
            raise ValueError(
                f"torchnative quantisation cannot run {dtype} activations: candle's "
                "QMatMul accepts float32 and float16 only (docs/QUANT2.md §7, wall 4). "
                "Pass dtype=torch.float32 to from_pretrained."
            )
        return dtype

    # -- before the weights ------------------------------------------------

    def _process_model_before_weight_loading(self, model, **kwargs):
        config = self.quantization_config
        fmt = config.format

        skip = HfQuantizer.get_modules_to_not_convert(model, config.modules_to_not_convert)
        self.modules_to_not_convert = sorted(skip)

        tied = set()
        for a, b in (getattr(model, "all_tied_weights_keys", None) or {}).items():
            tied.add(a.removesuffix(".weight"))
            tied.add(b.removesuffix(".weight"))

        refused = []
        for parent_name, parent in list(model.named_modules()):
            for child_name, child in list(parent.named_children()):
                if not isinstance(child, torch.nn.Linear):
                    continue
                full = f"{parent_name}.{child_name}" if parent_name else child_name
                if not should_convert_module(full, skip):
                    self.skipped.append((full, "modules_to_not_convert"))
                    continue
                if full in tied:
                    refused.append((full, _TIED.format(name=full), _TIED_WAY_OUT))
                    continue
                why = _probe_shape(fmt, child.in_features)
                if why is not None:
                    refused.append((full, why, _SHAPE_WAY_OUT))
                    continue
                if child.weight.device.type != "meta":
                    self.swapped_before_weights = False
                setattr(parent, child_name, QuantizedLinear.pending_from_linear(child, fmt))
                self.converted.append(full)

        if refused:
            raise ValueError(self._refusal(refused))
        if not self.converted:
            raise ValueError(
                f"quantization_config asked for {fmt} but nothing was converted in "
                f"{type(model).__name__}: {len(self.skipped)} nn.Linear excluded by "
                f"modules_to_not_convert={self.modules_to_not_convert}, and no other "
                "nn.Linear was found. A model that loads dense under a quantisation "
                "config is the failure that reads as success."
            )

    def _refusal(self, refused):
        """Refuse the load, rather than quietly leaving those layers dense.

        This is where the plugin has to differ from `quantize_`, and the reason
        is the channel and not the judgement. `quantize_` returns a `_Report`
        that groups its skips by reason, so a caller who reads it sees the wall
        by name. `from_pretrained` returns a model and nothing else, so the
        same "skip it and say so" would land in a log line beside a model that
        looks quantised and mostly is not -- for `q4_k` on a 576-wide model
        that is *every* layer. So the refusal is raised, and it names the way
        forward.
        """
        by_reason = {}
        for name, why, way_out in refused:
            by_reason.setdefault((why, way_out), []).append(name)
        lines = [
            f"format={self.quantization_config.format!r} cannot be applied to "
            f"{len(refused)} layer(s), so nothing was loaded:"
        ]
        for (why, way_out), names in sorted(by_reason.items()):
            lines.append(f"  {len(names)} layer(s), e.g. {names[0]}:")
            lines.append(f"    {why}")
            lines.append(f"    {way_out}")
        return "\n".join(lines)

    # -- during the load ---------------------------------------------------

    def param_needs_quantization(self, model, param_name, **kwargs):
        module, tensor_name = get_module_from_name(model, param_name)
        return tensor_name == "weight" and isinstance(module, QuantizedLinear)

    def get_quantize_ops(self):
        return _QuantizeOnLoad(self)

    # -- after the load ----------------------------------------------------

    def _process_model_after_weight_loading(self, model, **kwargs):
        unfilled = [
            name
            for name, module in model.named_modules()
            if isinstance(module, QuantizedLinear) and module.qweight is None
        ]
        if unfilled:
            raise RuntimeError(
                f"{len(unfilled)} layer(s) were replaced but never received a weight "
                f"(e.g. {unfilled[0]}). The checkpoint had no tensor for them, so the "
                "model would fail at its first forward instead of here."
            )
        model.torchnative_quantization = _LoadReport(
            format=self.quantization_config.format,
            converted=list(self.converted),
            skipped=list(self.skipped),
            modules_to_not_convert=list(self.modules_to_not_convert),
            default_skips=self.quantization_config.modules_to_not_convert is None,
            swapped_before_weights=self.swapped_before_weights,
            widened_from=self.widened_from,
        )
        return model

    # -- capabilities ------------------------------------------------------

    def is_serializable(self, safe_serialization=None):
        """No. `qweight` is not in `state_dict()`, by design -- see `validate_environment`."""
        return False

    @property
    def is_trainable(self):
        """No. There is no gradient for a packed block (`QuantizedLinear`'s docstring)."""
        return False


class _LoadReport:
    """What the plugin replaced, what it left alone, and on whose say-so.

    Published as `model.torchnative_quantization`. `quantize_` returns its
    `_Report` to the caller; `from_pretrained` returns only a model, so the
    same information has to be reachable from the model or it is not reachable
    at all -- and "how many layers were replaced" is exactly the number
    CLAUDE.md §5.3 warns about reading on its own.
    """

    def __init__(
        self,
        format,
        converted,
        skipped,
        modules_to_not_convert,
        default_skips,
        swapped_before_weights,
        widened_from=None,
    ):
        self.format = format
        self.converted = converted
        self.skipped = skipped
        self.modules_to_not_convert = modules_to_not_convert
        self.default_skips = default_skips
        self.swapped_before_weights = swapped_before_weights
        self.widened_from = widened_from

    def __str__(self):
        origin = (
            "transformers' default (get_keys_to_not_convert: output embedding, "
            "last parameter, tied weights)"
            if self.default_skips
            else "given by the caller"
        )
        lines = [
            f"format={self.format} converted={len(self.converted)} "
            f"left dense={len(self.skipped)}",
            f"  modules_to_not_convert={self.modules_to_not_convert} -- {origin}",
            "  every replaced leaf was swapped before its weight had storage"
            if self.swapped_before_weights
            else "  WARNING: at least one leaf was replaced after its weight existed",
        ]
        if self.widened_from is not None:
            lines.append(
                f"  caller asked for dtype={self.widened_from}; there is no such tensor on "
                f"this stack, so the weights are {self.format} and the activations are "
                "torch.float32 (model.dtype says float32)"
            )
        reasons = {}
        for name, why in self.skipped:
            reasons.setdefault(why, []).append(name)
        for why, names in sorted(reasons.items()):
            lines.append(f"  left dense ({len(names)}): {why}")
            lines.append(f"    e.g. {names[0]}")
        return "\n".join(lines)

    __repr__ = __str__


def _register():
    """Put the name in transformers' tables. Idempotent, and inert until used.

    `register_quantizer` raises on a duplicate name, which is right for a
    registry and wrong for a module that may be imported twice under different
    paths (this package is reached both as `torchnative.quant.hf` and, in the
    vendored tree, through a `PYTHONPATH` that can differ per process). The
    guard checks the table rather than a module-level flag for that reason.

    Registering changes no default. `AutoHfQuantizer.from_config` dispatches on
    `quantization_config.quant_method`, so nothing reaches this quantizer
    unless a `TorchnativeConfig` was passed; a load that never mentions
    torchnative behaves exactly as it did before this import.
    """
    from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING, AUTO_QUANTIZER_MAPPING

    if QUANT_METHOD not in AUTO_QUANTIZATION_CONFIG_MAPPING:
        register_quantization_config(QUANT_METHOD)(TorchnativeConfig)
    if QUANT_METHOD not in AUTO_QUANTIZER_MAPPING:
        register_quantizer(QUANT_METHOD)(TorchnativeHfQuantizer)


_register()
