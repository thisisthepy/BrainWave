"""Quantisation by module replacement.

    import torchnative.quant as q
    report = q.quantize_(model, format="q8_0")
    print(report)     # what was replaced, what was not, and why not

**Why replacement and not a dtype.** `torch.int8` tensors cannot be built on
this stack at all -- candle-core 0.11's `DType` has no `I8`, so the storage
does not exist (docs/graph/QUANT.md §2.1) -- and candle's quantisation is not a
`DType` either: `QTensor`/`GgmlDType` is a separate type system that
`Tensor`/`DType` never sees (§5.1). Teaching `aten.mm` a new element type
would therefore reach none of candle's fast kernels. Swapping the leaf reaches
them, and it is the same move `torchao` and `bitsandbytes` make, and the same
one Intel's (now archived) NPU library offered beside `torch.compile`:

    intel_npu_acceleration_library.compile(model, dtype=torch.int8)

The model stays real `transformers`; nothing is rewritten, subclassed or
patched. Instances are exchanged.

**The ceiling that comes with it.** A replaced leaf is all this covers. There
is no fusion, no view of the surrounding graph, and nothing inside a module
boundary is reachable -- an attention block handed over whole needs a graph,
which is docs/graph/DECOMP.md's path and not this one. The two are not exclusive.

**What this module does not do.** It does not quantise activations (the
activation stays `float32`, and candle quantises it per call inside `vec_dot`),
it does not calibrate, and it has no importance matrix. Every scale comes from
the block it belongs to and nothing else.
"""

import torch

__all__ = [
    "QuantizedLinear",
    "quantize_",
    "storage_bytes",
    "FORMATS",
    "TorchnativeConfig",
    "TorchnativeHfQuantizer",
]


def __getattr__(name):
    """`TorchnativeConfig` is fetched lazily, and that is what keeps this import cheap.

    `torchnative.quant.hf` imports `transformers` and, on import, registers
    the name `"torchnative"` in transformers' quantizer tables. Two rules make
    that an attribute lookup rather than a top-level import:

      * **`transformers` is not a hard dependency.** `import torchnative.quant`
        has to keep working on a machine that has never installed it, because
        `quantize_` does not need it. A top-level `from .hf import ...` would
        make the whole module fail to import on that machine.
      * **Registration must be something you ask for.** Registering a name is
        inert -- nothing dispatches to `"torchnative"` unless a
        `TorchnativeConfig` is handed to `from_pretrained` -- but it should
        still not happen behind the back of someone who imported this module
        for `quantize_` alone. `from torchnative.quant import TorchnativeConfig`
        is the ask; PEP 562 routes it here.
    """
    if name in ("TorchnativeConfig", "TorchnativeHfQuantizer"):
        from . import hf

        return getattr(hf, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def FORMATS():
    """The formats `_C` will accept, in GGUF's spelling."""
    return torch._C._quantized_formats()


class QuantizedLinear(torch.nn.Module):
    """`nn.Linear` with a block-quantised weight.

    Interface-compatible with what a transformer actually calls: `forward`,
    `in_features`, `out_features`, `bias`. Not compatible with what tooling
    calls -- see the two deliberate gaps below.

    **`weight` is not a `Parameter` and not a buffer.** It is a plain
    attribute, so it is absent from `parameters()`, `buffers()` and
    `state_dict()`. That is not an oversight to fix later:

      * `parameters()` feeds optimisers, and there is no gradient for a packed
        4-bit block -- appearing there would offer training that cannot happen.
      * `state_dict()` feeds `torch.save`, and `_apply`/`.to()` walks buffers
        calling dense ops on them, which a quantised tensor refuses by name
        (`Repr::Quantized` has no dense storage). Registering it would turn
        `model.to(...)` -- which `from_pretrained` calls -- into a failure.

    So a quantised model here is a runtime object, not a serialisable one.
    Writing one out means writing GGUF, which is `_C._quantized_blob()` plus a
    container this repository does not have (docs/graph/QUANT2.md §7).

    **The bias stays dense `float32`.** It is `out_features` numbers against a
    weight of `in_features * out_features`, so quantising it would buy a
    fraction of a percent and cost the one part of the layer where an absolute
    error is not attenuated by a dot product.
    """

    def __init__(self, in_features, out_features, qweight, bias, fmt):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.format = fmt
        # Assigned before `bias` on purpose: `nn.Module.__setattr__` routes a
        # `Parameter` into `_parameters` and anything else into `__dict__`, and
        # `qweight` must take the second path. It does, because `_C._quantize`
        # returns a plain `torch.Tensor` and never a `Parameter`.
        self.qweight = qweight
        self.bias = bias

    @classmethod
    def from_linear(cls, mod, fmt):
        """Build the replacement, or raise saying why the layer cannot take it.

        The raise is the point. A silent skip on a shape the format cannot hold
        would report a quantised model that is mostly dense -- and the shape in
        question is not exotic: SmolLM2-135M is 576 wide, which no 256-element
        k-quant can hold (docs/graph/QUANT2.md §5.2).
        """
        weight = mod.weight
        out_features, in_features = weight.shape
        qweight = torch._C._quantize(weight, fmt)
        return cls(in_features, out_features, qweight, mod.bias, fmt)

    @classmethod
    def pending_from_linear(cls, mod, fmt):
        """The same swap as `from_linear`, made *before* the weight exists.

        This is what lets the replacement happen ahead of the checkpoint
        instead of after it. `from_pretrained` builds the model on the meta
        device and only then streams weights in, so at replacement time
        `mod.weight` is a shape and a dtype and no storage. This keeps that
        meta `Parameter` registered under the name `weight` and leaves
        `qweight` as `None` until `adopt` is called with the real tensor.

        **The placeholder is not a design change to `QuantizedLinear`.** It
        exists for exactly as long as loading does. transformers matches
        checkpoint keys against `model.state_dict()`, so a module with no
        `weight` entry makes `...q_proj.weight` an *unexpected key* -- the
        tensor is then never read, never routed anywhere, and the layer is
        silently left without a weight. `adopt` removes the placeholder again,
        so a loaded model has the `state_dict` this class documents.
        """
        out_features, in_features = mod.weight.shape
        self = cls(in_features, out_features, None, mod.bias, fmt)
        # Reuse the very `Parameter` the loader will look at, rather than a
        # new meta tensor of the same shape: the shape check in
        # `set_param_for_module` reads `getattr(module, "weight").shape`, and
        # anything that made the two disagree would be a bug this class
        # invented for itself.
        self.register_parameter("weight", mod.weight)
        return self

    def adopt(self, weight):
        """Quantise `weight` into this layer and drop the loading placeholder.

        Called once per layer, by the conversion op that transformers runs on
        the tensor as it comes off disk. Quantising here rather than after the
        load is the whole point of the pre-load swap: the dense tensor is the
        op's only reference, so it is freed as soon as this returns and the
        dense model never exists all at once.
        """
        if self.qweight is not None:
            raise ValueError(
                f"{type(self).__name__} already holds a quantised weight; "
                "adopt() is for the loading placeholder and runs once."
            )
        self.qweight = torch._C._quantize(weight, self.format)
        del self._parameters["weight"]

    def forward(self, x):
        if self.qweight is None:
            raise RuntimeError(
                "QuantizedLinear was asked to run before its weight arrived. "
                "This instance is still the loading placeholder built by "
                "`pending_from_linear`; nothing called `adopt`."
            )
        return torch._C._quantized_linear(x, self.qweight, self.bias)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, format={self.format}"
        )


class _Report:
    """What `quantize_` did, and what it refused to do.

    A count of replacements on its own is the metric CLAUDE.md §5.3 warns
    about: it goes up whether or not anything was gained. So this also carries
    the bytes on both sides and every skip with its reason, and `__str__`
    prints all three.
    """

    def __init__(self, fmt):
        self.format = fmt
        self.replaced = []
        self.skipped = []
        self.dense_bytes = 0
        self.quantized_bytes = 0

    @property
    def ratio(self):
        if self.quantized_bytes == 0:
            return float("nan")
        return self.dense_bytes / self.quantized_bytes

    def __str__(self):
        lines = [
            f"format={self.format} replaced={len(self.replaced)} "
            f"skipped={len(self.skipped)}",
            # Say which bytes. This counts the weights that were *replaced*,
            # and nothing else in the model shrank -- SmolLM2's embedding is
            # 113 MB that stays dense, so a reader who takes this ratio for the
            # model's is off by nearly two (3.76x here against 2.10x all-in,
            # docs/graph/QUANT2.md §8). A true number that reads as a bigger claim
            # than it supports is the failure this repository keeps paying for.
            f"replaced weight bytes {self.dense_bytes} -> {self.quantized_bytes} "
            f"({self.ratio:.2f}x); unreplaced parameters are unchanged"
            if self.replaced else "weight bytes 0 -> 0",
        ]
        reasons = {}
        for name, why in self.skipped:
            reasons.setdefault(why, []).append(name)
        for why, names in sorted(reasons.items()):
            lines.append(f"  skipped ({len(names)}): {why}")
            lines.append(f"    e.g. {names[0]}")
        return "\n".join(lines)


def quantize_(model, format="q8_0", predicate=None):
    """Replace every `nn.Linear` in `model` with a `QuantizedLinear`, in place.

    `predicate(name, module) -> bool` narrows it; the default takes all of
    them. Returns a `_Report`; the model is modified in place and also
    returned by `.model`-free convention (the caller already has it).

    **`lm_head` is not special-cased.** It is the largest single weight in a
    small model (49152 x 576 in SmolLM2-135M, 63% of the parameters) and also
    the one whose error lands directly on the logits with no further layer to
    attenuate it. Both facts are real and pull opposite ways, so the choice is
    left to `predicate` rather than made here silently -- docs/graph/QUANT2.md §5.3
    has the measurement of what skipping it costs and buys.
    """
    report = _Report(format)
    targets = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, torch.nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            if predicate is not None and not predicate(full, child):
                report.skipped.append((full, "excluded by predicate"))
                continue
            targets.append((module, child_name, full, child))

    for parent, child_name, full, child in targets:
        try:
            replacement = QuantizedLinear.from_linear(child, format)
        except NotImplementedError as exc:
            report.skipped.append((full, _reason(exc)))
            continue
        setattr(parent, child_name, replacement)
        report.replaced.append(full)
        report.dense_bytes += child.weight.numel() * 4
        report.quantized_bytes += torch._C._quantized_nbytes(replacement.qweight)

    # **A no-op is an error, not an empty report.** This walks `named_children`
    # and rebinds with `setattr`, so it can only replace a module that has a
    # parent -- Python gives no way to rebind the caller's own reference. Hand
    # it a bare `nn.Linear`, or one layer picked out of a model
    # (`quantize_(model.lm_head)`, which is a natural thing to try), and it
    # walks nothing, replaces nothing, and returns a report whose `replaced`
    # and `skipped` are both empty and whose `ratio` is `nan`.
    #
    # That reads as success. The caller has to notice an empty list to find out
    # otherwise, and the shape of the mistake -- reaching for one layer rather
    # than the model -- is exactly the case where they are least likely to look.
    #
    # `skipped` being non-empty is a different thing and stays quiet: the
    # predicate excluded them, or `from_linear` refused them by name, and both
    # are answers rather than silence.
    if not report.replaced and not report.skipped:
        if isinstance(model, torch.nn.Linear):
            raise ValueError(
                "quantize_ cannot replace the module it was handed, only that "
                "module's children -- rebinding the caller's own reference is "
                "not something Python allows. Assign it instead: "
                "`parent.name = QuantizedLinear.from_linear(parent.name, "
                f'"{format}")`, or pass the model that contains this layer.'
            )
        raise ValueError(
            "quantize_ found no nn.Linear to replace in "
            f"{type(model).__name__}. Nothing was quantised."
        )
    return report


def _reason(exc):
    """The refusal, shortened to its distinguishing clause.

    Kept as the message rather than an error code because the messages are the
    discovery mechanism this shim is built around (DESIGN.md §6): grouping
    skips by their text is how a new wall shows up as a new line in the report
    instead of as a number that got smaller.
    """
    text = str(exc)
    marker = "must have their last dim divisible by block size"
    if marker in text:
        return "in_features is not a multiple of the format's block size"
    return text.split("\n")[0][:120]


def storage_bytes(model):
    """`(dense, quantised)` weight bytes over the whole model.

    Counts `QuantizedLinear.qweight` through `_C._quantized_nbytes` and
    everything else through `numel() * element_size()`. It has to be counted
    this way round: `element_size()` *refuses* on a quantised tensor, because
    answering from the dtype tag would report 4 bytes per element for a weight
    that stores 0.5625 (tensor.rs, `element_size`).

    **Tied weights are counted once**, by object identity, and that is not
    bookkeeping pedantry -- it changes the answer. SmolLM2-135M has
    `tie_word_embeddings=True`, so `lm_head.weight is embed_tokens.weight` and
    the dense model is 538 MB rather than the 651 MB a naive walk reports.
    Measured: without the dedupe this function claimed 226 MB of dense weight
    for a model holding 113 MB of it.

    That identity has a consequence worth knowing before quantising:
    **replacing a tied `lm_head` breaks the tie and costs memory.** The dense
    tensor stays alive inside the embedding, and the quantised head is an
    additional copy -- so on SmolLM2-135M, `format="q8_0"` over everything
    lands at 2.10x while the same format *skipping* `lm_head` lands at 2.38x
    (docs/graph/QUANT2.md §5.4). It is still worth doing for speed, because the
    dense `lm_head` is where the time is; it is a trade and not a free win.
    """
    dense = 0
    quantised = 0
    seen = set()
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            if id(module.qweight) not in seen:
                seen.add(id(module.qweight))
                quantised += torch._C._quantized_nbytes(module.qweight)
            tensors = [module.bias] if module.bias is not None else []
        else:
            tensors = list(module.parameters(recurse=False)) + list(
                module.buffers(recurse=False)
            )
        for t in tensors:
            if id(t) in seen:
                continue
            seen.add(id(t))
            dense += t.numel() * t.element_size()
    return dense, quantised
