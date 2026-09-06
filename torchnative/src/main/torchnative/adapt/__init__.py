"""Test-time learning methods.

TTL contains TTA contains TTT; the nesting is not flattened into sibling
modules. Each method declares its own differentiation requirement rather than
living in a directory named for one, because normalization calibration sits on
both sides of that line -- recomputing statistics needs no backward pass,
updating affine parameters by a loss does. See DESIGN.md §3.

    from torchnative import adapt

    model = adapt.wrap(model, method=adapt.Tent(), lr=1e-3)
    model.online()
    out = model(input_ids=ids)      # predicts, then adapts on what it predicted

    model.adapted.norm()            # how far the weights have moved
    model.revert()                  # the base weights are back, byte for byte

**A method is not the central type; the delta is.** DESIGN.md §3 says every
adaptation method reduces to a weight delta over base weights, methods differing
only in lifetime and destination. So :class:`Method` declares three things and
owns no state: *which* parameters it moves, *what* scalar it descends, and which
of §3's differentiation stages it needs. Everything about keeping, measuring,
reverting and shipping the result lives on ``torchnative.delta.Delta``, once,
for every method. :class:`Tent` below is 40 lines because of that.

**How the step is taken here, and why it is not ``loss.backward()``.**
``Tensor.backward()`` refuses on this stack and docs/BACKWARD.md §8 says why:
it would need a node per op and a flag that propagates, which is upstream's
``VariableType`` half. What exists instead is a tape over a *captured region* --
so an adaptation step records the forward, seeds the gradient at the objective,
walks the record backwards and hands the gradients to a real ``torch.optim``
optimiser. The consequence a caller can see is that **an adaptation step is
subject to every refusal capture makes** (docs/CAPTURE.md §4): no ``.item()``
inside the region, no in-place ops, no unseeded randomness. The forward still
runs when capture refuses -- capture is an observation -- but the *step* does
not, and says which op stopped it.
"""

from __future__ import annotations

import torch

from torchnative.delta import Delta

__all__ = ["Method", "Tent", "Adapted", "wrap"]


# DESIGN.md §3 axis 1. Stated as a module constant rather than a bare integer at
# each use, because "stage 2 is desktop-only, permanently" is a decision and not
# a magic number.
STAGE_FORWARD_ONLY = 0
STAGE_NARROW_BACKWARD = 1
STAGE_FULL_AUTOGRAD = 2


class Method:
    """What an adaptation method has to declare.

    Three things, and no state. A method that held state would be holding the
    delta, and then the second method would hold its own copy of the same
    lifetime logic -- which is exactly what DESIGN.md §3 arranges against.

    ``stage`` is DESIGN.md §3's first axis, declared per method rather than by
    directory, because normalisation calibration sits on both sides of the line:
    recomputing statistics needs no backward, updating the affine parameters by
    a loss does. A build without a backward can refuse a method at ``wrap``
    time by reading this, instead of at the first step by exploding.
    """

    #: One of the ``STAGE_*`` constants above.
    stage = None

    def select(self, model):
        """The names of the parameters this method adapts.

        Names, not tensors, because the delta is keyed on names -- an object it
        can outlive, unlike a ``Parameter`` identity.
        """
        raise NotImplementedError

    def objective(self, outputs):
        """The scalar to descend, computed from what ``model(...)`` returned.

        Evaluated *inside* the capture region, so every op it uses is recorded
        and every op it uses needs a derivative rule. ``_C._tape_rules()`` is
        the list; ``trace.differentiable()`` names what a given trace is missing.
        """
        raise NotImplementedError


def _is_normalisation(module):
    """Is this module a normalisation layer carrying affine parameters?

    Decided on the *class name*, which is a heuristic and is here on purpose:
    the library has to answer for classes it has never seen. ``nn.LayerNorm``,
    ``LlamaRMSNorm``, ``Gemma2RMSNorm``, ``T5LayerNorm`` and
    ``BatchNorm1d`` all contain "norm" and none of them shares a base class we
    could test for. What keeps the heuristic from over-reaching is the second
    half: the module must carry ``weight`` or ``bias`` as its *own* parameter,
    so a container whose class happens to be named for normalisation selects
    nothing.

    ``Tent(select=...)`` overrides it for a model where this is wrong, and
    :meth:`Adapted.online_parameters` prints what was chosen so that "wrong" is
    visible rather than inferred.
    """
    if "norm" not in type(module).__name__.lower():
        return False
    own = dict(module.named_parameters(recurse=False))
    return "weight" in own or "bias" in own


class Tent(Method):
    """Entropy minimisation on the normalisation affine parameters.

    Wang et al., *Tent: Fully Test-Time Adaptation by Entropy Minimization*
    (ICLR 2021): adapt to unlabelled test data by descending the entropy of the
    model's own predictions, moving only the affine parameters of the
    normalisation layers.

    It is DESIGN.md §3's stage 1 -- "optimisation-based, normalisation
    calibration, updating the affine {gamma, beta} by a loss" -- and that is
    the row of the survey table that needs a backward. The row above it, which
    only recomputes statistics, needs none; both are normalisation calibration,
    which is why the stage is declared here and not read off a directory name.

    **What is deliberately not done.** The paper also puts normalisation layers
    into batch-statistic mode, because its models are BatchNorm ones and the
    test-time statistics are half the method. This selects and updates affine
    parameters only. On a model whose normalisation has no running statistics --
    LayerNorm, RMSNorm, so every transformer -- the two halves coincide and
    nothing is missing. On a BatchNorm model they do not.

        Check: any(hasattr(m, "running_mean") and m.running_mean is not None
                   for m in model.modules())

    If that is True for your model, this class is implementing half of Tent and
    :meth:`select` will tell you which layers it picked.
    """

    stage = STAGE_NARROW_BACKWARD

    def __init__(self, select=None, affine_only=True):
        self._select = select
        self.affine_only = affine_only

    def select(self, model):
        if self._select is not None:
            return list(self._select(model) if callable(self._select) else self._select)
        names = []
        for mod_name, module in model.named_modules():
            if not _is_normalisation(module):
                continue
            for own_name, param in module.named_parameters(recurse=False):
                if self.affine_only and own_name not in ("weight", "bias"):
                    continue
                names.append(f"{mod_name}.{own_name}" if mod_name else own_name)
        return names

    def objective(self, outputs):
        """Mean prediction entropy, over every position of the batch.

        ``-(p * log p).sum(-1)`` with ``p`` from ``softmax`` and ``log p`` from
        ``log_softmax`` rather than from ``log(p)``: the second spelling is one
        op shorter and loses the large negative logits, which at a 49152-wide
        vocabulary is most of them.

        Both spellings have derivative rules, so the choice here is numerical
        and not a matter of what the tape can carry.
        """
        logits = _logits_of(outputs)
        p = torch.softmax(logits, dim=-1)
        logp = torch.log_softmax(logits, dim=-1)
        return -(p * logp).sum(-1).mean()


def _logits_of(outputs):
    """The tensor a method's objective is a function of.

    ``transformers`` returns a ``ModelOutput``; ``nn.Module`` returns a tensor.
    Both are handled and anything else is refused by name rather than guessed
    at -- a wrong guess here would descend the entropy of some other tensor and
    still report a step.
    """
    if isinstance(outputs, torch.Tensor):
        return outputs
    logits = getattr(outputs, "logits", None)
    if logits is not None:
        return logits
    raise TypeError(
        "torchnative.adapt: the model returned %r, which is neither a Tensor "
        "nor something with .logits, so there is no prediction to take an "
        "entropy of. Pass a method with its own objective()."
        % (type(outputs).__name__,)
    )


def _tensor_inputs(args, kwargs):
    """Every tensor the traced region is a function of, in a stable order.

    Capture burns in every tensor it was not handed (docs/CAPTURE.md §2), so a
    tensor argument that is *not* declared here becomes a constant -- and a
    constant is a gradient target. Missing one would therefore not fail loudly;
    it would put a gradient somewhere nobody asked for. Nested lists and tuples
    are walked for the same reason.
    """
    found = []

    def walk(v):
        if isinstance(v, torch.Tensor):
            found.append(v)
        elif isinstance(v, (list, tuple)):
            for e in v:
                walk(e)
        elif isinstance(v, dict):
            for e in v.values():
                walk(e)

    for a in args:
        walk(a)
    for _, v in sorted(kwargs.items()):
        walk(v)
    return found


class Adapted(torch.nn.Module):
    """A model with a method attached, and the delta that method is producing.

    ``offline`` is the default and is *exactly* the wrapped model: ``forward``
    calls it and returns, with nothing recorded and nothing updated. That is
    load-bearing rather than tidy -- docs/ADAPT.md §7 re-measures the prefill
    logits sha256 through this wrapper at every length docs/SEQLEN.md records,
    and an adaptation API that moves a plain forward is a bug.

    ``online()`` opens a :class:`~torchnative.delta.Delta` over the method's
    parameters and an optimiser over the same ones. From then on ``forward``
    predicts *and then* adapts, in that order, which is the online protocol the
    method is written for: the caller gets the prediction the model made before
    it saw its own entropy.
    """

    def __init__(self, model, method, optimizer=None, lr=1e-3, **optimizer_kwargs):
        super().__init__()
        if method.stage is None:
            raise ValueError(
                "torchnative.adapt: %r does not declare a stage. DESIGN.md §3 "
                "splits methods by differentiation requirement, and a build "
                "without a backward has to be able to refuse at wrap time "
                "rather than at the first step" % (type(method).__name__,)
            )
        if method.stage == STAGE_FORWARD_ONLY:
            raise NotImplementedError(
                "torchnative.adapt: %r declares stage 0 (forward only), and the "
                "step implemented here is stage 1 -- it captures a region and "
                "walks a tape. A stage-0 method updates statistics inside the "
                "forward and needs no step at all; nothing here provides that "
                "path yet." % (type(method).__name__,)
            )
        if method.stage == STAGE_FULL_AUTOGRAD:
            raise NotImplementedError(
                "torchnative.adapt: %r declares stage 2 (full autograd through "
                "an inner update). DESIGN.md §3 excludes stage 2 from device "
                "targets permanently, and this stack has no autograd outside a "
                "captured region.\n"
                "Check: torch.ones(1, requires_grad=True).sum().backward(). If "
                "that returns instead of refusing, an autograd exists that this "
                "refusal predates." % (type(method).__name__,)
            )
        self.model = model
        self.method = method
        self._lr = lr
        self._optimizer_cls = optimizer or torch.optim.SGD
        self._optimizer_kwargs = optimizer_kwargs
        self._optimizer = None
        self._delta = None
        self._online = False
        self._history = []
        self._steps = 0
        self._grad_hooks = []

    # -- state -------------------------------------------------------------

    def online(self):
        """Arm adaptation. Idempotent; the delta accumulates over one base.

        Calling it twice does not re-snapshot the base. A second snapshot would
        make the base whatever the first round of adaptation left behind, and
        then ``revert`` would restore an adapted model and report success.
        """
        if self._delta is None:
            names = self.method.select(self.model)
            if not names:
                raise ValueError(
                    "torchnative.adapt: %r selected no parameters of this model, "
                    "so every step would run and change nothing.\n"
                    "Check: [type(m).__name__ for m in model.modules()] -- "
                    "%s picks modules whose class name contains 'norm' and which "
                    "carry weight or bias directly. Pass select= if that is "
                    "wrong for this architecture."
                    % (type(self.method).__name__, type(self.method).__name__)
                )
            self._delta = Delta.over(self.model, names)
            params = dict(self.model.named_parameters())
            self._optimizer = self._optimizer_cls(
                [params[n] for n in names], lr=self._lr, **self._optimizer_kwargs
            )
        self._online = True
        return self

    def offline(self):
        """Disarm. The weights keep whatever the delta put there; use
        :meth:`revert` to undo it."""
        self._online = False
        return self

    @property
    def is_online(self):
        return self._online

    @property
    def adapted(self):
        """The :class:`~torchnative.delta.Delta` this wrapper is producing.

        Recorded up to the last step, so ``.norm()`` answers "how far has this
        model moved" without a second walk of the weights.
        """
        return self._delta

    @property
    def online_parameters(self):
        """The names being adapted -- what the method selected, not what it meant."""
        return () if self._delta is None else self._delta.covers

    @property
    def history(self):
        """The objective at every step taken, in order. The curve."""
        return list(self._history)

    # -- gradient hooks ----------------------------------------------------

    def add_grad_hook(self, hook):
        """Call ``hook(adapted, params, names)`` after backward, before the step.

        The one place a *federated* method can change the local objective
        without this class knowing what federation is.  ``FedProx``'s
        difference from ``FedAvg`` is entirely here -- the server step of the
        two is the same weighted mean, and what FedProx adds is
        ``mu * (w - w_global)`` on the gradient of every adapted parameter
        (Li et al. 2020, §3).  Without a hook there is nowhere to put it, and
        an aggregator that called itself FedProx while doing FedAvg's
        arithmetic would report success -- docs/DESIGN.md §6 puts that below a
        refusal, so :class:`torchnative.nn.federated.FedProx` refuses to
        aggregate at all unless it has installed one of these.

        ``params`` is ``{name: Parameter}`` for the whole model and ``names``
        the parameters this step has gradients for; a hook writes into
        ``params[name].grad`` in place.  It runs *after* the gradients are
        assigned and *before* ``optimizer.step()``, which is the only point at
        which "the gradient this step will apply" exists as an object.

        Returns a callable that removes the hook again, so that installing one
        is reversible without reaching into the list.
        """
        if not callable(hook):
            raise TypeError(
                "torchnative.adapt.Adapted.add_grad_hook: %r is not callable"
                % (hook,)
            )
        self._grad_hooks.append(hook)

        def remove():
            if hook in self._grad_hooks:
                self._grad_hooks.remove(hook)

        return remove

    @property
    def grad_hooks(self):
        """The installed gradient hooks, in call order."""
        return tuple(self._grad_hooks)

    def revert(self):
        """Put the base weights back byte for byte, keeping the delta."""
        if self._delta is not None:
            self._delta.revert(self.model)
        return self

    # -- the step ----------------------------------------------------------

    def forward(self, *args, **kwargs):
        if not self._online:
            return self.model(*args, **kwargs)
        outputs, _ = self.step(*args, **kwargs)
        return outputs

    def step(self, *args, **kwargs):
        """One adaptation step. Returns ``(outputs, objective)``.

        The order is predict-then-adapt: ``outputs`` is what the model computed
        *before* the update, which is the prediction an online serving loop has
        already had to emit.
        """
        if self._delta is None:
            raise RuntimeError(
                "torchnative.adapt: step() before online() -- there is no delta "
                "to write into and no optimiser to write it with"
            )

        inputs = _tensor_inputs(args, kwargs)
        _C = torch._C
        _C._capture_begin(inputs)
        try:
            outputs = self.model(*args, **kwargs)
            objective = self.method.objective(outputs)
            trace = _C._capture_end(objective)
        except BaseException:
            # Capture is an observation and the forward has already happened;
            # what must not survive is the *recording*, which would otherwise
            # still be open when the next call begins.
            if _C._capture_active():
                _C._capture_abandon()
            raise

        slots = self._slots(trace)
        wrt = sorted(slots.values())
        if self._steps == 0:
            # Once, on the first step. `differentiable()` answers "what stops
            # this model" without running a backward and reading an exception
            # (docs/BACKWARD.md §1.2), and the first step is where a model that
            # cannot be adapted at all should say so -- with the whole list,
            # rather than with whichever missing rule the walk happened to reach
            # first. Skipped afterwards because the trace shape does not change
            # and the walk is not free.
            report = trace.differentiable(wrt_constants=wrt)
            if report["missing"]:
                raise NotImplementedError(
                    "torchnative.adapt: this model cannot take a stage-1 step -- "
                    "%d op(s) on the gradient path have no derivative rule: %s.\n"
                    "Check: torch._C._tape_rules() is the list that does exist, "
                    "and trace.differentiable() produced this one."
                    % (len(report["missing"]), sorted(report["missing"]))
                )
        grads = trace.backward(inputs, wrt_constants=wrt)["constants"]

        params = dict(self.model.named_parameters())
        got = 0
        for name, slot in slots.items():
            g = grads[slot]
            if g is None:
                continue
            params[name].grad = g
            got += 1
        if got == 0:
            raise RuntimeError(
                "torchnative.adapt: the objective produced a gradient for none "
                "of the %d selected parameters, so a step would run and change "
                "nothing. The usual cause is an objective computed on a "
                "detached tensor: the tape's rule for aten.detach.default is to "
                "stop, which is what detach is for.\n"
                "Check: trace.differentiable(wrt_constants=[...]) -- "
                "'nodes_on_a_gradient_path' is 0 when nothing connects."
                % (len(slots),)
            )

        # After the gradients are on the parameters and before they are
        # applied: the only moment at which "what this step will apply" is an
        # object. `FedProx` puts its proximal term here -- see add_grad_hook.
        for hook in self._grad_hooks:
            hook(self, params, sorted(slots))

        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self._delta.record(self.model)
        self._steps += 1
        value = float(objective.item())
        self._history.append(value)
        return outputs, value

    def _slots(self, trace):
        """Map each selected parameter to its slot among the trace's constants.

        By object identity, which is what ties "the gradient of the objective
        with respect to this parameter" to "the gradient at this constant"
        (docs/BACKWARD.md §1.2). A selected parameter that is *not* a constant
        of this trace did not participate in this forward, and that is refused
        rather than skipped -- silently adapting a subset of what was asked for
        is the failure this class is arranged against.
        """
        by_id = {id(o): i for i, o in enumerate(trace.constant_values)}
        params = dict(self.model.named_parameters())
        slots = {}
        absent = []
        for name in self._delta.covers:
            slot = by_id.get(id(params[name]))
            if slot is None:
                absent.append(name)
            else:
                slots[name] = slot
        if absent:
            raise RuntimeError(
                "torchnative.adapt: %d of %d selected parameters are not "
                "constants of this trace, so this forward did not use them; "
                "first is %r. Adapting the rest would report a step over "
                "parameters that were never involved."
                % (len(absent), len(self._delta.covers), absent[0])
            )
        return slots

    def extra_repr(self):
        return "method=%s, %s, steps=%d, covers=%d" % (
            type(self.method).__name__,
            "online" if self._online else "offline",
            self._steps,
            len(self._delta) if self._delta is not None else 0,
        )


def wrap(model, method=None, optimizer=None, lr=1e-3, **optimizer_kwargs):
    """Attach an adaptation method to a model. See :class:`Adapted`.

    Returns a wrapper rather than mutating the model, so that the model the
    caller already had is still the unadapted one and ``offline`` is provably
    the identity.
    """
    if method is None:
        raise TypeError(
            "torchnative.adapt.wrap: method= is required. There is no default "
            "adaptation method, because the choice determines which parameters "
            "move and what is minimised, and neither has a safe guess"
        )
    return Adapted(model, method, optimizer=optimizer, lr=lr, **optimizer_kwargs)
