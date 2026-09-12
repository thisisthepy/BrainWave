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

**Both stages of DESIGN.md §3 that target a device are here.** :class:`Tent` is
stage 1 -- descend a loss on the affine parameters, which needs a narrow
backward -- and :class:`BatchNormStats` is stage 0, which recomputes the running
statistics and needs no backward at all. The two open different state: a stage-1
wrapper opens a ``delta.Delta`` over parameters, a stage-0 wrapper opens a
``delta.BufferSnapshot`` over buffers, and ``revert`` puts back whichever is
open. That a delta covers parameters and *not* buffers is a decision with an
argument behind it, written out in DESIGN.md §3 -- a running statistic is a
re-estimate rather than an additive offset, so ``apply`` and ``publish`` are not
defined on it.

**A method is not the central type; the delta is.** DESIGN.md §3 says every
adaptation method reduces to a weight delta over base weights, methods differing
only in lifetime and destination. So :class:`Method` declares three things and
owns no state: *which* parameters it moves, *what* scalar it descends, and which
of §3's differentiation stages it needs. Everything about keeping, measuring,
reverting and shipping the result lives on ``torchnative.delta.Delta``, once,
for every method. :class:`Tent` below is 40 lines because of that.

**How the step is taken here, and why it is not ``loss.backward()``.**
``Tensor.backward()`` refuses on this stack and docs/training/BACKWARD.md §8 says why:
it would need a node per op and a flag that propagates, which is upstream's
``VariableType`` half. What exists instead is a tape over a *captured region* --
so an adaptation step records the forward, seeds the gradient at the objective,
walks the record backwards and hands the gradients to a real ``torch.optim``
optimiser. The consequence a caller can see is that **an adaptation step is
subject to every refusal capture makes** (docs/graph/CAPTURE.md §4): no ``.item()``
inside the region, no in-place ops, no unseeded randomness. The forward still
runs when capture refuses -- capture is an observation -- but the *step* does
not, and says which op stopped it.
"""

from __future__ import annotations

import torch

from torchnative.delta import BufferSnapshot, Delta

__all__ = [
    "Method", "StatisticsMethod", "Tent", "BatchNormStats", "Adapted", "wrap",
]


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


def _has_running_statistics(module):
    """Does this module keep running statistics of its own?

    By the buffers it carries, not by its class name. That is the opposite of
    :func:`_is_normalisation`'s decision and the difference is deliberate: the
    affine parameters of a normalisation layer are only *recognisable* by name,
    because ``nn.LayerNorm``, ``LlamaRMSNorm`` and ``BatchNorm1d`` share no base
    class -- but running statistics are not a naming question at all. A module
    either registered a ``running_mean`` and a ``running_var`` or it did not,
    and that is the whole of what stage 0 needs to know.

    So this reaches ``InstanceNorm``, ``SyncBatchNorm`` and a third-party layer
    that registered the same two buffers, and does not reach a
    ``BatchNorm(track_running_stats=False)`` -- which registers both names as
    ``None`` and computes batch statistics on every forward, so there is
    nothing for stage 0 to recalibrate and no state for it to revert.
    """
    own = dict(module.named_buffers(recurse=False))
    return own.get("running_mean") is not None and own.get("running_var") is not None


class StatisticsMethod(Method):
    """A stage-0 method: recompute the normalisation statistics, no backward.

    DESIGN.md §3's survey table puts normalisation calibration on both sides of
    the differentiation line. This is the row above :class:`Tent` -- "테스트
    배치에서 통계(mu, sigma)만 다시 계산" -- and it is the cheap half of the two
    on a device, because it needs no capture, no tape, no optimiser and no
    gradient at all.

    **It declares a different thing from :class:`Method`, and says so with a
    different name.** ``Method.select`` returns *parameter* names, because a
    stage-1 delta is keyed on them. A stage-0 method moves no parameters; what
    it names is the **modules** whose statistics it recalibrates, so it declares
    :meth:`select_modules` and leaves ``select``/``objective`` alone. Reusing
    ``select`` for both would put two meanings behind one name, which is how
    this project's worst defects have started.

    :class:`Adapted` reads this at ``wrap`` time: a method declaring stage 0
    without a ``select_modules`` is refused there, before a forward, the same
    way a method declaring no stage at all is.
    """

    stage = STAGE_FORWARD_ONLY

    def select_modules(self, model):
        """The names of the modules whose running statistics this recalibrates.

        Names, not modules, for the reason :meth:`Method.select` gives: a name
        outlives an object identity, and the snapshot the wrapper opens is
        keyed on the buffer names those modules own.
        """
        raise NotImplementedError


class BatchNormStats(StatisticsMethod):
    """Recalibrate the running statistics of every normalisation layer that has them.

    DESIGN.md §3's stage 0. This is the *other* half of what Wang et al.'s Tent
    does on a BatchNorm model: the paper both puts the normalisation layers into
    batch-statistic mode and descends the entropy of the predictions on the
    affine parameters. :class:`Tent` here does the second; this does the first,
    and does it with no backward at all.

        model = adapt.wrap(model, method=adapt.BatchNormStats())
        model.online()
        out = model(x)             # predicts, then recalibrates on what it saw

        model.adapted.drift(model.model)   # how far the statistics moved
        model.revert()                     # the base statistics are back

    **Selection is by buffer, not by name** -- see :func:`_has_running_statistics`.
    On a model whose normalisation keeps no running statistics (LayerNorm,
    RMSNorm, so every transformer) this selects nothing, and ``online()``
    refuses rather than running a method that would change nothing and report
    every step as having happened. That is not a shortcoming of this class: a
    layer that computes its statistics from the batch on every forward is
    *already* calibrated to the test distribution, so there is nothing stage 0
    could add.

    **It is source-free and label-free**, which is what makes it TTA under the
    survey's definition (DESIGN.md §3): nothing but the test batches is read,
    and no target is required.
    """

    def __init__(self, select=None):
        self._select = select

    def select_modules(self, model):
        if self._select is not None:
            return list(self._select(model) if callable(self._select) else self._select)
        return [
            name for name, module in model.named_modules()
            if _has_running_statistics(module)
        ]


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

    Capture burns in every tensor it was not handed (docs/graph/CAPTURE.md §2), so a
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
    load-bearing rather than tidy -- docs/models/ADAPT.md §7 re-measures the prefill
    logits sha256 through this wrapper at every length docs/numerics/SEQLEN.md records,
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
        if method.stage == STAGE_FORWARD_ONLY and not hasattr(method, "select_modules"):
            raise NotImplementedError(
                "torchnative.adapt: %r declares stage 0 (forward only) but does "
                "not declare select_modules(model). A stage-0 method does not "
                "move parameters -- it recalibrates the running statistics of "
                "named modules -- so Method.select, which returns parameter "
                "names, is not its contract.\n"
                "Check: subclass torchnative.adapt.StatisticsMethod (or use "
                "BatchNormStats), which declares select_modules."
                % (type(method).__name__,)
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
        # Stage 0's state. A second attribute rather than a second meaning for
        # `_delta`: the two are different types answering different questions
        # (DESIGN.md §3), and exactly one of them is ever open.
        self._snapshot = None
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
        if self.stage == STAGE_FORWARD_ONLY:
            if self._snapshot is None:
                self._snapshot = BufferSnapshot.over(
                    self.model, self._statistic_buffers()
                )
            self._online = True
            return self
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
        return self._delta if self._snapshot is None else self._snapshot

    @property
    def stage(self):
        """Which of DESIGN.md §3's differentiation stages this wrapper is running."""
        return self.method.stage

    @property
    def online_parameters(self):
        """The names being adapted -- what the method selected, not what it meant.

        Empty for a stage-0 wrapper, and that is the truth rather than a hole:
        stage 0 moves no parameters at all. :attr:`online_buffers` is where its
        selection shows.
        """
        return () if self._delta is None else self._delta.covers

    @property
    def online_buffers(self):
        """The buffer names a stage-0 wrapper is recalibrating.

        Empty for a stage-1 wrapper, symmetrically: `Tent` moves parameters and
        touches no buffer. Kept as a second property rather than as a
        stage-dependent meaning for :attr:`online_parameters`, because a caller
        that printed "adapting: [...]" would otherwise be printing two different
        kinds of name under one heading.
        """
        return () if self._snapshot is None else self._snapshot.covers

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
        arithmetic would report success -- docs/design/DESIGN.md §6 puts that below a
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
        """Put the base state back byte for byte, keeping it.

        **Whichever state is open**, which for a stage-0 wrapper is the buffer
        snapshot and not a delta. That distinction is the reason this method is
        not one line: a stage-0 run mutates ``running_mean``, ``running_var``
        and ``num_batches_tracked``, and a revert that only knew about
        ``named_parameters()`` would return, report success, and leave the model
        carrying the test distribution's statistics. Held by
        `rust/torch_c/pytests/test_stage0.py`.
        """
        if self._delta is not None:
            self._delta.revert(self.model)
        if self._snapshot is not None:
            self._snapshot.revert(self.model)
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
        if self.stage == STAGE_FORWARD_ONLY:
            return self._forward_only_step(*args, **kwargs)
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
            # (docs/training/BACKWARD.md §1.2), and the first step is where a model that
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

    # -- stage 0 -----------------------------------------------------------

    def _statistic_modules(self):
        """The modules the stage-0 method named, resolved and checked.

        Resolved here rather than in the method, so that a name the model does
        not have is refused by the wrapper that is about to put it into training
        mode -- a stage-0 step over a mis-named module would otherwise recompute
        nothing and still report a step.
        """
        by_name = dict(self.model.named_modules())
        names = list(self.method.select_modules(self.model))
        absent = [n for n in names if n not in by_name]
        if absent:
            raise KeyError(
                "torchnative.adapt: %d of %d names %s selected are not modules "
                "of this model, first is %r"
                % (len(absent), len(names), type(self.method).__name__, absent[0])
            )
        return [(n, by_name[n]) for n in names]

    def _statistic_buffers(self):
        """The buffer names a stage-0 run will move, and the refusal if there are none.

        ``running_mean``/``running_var``/``num_batches_tracked`` where the module
        registered them, by walking the module's own buffers rather than by
        assuming the three names -- a layer that keeps a fourth statistic would
        otherwise have it moved by the run and left behind by the revert.
        """
        selected = self._statistic_modules()
        names = []
        for mod_name, module in selected:
            for own_name, buf in module.named_buffers(recurse=False):
                if buf is None:
                    continue
                names.append(f"{mod_name}.{own_name}" if mod_name else own_name)
        if not names:
            raise ValueError(
                "torchnative.adapt: %r selected %d module(s) of this model and "
                "none of them keeps running statistics, so every step would run "
                "and recalibrate nothing.\n"
                "Check: [n for n, m in model.named_modules() "
                "if getattr(m, 'running_mean', None) is not None] -- LayerNorm "
                "and RMSNorm keep none, so on a transformer there is nothing "
                "for stage 0 to do and Tent (stage 1) is the whole method."
                % (type(self.method).__name__, len(selected))
            )
        return names

    def _forward_only_step(self, *args, **kwargs):
        """DESIGN.md §3's stage 0. Returns ``(outputs, None)``.

        No capture, no tape, no optimiser and no gradient: the whole method is
        that the selected modules see the batch in training mode, so their
        running statistics take it in. So none of stage 1's refusals apply here
        -- a model that capture turns away (``.item()``, in-place ops, unseeded
        randomness -- docs/graph/CAPTURE.md §4) can still be adapted at stage 0.

        **Predict first, then adapt**, the same order and for the same reason as
        the stage-1 step: the caller gets the prediction the model made with the
        statistics it had going in, which is the one an online serving loop has
        already had to emit. So the forward runs twice -- once in whatever mode
        the model is in, for the answer, and once with the selected modules in
        training mode, for the update. A single training-mode forward would be
        one forward cheaper and would return a prediction normalised by the
        batch's own statistics, which is a different method.

        **The second element is ``None``, not ``0.0``.** A stage-0 method
        descends nothing, so there is no objective; a zero would go into
        :attr:`history` and draw a flat curve that looked like a method that had
        converged. :attr:`history` stays empty for the same reason, and what
        this run *does* have a number for is ``adapted.drift(model)``.
        """
        if self._snapshot is None:
            raise RuntimeError(
                "torchnative.adapt: step() before online() -- there is no "
                "snapshot of the statistics this would move, so revert() would "
                "have nothing to put back"
            )
        outputs = self.model(*args, **kwargs)

        modules = [m for _, m in self._statistic_modules()]
        was = [m.training for m in modules]
        for m in modules:
            m.train()
        try:
            # No gradient: stage 0 has no objective, so a graph built here
            # would be built and dropped once per batch for nothing.
            with torch.no_grad():
                self.model(*args, **kwargs)
        finally:
            # Restored even if the forward raised. Leaving a module in training
            # mode would turn its statistic update on for every later call --
            # including the plain forwards of a wrapper that had gone back
            # offline, which docs/models/ADAPT.md §7 asserts are the model's own.
            for m, mode in zip(modules, was):
                m.train(mode) if mode else m.eval()
        self._steps += 1
        return outputs, None

    def _slots(self, trace):
        """Map each selected parameter to its slot among the trace's constants.

        By object identity, which is what ties "the gradient of the objective
        with respect to this parameter" to "the gradient at this constant"
        (docs/training/BACKWARD.md §1.2). A selected parameter that is *not* a constant
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
            len(self.adapted) if self.adapted is not None else 0,
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
