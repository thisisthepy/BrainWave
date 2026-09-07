# `nonzero`

## `where.default` and `nonzero.default`
Upstream maps the single-argument `torch.where(condition)` to `aten.where.default` and `torch.nonzero` to `aten.nonzero.default`.
While they are conceptually similar, they return different shapes:
- `where.default` returns a tuple of 1-D tensors (one for each dimension).
- `nonzero.default` returns a single 2-D tensor of shape `(z, ndim)`.
- For a 0-D input tensor (e.g. `torch.tensor(5)`), `nonzero.default` returns a 2-D tensor of shape `(1, 0)` (if non-zero) or `(0, 0)` (if zero). `where.default` returns a tuple of one 1-D tensor of size `[1]` (if non-zero) or `[0]`.

Because of these shape differences and the 0-D corner cases, `nonzero.default` cannot be trivially implemented as a spelling over `where.default`. A separate kernel is needed to emit the `(z, ndim)` flat tensor efficiently. We implemented `nonzero_default` in `rust/torch_c/src/aten.rs` and bound it to `aten.nonzero.default`.

We also intercepted `nonzero` in `rust/torch_c/src/bootstrap.py` for both the `torch._C._VariableFunctions` namespace and `TensorBase` to handle the Python-only `as_tuple` argument. If `as_tuple=True`, it dispatches to `aten.where.default`, matching upstream's behavior. If `as_tuple=False`, it dispatches to `aten.nonzero.default`.

## Capture Refusal
Both `aten.nonzero.default` and `aten.where.default` produce an output whose shape depends on the tensor values (specifically, the number of non-zero elements). A trace whose node output shape is not a function of its inputs cannot be recorded, because a replay with different inputs might produce a different shape, invalidating the rest of the graph.
We explicitly added both ops to `DATA_DEPENDENT_SHAPE` in `rust/torch_c/src/capture.rs` so that graph capture refuses them by name rather than recording an operation whose replay would be unsound.

## Meta Path
`aten.nonzero.default` and `aten.where.default` bypass standard kernels for meta tensors and are routed to `meta_dispatch`. Upstream raises a `RuntimeError` by default because the shape is data-dependent, unless `torch.fx.experimental._config.meta_nonzero_assume_all_nonzero` is True.
We implemented this exact behavior in `meta_dispatch()`: if the config is not set, it throws upstream's exact exception. If it is set, it assumes all elements are non-zero and returns fake shapes (`(numel, ndim)` for `nonzero`, and `ndim` 1-D tensors of size `numel` for `where`). This is required because `from_pretrained` initializes on the meta device.

## Models unblocked
1. `switch_transformers`: Hit `TensorBase.nonzero` inside MoE top-1 expert routing (`expert_hit = torch.greater(..., 0).nonzero()`). With `nonzero` implemented, it successfully moved past this and hit the next wall: `NotImplementedError: not implemented in torch._C shim: TensorBase.index_add_`.
2. `whisper.generate()`: Hit `torch.where` inside the autoregressive decode loop. With the fix to `__getitem__` (which previously mishandled `None` / `np.newaxis` when paired with tensor indices, breaking the downstream `aten.index.Tensor` dimensions), `whisper.generate()` completed entirely without hitting any further walls!

## Gates
```
329 marker(s) checked: PASS=329

DOCWATCH: PASS -- 329/329 evaluated marker(s) hold
RUN EXIT=0
```

```
SUMMARY: 8476/8476 cases passed, 0 failed, ops covered=203, pending case builders=0
```

`pytests/run.sh` passed with 381 `ok`.
