# SAVE2 — the three items docs/SAVE.md §6 left open

docs/SAVE.md §6 already lists three features as deliberate refusals: the legacy
(non-zipfile) container, `torch.serialization.skip_data`, and
`write_record(compress=True)`. This round's job was to check that verdict
against evidence rather than take it on faith, and to confirm the existing
save path holds up on a real 135M-parameter checkpoint.

## 1. Evidence for each of the three

### 1.1 Legacy (non-zipfile) container

`grep -rn "_use_new_zipfile_serialization" torch/serialization.py` shows the
flag exists and is read at save time (`serialization.py:1001`), but
`grep -rn "_use_new_zipfile_serialization\|use_new_zipfile" transformers/*.py`
in the vendored/installed `transformers` tree returns **zero matches** —
nothing in `transformers` ever asks for the legacy container.

The real checkpoint this round was told to use, `HuggingFaceTB/SmolLM2-135M`,
ships **only** `model.safetensors` on the Hub (checked the cached snapshot
directory directly: one `.safetensors` file, no `.bin`). So even the "`.bin`
checkpoints in the wild" case this task description raised does not apply to
the checkpoint actually in scope here — there is no `.bin` file, legacy or
otherwise, for this model.

`transformers/modeling_utils.py:load_state_dict` does have a `.bin` path
(`WEIGHTS_NAME`), for older repos that predate safetensors. But PyTorch's
zipfile format has been the *default* writer since 1.6 (2020); a `.bin` found
on the Hub today only omits safetensors, not the zip container. I did not
find a code path, in `torch` or `transformers`, that still asks
`torch.save` to write the legacy layout.

**Verdict: not implemented.** The existing refusal in
`_ZipRecords`/`storage.rs` (the shim only knows how to read the record it can
also write, per docs/SAVE.md §6 and docs/CKPT.md §3.3) stands. This is not
new work — the previous round already reasoned through the same asymmetry
(`set_` copies-then-fills, legacy fills-then-`set_`s) and reached the same
conclusion; this round just re-derived it from fresh grep evidence instead of
trusting the earlier doc.

### 1.2 `skip_data`

`grep -n skip_data torch/serialization.py` shows it guarded by
`torch.serialization.skip_data`'s own docstring: *"an early prototype and is
subject to change."* `grep -rl skip_data` over the installed `transformers`
tree: **zero files.** Nothing outside torch's own test suite calls it.

**Verdict: not implemented.** `write_record_metadata` still raises
`NotImplementedError` (`bootstrap.py:3362`) rather than writing a
zero-filled payload — the exact case docs/CKPT.md §4 spent a section on (a
file that loads, matches every key, and is silently all-zero).

### 1.3 `write_record(compress=True)`

Every `zip_file.write_record(...)` call inside `torch/serialization.py`
(lines 1259-1312, checked by grep) passes no `compress` argument, i.e.
`compress=False`. Nothing in `torch` itself ever asks for a compressed
record, and `transformers`'s save path goes through `torch.save`/
`safetensors`, neither of which threads a `compress` flag down to
`write_record`.

**Verdict: not implemented.** `write_record` still raises
`NotImplementedError` for `compress=True` (`bootstrap.py:3295-3300`).

### 1.4 Summary

All three are **API surface nothing calls**, for the checkpoint and library
versions in scope here. No code changes were made to `bootstrap.py`,
`storage.rs`, or `tensor.rs` for this round — the refusals already in place
from the previous round are the correct shape, confirmed with a second,
independent evidence pass (grep over the installed `torch`/`transformers`
trees plus inspection of the actual Hub artefact) rather than repeated from
the earlier doc.

## 2. What was proven instead: the real-checkpoint round trip

The task's other requirement — that the *existing*, already-implemented save
path survives a real 135M-parameter model, not just synthetic tensors —
had not been measured. Ran three checks, foreground, single interpreter
each, `HF_HOME=/Volumes/macMini/caches/hf-home` (already-cached weights, no
network needed beyond a metadata HEAD).

### 2.1 shim writes -> shim reads (in-model forward pass)

`AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")` under the
shim, `torch.save(model.state_dict(), ckpt)`, a fresh
`from_pretrained` instance, `load_state_dict(torch.load(ckpt,
weights_only=True))`, forward pass on both instances with identical input
ids (`"The capital of France is"`).

```
checkpoint size: 269121572 bytes
max abs diff (logits, shape (1, 5, 49152)): 0.0
```

Bit-exact, not approximate.

### 2.2 shim writes -> upstream torch reads (cross-process)

Wrote `state_dict()` with the shim in one process
(`PYTHONPATH=.../torchnative/src/main TORCH_USE_RTLD_GLOBAL=1`), then loaded
the resulting file with plain upstream torch in a separate process
(`env -u PYTHONPATH -u TORCH_USE_RTLD_GLOBAL`):

```
=== write with shim ===
shim
wrote 269123426
=== read with upstream ===
upstream
keys: 273
sample sum: -14720.0
```

Upstream opens the archive, all 273 keys present, no exception. (Byte count
differs by ~2 KB from §2.1's file because this run wrote the un-reloaded
`from_pretrained` state dict directly rather than round-tripping twice; not a
discrepancy in either writer.)

### 2.3 upstream writes -> shim reads

This direction is already covered by the existing suite
(`test_ckpt_torch_load_zip_round_trip_matches_upstream_within_measured_tolerance`
and neighbours in `pytests/test_shim.py`, `_ckpt_fixture`), which builds a
checkpoint with upstream torch and reads it back with the shim in a
subprocess. Not re-derived here since it already runs on every suite pass;
confirmed it still passes as part of the full-suite re-run below.

## 3. Regression

Rebuilt `lib_C.dylib` from this worktree, reinstalled via
`vendor/install_shim.sh`, cleared `build/test-results` first, and reran the
full suite plus golden compare. No source files were touched by this round
(`git status --short` in this worktree shows only this doc as new) — the
rebuild exists to prove the binary the suite runs against is the one in this
tree, not a stale artefact from before this session started.

```
pytests/run.sh                 389 ok
tools/docwatch/check_docs.py   DOCWATCH: PASS -- 350/350 evaluated marker(s) hold
tools/golden/compare.py        SUMMARY: 8509/8509 cases passed, 0 failed, ops covered=203
```

All three meet or exceed the round's gates (389 ok / 350+ DOCWATCH / 8509/8509,
ops=203).

## 4. What this round did NOT touch

No changes to `aten.rs`, `capture.rs`, `tape.rs`, `tools/golden/`,
`.github/`, or `torchnative/src/main/torch/` (the generated tree). No new
permanent pytest was added for the SmolLM2 round trip: it depends on a
network-cacheable Hub download and a full 135M-parameter forward pass, which
does not fit the existing `pytests/test_shim.py` fixtures (all synthetic,
in-repo, no network dependency) without either committing weights or adding
a network-conditional skip the rest of the suite does not have a precedent
for. The three round-trip runs above were captured directly in this document
instead, matching how docs/SAVE.md itself records its cross-process proofs.
