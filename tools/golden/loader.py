"""Load the built `torch._C` shim and resolve `torch.ops.aten.*` callables.

Kept deliberately independent of `sys.path`/cwd tricks: an earlier manual
probe while building this harness found a *different* stray `_C.so` (an
Android build artefact from unrelated, concurrent work) sitting in `/tmp`
and getting picked up ahead of the intended one purely because the shell's
cwd was `/tmp`. Loading by explicit file path through `importlib` sidesteps
that whole class of collision -- nothing here ever depends on `sys.path[0]`
or the process's current directory.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import shutil
import tempfile
from types import ModuleType


class ShimLoadError(RuntimeError):
    pass


def _candidate_artefacts(explicit_path: str | None) -> list[str]:
    if explicit_path:
        return [explicit_path]
    env_path = os.environ.get("TORCH_C_ARTEFACT")
    if env_path:
        return [env_path]
    # **The default is one shared path, and several checkouts build into it.**
    # Falling back here from a worktree grades whatever another agent most
    # recently built, and it does not look like an error -- it looks like a
    # result. It happened: a round reported `8470/8476` with six float8
    # in-place cases announcing a gap "appears CLOSED", because it had picked
    # up a concurrent round's binary. Same family as the `$TMPDIR` staging
    # collision and the shared `CARGO_TARGET_DIR`, and the same shape every
    # time: shared mutable build state, failing as a wrong verdict rather than
    # as an error.
    #
    # So the fallback stays -- a single-checkout run should not have to set an
    # environment variable -- but it says out loud which artefact it took, and
    # whether that artefact belongs to the tree it was invoked from.
    default = [
        "/Volumes/macMini/caches/cargo-target/release/lib_C.dylib",
        "/Volumes/macMini/caches/cargo-target/release/lib_C.so",
    ]
    here = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for path in default:
        if os.path.exists(path):
            print(
                f"golden: TORCH_C_ARTEFACT is unset, taking the shared default\n"
                f"        {path}\n"
                f"        invoked from {here}\n"
                f"        Several checkouts build into that path. If another is "
                f"building, this grades its binary.\n"
                f"        Set TORCH_C_ARTEFACT to the one you built.",
                file=sys.stderr,
            )
            break
    return default


def load_shim(explicit_path: str | None = None) -> ModuleType:
    """Copy the built artefact into a private temp dir named `_C.so` (the
    extension loader keys off that suffix) and import it as module `_C`.

    A private `tempfile.mkdtemp` is used instead of a shared, predictable
    path so this never collides with another process's staging directory.
    """
    for candidate in _candidate_artefacts(explicit_path):
        if os.path.isfile(candidate):
            artefact = candidate
            break
    else:
        tried = ", ".join(_candidate_artefacts(explicit_path))
        raise ShimLoadError(
            "no torch._C artefact found. Tried: "
            f"{tried}. Build it per docs/TORCH_C.md §7, or pass "
            "--artefact/TORCH_C_ARTEFACT explicitly."
        )

    stage = tempfile.mkdtemp(prefix="golden-harness-")
    so_path = os.path.join(stage, "_C.so")
    shutil.copy(artefact, so_path)

    spec = importlib.util.spec_from_file_location("_C", so_path)
    if spec is None or spec.loader is None:
        raise ShimLoadError(f"could not create an import spec for {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_torch_overload(torch_module, op_name: str):
    """`"aten.add.Tensor"` -> `torch.ops.aten.add.Tensor`.

    Overload is part of the identity (docs/TORCH_C.md §1: "오버로드가 키의
    일부입니다"), so this refuses to guess one when it is missing.
    """
    parts = op_name.split(".")
    if len(parts) != 3:
        raise ShimLoadError(
            f"op name {op_name!r} is not in the expected "
            "'<namespace>.<op>.<overload>' shape"
        )
    namespace, op, overload = parts
    try:
        ns_obj = getattr(torch_module.ops, namespace)
        packet = getattr(ns_obj, op)
        return getattr(packet, overload)
    except AttributeError as e:
        raise ShimLoadError(
            f"torch has no op matching {op_name!r} "
            f"(torch.ops.{namespace}.{op}.{overload}): {e}"
        ) from e
