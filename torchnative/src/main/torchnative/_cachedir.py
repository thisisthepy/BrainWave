"""Where torchnative is allowed to write derived artefacts, on each platform.

`docs/devices/NPUCACHE.md` is the design record. This module exists because the
Intel NPU round needed somewhere to point `ov::cache_dir` and this repository
had **no cache-path convention at all** -- no `XDG_CACHE_HOME`, no
`%LOCALAPPDATA%`, no `~/Library/Caches`, no `HF_HOME` handling anywhere under
`torchnative/`. So this file is the house rule, and the QNN and CoreML caches
that come later are expected to call `backend_cache_dir` rather than invent a
second one.

**Pure, and injectable, on purpose.** `cache_root` takes the platform string and
the environment as arguments instead of reading `sys.platform` and `os.environ`,
which is the shape `torchnative.export.intelnpu.library_candidates(platform=...)`
already established for the same reason: this project ships wheels for macOS,
Linux, Windows, Android, iOS and wasm32, and a path rule that is right on Linux
and wrong on Windows is exactly the defect no single-platform test run would
catch. Every one of the six answers is decided here and asserted in
`rust/torch_c/pytests/test_ovcache.py`, from whichever one machine is running.

**What each platform gets, and why that one:**

  ``win32``        ``%LOCALAPPDATA%\\torchnative\\Cache``. Local, never
                   ``%APPDATA%``: a compiled NPU blob is keyed on this machine's
                   driver and device, so roaming it to another host is shipping
                   a cache entry to a machine it was not compiled for.
                   ``XDG_CACHE_HOME`` is deliberately *not* consulted here --
                   a Windows user with git-bash or WSL interop in their
                   environment must not get a different answer than one without.
  ``darwin``       ``~/Library/Caches/torchnative``. The macOS convention, and
                   the directory ``NSCachesDirectory`` names.
  ``linux``        ``$XDG_CACHE_HOME/torchnative``, defaulting to
                   ``~/.cache/torchnative`` per the XDG base directory spec.
  ``android``      the same XDG shape, which lands *inside app-private storage*
                   because CPython on Android (PEP 738, and Chaquopy before it)
                   sets ``HOME`` to the application's own files directory. There
                   is no world-writable location on Android and this does not
                   look for one.
  ``ios``          ``~/Library/Caches/torchnative`` -- the same rule as macOS,
                   and correct for the same reason: ``HOME`` inside the iOS
                   sandbox is the app container, so this is
                   ``<container>/Library/Caches``. That directory is purgeable
                   by the OS under disk pressure, which is the right property
                   for a cache and the reason it is not ``Application Support``.
  ``emscripten``   ``None``.
  ``wasi``         ``None``. Neither has a persistent filesystem by default, so
                   there is nothing a cache could survive in; returning ``None``
                   says that rather than writing into a MEMFS that vanishes.

**Why not under the Hugging Face cache.** These artefacts are derived from HF
checkpoints, so ``HF_HOME``/``HF_HUB_CACHE`` is a genuine candidate: it keeps a
model's derived files next to it and inherits a disk-location choice the user
has already made. It was rejected. ``huggingface_hub`` owns that tree's layout
and prunes it -- ``huggingface-cli delete-cache`` walks ``models--*/blobs`` and
``snapshots`` and removes what it does not recognise as its own -- so a
directory we put there is somewhere between "at risk" and "someone else's". It
is also undefined for the inputs that never came from the Hub (a local
``safetensors`` file, a ``state_dict``), which is a real fraction of what this
library lowers. And PROJECT.md's position on not writing into another
distribution's directory layout applies to a cache tree as much as to a package
tree. What is kept from the argument on the other side is the *inheritance*: the
user's disk-location choice is still honoured, one level up, through
``XDG_CACHE_HOME``/``%LOCALAPPDATA%`` and through the two overrides below.

**Off is a first-class answer.** ``None`` means "do not cache", and it is the
same value the wasm platforms return, so a caller has one no-cache state to
handle and not two. A shared or network home, CI, and a read-only filesystem all
need that value to be reachable, which is what ``TORCHNATIVE_CACHE_DIR=0`` does.
"""

from __future__ import annotations

import os

__all__ = [
    "CACHE_ROOT_ENV",
    "DISABLE_VALUES",
    "is_disabled",
    "cache_root",
    "backend_cache_dir",
]

#: The house-wide override. Set it to a path to move every torchnative cache;
#: set it to one of `DISABLE_VALUES` to turn caching off everywhere. Named after
#: `TORCHNATIVE_OPENVINO_C`, the naming precedent in `export/intelnpu.py`.
CACHE_ROOT_ENV = "TORCHNATIVE_CACHE_DIR"

#: Spellings of "off", matched case-insensitively after stripping. The empty
#: string is included: `FOO= python ...` is how a shell user most often means
#: "unset this", and treating it as a path would put the cache at the filesystem
#: root. `"none"` is included because that is what a Python user types.
DISABLE_VALUES = frozenset({"", "0", "off", "no", "none", "false", "disable", "disabled"})


def is_disabled(value: "str | None") -> bool:
    """Is this environment-variable value one of the ways of saying "off"?"""
    if value is None:
        return False
    return value.strip().lower() in DISABLE_VALUES


def _home(env: "dict[str, str]") -> "str | None":
    home = env.get("HOME")
    if home:
        return home
    profile = env.get("USERPROFILE")
    return profile or None


def cache_root(
    platform: "str | None" = None,
    env: "dict[str, str] | None" = None,
    android: "bool | None" = None,
) -> "str | None":
    """The directory torchnative may create per-backend cache trees under.

    Args:
        platform: a `sys.platform` string. Defaults to the running one. Taken as
            an argument so the Windows answer is checkable from a Mac.
        env: the environment. Defaults to `os.environ`. A dict, not a mutable
            view, so a test can supply exactly the variables it means to.
        android: whether this is an Android host. CPython 3.13 reports
            `sys.platform == "android"` (PEP 738) and this is then redundant, but
            older embeddings (Chaquopy, python-for-android) report `"linux"` and
            are only distinguishable by `sys.getandroidapilevel`.

    Returns:
        An absolute directory, or `None` when caching is off -- either because
        the user disabled it, because the platform has no persistent filesystem,
        or because there is no home directory to anchor to. `None` is not an
        error; it means "compile without a cache".
    """
    env = os.environ if env is None else env
    if android is None:
        import sys as _sys

        android = hasattr(_sys, "getandroidapilevel")
    if platform is None:
        import sys as _sys

        platform = _sys.platform

    override = env.get(CACHE_ROOT_ENV)
    if override is not None:
        return None if is_disabled(override) else os.path.abspath(override)

    if platform == "win32":
        # %LOCALAPPDATA%, and the `AppData\Local` reconstruction when it is
        # absent -- a stripped-down service environment often has USERPROFILE
        # and not LOCALAPPDATA. HOME is not consulted on Windows: an MSYS or
        # Cygwin HOME points somewhere the native process should not cache.
        local = env.get("LOCALAPPDATA")
        if not local:
            profile = env.get("USERPROFILE")
            if not profile:
                return None
            local = os.path.join(profile, "AppData", "Local")
        return os.path.join(local, "torchnative", "Cache")

    if platform in ("darwin", "ios"):
        home = _home(env)
        return os.path.join(home, "Library", "Caches", "torchnative") if home else None

    if platform in ("emscripten", "wasi"):
        # No persistent filesystem by default. Writing into a MEMFS that is
        # discarded at the end of the page's life is not a cache; it is work
        # plus a directory. Say so with None instead.
        return None

    if platform.startswith("linux") or platform == "android" or android:
        xdg = env.get("XDG_CACHE_HOME")
        if xdg and not is_disabled(xdg):
            return os.path.join(os.path.abspath(xdg), "torchnative")
        home = _home(env)
        return os.path.join(home, ".cache", "torchnative") if home else None

    # An unrecognised platform. XDG is the most portable guess, but guessing is
    # what produces the wrong-on-Windows answer this module exists to prevent,
    # so it declines instead. A caller who knows better sets CACHE_ROOT_ENV.
    return None


def backend_cache_dir(
    backend: str,
    backend_env: "str | None" = None,
    platform: "str | None" = None,
    env: "dict[str, str] | None" = None,
    android: "bool | None" = None,
) -> "str | None":
    """`<cache_root>/<backend>`, or a backend-specific override, or `None`.

    Precedence, most specific first:

    1. `backend_env` (e.g. `TORCHNATIVE_OPENVINO_CACHE_DIR`) -- used verbatim as
       the directory, with no `<backend>` component appended, because a user who
       names a directory means that directory. Also the place to disable one
       backend's cache while leaving the others alone.
    2. `CACHE_ROOT_ENV` -- moves or disables every backend at once.
    3. The platform convention from `cache_root`.

    This does **not** touch the filesystem. Creating the directory, and deciding
    what to do when that fails, belongs to the caller that knows how to announce
    it -- see `torchnative.export.intelnpu.ensure_cache_dir`.
    """
    env = os.environ if env is None else env
    if backend_env:
        specific = env.get(backend_env)
        if specific is not None:
            return None if is_disabled(specific) else os.path.abspath(specific)
    root = cache_root(platform=platform, env=env, android=android)
    return None if root is None else os.path.join(root, backend)
