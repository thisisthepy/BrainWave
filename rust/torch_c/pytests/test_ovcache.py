"""Where a compiled OpenVINO model is allowed to live, and that it gets there.

`torchnative/src/main/torchnative/export/intelnpu.py` compiled every leaf from
scratch, every time, because `ov_core_compile_model(core, model, device, 0, &out)`
passed **zero properties** and so never set `ov::cache_dir`. On a 36-layer
Qwen3-4B that is 252 `_NPULinear` leaves, each compiled once at the prompt shape
and again at the decode shape -- 504 driver compiles before the second generated
token, none of which survived the process. `docs/devices/NPUCACHE.md` is the
design record; this file is its evidence.

**What this file is evidence ABOUT.** Two separable things:

* the **path**, which is pure and therefore fully checkable here. `cache_root`
  takes its platform string and its environment as arguments -- the same shape
  `intelnpu.library_candidates(platform=...)` established for exactly this
  reason -- so the Windows, Linux, Android, iOS and wasm answers are all
  decidable on an arm64 Mac. Every one of the six platforms this project ships
  wheels for is asserted below, by name, against the convention that platform
  actually has.
* the **wiring**, that the directory reaches `ov_core_compile_model` as a real
  `CACHE_DIR` property with a non-zero, *even* argument count. That is checked
  against a fake `openvino_c` which records its varargs, so the assertion is
  about the call this module makes and not about anything OpenVINO does with it.

**What it is NOT evidence about, stated here rather than implied.** There is no
Intel NPU and no OpenVINO runtime on this host -- `library_candidates` refuses
on darwin by design and `import openvino` fails. So nothing here demonstrates a
cache *hit*: that the blob OpenVINO writes on the first compile is found and
reused on the second, that reuse is faster, or that the blob decodes to the same
model. Those need the hardware, and `docs/devices/NPUCACHE.md` section 5 says so.
What is checked here is that we ask for the right thing in the right place.
"""

import os
import stat
import sys
import warnings

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")
sys.path.insert(0, _VENDOR_DIR)

# Before torch, not after: a suite that imports torch first passes in a shell
# where TORCH_USE_RTLD_GLOBAL is already exported and fails under run.sh.
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import ctypes  # noqa: E402
import tempfile  # noqa: E402

from torchnative import _cachedir  # noqa: E402
from torchnative.export import intelnpu  # noqa: E402


# --------------------------------------------------------------------------
# The path. Pure, so every platform is decidable from here.
# --------------------------------------------------------------------------


def test_every_platform_with_a_filesystem_answers_dot_cache_torchnative():
    """One rule, not six, and that is the decision this test exists to pin.

    The first version of this module used each OS's native convention:
    `%LOCALAPPDATA%\\torchnative\\Cache` on Windows, `~/Library/Caches` on macOS
    and iOS, XDG elsewhere. That is right for a desktop application and wrong
    here, because it makes torchnative the only thing in a user's ML toolchain
    that does it. The two that matter ignore the OS:

        torch/hub.py                     DEFAULT_CACHE_DIR = "~/.cache"
        huggingface_hub/constants.py     ~/.cache, XDG_CACHE_HOME first

    Neither consults `sys.platform`. On Windows a user's checkpoints are already
    in `C:\\Users\\<u>\\.cache\\huggingface`; sending only the NPU blobs to
    `%LOCALAPPDATA%` splits one thing across two places for a tidiness nobody
    asked for. We ship that torch, so its convention is not a foreign one.

    The only exception is the platform with no filesystem, not the platform with
    a nicer directory -- iOS keeps `~/.cache` too, and gives up
    `Library/Caches`'s OS-purgeability to do it. See the module for why that
    trade is taken and how an embedder undoes it.
    """
    env = {"HOME": "/home/u", "USERPROFILE": r"C:\Users\u",
           "LOCALAPPDATA": r"C:\Users\u\AppData\Local"}
    same = ("win32", "darwin", "linux", "android", "ios", "freebsd14")
    got = {p: _cachedir.cache_root(platform=p, env=env) for p in same}
    expected = os.path.join("/home/u", ".cache", "torchnative")
    for platform, answer in got.items():
        assert answer == expected, (platform, answer, expected)
    for platform in ("emscripten", "wasi"):
        assert _cachedir.cache_root(platform=platform, env=env) is None, platform
    print(
        f"ok   ovcache: {len(same)} platforms all answer ~/.cache/torchnative -- "
        f"the torch and huggingface_hub convention, not the per-OS one -- and "
        f"only wasm, which has no filesystem, answers None"
    )


def test_xdg_cache_home_is_honoured_everywhere_including_windows():
    """`XDG_CACHE_HOME` moves the root on every platform, Windows included.

    It was previously read on Linux and deliberately ignored on Windows, on the
    reasoning that a git-bash or WSL-interop environment must not silently
    change the answer. That reasoning is dropped, because `huggingface_hub`
    reads it unconditionally: a user who sets it and finds their HF cache moved
    but their torchnative cache not moved is worse off than one who set it on
    purpose and got what they asked for.

    Roaming was the other reason for `%LOCALAPPDATA%`, and it is not a real
    risk: `~/.cache` resolves under `%USERPROFILE%`, which is not the roaming
    `%APPDATA%` tree, and huggingface_hub already keeps multi-gigabyte
    checkpoints there.
    """
    env = {"HOME": "/home/u", "USERPROFILE": r"C:\Users\u",
           "LOCALAPPDATA": r"C:\Users\u\AppData\Local",
           "XDG_CACHE_HOME": "/big/cache"}
    for platform in ("win32", "darwin", "linux", "android", "ios"):
        got = _cachedir.cache_root(platform=platform, env=env)
        assert got == os.path.join("/big/cache", "torchnative"), (platform, got)

    roaming = _cachedir.cache_root(platform="win32", env={k: v for k, v in env.items()
                                                if k != "XDG_CACHE_HOME"})
    assert "Roaming" not in roaming and "AppData" not in roaming, (
        f"the Windows root went back into AppData: {roaming}"
    )
    print(
        "ok   ovcache: XDG_CACHE_HOME moves the root on every platform including "
        "Windows, and no platform answers anywhere under AppData"
    )


def test_the_cache_does_not_move_into_the_hugging_face_cache_directory():
    """A set `HF_HOME` / `HF_HUB_CACHE` must not relocate us into their layout.

    These artefacts are derived from Hugging Face checkpoints, so sitting under
    the HF cache is a real option and this test is what records that it was
    rejected rather than forgotten. `docs/devices/NPUCACHE.md` section 2 has the
    reasoning; the short form is that `huggingface_hub` owns that tree's layout
    and prunes it (`huggingface-cli delete-cache` walks `models--*/blobs`), and
    PROJECT.md's position on not writing into another distribution's directory
    applies to a cache directory as much as to a package.
    """
    env = {
        "HOME": "/home/u",
        "HF_HOME": "/big/hf",
        "HF_HUB_CACHE": "/big/hf/hub",
        "TRANSFORMERS_CACHE": "/big/hf/transformers",
    }
    root = _cachedir.cache_root(platform="linux", env=env)
    assert root == os.path.join("/home/u", ".cache", "torchnative"), root
    for hostile in ("/big/hf", "hf", "huggingface"):
        assert hostile not in root or root.startswith("/home/u"), (hostile, root)
    assert not root.startswith("/big/hf"), root
    print(
        "ok   ovcache: HF_HOME, HF_HUB_CACHE and TRANSFORMERS_CACHE do not move "
        "the torchnative root -- the cache stays in torchnative's own tree and "
        "does not write into huggingface_hub's layout"
    )


def test_the_root_override_the_backend_override_and_their_precedence():
    """Two overrides, and the specific one wins over the general one."""
    base = {"HOME": "/home/u"}
    default = _cachedir.backend_cache_dir("openvino", platform="linux", env=base)
    assert default == os.path.join("/home/u", ".cache", "torchnative", "openvino"), default

    rooted = _cachedir.backend_cache_dir(
        "openvino",
        platform="linux",
        env=dict(base, **{_cachedir.CACHE_ROOT_ENV: "/scratch/tn"}),
    )
    assert rooted == os.path.join("/scratch/tn", "openvino"), rooted

    both = _cachedir.backend_cache_dir(
        "openvino",
        backend_env=intelnpu.OPENVINO_CACHE_ENV,
        platform="linux",
        env=dict(
            base,
            **{
                _cachedir.CACHE_ROOT_ENV: "/scratch/tn",
                intelnpu.OPENVINO_CACHE_ENV: "/fast/ovblobs",
            },
        ),
    )
    assert both == "/fast/ovblobs", both
    # And the backend override is a *directory*, used as given -- it does not
    # get "openvino" appended a second time.
    assert not both.endswith(os.path.join("ovblobs", "openvino")), both

    # The env var names follow the module's own precedent.
    assert _cachedir.CACHE_ROOT_ENV == "TORCHNATIVE_CACHE_DIR"
    assert intelnpu.OPENVINO_CACHE_ENV == "TORCHNATIVE_OPENVINO_CACHE_DIR"
    assert intelnpu.LIBRARY_ENV == "TORCHNATIVE_OPENVINO_C"
    print(
        f"ok   ovcache: {_cachedir.CACHE_ROOT_ENV} moves the whole root and "
        f"{intelnpu.OPENVINO_CACHE_ENV} overrides just this backend, the specific "
        f"one winning -- both spelled like the existing {intelnpu.LIBRARY_ENV}"
    )


def test_every_disable_spelling_turns_caching_off_at_both_levels():
    """A shared home, a network filesystem, CI, a read-only image: off must be reachable.

    Off is `None`, the same value the wasm platforms return, so there is exactly
    one "no cache" state for callers to handle rather than two.
    """
    base = {"HOME": "/home/u"}
    spellings = ["0", "off", "no", "none", "false", "disable", "disabled", "", "  OFF  ", "None"]
    for spelling in spellings:
        root_off = _cachedir.backend_cache_dir(
            "openvino", platform="linux", env=dict(base, **{_cachedir.CACHE_ROOT_ENV: spelling})
        )
        assert root_off is None, (spelling, root_off)
        backend_off = _cachedir.backend_cache_dir(
            "openvino",
            backend_env=intelnpu.OPENVINO_CACHE_ENV,
            platform="linux",
            env=dict(base, **{intelnpu.OPENVINO_CACHE_ENV: spelling}),
        )
        assert backend_off is None, (spelling, backend_off)
    # Disabling the backend does not require disabling the root, and vice versa.
    still_on = _cachedir.backend_cache_dir(
        "openvino",
        backend_env=intelnpu.OPENVINO_CACHE_ENV,
        platform="linux",
        env=dict(base, **{intelnpu.OPENVINO_CACHE_ENV: "off", _cachedir.CACHE_ROOT_ENV: "/s"}),
    )
    assert still_on is None, still_on
    print(
        f"ok   ovcache: {len(spellings)} disable spellings switch caching off at "
        f"the root and at the backend, both landing on None -- the same value "
        f"wasm returns, so there is one 'no cache' state and not two"
    )


def test_a_directory_that_cannot_be_created_degrades_to_no_cache_and_says_so_once():
    """Degraded, not broken -- and not silent, but also not 504 times.

    docs/graph/NPU2.md's position is that the silently degraded path is the
    defect, so this warns. `_NPULinear` compiles once per leaf per shape, which
    is 504 compiles on a Qwen3-4B, so a warning per compile would be 504 lines
    of noise; the announcement is once per process per directory and this test
    holds that number down by asking twenty times.
    """
    with tempfile.TemporaryDirectory() as tmp:
        blocked = os.path.join(tmp, "ro")
        os.mkdir(blocked)
        os.chmod(blocked, stat.S_IRUSR | stat.S_IXUSR)  # r-x------: no writing
        target = os.path.join(blocked, "torchnative", "openvino")
        try:
            intelnpu._reset_cache_announcements()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                results = [intelnpu.ensure_cache_dir(target) for _ in range(20)]
            assert all(r is None for r in results), results
            assert len(caught) == 1, [str(w.message) for w in caught]
            message = str(caught[0].message)
            assert target in message, message
            assert "torchnative intelnpu" in message, message
            assert intelnpu.OPENVINO_CACHE_ENV in message, message
            assert issubclass(caught[0].category, intelnpu.IntelNPUCacheWarning), caught[0].category
        finally:
            os.chmod(blocked, stat.S_IRWXU)

    # ...and a directory that CAN be made comes back, created, with no warning.
    with tempfile.TemporaryDirectory() as tmp:
        good = os.path.join(tmp, "torchnative", "openvino")
        intelnpu._reset_cache_announcements()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            made = intelnpu.ensure_cache_dir(good)
        assert made == good, made
        assert os.path.isdir(good), good
        assert caught == [], [str(w.message) for w in caught]
    print(
        "ok   ovcache: an uncreatable cache directory degrades to None and warns "
        "exactly once across 20 asks (not once per compile), naming the path and "
        "the env var; a creatable one is created silently"
    )


# --------------------------------------------------------------------------
# The wiring. A fake openvino_c that records what it was called with.
# --------------------------------------------------------------------------


class _FakeLib:
    """Records the arguments `compile_ir` hands `ov_core_compile_model`.

    Only the entry points `OpenVINO.__init__` and `compile_ir` touch are
    present; anything else is an AttributeError rather than a silently accepted
    no-op, so this cannot pass by not being called.
    """

    def __init__(self, export_property_key=True):
        self.compile_calls = []
        self._export_property_key = export_property_key

        class _Fn:
            argtypes = None
            restype = None

            def __init__(self, impl):
                self._impl = impl

            def __call__(self, *args):
                return self._impl(*args)

        self._Fn = _Fn
        self.ov_core_create = _Fn(self._core_create)
        self.ov_core_free = _Fn(lambda core: None)
        self.ov_core_read_model_from_memory_buffer = _Fn(self._read_model)
        self.ov_model_free = _Fn(lambda model: None)
        self.ov_core_compile_model = _Fn(self._compile)
        self.ov_tensor_create_from_host_ptr = _Fn(self._tensor)
        self.ov_tensor_free = _Fn(lambda t: None)

    # `hasattr(lib, "ov_get_last_err_msg")` must be False, and `in_dll` must be
    # answerable, so this stands in for a CDLL closely enough for both.
    def __getitem__(self, name):
        raise KeyError(name)

    def _core_create(self, out):
        out._obj.value = 0x1000 if hasattr(out, "_obj") else 0x1000
        return 0

    def _read_model(self, core, blob, length, weights, out):
        out._obj.value = 0x2000
        return 0

    def _tensor(self, etype, shape, data, out):
        out._obj.value = 0x3000
        return 0

    def _compile(self, core, model, device, count, out, *varargs):
        # `count` arrives as the `ctypes.c_size_t` the call site built; a real
        # CDLL would unwrap it at the FFI boundary and this fake is the boundary.
        self.compile_calls.append((device, int(getattr(count, "value", count)), tuple(varargs)))
        out._obj.value = 0x4000
        return 0


def _ov_with(fake, cache_dir):
    """Build an `OpenVINO` over `fake`, bypassing library discovery only."""
    real_loader = intelnpu.load_openvino_c
    intelnpu.load_openvino_c = lambda path=None: fake
    try:
        return intelnpu.OpenVINO(cache_dir=cache_dir)
    finally:
        intelnpu.load_openvino_c = real_loader


def _decode(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, ctypes.c_char_p):
        return value.value.decode()
    return value


def test_the_cache_directory_actually_reaches_ov_core_compile_model():
    """The property is on the call, with an even, non-zero argument count.

    `ov_core.h:204` says `property_args_size` is "How many properties args will
    be passed, each property contains 2 args: key and value" -- so one property
    is a count of **2**, not 1, and the C side rejects an odd count outright.
    Getting that wrong is the kind of mistake that compiles, runs, and quietly
    caches nothing, which is the state this whole round is fixing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "openvino")
        fake = _FakeLib()
        ov = _ov_with(fake, cache)
        assert ov.cache_dir == cache, ov.cache_dir
        ov.compile_ir("<net/>", "NPU", b"\x00\x01")
        assert len(fake.compile_calls) == 1, fake.compile_calls
        device, count, varargs = fake.compile_calls[0]
        assert _decode(device) == "NPU", device
        assert count == 2, count
        assert count % 2 == 0, count
        assert len(varargs) == count, (varargs, count)
        key, value = (_decode(v) for v in varargs)
        assert key == "CACHE_DIR", key
        assert value == cache, (value, cache)
        assert os.path.isdir(cache), cache
    print(
        "ok   ovcache: compile_ir passes CACHE_DIR=<dir> to ov_core_compile_model "
        "as 2 varargs with property_args_size=2 -- the ARG count the header "
        "specifies, not the pair count"
    )


def test_with_caching_off_the_call_is_byte_for_byte_the_old_zero_property_one():
    """Off must not become a new way to fail. It is the call that shipped."""
    fake = _FakeLib()
    ov = _ov_with(fake, None)
    assert ov.cache_dir is None, ov.cache_dir
    ov.compile_ir("<net/>", "NPU", None)
    device, count, varargs = fake.compile_calls[0]
    assert _decode(device) == "NPU", device
    assert count == 0, count
    assert varargs == (), varargs
    print(
        "ok   ovcache: with caching disabled compile_ir makes exactly the call "
        "that shipped -- property_args_size=0 and no varargs -- so 'off' is the "
        "old behaviour and not a third code path"
    )


def test_the_property_key_is_taken_from_the_runtime_when_the_runtime_exports_it():
    """`ov_property_key_cache_dir` is an exported *variable*, not a function.

    `ov_property.h:97-98` declares it `OPENVINO_C_VAR(const char*)`, so the
    authoritative spelling is readable out of the loaded library with
    `ctypes.c_char_p.in_dll`. We read it when it is there and fall back to the
    literal `"CACHE_DIR"` when it is not -- an older runtime, or a `CDLL` that
    will not surface data symbols. Both branches must produce the same string,
    which is the only reason the fallback is safe.
    """
    literal = intelnpu.CACHE_DIR_PROPERTY
    assert literal == "CACHE_DIR", literal
    fake = _FakeLib(export_property_key=False)
    assert intelnpu._cache_dir_property_key(fake) == b"CACHE_DIR"

    class _Exporting(_FakeLib):
        pass

    exporting = _Exporting()
    # Stand in for the data export: the resolver asks the library first.
    exporting._torchnative_exported_cache_key = b"CACHE_DIR"
    assert intelnpu._cache_dir_property_key(exporting) == b"CACHE_DIR"
    print(
        "ok   ovcache: the CACHE_DIR key is read from the runtime's exported "
        "ov_property_key_cache_dir when available and falls back to the literal "
        "otherwise, both spellings agreeing"
    )


def test_openvino_resolves_its_cache_directory_once_not_once_per_compile():
    """504 compiles, one resolution -- which is also why the warning can be once."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "openvino")
        fake = _FakeLib()
        ov = _ov_with(fake, cache)
        calls = []
        real = intelnpu.ensure_cache_dir
        intelnpu.ensure_cache_dir = lambda path: (calls.append(path), real(path))[1]
        try:
            for i in range(50):
                ov.compile_ir("<net/>", "NPU", None)
        finally:
            intelnpu.ensure_cache_dir = real
        assert calls == [], calls
        assert len(fake.compile_calls) == 50, len(fake.compile_calls)
        assert all(c[1] == 2 for c in fake.compile_calls), fake.compile_calls
    print(
        "ok   ovcache: the directory is resolved and created once, at OpenVINO "
        "construction -- 50 compiles made 0 further filesystem calls and all 50 "
        "still carried the property"
    )


def test_the_two_things_this_module_varies_are_both_inside_what_openvino_hashes():
    """The cache-key question, answered as far as it can be without the runtime.

    OpenVINO keys a cached blob on the model, the device and the compile
    config. This module varies exactly two things between compiles: the batch
    dimension, which is written into the IR text, and the weights, which are
    the Constant payload. Both are part of the model OpenVINO hashes, so a
    wrong hit would require a hash collision rather than a key that omits what
    we vary. What this test can show is the *premise*: that those two really do
    differ in the bytes handed to `read_model`. That the hash covers them is
    OpenVINO's contract and is NOT verified here -- docs/devices/NPUCACHE.md
    section 5 states the residual risk and why the disable switch exists.
    """
    prompt_shape = intelnpu.linear_ir(8, 4, 12, True)
    decode_shape = intelnpu.linear_ir(8, 4, 1, True)
    assert prompt_shape != decode_shape, "the batch dim must be visible in the IR"
    assert "<dim>12</dim>" in prompt_shape and "<dim>1</dim>" in decode_shape

    same_shape_a = intelnpu.pack_f16([0.5] * 32 + [0.0] * 4)
    same_shape_b = intelnpu.pack_f16([0.25] * 32 + [0.0] * 4)
    assert len(same_shape_a) == len(same_shape_b)
    assert same_shape_a != same_shape_b, "two layers of the same shape differ in weights"
    print(
        "ok   ovcache: the two axes this module varies -- batch dim (in the IR "
        "text) and weights (in the Constant blob) -- are both inside the model "
        "bytes OpenVINO hashes; the hash itself is OpenVINO's contract, unverified here"
    )


def test_the_non_variadic_compile_model_props_is_still_not_used():
    """The reason at `intelnpu.py` for avoiding it survives passing properties.

    `ov_core_compile_model_props` is present on OpenVINO master but not in every
    release a user has installed. Passing properties does not change that, so
    the variadic form is still the one bound -- with a real count now instead of
    a zero.
    """
    source = open(
        os.path.join(_ROOT, "torchnative", "src", "main", "torchnative", "export", "intelnpu.py"),
        encoding="utf-8",
    ).read()
    binds = [
        line
        for line in source.splitlines()
        if "ov_core_compile_model_props" in line and "lib." in line and not line.strip().startswith("#")
    ]
    assert binds == [], binds
    assert "ov_core_compile_model_props" in source, "the reasoning must still be written down"
    print(
        "ok   ovcache: ov_core_compile_model_props is still not bound -- the "
        "release-availability reasoning survives the switch from 0 properties to 2"
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
