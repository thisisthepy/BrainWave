"""Are the nine wheel targets real, and does each one's tag mean what it says?

`tools/wheel/build.py` grew from one architecture per platform to two
(docs/platform/WHEELMATRIX.md). That change is almost entirely *parameterisation* -- one
`AndroidTarget` became two, one `LinuxTarget` became two, one `WindowsTarget`
became two -- and parameterisation is the shape of change this repository has
been burned by before, because the second instance inherits the first one's
constants without saying so. Three concrete instances, all of which this file
would have caught and none of which any existing test did:

  * `_confirm_pep600_spelling` checked a floor of glibc 2.5, which is PEP 600's
    grammatical floor and is *correct for x86-64 only*. `manylinux_2_12_aarch64`
    passes that check and matches no installer: aarch64 is not in PEP 513 or PEP
    571, so pip's own generator floors it at 2.17. A tag no installer generates
    is a wheel nobody can install, and it looks exactly like a working one on
    the machine that built it.
  * `verify_linux.py` and `verify_windows.py` resolved every wheel's symbols
    against a *hardcoded* x86-64 distribution. Because `libpython3.13.so` and
    `python3.dll` export the same names on both architectures, an aarch64 wheel
    checked against the x86-64 interpreter prints `0 unresolved` and PASSes.
    The check does not fail; it stops being a check.
  * `TARGETS` is a dict comprehension keyed on `t.key`. Two entries sharing a
    key collapse into one silently, leaving eight targets where nine were
    written -- and the survivor is a perfectly good target, just not the one
    that went missing.

So this file asserts three things about each target: its **tag** is a name
`packaging` -- pip's own code -- would accept, its **place in the registry** is
held rather than assumed, and if it **refuses** it does so by name with a
reason. It builds nothing and needs no network; the cases that read a target
CPython distribution skip by name when it is absent, because a check that could
not run must not read as one that passed.
"""

import os
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
BUILD_PY = REPO / "tools" / "wheel" / "build.py"

sys.path.insert(0, str(REPO / "tools" / "wheel"))
import build as _build  # noqa: E402

SKIPPED = []


def _skip(why):
    SKIPPED.append(why)
    print(f"     SKIP {why}")


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------

def test_the_registry_holds_exactly_what_it_declares():
    """`check_registry` is the guard; this is the guard passing on the real
    registry. Both halves matter -- the next test breaks it on purpose."""
    _build.check_registry()
    assert sorted(_build.TARGETS) == sorted(_build.EXPECTED_TARGET_KEYS)


def test_the_registry_cannot_silently_lose_an_entry():
    """A `key` copied and not edited makes two entries collide, and the dict
    comprehension keeps the last one. Nothing is raised, nothing is printed,
    and `--target` quietly offers one fewer choice.

    Reproduced rather than argued: the registry is rebuilt with a duplicate
    key, and `check_registry` has to name the key that vanished.
    """
    original = dict(_build.TARGETS)
    try:
        collided = {k: v for k, v in original.items() if k != "linux-aarch64"}
        _build.TARGETS.clear()
        _build.TARGETS.update(collided)
        try:
            _build.check_registry()
        except SystemExit as exc:
            said = str(exc)
        else:
            raise AssertionError(
                "check_registry passed a registry with 8 of its 9 targets")
        assert "linux-aarch64" in said, (
            f"the refusal does not name the lost target: {said!r}")
        assert "missing" in said, f"the refusal does not say what happened: {said!r}"
    finally:
        _build.TARGETS.clear()
        _build.TARGETS.update(original)


def test_every_non_apple_platform_carries_two_architectures():
    """The point of the whole change. Apple's two targets are a device and a
    simulator rather than two CPUs, and WASM has one architecture by
    construction, so neither is counted here."""
    by_platform = {}
    for key in _build.TARGETS:
        platform = key.split("-")[0]
        by_platform.setdefault(platform, []).append(key)
    for platform in ("android", "linux", "windows"):
        keys = sorted(by_platform.get(platform, []))
        assert len(keys) == 2, (
            f"{platform} has {len(keys)} target(s) ({keys}), expected two -- "
            "one per CPU architecture")


def test_every_target_names_a_rust_target_that_is_installed():
    """A registry entry whose triple rustup does not have is a target that
    cannot be built, and it should not look like one that merely has not been
    built yet."""
    try:
        proc = subprocess.run(["rustup", "target", "list", "--installed"],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return _skip(f"rustup not runnable here ({exc.__class__.__name__})")
    if proc.returncode != 0:
        return _skip("rustup target list failed")
    installed = set(proc.stdout.split())
    missing = sorted(t.rust_target for t in _build.TARGETS.values()
                     if t.rust_target not in installed)
    assert not missing, (
        f"registry entries whose rust target is not installed: {missing}\n"
        f"  fix: rustup target add {' '.join(missing)}")


# --------------------------------------------------------------------------
# The tags -- each on its own terms
# --------------------------------------------------------------------------

def test_the_android_tags_are_what_packaging_generates():
    """PEP 738, and the one family `packaging` can be asked directly:
    `android_platforms` takes the api level and abi as arguments rather than
    reading the running system, so a cross build can put the question."""
    try:
        from packaging import tags as ptags
    except ImportError:
        return _skip("packaging not importable")
    if not hasattr(ptags, "android_platforms"):
        return _skip(f"packaging {getattr(__import__('packaging'), '__version__', '?')}"
                     " has no android_platforms")
    for abi in ("arm64_v8a", "x86_64"):
        accepted = list(ptags.android_platforms(api_level=21, abi=abi))
        assert f"android_21_{abi}" in accepted, (
            f"packaging.tags.android_platforms(21, {abi!r}) does not yield "
            f"android_21_{abi}; it starts {accepted[:3]}")
    # ...and the two do not collide. `_ANDROID_ABIS` maps CPython's MULTIARCH
    # architecture to the NDK's ABI name, and `x86_64` is the one entry where
    # the two vocabularies happen to agree -- which is exactly the case where a
    # mapping is easiest to leave out and never notice.
    assert _build._ANDROID_ABIS["aarch64"] == "arm64-v8a"
    assert _build._ANDROID_ABIS["x86_64"] == "x86_64"


def test_the_manylinux_floor_is_per_architecture_and_packaging_agrees():
    """PEP 600's grammar is not the whole constraint, and this is the case the
    single-architecture version of this file got wrong.

    `manylinux_2_12_aarch64` is well-formed, is above manylinux1's glibc 2.5,
    and matches nothing. aarch64 does not appear in PEP 513 (manylinux1) or PEP
    571 (manylinux2010); it enters the scheme at PEP 599 (manylinux2014, glibc
    2.17), so `packaging._manylinux.platform_tags` -- the generator pip runs --
    never emits a lower name for it. The same number is fine on x86-64.
    """
    def refused(tag, arch):
        try:
            _build._confirm_pep600_spelling(tag, arch)
        except SystemExit as exc:
            return str(exc)
        return ""

    assert refused("manylinux_2_17_aarch64", "aarch64") == ""
    assert refused("manylinux_2_17_x86_64", "x86_64") == ""
    assert refused("manylinux_2_12_x86_64", "x86_64") == "", (
        "glibc 2.12 is manylinux2010 and is legitimate on x86-64")
    said = refused("manylinux_2_12_aarch64", "aarch64")
    assert "2.17" in said, (
        f"manylinux_2_12_aarch64 was not refused for the right reason: {said!r}")
    # A tag naming an architecture other than the target's. The two Linux
    # targets share this function, so this is a live way to mislabel a wheel.
    said = refused("manylinux_2_17_x86_64", "aarch64")
    assert "names architecture" in said, f"got: {said!r}"


def test_packaging_itself_yields_the_manylinux_tags_this_repo_emits():
    """Not this file's reading of PEP 599 -- pip's code, run.

    `packaging.tags` has no `manylinux_platforms`, which is why every Linux
    build printed "tag spelling unchecked" for months. The generator exists;
    it just reads the *host* glibc, which on a Mac is none. Substituting that
    one host probe is what `_confirm_manylinux_with_packaging` does, and this
    asserts the substitution still lets a wrong tag through as wrong.
    """
    try:
        import packaging._manylinux  # noqa: F401
    except ImportError:
        return _skip("packaging._manylinux not importable")

    def refused(tag, arch):
        try:
            _build._confirm_manylinux_with_packaging(tag, arch)
        except SystemExit as exc:
            return str(exc)
        return ""

    assert refused("manylinux_2_17_aarch64", "aarch64") == ""
    assert refused("manylinux_2_17_x86_64", "x86_64") == ""
    assert refused("manylinux_2_12_aarch64", "aarch64") != "", (
        "packaging's own generator was asked and did not refuse "
        "manylinux_2_12_aarch64 -- either the substitution is too generous or "
        "packaging's per-arch floor has changed")
    # An architecture manylinux does not cover at all.
    assert refused("manylinux_2_17_mips64", "mips64") != ""


def test_the_windows_tag_is_a_name_read_out_of_the_distribution():
    """`win_arm64` is right and cannot be reached by analogy.

    Windows tags carry no version, so there is no floor to compute and nothing
    to get subtly wrong -- which is precisely why the *name* is the thing to
    get wrong: `win_aarch64` and `win_arm64ec` are equally plausible spellings
    of "64-bit ARM Windows" and match no installer.

    The chain `WindowsTarget` follows is CPython's own and every link is on
    disk: `packaging` has no Windows tag code at all (it falls through to
    `_normalize_string(sysconfig.get_platform())`), `get_platform()` on `nt`
    tests `sys.version` for `amd64`/`(arm)`/`(arm64)` in that order, and
    `sys.version` embeds the `[MSC v.… (ARM64)]` string compiled into
    `python313.dll`. So the tag is read, not assumed -- and this checks it
    against both distributions, because a derivation that answers the same
    thing for both is not a derivation.
    """
    try:
        from packaging.tags import _normalize_string
    except ImportError:
        return _skip("packaging not importable")
    assert _normalize_string("win-arm64") == "win_arm64"
    assert _normalize_string("win-amd64") == "win_amd64"

    got = {}
    for key in ("windows-x86_64", "windows-arm64"):
        target = _build.TARGETS[key]
        if not (target.python_root / "python313.dll").exists():
            _skip(f"{key}: no python313.dll under {target.python_root}")
            continue
        got[key] = _build._confirm_windows_normalisation(target._platform_name())
    if len(got) < 2:
        return
    assert got["windows-x86_64"] == "win_amd64", got
    assert got["windows-arm64"] == "win_arm64", got
    assert got["windows-x86_64"] != got["windows-arm64"], (
        "both Windows distributions derive the same tag, so the derivation is "
        "reading something that does not vary -- which is the same as writing "
        "the tag down")


def test_the_windows_targets_refuse_the_other_architectures_distribution():
    """The two Windows distributions differ in no filename, so the file-shape
    check alone is satisfied by either. Without a machine check, pointing
    `--target windows-arm64` at the x86-64 tree derives `win_amd64` from it and
    the ARM64 DLL gets shipped under it -- and `check_image` would not catch
    that, because the *artefact* is correct."""
    arm, amd = _build.TARGETS["windows-arm64"], _build.TARGETS["windows-x86_64"]
    if not (amd.python_root / "python313.dll").exists():
        return _skip("no Windows CPython distributions on this machine")
    saved = arm.python_root
    try:
        arm.python_root = amd.python_root
        try:
            arm._check_distribution()
        except SystemExit as exc:
            said = str(exc)
        else:
            raise AssertionError(
                "windows-arm64 accepted the x86-64 distribution")
        assert "aarch64" in said and "windows-arm64" in said, f"got: {said!r}"
    finally:
        arm.python_root = saved


def test_the_android_targets_refuse_the_other_abis_distribution():
    """`AndroidTarget._api_and_abi`'s MULTIARCH check, and the same trap as
    above: two Android CPython trees differ in nothing but their directory
    name, so without this an x86-64 extension gets tagged `arm64_v8a`."""
    arm = _build.TARGETS["android-arm64-v8a"]
    x86 = _build.TARGETS["android-x86_64"]
    if not arm.python_root.exists():
        return _skip("no Android CPython distribution on this machine")
    saved = x86.python_root
    try:
        x86.python_root = arm.python_root
        try:
            x86._api_and_abi()
        except SystemExit as exc:
            said = str(exc)
        else:
            raise AssertionError("android-x86_64 accepted the aarch64 tree")
        assert "x86_64" in said and "android-x86_64" in said, f"got: {said!r}"
    finally:
        x86.python_root = saved


def test_the_linux_loader_is_per_architecture_and_not_in_the_policy_list():
    """PEP 599's external-library table is architecture-independent; the
    dynamic loader is not in it at all, and its soname *is* per architecture.
    Keeping the loader inside `POLICY_LIBRARIES` would have let an aarch64
    wheel name `ld-linux-x86-64.so.2` and pass."""
    policy = _build.LinuxTarget.POLICY_LIBRARIES
    loaders = _build.LinuxTarget.LOADER
    assert not (set(loaders.values()) & policy), (
        "a loader soname is in POLICY_LIBRARIES, which makes it allowed for "
        "every architecture")
    linux_arm = _build.TARGETS["linux-aarch64"]
    linux_x86 = _build.TARGETS["linux-x86_64"]
    assert loaders["x86_64"] in linux_x86._policy_libraries()
    assert loaders["x86_64"] not in linux_arm._policy_libraries()
    assert loaders["aarch64"] in linux_arm._policy_libraries()


# --------------------------------------------------------------------------
# The refusal
# --------------------------------------------------------------------------

def test_the_refusing_target_is_still_a_choice():
    """A target that cannot be built here stays in `--target`'s choices. Being
    dropped from the registry would make it look like one nobody thought
    about, which is the state `--target linux-x86_64` was listed to avoid."""
    proc = subprocess.run([sys.executable, str(BUILD_PY), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    for key in _build.EXPECTED_TARGET_KEYS:
        assert key in proc.stdout, f"--target does not offer {key}"


def test_the_refusing_target_refuses_by_name_with_a_reason():
    """"Refuses" has to mean a sentence about this machine, not a missing-file
    error with a rebuild hint that cannot succeed. `main` checks `refusal`
    before the artefact for exactly that reason, so this must fail on the
    refusal even though no cross artefact exists either."""
    refusing = [t for t in _build.TARGETS.values() if t.refusal]
    assert refusing, "no target refuses; this file's premise has changed"
    for target in refusing:
        env = dict(os.environ, TORCHNATIVE_TARGET_PYTHON="/nonexistent")
        proc = subprocess.run(
            [sys.executable, str(BUILD_PY), "--target", target.key],
            capture_output=True, text=True, timeout=180, cwd=str(REPO), env=env)
        assert proc.returncode != 0, f"--target {target.key} did not refuse"
        said = proc.stdout + proc.stderr
        assert target.key in said, (
            f"the refusal does not name the target: {said[:400]!r}")
        assert "cannot be built on this machine" in said, (
            f"the refusal does not say it is about this machine: {said[:400]!r}")
        # Not a rebuild hint. That sentence means "it has not happened yet";
        # this one means "no command here can make it happen".
        assert "no cross-built extension at" not in said, (
            f"--target {target.key} answered with the missing-artefact message "
            "instead of its refusal -- the refusal is checked too late")


def test_the_refusal_names_what_is_missing_and_not_merely_that_it_failed():
    """docs/platform/WHEELMATRIX.md §3.3's measurements, kept next to the code that
    rests on them. A refusal whose reason is unfalsifiable is an excuse."""
    reason = _build._ANDROID_X86_64_REFUSAL
    for fragment in ("x86_64-linux-android", "python-build-standalone",
                     "verify_android.py", "arm64-v8a"):
        assert fragment in reason, (
            f"the android-x86_64 refusal does not mention {fragment!r}; a "
            "reader cannot check it")


# --------------------------------------------------------------------------
# The verifiers, which are where a wrong architecture passes silently
# --------------------------------------------------------------------------

def test_the_verifiers_know_every_architecture_the_registry_builds():
    """`verify_linux.py` and `verify_windows.py` select a target CPython to
    resolve symbols against. Both read one distribution by name until there
    were two of each, and the failure mode is a silent PASS: the export names
    of `libpython3.13.so` and `python3.dll` do not vary by architecture, so
    resolving against the wrong one succeeds.

    So the registry is the ground truth for what they must cover, and this
    fails when a target is added and a verifier is not taught about it.
    """
    sys.path.insert(0, str(REPO / "tools" / "wheel"))
    import verify_linux
    import verify_windows

    want_linux = {t.arch for t in _build.TARGETS.values()
                  if isinstance(t, _build.LinuxTarget)}
    assert want_linux <= set(verify_linux.ARCHES), (
        f"verify_linux.py covers {sorted(verify_linux.ARCHES)} but the "
        f"registry builds {sorted(want_linux)}")
    for arch, subdir in verify_linux.ARCHES.items():
        assert any(t.rust_target == subdir for t in _build.TARGETS.values()), (
            f"verify_linux.ARCHES[{arch!r}] points at {subdir!r}, which no "
            "registry entry builds")

    want_windows = {t.expected_tag for t in _build.TARGETS.values()
                    if isinstance(t, _build.WindowsTarget)}
    assert want_windows <= set(verify_windows.ARCHES), (
        f"verify_windows.py covers {sorted(verify_windows.ARCHES)} but the "
        f"registry builds {sorted(want_windows)}")


def test_a_manylinux_wheel_is_refused_rather_than_resolved_against_another_arch():
    """The silent-PASS case, driven. A tag this machine has no CPython for must
    stop the run, not fall back to whichever distribution is on hand."""
    sys.path.insert(0, str(REPO / "tools" / "wheel"))
    import verify_linux
    try:
        verify_linux.select_arch(
            pathlib.Path("torchnative-0.0.13a0-cp313-abi3-manylinux_2_17_s390x.whl"))
    except SystemExit as exc:
        said = str(exc)
    else:
        raise AssertionError("an s390x tag was accepted")
    assert "s390x" in said and "PASS" in said, (
        f"the refusal does not explain why falling back would be worse: {said!r}")


def test_selecting_an_arch_points_at_that_arch_and_not_the_default():
    sys.path.insert(0, str(REPO / "tools" / "wheel"))
    import verify_linux
    import verify_windows
    verify_linux.select_arch(
        pathlib.Path("t-0.0.13a0-cp313-abi3-manylinux_2_17_aarch64.whl"))
    assert verify_linux.ARCH == "aarch64"
    assert verify_linux.LINUX_PYTHON.name == "aarch64-unknown-linux-gnu"
    verify_windows.select_arch(pathlib.Path("t-0.0.13a0-cp313-abi3-win_arm64.whl"))
    assert verify_windows.ARCH == "aarch64"
    assert verify_windows.WINDOWS_PYTHON.name == "aarch64-pc-windows-msvc"


# --------------------------------------------------------------------------
# The builder's own self-test, and the document
# --------------------------------------------------------------------------

def test_build_pys_own_self_test_passes():
    """It builds nothing and takes a second, and it covers the Linux tag
    derivation against real ELF images -- including the per-architecture floor
    cases this file asserts from the other side."""
    proc = subprocess.run([sys.executable, str(BUILD_PY), "--self-test"],
                          capture_output=True, text=True, timeout=300,
                          cwd=str(REPO))
    assert proc.returncode == 0, (
        f"build.py --self-test failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}")
    assert "LINUX SELF-TEST: PASS" in proc.stdout


def test_the_document_exists_and_names_each_target_and_its_verdict():
    """docs/platform/WHEELMATRIX.md is where "built" and "executed" are kept apart. A
    target present in the registry and absent from the document is one whose
    claim nobody wrote down."""
    doc = REPO / "docs" / "platform" / "WHEELMATRIX.md"
    assert doc.exists(), "no docs/platform/WHEELMATRIX.md"
    text = doc.read_text()
    for key in _build.EXPECTED_TARGET_KEYS:
        assert key in text, f"docs/platform/WHEELMATRIX.md does not mention {key}"
    for tag in ("manylinux_2_17_aarch64", "win_arm64", "android_21_x86_64"):
        assert tag in text, f"docs/platform/WHEELMATRIX.md does not name the tag {tag}"


def test_the_readme_matrix_has_a_column_for_every_new_architecture():
    """The table is the claim a reader sees first, and it is the one that has
    gone stale before (four times, by this repository's own count)."""
    text = (REPO / "README.md").read_text()
    header = next((line for line in text.splitlines()
                   if "macOS<br>arm64" in line), None)
    assert header, "the README platform table's header row has moved"
    for column in ("Android<br>x86_64", "Linux<br>aarch64", "Windows<br>arm64"):
        assert column in header, (
            f"the README platform table has no {column!r} column")


def test_the_readme_does_not_call_a_never_executed_target_measured():
    """⚠️ means "builds and symbols resolve but nothing has computed here".
    Windows ARM64 has no machine anywhere in this project's reach, so its
    execution rows cannot be ✅ -- and this is the assertion that has to be
    edited, deliberately, on the day one runs."""
    text = (README := REPO / "README.md").read_text()
    rows = [line for line in text.splitlines()
            if line.startswith("| ") and "|" in line[2:]]
    header = next((r for r in rows if "macOS<br>arm64" in r), None)
    assert header, f"{README.name}: platform table header not found"
    columns = [c.strip() for c in header.split("|")]
    try:
        win_arm = columns.index("Windows<br>arm64")
    except ValueError:
        raise AssertionError("no Windows<br>arm64 column")
    for label in ("installs", "`import torch`", "computes"):
        row = next((r for r in rows if [c.strip() for c in r.split("|")][1] == label),
                   None)
        if row is None:
            continue
        cell = [c.strip() for c in row.split("|")][win_arm]
        assert "✅" not in cell, (
            f"the README claims Windows arm64 {label} = {cell!r}. Nothing has "
            "executed a win_arm64 wheel in this project; ✅ requires a run.")


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    if SKIPPED:
        print(f"     ({len(SKIPPED)} case(s) skipped, by name above -- a skip "
              "is not a pass)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
