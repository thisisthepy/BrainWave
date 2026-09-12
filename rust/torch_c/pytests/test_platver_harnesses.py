"""The runtime wheel harnesses, and the judgement they are only useful for.

`docs/platform/RELEASE_0_1_0b3.md` §6 grades a platform **reaches** when an
interpreter *for that platform* unpacked the published wheel into its own
site-packages and `torch.__file__` came back out of that site-packages. That one
assertion is the whole difference between "a torch answered" and "this wheel's
torch answered" -- a stray `PYTHONPATH`, or a working directory inside the
repository, is enough to make the second false while the first stays true.

Three harnesses now make it: `verify_android.py` on a device, `verify_ios_sim.py`
in a simulator, and `verify_wasm_browser.py` under Pyodide. The first two were
already here; this file exists because the third arrived with two traps the other
two structurally cannot have, and because one shared helper had already silently
blocked all of them.

  * **one interpreter, two modes.** A browser page is not two processes. The
    Android and iOS harnesses run their `bare` and `stubbed` probes as separate
    `python3.13` invocations, so a half-imported `torch` left behind by the first
    cannot be found in `sys.modules` by the second. Under Pyodide both modes run
    in the same interpreter, so the second answer is about the first run unless
    the modules are cleared. A test that only checked "the wasm run passed" would
    not see this: the *stubbed* run is the one the judgement is taken from, and it
    is the one that would be answered from cache.

  * **JSON through JavaScript.** The wasm result crosses a JS runtime, where
    `4.0` and `4` are the same number. `mm == [[4.0, ...]]` therefore stays true
    for a result that is no longer float32, so the dtype has to be carried and
    checked separately.

  * **`pkg_resources`.** `stage_dependencies` mapped the `setuptools` requirement
    to `["setuptools", "pkg_resources"]`, and setuptools stopped shipping
    `pkg_resources` in 82. With 84 in the spike venv the helper raised
    `SystemExit` naming *the wheel's METADATA* -- so a fact about the staging
    source read as a fact about the wheel, and it blocked the iOS harness too,
    which imports the same function. That is checked here by behaviour rather
    than by reading the source, because the next edit that reintroduces it will
    not spell it the same way.

What this file does NOT do is run a wheel. The runtime harnesses take minutes,
boot simulators and emulators, and reach the network; the gate runs on every
commit. The live case here is the cheapest possible one -- spawn the already-built
simulator launcher and ask it for `sys.platform` -- and it **skips by name** when
no simulator is booted, because a check that quietly passes when its subject is
absent is the shape `CLAUDE.md` §5.5 is about.
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[3]
WHEEL_TOOLS = REPO / "tools" / "wheel"
RELEASE_DOC = REPO / "docs" / "platform" / "RELEASE_0_1_0b3.md"
WASM_HARNESS = WHEEL_TOOLS / "verify_wasm_browser.py"
IOS_SCRATCH = pathlib.Path("/Volumes/macMini/caches/ios-wheel-check")

SKIPPED = []


def _skip(why):
    SKIPPED.append(why)
    print(f"     SKIP {why}")


def test_the_wasm_harness_exists_and_parses():
    assert WASM_HARNESS.is_file(), (
        f"{WASM_HARNESS} is gone. RELEASE_0_1_0b3.md §6 grades WASM *reaches* on "
        "the strength of it; without the script the grade has no harness behind it")
    compile(WASM_HARNESS.read_text(), str(WASM_HARNESS), "exec")


def test_the_wasm_harness_keeps_the_load_bearing_judgement():
    """`torch.__file__` out of site-packages, on an Emscripten interpreter.

    Stated as three separate required fragments rather than one, because each one
    removed alone still leaves a script that runs and prints PASS.
    """
    source = WASM_HARNESS.read_text()
    for fragment, why in [
        ('not plain["torch_file"].startswith(site + "/")',
         "torch.__file__ is no longer required to come out of site-packages -- the "
         "run would pass against any torch the interpreter could find"),
        ('plain["platform"] != "emscripten"',
         "the harness no longer checks it ran on an Emscripten interpreter"),
        ('plain["mm"] != [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]',
         "no arithmetic is checked, so the grade would rest on import alone"),
        ('plain["mm_dtype"] != "torch.float32"',
         "the dtype is unchecked, and JSON through JavaScript has already made "
         "the value comparison blind to it"),
    ]:
        assert fragment in source, f"{WASM_HARNESS.name}: {why}"


def test_the_wasm_probe_clears_modules_between_its_two_modes():
    """Both modes share one interpreter; without this the second reads the first.

    Checked on the probe *source string* the page executes, not on the file as a
    whole, so a clearing loop that lives somewhere the probe never runs does not
    satisfy it.
    """
    sys.path.insert(0, str(WHEEL_TOOLS))
    try:
        import verify_wasm_browser as harness
    finally:
        sys.path.pop(0)
    probe = harness.PROBE
    assert "del sys.modules[" in probe, (
        "verify_wasm_browser.PROBE no longer clears sys.modules. Its `bare` and "
        "`stubbed` runs share one Pyodide interpreter, so the stubbed run -- the "
        "one the judgement is taken from -- would be answered out of the cache "
        "the failed bare import left behind")
    for name in ("torch", "_multiprocessing", "_posixshmem"):
        assert f'"{name}"' in probe or f"'{name}'" in probe, (
            f"the probe's module purge does not mention {name!r}; it was one of "
            "the three things the bare run leaves behind")


def _fake_wheel(path, requires):
    with zipfile.ZipFile(path, "w") as zf:
        meta = "Metadata-Version: 2.1\nName: fake\nVersion: 0\n" + "".join(
            f"Requires-Dist: {r}\n" for r in requires)
        zf.writestr("fake-0.dist-info/METADATA", meta)


def test_staging_does_not_demand_a_pkg_resources_setuptools_no_longer_ships():
    """setuptools >= 82 has no `pkg_resources`, and this used to be fatal.

    Behavioural: a staging source with `setuptools` and no `pkg_resources` is
    exactly the spike venv's shape today, and it must stage rather than raise.
    """
    sys.path.insert(0, str(WHEEL_TOOLS))
    try:
        import verify_android as harness
    finally:
        sys.path.pop(0)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        source = tmp / "site-packages"
        for name in ("packaging", "yaml", "setuptools"):
            (source / name).mkdir(parents=True)
            (source / name / "__init__.py").write_text("")
        wheel = tmp / "fake.whl"
        _fake_wheel(wheel, ["setuptools>=77.0.3"])
        into = tmp / "into"
        into.mkdir()

        original = harness.SPIKE_SITE
        harness.SPIKE_SITE = source
        try:
            staged = harness.stage_dependencies(into, wheel)
        except SystemExit as exc:
            raise AssertionError(
                "stage_dependencies refused a staging source that has setuptools "
                f"but no pkg_resources -- which is every setuptools >= 82: {exc}")
        finally:
            harness.SPIKE_SITE = original

        assert "setuptools" in staged, staged
        assert (into / "setuptools").is_dir()
        assert not (into / "pkg_resources").exists(), (
            "pkg_resources was staged from a source that does not have it")


def test_staging_still_refuses_a_dependency_that_is_genuinely_missing():
    """The relaxation above must not have turned the guard off wholesale.

    `pkg_resources` became conditional; a requirement the source simply does not
    have must still stop the run before a device is touched, because the failure
    would otherwise surface inside `import torch` and read as a wheel defect.
    """
    sys.path.insert(0, str(WHEEL_TOOLS))
    try:
        import verify_android as harness
    finally:
        sys.path.pop(0)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        source = tmp / "site-packages"
        for name in ("packaging", "yaml"):
            (source / name).mkdir(parents=True)
            (source / name / "__init__.py").write_text("")
        wheel = tmp / "fake.whl"
        _fake_wheel(wheel, ["filelock"])
        into = tmp / "into"
        into.mkdir()

        original = harness.SPIKE_SITE
        harness.SPIKE_SITE = source
        try:
            harness.stage_dependencies(into, wheel)
        except SystemExit:
            pass
        else:
            raise AssertionError(
                "stage_dependencies accepted a wheel requiring filelock from a "
                "source with no filelock -- the device would then import a torch "
                "missing a dependency and the failure would look like a wheel "
                "defect")
        finally:
            harness.SPIKE_SITE = original


def _release_rows():
    rows = []
    for line in RELEASE_DOC.read_text().splitlines():
        if not line.startswith("| ") or "|---" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3:
            rows.append(cells)
    return rows


def test_every_reaches_row_names_a_harness_that_exists():
    """A grade is only as good as the script behind it.

    `reaches` means an interpreter for that platform ran the wheel, which means
    some harness did it. If the row names a `verify_*.py` that is not in the
    tree, the claim has nothing standing under it.
    """
    rows = _release_rows()
    assert rows, f"{RELEASE_DOC.name}: found no platform table rows at all"
    reaching = [r for r in rows if "**reaches**" in r[1]]
    assert len(reaching) >= 4, (
        f"{RELEASE_DOC.name} §6 grades only {len(reaching)} platform(s) as "
        "reaches; macOS, iOS simulator, Android and WASM all did")
    for platform, _grade, detail in reaching:
        named = [w for w in detail.replace("`", " ").split()
                 if w.startswith("verify") and w.endswith(".py")]
        assert named, (
            f"{RELEASE_DOC.name} §6: the {platform!r} row is graded reaches but "
            "names no verify_*.py -- nobody can reproduce or re-check it")
        for script in named:
            assert (WHEEL_TOOLS / script).is_file(), (
                f"{RELEASE_DOC.name} §6: the {platform!r} row names "
                f"{script}, which is not in tools/wheel/")


def test_no_platform_is_graded_agrees():
    """`agrees` is comparison against a reference, and nothing here does that.

    The harnesses check `aten.mm.default` against a constant written in the
    harness. That is arithmetic executing; it is not agreement with upstream
    torch, and the release doc must not say it is.
    """
    for platform, grade, _detail in _release_rows():
        assert "**agrees**" not in grade, (
            f"{RELEASE_DOC.name} §6 grades {platform!r} as agrees. No wheel "
            "harness in tools/wheel/ compares against a reference torch; the "
            "strongest thing any of them does is check a hardcoded 3x2 of 4.0")


def test_ios_device_is_still_recorded_as_never_executed():
    """The one row that must not quietly improve.

    No iPhone has ever run this project's wheel. The device artefact cannot be
    run in a simulator or on macOS -- dyld refuses both -- so nothing short of
    attached hardware can change this, and a doc edit that upgrades it without
    one would be the most expensive kind of false claim here.
    """
    text = RELEASE_DOC.read_text()
    row = next((r for r in _release_rows() if "iOS device" in r[0]), None)
    assert row is not None, f"{RELEASE_DOC.name} §6: no iOS device row"
    assert "**builds**" in row[1], (
        f"{RELEASE_DOC.name} §6 grades the iOS device wheel {row[1]!r}. "
        "Nothing has executed it, on any release")
    assert "never executed" in text, (
        f"{RELEASE_DOC.name} no longer says the iOS device wheel has never been "
        "executed. It still has not been")


def test_a_booted_simulator_runs_the_harness_launcher():
    """The live case, and it skips by name rather than passing when absent.

    Cheapest possible: the launcher `verify_ios_sim.py` builds is a real
    `python3.13` for the simulator platform, so asking it for `sys.platform`
    costs a second and still proves the road the whole harness rides on -- a
    simulator process running an interpreter that reports `ios`.
    """
    launcher = IOS_SCRATCH / "python3.13"
    prefix = IOS_SCRATCH / "prefix"
    if not launcher.exists() or not prefix.exists():
        _skip("test_a_booted_simulator_runs_the_harness_launcher: no staged "
              f"simulator prefix at {IOS_SCRATCH} (run tools/wheel/verify_ios_sim.py)")
        return
    listed = subprocess.run(["xcrun", "simctl", "list", "devices", "booted",
                             "--json"], capture_output=True, text=True)
    if listed.returncode != 0:
        _skip("test_a_booted_simulator_runs_the_harness_launcher: xcrun simctl "
              f"failed: {listed.stderr.strip()[:200]}")
        return
    booted = [d["udid"] for ds in json.loads(listed.stdout)["devices"].values()
              for d in ds]
    if not booted:
        _skip("test_a_booted_simulator_runs_the_harness_launcher: no simulator is "
              "booted (this test does not boot one -- simulators are shared)")
        return

    environment = dict(**{k: v for k, v in __import__("os").environ.items()})
    environment["SIMCTL_CHILD_PYTHONHOME"] = str(prefix)
    proc = subprocess.run(
        ["xcrun", "simctl", "spawn", booted[0], str(launcher), "-s", "-P",
         "-c", "import sys; print('BW_PLATFORM ' + sys.platform)"],
        capture_output=True, text=True, env=environment, cwd=tempfile.gettempdir())
    output = proc.stdout + proc.stderr
    assert "BW_PLATFORM ios" in output, (
        "the simulator launcher did not report sys.platform == 'ios':\n"
        + output.strip()[-2000:])


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
