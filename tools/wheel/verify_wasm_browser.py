#!/usr/bin/env python3
"""Install the Pyodide wheel into a Pyodide interpreter and prove its torch computes.

    python tools/wheel/verify_wasm_browser.py dist/torchnative-*pyemscripten*.whl

`tools/wheel/verify_cross.py` stops at the artefact for this wheel and says so: it
establishes that the members are wasm32 side modules exporting `PyInit__C`, and
explicitly declines to claim that anything loads, imports or computes. This script
makes the same judgement the Android and iOS-simulator harnesses make -- *`torch.__file__`
must come out of the install location* -- inside a real Pyodide interpreter.

Why a browser rather than node
------------------------------

Pyodide's non-browser runner is node, and node is not installed on this machine
(2026-09-13; `node`, `npm`, `deno` and `bun` are all absent). A Pyodide build is
Emscripten output: `pyodide.asm.wasm` plus JS glue, so a bare wasm runtime
(`wasmtime`, `wasmer`) cannot run it and macOS's `jsc` lacks the host functions the
glue calls. What *is* present is Safari, and a browser is the environment Pyodide is
primarily built for -- so the road taken here is a page served over loopback that
loads the local Pyodide distribution, stages the wheel, runs the probe, and POSTs one
JSON object back to the server that served it. `safaridriver` is deliberately not
used: enabling Safari's remote automation is an interactive, administrator-authorised
step, and this needs none of it.

What "installed" means, precisely, because it is not `pip install`
------------------------------------------------------------------

Identical to `verify_android.py`, by importing its functions rather than restating
them: the archive is unpacked into the interpreter's **site-packages** (not onto a
`sys.path` entry that could shadow a broken install), `.data/purelib/` is relocated so
`importlib.metadata` sees `torch-<v>.dist-info`, and the pure-Python distributions the
wheel's own `Requires-Dist` names are staged beside it. `micropip` is not used --
the local distribution is `pyodide-core`, which does not ship it, and resolving through
it would reach the network for versions nobody chose.

The staged tree is shipped to the browser as one `.tar.gz` and unpacked inside the
Pyodide filesystem, because ~2,700 separate `fetch` calls over loopback is the
difference between a minute and an hour.

What this cannot answer
-----------------------

Nothing here measures throughput, and the tag's `2026_0` ABI half is still only
checked against the Pyodide distribution on this machine, exactly as
`verify_cross.py` says -- no wasm module records it. The local distribution is
CPython 3.14; the wheel is `cp313-abi3`, so what runs here is the abi3 forward
compatibility path, which is the one a Pyodide user gets but is not the same as
a cp313 Pyodide.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import queue
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_android import unpack, stage_dependencies  # noqa: E402

PYODIDE = Path(os.environ.get("PYODIDE_DIST", "/Volumes/macMini/caches/pyodide/pyodide"))
PORT = int(os.environ.get("WASM_PROBE_PORT", "8731"))

# Runs inside Pyodide. Mirrors verify_android.PROBE's shape; `_multiprocessing` is
# absent there too, so the same stub question is asked and both answers reported.
PROBE = r'''
import json, os, sys, traceback, types

def install_stubs():
    def unavailable(*a, **k):
        raise OSError("shared memory is unavailable under Emscripten")
    sys.modules["_multiprocessing"] = types.ModuleType("_multiprocessing")
    shm = types.ModuleType("_posixshmem")
    shm.shm_unlink = unavailable
    shm.shm_open = unavailable
    sys.modules["_posixshmem"] = shm

# Unlike the Android and iOS harnesses, the two modes share ONE interpreter --
# a browser page is not two processes. A half-imported torch left behind by the
# bare run, or a stub left behind by the stubbed one, would make the second
# answer be about the first run. Both are cleared here.
for _name in [n for n in sys.modules
              if n == "torch" or n.startswith("torch.")
              or n in ("_multiprocessing", "_posixshmem")]:
    del sys.modules[_name]

out = {"stubbed": MODE == "stubbed", "sys_path": sys.path,
       "platform": sys.platform, "py": sys.version.split()[0]}
if out["stubbed"]:
    install_stubs()
try:
    import torch
    out["torch_file"] = torch.__file__
    out["torch_version"] = torch.__version__
    out["C_file"] = sys.modules["torch._C"].__file__
    out["C_names"] = len(dir(sys.modules["torch._C"]))
    out["aten_ops"] = len(dir(torch.ops.aten))
    x = torch.ones(3, 4)
    y = torch.ones(4, 2)
    _mm = torch.ops.aten.mm.default(x, y)
    out["mm"] = _mm.tolist()
    # The result crosses JSON through JavaScript, where 4.0 and 4 are one number.
    # The dtype is carried separately so that "the arithmetic agreed" does not have
    # to rest on a distinction the transport cannot keep.
    out["mm_dtype"] = str(_mm.dtype)
    out["add"] = (x + x).tolist()[0]
    import torch.nn as nn
    m = nn.Linear(4, 3)
    out["linear_shape"] = list(m(torch.ones(2, 4)).shape)
    out["linear_dtype"] = str(m(torch.ones(2, 4)).dtype)
    import importlib.metadata as md
    try:
        out["metadata_version_torch"] = md.version("torch")
    except Exception as exc:
        out["metadata_version_torch"] = "<%s: %s>" % (type(exc).__name__, exc)
    out["ok"] = True
except BaseException as exc:
    out["ok"] = False
    out["error"] = "%s: %s" % (type(exc).__name__, exc)
    out["traceback"] = traceback.format_exc().splitlines()[-8:]
RESULT = json.dumps(out)
'''

PAGE = r'''<!doctype html><html><head><meta charset=utf-8><title>torchnative wasm probe</title>
</head><body><pre id=log>starting...
</pre><script type=module>
const log = (m) => { document.getElementById('log').textContent += m + "\n"; };
const post = (o) => fetch('/result', {method:'POST', body: JSON.stringify(o)});
try {
  const { loadPyodide } = await import('/pyodide/pyodide.mjs');
  log('loading pyodide');
  const py = await loadPyodide({ indexURL: '/pyodide/' });
  log('pyodide ' + py.version + ' python ' + py.runPython('import sys; sys.version'));
  log('fetching staged tree');
  const buf = new Uint8Array(await (await fetch('/site.tar.gz')).arrayBuffer());
  py.FS.writeFile('/site.tar.gz', buf);
  log('unpacking ' + buf.length + ' bytes');
  py.runPython(`
import sys, tarfile, os
site = [p for p in sys.path if p.endswith('site-packages')][0]
with tarfile.open('/site.tar.gz') as tf:
    tf.extractall(site)
os.remove('/site.tar.gz')
`);
  const results = {};
  for (const mode of ['bare', 'stubbed']) {
    log('probe: ' + mode);
    py.globals.set('MODE', mode);
    py.runPython(PROBE_SRC);
    results[mode] = JSON.parse(py.globals.get('RESULT'));
    log('  ok=' + results[mode].ok + (results[mode].ok ? '' : ' ' + results[mode].error));
  }
  await post({stage: 'done', results, pyodide: py.version});
  log('reported');
} catch (e) {
  await post({stage: 'error', error: String(e), stack: String(e && e.stack)});
  log('ERROR ' + e);
}
</script></body></html>'''


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wheel", type=Path)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    if not args.wheel.exists():
        sys.exit(f"no such wheel: {args.wheel}")
    if "pyemscripten" not in args.wheel.name:
        sys.exit(f"{args.wheel.name} is not a Pyodide wheel. The other platforms are "
                 "separate artefacts and this harness cannot run them.")
    if not PYODIDE.exists():
        sys.exit(f"no Pyodide distribution at {PYODIDE}; set PYODIDE_DIST")

    root = Path(tempfile.mkdtemp(prefix="wasmprobe-"))
    try:
        _run(args, root)
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


def _run(args, root: Path) -> None:
    staging = root / "site-packages"
    staging.mkdir()
    print(f"+ unpacking {args.wheel.name} into a staging site-packages")
    unpack(args.wheel, staging)
    deps = stage_dependencies(staging, args.wheel)
    print(f"  + {len(deps)} dependencies: {', '.join(deps)}")

    # Nothing host-native may reach the interpreter; a Mach-O that arrived with a
    # dependency would fail in a way that reads like a wheel defect.
    allowed = {"_C.abi3.so", "libtorch_global_deps.so"}
    strays = [p for p in staging.rglob("*")
              if p.suffix in (".so", ".dylib", ".pyd") and p.name not in allowed]
    if strays:
        sys.exit(f"refusing to stage {len(strays)} host-native artefact(s): "
                 f"{[str(p) for p in strays[:5]]}")

    tarball = root / "site.tar.gz"
    print("+ packing the staged tree")
    with tarfile.open(tarball, "w:gz", compresslevel=1) as tf:
        for entry in sorted(staging.iterdir()):
            tf.add(entry, arcname=entry.name)
    print(f"  {tarball.stat().st_size / 1e6:.0f} MB")

    (root / "index.html").write_text(
        PAGE.replace("PROBE_SRC", json.dumps(PROBE)))
    os.symlink(PYODIDE, root / "pyodide")

    answers: queue.Queue = queue.Queue()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
            answers.put(json.loads(body))

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}/index.html"
    print(f"+ serving {root} at {url}")
    print("+ opening Safari (node is not installed; see the module docstring)")
    subprocess.run(["open", "-a", "Safari", url], check=True)

    try:
        answer = answers.get(timeout=args.timeout)
    except queue.Empty:
        sys.exit(f"no result after {args.timeout:.0f}s -- the page never reported. "
                 "Is a Safari window open and allowed to run scripts?")
    finally:
        server.shutdown()

    if answer.get("stage") != "done":
        print(json.dumps(answer, indent=2))
        sys.exit("the page failed before the probe ran")

    results = answer["results"]
    plain = results["stubbed"]
    print(f"\n+ pyodide {answer['pyodide']}")
    for label in ("bare", "stubbed"):
        r = results[label]
        print(f"\n+ {label}")
        if r["ok"]:
            print(f"  torch_file   {r['torch_file']}")
            print(f"  C_file       {r['C_file']}")
            print(f"  C_names      {r['C_names']}   aten_ops {r['aten_ops']}")
            print(f"  aten.mm      {r['mm']}  ({r['mm_dtype']})")
            print(f"  x + x        {r['add']}")
            print(f"  nn.Linear    {r['linear_shape']} {r['linear_dtype']}")
            print(f"  metadata     torch {r['metadata_version_torch']}")
        else:
            print(f"  {r['error']}")
            for frame in r.get("traceback", []):
                print(f"    {frame}")

    problems: list[str] = []
    if not plain["ok"]:
        problems.append(f"import torch failed under Pyodide: {plain.get('error')}")
    else:
        site = next(p for p in plain["sys_path"] if p.endswith("site-packages"))
        if not plain["torch_file"].startswith(site + "/"):
            problems.append(
                f"torch.__file__ is {plain['torch_file']}, not under {site} -- "
                "Pyodide imported something that did not come from the wheel")
        if not plain["C_file"].startswith(site + "/"):
            problems.append(f"torch._C came from {plain['C_file']}")
        if not plain["C_file"].endswith("_C.abi3.so"):
            problems.append(f"torch._C is {plain['C_file']}, not our extension")
        if plain["mm"] != [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]:
            problems.append(f"aten.mm.default gave {plain['mm']}")
        if plain["mm_dtype"] != "torch.float32":
            problems.append(f"aten.mm.default produced {plain['mm_dtype']}, not "
                            "torch.float32 -- the values matched a float expectation "
                            "that JSON had already flattened to integers")
        if plain["platform"] != "emscripten":
            problems.append(f"sys.platform is {plain['platform']!r} -- this did not "
                            "run on an Emscripten interpreter")

    print()
    bare = results["bare"]
    print("  _multiprocessing stub  "
          + ("still required -- bare run: " + str(bare.get("error"))
             if not bare["ok"] else
             "NOT required -- the bare run imported torch"))

    if problems:
        print()
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        sys.exit(1)
    print()
    print(f"PASS -- {args.wheel.name} unpacks into a Pyodide interpreter's "
          "site-packages and its torch computes in the browser")
    print("        The tag's 2026_0 ABI half is still only checked against the local "
          "Pyodide distribution; no wasm module records it.")


if __name__ == "__main__":
    main()
