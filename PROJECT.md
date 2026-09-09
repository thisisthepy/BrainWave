# Packaging decisions

Why `pyproject.toml`, `setup.py` and the wheel layout are the way they are.

Every entry below is a mistake this project actually made and corrected. They
lived as comments inside `pyproject.toml` until that file was two thirds prose;
they are here so the file can be read as configuration and the reasoning can be
read as prose. **If you are about to change one of these fields, read its
section first** — several of them have been changed to the "obvious" value
before, and the section says what broke.

Related: [`docs/design/DESIGN.md`](docs/design/DESIGN.md) for what this project
is, [`docs/platform/WHEEL.md`](docs/platform/WHEEL.md) for how a wheel is built,
[`docs/design/ABI3.md`](docs/design/ABI3.md) for the stable-ABI decision.

---

## `setup.py` exists, and deleting it breaks the wheel tag

Everything declarative is in `pyproject.toml`. What is left in `setup.py` is the
pair of facts that decide the **wheel tag**, neither of which has a
`[tool.setuptools]` spelling:

1. **`has_ext_modules()` must answer `True`.** This distribution is not pure
   Python — it carries `torch/_C.abi3.so` — but setuptools cannot see that,
   because the extension is *pre-built* by `vendor/install_shim.sh` and arrives
   as package data rather than as an `Extension()` setuptools compiled itself.
   Left alone, `Distribution.is_pure()` answers `True` and the wheel goes out
   tagged `py3-none-any`: installable on Android, on iOS, on any machine at all,
   and functional on none of them, because the `.so` inside is Mach-O arm64.
   **`0.0.1a0` on PyPI is that wheel**; every release from `0.0.2a0` is tagged
   correctly, which is `setup.py` doing its job.
2. **`py_limited_api = "cp313"`.** This is a `bdist_wheel` *command option*, not
   project metadata, so it has to be passed through `options=`. It turns the tag
   from `cp313-cp313-<plat>` into `cp313-abi3-<plat>`.

Both are load-bearing for the tag and nothing else. If the file were deleted the
wheel would still build, and would be wrong in both directions.

`setup.py` as a *configuration* file is not deprecated. What is deprecated is
invoking `python setup.py <command>` directly; this project uses PEP 517
(`build-backend = "setuptools.build_meta"`), under which `setup.py` is read as
configuration and nothing else.

**Hatchling was considered and rejected.** Neither fact is declarative there
either: `pure_python` and `infer_tag` are *build data* that only a build hook
can set, and there is no `py_limited_api` option at all — the ABI tag would have
to be assembled by hand in a hook. That trades 56 lines (mostly comment) for a
hook that owns tag computation, on the one surface that has already gone
silently wrong across seven wheels. **maturin** does not fit for a different
reason: it assumes it builds the Rust extension itself, and here `cargo` builds
it separately and it is injected into a vendored upstream tree.

---

## `license` is upstream's expression, and cannot say which term is ours

```toml
license = "Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT"
```

That is torch 2.13.0's own `License-Expression`, **verbatim**. A platform wheel
carries the upstream Python tree, so declaring our licence alone would describe
a few thousand lines of this distribution and misdescribe two million.

**This field cannot express which term is ours.** Upstream's expression already
contains Apache-2.0 *and* MIT, so ours is indistinguishable inside it, and
changing our own licence does not change one character of this line. A comment
that stood here for a long time claimed MIT was "ours" while the rest was
upstream's — that was never true.

The upstream licence texts ride along: `tools/wheel/build.py` injects torch's
`dist-info`, third-party notices included. This line is about the metadata being
true, not about supplying attribution that was missing.

---

## `dependencies` — why `torch` is absent, and why upstream's deps are not

**`torch` is deliberately not listed.** This distribution *provides* `torch` —
the upstream Python tree with `_C` replaced ([`docs/design/DESIGN.md`](docs/design/DESIGN.md) §2)
— rather than consuming somebody else's, so requiring it would be a package
declaring a dependency on itself.

Two earlier attempts at that line were wrong in **opposite** directions:

- **Empty**, with a module-scope `from torch import nn`: installed cleanly and
  then failed on import.
- **Requiring `torch` unconditionally**: unresolvable on Android and iOS, where
  upstream publishes no wheel at all — that is, on exactly the platforms this
  project exists for.

The mistake underneath both was reading upstream torch as a runtime dependency
when it is a **comparison baseline**: the golden harness needs one installed to
diff against, and nothing at runtime does. It follows that this cannot coexist
with an installed PyTorch once the wheels carry the tree — not a defect to route
around, but a consequence of there being one `torch` on an interpreter and this
being a build of it.

**What *is* listed** is upstream torch's own pure-Python dependency set, copied
from the `Requires-Dist` lines of the vendored `torch-2.13.0.dist-info`.
Shipping the tree means inheriting them: `torch/__init__.py:35` imports
`typing_extensions` before it reaches anything of ours, so an empty list is not
a smaller promise — it is a wheel that installs and then raises
`ModuleNotFoundError`, which is what the first platform wheel built here did.

**Upstream's CUDA and triton requirements are not copied.** They are all guarded
`platform_system == "Linux"`, and this build has no CUDA path at all; carrying
them would drag ~2 GB of nvidia wheels onto every Linux install, including the
aarch64 boards that are the point of the exercise.

---

## `classifiers`

**`Development Status :: 2 - Pre-Alpha`** is on purpose. The Python surface is
still a skeleton; the working part of this project is the `torch._C` replacement
under `rust/torch_c`, built into the platform wheels
([`docs/platform/WHEEL.md`](docs/platform/WHEEL.md)).

**`Python :: 3.13` is the abi3 FLOOR, not a ceiling.** One `cp313-abi3` wheel
installs on 3.13 and on every later CPython — that is what `setup.py`'s
`py_limited_api` buys, and building a `cp314-abi3` wheel beside it would
*narrow* support rather than widen it. 3.14 and 3.15 are advertised on a
measurement, not on the argument: the `0.0.12a0` macOS arm64 wheel installs and
computes on 3.14.7 and on 3.15.0rc1 — same wheel, `a @ b` sums to 134.0 on both.
`test_release.py::test_the_abi3_wheel_loads_on_later_cpythons` keeps it honest
by loading the built extension under every newer `python3.N` on PATH, and
skipping *by name* when there is none.

**`Operating System :: POSIX :: Linux` and `:: Microsoft :: Windows`** are
declared because the wheels exist and pip will hand them to those users;
shipping a `manylinux_2_17_x86_64` and a `win_amd64` wheel while withholding the
classifier would be the more misleading option, since the platform tag already
made the claim. What they have is CI execution against the *published* wheel,
not local execution — the README's platform table is the precise record.

---

## `[project.optional-dependencies]` — the three backend extras install the same wheel, except npu

```toml
cpu = []
gpu = []
npu = [
    "openvino; (sys_platform == 'win32' and platform_machine == 'AMD64') or (sys_platform == 'linux' and platform_machine == 'x86_64')",
]
```

**`cpu` and `gpu` are empty deliberately, not accidentally**, and the shape is
forced by packaging rather than chosen
([`docs/platform/WHEEL.md`](docs/platform/WHEEL.md) §13). A wheel filename
carries no backend axis — pip selects on the python, abi and platform tags
alone, and the `build tag` slot it does have is a tie-breaker it never selects
on — so CPU, GPU and NPU builds cannot sit side by side under one platform
tag. Upstream torch hits exactly this and answers it by leaving PyPI: its 24
files for one version are 4 platforms × 6 pythons, and the CUDA builds live on
a separate index.

So one binary carries all three and chooses at runtime. That is possible because
the accelerator paths are **dlopened rather than linked**:
[`docs/devices/VULKAN.md`](docs/devices/VULKAN.md) measured `libvulkan` absent
from `NEEDED`, so a device with no driver loses the GPU path and not
`import torch`; NNAPI and CoreML compile at runtime with the same property.

An extra cannot change what is in a wheel — only add dependencies. What belongs
here is whatever a backend needs on the *Python* side at runtime. Nothing does
for CPU or GPU, so those two stay empty; the names are declared so
`torchnative[gpu]` resolves rather than errors, and so eventual contents have
somewhere to land.

**`npu` is no longer empty.** The Intel NPU path
(`torchnative/export/intelnpu.py`) reaches the device through OpenVINO's C
API over `ctypes`, loaded from a shared library that has to come from
somewhere — previously the user's own system-wide OpenVINO install, found by
hand and put on `PATH` or named in `TORCHNATIVE_OPENVINO_C`. `pip install
openvino` ships the *entire* runtime inside the Python package (`openvino_c`
plus every plugin, including `openvino_intel_npu_plugin`), so it is exactly
the kind of Python-side runtime dependency this section describes, not a
wheel-shape workaround. `load_openvino_c` finds and loads it automatically
once installed; an explicit path or `TORCHNATIVE_OPENVINO_C` still wins over
it, so naming a library by hand is unaffected. The marker keeps it off
platforms with no Intel NPU to reach — macOS and non-x86-64 hosts — matching
the refusal `library_candidates` already makes by platform name.

**`federated = []`** exists so that using adaptation alone does not pull in the
federated stack ([`docs/design/DESIGN.md`](docs/design/DESIGN.md) §10).

**`test = ["torch>=2.13,<2.14"]`** is upstream PyTorch as the thing the golden
harness diffs against — not a runtime dependency. Pinned to what is actually
compared: this shim implements one release's `_C` surface, so a tree from
another release expects different symbols from it. Installing it alongside the
wheels that carry our own tree will conflict, which is why it is an extra.

---

## `[tool.setuptools]` — the source-set layout has to be spelled out

```toml
package-dir = { "" = "torchnative/src/main" }
```

Without this, setuptools auto-discovery treats `src` as the root and ships
`main/` and `test/` as importable packages — which it did, so a wheel built
before this section answered `import main.torchnative` and not
`import torchnative`.

### `packages.find` includes `torch`, and that is the whole point

`torch` **is** shipped from this distribution. An earlier revision excluded it on
the grounds that writing into another distribution's package breaks its
uninstall — true, but that describes a *graft onto somebody else's torch*, which
is not what `src/main/torch` is. `vendor/vendor_torch.sh` assembles the whole
upstream Python tree there and `vendor/install_shim.sh` puts our `_C` in the
hole it leaves. **Excluding it is what produced the `py3-none-any` distribution
on PyPI**, which installs and then cannot `import torch`.

The tree is not in git (see `.gitignore`), so this has nothing to find until
those two scripts have run. `tools/wheel/build.py` refuses to build in that
state rather than quietly emitting the empty shell again.

**Three upstream packages, not one.** `torch-<v>.dist-info/top_level.txt` names
`functorch`, `torch` and `torchgen`, and `import torch` reaches the second of
those (`torch/utils/_python_dispatch.py:13`) 2254 lines into
`torch/__init__.py`. Shipping only `torch` looked fine for a long time because
the `PYTHONPATH` workflow shadows site-packages for `torch` alone and let the
other two resolve to the reference installation underneath.

### `package-data` is a recursive catch-all on purpose

Everything under the package roots that is not a `.py` module: the `_C`
extension, upstream's `.pyi` stubs (note `torch/_C/` is a *directory* beside the
`torch/_C.abi3.so` extension — importlib resolves the extension first, and
upstream ships exactly this shape), `py.typed`, the inductor Jinja templates and
linker script, the CMake config, and the `_dynamo/graph_break_registry.json`
read at runtime.

A per-extension list was tried first and **silently dropped files**, so this is a
recursive catch-all with an exclusion list instead. `torchgen` in particular is
mostly *not* Python: `torchgen/packaged/ATen/native/native_functions.yaml` and
the rest of `packaged/` are data files a per-extension list would have to
enumerate.

### `exclude-package-data`

Byte-code caches are build droppings — running the test suite in the source tree
leaves ~400 of them under the vendored tree — and `.pyc` compiled by the
*building* interpreter would pin the wheel to that interpreter's magic number,
which is the one thing an abi3 wheel must not do.

**`.DS_Store` is listed because one got into a wheel.** Finder writes it into any
directory it displays, `.gitignore` cannot help (the vendored tree is not in
git), and `build.py` skipped it only on the path that copies the tree, not on
the one setuptools takes. The copy at `torchnative/src/main/.DS_Store` is the
package root, so it landed at the archive root. It was caught by the iOS check
comparing two wheels member by member: 2,566 shared, 1 differing.

---

## `[tool.ppp]`

Platform source sets follow the pypackpack layout: `src/main` is scanned for
top-level Python packages, which is what lets one package provide both `torch`
and `torchnative`. See [`docs/design/DESIGN.md`](docs/design/DESIGN.md) §10.
