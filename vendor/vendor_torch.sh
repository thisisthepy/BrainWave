#!/bin/sh
# Vendor the upstream PyTorch *Python* tree, and nothing else.
#
# "The tree" is the three top-level packages the torch distribution installs --
# `torch`, `torchgen`, `functorch` -- not just `torch`. See the `siblings` loop
# below for why copying only the first one hid a missing dependency for months.
#
# DESIGN.md §2 is the bet this script exists to test: "파이썬 계층은 벤더링하고
# `_C` 만 교체한다". That sentence is only meaningful if the vendored tree has
# exactly one hole in it, so the copy is defined by what it *drops*:
#
#   torch/lib/         353 MB of libtorch/libc10 dylibs -- the native runtime
#   torch/include/      61 MB of C++ headers, build inputs only
#   torch/bin/           7 MB of host tools
#   torch/test/          upstream's own test binaries
#   *.so *.dylib *.a     every compiled artefact anywhere in the tree
#
# In torch 2.13.0 that last line removes exactly one file from the Python tree:
# `_C.cpython-313-darwin.so`. Everything else under `torch/` is Python source.
# That is the measurement, not an assumption -- `find` it yourself if in doubt.
#
# The result is a torch that cannot import until `install_shim.sh` puts our
# `_C` in the hole, which is the property we want: if the tree could import
# without it, the experiment would be proving nothing.
#
# The tree is NOT committed (see /.gitignore). It is ~1084 modules and tens of
# megabytes of third-party source; this script is the reproduction.
set -eu

repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
dest=${TORCHNATIVE_VENDOR_DIR:-$repo/torchnative/src/main}

# The spike venv that IMPORT_WALLS 3차/5차 measured against. Override to vendor
# from a different upstream; the stamp records which one was used.
src=${TORCHNATIVE_TORCH_SRC:-/Volumes/macMini/caches/spike-venv/lib/python3.13/site-packages}

if [ ! -d "$src/torch" ]; then
    echo "no torch under $src -- set TORCHNATIVE_TORCH_SRC" >&2
    exit 1
fi

version=$(sed -n "s/^__version__ = '\\(.*\\)'/\\1/p" "$src/torch/version.py")
if [ -z "$version" ]; then
    echo "could not read __version__ from $src/torch/version.py" >&2
    exit 1
fi

echo "vendoring torch $version"
echo "  from $src"
echo "  into $dest"

mkdir -p "$dest"
rm -rf "$dest"/torch-*.dist-info

# `--delete` rather than `rm -rf "$dest/torch"`: the destination is now inside
# the package (DESIGN.md §2's "주입 지점을 일원화"), so it already holds files
# that are tracked in git -- the add-hook and its README. Deleting the whole
# directory would take those with it. The excludes below are that union: rsync
# still removes anything stale from a previous upstream, and leaves ours alone.
rsync -a --delete \
    --exclude '/nn/federated.py' \
    --exclude '/README.md' \
    --exclude '/lib/' \
    --exclude '/include/' \
    --exclude '/bin/' \
    --exclude '/test/' \
    --exclude '__pycache__/' \
    --exclude '*.so' \
    --exclude '*.dylib' \
    --exclude '*.dll' \
    --exclude '*.pyd' \
    --exclude '*.a' \
    --exclude '*.lib' \
    "$src/torch/" "$dest/torch/"

# The torch *distribution* is three top-level packages, not one --
# `torch-$version.dist-info/top_level.txt` says `functorch`, `torch`,
# `torchgen`. Only `torch` used to be copied, and for the PYTHONPATH workflow
# that was invisible: `$PWD/torchnative/src/main` shadows site-packages for
# `torch`, and `torchgen`/`functorch` then quietly resolved to the *upstream
# install* underneath. So "import torch completes" was measured with two thirds
# of the distribution supplied by the reference installation.
#
# It stops being invisible the moment the tree is packaged into a wheel and put
# on a machine that has no upstream torch: `torch/nn/modules/module.py:17` ->
# `torch/utils/_python_dispatch.py:13` -> `import torchgen`, and the import dies
# 2254 lines into `torch/__init__.py`. Recorded in docs/platform/WHEEL.md.
#
# Read from top_level.txt rather than hard-coded so an upstream bump that adds
# or drops a sibling is followed, not silently missed the same way.
siblings=$(sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)$/\1/p' \
    "$src/torch-$version.dist-info/top_level.txt" 2>/dev/null | grep -v '^torch$' || true)
for pkg in $siblings; do
    if [ ! -d "$src/$pkg" ]; then
        echo "  warning: top_level.txt names $pkg but $src/$pkg does not exist" >&2
        continue
    fi
    echo "  sibling: $pkg"
    rsync -a --delete \
        --exclude '__pycache__/' \
        --exclude '*.so' \
        --exclude '*.dylib' \
        --exclude '*.dll' \
        --exclude '*.pyd' \
        --exclude '*.a' \
        --exclude '*.lib' \
        "$src/$pkg/" "$dest/$pkg/"
done

# IMPORT_WALLS 1차 §"관문은 is_torch_available() 하나다": transformers gates on
# `importlib.metadata.version("torch") >= 2.5.0`, so the distribution metadata is
# load-bearing, not decoration. Copying upstream's dist-info makes the vendored
# tree declare the version it actually came from.
if [ -d "$src/torch-$version.dist-info" ]; then
    rsync -a --exclude 'RECORD' --exclude '__pycache__/' \
        "$src/torch-$version.dist-info/" "$dest/torch-$version.dist-info/"
else
    echo "  warning: no torch-$version.dist-info in $src -- is_torch_available() will fail" >&2
fi

# What is left behind, so a later reader does not have to re-derive it.
roots="$dest/torch"
for pkg in $siblings; do
    [ -d "$dest/$pkg" ] && roots="$roots $dest/$pkg"
done
{
    echo "source=$src"
    echo "version=$version"
    echo "packages=$(echo torch $siblings | tr ' ' ',')"
    echo "vendored_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "py_modules=$(find $roots -name '*.py' | wc -l | tr -d ' ')"
    # "How many of *upstream's* compiled artefacts survived the copy" -- which
    # is what VENDOR.md §2 reads this number as, and it must stay 0. Our own
    # `_C.abi3.so` is excluded because rsync's `--exclude '*.so'` also protects
    # it from `--delete`, so re-vendoring over a tree that install_shim.sh has
    # already populated would otherwise count it and report 1.
    echo "native_left=$(find $roots \( -name '*.so' -o -name '*.dylib' \) \
        ! -name '_C.abi3.so' | wc -l | tr -d ' ')"
    echo "add_hooks=$(find "$dest/torch" -name 'federated.py' | wc -l | tr -d ' ')"
} > "$dest/.stamp"

cat "$dest/.stamp"
echo
echo "next: vendor/install_shim.sh"
