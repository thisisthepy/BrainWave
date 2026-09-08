#!/bin/sh
# Build `rust/torch_c` for the host and drop it into the hole that
# `vendor_torch.sh` left in the vendored tree.
#
# The filename is `_C.abi3.so` (ABI3.md §7 item 2). Upstream ships
# `_C.cpython-313-darwin.so`; both are in importlib's EXTENSION_SUFFIXES, and
# the abi3 tag is the honest one for a Limited-API build.
#
# `cd` into the crate rather than using `--manifest-path`: cargo discovers
# `.cargo/config.toml` from the working directory, and without it the host link
# loses `-undefined dynamic_lookup` (docs/design/TORCH_C.md §3).
set -eu

repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
dest=${TORCHNATIVE_VENDOR_DIR:-$repo/torchnative/src/main}
crate=$repo/rust/torch_c
target_dir=${CARGO_TARGET_DIR:-$crate/target}

if [ ! -d "$dest/torch" ]; then
    echo "no vendored tree at $dest/torch -- run vendor/vendor_torch.sh first" >&2
    exit 1
fi

cd -- "$crate"
cargo build --release

if [ -f "$target_dir/release/lib_C.dylib" ]; then
    artefact=$target_dir/release/lib_C.dylib
elif [ -f "$target_dir/release/lib_C.so" ]; then
    artefact=$target_dir/release/lib_C.so
else
    echo "no host artefact under $target_dir/release" >&2
    exit 1
fi

rm -f "$dest/torch/_C.abi3.so"
cp "$artefact" "$dest/torch/_C.abi3.so"
echo "installed $(basename "$artefact") -> $dest/torch/_C.abi3.so"

# `torch/__init__.py:2179 _manager_path()` refuses to import unless
# `torch/bin/torch_shm_manager` exists on disk -- unconditionally, on every
# non-Windows platform. It is the helper process for sharing tensor storages
# between OS processes; on a phone there is nothing to share with, and we are
# never going to ship it.
#
# Only *existence* is checked here; the path is handed to `_initExtension` and
# is not executed until someone actually uses `torch.multiprocessing`. So a
# zero-byte marker is enough, and it is left as a marker rather than quietly
# fixed up so that the requirement stays visible. Recorded as wall 4 in
# docs/platform/VENDOR.md.
mkdir -p "$dest/torch/bin"
: > "$dest/torch/bin/torch_shm_manager"
echo "placed empty torch/bin/torch_shm_manager (wall 4 -- see docs/platform/VENDOR.md)"
