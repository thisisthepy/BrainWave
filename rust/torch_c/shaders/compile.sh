#!/bin/sh
# Regenerates the checked-in SPIR-V beside each `.comp`.
#
# The words are checked in and `vulkan.rs` `include_bytes!`s them, rather than a
# `build.rs` compiling them. That is deliberate: `torch_c` builds for Android,
# iOS and the host, and a build script that shells out to `glslc` would make all
# three depend on a shader compiler being present for a file that changes about
# once a round. Checking the words in keeps `cargo build --release` exactly what
# `vendor/install_shim.sh` already runs, on a machine with no Vulkan toolchain.
#
# The cost is that an edit to a `.comp` does nothing until this is run. The
# guard against that is in the test suite, not in the build: it asserts the
# `.spv` is newer than its `.comp`.
#
# glslc comes from the NDK's shader-tools; override with GLSLC.
set -e
cd "$(dirname "$0")"
NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-$HOME/Library/Android/sdk/ndk/27.1.12297006}}"
if [ -z "$GLSLC" ]; then
  for host in darwin-x86_64 darwin-arm64 linux-x86_64; do
    if [ -x "$NDK/shader-tools/$host/glslc" ]; then GLSLC="$NDK/shader-tools/$host/glslc"; break; fi
  done
fi
GLSLC="${GLSLC:-glslc}"
for src in *.comp; do
  "$GLSLC" -fshader-stage=compute -O "$src" -o "${src%.comp}.spv"
  echo "compiled $src -> ${src%.comp}.spv"
done
