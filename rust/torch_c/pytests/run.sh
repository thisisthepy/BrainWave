#!/bin/sh
# Build the host artefact, rename it to `_C.abi3.so`, and run the smoke tests
# against it. Renaming is not incidental: cargo emits `lib_C.dylib`, and Python
# only loads a file whose name ends in one of importlib's extension suffixes
# (RUST_CROSSBUILD.md §2).
#
# The suffix is `.abi3.so`, not a bare `.so`: ABI3.md §7 item 2. An untagged
# `_C.so` loads into *any* interpreter, which is precisely the silent failure
# the abi3 build exists to remove -- so the filename should say what it is.
set -eu

crate_dir=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
repo_root=$(CDPATH='' cd -- "$crate_dir/../.." && pwd)
target_dir=${CARGO_TARGET_DIR:-$crate_dir/target}
# Per-checkout, not per-machine. The default used to be a single
# $TMPDIR/torch-c-stage shared by every worktree on the box, and this project
# runs several concurrently: two suites overlapping would stage into the same
# path and each would validate the other's artefact. It was caught by the
# documentation checker reporting ops=168 while compare.py in the same tree
# reported 166 -- a disagreement only possible if the two were reading
# different binaries. That is the false-green shape, and a suite that can
# silently check somebody else's build is worse than one that refuses.
#
# The hash is of the checkout path, so each worktree gets its own and reruns
# in the same worktree still reuse theirs.
stage=${TORCH_C_STAGE:-${TMPDIR:-/tmp}/torch-c-stage-$(printf '%s' "$repo_root" | cksum | cut -d' ' -f1)}

# `cd` rather than `--manifest-path`: cargo discovers `.cargo/config.toml` from
# the *working directory*, not from the manifest. Building this crate from
# elsewhere silently drops `-undefined dynamic_lookup` and the link fails with
# a wall of undefined `_Py*` symbols. Same trap as the hardcoded iOS `-F` path,
# from the other side -- see build.rs.
cd -- "$crate_dir"
cargo build --release

mkdir -p "$stage"
rm -f "$stage/_C.so"
if [ -f "$target_dir/release/lib_C.dylib" ]; then
    cp "$target_dir/release/lib_C.dylib" "$stage/_C.abi3.so"
elif [ -f "$target_dir/release/lib_C.so" ]; then
    cp "$target_dir/release/lib_C.so" "$stage/_C.abi3.so"
else
    echo "no host artefact under $target_dir/release" >&2
    exit 1
fi

# Four tests in test_shim.py (capture/checkpoint/device/meta) do not import
# the artefact staged above -- they shell out to a subprocess that puts the
# vendored tree on PYTHONPATH, and that subprocess loads
# `$vendor_dir/torch/_C.abi3.so`. `vendor/install_shim.sh` is the only thing
# that writes that file; this script never has. So a source change that only
# this script rebuilds does not reach those four tests -- they would keep
# testing whatever `install_shim.sh` last installed, silently, with a green
# result (docs/graph/CAPTURE.md §8).
#
# Refuse by name rather than install it here: installing would make this
# script a build step for a *different* artefact (the one that ships inside
# the vendored tree) with its own failure modes (torch/bin/torch_shm_manager,
# TORCHNATIVE_VENDOR_DIR) that have nothing to do with "run the smoke tests".
# Comparing bytes rather than mtimes because this crate rebuilds
# byte-identical output when nothing relevant changed (measured: a `touch` +
# rebuild of dtype.rs reproduced the previous dylib exactly on this
# toolchain), so a byte compare does not nag on a no-op rebuild the way an
# mtime check would.
#
# `cmp` distinguishes three outcomes and this has to distinguish them too: 0 is
# same, 1 is differ, anything above 1 is "the comparison itself failed". Reading
# every non-zero as "stale" conflates the check with its own failure -- which
# happened: under memory pressure from a concurrent build the kernel killed
# `cmp` with SIGKILL, the guard read exit 137 as a difference, and the suite
# refused with a message telling the reader to reinstall a shim that was already
# current. That is the repeated defect of this repository wearing a new hat, so
# the two are separated and only one of them is a staleness claim.
vendor_dir=${TORCHNATIVE_VENDOR_DIR:-$repo_root/torchnative/src/main}
vendor_shim="$vendor_dir/torch/_C.abi3.so"
if [ -f "$vendor_shim" ]; then
    cmp -s "$stage/_C.abi3.so" "$vendor_shim" && cmp_status=0 || cmp_status=$?
else
    cmp_status=0
fi
if [ "$cmp_status" -gt 1 ]; then
    cat >&2 <<EOF
run.sh: refusing to run -- could not compare against $vendor_shim.

\`cmp\` exited $cmp_status, which is neither "same" (0) nor "different" (1), so
whether the vendored shim is current is unknown. This is not a staleness
report. A SIGKILL here has meant memory pressure from a concurrent build;
re-running once the machine is quieter has been enough.
EOF
    exit 1
fi
if [ "$cmp_status" -eq 1 ]; then
    cat >&2 <<EOF
run.sh: refusing to run -- $vendor_shim is stale.

It does not match what was just built from rust/torch_c/src. The capture,
checkpoint, device, and meta tests read that file (not this script's
staged artefact) through a vendored-tree subprocess, so running them now
would silently re-test the old build instead of catching a change here.

Fix: run vendor/install_shim.sh, then re-run this script.
EOF
    exit 1
fi

# macOS SIP strips every `DYLD_*` variable from the environment when it execs
# a protected binary, and `/bin/sh` is one. So `DYLD_LIBRARY_PATH=... sh run.sh`
# arrives here with that variable already gone -- it was in the caller's shell
# and is not in this one. Nothing in this script can restore it, because
# nothing in this script ever saw it.
#
# docs/devices/VULKAN3.md §6.1 is what that cost: the four Vulkan tests skipped saying
# "no vulkan" while the loader was pointed at correctly, and the person who had
# just supplied the loader had no way to tell. A skip that gives a false reason
# is worse than a failure, because it is counted as a pass.
#
# `TORCH_C_DYLD_LIBRARY_PATH` is the way in. It is not a `DYLD_*` name, so it
# survives the exec, and it is re-exported below for the two Python
# invocations that need it -- exporting it here rather than at the top so it is
# obvious that its only purpose is to reach the loader.
if [ -n "${TORCH_C_DYLD_LIBRARY_PATH:-}" ]; then
    DYLD_LIBRARY_PATH="$TORCH_C_DYLD_LIBRARY_PATH${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export DYLD_LIBRARY_PATH
fi

# The trap leaves no trace of itself -- a stripped variable is simply absent --
# but it does leave a signature, and this is it. `VK_DRIVER_FILES` is not a
# `DYLD_*` name, so SIP does not strip it; a caller who set one and not the
# other set both and lost one on the way in. Said here as well as in the skip
# line, because this is where a reader is looking when they wonder why.
if [ -n "${VK_DRIVER_FILES:-}" ] && [ -z "${DYLD_LIBRARY_PATH:-}" ]; then
    cat >&2 <<EOF
run.sh: VK_DRIVER_FILES is set but DYLD_LIBRARY_PATH is not.

macOS SIP strips DYLD_* when exec'ing /bin/sh, so if you passed
DYLD_LIBRARY_PATH on this command line it did not reach this script and will
not reach Python. The Vulkan tests will skip -- truthfully, but for a reason
that is about this process and not about your machine.

Pass it as TORCH_C_DYLD_LIBRARY_PATH instead; this script re-exports it.
(docs/devices/VULKAN3.md §6.1)
EOF
fi

# The crate's own unit tests, before the Python suite.
#
# There are 30 of them and **nothing ran them until docs/training/TRAIN2.md noticed**.
# One had been red for an unknown length of time -- `RULE_OPS` was not sorted,
# and its own `assert_eq!(sorted, RULE_OPS)` said so to nobody. A test that
# nothing invokes is not a gate, which is the same finding this repository
# already recorded for the golden self-test and the documentation checker, both
# of which are invoked from this script for exactly that reason.
#
# They cost about a second: the crate is already built by the line above, and
# these are pure-Rust assertions over constant tables with no Python involved.
cargo test --release --quiet || exit $?

# Every `test_*.py` in `pytests/`, not just `test_shim.py`.
#
# One file was the whole suite for a long time, and the cost showed up in
# merges rather than in tests: several rounds land in parallel, all of them
# append before `if __name__ == "__main__"`, and git resolves that as one
# conflict hunk spanning thousands of lines. Reconstructing it by hand has
# twice silently dropped tests that the branch had added -- once seven of
# them, caught only because DOCWATCH markers named them.
#
# Splitting by topic makes those merges disjoint. The files share helpers by
# importing `test_shim`, which is why `pytests/` is on PYTHONPATH; each one's
# `__main__` guard keeps that import from running anything.
# `TORCHNATIVE_VULKAN_DYLD`: an opt-in way to let the vulkan tests actually run.
#
# docs/devices/VULKAN3.md §6.1 recorded the trap and left it: macOS SIP **strips
# `DYLD_*` from the environment when a protected binary is exec'd**, and
# `/bin/sh` is one. So `DYLD_LIBRARY_PATH=... sh run.sh` sets a variable that
# this script can still see and the interpreter it launches cannot -- the
# vulkan tests skip saying "no loader", on a machine where the caller supplied
# one. That is the closest thing to a false green this device has produced.
#
# Re-exporting it here, immediately before the interpreter is launched, is the
# fix that document proposed. Under a *different* name, so that a caller who
# exports `DYLD_LIBRARY_PATH` and watches it vanish is not told twice that it
# worked; and opt-in, so a run that does not set it behaves exactly as before
# and the tests skip by name as they always did.
vk_env=""
if [ -n "${TORCHNATIVE_VULKAN_DYLD:-}" ]; then
    vk_env="DYLD_LIBRARY_PATH=$TORCHNATIVE_VULKAN_DYLD"
    echo "vulkan: re-exporting DYLD_LIBRARY_PATH=$TORCHNATIVE_VULKAN_DYLD (SIP strips it through /bin/sh)"
fi

suite_failed=0
for suite in "$crate_dir"/pytests/test_*.py; do
    echo "--- $(basename "$suite") ---"
    env $vk_env PYTHONPATH="$stage:$crate_dir/pytests" "${PYTHON:-python3}" "$suite" || suite_failed=1
done
[ "$suite_failed" -eq 0 ] || exit 1

# The golden harness has its own self-test -- it injects a fault shaped like a
# plausible misimplementation at each comparator and checks the comparator
# rejects it. Nothing invoked it, so the gate existed without ever being pulled;
# it caught that the previous injection reached exactly one case out of 1781.
# It builds nothing, so it costs a few seconds here.
#
# TORCH_C_ARTEFACT points it at the artefact this script just staged, since
# tools/golden/loader.py otherwise falls back to a fixed cache path that may
# hold an entirely different build.
TORCH_C_ARTEFACT="$stage/_C.abi3.so" \
    "${PYTHON:-python3}" "$repo_root/tools/golden/compare.py" --self-test || exit $?

# The documentation checker, for the same reason the golden self-test is here:
# a gate nobody pulls is not a gate. An audit found false claims spread across
# six of eleven load-bearing documents, and every one arrived the same way --
# a later commit closed a gap and nobody returned to the document that had
# named it. The markers assert only what has a single ground truth (an op in
# `_aten_implemented()`, a key in a table, a count read from a suite's own
# summary line), so this cannot cry wolf on prose; docs/verification/DOCWATCH.md says what
# it structurally cannot see.
#
# README.md is passed explicitly. With no arguments the checker scans `docs/*.md`
# only -- so the fifteen markers on the README, which hold down every number a
# reader of the front page sees, were outside the gate while every number in
# `docs/` was inside it. The claims most likely to be read were the ones least
# likely to be checked.
# `find`, not `"$repo_root"/docs/*.md`. A shell glob does not recurse, and the
# documents now live in `docs/<folder>/` (docs/README.md is the index). With the
# glob, the moment a document moved into a subfolder it left DOCWATCH silently:
# the gate stays green and the marker count gets *smaller*, which is the
# "verification that cannot fail" shape CLAUDE.md §5.5 records. The count is the
# proof -- it went 1070 -> 1070 across the move, not down.
# `test_docrefs.py` fails if this glob comes back, here or in check_docs.py.
doc_files=$(find "$repo_root/docs" -name '*.md' | sort)
TORCH_C_ARTEFACT="$stage/_C.abi3.so" \
    exec "${PYTHON:-python3}" "$repo_root/tools/docwatch/check_docs.py" \
        $doc_files "$repo_root/README.md"
