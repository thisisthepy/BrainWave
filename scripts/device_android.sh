#!/bin/sh
# Stage the vendored torch tree on an Android device and run it there.
#
# `docs/platform/RUST_CROSSBUILD.md` establishes that `aarch64-linux-android` links, and
# `docs/devices/DEVICE_LOAD.md` establishes that the resulting `_C.so` loads. Neither
# says anything about `import torch`, because linking is not loading and loading
# is not importing. This script closes that last gap, and `docs/devices/DEVICE.md`
# records what it found.
#
#   ./scripts/device_android.sh build     cross-compile _C for aarch64-linux-android
#   ./scripts/device_android.sh stage     push CPython + deps + vendored torch + _C
#   ./scripts/device_android.sh run <py>  run a host-side script on the device
#   ./scripts/device_android.sh parity    run device_parity.py on host and device, diff
#   ./scripts/device_android.sh diff <h> <d>   diff two existing parity JSONs
#   ./scripts/device_android.sh shell     interactive python on the device
#
# Set ANDROID_SERIAL when more than one device is attached; with several up and
# no choice made, every subcommand that needs one refuses and lists them.
#
# `parity` builds its own host `_C` -- deliberately not the shipped one. See
# cmd_parity, and docs/devices/DEVICE.md §5.1 for what that costs.
#
# Nothing here installs an app or touches anything outside /data/local/tmp. The
# emulator is shared with other projects (CLAUDE.md: one device test at a time).
set -eu

repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
crate=$repo/rust/torch_c

# The vendored tree moved into the package (`vendor/vendor_torch.sh` writes to
# `$TORCHNATIVE_VENDOR_DIR`, default `torchnative/src/main`); it is no longer
# `vendor/torch`. Read the same variable so the two cannot drift apart.
vendor_root=${TORCHNATIVE_VENDOR_DIR:-$repo/torchnative/src/main}

: "${CARGO_TARGET_DIR:=/Volumes/macMini/caches/cargo-target-device}"
: "${ANDROID_SDK_ROOT:=$HOME/Library/Android/sdk}"
: "${ANDROID_NDK_HOME:=$ANDROID_SDK_ROOT/ndk/27.1.12297006}"
: "${TARGET_PYTHON:=/Volumes/macMini/caches/target-python/aarch64-linux-android/prefix}"
: "${SPIKE_SITE:=/Volumes/macMini/caches/spike-venv/lib/python3.13/site-packages}"
: "${HOST_PYTHON:=/Volumes/macMini/caches/spike-venv/bin/python}"
: "${DEVICE_ROOT:=/data/local/tmp/bw_device}"
export CARGO_TARGET_DIR

# `parity` builds its own host artefact, with Accelerate off, and it must not
# land in the directory the shipping build uses -- one `cargo build` in each
# would otherwise evict the other's objects on every alternation. See
# `cmd_parity` for why the host end is built differently at all.
: "${PARITY_HOST_TARGET_DIR:=${CARGO_TARGET_DIR}-hostgemm}"
: "${PARITY_HOST_TREE:=${TMPDIR:-/tmp}/bw_parity_host}"

ADB_BIN=${ADB:-$ANDROID_SDK_ROOT/platform-tools/adb}
[ -x "$ADB_BIN" ] || ADB_BIN=$(command -v adb || true)
[ -n "$ADB_BIN" ] || { echo "adb not found -- set ADB" >&2; exit 1; }

# Two emulators are routinely up on this machine, and every bare `adb` call
# below then dies with "more than one device/emulator" -- a failure that arrives
# halfway through a staging run rather than at the start. Resolve the target
# once, here, and pass `-s` explicitly on every call afterwards: `ANDROID_SERIAL`
# is honoured (it is adb's own variable), a single attached device is taken
# without ceremony, and an ambiguous choice is refused *by name* rather than
# guessed at.
select_device() {
    if [ -n "${ANDROID_SERIAL:-}" ]; then
        export ANDROID_SERIAL
        return 0
    fi
    attached=$("$ADB_BIN" devices | awk '$2 == "device" { print $1 }')
    count=$(printf '%s\n' "$attached" | awk 'NF' | wc -l | tr -d ' ')
    case $count in
        0) echo "no device attached -- start an emulator or plug one in" >&2; exit 1 ;;
        1) ANDROID_SERIAL=$(printf '%s\n' "$attached" | awk 'NF'); export ANDROID_SERIAL ;;
        *) {
               echo "$count devices attached, and this script will not pick one for you:"
               printf '%s\n' "$attached" | awk 'NF { print "  " $0 }'
               echo "set ANDROID_SERIAL to the one you mean, e.g."
               echo "  ANDROID_SERIAL=$(printf '%s\n' "$attached" | awk 'NF' | head -1) $0 $*"
           } >&2
           exit 1 ;;
    esac
}

adb_() { "$ADB_BIN" -s "$ANDROID_SERIAL" "$@"; }

ANDROID_SO=$CARGO_TARGET_DIR/aarch64-linux-android/release/lib_C.so

# The device shell's exit code is not trustworthy (`adb shell` has returned 0
# for failing device commands), so every judgement below reads an explicit
# DEVICE_EXIT marker printed by the device shell itself rather than `$?` here.
device_run() {
    adb_ shell "cd $DEVICE_ROOT && \
        BW_STUB_MULTIPROCESSING=1 \
        TORCH_USE_RTLD_GLOBAL=1 \
        LD_LIBRARY_PATH=$DEVICE_ROOT/lib \
        PYTHONHOME=$DEVICE_ROOT \
        PYTHONPATH=$DEVICE_ROOT/site \
        ./bin/python3.13 $* 2>&1; echo DEVICE_EXIT=\$?"
}

cmd_build() {
    command -v cargo-ndk >/dev/null || { echo "cargo-ndk not installed" >&2; exit 1; }
    echo "building _C for aarch64-linux-android"
    ( cd "$crate" && \
      ANDROID_NDK_HOME=$ANDROID_NDK_HOME \
      PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 \
      PYO3_CROSS_LIB_DIR=$TARGET_PYTHON/lib \
      cargo ndk -t arm64-v8a --platform 21 build --release )
    ls -l "$ANDROID_SO"
}

cmd_stage() {
    [ -f "$ANDROID_SO" ] || { echo "no $ANDROID_SO -- run '$0 build'" >&2; exit 1; }
    [ -d "$vendor_root/torch" ] || { echo "no vendored tree -- run vendor/vendor_torch.sh" >&2; exit 1; }

    stage=$(mktemp -d)
    trap 'rm -rf "$stage"' EXIT
    mkdir -p "$stage/site"

    # The vendored Python tree, plus our `_C` in the hole vendor_torch.sh left.
    # `.abi3.so` is in CPython's EXTENSION_SUFFIXES on Android -- verified on
    # device, the list is
    #   ['.cpython-313-aarch64-linux-android.so', '.abi3.so', '.so']
    rsync -a --exclude '__pycache__/' --exclude '*.so' --exclude '*.dylib' \
        "$vendor_root/torch/" "$stage/site/torch/"
    for info in "$vendor_root"/torch-*.dist-info; do
        [ -d "$info" ] && rsync -a --exclude '__pycache__/' "$info" "$stage/site/"
    done
    cp "$ANDROID_SO" "$stage/site/torch/_C.abi3.so"

    # Wall 4 of docs/platform/VENDOR.md: `_manager_path()` checks this file exists on
    # every non-Windows platform before it will let the import finish.
    mkdir -p "$stage/site/torch/bin"
    : > "$stage/site/torch/bin/torch_shm_manager"

    # torch's pure-Python dependencies. Compiled artefacts are excluded because
    # the host copies are Mach-O/arm64-darwin and would not load here; every
    # package below has a pure-Python fallback path. `torchgen` is required by
    # `torch/utils/_python_dispatch.py` and is easy to miss -- it is a sibling
    # distribution of torch, not part of the vendored tree.
    for pkg in typing_extensions.py filelock fsspec jinja2 markupsafe mpmath \
               networkx sympy packaging torchgen functorch yaml; do
        if [ -e "$SPIKE_SITE/$pkg" ]; then
            rsync -a --exclude '__pycache__/' --exclude '*.so' --exclude '*.dylib' \
                --exclude '*.pyd' "$SPIKE_SITE/$pkg" "$stage/site/"
        else
            echo "  warning: no $pkg under $SPIKE_SITE" >&2
        fi
    done
    for info in "$SPIKE_SITE"/*.dist-info; do
        case $(basename "$info") in
            filelock-*|fsspec-*|jinja2-*|markupsafe-*|mpmath-*|networkx-*|sympy-*|packaging-*|typing_extensions-*|pyyaml-*)
                rsync -a --exclude RECORD --exclude '__pycache__/' "$info" "$stage/site/" ;;
        esac
    done

    left=$(find "$stage" \( -name '*.so' -o -name '*.dylib' \) ! -name '_C.abi3.so' | wc -l | tr -d ' ')
    [ "$left" = 0 ] || { echo "refusing to stage: $left host-native artefacts in tree" >&2; exit 1; }

    echo "staging CPython runtime on device"
    adb_ shell "mkdir -p $DEVICE_ROOT/lib $DEVICE_ROOT/bin"
    adb_ push --sync "$TARGET_PYTHON/bin/python3.13" "$DEVICE_ROOT/bin/" > /dev/null
    adb_ push --sync "$TARGET_PYTHON/lib/libpython3.13.so" "$DEVICE_ROOT/lib/" > /dev/null
    # The stdlib tree has no symlinks, so --sync carries it as-is. The sibling
    # libssl/libcrypto/libsqlite3 in prefix/lib *are* symlinks and `adb push`
    # cannot create those as uid 2000; they are skipped because `_C.so` does not
    # list any of them as NEEDED. Add `cp -L` staging if that ever changes.
    adb_ push --sync "$TARGET_PYTHON/lib/python3.13" "$DEVICE_ROOT/lib/" > /dev/null

    echo "staging torch tree on device"
    adb_ push --sync "$stage/site" "$DEVICE_ROOT/" > /dev/null
    adb_ shell "ls -l $DEVICE_ROOT/site/torch/_C.abi3.so; du -sh $DEVICE_ROOT"
}

cmd_run() {
    [ $# -ge 1 ] || { echo "usage: $0 run <script.py> [args]" >&2; exit 1; }
    script=$1; shift
    adb_ push --sync "$script" "$DEVICE_ROOT/$(basename "$script")" > /dev/null
    device_run "$(basename "$script")" "$@"
}

# The host end of `parity` is NOT the artefact we ship, and that is deliberate.
#
# The shipped Apple build links Accelerate (docs/perf/PERF.md §3); the Android build
# cannot, and calls candle's `gemm` instead. Comparing those two ends bit for
# bit compares two different BLAS implementations, which can only ever restate
# something already known -- and it does so loudly enough (8 cases, 1-2 ULP) to
# bury the device-specific kernel fault this script exists to find. So the host
# end is rebuilt with the Accelerate exception switched off, which puts `gemm`
# on both ends and makes bit equality the right thing to demand again.
#
# What that costs: `parity` no longer says anything about the artefact Apple
# users receive. docs/devices/DEVICE.md §5.1 carries that, because it is a real
# difference and not a defect, and it does not go away by being measured
# differently.
#
# `--config target."cfg(...)".rustflags` rather than the `RUSTFLAGS` variable:
# `RUSTFLAGS` replaces `.cargo/config.toml`'s rustflags instead of adding to
# them, and the host link needs `-undefined dynamic_lookup` from that file or it
# fails with a wall of undefined `_Py*` symbols.
NO_ACCELERATE_CONFIG='target."cfg(target_vendor = \"apple\")".rustflags = ["--cfg", "torch_c_no_accelerate"]'

host_parity_artefact() {
    command -v cargo >/dev/null || {
        echo "cargo not found -- parity has to build its own host artefact" >&2; exit 1; }
    echo "building host _C without Accelerate into $PARITY_HOST_TARGET_DIR"
    ( cd "$crate" && CARGO_TARGET_DIR=$PARITY_HOST_TARGET_DIR \
        cargo build --release --config "$NO_ACCELERATE_CONFIG" >&2 )

    if [ -f "$PARITY_HOST_TARGET_DIR/release/lib_C.dylib" ]; then
        host_so=$PARITY_HOST_TARGET_DIR/release/lib_C.dylib
    elif [ -f "$PARITY_HOST_TARGET_DIR/release/lib_C.so" ]; then
        host_so=$PARITY_HOST_TARGET_DIR/release/lib_C.so
    else
        echo "no host artefact under $PARITY_HOST_TARGET_DIR/release" >&2; exit 1
    fi

    # The cfg key above is the only thing standing between this and a
    # meaningless comparison, and if it ever stops reaching the manifest the
    # symptom is not an error -- it is eight quiet mismatches that look like a
    # device regression. Check the artefact rather than trusting the flag.
    if command -v otool >/dev/null; then
        linked=$(otool -L "$host_so" | grep -c -i Accelerate || true)
        [ "$linked" = 0 ] || {
            echo "refusing to measure $host_so: it still links Accelerate." >&2
            echo "The 'torch_c_no_accelerate' cfg in rust/torch_c/Cargo.toml did not take." >&2
            exit 1; }
    fi
}

# `parity` reads a file that `stage` wrote, possibly days ago, from a build that
# `build` may have replaced since. Neither leaves a receipt, so compare the two
# by content -- the same reasoning as run.sh's stale-shim refusal
# (docs/graph/CAPTURE.md §8): a comparison against the wrong artefact is worse than no
# comparison, because it reports a verdict.
require_fresh_device_artefact() {
    [ -f "$ANDROID_SO" ] || { echo "no $ANDROID_SO -- run '$0 build'" >&2; exit 1; }
    device_sum=$(adb_ shell "md5sum $DEVICE_ROOT/site/torch/_C.abi3.so 2>/dev/null || true" \
        | tr -d '\r' | awk '{ print $1 }')
    [ -n "$device_sum" ] || {
        echo "nothing staged at $DEVICE_ROOT/site/torch/_C.abi3.so -- run '$0 stage'" >&2; exit 1; }
    if command -v md5 >/dev/null; then host_sum=$(md5 -q "$ANDROID_SO")
    else host_sum=$(md5sum "$ANDROID_SO" | awk '{ print $1 }'); fi
    [ "$device_sum" = "$host_sum" ] || {
        echo "refusing to measure: the staged _C.abi3.so is not $ANDROID_SO" >&2
        echo "  device $device_sum" >&2
        echo "  host   $host_sum" >&2
        echo "run '$0 stage' to push the build you just made." >&2
        exit 1; }
}

# Import the vendored tree with a *different* `_C` in it, without touching the
# installed one. Everything under `torch/` is symlinked through to the vendored
# tree except the one file being substituted, and `$vendor_root` stays second on
# PYTHONPATH so its siblings (`torchnative`, the dist-info) resolve as usual.
#
# The alternative -- swapping `$vendor_root/torch/_C.abi3.so` out and back --
# leaves the wrong artefact installed if the run dies in between, and what is
# installed there is what four of run.sh's tests read.
host_parity_tree() {
    rm -rf "$PARITY_HOST_TREE"
    mkdir -p "$PARITY_HOST_TREE/torch"
    for entry in "$vendor_root"/torch/*; do
        [ -e "$entry" ] || continue
        name=$(basename "$entry")
        case $name in
            _C.abi3.so) continue ;;
        esac
        ln -s "$entry" "$PARITY_HOST_TREE/torch/$name"
    done
    cp "$host_so" "$PARITY_HOST_TREE/torch/_C.abi3.so"
}

cmd_parity() {
    host_json=${1:-/tmp/bw_host.json}
    device_json=${2:-/tmp/bw_device.json}

    [ -d "$vendor_root/torch" ] || {
        echo "no vendored tree at $vendor_root/torch -- run vendor/vendor_torch.sh" >&2; exit 1; }
    require_fresh_device_artefact
    host_parity_artefact
    host_parity_tree

    echo "=== host ==="
    echo "host _C: $host_so (no Accelerate -- see cmd_parity)"
    TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$PARITY_HOST_TREE:$vendor_root \
        "$HOST_PYTHON" "$repo/scripts/device_parity.py" "$host_json" || true

    echo "=== device ==="
    adb_ push --sync "$repo/scripts/device_parity.py" "$DEVICE_ROOT/device_parity.py" > /dev/null
    device_run "device_parity.py $DEVICE_ROOT/parity.json"
    adb_ pull "$DEVICE_ROOT/parity.json" "$device_json" > /dev/null

    cmd_diff "$host_json" "$device_json"
}

# Split out from `parity` so the comparison can be re-run, and negative-control
# tested, without re-running either end. `parity ... && diff <doctored>` is how
# the exemption list below was shown to actually fire.
cmd_diff() {
    echo "=== diff ==="
    "$HOST_PYTHON" - "$1" "$2" <<'PYEOF'
import json, sys

# Measured on emulator-5554 (pmp_api26, API 26, arm64-v8a) against an
# aarch64-darwin host -- docs/devices/DEVICE.md records the run. These four cases and
# only these four differ, by at most 1 ULP, and both sides straddle the
# correctly-rounded double-precision reference in both directions, so neither is
# "the wrong one": Apple's libm and bionic's are different implementations of
# expf/tanhf. Everything else -- including the two nn.Module forwards -- is
# bit-identical.
#
# `exp.default` and `softplus.default` joined `_softmax.default` and
# `tanh.default` when the battery grew past its original 33 cases
# (docs/devices/DEVICE.md records the reference-value check): `softplus` calls `exp`
# internally, so its own 1-ULP divergence is inherited, not independent --
# the same `expf` call underneath, the same directionless straddle.
#
# Listing them by name rather than by a global tolerance is the point. A
# tolerance would also swallow a real divergence in `mm` or `cumsum`, which is
# exactly what this script exists to catch. Anything not on this list is a
# regression, including any of these four growing past 1 ULP.
#
# The list stays this short because `cmd_parity` builds the host end with
# Accelerate off. Against the *shipped* Apple artefact it would need eight
# more entries at 1-2 ULP -- `mm`/`addmm`/`bmm`/`native_layer_norm`/`nn.Linear`
# from BLAS accumulating in a different order, `sin`/`cos`/`rsqrt` from vForce
# -- and a list that long stops being a list of known exceptions and starts
# being the tolerance this comparison refused to have.
EXPECTED_LIBM_DIVERGENCE = {
    "_softmax.default": 1,
    "tanh.default": 1,
    "exp.default": 1,
    "softplus.default": 1,
}

host = json.load(open(sys.argv[1]))
device = json.load(open(sys.argv[2]))
print(f"host   {host['platform']}/{host['machine']} torch {host['torch']} kernels {host['aten_implemented']}")
print(f"device {device['platform']}/{device['machine']} torch {device['torch']} kernels {device['aten_implemented']}")
# Which two artefacts this verdict is actually about. Both ends record it in the
# JSON, so a re-run of `diff` on saved files says so too.
print(f"host   _C {host.get('c_extension', '?')}")
print(f"device _C {device.get('c_extension', '?')}")
shared = set(host["results"]) & set(device["results"])
mismatched = [k for k in sorted(shared) if host["results"][k] != device["results"][k]]
unexpected = []
for key in mismatched:
    hb, db = host["results"][key], device["results"][key]
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(hb["bits"], db["bits"])) if x != y]
    ulps = [abs(int(x, 16) - int(y, 16)) for _, x, y in diffs
            if isinstance(x, str) and isinstance(y, str)]
    worst = max(ulps) if ulps else None
    budget = EXPECTED_LIBM_DIVERGENCE.get(key)
    verdict = "MISMATCH" if budget is None or worst is None or worst > budget else "libm"
    if verdict == "MISMATCH":
        unexpected.append(key)
    print(f"{verdict} {key}: {len(diffs)}/{len(hb['bits'])} elements, max {worst if worst is not None else '?'} ULP")
    for i, x, y in diffs:
        print(f"    [{i}] host={x} device={y}")
print(f"identical {len(shared) - len(mismatched)}/{len(shared)}")
print(f"host failures:   {sorted(host['failures'])}")
print(f"device failures: {sorted(device['failures'])}")

# Two separate ways for the ends to disagree, reported separately so a run that
# fails says which one it was.
verdicts = []
if sorted(host["failures"]) != sorted(device["failures"]):
    verdicts.append("failure sets differ")
if unexpected:
    verdicts.append(f"unexpected bit divergence: {unexpected}")
stale = sorted(set(EXPECTED_LIBM_DIVERGENCE) - set(mismatched))
if stale:
    # Not a failure: if bionic's libm gets fixed this list should shrink, and
    # the run should say so rather than quietly keep an obsolete exemption.
    print(f"note: expected-divergence entries that now agree: {stale}")
print("PARITY: " + ("; ".join(verdicts) if verdicts else "ok"))
sys.exit(1 if verdicts else 0)
PYEOF
}

cmd_shell() { device_run "$@"; }

# `build` and `diff` never talk to a device, so they are not made to wait on one
# being attached; everything else resolves the target first, before it has done
# any work that a "more than one device/emulator" would waste.
case ${1:-} in
    build)  shift; cmd_build "$@" ;;
    stage)  shift; select_device stage; cmd_stage "$@" ;;
    run)    shift; select_device run; cmd_run "$@" ;;
    parity) shift; select_device parity; cmd_parity "$@" ;;
    diff)   shift; cmd_diff "$@" ;;
    shell)  shift; select_device shell; cmd_shell "$@" ;;
    *) sed -n '2,24p' "$0"; exit 2 ;;
esac
