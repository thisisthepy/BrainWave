//! Per-target link wiring that cannot live in `.cargo/config.toml`.
//!
//! RUST_CROSSBUILD.md §0.5 records that iOS is the one target whose link line
//! needs a *path* -- `-F <dir containing Python.framework>` -- and that this
//! path was hardcoded to one machine in the committed `.cargo/config.toml`.
//! Cargo does no environment expansion inside `rustflags`, so the fix cannot be
//! a variable in that file; it has to be a build script, which is also what the
//! same document calls "`Cargo.kt` 가 주입해야 할 값".
//!
//! The contract is therefore:
//!
//! | variable | who sets it | when |
//! |---|---|---|
//! | `TORCHNATIVE_PYTHON_FRAMEWORK_DIR` | the build driver (`Cargo.kt`, or the dev shell) | iOS targets only |
//! | `PYO3_CONFIG_FILE` | same, with `suppress_build_script_link_lines=true` | iOS targets only |
//!
//! Both are needed together and neither is sufficient alone -- PyO3 hardcodes
//! iOS as "links libpython" regardless of the `extension-module` feature, so it
//! keeps emitting `-lpython3.13` for a library that does not exist in the iOS
//! distribution until it is suppressed. §0.5 measured both halves.
//!
//! Missing wiring fails here with a message rather than at link time with
//! `library 'python3.13' not found`, which is what the hardcoded path produced
//! on any machine that was not this one.

use std::path::Path;

const FRAMEWORK_DIR_VAR: &str = "TORCHNATIVE_PYTHON_FRAMEWORK_DIR";

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    println!("cargo::rustc-env=TORCH_C_TARGET={target}");
    println!("cargo::rerun-if-env-changed={FRAMEWORK_DIR_VAR}");

    // `torch_c_cuda` is the cfg key `Cargo.toml`'s CUDA entry is gated on, and
    // `device.rs` reads it too. Declaring it here is what keeps `#[cfg(...)]`
    // in the source from tripping rustc's `unexpected_cfgs` lint -- Cargo
    // auto-declares the keys it knows (features, targets) and cannot know a
    // custom one used inside a `[target.'cfg(...)']` table.
    //
    // Only this key is declared. `torch_c_no_accelerate` is spelled in
    // `Cargo.toml` and nowhere in the source, so it needs no declaration and
    // adding one here would be a change to something this round did not touch.
    println!("cargo::rustc-check-cfg=cfg(torch_c_cuda)");

    // The compute capability the CUDA kernels were compiled for, read at *this*
    // crate's compile time and baked in so `device.rs` can compare a running
    // GPU against it (`option_env!("CUDA_COMPUTE_CAP")`).
    //
    // It is not this crate's variable: `candle-kernels`'s build script requires
    // it on any machine without a GPU, because its own detection is
    // `nvidia-smi --query-gpu=compute_cap` and there is nothing to ask. So on a
    // CUDA build it is always set, and reading it costs nothing; on every other
    // build it is absent and the comparison is skipped rather than guessed.
    println!("cargo::rerun-if-env-changed=CUDA_COMPUTE_CAP");

    if is_ios_device(&target) {
        link_ios_python_framework(&target);
    }
}

/// Physical iOS only. The simulator resolves CPython symbols the way macOS does
/// (`-undefined dynamic_lookup`, in `.cargo/config.toml`), and Mac Catalyst is a
/// third case that has not been tried at all -- so neither is claimed here.
fn is_ios_device(target: &str) -> bool {
    target.contains("-apple-ios") && !target.ends_with("-sim") && !target.contains("macabi")
}

fn link_ios_python_framework(target: &str) {
    let dir = match std::env::var(FRAMEWORK_DIR_VAR) {
        Ok(dir) if !dir.trim().is_empty() => dir,
        _ => panic!(
            "{FRAMEWORK_DIR_VAR} is not set, and target {target} has no linkable \
             libpython -- only Python.framework. Set it to the directory that \
             *contains* Python.framework (for the python-build-standalone iOS \
             distribution, that is the `arm64-iphoneos` directory), and set \
             PYO3_CONFIG_FILE with suppress_build_script_link_lines=true \
             alongside it. See docs/platform/RUST_CROSSBUILD.md §0.5 and docs/design/TORCH_C.md."
        ),
    };

    // Catch the wrong-directory mistake here rather than in the linker, where it
    // reappears as the same "library 'python3.13' not found" this whole path
    // exists to avoid.
    let framework = Path::new(&dir).join("Python.framework");
    if !framework.is_dir() {
        panic!(
            "{FRAMEWORK_DIR_VAR}={dir} does not contain Python.framework. It must \
             point at the directory holding the framework, not at the framework \
             itself and not at the distribution's `lib`."
        );
    }

    // `-L framework=<dir>` is rustc's spelling of clang's `-F <dir>`.
    println!("cargo::rustc-link-search=framework={dir}");
    println!("cargo::rustc-link-lib=framework=Python");
}
