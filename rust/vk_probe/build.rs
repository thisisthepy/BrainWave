// Compiles the GLSL compute shaders to SPIR-V at build time and drops the words
// in OUT_DIR for `include_bytes!`.
//
// This is the whole answer to "how are shaders supplied". `glslc` runs on the
// *build* machine, never on the device -- nothing here ships a shader compiler
// into the app. Kernels are a fixed set that we author, so compiling them ahead
// of time is not the same thing as converting a model ahead of time; the
// premise the project protects is about models, and this does not touch it.
//
// The NDK ships glslc under shader-tools/, so there is no extra toolchain to
// install. Override with GLSLC if you have your own.

use std::path::{Path, PathBuf};
use std::process::Command;

fn find_glslc() -> PathBuf {
    if let Ok(p) = std::env::var("GLSLC") {
        return PathBuf::from(p);
    }
    // The NDK's copy. The host directory is named for the toolchain's own
    // architecture, not ours -- on Apple silicon the x86_64 binary runs under
    // Rosetta, which is why this is not gated on the host arch.
    let ndk = std::env::var("ANDROID_NDK_HOME")
        .or_else(|_| std::env::var("ANDROID_NDK_ROOT"))
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").expect("HOME");
            format!("{home}/Library/Android/sdk/ndk/27.1.12297006")
        });
    for host in ["darwin-x86_64", "darwin-arm64", "linux-x86_64"] {
        let c = Path::new(&ndk).join("shader-tools").join(host).join("glslc");
        if c.exists() {
            return c;
        }
    }
    PathBuf::from("glslc")
}

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let glslc = find_glslc();

    for name in ["vecadd", "matmul"] {
        let src = Path::new("shaders").join(format!("{name}.comp"));
        let dst = out.join(format!("{name}.spv"));
        println!("cargo:rerun-if-changed={}", src.display());

        let status = Command::new(&glslc)
            .arg("-fshader-stage=compute")
            .arg("-O")
            .arg(&src)
            .arg("-o")
            .arg(&dst)
            .status()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", glslc.display()));
        assert!(status.success(), "glslc failed on {}", src.display());
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=GLSLC");
}
