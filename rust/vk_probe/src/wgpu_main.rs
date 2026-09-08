// The same three cases as `main.rs`, over `wgpu` instead of `ash`.
//
// This exists because the choice between the two is not decidable from build
// logs. `ash` was proven on the device first; recommending `wgpu` -- which is
// the only route with a story for the iOS and macOS targets as well -- on the
// strength of "it cross-compiles" would be exactly the unverified-claim
// failure this repository keeps paying for. So it gets run on the device too,
// graded by `check.rs`, the same code that graded `ash`.
//
// Like the ash probe: no timings. See docs/devices/VULKAN.md §7.

#[path = "check.rs"]
mod check;
use check::*;

use wgpu::util::DeviceExt;

// WGSL, not GLSL. wgpu translates this to SPIR-V through `naga`, in pure Rust,
// with no C++ compiler anywhere -- which is the whole reason this route's
// shader-supply story is worth checking. The arithmetic is written to match
// `check.rs`'s CPU reference exactly: same accumulation order, no fused
// multiply-add asked for.
const VECADD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.x) { return; }
    c[i] = a[i] + b[i];
}
"#;

const MATMUL_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform>             dims: vec4<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = dims.x; let n = dims.y; let k = dims.z;
    let col = gid.x;
    let row = gid.y;
    if (row >= m || col >= n) { return; }

    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < k; t = t + 1u) {
        acc = acc + a[row * k + t] * b[t * n + col];
    }
    c[row * n + col] = acc;
}
"#;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// Bind a, b, c and a uniform `dims`, dispatch, and read `c` back.
/// Everything is per-call for the same reason as in the ash probe: this is a
/// probe, not a runtime.
fn run(gpu: &Gpu, wgsl: &str, a: &[f32], b: &[f32], out_len: usize, dims: [u32; 4], groups: (u32, u32)) -> Vec<f32> {
    let dev = &gpu.device;

    let buf_a = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("a"),
        contents: bytemuck_cast(a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_b = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("b"),
        contents: bytemuck_cast(b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_bytes = (out_len * 4) as u64;
    let buf_c = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("c"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // wgpu has no push constants without a feature, so the shape goes in a
    // uniform. It is still one shader for every shape -- which is the property
    // that matters, see docs/devices/VULKAN.md §3.
    let buf_d = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dims"),
        contents: &dims.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<u8>>(),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: out_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_d.as_entire_binding() },
        ],
    });

    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, 1);
    }
    enc.copy_buffer_to_buffer(&buf_c, 0, &staging, 0, out_bytes);
    gpu.queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .expect("poll");
    let data = slice.get_mapped_range().expect("get_mapped_range");
    let out: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    drop(data);
    staging.unmap();
    out
}

/// Avoids taking a `bytemuck` dependency for four lines.
fn bytemuck_cast(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn main() {
    println!("vk_probe_wgpu: wgpu compute vs CPU, {} {}", std::env::consts::OS, std::env::consts::ARCH);
    let tampering = tamper_ulp().is_some();
    if let Some(n) = tamper_ulp() {
        println!("VK_PROBE_TAMPER={n}: reference perturbed by {n} ULP, every case MUST report MISMATCH");
    }

    // `new_without_display_handle` is wgpu 30's headless idiom -- it is the
    // constructor that says out loud there is no window, which is exactly the
    // claim being tested here.
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(desc);
    // No surface: headless compute from a plain binary, no Activity, no JNI.
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    })) {
        Ok(a) => a,
        Err(e) => {
            println!("RESULT: FAIL (init) -- request_adapter: {e}");
            std::process::exit(2);
        }
    };
    let info = adapter.get_info();
    println!("adapter: {:?} backend={:?} type={:?} driver={:?}", info.name, info.backend, info.device_type, info.driver);

    let (device, queue) = match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        experimental_features: Default::default(),
        memory_hints: Default::default(),
        trace: wgpu::Trace::Off,
    })) {
        Ok(p) => p,
        Err(e) => {
            println!("RESULT: FAIL (init) -- request_device: {e}");
            std::process::exit(2);
        }
    };
    let gpu = Gpu { device, queue };

    let mut all_ok = true;

    {
        let n = 1 << 20;
        let a = fill(1, n);
        let b = fill(2, n);
        let mut want = cpu_vecadd(&a, &b);
        tamper(&mut want);
        println!("vecadd n={n}");
        let got = run(&gpu, VECADD_WGSL, &a, &b, n, [n as u32, 0, 0, 0], (n.div_ceil(64) as u32, 1));
        all_ok &= report("vecadd", &compare(&got, &want));
    }

    for &(m, n, k) in &[(64usize, 64usize, 64usize), (96, 64, 80)] {
        let a = fill(3, m * k);
        let b = fill(4, k * n);
        let mut want = cpu_matmul(&a, &b, m, n, k);
        tamper(&mut want);
        println!("matmul {m}x{k} * {k}x{n}");
        let got = run(
            &gpu,
            MATMUL_WGSL,
            &a,
            &b,
            m * n,
            [m as u32, n as u32, k as u32, 0],
            (n.div_ceil(8) as u32, m.div_ceil(8) as u32),
        );
        all_ok &= report(&format!("matmul {m}x{k}x{n}"), &compare(&got, &want));
    }

    let verdict = if tampering { !all_ok } else { all_ok };
    if tampering {
        println!("RESULT: {} (tamper: comparison {} the perturbation)",
                 if verdict { "PASS" } else { "FAIL" },
                 if verdict { "caught" } else { "MISSED" });
    } else {
        println!("RESULT: {}", if verdict { "PASS" } else { "FAIL" });
    }
    std::process::exit(if verdict { 0 } else { 1 });
}
