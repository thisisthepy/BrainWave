// Smallest thing that answers "can a Vulkan compute shader run on our Android
// target, and does it agree with the CPU".
//
// Deliberately not a benchmark. It reports no timings: on the emulator the
// `ranchu` driver forwards to the host GPU, so any number measured here
// describes this Mac and not an Adreno or a Mali. Printing one would invite
// exactly the misreading this probe exists to avoid.
//
// The CPU reference is computed in this same process on the same device, so a
// mismatch is GPU-vs-CPU on one machine and cannot be blamed on a different
// libm or a different host.

use std::ffi::CStr;

use ash::vk;

#[path = "check.rs"]
mod check;
use check::*;

// ---------------------------------------------------------------------------
// Vulkan
// ---------------------------------------------------------------------------

const VECADD_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vecadd.spv"));
const MATMUL_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matmul.spv"));

/// `include_bytes!` gives a `&[u8]` with no alignment guarantee, and
/// `VkShaderModuleCreateInfo` wants `u32`. Copying is the honest fix; the blobs
/// are a few kilobytes.
fn spv_words(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len() % 4 == 0, "SPIR-V length not a multiple of 4");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct Buf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

struct Ctx {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    qfi: u32,
    mem_props: vk::PhysicalDeviceMemoryProperties,
}

impl Ctx {
    unsafe fn alloc(&self, bytes: usize, host_visible: bool) -> Result<Buf, String> {
        let size = bytes.max(4) as vk::DeviceSize;
        let ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = self.device.create_buffer(&ci, None).map_err(|e| format!("create_buffer: {e}"))?;
        let req = self.device.get_buffer_memory_requirements(buffer);

        let want = if host_visible {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let idx = (0..self.mem_props.memory_type_count)
            .find(|&i| {
                req.memory_type_bits & (1 << i) != 0
                    && self.mem_props.memory_types[i as usize].property_flags.contains(want)
            })
            .ok_or_else(|| format!("no memory type with {want:?}"))?;

        let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(idx);
        let memory = self.device.allocate_memory(&ai, None).map_err(|e| format!("allocate_memory: {e}"))?;
        self.device.bind_buffer_memory(buffer, memory, 0).map_err(|e| format!("bind: {e}"))?;
        Ok(Buf { buffer, memory, size })
    }

    unsafe fn write(&self, b: &Buf, data: &[f32]) -> Result<(), String> {
        let p = self
            .device
            .map_memory(b.memory, 0, b.size, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("map: {e}"))? as *mut f32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len());
        self.device.unmap_memory(b.memory);
        Ok(())
    }

    unsafe fn read(&self, b: &Buf, n: usize) -> Result<Vec<f32>, String> {
        let p = self
            .device
            .map_memory(b.memory, 0, b.size, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("map: {e}"))? as *const f32;
        let mut out = vec![0.0f32; n];
        std::ptr::copy_nonoverlapping(p, out.as_mut_ptr(), n);
        self.device.unmap_memory(b.memory);
        Ok(out)
    }

    unsafe fn free(&self, b: Buf) {
        self.device.destroy_buffer(b.buffer, None);
        self.device.free_memory(b.memory, None);
    }

    /// Bind three storage buffers, push `push` as the push-constant block, and
    /// dispatch. Everything is created and torn down per call -- this is a
    /// probe, not a runtime, and reusing objects would only add ways to be
    /// wrong about what was tested.
    unsafe fn run(
        &self,
        spv: &[u8],
        bufs: [&Buf; 3],
        push: &[u32],
        groups: (u32, u32, u32),
    ) -> Result<(), String> {
        let words = spv_words(spv);
        let sm_ci = vk::ShaderModuleCreateInfo::default().code(&words);
        let module = self.device.create_shader_module(&sm_ci, None).map_err(|e| format!("shader_module: {e}"))?;

        let bindings: Vec<_> = (0..3u32)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let dsl = self.device.create_descriptor_set_layout(&dsl_ci, None).map_err(|e| format!("dsl: {e}"))?;

        let pc_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size((push.len() * 4) as u32)];
        let set_layouts = [dsl];
        let pl_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&pc_range);
        let layout = self.device.create_pipeline_layout(&pl_ci, None).map_err(|e| format!("pipeline_layout: {e}"))?;

        let entry = CStr::from_bytes_with_nul(b"main\0").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);
        let cp_ci = [vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)];
        let pipelines = self
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &cp_ci, None)
            .map_err(|(_, e)| format!("create_compute_pipelines: {e}"))?;
        let pipeline = pipelines[0];

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3)];
        let dp_ci = vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes);
        let dpool = self.device.create_descriptor_pool(&dp_ci, None).map_err(|e| format!("desc_pool: {e}"))?;
        let ds_ai = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(dpool)
            .set_layouts(&set_layouts);
        let dset = self.device.allocate_descriptor_sets(&ds_ai).map_err(|e| format!("alloc_desc_set: {e}"))?[0];

        let infos: Vec<_> = bufs
            .iter()
            .map(|b| [vk::DescriptorBufferInfo::default().buffer(b.buffer).offset(0).range(vk::WHOLE_SIZE)])
            .collect();
        let writes: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(dset)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        self.device.update_descriptor_sets(&writes, &[]);

        let cp_ci = vk::CommandPoolCreateInfo::default().queue_family_index(self.qfi);
        let cpool = self.device.create_command_pool(&cp_ci, None).map_err(|e| format!("cmd_pool: {e}"))?;
        let cb_ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(cpool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cb = self.device.allocate_command_buffers(&cb_ai).map_err(|e| format!("alloc_cb: {e}"))?[0];

        let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.begin_command_buffer(cb, &begin).map_err(|e| format!("begin: {e}"))?;
        self.device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
        self.device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, layout, 0, &[dset], &[]);
        let push_bytes: Vec<u8> = push.iter().flat_map(|v| v.to_ne_bytes()).collect();
        self.device.cmd_push_constants(cb, layout, vk::ShaderStageFlags::COMPUTE, 0, &push_bytes);
        self.device.cmd_dispatch(cb, groups.0, groups.1, groups.2);
        self.device.end_command_buffer(cb).map_err(|e| format!("end: {e}"))?;

        let fence = self
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| format!("fence: {e}"))?;
        let cbs = [cb];
        let submit = [vk::SubmitInfo::default().command_buffers(&cbs)];
        self.device.queue_submit(self.queue, &submit, fence).map_err(|e| format!("submit: {e}"))?;
        // Ten seconds. The emulator's driver forwards to the host, so a
        // deadlock there would otherwise hang the probe with no output.
        self.device
            .wait_for_fences(&[fence], true, 10_000_000_000)
            .map_err(|e| format!("wait_for_fences: {e}"))?;

        self.device.destroy_fence(fence, None);
        self.device.destroy_command_pool(cpool, None);
        self.device.destroy_descriptor_pool(dpool, None);
        self.device.destroy_pipeline(pipeline, None);
        self.device.destroy_pipeline_layout(layout, None);
        self.device.destroy_descriptor_set_layout(dsl, None);
        self.device.destroy_shader_module(module, None);
        Ok(())
    }
}

unsafe fn init() -> Result<Ctx, String> {
    // dlopen, not a link-time dependency: on a device without Vulkan this
    // returns an error we can print instead of killing the process at exec.
    let entry = ash::Entry::load().map_err(|e| format!("failed to load libvulkan.so: {e}"))?;

    let ver = match entry.try_enumerate_instance_version() {
        Ok(Some(v)) => v,
        Ok(None) => vk::make_api_version(0, 1, 0, 0),
        Err(e) => return Err(format!("enumerate_instance_version: {e}")),
    };
    println!(
        "loader instance version: {}.{}.{}",
        vk::api_version_major(ver),
        vk::api_version_minor(ver),
        vk::api_version_patch(ver)
    );

    let app = vk::ApplicationInfo::default()
        .api_version(vk::make_api_version(0, 1, 1, 0));
    let ci = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = entry.create_instance(&ci, None).map_err(|e| format!("create_instance: {e}"))?;

    let pds = instance.enumerate_physical_devices().map_err(|e| format!("enumerate_physical_devices: {e}"))?;
    if pds.is_empty() {
        return Err("no physical devices".into());
    }
    println!("physical devices: {}", pds.len());

    let mut chosen = None;
    for &pd in &pds {
        let p = instance.get_physical_device_properties(pd);
        let name = CStr::from_ptr(p.device_name.as_ptr()).to_string_lossy().into_owned();
        println!(
            "  device: {name:?} type={:?} api={}.{}.{} driver={:#x}",
            p.device_type,
            vk::api_version_major(p.api_version),
            vk::api_version_minor(p.api_version),
            vk::api_version_patch(p.api_version),
            p.driver_version
        );
        println!(
            "    maxComputeWorkGroupInvocations={} maxComputeSharedMemorySize={} maxStorageBufferRange={}",
            p.limits.max_compute_work_group_invocations,
            p.limits.max_compute_shared_memory_size,
            p.limits.max_storage_buffer_range
        );
        let qfs = instance.get_physical_device_queue_family_properties(pd);
        for (i, qf) in qfs.iter().enumerate() {
            println!("    queue family {i}: count={} flags={:?}", qf.queue_count, qf.queue_flags);
        }
        if chosen.is_none() {
            if let Some(i) = qfs.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)) {
                chosen = Some((pd, i as u32, name));
            }
        }
    }
    let (pd, qfi, name) = chosen.ok_or("no queue family supports COMPUTE")?;
    println!("using {name:?} queue family {qfi}");

    // What a quantised kernel would need. Reported, not required -- this probe
    // is f32 only.
    //
    // The extension list alone is not the answer and reading it as one would be
    // a mistake: VK_KHR_16bit_storage, VK_KHR_variable_pointers and
    // VK_KHR_storage_buffer_storage_class were all promoted into Vulkan 1.1
    // core, so on a 1.1+ device they are *absent from the extension list
    // precisely because they are always available*. The feature bits below are
    // the real answer, and they are what gets reported as the verdict.
    match instance.enumerate_device_extension_properties(pd) {
        Ok(exts) => {
            let names: Vec<String> = exts
                .iter()
                .map(|e| CStr::from_ptr(e.extension_name.as_ptr()).to_string_lossy().into_owned())
                .collect();
            println!("device extensions: {} (promoted-to-core ones do not appear here)", names.len());
            for want in ["VK_KHR_shader_float16_int8", "VK_KHR_8bit_storage", "VK_KHR_cooperative_matrix"] {
                println!("  ext {:<40} {}", want, if names.iter().any(|n| n == want) { "present" } else { "absent" });
            }
        }
        Err(e) => println!("device extensions: query failed: {e}"),
    }

    let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    // `f2` borrows both chained structs for as long as it lives, so it is
    // scoped and the bits are copied out.
    let base = {
        let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f11).push_next(&mut f12);
        instance.get_physical_device_features2(pd, &mut f2);
        f2.features
    };
    let yn = |b: vk::Bool32| if b == vk::TRUE { "YES" } else { "no" };
    println!("quantisation-relevant feature bits:");
    println!("  storageBuffer16BitAccess (fp16/i16 in an SSBO) {}", yn(f11.storage_buffer16_bit_access));
    println!("  storageBuffer8BitAccess  (int8 in an SSBO)     {}", yn(f12.storage_buffer8_bit_access));
    println!("  shaderFloat16            (fp16 arithmetic)     {}", yn(f12.shader_float16));
    println!("  shaderInt8               (int8 arithmetic)     {}", yn(f12.shader_int8));
    println!("  shaderInt16              (int16 arithmetic)    {}", yn(base.shader_int16));

    let prio = [1.0f32];
    let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qfi).queue_priorities(&prio)];
    let dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci);
    let device = instance.create_device(pd, &dci, None).map_err(|e| format!("create_device: {e}"))?;
    let queue = device.get_device_queue(qfi, 0);
    let mem_props = instance.get_physical_device_memory_properties(pd);

    Ok(Ctx { _entry: entry, instance, device, queue, qfi, mem_props })
}

fn main() {
    println!("vk_probe: Vulkan compute vs CPU, {} {}", std::env::consts::OS, std::env::consts::ARCH);
    let tampering = tamper_ulp().is_some();
    if let Some(n) = tamper_ulp() {
        println!("VK_PROBE_TAMPER={n}: reference perturbed by {n} ULP, every case MUST report MISMATCH");
    }

    let ctx = match unsafe { init() } {
        Ok(c) => c,
        Err(e) => {
            println!("RESULT: FAIL (init) -- {e}");
            std::process::exit(2);
        }
    };

    let mut all_ok = true;

    // --- vector add -------------------------------------------------------
    {
        let n = 1 << 20;
        let a = fill(1, n);
        let b = fill(2, n);
        let mut want = cpu_vecadd(&a, &b);
        tamper(&mut want);
        println!("vecadd n={n}");
        // An immediately-invoked closure so `?` has somewhere to return to
        // without giving up the per-case error reporting below.
        let run = || -> Result<Vec<f32>, String> {
            unsafe {
                let ba = ctx.alloc(n * 4, true)?;
                let bb = ctx.alloc(n * 4, true)?;
                let bc = ctx.alloc(n * 4, true)?;
                ctx.write(&ba, &a)?;
                ctx.write(&bb, &b)?;
                ctx.run(VECADD_SPV, [&ba, &bb, &bc], &[n as u32], (n.div_ceil(64) as u32, 1, 1))?;
                let got = ctx.read(&bc, n)?;
                ctx.free(ba);
                ctx.free(bb);
                ctx.free(bc);
                Ok(got)
            }
        };
        match run() {
            Ok(got) => all_ok &= report("vecadd", &compare(&got, &want)),
            Err(e) => {
                println!("  vecadd: ERROR {e}");
                all_ok = false;
            }
        }
    }

    // --- matmul, square and non-square ------------------------------------
    // The non-square case is the one that catches a transposed index: with
    // m == n == k a row-major/column-major mixup can still produce plausible
    // numbers, and with 96x64x80 it cannot.
    for &(m, n, k) in &[(64usize, 64usize, 64usize), (96, 64, 80)] {
        let a = fill(3, m * k);
        let b = fill(4, k * n);
        let mut want = cpu_matmul(&a, &b, m, n, k);
        tamper(&mut want);
        println!("matmul {m}x{k} * {k}x{n}");
        let run = || -> Result<Vec<f32>, String> {
            unsafe {
                let ba = ctx.alloc(m * k * 4, true)?;
                let bb = ctx.alloc(k * n * 4, true)?;
                let bc = ctx.alloc(m * n * 4, true)?;
                ctx.write(&ba, &a)?;
                ctx.write(&bb, &b)?;
                ctx.run(
                    MATMUL_SPV,
                    [&ba, &bb, &bc],
                    &[m as u32, n as u32, k as u32],
                    (n.div_ceil(8) as u32, m.div_ceil(8) as u32, 1),
                )?;
                let got = ctx.read(&bc, m * n)?;
                ctx.free(ba);
                ctx.free(bb);
                ctx.free(bc);
                Ok(got)
            }
        };
        match run() {
            Ok(got) => all_ok &= report(&format!("matmul {m}x{k}x{n}"), &compare(&got, &want)),
            Err(e) => {
                println!("  matmul: ERROR {e}");
                all_ok = false;
            }
        }
    }

    unsafe {
        ctx.device.destroy_device(None);
        ctx.instance.destroy_instance(None);
    }

    // adb shell does not carry the exit code back, so the verdict has to be in
    // the output. It is printed as a single greppable line.
    // Under tamper the pass/fail sense is inverted: the run is correct exactly
    // when every case noticed the perturbation.
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
