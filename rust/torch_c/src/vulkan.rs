//! The `vulkan` device: a tensor representation that lives outside candle.
//!
//! `docs/VULKAN2.md` §5 sized this and deliberately wrote no code. Its finding
//! is the shape of this file: `candle_core::Device` is a closed enum with
//! nowhere to put a Vulkan handle, so a Vulkan tensor **cannot be a
//! `candle::Tensor` wearing a label**. It has to be a fourth arm of
//! `tensor::Repr`, exactly as `Repr::Quantized` is a third arm for the same
//! structural reason (candle's `QTensor` is a separate type system).
//!
//! **That inheritance is the single most important property here and it must
//! not be weakened.** `PyTensorBase::tensor()` refuses on every non-`Dense`
//! arm, and there are 396 call sites of it in this crate. So a kernel that has
//! not been taught the Vulkan arm *cannot* read CPU storage off a Vulkan
//! tensor by forgetting to check -- it gets a `PyResult` it has to handle, and
//! the only thing it can do with it is raise. **A silent CPU fallback is
//! structurally impossible**, which is what `docs/VULKAN.md` §5 names as the
//! worst available outcome. Ops opt in one at a time, by name, in `dispatch`
//! below; everything else refuses *naming the op*.
//!
//! What is deliberately not here:
//!
//! * **No performance claim of any kind.** The only Vulkan driver reachable on
//!   this machine is `kosmickrisp`, a Vulkan-on-Metal translation layer, so a
//!   number measured here would describe the translator. `docs/VULKAN2.md` §4.4
//!   says so and nothing in this file or its tests times anything.
//! * **No fallback.** If the loader is absent, `ones(..., device="vulkan")`
//!   raises with the loader's own error text. It never quietly returns a CPU
//!   tensor.
//! * **f32 only, contiguous only, same-shape only.** Every other case refuses
//!   by name rather than being approximated.
//!
//! ## Where the loader comes from
//!
//! Nothing is installed. `ash::Entry::load()` *dlopens* the loader rather than
//! linking it, so this module compiles and the artefact loads on a machine with
//! no Vulkan at all -- absence is a value that gets reported, not a link error.
//! On this host the loader and ICDs already exist inside the Android emulator's
//! bundle (`docs/VULKAN2.md` §4), and pointing `DYLD_LIBRARY_PATH` and
//! `VK_DRIVER_FILES` at `libkosmickrisp_icd.json` gives the real `Apple M1`
//! GPU. **Those files are the Android SDK's private implementation detail and
//! this is a development/test path, never a shipping one.**

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ash::vk;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};

use crate::device::PyDevice;
use crate::dtype::TorchDType;
use crate::err::not_implemented;
use crate::tensor::PyTensorBase;

/// The one kernel this round ships, as SPIR-V words.
///
/// Checked in beside its GLSL rather than compiled by a `build.rs`: this crate
/// builds for Android, iOS and the host, and making all three depend on a
/// shader compiler being on PATH for a file that changes about once a round is
/// a bad trade. `shaders/compile.sh` regenerates it and
/// `test_the_checked_in_spirv_is_not_stale` is the guard that an edit to the
/// `.comp` was actually compiled.
const ADD_F32_SPV: &[u8] = include_bytes!("../shaders/add_f32.spv");

macro_rules! spv {
    ($name:ident, $file:literal) => {
        const $name: &[u8] = include_bytes!(concat!("../shaders/", $file, ".spv"));
    };
}
spv!(SUB_F32_SPV, "sub_f32");
spv!(MUL_F32_SPV, "mul_f32");
spv!(DIV_F32_SPV, "div_f32");
spv!(RELU_F32_SPV, "relu_f32");
spv!(NEG_F32_SPV, "neg_f32");
spv!(COPY_F32_SPV, "copy_f32");
spv!(TRANSPOSE2D_F32_SPV, "transpose2d_f32");
spv!(MATMUL_F32_SPV, "matmul_f32");
spv!(BIAS_ADD_F32_SPV, "bias_add_f32");

// ---------------------------------------------------------------------------
// The instrument: how "it ran on the GPU" stops being an inference
// ---------------------------------------------------------------------------

/// Three process-wide counters, incremented at the only three places where
/// this module can cross the host/device boundary or make the GPU do work.
///
/// **Why these exist rather than a source-scanning test.** `docs/MPSATTN.md`
/// §3.1 records, against its own round, a way to take an op off a refusal list
/// while keeping its host readback that *both* of that device's derivation
/// tests would still pass: the per-op scan looks for six helper names in a
/// kernel body and the classification test looks for three markers, so moving
/// the readback one call deeper -- into a helper named something else -- is
/// invisible to both. Every check of that shape reads the source and can be
/// defeated by moving the thing it greps for.
///
/// These counters cannot be defeated that way, because they are not a
/// description of the code: `SHADER_DISPATCHES` is incremented inside
/// `dispatch_kernel`, after `vkWaitForFences` has returned success, and
/// `HOST_DOWNLOADS` is incremented inside `download`, which is the only
/// `vkMapMemory`-for-reading in the module. An op that computed on the host
/// would have to read its operands, and reading them goes through `download`
/// however many helpers deep it is buried. So the test asserts, at runtime and
/// per op, **`shader_dispatches` went up by the expected number and
/// `host_downloads` did not move at all** -- which is a statement about what
/// the process did, not about what the source looks like.
static SHADER_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static HOST_UPLOADS: AtomicU64 = AtomicU64::new(0);
static HOST_DOWNLOADS: AtomicU64 = AtomicU64::new(0);

fn push_bytes(push: [u32; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (i, v) in push.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Instance, device, queue and the two caches, created once per process.
///
/// **Lifetime is the piece `vk_probe` does not have and `docs/VULKAN2.md` §5.2
/// item 2 names.** The probe builds an instance, a pipeline and a command pool
/// per dispatch and tears them all down again, which is correct for a probe and
/// indefensible for a runtime: creating a `VkDevice` per tensor op would put a
/// driver-side allocation on the hot path and lose the pipeline cache entirely.
/// Here the context is a process-lifetime singleton and pipelines are built
/// once per kernel.
///
/// It is never destroyed, and that is a decision rather than an omission.
/// `vkDestroyDevice` at process exit would have to run *after* every
/// `Repr::Vulkan` tensor Python still holds has been dropped, and Python
/// guarantees no such ordering for module teardown. Leaking the device at exit
/// is what the driver already survives (the OS reclaims it); freeing it while a
/// live `VkBuffer` still names it is a use-after-free. So `Drop` frees buffers
/// and nothing frees the device.
pub struct VkContext {
    // Kept alive because the `Instance`/`Device` function pointers were loaded
    // out of it; dropping it would unload the library underneath them.
    _entry: ash::Entry,
    // Kept for the same reason `_entry` is: the `Device`'s function pointers
    // were loaded out of it. Never called after `init`, hence the underscore.
    _instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    qfi: u32,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    /// For the report `_vulkan_probe()` gives Python, so a skipped test can say
    /// which driver it skipped and a passing one can say what it ran on.
    pub device_name: String,
    pub device_type: String,
    /// Vulkan requires *external* synchronisation on a queue and on a command
    /// pool -- the driver does no locking of its own. One mutex covers both,
    /// which is right while there is one queue: the critical section is the
    /// whole record-submit-wait, so a finer lock would buy nothing.
    submit: Mutex<vk::CommandPool>,
    /// The pipeline cache of `docs/VULKAN2.md` §5.2 item 3. Keyed by kernel
    /// name; a `&'static str` because the set of kernels is closed and authored
    /// here, not discovered.
    pipelines: Mutex<HashMap<&'static str, Kernel>>,
}

/// Everything creating a compute pipeline produces, kept so it is created once.
#[derive(Clone, Copy)]
struct Kernel {
    dsl: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    dpool: vk::DescriptorPool,
}

// The handles above are opaque `u64`s and `ash` marks them `Send`/`Sync`; the
// mutability that Vulkan requires to be externally synchronised is behind the
// two mutexes.
unsafe impl Send for VkContext {}
unsafe impl Sync for VkContext {}

static CONTEXT: OnceLock<Result<VkContext, String>> = OnceLock::new();

/// The process's Vulkan context, or why there isn't one.
///
/// The error is a `String` and it is the *driver's* message, not ours: on a
/// machine with no loader the useful thing to print is `dlopen`'s own text.
/// `OnceLock` means a machine with no Vulkan pays one failed `dlopen` for the
/// life of the process rather than one per call.
pub fn context() -> Result<&'static VkContext, &'static str> {
    match CONTEXT.get_or_init(|| unsafe { init() }) {
        Ok(ctx) => Ok(ctx),
        Err(message) => Err(message.as_str()),
    }
}

/// The context, or a `NotImplementedError` carrying the driver's reason.
fn require(op: &str) -> PyResult<&'static VkContext> {
    context().map_err(|reason| {
        not_implemented(format!(
            "{op}: the vulkan device is not available in this process -- {reason}. \
             Nothing falls back to the CPU here (docs/VULKAN.md §5): a vulkan \
             tensor is a VkBuffer, so there is nothing to fall back with."
        ))
    })
}

unsafe fn init() -> Result<VkContext, String> {
    let entry = ash::Entry::load().map_err(|e| format!("failed to load the Vulkan loader: {e}"))?;
    let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 1, 0));
    let ci = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = entry
        .create_instance(&ci, None)
        .map_err(|e| format!("vkCreateInstance: {e}"))?;

    let pds = instance
        .enumerate_physical_devices()
        .map_err(|e| format!("vkEnumeratePhysicalDevices: {e}"))?;
    // Prefer a real GPU over a software rasteriser when both ICDs are loaded.
    // Both are useful -- lavapipe exercises the whole stack on a machine with
    // no GPU -- but if an `Apple M1` is enumerated it is the one to run on, and
    // `_vulkan_probe()` reports which was chosen so a test can say so.
    let mut best: Option<(vk::PhysicalDevice, u32, String, String, u32)> = None;
    for &pd in &pds {
        let p = instance.get_physical_device_properties(pd);
        let qfs = instance.get_physical_device_queue_family_properties(pd);
        let Some(qfi) = qfs
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        else {
            continue;
        };
        let name = CStr::from_ptr(p.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();
        let rank = match p.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
            _ => 0,
        };
        let better = match &best {
            None => true,
            Some(b) => rank > b.4,
        };
        if better {
            best = Some((pd, qfi as u32, name, format!("{:?}", p.device_type), rank));
        }
    }
    let (pd, qfi, device_name, device_type, _) =
        best.ok_or_else(|| "no physical device has a COMPUTE queue family".to_string())?;

    let prio = [1.0f32];
    let qci = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(qfi)
        .queue_priorities(&prio)];
    let dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci);
    let device = instance
        .create_device(pd, &dci, None)
        .map_err(|e| format!("vkCreateDevice: {e}"))?;
    let queue = device.get_device_queue(qfi, 0);
    let mem_props = instance.get_physical_device_memory_properties(pd);

    let cp_ci = vk::CommandPoolCreateInfo::default()
        .queue_family_index(qfi)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let pool = device
        .create_command_pool(&cp_ci, None)
        .map_err(|e| format!("vkCreateCommandPool: {e}"))?;

    Ok(VkContext {
        _entry: entry,
        _instance: instance,
        device,
        queue,
        qfi,
        mem_props,
        device_name,
        device_type,
        submit: Mutex::new(pool),
        pipelines: Mutex::new(HashMap::new()),
    })
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

/// One device allocation, freed when the last tensor naming it is dropped.
///
/// **One allocation per tensor, and that is stated as a limitation rather than
/// presented as an allocator.** `docs/VULKAN2.md` §5.2 item 3 asks for a
/// suballocating allocator, and this is not one: a real one matters because
/// `maxMemoryAllocationCount` is a few thousand on desktop drivers and can be
/// 4096 on mobile, so a model's worth of weights would exhaust it. It is the
/// right shape for a two-by-two and the wrong shape for a model, and the next
/// round is where that changes.
pub struct VkBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: usize,
}

unsafe impl Send for VkBuffer {}
unsafe impl Sync for VkBuffer {}

impl Drop for VkBuffer {
    fn drop(&mut self) {
        // The context outlives every buffer by construction (it is a
        // `OnceLock` that is never cleared), so this cannot be `None` for a
        // buffer that exists -- a buffer can only have been made through it.
        if let Ok(ctx) = context() {
            unsafe {
                ctx.device.destroy_buffer(self.buffer, None);
                ctx.device.free_memory(self.memory, None);
            }
        }
    }
}

/// A tensor whose bytes are on the GPU. The fourth arm of `Repr`.
///
/// `Arc` because `Repr` is `Clone` and cloning a tensor here must not copy a
/// GPU allocation. Two clones therefore *share* storage, which matches candle's
/// `Dense` arm (a candle clone is an `Arc` clone) -- and it is safe in the same
/// way, because nothing on this device is in-place: every taught op allocates
/// its output.
#[derive(Clone)]
pub struct VkTensor {
    pub buffer: Arc<VkBuffer>,
    pub shape: Vec<usize>,
}

impl VkTensor {
    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }
}

impl VkContext {
    /// Allocate `bytes` of host-visible device memory.
    ///
    /// **Host-visible, and on this machine that is not a compromise.** The
    /// `Apple M1` reports as `INTEGRATED_GPU` with unified memory, so the
    /// memory type that is `DEVICE_LOCAL` is also `HOST_VISIBLE` and the map is
    /// a pointer into the same bytes the shader reads -- the upload is real and
    /// the copy is the only one there is. The `DEVICE_LOCAL` bit is *preferred*
    /// and falls back to plain host-visible, so a discrete GPU still works and
    /// is simply slower than it should be; that is where a staging buffer plus
    /// `vkCmdCopyBuffer` belongs, and it is not here.
    unsafe fn alloc(&self, bytes: usize) -> Result<VkBuffer, String> {
        let size = bytes.max(4) as vk::DeviceSize;
        let ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = self
            .device
            .create_buffer(&ci, None)
            .map_err(|e| format!("vkCreateBuffer: {e}"))?;
        let req = self.device.get_buffer_memory_requirements(buffer);

        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mut chosen = None;
        for want in [host | vk::MemoryPropertyFlags::DEVICE_LOCAL, host] {
            chosen = (0..self.mem_props.memory_type_count).find(|&i| {
                req.memory_type_bits & (1 << i) != 0
                    && self.mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(want)
            });
            if chosen.is_some() {
                break;
            }
        }
        let Some(idx) = chosen else {
            self.device.destroy_buffer(buffer, None);
            return Err("no host-visible coherent memory type".to_string());
        };

        let ai = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(idx);
        let memory = match self.device.allocate_memory(&ai, None) {
            Ok(m) => m,
            Err(e) => {
                self.device.destroy_buffer(buffer, None);
                return Err(format!("vkAllocateMemory: {e}"));
            }
        };
        if let Err(e) = self.device.bind_buffer_memory(buffer, memory, 0) {
            self.device.destroy_buffer(buffer, None);
            self.device.free_memory(memory, None);
            return Err(format!("vkBindBufferMemory: {e}"));
        }
        Ok(VkBuffer {
            buffer,
            memory,
            bytes: req.size as usize,
        })
    }

    /// Host to device. The upload half of `docs/VULKAN2.md` §5.2 item 5.
    unsafe fn upload(&self, buf: &VkBuffer, data: &[f32]) -> Result<(), String> {
        let p = self
            .device
            .map_memory(
                buf.memory,
                0,
                buf.bytes as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("vkMapMemory: {e}"))? as *mut f32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len());
        self.device.unmap_memory(buf.memory);
        HOST_UPLOADS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Device to host. The download half -- what `.cpu()` runs.
    unsafe fn download(&self, buf: &VkBuffer, n: usize) -> Result<Vec<f32>, String> {
        let p = self
            .device
            .map_memory(
                buf.memory,
                0,
                buf.bytes as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| format!("vkMapMemory: {e}"))? as *const f32;
        let mut out = vec![0.0f32; n];
        std::ptr::copy_nonoverlapping(p, out.as_mut_ptr(), n);
        self.device.unmap_memory(buf.memory);
        HOST_DOWNLOADS.fetch_add(1, Ordering::Relaxed);
        Ok(out)
    }

    /// Build (or fetch) the compute pipeline for a kernel.
    ///
    /// Three storage-buffer bindings and a `uint` push constant is the whole
    /// interface every kernel here uses, so the layout is shared rather than
    /// described per kernel.
    unsafe fn kernel(&self, name: &'static str, spv: &[u8]) -> Result<Kernel, String> {
        let mut cache = self.pipelines.lock().map_err(|_| "pipeline cache poisoned")?;
        if let Some(k) = cache.get(name) {
            return Ok(*k);
        }
        // `include_bytes!` has no alignment guarantee and
        // `VkShaderModuleCreateInfo` wants `u32`; copying is the honest fix and
        // it happens once per kernel per process.
        if spv.len() % 4 != 0 {
            return Err(format!("{name}: SPIR-V length is not a multiple of 4"));
        }
        let words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let module = self
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .map_err(|e| format!("vkCreateShaderModule({name}): {e}"))?;

        let bindings: Vec<_> = (0..3u32)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl = self
            .device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .map_err(|e| format!("vkCreateDescriptorSetLayout({name}): {e}"))?;

        // Sixteen bytes, not four. Every kernel here declares
        // `Push { uint n; uint p1; uint p2; uint p3; }` and most of them use
        // only `n`; a shader that reads fewer bytes than the range declares is
        // legal, which is why widening this did not require recompiling
        // `add_f32.spv` -- and `shaders/compile.sh` reproduced that file
        // byte-for-byte, which is the control that the toolchain here is the
        // one that produced the committed kernel (docs/VULKAN4.md §3).
        let pc = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)];
        let set_layouts = [dsl];
        let layout = self
            .device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&pc),
                None,
            )
            .map_err(|e| format!("vkCreatePipelineLayout({name}): {e}"))?;

        let entry = CStr::from_bytes_with_nul(b"main\0").expect("literal");
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);
        let cp = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout)];
        let pipeline = self
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &cp, None)
            .map_err(|(_, e)| format!("vkCreateComputePipelines({name}): {e}"))?[0];
        // The module is only needed while the pipeline is being created.
        self.device.destroy_shader_module(module, None);

        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(3 * 64)];
        let dpool = self
            .device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(64)
                    .pool_sizes(&sizes),
                None,
            )
            .map_err(|e| format!("vkCreateDescriptorPool({name}): {e}"))?;

        let k = Kernel {
            dsl,
            layout,
            pipeline,
            dpool,
        };
        cache.insert(name, k);
        Ok(k)
    }

    /// Bind three buffers, push `n`, dispatch, wait.
    ///
    /// Synchronous on purpose: an asynchronous device needs a stream and an
    /// event per tensor, and a half-built one is a class of bug (reading a
    /// buffer whose dispatch has not retired) that produces *plausible wrong
    /// numbers*. The fence here is the same one the probe used and it means
    /// every tensor this device produces is complete when it is returned.
    unsafe fn dispatch_kernel(
        &self,
        name: &'static str,
        spv: &[u8],
        bufs: [&VkBuffer; 3],
        push: [u32; 4],
    ) -> Result<(), String> {
        let k = self.kernel(name, spv)?;
        let pool = self.submit.lock().map_err(|_| "submit lock poisoned")?;

        let set_layouts = [k.dsl];
        let dset = self
            .device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(k.dpool)
                    .set_layouts(&set_layouts),
            )
            .map_err(|e| format!("vkAllocateDescriptorSets: {e}"))?[0];
        let infos: Vec<_> = bufs
            .iter()
            .map(|b| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)]
            })
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

        let cb = self
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(*pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| format!("vkAllocateCommandBuffers: {e}"))?[0];
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device
            .begin_command_buffer(cb, &begin)
            .map_err(|e| format!("vkBeginCommandBuffer: {e}"))?;
        self.device
            .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, k.pipeline);
        self.device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            k.layout,
            0,
            &[dset],
            &[],
        );
        self.device.cmd_push_constants(
            cb,
            k.layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &push_bytes(push),
        );
        // One invocation per output element is the convention every `.comp` in
        // `shaders/` follows, so the grid is sized from `push[0]` alone.
        self.device.cmd_dispatch(cb, push[0].div_ceil(64), 1, 1);
        self.device
            .end_command_buffer(cb)
            .map_err(|e| format!("vkEndCommandBuffer: {e}"))?;

        let fence = self
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| format!("vkCreateFence: {e}"))?;
        let cbs = [cb];
        let submit = [vk::SubmitInfo::default().command_buffers(&cbs)];
        let result = self
            .device
            .queue_submit(self.queue, &submit, fence)
            .map_err(|e| format!("vkQueueSubmit: {e}"))
            .and_then(|()| {
                // Ten seconds, for the same reason the probe has a timeout: a
                // translation layer that deadlocks would otherwise hang the
                // interpreter with no output.
                self.device
                    .wait_for_fences(&[fence], true, 10_000_000_000)
                    .map_err(|e| format!("vkWaitForFences: {e}"))
            });
        self.device.destroy_fence(fence, None);
        self.device.free_command_buffers(*pool, &cbs);
        let _ = self.device.free_descriptor_sets(k.dpool, &[dset]);
        drop(pool);
        // After the fence, and only on success: the counter means "a compute
        // shader ran to completion on the device", not "one was submitted".
        if result.is_ok() {
            SHADER_DISPATCHES.fetch_add(1, Ordering::Relaxed);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// The pieces the rest of the crate calls
// ---------------------------------------------------------------------------

/// A tensor of `size` filled with `value`, on the GPU.
///
/// This is `torch.ones(2, 2, device="vulkan")`, and there is no kernel in it:
/// the fill is built on the host and *uploaded*, which is the honest minimum --
/// the bytes really do end up in a `VkBuffer` and really do come back out of
/// one through `.cpu()`. A `fill` shader would move the loop to the GPU and
/// prove nothing more about the wiring; `docs/VULKAN2.md` §5.3 says the same.
pub fn factory(
    py: Python<'_>,
    op: &str,
    size: Vec<usize>,
    tag: TorchDType,
    value: f32,
) -> PyResult<Py<PyAny>> {
    let ctx = require(op)?;
    check_dtype(op, tag)?;
    let n: usize = size.iter().product();
    let data = vec![value; n];
    let buffer = unsafe {
        let buf = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        ctx.upload(&buf, &data).map_err(|e| vk_error(op, e))?;
        buf
    };
    let vk_tensor = VkTensor {
        buffer: Arc::new(buffer),
        shape: size,
    };
    let wrapped = PyTensorBase::vulkan(vk_tensor, tag);
    crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())
}

/// f32 is the only dtype with a kernel, and every other one refuses here rather
/// than being silently widened or narrowed.
fn check_dtype(op: &str, tag: TorchDType) -> PyResult<()> {
    if tag == TorchDType::Float32 {
        return Ok(());
    }
    Err(not_implemented(format!(
        "{op}: the vulkan device in this build stores float32 only, not {}. \
         The shader is f32 and there is no conversion path that would not be a \
         silent one (docs/VULKAN3.md).",
        tag.name()
    )))
}

fn vk_error(op: &str, message: String) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("{op}: vulkan: {message}"))
}

/// Bring a Vulkan tensor's bytes back to the CPU as a candle tensor.
pub fn to_cpu(op: &str, vk_tensor: &VkTensor) -> PyResult<candle_core::Tensor> {
    let ctx = require(op)?;
    let n = vk_tensor.elem_count();
    let host = unsafe { ctx.download(&vk_tensor.buffer, n) }.map_err(|e| vk_error(op, e))?;
    candle_core::Tensor::from_vec(host, vk_tensor.shape.clone(), &candle_core::Device::Cpu)
        .map_err(|e| crate::err::candle_err(op, e))
}

/// The `vulkan` half of the dispatcher, and the *only* way an op computes on a
/// Vulkan tensor.
///
/// Structured exactly like `aten.rs::meta_dispatch` and for the same reason:
/// the answer to "does this op work on the vulkan device?" is a list here, not
/// a reading of ninety-odd kernels. The `other` arm is what makes the whole
/// design honest -- **an op that has not been taught this device refuses and
/// names itself.**
pub fn dispatch(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    match op {
        "aten._to_copy.default" => to_copy(py, op, args, kwargs),

        // Elementwise, equal shapes, f32. One exactly-rounded IEEE operation
        // per element, so each of these is compared bit-for-bit against the
        // CPU kernel rather than within a tolerance (docs/VULKAN4.md §4).
        "aten.add.Tensor" => add_tensor(py, op, args, kwargs),
        "aten.sub.Tensor" => binary(py, op, args, kwargs, "sub_f32", SUB_F32_SPV, true),
        "aten.mul.Tensor" => binary(py, op, args, kwargs, "mul_f32", MUL_F32_SPV, false),
        "aten.div.Tensor" => binary(py, op, args, kwargs, "div_f32", DIV_F32_SPV, false),

        // Unary elementwise.
        "aten.relu.default" => unary(py, op, args, kwargs, "relu_f32", RELU_F32_SPV),
        "aten.neg.default" => unary(py, op, args, kwargs, "neg_f32", NEG_F32_SPV),

        // `clone` allocates and copies on the device. `detach`/`alias` share
        // the buffer, which is what the dense arm does too (a candle clone is
        // an `Arc` clone) and is safe here for a stronger reason: no op on
        // this device writes in place.
        "aten.clone.default" => unary(py, op, args, kwargs, "copy_f32", COPY_F32_SPV),
        "aten.detach.default" | "aten.alias.default" | "aten.contiguous.default" => {
            let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
            let vk_tensor = input.vk_tensor(op)?.clone();
            let out = PyTensorBase::vulkan(vk_tensor, input.tag());
            crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind())
        }

        // Shape-only. Every tensor on this device is contiguous by
        // construction -- there are no strides to be non-contiguous with --
        // so a reshape is a new `shape` over the same buffer and moves no
        // bytes. `_vulkan_counters()` shows zero shader dispatches for these,
        // which is the honest answer and not a claim that the GPU did work.
        "aten.view.default" | "aten._unsafe_view.default" | "aten.reshape.default" => {
            reshape(py, op, args, kwargs)
        }

        // Transpose *materialises* here, because a `VkTensor` has a shape and
        // no strides. 2-D only; higher ranks refuse by name (docs/VULKAN4.md §6).
        "aten.t.default" => t_default(py, op, args, kwargs),
        "aten.transpose.int" => transpose_int(py, op, args, kwargs),

        // The matmuls -- the ops the measured trace says a forward pass
        // actually spends itself on (docs/VULKAN4.md §2).
        "aten.mm.default" => mm(py, op, args, kwargs),
        "aten.addmm.default" => addmm(py, op, args, kwargs),

        other => Err(not_implemented(format!(
            "{other}: not implemented for the vulkan device. This build teaches \
             the vulkan device {} ops by name -- {} -- and every other op \
             refuses here rather than falling back to the CPU \
             (docs/VULKAN4.md). Move the tensor with .cpu() to compute {other}.",
            vulkan_ops().len(),
            vulkan_ops().join(", ")
        ))),
    }
}

// ---------------------------------------------------------------------------
// The taught ops
// ---------------------------------------------------------------------------

/// Every op here is f32-only, and this is where that is enforced for a whole
/// tensor list rather than once per call site.
fn check_all_f32(op: &str, tensors: &[&PyTensorBase]) -> PyResult<()> {
    for t in tensors {
        check_dtype(op, t.tag())?;
    }
    Ok(())
}

/// `self` for an op whose only tensor argument is the receiver.
fn self_vk(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(PyTensorBase, VkTensor)> {
    let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    check_dtype(op, input.tag())?;
    let vk = input.vk_tensor(op)?.clone();
    Ok((input, vk))
}

/// Wrap a freshly produced buffer as a tensor of `shape` with `tag`.
fn wrap(
    py: Python<'_>,
    buffer: VkBuffer,
    shape: Vec<usize>,
    tag: TorchDType,
) -> PyResult<Py<PyAny>> {
    let out = PyTensorBase::vulkan(
        VkTensor {
            buffer: Arc::new(buffer),
            shape,
        },
        tag,
    );
    crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind())
}

/// One input, one output, same shape.
fn unary(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    name: &'static str,
    spv: &'static [u8],
) -> PyResult<Py<PyAny>> {
    let (input, a) = self_vk(op, args, kwargs)?;
    let ctx = require(op)?;
    let n = a.elem_count();
    let out = unsafe {
        let out = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        // Binding 1 is the input again: the descriptor set layout is three
        // storage buffers for every kernel here and the shader declares the
        // slot it does not read (`shaders/*_f32.comp`).
        ctx.dispatch_kernel(name, spv, [&a.buffer, &a.buffer, &out], [n as u32, 0, 0, 0])
            .map_err(|e| vk_error(op, e))?;
        out
    };
    wrap(py, out, a.shape.clone(), input.tag())
}

/// Two inputs of equal shape, one output. `alpha` is accepted only when it is
/// 1, for the ops that have one.
fn binary(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    name: &'static str,
    spv: &'static [u8],
    has_alpha: bool,
) -> PyResult<Py<PyAny>> {
    let lhs = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = crate::aten::tensor_arg(op, args, kwargs, 1, "other")?;
    if has_alpha {
        reject_alpha(op, args, kwargs)?;
    }
    check_all_f32(op, &[&lhs, &rhs])?;
    let a = lhs.vk_tensor(op)?.clone();
    let b = rhs.vk_tensor(op)?.clone();
    same_shape(op, &a, &b)?;
    let ctx = require(op)?;
    let n = a.elem_count();
    let out = unsafe {
        let out = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel(name, spv, [&a.buffer, &b.buffer, &out], [n as u32, 0, 0, 0])
            .map_err(|e| vk_error(op, e))?;
        out
    };
    wrap(py, out, a.shape.clone(), lhs.tag())
}

/// Broadcasting is refused rather than emulated, by name and with both shapes
/// in the message. `docs/VULKAN3.md` is explicit that reaching for the CPU
/// implementation half a metre away is exactly the silent fallback this device
/// exists to make impossible.
fn same_shape(op: &str, a: &VkTensor, b: &VkTensor) -> PyResult<()> {
    if a.shape != b.shape {
        return Err(not_implemented(format!(
            "{op}: the vulkan kernels are elementwise over equal shapes and do \
             not broadcast {:?} with {:?} (docs/VULKAN4.md §6).",
            a.shape, b.shape
        )));
    }
    Ok(())
}

fn reject_alpha(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if let Some(alpha) = crate::aten::optional(args, kwargs, 2, "alpha")? {
        if !alpha.is_none() && alpha.extract::<f64>().unwrap_or(1.0) != 1.0 {
            return Err(not_implemented(format!(
                "{op}: the vulkan kernel has no alpha (docs/VULKAN4.md §6)."
            )));
        }
    }
    Ok(())
}

/// `view` / `_unsafe_view` / `reshape`: a new shape over the same buffer.
fn reshape(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (input, a) = self_vk(op, args, kwargs)?;
    // `aten.view` and `aten._unsafe_view` spell the argument `size`;
    // `aten.reshape` spells it `shape`. Both are accepted rather than one
    // being assumed -- the first draft assumed `size` and `t.reshape(-1)`
    // raised "missing argument 'size'" on a tensor that had one.
    let requested = match crate::aten::optional(args, kwargs, 1, "size")? {
        Some(v) if !v.is_none() => Some(v),
        _ => crate::aten::optional(args, kwargs, 1, "shape")?,
    };
    let requested = requested
        .filter(|v| !v.is_none())
        .ok_or_else(|| not_implemented(format!("{op}: vulkan: missing argument 'size'")))?;
    let requested: Vec<i64> = match requested.extract::<Vec<i64>>() {
        Ok(v) => v,
        Err(_) => vec![requested.extract::<i64>()?],
    };
    let n = a.elem_count();
    let shape = resolve_shape(op, &requested, n)?;
    let out = PyTensorBase::vulkan(
        VkTensor {
            buffer: a.buffer.clone(),
            shape,
        },
        input.tag(),
    );
    crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind())
}

/// The `-1` sentinel and the element-count check, in upstream's words.
fn resolve_shape(op: &str, requested: &[i64], n: usize) -> PyResult<Vec<usize>> {
    let mut wild: Option<usize> = None;
    let mut known: usize = 1;
    for (i, &d) in requested.iter().enumerate() {
        if d == -1 {
            if wild.is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "only one dimension can be inferred",
                ));
            }
            wild = Some(i);
        } else if d < 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: invalid shape dimension {d}"
            )));
        } else {
            known = known.saturating_mul(d as usize);
        }
    }
    let mut shape: Vec<usize> = requested
        .iter()
        .map(|&d| if d == -1 { 0 } else { d as usize })
        .collect();
    match wild {
        Some(i) => {
            if known == 0 || n % known != 0 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "shape {requested:?} is invalid for input of size {n}"
                )));
            }
            shape[i] = n / known;
        }
        None => {
            if known != n {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "shape {requested:?} is invalid for input of size {n}"
                )));
            }
        }
    }
    Ok(shape)
}

/// `aten.t.default`. Rank 0 and 1 are the identity upstream, and are here too;
/// rank 2 materialises through the transpose kernel; anything else is upstream's
/// own error rather than a vulkan one.
fn t_default(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (input, a) = self_vk(op, args, kwargs)?;
    match a.shape.len() {
        0 | 1 => {
            let out = PyTensorBase::vulkan(a, input.tag());
            crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind())
        }
        2 => transpose2d(py, op, &input, &a),
        rank => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "t() expects a tensor with <= 2 dimensions, but self is {rank}D"
        ))),
    }
}

/// `aten.transpose.int`. 2-D only on this device -- see `transpose2d`.
fn transpose_int(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let (input, a) = self_vk(op, args, kwargs)?;
    let rank = a.shape.len();
    let dim = |i: usize, name: &str| -> PyResult<isize> {
        let v = crate::aten::optional(args, kwargs, i, name)?
            .ok_or_else(|| not_implemented(format!("{op}: vulkan: missing argument '{name}'")))?;
        v.extract::<isize>()
    };
    let extent = rank.max(1) as isize;
    let norm = |d: isize| -> PyResult<usize> {
        let i = if d < 0 { d + extent } else { d };
        if i < 0 || i >= extent {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Dimension out of range (expected to be in range of [{}, {}], but got {d})",
                -extent,
                extent - 1
            )));
        }
        Ok(i as usize)
    };
    let d0 = norm(dim(1, "dim0")?)?;
    let d1 = norm(dim(2, "dim1")?)?;
    if d0 == d1 || rank < 2 {
        let out = PyTensorBase::vulkan(a, input.tag());
        return crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind());
    }
    if rank != 2 {
        return Err(not_implemented(format!(
            "{op}: the vulkan device transposes 2-D tensors and was asked for a \
             {rank}-D one {:?}. A `VkTensor` carries a shape and no strides, so a \
             transpose here has to move bytes, and the kernel that moves them is \
             a 2-D one. Higher ranks refuse rather than being permuted by some \
             other route (docs/VULKAN4.md §6).",
            a.shape
        )));
    }
    transpose2d(py, op, &input, &a)
}

/// The materialising 2-D transpose. Pure data movement, so the result is
/// bit-identical to the CPU answer by construction.
fn transpose2d(
    py: Python<'_>,
    op: &str,
    input: &PyTensorBase,
    a: &VkTensor,
) -> PyResult<Py<PyAny>> {
    let (rows, cols) = (a.shape[0], a.shape[1]);
    let ctx = require(op)?;
    let n = rows * cols;
    let out = unsafe {
        let out = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel(
            "transpose2d_f32",
            TRANSPOSE2D_F32_SPV,
            [&a.buffer, &a.buffer, &out],
            [n as u32, rows as u32, cols as u32, 0],
        )
        .map_err(|e| vk_error(op, e))?;
        out
    };
    wrap(py, out, vec![cols, rows], input.tag())
}

/// The product itself, shared by `mm` and `addmm`.
fn matmul_into(op: &str, a: &VkTensor, b: &VkTensor) -> PyResult<(VkBuffer, usize, usize)> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(not_implemented(format!(
            "{op}: the vulkan matmul kernel is 2-D and was given {:?} and {:?}. \
             Batched matmul (`aten.bmm.default`) is not taught this device \
             (docs/VULKAN4.md §6).",
            a.shape, b.shape
        )));
    }
    let (m, k) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    if k != k2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "mat1 and mat2 shapes cannot be multiplied ({m}x{k} and {k2}x{n})"
        )));
    }
    let ctx = require(op)?;
    let count = m * n;
    let out = unsafe {
        let out = ctx.alloc(count * 4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel(
            "matmul_f32",
            MATMUL_F32_SPV,
            [&a.buffer, &b.buffer, &out],
            [count as u32, m as u32, k as u32, n as u32],
        )
        .map_err(|e| vk_error(op, e))?;
        out
    };
    Ok((out, m, n))
}

fn mm(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let lhs = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = crate::aten::tensor_arg(op, args, kwargs, 1, "mat2")?;
    check_all_f32(op, &[&lhs, &rhs])?;
    let a = lhs.vk_tensor(op)?.clone();
    let b = rhs.vk_tensor(op)?.clone();
    let (out, m, n) = matmul_into(op, &a, &b)?;
    wrap(py, out, vec![m, n], lhs.tag())
}

/// `aten.addmm.default(bias, mat1, mat2, *, beta=1, alpha=1)`.
///
/// Two dispatches, not one: the product, then the bias. `beta` and `alpha` are
/// refused unless they are 1 -- `nn.Linear` never sets them, and scaling here
/// would be a third arithmetic operation with no kernel behind it.
fn addmm(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let bias = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let lhs = crate::aten::tensor_arg(op, args, kwargs, 1, "mat1")?;
    let rhs = crate::aten::tensor_arg(op, args, kwargs, 2, "mat2")?;
    for (i, name) in [(3usize, "beta"), (4usize, "alpha")] {
        if let Some(v) = crate::aten::optional(args, kwargs, i, name)? {
            if !v.is_none() && v.extract::<f64>().unwrap_or(1.0) != 1.0 {
                return Err(not_implemented(format!(
                    "{op}: the vulkan device implements addmm with beta = alpha = 1 \
                     and was given {name} != 1 (docs/VULKAN4.md §6)."
                )));
            }
        }
    }
    check_all_f32(op, &[&bias, &lhs, &rhs])?;
    let c = bias.vk_tensor(op)?.clone();
    let a = lhs.vk_tensor(op)?.clone();
    let b = rhs.vk_tensor(op)?.clone();
    let (product, m, n) = matmul_into(op, &a, &b)?;

    // A 1-D bias of width `n` is what `nn.Linear` passes. A 2-D `[m, n]` bias
    // is elementwise and goes through the same kernel with `p1 = n`, because
    // `col` then indexes the row too. Everything else refuses.
    let bias_ok = match c.shape.as_slice() {
        [w] => *w == n,
        [r, w] => *r == m && *w == n,
        _ => false,
    };
    if !bias_ok {
        return Err(not_implemented(format!(
            "{op}: the vulkan device adds a bias of shape [{n}] or [{m}, {n}] and \
             was given {:?}. Wider broadcasting refuses rather than being \
             emulated (docs/VULKAN4.md §6).",
            c.shape
        )));
    }
    let ctx = require(op)?;
    let count = m * n;
    let out = unsafe {
        let out = ctx.alloc(count * 4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel(
            "bias_add_f32",
            BIAS_ADD_F32_SPV,
            [&product, &c.buffer, &out],
            [count as u32, n as u32, 0, 0],
        )
        .map_err(|e| vk_error(op, e))?;
        out
    };
    wrap(py, out, vec![m, n], lhs.tag())
}

/// `x.to("vulkan")` for a tensor that is **not** on Vulkan yet -- the upload
/// half of `docs/VULKAN2.md` §5.2 item 5, which did not exist until this round.
///
/// **Why this had to be added before anything could be measured.** Before it,
/// the only way onto this device was `torch.ones`/`zeros`/`empty`, so every
/// tensor that had ever reached a Vulkan kernel was a constant. A matmul of
/// all-ones agrees with any implementation that sums the right number of ones;
/// it cannot distinguish a correct kernel from one that transposed an index or
/// accumulated in the wrong order. Comparing element-wise against upstream on
/// real data (`docs/VULKAN4.md` §4) needs real data, and this is how it gets
/// there.
///
/// Returns `None` when the call is not a copy *to* vulkan, so the caller falls
/// through to its ordinary path and nothing else changes.
pub fn maybe_upload(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<Py<PyAny>>> {
    if op != "aten._to_copy.default" {
        return Ok(None);
    }
    let Ok(input) = crate::aten::tensor_arg(op, args, kwargs, 0, "self") else {
        return Ok(None);
    };
    let current = input.device_label();
    let label = crate::aten::device_arg_or_label(args, kwargs, 3, "device", &current)?;
    if label.kind != "vulkan" || current.kind == "vulkan" {
        return Ok(None);
    }
    // From here on the call *is* a copy to vulkan, so every remaining problem
    // refuses by name rather than returning `None` and letting `resolve()`
    // answer with the generic "device not available", which would say nothing
    // about which narrowing was hit.
    let tag = crate::aten::dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(input.tag());
    if tag != input.tag() {
        return Err(not_implemented(format!(
            "{op}: the vulkan device cannot change dtype on the way in ({} to \
             {}) -- there is no conversion shader. Cast on the cpu first \
             (docs/VULKAN4.md §6).",
            input.tag().name(),
            tag.name()
        )));
    }
    check_dtype(op, tag)?;
    if current.kind != "cpu" {
        return Err(not_implemented(format!(
            "{op}: the vulkan device accepts a copy from the cpu, not from {} \
             (docs/VULKAN4.md §6).",
            current.kind
        )));
    }
    let host = input.tensor()?;
    let shape: Vec<usize> = host.dims().to_vec();
    let flat = host
        .flatten_all()
        .and_then(|t| t.contiguous())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| crate::err::candle_err(op, e))?;
    let ctx = require(op)?;
    let buffer = unsafe {
        let buf = ctx.alloc(flat.len() * 4).map_err(|e| vk_error(op, e))?;
        ctx.upload(&buf, &flat).map_err(|e| vk_error(op, e))?;
        buf
    };
    Ok(Some(wrap(py, buffer, shape, tag)?))
}

/// `.cpu()`, `.to("cpu")` and `.to("vulkan")` for a tensor already on Vulkan.
fn to_copy(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let tag = crate::aten::dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(input.tag());
    let label = crate::aten::device_arg_or_label(args, kwargs, 3, "device", &input.device_label())?;
    if tag != input.tag() {
        return Err(not_implemented(format!(
            "{op}: the vulkan device cannot change dtype ({} to {}) -- there is \
             no conversion shader. Bring the tensor to the cpu first \
             (docs/VULKAN3.md).",
            input.tag().name(),
            tag.name()
        )));
    }
    let vk_tensor = input.vk_tensor(op)?.clone();
    match label.kind.as_str() {
        "cpu" => {
            let host = to_cpu(op, &vk_tensor)?;
            let out = PyTensorBase::new(host)?;
            crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind())
        }
        "vulkan" => {
            let out = PyTensorBase::vulkan(vk_tensor, tag);
            crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind())
        }
        other => Err(not_implemented(format!(
            "{op}: the vulkan device can copy to cpu and to vulkan, not to \
             {other} (docs/VULKAN3.md)."
        ))),
    }
}

/// `aten.add.Tensor` -- the one elementwise op, f32, same shape, alpha == 1.
///
/// Every narrowing refuses by name instead of being emulated: a broadcast, a
/// scalar `other` and an `alpha` all have perfectly good CPU implementations
/// half a metre away and reaching for one of them here is exactly the silent
/// fallback this device exists to make impossible.
fn add_tensor(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let lhs = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = crate::aten::tensor_arg(op, args, kwargs, 1, "other")?;
    if let Some(alpha) = crate::aten::optional(args, kwargs, 2, "alpha")? {
        if !alpha.is_none() && alpha.extract::<f64>().unwrap_or(1.0) != 1.0 {
            return Err(not_implemented(format!(
                "{op}: the vulkan kernel is a+b and has no alpha (docs/VULKAN3.md)."
            )));
        }
    }
    check_dtype(op, lhs.tag())?;
    check_dtype(op, rhs.tag())?;
    let a = lhs.vk_tensor(op)?.clone();
    let b = rhs.vk_tensor(op)?.clone();
    if a.shape != b.shape {
        return Err(not_implemented(format!(
            "{op}: the vulkan kernel adds equal shapes elementwise and does not \
             broadcast {:?} with {:?} (docs/VULKAN3.md).",
            a.shape, b.shape
        )));
    }
    let ctx = require(op)?;
    let n = a.elem_count();
    let out = unsafe {
        let out = ctx.alloc(n * 4).map_err(|e| vk_error(op, e))?;
        ctx.dispatch_kernel(
            "add_f32",
            ADD_F32_SPV,
            [&a.buffer, &b.buffer, &out],
            [n as u32, 0, 0, 0],
        )
        .map_err(|e| vk_error(op, e))?;
        out
    };
    let result = VkTensor {
        buffer: Arc::new(out),
        shape: a.shape.clone(),
    };
    let wrapped = PyTensorBase::vulkan(result, lhs.tag());
    crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())
}

/// The label a Vulkan tensor reports. There is one device, so there is no index
/// to reconstruct and none is invented -- the mirror of `PyDevice::meta()`.
pub fn label() -> PyDevice {
    PyDevice::checked("vulkan", None).expect("\"vulkan\" is in DEVICE_TYPES")
}

// ---------------------------------------------------------------------------
// What Python can ask
// ---------------------------------------------------------------------------

/// `_C._vulkan_probe()` -- **the reason tests can skip by name.**
///
/// A test that fails where there is no Vulkan is a broken gate, and a test that
/// silently passes there is worse. This returns a dict, so a skip can say
/// *which* driver was missing and a pass can say which GPU it ran on, and the
/// report distinguishes the two.
#[pyfunction]
#[pyo3(name = "_vulkan_probe")]
fn vulkan_probe(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    match context() {
        Ok(ctx) => {
            d.set_item("available", true)?;
            d.set_item("device", ctx.device_name.as_str())?;
            d.set_item("type", ctx.device_type.as_str())?;
            d.set_item("queue_family", ctx.qfi)?;
            d.set_item("error", py.None())?;
        }
        Err(reason) => {
            d.set_item("available", false)?;
            d.set_item("device", py.None())?;
            d.set_item("type", py.None())?;
            d.set_item("queue_family", py.None())?;
            d.set_item("error", reason)?;
        }
    }
    Ok(d.into_any().unbind())
}

/// `_C._vulkan_ops()` -- the closed list of ops taught this device, so a test
/// can pick something that is *not* on it and check the refusal names it.
#[pyfunction]
#[pyo3(name = "_vulkan_ops")]
fn vulkan_ops() -> Vec<&'static str> {
    vec![
        "aten._to_copy.default",
        "aten._unsafe_view.default",
        "aten.add.Tensor",
        "aten.addmm.default",
        "aten.alias.default",
        "aten.clone.default",
        "aten.contiguous.default",
        "aten.detach.default",
        "aten.div.Tensor",
        "aten.mm.default",
        "aten.mul.Tensor",
        "aten.neg.default",
        "aten.relu.default",
        "aten.reshape.default",
        "aten.sub.Tensor",
        "aten.t.default",
        "aten.transpose.int",
        "aten.view.default",
    ]
}

/// `_C._vulkan_counters()` -- **the runtime answer to "did the GPU do it?"**
///
/// Returns the three counters described beside their definitions above:
///
/// * `shader_dispatches` -- compute shaders that ran to completion on the
///   device, counted after `vkWaitForFences` returned success.
/// * `host_uploads` / `host_downloads` -- crossings of the host boundary,
///   counted inside the only two `vkMapMemory` sites in the module.
///
/// A test asserts the delta across one op. That is a statement about what the
/// process did rather than about what the source looks like, which is the
/// distinction `docs/MPSATTN.md` §3.1 says its own evidence failed to make.
#[pyfunction]
#[pyo3(name = "_vulkan_counters")]
fn vulkan_counters(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("shader_dispatches", SHADER_DISPATCHES.load(Ordering::Relaxed))?;
    d.set_item("host_uploads", HOST_UPLOADS.load(Ordering::Relaxed))?;
    d.set_item("host_downloads", HOST_DOWNLOADS.load(Ordering::Relaxed))?;
    Ok(d.into_any().unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vulkan_probe, m)?)?;
    m.add_function(wrap_pyfunction!(vulkan_ops, m)?)?;
    m.add_function(wrap_pyfunction!(vulkan_counters, m)?)?;
    Ok(())
}
