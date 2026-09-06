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

        let pc = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(4)];
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
        n: u32,
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
            &n.to_ne_bytes(),
        );
        self.device.cmd_dispatch(cb, n.div_ceil(64), 1, 1);
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
        "aten.add.Tensor" => add_tensor(py, op, args, kwargs),
        // `detach`/`alias`/`clone` share the buffer, which is what the dense
        // arm does too (a candle clone is an `Arc` clone) and is safe here for
        // a stronger reason: no op on this device writes in place.
        "aten.detach.default" | "aten.alias.default" => {
            let input = crate::aten::tensor_arg(op, args, kwargs, 0, "self")?;
            let vk_tensor = input.vk_tensor(op)?.clone();
            let out = PyTensorBase::vulkan(vk_tensor, input.tag());
            crate::tensor::promote(py, out.into_pyobject(py)?.into_any().unbind())
        }
        other => Err(not_implemented(format!(
            "{other}: not implemented for the vulkan device. This build teaches \
             the vulkan device four ops by name -- aten.add.Tensor, \
             aten._to_copy.default, aten.detach.default and aten.alias.default \
             -- and every other op refuses here rather than falling back to the \
             CPU (docs/VULKAN3.md). Move the tensor with .cpu() to compute {other}."
        ))),
    }
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
        ctx.dispatch_kernel("add_f32", ADD_F32_SPV, [&a.buffer, &b.buffer, &out], n as u32)
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
        "aten.add.Tensor",
        "aten.alias.default",
        "aten.detach.default",
    ]
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(vulkan_probe, m)?)?;
    m.add_function(wrap_pyfunction!(vulkan_ops, m)?)?;
    Ok(())
}
