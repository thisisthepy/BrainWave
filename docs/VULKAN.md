# Vulkan 컴퓨트 경로 — 설 수 있는가

**결론: 섭니다.** 안드로이드 `aarch64-linux-android` 바이너리가 `libvulkan.so` 를 열고, 컴퓨트
셰이더를 디스패치하고, 그 결과가 같은 기기의 CPU 계산과 **세 케이스 전부 비트 단위로 일치**했습니다.
두 경로(`ash`, `wgpu`)를 각각 기기에서 끝까지 돌렸고 **둘 다 통과**했습니다.

**권고는 `ash` 입니다** (§5). 다만 그 권고를 뒤집는 조건이 하나 있고, 그것은 이 작업의 범위를
넘는 결정이라 §5.3 에 따로 적어 두었습니다 — 조율 세션이 판단할 항목입니다.

**성능은 재지 않았습니다.** 재서는 안 됩니다 — 이유는 §7. 이 문서는 *되는가*에만 답합니다.

---

## 1. 무엇을 증명했나

`rust/vk_probe` — `rust/torch_c` 와 워크스페이스가 분리된 독립 크레이트입니다. 두 개의 바이너리가
**같은 세 케이스**를 돌리고 **같은 채점 코드**(`src/check.rs`)로 판정합니다.

| 케이스 | 크기 | `ash` | `wgpu` |
|---|---|---|---|
| `vecadd` `c[i] = a[i] + b[i]` | n = 1,048,576 | **1048576/1048576 비트 동일** | **동일** |
| `matmul` 정사각 | 64×64 · 64×64 | **4096/4096 비트 동일** | **동일** |
| `matmul` 비정사각 | 96×80 · 80×64 | **6144/6144 비트 동일** | **동일** |

채점 코드를 공유하는 것이 핵심입니다. 두 경로가 각자의 판정 코드를 가지면 **한쪽이 더 후하게
채점해서 생긴 차이**를 GPU 차이로 오독하게 됩니다.

비정사각 케이스를 따로 둔 이유는 `m == n == k` 일 때 행/열 우선 혼동이 그럴듯한 숫자를 그대로
내놓기 때문입니다. 96×80×64 에서는 그러지 못합니다.

### 실행 방법

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-vulkan
export ANDROID_NDK_HOME=~/Library/Android/sdk/ndk/27.1.12297006
ADB=~/Library/Android/sdk/platform-tools/adb
BIN=$CARGO_TARGET_DIR/aarch64-linux-android/release

cd rust/vk_probe
cargo ndk -t arm64-v8a --platform 26 build --release > /tmp/vk.log 2>&1; echo "EXIT=$?"
cargo ndk -t arm64-v8a --platform 26 build --release --features wgpu-route > /tmp/wg.log 2>&1; echo "EXIT=$?"

$ADB -s emulator-5556 shell 'mkdir -p /data/local/tmp/bw_vk'
$ADB -s emulator-5556 push $BIN/vk_probe      /data/local/tmp/bw_vk/vk_probe
$ADB -s emulator-5556 push $BIN/vk_probe_wgpu /data/local/tmp/bw_vk/vk_probe_wgpu
$ADB -s emulator-5556 shell 'chmod 755 /data/local/tmp/bw_vk/vk_probe*; /data/local/tmp/bw_vk/vk_probe'
$ADB -s emulator-5556 shell '/data/local/tmp/bw_vk/vk_probe_wgpu'
```

### 실제 출력 — `ash` (emulator-5556 = `pmp_api36`, API 36, `-gpu host`)

```
vk_probe: Vulkan compute vs CPU, android aarch64
loader instance version: 1.4.0
physical devices: 1
  device: "Apple M1" type=INTEGRATED_GPU api=1.3.0 driver=0x28a0
    maxComputeWorkGroupInvocations=1024 maxComputeSharedMemorySize=32768 maxStorageBufferRange=4294967295
    queue family 0: count=2 flags=GRAPHICS | COMPUTE | TRANSFER
using "Apple M1" queue family 0
device extensions: 44 (promoted-to-core ones do not appear here)
  ext VK_KHR_shader_float16_int8               present
  ext VK_KHR_8bit_storage                      absent
  ext VK_KHR_cooperative_matrix                absent
quantisation-relevant feature bits:
  storageBuffer16BitAccess (fp16/i16 in an SSBO) YES
  storageBuffer8BitAccess  (int8 in an SSBO)     YES
  shaderFloat16            (fp16 arithmetic)     YES
  shaderInt8               (int8 arithmetic)     YES
  shaderInt16              (int16 arithmetic)    YES
vecadd n=1048576
  vecadd                 n=1048576 bit-identical=1048576/1048576 max_abs=0e0 max_ulp=0 nonfinite=0
                         VERDICT: bit-identical to CPU
matmul 64x64 * 64x64
  matmul 64x64x64        n=4096 bit-identical=4096/4096 max_abs=0e0 max_ulp=0 nonfinite=0
                         VERDICT: bit-identical to CPU
matmul 96x80 * 80x64
  matmul 96x80x64        n=6144 bit-identical=6144/6144 max_abs=0e0 max_ulp=0 nonfinite=0
                         VERDICT: bit-identical to CPU
RESULT: PASS
```

### 실제 출력 — `wgpu`

```
vk_probe_wgpu: wgpu compute vs CPU, android aarch64
adapter: "Apple M1" backend=Vulkan type=IntegratedGpu driver="MoltenVK"
vecadd n=1048576
  vecadd                 n=1048576 bit-identical=1048576/1048576 max_abs=0e0 max_ulp=0 nonfinite=0
                         VERDICT: bit-identical to CPU
matmul 64x64x64 ... VERDICT: bit-identical to CPU
matmul 96x80x64 ... VERDICT: bit-identical to CPU
RESULT: PASS
```

`driver="MoltenVK"` 가 §2 의 경고를 **추측이 아니라 확인**으로 바꿔 줍니다. 게스트 Vulkan 호출이
호스트로 넘어가 MoltenVK 를 거쳐 Metal 로 번역되고 있다는 것을 드라이버 자신이 말합니다.

### 이 검증은 실패할 수 있다 — 확인했습니다

CLAUDE.md §5.5 의 항목입니다. `VK_PROBE_TAMPER=<n>` 은 CPU 기준값 한 원소를 n ULP 흔들고
**합격/불합격 판정을 뒤집습니다** — 모든 케이스가 MISMATCH 를 내야 그 실행이 옳습니다.

```
$ VK_PROBE_TAMPER=1024 ./vk_probe
VK_PROBE_TAMPER=1024: reference perturbed by 1024 ULP, every case MUST report MISMATCH
  vecadd          bit-identical=1048575/1048576 max_ulp=1024   VERDICT: MISMATCH
  matmul 64x64x64 bit-identical=4095/4096       max_ulp=1024   VERDICT: MISMATCH
  matmul 96x80x64 bit-identical=6143/6144       max_ulp=1024   VERDICT: MISMATCH
RESULT: PASS (tamper: comparison caught the perturbation)

$ VK_PROBE_TAMPER=1024 ./vk_probe_wgpu
RESULT: PASS (tamper: comparison caught the perturbation)
```

**대조군이 실제로 무언가를 잡아냈습니다.** 처음에는 1 ULP 로 흔들었는데 `RESULT: FAIL (tamper:
comparison MISSED the perturbation)` 이 나왔습니다. 버그가 아니라 판정 기준의 성질입니다 —
`report()` 는 2 ULP 이내를 통과시키고, 1~2 ULP 는 곱셈-덧셈 융합(FMA)의 서명이라 우리가 의도적으로
허용하는 범위입니다. **기준을 시험하려면 기준 바깥으로 흔들어야** 하므로 기본값을 1024 로
두었습니다. `VK_PROBE_TAMPER=1` 은 지금도 FAIL 을 내며, 그것이 허용 경계가 어디인지 기록합니다.

---

## 2. 안드로이드 쪽에서 증명된 것과, 호스트의 사실인 것

`"Apple M1"` 이라는 장치 이름과 `driver="MoltenVK"` 가 이 구분의 전부입니다.
`ro.hardware.vulkan = ranchu` 이고 `vulkan.ranchu.so` (gfxstream) 가 **호스트의
`VkPhysicalDeviceProperties` 를 그대로 게스트로 전달**합니다.

| | 판정 | 근거 |
|---|---|---|
| 안드로이드 로더(`/system/lib64/libvulkan.so`)가 열린다 | **증명됨** | `Entry::load()` 성공 |
| `shell` 도메인(uid 2000)이 SELinux Enforcing 아래에서 GPU 에 닿는다 | **증명됨** | 앱 설치 없이 `/data/local/tmp` 바이너리로 디스패치 성공 |
| ICD 가 우리 SPIR-V 를 받는다 | **증명됨** | NDK `glslc` 산출물, 그리고 `naga` 산출물 둘 다 통과 |
| `ash`·`wgpu` 가 `aarch64-linux-android` 로 크로스컴파일된다 | **증명됨** | 둘 다 `EXIT=0`, ELF aarch64 PIE |
| 큐·버퍼·디스크립터·디스패치·펜스 전 과정 | **증명됨** | 세 케이스 완주 |
| 헤드리스 · Activity 없음 · JNI 없음 | **증명됨** | 평범한 `fn main()` 바이너리 |
| GPU 결과가 CPU 와 맞는다 | **증명됨, 단 이 GPU 에 한해** | 비트 동일 |
| **fp16·int8 능력** | **호스트의 사실** | 아래 |
| **Adreno·Mali 에서의 정확성** | **미지** | |
| **성능** | **측정하지 않음. 측정해서도 안 됨** | §7 |

### DEVICE.md §10 의 fp16 관련 서술을 좁힙니다

`docs/DEVICE.md` §10 은 게스트 확장 문자열 `ANDROID_EMU_vulkan_shader_float16_int8` 을 근거로
"fp16·int8 셰이더가 된다는 뜻이라 양자화 경로까지 시험 대상" 이라고 적었습니다. 위 실측은 그보다
**약한 것만** 뒷받침합니다: 켜져 있다고 보고된 기능 비트는 **호스트 M1 의 것이 전달된 값**이지
안드로이드 기기 일반에 대한 진술이 아닙니다. 실제 폰의 Adreno/Mali 가 `shaderFloat16` 이나
`storageBuffer8BitAccess` 를 주는지는 **이 에뮬레이터로 알 수 없습니다.**

같은 조사에서 반대 방향의 함정도 하나 확인했습니다. 확장 목록만 보면 `VK_KHR_8bit_storage` 가
**absent** 인데 기능 비트 `storageBuffer8BitAccess` 는 **YES** 입니다. 모순이 아니라
**코어로 승격된 확장은 확장 목록에 나타나지 않기 때문**입니다 (`VK_KHR_16bit_storage`·
`VK_KHR_variable_pointers`·`VK_KHR_storage_buffer_storage_class` 는 1.1, `VK_KHR_8bit_storage` 는
1.2 로 승격). **확장 문자열만 세어 능력을 판정하면 거짓 음성이 납니다** — 그래서 프로브는 확장
목록이 아니라 `VkPhysicalDeviceVulkan11Features`/`Vulkan12Features` 를 조회해 보고합니다.

### 비트 동일은 기대 이상이며, 기기에서 유지된다고 약속하면 안 됩니다

셰이더와 CPU 참조가 같은 순서로 누적하도록 썼고 `mul_add` 를 쓰지 않았습니다 — GPU 가 **우리가
시키지 않은 융합을 하는지** 보려는 것이었습니다. 이 경로에서는 하지 않았습니다(0 ULP). 그러나
실제 모바일 드라이버는 곱셈-덧셈을 FMA 로 접는 것이 흔하고, 그러면 **1~2 ULP** 가 나옵니다.
그것은 결함이 아니며 프로브도 통과로 판정합니다. **기기에서 비트 동일을 기대하지 마십시오.**

---

## 3. 셰이더는 어떻게 공급하는가

**두 경로 모두 기기에 셰이더 컴파일러를 싣지 않습니다.** 이것이 전제와의 충돌 여부를 가르는
지점입니다.

| | `ash` | `wgpu` |
|---|---|---|
| 소스 언어 | GLSL | WGSL |
| SPIR-V 로 바꾸는 시점 | **빌드 타임** (`build.rs` → `glslc`) | 파이프라인 생성 시점 (`naga`) |
| 컴파일러가 어디서 도는가 | 빌드 머신 | **기기 — 단 순수 Rust** |
| C++ 툴체인이 앱에 실리는가 | 아니오 | 아니오 |
| 산출물 | `vecadd.spv` 1,296 B · `matmul.spv` 1,988 B | 바이너리에 WGSL 문자열 |

`ash` 쪽은 NDK 에 이미 들어 있는 `glslc` 를 부릅니다 —
`$ANDROID_NDK_HOME/shader-tools/darwin-x86_64/glslc` (Apple Silicon 에서는 Rosetta 로 돕니다).
**추가로 설치할 툴체인이 없습니다.**

`wgpu` 쪽의 `naga` 번역은 기기에서 돌지만 **순수 Rust 이고 C++ 컴파일러가 아닙니다.** 필요하면
빌드 타임에 `naga` 를 돌려 SPIR-V 를 미리 굽고 그대로 넘길 수도 있습니다 — 즉 이 축에서는
두 경로가 실질적으로 같은 자리에 있습니다.

### 이것이 "모델을 미리 변환하지 않는다" 전제와 충돌하지 않는 이유

충돌하지 않습니다. **커널은 모델이 아닙니다.** 전제가 지키는 것은 사용자의 모델이 export/변환
단계를 거치지 않고 그대로 `import` 되어 도는 것이지, *우리가 작성한 고정된 커널 집합*이 미리
컴파일되어 있으면 안 된다는 뜻이 아닙니다. 지금도 `candle` 의 CPU 커널은 미리 컴파일된 기계어이고,
누구도 그것을 "모델을 미리 변환했다" 고 말하지 않습니다. SPIR-V 커널은 같은 지위입니다.

전제를 실제로 위협하는 것은 **모양(shape)마다 셰이더를 다시 만들어야 하는 경우**입니다. 프로브는
그 길을 피했고, 그것이 설계에서 유일하게 신경 쓴 지점입니다 — **모양은 데이터로 넘깁니다**
(`ash` 는 푸시 상수, `wgpu` 는 유니폼 버퍼). **셰이더 하나가 모든 크기를 처리하고, 재컴파일이
일어나지 않습니다.** 96×80×64 케이스가 64×64×64 와 같은 SPIR-V 모듈로 돈 것이 그 증거입니다.

---

## 4. 후보 수단 비교

### 4.1 실측 요약

| | `ash` | `wgpu` | `vulkano` | ExecuTorch `vulkan` | candle |
|---|---|---|---|---|---|
| Vulkan 컴퓨트가 되는가 | **기기에서 확인** | **기기에서 확인** | 빌드만 확인 | 미시도 | **백엔드 없음** |
| 안드로이드 크로스빌드 | ✅ | ✅ | ✅ | 미시도 | — |
| 외부 크레이트 수 | **3** | **60** ※ | 79 † | (C++) | — |
| 산출물 크기 | **718,864 B** | 6,766,904 B | — | — | — |
| 클린 빌드 | **8.5 s** | 24 s | 2m47s | — | — |
| `NEEDED` | `libdl` `libc` | `libdl` `libm` `libc` | — | — | — |
| 빌드 머신 C++ 툴체인 | 불필요 | 불필요 | **필요** (cmake+shaderc) | **필요** | — |

※ `default-features = false, features = ["vulkan", "wgsl"]` 기준으로 이 저장소에서 직접 센 값입니다.
피처를 늘리면 늘어납니다.
† `vulkano` 행은 `/tmp` 의 별도 프로브에서 나온 값이고 **이 저장소에서 재현하지 않았습니다.**
`ash`·`wgpu` 행은 전부 `rust/vk_probe` 에서 직접 측정했습니다.

**두 경로 모두 `libvulkan.so` 가 `NEEDED` 에 없습니다.** `dlopen` 으로 열기 때문입니다. 이것은
편의가 아니라 요구사항입니다 — 링크했다면 Vulkan 없는 폰에서 `_C.so` 자체가 로드에 실패하고,
그것은 **GPU 가 없다는 이유로 `import torch` 가 통째로 깨진다**는 뜻입니다. 호스트 macOS 에서 그
폴백 경로를 실제로 밟아 확인했습니다:

```
$ ./vk_probe            # macOS, Vulkan 로더 없음
RESULT: FAIL (init) -- failed to load libvulkan.so: dlopen(libvulkan.dylib, 0x0005): ... (no such file)
rc=2
```

크래시가 아니라 진단입니다.

> **정정 (VULKAN2.md §4).** 이 "macOS 에는 로더가 없다" 는 전제는 더 이상 맞지 않습니다.
> 디스크에 이미 있는 안드로이드 에뮬레이터가 Vulkan 로더와 ICD 4개를 함께 배포하고 있고,
> 환경변수 둘만 주면 **같은 프로브 바이너리가 수정 없이** 이 맥의 실제 GPU(`Apple M1`,
> `INTEGRATED_GPU`)에서 통과합니다. 위 실패는 "로더를 못 찾았다" 이지 "이 기계에 없다" 가
> 아니었습니다. 다만 그 파일들은 안드로이드 SDK 내부물이므로 **개발 기계 경로일 뿐 배포 경로가
> 아닙니다.**

### 4.2 `candle` 에는 Vulkan 백엔드가 없습니다 — 확정

핀 고정된 `0.11.0` 태그에서 확인했습니다. `candle-core/Cargo.toml` 의 `[features]` 는
`default`·`cuda`·`cudnn`·`nccl`·`mkl`·`accelerate`·`metal`·`metal-debug-labels`·`ug` 이고
**`vulkan` 은 없습니다.** `candle-core/src/lib.rs` 도 `cuda_backend`/`metal_backend`/`cpu_backend`
만 조건부 컴파일하며 `vulkan_backend` 모듈이 존재하지 않습니다.

**따라서 어느 경로를 고르든 Vulkan 커널은 우리가 씁니다.** 이것은 선택지가 아니라 전제입니다.

### 4.3 `vulkano` — 되지만 고를 이유가 없음

빌드는 됩니다(79 크레이트, 2m47s). `shaderc-sys` 가 shaderc+glslang+SPIRV-Tools **C++ 트리 전체를
CMake 로 빌드**하는 것이 눈에 띄는 비용인데, 확인해 보니 **앱에는 실리지 않습니다** —
`vulkano-shaders` 는 `proc-macro = true` 크레이트라 호스트에서만 돌고, 최종 안드로이드 바이너리의
`NEEDED` 에 shaderc 계열이 전혀 없습니다. 즉 위험은 **빌드 머신/CI 요구사항**(cmake·C++·python3와
클린 빌드마다 ~3분)이지 런타임이 아닙니다.

그래도 고를 이유가 없습니다. `ash` 와 같은 Vulkan 전용 API 면서 크레이트가 79 개이고, GLSL 컴파일을
위해 CI 에 C++ 툴체인을 요구합니다. **`ash` 가 NDK 의 `glslc` 로 같은 일을 0 개의 추가 요구사항으로
합니다.**

### 4.4 ExecuTorch `vulkan` 델리게이트 — 권고하지 않음

세 가지가 걸립니다.

1. **문서화된 유일한 경로가 AOT 입니다.** `torch.export` →
   `to_edge_transform_and_lower(VulkanPartitioner())` → `.pte` → 런타임이 그 `.pte` 를 로드.
   이것은 이 프로젝트의 전제와 정면으로 부딪히며, `README.md` 가 이미 ExecuTorch 를 그 이유로
   배제하고 있습니다 — *"export step, and what runs is not what you wrote"*.
2. **`.pte` 없이 쓸 수 있는 층이 있기는 합니다.** `vkcompute::ComputeGraph` + `api::Context` 를
   직접 불러 단일 op 그래프를 세워 돌릴 수 있고, ExecuTorch 자신의 테스트
   (`backends/vulkan/test/vulkan_compute_api_test.cpp` 의 `test_mm`) 가 `mm` 에 대해 정확히 그렇게
   합니다. 다만 **`test/` 아래에 있고, 문서화되어 있지 않으며, ABI 가 고정되어 있지 않습니다.**
   공개 계약이 아닌 내부 면에 의존하게 됩니다.
3. **빌드가 독립적이지 않습니다.** `Vulkan-Headers`·`volk`·`VulkanMemoryAllocator` 서브모듈,
   `glslc`, `flatc` 가 필요하고, 그 "`.pte` 없이 쓰는" 경로조차
   `executorch/runtime/core/exec_aten/exec_aten.h` 를 끌어와 ExecuTorch 코어 런타임을 함께
   가져옵니다. `candle-core` 처럼 깔끔히 벤더링되지 않습니다. 셰이더는 165 개 GLSL + 165 개 YAML
   변형 설정에서 빌드 타임에 생성됩니다(이 점 자체는 우리와 같습니다).

**`thisisthepy/executorch` 포크 상태:** `main` HEAD 가 `a6cd1dce3e83` (2025-03-21), 업스트림
`pytorch/executorch` 는 `9a2d135d511d` (2026-08-24) 로 **7069 커밋 앞서 있습니다.** 포크에 고유
커밋이 기록되어 있지 않아 **리베이스라기보다 현재 업스트림에서 다시 포크하는 것에 가깝습니다.**
저장소 안에 이 포크를 빌드하거나 링크하는 배선은 **없습니다** — 참조는 문서와
`rust/torch_c/src/device.rs` 가 `"vulkan"` 장치 문자열을 예약해 둔 것뿐입니다.

**포기하는 것:** `docs/DESIGN.md` §5 가 candle 을 고른 근거 자체 — "소유하는 코드가 작다".
Vulkan 하나를 얻으려고 그 논거를 되돌리게 됩니다.

---

## 5. 권고

### 5.1 `ash` 로 갑니다

1. **기기에서 통과했고, 가장 가볍습니다.** 4 크레이트(자신 포함), 719 KB, 8.5 초. `wgpu` 도 같은
   결과를 냈지만 **6.0 MB 를 더 쓰고 60 크레이트를 끌고 옵니다** — 같은 답에 9.4 배입니다.
2. **셰이더 툴체인이 이미 있습니다.** NDK 의 `glslc` 로 끝나고, 빌드 머신에 아무것도 추가로
   요구하지 않습니다. `vulkano` 가 CI 에 C++/CMake 를 요구하는 것과 대비됩니다.
3. **`docs/DESIGN.md` §5 의 "소유하는 코드가 작다" 와 정합합니다.** `ash` 는 런타임이 아니라
   Vulkan 헤더의 생성된 바인딩입니다 — 추상화를 사는 것이 아니라 호출 규약을 사는 것입니다.
4. **실패 모드가 안전합니다.** `dlopen` 이라 Vulkan 없는 기기에서 GPU 경로만 조용히 꺼지고
   `import torch` 는 그대로 됩니다.

### 5.2 포기하는 것

**안전성 추상화입니다.** `ash` 는 Vulkan 을 그대로 노출하므로 수명·동기화·메모리 타입 선택이
전부 우리 책임입니다. 프로브는 매 디스패치마다 파이프라인·디스크립터 풀·커맨드 풀을 전부 만들고
부수는데, **그건 프로브라서 그런 것이지 런타임이 그래도 된다는 뜻이 아닙니다.** 커널이 몇 개를
넘어가면 얇은 내부 계층(할당자, 디스크립터·파이프라인 캐시)이 필요해지고, 그 코드는 `wgpu` 를
골랐다면 공짜로 얻었을 것입니다. **이것이 이 권고의 실제 비용입니다.**

### 5.3 이 권고를 뒤집는 조건 — 조율 세션이 판단할 것

**`wgpu` 는 Apple 타깃에서 MoltenVK 없이 네이티브 Metal 백엔드로 갑니다.** `ash` 는 그러지
못합니다(컴파일은 되지만 — `aarch64-apple-ios` 확인함 — 로더가 없어 런타임에 위 macOS 진단이
납니다). 즉 질문은 이렇게 갈립니다.

| Apple GPU 를 어떻게 채울 것인가 | 그러면 |
|---|---|
| candle 의 기존 `metal` 피처를 켠다 | **`ash` 가 맞습니다.** Vulkan 은 안드로이드 구멍만 메우면 되고, 커널을 두 번 쓸 일이 없습니다 |
| 우리가 커널을 직접 소유한다 (candle 에 없는 융합 커널 등) | **`wgpu` 가 맞습니다.** WGSL 을 한 번 써서 세 타깃을 다 덮습니다 |

candle 에는 `metal` 피처가 **있고**, 지금 `rust/torch_c/Cargo.toml` 은 Apple 에서 `accelerate` 만
켜고 `metal` 은 켜지 않았습니다. 첫 번째 칸이 사실이라면 `ash` 가 분명히 맞습니다.

**저는 이 결정을 하지 않았습니다.** 세 타깃에 걸친 커널 소유권 결정이라 이번 지시(안드로이드
Vulkan 이 서는가)의 범위를 넘습니다 — CLAUDE.md §5.7 항목입니다. **`wgpu` 프로브를 지우지 않고
남겨 둔 이유가 이것입니다:** 두 번째 칸으로 결정되면 그 경로는 이미 기기에서 검증되어 있습니다.

### 5.4 아직 하지 않은 것 (제안)

**`rust/torch_c` 에 Vulkan 의존성을 넣지 않았습니다.** 지시대로입니다. 확인:

- `cargo metadata` — `torch_c` 의 워크스페이스 멤버는 자기 자신뿐 (`vk_probe` 는 별도 워크스페이스)
- `cargo check --release` — `torch_c` 는 그대로 `EXIT=0`
- `git status --short` — 변경은 `rust/vk_probe/` 와 `docs/VULKAN.md` 뿐

다음 단계로 제안하는 것:

1. `ash` 를 `[target.'cfg(target_os = "android")'.dependencies]` 로만 넣습니다. iOS/macOS 에는
   넣지 않습니다.
2. `rust/torch_c/src/device.rs` 가 이미 예약해 둔 `"vulkan"` 장치 문자열 뒤에 배선합니다.
3. 커널은 `matmul` 하나부터. 골든 스위트를 CPU 대조로 그대로 돌립니다.
4. §5.3 을 먼저 결정합니다 — 1번보다 앞섭니다.

---

## 6. 규율 — 이 작업에서 지킨 것

- **emulator-5556 (`pmp_api36`) 만** 썼습니다. 5554 는 건드리지 않았습니다.
- 파일은 **`/data/local/tmp/bw_vk` 안에만** 올렸습니다. 같은 AVD 의 `pmp`,
  `pmp-nativetest-arm64-v8a` 는 손대지 않았고 앱 설치도 하지 않았습니다.
- `adb shell` 종료 코드를 믿지 않고 **판정을 출력의 `RESULT:` 한 줄로** 냅니다.
- 빌드 종료 코드는 파일로 리다이렉트한 뒤 `$?` 로 읽었습니다.
- 커밋하지 않았습니다.

---

## 7. 에뮬레이터로 답할 수 없는 것

**성능은 재지 않았고, 재서도 안 됩니다.** `ranchu` 가 호스트로 번역하고 드라이버가 스스로
`MoltenVK` 라고 보고합니다 — 여기서 나온 수치는 이 Mac 의 M1 을 설명합니다. 두 프로브 모두
타이밍을 **하나도 출력하지 않습니다** — 숫자가 있으면 누군가 인용하기 때문입니다.
`docs/PERF.md` §7.3 의 Apple 측정(디코딩 모양 행렬×벡터에서 n=4096 까지 GPU 우세)은 실기 측정의
**가설**이지 답이 아닙니다.

그 외에 이 문서가 답하지 않는 것:

- **실물 단말이 없습니다.** Adreno·Mali 드라이버의 SPIR-V 수용, 정확성, 기능 비트 전부 미지입니다.
- **기능 비트는 호스트의 것입니다.** §2 참조. **양자화 경로가 기기에서 가능한지는 열려 있습니다.**
- **비트 동일이 기기에서도 유지되는지 모릅니다.** FMA 융합으로 1~2 ULP 가 정상 범위입니다.
- **APK 안에서의 동작을 확인하지 않았습니다.** `/data/local/tmp` 의 `shell` 도메인 바이너리로만
  돌렸습니다. 앱 프로세스는 SELinux 컨텍스트가 다릅니다. GPU 접근은 오히려 앱 쪽이 정상 경로라
  악화될 가능성은 낮지만 **확인한 것은 아닙니다.**
- **API 26 에서 시도하지 않았습니다.** 5554 가 사용 중이라 손대지 않았습니다. NDK 27 의
  `libvulkan.so` 스텁은 **API 24** 부터 있고(21~23 에는 없음) 프로브는 `--platform 26` 으로
  빌드했지만, **돈 것은 API 36 뿐입니다.**
- **커널 하나짜리 프로브입니다.** 공유 메모리 타일링, 서브그룹 연산, 여러 디스패치의 배리어
  동기화 — 실제 커널이 필요로 할 것 중 어느 것도 시험하지 않았습니다.
- **큰 버퍼·메모리 압박 없음.** 최대 4 MB 입니다. `wgpu` 쪽은 `Limits::downlevel_defaults()` 로
  돌렸는데 이것은 기기 한계가 아니라 **wgpu 가 거는 하한**이며, 큰 텐서에는
  `adapter.limits()` 로 올려야 합니다 — 그 경로는 시험하지 않았습니다.
