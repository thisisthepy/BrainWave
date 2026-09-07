# Vulkan 배선 — 사이징 라운드

**결론: 이번 라운드는 배선이 아니라 사이징입니다.** 그리고 사이징 도중 지시가 전제한 사실 하나가
**틀렸다는 것을 실측으로 확인했습니다.**

| 질문 | 답 |
|---|---|
| 기존 프로브가 증명한 것은 디스패치된 **op** 인가, 돈 **셰이더**인가 | **셰이더**입니다. torch op 는 하나도 지나가지 않았습니다 |
| 핀 고정된 `candle-core 0.11.0` 에 Vulkan 백엔드가 있는가 | **없습니다.** `Device`·`Storage` 가 닫힌 enum 이고 소스에 `vulkan` 문자열이 0 건 |
| 이 기계에서 Vulkan 을 시험할 수 있는가 | **있습니다.** README 와 `VULKAN.md` §4.1 의 "호스트에 Vulkan 로더 없음" 은 **더 이상 사실이 아닙니다** |

세 번째 줄이 이 문서의 새로운 내용입니다. 나머지 둘은 `docs/devices/VULKAN.md` 가 이미 답해 두었고,
저는 그것을 재확인만 했습니다.

---

## 1. 프로브가 실제로 증명한 것 — 코드 이전에 답한다

**지시대로 코드보다 먼저 적습니다.** `rust/vk_probe` 는 **`rust/torch_c` 와 워크스페이스가 분리된
독립 크레이트**입니다 (`Cargo.toml` 주석이 그렇게 하도록 의도했다고 명시). 그것이 증명한 것은:

- Vulkan 인스턴스·물리 장치·큐·버퍼·디스크립터·**컴퓨트 셰이더 디스패치**·펜스가 끝까지 돈다
- 그 셰이더의 출력이 같은 기계의 CPU 참조와 **비트 단위로 일치**한다 (세 케이스)

증명하지 **않은** 것:

- `torch.device("vulkan")` 이 무언가에 닿는 것 — **프로브는 `torch._C` 를 임포트하지 않습니다**
- candle `Tensor` 가 GPU 메모리에 사는 것 — 프로브의 버퍼는 순수 Vulkan `VkBuffer` 입니다
- 디스패처·스토리지·dtype 중 어느 것 하나도 — **프로브는 이 프로젝트 타입을 하나도 쓰지 않습니다**

즉 **"셰이더가 돌았다" 와 "op 이 디스패치됐다" 사이의 거리 전부가 아직 남아 있습니다.**
프로브는 하한을 세웠습니다 — *하드웨어와 툴체인은 우리를 막지 않는다*. 그 이상은 아닙니다.

---

## 2. `torch.device("vulkan")` 은 지금 무엇에 걸리는가 — 층위별

**세 층 중 두 번째입니다.** 어휘가 아니라 백엔드 해석에서 걸립니다.

| 층 | 상태 | 근거 |
|---|---|---|
| 1. 장치 어휘 | **통과합니다.** 거부하지 않습니다 | `device.rs:69` — `DEVICE_TYPES` 20개에 `"vulkan"` 이 이미 있습니다. 업스트림 어휘를 그대로 전사한 것이라 우리가 넣은 것이 아니라 **원래 있던 것**입니다. `torch.device("vulkan")` 은 오늘도 객체를 만듭니다 |
| 2. **백엔드 해석** | **여기서 걸립니다** | `PyDevice::resolve()` 의 `other =>` 팔 → `not_implemented("device not available in torch._C shim: vulkan")` |
| 3. 스토리지 할당자 | 도달하지 않습니다 | `storage.rs` 는 CPU 전용(`storage.rs:103` 주석)이지만 2층에서 이미 멈춥니다 |
| 4. 디스패처 | 도달하지 않습니다 | 마찬가지 |

**그래서 "몇 층 깊은가"의 답은 2층이고, 그것이 좋은 소식이 아니라 나쁜 소식입니다.** 2층이 얕아서
고치기 쉬운 것이 아니라, **2층의 반환 타입이 `candle_core::Device` 이기 때문**입니다:

```rust
pub fn resolve(&self) -> PyResult<Device>     // Device = candle_core::Device
```

`candle_core::Device` 는 `Cpu | Cuda | Metal` 세 변형의 **닫힌 enum** 입니다. Vulkan 핸들을
넣을 자리가 없습니다. 따라서 2층은 *팔 하나 추가*로 고쳐지지 않고 **시그니처가 바뀌어야** 합니다.

---

## 3. candle 에 Vulkan 백엔드가 있는가 — 없습니다, 재확인

`docs/devices/VULKAN.md` §4.2 의 판정을 이번에 독립적으로 다시 확인했습니다
(`~/.cargo/registry/.../candle-core-0.11.0`):

```
pub enum Device  { Cpu, Cuda(CudaDevice), Metal(MetalDevice) }      // 닫힘
pub enum Storage { Cpu(CpuStorage), Cuda(CudaStorage), Metal(MetalStorage) }  // 닫힘
[features] accelerate cuda cudnn default metal ...                   // vulkan 없음
$ grep -rni vulkan candle-core-0.11.0/src/     ->  0 건
```

그리고 **candle 은 벤더링되어 있지 않습니다** — `Cargo.toml` 이 crates.io 의 `0.11.0` 을
버전으로 참조합니다. 즉 백엔드를 추가하려면 **candle 을 포크해야** 하고, 그것은 직전 커밋
`fc1f038` 이 *"float8 computes what upstream computes, **without forking candle**"* 로 명시적으로
피한 길입니다.

**따라서 이것은 "가져오기" 라운드가 아니라 "직접 쓰기" 라운드입니다** — 지시가 나눈 두 갈래 중
후자이고, 그 둘은 서로 다른 회차라는 지시의 판단이 맞습니다.

---

## 4. 새로 확인한 사실 — 이 기계에서 Vulkan 이 **돕니다**

`docs/devices/VULKAN.md` §4.1 은 호스트 macOS 에서 프로브가 이렇게 죽는 것을 기록했습니다:

```
RESULT: FAIL (init) -- failed to load libvulkan.so: dlopen(libvulkan.dylib, 0x0005): ... (no such file)
```

**이번에 그대로 재현했습니다.** 시스템 위치에는 아무것도 없습니다 — `/usr/local/lib`,
`/opt/homebrew/lib`, `~/VulkanSDK`, `/Applications/VulkanSDK` 전부 없음, `brew` 에도 없음.

그런데 **이미 디스크에 있습니다.** 안드로이드 에뮬레이터가 자기 번들 안에 로더 + ICD 넷을
싣고 다닙니다:

```
~/Library/Android/sdk/emulator/lib64/vulkan/
    libvulkan.dylib              로더 (instance version 1.4.344)
    libMoltenVK.dylib            + MoltenVK_icd.json          portability
    libvulkan_kosmickrisp.dylib  + libkosmickrisp_icd.json    Metal 백엔드, 네이티브
    libvulkan_lvp.dylib          + lvp_icd.json               lavapipe, CPU
    libvk_swiftshader.dylib      + vk_swiftshader_icd.json    SwiftShader, CPU
```

**아무것도 설치하지 않았습니다.** 환경변수 둘만 주고 기존 프로브 바이너리를 그대로 돌렸습니다.

### 4.1 실측 — 호스트 macOS, `vk_probe` 무수정

```sh
V=~/Library/Android/sdk/emulator/lib64/vulkan
export DYLD_LIBRARY_PATH=$V
VK_DRIVER_FILES=$V/<icd>.json  $CARGO_TARGET_DIR/release/vk_probe
```

| ICD | 보고된 장치 | 타입 | 판정 |
|---|---|---|---|
| `libkosmickrisp_icd.json` | **`"Apple M1"`** api=1.3.335 | **INTEGRATED_GPU** | **RESULT: PASS** — 세 케이스 비트 동일 |
| `lvp_icd.json` | `"llvmpipe (LLVM 21.1.4, 128 bits)"` | CPU | RESULT: PASS — 비트 동일 |
| `vk_swiftshader_icd.json` | `"SwiftShader Device (LLVM 10.0.0)"` | CPU | RESULT: PASS — 비트 동일 |
| `MoltenVK_icd.json` | — | — | **FAIL (init)** — 아래 §4.2 |

`kosmickrisp` 줄이 핵심입니다. **`type=INTEGRATED_GPU`, 이름이 `"Apple M1"`** 입니다 —
소프트웨어 래스터라이저가 아니라 **이 Mac 의 실제 GPU** 이고, Mesa 의 Vulkan-on-Metal 드라이버가
그것을 Metal 로 번역합니다. 즉 **에뮬레이터 없이, 호스트에서, 실제 Apple GPU 위에서 Vulkan 컴퓨트
셰이더가 돌고 CPU 와 비트 단위로 일치합니다.**

```
  device: "Apple M1" type=INTEGRATED_GPU api=1.3.335 driver=0x6463063
  vecadd          n=1048576 bit-identical=1048576/1048576 max_ulp=0   VERDICT: bit-identical to CPU
  matmul 64x64x64 n=4096    bit-identical=4096/4096       max_ulp=0   VERDICT: bit-identical to CPU
  matmul 96x80x64 n=6144    bit-identical=6144/6144       max_ulp=0   VERDICT: bit-identical to CPU
RESULT: PASS
```

### 4.2 MoltenVK 는 실패하고, 그 실패가 진단입니다

`MoltenVK_icd.json` 은 `"is_portability_driver": true` 이고, 로더가 이렇게 거부합니다:

```
[Vulkan Loader] ERROR | DRIVER: vkCreateInstance: Found drivers that contain devices which support
the portability subset, but the instance does not enumerate portability drivers! Applications ...
must set the VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR bit ... and enable the
VK_KHR_portability_enumeration instance extension.
```

**프로브의 결함이 아니라 프로브가 그 플래그를 안 켠 것이고, 안 켤 이유가 있었습니다** — 안드로이드
타깃에는 portability 드라이버가 없습니다. 호스트 macOS 를 시험 대상으로 삼기로 결정하면
`create_instance` 에 확장 하나와 플래그 하나를 추가하는 **몇 줄짜리 변경**입니다.
`kosmickrisp` 이 그것 없이 이미 되므로 급한 일은 아닙니다.

### 4.3 검증이 실패할 수 있는지 — 확인했습니다

CLAUDE.md §5.5. `kosmickrisp` (실제 M1 GPU) 경로에서 대조군을 돌렸습니다:

```
$ VK_PROBE_TAMPER=1024 ./vk_probe    ->  MISMATCH ×3, RESULT: PASS (comparison caught the perturbation)
$ VK_PROBE_TAMPER=1    ./vk_probe    ->  RESULT: FAIL (comparison MISSED the perturbation)
```

`VULKAN.md` §1 이 기록한 경계(2 ULP 허용, FMA 서명)가 이 드라이버에서도 그대로입니다.
**초록이 무조건 나오는 검증이 아니라는 것이 확인됐습니다.**

### 4.4 이것이 바꾸는 것과, 바꾸지 않는 것

바꾸는 것:

- **README `line 361` 의 근거와 `line 468` 의 "Wiring and correctness are testable on an emulator"**
  — 에뮬레이터가 필요 없습니다. 호스트에서 직접 됩니다. 반복이 훨씬 싸집니다.
- `VULKAN.md` §4.1 의 macOS FAIL 출력은 **여전히 맞지만 이제 조건부**입니다 —
  "시스템 위치에 로더가 없다" 이지 "이 기계에서 Vulkan 이 불가능하다" 가 아닙니다.
- `VULKAN.md` §5.3 이 조율 세션에 남긴 `ash` 대 `wgpu` 결정의 전제 하나가 흔들립니다.
  그 표는 *"`ash` 는 Apple 타깃에서 로더가 없어 못 돈다"* 를 근거로 들었는데, **돕니다.**
  다만 §5.3 의 결론까지 뒤집지는 않습니다 — 아래를 보십시오.

바꾸지 **않는** 것:

- **이 드라이버들은 우리 것이 아닙니다.** Android SDK 의 내부 구현 세부이고, 그것에 의존하는
  테스트는 사용자가 Android Studio 를 지웠거나 업데이트하면 조용히 사라집니다.
  **테스트 전용 · 개발 머신 전용으로만 써야 하고, 출하 경로가 되어서는 안 됩니다.**
- **여전히 실물 폰이 없습니다.** Adreno/Mali 는 미지입니다 (`VULKAN.md` §7 그대로).
- **성능은 여전히 재면 안 됩니다.** `kosmickrisp` 은 Metal 번역 계층이고, 여기서 나온 수치는
  Vulkan 의 성능이 아니라 번역기의 성능입니다.
- **§5.3 결정은 그대로 조율 세션의 것입니다.** Apple GPU 를 candle 의 `metal` 피처로 채울지
  우리 커널로 채울지는 이 관측이 답하지 않습니다. 오히려 관측은 **`ash` 쪽에 유리**합니다 —
  `ash` 의 유일한 약점이라던 "Apple 에서 못 돈다" 가 개발 머신 한정으로 해소되었으니까요.

### 4.5 새 시스템 의존성에 대한 판단 — 조율 세션의 몫

지시가 요구한 대로 **먼저 적고 실행하지 않았습니다.** Vulkan SDK 도 드라이버도
**설치하지 않았습니다.** 위 실측은 전부 이미 디스크에 있던 파일로 했습니다.

결정이 필요한 것은 하나입니다: **개발/CI 검증이 Android SDK 번들 ICD 를 가리키게 해도 되는가.**

| 선택지 | 비용 | 위험 |
|---|---|---|
| A. 에뮬레이터 번들 ICD 를 테스트에서 가리킨다 | **0** (이미 있음) | SDK 경로·버전에 묶임. SDK 업데이트로 조용히 깨짐 |
| B. Vulkan SDK 를 설치한다 (LunarG, MoltenVK 포함) | 설치 ~1 GB | 새 시스템 의존성 — **조율 세션 승인 필요** |
| C. 호스트 검증을 포기하고 에뮬레이터만 쓴다 | 반복 비용 큼 | 지금 상태 |

**추천은 A** 이되, 경로를 하드코딩하지 말고 `VK_DRIVER_FILES` 가 이미 설정된 경우에만 GPU 테스트를
돌리고 **없으면 건너뛰는 것이 아니라 명시적으로 skip 을 보고**하는 형태여야 합니다.
조용히 통과하는 테스트가 이 프로젝트에서 가장 나쁜 결과입니다.

---

## 5. 배선의 비용 — 아무도 만들지 않은 조각이 무엇인가

지시의 표현대로 **"배선에 아무도 만들지 않은 조각이 필요하다면 그것이 끝난 라운드"** 입니다.
그 조각은 이것입니다.

### 5.1 없는 조각: candle 밖에 사는 텐서 표현

`resolve()` 가 `candle_core::Device` 를 반환하고 그 enum 이 닫혀 있으므로,
**Vulkan 텐서는 `candle::Tensor` 가 될 수 없습니다.** 그러면 `PyTensorBase.inner: Repr` 에
네 번째 팔이 필요합니다.

**전례가 이미 있고, 그것이 이 사이징을 신뢰할 수 있게 만듭니다.** `Repr` 은 이미 세 팔입니다:

```rust
pub enum Repr {
    Dense(Tensor),                  // candle
    Meta { shape: Vec<usize> },     // 스토리지 없음
    Quant(...),                     // candle 의 QTensor -- 별도 타입 체계
}
```

`Quant` 팔의 주석이 정확히 우리 상황을 서술합니다 — *"candle 의 양자화는 `DType` 이 아니다 …
별도 타입 체계에 살고 … `&Tensor` 로 변환되지 않는다"*. **`Vulkan` 팔은 같은 이유로 같은 모양이
됩니다.**

그리고 그 전례가 **안전 기본값**까지 같이 줍니다. `tensor()` 헬퍼가 비-`Dense` 팔에서 거부하고,
**호출 지점이 396 곳**입니다:

```
$ grep -rn "\.tensor()" rust/torch_c/src/*.rs | wc -l     ->  396
$ grep -rn "Repr::Quant" rust/torch_c/src/*.rs | wc -l    ->   20
```

즉 **`Repr::Vulkan` 을 추가하면 396 개 호출 지점이 전부 자동으로 거부합니다.** 지시가 금지한
"조용히 CPU 로 폴백하는 장치" 가 **구조적으로 불가능**합니다 — 커널이 검사를 빼먹는 것으로
CPU 스토리지를 읽을 수 없습니다. op 은 하나씩 명시적으로 opt-in 됩니다.
`Quant` 가 20 곳만 다루는 것이 그 점진성의 실증입니다.

### 5.2 만들어야 하는 것, 순서대로

| # | 조각 | 없는가 | 크기 |
|---|---|---|---|
| 1 | `resolve()` 의 반환 타입을 `candle::Device` 밖으로 (예: `enum Backend { Candle(Device), Vulkan(VkCtx) }`) | **없음** | 시그니처 변경 — `resolve()` 호출 지점 전부 |
| 2 | Vulkan 컨텍스트 (인스턴스·장치·큐·커맨드 풀) 의 **수명 관리** — 프로브는 매 디스패치마다 만들고 부숩니다 (`VULKAN.md` §5.2 가 "런타임이 그래도 된다는 뜻이 아니다" 라고 명시) | **없음** | 새 모듈 |
| 3 | 버퍼 할당자 + 디스크립터·파이프라인 캐시 (`VULKAN.md` §5.2 가 `ash` 선택의 "실제 비용" 이라 부른 것) | **없음** | 새 모듈 |
| 4 | `Repr::Vulkan` 팔 + `Storage` 의 device 필드 확장 | **없음** | `Quant` 전례를 따름 |
| 5 | H2D / D2H 복사 (`.to("cpu")`, `.cpu()`, `torch.ones(..., device="vulkan")` 의 초기화) | **없음** | — |
| 6 | 커널 — op 하나당 SPIR-V 하나 | vecadd/matmul 셰이더는 **프로브에 있음**, 그 외 전부 없음 | 골든 `ops=203` 대비 **2** |
| 7 | dtype — 프로브는 f32 뿐. bf16/f16/int8 은 기능 비트만 확인됨, 커널 없음 | **없음** | — |

**6번이 규모를 말합니다.** 골든 스위트가 다루는 op 이 203 개인데 셰이더는 2 개 있고,
그중 프로젝트 op 에 대응하는 것은 사실상 `matmul` 하나입니다.

### 5.3 최소 end-to-end 하나의 비용 — `torch.ones(2,2, device="vulkan")`

지시가 제안한 가장 작은 것조차 위 표의 **1·2·3·4·5 전부**를 요구합니다. `ones` 는 커널조차
필요 없을 수 있지만(버퍼 채우기), 그 값을 파이썬으로 되돌리려면 D2H 가 있어야 하고, D2H 가
있으려면 컨텍스트와 할당자가 있어야 하고, 그것을 담으려면 `Repr` 과 `resolve()` 가 바뀌어야
합니다. **가장 얇은 수직 절단이 인프라 다섯 조각을 전부 지나갑니다.** 커널 하나를 더 얹는 비용은
그다음부터 작아집니다 — 곡선이 그런 모양이라는 것이 이 사이징의 핵심입니다.

`docs/training/BACKWARD5.md` 의 교훈대로 **이 숫자들은 전부 세거나 실행한 것**입니다 (396, 20, 21, 203, 2,
빌드 `EXIT=0`, 프로브 출력). 추정한 것은 하나도 없고, 대신 **사람-시간 추정을 내지 않았습니다** —
그것이야말로 측정에서 나오지 않는 숫자이기 때문입니다.

---

## 6. 이번 라운드에서 코드를 바꾸지 않은 이유

- §5.1 이 요구하는 `resolve()` 시그니처 변경은 **`tensor.rs`·`aten.rs` 를 광범위하게 건드립니다.**
  이번 지시는 `aten.rs` 의 커널 본문, `capture.rs`, `tape.rs` 를 만지지 말라고 했고, 다섯 에이전트가
  같은 크레이트에서 동시에 돌고 있습니다.
- **절반만 배선된 Vulkan 장치는 최악의 결과입니다.** `resolve()` 만 열고 커널을 안 넣으면
  거부 메시지가 `not_implemented` 에서 다른 곳으로 옮겨갈 뿐이고, 실수로 `Dense` 로 떨어지면
  **모든 테스트가 통과하면서 CPU 를 씁니다** — 지시가 명시적으로 금지한 결과입니다.
- 그래서 이번 라운드의 산출물은 **§4 의 관측**(README 의 한 줄을 정정하고 다음 라운드를 훨씬 싸게
  만듦)과 **§5 의 사이징**입니다.

## 7. 다음 라운드에 넘기는 것

1. **§4.5 의 A/B/C 를 결정합니다.** 호스트 GPU 검증 경로가 없으면 다음 라운드가 에뮬레이터에
   묶입니다.
2. **`VULKAN.md` §5.3 (`ash` vs `wgpu`) 를 결정합니다.** §4.4 가 `ash` 쪽 근거를 강화했지만
   결정 자체는 그대로 남아 있고, 이것이 §5.2 의 2·3번 코드를 누가 쓰는지를 정합니다.
3. 그다음이 §5.2 의 1→5 이고, **커널은 그 뒤**입니다. 순서를 바꾸면 인프라 없는 커널이 됩니다.

## 8. 규율

- **설치한 것 없음.** 로더·ICD 는 전부 이미 디스크에 있던 Android SDK 번들입니다.
- **`rust/torch_c` 를 변경하지 않았습니다.** 이 라운드의 변경은 `docs/devices/VULKAN2.md` 하나입니다.
- 빌드 종료 코드는 파일로 리다이렉트한 뒤 `$?` 로 읽었습니다. 파이프로 읽지 않았습니다.
- 프로브는 전부 포그라운드에서 돌았고 스스로 종료했습니다. **남긴 프로세스가 없습니다.**
- 커밋하지 않았습니다.
