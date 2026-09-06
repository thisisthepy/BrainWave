# Vulkan 배선 — 그런데 Metal 이 먼저다

**결론부터: 이번 라운드는 두 개의 장치를 붙였고, 순서를 바꿨습니다.**

| 질문 | 답 |
|---|---|
| `torch.ones(2,2,device="mps").cpu()` 가 진짜 GPU 에서 계산되는가 | **예.** Apple M1, candle 의 Metal 백엔드 |
| 그 기능을 켜는 비용은 | 의존 크레이트 12개, 아티팩트 **+1,065,984 B (+17.2%)**, Android·wasm 에는 **0** |
| `torch.ones(2,2,device="vulkan").cpu()` 가 진짜 GPU 메모리를 지나는가 | **예.** `VkBuffer` 왕복, ICD 는 `kosmickrisp` → `Apple M1` |
| 조용한 CPU 폴백이 일어날 수 있는가 | `vulkan` 은 **구조적으로 불가능**. `mps` 는 **가능하다** — §3 |

`docs/VULKAN2.md` 는 사이징 라운드였고 코드를 쓰지 않았습니다. 이 문서는 그 사이징 위에 실제로
얹은 것과, **사이징이 예상하지 않았던 것 하나**(§3)를 적습니다.

---

## 1. 왜 Metal 을 먼저 했는가 — 비대칭이 하나뿐이기 때문

`docs/VULKAN2.md` §5.1 의 발견은 이렇게 요약됩니다:

```rust
pub enum Device { Cpu, Cuda(CudaDevice), Metal(MetalDevice) }   // 닫힌 enum
```

Vulkan 핸들을 **넣을 자리가 없다**는 것이 그 문서의 핵심이었습니다. 그런데 같은 문장이
`Metal` 에 대해서는 정반대를 말합니다 — **자리가 이미 있습니다.** 그래서 두 장치의 작업량은
같은 종류가 아닙니다:

| | `mps` | `vulkan` |
|---|---|---|
| candle 백엔드 | **있음** (`metal` 피처, 꺼져 있었을 뿐) | 없음. 소스에 `vulkan` 0 건 |
| 텐서 표현 | 평범한 `Repr::Dense(candle::Tensor)` | **`Repr::Vulkan(VkTensor)` — 네 번째 팔** |
| 커널을 가르쳐야 하는가 | **아니오. 한 개도.** candle 의 op 이 GPU 에서 돈다 | 예, `dispatch` 에 이름으로 하나씩 |
| 이번 라운드의 변경 | `resolve()` 팔 하나 + 캐시 + `Cargo.toml` 한 줄 | 새 모듈 827 줄, SPIR-V 셰이더, `Repr` 팔 |
| 성능 | **실제 가속기 전부** | 번역 계층(kosmickrisp) 위의 한 조각 |

지시의 판단 — *"mps 가 Vulkan 보다 훨씬 싸고, 가속기의 조각이 아니라 진짜 가속기를 준다"* — 는
맞았습니다. `resolve()` 변경 **하나만으로** `torch.ones(2,2,device="mps")` 가 닿았고,
`Repr` 팔은 필요 없었습니다.

### 1.1 `metal` 피처는 왜 꺼져 있었는가

`Cargo.toml` 이 `candle-core = { default-features = false }` 로 **전부** 못박고 있었고, 그 주석의
이유는 *"미래의 upstream 기본값이 Accelerate/Metal/MKL/CUDA 를 디바이스 빌드에 조용히 링크하지
못하게"* 였습니다. 즉 **Metal 을 배제한 것이 아니라 기본값을 배제한 것**이고, README 가 적어둔
"build isolation" 은 그것을 가리킵니다. 그 근거는 **지금도 유효합니다** — 그래서 블랭킷 핀은
그대로 두고, Accelerate 가 이미 쓰던 방식대로 **타깃별 예외 항목**을 하나 더 놓았습니다.

```toml
[target.'cfg(target_vendor = "apple")'.dependencies]
candle-core = { version = "0.11.0", default-features = false, features = ["metal"] }
```

Accelerate 항목에 합치지 **않은** 이유가 하나 있고, 그것이 잘못된 게이트를 하나 막습니다.
Accelerate 항목은 `not(torch_c_no_accelerate)` 로 묶여 있는데, 그 cfg 는
`scripts/device_android.sh parity` 가 호스트 빌드에서 **BLAS 를 빼려고** 켜는 것입니다.
거기에 metal 을 얹으면 parity 빌드가 **`mps` 장치까지 잃고**, §5 의 mps 테스트들은 실패가 아니라
**스킵**됩니다 — 아무 말 없이 검사를 그만두는 게이트입니다. Cargo 는 같은 의존성의 여러 항목에서
피처를 합집합으로 모으므로, 두 항목은 평소 `["accelerate","metal"]`, parity cfg 아래에서는
`["metal"]` 이 되어 각각이 원하는 것과 일치합니다.

### 1.2 비용 — 실측

같은 체크아웃에서 항목만 넣었다 뺐다 하며 잰 값입니다.

| | metal 없이 | metal 켜고 | 차이 |
|---|---|---|---|
| `release/lib_C.dylib` | 6,198,736 B | 7,264,720 B | **+1,065,984 B (+17.2%)** |
| 증분 재링크 | 15 s | 16 s | 무시 가능 |
| 최초 1회 의존성 컴파일 | — | — | 크레이트 12개, 약 70 s (8코어 M1, 다른 에이전트 9개 동시 실행 중) |

늘어난 크레이트 12개: `objc2`, `objc2-encode`, `block2`, `objc2-core-foundation`, `dispatch2`,
`objc2-foundation`, `objc2-metal`, `candle-metal-kernels`, 그리고 그것들이 끌고 오는
`tracing`, `tracing-core`, `tracing-attributes`, `pin-project-lite`.
**셰이더는 candle 이 런타임에 자기가 임베드한 소스에서 컴파일**하므로 빌드 스텝도, 함께 배포할
`.metallib` 도 없습니다.

### 1.3 크로스 빌드 — 실측, 하나도 깨지지 않았다

`cargo tree -e features` 로 확인한 실제 도달 범위:

| 타깃 | candle-core 피처 | 빌드 |
|---|---|---|
| `aarch64-apple-darwin` | `accelerate`, `metal` | EXIT=0 |
| `aarch64-apple-ios-sim` | `accelerate`, `metal` | **EXIT=0** — `otool -L` 에 `Metal.framework` 가 실제로 박힘 |
| `aarch64-linux-android` | **없음** | EXIT=0 (`scripts/device_android.sh build`) |
| `wasm32-unknown-emscripten` | **없음** | 해당 없음 |

Android 와 wasm 은 `target_vendor = "apple"` 에 매칭되지 않으므로 **의존성 그래프에 들어오지도
않습니다.** iOS 시뮬레이터는 매칭되는 유일한 다른 타깃이라 실제로 링크까지 돌려서 확인했습니다.

두 크로스 빌드 모두 **아무 환경변수 없이 부르면 실패하는데, 그것은 이 변경과 무관합니다** —
`pyo3-ffi` 는 `PYO3_CROSS_LIB_DIR`(+ iOS 는 `PYO3_CONFIG_FILE`, `docs/RUST_CROSSBUILD.md` §의
레시피)를, Android 의 `onig_sys` 는 NDK 를 요구합니다. 문서의 레시피대로 부르면 둘 다 EXIT=0 입니다.

---

## 2. `resolve()` 한 팔 — 그리고 그 팔이 처음에 틀렸던 방식

```rust
#[cfg(target_vendor = "apple")]
"mps" => Self::metal_device(self.index.unwrap_or(0).max(0) as usize),
```

**첫 판은 여기서 바로 `Device::new_metal(...)` 을 불렀고, 그것이 버그였습니다.**
`new_metal` 은 조회가 아니라 **생성자**입니다 — 자기 `MTLCommandQueue` 를 열고 새 id 를 받습니다.
candle 의 `same_device` 는 그 id 를 비교하므로, 두 번 `resolve()` 한 두 `mps` 텐서가 서로를
다른 장치로 보았고 `aten.rs` 의 mixed-device 게이트가 이렇게 거절했습니다:

```
Expected all tensors to be on the same device, but found at least two devices, mps:0 and mps:0!
```

**같은 장치를 두 번 부르는 메시지**가 호출당 생성자의 겉모습입니다. `device.rs::metal_device` 가
인덱스별로 하나씩 캐시해서 `resolve()` 를 조회로 만듭니다 — 모든 호출자가 이미 그렇다고 가정하고
있었고 `Cpu` 팔은 언제나 그랬던 것. `OnceLock<Device>` 가 아니라 인덱스 키의 `Vec` 인 이유는
두 GPU 짜리 Mac 에서 `mps:1` 이 조용히 0번을 받으면 안 되기 때문입니다.

`test_two_mps_tensors_are_on_the_same_mps_device` 가 그 캐시를 지우면 깨지는 단 하나의 테스트입니다.

---

## 3. 사이징이 예상하지 않았던 것 — **`mps` 는 조용한 CPU 폴백이 가능하다**

이것이 이 라운드에서 가장 보고할 가치가 있는 발견이고, **두 장치가 같은 등급의 안전성을 갖지
않는다**는 뜻입니다.

`vulkan` 은 `Repr::Vulkan` 이라 `PyTensorBase::tensor()` 가 거절합니다. 그 함수의 호출처는
**396곳**이고, 가르치지 않은 커널은 `PyResult` 를 손에 쥐게 되어 **거절하는 것 말고 할 수 있는
일이 없습니다.** 조용한 CPU 폴백은 규율이 아니라 **표현 불가능**합니다. 이 설계 결론은
`docs/VULKAN2.md` 가 세운 것이고 이번에 약화시키지 않았습니다.

`mps` 에는 그 보호가 **없습니다.** mps 텐서는 `Repr::Dense` 이므로 `tensor()` 가 그냥 내어주고,
호출처는 아무것도 확인하도록 강제되지 않습니다. 사이에 서 있는 것은 이 크레이트의 타입 시스템이
아니라 **candle 의 Metal 백엔드가 거절해 주는 것**뿐입니다. 실측:

| 호출 | 결과 |
|---|---|
| `aten.sort.default` | `RuntimeError: aten.sort.default: candle: Metal contiguous to_dtype F32 F64 not implemented` |
| `aten.cumsum.default` | 같은 형태 |
| `aten.full.default` (f64 채움) | `candle: unsupported const-set f64` |
| `aten.median.default` | `NotImplementedError` — 이 shim 이 아예 구현하지 않은 op |
| **`aten.nonzero.default`** | **성공. 값도 맞음. 그런데 계산은 CPU 에서 일어났다** |
| **`aten.tril.default`** | 같음 |

마지막 두 줄이 핵심입니다. candle 은 필요한 경우 Metal 텐서를 호스트로 **되읽어** 계산하고
결과를 다시 mps 로 올립니다. **답은 맞습니다 — 틀린 답이 나오는 경로는 아닙니다.** 하지만
`mps` 라벨이 붙은 텐서에 대해 **GPU 가 계산하지 않았고, 아무도 그렇게 말해주지 않습니다.**

지시가 "가장 나쁜 결과" 로 지목한 것이 정확히 이 모양이므로 감추지 않고 적습니다. 다만 두 가지가
다릅니다: (1) 그 폴백은 이 크레이트가 아니라 candle 안에서 일어나고, (2) 잘못된 값이 아니라
잘못된 *장치* 를 만듭니다. `vulkan` 에서는 둘 다 일어날 수 없습니다.

`test_an_op_mps_cannot_run_refuses_and_names_the_op` 는 이 약한 성질을 **약한 채로** 단언합니다 —
강하게 적었다면 그 테스트가 거짓말이 됩니다.

---

## 4. Vulkan — `docs/VULKAN2.md` 가 시킨 대로

`VULKAN2.md` §5 의 설계 결론을 그대로 구현했습니다. 약화시킨 곳은 없습니다.

* `Repr::Vulkan(VkTensor)` — 네 번째 팔. `Repr::Quantized` 와 같은 구조적 이유(§5.1).
* `tensor()` 는 이 팔에서 거절하고, 그 거절이 396 호출처에 자동으로 걸린다.
* `vulkan::dispatch` 가 **이름으로** 가르친 op 만 계산한다. 나머지는 **op 이름을 대며** 거절한다.
* 폴백 없음. 로더가 없으면 `dlopen` 의 원문을 그대로 올려 raise 한다.
* f32 · 연속 · 같은 shape 만. 나머지는 근사하지 않고 이름을 대며 거절한다.

**실측 (호스트 macOS, 아무것도 설치하지 않음):**

```
$ V=~/Library/Android/sdk/emulator/lib64/vulkan
$ DYLD_LIBRARY_PATH=$V VK_DRIVER_FILES=$V/libkosmickrisp_icd.json ...

_C._vulkan_probe() -> {'available': True, 'device': 'Apple M1',
                       'type': 'INTEGRATED_GPU', 'queue_family': 0, 'error': None}
_C._vulkan_ops()   -> ['aten._to_copy.default', 'aten.add.Tensor',
                       'aten.alias.default', 'aten.detach.default']

ones(2,2,device="vulkan").cpu()   -> [[1.0, 1.0], [1.0, 1.0]]
add(ones(2,3), ones(2,3)).cpu()   -> [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]     SPIR-V 커널
mul(...)                          -> NotImplementedError:
    aten.mul.Tensor: not implemented for the vulkan device. This build teaches
    the vulkan device four ops by name -- ...
```

`~/Library/Android/` 에는 **읽기만 했고 아무것도 쓰지 않았습니다.** 설치한 것도 없습니다.

**성능은 재지 않았습니다.** kosmickrisp 은 Vulkan-on-Metal 번역 계층이고
(`VULKAN2.md` §4.4), 이 기계에서는 에이전트 아홉 개가 동시에 돌고 있었습니다. 여기서 나온 숫자는
번역기와 부하를 묘사할 뿐 장치를 묘사하지 않습니다.

---

## 5. 두 장치가 검사받는 방식이 다르다 — 그리고 그래야 한다

`pytests/test_shim.py` 에 11개가 늘었습니다. 두 묶음이 **서로 다른 성질**을 단언합니다.

| 테스트 | 무엇을 단언하는가 |
|---|---|
| `test_mps_ones_round_trips_through_the_gpu` | 라벨이 팩토리를 통과하고 `.cpu()` 가 값을 되가져온다 |
| `test_mps_elementwise_and_matmul_agree_with_cpu_element_for_element` | **값**. `mul`·`matmul`·`sum` 이 CPU 와 원소 단위로 일치 |
| `test_two_mps_tensors_are_on_the_same_mps_device` | §2 의 회귀. 캐시를 지우면 이것만 깨진다 |
| `test_an_op_mps_cannot_run_refuses_and_names_the_op` | `sort` 가 op 이름과 `Metal` 을 대며 거절. §3 의 약한 성질을 약하게 |
| `test_mps_is_refused_by_name_where_it_is_not_compiled_in` | 스킵이 안전한 이유: 없는 곳에서는 **시끄럽게** 실패한다 |
| `test_vulkan_probe_answers_the_same_question_the_device_does` | 프로브와 장치가 어긋나면 모든 스킵이 거짓 통과가 된다 |
| `test_vulkan_ones_round_trips_through_a_real_gpu_buffer` | `VkBuffer` 왕복. `zeros` 도 함께 — `1.0` 이 상수가 아님을 보이려고 |
| `test_vulkan_add_runs_the_spirv_kernel_and_agrees_with_cpu` | 두 번 더해 피연산자를 구별시킨다. `2.0` 을 쓰기만 하는 커널은 통과 못 함 |
| `test_an_op_vulkan_was_not_taught_refuses_and_names_the_op` | **후보를 `_vulkan_ops()` 에 없다는 이유로 고른다.** 다음 라운드가 `mul` 을 가르치면 스스로 다른 것을 고른다 |
| `test_a_vulkan_tensor_has_no_cpu_storage_to_read` | 위 성질의 *기전*: `tolist()` 가 거절하고 `.cpu()` 를 가리킨다 |
| `test_the_checked_in_spirv_is_not_stale` | `.comp` 를 고치고 `compile.sh` 를 잊으면 옛 커널이 조용히 나간다 |

**스킵은 이름을 대고 합니다.** Metal 도 Vulkan 도 없는 Linux·Windows 러너에서 부재로 실패하는
테스트는 망가진 게이트이고, 조용히 통과하는 테스트는 게이트가 아닙니다. 그래서 스킵 줄이
로더의 원문이나 `resolve()` 의 거절문을 그대로 인쇄합니다.

두 묶음을 섞지 않은 것이 의도입니다. `mps` 에서 볼 것은 **산술**이고 (§1 의 이유로 배선은 팔 하나뿐),
`vulkan` 에서 볼 것은 **거절**입니다 (§3 의 이유로 그것이 이 표현이 존재하는 이유).

---

## 6. 게이트

| | 값 |
|---|---|
| `pytests/run.sh` (로더 없음) | **426 ok**, 0 FAIL, EXIT=0 — vulkan 넷은 이름을 대고 스킵 |
| 같은 스위트, 로더 있음 | **426 ok**, 0 FAIL, EXIT=0 — **스킵 0개.** mps 와 vulkan 이 둘 다 실제 M1 위에서 |
| DOCWATCH | **PASS — 417/417** |
| golden `compare.py` | **8681/8681**, ops=207 — **움직이지 않음** |
| `aarch64-linux-android` | EXIT=0 (`scripts/device_android.sh build`) |
| `aarch64-apple-ios-sim` | EXIT=0 (`PYO3_CONFIG_FILE` 레시피) |

golden 이 움직이지 않은 것이 이 라운드의 음성 대조군입니다. 새 장치는 CPU 결과를 하나도 바꾸지
않아야 하고, 바꿨다면 의도하지 않은 것을 건드린 것입니다.

### 6.1 함정 — `run.sh` 를 통해서는 Vulkan 이 절대 보이지 않는다

`DYLD_LIBRARY_PATH=... sh rust/torch_c/pytests/run.sh` 로 돌리면 **로더를 정확히 가리켰는데도
vulkan 테스트 넷이 전부 스킵됩니다.** macOS 의 SIP 가 보호된 바이너리(`/bin/sh`)를 exec 할 때
환경에서 `DYLD_*` 를 **떼어냅니다.** 그래서 변수는 셸에는 있고 파이썬에는 없습니다.

**이것이 이 라운드에서 거짓 초록에 가장 가까웠던 지점입니다** — 스킵 줄이 "no vulkan" 이라고
말하는데 그 이유가 사실이 아니고, 로더를 준 사람은 자기가 준 줄 압니다. 스킵이 로더의 원문을
인쇄하도록 해 둔 것이 이것을 잡았습니다.

로더가 붙은 채로 스위트를 돌리려면 **파이썬을 직접** 부릅니다:

```sh
V=~/Library/Android/sdk/emulator/lib64/vulkan
env DYLD_LIBRARY_PATH=$V VK_DRIVER_FILES=$V/libkosmickrisp_icd.json \
    PYTHONPATH=<stage> $PY rust/torch_c/pytests/test_shim.py
```

`run.sh` 는 이 라운드의 담당 범위가 아니라 고치지 않았습니다. 고친다면 `run.sh` 가 파이썬을
exec 하기 직전에 `DYLD_LIBRARY_PATH` 를 다시 세우는 것이 맞습니다.

---

## 7. 남은 것

* **`mps` 의 CPU 되읽기(§3)를 관측 가능하게 만들 것.** 지금은 조용합니다. `Repr` 을 건드리지 않고
  할 수 있는 최소한은, mps 텐서를 받은 커널이 dense storage 를 실제로 읽었는지 세는 계수기입니다.
* **Vulkan 의 할당자.** 텐서당 `vkAllocateMemory` 하나입니다. `maxMemoryAllocationCount` 가
  모바일에서 4096 인 것을 감안하면 2×2 에는 맞고 모델에는 틀린 모양입니다 (`VULKAN2.md` §5.2 3번).
* **Vulkan 커널 넷.** `mul`·`sub`·브로드캐스트·`alpha` 는 전부 이름을 대며 거절 중입니다. 늘리는
  방법은 `dispatch` 에 한 줄씩이고, 늘리지 않는 동안에도 답이 틀릴 수는 없습니다.
* **`mps` 의 f64.** `full` 이 f64 채움에서 candle 의 `unsupported const-set f64` 로 막힙니다.
  `dtype=float32` 를 주면 되지만, 팩토리가 그것을 알아서 하지는 않습니다.
