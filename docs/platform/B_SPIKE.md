# B 경로 빌드 스파이크 — selective libtorch 크로스 컴파일

DESIGN.md §6 이 "B 에 남은 유일한 미지수" 라고 지목한 것 — **크로스 컴파일이 완료되는가** — 을
타임박스를 걸고 실제로 빌드해서 판정한 기록입니다.

- 실행일: 2026-08-23
- 타임박스: 60분 (실제 소요 ~31분)
- 작업 디렉터리: `/Volumes/macMini/caches/pytorch-spike`
- 대상: PyTorch main `16ee93b8a9e7e2dd215a2f5d07eada25e31ec123` (2026-08-23), `version.txt` = `2.15.0a0`
- 타깃: `aarch64-none-linux-android21` (arm64-v8a), NDK 27.1.12297006

---

## 0. 요약

**크로스 컴파일은 뚫렸습니다.** 전체 op 빌드와 최소 op 선택 빌드 둘 다 `libtorch.a` /
`libtorch_cpu.a` 를 aarch64 로 산출했고, 두 빌드 모두 종료 코드 0 으로 끝났습니다.
막힌 지점은 다섯 곳이었으나 전부 얕았고, 총 31분 안에 양쪽이 완료됐습니다.

**그러나 이 결과는 B 를 승인하지 않습니다.** 스파이크 도중 §6 의 전제 자체가 틀렸다는 것이
드러났기 때문입니다 — 모바일 빌드 경로는 `BUILD_PYTHON` 을 **옵션이 아니라 강제로 OFF** 로
덮어씁니다 (`CMakeLists.txt:917`). 즉 이 경로가 만들어 주는 것은 lite interpreter 용
`libtorch.a` 이지 **torchnative 가 필요로 하는 `torch._C` (CPython 확장 모듈) 가 아닙니다.**

| | 결과 |
|---|---|
| Android arm64 크로스 컴파일이 완료되는가 | **예** (§2, §3) |
| 그 산출물이 `torch._C` 인가 | **아니오** — 모바일 경로가 BUILD_PYTHON 을 강제 OFF (§4) |
| 상류 지원이 있는가 | **없음.** CI 참조 0건, 진입점 스크립트 삭제됨 (§1) |
| iOS 도 같은가 | **미확인, 그리고 Android 보다 확실히 나쁨** — 툴체인 파일 자체가 삭제됨 (§5) |

---

## 1. (a) `build_mobile.sh` 와 선택 빌드 경로는 살아 있는가

### 진입점 스크립트는 삭제됐다 — DESIGN.md §5 의 서술은 낡았습니다

DESIGN.md §5 는 "`scripts/build_mobile.sh` 가 지금도 main 에 있음" 이라고 적고 있으나,
**현재 main 에 없습니다.**

```
$ ls scripts/
analysis  build_host_protoc.sh  codeowners  compile_tests  export  hud
install_triton_wheel.sh  jit  lint_urls.sh  lint_xrefs.sh  lintrunner.py
onnx  README.md  release  release_notes  setup_hooks.py  write_metallib_headers.py
```

`build_mobile.sh` · `build_android.sh` · `build_ios.sh` 전부 없습니다
(`find` 로 트리 전체 확인, 일치 0건).

삭제 시점을 GitHub API 로 특정했습니다.

| 커밋 | 날짜 | 제목 |
|---|---|---|
| `91602a92548d` | 2025-07-23 | Cleanup old caffe2 scripts (#158475) |
| `ced5cf042de1` | 2025-07-17 | Revert "Cleanup old caffe2 scripts (#158475)" |
| `94d7f0c1ef9a` | 2025-07-17 | Cleanup old caffe2 scripts (#158475) |

한 번 머지됐다가 되돌려지고 엿새 뒤 다시 머지됐습니다. 태그로도 확인됩니다 —
`raw.githubusercontent.com` 조회 결과 `v2.8.0` 은 200, `v2.9.0` 은 404 입니다.

### 그러나 CMake 기계 장치는 남아 있다

삭제된 것은 **셸 진입점뿐이고**, 선택 빌드를 실제로 수행하는 CMake/codegen 층은 전부 살아 있습니다.

| 요소 | 위치 | 상태 |
|---|---|---|
| `SELECTED_OP_LIST` 캐시 변수 | `CMakeLists.txt:658` | 있음 |
| `TRACING_BASED` 옵션 | `CMakeLists.txt:677` | 있음 |
| `BUILD_LITE_INTERPRETER` 옵션 | `CMakeLists.txt:300` | 있음 |
| `INTERN_BUILD_MOBILE` 분기 | `CMakeLists.txt:815-816` (ANDROID 또는 IOS 이면 자동 ON) | 있음 |
| 선택적 codegen | `cmake/Codegen.cmake:185-217` (`--op_selection_yaml_path`) | 있음, **동작 확인** |
| `selected_mobile_ops.h` 생성 | `caffe2/CMakeLists.txt:1296-1335` | 있음, **동작 확인** |
| `model_tracer` | `torch/csrc/jit/mobile/model_tracer/` | 존재 (실행은 미확인) |
| `android/` Gradle 프로젝트 | `android/` | 존재 |

즉 **문서에만 있고 코드가 없는 경우는 아닙니다.** pypackpack `Cargo.kt` 전례와는 다릅니다 —
알맹이는 있고 손잡이만 떨어져 나갔습니다. 그래서 `v2.8.0` 의 `build_android.sh` 를 받아
그 안의 cmake 인자를 그대로 재구성하는 것으로 진입할 수 있었습니다.

### 상류 지원 0 은 정량적으로 확인된다

DESIGN.md §5 의 "상류 지원: **0**" 은 과장이 아니라 정확한 서술입니다.

- **CI 참조 0건.** `.github/` 와 `.ci/` 전체에서
  `custom_build` · `build_mobile` · `SELECTED_OP_LIST` · `TRACING_BASED` ·
  `BUILD_LITE_INTERPRETER` · `pytorch_android` 를 grep 한 결과 **일치 파일 0개**입니다.
  모바일 빌드를 도는 CI 잡이 하나도 없습니다.
- **in-tree 테스트가 dangling reference 다.** `test/mobile/custom_build/build.sh:41,55` 가
  `${SRC_ROOT}/scripts/build_mobile.sh` 를 호출하는데 그 파일이 없습니다. 이 테스트는
  현재 실행 불가능하며, 실행하는 CI 도 없어서 아무도 모릅니다.
- **`android/common.sh:66` 도 같습니다** — 삭제된 `scripts/build_android.sh` 를 호출합니다.

**호출자를 남겨둔 채 피호출자만 지웠다는 것이 이 경로의 상태를 가장 잘 말해줍니다.**

---

## 2. (b) 실제 빌드 시도 — 전체 op 빌드

`v2.8.0` `build_android.sh` 의 인자를 현재 main 에 그대로 적용했습니다. 막힌 지점이 셋이었고,
셋 다 우회 가능했습니다.

### 막힌 지점 1 — Android SDK 번들 cmake 가 너무 낡음 (환경 문제)

```
CMake Error at CMakeLists.txt:46 (cmake_minimum_required):
  CMake 3.27 or higher is required.  You are running version 3.22.1-g37088a8
```

`~/Library/Android/sdk/cmake/` 에는 3.22.1 만 설치돼 있습니다. Homebrew 의 cmake 3.28.3 을 쓰고,
`ninja` 만 Android SDK 쪽(`cmake/3.22.1/bin/ninja`)에서 가져오는 것으로 해결했습니다
(이 기계에 별도 ninja 가 없습니다). **저장소 문제가 아니라 환경 문제입니다.**

### 막힌 지점 2 — eigen 서브모듈이 사라졌는데 모바일 전용 경로가 아직 참조함

```
CMake Error at cmake/External/EigenBLAS.cmake:46 (add_library):
  Cannot find source file:
    .../pytorch/third_party/eigen/blas/single.cpp
```

`cmake/External/EigenBLAS.cmake:16` 이 `third_party/eigen/blas` 를 하드코딩하는데,
**eigen 은 더 이상 서브모듈이 아닙니다** — `.gitmodules` 에 eigen 항목이 없고,
`third_party/` 에는 `eigen_pin.txt` (내용: `5.0.1`) 만 있습니다. 빌드 타임에 받아오는
의존성으로 바뀌었습니다.

이 경로는 `EigenBLAS.cmake:6` 의 `if(NOT INTERN_BUILD_MOBILE OR NOT INTERN_USE_EIGEN_BLAS) return()`
가드 때문에 **모바일 빌드에서만 도달합니다.** 그리고 `CMakeLists.txt:923-924` 가
`USE_BLAS` 가 켜져 있으면 `INTERN_USE_EIGEN_BLAS` 를 ON 으로 만들므로, 기본 설정의 모든 모바일
빌드가 여기 걸립니다.

**eigen 을 서브모듈에서 뺀 변경이 모바일 BLAS 경로를 깼고, 모바일을 도는 CI 가 없어 아무도
모르는 상태입니다.** `-DUSE_BLAS=OFF` 로 우회했습니다.

> 주의: `USE_BLAS=OFF` 는 성능에 영향이 있습니다. 이 스파이크는 빌드 성립 여부만 판정하므로
> 성능은 측정하지 않았습니다 (§6 미확인 항목).

### 막힌 지점 3 — NDK 가 Vulkan 래퍼를 없앴는데 Android 기본값이 Vulkan ON

```
CMake Error at cmake/VulkanDependencies.cmake:18 (add_library):
  Cannot find source file:
    .../ndk/27.1.12297006/sources/third_party/vulkan/src/common/vulkan_wrapper.h
```

`CMakeLists.txt:410` 이 `cmake_dependent_option(USE_VULKAN "Use Vulkan GPU backend" ON "ANDROID" OFF)`
— **Android 이면 Vulkan 이 기본 ON** 입니다. 그런데 `cmake/VulkanDependencies.cmake:11-22` 가
`$ANDROID_NDK/sources/third_party/vulkan/` 를 기대하고, NDK 27 에는 그게 없습니다.

```
$ ls ~/Library/Android/sdk/ndk/27.1.12297006/sources/third_party/
googletest  shaderc
```

Google 이 NDK 에서 Vulkan 래퍼를 제거했고 PyTorch 쪽은 따라가지 않았습니다.
**즉 최신 NDK 로는 Android 빌드가 기본 설정에서 무조건 실패합니다.**
`-DUSE_VULKAN=OFF` 로 우회했습니다.

### 막힌 지점 4 — `cpuinfo` 가 WHOLE_ARCHIVE 와 일반 링크로 동시에 걸림

```
CMake Error at caffe2/CMakeLists.txt:855 (add_library):
  Impossible to link target 'torch_cpu' because the link item 'cpuinfo',
  specified with the feature 'WHOLE_ARCHIVE', has already occurred without
  any feature or 'DEFAULT' feature, which is not allowed.
```
(`caffe2/CMakeLists.txt:978` 에서 `torch` 에 대해서도 동일)

원인은 `c10/CMakeLists.txt:122-124` 입니다.

```cmake
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(s390x|ppc64le)$")
  target_link_libraries(c10 INTERFACE "$<BUILD_INTERFACE:cpuinfo>")
  target_link_libraries(c10 PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,cpuinfo>")
endif()
```

같은 `cpuinfo` 를 INTERFACE 로는 feature 없이, PRIVATE 로는 `WHOLE_ARCHIVE` 로 겁니다.
CMake 는 한 링크 항목에 상충하는 feature 가 붙는 것을 거부합니다.

> **추론 (미검증):** 정적 빌드(`BUILD_SHARED_LIBS=OFF`, 모바일이 강제)에서만 PRIVATE 의존성이
> `$<LINK_ONLY:>` 로 소비자에게 전파되어 INTERFACE 쪽과 충돌하는 것으로 보입니다. 공유 빌드에서는
> 전파되지 않아 드러나지 않습니다. **이 가설은 공유 빌드로 대조 실험을 하지 않았으므로 미검증이며,
> 확인된 사실은 "이 정적 모바일 구성에서 재현된다" 까지입니다.**

`c10/CMakeLists.txt:124` 를 일반 링크로 바꿔 우회했습니다.

### 결과 — 전체 op 빌드는 완료된다

위 셋을 우회한 뒤 configure 가 **종료 코드 0** 으로 끝났고, 컴파일도 완주했습니다.

```
$ ninja -j8 ; echo "BUILD_EXIT=$?"
[849/850] Linking CXX static library lib/libtorch_cpu.a
[850/850] Linking CXX static library lib/libtorch.a
BUILD_EXIT=0
```

산출물이 진짜 Android arm64 인지 확인했습니다.

```
$ llvm-objdump -f lib/libtorch_cpu.a
libtorch_cpu.a(CompositeViewCopyKernels.cpp.o):  file format elf64-littleaarch64
architecture: aarch64
```

| 산출물 | 크기 |
|---|---|
| `libtorch_cpu.a` | 194.2 MB |
| `libc10.a` | 3.5 MB |
| `libkineto.a` | 36.6 MB |
| `lib/` 전체 | 253 MB |

ninja 기준 컴파일 시간 **6.8분** (8코어, `-j8`).

---

## 3. (b) 실제 빌드 시도 — 최소 op 선택 빌드

op 10개짜리 목록으로 선택 빌드를 시도했습니다
(`/Volumes/macMini/caches/pytorch-spike/minimal_ops.yaml`):
`add.Tensor` · `mul.Tensor` · `addmm` · `mm` · `t` · `relu` · `view` ·
`empty.memory_format` · `softmax.int` · `matmul`.

`torchgen.selective_build.selector.SelectiveBuilder.from_yaml_path` 로 파싱 검증을 먼저 했습니다
(`include_all=False, n_ops=10`).

Configure 는 통과했고, 선택 codegen 이 실제로 동작합니다.

```
running gen_selected_mobile_ops_header for: .../minimal_ops.yaml
--   SELECTED_OP_LIST    : .../minimal_ops.yaml
```

### 막힌 지점 5 — 정적 디스패치 codegen 이 "모든 op 은 CPU 커널이 있다" 를 가정함

컴파일이 `[324/851]` 에서 멈췄습니다.

```
.../aten/src/ATen/ops/_scaled_grouped_mm_v2.h:19:10: fatal error:
  'ATen/ops/_scaled_grouped_mm_v2_cpu_dispatch.h' file not found
```

**처음에는 "선택 빌드가 헤더를 덜 생성한 것" 으로 오인했는데, 대조해 보니 아니었습니다.**
`*_cpu_dispatch.h` 개수는 전체 빌드와 선택 빌드가 **532개로 동일**하고, 문제의 헤더는
**양쪽 모두에 없습니다.** 차이는 생성된 헤더의 내용에 있었습니다.

```
$ diff full/ops/_scaled_grouped_mm_v2.h sel/ops/_scaled_grouped_mm_v2.h
18a19
> #include <ATen/ops/_scaled_grouped_mm_v2_cpu_dispatch.h>
```

즉 **선택 빌드에서만 존재하지 않는 헤더를 `#include` 하도록 생성됩니다.** 전체 빌드는 같은
소스 파일(`aten/src/ATen/native/mkldnn/ConvPrepack.cpp`)을 문제없이 컴파일합니다.

근본 원인을 특정했습니다.

1. `SELECTED_OP_LIST` 가 주어지고 정적 빌드이면 cmake 가 정적 디스패치로 자동 전환합니다 —
   빌드 로그에 `Switching to STATIC_DISPATCH_BACKEND=CPU.` (전체 빌드 로그에는 이 줄이 없음).
2. 정적 디스패치 codegen 은 op 헤더마다 `<ATen/ops/{op}_cpu_dispatch.h>` 를 **조건 없이** 넣습니다.
3. 그런데 `*_cpu_dispatch.h` 는 **CPU 커널이 있는 op 에 대해서만** 생성됩니다.
4. `_scaled_grouped_mm_v2` 는 `structured_delegate: _scaled_grouped_mm_v2.out` 이고,
   그 `.out` 의 디스패치 표가 **CUDA 전용**입니다 (`aten/src/ATen/native/native_functions.yaml:7158-7166`):

```yaml
- func: _scaled_grouped_mm_v2.out(...) -> Tensor(a!)
  structured: True
  variants: function
  dispatch:
    CUDA: _scaled_grouped_mm_cuda_v2_out
```

**규모는 작습니다.** 생성된 헤더 전체에서 `*_cpu_dispatch.h` include 는 1074건이고,
그중 파일이 없는 것은 **정확히 1건**입니다. 즉 구조적 붕괴가 아니라 **최근 추가된 CUDA 전용 op
하나가 정적 디스패치 경로를 깬 것**입니다.

빈 스텁 헤더만으로는 부족했습니다 — codegen 이 실제로 호출도 생성합니다
(`aten/src/ATen/Operators_3.cpp:4856`, `return at::cpu::_scaled_grouped_mm_v2(...)`).
`error: no member named '_scaled_grouped_mm_v2' in namespace 'at::cpu'`.
그래서 `TORCH_CHECK(false, ...)` 로 던지는 inline 스텁을 넣었습니다.

### 결과 — 선택 빌드도 완료된다

```
$ ninja -j8 ; echo "BUILD_SEL4_EXIT=$?"
[249/250] Linking CXX static library lib/libtorch_cpu.a
[250/250] Linking CXX static library lib/libtorch.a
BUILD_SEL4_EXIT=0
```

`elf64-littleaarch64` 로 확인했습니다. ninja 기준 마지막 패스 3.2분.

### 크기 — 선택 빌드의 이득이 이 측정에서는 작다

| | `libtorch_cpu.a` | `lib/` 전체 |
|---|---|---|
| 전체 op | 194.2 MB | 253 MB |
| 최소 op 10개 | **160.1 MB** | 226 MB |

**17.5% 감소에 그쳤습니다.** 다만 이 숫자를 DESIGN.md §5 의 "arm-v7 압축 4.5MB" 와 직접 비교하면
안 됩니다 — 여기서 잰 것은 **링크 전 정적 아카이브**이고, 선택 빌드의 크기 이득은 대부분
`-Wl,--gc-sections` 로 링크하고 strip 한 뒤에 나타납니다. 이 스파이크는 최종 바이너리를 링크하지
않았으므로 **배포 크기는 미확인입니다** (§6).

---

## 4. (d) 판단에 가장 크게 작용하는 발견 — 이 경로는 `torch._C` 를 만들지 않는다

스파이크의 원래 질문("크로스 컴파일이 완료되는가")에는 **예**라고 답했습니다. 그런데 그 과정에서
**질문 자체가 잘못 놓였다는 것**이 드러났습니다.

`CMakeLists.txt:907-929` 의 `if(INTERN_BUILD_MOBILE)` 블록이 하는 일입니다.

```cmake
if(INTERN_BUILD_MOBILE)
  ...
  set(BUILD_PYTHON OFF)      # :917
  set(BUILD_FUNCTORCH OFF)   # :918
  set(USE_DISTRIBUTED OFF)   # :919
  set(NO_API ON)             # :920
  set(USE_FBGEMM OFF)        # :921
  set(INTERN_DISABLE_ONNX ON)
  ...
```

`BUILD_PYTHON OFF` 는 **옵션이 아니라 강제 덮어쓰기**입니다. 그리고 `INTERN_BUILD_MOBILE` 은
`CMakeLists.txt:815` 에서 **`ANDROID` 또는 `IOS` 이면 자동으로 켜집니다.** 즉 NDK 툴체인 파일을
쓰는 순간 자동으로 켜지고, 파이썬 바인딩은 끌 수 없게 됩니다. 우리 빌드 요약에도
`BUILD_PYTHON : OFF` 로 찍혔고, 실제로 `torch/csrc` 의 파이썬 바인딩 산출물은 없습니다.

**이것이 A/B 결정에 결정적입니다.**

DESIGN.md §2 는 "네이티브인 것은 `torch._C` 하나뿐" 이라고 정리하고, §5 의 B 를
"selective libtorch 빌드" 라고 부릅니다. 그런데 **libtorch 와 `torch._C` 는 같은 물건이
아닙니다.** `torch._C` 는 libtorch + `torch/csrc/**` 의 CPython 바인딩 층이고,
**모바일 빌드 경로는 정확히 그 층을 끄는 것으로 정의돼 있습니다.**

그래서 DESIGN.md §5 의 표를 다시 읽으면 이 스파이크가 무엇을 재확인했는지 분명해집니다.

| | 실상 | 이 스파이크가 답한 것 |
|---|---|---|
| libtorch(C++) 를 Android / iOS 로 빌드 | 된다 | **재확인함 (예, 우회 5건 필요)** |
| TorchScript / `.pte` 를 기기에서 실행 | 된다 | 건드리지 않음 |
| 임베디드 CPython 에서 `import torch` | **아무도 안 했다** | **여전히 답하지 못함** |

**§6 이 "B 에 남은 유일한 미지수는 크로스 컴파일" 이라고 적은 것은 부정확합니다.**
크로스 컴파일은 셋 중 첫째 칸이고, 이미 된다고 알려져 있던 칸입니다. torchnative 가 필요로 하는
것은 셋째 칸이며, 그것은 이 스파이크로 좁혀지지 않았습니다. B 를 실제로 진행한다면
`CMakeLists.txt:917` 을 뚫고 크로스 컴파일 환경에서 `torch/csrc` 파이썬 바인딩을 빌드해야 하고,
**그 조합에는 선례가 없습니다** (그게 "아무도 안 했다" 칸입니다).

### 부수 발견 — 모바일 기본값은 autograd 도 끈다

`CMakeLists.txt:911-915` 가 `BUILD_MOBILE_AUTOGRAD` (옵션, `CMakeLists.txt:321`) 가
꺼져 있으면 `INTERN_DISABLE_AUTOGRAD ON` 으로 둡니다. **기본값이 autograd 없음**입니다.

이것은 DESIGN.md §3 축 1 의 **단계 1 (forward + 좁은 backward)** 에 직접 걸립니다.
다만 되살리는 옵션이 존재하므로 구조적 배제는 아닙니다. **`BUILD_MOBILE_AUTOGRAD=ON` 으로
빌드가 성립하는지는 이 스파이크에서 시도하지 않았습니다 (미확인).**

---

## 5. iOS — Android 결과를 외삽할 수 없다

torchnative 는 iOS 도 타깃이므로 따로 확인했습니다. **Android 보다 나쁩니다.**

| | v2.8.0 | v2.9.0 | 현재 main |
|---|---|---|---|
| `scripts/build_ios.sh` | 있음 (200) | **없음 (404)** | 없음 |
| `cmake/iOS.cmake` (툴체인) | 있음 (200) | 있음 (200) | **없음 (404)** |
| `ios/` 디렉터리 | 없음 | 없음 | 없음 |

Android 는 진입점 스크립트만 사라지고 `android/` Gradle 프로젝트와 cmake 배선이 남아 있어
`v2.8.0` 스크립트를 복원하는 것으로 진입할 수 있었습니다. **iOS 는 툴체인 파일
(`cmake/iOS.cmake`) 자체가 main 에서 사라졌습니다.** 남은 것은
`cmake/Dependencies.cmake:359` 의 `if(IOS)` 한 줄뿐입니다.

`CMakeLists.txt:815` 는 여전히 `IOS` 를 보고 `INTERN_BUILD_MOBILE` 을 켜므로 변수는 살아 있지만,
그 변수를 켜 줄 툴체인 파일이 없습니다. 외부 툴체인(예: `leetal/ios-cmake`)을 가져오면 될
가능성이 있으나 **시도하지 않았고, 미확인입니다.**

**iOS 는 "Android 가 됐으니 될 것" 으로 볼 수 없습니다.** Android 가 뚫린 이유는 삭제되지 않은
부분이 남아 있었기 때문인데, iOS 는 그 부분이 남아 있지 않습니다.

---

## 6. (c) 소요 시간과 디스크

| 항목 | 값 |
|---|---|
| **총 소요** | **~31분** (타임박스 60분 내) |
| shallow clone (`--depth 1`) | ~1분 |
| 서브모듈 (`--recursive --depth 1 --jobs 8`) | ~2분 |
| venv + 코드젠 의존성 (pyyaml, typing_extensions, setuptools) | ~1분 |
| configure 시도 4회 (실패 3 + 성공 1) | ~3분 |
| 전체 op 빌드 (ninja, `-j8`) | **6.8분** |
| 선택 빌드 (configure + 컴파일, 증분 포함) | **~5분** |
| 나머지 (조사·진단·GitHub API 조회) | ~12분 |

디스크 사용량 (전부 외장 `/Volumes/macMini`):

| 경로 | 크기 |
|---|---|
| `pytorch-spike/pytorch` (소스 + 서브모듈) | 3.6 GB |
| `pytorch-spike/build_android_arm64` (전체 op) | 607 MB |
| `pytorch-spike/build_android_arm64_sel` (선택) | 544 MB |
| `pytorch-spike/venv` | 21 MB |
| **합계** | **4.7 GB** |

외장 여유는 작업 후 170 GB 입니다. 내부 디스크는 건드리지 않았습니다.

### 확인하지 못한 것 (미확인)

추측으로 채우지 않고 명시합니다.

- **배포 크기.** 정적 아카이브만 쟀습니다. `--gc-sections` 링크 + strip 후의 실제 `.so`/실행 파일
  크기는 재지 않았으므로, DESIGN.md §5 의 "4.5MB" 와 비교 가능한 숫자가 없습니다.
- **기기 실행.** 산출물을 Android 기기/에뮬레이터에 올려 돌려보지 않았습니다. 아키텍처가
  `elf64-littleaarch64` 라는 것만 확인했습니다.
- **`import torch`.** 이 스파이크의 산출물에는 파이썬 바인딩이 없으므로 시도 자체가 불가능했습니다 (§4).
- **`TRACING_BASED=1` 과 `model_tracer`.** 소스는 존재하나 실행하지 않았습니다.
  DESIGN.md §5 가 기대한 "op 목록 자동 추출" 이 현재도 동작하는지는 **미확인**입니다.
- **`BUILD_MOBILE_AUTOGRAD=ON` 빌드.** 옵션 존재만 확인했습니다.
- **`USE_BLAS=OFF` 의 성능 영향.** 우회로 껐을 뿐 대안(eigen 을 pin 대로 받아 배치)을 시도하지 않았습니다.
- **`cpuinfo` WHOLE_ARCHIVE 충돌이 정적 빌드 전용인지.** 공유 빌드 대조 실험 미실시 (§2 막힌 지점 4).
- **iOS 크로스 컴파일** (§5).

---

## 7. (d) A/B 결정에 대한 결론

### B 의 크로스 컴파일은 "안 되는 것" 이 아니다

DESIGN.md §5 가 B 의 성격을 **"빌드 문제 (유한하고 기계적)"** 이라고 적은 것은 맞았습니다.
막힌 지점 다섯 곳 중 넷이 한 줄짜리 우회로 풀렸고, 다섯째도 op 하나에 국한된 스텁이었습니다.
30분 만에 두 개의 빌드가 완주했습니다. **"PyTorch 는 모바일로 크로스 컴파일이 안 된다" 는
통념은 이 측정으로 반박됩니다.**

그리고 §5 의 리스크 예상("`native_functions.yaml` 코드젠이 host==target 을 가정")은
**절반만 맞았습니다.** 코드젠에서 문제가 난 것은 맞지만 원인은 host/target 불일치가 아니라
**정적 디스패치 codegen 이 모든 op 에 CPU 커널이 있다고 가정한 것**이었습니다.

### 그러나 B 를 고를 근거는 되지 않는다

세 가지 이유입니다.

**1. 이 경로의 산출물은 `torch._C` 가 아니다 (§4).** 가장 중요한 이유입니다.
모바일 빌드는 `BUILD_PYTHON` 을 강제로 끄도록 정의돼 있고
(`CMakeLists.txt:917`), 그것이 켜지는 것은 `ANDROID`/`IOS` 를 보고 자동입니다
(`CMakeLists.txt:815`). 우리가 방금 증명한 것은 "libtorch.a 가 만들어진다" 이지
"`torch._C` 가 만들어진다" 가 아닙니다. **B 의 진짜 미지수는 크로스 컴파일이 아니라, 이 강제
설정을 뚫고 크로스 컴파일 환경에서 CPython 확장을 빌드·링크하는 것이며, 그것은 선례가 없습니다.**
스파이크는 미지수를 해소한 것이 아니라 **미지수의 위치를 옮겼습니다.**

**2. 부패는 계속 쌓인다.** 다섯 지점 각각이 얕았다는 것보다 중요한 것은 **그것들이 왜 거기
있었는가** 입니다. CI 참조가 0건이고, in-tree 테스트가 삭제된 스크립트를 부르고 있으며,
호출자를 남긴 채 피호출자만 지워졌습니다. 막힌 지점 5 가 이 성질을 가장 잘 보여줍니다 —
**최근 추가된 CUDA 전용 op 하나가 선택 빌드를 깼고, 아무도 몰랐습니다.** 이번에 우회한 다섯은
"남은 부채" 가 아니라 **작년 7월 이후 1년간 쌓인 표본**이고, 리베이스할 때마다 새로 생깁니다.
B 를 고른다는 것은 이 부패를 영구히 우리가 떠맡는다는 뜻입니다. §5 의 "상류 지원 0" 이
정량적으로 확인된 지금, 그 비용은 일회성이 아니라 **경상비**입니다.

**3. iOS 는 Android 만큼도 남아 있지 않다 (§5).** `cmake/iOS.cmake` 가 main 에서 사라졌습니다.
Android 가 뚫린 것은 삭제되지 않은 배선이 남아 있었기 때문인데, iOS 에는 그것이 없습니다.
torchnative 는 iOS 가 타깃이므로 **B 의 비용을 Android 결과로 추정하면 과소평가하게 됩니다.**

### 권고

**A(candle + PyO3) 를 기본 경로로 두는 것을 지지합니다.** 다만 근거가 §5 가 예상한 것과 다릅니다 —
"B 의 크로스 컴파일이 안 뚫려서" 가 아니라, **뚫어봤더니 그것이 B 의 실제 관문이 아니었기
때문**입니다. B 의 관문은 파이썬 바인딩 층이고, 그 층은 모바일 빌드 정의가 명시적으로 배제합니다.

동시에 §6 의 "타임박스를 건 빌드 스파이크 한 번으로 판정하고, 뚫리지 않으면 A 로 간다" 는
**판정 기준을 수정해야 합니다.** 크로스 컴파일 성공/실패는 B 를 판정하지 못합니다.
B 를 정말로 판정하려면 질문이 이것이어야 합니다 — **"모바일 크로스 컴파일 환경에서
`torch/csrc` 파이썬 바인딩이 빌드되고, 임베디드 CPython 이 그것을 `import` 할 수 있는가."**

그리고 그 질문은 §6 이 이미 옳게 정리한 순서를 바꾸지 않습니다. 두 경로가 공유하는 일 —
torch 파이썬 트리 벤더링, `import transformers` 성립, `_C` 경계 확정 — 이 여전히 먼저이고,
**그 일이 끝나면 위 질문에 답하는 비용도 훨씬 싸집니다** (벤더링한 트리에 우리 `_C` 를 끼우는
배선이 이미 서 있을 것이므로). 결정을 지금 닫을 필요는 없되, **B 를 "빌드 한 번이면 판정되는
것" 으로 취급하는 것은 그만두어야 합니다.**

---

## 부록 — 재현 방법

```bash
WS=/Volumes/macMini/caches/pytorch-spike
mkdir -p $WS && cd $WS
git clone --depth 1 https://github.com/pytorch/pytorch.git pytorch
cd pytorch && git submodule update --init --recursive --depth 1 --jobs 8

python3 -m venv $WS/venv
$WS/venv/bin/pip install -U pip setuptools wheel pyyaml typing_extensions

# 우회 1: Android SDK 번들 cmake(3.22.1)는 3.27 미만이라 못 쓴다. ninja 만 거기서 가져온다.
export PATH=$PATH:$HOME/Library/Android/sdk/cmake/3.22.1/bin   # ninja 용, 뒤에 붙일 것
export ANDROID_NDK=$HOME/Library/Android/sdk/ndk/27.1.12297006

# 우회 2·3: eigen 서브모듈 부재 / NDK Vulkan 래퍼 부재
#   -> -DUSE_BLAS=OFF -DUSE_VULKAN=OFF

# 우회 4: c10/CMakeLists.txt:124 의 WHOLE_ARCHIVE 를 일반 링크로
sed -i '' 's|target_link_libraries(c10 PRIVATE "\$<LINK_LIBRARY:WHOLE_ARCHIVE,cpuinfo>")|target_link_libraries(c10 PRIVATE "cpuinfo")|' c10/CMakeLists.txt

sed -i '' -e "s/__cplusplus >= 201703L/0/" third_party/pocketfft/pocketfft_hdronly.h

BUILD_ROOT=$WS/build_android_arm64
mkdir -p $BUILD_ROOT && cd $BUILD_ROOT
cmake $WS/pytorch -GNinja \
  -DCMAKE_INSTALL_PREFIX=$BUILD_ROOT/install -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE=$WS/venv/bin/python -DBUILD_CUSTOM_PROTOBUF=OFF \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DBUILD_TEST=OFF -DBUILD_BINARY=OFF \
  -DBUILD_LITE_INTERPRETER=ON -DTRACING_BASED=OFF -DUSE_LIGHTWEIGHT_DISPATCH=OFF \
  -DBUILD_MOBILE_BENCHMARK=0 -DBUILD_MOBILE_TEST=0 \
  -DBUILD_PYTHON=OFF -DBUILD_SHARED_LIBS=OFF -DANDROID_TOOLCHAIN=clang \
  -DUSE_CUDA=OFF -DUSE_ITT=OFF -DUSE_GFLAGS=OFF -DUSE_MPI=OFF -DUSE_OPENMP=OFF \
  -DUSE_VULKAN=OFF -DUSE_BLAS=OFF \
  -DANDROID_NDK=$ANDROID_NDK -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=21 \
  -DANDROID_CPP_FEATURES="rtti exceptions" \
  > /tmp/cfg.log 2>&1 ; echo "CFG_EXIT=$?"

ninja -j8 > /tmp/build.log 2>&1 ; echo "BUILD_EXIT=$?"
```

선택 빌드는 위에 `-DSELECTED_OP_LIST=<yaml> -DCMAKE_CXX_FLAGS="-DSTRIP_ERROR_MESSAGES"` 를
추가하고, 우회 5(`_scaled_grouped_mm_v2_cpu_dispatch.h` 스텁)를 넣습니다.

> 성공/실패는 반드시 **종료 코드**로 판정합니다. 빌드 로그에는 소스 줄이 그대로 찍히므로
> 출력 grep 으로 판정하면 안 됩니다.
