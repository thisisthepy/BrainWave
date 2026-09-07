# iOS — 시뮬레이터는 계산까지, 기기는 링크까지

**결론: 시뮬레이터 휠은 로드·임포트·계산까지 전부 확인됐다.** `docs/platform/WHEEL.md` §7 이
"아티팩트만 확인" 으로 남겨 두었던 칸이 시뮬레이터 쪽에 한해 채워졌다.

**기기 휠은 실행이 확인되지 않았다.** 이유는 §7 에 있다 — 편의상의 유보가 아니라 별개의
아티팩트이기 때문이다. **다만 "실기가 없으니 전부 미검증" 은 사실이 아니었다.** 두 슬라이스의
실질적 차이는 **CPython 심볼을 어떻게 푸느냐** 하나이고, 그 하나는 실기 없이 답할 수 있다 (§11).

## 0. 판정표

| | 빌드됨 | 심볼 해결됨 | 설치됨 | 임포트됨 | 계산됨 |
|---|---|---|---|---|---|
| `ios_14_0_arm64_iphonesimulator` | ✅ | ✅ (플랫 조회로 — §11.1) | ✅ 시뮬레이터 CPython 의 site-packages 에 언팩 | ✅ | ✅ `aten.mm`, `x+x`, `nn.Linear` |
| `ios_12_0_arm64_iphoneos` (실기) | ✅ | ✅ **222/222**, 각 심볼이 *묶인 그 라이브러리*에서 실제로 export 됨 (§11) | ❌ 기기 없음 | ❌ | ❌ |

판정 하네스는 둘이고, **말하는 것이 다르다**:

| | |
|---|---|
| `tools/wheel/verify_ios_sim.py` | 시뮬레이터 안에서 임포트·계산까지. `verify_android.py` 와 **판정 문장이 같다** — *`torch.__file__` 이 설치 위치 안을 가리켜야 한다* |
| `tools/wheel/verify_ios_device.py` | 기기 휠. **아무것도 실행하지 않는다.** 링크가 풀리는지만 본다 (§11). 이 도구가 말하는 것은 **"심볼이 다 풀린다"** 이지 **"기기에서 돈다"** 가 아니다 |

```
PASS -- torchnative-0.0.1a0-cp313-abi3-ios_14_0_arm64_iphonesimulator.whl unpacks into
        an iOS CPython's site-packages and its torch computes in the simulator
```

## 1. 시뮬레이터가 답할 수 없는 것 — 먼저 적는다

**시뮬레이터는 이 M1 맥 위에서 도는 일반 프로세스다.** `simctl spawn` 이 해 주는 것은
`DYLD_ROOT_PATH` 로 시뮬레이터 런타임 루트를 씌우는 것뿐이고, **CPU 는 호스트 M1 그대로다.**

| 시뮬레이터가 답하는 것 | 시뮬레이터가 **답하지 않는** 것 |
|---|---|
| 확장이 로드되는가 (`dlopen`, 심볼 해결) | 아이폰 처리량 · 지연 |
| `import torch` 가 끝까지 가는가 | 실기 AMX / NEON 경로의 실제 성능 |
| 계산 결과가 호스트와 일치하는가 | 발열 · 스로틀링 · 전력 |
| 임베드 경로 (`PYTHONHOME`, 프레임워크 링크) | 실기 코드 서명 · 프로비저닝 |

**그래서 이 문서에는 성능 수치가 하나도 없다.** 재면 다음 사람이 그것을 아이폰 숫자로 읽는다.
이 저장소는 이미 같은 함정에 빠진 적이 있다 — 안드로이드 에뮬레이터가 광고한 `shaderFloat16`
비트가 사실 **호스트 M1 의 것이 그대로 전달된 것**이었다 (`docs/devices/DEVICE.md` §10 정정).

### 그 노출이 여기서도 그대로 보인다

시뮬레이터 안에서 찍은 `platform.uname()` 이다:

```
system  = 'iOS'                     ← 시뮬레이터의 것
release = '18.0'                    ← 시뮬레이터의 것
machine = 'arm64'
version = 'Darwin Kernel Version 25.5.0: ... xnu-12377.121.6~2/RELEASE_ARM64_T8103'
                                    ↑ 호스트 맥의 커널. T8103 은 M1 이다
```

`system`/`release` 는 iOS 인데 **커널은 호스트의 것이 그대로 보인다.** 시뮬레이터는 iOS 커널을
흉내 내지 않고 호스트 커널 위에 iOS 사용자 공간만 씌운다. 성능이나 CPU 특성을 여기서 읽으면
안 되는 이유가 이 한 줄에 다 들어 있다. 하네스는 이 값을 매번 출력하고 그 옆에 경고를 찍는다.

## 2. 사다리 — 작은 것부터, 각 단계가 실제로 관측된 것

| 단계 | 결과 |
|---|---|
| 런처가 `print("hello")` | ✅ |
| `import sys` | ✅ `sys.platform == 'ios'`, CPython `3.13.0+` |
| stdlib (`json` `zipfile` `ctypes` `sysconfig`) | ✅ |
| `import _multiprocessing` | ❌ 배포본에 없음 — 스텁 필요 (§5) |
| 휠을 site-packages 에 언팩 | ✅ 호스트 네이티브 침입 0건 |
| `import torch` | ✅ 단, 런처가 UIKit 를 링크해야 한다 (§4) |
| `aten.mm` · `x+x` · `nn.Linear` | ✅ 호스트와 동일 (§3) |

## 3. 계산 결과 — 호스트와 대조

| | 시뮬레이터 (iOS 18.0) | 호스트 (macOS arm64) |
|---|---|---|
| `torch.__version__` | 2.13.0 | 2.13.0 |
| `len(dir(torch._C))` | **1260** | **1260** |
| `len(dir(torch.ops.aten))` | **896** | **896** |
| `aten.mm.default(ones(3,4), ones(4,2))` | `[[4,4],[4,4],[4,4]]` | `[[4,4],[4,4],[4,4]]` |
| `x + x` | `[2,2,2,2]` | `[2,2,2,2]` |
| `nn.Linear(4,3)(ones(2,4))` | `[2,3] float32` | `[2,3] float32` |
| `importlib.metadata.version("torch")` | 2.13.0 | 2.13.0 |

**전부 일치한다.** 개수 두 개(1260·896)까지 같다는 것은 크로스 빌드가 op 등록을 하나도 흘리지
않았다는 뜻이다. (이 표는 처음 1251·896 으로 기록됐다. 2026-08-29 재측정에서 양쪽 다 1260 이다 —
그 사이 `rust/torch_c` 에 착지한 것들이 늘린 수이고, **호스트와 시뮬레이터가 여전히 같다**는 것이
이 표가 말하는 바다.)

`sys.path` 에 저장소로 이어지는 항목이 **하나도 없다** — 하네스가 이것을 단언으로 검사한다:

```
<prefix>/lib/python313.zip
<prefix>/lib/python3.13
<prefix>/lib/python3.13/lib-dynload
<prefix>/lib/python3.13/site-packages     ← torch.__file__ 이 여기 안
```

`PYTHONPATH` 를 쓰지 않는다. 안드로이드에서 얻은 교훈 그대로 — `PYTHONPATH` 엔트리는 **망가진
설치를 가려 주는 바로 그것**이다 (`WHEEL.md` §7.3).

## 4. 무엇이 막고 있었나 — `platform.system()` 이 UIKit 를 부른다

`import torch` 는 처음에 **`torch/__init__.py:370` 에서** 죽었다.

```
torch/__init__.py:444  _load_global_deps()
torch/__init__.py:370    if platform.system() == "Windows":
platform.py:1072         return uname().system
platform.py:1056         system, release, _, _ = ios_ver()
platform.py:517          result = _ios_support.get_platform_ios()
_ios_support.py:67       system = objc.objc_msgSend(device_systemName, SEL_UTF8String).decode()
AttributeError: 'NoneType' object has no attribute 'decode'
```

**휠의 결함이 아니다.** iOS 의 CPython 은 `platform.system()` 을 Objective-C 런타임에 물어본다
(`_ios_support.py:40` — `objc_getClass(b"UIDevice")`). `simctl spawn` 으로 띄운 **맨 프로세스에는
UIKit 가 로드되어 있지 않아** 그 클래스 조회가 nil 을 돌려주고, 이어지는 `objc_msgSend` 가 전부
nil 을 타고 내려가 마지막 `.decode()` 에서 터진다.

**진짜 앱에는 UIKit 가 항상 로드되어 있다.** 즉 이것은 앱 경로에는 없고 **spawn 하네스에만 있는
결손**이다. 그래서 프로브에 우회를 넣지 않고 **런처를 `-framework UIKit` 로 링크했다** — 하네스를
앱에 가깝게 만든 것이지 측정을 피해 간 것이 아니다.

```
UIKit 링크 후:  platform.system() -> 'iOS'
```

이 결론은 추측이 아니라 음성 대조로 확인했다 (§6).

## 5. `DEVICE.md` §4 의 환경 변수를 하나씩 빼서 쟀다

안드로이드에 필수라고 기록된 것들이 iOS 에서도 필요한지 **하나씩 제거하며** 측정했다.
안드로이드에서 그중 하나(`TORCH_USE_RTLD_GLOBAL`)가 실은 불필요했던 것이 최근 밝혀졌기 때문에
(`WHEEL.md` §7.3.1), 물려받지 않고 다시 쟀다.

| 뺀 것 | 결과 | 판정 |
|---|---|---|
| `PYTHONHOME` | `ModuleNotFoundError: No module named 'encodings'` (인터프리터 초기화 단계에서 죽음) | **필수** |
| `_multiprocessing`/`_posixshmem` 스텁 | `ModuleNotFoundError: No module named '_multiprocessing'` (`torch/multiprocessing/__init__.py:110`) | **필수** |
| `TORCH_USE_RTLD_GLOBAL` | 없어도 성공. 붙여도 같은 값 | **불필요** |
| UIKit 링크 | `AttributeError: 'NoneType' object has no attribute 'decode'` | **필수** (하네스에만) |

**`TORCH_USE_RTLD_GLOBAL` 은 안드로이드와 같은 이유로 불필요하다** — 휠이
`torch/lib/libtorch_global_deps.so` 를 싣기 때문이다. 그리고 그것이 장식이 아니라 실제로
로드되고 있다는 것을 음성 대조로 확인했다 (§6).

**`_multiprocessing` 은 휠의 결함이 아니라 iOS CPython 배포본의 성질이다** — 안드로이드 배포본과
똑같이 그것도 `_posixshmem` 도 빌드하지 않는다. 하네스는 스텁 없이 한 번 돌려서 그 실패를
**출력에 남긴다** — 필요한 것을 배경에 숨기지 않기 위해서다.

## 6. 실패할 수 있는 검증인지 확인했다

"실패할 수 없는 검증은 검증이 아니다." 통과를 만들어 낸 것 세 개를 각각 없애 보고, **올바른
이유로** 깨지는지 확인했다.

| 망가뜨린 것 | 관측된 실패 | 무엇을 증명하는가 |
|---|---|---|
| `libtorch_global_deps.so` 를 옆으로 치움 | `OSError: dlopen(.../libtorch_global_deps.so ...): (no such file)` | 그 파일이 **실제로 로드되고 있다.** `TORCH_USE_RTLD_GLOBAL` 이 필요 없어진 진짜 이유 |
| 런처를 UIKit 없이 재링크 | `AttributeError: 'NoneType' ... decode` 재현 | UIKit 링크가 §4 의 실제 원인 수정임 |
| `PYTHONHOME` 제거 | `No module named 'encodings'` | 스테이징한 prefix 를 실제로 쓰고 있다 |

하네스 자체도 잘못된 아티팩트를 거절한다:

```
$ verify_ios_sim.py dist/...-macosx_11_0_arm64.whl
torchnative-...-macosx_11_0_arm64.whl is not an iOS-simulator wheel.
The device wheel is a different artefact and this harness cannot run it.
```

## 7. 시뮬레이터를 검증해도 기기 휠은 검증되지 않는다

**둘은 별개의 아티팩트다.** Mach-O `LC_BUILD_VERSION` 의 platform 필드가 다르다:

| | platform | 값 |
|---|---|---|
| `ios_14_0_arm64_iphonesimulator` | `IOSSIMULATOR` | **7** |
| `ios_12_0_arm64_iphoneos` | `IOS` | **2** |

**크기도 아키텍처도 심볼도 같아서 그 필드 말고는 구별되지 않는다** (`WHEEL.md` §7.4).
게다가 링크 방식 자체가 다르다:

- **기기** 산출물은 `@rpath/Python.framework/Python` 을 `LC_LOAD_DYLIB` 에 박는다. 실기에는
  폴백할 libpython 이 없으므로 이것이 없으면 로드 자체가 불가능하다.
- **시뮬레이터** 산출물에는 그 의존이 **없다.** `-undefined dynamic_lookup` 으로 해소되고,
  그것이 정상이다 (`WHEEL.md` §7.1).

즉 시뮬레이터에서 심볼이 해결됐다는 것은 **기기에서 프레임워크 링크가 맞다는 근거가 되지
않는다.** 서로 다른 메커니즘이다. 기기 쪽을 **실행**해 보려면 실기 + 프로비저닝 프로파일이
필요하고 이 기계에 둘 다 없다.

**그런데 그 "서로 다른 메커니즘" 이야말로 실기 없이 검사할 수 있는 것이다.** 위 문단은 2026-08-28
에 쓰였고, 그때는 "시뮬레이터 결과가 기기 칸을 채워 주지 않는다" 에서 멈췄다. 멈출 이유가
없었다 — 기기 산출물이 프레임워크에 대해 하는 주장은 **파일 안에 심볼 이름으로 적혀 있고**,
그 프레임워크는 이 디스크에 있다. 대조하면 된다. §11 이 그 대조다.

또 하나 여기서 명확히 해 둘 것: 위 표는 두 슬라이스가 "크기도 아키텍처도 심볼도 같다" 고
적었는데, **심볼은 같지 않다.** 정의된 심볼 집합은 같지만 **미해결 심볼의 성질이 다르다** —
기기 쪽 118 개는 `Python` 에 이름으로 묶여 있고, 시뮬레이터 쪽 같은 118 개는 아무 데도 묶여
있지 않다(`dynamically looked up`). 그 차이가 §11 의 검사가 성립하는 이유이고, 동시에 시뮬레이터
산출물을 기기 휠에 넣었을 때 그 검사가 잡아내는 근거다.

## 8. 재현

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-ios
BPY=/Volumes/macMini/caches/wheel-build-venv/bin/python
PY=/Volumes/macMini/caches/spike-venv/bin/python

bash vendor/vendor_torch.sh
bash vendor/install_shim.sh

# 1) 시뮬레이터용 _C 를 크로스 빌드한다.
#    시뮬레이터도 PYO3_CONFIG_FILE 이 필요하다 (WHEEL.md §7.1). 기기와 달리
#    TORCHNATIVE_PYTHON_FRAMEWORK_DIR 은 필요 없다 — 프레임워크를 링크하지 않는다.
cat > /tmp/pyo3-ios.cfg <<'EOF'
implementation=CPython
version=3.13
shared=true
lib_name=Python
pointer_width=64
suppress_build_script_link_lines=true
EOF

( cd rust/torch_c && PYO3_CONFIG_FILE=/tmp/pyo3-ios.cfg \
  PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 \
  PYO3_CROSS_LIB_DIR=/Volumes/macMini/caches/target-python/arm64-iphonesimulator/lib \
  cargo build --release --target aarch64-apple-ios-sim )

# 1b) 기기용도 같이 만든다 — §11 의 대조에는 두 휠이 다 필요하다.
( cd rust/torch_c && PYO3_CONFIG_FILE=/tmp/pyo3-ios.cfg \
  PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 \
  PYO3_CROSS_LIB_DIR=/Volumes/macMini/caches/target-python/arm64-iphoneos/lib \
  TORCHNATIVE_PYTHON_FRAMEWORK_DIR=/Volumes/macMini/caches/target-python/arm64-iphoneos \
  cargo build --release --target aarch64-apple-ios )

# 2) 휠을 만든다.
#    build.py 는 여기서 크로스 산출물이 **현재 소스로 빌드된 것인지** 확인하고,
#    아니면 이름을 대고 거절한다 (WHEEL.md §11). 위 cargo 단계를 건너뛰면 이 줄이 멈춘다.
$BPY tools/wheel/build.py --target ios-arm64-sim
$BPY tools/wheel/build.py --target ios-arm64

# 3) 판정한다 — 시뮬레이터를 띄우고, 임포트시키고, 계산시킨다
$PY tools/wheel/verify_ios_sim.py dist/torchnative-*iphonesimulator*.whl

# 4) 기기 휠은 실행할 수 없다. 링크가 풀리는지만 본다 (§11).
#    시뮬레이터 휠이 dist/ 에 같이 있어야 §11.3 의 대조가 성립한다.
$PY tools/wheel/verify_ios_device.py dist/torchnative-*iphoneos*.whl
$PY tools/wheel/verify_ios_device.py --self-test dist/torchnative-*iphoneos*.whl
```

하네스는 시뮬레이터를 **자기가 부팅했을 때만** 종료시킨다. 이미 떠 있던 것은 건드리지 않는다.
`--udid` 또는 `IOS_SIMULATOR_UDID` 로 기기를 고정할 수 있고, 없으면 사용 가능한 첫 iPhone 을
고르고 **어느 것을 골랐는지 출력한다.**

측정에 쓴 것: iPhone 16 Pro / iOS 18.0 (`8F9F769D-...`), Xcode 26.6, macOS 26.5.1.

## 9. 하네스가 안드로이드와 다른 지점

| | Android (`verify_android.py`) | iOS 시뮬레이터 (`verify_ios_sim.py`) |
|---|---|---|
| 배포본에 인터프리터가 있는가 | 있음 (`bin/python3.13`) | **없음** — `Python.framework/Python` 은 `MH_DYLIB` 뿐 |
| 실행 파일 | 배포본의 것을 push | **직접 컴파일한다** — `Py_BytesMain` 3줄 (아래) |
| 기기로 옮기기 | `adb push` | **불필요** — 시뮬레이터는 호스트 파일시스템을 그대로 본다 |
| 실행 진입점 | `adb shell` | `xcrun simctl spawn <UDID>` (`DYLD_ROOT_PATH` 를 대신 채워 줌) |
| 환경 변수 전달 | 그대로 | `SIMCTL_CHILD_` 접두사 필요 |
| 추가로 필요한 것 | — | **UIKit 링크** (§4) |
| 스텁 | `_multiprocessing`, `_posixshmem` | **같음** |

### 인터프리터를 직접 만든 것에 대해

iOS 배포본에는 실행 파일이 없다. 그래서 하네스가 이것을 컴파일한다:

```c
#include <Python.h>
int main(int argc, char **argv) { return Py_BytesMain(argc, argv); }
```

`Py_BytesMain` 은 CPython 의 **CLI 진입점 전체**이고 프레임워크가 그것을 export 한다
(`nm -gU` 로 확인). 그래서 결과물은 시뮬레이터용 진짜 `python3.13` 이고, 프로브를
`verify_android.py` 와 **똑같은 `-s -P <script>` 인자로** 돌릴 수 있다.

이것이 `PyRun_SimpleString` 보다 나은 이유가 있다. `DEVICE_LOAD_IOS.md` 는 임베딩 API 경로에서
`sys.path` 에 cwd 가 안 붙어 `PYTHONPATH` 를 손으로 채워야 했다고 적었는데, **그 차이가 바로 두
플랫폼의 측정을 비교 불가능하게 만드는 지점**이다. `Py_BytesMain` 은 `-s -P` 를 진짜로 해석하므로
그 차이가 사라진다.

빌드된 런처가 시뮬레이터 플랫폼인지 하네스가 매번 확인한다:

```
$ xcrun vtool -show-build-version <launcher>
 platform IOSSIMULATOR
    minos 14.0
```

### 공유 캐시를 오염시키지 않는다

`/Volumes/macMini/caches/target-python/arm64-iphonesimulator` 는 다른 작업과 공유된다.
하네스는 그것을 `/Volumes/macMini/caches/ios-wheel-check/prefix` 로 **복사한 뒤** 그 사본의
site-packages 에 휠을 푼다 (205 MB). 공유본의 site-packages 는 그대로 남는다.

## 10. 아직 남은 것

| 빈 칸 | 필요한 것 |
|---|---|
| iOS **실기** 로드·임포트·계산 | 기기 + 프로비저닝 프로파일. 이 기계에 없다. §7 때문에 시뮬레이터 결과로 대신할 수 없다. **링크 해결까지는 §11 이 채웠다** — 남은 것은 dyld 가 실행 시점에 `@rpath/Python.framework/Python` 을 *찾는가*, 코드 서명, 그리고 그 위의 전부다 |
| **실제 앱 번들** 경로 | `Python.framework` 를 `Frameworks/` 에 `Embed & Sign` 하고 앱 프로세스가 스스로 `Py_Initialize` 를 부르는 형태. rpath 가 `@rpath` 에서 `@executable_path/Frameworks` 로 바뀐다. §11 이 답하지 **못하는** 것이 정확히 이것이다 |
| 아이폰 성능 | 실기에서만. 시뮬레이터에서 재면 M1 숫자다 (§1) |
| `_multiprocessing` | iOS CPython 배포본 재빌드, 또는 상류 `torch/multiprocessing` 지연 임포트. 안드로이드와 같은 항목이고 휠 작업의 몫이 아니다 |
| PEP 730 휠의 `.so` → `.framework` 변환 | App Store 규칙상 바이너리 모듈은 `.framework` 여야 한다. 그것은 앱 패키징 도구의 일이고(briefcase 가 그렇게 한다), 이 저장소 안에서 일어나는지는 아직 아무도 정하지 않았다 (`WHEEL.md` §7.4) |

---

## 11. 기기 휠 — 실기 없이 어디까지 확인되는가 (2026-08-29)

`tools/wheel/verify_ios_device.py`.

```sh
$PY tools/wheel/verify_ios_device.py dist/torchnative-*iphoneos*.whl
$PY tools/wheel/verify_ios_device.py --self-test dist/torchnative-*iphoneos*.whl
```

### 11.0 먼저, 이 도구가 하지 않는 것

**아무것도 실행하지 않는다.** 실행할 수가 없고, 그것을 주장이 아니라 **dyld 의 말**로 확인했다.
두 기계 모두에서 거절된다:

| 어디에 놓았나 | dyld |
|---|---|
| 시뮬레이터의 site-packages 에 넣고 `import torch` | `ImportError: dlopen(...): incompatible platform (have 'iOS', need 'iOS-sim')` |
| macOS 에서 `ctypes.CDLL` | `OSError: dlopen(...): incompatible platform (have 'iOS', need 'macOS')` |

첫 줄은 시뮬레이터(iPhone 16 Pro / iOS 18.0)를 띄우고 `verify_ios_sim.py` 가 스테이징한
site-packages 의 `_C.abi3.so` 를 **기기 슬라이스로 바꿔 넣고** 실제로 재현한 것이다. 둘째 줄은
`verify_ios_device.py` 가 **매 실행마다 찍는다** — "심볼이 풀린다" 를 "돌아간다" 로 읽지 않게
하려고 출력 맨 앞에 둔다.

```
+ dyld will not load this here, which is why nothing below runs it
  torch/_C.abi3.so: incompatible platform (have 'iOS', need 'macOS')
```

### 11.1 그러면 무엇이 남아 있나 — 링크는 실기 없이 답이 나온다

두 슬라이스가 실제로 다른 것은 **CPython 심볼을 푸는 방식** 하나다.

| | 시뮬레이터 | 기기 |
|---|---|---|
| CPython 심볼 118 개 | `dynamically looked up` — **아무 라이브러리에도 묶여 있지 않다** | 전부 `Python` 에 **이름으로 묶여 있다** (2단계 네임스페이스) |
| `LC_LOAD_DYLIB` 의 Python | **없음.** `-undefined dynamic_lookup` 으로 해소 (§7, `WHEEL.md` §7.1) | `@rpath/Python.framework/Python` |
| 실기에서 폴백 | — | **없다.** 그래서 이 링크가 유일한 경로다 |

기기 산출물이 프레임워크에 대해 하는 주장은 **파일 안에 심볼 이름으로 적혀 있다.** 그리고 그
프레임워크는 이 디스크에 있다(`target-python/arm64-iphoneos/Python.framework/Python`).
그러니 대조하면 된다 — 실기가 필요한 것은 *실행*이지 *대조*가 아니다.

### 11.2 측정

```
torchnative-0.0.1a0-cp313-abi3-ios_12_0_arm64_iphoneos.whl
  torch/_C.abi3.so   222 undefined
    Accelerate     16  <- Accelerate.tbd  (iPhoneOS26.5.sdk)
    Python        118  <- Python  (the device CPython distribution)
    libSystem      88  <- libSystem.B.tbd  (iPhoneOS26.5.sdk)
    unresolved      0
  torch/lib/libtorch_global_deps.so   0 undefined
    unresolved      0
```

**222 개 전부가, 자기가 묶인 그 라이브러리에서 풀린다.**

읽는 법:

| | |
|---|---|
| **묶인 곳별로** 본다 | `nm -m` 은 2단계 네임스페이스 이미지의 각 미해결 심볼에 대해 **링커가 적어 둔 소속 라이브러리**를 알려 준다. 합집합으로 검사하면 `_Py*` 심볼이 우연히 libSystem 에 있어도 통과하는데, dyld 는 거기를 보지 않으므로 그 통과는 거짓이다. 그래서 라이브러리별로 나눠 묻는다 |
| `Python` 118 | 기기용 `Python.framework/Python` 의 `nm -gU` 출력(1670 개)과 대조 |
| `libSystem` 88 · `Accelerate` 16 | SDK 에는 arm64 dylib 이 없고 `.tbd` 텍스트 스텁만 있다. `libSystem.B.tbd` 는 **자기가 re-export 하는 라이브러리들의 스텁을 같은 파일 안에 다 담은 다중 문서**라서, 그 파일 하나를 읽는 것이 재export 폐포를 읽는 것이다 |
| `libiconv` | `LC_LOAD_DYLIB` 에는 있지만 미해결 심볼이 **0 개**라 표에 나오지 않는다 |
| `libtorch_global_deps.so` | 심볼이 하나도 없는 것이 정상이다 — 설계상 빈 라이브러리다 (`WHEEL.md` §3.2) |

### 11.3 나머지 아카이브는 시뮬레이터 휠과 **바이트 단위로 같다**

```
  2,565 shared members, 0 differing (compared byte-for-byte;
  torch/_C.abi3.so, torch/lib/libtorch_global_deps.so and the dist-info are
  excluded, being the platform-shaped parts)
```

이것이 왜 판정에 들어가는가: `verify_ios_sim.py` 가 시뮬레이터 안에서 임포트한 벤더링 파이썬
트리가 **기기 휠 안의 그것과 같은 파일**이라는 뜻이다. 즉 기기 휠의 *파이썬 절반*은 시뮬레이터에서
이미 실행됐다. 별개의 아티팩트인 것은 Mach-O 절반뿐이고, `WHEEL.md` §7.4 가 말로 적어 둔 그
구분을 여기서는 측정으로 적는다.

### 11.4 실패할 수 있는지 확인했다

"실패할 수 없는 검증은 검증이 아니다." 판정마다 하나씩 망가뜨려서, **올바른 종류의 답**으로
깨지는지 봤다. `--self-test` 가 매번 돈다.

```
SELF-TEST against torchnative-0.0.1a0-cp313-abi3-ios_12_0_arm64_iphoneos.whl
  caught    a Python.framework that is not CPython
  caught    no device Python.framework on disk
  caught    an SDK with no .tbd stubs
  caught    the simulator extension passed off as the device one
  caught    a vendored file differing between the two wheels

SELF-TEST: PASS -- 5/5 fault modes rejected, and each
  as the right kind of answer (a finding about the wheel, or the check unable to look)
```

| 망가뜨린 것 | 나와야 하는 답 | 무엇을 증명하는가 |
|---|---|---|
| `Python.framework` 자리에 CPython 이 아닌 진짜 Mach-O(빈 global-deps)를 놓음 | **FAIL** — 118 개 미해결 | 프레임워크를 **실제로 조회하고 있다.** 안 그러면 무엇을 놓든 통과한다 |
| `Python.framework` 를 아예 치움 | **CANNOT JUDGE** | 검사가 못 보는 것과 휠의 결함을 **섞지 않는다** |
| `.tbd` 가 없는 빈 SDK 를 가리킴 | **CANNOT JUDGE** | 같은 구분을 다른 제공자에서 |
| 기기 슬라이스 자리에 **시뮬레이터 슬라이스**를 넣음 | **FAIL** | `WHEEL.md` §7.4 가 "이 필드 말고는 구별되지 않는다" 고 한 그 치환을 잡는다 |
| 벤더링 파일 하나를 두 휠 사이에서 다르게 만듦 | **FAIL** | §11.3 이 시뮬레이터 결과를 실어 나르는 근거이므로, 근거가 없을 때 없다고 말해야 한다 |

**두 종류의 답을 섞지 않는 것이 이 도구의 골격이다.** `FAIL:` 은 휠에 대한 발견이고
`CANNOT JUDGE:` 는 검사가 못 본 것이다. 둘 다 종료 코드 1 이지만 문장이 다르고, self-test 가
그 문장까지 맞춰 본다. `rust/torch_c/pytests/run.sh` 주석에 있는 실패 — SIGKILL 당한 `cmp` 의
종료 코드를 "다름" 으로 읽어 멀쩡한 아티팩트를 낡았다고 보고한 것 — 의 같은 뿌리다.

### 11.5 그래서 기기 휠에 대해 지금 말할 수 있는 것

도구가 매 실행 끝에 이 사다리를 찍는다.

```
  the device wheel, rung by rung
    [yes] built                              device Mach-O, tag an installer matches -- verify_cross.py
    [yes] symbols resolved                   every undefined symbol exported by the library it is bound to
    [yes] same tree as the simulator wheel   everything but the extension is byte-identical
    [NO ] installed                          needs a device: no iOS filesystem here
    [NO ] imported                           needs a device: dyld refuses this slice on macOS and in the simulator
    [NO ] computed                           needs a device
```

**남은 것을 정확히 적는다.** 심볼이 존재하는 것은 필요조건이지 충분조건이 아니다.

| 남은 것 | 왜 실기여야 하나 |
|---|---|
| **런타임 로드** | dyld 가 실행 시점에 `@rpath/Python.framework/Python` 을 **찾아야** 한다. 이 도구는 캐시 디렉터리의 프레임워크 파일을 읽었을 뿐이고, 앱에서는 `@rpath` 가 `@executable_path/Frameworks` 로 해석된다 (§10) |
| **코드 서명** | 앱 번들 안의 모든 Mach-O 는 기기가 신뢰하는 프로파일로 서명돼야 한다. 여기서는 아무것도 서명하지 않는다 |
| **`import torch` 완주** | 시뮬레이터는 `_multiprocessing` 스텁과 UIKit 가 로드된 프로세스가 필요했다 (§4 · §5). 실제 앱이 그 둘을 만족하는지는 측정되지 않았다 |
| **계산 · 성능** | 실기에서만. §1 |
