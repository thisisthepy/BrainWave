# PyO3 CPython 3.13 크로스 빌드 조사 보고서

> 초안은 `agy` 가 작성했고, 경로와 디스크 결론은 조율 세션에서 아카이브를 직접 열어 검증·수정했습니다.
> 수정된 항목: §1 의 Android · iOS 경로(둘 다 틀렸음), §5 의 결론. §1 의 abi3 항목은 **미결**입니다.

## 0. 툴체인 현황 (실측)

**디스크 블로커는 해소됐습니다.** `PythonMultiplatform` 세션이 worktree 86 개의 `build/` 만
삭제해 159 GB 를 회수했습니다 — 소스 · git 상태 · 브랜치는 전부 남았습니다. 표본 검증에서
worktree 가 그대로 있고 HEAD 도 일치하며 미커밋 변경이 0 개임을 확인했습니다.

| | 상태 | 크기 | 위치 |
|---|---|---|---|
| 외장 `/Volumes/macMini` | 여유 **164Gi** (이전 4.6Gi) | | |
| 내부 | 여유 **48Gi** | | |
| Rust 툴체인 (minimal, 1.98.0) | **설치됨** | 863 MB | 내부 (`~/.rustup` 852M + `~/.cargo` 11M) |
| 타깃 | **설치됨** | | `aarch64-linux-android` · `aarch64-apple-ios` · `aarch64-apple-ios-sim` · `aarch64-apple-darwin` |
| Android NDK 27.1.12297006 | **이미 있었음** | 2.4 GB | 내부 (`~/Library/Android/sdk/ndk/`) |
| Xcode 26.6 (Build 17F113) | **이미 있었음** | | |
| `cargo-ndk` | 설치 중 | | |

**툴체인은 내부, 산출물은 외장.** `PythonMultiplatform` 세션의 판단을 따랐습니다 — 외장이
언마운트되면 `~/.cargo/bin` 이 PATH 에 물려 있어 빌드가 아니라 셸이 깨지고, cargo 레지스트리와
`target/` 은 파일 수만 개짜리 워크로드라 외장에서 눈에 띄게 느립니다. 큰 것은 `target/` 이고
그건 언제든 재생성되므로 그것만 뺍니다.

```
CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target
```

**전역 환경변수로 설정하지 않았습니다** — 사용자 셸 설정을 바꾸는 것이므로, 빌드 배선에서
명시적으로 주는 편이 낫습니다.

`build/` 는 앞으로도 쌓이므로 주기적으로 지웁니다. 지우기 전에 활성 빌드가 없고 미커밋 변경이
없는지 확인하되, **프로세스가 떠 있다는 것이 빌드 중이라는 뜻은 아닙니다** — 이번에 `aapt2` 가
떠 있는 것을 활성으로 오독했는데 실제로는 4 일 8 시간 묵은 좀비였습니다. `ps -o etime` 을 함께
보아야 합니다.

## 0.5 실제로 통과한 빌드 (`rust/torch_c`)

**세 타깃 모두 빌드됩니다.** 최소 PyO3 크레이트(`rust/torch_c`, PyO3 0.29.2)로 확인했습니다.

| 타깃 | 산출물 | 검증 |
|---|---|---|
| `aarch64-apple-darwin` (호스트) | `lib_C.dylib` 470,928 B | **`_C.so` 로 이름 바꿔 `import _C` 성공**, 함수 호출까지 |
| `aarch64-linux-android` | `lib_C.so` 602,952 B | `ELF 64-bit LSB, ARM aarch64`. Python 심볼 48 개가 undefined 로 남음 — 로드 시 인터프리터가 해결하는 올바른 형태 |
| `aarch64-apple-ios` | `lib_C.dylib` 463,584 B | `Mach-O 64-bit dylib arm64`, `@rpath/Python.framework/Python` 로 실제 프레임워크 링크(심볼릭 링크 우회 없음) |

### 타깃마다 링크 배선이 다르다 — `Cargo.kt` 가 인코딩해야 할 것

**호스트/macOS.** `extension-module` 은 libpython 을 링크하지 않고 로드 시점에 해결하는데,
Apple 링커에 그것을 명시하지 않으면 모든 `Py*` 가 undefined 오류로 떨어집니다.

```toml
[target.aarch64-apple-darwin]
rustflags = ["-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]
```

**Android.** `cargo-ndk` 로 통과합니다. 추가 링크 플래그가 필요 없습니다 — ELF 는 공유 라이브러리의
undefined 심볼을 허용합니다.

```
ANDROID_NDK_HOME=~/Library/Android/sdk/ndk/27.1.12297006
PYO3_CROSS=1  PYO3_CROSS_PYTHON_VERSION=3.13
PYO3_CROSS_LIB_DIR=<배포본>/aarch64-linux-android/prefix/lib
cargo ndk -t arm64-v8a --platform 21 build --release
```

**iOS — 여기만 특별합니다.** 두 가지가 걸립니다.

1. `ld: warning: -undefined dynamic_lookup is deprecated on iOS` — macOS 의 방법이 그대로 넘어오지
   않습니다.
2. `ld: library 'python3.13' not found` — PyO3 가 `-lpython3.13` 을 내보내는데 **배포본에
   `libpython3.13.{a,dylib}` 이 없습니다.** 링크 가능한 것은 `Python.framework/Python` 뿐입니다.

이전 판본은 프레임워크 바이너리를 `libpython3.13.dylib` 로 심볼릭 링크해 통과시켰습니다(스파이크용
우회). **이번에 정식 방법을 실제로 빌드해 판정했고, 심볼릭 링크 없이 통과합니다.**

**두 후보 중 하나만으로는 안 됩니다 — 실측으로 확인했습니다.**

*   **`-F <프레임워크 디렉터리> -framework Python` 만 추가** — 실패합니다.
    `ld: library 'python3.13' not found` 이 그대로 남습니다. 이유: `pyo3-build-config` 의
    `is_linking_libpython_for_target()` 가 iOS 를 **`extension-module` 피처와 무관하게 무조건
    링크 대상으로 하드코딩**합니다(`OperatingSystem::IOS(_)` 매칭). 그래서 PyO3 의 빌드 스크립트가
    `-F`/`-framework` 를 추가하든 말든 **자기 몫의 `-lpython3.13` 을 별도로 계속 내보내고**, 그
    이름의 라이브러리가 없으니 그대로 링크 실패합니다.
*   **`PYO3_NO_PYTHON=1` 만 설정** — 더 나쁘게 실패합니다. 컴파일조차 안 됩니다.
    ```
    error: Neither abi3 or abi3t features are enabled
    ```
    `PYO3_NO_PYTHON` 은 인터프리터 탐색을 끄지만, 그 경로는 stable ABI(`abi3`) 빌드로 폴백하는데
    이 크레이트는 `abi3` 피처를 켜지 않았습니다. **`Cargo.toml` 의 abi3 여부는 §1 에서 여전히
    미결이므로, 이 크레이트 상태로는 이 방법 자체가 성립하지 않습니다.**

**실제로 통과한 조합.** PyO3 자신의 링크 지시를 막는 것과, 우리가 진짜 프레임워크 링크 인자를
주는 것 **둘 다** 필요합니다.

1. `.cargo/config.toml` 의 `[target.aarch64-apple-ios]` 에 `-F`/`-framework Python` 을 유지합니다
   (이미 반영됨).
2. `PYO3_CONFIG_FILE` 환경 변수로 PyO3 자신의 `-lpython3.13` 방출을
   `suppress_build_script_link_lines=true` 로 끕니다. `PYO3_CONFIG_FILE` 이 설정되면
   `PYO3_CROSS*` 경로보다 **완전히 우선**하므로(둘 다 줘도 무해 — 무시될 뿐), 베이스라인 빌드
   명령은 그대로 두고 이 변수만 추가하면 됩니다.

```
cat > <config 경로> <<'EOF'
implementation=CPython
version=3.13
shared=true
lib_name=Python
pointer_width=64
suppress_build_script_link_lines=true
EOF

PYO3_CONFIG_FILE=<config 경로> \
PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 \
PYO3_CROSS_LIB_DIR=<배포본>/arm64-iphoneos/lib \
cargo build --release --target aarch64-apple-ios
```

**검증.** 심볼릭 링크(`<linkstub>/libpython3.13.dylib`) 를 치워도, 위 조합만으로 `EXIT=0` 이고
산출물이 진짜입니다.

```
$ file lib_C.dylib
Mach-O 64-bit dynamically linked shared library arm64
$ otool -L lib_C.dylib
	...
	@rpath/Python.framework/Python (compatibility version 3.13.0, current version 3.13.0)
$ nm -u lib_C.dylib | grep -c '^_Py'
106
```

`@rpath/Python.framework/Python` 로그가 실제 프레임워크 의존성이 Mach-O 로드 커맨드에 박혔다는
뜻이고, `Py*` 심볼 106 개가 undefined 로 남은 것은 Android 경로와 같은 형태 — 로드 시점에
그 프레임워크가 해결하는 올바른 모양입니다. `-undefined dynamic_lookup` 처럼 "아무 데서나 찾아라"
가 아니라 **어느 라이브러리에서 찾을지가 명시적으로 기록**되므로 이쪽이 더 견고합니다.

**주의: 지금 `.cargo/config.toml` 의 `-F` 경로가 이 기계에 하드코딩되어 있습니다.**

```
-F/Volumes/macMini/caches/target-python/arm64-iphoneos
```

**커밋된 파일에 절대 경로가 들어가 있으므로 다른 기계에서는 그대로 깨집니다.** 스파이크 단계라
남겨두지만, 프레임워크 위치는 `Cargo.kt` 가 `XCode.kt` 의 경로 조회로 주입해야 할 값이지
저장소에 박힐 값이 아닙니다. **아래 `PYO3_CONFIG_FILE` 주의사항과 같은 항목입니다** — 둘 다
"빌드 배선이 타깃별로 주입" 으로 가야 합니다.

**주의: `PYO3_CONFIG_FILE` 은 실제 `_sysconfigdata` 파싱을 완전히 건너뜁니다.** 위 파일은 손으로
채운 최소 필드(`version`, `shared`, `pointer_width`) 뿐이고, `build_flags`(`Py_DEBUG`,
`WITH_PYMALLOC` 등)는 기본값입니다. 이 최소 크레이트는 그것으로 충분했지만, **`Cargo.kt` 가 이
경로를 실제로 구현할 때는 이 파일을 손으로 박아 넣지 말고, `PYO3_CROSS_LIB_DIR` 의 진짜
`_sysconfigdata__ios_arm64-iphoneos.py` 를 파싱해 `PYO3_CONFIG_FILE` 내용을 생성해야 합니다** —
그래야 빌드 플래그 불일치를 피합니다. 또한 `PYO3_CONFIG_FILE` 은 프로세스 전역 환경 변수라
`.cargo/config.toml` 의 `[env]` 로는 타깃별로 분기할 수 없습니다 — Android/host 빌드에는 주면 안
되므로, 이 변수는 **`Cargo.kt` 가 iOS 타깃 빌드를 호출할 때만** 주입해야 합니다.

### `PYO3_CROSS_LIB_DIR` 은 libpython 이 아니라 `_sysconfigdata` 를 찾는다

프레임워크를 가리켰더니 `Could not find _sysconfigdata*.py` 로 실패했습니다. PyO3 가 이 경로에서
찾는 것은 **stdlib 의 `python3.13/_sysconfigdata*.py`** 이지 라이브러리가 아닙니다.

| | `PYO3_CROSS_LIB_DIR` | 그 안의 파일 |
|---|---|---|
| Android | `aarch64-linux-android/prefix/lib` | `python3.13/_sysconfigdata__android_aarch64-linux-android.py` |
| iOS | `arm64-iphoneos/lib` | `python3.13/_sysconfigdata__ios_arm64-iphoneos.py` |

> **정정.** 이 문서의 이전 판본에서 "iOS `lib/` 는 비어 있음(`.DS_Store` 뿐)" 이라고 적었는데
> **틀렸습니다.** `unzip -l` 결과를 직접 자식만 매칭하는 정규식으로 걸러 `python3.13/` 하위를
> 통째로 놓쳤습니다. 제 필터가 만든 착시입니다. **`lib/` 에 stdlib 이 있고, 없는 것은
> `libpython*.{a,so,dylib}` 입니다** — 링크 대상이 프레임워크뿐이라는 결론은 그대로입니다.

## 1. PyO3 CPython 3.13 확장 크로스 빌드 필수 요소

*   **abi3 사용 여부 — 조사 완료. 권고는 `abi3-py313` 을 켜는 것.** 상세는 `docs/design/ABI3.md`.

    > **정정 (두 번).** 이 항목의 이전 두 판본이 모두 틀렸습니다.
    >
    > **초안**은 "3.13 전용이므로 abi3 불필요" 라고 했는데, 전용으로 삼을 3.13 자체가 근거가
    > 없었습니다. `PythonMultiplatform/gradle.properties:26` 이 **`pythonVersion=3.14.7`**
    > 입니다. `binary/` 의 아카이브가 `cpython-3.13.0+20241008` 로 멈춰 있어 제가 그것을 현재
    > 버전으로 읽었을 뿐입니다. **3.13 고정은 시작하는 순간 이미 한 버전 뒤처집니다.**
    >
    > **제 반박**도 틀렸습니다. "이 생태계가 Stable ABI 를 전제한다" 를 abi3 근거로 들었는데,
    > `PythonMultiplatform/ROADMAP.md:1236-1238` 이 명시합니다 — **`Py_LIMITED_API` 는 어디에도
    > 정의되지 않고**, 이 코드베이스에서 "abi3" 는 *어느 함수를 부를지에 대한 자율 규율*이지
    > 컴파일 모드가 아닙니다. 저는 CLAUDE.md 의 "Stable ABI 에 대한 expect 선언" 이라는 표현을
    > 컴파일 모드로 읽었습니다.
    >
    > 그리고 그 규율은 **Rust 로 넘어오지 않습니다.** Kotlin 은 Panama 로 심볼을 이름으로 찾아
    > 버전 이동에 견디지만, 컴파일된 cdylib 에는 그 도피처가 없습니다. 즉 이 근거는 방향이
    > 반대였습니다 — Rust 쪽이 **더** 취약하므로 abi3 가 **더** 필요합니다.

    조사에서 확인된 것 (실측, `docs/design/ABI3.md`):

    - **abi3 는 크로스 빌드를 쉽게 만들지 않습니다.** 대조 실험에서 abi3 와 non-abi3 가 **같은**
      `ld: library 'python3.13' not found` 로 실패합니다. **이 항목은 판단에서 빼야 합니다.**
    - **버전 불일치가 조용히 통과합니다.** 3.13 으로 빌드한 non-abi3 모듈이 3.11.10 에서 그냥
      돌았습니다 — 산출물 `_C.so` 에 ABI 태그가 없어 어느 인터프리터에나 붙습니다. 위험은
      "안 돌아감" 이 아니라 **"돌다가 참조 카운트가 깨짐"** 입니다.
    - **기능 손실이 확인되지 않았습니다.** torch 모양의 `#[pyclass(subclass, dict, weakref)]` +
      `__getbuffer__` + PyCapsule(DLPack) + `create_exception!` 이 `abi3-py313` 으로 전부
      동작했고, iOS·Android 산출물의 미해결 CPython 심볼 89 개가 **전부** `Misc/stable_abi.toml`
      안이었습니다(위반 0).
    - **비용은 경계 호출당 +1.1~1.2 ns (+4~5%)**, 컨테이너 원소 접근당 +2.25 ns. 순전파당
      파이썬 호출 186 건(IMPORT_WALLS §5)에 대입하면 마이크로초 단위로 무시 가능하되,
      **`_C` 안에서 파이썬 컨테이너를 원소 단위로 순회하지 않는다**는 규칙이 따라붙습니다.

    **결정의 핵심은 비대칭입니다.** abi3 → 버전 고정은 소스 수정 0(Limited API 가 부분집합).
    버전 고정 → abi3 는 사설·구조체 API 를 전부 찾아 대체해야 하고 일부는 대체가 없습니다.
    **되돌리기 싼 쪽에서 시작합니다.**

    **실행 전 최우선 확인:** 3.14.7 인터프리터에서 abi3 모듈 로드가 **미확인**입니다(이 기계에
    3.14 가 없어 3.11↔3.13 전방 호환만 확인). 플랫폼이 3.14.7 이므로 이것부터 봐야 합니다.
    반대 근거로 남는 것: `abi3t` 는 3.15 부터이고 3.14 free-threaded 에는 Limited API 가 없어
    abi3 를 켜면 free-threaded 인터프리터에 못 싣습니다 — 다만 free-threaded 프리빌트가
    Android·iOS 둘 다 없어 데스크톱 전용이므로 기기 추론과는 겹치지 않습니다.
*   **크로스 빌드 환경 변수 (PyO3 공식 문서 기준)**:
    *   `PYO3_CROSS=1`: PyO3에 크로스 컴파일 중임을 명시합니다.
    *   `PYO3_CROSS_PYTHON_VERSION=3.13`: 대상 파이썬 버전을 명시합니다.
    *   `PYO3_CROSS_LIB_DIR`: 타깃에 맞는 파이썬 라이브러리(`libpython`)가 위치한 경로입니다.
*   **타깃 파이썬 헤더/라이브러리 경로** — 아카이브를 직접 열어 확인했습니다. **두 플랫폼의 링크 방식이 다릅니다.**

    **Android (`aarch64-linux-android.tar.xz`)** — 통상적인 prefix 레이아웃.

    | | 경로 |
    |---|---|
    | 헤더 | `aarch64-linux-android/prefix/include/python3.13/` |
    | `PYO3_CROSS_LIB_DIR` | `aarch64-linux-android/prefix/lib/` — `libpython3.13.so`, `libpython3.so` |
    | 정적 대안 | `aarch64-linux-android/build/libpython3.13.a` |

    `prefix/lib/` 에 `libssl.a` · `libffi` · `libbz2.a` · `liblzma.a` · `libsqlite3` 가 함께 들어 있어
    정적 링크 시 이것들도 걸립니다. **`build/` 가 아니라 `prefix/` 를 씁니다** — 전자는 빌드 트리,
    후자가 설치 트리이고 헤더와 짝이 맞는 쪽이 후자입니다.

    **iOS (`arm64-iphoneos.zip`)** — **프레임워크입니다.**

    | | 경로 |
    |---|---|
    | 헤더 | `arm64-iphoneos/Python.framework/Headers/` (최상위 `include/python3.13` 에도 있음) |
    | 링크 대상 | `arm64-iphoneos/Python.framework/Python` — 5,335,216 B |
    | `lib/` | **비어 있음** (`.DS_Store` 뿐) |

    **`libpython*.a` 도 `.so` 도 없습니다.** 링크 가능한 산출물이 프레임워크 바이너리 하나뿐이므로
    `-lpython3.13` 이 아니라 프레임워크 링크가 되고, `PYO3_CROSS_LIB_DIR` 도 프레임워크를 가리켜야
    합니다. **`Cargo.kt` 는 이 비대칭을 다뤄야 합니다** — Android 는 lib, iOS 는 framework.

## 2. Android: cargo-ndk 필요성 및 설정

*   **cargo-ndk 필요 여부**: 필요합니다. Rust의 `cargo`만으로는 Android NDK 경로와 링커/컴파일러 플래그 설정이 매우 까다로우므로, 이를 대신 처리해주는 `cargo-ndk` 래퍼 도구를 사용하는 것이 표준이자 권장 방식입니다.
*   **NDK 버전**: 특정한 버전이 강제되지는 않으나, `pypackpack/docs/SPEC.md`의 `build/crossenv/android_21_arm64` 예시에 나타나듯 최소 API Level 21 이상을 기준으로 빌드해야 합니다.
*   **링커 설정**: 안드로이드에는 시스템 파이썬이 존재하지 않습니다. 따라서 `libpython.so` 등 크로스 컴파일된 파이썬 런타임을 링커가 찾을 수 있도록 링커 경로를 설정해야 하며, 배포 시 앱에 파이썬 런타임을 임베딩해야 합니다.
*   **`.so` 이름 규약**: 파이썬에서 `import torch._C`로 로드되려면 최종 산출물 파일명이 `libtorch_C.so`가 아닌 `_C.so`여야 합니다. Rust 빌드는 기본적으로 `lib` 접두사를 붙이므로, `Cargo.toml` 구성이나 빌드 후 이름 변경 처리가 필요합니다.

## 3. iOS: XCFramework 및 cargo-lipo 동향

*   **cargo-lipo 상태**: 현재 **Deprecated(유지보수 중단)** 상태입니다. 과거에는 Universal Binary 생성을 위해 쓰였으나 최신 워크플로우에서는 권장되지 않습니다.
*   **현재 권장 방식 (xcframework)**: `cargo build --target aarch64-apple-ios` (실기기용)와 `--target aarch64-apple-ios-sim` (시뮬레이터용) 등 개별 아키텍처로 타깃을 각각 빌드합니다. 이후 생성된 정적 라이브러리(`.a`)들을 Apple의 `xcodebuild -create-xcframework` 명령어를 통해 하나의 `XCFramework` 묶음으로 패키징하는 방식이 최신 표준입니다.
*   **iOS 프레임워크 패키징과 Rust cdylib 제약**: iOS는 동적 라이브러리 임의 로드에 대한 코드 서명(Code Signing) 및 보안 제약이 매우 엄격합니다. 파이썬 확장을 동적 라이브러리(`cdylib`, `.so`) 형태로 임의 로드하는 것은 제약이 따르므로, 통상적으로 Rust 코드를 정적 라이브러리(`staticlib`)로 빌드한 후 앱 메인 바이너리나 프레임워크와 정적으로 묶어서(Static Linking) 배포하는 것이 일반적입니다.

## 4. pypackpack 의 Cargo.kt 구현 방향

`pypackpack/docs/SPEC.md` 와 `BackendInterface.kt`, `Meson.kt` 소스코드를 분석한 결과, `Cargo.kt`가 구현되어야 할 인터페이스는 다음과 같습니다.

*   **인터페이스 준수**: `BackendInterface`를 상속받아 `initialize()`와 `suspend fun compile(packageName: String, extraArgs: Map<String, String>?)` 메서드를 구현해야 합니다.
*   **어댑터 패턴**: `Meson.kt`가 내부적으로 `meson` 프로세스를 호출하듯, `Cargo.kt`는 `cargo`를 래핑해야 합니다.
*   **NDK / XCode 어댑터 연계**:
    *   **Android 타깃**: `NDK.kt` 어댑터와 연계하여 내부적으로 `cargo` 대신 `cargo-ndk` 커맨드로 치환하거나, NDK 관련 링커 환경 변수를 주입하도록 분기해야 합니다.
    *   **iOS 타깃**: `XCode.kt` 어댑터와 연계하여 `cargo build --target aarch64-apple-ios` 등을 호출하고, 산출물을 `xcodebuild -create-xcframework`로 패키징하는 래퍼 역할을 수행해야 합니다.

## 5. 디스크 비용 추정 (rustup + 두 타깃 + cargo-ndk)

*   **rustup (Rust Toolchain)**: 약 450MB (Minimal Profile 사용 시) ~ 1.2GB (Default Profile 사용 시)
*   **두 타깃 (Android, iOS)**: `aarch64-linux-android`와 `aarch64-apple-ios` 툴체인은 각각 약 100~150MB를 차지하여 도합 약 300MB 소모
*   **cargo-ndk**: Cargo 플러그인용 단일 바이너리로 약 10~20MB 미만 소모
*   **툴체인 소계**: 대략 **750MB ~ 1.5GB**.

### 그러나 이 숫자로 블로커가 풀렸다고 볼 수 없다

위 추정은 **툴체인만** 셉니다. 실제로 `torch._C` 를 빌드할 때 자리를 차지하는 것은 빠져 있습니다.

| 빠진 항목 | 규모 |
|---|---|
| Android NDK | **수 GB** (위 보고서도 제외를 명시함) |
| `~/.cargo/registry` — 크레이트 소스와 캐시 | candle + PyO3 의존 트리 기준 **1~2GB** 추정 |
| `target/` 빌드 산출물 | Rust 디버그 빌드가 크고, **타깃마다 별도**로 쌓임. 수 GB |
| Xcode / iOS SDK | 수십 GB (이미 설치돼 있을 가능성이 높음 — 확인 필요) |

즉 "여유 4.6GB 로 도구 설치에는 무리가 없다" 는 **맞지만 답하는 질문이 다릅니다.** rustup 을
설치할 수 있느냐와 `torch._C` 를 빌드할 수 있느냐는 다른 질문이고, **후자는 여전히 열려 있습니다.**

`/Volumes/macMini` 는 349Gi 중 4.6Gi 여유(99% 사용)이고 내부 디스크는 48Gi 여유입니다. 저장소의
CLAUDE.md 는 외장 335GB 여유를 기록하고 있으나 **그 기록은 낡았습니다.** 공간을 확보하거나 캐시
위치를 정하는 것은 판단이 필요한 사안이라 여기서는 사실만 남깁니다.

> **정정 (문서 감사, 2026-09):** 이 문단의 "4.6Gi 여유"는 이 문서 자신의 §0 이 이미 "164Gi (이전
> 4.6Gi)"로 갱신해 둔 그 이전 값이다 — 같은 문서 안에서 최신 값과 낡은 값이 함께 남았다. 디스크
> 여유는 그 자체로 계속 변하는 값이라(오늘 재확인: 외장 86Gi, 내부 12Gi) 특정 숫자로 다시 고정하지
> 않고, "낡은 기록이 최신 절 바로 아래 남아 있었다"는 사실만 표시해 둔다.
