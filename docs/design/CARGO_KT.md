# `Cargo.kt` 인터페이스 설계안

> 이 문서는 설계안입니다. 코드는 구현하지 않았고, `pypackpack` 저장소는 읽기만 했습니다.
> 모든 주장에 파일 경로·행 번호 근거를 달았고, 실행해서 확인하지 못한 것은 "미확인"으로 표시했습니다.

## 0. 결론 요약

- `Cargo.kt` 는 `BackendInterface` 를 직접 구현하지 않습니다. `Meson.kt` 가 그렇듯 평범한 `open class` 로 두고,
  `DefaultBackend` 가 필요할 때 두 번째 협력자로 조립하는 것을 권장합니다(§2, §3).
- `NDK.kt`/`XCode.kt` 는 현재 이름만 있는 빈 파일이고, `Meson.kt` 조차 이들을 참조하지 않습니다(§3-1). "어댑터
  패턴" 이라는 주석은 아직 실현되지 않은 설계 의도이지, 따라야 할 기존 배선이 아닙니다.
- 타깃별 링크 비대칭(macOS/Android/iOS)은 `Cargo.kt` 내부에 흡수하고, `NDK.kt`/`XCode.kt` 에는 "경로 탐색"만
  맡기는 것을 권장합니다(§3-3).
- `lib_C.so → _C.so` 이름 변경은 `Cargo.kt` 의 `install()` 단계(= `Meson.install()` 대응)에서 처리합니다(§4).
- Rust 크레이트 크로스 컴파일은 이 저장소에서 **처음으로 진짜 타깃별 분기가 필요한 백엔드**입니다. Meson 백엔드는
  현재 호스트 빌드만 하고(§2-4), `Platforms.kt` 의 타깃 표기 중 일부는 Rust 트리플과 맞지 않습니다(§3-4). 이
  두 가지는 `Cargo.kt` 를 구현하기 전에 반드시 짚어야 할 선행 문제로 남겨둡니다.

---

## 1. `BackendInterface` 가 요구하는 시그니처 (실측)

`packpack/src/main/kotlin/.../compile/backend/BackendInterface.kt:1-17`:

```kotlin
package org.thisisthepy.python.multiplatform.packpack.compile.backend

enum class BackendType {
    MESON,
}

interface BackendInterface {
    fun initialize()

    suspend fun compile(
        packageName: String,
        extraArgs: Map<String, String>? = null,
    ): Result<String>
}
```

확인된 사실:

- 시그니처는 `initialize()` 와 `suspend fun compile(packageName, extraArgs): Result<String>` 두 개뿐입니다.
  다른 백엔드(예: `dependency.backend.BackendInterface`, 같은 저장소지만 다른 패키지)처럼 `companion object`
  팩토리 메서드가 없습니다 — `compile.backend.BackendInterface.kt` 전체 17줄에 `companion object` 가 없다는
  것을 확인했습니다.
- `BackendType` enum 은 `MESON` 하나뿐이고, 검색 결과 `compile.backend` 패키지 어디에서도 이 enum 을
  소비하는 코드가 없습니다(`BackendType.MESON` 을 참조하는 곳이 선언부 자체 외에는 없음). 즉 이 enum 은
  현재 죽은 코드이며, "팩토리 패턴" 이라는 인터페이스 주석(`BackendInterface.kt:7`)은 아직 실현되지 않았습니다.
- 실제 구현체는 `DefaultBackend` 하나뿐이고, 이를 소비하는 `DefaultMiddleware.compile()` 은 타입 분기 없이
  `DefaultBackend()` 를 그대로 생성합니다(`compile/middleware/DefaultMiddleware.kt:12`). 즉 오늘 시점에는
  "패키지 하나 = 백엔드 하나" 가 아니라 "패키지 하나 = `DefaultBackend` 하나, 그 안에서 `Meson` 을 호출" 구조입니다.

이 사실이 `Cargo.kt` 설계에 갖는 함의: `Cargo.kt` 를 `BackendInterface` 의 두 번째 구현체로 만들고
`BackendType.CARGO` 를 추가해 팩토리로 분기하는 "정석적인" 접근은, 이미 있는 `BackendType` 을 처음으로
실제로 쓰는 변경이 됩니다 — 기존 패턴을 따르는 것이 아니라 새로 만드는 것입니다. §3 에서 이 선택지와
대안(직접 조립)을 비교합니다.

---

## 2. `Meson.kt` 가 인터페이스를 구현한 방식 — `Cargo.kt` 가 따를 패턴

`Meson.kt` 는 `BackendInterface` 를 구현하지 **않습니다**. 322줄 전체(`compile/backend/external/Meson.kt`)에
`: BackendInterface` 선언이 없고, 클래스 선언은 다음과 같습니다(`Meson.kt:13-15`):

```kotlin
open class Meson(
    private val uv: UVBackend = UVBackend(),
) {
```

즉 `Meson` 은 독립된 어댑터 클래스이고, `BackendInterface` 를 구현하는 쪽은 이를 감싸는 `DefaultBackend`
입니다(`compile/backend/DefaultBackend.kt:11-13`):

```kotlin
class DefaultBackend(
    val meson: Meson = Meson(),
) : BackendInterface {
```

### 2-1. 3단계 생명주기

`Meson` 은 `setup` / `compile` / `install` 세 개의 `open suspend fun` 을 노출합니다
(`Meson.kt:55-89`). `DefaultBackend.compile()` 은 이 셋을 순서대로 호출합니다(`DefaultBackend.kt:40-42`):

```kotlin
meson.setup(buildDir = buildDir, options = listOf("--buildtype=$type"), workingDir = packageDir, overwrite = overwrite).getOrThrow()
meson.compile(buildDir = buildDir, options = null, workingDir = packageDir).getOrThrow()
meson.install(buildDir = buildDir, options = null, workingDir = packageDir, destdir = destdir).getOrThrow()
```

**권장:** `Cargo.kt` 도 `BackendInterface` 를 구현하지 않는 평범한 `open class Cargo` 로 두고, 이 3단계에
대응하는 메서드를 노출합니다. 다만 `cargo` 에는 "setup" 개념이 없고(빌드 디렉터리를 미리 생성하는
`meson setup` 과 달리 `cargo build` 는 그 자체로 완결), "install" 도 없습니다(`cargo install` 은 바이너리
설치용이지 라이브러리 산출물 복사가 아닙니다). 따라서 1:1 이름 대응이 아니라 **역할 대응**을 권장합니다:

| Meson 3단계 | 역할 | Cargo 대응 | 비고 |
|---|---|---|---|
| `setup` | 빌드 시스템 파일 생성 + 준비 | (생략 또는 no-op) | `Cargo.toml` 은 사용자가 이미 작성해 둔 것을 전제(§5) |
| `compile` | 실제 컴파일러/빌드 도구 실행 | `compile()` — `cargo build` / `cargo ndk ... build` 실행 | §3 참고 |
| `install` | 산출물을 `destdir` 로 복사 | `install()` — 산출물을 찾아 이름을 바꿔 `destdir` 로 복사 | §4 참고 |

`setup` 을 완전히 없애지 않고 "Cargo.toml 존재 확인 + 크레이트 이름 읽기" 정도의 얇은 검증 단계로 남겨
두면, `DefaultBackend` 쪽 오케스트레이션 코드가 Meson/Cargo 양쪽에서 형태를 맞추기 쉬워집니다(§3-2).

### 2-2. 프로세스 실행 패턴

`Meson.executeCommand()` (`Meson.kt:281-321`) 은 다음을 합니다:

- `withContext(Dispatchers.IO)` 로 감싸기
- `ProcessBuilder` 로 실행하고 `redirectErrorStream(true)` 로 stdout/stderr 합치기
- 실행 파일이 PATH 에 없으면(`uv tool install` 로 설치된 도구) 환경변수 `PATH`/`Path` 에 bin 경로를 앞에 붙이기
- exit code != 0 이면 출력 전체를 메시지로 담아 예외를 던지고, `runCatching` 으로 감싸 `Result` 로 변환

`Cargo.kt` 는 이 패턴을 그대로 재사용할 수 있습니다. 다만 PATH 주입 대신 **환경변수 주입**(`RUSTFLAGS`,
`PYO3_CROSS`, `PYO3_CROSS_LIB_DIR`, `ANDROID_NDK_HOME` 등, §3-3)이 핵심이 되므로
`processBuilder.environment()[...] = ...` 형태의 헬�퍼가 추가로 필요합니다. `Meson.executeCommand` 는 이미
`processBuilder.environment()` 를 만지는 선례가 있으므로(`Meson.kt:301-304`) 같은 스타일로 확장하면 됩니다.

### 2-3. 테스트 가능성 — `open` 이 곧 계약

`Meson` 의 메서드가 전부 `open` 인 이유는 인터페이스가 아니라 **서브클래싱으로 페이크를 만들기 위해서**
입니다. 실제 테스트에서 확인했습니다(`packpack/src/test/kotlin/.../compile/backend/DefaultBackendTest.kt:121-154`):

```kotlin
private class RecordingMeson : Meson() {
    var lastSetupBuildDir: String? = null
    override suspend fun isMesonInstalled(): Boolean = true
    override suspend fun setup(...): Result<String> { lastSetupBuildDir = buildDir; return Result.success("ok") }
    ...
}
```

그리고 `DefaultBackend(recordingMeson)` 처럼 생성자 주입으로 교체합니다(`DefaultBackendTest.kt:33,82`).
같은 파일에 있는 `MesonTest.kt` 도 동일하게 `RecordingMeson : Meson()` 패턴을 씁니다(빌드 산출물 클래스명
`MesonTest$RecordingMeson.class` 로 확인).

**권장:** `Cargo` 의 모든 public 메서드(`compile`, `install`, 프로세스 실행 헬퍼)를 `open suspend fun` 으로
선언하고, `DefaultBackend` 생성자에 `cargo: Cargo = Cargo()` 를 추가해 같은 방식으로 페이크를 주입할 수
있게 합니다. TDD 로 진행할 때 실제 `cargo`/`cargo-ndk` 바이너리 없이도 `DefaultBackend` 의 오케스트레이션
로직(어떤 인자로 어떤 메서드를 호출하는지)을 검증할 수 있는 것이 이 패턴의 핵심 가치입니다.

### 2-4. Meson 백엔드가 실제로 크로스 컴파일을 하지 않는다는 사실

`Meson.kt` 322줄 전체를 읽었지만 `--cross-file`, NDK, XCode, 타깃 트리플 관련 코드가 전혀 없습니다.
`meson setup` 호출부(`Meson.kt:60-68`, `DefaultBackend.kt:40`)에 넘어가는 옵션은
`listOf("--buildtype=$type")` 뿐이고 `target` 은 쓰이지 않습니다. `DefaultBackendTest.kt:88,94` 의 테스트도
`"aarch64-apple-darwin"` 과 `"x86_64-linux-gnu"`(주의: `Platforms.SUPPORTED_TARGETS` 에 없는 오타 같은
문자열 — 정식 표기는 `x86_64-unknown-linux-gnu`, `Platforms.kt:33`)를 그냥 디렉터리 이름 조각으로만
쓰고, `Platforms.normalizeTarget()` 을 거치지 않는다는 것을 보여줍니다. `SPEC.md:434` 도 이를
"`--target` 이 accepted 되지만 compile 단계로 forward 되지 않는다"고 명시합니다.

**함의:** "Meson 백엔드처럼 만들어라" 라는 지침을 문자 그대로 따르면 `Cargo.kt` 도 타깃을 무시하게
됩니다. 하지만 Rust 크로스 컴파일은 `cargo build --target <triple>` 처럼 타깃 지정이 선택이 아니라
필수이므로, `Cargo.kt` 는 Meson 이 아직 풀지 않은 문제(타깃 문자열 → 실제 툴체인 선택)를 스스로 풀어야
합니다. 이는 결함이 아니라 범위이지만, "기존 패턴을 따랐다" 는 주장만으로 타깃 처리를 생략할 수는
없다는 점을 설계 단계에서 명확히 해 둡니다.

---

## 3. `NDK.kt`/`XCode.kt` 어댑터 연계 — 타깃별 링크 비대칭을 어디서 흡수하나

### 3-1. 현재 상태 (실측)

```
$ wc -l .../backend/external/{Clang,MSVC,NDK,XCode,Emscripten,Cargo}.kt
4 Clang.kt
4 MSVC.kt
4 NDK.kt
4 XCode.kt
4 Emscripten.kt
4 Cargo.kt
```

여섯 파일 모두 `package` 선언과 한 줄 주석뿐입니다. 예를 들어 `NDK.kt` 전체(`NDK.kt:1-5`):

```kotlin
package org.thisisthepy.python.multiplatform.packpack.compile.backend.external

/**
 * Android NDK wrapper (adapter pattern for Clang)
 */
```

`SPEC.md:113-116` 는 이들을 "adapter pattern (Clang.kt)" 라고 문서화하지만, 이는 **의도**를 적어 둔 것이지
구현이 아닙니다. `SPEC.md:435` 도 명시적으로 "Clang/MSVC/NDK/XCode/Emscripten/Cargo backend adapters ...
are all empty placeholder files" 라고 확인합니다. 그리고 §2-4 에서 본 것처럼, 유일하게 구현된 백엔드인
`Meson.kt` 조차 이 여섯 파일 중 어느 것도 import/참조하지 않습니다. 즉 "adapter pattern (Clang.kt, MSVC.kt,
NDK.kt, XCode.kt)" 라는 서술이 붙은 `Meson.kt` 자신도 그 패턴을 실천하고 있지 않습니다
(`SPEC.md:117` 의 서술과 `Meson.kt` 실제 코드 사이의 괴리).

**함의:** `Cargo.kt` 가 `NDK.kt`/`XCode.kt` 를 어떻게 쓸지는 베낄 선례가 없고, 처음부터 설계해야 합니다.

### 3-2. `Cargo.kt` 자신이 어디까지 할지 — 두 가지 선택지

**선택지 A (권장): `DefaultBackend` 가 `Cargo` 를 두 번째 협력자로 직접 조립**

```kotlin
class DefaultBackend(
    val meson: Meson = Meson(),
    val cargo: Cargo = Cargo(),
) : BackendInterface
```

패키지 디렉터리에 `Cargo.toml` 이 있으면(`Meson.kt` 가 `.c`/`.cc`/`.cpp` 를 스캔하는 것과 대응되는 방식으로
`Meson.findPythonPackages`/`collectExtensionModules`, `Meson.kt:169-220` 참고) `cargo.compile(...)` 을
추가로 호출합니다. `BackendType` enum 은 건드리지 않습니다. 기존 `DefaultBackend` 의 "Strategy pattern"
자기 서술(`DefaultBackend.kt:9`)을 "패키지 하나의 compile() 안에서 여러 빌드 도구 전략을 조합" 으로
자연스럽게 확장하는 방식이고, 변경 범위가 `DefaultBackend.kt` + `Cargo.kt` 로 국한됩니다.

**선택지 B: `BackendType.CARGO` 를 추가하고 `compile.backend.BackendInterface` 에 `companion object` 팩토리를
새로 만들어 `dependency.backend.BackendInterface` (`dependency/backend/BackendInterface.kt:198-204`)와
대칭을 맞춤**

이쪽은 인터페이스 주석의 "Factory pattern" 을 문자 그대로 실현하지만, `DefaultMiddleware.compile()` 이
지금은 패키지당 백엔드를 하나만 고르는 구조(`DefaultMiddleware.kt:12` — `DefaultBackend()` 고정)라서, 한
패키지에 C 확장과 Rust 확장이 공존하는 경우를 팩토리 하나로 표현하기 어렵습니다. `MiddlewareInterface.compile()`
자체가 패키지당 한 번만 호출되는 시그니처(`MiddlewareInterface.kt`)이므로, "패키지 = 언어 하나" 라는
전제가 깨지는 순간 팩토리 분기보다 §A 의 조합 방식이 더 자연스럽습니다.

**결론:** 선택지 A 를 권장합니다. `BackendType` enum 은 그대로 죽은 코드로 남겨두거나, 필요하다면
문서 주석만 업데이트합니다.

### 3-3. 타깃별 링크 비대칭 — 흡수 지점

`/Volumes/macMini/thisisthepy/torchnative/docs/platform/RUST_CROSSBUILD.md` §0.5 에서 실측된 비대칭(같은 문서
49-100줄)을 그대로 인용하면:

| 타깃 | 필요한 것 | 성격 |
|---|---|---|
| macOS(호스트) | `RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup"` | 고정 플래그, 조회 불필요 |
| Android | `ANDROID_NDK_HOME`, `cargo ndk -t <abi> --platform <api>`, `PYO3_CROSS*` | NDK 설치 경로 + API 레벨 **조회** 필요 |
| iOS | `Python.framework/Python` 위치, 심볼릭 링크 또는 `-F <dir> -framework Python`, `PYO3_CROSS_LIB_DIR` | 배포본 내 프레임워크 **경로 조회** 필요 |

이 표에서 "조회가 필요한 것"과 "cargo 전용 조합 로직"을 분리하는 것이 설계의 핵심입니다:

- **`NDK.kt`**: `ANDROID_NDK_HOME` 후보 경로 탐색(`~/Library/Android/sdk/ndk/<version>`, 환경변수 우선)과,
  현재 `Platforms.kt` 의 canonical 표기에서 유실된 API 레벨을 복원하는 책임을 맡깁니다(3-4 참고). 이렇게
  하면 향후 Clang 기반 Android 빌드(C/C++ 확장의 실제 크로스 컴파일, 아직 없음 — §2-4)가 생기더라도 같은
  조회 로직을 재사용할 수 있어 "Clang 용 어댑터" 라는 원래 취지도 살릴 수 있습니다.
- **`XCode.kt`**: 배포본 디렉터리에서 `Python.framework` 경로를 찾는 책임을 맡깁니다. `RUST_CROSSBUILD.md:76-85`
  가 스파이크로 썼던 심볼릭 링크 우회(`ln -s .../Python.framework/Python <linkstub>/libpython3.13.dylib`)
  는 "정식으로는 `-F <프레임워크 디렉터리> -framework Python` 를 주거나 `PYO3_NO_PYTHON` 으로 PyO3 자체
  링크 지시를 막는 쪽이 맞다" 고 그 문서 자신이 명시하며 **미결**로 남겨 두었습니다(`RUST_CROSSBUILD.md:83-85`).
  이 설계 문서도 그 판단을 이어받아 미결로 남기고(§6), `XCode.kt` 는 "프레임워크 경로 조회" 까지만 책임지고
  실제 링크 인자 조합(`-F`/심볼릭 링크/`PYO3_NO_PYTHON` 중 무엇을 쓸지)은 `Cargo.kt` 구현 시점의 결정
  사항으로 명시적으로 미룹니다.
- **`Cargo.kt`**: `Platforms.describeTarget(target).family` (`Platforms.kt:174-235`, 이미 `windows`/`android`/
  `ios`/`macos`/`linux`/`wasm` 로 분기하는 로직이 있음)로 패밀리를 얻고, family 에 따라 `RUSTFLAGS`/
  `PYO3_CROSS*` 조합과 `cargo` vs `cargo ndk` 서브커맨드 선택을 직접 수행합니다. `NDK.kt`/`XCode.kt` 에는
  경로만 물어봅니다. macOS(호스트)는 조회할 것이 없으므로 애초에 어댑터를 거치지 않고 `Cargo.kt` 안에
  하드코딩된 분기로 충분합니다.

이렇게 나누면 `NDK.kt`/`XCode.kt` 는 "Clang 을 위한 어댑터" 라는 원래 문서 상 정체성(`SPEC.md:113-114`)을
잃지 않으면서도, `Cargo.kt` 가 그 경로 조회 결과를 재사용하는 형태로 `SPEC.md:116` 의
"adapter pattern (Clang.kt, MSVC.kt, NDK.kt, XCode.kt)" 서술을 (부분적으로) 실현할 수 있습니다.

### 3-4. 타깃 트리플 불일치 — `Cargo.kt` 가 새로 풀어야 하는 문제

`Platforms.kt` 의 `SUPPORTED_TARGETS`(`Platforms.kt:22-77`)는 Windows/Linux/Android/macOS 표기는
rustc 공식 타깃 트리플과 그대로 일치하지만, **iOS 표기는 일치하지 않습니다**:

| `Platforms.kt` 표기 | 실제 rustc 타깃 (Tier 2, `rustup target add` 로 설치 가능) |
|---|---|
| `arm64-apple-ios` (`Platforms.kt:74`) | `aarch64-apple-ios` |
| `arm64-apple-ios-simulator` (`Platforms.kt:75`) | `aarch64-apple-ios-sim` |
| `x86_64-apple-ios-simulator` (`Platforms.kt:76`) | `x86_64-apple-ios` (시뮬레이터에도 `-sim` 접미사가 붙지 않는 x86_64 쪽 표기가 별도로 있음) |

`aarch64-apple-ios` 가 Tier 2 공식 타깃이라는 점은 웹 검색으로 확인했습니다(rustc book,
`https://doc.rust-lang.org/rustc/platform-support/apple-ios.html`). 이 머신에는 `rustc`/`rustup` 이
설치되어 있지 않아(`which rustc rustup` 실패) `rustc --print target-list` 로 로컬 재검증은 못 했습니다 —
**웹 소스로만 확인, 로컬 미검증**으로 표시합니다.

또한 Android 쪽은 트리플 자체는 일치하지만(`aarch64-linux-android`, `Platforms.kt:69`), **API 레벨
정보가 canonical 표기 단계에서 유실됩니다**. `TARGET_ALIASES` 는 `android_21_arm64` 와 `android_24_arm64`
를 둘 다 `"aarch64-linux-android"` 로 매핑합니다(`Platforms.kt:100,102`). `cargo ndk --platform <API>`
는 이 API 레벨 인자가 필수인데(`RUST_CROSSBUILD.md:66`, `cargo ndk -t arm64-v8a --platform 21 build`),
정규화를 거친 뒤에는 21 인지 24 인지 알 방법이 없습니다.

**함의 (미해결로 남김, §6):** `Cargo.kt` 는 (a) `Platforms.normalizeTarget()` 을 거치지 않은 **원본** 타깃
문자열을 `extraArgs["target"]` 으로 그대로 받아 자체적으로 파싱하거나, (b) `extraArgs` 에 API 레벨을 위한
별도 키(예: `"apiLevel"`)를 추가해야 합니다. 둘 다 `Platforms.kt` 나 `BuildCommand.kt`(`--target` 옵션이
`String?` 하나뿐, `BuildCommand.kt:18`) 를 함께 고쳐야 하는 범위이므로, 이 설계 문서는 `Cargo.kt` 파일
하나의 범위를 벗어난다고 보고 문제만 기록합니다. iOS 트리플 불일치는 `Cargo.kt` 내부의 로컬 매핑
테이블(`Platforms.kt` 표기 → rustc 트리플)로 흡수할 수 있어 파일 하나로 해결 가능하지만, Android API
레벨 유실은 상위 레이어(`Platforms.kt`/`BuildCommand.kt`)의 협조가 필요합니다.

---

## 4. `lib_C.so → _C.so` 이름 변경은 어느 단계에서 하는가

### 4-1. Meson 이 이 문제를 어떻게 피하는지

`Meson.kt` 의 `py.extension_module()` 생성 코드(`Meson.kt:145-153`)는 확장 모듈 이름을 파일명에서 그대로
가져옵니다(`extension.name = source.nameWithoutExtension`, `Meson.kt:214`). Meson 의 `py.extension_module()`
자체가 플랫폼별 관례(접두사 없음, `.so`/`.pyd` 확장자)를 이미 처리하므로 `lib` 접두사 문제가 애초에
발생하지 않습니다. 즉 Meson 백엔드에는 이 설계 문서가 다루는 "이름 변경 단계" 자체가 없습니다 —
`Cargo.kt` 가 처음으로 이 문제를 스스로 풀어야 합니다.

### 4-2. Cargo 쪽 사실관계

`RUST_CROSSBUILD.md` §0.5 표(46-48줄)가 실측한 산출물명:

| 타깃 | cargo 산출물 |
|---|---|
| `aarch64-apple-darwin` | `lib_C.dylib` |
| `aarch64-linux-android` | `lib_C.so` |
| `aarch64-apple-ios` | `lib_C.dylib` |

세 경우 모두 `lib` 접두사가 붙고, macOS/iOS 는 `.dylib` 확장자입니다. 같은 문서가 macOS 호스트에서
"`_C.so` 로 이름 바꿔 `import _C` 성공" 을 실측 검증했다고 밝히고 있으므로(`RUST_CROSSBUILD.md:45`),
**CPython 은 macOS 에서도 `.dylib` 이 아니라 `.so` 확장자를 요구**한다는 것이 확인된 사실입니다.
크레이트 이름은 `[package].name` 의 `-` 를 `_` 로 치환한 것이 기본값이고, `[lib].name` 이 있으면 그것이
우선합니다(cargo 자체 규칙 — 이 저장소 밖의 일반 지식이므로 여기서는 "cargo 공식 문서 기준" 으로만
표시하고 이 세션에서 별도 검증은 하지 않았습니다).

### 4-3. 권장 위치: `Cargo.install()`

§2-1 의 3단계 대응표를 따라, 이름 변경은 `compile()` 이 아니라 `install()` 에서 처리할 것을 권장합니다:

- `compile()` 은 `cargo build`/`cargo ndk build` 를 실행하고 **원시 산출물 경로**
  (`target/<triple>/<profile>/lib<crate_name>.<원래 확장자>`)만 반환합니다. cargo 프로세스 실행과
  파일시스템 조작을 분리해 두면, `compile()` 단계만 페이크로 테스트하고 `install()` 의 이름 변경 로직은
  진짜 `cargo` 없이 순수 파일 조작으로 별도 테스트할 수 있습니다.
- `install()` 은 Meson 의 `install()` 과 대칭되는 위치(`Meson.kt:79-89`, `DefaultBackend.kt:42` 에서
  `destdir` 을 받는 지점)에서 다음을 합니다: 크레이트 이름을 `Cargo.toml` 에서 읽고(§2-1의 "얇은 setup"
  단계에서 `TomlEditor` 로 미리 읽어 둔 값을 재사용 — `Meson.kt:116` 의 `TomlEditor(pyproject.readText())`
  선례와 같은 방식), `lib` 접두사를 벗기고, 확장자를 `.so` 로 통일하고, `destdir` 아래 Python 패키지가
  기대하는 위치(`Meson.installSubdir` 이 만드는 것과 같은 `subdir`)로 복사합니다.
- 이렇게 하면 `DefaultBackend` 쪽에서 Meson 경로와 Cargo 경로가 `setup/compile/install` 이라는 같은
  모양의 3단계로 보이고, 실제 파일 이름 규칙 차이는 각 어댑터 내부에 캡슐화됩니다.

---

## 5. 구현하지 않고 남길 것과 그 이유

| 항목 | 왜 남기는가 | 근거 |
|---|---|---|
| **abi3(Limited API) 사용 여부** | 이 저장소가 CPython Stable ABI 를 대상으로 설계돼 있다는 근거와, Limited API 의 API 표면 제약이 서로 충돌하는 실질적 트레이드오프이고 이미 상위 조사 문서가 "결정 항목으로 남김" 이라고 명시했습니다. `Cargo.kt` 설계 문서에서 임의로 정하면 상위 판단을 대신하는 것이 됩니다. | `RUST_CROSSBUILD.md:104-109` |
| **Cargo.toml 자동 생성** | Meson 은 `.c`/`.cc`/`.cpp` 파일을 스캔해 `meson.build` 를 기계적으로 생성합니다(`Meson.kt:112-167`) — 개별 번역 단위 나열만으로 충분하기 때문입니다. 반면 Cargo 크레이트는 이름·버전·의존성(pyo3 버전 등)·`crate-type` 을 사람이 정해야 하는 메타데이터이고, `RUST_CROSSBUILD.md` 가 보여준 타깃별 `rustflags` 조합(macOS 예시, 54-57줄)도 자연스럽게 `Cargo.toml` 의 `[target.*]` 테이블에 들어갈 수 있는 정보입니다. 자동 생성 대신 "기존 `Cargo.toml` 을 찾아 검증만 한다" 는 좁은 범위를 권장합니다. | `Meson.kt:112-167` (대비), `RUST_CROSSBUILD.md:54-57` |
| **iOS 링크 방식의 최종 선택** (`-F`/심볼릭 링크/`PYO3_NO_PYTHON`) | 조사 문서 자신이 "스파이크용 우회" 라고 명시하며 정식 방식을 `Cargo.kt` 구현 시점의 결정 사항으로 미뤘습니다. 세 방식의 실제 동작 차이를 검증한 실측 데이터가 아직 없습니다. | `RUST_CROSSBUILD.md:76-85` |
| **Android API 레벨의 상위 레이어 배선** (`Platforms.kt`/`BuildCommand.kt` 수정) | §3-4 에서 설명한 대로 `Cargo.kt` 파일 하나의 범위를 벗어나고, `Platforms.kt` 의 다른 소비자(예: `remove --target` 의 마커 계산, `SingleWheelBundler.androidPlatformTag`)에 영향을 줄 수 있는 변경이라 별도 설계·검토가 필요합니다. | `Platforms.kt:100-103`, `Platforms.kt:352-361` (API 레벨을 이미 다루는 다른 소비자 존재) |
| **iOS 앱 배포용 XCFramework 패키징** (`cargo-lipo`/`xcodebuild -create-xcframework`) | pypackpack 은 `compile` 단계에서 wheel 을 만드는 도구이고, 앱에 임베딩할 프레임워크 패키징은 `bundle` 단계(이미 구현된 `SingleWheelBundler`/`FatWheelBundler`/`ResourceBundler`)의 책임 범위입니다. `compile` 백엔드가 프레임워크까지 만들면 단계 경계가 흐려집니다. | `SPEC.md:412-419` (compile/bundle 단계 정의), `RUST_CROSSBUILD.md:147-151` |
| **rustup/cargo 자체의 자동 설치** | Meson 은 `uv tool install meson`(PyPI 패키지)으로 자동 설치가 가능하지만(`Meson.kt:16-20`), rustup 은 PyPI 패키지가 아니라 별도 설치 스크립트(`rustup-init`)가 필요해 같은 패턴을 재사용할 수 없습니다. `cargo install cargo-ndk` 는 cargo 가 이미 있다는 전제 하에 가능하므로 이것만 자동화 대상으로 남기고, rustup 자체는 "사전 설치를 전제하고 없으면 실행 가능한 에러 메시지를 낸다" 수준으로 좁히는 것을 권장합니다. | `Meson.kt:16-20` (대비), `RUST_CROSSBUILD.md:16-21` (rustup 이 이미 수동 설치되어 있었다는 실측) |
| **Windows/MSVC 타깃 지원** | `RUST_CROSSBUILD.md` 는 macOS 호스트, Android, iOS 세 타깃만 실측했습니다. Windows 는 데이터가 전혀 없습니다. | `RUST_CROSSBUILD.md:43-48` (표에 Windows 없음) |

---

## 6. 미확인 사항 정리

- **`aarch64-apple-ios-sim`/`x86_64-apple-ios` 가 정확한 rustc Tier 2 트리플 표기인지**: `aarch64-apple-ios`
  는 웹 검색으로 확인했지만(rustc book), 시뮬레이터 쪽 두 트리플은 이번 세션에서 별도로 재검증하지
  않았습니다. `RUST_CROSSBUILD.md:17` 이 "`aarch64-apple-ios-sim`" 설치를 언급하므로 앞뒤가 맞지만, 로컬
  `rustc --print target-list` 실행으로는 확인하지 못했습니다(이 머신에 rustc 없음).
- **`[lib].name` 이 없을 때 cargo 의 기본 crate 이름 규칙**(`-` → `_` 치환)이 이 사용 사례(PyO3 cdylib)에도
  그대로 적용되는지: 일반적인 cargo 지식으로 서술했지만 이 세션에서 실제 `cargo build` 로 재검증하지
  않았습니다.
- **iOS `-F <프레임워크 디렉터리> -framework Python` 방식이 실제로 동작하는지**: `RUST_CROSSBUILD.md`
  자신도 이를 시도하지 않고 "정식으로는 이쪽이 맞다" 는 추정만 남겼습니다(83-85줄). 미검증입니다.
- **`PYO3_NO_PYTHON` 이 이 프로젝트의 배포본(embedded CPython, stable ABI 대상)과 함께 썼을 때 안전한지**:
  PyO3 문서 수준의 일반 지식이며, 이 저장소의 FFI 계층 전제(Stable ABI, `EmbedAPI.kt`)와 상호작용하는지는
  검토하지 않았습니다.
- **Windows 타깃**: §5 에서 명시한 대로 데이터 없음.
