# `_C.dylib` iOS 시뮬레이터 로드 검증

> **Correction (문서 감사, 2026-09):** 이 문서도 `docs/devices/DEVICE_LOAD.md`(안드로이드 쪽 자매
> 문서, 이 감사가 이미 확인)와 같은 3-op 단계의 스냅샷입니다. `docs/platform/IOS.md` 가 이 계보의
> 후속 문서로 보이나(이 감사가 아직 읽지 않음), 이 문서 자체의 결론("로드 성공")은 여전히
> 유효합니다 — 뒤집혔다는 근거는 없습니다.

**결론: 로드 성공.** `import _C`, `_C._aten_implemented()`, `_C._aten_dispatch("aten.full.default",
[2, 3], 1.5)` 모두 실제 iOS 시뮬레이터(iPhone 16 Pro, iOS 18.0) 프로세스 안에서 정상 동작했다.
`docs/platform/RUST_CROSSBUILD.md` §0.5 가 확인한 것은 **`aarch64-apple-ios`(실기기) 산출물의 링크**뿐이었고,
이번에는 (1) 시뮬레이터 타깃(`aarch64-apple-ios-sim`)으로 별도 빌드하고 (2) 그 산출물을 실제
시뮬레이터 프로세스 안에서 실행해 심볼 해결까지 실측했다.

## 사용한 환경

- 시뮬레이터: iPhone 16 Pro, iOS 18.0 (`8F9F769D-BAB2-4316-B348-CF1F77220051`). `xcrun simctl list
  devices` 로 나열되는 것 중 아무도 부팅해 두지 않은 것을 골라 `simctl boot` 로 직접 띄웠고, 검증
  후 `simctl shutdown` 으로 되돌려 놓았다.
- 호스트: macOS 26.5.1 (25F80), Xcode 26.6 (17F113).
- 타깃 CPython: `PythonMultiplatform` 저장소의 `binary/arm64-iphonesimulator.zip` 을
  `/Volumes/macMini/caches/target-python/arm64-iphonesimulator/` 에 압축 해제(205 MB). 저장소
  자체는 읽기 전용으로 두고, 압축 해제본은 캐시 디렉터리에 새로 만들었다 — `arm64-iphoneos`
  와 나란한 형제 디렉터리로, `RUST_CROSSBUILD.md` §0 이 이미 그 위치를 캐시 관례로 쓰고 있다.
- 대상 아티팩트: `/Volumes/macMini/caches/cargo-target/aarch64-apple-ios-sim/release/lib_C.dylib`
  (1,704,928 바이트, Mach-O 64-bit dylib arm64) — 이번에 새로 빌드했다(아래 "빌드" 절).

## 빌드 — `rust/torch_c` 를 건드리지 않고 시뮬레이터 타깃 추가

`.cargo/config.toml` 에는 `[target.aarch64-apple-ios]`(실기기) 규칙만 있고 `aarch64-apple-ios-sim`
은 **다른 타깃 트리플이라 그 규칙을 상속하지 않는다.** 지시대로 `rust/torch_c` 를 전혀 열지도
고치지도 않고, `RUST_CROSSBUILD.md` §0.5 가 이미 문서화해 둔 값(프레임워크 경로, `PYO3_CONFIG_FILE`
내용)만 가져와 환경 변수·명령줄 인자로 재현했다.

```bash
rustup target list --installed   # aarch64-apple-ios-sim 이미 설치돼 있음 확인

FRAMEWORK_DIR=/Volumes/macMini/caches/target-python/arm64-iphonesimulator
CFG=/tmp/ios_sim_check/pyo3_config_sim.txt
cat > "$CFG" <<'EOF'
implementation=CPython
version=3.13
shared=true
lib_name=Python
pointer_width=64
suppress_build_script_link_lines=true
EOF

CARGO_TARGET_AARCH64_APPLE_IOS_SIM_RUSTFLAGS="-C link-arg=-F$FRAMEWORK_DIR -C link-arg=-framework -C link-arg=Python" \
PYO3_CONFIG_FILE="$CFG" \
PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 \
PYO3_CROSS_LIB_DIR="$FRAMEWORK_DIR/lib" \
CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target \
cargo build --release --target aarch64-apple-ios-sim
```

**`EXIT=0`.** `CARGO_TARGET_<TRIPLE>_RUSTFLAGS` (트리플 전용, 대문자+언더스코어) 를 써서
`.cargo/config.toml` 의 실기기 규칙과 섞이지 않게 했다 — 전역 `RUSTFLAGS` 를 썼다면 다른 세 작업이
동시에 돌리고 있는 실기기/Android 빌드에 영향을 줄 위험이 있었다.

```
$ file lib_C.dylib
Mach-O 64-bit dynamically linked shared library arm64
$ otool -L lib_C.dylib
	@rpath/Python.framework/Python (compatibility version 3.13.0, current version 3.13.0)
	/usr/lib/libiconv.2.dylib
	/usr/lib/libSystem.B.dylib
$ nm -u lib_C.dylib | grep -c '^_Py'
98
```

실기기 빌드와 같은 형태 — 심볼릭 링크 우회 없이 `@rpath/Python.framework/Python` 으로 진짜
프레임워크 의존성이 박혔고, `Py*` 심볼 98개가 undefined 로 남아 로드 시점 해결을 기다린다.

**빌드 로그에 경고 하나가 남는다:** `linker stderr: ld: -undefined dynamic_lookup is deprecated on
iOS-simulator`. 컴파일은 `EXIT=0` 으로 통과했고 `otool -L` 로 프레임워크가 실제로 링크됐음을
확인했으므로 이번 검증(로드 여부)에는 영향이 없었지만, 어디서 `-undefined dynamic_lookup` 이
아직 나오는지는 `rust/torch_c` 안을 보지 않고는 특정할 수 없었다 — 다른 작업이 그 디렉터리를
쓰고 있어 확인하지 않았다. `Cargo.kt` 구현 시점에 다시 볼 필요가 있다.

## 기기(시뮬레이터)에 올린 방법 — Android 와 근본적으로 다른 지점

**Android 문서(`DEVICE_LOAD.md`)의 패턴을 그대로 반복할 수 없었다.** Android 배포본에는
`bin/python3.13` 실행 파일이 있어 `adb push` + `adb shell ./bin/python3.13 -c '...'` 로 끝났지만,
**iOS 배포본(`arm64-iphonesimulator.zip`)에는 실행 파일이 아예 없다.**

```
$ file Python.framework/Python
Python.framework/Python: Mach-O 64-bit dynamically linked shared library arm64   ← MH_DYLIB, MH_EXECUTE 아님
$ find arm64-iphonesimulator -iname "python*" -type f
Python.framework/Python                 # 라이브러리
Python.framework/Headers/pythonrun.h    # 헤더
Python.framework/Headers/Python.h       # 헤더
```

`bin/` 아래 있는 것은 `arm64-apple-ios-simulator-clang` 류의 **크로스 컴파일러 래퍼**뿐이다.
즉 iOS 배포본은 "인터프리터를 그대로 실행"하는 형태가 아니라 "앱(또는 최소한 앱을 흉내 낸
실행 파일)이 프레임워크를 링크해 `Py_Initialize` 를 스스로 호출"하는 형태만 지원한다. 이것이
`DESIGN.md` §8 이 적어둔 "iOS 는 임의 경로의 dylib 를 `dlopen` 할 수 없고, 네이티브 코드는 서명된
앱 번들 안에 있어야 한다" 는 제약과 같은 결의 것이다 — CPython 배포본 자체가 이미 "독립 실행형
인터프리터를 기기에 그냥 올려 돌리는" 경로를 막아 둔 셈이다.

**앱은 만들지 않되, 최소 실행 파일 하나는 필요했다.** `rust/torch_c` 를 건드리지 않는 선에서
`/tmp` 에 15줄짜리 C 드라이버를 작성해 `Py_Initialize` → `PyRun_SimpleString("import _C; ...")` →
`Py_Finalize` 만 하도록 했다(Info.plist, 번들 구조, 코드 서명 엔타이틀먼트 없음).

```c
// /tmp/ios_sim_check/driver.c (torchnative/PythonMultiplatform 저장소 밖)
#include <Python.h>
int main(void) {
    Py_Initialize();
    int rc = PyRun_SimpleString(
        "import sys; print(sys.version); print(sys.path)\n"
        "import _C\n"
        "print(_C._aten_implemented())\n"
        "t = _C._aten_dispatch('aten.full.default', [2, 3], 1.5)\n"
        "print(t)\n"
    );
    if (PyErr_Occurred()) PyErr_Print();
    Py_Finalize();
    return rc;
}
```

시뮬레이터용 clang 래퍼로 컴파일(배포본의 `bin/arm64-apple-ios-simulator-clang` 이
`xcrun --sdk iphonesimulator clang -target arm64-apple-ios-simulator` 를 감싼 셸 스크립트임을
확인하고 그대로 사용):

```bash
FRAMEWORK_DIR=/Volumes/macMini/caches/target-python/arm64-iphonesimulator
"$FRAMEWORK_DIR/bin/arm64-apple-ios-simulator-clang" \
  -I"$FRAMEWORK_DIR/include/python3.13" \
  -F"$FRAMEWORK_DIR" -framework Python \
  -Wl,-rpath,"$FRAMEWORK_DIR" \
  -o driver driver.c
```

`EXIT=0`. `otool -L driver` → `@rpath/Python.framework/Python`, `/usr/lib/libSystem.B.dylib` 만
있는 `MH_EXECUTE` 바이너리가 나왔다.

### 코드 서명 — 걸릴 것으로 예상했는데 걸리지 않았다

빌드된 `driver` 와 `_C.so`(=`lib_C.dylib` 이름만 바꿈) 를 `codesign -dv` 로 보면 **아무 서명 절차도
밟지 않았는데 이미 서명돼 있다.**

```
CodeDirectory v=20400 ... flags=0x20002(adhoc,linker-signed)
Signature=adhoc
TeamIdentifier=not set
```

Apple 링커(`ld`)가 arm64 Mach-O 를 링크할 때 자동으로 ad-hoc 서명을 붙인다 — Apple Developer
인증서나 프로비저닝 프로파일이 전혀 필요 없었다. **그리고 iOS 시뮬레이터는 이 ad-hoc 서명만으로
실행을 허용한다.** `DESIGN.md` §5·§8 이 우려한 "코드 서명 제약"은 **실기기·앱스토어 배포 경로의
것이지, 시뮬레이터 경로에는 적용되지 않는다** — 시뮬레이터는 호스트 macOS 커널 위에서 도는
일반 프로세스라 실기기 수준의 서명 강제(entitlement, provisioning profile, 앱 샌드박스)가 없다.
이번 검증 범위에서 코드 서명은 결국 막힘 지점이 아니었다.

### 실행 — `simctl spawn`, 그리고 `DYLD_ROOT_PATH` 가 갈리는 지점

먼저 호스트에서 직접 실행을 시도했다(시뮬레이터 프로세스도 결국 arm64 macOS 프로세스이므로).
**이건 막혔다:**

```
$ PYTHONHOME=... DYLD_FRAMEWORK_PATH=... ./driver
dyld[83201]: DYLD_ROOT_PATH not set for simulator program
```

시뮬레이터 타깃(`-target arm64-apple-ios-simulator`)으로 링크된 바이너리는 dyld 가 "이것은
시뮬레이터 프로그램" 이라는 마커(`LC_BUILD_VERSION PLATFORM_IOSSIMULATOR`)를 로드 커맨드에서
읽고, `DYLD_ROOT_PATH`(시뮬레이터 런타임 루트) 없이는 실행을 거부한다. 즉 **호스트 프로세스로
그냥 띄우는 것과 "시뮬레이터 프로세스로 띄우는 것" 은 다르다** — Android 에서 `adb shell` 이
기기 위 프로세스를 직접 실행했던 것과 대응되는 지점이 iOS 에서는 `simctl spawn` 이다.

`xcrun simctl boot <UDID>` 로 iPhone 16 Pro(iOS 18.0)를 띄운 뒤:

```bash
UDID=8F9F769D-BAB2-4316-B348-CF1F77220051
SIMCTL_CHILD_PYTHONHOME="$FRAMEWORK_DIR" \
SIMCTL_CHILD_DYLD_FRAMEWORK_PATH="$FRAMEWORK_DIR" \
SIMCTL_CHILD_PYTHONPATH="$FRAMEWORK_DIR/lib/python3.13:/tmp/ios_sim_check" \
xcrun simctl spawn "$UDID" /tmp/ios_sim_check/driver
```

`simctl spawn` 은 필요한 `DYLD_ROOT_PATH` 등 시뮬레이터 런타임 배선을 자동으로 채워주고,
`SIMCTL_CHILD_` 접두사가 붙은 변수만 골라 자식 프로세스 환경으로 넘긴다(`iosMain/README.md` 가
이미 적어 둔 관례와 일치).

**1차 시도(`PYTHONPATH` 에 드라이버 디렉터리를 안 넣었을 때):**

```
ModuleNotFoundError: No module named '_C'
sys.path= ['.../lib/python3.13', '.../lib/python313.zip', '.../lib/python3.13/lib-dynload',
           '.../lib/python3.13/site-packages']
```

Android 는 `sys.path` 에 `''`(cwd)가 자동으로 들어 있어 `_C.so` 를 루트에 두는 것만으로 충분했지만,
**여기서는 cwd 가 `sys.path` 에 없었다.** `-c` 문자열 실행이 아니라 임베딩 API(`PyRun_SimpleString`)
경로라 `-c`/스크립트 인자에 따라 파이썬이 자동으로 넣어주는 `''` 항목이 안 붙은 것으로 보인다.
`SIMCTL_CHILD_PYTHONPATH` 에 드라이버 디렉터리(`/tmp/ios_sim_check`)를 명시적으로 추가하니 해결됐다.

## 결과 (판정은 종료 코드로)

`simctl spawn` 이 자식 프로세스의 종료 코드를 그대로 돌려주는 것을 먼저 확인했다 — `import _C`
가 실패했던 1차 시도에서 `PyRun_SimpleString` 이 `rc=-1` 을 반환하자 드라이버가 그 값을 `return`
했고, 셸에서 관측한 `$?` 는 `255`(부호 없는 8비트로 감싼 `-1`)였다. 값이 정확히 대응하므로,
2차 시도의 `$?` 를 그대로 판정 근거로 쓴다.

```
$ echo $?
0
```

출력:

```
sys.version= 3.13.0+ (heads/3.13:d894d467a61, Oct 18 2024, 02:15:03) [Clang 16.0.0 (clang-1600.0.26.3)]
sys.path= ['.../arm64-iphonesimulator/lib/python3.13', '/tmp/ios_sim_check',
           '.../lib/python313.zip', '.../lib/python3.13/lib-dynload', '.../lib/python3.13/site-packages']
PROBE_A_OK
_C module: <module '_C' from '/tmp/ios_sim_check/_C.so'>
PROBE_B_OK
implemented: ['aten.add.Tensor', 'aten.full.default', 'aten.mm.default']
PROBE_C_OK
dispatch: TensorBase(shape=[2, 3], dtype=float32, device=cpu)
PROBE_D_OK
PyRun_SimpleString rc=0
DRIVER_EXIT_RC=0
```

**98개 미해결 `Py*` 심볼이 시뮬레이터 프로세스 안에서 `Python.framework/Python` 으로 실제
해결됐고**, `_aten_implemented()` 가 구현된 op 3개를 그대로 반환했으며, `_aten_dispatch` 가
`TensorBase(shape=[2, 3], dtype=float32, device=cpu)` 를 크래시·트레이스백 없이 반환했다.
프로세스는 `PROBE_D_OK` 까지 전부 찍고 `DRIVER_EXIT_RC=0` 으로 정상 종료했다.

검증 후 `xcrun simctl shutdown <UDID>` 로 부팅했던 시뮬레이터를 되돌려 놓았다.

## Android 와 무엇이 달랐는가 (요약)

| | Android | iOS 시뮬레이터 |
|---|---|---|
| 배포본에 실행 파일이 있는가 | 있음 (`bin/python3.13`) | **없음** — `Python.framework/Python` 은 라이브러리(`MH_DYLIB`)뿐 |
| 최소 실행 경로 | `adb shell ./bin/python3.13 -c '...'` | 프레임워크를 링크하는 드라이버 실행 파일을 직접 만들어야 함 |
| 기기 실행 진입점 | `adb shell` (파일시스템 직접 접근) | `xcrun simctl spawn <UDID> <경로>` (`DYLD_ROOT_PATH` 를 대신 채워줌) |
| 환경 변수 전달 | 그대로 `adb shell` 인자/환경 | `SIMCTL_CHILD_` 접두사 필요 |
| `sys.path` 에 cwd 자동 포함 | 됨 (`-c` 실행) | **안 됨** (임베딩 API 경로) — `PYTHONPATH` 로 직접 추가해야 함 |
| 코드 서명 | 해당 없음(ELF, 서명 개념 없음) | **예상과 달리 걸리지 않음** — 링커가 자동으로 ad-hoc 서명, 시뮬레이터는 그것만으로 실행 허용 |
| 심볼릭 링크(`.so` 부속 라이브러리) 제약 | `adb push --sync` 가 심볼릭 링크 거부(root 아님) | 이번 최소 검증(프레임워크 하나만 링크)에는 해당 사항 없음 — `_C.dylib` 가 추가 `.dylib` 를 요구하게 되면 다시 볼 문제 |

## 이번 검증이 답하지 않은 것 (범위 밖)

- **실기기(`aarch64-apple-ios`) 실행은 검증하지 않았다.** `RUST_CROSSBUILD.md` §0.5 는 실기기
  산출물의 **링크**만 확인했고, 이번 작업은 **시뮬레이터** 산출물을 새로 빌드해 **실행**까지
  확인했다. 실기기는 프로비저닝 프로파일·엔타이틀먼트·개발자 서명이 필요해 시뮬레이터의 ad-hoc
  서명 결론을 그대로 외삽할 수 없다 — 오히려 `DESIGN.md` §5·§8 이 우려한 서명 제약은 실기기
  쪽에서 다시 확인해야 한다.
- **실제 앱 번들(Xcode 프로젝트) 안에서의 로드는 확인하지 않았다.** `simctl spawn` 으로 띄운
  독립 프로세스와, 앱 번들에 `Python.framework` 를 `Embed & Sign` 으로 넣고 앱 자체 프로세스에서
  `Py_Initialize` 를 부르는 실제 배포 경로는 번들 구조·`@executable_path` rpath 해석이 다르다.
  다만 PythonMultiplatform 의 `iosMain/README.md` 가 이미 `PYTHONHOME`·`SIMCTL_CHILD_` 요구사항을
  같은 결로 적어 두고 있어, 이번 결과가 그 경로와 같은 종류의 신뢰도를 준다고 볼 수는 있다.
- **`x86_64-apple-ios-simulator`(Intel Mac 시뮬레이터) 타깃은 시도하지 않았다.** 호스트가 Apple
  Silicon 이라 `aarch64-apple-ios-sim` 만으로 충분했다.
- **`libssl`/`libcrypto`/`libsqlite3` 등 부속 라이브러리는 다루지 않았다.** `_C.dylib` 가 지금은
  `Python.framework` 하나만 요구해 해당 없었지만, op 구현이 늘어 이들에 의존하게 되면 iOS
  프레임워크 배포본에 그런 부속 `.dylib` 가 아예 없다는 점(§`RUST_CROSSBUILD.md`)을 다시 봐야 한다.
- **빌드 로그의 `-undefined dynamic_lookup is deprecated on iOS-simulator` 경고의 근원은 특정하지
  않았다.** `rust/torch_c` 를 열지 않기로 한 제약 때문이다. `EXIT=0` 이고 `otool -L` 로 실제
  프레임워크 링크를 확인했으므로 이번 로드 검증 결과에는 영향이 없었지만, `Cargo.kt` 가 이
  경로를 정식으로 구현할 때 재확인이 필요하다.

## iOS 검증을 제대로 하려면 앞으로 필요한 것

1. **실기기 실행.** 시뮬레이터의 ad-hoc 서명 결론은 실기기에 적용되지 않는다 — 개발자 서명 +
   프로비저닝 프로파일이 있는 상태에서 `xcrun devicectl`(또는 Xcode)로 실기기에 올려 같은 패턴을
   재현해야 진짜 결론이 나온다.
2. **실제 앱 번들 경로.** `Python.framework` 를 `Frameworks/` 로 `Embed & Sign` 하고, 앱 프로세스
   자체(별도 `simctl spawn` 실행 파일이 아니라)에서 `Py_Initialize` 를 부르는 것이 최종 배포
   형태다. rpath 가 `@rpath` 에서 `@executable_path/Frameworks` 로 바뀌는 지점이라 이번 검증의
   `-Wl,-rpath,"$FRAMEWORK_DIR"`(절대 경로) 그대로는 안 옮겨진다.
3. **`Cargo.kt`/`XCode.kt` 배선에 이번 환경 변수 조합을 반영.** `CARGO_TARGET_AARCH64_APPLE_IOS_SIM_RUSTFLAGS`
   + `PYO3_CONFIG_FILE`(`suppress_build_script_link_lines=true`) + `PYO3_CROSS_LIB_DIR` 3종 조합이
   시뮬레이터 타깃에서도 실기기와 같은 패턴으로 동작함을 확인했으니, 실기기용 배선을 만들 때
   타깃 트리플만 분기해 재사용할 수 있다.
4. **`x86_64-apple-ios-simulator` 재현.** Intel Mac CI 러너가 있다면 같은 절차가 필요하다.
