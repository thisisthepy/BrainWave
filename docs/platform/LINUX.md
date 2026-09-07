# LINUX.md — Linux x86_64 크로스 빌드

`docs/design/DESIGN.md` §722 매트릭스(732 행 `| Linux x86_64 | cpu + cuda | 상류 그대로 | 런타임 (Hub) |`)에
Linux x86_64 가 올라 있는데 실제로 만든 적이 없다. 이 문서는 **맥에서 Linux x86_64 휠을 만드는 경로**를
한 층씩 밟아 본 기록이다.

**범위:** Linux x86_64 하나. Windows 는 이번 범위가 아니다.
**실행 검증은 이 기계에서 불가능하다** — `docker` · `colima` · `podman` · `lima` · `qemu` 가 전부 없고
설치하지 않는다. 도달 가능한 최하단은 iOS 가 도달한 것과 같다: **빌드 + 심볼 해결**.

작업 트리: 1회차 `/Volumes/macMini/worktrees/bw-linux` (`work/linux`, develop 에 머지됨),
2회차 `/Volumes/macMini/worktrees/bw-desk2` (`work/desk2`). 커밋하지 않는다.

---

## 진행 상황 요약

| 층 | 항목 | 상태 | 한 줄 |
|---|---|---|---|
| 1 | rust 타깃 `x86_64-unknown-linux-gnu` | **넘음** | 이미 설치되어 있었다. `rustup target add` 는 no-op |
| 2 | 링커 (맥 → Linux ELF) | **넘음 (2회차)** | `cargo-zigbuild` 설치로 열렸다. `zig cc` 가 드라이버·헤더·glibc stub·링커 넷을 다 준다 — §9 |
| 3 | 타깃 CPython (`PYO3_CROSS_LIB_DIR`) | **넘음** | 배포본이 이미 캐시에 있다. `build.py` 가 쓸 것이 전부 들어 있다 |
| 4 | `cargo build --target ...` | **넘음 (2회차)** | `cargo zigbuild --target x86_64-unknown-linux-gnu.2.17` — §9.2 |
| 5 | `build.py --target linux-x86_64` | **넘음** | `LinuxTarget` 추가. 1회차는 스탠드인 ELF 로, 2회차는 **우리 아티팩트로** 돌았다 |
| 6 | 심볼 해결 검증 | **넘음(약함)** | `verify_linux.py`. **iOS 만큼 강하지 않다** — §6.1 |
| 7 | Linux 에서 실행 | **불가** | 컨테이너 런타임도 Linux 기계도 없고 설치하지 않는다 — §7.2 |

**1회차가 남긴 진단이 정확했다.** 막고 있던 것은 링커가 아니라 glibc 스텁·타깃 헤더·C 드라이버
셋이었고, 권고했던 `cargo-zigbuild` 가 그 셋을 한 번에 채웠다. 2회차는 설치 허가를 받아 그것을
실행했을 뿐이고 **설계 판단은 바꾸지 않았다.**

기존 셋은 그대로다: pytests **197**, golden **2811/2811 ops=119**, `build.py --self-test` **8/8**,
호스트 휠은 빌드되고 깨끗한 venv 에 설치되어 계산까지 한다 (§4.1, §8).

`.cargo/config.toml` 은 **손대지 않았다.** 이유는 §4.4.

---

## 0. 환경

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cd /Volumes/macMini/worktrees/bw-linux
bash vendor/vendor_torch.sh
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-linux
BPY=/Volumes/macMini/caches/wheel-build-venv/bin/python
PY=/Volumes/macMini/caches/spike-venv/bin/python
bash vendor/install_shim.sh
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
```

기준선 회귀 (전부 exit 0 이어야 한다):

```sh
PYTHON=$PY sh rust/torch_c/pytests/run.sh        # 197
$PY tools/golden/compare.py                      # 2811/2811 ops=119
$BPY tools/wheel/build.py --self-test            # 8/8
```

---

## 1. rust 타깃 — 이미 있다

```sh
$ rustup target list --installed
aarch64-apple-darwin
aarch64-apple-ios
aarch64-apple-ios-sim
aarch64-linux-android
wasm32-unknown-emscripten
wasm32-unknown-unknown
wasm32-wasip1
x86_64-pc-windows-gnu
x86_64-pc-windows-msvc
x86_64-unknown-linux-gnu      <-- 이것
x86_64-unknown-linux-musl
```

`rustup target add x86_64-unknown-linux-gnu` 는 no-op 이다. **표준 라이브러리는 이미 있다.**
호스트 툴체인은 `stable-aarch64-apple-darwin` 하나뿐이다.

**넘어감.** 이 층은 아무것도 막지 않는다.

---

## 2. 링커 — 여기서 막힌다

### 2.1 이 기계에 있는 것

```sh
$ for t in cargo-zigbuild zig cross docker lld ld.lld rust-lld \
           x86_64-unknown-linux-gnu-gcc x86_64-linux-gnu-gcc clang; do ... done
cargo-zigbuild                 MISSING
zig                            MISSING
cross                          MISSING
docker                         MISSING
lld                            MISSING      (PATH 상)
ld.lld                         MISSING      (PATH 상)
rust-lld                       MISSING      (PATH 상)
x86_64-unknown-linux-gnu-gcc   MISSING
x86_64-linux-gnu-gcc           MISSING
clang                          /usr/bin/clang   (Apple clang)
```

다만 `rust-lld` 는 **PATH 에 없을 뿐 rustup 안에 있다**:

```
$(rustc --print sysroot)/lib/rustlib/aarch64-apple-darwin/bin/rust-lld
$(rustc --print sysroot)/lib/rustlib/aarch64-apple-darwin/bin/gcc-ld/ld.lld
```

LLD 는 크로스 링커라 ELF x86-64 를 링크할 수 있다. **그래서 "링커가 없다" 는 정확한 진단이 아니다.**
없는 것은 링커가 아니라 **링크 대상**이다 — 아래가 그 근거다.

### 2.2 없는 것은 링커가 아니라 glibc 스켈레톤이다

```sh
$ ls $(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/lib/self-contained/
ls: ... No such file or directory

$ ls $(rustc --print sysroot)/lib/rustlib/x86_64-unknown-linux-musl/lib/self-contained/
crt1.o  crtbegin.o  crtbeginS.o  crtend.o  crtendS.o  crti.o  crtn.o
libc.a  libunwind.a  rcrt1.o
```

**musl 타깃은 libc 를 통째로 들고 오고, gnu 타깃은 아무것도 안 들고 온다.** 이것이 층 2 의 전부다.
`x86_64-unknown-linux-gnu` 로 링크하려면 `crti.o` · `crtn.o` · `crtbeginS.o` · `crtendS.o` ·
`libc.so` · `libgcc_s.so` 가 기계 어딘가에 있어야 하는데, 이 기계에는 그중 하나도 없다.
타깃 CPython 배포본(§3)도 libc 를 담고 있지 않다 — 확인:

```sh
$ tar tzf /Volumes/macMini/caches/target-python/_download/linux.tar.gz \
    | grep -E "libc\.|crt[in1]\.o|libgcc"
python/lib/python3.13/site-packages/pip/_internal/utils/glibc.py    # 파이썬 소스, 무관
```

### 2.3 실측 — 기본 링커

```sh
mkdir -p /Volumes/macMini/caches/linux-probe/src
cd /Volumes/macMini/caches/linux-probe
printf '[package]\nname="probe"\nversion="0.0.0"\nedition="2021"\n\n[lib]\ncrate-type=["cdylib"]\n' > Cargo.toml
printf '#[no_mangle]\npub extern "C" fn probe_answer() -> i32 { 42 }\n' > src/lib.rs
cargo build --release --target x86_64-unknown-linux-gnu
```

```
error: linking with `cc` failed: exit status: 1
  = note: ld: unknown options: --version-script=... --no-undefined-version
          --as-needed -Bstatic -Bdynamic --eh-frame-hdr -z --gc-sections -z -z --strip-debug
          clang: error: linker command failed with exit code 1
```

기본 `cc` 는 Apple clang → `ld64` 다. GNU ld 옵션을 하나도 모른다. 예상대로다.

### 2.4 실측 — rust-lld

```sh
S=$(rustc --print sysroot)
LLD=$S/lib/rustlib/aarch64-apple-darwin/bin/rust-lld
RUSTFLAGS="-Clinker=$LLD -Clinker-flavor=ld.lld" \
  cargo build --release --target x86_64-unknown-linux-gnu
```

```
  = note: "..." "-flavor" "gnu" "--version-script=..." "--no-undefined-version"
          ... "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" ...
  = note: rust-lld: error: unable to find library -lgcc_s
          rust-lld: error: unable to find library -lutil
          rust-lld: error: unable to find library -lrt
          rust-lld: error: unable to find library -lpthread
          rust-lld: error: unable to find library -lm
          rust-lld: error: unable to find library -ldl
          rust-lld: error: unable to find library -lc
```

**이것이 이 조사에서 가장 중요한 한 줄이다.** rust-lld 는 ELF 링크를 **거절하지 않았다** —
`-flavor gnu` 로 정상 기동해 GNU 옵션을 전부 받아들이고, 마지막에 **라이브러리 7개를 못 찾아서만**
멈췄다. 즉 층 2 의 결손은 링커가 아니라 **glibc 스켈레톤 7개**로 정확히 한정된다.

> `-Clinker-flavor=gnu-lld` 는 stable 에서 거부된다
> (`the linker flavor gnu-lld is unstable`). 위처럼 레거시 이름 `ld.lld` 를 써야 한다.

참고: `rust-lld` 를 **직접** 부르면 `Library not loaded: @rpath/libLLVM.dylib` 로 죽는다.
rustc 가 부를 때는 문제가 없다. 수동 호출에는 `DYLD_LIBRARY_PATH=$(rustc --print sysroot)/lib` 가 필요하다.

### 2.5 후보 비교 — 무엇을 요구하는가

| 후보 | 요구하는 것 | 이 기계에서 |
|---|---|---|
| **`cross`** | `docker` 또는 `podman` | **불가.** 둘 다 없고 설치 금지다. 판정 끝 |
| **`lld` 단독** | 이미 있음 (`rust-lld`) + **glibc 스켈레톤을 따로 구해야 함** | 링커는 됐다. 스켈레톤이 없다 |
| **`cargo-zigbuild`** | `zig` + `cargo-zigbuild` | **미설치.** 아래가 근거 |
| **crosstool-ng (`brew install x86_64-unknown-linux-gnu`)** | homebrew tap `messense/macos-cross-toolchains`, 약 1 GB | 미설치 |

**추천은 `cargo-zigbuild` 다.** 근거는 세 가지이고, 전부 이 조사에서 실제로 부딪힌 것이다.

1. **결손이 정확히 zig 가 채우는 것이다.** §2.4 가 보여준 결손은 `libc`/`libgcc_s`/`libm`/`libdl`/
   `libpthread`/`librt`/`libutil` 이다. zig 는 glibc 의 **버전별 stub `.so` 를 합성**해서 들고 다닌다
   (2.17 부터). 스켈레톤만 필요하고 실행 이미지는 필요 없는 이 상황에 정확히 맞는다.
2. **§5 의 태그 문제를 같은 도구가 푼다.** manylinux 태그는 glibc 최소 버전이다(§5.2).
   zig 는 `--target x86_64-unknown-linux-gnu.2.17` 로 그 버전을 **입력으로 받는다.**
   `cross`/crosstool-ng 는 이미지·툴체인이 고정한 glibc 버전을 그냥 물려받으므로,
   태그를 바꾸려면 툴체인을 바꿔야 한다.
3. **docker 가 필요 없다.** 이 기계의 하드 제약이다.

crosstool-ng 는 2·3 은 만족하지만 1 에서 진다 — glibc 버전이 툴체인에 박혀 있고 용량이 1 GB 다.

**설치하지 않았다. 사용자가 정한다.** 필요한 것을 그대로 적으면:

```sh
# (a) zig — cargo-zigbuild 가 부르는 C 툴체인 겸 glibc stub 공급원
pip install ziglang            # 파이썬 휠로 오는 zig. brew 보다 작다
#   또는
brew install zig

# (b) cargo-zigbuild — cargo 의 --target 을 zig cc 로 넘기는 얇은 래퍼
cargo install cargo-zigbuild
#   또는
pip install cargo-zigbuild
```

그다음 §4 는 `cargo build` 대신 이렇게 된다:

```sh
cargo zigbuild --release --target x86_64-unknown-linux-gnu.2.17
```

**아직 아무것도 검증되지 않은 명령이다.** 위 두 개가 설치되기 전까지는 §4 가 막혀 있다.

### 2.6 고려했다가 버린 것 — 빈 stub 로 링크하기

ELF 는 `-shared` 링크에서 **정의되지 않은 심볼을 기본으로 허용한다.** 그러니 `libc.so` 자리에
빈 `.so` 7개를 만들어 두면 §2.4 의 에러 7줄은 사라지고 링크는 통과한다. 파이썬 확장 모듈은
이미 glibc 가 올라와 있는 프로세스에 `dlopen` 되므로 실제로 로드도 될 것이다.

**하지 않았다.** 그렇게 만든 `.so` 는 `DT_NEEDED` 도 `.gnu.version_r` 도 갖지 못한다.
그러면 §6 의 심볼 해결 검증이 **검사할 대상 자체를 잃는다** — "libc 심볼이 진짜 glibc 에 있는가" 를
묻는 검사가, 빈 stub 로 링크한 이미지에 대해서는 언제나 통과한다. 실패할 수 없는 검증이다.
iOS 의 `-undefined dynamic_lookup` 과 겉모습이 같아 보이지만 다르다 — iOS 는 **CPython 심볼만**
미정의로 두고 libSystem 은 정상적으로 링크한다. 여기서 하려던 것은 **libc 를 통째로** 미정의로
두는 것이고, 그것은 같은 종류의 주장이 아니다.

### 2.7 다만 — 빈 global-deps 라이브러리는 지금 도구로 만들어진다

§5 의 `Target.cc()` 가 만드는 `libtorch_global_deps.so` 는 **설계상 비어 있다**
(build.py `global_deps_stub`, docs/platform/VENDOR.md wall 1). 빈 라이브러리는 libc 심볼을 하나도 참조하지
않으므로 **스켈레톤이 필요 없다.** 실측:

```sh
printf 'int torchnative_global_deps_placeholder = 0;\n' > /tmp/gd.c
/usr/bin/clang -target x86_64-unknown-linux-gnu -fPIC -c /tmp/gd.c -o /tmp/gd.o
file /tmp/gd.o
#   ELF 64-bit LSB relocatable, x86-64, version 1 (SYSV), not stripped

S=$(rustc --print sysroot)
DYLD_LIBRARY_PATH=$S/lib $S/lib/rustlib/aarch64-apple-darwin/bin/rust-lld \
  -flavor gnu -shared -soname libtorch_global_deps.so \
  -o /tmp/libtorch_global_deps.so /tmp/gd.o
file /tmp/libtorch_global_deps.so
#   ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, not stripped
```

**Apple clang 이 ELF x86-64 재배치가능 오브젝트를 낸다** — LLVM 백엔드는 전부 들어 있고, 막히는 것은
링크뿐이다. 그리고 그 링크도 libc 를 안 쓰는 이 경우에는 rust-lld 로 통과한다.

따라서 층 2 의 결손은 **`_C.abi3.so` 하나에만** 걸린다. 세 갈래 중 두 갈래는 이미 뚫려 있다:

| 휠에 들어가는 것 | 이 기계에서 만들 수 있나 |
|---|---|
| `torch/_C.abi3.so` (Rust cdylib) | **아니오** — glibc 스켈레톤 필요 (§2.5) |
| `torch/lib/libtorch_global_deps.so` (빈 C) | **예** — Apple clang + rust-lld (§2.7) |
| 나머지 벤더 트리 (순수 파이썬) | 예 — 플랫폼 무관 |

---

## 3. 타깃 CPython — 이미 있다

`/Volumes/macMini/caches/target-python/x86_64-unknown-linux-gnu/` 에 CPython 3.13 Linux x86_64
배포본이 풀려 있다. `_download/linux.tar.gz` (119,852,836 B) 에서 나왔다.

**출처 판정.** 아카이브 최상위가 `python/` 하나이고 `python/build/` 가 없다 —
python-build-standalone 의 `install_only` 계열 레이아웃이다. 같은 캐시의 기존 것들과 비교하면:

| 디렉터리 | 레이아웃 | 비고 |
|---|---|---|
| `aarch64-linux-android/` | `build/` + `prefix/` | 직접 크로스 빌드한 흔적. `build.py` 는 `prefix/` 를 가리킨다 |
| `arm64-iphoneos/` | `bin/ include/ lib/ Python.framework/` | PEP 730 배포본 |
| `arm64-iphonesimulator/` | 같음 | |
| `x86_64-unknown-linux-gnu/` | `bin/ include/ lib/ share/` | **`install_only`. `prefix/` 하위 디렉터리가 없다** |

→ `build.py` 의 `python_root` 는 Android 처럼 `.../prefix` 가 아니라 **디렉터리 자체**를 가리켜야 한다.
iOS 타깃이 그렇게 하고 있으므로 그쪽이 본보기다.

**`build.py` 가 필요로 하는 것은 전부 있다:**

```
lib/python3.13/_sysconfigdata__linux_x86_64-linux-gnu.py   <-- target_sysconfig() 가 읽는 것
lib/libpython3.13.so -> libpython3.13.so.1.0               <-- PYO3_CROSS_LIB_DIR 이 가리킬 곳
include/python3.13/
```

`build_time_vars` 에서 태그에 쓸 값:

| 키 | 값 |
|---|---|
| `MULTIARCH` | `x86_64-linux-gnu` |
| `SOABI` | `cpython-313-x86_64-linux-gnu` |
| `EXT_SUFFIX` | `.cpython-313-x86_64-linux-gnu.so` |
| `HOST_GNU_TYPE` | `x86_64-unknown-linux-gnu` |
| `Py_DEBUG` | `0` |
| `SHLIB_SUFFIX` | `.so` |
| `CC` | `clang -pthread` |

**주의 — Android · iOS 와 다른 점.** `ANDROID_API_LEVEL` / `IPHONEOS_DEPLOYMENT_TARGET` 에 해당하는
**"최소 OS 버전" 필드가 sysconfig 에 없다.** `ANDROID_API_LEVEL` 은 `0` 으로 들어 있다.
즉 §5 의 태그는 **Android · iOS 가 하듯 배포본에서 유도할 수 없다.** 그 결과가 §5 다.

libpython 은 스트립되지 않았다 (241,980,216 B):

```
ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked,
BuildID[sha1]=8336b419..., with debug_info, not stripped
```

**넘어감.** 이 층은 아무것도 막지 않는다.

## 4. cargo build — 막힌다. 그리고 §2 가 짐작한 것보다 한 칸 더 깊다

### 4.1 기준선 먼저

크로스를 시도하기 전에 기존 셋이 그대로인지 확인했다 (§0 의 환경 그대로):

| 검사 | 결과 |
|---|---|
| `PYTHON=$PY sh rust/torch_c/pytests/run.sh` | **exit 0, `^ok ` 197줄** |
| `$PY tools/golden/compare.py` | **exit 0, 2811/2811, ops=119** |
| `$BPY tools/wheel/build.py --self-test` | **exit 0, 8/8** |

`vendor/vendor_torch.sh` → `native_left=0`, `py_modules=2372`.
`vendor/install_shim.sh` → `Finished release profile in 55.94s`.

### 4.2 크로스 시도

```sh
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-linux
S=$(rustc --print sysroot)
LLD=$S/lib/rustlib/aarch64-apple-darwin/bin/rust-lld
cd rust/torch_c
PYO3_CROSS_LIB_DIR=/Volumes/macMini/caches/target-python/x86_64-unknown-linux-gnu/lib \
RUSTFLAGS="-Clinker=$LLD -Clinker-flavor=ld.lld" \
  cargo build --release --target x86_64-unknown-linux-gnu
```

```
error occurred in cc-rs: failed to find tool "x86_64-linux-gnu-gcc":
  No such file or directory (os error 2)

  process didn't exit successfully:
    .../build/onig_sys-259239ca063cb03a/build-script-build (exit status: 1)
EXIT=101
```

**링크까지 가지도 못한다.** `onig_sys` 는 C 소스를 들고 있는 `-sys` 크레이트라
빌드 스크립트가 `cc-rs` 로 **타깃용 C 컴파일러**를 부른다. 의존 사슬:

```
torch_c -> candle-core -> tokenizers -> onig -> onig_sys   (C: Oniguruma)
```

### 4.3 그래서 요구사항이 커졌다

§2 는 "glibc 스켈레톤(`.so` 7개)만 있으면 된다" 로 읽혔지만, §4.2 가 그것을 정정한다.
**타깃 C 헤더도 있어야 한다.** Apple clang 은 ELF 코드 생성은 하지만(§2.7) 타깃 libc 헤더가 없다:

```sh
printf '#include <stdio.h>\nint f(void){return 0;}\n' > /tmp/hdr.c
/usr/bin/clang -target x86_64-unknown-linux-gnu -fPIC -c /tmp/hdr.c -o /tmp/hdr.o
#   hdr.c:1:10: fatal error: 'stdio.h' file not found

/usr/bin/clang -target x86_64-unknown-linux-gnu -E -v /tmp/hdr.c
#   #include <...> search starts here:
#    /Applications/Xcode.app/.../lib/clang/21/include
#   End of search list.
```

**깨끗한 실패다.** macOS SDK 헤더로 조용히 대체하지 않고 아예 없다고 말한다 —
그러니 "컴파일은 됐는데 잘못된 헤더였다" 는 종류의 사고는 여기서 안 난다.

정정된 §2 요구사항:

| 필요한 것 | 이 기계에 | 채우는 것 |
|---|---|---|
| ELF 링커 | **있다** (`rust-lld`, §2.4 로 확인) | — |
| glibc `.so` 스켈레톤 7개 | 없다 | zig / crosstool-ng |
| **타깃 libc 헤더** | 없다 | zig / crosstool-ng |
| **`TARGET_CC` (타깃 C 컴파일러 드라이버)** | 없다 | `zig cc` / `x86_64-linux-gnu-gcc` |

`cargo-zigbuild` 는 넷을 한 번에 채운다 — `zig cc` 가 드라이버·헤더·스켈레톤·링커 전부다.
`cross` 는 채우지만 docker 가 필요하고, crosstool-ng 는 채우지만 1 GB 다.
**§2.5 의 추천이 §4.2 로 더 강해졌지 약해지지 않았다.**

`cargo-zigbuild` 를 쓸 때 `onig_sys` 에 필요한 추가 배선(래퍼가 자동으로 하지만, 손으로 하면 이것):

```sh
export CC_x86_64_unknown_linux_gnu="zig cc -target x86_64-linux-gnu.2.17"
export AR_x86_64_unknown_linux_gnu="zig ar"
```

### 4.4 `.cargo/config.toml` 에 링커를 박지 않았다

`[target.x86_64-unknown-linux-gnu] linker = "..."` 를 넣고 싶어지지만 **넣지 않았다.**
두 가지 이유가 있고 둘 다 이 저장소가 이미 겪은 것이다:

- 값이 `/Users/ibrew/.rustup/...` 로 시작하는 **한 기계의 절대 경로**가 된다. cargo 는
  `rustflags` 안에서 환경변수를 전개하지 않는다. iOS 의 `-F` 가 정확히 이렇게 한 기계에
  묶였고 `docs/platform/RUST_CROSSBUILD.md` §0.5 가 그것을 결함으로 기록했다. `rust/torch_c/build.rs`
  전체가 그 정정이다.
- `cargo-zigbuild` 를 쓰면 래퍼가 링커를 **스스로** 지정한다. 미리 박아둔 값은 그것과 싸운다.

따라서 §2.4 · §4.2 의 `RUSTFLAGS` 는 **측정용 일회성 환경변수이지 커밋된 설정이 아니다.**
`rust/torch_c/.cargo/config.toml` 은 **손대지 않았다.**

## 5. `build.py --target linux-x86_64` — 넘었다

`tools/wheel/build.py` 에 `LinuxTarget` 을 넣었다. `AndroidTarget`/`IOSTarget` 이 본보기라는 지시대로
같은 세 가지(아티팩트 · 컴파일러 · 태그)를 채우지만, **태그의 출처가 반대다.** 그것이 이 층의 전부다.

### 5.1 태그 출처가 뒤집힌다

Android · iOS 는 **타깃 CPython** 에서 최소 OS 버전을 읽는다:

| 타깃 | 읽는 필드 | 어디에 있나 |
|---|---|---|
| Android | `ANDROID_API_LEVEL` | `_sysconfigdata_*.py` |
| iOS | `IPHONEOS_DEPLOYMENT_TARGET` | `_sysconfigdata_*.py` |
| **Linux** | — | **없다** (§3) |

없는 것이 실수가 아니다. **glibc 호환성은 인터프리터 빌드의 속성이 아니라 그 아티팩트의 속성이다.**
링커가 `.gnu.version_r` 에 "이 파일은 `GLIBC_2.17` 을 쓴다" 를 적어두고, 그중 가장 높은 것이
이 파일을 로드할 수 있는 가장 오래된 glibc다. `auditwheel` 이 읽는 곳이 정확히 거기다.

그래서 `LinuxTarget.platform_tag()` 는 **아티팩트를 읽고 인터프리터를 무시한다.**
(인터프리터는 여전히 확인한다 — `MULTIARCH` 가 `x86_64-linux-gnu` 가 아니면 배포본이 다른 것이므로 거절.)

`tools/wheel/binfmt.py` 에 `elf_dynamic()` 을 추가했다: `DT_SONAME` · `DT_NEEDED` ·
`.gnu.version_r` 을 읽는다. Mach-O 에도 Android 에도 대응물이 없는 섹션이다.

### 5.2 manylinux 태그 규칙 — 확인한 것

- **PEP 600**: `manylinux_${GLIBCMAJOR}_${GLIBCMINOR}_${ARCH}`. 이것을 쓴다.
- 레거시 별칭 `manylinux1`(2.5) · `manylinux2010`(2.12) · `manylinux2014`(2.17) 은 쓰지 않는다.
- **PEP 599** 의 외부 라이브러리 허용 목록도 태그의 일부다. manylinux 라는 태그의 약속이
  "목록 밖의 것에 의존하지 않는다" 이므로, `DT_NEEDED` 를 그 목록과 대조한다.
  이걸 빼면 `libopenblas.so` 를 링크한 휠이 **그러지 않았다고 약속하는** 태그를 달 수 있다.
- 하한은 2.5 (manylinux1). 그 아래는 어떤 인스톨러도 매칭하지 않으므로 거절한다.

**`packaging` 으로는 이 태그를 교차 확인할 수 없다.** Android·iOS 와 다른 점이다:

```
! packaging 26.3 has no manylinux_platforms -- tag spelling unchecked
```

`packaging.tags` 에는 `android_platforms` · `ios_platforms` 가 있고 **`manylinux_platforms` 는 없다**
(manylinux 매칭은 실행 중인 인터프리터의 glibc 를 봐야 하므로 인자로 만들 수 없다).
기존 `_confirm_with_packaging` 이 이 경우를 **시끄럽게 건너뛰도록** 이미 만들어져 있어서 그대로 썼고,
대신 PEP 600 자체의 형태 검사(`_confirm_pep600_spelling`)를 추가했다.
**Android·iOS 보다 이 한 칸이 약하다.**

### 5.3 지시받은 두 가지를 유지했다

- **낡음 검사가 새 타깃도 덮는다.** 상속이라 자동이지만, 자동이라고 믿지 않고 실제로 재현했다.
  `CARGO_TARGET_DIR/x86_64-unknown-linux-gnu/release/` 에 진짜 Linux ELF 를 놓고 mtime 을 2020 으로
  돌린 뒤:

  ```
  tools/wheel/build.py: .../x86_64-unknown-linux-gnu/release/lib_C.so is stale.
    rust/torch_c/src/lib.rs was modified 58389.0 h ... after lib_C.so was written
    ...
    Fix: docs/platform/LINUX.md §2.5 -- no toolchain on this machine can produce it yet.
  ```

- **`.DS_Store` 거절(`SKIP_NAMES`)은 건드리지 않았다.** `preflight()` 는 타깃과 무관하게 먼저 돌고,
  호스트 휠 빌드에서 그대로 통과하는 것을 확인했다.

### 5.4 부수적으로 고친 것 — 거절이 남의 타깃 이름을 대고 있었다

아티팩트가 없을 때의 메시지가 **모든 타깃에 대해** 이렇게 나오고 있었다:

```
build it for x86_64-unknown-linux-gnu first
  (scripts/device_android.sh build, or docs/platform/RUST_CROSSBUILD.md §0.5 for iOS)
```

Linux 사용자에게 `scripts/device_android.sh build` 를 실행하라고 말한다.
각 타깃은 이미 `rebuild_hint` 로 자기 답을 들고 있고 낡음 검사는 그것을 인용하고 있었으므로,
이 메시지도 같은 것을 쓰게 했다. 실측 결과:

```
--- android-arm64-v8a
  Fix: scripts/device_android.sh build
--- ios-arm64
  Fix: re-run the cross build for this target -- docs/platform/WHEEL.md §7.1 has the exact
       command (cargo build --release --target aarch64-apple-ios, with
       PYO3_CONFIG_FILE and PYO3_CROSS_LIB_DIR, TORCHNATIVE_PYTHON_FRAMEWORK_DIR)
--- linux-x86_64
  Fix: docs/platform/LINUX.md §2.5 -- no toolchain on this machine can produce it yet ...
```

`verify()` 의 `plat.startswith(("android_", "ios_"))` 도 `CROSS_TAG_PREFIXES` 로 바꿨다.
그대로 뒀으면 **올바른 manylinux 휠을 "타깃과 태그가 다르다" 며 거절**했을 자리다.

### 5.5 어디까지 실제로 돌았나

태그 유도 경로 전체가 **진짜 Linux ELF 로 끝까지** 돌았다. 아티팩트 자리에 배포본의 실제
확장 모듈(`_dbm.cpython-313-x86_64-linux-gnu.so`)을 놓고:

```
target linux-x86_64: lib_C.so (2,511,824 B)
      ELF 64-bit little-endian x86_64 dyn
  tag floor from the artefact's .gnu.version_r (glibc 2.17), not from CPython
      -- the distribution records no glibc minimum at all
  DT_NEEDED within the PEP 599 policy list: ['libpthread.so.0', 'libc.so.6']
  ! packaging 26.3 has no manylinux_platforms -- tag spelling unchecked
  tag manylinux_2_17_x86_64 is PEP 600-shaped (glibc 2.17, x86_64)
```

그다음 `cc()` 에서 이름을 대고 멈춘다 (아래). **그 스탠드인은 지웠다** — 남겨두면 다음 실행이
`_dbm.so` 를 우리 확장으로 착각한다.

```
tools/wheel/build.py: no C compiler that targets x86_64-unknown-linux-gnu.
  Tried, in order: $CC_x86_64_unknown_linux_gnu, $TARGET_CC, `zig` on PATH.
  ...
  Fix: install zig (docs/platform/LINUX.md §2.5), or point CC_x86_64_unknown_linux_gnu
  at a cross gcc such as x86_64-linux-gnu-gcc.
```

`cc()` 가 zig 를 요구하는 이유는 §2.7 이 아니라 **한 명령이어야 하기 때문**이다.
`global_deps_stub` 은 `[*cc(), "-o", out, src]` 를 한 번 실행한다. 이 기계는 컴파일과 링크를
**따로** 할 수 있지만(§2.7) 한 명령으로는 못 한다 — `clang --ld-path=<rust-lld>` 는
`Library not loaded: @rpath/libLLVM.dylib` 로 죽고, SIP 가 `/usr/bin/clang` 을 exec 할 때
`DYLD_LIBRARY_PATH` 를 벗기기 때문에 우회할 수 없다.

### 5.6 새 자체 검사 — 10 케이스

`build.py --self-test` 에 `self_test_linux()` 를 추가했다. **기존 8/8 줄은 그대로 두고**
별도 블록으로 붙였다 (기준선 신호를 바꾸지 않기 위해).

```
SELF-TEST: PASS -- 8/8 cases answered as specified

LINUX SELF-TEST of the manylinux tag derivation
  ok    glibc floor read off a real ELF -> 2.14
  ok    the target CPython records no glibc minimum to derive from
  ok    DT_NEEDED ['libc.so.6'] accepted by the PEP 599 list
  ok    manylinux_2_14_x86_64 accepted as PEP 600-shaped
  ok    a floor below manylinux1's glibc 2.5 is refused
  ok    an off-policy DT_NEEDED is refused, naming it
  ok    no version requirements -> refused, and disclaimed as a finding
  ok    unreadable image -> refused as unavailable, not as a floor
  ok    an aarch64 ELF is refused by the x86-64 target
  ok    a Mach-O is refused by the Linux target
LINUX SELF-TEST: PASS -- 10/10 cases; derivation exercised on real Linux ELF,
  no artefact of this crate (none is buildable here)
```

**이 테스트가 즉시 결함 하나를 잡았고, 그것은 내 것이었다.** 기대값을 `(2, 7)` 로 적었는데
실제는 `(2, 14)` 였다 — 처음에 버전 목록을 **문자열로 정렬해서** 보고 `GLIBC_2.7` 이 마지막인 줄
알았다. 코드는 숫자로 비교하고 있어서 맞았다. 만약 반대였다면 **하한이 너무 낮은 태그**가 되고,
그것은 로드할 수 없는 glibc 에 설치되는 휠이다.

**이 검사가 실패할 수 있는지 확인했다.** `floor = max(floor, version)` 을 `floor = version` 으로
바꾸자:

```
TAMPERED_EXIT=1
  WRONG glibc floor read off a real ELF -> 2.7
LINUX SELF-TEST: FAIL -- 1/10 wrong
```

되돌린 뒤 다시 0.

---

## 6. 심볼 해결 검증 — 넘었다. 다만 **iOS 만큼 강하지 않다**

`tools/wheel/verify_linux.py` 를 만들었다. `verify_ios_device.py` 가 본보기라는 지시대로 같은
구조지만, **먼저 말할 것은 어디까지 못 미치는가다.**

### 6.1 왜 약한가 — ELF 에는 two-level namespace 가 없다

iOS 기기 검증이 강한 이유는 Mach-O 의 **2단계 네임스페이스** 때문이다. 미정의 심볼마다
**링커가 어느 라이브러리에 바인딩했는지 파일에 적어둔다.** 그래서 심볼별로 그 라이브러리 하나에만
물어볼 수 있고, `libSystem` 에 우연히 같은 이름이 있어도 통과시키지 않는다.

**ELF 에는 그것이 없다.** `ld.so` 는 프로세스에 로드된 전부를 로드 순서대로 평면 검색한다.
그러므로 일반적으로 **이미지는 무엇이 어디서 오는지 말하지 않는다.**

예외가 정확히 하나 있고, 그것이 꽤 큰 몫을 진다:

> **심볼 버저닝.** `.gnu.version` 이 `.dynsym` 항목마다 인덱스를 주고, 미정의 심볼의 그 인덱스는
> **라이브러리별로 묶인** `.gnu.version_r` 을 가리킨다. glibc 는 자기 export 를 전부 버저닝하므로
> **모든 libc import 는 자기 라이브러리와 최소 glibc 를 명시한다** — Mach-O import 가 자기 dylib 을
> 명시하는 것과 똑같이.

**CPython 은 자기 export 를 하나도 버저닝하지 않는다.** 그래서 `Py*` import 는 아무것도 명시하지
않고, 배포본의 `libpython3.13.so` 에 대한 **합집합 검사**밖에 할 수 없다.

사다리로 적으면:

| | 미정의 심볼의 라이브러리 바인딩 |
|---|---|
| Mach-O 기기 (iOS) | **전부** 파일에 적혀 있다 |
| ELF (Linux) | **버저닝된 것만** (glibc). 나머지는 libpython 합집합. 둘 다 아니면 아무것도 없다 |

**지시받은 대로 적는다: iOS 만큼 강하지 않다.**

### 6.2 `readelf -d` / `nm -D` 로 무엇을 할 수 있는지 재봤다

이 기계의 ELF 도구 실측:

| 도구 | 있나 |
|---|---|
| `readelf` · `llvm-readelf` · `greadelf` | **없다** |
| `nm` (`/usr/bin/nm`, LLVM) | 있다. **ELF 를 읽는다** (`nm -D` 로 확인) |
| `objdump` (`/usr/bin/objdump`) · `xcrun llvm-objdump` | 있다 |

즉 `nm -D` 는 쓸 수 있고 `readelf -d` 는 못 쓴다. 그런데 **둘 다 쓰지 않았다.**
`verify_ios_device.py` 가 `nm -m` 을 부르는 것은 Mach-O 2단계 바인딩이 `nm` 출력에만 나오기
때문인데, ELF 에서 필요한 것(`.gnu.version` → `.gnu.version_r` 매핑)은 **`nm` 이 아예 출력하지 않고**
`readelf -V` 만 내며 그 `readelf` 가 없다. 그래서 `binfmt.py` 에 `elf_symbols()` 로 직접 읽었다 —
zip 아카이브 안의 바이트를 다루므로 어차피 경로가 아니라 바이트를 받아야 한다는 기존 이유와도 맞는다.

### 6.3 약한 심볼을 세지 않는다

컴파일러가 내는 모든 공유 객체는 `__gmon_start__` · `_ITM_registerTMCloneTable` ·
`_ITM_deregisterTMCloneTable` 을 **미정의 약한 심볼**로 달고 다니며, 이것들은 보통 시스템 어디에도
없다. 이를 실패로 세면 **매번 3개가 뜨고 읽는 사람이 무시하는 법을 배운다.** `STB_WEAK` 를 읽어
따로 센다.

### 6.4 어디까지 실제로 돌았나

**우리 아티팩트로는 못 돌렸다** — 존재하지 않는다(§4). 대신 배포본 안의 **진짜 Linux x86-64 CPython
확장 모듈**로 리졸버를 돌렸다. 우리 `_C.abi3.so` 와 같은 모양이다: `Py*` 를 libpython 에서 버전 없이,
libc 심볼을 glibc 에서 버전 붙여 가져온다.

```
$ python tools/wheel/verify_linux.py --self-test
  _dbm.cpython-313-x86_64-linux-gnu.so
    152 undefined (3 exported)
      84  -> libc.so.6  (bound by .gnu.version_r; needs GLIBC_2.14 ... GLIBC_2.17)
            not resolved here -- no libc.so.6 on this machine
      33  -> libpthread.so.0  (bound by .gnu.version_r; needs GLIBC_2.2.5, GLIBC_2.3.2)
      31  unversioned -- ELF records no library for these; checked as a union
            31  found in libpython (target distribution)
       4  weak, allowed to stay unresolved
       0  unresolved
```

**합집합 검사가 실패할 수 있다는 것을 같은 스위트에서 보인다.** `_tkinter` 는 unversioned import 중
70개가 libpython 이 아니라 Tcl/Tk 에서 온다. Tcl 을 안 주고 한 번, 주고 한 번 돌린다:

```
  _tkinter...so  [without Tcl]      67  unresolved   <-- 실패한다
  _tkinter...so  [with Tcl]          0  unresolved
                                    98  found in libpython
                                    64  found in libtcl9.0.so
                                     3  found in libtcl9tk9.0.so
```

```
  ok    a real Linux CPython extension resolves completely
  ok    ...and _tkinter does NOT, when Tcl is left out
  ok    ...and does resolve once Tcl is supplied
  ok    85 imports name libc.so.6 via .gnu.version_r
  ok    CPython imports name no library (the ELF limit)
SELF-TEST: PASS -- 5/5 cases, on real Linux ELF from the target distribution.
```

마지막 두 케이스가 §6.1 의 주장을 **검사로** 바꾼 것이다: libc import 는 라이브러리를 명시하고
(85개), CPython import 는 전부 `None` 이다.

### 6.5 이 검증이 말하지 않는 것

- **glibc 쪽은 해결되지 않는다.** 버저닝된 import 는 `libc.so.6` 과 `GLIBC_x.y` 를 명시하지만,
  대조할 glibc 가 이 기계에 없다. 검증된 것은 요구가 **내부적으로 일관되고 manylinux 정책 안에
  있다**는 것까지이고, 그 심볼들이 그 버전의 진짜 glibc 에 있다는 것은 **링커의 말을 믿는 것**이다.
- **우리 확장으로는 한 번도 돌지 않았다.**

---

## 7. 실행 검증에 필요한 것

> **2회차 정정.** 이 절은 층 4 가 막혀 있을 때 쓰였다. 그 층은 §9 에서 열렸으므로
> **§7.1 은 §9.1 · §9.2 로 대체되었다.** 실행 검증(§7.2)만 그대로 남는다.

### 7.1 층 4 를 열려면 (빌드) — **§9 에서 실행됨**

1회차가 여기에 적어둔 명령이 그대로 맞았다. 실제로 돌린 것과 실측은 §9 에 있다.

### 7.2 실행 검증을 하려면

이 기계에는 `docker` · `colima` · `podman` · `lima` · `qemu` 가 **전부 없고 설치하지 않는다.**
그러니 다음 중 하나가 필요하다:

| 수단 | 무엇이 열리나 |
|---|---|
| **Linux x86_64 기계 / VM** (실물, 클라우드 인스턴스, CI 러너) | `pip install` → `import torch` → 계산. iOS 시뮬레이터가 하는 것 전부 |
| **컨테이너 런타임** (colima + docker, podman, lima) | 위와 같음. `quay.io/pypa/manylinux2014_x86_64` 가 표준 이미지다 |
| **`qemu-x86_64` 유저모드 에뮬레이션** | 로드와 import 는 될 수 있으나 성능 수치는 무의미하다 |

**추가로, 이 기계에 glibc 가 생기면 §6.5 의 첫 칸이 닫힌다** — 컨테이너를 띄우지 않고
`libc.so.6` 파일만 있어도 버저닝된 84개 심볼을 그 파일에 대조할 수 있다. zig 를 설치하면
zig 가 들고 오는 glibc stub 이 그 대조 대상이 될 수 있는지는 **재보지 않았다.**

### 7.3 사다리 (2회차 기준)

```
built              됨     -- cargo-zigbuild. lib_C.so 4,809,768 B, ELF x86-64 dyn (§9.2)
tagged             됨     -- manylinux_2_17_x86_64, 우리 아티팩트에서 유도 (§9.3)
symbols resolve    부분   -- 우리 아티팩트로 돌았고 unresolved 0. 다만 glibc 쪽 117개는
                            여전히 링커의 말을 믿는다 (§6.5 · §9.4)
imports on Linux   불가   -- Linux 기계나 컨테이너 필요 (§7.2)
computes           불가   -- 같음
```

1회차 사다리는 위 두 칸이 `막힘`/`됨(스탠드인으로)` 이었다. 바뀐 것은 그 둘이고,
**아래 세 칸은 그대로다** — 도구를 설치했다고 실행 검증이 열리지는 않는다.

---

## 8. 기존 셋이 그대로인지 — 재현 명령과 실측

**변경 후** 전부 다시 돌렸다. `TORCH_C_ARTEFACT` 를 export 하지 않으면 골든이 캐시의 다른 빌드를
재므로 반드시 넣는다.

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cd /Volumes/macMini/worktrees/bw-linux
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-linux
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
BPY=/Volumes/macMini/caches/wheel-build-venv/bin/python
PY=/Volumes/macMini/caches/spike-venv/bin/python
bash vendor/vendor_torch.sh
bash vendor/install_shim.sh
```

| 명령 | 기준선 | 변경 후 |
|---|---|---|
| `PYTHON=$PY sh rust/torch_c/pytests/run.sh` | exit 0, 197 | **exit 0, 197** |
| `$PY tools/golden/compare.py` | exit 0, 2811/2811 ops=119 | **exit 0, 2811/2811 ops=119** |
| `$BPY tools/wheel/build.py --self-test` | exit 0, 8/8 | **exit 0, 8/8 + 새 10/10** |
| `$BPY tools/wheel/verify_linux.py --self-test` | (신규) | **exit 0, 5/5** |
| `$BPY tools/wheel/build.py` (호스트 휠) | — | **exit 0**, `macosx_11_0_arm64`, 2,687 entries |
| `$BPY tools/wheel/verify.py <위 휠>` | — | **exit 0**, 깨끗한 venv 에 설치 후 `aten.mm.default` 계산 |
| `$BPY tools/wheel/verify_{cross,android,ios_sim,ios_device}.py --help` | — | **전부 import 됨** |

`binfmt.py` 변경은 **추가만**이다 — `elf_info` · `describe` · `macho_*` 는 그대로이므로
Android · iOS 검증기가 읽는 것은 바뀌지 않았고, 위 마지막 줄이 그것을 확인한다.

Android · iOS 타깃은 이 워크트리에 아티팩트가 없어 휠까지 가지 못한다(기기·시뮬레이터 금지).
도달 가능한 지점까지는 확인했다 — §5.4 의 세 거절 메시지가 각각 자기 타깃의 지시를 낸다.

---

# 2회차 — 툴체인을 설치하고 층 2·4 를 열었다

작업 트리 `/Volumes/macMini/worktrees/bw-desk2` (`work/desk2`). 사용자가 빌드 도구 설치를 허가했고,
1회차가 §2.5 에서 권고한 그대로 `cargo-zigbuild` 를 설치했다. **설계 판단은 하나도 바꾸지 않았다** —
1회차의 진단(막는 것은 링커가 아니라 glibc 스텁·타깃 헤더·C 드라이버 셋)이 정확했고, 그 셋을 채우자
남은 층이 전부 열렸다.

## 9. 설치한 것

### 9.1 정확히 무엇을 설치했나

```sh
# (a) zig 0.16.0 — 전용 venv 에. 기존 두 venv 를 오염시키지 않는다
/usr/bin/python3 -m venv /Volumes/macMini/caches/zig-venv
/Volumes/macMini/caches/zig-venv/bin/pip install ziglang
/Volumes/macMini/caches/zig-venv/bin/python -m ziglang version    # 0.16.0

# (b) cargo-zigbuild 0.23.3 — ~/.cargo/bin 에
cargo install cargo-zigbuild
```

**`brew install zig` 를 쓰지 않았다.** `ziglang` 휠은 같은 컴파일러를 파이썬 패키지로 담고 있고,
전용 venv 안에 있으므로 지우는 것이 디렉터리 하나 삭제다. brew 는 시스템 상태를 바꾼다.

**`zig` 는 PATH 에 실행 파일로 존재하지 않는다.** `ziglang` 휠은 `zig` 콘솔 스크립트를 만들지 않고
`<python> -m ziglang` 로만 부른다. `cargo-zigbuild` 는 이 경우를 알고 있어서 `zig` 를 못 찾으면
`python3 -m ziglang` → `python -m ziglang` 순으로 찾는다. 그래서 zig-venv 의 `bin` 을 PATH 앞에
두는 것만으로 배선이 끝난다 — **`zig` 셸 셰임을 만들지 않았고, 어떤 설정 파일에도 경로를 박지 않았다.**

`build.py` 쪽은 이것을 몰라서 §9.5 에서 고쳤다.

### 9.2 층 2·4 — `cargo zigbuild` 로 한 번에 열렸다

먼저 자명한 cdylib 하나로 링크만 확인했다 (1회차 §2.3 과 같은 프로브):

```sh
export PATH="/Volumes/macMini/caches/zig-venv/bin:$HOME/.cargo/bin:$PATH"
export ZIG_GLOBAL_CACHE_DIR=/Volumes/macMini/caches/zig-cache
cd /Volumes/macMini/caches/linux-probe
cargo zigbuild --release --target x86_64-unknown-linux-gnu.2.17
#   EXIT=0, 2.83s
file .../libprobe.so
#   ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), dynamically linked, stripped
```

1회차 §2.4 에서 `-lc` 부터 일곱 줄로 죽던 자리다. 그다음 진짜 크레이트:

```sh
export PATH="/Volumes/macMini/caches/zig-venv/bin:$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-desk2
export ZIG_GLOBAL_CACHE_DIR=/Volumes/macMini/caches/zig-cache
export PYO3_CROSS_LIB_DIR=/Volumes/macMini/caches/target-python/x86_64-unknown-linux-gnu/lib
cd /Volumes/macMini/worktrees/bw-desk2/rust/torch_c
cargo zigbuild --release --target x86_64-unknown-linux-gnu.2.17
#   EXIT=0, Finished `release` profile in 59.22s
```

**`onig_sys` 가 그냥 지나갔다.** 1회차 §4.2 가 `failed to find tool "x86_64-linux-gnu-gcc"` 로
죽던 곳이다 — `cargo-zigbuild` 가 `CC_x86_64_unknown_linux_gnu` 등을 `zig cc` 로 설정해 주므로
`cc-rs` 가 드라이버를 찾는다. §4.3 표의 넷(**드라이버 · 헤더 · glibc 스텁 · 링커**)이 한 번에 채워졌다.

`.cargo/config.toml` 은 **여전히 손대지 않았다.** 위 명령은 전부 환경변수이고, 링커는
`cargo-zigbuild` 가 스스로 지정한다 (1회차 §4.4 가 예상한 그대로).

경고 한 줄이 나오는데 무해하다: `linker stderr: ignoring deprecated linker optimization setting '1'`
— rustc 가 `-O1` 을 넘기고 zig 0.16 의 lld 가 그것을 무시한다는 뜻이다. 프로브에서도 나온다.

```
lib_C.so   4,809,768 B   ELF 64-bit LSB shared object, x86-64, dynamically linked, stripped
SONAME     없음
DT_NEEDED  libm.so.6, libc.so.6, ld-linux-x86-64.so.2, libpthread.so.0, libdl.so.2
```

### 9.3 층 5 — 휠. 태그가 **우리 아티팩트에서** 나왔다

1회차는 배포본의 `_dbm...so` 를 스탠드인으로 놓고 태그 유도를 돌렸다. 이번엔 진짜다:

```sh
$BPY tools/wheel/build.py --target linux-x86_64
```

```
target linux-x86_64: lib_C.so (4,809,768 B)
      ELF 64-bit little-endian x86_64 dyn
  tag floor from the artefact's .gnu.version_r (glibc 2.17), not from CPython
      -- the distribution records no glibc minimum at all
  DT_NEEDED within the PEP 599 policy list:
      ['libm.so.6', 'libc.so.6', 'ld-linux-x86-64.so.2', 'libpthread.so.0', 'libdl.so.2']
  ! packaging 26.3 has no manylinux_platforms -- tag spelling unchecked
  tag manylinux_2_17_x86_64 is PEP 600-shaped (glibc 2.17, x86_64)
  + torch/lib/libtorch_global_deps.so (2,752 B, empty by design -- VENDOR.md wall 1)
  retag: macosx_11_0_universal2 -> manylinux_2_17_x86_64

dist/torchnative-0.0.2a0-cp313-abi3-manylinux_2_17_x86_64.whl
  2,687 entries, 13.9 MB compressed, 58.2 MB installed
```

아티팩트가 실제로 요구하는 버전 집합은 이렇다 (`.gnu.version_r`):

| 라이브러리 | 요구 |
|---|---|
| `libc.so.6` | 2.2.5, 2.3, 2.3.4, 2.9, 2.14, 2.15, 2.16, **2.17** |
| `libm.so.6` | 2.2.5 |
| `libpthread.so.0` | 2.2.5, 2.12 |
| `libdl.so.2` | 2.2.5 |
| `ld-linux-x86-64.so.2` | 2.3 |

최댓값이 2.17 이고 그것이 태그다. **`--target ...gnu.2.17` 이라는 입력이 출력 태그를 정한 것이
맞지만, `build.py` 는 그 입력을 모르고 아티팩트만 읽는다** — 즉 둘이 어긋나면 태그가 아티팩트를
따라간다. `2.28` 로 다시 빌드하면 태그도 따라 올라간다는 뜻이고, 고칠 곳은 없다.

**빈 global-deps 스텁은 버전 요구를 하나도 기록하지 않는다** (`DT_NEEDED []`, undefined 0).
설계상 비어 있으므로 libc 심볼을 참조하지 않기 때문이다. 이것이 `LinuxTarget.GLIBC_TARGET` 과
아티팩트 floor 사이에 **교차 검사를 넣지 않은 이유**다 — 넣어도 실패할 수 없다 (CLAUDE.md §5.5).

### 9.4 층 6 — `verify_linux.py` 가 처음으로 우리 아티팩트를 봤다

```sh
$BPY tools/wheel/verify_linux.py dist/torchnative-*manylinux*.whl     # EXIT=0
```

```
  torch/_C.abi3.so
    243 undefined (1 exported)
       1  -> ld-linux-x86-64.so.2  (needs GLIBC_2.3)
      87  -> libc.so.6             (needs GLIBC_2.14 … 2.17)
       2  -> libdl.so.2            (needs GLIBC_2.2.5)
      14  -> libm.so.6             (needs GLIBC_2.2.5)
      13  -> libpthread.so.0       (needs GLIBC_2.12, 2.2.5)
     118  unversioned -- ELF records no library for these; checked as a union
           118  found in libpython (target distribution)
       8  weak, allowed to stay unresolved (__cxa_thread_atexit_impl, getrandom, gettid…)
       0  unresolved

  torch/lib/libtorch_global_deps.so
    0 undefined (1 exported)          DT_NEEDED []
       0  unresolved
```

**§6.5 의 두 번째 칸이 닫혔다** — "우리 확장으로는 한 번도 돌지 않았다" 가 이제 거짓이다.
**첫 번째 칸은 그대로 열려 있다**: 버전 붙은 117개는 라이브러리와 최소 glibc 를 명시하지만
대조할 glibc 가 이 기계에 없다. §6.1 의 사다리(iOS 만큼 강하지 않다)는 **바뀌지 않았다.**

1회차 §7.2 가 남긴 미확인 항목 하나에 답한다: **zig 가 들고 오는 glibc stub 을 대조 대상으로
쓸 수 있는가.** 쓸 수 있다 — `ziglang/lib/libc/glibc/abilists` 가 심볼별로 어느 glibc 버전부터
있는지를 담은 단일 바이너리 테이블이고(`bits/ csu/ include/ …` 가 나머지 소스), zig 는 링크
시점에 여기서 `libc.so.6` 을 합성한다. **다만 하지 않았다.** 그렇게 만든 대조
대상은 링커가 이 아티팩트를 링크할 때 쓴 바로 그 데이터이므로, 그것으로 검증하면
**링커의 말을 링커에게 확인받는 것**이 된다. §6.5 가 "링커의 말을 믿는다" 고 적은 상태에서
한 발짝도 나아가지 않으면서 초록만 하나 늘어난다.

### 9.5 `build.py` 를 두 군데 고쳤다 — 실행해 보니 드러난 것

**(a) `cc()` 가 `zig` 실행 파일만 찾고 있었다.** `shutil.which("zig")` 하나였고, `ziglang` 휠은
실행 파일을 만들지 않는다(§9.1). 그대로 뒀으면 아티팩트는 `cargo-zigbuild` 가 zig 로 만들고
스텁은 "zig 가 없다" 며 거절하는, **한 휠의 두 네이티브 멤버가 다른 툴체인을 요구하는** 상태가 된다.
`LinuxTarget.zig_command()` 를 추가해 **`cargo-zigbuild` 와 같은 순서**로 찾게 했다:
`zig` → `python3 -m ziglang` → `python -m ziglang` → `sys.executable -m ziglang`.
`CC_x86_64_unknown_linux_gnu` / `TARGET_CC` 오버라이드는 그대로 우선한다.

**(b) `2, 17` 하드코딩을 `GLIBC_TARGET` 으로 올렸다.** `cc()` 안에 있던 매직 넘버이고,
§9.2 의 `cargo zigbuild --target ...gnu.2.17` 과 같은 숫자여야 하는데 그 사실이 어디에도 없었다.
**태그를 정하지는 않는다** — 태그는 아티팩트에서 나온다(§9.3). 그 점을 주석에 적었다.

낡은 문구 둘도 정정했다: `LinuxTarget` 독스트링의 "Nothing on this machine can build the artefact"
와 `rebuild_hint` 의 "no toolchain on this machine can produce it yet". 둘 다 §9.2 로 거짓이 되었고,
`rebuild_hint` 는 이제 **실제로 통한 명령**을 인용한다.

### 9.6 `verify_cross.py` 가 manylinux 를 거절하고 있었다

`build.py` 는 Linux 휠을 만들고 나서 `next: verify_cross.py …` 를 찍는데, 그것을 실행하면:

```
'manylinux_2_17_x86_64' is neither a PEP 738 android_* nor a PEP 730 ios_* tag.
```

**1회차에는 보일 수 없던 결함이다** — 거절당할 휠이 존재하지 않았다. `verify_linux.py` 는 심볼만
보므로, 이 상태에서는 **RECORD 해시 · 확장자 검색 가능성 · WHEEL Tag 일치 · 파일 목록 대조**를
Linux 휠에 대해서만 아무도 안 했다. `LinuxExpectation` 을 추가했다.

Android · iOS 와 다른 두 가지만 적는다:

- **`packaging` 이 철자를 확인해 주지 못한다.** `manylinux_platforms` 생성기가 없기 때문이고
  (matching 이 실행 중 인터프리터의 glibc 를 봐야 해서 인자로 만들 수 없다), `build.py` 가 이미
  같은 이유로 시끄럽게 건너뛰고 있었다. 여기서도 **문제로 올리지 않고 출력에 남긴다** —
  이 한 칸이 Android · iOS 보다 약하다는 것이 사실이고, FAIL 로 만들면 그 사실이 사라진다.
- **태그의 glibc 하한을 멤버와 대조한다.** iOS 의 `minos` 검사와 같은 모양이다. Android 에는
  대응물이 없다(ELF 는 API 레벨을 기록하지 않는다). 이것이 이 계열에서 **유효 범위 안에서
  재태깅해도 틀릴 수 있는** 유일한 자리다: `manylinux_2_5_x86_64` 는 어떤 최신 배포판이든 매칭하는
  멀쩡한 태그이고, 그 휠은 2.17 을 요구하는 `_C.abi3.so` 를 로드 못 하는 glibc 에 설치된다.

PEP 599 목록과 `GLIBC_ABI_DT_RELR` 순서 규칙은 **`build.LinuxTarget` 에서 import** 했다.
복사하면 둘이 조용히 어긋난다.

```
$BPY tools/wheel/verify_cross.py dist/torchnative-*manylinux*.whl      # EXIT=0
  tag                 manylinux_2_17_x86_64  (PEP 600-shaped: glibc 2.17, x86_64)
  ! packaging has no manylinux_platforms, so unlike the android and ios tags
    this spelling is not confirmed against pip's own generator
  glibc               torch/_C.abi3.so needs 2.17 <= the tag's 2.17
  glibc               torch/lib/libtorch_global_deps.so records no requirement
  binaries            2   (둘 다 ELF 64-bit little-endian x86_64 dyn)
  ext suffix          .abi3.so present in libpython3.13.so
  file list           identical to …macosx_11_0_arm64.whl (2,685 entries)
```

### 9.7 자체검사가 **자기 자신에서** 실패할 수 없는 검사를 하나 찾아냈다

`verify_cross.py --self-test` 를 Linux 휠로 처음 돌리자 9개 중 하나가 안 잡혔다:

```
  NOT CAUGHT  extension built for the wrong platform
```

`_wrong_elf_machine()` 이 `e_machine := EM_X86_64` **대입**이었다. aarch64 Android 휠에는 손상이지만
**x86-64 manylinux 휠에는 무연산이다.** 그 결함 모드는 "없어서" 통과하고 있었다 —
CLAUDE.md §5.5 가 말하는 바로 그 모양이고, **Linux 휠이 존재하기 전까지는 드러날 수 없었다.**
대입을 뒤집기(x86_64 ↔ aarch64)로 바꿨다. Android 쪽 동작은 그대로다(aarch64 → x86_64).

그리고 이 계열 전용 결함 모드를 하나 더 넣었다 — §9.6 의 하한 검사가 실제로 실패하는지:

```
$BPY tools/wheel/verify_cross.py dist/torchnative-*manylinux*.whl --self-test   # EXIT=0

  caught      extension built for the wrong platform
  caught      global-deps library missing
  caught      global-deps library under the host's name
  caught      wall-4 marker missing
  caught      a member edited without updating RECORD
  caught      part of the vendored tree dropped
  caught      WHEEL Tag: out of step with the filename
  caught      platform tag no installer would generate
  caught      abi tag downgraded from abi3
  caught      tag floor below the glibc the members actually need     <-- 새것
SELF-TEST: PASS -- 10/10 fault modes rejected
```

### 9.8 자체검사에 비순환 케이스를 하나 넣었다

`self_test_linux()` 의 케이스 1~6 은 전부 배포본의 ELF 를 읽고 **그 파서가 읽어낸 것**을 확인한다.
아티팩트가 존재하지 않을 때는 그것이 할 수 있는 최선이었다. 이제 케이스 7 을 넣었다:

> **`cargo zigbuild --target x86_64-unknown-linux-gnu.2.17` 에 준 버전이, 유도를 거쳐 그대로
> 돌아오는가.** 다른 도구에 준 *입력*을 이 코드의 *출력*과 대조하므로 순환하지 않는다.
> zig 가 요청한 버전을 무시했거나, 의존성이 더 새 glibc 심볼을 끌어왔거나,
> `GLIBC_TARGET` 이 §9.2 의 명령과 어긋나면 실패한다.

**세 갈래를 다 실측했다.**

```
정상            ok    our own lib_C.so derives manylinux_2_17_x86_64, the version zig was asked for
                LINUX SELF-TEST: PASS -- 11/11 cases on real Linux ELF, including this crate's own artefact

GLIBC_TARGET 을 (2,28) 로 바꿈:
                TAMPERED_EXIT=1
                WRONG our own lib_C.so derives manylinux_2_28_x86_64, ...
                      got manylinux_2_17_x86_64; either rebuild at 2.28 or move GLIBC_TARGET to match
                LINUX SELF-TEST: FAIL -- 1/11 wrong          (되돌린 뒤 다시 0)

아티팩트 없음 (CARGO_TARGET_DIR 미설정):
                ! case 7 skipped -- no cross-built lib_C.so (...)
                  This is a SKIP, not a pass: the one non-circular case did not run.
                  Build it with: PYO3_CROSS_LIB_DIR=... cargo zigbuild --release --target ...
                LINUX SELF-TEST: PASS -- 10/10 ...; this crate's own artefact was NOT among them
```

세 번째가 중요하다. **PASS 줄이 자기가 무엇을 안 봤는지 말한다** — 아티팩트가 없는 기계에서
`11/11` 이 `10/10` 으로 조용히 줄어들면 그것은 초록으로 읽힌다.

---

## 10. 2회차 회귀 — 기존 셋이 그대로인지

```sh
export PATH="/Volumes/macMini/caches/zig-venv/bin:$HOME/.cargo/bin:$PATH"
cd /Volumes/macMini/worktrees/bw-desk2
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-desk2
export ZIG_GLOBAL_CACHE_DIR=/Volumes/macMini/caches/zig-cache
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
BPY=/Volumes/macMini/caches/wheel-build-venv/bin/python
PY=/Volumes/macMini/caches/spike-venv/bin/python
bash vendor/vendor_torch.sh          # py_modules=2372, native_left=0
bash vendor/install_shim.sh
```

| 명령 | 기준선 | 2회차 |
|---|---|---|
| `PYTHON=$PY sh rust/torch_c/pytests/run.sh` | exit 0, 197 | **exit 0, `^ok ` 197줄** |
| `$PY tools/golden/compare.py` | exit 0, 2811/2811 ops=119 | **exit 0, 2811/2811 ops=119** |
| `$BPY tools/wheel/build.py --self-test` | exit 0, 8/8 + 10/10 | **exit 0, 8/8 + 11/11** (케이스 7 추가, §9.8) |
| `$BPY tools/wheel/verify_linux.py --self-test` | exit 0, 5/5 | **exit 0, 5/5** |
| `$BPY tools/wheel/build.py` (호스트 휠) | exit 0, `macosx_11_0_arm64` | **exit 0**, 2,687 entries |
| `$BPY tools/wheel/verify.py <위 휠>` | exit 0 | **exit 0**, 깨끗한 venv 에서 `aten.mm.default` 계산 |
| `$BPY tools/wheel/build.py --target linux-x86_64` | (1회차엔 불가) | **exit 0**, `manylinux_2_17_x86_64`, 2,687 entries |
| `$BPY tools/wheel/verify_cross.py <manylinux 휠>` | (1회차엔 거절) | **exit 0** |
| `$BPY tools/wheel/verify_cross.py <manylinux 휠> --self-test` | (신규) | **exit 0, 11/11** |
| `$BPY tools/wheel/verify_linux.py <manylinux 휠>` | (1회차엔 불가) | **exit 0**, unresolved 0 |

> **10/10 이 아니라 11/11 인 이유.** 이 절을 처음 쓸 때는 10개였다. Windows 를 붙이면서
> (`docs/platform/WINDOWS.md`) 결함 목록이 계열별로 구성되게 바뀌었고, 그때 "확장 모듈 자체가 없음" 이
> 별도 결함 모드로 분리됐다. Linux 쪽 결함 11개는 위 표의 나머지 줄을 바꾸지 않는다.

**Android · iOS 검증기에 대한 회귀 우려 한 가지를 직접 확인했다.** §9.7 에서
`_wrong_elf_machine()` 을 대입에서 뒤집기로 바꿨는데, Android 휠(aarch64)에서는 전과 같은 방향
(aarch64 → x86_64)으로 동작한다. `verify_cross.py` 의 나머지 변경은 **`manylinux` 로 시작하는
태그에서만 도는 새 분기**이고, `Expectation.parse` 의 android/ios 두 줄은 그대로다.
이 워크트리에는 Android · iOS 아티팩트가 없어 그 휠들을 실제로 만들 수 없다(기기·시뮬레이터 금지) —
**그러므로 위 문장은 코드를 읽은 결과이지 실측이 아니다.** Android · iOS 휠이 있는 트리에서
`verify_cross.py --self-test` 를 한 번 돌리는 것이 남아 있다.

## 11. 설치 목록 — 다음 사람이 같은 환경을 만들려면

2회차에서 새로 설치한 것은 **둘뿐이고, 둘 다 빌드 도구다.**

| 설치한 것 | 버전 | 위치 | 명령 |
|---|---|---|---|
| `ziglang` (zig 컴파일러) | 0.16.0 | `/Volumes/macMini/caches/zig-venv` | `/usr/bin/python3 -m venv <위>` 후 `pip install ziglang` |
| `cargo-zigbuild` | 0.23.3 | `~/.cargo/bin/cargo-zigbuild` | `cargo install cargo-zigbuild` |

부수적으로 생기는 것:

| 경로 | 무엇 | 지워도 되나 |
|---|---|---|
| `/Volumes/macMini/caches/zig-cache` | `ZIG_GLOBAL_CACHE_DIR`. 합성된 glibc stub 과 컴파일 캐시 | 예 (다시 만들어진다) |
| `/Volumes/macMini/caches/cargo-target-desk2/x86_64-unknown-linux-gnu/` | 크로스 빌드 산출물 | 예 |
| `/Volumes/macMini/caches/linux-probe*` | §9.2 의 프로브 크레이트 | 예 |

**설치하지 않은 것과 그 이유:**

- `docker` · `colima` · `podman` · `lima` · `qemu` — 지시에서 금지. 실행 검증은 그래서 열리지 않는다
- `brew install zig` — 시스템 상태를 바꾼다. venv 쪽이 되돌리기 쉽다 (§9.1)
- `cross` — docker 를 요구한다 (1회차 §2.5)
- crosstool-ng (`messense/macos-cross-toolchains`) — 1 GB 이고 glibc 버전이 툴체인에 박힌다 (1회차 §2.5)

**설정 파일에 아무것도 박지 않았다.** `.cargo/config.toml` · `~/.zshrc` · 저장소의 어떤 파일에도
`zig` 경로나 링커 경로가 들어가지 않는다. 배선은 전부 위 표의 `export` 로만 이루어진다.

---

## 12. 다음 — Windows

Linux 가 선 뒤 같은 사다리를 Windows x86_64 에 대해 밟았고, 그 기록은 **`docs/platform/WINDOWS.md`** 에 있다.
한 줄로 옮기면:

| | Linux x86_64 | Windows x86_64 |
|---|---|---|
| 툴체인 | `cargo-zigbuild` (한 프로그램이 넷을 다 준다) | `cargo-xwin` + **셰임 넷** (xwin 은 데이터만 준다) |
| 태그 | `manylinux_2_17_x86_64`, **아티팩트에서 유도** | `win_amd64`, **고정 이름. 유도할 것이 없다** |
| 휠 안의 확장 이름 | `torch/_C.abi3.so` | **`torch/_C.pyd`** (Windows 의 dynload 표에 `.abi3.so` 가 없다) |
| global-deps 라이브러리 | 빈 `.so` 를 싣는다 | **싣지 않는다** (`_load_global_deps()` 가 Windows 에서 즉시 return) |
| 심볼 해결 | **iOS 만큼 강하지 않다** — 버저닝된 것만 귀속 | **iOS 만큼 강하다** — import table 이 심볼마다 DLL 을 적는다 |

마지막 줄이 이 두 문서를 나란히 읽어야 하는 이유다. §6.1 이 ELF 의 한계로 지목한 것이
PE 에는 없고, 그 차이는 도구가 아니라 **파일 형식**에서 온다.

**Windows 를 넣으면서 공유 코드를 건드렸으므로**(`Target.extension_member`,
`global_deps_name = None`, `_repack(renames=…)`), 위 §10 의 Linux 회귀는 그 변경 **이후**
숫자로 갱신되어 있다. `docs/platform/WINDOWS.md` §9 가 두 타깃을 한 표에 놓은 것이다.
