# WINDOWS.md — Windows x86_64 크로스 빌드

`docs/platform/LINUX.md` 가 Linux x86_64 를 여섯 층으로 나눠 밟은 기록이고, 이 문서는 같은 사다리를
Windows x86_64 에 대해 밟은 기록이다. **Linux 가 먼저 서고 나서 시작했다.**

**실행 검증은 이 기계에서 불가능하다** — Windows 기계도 VM 도 없고, 컨테이너 런타임도 없으며
설치하지 않는다. 도달 가능한 최하단은 Linux 와 같은 **빌드 + 심볼 해결**이다.
다만 아래 §5 가 말하듯 **PE 에서는 그 "심볼 해결" 이 ELF 보다 훨씬 강하다.**

작업 트리: `/Volumes/macMini/worktrees/bw-desk2` (브랜치 `work/desk2`). 커밋하지 않는다.

---

## 진행 상황 요약

| 층 | 항목 | 상태 | 한 줄 |
|---|---|---|---|
| 1 | rust 타깃 `x86_64-pc-windows-msvc` | **넘음** | 이미 설치되어 있었다 (`rustup target list --installed`) |
| 2 | MSVC 툴체인 (컴파일러 · 링커 · 헤더 · import lib) | **넘음** | `cargo-xwin` + **셰임 넷**. §3.2 가 그 넷이고, 이 문서에서 가장 손이 많이 간 부분이다 |
| 3 | 타깃 CPython | **넘음** | 앞선 회차가 받아둔 배포본이 그대로 쓰인다. `_sysconfigdata` 가 **없는 것이 정상**이다 — §2 |
| 4 | `cargo xwin build --target ...` | **넘음** | `_C.dll` 5,018,112 B, PE32+ x86-64. 40.80s |
| 5 | `build.py --target windows-x86_64` | **넘음** | `WindowsTarget`. `win_amd64`, 2,686 entries. 멤버 이름과 global-deps 가 다른 셋과 다르다 — §4 |
| 6 | 심볼 해결 검증 | **넘음 (강함)** | `verify_windows.py`. **iOS 만큼 강하다** — §5 |
| 7 | Windows 에서 실행 | **불가** | Windows 기계 필요 |

**Linux 와 다른 결론이 하나 있다.** `docs/platform/LINUX.md` §6.1 은 ELF 의 심볼 해결이 iOS 만큼 강할 수
없다고 적었고 그것은 맞았다. **PE 는 다르다** — import table 이 심볼마다 DLL 을 직접 적으므로
귀속(attribution)이 100% 다. §5 가 그것을 실측으로 보인다.

---

## 0. 환경

```sh
export PATH="/Volumes/macMini/caches/msvc-shims:$HOME/.cargo/bin:$PATH"
cd /Volumes/macMini/worktrees/bw-desk2
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-desk2
export XWIN_CACHE_DIR=/Volumes/macMini/caches/xwin
export XWIN_ACCEPT_LICENSE=1
BPY=/Volumes/macMini/caches/wheel-build-venv/bin/python
```

---

## 1. rust 타깃 — 이미 있다

`rustup target list --installed` 에 `x86_64-pc-windows-msvc` 와 `x86_64-pc-windows-gnu` 가
둘 다 들어 있다. **`msvc` 를 쓴다** — CPython 공식 Windows 배포본이 MSVC 로 빌드되고
`vcruntime140.dll` 을 쓰므로, `gnu`(MinGW) 로 만들면 런타임이 갈린다.

Linux 와 달리 `self-contained/` 문제가 없다: MSVC 타깃의 결손은 표준 라이브러리가 아니라
**툴체인 전체**이고, 그것이 §3 이다.

---

## 2. 타깃 CPython — 이미 있고, POSIX 와 모양이 다르다

`/Volumes/macMini/caches/target-python/x86_64-pc-windows-msvc/` 에 풀려 있다
(`_download/windows.tar.gz`, 47,207,576 B). 배포본 레이아웃:

```
python.exe  pythonw.exe
python3.dll        <-- 안정 ABI(abi3) 포워더.  902 exports
python313.dll      <-- 실제 구현.             1,657 exports
vcruntime140.dll   vcruntime140_1.dll
libs/python3.lib   libs/python313.lib        <-- import library
include/  Lib/  DLLs/  Scripts/  tcl/
```

**`_sysconfigdata_*.py` 가 없다. 그것이 결손이 아니다.** `sysconfig` 는 그 파일을 POSIX 에서만
생성하고, Windows 에서는 `_init_non_posix()` 가 `sys.prefix` 로부터 값을 계산한다. 따라서
Android · iOS · Linux 가 하듯 배포본에서 무언가를 읽어 태그를 유도하는 것이 **여기서는 불가능하고,
필요하지도 않다** (§4.1). `WindowsTarget.sysconfig()` 는 그 사실을 말하며 거절하고,
대신 `_check_distribution()` 이 위 세 파일(`libs/python3.lib`, `python3.dll`, `python313.dll`)의
존재로 배포본을 확인한다.

**`python3.lib` 대 `python313.lib` 가 abi3 의 전부다.** 전자를 링크하면 확장이 `python3.dll` 에
바인딩되어 3.13 과 그 이후를 다 서비스하고, 후자를 링크하면 3.13 하나만 서비스한다.
`PYO3_CROSS_LIB_DIR` 이 `libs/` 를 가리키고 PyO3 가 `abi3` 피처를 켠 상태이므로 전자가 선택되는데,
**그것을 빌드 플래그가 아니라 파일에서 확인한다** — §5.2.

---

## 3. MSVC 툴체인 — 여기가 실제 작업이었다

### 3.1 `cargo-xwin` 이 무엇을 주고 무엇을 안 주는가

`cargo install cargo-xwin` (0.23.1). 첫 실행에서 Microsoft 로부터 **CRT 헤더·라이브러리와
Windows SDK** 를 받아 `$XWIN_CACHE_DIR` 에 풀고, cc-rs 와 rustc 에 넘길 플래그를 만들어 준다:

```
CFLAGS_x86_64_pc_windows_msvc = --target=x86_64-pc-windows-msvc -fuse-ld=lld-link
    /imsvc <xwin>/crt/include  /imsvc <xwin>/sdk/include/ucrt
    /imsvc <xwin>/sdk/include/um  /imsvc <xwin>/sdk/include/shared ...
CC_x86_64_pc_windows_msvc = clang-cl
AR_x86_64_pc_windows_msvc = llvm-lib
```

**즉 헤더와 라이브러리는 주지만 도구는 주지 않는다.** 그래서 그대로 돌리면:

```
error occurred in cc-rs: failed to find tool "llvm-lib": No such file or directory
EXIT=101
```

`cargo-zigbuild` 와 대비된다. zig 는 드라이버·헤더·스텁·링커를 **한 프로그램 안에** 다 들고 있고,
xwin 은 **데이터만** 들고 온다. 이 기계에 없는 것:

| 필요한 것 | 이 기계에 | 무엇으로 대신했나 |
|---|---|---|
| `clang-cl` (MSVC 모드 C 드라이버) | 없음 | **`/usr/bin/clang --driver-mode=cl`** — Apple clang 이 지원한다 |
| `llvm-lib` (`lib.exe` 대체) | 없음 | **`zig lib`** — llvm-lib 자체다 (`/llvmlibempty` 안내를 그대로 낸다) |
| `lld-link` (COFF 링커) | PATH 에 없음 | **rustup 의 `rust-lld`** — §3.3 |
| `llvm-rc` · `llvm-dlltool` | 없음 | **`zig rc` · `zig dlltool`** (이 크레이트는 쓰지 않지만 넣어 두었다) |
| MSVC CRT 헤더 · Windows SDK | 없음 | **`cargo-xwin` 이 받아온다** |

**넷 중 셋이 이미 이 기계에 있었다.** Linux 때와 같은 구조다 (`docs/platform/LINUX.md` §2.1): 없는 것은
도구가 아니라 **그 도구를 그 이름으로 부를 방법**이었다.

### 3.2 셰임 넷 — `/Volumes/macMini/caches/msvc-shims`

`.cargo/config.toml` 에도 `~/.zshrc` 에도 아무것도 박지 않는다 (`docs/platform/LINUX.md` §4.4 와 같은 이유).
셰임은 캐시 디렉터리에 두고 PATH 로만 붙인다.

```sh
D=/Volumes/macMini/caches/msvc-shims; mkdir -p $D
S=$(rustc --print sysroot)
Z=/Volumes/macMini/caches/zig-venv/bin/python     # docs/platform/LINUX.md §9.1 의 그 venv

cat > $D/clang-cl <<'EOF'
#!/bin/sh
exec /usr/bin/clang --driver-mode=cl "$@"
EOF

cat > $D/lld-link <<EOF
#!/bin/sh
export DYLD_LIBRARY_PATH="$S/lib\${DYLD_LIBRARY_PATH:+:\$DYLD_LIBRARY_PATH}"
L="$S/lib/rustlib/aarch64-apple-darwin/bin/rust-lld"
if [ "\$1" = "-flavor" ]; then exec "\$L" "\$@"; else exec "\$L" -flavor link "\$@"; fi
EOF

for pair in lib:llvm-lib rc:llvm-rc dlltool:llvm-dlltool ar:llvm-ar ranlib:llvm-ranlib; do
  sub=\${pair%%:*}; n=\${pair##*:}
  printf '#!/bin/sh\nexec "%s" -m ziglang %s "$@"\n' "$Z" "$sub" > $D/$n
done
chmod +x $D/*
```

### 3.3 `lld-link` 셰임에 조건문이 들어간 이유 — 실측

rustup 은 `rust-lld` 하나와, argv[0] 으로 flavor 를 정하는 심볼릭 링크 몇 개를 함께 둔다:

```
$(rustc --print sysroot)/lib/rustlib/aarch64-apple-darwin/bin/rust-lld
$(rustc --print sysroot)/lib/rustlib/aarch64-apple-darwin/bin/gcc-ld/lld-link
```

`gcc-ld/lld-link` 를 그대로 가리켰더니 **rustc 가 링크에서 죽었다**:

```
= note: "lld-link" "-flavor" "link" "/DEF:..." "/NOLOGO" ...
= note: rust-lld: warning: ignoring unknown argument '-flavor'
        rust-lld: error: could not open 'link': No such file or directory; did you mean '/link'
```

**rustc 는 링커를 `lld-link -flavor link …` 로 부른다.** 그런데 `gcc-ld/lld-link` 는 argv[0] 으로
이미 flavor 가 정해져 있어 `-flavor` 를 인자로 알아듣지 못하고, `link` 를 입력 파일로 읽는다.
그래서 셰임이 **flavor 없는 `rust-lld`** 를 가리키고, 첫 인자가 `-flavor` 가 아닐 때만 넣어 준다.
그래야 rustc(넣어서 부름)와 cc-rs 계열(안 넣고 부름) 양쪽이 다 동작한다.

`DYLD_LIBRARY_PATH` 가 필요한 것은 `rust-lld` 가 `@rpath/libLLVM.dylib` 로 rustup 의 LLVM 을 찾기
때문이다 (`docs/platform/LINUX.md` §2.4 각주와 같은 사실). **여기서는 SIP 에 막히지 않는다** —
`rust-lld` 를 exec 하는 것이 rustc 이고 rustc 는 SIP 보호 대상이 아니다.
`docs/platform/LINUX.md` §5.5 가 `/usr/bin/clang` 에서 막혔던 것과 대비되는 지점이다.

### 3.4 층 2 · 4 실측

먼저 자명한 cdylib 하나로 (Linux 때와 같은 프로브 크레이트):

```sh
cd /Volumes/macMini/caches/linux-probe
cargo xwin build --release --target x86_64-pc-windows-msvc
#   EXIT=0
file .../probe.dll
#   PE32+ executable (DLL) (GUI) x86-64, for MS Windows
```

그다음 진짜 크레이트:

```sh
cd /Volumes/macMini/worktrees/bw-desk2/rust/torch_c
export PYO3_CROSS_LIB_DIR=/Volumes/macMini/caches/target-python/x86_64-pc-windows-msvc/libs
export PYO3_CROSS_PYTHON_VERSION=3.13
cargo xwin build --release --target x86_64-pc-windows-msvc
#   EXIT=0, Finished `release` profile in 40.80s
#   _C.dll   5,018,112 B   PE32+ executable (DLL) (GUI) x86-64
```

**`onig_sys` 가 여기서도 그냥 지나갔다.** Linux 에서 `cargo-zigbuild` 가 그랬듯,
`clang-cl` 셰임이 cc-rs 에게 타깃 C 드라이버로 보인다.

산출물 이름이 `lib_C.dll` 이 아니라 **`_C.dll`** 이다 — cargo 는 Windows 에서 cdylib 에 `lib`
접두사를 붙이지 않는다. `WindowsTarget.__init__` 이 그 이름을 쓴다.

---

## 4. `build.py --target windows-x86_64` — 넘었다. 다른 셋과 구조가 두 군데 다르다

### 4.1 태그는 유도되지 않는다. 그것이 이 층에서 가장 중요한 사실이다

| 타깃 | 태그의 하한이 어디서 오나 |
|---|---|
| Android | `_sysconfigdata` 의 `ANDROID_API_LEVEL` |
| iOS | `_sysconfigdata` 의 `IPHONEOS_DEPLOYMENT_TARGET` |
| Linux | **아티팩트**의 `.gnu.version_r` (`docs/platform/LINUX.md` §5.1) |
| **Windows** | **없다. 태그에 버전 성분 자체가 없다** |

Windows 휠 태그는 `win32` · `win_amd64` · `win_arm64` 셋뿐이고 OS 버전을 담지 않는다.
PE 헤더에 `MajorSubsystemVersion` 이 있지만 **어떤 인스톨러도 그것을 보지 않는다.**

그래서 `WindowsTarget.platform_tag()` 는 유도하지 않고 **`win_amd64` 를 그냥 반환한다.**
`packaging` 도 여기서는 도와주지 못한다 — `windows_platforms` 생성기가 없다
(Windows 에서 pip 는 실행 중 인터프리터의 `sysconfig.get_platform()` 을 쓰므로 인자로 만들 수 없다).
`manylinux` 와 같은 모양의 결손이고, 같은 방식으로 **시끄럽게 건너뛴다:**

```
! packaging 26.3 has no windows_platforms -- tag spelling unchecked
```

대신 `platform_tag()` 는 태그가 아니라 **파일**에 대해 확인할 수 있는 것을 확인한다 — §5.2 의
abi3 바인딩이다.

### 4.2 멤버 이름이 `torch/_C.pyd` 다

CPython 의 확장 검색 테이블은 플랫폼마다 다른 파일에 있다:

```
dynload_shlib.c (POSIX)   { SOABI 접미사, ".abi3.so", SHLIB_SUFFIX }
dynload_win.c   (Windows) { "_d.pyd", ".cp313-win_amd64.pyd", ".pyd" }
```

**Windows 표에는 `.abi3.so` 가 없다.** 그러니 다른 셋과 같은 이름으로 넣으면 그 파일은
**영원히 발견되지 않고**, 실패는 여기를 전혀 가리키지 않는 `ModuleNotFoundError` 로 나온다.
abi3 확장의 Windows 철자는 접미사 없는 `.pyd` 다 (`.cp313-win_amd64.pyd` 는 버전 고정용).

`Target.extension_member` 를 새로 두고 기본값을 `torch/_C.abi3.so` 로, `WindowsTarget` 만
`torch/_C.pyd` 로 두었다. `_repack()` 에 `renames` 인자가 생긴 것이 이것 때문이고,
**기존 세 타깃은 기본값을 쓰므로 동작이 바뀌지 않는다.**

`verify()` 는 두 가지를 본다: 그 이름으로 존재하는가, 그리고 **두 이름으로 동시에 존재하지 않는가.**
후자는 리네임이 절반만 적용된 상태이고, 그러면 인터프리터가 표 순서대로 엉뚱한 쪽을 먼저 찾는다.

### 4.3 global-deps 라이브러리를 **싣지 않는다**

`torch/__init__.py` 를 읽고 결정했다. 두 줄이 전부다:

```python
def _load_global_deps() -> None:
    if platform.system() == "Windows":
        return                      # <-- 아무것도 하지 않는다
```

```python
if sys.platform == "win32":
    def _load_dll_libraries() -> None:
        ...
        dlls = glob.glob(os.path.join(th_dll_path, "*.dll"))   # th_dll_path = torch/lib
        for dll in dlls:
            res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
            ...
            if res is None: raise err                          # <-- 삼키지 않는다
```

즉 Windows 에서 `libtorch_global_deps` 는 **찾지도 않고**, 반대로 `torch/lib/` 에 놓인 DLL 은
전부 `LoadLibrary` 된다. 빈 DLL 을 넣으면 **아무 목적 없이 로드되고, 로드에 실패하면 예외가
그대로 올라온다.** 그래서 `Target.global_deps_name` 에 `None` 을 허용하고 Windows 에서 그렇게 두었다.

**"만들 수 없어서 안 넣었다" 가 아니라 "넣을 것이 없다" 이고, 그 차이가 출력에 적힌다:**

```
+ no torch/lib/ global-deps library: _load_global_deps() returns immediately on
    Windows, and _load_dll_libraries() would LoadLibrary an empty one for nothing
```

`torch/bin/torch_shm_manager`(VENDOR.md wall 4) 도 같은 이유로 Windows 에서는 요구하지 않는다 —
`_manager_path()` 가 Windows 에서 확인 전에 `b""` 를 반환한다. 다만 **지금 휠에는 들어 있다**
(벤더 트리에 있으므로). 있어서 해로울 것은 없고, `verify_cross.py` 가 그것을 *요구*하지 않을 뿐이다.

### 4.4 실측

```sh
$BPY tools/wheel/build.py --target windows-x86_64          # EXIT=0
```

```
target windows-x86_64: _C.dll (5,018,112 B)
      PE32+ x86_64 dll
  tag is a fixed name, not a derivation -- Windows wheel tags carry no OS version
      floor for either the interpreter or the artefact to supply
  imports 119 names from python3.dll (the abi3 forwarder), so the
      abi3 tag is about the file and not only about the build flags
  ! packaging 26.3 has no windows_platforms -- tag spelling unchecked
  member: torch/_C.abi3.so -> torch/_C.pyd (this interpreter's dynload table
      has no .abi3.so in it)
  + no torch/lib/ global-deps library: ...
  retag: macosx_11_0_universal2 -> win_amd64

dist/torchnative-0.0.2a0-cp313-abi3-win_amd64.whl
  2,686 entries, 13.8 MB compressed, 58.4 MB installed
```

Linux 휠보다 entry 가 하나 적다 — global-deps 가 없기 때문이다.

---

## 5. 심볼 해결 검증 — **iOS 만큼 강하다**

`tools/wheel/verify_windows.py`. `docs/platform/LINUX.md` §6.1 이 ELF 에 대해 "iOS 만큼 강하지 않다" 고
적었고 그것은 맞았다. **PE 는 그 제약을 받지 않는다.**

### 5.1 왜 강한가 — import table 이 곧 답이다

세 형식이 "이 미정의 심볼은 어디서 오는가" 에 답하는 방식:

| | 미정의 심볼의 라이브러리 귀속 |
|---|---|
| Mach-O (iOS) | 2단계 네임스페이스. 링커가 심볼마다 dylib 을 적어둔다. **전부 답한다** |
| ELF (Linux) | 평면 검색. **버저닝된 것만** 답한다(glibc). CPython 심볼은 아무것도 안 답한다 |
| **PE (Windows)** | **import 는 "찾을 이름" 이 아니라 DLL 로 묶인 테이블의 항목이다. 귀속 안 된 import 가 존재할 수 없다** |

Linux 에서 우리 아티팩트의 CPython import 118개는 **어느 라이브러리에서 오는지 파일이 말하지
않아** libpython 에 대한 합집합 검사밖에 못 했다 (`docs/platform/LINUX.md` §6.4). Windows 에서 같은
119개는 **파일이 `python3.dll` 이라고 적어 두었다.** 다른 DLL 에 우연히 같은 이름이 있어도
빠진 것을 가려줄 수 없다.

### 5.2 어디까지 실제로 돌았나 — 우리 아티팩트로

```sh
$BPY tools/wheel/verify_windows.py dist/torchnative-*win_amd64.whl     # EXIT=0
```

```
python3.dll: 902 exported symbols | vcruntime140.dll: 71 | vcruntime140_1.dll: 3

  torch/_C.pyd: PE32+ x86_64 dll
    235 imports from 11 DLL(s), every one attributed by the import table
      119  -> python3.dll        resolved against .../python3.dll
        8  -> VCRUNTIME140.dll   resolved against .../vcruntime140.dll
       60  -- kernel32.dll       a Windows component; attributed but not resolvable here
       19  -- api-ms-win-crt-math-l1-1-0.dll
       10  -- api-ms-win-crt-runtime-l1-1-0.dll
        5  -- api-ms-win-crt-stdio-l1-1-0.dll
        4  -- api-ms-win-crt-heap-l1-1-0.dll
        3  -- api-ms-win-crt-string-l1-1-0.dll
        3  -- api-ms-win-core-synch-l1-2-0.dll
        3  -- ntdll.dll
        1  -- bcryptprimitives.dll
```

**127 / 235 가 실제로 해결되었고, 나머지 108 은 전부 Windows 자체 구성요소다.**
`python3.dll` 119개가 하나도 빠짐없이 배포본의 export 902개 안에 있고, `VCRUNTIME140` 8개도
배포본이 싣는 `vcruntime140.dll` · `vcruntime140_1.dll` 에 다 있다.

**abi3 주장이 파일로 확인된다.** `python313.dll` import 는 0 이다. 만약 `python313.lib` 를 링크했다면
여기에 나타났을 것이고, 그 휠은 `abi3` 태그를 달고 3.13 하나만 서비스했을 것이다.
**빌드 플래그가 아니라 아티팩트에 대한 확인이다.**

### 5.3 파서를 `objdump` 와 대조했다

`binfmt.py` 에 `pe_info` · `pe_imports` · `pe_exports` 를 새로 넣었으므로,
독립적인 구현과 맞춰 봤다 (`/usr/bin/objdump -p`, LLVM):

```
   objdump   ours   DLL
       119    119   python3.dll
        60     60   kernel32.dll
        19     19   api-ms-win-crt-math-l1-1-0.dll
        10     10   api-ms-win-crt-runtime-l1-1-0.dll
         8      8   VCRUNTIME140.dll
         5      5   api-ms-win-crt-stdio-l1-1-0.dll
         4      4   api-ms-win-crt-heap-l1-1-0.dll
         3      3   api-ms-win-core-synch-l1-2-0.dll
         3      3   api-ms-win-crt-string-l1-1-0.dll
         3      3   ntdll.dll
         1      1   bcryptprimitives.dll
```

11개 DLL 전부, 심볼 수까지 일치한다.

### 5.4 자체검사 — 실패할 수 있는지 확인했다

```sh
$BPY tools/wheel/verify_windows.py --self-test        # EXIT=0
  ok    _socket.pyd: all 181 imports name a DLL
  ok    our torch/_C.pyd attributes 119 imports to python3.dll, by name
  ok    ...and none to python313.dll, which would make the abi3 tag false
  ok    an import attributed to a DLL nothing provides is refused, naming it (119 symbols)
  ok    python313.dll is not offered as a resolution target
SELF-TEST: PASS -- 5/5 cases, on real Windows PE from the target distribution
```

**네 번째가 이 검증이 합집합 검사가 아님을 증명한다.** 우리 `_C.pyd` 의 import descriptor 에서
`python3.dll` 이라는 **문자열만** 같은 길이의 `pythonX.dll` 로 바꾼다. 심볼 이름은 하나도 안 바뀌고,
그 119개는 여전히 전부 `python3.dll` 이 export 하는 것들이다. **"이 심볼이 내가 가진 어딘가에
있는가" 를 묻는 검사라면 그대로 통과한다.** 여기서는 119개가 전부 거절된다.

다섯 번째는 그 함정을 막아 둔 것이다: `python313.dll` 을 해결 대상에 넣었다면 잘못된 DLL 에
바인딩된 확장이 네 번째를 빠져나갔을 것이다.

### 5.5 이 검증이 말하지 않는 것

- **Windows 쪽 108개는 해결되지 않는다.** `kernel32.dll` · `ntdll.dll` · ucrt 포워더는 Windows
  구성요소이고 이 기계에 없다. 확인된 것은 *귀속*(어느 DLL 이 줘야 하는가)과 *그 DLL 집합에
  이상한 것이 없다*는 것까지다. Linux 의 glibc 칸과 같은 성격이되, **거기서는 귀속조차 절반이었다.**
- **ordinal import 는 이름을 확인할 수 없다.** 우리 아티팩트에는 0개지만, 있으면 따로 센다.
- **`import torch` 를 돌리지 않았다.** `_load_dll_libraries()` 는 `os.add_dll_directory`,
  `LoadLibraryExW`, `vcruntime140.dll` CDLL 을 import 시점에 하는데 그중 무엇도 실행되지 않았다.

---

## 6. `verify_cross.py` 도 Windows 를 안다

`WindowsExpectation` 을 넣었다. `LinuxExpectation` 과 같은 자리(태그 · 아키텍처 · RECORD ·
확장자 검색 가능성 · 파일 목록)를 채우되, 계열별로 다른 세 가지가 있다:

- **`packaging` 확인 불가.** manylinux 와 같고, 같은 방식으로 출력에 남긴다.
- **확장자 검사가 `.pyd` 를 본다.** `python313.dll` 안에 `.pyd\0` 문자열이 있는지 확인한다
  (`python3.dll` 이 아니다 — 포워더에는 코드가 없다). 이 검사는 이제 `exp.extension_member` 에서
  접미사를 유도하므로 계열마다 따로 쓰지 않는다.
- **global-deps 의 결함 모드가 뒤집힌다.** 다른 셋에서는 *없는 것*이 결함이고, Windows 에서는
  *있는 것*이 결함이다 (§4.3).

자체검사 8개 전부 잡힌다:

```
$BPY tools/wheel/verify_cross.py dist/torchnative-*win_amd64.whl --self-test   # EXIT=0
  caught      extension built for the wrong platform
  caught      a global-deps library on a platform that loads none
  caught      a member edited without updating RECORD
  caught      part of the vendored tree dropped
  caught      extension under the POSIX name this platform does not search
  caught      WHEEL Tag: out of step with the filename
  caught      platform tag no installer would generate
  caught      abi tag downgraded from abi3
SELF-TEST: PASS -- 8/8 fault modes rejected
```

Linux 는 11개다 (global-deps 결함 2개 + wall-4 마커 + glibc 하한이 더 있다).

### 6.1 여기서도 "실패할 수 없는 검사" 를 하나 더 찾았다

`_wrong_elf_machine()` 이 `e_machine := EM_X86_64` **대입**이라 x86-64 휠에는 무연산이었다.
`docs/platform/LINUX.md` §9.7 에서 뒤집기로 고쳤는데, PE 를 붙이면서 같은 함정이 반복될 수 있어
`_wrong_pe_machine()` 도 처음부터 **뒤집기**(amd64 ↔ arm64)로 썼다.

---

## 7. 사다리

```
built              됨     -- cargo-xwin + 셰임 넷. _C.dll 5,018,112 B, PE32+ x86-64 (§3.4)
tagged             됨     -- win_amd64. 유도가 아니라 고정 이름이다 (§4.1)
symbols resolve    됨(강함) -- 235개 전부 DLL 에 귀속. 127개는 실제로 해결,
                            108개는 Windows 구성요소라 이 기계에서 대조 불가 (§5.2)
imports on Windows 불가   -- Windows 기계 필요
computes           불가   -- 같음
```

Linux 사다리(`docs/platform/LINUX.md` §7.3)와 비교하면 **세 번째 칸만 다르고 나머지는 같다.**
그리고 그 차이는 도구가 아니라 **파일 형식**에서 온다.

### 7.1 실행 검증을 하려면

| 수단 | 무엇이 열리나 |
|---|---|
| **Windows x86_64 기계 / VM** (실물, 클라우드 인스턴스, CI 러너) | `pip install` → `import torch` → 계산 |
| **Wine** | `LoadLibrary` 와 import 는 될 수 있으나 `kernel32`/`ntdll` 이 재구현이라 계산 결과는 몰라도 로드 실패는 Wine 탓일 수 있다 |

**어느 것도 설치하지 않았다.** 컨테이너 런타임과 같은 층위의 결정이다.

---

## 8. 설치 목록

`docs/platform/LINUX.md` §11 의 둘에 더해, Windows 를 위해 새로 설치한 것은 **하나뿐이다.**

| 설치한 것 | 버전 | 위치 | 명령 |
|---|---|---|---|
| `cargo-xwin` | 0.23.1 | `~/.cargo/bin/cargo-xwin` | `cargo install cargo-xwin` |

나머지는 **이미 있던 것을 이름만 붙여준 것**이다:

| 셰임 | 실제로 부르는 것 | 원래 있던 이유 |
|---|---|---|
| `clang-cl` | `/usr/bin/clang --driver-mode=cl` | Xcode |
| `lld-link` | rustup 의 `rust-lld` | rustup |
| `llvm-lib` · `llvm-rc` · `llvm-dlltool` · `llvm-ar` · `llvm-ranlib` | `python -m ziglang <sub>` | `docs/platform/LINUX.md` §9.1 |

부수적으로 생기는 것:

| 경로 | 무엇 | 지워도 되나 |
|---|---|---|
| `/Volumes/macMini/caches/msvc-shims` | 위 셰임 일곱 개 (셸 스크립트, 총 1 KB 미만) | 예 (§3.2 로 다시 만든다) |
| `/Volumes/macMini/caches/xwin` | `XWIN_CACHE_DIR`. Microsoft CRT + Windows SDK | 예 (다시 받는다) |
| `/Volumes/macMini/caches/cargo-target-desk2/x86_64-pc-windows-msvc/` | 크로스 빌드 산출물 | 예 |

**`XWIN_ACCEPT_LICENSE=1` 이 필요하다.** cargo-xwin 은 Microsoft 의 라이선스 동의를 묻고,
비대화형 실행에서는 이 변수가 없으면 멈춘다.

**설정 파일에 아무것도 박지 않았다.** `.cargo/config.toml` · `~/.zshrc` · 저장소의 어떤 파일에도
셰임 경로나 링커 경로가 들어가지 않는다. 배선은 §0 의 `export` 로만 이루어진다.

---

## 9. 회귀 — 기존 넷이 그대로인지

Windows 를 넣으면서 **공유 코드**를 건드렸다 (`Target.extension_member`,
`Target.global_deps_name = None`, `_repack(renames=…)`, `verify()`, `verify_cross.py` 의
파일 검사 세 개). 그래서 나머지 타깃을 전부 다시 돌렸다.

| 명령 | 결과 |
|---|---|
| `PYTHON=$PY sh rust/torch_c/pytests/run.sh` | **exit 0, 197** |
| `$PY tools/golden/compare.py` | **exit 0, 2811/2811 ops=119** |
| `$BPY tools/wheel/build.py --self-test` | **exit 0, 8/8 + LINUX 11/11** |
| `$BPY tools/wheel/build.py` (호스트) | **exit 0**, `macosx_11_0_arm64`, 2,687 entries |
| `$BPY tools/wheel/verify.py <호스트 휠>` | **exit 0**, 깨끗한 venv 에서 `aten.mm.default` 계산 |
| `$BPY tools/wheel/build.py --target linux-x86_64` | **exit 0**, `manylinux_2_17_x86_64` |
| `$BPY tools/wheel/verify_cross.py <manylinux>` | **exit 0** |
| `$BPY tools/wheel/verify_cross.py <manylinux> --self-test` | **exit 0, 11/11** |
| `$BPY tools/wheel/verify_linux.py <manylinux>` | **exit 0**, unresolved 0 |
| `$BPY tools/wheel/verify_linux.py --self-test` | **exit 0, 5/5** |
| `$BPY tools/wheel/build.py --target windows-x86_64` | **exit 0**, `win_amd64` |
| `$BPY tools/wheel/verify_cross.py <win_amd64>` | **exit 0** |
| `$BPY tools/wheel/verify_cross.py <win_amd64> --self-test` | **exit 0, 8/8** |
| `$BPY tools/wheel/verify_windows.py <win_amd64>` | **exit 0** |
| `$BPY tools/wheel/verify_windows.py --self-test` | **exit 0, 5/5** |

**Android · iOS 는 이 워크트리에서 휠까지 갈 수 없다** — 아티팩트가 없고 기기·시뮬레이터가
금지되어 있다. 그 둘에 대해 확인한 것은 코드 수준이다:

- `Target.extension_member` 와 `global_deps_name` 의 **기본값이 이전 동작 그대로**이고,
  `AndroidTarget` · `IOSTarget` 은 둘 다 재정의하지 않는다
- `_repack(renames=…)` 는 `renames` 가 비면 이전과 같은 코드 경로를 탄다
- `verify_cross.py` 의 `Expectation.parse` 는 android/ios 두 줄이 그대로다

**이것은 코드를 읽은 결과이지 실측이 아니다.** Android · iOS 아티팩트가 있는 트리에서
`verify_cross.py --self-test` 를 한 번 돌리는 것이 남아 있다 — `docs/platform/LINUX.md` §10 과 같은 항목이다.
