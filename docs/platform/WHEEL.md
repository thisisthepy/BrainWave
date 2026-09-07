# 실제로 설치되는 휠 — 호스트 플랫폼 휠을 처음 만든 기록

PyPI 의 `torchnative 0.0.1a0` 은 **껍데기**입니다. 열어 보면 `torchnative/` 파이썬 스켈레톤뿐이고
`_C` 확장도, 벤더링된 `torch/` 트리도 없습니다. 태그는 `py3-none-any` 라서 어느 기계에나 설치되고,
설치된 다음 `import torch` 가 `ModuleNotFoundError` 로 죽습니다.

이 문서는 그것을 대체할 **진짜 플랫폼 휠**을 만들고, 깨끗한 가상환경에 설치해 동작을 확인한
기록입니다.

**판정은 하나입니다 — 저장소를 한 번도 본 적 없는 인터프리터가 `pip install` 한 뒤 `import torch`
하고 연산이 돌아야 하고, 그때 `torch.__file__` 이 그 가상환경 안을 가리켜야 합니다.** 개발 트리를
가리키면 휠이 아니라 소스를 쓴 것이고 아무것도 증명하지 못합니다.

---

## 0. 한눈에

| | 이전 (`0.0.1a0` on PyPI) | 지금 |
|---|---|---|
| 태그 | `py3-none-any` 하나 | **플랫폼 휠 4 개** (아래) |
| `_C` 확장 | 없음 | **`torch/_C.abi3.so`**, 타깃별로 진짜 그 플랫폼의 바이너리 |
| 벤더링 트리 | 없음 | **`torch` · `torchgen` · `functorch`**, `.py` 2372 개 |
| 파일 수 | 8 | **2,686** (네 휠 모두 같음) |
| `pip install` → `import torch` | **ImportError** | **동작** (§2) |
| `torch.ops.aten.mm.default` | 도달 불가 | **동작**, 값 일치 |
| `nn.Linear` 순전파 | 도달 불가 | **동작** |
| `importlib.metadata.version("torch")` | 없음 | **`2.13.0`** |
| CPython 3.14.7 | — | **같은 휠로 동작** (§2.3) |
| `twine check` | — | **4/4 PASSED** |

| 휠 | 압축 | 설치 후 | `_C` | 어디까지 확인됐나 |
|---|---|---|---|---|
| `cp313-abi3-macosx_11_0_arm64` | 13,287,413 B | 56.9 MB | 3,509,008 B | **계산됨** — 깨끗한 venv (§2) |
| `cp313-abi3-android_21_arm64_v8a` | 13,623,651 B | 58.2 MB | 4,821,448 B | **계산됨** — 에뮬레이터 (§7.3) |
| `cp313-abi3-ios_12_0_arm64_iphoneos` | 13,313,090 B | 57.1 MB | 3,618,000 B | **링크 해결까지** — 미해결 심볼 0/222 (§7.4.1). 로드·임포트·계산은 실기 필요 |
| `cp313-abi3-ios_14_0_arm64_iphonesimulator` | 13,270,683 B | 56.9 MB | 3,476,720 B | **계산됨** — 시뮬레이터 (§7.4, `docs/platform/IOS.md`) |

*(이 표의 바이트 수는 2026-08-28 회차의 것입니다. 2026-08-29 재빌드에서는 엔트리 2,687 개,
휠 13.5 MB, iOS 실기 `_C` 4,160,720 B 로 커졌습니다 — 그 사이 `rust/torch_c` 에 착지한 것들
때문입니다. 네 번째 휠(안드로이드)을 이 회차에 다시 만들지 않아 표 전체를 갱신하지 않았습니다.)*

기존 검증은 그대로입니다 — shim 테스트 **168/168**, 골든 하네스 **2702/2702, ops=118**.
둘 다 이 작업 전후로 같은 값이고, 종료 코드로 판정했습니다.

**아직 안 되는 것**: iOS **실기** 휠이 로드·임포트·계산되는지는 **측정하지 않았습니다**
(§7.0). 시뮬레이터 쪽은 채워졌고(§7.4), 실기 쪽은 **링크가 풀리는 것까지** 확인됐습니다
(§7.4.1) — 그 위는 기기가 있어야 합니다. 안드로이드는 `_multiprocessing` 스텁이 여전히
필요한데, 휠이 아니라 안드로이드 CPython 배포본의 성질입니다 (§7.3.2).

---

## 1. 만드는 법

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-wheel   # 선택

bash vendor/vendor_torch.sh      # 상류 파이썬 트리 3 개 패키지를 가져온다
bash vendor/install_shim.sh      # _C 를 빌드해 트리의 구멍에 넣는다
python tools/wheel/build.py      # 휠을 만든다  -> dist/*.whl
python tools/wheel/verify.py dist/torchnative-*macosx*.whl   # 진짜 되는지 본다
```

안드로이드 · iOS 휠은 `--target` 을 줍니다. 배선과 판정은 §7 에 있습니다.

빌드 인터프리터에는 `pip` · `setuptools` · `wheel` 이 필요합니다 (`build` 는 필요 없습니다 —
`tools/wheel/build.py` 가 `pip wheel --no-build-isolation` 으로 몰아넣습니다). 이 기록은
`/Volumes/macMini/caches/wheel-build-venv` (CPython 3.13.0, setuptools 84.0.0, wheel 0.48.0)
에서 만들었습니다.

**빈 체크아웃에서 위 네 줄이 그대로 돕니다.** 벤더링 트리를 통째로 지우고 (`git checkout` 으로
추적 파일 2 개만 복구) 처음부터 다시 돌려 같은 휠을 얻는 것으로 확인했습니다.

### `tools/wheel/build.py` 가 `pip wheel .` 보다 더 하는 것

| | 왜 |
|---|---|
| **preflight** | 벤더링 트리와 `_C` 가 없으면 **빌드를 거부**한다. 둘 다 `.gitignore` 라서 새 클론에는 없고, 그 상태로 setuptools 를 돌리면 **PyPI 에 있는 그 껍데기가 다시 나온다**. 조용히 실패하면 안 되는 지점이다. `--target` 을 줘도 이 검사는 **그대로 돕니다** — 크로스 산출물은 그 위에 얹는 *추가* 요구사항이지 대체가 아니다. **`build/` 밑에 이전 실행이 남긴 setuptools 캐시가 있으면 그것도 거부한다** — `build_py` 는 더 이상 요청받지 않는 파일을 캐시에서 지우지 않으므로, 거기 우연히 들어간 파일(Finder `.DS_Store`, 낡은 `.pyc`, 이전 타깃의 `.so`)은 그 캐시가 남아 있는 한 이후 모든 휠에 조용히 다시 복사된다. 2026-08-30 에 그렇게 여섯 휠 전부가 `.DS_Store` 를 실었다. 자동으로 지우지 않고 거부만 한다 — 무엇을 지울지는 사람이 정한다. `rm -rf build/` 통짜가 아니라 캐시 안의 각 항목을 이름으로 대는 좁은 명령을 낸다 |
| **신선도** | 담을 `_C` 가 **지금 디스크에 있는 소스로 빌드된 것인지** 확인하고, 아니면 이름을 대고 거부한다. 호스트와 크로스 양쪽 다. `preflight` 가 *존재*만 묻던 자리인데, 5 일 묵은 산출물도 그 질문에는 똑같이 답한다 — 실제로 그렇게 통과한 적이 있다 (§11) |
| **global-deps 스텁** | `torch/lib/libtorch_global_deps.{dylib,so}` 를 빈 라이브러리로 만들어 넣는다. **타깃의 컴파일러로** 만들고, 만든 것이 정말 그 플랫폼인지 확인한다 (§3.2 · §7.4) |
| **retag** | 호스트: `universal2` → `arm64`, setuptools 는 확장의 실제 아키텍처를 보지 않는다 (§3.3). 크로스: PEP 738/730 태그를 타깃 CPython 에서 유도한다 (§7.2) |
| **install name** | cargo 가 박아 넣은 빌드 머신 절대경로를 `@rpath/_C.abi3.so` 로 바꾼다 (§3.4). 이미지 형식을 보고 판단하므로 ELF(안드로이드)에는 걸지 않는다 |
| **상류 dist-info 주입** | `importlib.metadata.version("torch")` 가 답하게 한다 (§3.5) |
| **verify** | 아카이브를 소스 트리와 **파일 단위로 대조**하고 빠진 것이 있으면 실패한다. 작기만 한 휠은 설치는 되고 나중에 아무 임포트에서나 죽는다. **그리고 그 반대 방향도 본다 — 소스 트리에도, 이 빌드가 스스로 만들어 넣은 것(global-deps 스텁, 상류 dist-info)에도 없는 멤버가 하나라도 있으면 이름을 대고 거부한다.** 이름 허용 목록이 아니라 "이 빌드가 실제로 넣으려 한 것과의 대조" 인 이유는 벤더링 트리 하나가 수천 개의 정상 파일을 담기 때문이고, 예외는 이 휠 자신의 `<name>-<version>.dist-info/` 뿐이다 — setuptools 가 쓰는 그 디렉터리의 멤버 목록은 이 스크립트가 정하는 것이 아니라서, 이름이 아니라 접두사로만 빠져나간다. 이 방향이 없어서 2026-08-30 에 `.DS_Store` 가 여섯 휠 모두를 통과했다. 크로스면 완성된 아카이브 안의 바이너리를 다시 열어 플랫폼을 확인한다 |

---

## 2. 실제로 설치해서 동작하는 것

### 2.1 판정 출력

`tools/wheel/verify.py` 는 새 venv 를 만들고, 휠을 **의존성까지 함께** 설치하고, 저장소 **밖**
디렉터리에서 `-I`(PYTHONPATH·user site 무시)로 프로브를 돌립니다.

```
+ /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 -m venv /tmp/wheeltest
+ pip install torchnative-0.0.1a0-cp313-abi3-macosx_11_0_arm64.whl
  pulled in: filelock 3.32.4, fsspec 2026.7.0, Jinja2 3.1.6, MarkupSafe 3.0.3, mpmath 1.3.0,
             networkx 3.6.1, setuptools 84.0.0, sympy 1.14.0, torch 2.13.0,
             typing_extensions 4.16.0
+ import torch

  py                       3.13.0
  prefix                   /private/tmp/wheeltest
  torch_version            2.13.0
  torch_file               /private/tmp/wheeltest/lib/python3.13/site-packages/torch/__init__.py
  C_file                   /private/tmp/wheeltest/lib/python3.13/site-packages/torch/_C.abi3.so
  C_names                  1242
  aten_ops                 896
  metadata_version_torch   2.13.0
  torchnative              /private/tmp/wheeltest/lib/python3.13/site-packages/torchnative/__init__.py
  aten.mm.default          [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]  (torch.float32)
  x + x                    [2.0, 2.0, 2.0, 2.0]

PASS -- torchnative-0.0.1a0-cp313-abi3-macosx_11_0_arm64.whl installs into a clean venv and its torch computes
```

**`torch.__file__` 도 `torch._C.__file__` 도 `/private/tmp/wheeltest/` 안입니다.** 이것이 이
작업의 증명입니다. `torch_version` 이 `2.13.0` 인 것은 벤더링한 상류 트리의 버전이고,
`C_file` 이 `_C.abi3.so` 인 것은 그 트리가 상류 `_C` 가 아니라 우리 것을 쓰고 있다는 뜻입니다.

조금 더 얹은 것 (같은 venv):

```
$ ./bin/python -I -c "import torch, torch.nn as nn; m = nn.Linear(4,3); y = m(torch.ones(2,4)); print(tuple(y.shape), y.dtype)"
(2, 3) torch.float32
```

### 2.2 이 판정은 실패할 수 있다 — 확인함

"실패할 수 없는 검증은 검증이 아니다". `torch.__file__` 검사가 실제로 개발 트리를 잡아내는지
직접 확인했습니다. 같은 venv, 같은 명령, 차이는 `-I` 뿐입니다.

```sh
$ (cd /tmp/wheeltest && PYTHONPATH=<repo>/torchnative/src/main ./bin/python -I -c \
     "import torch; print(torch.__file__)")
/private/tmp/wheeltest/lib/python3.13/site-packages/torch/__init__.py     # 휠

$ (cd /tmp/wheeltest && PYTHONPATH=<repo>/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
     ./bin/python -c "import torch; print(torch.__file__)")
<repo>/torchnative/src/main/torch/__init__.py                             # 개발 트리
```

아래쪽이 `verify.py` 가 `FAIL` 로 잡는 상태이고, 실제로 개발 중에 한 번 잡혔습니다 — macOS 의
`/tmp` → `/private/tmp` 심볼릭 링크 때문에 접두사 비교가 어긋나 **통과한 실행을 실패로** 보고했고,
그래서 양쪽을 `realpath` 로 정규화했습니다.

### 2.3 abi3 는 말뿐이 아니다 — 3.14 에서 같은 휠이 돈다

`cp313-abi3` 은 "3.13 **이상** 전부" 라는 주장입니다. 그 주장을 그대로 시험했습니다.

```
$ /Users/ibrew/.local/bin/python3.14 -m venv /tmp/wheeltest314
$ /tmp/wheeltest314/bin/pip install <같은 휠>      # 재빌드 없음, 태그 그대로
$ /tmp/wheeltest314/bin/python -I -c "..."
py 3.14.7
torch 2.13.0 /private/tmp/wheeltest314/lib/python3.14/site-packages/torch/__init__.py
_C /private/tmp/wheeltest314/lib/python3.14/site-packages/torch/_C.abi3.so
mm [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]
add [2.0, 2.0, 2.0, 2.0]
```

**3.13 으로 빌드한 바이너리 하나가 3.14.7 에서 그대로 로드되고 계산합니다.** docs/design/ABI3.md 가
비용 대칭성으로 고른 선택인데, 여기서 처음으로 그 이득이 실물로 확인됐습니다 — 버전 고정이었다면
3.14 용 휠을 따로 빌드해야 했습니다.

---

## 3. 무엇을 고쳐야 했는가

### 3.1 상류 torch 는 패키지 **하나가 아니다** — 이것이 제일 컸다

`vendor_torch.sh` 는 `torch/` 만 복사하고 있었습니다. 상류 배포본의 `top_level.txt` 는 세 개를
말합니다.

```
functorch
torch
torchgen
```

**PYTHONPATH 워크플로에서는 이 결손이 보이지 않습니다.** `PYTHONPATH=$PWD/torchnative/src/main`
은 `torch` 에 대해서만 site-packages 를 가리고, `torchgen` 과 `functorch` 는 **그 밑의 상류
설치본으로 조용히 해소**됩니다. 즉 지금까지의 "`import torch` 완주" 는 배포본의 3 분의 2 를
참조 설치가 대주는 상태에서 측정된 것입니다.

상류 torch 가 없는 기계에 휠을 얹는 순간 드러납니다:

```
torch/__init__.py:2254  from torch import _VF, functional
  -> torch/nn/modules/module.py:17  from torch.utils._python_dispatch import ...
     -> torch/utils/_python_dispatch.py:13  import torchgen
ModuleNotFoundError: No module named 'torchgen'
```

`torch/__init__.py` 3087 행 중 2254 행에서 죽습니다. 고친 방법은 `vendor_torch.sh` 가
`top_level.txt` 를 **읽어서** 형제 패키지를 함께 벤더링하게 한 것입니다 — 하드코딩하지 않은 이유는
상류가 형제를 추가·삭제했을 때 같은 방식으로 다시 놓치지 않기 위해서입니다.

부수 효과로 `.stamp` 의 `native_left` 계산을 고쳤습니다. rsync 의 `--exclude '*.so'` 는
`--delete` 로부터도 그 파일을 **보호**하므로, `install_shim.sh` 를 돌린 트리 위에 다시 벤더링하면
우리 `_C.abi3.so` 가 세어져 `native_left=1` 이 나옵니다. 그 숫자는 VENDOR.md §2 에서
"상류 네이티브가 하나도 안 남았다" 는 뜻으로 읽히는 값이므로, 우리 것을 세지 않도록 했습니다.

### 3.2 `libtorch_global_deps` — 벽 1 은 휠에서는 우회로 못 넘는다

VENDOR.md 벽 1: `torch/__init__.py:_load_global_deps()` 가 무조건
`ctypes.CDLL(torch/lib/libtorch_global_deps.dylib, RTLD_GLOBAL)` 을 하는데,
`vendor_torch.sh` 는 `torch/lib/` 를 통째로 버립니다. 이 저장소의 모든 실행이
`TORCH_USE_RTLD_GLOBAL=1` 로 그 분기를 건너뜁니다.

**소스 트리에서는 그래도 됩니다. 휠에서는 안 됩니다.** `pip install torchnative` 한 사람에게
"이 환경변수를 설정하세요" 라고 할 수는 없습니다. 게다가 그 우회 분기는 공짜가 아닙니다 —
`from torch._C import *` 전체를 `sys.setdlopenflags(RTLD_GLOBAL | RTLD_LAZY)` 아래에서
수행하고, 그 기기 영향은 VENDOR.md §7 항목 4 가 **미확인**으로 남긴 것입니다.

그래서 **빈 공유 라이브러리를 하나 넣습니다.** 상류의 그 파일은 MKL/OpenMP 를 libtorch 보다 먼저
전역 네임스페이스에 올리려고 존재하는데, 이 빌드는 둘 다 링크하지 않습니다 (`_C.abi3.so` 는
Accelerate 에 대해 자기완결적입니다). 즉 **비어 있는 것이 스텁이 아니라 정확한 내용**입니다.
다만 `install_shim.sh` 가 `torch/bin/torch_shm_manager` 에 쓰는 0 바이트 마커와는 달리 이쪽은
실제로 `dlopen` 되므로 **로드 가능한 이미지**여야 합니다.

`install_shim.sh` 가 아니라 휠 빌드에서 만드는 이유: 소스 트리는 기존 문서가 기술하는 대로
그대로 두기 위해서입니다. 벽 1 은 거기서 여전히 서 있고, 환경변수를 설정하는 스위트들은
자기가 잰다고 말하는 것을 계속 잽니다.

**이 파일이 실제로 로드되고 있다는 것은 이후에 기기에서 확인됐습니다** — 에뮬레이터에서 이
파일만 치우면 `import torch` 가 `dlopen failed: ... libtorch_global_deps.so not found` 로 죽습니다
(§7.3.1). 처음 만들 때는 "있으면 우회 분기를 안 탄다" 는 논증뿐이었습니다.

#### 3.2.1 파일 이름과 배포 대상 — 나중에 드러난 두 가지

**이름은 `.dylib` 가 아닐 수 있습니다.** `_load_global_deps()` 는
`".dylib" if platform.system() == "Darwin" else ".so"` 로 이름을 만드는데, `platform.system()` 은
**안드로이드에서 `"Android"`, iOS 에서 `"iOS"`** 입니다. 즉 **iOS 도 `.so` 를 찾습니다** — 내용은
Mach-O dylib 인 채로. 이것을 틀리면 조용합니다: `CDLL` 이 던진 `OSError` 를
`_load_global_deps` 가 `_preload_cuda_deps` 경로로 삼켜서, 임포트는 전혀 관계없는 곳에서 죽습니다.

**호스트용도 배포 대상이 틀려 있었습니다** — 이번에 고쳤습니다. 호스트 `cc` 는 자기가 가진 SDK
로 스탬프를 찍으므로 `macosx_11_0_arm64` 로 태그된 휠 안에 `Mach-O arm64 macos 26.0+` 짜리
파일이 들어가 있었습니다. dyld 는 그 필드를 강제하므로 **태그가 약속한 macOS 대부분에서 로드할
수 없는 파일**입니다. §3.3 과 같은 종류의 불일치이고, 같은 방향입니다 — 남의 기계에서만 드러나는
쪽. 지금은 빌드 인터프리터의 `MACOSX_DEPLOYMENT_TARGET` 을 `-mmacosx-version-min` 으로 넘겨
`macos 11.0+` 로 맞춥니다.

(이 결함이 실제로 사용자를 깨뜨린 것은 확인하지 않았습니다 — macOS 11~25 인 기계가 없습니다.
고친 근거는 재현이 아니라 **아카이브 내용과 태그가 어긋나 있다**는 것 자체입니다.)

### 3.3 태그의 아키텍처가 거짓말이었다

setuptools 는 플랫폼 태그를 `sysconfig.get_platform()` 에서 가져온 뒤
`calculate_macosx_platform_tag` 로 아카이브를 훑는데, **그 훑기는 배포 대상 버전만 올리고
아키텍처는 건드리지 않습니다.** python.org 의 universal2 인터프리터로 빌드하면 안에 무엇이 있든
`macosx_11_0_universal2` 가 나옵니다. 우리 안에는 arm64 Mach-O 하나뿐입니다.

이것은 미관 문제가 아닙니다. `universal2` 는 인텔 맥의 pip 에게 "이건 네 것" 이라고 말하고,
설치는 성공하고, `import torch` 가 `dlopen` 의 아키텍처 불일치로 죽습니다 — 그러면 사용자의 증거는
이 태그가 아니라 자기 기계를 가리킵니다. 좁히면 그 사용자는 "no matching distribution" 을 받고,
그게 사실입니다.

`tools/wheel/build.py` 가 Mach-O 헤더(fat/thin, cputype)를 직접 읽어 `arm64` / `x86_64` /
`universal2` 를 정하고, 파일 이름과 `.dist-info/WHEEL` 의 `Tag:` 를 함께 고칩니다.

```
retag: macosx_11_0_universal2 -> macosx_11_0_arm64 (extension is arm64)
```

### 3.4 `_C.abi3.so` 가 빌드 머신 경로를 광고하고 있었다

```
$ otool -L torchnative/src/main/torch/_C.abi3.so
	/Volumes/macMini/caches/cargo-target-wheel/release/deps/lib_C.dylib   # LC_ID_DYLIB
```

cargo 가 cdylib 을 `-install_name <CARGO_TARGET_DIR>/...` 로 링크합니다. **동작에는 영향이
없습니다** — 경로로 `dlopen` 되는 모듈은 `LC_ID_DYLIB` 을 보지 않고, 그래서 이걸 고치기 전에도
임포트는 됐습니다. 다만 배포되는 산출물이 빌드한 기계의 디렉터리를 적어 두는 것은 결함입니다.

휠 안의 사본에만 `install_name_tool -id @rpath/_C.abi3.so` 를 걸고 **애드혹 재서명**합니다
(arm64 Mach-O 는 서명이 *없는* 것은 괜찮아도 *깨진* 것은 로드가 거부됩니다). 소스 트리의 파일은
cargo 가 낸 그대로 두므로, 스위트가 초록인 것은 이 재작성이 아니라 그 산출물에 대한 증거입니다.

```
$ otool -L /tmp/wheeltest/lib/python3.13/site-packages/torch/_C.abi3.so
	@rpath/_C.abi3.so
```

### 3.5 상류 `torch-2.13.0.dist-info` 를 어디에 넣는가

`importlib.metadata.version("torch")` 는 transformers 의 `is_torch_available()` 이 걸려 있는
호출이므로 (VENDOR.md) 장식이 아닙니다. 그런데 아카이브 **루트**에 둘 수는 없습니다 — pip 는
최상위에서 `.dist-info` 로 끝나는 이름을 훑어 자기 메타데이터 디렉터리를 찾고, 둘 이상이면
`UnsupportedWheel: multiple .dist-info directories found` 로 거부합니다.

`torchnative-0.0.1a0.data/purelib/torch-2.13.0.dist-info/` 로 넣습니다. 설치 시점에
site-packages 로 옮겨지고, 루트에는 여전히 하나만 있습니다. 상류의 `RECORD` 는 복사하지
않습니다 (`vendor_torch.sh` 도 안 합니다) — 우리 `RECORD` 가 이 파일들을 들고 있어서
`pip uninstall torchnative` 가 정리하고, 자기 `RECORD` 를 주면 `pip uninstall torch` 가
트리를 통째로 지우도록 초대하는 셈이 됩니다.

실측한 결과:

```
$ pip list | grep -i torch
torch             2.13.0
torchnative       0.0.1a0

$ pip install --dry-run --no-deps torch
Requirement already satisfied: torch in ./lib/python3.13/site-packages (2.13.0)

$ pip uninstall -y torch
× Cannot uninstall torch 2.13.0
╰─> The package's contents are unknown: no RECORD file was found for torch.

$ pip uninstall -y torchnative
Successfully uninstalled torchnative-0.0.1a0
$ ls site-packages | grep -i torch     # 남는 것 없음
```

앞의 셋은 의도한 것입니다 — 상류 torch 로 덮어쓰이지 않고, `is_torch_available()` 이 참이 되고,
제거는 우리 배포본을 통해서만 일어납니다. `pip uninstall torch` 의 거절 메시지가 "uv 가 설치한 것"
이라고 잘못 추측하는 것은 다듬을 여지입니다 (§6).

### 3.6 벤더링 트리는 자기 의존성을 데려온다

`dependencies = []` 가 옳았던 이유는 "**torch** 를 요구하면 자기 자신을 요구하는 것" 이라서인데,
그 문장이 상류의 **다른** 의존성까지 덮는 것으로 읽혀 있었습니다. 첫 플랫폼 휠은 그래서 설치는
되고 `torch/__init__.py:35` 의 `from typing_extensions import ...` 에서 죽었습니다.

상류 `Requires-Dist` 중 순수 파이썬 7 개를 그대로 가져왔습니다: `filelock`,
`typing-extensions>=4.10.0`, `setuptools>=77.0.3`, `sympy>=1.13.3`, `networkx>=2.5.1`,
`jinja2`, `fsspec>=0.8.5`.

**CUDA · triton 항목은 가져오지 않았습니다.** 전부 `platform_system == "Linux"` 가드가 붙어
있고 이 빌드에는 CUDA 경로가 아예 없습니다. 그대로 두면 모든 리눅스 설치가 ~2 GB 의 nvidia 휠을
끌어옵니다 — 이 프로젝트의 대상인 aarch64 보드까지 포함해서.

---

## 4. 휠 안에 무엇이 있는가

```
torch/                     상류 2.13.0 파이썬 트리 + 우리 _C
  _C.abi3.so               3,185,936 B   arm64, abi3, install name @rpath
  _C/*.pyi                 상류 스텁 26 개. `_C` 확장 옆의 *디렉터리*이고
                           importlib 는 확장을 먼저 고른다 (상류도 같은 모양)
  lib/libtorch_global_deps.dylib   16,840 B   빈 것 (§3.2)
  bin/torch_shm_manager    0 B   존재만 검사되는 마커 (VENDOR.md 벽 4)
  nn/federated.py          우리 add-hook
torchgen/                  native_functions.yaml 등. import torch 가 요구한다
functorch/
torchnative/               API 스켈레톤
torchnative-0.0.1a0.data/purelib/torch-2.13.0.dist-info/    §3.5
torchnative-0.0.1a0.dist-info/
```

크기:

| | |
|---|---|
| 휠 (압축) | **13,175,324 B** = 12.6 MiB |
| 아카이브 엔트리 | 2,683 |
| 압축 해제 | 56.6 MB |
| 설치 후 (pip 가 `.pyc` 생성) | torch 110 MB + torchgen 3.5 MB + functorch 420 KB + torchnative 52 KB |

**PyPI 의 기본 파일 크기 한도는 100 MB 이고 여유가 큽니다.** 벤더링 트리가 수백 MB 가 될 것이라는
예상은 빗나갔는데, 이유는 `vendor_torch.sh` 가 버리는 것이 바로 무거운 것들이기 때문입니다 —
`torch/lib/` 353 MB, `torch/include/` 61 MB, `torch/bin/` 7 MB. 남는 것은 파이썬 소스이고
잘 압축됩니다.

---

## 5. 이 휠로 아직 안 되는 것 — `print(tensor)`

> **닫혔습니다 (2026-08-28). `docs/models/E2E_REAL.md` §2 를 보십시오.** 아래 측정은 그대로 두되,
> 이 절이 "최소 8개" 라고 적은 목록은 **거절한 이름의 목록이지 필요한 셈의 목록이 아니었습니다.**
> 상류의 `repr` 이 실제로 무슨 op 을 디스패치하는지 재 보니 없던 것은 커널 6개였고,
> 여기 3번으로 적힌 `torch.get_default_dtype` 은 그 사이에 이미 구현되었으며
> (`docs/distributed/DISTRIBUTED.md` §3.4), 반대로 이 목록에 없던 `abs`·`ceil`·`gt.Scalar`·`min`·
> `unbind.int`·`masked_select` 가 필요했습니다. 그 차이를 만든 것이 계측기의 위치입니다 —
> `_str` 은 `_disable_current_modes()` 로 시작하므로 `TorchDispatchMode` 를 **밖에** 걸면
> 아무것도 기록되지 않습니다.

작업 지시의 판정 스니펫은 이렇게 되어 있었습니다.

```python
print(torch.ops.aten.mm.default(x, torch.ones(4, 2)))
```

**연산은 됩니다. `print` 가 안 됩니다.**

```
File "torch/_tensor_str.py", line 409, in _str_intern
    if torch._C._functorch.is_functorch_wrapped_tensor(inp):
NotImplementedError: not implemented in torch._C shim:
    torch._C._functorch.is_functorch_wrapped_tensor
```

**이것은 휠의 결함이 아닙니다.** 개발 트리에서 `TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=...` 로
돌려도 같은 자리에서 같은 예외가 납니다. `_C` 표면의 구멍이고 포장과 무관합니다.

막고 있는 이름을 하나씩 스텁하며 `repr` 경로를 따라가 본 결과 **최소 8 개**입니다.

```
torch._C._functorch.is_functorch_wrapped_tensor
TensorBase.is_nested
torch.get_default_dtype                 (torch._C._get_default_dtype 자체가 없음)
TensorBase.is_sparse
TensorBase.layout
TensorBase.is_quantized
torch._is_functional_tensor
TensorBase._is_zerotensor
```

"최소" 인 이유: 이 열거는 막힌 이름을 `False` 로 갈아끼우고 다시 돌리는 거친 방법이라, 8 번째
뒤에서 스텁 자체의 타입 오류로 멈췄습니다. **8 은 하한이지 총계가 아닙니다.**

값을 꺼내는 것은 문제없습니다 — `.tolist()`, 산술, `nn.Linear` 순전파 전부 동작합니다. 그래서
`verify.py` 는 `repr` 대신 `.tolist()` 로 판정합니다. 막힌 것은 **텐서 서식화**입니다.

이 여덟 개는 거의 전부 "이 빌드에는 그런 것이 없다" 로 답이 정해져 있어 보이지만
(functorch 래핑 없음, 희소 텐서 없음, 양자화 없음), **그 판단은 `_C` 표면 작업의 몫이지 휠
작업의 몫이 아닙니다.** 여기서 고치지 않고 남겼습니다.

---

## 6. 아직 판단이 필요한 것

| | |
|---|---|
| **라이선스 메타데이터** | `pyproject.toml` 은 `license = "MIT"` 인데 휠의 대부분은 **BSD-3 인 상류 PyTorch 소스**다. 상류 LICENSE 와 third-party 라이선스들은 주입한 dist-info 안에 함께 실려 있지만, 배포본 메타데이터가 그것을 진술하지 않는다. **업로드 전에 정리해야 한다** |
| **`torch` 이름을 점유한다** | 설치되면 `pip list` 에 `torch 2.13.0` 이 뜨고 `pip install torch` 가 무시된다. 의도한 것이지만(pyproject 주석), 사용자가 상류 torch 를 원할 때 되돌리는 절차가 문서화돼 있지 않다. 지금은 `pip uninstall torchnative` 뒤 재설치가 유일한 경로다 |
| **`pip uninstall torch` 의 오진단** | 거절 메시지가 "uv 가 설치한 것 같다" 고 추측한다. RECORD 가 없기 때문인데, 우리가 일부러 안 넣은 것이므로 메시지가 사용자를 엉뚱한 곳으로 보낸다 |
| **sdist** | 만들지 않았다. 벤더링 트리가 git 에 없으므로 sdist 는 지금의 `py3-none-any` 와 같은 껍데기가 된다. `python -m build` 를 기본으로 쓰지 않는 이유이기도 하다 |

---

## 7. 호스트 밖 — Android · iOS

**이제 네 개의 휠이 있습니다.** 다만 넷이 같은 정도로 증명된 것은 아니고, 그 차이가 이 절의
전부입니다.

### 7.0 어디까지 확인됐는가

| | 빌드됨 | 심볼 해결됨 | 설치됨 | 임포트됨 | 계산됨 |
|---|---|---|---|---|---|
| `macosx_11_0_arm64` | ✅ | ✅ (설치돼서 돌았으므로) | ✅ 깨끗한 venv 에 `pip install` (§2) | ✅ | ✅ `aten.mm`, `nn.Linear` |
| `android_21_arm64_v8a` | ✅ | ✅ (설치돼서 돌았으므로) | ✅ **에뮬레이터의 site-packages 에 언팩** (§7.3) | ✅ | ✅ `aten.mm`, `nn.Linear` |
| `ios_12_0_arm64_iphoneos` | ✅ | ✅ **222/222 정적 대조** (§7.4.1) | ❌ 실기 없음 | ❌ | ❌ |
| `ios_14_0_arm64_iphonesimulator` | ✅ | ✅ | ✅ **시뮬레이터 CPython 의 site-packages 에 언팩** (§7.4) | ✅ | ✅ `aten.mm`, `nn.Linear` |

**"심볼 해결됨" 칸을 새로 나눴습니다** (2026-08-29). 앞의 셋에서는 이 칸이 뒤 칸들에 흡수돼
있었지만 — 임포트됐다면 심볼은 당연히 풀린 것이므로 — **실기 칸에서는 이것만 따로 답할 수
있고, 실제로 답이 나옵니다.** 나누지 않으면 그 답을 적을 자리가 없습니다.

**시뮬레이터 칸은 채워졌습니다** (2026-08-28). `tools/wheel/verify_ios_sim.py` 가 시뮬레이터
안에서 휠을 임포트하고 계산시키며, 값이 호스트와 정확히 일치합니다. 전체 기록은
**`docs/platform/IOS.md`** 에 있습니다.

**실기(`ios_12_0_arm64_iphoneos`)의 로드·임포트·계산은 여전히 측정하지 않았습니다.** 다만
"아티팩트가 맞다" 에서 멈춰 있던 것이 한 칸 나아갔습니다 — 기기 산출물이 링크하는
`Python.framework` 의 심볼 118 개와 SDK 시스템 라이브러리의 104 개, 합쳐 **222 개 전부가 각자
묶여 있는 그 라이브러리에서 실제로 export 된다**는 것을 확인했습니다 (§7.4.1,
`tools/wheel/verify_ios_device.py`).

**그래도 시뮬레이터 결과가 실기 칸을 채워 주지는 않습니다.** 둘은 Mach-O
`LC_BUILD_VERSION.platform` 이 7 과 2 로 다른 별개 아티팩트이고, 링크 방식마저 다릅니다 —
실기 쪽만 `Python.framework` 를 링크하고 시뮬레이터 쪽은 `-undefined dynamic_lookup` 으로
해소합니다(§7.1). 서로 다른 메커니즘이라 한쪽의 성공이 다른 쪽의 근거가 되지 않습니다
(`docs/platform/IOS.md` §7). **§7.4.1 이 하는 것은 시뮬레이터 결과를 실기로 옮기는 것이 아니라, 실기
쪽 메커니즘을 그 자체로 검사하는 것입니다.**

빈 칸을 채우는 방법은 §7.6 에 적어 두었습니다.

### 7.1 만드는 법

```sh
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-wheel2
export ANDROID_NDK_HOME=$HOME/Library/Android/sdk/ndk/27.1.12297006
TP=/Volumes/macMini/caches/target-python

# 1) _C 를 타깃별로 크로스 빌드한다
scripts/device_android.sh build                       # aarch64-linux-android

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
  PYO3_CROSS_LIB_DIR=$TP/arm64-iphoneos/lib \
  TORCHNATIVE_PYTHON_FRAMEWORK_DIR=$TP/arm64-iphoneos \
  cargo build --release --target aarch64-apple-ios )

( cd rust/torch_c && PYO3_CONFIG_FILE=/tmp/pyo3-ios.cfg \
  PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 \
  PYO3_CROSS_LIB_DIR=$TP/arm64-iphonesimulator/lib \
  cargo build --release --target aarch64-apple-ios-sim )

# 2) 휠을 만든다
python tools/wheel/build.py --target android-arm64-v8a
python tools/wheel/build.py --target ios-arm64
python tools/wheel/build.py --target ios-arm64-sim

# 3) 판정한다
python tools/wheel/verify_cross.py   dist/torchnative-*android*.whl
python tools/wheel/verify_cross.py   --self-test dist/torchnative-*android*.whl
ANDROID_SERIAL=emulator-5554 \
  python tools/wheel/verify_android.py dist/torchnative-*android*.whl
```

**시뮬레이터도 `PYO3_CONFIG_FILE` 이 필요합니다.** RUST_CROSSBUILD.md §0.5 는 그것을 실기
전용으로 적어 두었는데(`.cargo/config.toml` 주석과 `build.rs` 의 `is_ios_device` 도 같습니다),
실제로 돌려 보니 시뮬레이터 배포본에도 `libpython3.13.{a,dylib}` 이 없어 PyO3 의 `-lpython3.13`
이 그대로 실패합니다.

```
ld: warning: -undefined dynamic_lookup is deprecated on iOS-simulator
ld: library 'python3.13' not found
```

`suppress_build_script_link_lines=true` 만 추가하면 통과합니다 — 실기와 달리 `-framework Python`
은 필요 없고, `.cargo/config.toml` 이 이미 주는 `-undefined dynamic_lookup` 으로 해소됩니다.
그래서 시뮬레이터 산출물에는 `Python.framework` 의존이 **없고**, 그것이 정상입니다.
(`build.rs` 와 `.cargo/config.toml` 은 이 작업의 소유 범위 밖이라 손대지 않았습니다. 고칠 곳은
그 두 파일의 주석이지 배선이 아닙니다.)

### 7.2 태그 — 추측하지 않고 확인한 것

| | Android | iOS |
|---|---|---|
| PEP | 738 | 730 |
| 형태 | `android_<api>_<abi>` | `ios_<major>_<minor>_<multiarch>` |
| 실제 값 | `android_21_arm64_v8a` | `ios_12_0_arm64_iphoneos` · `ios_14_0_arm64_iphonesimulator` |
| 그 숫자의 출처 | 타깃 CPython 의 `ANDROID_API_LEVEL` | 타깃 CPython 의 `IPHONEOS_DEPLOYMENT_TARGET`, **단 산출물 쪽이 더 높으면 그쪽** |
| 숫자의 의미 | **하한**. `packaging` 은 기기의 레벨에서 16 까지 내려가며 태그를 만든다 | **하한**. 12.0 까지 내려간다 |

두 가지를 실제로 확인했습니다.

**하나. 형태를 명세가 아니라 `packaging` 에 물었습니다.** pip 가 "이 휠이 내 것인가" 를 판정할 때
쓰는 코드가 `packaging.tags.{android,ios}_platforms` 이므로, 거기서 나오는 목록에 우리 태그가
들어 있는지를 빌드가 검사합니다. 들어 있지 않으면 빌드가 실패합니다.

```
  tag android_21_arm64_v8a accepted by packaging.tags.android_platforms
  tag ios_12_0_arm64_iphoneos accepted by packaging.tags.ios_platforms
```

이 기계의 `packaging` 은 26.3 이고 두 생성기를 모두 가지고 있습니다. 없는 버전이면 검사를
**건너뛰었다고 출력**합니다 — 조용히 통과시키지 않습니다.

**둘. 시뮬레이터의 하한은 CPython 이 아니라 산출물이 정했습니다.** Rust 의
`aarch64-apple-ios-sim` 기본 배포 대상이 14.0 이라, CPython 의 12.0 을 그대로 쓰면 **자기 확장을
로드할 수 없는 OS 를 광고하는 휠**이 됩니다. `LC_BUILD_VERSION` 의 `minos` 를 읽어 더 높은 쪽을
택합니다.

```
  tag floor from the artefact (14.0), not from CPython (12.0)
```

실기 쪽은 반대로 산출물이 10.0(`LC_VERSION_MIN_IPHONEOS`)이라 CPython 의 12.0 이 이겼습니다.

### 7.3 Android — 기기에서 실제로 임포트하고 계산한다

`tools/wheel/verify_android.py` 가 하는 것은 §2 의 판정을 기기로 옮긴 것입니다. 판정 문장이
같습니다 — **`torch.__file__` 이 설치 위치 안을 가리켜야 한다.**

`PYTHONPATH` 를 쓰지 않는 것이 이전 측정과의 차이입니다. docs/devices/DEVICE.md 의 모든 안드로이드
측정은 `PYTHONPATH=$ROOT/site` 로 트리를 얹은 것이고, **`PYTHONPATH` 엔트리는 망가진 설치를
가려 줄 수 있는 바로 그것**입니다. 여기서는 휠을 기기 CPython 의 **site-packages** 에 풀고,
`-s -P` 로(유저 site 없음, 작업 디렉터리 없음) 돌립니다.

```
$ ANDROID_SERIAL=emulator-5554 python tools/wheel/verify_android.py \
      dist/torchnative-0.0.1a0-cp313-abi3-android_21_arm64_v8a.whl

  sys.path  ['/data/local/tmp/bw_wheel/lib/python313.zip',
             '/data/local/tmp/bw_wheel/lib/python3.13',
             '/data/local/tmp/bw_wheel/lib/python3.13/lib-dynload',
             '/data/local/tmp/bw_wheel/lib/python3.13/site-packages']

  torch_file   .../site-packages/torch/__init__.py
  C_file       .../site-packages/torch/_C.abi3.so
  C_names      1249   aten_ops 896
  aten.mm      [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]
  x + x        [2.0, 2.0, 2.0, 2.0]
  nn.Linear    [2, 3] torch.float32
  metadata     torch 2.13.0

PASS -- torchnative-0.0.1a0-cp313-abi3-android_21_arm64_v8a.whl unpacks into an
        Android CPython's site-packages and its torch computes on the device
```

`sys.path` 에 저장소로 이어지는 항목이 하나도 없습니다. 기기에는 개발 트리 자체가 없으므로
`torch.__file__` 검사는 §2 만큼 날카롭지는 않지만, **`PYTHONPATH` 를 쓰지 않는다는 것**이 그
자리를 대신합니다.

`pip install` 은 아닙니다 — 기기 CPython 용 pip 가 이 기계에 없고 기기에서 받게 하면 네트워크가
필요합니다. 대신 이 휠이 실제로 쓰는 PEP 427 설치 동작만 흉내 냅니다: 아카이브를 site-packages
에 풀고, `.data/purelib/` 를 그 위로 옮기고(그래서 `importlib.metadata.version("torch")` 가
`2.13.0` 을 답합니다), 휠의 `Requires-Dist` 가 이름 대는 순수 파이썬 배포본들을 옆에 놓습니다.
그 목록은 여기 적어 둔 것이 아니라 **휠의 METADATA 를 읽어서** 만듭니다.

#### 7.3.1 `TORCH_USE_RTLD_GLOBAL` 이 더 이상 필요 없다

docs/devices/DEVICE.md §4 는 이 변수를 안드로이드에서 **필수**로 기록했습니다. 그 측정은
`torch/lib/libtorch_global_deps.so` 가 **없는** 트리에 대한 것이었고, 휠은 그것을 싣습니다
(§3.2). 그래서 같은 기기에서 셋을 다 돌렸습니다.

| 실행 | 결과 |
|---|---|
| 스텁만, `TORCH_USE_RTLD_GLOBAL` 없이 | **성공** — 위 출력 |
| 스텁 + `TORCH_USE_RTLD_GLOBAL=1` | 성공, 같은 값 |
| 스텁, 변수 없이, **기기의 global-deps 파일을 치우고** | **실패** |

세 번째가 음성 대조입니다. 파일 이름만 바꾸고 같은 명령을 돌리면:

```
OSError: dlopen failed: library ".../site-packages/torch/lib/libtorch_global_deps.so" not found
  ctypes.CDLL(global_deps_lib_path, mode=ctypes.RTLD_GLOBAL)
```

**즉 그 파일은 장식이 아니라 실제로 로드되고 있고, 그것이 있어서 우회 변수가 필요 없어진
것입니다.** 이것으로 VENDOR.md 벽 1 이 기기에서도 닫혔습니다.

#### 7.3.2 아직 필요한 것 — `_multiprocessing` 스텁

`_multiprocessing` 스텁 없이 돌리면 여전히 죽습니다.

```
torch/multiprocessing/__init__.py:110  from multiprocessing.resource_tracker import ...
  -> lib/python3.13/multiprocessing/resource_tracker.py:41  import _multiprocessing
ModuleNotFoundError: No module named '_multiprocessing'
```

**이것은 휠의 결함이 아니라 안드로이드 CPython 배포본의 성질입니다** — 그 배포본은
`_multiprocessing` 도 `_posixshmem` 도 빌드하지 않습니다(안드로이드에 SysV IPC 가 없습니다).
`scripts/device_parity.py` 가 같은 이유로 같은 스텁을 답니다. `verify_android.py` 는 스텁 없이도
한 번 돌려서 그 실패를 **출력에 남깁니다** — 필요한 것을 배경에 숨기지 않기 위해서입니다.

이것을 진짜로 닫으려면 안드로이드 CPython 을 다시 빌드하거나 `torch/multiprocessing` 을
지연 임포트로 바꿔야 하고, 둘 다 휠 작업의 몫이 아닙니다.

### 7.4 iOS — 시뮬레이터는 계산까지, 실기는 아티팩트까지

**시뮬레이터는 더 이상 아티팩트 검사에 머물지 않습니다.** `tools/wheel/verify_ios_sim.py` 가
시뮬레이터 CPython 의 site-packages 에 휠을 풀고, 시뮬레이터 프로세스 안에서 `import torch` 를
시키고, `aten.mm`·`x+x`·`nn.Linear` 를 계산시킵니다. `dir(torch._C)` 1251 개와
`dir(torch.ops.aten)` 896 개까지 호스트와 정확히 같습니다.

iOS 배포본에는 실행 파일이 없어서(`Python.framework/Python` 은 `MH_DYLIB`) 하네스가
`Py_BytesMain` 3줄로 시뮬레이터용 `python3.13` 을 직접 컴파일합니다. 그 런처는 **UIKit 를
링크해야 합니다** — iOS 의 `platform.system()` 이 `UIDevice` 에 물어보는데, 맨
`simctl spawn` 프로세스에는 UIKit 가 없어 `torch/__init__.py:370` 에서 죽습니다. 전체 기록과
환경 변수 절제 측정은 **`docs/platform/IOS.md`** 에 있습니다.

**아래는 실기 쪽 이야기입니다.** 실기가 없으므로 **로드·임포트·계산은 측정하지 않았습니다.**
확인한 것은 파일 안에 무엇이 들어 있는가입니다.

```
torchnative-0.0.1a0-cp313-abi3-ios_12_0_arm64_iphoneos.whl
  binaries            2
    torch/_C.abi3.so                   Mach-O arm64 ios 10.0+
    torch/lib/libtorch_global_deps.so  Mach-O arm64 ios 12.0+
  ext suffix          .abi3.so present in Python
  file list           identical to torchnative-...-macosx_11_0_arm64.whl (2,684 entries)
```

각 줄이 무엇을 말하는지:

| | |
|---|---|
| `Mach-O arm64 ios` | `LC_BUILD_VERSION`/`LC_VERSION_MIN_*` 의 플랫폼 필드를 직접 읽은 것. **`ios`(2) 와 `iossimulator`(7) 은 이것 말고 구별되지 않습니다** — 크기도 아키텍처도 심볼도 같습니다 |
| `Python.framework` | 실기 산출물의 `LC_LOAD_DYLIB` 에 `@rpath/Python.framework/Python` 이 있습니다. 실기에는 폴백할 libpython 이 없으므로 이것이 없으면 로드 자체가 불가능합니다. 시뮬레이터 쪽에는 없고, 그것이 맞습니다(§7.1) |
| `.so` 인데 Mach-O | 파일 **이름**이 `.so` 여야 합니다. `_load_global_deps()` 가 `".dylib" if platform.system() == "Darwin" else ".so"` 로 이름을 만드는데 **iOS 에서 `platform.system()` 은 `"iOS"`** 입니다. 내용은 Mach-O dylib 이고 `dlopen` 은 확장자를 보지 않습니다 |
| `ext suffix` | `.abi3.so` 라는 문자열이 그 배포본의 `Python.framework/Python` 안에 실제로 있습니다. CPython 의 `_PyImport_DynLoadFiletab` 이 `{SOABI 접미사, ".abi3.so", SHLIB_SUFFIX}` 이고 그 상수들이 인터프리터에 컴파일되어 들어가므로, **기기 없이 이 질문에 답할 수 있는 가장 강한 형태**입니다. 안드로이드 쪽은 기기에서 직접 목록을 찍은 측정이 따로 있습니다(`scripts/device_android.sh` 주석) |
| `file list identical` | 트리가 하나도 빠지지 않았다는 것. 호스트 휠과 엔트리 집합이 정확히 같습니다 |

**"빌드된다" 는 판정이 아니라는 것을 호스트 휠에서 배웠고, 그 교훈은 여기에도 그대로
적용됩니다.** 위 표는 2026-08-28 에 "판정할 수단이 없다" 로 끝났는데, 그것이 사실이 아니었던
부분이 §7.4.1 입니다.

#### 7.4.1 실기 휠 — 실행하지 않고 확인할 수 있는 것 (2026-08-29)

`tools/wheel/verify_ios_device.py`. 전체 기록은 **`docs/platform/IOS.md` §11** 에 있고, 여기에는
판정만 적습니다.

**실행은 여전히 불가능하고, 그것을 주장이 아니라 dyld 의 말로 확인했습니다.** 기기 슬라이스를
시뮬레이터의 site-packages 에 넣으면 `incompatible platform (have 'iOS', need 'iOS-sim')`,
macOS 에서 `dlopen` 하면 `(have 'iOS', need 'macOS')` 입니다. 도구는 후자를 **매 실행 첫 줄에
찍습니다** — "심볼이 풀린다" 를 "돌아간다" 로 읽지 않게 하려고요.

**그런데 두 슬라이스의 실질적 차이는 심볼을 푸는 방식 하나이고, 그것은 정적으로 답이 나옵니다.**

```
torch/_C.abi3.so   222 undefined
  Accelerate     16  <- Accelerate.tbd  (iPhoneOS26.5.sdk)
  Python        118  <- Python  (the device CPython distribution)
  libSystem      88  <- libSystem.B.tbd  (iPhoneOS26.5.sdk)
  unresolved      0
```

**합집합이 아니라 라이브러리별로 묻습니다.** `nm -m` 은 2단계 네임스페이스 이미지의 각 미해결
심볼에 대해 **링커가 적어 둔 소속 라이브러리**를 알려 주므로, `_Py*` 심볼이 우연히 libSystem 에
있어도 통과해 버리는 합집합 검사와 달리 dyld 가 실제로 보는 곳에 있는지를 묻습니다.

그리고 **확장과 메타데이터를 빼면 두 iOS 휠은 바이트 단위로 같습니다** (공유 멤버 2,565 개, 차이
0). 즉 `verify_ios_sim.py` 가 시뮬레이터 안에서 임포트한 벤더링 파이썬 트리가 실기 휠 안의
그것과 같은 파일입니다. §7.4 가 말로 적어 둔 "파일 목록이 동일하다" 를 내용 비교로 바꾼 것입니다.

**어느 시뮬레이터 휠과 비교하는지가 그 비교의 전부입니다.** `find_sibling` 은 `dist/` 옆의
`*iphonesimulator*.whl` 을 **버전으로 짝짓습니다** — 2026-08-30 까지는 `sorted(glob(...))[0]`
이라서, `dist/` 에 여러 버전이 쌓여 있으면 실기 휠의 버전과 무관하게 **가장 오래된** 시뮬레이터
휠을 골랐습니다. 그날 0.0.4a0 실기 휠이 0.0.2a0 시뮬레이터 휠과 비교되고도 PASS 가 났는데, 그
사이 벤더링 트리가 안 바뀌어서 우연히 무해했을 뿐입니다. 지금은 버전이 안 맞으면 — 없어도, 여럿이
맞아도 — **거부**하고 이유를 댑니다; `--against` 로 명시할 수 있습니다.

**과장하지 않기 위해 도구가 사다리를 찍습니다.**

```
[yes] built · [yes] symbols resolved · [yes] same tree as the simulator wheel
[NO ] installed · [NO ] imported · [NO ] computed
```

남는 것은 **런타임 로드**(dyld 가 실행 시점에 `@rpath/Python.framework/Python` 을 *찾는가* —
앱에서는 `@executable_path/Frameworks` 로 해석됩니다), **코드 서명**, **`import torch` 완주**,
**계산과 성능**입니다. 전부 실기가 있어야 합니다.

도구는 여덟 가지로 스스로를 망가뜨려 보고 각각이 **올바른 종류의 답**으로 깨지는지 확인합니다
(`--self-test`, 8/8). 특히 **"휠에 대한 발견"(`FAIL:`)과 "검사가 못 봄"(`CANNOT JUDGE:`)을
섞지 않습니다** — 프레임워크를 엉뚱한 Mach-O 로 바꾸면 118 개 미해결로 `FAIL`, 프레임워크를
아예 치우면 `CANNOT JUDGE` 입니다.

**그 구분이 사다리의 각 칸에도 따로 적용됩니다 (2026-08-31 정정).** 사다리를 찍는 `ladder()`
는 한동안 "symbols resolved" 칸을 `symbols_ok and not findings.blind` 로 계산했는데, 여기
`findings.blind` 는 심볼 검사와 형제 비교(시뮬레이터 휠과의 파일 비교) 가 **함께 쓰는 목록**
이었습니다. 그래서 형제 비교 쪽이 눈멀면(예: 짝지을 시뮬레이터 휠이 없으면) 심볼과 아무 상관
없는 그 눈멂이 "symbols resolved" 칸을 `[NO]` 로 끌어내렸습니다. 실기 휠로 재현: 222 개 심볼이
전부 자기 라이브러리에서 풀려도(0 unresolved), 형제 비교만 눈멀게 하면 그 칸이 여전히 `[NO]`
였습니다. 지금은 심볼 검사와 형제 비교가 각자의 `Findings` 를 씁니다 — 사다리의 각 칸이 자기
자신의 증거로만 판정되고, 다른 칸의 눈멂이 새어 들어올 공유 목록 자체가 없습니다.

또 하나 미결로 남는 것: iOS 앱은 App Store 규칙상 바이너리 모듈을 `.framework` 로 담아야 하고,
PEP 730 휠 안의 `.so` 를 프레임워크로 바꾸는 것은 **앱 패키징 도구의 일**입니다(briefcase 가
그렇게 합니다). 이 휠은 규격대로 `.so` 를 싣고, 그 변환이 이 저장소 안에서 일어나는지는
아직 아무도 정하지 않았습니다.

### 7.5 검증이 실제로 실패하는지 확인했다

"실패할 수 없는 검증은 검증이 아니다". `verify_cross.py --self-test` 는 좋은 휠을 여러 가지로
망가뜨린 뒤, 각각이 **올바른 이유로** 거절되는지 봅니다. 종료 코드만 보지 않고 보고 문구를
맞춰 보는 이유는, 파일 목록 비교가 나머지를 전부 흡수해 버리기 때문입니다.

```
SELF-TEST against torchnative-0.0.4a0-cp313-abi3-android_21_arm64_v8a.whl
  caught      extension built for the wrong platform
  caught      global-deps library missing
  caught      global-deps library under the host's name
  caught      a member edited without updating RECORD
  caught      part of the vendored tree dropped
  caught      extension missing
  caught      WHEEL Tag: out of step with the filename
  caught      wall-4 marker missing
  caught      platform tag no installer would generate
  caught      abi tag downgraded from abi3
  caught      _default_reference picks the SAME-VERSION macosx wheel beside a
              wheel, not whichever version sorts last

SELF-TEST: PASS -- 11/11 fault modes rejected
```

**공통 고장 모드는 아홉이지만, 세 크로스 휠이 같은 개수는 아닙니다** — manylinux 는 §2.6 함정을
잡는 태그-바닥 케이스가 하나 더 있어 12/12, Windows 는 global-deps 관련 두 케이스 대신 "이
플랫폼은 아무것도 로드하지 않는데 있는" 한 케이스만 있어 9/9 입니다. 손상은 헤더 필드를 직접
고쳐 만들므로 — ELF 의 `e_machine`, Mach-O 의 `LC_BUILD_VERSION.platform` — **디스크에 다른
아티팩트가 있든 없든 그 개수가 매번 다 돕니다.** 건너뛴 항목이 통과처럼 보이는 일이 없습니다.

**`--self-test` 자신의 기준 휠 고르개도 버전으로 짝짓습니다 (2026-08-31 정정).** `--reference`
없이 `--self-test` 를 돌리면 비교 대상 macosx 휠을 고르는데, 한동안 이 기본값이 `find_sibling`
의 원래 결함과 같은 모양이었습니다 — `sorted(glob("*macosx*.whl"))[-1]`, 버전 구분 없이
정렬상 마지막 것. 운영 경로의 기본값(§7.4 의 `find_sibling` 바로 아래, 이 파일 내
`_default_reference`)은 이미 `{name}-{version}-*macosx*.whl` 로 버전을 넣어 골랐지만
`--self-test` 쪽만 그 필터가 빠져 있었습니다. `dist/` 에 0.0.2a0 부터 0.0.4a0 까지 나란히
있는 지금, 이것은 가정이 아니라 실측입니다: 0.0.2a0 안드로이드 휠을 셀프테스트하면서
`--reference` 를 생략하면 0.0.4a0 macosx 휠이 골렸고, 두 버전의 `{name}-{version}.data/...`
경로가 구성상 항상 다르므로 그 자체로 "missing here" 문구가 나와 — "part of the vendored tree
dropped" 케이스가 **실제로 드롭이 있었는지와 무관하게** 통과했습니다. 지금은 `--self-test` 도
운영 경로와 같은 `_default_reference` 를 쓰고, 그 자체를 시험하는 케이스(위 목록의 마지막 줄)
가 추가되었습니다.

기기 쪽 음성 대조는 §7.3.1 의 세 번째 줄입니다.

### 7.6 남은 칸을 채우려면

| 빈 칸 | 필요한 것 |
|---|---|
| ~~iOS 시뮬레이터 임포트~~ | **채워졌습니다** — `tools/wheel/verify_ios_sim.py`, `docs/platform/IOS.md`. 앱 번들은 결국 필요 없었습니다: `Py_BytesMain` 으로 만든 최소 실행 파일을 `simctl spawn` 으로 돌리는 것으로 충분했고, 앱 번들보다 훨씬 쌉니다 |
| ~~iOS 실기 **링크 해결**~~ | **채워졌습니다** — `tools/wheel/verify_ios_device.py`, §7.4.1, `docs/platform/IOS.md` §11. 실기가 필요한 것은 *실행*이지 *대조*가 아니었습니다 |
| iOS 실기 **로드·임포트·계산** | 기기 · 프로비저닝 프로파일. 이 기계에 없습니다. 시뮬레이터 결과로 대신할 수 없습니다 (`docs/platform/IOS.md` §7). 심볼이 푸는 것은 필요조건이지 충분조건이 아닙니다 — dyld 가 실행 시점에 프레임워크를 *찾는* 것과 코드 서명이 남습니다 |
| iOS 실제 앱 번들 경로 | `Python.framework` 를 `Embed & Sign` 으로 넣고 앱 프로세스가 스스로 `Py_Initialize` 를 부르는 형태. rpath 해석이 `@executable_path/Frameworks` 로 바뀝니다 |
| Android `_multiprocessing` | 배포본 재빌드, 또는 상류 `torch/multiprocessing` 지연 임포트 |
| `pip install` 로서의 안드로이드 설치 | 기기용 pip. 지금은 언팩으로 대신하고 있고, 그 차이를 §7.3 이 적어 두었습니다 |

### 7.7 `docs/design/CARGO_KT.md` 와의 관계

설계 문서는 이 배선을 `pypackpack` 저장소의 Cargo 백엔드가 맡는 것으로 상정합니다. **이 작업은
그 저장소를 건드리지 않았습니다.** `tools/wheel/build.py` 의 `--target` 은 그 백엔드가 생기기
전까지의 경로이고, `TARGETS` 표가 그때 옮겨질 것 — 타깃 하나는 "산출물 경로 · 컴파일러 ·
태그" 세 답이고, 그 세 개가 `Cargo.kt` 가 인코딩해야 할 것과 같습니다.

---

## 8. PyPI 로 갈 때

**아직 올리지 않았습니다.** 확인한 것과 남은 것:

| | |
|---|---|
| `twine check` | **4/4 PASSED** (macOS · Android · iOS · iOS 시뮬레이터) |
| 파일 크기 | 12.7~13.0 MiB — 기본 한도 100 MB 대비 여유 |
| 프로젝트 총량 | 플랫폼당 ~13 MB. 지금 있는 넷을 다 내도 기본 총량 한도에 한참 못 미친다 |
| 태그 | 전부 `cp313-abi3-*`. abi3 이므로 **파이썬 버전마다 낼 필요가 없다** — 3.13 · 3.14 를 하나로 덮는 것을 §2.3 에서 확인했다 |
| 여러 플랫폼 | 태그가 다르므로 같은 버전에 여러 파일을 올린다 |
| **iOS 두 개** | 만들어졌고 아티팩트는 맞지만 **한 번도 실행된 적이 없다** (§7.0). 검증되지 않은 휠을 올리는 것은 지금 PyPI 에 있는 껍데기와 같은 종류의 약속이다 — 설치는 되고 동작은 확인된 바 없는 것. **업로드 판단이 필요한 지점이고, 이 작업은 아무것도 올리지 않았다** |
| **라이선스** | §6 의 첫 줄. **이것이 업로드 전 진짜 블로커다** |
| `0.0.1a0` 재사용 불가 | PyPI 는 같은 버전에 같은 파일명을 다시 받지 않는다. 지금 `0.0.1a0` 에는 `py3-none-any` 가 이미 올라가 있다. 플랫폼 휠은 파일명이 다르므로 **같은 버전에 추가**할 수는 있는데, 그러면 그 버전이 "설치는 되는데 안 되는 것" 과 "되는 것" 을 동시에 갖게 된다. 새 버전을 쓰는 편이 낫다 — 판단 필요 |

`py3-none-any` 는 다른 모든 태그보다 **덜** 구체적이라, 같은 버전에 둘 다 있으면 pip 는 플랫폼
휠을 먼저 고릅니다. 그래도 태그가 맞지 않는 플랫폼(예: 리눅스)에서는 여전히 껍데기가 설치됩니다.

---

## 9. 이 작업이 만든 것 / 고친 것

**기능 추가**

- `tools/wheel/build.py` — 플랫폼 휠 빌드. preflight · global-deps 스텁 · retag · install name ·
  상류 dist-info 주입 · 아카이브 대조
- `tools/wheel/verify.py` — 깨끗한 venv 설치 + `torch.__file__` 판정
- `setup.py` — `has_ext_modules()` 와 `py_limited_api="cp313"`. 이 둘이 태그를
  `py3-none-any` 에서 `cp313-abi3-<plat>` 로 바꾼다

**결함 수정**

- `vendor/vendor_torch.sh` — `torchgen` · `functorch` 미벤더링 (§3.1). PYTHONPATH 워크플로가
  가리고 있던 결손
- `vendor/vendor_torch.sh` — 재벤더링 시 `native_left` 가 우리 `_C` 를 세던 것
- `pyproject.toml` — `dependencies = []` 가 상류의 순수 파이썬 의존성까지 비우고 있던 것 (§3.6)
- `pyproject.toml` — `packages.find` 가 `torch` 를 제외해 껍데기를 만들던 것

**문서**

- 이 문서
- `README.md` Install 절 — "`0.0.1a0` 은 이름 예약" 서술을 현재 상태로 갱신

**변경하지 않은 것**

- `rust/torch_c/` — 한 줄도. §5 의 구멍은 그대로 남겼다
- `vendor/install_shim.sh` — 소스 트리 동작을 기존 문서대로 유지
- 벤더링 트리 자체 — 전부 `vendor_torch.sh` 가 생성한 것
- `pypackpack` 저장소

---

## 10. 크로스 휠 회차 (2026-08-28) — 만든 것 / 고친 것

앞 절들은 호스트 휠을 처음 만든 기록이고, 이 절은 안드로이드 · iOS 를 붙인 회차입니다.
칸을 나눈 이유는 CLAUDE.md §5.3 입니다 — 테스트 수는 진척이 아니고, 무엇이 어느 칸인지
밝히지 않으면 삭제와 구현이 같은 줄에 섞입니다.

**기능 추가**

- `tools/wheel/build.py --target {android-arm64-v8a,ios-arm64,ios-arm64-sim}` —
  크로스 산출물 주입 · 타깃 컴파일러로 global-deps · PEP 738/730 태그 유도. 호스트 경로는
  동작이 바뀌지 않았고, 태그도 그대로 `macosx_11_0_arm64` 입니다
- `tools/wheel/binfmt.py` — Mach-O / ELF 를 직접 읽는다. `LC_BUILD_VERSION` ·
  `LC_VERSION_MIN_*` · `LC_LOAD_DYLIB` · ELF `e_machine`. 아카이브 안의 **바이트**에 대해
  물어야 하므로 `file(1)` 로는 안 됩니다
- `tools/wheel/verify_cross.py` — 크로스 휠 정적 판정 + `--self-test` (9 개 오류 모드)
- `tools/wheel/verify_android.py` — 기기 site-packages 설치 + `import torch` + 연산 판정

**결함 수정**

- `tools/wheel/build.py` — 호스트 global-deps 가 빌드 SDK 의 배포 대상(`macos 26.0+`)으로
  스탬프되어 태그(`macosx_11_0`)와 어긋나 있던 것 (§3.2.1)

**측정 (새 사실)**

- 안드로이드 휠에서 **`TORCH_USE_RTLD_GLOBAL=1` 이 더 이상 필요 없다.** docs/devices/DEVICE.md §4 는
  이것을 필수로 기록하고 있고, 그 표는 이 결과로 정정되어야 합니다 (§7.3.1). 파일을 치우면
  다시 실패하는 것까지 확인했습니다
- iOS **시뮬레이터도 `PYO3_CONFIG_FILE` 이 필요하다.** RUST_CROSSBUILD.md §0.5 와
  `rust/torch_c/build.rs` 주석은 실기 전용으로 적고 있습니다 (§7.1)
- 세 타깃 산출물의 배포 대상: 실기 `ios 10.0+`, 시뮬레이터 `iossimulator 14.0+`,
  안드로이드 API 21. 시뮬레이터 쪽이 CPython 의 12.0 보다 높아 **태그 하한을 산출물이 정합니다**

**문서**

- 이 문서 §0 · §1 · §3.2 · §7 · §8. §0 의 호스트 숫자와 스위트 수(113/2268 → 168/2702)가
  낡아 있어 실측으로 갱신했습니다

**손대지 않은 것**

- `rust/torch_c/` · `tools/golden/` · `scripts/` · `vendor/` — 한 줄도
- `rust/torch_c/build.rs` 와 `.cargo/config.toml` — §7.1 이 지적하는 것은 그 두 파일의
  **주석**이지 배선이 아니고, 소유 범위 밖이라 남겼습니다
- `docs/devices/DEVICE.md` — §7.3.1 이 그 문서의 표를 정정해야 하지만 소유 범위 밖입니다
- 업로드. 아무것도 PyPI 에 올리지 않았습니다

**기존 검증** (이 작업 전후 동일, 종료 코드로 판정)

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh   ->  EXIT=0,  ok 168
$PY tools/golden/compare.py                 ->  EXIT=0,  2702/2702, ops=118
$BPY tools/wheel/build.py                   ->  EXIT=0
$BPY tools/wheel/verify.py dist/*macosx*    ->  EXIT=0,  PASS
$BPY -m twine check dist/*.whl              ->  EXIT=0,  4/4 PASSED
```

---

## 11. 낡은 아티팩트가 어떻게 걸리는가 (2026-08-29)

### 11.1 무엇이 일어났는가

측정된 사실입니다.

```
CARGO_TARGET_DIR/release/lib_C.dylib                 2026-08-29 10:18   ← 그날
CARGO_TARGET_DIR/aarch64-apple-ios/release/...       2026-08-25 02:53   ← 4 일 전
CARGO_TARGET_DIR/aarch64-apple-ios-sim/release/...   2026-08-24 11:17   ← 5 일 전
```

`python tools/wheel/build.py --target ios-arm64-sim` 이 **exit 0 을 내면서 다시 빌드하지
않았습니다.** 당연합니다 — `build.py` 는 크로스 산출물을 **만들지 않고**
`CARGO_TARGET_DIR/<triple>/release/` 에서 **집어옵니다** (§7.1 이 cargo 를 먼저 돌리라고 적어
둔 이유가 그것입니다). 그리고 `preflight` 는 벤더 트리와 호스트 `_C` 만 봤습니다.

**결과는 5 일 묵은 소스로 만들어진 시뮬레이터 휠이었고, 그 사이 착지한 bf16 수정 · 스키마 ·
분해 · dtype · 양자화 · linear · 골든이 하나도 들어 있지 않았습니다.** 그런데 검증은
통과했습니다 — 태그도, 플랫폼 검사도, 파일 목록도, `verify_cross.py` 도, **아무것도 코드의
나이를 보지 않기 때문입니다.** 알아챈 것은 실행 중 나온 에러 문구가 **현재 소스에 없는
것**("overload resolution is not implemented")이었기 때문이고, 그것은 운입니다.

### 11.2 무엇으로 판정하는가 — cargo 의 dep-info

`build.py` 는 이제 담을 `_C` 가 **지금 디스크에 있는 소스로 빌드된 것인지** 확인합니다. 기준은
여기에 적어 둔 glob 이 아니라 **cargo 가 산출물 옆에 써 둔 `lib_C.d`** 입니다.

```
$ cat $CARGO_TARGET_DIR/aarch64-apple-ios/release/lib_C.d
.../aarch64-apple-ios/release/lib_C.dylib: .../build.rs .../src/aten.rs .../src/bootstrap.py
  .../src/methods.json .../src/overloads.json .../src/surface.json ... (18 개)
```

그것을 고른 이유가 셋입니다.

| | |
|---|---|
| **빌드가 실제로 읽은 것**이 적혀 있다 | 이 크레이트는 `src/methods.json` · `src/overloads.json` · `src/surface.json` · `src/bootstrap.py` 를 `include_str!` 합니다. 손으로 쓴 glob 이 그중 하나를 빠뜨리면 **그 파일의 변경에만 조용히 눈이 먼 검사**가 됩니다 |
| **타깃별**이다 | 자기가 설명하는 산출물 바로 옆에 있으므로, 실기 답과 시뮬레이터 답이 섞일 수 없습니다 |
| **절대경로**다 | 다른 체크아웃에서 빌드된 산출물이 보입니다. 소스 glob 은 이것을 **아예 볼 수 없습니다** — 이 트리의 mtime 을 남의 트리에서 나온 바이너리와 비교하고 최신이라고 답할 것입니다 |

**바이트 비교가 아니라 mtime 인 이유.** `rust/torch_c/pytests/run.sh` 는 바이트를 비교하는데,
그쪽은 **직접 빌드해서** 비교 기준이 손에 있고, 그래야 no-op 재빌드에서 헛경보가 나지 않기
때문입니다. 여기서는 빌드하지 않기로 했으므로(§11.3) 비교할 대상이 없습니다. 남는 것은 mtime
이고, 그것을 dep-info 에서 읽으면 **cargo 가 재빌드를 결정할 때 던지는 것과 같은 질문**이
됩니다 — 여기서 새로 발명한 더 엄한 질문이 아니라. (확인함: 소스를 `touch` 하고 재빌드하면
cargo 는 바이트 동일한 산출물을 내면서 mtime 을 갱신하므로, 헛경보가 나지 않습니다.)

### 11.3 거절인가 재빌드인가 — 거절입니다

`run.sh` 가 낡은 shim 을 발견했을 때 설치하지 않고 거절한 것과 같은 이유입니다.

1. **여기서 빌드한다는 것은 크로스 빌드의 두 번째 철자를 적는다는 뜻입니다.** 실기 쪽은
   `PYO3_CONFIG_FILE`(그 내용은 이 저장소가 아니라 §7.1 에 적혀 있습니다) ·
   `PYO3_CROSS_LIB_DIR` · `TORCHNATIVE_PYTHON_FRAMEWORK_DIR` 이 필요하고, 안드로이드 쪽은
   `scripts/device_android.sh build` 와 `cargo ndk --platform 21` 을 거칩니다. 두 번째 철자는
   첫 번째와 어긋날 수 있고, **어긋난 것이 기기가 파일을 거절할 때까지 안 보이는 것**이 이
   저장소가 반복해 온 결함의 종류입니다.
2. **`build.py` 는 이미 산출물이 *없을* 때 거절하고 그 문서를 가리킵니다.** 낡음은 그 질문의
   한 칸 안쪽이고, 다르게 답하면 "없는 것은 네 문제, 낡은 것은 내 문제" 가 됩니다.

### 11.4 세 갈래를 다 시험했습니다

**`cmp` 는 같음(0) · 다름(1) · 비교 실패(>1) 를 구분하고, 뒤의 둘을 합치면 검사가 자기 실패와
검사 대상을 구분하지 못합니다.** `run.sh` 주석에 그 사고가 적혀 있습니다 — 동시 빌드의 메모리
압박으로 커널이 `cmp` 를 SIGKILL 했고, 종료 코드 137 을 "다름" 으로 읽어 **멀쩡한 shim 을 낡았다고
보고**했습니다.

같은 함정이 여기에도 있고, 방향이 반대라 더 빠지기 쉽습니다. **빈 의존성 목록에 대한 `max()`
는 던질 에러가 없습니다** — 그냥 "산출물보다 새로운 것이 없다" 고 답합니다. 즉 읽을 수 없거나,
파싱할 수 없거나, 남의 체크아웃에서 온 dep-info 가 **`fresh` 로 읽힙니다.**

그래서 답이 셋이고, **낡음을 주장하는 것은 그중 하나뿐입니다.**

```
$ $BPY tools/wheel/build.py --self-test
SELF-TEST of the artefact freshness check (8 cases)
  ok    fresh   current artefact
  ok    stale   a prerequisite modified after the build
  ok    unknown no dep-info at all
  ok    unknown dep-info with no rule in it
  ok    unknown dep-info describing a different artefact
  ok    unknown rule with no prerequisites
  ok    unknown a prerequisite that is gone
  ok    unknown prerequisites from another checkout

  1 staleness report(s), 6 'cannot judge', 1 clean -- and the two refusal texts
  assert they are not each other
SELF-TEST: PASS -- 8/8 cases answered as specified
```

self-test 는 **판정 문자열까지 맞춰 봅니다** — `stale` 메시지는 `is stale.` 을 담고
`NOT a staleness report` 를 담지 않아야 하고, `unknown` 메시지는 그 반대여야 합니다. 검사가
그 둘을 섞으면 self-test 가 깨집니다.

### 11.5 배선이 실제로 발동하는지 확인했습니다

self-test 는 함수를 시험하는 것이고, 그 함수가 **빌드 경로에 실제로 걸려 있는지**는 다른
질문입니다. §11.1 의 상황을 그대로 재현했습니다 — 소스를 `touch` 하고, **호스트만 다시
빌드**한 뒤 크로스 휠을 만들었습니다.

```
$ $BPY tools/wheel/build.py --target ios-arm64
vendored torch 2.13.0 (2372 modules) + _C.abi3.so (4,047,888 B)
  current: 18 recorded inputs, newest rust/torch_c/src/aten.rs, 0.0 h before it
  current: torch/_C.abi3.so is byte-identical to lib_C.dylib
tools/wheel/build.py: .../aarch64-apple-ios/release/lib_C.dylib is stale.
  rust/torch_c/src/aten.rs was modified 0.2 h (554 s) after lib_C.dylib was written,
  and it is one of the 18 inputs that build read.
  ...
  Fix: re-run the cross build for this target -- docs/platform/WHEEL.md §7.1 has the exact
  command (cargo build --release --target aarch64-apple-ios, with PYO3_CONFIG_FILE
  and PYO3_CROSS_LIB_DIR, TORCHNATIVE_PYTHON_FRAMEWORK_DIR)   [EXIT=1]
```

**호스트 두 줄은 통과하고 크로스만 거절합니다.** 전부 거절하는 검사는 아무것도 구분하지 않는
검사이므로, 이 비대칭이 확인 대상이었습니다.

### 11.6 호스트도 같이 걸립니다 — 다만 질문이 두 겹입니다

호스트 휠의 `_C` 는 `vendor/install_shim.sh` 가 `cargo build` 후 **복사한 사본**이고, 사본에는
무엇이 만들었는지가 적혀 있지 않습니다. 그래서 두 겹으로 묻습니다.

| | 어떻게 | 왜 그 방법인가 |
|---|---|---|
| cargo 산출물이 최신인가 | dep-info (§11.2) | 위와 같음 |
| shim 이 **그** 산출물인가 | **바이트 비교** | `cp` 는 사본에 새 mtime 을 찍으므로 mtime 은 "더 새롭다" 고만 말하고 아무 의미가 없습니다. `run.sh` 가 벤더 shim 에 대해 하는 것과 같은 검사입니다 |

`CARGO_TARGET_DIR` 아래에 호스트 산출물이 아예 없으면 **`CANNOT JUDGE` 로 거절합니다** — 다른
`CARGO_TARGET_DIR` 로 shim 을 설치했을 수 있고, 그때 이 검사가 할 수 있는 정직한 답은 "모른다"
이지 "낡았다" 가 아닙니다. 메시지가 그렇게 말하고 `CARGO_TARGET_DIR` 을 가리킵니다.

크로스 경로에는 이 두 번째 겹이 필요 없습니다 — cargo 산출물을 **직접** 읽으므로 사이에 사본이
없습니다.

---

## 13. 백엔드(CPU · GPU · NPU)를 휠 이름으로 가를 수 없다

**휠 파일명에는 백엔드 축이 없습니다.** 이름이 나르는 것은
`(배포판, 버전, [build tag], python 태그, abi 태그, 플랫폼 태그)` 뿐이고, pip 은 뒤의 셋으로
고릅니다. `build tag` 자리는 존재하지만(`torchnative-0.0.2a0-1-cp313-abi3-...`) pip 은 그것으로
**선택하지 않고 동점 처리만** 합니다.

그러니 같은 `(python, abi, platform)` 에 CPU 판과 GPU 판을 나란히 올릴 방법이 없습니다.

**상류가 정확히 이 문제를 겪고 있고, PyPI 밖으로 나가는 것으로 풉니다.** torch 2.13.0 은 한
버전에 24 개 파일을 올리는데 그 축은 플랫폼 4 × 파이썬 6 이고, **CUDA 빌드는 PyPI 에 없습니다** —
`download.pytorch.org/whl/cu121` 같은 별도 인덱스로 냅니다.

### 13.1 이 프로젝트의 방침

**PyPI 에는 CPU · GPU · NPU 를 한 바이너리로 담은 것 하나만 올린다.** 백엔드는 런타임에 고른다.

**이 방침이 성립하는 근거가 이미 측정돼 있습니다.** Vulkan 프로브(`docs/devices/VULKAN.md`)가 `ash` 를
고른 이유 중 하나가 **`libvulkan` 을 `NEEDED` 가 아니라 `dlopen` 으로 연다**는 것이었습니다.
드라이버가 없는 기기는 **GPU 경로를 잃지 `import torch` 를 잃지 않습니다** — macOS 에서 그
폴백을 실측했습니다. NPU(NNAPI · CoreML)도 같은 성질입니다: 런타임에 컴파일하고, 없으면
없는 대로 CPU 로 떨어집니다.

즉 **한 바이너리가 셋을 다 담고 있다가 있는 것을 쓰는 것**이 가능하고, 그것이 휠 하나로 가는
유일한 길입니다.

### 13.2 그러면 `torchnative[gpu]` 는 무엇을 하는가

**엑스트라는 휠의 내용을 바꾸지 않습니다.** `pip install torchnative[gpu]` 는 같은 휠을 설치하고
**의존성을 더** 얹을 뿐입니다. 그러니 엑스트라가 담당하는 것은 이렇게 갈립니다:

| | 무엇 |
|---|---|
| **휠 안** | 백엔드 코드 자체 — 컴파일된 커널, `dlopen` 하는 로더 |
| **엑스트라** | 그 백엔드가 **런타임에 필요로 하는 파이썬 쪽 부속** |

`torchnative[cpu]` · `[gpu]` · `[npu]` 를 선언하되, **셋이 같은 바이너리를 받는다는 것을
엑스트라 설명에 적어야 합니다.** 안 적으면 `[cpu]` 가 더 작은 것을 받는다고 읽힙니다 — 안 그렇습니다.

**아직 무엇을 넣을지 정할 단계가 아닙니다.** GPU 도 NPU 도 `cpu` 밖으로 나간 적이 없고
(`torch.device("vulkan")` 은 이름을 대고 거절합니다), 엑스트라가 끌어와야 할 부속이 무엇인지는
그 경로가 실제로 서야 압니다. 지금은 **자리만 잡고 비워둡니다.**

### 13.3 지켜볼 것

파이썬 패키징이 이 구멍을 메우려는 시도가 있습니다 — "wheel variants" 계열 제안이 바로 이
축(같은 플랫폼, 다른 가속기)을 겨냥합니다. **표준이 되면 별도 인덱스 없이 갈라 낼 수 있습니다.**
표준화 전에는 위 방침(한 바이너리 + 런타임 선택)이 PyPI 안에 머무는 유일한 방법입니다.

---

## 12. 이 회차 (2026-08-29) — 만든 것 / 고친 것

**결함 수정**

- `tools/wheel/build.py` — **크로스 산출물의 나이를 아무도 보지 않던 것** (§11). 5 일 묵은
  아티팩트가 exit 0 으로 휠에 들어가고 모든 검증을 통과했습니다. 호스트 shim 도 같은
  구멍이었고 (§11.6) 함께 막았습니다

**기능 추가**

- `tools/wheel/build.py --self-test` — 신선도 판정의 8 개 사례. 빌드하지 않습니다 (§11.4)
- `tools/wheel/verify_ios_device.py` — 기기 휠의 링크 해결 검사 + `--self-test` (5 개 오류
  모드). `verify_ios_sim.py` 가 시뮬레이터에 대해 하는 것의 **기기판이 아니라**, 기기에서
  *정적으로 답이 나오는 것*만 골라 답하는 도구입니다 (§7.4.1, `docs/platform/IOS.md` §11)

**측정 (새 사실)**

- 기기 `_C.abi3.so` 의 미해결 심볼 **222 개가 전부 자기가 묶인 라이브러리에서 풀립니다** —
  `Python.framework` 118, iOS SDK 104, 미해결 0
- 기기/시뮬레이터 두 슬라이스는 **미해결 심볼의 성질이 다릅니다.** 시뮬레이터 쪽 118 개는
  `dynamically looked up` 이고 기기 쪽은 `Python` 에 이름으로 묶여 있습니다. `docs/platform/IOS.md` §7
  의 "심볼도 같다" 는 서술을 이것으로 정정했습니다
- 두 iOS 휠은 확장과 dist-info 를 빼면 **바이트 단위로 동일**합니다 (공유 2,565, 차이 0)
- 기기 슬라이스는 시뮬레이터에서 `incompatible platform (have 'iOS', need 'iOS-sim')`,
  macOS 에서 `(have 'iOS', need 'macOS')` — 실행 검증이 실기에서만 가능한 이유를 dyld 의
  말로 확인했습니다

**문서 정정**

- `docs/platform/IOS.md` §3 의 `len(dir(torch._C))` 가 1251 로 낡아 있었습니다. 재측정 1260 이고,
  **호스트와 시뮬레이터가 여전히 같다**는 것이 그 표의 요지입니다
- 이 문서 §7.0 의 판정표에 **"심볼 해결됨" 칸**을 나눴습니다. 앞의 세 휠에서는 뒤 칸에 흡수돼
  있던 구분인데, 실기 칸에서만 따로 답이 나옵니다

**손대지 않은 것**

- `rust/torch_c/` · `tools/golden/` · `scripts/` · `vendor/` — 한 줄도
- 업로드. 아무것도 PyPI 에 올리지 않았습니다

**검증** (전부 종료 코드로 판정)

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh              ->  EXIT=0,  ok 197
$PY tools/golden/compare.py                            ->  EXIT=0,  2811/2811, ops=119
$PY rust/torch_c/pytests/verify_schemas.py             ->  EXIT=0,  4203/4203
$BPY tools/wheel/build.py --self-test                  ->  EXIT=0,  8/8
$BPY tools/wheel/build.py                              ->  EXIT=0
$BPY tools/wheel/build.py --target ios-arm64           ->  EXIT=0
$BPY tools/wheel/build.py --target ios-arm64-sim       ->  EXIT=0
$BPY tools/wheel/verify.py       dist/*macosx*         ->  EXIT=0,  PASS
$BPY tools/wheel/verify_cross.py dist/*ios*  (2 개)     ->  EXIT=0,  PASS
$BPY tools/wheel/verify_cross.py --self-test (2 개)     ->  EXIT=0,  9/9
$PY  tools/wheel/verify_ios_sim.py    dist/*iphonesimulator*  ->  EXIT=0,  PASS
$PY  tools/wheel/verify_ios_device.py dist/*iphoneos*         ->  EXIT=0,  PASS
$PY  tools/wheel/verify_ios_device.py --self-test dist/*iphoneos*  ->  EXIT=0,  5/5
```

안드로이드는 이 회차에서 돌리지 않았습니다 — 이 worktree 의 `CARGO_TARGET_DIR` 에 안드로이드
산출물을 만들지 않았습니다. **`build.py --target android-arm64-v8a` 를 실제로 돌려 확인한
결과**는 신선도 검사가 아니라 그 앞 단계인 **기존의 "산출물이 없다" 거절**입니다 (EXIT=1).
§11.3 의 2 번이 말하는 대칭이 그것입니다 — 없음과 낡음이 같은 자리에서 같은 문서를 가리킵니다.

---

## 15. 검증 도구 자체를 감사한 회차 (2026-08-31) — 실패할 수 없는 검사를 찾는다

하루 안에 `tools/wheel/` 에서 결함 셋이 같은 모양으로 나왔습니다: 전부 "무언가를 잡으려던
검사인데 잡지 못했다" 는 뜻이었습니다 — `verify()` 는 MISSING 만 알아서 `.DS_Store` 가 여섯
휠에 실렸고, `preflight()` 는 살아있는 소스 트리만 봐서 낡은 setuptools 빌드 캐시를 못 봤고,
`find_sibling()` 은 `sorted(glob(...))[0]` 이라 0.0.4a0 실기 휠을 0.0.2a0 시뮬레이터 휠과
비교하고도 PASS 를 냈습니다. 앞의 둘은 그 자리에서 고쳐졌고, 세 번째는 **보고만 되고 고쳐지지
않은 채** 이 회차의 출발점이었습니다: `verify_ios_device.py` 의 `ladder()` 가 "symbols
resolved" 칸을 `symbols_ok and not findings.blind` 로 계산하는데, 이 `findings.blind` 가 심볼
검사와 형제 비교가 **함께 쓰는 목록**이라 형제 비교 쪽의 눈멂이(심볼과 무관해도) 그 칸을
`[NO]` 로 끌어내렸습니다. §7.4 에 재현과 정정을 적어 두었습니다.

**결함 수정**

- `tools/wheel/verify_ios_device.py` `ladder()` — 심볼 검사와 형제 비교가 각자의 `Findings`
  를 쓰도록 분리. 사다리의 각 칸이 자기 증거로만 판정되고, 다른 칸의 눈멂이 새어 들어올 공유
  목록이 없어졌습니다 (§7.4)
- `tools/wheel/verify_cross.py` `--self-test` 의 기준 휠 고르개 — `sorted(glob("*macosx*.whl"))
  [-1]` (버전 구분 없음) 을 운영 경로와 같은 `_default_reference`(`{name}-{version}-
  *macosx*.whl`) 로 교체. `dist/` 에 여러 버전이 나란히 있는 지금 실측으로 재현: 버전이 안
  맞는 기준 휠을 골랐을 때, 두 버전의 `.data/...` 경로 차이만으로 "part of the vendored tree
  dropped" 케이스가 실제 드롭 여부와 무관하게 통과했습니다 (§7.5)
- `tools/wheel/binfmt.py` `elf_symbols()` — 읽을 수 있는 섹션 헤더 안에 `SHT_DYNSYM` 이 아예
  없으면 `None` (읽기 실패) 을 답하도록 정정. 전에는 `{"defined": set(), "undefined": []}` 를
  답해서, 동적 공유 객체의 섹션 헤더 하나만 손상돼도(다른 모든 것은 멀쩡해도) `verify_linux.py`
  가 "0 undefined, 0 unresolved" 로 깨끗하게 통과했습니다. 실제 `libtorch_global_deps.so` 의
  섹션 하나의 `sh_type` 을 `SHT_DYNSYM` 에서 `SHT_NULL` 로 바꿔 재현
- `tools/wheel/build.py` `LinuxTarget._check_policy()` — `elf_dynamic(artefact) or
  {"needed": []}` 가 "읽을 수 없음" 과 "링크한 게 없음" 을 같은 답으로 접었습니다. 두 메서드
  아래 `_glibc_floor()` 는 이미 그 둘을 구분하는데(주석이 그 이유까지 적어 두었습니다)
  `_check_policy()` 만 빠져 있었습니다. `platform_tag()` 가 같은 바이트로 `_glibc_floor()` 를
  바로 뒤에 부르는 덕에 오늘까지는 가려져 있었을 뿐 — 그 호출 순서가 바뀌거나 `_check_policy`
  가 다른 데서도 불리면 그 즉시 드러났을 결함입니다
- `tools/wheel/build.py` `upstream_dist_info()` — 벤더링 트리에 `torch-*.dist-info` 가 없으면
  `print()` 만 하고 빈 dict 를 반환했습니다. 그 dict 가 `extra` 로 들어가고, `verify()` 의
  "누락"/"안 부른 것" 두 검사 모두 `expected`(패키지 트리 걷기) 와 `extra` 로만 판단하므로 —
  둘 다 이 파일을 애초에 이름 붙인 적이 없어 그 부재를 볼 방법이 없었습니다. 지금은 거절합니다

**테스트 추가** (기존 스위트에 케이스만 더한 것 / 새 스위트를 만든 것을 나눕니다)

- `verify_ios_device.py --self-test` 7/7 → 8/8 — `ladder()` 자체를 시험하는 케이스. 심볼 검사가
  깨끗해도(0/222 미해결) 형제 비교가 눈멀면 "symbols resolved" 칸은 여전히 `yes` 여야 함을 확인
- `verify_cross.py --self-test` — 안드로이드/실기 10/10 → 11/11, manylinux 11/11 → 12/12,
  Windows 8/8 → 9/9. `_default_reference` 가 버전으로 짝짓는지를 직접 시험하는 케이스
- `verify_linux.py --self-test` 5/5 → 6/6 — `SHT_DYNSYM` 이 없는 ELF 가 `None` 을 답하는지
  직접 시험하는 케이스
- `build.py --self-test` — LINUX 스위트 10/10 → 11/11 (`_check_policy` 케이스), VERIFY 스위트
  3/3 → 4/4 (**"누락" 방향이 이 스위트에 한 번도 없었습니다** — 세 케이스 다 `expected` 의
  모든 파일을 항상 포함해서 만들었으므로, `verify()` 의 "누락" 검사가 통째로 회귀해도 3/3 은
  그대로 PASS 였을 것입니다: 규칙 문자열 하나를 지운 채로 실제로 돌려서 확인), UPSTREAM-DIST-
  INFO 새 스위트 2/2 (`upstream_dist_info` 전용)

**문서 정정**

- §7.4 의 "5/5" 를 8/8 로, §7.5 의 "9/9"(세 크로스 휠 공통) 를 실측(안드로이드/실기 11/11,
  manylinux 12/12, Windows 9/9) 으로 정정. §11.4/§12 의 8/8 은 신선도 스위트 것으로 그대로
  둡니다 — 이번 회차가 건드리지 않았습니다

**감사했지만 결함이 아니라고 판정한 것** (기준과 함께)

- `verify_windows.py --self-test` 의 `for candidate in sorted(REPO.glob("dist/*win_amd64.whl")):`
  (break 없이 마지막 것이 남음) — `_default_reference` 와 같은 모양이지만, 세 버전
  (0.0.2a0/0.0.3a0/0.0.4a0) 의 `torch/_C.pyd` PE import 테이블을 직접 비교해 확인: 셋 다
  `python3.dll` 119 개 · `python313.dll` 0 개 · DLL 11 개로 **완전히 같습니다** — abi3 안정
  ABI 표면이라 어느 버전을 고르든 이 self-test 의 판정이 달라지지 않습니다. 구조는 손봐야 할
  냄새지만 오판을 만들지는 않습니다
- `verify_android.py` · `verify_ios_sim.py` — `--self-test` 자체가 없고(실기/시뮬레이터가
  있어야 함), `adb shell` 의 종료 코드를 신뢰하지 않는다는 것도 주석에 명시돼 있어(마커 줄로만
  판정) 같은 모양의 결함을 찾지 못했습니다
- `verify_cross.py` 의 `AndroidExpectation`/`LinuxExpectation`/`WindowsExpectation.interpreters`
  — `sorted(glob(...))` 로 여러 인터프리터를 모을 수 있지만 `check_suffix_is_searched()` 가
  **전부**를 순회해 확인하므로(`[0]`/`[-1]` 로 하나만 고르지 않음) 여러 개 중 하나만 보는 결함이
  아닙니다

**손대지 않은 것**

- `rust/torch_c/` · `tools/golden/` — 한 줄도 (다른 에이전트 담당)
- `dist/` — PyPI 에 올라간 0.0.4a0 휠들에 손대지 않았습니다. 검증은 전부 `/tmp` 사본과 실제
  배포된 `torchnative` 저장소의 `dist/` (읽기 전용) 로 했습니다 — `twine check dist/*0.0.4a0*`
  와 sha256 로 그대로임을 확인했습니다
- 업로드 · 커밋 · 빌드 (휠을 새로 만들지 않았습니다 — 감사 대상은 도구지 산출물이 아닙니다)
호스트 두 줄은 그 앞에서 통과했습니다.
