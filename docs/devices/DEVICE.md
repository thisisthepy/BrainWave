# `import torch` 안드로이드 실기기(에뮬레이터) 검증

**결론: `import torch` 가 되고, aten op 이 돌고, `nn.Linear`/`nn.Sequential` 순전파가
호스트와 비트 단위로 일치한다.** 목표(“`import torch` 가 되고 aten op 하나가 도는 것”)를
넘어 배터리를 돌렸다. **2026-08-28 기준 배터리는 54개 케이스**(원래 33개 + mamba·mixtral
12개 op·repr 커널 6개·`cat`/`max` 결함 수정·gemm 스레딩 임계값·`sum` 결함 수정 각각의 검증
케이스)이고, 호스트와 비교 가능한 54개 중 **50개가 비트 동일**, 나머지 4개는 **1 ULP** 차이다
(§5.3).

이전 단계와의 구분:

| 문서 | 확인한 것 | 확인하지 **않은** 것 |
|---|---|---|
| `RUST_CROSSBUILD.md` | `aarch64-linux-android` 로 **링크**된다 | 로드 |
| `DEVICE_LOAD.md` | 기기에서 `import _C` 가 되고 op 3개가 돈다 | `import torch` |
| **이 문서** | 기기에서 `import torch` · aten 96개 · nn 순전파 · 호스트 대조 | APK 안에서의 로드, 실물 단말, 학습/autograd |

---

## 1. 환경 (실측)

| | 값 |
|---|---|
| 기기 | `emulator-5554` = AVD `pmp_api26`, API **26** / Android 8.0.0, `arm64-v8a`, `uid=2000(shell)` |
| 기기 CPython | `3.13.0+ (heads/3.13-dirty:b4c504d76ff, Oct 13 2024)` `[Clang 17.0.2]`, `sys.platform='android'`, `os.uname().machine='aarch64'` |
| 배포본 | `/Volumes/macMini/caches/target-python/aarch64-linux-android/prefix` |
| 호스트 | `darwin/arm64`, `/Volumes/macMini/caches/spike-venv/bin/python` (CPython 3.13.0) |
| 벤더링 트리 | `torchnative/src/main/torch` — torch **2.13.0**, `vendor/vendor_torch.sh` 로 생성 (`py_modules=2286`, `native_left=0`) |
| `_C` | `rust/torch_c`, PyO3 0.29.2 `abi3-py313` + candle-core 0.11.0 |

**`pmp_api26` 은 `PythonMultiplatform` 이 쓰는 공용 에뮬레이터다.** 앱 설치·`pm` 조작을 전혀
하지 않았고 `/data/local/tmp/bw_device` 아래에만 파일을 올렸다. `DEVICE_LOAD.md` 와 같은 규율이다.

**에뮬레이터이지 실물 단말이 아니다.** 아래 어떤 수치도 실물 폰에 대해 말하지 않는다.

## 2. 빌드

```
CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-device2
cd rust/torch_c && ANDROID_NDK_HOME=~/Library/Android/sdk/ndk/27.1.12297006 \
  PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 \
  PYO3_CROSS_LIB_DIR=<배포본>/prefix/lib \
  cargo ndk -t arm64-v8a --platform 21 build --release
```

`EXIT=0`, 52초. 산출물 `lib_C.so` **4,255,872 B**, `ELF 64-bit LSB shared object, ARM aarch64`.

```
NEEDED  libpython3.13.so   libdl.so   libm.so   libc.so
미해결 동적 심볼 185개 = Py*/_Py* 112개 + bionic 73개
```

**`libm.so` 가 `NEEDED` 에 들어 있고 미해결 심볼에 `expf`·`tanhf`·`sinf`·`cosf`·`log`·`pow` 가
있다.** §5 의 불일치가 여기서 설명된다 — 이 심볼들은 호스트에서 Apple libm 으로, 기기에서
bionic 으로 해결되며, 그 사실이 링크 정보에 그대로 적혀 있다.

## 3. 스테이징

`scripts/device_android.sh stage` 가 `/data/local/tmp/bw_device` 에 올린 것:

```
bw_device/            399 MB
├── bin/python3.13
├── lib/libpython3.13.so + lib/python3.13/   279 MB (stdlib 7,499 파일)
└── site/                                    120 MB
    ├── torch/            벤더링 트리 + _C.abi3.so (안드로이드 산출물로 덮어씀)
    ├── torch-2.13.0.dist-info/
    └── filelock fsspec functorch jinja2 markupsafe mpmath networkx packaging
        sympy torchgen typing_extensions.py yaml
```

스크립트는 스테이징 직전에 `_C.abi3.so` 를 뺀 모든 `*.so`/`*.dylib` 을 세고 0이 아니면
거부한다 — 호스트용 Mach-O 가 섞여 들어가는 사고를 막는다.

`libssl`/`libcrypto`/`libsqlite3` 는 여전히 올리지 않는다. `prefix/lib` 에서 그것들은
심볼릭 링크이고 `uid=2000` 의 `adb push` 는 원격 심볼릭 링크를 만들 수 없다
(`remote symlink failed: Permission denied`). `lib_C.so` 의 `NEEDED` 에 없으므로 이번에도
필요 없었다 — `DEVICE_LOAD.md` 의 판단 그대로다.

## 4. 무엇이 실제로 필요한 환경변수인가 (실측)

`import torch` 를 성공시킨 조합에서 변수를 하나씩 빼서 잰 결과다.

| 변수 | 빼면 | 판정 |
|---|---|---|
| `LD_LIBRARY_PATH=$ROOT/lib` | `CANNOT LINK EXECUTABLE "./bin/python3.13": library "libpython3.13.so" not found`, 종료 134 | **필수** |
| `TORCH_USE_RTLD_GLOBAL=1` | `OSError: dlopen failed: library ".../torch/lib/libtorch_global_deps.so" not found` | **필수 — 단, 이 스테이징 방식에서만. §4.1** |
| `PYTHONPATH=$ROOT/site` | (자명) | **필수** |
| `PYTHONHOME=$ROOT` | **없어도 성공한다** | **불필요** |
| `BW_STUB_MULTIPROCESSING=1` | `ModuleNotFoundError: No module named '_multiprocessing'` | **필수** (§6) |

`PYTHONHOME` 이 불필요한 것은 인터프리터를 `$ROOT/bin/python3.13` 로 실행하면 CPython 이
argv[0] 에서 prefix 를 유도하기 때문이다. `DEVICE_LOAD.md` 는 이것을 주고 있었지만,
**필요해서 준 것이 아니었다.** 앱 안에서는 실행 파일이 아니라 임베딩된 인터프리터이므로
이 결론이 그대로 넘어가지 않는다 — 그때는 다시 재야 한다.

### 4.1 정정 (2026-08-28) — 휠로 설치하면 이 변수가 필요 없다

위 표의 "필수" 는 **이 문서의 스테이징 방식에 대해서만** 참입니다. 그때 기기에 올린 트리에는
`torch/lib/libtorch_global_deps.so` 가 아예 없었고, 그래서 `_load_global_deps()` 를 건너뛰는
스위치가 필요했습니다. **없는 파일을 우회한 것이지 안드로이드의 성질이 아닙니다.**

플랫폼 휠은 그 파일을 싣습니다(`docs/platform/WHEEL.md`). 그러니 휠로 설치하면:

```
torch       : /data/local/tmp/bw_wheel/lib/python3.13/site-packages/torch/__init__.py
RTLD_GLOBAL : False          ← 주지 않았다
repo on path: []             ← 개발 트리가 아니라 진짜 설치본이다
mm          : [[3.0, 3.0], [3.0, 3.0]]
```

**음성 대조로 확인했습니다.** 기기에서 그 `.so` 의 이름을 바꾸면 예전 에러가 그대로 돌아오고,
되돌리면 다시 계산합니다. 즉 이 변수를 몰아낸 것은 정확히 그 파일입니다.

`_multiprocessing` 과 `_posixshmem` 스텁은 **여전히 필요합니다.** 그쪽은 휠이 아니라
**안드로이드 CPython 배포본이 그 둘을 빌드하지 않는** 성질입니다. 조율 세션이 검증 중
빈 `_posixshmem` 으로는 부족한 것도 확인했습니다 — `resource_tracker.py:54` 가 `shm_unlink` 를
가드 없이 읽으므로 그 이름은 존재해야 하고, 실제로 쓰이면 실패하도록 두는 것이 맞습니다.

`TORCH_USE_RTLD_GLOBAL` 은 회피가 아니라 상류가 그 목적으로 둔 스위치다
(`torch/__init__.py:406`). 켜면 `sys.setdlopenflags(RTLD_GLOBAL)` 로 `torch._C` 를 직접
임포트하고, `libtorch_global_deps` 를 dlopen 하는 `_load_global_deps()` 경로를 건너뛴다.
`VENDOR.md` 벽 1 이 호스트에서 기록한 것이 **기기에서도 같다** — `IMPORT_TORCH.md:520`
"`TORCH_USE_RTLD_GLOBAL` 의 기기 영향" 미결 항목의 답이다.

## 5. 결과 — 호스트 대비 비트 대조

`scripts/device_parity.py` 를 양쪽에서 돌리고 모든 결과를 **big-endian IEEE-754 hex** 로
찍어 비교했다. 오차 허용치가 아니라 비트 비교다.

```
host   darwin/arm64   torch 2.13.0  kernels 96
device android/aarch64 torch 2.13.0 kernels 96
identical 30/32
host failures:   ['nn.ReLU forward']
device failures: ['nn.ReLU forward']
PARITY: ok
```

**비트 동일 (30):** `add.Tensor` `sub.Tensor` `mul.Tensor` `div.Tensor` `mm.default`
`addmm.default` `bmm.default` `cos.default` `sin.default` `gelu.default` `silu.default`
`rsqrt.default` `reciprocal.default` `pow.Tensor_Scalar` `relu.default` `sum.default`
`mean.dim` `cumsum.default` `native_layer_norm.default` `max.dim` `argmax.default`
`topk.default` `sort.default` `cat.default` `permute.default` `slice.Tensor`
`embedding.default` `where.self` **`nn.Linear forward`** **`nn.Sequential 2-layer`**

**불일치 (2), 둘 다 최대 1 ULP:**

| 케이스 | 다른 원소 | 최대 차 |
|---|---|---|
| `_softmax.default` | 12개 중 6개 | 1 ULP |
| `tanh.default` | 12개 중 2개 | 1 ULP |

### 이 1 ULP 가 무엇인지 — 어느 쪽이 틀린 것이 아니다

배정밀도로 계산해 f32 로 올림한 값(정확 반올림 기준)과 양쪽을 대조했다.

`tanh.default`: 기준 `3f5ebc5d`. **호스트가 정확 반올림, 기기가 −1 ULP.**

`_softmax.default` (원소별 기준값 대비 오프셋):

```
idx    host    device   dbl-ref   host_off  device_off
  0  3e4c02c8 3e4c02c8 3e4c02c9     -1        -1
  1  3e6b56fe 3e6b56fd 3e6b56fe     +0        -1
  5  3e6b56fe 3e6b56fd 3e6b56fd     +1        +0
  7  3e9c95b5 3e9c95b4 3e9c95b4     +1        +0
  8  3e4c02c8 3e4c02c9 3e4c02c9     -1        +0
```

**양쪽이 기준값을 위아래로 번갈아 벗어난다.** 한쪽이 우월한 것이 아니라 Apple libm 과
bionic 의 `expf`/`tanhf` 가 서로 다른 구현일 뿐이다. §2 의 미해결 심볼 목록이 이것을
직접 뒷받침한다 — `expf`·`tanhf` 는 `lib_C.so` 가 로더에게 맡긴 심볼이다.

**같은 목록에 있는 `sinf`·`cosf`·`pow`·`log` 는 이번 입력에서 비트 동일했다.** 즉 이것은
"libm 이 다르면 다 다르다" 가 아니라 **함수별** 현상이고, 그래서 아래처럼 함수 이름으로
면제 목록을 둔다.

### 5.1 Apple 사용자와 안드로이드 사용자는 실제로 다른 값을 받는다 (2026-08-25)

**이 절은 위 대조가 답하지 않는 것을 적는다.** 위의 "30/32 비트 동일" 은 양쪽이 같은
행렬곱 커널(candle 의 `gemm`)을 쓸 때의 값이다. **배송되는 빌드는 그렇지 않다.**
`docs/perf/PERF.md` §3 이 Apple 타깃에만 `accelerate` 를 켰으므로,

| | 행렬곱 | 초월함수 |
|---|---|---|
| macOS · iOS 배송 빌드 | Accelerate (AMX 코프로세서) | vForce + Apple libm |
| 안드로이드 배송 빌드 | candle `gemm` (NEON) | bionic libm |

두 벌은 **비트 동일할 수 없다.** 서로 다른 BLAS 는 누적 순서가 다르고, 그 사실은
`docs/perf/PERF.md` §3 이 `accelerate` 를 켤 때 골든을 다시 돌려야 했던 이유이기도 하다.

**실측 (같은 기기 `.so`, 호스트만 배송 빌드로 바꿔 대조):**

```
identical 23/33
MISMATCH addmm.default             3/9  elements, max 1 ULP
MISMATCH bmm.default               3/9  elements, max 1 ULP
MISMATCH mm.default                1/9  elements, max 1 ULP
MISMATCH native_layer_norm.default 3/12 elements, max 2 ULP
MISMATCH nn.Linear forward         3/9  elements, max 2 ULP
MISMATCH cos.default               6/12 elements, max 1 ULP
MISMATCH sin.default               3/12 elements, max 1 ULP
MISMATCH rsqrt.default             1/12 elements, max 1 ULP
```

앞의 다섯은 BLAS 누적 순서, 뒤의 셋은 vForce 가 스칼라 `sinf`/`cosf`/`rsqrt` 를 벡터
근사로 대체하는 것이다. **1~2 ULP 이고, 어느 쪽이 틀린 것이 아니다** — §5 의 `_softmax`
분석과 같은 성격이다.

**이것은 결함이 아니다.** 상류 torch 도 플랫폼마다 다른 BLAS(MKL · Accelerate · OpenBLAS)
를 링크하므로 같은 성질을 갖는다. 다만 **결함이 아니라는 것과 존재하지 않는다는 것은
다르다.** 다음 두 가지가 여기서 나온다.

- **비트 재현성을 요구하는 기능(체크포인트 해시, 결정론적 재생, 크로스 플랫폼 골든)은
  이 차이 위에 세울 수 없다.** f32 행렬곱을 지나는 순간 플랫폼이 답을 바꾼다.
- **`scripts/device_android.sh parity` 는 이 차이를 재지 않는다.** 그 스크립트는 호스트
  쪽을 `accelerate` 없이 따로 빌드해서(`rust/torch_c/Cargo.toml` 의
  `torch_c_no_accelerate` cfg) **gemm 대 gemm** 으로 비교한다.

### 5.2 왜 parity 는 배송 빌드를 재지 않는가

위 8 건을 면제 목록에 추가하는 선택지도 있었고, **그것을 고르지 않았다.**

이 검사가 답하는 질문은 **"같은 코드가 다른 CPU 에서 같은 답을 내는가"** 다. AMX 와 NEON 을
비교하면 **이미 아는 사실**(다른 BLAS 는 다른 답을 낸다)만 확인하게 되고, 정작 잡아야 할
기기 고유 커널 결함은 그 1~2 ULP 잡음에 묻힌다. 면제 목록이 열 항목이 되는 순간 그것은
"알려진 예외 목록" 이 아니라 이 스크립트가 갖지 않기로 한 **전역 허용오차**가 된다
(§9 의 판정 원칙).

그래서 `parity` 는 호스트 쪽만 `accelerate` 를 끈 채 다시 빌드하고, **그 산출물이 정말
Accelerate 를 링크하지 않는지 `otool` 로 확인한 뒤에만** 잰다. cfg 가 어느 날 동작을
멈추면 그 결과는 오류가 아니라 **기기 회귀처럼 보이는 조용한 8 건**이므로, 플래그를 믿지
않고 산출물을 본다. (음성 대조: cfg 이름을 일부러 어긋나게 바꾸고 돌리면
`refusing to measure ...: it still links Accelerate` 로 exit 1 이다 — 확인함.)

**대가를 분명히 해 둔다: `PARITY: ok` 는 배송되는 Apple 아티팩트에 대해 아무 말도 하지
않는다.** 그것에 대해 말하는 것은 이 §5.1 이고, 위 8 건이 그 전부다.

### 5.3 배터리 확장 (2026-08-28) — ops 97→116 가 기기에서 한 번도 확인되지 않았다

**여기서 재확인한 것.** `parity` 가 마지막으로 초록이었던 것은 `_aten_implemented()` 가
97 이던 시점이다. 그 뒤로 op 이 **116** 개가 됐다:

- mamba·mixtral 12 개 (`clamp_.default` · `convolution.default` · `div_.Tensor` ·
  `empty_like.default` · `exp.default` · `floor_divide.default` · `ge.Scalar` ·
  `histc.default` · `index_put_.default` · `masked_fill_.Scalar` · `softplus.default` ·
  `zeros_like.default`, `3b9a67a`)
- "repr 커널" 6 개 (`abs.default` · `ceil.default` · `gt.Scalar`/`gt.Tensor` · `min.default` ·
  `masked_select.default` · `unbind.int`, `test_the_six_repr_kernels_dispatch` 가 도로 이름이다)
- `cat.default` 의 legacy-empty 규칙 변경 (`test_cat_skips_a_tensor_of_shape_zero_and_only_that_shape`)
- `max.default` 의 NaN 전파 수정 (candle 이 NaN 을 건너뛰던 것을 IEEE `maximum` 규칙으로 고침)
- gemm 스레딩 임계값 589,824 → 4,000,000 (`25a79df`) — n=128 이 기기에서 6.9배 빨라졌다는
  주장이 있었지만 **정확성 쪽은 기기에서 한 번도 재지 않았다**
- `sum.dim_IntList`([]) 결함 수정 (`docs/graph/DECOMP.md` §6.1, 이 회차에 같이 고쳤다)

**이 중 어느 것도 기기에서 확인된 적이 없었다.** `scripts/device_parity.py` 의 배터리는
33 케이스에 멈춰 있었고 위 항목 전부가 빠져 있었다.

#### 무엇을 더했는가

`scripts/device_parity.py` 에 21 케이스를 더해 **54 개**로 늘렸다 (원래 33 개는 그대로 두고
추가만 했다 — 기존 케이스 이름과 겹치지 않는 새 키를 썼다):

```
abs.default   ceil.default   gt.Scalar   masked_select.default   min.default   unbind.int
max.default                              ← NaN 을 포함한 입력, IEEE 전파 규칙 확인
cat.default (empty)                      ← legacy-empty 규칙: (0,) 1-D 는 스킵, 다른 empty 는 아님
clamp_.default   convolution.default   div_.Tensor   exp.default   floor_divide.default
ge.Scalar   histc.default   index_put_.default   masked_fill_.Scalar   softplus.default
zeros_like.default                       ← mamba·mixtral 12 개 중 11 개 (mixtral 값은 doc-comment
                                            를 베끼지 않고 tools/golden/cases.py 의 이미 측정된
                                            픽스처 값을 그대로 재사용했다)
mm.default (n=128, gemm threading threshold)  ← 정확성만, 성능은 안 쟀다 (CLAUDE.md)
sum.dim_IntList (dim=[])                 ← 이번에 고친 커널이 기기에서도 같은 값을 내는지
```

**뺀 것 하나, 의도적으로.** `empty_like.default` 는 안 넣었다 — 초기화되지 않은 메모리는
비트로 비교할 대상이 없다. `tools/golden/cases.py` 의 `_dtype_shape_only_check` 가 골든
하네스에서 이미 같은 이유로 값 비교를 건너뛰고, 이 배터리는 값 비교(비트 대조)가 전부이므로
같은 논리로 뺐다 — shape/dtype 만 비교하는 별도 경로는 이 스크립트에 없다.

#### 결과

```
host   darwin/arm64   torch 2.13.0  kernels 116
device android/aarch64 torch 2.13.0 kernels 116
identical 50/54
host failures:   []
device failures: []
PARITY: ok
```

**새로 추가한 20 케이스(원래 33 개 + `sum.dim_IntList([])` 를 뺀 나머지) 전부 비트 동일했다
— `max.default` 의 NaN 전파, `cat.default` 의 empty 스킵, `sum.dim_IntList([])` 의 전체 축약,
`mm.default` n=128 (gemm 스레딩 경로 변경), mamba·mixtral 11 개 op 모두.** 원래 33 개는
30 개 비트 동일 · 2 개(`tanh.default` · `_softmax.default`) 1 ULP 그대로였다 — op 이
97→116 으로 늘어난 것이 기존 케이스의 답을 바꾸지 않았다는 뜻이다.

**딱 둘, 새로 갈렸다: `exp.default` 와 `softplus.default`, 둘 다 1 ULP.** 면제 목록을 늘리기
전에 먼저 쟀다 (§9 의 원칙 그대로):

```
exp.default 입력 12/7 (index 11)
  배정밀도 기준 (올림)  40b1afc8
  호스트                40b1afc9   ← 기준보다 +1
  기기                  40b1afc8   ← 기준과 정확히 일치

softplus.default 입력 5/7 (index 4)
  배정밀도 기준 (올림)  3f8e6ebb
  호스트                3f8e6ebb   ← 기준과 정확히 일치
  기기                  3f8e6eba   ← 기준보다 −1
```

**한쪽이 항상 이기지 않는다** — `exp` 에서는 기기가 정확 반올림이고 호스트가 벗어났는데,
`softplus` 에서는 반대다. §5 의 `tanh`/`_softmax` 분석과 같은 모양이다: Apple libm 과
bionic 의 `expf` 가 서로 다른 구현일 뿐, 어느 쪽도 "틀린" 것이 아니다. `softplus` 가
내부적으로 `exp` 를 부르므로(`log(1+exp(βx))` 형태), 이 발산은 **독립된 결함이 아니라
같은 `expf` 차이가 한 겹 위로 전파된 것**이다 — 새 op 이 아니라 새로 이 `expf` 경로를
지나는 op 일 뿐이다.

그래서 `scripts/device_android.sh` 의 `EXPECTED_LIBM_DIVERGENCE` 에 두 항목을 추가했다:

```python
EXPECTED_LIBM_DIVERGENCE = {
    "_softmax.default": 1,
    "tanh.default": 1,
    "exp.default": 1,
    "softplus.default": 1,
}
```

#### 계측기가 우는지 확인했다

새 케이스를 늘렸다고 감도가 죽지 않았는지 음성 대조 두 개로 확인했다 (`device_android.sh
diff` 에 손댄 JSON을 준 것 — §9 의 기존 대조군과 같은 방법):

| 대조군 | 결과 |
|---|---|
| 면제 목록에 없는 새 op(`masked_fill_.Scalar`)을 1 ULP 틀어놓음 | `EXIT=1  PARITY: unexpected bit divergence: ['masked_fill_.Scalar']` |
| 새로 면제한 `exp.default` 를 5 ULP 로 키움(예산 1 ULP 초과) | `EXIT=1  PARITY: unexpected bit divergence: ['exp.default']` |
| 손대지 않은 원본 | `EXIT=0  PARITY: ok` |

둘 다 정확히 잡혔다. 면제 목록이 넷으로 늘어난 뒤에도 새 op 을 1 ULP 만 틀어도 실패하고,
면제된 op 이 예산을 넘겨도 실패한다 — 배터리를 키운 것이 계측기를 무디게 만들지 않았다.

#### 이 확장이 답하지 않은 것

**54 개는 116 개 전부가 아니다.** `eq`/`ne`/`lt`/`le` 의 Tensor 오버로드, `bitwise_and`/
`bitwise_or`/`bitwise_not`, `isin.Tensor_Tensor`, `rsub.Scalar`, `transpose.int`,
`stack.default`, `split.Tensor`/`split_with_sizes.default`, `scatter.src`, `gather.default`,
`index.Tensor`, `select.int`, `squeeze.dim`, `expand.default`, `arange.*`, `scalar_tensor.default`,
`neg.default`, `fill_.*`, `copy_.default`, `any.*` 는 여전히 기기에서 한 번도 안 돌았다.
이것들 중 다수는 부동소수점 libm 을 부르지 않는 정수/비트 연산이라 플랫폼 발산 가능성이
낮다고 **추정**은 할 수 있지만, 그것은 추정이지 이 회차가 잰 것이 아니다. `randint`/
`normal_`/`uniform_`/`multinomial` 은 기존 33개 배터리부터 의도적으로 빠져 있다 — RNG 스트림
자체가 플랫폼 비교 대상이 아니다.

## 6. 벽 하나 — Android CPython 에 `_multiprocessing` 이 없다

```
torch/multiprocessing/__init__.py:110
  from multiprocessing.resource_tracker import ResourceTracker
    -> multiprocessing/resource_tracker.py:41  import _multiprocessing
ModuleNotFoundError: No module named '_multiprocessing'
```

`torch/__init__.py:2298` 이 `torch.multiprocessing` 을 **무조건** 임포트하므로 우회가 없다.
안드로이드 CPython 배포본은 `_multiprocessing` 도 `_posixshmem` 도 빌드하지 않는다 —
안드로이드에 SysV IPC 가 없고 POSIX 명명 세마포어가 쓸 수 없기 때문이다.

`device_parity.py::_install_android_stubs()` 가 `BW_STUB_MULTIPROCESSING=1` 일 때만
두 모듈을 세운다. **고친 것이 아니라 계측 도구다.**

- `_multiprocessing` 은 **비워 둔다.** `resource_tracker.py:49` 가
  `hasattr(_multiprocessing, 'sem_unlink')` 로 감싸고 있고, 명명 세마포어가 없는 빌드에는
  정리할 세마포어도 없다는 것이 정직한 답이다.
- `_posixshmem.shm_unlink` 는 `resource_tracker.py:54` 가 **가드 없이** 읽으므로 이름이
  존재해야 한다. no-op 이 아니라 **`OSError` 를 던지게** 배선했다 — 실제로 쓰이면 조용히
  새는 대신 시끄럽게 실패한다.

**앱 배선이 정해질 때 이것을 결정해야 한다.** 선택지는 (a) 배포본에 두 모듈을 스텁으로
넣는다, (b) `torch/multiprocessing/__init__.py` 를 벤더링 시점에 패치한다, (c) 지금처럼
런타임 스텁을 앱 부트스트랩에 둔다. 이 문서는 **셋 중 무엇도 고르지 않았다.**

## 7. 양쪽에서 똑같이 실패하는 것 하나 — `_C` 의 갭이지 기기 문제가 아니다

> **Correction (문서 감사, 2026-09):** 닫혔습니다. `torch.relu` 가 지금
> `rust/torch_c/src/overloads.json` 에 있고(`grep -c '"relu"'` → 1), 실측:
> `torch.relu(torch.tensor([-1.0, 2.0]))` → `tensor([0., 2.])`, 성공. 어느 라운드가 채웠는지는
> 추적하지 않았습니다 — `docs/models/SAMPLING.md` 감사에서 방금 발견한 것과 같은 모양입니다(overloads
> 테이블이 이 문서를 쓴 뒤 다른 라운드에서 계속 채워졌다).
> <!-- DOCWATCH: json-key rust/torch_c/src/overloads.json relu present -->

```
nn.ReLU()(x)  ->  F.relu(x)  ->  torch.relu(x)
NotImplementedError: not implemented in torch._C shim: torch.relu(...)
  -- overload resolution has no table entry for this op
```

`torch.relu` (오버로드 접미사 없는 스펠링) 가 `rust/torch_c/src/overloads.json` 에 없다.
`torch.ops.aten.relu.default` 는 양쪽에서 **비트 동일하게 돈다.** 호스트와 기기가 **같은**
메시지로 실패하므로 이것은 기기 문제가 아니라 오버로드 테이블의 빠진 항목이다 —
양쪽에서 돌린 것의 값이 여기 있다. `nn.Sequential` 케이스가 `nn.ReLU` 대신 `nn.Tanh` 를
쓰는 이유도 이것이다.

**이 문서는 `rust/` 를 고치지 않았다.** 수정은 별도 작업이다.

> **이후 고쳐졌다 (2026-08-25 확인).** 오버로드 테이블에 그 스펠링이 들어가면서 `nn.ReLU`
> 가 양쪽에서 돈다. 같은 배터리를 지금 돌리면 `cases=33 ok=33 failed=0` 이고 양쪽
> 실패 목록이 비어 있으며, 비교 가능한 33 건 중 **31 건이 비트 동일**(나머지 둘은 §5 의
> 면제된 libm 두 건)이다. 위 32 건·30 건은 그 시점의 기록으로 남긴다.

## 8. 대략적인 시간 (측정 아님 — 부하 있는 기계, 에뮬레이터)

load average 2.98 인 상태에서 잰 값이고 에뮬레이터다. **회귀 판정에 쓰지 마라.**
자릿수만 보라는 뜻으로 남긴다.

| | 기기 | 호스트 |
|---|---|---|
| `import torch` | 0.372 s (`sys.modules` 1001) | 0.347 s (1100) |
| `aten.add.Tensor` 64×64 | 1.2 µs/call | 1.0 µs/call |
| `aten.mm.default` 64×64 | 10.7 µs/call | 8.9 µs/call |

에뮬레이터는 Apple Silicon 위에서 같은 aarch64 를 거의 네이티브로 돌리므로 두 값이 가까운
것은 당연하다. **실물 단말은 이 표에 없다.**

## 9. 재현

```sh
export PATH="$HOME/.cargo/bin:$HOME/Library/Android/sdk/platform-tools:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-device2
export ANDROID_SERIAL=emulator-5554  # 여러 대가 붙어 있으면 필수

bash vendor/vendor_torch.sh          # 벤더링 트리
bash vendor/install_shim.sh          # 호스트 _C (배송 설정 — 골든·pytests 용)
sh scripts/device_android.sh build   # 안드로이드 _C
sh scripts/device_android.sh stage   # 기기에 올림
sh scripts/device_android.sh parity  # 양쪽 실행 + 비트 대조
```

`parity` 는 실패하면 종료 코드 1 과 `PARITY: ...` 한 줄을 낸다.

**`ANDROID_SERIAL` 을 주지 않아도 되는 것은 기기가 한 대일 때뿐이다.** 여러 대가 붙어 있고
고르지 않으면, 스크립트는 예전처럼 도중에 `adb: more than one device/emulator` 로 죽지 않고
**시작 전에 목록을 대고 거절한다.**

### `parity` 는 자기 호스트 아티팩트를 직접 만든다

**`install_shim.sh` 가 설치한 것을 재지 않는다.** §5.2 의 이유로 호스트 쪽을 `accelerate`
없이 따로 빌드해서(`$CARGO_TARGET_DIR-hostgemm`, `PARITY_HOST_TARGET_DIR` 로 덮어쓸 수 있음)
`$TMPDIR/bw_parity_host` 에 벤더 트리를 심볼릭 링크로 비추고 `_C.abi3.so` **하나만** 그
빌드로 바꿔 임포트한다. 벤더 트리에 설치된 아티팩트는 건드리지 않는다 — 그것을 잠깐
바꿔치기했다가 중간에 죽으면 `run.sh` 의 도로 테스트 넷이 잘못된 것을 읽게 된다
(`docs/graph/CAPTURE.md` §8).

같은 이유로 **기기 쪽도 조용히 낡을 수 없게** 했다. `parity` 는 시작할 때
`$DEVICE_ROOT/site/torch/_C.abi3.so` 의 md5 를 방금 빌드한 `lib_C.so` 와 대조하고, 다르면
두 해시를 대고 거절한다. `stage` 를 잊은 채 며칠 전 산출물에 대해 `PARITY: ok` 가 나오는
것이 이 검사의 최악의 실패 모드다.

### 판정은 허용치가 아니라 **이름 기반 면제 목록**이다

`cmd_diff` 는 `EXPECTED_LIBM_DIVERGENCE` 바깥의 어떤 비트 차이도 실패로 본다. §5.3 이후
넷이다 — `_softmax.default` · `tanh.default` · `exp.default` · `softplus.default`, 전부
1 ULP, 전부 `expf` 를 부르는 경로. 전역 허용치를 두면 `mm` 이나 `cumsum` 의 진짜 불일치까지
함께 삼키는데, 이 스크립트는 바로 그것을 잡으려고 있다. **이 목록이 짧게 유지되는 것이
§5.2 의 설계를 고른 이유**이고, 늘어나기 시작하면 그것이 허용오차다 — 늘 때마다 §5.3 처럼
배정밀도 기준과 대조해 방향이 없는 발산인지 재고 나서 늘렸다.

**계측기가 실제로 울리는지 확인했다** (`device_android.sh diff <host> <doctored>`,
2026-08-25 재확인):

| 대조군 | 결과 |
|---|---|
| 면제 목록에 없는 `mm.default` 를 1 ULP 틀어놓음 | `EXIT=1  PARITY: unexpected bit divergence: ['mm.default']` |
| 면제된 `tanh.default` 를 5 ULP 로 키움 | `EXIT=1  PARITY: unexpected bit divergence: ['tanh.default']` |
| 손대지 않은 원본 | `EXIT=0  PARITY: ok` |
| `torch_c_no_accelerate` cfg 이름을 어긋나게 함 | `EXIT=1  refusing to measure ...: it still links Accelerate` |
| 기기에 다른 빌드를 올려둔 채 `parity` | `EXIT=1  refusing to measure: the staged _C.abi3.so is not ...` |

bionic 쪽이 나중에 고쳐지면 면제 항목이 남아돌게 되는데, 그때는 실패가 아니라
`note: expected-divergence entries that now agree: [...]` 로 알린다.

> **이 절을 쓰는 과정에서 계측기를 한 번 잘못 읽었다.** `echo "$(basename $f) EXIT=$?"`
> 로 종료 코드를 찍었는데 명령 치환이 `$?` 보다 먼저 실행되어 **세 대조군이 전부 `EXIT=0`**
> 으로 보였다. 스크립트는 처음부터 옳았고 재는 쪽이 틀렸다. 이 저장소에서 같은 종류의
> 계측 오류가 반복되고 있다 — 종료 코드는 다른 무엇을 하기 전에 변수에 먼저 담아라.

## 10. 이 검증이 답하지 않은 것

- **APK 안에서의 로드.** `/data/local/tmp` 의 `adb shell` 실행 경로만 다뤘다. 앱 전용
  저장소 + 임베딩 인터프리터는 권한 모델·SELinux 컨텍스트·`argv[0]` 이 모두 다르다.
  특히 §4 의 "`PYTHONHOME` 불필요" 결론은 그쪽으로 넘어가지 않는다.
- **API 26 하나뿐이다.** `pmp_api36` 에서는 시도하지 않았다. 16 KB 페이지처럼 API 레벨에
  따라 갈리는 지점은 범위 밖이다.
- **실물 단말 없음.** 전부 에뮬레이터다.
- **autograd·학습 없음.** 순전파만이다. `nn.Linear`/`nn.Sequential` 도 `no_grad` 아래에서
  가중치를 채워 넣고 forward 만 돌렸다.
- **큰 텐서·메모리 압박 없음.** 최대 64×64 다. 폰의 메모리 한계에 대해 아무것도 말하지 않는다.
- **`float64`/`float16`/`bfloat16` 미검증.** 배터리는 전부 f32 다.
- **stdlib 279 MB 를 그대로 올렸다.** 배포 크기 문제는 손대지 않았다.

## 11. 다음에 무엇을 해야 다음 단계가 열리는가

1. **`rust/torch_c/src/overloads.json` 에 접미사 없는 스펠링을 채운다** (§7). `torch.relu`
   하나가 아니라 `F.*` 가 부르는 bare 스펠링 전반의 문제일 가능성이 높다 — `nn` 모듈이
   순전파에서 어떤 bare 스펠링을 부르는지 세어보는 것이 먼저다.
2. **`_multiprocessing` 부재를 어디서 처리할지 정한다** (§6). 지금은 계측용 런타임 스텁이라
   앱에는 그대로 못 넣는다.
3. **APK 경로 검증** (§10 첫 항목). `PythonMultiplatform` 의 샘플 앱 배선이 참고 대상이다.
4. **dtype 확대와 큰 텐서.** 지금 배터리는 f32·64×64 까지만 말한다.

---

## 10. Vulkan 은 실기 없이도 열린다 — 꺼져 있었을 뿐이다 (2026-08-25)

§5 까지의 작업은 API 26 AVD (`pmp_api26`) 에서 했고, 거기서 `pm list features` 에 vulkan 항목이
**하나도** 없었습니다. 그것을 근거로 "Vulkan 은 실기가 필요하다" 고 판단했는데 **틀렸습니다.**

원인은 하드웨어가 아니라 설정입니다:

```
~/.android/avd/pmp_api26.avd/config.ini   hw.gpu.enabled = no
~/.android/avd/pmp_api36.avd/config.ini   hw.gpu.enabled = no
```

**두 AVD 모두 GPU 를 꺼둔 상태였습니다.** 이미지에 없는 것이 아니라 켜지지 않았을 뿐입니다.

### 실측 — API 36 을 `-gpu host` 로 띄운 결과

```
$ emulator -avd pmp_api36 -gpu host -no-audio -no-snapshot -port 5556

feature:android.hardware.vulkan.compute            ← 컴퓨트 셰이더
feature:android.hardware.vulkan.level=1
feature:android.hardware.vulkan.version=4206592    = VK_MAKE_VERSION(1, 3, 0)
feature:android.software.vulkan.deqp.level=132711169
```

게스트 확장에 `ANDROID_EMU_vulkan`, `ANDROID_EMU_deferred_vulkan_commands`,
`ANDROID_EMU_vulkan_async_queue_submit`, 그리고 `ANDROID_EMU_vulkan_shader_float16_int8`
이 있습니다.

> **정정 (2026-08-25).** 이 절의 초판은 마지막 항목을 근거로 *"fp16·int8 셰이더가 되니
> 양자화 경로까지 시험 대상"* 이라고 썼습니다. **너무 강한 주장이었습니다.** `docs/devices/VULKAN.md`
> 의 프로브가 어댑터를 **`"Apple M1"`** 으로, wgpu 쪽은 드라이버를 **`MoltenVK`** 로 보고합니다 —
> 저 피처 비트는 gfxstream 이 **호스트의 것을 그대로 전달한 것**입니다. Adreno 나 Mali 가
> `shaderFloat16` 을 주는지는 **여기서 답할 수 없습니다.** 양자화 경로는 여전히 열린 문제입니다.
>
> 같은 절에서 배운 두 번째 것: **확장 문자열을 세면 거짓 음성이 난다.** `VK_KHR_8bit_storage`
> 는 "absent" 로 나오는데 `storageBuffer8BitAccess` 는 YES 입니다 — 코어로 승격된 확장은
> 목록에 안 나옵니다. `Vulkan11Features`/`Vulkan12Features` 를 조회해야 합니다.

드라이버는 `ro.hardware.vulkan = ranchu` (에뮬레이터의 virtio-gpu) 로, **게스트 Vulkan 을 호스트로
번역**합니다.

### 그래서 무엇이 열리고 무엇이 안 열리는가

| | 지금 가능 | 근거 |
|---|---|---|
| Vulkan 경로가 도는지 | **예** | 실제로 돌렸다 — `docs/devices/VULKAN.md` |
| 양자화(fp16·int8) 셰이더가 실기에서 되는지 | **아니오** | 위 정정 참고 — 호스트 비트다 |
| 값이 CPU 와 맞는지 | **예** | §4 의 비트 단위 대조 방식 그대로 |
| ExecuTorch `vulkan` 델리게이트 배선 | **예** | |
| **성능이 Adreno·Mali 를 대표하는가** | **아니오** | `ranchu` 가 호스트 GPU 로 번역한다 |

Apple Silicon 호스트에서 `-gpu host` 는 결국 Metal 로 갑니다. **정확성과 통합은 여기서 끝낼 수
있고, "GPU 가 디코딩에서 이기는가" 만 실기 질문으로 남습니다.**

`docs/perf/PERF.md` §7.3 이 Apple 에서 잰 것 — 디코딩 모양의 행렬×벡터는 n=4096 까지 GPU 가 진다 —
은 그 실기 측정의 **가설**이지 답이 아닙니다. 모바일 GPU 는 CPU 대비 우위가 데스크톱과 다릅니다.

### 주의

`pmp_api36` 에도 `pmp`, `pmp-nativetest-arm64-v8a` 디렉터리가 있습니다 — **다른 프로젝트가
같은 AVD 를 씁니다.** §1 의 규칙대로 `/data/local/tmp/bw_device` 밖으로 나가지 마십시오.
API 26 쪽 작업을 내리지 않고 포트 5556 에 나란히 띄웠습니다.
