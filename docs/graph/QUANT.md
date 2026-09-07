# 양자화 — 이 스택에서 실제로 무엇을 주는가

측정일 2026-08-28. 브랜치 `work/quant`. 호스트 Apple M1 (P 4 + E 4, macOS darwin 25.5.0),
CPython 3.13.0, 상류 torch 2.13.0, candle-core 0.11.0.
기기 `emulator-5554` (`pmp_api26`, API 26, arm64-v8a, vCPU 4).
**벤더링 트리는 한 줄도 고치지 않았고, 저장소 코드도 고치지 않았습니다.** 이 문서는 측정만 담습니다.

> **후속.** `docs/numerics/DTYPE.md` 가 §3 의 "축소 정밀도가 느린 이유" 를 층별로 분해하고 그 중
> 셋을 고쳤습니다. `docs/graph/QUANT2.md` 가 **§8 6 번(양자화용 검증 축)을 닫고**, `Repr` 세 번째
> 변형과 모듈 교체 경로를 착지시켰습니다 — 아래 §8 의 우선순위는 그 두 문서로 갱신되었습니다.
> 특히 **§8 2 번의 "가중치 7.8× 압축" 은 KleidiAI 경로에서만 참**이고(DTYPE.md §6.1),
> **§7 의 "검증 축이 지금 저장소에 없다" 는 더 이상 참이 아닙니다**(QUANT2.md §2).

> **결론 먼저.**
>
> 1. **dtype 을 낮추면 이 스택은 느려집니다. 빨라지지 않습니다.** `float16`·`bfloat16` 연산은
>    전부 `f32` 로 넓혀서 계산하고 다시 좁힙니다(`opmath_in`). 그래서 **커널은 f32 커널 그대로이고
>    변환 비용만 추가**됩니다. 호스트에서 `mm` 1024 는 f32 1.91 ms → bf16 2.27 ms → f16 3.59 ms 로
>    **단조 증가**합니다. 디코딩 모양에서는 더 나빠서 f16 이 f32 의 **27 배**입니다.
> 2. **`int8` 은 미구현이 아니라 저장이 불가능합니다.** candle-core 0.11 의 `DType` 에 `I8` 이
>    없어서 `torch.int8` 텐서를 **만들 수조차 없습니다**(`quint8`·`qint8` 도 같음, 실측 §2).
>    정수 양자화는 op 을 더 쓰는 문제가 아니라 **백엔드 저장 타입이 없는 문제**입니다.
> 3. **그런데 양자화 자체는 이 하드웨어에서 크게 이깁니다 — 단, f32 가 AMX 로 가지 않는 곳에서만.**
>    상류의 KleidiAI 4-bit 는 Apple 에서 **디코딩 6.6 배**를 주지만 **프리필에서는 8.3 배 집니다**
>    (f32 가 AMX 로 가므로). AMX 가 없는 쪽(안드로이드·`gemm` 백엔드)에서는 **프리필에서도 이깁니다** —
>    기기에서 candle Q4K 가 f32 대비 **3.3 배**(§5.2).
> 4. **`docs/perf/PERF_ANDROID.md` §3 이 f32 에 대해 내린 "컴파일 플래그는 문이 아니다" 는 결론이
>    양자화에는 그대로 적용되지 않습니다.** `gemm` 은 런타임 검출이지만 **candle 의 양자화 커널은
>    컴파일타임 `#[cfg(target_feature = "dotprod")]`** 으로 게이팅됩니다. 그리고
>    `aarch64-linux-android` 와 `aarch64-apple-ios` 의 기본 target feature 는 **`neon` 하나뿐**입니다.
>    즉 **지금 안드로이드와 iOS 는 `sdot` 없는 폴백 경로를 컴파일하고 있고**, 플래그 한 줄이
>    기기에서 **2.05 배**입니다(§6).

---

## 0. 이 문서의 숫자로 하면 안 되는 것

- **에뮬레이터는 실기가 아닙니다.** 게스트 `/proc/cpuinfo` 의 Features 는
  `fp asimd evtstrm aes pmull sha1 sha2 crc32` 뿐으로 **`asimddp`(dotprod) 를 광고하지 않습니다.**
  같은 기기에서의 A/B 비율만 의미가 있고, 절대값은 실기가 아닙니다.
- **기기의 디코딩(gemv) 수치는 재현되지 않았습니다.** `1x4096x4096/Q4K` 가 같은 바이너리 4 회에서
  0.095 ~ 0.700 ms 를 오갔습니다(7 배). HVF 게스트의 메모리 경로가 흔들립니다. **그래서 기기
  결론에는 프리필 수치만 씁니다** — 프리필은 회차 간 2% 이내로 재현됩니다.
- **§7 의 정확도 수치는 무작위 가우시안 가중치입니다.** 실제 모델 가중치가 아니므로
  **오차의 상한이지 대표값이 아닙니다.** 실모델 평가는 하지 않았습니다.
- **`_dyn_quant_matmul_4bit` 의 출력값을 검증하지 않았습니다.** 무작위 uint8 을 4-bit 가중치로
  먹여 **시간만** 쟀습니다. 커널이 무엇을 계산하는지는 이 문서가 확인한 것이 아닙니다.

---

## 1. 측정 조건

**단독 실행입니다.** 다른 에이전트도 동시 빌드도 없습니다. 각 측정 직전·직후에 `uptime` 을
기록했습니다.

| 측정 | load (1분) |
|---|---|
| 호스트 dtype 스윕 (시임·상류·no-accel) | 1.34 ~ 1.81 |
| 상류 KleidiAI 4-bit | **0.86** |
| candle 양자화 (호스트) | 1.46 |
| candle 양자화 (기기, 2 회차) | 1.74 ~ 1.92 |

- **반복 15 회, 워밍업 3 회, 보고값은 최솟값**입니다. 최솟값이 다른 부하에 가장 덜 오염됩니다.
- **전부 단일 스레드**입니다 (`RAYON_NUM_THREADS=1`, 상류는 `torch.set_num_threads(1)`).
  `docs/perf/PERF_ANDROID.md` §0 이 단일 스레드 수치만 재현된다고 기록했고, 이번에도 그랬습니다 —
  회차 간 편차가 대부분 **2% 미만**입니다(예외 두 개는 아래).
- **스위트를 필터링하지 않았습니다.** 전체를 그대로 돌린 값입니다.
- 연산은 양쪽 다 `torch.ops.aten.*` 로 불러 스펠링 차이가 섞이지 않게 했습니다.

**재현성 실측** (호스트, 시임, 같은 바이너리 2 회차):

```
mm512/f32   0.2083 / 0.2087   0.2%       mv2048/f16  3.0693 / 3.0663   0.1%
mm512/bf16  0.3005 / 0.3007   0.1%       cvt1024/bf16 0.0717 / 0.0730  1.8%
mm1024/f32  1.9132 / 1.7276  10.7%  <-   add1024/f32  0.1268 / 0.1137 11.5%  <-
```

**흔들리는 두 칸은 `mm1024/f32` 와 `add1024/f32` 뿐**이고 둘 다 f32 입니다. 이 문서의 결론은
전부 dtype 간 **배수**에 걸려 있고 그 배수는 3 배 이상이므로, 10% 흔들림이 결론을 바꾸지 않습니다.

**검증된 기준선** (이 측정을 시작한 트리 상태, 전부 exit 0):

```
sh rust/torch_c/pytests/run.sh        171 통과, 실패 0
tools/golden/compare.py               2744/2744, 실패 0, ops covered=118
rust/torch_c/pytests/verify_schemas.py 3076/3076
git status --short                    (비어 있음)
```

> **함정 하나 기록.** `vendor/install_shim.sh` 와 `pytests/run.sh` 는 `cargo` 를 부르는데,
> `PATH` 에 `~/.cargo/bin` 이 없으면 **exit 127 로 죽고 낡은 산출물이 남습니다.** 이번에 두 번
> 걸렸습니다. 파이프로 종료 코드를 읽었다면 성공으로 보였을 것입니다.

---

## 2. 지금 무엇이 되고 무엇이 안 되는가 — dtype 별

### 2.1 저장 가능한 dtype 은 9 종 (스펠링 16 개)

`torch` 가 이름을 가진 dtype 55 개 각각에 `aten.full.default` 를 실제로 호출해 봤습니다.

| | dtype |
|---|---|
| **저장 가능 (9)** | `float32` · `float64` · `float16` · `bfloat16` · `int16` · `int32` · `int64` · `uint8` · `bool` |
| **거부 (39)** | `int8` · `qint8` · `quint8` · `quint4x2` · `quint2x4` · `qint32` · `uint16` · `uint64` · `complex*` (3) · `float8_*` (4) · `float4_e2m1fn_x2` · `bits*` (5) · sub-byte `int1..7`/`uint1..7` (14) |
| **미측정 (1)** | `float8_e4m3fn` — 독립 프로브에서 **무한 대기**하는 것이 이미 알려져 있어 건너뛰었습니다 (`tools/golden/dtypes.py` 의 기록) |

거부는 전부 같은 한 줄입니다:

```
NotImplementedError: aten.full.default: dtype not storable by the candle backend
                     in torch._C shim: torch.int8
```

**원인은 op 커버리지가 아니라 백엔드 저장 타입입니다.** `dtype.rs` 의 `storage()` 가 유일한
관문이고, candle-core 0.11 의 `DType` 열거형에는 `U8, U32, I16, I32, I64, BF16, F16, F32, F64,
F8E4M3` 과 MX 계열뿐 — **`I8` 이 없습니다.**

    따라서 int8 양자화는 "아직 구현하지 않은 op" 이 아닙니다.
    **int8 텐서를 담을 그릇이 백엔드에 없습니다.**

`torch.quantize_per_tensor`·`dequantize`·`_int_mm`·`_weight_int8pack_mm`·
`_dyn_quant_pack_4bit_weight`·`_dyn_quant_matmul_4bit` 은 **하나도 구현되어 있지 않습니다**
(스키마는 벤더 트리에서 전부 정확히 답합니다 — 이름과 시그니처는 알고 커널이 없는 상태).
`TensorBase.is_quantized` 는 **무조건 `False`** 를 돌려주고, `torch._C._has_kleidiai` 는
**하드코딩 `False`** 입니다(`bootstrap.py:3308`, 근거 주석 있음). `Repr` 열거형은
`Dense(candle Tensor)` 와 `Meta` 둘뿐이라 **양자화 텐서가 들어갈 자리가 타입 수준에 없습니다.**

### 2.2 op 별 dtype — reduced float 에 구멍이 **없습니다**

`_aten_implemented()` 의 118 개 각각에 대해 벤더 트리의 스키마를 읽어 인자를 합성하고,
dtype 을 바꿔가며 실제로 호출했습니다. **한 dtype 도 성공하지 못한 op 은 "인자 합성 실패" 로
분류하고 dtype 판정에서 제외**했습니다 — 그래야 내 프로브의 한계를 op 의 한계로 오독하지 않습니다.

```
전체 118  ->  프로브 성립 78 · 인자 합성 실패 40
```

프로브가 성립한 78 개에 대해:

| dtype | ok | NotImplemented | 기타 거절 |
|---|---:|---:|---:|
| `float32` | 68 | 0 | 10 |
| `float64` | 68 | 0 | 10 |
| **`float16`** | **68** | **0** | **10** |
| **`bfloat16`** | **68** | **0** | **10** |
| `int64` / `int32` / `int16` | 63 | 7 | 8 |
| `uint8` | 65 | 7 | 6 |
| `bool` | 55 | 19 | 4 |
| `int8` / `qint8` / `quint8` | — | — | **텐서 생성 자체가 불가 (§2.1)** |

**`float32` 를 받고 `float16` 을 거부하는 op 이 0 개, `bfloat16` 을 거부하는 op 도 0 개입니다.**
f32 에서 실패한 10 개는 dtype 구멍이 아니라 **의미론적 거절**입니다 — `bitwise_*` 가 float 을
거부하고, `masked_fill`·`masked_select`·`where` 가 bool 마스크를 요구합니다. 상류와 같은 거절입니다.

    즉 **reduced float 의 표면 커버리지는 이미 완전합니다.**
    문제는 "받느냐" 가 아니라 "받아서 무엇을 하느냐" 입니다 — §3.

> 이 표로 **알 수 없는 것**: 인자 합성이 실패한 40 개(`bmm`·`baddbmm`·`convolution`·in-place
> 계열 등)의 dtype 거동. 이들은 3D 입력이나 특수 인자가 필요해 일반 합성이 닿지 않았습니다.
> **f16/bf16 구멍이 그 40 개 안에 숨어 있을 가능성은 배제하지 못했습니다.**

---

## 3. dtype 을 낮추면 빨라지는가 — **아니오, 느려집니다**

### 3.1 이유는 소스에 적혀 있습니다

`aten.rs` 의 `opmath_in` 이 상류의 `at::opmath_type` 을 구현합니다 — reduced float 은 **f32 로
넓혀서 계산하고 마지막에 한 번 narrow** 합니다. `gemm_accumulate_in` 이 그것을 부르고,
`add`/`sub`/`mul`/`div`/`sum`/`mean` 도 같은 규칙을 씁니다. `docs/numerics/BF16.md` 가 이 규칙을 **수치
정확성을 위해** 도입했고, 그 판단은 옳습니다(로짓 오차 11.75 → 0.47).

**그 대가가 성능이고, 지금까지 아무도 재지 않았습니다.**

### 3.2 호스트, 시임 (배송 설정 = `accelerate` 켬), 단일 스레드, 15 회 최솟값

| case | f32 (ms) | bf16 (ms) | f16 (ms) | bf16/f32 | f16/f32 |
|---|---:|---:|---:|---:|---:|
| `mm` 128 | 0.0059 | 0.0130 | 0.0363 | **2.20×** | **6.15×** |
| `mm` 256 | 0.0317 | 0.0551 | 0.1367 | **1.74×** | **4.31×** |
| `mm` 512 | 0.2083 | 0.3005 | 0.6275 | **1.44×** | **3.01×** |
| `mm` 1024 | 1.9132 | 2.2741 | 3.5851 | **1.19×** | **1.87×** |
| `mv` 1024 (디코딩) | 0.0255 | 0.1006 | 0.6977 | **3.95×** | **27.4×** |
| `mv` 2048 (디코딩) | 0.2730 | 0.7130 | 3.0693 | **2.61×** | **11.2×** |
| `add` 1024 | 0.1268 | 0.5440 | 1.8589 | **4.29×** | **14.7×** |

**전부 1 보다 큽니다. dtype 을 낮춰서 빨라지는 칸이 하나도 없습니다.**
그리고 **디코딩 모양에서 가장 나쁩니다** — 자기회귀 생성이 실제로 도는 그 모양입니다.

### 3.3 변환 비용이 그 전부입니다 — 분해해서 확인

`f32` 로 넓히는 비용만 따로 쟀습니다:

| | 512×512 | 1024×1024 |
|---|---:|---:|
| `bf16 -> f32` | 0.0193 ms | 0.0717 ms |
| `f16 -> f32` | **0.1685 ms** | **0.6802 ms** |
| | **8.7 배** | **9.5 배** |

`mm` 512 를 이것으로 재구성하면:

```
bf16   0.2083 (f32 GEMM) + 2 x 0.0193 (입력 2 개 넓히기) + narrow   =~ 0.27   실측 0.3005
f16    0.2083 (f32 GEMM) + 2 x 0.1685                    + narrow  =~ 0.71   실측 0.6275
```

**모형이 맞습니다.** f16 `mm` 512 시간의 약 2/3 가 dtype 변환입니다.
`bf16 -> f32` 는 비트 시프트라 싸고, `f16 -> f32` 는 그렇지 않습니다.

### 3.4 세 dtype 이 같은 커널을 탄다는 직접 증거

`accelerate` 를 끈 빌드로 같은 것을 다시 쟀습니다 (백엔드만 바뀌고 소스는 동일):

| case | accelerate 켬 | accelerate 끔 | 끔/켬 |
|---|---:|---:|---:|
| `mm` 512 f32 | 0.2083 | **2.8383** | 13.6× |
| `mm` 512 bf16 | 0.3005 | **2.9334** | 9.8× |
| `mm` 512 f16 | 0.6275 | **3.0900** | 4.9× |

**`accelerate` 를 끄면 세 dtype 이 2.84 ~ 3.09 ms 로 모입니다.** f32 GEMM 을 느리게 만들면
f16·bf16 도 정확히 같은 만큼 느려집니다 — **셋이 한 커널을 탄다는 뜻**이고, 남는 차이가 정확히
§3.3 의 변환 비용입니다.

그리고 그 f32 값이 독립 검증이 됩니다: `2 × 512³ / 2.8383 ms = 94.6 GFLOP/s`,
`docs/perf/PERF_ANDROID.md` §1 이 따로 잰 **88 ~ 93 GFLOP/s** 와 같은 자리입니다.

### 3.5 심볼 수준 확인 — f16 커널은 아예 링크되지도 않습니다

`docs/perf/PERF_ANDROID.md` §3.1 은 `gemm` 크레이트의 fp16 마이크로커널이 산출물 안에 있다는 것을
확인했습니다. **그런데 시임은 그것을 절대 부르지 않습니다.**

```
호스트 lib_C.dylib, accelerate 켬 (배송)     gemm_f16 심볼   0 개
호스트 lib_C.dylib, accelerate 끔            gemm_f16 심볼  27 개
```

배송 빌드에서 **0 개**입니다. candle 의 accelerate matmul 은 `DType::F16` 에서
`bail!("the accelerate backend does not support f16 matmul")` 로 즉시 거절하고(`cpu_backend/mod.rs:1497`),
`BF16` 은 `UnsupportedDTypeForOp` 로 떨어집니다 — **accelerate 백엔드의 matmul 은 F32 와 F64
둘뿐입니다.** 시임이 그 전에 f32 로 넓혀 두므로 그 거절 경로는 실행되지 않고, 결과적으로 f16
GEMM 코드가 도달 불가가 되어 링커가 걷어냅니다.

**존재하는 저정밀 커널을 우리가 우회하고 있습니다.**

### 3.6 상류와의 대조 — 상류도 여기서 좋지 않습니다

| case | 시임 | 상류 2.13.0 | 시임/상류 |
|---|---:|---:|---:|
| `mm` 512 f32 | 0.2083 | 0.2026 | 1.03× |
| `mm` 1024 f32 | 1.9132 | 1.7127 | 1.12× |
| `mm` 512 f16 | 0.6275 | **89.17** | **0.007×** |
| `mm` 512 bf16 | 0.3005 | **89.14** | **0.003×** |
| `mm` 1024 f16 | 3.5851 | **727.7** | **0.005×** |
| `mv` 1024 f16 | 0.6977 | 0.0952 | 7.33× |
| `add` 1024 f16 | 1.8589 | 0.0680 | 27.3× |

**상류의 f16/bf16 `mm` 은 3 GFLOP/s 로 무너집니다** — f32 의 400 배 느립니다. 이 호스트의 상류
빌드에 reduced-float BLAS 가 없어서 참조 루프로 떨어지는 것으로 보입니다(**추론입니다 — 상류
커널을 프로파일하지 않았습니다**). 즉 **큰 `mm` 에서는 우리가 상류보다 100 배 이상 빠릅니다.**

하지만 그것이 우리가 잘한다는 뜻은 아닙니다 — `mv`·`add` 처럼 상류가 제대로 된 벡터화 커널을
가진 자리에서는 **우리가 7 ~ 27 배 집니다.** 두 결과가 같은 원인을 가리킵니다:
**우리는 reduced float 에 전용 경로가 없습니다.**

---

## 4. 상류는 무엇을 하는가 — KleidiAI 4-bit

`docs/perf/PERF_ANDROID.md` §2 가 "KleidiAI 가 걸리는 자리는 `aten._dyn_quant_pack_4bit_weight`
하나" 라고 기록했습니다. **그 게이트가 무엇을 요구하는지 읽고, 이 호스트에서 실제로 쟀습니다.**

### 4.1 요구 사항 — 새 dtype 이 필요 없습니다

`torch/_meta_registrations.py:4262` 의 계약:

| 인자 | 타입 | 이 스택에서 |
|---|---|---|
| `weights` | **`torch.uint8`** (4-bit 두 개를 한 바이트에 팩) | **저장 가능** |
| `scales_zeros` | `float32` (블록=행 전체) 또는 `bfloat16` (블록별, `block_size % 32 == 0`) | **둘 다 저장 가능** |
| `inp` / 출력 | `float32` 또는 `bfloat16` | **둘 다 저장 가능** |
| 팩된 가중치 | KleidiAI 있으면 불투명 `uint8` 블롭, **없으면 평범한 `float` 텐서** (`weights.numel() + scales_zeros.numel()`) | **저장 가능** |

**이것이 이 조사에서 가장 실행 가능한 발견입니다.** 상류의 4-bit 경로는 `int8` 도 `qint8` 도
쓰지 않습니다 — **`uint8` + 스케일**이고, 그 셋 다 이 스택이 이미 담을 수 있습니다.
그리고 **KleidiAI 가 없을 때의 팩 레이아웃은 단순 연결(concat)** 이라 그대로 재현 가능합니다.
§2 의 `I8` 벽을 **건드리지 않고** 우회할 수 있는 유일한 경로입니다.

### 4.2 이 호스트의 상류에는 KleidiAI 가 **켜져 있습니다** — 그래서 천장을 직접 쟀습니다

```
upstream torch._C._has_kleidiai        = True      (시임은 하드코딩 False)
```

| 모양 | 상류 f32 `mm` (ms) | 상류 4-bit (ms) | 배수 |
|---|---:|---:|---:|
| **디코딩** `1×1024×1024` | 0.0256 | 0.0167 | **1.53× 빠름** |
| **디코딩** `1×2048×2048` | 0.3383 | 0.0652 | **5.19× 빠름** |
| **디코딩** `1×4096×4096` | 1.6397 | 0.2479 | **6.61× 빠름** |
| **프리필** `128×1024×1024` | 0.2087 | 1.7713 | **8.49× 느림** |
| **프리필** `512×512×512` | 0.2027 | 1.7813 | **8.79× 느림** |
| **프리필** `1024×1024×1024` | 1.7510 | 14.447 | **8.25× 느림** |

**같은 커널이 디코딩에서 6.6 배 이기고 프리필에서 8.3 배 집니다.**

설명은 `docs/perf/PERF_ANDROID.md` §2 가 이미 준비해 두었습니다 — **f32 쪽이 AMX 로 갑니다.**
프리필에서 상대는 1300 GFLOP/s 짜리 행렬 코프로세서이고, KleidiAI 의 4-bit 커널은 NEON 입니다.
디코딩은 연산이 아니라 **메모리 대역폭**에 걸리므로 압축률이 그대로 이득이 됩니다:

```
가중치 1024×1024   f32 4.19 MB  ->  4-bit 팩 0.537 MB   = 7.8 배 작음
```

`block_size` 를 32 로 줄여도(그룹별 bf16 스케일) 시간은 거의 같고 크기만 11% 늘어납니다
(0.537 → 0.598 MB). **정확도-크기 절충이 시간을 거의 안 씁니다.**

> **이 표가 뜻하지 않는 것.** 4-bit 커널의 출력을 검증하지 않았습니다(무작위 `uint8` 입력).
> 그리고 이것은 **Apple 의 그림**입니다 — AMX 가 없는 타깃에서는 프리필 칸의 부호가 바뀝니다(§5).

---

## 5. candle 은 무엇을 주는가

### 5.1 형식과 구조

candle-core 0.11 의 양자화는 **`DType` 이 아니라 별도의 타입 체계**입니다:
`QTensor` + `GgmlDType` (`Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q8_1 Q2K Q3K Q4K Q5K Q6K Q8K`) — GGML/GGUF
k-quant 입니다. `QMatMul` 이 그것을 소비합니다. 읽기 경로(`gguf_file`·`ggml_file`)도 있습니다.

**이것이 §2 의 `I8` 벽과 무관하게 존재한다**는 점이 중요합니다. 양자화 가중치는 `DType` 을
거치지 않으므로 `I8` 이 없어도 됩니다. 대신 **`QTensor` 는 `Tensor` 가 아니어서** 시임의
`Repr::Dense(Tensor)` 에 들어가지 않습니다 — 세 번째 variant 가 필요합니다.

`candle-transformers` 는 이 저장소가 의존하지 않습니다(`DESIGN.md` §1 이 거절). 필요한 것은
`candle-core` 안에 다 있습니다.

### 5.2 성능 — f32 가 AMX 로 가지 않는 곳에서는 **프리필에서도 이깁니다**

`candle-core` 만 쓰는 별도 바이너리로 쟀습니다 (`default-features = false`, 즉 **accelerate 없음
= 안드로이드와 같은 f32 백엔드**). 15 회 최솟값, 단일 스레드.

**호스트 M1** (f32 기준선 = `gemm` NEON 경로):

| 모양 | f32 | Q4K (`+dotprod`) | 배수 | Q4K (dotprod 없음) | 배수 |
|---|---:|---:|---:|---:|---:|
| 디코딩 `1×4096×4096` | 1.3324 | **0.0891** | **15.0×** | 0.1645 | 8.3× |
| 프리필 `1024×1024×1024` | 22.609 | **6.187** | **3.65×** | 11.111 | 2.05× |

**기기 `emulator-5554`** (프리필만 — 디코딩은 재현 안 됨, §0). 2 회차 교대, 편차 2% 미만:

| 모양 | f32 | Q4K (dotprod 없음 = **현재 배송**) | Q4K (`+dotprod`) | Q4_0 | Q8_0 |
|---|---:|---:|---:|---:|---:|
| `512×512×512` | 2.970 | 1.913 (**1.55×**) | **0.931 (3.19×)** | 4.20 (0.71× 느림) | 3.34 (0.89× 느림) |
| `1024×1024×1024` | 23.87 | 14.91 (**1.60×**) | **7.26 (3.29×)** | 34.5 (0.69× 느림) | 25.8 (0.92× 느림) |

세 가지가 읽힙니다.

1. **AMX 가 없으면 양자화가 프리필에서도 이깁니다** — 기기에서 Q4K 가 3.29 배.
   `docs/perf/PERF_ANDROID.md` §8 이 "f32 NEON 으로 회수할 여지는 최대 1.14 배" 라고 닫아둔 그 자리에서,
   **dtype 쪽으로는 3.29 배가 남아 있습니다.** 그 문서의 가설이 수치로 확인됩니다.
2. **`Q4_0` 은 프리필에서 f32 보다 느립니다** (0.69×). 형식 선택이 공짜가 아닙니다 — **`Q4K` 를
   써야 합니다.** `Q8_0` 도 프리필에서는 거의 이득이 없습니다(0.92×).
3. **기기 f32 1024³ = 23.87 ms 가 호스트 no-accel 22.61 ms 와 같습니다.** 서로 다른 두 바이너리가
   `docs/perf/PERF_ANDROID.md` §1 의 "기기와 호스트의 f32 GEMM 처리량이 동일" 을 재확인합니다.

---

## 6. 컴파일 플래그가 **여기서는** 문이다

`docs/perf/PERF_ANDROID.md` §3 은 `-C target-cpu` / `-C target-feature` 가 아무것도 바꾸지 않는다는
것을 소스와 측정 양쪽에서 확인했습니다. **그 결론은 f32 `gemm` 에 대해 옳고, 지금도 옳습니다** —
`gemm` 0.19 는 `is_aarch64_feature_detected!` 로 **런타임 검출**합니다.

**candle 의 양자화 커널은 그렇지 않습니다.**

```rust
// candle-core-0.11.0/src/quantized/neon.rs:17
unsafe fn vdotq_s32(a: int8x16_t, b: int8x16_t) -> int32x4_t {
    #[cfg(target_feature = "dotprod")]        // <- 컴파일타임
    { ... asm!("sdot {acc:v}.4s, ...") ... }
    #[cfg(not(target_feature = "dotprod"))]
    { let p0 = vmull_s8(...); ... }           // <- 곱셈-누산 에뮬레이션
}
```

`neon.rs` 안에 이런 게이트가 **8 개**입니다. 그리고 각 타깃의 기본 target feature 는:

```
aarch64-apple-darwin    aes crc dit dotprod dpb dpb2 fcma fhm flagm fp16 ...
aarch64-linux-android   neon                          <- 이것뿐
aarch64-apple-ios       aes neon pmuv3 sha2           <- dotprod 없음
```

**안드로이드와 iOS 는 `sdot` 없는 폴백을 컴파일합니다. macOS 만 빠른 경로를 받습니다.**
그리고 그 차이를 기기에서 쟀습니다:

```
기기 Q4K 1024³      dotprod 없음 14.91 ms  ->  +dotprod 7.26 ms     2.05 배
기기 Q4K 512³       dotprod 없음  1.913 ms ->  +dotprod 0.931 ms    2.05 배
호스트 Q4K 1024³    dotprod 없음 11.11 ms  ->  +dotprod 6.19 ms     1.80 배
```

**플래그 한 줄이 2 배입니다.** f32 에서는 0 이었던 그 줄이, 양자화에서는 지렛대입니다.

> **다만 그냥 켤 수는 없습니다.** `dotprod` 는 ARMv8.2-A 확장이고 모든 arm64 안드로이드 기기에
> 있지 않습니다. `+dotprod` 로 빌드한 바이너리는 ARMv8.0 기기에서 **SIGILL** 입니다.
> candle 은 이 경로에 런타임 디스패치가 없으므로, 실제 해법은 (a) 멀티 버저닝, (b) candle 에
> 런타임 검출 패치, (c) `minSdk`/ABI 로 대상을 좁히기 중 하나입니다. **어느 것도 한 줄이 아닙니다.**
>
> **이 에뮬레이터가 `+dotprod` 빌드를 돌릴 수 있었던 것은 HVF 게스트가 실제 M1 위에 있기
> 때문입니다** — 게스트는 HWCAP 에 `asimddp` 를 광고하지 않지만 `sdot` 명령 자체는 실행됩니다.
> **그러므로 위 2.05 배는 속도 측정으로는 유효하고, 호환성 측정으로는 무효입니다.**

그리고 천장이 하나 더 있습니다: **candle 에는 `i8mm`/`smmla`/`bfmmla` 가 한 줄도 없습니다**
(전체 소스 grep, 일치 0). ARM 의 저정밀 행렬 명령을 candle 은 쓰지 않습니다. 상류의 KleidiAI 는
씁니다. **즉 candle 경로의 상한은 `sdot` 이고, KleidiAI 의 상한보다 낮습니다.**

---

## 7. 정확도 — 공짜가 아니다

candle `QMatMul` 출력을 f32 `matmul` 과 대조 (**무작위 가우시안 가중치**, `1024×1024`):

| 형식 | 최대 절대오차 | 평균 절대오차 | 상대 RMS |
|---|---:|---:|---:|
| `Q4_0` | 9.10 | 2.23 | **8.5%** |
| `Q4K` | 8.18 | 1.98 | **7.5%** |
| `Q8_0` | 0.736 | 0.186 | **0.70%** |

**이 숫자를 모델 품질로 읽지 마십시오.** 무작위 가우시안은 양자화에 가장 불리한 입력이고
(구조가 없어 스케일이 아무것도 잡아내지 못함), 실제 모델 가중치는 훨씬 잘 양자화됩니다.
**실모델 평가를 하지 않았으므로 이 표가 말하는 것은 "0 이 아니다" 와 형식 간 순서뿐입니다.**

다만 이 저장소 맥락에서는 그것만으로도 결론이 하나 나옵니다. `docs/numerics/BF16.md` 는 **1 ULP 편향이
30 층을 지나 로짓을 11.75 움직이는 것**을 추적했고, 골든은 비트 단위 검사를 씁니다.
**양자화는 그 체계와 같은 판정 기준을 쓸 수 없습니다** — 7.5% 오차는 어떤 허용오차 정책으로도
"상류와 같다" 가 아닙니다. 양자화를 들이면 **"상류와 비트 일치" 와는 다른 검증 축**(perplexity,
토큰 일치율 같은 것)이 필요하고, 그 축이 지금 저장소에 없습니다.

---

## 8. 다음 작업 항목 — 구멍과 비용

| # | 항목 | 기대치 | 비용 | 근거 |
|---|---|---|---|---|
| 1 | **`f16 -> f32` 변환 경로** | f16 `mm` 512 에서 **~2.2×**, `add` 에서 더 | 작음 — 변환 커널 하나 | §3.3. bf16 은 0.072 ms 인데 f16 은 0.680 ms (9.5 배). 같은 원소 수, 같은 목적지 |
| 2 | **`_dyn_quant_pack_4bit_weight` / `_dyn_quant_matmul_4bit` (비-KleidiAI 형태)** | 상류 표면 호환 + 가중치 **7.8× 압축** | 중간 — **새 dtype 불필요** (§4.1) | 상류의 두 op 이 `uint8` + `f32`/`bf16` 스케일만 쓰고, 셋 다 이미 저장 가능 |
| 3 | **`+dotprod` 를 안드로이드/iOS 에 넣기** | 기기 Q4K **2.05×** | **한 줄이 아님** — SIGILL 위험, 멀티버저닝 또는 candle 런타임 검출 패치 필요 | §6 |
| 4 | **candle `QTensor` 를 `Repr` 에 넣기** | 프리필 기기 **3.29×**, 호스트 디코딩 15× | 큼 — `Repr` 세 번째 variant + 수명·GIL 규약 | §5.1. `QTensor` 는 `Tensor` 가 아니다 |
| 5 | **candle-core 에 `I8` 추가 (상류 기여)** | `torch.int8` 저장 가능 → `_int_mm`·`_weight_int8pack_mm` 이 열림 | 큼 — 상류 PR, 머지 보장 없음 (`docs/design/CANDLE_DEPS.md` §2a 가 4 개월 정체된 PR 사례) | §2.1 |
| 6 | **양자화용 검증 축** | 없으면 4·5 를 착지시킬 수 없음 | 중간 — perplexity 또는 토큰 일치율 하네스 | §7 |

**순서에 대한 의견 (판단이지 측정이 아닙니다):** 1 번이 가장 싸고 지금 배송 중인 경로를
직접 개선합니다. 2 번이 그다음 — 상류 표면과 정확히 맞고 새 dtype 이 필요 없다는 점에서
**§2 의 `I8` 벽을 우회하는 유일하게 확인된 경로**입니다. 4 번이 가장 큰 수를 주지만 6 번이
먼저 있어야 착지 판정이 됩니다. 3 번은 **수치가 크지만 호환성 리스크가 그 크기를 상쇄**하므로
단독으로 하지 말고 4 번과 같이 다뤄야 합니다.

---

## 9. 확인하지 않은 것

| 항목 | 상태 |
|---|---|
| 실기 안드로이드 | **없음.** 전부 에뮬레이터 (§0) |
| 기기의 디코딩(gemv) 수치 | **재현 안 됨.** 같은 바이너리가 0.095~0.700 ms (§0). 결론에 쓰지 않음 |
| iOS | **미측정.** target feature 목록만 확인했습니다(§6). 기기에서 재지 않았습니다 |
| 인자 합성 실패한 40 개 op 의 dtype 거동 | **미측정** (§2.2). f16/bf16 구멍이 그 안에 있을 가능성 배제 못 함 |
| `float8_e4m3fn` | **미측정.** 프로브가 무한 대기하는 것이 이미 알려져 있어 건너뜀 |
| 4-bit 커널의 **출력값** | **미검증.** 무작위 `uint8` 로 시간만 쟀습니다 (§4.2) |
| 실모델 양자화 품질 | **미측정.** §7 은 무작위 가중치이고 상한입니다 |
| KleidiAI **없을 때** 상류의 4-bit 커널 | **미측정.** 이 호스트는 `_has_kleidiai=True` 라 그 분기를 못 밟았습니다 |
| 상류 f16 `mm` 이 왜 3 GFLOP/s 인지 | **추론.** 상류 커널을 프로파일하지 않았습니다 (§3.6) |
| `accelerate` 가 f16 변환을 2 배 느리게 하는 것 | **측정됐고 설명 못 함.** 아래 §9.1 |
| 스레드 | 전부 단일 스레드. 멀티스레드에서 배수가 어떻게 변하는지 미측정 |
| 메모리 사용량 · 양자화 자체의 비용 | 안 쟀습니다. `QTensor::quantize` 시간은 측정 밖 |

### 9.1 설명하지 못한 것 하나

`accelerate` feature 를 켜면 **f16 관련 연산이 느려집니다.** 재현됩니다(2 회차, 오염된 회차 포함 3 회):

| | accelerate 켬 | accelerate 끔 | 켬/끔 |
|---|---:|---:|---:|
| `cvt` 1024 `f16->f32` | 0.6802 | 0.3355 | **2.03×** |
| `cvt` 512 `f16->f32` | 0.1685 | 0.0852 | **1.98×** |
| `add` 1024 f16 | 1.8589 | 1.1653 | **1.60×** |
| `cvt` 1024 `bf16->f32` | 0.0717 | 0.0717 | 1.00× |
| `add` 1024 bf16 | 0.5440 | 0.5261 | 0.97× |

**bf16 은 영향이 없고 f16 만 정확히 2 배입니다.** candle 에서 `accelerate` feature 가 게이팅하는
것은 matmul 과 f32/f64 vForce 초월함수뿐이고(전체 grep), **dtype 변환은 건드리지 않습니다.**
그러므로 이것은 코드 경로 차이가 아니라 **코드 생성/인라이닝 부작용**으로 보이지만,
**확인하지 않았습니다.** §8 의 1 번 항목과 같은 자리를 가리키므로 함께 조사할 값이 있습니다.

---

## 10. 재현

```sh
export PATH="$HOME/.cargo/bin:$HOME/Library/Android/sdk/platform-tools:$PATH"
cd /Volumes/macMini/worktrees/bw-quant
bash vendor/vendor_torch.sh
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-quant
export HF_HOME=/Volumes/macMini/caches/hf-home
PY=/Volumes/macMini/caches/spike-venv/bin/python
bash vendor/install_shim.sh                       # PATH 에 cargo 가 있어야 함 -- §1 의 함정
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
```

측정 스크립트는 저장소 밖 `/Volumes/macMini/caches/quant-scratch/` 에 있습니다:

| 파일 | 무엇 | 절 |
|---|---|---|
| `storable.py` | dtype 55 종에 `aten.full.default` | §2.1 |
| `dtype_survey.py` | 118 op × 12 dtype, 스키마로 인자 합성 | §2.2 |
| `bench.py` | dtype × 모양 스윕 (시임·상류 공용) | §3 |
| `q4bench.py` | 상류 KleidiAI 4-bit | §4.2 |
| `qbench/` | candle 양자화 (`src/main.rs` 속도, `src/bin/qerr.rs` 오차) | §5, §7 |

```sh
# §3  호스트 dtype 스윕
RAYON_NUM_THREADS=1 TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$PWD/torchnative/src/main \
    $PY /Volumes/macMini/caches/quant-scratch/bench.py ours
OMP_NUM_THREADS=1 $PY /Volumes/macMini/caches/quant-scratch/bench.py upstream   # PYTHONPATH 없이

# §3.4  accelerate 끈 빌드 (환경변수 RUSTFLAGS 로 주지 말 것 -- PERF_ANDROID.md §7.2)
CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-quant-noaccel \
  ( cd rust/torch_c && cargo build --release \
      --config 'target."cfg(target_vendor = \"apple\")".rustflags = ["--cfg", "torch_c_no_accelerate"]' )
# 산출물 교체는 cp 로 백업하고 cp 로 복구합니다 -- `git checkout --` 금지

# §5, §6  candle 양자화
cd /Volumes/macMini/caches/quant-scratch/qbench
CARGO_TARGET_DIR=.../cargo-target-qbench       cargo build --release          # macOS 기본 = +dotprod
CARGO_TARGET_DIR=.../cargo-target-qbench-nodot RUSTFLAGS="-C target-feature=-dotprod" cargo build --release

# §5.2  기기. NDK clang 을 CC 로 줘야 onig_sys 가 빌드됩니다 (CANDLE_DEPS.md §2a 의 그 크레이트)
NDK=/Volumes/macMini/caches/android-ndk/27.1.12297006/toolchains/llvm/prebuilt/darwin-x86_64/bin
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$NDK/aarch64-linux-android26-clang
export CC_aarch64_linux_android=$NDK/aarch64-linux-android26-clang
export AR_aarch64_linux_android=$NDK/llvm-ar
cargo build --release --target aarch64-linux-android

export ANDROID_SERIAL=emulator-5554               # 5556 은 건드리지 않습니다
adb push .../qbench /data/local/tmp/bw_device/qbench/qbench_nodot
adb shell "cd /data/local/tmp/bw_device/qbench && RAYON_NUM_THREADS=1 ./qbench_nodot; echo DEVICE_EXIT=\$?"
# adb shell 의 종료 코드를 읽지 마십시오 -- 기기 셸이 찍는 DEVICE_EXIT= 를 봅니다
```

회귀 (전부 exit 0, 이 문서를 쓰는 동안 변하지 않았습니다):

```sh
PYTHON=$PY sh rust/torch_c/pytests/run.sh          # 171
$PY tools/golden/compare.py                        # 2744/2744 ops=118
$PY rust/torch_c/pytests/verify_schemas.py         # 3076/3076
```

---

## 11. 보고 분류

`CLAUDE.md` §5.3 에 따라 종류를 나눕니다.

| 종류 | 무엇 |
|---|---|
| **측정** | dtype 저장 가능 집합(§2.1) · 118 op 의 dtype 수용(§2.2) · reduced float 의 성능 대가(§3) · 상류 KleidiAI 4-bit 천장(§4.2) · candle 양자화 호스트/기기(§5.2) · `dotprod` 게이트의 값(§6) · 양자화 오차 상한(§7) |
| **문서 정정** | `docs/perf/PERF_ANDROID.md` §3 의 "컴파일 플래그는 문이 아니다" 는 **f32 에 한정**됩니다. candle 양자화 커널에는 적용되지 않습니다(§6) |
| **기능 추가** | **없음** |
| **결함 수정** | **없음** |
| **테스트 추가** | **없음** — 측정 스크립트는 저장소 밖입니다(§10) |
| **삭제** | **없음** |

**저장소 코드는 한 줄도 바뀌지 않았습니다.** 이 회차의 산출물은 이 문서 하나입니다.
