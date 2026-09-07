# 축소 정밀도 — 어느 층이 비쌌는가, 그리고 int8 로 가는 길

측정일 2026-08-28. 브랜치 `work/dtype`. 호스트 Apple M1 (P 4 + E 4, macOS darwin 25.5.0),
CPython 3.13.0, 상류 torch 2.13.0, candle-core 0.11.0, half 2.7.1, rustc 1.98.0.
**벤더링 트리는 한 줄도 고치지 않았습니다.**

`docs/graph/QUANT.md` 가 "dtype 을 낮추면 느려진다" 를 측정으로 닫아둔 자리에서 시작합니다.
이 문서는 두 가지를 답합니다 — **왜 느렸는지(그리고 그 중 무엇을 고쳤는지)**, 그리고
**int8 로 가는 길이 실제로 무엇을 요구하는지**.

> **결론 먼저.**
>
> 1. **`opmath_in` 규칙은 무죄입니다. 비싼 것은 그 규칙의 구현이었습니다.** 비용은 네 층에
>    나뉘어 있었고 세 층은 우리 쪽이었습니다 — candle 의 원소 단위 `to_dtype`, 넓힌 텐서를
>    **메모리에 실체화**하던 3-패스 구조, 그리고 결과 버퍼를 `vec![0f32; n]` 으로 **0 으로 채우던**
>    것. 넷째 층만 하드웨어입니다.
> 2. **`float16` 이 정상화됐습니다.** 같은 세션 A/B 로 변환 **4.7 배**, `add` **11.2 배**,
>    디코딩 `mv` **3.7 배** 빨라졌고, **`float16` 의 `add` 는 이제 `float32` 보다 빠릅니다**
>    (0.106 ms 대 0.136 ms). QUANT.md §3.2 의 27.4 배가 3.84 배로 내려왔습니다.
>    정확성은 한 비트도 바뀌지 않았습니다 — 새 커널은 `half` 와 **비트 단위로 같은 함수**이고,
>    65536 개 비트 패턴 전부에 대해 그것을 검사합니다.
> 3. **`bfloat16` 은 거의 그대로입니다. 그리고 그것이 하드웨어입니다.** ARMv8.6 (`FEAT_BF16`)
>    이전의 ARM 에는 `bfloat16` 변환 명령이 **없습니다.** `float16` 이 `fcvtl`/`fcvtn` 한 개로
>    하는 일을 `bfloat16` 은 정수 연산 9 개로 해야 하고(round-to-nearest-even + NaN 처리),
>    그래서 `bfloat16` 의 elementwise 는 여전히 `float32` 의 2.7 배입니다. M1 에 `FEAT_BF16` 이
>    없습니다.
> 4. **candle 에는 축소 정밀도 GEMM 이 있고, 우리는 그것을 쓰면 안 됩니다.** `gemm-f16` 의
>    네이티브 커널은 accelerate 를 끄면 실제로 링크되고 f32 보다 **1.92 배 빠릅니다.** 그런데
>    **`float16` 으로 누산합니다** — 실측으로 k=512 에서 4096 개 출력 중 **3818 개**가 f32 누산과
>    다르고 최대 절대오차 0.172 입니다. `docs/numerics/BF16.md` 가 산 것을 그대로 되돌리는 커널입니다.
>    QUANT.md §3.5 의 "존재하는 저정밀 커널을 우회하고 있다" 는 관찰은 맞았지만, **그 커널은
>    우리가 원하는 함수를 계산하지 않습니다.**
> 5. **int8 은 그 자체로는 목표가 아닙니다.** candle 에 `I8` 을 넣어도 candle 에는 **int8
>    matmul 커널이 없습니다** — 저장 타입이 생기고 커널은 우리가 씁니다. 반대로 상류의
>    비-KleidiAI 4-bit 팩 레이아웃은 **전부 `float32` 로 캐스팅해서 concat 한 것**이라
>    압축이 0 이고, 이 환경에는 그 op 을 부르는 것이 **없습니다**(`torchao` 미설치).
>    **권고는 (b) — candle `QTensor` 를 `Repr` 의 세 번째 변형으로 들이는 것**이고,
>    **그 앞에 반드시 §6.4 의 검증 축이 있어야 합니다.**

---

## 0. 이 문서의 숫자로 하면 안 되는 것

- **기계가 공유 중이었습니다.** 다른 에이전트가 동시에 돌았고 load 는 1.6 ~ 11.7 을 오갔습니다.
  그래서 **같은 실행 안의 상대 비교만** 씁니다. §3 의 A/B 는 한 세션 안에서 base 와 new 를
  **번갈아** 돌린 것이고, §2 의 커널 비교는 한 프로세스 안에서 연달아 잰 것입니다.
  **절대 수치를 다른 문서의 절대 수치와 비교하지 마십시오.**
- **호스트만 쟀습니다.** 안드로이드·iOS 는 컴파일만 확인했습니다(§5.3). 기기 측정은 없습니다 —
  다른 에이전트가 에뮬레이터를 쓰고 있었습니다.
- **§4.3 의 손으로 쓴 gemv 는 프로토타입입니다.** 블로킹도 언롤도 없습니다. 그 절대값을
  "이것이 가능한 최선" 으로 읽으면 안 됩니다. 서로 **같은 구조**의 f32/f16/bf16 세 커널을
  비교하는 용도입니다.
- **§6 의 int8 권고는 측정이 아니라 판단입니다.** 근거가 되는 사실에는 전부 출처를 달았고,
  판단인 부분은 판단이라고 적었습니다.

---

## 1. 검증된 기준선

이 작업을 시작한 트리 상태(`6076cf4`)와 끝낸 상태 양쪽에서, 전부 exit 0:

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh        전 171 통과 -> 후 175 (+4)
$PY tools/golden/compare.py                      2744/2744, ops=118        (변화 없음)
$PY rust/torch_c/pytests/verify_schemas.py       3076/3076                 (변화 없음)
cd rust/torch_c && cargo test --release          전 2 통과 -> 후 7 (+5)
```

> **문서 정정.** `docs/devices/DEVICE_ABS.md` §5.1 은 *"이 크레이트에는 돌릴 수 있는 Rust 단위 테스트가
> 없습니다 — `crate-type = ["cdylib"]` + `extension-module` 이라 `cargo test` 가 `dyld: symbol
> not found in flat namespace '_PyExc_BaseException'` 로 죽습니다"* 라고 적고 있습니다.
> **rustc 1.98.0 에서는 죽지 않습니다.** `capture.rs` 에 이미 있던 두 개를 포함해 7 개가 돕니다
> (실측, 이 회차 전후 모두). 그 문장을 근거로 Rust 테스트를 쓰지 않기로 한 판단은 이제 유효하지
> 않습니다. 다만 `pytests/run.sh` 는 `cargo test` 를 부르지 않으므로 **표준 검증 경로에는 여전히
> 들어 있지 않고**, 그래서 이 회차의 정확성 검사는 `test_shim.py` 쪽에도 두었습니다(§5).

---

## 2. 비용은 네 층에 나뉘어 있었다

측정 도구는 저장소 밖 `/Volumes/macMini/caches/dtype-scratch/cvtbench` 의 독립 크레이트입니다
(candle-core + half 만 의존, accelerate 켠 빌드와 끈 빌드 두 개). 100 만 원소, 40 회 최솟값,
단일 스레드.

### 2.1 candle 은 원소 하나씩 변환한다 — 그리고 `f16::to_f32` 는 인라인 어셈블리다

`CpuStorage::to_dtype` (`cpu_backend/mod.rs:1862`) 은 전부 `unary_map` 에 **원소 단위 클로저**를
넘깁니다:

```rust
(Self::F16(storage), DType::F32) => {
    let data = unary_map(storage, layout, |v| v.to_f32());   // <- 원소마다 한 번
    Ok(Self::F32(data))
}
```

`half::f16::to_f32` 는 aarch64 에서 이것입니다 (`half-2.7.1/src/binary16/arch/aarch64.rs:13`):

```rust
asm!("fcvt {0:s}, {1:h}", out(vreg) result, in(vreg) i, options(pure, nomem, nostack));
```

**인라인 어셈블리는 벡터화기에게 불투명합니다.** 그래서 루프가 스칼라로 남습니다.
`aarch64-apple-darwin` 의 기본 target feature 에 `fp16` 이 **들어 있어서**(확인함:
`rustc --print cfg`) `half` 의 `convert_fn!` 매크로는 런타임 검출조차 하지 않고 곧장 이
어셈블리 경로를 고릅니다 — 즉 **플래그 문제가 아니라 형태 문제**입니다.

`bfloat16` 이 이 함정을 피한 것은 그 변환이 시프트 한 번(`f32::from_bits((x as u32) << 16)`)이라
LLVM 이 `shll` 로 벡터화하기 때문입니다. **한 코드 경로에서 두 dtype 의 운명이 갈린 자리가
여기입니다.**

```
100만 원소, 한 프로세스 안에서 연달아                 Gelem/s
candle to_dtype   bf16 -> f32     0.1719 ms            6.10
candle to_dtype   f16  -> f32     0.3956 ms            2.65    <- 2.3 배 느림
half slice-ext    f16  -> f32     0.0835 ms           12.55
NEON fcvtl        f16  -> f32     0.0704 ms           14.90    <- candle 대비 5.6 배
```

`half` 자신은 **빠른 슬라이스 변환을 가지고 있습니다**(`f16_to_f32_slice`, 4 개씩 `fcvtl`).
candle 이 그것을 부르지 않을 뿐입니다.

### 2.2 넓히는 지점이 틀렸다 — 상류는 누산기만, 우리는 텐서 전체

`aten.rs` 의 모든 opmath 경로가 이 모양이었습니다:

```rust
let lhs = lhs.to_dtype(F32)?;          // float32 텐서 하나를 메모리에 쓴다
let rhs = rhs.to_dtype(F32)?;          // 또 하나
lhs.broadcast_add(&rhs)?.to_dtype(BF16)  // 또 하나, 그리고 좁혀서 또 하나
```

원소당 메모리 트래픽이 **30 바이트**입니다(읽기 2×2 + 쓰기 2×4 + 읽기 2×4 + 쓰기 4 + 읽기 4 +
쓰기 2). 상류의 `TensorIterator` 는 넓히기를 **레지스터 안에서** 하므로 **6 바이트**입니다.

**이것이 "`at::opmath_type` 규칙" 과 "그 규칙의 비싼 구현" 의 차이 전부입니다.** 규칙은
"누산을 `float` 로 하라" 이지 "텐서를 `float32` 로 승격하라" 가 아닙니다.

### 2.3 결과 버퍼를 0 으로 채우고 있었다

이건 처음 고칠 때 **내가 새로 만든 손해**입니다. NEON 커널을 넣고 재보니 `bfloat16` 변환이
**3.6 배 느려졌습니다**(0.0715 → 0.2563 ms). 원인은 커널이 아니라 `vec![0f32; n]` —
4 MB 를 0 으로 쓴 다음 그 위에 변환 결과를 씁니다. candle 은 `.map().collect()` 라서 그 패스가
없습니다.

```
bf16 -> f32, 할당을 타이밍 안에 포함              Gelem/s
NEON into vec![0f32; n]        0.1663 ms          6.30
NEON into uninit capacity      0.1266 ms          8.28     <- 1.31 배
f16 -> f32
NEON into vec![0f32; n]        0.1040 ms         10.08
NEON into uninit capacity      0.0704 ms         14.90     <- 1.48 배
```

`Vec::with_capacity` 뒤에 커널이 원시 포인터로 채우고 `set_len` 합니다. `reduced.rs` 의
`built()` 가 그것이고, 모든 커널이 `&mut [T]` 가 아니라 `*mut T` 를 받는 이유입니다.

> **여기서 배운 것.** 빠른 커널을 넣고 **전체를 다시 재지 않았다면** 이 손해는 그대로 배송됐을
> 것입니다. `bfloat16` 은 원래 안 느렸으므로 "f16 이 빨라졌다" 만 보고 끝냈을 것입니다.

### 2.4 어느 구현이 가장 빠른지는 균일하지 않다 — 넷 다 따로 쟀다

"NEON 이 항상 이긴다" 가 아니었습니다.

| 변환 | 가장 빠른 것 | 왜 |
|---|---|---|
| `bf16 -> f32` | **평범한 루프** (9.35 Gelem/s) | LLVM 이 `shll` 하나로 넓히고 시프트한다. 손으로 쓴 NEON 은 `ushll` + `shl` 두 개가 되어 8.28 로 **졌다** |
| `f16 -> f32` | **NEON `fcvtl`** (14.90) | `half` 12.55, candle 2.65 |
| `f32 -> f16` | **NEON `fcvtn`** (11.03) | `half` 7.81 |
| `f32 -> bf16` | **NEON 정수 RNE** (5.94) | `half` 3.93. 넷 중 유일하게 하드웨어 명령이 없는 방향 |

그래서 `reduced.rs` 의 `widen_bf16_into` 에는 **NEON 팔이 없습니다.** 손으로 쓴 것이 컴파일러가
쓴 것보다 느렸고, 재보지 않았으면 그대로 넣었을 것입니다.

---

## 3. 무엇을 고쳤나 — 그리고 A/B

`rust/torch_c/src/reduced.rs` (신규, 746 줄) 이 두 가지를 합니다.

1. **`{f16,bf16} <-> f32` 변환 4 개**를 candle 대신 직접 합니다. `candle_core::CustomOp1`
   두 개(`Widen`, `Narrow`)로 들어가고, `FastDType::fast_to` 라는 이름의 확장 트레이트로
   `aten.rs` 의 **80 개 opmath 호출 지점**이 그것을 부릅니다
   (`.to_dtype(acc)` 28 개 + `.to_dtype(storage)` 52 개를 기계적으로 치환).
   축소 float 이 아니거나 비연속이거나 CPU 가 아니면 **candle 로 그대로 넘깁니다.**
2. **`add`/`sub`/`mul`/`div` 를 한 패스로** 합니다 (`CustomOp2`). 조건은 두 피연산자가 같은
   축소 float 이고, CPU 이고, 브로드캐스트 후 같은 모양이고, `alpha == 1` 인 것.
   하나라도 아니면 `None` 을 돌려주고 호출자가 기존 경로를 탑니다 — **다른 함수를 계산하는
   팔이 하나도 없습니다.**

**정확성은 구조로 보장됩니다.** 모든 커널이 `half` 가 원소 단위로 계산하는 바로 그 함수이고
(`fcvtl`/`fcvtn` 은 RNE, `bfloat16` 좁히기는 `bf16::from_f32` 의 `if` 를 `bsl` 로 바꾼 것),
융합 커널은 정의상 `narrow(widen(a) OP widen(b))` 입니다. §5 가 그것을 **논증하지 않고 검사**합니다.

### 3.1 A/B — 같은 세션에서 두 산출물을 번갈아

base 는 `6076cf4` 의 소스를 별도 `CARGO_TARGET_DIR` 로 빌드한 것이고(의존성 버전 동일 — 
`Cargo.lock` 의 변화는 `torch_c` 의 의존 목록에 `half` 한 줄이 늘어난 것뿐), 두 벤더 트리를
만들어 `PYTHONPATH` 로 갈아끼우며 **base → new → base → new** 순으로 돌렸습니다.
load 10.2 ~ 11.7. 각 셀은 두 회차의 최솟값.

| case | base (ms) | new (ms) | 개선 |
|---|---:|---:|---:|
| `mm` 128 f16 | 0.0233 | **0.0105** | **2.22×** |
| `mm` 256 f16 | 0.0955 | **0.0452** | **2.11×** |
| `mm` 512 f16 | 0.4624 | **0.2625** | **1.76×** |
| `mm` 1024 f16 | 2.9315 | **2.1754** | **1.35×** |
| `mv` 1024 f16 (디코딩) | 0.3630 | **0.0992** | **3.66×** |
| `mv` 2048 f16 (디코딩) | 1.7379 | **0.7347** | **2.37×** |
| `cvt` 512 f16→f32 | 0.0857 | **0.0190** | **4.51×** |
| `cvt` 1024 f16→f32 | 0.3377 | **0.0723** | **4.67×** |
| `add` 1024 f16 | 1.1888 | **0.1060** | **11.2×** |
| `add` 1024 bf16 | 0.5484 | **0.3727** | **1.47×** |
| `mm` 128/256/512/1024 bf16 | 0.0130 / 0.0549 / 0.3040 / 2.3178 | 0.0121 / 0.0516 / 0.2902 / 2.2423 | 1.03 ~ 1.07× |
| `mv` 1024 / 2048 bf16 | 0.0995 / 0.7241 | 0.0998 / 0.7210 | 1.00× |
| `cvt` bf16→f32 | 0.0196 / 0.0734 | 0.0193 / 0.0726 | 1.01× |
| `mm` f32 (회귀 확인) | 0.0058 / 0.2086 / 1.7974 | 0.0058 / 0.2086 / 1.7435 | 변화 없음 |

**`bfloat16` 의 `mm`·`mv`·`cvt` 가 그대로인 것은 예상대로입니다** — 그 경로의 비용은 이미
`shll` 한 개였고 §2.4 가 그것을 확인했습니다. 움직인 것은 융합이 걸리는 `add` 뿐입니다.

### 3.2 이제 `float32` 대비 어디에 서 있나

new 빌드 안에서, 같은 실행:

| case | f32 | f16 | bf16 | f16/f32 | bf16/f32 | (QUANT.md §3.2 의 f16/f32) |
|---|---:|---:|---:|---:|---:|---:|
| `mm` 128 | 0.0058 | 0.0105 | 0.0121 | 1.81× | 2.09× | *6.15×* |
| `mm` 256 | 0.0309 | 0.0452 | 0.0516 | 1.46× | 1.67× | *4.31×* |
| `mm` 512 | 0.2086 | 0.2625 | 0.2902 | 1.26× | 1.39× | *3.01×* |
| `mm` 1024 | 1.7435 | 2.1754 | 2.2423 | 1.25× | 1.29× | *1.87×* |
| `mv` 1024 | 0.0258 | 0.0992 | 0.0998 | 3.84× | 3.87× | *27.4×* |
| `mv` 2048 | 0.2780 | 0.7347 | 0.7210 | 2.64× | 2.59× | *11.2×* |
| **`add` 1024** | 0.1362 | **0.1060** | 0.3727 | **0.78×** | 2.74× | *14.7×* |

**`float16` 의 `add` 한 칸만 1 보다 작습니다.** QUANT.md §3.2 의 "1 보다 작은 칸이 하나도 없다"
가 깨졌고, 깨진 자리가 **메모리 대역폭에 걸리는 op** 이라는 것이 §4 의 설명 전부입니다.

---

## 4. 남은 것 — 왜 나머지는 `float32` 를 못 이기는가

### 4.1 elementwise: `bfloat16` 에는 변환 명령이 없다

`float16` 의 융합 `add` 가 이기고 `bfloat16` 이 지는 이유는 하나입니다.

```
8 개 원소를 처리하는 데 드는 명령 수
f16    ld1q, fcvtl, fcvtl2, fcvtl, fcvtl2, fadd, fadd, fcvtn, fcvtn2, st1q     ~ 11
bf16   ld1q, ushll, ushll2, shl, shl, (x2 피연산자), fadd, fadd,
       그리고 좁히기: shr/and/add/add/shrn/shrn/orr/and/cmp/shrn/bsl x2        ~ 30
```

**`float32 -> bfloat16` 만 하드웨어 명령이 없습니다.** round-to-nearest-even 과 NaN 처리를
정수 연산으로 해야 하고(§2.4 의 5.94 Gelem/s), 그것이 그대로 2.74 배입니다.
`FEAT_BF16` (ARMv8.6) 의 `BFCVT` 가 이 9 개를 1 개로 만들지만 **M1 에 그 확장이 없습니다.**

즉 **이 하드웨어에서 `bfloat16` elementwise 가 `float32` 를 이기는 방법은 없습니다.**
`opmath_in` 을 버리면(= 절사) 이길 수 있고, 그것이 `docs/numerics/BF16.md` 가 고친 결함입니다.

### 4.2 `mm`: AMX 는 `float32` 전용이고, candle 의 `float16` GEMM 은 함수가 다르다

Apple 에서 f32 `mm` 512 는 0.2086 ms = **1289 GFLOP/s** 입니다. AMX 입니다. 축소 정밀도로
그 코프로세서에 닿는 경로가 **없습니다**:

```
accelerate 켬 (배송):  candle f16 matmul   -> "the accelerate backend does not support f16 matmul"
                       candle bf16 matmul  -> "unsupported dtype BF16 for op matmul"
```

accelerate 를 끄면 `gemm-f16` 의 네이티브 커널이 실제로 링크되고 **f32 보다 1.92 배 빠릅니다**
(512³ 에서 1.4723 ms 대 2.8296 ms). QUANT.md §3.5 가 "배송 빌드에서 `gemm_f16` 심볼 0 개" 로
관찰한 그 커널이고, **도달 불가인 것은 맞습니다.** 그런데:

```
k=512, 64x64 출력. candle 네이티브 f16 matmul vs 같은 값을 f32 로 넓혀 곱한 것
  3818 / 4096 개가 다름,  최대 절대오차 0.171875
```

**`float16` 으로 누산합니다.** `gemm-f16` 에는 `float32` 로 패킹해 f32 마이크로커널을 쓰는
경로(`MixedSimd<T,T,T,f32>`)와 `neonfp16` 네이티브 경로가 **둘 다** 있고, `fp16` 이 있는 이
호스트는 후자를 고릅니다. `gemm_accumulate_in` 의 doc comment 가 적어둔 그 실패
("k=512 에서 64 개 중 15 개가 허용오차 밖") 를 재현합니다.

    따라서 "candle 이 축소 정밀도 커널을 갖고 있는가" 의 답은:
    **f16 은 있고 bf16 은 없으며, 있는 그것은 우리가 원하는 함수를 계산하지 않는다.**

`aarch64-linux-android` 의 기본 target feature 는 `neon` 하나뿐이므로(QUANT.md §6) 그 타깃에서는
`gemm-f16` 이 f32 누산 경로로 갈 **가능성**이 있습니다. 그렇다면 안드로이드에서는 정확성을 잃지
않고 1.9 배를 얻을 수도 있습니다. **기기에서 재지 않았습니다.** 런타임 디스패치이므로 소스만
읽어서는 판정할 수 없고, 그래서 이것은 가설로 남깁니다.

### 4.3 `mv` (디코딩): 융합 gemv 는 f32 gemv 를 이기지만 Accelerate 는 못 이긴다

가중치를 넓히는 것은 **O(n²) 작업이고 gemv 도 O(n²)** 입니다. 상각될 여지가 구조적으로 없으므로,
`mv` 의 3.8 배를 없애려면 **가중치를 축소 정밀도로 읽는 커널**이 필요합니다.
같은 구조로 세 개를 써서 쟀습니다(블로킹 없음, 프로토타입):

| 1×1024×1024 | ms | GMAC/s |
|---|---:|---:|
| Accelerate `sgemv` (= 지금 넓힌 뒤 부르는 것) | **0.0887** | 11.8 |
| 지금의 시임 경로 (넓히고 gemv 하고 좁힘) | 0.5797 | 1.8 |
| 손으로 쓴 NEON f32 gemv | 0.1976 | 5.3 |
| 손으로 쓴 NEON **f16** gemv, f32 누산 | **0.1273** | 8.2 |
| 손으로 쓴 NEON **bf16** gemv, f32 누산 | 0.1419 | 7.4 |

**같은 구조끼리 비교하면 축소 정밀도가 이깁니다** (f16 이 f32 대비 1.55 배, bf16 1.39 배 —
바이트를 절반만 읽으니 당연합니다). 그런데 **Accelerate 가 손으로 쓴 f32 보다 2.2 배 빠르므로**,
융합 f16 gemv 를 넣어도 지금의 "넓히고 Accelerate" 보다 느립니다.

> **이 표의 중요한 한계.** 1024² f32 = 4 MB 이고 M1 의 캐시에 들어갑니다.
> 실제 디코딩은 수백 MB 의 가중치를 **DRAM 에서** 읽으므로 그 영역에서는 바이트 수가 그대로
> 시간이 됩니다. **모델 규모에서 이 부호가 유지되는지 재지 않았습니다.**
> 이 표가 말하는 것은 "캐시에 들어가는 크기에서는 Accelerate 가 이긴다" 뿐입니다.

### 4.4 목표에 대한 답

> *"정확성을 잃지 않고 축소 정밀도가 f32 보다 느리지 않게 하라. 못 하면 왜 못 하는지가 답이다."*

**하나는 했고**(`float16` elementwise, 0.78×), **나머지는 이 하드웨어에서 못 합니다.** 층별로:

| 층 | 남은 비용 | 고칠 수 있나 |
|---|---|---|
| `bfloat16` 좁히기 | `float32` 의 2.74× (elementwise) | **아니오.** `FEAT_BF16` 이 없는 CPU 에서는 정수 9 연산 |
| `mm` | 1.25 ~ 1.81× | **아니오.** AMX 가 f32 전용이고, 축소 정밀도로 거기 가는 경로가 없다 |
| `mv` | 2.6 ~ 3.9× | **부분적.** 융합 gemv 가 필요하고, Accelerate 수준으로 써야 이긴다 (§4.3) |
| `float16` elementwise | **없음** | 끝났다 |

**"opmath 때문" 은 답이 아니었습니다.** 규칙은 그대로 두고 `float16` 을 11 배 빠르게 만들었고,
남은 것은 규칙이 아니라 **명령 집합**입니다.

---

## 5. 검증 — 실패할 수 있는 것으로

`CLAUDE.md` §5.5. 이 회차의 주장은 "같은 함수를 더 빨리 계산한다" 이므로, 검사는 전부
**그 동일성**을 겨냥합니다. 허용오차를 쓰는 검사는 하나도 없습니다.

### 5.1 Rust 단위 테스트 5 개 (`reduced.rs`)

- `reduced_kernels_agree_with_half_on_every_f16_bit_pattern` — `float16` 65536 개, `bfloat16`
  65536 개 **전부**. 표본이 아니라 전수입니다.
- `reduced_narrowing_rounds_to_nearest_even_including_nan_and_the_tail` — 무작위 20 만 개 +
  Inf/NaN/subnormal/signalling NaN/정확한 tie. 길이를 8 의 배수가 아니게 두어 **스칼라 꼬리도**
  돕니다.
- `fused_arithmetic_equals_widen_compute_narrow_element_by_element` — 4 연산 × 2 dtype ×
  4099 원소(8 의 배수 아님).
- `fast_to_dtype_matches_candle_bit_for_bit`, `fused_arithmetic_broadcasts_and_still_matches_the_slow_path`.

### 5.2 `test_shim.py` 4 개 (상류와 대조, `run.sh` 가 실제로 돌린다)

- `..._matches_upstream_at_every_vector_boundary` — 4 연산 × 2 dtype × **13 개 길이**
  (1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1000, 1001). 8 의 배수만 재면 꼬리 결함이 안 보입니다.
- `..._when_an_operand_broadcasts` — rotary 의 모양. 융합 커널이 브로드캐스트를 실체화하므로,
  **모양은 맞고 값이 틀린** 답이 나올 수 있는 유일한 팔입니다.
- `..._on_non_contiguous_operands` — 전치된 피연산자는 빠른 경로에 안 갑니다. 폴백이 여전히
  일치하는지 고정합니다.
- `..._carries_the_values_no_shift_would` — §5.4 참조.

### 5.3 컴파일 타깃

```
aarch64-apple-darwin              build   OK   (경고 0, 기존 `by_name` 제외)
aarch64-linux-android             check   OK   (NDK clang 을 CC 로)
비-aarch64 폴백 경로               test    OK   (7/7) -- cfg 를 강제로 뒤집어 확인
aarch64-apple-ios / -sim          미확인   pyo3-ffi 가 PYO3_CONFIG_FILE 을 요구 (기존 제약)
```

**폴백 경로를 확인한 방법**: `target_arch = "aarch64"` 14 곳을 없는 arch 로 바꿔 호스트에서
빌드·테스트했습니다. 포터블 본문(= `half` 의 슬라이스 변환)이 컴파일되고 **같은 정확성 검사를
통과**합니다. iOS 는 aarch64 이므로 android 와 같은 코드 경로입니다.

### 5.4 고장 내서 확인했다 — 그리고 한 번은 검증이 거짓말을 했다

`cp` 백업, `git checkout` 아님.

| 무엇을 깼나 | 무엇이 빨개졌나 |
|---|---|
| `bfloat16` 좁히기에서 반올림 덧셈 제거 (= 절사 재도입) | **골든 6 개** + pytest **5 개** |
| `float16` 융합 커널의 **스칼라 꼬리**를 0 으로 | pytest 3 개 (n=1 에서 즉시) |
| `bfloat16` 좁히기의 **NaN 선택 제거** | pytest **1 개**, 그리고 그것뿐 |

세 번째가 이 회차에서 가장 값이 있습니다. NaN 선택을 빼면 `0x7fff_ffff`(가수가 전부 1 인 NaN)
가 반올림 덧셈에서 지수로 올림되어 **`0x8000` = 음의 0** 이 됩니다. 실제 실패 메시지:

```
FAIL test_reduced_float_conversion_carries_the_values_no_shift_would:
     bfloat16 narrowed: NaN came back as [(0, -0.0), (1, 0.0), (4, -0.0), (5, 0.0)]
```

**골든은 이것을 못 잡습니다** — 축소 float 정확 케이스에 NaN 이 없고, 넣더라도 상류의 NaN
페이로드(`0x7fc0`)와 `half` 의 것(`0x7fff`)이 달라 값 비교로는 판정할 수 없습니다(둘 다 NaN 입니다).
그래서 이 검사는 "NaN 인가" 만 묻습니다. 그리고 **`float("nan")` 만 넣는 프로브는 이 결함을
통과시킵니다** — 기본 quiet NaN 은 절사해도 NaN 으로 남기 때문입니다. 실측으로 확인했습니다.

> **그리고 검증이 한 번 거짓말을 했습니다.** 처음 두 번의 고장 확인에서 `TORCH_C_ARTEFACT` 를
> export 하지 않은 채 `compare.py` 를 돌렸고, **골든이 고정 캐시 경로의 다른 빌드를 재면서
> 2744/2744 초록을 냈습니다.** `pytests/run.sh` 의 주석이 경고하는 바로 그 함정입니다.
> 절사를 재도입한 빌드가 골든을 통과했다고 보고할 뻔했습니다. 위 표는 산출물을 고정하고
> **다시 돌린** 결과입니다.

---

## 6. int8 로 가는 길

### 6.1 사실 — 그리고 QUANT.md §4.1 이 낙관적이었던 지점

`_dyn_quant_pack_4bit_weight` 의 **비-KleidiAI** 커널
(`aten/src/ATen/native/cpu/int4mm_kernel.cpp:958-968`):

```cpp
packed_weights = packed_weights.to(kFloat);
auto weight_reshaped = weights.reshape({-1}).to(kFloat);        // uint8 -> float32
auto scales_zeros_reshaped = scales_zeros.reshape({-1}).to(kFloat);
auto res = at::cat({weight_reshaped, scales_zeros_reshaped, /*bias*/}, 0);
```

**팩된 결과가 `float32` 1-D 텐서입니다.** uint8 니블조차 `float` 로 캐스팅됩니다.
`_meta_registrations.py:4283` 의 `weights.new_empty(size, dtype=torch.float)` 가 그것과 일치합니다.

    즉 QUANT.md §8 의 2 번 항목이 기대치로 적은 **"가중치 7.8× 압축" 은
    KleidiAI 가 켜졌을 때만 참입니다.** 우리가 재현할 수 있는 폴백 레이아웃은
    압축률이 0 이고, 오히려 원본보다 **큽니다** (4-bit 니블 하나당 4 바이트).

두 개의 게이트가 서로 다르다는 것도 확인됐습니다:

- 메타 등록의 게이트는 `torch._C._has_kleidiai` = `AT_KLEIDIAI_ENABLED()` — **컴파일 타임**.
- 실제 CPU 커널의 게이트는 `can_use_kleidiai()` (`int4mm_kernel.cpp:778-794`) 로,
  거기에 **런타임 `cpuinfo_has_arm_neon_dot()`** 이 더 붙습니다.

그리고 **이 환경에는 그 두 op 을 부르는 것이 없습니다.** 벤더 트리와 설치된 torch 전체에서
호출자는 `test/test_linalg.py` 의 단위 테스트뿐이고, `torchao` 는 설치되어 있지 않습니다
(`ModuleNotFoundError`). 사용자 대면 API 가 어디인지는 **판정하지 못했습니다** — 그 패키지가
디스크에 없기 때문입니다.

### 6.2 candle `QTensor` 의 실제 제약

| | |
|---|---|
| 활성화 dtype | `QMatMul::forward` 는 **`f32` 또는 `f16` 만** 받습니다 (`"Expected f32/f16"`). **`bf16` 을 안 받습니다** — 이 저장소의 기본 경로가 bf16 이므로 넓혀야 합니다 |
| 랭크 | 구성은 임의 랭크로 되지만 **소비자 둘 다 2-D 를 요구**합니다 (`QMatMul::cpu_fwd`, `embedding` 모두 `dims2()?`) |
| 역양자화 | `dequantize(&Device) -> Tensor`, `dequantize_f16` 있음 |
| 내부 가변성 | **있습니다.** `repacked_qs: OnceLock<Option<Vec<u8>>>` 를 `cpu_fwd(&self)` 안에서 `get_or_init` 합니다. 즉 **공유 참조 뒤에서 한 번 쓰입니다** — GIL 규약에 한 줄이 필요합니다 |
| 장치 | `quantize` 는 `Device` 를 암묵적으로 소스에서 가져오고, `quantize_onto` 는 명시적으로 받으며 소스가 CPU 일 것을 요구합니다 |

### 6.3 `Repr` 세 번째 변형의 실제 비용

`tensor.rs` 의 `Repr` 위를 **와일드카드 없이** 매치하는 곳이 **14 개**입니다
(`tensor.rs` 12 개, `aten.rs` 2 개). 와일드카드가 없으므로 변형을 늘리면 **14 곳 전부에서
컴파일이 깨지고**, 그것은 좋은 성질입니다 — 조용히 지나가는 자리가 없습니다.

그런데 그 14 개는 **설계 논쟁이 아닙니다.** 관문은 하나입니다:

```rust
pub fn tensor(&self) -> PyResult<&Tensor> {
    match &self.inner {
        Repr::Dense(tensor) => Ok(tensor),
        Repr::Meta { .. } => Err(no_data()),
    }
}
```

`QTensor` 는 `Tensor` 가 아니므로 **`&Tensor` 를 만들어 줄 방법이 없습니다.** 따라서 새 변형은
`Meta` 와 똑같이 `Err` 를 돌려주게 되고, 그러면 **96 개 커널이 자동으로 거절합니다.**
그 enum 의 doc comment 가 적어둔 목적("어떤 커널도 meta 텐서의 저장소를 읽지 못한다는 것을
타입의 성질로") 이 그대로 양자화에도 적용됩니다.

    즉 enum 을 늘리는 것은 기계적이고, 어려운 것은 **어느 op 에게 새 팔을 가르칠 것인가** 입니다.
    최소 집합은 `mm` / `linear` / `embedding` 셋입니다.

### 6.4 권고 — (b), 다만 순서가 있다

> **정정 (문서 감사, 2026-09):** 권고가 채택됐다. `docs/graph/QUANT2.md`(같은 날 22:08 착지, 이 문서의
> 18:23 착지보다 약 4시간 뒤 — `git log` 로 순서 확인)가 `Repr::Quantized(Arc<QTensor>)` 를
> `tensor.rs` 에 실제로 넣었다(오늘도 존재, 확인함). SmolLM2-135M q8_0/q4_0 로 20/20 토큰 일치를
> 검증축으로 삼았다 — 아래 §6.4 순서표의 1번("양자화용 검증 축")과 2번("`Repr::Quantized` +
> `dequantize` 왕복")이 실제로 그 순서로 닫혔다. `torch.int8` dtype 자체는 오늘도 여전히 이름을
> 대고 거절한다(`torch.tensor([1,2,3], dtype=torch.int8)` → `NotImplementedError: ... dtype not
> storable by the candle backend`) — §6.4 항목 2 의 "결함이 아니라 정확한 보고" 라는 판단은
> 그대로 유효하다. 아래 원문은 권고 시점 그대로 남긴다.
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs Quantized present -->

**추천: (b) candle `QTensor` 를 `Repr` 의 세 번째 변형으로.** 근거는 셋입니다.

1. **(a) 는 지금 형태로는 아무것도 사지 못합니다.** 비-KleidiAI 팩 레이아웃은 압축이 0 이고
   (§6.1), 커널은 어차피 우리가 써야 하며, 이 환경에 호출자가 없습니다. 남는 값은 **상류
   op 이름과의 표면 호환** 하나뿐입니다. 그것이 필요해지는 날 (b) 위에 이름만 얹으면 됩니다 —
   상류의 계약이 "KleidiAI 가 있으면 팩된 가중치는 불투명한 블롭" 이므로, 우리도 불투명한
   것을 돌려줄 자유가 있습니다.
2. **QUANT.md §8 의 5 번(candle 에 `I8` 추가)은 저장 타입만 삽니다.** candle 에는 int8
   matmul 커널이 **없습니다.** `I8` 을 얻어도 `_int_mm`·`_weight_int8pack_mm` 의 커널은 우리가
   씁니다. 상류 PR 의 불확실성(`docs/design/CANDLE_DEPS.md` §2a)을 그 대가로 치를 이유가 없습니다.
   **`dtype.rs` 가 `torch.int8` 을 이름 대고 거절하는 것은 결함이 아니라 정확한 보고입니다.**
3. **(b) 만이 반대편에 이미 최적화된 커널이 있습니다** — QUANT.md §5.2 의 기기 Q4K 프리필
   1.60×(그리고 `+dotprod` 로 3.29×), 호스트 디코딩 15×.

**다만 먼저 있어야 하는 것이 코드가 아닙니다.** 이 저장소의 모든 판정은 상류와의 **비트 일치**이고
(골든 2744 개, `_exact_value_check`, 이 회차의 검사 전부), **7.5% RMS 오차는 어떤 허용오차
정책으로도 "상류와 같다" 가 아닙니다**(QUANT.md §7). 검증 축 없이 (b) 를 착지시키면
**어떤 테스트도 옳다고 말할 수 없는 것을 착지시키는 것**입니다 — `CLAUDE.md` §5.5 가 금지하는
바로 그 모양입니다.

권장 순서:

| # | 단계 | 왜 이 순서인가 |
|---|---|---|
| 1 | **양자화용 검증 축** (perplexity 또는 토큰 일치율 하네스) | 없으면 2~4 의 착지 판정이 불가능. QUANT.md §8 의 6 번 |
| 2 | `Repr::Quantized(QTensor)` + `tensor()` 가 `Meta` 처럼 거절 + `dequantize` 왕복 | 14 개 매치는 기계적. 이 단계만으로 **양자화 텐서를 담고 되돌릴 수 있는지**가 판정된다 |
| 3 | `mm`/`linear` 이 양자화 가중치를 만나면 `QMatMul` | 활성화가 bf16 이면 f32 로 넓혀야 한다(§6.2). 그 비용을 재고 나서 판단 |
| 4 | 상류 4-bit op 이름 (`_dyn_quant_*`) 을 2~3 위에 얹기 | 표면 호환이 필요해질 때. 재양자화가 일어나면 **그것을 문서에 적어야 한다** — 상류의 계약은 "이 가중치를 팩하라" 이지 "다시 양자화하라" 가 아니다 |
| 5 | `+dotprod` (QUANT.md §6) | 기기에서 2.05× 지만 SIGILL 위험. **3 번과 함께 다루고 단독으로 하지 말 것** |

**하지 말아야 할 것 하나.** 4.2 가 보인 대로 candle 의 네이티브 `float16` GEMM 은 `float16` 으로
누산합니다. 속도가 매력적이라는 이유로 `mm` 을 거기로 보내면 `docs/numerics/BF16.md` 가 되돌린 결함을
`float16` 쪽에 다시 만드는 것입니다. **그 경로는 열지 마십시오.**

---

## 7. 확인하지 않은 것

| 항목 | 상태 |
|---|---|
| 안드로이드·iOS 기기 측정 | **없음.** 컴파일만 확인 (§5.3) |
| `gemm-f16` 이 `neon` 만 있는 타깃에서 f32 누산으로 가는가 | **미측정** (§4.2). 런타임 디스패치라 소스로는 판정 불가 |
| 모델 규모(DRAM 바운드)에서 융합 gemv 의 부호 | **미측정** (§4.3). 잰 것은 캐시에 들어가는 크기뿐 |
| `bfloat16` 좁히기를 더 줄일 수 있는가 | **부분적.** 11 연산 중 1 개를 `vsraq` 로 줄일 여지가 보이지만 재지 않았습니다. `FEAT_BF16` 없이 한 자릿수 개선은 없어 보입니다 |
| SmolLM2 실모델의 로짓·토큰 | **미측정.** 이 회차는 비트 동일성만 주장하므로 모델 수준 재측정을 하지 않았습니다. **비트가 같으면 모델도 같다**는 것이 그 논거이고, 그 전제는 §5 가 검사합니다 |
| QUANT.md §9.1 (accelerate 가 f16 을 2 배 느리게) | **candle 층에서는 재현되지 않습니다** — accelerate 켬 0.6074 ms, 끔 0.6075 ms (같은 프로세스, 같은 크레이트). 대신 시임 빌드 사이에서 f16 변환이 0.338 ~ 0.669 ms 로 **흔들리는 것**을 관측했습니다. 새 경로는 네 번의 측정에서 0.0715 ~ 0.0742 로 안정적입니다. **원인은 여전히 설명하지 못했고, 새 경로에서는 증상이 사라졌습니다** |
| `torchao` 의 사용자 대면 API | **판정 불가.** 패키지가 디스크에 없습니다 (§6.1) |
| `QTensor` 하위 타입들의 `Send`/`Sync` | **부분 확인.** `QTensor` 자체에 부정 impl 은 없고 `Arc<QTensor>` 가 쓰이므로 성립하지만, `QStorage` 변형들의 경계는 확인하지 않았습니다 |

---

## 8. 재현

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cd /Volumes/macMini/worktrees/bw-dtype
bash vendor/vendor_torch.sh
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-dtype
PY=/Volumes/macMini/caches/spike-venv/bin/python
bash vendor/install_shim.sh                      # PATH 에 cargo 가 있어야 함
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib   # <- 빼먹으면 §5.4 의 거짓 초록

PYTHON=$PY sh rust/torch_c/pytests/run.sh        # 175
$PY tools/golden/compare.py                      # 2744/2744 ops=118
$PY rust/torch_c/pytests/verify_schemas.py       # 3076/3076
( cd rust/torch_c && cargo test --release )      # 7
```

측정 스크립트는 저장소 밖 `/Volumes/macMini/caches/dtype-scratch/` 에 있습니다:

| 파일 | 무엇 | 절 |
|---|---|---|
| `bench.py` | dtype × 모양 스윕 (QUANT.md §10 에서 그대로 가져옴) | §3 |
| `cvtbench/src/main.rs` | 변환·add·matmul·gemv 층별 분해 | §2.1, §4.2 |
| `cvtbench/src/bin/b2.rs` | 손으로 쓴 branchless 커널, f16 matmul 누산 판정 | §4.2 |
| `cvtbench/src/bin/b3.rs` | NEON 커널과 그 정확성 (전수 비트 패턴) | §2.4 |
| `cvtbench/src/bin/b4.rs` | **할당을 타이밍에 포함**한 변환 | §2.3 |
| `cvtbench/src/bin/b5.rs` | 융합 gemv 프로토타입 | §4.3 |

§3 의 A/B 는 `6076cf4` 를 별도 타깃 디렉터리로 빌드해 두 벤더 트리에 각각 심고
`PYTHONPATH` 로 갈아끼우며 **base → new → base → new** 로 돌린 것입니다.
`git stash push -- <파일>` / `git stash pop` 을 썼고, 복원 후 `diff -r` 로 소스가 동일함을
확인했습니다.

---

## 9. 보고 분류

`CLAUDE.md` §5.3.

| 종류 | 무엇 |
|---|---|
| **기능 추가** | `rust/torch_c/src/reduced.rs` — 축소 float 변환 4 개와 융합 산술 4 개. `aten.rs` 의 opmath 경로 80 곳이 그것을 탄다 |
| **결함 수정** | 없음. **한 비트도 바뀌지 않았습니다** — 골든 2744 개와 스키마 3076 개가 변화 없음 |
| **테스트 추가** | Rust 5 개(전수 비트 패턴 포함) · `test_shim.py` 4 개 (171 → 175). 전부 허용오차 없음 |
| **측정** | 층별 분해(§2) · A/B(§3) · f32 대비 위치(§3.2) · candle f16 GEMM 의 누산 dtype(§4.2) · 융합 gemv 프로토타입(§4.3) |
| **문서 정정** | `docs/devices/DEVICE_ABS.md` §5.1 의 "`cargo test` 가 죽는다" 는 **rustc 1.98 에서 더 이상 참이 아닙니다**(§1) · `docs/graph/QUANT.md` §8 2 번의 "7.8× 압축" 은 **KleidiAI 경로에서만** 참입니다(§6.1) · QUANT.md §3.5 의 dead-strip 관찰은 맞지만 **그 커널은 쓰면 안 됩니다**(§4.2) |
| **삭제** | 없음 |
