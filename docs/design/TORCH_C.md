# `torch._C` — 바닥 놓기

A/B 결정이 A(candle 위 PyO3 어댑터)로 확정된 뒤, `rust/torch_c` 를 함수 하나짜리 스파이크에서
**실제 시작점**으로 키운 작업의 기록입니다.

**목표는 커버리지가 아니라 바닥입니다.** 구현한 aten op 은 3 개이고, 그것이 적은 것이 아니라
의도한 것입니다 — DESIGN.md §6 이 "op 집합을 미리 세는 계획은 폐기한다" 고 정한 이상, 이 단계에서
확인해야 하는 것은 *얼마나 만들었는가*가 아니라 **다음 op 이 붙을 자리가 옳은가**입니다.

---

## 0. 한눈에

| | |
|---|---|
| 텐서 엔진 | `candle-core 0.11.0`, 기본 피처 전부 끔 |
| 노출한 타입 | `TensorBase` · `dtype` · `device` |
| 구현한 aten op | `aten.full.default` · `aten.add.Tensor` · `aten.mm.default` |
| 미구현 op | `NotImplementedError: aten op not implemented in torch._C shim: <이름>` |
| 세 타깃 빌드 | **전부 통과** (종료 코드 0) |
| 호스트 실제 임포트 | **통과** — `_C.so` 로 이름 바꿔 13 개 스모크 테스트 |
| 하드코딩된 iOS 경로 | **제거됨** — 환경 변수 → `build.rs` 주입 |

---

## 1. 왜 이 구조인가

### 파일 배치

```
rust/torch_c/
├─ Cargo.toml            candle-core, PyO3
├─ build.rs              타깃별 링크 배선 중 "경로" 인 것
├─ .cargo/config.toml    타깃별 링크 배선 중 "상수" 인 것
├─ src/
│  ├─ lib.rs             #[pymodule] _C — 등록만
│  ├─ tensor.rs          TensorBase — 텐서의 정체성(shape · dtype · device)
│  ├─ dtype.rs           torch.float32 … 을 _C 가 소유하는 타입으로
│  ├─ device.rs          torch.device — 살아 있는 백엔드가 아니라 라벨
│  ├─ aten.rs            디스패치 단일 관문 + 구현된 op
│  └─ err.rs             예외 문구. §6 의 발견 장치가 여기 얹힘
└─ pytests/
   ├─ test_shim.py       빌드된 _C.so 에 대고 도는 스모크 테스트
   └─ run.sh             빌드 → `_C.so` 로 개명 → 실행
```

### `TensorBase` 라는 이름은 임의가 아니다

상류 `torch/_tensor.py` 의 첫 줄이 `class Tensor(torch._C.TensorBase):` 입니다. 그리고
IMPORT_WALLS §5 가 추론 중 **실제로 파이썬이 실행되는 14 개 모듈** 중 하나로 `torch._tensor` 를
지목했습니다. 즉 이 이름은 벤더링한 파이썬 트리가 **반드시 상속할 이름**이고, 지금 맞춰 두는 데
드는 비용은 0 이지만 나중에 틀린 것을 고치는 비용은 벤더링 파일 수정입니다.

스모크 테스트가 실제로 `class Tensor(_C.TensorBase)` 를 만들어 상속 가능함을 확인합니다
(`#[pyclass(subclass)]`).

### 산술은 `TensorBase` 에 없다 — 관문이 하나여야 하기 때문

`aten.rs` 의 `_aten_dispatch(op, *args, **kwargs)` 가 **유일한 입구**입니다. `TensorBase.__add__`
같은 편의 메서드를 두면 그 경로로 들어온 호출은 계측기에 잡히지 않습니다. DESIGN.md §6 의
"발견은 shim 이 스스로 한다" 는 **모든 호출이 한 지점을 지날 때만** 성립합니다.

```python
>>> _C._aten_dispatch("aten.embedding.default")
NotImplementedError: aten op not implemented in torch._C shim: aten.embedding.default
```

문구는 `err.rs::aten_not_implemented` 한 곳에만 있습니다. 트레이스를 긁어 다음 op 을 뽑는 도구가
이 문자열에 매칭할 것이므로, 호출부마다 손으로 쓰면 표현이 흔들립니다.

**오버로드가 키의 일부입니다** (`aten.add.Tensor`, `aten.add` 아님). torch 는 커널에 닿기 전에
오버로드를 이미 해소하고, `add.Tensor` 와 `add.Scalar` 는 스키마가 다릅니다. 뭉치면 하나를
구현하고 둘을 구현했다고 계측기가 거짓말합니다.

`_C._aten_implemented()` 가 구현된 이름 목록을 돌려줍니다. 파이썬 쪽이 사본을 들고 있으면
어긋나므로 물어보게 했습니다. 스모크 테스트에 **"목록에 있는 이름이 전부 실제로 디스패치되는가"**
불변식이 있습니다 — 목록에만 있고 폴백으로 떨어지는 이름이 생기면 계측기가 커버리지를 과장합니다.

### `device` 는 candle 의 `Device` 를 감싸지 않는다

torch 에서 `torch.device("cuda")` 는 CPU 전용 빌드에서도 **만들어집니다.** 라벨이고, 쓸 때만
실패합니다. candle 의 `Device` 는 반대로 enum variant 가 살아 있는 핸들을 들고 있어 백엔드가 없는
장치를 표현할 수 없습니다. 그래서 라벨(`type`, `index`)만 저장하고 쓸 때 `resolve()` 합니다.
실패 지점이 torch 와 같은 자리에 놓입니다.

### `dtype` 은 파이썬 상수가 아니라 `_C` 가 소유하는 타입이다

`torch.float32` 는 상류에서도 C 가 정의한 타입의 인스턴스이고, `torch/__init__.py` 는 그것을
re-export 할 뿐입니다. 그러므로 shim 이 타입을 소유해야 합니다 — 이름만 흉내 내면
`isinstance(x, torch.dtype)` 이 깨집니다.

**매핑은 전단사가 아닙니다.** 대응이 확실한 쌍만 등록하고, 나머지는 이름을 빌려주지 않습니다.

| torch | candle | |
|---|---|---|
| `float64` `float32` `float16` `bfloat16` | `F64` `F32` `F16` `BF16` | |
| `int64` `int32` `int16` | `I64` `I32` `I16` | |
| `uint8` `uint32` | `U8` `U32` | |
| `float8_e4m3fn` | `F8E4M3` | |
| **`bool`** | — | **candle 에 없음** |
| **`int8`** | — | **candle 에 없음** (부호 있는 8 비트 부재) |
| `complex64` `complex128` | — | 없음 |
| — | `F6E2M3` `F6E3M2` `F4` `F8E8M0` | torch 에 이름이 없음. `torch._C.dtype(candle:f4)` 로 표기 |

가까운 이웃에 얹지 않은 이유는 DESIGN.md §5 가 A 의 주 리스크로 지목한
**"수치 불일치가 조용히 번짐"** 그 자체이기 때문입니다. `bool` 을 `uint8` 로 별칭하면 마스킹
경로(`masked_fill`, `any`, `ne`)가 조용히 다른 답을 냅니다.

**`torch.bool` 의 부재는 이 계획의 실제 항목입니다.** CORE_ATEN §2 목록의
`aten.bitwise_and.Tensor` · `bitwise_not` · `bitwise_or` · `any.default` · `any.dim` ·
`eq.Scalar` · `ne.Tensor` · `lt.Scalar` · `masked_fill.Scalar` 가 전부 불리언 텐서를 오갑니다.
아래 §5 에 남깁니다.

---

## 2. 구현한 op 3 개와 고른 이유

**같은 종류를 셋 고르지 않았습니다.** `add` 와 `mul` 과 `sub` 를 구현하면 세 개가 아니라 한 개를
세 번 구현한 것입니다. *종류*가 다른 셋을 골라야 패턴이 일반화된다는 것이 보입니다.

| op | 종류 | 왜 이것인가 |
|---|---|---|
| `aten.full.default` | **팩토리** | 텐서를 만드는 경로가 디스패처를 지나야 한다. 팩토리가 없으면 모든 텐서가 계측기가 못 보는 뒷문으로 들어온다 |
| `aten.add.Tensor` | **원소별 이항** | torch 와 candle 의 의미론 차이가 **가장 싸게 드러나는 지점**. 브로드캐스팅은 양쪽이 같고, dtype 승격은 torch 에만 있다 |
| `aten.mm.default` | **행렬곱** | 유일하게 *뜨거운* op. IMPORT_WALLS §5 의 살아 있는 10 개 모듈 중 계산이 무거운 것은 `nn.Linear` 뿐이고 그것이 이 op 이 된다 |

셋 다 CORE_ATEN §4 의 "우리 모델이 실제로 부르는 Core ATen 원시" 목록 안에 있습니다.

### 각각에서 드러난 것

**`full.default`** — dtype 추론 규칙이 파이썬 타입에 걸립니다. torch 는 정수 fill 이면 `int64`,
아니면 기본 부동소수 dtype 을 씁니다. 파이썬 `bool` 은 `int` 의 서브클래스라 정수 분기로 떨어지는데
torch 는 `torch.bool` 을 줍니다 — **candle 에 `bool` 이 없어 지금은 맞출 수가 없습니다.**
§5 에 남깁니다.

그리고 `layout` · `pin_memory` 는 **무시하지 않고 거부합니다.** 조용히 버리면 호출이 성공한 것처럼
보이면서 답만 다릅니다. `layout=torch.sparse_coo` 가 그대로 통과하는 쪽이 미구현으로 터지는 쪽보다
훨씬 나쁩니다.

**`add.Tensor`** — **dtype 승격을 구현하지 않았고, 추측하지도 않습니다.**

```python
NotImplementedError: aten.add.Tensor: dtype promotion not implemented in torch._C shim: f32 vs f64
```

candle 은 dtype 이 다르면 거부하고, torch 는 승격표에 따라 올립니다. 승격표를 대충 짜 넣으면
정확히 §5 가 경고한 조용한 수치 오차가 됩니다. **미구현으로 두는 것이 이 단계에서 옳은 선택**이고,
승격표는 그 자체로 따로 정해야 할 항목입니다 (§5).

`alpha` 는 `affine(alpha, 0)` 으로 처리합니다. 브로드캐스팅은 `broadcast_add` 가 numpy 규칙을
따르므로 torch 와 같습니다.

**`mm.default`** — candle 의 `matmul` 은 **배치를 받습니다.** 그대로 노출하면 `mm` 이 아니라
`bmm`/`matmul` 을 구현한 것이 되고, torch 에서는 서로 다른 오버로드입니다. 그래서 2 차원을
명시적으로 강제합니다. 스모크 테스트에 3 차원 입력이 거부되는지 확인하는 케이스가 있습니다.

### 일부러 구현하지 않은 것

**`aten.view.default`.** DESIGN.md §4 가 candle 의 알려진 임피던스로 지목한 바로 그 op 입니다 —
candle 은 복사 지향이고 torch 의 `view` 는 별칭(alias)입니다. 여기에 더해 `-1` 추론이 candle 의
`reshape` 에는 없습니다. **빨리 짜 넣을 수 있지만 그러면 안 되는 종류**입니다. 별칭 의미론을
어디까지 재현할지는 KV 캐시 갱신 경로(`transformers` 가 in-place 로 밟는 곳)와 함께 한 번에
정해야 할 설계 판단이고, 이번 작업의 "바닥 놓기" 범위 밖입니다.

### 뒷문 하나 — `_tensor_from_flat`

값이 있는 텐서를 만들 aten 경로가 아직 없습니다. `torch.tensor([...])` 는 파이썬 계층의
팩토리이고 `lift_fresh` / `_to_copy` 로 내려가는데, **`aten.lift_fresh.default` 는 CORE_ATEN §0 이
"Core ATen 도 아니고 분해 테이블에도 없는" 두 개 중 하나로 특정한 op** 입니다. 그것을 어떻게 다룰지
정하기 전에 임의로 구현하면 안 되므로, 테스트가 실제 데이터를 넣을 수 있게 **밑줄 접두사에 aten
이름이 없는** 임시 함수를 하나 두었습니다. 승격 대상이 아니라 **삭제 대상**입니다.

---

## 3. 빌드 배선 — 하드코딩된 iOS 경로를 걷어냈다

RUST_CROSSBUILD.md §0.5 가 지적한 항목입니다.

```toml
# 이전 — 커밋된 파일 안의 절대 경로. 다른 기계에서 그대로 깨진다
[target.aarch64-apple-ios]
rustflags = ["-C", "link-arg=-F/Volumes/macMini/caches/target-python/arm64-iphoneos", ...]
```

**환경 변수를 `.cargo/config.toml` 에 쓸 수는 없습니다.** cargo 는 `rustflags` 값 안에서 환경 변수를
전개하지 않습니다. 그래서 `build.rs` 로 옮겼습니다 — 이것이 RUST_CROSSBUILD.md 가
"`Cargo.kt` 가 주입해야 할 값" 이라고 적은 형태와 같습니다.

### 새 규약

| 변수 | 누가 주는가 | 언제 |
|---|---|---|
| `TORCHNATIVE_PYTHON_FRAMEWORK_DIR` | 빌드 드라이버 (`Cargo.kt`, 또는 개발 셸) | **iOS 실기기 타깃만** |
| `PYO3_CONFIG_FILE` (`suppress_build_script_link_lines=true`) | 같음 | **iOS 실기기 타깃만** |

`build.rs` 가 `TARGET` 을 보고 iOS 실기기일 때만 다음을 방출합니다.

```
cargo::rustc-link-search=framework=<dir>     # clang 의 -F <dir>
cargo::rustc-link-lib=framework=Python       # clang 의 -framework Python
```

시뮬레이터와 macOS 는 `-undefined dynamic_lookup` 이면 되고 **경로가 없으므로**
`.cargo/config.toml` 에 그대로 둡니다. 분할 기준은 **"타깃의 상수인가, 빌드 기계의 경로인가"**
입니다.

### 규약이 실제로 작동하는지 확인했다

| 조건 | 결과 |
|---|---|
| 변수 미설정 | **EXIT=101**, `TORCHNATIVE_PYTHON_FRAMEWORK_DIR is not set, and target aarch64-apple-ios has no linkable libpython …` |
| 변수가 엉뚱한 디렉터리(`…/lib`) | **EXIT=101**, `does not contain Python.framework. It must point at the directory holding the framework …` |
| 변수가 **원래와 다른 경로**(`/tmp/bw-alt-fw`) | **EXIT=0**, `otool -L` 에 `@rpath/Python.framework/Python` |

첫 줄이 중요합니다 — **하드코딩이 정말로 사라졌다는 증거**입니다. 남아 있었다면 변수를 지워도
빌드가 성공했을 것입니다. 셋째 줄은 경로가 진짜로 재배치 가능함을 보입니다.

`build.rs` 는 잘못된 디렉터리를 **링커가 아니라 자기 자리에서** 잡습니다. 링커까지 흘려보내면
이 경로가 존재하는 이유였던 `library 'python3.13' not found` 로 되돌아옵니다.

### 곁다리로 잡힌 함정 — `--manifest-path` 는 `.cargo/config.toml` 을 안 읽는다

`pytests/run.sh` 를 처음에 `cargo build --manifest-path <crate>/Cargo.toml` 로 썼더니 링크가
`_Py*` 미정의 심볼 벽으로 실패했습니다. **cargo 의 config 탐색은 매니페스트가 아니라 작업
디렉터리 기준**이라 `-undefined dynamic_lookup` 이 통째로 빠진 것입니다. 스크립트는 `cd` 하도록
고쳤습니다. 하드코딩된 `-F` 와 같은 함정의 반대편이고, **링크 배선을 `.cargo/config.toml` 에
두는 것 자체가 취약하다**는 방증이므로 `Cargo.kt` 는 이 점도 인코딩해야 합니다.

---

## 4. 세 타깃 빌드 결과

**판정은 전부 종료 코드입니다.** 명령과 환경은 RUST_CROSSBUILD.md §0.5 그대로이고, iOS 에만
`TORCHNATIVE_PYTHON_FRAMEWORK_DIR` 이 추가됐습니다.

| 타깃 | 종료 코드 | 산출물 | 검증 |
|---|---|---|---|
| `aarch64-apple-darwin` (호스트) | **0** | `lib_C.dylib` 1,425,952 B | **`_C.so` 로 개명 → 임포트 성공 → 스모크 13/13 통과** |
| `aarch64-linux-android` | **0** | `lib_C.so` 2,280,968 B | `ELF 64-bit LSB, ARM aarch64`. Python 심볼 92 개 undefined |
| `aarch64-apple-ios` | **0** | `lib_C.dylib` 1,495,064 B | `Mach-O 64-bit dylib arm64`, `@rpath/Python.framework/Python`, `_Py*` 87 개 undefined |

undefined 로 남은 `Py*` 심볼은 **올바른 모양**입니다 — 로드 시점에 인터프리터가 해결합니다.

### 링크 성공은 증명이 아니다 — 그래서 호스트에서 돌렸다

```
$ ./pytests/run.sh
ok   test_add_broadcasts_and_applies_alpha
ok   test_add_refuses_to_guess_a_promotion
ok   test_device_is_a_label_not_a_backend
ok   test_dtype_is_a_type_owned_by_c
ok   test_every_advertised_op_is_actually_dispatchable
ok   test_full_infers_dtype_from_the_fill_value
ok   test_full_rejects_arguments_it_does_not_honour
ok   test_mm_is_2d_only
ok   test_mm_matches_torch
ok   test_module_loads
ok   test_tensor_base_is_subclassable
ok   test_tensor_exposes_shape_dtype_device
ok   test_unimplemented_op_names_itself

target=aarch64-apple-darwin implemented=['aten.add.Tensor', 'aten.full.default', 'aten.mm.default']
```

`mm` 은 상류 torch 값과 대조한 것입니다 —
`torch.mm([[1,2],[3,4]], [[5,6],[7,8]]) == [[19,22],[43,50]]`. 다만 **이 기계에 torch 가 설치돼
있지 않아 골든 값을 손으로 박았습니다.** 진짜 골든 대조는 §11 의 3 단계 몫입니다.

### 크기 — 스파이크 대비 3 배

| | 스파이크 (PyO3 만) | 지금 (+ candle) | 배수 |
|---|---|---|---|
| 호스트 | 470,928 B | 1,425,952 B | 3.0× |
| Android | 602,952 B | 2,280,968 B | 3.8× |
| iOS | 463,584 B | 1,495,064 B | 3.2× |

**전부 스트립 전 숫자입니다.** upytorch 압축 430KB, 선택 빌드 libtorch 4.5~20MB 라는 §5 의
범위 안에 있으므로 지금 걱정할 수치는 아니지만, **op 을 늘리기 전에 잰 기준선**으로 남깁니다.

---

## 5. 다음에 와야 하는 것

우선순위 순입니다.

### 1. candle-core 가 `tokenizers` 를 **비선택적으로** 끌고 온다

이번 작업에서 나온 가장 큰 발견입니다.

```
onig_sys v69.9.3 → onig v6.5.3 → tokenizers v0.22.2 → candle-core v0.11.0
```

`candle-core/Cargo.toml` 이 `cfg(not(target_arch = "wasm32"))` 에서 `tokenizers`(피처 `onig`)를
**optional 이 아닌 필수 의존성**으로 겁니다. 쓰는 곳은 `src/quantized/tokenizer.rs` — GGUF 안의
토크나이저를 읽는 편의 기능 하나입니다. 결과:

- **C 라이브러리(oniguruma)가 텐서 코어에 딸려 들어옵니다.** `onig_sys` 가 `cc` 로 C 를 빌드하므로
  타깃마다 C 크로스 툴체인이 필요합니다. 이번엔 Android · iOS 둘 다 통과했지만, WASM 이나 다른
  타깃에서 먼저 깨질 자리입니다. iOS 산출물에 `/usr/lib/libiconv.2.dylib` 의존이 새로 생긴 것도
  여기서 옵니다.
- **의존 트리가 129 크레이트 / 락파일 150 패키지**가 됐습니다.
- **중복입니다.** DESIGN.md §2 는 `tokenizers` 를 파이썬 계층이 이미 갖고 있는 것으로 셉니다.
  같은 크레이트가 `_C` 안에 한 벌 더 들어갑니다.

지금은 **막을 수단이 없습니다** (피처 게이트가 없음). 선택지는 (a) 그대로 두고 크기를 감수,
(b) `[patch.crates.io]` 로 이 의존을 뺀 포크를 물리기, (c) 상류에 optional 화 PR.
**정해야 할 항목이지 지금 정한 항목이 아닙니다.**

### 2. dtype 승격표

`add.Tensor` 가 지금 `NotImplementedError` 로 막는 자리입니다. torch 의 승격은 표로 공개돼 있으므로
(`torch.promote_types`) **추측이 아니라 이식**입니다. 다만 이식 범위(카테고리 승격, 스칼라 참여
규칙, `_to_copy` 와의 관계)를 정해야 하고, 이것 없이는 두 번째 이항 op 부터 전부 같은 벽에 막힙니다.

> **Correction (문서 감사, 2026-09):** 결정이 이 문서가 예상한 방향과 다르게 났습니다 —
> 이식하지 않기로 **의도적으로** 정해졌습니다. `torch.ops.aten.add.Tensor(float32_t, float64_t)`
> 는 오늘도 정확히 이 메시지로 거부합니다(실측, 2026-09). `docs/bindings/TENSORBASE.md` §2.3 이 그
> 결정을 명시합니다: "승격은 여전히 하지 않는다 ... `DESIGN.md` §5 가 candle 의 주된 위험으로
> 꼽은 '조용한 수치 드리프트' 를 만들지 않기 위한 기존 규칙." 즉 **미해결이 아니라 해결된
> 결정**입니다 — "텐서끼리는 승격하지 않고 이름을 댄다, 파이썬 스칼라는 wrapped-number 규칙을
> 재현한다"로 갈렸습니다.
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs same_dtype present -->

### 3. `torch.bool`

candle 에 없습니다. CORE_ATEN §2 목록의 마스킹·비교 op 9 개가 전부 여기 걸립니다
(`bitwise_*`, `any.*`, `eq.Scalar`, `ne.Tensor`, `lt.Scalar`, `masked_fill.Scalar`).
`U8` 을 불리언으로 쓰되 **dtype 라벨만 `torch.bool` 로 다는** 층을 `_C` 안에 둘지, candle 을
건드릴지 정해야 합니다. `full.default` 의 bool fill 도 여기 묶입니다.

> **Correction (문서 감사, 2026-09):** 닫혔습니다 — `docs/numerics/BOOL.md` 가 바로 이 결정을 다룹니다.
> `torch.bool` 이 지금 존재하고(`hasattr(torch, 'bool')` → `True`), `aten.eq.Scalar` 등의 결과가
> `torch.bool` dtype 을 답합니다(실측). 이 문서가 물었던 "`U8` 라벨링이냐 candle 을 건드리냐"는
> 전자로 정해진 것으로 보입니다 — 자세한 내용은 `docs/numerics/BOOL.md`.

### 4. `torch.ops.aten.<op>.<overload>` 진입로

지금 입구는 `_C._aten_dispatch(name, *args, **kwargs)` 하나입니다. 벤더링한 파이썬 트리는
`torch._C._jit_get_operation(name)` 이 **호출 가능한 객체를 돌려주는** 모양을 기대합니다.
디스패처 자체는 그대로 두고 그 위에 얇은 조회 층을 얹으면 되지만, **§11 의 1 단계
(`import transformers`)를 실제로 해 보기 전에는 어떤 모양이 요구되는지 확정할 수 없습니다.**
그러므로 1 단계가 먼저입니다.

> **Correction (문서 감사, 2026-09):** 닫혔습니다 — `docs/bindings/OVERLOAD.md` 가 `torch.<op>(...)`
> 사용자 표면(오버로드 해석기, `PythonArgParser::raw_parse` 재현)을 열었고,
> `docs/bindings/TENSORBASE.md` 가 `TensorBase` 메서드 쪽을 열었습니다(이 감사가 이미 두 문서를
> 확인함, 위 참고).

### 5. `aten.view.default` 와 별칭 의미론

§2 에서 미룬 항목. DESIGN.md §4 가 "스파이크 초기에 여기부터 확인해야 한다" 고 적은 바로 그것이고,
KV 캐시 갱신(`add_`, `copy_`) 과 한 묶음입니다.

> **Correction (문서 감사, 2026-09):** 닫혔습니다 — `docs/kernels/VIEWS.md` 가 정확히 이 항목을
> 다룹니다(이 감사가 round 1 에서 이미 확인: in-place 쓰기가 뷰를 관통, `aten.ge.Tensor` 등).
> 완전히는 아닙니다 — `docs/kernels/VIEWS.md` §6.4 가 `slice.Tensor` step>1 과 `view.dtype` 은 여전히
> 구조적으로(candle 의 `pub(crate)` storage 경계) 못 고친다고 남겨 둡니다.

### 6. abi3

**이번 작업에서 켜지 않았습니다** (지시대로). ABI3.md 의 권고는 `abi3-py313` 이지만
3.14.7 인터프리터 확인이 미완입니다. 다만 이번에 붙은 candle 이 그 판단에 새 변수를 넣지는
않습니다 — 경계 호출 비용은 `_C` 표면에만 걸리고 candle 은 그 아래이기 때문입니다.

### 7. Android · iOS 기기에서 실제 임포트

지금 확인한 것은 **호스트 임포트 + 두 타깃의 링크**입니다. 기기에서 `import _C` 가 되는지는
확인되지 않았고, PythonMultiplatform 의 임베딩 경로를 태워야 답이 나옵니다.

---

## 6. 미확인 항목

| 항목 | 상태 |
|---|---|
| Android · iOS **기기**에서의 임포트 | **미확인** — 링크만 확인. 위 §5-7 |
| 상류 torch 와의 골든 대조 | **미확인** — 이 기계에 torch 미설치. `mm` 기대값은 손으로 박음 |
| 스트립 후 배포 크기 | **미측정** — §4 는 전부 스트립 전 |
| `tokenizers`/`onig` 를 뺐을 때의 크기 | **미측정** — 뺄 수단부터 정해야 함 (§5-1) |
| 시뮬레이터(`aarch64-apple-ios-sim`) | **미검증** — 배선은 `.cargo/config.toml` 에 있으나 이번에 빌드하지 않음 |
| `candle-ug` 가 iOS 에서 제외되는 이유 | **미확인** — candle 이 `cfg(not(target_os = "ios"))` 로 끊어 두었음. 지금은 optional 이라 무해하나 나중에 커널 경로(§8)에서 걸릴 수 있음 |
| `affine` 의 정수 dtype 동작 | **미검증** — `alpha` 경로가 정수 텐서에서 어떻게 도는지 테스트하지 않음 |
| `_tensor_from_flat` 의 f64 경유 손실 | **알려진 제약** — 입력을 f64 로 받아 캐스팅하므로 큰 `int64` 를 정확히 넣을 수 없음. 임시 함수이므로 그대로 둠 |

---

## 7. 재현 방법

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target
DIST=/Volumes/macMini/caches/target-python
cd rust/torch_c            # cd 필수 — .cargo/config.toml 은 cwd 기준으로 찾는다

# 호스트 + 실제 임포트 검증
./pytests/run.sh; echo "EXIT=$?"

# Android
ANDROID_NDK_HOME=~/Library/Android/sdk/ndk/27.1.12297006 \
PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 \
PYO3_CROSS_LIB_DIR=$DIST/aarch64-linux-android/prefix/lib \
cargo ndk -t arm64-v8a --platform 21 build --release; echo "EXIT=$?"

# iOS — PYO3_CONFIG_FILE 내용은 RUST_CROSSBUILD.md §0.5 참고
TORCHNATIVE_PYTHON_FRAMEWORK_DIR=$DIST/arm64-iphoneos \
PYO3_CONFIG_FILE=<config> \
PYO3_CROSS=1 PYO3_CROSS_PYTHON_VERSION=3.13 \
PYO3_CROSS_LIB_DIR=$DIST/arm64-iphoneos/lib \
cargo build --release --target aarch64-apple-ios; echo "EXIT=$?"
```
