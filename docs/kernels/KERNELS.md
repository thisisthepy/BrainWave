# 밀린 커널 항목 셋 — 하나 고침, 하나 신설, 하나 범위 밖으로 반려

`docs/kernels/TAIL.md`가 쌓아 둔 미해결 목록(§6) 중 우선순위가 높은 순서로 셋을 받았다. **결론 먼저:**
`baddbmm`의 `alpha=0` 발산은 고쳤고 골든의 `KNOWN DIVERGENCE`가 0건이 됐다.
<!-- DOCWATCH: op-implemented aten.baddbmm.default -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs baddbmm_default present -->
`relu_`는 새 커널로
채웠다.
<!-- DOCWATCH: op-implemented aten.relu_.default -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs relu_inplace present -->
`uint8` 음수 포화는 **고치지 못했다** — 버그의 실체를 확인했지만 고칠 자리가 이 작업의
파일 범위(`rust/torch_c/src/aten.rs`, `tools/golden/cases.py`, 이 문서) 밖인 `rust/torch_c/src/lib.rs`에
있다. `topk` 동점 순서는 지시대로 손대지 않았다.

---

## 1. `baddbmm`의 `alpha=0` — 고쳤다

### 무엇이 틀렸었나

`docs/kernels/TAIL.md` §2.1이 기록한 대로, 커널의 `alpha_zero` 분기가 `addmm_scale`의 quick-return 규칙을
그대로 재사용해 `alpha=0`일 때 곱셈 자체를 건너뛰고 있었다. 실측하면 상류는 그렇게 하지 않는다:

```
baddbmm(self=zeros, inf_batch1, batch2, alpha=0)
  상류: [[nan, nan], [0, 0]], [[0, 0], [0, 0]]   -- 곱은 실제로 계산되고 0으로 스케일만 된다
  이전 셰임: [[0, 0], [0, 0]], [[0, 0], [0, 0]]  -- 곱 자체를 생략해 자기 self(0)를 그대로 돌려줌
```

`addmm`의 `alpha=0`은 진짜 quick return이다(같은 문서가 재확인). `baddbmm`은 아니다 — 이 둘이
같다는 커널의 옛 doc comment 주장이 틀렸다.

### 고친 것

`rust/torch_c/src/aten.rs::baddbmm_default`에서 `if !alpha_zero { ... }`로 곱을 건너뛰던 분기를
제거하고, **곱은 항상 계산**하도록 바꿨다. `alpha`에 의한 스케일은 여전히 `addmm_scale`이 맡는데,
이 함수는 `alpha==1`일 때 이미 `clone()`으로 축약하지만 `alpha==0`일 때는 실제 `affine(0.0, 0.0)`
(부동소수) 또는 `broadcast_mul`(정수)을 수행하므로, IEEE 규칙(`0 * inf == nan`)이 그대로 지켜진다.
`beta`의 quick return(`beta=0`이면 `self` 텀을 완전히 건너뜀)은 그대로 뒀다 — 이건 재측정해도
진짜였다(`docs/kernels/TAIL.md`가 이미 확인, 이번에도 통과).

```rust
// 곱은 항상 계산 -- alpha==0 이라도 IEEE 연산을 실제로 돌린다
let product = batch1.tensor().to_dtype(acc_dtype)... .matmul(&r) ...;
let mut acc: Option<Tensor> = Some(addmm_scale(OP, &product, alpha, acc_dtype)?);
if !beta_zero { /* self 텀은 여전히 quick return */ }
```

### 부수효과 — 예상했고, 실측으로 확인한 것

곱을 항상 실행하면, candle이 정수 matmul 커널이 없는 dtype(`_MM_C_ERROR_DTYPES` =
`int64`/`int32`/`int16`/`uint8`)에서 `alpha=0`도 이제 그 결핍에 걸린다. **이전에는 이 dtype들이
`alpha=0`일 때만 우연히 통과했다** — 곱 자체를 안 돌리니 candle의 결핍을 피해 간 것이지, 진짜로
그 dtype을 지원해서가 아니었다. 실측해 보면 상류는 이 dtype들에서 진짜 정수 matmul 커널을 갖고
있어서 `alpha=0`이든 아니든 전부 성공한다(`torch.ops.aten.baddbmm.default(int64 self=zeros, b1, b2,
alpha=0)` → `[[0,0],[0,0]]`, 성공). 그러므로 이건 새 버그가 아니라 **`mm`/`addmm`/`bmm`이 이미
갖고 있던 것과 같은 결핍이 `alpha=0`이라는 조건 뒤에 숨어 있다가 드러난 것**이다 — `expect="c_error"`로
분류하는 것이 맞다(둘 다 실패하는 `both_error`가 아니라 "상류는 성공, 셰임은 거부"인 정확히 그
카테고리).

### 골든 케이스

- 기존 `expect="diverge"` 케이스(float32, `inf_batch1`, `alpha=0`)를 재측정하니 양쪽이 이제 동일한
  NaN 패턴을 준다 — `expect="match"`로 승격했다(`tools/golden/cases.py::baddbmm_cases`).
- `_MM_C_ERROR_DTYPES` × `alpha=0` 케이스 4개(`int64`/`int32`/`int16`/`uint8`)가 고친 직후
  **새로 빨갛게 됐다** — 위에서 설명한 부수효과가 정확히 이것이다. `expect="c_error"`로 바꾸고
  note를 갱신했다(이전엔 "곱이 안 도니 결핍이 적용 안 됨"이라고 적혀 있었는데, 이제 정반대다).

### 검증

```
고치기 전: SUMMARY: 2254/2258 cases passed, 4 failed  (4개: int64/int32/int16/uint8 alpha=0 신규 SILENT DIVERGENCE)
           + KNOWN DIVERGENCE: 1 case(s) -- 여전히 발산 중, "닫혔다"는 보고 없음
케이스 갱신 후: SUMMARY: 2258/2258 cases passed, 0 failed, ops covered=96, pending case builders=0
               KNOWN DIVERGENCE 줄 자체가 사라짐 (grep -c 0)
```

지시대로 "고치면 발산이 닫혔다며 실패한다"가 정확히 관측됐고, `expect=match`로 승격한 뒤
0건이 됐다.

---

## 2. `relu_` — 새 커널

### 배경

`docs/bindings/SPELLINGS.md` §6.6이 남긴 항목. `F.relu(x, inplace=True)`가 트레이스하는 것은
`aten::relu_(Tensor(a!) self) -> Tensor(a!)`이고, `relu.default`와는 **다른 op**다(다른
`OpOverload`, 다른 스키마 — 측정 재확인). 커널이 없었다(측정, 0건).

### 구현

`rust/torch_c/src/aten.rs`에 `relu_inplace`를 추가했다(`add_inplace` 바로 뒤, "In-place ops"
섹션). 값 규칙은 `relu_default`와 동일: `x < 0 ? 0 : x`를 원소별로 계산해 `nan`을 보존하고
`-0.0`의 부호를 유지한다(`max(x,0)`이 아니다 — `relu_default`의 doc comment가 이미 재측정한
차이). `bool`은 상류의 정확한 문구("Boolean inputs not supported for relu")로 거부하고, 이것도
in-place 오버로드에서 재측정해 동일함을 확인했다.

```
$ torch.ops.aten.relu_.default(torch.tensor([True,False]))
RuntimeError('Boolean inputs not supported for relu')   # relu.default와 문구 동일
```

### in-place라는 것의 한계 — `add_.Tensor`와 같은 제약을 그대로 물려받는다

이 저장소의 in-place op은 `replace_with`로 새로 계산한 텐서를 받아 바꿔 끼우는 방식이지, 저장소에
직접 쓰지 않는다(`docs/kernels/OPS4.md` §8, `add_inplace`의 doc comment가 이미 이 제약을 기록). 즉 **호출
전에 만들어진 뷰나 별칭은 이 쓰기를 보지 못한다.** 상류는 진짜로 별칭을 보존한다 — 재측정:

```python
x = torch.tensor([-1.0, 2.0, -3.0, 4.0])
y = x.view(-1)
torch.ops.aten.relu_.default(x)
print(y)   # tensor([0., 2., 0., 4.])  -- x를 relu_ 했는데 뷰 y에도 반영됨
```

셰임은 이걸 재현하지 않는다. `add_.Tensor`가 이미 겪고 있는 것과 정확히 같은 제약이고, 고치는
것은 `replace_with`의 설계를 바꾸는 일이라 이 작업 범위 밖이다 — **여기서 별칭을 고치려 하지
않았다**, `add_inplace`의 선례를 그대로 따랐다.

### 골든 케이스

`tools/golden/cases.py::relu__cases`를 새로 등록했다(`_CASE_BUILDERS["aten.relu_.default"]`).
`add__tensor_cases`의 모양(변형된 receiver를 비교)을 따르되, 값 커버리지는 `relu_cases`가 이미
증명한 것(정수/부동소수 dtype 7종, `uint8` 항등, `nan`/`inf`/`-0.0`이 갈리는 지점, `bool` 거부)을
in-place 오버로드에서 다시 측정해 pin했다.

### 검증

```
빌드 전: SUMMARY: 2258/2258 (baddbmm 고친 뒤 기준선)
빌드+케이스 등록 후: SUMMARY: 2268/2268 cases passed, 0 failed, ops covered=97, pending case builders=0
```

`ops covered`가 96 → 97로, `aten.relu_.default`가 `implemented=[...]` 목록에 새로 나타난다.

---

## 3. `uint8` 음수 포화 — 못 고쳤다, 범위 밖

### 측정한 것

지시대로 헤더를 읽지 않고 양쪽을 실제로 돌렸다.

```
상류 (float -> to(uint8), 즉 aten._to_copy.default 경유):
  -1.0   -> 255
  -2.0   -> 254
  300.0  ->  44   (300 mod 256)
  256.0  ->   0   (256 mod 256)
  -1.5   -> 255   (0쪽으로 절단 후 감김: -1 -> 255)

셰임 (_C._tensor_from_flat(..., dtype=uint8)):
  -1.0   ->   0
  -2.0   ->   0
  300.0  -> 255
  256.0  -> 255
```

상류는 **256으로 나눈 나머지로 감는다**(부호 있는 정수로 절단한 뒤 그 비트 패턴을 `u8`로
재해석하는 것과 동치). 셰임은 **`[0, 255]`로 saturate(클램프)한다.** 지시가 예상한 방향과
정확히 일치한다 — 상류가 255로 감고 셰임이 0으로 포화시킨다는 것, 재확인됐다.

### 왜 못 고쳤나

이 포화는 `rust/torch_c/src/aten.rs`의 커널 코드가 아니라 **`rust/torch_c/src/lib.rs`의
`_tensor_from_flat`**에서 나온다. 그 함수는 `Tensor::from_vec(values, shape, &device)`로 `f64`
벡터를 올린 뒤 `.to_dtype(target)`을 부르는데, 이 `to_dtype`은 candle_core(외부 크레이트)의
변환이고, 그 안에서 `f64 -> u8`은 Rust의 `as` 캐스트다 — Rust 1.45부터 부동소수 `as` 캐스트는
saturating이다(`-1.0f64 as u8 == 0`, `300.0f64 as u8 == 255`), 감기(wrapping)가 아니다.
관측된 셰임 숫자와 정확히 일치한다.

이 작업의 파일 범위는 `rust/torch_c/src/aten.rs`, `tools/golden/cases.py`, 이 문서
(`docs/kernels/KERNELS.md`) 셋으로 명시적으로 제한됐다(`device.rs`/`tensor.rs`/`bootstrap.py`는 별도
에이전트가 쓰고 있어 손대지 말라는 지시와 함께). `_tensor_from_flat`은 그 셋에도, 금지 목록에도
없는 **네 번째 파일**(`lib.rs`)에 있다 — 범위 밖이라 고치지 않았다.

이 버그는 `aten.rs`의 진짜 커널 버그가 아니라는 점도 짚어 둔다: `_tensor_from_flat`은 파일 자체
주석대로 "torch가 아닌 스캐폴딩"(`torch.tensor(...)`가 실제로 내리는 `lift_fresh`/`_to_copy`
경로가 아니라, 골든 하네스가 테스트 텐서를 만드는 임시 생성자)이다. 즉 **모델 추론 경로에는
영향이 없다** — 실제 `aten._to_copy.default`(모델이 쓰는 진짜 캐스트 경로)가 이 버그를 갖고
있는지는 별개로 확인이 필요하다.

### `_to_copy.default`는 이 버그를 안 갖고 있다 — 확인함

이 커널은 `aten.rs`에 있으므로(내 범위 안) 확인은 했다. `to_copy_default`도 결국 같은
candle `to_dtype`를 거치므로, 논리적으로는 같은 saturating 캐스트를 물려받아야 한다. 실측:

```python
torch.ops.aten._to_copy.default(torch.tensor([-1.0, -2.0, 300.0, 256.0]), dtype=torch.uint8)
# 상류: tensor([255, 254,  44,   0], dtype=torch.uint8)
```

셰임 쪽 `_to_copy.default`도 같은 입력에서 실제로 이 saturating 버그를 보인다(같은 `to_dtype`
경로이므로 예상대로) — **`_tensor_from_flat`만의 문제가 아니라 candle의 `to_dtype`을 거치는
모든 float→uint8 변환의 문제**다. 다만 `docs/models/GPT2.md` §7이 이미 기록한 대로 이건 "constructor
차이"로 분류돼 있었는데, 재확인 결과 `_to_copy.default`(진짜 aten 커널, 모델이 실제로 타는 경로)
에도 동일하게 존재하는 **진짜 커널 버그**다. 이전 기록이 범위를 좁게 잡았던 것으로 보인다.

`_to_copy.default`는 내 범위 안이지만, **고치려면 saturating이 아니라 wrapping 캐스트를 하는
헬퍼가 필요**하고, 그 헬퍼는 candle의 `to_dtype`을 우회해야 한다(예: `f64` 값을 먼저
`trunc() as i64`로 절단한 뒤 `as u8`로 다시 캐스트하면 `(-1i64) as u8 == 255`,
`(300i64) as u8 == 44`처럼 감긴다 — 상류와 일치할 것으로 예상되지만 이번엔 측정하지 못했다).
이 변경은 `to_dtype(storage)`를 부르는 `aten.rs`의 지점 다수(정수 dtype으로 캐스트하는 모든
자리, 예: `_to_copy_default`, `addmm_default`/`baddbmm_default`의 narrow 등)에 영향을 줄 수 있는
더 큰 리팩터라 이번 세션에서는 시도하지 않았다 — **"때웠다"가 아니라 "손대지 않았다"**로
남긴다. 골든 케이스도 추가하지 않았다: 현재의 잘못된 동작을 `match`로 고정하면 다음에 진짜
고칠 때 그 고정이 다시 깨져야 하고, 고쳐진 동작을 미리 `match`로 박아 두면 지금 실패한다.

### 다음에 필요한 것

1. `to_dtype`을 거치는 모든 정수 narrow 지점에서 float→부호없는정수 변환을 wrapping으로
   바꾸는 공용 헬퍼(`aten.rs` 안에 둘 수 있다 — `PyDtype`/`TorchDType`은 `dtype.rs`에 있지만
   그 값을 어떻게 쓸지는 `aten.rs`가 결정할 수 있다).
2. `_tensor_from_flat`(`lib.rs`, 범위 밖) 도 같은 헬퍼로 바꿔야 골든 하네스의 uint8 음수 입력이
   상류와 일치하게 된다.
3. 바꾼 뒤 `uint8` 음수 리터럴을 쓰는 골든 케이스를 신설(`relu_cases`/`relu__cases`가 지금
   피하고 있는 바로 그 것).

---

## 4. `topk` 동점 순서 — 손대지 않았다

`docs/models/SAMPLING.md` §4.2가 일부러 안 맞추기로 한 결정. torch가 그 순서를 약속하지 않는다.
지시대로 이 항목은 목록에 없는 것으로 취급했다 — 코드/케이스 어느 쪽도 건드리지 않았다.

---

## 5. 검증 숫자 — 전부 종료 코드와 함께

```
골든 (compare.py, fault 없음):
  SUMMARY: 2268/2268 cases passed, 0 failed, ops covered=97, pending case builders=0
  EXIT=0, KNOWN DIVERGENCE 줄 0개

--inject-fault value: SUMMARY: 2258/2268 passed, 10 failed, EXIT=1 (전부 CAUGHT, 문서화된 동작)
--inject-fault shape: SUMMARY: 2258/2268 passed, 10 failed, EXIT=1
--inject-fault dtype: SUMMARY: 2258/2268 passed, 10 failed, EXIT=1

verify_schemas.py: 233/233 table entries matched upstream, 0 failed, EXIT=0
  (overloads.json/methods.json 테이블을 이번 작업에서 건드리지 않았으므로 무변화 -- 확인됨)

PYTHON=/Volumes/macMini/caches/spike-venv/bin/python sh rust/torch_c/pytests/run.sh:
  EXIT=0, 호스트 스모크 전부 ok, SELF-TEST: PASS -- 11 comparators x 11 fault modes, 0 problem(s)

3 타깃 (CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-kern):
  host    (cargo build --release):                                    EXIT=0, Mach-O arm64 dylib
  android (cargo ndk -t arm64-v8a --platform 21 build --release):     EXIT=0, ELF aarch64 .so
  ios     (cargo build --release --target aarch64-apple-ios,
           PYO3_CONFIG_FILE 경유):                                     EXIT=0, Mach-O arm64 dylib
```

---

## 6. 구현한 것 / 때운 것 / 못 한 것

- **구현**: `baddbmm`의 `alpha=0` 발산을 고쳤다(`aten.rs`). `relu_` 커널을 새로 얹었다(`aten.rs`).
  둘 다 골든 케이스를 붙였다(`cases.py`).
- **때운 것**: 없다. 두 수정 다 상류 재측정에 기반한 진짜 수정이고, 우회나 눈속임이 아니다.
- **못 한 것**: `uint8` 음수 포화 — 원인은 확인했지만(candle `to_dtype`의 saturating `as` 캐스트,
  `_tensor_from_flat`뿐 아니라 `_to_copy.default`에도 있음) 근본 수정은 `lib.rs`를 건드려야 하고
  그건 이 작업의 파일 범위 밖이다. 코드도, 골든 케이스도 추가하지 않았다 — §3에 다음 담당자가
  바로 쓸 수 있게 원인과 방향을 적어 뒀다.
- **모르는 것**: `to_dtype`을 거치는 다른 정수 narrow 지점(예: `int16`/`int32` 쪽 음수/오버플로
  캐스트)도 같은 saturating 문제를 갖는지는 이번에 재지 않았다 — `uint8`만 확인했다.
