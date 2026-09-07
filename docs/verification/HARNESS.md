# HARNESS — 검사기를 검사한다

`tools/golden/compare.py` 의 자가검사(`--inject-fault`)가 **비교기 열 개 중 하나만**
검사하고 있었습니다. 나머지 아홉은 한 번도 "틀린 답을 거부하는가" 를 확인받은 적이 없고,
그 아홉이 판정하던 케이스가 **1781 중 404 개(22.7%)** 입니다.

이 문서는 그 구멍의 크기, 메운 방법, 그리고 메우면서 나온 **비교기 세 곳의 실제 결함**을
적습니다.

---

## 1. 판정 요약

```
정상 실행            1781/1781 passed, 0 failed, ops covered=82, pending 0     exit 0
--inject-fault       11개 모드 전부                                            exit 1
--self-test          10 비교기 × 11 모드, problem 0, 미검사 비교기 0           exit 0
verify_schemas.py    170/170                                                   exit 0
스모크 (run.sh)                                                                exit 0
```

`--inject-fault` 의 기존 세 모드(value/shape/dtype)는 **그대로 exit 1** 입니다. 정상 실행의
1781/1781 도 그대로입니다 — 자가검사를 고치면서 평시 검사는 한 줄도 바꾸지 않았습니다.

---

## 2. 구멍은 왜 있었나 — 원래 이유는 정당했다

`--inject-fault` 는 `case.value_check` 가 붙은 케이스를 건너뜁니다. 그렇게 짠 이유가
코드와 커밋 메시지에 남아 있고, **당시에는 옳았습니다.**

`48d393e` ("Test: Seed golden cases for the twelve operators still to come") 시점에
`value_check` 를 쓰는 비교기는 셋뿐이었습니다.

| 비교기 | 왜 기본 파이프라인으로 못 보나 |
|---|---|
| `_dtype_shape_only_check` (`empty`) | **초기화되지 않은 메모리**라 "맞는 값" 자체가 없다 |
| `_scalar_match_check` (`is_floating_point`) | 파이썬 `bool` 이라 `.tolist()` 가 없다 |
| `_range_check` (`randint`) | 난수. 두 RNG 는 같은 시드로도 같은 수열을 못 낸다 |

그리고 `_corrupt()` 는 `.tolist()` 를 가진 **단일 텐서**를 가정하고 짜여 있었습니다.
그 셋에 값 결함을 주입하면:

- `empty` 는 값을 안 보므로 **정당하게** 안 잡히는데, 코드는 그것을 `COMPARATOR BUG` 로
  출력합니다 — **거짓 경보**입니다.
- `is_floating_point` 는 `.tolist()` 가 없어 `_corrupt` 가 그냥 원본을 돌려주고, 주입이
  일어나지 않았는데 일어난 것처럼 계속 진행합니다.

즉 **건너뛴 것은 회피가 아니라 당시 정확한 판단**이었고, `compare.py` 의 주석도 그렇게
적어 두었습니다. 모르고 부술 뻔한 자리였습니다.

### 정당하던 이유가 언제 정당하지 않게 되었나

`value_check` 가 **다중 결과 op 의 표준 비교 수단이 되면서**입니다. 지금 `value_check` 는
"이상한 op 세 개의 예외" 가 아니라 `(values, indices)`, `(output, logsumexp)`,
`(out, mean, rstd)`, `조각 리스트` 를 비교하는 **본류**입니다. 이것들은 전부 실제 수치
비교를 하고, 전부 자가검사 밖에 있었습니다.

측정한 크기:

| 비교기 | 케이스 수 | 담당 op |
|---|---:|---|
| `<default pipeline>` | 1377 | (자가검사 대상이던 유일한 하나) |
| `_rng_stream_check` | 168 | `normal_`, `uniform_` |
| `_pair_result_check` | 91 | `max.dim`, `sort`, `topk` |
| `_triple_result_check` | 41 | `native_layer_norm` |
| `_chunk_list_check` | 31 | `split.Tensor` |
| `_scalar_match_check` | 25 | `_local_scalar_dense`, `is_floating_point` |
| `_sdpa_pair_check` | 18 | `_scaled_dot_product_flash_attention_for_cpu` |
| `_dtype_shape_only_check` | 14 | `empty.memory_format` |
| `_range_check` | 12 | `randint.low` |
| `_topk_multiset_check` | 4 | `topk` (동점 · `sorted=False`) |
| **자가검사 밖 합계** | **404 / 1781** | **비교기 9종** |

그리고 더 정확히 말하면 구멍은 404 보다 큽니다. 옛 `--inject-fault` 는 **실행 전체에서
케이스 딱 하나**를 오염시켰습니다(`aten._softmax.default` 의 첫 케이스). `exit 1` 이
증명한 것은 "기본 파이프라인이 어떤 한 종류의 틀린 답 하나를 거부한다" 이지,
"비교기들이 살아 있다" 가 아니었습니다.

---

## 3. 어떻게 메웠나

두 가지를 바꿨습니다. **둘 다 `compare.py` 안에서 끝납니다** — `cases.py` 와
`rust/torch_c/` 는 한 글자도 건드리지 않았습니다.

### 3.1 주입 단위를 "실행당 1건" 에서 "비교기당 1건" 으로

`--inject-fault MODE` 는 이제 **비교기마다 대표 케이스 하나씩** 오염시킵니다. 한 번
돌리면 열 개 비교기가 각각 그 모드를 잡는지 한꺼번에 보입니다.

```
INJECTION VERDICT (--inject-fault value) -- one representative case per comparator:
  CAUGHT  <default pipeline>        [aten._softmax.default]
  CAUGHT  _chunk_list_check         [aten.split.Tensor]
  blind   _dtype_shape_only_check   [aten.empty.memory_format] -- 초기화 안 된 메모리...
  CAUGHT  _pair_result_check        [aten.max.dim]
  ...
```

`blind` 는 **의도된 무시**입니다. 거짓 `COMPARATOR BUG` 를 없애기 위해
`BLIND_BY_DESIGN` 표를 두고, 각 항목의 근거를 `cases.py` 의 그 비교기 자신의
docstring 에서 인용해 적었습니다.

또 `Outcome.fault_applied` 를 추가했습니다. **"주입했는데 안 잡혔다" 와 "애초에 주입이
안 됐다" 를 구분하지 못하면 자가검사는 자기 자신에 대해 거짓말을 합니다.** 단일 텐서
결과에는 "마지막 조각" 이 없고, 균등 분할에는 패딩할 자리가 없습니다. 그런 조합은
`n/a` 로 보고하지, 조용히 통과하지 않습니다.

### 3.2 결함 모드를 "그럴듯한 오구현" 모양으로 만들었다

값을 1 더하는 주입은 통과하는 자가검사를 만들 뿐입니다. 비교기마다 "틀린 답" 의 모양이
다르므로 모드를 그 모양에 맞춰 열한 개로 늘렸습니다.

| 모드 | 무슨 오구현을 흉내내나 |
|---|---|
| `value` / `value-last` | 틀린 수. `-last` 는 **다중 결과의 마지막 멤버** — 쌍의 `indices`, layer-norm 의 `rstd`, split 의 마지막 조각 |
| `shape` / `shape-last` | 틀린 모양, 같은 두 자리 |
| `dtype` / `dtype-last` | 틀린 dtype. `dtype-last` 가 **`logsumexp` 의 float16 입력 → float32 출력 비대칭**을 실제로 보고 있는지 판정한다 |
| `permute` | **첫 멤버만** 순서를 뒤집는다 → 쌍에서는 순서 결함이 아니라 **짝짓기 결함** (값 i 가 인덱스 j 를 주장한다) |
| `permute-all` | 모든 멤버를 **함께** 뒤집는다 → 진짜 순서 결함. multiset 비교기가 무시해도 되는 유일한 것 |
| `constant` | 모든 원소를 첫 원소로 붕괴 → 망가진 RNG 의 모양 |
| `chunk-count` | 조각 하나 누락 |
| `chunk-pad` | **마지막 조각을 짧게 두지 않고 패딩** — `docs/models/GPT2.md` 가 `split` 의 가장 그럴듯한 오구현으로 지목한 것. 원소 단위 비교로는 절대 안 보인다 |

> **`permute` 와 `permute-all` 이 갈라진 것은 계획이 아니라 측정 결과입니다.**
> 표를 처음 돌렸을 때 `_topk_multiset_check` 가 `permute` 를 **잡았고**, 제가 적어 둔
> "이건 원래 못 잡는다" 표와 어긋났습니다. 틀린 것은 표가 아니라 **모드**였습니다 —
> 값 텐서만 뒤집으면 그건 순서가 아니라 짝짓기가 깨진 것이고, multiset 비교기는 그것을
> 잡는 게 맞습니다. 그래서 `permute-all` 을 만들어 둘을 분리했습니다.
> `--self-test` 가 "의도된 무시라고 적힌 것이 실제로는 잡혔다" 를 **실패로 취급**하지
> 않았다면 이 오류는 그냥 넘어갔을 것입니다.

### 3.3 `--self-test` — 표 전체를 돌리는 게이트

```bash
$PY tools/golden/compare.py --self-test        # exit 0 이어야 정상
```

`--inject-fault` 는 역사적으로 **exit 1 이 정상**입니다(잡힌 결함이 케이스를 실패시키므로).
그래서 그것을 게이트로 쓰면 "모든 비교기가 살아 있다" 를 표현할 수 없습니다.
`--self-test` 는 반대로 **exit 0 이 정상**이고, 다음 중 하나라도 있으면 1 입니다.

- 잡아야 하는 결함을 못 잡음 (`MISSED`)
- 어떤 결함에도 안 걸린 비교기가 있음
- `BLIND_BY_DESIGN` / `KNOWN_GAP` 에 "못 잡는다" 고 적힌 것이 **잡힘** → 표가 낡았다

마지막 항목이 §3.2 의 `permute` 오류를 잡아낸 규칙입니다.

---

## 4. 비교기별 커버리지 표 (측정값)

`--self-test`, 2026-08-24, `aarch64-apple-darwin`, 아티팩트
`cargo-target-harness/release/lib_C.dylib`.

```
comparator              | value  | value-last | shape  | shape-last | dtype  | dtype-last | permute | permute-all | constant | chunk-count | chunk-pad
------------------------+--------+------------+--------+------------+--------+------------+---------+-------------+----------+-------------+----------
<default pipeline>      | CAUGHT | CAUGHT     | CAUGHT | n/a        | CAUGHT | n/a        | CAUGHT  | CAUGHT      | CAUGHT   | n/a         | n/a
_chunk_list_check       | CAUGHT | CAUGHT     | CAUGHT | CAUGHT     | CAUGHT | CAUGHT     | CAUGHT  | CAUGHT      | CAUGHT   | CAUGHT      | CAUGHT
_dtype_shape_only_check | blind  | blind      | CAUGHT | n/a        | CAUGHT | n/a        | n/a     | n/a         | n/a      | n/a         | n/a
_pair_result_check      | CAUGHT | CAUGHT     | CAUGHT | CAUGHT     | CAUGHT | GAP        | CAUGHT  | CAUGHT      | CAUGHT   | n/a         | n/a
_range_check            | CAUGHT | CAUGHT     | CAUGHT | n/a        | CAUGHT | n/a        | blind   | blind       | blind    | n/a         | n/a
_rng_stream_check       | CAUGHT | CAUGHT     | CAUGHT | n/a        | CAUGHT | n/a        | CAUGHT  | CAUGHT      | CAUGHT   | n/a         | n/a
_scalar_match_check     | CAUGHT | CAUGHT     | n/a    | n/a        | n/a    | n/a        | n/a     | n/a         | n/a      | n/a         | n/a
_sdpa_pair_check        | CAUGHT | CAUGHT     | CAUGHT | CAUGHT     | CAUGHT | CAUGHT     | CAUGHT  | CAUGHT      | CAUGHT   | n/a         | n/a
_topk_multiset_check    | CAUGHT | CAUGHT     | CAUGHT | GAP        | CAUGHT | GAP        | CAUGHT  | blind       | CAUGHT   | n/a         | n/a
_triple_result_check    | CAUGHT | CAUGHT     | CAUGHT | CAUGHT     | CAUGHT | CAUGHT     | CAUGHT  | CAUGHT      | CAUGHT   | n/a         | n/a
```

- `CAUGHT` — 그 결함을 거부했다
- `blind` — **의도된 무시**. 근거는 §5
- `GAP` — **실제 결함**. 잡아야 하는데 못 잡는다. §6. **정정 (문서 감사, 2026-09): 위 표의 세
  `GAP` 칸(`_pair_result_check`+`dtype-last`, `_topk_multiset_check`+`shape-last`/`dtype-last`)은
  전부 닫혔다 — §6 의 정정 참조. `compare.py` 의 `KNOWN_GAP` 은 오늘 빈 딕셔너리다**
- `n/a` — 그 결과 모양에 그 결함을 만들 수 없다 (단일 텐서에는 "마지막 멤버" 가 없다 등)

**어떤 결함에도 안 걸린 비교기는 없습니다.** 열 개 전부 최소 두 개 이상의 주입을
거부했습니다. 가장 적은 둘은 `_dtype_shape_only_check`(shape, dtype — 값을 안 보는 것이
설계다)와 `_scalar_match_check`(value, value-last — dtype 도 shape 도 없는 파이썬
스칼라라서 그 둘 말고는 만들 결함이 없다)입니다.

---

## 5. `blind` 항목 — 무시가 맞는 것들

| 비교기 + 모드 | 근거 |
|---|---|
| `_dtype_shape_only_check` + `value`, `value-last` | `aten.empty` 는 초기화되지 않은 메모리를 준다. **맞는 값이 존재하지 않는다** |
| `_range_check` + `permute`, `permute-all` | 두 RNG 는 같은 시드로도 같은 수열을 못 낸다. 수열은 **의도적으로** 안 본다 |
| `_range_check` + `constant` | `[lo, hi)` 소속만 보고 **분포는 전혀 안 본다.** 범위 안 상수를 돌려주는 셰임은 통과한다 |
| `_topk_multiset_check` + `permute-all` | 동점·`sorted=False` 에서 상류의 순서는 partition 부산물이지 약속이 아니다 |

**`_range_check` + `constant` 는 blind 로 적었지만 읽는 사람이 알아야 할 한계입니다.**
`randint` 가 항상 `lo` 를 돌려줘도 이 하니스는 통과합니다. `cases.py` 가 그 한계를
스스로 적어 두었고, 시드를 맞출 수 없다는 §2 의 논거가 그 한계의 이유입니다. 다만
`normal_`/`uniform_` 은 다릅니다 — 그쪽은 `_rng_stream_check(bitwise=True)` 로 **비트
단위 일치**를 요구하고, `constant` 주입을 잡습니다. 즉 분포를 안 보는 것은
`randint` **하나**입니다.

---

## 6. 발견 — 비교기 세 곳이 `indices` 를 덜 본다

표에서 `GAP` 인 세 칸은 의도가 아니라 **결함**입니다. 셋 다 같은 뿌리입니다:
`(values, indices)` 쌍에서 **`indices` 의 dtype/shape 을 안 본다.**

| 비교기 + 모드 | 무엇을 놓치나 |
|---|---|
| `_pair_result_check` + `dtype-last` | `indices` 의 **dtype 을 비교하지 않는다.** shape 과 평탄화된 값만 본다. 상류는 `int64` 를 주는데, 값이 같은 `int32` 를 돌려주는 셰임이 통과한다 |
| `_topk_multiset_check` + `shape-last` | `indices` 의 **shape 을 비교하지 않는다.** 값 쪽 shape 과 `(값, 인덱스)` multiset 만 보는데, multiset 은 인덱스 텐서의 reshape 을 견딘다 |
| `_topk_multiset_check` + `dtype-last` | 위와 같은 dtype 구멍 |

세 결함 다 **`tools/golden/cases.py` 안에 있고, 이번 작업의 파일 범위 밖**입니다.
고치지 않았습니다. 필요한 수정은 각각 한 줄입니다.

> **정정 (문서 감사, 2026-09):** 전부 닫혔다 — `tools/golden/cases.py` 에 `indices dtype
> mismatch`/`indices shape mismatch` 검사가 오늘 두 자리(6287·6295, 9575·9578행)에 있고, 그
> 텍스트가 이 절이 제안한 한 줄과 그대로 일치한다. `tools/golden/compare.py` 의 `KNOWN_GAP` 은
> 오늘 빈 딕셔너리다 — 아래 §6 의 요구대로 고친 뒤 지워졌다. `tools/golden/`/`rust/` 는 이
> 라운드의 금지 영역이라 직접 고치지 않았지만, 이미 다른 작업이 고쳐 두었다.
> <!-- DOCWATCH: symbol-in-file tools/golden/cases.py "indices dtype mismatch" present -->

```python
# _pair_result_check (cases.py:2108 근처, indices shape 비교 옆)
t_idx_dtype, c_idx_dtype = dt.dtype_name(t_indices.dtype), dt.dtype_name(c_indices.dtype)
if t_idx_dtype != c_idx_dtype:
    return False, f"indices dtype mismatch: torch={t_idx_dtype} c={c_idx_dtype}"

# _topk_multiset_check (cases.py:3540 근처, multiset 비교 앞)
#   위와 같은 dtype 비교 + indices shape 비교를 추가
```

**고친 뒤에는 `compare.py` 의 `KNOWN_GAP` 에서 해당 항목을 지워야 합니다.**
안 지우면 `--self-test` 가 "못 잡는다고 적힌 것이 잡혔다 → 표가 낡았다" 로 **실패**합니다.
방치되지 않게 일부러 그렇게 만들었습니다.

`indices` dtype 이 실제로 중요한 이유: 상류의 `max.dim`/`sort`/`topk` 는 `int64` 인덱스를
약속하고, 그 인덱스는 `index_select`·`gather`·`embedding` 으로 그대로 들어갑니다.
값만 같고 dtype 이 다른 인덱스는 **하니스가 통과시킨 뒤 하류에서 터지는** 종류의 divergence 이고,
`DESIGN.md` §5 가 이 하니스를 만든 이유가 정확히 그것입니다.

---

## 7. 허용오차 — 절대냐 상대냐에 대한 의견

`docs/models/GPT2.md` §3.3 이 `Linear(512,512)` 에서 절대오차 `1.526e-05` 를 보고했고, 그것이
`float32` 허용오차 `1e-5` 를 넘습니다. **케이스는 제가 만들지 않았습니다**(다른 에이전트
소유). 대신 지금 체계가 실제로 무엇을 하고 있는지 **쟀습니다.**

### 7.1 먼저 잰 것 — 지금 허용오차는 거의 아무 일도 안 하고 있다

`math.isclose(x, y, rel_tol, abs_tol)` 는 **선언(OR)** 입니다:
`|x-y| <= max(rtol*max(|x|,|y|), atol)`. 즉 고정 `atol` 은 모든 원소 밑에 깔린 바닥입니다.
스위트 전체의 float 원소 비교 **29,663 건**을 무엇이 판정했는지 분류했습니다.

```
exact      29654   99.97%      두 쪽이 비트 단위로 같다
both           9    0.03%      다르지만 atol 로도 rtol 로도 통과
rel-only       0
abs-only       0
FAIL           0
```

| dtype | n | exact | both |
|---|---:|---:|---:|
| `float64` | 7438 | 7433 | 5 |
| `float32` | 7717 | 7716 | 1 |
| `float16` | 7339 | 7338 | 1 |
| `bfloat16` | 7169 | 7167 | 2 |

**`atol` 때문에만 통과한 비교는 0 건입니다.** 스위트를 초록으로 만들고 있는 것은
허용오차가 아니라 **비트 일치**입니다. 지금의 `TOLERANCES` 표는 사실상 놀고 있습니다.

여유도 마찬가지입니다. 스케일 정규화 오차(`|diff| / max(1, max|torch|)`) 최대값:

| 케이스 | 정규화 오차 | 그 dtype 예산 | 소진율 |
|---|---:|---:|---:|
| `_softmax(bfloat16, dim=0)` | `1.95e-03` | `6e-2` | 3.3% |
| `_softmax(float16, dim=0)` | `6.10e-05` | `5e-3` | 1.2% |
| `cos(float32)` | `5.96e-08` | `1e-5` | 0.6% |

**즉 지금 허용오차를 어떻게 바꿔도 1781 의 판정은 안 바뀝니다.** 그래서 안전하고,
동시에 급하지 않습니다.

### 7.2 의견 — 절대도 상대도 아니고 **텐서 스케일 기준**이 맞다

`k=512` 숫자를 뜯어보면 셋 다 확인됩니다.

```
절대오차   1.526e-05      원소 크기 4.967e-03 에서    →  float32 atol 1e-5 초과, 빨강
상대오차   1.650e-04      같은 원소                    →  float32 rtol 1e-5 초과, 빨강
정규화     3.51e-07       텐서 최대 크기 43.46 기준    →  예산 1e-5 의 3.5%, 초록
```

- **원소별 상대오차를 기준으로 삼으면 안 됩니다.** 상쇄가 일어난 원소는 결과가 작아지지만
  오차는 **누적 중간값의 크기**가 정합니다. `4.967e-03` 짜리 원소에 `1.65e-04` 의
  상대오차는 부동소수점이 정상 동작한 결과이지 divergence 가 아닙니다.
- **고정 절대오차도 기준이 아닙니다.** 데이터 크기와 함께 커져야 하는데 안 커집니다.
  크기 ~1 텐서에는 맞고, 크기 43 텐서에는 좁고, 크기 `1e-6` 텐서에는 **터무니없이 넓습니다**
  (지금은 원소가 전부 `1e-6` 이어도 `1e-5` 바닥이 깔려 전부 무조건 통과합니다).

권장: **`abs_tol` 을 상수가 아니라 텐서 스케일에 비례시킨다.**

```python
# dtypes.py 의 atol 은 그대로 두고, compare.py 의 _values_close 에서
scale = max(abs(v) for v in flattened torch values if finite)   # 유한값의 최대 크기
ok = math.isclose(xf, yf, rel_tol=rtol, abs_tol=atol * scale)
```

- `k=512` 케이스: 예산 `1e-5 * 43.46 = 4.35e-04`, 실측 `1.53e-05` → **28배 여유로 통과.**
  새 숫자를 발명하지 않고 통과합니다.
- 지금 1781 은 그대로 통과합니다 — 위 측정에서 `atol` 로만 통과한 비교가 **0 건**이고,
  비-정확 9 건은 전부 `rtol` 로도 통과하기 때문입니다(`both`). 증명이지 추측이 아닙니다.
- 작은 크기 텐서에서는 지금보다 **좁아집니다.** 그게 개선입니다.

**`k` (축소 차원)를 계수로 넣는 것은 반대합니다.** 오차 한계가 `sqrt(k)` 로 커지는 것은
맞지만 (1) 비교기는 결과만 볼 뿐 op 의 축소 차원을 모르므로 케이스마다 자기 `k` 를
선언해야 하고, 그건 **빨간 케이스를 초록으로 만들려고 조용히 넓히는 자리**가 됩니다.
(2) 스케일 정규화만으로 `k=512` 에 28배 여유가 있습니다. `k=2304`(GPT-2 small) 에서
넘친다면 그건 **정보**이고, 미리 넓혀 두면 그 정보를 못 받습니다.

### 7.3 그래서 이번에 안 바꿨다

`dtypes.py` 를 고치지 않았습니다. 이유:

1. 지금 1781 중 허용오차가 판정하는 케이스가 **하나도 없어서** 바꿔도 아무것도 증명되지
   않습니다. 검증할 수 없는 변경입니다.
2. `k=512` 급 케이스를 **다른 에이전트가 추가하는 중**입니다. 그 케이스가 들어온 뒤에
   바꿔야 "이 변경이 이 케이스를 옳은 이유로 통과시킨다" 를 보일 수 있습니다.
3. 위 §7.2 의 변경은 다섯 줄이고, 근거 숫자는 전부 이 문서에 있습니다. 큰 케이스가
   들어오는 순간 한 번에 적용하면 됩니다.

---

## 8. 모르는 것 / 안 한 것

- **`_dtype_shape_only_check` 의 `permute`·`constant`·`permute-all` 이 `n/a` 인 것은
  재서 확인했습니다.** `_C` 의 `empty` 는 이 호스트에서 **원소가 전부 같은 값**으로
  나옵니다(14 케이스 전부 `distinct=1`). 그래서 뒤집어도 같고(palindrome) 이미 상수라
  두 결함 다 구성이 안 됩니다. **다만 그것은 이 호스트 할당자가 0 페이지를 주는 관측이지
  약속이 아닙니다** — 다른 플랫폼에서 쓰레기 값이 나오면 이 칸은 `n/a` 에서 `blind` 로
  바뀝니다. 어느 쪽이든 `value`/`value-last` 가 같은 성질을 이미 `blind` 로 덮으므로
  커버리지 결론은 안 바뀝니다.
- **`chunk-count`/`chunk-pad` 는 `_chunk_list_check` 하나에만 적용됩니다.** 리스트를
  돌려주는 op 이 `split` 뿐이라 그렇습니다. 다른 리스트 반환 op 이 생기면 자동으로 덮입니다.
- **`_scalar_match_check` 의 타입 변경은 별도 주입 모드로 만들지 않았지만, 비교기를 직접
  불러 확인은 했습니다.** `f(3, 3.0)` → `type mismatch: torch=int(3) c=float(3.0)`,
  `f(True, 1)` → `type mismatch: torch=bool(True) c=int(1)`. 즉 값이 같아도 타입이 다르면
  거부합니다. 주입 모드로 만들지 않은 이유는 `value` 가 이미 이 비교기를 덮기 때문입니다.
- **비교기들이 "맞는 답을 틀렸다고 하지 않는가"(false positive)** 는 이 표가 다루지
  않습니다. 정상 실행 1781/1781 이 그 방향의 증거이긴 하지만, 결함 주입은 한 방향만
  검사합니다.
- `--self-test` 는 비교기당 최대 `--self-test-scan`(기본 120)개 케이스까지만 훑고
  결함을 구성할 수 없으면 `n/a` 로 적습니다. 지금은 전부 훨씬 앞에서 구성됐지만,
  케이스 순서가 크게 바뀌면 `n/a` 가 늘 수 있습니다. **`n/a` 가 늘면 커버리지가 준
  것이므로 그냥 넘기면 안 됩니다.**

---

## 9. 재현

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-harness
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
PY=/Volumes/macMini/caches/spike-venv/bin/python

sh vendor/vendor_torch.sh                              # 새 worktree 라면 먼저
(cd rust/torch_c && cargo build --release)

$PY tools/golden/compare.py            > /tmp/g.log 2>&1; echo "EXIT=$?"   # 0
$PY tools/golden/compare.py --self-test > /tmp/s.log 2>&1; echo "EXIT=$?"  # 0
for m in value value-last shape shape-last dtype dtype-last \
         permute permute-all constant chunk-count chunk-pad; do
    $PY tools/golden/compare.py --inject-fault $m > /tmp/fi-$m.log 2>&1
    echo "$m EXIT=$?"                                                      # 전부 1
done
```

`compare.py`/`verify_schemas.py` 는 `PYTHONPATH=$PWD/vendor` **없이** 돌립니다.
파이프로 종료 코드를 읽지 마십시오 — 파일로 리다이렉트한 뒤 `$?` 를 읽습니다.
