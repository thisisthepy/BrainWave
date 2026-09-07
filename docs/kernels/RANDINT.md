# RANDINT — 정수 뽑기가 조용히 갈라진 곳

`docs/architectures/DEMAND3.md` 마지막 절이 남긴 한 줄짜리 결함을 닫는 문서입니다. 같은 `manual_seed` 에서
`randn` 과 `rand` 는 upstream 과 **비트 단위로 같은데** `randint` 만 다른 값을 냈고, **에러도
경고도 없었습니다.**

```
manual_seed(1234)      upstream                        shim (수정 전)
randn(4)               [0.04613, 0.402403, …]          동일
rand(4)                [0.028979, 0.401899, …]         동일
randint(0,100,(6,))    [75, 71, 6, 65, 16, 64]         [96, 17, 58, 14, 11, 20]
randperm(6)            [3, 2, 4, 5, 1, 0]              미구현
```

`randn`/`rand` 가 맞는다는 것은 **생성기 자체는 옳다**는 뜻입니다(`docs/numerics/RNG.md` 가 옮긴
MT19937). 갈라진 곳은 `randint` 가 그 생성기를 **소비하는 방식**이었습니다 — 수정 전 커널은
생성기를 아예 쓰지 않고 candle 의 시드 불가능한 `Tensor::rand` 로 `[0,1)` 균일난수를 뽑아
아핀 변환하고 `floor` 했습니다(`aten.rs`, 옛 `randint()`). 값도 다르고 스트림도 전혀 움직이지
않으므로, `randint` 뒤에 오는 모든 뽑기까지 어긋납니다.

이 문서는 **코드를 고치기 전에** upstream 의 알고리즘을 실측으로 확정한 기록입니다. 측정
스크립트는 `/Volumes/macMini/caches/randint-probe/` 에 있습니다(커밋 대상 아님). 환경은
`/Volumes/macMini/caches/spike-venv/bin/python` (torch 2.13.0, CPython 3.13.0,
Apple Silicon / darwin 25.5.0).

---

## 1. 소비량 — 원소당 한 번, 그리고 **폭이 그 한 번의 크기를 바꾼다**

측정 방법: `manual_seed(1234)` → 대상 호출 → `torch.rand(3)`. `rand` 는 float32 이므로
MT 워드 하나를 그대로 `uniform_real<float>` 로 바꾼 값입니다. 시드에서 재생성한 워드
스트림에서 그 세 값이 나타나는 위치가 곧 **직전 호출이 소비한 32비트 워드 수**입니다.

| 호출 | 소비한 32비트 워드 |
|---|---|
| `randint(0, 100, (0,))` | 0 |
| `randint(0, 100, (1,))` | 1 |
| `randint(0, 100, (6,))` | 6 |
| `randint(0, 100, (20,))` | 20 |
| `randint(0, 100, (3,4))` | 12 |
| `randint(0, 2**40, (6,))` | **12** |
| `randperm(1)` | 0 |
| `randperm(6)` | 5 |
| `randperm(20)` | 19 |

**원소당 정확히 한 번의 뽑기**입니다. 거부 샘플링이 아닙니다 — 소비량이 데이터에 의존하지
않고, 같은 `n` 이면 시드·범위와 무관하게 항상 같습니다. 재시도가 있었다면 여기서 숫자가
흔들렸을 것입니다.

**그 한 번이 32비트냐 64비트냐를 폭이 결정합니다.** 경계를 직접 찍었습니다
(`n=6`, `low=0`):

| 폭(`high - low`) | 소비 워드 |
|---|---|
| 2²⁷ | 6 |
| 2²⁸ − 2 | 6 |
| 2²⁸ − 1 | 6 |
| **2²⁸** | **12** |
| 2²⁸ + 1 | 12 |
| 2²⁹ | 12 |
| 2³² − 1 | 12 |
| 2³² | 12 |

경계는 **폭 ≥ 2²⁸** 이고, 결정하는 것은 `low` 도 dtype 도 아니라 **폭 하나뿐**입니다:

```
[5, 5+2^28-1)   폭 2^28-1   →  6 워드
[5, 5+2^28)     폭 2^28     → 12 워드
[-2^27, 2^27)   폭 2^28     → 12 워드
```

### 1.1 소스를 그대로 읽으면 틀린다 — `#ifdef FBCODE_CAFFE2`

`ATen/core/DistributionsHelper.h:44-57` 는 분기가 둘입니다.

```cpp
#ifdef FBCODE_CAFFE2
    if ((
      std::is_same_v<T, int64_t> ||
      std::is_same_v<T, double> ||
      std::is_same_v<T, float> ||
      std::is_same_v<T, at::BFloat16>) && range_ >= 1ULL << 32)
#else
    if (range_ >= 1ULL << 28) // allow approx 5% skew in uniform int generation using %
#endif
    {
      return transformation::uniform_int_from_to<T>(generator->random64(), range_, base_);
    } else {
      return transformation::uniform_int_from_to<T>(generator->random(), range_, base_);
    }
```

**공개 휠은 `#else` 쪽입니다** — 임계값 **2²⁸**, dtype 무관. 위쪽(FBCODE) 조건을 그대로
옮기면 임계값이 2³² 이 되고 dtype 목록까지 붙어서, `[0, 2**28)` 부터 `[0, 2**32)` 사이의
모든 폭에서 **값과 스트림 위치가 동시에 어긋납니다.** 실측이 없었다면 이 함정을 그대로
밟았을 것입니다.

dtype 무관이라는 것도 실측으로 확인했습니다 — `int16`/`uint8`/`bool` 처럼 2²⁸ 근처의
폭을 애초에 담을 수 없는 dtype 은 경계 검사에 먼저 걸리므로 관측 자체가 안 되고, 담을 수
있는 `int32`·`int64`·`float32`·`float64`·`bfloat16` 은 전부 같은 자리에서 갈립니다.

## 2. 변환 — **모듈로**, 스케일 곱도 거부 샘플링도 아니다

`ATen/core/TransformationHelper.h:42-44`:

```cpp
template <typename T, typename V>
C10_HOST_DEVICE inline T uniform_int_from_to(V val, uint64_t range, int64_t base) {
  return static_cast<T>(static_cast<int64_t>((val % range) + base));
}
```

- `range = static_cast<uint64_t>(to) - static_cast<uint64_t>(from)`
  (`ATen/native/DistributionTemplates.h`, `random_from_to_impl`)
- `(val % range) + base` 는 **uint64 산술**이고, 그 뒤 `static_cast<int64_t>` 로 감쌉니다.
  `base` 가 음수여도 랩어라운드가 정확히 원하는 값을 냅니다.
- 주석이 스스로 밝히듯(`allow approx 5% skew`) 이것은 **편향된** 모듈로입니다. 균일하게
  만들려고 재시도하지 않습니다 — 그래서 소비량이 고정입니다.

**`low` 는 오프셋 그 이상이 아닙니다.** `[low, high)` 의 값은 `[0, high-low)` 의 값에
`low` 를 더한 것과 정확히 같고(실측), 소비량도 같습니다.

## 3. dtype 이 바꾸는 것 — 캐스트와 경계, 소비량이 아니다

`random_from_to_kernel` 은 dtype 별로 인스턴스화되지만, §1 에서 본 대로 **뽑기의 폭은
dtype 이 아니라 범위 폭이 정합니다.** dtype 이 바꾸는 것은 두 가지입니다.

### 3.1 마지막 캐스트

`static_cast<T>(static_cast<int64_t>(...))`. 정수 dtype 이면 C 의 랩어라운드 변환이고
(경계 검사를 통과했다면 값이 이미 범위 안이므로 무해), **부동소수 dtype 이면 반올림**입니다.
`float32` 에 `[0, 2**33)` 을 요구하면 값이 512 의 배수로 떨어지는 것이 그 결과입니다.

### 3.2 부동소수 dtype 은 **경계 자체가 이동한다** (`update_from` / `update_to`)

`ATen/native/DistributionTemplates.h:42-77`. `to - 1` 을 dtype 으로 캐스트했을 때
`to` 이상으로 반올림되어 버리면, 그 dtype 에서 표현 가능한 **바로 아래 값**으로 `to` 를
끌어내립니다:

```cpp
const auto to_minus_1 = static_cast<int64_t>(static_cast<scalar_t>(to - 1));
if (to_minus_1 >= to) {
  int64_t to_ = std::abs(to - 1);
  int n = 0;
  while (to_ >>= 1) ++n;
  to = to_minus_1 - (1LL << (n - std::numeric_limits<scalar_t>::digits + 1));
}
```

`from` 쪽에 대칭인 `update_from` 이 있습니다. **이 보정이 `range` 를 바꾸므로 §1 의 임계값
판정과 §2 의 모듈로 둘 다 보정된 값으로 해야 합니다.** 예: `float32`(digits=24) 에서
`randint(0, 2**33)` 은 `to` 가 `2**33 - 512` 로 내려가고, 폭은 `8589934080` 이 됩니다.

`digits` 는 `float32`=24, `float64`=53, `float16`=11, `bfloat16`=8 입니다.

### 3.3 경계 검사 — 순서까지 관측 가능하다

`check_from_to_in_range` (같은 파일). 순서가 중요합니다:

1. `TORCH_CHECK(from < to)` — 원래 값으로
   → `random_ expects 'from' to be less than 'to', but got from=5 >= to=5`
2. 부동소수면 `update_from`/`update_to`, 그리고 다시 `from < to`
   → `random_ expects 'from' casted to dtype to be less than 'to' casted to dtype, …`
3. `check_from_to_in_range(from, to - 1, dtype)` — dtype 의 표현 범위 밖이면 거부
   → `to - 1 is out of bounds for int` / `from is out of bounds for int`
4. **그다음에야** `CHECK_EMPTY_AND_RETURN` — 그래서
   `randint(0, 2**32, (0,), dtype=torch.int32)` 는 **빈 텐서가 아니라 에러**입니다(실측).

3 번의 dtype 이름은 `caffe2::TypeMeta` 의 C++ 이름입니다(실측):

| dtype | 메시지에 찍히는 이름 |
|---|---|
| `int64` | `long` (도달 불가 — int64 범위를 벗어나는 경계가 없다) |
| `int32` | `int` |
| `int16` | `short` |
| `int8` | `signed char` |
| `uint8` | `unsigned char` |
| `uint16` | `unsigned short` |
| `uint32` | `unsigned int` |
| `bool` | `bool` |
| `float16` | `c10::Half` |
| `float32`/`float64`/`bfloat16` | 도달 불가 — 최대값이 int64 범위를 덮는다 |

부동소수에는 거부가 아닌 **경고**도 있습니다 — `|from|` 또는 `|to-1|` 이 `2^digits` 를
넘으면 `UserWarning: to - 1 is out of bounds [-(2^24), 2^24]. Due to precision limitations
float can support discrete uniform distribution only within this range. …`. 값에는 영향이
없고(§3.2 의 보정이 이미 한 일을 설명할 뿐입니다) **이 구현은 이 경고를 내지 않습니다** —
아래 §7 에 미구현으로 명시합니다.

## 4. `randperm` 은 같은 기계에서 나온다

`randperm` 은 별도의 알고리즘이 아니라 **같은 MT19937 의 `random()` 을 쓰는 Fisher–Yates**
입니다. `n-1` 번의 32비트 뽑기(§1 의 표)가 그 증거이고, 값까지 재현했습니다:

```
r = [0, 1, …, n-1]
for i in 0 .. n-2:
    z = generator->random() % (n - i)
    swap(r[i], r[z + i])
```

`n` 에 따른 분기(예전 코드가 가지고 있던 `n < 30000` 임계값)는 **없습니다** — `n` = 0, 1, 2,
6, 17, 20, 100, 1000, 29999, 30000, 30001, 50000 을 시드 4 개로 전부 재현했고
(**48/48 일치**), 소비량도 언제나 `max(n-1, 0)` 이었습니다.

따라서 `randperm` 은 `randint` 를 고치는 것과 같은 기계에서 떨어져 나오므로 **이 라운드에서
함께 구현합니다.** 이름을 대며 거부하는 쪽으로 남기지 않았습니다.

dtype 은 `int64`(기본) 외에 `int32`/`int16`/`uint8`/`float*` 이 모두 같은 순열을 내고
(캐스트만 다름), `bool` 은 upstream 이 `"randperm" not implemented for 'Bool'` 로 거부합니다.
`n` 이 dtype 의 정밀도를 넘으면 `n cannot be greater than 2049 for Half type.` 처럼 거부합니다.

## 5. 모델과 upstream 의 대조 — 값 기준

§1~§4 를 순수 파이썬으로 옮겨(`/Volumes/macMini/caches/randint-probe/model.py`) upstream 과
직접 대조했습니다. **코드를 고치기 전에** 돌린 것입니다.

    시드      0, 1234, 42, 7
    dtype     int64, int32, int16, int8, uint8, bool, float32, float64, float16, bfloat16
    범위      [0,100) [0,2) [-5,5) [3,4) [0,128) [0,2^8) [0,2^11) [0,2^24)
              [0,2^28-1) [0,2^28) [0,2^28+1) [0,2^31) [0,2^32) [0,2^32+1)
              [0,2^40) [0,2^53) [-2^30,2^30) [1e9,1e9+7) [-2^40,2^40)
    n         1, 6, 17

**2280 조합 중 값이 어긋난 것은 0 개입니다.** 남은 756 개는 전부 upstream 이 §3.3 의 경계
검사로 거부한 조합인데 모델에 그 검사를 넣지 않아 난 차이이고, 값 불일치가 아닙니다 —
그 목록이 §3.3 의 표를 만들었습니다.

`randperm` 은 위의 12 개 `n` × 시드 4 개 = **48/48 일치**입니다.

## 6. 스트림 위치 — 값만 맞추면 놓치는 것

값이 맞는데 소비량이 틀린 구현은 **값만 보는 테스트를 통과합니다.** 그리고 그 다음 뽑기에서
무너집니다. 그래서 기준선을 값이 아니라 **끼워넣기(interleaving)** 로도 박아 둡니다
(`manual_seed(1234)`, 실측):

```
randint(0,100,(6,))    [75, 71, 6, 65, 16, 64]
randn(4)               [-0.85447514, 0.50984222, -0.08205455, 0.66073167]

randint(0,2**40,(3,))  [204000912083, 166721346613, 878802328948]
randn(4)               [-0.85447514, 0.50984222, -0.08205455, 0.66073167]      ← 같다
```

두 줄이 같은 `randn` 을 내는 것이 §1 의 표를 값으로 다시 말한 것입니다 — 폭이 작은 6 원소
(6 워드)와 폭이 큰 3 원소(3×2 = 6 워드)가 **같은 자리에서 끝납니다.** 폭 임계값을 2³² 로
잘못 옮긴 구현은 첫 줄은 맞히고 둘째 줄에서 갈라집니다.

```
randn(3)               [0.04613046, 0.40240282, -1.01152909]
randint(0,100,(4,))    [75, 29, 43, 41]
rand(2)                [0.77485049, 0.82080257]

randperm(6)            [3, 2, 4, 5, 1, 0]
randn(4)               [-1.30546951, -1.01470625, -0.68631357, -0.96611220]
```

## 7. 이 구현이 하지 않는 것 — 명시

- **`UserWarning` 을 내지 않습니다.** §3.3 의 정밀도 경고(`to - 1 is out of bounds
  [-(2^24), 2^24] …`)는 upstream 이 내고 이 구현은 내지 않습니다. **값은 같습니다** —
  경고가 설명하는 보정(§3.2)은 구현되어 있고 대조로 확인했습니다. 이 저장소의 rust 소스에는
  파이썬 경고를 내는 선례가 아직 없어서, 선례를 만드는 것을 이 라운드의 범위 밖으로 두었습니다.
- **`generator=` 오버로드는 열지 않았습니다.** `aten::randint.generator`,
  `aten::randint.low_generator`, `aten::randperm.generator` 는 `overloads.json` 에는
  있지만 커널이 없습니다. 이 shim 에는 프로세스당 생성기가 하나뿐이므로
  (`rng.rs::default_generator`) 별도 `torch.Generator` 는 `uniform_`/`normal_` 이 이미
  하는 것과 같은 이유로 거부됩니다.
- **`.out` 오버로드는 열지 않았습니다.** `aten::randint.out` 계열은 `out=` 텐서를
  리사이즈해야 하고 `docs/kernels/RANDOM.md` §3 이 `randn`/`rand` 에 대해 같은 이유로 이미
  거부하고 있습니다.
- **AVX2/x86.** `random_from_to_kernel` 은 `cpu_serial_kernel` 이라 SIMD 특수화가 없고,
  `normal_` 과 달리 플랫폼 의존이 없습니다 — 다만 이 문서의 실측은 전부 aarch64 입니다.

## 8. 무엇이 이 결함에 오염되어 있었나

`randint` 가 upstream 과 다른 값을 냈으므로, **`randint` 로 입력을 만들어 두 쪽을 비교한
모든 것은 서로 다른 입력을 비교하고 있었습니다.** 실제로 무엇이 그랬는지는 §8.1 에,
"그런데 왜 통과했나" 는 §8.2 에 있습니다.

### 8.1 목록

| 자리 | 무엇이었나 | 잘못된 이유로 통과했나 |
|---|---|---|
| `tools/golden/cases.py::randint_low_cases` (13 케이스) | `_range_check(low, high)` — dtype·shape·`[low, high)` 소속만 보고 **수열은 보지 않는다**. 양쪽을 시드로 맞추지도 않았다 | **아니다.** 통과 근거가 문서화되어 있었고(`cases.py` 모듈 주석, `RNG.md` §5 의 표) 그 근거가 당시엔 참이었다 — candle 생성기는 시드를 받지 못한다. 값이 다른 것을 *알고* 비교하지 않은 것이지, 같다고 오판한 것이 아니다. **다만 그 근거는 `rng.rs` 가 들어온 순간 낡았고**(RNG.md §5 표의 `randint` 행이 "포팅 이후 승격"이라고 예고한 그대로), 그때 갱신되지 않았다 |
| `docs/architectures/DEMAND3.md` 의 11-모델 스윕 | 토큰 id 를 `torch.randint` 로 뽑아 양쪽에 먹였다 | **그렇다 — 그리고 그것이 이 결함을 찾아낸 경로다.** `t5` 는 가중치가 비트 단위로 같은데 출력이 단위 단위로 어긋났고, 원인이 하네스가 두 쪽에서 다른 토큰을 뽑은 것이었다. DEMAND3 는 그 뒤 손으로 만든 토큰 리스트로 바꿔서 스윕을 마쳤다(§5) |
| `rust/torch_c/pytests/test_shim.py:6004` | `d("aten.randint.default", 10, [2])` 를 **거부 경로**로만 쓴다(잘못된 인자에 op 이름이 찍히는지) | 아니다 — 값을 보지 않는다 |
| `rust/torch_c/pytests/decomp_sweep.py:47-48` | 이름만 등장 | 아니다 |

**시드를 걸고 정수를 뽑아 값을 비교하던 자리는 이 라운드 이전의 저장소에 없었습니다.**
스위트에서 `manual_seed`/`_shim_manual_seed` 를 부르는 **72 곳**(이 라운드 이전 기준)을 전부
훑었고, 그것들이 뽑는 것은 `uniform_` · `normal_` · `bernoulli_` · `multinomial` · `dropout` ·
`randn`/`rand` 이지 `randint` 가 아닙니다.

그래서 이 결함으로 **빨개졌어야 하는데 초록이던 테스트는 하나도 없습니다.** 위 표의 골든
13 케이스는 잘못된 이유로 통과한 것이 아니라 **애초에 값을 묻지 않았고**, 나머지 셋은 값을
보지 않습니다. 이것이 이 결함의 진짜 모양입니다 — 틀린 단언이 아니라 **없는 단언**이었고,
그래서 `docs/architectures/DEMAND3.md` 의 모델 스윕까지 가서야 드러났습니다. 그 자리를 지금 §9 의
케이스들이 메웁니다.

**따라서 무효화된 측정은 하나입니다** — `docs/architectures/DEMAND3.md` 가 `randint` 로 토큰을 뽑던 첫 회차.
그 문서가 §5 에서 이미 손으로 만든 토큰으로 바꿔 다시 돌렸으므로, **그 문서의 표는 다시
돌릴 필요가 없습니다.** 다시 돌릴 값이 있는 것은 그 문서가 §0 에서 버렸다고 적은 첫 회차뿐이고,
그 회차의 숫자는 어디에도 남아 있지 않습니다.

### 8.1a 고친 뒤 다시 잰 결과

같은 스윕을 **구현된 shim** 으로 다시 돌려 upstream 과 대조했습니다 (`verify.py`, 위와 같은
프로브 디렉터리). 값·에러 메시지·끼워넣기를 전부 한 JSON 에 담아 통째로 비교합니다.

```
2399 / 2399 identical to upstream, 0 differ
```

여기에는 §6 의 끼워넣기 9 개가 모두 포함되고, §3.3 의 거부 메시지 전부가 포함됩니다.

한 번 갈렸다가 고친 것이 하나 있습니다 — **`float16` 의 무한대 경유**. 첫 구현은
`static_cast<Half>` 를 정수 산술로만 옮겨 가수 비트만 반올림했는데, upstream 은 65504 를 넘으면
**inf** 가 되고 그 뒤 `static_cast<int64_t>` 가 포화합니다. 그래서
`randint(10**9, 10**9+7, dtype=float16)` 에서 upstream 은 `from is out of bounds for c10::Half`
를 내고 첫 구현은 `'from' casted to dtype …` 을 냈습니다. 값이 아니라 **어느 거부가 먼저
나오는가** 가 갈린 것이고, 값만 보는 대조로는 안 보였을 자리입니다
(`rng.rs::FloatFormat::max_finite`).

### 8.1b 임계값이 실제로 지켜지는지 — 무력화해서 확인했다

**실패할 수 없는 검증은 검증이 아니므로**, `RANDINT_WIDE_THRESHOLD` 를 `1 << 28` 에서
`1 << 32`(헤더의 FBCODE 쪽 값)로 바꾸고 다시 빌드해서 게이트가 실제로 빨개지는지 확인했습니다:

```
스위트   4 개 실패 — randint 값 대조, 스트림 위치, 끼워넣기 체인, high-only 오버로드
골든     8279/8304, 25 실패
```

그 뒤 되돌리고 다시 초록을 확인했습니다. 즉 §1.1 의 임계값은 주석이 아니라 **테스트가 잡고
있는 것**입니다.

### 8.2 왜 하네스가 못 잡았나 — 규칙이 스스로를 면제했다

`_range_check` 는 세 개뿐인 "값을 비교하지 않는" 비교자 중 하나였고, 나머지 둘(`empty`,
`is_floating_point`)과 같은 칸에 묶여 있었습니다. `empty` 는 **비교할 옳은 값이 없고**,
`randint` 는 **비교할 옳은 값이 있는데 당시 생성기로는 만들 수 없었습니다.** 두 가지가
같은 예외 칸에 들어가 있었기 때문에, 생성기가 바뀌어 두 번째 이유가 사라졌을 때 아무것도
그것을 알려주지 않았습니다.

이번 라운드는 그 칸에서 `randint` 를 꺼내 **시드를 맞춘 값 비교**로 바꿉니다. `empty` 는
그대로 둡니다 — 그쪽 이유는 여전히 참입니다.

`_range_check` 는 삭제됐고, `compare.py` 의 `BLIND_BY_DESIGN` 에 있던 그 세 항목
(`permute` · `permute-all` · `constant`)도 함께 사라졌습니다. **낡아서 지운 것이 아니라 그
맹점이 없어져서 지운 것입니다** — `_rng_stream_check` 는 셋 다 잡습니다(자체 테스트 실측:
`_range_check` 4/11 → `_rng_stream_check` 7/11). 그것이 검사하던 `[lo, hi)` 소속은
`_rng_stream_check(bounds=…)` 로 그대로 남아 있으므로 **잃은 검사는 없습니다.**

## 9. 게이트

```
$ PYTHON=$PY sh rust/torch_c/pytests/run.sh
ok 358   (기준선 348)
SELF-TEST: PASS -- 20 comparators x 11 fault modes, 0 problem(s), 0 comparator(s) never exercised
DOCWATCH: PASS -- 283/283 evaluated marker(s) hold        (기준선 275/275)
EXIT=0

$ $PY tools/golden/compare.py
SUMMARY: 8304/8304 cases passed, 0 failed, ops covered=187, pending case builders=1
                                          (기준선 8126 / 0 / 185)

$ $PY rust/torch_c/pytests/verify_schemas.py
SUMMARY: 4583/4583 table entries matched upstream, 0 failed     (기준선 4574 — `randperm` 4 개 추가)
```

비교자 수가 21 → 20 인 것은 `_range_check` 삭제입니다. 골든 케이스가 8126 → 8304 인 것은
`randint.low` 의 재작성분과 `randint.default`·`randperm.default` 의 새 빌더입니다.

<!-- DOCWATCH: op-implemented aten.randint.low -->
<!-- DOCWATCH: op-implemented aten.randint.default -->
<!-- DOCWATCH: op-implemented aten.randperm.default -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/rng.rs randint_from_to_fill present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/rng.rs randperm_fill present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/rng.rs RANDINT_WIDE_THRESHOLD present -->
<!-- DOCWATCH: symbol-in-file tools/golden/cases.py _rng_stream_check present -->
<!-- DOCWATCH: json-key rust/torch_c/src/overloads.json randperm present -->
