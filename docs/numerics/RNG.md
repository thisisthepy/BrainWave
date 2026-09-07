# RNG — candle 은 torch 의 CPU 난수 스트림을 재현할 수 있는가

`docs/models/FROM_CONFIG.md` §4.3 이 **미확인**으로 남긴 질문에 답하는 문서입니다. 그 문서는
`AutoModelForCausalLM.from_config` 이 요구하는 aten op 14 개 중 최다 호출 상위 둘이 난수
(`aten.normal_.default` 17 회, `aten.uniform_.default` 15 회 — 총 75 회 중 32 회)이고,
`torch.Generator` 가 한 번도 생성되지 않으므로 **그 두 op 을 구현하는 것이 곧 RNG 요구사항**이라고
적었습니다. 남은 질문은 하나였습니다: **candle 의 난수로 그것을 구현하면 torch 와 값이 같은가?**

**판정: 같을 수 없습니다. 그리고 candle 을 고쳐서 같게 만드는 길도 없습니다** — candle 의 CPU
백엔드는 시드를 받는 것 자체를 거부합니다(§2.1). 대신 **torch 의 CPU RNG 를 직접 포팅하는 것은
가능하고, 비용도 작습니다** — 이 문서가 그 포팅을 순수 파이썬으로 먼저 해서 `uniform_` 은
**비트 단위 완전 일치**, `normal_` 은 **최악 3 ulp** 로 재현되는 것을 실측했습니다(§3).

실험 스크립트는 `/Volumes/macMini/caches/rng-probe/` 에 있습니다(커밋 대상 아님). 환경은
`/Volumes/macMini/caches/spike-venv/bin/python` (torch 2.13.0, transformers 5.15.1, CPython 3.13.0,
Apple Silicon / darwin 25.5.0). 이 문서는 `docs/numerics/RNG.md` 외의 어떤 파일도 만들거나 고치지
않았습니다.

> **Correction (문서 감사, 2026-09):** §5 의 권고가 그대로 채택되어 포팅됐습니다. `git log
> -S'"aten.uniform_.default"' -- rust/torch_c/src/aten.rs` 가 찾는 `2d3663f` ("Feat: Port
> torch's CPU generator, and give `_C._dynamo` the two names that do work") 의 커밋 메시지가
> 이 문서를 그대로 요약합니다 — "candle's CPU backend refuses seeding outright: MT19937 with the
> pre-decrement that twists before the first draw, uniform_real at 24 and 53 bits, and the
> kernel shapes that decide which stream a draw comes from". `aten.uniform_.default` /
> `aten.normal_.default` 둘 다 지금 `_aten_implemented()` 에 있습니다 — §5 권고 1·2 번이 실행된
> 결과이자, `docs/bindings/TENSORBASE.md` §7 이 "RNG 에서 멈췄다"고 적은 벽이 사라진 이유이기도 합니다
> (같은 커밋). §3.3 의 비트 단위 재현 주장, §6 의 미확인 목록(AVX2 경로 · `randint` 알고리즘 ·
> bf16/fp16 정확 재현)은 재측정하지 않았습니다 — 이 감사는 "포팅했는가" 만 확인했고 "포팅이
> 이 문서가 예측한 정확도로 재현하는가" 는 확인하지 못했습니다.
> <!-- DOCWATCH: op-implemented aten.uniform_.default -->
> <!-- DOCWATCH: op-implemented aten.normal_.default -->

---

## 1. torch 의 CPU RNG — 실체

### 1.1 엔진은 MT19937 이고, 시딩은 `init_genrand` 다

`torch.manual_seed(s)` 는 `torch.default_generator`(프로세스당 하나인 C 소유 싱글턴,
`at::CPUGeneratorImpl`)의 시드를 바꿉니다. 그 안의 비트 생성기는 Mersenne Twister 이고, 상수와
순서까지 Matsumoto 의 `mt19937ar-cok.c` 원본 그대로입니다. 설치된 torch 헤더에 소스가 그대로
들어 있습니다:

| 사실 | 근거 |
|---|---|
| 상태 크기 624 워드 | `torch/include/ATen/core/MT19937RNGEngine.h:20` (`MERSENNE_STATE_N = 624`), `:108` (`std::array<uint32_t, 624> state_`) |
| 시딩 = Knuth 1812433253 재귀 (`init_genrand`) | 같은 파일 `:160-163` — `state_[j] = 1812433253 * (state_[j-1] ^ (state_[j-1] >> 30)) + j`, 그리고 `left_ = 1` |
| 인출 순서 = `if (--left_ == 0) next_state();` | 같은 파일 `:141` |
| twist 재귀 | 같은 파일 `:177-188` |

**중요한 함정**: `left_` 를 **먼저 감소시키고** 0 인지 본다는 것. 초기 `left_ = 1` 이므로 **첫 번째
인출 이전에 이미 한 번 twist 가 돕니다.** 이 문서를 쓰면서 처음에 검사 순서를 뒤집어 구현했고,
그 결과 나온 값은 numpy 의 `np.random.seed(0)` 스트림과 비슷한 다른 수열이었으며 torch 와 전혀
맞지 않았습니다. 순서를 바로잡자 즉시 일치했습니다.

`torch.default_generator.get_state()` 는 5056 바이트 `uint8` 텐서를 돌려줍니다(실측). 앞
40 바이트를 뜯어보면 `seed=0 / left=1 / seeded=1 / next=0 / state[0]=0 / state[1]=1` 로,
`state[1] = 1812433253*(0^0)+1 = 1` 이라는 위 재귀의 예측과 정확히 맞습니다 — 즉 이 blob 은 레거시
포맷으로 노출된 MT 상태 그 자체입니다.

### 1.2 `uniform_` 은 24/53 비트 마스크 곱

`torch/include/ATen/core/TransformationHelper.h:85-88`:

```cpp
C10_HOST_DEVICE inline dist_acctype<T> uniform_real(V val, T from, T to) {
  constexpr auto MASK = static_cast<V>((static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
  constexpr auto DIVISOR = static_cast<dist_acctype<T>>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
  dist_acctype<T> x = (val & MASK) * DIVISOR;
```

즉 **하위 비트 마스크**이지 상위 비트 시프트(`>> 8`)가 아닙니다. 두 후보를 다 실측해 갈랐습니다:

| dtype | 소비 | 변환 | seed 0/42/1234 실측 |
|---|---|---|---|
| float32 | `random()` = MT 1 워드 | `(v & (2²⁴−1)) · 2⁻²⁴` | **3/3 일치** |
| float32 | — | `(v >> 8) · 2⁻²⁴` | 3/3 **불일치** |
| float64 | `random64()` = MT 2 워드, **hi 먼저** | `(((hi<<32)\|lo) & (2⁵³−1)) · 2⁻⁵³` | **3/3 일치** |

범위 지정판 `uniform_(from, to)` 은 `x * (to - from) + from` 을 **누적 타입에서** 한 번 더 곱합니다.
`torch.empty(1000).uniform_(-0.5, 0.5)` 를 20 개 시드로 20,000 개 뽑아 대조한 결과
**비트 단위로 20,000/20,000 일치, 불일치 0 개**입니다.

`torch.nn.init.kaiming_uniform_` 은 경계를 계산해 `uniform_(-bound, bound)` 를 부르는 파이썬 함수일
뿐이므로 자동으로 따라옵니다 — `kaiming_uniform_(torch.empty(4,64), a=√5)` 256 개 값을 직접 계산한
스트림으로 재현했고 **전부 일치**했습니다.

### 1.3 `normal_` 은 Box–Muller — 그런데 **커널의 블로킹 구조가 출력의 일부다**

여기가 이 조사에서 가장 중요한 발견입니다. `torch/include/ATen/native/cpu/DistributionTemplates.h:223-235`:

```cpp
void normal_kernel(const TensorBase &self, double mean, double std, RNG generator) {
  auto size = self.numel();
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, self.scalar_type(), "normal_kernel_cpu", [&] {
    if (size >= 16 && self.is_contiguous()) {
      normal_fill<scalar_t>(self, mean, std, generator);      // 경로 A
    } else {
      ...
      cpu_serial_kernel(iter, [mean, std, generator]() -> scalar_t {
        at::normal_distribution<double> normal(mean, std);     // 경로 B
        return static_cast<scalar_t>(normal(generator));
      });
```

**같은 시드, 같은 dtype 인데 원소 개수와 연속성에 따라 완전히 다른 수열이 나옵니다.** 실측:

```
torch.manual_seed(0); torch.empty(n).normal_()   (float32)
n= 1..15  →  [ 1.5410, -0.2934, -2.1788,  0.5684, -1.0845, ...]   경로 B
n=16      →  [-1.1258, -1.1524, -0.2506, -0.4339,  0.8487, ...]   경로 A
n=17      →  [-1.1258, -1.6959,  0.5667,  0.7935,  0.5988, ...]   경로 A + 꼬리 재계산
n=20      →  [-1.1258, -1.1524, -0.2506, -0.4339,  0.5988, ...]   경로 A + 꼬리 재계산
n=32      →  [-1.1258, -1.1524, -0.2506, -0.4339,  0.8487, ...]   경로 A, 꼬리 없음
```

비연속 텐서는 크기가 커도 경로 B 로 떨어집니다 — `torch.empty(40)[::2].normal_()`(numel 20,
`is_contiguous() == False`) 은 `n=20` 이 아니라 `n≤15` 와 같은 앞머리(`1.5410, -0.2934, ...`)를
냈습니다. `size >= 16 && is_contiguous()` 조건이 관측 가능한 값을 가른다는 직접 증거입니다.

**경로 B (`size < 16` 또는 비연속)** — `DistributionsHelper.h:173-203` 의 Box–Muller 이고, 캐시된
짝을 한 개 들고 있습니다(`maybe_set_next_normal_sample`). 결정적으로, 커널은 스칼라 타입과 무관하게
**항상 `normal_distribution<double>`** 을 인스턴스화합니다 — 즉 float32·float16 도 **double 균일난수를
소비**(호출당 MT 2 워드)한 뒤 마지막에 캐스팅합니다. 이 때문에 `float32 n=5` 와 `float64 n=5` 의
값이 float 정밀도까지 같습니다(실측). 처음에 float32 는 float 균일난수를 쓸 것으로 가정했다가
전부 불일치했고, double 로 바꾸자 **seed 0/42/7 × n 2/5/15 × dtype f32/f64/f16 전 18 조합 일치**했습니다.

**경로 A (`size >= 16` 이고 연속)** — `DistributionTemplates.h:168-220` 의 `normal_fill`:
출력 버퍼 전체를 먼저 `uniform_real_distribution<opmath_t>` 로 채운 뒤, **16 개 블록 단위로 제자리
Box–Muller** 를 겁니다(`NormalFill16`, `:93-110`) — `data[j]` 와 `data[j+8]` 을 짝지어 cos/sin 을
씁니다. 그리고 `size % 16 != 0` 이면 **마지막 16 개를 새 균일난수로 다시 뽑아 덮어씁니다**:

```cpp
    if (size % 16 != 0) {
      data = data + size - 16;
      for (const auto i : c10::irange(16)) { data[i] = uniform(generator); }
      normal_fill_16(data);
    }
```

n=17 과 n=20 에서 앞쪽 원소가 다시 바뀌는 것이 이 겹침 재계산입니다. 이 구조를 그대로 옮긴 구현으로
seed 0/42 × n 16/17/20/32/40 (float32) 및 n 16/20/33 (float64) **전부 재현**했습니다. float32 는
`opmath_t == float` 이라 버퍼가 곧 출력이고, bf16/fp16 은 16 칸짜리 스택 버퍼를 써서 소비 패턴이
또 다릅니다(`:196-219`).

### 1.4 병렬성은 관여하지 않는다

RNG 커널은 `cpu_serial_kernel` / 뮤텍스 잠금이라 스레드 수와 무관합니다. `torch.empty(100000).uniform_()`
을 포팅 구현으로 **10 만 개 전부 비트 단위 재현**했고, `torch.set_num_threads(1)` 로 바꿔도 같은
값이었습니다(기본 4 스레드). 즉 재현을 방해하는 비결정성은 없습니다.

---

## 2. candle 의 RNG — 실체

소스는 `/Users/ibrew/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/candle-core-0.11.0/`.
torchnative 의 `rust/torch_c/Cargo.lock` 이 `rand 0.9.5`, `rand_distr 0.5.1` 을 고정하고 있습니다
(`Cargo.lock:866-868`, `:895-898`; candle 의 요구는 `Cargo.toml:231-234` 의 `rand 0.9.0` /
`rand_distr 0.5.1`).

### 2.1 CPU 백엔드는 시드를 받는 것 자체를 거부한다

`src/cpu_backend/mod.rs:3096-3102`:

```rust
fn set_seed(&self, _seed: u64) -> Result<()> {
    crate::bail!("cannot seed the CPU rng with set_seed")
}
fn get_current_seed(&self) -> Result<u64> {
    crate::bail!("cannot get the CPU rng seed with get_current_seed")
}
```

`Device::set_seed`(`src/device.rs:278-284`)가 CPU 로 위임하는 곳이 바로 여기입니다. **CUDA 와 Metal
백엔드는 시드를 받지만 CPU 는 런타임 에러를 냅니다.** 우리가 목표로 하는 것은 CPU 경로이므로,
"candle 을 torch 와 같은 시드로 맞춘다"는 선택지는 시작점에서 이미 없습니다.

### 2.2 스트림은 프로세스마다 다른 암호학적 RNG 다

`rand_uniform`(`:3104-3166`)과 `rand_normal`(`:3168-3227`) 은 둘 다 본문 첫 줄이
`let mut rng = rand::rng();` 입니다. `rand::rng()` 는 스레드 로컬 `ThreadRng` 이고:

- 알고리즘은 **ChaCha (12 라운드)** — `rand-0.9.5/src/rngs/thread.rs:59`
- 초기 시드는 **OS 엔트로피**(`OsRng`), 그리고 64 KB 생성마다 **자동 재시드** —
  같은 파일 `:36-39` (`THREAD_RNG_RESEED_THRESHOLD: u64 = 1024 * 64`), `:56`
- 외부에서 시드를 주입할 API 가 없습니다(`ThreadRng::reseed()` 는 OS 에서 다시 뽑을 뿐)

즉 candle CPU 의 난수는 **매 프로세스 실행마다 다르고, 같은 프로세스 안에서도 64 KB 마다 외부
엔트로피가 섞입니다.** 시드로 고정한다는 개념 자체가 성립하지 않습니다.

### 2.3 분포 구현도 다르다

| | torch CPU | candle CPU |
|---|---|---|
| 비트 생성기 | MT19937 (624×32bit, 결정적, 시드 가능) | ChaCha12 (암호학적, OS 시드, 재시드) |
| uniform f32 | `(v & (2²⁴−1)) · 2⁻²⁴` + 아핀 | `rand::distr::Uniform` — 지수 조작으로 [1,2) 를 만든 뒤 `value0_1 * scale + low` (`rand-0.9.5/src/distr/uniform_float.rs:140-153`) |
| normal | **Box–Muller** (경로 B) / 16 칸 블록 Box–Muller (경로 A) | **Ziggurat** — `rand_distr-0.5.1/src/normal.rs:60-100`, `ziggurat_tables::ZIG_NORM_X/F` |
| normal f32 | 경로 B 는 double 로 계산 후 캐스팅 | `rand_distr` 도 f64 로 뽑아 `x as f32` (`normal.rs:51-58`, 주석 `TODO: use optimal 32-bit implementation`) |
| 소비량 | 원소당 1 워드(f32 균일) / 2 워드(f64 균일) — 결정적 | Ziggurat 은 **거부 샘플링**이라 원소당 소비량이 가변 |

Box–Muller 와 Ziggurat 은 같은 분포를 내지만 **같은 수열을 내지 않습니다.** Ziggurat 은 거부
샘플링이라 소비 개수조차 데이터 의존적이어서, 설령 두 엔진의 비트 스트림을 같게 맞춘다 해도
정렬이 어긋납니다.

`rust/torch_c/src/aten.rs:858-868` 의 `randint` 주석은 이 상황을 이미 정직하게 적어 두었습니다 —
"The generator is candle's, not torch's, so the *values* will not match a seeded torch run."
다만 그 주석의 **"torch's Philox stream" 이라는 표현은 CPU 에 대해서는 틀립니다** — Philox 는 CUDA
생성기(`CUDAGeneratorImpl`)이고 CPU 는 §1.1 의 MT19937 입니다. (이 문서는 그 파일을 고치지 않았습니다.
다른 작업이 `rust/torch_c/` 를 동시에 쓰고 있으므로 지적만 남깁니다.)

---

## 3. 재현 가능성과 비용

### 3.1 `rand` 계열에 호환 구현은 없다 — 직접 옮겨야 한다

`rand` 생태계에 MT19937 구현체(`rand_mt` 등)가 존재하기는 하지만, **비트 생성기를 맞추는 것은
문제의 일부일 뿐입니다.** torch 와 값을 맞추려면 §1.2~1.3 의 세 층을 전부 맞춰야 합니다:

1. 엔진 (MT19937 + `init_genrand` 시딩 + `--left` 인출 순서)
2. 변환 (`(v & MASK) · DIVISOR`, f32 는 1 워드 / f64 는 2 워드 hi-first)
3. **커널의 블로킹 구조** — `size >= 16 && is_contiguous()` 분기, 16 칸 Box–Muller, 꼬리 겹침
   재계산, 경로 B 의 `double` 강제, bf16/fp16 의 스택 버퍼 경로

3 번은 어느 RNG 라이브러리도 제공하지 않습니다. 이것은 라이브러리가 아니라 **ATen 커널의 형태**이고,
옮기는 수밖에 없습니다.

**그리고 candle 의 `rand_uniform`/`rand_normal` 은 쓸 수 없습니다** — 시드를 못 받으므로(§2.1)
우회가 아니라 배제입니다. 대신 값을 **직접 `Vec` 에 채워 `Tensor::from_vec` 으로 만드는** 길이
있고, `rust/torch_c` 는 이미 그 패턴을 여러 곳에서 씁니다(`src/aten.rs:385, 390, 520, 538, 837`,
`src/lib.rs:81, 228, 238, 248`). 즉 배선 비용은 새로 드는 것이 아닙니다.

### 3.2 비용은 작다 — 파이썬으로 먼저 해서 재봤다

이 조사는 위 세 층을 **순수 파이썬 약 60 줄**로 포팅해서 실제로 재현되는지 확인했습니다
(`/Volumes/macMini/caches/rng-probe/mt.py` + 프로브 스크립트). 결과:

| 대상 | 결과 |
|---|---|
| `uniform_()` f32 / f64 | seed 0/42/1234 **완전 일치** |
| `uniform_(-0.5, 0.5)` f32, 20 시드 × 1000 개 | **20,000/20,000 비트 단위 일치**, 불일치 0 |
| `uniform_()` f32, n = 100,000 | **100,000 개 전부 비트 단위 일치** |
| `kaiming_uniform_(4×64, a=√5)` | 256 개 **전부 일치** |
| `normal_()` 경로 B (n<16, f32/f64/f16) | seed 0/42/7 × 6 조합 **전부 일치** |
| `normal_()` 경로 A (n≥16, f32/f64) | seed 0/42 × n 16/17/20/32/40 **전부 일치** |
| `normal_(0, 0.02)` f32, 20 시드 × 256 개 | **최악 3 ulp** — 아래 참조 |

Rust 로 옮기면 대략: MT19937 엔진 ~80 줄, 변환 ~15 줄, `normal_` 두 경로 ~70 줄, 생성기 상태 배선
(`manual_seed`/`seed`/`initial_seed`/`get_state`/`set_state`, 5056 바이트 레거시 blob) ~150 줄.
**300 줄 규모의 유한하고 기계적인 작업**이고, `torch.nn.init.*` 14 개는 벤더링된 파이썬 트리
그대로이므로 추가 비용이 없습니다(`FROM_CONFIG.md` §4.2 의 결론과 같음).

### 3.2a 중요한 발견: FMA 축약으로 인한 1 ulp 오차

**clang 이 `x * (to - from) + from` 을 Fused Multiply-Add(FMA) 로 접고, torch 가 그렇게 빌드되어 있습니다.**
결과적으로 **쓰여 있는 C++ 는 컴파일된 C++ 의 1 ulp 틀린 필사본**이 됩니다.

- 사람들이 먼저 시도하는 범위(예: `(0,1)`, `(-1,1)`, `(-0.5,0.5)`)는 전부 **폭이 2 의 거듭제곱**이라
  곱셈이 정확해져서 **이 오차가 보이지 않습니다.**
- `(2.0, 7.5)` 범위에서는 **뽑기의 약 9.5% 가 1 ulp 낮게** 나왔습니다. `uniform_real` 구현을
  `std::fma` 를 명시적으로 쓰는 `mul_add` 로 바꾼 후 이 오차가 사라졌습니다.
- `transformation::normal` 의 `val * std + mean` 도 같은 패턴입니다.

**이것이 `RNG.md` 의 순수 파이썬 이식이 이 오차를 못 본 정확한 이유입니다** — 2 의 거듭제곱 범위만
테스트했기 때문입니다. torch 커널을 이식하는 누구나 이 함정을 마주치므로, 이 문서에서 가장 재사용 가치가
큰 항목입니다.

### 3.3 `normal_` 의 정확도는 libm 과 스칼라 경로 가용성에 걸린다

**예측(이식 전)**: `normal_` 의 최악 오차 3 ulp — libm 이 갈릴 것으로 가정.
**실측(2026-08)**: **비트 완전 일치 (0 ulp)** — 예측을 벗어남.

Rust 의 `f32::ln/cos/sin` 이 aarch64 에서 torch 가 부르는 것과 같은 libSystem 함수로 풀렸고,
`mul_add` 가 상류의 명시적 `std::fma` 와 일치했기 때문입니다. **다만 이것은 스칼라 경로
(`NormalFill16`)이 살아 있는 곳에서만 성립합니다** — AVX2/VSX 특수화(`NormalFill16<float, true>`)는
aarch64 에서 컴파일되지 않으므로 다른 호스트에서의 동작은 미확인입니다.

**두 숫자는 플랫폼이 아니라 이식 방식이 다릅니다.** 둘 다 같은 aarch64 기계에서 났습니다.

```
이식 전, 파이썬 포팅 vs torch (5,120 개)
0 ulp: 2803    1 ulp: 1979    2 ulp: 313    3 ulp: 25      (최악 3 ulp)

이식 후, Rust 포팅 vs torch (227,040 개)
0 ulp: 전부                                                 (완전 일치)
```

앞의 것은 **double 로 계산해 1 회 반올림한 파이썬 포팅**을 torch 와 비교한 것입니다. 정확
반올림과 f32 초월함수의 반올림이 다르니 몇 ulp 가 갈리는 것이 당연했고, 그래서 §3.3 초안은
"libm 때문에 비트 일치는 불가능하다"고 결론지었습니다. **그 결론은 파이썬 포팅에 대해서만
참이었습니다.** Rust 포팅은 double 을 경유하지 않고 torch 와 같은 f32 libm 을 그대로 부르므로
갈릴 곳이 없습니다.

**AVX2/VSX 에서 어떻게 되는지는 측정된 적이 없습니다.** 위의 3 ulp 를 AVX2 의 오차로 인용하지
마십시오 — 그 숫자는 AVX2 에서 나온 것이 아닙니다. 스칼라 경로가 아닌 곳에서는 torch 가 다른
수열을 낼 수도 있고 같을 수도 있으며, **둘 다 확인되지 않았습니다.**

**의미**: aarch64 스칼라 경로는 비트 단위 동일을 기준으로 삼아도 됩니다. 그 외 경로는 근거가
없으므로 float32 허용오차(~1e-6 상대)가 안전한 기본값입니다 — `tools/golden/compare.py:211` 의
`dt.tolerance_for(t_dtype)` 경로가 이미 하는 일입니다. **측정되지 않은 플랫폼에 비트 일치를
요구하지 마십시오** — 실패할지 아닐지를 아는 사람이 없습니다.

---

## 4. 재현하지 못하면 무엇이 무너지는가

이것이 이 문서에서 가장 중요한 절입니다. 결론부터: **무너지는 것은 "만드는 것"이고, 무너지지
않는 것은 "쓰는 것"입니다.**

### 4.1 무너지지 않는 것 (전부 실측)

**(a) 추론 순전파에는 난수가 한 번도 안 쓰인다.** 작은 Llama(`hidden_size=64`, 2 층,
`vocab_size=100`)를 `model.eval()` 로 두고 `TorchDispatchMode` 로 순전파를 계측했습니다:

```
FORWARD eval: total dispatch ops: 178  unique: 25
FORWARD eval: RANDOM ops: {}          <- 난수 op 0 종, 0 회
```

**(b) `dropout` 은 `eval()` 에서 aten 호출 자체가 사라진다.**

```
dropout mode=train : {'aten.empty_like.default': 1, 'aten.bernoulli_.float': 1,
                      'aten.div_.Scalar': 1, 'aten.mul.Tensor': 1}
dropout mode=eval  : {}                <- dispatch 자체가 0
```

`nn.Dropout` 은 `self.training` 이 거짓이면 입력을 그대로 돌려주므로 **`aten.bernoulli_` 를
구현할 필요조차 없습니다**(추론 전용인 한). 참고로 이 Llama 설정의 dropout 관련 항목은
`attention_dropout = 0.0` 하나뿐입니다.

**(c) `generate(do_sample=False)` — 그리디 디코딩에는 난수가 없다.**

```
GENERATE {'do_sample': False}: RANDOM ops: {}
```

**(d) 체크포인트를 적재하면 랜덤 초기화가 아예 실행되지 않는다.** `from_pretrained` 는 모델을
**meta 디바이스**에서 만듭니다 — `modeling_utils.py:3796` 이 `init_contexts` 에
`torch.device("meta")` 를 넣고, `:3180-3183` 의 `post_init` 은

```python
# If we are initializing on meta device, there is no point in trying to run inits
if get_torch_context_manager_or_global_device() != torch.device("meta"):
    self.initialize_weights()
```

로 **초기화를 통째로 건너뜁니다.** deepspeed 등 meta 를 못 쓰는 경로에서도
`initialization.py:254-274` 의 `no_init_weights()` 가 `TORCH_INIT_FUNCTIONS` 14 개를 `empty_func`
로 바꿔치기해 같은 결과를 냅니다. 랜덤 초기화가 실제로 도는 것은
`_initialize_missing_keys`(`:4774-4808`) — **체크포인트에 없는 키에 한해서**입니다.

> 근거의 성격: (a)(b)(c) 는 실측, **(d) 는 코드 판독**입니다. 실제 사전학습 체크포인트를 내려받아
> `from_pretrained` 를 끝까지 돌려본 것은 아닙니다 — **미확인.** 다만 "체크포인트에 없는 키는
> 랜덤 초기화된다"는 것은 코드가 명시하므로, **부분 체크포인트(예: 새로 붙인 분류 헤드)를 쓰면
> 그만큼 §4.2 의 문제가 되돌아옵니다.**

### 4.2 무너지는 것

**(a) `from_config` 로 만든 모델을 골든 하네스로 값 대조하는 것 — 불가능해집니다.**
`FROM_CONFIG.md` §2.1 이 센 75 회 호출 중 32 회가 난수이고, 그 결과가 모든 파라미터의 값입니다.
초기 가중치가 다르면 그 뒤의 순전파 출력도 전부 다릅니다. **`DESIGN.md` §5(391·394 행)가 A 경로의
주된 위험으로 지목한 "수치 불일치가 조용히 번짐" 을 잡는 도구가, 정작 랜덤 초기화된 모델에는
쓸 수 없다**는 뜻입니다.

**(b) 시드 고정 재현성 — 사용자 코드가 `torch.manual_seed(0)` 을 걸고 특정 값을 기대하는 모든
경우가 깨집니다.** 재현성 자체(같은 시드 → 같은 결과)는 우리 RNG 로도 줄 수 있지만, **torch 와
같은 값**은 못 줍니다.

**(c) `torch.Generator` 의 상태 왕복(`get_state`/`set_state`)** — 5056 바이트 레거시 blob 의
바이트 레이아웃까지 맞춰야 상호운용이 됩니다. 우리끼리 왕복만 하면 되는 것이라면 형식은 자유지만,
**저장된 상태를 실물 torch 와 주고받는 것은 포기**해야 합니다.

**(d) `generate(do_sample=True)` 의 샘플링 스트림.** 실측:

```
GENERATE {'do_sample': True, 'top_k': 5, 'temperature': 1.0}:
    RANDOM ops: {'aten.multinomial.default': 4}   (max_new_tokens=4 → 토큰당 1 회)
```

**추론 중에도 난수를 쓰는 유일한 지점이 여기입니다.** 다만 성격이 다릅니다 — 샘플링은 **분포가
맞으면 되는 것**이지 특정 토큰이 나와야 하는 것이 아닙니다. 재현이 필요한 것은 "시드를 고정해
디버깅할 때"뿐입니다. 대신 `aten.multinomial.default` **자체는 반드시 구현해야 하고**, 이것은
`FROM_CONFIG.md` §2.1 의 14 개 목록에 **없습니다** — `from_config` 만 계측했기 때문입니다.
(`docs/design/C_SURFACE.md` 가 `generate()` 를 추적했다면 겹칠 수 있으나 이 문서는 그 파일을 열지 않았습니다.)

**(e) 현재 `aten.randint.low` 의 값 비교 승격.** `tools/golden/cases.py` 는 `_range_check` 로
dtype·shape·범위만 보고 수열은 안 봅니다. 그 파일의 모듈 주석이 적은 판단 — "seed 는 한 생성기의
스트림을 고정할 뿐, 다른 알고리즘이 같은 값을 내게 만들지 못한다" — 은 **candle 을 쓰는 한
정확하고**, 포팅하면 그 제약이 사라집니다. 다만 torch 의 `randint` 가 균일 정수를 뽑는 정확한
알고리즘(`random_` → `uniform_int_from_to_distribution` 계열, 범위별 분기와 모듈로/거부 처리)은
**이 조사에서 재현하지 않았습니다 — 미확인.**

> **Correction (2026-09, `docs/kernels/RANDINT.md`):** 위 문단이 예고한 승격은 **일어나지 않았고, 그
> 사이에 결함이 하나 숨었습니다.** `uniform_`/`normal_` 이 포팅될 때 `randint` 는 따라오지
> 않아서 candle 의 시드 불가능한 생성기를 계속 썼고, `_range_check` 는 범위 소속만 보므로
> 그것을 **잡을 수 없었습니다** — 잘못된 생성기도 `[low, high)` 는 완벽히 만족합니다. 결국
> `docs/architectures/DEMAND3.md` 의 모델 스윕에서 토큰 id 가 양쪽에서 다르게 뽑히는 것으로 드러났습니다.
> 지금은 포팅됐고(`rng.rs::randint_from_to_fill`), `_range_check` 는 사라졌으며,
> `randint`/`randperm` 은 `_rng_stream_check` 로 **시드를 맞춘 원소 단위 비교**를 받습니다.
> 아래 §6 이 "미확인" 으로 남긴 알고리즘은 `docs/kernels/RANDINT.md` §1~§4 가 실측했습니다 —
> **원소당 한 번의 모듈로, 폭 2²⁸ 에서 32→64비트, 거부 샘플링 아님.**

### 4.3 요약표

| | torch RNG 를 포팅하지 않으면 |
|---|---|
| `import torch` | 영향 없음 |
| `from_config` 로 모델 생성 (구조적 통과) | **된다** — 값만 다름 |
| `from_config` 결과를 골든 대조 | **불가능** |
| `from_pretrained` 로 체크포인트 적재 | **영향 없음** (초기화가 실행되지 않음, 코드 판독) |
| 체크포인트에 없는 키(새 헤드 등) | **영향 있음** — 그 부분만 값이 달라짐 |
| `eval()` 순전파 | **영향 없음** (난수 op 0 회, 실측) |
| `dropout` (`eval`) | **영향 없음** (dispatch 0, 실측) |
| `generate(do_sample=False)` | **영향 없음** (난수 op 0 회, 실측) |
| `generate(do_sample=True)` | 분포는 맞음, **시드 재현은 불가** |
| 사용자 코드의 `torch.manual_seed` 기대값 | **깨진다** |
| `Generator.get_state()` 를 실물 torch 와 교환 | **깨진다** |

---

## 5. 권고

**1. torch 의 CPU RNG 를 직접 포팅한다. candle 의 `rand_uniform`/`rand_normal` 은 쓰지 않는다.**

이유는 품질 선호가 아니라 **선택지가 없어서**입니다 — candle CPU 는 시드를 거부하므로(§2.1)
"candle 로 하되 시드를 맞춘다"는 중간 안이 존재하지 않습니다. 그리고 포팅 비용이 §3.2 에서 실측한
대로 작습니다. 값을 `Vec` 에 채워 `Tensor::from_vec` 으로 넘기는 배선은 이미 코드베이스에 있습니다.

**2. 순서는 `uniform_` → `normal_` 경로 B → `normal_` 경로 A 다.**

- `uniform_` 만으로 `kaiming_uniform_`·`uniform_`·`xavier_uniform_` 이 전부 열리고, **비트 단위
  일치**라 골든 하네스에서 가장 강한 판정을 줄 수 있습니다.
- 경로 B 는 로직이 단순하고(캐시 1 개짜리 Box–Muller, double 고정), 작은 텐서 전부를 덮습니다.
- 경로 A 는 `size >= 16 && is_contiguous()` 분기와 꼬리 겹침 재계산 때문에 가장 틀리기 쉽습니다.
  **`n = 15, 16, 17, 20, 32` 를 반드시 케이스로 두십시오** — 이 다섯 개가 두 경로의 경계와 꼬리
  재계산을 모두 건드립니다(§1.3 의 실측값을 기준선으로 그대로 쓸 수 있습니다).

**3. 골든 하네스의 기대치를 op 별로, 그리고 플랫폼/구현 경로별로 나눈다.**

| op | 대상 | `expect` | 비교 기준 |
|---|---|---|---|
| `aten.uniform_.default` | 모든 플랫폼 | `match` | **비트 단위** 가능 |
| `aten.normal_.default` | aarch64 스칼라 경로 | `match` | **비트 단위** (§3.3 실측) |
| `aten.normal_.default` | 그 외 (AVX2/VSX 포함) | `match` | float32 허용오차 — **미측정이라서지, 3 ulp 로 재서가 아닙니다** |
| `aten.randint.low` | 모든 플랫폼 | ~~현행 `_range_check` 유지~~ → **`match`** | ~~알고리즘 재현이 **미확인**이므로 승격은 그 확인 이후~~ → `docs/kernels/RANDINT.md` 가 재현했고 **비트 단위**로 승격됐습니다. `_range_check` 는 삭제됐습니다 |
| `aten.randint.default` / `aten.randperm.default` | 모든 플랫폼 | `match` | 같은 근거. `randperm` 은 같은 엔진의 Fisher–Yates라 함께 들어왔습니다 |

`cases.py` 모듈 주석의 "두 RNG 는 시드로 맞출 수 없다"는 서술은 **candle 을 쓰는 동안은 맞고,
포팅 후에는 틀립니다.** 포팅을 하면 그 주석과 `_range_check` 의 근거를 함께 갱신해야 합니다.

**4. 시드를 안 맞춰도 되는 곳에 비용을 쓰지 않는다.**

`multinomial`(do_sample)은 **분포만 맞으면 됩니다.** 여기까지 스트림을 맞추려 하지 마십시오 —
비용 대비 얻는 것이 "시드 고정 디버깅" 뿐입니다. 다만 `aten.multinomial.default` 는 `IMPLEMENTED`
목록에 없고 `FROM_CONFIG.md` 의 14 개에도 없으므로, **`generate` 를 목표로 삼는 순간 새 항목으로
추가**해야 합니다.

**5. 골든 하네스의 진짜 결론: 모델 단위 대조는 체크포인트로 해라.**

`DESIGN.md` §5 가 걱정한 "수치 불일치가 조용히 번짐" 은 **랜덤 초기화 모델로는 관측할 수 없습니다.**
`from_pretrained` 는 고정된 가중치를 읽으므로(§4.1(d)) 양쪽이 **같은 입력·같은 가중치**에서
출발하고, 그때 비로소 순전파 출력을 값으로 대조할 수 있습니다. 즉 op 단위 골든은 지금처럼
`tools/golden/` 이 맡고, **모델 단위 골든은 체크포인트 적재 이후로 미루는 것이 옳습니다.**
RNG 포팅은 그 대조를 가능하게 하려고 하는 것이 아니라, `from_config` 를 통과시키고 시드 재현성을
주장할 수 있게 하려고 하는 것입니다 — **두 목적을 섞지 마십시오.**

---

## 6. 미확인 — 명시

- **`from_pretrained` 실물 확인.** §4.1(d) 는 `modeling_utils.py` 판독이고, 실제 사전학습
  체크포인트를 내려받아 초기화가 건너뛰어지는지 계측하지 않았습니다.
- ~~**torch `randint` 의 정수 균일 알고리즘.**~~ **닫혔습니다 — `docs/kernels/RANDINT.md`.** 거부
  샘플링이 아니라 **모듈로**이고(`(val % range) + base`, 헤더가 스스로 "approx 5% skew" 라고
  적습니다), 소비량은 원소당 정확히 한 번이며, 그 한 번이 32비트냐 64비트냐는 **폭 2²⁸**
  에서 갈립니다 — dtype 은 관여하지 않습니다. 헤더의 첫 분기(`#ifdef FBCODE_CAFFE2`, 2³²
  + dtype 목록)는 공개 휠이 컴파일하는 쪽이 아니므로 그대로 옮기면 틀립니다. 부동소수
  dtype 은 `update_from`/`update_to` 로 경계가 먼저 이동합니다.
- **`bernoulli_`(train 모드 dropout)의 스트림.** `eval` 에서 안 쓰이므로 조사 범위 밖으로 두었습니다.
  (그 뒤 `aten.rs::bernoulli_inplace_float` 로 구현됐습니다 — 이 문서가 재측정한 것은 아닙니다.)
- **bf16 / fp16 `normal_` 경로 A 의 완전 재현.** `DistributionTemplates.h:196-219` 의 스택 버퍼
  경로를 코드로 확인했으나, f32/f64 처럼 값까지 대조하지는 않았습니다. fp16 은 `n=16` 에서 f32 와
  근사하게 일치하고 `n=20` 에서 갈리는 것만 관측했습니다.
- **AVX2 경로.** `NormalFill16<float, true>` 특수화(`:112-165`)는 `CPU_CAPABILITY_AVX2` 에서만
  컴파일됩니다. 이 조사는 Apple Silicon 에서만 재고, x86 에서 SIMD 특수화가 스칼라판과 **비트 단위로
  같은 값을 내는지** 확인하지 않았습니다. 다르다면 §3.3 의 ulp 예산이 플랫폼마다 달라집니다.
- **다른 아키텍처가 요구하는 초기화 분포.** `FROM_CONFIG.md` §4.2 가 남긴 `xavier_*`·
  `kaiming_normal_`·`trunc_normal_`·`orthogonal_`·`dirac_`·`eye_`·`sparse_` 는 여기서도 다루지
  않았습니다. 다만 이들 대부분은 `uniform_`/`normal_` 위에 얹힌 파이썬 계층이라 §5 의 1·2 번을
  하면 대부분 따라옵니다(`orthogonal_` 은 QR 분해를 추가로 요구 — 미확인).
- **`torch.default_generator` 를 여러 스레드에서 동시에 쓰는 경우.** torch 는 뮤텍스로 잠그지만
  (§1.4), 잠금 순서에 따른 스트림 분할은 재지 않았습니다.
