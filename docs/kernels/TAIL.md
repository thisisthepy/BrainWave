# 꼬리 다섯 개 — 골든 케이스는 붙었고, 붙이는 과정에서 커널 버그 두 종류를 찾았다

앞선 세션이 `aten.baddbmm.default` · `aten.split_with_sizes.default` · `aten._safe_softmax.default` ·
`aten.add_.Tensor` · `aten.mul.Scalar` 다섯 개를 `rust/torch_c/src/aten.rs` 에 구현하고 커밋
(`9612146`)까지 마친 뒤 인터럽트로 죽었다. 다섯 다 빌드는 깨끗했고 `_aten_implemented()` 에도
이미 올라 있었지만, `tools/golden/cases.py` 에 케이스 빌더가 하나도 없어 `compare.py` 가 다섯
전부를 `<no case builder registered>` 로 하드 실패시키고 있었다. 이 문서는 그 다섯 개의 케이스
빌더를 채운 기록이다.

**결론 먼저.** 다섯 다 채웠고, 골든은 **2257/2258** 이다 (1 개는 의도적으로 빨갛게 남겨둔 것 — 아래
§2 참고). 케이스를 쓰면서 커널의 자기 자신 doc comment 를 다시 재봤고, doc comment 가 주장하는
것과 실제 torch 2.13.0 의 동작이 다른 지점을 세 군데 찾았다. **`rust/torch_c/src/aten.rs` 는 건드리지
않았다** — 작업 범위 밖이고, 지시가 명시적으로 "고치지 말고 보고" 였다.

---

## 0. 도달한 숫자 — 전부 종료 코드와 함께

| 검증 | 이전 (커밋 브리핑 기준) | 이후 | 종료 코드 |
|---|---|---|---|
| 골든 케이스 | 2095/2100 (5개 `<no case builder registered>`) | **2257/2258**, 1 실패 (의도됨, §2) | `1` (의도된 적색 1개 때문) |
| ops covered (`_aten_implemented()`) | 91 (5개는 광고만 되고 골든 비교 없음) | **96** (다섯 다 실제로 골든 비교됨) | — |
| pending case builders | 5 | **0** | — |
| `--inject-fault value` | — | 2247/2258 (11개 CAUGHT) | `1` (문서화된 동작 — §4) |
| `--inject-fault shape` | — | 2247/2258 (11개 CAUGHT) | `1` |
| `--inject-fault dtype` | — | 2247/2258 (11개 CAUGHT) | `1` |
| `--self-test` (`pytests/run.sh`) | — | 11 comparators × 11 fault modes, 0 problem, 0 미가동 | `0` |
| 스키마 (`verify_schemas.py`) | — | 204/204 (overloads 93/93, methods 111/111) | `0` |
| 호스트 빌드 + 실제 임포트 | — | 통과 (`pytests/run.sh` 자체가 이걸 포함) | `0` |
| Android (`cargo ndk`) | — | 통과 | `0` |
| iOS (`aarch64-apple-ios`) | — | 통과, `@rpath/Python.framework/Python` 링크 확인 | `0` |
| falcon/bloom/gpt_bigcode 미구현 (재측정) | — | **0 / 0 / 0** (§5) | — |

`_aten_implemented()` 자체는 91 → 96 이 아니라 **원래부터 96 이었다** — 이 리스트는 Rust 쪽 정적
배열이고 골든 케이스가 있든 없든 값이 바뀌지 않는다. 바뀐 것은 "96개 중 5개가 케이스 없이 하드
실패하던 상태"에서 "96개 전부 실제로 골든 비교되는 상태"로 넘어간 것이다.

---

## 1. 다섯 개의 케이스 빌더 — 무엇을 어떻게 쟀는지

전부 `tools/golden/cases.py` 에 있다. 커널의 doc comment 를 출발점으로 삼되, **doc comment 를
그대로 베끼지 않고 하나씩 상류 torch 2.13.0 (`/Volumes/macMini/caches/spike-venv/bin/python`)
으로 재확인**한 뒤 케이스를 썼다 — 이 모듈 자체의 규칙(`_pair` 위의 note)이 그렇게 하라고 되어
있다.

### `aten.baddbmm.default` — `baddbmm_cases`

`addmm`(GEMM 스케일 규칙, self 브로드캐스트)과 `bmm`(배치 랭크 검사, 배치 브로드캐스트 금지)을
합친 모양. 재확인한 것:

- **beta=0/알파=0 두 quick return 중 beta=0 은 진짜다.** `nan` self 에 `beta=0` 을 주면 상류가
  깨끗한 곱을 준다 (0*nan 이 아니라) — 확인됨, 통과.
- **alpha=0 은 addmm 의 규칙이 아니다.** §2 에서 자세히.
- **self 브로드캐스트** — 1-D, 0-d, 2-D(배치 차원 없음), 배치=1 인 3-D 넷 다 실측; 메시지까지
  addmm 의 expand 에러 문구와 글자 그대로 일치 (`"The expanded size of the tensor (...) must
  match the existing size (...) ..."`, `"expand(torch.{}Tensor{...}, size=...): the number of
  sizes provided (...) must be greater or equal..."`).
- **int64 alpha 절단** — doc comment 는 "alpha=1.9 와 alpha=1 이 int64 삼중에서 bit-for-bit 일치"
  라고 주장하지만, candle 에 int64 matmul 커널 자체가 없어서 (`mm`/`addmm`/`bmm` 이 이미 갖고 있는
  gap, `_MM_C_ERROR_DTYPES`) alpha 값과 무관하게 항상 `c_error` 다. 이 주장은 **오늘 시점에
  검증 불가능**하다 — §2 에 기록.
- **랭크/dtype 거부 8종** — batch1 이 2D, batch2 가 2D, dtype 불일치(batch1/batch2, self/batch2),
  배치 수 불일치, 내부 차원 불일치, self expand 실패, self 랭크 초과, `Bool`/`UInt32` dtype 거부.
  전부 `expect="both_error"` (양쪽 다 raise 하면 되고 메시지는 비교 안 함 — `compare.py` 의
  `both_error` 정의 그대로). 다만 랭크 거부 하나는 상류의 실제 체크 순서가 커널의 것과 다르다는
  것도 측정했다 — §3.
- **모델 스케일** — `_big_gemm_case` 재사용, `batch=2, with_bias=True, k=512`, float32/float16.

### `aten.split_with_sizes.default` — `split_with_sizes_cases`

`split.Tensor` 의 `_chunk_list_check` 비교기를 그대로 재사용 (리스트-of-텐서 반환). 재확인한 것:

- **크기 합이 정확히 맞아야 한다** — `split.Tensor` 와 달리 "마지막 청크는 짧아도 된다"는 관용이
  없다. `[3,3,3]` (합 9, extent 10) 과 `[5,10]` (합 15) 둘 다 상류/샤임 양쪽에서 정확히 같은
  문구로 거부됨 (`"split_with_sizes expects split_sizes to sum exactly to {N} ... but got
  split_sizes={...}"`).
- **음수 항목 거부** — `[5,-5,10]` (합은 10 으로 맞아떨어짐에도) 거부. 문구 일치 확인.
- **크기 0 인 청크가 dimension 이 비어있지 않아도 허용됨** — `[0,10]`, `[10,0]`, `[4,0,6]` 셋 다
  실측, 상류와 값 일치.
- **0-d 텐서 거부** — `split.Tensor` 와 같은 문구 (`"split expects at least a 1-dimensional
  tensor"`), 오버로드를 구분하지 않는다는 것까지 확인.
- **gpt_bigcode 의 실제 호출 모양** — `c_attn(x).split((embed_dim, kv_dim, kv_dim), dim=2)` 를
  본떠 `(2,2,6)` 을 `[3,1,2]` 로, `dim=2` 와 `dim=-1` 둘 다.

### `aten._safe_softmax.default` — `safe_softmax_cases`

`_softmax.default` 와 유일하게 다른 지점(전부 `-inf` 인 행)을 중심으로 구성. 재확인한 것:

- **전부 `-inf` 인 행은 0, NaN 이 아니다** — `_softmax.default` (plain) 는 같은 입력에 NaN 을
  주는데(`-inf - (-inf)`), `_safe_softmax` 는 0. 1-D, 2-D(배치 하나만 마스킹), 완전 마스킹 셋 다
  실측.
- **부분 마스킹은 plain softmax 와 같다** — max-subtraction 이 이미 안전하게 만들기 때문. 확인.
- **`dtype` 인자는 정수 거부보다 먼저 캐스팅한다** — `_safe_softmax(int64_tensor, 0,
  torch.float32)` 는 성공하는데, `dtype` 없이 `_safe_softmax(int64_tensor, 0)` 는
  `NotImplementedError` (`"softmax_lastdim_kernel_impl" not implemented for 'Long'`). 둘 다
  케이스로 pin.

### `aten.add_.Tensor` — `add__tensor_cases`

`fill_.Scalar`/`copy_.default` 의 관례(케이스마다 새 operand, in-place 뮤테이션 비교)를 따름.
재확인한 것 — 그리고 여기서 두 번째 진짜 버그를 찾았다: §2.

### `aten.mul.Scalar` — `mul_scalar_cases`

`rsub_scalar_cases` 와 같은 `arith_tag` 승격 규칙(정수 텐서 + 정수 스칼라 → 그대로, + 실수
스칼라 → 기본 float 로 승격)을 재확인. 여기서 세 번째 발견 — §2.

---

## 2. 커널이 틀린 곳 — 고치지 않고 보고만 한다

지시대로 `rust/torch_c/src/aten.rs` 는 건드리지 않았다. 아래 셋은 케이스 작성 중 상류와 대조하다
찾은, doc comment 의 주장과 실제 동작이 갈리는 지점이다.

### 2.1 `baddbmm` 의 `alpha=0` 은 `addmm` 의 quick return 이 아니다 — 골든에 빨간 케이스로 pin

커널의 doc comment: *"the zero-fast-return and integral-truncation rules are addmm's, reused
rather than re-derived, and measured to still hold batched: ... baddbmm(self, inf_b1, b2,
alpha=0) gives back self unchanged."*

실측 (상류 torch 2.13.0):

```python
self_full = torch.zeros(2,2,2)
inf_b1 = b1.clone(); inf_b1[0,0,0] = float('inf')
torch.ops.aten.baddbmm.default(self_full, inf_b1, b2, alpha=0)
# tensor([[[nan, nan], [0., 0.]], [[0., 0.], [0., 0.]]])
```

`self` 가 전부 0인데도 결과에 **NaN 이 남는다.** `addmm` 의 같은 실험 (`addmm(self=zeros, inf_m1,
m2, alpha=0)`) 은 실제로 깨끗한 0 을 준다 — 그건 `addmm_cases` 가 이미 pin 하고 있고 재확인해도
맞다. `baddbmm` 은 다르다: 상류는 `alpha=0` 이어도 곱을 건너뛰지 않고 실제 IEEE 연산으로
`0 * inf = nan` 을 계산한다. 커널의 Rust 코드는 `!alpha_zero` 분기로 곱셈 자체를 건너뛰므로
(`addmm_scale` 의 규칙을 그대로 재사용), 상류가 NaN 을 주는 자리에서 깨끗한 0 을 준다.

`tools/golden/cases.py::baddbmm_cases` 에 이 케이스를 **`expect="match"` 로 그대로 남겨뒀다** —
`c_error`/`torch_error` 어느 쪽도 맞지 않는다 (양쪽 다 *성공*하고, 값만 다르다). `_FULL_FILLS`
모듈이 이미 세운 선례와 같은 방식: 실제 버그를 담은 **살아있는 회귀 트랩**으로 남겨서, 커널이
고쳐질 때까지 골든이 계속 빨갛게 이 사실을 말하게 했다. 케이스 이름에
`"KNOWN KERNEL BUG (docs/kernels/TAIL.md)"` 를 박아뒀다.

### 2.2 `.Scalar` 오버로드의 `torch.bool` 거부는 과잉이다 — `mul.Scalar` 와 `add_.Tensor` 양쪽에서 확인

`arith_tag` (mul.Scalar/add.Scalar/sub.Scalar/div.Scalar/rsub.Scalar/mul.Tensor/add.Tensor/... 가
전부 공유하는 승격 함수)는 `tensor == TorchDType::Bool` 이면 무조건 거부한다. `.Tensor` 오버로드에는
맞는 규칙이다 (상류의 bool `*`/`+` tensor-tensor 는 진짜 논리 연산). **`.Scalar` 오버로드에는
틀렸다:**

```python
torch.ops.aten.mul.Scalar(torch.tensor([True,False,True]), 3)
# tensor([3, 0, 3])            <- int64, 산술 곱셈이지 논리곱이 아니다
torch.ops.aten.mul.Scalar(torch.tensor([True,False,True]), 2.5)
# tensor([2.5, 0., 2.5])       <- float scalar 는 float32 로 승격, 정수 스칼라와 같은 규칙
```

`True`/`False` 를 `1`/`0` 으로 읽어 **일반 정수 텐서와 똑같이 승격 규칙을 따라 산술 곱셈**을
한다. 샤임은 dtype 을 보자마자 거부한다. `mul_scalar_cases` 에 `expect="c_error"` 로 두 케이스
(정수 스칼라, 실수 스칼라) pin.

`add_.Tensor` 에서도 같은 종류를 찾았다 — 다만 여기는 커널 doc comment 가 **스스로 자기모순**을
담고 있다. doc comment: *"torch.bool is refused, matching add.Tensor's own refusal — upstream's
in-place bool add is a logical or (measured: tensor([True,False]).add_(tensor([True,True]))
gives [True, True])."* 이 문장 자체가 "상류는 계산한다"고 말하면서 "샤임이 거부하는 게
`add.Tensor` 의 거부와 일치한다"고 결론짓는다 — 하지만 **`add.Tensor`(out-of-place) 의 거부가
상류와 일치하는지는 검증되지 않았다.** 재확인:

```python
torch.ops.aten.add_.Tensor(torch.tensor([True,False]), torch.tensor([True,True]))
# tensor([True, True])   <- 상류는 성공한다 (논리합)
```

`_C._aten_dispatch("aten.add_.Tensor", bool_t, bool_t)` 는 `NotImplementedError` 를 던진다.
**상류 성공, 샤임 거부 — `c_error`**, doc comment 가 암시하는 `both_error` 가 아니다.
`add__tensor_cases` 에 pin. (`aten.add.Tensor` 의 기존 케이스 빌더는 bool 을 아예 테스트하지
않아서 — grep 해서 확인 — 같은 종류의 갭이 out-of-place 쪽에도 있는지는 이 작업의 범위 밖이라
확인하지 않았다. 후속 작업감으로 남긴다.)

같은 파일에서 `add_.Tensor` 의 doc comment 가 언급한 **다른** known gap 도 재확인해 pin 했다
(이건 doc comment 가 맞았다): `int32.add_(float32_tensor)` 는 상류가 거부하는데
(`"result type Float can't be cast to the desired output type Int"`) 샤임은 `other` 를 receiver
dtype 으로 캐스팅해 계산해버린다 — `torch_error`.

### 2.3 (작은 것) `baddbmm` 의 랭크 거부 메시지 — 상류의 실제 체크 순서가 다르다

커널 doc comment 는 `bmm` 의 랭크 검사를 그대로 재사용한다고 적었지만, 2D `batch1` 을 실제로 줘
보면 상류는 `"batch1 must be a 3D tensor"` 를 주지 않는다:

```python
torch.ops.aten.baddbmm.default(torch.zeros(2,2,2), torch.ones(2,3), b2_3d)
# RuntimeError: The expanded size of the tensor (3) must match the existing size (2)
# at non-singleton dimension 1. Target sizes: [2, 3, 2]. Tensor sizes: [2, 2, 2]
```

상류는 `batch1`/`batch2` 의 랭크를 검사하기 전에 `self` 의 broadcast 가능성부터 확인하는
것으로 보인다 (`batch1.size(0)`, `batch1.size(1)`, `batch2.size(-1)` 만 읽어 target shape 을
만들고, 그 자리에서 self 검사가 먼저 실패한다). 샤임은 `batch1`/`batch2` 랭크를 제일 먼저
확인해 그 자리에서 즉시 `"batch1 must be a 3D tensor"` 를 던진다.

**골든 결과에는 영향 없다** — `expect="both_error"` 는 양쪽이 raise 하기만 하면 되고 메시지를
비교하지 않는다 (`compare.py` 의 정의 그대로), 그리고 실제로 양쪽 다 raise 한다. 다만 커널의 doc
comment 가 "bmm 의 랭크 체크를 그대로 재사용한다"고 적은 것은 **체크 순서까지는** 맞지 않는다는
걸 기록해둔다.

---

## 3. golden 케이스를 쓰면서 확인한 재현성

- 다섯 op 모두 **케이스 리스트 생성 시점에 텐서를 만들지 않는 위험은 없었다** — 다섯 다 이미
  `_aten_implemented()` 에 있었고, `_tensor_from_flat` 로 bool 텐서 구성도 (모듈 상단 주석이
  경고하는 옛 BOOL.md 제약과 달리) 이 빌드에서는 성공한다 (`_tensor_from_flat([1,0,1],[3],
  dtype=c.bool)` 실측 성공). 다만 파일 관례를 따라 `_pair` 를 통한 지연 생성은 그대로 지켰다.
- `_MM_C_ERROR_DTYPES` (int64/int32/int16/uint8 — candle 에 정수 matmul 커널 없음) 는
  `baddbmm` 에도 그대로 상속된다는 걸 실측으로 재확인했다.
- `uint16`/`uint64` 는 `_tensor_from_flat` 로 구성 자체가 안 된다 (`"dtype not storable by the
  candle backend"`) — `baddbmm` 자체의 거부와는 무관한, 별개의 저장소 갭이라 `baddbmm` 의 dtype
  거부 케이스는 `bool`/`uint32` (둘 다 구성은 되고, `baddbmm` 자체가 거부하는 걸 보여준다) 로
  좁혔다.

---

## 4. `--inject-fault` 종료 코드 1 은 정상이다

`value`/`shape`/`dtype` 세 모드 다 `exit=1` 이고 `2247/2258` 통과, `11` 실패다. `compare.py` 의
안내 문구 그대로: *"exit code stays 1 whenever any fault was CAUGHT ... Use --self-test for the
pass/fail gate."* 게이트는 `pytests/run.sh` (자기검사 포함) 이고 그건 `exit=0` 이다.

---

## 5. falcon/bloom/gpt_bigcode 재측정 — `_aten_implemented()` 미구현 0

`docs/architectures/ARCH.md` 의 방법(2 층, hidden 64, heads 2, intermediate 128, `TorchDispatchMode`,
`generate` 의 greedy ∪ do_sample 합집합)을 그대로 따르되 표본을 이 세 아키텍처로 좁혀 다시 쟀다.
**권위 있는 소스는 `_C._aten_implemented()` 다** (`_aten_all_implemented()` 가 아니다 — 후자는
골든 비교가 없는 op 도 포함하므로 "미구현" 의 기준이 느슨해진다).

```
falcon:       total_ops=55   missing_from_aten_implemented=0
bloom:        total_ops=53   missing_from_aten_implemented=0
gpt_bigcode:  total_ops=46   missing_from_aten_implemented=0

TOTAL MISSING ACROSS ALL THREE: 0
```

세 아키텍처 다 `generate()` (greedy + do_sample, 4 토큰씩) 가 부르는 op 전부가 `_aten_implemented()`
안에 있다.

### 로짓 최대 차이 — 미완료, 추측하지 않고 그렇게 적는다

**토큰만 보고 판정하지 말라는 지시대로, 실제 로짓 수치 대조를 시도했다.** `docs/architectures/ARCH.md` §5
(Gemma/BERT) 가 쓴 손-전사 방식 대신, `TorchDispatchMode` 로 매 aten 호출을 가로채 실제 `_C`
샤임으로 계산하고 결과를 다시 진짜 `torch.Tensor` 로 되돌리는 **범용 자동 라우터**를 새로 짜서
`FalconForCausalLM`/`BloomForCausalLM`/`GPTBigCodeForCausalLM` 을 (HF 모듈 코드 그대로) 상류
위에서 한 번, 샤임 라우팅으로 한 번 돌려 로짓을 직접 빼려고 했다.

**끝내지 못했다.** 디버깅 중 나온 문제들:

- `t + 1` 같은 연산자 경로는 `aten.add.Tensor` 오버로드에 **감싸지 않은 파이썬 스칼라**를 그대로
  넘긴다는 걸 처음 알았다 (측정: `TorchDispatchMode` 로 `a + 1` 을 가로채면 `other` 의 타입이
  `int`) — `.Tensor` 오버로드가 스칼라를 못 받는 샤임 커널과 안 맞아서, `add`/`sub`/`mul`/`div`/
  `rsub`/`pow` 로 범위를 좁혀 `.Scalar` 오버로드로 리다이렉트하는 임시방편을 넣었다.
- `torch.dtype` 객체를 샤임의 dtype 객체로 변환하는 게 빠져 있어서 `aten.full.default` 가
  깨졌다 — 고쳤다.
- `pin_memory`/`layout`/`device` 같은, 실제 ATen 디스패처가 항상 채워 넣는 기본값 kwargs 를
  샤임이 안 받아서 필터링을 넣었다.
- 이 셋을 고친 뒤에도 `gpt_bigcode`/`falcon` 은 `aten.cat.default` 에서 **랭크가 안 맞는다는
  IndexError** 로 죽었고 (`dim=-2` 를 요구하는데 실제 들어온 텐서가 rank 1 로 보임), `bloom` 은
  `aten.pow.Tensor_Tensor` 에서 `float32 vs int32` 프로모션이 안 된다는 `NotImplementedError` 로
  죽었다. 후자는 진짜 샤임 갭일 수 있지만, 전자는 **이 자동 라우터 자체의 변환 버그로 보인다**
  (스칼라 재라우팅 휴리스틱이 `.Tensor` 오버로드를 쓰는 다른 op — 예를 들어 `slice.Tensor` —
  에도 잘못 걸려 존재하지 않는 `.Scalar` 오버로드를 부르는 걸 먼저 봤고, 범위를 좁혔음에도
  비슷한 종류의 오염이 남아있을 가능성이 높다).

이 둘을 진짜 샤임 갭인지 하니스 버그인지 가르려면 ATen 오버로드 해석을 제대로 재구현해야 하는데,
이 작업(다섯 개 케이스 빌더)의 범위를 크게 벗어난다. **로짓 최대 차이는 모른다 — 추측해서 적지
않는다.** §0 표의 "미구현 0" 세 개는 신뢰도가 높다(직접 측정, 권위 있는 소스 사용); 로짓 수치는
미검증으로 남겨둔다.

---

## 6. 다음에 와야 하는 것

- **2.1** — `baddbmm` 의 `alpha=0` 가 upstream 처럼 곱을 스킵하지 않도록(또는 스킵하되 그게
  틀렸다는 걸 알고) 고치는 것. 골든의 빨간 케이스 하나가 정확히 이걸 가리킨다.
- **2.2** — `arith_tag` 의 blanket bool 거부를 오버로드별로 나누는 것 (`.Scalar` 는 산술,
  `.Tensor` 는 논리). `add.Tensor`(out-of-place) 도 bool 을 테스트하는 케이스가 없어서 같은
  종류의 갭이 있는지 확인이 안 됐다 — 먼저 확인부터.
- **§5 로짓 대조** — 자동 라우터를 계속 다듬거나, `docs/architectures/ARCH.md` §5 가 Gemma/BERT 에 쓴 손-전사
  방식을 falcon/bloom/gpt_bigcode 에도 적용하는 것. 후자가 더 느리지만 이 저장소에서 이미 검증된
  방법이다.

---

## 7. §2.2 의 후속 — 거부의 *이유* 가 틀려 있었다

§6 의 두 번째 항목(`arith_tag` 의 blanket bool 거부를 오버로드별로 나눌 것)에 대해, 그 사이에
`docs/verification/AUDIT.md` 가 **살아 있는 코드 결함**을 하나 더 찾아 보고만 하고 두었다. 이 절이 그것이다.

§2.2 가 쓰인 뒤 어느 회차에서 거부 메시지가 바뀌어 있었다:

```
aten.mul.Scalar: torch.bool operands are logical, not arithmetic, in torch
                 (BOOL.md §2.2) and are not implemented in torch._C shim
```

**그 문장 자체가 §2.2 의 측정과 모순된다.** `docs/numerics/BOOL.md` §2.2 의 표가 잰 것은 bool 텐서 둘의
`x + x` 였는데, 그 결론이 모든 오버로드에 대한 blanket 메시지로 일반화되면서 `.Scalar` 에 대해
**사실이 아닌 주장**을 하게 되었다. 이 저장소가 이번 세션에만 다섯 번 겪은 "낡은 거부" 의
여섯 번째이고, 앞의 다섯이 각각 작업을 잘못된 곳으로 보냈다.

### 7.1 실제로 참인 것 — 12칸 전부 재측정 (torch 2.13.0)

|  | `add` | `sub` | `mul` | `div` |
|---|---|---|---|---|
| `.Tensor` | **논리합**, `bool` | **거부** | **논리곱**, `bool` | **산술**, `float32` (`[1.0, nan, 1.0]`) |
| `.Tensor` in-place | 논리합, `bool` | 거부 | 논리곱, `bool` | 거부 (캐스트 불가) |
| `.Scalar` | **산술**, `int64` (`[4,3,4]`) | **거부** | **산술**, `int64` | **산술**, `float32` |
| `.Scalar` in-place | 거부 (캐스트 불가) | 거부 | 거부 | 거부 |

`rsub.Scalar` 는 `Arith::Sub` 로 들어오고, 상류도 같은 문구로 거부한다.

**"logical, not arithmetic" 이 참인 칸은 12칸 중 2칸뿐이다** (`.Tensor` 의 `add` 와 `mul`).
그리고 같은 doc comment 가 적고 있던 *"`-` and `/` are refused by upstream itself"* 도 절반이
틀렸다 — `div.Tensor(bool, bool)` 은 상류가 계산한다. 세 칸이 틀린 셈이다.

세 종류의 거부 사유가 있고, 셋 다 서로 다르다:

* **`sub` / `rsub`** — 상류도 거부한다. 이유는 "논리 연산이라서" 가 아니라 bool 뺄셈에 의미가
  없어서이고, 상류는 그렇게 말한다(`"Subtraction, the `-` operator, with a bool tensor is not
  supported"`). 즉 **양쪽 다 거부**이며 갭이 아니다.
* **`.Tensor` 의 `add`** — 상류는 논리합을 계산해 `bool` 을 돌려준다. candle 은 2 를 만들고,
  2 는 여전히 truthy 이므로 **조용히 틀린다** (docs/numerics/BOOL.md §6.3 의 불변식). 여기서만
  "논리적이다" 가 참이다.
* **`.Scalar` 전부와 `.Tensor` 의 `div`** — 상류는 **산술로 계산하고 승격한다.** 여기에 없는
  것은 그 승격이지 논리 연산이 아니다.
* **in-place + `.Scalar`** — 상류도 거부하지만 네 번째 이유다: 승격된 결과를 bool 수신자에
  되쓸 수 없다 (`"result type Long can't be cast to the desired output type Bool"`).

### 7.2 고친 것 — 동작이 아니라 메시지

**거부 자체는 하나도 바뀌지 않았고, 전부 여전히 옳다.** 바뀐 것은 각 거부가 대는 이유뿐이다.
`arith_tag` 는 이제 `(kind, .Scalar 인가, in-place 인가)` 로 네 갈래를 나눠 위 표의 사실을
말한다. `mul.Tensor` 의 예외는 그대로다 — {0,1} 위의 산술 곱이 곧 논리곱이라는 §2.2 이후의
논증은 재검증했고 여전히 성립한다.

골든에서 이 칸들이 어떻게 분류되는지도 §8 에서 정리됐다: `add.Scalar(bool, ·)` 는 `c_error`
(상류는 계산, 샤임은 거부), `sub.Scalar(bool, ·)` 는 `both_error` (양쪽 거부). **한 메시지가
이 둘 모두에 맞을 수는 없었다**는 것이 이 결함의 요약이다.

### 7.3 다시 낡지 않게 하는 장치

낡은 거부가 여섯 번 반복된 이유는 **거부 사실은 테스트되고 거부 사유는 테스트되지 않기**
때문이다. `test_the_bool_arithmetic_refusals_each_give_upstreams_actual_reason` 은 12칸의
사유를 문자열로 단언한다 — `.Scalar` 메시지에 `"logical"` 이 다시 나타나면 실패하고,
`div.Tensor` 메시지가 "상류도 거부한다" 로 돌아가도 실패한다.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_bool_arithmetic_refusals_each_give_upstreams_actual_reason present -->
<!-- DOCWATCH: op-implemented aten.add.Scalar -->
<!-- DOCWATCH: op-implemented aten.sub.Scalar -->

§6 의 2.2 항목은 **절반 닫혔다**: 오버로드별로 나뉜 것은 *메시지*이고, `.Scalar` 의 bool 산술을
실제로 *계산*하는 것은 아직 없다. 그것은 승격 경로를 새로 만드는 일이고, `c_error` 케이스로
pin 되어 있다. §6 이 함께 요구한 "`add.Tensor`(out-of-place) 도 bool 케이스가 없어서 확인이
안 됐다" 는 확인은 위 표가 끝냈다 — 상류는 논리합을 계산하고, 샤임은 거부하며, 그것이 갭이다.
