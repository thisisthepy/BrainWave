# `mps` — an eager attention block, and the sentence in §5 that was half right

**결론부터: BERT 가 `mps` 에서 forward 합니다. 그리고 §5 의 두 문장은 각각 절반씩
틀렸는데, 틀린 절반이 서로 반대쪽이었습니다.**

| 질문 | 답 |
|---|---|
| `docs/RELEASE_0_0_13a0.md` §5 의 "No transformer forwards on `mps`" 는 맞았나 | **아니오.** 쓰였을 때 이미 거짓이었다 — `docs/MPSFWD.md` 가 SmolLM2-135M 을 거기서 돌렸고 §5 이 갱신되지 않았다 |
| "every attention block passes through `_softmax`" 는 맞았나 | **SDPA 경로에서는 아니고, eager 경로에서는 맞다.** `docs/MPSFWD.md` 는 SmolLM2 하나로 "어떤 어텐션도 지나지 않는다" 를 결론냈다. `attn_implementation="eager"` 인 BERT 는 레이어마다 두 번 지난다 |
| 그래서 §5 이 지목한 공백은 실재했나 | **실재했다.** 단, 그것을 밟을 수 있는 모델을 `docs/MPSFWD.md` 가 돌리지 않았을 뿐이다 |
| 어텐션 블록을 `mps` 에서 돌리면 실제로 무엇이 거절했나 | 게이트 거절은 **`aten._softmax.default` 하나.** 나머지 세 개는 거절이 아니라 **벽**이었다 (§3) |
| 지금 도는가 | **돈다.** shrunk BERT (2 layer, eager) 가 `mps` 에서 forward 하고, 출력 256개가 upstream 의 `float64` 진실로부터 **5.257e-07** 떨어져 있다 — upstream 자신의 `float32` 오차 **4.320e-07** 의 **1.22배** (§4) |
| 목록은 어떻게 됐나 | **87 → 85.** 둘 다 **다시 쓰여서** 빠졌다. 게이트를 넓힌 곳은 없다 |

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs MPS_HOST_READBACK_OPS present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs softmax_on_device present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsattn.py test_a_bert_encoder_forwards_on_mps_and_agrees_with_upstream present -->

---

## 1. 주장을 먼저 다시 쟀다 — 이 저장소에서 §5 항목이 이미 거짓이었던 것이 세 번째다

지시가 "믿지 말고 다시 재라" 고 한 이유가 이것이고, 이번에도 재는 쪽이 옳았습니다. 다만
**결과가 "이미 닫혀 있었다" 가 아니라 "반쯤 열려 있었다"** 였습니다.

§5 의 문장:

> **No transformer forwards on `mps`.** `aten._softmax.default` is in the
> set of ops refused there, and every attention block passes through it.

두 문장을 따로 재면:

| 문장 | 판정 | 근거 |
|---|---|---|
| "No transformer forwards on `mps`" | **거짓, 쓰였을 때 이미** | `docs/MPSFWD.md` 가 SmolLM2-135M 을 `mps` 에서 forward 시켰다. §5 은 그것을 반영하지 않았다 |
| "every attention block passes through `_softmax`" | **eager 에서 참, SDPA 에서 거짓** | 아래 |

그리고 `docs/MPSFWD.md` 는 그 두 번째 문장을 **정정했는데, 반대 방향으로 지나쳤습니다.**
그것이 적은 것은:

> `_softmax` 는 **한 번도 불리지 않습니다** — 어텐션은
> `F.scaled_dot_product_attention` 으로 내려가고 (…)

이것은 **SmolLM2 에 대한 사실**이고, 그 문서는 그것을 **어텐션 일반에 대한 사실**로 적었습니다.
한 모델 한 입력에서 나온 관측을 전체로 일반화한 것이고, 그 문서 자신의 §2.2 가
*"한 입력, 한 아키텍처"* 라고 경고해 둔 바로 그 자리입니다.

`attn_implementation="eager"` 로 만든 BERT 를 `mps` 에서 돌리면 `_softmax` 가 **레이어마다
두 번** 나옵니다. 2-layer 모델에서 정확히 2회 (§2 의 추적):

```
   2 aten._softmax.default
```

**그러므로 §5 이 지목한 공백은 실재했습니다.** 실재했지만, 그것을 밟는 모델을 아무도 돌리지
않아서 `docs/MPSFWD.md` 가 "그럴 필요가 없다" 로 닫아 두었던 것입니다.

> **이 라운드가 남기는 교훈은 §5 가 틀렸다는 것이 아니라, §5 를 정정한 문서가 그 자리에서
> 새 일반화를 하나 만들었다는 것입니다.** "SmolLM2 는 `_softmax` 를 부르지 않는다" 와
> "어텐션 블록은 `_softmax` 를 지나지 않는다" 는 다른 문장이고, 두 번째를 뒷받침하는 측정은
> 없었습니다.

---

## 2. 무엇이 실제로 거절하는가 — 이름으로

추측하지 않고 쟀습니다. `docs/MPSFWD.md` §2.1 의 방법을 그대로 씁니다: 게이트를 일회용으로
log-and-allow 로 바꾸고, 별도로 door 에 op 이름을 찍는 세 줄을 넣어 빌드한 뒤 실제 forward 를
돌리고 **되돌렸습니다.** 최종 트리에는 둘 다 없습니다.

게이트를 켠 채로 한 번에 하나씩 고치는 방법으로는 알 수 없는 것이 있기 때문입니다 —
첫 거절에서 멈추므로 그 뒤가 보이지 않고, 실제로 이번에도 **첫 거절(`gather`)이 어텐션이 아니라
임베딩에 있었습니다.**

### 2.1 shrunk BERT (eager) 가 `mps` 에서 부르는 op 전부

`MPSATTN_TRACE=1` 로 찍은 것, 99 dispatch / 16 종:

```
  30 aten.view.default          4 aten.matmul.default        2 aten.gelu.default
  13 aten.t.default             4 aten.contiguous.default     2 aten._softmax.default
  13 aten.addmm.default         3 aten.embedding.default      1 aten.tanh.default
  10 aten.transpose.int         2 aten.reshape.default        1 aten.slice.Tensor
   6 aten.add.Tensor            2 aten.mul.Scalar             1 aten.select.int
   5 aten.native_layer_norm.default
```

**이 16개 중 게이트 목록에 있었던 것은 `aten._softmax.default` 하나뿐입니다.** §5 이 지목한
바로 그 이름이고, 이 라운드가 옮긴 것이 그것입니다.

### 2.2 그러나 거절만 고쳐서는 돌지 않았다 — 벽이 셋 더 있었다

`docs/MPSFWD.md` §2.2 가 적어 둔 명제 *"거절되는 op 을 다 옮기면 돈다 는 거짓"* 이 이번에도
그대로 성립했습니다. 셋 다 **거절 목록 밖**이고, 셋 다 되읽기가 아닙니다.

| # | 증상 | 원인 | 고친 방법 |
|---|---|---|---|
| 1 | `aten.matmul.default: Metal error Invalid matmul arguments [256, 8, 32, 1] [256, 8, 1, 32] (8, 8, 8)` | `gemm_with_layout_fallback` 의 재시도가 **CPU 백엔드의 철자만** 알고 있었다 (§3.1) | `is_matmul_striding_refusal` 에 Metal arm |
| 2 | `torch.tensor: Metal contiguous to_dtype F64 F32 not implemented` | `Tensor::from_vec(f64, .., device)` + `to_dtype` — **장치에게** f64 버퍼를 만들라고 시킨다 | `host_built`: 호스트에서 만들고 호스트에서 좁힌 뒤 올린다 |
| 3 | `aten.where.ScalarOther: unsupported const-set f64` | `Tensor::full(f64, (), device)` — `docs/MPSFWD.md` §4 wall 1 과 **같은 실수, 다른 자리** | `host_const` |

2·3 은 `docs/MPSFWD.md` §4 가 이미 한 번 고친 형태입니다. 같은 모양이 두 자리 더 남아
있었다는 뜻이고, **그 문서가 `host_const` 를 만들면서 남은 호출자를 쓸지 않았다는 뜻**입니다.
`Tensor::full(..., device)` 와 `Tensor::from_vec(f64, ..., device)` 는 앞으로도 같은 방식으로
실패하므로 여기 적어 둡니다.

---

## 3. 무엇을 바꿨나

### 3.1 `_softmax` / `_safe_softmax` — 되읽기를 지우는 것 말고 다른 길이 없다

게이트는 **문 앞에서 op 이름으로** 거절하므로, `_softmax` 가 `mps` 에서 돌려면 그 op 이
`MPS_HOST_READBACK_OPS` 를 떠나야 합니다. 그리고 그 목록은 손으로 쓴 것이 아니라
`aten.rs` 를 스캔해 **유도**된 것이므로, 커널이 `read_flat` 을 부르는 한 스캔이 다시 넣습니다.

> **여기서 하지 않은 것을 적어 둡니다.** `read_flat` 호출을 `host_softmax_cpu_only` 같은
> 이름의 함수 한 단계 아래로 옮기면 **두 테스트가 모두 통과합니다** — per-op 스캔은 커널
> 본문에서 여섯 헬퍼 이름만 찾고, 분류 테스트는 마커(`to_vec*`/`to_scalar`/`to_cpu`)만
> 찾는데 `read_flat` 은 마커가 아니기 때문입니다. 즉 **거절 목록에서 이름을 빼면서 되읽기는
> 그대로 두는 방법이 존재하고, 지금의 검사는 그것을 잡지 못합니다.**
> `docs/VOICE3.md` 가 "한 단계만 따라간다" 로 경고한 사각지대의 정확한 모양이고,
> 이 문단이 그것을 실행하지 않았다는 기록입니다.

그래서 되읽기를 **양쪽 장치에서 전부** 지웠습니다. `softmax_on_device` 는 candle op 다섯 개입니다:

```
max_keepdim  →  broadcast_sub  →  exp  →  sum_keepdim  →  broadcast_div
```

**왜 이것이 근사가 아니라 같은 연산인가.** 옛 스칼라 루프는 들어올 때 `f64` 로 넓히고
축소 dtype 경로에서는 **모든 연산을 `f32` 로 캐스팅해서** 했습니다 — 즉 실제로 수행한 산술은
`f32` max, `f32` 뺄셈, `f32::exp`, `f32` 순차 합, `f32` 나눗셈입니다. 각각이 같은 dtype 의
candle op 하나이고, candle 의 CPU `exp` 는 같은 `f32::exp` 입니다. 구성으로 고정되지 않는
것은 **`sum_keepdim` 안의 합산 순서** 하나이고, 그것에 대한 검사는 golden 코퍼스입니다 —
**11405/11405, 움직이지 않았습니다** (§4).

`acc` 는 이전과 같이 `opmath_type<scalar_t>` 입니다. `float16`/`bfloat16` 은 여전히 `f32` 에서
계산하고 마지막에 한 번 좁힙니다.

### 3.2 `_safe_softmax` 의 죽은 행 — `max == -inf` 는 Metal 에서 답이 아니다

`_safe_softmax` 가 `_softmax` 와 갈라지는 유일한 지점은 **전부 `-inf` 인 행**입니다. 옛 루프는
`max == -inf` 로 그것을 찾았고, 그것은 CPU 에서 참입니다.

**Metal 에서는 아닙니다.** 전부 `-inf` 인 행에 `max_keepdim` 을 하면 `-inf` 가 돌아오지 않고,
그래서 행이 검출되지 않고, `exp(-inf - max)` 가 `0` 으로 언더플로하고, `0 / 0` 이 이 분기가
존재하는 이유인 그 `NaN` 을 돌려줍니다. 처음 구현이 정확히 그렇게 틀렸고, **값 테스트가
그것을 잡았습니다** (§5).

지금의 판정은 torch 자신의 분해와 같은 것입니다 —
`torch/_decomp/decompositions.py::safe_softmax` 는 `dim` 을 따라 **모든** 원소가 `-inf` 인 곳을
마스킹하므로, 그 개수를 세서 폭과 비교합니다. 비교와 1의 합이라 백엔드에 의존하지 않고,
**옳은 방향으로 더 엄격합니다**: `NaN` 을 품은 행은 전부 `-inf` 가 아니고, upstream 도 거기서
`NaN` 을 답합니다.

### 3.3 축소할 것이 없는 두 모양 — golden 이 잡았고 이 파일은 잡지 않았다

rank 0 과 원소 0개. candle 의 축소는 둘 다 **답하지 않고 거절**합니다
(`max: dimension index 0 out of range for shape []`, `empty tensor for reduce`). 옛 스칼라 루프는
`(outer, n, inner) = (1, 1, 1)` 특례로 삼켰기 때문에 이것을 생각할 필요가 없었습니다.

빠뜨린 채로 golden 을 돌려 **9개 케이스가 빨개졌습니다** — 모든 float dtype 에 걸쳐
`_softmax`/`_safe_softmax` 의 0-d 와 empty. §5 에 그 항목이 있습니다.

### 3.4 Metal 의 matmul 거절은 다른 철자였다

`gemm_with_layout_fallback` 은 candle 이 레이아웃을 거절하면 contiguous 로 재시도합니다. 그런데
그 재시도를 켜는 술어가 **CPU 백엔드의 철자(`MatMulUnexpectedStriding`)만** 알고 있었습니다.
Metal 은 `MetalKernelError::MatMulNonContiguous` 로 답하고, 그것은 `Error::Metal` 로 도착하므로
arm 에 걸리지 않았습니다 — **`mps` 에서는 재시도가 한 번도 돈 적이 없습니다.**

eager 어텐션의 `query @ key.transpose(2, 3)` 은 왼쪽 피연산자가 `view`+`permute` 라 contiguous 가
아니고 (`[1,8,4,8]` → `[1,4,8,8]`, stride `[256, 8, 32, 1]`), 그래서 정확히 거기서 멈췄습니다.

메시지로 매칭합니다. 변형이 `candle-metal-kernels` 안에 있고 이 크레이트는 그것을 이름으로
의존하지 않기 때문이며, 그 메시지는 `MetalKernelError` 자신의 `#[error(...)]` 텍스트입니다.

---

## 4. 값 — 무엇과 얼마나 맞는가, 그리고 그 허용치는 어디서 왔는가

`docs/AGREE.md` 의 규칙을 그대로 씁니다: **허용치를 고르지 않고, upstream 자신의
`float32`-대-`float64` 오차에서 읽어냅니다.** 같은 shrunk BERT 를 upstream 에서 `float64` 로도
돌려 진실을 만들고, 세 개의 `float32` 답을 모두 그것에 대고 잽니다.

```
BertModel(vocab 64, hidden 32, 2 layer, 4 head, ffn 64, attn_implementation="eager")
같은 가중치, 같은 입력, 8 토큰.  출력 256개.  max|truth| = 2.77166
```

| | `float64` 진실로부터 | upstream 자신의 오차 대비 |
|---|---|---|
| **upstream `float32`** (오라클 자신) | **4.320e-07** (p90 2.485e-07) | 1.00× — 기준 |
| shim `cpu` | **4.632e-07** | 1.07× |
| shim `mps` | **5.257e-07** | **1.22×** |

`docs/AGREE.md` 의 판정 규칙은 *"같은 출력에 대해 upstream 자신이 `float64` 진실에서 떨어진
거리의 4배 안이면 결함이 아니다"* 입니다. **1.22× 는 4× 안입니다.** 그리고 AGREE 의 헤드라인
상대 허용치 1.186e-06 에 대해서도 이 모델의 `mps`-대-upstream 상대 오차는 **1.720e-07** 로,
한참 아래입니다.

**백엔드끼리의 비교는 그보다 더 촘촘합니다:**

```
max |mps - cpu|  =  2.384e-07      이 크기에서 float32 의 1 ulp = 3.305e-07
                                   즉 0.72 ulp — 원소당 최대 1 ulp
비트 동일         =  68 / 256
```

**그래서 이 문서는 "mps 가 cpu 와 일치한다" 고 쓰지 않습니다.** 쓰는 것은 위의 숫자이고,
남은 차이는 `docs/MPSFWD.md` §5 가 이미 원인을 잰 것과 같은 종류입니다 — Metal 의 초월함수
커널이 호스트 libm 과 마지막 비트에서 다르고, 부동소수 리덕션의 합산 순서가 다릅니다.
softmax 는 `exp` 하나와 리덕션 둘이므로 정확히 그 두 가지가 다 걸립니다.

**이것이 정밀도이지 결함이 아니라는 증거는 `float64` 답입니다**, 그리고 그 방향입니다:
`mps` 가 `cpu` 보다 진실에서 **더 멀리** 있지만(5.257e-07 대 4.632e-07) 그 차이는 6.25e-08 로
**1 ulp 의 5분의 1** 이고, 둘 다 upstream 자신이 만드는 오차와 같은 크기입니다. 커널이 틀렸다면
이 숫자는 `exp` 한 번의 마지막 비트가 아니라 그보다 훨씬 큰 곳에 있어야 합니다.

**golden 은 `cpu` 에서 돕니다 — 이 라운드의 음성 대조군이고, 움직이지 않아야 하며, 움직이지
않았습니다: 11405/11405, ops=301.** `_softmax` 의 CPU 산술을 스칼라 루프에서 candle op 로
바꿨는데도 그렇다는 것이 §3.1 의 "같은 연산" 주장의 실측 근거입니다.
<!-- DOCWATCH: count golden_cases_passed ge 11405 --> <!-- DOCWATCH: count golden_ops_covered ge 301 -->

---

## 5. GPU 에서 계산했는가 — op 별로, 추론이 아니라 런타임 단언으로

"호스트 되읽기로 폴백하는 것은 Metal 에서 도는 것과 같지 않다" 는 이 프로젝트가
`docs/MPS.md` 에서 세운 구분입니다. 그러므로 §2.1 의 16개 op 각각에 대해 답합니다.

**단언의 형태.** 세 가지가 함께여야 결론이 나옵니다:

1. **candle 의 Metal 백엔드에는 조용한 CPU 폴백이 없다** (`docs/MPS.md` §1.1, candle 소스에
   대한 사실). 못 하는 op 은 전부 `bail!` 한다. 그러므로 `mps` 텐서의 산술이 호스트에서
   일어날 수 있는 유일한 길은 **이 크레이트의 커널이 되읽는 것**이다.
2. **그 되읽기 집합은 `_C._shim_mps_host_readback_ops()` 가 런타임에 돌려주는 표**이고,
   `test_the_mps_readback_list_is_what_the_kernels_actually_do` 가 그것을 **소스에서 다시
   유도해** 로드된 아티팩트와 대조한다.
3. **forward 가 `mps` 에서 완료됐고 출력이 `mps:0` 이다.** 게이트는 커널이 돌기 *전에*
   거절하므로, 완료했다는 것은 경로 위의 어떤 op 도 그 표에 없었다는 뜻이다.

셋을 합치면: **표에 없는 op 은 되읽지 않고, 되읽지 않았으면 GPU 가 계산했다.**
`test_a_bert_encoder_forwards_on_mps_and_agrees_with_upstream` 이 그 표를 서브프로세스 안에서
읽어 단언합니다.

| op | GPU 인가 | 어떻게 아는가 |
|---|---|---|
| `_softmax` (×2) | **GPU** | 표에 없음(이 라운드에 빠짐) + `softmax_on_device` 에 마커·헬퍼 없음을 테스트가 소스에서 확인 |
| `matmul` (×4) | **GPU** | 표에 없음. `docs/MPSFWD.md` §5 가 이 shape 에서 CPU 와 **비트 동일** 을 측정했고, 이 파일의 permuted matmul 테스트도 동등으로 비교해 통과한다 |
| `addmm` (×13), `native_layer_norm` (×5), `add.Tensor` (×6), `mul.Scalar` (×2), `gelu` (×2), `tanh` | **GPU** | 전부 표에 없음 |
| `embedding` (×3) | **GPU** | 표에 없음 (`index_select` 는 candle 의 Metal 커널) |
| `view`(×30), `t`(×13), `transpose`(×10), `reshape`(×2), `contiguous`(×4), `slice`, `select` | **GPU 또는 레이아웃 전용** | 표에 없음. 이들은 대부분 바이트를 옮기지 않는 레이아웃 연산이고, `contiguous` 가 실제로 복사할 때는 candle 의 Metal 복사다 (`contiguous_blocked` 의 `is_metal` 분기) |
| `gather` | **경로에 없음 — 거절된다** | 표에 있음. §6 |

**정직하게 좁혀 두는 것:** 이 단언은 *"이 크레이트의 커널이 되읽지 않았다"* 를 말하지,
*"모든 산술이 GPU 코어에서 일어났다"* 를 candle 안쪽까지 추적해 말하지 않습니다. 후자는
1번 항목(candle 에 폴백이 없다)에 기대고 있고, 그것은 candle 소스에 대한 읽기이지 이
라운드가 실행한 측정이 아닙니다.

### 5.1 무력화하면 빨개지는가 — 여섯 번 다 해봤다

실패할 수 없는 검증은 검증이 아니므로, 이 라운드가 넣은 것을 하나씩 되돌려 다시 빌드하고
실제로 빨간 것을 봤습니다.

| 무력화한 것 | 잡혔나 | 어디서 |
|---|---|---|
| `softmax_default` 에 `read_flat` 을 되돌림 | **예** | `test_mpsattn.py` 4개 + `test_shim.py` 다수 |
| `is_matmul_striding_refusal` 의 Metal arm 을 `false` 로 | **예** | `test_mpsattn.py` 3개 |
| `host_built` → `from_vec(.., device)` | **예** | `test_a_float_literal_and_a_scalar_where_reach_the_device_at_all` |
| `where.ScalarOther` 의 `host_const` → `Tensor::full(.., device)` | **예** | 같은 테스트 |
| `_safe_softmax` 의 죽은 행 판정을 `max == -inf` 로 되돌림 | **예** | `test_safe_softmax_on_mps_answers_zero_for_an_all_minus_infinity_row` |
| 0-d / empty 가드 제거 | **golden 만** | ↓ |

**마지막 줄이 이 절에서 가장 중요합니다.** 0-d 와 empty 가드를 지웠을 때
`test_mpsattn.py` 는 **10개 전부 초록**이었고, 빨개진 것은 golden 9개뿐이었습니다. golden 은
게이트 안에 있으므로 회귀는 잡혔지만, **이 파일을 읽은 사람은 그 경계가 여기서 덮인다고
결론냈을 것이고 아니었습니다.** 그래서
`test_softmax_of_a_scalar_and_of_an_empty_tensor_still_answer` 를 추가했고, 같은 무력화를 다시
걸어 그것이 실제로 빨개지는 것을 확인했습니다.
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsattn.py test_softmax_of_a_scalar_and_of_an_empty_tensor_still_answer present -->

---

## 6. 남은 것 — 그리고 무엇을 재고 무엇을 재지 않았는지

* **`aten.gather.default` 가 BERT 의 임베딩에 남아 있다.** `token_type_ids` 를 주지 않으면
  `BertEmbeddings` 가 `torch.gather` 로 그것을 만들고, 그 op 은 여전히 거절됩니다. 위의 측정은
  전부 `token_type_ids` 를 **명시적으로 준** 것이고, 그 사실을 테스트 안에도 적어 두었습니다.
  **어텐션 블록이 아니라 임베딩입니다.** 옮기지 않은 이유는 candle 에 `gather` 가 있는데도
  단순 치환이 아니기 때문입니다 — 지금 커널은 인덱스를 호스트에서 훑으며
  `index N is out of bounds for dimension D with size S` 라는 **upstream 의 메시지를 재현**하고,
  device 위에서는 그 검사를 할 수 없습니다. 정확한 거절 메시지를 잃는 것은 이 라운드가
  혼자 결정할 일이 아니라고 판단해 남깁니다.
* **GPT-2 는 돌지 않는다.** 벽 두 개(§2.2 의 2·3)를 지나고 나면
  `torch.tensor: Metal error Failed to create metal resource: Buffer` 에서 멈춥니다 —
  Metal 이 **0바이트 버퍼를 만들지 못하는** 것이고, `docs/MPSFWD.md` §6 이 `use_cache=True`
  경로에서 이미 같은 벽을 기록하고 담당 범위 밖으로 둔 그것입니다. 커널이 아니라 할당의
  문제이고 `tensor.rs` 에 있습니다. **여기서도 손대지 않았습니다.**
* **`aten.gt.Scalar` 등 비교 op 은 `mps` 에서 거절이 아니라 벽이다.** `compare_common` 이
  정확한 비교를 위해 `f64` 로 넓히는데 Metal 에 `f64` 가 없습니다. 어텐션 경로에 없어서
  건드리지 않았고, 이 파일의 테스트는 그것을 피해 bool 리터럴로 마스크를 만듭니다.
  `docs/MPS.md` §7 의 *"`f32 -> f64` 가 Metal 에 없다"* 와 같은 항목입니다.
* **`_log_softmax` 는 여전히 거절된다.** `log_softmax_body` 는 `softmax_body` 와 구조가 달라
  (`narrow` 인자가 하나 더 있고 upstream 소스를 그대로 옮긴 것) 같은 치환이 되지 않고,
  **BERT 의 eager 어텐션 경로에 없습니다.** 손대지 않았습니다.
* **거절 목록의 나머지 85개.** 이 경로에 없으므로 건드리지 않았습니다.
* **한 모델, 한 입력, 한 dtype.** `float32`, 8 토큰, 2 레이어, shrunk config, `no_grad`.
  `docs/MPSFWD.md` §2.2 가 자기 측정에 붙인 경고를 그대로 물려받습니다 — **이 문서도
  "eager 어텐션은 `mps` 에서 돈다" 를 하나의 관측에서 일반화하고 있고**, §1 이 기록한 실수가
  정확히 그 모양이었습니다. 다른 어텐션 구현(`sdpa`, `flash`), 다른 길이, `use_cache=True`,
  reduced dtype 은 재지 않았습니다.
* **생성(generate) 은 확인하지 않았습니다.** forward 한 번입니다.

---

## 7. 게이트

| | 값 |
|---|---|
| `pytests/run.sh` | **976 ok**, 0 FAIL, EXIT=0 (기준선 966 + `test_mpsattn.py` 의 10개) |
| `cargo test` | **30 passed, 0 failed** |
| DOCWATCH | **PASS — 883/883** |
| golden `compare.py` | **11405/11405, ops=301 — 움직이지 않았다** |
| `MPS_HOST_READBACK_OPS` | 87 → **85** |
| 무력화 시 | §5.1 의 여섯 줄 |

마지막 줄이 앞의 줄들을 의미 있게 만듭니다.
