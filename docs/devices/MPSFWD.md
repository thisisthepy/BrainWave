# `mps` — SmolLM2-135M forwards, and what was actually in the way

**결론부터: 돈다. 그리고 막고 있던 것은 `_softmax` 가 아니었습니다.**

| 질문 | 답 |
|---|---|
| SmolLM2-135M 이 `mps` 에서 forward 하는가 | **한다.** 393,216 개 로짓, `device=mps:0` |
| 값이 `cpu` 와 맞는가 | **원소별로 5.33e-06 (상대)** 안에서 맞는다. **비트 동일은 아니다** — §5 가 그 차이가 어디서 오는지 잰다. `argmax` 는 같다 |
| 거절 목록(`MPS_HOST_READBACK_OPS`)에서 몇 개가 이 경로에 있었나 | **3개.** `pow.Tensor_Scalar`, `neg.default`, `cumsum.default`. `_softmax` 는 **경로에 없다** |
| 그 3개는 candle 의 공백이었나, 우리 되읽기였나 | **셋 다 우리 것.** candle 이 못 하는 것은 하나도 없었다 (§3) |
| 목록은 어떻게 됐나 | **71 → 67.** 넷(`prims.neg` 포함)이 **다시 쓰여서** 빠졌다. 게이트를 넓힌 곳은 없다 |
| 거절 아닌 벽은 몇 개였나 | **다섯.** f64 상수, `U8 -> F64` 위닝 두 곳, `cpu_fwd` 만 있는 `CustomOp1` 셋 (§4) |

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs MPS_HOST_READBACK_OPS present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_mpsfwd.py test_the_ops_that_left_the_refusal_list_no_longer_read_back present -->

---

## 1. 표 — 경로 위의 op 하나하나

**"거절되는 op 중 무엇이 트랜스포머 경로에 있는가"** 를 추측하지 않고 쟀습니다. 게이트를
`eprintln!` 후 통과시키도록 임시로 바꿔 빌드하고(§2.1), SmolLM2-135M 을 `mps` 에서 돌려
실제로 걸린 이름만 모았습니다.

| op | 왜 거절됐나 | candle 의 공백인가 | 옮겼나 | 어떻게 |
|---|---|---|---|---|
| `aten.pow.Tensor_Scalar` | `side_from_tensor` → `Vec<f64>` → Rust `powf` → `from_vec` | 아니오 (`powf` 는 Metal 에 `powf_f32` 가 있고, `F64` 는 CPU 에서 같은 `f64::powf`) | **예** | `widen_f64` + candle `powf`; 정수 지수는 device 위 제곱-반복 |
| `aten.neg.default` / `prims.neg.default` | 정수 경로가 `to_vec1::<i64>()` + `wrapping_neg` | 아니오 (candle `neg` 의 정수 arm 은 `todo!()` 라 못 쓴다는 것이 원래 이유였고, `0 - x` 는 된다) | **예** | `int64` 에서 `0 - x`, 같은 wrap |
| `aten.cumsum.default` | `to_vec1` + 스칼라 루프 | 아니오 (candle `cumsum` 은 삼각행렬 matmul 이라 `I64` 가 안 되는 것이 원래 이유) | **예** | `n-1` 개 `narrow`/`add`, **루프와 같은 순서** |
| `aten._softmax.default` | (이 라운드에서는 거절) | — | **아니오, 그럴 필요가 없다** | SmolLM2 는 이 op 을 부르지 않는다 — §3.1. **그러나 eager 어텐션은 부른다: docs/devices/MPSATTN.md** |

> **이 정정 자체가 한 걸음 더 갔습니다 (docs/devices/MPSATTN.md §1).** 아래에서 얻은 것은
> *"SmolLM2 는 `_softmax` 를 부르지 않는다"* 인데, 이 문서는 그것을 *"어텐션 블록은
> `_softmax` 를 지나지 않는다"* 로 적었습니다. 둘은 다른 문장이고 두 번째를 뒷받침하는
> 측정은 없었습니다 — `attn_implementation="eager"` 인 BERT 는 레이어마다 두 번 지납니다.
> 이 문서 자신의 §2.2("한 입력, 한 아키텍처")가 경고해 둔 바로 그 자리입니다.

`_softmax` 가 이 표에서 빠지는 것이 이 라운드의 첫 번째 정정입니다. `docs/devices/MPS.md` §1.3 은
*"`mps` 위의 트랜스포머는 어텐션 블록마다 그곳을 지난다"* 고 적었는데, **지나지 않습니다.**
`docs/numerics/SEQLEN.md` §7.4 가 이미 반대로 측정해 두었고(`softmax_body` 는 프로파일에 0 회,
`sdpa_flash_cpu` 는 매번), 그 사실이 이 라운드에서 그대로 확인됐습니다. 두 문서가 같은 저장소
안에서 서로 다른 말을 하고 있었고, **경로를 실제로 밟아 본 쪽이 맞았습니다.**

거절 목록의 나머지 67개는 이 경로에 없으므로 **이 라운드에서 건드리지 않았습니다.** 거절은
그대로입니다.

---

## 2. 측정 방법 — 그리고 그것이 왜 답을 정하지 않는가

### 2.1 게이트를 log-and-allow 로 바꾼 일회용 빌드

```rust
if std::env::var("MPSFWD_SURVEY").is_ok() {
    eprintln!("MPSFWD-REFUSED {op}");
    return Ok(());
}
```

`device.rs` 에 이 세 줄을 넣고 빌드해 forward 를 돌린 다음 **되돌렸습니다.** 최종 트리에는
없습니다(있으면 그것이 바로 이 프로젝트가 한 라운드를 들여 닫은 조용한 CPU 폴백입니다).

이 방법이 아니면 알 수 없는 것이 있습니다: 게이트가 켜져 있으면 **첫 번째 거절에서 멈추므로**
그 뒤에 무엇이 있는지 보이지 않습니다. 실제로 이 조사는 "첫 실패를 고치고 다시 돌린다" 로
시작했는데, 세 번째 왕복에서 `use_cache` 유무에 따라 **경로 자체가 갈린다**는 것이 드러났습니다
(캐시 경로는 `cumsum` 대신 0-크기 버퍼에서 죽습니다 — §6). 한 번에 전부 보는 방법이 필요했던
이유입니다.

### 2.2 이 측정이 못 보는 것

- **한 입력, 한 아키텍처.** 8 토큰 프리필, `float32`, `use_cache=False`. 다른 길이·dtype·
  `use_cache=True` 는 다른 op 을 부를 수 있고, 실제로 §6 이 그 하나를 붙잡았습니다.
- **`torch.tensor` 같은 팩토리는 게이트를 지나지 않습니다.** 거절 목록은 커널의 되읽기에
  대한 것이지 Metal 이 못 하는 모든 것에 대한 것이 아니고, §4 의 벽 다섯 개는 전부
  **거절 목록 밖**에서 나왔습니다. "거절되는 op 을 다 옮기면 돈다" 는 명제는 **거짓**이며,
  이 라운드가 그것을 실물로 보여줍니다.

---

## 3. 세 개 다 우리 되읽기였다 — candle 은 아무것도 막지 않았다

`docs/devices/MPS.md` §1.1 이 세운 명제("되읽는 것은 candle 이 아니라 이 크레이트다")가 이 경로에서
**3/3 으로 유지**됩니다. 셋 다 커널을 다시 써서 device 위에 남겼고, 셋 다 candle 의 Metal
백엔드에 필요한 커널이 이미 있었습니다.

### 3.1 `pow.Tensor_Scalar` — RMSNorm

```
hidden_states.pow(2).mean(-1, keepdim=True)      LlamaRMSNorm, 레이어마다 두 번
```

부동소수 경로는 `widen_f64` 후 candle `powf` 입니다. **CPU 값이 움직이지 않는 이유는 tolerance 가
아니라 같은 호출이기 때문입니다** — candle 의 `F64` arm 은 `unary_map(|v| v.powf(e))`, 즉
호스트 루프가 부르던 바로 그 `f64::powf` 입니다.

Metal 에는 `f64` 가 없으므로 **제곱이 아닌 지수는 `mps` 에서 시끄럽게 실패합니다.**
`f32` 로 낮춰 계산하면 CPU 가 내지 않았을 숫자를 `mps` 라벨로 돌려주게 되는데, 그것이 이
프로젝트가 문 앞에서 거절하기로 한 바로 그 형태입니다. RMSNorm 이 도는 이유는 지수 2.0 이
`pow_square_fast_path` 의 곱셈이라서이고, 그 곱셈은 어느 장치에서나 정확히 반올림된 곱입니다.
`test_pow_with_a_general_exponent_on_mps_refuses_rather_than_lowering_precision` 이 그 선택을
잡아 둡니다 — 나중에 누가 "Metal 에서는 f32 로" 를 넣으면 테스트를 고쳐야 합니다.

정수 경로는 `int64` 제곱-반복이고 `wrapping_pow` 와 같은 곱셈열입니다.

### 3.2 `neg.default` — `rotate_half`

```
torch.cat((-x2, x1), dim=-1)                     레이어마다 두 번 (q, k)
```

`0 - x` 를 `int64` 로 합니다. `wrapping_neg` 와 같은 wrap 입니다: `0i64.wrapping_sub(i64::MIN)`
은 `i64::MIN` 이고, 좁은 폭의 wrap 은 예전처럼 뒤의 narrowing 이 냅니다. 테스트가 `int64` 최솟값을
따로 넣습니다.

원래 되읽던 이유는 "candle 의 정수 `neg` 는 `todo!()` 라 패닉한다" 였는데, **그것은 `neg` 에
대한 사실이지 뺄셈에 대한 사실이 아니었습니다.**

### 3.3 `cumsum.default` — 어텐션 마스크

`n-1` 개의 `narrow`/`add` 쌍이고, `docs/architectures/VOICE.md` 의 `cumprod` 와 같은 모양입니다. 순서가
**호스트 루프의 순서 그대로**(`running + x[i]`)라 부동소수 결과가 근사가 아니라 비트 동일합니다.
병렬 스캔이었다면 재결합이므로 논증이 필요했을 텐데, 그러지 않았습니다. 대가는 한 번의 패스가
아니라 `n` 번의 커널 실행입니다.

부동소수 `cumsum` 은 `f64` 로 누산하므로 `mps` 에서는 `widen_f64` 에서 시끄럽게 실패합니다.
정수 `cumsum`(마스크가 부르는 것)은 돕니다.

---

## 4. 거절 목록 밖의 벽 다섯 — 여기가 이 라운드의 대부분이었다

세 커널을 옮긴 뒤에도 forward 는 **다섯 번 더** 멈췄습니다. 전부 거절 목록과 무관하고, 전부
"Metal 이 못 하는 것을 우리가 요구하고 있었다" 입니다.

| # | 증상 | 원인 | 고친 방법 |
|---|---|---|---|
| 1 | `aten.mul.Scalar: candle: unsupported const-set f64` (rotary) | `Tensor::full(f64, (), device)` 가 **장치에게** f64 상수를 만들라고 시킨다. Metal 에 f64 는 없다 | `host_const` — 호스트에서 만들고 dtype 변환까지 호스트에서 한 뒤 올린다. 되읽기가 아니라 업로드다 |
| 2 | `aten.all.default: Metal contiguous to_dtype U8 F64 not implemented` | `any_from` 이 마스크를 f64 로 넓혀 `!= 0` 을 한다 | 정수·bool 은 자기 dtype 에서 비교. `x != 0` 은 폭에 의존하지 않는다 |
| 3 | 같은 메시지, `_to_copy` 의 bool arm (`bool_mask.to("mps")`) | 같은 위닝 | 같은 수정 |
| 4 | `no metal implementation for torch._C shim: transposed copy` | `cpu_fwd` 만 있는 `CustomOp1` | Metal 에서는 candle 의 `contiguous`. **구성상 비트 동일** — 모든 출력 원소가 입력 한 원소의 복사라 재결합이 없다 |
| 5 | `... scale + causal mask`, `... amax` | 같은 형태의 `CustomOp1` 둘 | 각각 그것이 **측정 대상으로 삼았던 원래 스펠링**으로: `affine(scale,0)` + `-inf` 마스크 `broadcast_add`, 그리고 candle `max_keepdim`. 둘 다 원 문서가 비트 동일이라고 논증해 둔 쌍이다 |

4·5 는 **성능을 위해 도입된 커널이 이식성을 잃은 자리**입니다. 셋 다 CPU 에서 측정된 이득이고
(`docs/numerics/SEQLEN.md` §7, §8.3, §8.12), Metal 위에서 그 이득은 존재하지 않으므로 폴백은 잃는 것이
없습니다. 중요한 것은 **폴백이 근사가 아니라는 것**입니다 — 세 경우 모두 원 문서가 "이것은
tolerance 가 아니라 논증" 이라고 적어 둔 그 상대편으로 돌아갑니다.

`is_metal` 가드를 `if true` 로 무력화하면 §7 대로 테스트 둘이 빨개집니다.

---

## 5. 값 — `cpu` 와 얼마나 맞는가, 그리고 남은 차이는 무엇인가

```
SmolLM2-135M  float32  8 토큰 프리필  use_cache=False
393,216 로짓   비트 동일 13,354 (3.40%)
max |cpu - mps| = 1.907e-04     max |logit| = 35.78     상대 5.33e-06
argmax(cpu) = argmax(mps) = 2034
```

**비트 동일이 아닙니다.** 그것을 "맞다" 로 적지 않기 위해, 남은 차이가 어디서 오는지 같은
아티팩트로 재었습니다.

| | `cpu` 와 비트 동일 | 최대 차이 |
|---|---|---|
| `matmul` (576×576, f32) | **331,776 / 331,776** | 0 |
| `mul.Tensor` (동상동) | **331,776 / 331,776** | 0 |
| `exp` (4096, f32) | 1,638 / 4,096 | 5.7e-06 |
| `sigmoid` | 2,456 / 4,096 | 1.2e-07 |
| `silu` | 2,273 / 4,096 | 4.8e-07 |
| `sum` (4096 → 스칼라) | 0 / 1 | 5.0e-05 |
| `mean` | 0 / 1 | 1.2e-08 |

**GEMM 이 아닙니다** — 그쪽이 의심스러웠고, 실제로 재 보니 비트 동일이었습니다. 남은 것은
(a) Metal 의 초월함수 커널이 호스트 libm 과 마지막 비트에서 다르고, (b) 부동소수 **리덕션의
합산 순서**가 다르다는 것입니다. 둘 다 이 크레이트가 고를 수 있는 것이 아니고, 둘 다
"틀린 장치에서 계산됐다" 와는 다른 종류입니다 — GPU 가 계산했고, GPU 의 `expf` 로 계산했습니다.

그래서 이 문서는 **"로짓이 cpu 와 일치한다" 고 쓰지 않습니다.** 쓰는 것은 위의 숫자이고,
그 숫자의 출처는 위의 표입니다.

`pytests/test_mpsfwd.py` 안의 검증은 두 종류로 나뉘고 그 경계가 위 표와 같습니다:
`neg`·`pow(2)`·`cumsum`·`all` 은 **동등(equality)** 으로, SDPA 와 트랜스포머 블록은
**1e-5 상대** 로 비교합니다. 전자에 tolerance 를 쓰면 틀린 커널을 숨기고, 후자에 동등을 쓰면
GPU 의 `exp` 하나에 빨개집니다.

---

## 6. 아직 막혀 있는 것

* **`use_cache=True` 의 첫 스텝.** `transformers` 의 `DynamicCache` 가 빈 텐서를 만들고
  (`cache_utils.py::lazy_initialization`), Metal 은 **0 바이트 버퍼를 만들지 못합니다**:
  `torch.tensor: candle: Metal error Failed to create metal resource: Buffer`. 위 측정은
  `use_cache=False` 입니다. 이것은 커널이 아니라 할당의 문제이고 `tensor.rs` 에 있습니다 —
  이 라운드의 담당 범위 밖이라 손대지 않았습니다.
* **`f64` 가 필요한 모든 것.** 제곱 아닌 `pow`, 부동소수 `cumsum`, `_softmax` 의 `read_flat`
  경로. `docs/devices/MPS.md` §7 이 남긴 그대로이고, candle 쪽 문제입니다.
* **거절 목록의 67개.** 이 경로에 없으므로 건드리지 않았습니다. 옮길 수 있는 것과 없는 것이
  섞여 있고(`nonzero` 는 인덱스 산술이라 어렵고, `abs`/`bitwise_*` 는 `neg` 와 같은 모양이라
  쉽습니다), **하나씩 다시 쓰는 것 말고 목록을 줄이는 방법은 없습니다.**
* **생성(generate) 은 확인하지 않았습니다.** forward 한 번입니다. `argmax` 가 일치한다는 것은
  첫 토큰에 대한 사실이지 20 토큰에 대한 사실이 아닙니다.

---

## 7. 게이트

| | 값 |
|---|---|
| `pytests/run.sh` | **677 ok**, 0 FAIL, EXIT=0 |
| DOCWATCH | **PASS — 616/616** |
| golden `compare.py` | **10039/10039, ops=270 — 움직이지 않았다** <!-- DOCWATCH: count golden_cases_passed ge 10039 --> <!-- DOCWATCH: count golden_ops_covered ge 270 --> |
| `aarch64-linux-android` | EXIT=0 (`scripts/device_android.sh build`) |
| `aarch64-apple-ios-sim` | EXIT=0 (`PYO3_CONFIG_FILE` 레시피) |
| `MPS_HOST_READBACK_OPS` | 71 → **67** |

**무력화하면 빨개지는가 — 두 번 실제로 해봤습니다.**

| 무력화 | 결과 |
|---|---|
| `scale_and_causal_mask_anywhere` 의 `is_metal` 가드를 `if true` 로 | `test_sdpa_on_mps_agrees_with_cpu`, `test_a_transformer_block_forwards_on_mps_and_agrees_with_cpu` **FAIL** |
| `neg_default` 의 device 경로를 옛 `to_vec1` 로 되돌리고 **목록에는 넣지 않음** (= 조용한 CPU 계산의 모양) | `test_the_ops_that_left_the_refusal_list_no_longer_read_back` **FAIL**, 그리고 `test_shim.py` 의 파생 테스트 둘 **FAIL** |

두 번째가 이 라운드에서 가장 중요한 줄입니다. 되돌린 상태에서도 **`mps` 위의 값 테스트는 전부
초록이었습니다** — 값이 맞기 때문입니다. 빨개진 것은 소스에 대한 주장을 하는 테스트뿐이고,
그것이 `docs/devices/MPS.md` 가 파생 검사를 만든 이유이자, 이 라운드가 그 검사에 이름 네 개짜리 조항을
하나 더 붙인 이유입니다.
