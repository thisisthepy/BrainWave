# 60개 구현 vs 실사용 op — 갭 측정

`fac5702` 로 `_aten_implemented()` 가 60개가 됐습니다. `docs/design/CORE_ATEN.md` §2 가 예전에 잰
"소형 Llama 가 부르는 op" 48개와 정확히 대조하고, `do_sample=True` 경로와 다른 아키텍처(GPT-2)로
같은 방법을 반복 적용한 결과입니다.

**결론 먼저: "다 됐다"가 아닙니다.** greedy 2층 Llama 기준으로도 8개가 비어 있고, 그중
`bmm.default` 는 어텐션 자체가 쓰는 op 이라 지금 이 shim 으로는 그 모델을 끝까지 못 돌립니다.

---

## 0. 측정 방법과 그 신뢰도

- **구현 목록 (60):** `rust/torch_c/src/aten.rs` 의 `IMPLEMENTED` 상수를 소스에서 직접 파싱.
  빌드하지 않았습니다 — 다른 두 에이전트가 같은 크레이트를 빌드 중이라는 지시를 따라 정적 소스만
  읽었습니다. `_aten_implemented()` 는 이 상수를 그대로 반환하는 함수(`aten.rs:270`)이므로 소스
  파싱 = 런타임 값입니다.
- **실사용 목록:** `TorchDispatchMode` 로 상류 torch(2.13.0) 위에서 직접 재측정했습니다
  (`/Volumes/macMini/caches/spike-venv/bin/python`, transformers 5.15.1). `docs/design/CORE_ATEN.md` §2 의
  48개 목록을 그대로 재현해 봤는데 **47/48 이 일치**했고, 빠진 하나(`aten.alias.default`)는 원본
  문서 목록에 있으므로 원본 48개 쪽을 그대로 신뢰 기준으로 썼습니다(제가 만든 모델의 랜덤 초기화 ·
  transformers 마이너 버전 차이로 짐작되는 1개 차이이며, 재현 스크립트가 갖고 있던 결함 — `randint`
  호출이 `TorchDispatchMode` 컨텍스트 밖에 있었던 것 — 은 고쳐서 `randint.low` 는 정확히 일치시켰습니다).
- overload 까지 맞춰서 비교했습니다 (`sum.default` ≠ `sum.dim_IntList`).
- 사용한 측정 스크립트는 `/tmp/measure_llama2.py`, `/tmp/measure_gpt2.py` (저장소 밖, 읽기전용
  규정을 지키기 위해 저장소에는 안 둠). 원시 출력은 `/tmp/llama_sample2.txt`, `/tmp/gpt2_greedy.txt`,
  `/tmp/gpt2_sample.txt`.

**중요한 구조적 발견 — 분해 테이블은 eager 실행에 적용되지 않습니다.**

> **정정 (조율 세션).** 이 절의 결론은 맞지만 **근거가 틀려서 아래로 대체합니다.** 초안은
> "`torch/_decomp` 가 벤더링되어 있지 않다"고 했는데, `torchnative/src/main/torch/_decomp/` 는 560K 로
> **벤더링되어 있고** `decompositions.py` 도 있습니다 — `rust/torch_c/src/lib.rs:4` 가 명시적으로
> 그렇게 적고 있습니다. 초안이 확인한 `torchnative/src/main/torch/` 는 이 저장소에 없는 경로입니다.

진짜 이유는 벤더링 여부가 아니라 **분해가 적용되는 시점**입니다. 분해표는 추적·컴파일
(`torch.compile` / `export`) 시점의 그래프 변환이지, eager 디스패치의 폴백이 아닙니다.
상류 torch 로 직접 확인했습니다:

```
eager 에서 관측된 op: ['aten.silu.default', 'aten.rsub.Scalar', 'aten.t.default']
silu 가 분해표에 있나: True
```

**`silu` 는 분해표에 있는데도 eager 에서 `aten.silu.default` 가 그대로 디스패치됩니다.**
표에 있다는 것이 "우리가 안 넣어도 된다"를 뜻하지 않습니다.

그러므로 `_aten_dispatch` 의 `match` 에 이름이 없는 op 은 즉시 `aten_not_implemented` 로 죽고,
아래 "missing" 목록은 전부 지금 당장 모델을 못 돌리는 하드 블로커입니다.

**이것은 `CORE_ATEN.md` §5 와 `DESIGN.md` §5 의 전제를 뒤집습니다** — "12 개는 벤더링한
`decompositions.py` 가 기계적으로 처리한다"는 서술은 eager 경로에서 성립하지 않습니다.
분해 층을 실제로 쓰려면 **우리가 직접 적용하는 배선**(디스패처가 미구현 op 을 만났을 때
분해표를 조회해 재귀 실행)이 필요하며, 그것은 아직 없습니다. 이 배선을 할지, 아니면 필요한
op 을 전부 직접 구현할지가 **미결 설계 판단**입니다.

---

## 1. 구현된 60개 (원문 그대로, `aten.rs:37-98`)

```
aten._local_scalar_dense.default   aten.eq.Scalar               aten.mul.Tensor
aten._to_copy.default              aten.eq.Tensor                aten.ne.Scalar
aten.add.Tensor                    aten.expand.default           aten.ne.Tensor
aten.any.default                   aten.fill_.Scalar             aten.new_ones.default
aten.any.dim                       aten.full.default             aten.ones.default
aten.arange.default                aten.index.Tensor             aten.pow.Scalar
aten.arange.start                  aten.is_floating_point.default aten.pow.Tensor_Scalar
aten.arange.start_step             aten.isin.Tensor_Tensor       aten.pow.Tensor_Tensor
aten.argmax.default                aten.lift_fresh.default       aten.randint.low
aten.bitwise_and.Scalar            aten.lt.Scalar                aten.reciprocal.default
aten.bitwise_and.Tensor            aten.lt.Tensor                aten.rsqrt.default
aten.bitwise_not.default           aten.masked_fill.Scalar       aten.select.int
aten.bitwise_or.Scalar             aten.max.default              aten.sin.default
aten.bitwise_or.Tensor             aten.max.dim                  aten.slice.Tensor
aten.cat.default                   aten.mean.default             aten.sub.Tensor
aten.clone.default                 aten.mean.dim                 aten.sum.default
aten.copy_.default                 aten.mm.default                aten.sum.dim_IntList
aten.cos.default                                                  aten.transpose.int
aten.cumsum.default                                               aten.unsqueeze.default
aten.detach.default                                               aten.view.default
aten.div.Tensor
aten.embedding.default
aten.empty.memory_format
```

(`IMPLEMENTED_AWAITING_GOLDEN` 에 13개가 더 있어서 실제 디스패치 가능한 op 은 73개입니다 — 이들은
동작은 하지만 `_aten_implemented()` 가 광고하지 않고 골든 비교도 안 걸려 있습니다: `add.Scalar`,
`any.dims`, `contiguous.default`, `div.Scalar`, `fill_.Tensor`, `masked_fill.Tensor`,
`matmul.default`, `max.other`, `mul.Scalar`, `randint.default`, `reshape.default`, `sub.Scalar`,
`zeros.default`.)

---

## 2. §0 이 지적한 "분해로도 안 풀리는 둘" — 확인

| op | 상태 |
|---|---|
| `aten.lift_fresh.default` | **구현됨** (`aten.rs:69`, 디스패치 `aten.rs:187`) |
| `aten.max.default` | **구현됨** (`aten.rs:73`, 디스패치 `aten.rs:233`) |

둘 다 해결됐습니다. `fac5702` 가 이 문서(§0)의 지적을 정확히 메웠습니다.

---

## 3. greedy 2층 Llama (`docs/design/CORE_ATEN.md` §2, 48개) 대조

### 필요한데 없음 — 8개 (진짜 남은 작업)

```
aten._scaled_dot_product_flash_attention_for_cpu.default
aten._unsafe_view.default
aten.alias.default
aten.bmm.default
aten.neg.default
aten.rsub.Scalar
aten.silu.default
aten.t.default
```

분해 테이블이 아직 안 걸려 있으므로(위 §0 참고), 이 8개는 전부 **지금 당장** shim 이 직접 처리해야
합니다. `bmm.default` 는 어텐션의 QK^T/PV 배치 행렬곱이라 **이게 없으면 이 아키텍처의 셀프어텐션이
아예 안 돕니다** — greedy 2층 Llama 는 지금 이 60개로는 forward pass 자체를 통과 못 합니다.
(`_scaled_dot_product_flash_attention_for_cpu.default` 도 어텐션 커널이라 같은 이유로 블로커입니다 —
`sdpa` 자체가 처리 안 되면 `bmm` 경로로 수동 구현하지 않는 한 어텐션이 없습니다.)

`silu.default` 는 SwiGLU MLP 의 활성화라 이것도 forward path 블로커입니다. `t.default` 는
`nn.Linear` 가 가중치를 전치할 때 부르는 것으로 짐작되고(§4 원본 목록에도 있음), `rsub.Scalar` 는
RoPE 나 마스크 생성 어딘가의 `scalar - tensor` 패턴, `neg.default` 는 RoPE 의 `rotate_half`,
`alias.default` 와 `_unsafe_view.default` 는 뷰 연산입니다 — 전부 forward pass 안에서 실제로
지나가는 경로입니다(§0 의 "분해로 안 풀림" 둘과 달리 이 8개는 이미 §4 의 decomposition 교집합에
있었으므로 "분해되면 없어도 된다"는 전제였는데, 그 분해 층이 없으니 지금은 그 전제가 성립하지
않습니다).

### 구현됐는데 이 시나리오는 안 씀 — 18개

```
aten.arange.start          aten.eq.Tensor              aten.mean.default
aten.arange.start_step     aten.fill_.Scalar           aten.ne.Scalar
aten.bitwise_and.Scalar    aten.index.Tensor           aten.pow.Scalar
aten.bitwise_or.Scalar     aten.is_floating_point.default  aten.pow.Tensor_Tensor
aten.copy_.default         aten.max.dim                aten.reciprocal.default
aten.detach.default        aten.div.Tensor(*)          aten.sum.dim_IntList
aten.empty.memory_format
```

(`div.Tensor` 는 §2 목록엔 없지만 do_sample 경로와 GPT-2 양쪽에서 쓰여서 실전에서는 쓰입니다 —
아래 §4, §5 참고. 나머지 17개는 `docs/bindings/TENSORBASE.md` 가 설명하는 **`TensorBase` 파이썬 메서드
표면**(`x.fill_()`, `x.copy_()`, `x.detach()`, `x[idx]`, `x.is_floating_point()`, `x.max(dim=)`,
`x.mean()`, `x == y`/`x != 2`, `x ** 2`, `x.reciprocal()`, `x.sum(dim=)`, `x.bitwise_and(2)` 등)
때문에 들어간 것으로 보입니다 — `aten.rs:32-36` 의 주석이 이 겹침을 명시적으로 설계로 인정합니다:
"`_aten_implemented()` 가 뜻하는 것은 커널이 있고 골든 비교가 걸려 있다는 것뿐, 어떤 파이썬 철자로
거기 도달하는지는 그 의미에 안 들어간다." 즉 이 op 들은 Llama forward 가 안 불러도 `TensorBase`
편의 메서드를 통해 golden 비교가 걸려 있어서 구현/카운트된 것으로 보이며, "왜 넣었는지" 질문의
답은 대체로 이것입니다.

---

## 4. `do_sample=True` — 추가로 필요한 것

측정: 같은 소형 Llama, `generate(max_new_tokens=4, do_sample=True, top_k=50, top_p=0.95,
temperature=0.9)`. 총 56개 op (greedy 48개 대비 +10, 그리고 `randint.low` 는 입력 생성에서도
찍히므로 이미 겹침).

### greedy 대비 새로 등장하는 10개

```
aten._softmax.default      aten.multinomial.default
aten.div.Tensor  (구현됨)   aten.scatter.src
aten.fill_.Tensor (awaiting_golden 에 있음, 디스패치는 됨)
aten.le.Scalar              aten.sort.default
aten.lt.Tensor  (구현됨)    aten.squeeze.dim
                             aten.topk.default
```

| 상태 | 목록 |
|---|---|
| 이미 구현됨 (`IMPLEMENTED`) | `div.Tensor`, `lt.Tensor` |
| 디스패치는 되지만 광고 안 됨 (`AWAITING_GOLDEN`) | `fill_.Tensor` |
| **완전히 없음 — 신규 작업** | `_softmax.default`, `le.Scalar`, `multinomial.default`, `scatter.src`, `sort.default`, `squeeze.dim`, `topk.default` |

**7개가 신규 블로커입니다.** 예상대로 `multinomial.default` (샘플링 자체)가 있고, top-k/top-p
필터링 체인이 `topk.default` → `sort.default`(top-p 정렬용) → `le.Scalar`(임계치 비교) →
`scatter.src`(마스크 되쓰기) → `_softmax.default`(logits→확률) → `squeeze.dim`(shape 정리) 로
이어집니다. 이 7개 중 하나라도 없으면 `do_sample=True` 경로는 greedy 8개 블로커를 다 메워도 여전히
못 돕니다.

---

## 5. 두 번째 아키텍처 — GPT-2 (재측정함)

Llama 는 RMSNorm · RoPE · SwiGLU · GQA-friendly 구조라 다른 아키텍처와 겹치지 않는 op 이 있을
수 있다는 우려에 따라, **같은 방법으로 GPT-2 도 쟀습니다** (n_embd=64, n_layer=2, n_head=2,
n_inner=128, vocab_size=100 — Llama 픽스처와 비슷한 규모).

- **greedy:** 43개 op
- **do_sample=True (top_k=50, top_p=0.95):** 52개 op

### GPT-2 greedy 가 Llama greedy 에 없는 op — 4개, 전부 미구현

```
aten.addmm.default        (nn.Linear + bias, GPT-2 의 Conv1D 가 이걸로 내려감)
aten.native_layer_norm.default  (LayerNorm — Llama 의 RMSNorm 과 다른 정규화)
aten.split.Tensor          (QKV projection 을 하나의 Linear 뒤에서 3분할)
aten.tanh.default          (GPT-2 의 활성화가 gelu_new/tanh 근사를 씀)
```

**넷 다 `_aten_implemented()` 에도 `AWAITING_GOLDEN` 에도 없습니다** — Llama 만 기준으로 8개를
다 메워도 GPT-2 는 여전히 4개가 더 필요합니다. 이는 질문이 우려한 대로 **"남은 작업이 모델마다
다르다"**는 것을 확인합니다: Llama 쪽 블로커(`bmm`, `silu`, `rsub.Scalar`, RoPE 관련 `neg`)는
GPT-2 에는 필요 없고(GPT-2 는 절대위치임베딩·MHA 를 다른 방식으로 구성), 대신 `addmm`/
`native_layer_norm`/`split`/`tanh` 가 새로 필요합니다.

### Llama greedy 에 있는데 GPT-2 는 안 쓰는 9개 (참고용 — 아키텍처 특이적임을 보여줌)

```
aten.alias.default   aten.cos.default   aten.mean.dim   aten.rsqrt.default
aten.bmm.default      aten.expand.default aten.neg.default aten.silu.default
                                          aten.sin.default
```

(RoPE 의 `sin`/`cos`/`neg`, RMSNorm 의 `mean.dim`+`rsqrt`, SwiGLU 의 `silu`, GQA 확장의
`expand`/`alias` — 전부 Llama 고유 구조와 대응됩니다.)

### GPT-2 sample 이 추가로 요구하는 것

GPT-2 sample(52개) − (GPT-2 greedy ∪ Llama sample) = **빈 집합**. 즉 GPT-2 의 샘플링 경로가 필요로
하는 op 은 이미 Llama sample 측정에서 다 나왔습니다(`_softmax`, `multinomial`, `topk`, `sort`,
`le.Scalar`, `scatter.src`, `squeeze.dim`, `fill_.Tensor`, `div.Tensor`, `lt.Tensor` 등) — 샘플링
로직 자체는 `generate()` 공통 코드이므로 아키텍처에 안 걸리는 것으로 보입니다. 다만 표본이 GPT-2
하나뿐이라 일반화하기엔 근거가 약합니다.

---

## 6. 종합 — 전체 4개 시나리오의 합집합

Llama-greedy(48) ∪ Llama-sample(56) ∪ GPT2-greedy(43) ∪ GPT2-sample(52) = **62개 유일 op**.

이 62개 중 `_aten_implemented()`(60개)에 없는 것 = **20개**:

```
aten._scaled_dot_product_flash_attention_for_cpu.default   aten.native_layer_norm.default
aten._softmax.default                                       aten.neg.default
aten._unsafe_view.default                                   aten.rsub.Scalar
aten.addmm.default                                           aten.scatter.src
aten.alias.default                                           aten.silu.default
aten.bmm.default                                              aten.sort.default
aten.le.Scalar                                                aten.split.Tensor
aten.multinomial.default                                      aten.squeeze.dim
                                                                aten.t.default
                                                                aten.tanh.default
                                                                aten.topk.default
```

`fill_.Tensor` 만 `AWAITING_GOLDEN` 이라 실제로는 디스패치되므로(골든 비교만 안 걸림) 이 20개에서
빠집니다 — 즉 **완전히 새 작업이 필요한 것은 20개**, "동작은 하는데 카운트가 안 되는 것"이 1개
(`fill_.Tensor`, 골든 케이스 빌더만 추가하면 됨) 입니다.

---

## 7. 우선순위 제안

측정만으로 판단할 수 있는 범위에서, "무엇을 먼저 넣으면 무엇이 열리는가" 기준입니다.

1. **`bmm.default` + `_scaled_dot_product_flash_attention_for_cpu.default`** — 이 둘이 없으면
   **Llama forward pass 자체가 안 돕니다** (self-attention). 8개 greedy 블로커 중 가장 앞에 둘
   이유는 다른 6개(`silu`, `t`, `rsub.Scalar`, `neg`, `alias`, `_unsafe_view`)를 다 메워도 이 둘이
   없으면 여전히 아무것도 안 돌기 때문입니다. `bmm.default` 는 §8 에서 확인했듯 커널 로직
   (`candle` 의 `broadcast_matmul`)이 `matmul_default` 에 이미 있고 op 라우팅 한 줄만 빠져
   보이므로 우선순위상 가장 저렴할 가능성이 있습니다. `_scaled_dot_product_flash_attention_for_cpu`
   는 별도 커널이 필요해 보이고 작업량이 더 큽니다.
2. **`silu.default`, `t.default`, `rsub.Scalar`, `neg.default`, `alias.default`,
   `_unsafe_view.default`** — 나머지 6개. 이걸 다 넣으면 **greedy 2층 Llama 가 통째로 돕니다**
   (다른 미측정 op 이 없다는 전제 하에 — 2층이라는 표본 크기의 한계는 §0/원본 문서가 이미 밝힘).
3. **GPT-2 를 같이 지원하려면** 위 8개 더하기 `addmm.default`, `native_layer_norm.default`,
   `split.Tensor`, `tanh.default` 4개가 추가로 필요합니다. Llama 만 목표라면 이 4개는 지금 우선순위
   밖입니다.
4. **`do_sample=True` 를 열려면** `multinomial.default`(샘플링 자체가 이것 없이는 원천적으로 불가),
   `topk.default`/`sort.default`/`le.Scalar`/`scatter.src`(top-k/top-p 체인), `_softmax.default`,
   `squeeze.dim` 7개가 필요합니다. **greedy 가 도는 것이 이 경로의 전제조건**이므로(§3 의 8개가
   `do_sample=True` 에도 그대로 필요) 순서상 1~2 다음입니다.
5. **`fill_.Tensor`** 는 이미 동작하므로 우선순위가 아니라 잡무입니다 — golden 케이스 빌더 하나
   추가하고 `IMPLEMENTED` 로 옮기면 끝입니다 (`aten.rs:100-114` 의 주석이 이미 그렇게 적어 뒀습니다).

**분해 테이블 배선은 이 우선순위와 별개로 봐야 합니다.** §0 에서 지적했듯 아직 안 걸려 있고, 지금
막힌 8+7+4개는 모두 "분해되면 안 넣어도 되는" 종류가 아니라 forward path 가 직접 부르는 원시
op(들)입니다 — 분해 테이블을 지금 배선해도 이 갭은 안 줄어듭니다. (반대로 §0 표의 "분해 가능"
12개는 이미 대부분 직접 구현되어 있어 실질적으로 막혀 있지 않습니다.)

---

## 8. 한계 · 모르는 것

- **표본이 2개 아키텍처, 각 2층뿐입니다.** 레이어 수를 늘리거나 다른 모델군(Mistral·Qwen·Gemma —
  GQA·sliding window·다른 정규화)을 추가로 재면 더 나올 수 있습니다. 시간 제약으로 셋째 아키텍처는
  못 쟀습니다.
- **`bmm.default` 등이 왜 `IMPLEMENTED` 에서 빠졌는지는 모릅니다** — 커밋 로그나 이슈 트래커를
  안 봤습니다(코드만 읽으라는 지시 범위 안에서). 의도적 보류인지 누락인지는 원 작성자에게 확인이
  필요합니다.
- **`bmm.default` 가 없어도 `matmul.default`(AWAITING_GOLDEN)로 우회되지 않습니다.** 소스만
  읽어 확인했습니다(안 빌드함) — `matmul_default`(`aten.rs:1234`)는 candle 의 `broadcast_matmul`
  로 rank≥2(배치 포함)를 이미 처리하지만, 디스패치는 op 문자열 완전 일치로만 라우팅됩니다
  (`aten.rs` 헤더 주석: "exactly one entrance"). `aten.bmm.default` 콜은 `aten.matmul.default` 로
  안 새고 그대로 `match` 의 `other => Err(aten_not_implemented(...))` 로 떨어집니다. 즉 커널 로직은
  이미 있는데(배치 행렬곱을 이미 처리함) **op 이름 라우팅 한 줄만 없는 상태**로 보입니다 — 실행
  검증은 못 했지만, 이 부분은 다른 8개보다 작업량이 훨씬 작을 가능성이 있습니다.
- **do_sample=True 세부 하이퍼파라미터**(top_k, top_p, temperature 값)를 바꾸면 경로가 달라질 수
  있습니다(예: top_p=1.0 이면 `sort`/`le.Scalar` 체인을 안 탈 수도 있음) — 여기서는 top_k=50,
  top_p=0.95 로 한 조합만 쟀습니다.
