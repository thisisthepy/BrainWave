# Core ATen Operator Set 분석

torchnative 프로젝트의 요구 사항을 검증하기 위해 PyTorch Core ATen 공식 스펙과 실제 모델의 aten op 사용 현황을 대조한 문서입니다.

> **검증 결과 (조율 세션).** 초안은 서브 에이전트가 작성했고, **핵심 결론이 틀려 아래 §0 으로
> 대체합니다.** 초안은 Core ATen 목록을 얻지 못한 채 "분해 테이블에 없음 → Core ATen 원시" 로
> 추론했는데, 그 추론은 성립하지 않습니다. 분해 테이블에 없다는 것은 core 라는 뜻이 아니라
> **그 표에서 다루지 않는다**는 뜻입니다. 아래 §0 은 torch 가 직접 표시하는 태그로 다시 잰 것입니다.
> §1~§4 의 op 수집 자체는 유효하므로 남겨둡니다.

---

## 0. 실측 (권위 있는 출처로 재확인)

**Core ATen 목록은 웹 없이 로컬에서 얻어집니다.** PyTorch 가 op 마다 태그를 달아 두었습니다.

```python
torch.Tag.core in torch.ops.aten.<op>.<overload>.tags
```

| | 개수 |
|---|---|
| **Core ATen (`torch.Tag.core`)** | **189** |
| 우리 모델이 실제로 부른 aten op | **47** |
| └ Core ATen 안 | **33** |
| └ Core ATen 밖 | **14** |
| &nbsp;&nbsp;&nbsp;└ 분해 테이블로 풀림 | 12 |
| &nbsp;&nbsp;&nbsp;└ **분해로도 안 풀림** | **2** |

분해로도 안 풀리는 둘:

```
aten.lift_fresh.default
aten.max.default
```
<!-- DOCWATCH: op-implemented aten.lift_fresh.default -->
<!-- DOCWATCH: op-implemented aten.max.default -->
(둘 다 shim 은 이미 구현하고 있습니다 — 이 절이 말하는 "미분류"는 Core ATen/분해표 어느
분류표에도 안 걸린다는 뜻이지, shim 에 커널이 없다는 뜻이 아닙니다.)

**그러므로 "Core ATen 범위 밖의 op 은 없다" 는 초안의 결론은 틀렸습니다.** 14 개가 밖에 있고,
그중 2 개는 분해 테이블도 다루지 않습니다. 둘 다 사소해 보이지만 — `lift_fresh` 는 텐서 생성
표식이고 `max.default` 는 전체 최댓값 축약 — **shim 이 별도로 처리해야 할 항목**이고, 그 존재
여부가 "Core ATen 만 구현하면 되는가" 라는 계획의 전제를 바꿉니다.

### 이것이 §5 의 3 층 구조에 주는 것

DESIGN.md §5 는 shim 을 (1) Core ATen 구현 (2) 분해 테이블 벤더링 (3) 핫 op 손 최적화 로
정리했습니다. **여기에 네 번째가 필요합니다 — 어느 쪽에도 안 걸리는 잔여 op.** 이번 측정에서는
2 개이지만, 모델을 늘리면 늘어날 수 있습니다. 그 잔여를 어떻게 다룰지(직접 구현할지, 상류에
분해를 추가할지)가 정해져야 합니다.

**주의:** 이 측정은 2 층 Llama · 양자화 없음 · `do_sample=False` 기준입니다. 더 큰 모델과
다른 경로는 더 부를 수 있습니다.

---

## 1. Core ATen 공식 정의

**출처:** PyTorch 공식 문서 — [Core ATen Operator Set Definition](https://docs.pytorch.org/executorch/stable/ir-ops-set-definition.html)

**정의:** PyTorch 가 공개 저장소와 유명 오픈소스 모델을 조사하여 실제로 많이 쓰이는 aten op 을 추린 것. 목적은 명시적으로 "백엔드와 컴파일러가 처리해야 할 연산자 수를 줄이는 것".

**Core ATen 크기: 189** (§0 에서 `torch.Tag.core` 로 실측. 초안은 "미확인" 으로 남겼으나 웹 문서
없이 로컬에서 얻어집니다.)

---

## 2. 우리 모델의 실제 ATen Op 사용

### 실험 환경

- **모델:** 소형 Llama (hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128, vocab_size=100)
- **환경:** PyTorch 2.13.0, transformers 5.15.1
- **수집 방법:** `torch.utils._python_dispatch.TorchDispatchMode` 로 순전파 + `generate(max_new_tokens=4, do_sample=False)` 중 호출된 연산자 수집

### 실제 사용 Op 목록 (48개)

```
aten._local_scalar_dense.default
aten._scaled_dot_product_flash_attention_for_cpu.default
aten._to_copy.default
aten._unsafe_view.default
aten.add.Tensor
aten.alias.default
aten.any.default
aten.any.dim
aten.arange.default
aten.argmax.default
aten.bitwise_and.Tensor
aten.bitwise_not.default
aten.bitwise_or.Tensor
aten.bmm.default
aten.cat.default
aten.clone.default
aten.cos.default
aten.cumsum.default
aten.embedding.default
aten.eq.Scalar
aten.expand.default
aten.full.default
aten.isin.Tensor_Tensor
aten.lift_fresh.default
aten.lt.Scalar
aten.masked_fill.Scalar
aten.max.default
aten.mean.dim
aten.mm.default
aten.mul.Tensor
aten.ne.Tensor
aten.neg.default
aten.new_ones.default
aten.ones.default
aten.pow.Tensor_Scalar
aten.randint.low
aten.rsqrt.default
aten.rsub.Scalar
aten.select.int
aten.silu.default
aten.sin.default
aten.slice.Tensor
aten.sub.Tensor
aten.sum.default
aten.t.default
aten.transpose.int
aten.unsqueeze.default
aten.view.default
```

**총 개수:** 48개

---

## 3. PyTorch Decomposition Table (분해 규칙)

### 소개

PyTorch 가 제공하는 `torch._decomp.core_aten_decompositions()` 함수는 Core ATen 이 아닌 연산자를 Core ATen 원시 연산자로 분해하는 규칙표입니다. 이 규칙들은 파이썬 소스(`torch/_decomp/decompositions.py`)입니다.

### 분해 가능한 총 연산자 수

**940개** — `torch._decomp.core_aten_decompositions()` 의 반환 사전에 포함된 모든 op

### 우리 모델에서 분해 가능한 Op (교집합)

우리 모델의 48개 op 중에서 분해 테이블에 있는 것:

```
aten._scaled_dot_product_flash_attention_for_cpu.default
aten._unsafe_view.default
aten.arange.default
aten.isin.Tensor_Tensor
aten.masked_fill.Scalar
aten.new_ones.default
aten.ones.default
aten.rsub.Scalar
aten.silu.default
aten.sum.default
aten.t.default
aten.transpose.int
```

**개수:** 12개 (전체 48개 중 25%)

**의미:** 이들 op 은 더 기본적인 Core ATen 원시 연산자로 자동 분해될 수 있습니다.

---

## 4. Core ATen 원시 연산자 (우리 모델이 사용)

우리 모델의 48개 op 중에서 분해 테이블에 없는 것 = Core ATen 원시 연산자:

```
aten._local_scalar_dense.default
aten._to_copy.default
aten.add.Tensor
aten.alias.default
aten.any.default
aten.any.dim
aten.argmax.default
aten.bitwise_and.Tensor
aten.bitwise_not.default
aten.bitwise_or.Tensor
aten.bmm.default
aten.cat.default
aten.clone.default
aten.cos.default
aten.cumsum.default
aten.embedding.default
aten.eq.Scalar
aten.expand.default
aten.full.default
aten.lift_fresh.default
aten.lt.Scalar
aten.max.default
aten.mean.dim
aten.mm.default
aten.mul.Tensor
aten.ne.Tensor
aten.neg.default
aten.pow.Tensor_Scalar
aten.randint.low
aten.rsqrt.default
aten.select.int
aten.sin.default
aten.slice.Tensor
aten.sub.Tensor
aten.unsqueeze.default
aten.view.default
```

**개수:** 36개 (전체 48개 중 75%)

**의미:** 이들은 **분해 테이블이 다루지 않는** op 입니다. 초안은 여기서 "분해 테이블에 없음 →
Core ATen 원시" 로 넘어갔는데, **§0 이 그 추론을 무효로 판정했습니다** — 표에 없다는 것은 core
라는 뜻이 아니라 그 표에서 다루지 않는다는 뜻입니다. `torch.Tag.core` 로 실측하면 우리 모델이
부른 것 중 Core ATen 안은 33 개이고 밖이 14 개입니다. 목록 자체는 유효하니 남기되, **이름을
근거로 쓰지 마십시오.**

---

## 5. 요약: 대조 결과

| 항목 | 개수 | 비율 |
|---|---|---|
| 우리 모델이 사용하는 total aten op | **48** | 100% |
| 분해 테이블이 다루지 않는 op (직접 구현 필요) | **36** | 75.0% |
| 분해 가능한 연산자 (자동 분해) | **12** | 25.0% |

이 표의 36 은 "48 − 12" 이지 Core ATen 태그를 센 것이 아닙니다. **태그로 실측한 분류는
§0 의 표입니다** (Core ATen 안 33 / 밖 14 / 그중 분해로도 안 풀림 2). 두 표는 다른 것을
세고 있으므로 섞어 읽으면 안 됩니다.

### 결론

- **§0 의 재측정에 따르면, 우리 모델이 부른 op 중 14 개는 Core ATen 범위 밖입니다.**
  그중 12 개는 분해 테이블로 풀리지만, 2 개(`lift_fresh.default`, `max.default`)는 그것도 다루지 않습니다.
  따라서 초안의 "**모든 Op 이 Core ATen 또는 분해 테이블에 포함**" 이라는 결론은 틀렸습니다.
  (§0 37-40줄 참고)
- **분해 테이블이 다루지 않는 36 개** 가 shim 의 하드 플로어입니다. 이것을 "Core ATen 원시"
  라고 부르지 마십시오 — 위에 적은 이유로 그 이름은 측정된 것이 아닙니다.
- ~~**12개의 분해 가능한 op** 는 벤더링한 `torch/_decomp/decompositions.py` 가 기계적으로 처리합니다.~~
  **틀렸습니다 — `docs/kernels/GAP.md` §0 이 실측으로 뒤집었습니다.** 분해표는 추적·컴파일 시점의 그래프
  변환이지 eager 디스패치의 폴백이 아닙니다. `silu` 는 분해표에 있는데도 eager 에서
  `aten.silu.default` 가 그대로 디스패치됩니다. 파일이 벤더링되어 있는 것과(되어 있습니다)
  그것이 자동으로 적용되는 것은 다른 문제입니다. **이 12 개도 직접 구현하거나, 분해표를 조회하는
  배선을 우리가 만들어야 합니다.**
- **2개의 잔여 op** (`lift_fresh.default`, `max.default`)는 분해 테이블도 다루지 않으므로,
  shim 이 별도로 처리해야 할 항목입니다.

이 분석은 DESIGN.md §5 의 "계약을 측정에서 분리한다" 전략을 검증하되, 초안보다 정확한 측정 범위를 제시합니다.

---

## 6. 참고 자료

- **Core ATen 공식 정의:** https://docs.pytorch.org/executorch/stable/ir-ops-set-definition.html
- **PyTorch Decomposition Table:** https://pytorch.org/docs/main/generated/torch._decomp.core_aten_decompositions.html (또는 로컬: `torch._decomp.core_aten_decompositions()`)
- **torchnative DESIGN.md §5:** "사양서는 이미 공개되어 있다"

---

## 7. 미확인 항목

| 항목 | 상태 |
|---|---|
| Core ATen 공식 총 개수 | **189** — §0 에서 `torch.Tag.core` 로 실측 |
| 12개 분해 op 의 분해 경로 | 미확인 — 각각이 어떤 원시 op 으로 분해되는지 상세 추적 미완료 |
| 실사용 op 개수가 47 인지 48 인지 | **미해결** — §0 은 47, §2 는 48 로 적고 있습니다. §2 의 목록을 직접 세면 48 개입니다. 두 측정이 별도 실행이라 어느 쪽이 맞는지 확인되지 않았습니다. 어느 쪽이든 §0 의 분류 비율을 뒤집지는 않지만, **인용하기 전에 다시 재십시오** |

