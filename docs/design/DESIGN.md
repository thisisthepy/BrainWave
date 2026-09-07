# torchnative 설계 방향

온디바이스에서 PyTorch 생태계를 그대로 돌리기 위한 설계와, 그 판단의 근거를 정리한 문서입니다.

---

## 0. 이 문서가 답하는 것

torchnative 는 **온디바이스 인공지능 라이브러리**입니다. 연합학습(FL) 뿐 아니라
TTL 전반(그 안에 TTA 와 TTT 가 포함됩니다 — §3)을 커버하고, 그 위에 flash-attention /
flash-linear-attention 같은
**커널 최적화를 멀티플랫폼으로** 제공하는 것을 목표로 합니다.

그 전부가 하나의 전제에 걸려 있습니다 — **기기에서 PyTorch 모델이 실제로 돌아야 합니다.**
이 문서는 그것을 어떻게 가능하게 할지, 커널 계층을 어떤 계약 위에 올릴지, 그리고 아직 결정되지
않은 것이 무엇인지를 정리합니다.

관련 저장소:

| 저장소 | 역할 |
|---|---|
| `thisisthepy/PythonMultiplatform` | CPython 3.13 을 Kotlin Multiplatform 에 임베딩. **기기에서 파이썬이 도는 기반** |
| `robustaim/test-time-adapters` | 비전 TTA. `AdaptationEngine.online()` 이 실제 역전파를 요구 |
| `retentionlabs/theRiverLethe` | TTT-Linear / Titans. 손유도 닫힌 형식이라 autograd 불필요 |

---

## 1. 전제 — 파사드를 만들지 않는다

`PythonMultiplatform` 이 존재하는 이유는 **진짜 CPython 이 기기에서 돌면 진짜 pip 패키지가 기기에서
돌기 때문**입니다. transformers 모양의 API 를 흉내내는 파사드를 만드는 순간 그 기반이 무의미해집니다.

그러므로 목표는 "transformers 처럼 보이는 것" 이 아니라 **`import transformers` 가 기기에서
성공하는 것** 이고, 유일한 장애물은 `torch` 입니다.

이 판단을 뒷받침하는 관찰이 하나 더 있습니다. `test-time-adapters` 는 RT-DETR, YOLO11,
Grounding DINO, RCNN, ResNet 을 COCO / Cityscapes / SHIFT 위에서 돌리는 **비전 라이브러리**입니다.
llama.cpp 류의 LLM 추론 엔진이 커버하는 것이 하나도 없습니다. 범용 텐서 계층 외에 답이 없습니다.

---

## 2. 계층 구조

```
Kotlin Multiplatform (Android / iOS / Desktop / GraalVM native image)
└─ 임베디드 CPython 3.13                          ← PythonMultiplatform 이 이미 제공
   ├─ transformers (진짜, 또는 thelethe 의 포크)
   ├─ tokenizers · safetensors · huggingface_hub    ← 이미 torch-free, 일부는 이미 Rust
   ├─ ttadapters · thelethe · torchnative            ← 무수정으로 동작해야 함
   ├─ kernels                                       ← HF 표준. 인터페이스는 채용, 배포는 역전 (§8)
   │  └─ 번들 리졸버 → 앱 번들 안의 AOT 커널
   └─ torch                                         ← 만들 것
      ├─ torch/ 파이썬 트리    ← 상류에서 벤더링 (BSD)
      ├─ torch/_decomp/        ← 벤더링. Core ATen 밖 롱테일이 자동 분해됨
      └─ torch/_C              ← 여기만 새로 만듦

빌드 타임 (Gradle 플러그인 + 아티팩트 워커)
   ├─ 체크포인트 → 선택한 양자화 포맷으로 변환      (§7)
   └─ 커널 소스 → 타깃별 AOT 컴파일 → 번들에 적재   (§8)
```

### 왜 `torch._C` 만인가

**PyTorch 는 대부분 파이썬입니다.** `torch/nn/modules/*.py`, `torch/nn/functional.py`,
`torch/_tensor.py`, `torch/optim/`, `torch/utils/_pytree.py` 가 전부 파이썬 소스입니다.
네이티브인 것은 `torch._C` — ATen 텐서 + 디스패처 + autograd — 하나뿐입니다.

따라서 "PyTorch 를 다시 만든다" 가 아니라 **"파이썬 계층은 벤더링하고 `_C` 만 교체한다"** 입니다.
BSD 라이선스라 벤더링에 법적 문제가 없고, `nn.Module` · `Parameter` · `GenerationMixin` ·
캐시 클래스 · pytree 등록이 전부 따라옵니다. transformers 가 매달 바꾸는 것들은 거의 전부 aten
위쪽(파이썬)이므로 re-vendoring 으로 추적됩니다. 추격 대상이 "torch API 전체" 가 아니라
"aten op 집합" 으로 줄어들고, aten 은 훨씬 안정적입니다.

### torchnative 자체의 표면

위 그림에서 torchnative 가 차지하는 자리는 `torchnative` 한 줄이지만, 그것이 이 라이브러리가 하는
일의 전부입니다. **torchnative 는 torch 를 대체하지 않습니다** — torch 계층은 이 프로젝트가 성립하기
위한 *전제*이고, torchnative 는 그 위에서 §3 의 네 가지 능력을 제공합니다.

```
torchnative/
├─ delta/          ← 핵심 추상. 수명이 타입에 박힌 가중치 델타 (§3)
│                    수명 이름은 미정 — 벤치마크 시나리오에서 가져오지 않는다 (§3)
├─ adapt/          ← 단일 기기 안에서 닫히는 적응
│  ├─ stateless/     BN 통계 등, backward 없음      (단계 0)
│  └─ gradient/      엔트로피 · 보조과제 등          (단계 1)
├─ federated/      ← adapt 위에 얹히는 층. 집계 · 통신 · 프라이버시
├─ kernels/        ← 번들 리졸버. HF kernels 탐색 API 를 만족 (§8)
└─ api/            ← TorchNativeAPI. 배포 · 수명 정책 · 기기 오케스트레이션
```

설계상 강제해야 할 것 셋:

1. **`adapt/stateless/` 와 `adapt/gradient/` 를 타입으로 가릅니다.** §3 에서 본 대로 이름
   ("TTA")으로 묶으면 backward 없는 기기 빌드에 gradient 방법이 들어와 런타임에 터집니다.
   빌드 구성이 단계 0 만 지원하면 단계 1 타입은 **컴파일/임포트 시점에** 걸려야 합니다.
2. **`federated/` 는 `adapt/` 의 형제가 아니라 상위 층입니다.** FL 의 로컬 스텝은 TTA 와
   같은 기제를 씁니다. 형제로 두면 TTA 하나 쓰는 데 집계 스택이 딸려옵니다.
3. **수명 정책은 `delta/` 한 곳에만 둡니다.** `reset()` · 체크포인팅 · 영속화 · 집계가 방법마다
   재구현되지 않게 합니다.

### `torch.nn.federated` 네임스페이스 주입

현재 `torch/nn/federated.py` 가 `torchnative.nn.federated` 를 `torch` 네임스페이스에 얹는
add-hook 형태입니다. **우리가 `torch` 파이썬 트리를 소유하게 되면 이 주입이 해킹이 아니라 정식
경로가 됩니다.** 다만 두 가지를 정해야 합니다.

- **주입 지점을 일원화할 것.** 벤더링한 트리와 add-hook 이 같은 경로를 놓고 충돌하지 않도록,
  벤더링 트리 안에 확장 지점을 하나 두고 거기로만 들어갑니다.
- **주입 범위를 좁게 유지할 것.** 데스크톱에서는 **상류 torch 위에서도 동작해야** 합니다
  (§11 의 1~3 단계가 전부 데스크톱 상류 torch 에서 이뤄집니다). `torch` 를 소유한다는 전제를
  API 가 요구하기 시작하면 그 검증 경로가 막힙니다. add-hook 은 **편의**이지 **의존**이 아니어야
  합니다.

---

## 3. 범위를 정하는 두 축

### 용어 — 형제가 아니라 중첩이다

**TTL · TTA · TTT 는 나란한 범주가 아니라 포함 관계입니다.**

```
TTL  (Test-Time Learning)  테스트 타임에 학습이 일어나는 모든 경우
 └─ TTA  (Test-Time Adaptation)  그중 분포 변화에 대응하는 것. 레이블 없음, 소스 데이터 없음
     └─ TTT  (Test-Time Training)  그중 학습 시점에 보조 과제를 함께 훈련해야 하는 것
```

정의는 [Wang et al., *In Search of Lost Online Test-Time Adaptation: A Survey*, IJCV
2025](https://doi.org/10.1007/s11263-024-02213-5) 를 따릅니다.

- **TTA** — "adapting the model to unseen distributions using **unlabeled test data**"이고,
  비지도 도메인 적응과 달리 **소스 데이터에 접근하지 않습니다.**
- **TTT** — "introduces an auxiliary task for **both training and adaptation**. During training,
  the original backbone is modified into a **'Y'-shaped structure**, with one branch for image
  classification and another for an auxiliary task, such as rotation prediction."

**둘을 가르는 것은 "학습 시점을 건드리는가" 입니다.** 서베이가 OTTA 를 정의하면서 명시적으로
"the pre-trained model is expected to retain its original architecture ... **without modifying its
layers or introducing new model branches during training**" 이라고 못박습니다. TTT 는 Y자 분기를
요구하므로 이 조건 밖이고, 그래서 TTA 안에 있되 OTTA 는 아닙니다.

**FL 은 이 중첩의 바깥입니다.** TTL 계열은 *한 기기 안에서 닫히는* 학습이고, FL 만 **여러 기기에
걸친 집계**입니다. 그래서 FL 은 형제가 아니라 위에 얹히는 층입니다 — 로컬 스텝 자체는 TTL 의
기제를 쓰고, 그 위에 집계·통신·프라이버시가 붙습니다.

**따라서 torchnative 의 범위는 "TTL + FL" 입니다.** TTA 와 TTT 는 별도로 커버할 대상이 아니라
TTL 안의 좁은 영역이고, 라이브러리 구조도 넷을 나열할 것이 아니라 이 중첩을 반영해야 합니다.

### 아키텍처로서의 TTT 는 이 중첩에 들어가지 않는다

`TTT-Linear` · `Titans` 의 "TTT" 는 위 정의의 TTT 와 **같은 단어이지만 다른 범주**입니다.
적응 방법이 아니라 **아키텍처**입니다. `theRiverLethe` 의 분류가 그렇게 되어 있습니다 —
`architectures/protogenois` 는 "transformer 를 대체할 새 아키텍처 아이디어" (여기에 `ttt_linear`,
`ttt_mlp`), `architectures/titans` 는 "**메타러닝 메모리 아키텍처**" (LMM · MAC · MAE · MAG · MAL)
입니다. 적응 방법은 다른 저장소의 `ttadapters/methods/` 에 있습니다.

**따라서 두 축은 직교합니다.**

| | 무엇 | 어디 |
|---|---|---|
| **아키텍처 축** | 모델이 무엇인가. ResNet · ViT · Llama · TTT-Linear · Titans | `theRiverLethe/architectures/` |
| **적응 방법 축** | 그 모델을 테스트 타임에 어떻게 다룰 것인가. TTL ⊃ TTA ⊃ TTT | `ttadapters/methods/` |

TTT-Linear 모델 위에 TTA 를 돌릴 수도 있고 ResNet 위에 돌릴 수도 있습니다. **모델 내부에
가중치 갱신이 있다는 것은 그 모델의 성질이지 torchnative 의 방법이 아닙니다.**

설계에 미치는 결론이 분명해집니다.

- **`adapt/` 아래에 아키텍처를 두지 않습니다.** 아키텍처는 torchnative 의 *입력*이지 구성 요소가
  아닙니다.
- **아키텍처 내부의 fast weight 갱신은 미분 단계 축의 항목이 아닙니다.** torchnative 입장에서
  `model(x)` 안에서 일어나는 일이므로 그냥 forward 입니다. `torch._C` 가 그 op 들을 제공해야
  한다는 요구는 남지만, 그것은 §6 사다리의 대상(모델 목록)이지 §3 의 방법 분류가 아닙니다.

### 수명은 별개의 축이지만, `ScenarioType` 은 그 축이 아니다

> **이전 판본에서 `delta/` 의 수명 정책을 `ttadapters` 의 `ScenarioType` 에 맞추라고 썼으나,
> 철회합니다.** `ScenarioType` 은 **평가 프로토콜**이지 런타임 수명 정책이 아닙니다.
>
> 저장소가 이미 그렇게 분류해 두었습니다 — 경로가 `ttadapters/**datasets**/scenarios/base.py`
> 이고, `BaseScenario` 는 도메인을 키로 하는 데이터셋 `dict` 입니다. `play()` 가 도메인마다
> `DataLoader` 를 만들어 채점 `script` 에 넘기고, 끝나면 도메인 평균(`res["avg"]`)을 냅니다.
> **여러 방법을 비교하기 위한 벤치마크 하네스입니다.**
>
> 그래서 `STANDARD` 의 리셋은 **측정을 위한 장치**입니다. 도메인마다 기준 상태로 되돌려야 각
> 도메인 결과가 독립적으로 비교되기 때문이지, 제품이 그렇게 동작해야 해서가 아닙니다. 기기에는
> 도메인 레이블도, 도메인 경계 신호도, "심각도 1→5" 도 없습니다. 카메라 피드가 있을 뿐입니다.
>
> **그리고 서베이 대응을 확인한 것이 이 결정을 뒷받침한다고 적었는데, 반대입니다.** 그 대응은
> 이 분류가 *평가* 서베이에서 온 *평가* 분류라는 것을 확인해 줍니다. 가져오지 말아야 할 이유입니다.

수명 축 자체는 여전히 필요합니다. 다만 그것을 정하는 것은 도메인 경계가 아니라 **시스템 사건**
입니다 — 앱이 백그라운드로 갔다, 사용자가 바뀌었다, 저장 공간이 부족하다, 동기화 창이 열렸다.

정책이 답해야 할 질문은 셋이고, 셋 다 검증된 사실에서 나옵니다.

| 질문 | 왜 실제 문제인가 |
|---|---|
| 이 델타를 버리고 기준으로 되돌릴 수 있는가, 비용은 얼마인가 | `AdaptationEngine.reset()` 이 존재하고 지금은 전체 가중치 사본을 요구합니다 (§9-5) |
| 프로세스 재시작을 넘어 살아남는가 | TTL 은 테스트 타임 학습 전반이므로 세션을 넘는 경우를 포함합니다 |
| 기기 밖으로 나갈 수 있는가 | FL 만 해당하고, 그때 직렬화·보안 집계·프라이버시가 붙습니다 |

**이름은 아직 정하지 않습니다.** 한 번은 지어냈다가(`Ephemeral · Session · Persistent · Shared`)
근거가 없어 버렸고, 한 번은 남의 평가 분류를 가져왔다가 층이 달라 버렸습니다. 세 질문에 답이 필요한
것은 확실하니, 이름은 §11 의 1~3 단계에서 실제 사용처가 드러난 뒤에 붙입니다.

> **갱신 (docs/models/ADAPT.md, 2026-09) — 첫 실사용처가 생겼고, 이름은 필요하지 않았습니다.**
> `torchnative.adapt.Tent` 이 SmolLM2-135M 위에서 실제로 돕니다. 그것이 요구한 것은 이름 넷이
> 아니라 **위 세 질문에 대한 답 셋**이었고, `torchnative.delta.Delta` 는 그것을 라벨이 아니라
> **동작으로** 답합니다 — `revert()`/`nbytes` 는 되고, `persist()` 와 `publish()` 는 각각
> *실행 가능한 검사*를 대며 거절합니다. 지금 이름을 붙이면 그 둘에는 **아무것도 갖지 않은 능력의
> 이름**을 붙이게 되므로, 세 번째로 같은 실수를 하는 셈입니다. 하나가 거절을 멈출 때 붙입니다.
>
> > **정정 (2026-09-02, `docs/distributed/FEDERATED.md`).** 둘 다 거절을 멈췄습니다. `persist()` 는
> > `docs/training/BACKWARD.md` §14 에서, `publish()` 는 `world_size = 2` 전송(`docs/distributed/TRANSPORT.md`) 위에서.
> > 지금 `publish()` 는 두 OS 프로세스 사이에서 가중 평균을 내고, 그 결과는 같은 평균을 중앙에서
> > 계산한 것과 비트가 같습니다. 즉 세 질문이 전부 **동작으로** 답합니다 — 그런데도 이름은 여전히
> > 붙이지 않았습니다. 실사용처가 하나 더 생기기 전에는 근거가 같기 때문입니다.
>
> 세 질문 중 첫째의 비용도 이제 수치입니다. 델타는 **자기가 덮는 파라미터의 베이스만** 들고
> 있으므로 SmolLM2 Tent 기준 140,544 B 이고, 모델 전체 사본은 538,060,032 B 입니다 (§9 항목 5 의
> `base_state` 가 요구하던 것). 그리고 **사본을 없앨 수는 없습니다** — 오프셋을 도로 빼는 방식은
> 35,136 개 중 2 개가 어긋납니다(`(w+d)-d ≠ w`). 측정치는 docs/models/ADAPT.md §5.

**`ScenarioType` 이 무의미한 것은 아닙니다.** torchnative 가 적응 방법을 제공한다면 벤치마크 수치를
재현할 수 있어야 하고, 그러려면 리셋 프로토콜을 지원하는 평가 하네스가 필요합니다. 그것은 이미
`ttadapters` 가 하는 일이고, **평가 쪽에 있어야지 `delta/` 에 있어서는 안 됩니다.**

### 축 1 — 미분 요구

| 단계 | 무엇 | 필요한 것 | 어디서 |
|---|---|---|---|
| 0 | 통계만 갱신하는 정규화 보정, 데이터 기반 방법. **그리고 모든 아키텍처의 순전파** — 내부에 fast weight 갱신이 있는 것 포함 | **forward 만** | 기기 |
| 1 | 손실을 최소화하는 TTA 전반, 보조과제 TTT, 모듈·프롬프트 추가, FL 로컬 스텝 | forward + **좁은** backward | 기기 |
| 2 | 메타러닝 메모리 아키텍처의 **사전학습** | 내부 갱신 전체를 통과하는 full autograd | **데스크톱 전용, 영구히** |

**단계 2 를 기기에서 명시적으로 배제하는 것이 중요합니다.** 가장 어려운 요구 — scan 을 통과하는
역전파와, 2048 토큰 기준 1.5GB 를 넘는 활성값 보존 — 를 기기 타깃에서 통째로 들어냅니다.

**메타러닝 메모리 아키텍처의 추론이 단계 0 에 들어가는 것이 핵심입니다.** 이름에 "training" 이
들어가고 순전파 안에서 가중치가 갱신되지만, 그 갱신이 **손으로 유도한 닫힌 형식**이라 autograd
테이프가 없습니다. 구현된 둘 다에서 확인했습니다.

| | 확인 내용 |
|---|---|
| `ttt_linear` | `backward()` 가 L2 손실의 그래디언트를 손으로 전개 (`modular_ttt_linear.py:471-531`) |
| `titans/origin` | 같은 구조 (`modeling_origin.py:697`). **2271 줄 전체에서 autograd 흔적은 `@torch.no_grad()` 하나뿐** — `autograd` · `requires_grad` · `.backward()` 호출 · `optim.` · `create_graph` 가 0 회. `adapt_step` 이 momentary/past surprise(모멘텀)를 텐서로 직접 계산하고, `lr_gate` 가 네 게이트(token · momentary · past · forget)를 돌려줍니다 |

즉 **shim 에 backward 를 구현하지 않아도 이 아키텍처들의 추론이 성립합니다.** 배제되는 것은
**사전학습뿐**입니다 (단계 2).

원 논문도 같은 형태입니다 — Nested Learning 발표 자료가 "원본 TITANS 구현은 Optimizer 없이 동작
(closed-form solution, 직접적인 outer product update)" 이라고 적고 있고, NL 은 그것을 경사하강과
**수학적으로 동등**하다고 재해석할 뿐입니다. **재해석이지 구현 변경이 아니므로 단계 0 이 유지됩니다.**

단계 1 은 backward 가 필요하지만 `AdaptationEngine.online_parameters()` 가 갱신 대상을 명시적으로
좁히므로 (`base.py:196-222`), **backward 가 필요한 op 집합은 forward 집합보다 훨씬 작습니다.**

정리하면 `torch._C` 의 사양은 **"전체 forward op + 부분 backward op"** 이고, 두 집합 모두
측정 가능합니다 (§6).

### 이 축은 서베이의 분류로 판정한다

**단계 0/1 은 방법 이름이 아니라 방법이 무엇을 하느냐로 갈립니다.** 서베이가 OTTA 를 세 갈래로
나누는데, 그 분류가 그대로 미분 요구를 결정합니다.

| 서베이 분류 | 예 | 미분 요구 |
|---|---|---|
| **최적화 기반** — 정규화 보정 | 테스트 배치에서 통계(μ, σ)만 다시 계산 | **단계 0** |
| **최적화 기반** — 정규화 보정 | 손실로 affine {γ, β} 를 갱신 | 단계 1 |
| **최적화 기반** — mean-teacher · 비지도 목적함수 · 의사 레이블 | | 단계 1 |
| **데이터 기반** — 증강 · 메모리 뱅크 | 서베이가 TTAug 는 "does not require any modification to the model training process" 라고 명시 | **단계 0** |
| **모델 기반** — 모듈 추가 · 치환 · 프롬프트 | 추가한 모듈을 학습시켜야 함 | 단계 1 |

**정규화 보정이 양쪽에 걸치는 것에 주의하십시오.** "BatchNorm 계열이면 단계 0" 이 아닙니다 —
통계만 다시 계산하면 backward 가 없고, 같은 레이어의 affine 파라미터를 손실로 갱신하면 backward 가
있습니다. 디렉터리 이름으로 판정할 수 없고 **구현을 보고 판정해야 합니다.**

> **주의 — 이 표는 서베이 기준이지 `ttadapters` 의 구현 상태가 아닙니다.** 현재
> `methods/entropies/` 와 `methods/auxtasks/` 트리는 `__init__.py` 까지 포함해 **전부 0바이트**
> 입니다. 구현이 있는 것은 `batchnorms/` · `deepsupervisions/` · `pefts/` · `regularizers/`
> 뿐입니다. 각 방법이 실제로 어느 단계인지는 §6 의 사다리에서 확정되며, 그때까지 배정은
> 서베이에 근거한 예상입니다.

**그래도 단계 1 이 존재한다는 것 자체는 확정입니다.** `online()` 이 `online_parameters()` 에
`requires_grad` 를 켜고 SGD/Adam/Muon 옵티마이저를 생성합니다 (`base.py:169-225`). 프레임워크가
그래디언트 기반 적응을 전제로 설계되어 있습니다.

**API 를 하나로 묶으면 이 차이가 숨습니다.** 기기에서 backward 가 없는 빌드에 단계 1 방법이
들어오는 순간 런타임에 터집니다. 타입 수준에서 갈라둘 것.

### 축 2 — 상태의 수명과 소재

미분 요구만으로는 FL 이 설명되지 않습니다. 둘째 축이 필요합니다.

**수명은 방법이 아니라 정책입니다.** 같은 TENT 를 한 스트림에서 버리고 끝낼지 세션을 넘겨 이어
쓸지는 방법이 정하지 않습니다. 다만 **그 정책을 벤치마크의 시나리오에서 가져오면 안 됩니다** —
위 "수명은 별개의 축이지만" 참조. 기기에서는 도메인 경계가 관측되지 않으므로, 정책을 움직이는 것은
시스템 사건입니다.

소재는 별개입니다.

| | 상태가 기기 밖으로 나가나 |
|---|---|
| TTT · TTA | 아니오 |
| FL | **예 — 집계 서버로** |

FL 만 상태가 기기를 떠납니다. 그래서 FL 에만 직렬화 포맷 · 통신 · 보안 집계 · 차분 프라이버시가
붙습니다. 이걸 섞으면 TTA 하나 돌리는 데 FL 스택이 딸려옵니다.

### 그래서 핵심 추상은 하나다

적응 방법들을 관통하는 것은 **베이스 가중치 위의 델타**입니다 — TTA 가 적응시킨 파라미터와 FL 의
로컬 업데이트가 같은 물건이고 **수명과 행선지만 다릅니다.**

따라서 torchnative 의 중심 타입은 "적응 방법" 이 아니라 **수명이 타입에 박힌 가중치 델타** 여야
합니다. 그러면 `reset()` · 체크포인팅 · 집계 · 영속화가 각 방법마다 재구현되지 않고 델타의 수명
정책 하나로 정리됩니다. (`ttadapters` 가 지금 `base_state` 로 전체 가중치 사본을 들고 있는 문제도
여기서 해소됩니다 — §9 항목 5.)

> **구현됨 (docs/models/ADAPT.md §2).** `torchnative.delta.Delta` 가 베이스 · 오프셋 · 세 수명 질문을
> 전부 소유하고, `torchnative.adapt.Method` 는 **셋만 선언하고 상태를 갖지 않습니다** — 어느
> 파라미터를 움직이는가(`select`), 무엇을 내려가는가(`objective`), 이 축의 어느 단계인가
> (`stage`). 그래서 `Tent` 이 40 줄이고 `reset()` · 체크포인팅 · 직렬화를 하나도 갖지 않습니다.
> 두 번째 방법이 그것을 안 쓰는 것으로 물려받는 구조인지는 **아직 한 방법으로만 확인됐습니다**
> (docs/models/ADAPT.md §9.1).

**아키텍처 내부의 fast weight 는 이 델타가 아닙니다.** TTT-Linear 의 `TTTLinearCache` 처럼 모델이
스스로 관리하는 상태이고, 수명도 모델의 캐시 수명이지 시나리오 정책이 아닙니다. `delta/` 가 이것까지
소유하려 들면 §3 의 직교성이 깨지고 — 모델을 바꿀 때마다 `delta/` 를 고치게 됩니다. **경계는
`model(x)` 입니다.** 그 안은 모델의 것, 그 밖이 torchnative 의 것.

TTL 은 이 델타가 사는 범위이고 (§3 의 중첩), TTA · TTT 는 그 안의 좁은 영역입니다. 따라서
**`adapt/` 아래에 TTA 와 TTT 를 나란한 모듈로 두면 안 됩니다** — 중첩을 평평하게 펴는 것이라,
어느 쪽에 넣을지 모호한 방법이 반드시 생깁니다.

---

## 4. 만들 것과 빌려올 것

| | |
|---|---|
| **빌려옴** | torch 파이썬 트리, `torch/_decomp` 분해표, candle (텐서 엔진), tokenizers · safetensors, Core ATen 목록 (사양서) |
| **이미 있음** | CPython 임베딩, KMP FFI, Gradle 플러그인 + 아티팩트 워커 |
| **새로 만듦** | `torch._C` PyO3 어댑터, 양자화 변환 파이프라인 |

### 텐서 엔진은 candle

[huggingface/candle](https://github.com/huggingface/candle) 을 `candle-core` 만 씁니다. 근거 셋:

1. **동적 랭크.** torch API 는 런타임 동적 랭크입니다 (`x.view(-1, d)` 의 결과 랭크가 실행 시점에
   정해짐). candle 의 텐서는 shape 가 타입이 아니라 값이라 그대로 맞습니다. 대안인
   [burn](https://github.com/tracel-ai/burn) 은 `Tensor<B, D>` 정적 랭크(const generics)라
   torch API 를 씌울 수 없습니다.
2. **양자화가 이미 있음.** llama.cpp 의 GGML/GGUF k-quant 를 읽습니다. §7 의 양자화 문제가 크게
   줄어듭니다.
3. **계보.** 저자 Laurent Mazare 는 candle 이전에 `tch-rs`(libtorch 의 Rust 바인딩, 5.5k stars)
   와 `ocaml-torch` 를 만들었습니다. **PyTorch 를 다른 언어로 옮기는 일에 가장 경험이 많은 사람이,
   배포 크기와 파이썬 제거가 중요해지자 libtorch 바인딩을 버리고 처음부터 다시 쓴 것이 candle
   입니다.** 우리 조건과 같은 조건에서 내려진 판단입니다.

`candle-transformers` 는 **쓰지 않습니다.** 그것은 모델 아키텍처를 Rust 로 재구현한 것이라,
§1 에서 거부한 접근 그 자체입니다.

burn 은 나중에 FL 학습 경로를 별도 트랙으로 둘 때 검토합니다. 그쪽은 torch 흉내가 필요 없어
정적 랭크가 오히려 장점이고, `Autodiff<B>` 백엔드 데코레이터와 `Learner` · 옵티마이저 ·
체크포인팅이 이미 갖춰져 있습니다.

### 주의: candle 과 torch 의 의미론 차이

candle 은 복사 지향이라 **view / stride aliasing 과 in-place 연산**(`add_`, `copy_`)에서
임피던스가 생깁니다. transformers 의 KV 캐시 갱신이 정확히 그 경로를 밟으므로, 스파이크 초기에
여기부터 확인해야 합니다.

---

## 5. 아직 닫히지 않은 결정 — A 대 B

`torch._C` 를 무엇으로 만들 것인가.

- **A. candle 위 PyO3 어댑터**
- **B. selective libtorch 빌드**

### B 는 처음 생각보다 살아 있다

"PyTorch 는 모바일 빌드가 불가능하다" 는 통념을 분해하면 이렇습니다.

| | 실상 |
|---|---|
| libtorch(C++) 를 Android / iOS 로 빌드 | **된다** (실제로 빌드해 확인 — 아래 "B 는 판정됐다") |
| TorchScript / `.pte` 아티팩트를 기기에서 실행 | **된다** |
| 임베디드 CPython 에서 `import torch` | **아무도 안 했다** ← 우리가 필요한 것 |

**"JIT 때문에 안 된다" 는 틀린 설명입니다.** TorchScript 는 JIT 컴파일러가 아니라 직렬화된
그래프에 대한 바이트코드 인터프리터입니다. 기계어를 생성하지 않으므로 iOS 의 W^X 를 위반하지 않고,
PyTorch Mobile 이 실제로 iOS 에서 수년간 돌았습니다.

**진짜 불가능한 것은 `torch.compile` / TorchInductor 뿐입니다.** 런타임에 C++ 컴파일러를 호출해
커널을 생성하므로 iOS 에서 원리적으로 불가능합니다. 그리고 우리는 eager 실행이 목표이므로
이 손실은 받아들일 수 있습니다.

크기도 in-tree 해법이 있습니다. `SELECTED_OP_LIST=<yaml> BUILD_PYTORCH_MOBILE=1` 로 필요한 op 만
포함하면 **MobileNetV2 op 집합 기준 arm-v7 압축 ~4.5MB** 입니다. op 목록은 `TRACING_BASED=1` 로
계측 빌드를 만들면 `model_tracer` 가 뽑아줍니다. **B 를 고를 경우 op 목록을 손으로 만들 필요가
없다는 뜻이고, §6 의 사다리가 그 목록을 자연히 쌓아 줍니다.**

### 그럼 무엇이 실제 차이인가

**두 경로 모두 op 집합을 미리 확정해야 한다는 점은 같습니다.** selective build 도 닫힌 세계이고,
Rust 구현도 닫힌 세계입니다. 커버리지는 차이가 아닙니다.

| | A (candle + shim) | B (selective libtorch) |
|---|---|---|
| 소유하는 코드 | 작음 (`_C` 어댑터) | 300만 줄 C++ 의 크로스 컴파일 |
| 수치 의미론 | 다시 만들어야 함 (dtype promotion, broadcasting, stride/view) | **공짜** |
| 상류 지원 | 해당 없음 | **0** — 모바일 노력이 전부 ExecuTorch 로 이동 |
| 성격 | 의미론 문제 (torch 와 맞을 때까지 안 끝남) | 빌드 문제 (유한하고 기계적) |
| 주요 리스크 | 수치 불일치가 조용히 번짐 | `native_functions.yaml` 코드젠이 host==target 을 가정 |

**B 는 "이 논지가 성립하는가" 를 가장 빨리 답하고, A 는 "이걸 출시할 수 있는가" 의 답일 가능성이
높습니다.**

### B 는 판정됐다 — **A 로 간다**

스파이크를 돌렸고(32 분, 60 분 타임박스 안), **결론이 예상과 반대 방향에서 나왔습니다.**
상세는 `docs/platform/B_SPIKE.md`.

**크로스 컴파일은 됩니다.** 전체 op 빌드와 10 개 선택 빌드 둘 다 aarch64 `libtorch_cpu.a` 를
만들었고(각각 202MB · 175MB), 종료 코드 0, 외장 4.7GB 를 썼습니다. 막힌 것 다섯 개는 전부
얕았습니다 — 번들 cmake 버전, 삭제된 eigen 서브모듈 참조, NDK 27 의 Vulkan 래퍼, `cpuinfo` 이중
링크, 그리고 선택 빌드가 CUDA 전용 `at::cpu::_scaled_grouped_mm_v2` 를 호출하는 torchgen 버그
(디스패치 include 1074 개 중 1 개).

**그런데 그것이 B 의 관문이 아니었습니다.**

```cmake
# CMakeLists.txt:813-816
if(ANDROID OR IOS OR DEFINED ENV{BUILD_PYTORCH_MOBILE_WITH_HOST_TOOLCHAIN})
  set(INTERN_BUILD_MOBILE ON)
...
# CMakeLists.txt:917  — 같은 블록 안
  set(BUILD_PYTHON OFF)
```

**Android 나 iOS 툴체인이면 `BUILD_PYTHON` 이 자동으로 꺼집니다.** 옵션이 아니라 덮어쓰기입니다.
즉 **모바일 빌드 경로는 구조적으로 `torch._C` 를 만들 수 없습니다** — lite interpreter 용
libtorch 를 만듭니다. B 가 필요로 했던 바로 그것이 나오지 않습니다.

같은 블록이 `USE_DISTRIBUTED OFF` · `NO_API ON` 도 강제하고, `BUILD_MOBILE_AUTOGRAD` 를 켜지
않으면 `INTERN_DISABLE_AUTOGRAD ON` 입니다 — **§3 의 단계 1(TTA · FL 로컬 스텝)이 요구하는
backward 도 기본값으로 없습니다.**

> **정정 두 개.** 위 표에서 "`scripts/build_mobile.sh` 가 지금도 main 에 있음" 이라고 적었는데
> **틀렸습니다. main 에서 404 입니다.** 삭제 커밋은 `91602a92548d` "Cleanup old caffe2 scripts
> (#158475)" (2025-07-23) — v2.8.0 에는 있고 v2.9.0 에는 없습니다. `cmake/iOS.cmake` 도 main 에서
> 사라졌습니다. 저는 검색 결과 제목이 "…at main" 인 것을 보고 확인했다고 여겼습니다. **낡은 검색
> 색인을 저장소 확인으로 대신한 것입니다.**
>
> 다만 **CMake · codegen 기계는 온전하고 실제로 동작합니다** — `SELECTED_OP_LIST` ·
> `TRACING_BASED` · `INTERN_BUILD_MOBILE` · `gen_selected_mobile_ops_header`. pypackpack 의
> `Cargo.kt` 처럼 속이 빈 경우가 아니라 **손잡이만 떼어진 경우**입니다. 상류 지원이 0 이라는 것은
> 정량적으로도 확인됩니다 — `.github/` 와 `.ci/` 전체에서 모바일 경로 참조 **0 건**이고,
> `test/mobile/custom_build/build.sh:41,55` 와 `android/common.sh:66` 은 삭제된 스크립트를
> 부르는 끊긴 호출입니다.

**그리고 §6 의 판정 기준이 틀린 질문이었습니다.** "B 의 유일한 미지수는 크로스 컴파일이
완료되는가" 라고 적었는데, 크로스 컴파일은 §5 의 표에서 이미 "된다" 로 분류돼 있던 칸입니다.
스파이크는 미지수를 **해소한 것이 아니라 옮겼습니다** — 그리고 옮겨간 자리에서 B 가 성립하지
않는다는 것이 드러났습니다.

**결론: A(candle 위 `torch._C`)로 갑니다.** IMPORT_WALLS 4·5 차가 A 의 비용(실행되지 않는
1070 모듈을 임포트 가능하게 유지)을 드러냈지만, **B 는 애초에 목표물을 만들지 못합니다.**
비용 비교가 아니라 가능/불가능의 문제였습니다.

미확인으로 남은 것: 링크·스트립 후 배포 크기, 기기 실행, `TRACING_BASED`/`model_tracer` 동작,
`USE_BLAS=OFF` 의 성능 비용, 그리고 **iOS 전체**(`cmake/iOS.cmake` 가 없어 Android 결과를
외삽할 수 없음).

### 선행 사례

[`ljk53/upytorch`](https://github.com/ljk53/upytorch) — PyTorch 모바일 엔지니어가 만든
"MicroPython + PyTorch ATen 커널" 바인딩입니다.

| | 크기 |
|---|---|
| upytorch (비압축, x86-64) | **1MB 미만** (런타임 + 선택된 ATen 커널) |
| upytorch (압축) | **~430KB** |
| CPython 3.8.3 | ~2.6MB |
| PyTorch Mobile (선택 빌드) | 4.5 ~ 20MB |
| libtorch-cpu.so (전체) | **50MB 초과** |

**op 오버로드 약 40개로 AlexNet 이 돕니다.** 규모는 개념 증명 수준(53 stars, 85 commits,
추론 전용)이지만, 증명하는 것이 정확히 우리에게 필요한 두 가지입니다 — 파이썬 인터프리터에
ATen 커널을 torch 시그니처로 직접 붙이는 것이 되고, 결과가 작다.

### 사양서는 이미 공개되어 있다

op 집합을 추측할 필요가 없습니다. PyTorch 에 **[Core ATen Operator Set]**
(https://docs.pytorch.org/executorch/stable/ir-ops-set-definition.html) 이라는 공식 답이 있습니다.
공개 저장소와 유명 오픈소스 모델을 조사해 실제로 많이 쓰이는 aten op 을 추린 것이고, 목적이 명시적으로
"백엔드와 컴파일러가 처리해야 할 연산자 수를 줄이는 것" 입니다.

같이 딸려오는 **Core ATen Decomposition Table** 이 core 가 아닌 op 을 core op 으로 분해하는
규칙표이고, **이 규칙들은 파이썬 소스**(`torch/_decomp/decompositions.py`)입니다. 즉 §2 의
벤더링에 포함됩니다.

그러면 shim 이 3층으로 정리됩니다.

1. **Core ATen 을 Rust/candle 로 구현** — 하드 플로어
2. **분해 테이블을 벤더링** — 롱테일이 자동으로 core op 으로 분해됨
3. **핫한 op 만 손으로 최적화** — matmul, attention, 양자화 경로

부수 효과로, ExecuTorch 도 Core ATen 을 타깃하므로 나중에 무거운 모델만 `.pte` 로 빼는 하이브리드가
같은 계약 위에서 열립니다.

---

## 6. 측정 대신 계약과 발견

**op 집합을 미리 세는 계획은 폐기합니다.** 성립하지 않습니다.

- **셀 대상이 아직 없습니다.** `ttadapters` 의 `methods/entropies/` 와 `methods/auxtasks/` 는
  `__init__.py` 까지 전부 0바이트이고, `theRiverLethe` 의 `cronos` 는 클래스 골격 30줄입니다.
  구현되지 않은 것의 op 은 셀 수 없습니다.
- **셀 수 있는 것도 비용이 큽니다.** 저장소 둘에 걸친 모델 10 개를 각각 인스턴스화해 돌리려면
  의존성·설정·체크포인트를 전부 세워야 합니다. 숫자 하나 얻기 전에 며칠이 갑니다.
- **선택 빌드 크기와 크로스 컴파일은 측정이 아니라 공사입니다.** 상류 지원이 0 인 빌드를 뚫는
  것이므로, 결과가 나오기 전까지 아무것도 진행되지 않습니다.
- **대상이 움직입니다.** `cronos` 는 설계 중이고 `olympians` 는 미정입니다. 스냅샷을 찍어도
  다음 아키텍처에서 무효가 됩니다.

무엇보다 이 계획은 **시작하기 전에 답을 알려는 것**이었습니다. 그럴 필요가 없습니다.

### 계약을 측정에서 분리한다

op 집합은 이미 공개되어 있습니다 — **Core ATen** (§5). 그러니 "우리 모델이 무엇을 쓰는가" 를 묻는
대신 **"우리 모델을 Core ATen 으로 표현할 수 있는가"** 를 묻습니다. 대부분은 분해 테이블이
기계적으로 답하고, 답하지 못하는 것만 남습니다.

이 전환으로 op 집합이 **범위 산정 문제에서 우선순위 문제로 바뀝니다.** "얼마나 큰 일인가" 가 아니라
"Core ATen 중 무엇부터 구현할 것인가" 이고, 후자는 시작을 막지 않습니다.

### 발견은 shim 이 스스로 한다

미구현 op 은 **op 이름을 담아 시끄럽게 실패**하게 둡니다.

```python
raise NotImplementedError(f"aten op not implemented in torch._C shim: {op}")
```

모델을 돌리면 실패 트레이스가 다음에 구현할 op 을 지목합니다. **가장 많이 쓰이는 것부터 순서대로**
나오므로, 목록을 미리 만드는 것보다 우선순위가 정확합니다. 열거가 사라지고 shim 자체가 계측기가
됩니다.

### 무엇을 대상으로 삼는가

`theRiverLethe` 의 아키텍처는 `thelethe/architectures/__init__.py` 가 **세 계열**로 선언합니다.
그리스 신화의 세대 순서(프로토게노이 → 티탄 → 올림포스)를 분류 축으로 쓰며, 실제 배치가 계보
시기와 맞습니다.

| 계열 | 성격 | 온디바이스 범위 |
|---|---|---|
| `protogenois` | transformer 를 대체할 새 아키텍처 아이디어 — `llama` · `llama_mor` · `recursive_llama` · `relaxed_recursive_llama` · `ttt_linear` · `ttt_mlp` · `vit` | **구현된 것만** |
| `titans` | 메타러닝 메모리 아키텍처 (LMM · MAC · MAE · MAG · MAL) — `origin` · `atlas` · `cronos` | **구현된 것만** |
| `olympians` | Nested Learning 기반 AGI 미래 아키텍처. **설계 미정** | **범위 밖** |

`olympians/__init__.py` 는 비어 있습니다. **설계되지 않은 아키텍처의 런타임을 설계할 수는 없으므로
명시적으로 범위 밖에 둡니다.** 설계가 나오면 그때 사다리에 추가합니다.

**`titans` 에서 실제로 구현된 것은 `origin` 하나뿐입니다.** `atlas` 와 `cronos` 는 둘 다
클래스 골격만 있는 스텁입니다.

| | `modeling_*.py` | 상태 |
|---|---|---|
| `origin` | 89,793 B (+ `modular` 35,229 B, `configuration` 12,300 B) | 구현됨 |
| `atlas` | 1,010 B | **스텁** — `PreTrainedTitansModel` 상속 + `init_weights()` 뿐 |
| `cronos` | 1,027 B | **스텁** — 같은 골격 |

둘 다 자리를 잡은 뒤에 사다리에 넣습니다.

여기에 `ttadapters` 쪽의 RT-DETR · YOLO11 과 적응 방법들이 더해집니다.

### 사다리 — 작은 것부터 하나씩

전부를 한 번에 세는 대신 **가장 작은 모델 하나를 끝까지 통과시키고 하나씩 늘립니다.** 각 추가가
그 아키텍처의 한계 비용을 부산물로 알려줍니다.

| # | 대상 | 이 단계가 답하는 것 |
|---|---|---|
| 1 | `vit` 또는 소형 `llama` | 순전파 최소 집합. 벤더링·import 배선이 실제로 서는가 |
| 2 | `ttt_linear` | 손유도 갱신이 shim 위에서 도는가. §7 의 prefill 비용 실측 |
| 3 | `ttadapters` 의 `online()` 한 스텝 | backward 부분집합의 실제 크기 |
| 4 | `titans/origin` | 메타러닝 메모리의 한계 비용 (`atlas` · `cronos` 는 스텁이라 아직 없음) |
| 5 | 재귀 계열 (`llama_mor` 등) | 파라미터 공유가 델리게이트 가정을 건드리는가 |

**1 번이 가장 값싸고 가장 많이 알려줍니다.** 벤더링한 torch 파이썬 트리와 빈 `_C` 스텁만으로
`import transformers` 를 시도하는 것이 여기 포함되고, 그때 나오는 벽의 개수가 이 계획의 실현
가능성을 거의 다 말해줍니다.

### 정적 스캔은 계측이 아니라 조기 경보로 쓴다

AST 로 두 저장소의 `torch.*` · `F.*` 호출을 훑는 것은 **op 집합 산정에는 쓸모가 없습니다** —
파이썬 API 는 aten op 과 1:1 이 아닙니다 (`nn.Linear` 하나가 `addmm` 이 됩니다). 그러나 **깨질
것을 미리 잡는 데는 유용하고, 실행이 필요 없어 미구현 코드와 미래 아키텍처에도 걸립니다.**

찾을 것은 §9 에서 실제로 물렸던 종류입니다.

- `torch.compile` · `torch.jit` — iOS 에서 불가능 (§5)
- `torch.cuda.*` · `torch.backends.cudnn.*` — 가드 없는 호출
- `torch._*` 사설 API — shim 이 제공해야 하는 것
- `einsum` — shim 구현 비용이 큰 것
- 텐서 값에 대한 파이썬 분기 (`.item()`, `.any()` 뒤의 `if`) — 호스트 동기화이자 추적 불가

**이것이 CI 에 들어갑니다.** 새 아키텍처가 이 중 하나를 들고 들어오면 머지 시점에 드러납니다.
기기에서 터져서 알게 되는 것을 막는 장치이지, 범위를 재는 도구가 아닙니다.

### 그러면 A 와 B 는 무엇으로 정하나

**숫자가 아니라 순서로 정합니다.** 두 경로가 공유하는 일 — torch 파이썬 트리 벤더링, `import
transformers` 성립, `_C` 경계 확정 — 을 먼저 합니다. 그 일은 어느 쪽을 고르든 필요하고, 끝나면
양쪽에 대해 훨씬 많이 알게 됩니다.

그 시점에 남는 미지수는 하나뿐입니다 — **B 의 크로스 컴파일이 완료되는가.** 크기는 이미 공개된
숫자로 범위를 압니다 (선택 빌드 4.5~20MB, upytorch 압축 430KB, 전체 libtorch 50MB 초과 — §5).
그러니 B 는 **타임박스를 건 빌드 스파이크 한 번**으로 판정하고, 뚫리지 않으면 A 로 갑니다.

**결정을 미루는 것이 손해가 아닙니다.** 지금 정해도 공유 작업부터 해야 하고, 그 작업이 결정에
필요한 정보를 만들어 줍니다.
## 7. 런타임 특성

### 디코드는 살아남는다

배치 1 디코드는 산술 강도가 바이트당 1~2 FLOP 수준이라 메모리 바운드입니다. op 하나가 밀리초
단위로 도는 동안 파이썬 디스패치 오버헤드(마이크로초)는 그 밑에 숨습니다. ggml 의 인터프리터가
경쟁력 있는 것과 같은 이유이고, **이 논거가 우리 편입니다.**

### prefill 이 약점이다

거기서는 compute 바운드라 오버헤드가 숨지 않고, op 융합도 못 합니다.

TTT-Linear 기준 구체적으로: `adapt_step` 하나가 텐서 연산 25~30 개이고, `chunk_size` 16 에
2048 토큰이면 미니배치 128회 × 24레이어 = **forward 당 8만 회 안팎의 파이썬→네이티브 호출**입니다.

레버는 `chunk_size` 입니다 (16 → 128회, 64 → 32회). 다만 이것은 TTT 미니배치 크기라 **모델링
하이퍼파라미터**이고 `max_position_embeddings` 가 여기 묶여 있어 (`modular_ttt_linear.py:137`)
자유 변수가 아닙니다. 값을 바꾸려면 그 값으로 학습된 체크포인트가 필요합니다.

**그러므로 `adapt_step` 전체를 네이티브 커널 하나로 내리는 탈출구를 처음부터 설계에 넣어둘 것.**
당장 만들지는 않되, 나중에 끼워넣을 수 있는 형태로.

### 테스트 타임에 갱신되는 가중치가 양자화 가능 범위를 정한다

이것이 아키텍처 선택과 온디바이스 계획이 만나는 지점이고, **아키텍처가 진화할수록 나빠지는
방향**입니다.

델리게이트와 양자화 툴체인은 가중치를 상수로 취급합니다 — 미리 타일링해 패킹하고, 캘리브레이션을
끝내고, 컴파일된 blob 에 굽습니다. 그런데 테스트 타임에 갱신되는 텐서는 그 취급을 받을 수 없고,
누적이 수천 스텝 복리로 쌓이므로 int4 로 들고 있을 수도 없습니다 (분포가 입력에 따라 변해
캘리브레이션 대상 자체가 없습니다). **즉 갱신 대상이 되는 텐서는 양자화 밖으로 나갑니다.**

그 범위가 아키텍처마다 다릅니다.

| | 갱신되는 것 | 크기 |
|---|---|---|
| TTT-Linear / Titans | 헤드당 fast weight `[nh, d, d]` | 작음 |
| Nested Learning 의 CMS | **FFN 까지 포함** — "FFN 도 연관 메모리" | **파라미터의 대부분** |
| MAE (Memory As Embedding) | **미정 — 아래** | 확인 필요 |

**CMS 가 범위에 들어오면 계획이 크게 바뀝니다.** FFN 이 갱신 대상이면 폰에서 가장 큰 메모리 항목이
양자화 밖으로 나가고, 사전 패킹·상수 폴딩·weight-stationary 데이터플로우도 함께 못 씁니다.
`olympians` 가 설계 미정이므로 지금 결정할 것은 아니지만, **설계될 때 이 제약을 입력으로 넣어야
합니다.** 나중에 발견하면 양자화 파이프라인(§ 위)을 다시 짜게 됩니다.

**MAE 에 대해서는 조건부로만 적어 둡니다.** `TitansVariants` 에 `MAE = "mae"  # Memory As
Embedding` 이 LMM · MAC · MAG · MAL 과 나란히 1급 변형으로 선언되어 있고
(`titans/configuration_utils.py:10-15`), 이를 적용한 `cronos` 는 설계 중입니다. 설계를 모르는
상태에서 비용을 단정할 수 없으나, **임베딩 위치에 특유한 상호작용이 하나 있어 미리 짚어 둡니다.**

- 임베딩 테이블은 `vocab_size × hidden_size` 라 작은 모델에서 **단일 최대 텐서**인 경우가 많습니다.
- 그리고 이 저장소는 **가중치 묶기가 기본값**입니다 — `TTTLinearConfig.tie_word_embeddings = True`
  (`modular_ttt_linear.py:129`), `TTTLinearForCausalLM._tied_weights_keys =
  {"lm_head.weight": "model.embed_tokens.weight"}` (`:927`).
- **따라서 메모리가 임베딩 공간에 쓰이는 형태라면, 그 쓰기가 출력 투영도 함께 바꿉니다.**
  의도된 것이면 상관없지만, 의도되지 않았다면 MAE 에서는 묶기를 푸는 결정이 필요합니다.
- 온디바이스 관점에서는 어느 쪽이든 **가장 크고 가장 양자화하고 싶은 텐서가 갱신 대상이 되는지**가
  갈립니다. 설계가 확정되면 이 항목을 조건부에서 확정으로 옮길 것.

### 양자화는 torch 코어 밖에 있다

4비트 체크포인트는 bitsandbytes / torchao / GPTQ / AWQ / compressed-tensors 중 하나를 타는데
**전부 별개의 C 확장**이고, transformers 가 체크포인트 config 를 보고 거기로 디스패치합니다.
즉 `torch` 를 제공해도 4비트 모델은 안 돌아갑니다.

**해법:** 경로를 **하나만** 고르고 (candle 의 GGUF k-quant 가 유력), 그 op 을 Rust 계층에 넣고,
**Gradle 플러그인에서 빌드 타임에 체크포인트를 변환**합니다. `PythonMultiplatform` 에 아티팩트
워커가 이미 있으므로 자리가 맞습니다.

**대가:** 지원 매트릭스가 "HF 전체" 가 아니라 "빌드 타임에 우리 경로로 변환한 체크포인트" 가 됩니다.
괜찮은 트레이드지만 **지금 의식적으로 정할 일이지 8개월 차에 발견할 일이 아닙니다.**

---

## 8. 커널 전략 — 계약은 채용하고, 배포는 역전한다

flash-attention / flash-linear-attention 같은 융합 커널을 멀티플랫폼으로 제공하려면
[HuggingFace `kernels`](https://github.com/huggingface/kernels) 를 쓰는 것이 맞습니다.
다만 그 표준은 **분리 가능한 두 반쪽**으로 되어 있고, 우리에게 쓸모 있는 것은 한쪽뿐입니다.

### 채용할 반쪽 — 계약

`kernels` 의 계약은 우리 요구에 놀랄 만큼 잘 맞습니다.

- **백엔드 목록에 `metal` 이 이미 있습니다.** 지원 백엔드는 `cpu` · `cuda` · `metal` · `rocm` ·
  `xpu` 이고, 타입 목록에는 `cann` · `neuron` 도 있습니다. 즉 **CUDA 전용 표준이 아닙니다.**
- `kernel-builder` 의 `build.toml` 이 **하나의 커널에 대해 백엔드별 변형**을 선언하는 구조이고,
  백엔드별 의존성(`python-depends-backends`)까지 분리됩니다. 우리가 원하는 멀티플랫폼 패키징의
  형태가 이미 그것입니다.
- transformers 통합이 **모델 코드 수정 없이** 레이어를 교체합니다 (`from_pretrained(use_kernels=True)`,
  `use_kernel_forward_from_hub`). §1 의 "무수정으로 동작해야 함" 과 정확히 맞습니다.

**이 계약을 우리가 다시 발명할 이유가 없습니다.**

### 역전할 반쪽 — 배포

`get_kernel("kernels-community/activation")` 은 **런타임에 Hub 에서 사전 컴파일된 바이너리를
내려받아 캐시하고 로드**합니다. 이것이 모바일에서 성립하지 않습니다.

- **iOS**: 코드를 내려받아 실행할 수 없습니다. 네이티브 코드는 전부 서명된 앱 번들 안에 있어야
  하고, 임의 경로의 `dylib` 를 `dlopen` 하는 것은 금지됩니다. §5 에서 `torch.compile` 을 막는
  것과 같은 계열의 제약입니다.
- **Android**: 앱 전용 저장소에서 `dlopen` 은 기술적으로 되지만, 실행 코드를 내려받는 것은
  스토어 정책에 걸리고 서명되지 않은 네이티브 코드를 싣게 됩니다.

**해법: 해석 시점을 런타임에서 빌드 타임으로 옮깁니다.** Gradle 플러그인이 타깃별 변형을 골라
AOT 컴파일하고 앱 번들에 적재하며, 런타임에는 Hub 대신 **번들을 조회하는 리졸버**가 `kernels` 의
탐색 API 를 만족시킵니다. 위쪽 코드(transformers, thelethe)는 차이를 모릅니다.

이 단계가 §7 의 체크포인트 변환과 **같은 자리**입니다. 빌드 타임 해석 한 번이 모델과 커널을 함께
처리합니다.

**중요한 단순화 하나** — 역전이 필요한 것은 **모바일뿐입니다.** 데스크톱(macOS · Linux · Windows)
에서는 런타임 Hub 해석이 그대로 유효하고 `kernels-community` 의 상류 커널을 그냥 씁니다. 즉
리졸버 하나에 **소스가 둘**(Hub / 번들)이고, 플랫폼에 따라 고릅니다. 데스크톱 개발 경험이
상류와 동일하게 유지되는 것은 §1 의 "무수정" 전제에도 부합합니다.

### 타깃 매트릭스

"멀티플랫폼" 을 구체화하면 이렇습니다. `build.toml` 이 선언해야 할 변형이 곧 이 표입니다.

| 플랫폼 | `kernels` 백엔드 | 실제 커널 | 해석 시점 |
|---|---|---|---|
| Android arm64 (CPU) | `cpu` | NEON · dotprod · i8mm | **빌드 타임** |
| Android arm64 (GPU) | **없음** ← 아래 참조 | Vulkan compute | **빌드 타임** |
| iOS / iPadOS arm64 | `cpu` + `metal` | NEON + Metal | **빌드 타임** |
| macOS arm64 | `cpu` + `metal` | NEON + Metal | 런타임 (Hub) |
| Linux x86_64 | `cpu` + `cuda` | 상류 그대로 | 런타임 (Hub) |
| Windows x86_64 | `cpu` + `cuda` | 상류 그대로 | 런타임 (Hub) |

**Android GPU 에 구멍이 있습니다.** `kernels` 의 백엔드 열거는 `cpu` · `cuda` · `metal` ·
`rocm` · `xpu` (그리고 `cann` · `neuron`) 인데 **`vulkan` 이 없습니다.** Android 에는 Metal 이
없으므로 Android GPU 가속은 현재 표준 안에 자리가 없습니다. 선택지 셋:

1. **Android 는 CPU 만** — 가장 단순하고, §7 의 "디코드는 메모리 바운드" 논거상 손해가 생각보다
   작습니다. 1단계 기본값으로 적절합니다.
2. **상류에 `vulkan` 백엔드를 제안** — 표준에 남는 기여이지만 우리 일정에 종속되지 않습니다.
3. **로컬 확장으로 들고 감** — 빠르지만 상류와 갈라집니다.

**1 로 시작하고 2 를 병행하는 것을 권합니다.** 어차피 커널 구현은 사다리에서 핫스팟이 드러난
뒤이므로 (§11),
그 사이에 상류 논의를 열어둘 시간이 있습니다.

### flash-attention 은 prefill 에서만 의미가 있다

flash-attention 도 같은 취급을 받지만, 기대치를 정확히 잡아둘 필요가 있습니다.

FA2/FA3 의 실제 커널은 CUDA 라 모바일에서 쓸 수 없고, 필요한 것은 **NEON / Metal 위의 융합
어텐션**입니다. 통합 지점은 이미 있습니다 — transformers 가 `sdpa` 로 라우팅하므로 그 자리에
번들 커널을 물리면 됩니다.

다만 **이득이 나는 구간이 좁습니다.** flash-attention 의 본질은 S×S 어텐션 행렬을 메모리에
실체화하지 않는 것인데, **배치 1 디코드에는 쿼리 토큰이 하나뿐이라 애초에 S×S 행렬이 없습니다.**
따라서 기기에서 flash-attention 이 버는 것은 **prefill 과 긴 컨텍스트**이고, 디코드에서는 거의
없습니다.

이 결론이 §7 · §8 전체와 같은 방향입니다 — **커널 작업의 값어치는 전부 prefill 에 있습니다.**
디코드는 메모리 바운드라 이미 상한에 가깝고, 커널을 아무리 갈아도 대역폭이 안 늘어납니다.

### Triton 은 iOS 에서 불가능하다

**`flash-linear-attention` 은 Triton 기반입니다.** 그리고 Triton 은 JIT 입니다 — 런타임에
PTX/LLVM 으로 컴파일합니다. 따라서:

- **iOS**: 원리적으로 불가능 (런타임 코드 생성 금지)
- **Android**: Triton 런타임도, 인앱 GPU 컴파일러 툴체인도 없음
- 설령 가능하더라도 Triton 의 autotuning 워밍업은 **세션이 짧은 기기에서 상각되지 않습니다.**

그러므로 **"모바일에서 flash-linear-attention 을 지원한다" 는 FLA 의 Triton 커널을 돌린다는
뜻일 수 없습니다.** 같은 *융합 연산 집합* 을 AOT 컴파일된 커널로 제공한다는 뜻이어야 합니다.

선례가 있습니다 — FlexLA (ICLR 2026) 가 **Triton 위에 정적 커널 디스패처를 얹은 AOT 컴파일**로
런타임 오버헤드를 없앴습니다. AOT 방향은 연구된 경로이지 즉흥적인 발상이 아닙니다.

### 커널 소스 후보

| 소스 | 장점 | 단점 |
|---|---|---|
| 손으로 쓴 Metal / NEON | 최고 성능. `kernels` 의 `metal` · `cpu` 변형에 그대로 맞음 | 타깃마다 따로 씀 |
| **CubeCL** (burn 의 커널 언어) | Rust 안에서 한 번 쓰고 WGSL · SPIR-V · Metal · CUDA 로 컴파일 | 성숙도 확인 필요. burn 생태계 결합 |
| AOT 컴파일한 Triton | FLA 의 커널 정의를 재사용 | 툴체인 무겁고, 모바일 백엔드 커버리지가 관건 |

§4 에서 burn 을 shim 백엔드로는 기각했지만, **CubeCL 은 여기서 다시 후보가 됩니다.** 텐서 API 의
정적 랭크 문제와 무관한 계층이기 때문입니다.

### 그리고 이것이 prefill 탈출구다

§7 에서 "`adapt_step` 전체를 네이티브 커널 하나로 내리는 탈출구를 설계에 넣어둘 것" 이라고 했는데,
**그 탈출구가 바로 이 기제입니다.** 새로 만들 것이 아니라 TTT 적응 레이어에
`use_kernel_forward_from_hub` 를 걸고 번들된 융합 커널을 물리면 됩니다.

즉 커널 계층은 "나중에 추가할 최적화" 가 아니라 **prefill 문제의 예정된 해법**이고, 그래서 지금
계약을 정해둘 가치가 있습니다. 커널 자체는 나중에 써도 됩니다.

---

## 9. 기존 코드에서 나온 선결 과제

경로(A/B) 와 무관하게 옳은 것들입니다.

### `theRiverLethe`

| # | 위치 | 문제 |
|---|---|---|
| 1 | `thelethe/ops/normal_scan.py:214` | `compiled_scan = torch.compile(scan, mode="max-autotune")` 가 **모듈 스코프**. `ops/__init__.py` 가 export 하므로 import 시 실행됨. `torch.compile` 심볼이 없으면 import 단계에서 죽고, 호출되면 Inductor 가 런타임 C++ 컴파일러를 부름 → **iOS 불가**. `max-autotune` 은 커널 변형을 런타임 벤치마크하므로 그중 최악. **→ 지연 생성 + 플랫폼 가드** |
| 2 | `modular_ttt_linear.py:374-376` | `torch.isnan(bias).any()` 를 파이썬 `if` 로 분기. `rasterize` → `step()` → scan 반복마다 호출되어 **forward 당 3000회 넘는 호스트 동기화**. 게다가 데이터 의존 분기라 **export 를 영구히 막음**. bias 유무는 정적 정보이므로 **→ NaN 센티넬 대신 `use_bias` 불리언을 상태에 동반** |
| 3 | `normal_scan.py:11` | `from torch._higher_order_ops import scan` 이 아래 `def scan` 에 가려진 **죽은 import**. 그래도 모듈 경로가 존재해야 통과. **→ 삭제** |
| 4 | `modular_ttt_linear.py:319` | `torch.einsum` — 문자열 파싱 + 일반화 축약이라 shim 구현 비용이 큼. 여기서는 `"bhkc,hcd->bhkd"` 한 패턴뿐. **→ matmul 로 재작성** |

### `test-time-adapters`

| # | 위치 | 문제 |
|---|---|---|
| 5 | `methods/base.py:117` | `base_state` 가 전체 가중치의 CPU 사본을 엔진 수명 내내 보유. **폰에서 모델 메모리 2배**이고, 모바일은 CPU/GPU 가 같은 물리 RAM 이라 오프로드가 아니라 순수 중복. `reset()` 은 StandardTTA/GradualTTA **벤치마크 요구**이지 제품 요구가 아님. **→ 기기에서는 디스크 재로드로** |
| 6 | `methods/base.py:152-156` | `self._dtype = torch.dtype(*args, **kwargs)` — `torch.dtype` 은 생성자로 호출 불가라 항상 TypeError 이고 `except TypeError: pass` 가 삼킴. **`_dtype` 이 init 값에 고정**되고 `reset()` 이 `:254` 에서 그 값으로 되돌림. 데스크톱 fp32 고정에서는 안 보이지만 기기에서 fp16/fp32 를 오가면 즉시 물림 |
| 7 | `methods/base.py:37-55` | `torch.cuda.manual_seed()`, `torch.backends.cudnn.*` 가 가드 없이 호출됨 |
| 8 | `methods/base.py:180-191` | Muon / MuonWithAuxAdam 이 스텝마다 Newton-Schulz 직교화 반복(matmul 다회). `muon` 이 추가 의존성. **→ 기기 기본값은 SGD, Muon 은 데스크톱 게이팅** |

### carry in-place 에 관한 정정

`TTTLinearAdaptationState.step()` 이 매번 새 상태를 만드는 것은 **바깥쪽 학습(단계 2)의 역전파를
위해 필요하며 맞게 작성된 것입니다.** in-place 로 바꾸면 `W_{t-1}` 을 복원할 수 없어 PyTorch 가
버전 카운터로 잡아냅니다.

다만 판정 기준이 `self.training` 이 아닙니다. **`eval()` 은 grad 추적을 끄지 않습니다** —
파라미터의 `requires_grad` 가 True 면 eval 에서도 테이프가 쌓입니다. 올바른 술어는
`torch.is_grad_enabled()` 입니다.

```python
next_carry = carry.step(deltas) if torch.is_grad_enabled() else carry.step_(deltas)
```

`GenerationMixin.generate` 가 이미 `@torch.no_grad()` 로 감싸져 있으므로 기본 추론 경로는 분기만
넣으면 혜택을 받습니다. `inference_mode` 는 `no_grad` 보다 강해 그 안에서 만든 텐서를 이후
autograd 에 쓸 수 없으므로, **online/offline 을 오가는 `ttadapters` 에는 `no_grad` 가 안전합니다.**

**얻는 것의 크기는 크지 않습니다.** `memory_depth` 가 1 이라 스텝당 파이썬 객체는 2~3 개이고,
`no_grad` 하에서는 이전 carry 의 참조가 즉시 끊겨 allocator 가 블록을 재사용합니다. 실질적인 이득은
할당 제거보다 **항목 2 의 호스트 동기화가 같이 사라지는 쪽**입니다. prefill 비용의 본체는 디스패치
횟수이고 이것으로는 안 풀립니다 (§7).

### 이 저장소

- `torchnative/api/__init__.py:4` — `def __init__(self, *args, *kwargs)` 는 **SyntaxError**
  입니다 (`**kwargs` 여야 함). 현재 이 모듈은 import 되지 않습니다.
- `torchnative/nn/federated/__init__.py:1` — `from . import DistributedDataFederated` 인데 해당
  모듈이 디렉터리에 없어 ImportError 입니다.

둘 다 blank template 단계라 의도된 미완성일 수 있으나, 첫 실행 전에 걸립니다.

> **Correction (문서 감사, 재측정 2026-09): 둘 다 고쳐졌습니다.** `torchnative/api/__init__.py` 는
> 지금 `**kwargs` 문법 오류 없이 컴파일되고(`py_compile` 로 재확인), 실제로는 docstring 하나뿐인
> 지연 임포트 모듈입니다. `torchnative/nn/federated/__init__.py` 도 `DistributedDataFederated`
> 임포트가 사라지고 docstring 만 남아, `import torchnative.nn.federated` 가 예외 없이 성공합니다
> (재확인). 어느 커밋이 고쳤는지는 추적하지 않았다 — 이 문서 감사의 범위는 현재 상태 확인까지다.
>
> Standing check (docs/verification/DOCWATCH.md) — path is `torchnative/src/main/torchnative/...` in this
> tree, not the bare `torchnative/...` §9 above writes (the vendored tree's own root):
> <!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/nn/federated/__init__.py DistributedDataFederated absent -->
> <!-- DOCWATCH: symbol-in-file torchnative/src/main/torchnative/api/__init__.py "*args, *kwargs" absent -->

---

## 10. 저장소 구성과 빌드

빌드 도구는 [`pypackpack`](https://github.com/thisisthepy/pypackpack) 입니다 —
`crossenv + compiler(nuitka) + bundler + codepush` 로, Android(arm64 · x86_64) · iOS(arm64) ·
macOS · Linux · Windows · WASM 을 대상으로 합니다. **C · C++ · Rust 확장 빌드를 지원**하므로
`torch._C` 를 크로스 컴파일할 도구가 이미 있습니다.

> **이 여섯과 §722 의 다섯은 다른 것을 세고 있습니다** (2026-08-29 에 명시).
> 여기 여섯은 **pypackpack 이 배포할 수 있는 플랫폼**이고, §722 의 표는 **`kernels` 백엔드
> 매트릭스** — 각 플랫폼에서 어느 가속 백엔드가 해석되는가 — 입니다. WASM 이 §722 에 없는 것은
> **`kernels` 에 wasm 백엔드가 없기 때문**이고, 같은 이유로 `vulkan` 도 없습니다(§722 가 그 구멍을
> 이미 적어놨습니다).
>
> 그러니 **§722 에 WASM 을 넣는 것은 권하지 않습니다.** 넣으면 "가속 백엔드가 있다" 는 뜻이 되는데
> 없습니다. `docs/platform/WASM.md` 가 네 층을 다 재서 그 판단의 근거를 담고 있습니다 — candle 은
> `wasm32` 에서 빌드되고(오히려 crates 129 → 80 으로 줄어듭니다), PyO3 의 `abi3` + 확장 모듈도
> 되지만, **WASI 는 `dlopen` 이 없어 휠이 성립하지 않고** Emscripten 은 되는 대신 `abi3` 를
> 무효로 만듭니다.

그리고 **pypackpack 스펙에 이미 torchnative 가 들어 있습니다.** 배포 채널이 셋으로 갈라져 있고
그중 가중치를 torchnative 가 맡습니다.

| pypackpack `deploy/` | 무엇 |
|---|---|
| `code/FastTrackAPI.kt` | 코드 배포 |
| `resource/ResourceHubAPI.kt` | 리소스 배포 |
| **`weight/TorchNativeAPI.kt`** | **가중치 배포 클라이언트** |

즉 `torchnative/api/TorchNativeAPI` 는 그 **기기 쪽 상대편**입니다.

### 디렉터리

```
torchnative/
├─ pyproject.toml                루트 워크스페이스, [tool.ppp] 타깃
├─ uv.lock  .python-version
│
├─ torchnative/                   pypackpack 패키지 — 하나
│  ├─ pyproject.toml
│  └─ src/
│     ├─ main/                   ← 여기를 스캔해 최상위 파이썬 패키지를 찾음
│     │  ├─ torch/               최상위 `torch` 로 임포트됨
│     │  │  ├─ …                 벤더링한 상류 파이썬 트리 (BSD)
│     │  │  ├─ _decomp/          Core ATen 분해표
│     │  │  ├─ _C/               Rust + PyO3 → candle
│     │  │  └─ nn/federated.py   add-hook. 편의이지 의존이 아님 (§2)
│     │  └─ torchnative/
│     │     ├─ delta/            핵심 추상. 수명 이름 미정 (§3)
│     │     ├─ adapt/            TTL 방법
│     │     ├─ nn/federated/     FL
│     │     ├─ kernels/          번들 리졸버 (§8)
│     │     └─ api/              TorchNativeAPI
│     ├─ android/  ios/  macos/  linux/  windows/
│     └─ test/
│
├─ tools/scan/                   정적 스캔 (§6). 배포물 아님, CI 전용
└─ docs/design/DESIGN.md
```

### 왜 한 패키지인가

`torch` 는 **최상위 이름 `torch` 로 임포트되어야 합니다** — §1 의 전제가 그것입니다. pypackpack 이
`src/main` 을 스캔해 최상위 파이썬 패키지들을 찾으므로 (SPEC.md:427), 한 패키지가 `torch` 와
`torchnative` 을 함께 제공할 수 있습니다.

**별도 저장소로 빼지 않습니다.** torchnative 는 `theRiverLethe` 와 `ttadapters` 를 참조하지 않고
(소스 내 참조 0 건), 아키텍처는 §3 에서 정리한 대로 torchnative 의 *입력*이므로 의존 방향이 생기지
않습니다. `PythonMultiplatform` 이 CPython 배포본을 `binary/` 에 두어 라이브러리와 같은 저장소에서
관리하는 것과 같은 형태입니다.

### FL 의 결합은 extras 로 막는다

"TTA 만 쓰는 사람이 집계 · 통신 · 프라이버시 스택을 끌고 오지 않는다" 는 요구는 유효하되, 패키지
경계가 아니라 `[project.optional-dependencies].federated` 로 지킵니다. 네트워킹 · 암호 의존성이
기본 설치에 딸려오지 않게 합니다.

### `adapt/` 는 디렉터리로 단계를 나누지 않는다

미분 요구를 디렉터리 이름으로 판정할 수 없습니다 — 정규화 보정이 단계 0 과 1 에 걸칩니다 (§3).
**각 방법이 자기 요구를 선언하고 빌드가 그 선언으로 거릅니다.** backward 없는 기기 빌드에 단계 1
방법이 들어오면 임포트 시점에 걸립니다.

### 배포 채널을 섞지 않는다

pypackpack 의 코드 fast-track 은 **파이썬(소스 · 바이트코드)에만** 씁니다. iOS 가 내려받은
네이티브 코드의 실행을 금지하므로 — §8 에서 `kernels` 의 Hub 해석을 빌드 타임으로 역전시킨 것과
같은 제약 — **`torch._C` 와 융합 커널은 fast-track 대상이 아니라 번들에 구워야 합니다.**

| 무엇 | 채널 |
|---|---|
| 파이썬 계층 (`torchnative`, 벤더링한 `torch/` 트리) | fast-track 가능 |
| `torch._C`, 융합 커널 | **번들만** |
| 모델 가중치 | TorchNativeAPI |

### 빌드 레벨은 두 번째 손잡이다

pypackpack 이 `instant(.py) / bytecode / native / mixed` 를 지원하므로, 릴리스에서 `adapt/` 의
핫 루프를 nuitka 로 네이티브에 내릴 수 있습니다. §7 의 prefill 디스패치 비용에 대한 손잡이가
둘이 되는 셈입니다 — 첫째는 융합 커널(§8), 둘째가 이것.

### Rust 배선은 아직 없다 — 이것이 1 단계보다 앞선다

확인 결과 **pypackpack 에 Rust 확장 빌드가 구현되어 있지 않습니다.**

| | 상태 |
|---|---|
| `compile/backend/external/Meson.kt` | 322 줄, 구현됨 |
| `compile/backend/external/Cargo.kt` | **4 줄 — 패키지 선언과 주석뿐** |

SPEC.md:427 의 meson 자동 생성도 `.c` · `.cc` · `.cpp` 만 `py.extension_module()` 로 만들고
Rust 는 다루지 않습니다. 그러므로 **`torch/_C/` 는 지금 상태의 pypackpack 으로 빌드되지 않습니다.**

선택지가 둘입니다.

| | 내용 | 성격 |
|---|---|---|
| **A. `Cargo.kt` 를 구현** | pypackpack 에 Rust 백엔드를 채움 | **어댑터가 없어 함께 만들어야 함 — 아래 참조** |
| **B. 밖에서 빌드해 산출물만 넘김** | `cargo` + `cargo-ndk` 로 `.so` · `.dylib` 를 만들고 pypackpack 은 패키징만 | 빠르게 뚫음. 크로스 컴파일 설정을 이중 관리하게 됨 |

> **정정.** 위에서 A 를 "`Clang` · `NDK` · `XCode` 어댑터 위에 얹는 이미 설계된 형태" 라고 적었는데,
> **그 어댑터들이 존재하지 않습니다.** 실제로 세어 보니 `compile/backend/external/` 에서 구현된
> 것은 `Meson.kt`(322 줄) 하나뿐이고 **`Cargo.kt` · `Clang.kt` · `NDK.kt` · `XCode.kt` ·
> `MSVC.kt` · `Emscripten.kt` 가 전부 4 줄짜리 스텁**입니다. SPEC.md 의 어댑터 설명은 의도이지
> 배선이 아닙니다. 즉 A 는 "빈 칸 하나 채우기" 가 아니라 **크로스 컴파일 백엔드 계층 전체를
> 만드는 일**입니다.
>
> 그리고 `Meson.kt` 는 `BackendInterface` 를 구현하지도 않습니다 — `DefaultBackend` 에 직접
> 합성되는 `open class` 이고, `BackendType` enum 은 `MESON` 하나뿐인 채 참조되지 않습니다.
> **팩토리가 없습니다.**
>
> 타깃 디스패치에 걸리는 실제 결함도 둘 확인했습니다 (로컬 `rustc --print target-list` 로 대조).
>
> | 위치 | 문제 |
> |---|---|
> | `utils/Platforms.kt:74-75` | `arm64-apple-ios` · `arm64-apple-ios-simulator` — **rustc 가 모르는 트리플**. 실제는 `aarch64-apple-ios` · `aarch64-apple-ios-sim` |
> | `utils/Platforms.kt:100,102` | `android_21_arm64` 와 `android_24_arm64` 가 같은 트리플로 뭉개져 **API 레벨이 소실**. `cargo-ndk --platform` 이 그 값을 요구함 |
>
> **B 를 먼저 하라는 권고는 그대로이고, 근거가 더 강해졌습니다.** 설계안은 `docs/design/CARGO_KT.md`.

**B 로 뚫고 A 로 수렴하는 것을 권합니다.** `torch._C` 의 첫 스파이크가 Rust 툴체인 문제로 막히면
안 되고, 반대로 A 를 먼저 하면 아직 존재하지 않는 크레이트를 위해 백엔드를 설계하게 됩니다.
B 로 한 번 통과시켜 필요한 것이 드러난 뒤에 옮기는 편이 백엔드 설계도 정확해집니다.

**이것이 §11 의 1 단계보다 앞섭니다** — `torch._C` 가 빌드되지 않으면 스텁조차 세울 수 없습니다.

---

## 11. 순서

**부트스트랩 → 사다리 → 이식.** 3 단계까지는 KMP 도 기기도 건드리지 않습니다.
**선행 측정 단계가 없습니다** — 각 단계가 다음 단계에 필요한 것을 스스로 만들어 냅니다 (§6).

| # | 할 일 | 산출물 |
|---|---|---|
| 1 | 벤더링한 torch 파이썬 트리 + 빈 `_C` 스텁으로 `import transformers` 시도 | import-time 요구사항 전체 목록 |
| 2 | 정적 스캔을 CI 에 넣기 (§6) | `torch.compile` · CUDA · 사설 API · 텐서 값 분기의 조기 경보 |
| 3 | 사다리 1~2 단 — 가장 작은 모델을 `torch._C` 로 통과, 진짜 torch 와 골든 대조 | 수치 의미론 리스크의 실제 크기, 첫 op 우선순위 |
| ~~4~~ | ~~B 의 크로스 컴파일 스파이크~~ | **완료 — A 로 결정** (§5). 빌드는 됐으나 모바일 경로가 `BUILD_PYTHON` 을 강제로 끄므로 B 는 `torch._C` 를 만들지 못함 |
| 5 | 사다리 3~5 단 | 아키텍처별 한계 비용 |
| 6 | 기기 (Android 먼저 — iOS 보다 제약이 적음) | |
| 7 | GraalVM 네이티브 이미지 경로 | `PythonMultiplatform` 의 요구사항 |

### 11.1 1 단계가 멈춘 지점과, 거기서 나온 순서 (2026-08-24 갱신)

> **(2026-08-25 추기) 이 벽은 열렸습니다 — `docs/distributed/DISTRIBUTED.md`.** 아래 문단이 예상한 그대로,
> `torch.distributed` 를 `world_size = 1` 부터 실체로 구현하면서 부수 효과로 열렸고, 벤더 트리는
> 한 줄도 고치지 않았습니다. `from transformers import AutoModelForCausalLM` 과
> `AutoModelForCausalLM.from_config(...)` 이 통과합니다 — 손으로 옮겨 적은 것이 아닌 **진짜
> transformers 모델 객체**가 만들어집니다.
>
> **다만 순전파는 아직입니다.** 다음 벽은 분산이 아니라 `torch._C.is_autocast_enabled`
> (`transformers/utils/generic.py:250`) 이고, 따라서 바로 아래 문단이 말하는
> `from_pretrained` · 실제 체크포인트 경로는 **여전히 한 번도 실행되지 않았습니다.**
>
> > **Correction (문서 감사, 재측정 2026-09): 이 벽도 열렸고, `from_pretrained` 는 지금 실행됩니다.**
> > `docs/models/COMPAT.md` 가 `torch._C._is_autocast_available`(및 관련 다섯 개)을 구현해 이 자리의
> > autocast 벽을 닫았습니다. 재확인 완료 — 오늘의 빌드에서 `AutoModelForCausalLM.from_pretrained
> > ("HuggingFaceTB/SmolLM2-135M")` 이 실제 Hub 체크포인트를 내려받아 로드하고, forward 와
> > `generate()` 둘 다 성공합니다(직접 실행, 로짓 shape `(1, 3, 49152)`, `generate` 8 토큰 산출).
> > `docs/devices/META.md` §8.3/§7 도 같은 경로(`from_pretrained` + llama3 rope 초기화)를 상류와 로짓
> > 1e-5 이내로 대조해 독립적으로 확인해 두었다. 아래 두 단락("1 단계가 완료되지 않았습니다" ·
> > "`from_pretrained` 와 실제 체크포인트 경로는 아직 한 번도 실행되지 않았습니다")은 이 correction
> > 이전 시점의 서술로 남겨 두었다 — 무엇이 막혀 있었고 무엇이 그것을 열었는지의 기록으로서다.

**1 단계가 완료되지 않았습니다.** `import transformers` 가 아직 안 됩니다 —
`torch.distributed.Store` 재수출이 끊겨서이고, 그 진단과 결정은 `docs/design/SURFACE_HONESTY.md` §2 에
있습니다. **벤더 트리에 패치를 대지 않기로 정했으므로**, 이 벽은 `torch.distributed` 를
`world_size = 1` 부터 실체로 구현하면서 부수 효과로 열립니다.

그래서 3 · 5 단계의 "골든 대조" 는 지금까지 **전부 손으로 옮겨 적은 모델**로 해 왔습니다.
아키텍처 20 개 중 15 개가 미구현 op 0 이고 Llama · GPT-2 가 상류와 토큰·로짓이 맞지만,
**`from_pretrained` 와 실제 체크포인트 경로는 아직 한 번도 실행되지 않았습니다.** 그 둘은
이 벽 뒤에 있습니다.

**분산은 우회가 아니라 범위 안입니다.** FL 이 처음부터 목표에 있었고 연합 학습은 집합 통신
위에 섭니다 — `broadcast` · `gather` · 가중 `all_reduce` 가 곧 FedAvg 입니다.

**계획한 스택** (위가 아래에 의존):

```
torchnative.nn.federated   라운드 · 클라이언트 선택 · 집계 · 이탈 처리
  └ torch.distributed     ProcessGroup · 집합 통신 (전송 추상)
      └ 백엔드             register_backend 로 우리 것
          └ 장치 추상       CPU · Metal · Vulkan · NPU
```

**공통 기반은 분산 표면이 아니라 그 아래 장치 추상입니다.** 랭크가 장치를 가리키려면
`torch.device` 와 장치별 디스패치가 먼저 있어야 하고, 가속기도 전부 그 위에 얹힙니다.
분산을 먼저 세워도 랭크가 가리킬 것이 없으면 껍데기입니다.

가속기의 실제 지형 — **`candle` 에 `metal` feature 가 있고**, 우리 `Cargo.toml` 이 그것을 끈
것은 능력 부재가 아니라 "상류의 미래 기본값이 조용히 링크하지 못하게" 하는 격리 목적이었습니다.
되돌릴 수 있는 결정입니다.

| 가속기 | 상태 | 필요한 일 |
|---|---|---|
| Apple GPU (Metal) · Accelerate | candle 에 있음, 우리가 껐음 | feature + 장치 개념 |
| 안드로이드 GPU | candle 에 백엔드 없음 (`kernels` 표준에도 `vulkan` 슬롯 없음) | Vulkan/wgpu 백엔드 도입 또는 작성 |
| NPU (ANE · NNAPI · QNN) | **구조가 다름** | 그래프 캡처 층 |

**NPU 가 진짜 설계 문제입니다.** 그것들은 즉시 실행 장치가 아니라 **그래프를 통째로 받아 미리
컴파일하는 실행기**라, op 단위로 `_aten_dispatch` 를 지나는 이 구조와 정면으로 어긋납니다.
다만 **문이 하나라는 것이 캡처에 유리합니다** — 모든 op 이 반드시 한 곳을 지나므로 그 자리가
부분 그래프를 기록하기에 맞습니다. `_C._dynamo` 를 no-op 으로 둔 것(§7)과 모순되지 않습니다.
캡처가 필요해지면 상류 것을 켜는 것이 아니라 **우리 문에서 우리가** 합니다.

**1 단계에서 나오는 벽의 개수가 이 계획의 실현 가능성을 거의 다 말해줍니다.** 그리고 이제 이것이
맨 앞입니다 — 가장 값싸고 가장 많이 알려주며, A 와 B 어느 쪽을 고르든 필요한 일입니다.
`accelerate` 가 무조건 `import torch` 를 하는 것 같은 결합은 우리에게 유리합니다 — 그 이슈의
사람들은 torch 를 *피하려* 했고 우리는 *만족시키려는* 것이므로 방향이 반대입니다.

**A/B 결정이 4 번으로 내려간 것이 의도입니다.** 1~3 을 끝내면 양쪽에 대해 훨씬 많이 알게 되고,
그때 남는 미지수는 "B 의 빌드가 뚫리는가" 하나뿐입니다.

### 열려 있는 성능 결함 — `linear` 이 매 호출마다 가중치를 옮겨 적는다 (2026-08-28)

**순서와 무관하게 지금 가장 큰 것입니다.** `docs/graph/QUANT2.md` §6.4 가 양자화를 재다가 찾았고,
조율 세션이 상류와 직접 대조해 확인했습니다:

```
F.linear, lm_head 모양 (49152 × 576, 가중치 113 MB)
  seq=1    상류 2.83 ms   우리 86.6 ms    30.6 배
  seq=8    상류 4.70 ms   우리 98.2 ms    20.9 배
```

원인: `bootstrap.py` 가 `t(weight)` 를 넘기고 `aten.matmul.default` 가 두 피연산자에
`.contiguous()` 를 부릅니다. **매 순전파마다 113 MB 를 옮겨 적습니다.**

**고치지 않았고, 그것이 판단입니다.** 고치면 BLAS 에 넘기는 레이아웃이 바뀌고 그러면 누산
순서가 바뀔 수 있는데, 골든은 비트 비교입니다. 즉 이건 패치가 아니라 **"정확성과 속도 중
무엇을 약속하는가" 의 결정**이고, `docs/kernels/SDPA.md` §12 가 sdpa 에서 내린 것과 같은 종류입니다.
거기서는 옵트인으로 갈랐습니다.

> **Correction (문서 감사, 재측정 2026-09): 지금은 고쳐져 있습니다.** `docs/perf/LINEAR.md`
> (2026-08-29) 가 이 결함을 정확히 진단하고("접기" + 레이아웃 폴백 두 가지), 비트가 상류
> 쪽으로만 움직인다는 것(507 케이스 중 48 개 변경, 상류 이탈 0 · 상류 일치로 개선 35)을 재서
> 두 갈래 선택지(그대로 착지 vs `docs/kernels/SDPA.md` 식 옵트인)를 제시했다. `git log -S` 로 확인:
> 커밋 `2e00ec3` "Perf: Fold instead of broadcasting, and stop copying the weight every call"
> 가 그 변경을 **옵트인이 아니라 기본값으로 착지**시켰다 — `rust/torch_c/src/aten.rs`
> 의 `gemm_with_layout_fallback`/`batched_matmul` 가 지금 이 빌드에 있고,
> `lhs.contiguous()?, rhs.contiguous()?` 가 candle 이 스트라이드 실패로 거절할 때만 도는
> 폴백으로 바뀌어 있다(재확인). LINEAR.md §6 에 이후에도 남은 벽 다섯 개(bf16/f16 가중치가
> 여전히 호출마다 넓혀짐 등)가 이름으로 적혀 있다.
>
> Standing check (docs/verification/DOCWATCH.md):
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs gemm_with_layout_fallback present -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs batched_matmul present -->

**이 결함이 다른 수치를 오염시킵니다.** `docs/graph/QUANT2.md` 의 "27 배" 는 대부분 양자화가 아니라
이 복사를 피한 것입니다. 전치 복사가 없는 밀집 경로와 비교하면 양자화는 이 호스트에서
`lm_head` 소배치에서만 이깁니다.

### 이 순서와 병행할 수 있는 것

커널 계층(§8)은 **계약만 먼저 정하고 구현은 뒤로 미룰 수 있습니다.** 오히려 그래야 합니다 —
번들 리졸버가 만족시켜야 할 `kernels` 탐색 API 는 지금 확정 가능하고, 실제 융합 커널은 사다리에서
핫스팟이 드러난 뒤에 쓰는 것이 맞습니다. 지금 커널부터 쓰면 최적화할 대상을 모르는 채로 쓰게 됩니다.

축 정리(§3)도 병행 가능합니다. **수명이 타입에 박힌 가중치 델타**를 먼저 정의해두면 `ttadapters`
의 `base_state` 문제(§9-5)가 이식 전에 해소되고, FL · TTL 이 나중에 붙을 자리가 생깁니다.

§9 의 1~4, 6~7 은 이 순서와 무관하게 지금 고쳐도 되는 것들이고, 5 와 8 이 설계 판단입니다.

---

## 부록: 기각된 대안과 근거

| 대안 | 기각 사유 |
|---|---|
| **llama.cpp 바인딩 + transformers 파사드** | §1 의 전제를 위반. 그리고 `ttadapters` 의 비전 모델을 하나도 커버하지 못함 |
| **micrograd / hypergrad 계열에서 출발** | 참조한 hypergrad 는 커밋 5개 · 스타 11개의 **스칼라 값** autograd 로, 텐서도 BLAS 도 없음. 거기서 "폰에서 int4 로 Llama" 까지는 증분이 아니라 ggml/candle 코드베이스 전체 |
| **torch API 를 처음부터 완전 복제** | transformers 는 `isinstance(x, torch.Tensor)`, dtype promotion, view/stride aliasing, `__torch_function__` 에 의존 — 동작 호환이 아니라 **버그 호환**이 요구됨. 게다가 transformers v5 가 TF/Flax 를 유예 없이 삭제했고 명시 사유가 "유지 비용" — **풀타임 팀의 HuggingFace 본인이 다중 백엔드 패리티를 포기함**. 그리고 그 결정 방향이 "추상화를 걷어내는 것" 이라 모델링 코드는 앞으로 torch 에 **더** 밀착함 |
| **tch-rs 사용** | (a) libtorch 가 타깃에 있어야 함 (b) **화살표 방향이 반대** — tch-rs 는 torch 를 *Rust 에* 노출하는데, 우리는 Rust 엔진을 *파이썬에* torch 얼굴로 노출해야 함. 만들 것은 크레이트가 아니라 `torch._C` 자리의 PyO3 확장 모듈 |
| **burn 을 shim 백엔드로** | `Tensor<B, D>` 정적 랭크가 torch 의 동적 랭크와 근본적으로 불일치. FL 학습 트랙에서는 여전히 후보 |
| **torch-xla / torch-mlir** | torch-xla 는 libtorch 를 요구하는 **런타임 확장**, torch-mlir 은 AOT **컴파일러 프론트엔드**로 결과가 컴파일된 아티팩트 — 둘 다 "기기에서 진짜 파이썬 transformers" 가 아님. 다만 각각 훔칠 것이 있음: 전자는 "한 계층에서 가로채면 위가 전부 따라온다" 의 대규모 증명, 후자는 "aten 수천 개가 작은 핵심 집합으로 분해된다" 의 독립 검증 (TOSA 목록이 특히 하드웨어 지향) |
| **`optimum-executorch` 로 `.pte` 배포** | 이미 공식으로 존재하고 잘 동작하지만, 데스크톱에서 export 한 아티팩트를 배포하는 것이라 §1 의 전제와 다름. §5 의 Core ATen 계약을 공유하므로 **나중에 무거운 모델만 빼는 하이브리드로는 열려 있음** |
| **`kernels` 를 런타임 Hub 해석 그대로 사용** | iOS 가 내려받은 네이티브 코드의 실행을 금지하고 Android 도 스토어 정책에 걸림. **계약은 채용하고 해석 시점만 빌드 타임으로 옮김** (§8) |
| **FLA 의 Triton 커널을 기기에서 실행** | Triton 은 런타임에 PTX/LLVM 으로 컴파일하는 JIT — iOS 에서 원리적으로 불가능하고, 짧은 세션에서 autotuning 이 상각되지 않음. **같은 융합 연산 집합을 AOT 커널로 제공** (§8) |

---

## 참고

- [Core ATen Operator Set](https://docs.pytorch.org/executorch/stable/ir-ops-set-definition.html)
- [PyTorch's Tracing Based Selective Build](https://pytorch.org/blog/pytorchs-tracing-based-selective-build/)
- [`pytorch/scripts/build_mobile.sh`](https://github.com/pytorch/pytorch/blob/v2.8.0/scripts/build_mobile.sh) — **main 에서 삭제됨** (커밋 `91602a92548d`). v2.8.0 링크
- [`ljk53/upytorch`](https://github.com/ljk53/upytorch)
- [huggingface/candle](https://github.com/huggingface/candle) · [tracel-ai/burn](https://github.com/tracel-ai/burn) · [LaurentMazare/tch-rs](https://github.com/LaurentMazare/tch-rs)
- [torch-mlir architecture](https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md)
- [transformers v5 Migration Guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)
- Wang, Luo, Zheng, Chen, Wang, Huang — [*In Search of Lost Online Test-Time Adaptation: A Survey*, IJCV 133:1106–1139 (2025)](https://doi.org/10.1007/s11263-024-02213-5) — §3 의 TTL ⊃ TTA ⊃ TTT 정의와 시나리오·미분 요구 분류의 출처
- Behrouz et al. — *Nested Learning: The Illusion of Deep Learning Architectures*, Google Research, NeurIPS 2025 — §7 의 CMS·FFN 갱신 논의의 출처
- [huggingface/kernels](https://github.com/huggingface/kernels) · [Kernel requirements (백엔드 목록)](https://huggingface.co/docs/kernels/kernel-requirements) · [Writing Hub kernels with kernel-builder](https://huggingface.co/docs/kernels/en/builder/writing-kernels) · [Integrating kernels](https://huggingface.co/docs/kernels/integrating-kernels)
- [transformers — Loading kernels](https://huggingface.co/docs/transformers/kernel_doc/loading_kernels)
- [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- [FlexLA: AOT compilation with a static kernel dispatcher on Triton (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/file/d029c97ee0db162c60f2ebc9cb93387e-Paper-Conference.pdf)
