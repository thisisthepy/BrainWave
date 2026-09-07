# Vulkan — 네 개가 무엇이었는지 다시 재고, 열여덟 개로 넓힌 라운드

**결론부터: `docs/RELEASE_0_0_13a0.md` §5 의 "Vulkan is four ops" 는 틀리지 않았습니다.
맞는데 오해를 부르는 문장이었고, 그 오해가 어디에 있었는지가 이 라운드의 첫 번째 산출물입니다.**

| 질문 | 답 |
|---|---|
| §5 의 "Vulkan is four ops" 는 맞았나 | **맞다. 다만 넷 중 셰이더를 도는 것은 하나뿐이었다** — `add` 하나. `_to_copy` 는 메모리 복사, `detach`/`alias` 는 GPU 일을 전혀 하지 않는다 |
| 그 문장이 말하지 않은 것 | **임의의 데이터를 장치에 올릴 방법이 아예 없었다.** `ones`/`zeros`/`empty` 뿐이었고, 그래서 지금까지 Vulkan 커널은 **전부 상수로만** 시험되었다 |
| 지금은 몇 개인가 | **18개.** 넷은 전부 유지했고 열넷을 더했다 |
| 무엇을 기준으로 골랐나 | 실제 모델 forward 의 **디스패치 추적**(§2). 재고 순서가 아니라 잰 순서 |
| 값은 무엇과 맞나 | 정확히 반올림되는 op 11개 × 3 shape = **33 케이스가 upstream 과 비트 동일**. matmul 은 유도된 허용치 안이고, 남은 차이는 **FMA 계약임을 비트 단위로 증명**했다 (§4.2) |
| GPU 가 돌았다는 근거는 | 런타임 계수기. **결과가 맞다는 것에서 추론하지 않는다** (§5) |
| 모듈 하나가 forward 하나 | **한다.** `nn.Sequential(Linear, ReLU, Linear)` 가 셰이더 7회, 호스트 되읽기 **0회**로 돈다 |
| 트랜스포머는 | **돌지 않는다.** `layer_norm`·`_softmax`·`gelu`·`embedding`·`bmm` 이 전부 미구현이고 §2 의 추적에 전부 있다 |

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs SHADER_DISPATCHES present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/vulkan.rs maybe_upload present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_every_taught_op_ran_on_the_gpu_or_says_it_did_not present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_vulkan4.py test_the_matmul_residue_is_fma_contraction_and_not_a_defect present -->

---

## 1. 주장을 먼저 다시 쟀다 — 그리고 이번에는 "틀렸다"가 아니라 "덜 말했다"였다

이 저장소에서 §5 항목을 재측정한 것이 이번이 다섯 번째이고, 앞의 넷은 전부 문장이 **틀렸습니다.**
가장 최근 둘은 아무도 예상하지 않은 방식으로 틀렸습니다 — `mps` 항목은 두 주장이 각각 절반씩,
서로 **반대 방향으로** 틀렸고(`docs/MPSATTN.md` §1), collectives 항목은 네 개에 대해 거절을
약속해 놓고 실제로는 **조용히 틀린 숫자를 돌려주고** 있었습니다(`docs/COLLECT2.md` §1).

이번 문장은 다릅니다. **세어 보면 정확히 넷입니다.** 그런데 넷에게 "너는 무엇을 하느냐"고
하나씩 물으면 이렇게 갈라집니다:

| op | 무엇을 하는가 | 컴퓨트 셰이더 |
|---|---|---|
| `aten.add.Tensor` | `add_f32` SPIR-V 커널. **진짜 GPU 산술** | **1회** |
| `aten._to_copy.default` | `vkMapMemory` + `memcpy`. 산술 없음 | 0회 |
| `aten.detach.default` | shape 의 `Arc` 복제 | **0회** |
| `aten.alias.default` | 같음 | **0회** |

**즉 "네 개" 는 *도달 가능성* 을 센 숫자인데 *계산* 으로 읽힙니다.** 넷 중 하나가 산술의 전부였고,
둘은 GPU 커맨드 버퍼를 제출조차 하지 않습니다. 이것은 문장이 거짓이라는 뜻이 아니라,
**그 문장으로부터 독자가 얻는 그림이 실제보다 3배 크다**는 뜻입니다.

이 표는 추론이 아니라 §5 의 계수기로 잰 것이고, `test_the_four_ops_of_the_previous_round_were_one_shader_and_three_that_were_not`
이 그대로 단언합니다.

### 1.1 그리고 op 목록에서는 보이지 않는 것 하나 — 데이터를 올릴 수 없었다

`_vulkan_ops()` 를 아무리 읽어도 나오지 않는 사실입니다. **이 장치에 임의의 값을 올리는 경로가
없었습니다.**

```
>>> x = torch.randn(2, 3)
>>> x.to("vulkan")
NotImplementedError: device not available in torch._C shim: vulkan
>>> torch.tensor([[1.5, -2.5]], device="vulkan")
NotImplementedError: device not available in torch._C shim: vulkan
```

들어가는 문은 `torch.ones` / `zeros` / `empty` 세 개의 팩토리뿐이었습니다
(`aten.rs` 의 `label.kind == "vulkan"` 분기 하나). 그러므로 **지금까지 Vulkan 커널에 도달한 모든
텐서는 상수였습니다.**

이것이 왜 측정의 문제인지가 중요합니다. `docs/VULKAN3.md` §5 의 `add` 테스트는 이 한계를 알고
있었고 *"두 번 더해 피연산자를 구별시킨다"* 로 최선을 다했습니다. 하지만 **전부 1로 채운
행렬의 곱은 올바른 커널과, 인덱스를 뒤집었지만 개수는 맞는 커널을 구별하지 못합니다.**
지시 3번이 요구하는 "upstream 과 원소 단위 비교" 는 진짜 데이터를 필요로 하고,
그래서 이 라운드는 커널을 하나 더 얹기 전에 **업로드 경로부터** 만들었습니다 (`maybe_upload`).

---

## 2. 무엇을 가르칠지 — 재고 순서가 아니라 잰 순서

지시대로 **실제 모델의 forward 가 무엇을 디스패치하는지 기록해서** 골랐습니다.
`docs/MPSATTN.md` 가 쓴 것과 같은 shrunk BERT (vocab 64, hidden 32, 2 layer, 4 head, ffn 64,
`attn_implementation="eager"`, 8 토큰) 를 **upstream torch 에서** `TorchDispatchMode` 로 추적했습니다.

```
총 117 dispatch / 17 종

  40  aten.view.default              4  aten.bmm.default          2  aten.clone.default
  13  aten.t.default                 4  aten._unsafe_view.default 2  aten.gelu.default
  13  aten.addmm.default             3  aten.embedding.default    1  aten.slice.Tensor
  10  aten.transpose.int             2  aten.mul.Tensor           1  aten.select.int
   8  aten.expand.default            2  aten._softmax.default     1  aten.tanh.default
   6  aten.add.Tensor                5  aten.native_layer_norm.default
```

### 2.1 이 추적이 `docs/MPSATTN.md` §2.1 과 다른 곳, 그리고 그것이 방법의 대조군이다

`docs/MPSATTN.md` 는 **shim 안에서** 같은 모델을 추적해 99 dispatch / 16 종을 얻었습니다.
이번 것은 117 / 17 입니다. **다르고, 다른 이유가 정확히 하나입니다:**

- shim 은 `aten.matmul.default` 를 **하나의 op 으로** 유지합니다 (4회).
- upstream 은 그것을 `expand` (8) + `bmm` (4) + `_unsafe_view` (4) 로 분해합니다.

99 + 8 + 4 + 4 − 4×2(=matmul 4회와 contiguous 4회의 차이) 를 맞추는 산수를 하지는 않았고,
그럴 필요도 없습니다. **두 추적이 같은 이름들에서 같은 순위를 준다는 것이 여기서 쓰는 전부**이고,
서로 독립적으로 얻어졌다는 점이 이 순위가 한 하네스의 인공물이 아니라는 근거입니다.
*이 문서가 upstream 쪽을 쓴 이유는 그것이 계측 빌드를 필요로 하지 않기 때문*이고
(`docs/MPSATTN.md` §2 는 게이트를 일회용으로 고쳐 빌드했다가 되돌려야 했습니다), 여기서 재는
질문 — *"모델의 forward 는 무엇을 부르는가"* — 은 모델의 성질이지 shim 의 성질이 아닙니다.

### 2.2 순위가 말하는 것 — 그리고 그것이 직관과 다른 곳

**가장 많이 불리는 것은 산술이 아니라 레이아웃입니다.** `view`(40) + `t`(13) + `transpose`(10)
+ `expand`(8) + `_unsafe_view`(4) = **75 / 117, 64%.** 산술은 `addmm`(13) 이 단연 앞이고
`add`(6) · `layer_norm`(5) · `bmm`(4) 이 뒤를 따릅니다.

그래서 이 라운드가 고른 것은 **"레이아웃 전부 + `addmm` 이 서려면 필요한 것 전부"** 입니다:

| 순위 | op | 이번에 가르쳤나 | 무엇으로 |
|---|---|---|---|
| 40 | `view` / `_unsafe_view` / `reshape` | **예** | 셰이더 없음 — shape 만 (§3.2) |
| 13 | `t` | **예** | `transpose2d_f32` (실체화) |
| 13 | `addmm` | **예** | `matmul_f32` + `bias_add_f32` |
| 10 | `transpose.int` | **예** (2-D 만) | 같음 |
| 8 | `expand` | 아니오 | 브로드캐스트 = 스트라이드 또는 복사 커널. §6 |
| 6 | `add.Tensor` | 이미 있었음 | `add_f32` |
| 5 | `native_layer_norm` | **아니오** | 리덕션 커널이 필요. §6 |
| 4 | `bmm` | 아니오 | 배치 matmul. §6 |
| 3 | `embedding` | 아니오 | gather. §6 |
| 2 | `mul.Tensor` | **예** | `mul_f32` |
| 2 | `_softmax` | **아니오** | 리덕션 두 번 + `exp`. §6 |
| 2 | `clone` | **예** | `copy_f32` |
| 2 | `gelu` | **아니오** | 정확한 `erf` 가 GLSL 내장이 아니다. §6 |
| 1 | `tanh` / `slice` / `select` | 아니오 | §6 |

추적에 없지만 함께 넣은 것: `sub`·`div`·`neg`·`relu`·`mm`·`contiguous`.
`relu` 는 BERT 의 eager 경로에는 없지만(거기는 `gelu` 입니다) **§5 의 모듈 forward 를 성립시키는
비선형**이고, `mm` 은 `addmm` 커널의 절반이라 따로 노출하는 비용이 0 입니다.
`sub`·`div`·`neg` 는 `add`·`mul` 과 같은 셰이더 모양이라 한 줄씩입니다 — **이것들은 순위가
아니라 한계비용이 0 이라서 들어간 것이고, 그렇게 적습니다.**

---

## 3. 무엇을 만들었나

### 3.1 셰이더 아홉 개 — 그리고 툴체인 대조군 하나

`rust/torch_c/shaders/` 가 1개에서 **10개**가 되었습니다.

```
add_f32        (기존)     sub_f32   mul_f32   div_f32          원소별 이항
relu_f32   neg_f32   copy_f32                                  원소별 단항
transpose2d_f32                                                실체화 전치
matmul_f32     bias_add_f32                                    mm / addmm
```

전부 **출력 원소 하나당 인보케이션 하나** 라는 같은 규약을 따릅니다. 그래야 `dispatch_kernel` 이
격자 크기를 원소 수 하나로 정할 수 있습니다. 인터페이스는 스토리지 버퍼 3개와
`Push { uint n; uint p1; uint p2; uint p3; }` 로 통일했습니다 (기존은 `uint n` 하나).

**푸시 상수를 4바이트에서 16바이트로 넓혔는데 `add_f32.spv` 를 다시 컴파일할 필요가 없었습니다.**
선언된 범위보다 적게 읽는 셰이더는 Vulkan 에서 합법이기 때문입니다. 그리고 실제로
`shaders/compile.sh` 를 돌려 확인했습니다:

```
$ cmp <커밋된 add_f32.spv> <재컴파일한 add_f32.spv>   ->  IDENTICAL
```

**이것이 이 절의 대조군입니다.** 열 개의 새 `.spv` 를 이 저장소에 들이면서, 그것을 만든
컴파일러가 **이미 커밋되어 있던 커널을 바이트 단위로 재생산**한다는 것을 먼저 보였습니다.
컴파일러는 NDK 27.1 의 `shader-tools/darwin-x86_64/glslc` 이고, 설치한 것은 없습니다.

`test_the_checked_in_spirv_is_not_stale_for_any_shader` 는 `docs/VULKAN3.md` 가 `add_f32` 하나에
걸어 둔 최신성 검사를 **열 개 전부로** 넓히고, SPIR-V 매직 넘버까지 봅니다 — mtime 만 보면
`.comp` 를 `.spv` 로 복사해도 통과하기 때문입니다.

### 3.2 셰이더가 **없는** op 이 다섯 개 있고, 그것을 숨기지 않았다

`view`·`_unsafe_view`·`reshape`·`contiguous`·`detach`·`alias` 는 컴퓨트 셰이더를 **한 번도**
돌리지 않습니다. 이 장치의 모든 텐서는 구성상 연속이고(스트라이드가 없으니 불연속일 수가 없습니다),
그러므로 reshape 은 같은 버퍼 위의 새 `shape` 이며 바이트를 옮기지 않습니다.

**0 은 1 만큼이나 하나의 주장입니다.** §5 의 표가 이 op 들에 대해 `dispatches == 0` 을 단언하는
것은 게으름이 아니라, "GPU 가 일했다" 로 읽힐 숫자를 적지 않겠다는 것입니다.

### 3.3 업로드 경로 — `maybe_upload`

`x.to("vulkan")` 은 평범한 CPU `_to_copy` 로 도착해서 `resolve()` 까지 가 거절당하고 있었습니다.
`aten_dispatch` 의 마지막 팔에서 `crate::vulkan::maybe_upload` 를 먼저 묻습니다.
**vulkan 으로의 복사가 아닌 모든 호출에 대해 `None` 을 돌려주므로 다른 경로는 모양이 바뀌지
않습니다.** dtype 변환은 여기서도 거절합니다 — 변환 셰이더가 없고, 조용히 넓히거나 좁히는 것이
이 장치가 존재하는 이유에 반합니다.

`torch.tensor([...], device="vulkan")` 은 **여전히 거절합니다.** 다른 팩토리이고, 이번 라운드가
필요로 한 것은 `.to("vulkan")` 하나였습니다. 넓히지 않은 것을 §6 에 적습니다.

---

## 4. 값 — 허용치를 고르지 않았다

`docs/AGREE.md` §2 의 규칙을 그대로 씁니다: **upstream 을 `float64` 로도 돌려 진실을 만들고,
허용치를 그 분포에서 읽어냅니다.**

### 4.1 정확히 반올림되는 op 은 허용치가 **0** 이다 — 비트 동일

`add`·`sub`·`mul`·`div`·`neg`·`relu`·`clone`·`contiguous`·`view`·`t`·`transpose` 는 원소당
IEEE-754 단정도 연산이 **정확히 하나**(또는 0개)입니다. IEEE-754 는 그 연산들을 마지막 비트까지
규정하므로, 이 모집단에 대해 유도 규칙이 내놓는 허용치는 **0** 이고, 그러므로 **비트 동일**로
겁니다. `docs/COLLECT2.md` 가 숫자를 옮기기만 하는 collectives 를 비트 동일로 건 것과 같은
논리이고, 같은 자리에서 나온 것입니다.

`randn` 데이터, 3개 shape, 11개 op = **33 케이스 전부 upstream 과 비트 동일**입니다.
(이 문장이 이번 라운드 전에는 쓸 수 없었다는 점이 §1.1 입니다 — 데이터가 전부 1.0 이었습니다.)

### 4.2 matmul 은 다르고, 그 차이가 정밀도임을 **비트로 증명**했다

내적은 **합**이고 합에는 순서가 있습니다. 여기가 이 디렉터리에서 답이 강제되지 않는 유일한
자리이고, 그래서 허용치가 필요한 유일한 자리입니다.

31개 케이스(원소별 21 + mm/addmm 10)로 잰 유도값:

```
upstream 자신의 float32-대-float64 상대 오차, 31 케이스:
    median 3.205e-08    p90 1.618e-07    max 2.018e-07
float32 eps                                          1.192e-07
8 ulp 바닥 (docs/AGREE.md §2 의 바닥, 같은 이유로)     9.537e-07
유도된 허용치 = max(p90, 8 ulp)                       9.537e-07

shim 의 최악 상대 오차                                2.731e-07     <- 허용치의 0.29배
호스트 되읽기 (전 케이스 합계)                        0
```

`test_the_matmuls_agree_with_upstream_at_a_derived_tolerance` 는 같은 유도를 **테스트 안에서 다시**
합니다 — 상수를 붙여넣지 않고 그 자리의 모집단에서 계산하므로, 모집단이 바뀌면 상수가 낡는 대신
숫자가 바뀝니다. 그 모집단은 mm/addmm 12개로 위보다 작고, 게이트에서 이렇게 인쇄합니다:

```
derived tolerance = max(p90 2.228e-07, 8 ulp 9.537e-07) = 9.537e-07;
worst shim relative error 3.197e-07
```

두 모집단 모두에서 **8 ulp 바닥이 p90 을 이깁니다.** 즉 이 허용치를 실제로 정하고 있는 것은
`docs/AGREE.md` 의 바닥이지 이 라운드의 케이스 선택이 아니고, 그래서 케이스를 더하거나 빼서
허용치를 움직일 수 없습니다.

**그런데 허용치를 통과하는 것은 증명이 아닙니다.** 인덱스를 뒤집었거나 항을 하나 잃은 커널도
"작음" 안에 들어올 수 있고, 허용치는 그 둘을 구별하지 못합니다. 그래서 남은 차이가 무엇인지
직접 물었습니다.

512항 내적 하나를 골라 네 가지로 계산했습니다:

```
float64 진실 (upstream)             43.117996500412531
upstream torch.dot  (블록 gemm)     43.118000030517578   bits d5782c42
호스트에서 naive 순차 f32           43.118003845214844   bits d6782c42
shim VULKAN 커널                    43.118007659912109   bits d7782c42
호스트에서 순차 f32 + FMA           43.118007659912109   bits d7782c42   <-- 일치
```

세 답이 **연속한 3 ulp** 이고, 마지막 줄이 답입니다. GLSL 은 `acc += a*b` 를 **FMA 로 계약**하는
것을 허용하고, 이 드라이버가 그렇게 합니다. 두 float32 를 곱한 값은 double 에 정확히 들어가므로
`round_f32(acc + a*b)` 를 호스트에서 계산하면 그것이 곧 f32 FMA 이고, **그 모델이 GPU 답과
비트 단위로 같습니다.**

그리고 이것은 한 케이스의 우연이 아닙니다 — **여섯 shape 전부**에서 성립합니다:

| shape | 호스트 FMA 모델과 비트 동일 | upstream 과 비트 동일 |
|---|---|---|
| `mm[2,3,4]` | **예** | 예 (0/8 다름) |
| `mm[8,16,8]` | **예** | 예 (0/64) |
| `mm[5,64,7]` | **예** | 아니오 (1/35) |
| `mm[32,128,16]` | **예** | 예 (0/512) |
| `mm[1,512,1]` | **예** | 아니오 (1/1) |
| `mm[3,257,5]` | **예** | 아니오 (2/15) |

**커널이 서술하는 산술의 모델이 커널의 답을 모든 shape 에서 마지막 비트까지 재현한다** — 이것이
"정밀도이지 결함이 아니다" 의 증명이고, 허용치보다 훨씬 강한 진술입니다. 인덱스를 뒤집은 커널은
이것을 통과할 수 없습니다. `test_the_matmul_residue_is_fma_contraction_and_not_a_defect` 가
이것이고, **upstream 과 다른 원소가 0개가 되면 스스로 실패합니다** — 그러면 이 모집단이 더 이상
누적 순서 문제를 건드리지 않는다는 뜻이고, 그때는 증거가 자명해져서 의미가 없어지기 때문입니다.

### 4.3 그리고 유도 규칙이 **초과되는** 자리를 하나 찾았고, 숨기지 않는다

같은 512항 내적을 `docs/AGREE.md` 의 두 번째 규칙(*"upstream 자신이 float64 진실에서 떨어진
거리의 4배 안"*)으로 재면 이렇게 됩니다:

```
float64 진실로부터   upstream f32  2.282e-07      shim vulkan  2.136e-06      비율 9.4x
```

**4배를 넘습니다.** 이것을 통과하는 프레이밍으로 바꾸지 않고 그대로 적습니다. 왜 넘는지는
분명합니다: 이 **한 스칼라**에서 upstream 이 운 좋게 0.065 ulp 로 맞췄고(이 크기에서 1 ulp 는
3.499e-06), shim 은 0.61 ulp 입니다. **둘 다 1 ulp 아래인데 비율이 9.4 가 된 것은 분모가
작아서**입니다.

이것은 규칙의 오용이지 커널의 결함이 아닙니다. `docs/AGREE.md` 의 4배 규칙은 **아키텍처 하나의
출력 텐서 전체**에 대한 2차 규칙이고, 1차 규칙은 모집단 p90 상대 허용치입니다. 원소가 하나뿐인
출력에서는 비율의 분모가 통계가 아니라 한 번의 반올림 운이 됩니다. §4.2 의 FMA 증명이 같은
케이스에 대해 **비율보다 훨씬 강한 답**을 이미 주고 있고, 텐서 전체를 보는 §4.4 의 모듈
측정에서는 비율이 **0.72–0.90** 으로 편안하게 안쪽입니다.

**그래서 이 문서가 남기는 권고는:** 출력 원소가 소수인 케이스에 4배 비율 규칙을 적용하지 마라.
그 자리에서 물어야 할 것은 **ulp** 이고, 답은 0.61 ulp 다.

### 4.4 모듈 하나 — 텐서 전체

`nn.Sequential(Linear(12,32), ReLU, Linear(32,6))`, 배치 5, 출력 30개:

| | float64 진실로부터 | upstream 자신의 오차 대비 |
|---|---|---|
| upstream `float32` (오라클 자신) | 5.2120e-08 | 1.00× — 기준 |
| shim `cpu` | 5.2120e-08 | 1.00× |
| **shim `vulkan`** | **4.7031e-08** | **0.90×** |

`vulkan` 이 `cpu` 와 다른 정도는 **5.9605e-08 = 1.02 float32 ulp** 입니다.
(테스트가 쓰는 다른 시드에서는 0.72× / 1.09 ulp 로 나옵니다 — 같은 크기입니다.)

**`vulkan` 이 upstream 의 float32 보다 진실에 *더 가깝습니다*.** 이것을 이 커널의 우월함으로
읽으면 안 됩니다 — 30개 원소의 최댓값 하나가 어느 쪽으로 반올림되었는지의 문제이고,
방향은 시드를 바꾸면 바뀝니다. 적어 두는 이유는 이 크기의 차이에 방향이 없다는 것 자체가
"결함이 아니다"의 일부이기 때문입니다.

---

## 5. GPU 가 돌았는가 — 런타임 단언이지 추론이 아니다

지시 4번이 가리키는 `docs/MPSATTN.md` §3.1 을 읽고 설계했습니다. 그 문단이 기록한 것은
**자기 라운드에 대한 자백**입니다:

> `read_flat` 호출을 `host_softmax_cpu_only` 같은 이름의 함수 한 단계 아래로 옮기면 **두 테스트가
> 모두 통과합니다** — per-op 스캔은 커널 본문에서 여섯 헬퍼 이름만 찾고, 분류 테스트는 마커만
> 찾는데 `read_flat` 은 마커가 아니기 때문입니다.

**즉 소스를 grep 하는 모든 검사는 grep 하는 그것을 옮기면 무력화됩니다.** 저의 증거가 그 모양이
아니어야 합니다.

그래서 이 라운드의 증거는 **소스에 대한 서술이 아니라 프로세스가 한 일**입니다. 계수기 셋을
모듈 안에서 호스트/장치 경계를 넘거나 GPU 에 일을 시키는 **유일한 세 지점**에 두었습니다:

| 계수기 | 어디서 증가하는가 |
|---|---|
| `shader_dispatches` | `dispatch_kernel` 안, **`vkWaitForFences` 가 성공을 돌려준 뒤** |
| `host_uploads` | `upload` 안 — 모듈의 유일한 쓰기용 `vkMapMemory` |
| `host_downloads` | `download` 안 — 모듈의 유일한 읽기용 `vkMapMemory` |

**왜 이것은 같은 방식으로 무력화되지 않는가.** 호스트에서 계산하려는 op 은 피연산자를 읽어야
하고, 이 모듈에서 `VkBuffer` 를 읽는 길은 `download` 하나입니다. 헬퍼를 몇 단계 파고들어
이름을 무엇으로 바꾸든 그 함수를 지나가야 합니다. 그래서 테스트가 op 마다 단언하는 것은
**"기대한 개수의 컴퓨트 셰이더가 돌았고, 되읽기는 0이었다"** 입니다.

`test_every_taught_op_ran_on_the_gpu_or_says_it_did_not` 의 표(17개 op)와, 그 표가 `_vulkan_ops()`
와 **집합으로 일치하는지**까지 단언합니다 — op 을 가르치면서 셰이더 개수를 적지 않으면 실패합니다.

### 5.1 계수기가 살아 있는지 — 양성 대조군

*"이 계수기가 움직이지 않았다"* 는 단언은 **그 계수기가 움직일 수 있을 때에만** 값이 있습니다.
`test_the_readback_counter_moves_when_something_is_actually_read_back` 이 `.cpu()` 한 번에
`host_downloads` 가 정확히 1 오르는 것을 봅니다. 이것이 없으면, `download` 의 증가를 누가
지워도 위의 17개 단언이 **전부 조용히 초록**이 됩니다.

### 5.2 정직하게 좁혀 두는 것

이 단언이 말하는 것은 *"이 모듈이 버퍼를 호스트로 읽지 않았고 기대한 만큼의 컴퓨트 파이프라인이
펜스까지 돌았다"* 입니다. **드라이버 안까지 따라가 "산술이 GPU 코어에서 일어났다" 를 말하지
않습니다.** 이 기계에서 도달 가능한 유일한 ICD 는 `kosmickrisp` — Vulkan-을-Metal 로 번역하는
계층이고, 그것이 컴퓨트 파이프라인을 Metal 커널로 옮긴다는 것은 `docs/VULKAN2.md` §4 가
`type=INTEGRATED_GPU`, `"Apple M1"` 으로 확인한 것이지 이 라운드가 다시 잰 것이 아닙니다.

---

## 6. "Vulkan works" 가 무슨 뜻이고 어디까지 갔나 — 정직한 숫자

**"Vulkan works" 를 이 문서는 이렇게 정의합니다:** 사용자가 만든 `torch.nn.Module` 을
`.to("vulkan")` 하고 실제 입력으로 forward 했을 때, (a) 완주하고, (b) upstream 과 유도된
허용치 안에서 일치하고, (c) 그 계산이 **런타임 단언으로** GPU 에서 일어났음이 확인되는 것.

**그 정의로 도달한 곳:**

```
    nn.Sequential(Linear(12,32), ReLU, Linear(32,6))     forward    OK
        출력 장치        vulkan
        컴퓨트 셰이더    7회   = 2 x (transpose + matmul + bias) + 1 relu
        호스트 되읽기    0회
        upstream 대비    0.72-0.90x   (docs/AGREE.md 의 4배 규칙 안)
        shim cpu 대비    1.02-1.09 float32 ulp
```

**도달하지 못한 곳, 그리고 그것이 훨씬 중요합니다:**

- **트랜스포머는 forward 하지 않습니다.** §2 의 추적에 있는 op 중
  `native_layer_norm`(5회) · `_softmax`(2회) · `gelu`(2회) · `embedding`(3회) · `bmm`(4회) ·
  `expand`(8회) 가 전부 미구현이고, BERT 는 그중 첫 번째에서 멈춥니다.
  `test_a_transformer_still_does_not_forward_and_the_wall_is_named` 이 다섯 개가 여전히
  거절되는 것을 단언하므로, 누군가 하나를 가르치면 **이 문장이 실패로 갱신을 요구**합니다.
- **네 개에서 여덟 개가 아니라 열여덟 개입니다. 그런데 열여덟 중 컴퓨트 셰이더를 도는 것은
  열한 개**이고, 나머지 일곱은 shape 조작이거나 복사입니다. §1 이 지적한 셈법의 함정을 이
  문서 자신에게도 적용하면 그렇게 됩니다. **"18 ops"보다 "11 kernels + 7 metadata ops"가 정확한
  요약입니다.**
- **성능은 재지 않았습니다.** `docs/VULKAN2.md` §4.4 의 이유가 그대로입니다 — `kosmickrisp` 은
  번역 계층이고, 이 기계는 측정 중 load average 가 10 을 넘었습니다. 여기서 나올 숫자는 장치가
  아니라 번역기와 부하를 묘사합니다. §5 의 문장 후반 — *"performance needs a phone and has not
  been measured"* — 은 **여전히 그대로 참이고, 이 라운드는 그 절반을 건드리지 않았습니다.**
- **실물 폰에서 돌리지 않았습니다.** 공유 에뮬레이터는 건드리지 않았습니다 (§8).

### 6.1 좁혀 놓은 것들 — 전부 이름을 대며 거절한다

| 무엇 | 왜 근사하지 않았나 |
|---|---|
| 브로드캐스트 (`add([2,3], [3,2])`) | CPU 구현이 몇 줄 옆에 있고, 그것을 부르는 것이 바로 이 장치가 막으려는 조용한 폴백 |
| `alpha` / `beta` | `add`·`addmm` 커널에 곱셈이 없다. `nn.Linear` 는 주지 않는다 |
| 3-D 이상의 `transpose` | `VkTensor` 에 스트라이드가 없어 전치가 바이트를 옮겨야 하고, 그 커널이 2-D 다 |
| `bmm` | 같은 이유. 배치 matmul 커널이 없다 |
| f32 아닌 dtype | 변환 셰이더가 없다. 올릴 때도 계산할 때도 거절 |
| `torch.tensor([...], device="vulkan")` | 다른 팩토리. `.to("vulkan")` 만 만들었다 |
| `gelu` | upstream 의 기본 `gelu` 는 정확한 `erf` 이고 GLSL 내장이 아니다. tanh 근사로 바꾸면 값이 달라지는데, **그 차이는 정밀도가 아니라 다른 함수**다 |
| 리덕션 (`sum`·`softmax`·`layer_norm`) | 행 단위 리덕션 커널이 없다. 이 라운드가 하지 않은 가장 큰 덩어리이고, 트랜스포머로 가는 길이다 |

**전치가 실체화된다는 것은 upstream 과의 진짜 차이**이고 숨기지 않습니다. upstream 에서
`x.t()` 는 메타데이터이고 여기서는 바이트를 옮깁니다. 결과 값은 같지만(§4.1 이 비트 동일로
확인), **비용 모델이 다르고** 스트라이드를 도입하는 것은 이 라운드보다 훨씬 큰 변경입니다.

---

## 7. 무력화 — 실패할 수 없는 검증은 검증이 아니다

`CLAUDE.md` §5.5. 이번에 얹은 것을 하나씩 되돌려 다시 빌드하고 **실제로 빨간 것을 봤습니다.**

| # | 무력화한 것 | 잡은 테스트 수 | 어디서 (대표) |
|---|---|---|---|
| 1 | `add_tensor` 를 호스트 계산으로 (download → 더하기 → upload) | **3** | `..._ran_on_the_gpu...`: *"aten.add.Tensor ran 0 compute shaders, expected 1"* |
| 2 | `dispatch_kernel` 의 `SHADER_DISPATCHES` 증가 제거 | **3** | 모듈 forward: *"ran 0 compute shaders, expected 7"* |
| 3 | `download` 의 `HOST_DOWNLOADS` 증가 제거 | **1** | `..._readback_counter_moves...` — **이 하나뿐**. §7.1 |
| 4 | `matmul_f32` 의 B 인덱스를 전치 (`b[k*p3+col]` → `b[col*p2+k]`) | **3** | FMA 모델: *"8 of 8 elements differ from the host model"* |
| 5 | `transpose2d` 를 버퍼 공유로 (전치하지 않음) | **3** | 비트 동일: *"t[4,5] … 18 of 20 elements differ"*, 그리고 셰이더 0회 |
| 6 | `maybe_upload` 가 값 대신 0 을 올림 | **5** | 왕복 · 비트 동일 · FMA 모델 · 유도 허용치 · 모듈 forward |
| 7 | `bias_add_f32` 가 bias 를 `bias[0]` 으로 (열 인덱스 상실) | **2** | 유도 허용치: *"addmm[2,3,4]: shim 5.037e-01"* |

되돌린 뒤 다시 빌드해 **16개 전부 초록**인 것을 확인했고, 소스는 백업과 바이트 단위로 동일합니다.

**1번이 이 표의 이유입니다.** `docs/MPSATTN.md` §3.1 이 서술한 바로 그 무력화 — *"거절 목록에서
이름을 빼면서 되읽기는 그대로 두는"* — 을 이 장치에서 실행해 보았고, **잡힙니다.**
소스 스캔이 아니라 런타임 계수기이기 때문입니다. 덤으로
`test_every_shader_on_disk_is_reachable_from_the_dispatcher` 도 걸렸습니다 —
`add_f32.comp` 이 갑자기 아무도 부르지 않는 파일이 되었기 때문입니다.

**3번이 §5.1 의 양성 대조군을 정당화합니다.** 되읽기 계수기를 죽였을 때 빨개진 것은
**단 하나**, 그 대조군뿐이었습니다. 그것이 없었다면 계수기를 죽이는 변경이 조용히 통과하고,
그다음에 1번 같은 것이 들어와도 `host_downloads == 0` 이 **참인 이유가 사라진 채로** 초록이
됩니다.

**7번은 §4.2 의 FMA 모델이 잡지 못했습니다.** 그 모델은 `mm` 만 재현하고 `addmm` 의 bias 단계는
재현하지 않기 때문입니다. 값 테스트가 잡았지만, **증거가 어디까지 닿는지의 경계가 정확히
거기**라는 것을 적어 둡니다.

### 7.1 테스트가 잡지 **못하는** 것

- **`contiguous`/`view` 가 잘못된 shape 을 만드는 것**은 §5 의 계수기가 잡지 못합니다
  (셰이더가 0회인 것이 정상이므로). shape 단언과 값 테스트가 잡지만, **계수기는 이 부류에
  대해 아무 말도 하지 않습니다.** 계수기를 "GPU 검증" 으로 일반화해 읽으면 안 되는 이유입니다.
- **드라이버가 컴퓨트 파이프라인을 CPU 에서 에뮬레이트하는 경우.** `lvp`/`swiftshader` ICD 로
  돌리면 계수기는 똑같이 증가하고 답도 맞습니다. 계수기는 *이 크레이트가* 호스트로 되읽지
  않았음을 말하지, 드라이버가 실리콘을 썼음을 말하지 않습니다 (§5.2).
- **성능 회귀.** 아무것도 시간을 재지 않습니다. 의도적입니다 (§6).
- **`addmm` 의 bias 단계.** §4.2 의 FMA 모델은 `mm` 만 재현합니다. bias 를 망가뜨린 무력화 7번은
  값 테스트에만 걸렸고, "커널이 서술한 산술을 한다" 는 종류의 증거는 `addmm` 의 후반부에는
  **없습니다.**
- **되읽기 계수기 자체.** 무력화 3번은 테스트 **하나**에만 걸립니다. 그 하나가 §5.1 이고,
  그것이 지워지면 §5 의 17개 단언이 전부 공허해집니다.

---

## 8. 게이트와 규율

| | 값 |
|---|---|
| `pytests/run.sh` | §9 참조 |
| `cargo test` | 30 passed, 0 failed |
| DOCWATCH | §9 |
| golden `compare.py` | **움직이지 않아야 하고, 움직이지 않았다** — 이 라운드의 음성 대조군 |
| 공유 에뮬레이터 | **켜지 않았고 접속하지 않았습니다.** `ANDROID_SERIAL` 을 쓸 일이 없었습니다 |

<!-- DOCWATCH: count golden_cases_passed ge 11420 --> <!-- DOCWATCH: count golden_ops_covered ge 302 -->

golden 이 움직이지 않는 것이 왜 정확한 결과인가: 이 라운드는 `_aten_implemented()` 에 이름을
**하나도 더하지 않았습니다.** 가르친 18개는 전부 이미 CPU 커널이 있고 golden 이 이미 비교하던
op 들이며, 이 라운드가 준 것은 그 op 들이 `vulkan` 장치에서 도는 경로입니다. CPU 결과가 하나라도
바뀌었다면 의도하지 않은 것을 건드린 것입니다.

### 8.1 `run.sh` 를 통해 Vulkan 이 보이지 않던 문제를 고쳤다

`docs/VULKAN3.md` §6.1 이 기록하고 담당 범위 밖으로 남긴 함정입니다: macOS SIP 가 보호된
바이너리(`/bin/sh`)를 exec 할 때 환경에서 `DYLD_*` 를 **떼어냅니다.** 그래서
`DYLD_LIBRARY_PATH=... sh run.sh` 는 셸에는 변수를 주고 파이썬에는 주지 않고,
**로더를 정확히 가리킨 사람에게 "no vulkan" 이라고 말합니다.**

그 문서가 제안한 대로 파이썬을 부르기 직전에 다시 세웁니다. 다만 **다른 이름으로, 옵트인**입니다:

```sh
TORCHNATIVE_VULKAN_DYLD=~/Library/Android/sdk/emulator/lib64/vulkan \
VK_DRIVER_FILES=.../libkosmickrisp_icd.json \
    bash rust/torch_c/pytests/run.sh
```

설정하지 않으면 이전과 **정확히 같게** 동작하고 vulkan 테스트는 이름을 대며 스킵합니다.
`DYLD_LIBRARY_PATH` 를 그대로 재사용하지 않은 이유는, 그것을 export 했다가 사라지는 것을 본
사람에게 두 번 같은 말을 하지 않기 위해서입니다.

---

## 9. 이 라운드가 한 일 — 종류별로 나눈다

`CLAUDE.md` §5.3: 숫자는 진척이 아니고, 섞으면 진척처럼 보입니다.

**기능 추가 (14개 op, 9개 셰이더, 1개 경로)**
`mul`·`sub`·`div`·`neg`·`relu`·`clone`·`contiguous`·`view`·`_unsafe_view`·`reshape`·`t`·
`transpose.int`·`mm`·`addmm` 을 `vulkan` 에 가르쳤고, `x.to("vulkan")` 업로드 경로를 만들었습니다.

**계측 추가 (1개)**
`_vulkan_counters()` — GPU 실행을 런타임에서 단언하는 수단. 이것이 없으면 §5 의 어떤 문장도
쓸 수 없습니다.

**결함 수정 (1개)**
`run.sh` 가 SIP 때문에 vulkan 테스트를 항상 스킵하던 것(§8.1). 기존 거절을 약화시킨 곳은
없습니다.

**측정 (3개)**
§1 의 네 op 재측정, §2 의 forward 추적, §4 의 값 비교. 이 중 §1 과 §4 는 **이번에 만든 도구가
없었으면 불가능**했습니다.

**테스트 추가 (15개)**
`test_vulkan4.py`. 그중 §5·§5.1 의 둘은 *기능*이 아니라 *증거의 성질*을 검사합니다.

**삭제**
없습니다. `_vulkan_ops()` 의 네 개는 전부 그대로 있고, 어떤 거절도 약화시키지 않았습니다.

---

## 10. 다음 라운드에 넘기는 것

1. **리덕션 커널.** `sum`/`max` 를 행 단위로 하는 셰이더 하나가 `_softmax` 와
   `native_layer_norm` 을 동시에 엽니다. §2 의 순위에서 남은 가장 큰 덩어리이고,
   **트랜스포머까지의 거리 대부분**입니다.
2. **`expand` 와 브로드캐스트.** 8회로 순위가 높은데 손대지 않았습니다. 스트라이드를 도입할지
   복사 커널로 갈지는 §6 의 전치 문제와 같은 결정이므로 함께 정하는 것이 맞습니다.
3. **`bmm`.** 배치 차원 하나를 `matmul_f32` 에 더하는 것이라 작습니다.
4. **할당자.** 텐서당 `vkAllocateMemory` 하나라는 `docs/VULKAN3.md` §7 의 항목이 **그대로**
   입니다. 이 라운드가 텐서 개수를 늘렸으므로 압력은 커졌습니다 — §5 의 MLP forward 는 셰이더를
   7회 돌리고 **모든 커널이 자기 출력을 새로 할당**하므로, forward 한 번에 `vkAllocateMemory`
   가 최소 7회입니다(파라미터 업로드분은 별도). 모바일의 `maxMemoryAllocationCount` 가 4096 인
   것을 생각하면 모델 크기에는 틀린 모양이고, **성능을 재기 전에 이것이 먼저**입니다.
5. **폰.** §5 문장의 후반은 이 라운드가 건드리지 않았고, 여전히 참입니다.
