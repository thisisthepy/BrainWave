# `mps` — 조용한 CPU 폴백을 닫는다

**결론부터: 고른 것은 "문 앞에서 거절"이고, 문을 세운 자리가 설계입니다.**

| 질문 | 답 |
|---|---|
| `device="mps"` 로 쓴 사용자가, Metal 이 못 하는 op 에 대해 이제 무엇을 받는가 | **`NotImplementedError`.** op 이름과 `mps` 와 `.cpu()` 를 대며 거절 |
| 무엇을 거절하는가 | 이 크레이트의 커널이 텐서를 호스트로 되읽어 계산하는 op **54개** (구현된 228개 중) |
| 거절하지 않는 것은 왜 안전한가 | candle 의 Metal 백엔드에는 **조용한 CPU 폴백이 없다.** 못 하는 op 은 전부 에러를 낸다 |
| 가드를 떼면 무엇이 돌아오는가 | 아래 §2 — 프로브한 14개 중 **13개가 `mps:0` 라벨을 단 채 CPU 에서 계산된 값을 조용히 돌려준다** |

---

## 1. 넘겨받은 전제 두 개 중 하나가 틀렸다

`docs/VULKAN3.md` §3 은 이렇게 적었습니다 — 감추지 않고 적은 것은 옳았고, **원인 지목이
틀렸습니다.**

> candle 은 필요한 경우 Metal 텐서를 호스트로 되읽어 계산하고 결과를 다시 mps 로 올립니다.
> (…) `aten.nonzero.default` 와 `aten.tril.default` 가 CPU 에서 계산되었다.

두 문장 다 확인했고 두 문장 다 수정이 필요합니다.

### 1.1 되읽기는 candle 이 아니라 **이 크레이트**가 한다

`candle-core-0.11.0/src/metal_backend/mod.rs` 에는 CPU 폴백 경로가 **없습니다.** 구현하지 않은
op 은 `bail!` 합니다. 그래서 가드가 없던 상태에서도 이렇게 나왔습니다:

```
aten.sort.default    RuntimeError: candle: Metal contiguous to_dtype F32 F64 not implemented
aten._softmax.default (f32)  같은 형태
```

되읽는 것은 `aten.rs` 의 커널들입니다. `nonzero_default` 는 마스크를 `to_vec1::<u8>()` 로
호스트에 가져와 인덱스 산술을 Rust 에서 하고 `Tensor::from_vec(..., t.device())` 로 다시
올립니다. **답은 맞고, GPU 는 아무것도 하지 않았습니다.**

이 구분이 설계를 바꿉니다. 폴백이 candle 안에 있었다면 밖에서 할 수 있는 것이 거의 없지만,
**이 크레이트의 소스에 있다면 이 크레이트의 소스가 그 목록의 근거가 될 수 있습니다.**

### 1.2 `tril` 은 CPU 에서 계산되지 않았다

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs tril_triu present -->
<!-- DOCWATCH: op-implemented aten.tril.default -->

`tril_triu` 는 마스크를 호스트에서 만들지만 그 마스크는 **입력과 무관한 상수**이고, 계산은
`where_cond` 입니다. `where_cond` 는 candle 의 진짜 Metal 커널입니다
(`metal_backend/mod.rs:846`). 입력의 바이트는 한 번도 호스트로 오지 않습니다.

§3 이 `tril` 을 지목한 근거는 "성공했고 값이 맞았다" 였는데, **호스트 되읽기와 Metal 커널은
둘 다 맞는 값을 냅니다.** 그것으로는 구분되지 않습니다. 그래서 `tril` 은 거절 목록에 없고,
`test_tril_on_mps_computes_on_the_gpu_after_all` 이 그 정정을 붙잡아 둡니다.

### 1.3 그리고 진짜 크기는 둘이 아니라 54 였다

같은 스캔이 `aten._softmax.default` 를 찾아냅니다. **`mps` 위의 트랜스포머는 어텐션 블록마다
그곳을 지납니다.** §3 이 둘이라고 적은 것은 둘을 봤기 때문이지 둘뿐이어서가 아닙니다.

---

## 2. 가드를 떼면 무엇이 돌아오는가 — 실측

`mps_host_readback_gate` 를 `if true { Ok(()) }` 로 무력화하고 다시 빌드한 뒤 잰 것입니다.
전부 `device=mps:0` 을 달고, 전부 값이 맞고, 전부 CPU 가 계산했습니다.

```
SILENT-CPU nonzero(f32)        device=mps:0   [0, 0, 0, 1]
SILENT-CPU where(f32)          device=mps:0   [0, 0, 0, 1]
SILENT-CPU neg(int64)          device=mps:0   [0, -1, -2, -3]
SILENT-CPU abs(int64)          device=mps:0   [0, 1, 2, 3]
SILENT-CPU bitwise_not(int64)  device=mps:0   [-1, -2, -3, -4]
SILENT-CPU gather(int64)       device=mps:0   [0, 3, 6]
SILENT-CPU pow(int64)          device=mps:0   [0, 1, 4, 9]
SILENT-CPU remainder(int64)    device=mps:0   [0, 1, 0, 1]
SILENT-CPU index(int64)        device=mps:0   [0, 1, 2, 3]
SILENT-CPU max(int64)          device=mps:0   [8]
SILENT-CPU masked_select       device=mps:0   [0, 1, 2, 3]
SILENT-CPU cumsum(int64)       device=mps:0   [0, 1, 2, 3]
SILENT-CPU one_hot             device=mps:0   [1, 0, 0, 0]
loud       softmax(f32)        RuntimeError: candle: Metal ... to_dtype F32 F64
```

마지막 줄이 나머지 열셋을 설명합니다. `read_flat` 은 부동소수 경로에서 `widen_f64` 를 지나고
Metal 은 F32→F64 를 구현하지 않으므로 **거기서는 이미 시끄러웠습니다.** 조용했던 것은 정수
경로 — `to_dtype(I64)` 는 Metal 이 하므로 되읽기가 막힘 없이 성공하고, 그 뒤 산술이 Rust 에서
일어납니다. **가장 조용한 실패가 가장 늦게 발견되는 경로에 있었습니다.**

가드를 켠 상태에서 같은 것을 부르면:

```
NotImplementedError: aten.nonzero.default: not implemented for the mps device.
This kernel reads the tensor back to host memory and computes there, so it would
return a correct value that the GPU did not compute, under an mps label -- the
shim refuses that rather than doing it silently. Move the tensor with .cpu() to
ask for the CPU on purpose. 54 of the ops this build implements are refused on
mps for this reason; torch._C._shim_mps_host_readback_ops() lists them.
```

---

## 3. 왜 이 선택인가 — 셋 중에서

지시가 준 세 가지 중 **"문 앞에서 거절"** 입니다.

| | 왜 아닌가 / 왜 이것인가 |
|---|---|
| 시끄럽게 폴백 | op 당 한 번 경고. **아무도 안 읽는 경고는 침묵에 가깝고**, 값은 여전히 틀린 장치에서 나온다 |
| 구조적으로 불가능하게 | Vulkan 의 `Repr::Vulkan` 이 그것이고, **여기서는 성립하지 않는다.** mps 텐서는 진짜 candle 텐서라 `tensor()` 가 거절할 근거가 없고, 모든 Metal 텐서를 감싸면 `mps` 가 주는 유일한 이점(커널을 하나도 안 가르쳐도 된다)을 버리게 된다 |
| **문 앞에서 거절** | 문이 이미 하나다. `aten_dispatch` 의 `check_devices_agree` 가 인자의 장치를 **이미 스캔하고 있고**, meta 와 vulkan 이 이미 그 결과로 갈라진다. mps 는 거기에 팔 하나 |

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs mps_host_readback_gate present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/device.rs MPS_HOST_READBACK_OPS present -->

`aten.rs` 에 들어간 것은 그 팔 하나(3줄)이고, 나머지는 전부 `device.rs` 에 있습니다.

### 3.1 목록인가 — 그렇다. 그리고 무엇이 그것을 정직하게 유지하는가

지시가 물은 질문입니다. 답은 **"candle 이 무엇을 구현했는지 알아낼 필요가 없다"** 입니다.

§1.1 때문에 목록의 근거가 candle 이 아니라 **이 저장소의 소스**입니다. `aten.rs` 를 스캔해
장치 바이트를 호스트로 옮기는 네 가지 호출(`to_vec0..3`, `to_scalar`, `to_cpu`)을 찾고,
커널이 직접 부르거나 그것을 부르는 여섯 개 헬퍼(`read_flat`, `side_from_tensor`,
`mask_to_indices`, `nan_along_dim`, `narrow_through`, `local_scalar_dense`)를 부르면 목록에
올립니다. 그 스캔은 **테스트가 매번 다시 돌립니다:**

* `test_the_mps_readback_list_is_what_the_kernels_actually_do`
  <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_mps_readback_list_is_what_the_kernels_actually_do present -->
  소스에서 다시 유도한 집합을 **아티팩트가 게이트하고 있는 표**
  (`_C._shim_mps_host_readback_ops()`) 와 비교합니다. 소스끼리 비교했다면 아무도 다시 빌드하지
  않은 변경 뒤에도 자기 자신과 일치했을 것입니다. `to_vec1` 이 생긴 커널은 목록에 오르기
  전까지 스위트를 빨갛게 만들고, 없어진 커널은 사라진 이유로 계속 거절당하지 않습니다.
* `test_every_host_readback_in_aten_is_classified`
  <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_every_host_readback_in_aten_is_classified present -->
  스캔이 **헬퍼 호출을 이름으로 한 단계만** 따라간다는 것이 이 방식의 한계입니다. 두 단계
  아래 새로 생긴 되읽기는 어떤 op 에도 닿지 않고 조용히 빠집니다. 그래서 더 강한 것을
  단언합니다 — `aten.rs` 안에서 되읽기 표식을 가진 **모든 함수**는 (a) 이미 거절되는 커널,
  (b) 여섯 헬퍼 중 하나, (c) **이유가 적힌 면제 목록**의 하나여야 합니다. 셋 중 무엇도
  아니면 사람이 분류할 때까지 초록이 되지 않습니다.
* 두 테스트 모두 **Metal 이 없는 기계에서도 돕니다.** 소스에 대한 주장이기 때문이고, 그 아래
  커널을 바꿀 가능성이 가장 큰 것이 바로 게이트를 실행해 볼 수 없는 러너들이기 때문입니다.

### 3.2 왜 "안전한 op 의 허용 목록" 이 아닌가

허용 목록은 위험 집합을 알 수 없을 때의 보수적 선택이고, **여기서는 알 수 있습니다**(§1.1).
그리고 허용 목록에는 값이 있습니다: 과잉 거절.

`abs` 와 `neg` 는 **정수 경로에서만** 되읽고 부동소수는 candle 이 GPU 에서 합니다. op 단위
허용 목록은 그 구분을 표현하지 못해 부동소수까지 거절합니다.

**그런데 지금 고른 거절 목록도 같은 대가를 치릅니다** — `abs`·`neg`·`pow`·`max.other` 는
float 에서 GPU 에 남는데 통째로 거절됩니다. 이것을 고치려면 문 앞에서 `tag.is_floating_point()`
를 다시 쓰는 수밖에 없고, **다시 쓴 불변식은 표류하며, 여기서 표류는 열리는 쪽으로**
— 즉 조용히 — 실패합니다. 닫히는 쪽 실패(과잉 거절)는 시끄럽고 메시지에 이유가 적힙니다.
그래서 op 단위를 유지하고, 그 대가를 여기 적어 둡니다.

### 3.3 거절하지 **않는** 되읽기 둘

스캔은 이 둘도 찾아내므로, 이유 없이 빼면 실수로 보입니다. `MPS_READBACK_BUT_ALLOWED` 에
이름과 이유가 함께 있습니다.

| op | 왜 다른가 |
|---|---|
| `aten._local_scalar_dense.default` | `.item()`. **되읽기 자체가 요청된 것**이다 — `.cpu()` 와 같은 부류이고, 거절하면 장치에서 값을 읽을 방법이 없어진다 |
| `aten.uniform_.default` | 자기가 쓰는 텐서를 **읽지 않는다.** 표식은 호스트에서 만든 난수와 *상수*의 dtype 왕복(`narrow_roundtrip_f32`)에 있다. 틀린 장치에서 계산될 입력 자체가 없다 |

---

## 4. 비용

CPU 디스패치는 **아무것도 내지 않습니다.** 게이트는 `check_devices_agree` 가 이미 답한
`Where::Dense(device)` 에 대한 `matches!` 뒤에만 있고, 그 스캔은 이 변경 이전부터 문에
있었습니다(`docs/DEVICE_ABS.md` §6 이 그 스캔의 값을 잰 곳입니다). mps 디스패치는 54개
`&'static str` 표의 선형 탐색 하나를 냅니다.

golden 은 `cpu` 에서 돌므로 **움직이지 않아야 하고, 움직이지 않았습니다** — 8921/8921,
ops=222. 그것이 이 라운드의 음성 대조군입니다.

---

## 5. `run.sh` — 거짓말하던 스킵 (docs/VULKAN3.md §6.1)

§6.1 이 기록한 함정: `DYLD_LIBRARY_PATH=... sh rust/torch_c/pytests/run.sh` 는 로더를 정확히
가리켜도 vulkan 테스트 넷을 전부 스킵했고, **스킵 줄은 "no vulkan" 이라고 말했습니다.**
macOS SIP 가 `/bin/sh` 를 exec 할 때 `DYLD_*` 를 떼어내기 때문입니다.

**스크립트가 그 변수를 복원할 수는 없습니다 — 애초에 받은 적이 없습니다.** 그래서 두 가지를
했습니다.

1. **통과용 이름을 하나 만든다.** <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/run.sh TORCH_C_DYLD_LIBRARY_PATH present -->
   `TORCH_C_DYLD_LIBRARY_PATH` 는 `DYLD_*` 가 아니므로 exec 를
   살아서 통과하고, `run.sh` 가 파이썬을 부르기 직전에 `DYLD_LIBRARY_PATH` 로 다시 내보냅니다.

2. **함정을 탐지해 이름을 댄다.** 떼어진 변수는 흔적을 남기지 않지만 **서명은 남깁니다** —
   `VK_DRIVER_FILES` 는 `DYLD_*` 가 아니라 SIP 가 건드리지 않습니다. 둘 중 하나만 세팅된
   상태는 **둘 다 세팅하고 하나를 잃은 사람**의 모양입니다. `run.sh` 가 그때 경고하고,
   <!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py _sip_stripped_the_loader_path present -->
   `_vulkan_or_skip` 의 스킵 줄도 "no vulkan" 대신 SIP 와 통과용 변수를 말합니다.

**실측 — 이것이 §6.1 이 불가능하다고 적은 조합입니다:**

```
$ V=~/Library/Android/sdk/emulator/lib64/vulkan
$ TORCH_C_DYLD_LIBRARY_PATH=$V VK_DRIVER_FILES=$V/libkosmickrisp_icd.json \
      PYTHON=$PY sh rust/torch_c/pytests/run.sh

461 ok,  스킵 0개,  EXIT=0,  DOCWATCH PASS 452/452
```

`run.sh` 를 통해서, mps 와 vulkan 이 둘 다 진짜 M1 위에서, **하나도 스킵하지 않고** 돌았습니다.
옛 방식(`DYLD_LIBRARY_PATH=... sh run.sh`)으로 부르면 여전히 스킵하지만, 이제 스킵 줄이
왜인지를 말합니다.

---

## 6. 게이트

| | 값 |
|---|---|
| `pytests/run.sh` (로더 없음) | **461 ok**, 0 FAIL, EXIT=0 |
| 같은 스위트, `TORCH_C_DYLD_LIBRARY_PATH` 로 로더 지정 | **461 ok**, **스킵 0개**, EXIT=0 |
| DOCWATCH | **PASS — 452/452** |
| golden `compare.py` | 이 회차 기준 **8921/8921**, ops=222 — 이 회차가 **움직이지 않았다**는 뜻이지 그 수가 고정이라는 뜻이 아니다. 같은 배치의 `docs/FIXES.md` 가 `fmod` 로 224 로 올렸다. `eq` 로 적었다가 그 병합에서 바로 터졌고, 다른 회차가 올릴 수 있는 수는 `ge` 로 적는다 <!-- DOCWATCH: count golden_cases_passed ge 8921 --> <!-- DOCWATCH: count golden_ops_covered ge 222 --> |
| `aarch64-linux-android` | EXIT=0 (`scripts/device_android.sh build`) |
| `aarch64-apple-ios-sim` | EXIT=0 (`PYO3_CONFIG_FILE` 레시피) |
| 가드 무력화 시 | `test_an_mps_op_that_would_compute_on_the_cpu_is_refused_and_names_the_op` **FAIL**, EXIT=1 |

마지막 줄이 앞의 줄들을 의미 있게 만듭니다. 실패할 수 없는 검증은 검증이 아니므로,
`mps_host_readback_gate` 를 `if true` 로 바꿔 다시 빌드하고 실제로 빨간 것을 확인했습니다 —
`aten.nonzero.default must not compute on the CPU under mps`.

---

## 7. 남은 것

* **dtype 별 정밀도.** §3.2 의 과잉 거절 — `abs`·`neg`·`pow`·`max.other` 의 부동소수 경로는
  GPU 에 남는데 거절됩니다. 이것을 여는 유일한 안전한 방법은 그 커널들이 정수 경로를
  되읽지 **않도록** 고치는 것이지, 문 앞에서 dtype 을 다시 판단하는 것이 아닙니다.
* **`_softmax` 를 GPU 로.** 54개 중 이것 하나가 `mps` 에서 모델을 돌리는 것과 못 돌리는 것을
  가릅니다. `read_flat`/`widen_f64` 를 지나지 않는 경로가 필요합니다.
* **`f32 -> f64` 가 Metal 에 없다.** `widen_f64` 를 부르는 모든 것의 벽이고, `read_flat` 의
  부동소수 경로가 조용하지 않았던 이유이기도 합니다. candle 쪽 문제입니다.
* **`mps` 의 f64 `full`** — `VULKAN3.md` §7 이 남긴 그대로입니다.
