# 장치 추상 — 지금 `torch.device` 가 무엇인지 재고, 실제 경로가 요구하는 만큼 채웠다

`DESIGN.md` §11.1 이 장치 추상을 **공통 기반**으로 지목했습니다. 연합 학습이 집합 통신 위에 서고,
집합 통신이 백엔드 위에 서고, 백엔드가 장치 위에 섭니다 — 랭크가 가리킬 것이 없으면 분산은
껍데기입니다. 이 문서는 그 맨 아래 칸을 **먼저 실측하고** 그 결과에 따라 채운 기록입니다.

> **결론 먼저.** `torch.device` 라벨 자체는 대체로 동작하고 있었고, **라벨을 소비하는 쪽이 전부
> 죽어 있었습니다.** `nn.Module.to("cpu")` · `.cpu()` · `.float()` · `Module._apply` 가 벽 넷을
> 연달아 만나 하나도 통과하지 못했습니다. 82 개 스펠링 실측에서 상류와 **일치 41 → 56** 이 되었고,
> 남은 26 개 중 6 개는 "이 호스트에는 MPS 가 있고 우리 빌드에는 가속기가 없다" 는 **빌드 차이**,
> 5 개는 같은 부재(CUDA)를 다른 예외 타입으로 말하는 것입니다 — 둘 다 결함이 아닙니다(§9).
>
> **`meta` 장치와 `with torch.device(...)` 는 구현하지 않았습니다.** 둘 다 값이 아니라 구조가
> 걸린 문제이고, 무엇이 걸리는지는 §7 에 적었습니다.
>
> **(2026-08-25 추기) 둘 다 이후에 구현됐습니다 — `docs/devices/META.md`.** §7.1 이 요구한 "저장소
> 없는 표현"이 `PyTensorBase::Repr` 로 들어갔고, §7.2 가 요구한 "스택과 그것을 상의하는
> 팩토리"가 둘 다 들어갔습니다. 그리고 그 작업이 이 문서의 §10 미해결 하나를 닫으면서
> **게이트의 버그를 찾았습니다** (`META.md` §5).

측정일 2026-08-25, `develop`(`d6b0684`) 기준. 호스트 `darwin/arm64`, CPython 3.13.0,
상류 torch 2.13.0 (`/Volumes/macMini/caches/spike-venv`).

---

## 1. 무엇을 어떻게 쟀나

**설계부터 하지 않았습니다.** 같은 프로브 스크립트를 두 torch 에 대해 돌리고 전사(transcript) 를
줄 단위로 비교하는 방식입니다 — 한쪽은 벤더 트리(`PYTHONPATH=$PWD/torchnative/src/main`,
`TORCH_USE_RTLD_GLOBAL=1`), 다른 쪽은 spike-venv 의 상류 torch. 각 줄은
`이름 | 값` 또는 `이름 | <예외타입: 메시지>` 이고, **아무것도 단언하지 않습니다.** 전사가 곧
측정값입니다.

세 벌을 돌렸습니다.

| 프로브 | 무엇을 묻는가 | 항목 수 |
|---|---|---|
| 표면 프로브 | `torch.device` 의 생성자 · 속성 · 컨텍스트 · 가용성 · 텐서/모듈 쪽 스펠링 | 82 |
| 경로 프로브 | 2 층 모델을 실제로 만들고 옮기고 저장할 때 무엇이 걸리는가 | 40 |
| 디스패처 프로브 | 각 스펠링이 **ATen 디스패처에 닿는가** (`TorchDispatchMode` 계측) | 18 |

세 번째가 설계 판단을 가장 많이 바꿨습니다 — §3.3.

`_aten_implemented()` 가 권위 있는 출처라는 규정대로, `aten.rs` 를 grep 해서 op 목록을 만든 곳은
없습니다. 커버리지는 골든 하네스가 `_C._aten_implemented()` 에게 물어서 정합니다.

---

## 2. 실측 — 작업 전 `torch.device` 표면

82 개 스펠링 중 **상류와 일치 41, 불일치 41.** 불일치를 성격별로 묶으면 이렇습니다.

### 2.1 라벨 자체는 대체로 맞았다

`torch.device("cpu")`, `.type`, `.index`, `str`, `repr`, `__eq__`, `__hash__`, `x.device`,
`torch.zeros(2, device="cpu")`, `x.to("cpu") is x`, `x.to(torch.device("cpu")) is x` — 전부
상류와 같았습니다. `docs/numerics/BOOL.md` 의 "candle 의 `Device` 를 감싸지 않는다. 라벨이고, 쓸 때
`resolve()` 한다" 는 결정은 **여기까지는 버티고 있었습니다.**

### 2.2 라벨의 구멍 — 생성과 검증

| 스펠링 | 상류 | 우리(작업 전) |
|---|---|---|
| `torch.device(torch.device("cpu"))` | `device(type='cpu')` | `TypeError: 'device' object is not an instance of 'str'` |
| `torch.device(type="cpu")` | `device(type='cpu')` | `TypeError: unexpected keyword argument 'type'` |
| `torch.device("nosuchdevice")` | `RuntimeError: Expected one of cpu, cuda, ... device type` | **`device(type='nosuchdevice')`** |
| `torch.device("cuda", -1)` | `RuntimeError: Device index must not be negative` | **`device(type='cuda', index=-1)`** |
| `torch.device("")` | `RuntimeError: Device string must not be empty` | **`device(type='')`** |
| `torch.device("cuda:1").__reduce__()` | `(torch.device, ('cuda', 1))` | `TypeError: cannot pickle 'device' object` |
| `torch.device(0)` | `device(type='mps', index=0)` | `TypeError: 'int' object is not an instance of 'str'` |

**굵은 세 줄이 성격이 다릅니다.** 나머지는 "받아야 할 것을 못 받는" 것이고, 이 셋은 **받으면 안
되는 것을 받는** 것입니다. `torch.device("cuad")` 가 조용히 만들어졌다가 나중에 `resolve()` 에서
"device not available in torch._C shim: cuad" 로 죽습니다 — 아무도 요청한 적 없는 장치 이름을
대면서. 라벨이 아무 문자열이나 받으면 그것은 라벨이 아니라 자유 서술 메모입니다.

`torch.device(0)` 이 상류에서 `mps:0` 인 것이 흥미롭습니다 — 상류는 벌거벗은 정수를 **현재
가속기의 인덱스**로 읽습니다. 이 호스트에 MPS 가 있어서 그렇게 나온 것이고, 가속기가 없는
빌드에서 무엇이 나오는지는 **이 기계에서 잴 수 없었습니다.**

### 2.3 라벨을 소비하는 쪽은 통째로 죽어 있었다

이쪽이 실제 피해입니다.

| 스펠링 | 우리(작업 전) |
|---|---|
| `model.to("cpu")` | `NotImplementedError: torch._C._nn._parse_to` |
| `model.to(torch.device("cpu"))` | 〃 |
| `model.to(torch.float32)` | 〃 |
| `model.cpu()` | `NotImplementedError: TensorBase.cpu` |
| `model.float()` | `NotImplementedError: TensorBase.is_floating_point` |
| `model._apply(f)` | `NotImplementedError: torch._has_compatible_shallow_copy_type` |
| `x.cpu()` | `NotImplementedError: TensorBase.cpu` |
| `x.is_cpu` / `x.is_cuda` / `x.get_device()` | `NotImplementedError` |
| `torch.default_generator.device` | `NotImplementedError: Generator.device` |
| `torch.get_device_module()` | `NotImplementedError: torch._C._get_accelerator` |

**`nn.Module.to(...)` 는 어떤 철자로도 통하지 않았습니다.** 체크포인트를 읽어 모델에 올리는
길(`docs/models/CKPT.md`)이 열려 있는데 그 모델을 장치로 옮기는 길은 첫 걸음도 못 뗀 상태였습니다.
그리고 벽이 **하나가 아니라 넷을 연달아** 만납니다 — `_parse_to` 를 채우면 `Tensor.data =` 가
`AttributeError` 로 나오고, 그것을 채우면 `_has_compatible_shallow_copy_type`, 그다음
`is_floating_point`. 이것이 §5 가 넷을 한 묶음으로 채운 이유입니다.

### 2.4 상류가 이 호스트에서 세그폴트한다

측정 도중 발견한 것이라 여기 적어 둡니다.

```
$ python -c "import torch; a=torch.tensor([1.,2.]); m=torch.tensor([1.,2.],device='mps'); torch.cat([a,m])"
EXIT=139        # SIGSEGV, 출력 없음
```

torch 2.13.0 + MPS 에서 **`torch.cat([cpu_tensor, mps_tensor])` 가 프로세스를 죽입니다.**
같은 조합으로 `a + m` 은 `RuntimeError` 를 제대로 냅니다. **장치 검사가 op 마다 따로 있고,
`cat` 에는 없다**는 뜻이고, 이것이 §6 의 설계 판단 하나를 그대로 결정했습니다.

---

## 3. 설계 판단

### 3.1 `torch.device` 는 값인가 라벨인가 — **라벨이다. 그리고 이제 검증되는 라벨이다**

**유지합니다.** 근거는 상류 대조입니다: `torch.device("cuda")` 는 CUDA 없는 빌드에서 **만들어지고**,
쓸 때만 실패합니다(측정: `AssertionError: Torch not compiled with CUDA enabled` — 생성이 아니라
사용 지점에서). candle 의 `Device` 는 반대로 살아 있는 핸들을 들고 있어서 백엔드 없는 장치를
표현할 수 없습니다. 라벨이 아니면 상류 의미론을 낼 수 없습니다.

**바꾼 것은 라벨의 어휘가 닫혀 있어야 한다는 점입니다.** 상류는 20 개 장치 타입의 고정 목록에
대해 **생성 시점에** 검증하고, 우리는 아무것도 검증하지 않았습니다(§2.2). 이것이 취향 문제가
아닌 이유는 §11.1 의 다음 칸에 있습니다 — **`torch.distributed` 의 `register_backend` 가
키로 쓰는 어휘가 바로 이 목록**입니다. 라벨의 어휘가 열려 있으면 백엔드 등록의 키 공간도
열려 있게 됩니다.

목록은 상류 에러 메시지에서 글자 그대로 전사했습니다 (`device.rs::DEVICE_TYPES`). 여기서
읽어낼 것이 하나 더 있습니다: **`vulkan` 은 이미 torch 의 장치 타입이고, `npu` 는 없습니다.**
즉 안드로이드 GPU 백엔드는 새 철자를 필요로 하지 않고, NPU 는 `privateuseone` 으로 들어와야
합니다 — 상류가 정확히 그 용도로 비워 둔 칸입니다.

### 3.2 텐서가 자기 장치를 들고 있어야 하는가 — **오늘은 아니고, 언제부터인지는 정확히 말할 수 있다**

지금 `PyTensorBase.device` 는 candle 핸들에서 라벨을 **재구성**합니다
(`PyDevice::from_candle`). 이 방향은 **손실이 있고, 그 손실이 하중을 받습니다.**

```rust
Device::Cuda(_) => Self { kind: "cuda", index: Some(0) },   // 인덱스가 하드코딩
Device::Metal(_) => Self { kind: "mps",  index: Some(0) },
```

candle 의 `Cuda`/`Metal` 변형은 서수를 들고 있지만 이 크레이트가 읽을 수 없습니다(두 feature 가
꺼져 있어 내부 타입이 불투명). 그래서 `torch.zeros(2, device="cuda:1")` 을 왕복시키면
`cuda:0` 이 나옵니다.

**오늘 이것은 도달 불가능합니다** — `resolve()` 가 CPU 아닌 라벨을 전부 거부하므로 저 두 팔에는
핸들이 만들어지지 않습니다. 그래서 `unreachable!()` 로 막지 않고 **보이게 남겼습니다**: feature
하나를 켜는 순간 이 코드가 리뷰에 걸려야지, 조용히 텐서에 틀린 라벨을 붙이면 안 됩니다.

**언제 텐서가 라벨을 들어야 하는가:** 같은 종류의 장치가 둘 이상 주소 지정 가능해지는 순간입니다.
그때는 이 저장소가 **이미 한 번 푼 모양**입니다 — `PyTensorBase.tag: TorchDType` 이 존재하는
이유가 정확히 이것이기 때문입니다(`BOOL.md` §5-B). candle 의 dtype 이 torch 의 dtype 에 대해
손실이 있어서(`bool` 과 `uint8` 이 둘 다 `U8`) 래퍼가 태그를 따로 듭니다. 장치는 같은 모양의
문제입니다 — candle 의 `Device` 는 torch 의 장치 라벨에 대해 손실이 있습니다(인덱스, 그리고
candle 에 변형 자체가 없는 `meta`·`vulkan`·`xpu`·`privateuseone`).

**지금 넣지 않은 이유는 수요 기반 원칙(`DESIGN.md` §6)입니다.** 장치가 하나뿐인 지금 저장된
라벨은 순수한 중복이고, `aten.rs` 6467 줄의 모든 `finish()`/생성자를 건드리는 대가는 실측 이득
0 에 대해 지불하는 것입니다.

### 3.3 디스패처가 장치를 어떻게 보는가 — **대부분의 스펠링은 디스패처를 지나지 않는다**

이것이 이번 작업에서 가장 많이 바꾼 판단입니다. `TorchDispatchMode` 로 상류를 계측한 결과:

```
t.is_floating_point()   []                          t.to(float64)   ['aten._to_copy.default']
t.is_complex()          []                          t.to('mps')     ['aten._to_copy.default']
t.cpu()                 []                          t.double()      ['aten._to_copy.default']
t.to('cpu')             []                          t.to('cpu', copy=True)
t.get_device()          []                                          ['aten._to_copy.default']
t.is_cpu                []                          m.to(float64)   ['aten._to_copy.default'] x2
t.device                []
t.float()               []
m.to('cpu')             []
m.cpu()                 []
Generator.device        []
```

**왼쪽 열은 디스패처 호출을 하나도 만들지 않습니다.** TensorImpl 에 이미 앉아 있는 메타데이터를
읽을 뿐입니다. `tools/golden/cases.py` 가 이미 기록해 둔 "50 개 중 9 개는 ATen 디스패처에 전혀
닿지 않는다"(`device`, `dim`, `dtype`, `shape`, ...) 와 **같은 가족**입니다.

두 가지가 여기서 따라 나옵니다.

1. **§5 에서 추가한 스펠링들은 두 번째 문이 아닙니다.** `DESIGN.md` §6 이 금지하는 것은 계산이
   문을 우회하는 것이고, 이것들은 계산하지 않습니다. 실제로 무언가를 바꾸는 `.to()` 만 디스패처에
   내려가고, 그것은 이미 커널과 골든 케이스가 있는 `aten._to_copy.default` 입니다.
2. **골든 케이스를 붙일 자리가 없습니다.** 하네스는 `_aten_implemented()` 의 op 마다 케이스를
   요구하는데 이번에 추가된 op 이 없습니다(96 그대로). `<no case builder registered>` 위험은
   따라서 없고, 대신 `pytests/test_shim.py` 에 스펠링별 테스트를 붙였습니다(§8).

**그러므로 장치별 분기는 `_aten_dispatch` 안이 아니라 그 아래(커널)에 들어갑니다.** 문에 들어가는
것은 분기가 아니라 **거부**입니다 — §6.

### 3.4 혼합 장치 연산 — 상류를 재서 맞췄다

MPS 가 있는 이 호스트에서 상류를 직접 쟀습니다.

| 호출 | 상류 torch 2.13.0 |
|---|---|
| `cpu + mps` | `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!` |
| `mps + cpu` | 같은 메시지 |
| `mps + 1.0` (파이썬 스칼라) | **통과**, 결과는 `mps:0` |
| `cpu + tensor(1.0, device='mps')` (0-차원 텐서) | **거부** |
| `torch.mm(cpu, mps)` | `RuntimeError: Tensor for argument #1 'mat1' is on CPU, but expected it to be on GPU` |
| `torch.where(cpu_cond, cpu, mps)` | 첫 번째 메시지 |
| `torch.add(cpu, cpu, out=mps)` | 첫 번째 메시지 |
| `cpu.copy_(mps)` | **통과**, 결과는 `cpu` |
| `torch.cat([cpu, mps])` | **SIGSEGV** |
| `cpu + meta` | `RuntimeError: Tensor on device meta is not on the expected device cpu!` |

**규칙은 "암묵적 전송은 절대 없다" 입니다.** 예외는 둘 — 파이썬 스칼라(텐서가 아니므로), 그리고
`copy_`(전송이 그 op 의 정의). 0-차원 텐서는 예외가 **아닙니다**, 이건 추측하지 않고 따로 쟀습니다.

그리고 세 번째 관찰: **메시지가 op 마다 다르고 하나는 세그폴트합니다.** 상류의 검사는 커널 안에
있고, 커널마다 따로 기억해야 합니다. `cat` 은 잊었습니다.

---

## 4. 별칭 미해결과 어떻게 만나는가

`docs/kernels/OPS4.md` §8 이 남긴 미해결은 이것입니다: 이 셰임에는 op 별 별칭 규칙이 없고 규칙이 하나
있습니다 — **어떤 뷰를 통해서도 쓰기가 원본에 닿지 않습니다.** `replace_with` 가 저장소에 쓰는
대신 래퍼가 가리키는 텐서를 갈아끼우기 때문입니다.

장치 작업이 이 미해결과 만나는 지점은 **정확히 두 곳**입니다.

### 4.1 `Tensor.data =` 는 `replace_with` 이고, **이 철자에서는 상류와 일치한다**

`nn.Module._apply` 는 모든 `.to()`/`.cpu()`/`.float()` 을 `param.data = param_applied` 로
끝냅니다(`module.py:995`). 세터가 없어서 `AttributeError` 였고, `_shim_set_data` →
`replace_with` 로 채웠습니다.

`replace_with` 의 기록된 발산은 "할당 *전에* 뜬 뷰가 따라오지 않는다" 입니다. **상류의
`.data =` 도 TensorImpl 을 통째로 교체하므로 이전 뷰는 마찬가지로 따라오지 않습니다.** 즉 이
철자에 관한 한 두 쪽이 같고, OPS4 의 미해결(뷰를 **통한 쓰기**가 원본에 닿는가)은 다른 질문입니다.
`.data =` 는 쓰기가 아니라 재바인딩입니다.

`requires_grad` 는 일부러 건드리지 않습니다 — 상류의 `.data =` 도 그렇고, `_apply` 가 그 성질에
기대어 `Parameter` 를 `Parameter` 로 유지합니다. 벤더 트리 테스트가 파라미터 **객체 동일성**이
보존되는 것까지 확인합니다.

### 4.2 장치가 둘이 되면 `.to(device)` 가 진짜 복사를 만들고, 그때 이 규칙이 시험대에 오른다

지금 `x.to("cpu")` 는 `self` 를 그대로 돌려줍니다(측정: 상류도 같음). 복사가 없으니 별칭 질문이
생기지 않습니다. 장치가 둘이 되면 `x.to("mps")` 는 **반드시 새 저장소**를 만들고, 상류에서도
`a.to("mps") is a` 는 `False` 입니다(측정) — 즉 **장치 간 이동은 상류에서도 별칭이 아닙니다.**

그러므로 §3.2 를 실행할 때(텐서가 라벨을 들 때) 별칭 규칙을 함께 바꿀 필요는 없습니다. 두
미해결은 서로 독립입니다. **다만 하나 겹칩니다:** `copy_` 는 상류에서 장치 간 전송을 허용하는
유일한 op 이고(§3.4), 이 셰임의 `copy_` 는 `replace_with` 로 구현돼 있어 **수신자의 뷰에 쓰기가
보이지 않습니다.** 장치 간 `copy_` 를 구현하는 날 그것이 OPS4 의 미해결을 정면으로 건드립니다.
오늘은 도달 불가능합니다.

---

## 5. 구현한 것

`resolve()` 가 여전히 `cpu` 하나만 받습니다. **백엔드를 늘리지 않았습니다** — 늘린 것은 라벨이
정확한가, 그리고 라벨을 소비하는 길이 뚫려 있는가입니다.

### 5.1 라벨 (`rust/torch_c/src/device.rs`)

- `DEVICE_TYPES` — 상류 20 개 목록. 생성 시점 검증.
- `parse_device_string` — 빈 문자열 · 공백 · 대문자 · 음수 인덱스 · 비숫자 인덱스를 상류와
  **같은 예외 타입과 같은 메시지**로 거부.
- `PyDevice::coerce` — 문자열 · `device` · 정수를 받는 하나의 입구. `torch.device` 의 멱등성이
  벤더 트리 전체가 의존하는 성질이라 이것이 핵심입니다.
- `__new__` 를 손으로 파싱 — 첫 인자가 상류에서 `type` 과 `device` 두 이름을 갖는데 PyO3 는
  파라미터에 이름을 하나 줍니다. 둘 다 실제로 쓰이는 철자이고 둘 다 쟀습니다.
- `__reduce__` — 피클 가능. 인덱스 없는 형태는 두 번째 원소를 **떨어뜨립니다**(상류와 동일).
- `same_physical_device` — `==` 와 **다른** 관계. `cpu` 와 `cpu:0` 은 라벨로서 같지 않지만
  (측정: 해시도 다름) 한 장치를 가리킵니다.
- `_shim_same_device` — 위 규칙을 파이썬에서 부를 수 있게 노출. **이 크레이트에는 돌릴 수 있는
  Rust 단위 테스트가 없습니다** — `crate-type = ["cdylib"]` + `extension-module` 이라
  `cargo test` 가 `dyld: symbol not found in flat namespace '_PyExc_BaseException'` 로 죽습니다
  (변경 전에도 같음, 확인함). `#[test]` 를 넣으면 아무 데서도 안 도는 커버리지처럼 보입니다.

### 5.2 `_to_copy` 의 기본 장치 버그 (`aten.rs`)

`device_arg` 가 인자 부재 시 **무조건 `Device::Cpu`** 를 돌려주고 있었고, `_to_copy` 가 그것을
씁니다. `aten::_to_copy(self, dtype=None, ..., device=None, ...)` 에서 `device=None` 은 "있던
자리에 있어라" 이지 "CPU 로 가라" 가 아닙니다. `device_arg_or(..., fallback)` 로 갈라서
`_to_copy` 는 입력 텐서의 장치를 기본값으로 씁니다.

**장치가 하나뿐이라 관측되지 않는 버그였고**, 그래서 실패하는 테스트가 아니라 코드를 읽어서
찾았습니다. 장치가 둘이 되는 날 조용한 전송이 됩니다.

### 5.3 모듈 쪽 장치 도로 (`bootstrap.py`)

벽 넷을 순서대로:

| 이름 | 무엇이 막혔었나 | 근거 |
|---|---|---|
| `torch._C._nn._parse_to` | `Module.to` 의 유일한 입구 (`module.py:1340`) | 벤더 트리 자신의 폴리필(`_dynamo/polyfills/torch_c_nn.py`)을 기준으로, **실제 파서와 어긋나는 3 곳을 실측으로 교정** |
| `Tensor.data =` | `_apply` 의 in-place 분기 (`module.py:995`) | §4.1 |
| `torch._has_compatible_shallow_copy_type` | `_apply` 의 분기 판단 (`module.py:938`) | 상류가 dtype·장치·`Parameter` 차이에도 전부 `True` 인 것을 측정 |
| `TensorBase.is_floating_point()` / `is_complex()` | `Module.to` 의 `convert(t)` (`module.py:1365`) | 디스패처에 안 닿음(§3.3), dtype 이 이미 답을 들고 있음 |

`_parse_to` 에서 폴리필과 실제 파서가 어긋난 3 곳:

- 폴리필은 위치 인자 1 개, 실제 파서는 **4 개**(`device, dtype, non_blocking, copy`).
- 폴리필은 `memory_format` 을 위치로 받고, 실제 파서는 **키워드 전용**.
- `copy` 는 `.pyi` 오버로드에 있고 실제 파서는 **런타임에 거부**합니다
  (`RuntimeError: .to() does not accept copy argument`). 조용히 받으면
  `Module.to(copy=True)` 가 에러가 아니라 무동작이 됩니다.

### 5.4 텐서 쪽 스펠링

`x.is_cpu`, `x.is_cuda` 는 `tensor.rs` 의 게터입니다 — `is_meta` 와 같은 자리에서 같은 방식으로
(`PyDevice::from_candle(...).kind` 비교) 유도되므로 셋이 `device` 와 어긋날 수가 없습니다.
`x.cpu()`, `x.cuda()`, `x.get_device()`, `x.is_floating_point()`, `x.is_complex()` 는
`bootstrap.py` 에 있습니다 — 앞의 둘은 `_to_copy` 클로저를 재사용해야 "이미 거기 있으면 self"
단락과 "없는 백엔드면 거부" 가 한 곳에서 결정되고, 뒤의 둘은 dtype 이 이미 답을 들고 있습니다.

`get_device()` 만은 **때운 것이 아니라 PyO3 제약** 때문에 그 자리입니다: `#[pymethods]` 가
`device` **게터**와 `get_device` **메서드**에서 똑같이 `__pymethod_get_device__` 를 만들어 내고,
이 크레이트는 `multiple-pymethods` 없이 빌드됩니다. `tensor.rs` 에 그 사실을 주석으로 남겨
두었습니다.

`Tensor.to` 의 위치 불리언 처리도 고쳤습니다. 기존 코드는 **모든** 불리언을 `copy` 로 접었고
(`copy = copy or value`), `Module.to` 는 매 호출마다 `t.to(device, dtype, non_blocking)` 으로
`non_blocking` 을 위치로 넘깁니다. 지금은 순서를 세어 첫 번째가 `non_blocking`, 두 번째가 `copy`
입니다. 고치기 전이라면 `model.to("cpu", non_blocking=True)` 가 건드리지 말라고 한 파라미터를
전부 복사했을 것입니다.

### 5.5 "지금 가속기가 뭔가" 를 묻는 두 이름은 답이 다르다

이건 실측이 아니라 벤더 소스를 읽어서 나온 것이고(이 호스트에는 MPS 가 있어서 "가속기 없음"
분기를 낼 수 없음), 그래서 소스 근거로 적습니다.

| 이름 | 호출자 | `None` 처리 | 우리 답 |
|---|---|---|---|
| `_get_default_device` | `torch.get_default_device()` | — | `"cpu"` (**문자열**, 상류 실측) |
| `_get_accelerator` | `torch.get_device_module()` → `.type` (`__init__.py:2978`) | **가드 없음** | `device("cpu")` |
| `_accelerator_getAccelerator` | `torch.accelerator.current_accelerator()` (`accelerator/__init__.py:128`) | `is not None` 명시 | `None` |

둘을 같은 값으로 답하면 한쪽이 깨집니다. `get_device_module()` 에 `None` 을 주면 `AttributeError`,
`current_accelerator()` 에 `device("cpu")` 를 주면 CPU 를 가속기라고 주장하게 됩니다.

`torch.backends.mps.is_available()` 도 `False` 로 채웠습니다 — candle 의 `metal` feature 가
꺼져 있다는 사실의 파이썬 쪽 표현입니다.

> **(2026-09-12 추기) closed.** `torch.backends.mps.is_available()` 과 `is_built()` 는 이제 둘 다 `torch._C._mps_probe()` 를 통해 `True` 를 반환합니다 (`docs/numerics/DTYPEDEV.md` §2).

`Generator.device` 는 **클래스 속성**을 덮어썼습니다. `_install_default_generator` 의 기존 주석이
"장치는 인스턴스별 값이므로 클래스 속성을 덮는 것은 틀린 모양" 이라고 반대하고 있었는데, 그 논증은
상류에 대해서는 맞고 여기에 대해서는 틀립니다 — `resolve()` 가 `cpu` 외의 라벨을 전부 거부하므로
이 빌드가 만들 수 있는 `Generator` 는 전부 CPU 제너레이터이고, 인스턴스 속성이 들 두 번째 값이
없습니다. 두 번째 백엔드가 오는 날 다시 틀려집니다.

---

## 6. 문에 놓은 혼합 장치 거부 — 그리고 그것이 얼마인가

`_aten_dispatch` 가 텐서 인자들의 장치가 갈리면 거부합니다(`check_devices_agree`).

**어디에 두는가가 판단이지 둘지 말지가 아닙니다.** §2.4/§3.4 가 보여준 대로 상류는 커널마다
검사하고, 그래서 메시지가 op 마다 다르며 `cat` 은 검사를 **잊었고 세그폴트합니다.** 커널마다 하는
검사는 새 커널마다 기억해야 하는 검사입니다. 이 셰임에 문이 하나인 이유가 바로 이런 것이 갈 곳을
만들기 위해서입니다. 메시지는 상류의 가장 흔한 것을 그대로 씁니다.

**오늘 이 검사는 발화할 수 없습니다.** `resolve()` 가 CPU 아닌 라벨을 전부 거부하므로 모든
텐서가 CPU 위에 있습니다. 숨기지 않고 적습니다: 테스트가 닿는 것은 **통과하는 절반**입니다 —
장치가 맞는 텐서가 계속 디스패치되어야 하고, 평범한 인자를 통해서도 `Tensor[]` 를 통해서도
그래야 합니다(`cat`/`stack` 의 텐서는 한 단계 아래 숨어 있습니다).

### 비용 — 세 판본을 A/B 로 쟀다

두 산출물이 딱 한 줄(문의 호출)만 다르게 만들어 번갈아 돌렸습니다. 가장 싼 op 을 2 원소 텐서로
20 만 번, 7 회 반복 중 최솟값. `load average 2.5~2.8` (8 코어), 다른 에이전트 없음.

| 판본 | `add.Tensor` | `cat.default` |
|---|---:|---:|
| 문 없음 | 345 ns | 392 ns |
| 인자마다 라벨 재구성 | 424 ns (**+79, +23%**) | 480 ns (+88, +22%) |
| candle 핸들 직접 비교 | 396 ns (+51, +15%) | 445 ns (+53, +14%) |
| **위 + 텐서 캐스트를 먼저** | **366 ns (+21, +6%)** | **450 ns (+58, +15%)** |

두 번의 최적화 모두 **측정이 시켰습니다.** 첫 번째는 `PyDevice::from_candle` 이 인자마다
`String` 을 힙에 할당하고 있던 것이고, 두 번째는 리스트/튜플 캐스트를 텐서 캐스트보다 먼저
시도해서 흔한 경우가 실패하는 타입 검사 두 번을 무는 것이었습니다.

**두 번째 최적화는 `cat` 에는 도움이 되지 않았습니다** (445 → 450 ns, 잡음 범위). 당연한
결과입니다 — `cat` 의 인자는 리스트라서 텐서 캐스트를 먼저 시도하면 그것이 실패하는 검사가
됩니다. `add` 가 366 ns 로 내려간 것과 맞바꾼 것이고, 디스패치의 압도적 다수가 텐서를 직접
받는 쪽이므로 그렇게 맞바꿨습니다. 두 판본을 다 남기지 않고 하나를 골랐다는 뜻입니다.

**모델 수준에서는 묻힙니다.** `docs/perf/PERF.md` 의 2 층 블록이 2.22 ms 이고 디스패치 수십 회면
21 ns × 수십 = 마이크로초 단위, 0.1% 미만입니다. 하지만 **원소별 op 만 도는 마이크로벤치에서는
6% 가 보입니다** — 그 숫자가 필요한 사람이 있을 수 있으므로 감추지 않고 적습니다.

**측정 한계.** `docs/perf/PERF.md` §0 과 같습니다 — 절대값은 재현되지 않고, 유효한 것은 같은 조건에서
잰 A/B 비율뿐입니다. 벤치 스크립트는 `/tmp/dev_bench.py` 로 저장소 밖입니다.

---

## 7. 하지 않은 것, 그리고 왜

### 7.1 `meta` 장치 — **구조가 걸린다**

> **구현됨 (2026-08-25). `docs/devices/META.md`.** 아래 두 길 중 (b) 를 택했습니다.
> 아래 분석은 그대로 유효합니다.

상류에서 `meta` 는 저장소가 **없는** 텐서입니다: 모양과 dtype 은 있고 바이트는 없습니다.
`torch.zeros(2, device="meta")` 는 할당하지 않고, `.item()` 은 거부하며
(`RuntimeError: Tensor.item() cannot be called on meta tensors`), `.to("cpu")` 도 거부합니다
(`NotImplementedError: Cannot copy out of meta tensor; no data!`). 전부 실측입니다.

**candle 에는 저장소 없는 텐서가 없습니다.** 그래서 두 길뿐입니다 — (a) 진짜로 할당하고 `meta`
라벨을 붙인다, (b) `PyTensorBase` 에 저장소 없는 표현을 만든다. (a) 는 `meta` 의 존재 이유
자체가 할당하지 않는 것이므로 정확히 거꾸로 된 거짓말이고, (b) 는 `aten.rs` 의 모든 커널이
`meta` 를 처리하거나 거부해야 하는 작업입니다.

**이것이 아깝습니다**, 왜냐하면 `meta` 는 **백엔드가 필요 없는 두 번째 장치**이고, 그러므로
이 문서의 모든 "장치가 둘이 되면" 가정을 실제로 시험할 수 있는 **가장 싼 수단**이기 때문입니다.
§6 의 문도, §3.2 의 라벨 운반도, §5.2 의 기본 장치도 전부 `meta` 하나로 관측 가능해집니다.
다음 작업으로 가장 값이 높은 항목이라고 봅니다.

`accelerate` 의 `init_empty_weights` 가 `with torch.device("meta")` 인 것도 여기 걸립니다 —
§11.1 이 지목한 `from_pretrained` 벽 뒤에 이것이 있습니다.

### 7.2 `with torch.device(...)` 와 `set_default_device` — **토치 함수 모드 스택 전체가 걸린다**

> **구현됨 (2026-08-25). `docs/devices/META.md` §8.** 아래의 "스택만 만드는 것은 의미가 없고
> 위험하다" 가 그대로 설계 제약이 되어, 스택과 **팩토리가 그것을 상의하는 쪽** 둘 다
> 들어갔습니다. 976 개를 전부 고칠 필요는 없었습니다 — `_torch_level_function` 이 이미
> 그 전부의 단일 깔때기이기 때문입니다.

상류는 `torch.device.__enter__` 를 `torch.utils._device.DeviceContext` 라는
`TorchFunctionMode` 를 스택에 밀어 넣는 것으로 구현합니다. 필요한 `_C` 이름은
`_push_on_torch_function_stack`, `_pop_torch_function_stack`, `_len_torch_function_stack` 등입니다.

**스택만 만드는 것은 의미가 없고 위험합니다.** 그렇게 하면 `with torch.device("cpu"):` 가
성공하고 `torch.get_default_device()` 가 맞는 값을 돌려주지만, `torch.zeros(2)` 는 모드를
쳐다보지 않으므로 **조용히 무시합니다.** 지금처럼 `NotImplementedError` 로 죽는 편이 낫습니다.

실제로 효과를 내려면 976 개 `_VariableFunctions` 가 매 호출마다 모드 스택을 상의해야 합니다.
한 곳(`_torch_level_function`)에 장치 전용 특례를 넣는 길은 있지만, 그것은 상류의 일반적 기제를
장치 하나짜리 특례로 바꾸는 것이고 §6 에서 21 ns 를 놓고 고민한 것과 같은 자리에 조건 분기를
하나 더 얹는 일입니다. **그리고 이 기능의 주 사용처가 `with torch.device("meta")` 라서 §7.1 과
묶여 있습니다.** 둘을 같이 하는 편이 맞다고 판단했습니다.

### 7.3 `privateuse1` 백엔드 이름 변경

`torch.utils.rename_privateuse1_backend("brainwave")` → `_C._rename_privateuse1_backend` 가
미구현입니다. 이것은 **`torch.distributed` 백엔드 등록의 바로 옆 칸**이므로 §11.1 의 다음
단계에서 필요해질 가능성이 높습니다. 지금 구현하지 않은 이유는 측정된 수요가 없어서입니다 —
어느 경로도 아직 이것을 부르지 않습니다. 다만 `DEVICE_TYPES` 검증이 들어갔으므로, 이름을
바꾸면 그 목록도 함께 바뀌어야 한다는 결합이 생겼습니다. `device.rs` 에 적어 두었습니다.

### 7.4 Metal / Vulkan / NPU — 이번 작업에서 켜지 않았다

`Cargo.toml` 의 `[target.'cfg(target_vendor = "apple")']` 절이 `accelerate` 를 켠 자리이고,
candle 의 `metal` feature 도 **같은 자리**입니다(candle 0.11 의 feature 목록에서 확인:
`accelerate`, `cuda`, `metal`, `mkl`, `nccl`).

**`accelerate` 와 장치 작업은 서로 만나지 않습니다.** `accelerate` 는 **장치가 아니라 CPU 의
BLAS 백엔드**입니다 — `Device::Cpu` 를 빠르게 만들 뿐 새 `Device` 변형을 만들지 않습니다.
`docs/perf/PERF.md` 의 7.0× → 0.97× 가 장치 하나로 얻어진 것이라는 뜻이고, 이번 작업의 어떤 결정도
그것을 바꾸지 않습니다. 골든 2258/2258 이 `accelerate` 켠 상태로 그대로인 것도 확인했습니다
(§8).

`metal` 을 켜면 무엇이 생기는지는 지금 말할 수 있습니다:

- `PyDevice::resolve` 에 `"mps" => Device::new_metal(0)` 한 팔.
- **§3.2 의 하드코딩된 인덱스가 즉시 도달 가능해집니다.**
- **§6 의 문이 발화 가능해집니다** — 처음으로 혼합 장치 입력을 만들 수 있게 됩니다.
- 골든 하네스가 MPS 대조를 하게 되는데, 상류 MPS 는 `cat` 혼합에서 세그폴트합니다(§2.4).
- **Apple 전용이라 안드로이드는 여전히 장치 하나입니다.** 즉 첫 다중 장치 빌드는 타깃 간
  비대칭이 됩니다. 이것이 §3.2 를 "언제 하느냐" 를 정할 때 고려해야 할 점입니다.

켜지 않은 이유는 `DESIGN.md` §11.1 이 이 작업을 "가장 값싸고 가장 많이 알려주는" 단계로 놓았고,
백엔드를 늘리는 것은 그 단계가 아니기 때문입니다. 되돌릴 수 있는 결정이라는 §11.1 의 판단은
그대로 유효합니다.

---

## 8. 판정 — 전부 종료 코드로

파이프로 읽지 않았습니다. 전부 파일로 리다이렉트한 뒤 `$?`.

```
골든           2258/2258, ops=96, pending 0                      EXIT=0
fault value                                                      EXIT=1
fault shape                                                      EXIT=1
fault dtype                                                      EXIT=1
스키마         204/204 (overloads 93/93, methods 111/111)        EXIT=0
pytests        84 ok / 0 fail (run.sh, 골든 self-test 포함)       EXIT=0
호스트 빌드                                                       EXIT=0
android arm64  ELF 64-bit LSB shared object, ARM aarch64          EXIT=0
ios arm64      Mach-O 64-bit dynamically linked shared library    EXIT=0
```

**골든 op 수는 96 으로 그대로입니다.** 새 aten op 을 추가하지 않았기 때문이고, 그 이유는
§3.3 입니다 — 이번에 채운 스펠링들은 디스패처에 닿지 않습니다.

**스키마는 204/204 로 그대로입니다.** `overloads.json`/`methods.json` 에 항목을 넣지 않았습니다.
`_parse_to` 는 스키마가 아예 없는 손수 쓴 파서이므로 테이블에 들어갈 것이 아니고
(`docs/bindings/OVERLOAD.md` §9 항목 7), `is_floating_point`/`cpu`/`get_device` 는 §3.3 의 이유로
테이블 항목이 아닙니다.

새로 붙인 테스트 (`rust/torch_c/pytests/test_shim.py`):

| 테스트 | 무엇을 고정하나 |
|---|---|
| `test_device_label_is_validated_against_a_closed_vocabulary` | 상류가 거부하는 6 가지를 같은 예외로 거부 |
| `test_device_accepts_every_spelling_torch_normalises_through` | 복사 생성자 · `type=`/`device=` · 중복 인덱스 거부 |
| `test_device_is_picklable` | `__reduce__` 모양. 왕복은 §아래 |
| `test_indexed_and_bare_labels_are_unequal_but_name_one_device` | `cpu` ≠ `cpu:0`, 그러나 `device="cpu:0"` 로 만든 텐서는 `cpu` |
| `test_tensor_reports_its_device_through_every_spelling` | `is_cpu`·`is_cuda`·`is_meta`·`get_device`·`cpu()`·`is_floating_point` |
| `test_to_copy_with_no_device_keeps_the_tensor_where_it_is` | §5.2 의 계약 |
| `test_unavailable_device_fails_where_torch_fails_it` | 라벨은 만들어지고 사용에서 죽는다 |
| `test_data_setter_replaces_the_tensor_behind_a_parameter` | §4.1 |
| `test_parse_to_matches_the_real_parser_not_the_dynamo_polyfill` | 11 가지 호출 모양 + `copy` 거부 3 가지 |
| `test_shallow_copy_compatibility_answers_for_dense_tensors_only` | 텐서 아닌 것은 추측하지 않고 거부 |
| `test_the_two_accelerator_questions_get_two_different_answers` | §5.5 |
| `test_mixed_device_gate_lets_agreeing_tensors_through` | §6 의 통과하는 절반 + 라벨 규칙 |
| `test_generator_reports_a_device` | §5.5 |
| `test_device_road_through_the_vendored_tree` | **위를 전부 이어 붙인 것** — 서브프로세스에서 실제 `nn.Sequential` 을 `.to()`/`.cpu()`/`.float()`/`.double()` 하고, 파라미터 객체 동일성 보존과 순전파까지 확인. 피클 왕복도 여기(이름이 `torch._C` 로 해석되는 유일한 자리) |

마지막 것이 필요한 이유는 §2.3 입니다 — 벽이 넷 연달아 있어서, 각 조각을 따로 테스트하면 사슬이
이어졌는지는 증명되지 않습니다.

**`test_device_road_through_the_vendored_tree` 는 `nn.ReLU` 를 쓰지 않습니다.** `torch.relu` 에
오버로드 테이블 항목이 없어서 `nn.ReLU` 순전파가 별도로 깨져 있습니다(이번 작업 이전부터,
`docs/devices/DEVICE.md` §5 의 "host failures: ['nn.ReLU forward']" 와 같은 것). 이 테스트는 장치 도로에
관한 것이므로 그 구멍을 피해서 `nn.Linear` 두 개로 썼고, 여기 적어 둡니다.

---

## 9. 실측 요약 — 82 개 스펠링, 작업 전후

```
상류와 일치   41  →  56
불일치        41  →  26
```

남은 26 개의 성격:

| 갈래 | 개수 | 항목 |
|---|---:|---|
| **이 호스트에 MPS 가 있고 우리 빌드에 가속기가 없다** | 6 | `ctor.int`, `_get_accelerator`, `avail.mps`, `avail.mps.built`, `avail.accel`, `avail.accel.count` — 결함이 아니라 빌드 차이 |
| ~~**컨텍스트 매니저 / 기본 장치 미구현** (§7.2)~~ **정정 (문서 감사, 2026-09): §7.2 자신이 이미 "구현됨" 이라고 적은 것과 어긋남** | 6 | `attr.enter`, `ctx.with_cpu`, `ctx.with_meta`, `default.set_cpu`, `default.set_meta`, `m.meta_ctx` — §7.2 정정 참고, `docs/devices/META.md` §8 |
| **CUDA 부재를 다른 예외로 말한다** | 5 | `t.to.cuda`, `t.cuda`, `t.zeros_cuda`, `m.to_cuda`, `mixed.cpu_plus_cuda`. 상류 `AssertionError: Torch not compiled with CUDA enabled` vs 우리 `NotImplementedError: device not available in torch._C shim: cuda` — 둘 다 "없다" 이고, 우리 쪽 메시지는 어느 층에서 없는지를 말합니다 |
| ~~**`meta` 미구현** (§7.1)~~ **정정 (문서 감사, 2026-09): §7.1 자신이 이미 "구현됨" 이라고 적은 것과 어긋남** | 3 | `t.to.meta`, `t.zeros_meta`, `mixed.cpu_plus_meta` — §7.1 정정 참고, `docs/devices/META.md`. 실측 재확인: `torch.zeros(2, device='meta')` 와 `with torch.device('meta'): torch.zeros(3)` 둘 다 오늘 성공 |
| **`privateuse1` 이름 변경 미구현** (§7.3) | 3 | `reg.rename`, `reg.after_rename`, `reg.device_pu1` |
| **장치와 무관한 커널 구멍** | 2 | `t.ones_like_dev` (오버로드 테이블에 `ones_like` 없음), `t.pin_memory` |
| **예외 메시지 문면만 다름** | 1 | `ctor.none` — 타입과 앞부분은 같고 상류가 오버로드 목록을 덧붙입니다 |

---

## 10. 모르는 것

- **가속기가 없는 상류 빌드에서 `torch.device(0)` 이 무엇을 하는지 모릅니다.** 이 기계에 MPS 가
  있어서 그 분기를 낼 수 없었습니다. 우리는 이유를 말하는 `RuntimeError` 를 냅니다.
- **`_get_accelerator` 와 `_accelerator_getAccelerator` 의 "가속기 없음" 답은 실측이 아니라
  벤더 소스를 읽은 것입니다** (§5.5). 호출자의 `None` 처리와 docstring 이 근거이지, 돌려본 것이
  아닙니다.
- **`_has_compatible_shallow_copy_type` 이 상류에서 `False` 가 되는 조건을 완전히는 모릅니다.**
  dtype·장치·`Parameter` 차이에 전부 `True` 인 것은 쟀고, C++ 은 `DispatchKeySet` 을 보므로
  `FakeTensor` 같은 서브클래스가 `False` 일 것으로 읽었지만 만들어서 확인하지 않았습니다.
- ~~**문의 거부 절반은 한 번도 실행된 적이 없습니다** (§6). 발화시킬 입력을 만들 수 없습니다.~~
  **닫힘 (2026-08-25):** `meta` 로 입력을 만들 수 있게 되어 처음 발화시켰더니, 키워드 인자
  안의 시퀀스를 훑지 않는 버그가 나왔습니다 — 그리고 토치 레벨 호출은 인자를 전부 이름으로
  넘기므로 `torch.cat([cpu, meta])` 가 게이트를 통째로 지나가고 있었습니다. `docs/devices/META.md` §5.
- ~~**`meta` 를 구현하면 `aten.rs` 의 몇 개 커널이 영향받는지 모릅니다.**~~ **답: 0 개.**
  meta 는 커널 안의 분기가 아니라 별도의 디스패치 표로 들어갔고(상류의 `Meta` 키와 같은
  모양), dense 커널은 meta 를 모릅니다 — `tensor()` 가 타입 수준에서 거부합니다.
  바뀐 것은 팩토리 9 개의 `device=` 처리뿐입니다. `docs/devices/META.md` §3~§4.
- **다중 장치에서 `same_physical_device` 의 인덱스 없는 라벨 처리가 맞는지 모릅니다.** "인덱스
  없음 = 그 종류의 현재 장치" 라는 상류 규약을 따랐지만, 종류당 장치가 하나뿐인 상태에서만
  확인했습니다.
- **`accelerate` 가 켜진 상태에서 두 번째 장치를 켰을 때 무슨 일이 나는지 모릅니다.** 둘 다
  Apple 타깃의 같은 Cargo 절에 들어가고, candle 이 그 조합을 어떻게 다루는지는 안 봤습니다.
- **`docs/perf/PERF.md` 의 2 층 블록 수치를 이번 변경 후 다시 재지 않았습니다.** §6 의 마이크로벤치가
  모델 수준으로 어떻게 번역되는지는 계산이지 측정이 아닙니다.

---

## 11. 재현

```bash
cd /path/to/repo
bash vendor/vendor_torch.sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-devabs
bash vendor/install_shim.sh

PY=/Volumes/macMini/caches/spike-venv/bin/python
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib

$PY tools/golden/compare.py                     > /tmp/g.log 2>&1;  echo "EXIT=$?"
$PY tools/golden/compare.py --inject-fault value > /tmp/fv.log 2>&1; echo "EXIT=$?"
$PY rust/torch_c/pytests/verify_schemas.py      > /tmp/s.log 2>&1;  echo "EXIT=$?"
PYTHON=$PY sh rust/torch_c/pytests/run.sh       > /tmp/p.log 2>&1;  echo "EXIT=$?"

# 표면 대조: 같은 프로브를 두 torch 로 돌리고 전사를 diff
PYTHONDONTWRITEBYTECODE=1 TORCH_USE_RTLD_GLOBAL=1 \
  PYTHONPATH=$PWD/torchnative/src/main $PY <probe> > ours.txt
(cd /tmp && $PY <probe> > upstream.txt)
```

`<probe>` 는 §1 의 세 스크립트입니다. 저장소 밖(`/tmp`)에 두었습니다 — `docs/perf/PERF.md` 의
벤치 스크립트와 같은 규율입니다.
