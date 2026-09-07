# `meta` 장치 — 저장소 없는 텐서를 실제로 만들고, 그것이 열어준 검증을 열었다

`docs/DEVICE_ABS.md` §7.1 이 `meta` 를 **다음으로 가장 값진 항목**으로 지목했습니다. 이유는
`meta` 자체가 아니라 그것이 여는 것이었습니다 — **백엔드가 필요 없는 두 번째 장치**라서, 그 문서가
"장치가 둘이 되면" 이라고 논증만 해둔 것들이 **시험 가능**해집니다.

> **결론 먼저.** 저장소 없는 텐서를 만들었습니다. 할당해놓고 `meta` 라고 부르지 않았습니다 —
> `PyTensorBase` 가 `Repr::Dense(Tensor) | Repr::Meta { shape }` 두 표현을 갖고, `tensor()` 가
> `&Tensor` 대신 `PyResult<&Tensor>` 를 돌려주면서 **커널이 meta 의 바이트를 읽는 것이
> 타입 수준에서 불가능**해졌습니다.
>
> **그리고 그것이 실제로 버그를 하나 찾았습니다.** DEVICE_ABS §10 이 "한 번도 실행된 적 없다"
> 고 적은 혼합 장치 게이트의 거부 갈래를 처음 발화시켰더니, **키워드 인자 안의 리스트를 훑지
> 않고 있었습니다.** 그런데 토치 레벨 호출은 인자를 전부 이름으로 바인딩해서 넘기므로,
> `torch.cat([cpu, meta])` 가 게이트를 통째로 지나쳤습니다 (§5).
>
> **`with torch.device(...)` 와 `set_default_device` 도 함께 넣었습니다.** DEVICE_ABS §7.2 가
> 경고한 "스택만 만들면 `torch.zeros(2)` 가 조용히 무시한다" 를 피하려면 둘 다 필요했고,
> 둘 다 넣었습니다 (§8).
>
> **디스패치 비용이 늘었습니다: `add.Tensor` +13.6 ns (+3.7%).** 재서 §9 에 적었습니다.

측정일 2026-08-25, `develop`(`016d65b`) 기준. 호스트 `darwin/arm64`, CPython 3.13.0,
상류 torch 2.13.0 (`/Volumes/macMini/caches/spike-venv`).

---

## 1. 무엇을 어떻게 쟀나

DEVICE_ABS §1 과 같은 방식입니다 — **설계부터 하지 않았습니다.** 같은 프로브를 두 torch 로
돌리고 전사를 줄 단위로 비교합니다. 각 줄은 `이름 | 값` 또는 `이름 | <예외타입: 메시지>` 이고,
**아무것도 단언하지 않습니다.** 전사가 곧 측정값입니다.

프로브는 107 항목이고, 다섯 갈래를 묻습니다.

| 갈래 | 무엇을 묻는가 |
|---|---|
| 생성 | `torch.device("meta")`, 9 개 팩토리, 인덱스가 붙은 라벨 |
| 메타데이터 | `.shape` · `.dtype` · `.stride()` · `.numel()` · `.data_ptr()` · `.untyped_storage()` |
| 데이터 접근 | `.tolist()` · `.item()` · `.numpy()` · `float()` · `bool()` — **어떻게 거부하는가** |
| 연산 전파 | 산술 · 뷰 · 브로드캐스트 · dtype 승격 · 모양 오류 · 혼합 장치 |
| 컨텍스트 | `with torch.device(...)` · `set_default_device` · 모드 스택 · `nn.Module` 경로 |

프로브 스크립트는 저장소 밖(`/tmp/meta_probe/`)입니다 — `docs/PERF.md` 의 벤치 스크립트와 같은
규율입니다.

---

## 2. 실측 — 상류의 `meta` 는 무엇인가

**이것이 먼저였습니다.** 구현 방향을 정한 것은 아래 표이지 `meta` 라는 이름에 대한 짐작이
아닙니다.

### 2.1 메타데이터는 전부 있고 바이트만 없다

```
torch.zeros(2,3, device="meta")     tensor(..., device='meta', size=(2, 3))
  .shape                            torch.Size([2, 3])
  .dtype                            torch.float32
  .device                           device(type='meta')
  .stride()                         (3, 1)
  .numel()                          6
  .element_size()                   4
  .is_meta / .is_cpu                True / False
  .data_ptr()                       0                    <-- 이것이 정의다
  .untyped_storage().nbytes()       24                   <-- 크기는 알고 바이트는 없다
  .untyped_storage().data_ptr()     0
```

`data_ptr()` 이 `0` 이라는 것과 `nbytes()` 가 `24` 라는 것이 함께 성립합니다. **meta 는
"할당하지 않은 채로 할당했을 때의 모양을 아는 것"** 이지 "빈 텐서" 가 아닙니다.

### 2.2 데이터를 읽으려 하면 — 두 가지 다른 예외로 거부한다

| 호출 | 상류 |
|---|---|
| `.tolist()` | `NotImplementedError: Cannot copy out of meta tensor; no data!` |
| `.item()` / `float(t)` / `bool(t)` | `RuntimeError: Tensor.item() cannot be called on meta tensors` |
| `.numpy()` | `TypeError: can't convert meta device type tensor to numpy. ...` |
| `.to("cpu")` / `.cpu()` | `NotImplementedError: Cannot copy out of meta tensor; no data!` |
| `torch.zeros(2).copy_(meta)` | 〃 |
| `torch.allclose(meta, meta)` | `RuntimeError: Tensor.item() cannot be called on meta tensors` |

**세 가지가 다른 예외 타입입니다.** 하나로 합치고 싶어지는 모양이지만 합치지 않았습니다 — 상류를
재서 그대로 옮겼고, 셰임도 그대로 세 갈래입니다(§4).

### 2.3 전송은 한 방향뿐이다

```
torch.zeros(2).to("meta")           tensor(..., device='meta', size=(2,))     통과
meta.to("cpu")                      Cannot copy out of meta tensor; no data!  거부
meta.to("meta") is meta             True
meta.to(torch.float64)              tensor(..., device='meta', dtype=float64) 통과
meta.copy_(cpu)                     통과 (무동작, 수신자는 meta 로 남는다)
cpu.copy_(meta)                     거부
```

`meta.copy_(cpu)` 가 통과하고 **아무것도 하지 않는** 것이 중요합니다. `nn.Module.load_state_dict`
를 `assign=True` 없이 부르면 정확히 여기로 오고, 상류가 그 자리에서 경고합니다 —
*"copying from a non-meta parameter in the checkpoint to a meta parameter in the current model,
which is a no-op"*.

### 2.4 인덱스는 사라진다

```
torch.zeros(2, device="meta:0").device      device(type='meta')
torch.zeros(2, device="meta:1").device      device(type='meta')
torch.zeros(2, device="meta:7").device      device(type='meta')
torch.zeros(2, device="cpu:3").device       device(type='cpu')     <-- 같은 규칙
```

**meta 장치는 하나뿐이고 상류가 인덱스를 정규화해 버립니다.** 이것이 §3 의 표현 결정 하나를
그대로 정했습니다.

### 2.5 연산은 모양만 계산해서 전파한다

`add` · `mm` · `view` · `reshape` · `t` · `slice` · `cat` · `softmax` · dtype 승격 · in-place —
전부 통과하고 전부 meta 를 돌려줍니다. 모양 오류는 **여전히 오류**입니다
(`mm([2,3], [5,4])` → `RuntimeError: a and b must have same reduction dim`). 상류는 이 각각에
대해 **모양만 계산하는 커널**을 따로 등록해 두고 있습니다(`torch/_meta_registrations.py`).

### 2.6 혼합 장치 — 그리고 상류가 여기서도 한 번 새어나간다

| 호출 | 상류 |
|---|---|
| `cpu + meta` | `RuntimeError: Tensor on device meta is not on the expected device cpu!` |
| `meta + cpu` | `RuntimeError: Tensor on device cpu is not on the expected device meta!` |
| `torch.cat([cpu, meta])` | 위 첫 번째 메시지 |
| `torch.where(cpu_cond, cpu, meta)` | 〃 |
| `torch.add(cpu, cpu, out=meta)` | `RuntimeError: Attempting to copy from device cpu to device meta, ...` |
| `cpu + tensor(1.0, device="meta")` (0-차원) | 거부 |
| **`torch.mm(cpu, meta)`** | **통과.** `tensor([[0., 0., 0., 0.], [0., 0., 0., 0.]])` 를 돌려준다 |

**마지막 줄이 상류의 구멍입니다.** `torch.mm` 이 CPU 텐서와 meta 텐서를 받아 **CPU 텐서를
계산해 돌려줍니다.** DEVICE_ABS §2.4 가 `torch.cat([cpu, mps])` 가 세그폴트하는 것을 찾았던
것과 **같은 종류의 발견**이고, 원인도 같습니다 — 상류의 장치 검사는 커널 안에 있고, 커널마다
따로 기억해야 하며, `mm` 은 meta 에 대해 잊었습니다.

이 셰임은 문이 하나라서 잊을 곳이 없습니다. 같은 호출을 거부합니다(§5).

---

## 3. 구조 판단 — 저장소 없는 텐서를 어떻게 만들었나

과제가 미리 경고한 지점입니다: **candle 에는 저장소 없는 텐서가 없습니다.** 모든
`candle_core::Tensor` 가 저장소를 소유하고 `Tensor::zeros` 는 할당합니다.

DEVICE_ABS §7.1 이 두 길을 적었습니다 — (a) 진짜로 할당하고 `meta` 라벨을 붙인다,
(b) `PyTensorBase` 에 저장소 없는 표현을 만든다.

**(b) 를 했습니다.** (a) 는 하지 않았습니다.

### 3.1 표현

```rust
pub enum Repr {
    Dense(Tensor),
    Meta { shape: Vec<usize> },
}

pub struct PyTensorBase {
    inner: Repr,
    tag: TorchDType,     // 그대로
    ...
}
```

**저장하지 않은 것이 두 가지이고 둘 다 근거가 있습니다.**

*스트라이드를 저장하지 않습니다.* 이 셰임의 `TensorBase` 에는 `.stride()` 가 **아예 없습니다** —
dense 쪽도 보고하지 않습니다. meta 만 스트라이드를 들면 meta 가 dense 에 없는 표면을 갖게
됩니다. 상류의 meta 는 스트라이드를 갖고
(`torch.zeros(2,3,device="meta").t().stride()` 는 `(1, 3)`), 그래서 이것은 **좁힌 것**이고 §6 에
적었습니다. `t`/`permute` 의 meta 커널이 오는 날 그 커널이 이 필드를 추가해야 합니다.

*장치 라벨을 저장하지 않습니다.* §2.4 의 실측 때문입니다 — meta 장치는 하나뿐이고 인덱스가
정규화되어 사라집니다. 그러므로 라벨은 상수입니다. 종류당 장치가 둘 이상 주소 지정 가능해지는
날 이 필드가 생겨야 하고, 그때의 논증은 `DEVICE_ABS.md` §3.2 에 이미 있습니다.

### 3.2 대가를 한 곳에서 치른다 — `tensor()` 가 `PyResult` 를 돌려준다

```rust
pub fn tensor(&self) -> PyResult<&Tensor> {
    match &self.inner {
        Repr::Dense(tensor) => Ok(tensor),
        Repr::Meta { .. } => Err(no_data()),   // 상류의 그 메시지
    }
}
```

`aten.rs` 의 **241 곳**이 `.tensor()` → `.tensor()?` 로 바뀌었습니다. 기계적인 변경이고
컴파일러가 전부 검증했습니다 (7 곳만 손으로 고쳤습니다 — `and_then` 클로저 안에서 `?` 를 쓸 수
없어 호이스팅한 자리들).

**이 대가를 치른 이유는 방어가 두 겹이 되기 때문입니다.** 문의 meta 게이트(§4)가 먼저 거부하고
더 나은 메시지를 냅니다. 그 아래에서 타입이 거부합니다. **내일 meta 를 모르고 추가되는 커널이
안전한 이유가 이것입니다** — 96 개 커널이 각자 기억해야 하는 규칙이 아니라 타입의 성질입니다.
`check_devices_agree` 가 문에 있는 것과 같은 논증입니다.

### 3.3 meta 는 백엔드가 아니라 백엔드의 부재다

`PyDevice::resolve()` 는 여전히 `cpu` 하나만 핸들로 바꿉니다. `meta` 는 **다른 열아홉 개와 다른
메시지로** 거부합니다:

```
"the meta device has no backend to resolve to -- a meta tensor holds shape and dtype
 and no storage, so this call site has to branch on PyDevice::is_meta() before resolving"
```

`cuda` 는 이 빌드가 링크하지 않은 백엔드이고, `meta` 는 **정의상 백엔드가 없는** 장치입니다.
같은 메시지로 답하면 그 차이가 지워집니다.

그래서 팩토리는 `device_arg_or_label()` 로 **라벨**을 받고, `is_meta()` 를 먼저 물은 뒤에
`resolve()` 합니다. `device_arg`/`device_arg_or` (핸들을 돌려주던 것들)는 호출자가 하나도
남지 않아 삭제했습니다 — 모든 팩토리가 라벨 경로로 갔다는 뜻입니다.

---

## 4. meta 커널은 별도의 표다 — 상류가 그런 것처럼

```rust
match check_devices_agree(op, args, kwargs)? {
    Some(Where::Meta) => meta_dispatch(py, op, args, kwargs)?,
    _                 => aten_dispatch_inner(py, op, args, kwargs)?,
}
```

**커널 안의 분기가 아니라 별도의 표인 이유는 상류가 그렇기 때문입니다.** torch 에는 `Meta`
디스패치 키가 있고 등록이 따로 있습니다(`torch/_meta_registrations.py`). 이 모양을 따르면
dense 커널은 meta 를 몰라도 되고("모르면 `tensor()` 가 거부한다"), "이 op 이 meta 에서 되나?"
의 답이 96 개를 읽는 것이 아니라 **목록 하나**가 됩니다.

### 4.1 구현한 meta 커널

| op | 무엇을 하나 |
|---|---|
| `_to_copy.default` | dtype 변경은 통과, `device=None` 은 제자리, cpu 로 나가는 것은 거부 |
| `copy_.default` | `meta ← cpu` 는 무동작(수신자 유지), `cpu ← meta` 는 거부 |
| `detach` · `alias` · `clone` · `contiguous` · `lift_fresh` | 모양·dtype 그대로 통과 |
| `is_floating_point.default` | dtype 태그가 이미 답을 들고 있다 |
| `new_ones.default` | 모양은 인자, 장치는 입력 — meta 입력이면 meta 팩토리 |
| `uniform_` · `normal_` · `zero_` · `fill_.Scalar` | **무동작, 수신자를 돌려준다** |
| `_local_scalar_dense.default` | `Tensor.item() cannot be called on meta tensors` |

**이 표는 첫 회차의 것입니다.** 그 뒤로 `empty_like`(docs/CKPT2.md §3 이 멈춰 있던 자리),
`div.Scalar` · `mul.Scalar` · `pow.Scalar` · `reciprocal`(rope 초기화)이 더해졌고, 이번
회차가 **원소별 계열 전체와 모양 커널 셋**을 더했습니다. 현재 목록은 §7.1 과 §7.2 입니다.

**in-place 초기화 넷이 편의가 아닙니다.** `nn.Linear.__init__` 은 매번
`init.kaiming_uniform_(self.weight)` 로 끝나므로, 이것들 없이는
`with torch.device("meta"): nn.Linear(4, 8)` 이 파라미터를 하나도 만들지 못하고 멈춥니다 —
`accelerate.init_empty_weights` 가 통째로 그 호출입니다. 바이트가 없는 텐서에 쓰는 것은 관측
가능한 효과가 없고, 광고하는 모양도 바뀌지 않으므로 "아무것도 쓰지 않고 self 를 돌려준다" 가
상류의 meta 커널이기도 합니다. **읽어내는 쪽의 거부(`tolist`/`item`)는 그대로 살아 있으므로**,
초기화되지 않은 meta 파라미터를 초기화된 것으로 착각할 길은 여전히 없습니다.

`add_`/`mul_` 은 **일부러 넣지 않았습니다.** 브로드캐스트하므로 맞춰야 할 모양 규칙이 있고,
무동작으로 두면 상류가 거부하는 모양을 조용히 받게 됩니다.

### 4.2 meta 팩토리

`empty.memory_format` · `zeros.default` · `ones.default` · `full.default` ·
`scalar_tensor.default` · `arange.{default,start,start_step}` · `randint.{default,low}` ·
`new_ones.default` · `torch.tensor(...)`.

**검사는 meta 에서도 전부 합니다.** `full` 의 `checked_convert`(dtype 이 담을 수 없는 값),
`randint` 의 `high <= low`, `arange` 의 `step == 0` 과 부호 불일치, `torch.tensor` 의 ragged 검사 —
전부 meta 반환 **앞**에서 실행됩니다. meta 텐서는 *진짜 호출이 무엇을 만들었을지에 대한
주장*이고, 진짜 호출이 하는 검사를 건너뛴 주장은 주장이 아니기 때문입니다.

**딱 하나 일부러 건너뜁니다:** `arange` 의 `arange_has_cpu_kernel`. 그것은 상류의 *CPU 커널*
구멍을 재현하는 것인데 상류의 meta 커널에는 그 구멍이 없습니다 —
`torch.arange(5, dtype=torch.uint16, device="meta")` 는 텐서이고 CPU 철자는 예외입니다. 실측.

`arange` 의 원소 수 계산은 `arange_length()` 로 뽑아 dense 와 meta 가 **공유**합니다. 정수
분기가 부동소수 `ceil` 이 아니라 정수 올림나눗셈이라 두 벌을 두면 경계에서 갈라집니다.

### 4.3 그리고 나머지는 자기 이름을 댄다

```
NotImplementedError: torch._C shim has no meta kernel for aten.add.Tensor. A meta tensor
holds shape and dtype and no storage, so this op would have to infer its output shape
without computing -- which is a real kernel (upstream registers one in
torch/_meta_registrations.py), not a fallthrough. See docs/META.md §7.
```

**나머지 90 여 개의 모양 추론은 넣지 않았습니다.** 근거는 §7 에 적었습니다.

---

## 5. 열린 검증 (1) — 게이트의 거부 갈래가 실행됐고, **버그를 찾았다**

DEVICE_ABS §10 의 문장은 이것이었습니다:

> **문의 거부 절반은 한 번도 실행된 적이 없습니다** (§6). 발화시킬 입력을 만들 수 없습니다.

이제 만들 수 있습니다. 만들어서 돌렸더니 **게이트가 발화하지 않았습니다.**

```
torch.cat([torch.zeros(2), torch.zeros(2, device="meta")])
  기대:  RuntimeError: ... at least two devices, cpu and meta!
  실제:  NotImplementedError: Cannot copy out of meta tensor; no data!
```

거부는 맞았지만 **틀린 곳에서** 나왔습니다 — 게이트를 지나쳐 커널 안으로 들어가서
`tensor()?` 에 걸린 것입니다(§3.2 의 두 번째 방어선이 실제로 일한 순간이기도 합니다).

**원인:** `check_devices_agree` 의 키워드 루프가 시퀀스로 내려가지 않았습니다.

```rust
// 이전
for value in args.iter() { /* 텐서, 그리고 리스트/튜플 안까지 */ }
if let Some(kwargs) = kwargs {
    for (_, value) in kwargs.iter() { visit(&value)?; }   // 시퀀스로 안 내려간다
}
```

그런데 **`_torch_level_function` 은 인자를 전부 이름으로 바인딩해서 `dispatch(key, **bound)`
로 넘깁니다.** 즉 `torch.cat([a, b])` 는 `kwargs["tensors"] = [a, b]` 로 도착하고
**위치 인자 루프를 아예 건드리지 않습니다.** 두 루프의 차이는 장치가 하나인 동안 관측
불가능했고, 그래서 DEVICE_ABS §6 이 "`Tensor[]` 인자를 통해서도" 테스트했다고 적은 것은
**위치 인자 경로에 대해서만** 참이었습니다.

지금은 두 루프가 같은 `scan_for_device` 를 씁니다. 그리고 §2.6 의 표와 대조하면:

| 호출 | 상류 | 이 셰임 (지금) |
|---|---|---|
| `cpu + meta` | 거부 | 거부 |
| `torch.cat([cpu, meta])` | 거부 | 거부 |
| `torch.where(cpu, cpu, meta)` | 거부 | 거부 |
| `torch.add(cpu, cpu, out=meta)` | 거부 | 거부 |
| `cpu + tensor(1.0, device="meta")` | 거부 | 거부 |
| **`torch.mm(cpu, meta)`** | **통과 (CPU 결과를 계산한다)** | **거부** |

**마지막 줄에서 이 셰임이 상류보다 엄격하고, 그것이 이 설계의 값입니다.** 문이 하나라서
`mm` 이 잊을 수 없습니다. DEVICE_ABS §6 이 `cat`+MPS 세그폴트를 근거로 예측한 바로 그 일이
다른 op, 다른 장치에서 다시 관측됐습니다.

### 5.1 메시지는 상류와 다릅니다 — 의도한 발산

상류는 meta 에 대해 `Tensor on device meta is not on the expected device cpu!` 라고 말하고,
이 셰임은 `Expected all tensors to be on the same device, but found at least two devices,
cpu and meta! (aten.add.Tensor in torch._C shim)` 라고 말합니다.

문이 하나라는 것은 메시지도 하나라는 뜻입니다. 상류는 커널마다 다른 메시지를 내고(§2.6 의
표에 세 종류가 있습니다), 그중 가장 흔한 것을 DEVICE_ABS 가 이미 골라 두었습니다. op 이름이
괄호로 붙으므로 어느 op 이었는지는 잃지 않습니다. **7 개 항목이 이 이유로 전사에서 불일치로
집계됩니다** (§10).

---

## 6. 열린 검증 (2) — `PyDevice::from_candle` 의 인덱스 하드코딩

DEVICE_ABS §3.2 가 남긴 질문: `from_candle` 이 `Cuda`/`Metal` 에 대해 인덱스 0 을 박아넣고
있고, 그것이 언제 틀리는가.

**`meta` 는 이 질문에 답하지 못합니다. 다만 질문의 모양을 바꿉니다.**

- meta 텐서에는 candle 핸들이 **없으므로** `from_candle` 을 지나가지 않습니다. 그래서
  `PyTensorBase::device_label()` 이 두 갈래로 갈렸습니다 — dense 는 `from_candle`(손실 있는
  재구성), meta 는 상수. **즉 텐서가 자기 라벨을 드는 쪽으로 가는 첫 발이 이미 놓였습니다.**
- 그런데 §2.4 의 실측이 meta 에 대해서는 **인덱스가 없는 것이 정답**이라고 말합니다. 상류가
  `meta:7` 을 `meta` 로 정규화하기 때문입니다. 그래서 meta 는 하드코딩이 **틀리는** 사례가
  아니라, 하드코딩이 **필요 없는** 두 번째 사례입니다 — `cpu:3` → `cpu` 와 같은 갈래.

**여전히 모르는 것:** 인덱스가 살아남는 장치 종류에서 무슨 일이 나는지. `cuda:1` 을 왕복시키면
`cuda:0` 이 된다는 DEVICE_ABS 의 관찰은 그대로 미해결입니다. **meta 로는 그것을 시험할 수
없습니다** — 인덱스를 보존하는 장치가 아직 하나도 없기 때문입니다. `metal` feature 를 켜는 것이
그 질문에 답하는 유일한 길이고, 이번 작업은 그것을 하지 않았습니다.

---

## 7. 원소별 계열과 모양 커널 셋 — 그리고 아직 남은 것

**§7 의 이전 판은 "나머지 op 의 모양 추론은 하지 않았다" 였습니다.** 그 판단을 뒤집은 것은
논증이 아니라 **사용자 보고**입니다. 공개된 0.0.5a0 휠에서:

```
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
  ... transformers/modeling_rope_utils.py:655 _compute_llama3_parameters
  inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
NotImplementedError: torch._C shim has no meta kernel for aten.gt.Scalar.
```

**윈도우 전용이 아닙니다.** macOS 에서 재현했습니다 — 이 rope 초기화는 `cpu` 에서 성공하고
`meta` 에서 실패합니다. `from_pretrained` 가 가중치를 meta 에서 초기화하므로 rope 계산이
거기서 돕니다. SmolLM2 가 되던 것은 `rope_scaling` 이 없어서 기본 rope 초기화가 **비교를
하지 않기** 때문이었습니다.

그리고 **README 첫 코드 블록이 로드하는 모델이 바로 그것입니다.** 즉 §7 의 이전 판이
"측정된 수요가 없다" 고 적은 그 자리가, 이 프로젝트의 대표 예시였습니다. 이전 판의 세 근거 중
1번(짐작하지 말 것)과 3번(거부가 작업 큐다)은 그대로 유효하고 **이번 회차가 그 둘을 따랐습니다** —
2번만 반증됐습니다.

### 7.1 원소별 계열 — 모양 규칙과 dtype 규칙, 둘 다 빌려온 것

| 계열 | op | 모양 | dtype |
|---|---|---|---|
| 비교 | `eq`·`ne`·`lt`·`le`·`ge`·`gt` × `{Scalar, Tensor}` (12) | 입력 / 브로드캐스트 | **무조건 `bool`** |
| 산술 | `add`·`sub`·`mul`·`div` × `{Tensor, Scalar}` (8), `rsub.Scalar` | 브로드캐스트 / 입력 | `arith_tag` |
| 선택 | `where.self` · `where.ScalarOther` | **3-피연산자 브로드캐스트** | 값 쪽에서, 조건이 아니라 |
| 단항 승격 | `cos`·`sin`·`tanh`·`exp`·`log`·`expm1`·`rsqrt`·`reciprocal` | 그대로 | `unary_float_tag` |
| 단항 보존 | `neg` · `bitwise_not` | 그대로 | 입력 그대로, 거부 포함 |
| 사다리 | `clamp.default` | 그대로 | `clamp_result_tag` |
| 거듭제곱 | `pow.Tensor_Scalar` · `pow.Tensor_Tensor` | 입력 / 브로드캐스트 | `pow_result_tag` |

**모양 절반은 기계적이고 dtype 절반은 아닙니다.** 그래서 dtype 규칙을 **한 줄도 다시 쓰지
않았습니다** — dense 커널이 쓰는 바로 그 함수를 부릅니다. 근거는 docs/E2E_REAL.md §6.1 입니다:
dense 가 만들지 않을 dtype 을 meta 가 약속하면, 호출자는 그 dtype 으로 할당해 두고 dense 가
그 자리에서 거부합니다. 같은 함수를 부르면 **구성적으로** 어긋날 수 없고, 거부까지 같아집니다.

그 공유를 위해 dense 커널 안에 인라인으로 있던 규칙 셋을 함수로 꺼냈습니다. dense 의 동작은
바뀌지 않았고(골든 4284/4284 그대로), 각 함수는 이제 두 경로가 함께 씁니다:

| 꺼낸 함수 | 원래 있던 곳 | meta 가 쓰는 이유 |
|---|---|---|
| `unary_float_tag` | `unary_float` · `rsqrt_default` 본문 | "부동은 그대로, 나머지는 기본 float" |
| `neg_result_tag` | `neg_default` 본문 | dtype 보존 **과 두 거부**(`bool`, 넓은 unsigned) |
| `clamp_result_tag` | `clamp_default` 본문 | 골든이 한 번 고쳐준 사다리를 두 벌 두지 않기 위해 |
| `where_condition_check` | `where_self` · `where_scalar_other` 본문 | 조건 dtype 거부를 세 곳이 같은 문면으로 |
| `expand_target` | `expand_default` 본문 | 랭크 검사와 `-1` 해석 (§7.2) |

**dtype 이 짐작 불가능한 자리 셋**이 이 계열의 전부라고 해도 됩니다. 상류 meta 텐서로 직접
측정한 것입니다:

```
gt(float32_meta, 1.0)            torch.bool        입력이 무엇이든
div(int64_meta, int64_meta)      torch.float32     참 나눗셈이 정수 쌍을 띄운다
where(bool_meta, f32, f32)       torch.float32     조건이 아니라 값에서
mul(f16_meta, bf16_meta)         torch.float32     같은 랭크끼리는 위로 탈출
pow(int64_meta, 2)               torch.int64       정수 지수는 정수를 유지
pow(int64_meta, 2.0)             torch.float32     실수 지수는 띄운다
clamp(int32_meta, None, 2.0)     torch.float32     제자리 형제는 여기서 거부한다
neg(int64_meta)                  torch.int64       단항이라고 다 승격하지 않는다
```

`where.self` 의 모양은 **세 피연산자의 조인**이지 조건의 것이 아닙니다:
`where(bool(2,1), f32(1,3), f32(3))` 은 `(2,3)` 이고, 조건 모양을 답하면 `(2,1)` 입니다.

### 7.2 모양 커널 셋 — 그리고 그것을 고른 것은 작업 큐다

원소별 계열만으로 `llama` 는 통과했습니다. 그 다음을 **미리 정하지 않고** 스무 아키텍처를
`with torch.device("meta")` 아래에서 구성해 큐를 인쇄시켰습니다. ARCH20.md §0.2 의
*"벽 하나가 벽 하나가 아니다"* 를 따라 **매번 다시 재고** 다음 하나를 넣었습니다.

| 회차 | 통과 | 큐가 인쇄한 것 |
|---|---|---|
| 원소별 계열까지 | **14/20** | `aten.select.int` ×5, `aten.tril.default` ×1 |
| `select.int` · `tril`/`triu` 후 | **19/20** | `aten.expand.default` ×1 (`bert`) |
| `expand.default` 후 | **20/20** | — |

| op | 모양 규칙 | 함께 재현한 거부 |
|---|---|---|
| `select.int` | 그 차원을 **제거** (`slice` 와 다른 점) | 0-차원, `normalise_dim`, `normalise_index` |
| `tril` · `triu` | 그대로 — 바뀌는 것은 *어느 값이 0 이 되나*뿐 | 랭크 < 2 |
| `expand.default` | 앞에 차원을 붙이고 `-1` 은 기존 extent | 랭크 부족, 앞자리 `-1`, 비-singleton 확장 |

`expand` 만 dense 와 검사를 **완전히** 공유하지 못합니다. dense 는 extent 검사를 candle 의
`broadcast_as` 에서 공짜로 받는데 meta 에는 넘길 핸들이 없습니다. 그래서 그 절반만 여기서 직접
쓰고, 문면은 상류에서 실측해 옮겼습니다. **0 은 이 규칙에서 singleton 이 아닙니다** —
`(1,3) → (0,3)` 은 통과하고 `(0,3) → (2,3)` 은 거부입니다. `have <= 1` 로 쓰면 조용히 통과합니다.

### 7.3 상류 자신이 cpu 와 meta 에서 다른 답을 하는 자리 셋

§2.6 이 `torch.mm(cpu, meta)` 로 찾은 것과 **같은 종류**입니다. 이번에 셋 더 나왔습니다.
상류만으로 잰 것이고, 이 셰임은 관여하지 않습니다:

| 호출 | 상류 `cpu` | 상류 `meta` |
|---|---|---|
| `bitwise_not(float32)` | `NotImplementedError: "bitwise_not_cpu" not implemented for 'Float'` | **통과**, float32 를 돌려준다 |
| `clamp(float32, None, None)` | `RuntimeError: torch.clamp: At least one of 'min' or 'max' must not be None` | `ValueError: clamp called but both min and max are none!` |
| `where(uint8_cond, f32, f32)` | **통과** (deprecation 경고와 함께) | `RuntimeError: expected predicate to be bool, got torch.uint8` |

**이 셰임은 세 자리 모두에서 자기 dense 커널을 따릅니다** = 상류의 `cpu` 를 따릅니다. 문이
하나이므로 meta 가 dense 와 다른 답을 할 자리가 없고, 그것이 §5 가 `mm` 에 대해 적은 것과 같은
값입니다.

### 7.4 여전히 남은 것 — 그리고 그 크기

**작은 목록이 아닙니다. 숫자를 적습니다.** 커널이 있는 op 148 개 중 meta 에서 닿는 것은
**66 개**(이 표 + 자기 dense 커널 안에서 `is_meta()` 로 갈라지는 팩토리 10 개)이고,
**80 개가 여전히 닿지 않습니다.** 갈래별로:

> **2026-09 정정.** 이 표의 앞판은 82 개였고 축소에 `sum`, 뷰·모양에 `view` 가 들어 있었습니다.
> docs/VOICE4.md §4 가 그 둘의 `.default` 오버로드에 meta 커널을 넣었습니다 — voicestudio 의
> BigVGAN 이 `__init__` 에서 218 개의 리샘플링 필터를 `(taps / taps.sum()).view(1, 1, k)` 로
> 정규화하고, `from_pretrained` 가 그 `__init__` 을 meta 에서 돌리기 때문입니다.
>
> **계열이 닫힌 것이 아니라 각 계열의 한 멤버가 닫힌 것입니다.** `sum.dim_IntList` 와
> `mean.dim` 은 그대로 남아 있고, `reshape` 도 남아 있습니다 — `reshape` 는 *복사할 수*
> 있으므로 `view` 의 규칙으로 답하면 상류가 복사를 돌려줄 자리에서 뷰를 약속하게 됩니다.

| 갈래 | 수 | 예 |
|---|---:|---|
| 축소 | 21 | `sum.dim_IntList` · `mean` · `amax` · `max.dim` · `argmax` · `any` · `cumsum` · `topk` · `sort` |
| 뷰·모양 | 12 | `reshape` · `t` · `permute` · `transpose` · `slice` · `squeeze` · `unsqueeze` |
| 제자리 | 12 | `add_` · `mul_` · `sub_` · `div_` · `exp_` · `neg_` · `relu_` · `clamp_` |
| 축약 | 8 | `mm` · `bmm` · `matmul` · `addmm` · `baddbmm` · `_grouped_mm` · `convolution` · `embedding` |
| 인덱싱 | 7 | `index.Tensor` · `gather` · `masked_fill` · `index_put_` · `isin` |
| 합성·활성 | 6 | `_softmax` · `_safe_softmax` · `native_layer_norm` · SDPA · `gelu` · `silu` · `relu` |
| 결합·분할 | 4 | `stack` · `unbind` · `split_with_sizes` · `scatter` |
| 그 외 | 10 | `abs` · `ceil` · `bitwise_and`/`bitwise_or` · `floor_divide` · `constant_pad_nd` · `zeros_like` |

그러므로 **"meta 표면이 완성됐다" 고 읽으면 안 됩니다.** 이번 회차가 연 것은 정확히 하나입니다 —
**스무 아키텍처의 `init_empty_weights` 구성 경로**. 그것이 `from_pretrained` 가 요구하는
전부이고(§8.3), 측정으로 확인했습니다(§7.2 의 20/20). meta 텐서로 *순전파*를 돌리는 것 —
모양 추론 도구로 쓰는 것 — 은 위 표가 그대로 남아 있으므로 **여전히 안 됩니다.**

뷰 계열에는 §12 가 이미 적어둔 전제 조건이 하나 더 붙습니다: **meta 는 스트라이드를 들지
않습니다.** `t`/`permute` 의 meta 커널이 오는 날 그 커널이 `PyTensorBase` 에 그 필드를 먼저
추가해야 합니다. 이번 셋(`select`·`tril`·`expand`)은 전부 contiguous 결과라 그 문제를 건드리지
않습니다 — `expand` 만은 상류에서 stride-0 뷰이므로, **이 셰임의 meta `expand` 는 모양만 맞고
스트라이드 의미는 없습니다.** dense `expand` 도 `broadcast_as` 후 대개 `contiguous` 되므로 같은
갈래이고, §12 에 함께 적었습니다.

**`_aten_implemented()` 는 139 그대로이고 스키마도 4353 그대로입니다.** 그 상수는 "커널이 있고
*또한* `tools/golden/cases.py` 가 상류와 대조한다" 를 뜻하는데, **골든 하네스는 값을 비교하고
meta 는 정의상 값이 없습니다.** meta 지원은 이미 목록에 있는 op 들의 *성질*이므로 op 수가 늘지
않고, 새 철자를 만들지 않았으므로 `overloads.json`/`methods.json` 도 그대로입니다. 증거는
`pytests/test_shim.py` 에 있습니다(§11).

---

## 8. 함께 본 것 — `with torch.device(...)` 와 `set_default_device`

DEVICE_ABS §7.2 의 경고가 이 작업의 전제였습니다:

> **스택만 만드는 것은 의미가 없고 위험합니다.** `with torch.device("cpu"):` 가 성공하고
> `torch.zeros(2)` 는 모드를 쳐다보지 않으므로 **조용히 무시합니다.**

**그래서 둘 다 넣었습니다.** 한쪽만 넣을 바에는 지금처럼 `NotImplementedError` 로 죽는 편이
낫다는 §7.2 의 판단에 동의하고, 그래서 반쪽으로 끝내지 않았습니다.

### 8.1 세 조각

**(1) 진짜 모드 스택** (`bootstrap.py` `_install_torch_function_modes`). 벤더 트리는 이미
전부 갖고 있습니다 — `torch/overrides.py` 의 `TorchFunctionMode`/`_push_mode`/`_pop_mode`,
`torch/utils/_device.py` 의 `DeviceContext`, `torch/__init__.py` 의
`set_default_device`/`get_default_device`. 바닥의 `_C` 이름 다섯 개만 없었습니다:
`_push_on_torch_function_stack` · `_pop_torch_function_stack` ·
`_len_torch_function_stack` · `_get_function_stack_at` · `_is_torch_function_mode_enabled`.
`_DISCOVERED_RETURNS` 에서 상수 `0`/`False` 로 답하던 두 개를 빼고 실제 함수로 바꿨습니다.

**(2) `torch.device.__enter__` / `__exit__`** (`device.rs`). 상류의 `THPDevice_enter` 와 같은
모양입니다 — `torch.utils._device.DeviceContext` 를 만들어 스택에 밀어 넣고 **`self` 를**
돌려줍니다. 실측: `with torch.device("meta") as d: repr(d)` 는 `device(type='meta')` 입니다.
`DeviceContext.__enter__` 를 부르지 *않는* 것도 상류를 따른 것입니다 — 그 파이썬 쪽 진입은
모드를 스택 **바닥**으로 밀어내는 재배치를 하는데, 그것은 `set_default_device` 가 원하는
동작이고 어휘적으로 중첩된 `with` 가 가지면 안 되는 동작입니다.

**(3) 팩토리가 스택을 상의한다** (`bootstrap.py` `_torch_level_function`, `_tensor_factory`).
이것이 "조용히 무시" 를 막는 유일한 조각입니다. 모드가 없으면 전역 리스트의 진리값 검사
하나(`LOAD_GLOBAL` + 점프)이고, 있으면 상류의 `handle_torch_function` 과 같은 일을 합니다 —
**최상위 모드를 잠시 꺼내고** `mode.__torch_function__(fn, (), args, kwargs)` 를 부릅니다.
꺼내지 않으면 모드의 구현이 마지막에 `func(*args, **kwargs)` 를 다시 부르면서 무한 재귀합니다.

`fn` 이 **자기 자신을** `func` 로 넘기는 것이 필수입니다: `DeviceContext.__torch_function__` 은
`func in _device_constructors()` 로 판정하는데, 그 집합(36 개)은 `torch.zeros` · `torch.empty`
등을 `torch` 모듈에서 읽어 만든 것이고 그것이 바로 이 클로저 객체들입니다.

데코레이터로 감싸지 않고 두 갈래에 각각 써넣은 것은 비용 판단입니다 — 985 개 함수 전부에
파이썬 프레임을 하나씩 더하는 것보다 전역 검사 한 줄이 쌉니다.

### 8.2 실측 대조 — 상류와 이 셰임

같은 스크립트를 두 torch 로 돌린 결과입니다.

| | 상류 | 이 셰임 |
|---|---|---|
| `with meta: torch.zeros(2).device` | `meta` | `meta` |
| `with meta: torch.tensor([1.,2.]).device` | `meta` | `meta` |
| `with meta: torch.zeros(2, device="cpu").device` | `cpu` | `cpu` |
| 중첩 `with meta: with cpu:` (안 / 밖) | `cpu` / `meta` | `cpu` / `meta` |
| `with meta: torch.get_default_device()` | `device(type='meta')` | `device(type='meta')` |
| `with meta as d: repr(d)` | `device(type='meta')` | `device(type='meta')` |
| 블록 밖 | `cpu` | `cpu` |
| 스택 길이 (전 / 안 / 후) | 0 / 1 / 0 | 0 / 1 / 0 |
| `set_default_device("meta")` 후 | `meta` | `meta` |
| `set_default_device(None)` 후 | `cpu` | `cpu` |
| `with meta: nn.Linear(2,3)` 파라미터 장치 | `meta` | `meta` |

**`get_default_device()` 가 맞는 이유가 재미있습니다.** 벤더 코드는 인덱스 없는 라벨에 대해
`torch.tensor([]).device` 를 돌려주는데(`torch/__init__.py:1222`), `torch.tensor` 가 모드를
상의하므로 컨텍스트 안에서는 그것이 meta 텐서입니다. `torch.tensor` 의 meta 갈래
(`lib.rs`)가 없으면 이 줄이 깨집니다.

### 8.3 그리고 목적지 — `init_empty_weights`

```python
with torch.device("meta"):
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
# [('0.weight', (8,4), 'meta', 'Parameter'), ('0.bias', (8,), 'meta', 'Parameter'),
#  ('1.weight', (2,8), 'meta', 'Parameter'), ('1.bias', (2,), 'meta', 'Parameter')]

model.load_state_dict({...진짜 가중치...}, assign=True)
model(torch.ones(1, 4)).tolist()      # [[32.0, 32.0]]
```

**상류에서 같은 스크립트가 같은 숫자를 냅니다** (`[[32.0, 32.0]]`). 모양 · dtype ·
`Parameter` 타입 · `state_dict` 까지 전부 있고 **가중치용으로 할당된 바이트는 0** 입니다.
DEVICE_ABS §7.1 이 "`from_pretrained` 벽 뒤에 이것이 있다" 고 적은 그 경로입니다.

### 8.4 때운 것

**스택이 프로세스 전역입니다. 상류는 스레드 로컬입니다** (`PythonTorchFunctionTLS`). 한
스레드에서 들어간 모드를 다른 스레드가 보게 되며, 상류는 그러지 않습니다. 이 셰임의 측정된
경로 중 멀티스레드인 것이 없어서 `threading.local` 로 바꾸지 않았고, 바꾼다면 그 블록에서
두 줄입니다. 코드에 적어 두었습니다.

**`Tensor` 메서드는 모드를 상의하지 않습니다.** `_torch_level_function` 과 `torch.tensor` 만
연결했습니다. 상류는 메서드에 대해서도 `__torch_function__` 을 발화시킵니다.
`_device_constructors()` 의 36 개가 전부 모듈 수준 함수라서 장치 컨텍스트에 대해서는 차이가
없지만, **다른 종류의 모드가 오면 차이가 납니다.** 때운 것으로 적습니다.

**`_is_torch_function_enabled` 는 `False` 로 남겼습니다.** 모드 스택과 *다른* 질문입니다 —
서브클래스가 `__torch_function__` 을 오버라이드하는가이고, 벤더 트리에 그런 타입이 없다는
`_DISCOVERED_RETURNS` 위의 기존 근거가 그대로 유효합니다. **모드 쪽 절반만 진짜가 됐습니다.**

---

## 9. 디스패치 비용 — 늘었고, 얼마인지 쟀다

DEVICE_ABS §6 이 혼합 장치 게이트에 디스패치당 **+21 ns (+6%)** 를 이미 쓰고 있다고 적었고,
이번 변경이 그것을 더 늘리는지 재라는 지시가 있었습니다. **늘었습니다.**

방법은 §6 과 같습니다: 두 산출물을 만들어 **번갈아** 돌리고, 가장 싼 op 을 2 원소 텐서로
20 만 번, 프로세스당 7 회 반복 중 최솟값, 그 8 라운드 중 최솟값. 벤치 스크립트는
`/tmp/dev_bench.py` 로 저장소 밖입니다. 기준선은 이 브랜치의 부모(`016d65b`) 를
`git stash` 로 되돌려 같은 툴체인으로 빌드한 산출물입니다.

`load average 2.3~2.9` (8 코어), 다른 에이전트 없음. DEVICE_ABS 가 잰 `2.5~2.8` 과 같은 대역입니다.

| | `add.Tensor` | `cat.default` |
|---|---:|---:|
| 기준선 (`016d65b`) | 365.3 ns | 447.2 ns |
| 첫 판본 | 361.6→? (**+22 ns, +6%**) | (**+26 ns, +6%**) |
| **착지한 판본** | **378.9 ns (+13.6, +3.7%)** | **459.2 ns (+12.0, +2.7%)** |

**첫 판본에서 +22 를 +13.6 으로 줄인 것은 측정이 시킨 것입니다.** 두 가지였습니다.

1. **게이트가 클로저 안의 클로저였습니다.** `scan` 이 `visit` 를 부르고 둘 다 `first` 를
   가변 캡처해서, 안쪽 호출이 인라인되지 않았습니다. 둘 다 `first` 를 인자로 받는 자유
   함수(`scan_for_device`/`visit_for_device`, `#[inline]`)로 바꾸니 호출자 안으로 접혔습니다.
2. **인자마다 `Where` 를 만들고 있었습니다.** `Where::of` 는 `candle_core::Device` 를
   복제하는데, 이전 코드는 *첫* 텐서에서만 복제하고 비교는 참조로 했습니다. 지금은 표현을
   제자리에서 읽어 비교하고, `Where` 는 첫 텐서와 **거부 직전**에만 만듭니다.

`cat` 의 `collect::<PyResult<Vec<_>>>()` 를 미리 크기를 잡은 루프로 바꾼 것과
`PyTensorBase` 의 접근자들에 `#[inline]` 을 단 것도 같은 회차에 들어갔습니다.

**남은 +13.6 ns 는 무엇인가.** 정확히 분해하지 못했습니다. 후보는 셋이고, 셋을 따로 재는
산출물을 만들지 않았습니다:

- `tensor()` 가 `PyResult<&Tensor>` 가 되면서 커널의 모든 인자 접근에 분기가 하나씩
  (`add.Tensor` 는 2~4 회).
- `Repr` 이 enum 이 되면서 `PyTensorBase` 가 커졌습니다 (`Vec<usize>` 24 바이트).
- 게이트가 `Option<Where>` 를 돌려주고 `aten_dispatch` 가 그것을 match 합니다.

**모델 수준에서는 묻힙니다** — `docs/PERF.md` 의 2 층 블록이 2.22 ms 이고 디스패치 수십 회면
13.6 ns × 수십 = 마이크로초 단위입니다. **하지만 원소별 op 만 도는 마이크로벤치에서는
3.7% 가 보입니다.** DEVICE_ABS 의 21 ns 위에 얹히므로 게이트 이전 대비로는 약 35 ns 입니다.

**측정 한계.** `docs/PERF.md` §0 과 같습니다 — 절대값은 재현되지 않고, 유효한 것은 같은
조건에서 잰 A/B 비율뿐입니다. 그리고 **`docs/PERF.md` 의 2 층 블록 수치를 이번 변경 후 다시
재지 않았습니다.** 위 문단의 "묻힌다" 는 계산이지 측정이 아닙니다.

---

## 10. 실측 요약 — 107 항목 전사 대조

```
상류와 일치   41
불일치        66
```

**이 숫자를 그대로 읽으면 안 됩니다.** 불일치 66 개 중 meta 때문인 것은 19 개뿐이고, 나머지는
장치와 무관한 기존 구멍이 프로브에 걸린 것입니다. 갈래별로:

| 갈래 | 개수 | 성격 |
|---|---:|---|
| **텐서 `repr()`/`str()` 벽** | 18 | **기존 문제.** `torch._C._functorch.is_functorch_wrapped_tensor` 에서 막힙니다 — `docs/SPELLINGS.md` §4.1 이 이미 기록한 것이고, dense 텐서도 똑같이 막힙니다(확인함). 프로브가 `repr` 로 값을 찍어서 크게 잡혔습니다 |
| **meta 커널 없음** (§7) | 19 | 기록된 경계. `add`·`mm`·`view`·`reshape`·`t`·`slice`·`cat`·`sum`·`select`·`eq`·모양 오류·`m.meta_forward` 등 |
| **오버로드 테이블 항목 없음** | 9 | 장치와 무관. `rand`·`randn`·`eye`·`softmax`·`allclose`·`stack`·`empty_like`·`zeros_like`·`m.to_empty` |
| **`TensorBase` 멤버 부재** | 8 | 장치와 무관. `.stride()`·`.data_ptr()`·`.untyped_storage()`·`.numpy()`·`.add_`·`.new_empty`·`torch.save` |
| **게이트 메시지 문면** (§5.1) | 7 | 의도한 발산. 문이 하나면 메시지도 하나 |
| **autograd 부재** | 2 | 기존. `requires_grad=True` 를 이 셰임이 거부합니다 |
| **그 외** | 3 | ~~`torch.Size` 대신 튜플~~ — **해소됨, docs/SCALAR2.md §5.** `shape` 과 `size()` 가 이제 `torch.Size` 를 돌려주고 `state_dict` 의 shape 도 함께 따라옵니다. 당시 근거로 적힌 `TORCH_C.md` 에는 이 기록이 실제로 없었습니다(SCALAR2.md §5.3). 남은 둘: `__enter__` 의 `repr` 이 `torch._C.device` 로 찍힘 |

**meta 를 구현해서 새로 생긴 발산은 게이트 메시지 7 개뿐입니다.** 나머지 59 개는 meta 이전에도
같은 이유로 실패하던 것들이고, 19 개는 §7 이 의도적으로 남긴 경계입니다.

---

## 11. 판정 — 전부 종료 코드로

파이프로 읽지 않았습니다. 전부 파일로 리다이렉트한 뒤 `$?`.

```
골든           2258/2258, ops=96, pending 0                      EXIT=0
fault value                                                      EXIT=1
fault shape                                                      EXIT=1
fault dtype                                                      EXIT=1
스키마         233/233                                           EXIT=0
pytests        91 ok / 0 fail (84 -> 91, 골든 self-test 포함)     EXIT=0
호스트 빌드                                                       EXIT=0
android arm64  ELF 64-bit LSB shared object, ARM aarch64          EXIT=0
ios arm64      Mach-O 64-bit dynamically linked shared library    EXIT=0
```

**골든 op 수 96 과 스키마 233 은 그대로입니다.** 이유는 §7 입니다 — 새 aten op 도, 새 철자도
만들지 않았습니다.

### 새로 붙인 테스트 (`rust/torch_c/pytests/test_shim.py`, +7)

| 테스트 | 무엇을 고정하나 |
|---|---|
| `test_meta_tensors_carry_shape_and_dtype_and_no_data` | §2.1/§2.2 — 메타데이터 전부, 그리고 **두 개의 서로 다른** 거부 |
| `test_meta_drops_the_device_index_where_cpu_does_too` | §2.4 — `meta:7` 로 만든 텐서는 `meta`, 그러나 라벨끼리는 불일치 |
| `test_the_gate_refuses_a_mixed_device_op_and_finds_it_in_a_sequence` | **§5** — 거부 갈래, 위치·키워드 × 텐서·리스트·튜플 4 조합, 그리고 `copy_` 예외의 양방향 |
| `test_meta_transfers_go_one_way_only` | §2.3 — `device=` 부재가 "제자리" 라는 §5.2 의 계약이 처음으로 관측 가능 |
| `test_ops_without_a_meta_kernel_name_themselves` | §7 — 거부가 자기 이름을 대는 것, `_aten_implemented()` 가 그대로인 것 |
| `test_the_initialisers_a_module_constructor_runs_are_no_ops_on_meta` | §4.1 — in-place 넷이 무동작이고 수신자를 돌려주는 것 |
| `test_meta_road_through_the_vendored_tree` | **§8 전체** — 서브프로세스에서 컨텍스트 매니저 · 중첩 · 명시 인자 우선 · 스택 균형 · `set_default_device` · `init_empty_weights` · `load_state_dict(assign=True)` · 순전파(`[[32.0, 32.0]]`, 상류와 같은 숫자) |

### 11.1 이번 회차 (§7) 의 판정

```
pytests        260 ok / 0 fail   (253 -> 260)                     EXIT=0
골든           4284/4284, ops=139, pending 1 (기대치)             EXIT=0
골든 self-test 13 comparators x 11 fault modes, 0 problem(s)      EXIT=0
스키마         4353/4353                                          EXIT=0
20-아키텍처 순전파 스윕      20/20                                EXIT=0
20-아키텍처 meta 구성 스윕   20/20  (§7.2)                        EXIT=0
llama3-rope from_pretrained + generate                            EXIT=0
```

**골든 op 수 139 와 스키마 4353 은 그대로입니다** — 이유는 §7.4.

> **Correction (문서 감사):** 이 블록의 "139"/"4284"/"4353" 은 §7.4 본문이 이미 쓰고 있는
> "148" 과 같은 커밋(`3c9d000`) 안에서 나온 서로 다른 숫자입니다 — §7.4 의 "커널이 있는 op
> 148 개" 는 이 라운드가 끝난 시점의 `_aten_implemented()` 값이고, 이 §11.1 블록은 그보다
> 앞서 잰 스냅샷을 그대로 옮긴 것으로 보입니다. 둘 중 148 쪽이 이 라운드의 실제 종료 상태와
> 일치합니다 — 재측정 확인, 현재 베이스라인은 5634/5634, ops=148, pending 1
> (`docs/AUDIT.md` 베이스라인과 동일).

### 11.2 새로 붙인 테스트 (+7) 과 고친 것 하나

| 테스트 | 무엇을 고정하나 |
|---|---|
| `test_meta_comparisons_answer_bool_whatever_went_in` | 12 키 × 9 dtype × 3 스칼라, 그리고 8 가지 브로드캐스트 조합. **`bool` 을 별도 줄로 단언** |
| `test_meta_elementwise_arithmetic_broadcasts_and_promotes_like_the_dense_kernel` | `div` 가 정수 쌍을 띄우는 것, `mul` 만 승격하고 셋은 거부하는 것, `set_default_dtype` 결합 |
| `test_meta_where_broadcasts_three_operands_and_takes_dtype_from_the_values` | 3-피연산자 조인, 조건이 dtype 을 주지 않는 것, 조건 dtype 거부 |
| `test_meta_unary_promotions_are_the_dense_families_own` | 승격 계열 8 개와 **보존** 계열(`neg`·`bitwise_not`) 을 **양방향으로** |
| `test_meta_clamp_and_pow_share_the_dense_kernels_dtype_ladders` | 골든이 한 번 고쳐준 `clamp` 사다리, `pow` 의 wrapped-number 규칙 |
| `test_meta_shape_kernels_drop_expand_and_keep_the_triangle` | §7.2 셋의 모양 규칙과 거부 전부. `(1,3)→(0,3)` 통과 / `(0,3)→(2,3)` 거부 |
| `test_the_llama3_rope_init_runs_on_meta_end_to_end` | **사용자 보고 그 자체.** 같은 식을 `cpu` 로도 돌려 모양과 dtype 을 대조 |

고친 것: `test_ops_without_a_meta_kernel_name_themselves` 가 `add.Tensor` 를 "meta 커널이
없는 것" 으로 단언하고 있었습니다. 지우지 않고 **경계를 옮겼습니다** — 축소·축약·뷰 열 개로
바꾸고, **반대 방향 단언을 추가**했습니다(§7.1 이 넣은 것들은 자기 이름을 대면 안 된다).
그것이 없으면 이 테스트는 meta 표를 통째로 비워도 통과합니다.

### 11.3 사보타주 — 13 개 결함, 13 개 다 잡힘

*"실패할 수 없는 검증은 검증이 아니다"* (CLAUDE.md §5.5). meta 커널의 출력은 모양과 dtype
**둘뿐**이므로, `.shape` 만 읽는 테스트는 dtype 결함을 구조적으로 못 봅니다. 그래서 규칙마다
한 줄씩 고장 내고 세었습니다. 각 회차는 재빌드 + 전체 스위트입니다.

| 주입한 결함 | 실패한 테스트 |
|---|---:|
| 비교 `Scalar` 가 `bool` 대신 입력 dtype | **2** |
| 비교 `Tensor` 가 `bool` 대신 입력 dtype | 1 |
| 산술이 `arith_tag` 대신 피연산자 dtype (`div` 가 안 뜬다) | 1 |
| 이항 모양이 브로드캐스트 대신 왼쪽 피연산자 | 1 |
| `where` 가 값이 아니라 조건에서 dtype 을 가져옴 | **2** |
| `where` 모양이 3-조인 대신 조건의 것 | 1 |
| 단항 승격 계열이 입력 dtype 을 그대로 통과 | 1 |
| `neg` 가 보존 대신 승격 | 1 |
| `clamp` 가 사다리 대신 입력 dtype | 1 |
| `pow.Tensor_Scalar` 가 wrapped-number 대신 베이스 dtype | 1 |
| `select.int` 이 차원을 제거하지 않고 1 로 남김 | 1 |
| `expand` 가 0 extent 를 singleton 으로 취급 | 1 |
| `tril`/`triu` 가 랭크 거부를 잃음 | 1 |

**0 건이 조용히 통과했습니다.** 두 건이 두 테스트를 깨뜨린 것은 rope 종단 테스트가 같은
결함을 독립적으로 잡았기 때문입니다 — 단위 테스트와 종단 테스트가 겹치는 것이 의도입니다.

---

마지막 것이 서브프로세스인 이유는 `test_device_road_through_the_vendored_tree` 와 같습니다 —
`torch.device.__enter__` 는 벤더 트리의 `torch.utils._device` 를 필요로 하고, 독립 `_C` 주위에는
그것이 없습니다.

### 고친 기존 테스트 하나

`test_device_road_through_the_vendored_tree` 가 `t.to("meta")` 를 `cuda` 와 나란히 **거부되는
것**으로 단언하고 있었습니다. 그 기대가 낡았습니다 — `cuda` 는 이 빌드가 링크하지 않은
백엔드이고 `meta` 는 백엔드가 필요 없는 장치입니다. 목록에서 빼고, 왜 뺐는지를 그 자리에
주석으로 남기고, 대신 §11 의 새 테스트들이 "돌려준 텐서가 *맞는* 텐서인지" 를 잽니다.

---

## 12. 때운 것 / 못 한 것 / 모르는 것

### 때운 것

- **모드 스택이 프로세스 전역입니다** (상류는 스레드 로컬) — §8.4.
- **`Tensor` 메서드는 모드를 상의하지 않습니다** — §8.4. 장치 컨텍스트에 대해서는 차이가
  없지만 다른 종류의 모드에 대해서는 있습니다.
- **meta 는 스트라이드를 들지 않습니다.** `is_contiguous()` 가 meta 에 대해 무조건 `True` 를
  답하고, 상류는 `t()` 한 meta 에 대해 `False` 입니다. meta `t`/`permute` 커널이 오는 날
  그 커널이 `PyTensorBase` 에 그 필드를 먼저 추가해야 합니다 (§7.4).
  **§7.2 의 `expand` 가 이 항목의 첫 실제 사례입니다.** 상류에서 `expand` 는 stride-0 뷰인데
  이 셰임의 meta `expand` 는 모양만 맞고 스트라이드 의미가 없습니다. dense `expand` 도
  `broadcast_as` 뒤에 대개 `contiguous` 되므로 같은 갈래이지만, "meta 가 만든 것은 전부
  contiguous 다" 라는 문장이 **더 이상 자명하지 않습니다.**
- **`expand` 의 extent 검사만 dense 와 공유하지 못합니다** (§7.2). dense 는 candle 의
  `broadcast_as` 에서 받고 meta 는 직접 씁니다. 문면은 상류에서 실측해 옮겼지만, 두 벌인
  이상 갈라질 수 있는 유일한 자리입니다.
- **§7.1 의 원소별 커널은 상류의 *meta* 가 아니라 이 셰임의 *dense* 를 따릅니다.** 셋이
  갈리는 자리가 §7.3 에 있고, 거기서 상류 자신이 `cpu` 와 `meta` 로 다른 답을 합니다.
- **`arange` 의 `arange_has_cpu_kernel` 을 meta 에서 건너뜁니다** — 상류의 meta 커널이
  그렇다는 실측에 따른 것이고, 두 경로가 다른 검사를 하는 유일한 자리입니다 (§4.2).
- **게이트 메시지가 상류의 meta 문면과 다릅니다** (§5.1).

### 못 한 것

- **나머지 82 개 op 의 meta 모양 추론** (§7.4). 축소·축약·뷰·인덱싱·제자리·합성이
  통째로 남아 있습니다. 열린 것은 **구성 경로 하나**이지 meta 표면 전체가 아닙니다.
- **`meta-llama/Llama-3.2-1B` 자체로는 확인하지 못했습니다.** Hub 에서 게이트되어 있고
  `HF_HOME` 에 캐시되어 있지 않습니다(`SmolLM2-135M` 만 있습니다). 대신 상류 torch 로
  `rope_parameters={"rope_type": "llama3", ...}` 를 가진 작은 Llama 체크포인트를 써 두고,
  **같은 `AutoModelForCausalLM.from_pretrained` 경로로** 이 셰임에서 읽었습니다 —
  즉 `_compute_llama3_parameters` 를 meta 에서 실제로 통과시켰고, 손으로 만든 경로가
  아닙니다. 로짓은 상류와 1e-5 안에서 같고 `generate` 는 토큰 열이 같습니다.
- **`m.to("cpu")` (meta 모듈을 CPU 로) 와 `m.to_empty(device=...)`.** 둘 다 `empty_like` 에
  걸리는데, 그것은 오버로드 테이블에 항목이 없습니다 — **meta 와 무관한 기존 구멍**이고
  `zeros_like`·`ones_like` 와 같은 갈래입니다(DEVICE_ABS §9 가 `ones_like` 로 이미 기록).

  > **Correction (문서 감사, 재측정):** `empty_like` 는 이제 `overloads.json` 에 항목이
  > 있습니다 — `zeros_like`·`ones_like` 도 마찬가지입니다. `m.to_empty(device="cpu")` 는
  > 지금 실제로 동작합니다(재측정 확인). `m.to("cpu")` 는 여전히 거부되지만 이유가
  > 다릅니다 — 오버로드 누락이 아니라 `NotImplementedError: Cannot copy out of meta
  > tensor; no data! Please use torch.nn.Module.to_empty() instead ...`, 그리고 상류도
  > **동일한 문면**으로 거부합니다(재측정 확인). 즉 이 갈래는 "구멍"이 아니라 상류와 일치하는
  > 의도된 거부이고, 남은 것은 `m.to_empty` 가 아니라 `m.to("cpu")` 자체가 상류처럼
  > 영구히 거부되어야 한다는 것뿐입니다.
- **`torch.save(meta_tensor)`.** `PyTorchFileWriter.write_end_of_file` 에서 막힙니다 — 기존 구멍.
- **`torch.device.__enter__` 를 독립 `_C` 에서 부르면 `ImportError`.** 벤더 트리가 없으면
  `torch.utils._device` 도 없습니다. 상류도 같은 자리에서 같은 이유로 실패합니다.

### 모르는 것

- **남은 +13.6 ns 가 셋 중 어디서 오는지 모릅니다** (§9). 분해하는 산출물을 만들지 않았습니다.
- **`docs/PERF.md` 의 2 층 블록을 다시 재지 않았습니다.** 마이크로벤치가 모델 수준으로 어떻게
  번역되는지는 계산이지 측정이 아닙니다.
- **인덱스가 살아남는 장치에서 `from_candle` 의 하드코딩이 무엇을 하는지 여전히 모릅니다**
  (§6). meta 로는 시험할 수 없습니다.
- **`meta` 텐서에 대한 `Tensor.data =` (`replace_with`) 의 별칭 규칙을 따로 재지 않았습니다.**
  dense 와 같은 코드를 지나가고, `load_state_dict(assign=True)` 가 그 경로로 동작하는 것은
  확인했지만, meta→dense 로 갈아끼울 때 이전 뷰가 무엇을 보는지는 묻지 않았습니다.
- **`_local_scalar_dense` 외의 데이터 읽기 경로를 전부 훑지 않았습니다.** `tolist`·`item` 은
  막았고, `.numpy()`·`.data_ptr()`·`.untyped_storage()` 는 dense 에서도 미구현이라 meta 에서
  따로 막을 것이 없었습니다. 그 셋이 구현되는 날 meta 갈래를 함께 넣어야 합니다.
- **혼합 장치 `copy_` 에서 `meta ← cpu` 를 무동작으로 두었을 때, 수신자의 *뷰*가 무엇을 보는지
  모릅니다.** `docs/OPS4.md` §8 의 미해결과 만나는 자리인데(`copy_` 는 `replace_with` 입니다),
  이번 판본에서는 아무것도 바꾸지 않으므로 뷰 질문이 생기지 않습니다. 진짜 장치 간 전송이
  구현되는 날 다시 물어야 합니다.

---

## 13. 재현

```bash
cd /path/to/repo
bash vendor/vendor_torch.sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-meta
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
bash vendor/install_shim.sh

PY=/Volumes/macMini/caches/spike-venv/bin/python

$PY tools/golden/compare.py                       > /tmp/g.log 2>&1;  echo "EXIT=$?"
$PY tools/golden/compare.py --inject-fault value  > /tmp/fv.log 2>&1; echo "EXIT=$?"
$PY rust/torch_c/pytests/verify_schemas.py        > /tmp/s.log 2>&1;  echo "EXIT=$?"
PYTHON=$PY sh rust/torch_c/pytests/run.sh         > /tmp/p.log 2>&1;  echo "EXIT=$?"

# 전사 대조: 같은 프로브를 두 torch 로 돌리고 diff
PYTHONDONTWRITEBYTECODE=1 TORCH_USE_RTLD_GLOBAL=1 \
  PYTHONPATH=$PWD/torchnative/src/main $PY <probe> > ours.txt
(cd /tmp && $PY <probe> > upstream.txt)

# A/B 벤치: 기준선 산출물을 stash 로 만든다 (checkout 금지 -- CLAUDE.md)
git stash push -- rust/torch_c/src
(cd rust/torch_c && cargo build --release) && cp $TORCH_C_ARTEFACT /tmp/base_C.so
git stash pop
(cd rust/torch_c && cargo build --release) && cp $TORCH_C_ARTEFACT /tmp/meta_C.so
for i in 1 2 3 4; do $PY /tmp/dev_bench.py /tmp/base_C.so; $PY /tmp/dev_bench.py /tmp/meta_C.so; done
```

프로브와 벤치 스크립트는 저장소 밖(`/tmp`)입니다.
