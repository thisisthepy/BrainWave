# `from_pretrained` — 진짜 모델 · 진짜 가중치 · 진짜 파이썬

`docs/models/E2E_REAL.md` §6.2 가 남긴 벽에서 이어집니다. 그 회차는 `from_pretrained` 가 **모델을
만드는 데까지** 갔고 `torch.UntypedStorage.from_file` 에서 멈췄습니다 — 가중치를 한 바이트도
읽지 못했습니다.

측정일 2026-08-28. 호스트 `darwin/arm64`, CPython 3.13.0, 상류 torch 2.13.0 ·
transformers 5.15.1 · safetensors 0.8.0 (`/Volumes/macMini/caches/spike-venv`).
**벤더링 트리는 한 줄도 고치지 않았습니다.**

---

## 0. 한눈에

| | 전 (E2E_REAL) | 후 |
|---|---|---|
| `from_pretrained` — 모델 생성 | 통과 | 통과 |
| `from_pretrained` — **가중치 적재** | **미통과** (`from_file`) | **통과, 네 경로 전부** (§3) |
| 적재된 가중치 vs 상류 | — | **비트 단위 일치 (worst `0.0`)**, 21/21 텐서 |
| 그 모델의 순전파 로짓 | — | 상류와 **2.235e-08**, argmax 완전 일치 |
| `torch.load(mmap=True)` | 미통과 | **통과.** 21/21 텐서가 `mmap=False` 와 **`0.0`** 일치 |
| safetensors `mmap` 백엔드 | 이름 대고 거절 | **통과.** `pread` 백엔드와 **`0.0`** 일치 |
| shared tensor (가중치 공유) | **미측정** | **통과, 비트 일치** (§6) |
| sharded `index.json` | **미측정** | **통과, 비트 일치** (7 샤드) |
| `_metadata` 가 붙은 state dict | **미측정** | **통과, 비트 일치** |
| `bfloat16` 체크포인트 | 미시도 | **적재는 비트 일치.** 순전파는 다름 (§6.1) |
| **허브의 진짜 사전학습 모델** | 미시도 | **SmolLM2-135M, 273 텐서 · 1.63억 파라미터, worst `0.0`** (§7) |
| `pytests/run.sh` | 149 통과 | **155 통과** |
| `tools/golden/compare.py` | 2486/2486, ops=116 | **2496/2496, ops=117** |
| `verify_schemas.py` | 270/270 | **272/272** |

**이번 회차의 한 문장**: 이 저장소가 처음으로 **허브에서 받은 진짜 사전학습 체크포인트를
읽어**, 그 가중치가 상류와 비트 단위로 같음을 확인했습니다.

**아직 안 되는 것**: 그 진짜 모델의 **순전파**. 적재가 아니라 커널 둘이 남았고 §7.1 에
이름이 있습니다.

---

## 1. 판정 기준을 먼저 정한다 — "적재됐다" 는 판정이 아니다

`docs/models/CKPT.md` §4 가 기록한 사고가 이 절의 이유입니다. 순진한 구현에서 `torch.load` 가
성공하고 `load_state_dict` 가 `<All keys matched successfully>` 를 돌려주는데
**모든 가중치가 `0.0`** 이고 예외는 어디에도 없었습니다.

그래서 이 문서의 모든 판정은 **값**입니다. 두 층으로 봅니다.

1. **적재된 파라미터 하나하나를 상류의 것과 비트 단위로 비교** (`worst == 0.0`).
   이것이 0 을 잡고, 동시에 **오프셋이 레코드 헤더만큼 어긋나 옆 텐서의 바이트를
   읽는 경우**도 잡습니다.
2. **그 모델의 순전파 로짓을 상류와 비교.**

그리고 **음성 대조**를 함께 둡니다 — 가중치를 적재하지 않은 같은 모델의 로짓은 진실에서
멀어야 합니다. 실측 **0.2416** 이고, 적재된 쪽의 2.235e-08 과 일곱 자릿수 떨어져 있습니다.
이것이 없으면 위 두 비교는 "적재를 하든 안 하든 통과하는" 비교일 수 있습니다.

### 1.1 실제로 깨서 확인했다

`CLAUDE.md` §5.5. 국소 편집으로 구현을 고장 내고 다시 돌렸습니다 (`git checkout` 은 쓰지
않았습니다 — `cp` 백업).

| 무엇을 깼나 | 무엇이 빨개졌나 |
|---|---|
| `from_file` 이 파일 바이트 대신 **0 을 채우게** | 5개 테스트. 비트 비교가 `model.layers.0.self_attn.o_proj.weight` 를 이름으로 지목 |
| `__getitem__` 슬라이스가 **오프셋을 무시**하게 (옆 텐서 사고 재현) | 5개 테스트. 비트 비교가 `lm_head.weight`, 차이 **1.84e+37** |
| `view.dtype` 이 재해석 대신 **수치 변환**하게 | 골든 10개 중 **9개** (나머지 하나는 빈 텐서라 발산할 값이 없음) |

**첫 두 경우 모두 `from_pretrained` 자체는 예외 없이 성공했습니다.** 0 을 채운 쪽은
argmax `[0,0,0,0]` 을, 오프셋을 무시한 쪽은 1e37 규모의 로짓을 돌려주었고, 둘 다 모델
객체를 정상적으로 반환했습니다. §1 이 "적재됐다는 판정이 아니다" 라고 쓴 것이 바로 이
모양입니다.

---

## 2. `UntypedStorage.from_file` — 진짜 mmap 이 필요한가를 먼저 쟀다

지시받은 첫 단계가 이것이었고, 답은 **필요 없다** 입니다. 근거는 추론이 아니라 상류에서
실측한 두 가지입니다.

**(a) 호출자는 언제나 `shared=False` 를 넘긴다.**

```python
# torch/serialization.py:1591
shared = get_default_mmap_options() == MAP_SHARED
```

`torch/utils/serialization/config.py:15` 의 `mmap_flags` 기본값이 `2` = `MAP_PRIVATE`
이므로 이 식은 `False` 입니다. 실측:

```
get_default_mmap_options() = 2      mmap.MAP_PRIVATE = 2   MAP_SHARED = 1
torch.load(mmap=True) 이 넘기는 shared = False
```

safetensors 의 `mmap` 백엔드도 `shared=False` 로 부릅니다.

**(b) `MAP_PRIVATE` 매핑의 관측 가능한 내용은 파일의 바이트 그 자체다.**

```
s2 = from_file(p, False, N);  s2[0] = 0xEE
  s2[0]                   238
  디스크의 0번 바이트        0      <- 쓰기가 파일에 닿지 않는다
  새로 뜬 매핑의 0번 바이트   0      <- 다른 매핑에서도 보이지 않는다
반면 shared=True 로 같은 것을 하면 디스크의 바이트가 171 로 바뀐다
```

즉 `shared=False` 매핑은 **파일의 사본**이고, 그 바이트를 읽어 버퍼에 담은 것과
관측적으로 같습니다. 차이는 둘뿐입니다:

- **상주 방식** — mmap 은 페이지 단위로 게으르게, 읽기는 전부 미리. 성능/메모리 성질이고
  값이 아닙니다. 큰 체크포인트에서는 실제 비용이므로 §8 에 미해결로 적어 둡니다.
- **`_get_filename()`** — 상류도 `shared=False` 에 **`None`** 을 답합니다(실측).
  즉 사본이 답하는 것과 같은 답입니다.

**그래서 읽어서 채웁니다.** `shared=True` 는 이야기가 다릅니다.

### 2.1 `shared=True` 는 이름을 대고 거절한다

`MAP_SHARED` 는 **쓰기가 파일과 다른 프로세스에 도달해야** 한다는 요구입니다. 이 shim 의
저장소는 소유된 버퍼이고 그것을 할 수 없습니다. 조용히 사본을 건네면 §1 의 0 과 같은
계열의 실패 — 누군가 쓰기 전까지는 맞아 보이는 답 — 가 됩니다. `docs/models/E2E_REAL.md` §7 의
1b 항목이 이 판단을 예고했고, 이번에 그대로 착지했습니다.

### 2.2 상류와 대조한 것 — 14개 관측점

테스트는 숫자를 적어 두지 않습니다. **같은 파일에 대해 상류가 답하는 것을 그 자리에서
재고, shim 의 답과 비교합니다.** 14/14 일치:

```
크기       nbytes(full/16/0), 키워드 인자, nbytes 기본값(=0)
내용       앞 8바이트, s[16:24]
슬라이스   빈 슬라이스, 끝을 넘긴 슬라이스(클램프), 음수 인덱스
주소       s[16:24].data_ptr() - s.data_ptr() == 16
기타       element_size, device, filename(None)
```

그리고 상류가 내는 세 거절을 그대로 냅니다:

```
nbytes > 파일 크기   RuntimeError: file <p> size <1024> is smaller than the
                     required mapping size <1025>
없는 파일            RuntimeError: unable to open file <p> in read-only mode:
                     No such file or directory (2)
s[::2]               RuntimeError: Trying to slice with a step of 2, but only
                     a step of 1 is supported
```

### 2.3 슬라이스는 진짜 뷰다

`torch/serialization.py:2115` 는 **텐서마다 하나씩** 전체 파일 저장소에서 잘라냅니다:

```python
storage = overall_storage[storage_offset : storage_offset + nbytes]
```

슬라이스를 사본으로 만들면 (a) 체크포인트가 메모리에 두 벌이 되고, (b) `data_ptr()` 이
바이트가 실제로 있는 곳과 무관해집니다. 그래서 버퍼는 `Arc<Vec<u8>>` 이고 뷰는 그 안의
오프셋을 듭니다 — 상류에서 실측한 `data_ptr` 관계(`+16`)가 그대로 성립합니다.

`filled` 불변식은 이것을 견딥니다. **뷰는 부모의 `filled` 를 상속하고**, `_shim_fill` 은
버퍼를 공유하는 저장소에서 **거절합니다** (`Arc::get_mut` 이 `None`). 즉 "채운 뒤에
자른다" 만 가능하고, 그것이 zip 컨테이너가 하는 순서입니다.

### 2.4 `filled` 불변식의 문장이 하나 달라졌다

`storage.rs` 의 원래 문장은 "`_shim_fill` 이 유일한 문" 이었습니다. `from_file` 이 두
번째 문이 되므로, 불변식을 **원래 의도대로** 다시 적었습니다:

> `filled` 는 "한 함수만 세울 수 있다" 가 아니라 **"바이트를 실제로 전달한 것만 세울 수
> 있다"** 이다.

`_shim_fill` 은 버퍼를 받고 `from_file` 은 파일을 읽습니다. 순수한 할당은 아무것도
세우지 않습니다. legacy 컨테이너가 우연히 통과할 수 없다는 성질은 그대로입니다 —
`docs/models/CKPT.md` §3.3 의 거절이 이번에도 살아 있고, 테스트가 그것을 단언합니다.

---

## 3. 벽 넷을 순서대로

각 벽은 앞의 벽을 메운 뒤에야 보였습니다. 프로브는 네 경로를 **각각 독립된
`try/except`** 로 돌려 첫 벽이 다음 벽을 가리지 않게 했습니다.

| # | 벽 | 어디서 | 메운 것 |
|---|---|---|---|
| 1 | `UntypedStorage.from_file` | `serialization.py:1594`, safetensors `mmap` | `storage.rs` (§2) |
| 2 | `UntypedStorage.__getitem__(slice)` | `serialization.py:2115` | `storage.rs`, 진짜 뷰 |
| 3 | `torch.empty_like` | `modeling_utils.py:4763,4771` | `overloads.json` + **meta 커널** (§5) |
| 4 | `torch.asarray` | safetensors 의 `mmap` 백엔드 | `lib.rs::_asarray` + `aten.view.dtype` (§4) |

### 3.1 벽 하나를 넘을 때마다 어디까지 갔는지

프로브 출력 그대로입니다. 세 번째 열이 그 회차에 도달한 지점입니다.

```
착수 시점
  safetensors mmap    from_file
  safetensors bytes   from_file            <- E2E_REAL §6.2 가 기록한 것과 다름:
  .bin mmap           from_file               bytes 경로는 이미 21/21 을 읽고 있었고
  .bin no-mmap        from_file               empty_like 에서 멈춰 있었다

from_file + __getitem__ 후
  safetensors mmap    asarray              (21/21 텐서 읽음)
  나머지 셋           empty_like           (21/21 텐서 읽음)

empty_like 후
  safetensors mmap    asarray
  나머지 셋           OK

asarray + view.dtype 후
  네 경로 전부        OK
```

**두 번째 줄이 이번 회차가 정정한 것입니다.** `E2E_REAL.md` §6.2 는 `disable_mmap=True`
로도 우회되지 않는다고 적었는데, 그것은 `.bin` 경로에 대해서만 맞습니다. safetensors
쪽은 `modeling_utils.py:4462` 가 `disable_mmap` 일 때 `_safe_load_bytes` 로 갈라지므로
이미 파일을 끝까지 읽고 있었고, 막고 있던 것은 **`empty_like`** 였습니다.

---

## 4. `torch.asarray` 와 `aten.view.dtype` — safetensors 의 기본 백엔드

`torch.asarray` 는 aten op 이 **아닙니다** (`torch.ops.aten.asarray` 는 2.13.0 에서
`AttributeError`). `torch.frombuffer` 와 같은 계열의 `_C` 바인딩이고, 같은 이유로
`lib.rs` 에 있으며 같은 바이트 리더(`from_le_bytes`)를 씁니다.

**무엇이 부르는지 쟀습니다.** `torch.asarray` 를 감싸고 `safe_open(..., backend="mmap")`
을 돌린 결과:

```
torch.asarray(UntypedStorage, dtype=torch.uint8)                 <- get_slice()[...]
torch.asarray(UntypedStorage, dtype=torch.uint8, device='cpu')   <- get_tensor()
```

그 뒤 `TorchDispatchMode` 로 잰 op 순서:

```
empty.memory_format  set_.source_Storage  select.int  _local_scalar_dense
lift_fresh  view.dtype  detach  view.default  [alias]
```

즉 safetensors 는 바이트를 `uint8` 텐서로 만든 뒤 **`.view(dtype)` 으로 dtype 을
말합니다.** 그래서 `aten.view.dtype` 이 필요했습니다.

**`asarray` 는 저장소 형태로 좁혔고 그것을 이름으로 말합니다.** 상류의 `asarray` 는
텐서·시퀀스·스칼라·numpy 배열도 받지만, 그것들을 구현하는 것은 측정된 호출자 없이
`torch.tensor` 의 변환 규칙을 다시 유도하는 일입니다 — `E2E_REAL.md` §1.2 가 경계한 바로
그것. 저장소가 아닌 것은 받은 타입을 말하며 거절하고 `torch.frombuffer`/`torch.tensor`
를 가리킵니다.

### 4.1 `view.dtype` 은 캐스트가 아니다

`1.0` 을 `int32` 로 보면 `1` 이 아니라 **`1065353216`** 입니다. 비트 재해석이고, 그래서
**바이트로 나갔다 바이트로 들어옵니다** — `to_le_bytes` 다음 `from_le_bytes`, 후자는
`torch.load` 리더와 `torch.frombuffer` 가 이미 쓰던 그 함수입니다. 수치 변환을 경유하는
어떤 경로도 반올림하고, 반올림하는 체크포인트 리더는 거절하는 리더보다 나쁩니다.

상류의 세 거절을 실측해 그대로 냅니다. **C++ 쪽 dtype 이름**까지 포함해서입니다
(`dtype.rs::cpp_name`) — 문장은 상류의 것인데 문제를 지목하는 단어만 다르면, 그 메시지는
필요한 곳에서만 쓸모없어집니다.

```
self.dim() cannot be 0 to view Float as Byte (different element sizes)
self.size(-1) must be divisible by 4 to view Byte as Float (different element sizes), but got 23
self.stride(-1) must be 1 to view Byte as Float (different element sizes), but got 2
```

**마지막 것 때문에 이 커널은 사본을 만들면서도 stride 를 검사합니다.** `stride(-1) == 1`
이 바로 "사본과 진짜 뷰가 같은 답을 내는" 조건입니다 — 재해석은 **마지막 차원 안에서만**
바이트를 합치거나 쪼개므로, 그 차원이 빽빽하기만 하면 행 우선으로 읽어 다시 읽은 결과가
같은 자리에 같은 바이트를 놓습니다. 검사를 빼면 상류가 거절하는 모양을 조용히 답하게 됩니다.

**골든 케이스 10개**로 상류와 대조합니다 (`tools/golden/cases.py::view_dtype_cases`) —
폭 1/2/4/8 을 넓히는 방향과 좁히는 방향, 같은 폭의 정수↔부동소수, rank 3, 빈 텐서.
`ops covered` 가 116 → 117 이 된 것이 이것입니다.

### 4.2 `to_le_bytes` 는 `f16`/`bf16` 을 거절하고, 그 비대칭은 의도된 것이다

`f16`/`bf16` 텐서의 비트에 candle 을 통해 닿으려면 `half` 크레이트의 타입을 이름으로
불러야 하는데, 이 크레이트는 `half` 에 **의존하지 않습니다** — candle 의 의존성으로
따라올 뿐이고, `CANDLE_DEPS.md` 는 의존성을 얻어걸리지 않게 하자는 문서입니다.

**있는 경로에서는 비용이 0 입니다.** 체크포인트는 *바이트*로 도착하므로 `bf16` 가중치는
`uint8 → bf16` 방향으로 `from_le_bytes` 를 타고, 그것은 원시 버퍼를 읽으므로 이 함수를
부르지 않습니다. 닫혀 있는 것은 반대 방향(`bf16_tensor.view(torch.uint8)`)뿐이고 아무도
거기 도달하지 않았습니다. §6.1 의 `bf16` 체크포인트가 **비트 단위로 적재되는** 것이 그
증거입니다.

---

## 5. `empty_like` — 모델을 meta 에서 데리고 나오는 op

`from_pretrained` 는 모듈 트리 전체를 `init_empty_weights` 아래에서 만듭니다. 그래서
모든 파라미터·버퍼가 meta 에서 시작하고, **체크포인트가 주지 않은 것**은
`torch.empty_like(param, device=...)` 로 건너옵니다 —
`modeling_utils.py:4763`(누락 키)과 `:4771`(비영속 버퍼: 정의상 체크포인트에 없습니다).

그래서 필요한 것이 둘이었습니다:

1. `overloads.json` 항목 — `torch.empty_like(...)` 라는 **자유 함수**가 해석될 자리.
   `verify_schemas.py` 가 270 → 272 가 된 것이 이 두 줄입니다.
2. **meta 커널** — 입력이 meta 이므로 `meta_dispatch` 가 먼저 받습니다.

meta 커널은 조밀 커널의 규칙을 **같은 헬퍼로 같은 순서로** 다시 씁니다. `E2E_REAL.md`
§6.1 이 그 이유를 적어 두었습니다 — meta 커널이 조밀 커널과 다른 dtype 을 약속하면
그 shape·dtype 으로 할당한 뒤에 계산이 거절됩니다.

**"empty" 는 여기서도 0 을 답합니다.** 그것이 안전한 이유를 가정하지 않고 적어 둡니다:
이 커널이 만드는 값은 전부 읽히기 전에 덮어씌워집니다 — 누락 키는
`_initialize_missing_keys` 가, 비영속 버퍼는 모듈 자신의 초기화가. 만약 하나라도 그렇지
않다면 그 0 이 순전파에 도달하고, `pytests/test_shim.py` 의 로짓 비교가 말합니다.
실측 로짓 차이가 2.235e-08 이라는 것이 지금은 그렇지 않다는 증거입니다.

---

## 6. `E2E_REAL.md` §6.2 가 미측정으로 남긴 체크포인트 형태 셋 — 전부 쟀다

지시받은 항목입니다. 셋 다 **컨테이너의 성질**이지 수치의 성질이 아니고, 셋 다 조용히
틀리는 방식이 따로 있습니다.

| 형태 | 조용히 틀리는 방식 | 결과 |
|---|---|---|
| **shared tensor** (`tie_word_embeddings=True`) | safetensors 는 중복 저장소를 쓰지 않고 헤더에 적는다. 무시하는 리더는 `lm_head` 를 잃고 새로 초기화한다 — 예외 없음 | **비트 일치.** 로짓 4.47e-08 |
| **sharded** (`model.safetensors.index.json` + 7 샤드) | 첫 파일에서 멈추는 리더는 나머지 층을 초기화 상태로 둔다 — 예외 없음 | **비트 일치.** 로짓 7.45e-08 |
| **`_metadata`** (`nn.Module.state_dict()` 가 붙이는 속성, `.bin` 컨테이너) | 언피클이 그 속성을 만나 죽거나, 조용히 버린다 | **비트 일치.** 로짓 9.39e-07 |

셋 다 회귀 스위트에 박혀 있습니다
(`test_the_four_hard_checkpoint_shapes_load_with_the_right_weights`). 샤드 케이스는
**샤드가 실제로 둘 이상인지**를 픽스처가 단언합니다 — `max_shard_size` 가 언젠가 무시되면
그 테스트는 아무것도 검사하지 않는 테스트가 되기 때문입니다.

### 6.1 `bfloat16` — 적재는 정확하고, 순전파는 다르다

실제 체크포인트가 저장되는 dtype이므로 함께 쟀습니다.

```
가중치      비트 단위 일치 (worst 0.0)
로짓        0.042 차이,  argmax 는 일치
```

**차이는 적재가 아니라 계산에서 옵니다.** 이 빌드는 여러 커널에서 `bf16` 을 `f32` 로
올려서 계산합니다 (`aten.rs`, `DType::F16 | DType::BF16 => DType::F32` 가 다섯 곳).
즉 상류보다 **높은 정밀도로** 순전파를 돌고, 두 답은 bf16 자신의 해상도(1.0 근처 ulp
≈ 0.0078) 규모로 갈라집니다. 로짓 스케일이 ~0.4 이므로 0.042 는 그 규모입니다.

그래서 이 케이스의 테스트는 **가중치에는 비트 일치를, 로짓에는 느슨한 경계(0.1)를**
겁니다. 두 성질이 다른 것을 재고 있으므로 하나의 숫자로 묶지 않습니다. 이 상향 캐스트가
옳은지는 이 회차가 판정하지 않았고 §8 에 미해결로 둡니다.

---

## 7. 허브의 진짜 사전학습 모델 — SmolLM2-135M

네트워크가 되었으므로 **`E2E_REAL.md` §7 의 2번 항목**을 열었습니다.

```
HuggingFaceTB/SmolLM2-135M     model.safetensors 269,060,552 바이트
LlamaForCausalLM               273 텐서 · 162,826,560 파라미터 · 전부 bfloat16
```

```
from_pretrained    0.7 초
WEIGHTS worst difference: 0.0     at None
        all-zero weights: False
```

**273개 텐서 전부가 상류와 비트 단위로 같습니다.** 비교는 텐서마다 앞 8개 · 뒤 4개 ·
`abs().max()` 이고, `abs().max()` 는 전 원소를 지나므로 "가운데만 틀린" 경우를
놓치지 않습니다.

이것이 지금까지 이 저장소에서 가장 큰 실제 적재입니다. `docs/models/CKPT.md` §6 "모르는 것" 의
첫 줄 — *"실제 사전훈련 체크포인트로는 검증하지 못했습니다"* — 이 닫혔습니다.

### 7.1 그 모델의 순전파는 아직 안 됩니다 — 다음 벽 둘

> **Correction (문서 감사, 2026-09):** 둘 다 열렸습니다. `aten.where.ScalarOther` 는 지금
> `_aten_implemented()` 에 있고, `enable_gqa=True` 는 `bootstrap.py::_sdpa_math` 가
> (`docs/kernels/SDPA.md` 감사가 이 라운드에서 이미 지목한 그 composite) 헤드를 반복해 지원합니다
> (`bootstrap.py:5312` 의 `if enable_gqa:` 분기). 실측 재확인, 2026-09:
> `AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")` 의 순전파가
> **성공**하고(`logits.shape == (1, 2, 49152)`), `model.generate(ids, max_new_tokens=5)` 도
> **성공**합니다(§8 항목 2 의 `_prepare_attention_mask_for_generation` 벽도 함께 닫힌 것으로
> 보입니다 — 별도로 추적하지 않았지만 `generate()` 자체가 도는 것으로 그 벽이 더는 막지
> 않음을 알 수 있습니다). `docs/design/DESIGN.md` §11.1 감사(round 1)가 이미 같은 모델로 이것을
> 확인했었고, 이 문서 자체의 §7.1/§8 이 그 사실을 반영하지 못한 채 남아 있었습니다.
> <!-- DOCWATCH: op-implemented aten.where.ScalarOther -->
> <!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _sdpa_math present -->

적재가 아니라 **커널** 문제입니다. 두 어텐션 구현이 각각 다른 벽을 냅니다.

```
attn_implementation="sdpa"  (기본)
  sdpa_attention.py:154
  NotImplementedError: scaled_dot_product_attention(enable_gqa=True) --
    upstream's flash kernel broadcasts the key/value head dimension internally;
    this shim's does not. Repeat the heads before calling.

attn_implementation="eager"
  masking_utils.py:603   torch.where(mask, torch.tensor(0.0, ...), min_dtype)
  NotImplementedError: aten op not implemented in torch._C shim:
    aten.where.ScalarOther
```

SmolLM2-135M 은 `num_attention_heads=9`, `num_key_value_heads=3` 의 **그룹 질의
어텐션**이라 `enable_gqa=True` 로 갑니다. §4 의 작은 Llama 는
`num_key_value_heads == num_attention_heads` 라 그 분기에 닿지 않습니다.

**둘 다 이번 작업의 범위 밖입니다** — 체크포인트 리더가 아니라 커널이고, 각각 골든
케이스가 따로 필요합니다. 다음 회차의 시작점으로 §8 에 이름을 적어 둡니다.

`aten.where.ScalarOther` 는 이미 `overloads.json` 의 `where` 항목에 **스키마로는 있고
커널이 없는** 상태입니다 — 즉 해석은 되고 디스패치에서 이름을 대며 거절합니다.

---

## 8. 미확인 — 숨기지 않는 것

| # | 항목 | 상태 |
|---|---|---|
| 1 | **진짜 사전학습 모델의 순전파** | ~~**미통과.** 커널 둘: `sdpa(enable_gqa=True)` 와 `aten.where.ScalarOther` (§7.1)~~ **정정 (문서 감사, 2026-09): 통과함** — §7.1 정정 참고 |
| 2 | `generate()` — **진짜 모델로** | ~~**미통과.** `_prepare_attention_mask_for_generation` 에서 `aten.mul.Tensor: int64 vs bool` 승격이 없어 멈춥니다. 커널 문제이지 체크포인트 문제가 아닙니다~~ **정정 (문서 감사, 2026-09): 통과함** — 실측: `model.generate(ids, max_new_tokens=5)` 이 SmolLM2-135M 에서 성공 |
| 3 | `from_file(shared=True)` | **거절.** §2.1. 재현 불가능한 것이지 미구현이 아닙니다 |
| 4 | **메모리** — `from_file` 은 파일 전체를 미리 읽습니다 | **미측정.** mmap 의 게으른 상주를 잃었고, 135M 모델(269 MB)에서는 문제가 되지 않았습니다. 7B 급에서 이것이 실제 제약인지는 재지 않았습니다 |
| 5 | `bf16` 을 `f32` 로 올려 계산하는 것 | **판정 안 함.** §6.1. 적재는 정확하고 계산이 상류보다 정밀합니다. 어느 쪽이 옳은지는 이 회차의 질문이 아니었습니다 |
| 6 | `to_le_bytes` 의 `f16`/`bf16` | **거절.** §4.2. 체크포인트를 *읽는* 방향은 영향이 없습니다 |
| 7 | `torch.asarray` 의 저장소 외 형태 | **거절.** §4. 텐서·시퀀스·스칼라·numpy |
| 8 | `torch.load` legacy 포맷 | **여전히 거절, 그리고 그것이 옳습니다.** `docs/models/CKPT.md` §4. 테스트가 계속 단언합니다 |
| 9 | 가중치 공유의 **동일성** | §6 은 값이 보존되는 것만 확인했습니다. `docs/models/CKPT.md` §5 가 적은 대로 이 shim 은 복사하므로 `tied_a is tied_b` 의 저장소는 둘입니다 |
| 10 | `torch.zeros_like` 의 자유 함수 형태 | **미구현.** `modeling_utils.py:4746` 의 FSDP 분기가 부르는데 그 분기에 도달하지 않았습니다. `empty_like` 와 달리 넣지 않은 것은 §1.2 의 규칙 그대로입니다 |
| 11 | 안드로이드 · iOS | **미측정.** 호스트(darwin/arm64)에서만 돌렸습니다. 크로스 **컴파일**도 확인하지 못했습니다 — `cargo check --target aarch64-apple-ios` 와 `--target aarch64-linux-android` 는 둘 다 이 워크트리에 `docs/platform/RUST_CROSSBUILD.md` 가 요구하는 환경(타깃용 CPython 배포본, `PYO3_CROSS_LIB_DIR`, NDK 툴체인)이 없어 **빌드 스크립트 단계에서** 멈춥니다(`pyo3-ffi`, `onig_sys`). 즉 이 회차의 코드가 통과하는지 아닌지가 아니라 환경이 없어서 못 잰 것입니다. 새로 쓴 것은 `std::sync::Arc` · `std::fs::{metadata,read}` · `pyo3::types::{PySlice,PyType}` 뿐이고 조건부 컴파일이나 플랫폼 API 는 없지만, **그것은 근거이지 측정이 아닙니다** |
| 12 | `int8`·`uint16`·`uint64`·complex 로 저장된 체크포인트 | **거절.** candle 이 담지 못합니다. dtype 이름을 대며 거절합니다 |

---

## 9. 검증

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh     exit 0   155 통과 (전 149, +6)
$PY tools/golden/compare.py                   exit 0   2496/2496, ops=117 (전 2486/116)
$PY rust/torch_c/pytests/verify_schemas.py    exit 0   272/272 (전 270)
```

**보고를 종류별로 나눕니다** (`CLAUDE.md` §5.3):

| 종류 | 무엇 |
|---|---|
| 기능 추가 | `UntypedStorage.from_file` · `__getitem__`(정수/슬라이스, 진짜 뷰) · `_get_filename` · `is_shared` · `torch.asarray` · `aten.view.dtype` 커널 · `empty_like` 의 meta 커널과 오버로드 항목 · `tensor::to_le_bytes` · `TorchDType::cpp_name` |
| 결함 수정 | 없음 |
| 테스트 추가 | 6개 — `from_file` 의 상류 대조, 두 mmap 경로의 일치, 네 적재 경로의 비트 비교, 그 네 경로의 로짓, 어려운 형태 넷, 음성 대조. 골든 케이스 10개 |
| 문서 정정 | `E2E_REAL.md` §6.2 의 `disable_mmap` 서술(§3.1), 같은 문서의 미측정 항목 셋(§6), `CKPT.md` §3.1 의 mmap 백엔드 거절 → 일치 |
| 삭제 | 없음. 이름이 바뀐 테스트 하나(`..._are_refused_by_name` → `..._is_refused_by_name_and_the_mmap_backend_agrees`)는 절반이 반대 주장이 되었기 때문이고, legacy 거절 쪽 단언은 그대로입니다 |

### 9.1 테스트 수가 진척이 아닌 이유를 적어 둡니다

`CLAUDE.md` §5.3. +6 중 **다섯**은 이 회차가 새로 연 능력에 대한 값 비교이고, 하나
(어려운 형태 넷)는 **이미 되고 있었지만 아무도 확인한 적이 없던 것**을 고정한 것입니다.
후자는 기능 추가가 아닙니다 — `tied`/`shard`/`meta` 는 §3 의 벽 넷을 메우자 별도 작업
없이 통과했고, 이 회차가 한 일은 그것을 **재고 박은 것**입니다.

---

## 10. 재현

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cd /Volumes/macMini/worktrees/bw-ckpt2
bash vendor/vendor_torch.sh
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-ckpt2
PY=/Volumes/macMini/caches/spike-venv/bin/python
bash vendor/install_shim.sh
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib

PYTHON=$PY sh rust/torch_c/pytests/run.sh          # 155
$PY tools/golden/compare.py                        # 2496/2496 ops=117
$PY rust/torch_c/pytests/verify_schemas.py         # 272/272
```

§7 의 진짜 모델은 회귀 스위트에 **넣지 않았습니다** — 269 MB 를 받아야 하고 네트워크가
필요하므로 `pytests/run.sh` 의 성질(오프라인에서 몇 초)을 바꿉니다. 손으로 재현하는
방법은 이렇습니다:

```sh
export HF_HOME=/Volumes/macMini/caches/hf-home
# 상류가 진실을 적는다 (벤더 트리를 PYTHONPATH 에 넣지 않는다)
ATTN=eager $PY /Volumes/macMini/caches/ckpt2-scratch/real_model.py truth
# shim 이 같은 체크포인트를 읽고 대조한다
ATTN=eager PYTHONPATH=$PWD/torchnative/src/main TORCH_USE_RTLD_GLOBAL=1 \
    $PY /Volumes/macMini/caches/ckpt2-scratch/real_model.py shim
```

**`compare.py` 와 `verify_schemas.py` 는 벤더 트리를 `PYTHONPATH` 에 넣지 않고
돌립니다** — 넣으면 상류 torch 를 가려 기준선이 사라집니다.

`run.sh` 에는 함정이 하나 있습니다. 벤더 트리의 `_C.abi3.so` 가 방금 빌드한 것과 같은지
`cmp` 로 확인하는데, 이번에 그 `cmp` 가 한 번 `Killed: 9` 로 죽어 **최신 산출물을
"stale" 로 보고**했습니다. `install_shim.sh` 를 다시 돌리면 지나갑니다.

---

## 11. 지시받은 것에 대한 채점

| # | 목표 | 결과 |
|---|---|---|
| 1 | `from_file` 이 무엇을 요구하는지 재고, mmap 이 진짜 필요한지 근거를 대라 | **달성.** 필요 없습니다 — 근거는 §2 의 두 실측(기본 플래그가 `MAP_PRIVATE`, 그 매핑의 쓰기가 파일에 닿지 않음). `shared=True` 는 이름을 대고 거절 |
| 2 | TDD — 실패하는 테스트를 먼저 | **달성.** 다섯 테스트를 먼저 쓰고 넷이 빨간 것을 확인한 뒤 구현했습니다 (다섯 번째인 음성 대조는 처음부터 초록이어야 하고 그랬습니다) |
| 3 | **`from_pretrained` 로 실제 체크포인트를 읽어 순전파까지** | **달성.** 상류가 쓴 체크포인트로 네 경로 전부, 가중치 비트 일치 · 로짓 2.235e-08 · argmax 일치 |
| 4 | 조용한 0 을 경계하라 | **달성.** §1.1 — 0 을 채우도록 고장 냈을 때 `from_pretrained` 는 **예외 없이 성공했고** 값 비교만이 잡았습니다 |
| 5 | 판정은 값이다 | **달성.** 모든 판정이 상류와의 값 비교이고, 음성 대조가 붙어 있습니다 |
| 6 | 미측정 셋(shared/sharded/`_metadata`)을 마주치면 재라 | **달성.** §6. 셋 다 통과, 셋 다 회귀에 고정 |
| 7 | 끝까지 밀어라 | **부분.** 허브의 진짜 사전학습 모델까지 갔고 **적재는 완전**합니다(§7). 그 모델의 **순전파는 미통과**이고, 남은 것은 커널 둘입니다(§7.1) |

7번을 "부분" 이라고 적는 이유를 분명히 합니다. **"이 프로젝트가 진짜 모델·진짜 가중치·
진짜 파이썬을 끝까지 돌린다" 고 쓸 수 있는 상태는 두 가지 뜻이 있고, 하나만
달성했습니다.** 손으로 만든 작은 Llama 는 체크포인트에서 읽어 순전파까지 상류와 일치하고,
허브의 SmolLM2-135M 은 가중치를 완전히 읽었지만 아직 계산하지 못합니다.
