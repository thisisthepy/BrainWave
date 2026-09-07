# SAVE — 체크포인트를 **쓴다**

`docs/models/CKPT.md` §6 "못 한 것" 의 마지막 줄이 이 문서의 시작점입니다.

> | 체크포인트 **쓰기** | 범위 밖. `UntypedStorage.__getstate__` 가 이름을 대며 거부 |

읽는 쪽은 이미 끝까지 갑니다(CKPT.md, CKPT2.md). 이 문서는 **쓰는 쪽**을 재고, 어디까지
뚫었고, 무엇을 이름을 대며 거부한 채로 남겼는지를 적습니다.

측정일 2026-09-02. 호스트 `darwin/arm64`, CPython 3.13, 상류 torch 2.13.0
(`/Volumes/macMini/caches/spike-venv`). 벤더링 트리는 한 줄도 고치지 않았습니다.

---

## 1. 트레이스백이 가리킨 곳은 사고 현장이 아니다

지시받은 증상은 이것이었습니다.

```
torch.save(torch.ones(2,2), io.BytesIO())
NotImplementedError: not implemented in torch._C shim: PyTorchFileWriter.write_end_of_file
```

**이 메시지는 가리개(masking)입니다.** 실제로 무엇이 거부했는지는 예외 하나만 봐서는 알 수
없고, 계측해야 보입니다. 전체 트레이스백을 찍어 보면 두 덩어리가 나옵니다:

```
  File "torch/serialization.py", line 1253, in _save
    pickler.dump(obj)
  File "torch/_tensor.py", line 328, in _reduce_ex_internal
    not torch._C._has_storage(self)
NotImplementedError: not implemented in torch._C shim: torch._C._has_storage   <-- 진짜 첫 벽

During handling of the above exception, another exception occurred:

  File "torch/serialization.py", line 1002, in save
    with _open_zipfile_writer(f) as opened_zipfile:
  File "torch/serialization.py", line 854, in __exit__
    self.file_like.write_end_of_file()
NotImplementedError: ... PyTorchFileWriter.write_end_of_file                    <-- 보고된 것
```

**구조가 원인입니다.** `torch.save` 의 본체는 `with _open_zipfile_writer(f) as z:` 블록
안에 있고, `_open_zipfile_writer_buffer.__exit__` 는 무조건 `write_end_of_file()` 를
부릅니다(serialization.py:854). 블록 안에서 예외가 나면 `__exit__` 가 실행되고, 거기서
**두 번째** 예외가 나면서 전파되는 것은 두 번째 쪽입니다. 즉 이 컨텍스트 매니저는
**블록 안의 모든 실패를 자기 실패로 덮어씁니다.**

그래서 `PyTorchFileWriter` 만 구현했다면 다음 벽에서 다시 같은 메시지를 봤을 것입니다 —
`__exit__` 가 성공하기 시작할 때까지, 안쪽 벽은 한 번도 이름을 대지 못합니다.

### 1.1 계측으로 뽑은 진짜 벽의 순서

앞 벽을 파이썬에서 임시로 메우고 다음을 보는 방식으로 잰 **완전한 순서**입니다
(CKPT.md §3.2 가 읽기 쪽에 쓴 것과 같은 방법).

| # | 벽 | 어디서 부르나 | 무엇이 필요한가 |
|---|---|---|---|
| 1 | `torch._C._has_storage(tensor)` | `_tensor.py:328` `_reduce_ex_internal` | 텐서가 스토리지를 갖는지 |
| 2 | `TensorBase.untyped_storage()` | `_tensor.py:311` `_typed_storage` | 텐서 뒤의 **전체** 버퍼 |
| 3 | `TensorBase.storage_offset()` | `_tensor.py:519` | 뷰의 시작 원소 |
| 4 | `TensorBase.stride()` | `_tensor.py:520` | 뷰의 스트라이드 |
| 5 | `torch._C._get_tensor_metadata(tensor)` | `_tensor.py:531` | 상류는 `{}` (실측) |
| 6 | `PyTorchFileWriter(...)` 의 세 메서드 | `serialization.py:1257-1312`, `854` | zip 컨테이너 쓰기 |

`PyTorchFileWriter(buffer, crc32, alignment)` 의 **생성자 자체는 원래 통과했습니다** —
타입 캐치올이 만든 합성 타입이라 인스턴스는 만들어지고 메서드가 거부합니다. 그래서 실패가
`__exit__` 까지 늦춰졌고, 그것이 §1 의 가리개가 생긴 조건입니다.

`torch.is_storage`, `torch.serialization._get_storage_alignment`,
`torch.serialization.get_crc32_options` 는 전부 이미 통과합니다.

### 1.2 `docs/training/BACKWARD.md` §14 가 이미 여기까지 왔었다

이 회차가 처음 발견한 것이 아닙니다. `docs/training/BACKWARD.md` §14.1 이 같은 가리개를 이미
기록했고 §14.2 가 같은 사슬을 이미 쟀습니다. **바뀐 것은 그 회차의 결론입니다** — §3 이
그 이야기입니다. 그리고 `README.md` 는 그 정정을 반영하지 않아, 지금도
*"`torch.save` refuses"* 라고 적혀 있습니다.

§14.2 의 표에는 **틀린 줄이 둘 있고**, 둘 다 같은 오해에서 나왔습니다 —
`untyped_storage()` 가 *텐서의* 바이트를 돌려준다고 본 것입니다.

| §14.2 가 적은 것 | 실제 |
|---|---|
| `storage_offset()` — *"always 0 — storages here are copies, never windows"* | **0 이면 안 됩니다.** 상류의 `untyped_storage()` 는 텐서가 올라앉은 **버퍼 전체**이고, `torch.save` 는 `(storage, offset, size, stride)` 넷을 그 버퍼에 대한 색인으로 씁니다. 0 을 답하면 `x[1]` 이 버퍼 맨 앞에서 시작하는 것으로 저장됩니다 |
| `stride()` — *"contiguous strides from the shape"* | `w.t()` 의 stride 는 `(1, 4)` 입니다. `docs/kernels/VIEWS.md` §6 이후 candle 의 `Layout` 이 진짜 stride 를 들고 있으므로 정확히 답할 수 있습니다 |

§14.2 는 `_cdata` 를 *"an identity key, for storage de-duplication"* 라고만 적고, **무엇의
identity 인지**는 적지 않았습니다. 파이썬 객체의 identity 로 답하면 de-duplication 은
정확히 반대로 동작합니다(§3.2).

---

## 2. 판정 기준 — "저장됐다" 는 판정이 아니다

`docs/models/CKPT.md` §1 이 읽기 쪽에 세운 기준을 그대로 뒤집습니다. 예외 없이 파일이 만들어지는
것은 아무것도 증명하지 않습니다. 이 문서의 판정은 셋이고, **첫째만이 인수 기준**입니다.

1. **상류가 우리가 쓴 것을 읽는다.** 다른 프로세스의 상류 torch 2.13.0 이 파일을 열어,
   **원본 객체와 `torch.equal`** 로 같아야 합니다. 우리만 읽을 수 있는 파일은 `torch.save`
   가 아닙니다.
2. **상류가 쓴 것을 우리가 여전히 읽는다** (CKPT.md 의 성질, 회귀하지 않았는지 재확인).
3. **shim 안에서의 왕복** — 저장하고 다시 읽어 값·dtype·shape 가 같다.

**3 은 가장 약한 기준입니다.** 쓰는 쪽과 읽는 쪽이 *같은 틀린 배치*에 합의하면 통과합니다.
그래서 인수 기준이 1 이고, 3 은 "빨간 것이 어느 쪽인지" 를 국소화하는 용도로만 둡니다.

---

## 3. `untyped_storage()` — BACKWARD §14.3 의 결론을 뒤집는다

§14.3 이 이 작업을 멈춘 자리이고, 그 논거는 이렇습니다.

> So an `untyped_storage()` on this stack can only return a **copy**. `torch.save` would never
> notice, because it only reads. Every other caller of `untyped_storage()` would — a write through it
> would land nowhere, silently. (...) implementing item 6 for `torch.save`'s sake would put a lie on
> the public surface to satisfy the one caller that cannot detect it.

**위험에 대해서는 맞고, 처방에 대해서는 틀렸습니다.** 위험이 요구하는 것은 *별칭*이 아니라
**거절**입니다. 읽으면 정확하고 쓰려 하면 이름을 대며 거절하는 스토리지는 거짓말이 아니라
스스로 이름을 대는 축소이고, 그것은 반대편의 `filled` 불변식과 정확히 같은 모양입니다.
존재하지 않는 `torch.save` 는 그 축소가 아닙니다.

그래서 쓰기 문 **네 개 전부가 이름을 대며 거절합니다.** 셋은 상류에도 있고 벤더 트리에서
전부 **메시지 없는** `raise NotImplementedError` 였습니다(`torch/storage.py` 62·65·173행) —
DESIGN.md §6 이 금지하는 익명 거절이고, 그것이 있는 자리에서는 고칠 수 없습니다.
`UntypedStorage(torch._C.StorageBase, _StorageBase)` 의 MRO 가 우리 쪽을 앞에 둡니다.

```
UntypedStorage.__setitem__   거절, snapshot 이라고 이름을 댐
UntypedStorage.copy_         거절
UntypedStorage.resize_       거절
UntypedStorage._shim_fill    거절 (snapshot 에 한해서 — 로더가 쓰는 문은 그대로 열려 있음)
```

### 3.1 스냅숏은 뷰가 아니라 **버퍼 전체**다

`untyped_storage()` 가 돌려주는 것은 텐서의 원소들이 아니라 **텐서가 올라앉은 candle 버퍼
전체**입니다. 상류의 것이 그렇기 때문입니다. `torch.save` 는 텐서를
`(storage, storage_offset, size, stride)` 로 기록하고, 그 세 숫자는 건네받은 바이트에 대한
색인입니다. 뷰를 실체화해서 건네면 **stride 와 offset 이 자기 페이로드에 대해 거짓말하는
파일**이 나옵니다 — 읽히고, 조용히 틀립니다. `docs/models/CKPT.md` §4 와 §5 가 반대 방향에서
기록한 바로 그 실패 모양입니다.

그래서 `to_le_bytes` 가 아니라 새 함수(`tensor.rs::storage_snapshot`)입니다. 전자는 *뷰*를
읽고 후자는 *스토리지*를 읽습니다.

**반쪽 부동소수(`f16`/`bf16`)를 여기서 읽습니다.** `to_le_bytes` 는 그것을 거절하고, 이유로
*"this crate does not depend on `half`"* 를 답니다 — 그 문장은 `reduced.rs` 가 그 의존성을
가져온 시점에 사실이 아니게 되었습니다(`Cargo.toml` 에 `half = "2.7"`). 그대로 두면
**`bfloat16` 가중치를 저장할 수 없습니다.** `to_le_bytes` 자체는 건드리지 않았습니다 —
그것은 `aten.view.dtype` 의 역함수이고, 그 op 이 무엇을 받는지는 자기 골든 케이스를 가진
별개의 결정입니다.

### 3.2 사본이 잃는 것은 바이트가 아니라 **identity** 이고, 그것이 실제로 하중을 받는다

```python
# torch/serialization.py:1235
storage_key = id_map.setdefault(storage._cdata, str(len(id_map)))
```

**한 파일이 레코드를 몇 개 쓰는지는 `_cdata` 가 정합니다.** `x`, `x.t()`, `x[1]` 세 텐서가
한 버퍼를 볼 때, 상류는 레코드 **하나**를 쓰고 세 텐서에 서로 다른 offset/stride 를
붙입니다(실측 — §4). 사본이 자기 주소로 답하면 레코드 **셋**, 버퍼 사본 셋, 그리고
**셋이 서로의 뷰였다는 사실이 사라진 파일**이 나옵니다.

그래서 스냅숏은 자기가 복사해 온 candle 버퍼의 주소를 달고 다닙니다
(`storage.rs::origin`). `data_ptr()` 과 `_cdata` 는 그 주소로 답합니다 — 두 경로가 이 숫자에
묻는 유일한 질문이 *"이 둘이 같은 스토리지인가"* 이고, 사본 자신의 주소는 그 질문에
**아니오** 라고 답하기 때문입니다. 그 숫자는 한 번도 역참조되지 않습니다.

주소는 `Arc<RwLock<Storage>>` 안의 `Storage` 의 주소이고(candle 의
`Tensor::storage_and_layout()` 이 공개), 텐서가 살아 있는 동안 유효합니다 — 저장 경로에서는
피클되는 객체가 텐서를 붙들고 있으므로 항상 살아 있습니다.

**사보타주로 확인**: `origin` 을 떼어 사본 주소로 답하게 하면, 세 뷰가
`['data/0', 'data/1', 'data/2']` 를 쓰고 그 테스트 하나가 빨개집니다.

---

## 4. 컨테이너 — `_ZipWriter`

`_ZipRecords` 의 반쪽이고, 같은 자리(`bootstrap.py`)에 같은 이유로 있습니다: 컨테이너는
평범한 zip 이고, 그 안에서 계산되는 것은 없으며, CPython 이 검증된 zip 쓰기를 이미 들고
있습니다. 이 클래스가 가진 것은 `zipfile` 이 모르는 두 가지 — torch 의 **레코드 이름**과
torch 의 **페이로드 정렬**입니다.

### 4.1 상류 아카이브를 먼저 뜯어봤다

```
'archive/data.pkl'               local_extra= 18  data_off=  64  aligned64=True  cd_extra=0
     local extra bytes: 46420e00 5a5a5a5a...              <- id 0x4246, 길이 14, 'Z' 14개
'archive/data/0'                 local_extra= 62  data_off= 832  aligned64=True  cd_extra=0
'archive/version'                                          b'3\n'
'archive/.data/serialization_id'                           b'0843495070226412367704396183885188234126'
```

읽은 것 넷:

1. **모든 레코드의 페이로드가 64 바이트 경계에서 시작합니다.** torch 는 *로컬* 파일 헤더의
   extra 필드를 `0x4246` id 로 채워 그것을 만듭니다. 중앙 디렉터리의 extra 는 **비어
   있습니다**. CPython 의 `zipfile` 은 둘 다 하지 않으므로 직접 합니다(그리고 `writestr`
   직후 `info.extra` 를 비워 중앙 디렉터리에는 들어가지 않게 합니다).
2. **`version` 레코드는 `_save` 가 아니라 C++ 의 `writeEndOfFile()` 이 씁니다.** 값은
   `b"3\n"`.
3. `.data/serialization_id` 도 같은 자리에서 쓰입니다.
4. 아카이브 이름은 파일명에서 확장자 하나를 뗀 것입니다 — `x.pt` → `x/`,
   `a.b.c.pt` → `a.b.c/`, `no_ext` → `no_ext/`. 버퍼로 저장하면 `archive/`.

### 4.2 정렬은 필수가 아니라 **충실도**다 — 재서 확인했다

패딩을 0 으로 만들고(그리고 자체 검사도 꺼서) 다시 돌리면, **상류는 그 파일을 문제없이
읽습니다.** 빨개지는 것은 아카이브 모양을 단언하는 테스트 하나뿐입니다. 그래도 정렬을
하는 이유는 `_save` 가 `.storage_alignment` 레코드에 `64` 라고 **써 넣기 때문**입니다 —
정렬하지 않으면 그 레코드가 자기 파일에 대해 거짓말을 하고, 어떤 리더도 불평하지 않습니다.
`_ZipRecords.get_record_offset` 이 읽어 쓰는 숫자이기도 합니다.

`zip64` 도 계산에 넣습니다: `zipfile` 은 2 GB 를 넘는 항목의 로컬 헤더에 20 바이트를 더
붙이므로(`ZipFile._open_to_write`), 그것을 빼먹으면 큰 체크포인트의 페이로드가 전부 20
바이트씩 어긋납니다 — 예외 없이.

### 4.3 `version` 은 필수다 — 이것도 재서 확인했다

`version` 레코드를 쓰지 않으면 **상류가 거절합니다**:

```
RuntimeError: Expected hasRecord("version") to be true, but got false.
```

그런데 **shim 자신의 리더는 그 파일을 읽습니다** — `_ZipRecords` 는 `version` 을 보지
않기 때문입니다. 이것이 §2 가 "우리가 읽을 수 있다" 를 인수 기준으로 삼지 않는 이유의
구체적인 사례입니다. shim 만으로 검증했다면 이 결함은 통과했을 것입니다.

### 4.4 `serialization_id` 는 **쓰지 않습니다** — 쓰는 것이 더 나쁩니다

상류의 것은 40 자리이고, 20 자리씩 두 덩어리(레코드 이름 해시의 합성, 내용 CRC 의 합성)
입니다. 앞 절반이 `std::hash<std::string>` 이라 **구현 정의**이고, 이식 가능하게 같은 수를
낼 방법이 없습니다. 즉 여기서 만든 id 는 **같은 내용에 대해 상류와 다른 값**이 됩니다.
그 값의 유일한 소비자는 "두 체크포인트가 같은 파일인가" 를 id 로 비교하는 텔레메트리
콜백(`serialization.py:2226`)이고, 그 질문에 **결정적으로 틀린 답**을 주는 것은 답을 주지
않는 것보다 나쁩니다. 없으면 양쪽 리더가 `""` 를 답합니다.

### 4.5 덤으로 얻은 것: 바이트 재현성

`_ZipWriter` 는 zip 항목의 타임스탬프를 1980-01-01 로 고정합니다(상류는 벽시계를 씁니다).
그래서 **같은 객체를 두 번 저장하면 바이트가 같습니다** — 상류의 `torch.save` 에는 없는
성질이고, 페더레이션 델타를 해시로 주소 지정할 수 있게 하는 것입니다(README §2).

---

## 5. 실측

### 5.1 상류가 읽는다 — 인수 기준

14 개 텐서, 두 목적지(경로 · `io.BytesIO`), 별도 프로세스의 상류 torch 2.13.0:

```
shim_path.pt: 14 tensors, all bit-exact = True
shim_buf.pt : 14 tensors, all bit-exact = True
shim_sd.pt  :  3 tensors, all bit-exact = True      (nn.Module 의 state_dict)
```

비교는 허용오차가 아니라 `torch.equal` 입니다 — 경로 전체가 바이트 복사이므로, 정확히
같지 않은 것은 반올림이 아니라 결함입니다.

dtype 커버리지: `float32` · `float16` · **`bfloat16`** · `float64` · `int64` · `int32` ·
`uint8` · `bool`, 그리고 스칼라 · 빈 텐서 · rank-3.

### 5.2 스토리지 공유가 살아남는다

shim 안에서 만든 `x`, `x.t()`, `x[1]`:

```
레코드            ['data/0']                    <- 셋이 아니라 하나
상류가 본 것       세 텐서의 storage.data_ptr() 이 전부 같음
                  tr.stride() == (1, 4)
                  row.storage_offset() == 4
                  tr == base.t(),  row == base[1]
```

### 5.3 그 밖에 지나가는 것

```
torch.load(mmap=True)      우리가 쓴 파일에 대해 동작
weights_only=True / False  둘 다
페이로드 정렬               17/17 레코드가 64 바이트 경계
두 번 저장                  바이트 동일
```

### 5.4 사보타주 — 각 테스트가 실제로 빨간지

구현을 국소적으로 망가뜨리고 다시 빌드해서 셌습니다(`cp` 백업, `git checkout` 은 쓰지
않았습니다).

| 넣은 결함 | 빨개진 것 |
|---|---|
| `origin` 을 떼어 identity 를 사본 주소로 | 1 — 공유 테스트, `['data/0','data/1','data/2']` 를 이름으로 지목 |
| `storage_offset()` 을 항상 `0` 으로 | 1 — 공유 테스트 (`row` 가 `[0,1,2,3]`, 참값 `[4,5,6,7]`) |
| `stride()` 를 shape 의 연속 stride 로 | 1 — 공유 테스트 (`tr` 이 전치되지 않음) |
| `untyped_storage()` 가 뷰를 실체화 | 6 — 전부 |
| 정렬 패딩 제거 (자체 검사 유지) | 6 — 자체 검사가 먼저 잡음 |
| 정렬 패딩 제거 + 자체 검사 제거 | 1 — 아카이브 테스트만. **상류는 그래도 읽음**(§4.2) |
| `version` 레코드 미기록 | 4 — 상류 리더 셋 + 아카이브 테스트. **shim 왕복은 초록**(§4.3) |

**이 표에서 읽어야 할 것 하나**: `storage_offset`/`stride` 사보타주가 **인수 테스트를
초록으로 남깁니다.** 이유는 그 테스트의 입력이 전부 shim 의 *로더*를 거쳐 오는데, 로더가
뷰를 실체화하기 때문입니다(CKPT.md §5) — 재저장 시점에는 offset 0 · 연속 stride 가 정답인
텐서들입니다. 뷰 충실도를 실제로 검사하는 것은 shim 안에서 뷰를 만드는 공유 테스트
하나뿐이고, 그 사실을 인수 테스트의 독스트링에 적어 두었습니다.

---

## 6. 여전히 거절하는 것, 그리고 각각의 이유

| | 왜 |
|---|---|
| `_use_new_zipfile_serialization=False` (legacy 컨테이너) | 쓸 수는 있으나 **이 빌드가 다시 읽을 수 없습니다.** legacy 는 `set_` **뒤에** 스토리지를 채우고 이 shim 의 `set_` 은 복사하므로, 그 형식으로 쓴 것은 0 으로 읽힙니다(CKPT.md §4). 거절은 `UntypedStorage._write_file` 에서 이름을 대고 나옵니다 |
| `torch.serialization.skip_data` | 페이로드 없는 레코드 헤더만 쓰는 경로. 0 으로 채우면 CKPT.md §4 가 한 절을 들여 기록한 그 파일 — 로드되고, 모든 키가 맞고, 전부 0 — 이 나옵니다. `write_record_metadata` 가 이름을 대고 거절 |
| snapshot 에 대한 쓰기 (`__setitem__` · `copy_` · `resize_` · `_shim_fill`) | §3. 거절이 이 설계를 정직하게 만드는 부분입니다 |
| `write_record(compress=True)` | torch 는 압축하지 않고 두 리더 모두 `ZIP_STORED` 를 가정합니다 |
| meta 텐서의 `untyped_storage()` | 바이트가 없습니다(META.md §3). `tensor()` 에서 이름을 대고 거절 |
| 양자화 텐서의 `untyped_storage()` | 블록은 torch 가 레코드에 이름 붙일 수 있는 어떤 dtype 의 평면 스토리지도 아닙니다 |
| `UntypedStorage.__getstate__` | 스토리지를 *객체로* 피클하는 것. `torch.save` 는 `persistent_id` 로 가로채므로 저장 경로가 아닙니다 |
| `.data/serialization_id` 레코드 | §4.4 — 쓰는 것이 안 쓰는 것보다 나쁩니다 |
| `TensorBase.stride()` / `storage_offset()` on meta | `Repr::Meta` 는 layout 을 들고 있지 않습니다(META.md §6). 답하면 지어내는 것입니다 |

### 6.1 거절이 아니라 **좁혀진 것** — 저장은 되지만 상류와 다른 것

| | 무엇이 다른가 |
|---|---|
| **로드 후 재저장은 스토리지 공유를 복원하지 못합니다** | shim 의 로더가 복사하므로(CKPT.md §5), 상류 파일에서 한 버퍼를 보던 두 텐서는 shim 을 거치면 두 버퍼가 됩니다. 재저장하면 레코드가 둘입니다. **값은 동일**하고, 달라지는 것은 파일 크기와 별칭 관계입니다. 묶인 임베딩(`lm_head.weight is embed_tokens.weight`)처럼 **한 객체가 두 키에 들어 있는** 경우는 피클의 memo 가 나르므로 그대로 보존됩니다 — 두 기제가 다르고, 테스트가 둘을 따로 단언합니다 |
| `compute_crc32=False` | 무시하고 진짜 CRC 를 씁니다. `zipfile` 은 항상 계산하고, 상류가 0 을 쓰는 자리에 올바른 값이 들어갑니다. 어떤 리더도 구별할 수 없습니다 |
| `_cdata` | 상류는 `THPStorage` 객체의 주소, 여기서는 **버퍼**의 주소입니다. §3.2 가 이유이고, 상류에서는 둘이 1:1 이므로 관측 가능한 차이가 없습니다 |
| 빈 스토리지의 `data_ptr()` | 상류는 `0`, 여기서는 진짜 주소입니다. 상류가 `0` 에 부여하는 의미(`serialization.py:1224` 의 dtype 충돌 검사를 건너뜀)는 빈 스토리지끼리 충돌하지 않게 하기 위한 것인데, 여기서는 `Arc` 마다 주소가 다르므로 그 충돌이 애초에 없습니다 |
| 타임스탬프 | 1980-01-01 고정. §4.5 |

---

## 7. 이것이 무엇을 열었나 — 페더레이션 이음매

`README.md` §2 는 페더레이션이 기다리는 것을 **셋**으로 적습니다.

> a second rank (`ProcessGroupLocal` refuses `world_size != 1`), a rendezvous (`TCPStore`
> refuses; `HashStore` is process-local), and **a delta on the wire** — `torch.save` refuses,
> so the local update cannot be written to bytes at all.

**셋째는 닫혔습니다.** 실측:

```
torch.save(delta.value) -> io.BytesIO        1871 B   ok
torch.save(delta)       -> io.BytesIO        2943 B   ok      (Delta 객체 통째로)
delta.persist(...)      -> safetensors                ok      (원래부터 되던 것)
```

**`Delta.publish` 의 다음 벽은 `torch.save` 가 아니라 두 번째 랭크입니다.**
`torchnative.distributed` 를 임포트해 `local` 백엔드를 등록하고 지시받은 검사를 그대로
돌리면:

```
Delta.publish()
  NotImplementedError: torchnative.delta: a delta cannot leave this device yet --
  aggregation needs a process group with more than one rank.
  Check: torch.distributed.init_process_group(backend='local', rank=0, world_size=2,
         store=torch.distributed.HashStore())

그 Check 를 실행하면:
  init_process_group(world_size=2)  ->  NotImplementedError:
      torch._C._distributed_c10d.ProcessGroupLocal: world_size 2 needs a transport,
      and this build has none. Only world_size 1 is implemented
  init_process_group(world_size=1)  ->  통과
```

즉 `Delta.publish` 의 거절은 **여전히 옳고, 여전히 자기 검사를 통과하지 못합니다** — 그리고
그 검사가 가리키는 곳은 `bootstrap.py` 의 `ProcessGroupLocal.__init__` 입니다. 다음 회차가
겨눌 이름은 **transport** 이고, 그다음이 rendezvous(`TCPStore`)입니다.

### 7.1 낡은 문장 셋

이 회차가 사실을 바꿔 놓은 곳입니다. **이 문서는 그것을 기록만 하고 고치지 않았습니다** —
`README.md` 와 `torchnative/delta` 는 지시받은 파일 범위 밖입니다.

| 어디 | 무엇이 낡았나 |
|---|---|
| `README.md` §2 | *"`torch.save` refuses, so the local update cannot be written to bytes at all"* — 더 이상 거절하지 않습니다 |
| `torchnative/delta/__init__.py` 모듈 독스트링 | *"``persist`` deliberately does **not** go through ``torch.save``. (...) its blocker is a storage object that aliases its tensor, which this stack does not have and cannot honestly fake."* — 그 blocker 는 별칭이 필요하지 않았습니다. 필요했던 것은 identity 를 나르는 사본과, 쓰기에 대한 거절이었습니다(§3). `persist` 가 safetensors 를 쓰는 것 자체는 여전히 합리적입니다(더 작고, 피클이 없습니다) — 낡은 것은 **이유**입니다 |
| `docs/training/BACKWARD.md` §14.2 · §14.3 | §1.2 와 §3 |

---

## 8. 회귀

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh    339 ok   (이전 333, +6)
                                              DOCWATCH: PASS -- 248/248
$PY tools/golden/compare.py                  7763/7763, ops=168
```

`_aten_dispatch` 에 추가된 op 은 없습니다 — 이 회차가 더한 것 중 aten op 은 하나도 없고,
따라서 `tools/golden/cases.py` 에 붙일 케이스도 없습니다.

위 표의 회차 고유 숫자가 아니라, **뒤에 와도 계속 참이어야 하는 것**에 대한 상시 검사
(docs/verification/DOCWATCH.md). `339` 은 이 회차의 스냅숏이므로 하한으로만 겁니다 — 이 저장소의
다른 문서가 `smoke_ok` 를 다루는 방식과 같습니다.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _ZipWriter present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/storage.rs snapshot present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs storage_snapshot present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_save_upstream_reads_every_dtype_and_view_the_shim_wrote_bit_for_bit present -->
<!-- DOCWATCH: count smoke_ok ge 339 -->

---

## 9. 모르는 것

- **큰 체크포인트로 재보지 않았습니다.** §4.2 의 zip64 산술은 2 GB 넘는 레코드에 대한
  것인데, 이 회차의 가장 큰 레코드는 수백 바이트입니다. 그 분기는 **코드로만 있고
  실행되지 않았습니다.**
- **`torch.save(model)`**(state_dict 가 아니라 모듈 자체)는 `Delta` 객체가 통과한 것으로
  미루어 동작할 가능성이 높지만, 별도로 재지 않았습니다.
- **`storage_snapshot` 의 identity 가 주소 재사용에 노출되는 창**이 있는지 — 저장 경로에서는
  피클되는 객체가 모든 텐서를 붙들고 있으므로 없다고 논증했지만, `untyped_storage()` 를
  직접 부르는 다른 호출자에 대해서는 논증하지 않았습니다.
- **`stride()`/`data_ptr()`/`storage_offset()` 이 새 공개 표면**이라, 벤더 트리의 다른 코드가
  이제 다른 분기를 타게 되었을 수 있습니다. 스위트와 골든에는 변화가 없었지만, 그 둘이
  닿지 않는 경로가 있는지는 확인하지 않았습니다.
- **다른 아키텍처**에서는 돌리지 않았습니다 — Apple Silicon 뿐입니다. 바이트 순서를 쓰는
  코드가 새로 생겼으므로(`storage_snapshot`), 빅엔디언 타깃이 생기면 재봐야 합니다.
  (`torch.save` 는 `byteorder` 레코드를 쓰고, `_save` 가 `sys.byteorder` 를 그대로 넣습니다.)
