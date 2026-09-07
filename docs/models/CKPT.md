# CKPT — 훈련된 가중치를 읽는다

> **§1 · §3.1 · §6 의 "못 한 것" 세 줄과 §6 의 "모르는 것" 첫 줄은 뒤에 나온 측정으로
> 바뀌었습니다 — `docs/models/CKPT2.md`.** `UntypedStorage.from_file` 과 저장소 슬라이싱이
> 구현되어, **`torch.load(mmap=True)` 와 safetensors 의 기본 `mmap` 백엔드가 둘 다
> 동작합니다.** 셋째 리더인 `mmap` 백엔드는 여기 §1 의 두 리더와 **`0.0`** 으로
> 일치하고, `torch.load` 의 mmap/비-mmap 두 경로도 21/21 텐서가 **`0.0`** 입니다.
> 이 문서의 회귀 테스트 중 그 성질을 단언하던 것은 "거절한다" 에서 "일치한다" 로
> 바뀌었습니다(같은 파일, 이름이 바뀐 테스트).
>
> **legacy 포맷의 거절과 `filled` 불변식은 그대로이고, 그것이 옳습니다** — §4 가 그
> 이유이며 테스트가 계속 단언합니다. `filled` 를 세울 수 있는 것이 `_shim_fill` 하나에서
> 둘로 늘었는데, 불변식의 문장은 원래 의도대로 **"바이트를 실제로 전달한 것만 세울 수
> 있다"** 로 다시 적혔습니다(CKPT2.md §2.4).
>
> **§6 "모르는 것" 의 첫 줄** — *실제 사전훈련 체크포인트로는 검증하지 못했습니다* —
> 은 닫혔습니다: 허브의 SmolLM2-135M(273 텐서 · 1.63억 파라미터)이 상류와 비트 단위로
> 일치합니다(CKPT2.md §7).

이 shim 은 **체크포인트를 한 번도 읽은 적이 없었습니다.** 지금까지의 모든 검증 — 골든 2095 케이스,
E2E 토큰 일치, 샘플링 90 토큰 — 이 전부 **난수로 초기화한 가중치**로 한 것입니다. 모델을 조립하는
것과 훈련된 모델을 불러오는 것은 다른 경로이고, 후자는 한 번도 실행된 적이 없었습니다.

이 문서는 그 경로를 재고, 어디까지 뚫었고, 어디서 멈췄는지를 적습니다.

---

## 1. 판정 요약

**상류가 저장한 체크포인트를 shim 이 읽어, 그 가중치로 순전파를 돌리고 상류와 대조한 결과:**

| 경로 | 결과 | 로짓 최대 절대오차 |
|---|---|---|
| `torch.load` (zip, `weights_only=True`) | **동작** | **2.98e-08** |
| `torch.load` (zip, `weights_only=False`) | **동작** | 2.98e-08 |
| `safetensors.torch.load(bytes)` | **동작** | 2.98e-08 |
| `safetensors.torch.load_file(backend="pread")` | **동작** | 2.98e-08 |
| `safetensors.torch.load_file()` — 기본 `mmap` 백엔드 | 거부 | — |
| `torch.load` (legacy 비-zip 포맷) | 거부 | — |
| `Module.state_dict()` / `load_state_dict()` | **동작** | — |

정상 범위는 2.3e-09~5.2e-06 이고, 2.98e-08 은 그 안입니다.

그리고 **두 리더가 같은 답을 냅니다** — 같은 가중치를 `torch.load` 로 읽은 것과 safetensors 로
읽은 것의 차이는 `0.0` 입니다. 근사가 아니라 비트 단위로 같습니다.

### 회귀 없음

```
골든              2095/2095, ops=91, pending 0            exit 0   (develop 기준과 동일)
--inject-fault    value / shape / dtype                   전부 exit 1
pytests run.sh    (자가검사 11×11 포함)                     exit 0
verify_schemas    204/204                                  exit 0
3 타깃            host / android-arm64 / ios-arm64          전부 exit 0
```

---

## 2. `636a3cc` 가 무엇을 했나

이 작업은 인터럽트로 죽은 에이전트의 커밋(`636a3cc`, 4 파일 269 줄)을 이어받은 것입니다. 보고가
없었으므로 먼저 판정했습니다. **결론: 그 커밋은 safetensors 경로를 끝까지 뚫어 놓았고, 코드는
옳습니다.** 다시 하지 않았습니다.

들어 있던 것 세 가지:

| 위치 | 무엇 | 판정 |
|---|---|---|
| `lib.rs` `_frombuffer` | `torch.frombuffer` 의 구현 | **옳음.** safetensors 로드의 전부 |
| `dtype.rs` | `uint1..7` · `int1..7` 14 개 태그 | **옳음.** 아래 설명 |
| `tensor.rs` `is_meta` | `load_state_dict` 가 파라미터마다 읽음 | **옳음** |
| `bootstrap.py` | `varfns.frombuffer` 배선 | **옳음** |

**14 개 서브바이트 dtype 이 왜 필요한가**를 재확인했습니다. 커밋 주석의 주장이 맞습니다 —
`torch/_weights_only_unpickler.py` 가 허용목록을 만들 때

```python
for t in [getattr(torch, f"uint{x}") for x in range(1, 8)]: ...
```

를 **체크포인트를 한 바이트도 읽기 전에** 무조건 실행하므로, 14 개 중 하나만 없어도 모든
`torch.load` 가 `AttributeError` 로 죽습니다. 이 문서의 §3 이 그 다음 벽을 재는 것은 이 14 개가
있었기 때문에 가능했습니다.

**주석에 적힌 미검증 주장 하나를 실측으로 확인했습니다.** `_frombuffer` 의 doc 은 "이 함수 하나로
safetensors 경로가 첫 벽에서 완전한 state dict 까지 간다" 고 적었는데, 근거가 남아 있지 않았습니다.
재측정 결과 **사실입니다** (§3 표 참조).

**그 커밋이 하지 않은 것**: `torch.load` 는 전혀 손대지 않았고, 어떤 경로도 테스트되지 않았으며,
`docs/models/CKPT.md` 를 네 군데에서 참조하는데 그 파일이 없었습니다(이 문서가 그것입니다).

---

## 3. 각 경로가 어디서 막혔나 — 실측

측정 방법: 각 경로를 독립된 `try/except` 로 돌려 **첫 벽이 다음 벽을 가리지 않게** 했습니다.
`vendor/probe.py` 의 record 모드와 같은 발상이되, 대상이 import 가 아니라 로드 경로입니다.

### 3.1 safetensors

```
load_file()                → torch._C.StorageBase.from_file      (mmap 백엔드)
load_file(backend="pread") → 통과
load(bytes)                → 통과
```

**기본 백엔드가 `mmap` 이고, 그것만 막힙니다.** `safe_open(..., backend="mmap")` 은 safetensors
자신의 Rust 확장 안에서 `torch._C.StorageBase.from_file` 로 mmap 스토리지를 만들고, 그것은
"텐서가 파일에 별칭으로 붙는다"는 개념이라 candle 에 대응물이 없습니다. `backend="pread"` 는
같은 컨테이너를 읽고 `torch.frombuffer` 로 나오므로 통과합니다.

**즉 safetensors 는 인자 하나로 오늘 동작합니다.** 이것이 이 조사에서 나온 가장 실용적인 결과입니다.

### 3.2 `torch.load`, zip 컨테이너 (기본)

벽을 하나씩 뚫으며 잰 **완전한 순서**입니다. 각 줄이 앞줄을 메운 뒤에야 보였습니다.

| # | 벽 | 어디서 | 메운 방법 |
|---|---|---|---|
| 1 | `PyTorchFileReader.get_all_records` | `serialization.py:2232` | `_ZipRecords` (bootstrap.py) |
| 2 | `TypedStorage(wrap_storage=...)` 가 진짜 `UntypedStorage` 를 요구 | `storage.py:836` | `StorageBase` (storage.rs) |
| 3 | `TensorBase.set_` | `_utils.py:198` | tensor.rs |
| 4 | `TensorBase.element_size` | `set_` 내부 | tensor.rs |
| 5 | `tensor._backward_hooks = ...` 에 setter 없음 | `_utils.py:246` | tensor.rs, 실슬롯 |
| 6 | `torch._C._check_sparse_tensor_invariants` | `_utils.py:290` | bootstrap.py, `False` |

여섯 개를 메우면 zip 포맷이 끝까지 갑니다. `weights_only` 는 `True`/`False` 둘 다 통과합니다.

### 3.3 `torch.load`, legacy 포맷 — **막혔고, 그것이 옳습니다**

`torch.save(..., _use_new_zipfile_serialization=False)` 로 저장한 것은 **거부합니다.** 이것은
못 뚫은 것이 아니라 **뚫으면 안 되는 것**이며, §4 가 그 이유입니다.

---

## 4. 이 조사에서 나온 제일 중요한 것 — 조용히 0 이 되는 경로

**측정된 사실.** legacy 포맷을 소박하게 구현하면 `torch.load` 가 성공하고,
`load_state_dict` 가 `<All keys matched successfully>` 를 돌려주고, **모든 가중치가 `0.0` 입니다.**
예외는 한 곳에서도 나지 않습니다.

```
LOADED legacy ['down.bias', 'down.weight', 'embed.weight', ...]
  every loaded tensor is all-zero: True
  down.bias[:4] = [0.0, 0.0, 0.0, 0.0]
  truth         = [0.06125, 0.14750, 0.23375, -0.18000]
```

**원인은 별칭과 복사의 차이이고, 버그가 아니라 구조입니다.**

상류에서 스토리지는 텐서 메모리의 *소유자*이고 텐서는 그 위의 뷰입니다 — `set_` 은 별칭을
만듭니다. candle 은 자기 메모리를 소유하므로 `set_` 이 **복사**할 수밖에 없습니다. 그런데 두
컨테이너 포맷은 스토리지를 채우는 시점이 `set_` 기준으로 반대입니다.

```
zip     스토리지를 레코드에서 채움  →  _rebuild_tensor → set_      복사해도 옳다
legacy  _rebuild_tensor → set_      →  스토리지를 파일에서 채움     복사하면 0 이 된다
```

별칭 의미론에서는 두 순서가 같은 답을 내므로 상류는 한 경로로 둘 다 처리합니다. 복사
의미론에서는 두 번째 순서가 **`set_` 시점의 스토리지 내용** — 즉 갓 할당된 0 — 을 집습니다.

### 규약이 아니라 불변식으로 막았다

이것을 "legacy 를 구현하지 않는다" 는 합의로 두면, 나중에 `_set_from_file` 이 사소해 보여서
구현되는 순간 되살아납니다. 그래서 **`StorageBase` 가 `filled` 를 들고 있고, `set_` 은 한 번도
채워진 적 없는 스토리지를 거부합니다.**

```
NotImplementedError: TensorBase.set_: the storage has never been filled. This shim's
set_ copies out of the storage instead of aliasing it, so a tensor built from an empty
storage would be silently zero. The caller must deliver the bytes before set_, not
after (see storage.rs and docs/models/CKPT.md §4).
```

`filled` 를 세우는 문은 `_shim_fill` 하나뿐이고, 그것은 바이트를 실제로 받은 경우에만 불립니다.
**legacy 순서를 우연히 타는 것이 불가능해집니다.**

---

## 5. 뷰로 저장된 텐서 — 두 번째 조용한 오답을 닫았다

체크포인트의 텐서가 자기 스토리지의 **연속 구간이라는 보장이 없습니다.** `torch.save` 는 발견한
그대로 stride 와 storage_offset 을 기록하므로, `state_dict` 가 `w.t()` 를 들고 있으면
비연속으로 도착합니다.

처음 구현은 앞에서부터 `numel` 개를 읽었고, 그러면 전치 텐서가 **dtype 도 shape 도 맞는데 숫자만
틀린 채로** 나옵니다. 역시 예외가 없습니다.

지금은 `gather_strided` 가 `(storage_offset, size, stride)` 를 실제로 걸어 행 우선으로 모읍니다.
바이트 단위 gather 라 dtype 과 무관하고, 근사가 아니라 정의 그대로입니다.

**실측 — 상류가 저장한 14 개 항목 전부 비트 단위 일치:**

```
  [ok ] transposed     torch.float32    stride=(1, 4)      worst=0
  [ok ] slice_offset   torch.float32    storage_offset=5   worst=0
  [ok ] tied_a/tied_b  가중치 공유(한 스토리지, 두 키)        worst=0
  [ok ] w_f16 / w_bf16 / w_f64 / buf_i64 / buf_i32 / buf_bool / scalar / empty / rank3
```

**남은 차이 하나**: 가중치 공유(weight tying)는 **값으로는 보존되지만 동일성으로는 보존되지
않습니다.** 상류에서 `tied_a is tied_b` 의 스토리지는 하나인데, 여기서는 복사이므로 둘이
독립입니다. 읽기 전용 추론에서는 보이지 않고, 한쪽을 제자리 수정하면 갈립니다.

---

## 6. 구현한 것 / 때운 것 / 못 한 것

### 구현한 것

| | 어디 | |
|---|---|---|
| `StorageBase` | `rust/torch_c/src/storage.rs` (신규) | 바이트 버퍼 + `filled` 불변식 |
| `TensorBase.set_` | `tensor.rs` | strided gather, 네 가지 거부 |
| `TensorBase.element_size` | `tensor.rs` | dtype 태그 기준 |
| `gather_strided` | `tensor.rs` | 뷰를 정의대로 읽음 |
| `from_le_bytes` | `tensor.rs` | `_frombuffer` 와 `set_` 의 **공통** 바이트→텐서 |
| `PyTorchFileReader` | `bootstrap.py` `_ZipRecords` | zip 컨테이너 |

`from_le_bytes` 를 공유로 뽑은 것은 정리가 아니라 판정 근거입니다 — 두 리더가 dtype 처리와
`torch.bool` 정규화를 따로 갖고 있으면 §1 의 "두 리더가 비트 단위로 같다" 가 우연이 됩니다.

**`_aten_dispatch` 는 그대로 단일 출입구입니다.** 여기 추가된 것 중 aten op 은 하나도 없고
(`_aten_implemented()` 는 91 개 그대로, 골든 pending 0), 따라서 `tools/golden/cases.py` 에
붙일 케이스도 없습니다.

### 때운 것 (papered over — 구현이 아님)

| | 무엇을 했나 | 왜 무해한가 |
|---|---|---|
| `_backward_hooks` | 저장하되 아무도 읽지 않는 실슬롯 | autograd 가 없어 훅이 발화할 일이 없고, 이 경로가 쓰는 값은 항상 빈 `OrderedDict()` |
| `_check_sparse_tensor_invariants` | 항상 `False` | 상류 기본값이 `False`, 그리고 이 shim 에 sparse 텐서가 없어 검사 대상이 공집합 |
| `_RecordHolder` | 상류는 Tensor 를 돌려주는데 여기서는 두 메서드만 답하는 객체 | 호출자가 `._typed_storage()._untyped_storage` 만 함. Tensor 를 흉내내면 체크포인트를 통째로 한 번 더 복사 |
| `PyTorchFileReader` 가 Rust 가 아니라 bootstrap.py | 컨테이너 파싱은 텐서 연산이 아니고, CPython 이 검증된 zip 리더를 이미 들고 있음 | 대안은 손으로 쓴 zip 파서나 3 타깃에 새 크레이트 의존성 — 둘 다 더 나쁜 거래 |

### 못 한 것 (거부하고 이름을 답함)

| | 무엇이 필요한가 |
|---|---|
| `torch.load` legacy 포맷 | 진짜 별칭 스토리지. §4 |
| ~~`torch.load(mmap=True)`~~ **정정 (문서 감사, 2026-09): 닫힘** | `UntypedStorage.from_file` + 스토리지 슬라이싱 — `docs/models/CKPT2.md` 가 구현, 가중치가 비트 단위로 상류와 일치 |
| ~~`safetensors` 기본 `mmap` 백엔드~~ **정정 (문서 감사, 2026-09): 닫힘** | 같은 것 — `docs/models/CKPT2.md` §7 이 SmolLM2-135M 273 텐서 전부 비트 일치를 확인 |

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/storage.rs from_file present -->
| `get_record_offset_no_read` | torch 의 레코드 정렬 산술 재현. **틀린 오프셋은 예외가 아니라 옆 텐서의 바이트**라 추측하지 않음 |
| `int8` · `uint16` · `uint64` · complex 로 저장된 체크포인트 | candle 이 못 담음. dtype 이름을 대며 거부 |
| 음수 stride | torch 가 만들지 않으므로 추측하지 않음 |
| 체크포인트 **쓰기** | 범위 밖. `UntypedStorage.__getstate__` 가 이름을 대며 거부 |

### 모르는 것

- ~~**실제 사전훈련 체크포인트로는 검증하지 못했습니다.**~~ **정정 (문서 감사, 2026-09):
  닫힘 — `docs/models/CKPT2.md` §7 자신이 이 정확한 문장을 인용하며 닫혔다고 적습니다** ("`docs/models/CKPT.md`
  §6 '모르는 것' 의 첫 줄 ... 이 닫혔습니다"). `import transformers` 도 그 뒤 `docs/distributed/DISTRIBUTED.md`
  가 열었습니다(이 감사가 round 2 에서 이미 확인).
  `import transformers` 가 아직
  `torch.distributed` 에서 막히고(IMPORT_WALLS), 이 조사는 `torch.nn` 조립 + 수동 적재로
  판정했습니다. 여기서 쓴 것은 상류가 저장한 진짜 `.pt`/`.safetensors` 파일이지, 진짜 *모델* 은
  아닙니다. HF 체크포인트 특유의 것(공유 텐서 메타데이터, 샤딩된 `.index.json`, `_metadata`)은
  **미측정**입니다.
- **회귀 스위트에 박혀 있지 않습니다.** 위 숫자는 전부 `/Volumes/macMini/caches/ckpt-probe/`
  의 스크립트로 잰 것이고, 커밋 대상이 아닙니다(이 작업의 파일 범위가 `rust/torch_c/src/`,
  `tools/golden/cases.py`, 이 문서였습니다). **`pytests/test_shim.py` 에 넣는 것이 다음
  작업이고, 넣기 전까지 §1 의 어떤 성질도 회귀로부터 보호되지 않습니다.** docs/models/E2E.md 가 같은
  이유로 만들어졌던 자리입니다.
- `serialization_id()` 는 레코드가 없으면 빈 문자열을 답합니다. 상류가 그 값을 어떻게 쓰는지는
  텔레메트리 콜백 하나밖에 확인하지 못했습니다.

---

## 7. 재현 방법

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-ckpt
cd /path/to/repo
bash vendor/vendor_torch.sh
sh vendor/install_shim.sh

# 상류가 체크포인트를 만든다 (벤더 트리를 PYTHONPATH 에 넣지 않는다)
/Volumes/macMini/caches/spike-venv/bin/python make_ckpt.py

# shim 이 그것을 읽는다
TORCH_USE_RTLD_GLOBAL=1 PYTHONPATH=$PWD/torchnative/src/main \
    /Volumes/macMini/caches/spike-venv/bin/python verify.py
```

`TORCH_USE_RTLD_GLOBAL=1` 은 회피가 아니라 상류가 그 목적으로 둔 스위치입니다 —
`libtorch_global_deps` 가 없는 빌드 환경을 위한 것이고, 우리가 정확히 그 환경입니다
(VENDOR.md:181).

**`compare.py` / `verify_schemas.py` 는 벤더 트리를 `PYTHONPATH` 에 넣지 말고 돌립니다.**
그 둘은 상류 torch 와 대조하는 쪽이라 진짜 torch 가 보여야 합니다.

---

## 8. 회귀 스위트에 박아 넣기

§1-7 의 모든 숫자는 `/Volumes/macMini/caches/ckpt-probe/` 아래의, 커밋 대상이 아닌 스크립트
(`make_ckpt.py`/`verify.py`/`make_hard.py`/`verify_hard.py`)로 잰 것이었다. worktree 가 정리되면
그 스크립트도 함께 사라지고, 그때까지는 §1 의 어떤 성질도 회귀로부터 보호되지 않았다. 이 절은
그것을 `rust/torch_c/pytests/test_shim.py` 의 다섯 테스트로 옮겨 박은 기록이다. `docs/models/E2E.md` 가
샘플링 경로에 대해 이미 한 일과 같은 종류의 작업이다.

### 8.1 추가한 다섯 테스트와 각각이 잡는 것

| 테스트 | 잡는 것 |
|---|---|
| `test_ckpt_torch_load_zip_round_trip_matches_upstream_within_measured_tolerance` | zip 컨테이너(`weights_only=True`/`False` 둘 다), 키·shape·dtype 일치, 로짓 회귀 |
| `test_ckpt_safetensors_two_readers_agree_with_torch_load_bit_for_bit` | `load(bytes)`/`load_file(pread)` 로짓 회귀, 그리고 §1 이 주장하는 "두 리더가 정확히 0.0 으로 일치"가 우연이 아니라 `from_le_bytes` 공유(§6) 덕분이라는 것 |
| `test_ckpt_legacy_format_and_mmap_backend_are_refused_by_name` | legacy `torch.load` 와 safetensors `mmap` 백엔드, 이 두 미구현 경로가 계속 **거부**하는지 (§3.3, §6 "못 한 것") |
| `test_ckpt_filled_guard_refuses_set_on_unfilled_storage_then_gathers_strided_views` | **가장 중요한 안전장치**(§4) — `filled` 가드 자체, 그리고 채워진 뒤의 strided gather 세 가지(연속/전치/오프셋, §5)와 두 거부(범위 밖, 음수 stride) |
| `test_ckpt_fourteen_hard_dtypes_and_views_round_trip_bit_exact` | §5 의 14 개 항목(f16/bf16/f64/i64/i32/bool/스칼라/빈 텐서/rank-3, 가중치 공유, 전치 뷰, 0 아닌 오프셋) 전부 비트 일치, 그리고 가중치 공유의 값 보존 |

### 8.2 왜 한 프로세스가 아니라 두 인터프리터인가

`docs/models/E2E.md` 의 세 테스트는 "상류와 shim 을 한 프로세스에서 쓴다"는 것을 보였다 — 단, 그 트릭은
`_C` 를 `torch._C` 가 아니라 독립 모듈 `_C` 로 파일 경로째 로드하고, `torch.ops.aten.*` 를
`_aten_dispatch` 로 직접 부르는 한에서만 성립한다.

이번에는 그 트릭을 못 썼다. `torch.load`/`nn.Module`/`state_dict` 는 **순수 파이썬 torch**
(`torch/serialization.py`, `torch/nn/modules.py`, ...) 안에 있고, 그 코드가 셰임을 쓰게 하려면
벤더 트리(`torchnative/src/main/torch`, `_C.abi3.so` 가 이미 심어져 있는 그 패키지)를 **`torch`
라는 이름으로** import 해야 한다 — 상류 `torch` 와 이름이 같다. 한 인터프리터에서 `torch` 라는
이름은 하나뿐이고, 게다가 그렇게 하려면 이미 독립 모듈로 한 번 로드해 둔 셰임 네이티브 라이브러리를
**다른 경로에서 두 번째로 `dlopen`** 하게 되는데, 이게 안전한지는 이 작업에서도 이전 어디에서도
측정된 적이 없다. 그래서 이번에는 `docs/models/CKPT.md` §7 이 이미 검증해 둔 **두 스크립트 레시피**
(`make_ckpt.py` 로 만들고 `verify.py` 로 읽는다) 를 그대로 서브프로세스로 접었다:

1. 이 프로세스(`test_shim.py`, 상류 `torch` 가 `_upstream_torch`) 가 결정적 가중치로 `Tiny` 를
   조립하고 `torch.save`(zip·legacy)/`safetensors.save_file` 로 세 파일을 쓴다.
2. `subprocess.run([sys.executable, "-c", ...], env={"PYTHONPATH": <벤더 디렉터리>, ...})` 로
   **완전히 새로운 인터프리터**를 띄운다 — 그 프로세스의 `import torch` 는 벤더 트리(셰임)다.
3. 그 서브프로세스가 §1-6 의 모든 검사를 돌리고 결과를 JSON 한 덩어리로 stdout 에 찍는다.
4. 이 프로세스가 JSON 을 파싱해 assert 한다.

서브프로세스 실행은 `functools.lru_cache` 로 한 번만 한다 — 다섯 `test_ckpt_*` 함수가 전부 같은
결과를 나눠 쓴다. 실패 시에는 `lru_cache` 가 예외를 캐싱하지 않으므로 각 테스트가 독립적으로
재실행하고 독립적으로 빨갛게 보고한다(§8.4).

### 8.3 새 전제조건 — `pytests/run.sh` 가 보장하지 않는 것

`pytests/run.sh` 는 독립 모듈 `_C.abi3.so` 만 스테이징한다. 이 다섯 테스트가 필요로 하는
`torchnative/src/main/torch/_C.abi3.so` (벤더 트리 안에 심어진 셰임)는 **별도로**
`vendor/vendor_torch.sh` + `vendor/install_shim.sh` 를 돌려야 생긴다 — `run.sh` 자신은 이 경로를
전혀 건드리지 않는다. 그래서 다섯 테스트 모두 `_upstream_torch is None` 뿐 아니라
`torchnative/src/main/torch/_C.abi3.so` 존재 여부도 같이 확인하고(`_ckpt_shim_available()`), 둘
중 하나라도 없으면 `docs/models/E2E.md` 와 같은 이유로 조용히 통과한다(`pytest.skip` 을 쓰지 않는 이유도
같다 — 이 파일은 pytest 에 의존하지 않는다).

### 8.4 각 테스트가 실제로 빨간지 확인한 방법

구현(`rust/torch_c/src/`)은 건드리지 않았다 — 이 작업의 파일 범위 밖이다. 대신 매 테스트마다
`test_shim.py` 안의 **기대값**을 하나씩 흔들어 다시 돌리고, `FAIL` 을 직접 본 뒤 원본 사본
(`cp` 로 떠 둠)과 `diff` 로 바이트 단위 원상복구를 확인했다. 다섯 개 전부:

| 테스트 | 흔든 것 | 결과 |
|---|---|---|
| zip round trip | 통과 기준을 `1e-5` 대신 `1e-30` 으로 | `FAIL ...: AssertionError: 2.9802322387695312e-08` |
| safetensors 일치 | `reader_agreement_worst == 0.0` 을 `== 12345.0` 으로 | `FAIL ...: AssertionError: 0.0` |
| legacy/mmap 거부 | `assert r["legacy_refused"]` 를 `assert not ...` 로 반전 | `FAIL ...: AssertionError: flipped for verification` |
| filled 가드 — 가드 자체 | `assert r["unfilled_refused"]` 를 `assert not ...` 로 반전 | `FAIL ...: AssertionError: flipped for verification` |
| filled 가드 — strided gather | 연속 읽기 기대값을 `[0,1,2,3]` 대신 `[9,9,9,9]` 로 | `FAIL ...: AssertionError: [0.0, 1.0, 2.0, 3.0]` |
| 14 개 어려운 케이스 | 비트 일치 기준(`== 0.0`)을 `== 999.0` 으로 | `FAIL ...: AssertionError: ('w_f32', 0.0)` |

마지막에 `diff /tmp/test_shim.py.orig rust/torch_c/pytests/test_shim.py` 로 완전히 동일함을,
`git status --short` 로 `rust/torch_c/pytests/test_shim.py` 와 `docs/models/CKPT.md` 두 파일 외에는
아무것도 바뀌지 않았음을 확인했다.

**filled 가드에 대해 못 한 것.** 지시받은 것은 "가드를 우회했을 때 실제로 0.0 이 나오는 것을
보여라"였다. 위 표의 "가드 자체" 행은 **테스트가 그 성질을 실제로 검사한다**는 것(반전하면 빨갛게
됨)을 보이지만, "가드가 아예 없었다면 0.0 이 나온다"는 것 자체를 이번 세션에서 다시 재현하지는
못했다 — `set_` 이 그 확인이 걸리는 유일한 문이라(§4, storage.rs 주석: "the one door"), 소스를
고치지 않고는 우회할 방법이 없었다. `git stash`로 가드를 빼고 다시 빌드하는 것도 고려했지만
지시받은 파일 범위(`rust/torch_c/src/` 제외)를 넘는 일이라 하지 않았다. 대신 이미 §4 가 기록해
둔, 가드가 생기기 전에 실측된 값(`down.bias[:4] = [0.0, 0.0, 0.0, 0.0]`, 참값은
`[0.06125, 0.14750, 0.23375, -0.18000]`)을 그대로 근거로 남긴다. **이 부분은 재현이 아니라
인용이다.**

### 8.5 허용오차 근거

- **로짓 비교(`torch.load` zip, safetensors 두 경로)**: `pytests/test_shim.py` 에 이미 있는
  `_E2E_LOGIT_ATOL = 1e-5` 를 그대로 재사용했다. 새 상수를 만들지 않은 이유는 근거가 이미 같기
  때문이다 — §1 의 표가 스스로 적은 정상 범위(`2.3e-09~5.2e-06`)가 `_E2E_LOGIT_ATOL` 정의부
  주석이 `do_sample`/`greedy` 측정에서 뽑은 범위와 **동일**하고, 오늘 이 세션에서 재측정한
  `2.98e-08`(§1 표와 정확히 일치)도 그 안에 있다.
- **safetensors 두 리더의 상호 일치, 14 개 어려운 케이스**: `== 0.0` 비트 정확 일치. §1·§5 가
  스스로 "근사가 아니라 비트 단위" 라고 적었고, 오늘 재측정도 정확히 `0.0` 이었으므로 느슨하게
  잡을 이유가 없다 — 느슨하게 잡으면 두 리더가 실제로 갈리기 시작해도 통과한다.
  `float16`/`bfloat16` 은 `float32` 를 경유해도 표현 가능한 값들이라(§5 원본 데이터가
  `float32` det() 을 캐스팅한 것) 정확 일치가 성립하는 것이 맞다.
- **음성 대조(negative control)**: 무작위 초기화가 목표 로짓과 실제로 멀다는 것(`> 1e-3`)과,
  safetensors 페이로드를 1.0 만큼 흔들면 로짓이 실제로 움직인다는 것(`> 1e-3`)은 §1 의 원본
  프로브가 쓰던 문턱을 그대로 가져왔다 — 이 값 자체가 "정확히 얼마나 커야 하는가"의 근거는
  아니고, "로딩 전/후가 우연히 같지 않다"를 확인하는 스모크 문턱이다.

### 8.6 스위트 실행 시간 변화

`PYTHONPATH=<stage> python3 pytests/test_shim.py` 단독 실행, `spike-venv` 인터프리터:

```
이전 (테스트 65개)   2.09s
이후 (테스트 70개)   2.86s   (+0.77s)
```

증가분은 대부분 상류 `torch.save`/`safetensors.save_file`(이 프로세스, 1 회) +
서브프로세스 기동 및 그 안에서의 `import torch`(벤더+셰임)/`import safetensors`(1 회) 비용이다
— `functools.lru_cache` 덕분에 다섯 테스트가 그 비용을 한 번만 낸다. `pytests/run.sh` 전체(cargo
증분 빌드 + golden 자가검사 포함)는 약 5 초로, "몇 분씩" 걸리는 영역과는 여전히 자릿수가 다르다.

### 8.7 넣지 않은 것과 이유

- **`torch.load(mmap=True)`, 3rd-party 실제 사전훈련 체크포인트**: §6 이 이미 "못 한 것"/"모르는
  것"으로 적어 둔 범위이고, 이번 작업은 회귀 고정이 목적이라 새 범위를 열지 않았다.
  `int8`/`uint16`/`uint64`/complex dtype 도 같은 이유로 넣지 않았다.
- **safetensors mmap 백엔드가 던지는 예외의 정확한 타입 재확인**: 오늘 재측정으로
  `NotImplementedError`(메시지 없음)임을 확인했고 테스트도 그 타입만 잡는다. 이 예외가 셰임
  자체에서 나는지(`StorageBase` 에 `from_file` 이 아예 없어 `AttributeError` 가 나고 safetensors
  Rust 확장이 그것을 감싸는지) 정확한 경로는 추적하지 않았다 — 타입과 "거부한다"는 사실만
  검증 대상으로 삼았다.
- **가중치 공유(tying)의 동일성**: §5 가 이미 "값은 보존, 동일성은 보존 안 됨" 으로 기록해 둔
  것을 값 쪽만 고정했다(`tied_equal`). 동일성 쪽은 이 shim 의 알려진 설계 한계이지 회귀가 아니므로
  고정할 대상이 아니다.

### 8.8 모르는 것

- §8.4 에 적은 대로, filled 가드가 없었다면 실제로 0.0 이 나온다는 것을 **이번 세션에서 직접
  재현하지는 못했다** — §4 의 기존 기록을 인용했을 뿐이다.
- `docs/models/E2E.md` §3 이 이미 적은 것과 같은 한계가 여기에도 그대로 적용된다: 상류 torch 가 없는
  인터프리터, 그리고 벤더 셰임이 설치되지 않은 환경에서는 이 다섯 테스트가 **아무것도 검증하지
  않고 조용히 통과**한다. `pytests/run.sh` 만 돌리는 환경(벤더 트리 설치 없이)이 실제로 있는지는
  확인하지 않았다.
- 다른 아키텍처(AVX2/VSX)에서 서브프로세스 접근 자체나 `_E2E_LOGIT_ATOL` 재사용이 여전히
  유효한지는 §8 이 새로 확인하지 않았다 — `docs/models/E2E.md` §8 이 이미 같은 범위를 미확인으로 남겨
  둔 것과 동일하게, 이번 세션도 Apple Silicon 에서만 실행했다.
