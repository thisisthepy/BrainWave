# 스키마 텍스트 — `_schema` 가 진짜로 읽히게 만든다

## 0. 이 문서가 답하는 것

- **`is_mutable` 이 고쳐졌는가.** 고쳐졌습니다. 구현된 117 개 전부가 상류와 일치하고,
  그중 **12 개가 참**입니다 (docs/distributed/DISTRIBUTED.md §8.1 이 지목한 일곱 개를 포함합니다). §4
- **텍스트를 어디서 가져왔는가.** 벤더 트리의 `torchgen/packaged/ATen/native/native_functions.yaml`
  입니다. **손으로 옮겨 적은 것은 18 개**이고, 그 18 개가 왜 옮겨 적을 수밖에 없는지에 이름이
  붙어 있습니다. §2 · §5
- **몇 개가 진짜가 됐고 몇 개가 아직 자리표시자인가.** 상류의 aten 스키마 3754 개 중
  **2606 개**를 답할 수 있고 **1148 개**는 여전히 자리표시자입니다. §6
- **`_schema` 를 읽는 것 중 무엇이 이제 옳은 답을 받는가.** §3 이 그 목록이고, 그것이 이
  작업의 범위였습니다.
- **아직 거짓말하는 것.** §7. **84 개의 술어가 텍스트 없이 답하고 있습니다.** 전부 상류에
  없는 op 에 대한 것이고, 그것이 사실인지는 `verify_schemas.py` 가 매번 다시 확인합니다.
- **한 번 잘못 고쳤다가 되돌린 것 둘.** §8. 둘 다 측정이 되돌리게 했습니다.

바뀐 파일은 셋입니다 — `rust/torch_c/src/bootstrap.py`,
`rust/torch_c/pytests/test_shim.py` (테스트 9 개 추가),
`rust/torch_c/pytests/verify_schemas.py` (검사 4 개 추가).
`overloads.json` 과 `methods.json` 은 **한 글자도 바뀌지 않았습니다** — 이 작업에서 그 둘은
고칠 대상이 아니라 **오라클**이었습니다.

---

## 1. 문제 — 술어가 틀릴 수 없었다

docs/distributed/DISTRIBUTED.md §8.1:

```
상류   add.Tensor  is_mutable = False      add_.Tensor is_mutable = True
우리   add.Tensor  is_mutable = False      add_.Tensor is_mutable = False   ← 틀림
```

`is_mutable` 의 구현은 상류 규칙 그대로였습니다
(`any(a.alias_info.is_write for a in self.arguments)`). **입력이 없었습니다.**
`_get_schema` 가 모든 op 에 대해 인자도 반환도 없는 `_Schema` 를 돌려주었고,
`any([])` 는 거짓이므로 **모든 op 이 "변경하지 않는다"** 였습니다.

그 전 판본은 `is_mutable` 이 프로퍼티가 아니라 메서드였고, 바인드된 메서드는 참이므로
**모든 op 이 "변경한다"** 였습니다. **같은 결함이 방향만 바꿔 두 번 났습니다.** 두 번 다
"텍스트가 없다" 가 원인이었고 술어 자체는 옳았습니다.

그리고 이것은 술어 하나의 문제가 아니었습니다. 벤더 트리에서 `_schema` 를 읽는 곳:

| 읽는 것 | 개수 | 자리표시자가 주던 답 |
|---|---|---|
| `._schema.arguments` | 80 | 빈 리스트 |
| `._schema.returns` | 43 | 빈 리스트 |
| `._schema.name` | 31 | (이것만 옳았습니다) |
| `._schema.is_mutable` | 18 | 항상 거짓 |
| `._schema.overload_name` | 7 | (옳았습니다) |
| `._schema._is_view_op()` | 2 | 항상 거짓 |

`arguments` 를 읽는 80 곳이 이 작업의 진짜 범위입니다. `is_mutable` 은 그중
**측정 가능한 판정 하나**였을 뿐입니다.

---

## 2. 텍스트의 출처 — 옮겨 적지 않는다

### 벤더 트리에 이미 있습니다

`torchgen/packaged/ATen/native/native_functions.yaml` — **`- func:` 항목 2584 개**, 각각이
aten 스키마 한 줄입니다. 상류가 코드젠에 쓰는 그 파일이고, `pyproject.toml` 이 데이터 파일로
휠에 넣습니다 (docs/platform/WHEEL.md). **런타임에 상류 torch 를 요구하지 않습니다.**

같은 트리를 같은 방식으로 이미 읽고 있습니다 — `torchnative/export/decompose.py` 가 Core ATen
태그 집합을 여기서 읽습니다 (docs/graph/DECOMP.md §2). 그 파일의 두 가지 결정을 그대로 따랐습니다:

- **`torch.__file__` 기준으로 찾습니다.** `torch/` 와 `torchgen/` 은 설치된 휠에서도 소스
  트리에서도 형제이고, 경로에 트리가 둘 있으면 답은 **지금 op 을 묻고 있는 쪽**에서 나와야
  합니다. `_C.__file__`, 그다음 `sys.path` 훑기가 대안 경로입니다.
- **YAML 파서가 아니라 줄 스캐너입니다.** `pyyaml` 은 이 배포판의 선언된 의존성이 아니므로
  `import yaml` 은 올바르게 설치된 휠에서 `import torch` 를 깨뜨립니다.

파일을 못 찾으면 예외를 던지지 않습니다. `_C._shim_schema_source()` 가 **경로 대신 이유**를
돌려주고, 그 프로세스의 aten 스키마는 전부 자리표시자가 됩니다.

### 비용

색인은 15789 줄 한 번 훑기이고 파싱을 하지 않습니다. **정규화와 파싱은 op 단위로 지연**되므로
스키마 열두 개를 묻는 프로세스는 열두 개 값만 냅니다.

---

## 3. 상류가 인쇄하는 것은 파일에 적힌 것과 다르다

`str(FunctionSchema)` 는 C++ 인쇄기이고, **기본값을 다시 쓴 다음** 내보냅니다. 2584 개를
torch 2.13.0 의 `_jit_get_all_schemas()` 와 대조하면 **165 개가 다릅니다.** 다섯 가지뿐이고,
다섯 규칙을 넣으면 **잔차가 0/2584** 입니다.

| # | 규칙 | 파일 | 상류 |
|---|---|---|---|
| 1 | `DeviceIndex` 는 `int` 로 인쇄된다 | `DeviceIndex device_index` | `int device_index` |
| 2 | `float` 기본값은 C++ double 인쇄기를 통과한다 | `float std=1` | `float std=1.` |
| 3 | 문자열 기본값은 큰따옴표, `'` `"` `\` 를 모두 이스케이프 | `str a='none'` | `str a="none"` |
| 4 | 크기 있는 리스트 기본값은 펼쳐진다 — **`int[N>1]` 은 예외** | `SymInt[2] stride=1` | `[1, 1]` |
| 5 | 열거 기본값은 정수로 인쇄된다 | `ScalarType? dtype=long` | `dtype=4` |

**규칙 2** 는 `torch::jit` 이 double 에 쓰는 인쇄기 그대로입니다: 유한하고 `1e10` 미만이며
정수인 값은 `정수 + "."` 로(`1.`, `0.`, 음의 0 은 `-0.`), 나머지는 `max_digits10 == 17` 로.
상류가 `1/3` 을 `0.33333333333333331` 로 쓰는 이유입니다. `Scalar` 형에는 **리터럴이 소수로
적혀 있을 때만** 적용됩니다 — `Scalar alpha=1` 은 int IValue 이고 `1` 로 인쇄됩니다.

**규칙 4 의 예외가 이 표에서 가장 미묘합니다.** `SymInt[2] stride=1` 은 `[1, 1]` 이 되는데
`int[2] padding=0` 은 `0` 으로 남습니다. 상류 인쇄기가 `int` 리스트에 한해 **길이 2 이상이고
원소가 모두 같으면 스칼라로 되접기** 때문입니다 ("we want to faithfully replicate the schema
string"). 그래서 `int[1] padding=0` 은 되접기가 걸리지 않아 `[0]` 으로 인쇄됩니다. 추론이 아니라
측정입니다 — 파일 전체에서 이 모양의 인자 101 개가 정확히 그 선을 따라 갈립니다:

```
SymInt[1] 9 개 · SymInt[2] 28 개 · SymInt[3] 27 개 · int[1] 7 개   → 리스트로 인쇄
int[2] 16 개 · int[3] 14 개                                        → 스칼라로 남음
```

**규칙 5 의 표는 세 항목이 전부입니다.** 파일 전체에서 `None`/`True`/`False` 가 아닌 단어형
기본값은 `Mean`(33 회) · `long`(8 회) · `contiguous_format`(4 회) 뿐이고, 값은 각각 C++
열거자 값 `at::Reduction::Mean == 1`, `c10::ScalarType::Long == 4`,
`c10::MemoryFormat::Contiguous == 0` 입니다.

---

## 4. 판정 — `is_mutable`

`verify_schemas.py` 가 벤더 트리를 별도 프로세스로 띄워 상류와 대조합니다:

```
shim _schema text and is_mutable: 117/117 matched upstream
```

**텍스트와 술어를 따로 확인합니다.** 술어를 텍스트에서 따라 나오는 것으로 두지 않는 이유는,
이 술어가 낸 두 번의 실패가 **둘 다 텍스트에서 보이지 않았기** 때문입니다.

구현된 117 개 중 참인 것 **12 개**, 상류와 정확히 일치합니다:

```
add_.Tensor   clamp_.default   copy_.default        div_.Tensor
fill_.Scalar  fill_.Tensor     index_put_.default   masked_fill_.Scalar
normal_.default  relu_.default  uniform_.default    zero_.default
```

**§8.1 은 일곱 개를 지목했고 답은 열둘입니다.** §8.1 이 쓰일 때 구현 집합이 97 개였고 지금은
117 개입니다. 그 차이가 §8.1 의 요점을 다시 말해줍니다 — **틀리는 방향이 "변경하지 않는다"**
였으므로, op 집합이 커지는 동안 거짓말도 같이 커졌고 아무것도 시끄러워지지 않았습니다.
§8.1 이 이름을 댄 일곱 개는 `_SECTION_8_1_MUTABLE` 로 따로 고정해 두었습니다.

`verify_schemas.py` 는 **술어가 두 값을 다 가지는지**도 확인합니다. 상류와 op 마다 일치하면서
동시에 전부 같은 값인 실행은 위 루프를 통과하면서 결함을 재현할 수 있기 때문입니다.

---

## 5. 층 — 무엇이 먼저 답하는가, 그리고 순서가 왜 문제인가

`_get_schema` 는 네 곳을 순서대로 봅니다. `_C._shim_schema_provenance(qualname, overload)` 가
**어디가 답했는지**를 돌려줍니다.

| # | 이름 | 무엇 | 개수 |
|---|---|---|---|
| 1 | `registered` | `_c10d_functional` 계열 + 생성된 aten 스키마 + 트리의 `Library.define()` | 22 + 18 + 런타임 |
| 2 | `native_functions.yaml` | 파일이 선언한 것, 재인쇄 | 2584 |
| 3 | `tables` | `overloads.json` · `methods.json` | 4 (나머지 169 는 2 가 답함) |
| 4 | `placeholder` | 텍스트 없음. 인자·반환이 비고, 묻는 술어를 전부 기록 | |

### 2 와 3 의 순서가 load-bearing 입니다

처음 동작한 판본은 **3 을 2 보다 먼저** 보았습니다. 결과가 같아 보였고 실제로 같았습니다 —
두 곳이 겹치는 169 개에서 텍스트가 일치하므로. **그런데 그 순서에서는 재인쇄기를 검사하던
테스트가 오라클을 자기 자신과 비교하고 있었습니다.**
`test_schema_text_survives_the_round_trip_through_the_transcribed_tables` 는
`overloads.json`/`methods.json` 을 정답으로 놓고 shim 의 텍스트와 맞춰보는데, 표가 먼저
답하니 그 173 개 조회를 표 자신이 답했습니다. **부동소수 인쇄기를 통째로 지워도 초록이었습니다**
(실측). 순서를 뒤집고, 테스트가 **출처까지** 단언하도록 고쳤습니다 — 파일이 선언한 169 개는
`native_functions.yaml` 에서 와야 하고, 올 수 없는 4 개는 이름이 적혀 있습니다
(`div.Scalar_out`, `div.Scalar_mode_out`, `embedding.out`, `empty_like.out` — torchgen 이
생성하는 `.out` 변형이라 파일에 없습니다).

### 층 1 의 18 개는 왜 손으로 옮겨 적었는가

파일은 2584 개를 선언하고 상류 레지스트리에는 aten 스키마가 3754 개 있습니다. 차이는
`torchgen/native_function_generation.py` 가 **빌드 타임에 생성**하는 `.out`·functional·mutable
변형입니다. 그 생성기는 벤더링되어 있지만 **여기서 돌릴 수 없습니다** — 입력이 파싱된
`NativeFunction` 이고 파싱에 `pyyaml` 이 듭니다.

그래서 옮겨 적은 것은 **생성된 절반 전체가 아니라 트리가 실제로 질문하는 부분**이고, 그것을
추측이 아니라 **계측해서** 정했습니다 (§8.2). 18 개이고, `verify_schemas.py` 가
`_NON_ATEN_SCHEMA_TEXT` 와 똑같이 상류와 대조합니다 — **양방향으로**: 상류에 있어야 하고,
**파일에 없어야** 합니다. 파일에 있는 항목은 파일을 가리는 죽은 무게가 됩니다.

18 개 중 하나가 이 표를 선택 사항이 아니게 만듭니다:
**`aten::native_dropout_backward.out` 은 상류에서 `is_mutable = True`** 이고 자리표시자는
거짓을 답했습니다. 나머지 17 개는 오늘 자리표시자의 답과 우연히 일치하며, **"오늘 일치한다"에
기대지 않기 위해** 함께 넣었습니다.

---

## 6. 채운 것과 못 채운 것

| | 개수 | 출처 |
|---|---|---|
| 파일이 선언한 aten 스키마 | **2584** | `native_functions.yaml`, 재인쇄 (2584/2584 상류 일치) |
| 생성된 aten 스키마 중 옮겨 적은 것 | **18** | `_GENERATED_ATEN_SCHEMA_TEXT` (18/18 상류 일치) |
| 표에만 있는 `.out` 변형 | **4** | `overloads.json` · `methods.json` |
| — 합계 답할 수 있는 aten 오버로드 | **2606** | |
| **상류에 있으나 답할 수 없는 aten 오버로드** | **1148** | 물으면 자리표시자입니다 |
| 비-aten (`_c10d_functional` 계열) | 22 | `_NON_ATEN_SCHEMA_TEXT` (22/22 상류 일치) |

**구현된 117 개는 117/117 입니다.** 자리표시자가 하나도 없고, `torch.ops.aten.<op>.<ov>._schema`
로 가는 길과 `_get_schema` 로 가는 길이 같은 텍스트를 냅니다 (둘 다 단언합니다).

**1148 개는 숨겨져 있지 않습니다.** 그중 물어진 것은 `_C._shim_placeholder_schemas()` 에
들어가고, `str()` 이 `aten::foo(...) -> ...` 로 자기가 자리표시자임을 그대로 보여주며,
`.is_placeholder` 가 참입니다.

---

## 7. 아직 거짓말하는 것 — 텍스트 없이 답하는 술어 84 개

자리표시자의 `is_mutable` 은 **거짓을 답합니다.** 거절하지 않습니다. §8.1 이 지목한 바로 그
값이므로, 왜 그렇게 두었는지가 이 절입니다.

전체 실행(import · transformers 길 · FSDP · 분해 패스)에서 텍스트 없는 스키마에
`is_mutable`/`_is_view_op()` 를 묻는 `(op, 술어)` 쌍은 **102 개**입니다. 그중:

- **84 개는 상류에 op 자체가 없습니다.** 트리가 이름을 *합성해서* 물어봅니다 —
  `torch/distributed/tensor/_ops/autogen.py` 가 `<base>_` 와 `<base>_functional` 을 만들어
  캐묻고, `torch/_ops.py` 는 모든 패킷에 `default` 오버로드를 묻습니다
  (`aten::add` 는 상류에서 `add.Tensor`/`add.Scalar` 이고 `default` 가 없습니다).
  상류는 이 84 개를 **패킷 조회에서 AttributeError** 로 답하고, 호출자의 방어
  (`packet is None`, `except AttributeError`)는 **여기서 `False` 가 도달하는 것과 같은 가지**로
  갑니다.
- **18 개는 상류에 있습니다.** 그래서 §5 의 표로 옮겨 적었습니다. 지금은 0 개입니다.

**그러므로 남은 거짓말은 "존재하지 않는 op 에 대한" 것뿐이고, 조용하지 않습니다.**
`_C._shim_unanswered_predicates()` 가 전부 열거하므로 이 집합은 다시 발견하는 것이 아니라
**diff 하는 것**입니다. `verify_schemas.py --` 의 `check_unanswered` 가 매 실행마다
**"이 집합의 어느 op 도 상류에 없어야 한다"** 를 확인합니다. 하나라도 상류에 생기면
그 자리에서 이름을 대고 실패하며, `_GENERATED_ATEN_SCHEMA_TEXT` 에 넣으라고 말합니다.

이 검사가 실패할 수 있다는 것은 확인했습니다 —
`native_dropout_backward.out` 한 줄을 표에서 빼면 정확히 그것을 지목하고 종료 코드 1 을 냅니다.

---

## 8. 두 번 잘못 고쳤고 측정이 되돌렸다

### 8.1 "자리표시자는 거절한다" — 트리와 부딪혀 무너졌습니다

첫 설계는 자리표시자의 `is_mutable` 이 op 이름을 담아 `NotImplementedError` 를 던지는 것이었습니다.
DESIGN.md §6 의 거절 규약이고, "모른다" 를 "아니다" 와 절대 헷갈릴 수 없게 만드는 판본입니다.

**트리를 통과하지 못합니다.** `import transformers` 가 첫 번째에서 멈춥니다
(`aten::convolution_`, 고친 뒤에는 `aten::_native_batch_norm_legit_functional`). 하나씩 고치는
대신 거절을 **기록으로 바꿔** 한 번에 전수 조사했고, 그 결과가 §7 의 102/84/18 입니다.
**84 개가 상류에 없는 op 이었으므로 거절은 상류가 답하는 질문을 import 실패로 바꾸는 것**이었고,
그것이 이 설계를 버린 이유입니다.

### 8.2 "파일에 없는 aten 이름은 op 이 아니다" — 전제가 틀렸습니다

`aten::convolution_` 은 상류에 없습니다. 그러니 잘못은 스키마 층이 아니라 **레지스트리 층**에
있습니다 — shim 이 없는 op 의 패킷을 내주고 있었습니다. 그래서
`native_functions.yaml` 에 없는 aten 이름을 `_jit_get_operation` 이 거절하게 했습니다.

**`import torch` 가 즉시 깨졌습니다.** `torch/__init__.py:2395` 가
`quantized_lstm = ops.aten.quantized_lstm` 을 무조건 읽는데, `quantized_lstm` 은 상류에
있으면서 파일에는 없습니다. **파일은 aten op 의 완전한 목록이 아닙니다** — 상류의 aten 이름
1730 개 중 **176 개**가 파일에 없습니다.

파일이 완전한 것은 **자기가 선언한 op 의 in-place 변형**에 대해서입니다. `add_` 는 `add` 옆에,
`relu_` 는 `relu` 옆에 있습니다. 그래서 규칙을 그 모양으로만 좁혔습니다:

> `aten::<base>_` 를 거절한다 — 파일이 `<base>` 를 선언하고 `<base>_` 를 선언하지 않을 때에만.

측정: 이 모양의 이름 **1348 개** 중 상류가 등록하는 것은 **0 개**입니다. 반대 방향으로도,
파일이 빠뜨린 상류 in-place 이름 중 base 가 파일에 있는 것은 **0 개**입니다.
`test_an_in_place_variant_the_file_does_not_declare_is_not_an_operator` 가 `convolution_`·`mm_`
이 사라졌음과 `add_`·`relu_`·`quantized_lstm`·`zero` 가 남았음을 함께 단언합니다 —
넓은 규칙을 깨뜨린 반례를 좁은 규칙 옆에 붙여 둡니다.

---

## 9. 검사가 실패할 수 있는지 확인했다

다섯 규칙 중 어느 것도 "항상 통과하는" 검사에 기대지 않도록, 결함을 주입해 각 검사가 실제로
빨개지는지 확인했습니다.

| 주입한 결함 | 잡은 검사 | 결과 |
|---|---|---|
| `_GENERATED_ATEN_SCHEMA_TEXT` 에서 `native_dropout_backward.out` 삭제 | `verify_schemas.py check_unanswered` | 그 op 을 지목, exit 1 |
| 부동소수 인쇄기 삭제 (규칙 2) | `verify_schemas.py check_shim_schemas` | `_scaled_dot_product_flash_attention_for_cpu` 지목, exit 1 |
| 같은 결함, 상류 없이 | `test_shim.py` 왕복 테스트 | 순서를 고친 **뒤에만** 잡음 — §5 |
| `int` 되접기 삭제 (규칙 4) | `verify_schemas.py check_declared_schemas` | 17 개 지목, exit 1 |

**세 번째 줄이 §5 의 순서 문제입니다.** 순서를 고치기 전에는 같은 결함이 왕복 테스트를
통과했습니다.

---

## 10. 재현

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cd /Volumes/macMini/worktrees/bw-schema
bash vendor/vendor_torch.sh
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-schema
PY=/Volumes/macMini/caches/spike-venv/bin/python
bash vendor/install_shim.sh
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib

PYTHON=$PY sh rust/torch_c/pytests/run.sh        # 164 (기존 155 + 신규 9)
$PY tools/golden/compare.py                      # 2536/2536 ops=117
$PY rust/torch_c/pytests/verify_schemas.py       # 3075/3075
```

`verify_schemas.py` 는 **상류 torch 가 있는 환경**에서 돌고, shim 은 벤더 트리를 `PYTHONPATH`
에 얹은 별도 프로세스에서 답합니다 — 한 인터프리터가 둘을 다 가질 수 없기 때문이고,
`tools/golden/compare.py` 가 두 번째 프로세스로 가는 이유와 같습니다.

```
overloads.json:                          126/126
methods.json:                            124/124
_NON_ATEN_SCHEMA_TEXT:                     22/22
_GENERATED_ATEN_SCHEMA_TEXT:               18/18
shim _schema text and is_mutable:        117/117    ← §8.1 의 판정
native_functions.yaml re-printed:      2584/2584    ← 재인쇄기 전체
predicates answered without text:       84, 84/84 about ops upstream does not have
```

---

## 11. 새로 읽을 수 있게 된 것

| 함수 | 답하는 질문 |
|---|---|
| `_C._shim_schema_source()` | 어느 `native_functions.yaml` 을 읽었는가, 또는 왜 못 읽었는가 |
| `_C._shim_schema_provenance(qualname, overload="")` | 네 층 중 어디가 이 스키마를 답했는가 |
| `_C._shim_placeholder_schemas()` | 텍스트 없이 내준 스키마 전부 |
| `_C._shim_unanswered_predicates()` | 텍스트 없이 답한 `(op, 술어)` 전부 |
| `schema.is_placeholder` | 이 스키마에 텍스트가 있는가 |

`_shim_registrations` · `_shim_overloads` 와 같은 이유로 있습니다 — **빈틈의 크기는 산출물을
읽어 추론하는 것이 아니라 물어서 답을 얻는 것이어야 합니다.**

`_shim_schema_provenance` 는 그중 유일하게 **텍스트로는 확인할 수 없는 것**을 답합니다.
네 층이 전부 `_Schema` 를 돌려주고 대개 같은 답을 내므로, "파일이 이것을 답했다" 는 텍스트를
봐서는 알 수 없습니다. 그것을 물을 수 없으면, 파일을 조용히 안 보게 되는 재배치가 텍스트를
비교하는 모든 테스트를 통과합니다 — §5 에서 실제로 그랬습니다.

---

## 12. 남은 것

- **`.out`/functional 생성 변형 1148 개.** `torchgen/native_function_generation.py` 를 돌릴 수
  있으면(= `pyyaml` 을 의존성으로 받아들이면) 전부 채워집니다. 지금은 트리가 물은 18 개만
  옮겨 적혀 있고, 새로 물어지는 것은 `check_unanswered` 가 이름을 대고 실패합니다.
- **`prims::` 자리표시자.** 트리가 `Library.define()` 으로 정의하므로 층 1 이 답해야 하는데
  자리표시자로 남는 것이 있습니다 — 평범한 `import torch` 뒤 126 개,
  `verify_schemas.py` 가 쓰는 프로브 모듈까지 임포트하면 143 개.
  **조사하지 않았습니다** — `_schema` 를 읽는 코드가 이것들에 무엇을 묻는지부터 재야 합니다.
  §7 의 84 개 중 17 개가 `prims::<name>_` 모양이므로 §8.2 의 in-place 규칙이
  `prims` 에도 서는지가 첫 질문일 것입니다.
- **레지스트리는 여전히 열려 있습니다.** `_jit_get_operation` 은 §8.2 가 좁힌 한 모양을
  빼면 아무 이름에나 callable 을 돌려줍니다. 그것이 §7 의 84 개가 존재하는 이유입니다.
  닫으려면 aten op 의 완전한 목록이 필요하고, 이 트리에는 없습니다.
- **docs/distributed/DISTRIBUTED.md §8.1 은 아직 "미해결"로 적혀 있습니다.** 이 문서가 그 항목의 답이지만,
  그 파일은 이 작업의 소유 범위 밖이라 건드리지 않았습니다.
  > **정정 (문서 감사, 2026-09):** 이 문장은 쓰인 순간부터 자기모순이었습니다 — `git show
  > --stat`으로 확인하면, 이 문장이 들어 있는 바로 그 커밋(`e26e54b`, 이 문서 자체의 착지
  > 커밋)이 `docs/distributed/DISTRIBUTED.md` 도 함께 22줄 바꿔 §8.1 에 "해결됐습니다 (2026-08-28)." 블록쿼트를
  > 넣었습니다. "건드리지 않았다"는 "이 문서를 쓴 사람이 직접 고치지 않고 조율 세션이 착지시켰다"
  > 는 뜻이었을 가능성이 높지만, 문면 그대로 읽으면 오늘도 여전히 틀립니다 — `docs/distributed/DISTRIBUTED.md`
  > §8.1 은 지금 이 문서를 정확히 가리키는 "해결됐습니다" 블록을 담고 있습니다.
  > <!-- DOCWATCH: symbol-in-file docs/distributed/DISTRIBUTED.md 해결됐습니다 present -->
