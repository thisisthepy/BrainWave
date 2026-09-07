# 이름 붙이기 — 커널은 있는데 철자가 없던 19 개

조율 세션이 실측한 `Tensor` 메서드 25 개 중 19 개 거부 목록:

    argmax softmax flatten le topk sort masked_fill repeat split chunk
    narrow squeeze permute clamp tril triu flip gather index_select

이 문서는 그중 몇 개가 **커널이 이미 있어서 철자만 채우면 되는지**, 몇 개가 **커널부터 없어서
손댈 수 없는지**를 가르고, 채운 철자가 상류와 같은 값을 내는지 실측한 기록이다.

판정은 종료 코드로만 한다. grep 으로 성공을 판단하지 않는다.

---

## 0. 한눈에

| | 이전 | 지금 |
|---|---|---|
| `verify_schemas.py` | 154/154 | **170/170** (+16) |
| 골든 하네스 | 1616/1616, ops covered=78 | **1616/1616, ops covered=78** (무회귀) |
| `--inject-fault value/shape/dtype` | 1/1/1 | **1/1/1** (그대로) |
| 호스트 스모크 (`pytests/run.sh`) | — | **62/62, exit 0** |
| 3 타깃 (host / androidNdk arm64-v8a / aarch64-apple-ios) | — | **전부 exit 0** |

19 개 중 **커널이 있어서 실제로 고칠 수 있었던 것은 7 개** (`argmax`, `softmax`, `le`,
`topk`, `sort`, `masked_fill`, `squeeze` — 뒤의 세 개는 부분적으로), 나머지 12 개는 커널
자체가 `aten.rs` 에 없어 이 작업 범위(스펠링)로는 손댈 수 없다.

---

## 1. 채운 철자와 상류 대조 결과

전부 `/Volumes/macMini/caches/spike-venv/bin/python`(torch 2.13.0)을 상류로 두고,
`tools/golden/loader.py::load_shim()` 로 이 빌드의 `_C.so` 를 직접 로드해 `TensorBase.<method>`
와 `_VariableFunctions.<fn>` 양쪽을 실측했다. 재현 스크립트는 이 보고서 이후 폐기했으므로
숫자만 남긴다.

### `argmax` — 새 스펠링, 완전히 동작

`methods.json`(`argmax`) 과 `overloads.json`(기존 `argmax` 항목에 `.out` 은 이미 있었으나
`.default` 사용례를 재확인)을 채웠다. 커널: `aten.argmax.default`.

    x.argmax()                 -> 상류와 동일 (스칼라 인덱스)
    x.argmax(dim=1)             -> 상류와 동일 (shape/value)
    x.argmax(dim=1, keepdim=True) -> 상류와 동일
    torch.argmax(x, dim=1)       -> (`_VariableFunctions.argmax`) 상류와 동일

### `softmax` — 새 스펠링, 테이블이 아니라 Python 합성으로

`methods.json` 의 `_README` 가 이미 적어 둔 이유대로, `Tensor.softmax` 의 파서 레벨 키는
`aten.softmax.int` 인데 상류 디스패처가 실제로 보는 키는 `aten._softmax.default` 다
(`TorchDispatchMode` 로거로 재확인, `docs/bindings/NN_SURFACE.md` §6). 테이블은 스키마 문자열 그대로
디스패치 키를 만들기 때문에 이 어긋남을 테이블로는 표현할 수 없다 — 그래서 `bootstrap.py` 에
`_install_tensor_softmax` 를 새로 추가해 `to`/`item` 과 같은 "Python 레벨" 그룹에 넣었다.

측정한 실제 분기 (`dtype=None`, `dtype=self.dtype`, `dtype=다른 타입` 셋 다 재현):

    x.softmax(dim)                        -> aten._softmax.default(x, dim, False)
    x.softmax(dim, dtype=x.dtype)          -> aten._softmax.default(x, dim, False)  (변환 없음)
    x.softmax(dim, dtype=other)            -> aten._to_copy.default(x, dtype=other)
                                               다음 aten._softmax.default(_, dim, False)

    softmax(dim=-1)                 -> 상류와 float32 로 일치 (atol 1e-6)
    softmax(dim=-1, dtype=float64)  -> 상류와 float64 로 일치 (atol 1e-10)

`methods.json` 의 README 코멘트도 갱신해 "테이블에는 없지만 Python 합성으로는 있다"는 것을
명시했다.

### `topk` — 새 스펠링, 완전히 동작

`methods.json`/`overloads.json` 모두에 추가. 커널: `aten.topk.default` (유일한 오버로드 —
`.values` out 변형은 커널이 없고, `Tensor.topk` 의 벤더 `.pyi` 에도 out= 이 없어 아예 넣지
않았다). `values`/`indices` 둘 다 상류와 일치 확인(dim=2, k=2, largest 기본값).

### `sort` — 새 스펠링, 완전히 동작

`methods.json`/`overloads.json` 모두에 추가. 순서는 벤더 `__init__.pyi` 의 `@overload` 순서
그대로 — `stable=` 키워드 전용 변형(`aten.sort.stable`, 커널 없음)이 먼저, 일반형
(`aten.sort.default`, 커널 있음)이 나중. `stable` 인자에 기본값 텍스트가 없어(`bool? stable`
에 `=` 없음) `has_default_value()` 가 거짓이 되고, `stable=` 을 안 주면 첫 스키마가 바인딩에
실패해 두 번째로 정확히 넘어간다 — 상류 디스패치 로그와 일치 실측함.

    x.sort(dim=1)                 -> 상류와 일치 (values, indices 둘 다)
    x.sort(dim=1, descending=True) -> 상류와 일치
    torch.sort(x, dim=1)           -> (`_VariableFunctions.sort`) 상류와 일치

out 변형(`sort.values`, `sort.values_stable`)은 넣지 않았다 — 커널이 없고, `max.dim` 등 이미
있는 다중 반환 op 들도 이 표에서 out 변형을 넣지 않는 것이 기존 관례였다(값 두 개를 `out=` 튜플로
받는 바인딩은 이 `_Overloads` 기계가 아예 표현하지 못한다).

### `masked_fill` — 이미 있었다. 손대지 않음

`methods.json` 에 `masked_fill.Tensor`/`masked_fill.Scalar` 가 **이미** 있었고, `aten.rs` 도
둘 다 커널이 있다(`Scalar` 는 `IMPLEMENTED`, `Tensor` 는 `IMPLEMENTED_AWAITING_GOLDEN` — 골든
집계에서만 빠져 있을 뿐 디스패치는 정상). 실제로 두 형태 다 이 빌드에서 상류와 일치하는 값을
낸다:

    x.masked_fill(mask, -9.0)          -> 상류와 일치 (Scalar)
    x.masked_fill(mask, torch.tensor(3.5)) -> 상류와 일치 (Tensor)
    x.masked_fill(non_bool_mask, ...)   -> 상류처럼 거부(런타임 에러) — 정확히 그 자리에서

19 개 거부 목록에 있었던 이유는 **이 작업에서 알아낼 수 없었다** — 이번 빌드에서는 재현되지
않았다. 가능한 설명은 (a) 측정이 이 커밋 이전의 낡은 아티팩트에 대고 이루어졌거나,
(b) 측정에 쓰인 인자 모양이 여기서 시도한 것과 달랐거나(예: dtype 이 다른 mask, 브로드캐스트가
필요한 shape 등) 둘 중 하나로 보인다. 조율 세션이 원래 측정 스크립트로 다시 돌려 보는 편이
확실하다.

### `le` — 이미 있었다. 절반만 커널이 있다

`methods.json` 에 `le.Tensor`/`le.Scalar` 가 **이미** 있었다. 그러나 `aten.rs` 에는
`aten.le.Scalar` 커널만 있고 `aten.le.Tensor` 는 없다:

    x.le(0.5)   -> 상류와 일치 (Scalar, 커널 있음)
    x.le(other_tensor) -> "aten op not implemented in torch._C shim: aten.le.Tensor"

스펠링은 정확하게 해석되고(두 후보 중 맞는 것을 고름), 커널이 없는 쪽만 그 자리에서 거부한다 —
이것이 `methods.json` 의 README 가 말하는 "테이블 항목이 커널을 보장하지 않는다"는 바로 그
사례다. `le.Tensor` 커널이 이 작업의 파일 범위(`aten.rs` 금지) 밖이라 채울 수 없다. masked_fill
과 마찬가지로, 19 개 목록에 `le` 가 오른 것이 `le.Scalar` 쪽 실패였는지 `le.Tensor` 쪽
실패였는지는 알 수 없다 — 전자라면 이 빌드에서는 이미 고쳐져 있고, 후자라면 여전히 커널이
없어서 막혀 있다.

### `squeeze` — 부분적으로 새 스펠링

`methods.json`/`overloads.json` 모두에 세 오버로드를 벤더 `.pyi` 순서(무인자 → `dim:int` →
`dim:int[]`) 그대로 추가했다. 커널은 `aten.squeeze.dim` 하나뿐이다:

    x.squeeze(1)     -> 상류와 일치 (커널 있음)
    x.squeeze()      -> "aten op not implemented ...: aten.squeeze.default" (커널 없음)
    x.squeeze((1,3)) -> "aten op not implemented ...: aten.squeeze.dims" (커널 없음)

무인자·리스트 형태는 스펠링만 채워 두었다 — 정확한 키로 거부하므로 다음 작업의 작업 큐에
그대로 잡힌다.

### `multinomial` — 목록 밖이지만 요청받은 대로 채움

19 개 목록에는 없지만, 작업 지시가 "`torch.multinomial` 같은 함수 스펠링도 없다"고 직접
지목했고 커널(`aten.multinomial.default`)이 있으므로 `methods.json`/`overloads.json` 양쪽에
추가했다. `.out` 변형은 벤더 `.pyi` 의 함수 레벨에는 있지만 메서드 레벨에는 없어 `overloads.json`
에만 넣었다(`argmax` 와 같은 패턴).

    x.multinomial(3, replacement=True)      -> shape/dtype 상류와 일치, 인덱스 범위 내 (RNG 스트림은
    torch.multinomial(x, 3, replacement=True) -> 다르므로 값 자체는 비교 대상이 아님 — 다른 uniform_/
                                                   normal_ 항목들과 같은 근거)

---

## 2. 커널이 없어서 손대지 않은 것 (12 개 + `torch.randn`)

`aten.rs` 의 `IMPLEMENTED`/`IMPLEMENTED_AWAITING_GOLDEN` 어디에도 다음 이름의 커널이 없다.
`_aten_dispatch` 가 유일한 문이므로, 스펠링만 채워 봤자 정확한 이름으로 거부할 뿐 아무것도
동작시키지 못한다 — 그리고 이 작업은 커널 추가가 금지되어 있다. 손대지 않고 그대로 남겼다:

    flatten repeat split chunk narrow permute clamp tril triu flip gather index_select

추가로, 지시에서 예로 든 `torch.randn` 도 커널이 없다(`aten.rs` 에 `randn` 계열 op 자체가
없음 — `normal_`/`uniform_`/`randint` 만 있다). 이것도 목록으로만 남긴다.

이 12+1 개는 커널을 만드는 다른 에이전트에게 넘긴다.

---

## 3. 표 변경 요약

`overloads.json`: `argmax`(기존 항목 재확인) 뒤에 `topk`, `sort`, `squeeze`, `multinomial`
네 항목 신설. `methods.json`: `masked_fill` 뒤에 `argmax`, `topk`, `sort`, `squeeze`,
`multinomial` 다섯 항목 신설, `_README` 의 `softmax` 코멘트를 Python 합성 위치를 가리키도록
갱신. `bootstrap.py`: `_install_tensor_softmax` 신설, `_install_tensor_methods` 에서 호출.

모든 스키마 문자열은 `str(torch.ops.aten.<op>.<ov>._schema)` (torch 2.13.0)에서 그대로
전사했다 — 지어낸 것이 없다. `pytests/verify_schemas.py` 가 그것을 재확인한다(170/170).

---

## 4. 두 번째 회차 — 커널이 82 개로 늘어난 뒤 (`addmm`/`native_layer_norm`/`split.Tensor`/`tanh`)

`docs/models/GPT2.md`가 이 넷을 구현해 커널이 78 → 82 개가 됐고, §5에 "커널은 있는데 스펠링이 없다"는
표를 명시적으로 남겼다. 이 회차는 그 표를 메우는 작업이다. 근거는 두 가지뿐이었다 — `_aten_implemented()`가
돌려주는 82개 목록(권위 있는 출처, `aten.rs`를 grep하지 않았다)과, 상류 `torch.ops.aten.<op>.overloads()` /
벤더링된 `.pyi`.

### 4.0 82개 전부 감사

지난 회차가 열어 둔 14개(`flatten masked_fill repeat split chunk narrow squeeze permute clamp
tril triu flip gather index_select`) 중 커널이 **새로 생긴 것은 `split` 하나뿐**이다
(`aten.split.Tensor`). `masked_fill`/`squeeze`는 커널도 스펠링도 지난 회차 그대로(부분)이고,
나머지 11개는 여전히 커널 자체가 없다 — 이번에도 건드리지 않았다.

그래서 판단 기준을 "14개 목록"에서 "82개 커널 전부"로 넓혔다. `_aten_implemented()`의 82개
각각에 대해 `overloads.json`/`methods.json`의 스키마를 파싱해 `(op, overload)` 키를 뽑고,
82개와 대조하는 감사 스크립트를 돌렸다. 결과: **82개 전부가 어떤 형태로든 스펠링을 갖고
있다** — 표 항목(대다수), Python 레벨 합성(`softmax`/`to`/`item`/`__getitem__`/`linear`/
`silu`/`sdpa`/`layer_norm`, 기존 + 이번에 추가한 `layer_norm`), 또는 상류 자체에 공개
스펠링이 없는 내부 전용 op(`alias.default` — `hasattr(torch.Tensor, "alias")` /
`hasattr(torch, "alias")` 둘 다 실측 결과 `False`, 감쌀 스펠링이 존재하지 않는다).

### 4.1 채운 것

| 이름 | 파일 | 상류와 값 대조 |
|---|---|---|
| `tanh` (함수+메서드) | `overloads.json`/`methods.json` | 일치 (`torch.tanh(x)` == `x.tanh()`) |
| `addmm` (함수+메서드) | `overloads.json`/`methods.json` | 일치 |
| `mm` (메서드 — 함수는 이미 있었다) | `methods.json` | 일치 |
| `select` (함수+메서드) | `overloads.json`/`methods.json` | 일치 |
| `scatter` (함수+메서드) | `overloads.json`/`methods.json` | `.src` 일치. `.value`/`.reduce`/`.value_reduce`는 커널이 없어 정확한 이름으로 거부 (실측) |
| `split` (함수+메서드, int 형태) | `overloads.json`/`methods.json` | 일치 (`torch.split(x,1,0)` == `x.split(1,0)`) |
| `split_with_sizes` (함수+메서드) | `overloads.json`/`methods.json` | 커널 없음 — 정확한 이름으로 거부 (실측, 아래 4.2) |
| `native_layer_norm` (함수) | `overloads.json` | 3-튜플(out, mean, rstd) 전부 일치 |
| `layer_norm` (함수, Python 합성) | `bootstrap.py` `_install_composites` | 일치 (`F.layer_norm`/`torch.layer_norm`/`nn.LayerNorm`이 전부 도달하는 지점) |
| `_get_cudnn_enabled`/`_set_cudnn_enabled` | `bootstrap.py` `_install_behaviour` | 상태 게터/세터, `F.layer_norm`이 읽는 값을 실제로 반환 |

전부 `tools/golden/loader.py::load_shim()`으로 이 빌드의 `_C.so`를 직접 로드해
`TensorBase.<method>`와 `_VariableFunctions.<fn>` 양쪽을 실측했다 (이전 회차와 같은 방법).
`torch 2.13.0`을 상류로 두고 동일 입력에 `torch.allclose`로 비교했으며, 전부 일치했다
(`tanh`/`addmm`/`mm`/`select`/`scatter.src`/`split`/`layer_norm`/`native_layer_norm`의
세 출력 전부).

**전체 vendor 트리(`import torch`)로도 시도했으나 판단 근거로 쓰지 않았다.** `TORCH_USE_RTLD_GLOBAL=1`로
`vendor/`를 얹어 실제로 실행해 보면, `x.argmax()`처럼 이미 동작이 확인된 기존 스펠링조차
`torch._C._functorch.is_functorch_wrapped_tensor`에서 막힌다 — 이번 회차가 만든 문제가
아니라 vendor 트리를 통한 메서드 호출 전반에 걸친 기존 벽이다(범위 밖의 `_functorch` 서브모듈
배선 문제로 보인다, 원인은 추적하지 않았다). 그래서 지난 회차와 같은 방법(`loader.load_shim()`로
`_C`를 직접 로드해 `TensorBase`/`_VariableFunctions`를 부르는 것)으로 판단했다 — `docs/models/GPT2.md` §4가
판정을 aten 레벨로 내린 것과 같은 이유다.

### 4.2 `split`의 스키마 두 개가 이름이 다른 이유

`methods.json`의 `split` 항목은 `aten::split.Tensor`와 `aten::split_with_sizes`를 **한 키
아래** 담고 있다 — 이 표에서 유일하게 스키마들이 op 이름을 공유하지 않는 항목이다. 상류
`torch/_tensor.py:Tensor.split`을 직접 읽으면(`TorchDispatchMode`로도 재확인)
`isinstance(split_size, int)`로 분기해서 `torch._VF.split`과 `torch._VF.split_with_sizes`
중 하나를 호출한다 — 진짜로 다른 두 aten op다. `_Overloads.resolve()`는 키와 스키마의 op
이름이 같은지 검사하지 않고 스키마를 순서대로 시도하다 바인딩되는 것을 쓰므로, 이 방식으로도
정확히 상류와 같은 분기를 재현한다: int 인자는 `split.Tensor`(커널 있음)에 바인딩되고
list/tuple 인자는 `split_with_sizes`(커널 없음, `aten.split_with_sizes.default`로 정확히
거부)에 바인딩된다. `overloads.json`은 다르다 — `_VariableFunctions.split`은 실측상 int
형태로만 불리므로(벤더 트리가 리스트를 항상 `split_with_sizes`로 직접 보낸다)
`split`/`split_with_sizes`를 별개 키로 깔끔하게 나눴다.

### 4.3 `F.layer_norm`이 실제로 가는 키

`TorchDispatchMode` 로거로 재확인(torch 2.13.0): `F.layer_norm(...)`, `torch.layer_norm(...)`,
`nn.LayerNorm`의 forward가 **전부 `aten.native_layer_norm.default` 하나로만** 기록되고,
`aten.layer_norm.default`는 한 번도 찍히지 않는다. `torch/nn/functional.py:2966`의
`F.layer_norm`은 `torch.layer_norm(input, normalized_shape, weight, bias, eps,
torch.backends.cudnn.enabled)`을 그대로 부르고, `aten::layer_norm`은 `Composite
ImplicitAutograd`라 실제 커널을 갖지 않는다 — C++ 본체가 `std::get<0>(native_layer_norm(...))`
그 자체다. 그래서 `layer_norm`을 `overloads.json`에 원본 스키마로 넣지 않고
(`softmax`가 `methods.json`에서 빠진 것과 정확히 같은 이유 — 파서 키와 실제 디스패치 키가
다르면 표 항목은 절대 구현되지 않을 작업 항목의 이름만 짓는다), `bootstrap.py`
`_install_composites`에 `native_layer_norm.default`를 부르고 첫 번째 결과만 돌려주는
Python 합성으로 심었다. `native_layer_norm` 자체는 진짜 leaf 커널이라(디스패치 키가 그대로)
`overloads.json`에 평범한 표 항목으로 넣었다.

`F.layer_norm`은 이 합성에 닿기 전에 `torch.backends.cudnn.enabled`부터 읽는데, 이건
`torch._C._get_cudnn_enabled()`를 부르는 프로퍼티다. 이 함수가 지난 회차까지 raising 스텁이라
`F.layer_norm`이 그 자리에서 막혔다(`docs/models/GPT2.md` §5가 정확히 이 지점을 지목했다). 진짜
cuDNN 백엔드는 없어도(`_has_cudnn=False`, 안 바뀜) 게터/세터 자체는 평범한 불리언 상태
셀이면 되므로 `_install_behaviour`에 실제 게터/세터를 심었다(기본값 `True`, 상류 기본값과
동일).

**`nn.LayerNorm(...)`은 여전히 안 된다 — `TensorBase.zero_`가 다른 에이전트 담당이라 손대지
않았다.** `reset_parameters`가 `init.zeros_(self.bias)`를 부르는 자리에서 정확히 그 이름으로
막힌다(실측, `docs/models/GPT2.md`가 예고한 그대로). `F.layer_norm(x, [C], weight, bias)`처럼 이미
만들어진 파라미터 텐서를 직접 넘기는 경로는 `zero_`를 거치지 않으므로 이번 수정만으로 된다.

### 4.4 스키마 숫자

`verify_schemas.py`: **170/170 → 199/199** (+29 — `overloads.json` 72→90, `methods.json`
98→109). 골든 하네스는 **1781/1781, ops covered=82 그대로**(무회귀 — 스펠링 추가는 커널
집합을 바꾸지 않는다, `_aten_implemented()`가 유일한 출처이기 때문). `--inject-fault
value/shape/dtype` 전부 그대로 exit 1. 호스트 스모크(`pytests/run.sh`) exit 0. 3 타깃
(host / androidNdk arm64-v8a / aarch64-apple-ios) 전부 exit 0.

### 4.5 손대지 않은 것

`flatten repeat chunk narrow permute clamp tril triu flip gather index_select` — 여전히
`aten.rs`에 커널이 없다(`_aten_implemented()`에 없음). `masked_fill`/`squeeze`도 커널·스펠링
모두 지난 회차 상태 그대로 다시 확인만 했다. `torch.softmax`/`F.softmax`(함수 레벨)는
`Tensor.softmax`와 별개로 여전히 스펠링이 없다 — 지난 회차가 메서드 폼만 고쳤고 이번 회차도
지시 범위(14개 목록 + GPT2.md의 넷) 밖이라 손대지 않았다, 실측만 남긴다
(`_VariableFunctions.softmax`가 여전히 "overload resolution has no table entry" 로 거부).

---

## 5. 세 번째 회차 — `gelu`/`gather`/`zero_`, 그리고 `nn.LayerNorm(...)`이 실제로 열리는지

`docs/architectures/ARCH.md`가 이 셋의 커널을 넣고 나서(82 → 85개) 남긴 항목을 메우는 회차다. §0이 지목한 대로
`zero_`의 스펠링이 없으면 `nn.LayerNorm`의 `reset_parameters`가 `init.zeros_(self.bias)`에서
막힌다. 그 벽이 실제로 걷히는지가 이 회차의 진짜 판정이다.

### 5.0 세 이름 중 하나는 표(table) 항목이 아니다 — 측정으로 갈랐다

셋을 지시받은 대로 `overloads.json`/`methods.json`에 다 넣기 전에, 각각이 실제로 상류에서
어느 표면으로 도달하는지부터 쟀다(`TorchDispatchMode` 로거, torch 2.13.0):

    x.gather(dim, idx)        -> aten.gather.default   (Tensor 메서드로 존재)
    torch.gather(x, dim, idx) -> aten.gather.default    (_VariableFunctions 로 존재)
    x.zero_()                 -> aten.zero_.default     (Tensor 메서드로 존재)
    torch.zero_(x)            -> aten.zero_.default     (_VariableFunctions 로도 존재 -- 벤더 .pyi 에 실제로 있다)
    F.gelu(x, approximate=..) -> aten.gelu.default

`gather`와 `zero_`은 파서 키와 디스패치 키가 일치하는 평범한 leaf라서 표 항목으로 충분하다.
`gelu`는 다르다 — **`hasattr(torch.Tensor, "gelu")`가 `False`다.** `Tensor.gelu()`라는 메서드
자체가 상류에 없다. `torch/nn/functional.py:2054`가 `gelu = _add_docstr(torch._C._nn.gelu, ...)`이므로
`F.gelu`는 `_C._nn` 서브모듈의 이름이지 `TensorBase`도 `_VariableFunctions`도 아니다 — `linear`/
`silu`가 표가 아니라 `bootstrap.py`의 `_install_nn`에 있는 것과 정확히 같은 이유이고 같은 자리다.
그래서 `gelu`는 `overloads.json`/`methods.json`에 넣지 않았다 — 넣었다면 상류가 절대 보지 않는
파서 키(`aten.gelu.default`를 `Tensor.gelu`나 `torch.gelu`로 노출하는 것 자체가 상류에 없는 이름을
짓는 일)를 만드는 셈이었다.

### 5.1 채운 것

| 이름 | 파일 | 위치 |
|---|---|---|
| `gather` (메서드 `aten::gather`, 함수 `aten::gather`/`.out`) | `methods.json`/`overloads.json` | 평범한 표 항목. 메서드 쪽 벤더 `.pyi`(`__init__.pyi:4767`)에는 `out=`이 없어 `.out` 스키마를 안 넣었고, 함수 쪽 `.pyi`(`_VariableFunctions.pyi:13255`)에는 있어 `.out`을 먼저(관례대로) 넣었다 — 커널이 없어(`aten.gather.out`은 `_aten_implemented()`에 없다) `out=`을 주는 호출은 정확한 이름으로 거부된다(실측) |
| `zero_` (메서드 `aten::zero_`, 함수 `aten::zero_`) | `methods.json`/`overloads.json` | 평범한 표 항목. `torch.ops.aten.zero_.overloads()`가 `['default']` 하나뿐이라 `.out` 걱정이 없다 |
| `gelu` (`_C._nn.gelu`) | `bootstrap.py` `_install_nn` | `silu`/`linear`와 같은 자리의 Python 합성. `dispatch("aten.gelu.default", input, approximate=approximate)`로 넘긴다 |

셋 다 `_C.so`를 재빌드해 `tools/golden/loader.py`로 직접 로드하고 상류 torch 2.13.0과 값을
대조했다 — `gather`(메서드/함수 둘 다), `zero_`(메서드/함수 둘 다), `gelu`(`approximate='none'`/
`'tanh'` 둘 다)가 **전부 비트까지 일치**했다(`torch.tensor(...).tolist()`를 나란히 찍어 비교).
`gelu`의 두 근사식 값은 `docs/architectures/ARCH.md` §1이 실측한 상류 수치와도 그대로 일치한다:

    none [-3,-1,-0.5,0,0.5,1,3] -> [-0.00405, -0.158655, -0.154269, 0.0, 0.345731, 0.841345, 2.995950]
    tanh [-3,-1,-0.5,0,0.5,1,3] -> [-0.00364, -0.158808, -0.154286, 0.0, 0.345714, 0.841192, 2.996363]

### 5.2 처음 쓴 `gelu` 합성은 틀렸다 — `approximate`가 위치 인자로도 통과했다

첫 구현은 `def gelu(input, approximate="none")`이었다. `dispatch()`에는 항상 `approximate=`를
키워드로 넘기니 `aten.rs`의 키워드 전용 검사(`args.len() > 1`이면 거부)를 통과할 것이라고
**추론했는데, 틀렸다.** 그 검사는 *`dispatch`가 받는 튜플의 길이*만 보고, 이 합성은 자신의
호출자가 `approximate`를 위치로 줬든 키워드로 줬든 항상 `dispatch(..., approximate=approximate)`
한 개의 위치 인자로 재포장해서 넘긴다 — 그래서 **`F.gelu(x, "tanh")`(위치 인자)가 조용히
통과했다.** 상류는 이걸 `TypeError`로 거부한다(측정, `torch._C._nn.gelu(x, "tanh")` ->
`gelu() takes 1 positional argument but 2 were given`).

`vendor/probe.py --mode strict --target torch`로 벤더 트리 전체를 이 빌드의 `_C` 위에 얹어
`F.gelu(x, "tanh")`를 직접 불러보고서야 잡았다 — 추론이 아니라 실행해서 발견했다. 고친 것은
`aten.rs`가 아니라 **이 합성 자신의 Python 시그니처**다: `def gelu(input, *, approximate="none")`.
`*`가 없으면 `dispatch` 쪽 키워드 전용 검사가 아무리 정확해도 그 앞에서 이미 잘못된 값을
받아들인 뒤이므로 소용이 없다. 고친 뒤 같은 호출이 상류와 같은 자리에서 거부되는 것을
재확인했다:

    torch.nn.functional.gelu(x, "tanh")  ->  TypeError: gelu() takes 1 positional argument but 2 were given
    (상류와 셰임 양쪽 동일 메시지, 재빌드 후 재측정)

### 5.3 `nn.LayerNorm(...)`이 이제 열린다 — `vendor/probe.py`로 끝까지 확인했다

`docs/architectures/ARCH.md` §0/§4.3이 지목한 벽은 `reset_parameters`의 `init.zeros_(self.bias)` ->
`TensorBase.zero_`였다. `zero_` 스펠링을 채운 뒤 `TORCH_USE_RTLD_GLOBAL=1`로 벤더 트리 전체를
이 `_C` 위에서 `import torch`시키고(`vendor/probe.py --mode strict --target torch`, exit 0,
`torch.__version__ == 2.13.0`까지 확인), 그 프로세스 안에서 직접 실행했다:

    torch.nn.LayerNorm(8)                 -> 생성자 성공
      .weight.tolist()  == [1.0]*8        (fill_.Scalar 경로, 그대로)
      .bias.tolist()    == [0.0]*8        (zero_ 경로 -- 이번에 열림)
    ln(torch.ones(2, 8))                  -> forward 성공, shape (2, 8)
      결과 전부 0.0  (상수 입력의 LayerNorm은 분산이 0이라 정확히 0이 나오는 것이 맞다)

**생성자와 순전파 둘 다 통과했다.** `docs/architectures/ARCH.md` §0이 예고한 두 겹(`_get_cudnn_enabled`는
이미 해결, `zero_`가 남은 것) 뒤에 **셋째 겹은 없었다** — `zero_` 스펠링을 채우는 것만으로
`nn.LayerNorm(...)`이 끝까지 갔다.

주의할 것 하나: `repr(ln)`이나 `print(ln.weight)`(값이 아니라 텐서 객체 자체를 출력하는 경로)는
여전히 `torch._C._functorch.is_functorch_wrapped_tensor`에서 막힌다 — §4.1이 이미 기록한,
이 작업과 무관한 기존 벽이다(`_tensor_str.py`가 모든 텐서 repr에서 그 이름을 부른다). `.tolist()`로
값만 읽으면 이 벽을 타지 않는다. `torch.randn`도 여전히 커널이 없어 거부된다(스펠링 문제가 아니라
`aten.rs`에 `randn` 계열 자체가 없음, §2에서 이미 남긴 목록 그대로) — 그래서 위 forward 검증은
`torch.ones`로 입력을 만들었다.

### 5.4 11개 커널 없는 목록 재확인 — 새로 열린 것 없음

`docs/architectures/ARCH.md`가 이번에 커널을 넣은 것은 `gelu`/`gather`/`zero_` 셋뿐이다. 남은 11개
(`flatten repeat chunk narrow permute clamp tril triu flip index_select`)를 이번 빌드의
`_aten_implemented()`에 대고 다시 확인했다 — **전부 여전히 커널이 없다**(`permute`는 다른
에이전트가 지금 작업 중이라는 지시를 받아 건드리지 않았다). 커널이 없는 스펠링은 정확한 이름으로
거부되는 것 말고 할 수 있는 일이 없으므로, 이 11개는 그대로 다음 회차로 넘긴다.

### 5.5 숫자

`verify_schemas.py`: **199/199 → 204/204** (+5 — `overloads.json` 90→93 [`gather`+`.out`,
`zero_`], `methods.json` 109→111 [`gather`, `zero_`]; `gelu`는 `_install_nn` 합성이라 이 표에
없다). 골든 하네스는 **1934/1934, ops covered=85 그대로**(무회귀 — 스펠링은 `_aten_implemented()`가
답하는 커널 집합을 바꾸지 않는다). `--inject-fault value/shape/dtype` 전부 그대로 exit 1. 호스트
스모크(`pytests/run.sh`, `test_shim.py` 65개 + `compare.py --self-test`) exit 0. 3 타깃
(host / androidNdk arm64-v8a / aarch64-apple-ios) 전부 exit 0.

### 5.6 손대지 않은 것 / 이 회차 밖

`aten.rs`, `tools/golden/cases.py`, `tools/golden/compare.py`, `rust/torch_c/pytests/test_shim.py`는
이번 회차의 파일 범위 밖이라 한 줄도 고치지 않았다 — 커널 추가도 하지 않았다. `nn.LayerNorm`에
대한 회귀 테스트를 `test_shim.py`에 박아 두는 것은 다음 회차의 작업 항목이다(§4가 남긴 것과 같은
이유 — 이번에 손으로 확인한 것을 자동화하지 못했다).

---

## 6. 네 번째 회차 — 커널은 있는데 이름이 없는 부류를 F.\* 기준으로 전수 조사

기기 작업(`docs/devices/DEVICE.md`)이 `nn.ReLU()` -> `F.relu` -> `torch.relu(x)`가 `NotImplementedError`를
낸다고 보고했다 — 호스트와 기기 양쪽에서 똑같이, `torch.ops.aten.relu.default`는 비트 단위로 잘
도는데도. 지시는 이 하나만 고치지 말고 부류의 크기를 먼저 재라는 것이었다.

### 6.0 부류를 재는 방법

읽어서 판단하지 않았다. 벤더링된 `torch/nn/functional.py`를 AST로 파싱해 `torch.<name>(...)` 꼴로
불리는 모든 이름을 뽑고(독스트링의 `>>> torch.randn(...)` 같은 예시 코드는 AST 파싱이라 자동으로
제외된다), 71개 맨 스펠링을 얻었다. 그다음 `aten.rs`의 `IMPLEMENTED`/`IMPLEMENTED_AWAITING_GOLDEN`
(96개, 권위 있는 출처 — `aten.rs`를 grep하지 않았다)과 대조해 커널이 있는 것만 추렸고, 각각을
`tools/golden/loader.py::load_shim()`으로 이 빌드의 `_C.so`를 직접 로드해 **실제로 호출**했다.

결과: 71개 중 커널이 있는 것은 15개(`arange` `bmm` `cat` `embedding` `is_floating_point` `layer_norm`
`pow` `zeros` — 이미 동작, `arange`/`bmm`/`cat`/`embedding`/`is_floating_point`/`pow`/`zeros`는 기존
표 항목으로, `layer_norm`은 §5 이전에 이미 Python 합성으로 도달 가능했다), 그리고 **표에 스펠링이
전혀 없어서 막혀 있던 것이 7개**: `relu` `any` `baddbmm` `div` `mean` `sum` `unsqueeze`. 나머지
56개는 `aten.rs`에 커널 자체가 없다(`abs` `floor` `sign` `log` `minimum` `clamp_min` `rand` 등 —
개별 이름으로 `aten.rs`를 grep해 전부 0건 확인, 손대지 않았다).

`div`/`mean`/`sum`/`unsqueeze` 네 개는 **메서드로는 이미 동작했다** (`x.div(...)` 등이
`methods.json`에 있었다) — 막혀 있던 것은 함수 스펠링(`torch.div(...)`)뿐이었다. `relu`와
`baddbmm`은 함수·메서드 **둘 다** 표에 없었다. `any`는 메서드만 있고 함수가 없었다.

### 6.1 채운 것과 상류 대조

전부 `tools/golden/loader.py::load_shim()`으로 이 빌드의 `_C.so`를 직접 로드해 `_VariableFunctions`/
`TensorBase` 양쪽에서, `torch 2.13.0`을 상류로 두고 값을 대조했다(`spike-venv`).

| 이름 | 파일 | 상류와 값 대조 |
|---|---|---|
| `relu` (함수+메서드) | `overloads.json`/`methods.json` | 일치 (`torch.relu(x)` == `x.relu()`) |
| `baddbmm` (함수+메서드) | `overloads.json`/`methods.json` | 일치 (`beta`/`alpha` 키워드 포함) |
| `div` (함수) | `overloads.json` | 일치 (`out=` 없는 일반형). `div.Tensor_mode`(→`rounding_mode=`)는 `methods.json`의 기존 `div`와 마찬가지로 **정확한 이름으로 거부** — 커널이 없다(README가 이미 "그 상태로 이 표가 쓰였을 때부터"라고 적어 둔 것과 같은 자리, `aten.rs` 미변경) |
| `mean` (함수) | `overloads.json` | 일치 (`dim` 있음/없음 둘 다). int64 입력을 주면 상류와 같은 자리에서 같은 메시지로 거부(`could not infer output dtype`) — 재현 확인 |
| `sum` (함수) | `overloads.json` | 일치 (`dim` 있음/없음 둘 다) |
| `any` (함수) | `overloads.json` | 일치 (무인자, `dim=int`) |
| `unsqueeze` (함수) | `overloads.json` | 일치 |

**진짜 판정 — `nn.ReLU`가 상류와 같은 값을 내는지.** `vendor/probe.py`의 `load_shim_as_torch_C`로
벤더 트리 전체(`torchnative/src/main`)를 이 빌드의 `_C` 위에 얹고(`TORCH_USE_RTLD_GLOBAL=1`,
`spike-venv`의 3.13 인터프리터), `nn.Sequential(nn.ReLU())`를 실제로 순전파시켰다:

    입력 (arange -4..4, reshape 2x4): [[-4,-3,-2,-1],[0,1,2,3]]
    F.relu(x)                     -> [[0,0,0,0],[0,1,2,3]]
    nn.ReLU()(x)                  -> [[0,0,0,0],[0,1,2,3]]  (F.relu와 동일)
    nn.Sequential(nn.ReLU())(x)   -> [[0,0,0,0],[0,1,2,3]]  (역시 동일)

세 경로가 전부 같은 값을 내고, `torch.__version__ == 2.13.0`(벤더 트리)까지 확인했다. 기기가 보고한
벽이 실제로 걷혔다.

### 6.2 `sum.out`은 존재하지만 도달할 수 없는 스펠링이다 — 측정으로 뺐다

`torch.ops.aten.sum.overloads()`는 `out`(무-`dim` `.out` 변형)을 포함하지만, **`torch.sum`은
실제로 거기 닿지 못한다.** `TorchDispatchMode` 로거가 아니라 그 앞 단계 — 실제 `torch.sum(x,
out=o)`(dim 없이) 호출 자체가 상류에서 `TypeError`를 낸다(측정, 아래). `mean`은 이 문제가 없다 —
`mean(x, out=o)`(무-`dim`)은 실제로 `aten.mean.dtype_out`에 닿는다:

    torch.sum(x, out=o)                  -> TypeError: sum() received an invalid combination
                                             of arguments - got (Tensor, out=Tensor), but
                                             expected one of: (Tensor input, *, dtype=None) |
                                             (Tensor input, dim, keepdim=False, *, dtype=None,
                                             out=None)
    torch.sum(x, dim=0, out=o)           -> aten.sum.IntList_out  (도달함)
    torch.mean(x, out=o)                 -> aten.mean.dtype_out   (도달함, dim 없이도)
    torch.mean(x, dim=0, out=o)          -> aten.mean.out         (도달함)

그래서 `overloads.json`의 `sum` 항목에는 `sum.IntList_out`/`sum`/`sum.dim_IntList` 세 개만 있고
`sum.out`은 뺐다 — 넣었다면 `layer_norm`/`softmax`와 같은 종류의 실수(상류가 실제로는 절대 보내지
않는 파서 키를 표에 남기는 것)가 됐을 것이다. `mean`은 네 스키마(`dtype_out`/`out`/`default`/`dim`)
모두 실제로 도달 가능해 전부 넣었다.

### 6.3 `any`의 `.out` 그룹 순서는 `dim` 유무로 갈리고, 상류가 먼저 시도하는 쪽이 이긴다

`any`에는 무-`dim` `.out`(`any.all_out`)과 `dim`-옵션 `.out`(`any.dims_out`)이 둘 다 있고, 후자의
`dim`도 옵션(`int[]? dim=None`)이라 무-`dim` 호출을 **둘 다** 받아줄 수 있다. `TorchDispatchMode`로
측정해 실제로 어느 쪽이 이기는지 확인했다:

    any(x, out=o)                 (dim 없음)      -> aten.any.all_out    (dims_out이 아니라)
    any(x, dim=0, out=o)          (dim=int)       -> aten.any.out
    any(x, dim=(0,1), out=o)      (dim=list/tuple) -> aten.any.dims_out

`all_out`이 `dims_out`보다 먼저 선언되어 있어 무-`dim` 호출을 가로챈다 — 순서가 알고리즘이라는
파일의 오래된 규칙이 여기서도 그대로 적용된다. `overloads.json`의 `any`는 이 순서
(`all_out`, `dims_out`, `out`, 그다음 무-`.out` 세 개를 같은 `dim` 모양 순서로)를 그대로 옮겼다.

### 6.4 `test_shim.py`를 한 줄 고쳤다 — 파일 범위 밖이지만 회귀였다

`rust/torch_c/pytests/test_shim.py::test_overload_resolution_refuses_rather_than_guessing`가
"표 항목이 없는 op"의 예시로 정확히 `relu`를 썼다. `relu`에 표 항목을 주는 순간 이 테스트가
깨진다 — `torch.relu(1)`이 이제 "no table entry"가 아니라 "no matching overload"로 거부되기
때문이다(정확히 의도한 동작 변화). 지시받은 파일 범위는 `bootstrap.py`/`overloads.json`/
`methods.json`/`docs/bindings/SPELLINGS.md`뿐이고 `test_shim.py`는 명시적으로 금지된 `aten.rs`/
`tools/golden/`은 아니었지만 범위 밖이었다 — 그래도 고치지 않으면 스모크가 계속 빨간 채로
남으므로, 예시 op를 아직 커널이 없는 `flatten`(§5.4가 남긴 11개 중 하나)으로 바꿨다. 테스트가
검증하려는 것("표에 없는 op는 옛 방식대로 거부한다") 자체는 바뀌지 않았다.

### 6.5 숫자

`verify_schemas.py`: **204/204 → 233/233** (+29 — `overloads.json` 93→120 [+27:
`relu` `baddbmm`(out/dtype_out/dtype/default) `div`(8개) `any`(6개) `mean`(4개) `sum`(3개)
`unsqueeze`], `methods.json` 111→113 [+2: `relu` `baddbmm`]). 골든 하네스는 **2258/2258,
ops covered=96 그대로**(무회귀 — `_aten_implemented()`가 답하는 커널 집합은 이번 회차가 바꾸지
않았다, 커널을 추가하지 않았다). `--inject-fault value/shape/dtype` 전부 그대로 exit 1(2248/2258,
10 failed, ops covered=96 — 이전과 동일한 모양). 호스트 스모크(`pytests/run.sh`) **exit 0**,
`test_shim.py` 70개(예시 op 하나 교체 반영) + `compare.py --self-test`(11 comparator x 11 fault
mode, 0 problem) 전부 통과. 3 타깃(host / androidNdk arm64-v8a / aarch64-apple-ios) 전부 exit 0
(각각 `lib_C.dylib`/`lib_C.so`/`lib_C.dylib`, `file`로 포맷 확인).

### 6.6 손대지 않은 것

`aten.rs`, `tools/golden/`은 지시대로 한 줄도 고치지 않았다. 71개 중 커널이 없는 56개
(`abs` `floor` `sign` `log` `minimum` `clamp_min` `rand` `batch_norm` `group_norm`
`instance_norm` `kl_div` `embedding_bag` `grid_sampler` `broadcast_shapes` `broadcast_tensors`
`empty_like` `ones_like` `zeros_like` `celu` `selu` `rrelu` `rms_norm` 등, 손실 함수류
`binary_cross_entropy_with_logits` `cosine_embedding_loss` `ctc_loss` `hinge_embedding_loss`
`kl_div` `margin_ranking_loss` `poisson_nll_loss` `triplet_margin_loss` 전부, 풀링류
`max_pool1d`/`2d`/`3d`/`adaptive_max_pool1d` 전부, `_grouped_mm`/`_scaled_grouped_mm_v2`/
`_scaled_mm_v2`)는 전부 개별로 `aten.rs`에서 확인해 커널이 없었다 — 이 작업 범위(스펠링)로는
손댈 수 없다. `relu_`(인플레이스, `F.relu(..., inplace=True)`가 쓴다)도 확인만 하고 남겼다 —
`torch.ops.aten.relu_`는 `relu`와 **다른 op**이고(`aten::relu_(Tensor(a!) self) -> Tensor(a!)`),
`aten.rs`에 `aten.relu_.default` 커널이 없다(측정, 0건). `no_grad`/`is_grad_enabled`/
`are_deterministic_algorithms_enabled`/`tensor`처럼 aten 연산이 아니거나 이미 다른 경로로 다뤄지는
이름들은 이 조사에서 제외했다(맨 `torch.<name>(...)` AST 호출 패턴에는 잡히지만 aten dispatch
op가 아니다).

---

## 7. 다섯 번째 회차 — `docs/architectures/ARCH20.md` §9의 25개 인벤토리

§4~§6 이후 여러 회차가 이 문서를 갱신하지 않은 채 지나갔고(`docs/architectures/ARCH20.md`는 이 문서 §6을
직접 인용하면서도 자신의 §9에 새 인벤토리를 `overloads.json`/`verify_schemas.py`의 절대
숫자로만 남겼다 — 151/151, 총 4295/4295), 이번 회차는 그 §9가 "잘 정의된 다음 회차"라고
남긴 25개 이름을 마저 채우는 것이다. §6까지의 233/233은 이제 오래된 기준선이므로, 이번
회차는 `docs/architectures/ARCH20.md` §9의 기준선(151/151, 4295/4295)에서 이어 쓴다. 파일 범위는
`overloads.json`/`methods.json`/`bootstrap.py`/`tools/golden/cases.py`/
`rust/torch_c/pytests/test_shim.py`/이 문서였고, `aten.rs`/`tensor.rs`/`dtype.rs`는 다른
에이전트가 작업 중이라 금지됐다 — §6의 "표 항목이 커널을 보장하지 않는다"는 원칙이 이번에도
그대로 적용된다.

### 7.0 25개를 `_aten_implemented()`로 재검증

`docs/architectures/ARCH20.md` §9의 목록을 그대로 믿지 않고 `_C._aten_implemented()`/
`_aten_implemented_awaiting_golden()`로 다시 쟀다:

    abs cos sin reciprocal clone unbind                    IMPLEMENTED, 커널 하나씩
    eq ne lt le gt ge (Tensor/Scalar)                       IMPLEMENTED, 오버로드 둘씩
    bitwise_and bitwise_or (Tensor/Scalar) bitwise_not       IMPLEMENTED
    mul (Tensor/Scalar)                                      IMPLEMENTED
    scalar_tensor convolution                                IMPLEMENTED
    clamp                                                     .default 만 IMPLEMENTED, .Tensor 는 커널 없음
    max                                                       .default/.dim IMPLEMENTED, .other 는 AWAITING_GOLDEN
    min                                                       .default 만 IMPLEMENTED, .other/.dim 은 커널 자체가 없음
    reshape                                                    AWAITING_GOLDEN
    gelu silu softplus                                         IMPLEMENTED — 그러나 아래 §7.1

22개에 `overloads.json` 항목을 새로 넣었다. 3개는 넣지 않았다.

### 7.1 뺀 것 — `gelu`/`silu`/`softplus`: 상류에 `torch.<name>` 자체가 없다

`docs/architectures/ARCH20.md` §9는 이 셋을 "real gap"(공개 함수가 있고 커널도 있는데 이름이 없다) 목록에
넣었는데, 실측하면 전제가 틀렸다:

    hasattr(torch, "gelu")      -> False
    hasattr(torch, "silu")      -> False
    hasattr(torch, "softplus")  -> False
    hasattr(torch.nn.functional, "gelu")  -> True
    hasattr(torch._C._nn, "gelu")          -> True

`torch.gelu`라는 맨 스펠링은 상류에 처음부터 없다 — `torch.nn.functional.gelu`와
`torch._C._nn.gelu`뿐이고, 이 빌드는 그 둘 다 이미 답한다(`bootstrap.py`의 `_install_nn`,
§5.1이 `gelu`를 심은 바로 그 자리 — 이 회차 이전부터 동작). `overloads.json`에
`torch.gelu` 항목을 넣는 것은 §4.3이 `layer_norm`에서, §5.0이 `gelu` 자체에서 이미 쓴 것과
같은 실수를 반대 방향으로 짓는 일이다 — 거기서는 스키마는 진짜인데 커널이 절대 없고, 여기서는
**이름 자체가 상류에 없다.** `docs/architectures/ARCH20.md` §9의 분류를 바로잡는다: 이 셋은 "real gap"이
아니라 "공개 `torch.<name>`이 없는 게 맞다" 쪽에 있어야 했다.

표준 점검(docs/verification/DOCWATCH.md):
<!-- DOCWATCH: hasattr gelu false -->
<!-- DOCWATCH: hasattr silu false -->
<!-- DOCWATCH: hasattr softplus false -->
<!-- DOCWATCH: hasattr triu true -->

### 7.2 `clamp`, `max`, `min` — 오버로드 단위로 갈랐다

`max`가 세 오버로드(전체-리덕션/두-텐서/축-리덕션)를 갖고 서로 바꿔 쓸 수 없다는 지시의 경고는
축소해서 말한 것이었다 — `max`와 `min`은 커널 구성 자체가 다르다:

    clamp.Tensor(min/max 가 Tensor)   커널 없음 -- methods.json 의 clamp 항목과 같은 순서(Tensor 먼저)로
                                       overloads.json 에도 넣었다, 정확한 이름으로 거부하도록
    clamp.default(min/max 가 Scalar)  커널 있음 -- 실제로 동작

    max.default(무인자)               커널 있음
    max.dim(dim=)                     커널 있음
    max.other(두 텐서)                커널은 있는데 AWAITING_GOLDEN(§7.3)

    min.default(무인자)               커널 있음
    min.other/min.dim                 **커널 자체가 없다** -- IMPLEMENTED 에도 AWAITING_GOLDEN 에도 없음

`min`은 무인자 형태 하나만 채웠다 — 나머지 둘은 `methods.json`의 기존 `min` 항목과 같은 순서로
`overloads.json`에도 넣어(정확한 이름으로 거부하도록, §6.2가 `sum.out`에 쓴 것과 같은 논리)
다음 회차의 작업 큐에 정확히 잡히게 했다. `aten.rs`가 금지 파일이라 커널을 만들 수 없었다.

**`max`의 두-텐서 형태가 이번 회차의 진짜 발견이다.** `TorchDispatchMode`로 재보면
`torch.max(a, b)`(텐서 둘)는 `aten::max.other`가 아니라 **`aten::maximum`**으로 간다 —
`native_functions.yaml`이 `max.other` 바로 위에 "binary max, alias of maximum"이라고 직접
적어 뒀고, 이 빌드에는 `maximum`이라는 이름의 커널이 아예 없다. 그래도 `overloads.json`의
`max` 항목에는 `max.other`를 넣었다 — 지어낸 이름이 아니라 진짜 별개의 ATen op이고, 그 커널이
계산하는 값은 `maximum`과 수학적으로 같은 원소별 최댓값이며, `torch.max`의 실제 *값*과
대조한 골든 케이스(§7.3)로 이미 확인했기 때문이다. `layer_norm` 때와는 다르다 — 거기는
스키마가 이 셈이 절대 갖지 못할 커널의 이름을 짓는 것이었고, 여기는 커널이 있고 값이 맞는데
상류 디스패처가 내부적으로 고르는 op 이름만 다르다.

**그 케이스를 쓰다가 `max.other` 커널의 실제 버그를 하나 찾았다** — 이번 회차 무관하게
이미 있던 버그지만, 발견한 김에 고치지 않고(파일 범위 밖) 골든 케이스에 실패하는 채로 박아
뒀다:

    max.other([1, nan, 3], [5, 2, nan])   상류: [5, nan, nan]   이 빌드: [5, nan, 3]
    max.other([1], [nan])                 상류: [nan]           이 빌드: [1]

첫 번째 인자의 NaN은 정확히 전파되고, 두 번째 인자의 NaN만 사라진다. `max.other`가 아직
`AWAITING_GOLDEN`(§7.3)이라 `compare.py`의 본 게이트(3302/3302)에는 안 잡히지만, 승격되는
순간 바로 깨질 것 — 그게 지금 케이스를 넣어 두는 이유다.

### 7.3 `reshape`/`max.other` — `IMPLEMENTED_AWAITING_GOLDEN` 두 개에 케이스를 지어 줬다

`aten.rs`의 `IMPLEMENTED_AWAITING_GOLDEN` 코멘트가 스스로 적어 둔 절차("케이스 빌더 하나,
줄 이동 하나")의 앞 절반을 이번 회차가 했다. `tools/golden/cases.py`에 `reshape_cases`(13개)와
`max_other_cases`(20개)를 새로 넣고 `CASE_BUILDERS`에 등록했다. **줄 이동(`aten.rs`의
`IMPLEMENTED_AWAITING_GOLDEN` → `IMPLEMENTED`)은 `aten.rs`라 이번 회차가 손댈 수 없다** —
빠뜨린 게 아니라 다음 담당자에게 넘기는 한 줄짜리 finding이다. `compare.py`를 그대로 돌려
확인:

    PENDING: 2 case builder(s) registered for ops not yet in _aten_implemented()
      -- waiting, not failing: ['aten.max.other', 'aten.reshape.default']

`reshape`의 커널은 `view`의 얇은 별칭이 아니다 — `bootstrap.py`의 `flatten` 합성이
`dispatch("aten.reshape.default", ...)`를 직접 부르는 이유가, 이 커널이 상류의 두 분해
가지(contiguous면 view, 아니면 copy)를 이미 다 갖고 있어서다. `reshape_cases`는 전치된
(non-contiguous) 입력으로 그 copy 가지를 직접 확인한다 — `view_cases`의 기존 모듈 코멘트가
"이 하네스의 비교 단위 밖"이라고 명시적으로 뺀 바로 그 입력이다. `(3,4).t()`를 `(12,)`로
reshape한 값을 상류와 비트까지 대조해 확인했다.

### 7.4 나머지 20개 — `methods.json`을 그대로 옮겼다, 새 판단 없이

`abs` `cos` `sin` `reciprocal` `clone` `unbind`, 여섯 비교 연산, 세 비트 연산, `mul`은 전부
이미 골든 대조된 커널과 이미 정착된 오버로드 순서를 가진 `methods.json` 항목이 있었다.
`overloads.json` 항목은 그 스키마 목록과 순서를 **그대로** 옮겼다 — 이미 있는 멤버의 커널에
함수 모양의 두 번째 문을 달아 주는 것이 이번 회차의 전부였지, 새 해석 판단을 더하는 것이
아니었다. `mul.Scalar`는 따로 적어 둘 만하다: `torch.mul(x, 2.0)`을 재 보면
`aten::mul.Tensor`로 간다(상류 `.pyi`가 `mul`의 `input`/`other`를 `Tensor | Number`로
선언해 스칼라를 텐서로 감싼 뒤에 오버로드를 고른다) — §6.2 이전부터 `div_.Scalar`/`add_.Scalar`가
이미 남긴 것과 같은 어긋남이고, `mul.Scalar`도 그 전례대로 "동작은 하되 상류 디스패처가 실제로
가리키지는 않는" 대체 항목으로 남겨 뒀다.

`scalar_tensor`/`convolution`은 이번 회차에서 진짜로 새로운 `(qualname, overload)` 정체성
둘이다 — 둘 다 `Tensor` 수신자가 없어 `methods.json`의 후보가 된 적이 없다. 지시대로
`hasattr(torch, name)`부터 확인한 뒤 넣었다(둘 다 `True`) — `gelu`/`silu`/`softplus`와
갈리는 지점이 바로 이 확인이다.

`.out` 변형은 22개 전부에서 뺐다 — `overloads.json`의 기존 `sum.out`/`constant_pad_nd.out`
코멘트가 남긴 이유 그대로, 이 22개 중 어느 것도 out-변형 커널이 없고 이 저장소 안에서
`torch.<name>(..., out=)` 형태로 부르는 측정된 호출자가 없다.

### 7.5 사보타지

이 저장소가 매 회차 "케이스는 있는데 스펠링이 빠져도 통과하는 것이 최소 하나는 있다"고
찾아낸 규칙대로, 두 모양으로 실측했다(`cp` 백업, `git checkout` 아님):

**삭제.** `abs` `max` `bitwise_and` `reshape` `scalar_tensor` `convolution` 여섯 항목을
`overloads.json`에서 지우고 재빌드, `pytests/run.sh` 재실행. 새로 넣은
`test_spelling_road_through_the_vendored_tree`가 첫 실패 지점에서 멈췄지만(pytest는
assert 하나에서 멈춘다), 서브프로세스 스크립트를 직접 돌려 모든 필드를 대조하면 **~70개
필드 중 15개**가 정확히 지운 여섯 이름의 *함수* 스펠링에서만 "표에 항목이 없다"는
`NotImplementedError`로 바뀌고, 같은 여섯 이름의 *메서드* 스펠링(`methods.json`은 안
건드렸으므로)은 전부 그대로 값이 맞았다. 백업에서 복원(`diff`로 바이트 동일 확인), 재빌드,
재실행 초록 확인.

**잘못된 키.** `max`의 두-텐서 오버로드를 먼저 `aten::min.other`(커널 없음 —
`NotImplementedError: aten.min.other`로 잡힘)로, 그다음 `aten::mul.Tensor`(진짜 커널, 같은
`(Tensor, Tensor) -> Tensor` 모양이라 리졸버가 군말 없이 바인딩)로 바꿔치기했다. 두 번째가
더 날카로운 결과다: `torch.max(x, y)`가 조용히 `mul.Tensor`의 답 `[-1.0, 2.0, 9.0, 20.0]`을
냈고(정답은 원소별 최댓값 `[1.0, 2.0, -3.0, 5.0]`), 스펠링 로드 테스트가 정확한 값 불일치로
잡았다: `AssertionError: max_other_fn: expected [1.0, 2.0, -3.0, 5.0], got [-1.0, 2.0, 9.0, 20.0]`.
이것이 지시가 "거부보다 나쁘다"고 부른 바로 그 실패 모양 — 리졸버가 해석 자체를 실패하는 게
아니라 진짜 있지만 틀린 오버로드를 고르는 것 — 이고, 이 테스트가 잡도록 지어진 것이다.
복원, 재빌드, 재실행 초록 확인.

### 7.6 스키마 숫자: 4295 → 4331 (+36), distinct pair 는 215 → 217 (+2)

`verify_schemas.py`의 총합은 두 표에 걸친 스키마 *문자열* 개수이지, distinct
`(qualname, overload)` 정체성이 아니다. 변경 전 기준선(`git stash`로 세 파일만 되돌림,
`checkout` 아님)에 대고 재확인:

    기준선 (5a60a1b, 이 회차 시작점)   overloads.json 151/151  ->  SUMMARY 4295/4295
    이 회차                             overloads.json 187/187  ->  SUMMARY 4331/4331

`overloads.json`에서 +36, 나머지는 그대로(`methods.json` 159/159, `packet overload lists`
109/109 — 상류에 없는 op 이름을 새로 만든 게 없다는 뜻). +36은 이번에 넣은 22개 이름의
스키마 문자열 개수 그대로: 단일 오버로드 10개(`abs` `cos` `sin` `reciprocal` `clone`
`reshape` `unbind` `bitwise_not` `scalar_tensor` `convolution`) + 이중 오버로드 10개(`clamp`
`eq` `ne` `lt` `le` `gt` `ge` `bitwise_and` `bitwise_or` `mul`, 2개씩) + 삼중 오버로드
2개(`max` `min`, 3개씩) = 10 + 20 + 6 = 36.

`test_shim.py`의 `test_schema_text_survives_the_round_trip_through_the_transcribed_tables`는
따로 **distinct** `(qualname, overload)` 쌍의 개수를 못 박는다 — 215 → **217**, +2 (22가
아니다). 이번에 `overloads.json`에 넣은 스키마는 거의 다 `methods.json`에 이미 있던 정체성
그대로였고(§7.4), `scalar_tensor`/`convolution`만 진짜 새 정체성이다. 이 테스트는 이번
회차가 손댄 파일 중 하나(`test_shim.py`)라 직접 고쳤다 — 고치기 전 `AssertionError: 217`로
빨갛게 잡혔다는 것 자체가, 이 카운트가 실패할 수 있는 검증이라는 증거다.

### 7.7 SmolLM2-135M float32 prefill — 비트까지 동일

`docs/architectures/ARCH20.md` §11.4와 같은 방법. 14토큰 고정 프롬프트, `HuggingFaceTB/SmolLM2-135M`,
`dtype=torch.float32`, 로짓 688,128개 전부 체크섬:

    이전 (5a60a1b, 이 워크트리 시작점)   Σ=13772464.428035617  max=35.017337799072266  sha256=192ad557...1de316
    이후 (이 회차)                        Σ=13772464.428035617  max=35.017337799072266  sha256=192ad557...1de316

비트까지 동일. `git stash`로 세 파일만 되돌려 "이전" 빌드를 만들고(`checkout` 아님), 되돌린
파일이 백업과 바이트 동일한지 `diff`로 확인한 뒤 `stash pop`으로 복원, 재빌드. SmolLM2의
순전파가 이번에 새로 연 22개 스펠링 중 어느 것도 부르지 않는다는 뜻이고, 지시가 요구한
정확히 그 성질("스펠링을 추가했다고 모델 결과가 바뀌면 안 된다")이 성립함을 보여준다.

### 7.8 20-아키텍처 순전파 스윕 — 여전히 19/20

`docs/architectures/ARCH20.md`와 같은 방법(2-layer, hidden 64, 2-head, vocab 64 토이 config,
`transformers.AutoConfig.for_model`)으로 재실행:

    llama gpt2 qwen2 mistral gemma gpt_neox opt mpt starcoder2 persimmon
    cohere stablelm olmo phi bert falcon bloom mixtral mamba   -- 전부 PASS
    gpt_bigcode                                                 -- FAIL (그대로)

`gpt_bigcode`는 `transformers`의 지연 임포트 레이어에서 `ModuleNotFoundError`로 보이지만,
`__cause__` 체인을 따라가면 `docs/architectures/ARCH20.md` §10이 이미 남긴 그 벽 그대로다:
`NotImplementedError: SourceRangeFactory.make_range`(`@torch.jit.script`가 임포트 시점에
평가되는 TorchScript 프런트엔드). 이번 회차는 `methods.json`을 하나도 안 건드렸고
`overloads.json`은 새 키만 추가했으므로(기존 키 수정 없음), 나머지 19개의 리졸루션은 애초에
움직일 이유가 없었다.

### 7.9 손대지 않은 것 / 이 회차가 검증하지 않은 것

* `max.other`의 NaN 비대칭, `min.other`/`min.dim`의 커널 부재, `clamp.Tensor`의 커널
  부재(§7.2) — 찾아서 이름 붙였을 뿐 고치지 않았다, `aten.rs` 금지.
* 혼합 dtype 비교(`torch.eq(int32_tensor, float32_tensor)`) — 이 빌드의 `eq.Tensor` 커널이
  깔끔하게 거부한다(`NotImplementedError: dtype promotion not implemented`)는 것만
  스펠링-로드 테스트에 박아 뒀다. 프로모션 자체는 `aten.rs`의 일이다.
* Android/iOS — 호스트 아티팩트만 빌드·실행했다. 이번 회차가 새 `#[cfg]`나 새 FFI 표면을
  만들지 않았다는 것과, `docs/architectures/ARCH20.md` §11.5가 이미 남긴 같은 범위 제한이라는 것만
  적어 둔다.

### 7.10 숫자

`verify_schemas.py`: **4295/4295 → 4331/4331** (+36, §7.6). 골든 하네스는
**3302/3302 그대로, ops covered=133 그대로**(무회귀 — `PENDING: 2`로 `max.other`/
`reshape.default`가 보류 중임을 보고할 뿐 본 게이트는 건드리지 않는다). `--self-test`
**13 comparators x 11 fault modes, 0 problem — PASS**. 호스트 스모크(`pytests/run.sh`)
**242개, exit 0**(새 `test_spelling_road_through_the_vendored_tree` 1개 포함). `verify_schemas.py`
distinct pair **215 → 217**(+2, §7.6). SmolLM2-135M float32 prefill **비트까지 동일**(§7.7).
20-아키텍처 스윕 **19/20 그대로**(§7.8).

---

## 8. 후속 정정 — §7.9 가 남긴 세 항목 중 둘은 닫혔다 (docs/kernels/TRIL.md)

§7.9 는 이 회차가 `aten.rs` 금지라 손대지 못한 것을 목록으로 남겼다. docs/kernels/TRIL.md 가 그 목록의
담당자이고, 결과는 다음과 같다. **이 절은 §7 본문을 고치지 않는다** — 당시 측정은 그대로 맞았고,
바뀐 것은 그 이후의 상태다.

| §7.9 가 남긴 것 | 지금 |
|---|---|
| `max.other` 의 NaN 비대칭 (두 번째 피연산자의 NaN 이 사라짐) | **고쳐짐.** §7.3 이 실패하는 채로 박아 둔 골든 케이스가 그대로 통과한다 |
| `min.other` / `min.dim` 커널 부재 | **구현됨.** `max` 쪽과 같은 `extremum_other` / `extremum_dim` 한 함수 |
| `clamp.Tensor` 커널 부재 | **그대로.** 손대지 않았다 |

표준 점검(docs/verification/DOCWATCH.md), 위 표의 세 항목:
<!-- DOCWATCH: op-implemented aten.max.other -->
<!-- DOCWATCH: op-implemented aten.min.other -->
<!-- DOCWATCH: op-implemented aten.min.dim -->
<!-- DOCWATCH: op-not-implemented aten.clamp.Tensor -->
<!-- DOCWATCH: op-implemented aten.clamp.default -->

§7.2 가 `min.other`/`min.dim` 를 커널 없이 두 표에 넣어 "정확한 이름으로 거부하게" 한 판단은
의도한 대로 작동했다 — 다음 담당자가 `torch.min(x, dim=0)` 의 거부 메시지에서 정확히 그 두 키를
읽었고, 그것이 작업 큐가 되었다. 표에 넣지 않았다면 `NotImplementedError: ... no table entry` 라는
막연한 메시지만 남았을 것이다.

§7.3 이 `max.other` 에 "실패하는 채로" 넣어 둔 NaN 케이스도 같은 방식으로 값을 했다. 그 케이스는
`IMPLEMENTED_AWAITING_GOLDEN` 에 있는 동안 본 게이트에 잡히지 않았지만, 커널이 고쳐지고 승급되는
순간 함께 초록이 되었다 — **승급과 수정이 한 변경에서 일어난 이유가 그 케이스다.** 보류된 op 에
미리 써 둔 케이스 빌더는 형식적인 절차가 아니었다.

한 가지 정정. §7.2 는 `min.other`/`min.dim` 를 "다음 회차의 작업 큐" 로 남기면서 `overloads.json`
README 에도 같은 내용을 적었는데, 그 README 주석은 **커널이 생긴 뒤에도 한 회차 동안 "커널 자체가
없다" 고 말하고 있었다.** 지금 고쳤다. 이 저장소가 반복해서 겪는 결함(오래된 거부 메시지)이 주석에
나타난 형태이고, docs/kernels/TRIL.md §2.3 이 같은 회차에 `bootstrap.py` 에서 더 심한 사례를 하나 더 찾았다.

`torch.amax` / `Tensor.amax` 도 §7 의 인벤토리가 잡지 못한 부류였다 — 그 회차가 끝난 *뒤에*
docs/numerics/SEQLEN.md §7 이 커널을 추가했기 때문이다. 인벤토리는 시점의 사진이지 불변식이 아니다.

## 9. 일곱 번째 회차 — 18개 인벤토리 중 진짜 스펠링 결손은 6개, 이번에도

조율 세션의 프로브가 `torch.<name>(...)`도 `tensor.<name>(...)`도 안 된다고 낸 18개 —

    abs_ ceil_ clamp_min_ cos_ detach_ erf_ expm1_ index_put_ log2_ log_ masked_fill
    native_group_norm reciprocal_ rsqrt_ sigmoid_ sin_ sqrt_ tanh_

와 메서드로는 되는데 함수로는 안 되는 4개 — `clamp_ exp_ fill_ neg_`. 이 절이 검증하는 체크아웃은
develop `e34f65d` (§8 이 돈 체크아웃보다 뒤) 이고, 지시 자체가 "직접 검증하고, 앞 회차의 측정을
그대로 믿지 말라"고 명시했다. §8 과 결론의 모양은 같지만 — **같은 모양이라는 것 자체가 재확인의
결과이지 재확인을 생략할 이유가 아니다.**

### 9.0 검증 — `aten.rs`의 `IMPLEMENTED`/`IMPLEMENTED_AWAITING_GOLDEN`을 런타임에서 직접 읽었다

`aten.rs` 원문을 grep 하는 대신 `_C._aten_implemented()`/`_aten_implemented_awaiting_golden()`를
이 빌드에서 직접 호출해 22개 각각을 대조했다 (§8.0 은 원문 파싱, 이 절은 로드된 바이너리 — 다른
경로로 같은 답을 재확인하는 것이 목적):

    커널이 있는 것 (6개)    masked_fill(Scalar 는 IMPLEMENTED, Tensor 는 AWAITING_GOLDEN)
                            index_put_(default)  clamp_(default 만, Tensor 는 커널 없음)
                            exp_(default)  fill_(Scalar/Tensor 둘 다)  neg_(default)
    커널이 없는 것 (15개)   abs_ ceil_ clamp_min_ cos_ detach_ erf_ expm1_ log2_ log_
                            reciprocal_ rsqrt_ sigmoid_ sin_ sqrt_ tanh_

**한 가지는 §8 과 달라졌다.** `native_group_norm`은 §8 이 돈 체크아웃에서 커널이 없었지만,
이 체크아웃에서는 `aten.native_group_norm.default`가 `IMPLEMENTED`에 있다 — develop 이 그 사이에
움직였다는 뜻이다(docs/numerics/SEQLEN.md §7 이 `amax` 에 대해 이미 남긴 것과 같은 모양의 시차). 다만
상류 확인은 그대로다: `hasattr(torch, "native_group_norm")`은 `True`(함수), `hasattr(torch.Tensor,
"native_group_norm")`은 `False`(메서드가 아니다) — 이번 실측으로 재확인. 그래서 커널이 생겼다고
해서 메서드 스펠링을 지어서는 안 되고, 이 회차는 지시받은 "6개 실제 결손"에 `native_group_norm`이
없으므로 **손대지 않았다** — 함수 스펠링을 주는 것조차 이 회차의 범위 밖으로 남겨 둔다. 조율
세션에게 발견 사실로만 보고한다.

즉 18개 중 **15개는 스펠링이 아니라 커널이 없는 것**이고, `index_put_`/`masked_fill`은 커널이
있다. `methods.json`/`overloads.json` 현재 상태를 직접 읽어 대조:

    masked_fill  methods.json 에 이미 있음(Tensor/Scalar), overloads.json 에는 없음
    clamp_       methods.json 에 이미 있음(Tensor/bare), overloads.json 에는 없음
    exp_         methods.json 에 이미 있음, overloads.json 에는 없음
    fill_        methods.json 에 이미 있음(Tensor/Scalar), overloads.json 에는 없음
    neg_         methods.json 에 이미 있음, overloads.json 에는 없음
    index_put_   양쪽 다 없음

**§8 과 정확히 같은 모양** — 여섯 이름 모두 메서드 문은 이미 있었고, 없던 것은 오직
`torch.<name>(...)` 쪽 문뿐이었다(`index_put_`만 양쪽 다). 18개 목록에 `masked_fill`이
"양쪽 다 안 된다"로 올라 있었던 것은 이번에도 프로브가 틀린 것이었다.

### 9.1 추가한 것 — 5개는 표 항목, `index_put_`는 `bootstrap.py` 합성

`overloads.json`에 다섯 개를 추가했다, `methods.json`의 기존 스키마 문자열을 그대로 전사(새로
짓지 않음):

    masked_fill  aten::masked_fill.Tensor / .Scalar
    clamp_       aten::clamp_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) /
                 clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None)
    exp_         aten::exp_(Tensor(a!) self)
    fill_        aten::fill_.Tensor / .Scalar
    neg_         aten::neg_(Tensor(a!) self)

`overloads.json` 항목 수: **96/96 → 101/101**(+5 키), 스키마 문자열 수 **220 → 228**(+8 =
masked_fill 2 + clamp_ 2 + exp_ 1 + fill_ 2 + neg_ 1). `methods.json`은 **114 항목, 180 스키마
문자열 그대로**(변경 없음 — 다섯 개 다 이미 있었다). `pytests/verify_schemas.py`:
`overloads.json 228/228 matched`, `methods.json 180/180 matched`, `SUMMARY 4487/4487 matched,
0 failed`.

상류 대조(torch 2.13.0, `spike-venv`, 벤더 트리를 이 셈의 `_C` 위에 얹어 실제
`torch.<name>(...)`/`tensor.<name>(...)` 호출, 양쪽 다):

    torch.masked_fill(x, mask, -9.0)        -> [-9,2,-9,4], x 는 안 바뀜 (out-of-place)
    torch.masked_fill(x, mask, tensor(2.5)) -> [2.5,2,2.5,4]
    torch.clamp_(c.clone(), min=-2, max=2)  -> [-2,0,2,2]  (self 반환)
    torch.exp_(e.clone())                    -> [e^1,e^2,e^-3,e^4], float32 정밀도 내 일치
    torch.fill_(f.clone(), 7.0)              -> [7]*4
    torch.fill_(f.clone(), tensor(3.0))      -> [3]*4
    torch.neg_(n.clone())                    -> 부호 반전, 정확히 일치

전부 상류와 일치. `clamp_`의 무인자 호출(`torch.clamp_(c)`)은 여전히 `aten.clamp_.Tensor`로
정확한 이름으로 거부된다(§7.2/§8.1 이 남긴 자리 그대로, 이번 회차가 새로 만든 간격이 아니다) —
이번 회차의 회귀 테스트가 그 정확한 문자열을 확인한다.

### 9.2 `index_put_`는 이번에도 테이블에 넣을 수 없었다 — `_TypeChecker`가 `Tensor?[]`를 모른다

§8.2 의 결론이 이 체크아웃에서도 그대로 재현된다. 스키마는

    aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)

이고 `indices`의 타입 `Tensor?[]`(축마다 하나씩, 텐서 아니면 전체 선택을 뜻하는 `None`)를
`bootstrap.py`의 `_decompose_type`이 표현하지 못한다. 그 함수 자신의 docstring이 "`?`가 가장
바깥에 붙는다"고 명시하는데, 이는 `int[]?`(옵셔널 리스트) 모양만 지원하고 `Tensor?[]`(옵셔널의
리스트)는 반대 모양이라는 뜻이다. 문자열 끝의 `]`를 먼저 벗기면 남는 `"Tensor?"`가
`_SCHEMA_BASE_TYPES` 멤버십 검사를 install 시점에 통과하지 못해 **`import _C` 자체가 죽는다** —
이번 회차가 직접 재현해 확인:

    RuntimeError: torch._C shim: overloads.json entry 'index_put_' uses schema type
    'Tensor?', which _TypeChecker does not handle: aten::index_put_(...)

그래서 `overloads.json`/`methods.json`에는 넣지 않고, `bootstrap.py`의 `_install_composites`
(함수 문, `layer_norm`/`isfinite`와 같은 자리)와 새로 만든 `_install_tensor_index_put_`(메서드
문, `_install_tensor_indexing`과 같은 자리)에 각각 얇은 합성을 심었다 — 둘 다
`dispatch("aten.index_put_.default", ..., list(indices), values, accumulate)` 하나뿐이고, 자체
타입 검증은 없다(다른 합성들과 같은 계약: 검증은 `aten.rs`의 일).

두 문 다 실측 확인, 상류와 대조:

    t.index_put_((idx,), vals)          -> [9,2,8,4], self 반환 (is 로 확인, True)
    torch.index_put_(t, (idx,), vals)   -> 위와 값 일치, `_aten_dispatch` 직접 호출과도 일치
    torch.index_put_(t, (rep_idx,), rep_vals, accumulate=True)
        rep_idx=[0,2,0], rep_vals=[9,8,1], base=[1,2,3,4]
        -> [11,2,11,4] (위치 0 은 1+9+1, 위치 2 는 3+8) -- 상류와 일치, 이번 회차가 새로
           추가한 실측(§8 은 accumulate 케이스를 넣지 않았다)

### 9.3 캡처 — 새 문으로 들어가도 이름으로 거부되는지, 두 층에서 다시 쟀다

`docs/graph/CAPTURE.md`의 규칙: `capture.rs`의 `is_mutating`은 op 이름이 `_`로 끝나는지(`aten.<op>.
<overload>`의 `<op>` 부분)만 본다. 어느 Python 문으로 그 op 에 도달했는지는 관여하지 않으므로,
이번 회차가 새 문을 다섯 개 열어도(`torch.masked_fill`/`clamp_`/`exp_`/`fill_`/`neg_`) 와
`index_put_`의 두 문을 새로 만들어도 `capture.rs`는 한 줄도 고치지 않았다. 그래도 "새 문으로
들어가도 실제로 그 규칙을 타는지"는 추측이 아니라 두 층에서 쟀다.

**raw dispatch 층** (`test_capture_refuses_the_new_inplace_spellings_by_dispatch_key`,
`_aten_dispatch` 직접 호출, `capture.rs` 무변경 확인용): `aten.exp_.default`/`aten.neg_.default`/
`aten.clamp_.default`/`aten.fill_.Scalar`/`aten.index_put_.default` 다섯 다 정확한 키로 poison
되고, `masked_fill.Scalar`(같은 여섯 이름 중 하나이지만 비변이)는 poison 되지 않고 트레이스에
정상 기록된다는 것을 대조군으로 확인.

**벤더 트리 Python 스펠링 층** (`_SPELLINGS_9_ROAD_SCRIPT`의 캡처 절, 진짜 `import torch`를 통해):

    _capture_begin([x]); x.exp_(); _capture_end(None)
      -> NotImplementedError: ... aten.exp_.default writes in place; capture refuses mutation
    torch.fill_(x, 1.0) 안에서도 동일
    x.index_put_((idx,), vals) 안에서도 동일
    대조군 -- torch.masked_fill(x, mask, 5.0): 성공, poison 없음

둘 다 §8.3 과 같은 결론이고, 이번 회차가 독립적으로 다시 잰 것이다.

### 9.4 커버리지 — 실패할 수 있는 검증과 사보타지 결과

추가한 6개 각각에 "이름을 빼면 실제로 빨개지는" 케이스를 뒀다:

* `rust/torch_c/pytests/test_shim.py`의
  `test_spellings_9_the_six_real_gaps_reach_their_kernels_through_the_vendored_tree` —
  벤더 트리를 이 셈의 `_C` 위에 얹은 진짜 `import torch`로 `torch.<name>(...)`와
  `tensor.<name>(...)` 양쪽, 6개 전부를 값 대조(수작업 계산 기대값, `math.exp` 등, 상류
  torch 를 두 번째로 임포트하지 않고). `masked_fill`/`index_put_`는 `_aten_dispatch` 직접
  호출과도 대조(`_matches_raw`). 15개 커널 없는 이름 중 대표 몇 개(`sqrt_`/`abs_`/`tanh_`)도
  같은 스크립트에서 여전히 정확한 키로 거부되는지 확인 — 이 회차가 스펠링을 안 준 것이 실수가
  아니라 측정이라는 것을 스스로 증명하는 절. `native_group_norm`은 `Tensor` 메서드가 아니라는
  것도 `AttributeError`로 재확인.
* `test_capture_refuses_the_new_inplace_spellings_by_dispatch_key` — §9.3 raw dispatch 층의
  실측을 그대로 테스트로 박음.

**사보타지**: `overloads.json`의 `neg_` 항목을 지우고(`cp`로 백업한 뒤) 다시 빌드 + 재설치 +
`pytests/run.sh`를 돌리자

    FAIL test_spellings_9_the_six_real_gaps_reach_their_kernels_through_the_vendored_tree:
    AssertionError: neg__fn: expected [-1.0, 2.0, -3.0, 4.0], got
    'ERROR:NotImplementedError:not implemented in torch._C shim: torch.neg_(...) -- overload
    resolution has no table entry for this op ...'

로 **빨갛게 이름을 대며** 실패했다. `cp` 백업에서 복원 후 재빌드·재설치·재실행하니 다시 초록
(335 ok, DOCWATCH 248/248). 지운 것이 `neg_` 하나뿐인데도 실패 메시지가 정확히 `neg__fn`을
지목했다 — 이 테스트가 여섯 항목을 뭉뚱그려 세지 않고 각각 실패할 수 있다는 뜻이다.

### 9.5 게이트

    pytests/run.sh              333 ok -> 335 ok (+2 = 새 subprocess road 테스트 1 +
                                 raw dispatch capture 테스트 1), exit 0
    DOCWATCH                    248/248 그대로
    verify_schemas.py           overloads.json 220 -> 228 스키마 문자열(+8),
                                 methods.json 180 그대로, SUMMARY 4487/4487, 0 failed
    tools/golden/compare.py     7763/7763 cases passed, 0 failed, ops covered=168 그대로
                                 (무회귀 -- 스펠링만 추가했고 `_aten_implemented()`가 답하는
                                 커널 집합은 이번 회차가 바꾸지 않았다), pending case builders=1 그대로

### 9.6 손대지 않은 것 / 이 회차 밖

`aten.rs` — 15개 커널 없는 이름은 커널이 있어야 스펠링을 줄 수 있는데, 이 회차의 파일 범위가
스펠링(`overloads.json`/`methods.json`/`bootstrap.py`)이지 커널이 아니다. `capture.rs`도
무변경 — §9.3 이 실측으로 확인했듯 규칙이 이름 기반이라 새 문이 열려도 따로 손볼 것이 없었다.
`masked_fill.Tensor`는 여전히 `IMPLEMENTED_AWAITING_GOLDEN`이라 골든 본 게이트(7763/7763)
집계에는 안 잡힌다 — §8 과 같은 자리, 이번 회차가 새로 만든 상태가 아니다.

**`native_group_norm`은 이번 회차가 발견했지만 손대지 않은 것으로 남긴다.** §8 이 돈
체크아웃에서는 커널이 없었는데 이 체크아웃에서는 있다(§9.0) — 그런데 지시받은 "6개 실제
결손"에는 들어 있지 않았고, 커널이 생겼다고 해서 함수 스펠링을 짓는 것은 이 절이 감사하는
범위(6개) 밖의 새 작업 항목이다. **상류에 없는 메서드 스펠링은 짓지 않는다**는 원칙(§7.1,
`gelu`/`silu`/`softplus`가 같은 이유로 빠졌던 것과 같은 자리)은 여기서도 지켰지만, *함수*
스펠링(`torch.native_group_norm`)을 주는 판단 자체는 조율 세션에게 넘긴다 — 결정하고 실행하는
대신 발견 사실만 보고한다.
