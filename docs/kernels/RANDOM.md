# RANDOM — `torch.randn`/`torch.rand`가 왜 없었고, 어떻게 채웠는가

`torch.randn(4, 4)`와 `torch.rand(2, 2)`는 README 배너에 실릴 만큼 가장 먼저 실행되는 호출인데,
`NotImplementedError: ... overload resolution has no table entry`로 죽었습니다. 0.0.2a0 휠에서도
소스 트리에서도 동일하게 재현되므로 회귀가 아니라 최초부터 없던 스펠링입니다. 이 문서는 무엇이
없었는지, 왜 `overloads.json` 항목으로 채울 수 없는지, 그리고 시드 스트림이 upstream과 정확히
일치한다고 어떻게 실측했는지를 적습니다.

---

## 1. 왜 `overloads.json` 항목이 아닌가

`overloads.json`의 값은 "`aten::<op>` 스키마 문자열 목록"이고, 각 항목은 `aten.rs`의 `dispatch`
`match` 문에 실제로 존재하는 커널 키를 가리켜야 합니다(그렇지 않으면 `pytests/verify_schemas.py`가
잡습니다 — 그 스크립트가 세는 4203개가 정확히 이 두 테이블의 항목 수입니다). `aten.rs`에는
`aten::randn`도 `aten::rand`도 커널이 없습니다. 있는 것은 셋뿐입니다:

- `aten.empty.memory_format` — 초기화되지 않은 텐서를 만든다
- `aten.uniform_.default` — 제자리에서 `[from, to)` 균등분포로 채운다
- `aten.normal_.default` — 제자리에서 정규분포로 채운다

따라서 `randn`/`rand`를 `overloads.json`에 넣는 것은 존재하지 않는 커널 키를 이름 붙이는 일이
됩니다(`_README`가 `layer_norm`에 대해 이미 경고하는 것과 같은 모양의 실수). 대신 이 둘은
`bootstrap.py`의 `_install_composites` 안에서 **파이썬 레벨 합성**으로 구현했습니다 —
`dropout`/`layer_norm`/`isfinite`와 정확히 같은 자리, 같은 이유입니다.

## 2. 합성이 upstream과 같은 수열을 낸다는 근거

torch 2.13.0 자신의 C++ 구현도 같은 합성입니다. `TorchDispatchMode` 로거로 확인하면
`torch.randn(4, 4)`는 `aten.randn.default` **한 번**만 기록되어 융합 커널처럼 보이지만, 시드를
맞춘 두 번의 호출을 값으로 비교하면 합성과 융합 커널이 **비트 단위로 같습니다**:

```python
torch.manual_seed(0); a = torch.randn(4, 4)
torch.manual_seed(0); b = torch.empty(4, 4).normal_(0., 1.)
torch.equal(a, b)  # True
```

같은 방법으로 다음 다섯 조합을 모두 실측했고(스크립트는 `/Volumes/macMini/caches/` 아래, 커밋 대상
아님), 전부 `True`였습니다:

| upstream 호출 | 합성 |
|---|---|
| `torch.randn(*size)` | `torch.empty(size).normal_(0., 1.)` |
| `torch.rand(*size)` | `torch.empty(size).uniform_(0., 1.)` |
| `torch.rand_like(x)` | `torch.empty_like(x).uniform_(0., 1.)` |
| `torch.randn_like(x)` | `torch.empty_like(x).normal_(0., 1.)` |
| `torch.normal(mean=0., std=1., size=s)` | `torch.empty(s).normal_(mean, std)` |

`uniform_`/`normal_`은 이미 upstream과 비트 단위로 일치한다고 측정되어 있으므로(`docs/numerics/RNG.md`),
이 표는 "합성 자체가 옳다"가 아니라 "합성이 upstream이 실제로 하는 일과 같다"는 것을 보이는
목적입니다 — `aten.rs`를 건드리지 않고 `randn`/`rand`를 구현할 수 있는 유일한 근거입니다.

### 2.1 `torch.normal`의 네 오버로드

`torch.ops.aten.normal`은 스키마가 넷입니다(`.out` 변형 넷은 별도): `Tensor_float`,
`float_Tensor`, `Tensor_Tensor`, `float_float`. 뒤의 셋 다 실측으로 확인했습니다 — `mean`,
`std` 중 하나 이상이 Tensor일 때는 표준정규 `n`을 뽑아 `mean + std * n`으로 아핀 변환한 것이
upstream과 비트 단위로 같습니다:

```python
torch.manual_seed(0); a = torch.normal(mean_t, std_t)
torch.manual_seed(0); n = torch.empty(mean_t.shape).normal_(0., 1.)
torch.equal(a, mean_t + std_t * n)  # True
```

**`n`을 뽑는 크기는 `mean`도 `std`도 아니라 둘의 브로드캐스트 결과입니다.** `mean.shape=(3,1)`,
`std.shape=(1,4)`이면 출력은 `(3,4)`이고, `n`도 `(3,4)`에서 뽑아야 값이 맞습니다 — `mean.shape`인
`(3,1)`에서 뽑으면 값이 upstream과 어긋납니다(실측, 이 문서를 쓰며 처음에 그렇게 짰다가 걸렸습니다).
`bootstrap.py`의 `_broadcast_shape`가 이 계산을 담당하며, `aten.rs`를 호출하지 않는 순수 파이썬
산술입니다(브로드캐스트 형태를 알아야 표준정규를 뽑을 텐서를 만들 수 있으므로, `mul`/`add`가
자체적으로 하는 브로드캐스트보다 먼저 필요합니다).

`mul`/`add` 쪽은 `overloads.json`을 거치지 않고 `dispatch("aten.mul.Tensor", ...)` /
`dispatch("aten.add.Scalar", ...)`처럼 직접 호출합니다 — `isfinite` 합성이 이미 하는 것과 같은
패턴이며, `aten.mul.Tensor`/`aten.mul.Scalar`/`aten.add.Tensor`/`aten.add.Scalar`는 이미
`aten.rs`에 커널이 있습니다(`torch.mul`/`torch.add`라는 *스펠링*이 `overloads.json`에 없는 것과
무관합니다).

## 3. 지원하는 kwarg, 거부하는 kwarg

upstream이 받는 kwarg는 `dtype`, `device`, `layout`, `requires_grad`, `out`, `generator`,
`pin_memory`입니다.

| kwarg | 처리 |
|---|---|
| `dtype`, `layout`, `device` | 이미 검증된 `varfns.empty`/`varfns.empty_like`로 그대로 전달. 정수 `dtype`은 그 경로 끝의 `normal_`/`uniform_`이 이름을 대며 거부(`test_rng_ops_refuse_integer_tensors`와 같은 거부) |
| `generator` | `TensorBase.normal_`/`uniform_`로 전달. `torch.default_generator`가 아니면 거부하는 것도 그쪽이 이미 함 |
| `requires_grad=True` | `varfns.empty`가 이미 거부(`_strip_python_only_kwargs`). `False`는 무해하게 버려짐 |
| `pin_memory=True` | **이름을 대며 거부** — `aten.rs`의 `empty.memory_format`/`empty_like`는 `pin_memory`가 조금이라도 명시되면 위치 기반으로 거부한다(`reject_unsupported`). `pin_memory=False`(기본값)는 아예 전달하지 않아 이 거부에 걸리지 않게 함 — 이건 새로 만든 문제였다: 처음엔 `pin_memory=pin_memory`를 항상 넘겼더니 아무도 `pin_memory`를 말하지 않은 평범한 `torch.randn(4, 4)`까지 "pin_memory not implemented"로 죽었다(§4) |
| `out` | **이름을 대며 거부.** upstream은 `out=`을 준 텐서를 요청 크기로 **리사이즈**한다(`torch.randn(4, 4, out=torch.empty(2, 2))`는 `(4, 4)`를 반환, 실측). `aten::empty.out`도 범용 `resize_`도 `aten.rs`에 없으므로 잘못된 크기로 계산하거나 조용히 무시하는 대신 거부한다 |

`torch.normal`은 `size=`가 없고 `mean`/`std` 중 하나가 Tensor일 때 `dtype`/`layout`/`device`/
`pin_memory`를 받으면 이름을 대며 거부한다 — upstream 스키마에도 그 조합엔 그 kwarg들이 없다
(`Tensor_float`/`float_Tensor`/`Tensor_Tensor`는 `generator`, `out`만 받는다).

## 4. `pin_memory=False`를 늘 전달했다가 걸린 함정

첫 구현은 `varfns.empty(size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory,
requires_grad=requires_grad)`처럼 `pin_memory`(기본값 `False`)를 항상 넘겼습니다. `_bind`의
기본값 정리 로직은 값이 **스키마 기본값과 같을 때만** 지웁니다 — `pin_memory`의 스키마 기본값은
`None`이지 `False`가 아니므로, "아무도 pin_memory를 말하지 않았다"는 `False`가 그대로
`aten.rs`까지 도달해 `reject_unsupported`에 걸렸습니다. `requires_grad`는 이 문제가 없는데,
`requires_grad`는 애초에 아텐 스키마 인자가 아니라 `_strip_python_only_kwargs`가 별도로 다루는
파이썬 전용 키워드라서 `False`를 명시적으로 버리기 때문입니다. 고친 뒤에는 `pin_memory`를 참일
때만 넘깁니다 — `docs/kernels/RANDOM.md`를 쓰는 동안 `rust/torch_c/pytests/test_shim.py`에서
`test_randn_and_rand_are_wired_rather_than_refused` 등 다섯 개가 바로 이 이유로 빨갛게 실패하는
것을 보고서야 발견했습니다.

## 5. 검증

- `rust/torch_c/pytests/test_shim.py` — `randn`/`rand`/`rand_like`/`randn_like`/`normal`을
  다루는 섹션(순수 shim 동작 + `_upstream_torch`가 있을 때의 시드-스트림 비트 비교 둘 다).
- `tools/golden/compare.py`는 늘지 않는다 — `_C._aten_implemented()`에 새 aten 키가 생기지
  않았으므로(합성은 이미 있는 `empty`/`uniform_`/`normal_`/`mul`/`add`만 쓴다), 이 하네스가 보는
  집합은 변하지 않는 것이 옳다.
- `rust/torch_c/pytests/verify_schemas.py`는 4203/4203로 그대로 — `overloads.json`을 건드리지
  않았다.
