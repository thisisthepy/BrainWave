# CANDLE_DEPS — `candle-core` 의 `tokenizers` 비선택 의존 조사

`docs/design/TORCH_C.md` §5-1 이 최상위 미해결 항목으로 올린 것의 후속 조사입니다. **이 문서는 조사
결과만 담습니다 — 코드는 고치지 않았습니다.** `rust/torch_c/Cargo.toml` 은 읽기만 했고, 실험은
전부 `/Volumes/macMini/caches/candle-probe` 에서 했습니다.

## 0. 한눈에

| 질문 | 답 |
|---|---|
| 왜 필수인가 | GGUF 안의 토크나이저를 읽는 편의 기능(`quantized::tokenizer`) 하나 때문. **`torch_c` 는 이 기능을 쓰지 않는다** |
| 뗄 수 있는가 | 있다 — 상류에 이미 정확히 이 문제를 고치는 PR 이 열려 있다(#3490). **4 개월째 머지 안 됨** |
| 지금 뗄 수단이 있는가 | 있다 — 같은 3 줄짜리 패치를 로컬 포크/vendor 에 적용하면 됨. **직접 빌드해서 검증함** |
| 실제 비용 | 크레이트 **150 → 106 (−44, −29%)**, 릴리스 빌드 CPU 시간 **약 36% 감소** (측정치, 아래 §3) |
| candle 을 바꿔야 하는가 | 아니오 — DESIGN.md §4 의 근거(동적 랭크, GGUF 양자화, 저자 계보)는 이 문제와 무관하고 grep 결과 대안도 없음 |
| 지금 해야 하는가 | **급하지 않다.** 아래 §5 판단 참조 |

---

## 1. 왜 필수인가 — 확인함

### 1a. `Cargo.toml` 선언

`candle-core-0.11.0/Cargo.toml.orig`:

```
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
tokenizers = { workspace = true, features = ["onig"] }
```

`accelerate-src`, `candle-kernels`, `candle-metal-kernels`, `cudarc`, `intel-mkl-src`,
`objc2-metal`, `objc2-foundation` 등 다른 선택적 백엔드 의존은 전부 `optional = true` 가 붙어
있는데, **`tokenizers` 에만 없습니다.** `[features]` 블록의 `default = []` 도 이것과 무관합니다 —
optional 이 아니므로 애초에 feature 로 끌 수 있는 항목이 아닙니다. `default-features = false` 를
줘도 안 떨어진다는 `docs/design/TORCH_C.md` 의 기술이 소스 수준에서 확인됩니다.

### 1b. 쓰는 곳 — `quantized::tokenizer` 하나

`candle-core-0.11.0/src/quantized/mod.rs:18`:

```rust
#[cfg(not(target_arch = "wasm32"))]
pub mod tokenizer;
```

feature 게이트가 전혀 없고 `wasm32` 만 걸러냅니다. 이 모듈(`src/quantized/tokenizer.rs`, 약 250
줄)이 하는 일은 단 하나 — GGUF 메타데이터(`tokenizer.ggml.*` 키)에서 BPE vocab·merges·프리토크나
이저 파이프라인을 읽어 `tokenizers::Tokenizer` 를 조립하는 `TokenizerFromGguf` trait 구현입니다.
candle-core 자신의 소스 안에서 이 trait 을 호출하는 곳은 **없습니다**(`grep -rn
"TokenizerFromGguf" src/` 결과 정의부 한 곳뿐) — 순수하게 다운스트림(예: 사용자 코드나
`candle-transformers`)이 GGUF 로더에서 토크나이저까지 한 번에 얻고 싶을 때 쓰라고 얹어둔 공개
API 표면입니다.

**`torch_c` 는 이걸 쓰지 않습니다.** `rust/torch_c/src/` 전체(881 줄)를 `tokenizer|Tokenizer|
quantized` 로 grep 해도 일치가 없습니다 — `torch_c` 는 텐서 연산(`aten.*`)만 구현하고 있고
GGUF 로딩 경로 자체를 아직 붙이지 않았습니다.

### 1c. 양자화(quantized) 지원 자체와는 무관

DESIGN.md §4 가 candle 을 고른 두 번째 근거("양자화가 이미 있음 — GGML/GGUF k-quant 를
읽는다")는 `k_quants` · `ggml_file` · `gguf_file` 모듈이 담당하며, 이들은 `tokenizer.rs` 와
독립입니다(`quantized/mod.rs` 에서 별도 `pub mod` 로 선언되고 서로 참조하지 않음). **`tokenizers`
를 떼도 GGUF/양자화 읽기 자체는 그대로 남습니다** — 잃는 것은 "GGUF 안에 내장된 토크나이저를
바로 `tokenizers::Tokenizer` 로 바꿔주는 편의 함수" 뿐이고, 그건 파이썬 계층(`transformers`)이
이미 하는 일과 중복입니다(DESIGN.md §2).

---

## 2. 뗄 방법 — 셋을 조사함

### 2a. 상류 feature gate 제안 — **이미 있음, 안 됨**

GitHub 검색(`gh api graphql` search, 2026-08-24 기준)으로 정확히 이 문제를 고치는 PR 을 찾았습니다.

**[huggingface/candle#3490](https://github.com/huggingface/candle/pull/3490)** — "fix: make
tokenizers dependency optional (left on by default though) in candle-core"

- 작성자 `butterflysky`, **2026-04-23 오픈, 아직 OPEN**(2026-08-24 기준 약 4 개월).
- 패치 내용은 이 문서 §2c 에서 로컬로 검증한 것과 동일한 모양: `tokenizers` 에 `optional = true`
  추가 + `tokenizers` feature 신설(`default` 에 포함해 하위 호환 유지) + `quantized/mod.rs` 의
  `pub mod tokenizer;` 를 `#[cfg(feature = "tokenizers")]` 로 게이트.
- candle-nn / candle-transformers 도 같은 패턴으로 feature 를 전파하도록 고쳐, 워크스페이스
  전체가 opt-out 가능해집니다.
- 작성자 본인의 다운스트림 프로젝트(`memory-mcp`)에서 109 개 테스트 통과 + `onig` 완전 제거를
  확인했다고 PR 본문에 적혀 있습니다(**본인 보고, 제3자 검증 아님 — 미확인으로 취급**).
- 리뷰: 멤버 `ivarflakstad` 가 "Looks sensible to me" 라고 초기 코멘트(2026-04-24)했으나 **정식
  리뷰나 머지 액션은 없음.**
- 이후 독립적으로 같은 문제를 겪은 사용자 둘이 코멘트로 합류:
  - `Jerboas86`(2026-06-10) — Windows 지원에도 `onig` 가 문제라고 지적, 관련
    [tokenizers#1581](https://github.com/huggingface/tokenizers/issues/1581) 링크.
  - `fbbbkaya`(2026-07-30) — **PR 의 존재를 모른 채** `x86_64-pc-windows-gnu`(MinGW) 에서
    `onig_sys` 가 `uid_t`/`gid_t` 재정의 충돌로 빌드가 깨진다고 보고하며, **거의 동일한 패치를
    스스로 다시 유도**했습니다. candle 의 타깃 매트릭스(DESIGN.md "타깃 매트릭스")에 Windows
    x86_64 가 들어 있으므로 torchnative 에도 잠재적으로 관련됩니다(단, 기본 툴체인이 MSVC 인지
    GNU 인지는 **미확인**).
- **main 브랜치 재확인**: `git show origin/main:candle-core/Cargo.toml`(2026-08-24, 최신 커밋
  `81f247a8` / 2026-08-23) 에서 `tokenizers` 는 여전히 `optional` 없이 필수입니다. **PR 은
  머지되지도, main 에 리베이스되지도 않은 상태**입니다.

**결론:** 제안할 필요가 없습니다 — 이미 올바른 제안이 올라가 있고, 정체된 상태입니다. 우리가
할 수 있는 것은 (i) 코멘트로 신호를 더 보태거나(세 번째 독립 사용자로서 근거를 하나 더 얹음),
(ii) 그냥 이 PR 이 머지될 때까지 기다리는 것입니다.

### 2b. 포크해서 파일 제거 — **권장하지 않음**

`quantized/tokenizer.rs` 자체를 삭제하는 방식은 (a) 공개 API(`TokenizerFromGguf`)를 제거하는
호환성 깨는 변경이라 향후 candle 업데이트를 받을 때마다 충돌 지점이 되고, (b) 이미 상류에 더
안전한 대안(feature 게이트, §2a)이 나와 있는데 그것과 다른 모양의 패치를 만드는 것이라 나중에
상류 PR 이 머지됐을 때 우리 포크와 합류하기가 더 어렵습니다. **아래 §2c 로 대체합니다.**

### 2c. `[patch]` / vendoring — **검증함, 즉시 적용 가능**

PR #3490 의 패치를 그대로 `candle-core 0.11.0`(현재 `torch_c` 가 고정한 버전)에 옮겨 로컬에서
빌드까지 확인했습니다. 위치: `/Volumes/macMini/caches/candle-probe/candle-src`(candle 저장소
태그 `0.11.0`, 커밋 `31f35b1`, 얕은 클론). 적용한 diff:

```diff
--- a/candle-core/Cargo.toml
+++ b/candle-core/Cargo.toml
@@
 [target.'cfg(not(target_arch = "wasm32"))'.dependencies]
-tokenizers = { workspace = true, features = ["onig"] }
+tokenizers = { workspace = true, features = ["onig"], optional = true }
@@
 [features]
 default = []
+tokenizers = ["dep:tokenizers"]

--- a/candle-core/src/quantized/mod.rs
+++ b/candle-core/src/quantized/mod.rs
@@
-#[cfg(not(target_arch = "wasm32"))]
+#[cfg(all(not(target_arch = "wasm32"), feature = "tokenizers"))]
 pub mod tokenizer;
```

`cargo build --release` 로 정상 빌드됨을 확인(§3 에 시간 포함). **중요한 성질 하나**:
`rust/torch_c/Cargo.toml:30` 이 이미 `default-features = false` 로 `candle-core` 를 선언하고
있으므로(`# 이미 있음 절 참조`), 이 패치가 적용된 `candle-core` 로 바꿔 끼우기만 하면
**`torch_c/Cargo.toml` 을 추가로 고칠 필요가 없습니다** — `tokenizers` feature 를 명시적으로
요청하지 않는 한 자동으로 빠집니다.

적용 수단은 워크스페이스 루트의 `[patch.crates-io]` 로 거는 것이 표준이지만, **`rust/torch_c` 는
`[workspace]` 를 선언하지 않은 단독 크레이트이고 `rust/Cargo.toml` 워크스페이스 루트도 없습니다**
(`find . -iname Cargo.toml` 결과 `rust/torch_c/Cargo.toml` 하나뿐). Cargo 규약상 `[patch]` 는
그 크레이트 자신의 루트 매니페스트에 적어야 하므로, **적용하려면 `rust/torch_c/Cargo.toml` 을
고쳐야 합니다** — 이번 조사에서는 금지된 파일이라 실제로 걸지 않았습니다. 적용할 때 필요한
모양(다른 워크스트림이 가져다 쓸 수 있도록 기록):

```toml
[patch.crates-io]
candle-core = { git = "<우리 포크 URL>", rev = "<커밋>" }
# 또는
candle-core = { path = "<vendor 경로>" }
```

(주의: 테이블 이름은 `crates-io` — 대시입니다. `[patch.crates.io]` 로 쓰면 Cargo 가 `crates`
를 레지스트리 이름, `io` 를 그 아래 키로 파싱하려다 "should be a URL or registry name" 에러를
냅니다. §8 에서 실제로 이 오타를 밟고 고쳤습니다.)

포크/vendor 소스는 위 3 줄 diff 를 candle 0.11.0 태그에 적용한 것이면 충분합니다. 업스트림
PR #3490 이 머지되고 새 버전이 나오면 이 `[patch]` 를 지우고 버전만 올리면 되므로, **되돌리기
비용이 낮습니다.**

---

## 3. 실제 비용 — 측정함

### 3a. 크레이트 수

`rust/torch_c` 실제 그래프(수정 없이 `cargo metadata` 로 읽음, 2026-08-24):

- 전체 해석된 패키지: **150 개** (`docs/design/TORCH_C.md` 가 적은 "129 크레이트 / 락파일 150 패키지"의
  150 과 일치).
- `tokenizers` 로만 들어오는 서브그래프를 그래프 도달성 분석으로 계산(루트에서 `tokenizers`
  엣지를 끊고 재방문 가능한 노드 수 비교): **106 개**만 남음 → **−44 개 (−29.3%)**.
- 빠지는 44 개에는 `onig`/`onig_sys`(oniguruma C 라이브러리 본체) 뿐 아니라 그것을 빌드하는
  **C 툴체인 연결 크레이트**(`cc`, `pkg-config`, `find-msvc-tools`)와 `tokenizers` 자체가 쓰는
  `regex`/`regex-automata`/`regex-syntax`, `compact_str`, `derive_builder`(+`_core`/`_macro`),
  `esaxx-rs`, `spm_precompiled`, `unicode-normalization-alignments` 등 20 여 개 크레이트가
  포함됩니다. 역방향 확인 결과 이 그래프에서 `tokenizers` 를 요청하는 것은 `candle-core` 하나,
  `onig` 를 요청하는 것은 `tokenizers` 하나뿐이라 이 44 개는 정확히 이 의존 하나에 귀속됩니다.

`/Volumes/macMini/caches/candle-probe` 의 독립 실험 크레이트(patched vs baseline, 최소
`candle-core` 만 의존)로 교차 검증:

| | 잠긴 패키지 수 (`cargo generate-lockfile`) | `cargo metadata` 노드 수 |
|---|---|---|
| baseline (레지스트리 `candle-core 0.11.0`) | 142 | 143 |
| patched (§2c 의 3 줄 diff 적용) | 98 | 99 |
| 차이 | **−44** | **−44** |

`torch_c` 그래프 도달성 분석(−44)과 완전히 같은 수가 나와, 이 숫자가 `torch_c` 특유의 노이즈가
아니라 `tokenizers` 의존 그 자체의 크기임을 확인했습니다.

### 3b. 빌드 시간

같은 두 실험 크레이트를 `cargo clean && cargo build --release` 로 클린 빌드해 시간을 쟀습니다
(2026-08-24, 단독 실행, `/usr/bin/time -p`). 두 번씩 반복:

| | wall (run 1) | wall (run 2) | user CPU (run 1) | user CPU (run 2) |
|---|---|---|---|---|
| baseline | 51.50s | 45.85s | 221.31s | 221.60s |
| patched | 35.45s | 35.66s | 141.15s | 142.04s |
| 감소 | −31%(run1) | −22%(run2, load 상승 중 측정) | **−36%** | **−36%** |

wall 시간은 run 2 때 시스템 load 가 5→10 으로 오르는 도중이라 흔들렸지만(run 1 은 load 5.22 에서
단독 측정), **user CPU 시간은 두 번 다 거의 완벽히 재현**(221.3s/221.6s, 141.2s/142.0s)되어
스케줄링 노이즈에 덜 민감한 지표로 신뢰할 수 있습니다. **결론: 릴리스 빌드 CPU 시간 기준 약
36% 감소, 크레이트 수 기준 29% 감소.** 둘 다 추정이 아니라 같은 크레이트를 두 버전으로 실제로
빌드해서 잰 값입니다.

**미측정으로 남긴 것**: 스트립 후 바이너리 크기 차이(§4 는 TORCH_C.md 몫), Android/iOS
크로스컴파일 타깃에서의 시간 차이(호스트만 쟀음), incremental/디버그 빌드에서의 차이.

---

## 4. candle 을 바꿔야 하는가 — 아니오

DESIGN.md §4("텐서 엔진은 candle")가 candle 을 고른 근거 셋:

1. 동적 랭크(torch API 와 맞음, burn 은 정적 랭크라 탈락)
2. GGUF/GGML k-quant 양자화 지원
3. 저자(Laurent Mazare)의 libtorch 바인딩 경험 계보

이 셋 중 `tokenizers` 문제와 관련 있는 것은 없습니다. §1c 에서 확인했듯 양자화 지원(근거 2)은
`k_quants`/`gguf_file`/`ggml_file` 모듈이 담당하고 `tokenizer.rs` 와 독립이라, `tokenizers` 를
떼도 근거 2 는 그대로 유지됩니다. burn 은 DESIGN.md 가 이미 정적 랭크 이유로 기각했고 이번
조사에서 이를 뒤집을 근거를 찾지 못했습니다(burn 쪽 의존 트리 자체는 조사하지 않았습니다 —
**미확인**, 다만 애초에 정적 랭크가 확정적 결격 사유라 조사할 필요가 없다고 판단했습니다).
**candle 을 바꾸는 옵션은 기각합니다.**

---

## 5. 권고와 우선순위 판단

**권고:** §2c 의 3 줄 패치를 `[patch.crates-io]` 로 걸어 지금 뗄 수 있고, 위험은 낮습니다
(diff 가 상류 PR #3490 과 동일한 모양이라 즉흥 패치가 아니라 커뮤니티가 이미 검토한 형태이고,
`torch_c` 는 해당 기능을 아예 쓰지 않아 회귀 위험이 없습니다). 다만 이 패치는 `rust/torch_c/
Cargo.toml` 을 건드려야 적용되므로 **이 조사 세션의 범위 밖**입니다 — 해당 파일을 담당하는
워크스트림에 §2c 의 diff 와 `[patch]` 블록을 그대로 넘기면 됩니다.

**지금 당장 vs 나중:**

- **급하지 않습니다.** `docs/design/TORCH_C.md` §6 미확인 항목과 §5 의 다른 미해결 항목(dtype 승격표,
  `torch.bool`, `aten` 디스패치 진입로, `view`/별칭 의미론)은 **기능이 막히는** 항목인 반면, 이
  의존 문제는 **크기·빌드 표면**의 문제입니다 — 지금 당장 아무것도 깨뜨리고 있지 않고,
  Android·iOS 빌드도 이미 통과하고 있습니다(TORCH_C.md §5-1).
- 그러나 **방치할수록 비용이 오릅니다.** GGUF 양자화 로딩 경로를 실제로 연결하는 시점(§7 의
  "다음 단계"들)이 오면 `quantized` 모듈을 더 깊이 건드리게 되고, 그때 `tokenizers` 를 떼는 게
  지금보다 얽히게 됩니다. 또한 candle 버전을 올릴 때마다(§2a 의 PR 이 그 사이 머지되면 더더욱)
  이 패치를 다시 맞춰야 하므로, **패치를 만들 거면 지금처럼 진행이 얕을 때가 가장 쌉니다.**
  Windows 타깃(DESIGN.md 타깃 매트릭스에 포함)에서 `onig_sys` 가 MinGW 에서 깨진다는 상류
  보고(§2a)도 있어, Windows 빌드를 실제로 태우기 **전에** 처리해두는 편이 안전합니다.
- **권고하는 순서:** 지금 이 시점에 §2c 패치를 걸어 44 크레이트·36% 빌드시간을 회수하되,
  기능 블로커(dtype 승격표 등)를 막지는 않는 낮은 우선순위 작업으로 큐에 넣습니다. 상류
  PR #3490 에는 코멘트로 근거(§3 의 측정치)를 보태 머지를 독려하고, 머지되면 `[patch]` 를
  지우고 버전만 올립니다.

---

## 6. 미확인으로 남긴 것

| 항목 | 상태 |
|---|---|
| PR #3490 작성자의 "109 개 테스트 통과" 자체 보고 | 본인 보고, 제3자 검증 안 됨 |
| torchnative 의 Windows 빌드가 MSVC/GNU 중 무엇을 쓰는지 | 미확인 — `onig_sys` MinGW 버그(§2a) 관련 여부 불명 |
| Android/iOS 크로스컴파일에서의 크레이트 수·빌드시간 감소폭 | 미측정 — 호스트(macOS arm64)에서만 잼 |
| 스트립 후 바이너리 크기에 `tokenizers` 제거가 주는 영향 | 미측정 |
| burn 의 실제 의존 트리 크기 | 미조사 — 정적 랭크로 이미 기각되어 조사 불필요 판단 |
| WASM 타깃에서 이 이슈의 관련성 | candle 자체가 이미 `cfg(not(target_arch = "wasm32"))` 로 걸러둠 — torchnative 타깃 매트릭스에도 WASM 없음, 무관 |

---

## 7. 재현

```bash
export PATH="$HOME/.cargo/bin:$PATH"

# 크레이트 수 비교
cd /Volumes/macMini/caches/candle-probe/baseline && cargo clean && cargo generate-lockfile
cd /Volumes/macMini/caches/candle-probe/patched  && cargo clean && cargo generate-lockfile

# 빌드 시간 비교 (단독 실행 — 다른 에이전트/빌드와 동시에 돌리지 말 것)
cd /Volumes/macMini/caches/candle-probe/baseline && cargo clean && /usr/bin/time -p cargo build --release
cd /Volumes/macMini/caches/candle-probe/patched  && cargo clean && /usr/bin/time -p cargo build --release
```

`patched/Cargo.toml` 은 `/Volumes/macMini/caches/candle-probe/candle-src`(candle 0.11.0 태그에
§2c 의 3 줄 diff 적용)를 `path` 의존으로 가리킵니다. `candle-src/` 는 조사용 클론이며 torchnative
저장소의 일부가 아닙니다.

---

## 8. 적용함 (2026-08-24)

§5 의 권고를 실행했습니다. `rust/torch_c/Cargo.toml` 에 `[patch.crates-io]` 블록을 추가해
`candle-core` 를 §2c 의 패치가 적용된 로컬 vendor 사본으로 바꿔 끼웠습니다.

### 8a. 적용 방법

```toml
[patch.crates-io]
candle-core = { path = "/Volumes/macMini/caches/candle-vendor/candle-0.11.0-patched/candle-core" }
```

vendor 사본은 §2c 조사 때 만든 `/Volumes/macMini/caches/candle-probe/candle-src`(candle 0.11.0
태그, 커밋 `31f35b1`)를 `/Volumes/macMini/caches/candle-vendor/candle-0.11.0-patched` 로 복사하고,
§2c 의 3 줄 diff 를 로컬 커밋 하나(`e3b39f9`)로 고정한 것입니다. `torch_c/Cargo.toml` 자체는 3 줄
patch 블록 추가 외에 바꾸지 않았습니다 — `default-features = false` 가 이미 있어서 새로 생긴
`tokenizers` feature 를 아무도 요청하지 않으므로 그대로 빠집니다.

**표 이름 오타 주의:** 표준 이름은 `crates-io`(대시)입니다. §2·§5 초안에 `[patch.crates.io]`(점)로
적어둔 곳이 있었는데, 그대로 걸었더니 Cargo 가 `crates.io` 를 "레지스트리 `crates` 아래의 키
`io`"로 파싱하려다 `should be a URL or registry name` 에러를 냈습니다. 문서의 두 곳(§2c, §5)을
고쳤습니다.

### 8b. 재현성 — 그대로 로컬 경로 한계입니다

이 `[patch]` 는 `/Volumes/macMini/caches/candle-vendor/candle-0.11.0-patched` 라는 이 기계의
절대 경로를 가리킵니다. **다른 기계에서는 그 경로가 없으므로 그대로 깨집니다.** 재현 수단은
`rust/torch_c/Cargo.toml` 의 patch 블록 옆 주석에 그대로 적어 두었습니다 — candle 0.11.0 태그를
그 경로에 얕은 클론하고, §2c 의 diff 를 `candle-core/Cargo.toml` 과
`candle-core/src/quantized/mod.rs` 에 적용해 로컬 커밋 하나로 고정하면 됩니다. §2c 에서 이미
검토했듯 상류 포크(`git` 소스)로 옮기는 편이 더 재현 가능하지만, 그러려면 GitHub 에 새 포크를
만들어 커밋을 push 해야 하는데 이번 세션은 그 권한 범위 밖이라(자동 승인 분류기가 거부) 시도하지
않았습니다. `[workspace]` 를 가진 전체 candle 저장소(24MB, `candle-core` 가 `version.workspace =
true` 등으로 워크스페이스 상속을 쓰기 때문에 `candle-core` 디렉터리만 떼어 vendor 할 수 없습니다)를
이 저장소 git 에 커밋하는 방법도 검토했으나, 이번 작업 범위(`Cargo.toml` + 문서)를 벗어나는 대량
추가라 하지 않았습니다. 상류 PR #3490 이 머지되면 이 `[patch]` 블록을 지우고 `candle-core` 버전만
올리면 되므로, 어느 경로든 되돌리기 비용은 낮습니다.

### 8c. 다시 잰 숫자

**크레이트 수** — 이번엔 probe 가 아니라 `rust/torch_c` 실물 `Cargo.lock` 으로 직접 셌습니다
(`grep -c '^name = ' rust/torch_c/Cargo.lock`):

| | 패키지 수 |
|---|---|
| 패치 전 (`git show HEAD:rust/torch_c/Cargo.lock`) | **150** |
| 패치 후 | **106** |
| 차이 | **−44** (§3a 의 probe 측정과 정확히 일치) |

패치 후 lockfile 에 `tokenizers`/`onig`/`onig_sys` 가 전혀 없음을 확인했습니다.

**빌드 시간** — §3b 는 최소 probe 크레이트(candle-core 만 의존)를 단독 load 5.22 에서 쟀습니다.
이번엔 **`rust/torch_c` 실물 크레이트**를 `cargo clean && /usr/bin/time -p cargo build --release`
로 쟀습니다(`CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-deps`, 다른 에이전트 전용
디렉터리와 분리). **주의: 이 기계는 이 세션 내내 다른 에이전트 둘이 Android/iOS 빌드를 돌리고
있어 load 가 요동쳤습니다(측정 시작 시 8.7, 베이스라인 완료 시점엔 30 까지 치솟음, 패치 빌드
동안엔 18→14) — §3b 처럼 깨끗한 단독 측정이 아닙니다.** 그래도 두 빌드가 수 분 간격으로 붙어 있어
외부 load 가 상대적으로는 비슷하게 걸렸을 가능성이 높고, 방향은 §3b 와 일치합니다:

| | wall (`real`) | user CPU | 측정 시 load(1분 평균) |
|---|---|---|---|
| 패치 전 | 85.64s | 250.97s | 시작 8.7, 종료 시점 30 (급등) |
| 패치 후 | 50.38s | 169.11s | 시작 18.5, 종료 시점 14 |
| 감소 | **−41%** | **−33%** | — |

이 숫자는 §3b 의 −36%(user CPU, load 5.22 단독)와 같은 방향·비슷한 크기입니다. 다만 load 가
오염되어 있었다는 것을 그대로 적습니다 — **정밀한 재현치가 필요하면 load 가 낮을 때
`rust/torch_c` 에서 §7 재현 절차를 다시 돌리는 편이 낫습니다.** 크레이트 수(−44, §8c 위)는 load 와
무관한 지표라 그대로 신뢰할 수 있습니다.

### 8d. 판정 기준 확인

```
$PY tools/golden/compare.py                 # SUMMARY: 1212/1212 cases passed, 0 failed, ops covered=70   EXIT=0
$PY rust/torch_c/pytests/verify_schemas.py  # SUMMARY: 127/127 table entries matched upstream, 0 failed    EXIT=0
sh rust/torch_c/pytests/run.sh              # 62/62 스모크 테스트 전부 ok                                   EXIT=0
```

3 타깃:

| 타깃 | 명령 | 결과 |
|---|---|---|
| 호스트 (`aarch64-apple-darwin`) | `pytests/run.sh` (위) | **exit 0** |
| Android (`aarch64-linux-android`, `arm64-v8a`) | `cargo ndk -t arm64-v8a --platform 21 build --release` | **exit 0**, `Finished release ... in 47.37s` |
| iOS (`aarch64-apple-ios`) | `cargo build --release --target aarch64-apple-ios` (`PYO3_CONFIG_FILE=/Volumes/macMini/caches/bw-pyo3-ios.config`) | **exit 0**, `Finished release ... in 38.52s` |

전부 통과했습니다. 기능 회귀는 없습니다 — `torch_c` 는 애초에 `tokenizers`/`quantized::tokenizer`
를 쓰지 않으므로 이 패치로 잃는 표면이 없고, 골든·스키마·스모크 숫자는 패치 전(§0 이 인용하는
`docs/kernels/OPS8.md` 의 1212/1212, covered=70, 127/127)과 그대로 같습니다.

### 8e. 남은 일

- **재현성 한계(§8b)를 되도록 빨리 좁힐 것.** 지금은 이 기계 한 대에서만 재현됩니다. 후속 세션이
  GitHub 포크 생성 권한을 받으면, §2a 의 상류 PR #3490 브랜치를 `0.11.0` 에 리베이스해 우리
  포크에 올리고 `[patch.crates-io]` 를 `git` 소스로 바꾸는 편이 vendor 경로보다 낫습니다.
- **load 가 낮을 때 §8c 의 빌드 시간을 다시 잴 것.** 지금 수치는 방향은 맞지만 오염되어 있습니다.
- 상류 PR #3490 에 코멘트로 §3 의 측정치를 보태는 것(§5 가 이미 제안)은 이번에도 하지
  않았습니다 — 이 세션 권한 범위 밖입니다.

---

## 9. 조율 세션의 판정 — 패치는 착지시키지 않습니다

§8 의 `[patch.crates-io]` 블록은 **검증됐지만 커밋하지 않았습니다.** 문서만 착지시킵니다.

이유는 하나입니다. 그 블록이 `/Volumes/macMini/caches/candle-vendor/...` 라는 **이 기계의
절대 경로**를 가리키고, `[patch]` 의 경로가 없으면 cargo 는 경고가 아니라 **에러로 죽습니다.**
즉 커밋하는 순간 이 기계 밖의 모든 체크아웃에서 빌드가 불가능해집니다. 라이브러리가 될
저장소에서 로컬 빌드 40% 단축과 바꿀 만한 것이 아닙니다.

**측정과 진단은 유효하므로 남깁니다** — 크레이트 −44 개는 실제 `Cargo.lock` 으로 재확인됐고,
`[patch.crates.io]` (점)가 유효한 TOML 이 아니라는 §2c 의 오타 수정은 이 문서를 따라 하는
사람이 반드시 만나는 함정입니다.

착지 가능한 형태는 셋입니다:

1. 상류 PR #3490 이 머지된다 — 그러면 `[patch]` 자체가 필요 없습니다
2. 공개 포크에 패치를 올리고 `[patch]` 를 `git` 소스로 건다 — 이번 회차에 시도했으나
   포크 생성이 권한에서 막혔습니다
3. candle 을 이 저장소에 벤더링한다 — 24MB 가 git 히스토리에 들어갑니다

**셋 중 하나가 되기 전에는 이 최적화를 켜지 않습니다.** 각자 로컬에서 §8 대로 적용하는 것은
자유이고, 그것이 지금의 권장 형태입니다.
