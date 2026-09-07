# `_C.so` 실기기(에뮬레이터) 로드 검증

> **Correction (문서 감사, 2026-09):** 이 문서는 3-op 단계의 스냅샷입니다. `docs/devices/DEVICE.md`
> (이 감사가 이미 확인, round 2)가 같은 에뮬레이터에서 **54 케이스 배터리**(aten 96개 포함)로
> 후속 검증했습니다 — "다음에 필요한 것" 목록의 API 레벨 재현(항목 1)은 여전히 `pmp_api26`
> 한정으로 보입니다(`docs/devices/DEVICE.md` 도 같은 에뮬레이터). 항목 2·3(`libssl`류, APK 배포 경로)은
> 이 감사에서 추적하지 않았습니다.

**결론: 로드 성공.** `import _C` 및 구현된 op 3개(`aten.full.default`, `aten.add.Tensor`,
`aten.mm.default`) 모두 실제 Android 에뮬레이터의 임베디드 CPython 위에서 정상 동작했다.
크로스 컴파일·링크만 확인했던 이전 단계와 달리, 이번에는 기기 위에서 심볼 해결까지 실측했다.

## 사용한 환경

- 에뮬레이터: `pmp_api26` (API 26 / Android 8.0.0, `arm64-v8a`, `emulator-5554`)
  - `PythonMultiplatform` 저장소가 이미 이 에뮬레이터를 켜 둔 상태였다(`ps` 로 프로세스 확인,
    가동 시간 기준 다른 세션이 켜 놓은 것으로 추정). 그 세션과 충돌하지 않도록 앱 설치나
    `pm` 조작은 전혀 하지 않았고, `/data/local/tmp/torchc_check` 라는 전용 경로에만 파일을
    올렸다. 두 번째 에뮬레이터(`pmp_api36`)는 켜지 않았다.
- 타깃 CPython: `/Volumes/macMini/caches/target-python/aarch64-linux-android/prefix`
  (CPython 3.13.0, `Py_ENABLE_SHARED`, `libpython3.13.so` 가 실행 파일과 분리된 공유 라이브러리)
- 대상 아티팩트: `/Volumes/macMini/caches/cargo-target/aarch64-linux-android/release/lib_C.so`
  (2,280,968 바이트, ELF 64-bit aarch64, `not stripped`)

## 사전 점검 (기기에 올리기 전)

호스트에 `readelf`/`objdump` (GNU/aarch64용) 가 없어서, Xcode 의 크로스 타깃 지원 `llvm-objdump`
(`/Applications/Xcode.app/.../usr/bin/llvm-objdump`) 로 `lib_C.so` 를 정적으로 먼저 확인했다.

- `NEEDED`: `libpython3.13.so`, `libdl.so`, `libc.so` — **libpython 을 명시적으로 링크**하고 있다.
  즉 이 `.so` 는 "실행 파일의 동적 심볼 테이블에 얹혀 해결되는" 방식이 아니라, 로더가
  `libpython3.13.so` 를 별도 공유 객체로 찾아 링크하는 방식을 전제로 빌드되어 있다.
- 미해결(undefined) 동적 심볼 92개(전부 `Py*`/`_Py*` 접두) 를 타깃 `libpython3.13.so` 의
  export 테이블과 대조 — 확인한 것(`PyList_New`, `PyExc_TypeError`, `_Py_Dealloc`,
  `PyObject_VectorcallMethod` 등)은 전부 `libpython3.13.so` 가 전역(`g DF`/`g DO`) 로 내보내고
  있었고, `SONAME` 도 `libpython3.13.so` 로 `NEEDED` 항목과 정확히 일치했다.
- 이 시점에 "링크 대상과 실제 라이브러리의 심볼 집합이 맞는다" 는 것은 정적으로 확인됐지만,
  **동적 로더가 실제로 그것을 찾아 붙이는지는 기기에서 실행해봐야 안다** — 그래서 다음 단계로 넘어갔다.

## 기기에 올린 방법

`PythonMultiplatform` 저장소는 수정하지 않고, 그 저장소의 `python-multiplatform/build.gradle.kts`
에 있는 `androidNative*Test` 태스크(1584~1740줄 부근)가 쓰는 배선을 그대로 참고만 했다:
`PYTHONHOME` 을 스테이징 루트로, `LD_LIBRARY_PATH` 를 그 아래 `lib/` 로 잡고 `adb push --sync` 로
공유 라이브러리와 표준 라이브러리를 올린 뒤 `adb shell` 로 실행하는 패턴이다. 앱도, 인스트루먼테이션도,
JVM 도 필요 없었다 — 배선을 최소화하라는 지시에 맞춰 그 패턴만 재현했다.

전용 디렉터리 `/data/local/tmp/torchc_check` 아래에:

```
torchc_check/
├── bin/python3.13          # prefix/bin/python3.13 (7,176 바이트 — 대부분의 코드는 libpython3.13.so 쪽에 있다)
├── lib/libpython3.13.so    # 25.3 MB
├── lib/python3.13/         # stdlib, config-3.13-* 제외 (206 MB, 7,487개 파일)
└── _C.so                   # lib_C.so 를 이름만 바꿔 올림 (import 대상 이름과 파일명 일치 필요)
```

`libssl.so`/`libcrypto.so`/`libsqlite3.so` 등 심볼릭 링크로 된 부속 라이브러리는 **일부러 올리지
않았다** — 첫 시도에서 `adb push --sync` 가 그 심볼릭 링크들을 `remote symlink failed: Permission
denied` 로 거부했고(에뮬레이터 쉘이 `uid=2000(shell)`, root 아님), `lib_C.so` 의 `NEEDED` 목록에는
애초에 `libssl`/`libcrypto`/`libsqlite3` 가 없었으므로 이번 검증 목적(로드 여부)에는 불필요했다.
표준 라이브러리 트리 자체는 심볼릭 링크가 없어 `--sync` 가 그대로 통과했다(7,487개 파일, 198.5 MB,
1.77초).

실행 커맨드:

```
cd /data/local/tmp/torchc_check && \
  LD_LIBRARY_PATH=/data/local/tmp/torchc_check/lib \
  PYTHONHOME=/data/local/tmp/torchc_check \
  ./bin/python3.13 -c '...'
```

## 결과 (판정은 기기 내부 종료 코드로)

출력 grep 이 아니라 **파이썬 명령 뒤에 `; echo DEVICE_EXIT=$?` 를 붙여 기기 셸 내부에서 직접 받은
종료 코드**로 판정했다(이 프로젝트에서 트레이스백이 성공 마커 문자열을 출력에 남긴 전례가 있다는
경고에 따름).

1. `import sys; print(sys.version); print(sys.path)` — 정상 기동. `DEVICE_EXIT=0`.
   `sys.path` 는 `['', '.../lib/python313.zip', '.../lib/python3.13', '.../lib/python3.13/lib-dynload',
   '.../lib/python3.13/site-packages']` — cwd(`''`) 를 포함하므로 `_C.so` 를 루트에 둔 것만으로
   `import _C` 가 찾을 수 있었다(별도 `PYTHONPATH` 불필요).

2. `import _C; print(_C)` →
   ```
   <module '_C' from '/data/local/tmp/torchc_check/_C.so'>
   DEVICE_EXIT=0
   ```
   **92개 미해결 심볼이 기기 위에서 전부 `libpython3.13.so` 로 실제 해결됐다.**

3. `_C._aten_implemented()` →
   ```
   ['aten.add.Tensor', 'aten.full.default', 'aten.mm.default']
   ```

4. `_C._aten_dispatch("aten.full.default", [2, 3], 1.5)` →
   ```
   TensorBase(shape=[2, 3], dtype=float32, device=cpu)
   DEVICE_EXIT=0
   ```

5. 추가로 `add.Tensor`, `mm.default` 도 개별 실행해 확인:
   ```
   add: TensorBase(shape=[2, 2], dtype=float32, device=cpu)
   mm:  TensorBase(shape=[2, 2], dtype=float32, device=cpu)
   DEVICE_EXIT=0
   ```

세 op 모두 크래시나 트레이스백 없이 기대한 형태의 `TensorBase` 를 반환했고, 프로세스가 정상
종료(`DEVICE_EXIT=0`)했다.

## 이번 검증이 답하지 않은 것 (범위 밖)

- **API 26 한정 검증이다.** 다른 에뮬레이터(`pmp_api36`, API 36)에서는 시도하지 않았다 — 두 대뿐인
  에뮬레이터 중 다른 세션이 이미 켜 둔 `pmp_api26` 을 건드리지 않는 선에서 검증을 끝낼 수 있었기
  때문이다. `manylinux`/NDK ABI 가 API 레벨에 따라 달라지는 지점(예: 16 KB 페이지, execstack 등)은
  이 검증 범위 밖이다.
- **실제 앱(APK) 안에서의 로드는 확인하지 않았다.** `adb shell` 에서 `/data/local/tmp` 실행 파일로
  띄운 프로세스와, APK 로 설치되어 앱 전용 저장소(`/data/data/<pkg>`)에서 `PYTHONHOME` 을 잡는
  실제 배포 경로는 권한 모델과 SELinux 컨텍스트가 다르다. 다만 `PythonMultiplatform` 의 기존
  `androidNative*Test` 태스크가 같은 `/data/local/tmp` 패턴으로 실제 임베딩 테스트를 통과시키고
  있으므로, 이 경로가 그 저장소의 검증 기준과 같은 종류의 신뢰도를 갖는다고 볼 수 있다.
- **`libssl`/`libcrypto`/`libsqlite3` 는 기기에 올리지 않았다.** `_C.so` 가 지금은 필요로 하지
  않지만, op 구현이 늘어 이들 중 하나에 의존하는 순간 이번 방식(심볼릭 링크 제외)은 그대로
  재사용할 수 없다 — `adb push` 로 심볼릭 링크를 올리려면 `adb root` 가 필요하거나, 심볼릭 링크를
  실제 파일로 풀어서 올려야 한다.
- **메모리/성능 측정은 하지 않았다.** 로드와 3개 op 의 정확성만 확인했고, 디스패치 오버헤드나
  텐서 크기 확장에 따른 동작은 보지 않았다.

## 다음에 필요한 것

1. `pmp_api36` (또는 다른 API 레벨) 에서도 같은 절차로 재현해 API 26 이 특이 케이스가 아님을 확인.
2. `libssl_python.so`/`libcrypto_python.so`/`libsqlite3_python.so` 도 필요해지는 시점에는
   심볼릭 링크를 풀어(`cp -L` 등) 올리거나 `adb root` 가 되는 에뮬레이터로 전환.
3. 실제 APK 배포 경로(앱 전용 저장소 + `PYTHONHOME`)에서의 로드는 `PythonMultiplatform` 의
   샘플 앱 배선을 참고해 별도로 검증 필요 — 이번 검증은 `adb shell` 직접 실행 경로만 다룬다.
