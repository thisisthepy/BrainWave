# QNNCI — closing docs/QNN.md's host gap with a hosted Linux x86-64 runner

<!-- DOCWATCH: symbol-in-file tools/ci/qnn_lower.py verify_facts present -->
<!-- DOCWATCH: symbol-in-file tools/ci/qnn_lower.py collect_facts present -->
<!-- DOCWATCH: symbol-in-file tools/ci/qnn_lower.py render_summary present -->
<!-- DOCWATCH: symbol-in-file tools/ci/qnn_lower.py MIN_DELEGATED_FRACTION present -->
<!-- DOCWATCH: symbol-in-file tools/ci/qnn_lower.py DEFAULT_SOC_MODEL present -->
<!-- DOCWATCH: symbol-in-file tools/ci/qnn_lower.py integrity_ok present -->
<!-- DOCWATCH: symbol-in-file .github/workflows/qnn-lower.yml EXPECTED_QNN_SDK_VERSION present -->
<!-- DOCWATCH: symbol-in-file .github/workflows/qnn-lower.yml "ubuntu-24.04" present -->
<!-- DOCWATCH: symbol-in-file .github/workflows/qnn-lower.yml "steps.lower.outputs.integrity_ok" present -->
<!-- DOCWATCH: symbol-in-file .github/workflows/qnn-lower.yml "upload-artifact@v4" present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_qnnci.py test_a_program_with_no_qnn_delegate_is_never_uploaded present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_qnnci.py test_poor_coverage_fails_the_job_and_keeps_the_artefact present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_qnnci.py test_the_verification_runs_before_every_upload present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_qnnci.py test_the_threshold_cannot_be_lowered_from_the_command_line present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_qnnci.py test_nothing_in_the_lowering_job_disables_its_own_assertions present -->

## 0. At a glance

| | |
|---|---|
| What gap is this closing? | `docs/QNN.md` §11: "**`lower()` has never executed**" — it refuses on every host reachable from this project, so its body is *wired, not run* |
| Why can CI close it? | `docs/QNN.md` §1.3 measured that the wall is **the host**, not the code: the QNN SDK's host libraries exist only under `lib/x86_64-linux-clang`. A hosted `ubuntu-*` runner is that host |
| Has this job ever run? | **No. It has never run.** Every sentence below is about what it is written to do, not about what it did |
| Does it prove anything about the HTP? | **No.** §1 below, and `docs/QNN.md` §6.4 stays true |
| Runner | `ubuntu-24.04`, pinned. §3 is the decision and its reason |
| Where the logic lives | `tools/ci/qnn_lower.py`, not inline YAML. §5 |
| Tests | **30** in `rust/torch_c/pytests/test_qnnci.py`, none of which need executorch |
| Nullifications attempted / uncaught | **14 / 0**, and one of them found a real hole in this document's own test file first (§5.2) |

---

## 1. What the job proves, and what it does not

This is the first thing in the workflow file itself, and it is repeated here
because `docs/QNN.md` §6.4 is a paragraph that has to stay true and a job that
emits a file called `layer0_mlp.pte` built "for the HTP" is the most tempting
moment in the whole round to let *lowered for* slide into *ran on*.

**It proves:**

* that `to_edge_transform_and_lower_to_qnn` runs to completion on a real
  `transformers` submodule with the real `QnnPartitioner` and the real SDK;
* that the resulting `.pte` is a QNN program built for the named silicon —
  `docs/QNN.md` §5 step 7's three checks;
* what `delegated_fraction` actually is, which `docs/QNN.md` §11 records as
  **unknown**.

**It proves nothing about:**

* **HTP execution.** `docs/QNN.md` §6.1 lists four things that would prove it.
  This job supplies exactly one — (b), the decoded artefact, which §6.1 itself
  calls "still a claim about a file". (c), the optrace/QHAS hardware counters,
  and (d), the skeleton-removal negative control, both need the physical
  device, and §6.1 says (d) is "the one that makes it evidence rather than a
  plan." A hosted runner has no Snapdragon in it.
* **numbers.** No element-wise comparison against upstream happens here
  (`docs/QNN.md` §7.2), and no timing (§11).

In `CLAUDE.md` §4's three grades, a green run moves the QNN lowering from
**built** to **reached**. It does not touch **agreed**, and it does not touch
the separate question of *which unit* executed anything.

---

## 2. Every version the job pins

All four of `docs/QNN.md` §1.1's numbers, plus two more. Each is declared once,
in the workflow's `env:` block, and
`test_qnnci.py::test_every_version_the_job_pins_is_pinned_in_one_place`
refuses a second copy — `test_release.py`'s docstring records what a second
copy of a version cost this repository before (CI's default stayed three
releases behind, and the green runs the README cited were measuring an older
wheel than the README claimed).

| pin | value | why this value |
|---|---|---|
| `EXECUTORCH_VERSION` | `1.4.1` | `docs/QNN.md` §1.1 — the release §1.3's measurement was taken against |
| `EXPECTED_QNN_SDK_VERSION` | `2.37.0.250724` | ExecuTorch's own pin. **Asserted, not installed** — see below |
| `ANDROID_NDK_VERSION` | `26.3.11579264` (NDK r26c) | §1.1: "This example is verified with NDK 26c" |
| `TRANSFORMERS_VERSION` | `5.15.1` | the version in this project's `spike-venv`, i.e. the one every other claim in `docs/QNN.md` was measured under |
| `PYTHON_VERSION` | `3.13` | the abi3 floor this project targets, and the interpreter §1.3 measured on |
| model | `HuggingFaceTB/SmolLM2-135M`, `model.layers.0.mlp` | §4.1 and §11 ("No whole model, only one submodule") |
| `soc_model` | `SM8550` (default) | §6 below |

**The SDK version is asserted, not transcribed.** The workflow does not install
the SDK from a pinned URL. `docs/QNN.md` §5 step 4 records that
`pip install executorch` fetches it automatically on linux-x86
(`backends/qualcomm/__init__.py` gating on `is_linux_x86()`), and that is the
branch taken. The job then reads the version back through
`qnn.qnn_sdk_version()` — which resolves it from ExecuTorch's own
`download_qnn_sdk.py` — and **fails** if it is not `2.37.0.250724`. §1.1 is
explicit that transcribing the number is the mistake; comparing a literal
against a literal would be the same mistake with an extra step, and
`test_the_sdk_version_is_asserted_against_executorchs_own_answer` is what
prevents it collapsing into one.

### 2.1 Where the job deviates from `docs/QNN.md` §5, and why

The doc's procedure was written for a Linux box somebody owns. Two steps
cannot run verbatim on a hosted runner, and both deviations are in the
workflow's own comments:

| step | the doc says | the job does | reason |
|---|---|---|---|
| 4 | `export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.37.0.250724` and `source envsetup.sh` | provokes `install_qnn_sdk()` by importing `executorch.backends.qualcomm` | there is no `/opt/qcom` on a runner and no license-free way to put one there. The doc's *own* step 4 names this alternative in its first line |
| 5 | `cd "$EXECUTORCH_ROOT" && ./backends/qualcomm/scripts/build.sh` | clones `pytorch/executorch` at tag `v1.4.1` first | the doc assumes an `$EXECUTORCH_ROOT` already on the host. Pinned to the tag rather than `main`, because a runtime built from a different ExecuTorch than the one that lowered the `.pte` is a version-skew failure with no good symptom |

---

## 3. The Ubuntu decision

`docs/QNN.md` §1.1 records Qualcomm's verified host list: "**Ubuntu 22.04 LTS
(x64)**, CentOS Stream 9, WSL with Ubuntu 22.04". So the obvious pin is
`ubuntu-22.04`.

**It is pinned to `ubuntu-24.04` instead.** Three reasons, and the third
decides it:

1. `ubuntu-22.04` is on GitHub's runner-image retirement path. A job pinned to
   a retired image does not fail informatively — it fails to *schedule*, which
   reads as "CI is broken" rather than "the pin expired".
2. §1.1 also requires **g++ 13 or higher** for the AOT part. Ubuntu 22.04 ships
   g++ 11 and would need the toolchain PPA; 24.04 ships g++ 13 by default. So
   22.04 satisfies one half of §1.1 only by violating the other until patched.
3. What "verified on 22.04" buys is that the SDK's **prebuilt x86-64 host
   objects** (`libQnnHtp.so` and friends) find a new enough glibc/libstdc++.
   Prebuilt ELF objects are forward compatible with newer glibc and *not*
   backward compatible with older. So 24.04 (glibc 2.39) is the safe direction
   from 22.04 (glibc 2.35); the risky direction is an older image, and there
   is no older image.

It is `ubuntu-24.04` and **not `ubuntu-latest`**, so the host this measurement
is taken on cannot change under it without a commit —
`docs/QNN.md`'s numbers are keyed to hosts, and `ubuntu-latest` moves.
`test_the_runner_is_pinned_and_not_latest` enforces both halves.

**This reasoning is an argument, not a measurement.** Reason 3 in particular is
about how ELF symbol versioning works in general, not about this SDK on this
image. If the first run dies loading `libQnnHtp.so`, that argument was wrong;
the `runner` workflow input exists so `ubuntu-22.04` can be tried **without
editing the workflow first**, and §6 says to record the outcome here.

---

## 4. What the job asserts before it uploads anything

`docs/QNN.md` §6.2 is the reason this section exists: **QNN's dangerous
fallback is at compile time, not run time.** `QnnPartitioner` silently declines
whatever it cannot handle; those nodes stay in the program as portable CPU
kernels; the `.pte` loads, runs, and returns the right answer. So an artefact
can be uploaded, pushed to a phone, and executed entirely on the application
processor with nothing in its output saying so.

The checks are `docs/QNN.md` §5 step 7 plus step 6's assertion, and they are
split into **two classes that get handled differently**:

### 4.1 Integrity — failure means the artefact is NOT uploaded

The file is not what the job would be labelling it.

| check | source |
|---|---|
| `read_artefact(path).is_qnn` | §5 step 7, `assert a.is_qnn` |
| a delegate named `QnnBackend` is in `backend_ids` | §4.3 / `qnn.QNN_BACKEND_ID` |
| `match_device(art, SOC)` returns `(True, …)` | §5 step 7, `assert ok, why` |
| the delegation report describes a non-empty graph | see below |

`is_qnn` and `backend_ids` are checked **independently** rather than collapsed.
`is_qnn` is derived by `torchnative.export.qnn`; `backend_ids` is what the
artefact carries. If they ever disagree, that disagreement is the finding, and
a check that trusted the derived one could not see it.

The empty-graph branch is separate from the coverage check below because
`delegation_report` returns `delegated_fraction = 0.0` when `total_nodes` is 0.
Without it, "nothing was exported" would be reported as "the partitioner
claimed 0%" — a true number attached to a false cause, sending the reader to
QNN's op coverage when the problem is that there was no graph.

### 4.2 Coverage — failure fails the job and the artefact IS uploaded

`delegated_fraction > 0.9`, copied from `docs/QNN.md` §5 step 6 (strict `>`,
as the doc has it).

This one is separated because a genuine QNN program with poor coverage is
**the finding, not a defect in the file**. `docs/QNN.md` §11 says plainly that
this number for a real `LlamaMLP` under `QnnPartitioner` is unknown and may be
poor, citing `docs/REFOLD.md` §1.1 — this project's record of assuming op-table
coverage meant good partitioning and being wrong three times. Discarding the
artefact would discard the answer, so the job goes red and the file is uploaded
anyway, under a name that a red run makes unambiguous.

**The threshold is a module constant, not a workflow input.** A
`--min-delegated-fraction` flag would make dialling it down until the run went
green a one-word edit to a dispatch form, invisible in the diff of any file
that records a measurement.
`test_the_threshold_cannot_be_lowered_from_the_command_line` asserts that no
such flag exists and that the constant is still `0.9`.

> If the first run reports `delegated_fraction = 0.34`, the correct response is
> to write **0.34 into `docs/QNN.md` §11 as a measurement**, replacing the word
> "unknown". It is not to edit `MIN_DELEGATED_FRACTION`.

### 4.3 And it reports the number on every path

`docs/QNN.md` §11 asked for `delegated_fraction`, not for a pass/fail. It is
written to `$GITHUB_STEP_SUMMARY`, printed to the log, and saved to
`delegation_report.json` — which is uploaded **unconditionally**, before the
artefact gate, so a red run still hands back the measurement.

---

## 5. What was extracted, and what is tested on this machine

### 5.1 `tools/ci/qnn_lower.py`

`CLAUDE.md` §5.5 is the most-cited rule in this repository, and logic living
only inside a `run: |` block is its purest form: it cannot be imported, cannot
be nullified, and cannot be exercised anywhere except by pushing to a branch
and watching. `tools/ci/verify_published.py` is this repository's existing
answer and this file follows it.

Everything the job *decides* is in that file:

| | |
|---|---|
| `verify_facts(facts, soc, …)` | §4's checks, as a **pure function over a dict**. Every failure branch is reachable without executorch |
| `collect_facts(qnn, path, soc, report)` | reads §5 step 7's four answers off a real `.pte`. `qnn` is a parameter, so a double can stand in |
| `render_summary(verdict, soc)` | the step summary, including on the red paths |
| `Verdict` | `.integrity` / `.coverage` / `.integrity_ok`, which is what the upload gate reads |
| `MIN_DELEGATED_FRACTION = 0.9` | with `docs/QNN.md` §5 step 6 cited on the line |

The workflow step that runs it has **no** `if:`, no `continue-on-error` and no
`|| true`; `test_nothing_in_the_lowering_job_disables_its_own_assertions`
checks all three, in the `lower` job.

The one `continue-on-error` in the file is on the **`android-runtime` job**,
which builds `qnn_executor_runner` and `libqnn_executorch_backend.so`. That is
deliberate and narrow: those two are ordinary ExecuTorch build outputs
containing nothing measured, reproducible by any Linux x86-64 host with the
NDK, and an NDK or CMake breakage there must not turn a successful verified
lowering into a run somebody reads as failed. It is still reported red on its
own line.

### 5.2 `rust/torch_c/pytests/test_qnnci.py` — a new file, and why not `test_qnn.py`

**A new file.** `test_qnn.py`'s three fixtures all skip by name when their
environment is absent — `_qnn_et_fixture` skips unless
`$TORCHNATIVE_QNN_PYTHON` points at an interpreter with executorch. That is
right for what it tests and wrong for what is tested here: the decisions in
`qnn_lower.py` are exactly the ones a *successful* CI run never exercises, and
putting them behind an executorch-shaped skip would mean that on this project's
only machine they never run at all. A test that always skips is `CLAUDE.md`
§5.5 wearing a new hat.

So nothing in the file imports executorch, torch or torchnative. **30** tests,
two halves:

* **the verdict** — every refusal branch, driven by dicts, plus the ordering
  and independence properties: that poor coverage does *not* suppress the
  upload, that an empty graph is not reported as poor coverage, that the
  threshold is strict, that `collect_facts` asks `match_device` about the SoC
  it was told rather than a hardcoded one.
* **the workflow's honesty** — read as YAML: that an artefact is uploaded at
  all, that a `.pte` is among the uploaded paths, that the verifying step comes
  **before** every upload by index, that the `.pte` upload's `if:` consults
  `steps.lower.outputs.integrity_ok`, that no assertion is `|| true`'d or
  commented out, that the job invokes the extracted script rather than checking
  `is_qnn` inline, that the runner is pinned, and that the SoC default agrees
  across the workflow input, the push-trigger `||` fallbacks and the script.

**14 nullifications attempted, 0 uncaught — after one that was uncaught first.**
`test_the_sdk_version_is_asserted_against_executorchs_own_answer` originally
grepped the workflow's raw text for `qnn.qnn_sdk_version()`. Replacing the
actual call with a literal tuple went **undetected**, because the workflow also
names that function in a comment two lines above and the comment satisfied the
grep. The test now searches only the non-comment lines of `run:` scripts. This
is recorded rather than quietly fixed because it is §5.5's exact shape: a check
that a prose mention can satisfy is a check on the prose.

The other thirteen: gutting `verify_facts`; removing the `is_qnn`,
`match_device` and empty-graph branches; setting the threshold to `0.0`;
removing the upload gate; making the `lower` job `continue-on-error`; adding
`|| true` to the verifying step; switching to `ubuntu-latest`; drifting the SoC
default to SM8650; deleting the `.pte` from the upload; moving the upload above
the verification; and hardcoding the expected SDK literal into the comparison.

### 5.3 What is *not* tested here

The lowering itself. `verify_facts` is tested against dicts that *describe*
artefacts; no `.pte` is produced or read on this machine, because §1.3 is why
this whole document exists. The first real `PteArtefact` this code sees will be
on the runner. `test_qnn.py` already covers `read_artefact` and `match_device`
against real ExecuTorch-serialised programs where an interpreter allows it.

`actionlint` is not installed on this machine and could not be obtained, so the
workflow was validated by a **PyYAML parse plus a structural read** (job names,
step ordering, `if:` conditions, `env` keys) and a careful reading of the
expression syntax. GitHub Actions expression semantics — in particular
`inputs.runner || 'ubuntu-24.04'` resolving correctly on a `push` trigger,
where the `inputs` context is empty — is **argued, not verified**. §6 lists it.

---

## 6. First run: this job has never been run

Nobody has dispatched it. What follows is the list of things it is most likely
to fail on, roughly in the order it would hit them, and how the maintainer will
tell which one it was. `docs/VULKAN3.md` §6.1's rule applies to all of them —
the job is written so each refusal names the missing thing.

**1. The linux-x86_64 `executorch` wheel may also lack `PyQnnManagerAdaptor`.**
This is the most important unknown and the one most likely to end the run.
`docs/QNN.md` §1.3 measured the **macOS arm64** wheel and found
`executorch/backends/qualcomm/python/` absent. It did *not* measure the linux
x86-64 wheel; §1.3's argument is that the host libraries are x86-64 ELF, which
explains why the *Mac* wheel cannot have it — it does not establish that the
Linux wheel does.

*Symptom:* the job dies in the **"Fetch the QNN SDK"** step (or, if that
import succeeds and only the adaptor is missing, in "Lower and verify") with
`qnn_aot_refusal` printed verbatim, naming
`executorch.backends.qualcomm.python.PyQnnManagerAdaptor`. Both steps print
that refusal rather than letting an op builder's traceback be the top frame.

*What it means:* the AOT extension is only produced by a source build
(`backends/qualcomm/scripts/build.sh`, which the `android-runtime` job already
runs and which also emits `build-x86/`). The fix is to restructure the `lower`
job to build from source and put `build-x86` on `PYTHONPATH`, **not** to
weaken `qnn_aot_refusal()`. That refusal is correct and stays.

**2. `install_qnn_sdk()` may not fetch unattended.** The SDK download may
require accepting a Qualcomm license, or the URL may be gated.
*Symptom:* the "Fetch the QNN SDK" step fails at the
`import executorch.backends.qualcomm` line, or `QNN_SDK_ROOT` prints as `None`
and the version read fails.

**3. Disk.** The QNN SDK is several GB and a hosted runner has roughly 14 GB
free on `/`. *Symptom:* `No space left on device`, most likely during the
`android-runtime` job's submodule clone. *Fix:* the usual
`rm -rf /usr/share/dotnet /opt/ghc` reclaim step, or split the jobs further.

**4. The glibc argument in §3 may be wrong.** *Symptom:* an ELF load error
naming `libQnnHtp.so` and a `GLIBC_` or `CXXABI_` symbol version. *Action:*
re-dispatch with `runner: ubuntu-22.04`; if that also fails to schedule, the
SDK needs a container and this job needs rewriting around one. **Record the
outcome in §3 either way** — §3 is currently an argument and the first run is
what turns it into a measurement.

**5. `transformers==5.15.1` against the torch `executorch==1.4.1` pulls
(2.14.0).** *Symptom:* a pip resolver conflict in the install step, or an
`AttributeError` deep in `from_pretrained`. Note this pin is chosen to match
the interpreter every other `docs/QNN.md` claim was measured under; if it has
to move, that is a divergence worth stating rather than a free choice.

**6. The submodule path.** *Symptom:* `_resolve_submodule` refuses by name,
printing which component of `model.layers.0.mlp` was absent and what children
existed at that level. Almost certainly a `transformers` version issue, i.e.
symptom 5 in another costume.

**7. `delegated_fraction` below 0.9.** *Symptom:* the job is red with exactly
one `FAIL` line, the artefact **is** uploaded, and the step summary carries the
number. **This is a successful run, not a broken one.** `docs/QNN.md` §11 said
this number was unknown and might be poor; the job just measured it. Record it
in §11. Do not touch `MIN_DELEGATED_FRACTION`.

**8. `v1.4.1` may not be the tag name.** The `android-runtime` job clones
`--branch "v${EXECUTORCH_VERSION}"`. *Symptom:* `Remote branch v1.4.1 not
found`. This is `continue-on-error`, so it will not fail the run — check the
job's own line, do not read the run's overall colour.

**9. Actions expression syntax.** `actionlint` never ran (§5.3). *Symptom:*
either a workflow-parse error before any step executes, or — the quieter one —
`inputs.runner || 'ubuntu-24.04'` resolving oddly on the `push` trigger and the
job scheduling on an unexpected image. The "What we are standing on" step
prints `uname -a` first, so **read that line on the first run** and confirm it
says 24.04.

**And regardless of the outcome:** the artefact this produces is the input to
`docs/QNN.md` §5 step 8. Steps 9–14 — staging, running, the skeleton-removal
negative control, the optrace decode — are still a human with a phone, and
until they happen `docs/QNN.md` §6.4 stands unchanged.

---

## 7. What is not done

* **The job has not run.** Everything above is a description of a file.
* **No count marker.** This document adds no `DOCWATCH: count`, because this
  round measured no number. A `ge` bound on a number nothing has produced is a
  marker that cannot fail, which is the thing `CLAUDE.md` §5.5 and the
  `golden_cases_failed` history are both about. When the first run reports
  `delegated_fraction`, it belongs in `docs/QNN.md` §11 as a measurement.
* **The device half is untouched.** `docs/QNN.md` §5 steps 9–14 need the
  physical Galaxy Tab S9 Ultra and are unchanged by any of this.
* **Nothing in `torchnative/` changed.** `qnn.py` and `qnn_device.py` are
  byte-identical; in particular `qnn_aot_refusal()` still refuses on this
  machine, which is correct — the CI host is a host that *satisfies* the
  refusal, not a reason to remove it.
* **The quantised path is still not built.** `kHtpQuantized` is QNN's default
  and this job passes `fp16=True`. `docs/QNN.md` §11 and `docs/QUANT2.md` §2.
* **No `actionlint`.** §5.3.
