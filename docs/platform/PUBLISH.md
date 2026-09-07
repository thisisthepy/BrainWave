# `main` is a published snapshot, not a merge target

`develop` is the repository. `main` is what a person who has never read this
repository sees when they land on GitHub: the code that is on PyPI, the
workflows that verify it, a README, and nothing else. Nobody develops on it,
nothing is ever merged into it, and it carries exactly one commit per release.

The script is `tools/release/publish_main.sh`.

<!-- DOCWATCH: symbol-in-file tools/release/publish_main.sh EXCLUDE_PATHS present -->

---

## 1. Why this is not `git merge` followed by `git rm`

The obvious implementation is to merge `develop` into `main` and delete the
paths that should not ship. It does not work, and it fails quietly.

A deletion made in a merge commit does not survive the next merge. Three-way
merge resolves "deleted on one side, present on the other" as a deletion only
while the other side has not touched the path *since the merge base*. The merge
base advances with every release, so at release N+1 every document `develop` has
edited since release N comes back — and every document it has *not* edited has
to be deleted again by hand, every release, forever. Forgetting once ships the
whole 179-document `docs/` tree, and nothing in the process says so.

So `main`'s tree is **built from** `develop`'s rather than merged into:

```sh
D=$(git rev-parse develop)
git read-tree "$D"                 # develop's tree, into a SCRATCH index
git rm -r --cached <excluded>      # drop entries from that index only
TREE=$(git write-tree)
NEW=$(git commit-tree "$TREE" -p "$(git rev-parse main)" -m "Release $VERSION")
git update-ref refs/heads/main "$NEW"
```

Two consequences worth stating:

* **Nothing is ever restored.** There is no "the other side still has it",
  because there is no other side. Each release is a fresh derivation.
* **The working tree is never touched.** Everything runs through
  `GIT_INDEX_FILE` pointed at a temporary file, so `git rm --cached` removes
  index entries and no files. `git status` is identical before and after.

`main`'s history stays linear: one commit per release, each one's single parent
being the previous release.

<!-- DOCWATCH: symbol-in-file tools/release/publish_main.sh GIT_INDEX_FILE present -->
<!-- DOCWATCH: symbol-in-file tools/release/publish_main.sh commit-tree present -->

---

## 2. The exclusion list, with a reason per entry

It is an **exclusion** list. A path named in neither column is kept; silence
means keep. `rust/torch_c/pytests/test_publish.py` holds an independent copy and
asserts the two are identical, so neither can be edited alone.

### Excluded

| Path | Why |
|---|---|
| `docs/` | 179 documents recording what each round measured. The record of how the work was done, not part of what ships — and the bulk of what a user would otherwise clone. |
| `CLAUDE.md` | Instructions to agents working **on** the repository. Meaningless to somebody consuming the wheel. |
| `PROJECT.md` | The same, for project structure and reasoning. `README.md` is kept: it is the front page. |
| `rust/torch_c/pytests/` | The development gate — ~50 suites needing a vendored tree, a pinned spike venv, Android emulators and a Vulkan ICD. None of it runs from an installed wheel. |
| `tools/docwatch/` | Checks markers in `docs/`, which is excluded. Keeping it would ship a checker with nothing to check. |
| `tools/golden/` | Compares values against **upstream** torch, which must be installed alongside. A user of this distribution by definition does not have one. |
| `tools/spike/` | Scratch probes bound to this machine's spike venv. |
| `tools/bench/` | Measurement harnesses whose numbers only mean anything on this project's own hardware. |
| `tools/scan/` | An audit tool over upstream's source — the same class as `docwatch` and `golden` above. |
| `tools/colab/` | A notebook for verifying on a Colab box. Developer tooling, not part of the distribution. |

### Kept, and why the non-obvious ones are kept

| Path | Why |
|---|---|
| `.github/` | **All three workflows are `workflow_dispatch`, and GitHub only renders the "Run workflow" button for a workflow file present on the DEFAULT branch — which `main` is.** Stripping `.github/` would not merely tidy main's tree; it would remove the manual trigger for all three workflows repository-wide, while the files on `develop` continued to look perfectly healthy. |
| `tools/ci/` | A pair with the above: the workflows invoke `tools/ci/verify_published.py` and `tools/ci/qnn_lower.py` by path. Splitting them leaves a button that fails at "No such file". And `verify_published.py`'s subject is *what is on PyPI*, which is exactly what `main` represents. |
| `tools/wheel/` | Builds and verifies the wheel. Without it `main` cannot produce the thing it is a snapshot of — and its absence is invisible to the leak check (§4). |
| `scripts/` | Kept ON PURPOSE, not by silence: `tools/wheel/build.py` names `scripts/device_android.sh build` as the FIX in its own refusal message when the Android artefact is stale, so a `main` without it would print a recovery command `main` cannot run. |
| `vendor/` | Lays down the upstream tree and installs `_C`. `torchnative/src/main/torch/` is gitignored, so **every** checkout, `main`'s included, starts without one; `vendor/` is the only way to get it. |
| `rust/torch_c/src/` | The extension itself. |
| `torchnative/src/main/` | The Python package. |
| `setup.py`, `pyproject.toml`, `LICENSE`, `README.md` | The distribution's own metadata and front page. |

<!-- DOCWATCH: symbol-in-file tools/release/publish_main.sh workflow_dispatch present -->

---

## 3. README links

`README.md` carries ~63 clickable references into `docs/` — 44 markdown
`](docs/…)` targets and 19 HTML `href="docs/…"` ones. Every one of them 404s on
`main` once `docs/` is gone.

The script rewrites both forms to absolute URLs against `develop`
(`https://github.com/thisisthepy/torchnative/blob/develop/docs/…`) **in the
published tree only**. `develop`'s own README keeps its relative links, which
are correct there; the rewrite happens on a blob written straight into the
scratch index and never reaches the working tree.

Two things are deliberately **not** rewritten:

* **Prose citations in the README** — `` `docs/META.md` `` in running text, of
  which there are ~71. They are not links, nobody clicks them, and they do not
  404.
* **Comments in kept files** — the workflows, `tools/wheel/*.py` and
  `vendor/*.sh` cite `docs/…` in their comments throughout. Same reasoning, plus
  a cost: rewriting them would enlarge the release diff on every single round in
  exchange for nothing a user can see.

---

## 4. The two checks, and which one matters

```
check_leak     no excluded path exists in the published tree
check_build    the published tree still BUILDS A WHEEL
```

**The second is the real check.** A wrong exclusion — `tools/` where
`tools/bench` was meant, say — produces a branch that is spotlessly clean and
completely unable to ship, and `check_leak` passes it without a murmur. The
leak check can only see paths that are *present*; the failure that actually
reaches a user is a path that is *absent*.

> **The leak check alone gives you a tidy branch that cannot ship.**

`check_build` checks the new commit out into a scratch worktree, runs
`vendor/vendor_torch.sh` and `vendor/install_shim.sh` (which is what proves
`vendor/` survived), then `tools/wheel/build.py`, and fails unless a `.whl`
lands in `dist/`. `--skip-build` exists for rehearsals and says loudly that the
round proved tidiness and not shippability; do not publish on its green.

Both checks are nullified by `test_publish.py` rather than assumed: the leak
test runs a copy of the script with the exclusion step replaced by `true` and
requires it to go red, and the build test runs the same script against a
fixture that builds and one that refuses, requiring different verdicts
(CLAUDE.md §5.5).

---

## 5. Procedure for a maintainer

Publishing is a coordinating-session action taken with the user's approval, in
the same class as the PyPI upload (CLAUDE.md §5.7). The script moves a local ref
and **pushes nothing**.

1. Land the release on `develop` and upload to PyPI first. `main` is a snapshot
   of what *is* published, so it comes after.

   **Check that local `main` is at `origin/main` before you start.** The new
   release commit's parent is the *local* tip, so publishing on a stale one
   forks the published history off the remote's. The script warns when the two
   differ, but it does not refuse — a scratch target ref has no upstream.

2. **Rehearse on a scratch ref.** Never rehearse against `main`:

   ```sh
   tools/release/publish_main.sh --target-ref refs/heads/_publish_test
   ```

   Read the output: the excluded paths it dropped, the file count, the wheel it
   built. An `ABSENT` line means the exclusion list names a path that no longer
   exists — usually a rename, which means the new name is now shipping.

3. Inspect and delete the scratch ref:

   ```sh
   git ls-tree -r --name-only _publish_test | wc -l
   git branch -D _publish_test
   ```

4. **Publish**, with the user's approval:

   ```sh
   tools/release/publish_main.sh
   git push origin main
   ```

The gate (`rust/torch_c/pytests/run.sh`) runs `test_publish.py` on every round,
so the exclusion list, the non-merge mechanism and both checks are held in place
between releases rather than only at one.
