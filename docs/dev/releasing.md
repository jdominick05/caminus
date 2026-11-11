# Releasing Caminus: CI, the PyPI release and the nightly builds

Four GitHub Actions workflows (`.github/workflows/`) and a few tools carry a release and the nightly builds. One fact
shapes all of it: **CI never builds the shipped code objects.** The gfx1100 code objects in the wheel were built and validated with a TheRock ROCm nightly
(10.2.0a20261011: amdclang++ and the offload driver's device LTO), and the gates that validated them hold their exact
instructions and bitwise numerics (docs/dev/packaging.md, "Code objects"). That toolchain is not available at a public URL CI
could install from, and a different compiler gives different code. So the maintainer builds the release files on the
development machine, and CI checks them against hashes committed in the repository.

Every action the workflows use is pinned by its commit SHA, with the release tag in a trailing comment
(`actions/checkout@08c6903c... # v5.0.0`); to update one, replace both.

## ci.yml: every push and pull request

On `ubuntu-24.04`, no GPU, no model files:

| Step | What |
|---|---|
| Dependencies | numpy, Jinja2 and tokenizers at their tested versions; tests/serve's references (transformers' `apply_chat_template`, the anthropic SDK's stream accumulator); PyTorch's CPU wheel (nothing in CI launches a kernel); build and twine |
| Byte-compile | `python -I -m compileall -q python tools tests` |
| Privacy scan | `tools/release/privacy_scan.py .`: secret patterns, machine paths, agent job directories, session links, quoted maintainer instructions and private names (matched by salted digests, so the tool names none of them) |
| Tokenizer metadata | the first 11 MiB of the Coder's shard 1 at the manifest's pinned revision (an HTTP Range request), turned into a vocab-only GGUF by `tools/release/gguf_vocab_only.py` for tests/serve; optional: without it tests/serve is skipped |
| CPU suites | `tests/ci_cpu.py`: tests/serve, test_packaging and the other suites that need no model weights; the suites that need the Coder's weights are skipped with the reason (`tests/ci_cpu.py --list`) |
| sdist | `python -m build --sdist` and `twine check --strict` (CI's sdist has no code objects; the release's does) |
| Vendored code | `tools/release/vendor_gguf.py --check` against upstream llama.cpp at 5aeac37a35; `tools/release/vendor_mmq.py --check --commit e4e3f8eda` against the ignite-llama fork (github.com/jdominick05/ignite-llama): kernels/ggml_mmq's headers name the fork's development commit 842653396, published as e4e3f8eda with identical `ggml/`, `common/` and `src/`, so the check reads the files at e4e3f8eda and compares them with headers that carry the label 842653396 (`--label`, its default) |

The same checks run locally, except the downloads:

```bash
python -I tools/release/privacy_scan.py .
CODER=/path/to/a/Coder/shard-1-or-vocab-only.gguf python -I tests/ci_cpu.py --scratch <a new directory>
python -I tools/release/vendor_gguf.py --check --src <a llama.cpp clone holding 5aeac37a35>
python -I tools/release/vendor_mmq.py --check --src <an ignite-llama clone holding e4e3f8eda>   # --commit e4e3f8eda --label 842653396: the defaults
```

tests/serve's `test_cancel_latency` is timing-sensitive; on a busy shared runner it is the likeliest flake.

## kernels.yml: compile-only, weekly and on demand

It builds every kernel with a stock ROCm (the `rocm/dev-ubuntu-24.04` image) to catch source breakage, and prints how many
runtime objects happen to equal the shipped ones. It never produces release artifacts, uploads nothing, and its objects are
not expected to equal the shipped ones. build.sh compiles against the ignite-llama fork's ggml headers at e4e3f8eda (the K3
modules use helpers the fork adds), which the job fetches from github.com/jdominick05/ignite-llama.

## release.yml: a published GitHub Release (or by hand with its tag)

1. **verify** (`contents: read` only): checks out the tag, downloads the Release's attached `caminus-X.Y.Z-py3-none-any.whl`,
   `caminus-X.Y.Z.tar.gz` and `SHA256SUMS`, and runs `tools/release/verify_release.py DIST --tag vX.Y.Z`: one version everywhere,
   SHA256SUMS, the wheel's RECORD, the code objects exactly `caminus.device.kernels.RUNTIME_SET` with each sha256 equal to
   provenance.json and to the committed `packaging/gfx1100-kernels.sha256` at the tag, provenance naming the tag's commit and
   a clean tree, the kernel tree hash recomputed at the tag, the sdist's objects identical to the wheel's, and no private
   or secret text in either archive. Then `twine check --strict`. The verified files go to the next job as an artifact.
2. **publish** (environment `pypi`, no token permissions): waits for the environment's required reviewer, then uploads the
   two files with twine 6.1.0 (`twine upload`; not pypa/gh-action-pypi-publish, whose Docker image cannot be pinned by commit SHA) and the environment secret `PYPI_API_TOKEN` (a PyPI token scoped to the
   `caminus` project).

Every job runs only in `jdominick05/caminus` (`if: github.repository == ...`), so a fork or a mirror never publishes.

## Nightly builds: nightly.sh on a timer, then nightly.yml

A nightly is the head of `testing`, built like a release and published without a person in the loop: a GitHub
pre-release tagged `vX.Y.Z.devYYYYMMDD` with the wheel, the sdist and SHA256SUMS, and the wheel on PyPI as
`caminus X.Y.Z.devYYYYMMDD`. `X.Y.Z` is the next release (the tree's `__version__` when it has no tag yet, else its next
patch), so a nightly sorts after the last release and before the next one, and `pip install caminus` never picks one.

```bash
pip install "caminus>=0.2.2.dev0"            # the newest build, nightly or release
pip install "caminus==0.2.2.dev20261112"     # one nightly
```

Name a dev version in the request, as above, rather than passing `--pre`: pip then takes pre-releases of Caminus only,
while `--pre` also takes pre-releases of every dependency, which nothing here is tested against.

**The build runs on the maintainer's machine** for the reason at the top of this page: `tools/release/nightly.sh`, started at
midnight (local time) by a systemd user timer ([packaging/nightly/](../../packaging/nightly/README.md)). A run:

1. does nothing when `HOLD` exists in its state directory, or while a guarded GPU job other than the server is running
   (it builds the kernels twice, which would disturb a benchmark's timings);
2. reads the head of `testing` and the nightly tags. A head that has its nightly tag and pre-release is done: one nightly
   per commit, none when the branch did not move. The test keeps no state of its own
   (`tools/release/nightly_plan.py built`: a nightly tag whose commit's parent is the head), so a night that was skipped
   or failed is made up by the next run;
3. needs a completed, successful `ci.yml` push run on that commit;
4. in a fresh clone at the head: `release_build.sh --write-hashes` (the kernels, by the real toolchain), then **one
   commit** on top of the head with the version and that build's `packaging/gfx1100-kernels.sha256`, tagged
   `vX.Y.Z.devYYYYMMDD`. The commit is on no branch; the tag is its only name. It exists because the files are verified
   the way a release's are: a clean checkout whose `__init__.py` holds the version, hashes committed at that commit,
   provenance naming it;
5. in a second fresh clone at the tag: `release_build.sh`, which builds the kernels again, refuses objects that differ
   from the committed hashes (every nightly is also a reproducibility test of the build) and ends with
   `verify_release.py`;
6. checks the tag (`nightly_plan.py check-tag`) and scans the tagged tree (`privacy_scan.py`), and only then pushes the
   tag and creates the pre-release (`--prerelease --latest=false`, so the "Latest" release stays the last real one);
7. starts `nightly.yml` on the branch for that tag (`gh workflow run nightly.yml --ref testing -f tag=vX.Y.Z.devYYYYMMDD`).

`tools/release/nightly.sh --dry-run` does everything except the push, the pre-release and that start.

**nightly.yml** has one trigger, a start by hand or by the script, and it runs **on the branch `testing` only**. GitHub
runs the workflow file of the ref a run starts on. A run for a release event would execute the copy in the tag's commit,
so a crafted tag could carry an edited copy that skips every check and uploads any version of `caminus`; started on
`testing`, the file is always the reviewed one, because the branch takes changes through pull requests only. Two things
hold that in place: the verify job refuses any other ref, and the environment `pypi-nightly`, which the PyPI trusted
publisher names, admits the branch `testing` and no tag. GitHub starts a workflow this way only once its file is on
the default branch, so the upload begins after the first merge of `testing` into `main` that holds `nightly.yml`; until
then the script reports that the start was refused and the pre-release stands.

No person approves a nightly, so the verify job is the gate between a published pre-release and PyPI:

- the run is on `refs/heads/testing`, the tag has the nightly form and its Release is a pre-release;
- `nightly_plan.py check-tag --branch origin/testing`: the tag's commit has one parent, the parent is a commit of
  `testing`, and the commit changes only `python/caminus/__init__.py` (its `__version__` line, to the tag's version) and the
  hashes file (hash lines only). The job runs **its own checkout's copy of the checker** on a second checkout that
  holds the tag, before any file of the tag runs: a tag that passes holds no tool that `testing` does not, so the tag's
  own `verify_release.py` below is reviewed code;
- a completed, successful `ci.yml` push run exists on that parent;
- `verify_release.py --tag`: everything it checks for a release, which includes that every package file in the wheel is
  the checkout's tracked file byte for byte. Only the code objects, their provenance and the panel come from the
  maintainer's build, as in a release.

Its publish job (environment `pypi-nightly`, `id-token: write` and nothing else, no checkout) uploads **the wheel only**
with twine 6.1.0 through PyPI Trusted Publishing: the job's OIDC identity is exchanged for a short-lived upload token
(PyPI's documented manual exchange), so no PyPI secret is stored for it. It runs only when the repository variable
`NIGHTLY_PYPI` is `true`. The PyPI project's trusted publisher names owner `jdominick05`, repository `caminus`, workflow
`nightly.yml` and environment `pypi-nightly`. PyPI does not limit a publisher to dev versions, which is why the
environment's branch rule and the `testing`-only trigger matter: **do not widen the environment to tags.**

What a nightly is not: a release's GPU gates and manual tests (step 5 of "Making a release") are not run on it. Every
change in it passed its own pull request's gates on the way into `testing`.

Costs to know about: each nightly leaves a tag and a pre-release on GitHub and a 6.5 MB wheel on PyPI, for good (a PyPI
version cannot be replaced or reused; the project's default size limit is 10 GB).

## Making a release (the maintainer)

1. **The version:** set `__version__` in `python/caminus/__init__.py` (and the docs that name it). A version can be uploaded to
   PyPI once; a fix is the next patch version.
2. **The code-object hashes** (only when kernels/ or build.sh changed since the last release): in a fresh clone,
   `PY=<python with ROCm PyTorch> ROCM_PATH=<TheRock> IGNITE=<ignite-llama at e4e3f8eda> tools/release/release_build.sh --write-hashes`,
   review the change to `packaging/gfx1100-kernels.sha256` and commit it.
3. **Tag** the release commit `vX.Y.Z` and push the commit and the tag.
4. **Build** in a fresh clone at the tag: `PY=... ROCM_PATH=... IGNITE=... tools/release/release_build.sh`. It refuses a tree with
   changes or earlier build output, runs ./build.sh and tools/release/stage_kernels.py, refuses staged objects that differ from the
   committed hashes, builds the wheel and the sdist into `dist/`, makes them reproducible (`tools/release/repro_dist.py`, below),
   writes `dist/SHA256SUMS` and runs tools/release/verify_release.py. (`--untagged` does the same at an untagged commit for a
   rehearsal; the files name the commit, so a rehearsal's dist of the commit that is tagged later verifies against the tag,
   and equals a build at the tag byte for byte.)
5. **Try the wheel** in a fresh environment: `pip install torch --index-url https://download.pytorch.org/whl/rocm7.2`, then the
   wheel; `caminus doctor`, `caminus doctor --gpu`, a short `caminus serve` (docs/quickstart.md).
6. **Create the GitHub Release** with the three files attached, e.g.
   `gh release create vX.Y.Z dist/caminus-X.Y.Z-py3-none-any.whl dist/caminus-X.Y.Z.tar.gz dist/SHA256SUMS --title "Caminus X.Y.Z" --notes-file NOTES.md`.
   Publishing it starts release.yml.
7. **Approve** the `pypi` deployment once verify has passed (the run's page on GitHub). The upload follows.

## Reproducible release files

Two `release_build.sh` runs of the same commit give the same wheel, sdist and SHA256SUMS, byte for byte (the code objects
themselves are checked against `packaging/gfx1100-kernels.sha256`). Nothing in them depends on when or where they were built:

- `SOURCE_DATE_EPOCH` defaults to the commit's committer time (`git log -1 --format=%ct`; an exported value wins and is
  printed). It is provenance.json's `built` (`built_from: SOURCE_DATE_EPOCH`; tools/release/stage_kernels.py) and the wheel's entry
  times (setuptools' bdist_wheel reads it).
- `tools/release/repro_dist.py dist` rewrites the sdist, which setuptools writes with the checkout's file times, the build account's
  uid / gid and user / group names, the umask's modes and the clock's gzip time: every member at `SOURCE_DATE_EPOCH`, owner
  0:0 with no names, modes 0644 / 0755, gzip level 9 with that time and no file name; and the wheel's entry modes (0644 /
  0755; its RECORD hashes contents only). Member order and contents are unchanged and checked.
- `tools/release/verify_release.py` checks that form (one time, == provenance's `built`), so release.yml refuses files built another
  way.

**The code objects also depend on two absolute paths.** clang's HIP compile gives every object a compile-unit id, a symbol
`__hip_cuid_<hash>` whose hash covers the compile command's options, and those hold the absolute `-I<IGNITE>/ggml/...`
include paths and the toolchain's (`ROCM_PATH`) paths. The 28 objects that include ggml headers (`k3_gguf*`,
`prefill_mmq_*`, `prefill_moe_*`, `prefill_dense_*`) therefore differ when `IGNITE` or `ROCM_PATH` is another path string:
the same instructions and read-only data (`.text` and `.rodata` identical), 36 to 841 bytes of symbol names and the symbol
hash and string tables built from them; the 8 others do not. The repository's own path and the temporary directory
do not enter them. A byte-for-byte rebuild needs the same toolchain, the same fork commit at the same absolute `IGNITE` path and
the same `ROCM_PATH` string as the release build; `packaging/gfx1100-kernels.sha256` holds the release build's hashes, and
kernels.yml's "N of 36 equal" count is low for this reason as well as the compiler's.

To check a published release on the maintainer's layout: clone the tag, run `tools/release/release_build.sh` with those paths, and
compare `dist/SHA256SUMS` with the Release's.

## The privacy scan's private names

`tools/release/privacy_scan.py` holds salted sha256 digests of lowercased private terms (words, word pairs and path segment pairs),
never the terms. To add one: `python -I tools/release/privacy_scan.py --digest 'the term'` and add the digest to `PRIVATE`.
