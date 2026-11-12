# Verify your download

Every Caminus release is built twice from its git tag: on the maintainer's machine and on GitHub's runners, in the same
public build image. It is published only when the two builds are byte for byte equal. GitHub signs a record of the CI
build (SLSA Build Level 3 provenance: which repository, workflow, tag and commit built which file), and PyPI keeps a
signed record of who uploaded each file (a PEP 740 attestation). This page shows how to check both and how to rebuild a
release yourself, so you don't have to take a download on trust.

**Which files this covers.** Releases after 0.2.1 (0.1.0, 0.2.0 and 0.2.1 were built before this and carry no
provenance). Nightly builds carry provenance once their workflow can run, after the next merge of `testing` into
`main`; until then a nightly's files are checked only against the hashes in its tag. Replace `X.Y.Z` below with the
version you want.

## A file from a GitHub Release

You need [the GitHub CLI](https://cli.github.com) (`gh`, a recent one: it needs the `attestation` command).

```bash
gh release download vX.Y.Z -R jdominick05/caminus --pattern 'caminus-*' --pattern SHA256SUMS
sha256sum -c SHA256SUMS
gh attestation verify caminus-X.Y.Z-py3-none-any.whl -R jdominick05/caminus \
   --signer-workflow jdominick05/caminus/.github/workflows/build.yml \
   --source-ref refs/tags/vX.Y.Z --deny-self-hosted-runners
```

Each of the last command's options closes a gap:

- `--signer-workflow`: the file was built by the repository's release build (`.github/workflows/build.yml`), not by
  some other workflow in the repository;
- `--source-ref`: it was built from the release tag. Builds from test branches are signed by the same workflow, and this
  option tells them apart;
- `--deny-self-hosted-runners`: it was built on a GitHub-hosted runner, not on a machine someone controls.

Run the same command on `caminus-X.Y.Z.tar.gz` (the sdist) if you use it. `sha256sum -c` alone proves only that the
files match each other, not who made them; the attestation is the part that does.

## The same file from PyPI

PyPI serves the files the CI build made, so the same attestation applies to them:

```bash
pip download caminus==X.Y.Z --no-deps --only-binary :all: -d caminus-dl
gh attestation verify caminus-dl/caminus-X.Y.Z-py3-none-any.whl -R jdominick05/caminus \
   --signer-workflow jdominick05/caminus/.github/workflows/build.yml \
   --source-ref refs/tags/vX.Y.Z --deny-self-hosted-runners
```

To check the upload as well (that the file on PyPI was uploaded by this repository's release workflow, not by a
stolen account), use PyPI's attestation:

```bash
python -m pip install pypi-attestations
python -m pypi_attestations verify pypi --repository https://github.com/jdominick05/caminus \
   pypi:caminus-X.Y.Z-py3-none-any.whl
```

Then install exactly the file you checked, rather than downloading it again:

```bash
pip install ./caminus-dl/caminus-X.Y.Z-py3-none-any.whl
```

## Rebuild a release yourself

The strongest check needs no trust in the maintainer or in GitHub's signing: build the release from source and compare.
You need git and Docker; the build runs on the CPU (no GPU, nothing is installed on your system) and pulls the public
build image (about 350 MB unpacked).

```bash
git clone https://github.com/jdominick05/caminus.git && cd caminus
git checkout vX.Y.Z
tools/release/in_image.sh tools/release/release_build.sh
diff dist/SHA256SUMS <(gh release download vX.Y.Z -R jdominick05/caminus --pattern SHA256SUMS --output -)
```

No output from `diff` means your build and the published files are identical. The image is the one named by digest
in the tag's `packaging/build-image/IMAGE`; its contents and the licences of what it holds are listed in
[packaging/build-image/](../packaging/build-image/licenses/README.md), and what goes into a build in
[releasing](dev/releasing.md#the-build-image).

## What this does not cover

- **Caminus's dependencies.** The provenance covers the `caminus` package only. PyTorch for ROCm, numpy, tokenizers and
  the others come from their own publishers, with their own signatures, if any. To pin them, install from a lock file
  with `pip install --require-hashes`.
- **Model weights.** They come from Hugging Face, not from this project. `caminus download` checks each file's size and
  sha256 against the model catalog and refuses a mismatch ([models](models.md)); that check is separate from the one
  on this page.
- **The compiler.** The build image carries AMD's TheRock ROCm compiler as AMD built it. You can read and pull the exact
  image, but trusting those binaries means trusting AMD, as with any compiler.
- **Who checked the level.** SLSA levels are self-assessed; no one has audited this setup. How the builds are made and
  signed is in [releasing](dev/releasing.md#slsa), so you can judge it yourself.
- **A nightly's source.** A nightly's provenance names the branch `testing` and the commit the nightly was made from, not
  the nightly's own tag commit, which is that commit plus a version and hash update ([releasing](dev/releasing.md#slsa)).
  For a nightly, use `--source-ref refs/heads/testing`.
