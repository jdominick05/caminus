#!/usr/bin/env bash
# The release files, built on the maintainer's machine from a clean checkout of the public repository at a release tag
# (docs/dev/releasing.md). CI never builds the shipped code objects: the toolchain that built and validated them (TheRock ROCm
# 10.2.0a20261011) is not available to CI, so they are built here and checked in CI against the committed hashes.
#
#   tools/release/release_build.sh                 at a tag vX.Y.Z on HEAD (X.Y.Z = python/caminus/__init__.py): ./build.sh, the
#                                          staged runtime set == packaging/gfx1100-kernels.sha256, then dist/ (the wheel, the
#                                          sdist, SHA256SUMS) and tools/release/verify_release.py on it
#   tools/release/release_build.sh --untagged      the same at an untagged commit (a rehearsal; the files name the commit, not a tag)
#   tools/release/release_build.sh --write-hashes  build and stage, then write packaging/gfx1100-kernels.sha256 from the staged
#                                          provenance and stop: commit that file before tagging (it changes only when the
#                                          code objects do)
#
# Environment: PY (a Python 3.12 with pip and setuptools 84.2.1, the release image's; default python3), ROCM_PATH and IGNITE as build.sh
# reads them (TheRock, and a checkout of the ignite-llama fork at e4e3f8eda, github.com/jdominick05/ignite-llama, for its ggml
# headers: provenance.json records its commit). The objects that include ggml headers
# carry clang's HIP compile-unit id (__hip_cuid_<hash of the options>), so their bytes depend on the IGNITE and ROCM_PATH
# path strings too: build with the ones packaging/gfx1100-kernels.sha256 was made with (docs/dev/releasing.md).
# SOURCE_DATE_EPOCH (default: HEAD's committer time, i.e. the tagged commit's): provenance.json's `built`, the wheel's entry
# times and, through tools/release/repro_dist.py, the sdist's member and gzip times; with the owner, user / group names and modes
# normalized there too, two builds of the same commit give byte-identical files (docs/dev/releasing.md).
# Refused: a tree with changes or untracked files, earlier build output (build/, python/caminus/hsaco/, .pybuild/,
# *.egg-info, dist/: clone afresh), a tag that does not name HEAD, staged objects that differ from the committed hashes.
# Nothing is deleted. CPU only: building the kernels needs no GPU.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
mode=tag
case "${1:-}" in
    '') ;;
    --untagged) mode=untagged ;;
    --write-hashes) mode=hashes ;;
    *) echo "usage: tools/release/release_build.sh [--untagged | --write-hashes]" >&2; exit 2 ;;
esac
PY=${PY:-python3}
die() { echo "release_build: $*" >&2; exit 2; }

git rev-parse --git-dir > /dev/null 2>&1 || die "not a git checkout"
[ -z "$(git status --porcelain --untracked-files=all)" ] || die "the tree has changes or untracked files (git status): build from a clean checkout"
for d in build python/caminus/hsaco python/caminus/share/caminus-panel .pybuild dist; do
    [ ! -e "$d" ] || die "$d exists: build from a fresh clone (nothing here is deleted)"
done
ls -d python/*.egg-info > /dev/null 2>&1 && die "python/*.egg-info exists: build from a fresh clone"
VER=$(sed -n "s/^__version__ *= *['\"]\([^'\"]*\)['\"].*/\\1/p" python/caminus/__init__.py)
[ -n "$VER" ] || die "no __version__ in python/caminus/__init__.py"
HEAD=$(git rev-parse HEAD)
TAG=v$VER
if [ "$mode" = tag ]; then
    TC=$(git rev-parse --verify --quiet "refs/tags/$TAG^{commit}" || true)
    [ "$TC" = "$HEAD" ] || die "the tag $TAG does not name HEAD ($HEAD; the tag: ${TC:-none}): tag the release commit, or --untagged for a rehearsal"
fi
export BUILD_KERNELS_ONLY=1   # build.sh: the code objects only (the runtime tests are not shipped and need torch)
CT=$(git log -1 --format=%ct "$HEAD")
SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-$CT}
[[ "$SOURCE_DATE_EPOCH" =~ ^[0-9]+$ ]] || die "SOURCE_DATE_EPOCH must be whole seconds since 1970, not '$SOURCE_DATE_EPOCH'"
export SOURCE_DATE_EPOCH
umask 022
echo "release_build: caminus $VER at $HEAD ($mode); ROCM_PATH=${ROCM_PATH:-(build.sh default)} IGNITE=${IGNITE:-(build.sh default)}"
echo "release_build: SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH ($(date -u -d "@$SOURCE_DATE_EPOCH" +%Y-%m-%dT%H:%M:%SZ); $([ "$SOURCE_DATE_EPOCH" = "$CT" ] && echo "the commit's committer time" || echo "from the environment: the commit's is $CT"))"

mkdir -p build
./build.sh > build/release-build.log 2>&1 || { tail -30 build/release-build.log >&2; die "./build.sh failed (build/release-build.log)"; }
echo "release_build: ./build.sh done ($(ls build/*.hsaco | wc -l) code objects; build/release-build.log)"
"$PY" -I tools/release/stage_kernels.py
STAGED=python/caminus/hsaco/gfx1100
echo "release_build: provenance: $("$PY" -I -c "import json; p = json.load(open('$STAGED/provenance.json')); g = p['ggml_headers']; print(f\"built {p['built']} ({p['built_from']}); ggml headers {g['commit']} {'(dirty)' if g['dirty'] else '(clean)'}\")")"
# the panel (panel/, Go): built reproducibly (no VCS stamp, no build id, trimmed paths, the package's version) and staged as
# package data (pyproject.toml: share/caminus-panel), so `caminus panel` works from the wheel
command -v go > /dev/null || die "go is missing: the panel is part of the release (panel/; Go 1.22 or newer)"
(cd panel && CGO_ENABLED=0 GOFLAGS=-buildvcs=false go build -trimpath -ldflags "-s -w -buildid= -X main.version=$VER" \
    -o ../python/caminus/share/caminus-panel .) > build/release-panel.log 2>&1 || { tail -30 build/release-panel.log >&2; die "go build of the panel failed"; }
echo "release_build: the panel built ($(go version | cut -d' ' -f3); sha256 $(sha256sum python/caminus/share/caminus-panel | cut -c1-16))"
HASHES=packaging/gfx1100-kernels.sha256
staged_list() { "$PY" -I -c "import json; o = json.load(open('$STAGED/provenance.json'))['objects']; print(''.join(f'{o[n][\"sha256\"]}  {n}\n' for n in sorted(o)), end='')"; }
if [ "$mode" = hashes ]; then
    staged_list > "$HASHES"
    echo "release_build: wrote $HASHES ($(wc -l < "$HASHES") objects) from the staged provenance: review and commit it, then tag and run again"
    exit 0
fi
[ -f "$HASHES" ] || die "$HASHES is missing: run tools/release/release_build.sh --write-hashes and commit it first"
if ! diff <(staged_list) "$HASHES" > build/hashes.diff; then
    cat build/hashes.diff >&2
    die "the staged code objects differ from $HASHES (build/hashes.diff): a different toolchain or sources; not released"
fi
(cd "$STAGED" && sha256sum -c --strict --quiet "../../../../$HASHES") || die "sha256sum -c of the staged objects failed"
echo "release_build: the staged code objects == $HASHES ($(wc -l < "$HASHES") objects)"

mkdir dist
"$PY" -I -m pip wheel . --no-deps --no-build-isolation -w dist > build/release-wheel.log 2>&1 || { tail -30 build/release-wheel.log >&2; die "pip wheel failed"; }
"$PY" -I -c "from setuptools import build_meta as b; b.build_sdist('dist')" > build/release-sdist.log 2>&1 || { tail -30 build/release-sdist.log >&2; die "build_sdist failed"; }
"$PY" -I tools/release/repro_dist.py dist || die "tools/release/repro_dist.py refused the archives"
(cd dist && sha256sum "caminus-$VER-py3-none-any.whl" "caminus-$VER.tar.gz" > SHA256SUMS)
[ -z "$(git status --porcelain --untracked-files=no)" ] || die "the build changed tracked files (git status)"
if [ "$mode" = tag ]; then "$PY" -I tools/release/verify_release.py dist --tag "$TAG"; else "$PY" -I tools/release/verify_release.py dist --untagged; fi
echo "release_build: dist/ ready:"; cat dist/SHA256SUMS
