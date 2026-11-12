#!/usr/bin/env bash
# Build the release build image (packaging/build-image/Dockerfile; docs/dev/releasing.md, "The build image") from the
# maintainer's TheRock tree: only the files in packaging/build-image/therock-files.txt go in (the ones the kernel build
# opens, about 250 MB of the 11 GB tree, and TheRock's notices). The image is public; push it and write its digest into
# packaging/build-image/IMAGE.
#
#   ROCM_PATH=<TheRock 10.2.0a20261011> tools/release/build_image.sh [TAG]     TAG default: the name below
#
# Docker (rootless is fine). Network: the base image, Go, the pinned pip wheels and the ignite-llama commit. CPU only.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
TAG=${1:-ghcr.io/jdominick05/caminus-build:therock-10.2.0a20261011}
TR=${ROCM_PATH:?set ROCM_PATH to the TheRock 10.2.0a20261011 tree}
[ "$(basename "$TR")" = 10.2.0a20261011 ] || { echo "build_image: ROCM_PATH must be the 10.2.0a20261011 tree, not $TR" >&2; exit 2; }
mkdir -p build
ctx=$(mktemp -d "$PWD/build/build-image.XXXXXX")
trap 'rm -rf "$ctx"' EXIT
cp -r packaging/build-image/Dockerfile packaging/build-image/requirements.txt packaging/build-image/licenses "$ctx/"
grep -v '^#' packaging/build-image/therock-files.txt | (cd "$TR" && tar --create --file "$ctx/therock.tar" --no-recursion \
    --sort=name --owner=0 --group=0 --numeric-owner --mtime=@0 --verbatim-files-from --files-from=-)
echo "build_image: therock.tar $(du -h "$ctx/therock.tar" | cut -f1) ($(grep -vc '^#' packaging/build-image/therock-files.txt) entries)"
docker build --pull -t "$TAG" "$ctx"
echo "build_image: built $TAG ($(docker image inspect --format '{{.Id}}' "$TAG"))"
