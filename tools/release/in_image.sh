#!/usr/bin/env bash
# Run a command from this checkout inside the release build image (packaging/build-image/; docs/dev/releasing.md, "The
# build image"): the checkout is mounted at /src, the image sets PY, ROCM_PATH, IGNITE and Go. The maintainer's builds,
# the nightly timer (tools/release/nightly.sh) and CI (.github/workflows/build.yml) all build through this script, so they
# give the same bytes.
#
#   tools/release/in_image.sh tools/release/release_build.sh [--untagged | --write-hashes]
#
# The image: $CAMINUS_BUILD_IMAGE, else packaging/build-image/IMAGE (the pushed image, by digest). Docker, rootless or
# not: under a rootful daemon the command runs as the caller's uid, so the build output is the caller's. SOURCE_DATE_EPOCH
# is passed through when set. CPU only: the container gets no GPU device.
set -euo pipefail
[ $# -gt 0 ] || { echo "usage: tools/release/in_image.sh COMMAND [ARGS...]" >&2; exit 2; }
IMAGE=${CAMINUS_BUILD_IMAGE:-$(sed -n '/^[^#]/{p;q}' "$(dirname "$(readlink -f "$0")")/../../packaging/build-image/IMAGE")}
[ -n "$IMAGE" ] || { echo "in_image: no image (packaging/build-image/IMAGE is empty)" >&2; exit 2; }
user=()
docker info --format '{{.SecurityOptions}}' | grep -q rootless || user=(--user "$(id -u):$(id -g)")
exec docker run --rm "${user[@]}" -e SOURCE_DATE_EPOCH -v "$PWD:/src" -w /src "$IMAGE" "$@"
