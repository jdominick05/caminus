# The release build image (docs/dev/releasing.md, "The build image"): everything tools/release/release_build.sh needs, at
# fixed paths, so the maintainer's machine, the nightly timer and CI (.github/workflows/build.yml) build the same bytes.
#   /opt/rocm-therock/10.2.0a20261011   TheRock ROCm, only the files the kernel build opens (therock-files.txt)
#   /opt/ignite-llama                   the ignite-llama fork at e4e3f8eda, ggml/ only (sparse; provenance.json records the commit)
#   /opt/py                             the distribution's Python 3 with pip / setuptools / wheel at the versions below (hash-checked)
#   /usr/local/go                       Go 1.26.8 (sha256-checked); GOTOOLCHAIN=local: never a downloaded toolchain (go.mod's
#                                       toolchain line names the same version for builds outside the image)
# The kernels' bytes depend on the ROCM_PATH and IGNITE path strings (clang's __hip_cuid_<hash>), so those two paths are
# part of the build: packaging/gfx1100-kernels.sha256 is made in this image. Build it with tools/release/build_image.sh
# (the toolchain comes from the build context, never from the network). The image is public, so anyone can rebuild a
# release with it; CI pulls it by digest. Third-party licences: licenses/README.md.
# Ubuntu 26.04: the device code includes the host's C and C++ headers (<cmath>, <cstdlib> through clang's HIP wrappers), so
# libstdc++-15-dev, libc6-dev and linux-libc-dev are pinned to the versions the shipped kernels were validated with
FROM ubuntu@sha256:065aa8626841f16057b78b027b464cd59c4fac36c204c72dcbf48d66f5fa8a96

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl git python3 python3-venv xz-utils \
      libc6-dev=2.43-2ubuntu2.4 libstdc++-15-dev=15.2.0-16ubuntu1 libgcc-15-dev=15.2.0-16ubuntu1 linux-libc-dev=7.0.0-43.43 \
 && rm -rf /var/lib/apt/lists/*

ARG GO_VERSION=1.26.8
ARG GO_SHA256=d0f743b33e8d8945e6b1f432edd15785c70507121d6e2a723b21285eddf8b57b
RUN curl -fsSL -o /tmp/go.tgz "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" \
 && echo "${GO_SHA256}  /tmp/go.tgz" | sha256sum -c --strict \
 && tar -C /usr/local -xzf /tmp/go.tgz && rm /tmp/go.tgz

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/py \
 && /opt/py/bin/python -m pip install --no-cache-dir --no-deps --require-hashes -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

ARG IGNITE_COMMIT=e4e3f8edac36544e7c06139eb03e35288797e73e
RUN git init -q /opt/ignite-llama \
 && git -C /opt/ignite-llama fetch -q --depth 1 https://github.com/jdominick05/ignite-llama "${IGNITE_COMMIT}" \
 && git -C /opt/ignite-llama sparse-checkout set ggml \
 && git -C /opt/ignite-llama -c advice.detachedHead=false checkout -q FETCH_HEAD \
 && test "$(git -C /opt/ignite-llama rev-parse HEAD)" = "${IGNITE_COMMIT}"

# links the GHCR package to the repository
LABEL org.opencontainers.image.source=https://github.com/jdominick05/caminus

# therock.tar: tools/release/build_image.sh packs therock-files.txt from the maintainer's TheRock tree
ADD therock.tar /opt/rocm-therock/10.2.0a20261011/
COPY licenses/ /usr/share/doc/caminus-build-image/

# the checkout is bind-mounted and owned by another uid (CI's runner, a rootless host user): git must still read it
RUN git config --system --add safe.directory '*'

ENV PATH=/opt/py/bin:/usr/local/go/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PY=/opt/py/bin/python \
    ROCM_PATH=/opt/rocm-therock/10.2.0a20261011 \
    IGNITE=/opt/ignite-llama \
    GOTOOLCHAIN=local \
    GOFLAGS=-mod=readonly \
    GOCACHE=/tmp/go-cache \
    GOPATH=/tmp/go \
    HOME=/tmp \
    LC_ALL=C.UTF-8
