# Licences of the build image's third-party files

The release build image (packaging/build-image/Dockerfile) is public and redistributes a trimmed TheRock ROCm
10.2.0a20261011 (therock-files.txt). TheRock's own notices travel with it, inside the image at
/opt/rocm-therock/10.2.0a20261011/share/doc/: NOTICES.txt and the LICENSE files of hip, hipblas, hipblas-common and
hipcc. This directory adds the licences of the components among those files whose text TheRock does not ship; the
image copies it to /usr/share/doc/caminus-build-image/.

| File | Component in the image | Licence | Source |
|---|---|---|---|
| LLVM-LICENSE.TXT | lib/llvm (clang, lld, libLLVM, libclang-cpp, the device-library bitcode, clang's headers) | Apache-2.0 WITH LLVM-exception | github.com/ROCm/llvm-project, llvm/LICENSE.TXT |
| zstd-LICENSE | lib/rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1.5.7 | BSD-3-Clause (zstd is dual BSD / GPL-2.0; taken under BSD) | github.com/facebook/zstd v1.5.7, LICENSE |
| zlib-LICENSE | lib/rocm_sysdeps/lib/librocm_sysdeps_z.so.1.3.2 | Zlib | github.com/madler/zlib v1.3.2, LICENSE |

HIP (include/hip, lib/libamdhip64.so) and the hipBLAS headers are MIT (their LICENSE.md files above). Ubuntu, Go, pip /
setuptools / wheel and the ignite-llama fork (MIT) bring their own licence files in the image.
