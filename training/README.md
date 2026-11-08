# training/

The dormant S1 training stack's scripts (dormant since the 10-31 inference-first pivot): `tools/` (patches, profiling,
decision tests and scoring for the private trainer) and `runs/` (run scripts and their logs). Its Python modules are the
package's `caminus.train`, its tests `tests/train/`, and its C++ runtime the top-level `runtime/`, which stays at the root
because `build.sh` compiles it from there and build.sh is part of the kernel tree's provenance hash. Most of this needs the
private research repository, so the public tree leaves those files out (packaging/public-manifest.txt).
