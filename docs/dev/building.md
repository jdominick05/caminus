# Building from source

By the end of this page you'll have a checkout that builds the GPU code, runs the tests, and can make a wheel. Users don't
need this: the package ships the built GPU code.

## What you need

- Everything from the [quickstart](../quickstart.md): Linux, Python 3.12, PyTorch's ROCm build.
- **TheRock ROCm 10.2** (tested: the nightly 10.2.0a20261011) for `amdclang++` and device LTO. Point `ROCM_PATH` at it.
  Another compiler gives different machine code, and the bitwise tests compare exact bytes.
- **The ignite-llama fork of llama.cpp at commit e4e3f8eda**
  ([github.com/jdominick05/ignite-llama](https://github.com/jdominick05/ignite-llama)), in `IGNITE`. `build.sh` compiles
  against its ggml headers; the vendored quantized GEMMs in `kernels/ggml_mmq` come from it; and its HIP build is the
  reference the accuracy gates compare against. (Older notes name its development commit 842653396: the same `ggml/`.)
- `patchelf`, for the C++ runtime tests.
- Go 1.26 or newer, for the panel.

```bash
git clone https://github.com/jdominick05/ignite-llama && git -C ignite-llama checkout e4e3f8eda
git clone https://github.com/jdominick05/caminus && cd caminus
```

## Build

```bash
export ROCM_PATH=/path/to/therock IGNITE=/path/to/ignite-llama PATCHELF=$(command -v patchelf)
export TORCHLIB=$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
./build.sh                                   # every kernels/*.hip -> build/*.hsaco, then the C++ runtime tests
(cd panel && go build -o caminus-panel .)    # the panel
```

A checkout runs without installing: `python -I tools/caminus.py COMMAND` is the `caminus` command, and the package
imported from `python/` finds `build/` and `panel/caminus-panel` by itself.

## A wheel

```bash
python -I tools/release/stage_kernels.py     # the runtime GPU code + provenance.json into python/caminus/hsaco/gfx1100
python -I -m pip wheel . --no-deps --no-build-isolation -w dist
```

Release builds are reproducible: [releasing](releasing.md) has the recipe and the checks.

## Next

- [Testing](testing.md): what to run before you send a change.
- [GPU safety](gpu-safety.md): the rules for running GPU code on a machine whose card drives your monitors.
