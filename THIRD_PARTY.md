# Third-party code

Caminus is Copyright (C) 2026 jdominick05, licensed under the GNU Affero General Public License v3.0 or later
([LICENSE](LICENSE)). It includes, builds against or derives from the third-party code below: MIT-licensed (llama.cpp / ggml,
Strata, flash-linear-attention) and Apache-2.0-licensed (Hugging Face transformers; the license text is in
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)). Their terms require keeping these notices with the code. Model weights are
not part of this repository and stay under their own licenses (the last section).

When new third-party code comes in, list it here in the same commit and keep its notice in the file header.

## llama.cpp / ggml (MIT)

https://github.com/ggml-org/llama.cpp, used through the ignite-llama fork (https://github.com/jdominick05/ignite-llama at
commit e4e3f8eda: build.sh compiles against its ggml headers; docs/dev/building.md). The development
builds used the fork's commit 842653396, published as e4e3f8eda with identical `ggml/`, `common/` and `src/`; the vendored
files below name 842653396 in their headers.

| Where | How |
|---|---|
| `kernels/k3_gguf.hip` | compiles in ggml-cuda's GGUF block formats and dot products (`vecdotq.cuh` and the headers it includes) |
| `kernels/gg_mmvf_ref.hip` | compiles ggml-cuda's `mmvf.cu` unchanged, as the BF16 matvec reference for tests |
| `kernels/ggml_mmq/` (vendored) | ggml-cuda's MMQ for the batched prefill, copied from the fork at commit 842653396 (published as e4e3f8eda; identical `ggml/`, `common/`, `src/`) by `tools/release/vendor_mmq.py` (`--check --commit e4e3f8eda` compares the files against their headers' recorded label 842653396; each file keeps the MIT notice in its header and lists its changes): `mmq.cuh` (device code verbatim; only the RDNA3 config included, the host launch code removed), `mmq-config-rdna3.cuh`, `mmq-load-tiles.cuh`, `mmq-vec-dot.cuh` (unchanged), `quantize_mmq.cuh` (the `quantize_mmq_q8_1` kernel of `quantize.cu`), `mmid.cuh` (the `mm_ids_helper` kernel of `mmid.cu`). They build against the fork's base headers (`common.cuh`, `mma.cuh`, `vecdotq.cuh`, `ggml-common.h`), as `k3_gguf.hip` does |
| `kernels/prefill_mmq.hip`, `python/caminus/prefill/mmq.py` | instantiate and launch the vendored kernels (the launch logic follows ggml's `mmq.cu` / `mmq.cuh` host code: argument layout, J choice, q8_1 buffer padding, mul_mat_id grouping); the MIT notice is in `mmq.py`'s header |
| `kernels/prefill_moe.hip`, `python/caminus/prefill/moe.py` | the prefill's own MoE GEMMs: call the vendored `load_tiles` / `write_back`; the K loop is a modified copy of `mul_mat_q_process_tile` (mmq.cuh), the tile product `moe_vec_dot` a modified copy of `vec_dot_q8_0_q8_1_mma` / `vec_dot_q8_0_16_q8_1_mma` (mmq-vec-dot.cuh), the fused epilogue repeats `quantize_mmq_q8_1`'s D4 expressions (quantize.cu); the MIT notice is in the kernel file's header and in `moe.py`'s |
| `kernels/prefill_dense.hip`, `python/caminus/prefill/dense.py` | the prefill's own dense Q6_K GEMMs: call the vendored `load_tiles` / `vec_dot` (the control variant) and `write_back`; the K loop is a modified copy of `mul_mat_q` / `mul_mat_q_process_tile` (mmq.cuh), the weight tile `dense_load_x_q6_K` a modified copy of `ggml_cuda_mmq_load_tiles_q6_K` (mmq-load-tiles.cuh), the tile product `dense_vec_dot_q6_K` a modified copy of `ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma` (mmq-vec-dot.cuh); the MIT notice is in the kernel file's header and in `dense.py`'s |
| `tools/reference/gg_golden.cpp`, `tools/reference/gg_mmq_ref.cpp` | link libggml / libllama to dump reference tensors (gg_mmq_ref: ggml_mul_mat / ggml_mul_mat_id outputs and timings) |
| `tools/reference/dflash_dump.cpp` | a modified copy of llama.cpp's `examples/speculative-simple/speculative-simple.cpp` (the speculative loop kept; a per-pass JSON log and a cb_eval tensor dumper for the DFlash drafter added); links libllama; the MIT notice is in the file's header |
| `python/caminus/decode/dflash_torch.py` | a PyTorch port of the DFlash drafter's forward (`src/models/dflash.cpp`) and the draft-dflash loop (`common/speculative.cpp`), used as the reference of the K3 drafter module; the MIT notice is in the file's header |
| `python/caminus/iq_tables.py` | the codebooks of the IQ block formats as data (`kvalues_iq4nl`, `iq2xxs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid` of ggml-common.h, taken through gguf-py's `quants.py`): format constants read by Caminus's own decoders; the notice is in the file |
| `kernels/iq_grids.h` | the same three grids as C arrays for the kernels (`kernels/own_quant_iq.h`), generated from `python/caminus/iq_tables.py`; the notice is in the file |
| `kernels/iq2s_grid.h` | IQ2_S's grid (`iq2s_grid`) packed to 2 bits per magnitude for the kernels (`kernels/own_quant_iq2s.h`), generated from `python/caminus/iq_tables.py`; the notice is in the file |

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## llama.cpp gguf-py (MIT)

https://github.com/ggml-org/llama.cpp, `gguf-py/` (its own LICENSE below).

| Where | How |
|---|---|
| `python/caminus/_vendor/gguf/` (vendored) | gguf-py's GGUF reader at upstream commit 5aeac37a35 (the ignite-llama fork's gguf-py at e4e3f8eda is identical), copied by `tools/release/vendor_gguf.py` (`--check` verifies it): `constants.py`, `gguf_reader.py`, `quants.py`, `lazy.py` unchanged below a header with the MIT notice; `__init__.py` reduced to those modules (upstream's also imports the writer, tensor mapping, vocab, utility and metadata modules). Vendored because the Coder's Q2_0 tensors postdate gguf-py 0.19.0, the last version bump (docs/dev/packaging.md) |

```
MIT License

Copyright (c) 2023 Georgi Gerganov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Strata (MIT)

https://github.com/Niko1221/Strata. Code was rewritten and line-reviewed, never copied as is; nothing from Strata is run.

| Where | What follows Strata |
|---|---|
| `python/caminus/serve/template.py` | the Jinja environment setup (it shares its origin with transformers', below); LiteralMarker's mark / unmark scheme and its kept `<tool_response>` wrapper |
| `python/caminus/serve/normalize.py` | `pin_billing_header`; the OpenAI / Anthropic conversion shape |
| `python/caminus/serve/toolparse.py` | the tool-call parser's scheme (written from scratch) |

```
MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## flash-linear-attention (MIT)

https://github.com/fla-org/flash-linear-attention (the installed `flash-linear-attention` 0.6.0 package's LICENSE below).

| Where | What follows fla |
|---|---|
| `kernels/gdn.hip`, `python/caminus/train/gdn.py` | HIP implementations of fla's Gated DeltaNet chunk state recurrences `chunk_gated_delta_rule_fwd_h` / `_bwd_dhu` (`fla/ops/common/chunk_delta_h.py`): the same algorithm and operand roundings, written for gfx1100 (dormant training code). both files carry the MIT notice in their headers |

```
MIT License

Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Hugging Face transformers (Apache-2.0)

https://github.com/huggingface/transformers. Apache-2.0 is compatible with the AGPLv3; the license text is in
[LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) (copied from the installed transformers 5.20.2 package).

| Where | What follows transformers |
|---|---|
| `python/caminus/serve/template.py` | the chat template's Jinja environment (a sandboxed environment with `trim_blocks` / `lstrip_blocks` / `loopcontrols`), the `tojson` override, `raise_exception` and `strftime_now`, as in `src/transformers/utils/chat_template_utils.py`; modified (`raise_exception` raises the server's own TemplateRequestError); the notice is in the file's header |

```
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Model weights (not in this repository)

Caminus serves model files the user downloads; none is redistributed here, so the repository itself owes no model licence.
Serving and publishing results under a model's name follow each model's own terms:

| Model | Source | Licence | Note |
|---|---|---|---|
| Qwen3.8-Flash-Next (the base) | `Qwen/Qwen3.8-Flash-Next` | `other`, `license_name: qwen-community-1.0` (its model card) | **Read qwen-community-1.0's terms** (attribution, use limits) before you serve the model or publish results under its name. Its text was not reviewed when this table was written (no network access then) |
| GSQ-RCO Coder, IQ1_M pack (served) | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` | **the card contradicts itself**: its YAML tag says `apache-2.0`, its License section says the weights inherit the base model's licence; the GGUF carries no `general.license` | the conservative reading: the base's licence (qwen-community-1.0) |
| DFlash2 drafter (Q8_0, served) | `incoai/Qwen3.8-Flash-Next-DFlash2` (Inco AI) | `apache-2.0` (the GGUF's metadata) | the manifest's Q8_0 file (python/caminus/models.py): `incoai/Qwen3.8-Flash-Next-DFlash2-GGUF` serves it at the pinned revision with the manifest's size and LFS sha256 (Hugging Face's API, 2026-11-06); `caminus download` refuses a mismatch |
