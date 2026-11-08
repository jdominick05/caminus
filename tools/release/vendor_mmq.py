"""Vendor ggml-cuda's MMQ (llama.cpp, MIT) into kernels/ggml_mmq/ for the batched prefill (docs/specs/spec-prefill-cache.md
S2): the quantized GEMM kernels (mmq.cuh, its RDNA3 tile config, the per-type tile loaders and dot products), the q8_1
activation quantizer of mmq (quantize_mmq_q8_1) and the expert-grouping helper of mul_mat_id (mm_ids_helper).

  python -I tools/release/vendor_mmq.py --check [--commit e4e3f8eda] [--label 842653396] [--src DIR]   # exit 1 if the files differ
  python -I tools/release/vendor_mmq.py [--commit C] [--label L] [--src DIR]                           # (re)write kernels/ggml_mmq/

The files are taken from the fork's git history at --commit (`git show`), not its working tree, so a vendored copy is
reproducible. --src: an ignite-llama git tree holding --commit (default $IGNITE, else ~/src/ignite-llama). --commit: default
e4e3f8eda, the public fork's commit (github.com/jdominick05/ignite-llama). --label: the commit the generated headers name,
default 842653396, the one kernels/ggml_mmq records: the development commit, published as e4e3f8eda with identical ggml/,
common/ and src/. The vendored files are compiled into the shipped code objects, so they keep that label; --check reads the
files at --commit and compares them with headers that carry --label (only the commit token in each header takes it), which
passes exactly when the fork's files are the same at both commits.

Device code is kept verbatim; the only changes, all in mmq.cuh, are listed in EDITS (the host launch code is replaced by
python/caminus/prefill/mmq.py, and only the RDNA3 config is compiled in). kernels/prefill_mmq.hip instantiates the kernels; build.sh
compiles it against the same fork's base headers (common.cuh, mma.cuh, vecdotq.cuh, ggml-common.h), as the decode modules do.
"""
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, 'kernels', 'ggml_mmq')
CUDA = 'ggml/src/ggml-cuda'
COMMIT = 'e4e3f8eda'      # read from: the public fork commit (github.com/jdominick05/ignite-llama)
LABEL = '842653396'       # named in the headers: the development commit, published as e4e3f8eda (identical ggml/)

MIT = """MIT License

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
SOFTWARE."""

CONFIG_INCLUDES = """#include "mmq-config-pascal-older.cuh"
#include "mmq-config-pascal-dp4a.cuh"
#include "mmq-config-ampere.cuh"
#include "mmq-config-blackwell.cuh"

#include "mmq-config-gcn.cuh"
#include "mmq-config-cdna.cuh"
#include "mmq-config-rdna2.cuh"
#include "mmq-config-rdna3.cuh"
#include "mmq-config-rdna3-5.cuh"
#include "mmq-config-rdna4.cuh"
"""

# mmq.cuh: (what, old, new); every old must occur exactly once
EDITS = [
    ('gfx1100 only: refuse any other device target (only the RDNA3 config is vendored)',
     '#include "common.cuh"\n',
     '#include "common.cuh"\n\n#if defined(__HIP_DEVICE_COMPILE__) && !defined(RDNA3_0)\n'
     '#error "caminus: the vendored MMQ carries only the RDNA3 (gfx1100) config"\n#endif\n'),
    ('only the RDNA3 tile config is included (the other architectures\' config headers are not vendored)',
     CONFIG_INCLUDES, '#include "mmq-config-rdna3.cuh"\n'),
]
# the two config dispatchers (host, device): bodies replaced by the RDNA3 config (regex on the whole function)
DISPATCH = [
    ('host config dispatcher: the RDNA3 config (the host launch is python/caminus/mmq.py, which reads the configs from '
     'the device: kernels/prefill_mmq.hip k3_mmq_configs)',
     r'static __host__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config\(const ggml_type type, const int J, const bool fallback, const int cc, const ggml_prec prec_src1\) \{\n.*?\n\}\n',
     'static __host__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(const ggml_type type, const int J, const bool fallback, const int cc, const ggml_prec prec_src1) {\n'
     '    GGML_UNUSED_VARS(cc, prec_src1);\n    return ggml_cuda_mmq_get_config_rdna3(type, J, fallback); // [caminus] RDNA3 only\n}\n'),
    ('device config dispatcher: the RDNA3 config (gfx1100: the branch upstream takes for RDNA3 without RDNA3_5)',
     r'static constexpr __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config\(ggml_type type, int J, bool fallback, ggml_prec prec_src1\) \{\n.*?\n\}\n',
     'static constexpr __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(ggml_type type, int J, bool fallback, ggml_prec prec_src1) {\n'
     '    GGML_UNUSED(prec_src1);\n    return ggml_cuda_mmq_get_config_rdna3(type, J, fallback); // [caminus] RDNA3 only\n}\n'),
]
HOST_TAIL = 'struct mmq_args {'   # from here to the end: ggml's host launch code (ggml_backend_cuda_context, pools), dropped


def git_show(src, commit, path):
    return subprocess.run(['git', '-C', src, 'show', f'{commit}:{path}'], check=True, capture_output=True, text=True).stdout


# The banner names this tool by the path it had when the headers were vendored (tools/vendor_mmq.py; it moved to
# tools/release/ in the layout restructure). The headers are part of the kernel tree whose sha256 the code objects' provenance
# records, so the banner keeps that text until the headers are vendored again on purpose (with a rebuild); so do the change
# notes below that name the host launcher at its old path (python/caminus/mmq.py, now python/caminus/prefill/mmq.py).
GENERATOR = 'tools/vendor_mmq.py'


def header(name, origin, label, changes):
    lines = [f'// Vendored from llama.cpp / ggml (MIT): {origin}', f'// at the ignite-llama fork commit {label} (~/src/ignite-llama; '
             'upstream https://github.com/ggml-org/llama.cpp).', f'// Generated by {GENERATOR} (do not edit by hand; '
             'rerun it). Listed in THIRD_PARTY.md.']
    if changes:
        lines.append('// Changes from the original (device code otherwise verbatim):')
        lines += [f'//   - {c}' for c in changes]
    else:
        lines.append('// Unchanged from the original.')
    lines.append('//')
    lines += [('// ' + l).rstrip() for l in MIT.split('\n')]
    return '\n'.join(lines) + '\n\n'


def extract(text, start, end, what):
    i = text.find(start); assert i >= 0 and text.find(start, i + 1) < 0, f'{what}: start marker not found once'
    j = text.find(end, i); assert j > i, f'{what}: end marker not found'
    return text[i:j].rstrip() + '\n'


def build(src, commit, label):
    """the vendored files: their sources read at `commit`, their headers naming `label`"""
    out = {}
    # mmq.cuh
    t = git_show(src, commit, f'{CUDA}/mmq.cuh'); changes = []
    for what, old, new in EDITS:
        assert t.count(old) == 1, f'mmq.cuh: "{what}": the original text occurs {t.count(old)} times'
        t = t.replace(old, new); changes.append(what)
    for what, pat, new in DISPATCH:
        t, n = re.subn(pat, lambda m: new, t, count=1, flags=re.S); assert n == 1, f'mmq.cuh: "{what}": not found'
        changes.append(what)
    k = t.find(HOST_TAIL); assert k > 0 and t.find(HOST_TAIL, k + 1) < 0, 'mmq.cuh: host launch code not found once'
    t = t[:k].rstrip() + '\n'
    changes.append('the host launch code (struct mmq_args .. end: launch_mul_mat_q, mul_mat_q_switch_J, mul_mat_q_case, the '
                   'DECL_MMQ_CASE externs, ggml_cuda_mul_mat_q / ggml_cuda_should_use_mmq declarations) removed: '
                   'python/caminus/mmq.py launches the kernels')
    out['mmq.cuh'] = header('mmq.cuh', f'{CUDA}/mmq.cuh', label, changes) + t
    for f in ('mmq-config-rdna3.cuh', 'mmq-load-tiles.cuh', 'mmq-vec-dot.cuh'):
        out[f] = header(f, f'{CUDA}/{f}', label, []) + git_show(src, commit, f'{CUDA}/{f}')
    # quantize_mmq.cuh: the mmq activation quantizer (quantize.cu) and its block-size constant (quantize.cuh)
    q = git_show(src, commit, f'{CUDA}/quantize.cu'); qh = git_show(src, commit, f'{CUDA}/quantize.cuh')
    body = extract(q, '// scatter: grid over tokens, quantize once, write to all the token\'s compact rows\ntemplate <mmq_q8_1_ds_layout ds_layout, bool scatter>',
                   '\nvoid quantize_row_q8_1_cuda(', 'quantize_mmq_q8_1')
    cst = extract(qh, '#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128', '\n\n', 'CUDA_QUANTIZE_BLOCK_SIZE_MMQ')
    sa = extract(qh, 'static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0', '\n', 'static_assert')
    out['quantize_mmq.cuh'] = (header('quantize_mmq.cuh', f'{CUDA}/quantize.cu (the quantize_mmq_q8_1 kernel) and quantize.cuh '
                                      '(CUDA_QUANTIZE_BLOCK_SIZE_MMQ)', label,
                                      ['only the quantize_mmq_q8_1 kernel and the constant it is launched with are taken (the '
                                       'other quantizers and the host launchers are not)']) +
                               '#pragma once\n\n#include "common.cuh"\n#include "mmq.cuh"\n\n#include <cstdint>\n\n' + cst + '\n' + sa + '\n' + body)
    m = git_show(src, commit, f'{CUDA}/mmid.cu')
    mb = extract(m, '// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.',
                 '\ntemplate <int n_expert_used_template>\nstatic void launch_mm_ids_helper(', 'mm_ids_helper')
    out['mmid.cuh'] = (header('mmid.cuh', f'{CUDA}/mmid.cu (the mm_ids_helper kernel)', label,
                              ['only the mm_ids_helper kernel and its helpers are taken (the host launchers are not)']) +
                       '#pragma once\n\n#include "common.cuh"\n\n' + mb)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--commit', default=COMMIT, help=f'the fork commit the files are read from (default {COMMIT})')
    ap.add_argument('--label', default=LABEL, help=f'the commit the generated headers name (default {LABEL}, what kernels/ggml_mmq records)')
    ap.add_argument('--src', default=os.environ.get('IGNITE') or os.path.expanduser('~/src/ignite-llama'),
                    help='an ignite-llama git tree holding --commit (default $IGNITE, else ~/src/ignite-llama)')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    files = build(a.src, a.commit, a.label)
    if a.check:
        bad = [f for f, t in files.items() if not os.path.exists(os.path.join(OUT, f)) or open(os.path.join(OUT, f)).read() != t]
        print('vendor_mmq: ' + (f'OK, kernels/ggml_mmq matches {a.commit} (headers labelled {a.label})' if not bad
                                else f'DIFFERS from {a.commit} (headers labelled {a.label}): {bad}'))
        sys.exit(1 if bad else 0)
    os.makedirs(OUT, exist_ok=True)
    for f, t in files.items():
        with open(os.path.join(OUT, f), 'w') as fh: fh.write(t)
        print(os.path.join('kernels', 'ggml_mmq', f), len(t))


if __name__ == '__main__':
    main()
