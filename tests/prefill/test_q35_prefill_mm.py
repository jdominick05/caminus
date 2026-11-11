"""The dense qwen35 prefill's GEMMs (kernels/q35_prefill.hip) on a card against float64, on real rows of a model file: Q4_K
and Q6_K (q3m_prefill.hip's kernels, included as they are) and the two formats this object adds, Q8_0 and Q5_K.

  tools/guard.sh LOG 900 env HIP_VISIBLE_DEVICES=<a non-display card> python -I tests/prefill/test_q35_prefill_mm.py GGUF

For each format: the first rows of a tensor of that format (R = 200 and 256: a partial and a whole 128-row tile), fp16
activations of 1, 37, 64 and 130 tokens (partial and whole 64-token tiles), Y = X W^T in fp32 against X W^T in float64
from the host decoder (caminus.own_dequant). The error of each output is measured against sum |x| |w| of that output
(the scale of the fp32 rounding); the test asserts the worst at most 1e-5 of it, prints the mean and the worst per
format, and checks the Q4_K / Q5_K overflow branch: one token whose 32-value sums pass 65,504 (fp16's range).
Every tensor's bytes are read into ordinary memory (Tensor.pread); every allocation is checked against the card's cap.
"""
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch  # noqa: E402

from caminus import ggufio, own_dequant  # noqa: E402
from caminus.device import capacity, hip  # noqa: E402
from caminus.engine.own_layout import q6k_planes  # noqa: E402

FAILS = []
NULL = ctypes.c_void_p(0)
I = ctypes.c_int
BOUND = 1e-5   # (fp32 accumulation over K up to 17,408; the accepted Q4_K / Q6_K kernels are measured beside the new ones)


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        FAILS.append(what)


def main():
    path = sys.argv[1]
    F = ggufio.GGUFFile(path)
    dev = torch.device('cuda', 0)
    print(f'NOTE device: {torch.cuda.get_device_name(dev)} (HIP_VISIBLE_DEVICES={os.environ.get("HIP_VISIBLE_DEVICES")})', flush=True)
    picks = {}
    for name, t in F.T.items():   # (the first trunk tensor of each format with a K the kernels take)
        if t.type in ('Q4_K', 'Q6_K', 'Q8_0', 'Q5_K') and len(t.shape) == 2 and t.shape[0] % 256 == 0 and t.shape[1] >= 256 \
                and t.type not in picks and 'token_embd' not in name and 'output' != name.split('.')[0]:
            picks[t.type] = name
    print('NOTE tensors: ' + ', '.join(f'{k} {v} {tuple(F.T[v].shape)}' for k, v in sorted(picks.items())), flush=True)
    Kmax = max(F.T[n].shape[0] for n in picks.values())
    capacity.check([dev.index], [(130 * Kmax * 2 + 130 * 256 * 4 * 2 + 256 * Kmax * 2) / capacity.MIB * 1.5], 'q35 prefill GEMM test')
    mod = hip.Module('q35_prefill')
    rng = np.random.default_rng(11)
    for fmt in ('Q4_K', 'Q6_K', 'Q8_0', 'Q5_K'):
        if fmt not in picks:
            print(f'NOTE {fmt}: no tensor of this format in the file', flush=True)
            continue
        t = F.T[picks[fmt]]
        K = int(t.shape[0]); _, rb = t.rows()
        for R in (200, 256):
            raw = t.pread(0, R * rb)
            Wf = own_dequant.dequant(raw, fmt, R * K).reshape(R, K).astype(np.float64)
            wb = q6k_planes(raw, R, K) if fmt == 'Q6_K' else raw
            wd = torch.from_numpy(np.ascontiguousarray(wb)).to(dev)
            for n in (1, 37, 64, 130):
                x = (rng.standard_normal((n, K)) * 0.5).astype(np.float16)
                big = fmt in ('Q4_K', 'Q5_K') and n == 37
                if big:   # (token 5: block 0's first 32 values at 4,000: their sum 128,000 passes fp16's 65,504)
                    x[5, :32] = 4000.0
                xd = torch.from_numpy(x).to(dev)
                y = torch.zeros(n, R, dtype=torch.float32, device=dev)
                grid = (-(-n // 64), -(-R // 128))
                if fmt == 'Q4_K':
                    mod.q3p_mm_q4k(grid, (256,), wd, I(R), I(K), xd, I(K), NULL, y, I(R), NULL, NULL, NULL, I(n))
                elif fmt == 'Q6_K':
                    mod.q3p_mm_q6k(grid, (256,), wd, I(R), I(R), I(K), xd, I(K), NULL, y, I(R), NULL, NULL, NULL, I(n))
                elif fmt == 'Q8_0':
                    mod.q35p_mm_q8_0(grid, (256,), wd, I(R), I(K), xd, I(K), NULL, y, I(R), NULL, NULL, NULL, I(n))
                else:
                    mod.q35p_mm_q5k(grid, (256,), wd, I(R), I(K), xd, I(K), NULL, y, I(R), NULL, NULL, NULL, I(n))
                torch.cuda.synchronize()
                yg = y.cpu().numpy().astype(np.float64)
                x64 = x.astype(np.float64)
                ref = x64 @ Wf.T
                scale = np.abs(x64) @ np.abs(Wf).T
                rel = np.abs(yg - ref) / np.maximum(scale, 1e-30)
                fin = bool(np.isfinite(yg).all())
                check(fin and rel.max() <= BOUND,
                      f'{fmt} R={R} n={n} K={K}{" (an overflowing token)" if big else ""}: error / sum|x||w| mean {rel.mean():.2e}, '
                      f'worst {rel.max():.2e} (bound {BOUND:.0e}), finite {fin}')
            del wd
    print('RESULT', 'FAIL' if FAILS else 'PASS')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
