"""The dense qwen35 prefill's GEMMs (kernels/q35_prefill.hip) on a card against float64, on real rows of a model file: Q4_K
and Q6_K (q3m_prefill.hip's kernels, included as they are) and the two formats this object adds, Q8_0 and Q5_K. A file
with codebook formats (ThinkingCap) runs on kernels/q35_prefill_tcap.hip, which adds IQ4_XS / IQ3_XXS / IQ3_S (the IQ
study's kernels) and IQ2_S, the grid formats read in the record layout.

  tools/guard.sh LOG 900 env HIP_VISIBLE_DEVICES=<a non-display card> python -I tests/prefill/test_q35_prefill_mm.py GGUF

For each format: the first rows of a tensor of that format (R = 200 and 256: a partial and a whole 128-row tile), fp16
activations of 1, 37, 64 and 130 tokens (partial and whole 64-token tiles), Y = X W^T in fp32 against X W^T in float64
from the host decoder (caminus.own_dequant). The error of each output is measured against sum |x| |w| of that output
(the scale of the fp32 rounding); the test asserts the worst at most 1e-5 of it, prints the mean and the worst per
format, and checks the Q4_K / Q5_K overflow branch: one token whose 32-value sums pass 65,504 (fp16's range).
Every tensor's bytes are read into ordinary memory (Tensor.pread); every allocation is checked against the card's cap.
--ref-module NAME: also run every launch on another build of the same object (e.g. the previous commit's q35_prefill.hsaco
copied into the kernel dir as q35_prefill_base.hsaco) and require the outputs BITWISE equal: a change that keeps every
sum's arithmetic (prefetching, staging, tiling) must keep every output's bits.
  ... test_q35_prefill_mm.py GGUF [--ref-module q35_prefill_base]
"""
import ctypes
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch  # noqa: E402

from caminus import ggufio, own_dequant  # noqa: E402
from caminus.device import capacity, hip  # noqa: E402
from caminus.engine.own_layout import q6k_planes, iq_records, IQ_REC  # noqa: E402

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
    refname = sys.argv[sys.argv.index('--ref-module') + 1] if '--ref-module' in sys.argv else None
    F = ggufio.GGUFFile(path)
    dev = torch.device('cuda', 0)
    print(f'NOTE device: {torch.cuda.get_device_name(dev)} (HIP_VISIBLE_DEVICES={os.environ.get("HIP_VISIBLE_DEVICES")})', flush=True)
    picks = {}
    FMTS = ('Q4_K', 'Q6_K', 'Q8_0', 'Q5_K', 'IQ4_XS', 'IQ3_XXS', 'IQ3_S', 'IQ2_S')
    IQK = {'IQ4_XS': 'q3p_mm_iq4xs', 'IQ3_XXS': 'q3p_mm_iq3xxs', 'IQ3_S': 'q3p_mm_iq3s', 'IQ2_S': 'q35t_mm_iq2s'}
    for name, t in F.T.items():   # (the first trunk tensor of each format with a K the kernels take)
        if t.type in FMTS and len(t.shape) == 2 and t.shape[0] % 256 == 0 and t.shape[1] >= 256 \
                and t.type not in picks and 'token_embd' not in name and 'output' != name.split('.')[0]:
            picks[t.type] = name
    print('NOTE tensors: ' + ', '.join(f'{k} {v} {tuple(F.T[v].shape)}' for k, v in sorted(picks.items())), flush=True)
    Kmax = max(F.T[n].shape[0] for n in picks.values())
    capacity.check([dev.index], [(130 * Kmax * 2 + 130 * 256 * 4 * 2 + 256 * Kmax * 2) / capacity.MIB * 1.5], 'q35 prefill GEMM test')
    tc = any(f in picks for f in IQK)
    mod = hip.Module('q35_prefill_tcap' if tc else 'q35_prefill')
    print(f'NOTE object: {"q35_prefill_tcap" if tc else "q35_prefill"}', flush=True)
    refmod = hip.Module(refname) if refname else None
    if refmod: print(f'NOTE every launch also on {refname}: outputs bitwise', flush=True)
    nbits = [0, 0]
    rng = np.random.default_rng(11)
    for fmt in FMTS:
        if fmt not in picks:
            print(f'NOTE {fmt}: no tensor of this format in the file', flush=True)
            continue
        t = F.T[picks[fmt]]
        K = int(t.shape[0]); _, rb = t.rows()
        for R in ((200, 256, 300, 512) if fmt in ('Q4_K', 'Q6_K', 'Q8_0') else (200, 256)):
            if R > t.rows()[0]:
                print(f'NOTE {fmt}: R={R} skipped ({picks[fmt]} has {t.rows()[0]} rows)', flush=True)
                continue
            raw = t.pread(0, R * rb)
            Wf = own_dequant.dequant(raw, fmt, R * K).reshape(R, K).astype(np.float64)
            wb = q6k_planes(raw, R, K) if fmt == 'Q6_K' else iq_records(raw, R, K, fmt) if fmt in IQ_REC else raw
            wd = torch.from_numpy(np.ascontiguousarray(wb)).to(dev)
            # (R 300 / 512: the wide tile's partial and full 256-row workgroups, drawn from their own generator so the
            # other cases keep their inputs; at K 17,408 the Q5_K overflow case is close to BOUND, worst 1.05e-05 on one
            # other draw)
            g = rng if R in (200, 256) else np.random.default_rng(1000 + R)
            for n in (1, 37, 64, 130):
                x = (g.standard_normal((n, K)) * 0.5).astype(np.float16)
                big = fmt in ('Q4_K', 'Q5_K') and n == 37
                if big:   # (token 5: block 0's first 32 values at 4,000: their sum 128,000 passes fp16's 65,504)
                    x[5, :32] = 4000.0
                xd = torch.from_numpy(x).to(dev)
                grid = (-(-n // 64), -(-R // 128))
                y = launch(mod, fmt, IQK, grid, wd, R, K, xd, n, dev)
                if fmt == 'Q8_0':   # the wide tile (q35p_mm_q8_0_w) against q35p_mm_q8_0: bitwise
                    yw = torch.zeros(n, R, dtype=torch.float32, device=dev)
                    mod.q35p_mm_q8_0_w((-(-n // 64), -(-R // 256)), (256,), wd, I(R), I(K), xd, I(K), yw, I(R), I(n))
                    torch.cuda.synchronize()
                    wide_ok = torch.equal(y.view(torch.int32), yw.view(torch.int32))
                    check(wide_ok, f'Q8_0 R={R} n={n}: the wide tile q35p_mm_q8_0_w bitwise q35p_mm_q8_0'
                                   + ('' if wide_ok else f' ({int((y.view(torch.int32) != yw.view(torch.int32)).sum())} values differ)'))
                if fmt == 'Q6_K':   # the wide tile (q35p_mm_q6k_w) against q3p_mm_q6k: bitwise
                    yw = torch.zeros(n, R, dtype=torch.float32, device=dev)
                    mod.q35p_mm_q6k_w((-(-n // 64), -(-R // 256)), (256,), wd, I(R), I(R), I(K), xd, I(K), yw, I(R), I(n))
                    torch.cuda.synchronize()
                    wide_ok = torch.equal(y.view(torch.int32), yw.view(torch.int32))
                    check(wide_ok, f'Q6_K R={R} n={n}: the wide tile q35p_mm_q6k_w bitwise q3p_mm_q6k'
                                   + ('' if wide_ok else f' ({int((y.view(torch.int32) != yw.view(torch.int32)).sum())} values differ)'))
                if fmt == 'Q4_K':   # the wide tile (q35p_mm_q4k_w, what Ops.mm runs for dense Q4_K) against q3p_mm_q4k: bitwise
                    yw = torch.zeros(n, R, dtype=torch.float32, device=dev)
                    mod.q35p_mm_q4k_w((-(-n // 64), -(-R // 256)), (256,), wd, I(R), I(K), xd, I(K), yw, I(R), I(n))
                    torch.cuda.synchronize()
                    wide_ok = torch.equal(y.view(torch.int32), yw.view(torch.int32))
                    check(wide_ok, f'Q4_K R={R} n={n}: the wide tile q35p_mm_q4k_w bitwise q3p_mm_q4k'
                                   + ('' if wide_ok else f' ({int((y.view(torch.int32) != yw.view(torch.int32)).sum())} values differ)'))
                if refmod is not None:
                    yr = launch(refmod, fmt, IQK, grid, wd, R, K, xd, n, dev)
                    torch.cuda.synchronize()
                    same = torch.equal(y.view(torch.int32), yr.view(torch.int32))
                    nbits[0] += 1; nbits[1] += int(same)
                    if not same: check(False, f'{fmt} R={R} n={n}: bitwise {refname} ({int((y.view(torch.int32) != yr.view(torch.int32)).sum())} values differ)')
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
    if refmod is not None: check(nbits[0] > 0 and nbits[0] == nbits[1], f'{nbits[1]} of {nbits[0]} launches bitwise {refname}')
    print('RESULT', 'FAIL' if FAILS else 'PASS')
    return 1 if FAILS else 0


def launch(mod, fmt, IQK, grid, wd, R, K, xd, n, dev):
    """Y [n][R] fp32 = xd W^T through mod's kernel for fmt"""
    y = torch.zeros(n, R, dtype=torch.float32, device=dev)
    if fmt == 'Q4_K':
        mod.q3p_mm_q4k(grid, (256,), wd, I(R), I(K), xd, I(K), NULL, y, I(R), NULL, NULL, NULL, I(n))
    elif fmt == 'Q6_K':
        mod.q3p_mm_q6k(grid, (256,), wd, I(R), I(R), I(K), xd, I(K), NULL, y, I(R), NULL, NULL, NULL, I(n))
    elif fmt in IQK:
        getattr(mod, IQK[fmt])(grid, (256,), wd, I(R), I(R), I(K), xd, I(K), NULL, y, I(R), NULL, NULL, NULL, I(n))
    elif fmt == 'Q8_0':
        mod.q35p_mm_q8_0(grid, (256,), wd, I(R), I(K), xd, I(K), NULL, y, I(R), NULL, NULL, NULL, I(n))
    else:
        mod.q35p_mm_q5k(grid, (256,), wd, I(R), I(K), xd, I(K), NULL, y, I(R), NULL, NULL, NULL, I(n))
    return y


if __name__ == '__main__':
    sys.exit(main())
