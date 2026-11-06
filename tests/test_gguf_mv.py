"""GGUF formats in the K3 engine (kernels/k3_gguf.hip): on real tensors of the Flash-Next REAP-308-UI cut, the q8_1
quantizer matches a numpy port of llama.cpp's, and each format's matvec (llama.cpp's own vec_dot) matches a float64
reference built from gguf-py's dequantizer and the same q8_1 activations, within fp32 summation-order error.
  HIP_VISIBLE_DEVICES=0 python tests/test_gguf_mv.py [model.gguf]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus import gguf as G
from gguf import quants

from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
PATH = sys.argv[1] if len(sys.argv) > 1 else devpaths.reap()
CASES = [('blk.0.ffn_gate_exps.weight', 0, 640, 7), ('blk.0.ffn_down_exps.weight', 0, 2560, 7),
         ('blk.0.attn_qkv.weight', 0, 2048, 0), ('blk.0.hc_attn_down.weight', 0, 320, 0)]   # name, row0, nrows, expert


def q8_1_numpy(x):
    """llama.cpp quantize_q8_1_lane in float32: d = amax / 127, q = roundf(x / d); returns q (int8) and d as stored (half)"""
    xb = x.astype(np.float32).reshape(-1, 32)
    amax = np.abs(xb).max(1); d = (amax / np.float32(127.0)).astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.where(d[:, None] > 0, xb / d[:, None], 0).astype(np.float32)
    q = (np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))).astype(np.int8)
    return q, d.astype(np.float16)


r = G.GGUFReader(PATH, 'r'); T = {t.name: t for t in r.tensors}
torch.manual_seed(0); ok = True
for name, row0, nrows, slab in CASES:
    raw, ncols, qt = G.tensor_rows(T[name], row0, nrows, slab)
    x = torch.randn(ncols, device='cuda') * 0.5
    yq = G.quantize_q8_1(x)
    # the quantizer vs numpy
    yb = yq.cpu().numpy().reshape(-1, 36)
    d_gpu = yb[:, :2].copy().view(np.float16)[:, 0]; q_gpu = yb[:, 4:].view(np.int8)
    q_np, d_np = q8_1_numpy(x.cpu().numpy())
    qmis = int((q_gpu != q_np).sum()); dmis = int((d_gpu != d_np).sum())
    # matvec vs float64 reference on the GPU's own activations
    out = G.matvec(torch.from_numpy(raw).cuda(), qt, nrows, ncols, yq).cpu().numpy().astype(np.float64)
    W = quants.dequantize(raw.reshape(nrows, -1), qt).astype(np.float64).reshape(nrows, ncols)
    xq = (q_gpu.astype(np.float64) * d_gpu.astype(np.float64)[:, None]).reshape(-1)
    ref = W @ xq; scale = np.abs(W) @ np.abs(xq)
    rel = float(np.max(np.abs(out - ref) / np.maximum(scale, 1e-30)))
    good = rel < 1e-5 and qmis == 0 and dmis == 0; ok &= good
    print(f'{qt.name:8s} {name:28s} {nrows:5d} x {ncols:5d}: max |out - ref| / sum|w||x| = {rel:.2e}; q8_1 mismatches q {qmis} d {dmis}  {"ok" if good else "FAIL"}')
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
