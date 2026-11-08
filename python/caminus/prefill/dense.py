# Parts of this module follow ggml's MMQ host code (ggml-cuda/mmq.cuh launch_mul_mat_q: the column-tile (J) rule, the launch
# configs, the q8_1 activation layout and padding) from llama.cpp (https://github.com/ggml-org/llama.cpp, through the
# ignite-llama fork), which is distributed under the following license (THIRD_PARTY.md):
#
# MIT License
#
# Copyright (c) 2023-2026 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The batched prefill's dense quantized GEMMs of our own (kernels/prefill_dense.hip, impl:q6k-dense; project decision 11-03:
prefill GEMM kernels of our own), replacing the vendored MMQ's mul_mat_q (caminus.prefill.mmq.MMQ.mm) for the Q6_K dense matrices:

  mm(fmt, w, rows, K, y, ncols, dst)   dst[c * rows + r] = sum_k W[r, k] x[c, k] for c < ncols; W: [rows][K] raw Q6_K rows;
                                       y: x quantized by caminus.prefill.mmq.MMQ.quantize in the format's layout (D4), the same buffer
                                       MMQ reads (the D-wide mixed row is quantized once per half-layer for every format)

Numerics: BITWISE ggml's MMQ per output at every variant (VD) and column tile (J): MMQ's per-16-value sub-block sum
fma(fl(C * sc * d), dB, sum) in K order, with the scale product computed exactly another way (the kernel file's header says
why each form rounds once, the same real); tests/prefill/test_prefill_dense.py g1 against mmq.mm on real Coder Q6_K shards, g2b over
chunk splits. An output depends only on its weight row and its activation column, so T, J and the split points change no bit.

Bounds: every launch is refused (ValueError) unless the weights cover rows x K within their view, y covers mmq.y_bytes(K,
ncols) within its view, dst covers rows x ncols fp32 within its view, rows is a multiple of the row block (64) and no written
buffer overlaps another (caminus.device.bufcheck). The kernel reads those rows x K blocks and, like MMQ, whole J-column activation
tiles (up to J - 1 columns past ncols: the Y_TAIL inside mmq.y_bytes), and writes only dst[0, rows * ncols)."""
import ctypes
import torch
from . import mmq
from ..device.bufcheck import ALIGN, aligned, disjoint, view_bytes   # noqa: F401  (ALIGN: dense.ALIGN)
from ..device.hip import Module

FORMATS = ('Q6_K',)
JS = (16, 32, 48, 64, 96, 128)
VDS = (0, 1, 2, 3)            # the tile products (kernels/prefill_dense.hip): 0 the vendored one (the control), 1 .. 3 ours
JS_VD = {vd: JS for vd in VDS}
VD = 3                        # the default (measured: tests/prefill/test_prefill_dense.py speed)
CFG_WORDS = 9
SMPBO = mmq.SMPBO


SMALL_ROWS, J_SMALL = 1024, 64


def choose_J(ncols, rows=None):
    """the column tile: llama.cpp's own rule (the fewest column tiles, the smallest J on a tie), at most J_SMALL for a matrix
    of fewer than SMALL_ROWS rows (more workgroups: the Coder's 256 / 384-row attn_k / attn_v / shared-expert shards, measured
    11-04 on the XT: J 64 0.062 / 0.108 / 0.087 / 0.135 ms vs J 128 0.068 / 0.110 / 0.108 / 0.150 at n 1024 / 2048, ~1 ms per
    2K chunk in all; the larger shapes are within 1 % of their best J at llama.cpp's). Any J gives the same bits"""
    best, nt_best = 0, 1 << 62
    for J in JS:
        nt = (int(ncols) + J - 1) // J
        if nt < nt_best: best, nt_best = J, nt
    if rows is not None and int(rows) < SMALL_ROWS: best = min(best, J_SMALL)
    return best


class Dense:
    """the dense kernels on one device (every launch with that device current, on its current stream or the one given)"""
    def __init__(self, device, formats=FORMATS):
        self.dev = torch.device(device)
        assert self.dev.type == 'cuda', device
        self.mods, self.cfg, self.k, self.kdeq = {}, {}, {}, {}
        with torch.cuda.device(self.dev):
            for f in formats: self._load(f)

    def _load(self, fmt):
        if fmt not in FORMATS: raise ValueError(f'dense: no kernels for {fmt}')
        name = f'prefill_dense_{fmt.lower()}'
        with torch.cuda.device(self.dev):
            mod = Module(name); self.mods[name] = mod
            self.kdeq[fmt] = mod.k3_dense_deq_bf16
            out = torch.zeros(64 * CFG_WORDS, dtype=torch.int32, device=self.dev); cnt = torch.zeros(1, dtype=torch.int32, device=self.dev)
            mod.k3_dense_configs((1,), (1,), out, cnt); torch.cuda.synchronize(self.dev)
            n = int(cnt.item()); rows = out[:n * CFG_WORDS].view(n, CFG_WORDS).tolist()
            assert sorted((r[0], r[1]) for r in rows) == sorted((v, J) for v in VDS for J in JS_VD[v]), (fmt, rows)
            for vd, J, nth, I, occ, stride, qk, nbytes, iterk in rows:
                assert qk == mmq.BLOCK[fmt][0] and iterk == mmq.K_VRAM and nbytes <= SMPBO and nth % 32 == 0, (fmt, vd, J)
                self.cfg[(fmt, vd, J)] = dict(nthreads=nth, I=I, nbytes=nbytes)
                self.k[(fmt, vd, J)] = getattr(mod, f'k3_dense_mm_v{vd}_{J}')

    def mm(self, fmt, w, rows, K, y, ncols, dst, J=None, vd=None, stream=None):
        """dst[c * rows + r] = sum_k W[r, k] x[c, k] for c < ncols (bitwise mmq.MMQ.mm on the same w / y). Returns J"""
        rows, K, ncols = int(rows), int(K), int(ncols)
        vd = VD if vd is None else int(vd)
        J = choose_J(ncols, rows) if J is None else int(J)
        c = self.cfg.get((fmt, vd, J))
        if c is None: raise ValueError(f'dense.mm: no {fmt} kernel for VD {vd}, J {J}')
        qk = mmq.BLOCK[fmt][0]
        if K < qk or K % mmq.K_VRAM: raise ValueError(f'dense.mm: K {K} must be a positive multiple of {mmq.K_VRAM} ({fmt})')
        if rows < 1 or rows % c['I']: raise ValueError(f'dense.mm: rows {rows} must be a multiple of {c["I"]}')
        if not 1 <= ncols < (1 << 22): raise ValueError(f'dense.mm: ncols {ncols}')
        for t, what in ((w, 'w'), (y, 'y'), (dst, 'dst')):
            if not isinstance(t, torch.Tensor) or t.device != self.dev or not t.is_contiguous():
                raise ValueError(f'dense.mm: {what} must be a contiguous tensor on {self.dev}')
        if w.dtype != torch.uint8: raise ValueError('dense.mm: w must be the raw uint8 GGUF rows')
        if dst.dtype != torch.float32: raise ValueError('dense.mm: dst must be float32')
        wb = rows * mmq.row_bytes(fmt, K); yb = mmq.y_bytes(K, ncols); db = 4 * rows * ncols
        if view_bytes(w) < wb: raise ValueError(f'dense.mm: w holds {view_bytes(w)} B < {rows} x {mmq.row_bytes(fmt, K)}')
        if w.data_ptr() % 2: raise ValueError('dense.mm: w must be 2-byte aligned (the kernel reads 16-bit words)')
        if view_bytes(y) < yb: raise ValueError(f'dense.mm: y holds {view_bytes(y)} B < {yb} ({ncols} columns)')
        if y.data_ptr() % 16: raise ValueError('dense.mm: y must be 16-byte aligned')
        if view_bytes(dst) < db: raise ValueError(f'dense.mm: dst holds {view_bytes(dst)} B < {rows} x {ncols} fp32')
        disjoint('dense.mm', dict(w=(w, wb), y=(y, yb), dst=(dst, db)), ('dst',))
        self._launch(self.k[(fmt, vd, J)], (rows // c['I'], (ncols + J - 1) // J, 1), (32, c['nthreads'] // 32), w, y, dst, ctypes.c_int(rows),
                     ctypes.c_int(K // qk), ctypes.c_int(ncols), shmem=c['nbytes'], stream=stream)
        return J

    def deq_bf16(self, fmt, w, rows, K, out, stream=None):
        """out [rows][K] bf16 = W dequantized (ggml's dequantize_row_q6_K in fp32, each value rounded to bf16): candidate (b) of
        impl:q6k-dense, the B operand (W) of gemm.hip's WMMA (ChunkStack(q6k='wmma')). Not bitwise MMQ: a tolerance"""
        rows, K = int(rows), int(K)
        if fmt != 'Q6_K': raise ValueError(f'dense.deq_bf16: {fmt} (Q6_K only)')
        if K < mmq.K_VRAM or K % mmq.K_VRAM or rows < 1: raise ValueError(f'dense.deq_bf16: rows {rows}, K {K}')
        for t, what in ((w, 'w'), (out, 'out')):
            if not isinstance(t, torch.Tensor) or t.device != self.dev or not t.is_contiguous():
                raise ValueError(f'dense.deq_bf16: {what} must be a contiguous tensor on {self.dev}')
        if w.dtype != torch.uint8: raise ValueError('dense.deq_bf16: w must be the raw uint8 GGUF rows')
        if out.dtype != torch.bfloat16: raise ValueError('dense.deq_bf16: out must be bfloat16')
        wb, ob = rows * mmq.row_bytes(fmt, K), 2 * rows * K
        if view_bytes(w) < wb: raise ValueError(f'dense.deq_bf16: w holds {view_bytes(w)} B < {wb}')
        if w.data_ptr() % 2: raise ValueError('dense.deq_bf16: w must be 2-byte aligned')
        if view_bytes(out) < ob: raise ValueError(f'dense.deq_bf16: out holds {view_bytes(out)} B < {rows} x {K} bf16')
        disjoint('dense.deq_bf16', dict(w=(w, wb), out=(out, ob)), ('out',))
        nblk = rows * (K // mmq.BLOCK[fmt][0])
        self._launch(self.kdeq[fmt], (nblk, 1, 1), (64, 1, 1), w, out, ctypes.c_int(nblk), stream=stream)

    def _launch(self, k, *a, **kw):
        with torch.cuda.device(self.dev): k(*a, **kw)


# -- candidate (b)'s buffers (ChunkStack(q6k='wmma'); each ALIGN-aligned: caminus.device.bufcheck) -----------------------------


def bf16_layout(T, d):
    """name -> bytes of one card's buffers for ChunkStack(q6k='wmma') at chunks of up to T tokens (d: mmq.card_dims): w_bf the
    largest dense matrix of the card's shapes as bf16 (each Q6_K matrix is dequantized into it right before its GEMM), x_bf the
    activation's bf16 copy (tpad(T) rows x the widest K: gemm.hip's A operand)"""
    T = int(T); Tw = -(-T // mmq.WMMA_ROWS) * mmq.WMMA_ROWS
    L = dict(w_bf=2 * max(d['qkv'] * d['D'], d['z'] * d['D'], d['D'] * d['ssm_K'], d['q'] * d['D'], d['kv'] * d['D'],
                          d['D'] * d['ao_K'], d['ffc'] * d['D'], d['D'] * d['ffc']),
             x_bf=2 * Tw * max(d['D'], d['ssm_K'], d['ao_K'], d['ffc']))
    return aligned(L)


def bf16_bytes(T, d):
    return sum(bf16_layout(T, d).values())


class Bf16Workspace:
    """one card's preallocated buffers of ChunkStack(q6k='wmma') (bf16_layout): one uint8 allocation, .w_bf / .x_bf bfloat16
    views; checked against the card's cap (capacity.check) before it allocates"""
    def __init__(self, device, T, dims, check=True):
        from ..device import capacity
        self.dev, self.T, self.dims = torch.device(device), int(T), dict(dims)
        self.layout = bf16_layout(self.T, self.dims); self.nbytes = sum(self.layout.values())
        if check: capacity.check([self.dev.index], [self.nbytes / capacity.MIB], f'prefill bf16 dense buffers (T={self.T})')
        self.block = torch.zeros(self.nbytes, dtype=torch.uint8, device=self.dev)
        off = 0
        for n, b in self.layout.items():
            setattr(self, n, self.block[off:off + b].view(torch.bfloat16)); off += b
