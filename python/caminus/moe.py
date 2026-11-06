# Parts of this module follow ggml's MMQ host code (ggml-cuda/mmq.cu / mmq.cuh: mul_mat_id's grouping tables as
# mm_ids_helper builds them, the launch configs, the q8_1 activation layout) from llama.cpp
# (https://github.com/ggml-org/llama.cpp, through the ignite-llama fork), which is distributed under the following license
# (THIRD_PARTY.md):
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

"""The batched prefill's routed-expert GEMMs of our own (kernels/prefill_moe.hip; project decision 11-03: prefill GEMM kernels
of our own), replacing the vendored MMQ's mul_mat_id (caminus.mmq) for the MoE gate / up / down projections:

  group(ids, ...)   k3_mmq_ids_clamp (the ids clamped in range and distinct per token, in place; the flag word set), then
                    k3_moe_group: expert_bounds, ids_dst (compact column -> t * k + slot), ids_inv (t * k + slot -> compact
                    column): the SAME tables as ggml's mm_ids_helper (tests/test_prefill_moe.py), plus two flat tile lists
                    (one per GEMM's column tile J). One call per layer and micro-chunk (mul_mat_id's path needs two
                    mm_ids_helper calls, 0.36 ms each at 2048 tokens on the XT)
  gu(...)           gate + up of every tile, swiglu, and the down projection's q8_1 input written straight into its compact
                    columns (the D4 layout of the down format)
  mm(...)           one GEMM over a tile list (the down projection: dst[ids_dst[c] * rows + r], mul_mat_id's per-slot output)

Numerics: bitwise ggml's MMQ per output (the vendored load_tiles / vec_dot at the same configs; the swiglu and the quantizer's
expressions; tests/test_prefill_moe.py g1 on real routed Coder inputs). An output depends only on its weight row and its
activation column, so the results do not depend on T, the split points, J or the tile an output lands in (G2b).

Bounds: every launch is refused (ValueError) unless each buffer covers what the kernel touches within its own view
(caminus.bufcheck; the weights' over-read past a K = 384 shard is the one storage check, as in caminus.mmq) and no written
buffer overlaps another. The device checks every tile entry and every destination row id before using it (ROUTE_TILES in
the routing flag word, never a write outside the destination); the host checks the flag word (prefill_ops.check_routing)
before it uses the chunk."""
import ctypes
import torch
from . import mmq
from .bufcheck import disjoint, view_bytes
from .hip import Module

FORMATS_GU = ('IQ2_S', 'IQ3_S', 'IQ3_XXS', 'IQ4_XS')   # the Coder's ffn_gate_exps / ffn_up_exps
FORMATS_DOWN = ('IQ4_NL', 'Q2_0')                     # its ffn_down_exps
FORMATS = FORMATS_GU + FORMATS_DOWN
JS = (16, 32, 48, 64, 96, 128)
J_MIN = mmq.MOE_J_MIN
EMAX = 512                    # k3_moe_group's K3_MOE_EMAX
TCHUNK = 64                   # its K3_MOE_TCHUNK (tokens per placement round; k * TCHUNK <= 1024 threads)
CFG_WORDS = 9
TILE_BYTES = mmq.MOE_TILE_BYTES   # int4 per tile list entry
SMPBO = mmq.SMPBO
# the column tiles ChunkStack uses (measured 11-04 on the XT, tests/test_prefill_moe.py speed + the variant runs, real routing,
# layers 0 / 1 / 17, both cards' shapes, medians of 5-9): the fused gate / up kernel at J 64 (with the 16-column skip, J 64 beat
# 48 by 2-7 %; IQ2_S's J 64 config is its 128-row one, which the one-weight-tile LDS layout fits); the down kernel at J 32 for
# pieces of <= 512 tokens, 64 above (within 1-3 % of the best everywhere)
J_GU = 64


def j_down(ntok):
    return 32 if int(ntok) <= 512 else 64


def max_tiles(ncols, n_exp, J):
    """an upper bound on a tile list's length: sum_e ceil(c_e / J) <= n_exp + ceil(sum_e c_e / J)"""
    return int(n_exp) + (int(ncols) + int(J) - 1) // int(J)


class MoE:
    """the MoE kernels on one device (every launch with that device current, on its current stream or the one given).
    mm: the caminus.mmq.MMQ of the same device (its k3_mmq_ids_clamp runs before every grouping)"""
    def __init__(self, device, mm=None, formats=FORMATS):
        self.dev = torch.device(device)
        assert self.dev.type == 'cuda', device
        self.mods, self.cfg, self.kmm, self.kgu = {}, {}, {}, {}
        with torch.cuda.device(self.dev):
            self.glue = Module('prefill_moe_glue'); self.mods['prefill_moe_glue'] = self.glue
            self.kgroup = self.glue.k3_moe_group
            if mm is None:
                self._mglue = Module('prefill_mmq_glue'); self.kclamp = self._mglue.k3_mmq_ids_clamp
            else:
                assert mm.dev == self.dev, (mm.dev, self.dev); self.kclamp = mm.kclamp
            for f in formats: self._load(f)

    def _load(self, fmt):
        name = f'prefill_moe_{fmt.lower()}'
        with torch.cuda.device(self.dev):
            mod = Module(name); self.mods[name] = mod
            out = torch.zeros(16 * CFG_WORDS, dtype=torch.int32, device=self.dev); cnt = torch.zeros(1, dtype=torch.int32, device=self.dev)
            mod.k3_moe_configs((1,), (1,), out, cnt); torch.cuda.synchronize(self.dev)
            n = int(cnt.item()); rows = out[:n * CFG_WORDS].view(n, CFG_WORDS).tolist()
            assert [r[0] for r in rows] == list(JS), (fmt, rows)
            for J, nth, I, occ, stride, qk, nb_mm, nb_gu, iterk in rows:
                assert qk == mmq.BLOCK[fmt][0] and iterk == mmq.K_VRAM and nb_mm <= SMPBO and nb_gu <= SMPBO, (fmt, J)
                self.cfg[(fmt, J)] = dict(nthreads=nth, I=I, nbytes_mm=nb_mm, nbytes_gu=nb_gu)
                self.kmm[(fmt, J)] = getattr(mod, f'k3_moe_mm_{J}')
                if nb_gu: self.kgu[(fmt, J)] = getattr(mod, f'k3_moe_gu_{J}')

    def _launch(self, k, *a, **kw):
        with torch.cuda.device(self.dev): k(*a, **kw)

    # -- checks -------------------------------------------------------------------------------------------------------------
    def _chk_dev(self, t, what):
        if not isinstance(t, torch.Tensor) or t.device != self.dev or not t.is_contiguous():
            raise ValueError(f'moe: {what} must be a contiguous tensor on {self.dev}')

    def _chk(self, t, dtype, nbytes, what):
        self._chk_dev(t, what)
        if t.dtype != dtype: raise ValueError(f'moe: {what} must be {dtype}')
        if view_bytes(t) < nbytes: raise ValueError(f'moe: {what} holds {view_bytes(t)} B < {nbytes} B')

    def _chk_w(self, w, fmt, n_exp, rows, K, I, what):
        self._chk_dev(w, what)
        if w.dtype != torch.uint8: raise ValueError(f'moe: {what} must be the raw uint8 GGUF rows')
        if rows < 1 or rows % I: raise ValueError(f'moe: {what}: rows {rows} must be a multiple of the row block {I}')
        rb = mmq.row_bytes(fmt, K); need = n_exp * rows * rb
        if view_bytes(w) < need: raise ValueError(f'moe: {what} holds {view_bytes(w)} B < {n_exp} x {rows} x {rb}')
        if mmq.extent(w) < need + mmq.overread_bytes(fmt, K):   # load_tiles reads K in 256-value steps (mmq's over-read)
            raise ValueError(f'moe: {what}: {fmt} K={K} is read {mmq.overread_bytes(fmt, K)} B past the last row; the allocation ends '
                             f'{mmq.extent(w) - need} B after it (pad the tensor: mmq.pad_bytes)')
        return need + mmq.overread_bytes(fmt, K)

    def _cfg(self, fmt, J, gu):
        c = self.cfg.get((fmt, J))
        if c is None or (gu and (fmt, J) not in self.kgu): raise ValueError(f'moe: no {"gu" if gu else "mm"} kernel for {fmt} J={J}')
        return c

    # -- the grouping -------------------------------------------------------------------------------------------------------
    def group(self, ids, ntok, k, n_exp, bounds, ids_dst, ids_inv, tiles_a, ja, tiles_b, jb, ntiles, err, stream=None):
        """ids [ntok][k] int32 (clamped IN PLACE first: k3_mmq_ids_clamp, err |= ROUTE_RANGE / ROUTE_DUP) -> bounds [n_exp + 1],
        ids_dst [ntok * k], ids_inv [ntok * k], tile lists tiles_a (column tiles of ja) / tiles_b (jb), int4 entries,
        ntiles[0] / [1] their lengths"""
        ntok, k, n_exp, ja, jb = int(ntok), int(k), int(n_exp), int(ja), int(jb)
        if not 1 <= k <= mmq.IDS_KMAX or k * TCHUNK > 1024: raise ValueError(f'moe.group: k {k} (1 .. {min(mmq.IDS_KMAX, 1024 // TCHUNK)})')
        if not k <= n_exp <= EMAX: raise ValueError(f'moe.group: n_exp {n_exp} (k .. {EMAX})')
        if not 0 < ntok < (1 << 22) // k: raise ValueError(f'moe.group: ntok {ntok}')
        if ja not in JS or jb not in JS: raise ValueError(f'moe.group: column tiles {ja} / {jb} (one of {JS})')
        N = ntok * k; ma, mb = max_tiles(N, n_exp, ja), max_tiles(N, n_exp, jb)
        I32 = torch.int32
        self._chk(ids, I32, 4 * N, 'ids'); self._chk(bounds, I32, 4 * (n_exp + 1), 'bounds')
        self._chk(ids_dst, I32, 4 * N, 'ids_dst'); self._chk(ids_inv, I32, 4 * N, 'ids_inv')
        self._chk(tiles_a, I32, TILE_BYTES * ma, 'tiles_a'); self._chk(tiles_b, I32, TILE_BYTES * mb, 'tiles_b')
        self._chk(ntiles, I32, 8, 'ntiles'); self._chk(err, I32, 4, 'err')
        if tiles_a.data_ptr() % 16 or tiles_b.data_ptr() % 16: raise ValueError('moe.group: the tile lists must be 16-byte aligned')
        disjoint('moe.group', dict(ids=(ids, 4 * N), bounds=(bounds, 4 * (n_exp + 1)), ids_dst=(ids_dst, 4 * N), ids_inv=(ids_inv, 4 * N),
                                   tiles_a=(tiles_a, TILE_BYTES * ma), tiles_b=(tiles_b, TILE_BYTES * mb), ntiles=(ntiles, 8), err=(err, 4)),
                 ('ids', 'bounds', 'ids_dst', 'ids_inv', 'tiles_a', 'tiles_b', 'ntiles', 'err'))
        self._launch(self.kclamp, ((ntok + 255) // 256,), (256,), ids, ctypes.c_int(ntok), ctypes.c_int(k), ctypes.c_int(n_exp), err, stream=stream)
        self._launch(self.kgroup, (1,), (1024,), ids, ctypes.c_int(ntok), ctypes.c_int(k), ctypes.c_int(n_exp), bounds, ids_dst, ids_inv,
                     tiles_a, ctypes.c_int(ja), ctypes.c_int(ma), tiles_b, ctypes.c_int(jb), ctypes.c_int(mb), ntiles, err, stream=stream)

    # -- the GEMMs ----------------------------------------------------------------------------------------------------------
    def mm(self, fmt, w, n_exp, rows, K, y, ncols, ids_dst, tiles, ntiles, dst, err, J, stream=None):
        """dst[ids_dst[c] * rows + r] = sum_k W_e[r, k] y_c for every compact column c of every tile (e, c0, nc) of the list
        (W: [n_exp][rows][K] of fmt; y: block_q8_1_mmq [K_pad / 128][ncols] in fmt's layout; ncols = ntok * k)"""
        rows, K, ncols, n_exp, J = int(rows), int(K), int(ncols), int(n_exp), int(J)
        c = self._cfg(fmt, J, False)
        if K % 128: raise ValueError(f'moe.mm: K {K} must be a multiple of 128 (whole activation half-tiles)')
        if not 1 <= ncols < (1 << 22) or not 1 <= n_exp <= EMAX: raise ValueError(f'moe.mm: ncols {ncols}, n_exp {n_exp}')
        wb = self._chk_w(w, fmt, n_exp, rows, K, c['I'], 'w')
        mt = max_tiles(ncols, n_exp, J); I32 = torch.int32
        self._chk_dev(y, 'y')
        if view_bytes(y) < mmq.y_bytes(K, ncols): raise ValueError(f'moe.mm: y holds {view_bytes(y)} B < {mmq.y_bytes(K, ncols)}')
        self._chk(ids_dst, I32, 4 * ncols, 'ids_dst'); self._chk(tiles, I32, TILE_BYTES * mt, 'tiles'); self._chk(ntiles, I32, 4, 'ntiles')
        self._chk(dst, torch.float32, 4 * rows * ncols, 'dst'); self._chk(err, I32, 4, 'err')
        if tiles.data_ptr() % 16: raise ValueError('moe.mm: the tile list must be 16-byte aligned')
        disjoint('moe.mm', dict(w=(w, wb), y=(y, mmq.y_bytes(K, ncols)), ids_dst=(ids_dst, 4 * ncols), tiles=(tiles, TILE_BYTES * mt),
                                ntiles=(ntiles, 4), dst=(dst, 4 * rows * ncols), err=(err, 4)), ('dst', 'err'))
        qk = mmq.BLOCK[fmt][0]
        self._launch(self.kmm[(fmt, J)], (rows // c['I'], mt), (32, c['nthreads'] // 32), w, y, ids_dst, tiles, ntiles, dst, err,
                     ctypes.c_int(n_exp), ctypes.c_int(rows), ctypes.c_int(K // qk), ctypes.c_int(K), ctypes.c_int(ncols),
                     ctypes.c_int(ncols), shmem=c['nbytes_mm'], stream=stream)

    def gu(self, fmt, wg, wu, n_exp, rows, K, y, ncols, tiles, ntiles, yd, err, J, fmt_down, stream=None):
        """for every tile (e, c0, nc) and row r < rows: h = silu(Wg_e y_c) * (Wu_e y_c) (pf_swiglu), quantized as
        quantize_mmq_q8_1 (D4) into yd, the down projection's activations block_q8_1_mmq [k_pad(rows) / 128][ncols] (its value
        r of compact column c); fmt_down: the down projection's format (its activation layout must be D4)"""
        rows, K, ncols, n_exp, J = int(rows), int(K), int(ncols), int(n_exp), int(J)
        c = self._cfg(fmt, J, True)
        if mmq.TYPE_ID.get(fmt_down) is None or fmt_down not in FORMATS_DOWN: raise ValueError(f'moe.gu: down format {fmt_down}')
        if K % mmq.K_VRAM: raise ValueError(f'moe.gu: K {K} must be a multiple of {mmq.K_VRAM}')
        if rows % 128 or rows % mmq.BLOCK[fmt_down][0]: raise ValueError(f'moe.gu: rows {rows} (the down K) must be whole 128-value blocks')
        if not 1 <= ncols < (1 << 22) or not 1 <= n_exp <= EMAX: raise ValueError(f'moe.gu: ncols {ncols}, n_exp {n_exp}')
        wgb = self._chk_w(wg, fmt, n_exp, rows, K, c['I'], 'wg'); wub = self._chk_w(wu, fmt, n_exp, rows, K, c['I'], 'wu')
        mt = max_tiles(ncols, n_exp, J); I32 = torch.int32
        self._chk_dev(y, 'y'); self._chk_dev(yd, 'yd')
        if view_bytes(y) < mmq.y_bytes(K, ncols): raise ValueError(f'moe.gu: y holds {view_bytes(y)} B < {mmq.y_bytes(K, ncols)}')
        ydb = mmq.y_bytes(rows, ncols)
        if view_bytes(yd) < ydb: raise ValueError(f'moe.gu: yd holds {view_bytes(yd)} B < {ydb}')
        if yd.data_ptr() % 16: raise ValueError('moe.gu: yd must be 16-byte aligned')
        self._chk(tiles, I32, TILE_BYTES * mt, 'tiles'); self._chk(ntiles, I32, 4, 'ntiles'); self._chk(err, I32, 4, 'err')
        if tiles.data_ptr() % 16: raise ValueError('moe.gu: the tile list must be 16-byte aligned')
        disjoint('moe.gu', dict(wg=(wg, wgb), wu=(wu, wub), y=(y, mmq.y_bytes(K, ncols)), tiles=(tiles, TILE_BYTES * mt), ntiles=(ntiles, 4),
                                yd=(yd, ydb), err=(err, 4)), ('yd', 'err'))
        qk = mmq.BLOCK[fmt][0]
        self._launch(self.kgu[(fmt, J)], (rows // c['I'], mt), (32, c['nthreads'] // 32), wg, wu, y, tiles, ntiles, yd, err,
                     ctypes.c_int(n_exp), ctypes.c_int(rows), ctypes.c_int(K // qk), ctypes.c_int(ncols), shmem=c['nbytes_gu'], stream=stream)
