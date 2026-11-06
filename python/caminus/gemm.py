"""C = A . B^T on our gfx1100 WMMA GEMM (kernels/gemm.hip): A [M, K], B [N, K] bf16, K-contiguous; fp32 accumulation
in a fixed K order; bf16 or fp32 output; optional fp32 row scales rs [M] and column scales cs [N] on the accumulator.
round_bf16 (fp32 output, no scales): every result rounded to bf16 (round to nearest even) and stored as fp32 (the *_rbf
kernels: llama.cpp's batched BF16 mul_mat output precision on RDNA3, caminus.prefill_ops.Glue(wmma_out='bf16'))."""
import ctypes
import torch
from .hip import Module

_mod = {}
TILES = {(128, 128): 'gemm_tn_128x128', (128, 64): 'gemm_tn_128x64', (64, 128): 'gemm_tn_64x128'}
TILES_RBF = {(128, 128): 'gemm_tn_128x128_rbf', (128, 64): 'gemm_tn_128x64_rbf', (64, 128): 'gemm_tn_64x128_rbf'}


def _m():
    dev = torch.cuda.current_device()
    if dev not in _mod: _mod[dev] = Module('gemm')
    return _mod[dev]


def supported(M, N, K, tile=(128, 128)):
    return M % tile[0] == 0 and N % tile[1] == 0 and K % 32 == 0


def gemm_tn(a, b, out_dtype=torch.bfloat16, rs=None, cs=None, tile=(128, 128), out=None, kernel=None, round_bf16=False):
    M, K = a.shape; N = b.shape[0]
    assert b.shape[1] == K and a.dtype == b.dtype == torch.bfloat16 and a.stride(1) == 1 and b.stride(1) == 1
    assert supported(M, N, K, tile), f'gemm_tn: {M}x{N}x{K} does not tile by {tile} x 32'
    assert a.stride(0) % 8 == 0 and b.stride(0) % 8 == 0 and a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0
    c = out if out is not None else torch.empty(M, N, device=a.device, dtype=out_dtype)
    assert c.stride(1) == 1 and (c.dtype == torch.float32 or (c.stride(0) % 8 == 0 and c.data_ptr() % 16 == 0))
    nul = ctypes.c_void_p(None)
    if round_bf16:
        assert c.dtype == torch.float32 and rs is None and cs is None and kernel is None, 'gemm_tn round_bf16: fp32 output, no scales'
        getattr(_m(), TILES_RBF[tile])((N // tile[1], M // tile[0]), (128,), a, b, c, ctypes.c_int(K), ctypes.c_int(a.stride(0)),
                                        ctypes.c_int(b.stride(0)), ctypes.c_int(c.stride(0)))
        return c
    getattr(_m(), kernel or TILES[tile])((N // tile[1], M // tile[0]), (128,), a, b, c,
                                          rs if rs is not None else nul, cs if cs is not None else nul,
                                          ctypes.c_int(K), ctypes.c_int(a.stride(0)), ctypes.c_int(b.stride(0)),
                                          ctypes.c_int(c.stride(0)), ctypes.c_int(int(c.dtype == torch.float32)))
    return c


TILES_IU4 = {(128, 128): 'gemm_tn_iu4_128x128', (128, 64): 'gemm_tn_iu4_128x64', (64, 128): 'gemm_tn_iu4_64x128'}


def pack_int4(q):
    """int codes in [-8, 7], [R, K] (K even) -> uint8 [R, K / 2], element 2j in the low nibble of byte j"""
    q = q.to(torch.int16)
    return ((q[:, 0::2] & 0xF) | ((q[:, 1::2] & 0xF) << 4)).to(torch.uint8).contiguous()


def gemm_tn_iu4(qa, qb, rs=None, cs=None, out_dtype=torch.bfloat16, tile=(128, 128), out=None):
    """C = rs[:, None] * cs[None, :] * (codes_a . codes_b^T): packed int4 operands (pack_int4) [M, K / 2], [N, K / 2];
    int32 accumulation (exact), fp32 scale epilogue (cs, then rs), bf16 or fp32 output"""
    M, Kh = qa.shape; N = qb.shape[0]; K = 2 * Kh
    assert qb.shape[1] == Kh and qa.dtype == qb.dtype == torch.uint8 and qa.stride(1) == 1 and qb.stride(1) == 1
    assert M % tile[0] == 0 and N % tile[1] == 0 and K % 128 == 0, f'gemm_tn_iu4: {M}x{N}x{K} does not tile by {tile} x 128'
    assert qa.stride(0) % 16 == 0 and qb.stride(0) % 16 == 0 and qa.data_ptr() % 16 == 0 and qb.data_ptr() % 16 == 0
    c = out if out is not None else torch.empty(M, N, device=qa.device, dtype=out_dtype)
    nul = ctypes.c_void_p(None)
    getattr(_m(), TILES_IU4[tile])((N // tile[1], M // tile[0]), (128,), qa, qb, c,
                                    rs if rs is not None else nul, cs if cs is not None else nul,
                                    ctypes.c_int(K), ctypes.c_int(qa.stride(0)), ctypes.c_int(qb.stride(0)),
                                    ctypes.c_int(c.stride(0)), ctypes.c_int(int(c.dtype == torch.float32)))
    return c


TILES_IU4G = {(128, 64): 'gemm_tn_iu4g_128x64', (64, 128): 'gemm_tn_iu4g_64x128'}


def gemm_tn_iu4g(qa, qb, gs, rs=None, out_dtype=torch.bfloat16, tile=(64, 128), out=None):
    """C = rs[:, None] * acc with acc = fma(float(P_g), gs[:, g], acc) over the 128-groups g in order (P_g the exact int32
    dot of group g's codes): the g128 weight quantizer on iu4; packed int4 operands [M, K / 2], [N, K / 2], gs [N, K / 128]"""
    M, Kh = qa.shape; N = qb.shape[0]; K = 2 * Kh
    assert qb.shape[1] == Kh and qa.dtype == qb.dtype == torch.uint8 and qa.stride(1) == 1 and qb.stride(1) == 1
    assert M % tile[0] == 0 and N % tile[1] == 0 and K % 128 == 0, f'gemm_tn_iu4g: {M}x{N}x{K} does not tile by {tile} x 128'
    assert gs.shape == (N, K // 128) and gs.dtype == torch.float32 and gs.is_contiguous()
    c = out if out is not None else torch.empty(M, N, device=qa.device, dtype=out_dtype)
    getattr(_m(), TILES_IU4G[tile])((N // tile[1], M // tile[0]), (128,), qa, qb, c,
                                     rs if rs is not None else ctypes.c_void_p(None), gs,
                                     ctypes.c_int(K), ctypes.c_int(qa.stride(0)), ctypes.c_int(qb.stride(0)),
                                     ctypes.c_int(c.stride(0)), ctypes.c_int(int(c.dtype == torch.float32)))
    return c


class ASeg(ctypes.Structure):
    _fields_ = [('p', ctypes.c_void_p * 8), ('ld', ctypes.c_int * 8), ('kend', ctypes.c_int * 8), ('n', ctypes.c_int)]


TILES_SEG = {(128, 128): 'gemm_tn_seg_128x128', (128, 64): 'gemm_tn_seg_128x64', (64, 128): 'gemm_tn_seg_64x128'}


def gemm_tn_seg(segs, b, tile=(128, 128)):
    """C = cat(segs, -1) . b^T without the cat: segs = up to 8 bf16 [M, k_i] (k_i % 32 == 0, rows 16-byte aligned, any
    row stride), b [N, sum k_i] K-contiguous; bf16 output, the same accumulation order as on the concatenated A"""
    M = segs[0].shape[0]; N, K = b.shape
    assert len(segs) <= 8 and sum(s.shape[1] for s in segs) == K and b.dtype == torch.bfloat16 and b.stride(1) == 1
    assert M % tile[0] == 0 and N % tile[1] == 0, f'gemm_tn_seg: {M}x{N} does not tile by {tile}'
    a = ASeg(); kend = 0
    for i, s in enumerate(segs):
        assert s.shape[0] == M and s.dtype == torch.bfloat16 and s.stride(1) == 1 and s.stride(0) % 8 == 0
        assert s.shape[1] % 32 == 0 and s.data_ptr() % 16 == 0
        kend += s.shape[1]; a.p[i] = s.data_ptr(); a.ld[i] = s.stride(0); a.kend[i] = kend
    a.n = len(segs)
    c = torch.empty(M, N, device=b.device, dtype=torch.bfloat16)
    getattr(_m(), TILES_SEG[tile])((N // tile[1], M // tile[0]), (128,), a, b, c, ctypes.c_int(K), ctypes.c_int(b.stride(0)),
                                   ctypes.c_int(c.stride(0)))
    return c
