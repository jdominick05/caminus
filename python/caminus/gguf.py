"""GGUF tensors for the K3 engine (docs/INFERENCE-PLAN.md I1/I2): raw blocks read with llama.cpp's gguf-py (the
vendored reader, caminus.config.ensure_gguf) and uploaded as-is, activations quantized to q8_1 exactly as llama.cpp does, and the per-format
matvecs of kernels/k3_gguf.hip (llama.cpp's own vec_dot functions)."""
import ctypes, os, sys
import numpy as np
import torch
from .hip import Module

from . import config as _config
_config.ensure_gguf()
from gguf import GGUFReader, GGMLQuantizationType as QT, GGML_QUANT_SIZES   # noqa: E402

MV = {QT.IQ3_XXS: 'gg_mv_iq3_xxs', QT.IQ4_NL: 'gg_mv_iq4_nl', QT.Q6_K: 'gg_mv_q6_k', QT.Q8_0: 'gg_mv_q8_0',
      # the GSQ-RCO Coder's formats: their standalone kernels are only in build/k3_gguf_gsq.hsaco (MV_GSQ)
      QT.Q4_K: 'gg_mv_q4_k', QT.Q5_K: 'gg_mv_q5_k', QT.IQ4_XS: 'gg_mv_iq4_xs', QT.IQ2_S: 'gg_mv_iq2_s',
      QT.IQ3_S: 'gg_mv_iq3_s', QT.Q2_0: 'gg_mv_q2_0'}
MV_GSQ = frozenset({QT.Q4_K, QT.Q5_K, QT.IQ4_XS, QT.IQ2_S, QT.IQ3_S, QT.Q2_0})
Q8_1_BYTES = 36          # block_q8_1: half2 ds (d, sum) + 32 int8
_mod = {}


def _m(module='k3_gguf'):
    key = (torch.cuda.current_device(), module)
    if key not in _mod: _mod[key] = Module(module)
    return _mod[key]


def tensor_rows(t, row0, nrows, slab=0):
    """raw bytes of rows [row0, row0 + nrows) of a GGUF tensor (2D, or 3D expert stack: `slab` = expert index), as a
    contiguous uint8 array, plus (ncols, qtype)"""
    d = np.asarray(t.data)
    rows = d[slab] if d.ndim == 3 else d
    return np.ascontiguousarray(rows[row0:row0 + nrows]).reshape(-1), int(t.shape[0]), t.tensor_type


def quantize_q8_1(x):
    """x fp32 [n] (cuda) -> q8_1 blocks (uint8 [ceil(n / 32) * 36]); values past n quantize as 0"""
    n = x.numel(); nb = (n + 31) // 32; ng = (nb * 32 + 255) // 256
    # the kernel writes a block for every launched lane (ng * 8 blocks): quantize into a buffer that holds them all and
    # return the first nb (an exact-size buffer took writes past its end whenever nb % 8 != 0)
    y = torch.empty(ng * 8 * Q8_1_BYTES, dtype=torch.uint8, device=x.device)
    _m().gg_quantize_q8_1((ng,), (256,), x.contiguous(), y, ctypes.c_int(n))
    return y[:nb * Q8_1_BYTES]


def matvec(w, qtype, nrows, ncols, yq):
    """w: raw GGUF blocks of nrows contiguous rows (uint8, cuda); yq: q8_1 activations -> fp32 [nrows]"""
    out = torch.empty(nrows, dtype=torch.float32, device=w.device)
    getattr(_m('k3_gguf_gsq' if qtype in MV_GSQ else 'k3_gguf'), MV[qtype])(((nrows + 3) // 4,), (32, 4), w, yq, out,
                                                                          ctypes.c_int(nrows), ctypes.c_int(ncols))
    return out


def silu_mul(g, u):
    """h = silu(g) * u with the K3 engine's own gg_silu (build/k3_gguf_gsq.hsaco): OP_GG_MOE_GU's composition"""
    n = g.numel(); h = torch.empty_like(g)
    _m('k3_gguf_gsq').gg_silu_mul(((n + 255) // 256,), (256,), g.contiguous(), u.contiguous(), h, ctypes.c_int(n))
    return h
