"""Caminus's own host-side decoders of GGUF blocks to float32 (numpy), written from the block formats
(docs/specs/spec-own-quant.md); gguf-py's dequantize is the test oracle (tests/io/test_own_dequant.py: bitwise equal).
The arithmetic follows the formats' definition in float32: Q4_K y = (d * sc) * q - (dmin * m); Q6_K y = (d * sc) * q.

  dequant(raw, type_name, n_values) -> float32 [n_values]   (raw: the blocks' bytes, uint8)
  rows(t, r0, r1)                    -> float32 [r1 - r0, K] (a caminus.ggufio.Tensor's rows r0 .. r1 - 1)"""
import numpy as np


def _f16(b):
    return b.view(np.float16).astype(np.float32)


def q4_k(raw):
    """Q4_K: 144-byte blocks of 256: d, dmin (fp16), 12 bytes of 6-bit scales / mins, 128 bytes of nibbles"""
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 144)
    d, dmin = _f16(b[:, 0:2].copy()).reshape(-1, 1), _f16(b[:, 2:4].copy()).reshape(-1, 1)
    s = b[:, 4:16].astype(np.uint8)
    sc = np.empty((b.shape[0], 8), np.uint8); m = np.empty((b.shape[0], 8), np.uint8)
    sc[:, :4] = s[:, 0:4] & 63; m[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    m[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    dl = (d * sc.astype(np.float32)).reshape(-1, 8, 1)               # (float32 products, as the format defines)
    ml = (dmin * m.astype(np.float32)).reshape(-1, 8, 1)
    qs = b[:, 16:144].reshape(-1, 4, 32)                               # 4 chunks of 64 values: low nibbles, then high
    q = np.stack([qs & 0xF, qs >> 4], axis=2).reshape(-1, 8, 32).astype(np.float32)
    return (dl * q - ml).reshape(-1)


def q6_k(raw):
    """Q6_K: 210-byte blocks of 256: ql[128] (low 4 bits), qh[64] (high 2 bits), 16 int8 scales, d (fp16)"""
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 210)
    ql, qh = b[:, 0:128].reshape(-1, 2, 64), b[:, 128:192].reshape(-1, 2, 32)
    sc = b[:, 192:208].view(np.int8).astype(np.float32)                # [blocks, 16]
    d = _f16(b[:, 208:210].copy()).reshape(-1, 1)
    q = np.empty((b.shape[0], 2, 4, 32), np.int8)
    q[:, :, 0] = (ql[:, :, :32] & 0xF) | (((qh >> 0) & 3) << 4)
    q[:, :, 1] = (ql[:, :, 32:] & 0xF) | (((qh >> 2) & 3) << 4)
    q[:, :, 2] = (ql[:, :, :32] >> 4) | (((qh >> 4) & 3) << 4)
    q[:, :, 3] = (ql[:, :, 32:] >> 4) | (((qh >> 6) & 3) << 4)
    q = q.astype(np.float32) - 32.0
    ds = (d * sc).reshape(-1, 2, 4, 2, 1)                              # scale index n * 8 + 2 qd + (l >= 16)
    return (ds * q.reshape(-1, 2, 4, 2, 16)).reshape(-1)


DECODERS = {'Q4_K': (256, 144, q4_k), 'Q6_K': (256, 210, q6_k)}


def dequant(raw, type_name, n_values):
    if type_name == 'F32': return np.asarray(raw, dtype=np.uint8).view(np.float32)[:n_values].copy()
    if type_name == 'F16': return np.asarray(raw, dtype=np.uint8).view(np.float16)[:n_values].astype(np.float32)
    if type_name not in DECODERS: raise ValueError(f'own_dequant: {type_name} (have {sorted(DECODERS)} + F32 / F16)')
    qk, bs, f = DECODERS[type_name]
    return f(np.asarray(raw, dtype=np.uint8)[:n_values // qk * bs])


def rows(t, r0, r1):
    """float32 [r1 - r0, K] of a ggufio.Tensor's rows r0 .. r1 - 1 (rows: every dim but the innermost, flattened)"""
    _, rb = t.rows(); K = t.shape[0]
    return dequant(t.data()[r0 * rb:r1 * rb], t.type, (r1 - r0) * K).reshape(r1 - r0, K)
