"""Caminus's own host-side decoders of GGUF blocks to float32 (numpy), written from the block formats
(docs/specs/spec-own-quant.md); gguf-py's dequantize is the test oracle (tests/io/test_own_dequant.py: bitwise equal).
The arithmetic follows the formats' definition in float32: Q4_K and Q5_K y = (d * sc) * q - (dmin * m); Q6_K
y = (d * sc) * q; the codebook formats (their tables: caminus.iq_tables) IQ4_XS y = (d * (ls - 32)) * kv[q];
IQ3_XXS y = ((d * (0.5 + s)) * 0.5) * grid * sign; IQ3_S y = (d * (1 + 2 s)) * grid * sign; IQ2_XXS
y = ((d * (0.5 + s)) * 0.25) * grid * sign; the plain block formats Q8_0 y = d * q (int8), Q4_0 y = d * (q - 8),
Q4_1 y = d * q + m, Q2_0 (ggml type 42) y = d * (q - 1).

  dequant(raw, type_name, n_values) -> float32 [n_values]   (raw: the blocks' bytes, uint8)
  rows(t, r0, r1)                    -> float32 [r1 - r0, K] (a caminus.ggufio.Tensor's rows r0 .. r1 - 1)"""
import numpy as np
from caminus import iq_tables


def _f16(b):
    return b.view(np.float16).astype(np.float32)


def _k_scales(s):
    """the 12 bytes of 6-bit scales / mins Q4_K and Q5_K share -> sc [blocks, 8], m [blocks, 8] (uint8)"""
    sc = np.empty((s.shape[0], 8), np.uint8); m = np.empty((s.shape[0], 8), np.uint8)
    sc[:, :4] = s[:, 0:4] & 63; m[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    m[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc, m


def q4_k(raw):
    """Q4_K: 144-byte blocks of 256: d, dmin (fp16), 12 bytes of 6-bit scales / mins, 128 bytes of nibbles"""
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 144)
    d, dmin = _f16(b[:, 0:2].copy()).reshape(-1, 1), _f16(b[:, 2:4].copy()).reshape(-1, 1)
    sc, m = _k_scales(b[:, 4:16])
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


def _signs7(idx):
    """the 8 sign bits of a 7-bit sign index: the index itself, and bit 7 = its parity (the number of minus signs in a
    group of 8 is even)"""
    idx = idx.astype(np.uint8)
    par = idx ^ (idx >> 4); par = par ^ (par >> 2); par = par ^ (par >> 1)
    return idx | ((par & 1) << 7)


def _pm(bits):
    """uint8 [...] -> float32 [..., 8]: +1 where the bit is clear, -1 where it is set (bit i = value i)"""
    return np.where((bits[..., None] >> np.arange(8, dtype=np.uint8)) & 1 == 0, np.float32(1), np.float32(-1))


def _nib(b):
    """uint8 [n, m] -> uint8 [n, 2 m]: each byte's low nibble, then its high one"""
    return np.stack([b & 0xF, b >> 4], axis=2).reshape(b.shape[0], -1)


def iq4_xs(raw):
    """IQ4_XS: 136-byte blocks of 256: d (fp16), scales_h (u16: 2 high bits per 32 values), scales_l[4] (a low nibble
    per 32 values), qs[128]: per 32 values 16 bytes, their low nibbles = values 0 .. 15, the high ones = 16 .. 31"""
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 136); n = b.shape[0]
    d = _f16(b[:, 0:2].copy()).reshape(n, 1)
    sh = b[:, 2:4].copy().view(np.uint16).reshape(n, 1)
    ls = _nib(b[:, 4:8]).astype(np.int32) | (((sh >> (2 * np.arange(8, dtype=np.uint16))) & 3).astype(np.int32) << 4)
    dl = (d * (ls - 32).astype(np.float32)).reshape(n, 8, 1)
    qs = b[:, 8:136].reshape(n, 8, 16)
    q = iq_tables.KVALUES_IQ4[np.concatenate([qs & 0xF, qs >> 4], axis=2)].astype(np.float32)
    return (dl * q).reshape(-1)


def _su32(b):
    """the per-32-values u32 of IQ3_XXS / IQ2_XXS [n, 8] -> (scale float32 [n, 8], signs float32 [n, 8, 32]): four 7-bit
    sign indices (one per 8 values) and a 4-bit scale in the top bits"""
    u = b.copy().view(np.uint32).reshape(-1, 8)
    idx = (u[:, :, None] >> np.array([0, 7, 14, 21], np.uint32)) & 0x7F
    return (u >> 28).astype(np.float32), _pm(_signs7(idx)).reshape(-1, 8, 32)


def iq3_xxs(raw):
    """IQ3_XXS: 98-byte blocks of 256: d (fp16), qs[64] (a grid index per 4 values), 8 u32 (signs + scale per 32)"""
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 98); n = b.shape[0]
    d = _f16(b[:, 0:2].copy()).reshape(n, 1)
    s, sg = _su32(b[:, 66:98])
    db = ((d * (np.float32(0.5) + s)) * np.float32(0.5)).reshape(n, 8, 1)
    g = iq_tables.GRID_IQ3_XXS[b[:, 2:66]].astype(np.float32).reshape(n, 8, 32)
    return (db * g * sg).reshape(-1)


def iq3_s(raw):
    """IQ3_S: 110-byte blocks of 256: d (fp16), qs[64] (the low 8 bits of a 9-bit grid index per 4 values), qh[8] (the
    9th bits, one byte per 32 values), signs[32] (a bit per value), scales[4] (a nibble per 32 values)"""
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 110); n = b.shape[0]
    d = _f16(b[:, 0:2].copy()).reshape(n, 1)
    db = (d * (1 + 2 * _nib(b[:, 106:110])).astype(np.float32)).reshape(n, 8, 1)
    hi = ((b[:, 66:74, None] >> np.arange(8, dtype=np.uint8)) & 1).astype(np.uint16)          # [n, 8, 8]
    g = iq_tables.GRID_IQ3_S[b[:, 2:66].reshape(n, 8, 8) | (hi << 8)].astype(np.float32).reshape(n, 8, 32)
    return (db * g * _pm(b[:, 74:106]).reshape(n, 8, 32)).reshape(-1)


def iq2_xxs(raw):
    """IQ2_XXS: 66-byte blocks of 256: d (fp16), then per 32 values two u32: four grid indices (a byte per 8 values),
    then signs + scale"""
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 66); n = b.shape[0]
    d = _f16(b[:, 0:2].copy()).reshape(n, 1)
    w = b[:, 2:66].reshape(n, 8, 8)
    s, sg = _su32(w[:, :, 4:8].reshape(n, 32))
    db = ((d * (np.float32(0.5) + s)) * np.float32(0.25)).reshape(n, 8, 1)
    g = iq_tables.GRID_IQ2_XXS[w[:, :, 0:4]].astype(np.float32).reshape(n, 8, 32)
    return (db * g * sg).reshape(-1)


def q5_k(raw):
    """Q5_K: 176-byte blocks of 256: d, dmin (fp16), 12 bytes of 6-bit scales / mins (as Q4_K), qh[32] (the fifth bits:
    bit j of qh[l] belongs to value j * 32 + l), qs[128] (the low 4 bits: 4 chunks of 64 values, low nibbles then high)"""
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 176)
    d, dmin = _f16(b[:, 0:2].copy()).reshape(-1, 1), _f16(b[:, 2:4].copy()).reshape(-1, 1)
    sc, m = _k_scales(b[:, 4:16])
    dl = (d * sc.astype(np.float32)).reshape(-1, 8, 1)
    ml = (dmin * m.astype(np.float32)).reshape(-1, 8, 1)
    qh = b[:, 16:48]                                                   # [blocks, 32]
    qs = b[:, 48:176].reshape(-1, 4, 32)
    lo = np.stack([qs & 0xF, qs >> 4], axis=2).reshape(-1, 8, 32)      # sub-block j = 2 * chunk + (high nibble)
    hi = (qh[:, None, :] >> np.arange(8, dtype=np.uint8)[None, :, None]) & 1
    q = (lo | (hi << 4)).astype(np.float32)
    return (dl * q - ml).reshape(-1)


def q8_0(raw):
    """Q8_0: 34-byte blocks of 32: d (fp16), 32 int8"""
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 34)
    d = _f16(b[:, 0:2].copy()).reshape(-1, 1)
    return (d * b[:, 2:34].view(np.int8).astype(np.float32)).reshape(-1)


def q4_0(raw):
    """Q4_0: 18-byte blocks of 32: d (fp16), 16 bytes of nibbles (values 0 .. 15: low nibbles, values 16 .. 31: high)"""
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 18)
    d = _f16(b[:, 0:2].copy()).reshape(-1, 1)
    qs = b[:, 2:18]
    q = np.concatenate([qs & 0xF, qs >> 4], axis=1).astype(np.int8) - np.int8(8)
    return (d * q.astype(np.float32)).reshape(-1)



def q4_1(raw):
    """Q4_1: 20-byte blocks of 32: d (fp16), m (fp16), 16 bytes as Q4_0's; value = d q + m, q in 0 .. 15"""
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 20); qs = b[:, 4:]
    q = np.concatenate([qs & 0xF, qs >> 4], axis=1).astype(np.float32)
    return (_f16(b[:, 0:2].copy()).reshape(-1, 1) * q + _f16(b[:, 2:4].copy()).reshape(-1, 1)).reshape(-1)


def q2_0(raw):
    """Q2_0 (ggml type 42): 18-byte blocks of 64: d (fp16), 16 bytes (value j in bits 2 (j % 4) of byte j / 4); the
    levels are -1, 0, +1, +2 times d"""
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 18); qs = b[:, 2:]
    q = ((qs[:, :, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(-1, 64).astype(np.int8) - np.int8(1)
    return (_f16(b[:, 0:2].copy()).reshape(-1, 1) * q.astype(np.float32)).reshape(-1)



DECODERS = {'Q4_K': (256, 144, q4_k), 'Q6_K': (256, 210, q6_k), 'Q5_K': (256, 176, q5_k),
            'IQ4_XS': (256, 136, iq4_xs), 'IQ3_XXS': (256, 98, iq3_xxs), 'IQ3_S': (256, 110, iq3_s),
            'IQ2_XXS': (256, 66, iq2_xxs), 'Q8_0': (32, 34, q8_0), 'Q4_0': (32, 18, q4_0), 'Q4_1': (32, 20, q4_1),
            'Q2_0': (64, 18, q2_0)}


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
