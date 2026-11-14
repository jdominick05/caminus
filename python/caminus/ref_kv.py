"""CPU reference for the Coder's Q8 KV row bytes, not a runtime cache backend.

The source contract is op_kv_append in kernels/k3_gguf_ops.h and the Q8
append in kernels/prefill_attn.hip: each row is K then V, each flattened
head-major vector split into 32-value blocks of little-endian fp16 scale
and 32 signed bytes. No head padding or rotation is performed here.

Encode uses fp32 amax / 127, its fp32 reciprocal, fp32 multiplication and
roundf's ties away from zero. Codes use the unrounded scale; only the stored
scale is rounded to half (nearest, ties to even). Nonzero blocks require a
finite nonzero fp32 scale/reciprocal and finite stored half scale. A nonzero
fp32 scale may round to stored half zero: codes survive, decoded values vanish.
Other inputs are refused: the device source does not establish safe handling
of nonfinite values, fp32 scale underflow or half overflow. Half subnormals are modeled
as IEEE values here; compiled device rounding/flush behavior remains unverified.

Row counts describe the supplied state view. Allocation padding, speculative
rows and cache persistence versions belong to callers, not this wire codec.
CPU results follow source operations, not compiled-device division/reciprocal
instructions; they do not establish GPU parity, attention quality or model support.
"""
from dataclasses import dataclass
from operator import index

import numpy as np


def _count(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a nonnegative integer')
    try:
        value = index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be a nonnegative integer') from exc
    if value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


@dataclass(frozen=True)
class Q8KVLayout:
    """No padding: [K blocks for all heads | V blocks for all heads]."""
    kv_heads: int
    head_dim: int

    def __post_init__(self):
        heads = _count(self.kv_heads, 'kv_heads')
        dim = _count(self.head_dim, 'head_dim')
        if heads == 0 or dim == 0 or dim % 32:
            raise ValueError('positive heads and head_dim divisible by 32 required')
        object.__setattr__(self, 'kv_heads', heads)
        object.__setattr__(self, 'head_dim', dim)

    @property
    def values(self):
        return self.kv_heads * self.head_dim

    @property
    def row_bytes(self):
        return 2 * (self.values // 32) * 34

    def cache_bytes(self, rows):
        """Exact physical bytes for this many exposed rows (no allocation)."""
        return _count(rows, 'rows') * self.row_bytes


def encode_q8_kv(k, v, layout):
    """Pack float32 arrays [rows, kv_heads, head_dim] to uint8 [rows, bytes]."""
    for name, x in (('K', k), ('V', v)):
        if not isinstance(x, np.ndarray) or x.dtype != np.dtype(np.float32):
            raise ValueError(f'{name} must be a float32 numpy array')
        if x.ndim != 3 or x.shape[1:] != (layout.kv_heads, layout.head_dim):
            raise ValueError(f'{name} shape must be [rows, kv_heads, head_dim]')
        if not np.isfinite(x).all():
            raise ValueError(f'{name} contains nonfinite values')
    if k.shape != v.shape:
        raise ValueError('K and V shapes must match')
    rows = k.shape[0]
    x = np.concatenate((k.reshape(rows, layout.values),
                        v.reshape(rows, layout.values)), axis=1).reshape(-1, 32)
    amax = np.max(np.abs(x), axis=1)
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        d = amax / np.float32(127)
        inv = np.zeros_like(d)
        np.divide(np.float32(1), d, out=inv, where=d != 0)
        half = d.astype('<f2')
    nonzero = amax != 0
    if np.any(nonzero & ((d == 0) | ~np.isfinite(d) | ~np.isfinite(inv)
                         | ~np.isfinite(half))):
        raise ValueError('nonzero block scale is outside the reference encode domain')
    scaled = x * inv[:, None]  # fp32 product, before roundf
    # Promote only this already-rounded product: adding 0.5 in fp32 would
    # incorrectly round the float immediately below 0.5 up to one.
    rounded = np.copysign(np.floor(np.abs(scaled.astype(np.float64)) + 0.5), scaled)
    if np.any(np.abs(rounded) > 127):
        raise ValueError('quantized code outside [-127, 127]')
    blocks = np.empty((x.shape[0], 34), dtype=np.uint8)
    blocks[:, :2] = half.view(np.uint8).reshape(-1, 2)
    blocks[:, 2:] = rounded.astype(np.int8).view(np.uint8)
    return blocks.reshape(rows, layout.row_bytes)


def decode_q8_kv(raw, layout):
    """Decode literal finite Q8_0 blocks, including negative scales / -128 codes.

    This domain is wider than encode's: arbitrary finite half scales are valid
    for byte interpretation. Return K and V float32 arrays in input head order.
    """
    if not isinstance(raw, np.ndarray) or raw.dtype != np.dtype(np.uint8):
        raise ValueError('raw must be a uint8 numpy array')
    if raw.ndim != 2 or raw.shape[1] != layout.row_bytes:
        raise ValueError('raw shape must be [rows, row_bytes]')
    blocks = np.ascontiguousarray(raw).reshape(-1, 34)
    d = blocks[:, :2].copy().view('<f2').reshape(-1).astype(np.float32)
    if not np.isfinite(d).all():
        raise ValueError('nonfinite stored scale')
    values = blocks[:, 2:].view(np.int8).astype(np.float32) * d[:, None]
    values = values.reshape(raw.shape[0], 2, layout.kv_heads, layout.head_dim)
    return values[:, 0], values[:, 1]
