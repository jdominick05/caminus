"""Caminus's own weight layouts: GGUF blocks rearranged once at load into layouts our kernels read with aligned 16-byte
loads (docs/specs/spec-own-quant.md). Each keeps the matrix's byte count, so a token reads the same bytes.

  q6k_planes(raw, rows, K)   Q6_K (210-byte blocks: ql[128] | qh[64] | scales[16] | d) -> three planes over the whole
                             matrix: A = per row, per block, ql | qh (192 B); B = the 16 int8 scales; C = d (fp16).
                             Read by kernels/own_quant.h own_rows_q6k_planes.
  q6k_from_planes            the inverse (tests)."""
import numpy as np


def q6k_planes(raw, rows, K):
    """raw: uint8 [rows * K / 256 * 210] (GGUF Q6_K rows, contiguous) -> uint8, same size, the plane layout"""
    nb = K // 256
    if K % 256: raise ValueError(f'q6k_planes: K {K} is not a multiple of 256')
    b = np.asarray(raw, dtype=np.uint8).reshape(rows, nb, 210)
    return np.concatenate([b[:, :, :192].reshape(-1), b[:, :, 192:208].reshape(-1), b[:, :, 208:210].reshape(-1)])


def q6k_from_planes(pl, rows, K):
    nb = K // 256
    a = np.asarray(pl, dtype=np.uint8)
    n1, n2 = rows * nb * 192, rows * nb * 16
    out = np.empty((rows, nb, 210), dtype=np.uint8)
    out[:, :, :192] = a[:n1].reshape(rows, nb, 192)
    out[:, :, 192:208] = a[n1:n1 + n2].reshape(rows, nb, 16)
    out[:, :, 208:210] = a[n1 + n2:].reshape(rows, nb, 2)
    return out.reshape(-1)
