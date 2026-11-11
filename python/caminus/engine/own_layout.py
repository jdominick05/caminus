"""Caminus's own weight layouts: GGUF blocks rearranged once at load into layouts our kernels read with aligned 16-byte
loads (docs/specs/spec-own-quant.md). Each keeps the matrix's byte count, so a token reads the same bytes.

  q6k_planes(raw, rows, K)   Q6_K (210-byte blocks: ql[128] | qh[64] | scales[16] | d) -> three planes over the whole
                             matrix: A = per row, per block, ql | qh (192 B); B = the 16 int8 scales, reordered [n][h][qd]
                             (byte n * 8 + 4 h + qd = scale n * 8 + h + 2 qd: a kernel lane's four scales are contiguous);
                             C = d (fp16). Read by kernels/own_quant.h own_rows_q6k_planes.
  q6k_from_planes            the inverse (tests).
  iq_records(raw, rows, K, fmt)   IQ3_XXS / IQ3_S / IQ2_XXS (98 / 110 / 66-byte blocks) -> plane A: a record per unit of 32
                             weights (12 / 13 / 8 B: its index bytes, sign bits and, IQ3_S, its qh byte); plane B: a header
                             per block (d; IQ3_S: d | the four scale bytes); then IQ_PAD zero bytes (a kernel's 16- and
                             8-byte loads run past the last record and header). Read by kernels/own_quant_iq.h
                             own_rows_iqg_1rt.
  iq_from_records            the inverse (tests)."""
import numpy as np

# B plane order: position p = n * 8 + 4 h + qd holds GGUF scale Q6K_SC[p] = n * 8 + h + 2 qd
Q6K_SC = np.array([n * 8 + h + 2 * qd for n in range(2) for h in range(2) for qd in range(4)])


def q6k_planes(raw, rows, K):
    """raw: uint8 [rows * K / 256 * 210] (GGUF Q6_K rows, contiguous) -> uint8, same size, the plane layout"""
    nb = K // 256
    if K % 256: raise ValueError(f'q6k_planes: K {K} is not a multiple of 256')
    b = np.asarray(raw, dtype=np.uint8).reshape(rows, nb, 210)
    return np.concatenate([b[:, :, :192].reshape(-1), b[:, :, 192:208][:, :, Q6K_SC].reshape(-1), b[:, :, 208:210].reshape(-1)])


def q6k_from_planes(pl, rows, K):
    nb = K // 256
    a = np.asarray(pl, dtype=np.uint8)
    n1, n2 = rows * nb * 192, rows * nb * 16
    out = np.empty((rows, nb, 210), dtype=np.uint8)
    out[:, :, :192] = a[:n1].reshape(rows, nb, 192)
    out[:, :, 192 + Q6K_SC] = a[n1:n1 + n2].reshape(rows, nb, 16)
    out[:, :, 208:210] = a[n1 + n2:].reshape(rows, nb, 2)
    return out.reshape(-1)


IQ_PAD = 16
# format -> (block bytes, record bytes, header bytes)
IQ_REC = {'IQ3_XXS': (98, 12, 2), 'IQ3_S': (110, 13, 6), 'IQ2_XXS': (66, 8, 2)}


def iq_records(raw, rows, K, fmt):
    """raw: uint8 [rows * K / 256 * block bytes] (GGUF rows, contiguous) -> uint8, the record layout (the size + IQ_PAD)"""
    bs, rs, hs = IQ_REC[fmt]
    if K % 256: raise ValueError(f'iq_records: K {K} is not a multiple of 256')
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, bs)
    n = b.shape[0]
    if n != rows * (K // 256): raise ValueError(f'iq_records: {n} blocks, expected {rows} x {K // 256}')
    out = np.zeros(n * bs + IQ_PAD, dtype=np.uint8)
    A = out[:n * 8 * rs].reshape(n, 8, rs); B = out[n * 8 * rs:n * bs].reshape(n, hs)
    B[:, :2] = b[:, :2]
    if fmt == 'IQ3_XXS':
        A[:, :, :8] = b[:, 2:66].reshape(n, 8, 8); A[:, :, 8:] = b[:, 66:98].reshape(n, 8, 4)
    elif fmt == 'IQ3_S':
        A[:, :, :8] = b[:, 2:66].reshape(n, 8, 8); A[:, :, 8:12] = b[:, 74:106].reshape(n, 8, 4); A[:, :, 12] = b[:, 66:74]
        B[:, 2:] = b[:, 106:110]
    else:
        A[:, :, :] = b[:, 2:66].reshape(n, 8, 8)
    return out


def iq_from_records(rec, rows, K, fmt):
    bs, rs, hs = IQ_REC[fmt]
    n = rows * (K // 256)
    a = np.asarray(rec, dtype=np.uint8)
    A = a[:n * 8 * rs].reshape(n, 8, rs); B = a[n * 8 * rs:n * bs].reshape(n, hs)
    out = np.empty((n, bs), dtype=np.uint8)
    out[:, :2] = B[:, :2]
    if fmt == 'IQ3_XXS':
        out[:, 2:66] = A[:, :, :8].reshape(n, 64); out[:, 66:98] = A[:, :, 8:].reshape(n, 32)
    elif fmt == 'IQ3_S':
        out[:, 2:66] = A[:, :, :8].reshape(n, 64); out[:, 74:106] = A[:, :, 8:12].reshape(n, 32); out[:, 66:74] = A[:, :, 12]
        out[:, 106:110] = B[:, 2:]
    else:
        out[:, 2:66] = A.reshape(n, 64)
    return out.reshape(-1)
