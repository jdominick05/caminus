"""Caminus's own weight layouts: GGUF blocks rearranged once at load into layouts our kernels read with aligned 16-byte
loads (docs/specs/spec-own-quant.md). Each keeps the matrix's byte count, so a token reads the same bytes.

  q6k_planes(raw, rows, K)   Q6_K (210-byte blocks: ql[128] | qh[64] | scales[16] | d) -> three planes over the whole
                             matrix: A = per row, per block, ql | qh (192 B); B = the 16 int8 scales, reordered [n][h][qd]
                             (byte n * 8 + 4 h + qd = scale n * 8 + h + 2 qd: a kernel lane's four scales are contiguous);
                             C = d (fp16). Read by kernels/own_quant.h own_rows_q6k_planes.
  q6k_from_planes            the inverse (tests).
  iq_records(raw, rows, K, fmt)   IQ3_XXS / IQ3_S / IQ2_XXS (98 / 110 / 66-byte blocks) -> plane A: a record per unit of 32
                             weights (12 / 13 / 8 B: its index bytes, sign bits and, IQ3_S, its qh byte); plane B: a header
                             per block (d; IQ3_S: d | the four scale bytes; IQ2_S: a 10-byte record, d); then IQ_PAD zero bytes (a kernel's 16- and
                             8-byte loads run past the last record and header). Read by kernels/own_quant_iq.h
                             own_rows_iqg_1rt.
  iq_from_records            the inverse (tests).
  q4_0_planes(raw, rows, K)  Q4_0 (18-byte blocks: d | 16 code bytes; 2-byte aligned) -> plane A = every block's 16 code
                             bytes, row-major (one aligned 16-byte load per block), plane B = every block's d (fp16).
                             Read by kernels/own_quant_q4_0.h own_rows_q4_0_1rt.
  q4_0_from_planes           the inverse (tests).
  rs_planes(raw, rows, K, fmt)    the row-scale study's form (docs/dev/row-scale-study.md): a Q4_0 / Q2_0 tensor whose
                             blocks all repeat their row's scale -> (codes, scales): the codes alone, K * bits / 8 bytes
                             per row (4 bits: Q4_0's 16 code bytes per 32 values as they are; 2 bits: each 16 values
                             as a dword with value 4 i + k at bits 8 k + 2 i), and one fp32 scale per row in an array
                             of its own. Fewer bytes than the container (no repeated scales). Read by
                             kernels/own_quant_rs.h own_rows_rs_1rt.
  rs_from_planes             the inverse (tests)."""
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


def q4_0_planes(raw, rows, K):
    """raw: uint8 [rows * K / 32 * 18] (GGUF Q4_0 rows, contiguous) -> uint8, same size, the plane layout"""
    if K % 32: raise ValueError(f'q4_0_planes: K {K} is not a multiple of 32')
    b = np.asarray(raw, dtype=np.uint8)
    if b.size != rows * (K // 32) * 18: raise ValueError(f'q4_0_planes: {b.size} bytes, expected {rows} x {K // 32} x 18')
    b = b.reshape(rows, K // 32, 18)
    return np.concatenate([b[:, :, 2:].reshape(-1), b[:, :, :2].reshape(-1)])


def q4_0_from_planes(pl, rows, K):
    nb = K // 32
    a = np.asarray(pl, dtype=np.uint8)
    out = np.empty((rows, nb, 18), dtype=np.uint8)
    out[:, :, 2:] = a[:rows * nb * 16].reshape(rows, nb, 16)
    out[:, :, :2] = a[rows * nb * 16:].reshape(rows, nb, 2)
    return out.reshape(-1)


IQ_PAD = 16
# format -> (block bytes, record bytes, header bytes)
IQ_REC = {'IQ3_XXS': (98, 12, 2), 'IQ3_S': (110, 13, 6), 'IQ2_XXS': (66, 8, 2), 'IQ2_S': (82, 10, 2), 'IQ2_XS': (74, 9, 2)}


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
    elif fmt == 'IQ2_S':   # a unit: its 4 index bytes | 4 sign bytes | qh byte | scale byte (kernels/own_quant_iq2s.h)
        A[:, :, :4] = b[:, 2:34].reshape(n, 8, 4); A[:, :, 4:8] = b[:, 34:66].reshape(n, 8, 4)
        A[:, :, 8] = b[:, 66:74]; A[:, :, 9] = b[:, 74:82]
    elif fmt == 'IQ2_XS':  # a unit: its 4 u16 (9-bit grid index | 7-bit sign code << 9) | its scale byte (two nibbles)
        A[:, :, :8] = b[:, 2:66].reshape(n, 8, 8); A[:, :, 8] = b[:, 66:74]
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
    elif fmt == 'IQ2_S':
        out[:, 2:34] = A[:, :, :4].reshape(n, 32); out[:, 34:66] = A[:, :, 4:8].reshape(n, 32)
        out[:, 66:74] = A[:, :, 8]; out[:, 74:82] = A[:, :, 9]
    elif fmt == 'IQ2_XS':
        out[:, 2:66] = A[:, :, :8].reshape(n, 64); out[:, 66:74] = A[:, :, 8]
    else:
        out[:, 2:66] = A.reshape(n, 64)
    return out.reshape(-1)


Q2K_RS, Q2K_HS = 10, 4   # Q2_K records: a unit's 8 code bytes + its 2 scale bytes; a block's header: d, dmin (fp16 each)


def q2k_records(raw, rows, K):
    """raw: uint8 [rows * K / 256 * 84] (GGUF Q2_K rows, contiguous) -> uint8, the record layout (kernels/own_quant_q2k.h):
    plane A = a 10-byte record per unit of 32 values (its 32 2-bit codes packed four per byte, value 4 k + j at bits 2 j of
    byte k, then the block's scale bytes 2 ib and 2 ib + 1: scale nibble | min nibble << 4, one per 16 values), plane B = a
    4-byte header per block (d, dmin), then 16 bytes of padding. In the GGUF block a unit's codes are one bit pair of 32
    bytes that three other units share: read in place, a unit loads 4x its bytes"""
    if K % 256: raise ValueError(f'q2k_records: K {K} is not a multiple of 256')
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 84)
    n = b.shape[0]
    if n != rows * (K // 256): raise ValueError(f'q2k_records: {n} blocks, expected {rows} x {K // 256}')
    out = np.zeros(n * 8 * Q2K_RS + n * Q2K_HS + IQ_PAD, dtype=np.uint8)
    A = out[:n * 8 * Q2K_RS].reshape(n, 8, Q2K_RS); B = out[n * 8 * Q2K_RS:n * 8 * Q2K_RS + n * Q2K_HS].reshape(n, Q2K_HS)
    qs = b[:, 16:80].reshape(n, 2, 32)                                    # [n, half, byte]
    c = (qs[:, :, None, :] >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)) & 3   # [n, half, shift, 32]
    c = c.reshape(n, 8, 8, 4)                                             # unit ib = half * 4 + shift; byte k: values 4k .. 4k+3
    A[:, :, :8] = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
    A[:, :, 8] = b[:, 0:16:2]; A[:, :, 9] = b[:, 1:16:2]
    B[:, :] = b[:, 80:84]
    return out


def q2k_from_records(rec, rows, K):
    n = rows * (K // 256)
    a = np.asarray(rec, dtype=np.uint8)
    A = a[:n * 8 * Q2K_RS].reshape(n, 8, Q2K_RS); B = a[n * 8 * Q2K_RS:n * 8 * Q2K_RS + n * Q2K_HS].reshape(n, Q2K_HS)
    c = (A[:, :, :8, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3     # [n, unit, byte, j]
    c = c.reshape(n, 2, 4, 32)                                            # [n, half, shift, value]
    qs = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).reshape(n, 64)
    out = np.empty((n, 84), dtype=np.uint8)
    out[:, 0:16:2] = A[:, :, 8]; out[:, 1:16:2] = A[:, :, 9]; out[:, 16:80] = qs; out[:, 80:84] = B
    return out.reshape(-1)


IQ1M_RS, IQ1M_HS = 8, 2   # IQ1_M records: a unit's 4 index bytes, 2 qh bytes, its 2 scales; a block's header: d (fp16)


def iq1m_records(raw, rows, K):
    """raw: uint8 [rows * K / 256 * 56] (GGUF IQ1_M rows) -> uint8, the record layout (kernels/own_quant_iq1m.h): plane A
    = an 8-byte record per unit of 32 values (qs[4 ib .. 4 ib + 3], qh[2 ib], qh[2 ib + 1], the unit's two 3-bit scales
    s0 | s1 << 3, a zero byte), plane B = the block's fp16 d (reassembled from the four scale words' top nibbles), then 16
    bytes of padding"""
    if K % 256: raise ValueError(f'iq1m_records: K {K} is not a multiple of 256')
    b = np.asarray(raw, dtype=np.uint8).reshape(-1, 56)
    n = b.shape[0]
    if n != rows * (K // 256): raise ValueError(f'iq1m_records: {n} blocks, expected {rows} x {K // 256}')
    out = np.zeros(n * 8 * IQ1M_RS + n * IQ1M_HS + IQ_PAD, dtype=np.uint8)
    A = out[:n * 8 * IQ1M_RS].reshape(n, 8, IQ1M_RS); B = out[n * 8 * IQ1M_RS:n * 8 * IQ1M_RS + n * IQ1M_HS].reshape(n, IQ1M_HS)
    s16 = b[:, 48:56].copy().view(np.uint16).reshape(n, 4)
    sc = ((s16[:, :, None] >> np.array([0, 3, 6, 9], dtype=np.uint16)) & 7).reshape(n, 16).astype(np.uint8)
    A[:, :, 0:4] = b[:, 0:32].reshape(n, 8, 4); A[:, :, 4:6] = b[:, 32:48].reshape(n, 8, 2)
    A[:, :, 6] = sc[:, 0::2] | (sc[:, 1::2] << 3)
    d = ((s16[:, 0] >> 12) | ((s16[:, 1] >> 8) & 0x00F0) | ((s16[:, 2] >> 4) & 0x0F00) | (s16[:, 3] & 0xF000)).astype(np.uint16)
    B[:, :] = d.view(np.uint8).reshape(n, 2)
    return out


# container type -> (bits per code, values per block, block bytes): d (fp16) then the codes
RS_CONT = {'Q4_0': (4, 32, 18), 'Q2_0': (2, 64, 18)}
_RS_SH = np.array([0, 2, 4, 6], dtype=np.uint8)
# a container byte's four 2-bit values, one per byte of a dword (value k in byte k)
_RS_SPREAD = np.array([(v & 3) | ((v >> 2) & 3) << 8 | ((v >> 4) & 3) << 16 | ((v >> 6) & 3) << 24 for v in range(256)], dtype='<u4')


def rs_planes(raw, rows, K, fmt):
    """raw: uint8 [rows * K / block values * 18] (GGUF Q4_0 / Q2_0 rows, contiguous) -> (codes uint8 [rows * K * bits / 8],
    scales float32 [rows]). ValueError for a row with more than one scale: such a tensor has no row-scale form"""
    bits, bv, bs = RS_CONT[fmt]
    if K % 256: raise ValueError(f'rs_planes: K {K} is not a multiple of 256')
    b = np.asarray(raw, dtype=np.uint8)
    if b.size != rows * (K // bv) * bs: raise ValueError(f'rs_planes: {b.size} bytes, expected {rows} x {K // bv} x {bs}')
    b = b.reshape(rows, K // bv, bs)
    d = b[:, :, :2]
    if not (d == d[:, :1]).all(): raise ValueError(f'rs_planes: a {fmt} row with more than one scale (not a row-scale tensor)')
    scales = np.ascontiguousarray(d[:, 0]).view(np.float16).astype(np.float32).reshape(rows)
    if bits == 4: return np.ascontiguousarray(b[:, :, 2:]).reshape(-1), scales
    # 2 bits: the container's byte B holds values 4 B .. 4 B + 3 at bits 0, 2, 4, 6; the kernel's dword of 16 values
    # holds value 4 i + k at bits 8 k + 2 i, so (dword >> 2 i) & 0x03030303 is values 4 i .. 4 i + 3 as four bytes
    # (by table: container byte i of a dword spreads its four values over the dword's four bytes, shifted left by 2 i)
    out = np.empty((rows, K // 16), dtype='<u4')
    for r0 in range(0, rows, 16384):   # (a slice of rows at a time: the table's dwords are four times the codes)
        q = _RS_SPREAD[np.ascontiguousarray(b[r0:r0 + 16384, :, 2:]).reshape(-1, K // 16, 4)]
        out[r0:r0 + 16384] = q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)
    return out.view(np.uint8).reshape(-1), scales


def rs_from_planes(codes, scales, rows, K, fmt):
    bits, bv, bs = RS_CONT[fmt]
    out = np.empty((rows, K // bv, bs), dtype=np.uint8)
    out[:, :, :2] = np.asarray(scales, dtype=np.float32).astype(np.float16).reshape(rows, 1, 1).view(np.uint8).reshape(rows, 1, 2)
    a = np.asarray(codes, dtype=np.uint8)
    if bits == 4: out[:, :, 2:] = a.reshape(rows, K // bv, 16)
    else:
        k = a.reshape(rows, K // 16, 4)
        c = (k[..., None, :] >> _RS_SH[:, None]) & 3                       # [rows, dwords, i, k] = value 4 i + k
        out[:, :, 2:] = (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).reshape(rows, K // bv, 16)
    return out.reshape(-1)
