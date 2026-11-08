"""QSA indexer host helpers and numpy references (docs/specs/spec-qsa.md; device ops: kernels/k3_qsa_ops.h, builders:
caminus.engine.k3.Engine.idx_append / qsa_score / qsa_select).

- q8_0 rows exactly as llama.cpp's quantize_f32_q8_0_block (cpy-utils.cuh) and getrows' dequantize_q8_0, in numpy float32
  (IEEE, so bitwise the device's): 128 values -> 4 blocks of {half d; int8 qs[32]} = 136 bytes.
- requant_rows: the q8_0 bytes back from dequantized rows (llama.cpp's dumped indexer_k), checked exact.
- select_ref: the selection rule of OP_QSA_SELECT (all nv pools when nv <= 512, else the 512 largest scores with the lowest
  pool index first among equal ones), ascending.
- visible pools / tail cells of a position: nv = (pos + 1) // 4 (pools end at 4j + 3), n_tail = (pos + 1) % 4."""
import numpy as np

D, H, KPOOL, TOPK, ROWB, SELW = 128, 4, 4, 512, 136, 516
_BLK = np.dtype([('d', '<f2'), ('qs', 'i1', 32)])
assert _BLK.itemsize == 34


def nv_of(pos): return (pos + 1) // KPOOL


def ntail_of(pos): return (pos + 1) % KPOOL


def _roundf(v):
    """C roundf (half away from zero) on float32, exact: v - trunc(v) is exact in float"""
    t = np.trunc(v)
    return np.where(np.abs(v - t) >= np.float32(0.5), t + np.sign(v), t).astype(np.float32)


def q8_0_rows(x):
    """fp32 [n, 128] -> uint8 [n, 136] (4 block_q8_0 per row), quantize_f32_q8_0_block's arithmetic"""
    x = np.ascontiguousarray(x, np.float32).reshape(-1, D // 32, 32)
    amax = np.abs(x).max(-1)
    d = (amax / np.float32(127)).astype(np.float32)
    with np.errstate(divide='ignore'):
        idv = np.where(d != 0, np.float32(1) / np.where(d != 0, d, np.float32(1)), np.float32(0)).astype(np.float32)
    q = _roundf((x * idv[..., None]).astype(np.float32)).astype(np.int8)
    out = np.zeros(x.shape[:2], _BLK)
    out['d'] = d.astype(np.float16); out['qs'] = q
    return out.view(np.uint8).reshape(-1, ROWB)


def dq_rows(b):
    """uint8 [n, 136] -> fp32 [n, 128]: (float)q * (float)d"""
    blk = np.ascontiguousarray(b, np.uint8).reshape(-1, ROWB).view(_BLK).reshape(-1, D // 32)
    return (blk['qs'].astype(np.float32) * blk['d'].astype(np.float32)[..., None]).reshape(-1, D)


def requant_rows(v):
    """the q8_0 bytes of rows that are dequantized q8_0 (v = q * d exactly), e.g. llama.cpp's indexer_k: per block the
    largest |q| is 127, so d = amax / 127 in float32 is d exactly and round(v / d) gives q. Asserts the round trip"""
    v = np.ascontiguousarray(v, np.float32).reshape(-1, D)
    b = q8_0_rows(v)
    assert np.array_equal(dq_rows(b).view(np.uint32), v.view(np.uint32)), 'requant_rows: rows are not dequantized q8_0'
    return b


def key_of(s):
    """the scores' total order as uint64 keys (OP_QSA_SELECT's qsa_key)"""
    b = np.ascontiguousarray(s, np.float32).view(np.uint32).astype(np.uint64)
    return np.where(b & 0x80000000, (~b) & 0xFFFFFFFF, b | 0x80000000)


def select_ref(scores, nv):
    """(pool ids ascending, threshold key or 0): OP_QSA_SELECT's rule over scores[:nv]"""
    if nv <= TOPK: return np.arange(nv, dtype=np.int32), 0
    k = key_of(scores[:nv]).astype(np.int64)
    order = np.lexsort((np.arange(nv), -k))[:TOPK]
    return np.sort(order).astype(np.int32), int(k[order[-1]])


def tie_report(scores, nv, ours, theirs):
    """set comparison of two selections over scores[:nv]: (n differing, all differences at the threshold score?, ties at
    the threshold, picks with an exactly zero score, visible zero scores)"""
    a, b = set(int(i) for i in ours), set(int(i) for i in theirs)
    diff = sorted(a ^ b)
    s = np.asarray(scores[:nv], np.float32)
    if nv <= TOPK: thr = None
    else: thr = np.float32(s[np.lexsort((np.arange(nv), -key_of(s).astype(np.int64)))[TOPK - 1]])
    at_thr = all(thr is not None and s[i] == thr for i in diff)
    ties = int((s == thr).sum()) if thr is not None else 0
    return dict(ndiff=len(diff), diff_at_threshold=at_thr if diff else True, ties_at_threshold=ties,
                zero_picks=int(sum(1 for i in a if s[i] == 0)), zero_scores=int((s == 0).sum()), threshold=None if thr is None else float(thr))
