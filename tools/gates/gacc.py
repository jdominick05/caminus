"""docs/specs/spec-own-quant.md's G-acc: the pieces its harnesses share (tests/engine/test_k3_gguf_gsq.py (e),
benchmarks/q36_gacc.py, benchmarks/q35_mtp_gacc.py; the block cuts also tests/engine/test_own_q2_0.py).
Nothing here launches on import; the device helpers take the caller's Module (build/gg_mv_ref.hsaco, build/own_mv.hsaco)."""
import ctypes
import numpy as np

EXP_ROWS = 32        # rows sampled per routed expert
Q8_1_BYTES = 36      # block_q8_1: half2 ds (d, sum) + 32 int8


def block_cols(w, ts, b0, b1):
    """block columns [b0, b1) of every row of raw rows w [..., row bytes] (numpy or torch; ts bytes per block): a K cut"""
    sh = tuple(w.shape)
    v = w.reshape(*sh[:-1], sh[-1] // ts, ts)[..., b0:b1, :].reshape(*sh[:-1], (b1 - b0) * ts)
    return v.contiguous() if hasattr(v, 'contiguous') else np.ascontiguousarray(v)


def kcols(raw, ranges, bs, ts):
    """the blocks holding K values `ranges` [(k0, k1), ...] (in order) of each row of raw [r, row bytes] (numpy; bs values
    and ts bytes per block): a card's K shard, as K3 cuts it"""
    return np.ascontiguousarray(np.concatenate([block_cols(raw, ts, k0 // bs, k1 // bs) for k0, k1 in ranges], 1))


def ref_q8_1(gg, x32):
    """x fp32 [n] (cuda) -> q8_1 blocks with gg_mv_ref's quantizer (module gg). It writes a block for every launched lane
    (ng * 8 blocks), so the buffer holds them all and the first ceil(n / 32) are returned"""
    import torch
    n = x32.numel(); nb = (n + 31) // 32; ng = (nb * 32 + 255) // 256
    y = torch.empty(ng * 8 * Q8_1_BYTES, dtype=torch.uint8, device=x32.device)
    gg.gg_ref_quantize_q8_1((ng,), (256,), x32.contiguous(), y, ctypes.c_int(n))
    return y[:nb * Q8_1_BYTES]


def own_read(rd, w, sink, nbytes=None):
    """the plain read of the first nbytes (default all) of w (uint8, cuda, 16-byte aligned) with own_mv's own_read
    (module rd): the bandwidth ceiling a matvec over the same bytes is compared with"""
    n16 = (w.numel() if nbytes is None else nbytes) // 16
    rd.own_read((min(4096, (n16 + 255) // 256),), (256,), w, ctypes.c_long(n16), sink)


def expert_sets(data, ids, X, per_slot, rows=EXP_ROWS, ranges=None, bs=None, ts=None):
    """the row sets of one routed-expert tensor: for each expert some recorded vector routed to (ids [v, k_used]),
    `rows` of its rows (seeded by the expert id) with the vectors that routed to it.
    data [E, R, row bytes] raw; X [v, K] (per_slot False: every slot reads the vector, gate / up) or [v, k_used * kc]
    (per_slot True: slot j reads its own kc values, down); ranges (with bs, ts): cut each row to that K shard (kcols).
    Returns [(raw rows [rows, bytes] numpy, vectors [n, K] (X's type), global row ids)]"""
    E, Rw = data.shape[:2]; ku = ids.shape[1]; out = []
    for e in np.unique(ids):
        r = np.sort(np.random.default_rng(e).choice(Rw, rows, replace=False)); raw = np.asarray(data[e][r])
        hit = [(v, j) for v in range(ids.shape[0]) for j in range(ku) if ids[v, j] == e]
        if per_slot:
            kc = X.shape[1] // ku; Xe = X[[v for v, _ in hit]].reshape(len(hit), ku, kc)[np.arange(len(hit)), [j for _, j in hit]]
        else:
            Xe = X[[v for v, _ in hit]]
        out.append((kcols(raw, ranges, bs, ts) if ranges else np.ascontiguousarray(raw), Xe, e * Rw + r))
    return out


def rel_errors(Y, y):
    """G-acc's error of outputs y against the float64 reference Y, both [rows, vectors]: |y - Y| / rms over the rows of
    Y per vector, flattened vector by vector; vectors whose reference is all zero are left out and counted.
    Returns (errors, zero-reference count)"""
    Y = np.asarray(Y, np.float64); y = np.asarray(y, np.float64); errs, nz = [], 0
    for v in range(Y.shape[1]):
        sc = float(np.sqrt(np.mean(Y[:, v] ** 2)))
        if sc > 0: errs.append(np.abs(y[:, v] - Y[:, v]) / sc)
        else: nz += 1
    return (np.concatenate(errs) if errs else np.zeros(0)), nz


def summary(err):
    """mean and p99 of a G-acc error array"""
    return dict(mean=float(err.mean()), p99=float(np.percentile(err, 99)))
