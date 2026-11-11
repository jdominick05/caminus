"""Requantize a qwen3moe file's expert tensors from a Q8_0 donor into plain fixed-width integer codes: a study file
(docs/dev/row-scale-study.md) that is BASE with its expert tensors replaced, written the way tools/eval/gguf_splice.py
writes one (BASE's header and metadata bytes, a new tensor table, each tensor at the next multiple of the alignment).

  --bits 4   the Q4_0 container: 32 values per block, fp16 d + 16 bytes, w = d (q - 8), q 0 .. 15
  --bits 2   the Q2_0 container (ggml type 42 of the yardstick fork): 64 values per block, fp16 d + 16 bytes,
             w = d (q - 1), q 0 .. 3, so the levels are -1, 0, +1, +2 times d (not a symmetric set)
  --scale row    ONE d per output row: every block of a row holds the same d (a container of its own would store it
                 once per row and be 4.0 / 2.0 bits per weight; the model it decodes to is this file's)
  --scale block  one d per block (the container's own granularity): the control for what the row scale costs

The fit, the same for both scales: per group (a row, or a block) the d and codes that minimize
sum_j imp_j (w_j - d c_j)^2, with w the donor's decoded Q8_0 weights and imp_j the imatrix's value for the tensor's
expert and input column j (--imatrix, llama.cpp's legacy file: the mean squared activation; 1 without it). Candidates
d0 = +- max|w| / t over a grid of t, each refined by three rounds of (assign the nearest code, refit d by least
squares); the best d is rounded to fp16 and the codes are assigned again against the rounded value. A negative d is
allowed (the containers decode it; it mirrors the level set).

Also logged per tensor, not written: the fit's error with the level set shifted by half a step (codes +-0.5 .. : a
symmetric set without a zero), to tell what the container's level set costs.

  tools/guard.sh LOG 3600 python -I tools/eval/requant_rows.py BASE DONOR_Q8 OUT --bits 4|2 --scale row|block [--imatrix F]
The fit runs on one card (cuda:0), a chunk of experts at a time; its peak is projected and checked before the first
transfer. Prints the weighted and plain relative error per role and RESULT PASS | FAIL (exit 1) after reopening OUT:
metadata and every other tensor are BASE's bytes, every expert tensor has the container's type and size and, with
--scale row, one d per row."""
import argparse, collections, os, re, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
import torch
from caminus import ggufio
from caminus.device import capacity

ap = argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('donor'); ap.add_argument('out')
ap.add_argument('--bits', type=int, choices=(4, 2), required=True); ap.add_argument('--scale', choices=('row', 'block'), required=True)
ap.add_argument('--imatrix', default=None); ap.add_argument('--match', default=r'ffn_(gate|up|down)_exps')
ap.add_argument('--chunk', type=int, default=16, help='experts fitted at a time')
A = ap.parse_args()
CH = 64 << 20
TYPE_ID, BLK, LO, HI, ZERO = {4: (2, 32, -8, 7, 8), 2: (42, 64, -1, 2, 1)}[A.bits]   # (ggml type, values per block, code range, the stored code of 0)
BB = 18                                                                              # bytes per block, both containers
TS = {4: np.linspace(4.0, 10.0, 25), 2: np.linspace(0.5, 2.6, 22)}[A.bits]            # max|w| / d0 candidates
B, D = ggufio.GGUFFile(A.base), ggufio.GGUFFile(A.donor)
if os.path.exists(A.out): sys.exit(f'{A.out} exists')
assert ggufio.TYPES[TYPE_ID][1:] == (BLK, BB), ggufio.TYPES[TYPE_ID]
pat = re.compile(A.match); al = B.alignment
E = int(B.hp('expert_count'))


def imatrix(path):
    """llama.cpp's legacy imatrix file -> {tensor name: float32 [n]} (sums over the calls, as stored)"""
    out = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<i', f.read(4))[0]
        for _ in range(n):
            ln = struct.unpack('<i', f.read(4))[0]; name = f.read(ln).decode()
            _, nv = struct.unpack('<ii', f.read(8)); out[name] = np.frombuffer(f.read(4 * nv), dtype=np.float32)
    return out


IM = imatrix(A.imatrix) if A.imatrix else None
plan = []   # (tensor of BASE, new n_bytes or None)
for t in B.tensors:
    if not pat.search(t.name): plan.append((t, None)); continue
    d = D.T.get(t.name)
    if d is None or d.shape != t.shape or d.type != 'Q8_0': sys.exit(f'{t.name}: DONOR has {d!r} (a Q8_0 tensor of BASE\'s shape is needed)')
    K = int(t.shape[0]); R, _ = t.rows()
    if K % BLK or R % E: sys.exit(f'{t.name}: K {K} is not whole {BLK}-value blocks, or {R} rows are not {E} experts')
    if IM is not None and (t.name not in IM or len(IM[t.name]) != E * K or not np.isfinite(IM[t.name]).all()):
        sys.exit(f'{t.name}: the imatrix has no finite [{E} x {K}] entry')
    plan.append((t, R * (K // BLK) * BB))
if not any(nb for _, nb in plan): sys.exit(f'--match {A.match!r} names no tensor')
big = max(int(np.prod(t.shape)) // E for t, nb in plan if nb)     # weights of one expert of the largest tensor
# the fit's live tensors per chunk of experts: the donor's bytes (1.07 per weight), w, and at most ten fp32 temporaries
# of w's size inside a refinement round; + the HIP context and the allocator's slack
need_mib = A.chunk * big * (1.07 + 4 * 11) / 2 ** 20 + 768
capacity.check([0], [need_mib], f'requant_rows ({need_mib:.0f} MiB projected; lower --chunk)')
dev = torch.device('cuda:0')
torch.cuda.set_per_process_memory_fraction(min(1.0, (need_mib - 256) * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory), 0)
print(torch.cuda.get_device_name(0), f'{A.bits}-bit codes, one d per {A.scale}; projected {need_mib:.0f} MiB', flush=True)


def fit(w, imp, off):
    """w [e, r, nb, BLK] fp32, imp [e, 1, nb, BLK] -> (d [e, r, nb or 1, 1] already rounded to fp16, integer codes c
    (w ~ d (c + off)) as fp32, the weighted error per group); off 0 = the container's level set, 0.5 = the shifted one"""
    dims = (2, 3) if A.scale == 'row' else (3,)
    amax = w.abs().amax(dims, keepdim=True)
    best_e = torch.full_like(amax, float('inf')); best_d = torch.zeros_like(amax)

    def assign(d):
        return (torch.where(d != 0, w / d, torch.zeros_like(w)) - off).round_().clamp_(LO, HI).add_(off)
    for t in TS:
        for sg in (1.0, -1.0):
            d = amax * (sg / float(t))
            for _ in range(3):
                c = assign(d); ic = imp * c
                num = (ic * w).sum(dims, keepdim=True); den = (ic * c).sum(dims, keepdim=True)
                d = torch.where(den > 0, num / den, torch.zeros_like(num))
                e = (imp * (w - d * c).square_()).sum(dims, keepdim=True)
                up = e < best_e; best_e = torch.where(up, e, best_e); best_d = torch.where(up, d, best_d)
    d = best_d.half().float()
    c = assign(d)
    return d, c, (imp * (w - d * c).square_()).sum(dims, keepdim=True)


def pack(d, c):
    """-> uint8 [rows * nb * 18]: the container's blocks (c: the level-set codes of off 0)"""
    e, r, nb, _ = c.shape
    q = (c + ZERO).to(torch.uint8)
    if A.bits == 4: qs = q[..., :16] | (q[..., 16:] << 4)                                   # value j low nibble, j + 16 high
    else:
        q = q.view(e, r, nb, 16, 4); qs = q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)
    dh = d.expand(e, r, nb, 1).contiguous().half().view(e, r, nb, 1).view(torch.uint8).view(e, r, nb, 2)
    return torch.cat([dh, qs], -1).reshape(-1)


stats = collections.defaultdict(lambda: np.zeros(5))   # role -> sums of (weighted err, weighted energy, plain err, plain energy, shifted weighted err)


@torch.no_grad()
def requant(t):
    """the container bytes of BASE tensor t's requantized form, as numpy chunks in file order"""
    d8 = D.T[t.name]; K = int(t.shape[0]); R, rb = d8.rows(); ff = R // E; nb = K // BLK
    raw = np.asarray(d8.data()).reshape(E, ff * rb)
    im = IM[t.name].reshape(E, K) if IM is not None else np.ones((E, K), np.float32)
    role = re.sub(r'^blk\.\d+\.', '', t.name); st = np.zeros(5)
    for e0 in range(0, E, A.chunk):
        e1 = min(E, e0 + A.chunk)
        b = torch.from_numpy(np.ascontiguousarray(raw[e0:e1])).to(dev).view(e1 - e0, ff, K // 32, 34)
        w = (b[..., 0:2].contiguous().view(torch.float16).float() * b[..., 2:].contiguous().view(torch.int8).float()).view(e1 - e0, ff, nb, BLK)
        del b
        imp = torch.from_numpy(np.ascontiguousarray(im[e0:e1])).to(dev).view(e1 - e0, 1, nb, BLK)
        d, c, err = fit(w, imp, 0.0)
        st += [float(err.sum()), float((imp * w * w).sum()), float((w - d * c).square().sum()), float((w * w).sum()), 0.0]
        out = pack(d, c).cpu().numpy(); del d, c, err
        st[4] += float(fit(w, imp, 0.5)[2].sum())
        del w, imp
        yield out
    stats[role] += st
    print(f'  {t.name}: weighted relative error {st[0] / st[1]:.4e} (shifted level set {st[4] / st[1]:.4e}), plain {st[2] / st[3]:.4e}', flush=True)


table = bytearray(); off = 0; offs = []
for t, nb in plan:
    name = t.name.encode()
    table += struct.pack('<Q', len(name)) + name + struct.pack('<I', len(t.shape)) + struct.pack(f'<{len(t.shape)}Q', *t.shape)
    table += struct.pack('<IQ', TYPE_ID if nb else t.type_id, off); offs.append(off)
    off = (off + (nb or t.n_bytes) + al - 1) // al * al
head = bytes(B._mm[:B.table_start])
data_start = (len(head) + len(table) + al - 1) // al * al
with open(A.out, 'wb') as f:
    f.write(head); f.write(table); f.write(b'\0' * (data_start - len(head) - len(table)))
    for (t, nb), o in zip(plan, offs):
        assert f.tell() == data_start + o
        if nb is None:
            v = t.data()
            for i in range(0, t.n_bytes, CH): f.write(v[i:i + CH])
            n = t.n_bytes
        else:
            n = 0
            for chunk in requant(t): f.write(chunk.tobytes()); n += chunk.size
            assert n == nb, (t.name, n, nb)
        f.write(b'\0' * (-n % al))
    f.flush(); os.fsync(f.fileno())

# -- the checks ---------------------------------------------------------------------------------------------------------
O = ggufio.GGUFFile(A.out); bad = 0


def meta_eq(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) if isinstance(a[k], np.ndarray) else a[k] == b[k] for k in a)


if not meta_eq(O.meta, B.meta): bad += 1; print('FAIL the metadata differs from BASE')
if [t.name for t in O.tensors] != [t.name for t in B.tensors]: bad += 1; print('FAIL the tensor order differs from BASE')
nrep = 0
for o, (t, nb) in zip(O.tensors, plan):
    if nb is None:
        a, b = o.data(), t.data()
        if o.type_id != t.type_id or o.n_bytes != t.n_bytes or not all(np.array_equal(a[i:i + CH], b[i:i + CH]) for i in range(0, t.n_bytes, CH)):
            bad += 1; print(f'FAIL {o.name}: not BASE\'s bytes')
        continue
    nrep += 1
    if o.type_id != TYPE_ID or o.n_bytes != nb or o.shape != t.shape: bad += 1; print(f'FAIL {o.name}: {o.type} {o.n_bytes} bytes'); continue
    if A.scale == 'row':
        R, rb = o.rows()
        for r0 in range(0, R, 8192):   # (a slice of rows at a time: the d of every block of a row, against its first block's)
            dd = np.asarray(o.data()[r0 * rb:min(R, r0 + 8192) * rb]).reshape(-1, rb // BB, BB)[:, :, :2]
            if not (dd == dd[:, :1]).all(): bad += 1; print(f'FAIL {o.name}: a row with more than one d'); break
print(('PASS' if bad == 0 else f'FAIL {bad}') + f' {len(O.tensors)} tensors: {nrep} requantized to {ggufio.TYPES[TYPE_ID][0]} (one d per {A.scale}), '
      f'{len(O.tensors) - nrep} BASE\'s bytes, metadata BASE\'s; {os.path.getsize(A.out) / 1e9:.3f} GB (BASE {os.path.getsize(A.base) / 1e9:.3f} GB)')
for role, s in sorted(stats.items()):
    print(f'  {role}: weighted relative error {s[0] / s[1]:.4e} (shifted level set {s[4] / s[1]:.4e}), plain {s[2] / s[3]:.4e}')
print('RESULT', 'PASS' if bad == 0 else 'FAIL'); sys.exit(1 if bad else 0)
