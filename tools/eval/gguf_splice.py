"""Write a GGUF that is BASE with some of its tensors replaced by DONOR's tensors of the same name and shape (another
quantization of the same weights): a study file that differs from BASE in those tensors only. The header and the
metadata are BASE's bytes, the tensor table and the data are written here (each tensor at the next multiple of the
file's alignment, zero padding between), so general.file_type keeps naming BASE's quantization. Read with
caminus.ggufio; nothing here decodes a block.

After writing, the output is opened again and checked: its metadata equals BASE's, every replaced tensor's type and
bytes are DONOR's, every other tensor's are BASE's. With DONOR = BASE the output must also be BASE byte for byte
(--self).
  python -I tools/eval/gguf_splice.py BASE DONOR OUT [--match 'ffn_(gate|up|down)_exps'] [--self]
Prints the per-type census of the replaced tensors and RESULT PASS | FAIL (exit 1)."""
import argparse, collections, hashlib, os, re, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
from caminus import ggufio
ap = argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('donor'); ap.add_argument('out')
ap.add_argument('--match', default=r'ffn_(gate|up|down)_exps', help='regex: the tensors taken from DONOR (re.search on the name)')
ap.add_argument('--self', action='store_true', help='also require OUT == BASE byte for byte (DONOR is BASE)')
A = ap.parse_args()
CH = 64 << 20
B, D = ggufio.GGUFFile(A.base), ggufio.GGUFFile(A.donor)
if os.path.exists(A.out): sys.exit(f'{A.out} exists')
pat = re.compile(A.match); al = B.alignment
src = []
for t in B.tensors:
    if pat.search(t.name):
        d = D.T.get(t.name)
        if d is None or d.shape != t.shape: sys.exit(f'{t.name}: DONOR has {d!r}, BASE {t!r}')
        src.append(d)
    else: src.append(t)
if not any(s is not t for s, t in zip(src, B.tensors)) and not A.self: sys.exit(f'--match {A.match!r} names no tensor')
table = bytearray(); off = 0; offs = []
for s in src:
    nb = s.name.encode()
    table += struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(s.shape)) + struct.pack(f'<{len(s.shape)}Q', *s.shape)
    table += struct.pack('<IQ', s.type_id, off); offs.append(off)
    off = (off + s.n_bytes + al - 1) // al * al
head = bytes(B._mm[:B.table_start])
data_start = (len(head) + len(table) + al - 1) // al * al
with open(A.out, 'wb') as f:
    f.write(head); f.write(table); f.write(b'\0' * (data_start - len(head) - len(table)))
    for s, o in zip(src, offs):
        assert f.tell() == data_start + o
        v = s.data()
        for i in range(0, s.n_bytes, CH): f.write(v[i:i + CH])
        f.write(b'\0' * (-s.n_bytes % al))
    f.flush(); os.fsync(f.fileno())

# -- the checks ---------------------------------------------------------------------------------------------------------
O = ggufio.GGUFFile(A.out); bad = 0


def same(x, y):
    if x.n_bytes != y.n_bytes or x.type_id != y.type_id or x.shape != y.shape: return False
    a, b = x.data(), y.data()
    return all(np.array_equal(a[i:i + CH], b[i:i + CH]) for i in range(0, x.n_bytes, CH))


def meta_eq(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) if isinstance(a[k], np.ndarray) else a[k] == b[k] for k in a)


if not meta_eq(O.meta, B.meta): bad += 1; print('FAIL the metadata differs from BASE')
if [t.name for t in O.tensors] != [t.name for t in B.tensors]: bad += 1; print('FAIL the tensor order differs from BASE')
census = collections.Counter(); nrep = 0
for o, s, t in zip(O.tensors, src, B.tensors):
    if not same(o, s): bad += 1; print(f'FAIL {o.name}: not its source bytes')
    if s is not t:
        nrep += 1; census[(re.sub(r'^blk\.\d+\.', '', o.name), t.type, o.type)] += 1
print(('PASS' if bad == 0 else f'FAIL {bad}') + f' {len(O.tensors)} tensors: {nrep} from DONOR, {len(O.tensors) - nrep} from BASE, metadata BASE\'s; '
      f'{os.path.getsize(A.out) / 1e9:.3f} GB (BASE {os.path.getsize(A.base) / 1e9:.3f} GB)')
for (kind, was, now), n in sorted(census.items()): print(f'  {kind}: {n} tensors {was} -> {now}')
if A.self:
    def sha(p):
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            while True:
                b = f.read(CH)
                if not b: break
                h.update(b)
        return h.hexdigest()
    eq = os.path.getsize(A.out) == os.path.getsize(A.base) and sha(A.out) == sha(A.base)
    bad += not eq; print(('PASS' if eq else 'FAIL') + ' the output is BASE byte for byte')
print('RESULT', 'PASS' if bad == 0 else 'FAIL'); sys.exit(1 if bad else 0)
