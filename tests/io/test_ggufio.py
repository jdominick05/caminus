"""caminus.ggufio (our GGUF reader) against gguf-py (the oracle) on GGUF files: every metadata key and value, every
tensor's name, type, shape, absolute data offset and byte count, and the first / last bytes of each tensor's data; the
type table's block sizes against gguf-py's. Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/io/test_ggufio.py FILE.gguf [FILE.gguf ...]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
from caminus.config import ensure_gguf
ensure_gguf()
from gguf import GGUFReader, GGML_QUANT_SIZES, GGMLQuantizationType
from caminus import ggufio

ok = True
def check(c, what):
    global ok
    if not c: print('FAIL', what)
    ok &= bool(c)

for tid, (name, bs, ts) in ggufio.TYPES.items():
    try: qt = GGMLQuantizationType(tid)
    except ValueError: continue
    check(qt.name == name and tuple(GGML_QUANT_SIZES[qt]) == (bs, ts), f'type {tid}: ours {name} {bs}/{ts}, gguf-py {qt.name} {GGML_QUANT_SIZES[qt]}')
extra = {int(q): (q.name, *GGML_QUANT_SIZES[q]) for q in GGMLQuantizationType if int(q) not in ggufio.TYPES and q in GGML_QUANT_SIZES}
print('types gguf-py knows that ours does not:', extra)

for path in sys.argv[1:]:
    ours, ref = ggufio.GGUFFile(path), GGUFReader(path, 'r')
    rf = {f.name: f for f in ref.fields.values() if not f.name.startswith('GGUF.')}
    check(set(ours.meta) == set(rf), f'{path}: metadata keys differ: {set(ours.meta) ^ set(rf)}')
    nbad = 0
    for k, f in rf.items():
        v = f.contents()
        o = ours.meta.get(k)
        same = (np.array_equal(np.asarray(o), np.asarray(v)) if isinstance(o, np.ndarray) else
                (list(o) == list(v)) if isinstance(o, list) else (o == v or (isinstance(o, float) and np.float32(o) == np.float32(v))))
        if not same: nbad += 1; print('  meta mismatch', k, str(o)[:60], str(v)[:60])
    check(nbad == 0, f'{path}: {nbad} metadata values differ')
    check([t.name for t in ours.tensors] == [t.name for t in ref.tensors], f'{path}: tensor order / names differ')
    nbad = 0
    for a, b in zip(ours.tensors, ref.tensors):
        good = (a.type == b.tensor_type.name and list(a.shape) == [int(s) for s in b.shape] and a.offset == int(b.data_offset)
                and a.n_bytes == int(b.n_bytes))
        if good and a.n_bytes:
            d = a.data(); rb = np.asarray(b.data).reshape(-1).view(np.uint8)
            good = d[:64].tobytes() == rb[:64].tobytes() and d[-64:].tobytes() == rb[-64:].tobytes()
        if not good: nbad += 1; print('  tensor mismatch', a, b.name, b.tensor_type.name, list(b.shape), int(b.data_offset), int(b.n_bytes))
    check(nbad == 0, f'{path}: {nbad} of {len(ours.tensors)} tensors differ')
    print(f'{os.path.basename(path)}: {len(ours.meta)} keys, {len(ours.tensors)} tensors, arch {ours.arch}, data at {ours.data_start}')
    ours.close()
print('RESULT', 'PASS' if ok else 'FAIL')
