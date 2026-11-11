"""float64 reference logits of a dense qwen35 GGUF (caminus.ref_q35: the host forward on Caminus's own reader, decoders and
tokenizer, every quantized weight decoded exactly, activations in float64) for the first N tokens of a fixed public text
(tools/reference/q3moe_ref.py's), teacher-forced in one layer-major pass. The reference of the dense line's model gates.
CPU only, a few minutes whatever N is (each matrix is decoded once); it reads the whole file: never run it beside a GPU
timing or gate (its file reads and page faults evict GPU queues).
  python -I tools/reference/q35_ref.py GGUF OUT.npz [N=128] [--trace] [--census]
OUT.npz: seq (int64 [N]), logits (float32 [N, vocab]); --trace adds resid (float32 [blocks, N, D]: the residual stream
after each block). --census prints the largest |value| of each matvec input kind over the run (what an fp16 copy of that
input must hold: fp16's largest finite value is 65504)."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, HERE)
import numpy as np
from caminus.bpe import BPE
from caminus.ref_q35 import Q35Ref
from q3moe_ref import TEXT

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a not in ('--trace', '--census')]; trace = [] if '--trace' in sys.argv else None
    path, out = args[0], args[1]; N = int(args[2]) if len(args) > 2 else 128
    m = Q35Ref(path); seq = BPE(m.F.meta).encode(TEXT)[:N]
    if len(seq) < N: raise SystemExit(f'the text has {len(seq)} tokens < {N}')
    if '--census' in sys.argv: m.census = {}
    t0 = time.time()
    lg = m.feed(seq, trace=trace, log=lambda il: il % 16 == 15 and print(f'block {il + 1} / {m.L}: {time.time() - t0:.0f} s', flush=True))
    extra = {} if trace is None else {'resid': np.stack(trace).astype(np.float32)}
    np.savez(out, seq=np.array(seq, dtype=np.int64), logits=lg.astype(np.float32), **extra)
    top = lg.argmax(axis=1)
    print(f'{N} positions in {time.time() - t0:.0f} s; next-token top-1 hits on the text itself: {int((top[:-1] == np.array(seq[1:])).sum())} / {N - 1}')
    for k, v in (m.census or {}).items(): print(f'census {k}: max |value| {v:.6g}')
    print(f'RESULT DONE: {out}')
