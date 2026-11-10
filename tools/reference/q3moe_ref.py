"""float64 reference logits of a qwen3moe GGUF (caminus.ref_q3moe: the host forward on Caminus's own reader and decoders, every
quantized weight decoded exactly, activations in float64) for N tokens of a fixed public text, teacher-forced one position
at a time. The G-model gate's reference (tools/gates/q3moe_gmodel.py). CPU only, ~8 s per token on this machine's CPU: never
run it beside a GPU timing or gate (its file reads and page faults evict GPU queues).
  python -I tools/reference/q3moe_ref.py GGUF OUT.npz [N=128]     -> OUT.npz: seq (int64 [N]), logits (float32 [N, vocab])"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
from caminus import ggufio
from caminus.bpe import BPE
from caminus.ref_q3moe import Q3MoERef
TEXT = ("The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined a screw press with "
        "movable metal type, oil-based ink and a new casting process. Within fifty years presses were running in more than two "
        "hundred European cities, and the price of books fell sharply. Historians still debate how quickly literacy followed, "
        "but the spread of pamphlets during the Reformation shows how fast printed arguments could travel. In 1517 Martin "
        "Luther's theses were copied and reprinted across Germany within weeks, a speed that handwritten copies could never have "
        "matched. The same technology carried scientific results: Copernicus, Vesalius and later Newton reached readers")
if __name__ == '__main__':
    path, out = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    tk = BPE(ggufio.GGUFFile(path).meta); seq = tk.encode(TEXT)[:N]
    if len(seq) < N: raise SystemExit(f'the text has {len(seq)} tokens < {N}')
    m = Q3MoERef(path); lg = []; t0 = time.time()
    for i, t in enumerate(seq):
        lg.append(m.step(t))
        if i % 16 == 0: print(f'{i} / {N}: {time.time() - t0:.0f} s', flush=True)
    np.savez(out, seq=np.array(seq, dtype=np.int64), logits=np.stack(lg).astype(np.float32))
    print(f'RESULT DONE: {out}')
