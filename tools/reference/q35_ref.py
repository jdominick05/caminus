"""float64 reference logits of a qwen35, qwen35moe or qwen4exp GGUF (caminus.ref_q35 / caminus.ref_coder for qwen4exp, the
Coder: the host forward on Caminus's own reader and decoders, every quantized weight decoded exactly, activations in
float64; the file's architecture picks the forward) for the first N tokens of a fixed public text (tools/reference/
q3moe_ref.py's, through the own tokenizer) or of a token file (--tokens: ids separated by whitespace, as gg_golden's
tokens.txt; the Coder's gate texts are the stored short llama.cpp dumps' e2e_{code,prose,shell}_decode/tokens.txt
(tools/gates/common.py names their directory), whose one-token-decode logits sit beside them), teacher-forced in one
layer-major pass. The reference of the model gates (tools/gates/q35_gmodel.py: the qwen35 line and the Coder). CPU only, a
few minutes whatever N is (each matrix is decoded once); it reads the whole file: never run it beside a GPU timing or gate
(its file reads and page faults evict GPU queues).
  python -I tools/reference/q35_ref.py GGUF OUT.npz [N=128] [--tokens FILE] [--threads T] [--trace] [--census]
OUT.npz: seq (int64 [N]), logits (float32 [N, vocab]); --trace adds resid (float32 [blocks, N, D]: the residual stream
after each block; the Coder's [blocks, N, 4, D]: its four streams), and for a MoE file (qwen35moe, qwen4exp) topk (int16
[blocks, N, experts used]: each block's chosen experts, in the router's order) and topw (float32, the same shape: their
weights). --census prints the largest |value| of each matvec input kind over the run (what an fp16 copy of that input must
hold: fp16's largest finite value is 65504). The Coder's reference covers N <= 2,051 (dense attention is its QSA there:
caminus.ref_coder); its float64 logits take 2 MB a position."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, HERE)
import numpy as np
from caminus.bpe import BPE
from caminus.ref_q35 import reference
from q3moe_ref import TEXT


def opt(args, name):
    """the value after --name in args (removed from args with it), or None"""
    if name not in args: return None
    i = args.index(name); v = args[i + 1]; del args[i:i + 2]
    return v


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a not in ('--trace', '--census')]; trace = [] if '--trace' in sys.argv else None
    tokfile, threads, yarn = opt(args, '--tokens'), opt(args, '--threads'), opt(args, '--yarn')
    path, out = args[0], args[1]; N = int(args[2]) if len(args) > 2 else 128
    m = reference(path, threads=int(threads) if threads else None, yarn=float(yarn) if yarn else None)
    seq = [int(v) for v in open(tokfile).read().split()][:N] if tokfile else BPE(m.F.meta).encode(TEXT)[:N]
    if len(seq) < N: raise SystemExit(f'the {"token file" if tokfile else "text"} has {len(seq)} tokens < {N}')
    if '--census' in sys.argv: m.census = {}
    t0 = time.time()
    routes = [] if trace is not None and m.moe else None
    lg = m.feed(seq, trace=trace, routes=routes, log=lambda il: il % 16 == 15 and print(f'block {il + 1} / {m.L}: {time.time() - t0:.0f} s', flush=True))
    extra = {} if trace is None else {'resid': np.stack(trace).astype(np.float32)}
    if routes: extra.update(topk=np.stack([r[0] for r in routes]).astype(np.int16), topw=np.stack([r[1] for r in routes]).astype(np.float32))
    np.savez(out, seq=np.array(seq, dtype=np.int64), logits=lg.astype(np.float32), **extra)
    top = lg.argmax(axis=1)
    print(f'{N} positions in {time.time() - t0:.0f} s; next-token top-1 hits on the text itself: {int((top[:-1] == np.array(seq[1:])).sum())} / {N - 1}')
    for k, v in (m.census or {}).items(): print(f'census {k}: max |value| {v:.6g}')
    print(f'RESULT DONE: {out}')
