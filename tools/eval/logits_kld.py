"""Distance of logits files to a reference file's, position by position: KLD(REF || ARM) (mean, median, 99th
percentile, max) and top-1 agreement. The files are tools/reference/q3moe_ref32.py's OUT.npz (seq, logits [M, vocab]),
computed over the same token sequence; a file with another sequence or shape is refused. CPU only.
  python -I tools/eval/logits_kld.py REF.npz NAME=ARM.npz [NAME=ARM.npz ...]"""
import sys
import numpy as np


def logp(z):
    z = z.astype(np.float64); z -= z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


R = np.load(sys.argv[1]); lr = logp(R['logits']); pr = np.exp(lr); tr = lr.argmax(-1)
print(f'reference {sys.argv[1]}: {lr.shape[0]} positions')
print(f'{"arm":12s} {"mean KLD":>10s} {"median":>10s} {"p99":>10s} {"max":>10s}  top-1 agreement')
for a in sys.argv[2:]:
    name, path = a.split('=', 1); X = np.load(path)
    if X['logits'].shape != R['logits'].shape or not np.array_equal(X['seq'], R['seq']): sys.exit(f'{path}: another sequence or shape')
    lx = logp(X['logits'])
    k = (pr * (lr - lx)).sum(-1); ag = int((lx.argmax(-1) == tr).sum())
    print(f'{name:12s} {k.mean():10.3e} {np.median(k):10.3e} {np.percentile(k, 99):10.3e} {k.max():10.3e}  {ag}/{len(k)} ({100 * ag / len(k):.1f} %)')
