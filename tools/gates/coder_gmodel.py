"""The G-model gate for the Coder (docs/specs/spec-own-quant.md): FlashNextTP's logits and llama.cpp's against the float64
reference (tools/reference/q35_ref.py on the Coder: caminus.ref_coder), teacher-forced over the reference's tokens. Per
position KLD(ref || candidate), top-1 agreement and |logit difference| (tools/gates/q3moe_gmodel.py's score / line); the
gate: mean KLD <= llama.cpp's and top-1 >= llama.cpp's. The engine runs its defaults (both cards, the dense trunk below
2,051 positions, Q8 KV) and its replicas are checked after every token.
  tools/guard.sh LOG 900 python -I tools/gates/coder_gmodel.py $CODER --ref REF.npz --llama LLAMA   # both cards; the
      load dominates (tools/k3_smoke.py --model coder's guard is 600 s); 256 positions not yet timed
  python -I tools/gates/coder_gmodel.py --ref REF.npz --llama LLAMA --ref-only     # the reference against llama.cpp, no card
LLAMA: llama.cpp's one-token-decode logits rows of the same tokens: a gg_golden dump directory (index.tsv: stepN's
result_output, as the stored short dumps e2e_{code,prose,shell}_decode; tools/gates/common.py names their directory) or
a .bin of f32 [N, vocab] (build/gg_golden MODEL --lc JOBS with 'nctx N+64', 'kv f16', 'tfx ref TOKENS.txt 0 N 1 1 OUT':
two cards, the Coder does not fit one). Its KV type is what the dump used (the stored dumps: q8_0; --llama-kv says which).
--ref-only checks the reference's own reading of the model against llama.cpp: the reference passes when llama.cpp's mean KLD
against it is under --confirm (0.01; a gross misreading of a block costs 0.1 or more) and its top-1 agrees at 15 positions
of 16 or more. The line sits near llama.cpp's own noise on the prose and shell texts: its prefill-then-decode against its
one-token decode scores 9.9e-3 / 9.5e-3 there (code 1.1e-3; tools/gates/g4_floor.py short, n_ubatch 2048). That logits form cannot catch a subtle misreading (a router weighting, an expert count: see
tests/engine/test_coder_golden_cpu.py, which checks each block from llama.cpp's own input and the router weights by expert).
Prints PASS / FAIL and RESULT; exit 0 only on PASS."""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, HERE)
import numpy as np
from q3moe_gmodel import line, score


def llama_rows(path, n):
    """llama.cpp's logits rows 0 .. n - 1: from a gg_golden dump directory (stepI's result_output) or an f32 [N, vocab] .bin"""
    if os.path.isdir(path):
        got = {}
        for l in open(os.path.join(path, 'index.tsv')):
            r = l.rstrip('\n').split('\t')
            if r[2] == 'result_output' and int(r[0]) < n: got[int(r[0])] = r[8]
        if sorted(got) != list(range(n)): raise SystemExit(f'{path}: result_output for {len(got)} of the {n} steps')
        return np.stack([np.fromfile(os.path.join(path, got[i]), dtype=np.float32) for i in range(n)])
    a = np.fromfile(path, dtype=np.float32)
    if a.size % n: raise SystemExit(f'{path}: {a.size} values is not {n} rows')
    return a.reshape(n, -1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('gguf', nargs='?'); ap.add_argument('--ref', required=True)
    ap.add_argument('--llama', default=None); ap.add_argument('--llama-kv', default='q8_0', choices=('f16', 'q8_0'))
    ap.add_argument('--ref-only', action='store_true'); ap.add_argument('--confirm', type=float, default=0.01)
    ap.add_argument('--dump', default=None, help='save our logits rows (npy) for a bitwise comparison of two builds')
    A = ap.parse_args()
    z = np.load(A.ref); seq, ref = [int(v) for v in z['seq']], z['logits']
    sl = None
    if A.llama:
        ll = llama_rows(A.llama, len(seq))
        if ll.shape != ref.shape: raise SystemExit(f'llama.cpp rows {ll.shape} != the reference\'s {ref.shape}')
        sl = score(ref, ll); print(line('llama.cpp', sl) + f'  (its KV {A.llama_kv})', flush=True)
    if A.ref_only:
        if sl is None: ap.error('--ref-only needs --llama')
        c = sl['mean'] < A.confirm and sl['top1'] * 16 >= sl['n'] * 15
        print(('PASS' if c else 'FAIL') + f' the reference is confirmed by llama.cpp: mean KLD {sl["mean"]:.3e} < {A.confirm}, top-1 {sl["top1"]}/{sl["n"]} >= 15/16')
        print('RESULT', 'PASS' if c else 'FAIL'); sys.exit(0 if c else 1)
    if not A.gguf: ap.error('GGUF')
    import torch
    from caminus.engine.flashnext_tp import FlashNextTP
    print(torch.cuda.get_device_name(0), '+', torch.cuda.get_device_name(1), flush=True)
    eng = FlashNextTP(A.gguf, max_ctx=len(seq) + 8)
    print(f'module {eng.module}', flush=True)
    rows, bad = [], []
    for i in range(len(seq)):
        rows.append(eng.step(seq, i).float().cpu().numpy())
        if not all(eng.check_replicas()): bad.append(i)
    ours = np.stack(rows); ok = not bad
    print(('PASS' if ok else 'FAIL') + f' replicas equal on both cards after each of {len(seq)} tokens' + ('' if ok else f' (differ at {bad[:8]})'), flush=True)
    so = score(ref, ours); print(line('caminus', so), flush=True)
    if A.dump: np.save(A.dump, ours)
    if sl is not None:
        c1 = so['mean'] <= sl['mean']; c2 = so['top1'] >= sl['top1']
        print(('PASS' if c1 else 'FAIL') + f' mean KLD vs float64: caminus {so["mean"]:.3e} <= llama.cpp {sl["mean"]:.3e}')
        print(('PASS' if c2 else 'FAIL') + f' top-1 agreement: caminus {so["top1"]} >= llama.cpp {sl["top1"]}')
        ok = ok and c1 and c2
    print('RESULT', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
