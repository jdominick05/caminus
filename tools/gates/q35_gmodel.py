"""The G-model gate for the dense qwen35 line on own code (docs/specs/spec-own-quant.md): Dense35Own's logits and llama.cpp's
against the float64 reference (tools/reference/q35_ref.py), teacher-forced over the reference's tokens. Per position
KLD(ref || candidate), top-1 agreement and |logit difference|; the gate: mean KLD <= llama.cpp's and top-1 >= llama.cpp's.
  tools/guard.sh LOG 1800 python -I tools/gates/q35_gmodel.py GGUF --ref REF.npz [--llama LLAMA.bin] [--module M]
  python -I tools/gates/q35_gmodel.py --ref REF.npz --llama LLAMA.bin --ref-only     # the reference against llama.cpp, no card
LLAMA.bin: llama.cpp's one-token-decode logits rows of the same tokens, f32 [N, vocab] (build/gg_golden MODEL --lc JOBS with
the lines 'nctx N+64', 'kv f16', 'onecard', 'tfx ref TOKENS.txt 0 N 1 1 OUT'). --ref-only is the check that can catch a
wrong reading of the model in the reference itself (the engine and the reference share that reading): the reference
is confirmed when llama.cpp's mean KLD against it is under --confirm (0.01: llama.cpp's quantized activations cost
about 1e-3, a wrong reading of a block costs 0.1 or more) and its top-1 agrees at 15 positions of 16 or more. Prints PASS / FAIL and RESULT; exit 0 only on PASS."""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, HERE)
import numpy as np
from q3moe_gmodel import line, score

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('gguf', nargs='?'); ap.add_argument('--ref', required=True)
    ap.add_argument('--llama', default=None); ap.add_argument('--module', default=None); ap.add_argument('--ref-only', action='store_true')
    ap.add_argument('--confirm', type=float, default=0.01)
    A = ap.parse_args()
    z = np.load(A.ref); seq, ref = [int(v) for v in z['seq']], z['logits']
    sl = None
    if A.llama:
        ll = np.fromfile(A.llama, dtype=np.float32).reshape(len(seq), -1)
        sl = score(ref, ll); print(line('llama.cpp', sl), flush=True)
    if A.ref_only:
        if sl is None: ap.error('--ref-only needs --llama')
        c = sl['mean'] < A.confirm and sl['top1'] * 16 >= sl['n'] * 15
        print(('PASS' if c else 'FAIL') + f' the reference is confirmed by llama.cpp: mean KLD {sl["mean"]:.3e} < {A.confirm}, top-1 {sl["top1"]}/{sl["n"]} >= 15/16')
        print('RESULT', 'PASS' if c else 'FAIL'); sys.exit(0 if c else 1)
    if not A.gguf: ap.error('GGUF')
    import torch
    from caminus.engine.q35_dense import Dense35Own
    print(torch.cuda.get_device_name(0), flush=True)
    eng = Dense35Own(A.gguf, max_ctx=len(seq) + 8, module=A.module)   # (its own VRAM check; this script keeps nothing else on the card)
    ours = np.stack([eng.step(t).float().cpu().numpy() for t in seq])
    so = score(ref, ours); print(line('caminus', so), flush=True)
    ok = True
    if sl is not None:
        c1 = so['mean'] <= sl['mean']; c2 = so['top1'] >= sl['top1']
        print(('PASS' if c1 else 'FAIL') + f' mean KLD vs float64: caminus {so["mean"]:.3e} <= llama.cpp {sl["mean"]:.3e}')
        print(('PASS' if c2 else 'FAIL') + f' top-1 agreement: caminus {so["top1"]} >= llama.cpp {sl["top1"]}')
        ok = c1 and c2
    print('RESULT', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
