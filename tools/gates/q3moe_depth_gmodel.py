"""The G-model gate at depth for Qwen3-30B-A3B: the engine's batched prefill of positions [0, N - M), then M one-token K3
passes (teacher-forced), their logits against tools/reference/q3moe_ref32.py's fp32 rows for positions N - M .. N - 1, and
llama.cpp's own prefill-then-decode of the same tokens (build/gg_golden --lc 'tfx NAME TOKENS N-M M 2048 1 OUT' with
'kv f16' and 'onecard': its default KV type, one card). Gate: mean KLD(ref || ours) <= llama.cpp's and top-1 >= llama.cpp's.
  tools/guard.sh LOG 1800 python -I tools/gates/q3moe_depth_gmodel.py GGUF --ref REF32.npz [--llama LL.bin] [--tokens-out T.txt]
--tokens-out writes the reference's token ids (one line) for gg_golden's job file.
--decode-only: no prefill; all N positions as one-token K3 passes (a file whose expert formats the prefill has no
kernels for; about 6 ms per position on a 24 GB card).
--cards 2: the two-card engine (engine/q3moe_tp.py Q3MoETP on torch devices 0 and 1, its default split; CAMINUS_UPLOAD_LOG on
every two-card run). It has no batched prefill yet (a later change brings the two-card prefill), so it runs as
--decode-only: all N positions as one-token two-card passes, the last M scored. That is decode-then-decode, not the
one-card row's prefill-then-decode, against the same bound (llama.cpp's prefill-then-decode). The replicas (the residual
and every layer's router ids) are checked after every token; any difference, or a bailed pass, fails."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np


def lsm(v):
    v = v.astype(np.float64); v = v - v.max(); return v - np.log(np.exp(v).sum())


def score(ref, cand):
    kl = np.array([float(np.sum(np.exp(lsm(r)) * (lsm(r) - lsm(c)))) for r, c in zip(ref, cand)])
    return dict(mean=float(kl.mean()), median=float(np.median(kl)), p90=float(np.percentile(kl, 90)), max=float(kl.max()),
                top1=int(sum(np.argmax(r) == np.argmax(c) for r, c in zip(ref, cand))), n=len(kl))


def line(name, s):
    return (f'  {name:10s} mean KLD {s["mean"]:.3e} (median {s["median"]:.3e}, p90 {s["p90"]:.3e}, max {s["max"]:.3e}); '
            f'top-1 {s["top1"]}/{s["n"]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--ref', required=True)
    ap.add_argument('--llama', default=None); ap.add_argument('--tokens-out', default=None); ap.add_argument('--module', default=None)
    ap.add_argument('--decode-only', action='store_true'); ap.add_argument('--cards', type=int, default=1, choices=(1, 2))
    A = ap.parse_args()
    z = np.load(A.ref); seq, ref = [int(v) for v in z['seq']], z['logits']; N, M = len(seq), len(ref)
    if A.tokens_out:
        open(A.tokens_out, 'w').write(' '.join(map(str, seq)) + '\n'); print(f'tokens -> {A.tokens_out}'); sys.exit(0)
    import torch
    ok = True
    if A.cards == 2:
        from caminus.engine.q3moe_tp import Q3MoETP
        print(torch.cuda.get_device_name(0), '+', torch.cuda.get_device_name(1), f'N {N}, the last {M} positions', flush=True)
        eng = Q3MoETP(A.gguf, max_ctx=N + 8, module=A.module)
        print(f'two cards: expert rows {eng.exp_rows}, kv heads {eng.kvc}, head rows {eng.vh} / {eng.vocab - eng.vh}; decode only '
              f'(no two-card prefill yet): {N - M} one-token passes, then the {M} scored ones', flush=True)
        rep_bad, rows = [], []
        for i, t in enumerate(seq):
            if i < N - M: eng._run_tok(t, i, 0 if i == 0 else 1, True)   # (the argmax read-back only: no full logits copy)
            else: rows.append(eng.step(t).float().cpu().numpy())
            r = eng.check_replicas()
            if not all(r): rep_bad.append((i, r))
        ours = np.stack(rows)
        c0 = not rep_bad and not eng.bails
        print(('PASS' if c0 else 'FAIL') + f' replicas after each of {N} tokens (residual, every layer\'s router ids): '
              f'{N - len(rep_bad)} equal; bails {len(eng.bails)}' + (f'; first differences {rep_bad[:4]}' if rep_bad else ''), flush=True)
        ok = c0
    else:
        from caminus.engine.q3moe_one import Q3MoEOne
        from caminus.prefill.q3m import Q3MPrefill
        print(torch.cuda.get_device_name(0), f'N {N}, the last {M} positions', flush=True)
        eng = Q3MoEOne(A.gguf, max_ctx=N + 8, module=A.module)
        if A.decode_only:
            for t in seq[:N - M]: eng.step(t)
            print(f'decode only: {N - M} one-token passes, then the {M} scored ones', flush=True)
        else:
            pf = Q3MPrefill(eng, chunk=2048)
            pf.run(seq, 0, 0, N - M)
            eng.pos_host = N - M
        ours = np.stack([eng.step(t).float().cpu().numpy() for t in seq[N - M:]])
    so = score(ref, ours); print(line('caminus', so), flush=True)
    if A.llama:
        ll = np.fromfile(A.llama, dtype=np.float32).reshape(M, -1)
        sl = score(ref, ll); print(line('llama.cpp', sl), flush=True)
        c1, c2 = so['mean'] <= sl['mean'], so['top1'] >= sl['top1']
        print(('PASS' if c1 else 'FAIL') + f' depth {N}: mean KLD vs fp32 {so["mean"]:.3e} <= llama.cpp {sl["mean"]:.3e}')
        print(('PASS' if c2 else 'FAIL') + f' depth {N}: top-1 {so["top1"]} >= llama.cpp {sl["top1"]}')
        ok = ok and c1 and c2
    print('RESULT', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
