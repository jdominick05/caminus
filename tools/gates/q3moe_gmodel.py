"""The G-model gate for Qwen3-30B-A3B (docs/specs/spec-own-quant.md): the engine's logits and llama.cpp's against the float64
reference (tools/reference/q3moe_ref.py), teacher-forced over the reference's tokens. Per position KLD(ref || candidate),
top-1 agreement and |logit difference|; the gate: mean KLD <= llama.cpp's and top-1 >= llama.cpp's.
  tools/guard.sh LOG 900 python -I tools/gates/q3moe_gmodel.py GGUF --ref REF.npz [--llama LLAMA.bin] [--module M]
LLAMA.bin: llama.cpp's one-token-decode logits rows of the same tokens, f32 [N, vocab] (build/gg_golden MODEL --lc JOBS with
the lines 'nctx N+64', 'kv f16', 'onecard', 'tfx ref TOKENS.txt 0 N 1 1 OUT': its default KV type, one card). Without it only the
engine's numbers are printed. Prints PASS / FAIL and RESULT; exit 0 only on PASS (or with no llama.cpp numbers)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np


def lsm(v):
    v = v.astype(np.float64); v = v - v.max(); return v - np.log(np.exp(v).sum())


def score(ref, cand):
    kl, dl, t1 = [], [], 0
    for r, c in zip(ref, cand):
        lr, lc = lsm(r), lsm(c)
        kl.append(float(np.sum(np.exp(lr) * (lr - lc)))); dl.append(float(np.abs(r.astype(np.float64) - c).max()))
        t1 += int(np.argmax(r) == np.argmax(c))
    kl = np.array(kl)
    return dict(mean=float(kl.mean()), median=float(np.median(kl)), p90=float(np.percentile(kl, 90)), max=float(kl.max()),
                top1=t1, n=len(kl), dlogit_max=float(max(dl)), dlogit_median=float(np.median(dl)))


def line(name, s):
    return (f'  {name:10s} mean KLD {s["mean"]:.3e} (median {s["median"]:.3e}, p90 {s["p90"]:.3e}, max {s["max"]:.3e}); '
            f'top-1 {s["top1"]}/{s["n"]}; |d logit| median {s["dlogit_median"]:.3e}, max {s["dlogit_max"]:.3e}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--ref', required=True)
    ap.add_argument('--llama', default=None); ap.add_argument('--module', default=None)
    A = ap.parse_args()
    z = np.load(A.ref); seq, ref = [int(v) for v in z['seq']], z['logits']
    import torch
    from caminus.engine.q3moe_one import Q3MoEOne
    print(torch.cuda.get_device_name(0), flush=True)
    eng = Q3MoEOne(A.gguf, max_ctx=len(seq) + 8, module=A.module)
    ours = np.stack([eng.step(t).float().cpu().numpy() for t in seq])
    so = score(ref, ours); print(line('caminus', so), flush=True)
    ok = True
    if A.llama:
        ll = np.fromfile(A.llama, dtype=np.float32).reshape(len(seq), -1)
        sl = score(ref, ll); print(line('llama.cpp', sl), flush=True)
        c1 = so['mean'] <= sl['mean']; c2 = so['top1'] >= sl['top1']
        print(('PASS' if c1 else 'FAIL') + f' mean KLD vs float64: caminus {so["mean"]:.3e} <= llama.cpp {sl["mean"]:.3e}')
        print(('PASS' if c2 else 'FAIL') + f' top-1 agreement: caminus {so["top1"]} >= llama.cpp {sl["top1"]}')
        ok = c1 and c2
    print('RESULT', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
