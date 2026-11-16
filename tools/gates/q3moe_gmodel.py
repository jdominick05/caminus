"""The G-model gate for Qwen3-30B-A3B (docs/specs/spec-own-quant.md): the engine's logits and llama.cpp's against the float64
reference (tools/reference/q3moe_ref.py), teacher-forced over the reference's tokens. Per position KLD(ref || candidate),
top-1 agreement and |logit difference|; the gate: mean KLD <= llama.cpp's and top-1 >= llama.cpp's.
  tools/guard.sh LOG 900 python -I tools/gates/q3moe_gmodel.py GGUF --ref REF.npz [--llama LLAMA.bin] [--module M] [--cards 2]
--cards 2: the two-card engine (engine/q3moe_tp.py Q3MoETP on torch devices 0 and 1, its default split; CAMINUS_UPLOAD_LOG on
every two-card run), the same scores against the same bound (llama.cpp's ONE-card decode), and the replicas checked after
every token: the residual and every layer's router ids the same bits on both cards; any difference, or a bailed pass, fails.
LLAMA.bin: llama.cpp's one-token-decode logits rows of the same tokens, f32 [N, vocab] (build/gg_golden MODEL --lc JOBS with
the lines 'nctx N+64', 'kv f16', 'onecard', 'tfx ref TOKENS.txt 0 N 1 1 OUT': its default KV type, one card). Without it only the
engine's numbers are printed. Prints PASS / FAIL and RESULT; exit 0 only on PASS (or with no llama.cpp numbers).
--kv q8: the engine's KV cache as Q8_0 rows (Q3MoEOne(q8=True); with --cards 2 Q3MoETP(q8=True), each card its kv heads'
rows; the --against reference is then a --cards 2 F16 run's saved logits); the same scores and the
same rule against the --llama rows given (llama.cpp's F16-KV rows are the stricter bound; rows made with 'kv q8_0' in the job
file the like-for-like one: --llama-kv names which, for the printout). --save-logits PATH writes this run's scored logits
(f32 [rows, vocab]); --against PATH (another run's saved logits: the F16 cache's) gates KLD(that || this) and top-1 agreement
between the two runs (tools/gates/kv_against.py): against llama.cpp's own F16-KV -> q8_0-KV cost on the same tokens
(--against-llama ITS_F16KV.bin with --llama ITS_Q8_0KV.bin --llama-kv q8_0; then the --llama rule above is the
like-for-like one) or explicit bars (--against-kld MEAN --against-top1 PERCENT); --against without a bound is refused.
--yarn: the long-context mode (YaRN x4): REF made by tools/reference/q3moe_ref.py --yarn 4:32768 (it carries a yarn entry;
the mismatch either way is refused), the engine on its YaRN modules, LLAMA.bin from gg_golden with the job line
'yarn 4 32768' (llama.cpp's --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768) as the bound."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_against
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
    ap.add_argument('--cards', type=int, default=1, choices=(1, 2))
    ap.add_argument('--kv', default='f16', choices=('f16', 'q8')); ap.add_argument('--llama-kv', default='f16', choices=('f16', 'q8_0'))
    ap.add_argument('--yarn', action='store_true', help='the long-context mode (YaRN): the reference must be a YaRN one (its yarn '
                    'entry: tools/reference/q3moe_ref*.py --yarn), the engine runs its YaRN modules (rope=YarnMode)')
    kv_against.add_args(ap)
    A = ap.parse_args(); kv_against.check_args(ap, A)
    z = np.load(A.ref); seq, ref = [int(v) for v in z['seq']], z['logits']
    zy = tuple(float(v) for v in z['yarn']) if 'yarn' in z.files else None
    if bool(zy) != A.yarn:   # (a YaRN reference against a native engine, or the reverse, measures the RoPE change, not the engine)
        raise SystemExit(f'--ref {A.ref}: ' + (f'a YaRN reference ({zy}): pass --yarn' if zy else 'a native reference: --yarn needs one made with --yarn'))
    from caminus.rope import YarnMode
    rope = YarnMode(zy[0], int(zy[1])) if zy else None
    import torch
    ok = True
    if A.cards == 2:
        from caminus.engine.q3moe_tp import Q3MoETP
        print(torch.cuda.get_device_name(0), '+', torch.cuda.get_device_name(1), flush=True)
        eng = Q3MoETP(A.gguf, max_ctx=len(seq) + 8, module=A.module, q8=A.kv == 'q8', rope=rope)
        print(f'two cards: module {eng.module}, KV {getattr(eng, "kv_format_id", "f16")}, expert rows {eng.exp_rows}, kv heads {eng.kvc}, '
              f'head rows {eng.vh} / {eng.vocab - eng.vh}, grids {eng.per_wgp_c}, splits {eng.splits}', flush=True)
        rows, rep_bad = [], []
        for i, t in enumerate(seq):
            rows.append(eng.step(t).float().cpu().numpy())
            r = eng.check_replicas()
            if not all(r): rep_bad.append((i, r))
        ours = np.stack(rows)
        c0 = not rep_bad and not eng.bails
        print(('PASS' if c0 else 'FAIL') + f' replicas after each of {len(seq)} tokens (residual, every layer\'s router ids): '
              f'{len(seq) - len(rep_bad)} equal; bails {len(eng.bails)}' + (f'; first differences {rep_bad[:4]}' if rep_bad else ''), flush=True)
        ok = c0
    else:
        from caminus.engine.q3moe_one import Q3MoEOne
        print(torch.cuda.get_device_name(0), flush=True)
        eng = Q3MoEOne(A.gguf, max_ctx=len(seq) + 8, module=A.module, q8=A.kv == 'q8', rope=rope)
        print(f'one card: module {eng.module}, KV {getattr(eng, "kv_format_id", "f16")}', flush=True)
        ours = np.stack([eng.step(t).float().cpu().numpy() for t in seq])
    so = score(ref, ours); print(line('caminus' if A.kv == 'f16' else 'caminus q8', so), flush=True)
    if A.save_logits:
        np.ascontiguousarray(ours, dtype=np.float32).tofile(A.save_logits); print(f'logits -> {A.save_logits} ({ours.shape[0]} rows)', flush=True)
    if A.llama:
        ll = np.fromfile(A.llama, dtype=np.float32).reshape(len(seq), -1)
        sl = score(ref, ll); print(line('llama.cpp' if A.llama_kv == 'f16' else 'llama q8_0', sl) + f'  (its KV {A.llama_kv})', flush=True)
        c1 = so['mean'] <= sl['mean']; c2 = so['top1'] >= sl['top1']
        print(('PASS' if c1 else 'FAIL') + f' mean KLD vs float64: caminus {so["mean"]:.3e} <= llama.cpp {sl["mean"]:.3e}')
        print(('PASS' if c2 else 'FAIL') + f' top-1 agreement: caminus {so["top1"]} >= llama.cpp {sl["top1"]}')
        ok = ok and c1 and c2
    if A.against:   # (this run against another run of the same rows: the Q8 cache against the F16 cache, gated)
        n = ours.shape[0]
        lf = kv_against.load_rows(A.against_llama, n, '--against-llama') if A.against_llama else None
        lq = kv_against.load_rows(A.llama, n, '--llama') if A.against_llama else None
        c3, lines = kv_against.gate(score, ours, kv_against.load_rows(A.against, n, '--against'), lf, lq, A.against_kld, A.against_top1)
        if lf is not None: print(line('llama F16', score(ref, lf)) + '  (llama.cpp F16-KV against the reference: the stricter bound, printed)', flush=True)
        for l in lines: print(l, flush=True)
        ok = ok and c3
    print('RESULT', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
