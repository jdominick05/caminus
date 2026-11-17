"""The G-model gate at depth for Qwen3-30B-A3B: the engine's batched prefill of positions [0, N - M), then M one-token K3
passes (teacher-forced), their logits against tools/reference/q3moe_ref32.py's fp32 rows for positions N - M .. N - 1, and
llama.cpp's own prefill-then-decode of the same tokens (build/gg_golden --lc 'tfx NAME TOKENS N-M M 2048 1 OUT' with
'kv f16' and 'onecard': its default KV type, one card). Gate: mean KLD(ref || ours) <= llama.cpp's and top-1 >= llama.cpp's.
  tools/guard.sh LOG 1800 python -I tools/gates/q3moe_depth_gmodel.py GGUF --ref REF32.npz [--llama LL.bin] [--tokens-out T.txt]
--tokens-out writes the reference's token ids (one line) for gg_golden's job file.
--decode-only: no prefill; all N positions as one-token K3 passes (a file whose expert formats the prefill has no
kernels for; about 6 ms per position on a 24 GB card).
--cards 2: the two-card engine (engine/q3moe_tp.py Q3MoETP on torch devices 0 and 1, its default split; CAMINUS_UPLOAD_LOG on
every two-card run) with its batched prefill (prefill/q3m_tp.py Q3MPrefillTP, chunk 2048, its default micro-chunks and
overlap): positions [0, N - M) prefilled on both cards, then the M scored one-token two-card passes, as the one-card row.
With --decode-only: all N positions as one-token two-card passes (decode-then-decode, against the same bound). The replicas
are checked after the prefill (its residual rows and routing ids) and after every token (the residual and every layer's
router ids); any difference, or a bailed pass, fails.
--kv q8: the engine's KV cache as Q8_0 rows (Q3MoEOne(q8=True), its prefill Q3MPrefill's Q8 path; with --cards 2 Q3MoETP(q8=True)
and Q3MPrefillTP's Q8 path, the --against reference a --cards 2 F16 run's saved logits); the same scores and the
same rule against the --llama rows given (llama.cpp's F16-KV rows are the stricter bound; rows made with 'kv q8_0' in the job
file the like-for-like one: --llama-kv names which, for the printout). --save-logits PATH writes this run's scored logits
(f32 [rows, vocab]); --against PATH (another run's saved logits: the F16 cache's) gates KLD(that || this) and top-1 agreement
between the two runs (tools/gates/kv_against.py): against llama.cpp's own F16-KV -> q8_0-KV cost on the same tokens
(--against-llama ITS_F16KV.bin with --llama ITS_Q8_0KV.bin --llama-kv q8_0; then the --llama rule above is the
like-for-like one) or explicit bars (--against-kld MEAN --against-top1 PERCENT); --against without a bound is refused.
--yarn: the long-context mode (YaRN x4) at depth: REF made by tools/reference/q3moe_ref32.py --yarn 4:32768 (with --resume /
--budget-s past one window), the engine and its prefill on the YaRN builds, LLAMA.bin from gg_golden with 'yarn 4 32768'
(its LOAD line must say 'RoPE yarn (freq_scale 0.25, orig ctx 32768)'). Under --yarn the relative rule gets an absolute bar
on llama.cpp's mean KLD against the YaRN reference (--llama-max-kld, default 1.15e-02: twice its native 5.73e-03 at
40,896) and an optional --native-ref check (tools/gates/yarn_bound.py)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv_against
import numpy as np
import yarn_bound


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
    ap.add_argument('--kv', default='f16', choices=('f16', 'q8')); ap.add_argument('--llama-kv', default='f16', choices=('f16', 'q8_0'))
    ap.add_argument('--yarn', action='store_true', help='the long-context mode (YaRN): the reference must be a YaRN one (its yarn '
                    'entry: tools/reference/q3moe_ref*.py --yarn), the engine runs its YaRN modules (rope=YarnMode)')
    kv_against.add_args(ap); yarn_bound.add_args(ap, 'depth')
    A = ap.parse_args(); kv_against.check_args(ap, A); yarn_bound.check_args(ap, A)
    z = np.load(A.ref); seq, ref = [int(v) for v in z['seq']], z['logits']
    zy = tuple(float(v) for v in z['yarn']) if 'yarn' in z.files else None
    if bool(zy) != A.yarn:   # (a YaRN reference against a native engine, or the reverse, measures the RoPE change, not the engine)
        raise SystemExit(f'--ref {A.ref}: ' + (f'a YaRN reference ({zy}): pass --yarn' if zy else 'a native reference: --yarn needs one made with --yarn'))
    from caminus.rope import YarnMode
    rope = YarnMode(zy[0], int(zy[1])) if zy else None; N, M = len(seq), len(ref)
    if A.tokens_out:
        open(A.tokens_out, 'w').write(' '.join(map(str, seq)) + '\n'); print(f'tokens -> {A.tokens_out}'); sys.exit(0)
    import torch
    ok = True
    if A.cards == 2:
        from caminus.engine.q3moe_tp import Q3MoETP
        from caminus.prefill import q3m_tp
        print(torch.cuda.get_device_name(0), '+', torch.cuda.get_device_name(1), f'N {N}, the last {M} positions', flush=True)
        extra = (0.0, 0.0) if A.decode_only else q3m_tp.need_tp_mib(q3m_tp.tp_shapes(A.gguf), 2048)
        eng = Q3MoETP(A.gguf, max_ctx=N + 8, module=A.module, vram_extra_mib=extra, q8=A.kv == 'q8', rope=rope)
        print(f'two cards: module {eng.module}, KV {getattr(eng, "kv_format_id", "f16")}', flush=True)
        rep_bad, rows = [], []
        if A.decode_only:
            print(f'two cards: expert rows {eng.exp_rows}, kv heads {eng.kvc}, head rows {eng.vh} / {eng.vocab - eng.vh}; decode only: '
                  f'{N - M} one-token passes, then the {M} scored ones', flush=True)
            for i, t in enumerate(seq[:N - M]):
                eng._run_tok(t, i, 0 if i == 0 else 1, True)   # (the argmax read-back only: no full logits copy)
                r = eng.check_replicas()
                if not all(r): rep_bad.append((i, r))
        else:
            pf = q3m_tp.Q3MPrefillTP(eng, chunk=2048)
            print(f'two cards: expert rows {eng.exp_rows}, kv heads {eng.kvc}, head rows {eng.vh} / {eng.vocab - eng.vh}; the two-card '
                  f'prefill of [0, {N - M}) (chunk 2048, micro {pf.micro}, {pf.mode}), then the {M} scored passes', flush=True)
            pf.run(seq, 0, 0, N - M)
            r = pf.check_replicas()
            if not all(r): rep_bad.append(('prefill', r))
            eng.pos_host = N - M
        for i, t in enumerate(seq[N - M:], N - M):
            rows.append(eng.step(t).float().cpu().numpy())
            r = eng.check_replicas()
            if not all(r): rep_bad.append((i, r))
        ours = np.stack(rows)
        c0 = not rep_bad and not eng.bails
        print(('PASS' if c0 else 'FAIL') + f' replicas after the prefill or each prompt token, and each scored token (residual, router ids): '
              f'{"none differ" if not rep_bad else "differ"}; bails {len(eng.bails)}' + (f'; first differences {rep_bad[:4]}' if rep_bad else ''), flush=True)
        ok = c0
    else:
        from caminus.engine.q3moe_one import Q3MoEOne
        from caminus.prefill.q3m import Q3MPrefill
        print(torch.cuda.get_device_name(0), f'N {N}, the last {M} positions', flush=True)
        eng = Q3MoEOne(A.gguf, max_ctx=N + 8, module=A.module, q8=A.kv == 'q8', rope=rope)
        print(f'one card: module {eng.module}, KV {getattr(eng, "kv_format_id", "f16")}', flush=True)
        if A.decode_only:
            for t in seq[:N - M]: eng.step(t)
            print(f'decode only: {N - M} one-token passes, then the {M} scored ones', flush=True)
        else:
            pf = Q3MPrefill(eng, chunk=2048)
            pf.run(seq, 0, 0, N - M)
            eng.pos_host = N - M
        ours = np.stack([eng.step(t).float().cpu().numpy() for t in seq[N - M:]])
    so = score(ref, ours); print(line('caminus' if A.kv == 'f16' else 'caminus q8', so), flush=True)
    if A.save_logits:
        np.ascontiguousarray(ours, dtype=np.float32).tofile(A.save_logits); print(f'logits -> {A.save_logits} ({ours.shape[0]} rows)', flush=True)
    if A.llama:
        ll = np.fromfile(A.llama, dtype=np.float32).reshape(M, -1)
        sl = score(ref, ll); print(line('llama.cpp' if A.llama_kv == 'f16' else 'llama q8_0', sl) + f'  (its KV {A.llama_kv})', flush=True)
        c1, c2 = so['mean'] <= sl['mean'], so['top1'] >= sl['top1']
        print(('PASS' if c1 else 'FAIL') + f' depth {N}: mean KLD vs fp32 {so["mean"]:.3e} <= llama.cpp {sl["mean"]:.3e}')
        print(('PASS' if c2 else 'FAIL') + f' depth {N}: top-1 {so["top1"]} >= llama.cpp {sl["top1"]}')
        ok = ok and c1 and c2
    if A.yarn:   # (the relative rule above loosens with a wrong reference or llama.cpp run: an absolute bar, the native check)
        nat = yarn_bound.load_native(A.native_ref, seq, M) if A.native_ref else None
        ll_, sl_ = (ll, sl) if A.llama else (None, None)
        c4, lines = yarn_bound.gate(score, 'depth', ours, so, ll_, sl_, nat, A.llama_max_kld, A.llama_log, f'd{N}')
        for l in lines: print(l, flush=True)
        ok = ok and c4
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
