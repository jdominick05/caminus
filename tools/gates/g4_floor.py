"""The G4 floor (docs/specs/spec-prefill-cache.md section 3, G4): llama.cpp's OWN prefill-then-decode against its own
one-token decode, on the same positions, at a given n_ubatch. The end-to-end gate S6 runs is our prefill(N) then K3
decode against llama.cpp's one-token decode; this measures what llama.cpp itself scores on that comparison, so the bar
is llama.cpp's own floor, not zero (MMQ never equals the decode matvec bitwise: spec S3 note).

  python -I tools/gates/g4_floor.py jobs OUT --ub N [--save-short 0|1] [--save-ref 0|1] [--control]
  python -I tools/gates/g4_floor.py depth-jobs OUT --depths 8192,32768 [--ub 2048]   (the depth windows, M5 S7)
  python -I tools/gates/g4_floor.py depth-jobs OUT --depths 65536 --ctx FILE --runs-1tok 3 --runs-pd 2 [--ub 2048] [--nctx N]
                                                       (the sweep's multi-run floors: every run's logits saved)
  build/gg_golden CODER --lc OUT/jobs_ubN.txt          (under tools/guard.sh: llama.cpp on both cards)
  python -I tools/gates/g4_floor.py cmp OUT [--json F]  (CPU only)
  python -I tools/gates/g4_floor.py depth-cmp OUT [--json F]  (CPU only)
  python -I tools/gates/g4_floor.py pool OUT.json --ref LABEL=BIN ... --pd LABEL=BIN ... [--expect REF:PD=MEAN ...]
  python -I tools/gates/g4_floor.py pd-jobs OUT --refs DIR [--p0 128] [--ub 512]   (any model's e2e refs: the pd protocol)
  python -I tools/gates/g4_floor.py pd-cmp OUT --refs DIR [--p0 128] [--json F]    (CPU only)
                                                       (CPU only: a floor pooled over run pairings, the sweep's method)

Protocols (tokens: the D-281 short texts, refs/d281-short/e2e_{prose,code,shell}_decode/tokens.txt; the 2K contexts:
qsa/lc/d1792/ctx.ids and the first 2304 tokens of qsa/lc/d8192/ctx.ids):
  short  for every position p in 1 .. 255 of each text: memory cleared, tokens [0, p) as one prompt pass (one ubatch at
         any n_ubatch >= 256), then token p alone; its logits vs the stored one-token decode dump's row p (refs/d281-short:
         llama.cpp main mode, n_ctx 512). 765 positions (position 0 has no prompt to prefill).
  2K     prefill N tokens (one llama_decode of N: N / ubatch ubatches), then the next 256 tokens one per llama_decode;
         those 256 rows vs the one-token decode of the same tokens from position 0 (tfx B = 1, same process). N = 1792
         (positions 1792 .. 2047: below 2051, the dense regime S6's milestone A covers) and N = 2048 (positions 2048 ..
         2303: one full chunk, then decode past 2051, where the indexer selects: S7's regime).
  depth  (M5 S7, the QSA prompt path) the same as 2K at N = 8192 and 32768 (positions N .. N + 255 of qsa/lc/d{N}/ctx.ids,
         decode past the prefill with the indexer selecting): prefill N at --ub (2048: our chunk), then 256 one-token
         decodes, vs the one-token decode of the same tokens from position 0 (tfx B = 1, same process: ~22-30 ms a token,
         ~15 min at 32K); depth-jobs also writes jobs_depth_ub512.txt, the same prefill-then-decode at ub 512 (FNV only), to
         set beside the stored qsa/lc/ref/tf{N}_r1 (the D-281 long reference: llama.cpp's default ub). depth-cmp: the floor
         (mean KLD, its 95 % bootstrap interval over positions (seeded, 10,000 resamples), top-1, the max position and the
         mean without it) per depth. No bitwise control here: llama.cpp past 2051 is not run-to-run deterministic (measured
         11-04: the stored tf8192 / tf32768 _r1 vs _r2 differ in every row's FNV and in 2 argmaxes each; our pd32768 r1 vs
         r2 at ub 2048 in every row, 0 argmaxes; pd8192 r1 == r2), so r1 vs r2 and ub 512 vs the stored reference are
         reported (rows differing by FNV and by argmax) beside the stored pair's own spread.
  Each prefill-then-decode run twice (r1 with logits, r2 hashes only): the run-to-run spread.
  pool   (the 262K sweep, user ruling 11-05: llama.cpp past ~8K is not run-to-run reproducible, so one run pairing is not
         the floor) every pairing of a one-token run (--ref) with a prefill-then-decode run (--pd) of the same window: the
         per-pairing stats (mean KLD, its 95 % bootstrap, top-1, max) and the POOLED floor: per position the mean KLD over
         all pairings (a vector of NPOS), its mean, and its 95 % bootstrap interval from boot() resampling POSITIONS (the
         pairings of one position are not independent samples: resampling the pairs x positions values would narrow the
         interval), top-1 the mean over pairings. The G4a bars at a pooled depth: mean KLD <= the pooled upper bound,
         top-1 >= the pooled top-1 - 1 point (the g4d rule on the pooled floor). BIN: gg_golden's OUT.bin (rows in
         OUT.tsv) or a qsax job's DIR/logits.bin (rows in DIR/steps.tsv); every run must hold the same (position, token)
         rows (control). --expect REF:PD=MEAN: a pairing whose mean must reproduce an earlier computation (control, to 1e-9).
         Run to run (FNV / argmax rows differing) among the --ref runs and among the --pd runs is reported.
  pd     (any model's e2e refs, e.g. $ARTIFACTS/q36/refs-gsq: the protocol of tests/prefill/test_q36_prefill.py e2e) per text of
         DIR/e2e_{prose,code,shell}_decode (llama.cpp's stored one-token decode dumps): memory cleared, tokens [0, P0) as ONE
         prompt pass, then P0 .. n - 1 one per llama_decode; those rows vs the stored dump's (control: the lc-mode one-token
         decode == the dump by FNV-1a, every row), twice (r1 with logits, r2 hashes only: run to run). The floor over the
         three texts: mean KLD, its 95 % bootstrap interval over positions, top-1; per text as well.
Controls (cmp): the lc-mode one-token decode of the short texts == the stored dumps (FNV-1a of every row); the N = 1792
run at ub 512 == qsa/lc/ref/tf1792_r1 (gg_golden --lc's tf job, the stored long-context reference, by FNV); the
one-token references at ub 2048 == at ub 512 (FNV); the short protocol at ub 2048 == at ub 512 (FNV).
KLD: floor.py's math (refs/d281-short/floor.py): float64, full vocabulary, KL(decode || other); top-1 = equal argmax."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ARTIFACTS, D281_SHORT, LC, opt, tokens   # noqa: E402

TEXTS = ('prose', 'code', 'shell')
NV = 248320
NPOS = 256


def tokfile(t):
    return os.path.join(D281_SHORT, f'e2e_{t}_decode', 'tokens.txt')


def ctx2304(out):
    """the first 2304 tokens of the 8K probe context (one full 2048 chunk + 256 decoded positions), written once"""
    p = os.path.join(out, 'ctx2304.ids')
    if not os.path.exists(p):
        ids = tokens(os.path.join(LC, 'd8192', 'ctx.ids'))[:2048 + NPOS]
        assert len(ids) == 2048 + NPOS
        open(p, 'w').write(' '.join(map(str, ids)) + '\n')
    return p


def cmd_jobs(a):
    out = a.pop(0); ub = opt(a, '--ub', 512, int); save_short = opt(a, '--save-short', 1, int); save_ref = opt(a, '--save-ref', 1, int)
    control = '--control' in a
    os.makedirs(out, exist_ok=True); d = os.path.join(out, f'ub{ub}'); os.makedirs(d, exist_ok=True)
    c1792 = os.path.join(LC, 'd1792', 'ctx.ids'); c2304 = ctx2304(out)
    J = ['nctx 2560', f'ubatch {ub}']
    if control:
        for t in TEXTS: J.append(f'tfx dec_{t} {tokfile(t)} 0 {NPOS} 1 0 {d}/dec_{t}')
    for t in TEXTS: J.append(f'slide sl_{t} {tokfile(t)} 1 {NPOS} {save_short} {d}/sl_{t}')
    for N, c in ((1792, c1792), (2048, c2304)):
        J.append(f'tfx ref{N} {c} {N} {NPOS} 1 {save_ref} {d}/ref{N}')
        J.append(f'tfx pd{N}_r1 {c} {N} {NPOS} 2048 1 {d}/pd{N}_r1')
        J.append(f'tfx pd{N}_r2 {c} {N} {NPOS} 2048 0 {d}/pd{N}_r2')
    jf = os.path.join(out, f'jobs_ub{ub}.txt'); open(jf, 'w').write('\n'.join(J) + '\n')
    print(jf)


# ---- comparison (CPU) ---------------------------------------------------------------------------------------------------
def logp(x):
    x = x - x.max(); return x - np.log(np.exp(x).sum())


def kld(ref, oth):
    lp, lq = logp(ref), logp(oth); return float((np.exp(lp) * (lp - lq)).sum())


def tsv(path):
    """gg_golden's rows: (pos, token in, argmax, fnv)"""
    return [tuple(l.split('\t')[:4]) for l in open(path).read().splitlines() if l]


def fnv_rows(rows_u8):
    """FNV-1a 64 of every row (uint8 [n, bytes]), vectorised over rows (gg_golden's lc_fnv)"""
    cols = np.ascontiguousarray(rows_u8.T).astype(np.uint64)                  # [bytes, rows]: each step reads one row
    h = np.full(rows_u8.shape[0], 1469598103934665603, dtype=np.uint64); P = np.uint64(1099511628211)
    with np.errstate(over='ignore'):
        for k in range(cols.shape[0]):
            h ^= cols[k]; h *= P
    return [f'{int(v):016x}' for v in h]


def dump_rows(t, positions):
    """the stored one-token decode dump's logits rows (refs/d281-short, llama.cpp main mode) at positions"""
    GD = os.path.join(D281_SHORT, f'e2e_{t}_decode')
    idx = {int(r[0]): r for r in (l.rstrip('\n').split('\t') for l in open(os.path.join(GD, 'index.tsv'))) if r[2] == 'result_output'}
    return np.stack([np.fromfile(os.path.join(GD, idx[p][8]), dtype=np.float32) for p in positions])


def stats(ref, oth, pos):
    ks, ag, flips = [], [], []
    for i in range(len(pos)):
        r, o = ref[i].astype(np.float64), oth[i].astype(np.float64)
        ks.append(kld(r, o)); ag.append(int(o.argmax() == r.argmax()))
        if not ag[-1]: s = np.sort(r)[::-1]; flips.append(f'{pos[i]}:{s[0] - s[1]:.2f}')
    ks = np.array(ks)
    return dict(n=len(pos), kld_mean=float(ks.mean()), kld_p99=float(np.percentile(ks, 99)), kld_max=float(ks.max()),
                top1=float(100 * np.mean(ag)), flips=flips, kld=ks.tolist(), agree=ag)


def line(name, s):
    return (f'{name}: {s["n"]} pos, mean KLD {s["kld_mean"]:.4f}, p99 {s["kld_p99"]:.4f}, max {s["kld_max"]:.4f}, top-1 {s["top1"]:.1f}%'
            f'  (flips pos:gap {" ".join(s["flips"]) or "none"})')


def merge(parts):
    ks = np.concatenate([np.array(p['kld']) for p in parts]); ag = sum((p['agree'] for p in parts), [])
    return dict(n=len(ks), kld_mean=float(ks.mean()), kld_p99=float(np.percentile(ks, 99)), kld_max=float(ks.max()),
                top1=float(100 * np.mean(ag)), flips=sum((p['flips'] for p in parts), []), kld=ks.tolist(), agree=ag)


def cmd_cmp(a):
    out = a.pop(0); jout = opt(a, '--json', None)
    ubs = sorted(int(n[2:]) for n in os.listdir(out) if n.startswith('ub') and os.path.isdir(os.path.join(out, n)))
    R = dict(ubs=ubs, controls={}, short={}, k2={}); bad = []
    def ctl(name, ok, detail=''):
        R['controls'][name] = dict(ok=bool(ok), detail=detail); print(('PASS ' if ok else 'FAIL ') + name + (f' ({detail})' if detail else ''), flush=True)
        if not ok: bad.append(name)
    base = 512 if 512 in ubs else ubs[0]; B = os.path.join(out, f'ub{base}')
    # control: the lc-mode one-token decode of the short texts == the stored dumps (the short floor's reference)
    for t in TEXTS:
        f = os.path.join(B, f'dec_{t}.tsv')
        if not os.path.exists(f): continue
        rows = tsv(f); pos = [int(r[0]) for r in rows]
        want = fnv_rows(dump_rows(t, pos).view(np.uint8))
        nd = sum(1 for r, w in zip(rows, want) if r[3] != w)
        ctl(f'control: lc-mode one-token decode of {t} (n_ctx 2560, ub {base}) == the stored main-mode dump (n_ctx 512), '
            f'{len(rows)} rows by FNV-1a', nd == 0, f'{nd} rows differ')
    # control: N = 1792 at ub 512 == the stored tf1792_r1 (gg_golden --lc tf: the long-context D-281 reference)
    if 512 in ubs:
        mine = tsv(os.path.join(out, 'ub512', 'pd1792_r1.tsv')); ref = tsv(os.path.join(LC, 'ref', 'tf1792_r1.tsv'))
        nd = sum(1 for x, y in zip(mine, ref) if x != y) + abs(len(mine) - len(ref))
        ctl('control: prefill(1792) then decode at ub 512 == qsa/lc/ref/tf1792_r1 (pos, token, argmax, FNV of every row)', nd == 0, f'{nd} rows differ')
    # short protocol
    for ub in ubs:
        d = os.path.join(out, f'ub{ub}'); parts = []
        for t in TEXTS:
            rows = tsv(os.path.join(d, f'sl_{t}.tsv')); pos = [int(r[0]) for r in rows]
            assert pos == list(range(1, NPOS)), (t, pos[:3], pos[-3:])
            if ub != base:
                same = [r[3] for r in rows] == [r[3] for r in tsv(os.path.join(B, f'sl_{t}.tsv'))]
                ctl(f'short {t}: ub {ub} == ub {base}, every row (FNV)', same)
                continue
            oth = np.fromfile(os.path.join(d, f'sl_{t}.bin'), dtype=np.float32).reshape(len(pos), NV)
            s = stats(dump_rows(t, pos), oth, pos); parts.append(s); R['short'][f'ub{ub}_{t}'] = s
            print('  SHORT ' + line(f'ub {ub} {t}', s), flush=True)
        if parts:
            s = merge(parts); R['short'][f'ub{ub}_ALL'] = s; print('  SHORT ' + line(f'ub {ub} ALL', s), flush=True)
    # 2K protocol
    for N in (1792, 2048):
        for ub in ubs:
            d = os.path.join(out, f'ub{ub}')
            if ub != base:
                same = [r[3] for r in tsv(os.path.join(d, f'ref{N}.tsv'))] == [r[3] for r in tsv(os.path.join(B, f'ref{N}.tsv'))]
                ctl(f'2K N={N}: the one-token reference at ub {ub} == at ub {base} (FNV)', same)
            r1, r2 = tsv(os.path.join(d, f'pd{N}_r1.tsv')), tsv(os.path.join(d, f'pd{N}_r2.tsv'))
            nd = sum(1 for x, y in zip(r1, r2) if x[3] != y[3])
            print(f'  2K N={N} ub {ub}: run-to-run (r2 vs r1), {nd} of {len(r1)} rows differ (FNV)', flush=True)
            pos = [int(r[0]) for r in r1]; assert pos == list(range(N, N + NPOS)), (N, pos[:2])
            ref = np.fromfile(os.path.join(B, f'ref{N}.bin'), dtype=np.float32).reshape(NPOS, NV)
            oth = np.fromfile(os.path.join(d, f'pd{N}_r1.bin'), dtype=np.float32).reshape(NPOS, NV)
            s = stats(ref, oth, pos); s['r2_rows_differ'] = nd; R['k2'][f'N{N}_ub{ub}'] = s
            print('  2K ' + line(f'N={N} ub {ub}', s), flush=True)
    if jout:
        slim = {k: ({kk: {x: y for x, y in vv.items() if x not in ('kld', 'agree')} for kk, vv in v.items()} if k in ('short', 'k2') else v)
                for k, v in R.items()}
        json.dump(slim, open(jout, 'w'), indent=1)
    print('RESULT ' + ('PASS' if not bad else f'FAIL ({len(bad)} controls: {bad})'))
    return 0 if not bad else 1


def pd_dump(refs, t):
    GD = os.path.join(refs, f'e2e_{t}_decode')
    idx = {int(r[0]): r for r in (l.rstrip('\n').split('\t') for l in open(os.path.join(GD, 'index.tsv'))) if r[2] == 'result_output'}
    return GD, idx


def cmd_pd_jobs(a):
    out = a.pop(0); refs = opt(a, '--refs', None); p0 = opt(a, '--p0', 128, int); ub = opt(a, '--ub', 512, int)
    if not refs: raise SystemExit('pd-jobs: --refs DIR (e2e_{prose,code,shell}_decode)')
    os.makedirs(out, exist_ok=True); d = os.path.join(out, f'pd_ub{ub}'); os.makedirs(d, exist_ok=True)
    J = ['nctx 512', f'ubatch {ub}']
    for t in TEXTS:
        tf = os.path.join(refs, f'e2e_{t}_decode', 'tokens.txt'); n = len(tokens(tf))
        assert p0 < n <= 512, (t, p0, n)
        J.append(f'tfx dec_{t} {tf} 0 {n} 1 0 {d}/dec_{t}')
        J.append(f'tfx pd_{t}_r1 {tf} {p0} {n - p0} 512 1 {d}/pd_{t}_r1')   # (B 512 = n_batch at nctx 512: the prompt in one pass)
        J.append(f'tfx pd_{t}_r2 {tf} {p0} {n - p0} 512 0 {d}/pd_{t}_r2')   # (B 512 = n_batch at nctx 512: the prompt in one pass)
    jf = os.path.join(out, f'jobs_pd_ub{ub}.txt'); open(jf, 'w').write('\n'.join(J) + '\n')
    print(jf)


def cmd_pd_cmp(a):
    out = a.pop(0); refs = opt(a, '--refs', None); p0 = opt(a, '--p0', 128, int); jout = opt(a, '--json', None)
    if not refs: raise SystemExit('pd-cmp: --refs DIR')
    ubs = sorted(int(n[5:]) for n in os.listdir(out) if n.startswith('pd_ub'))
    R = dict(p0=p0, refs=refs, controls={}, floor={}); bad = []
    for ub in ubs:
        d = os.path.join(out, f'pd_ub{ub}'); parts = []
        for t in TEXTS:
            GD, idx = pd_dump(refs, t)
            dec = tsv(os.path.join(d, f'dec_{t}.tsv')); pos = [int(r[0]) for r in dec]
            rows = np.stack([np.fromfile(os.path.join(GD, idx[p][8]), dtype=np.float32) for p in pos])
            nd = sum(1 for r, w in zip(dec, fnv_rows(rows.view(np.uint8))) if r[3] != w)
            name = f'control: lc-mode one-token decode of {t} (ub {ub}) == the stored dump, {len(dec)} rows by FNV-1a'
            R['controls'][name] = dict(ok=nd == 0, rows_differ=nd); print(('PASS ' if nd == 0 else 'FAIL ') + name + f' ({nd} rows differ)', flush=True)
            if nd: bad.append(name)
            r1, r2 = tsv(os.path.join(d, f'pd_{t}_r1.tsv')), tsv(os.path.join(d, f'pd_{t}_r2.tsv'))
            pp = [int(r[0]) for r in r1]; assert pp == list(range(p0, p0 + len(r1))), (t, pp[:2])
            nv = os.path.getsize(os.path.join(d, f'pd_{t}_r1.bin')) // (4 * len(r1))
            oth = np.fromfile(os.path.join(d, f'pd_{t}_r1.bin'), dtype=np.float32).reshape(len(r1), nv)
            ref = np.stack([np.fromfile(os.path.join(GD, idx[p][8]), dtype=np.float32) for p in pp])
            s = stats(ref, oth, pp); s['r2_rows_differ'] = sum(1 for x, y in zip(r1, r2) if x[3] != y[3]); parts.append(s)
            R['floor'][f'ub{ub}_{t}'] = s
            print(f'  PD ' + line(f'ub {ub} {t} positions {p0} .. {pp[-1]}', s) + f'; run to run {s["r2_rows_differ"]} rows differ', flush=True)
        s = merge(parts); s['boot95'] = boot(s['kld']); R['floor'][f'ub{ub}_ALL'] = s
        print(f'  PD ' + line(f'ub {ub} ALL', s) + f'; 95 % bootstrap of the mean {s["boot95"][0]:.5f} .. {s["boot95"][1]:.5f}', flush=True)
    if jout:
        json.dump({k: ({kk: {x: y for x, y in vv.items() if x not in ('kld', 'agree')} for kk, vv in v.items()} if k == 'floor' else v)
                   for k, v in R.items()}, open(jout, 'w'), indent=1)
    print('RESULT ' + ('PASS' if not bad else f'FAIL ({len(bad)} controls)'))
    return 0 if not bad else 1


def boot(ks, n=10000, seed=0):
    """the 95 % bootstrap interval of the mean of ks (positions resampled with replacement, seeded)"""
    ks = np.asarray(ks, np.float64); rng = np.random.default_rng(seed)
    m = ks[rng.integers(0, len(ks), (n, len(ks)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cmd_depth_jobs(a):
    out = a.pop(0); depths = [int(x) for x in opt(a, '--depths', '8192,32768').split(',')]; ub = opt(a, '--ub', 2048, int)
    ctx = opt(a, '--ctx', None); r1 = opt(a, '--runs-1tok', None, int); rp = opt(a, '--runs-pd', None, int)
    nctx = opt(a, '--nctx', None, int)
    os.makedirs(out, exist_ok=True)
    nctx = nctx or (max(depths) + NPOS + 255) // 256 * 256
    if ctx is not None or r1 is not None or rp is not None:
        # the sweep's multi-run floors (pool): one token file for every depth (a prefix of one sequence), r1 one-token runs and
        # rp prefill-then-decode runs at ub, EVERY run's logits saved (each 254 MB); no ub 512 file (no stored reference there)
        r1 = 3 if r1 is None else r1; rp = 2 if rp is None else rp
        d = os.path.join(out, f'depth_ub{ub}'); os.makedirs(d, exist_ok=True)
        J = [f'nctx {nctx}', f'ubatch {ub}']
        for N in depths:
            c = ctx if ctx is not None else os.path.join(LC, f'd{N}', 'ctx.ids')
            assert len(tokens(c)) >= N + NPOS and N + NPOS <= nctx, (c, len(tokens(c)), N, nctx)
            J += [f'tfx pd{N}_r{k + 1} {c} {N} {NPOS} 2048 1 {d}/pd{N}_r{k + 1}' for k in range(rp)]
        jf = os.path.join(out, f'jobs_pd_ub{ub}.txt'); open(jf, 'w').write('\n'.join(J) + '\n'); print(jf)
        J = [f'nctx {nctx}', f'ubatch {ub}']
        for N in depths:
            c = ctx if ctx is not None else os.path.join(LC, f'd{N}', 'ctx.ids')
            J += [f'tfx ref{N}_r{k + 1} {c} {N} {NPOS} 1 1 {d}/ref{N}_r{k + 1}' for k in range(r1)]
        jf = os.path.join(out, f'jobs_1tok_ub{ub}.txt'); open(jf, 'w').write('\n'.join(J) + '\n'); print(jf)
        return
    for u, full in ((ub, True), (512, False)):
        d = os.path.join(out, f'depth_ub{u}'); os.makedirs(d, exist_ok=True)
        J = [f'nctx {nctx}', f'ubatch {u}']
        for N in depths:
            c = os.path.join(LC, f'd{N}', 'ctx.ids')
            assert len(tokens(c)) >= N + NPOS, (c, len(tokens(c)))
            if full:
                J.append(f'tfx pd{N}_r1 {c} {N} {NPOS} 2048 1 {d}/pd{N}_r1')
                J.append(f'tfx pd{N}_r2 {c} {N} {NPOS} 2048 0 {d}/pd{N}_r2')
                J.append(f'tfx ref{N} {c} {N} {NPOS} 1 1 {d}/ref{N}')
            else:
                J.append(f'tfx pd{N}_r1 {c} {N} {NPOS} 2048 0 {d}/pd{N}_r1')
        jf = os.path.join(out, f'jobs_depth_ub{u}.txt'); open(jf, 'w').write('\n'.join(J) + '\n'); print(jf)


def cmd_depth_cmp(a):
    out = a.pop(0); jout = opt(a, '--json', None)
    ubs = sorted(int(n[8:]) for n in os.listdir(out) if n.startswith('depth_ub') and os.path.isdir(os.path.join(out, n)))
    full = [u for u in ubs if os.path.exists(os.path.join(out, f'depth_ub{u}')) and any(f.startswith('ref') and f.endswith('.bin') for f in os.listdir(os.path.join(out, f'depth_ub{u}')))]
    R = dict(controls={}, depth={}); bad = []
    def ctl(name, ok, detail=''):
        R['controls'][name] = dict(ok=bool(ok), detail=detail); print(('PASS ' if ok else 'FAIL ') + name + (f' ({detail})' if detail else ''), flush=True)
        if not ok: bad.append(name)
    for ub in full:
        d = os.path.join(out, f'depth_ub{ub}')
        Ns = sorted(int(f[3:-4]) for f in os.listdir(d) if f.startswith('ref') and f.endswith('.bin'))
        for N in Ns:
            r1, r2 = tsv(os.path.join(d, f'pd{N}_r1.tsv')), tsv(os.path.join(d, f'pd{N}_r2.tsv'))
            nd = sum(1 for x, y in zip(r1, r2) if x[3] != y[3]); na = sum(1 for x, y in zip(r1, r2) if x[2] != y[2])
            st1, st2 = tsv(os.path.join(LC, 'ref', f'tf{N}_r1.tsv')), tsv(os.path.join(LC, 'ref', f'tf{N}_r2.tsv'))
            snd, sna = sum(1 for x, y in zip(st1, st2) if x[3] != y[3]), sum(1 for x, y in zip(st1, st2) if x[2] != y[2])
            R.setdefault('run_to_run', {})[f'N{N}'] = dict(r1_r2_fnv=nd, r1_r2_argmax=na, stored_r1_r2_fnv=snd, stored_r1_r2_argmax=sna)
            print(f'  depth N={N} ub {ub}: llama.cpp run to run (r2 vs r1): {nd} of {len(r1)} rows differ (FNV), {na} argmaxes; the stored '
                  f'qsa/lc/ref tf{N}_r1 vs _r2 (ub 512): {snd} rows (FNV), {sna} argmaxes', flush=True)
            pos = [int(r[0]) for r in r1]; assert pos == list(range(N, N + NPOS)), (N, pos[:2])
            ref = np.fromfile(os.path.join(d, f'ref{N}.bin'), dtype=np.float32).reshape(NPOS, NV)
            oth = np.fromfile(os.path.join(d, f'pd{N}_r1.bin'), dtype=np.float32).reshape(NPOS, NV)
            s = stats(ref, oth, pos); ks = np.array(s['kld']); i = int(ks.argmax())
            s['boot95'] = boot(ks); s['max_pos'] = pos[i]; s['mean_wo_max'] = float(np.delete(ks, i).mean()); s['median'] = float(np.median(ks))
            R['depth'][f'N{N}_ub{ub}'] = s
            print(f'  DEPTH {line(f"N={N} ub {ub}", s)}; 95 % bootstrap {s["boot95"][0]:.5f} .. {s["boot95"][1]:.5f}; max at {s["max_pos"]}, '
                  f'mean without it {s["mean_wo_max"]:.5f}, median {s["median"]:.6f}', flush=True)
            # reported: the ub 512 prefill-then-decode beside the stored D-281 long reference (same protocol, another process)
            c = os.path.join(out, 'depth_ub512', f'pd{N}_r1.tsv')
            if os.path.exists(c):
                mine, st = tsv(c), tsv(os.path.join(LC, 'ref', f'tf{N}_r1.tsv'))
                ok_tok = [(x[0], x[1]) for x in mine] == [(x[0], x[1]) for x in st]
                ctl(f'control: the ub 512 run and qsa/lc/ref/tf{N}_r1 cover the same positions and tokens', ok_tok)
                nd, na = sum(1 for x, y in zip(mine, st) if x[3] != y[3]), sum(1 for x, y in zip(mine, st) if x[2] != y[2])
                R['run_to_run'][f'N{N}'].update(ub512_vs_stored_fnv=nd, ub512_vs_stored_argmax=na)
                print(f'  depth N={N}: prefill-then-decode at ub 512 vs the stored tf{N}_r1: {nd} rows differ (FNV), {na} argmaxes '
                      f'(the stored r1 vs r2: {snd} / {sna})', flush=True)
    if jout:
        json.dump({k: ({kk: {x: y for x, y in vv.items() if x not in ('kld', 'agree')} for kk, vv in v.items()} if k == 'depth' else v)
                   for k, v in R.items()}, open(jout, 'w'), indent=1)
    print('RESULT ' + ('PASS' if not bad else f'FAIL ({len(bad)} controls: {bad})'))
    return 0 if not bad else 1


def rows_of(binpath):
    """a run's tsv rows (pos, token, argmax, fnv): OUT.bin -> OUT.tsv, DIR/logits.bin -> DIR/steps.tsv"""
    d, b = os.path.split(binpath)
    return tsv(os.path.join(d, 'steps.tsv') if b == 'logits.bin' else binpath[:-4] + '.tsv')


def pooled(perpair, agrees, pos):
    """the pooled floor of the per-pairing KLD vectors (each [NPOS], the same positions): per position the mean over pairings,
    its mean, the 95 % bootstrap interval resampling positions (boot()), top-1 = the mean over pairings"""
    K = np.mean(np.stack(perpair), axis=0); i = int(K.argmax())
    return dict(n=len(K), pairings=len(perpair), kld_mean=float(K.mean()), boot95=boot(K), kld_p99=float(np.percentile(K, 99)),
                kld_max=float(K.max()), max_pos=int(pos[i]), mean_wo_max=float(np.delete(K, i).mean()), median=float(np.median(K)),
                top1=float(np.mean([100 * np.mean(g) for g in agrees])), kld=K.tolist())


def cmd_pool(a):
    out = a.pop(0)
    def labelled(flag):
        r = []
        while flag in a:
            i = a.index(flag); lab, _, p = a[i + 1].partition('='); r.append((lab, p)); del a[i:i + 2]
        return r
    refs, pds = labelled('--ref'), labelled('--pd')
    exp = []
    while '--expect' in a:
        i = a.index('--expect'); k, _, v = a[i + 1].partition('='); exp.append((tuple(k.split(':')), float(v))); del a[i:i + 2]
    if not refs or not pds or a: print(__doc__); return 2
    R = dict(refs=dict(refs), pds=dict(pds), controls={}, pairs={}); bad = []
    def ctl(name, ok, detail=''):
        R['controls'][name] = dict(ok=bool(ok), detail=detail); print(('PASS ' if ok else 'FAIL ') + name + (f' ({detail})' if detail else ''), flush=True)
        if not ok: bad.append(name)
    rows = {lab: rows_of(p) for lab, p in refs + pds}
    base = rows[refs[0][0]]; pos = [int(r[0]) for r in base]; n = len(pos)
    same = all([(x[0], x[1]) for x in rw] == [(x[0], x[1]) for x in base] for rw in rows.values())
    ctl(f'control: every run holds the same {n} (position, token) rows ({len(refs)} one-token, {len(pds)} prefill-then-decode runs)',
        same and pos == list(range(pos[0], pos[0] + n)))
    if bad: return 1
    def r2r(labs):
        out = {}
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                x, y = rows[labs[i]], rows[labs[j]]
                out[f'{labs[i]}:{labs[j]}'] = dict(fnv=sum(1 for u, v in zip(x, y) if u[3] != v[3]), argmax=sum(1 for u, v in zip(x, y) if u[2] != v[2]))
        return out
    R['run_to_run'] = dict(one_token=r2r([l for l, _ in refs]), pd=r2r([l for l, _ in pds]))
    print(f'  run to run (rows differing by FNV / argmax): one-token {R["run_to_run"]["one_token"]}; prefill-then-decode {R["run_to_run"]["pd"]}', flush=True)
    mm = {lab: np.memmap(p, dtype=np.float32, mode='r').reshape(-1, NV) for lab, p in refs + pds}
    assert all(m_.shape[0] == n for m_ in mm.values()), {k: v.shape for k, v in mm.items()}
    ks = {(r, p): np.zeros(n) for r, _ in refs for p, _ in pds}; ag = {k: [] for k in ks}; flips = {k: [] for k in ks}
    for t in range(n):                                    # row by row: one position's rows of every run in float64
        lp = {lab: logp(mm[lab][t].astype(np.float64)) for lab in mm}
        for (r, p) in ks:
            ks[(r, p)][t] = float((np.exp(lp[r]) * (lp[r] - lp[p])).sum())
            a_ = int(lp[p].argmax() == lp[r].argmax()); ag[(r, p)].append(a_)
            if not a_: s_ = np.sort(lp[r])[::-1]; flips[(r, p)].append(f'{pos[t]}:{s_[0] - s_[1]:.2f}')
    for (r, p), k in ks.items():
        i = int(k.argmax())
        s = dict(n=n, kld_mean=float(k.mean()), boot95=boot(k), kld_p99=float(np.percentile(k, 99)), kld_max=float(k.max()), max_pos=pos[i],
                 mean_wo_max=float(np.delete(k, i).mean()), median=float(np.median(k)), top1=float(100 * np.mean(ag[(r, p)])), flips=flips[(r, p)])
        R['pairs'][f'{r}:{p}'] = s
        print(f'  pair {r} (one-token) vs {p} (prefill-then-decode): mean KLD {s["kld_mean"]:.5f}, 95 % bootstrap {s["boot95"][0]:.5f} .. {s["boot95"][1]:.5f}, '
              f'top-1 {s["top1"]:.2f} %, max {s["kld_max"]:.4f} at {s["max_pos"]}, mean without it {s["mean_wo_max"]:.5f}', flush=True)
    P = pooled(list(ks.values()), list(ag.values()), pos); R['pooled'] = {k: v for k, v in P.items() if k != 'kld'}; R['pooled_kld'] = P['kld']
    means = [s['kld_mean'] for s in R['pairs'].values()]
    R['spread'] = dict(min=float(min(means)), max=float(max(means)), sd=float(np.std(means, ddof=1)) if len(means) > 1 else 0.0)
    R['bars'] = dict(g4a_kld=P['boot95'][1], g4a_top1=P['top1'] - 1.0, rule='G4a mean KLD <= the pooled 95 % bootstrap upper bound, top-1 >= the pooled top-1 - 1 point')
    print(f'  POOLED over {P["pairings"]} pairings: mean KLD {P["kld_mean"]:.5f}, 95 % bootstrap (positions) {P["boot95"][0]:.5f} .. {P["boot95"][1]:.5f}, '
          f'top-1 {P["top1"]:.2f} %, max {P["kld_max"]:.4f} at {P["max_pos"]}, mean without it {P["mean_wo_max"]:.5f}, median {P["median"]:.6f}; '
          f'per-pairing means {R["spread"]["min"]:.5f} .. {R["spread"]["max"]:.5f} (sd {R["spread"]["sd"]:.5f}); G4a bars: KLD <= {P["boot95"][1]:.5f}, '
          f'top-1 >= {P["top1"] - 1.0:.2f}', flush=True)
    for (r, p), v in exp:
        got = R['pairs'].get(f'{r}:{p}', {}).get('kld_mean')
        ctl(f'control: pairing {r}:{p} reproduces the earlier mean {v:.10f}', got is not None and abs(got - v) <= 1e-9, f'got {got}')
    json.dump(R, open(out, 'w'), indent=1)
    print(f'wrote {out}')
    print('RESULT ' + ('PASS' if not bad else f'FAIL ({len(bad)} controls: {bad})'))
    return 0 if not bad else 1


def main():
    a = sys.argv[1:]
    if not a or a[0] not in ('jobs', 'cmp', 'depth-jobs', 'depth-cmp', 'pool', 'pd-jobs', 'pd-cmp'): print(__doc__); sys.exit(2)
    c = a.pop(0)
    if c == 'jobs': cmd_jobs(a); sys.exit(0)
    if c == 'depth-jobs': cmd_depth_jobs(a); sys.exit(0)
    if c == 'depth-cmp': sys.exit(cmd_depth_cmp(a))
    if c == 'pool': sys.exit(cmd_pool(a))
    if c == 'pd-jobs': cmd_pd_jobs(a); sys.exit(0)
    if c == 'pd-cmp': sys.exit(cmd_pd_cmp(a))
    sys.exit(cmd_cmp(a))


if __name__ == '__main__':
    main()
