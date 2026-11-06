"""The QSA depth investigation (impl:qsa-depth; docs/AGENTIC-PLAN.md, the S7 rulings: required before the 262K sweep).

At 32K depth G4a is 0.00128 (bf16) / 0.00118 (fp32) against llama.cpp's own floor mean 0.00113 (bootstrap 0.00009 ..
0.00293), with per-position tails 3-4x llama.cpp's (p99 0.050 / 0.036 vs 0.013). Question: is the tail the indexer's
normal near-tie sensitivity, or a defect? This tool collects the QSA selection (top-512 pools + the tail) of every QSA
layer (3, 7, .., 47) at the G4 window positions N .. N + 255 (N = 8192, the control, and 32768; tokens qsa/lc/d{N}/ctx.ids)
under each protocol, and compares them.

  K3_TEST_HOOKS=qsa-scores tools/guard.sh LOG 3600 $PY tools/qsa_depth.py k3 CODER --out DIR [--depths 8192,32768]
  tools/guard.sh LOG 3600 $PY tools/qsa_depth.py k3 CODER --out DIR --plain    (the default tape: the hook-changes-nothing check)
  $PY tools/qsa_depth.py jobs DIR [--depths 8192,32768]       -> DIR/llama/jobs_qsa.txt
  tools/guard.sh LOG 3600 build/gg_golden CODER --lc DIR/llama/jobs_qsa.txt   (llama.cpp, both cards, ~25 min)
  HIP_VISIBLE_DEVICES= $PY tools/qsa_depth.py cmp DIR [--json F]   (CPU only)

Protocols (the two families are never mixed):
  PD (prefill-then-decode, what G4 measures): the state of [0, N) is built one way, then the window is decoded one token
  per position (K3: teacher-forced T = 4 passes, bitwise one-token steps; llama.cpp: one token per llama_decode).
    K3  dec  one-token decode from position 0 (G4a's reference)          pd   the batched prefill of [0, N) (chunk 2048, the
                                                                              defaults: bf16 WMMA outputs, own MoE / Q6_K)
        t1   pd's state with dec's IDXRAW rows < N and IDXPOOL pools < N / 4 written in (the indexer codes transplanted)
        t2   dec's state with pd's IDXRAW / IDXPOOL rows written in      (t1 / t2: in-bounds row copies, both cards)
    llama dec  B = 1 fill (gg_golden qsax B 1)                           pd_r1 / pd_r2  ub 2048 fill, run twice (its own
                                                                              run-to-run selection noise)
  In-batch: the window as the last 256-row chunk / ubatch of a prefill of [0, N + 256) (the prompt path's own selection).
    K3  inb  Prefill.run(toks, 0, 0, N + 256): the QSA prompt path's SEL / score / query rows captured after each
             qsa_select / qsa_score (host copies on the launch's stream, card 0)
    llama inb  ub 2048 fill of [0, N), then the window as ONE 256-row llama_decode (qsab: the tile indexer, radix top-k)
  K3 captures: SEL rows of every QSA layer (FlashNextTP.qsa_sel, per layer already), and under K3_TEST_HOOKS=qsa-scores
  each layer's own score / indexer-query rows (FlashNextTP.qsa_score_l / qsa_q_l: only buffer pointers differ from the
  default tape). Controls: the hooked run's logits (dec, pd) bitwise == the --plain run's (sha256) and its G4a / G4b
  numbers == lead-g4d-ab's; llama.cpp's dump runs at 8K == the stored floor runs by FNV (ub 2048 is deterministic there).
  The backfill self-check: each K3 run's IDXRAW re-pooled by prefill_attn.idx_pool on the XT (bitwise OP_IDX_APPEND's
  pooling) == that run's IDXPOOL (pools < 513: the dual tape's backfill at 2051; >= 513: IDX_APPEND / the prompt path).

PRE-REGISTERED (written before any ours-vs-llama selection number was computed; not changed after):
  dis(X, Y)[p, l] = |X \\ Y| = |X delta Y| / 2, the pools one selection picks and the other does not (of 512), per window
  position p and QSA layer l. D(p) = mean over the 12 layers of dis(PD-run, dec-run)[p, l].
  first-difference rank: the smallest rank, in the reference run's score order (descending, lower pool index first among
  equal scores: OP_QSA_SELECT's order), of any pool in X delta Y (0-based; 512 = the first pool outside the selection).
  Reference runs: K3 dec for ours, llama dec for llama.cpp's, llama dec for the cross pairs.
  THE "WITHIN" TEST (binding, per depth): ours = D_ours(p) = dis(K3 pd, K3 dec), llama = D_llama(p) = dis(llama pd_r1,
  llama dec), the same 256 positions and tokens. Within iff BOTH (a) mean_p D_ours <= the 95 % bootstrap upper bound of
  mean_p D_llama (g4_floor.boot: 10,000 seeded resamples of positions, the 12 layers of a position kept together: the G4a
  precedent) and (b) the paired bootstrap of mean_p (D_ours(p) - D_llama(p)) has a 95 % interval whose lower end is <= 0
  (ours not significantly above llama.cpp's). Per layer the same two numbers are reported; a layer where ours is
  significantly above (lower end > 0) is flagged and must be localized before 'normal'.
  EXPLAINED (the KLD tail by the selection): Spearman rank correlation of the per-position G4a KLD with D_ours(p), and of
  llama.cpp's floor KLD (llama pd_r1 vs llama dec, the dump runs' own logits at 32K) with D_llama(p), reported; the 5
  largest-KLD positions of G4a with their D_ours percentile. CAUSE (transplant): the tail is attributed to the indexer's
  codes when KLD(t1 vs dec)'s top-5 mean falls to <= 50 % of KLD(pd vs dec)'s AND KLD(t2 vs dec)'s top-5 mean is >= 50 %
  of it; to the non-indexer state (KV rows, recurrent states, hidden states) when t1 keeps >= 50 % of it and t2 < 50 %;
  mixed otherwise.
Outputs (DIR = $ARTIFACTS/impl-qsa-depth): k3/k3_N{N}_{run}.npz (sel int32 [256, 12, 516]; score float32 [256, 12, nv]
for dec / pd / inb; q float32 [256, 12, 512]), k3/states_N{N}_{run}.npz (IDXRAW / IDXPOOL rows of card 0), k3/k3_{mode}.json
(per-position KLDs, logits sha256, the self-checks); llama/{run}{N}/ (dump.bin + index.tsv, steps.tsv, logits.bin at 32K);
report.json from cmp."""
import argparse, hashlib, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'python')); sys.path.insert(0, os.path.join(HERE, 'gates')); sys.path.insert(0, os.path.join(ROOT, 'tests'))
from common import ARTIFACTS, LC, tokens   # noqa: E402
import g4_floor as G                        # noqa: E402  (KLD / stats / bootstrap: the floor's math)

NL, SELW, TOPK, QD = 12, 516, 512, 512      # QSA layers, SEL row ints, picks, indexer query floats (4 x 128)
QSA_LAYERS = list(range(3, 48, 4))
NPOS = 256
HOOKS = 'qsa-scores'
S7G4 = os.path.join(ARTIFACTS, 'impl-prefill-s7', 'g4', 'depth_ub2048')
LEAD = {8192: dict(g4a=0.00156, g4b=0.00397), 32768: dict(g4a=0.00128, g4b=0.00127)}   # lead-g4d-ab bf16 (the defaults)


def comp():
    return open('/proc/sys/vm/compaction_proactiveness').read().strip()


def memavail():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'): return l.split()[1] + ' kB'


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---- K3 (GPU) -----------------------------------------------------------------------------------------------------------
class WindowCap:
    """the K3 decode window's rows after each pass: SEL of every QSA layer (card 0; card 1's checked equal), and with the
    hook each layer's score row [:nvmax] and indexer query; plus the logits rows"""
    def __init__(self, torch, m, lo, hi, nvmax, hooked):
        self.t, self.m, self.lo, self.hi, self.nvmax, self.hooked = torch, m, lo, hi, nvmax, hooked
        n = hi - lo
        self.sel = np.zeros((n, NL, SELW), np.int32); self.q = np.zeros((n, NL, QD), np.float32) if hooked else None
        self.score = np.zeros((n, NL, nvmax), np.float32) if hooked else None
        self.lg = []; self.sel_rep = True

    def __call__(self, i0, lo):
        t, m = self.t, self.m
        for d in (0, 1): t.cuda.synchronize(m.devs[d])
        f = lo.shape[0]; T = m.ntok
        a, b = max(i0, self.lo), min(i0 + f, self.hi)
        if a >= b: return
        self.lg.append(lo[a - i0:b - i0].float().cpu().numpy())
        r0, r1 = a - i0, b - i0
        for ai in range(NL):
            s0 = m.qsa_sel[0][ai].view(T, SELW)[r0:r1].cpu().numpy(); s1 = m.qsa_sel[1][ai].view(T, SELW)[r0:r1].cpu().numpy()
            self.sel_rep &= bool(np.array_equal(s0, s1))
            self.sel[a - self.lo:b - self.lo, ai] = s0
            if self.hooked:
                self.score[a - self.lo:b - self.lo, ai] = m.qsa_score_l[0][ai][r0:r1, :self.nvmax].cpu().numpy()
                self.q[a - self.lo:b - self.lo, ai] = m.qsa_q_l[0][ai].view(T, QD)[r0:r1].cpu().numpy()

    def logits(self):
        return np.concatenate(self.lg)


class InbCap:
    """the QSA prompt path's rows of positions [lo, hi) during a prefill (card 0): each qsa_select's SEL and score rows,
    each indexer qk_nr's query rows, copied to the host right after the launch on the same (current) stream"""
    def __init__(self, S, lo, hi, nvmax):
        self.S, self.lo, self.hi, self.nvmax = S, lo, hi, nvmax
        n = hi - lo
        self.sel = np.full((n, NL, SELW), -1, np.int32); self.score = np.zeros((n, NL, nvmax), np.float32)
        self.q = np.zeros((n, NL, QD), np.float32); self.seen = np.zeros((n, NL, 2), bool)
        self.A = S.A[0]; self.o_sel, self.o_qk = self.A.qsa_select, self.A.qk_nr
        self.A.qsa_select, self.A.qk_nr = self.select, self.qk

    def close(self):
        self.A.qsa_select, self.A.qk_nr = self.o_sel, self.o_qk

    def select(self, score, ld, pos0, T, prow, sel):
        r = self.o_sel(score, ld, pos0, T, prow, sel)
        a, b = max(int(pos0), self.lo), min(int(pos0) + int(T), self.hi)
        if a < b:
            ai = self.S._il // 4; ld = int(ld); r0, r1 = a - int(pos0), b - int(pos0)
            sv = sel.reshape(-1)[r0 * SELW:r1 * SELW].view(r1 - r0, SELW).cpu().numpy()
            sc = score.reshape(-1)[r0 * ld:r1 * ld].view(r1 - r0, ld)[:, :self.nvmax].cpu().numpy()
            self.sel[a - self.lo:b - self.lo, ai] = sv; self.score[a - self.lo:b - self.lo, ai] = sc
            self.seen[a - self.lo:b - self.lo, ai, 0] = True
        return r

    def qk(self, x, head_stride, heads, D, w, n_rot, theta_scale, eps, pos0, out, T):
        r = self.o_qk(x, head_stride, heads, D, w, n_rot, theta_scale, eps, pos0, out, T)
        if int(heads) == 4 and int(D) == 128:      # the indexer query (the attention q / k have head dim 256)
            a, b = max(int(pos0), self.lo), min(int(pos0) + int(T), self.hi)
            if a < b:
                ai = self.S._il // 4; r0, r1 = a - int(pos0), b - int(pos0)
                self.q[a - self.lo:b - self.lo, ai] = out.reshape(-1)[r0 * QD:r1 * QD].view(r1 - r0, QD).cpu().numpy()
                self.seen[a - self.lo:b - self.lo, ai, 1] = True
        return r


def idx_rows(torch, m, c, n_raw, n_pool):
    st = m.state
    with torch.cuda.device(m.devs[c]):
        return st[(c, 'idxraw')][:, :n_raw].cpu().numpy(), st[(c, 'idxpool')][:, :n_pool].cpu().numpy()


def put_idx(torch, m, raw, pool):
    """write IDXRAW rows [0, raw.shape[1]) and IDXPOOL pools [0, pool.shape[1]) of every QSA layer on both cards (in bounds:
    the slices of the engine's own tensors)"""
    for c in (0, 1):
        with torch.cuda.device(m.devs[c]):
            R, P = m.state[(c, 'idxraw')], m.state[(c, 'idxpool')]
            assert raw.shape[0] == R.shape[0] and raw.shape[1] <= R.shape[1] and raw.shape[2] == R.shape[2], (raw.shape, R.shape)
            assert pool.shape[0] == P.shape[0] and pool.shape[1] <= P.shape[1] and pool.shape[2] == P.shape[2], (pool.shape, P.shape)
            R[:, :raw.shape[1]].copy_(torch.from_numpy(raw).to(R.device)); P[:, :pool.shape[1]].copy_(torch.from_numpy(pool).to(P.device))
    for c in (0, 1): torch.cuda.synchronize(m.devs[c])


def repool_check(torch, m, S, n_raw):
    """re-pool every QSA layer's IDXRAW rows [0, n_raw) on the XT (card 1) with prefill_attn.idx_pool into a fresh buffer and
    compare with the engine's IDXPOOL (card 1): per layer the pools that differ, below 513 (backfill) and from 513"""
    from caminus import flashnext as F
    c = 1; A, AW = S.A[c], S.AW[c]; nj = n_raw // 4; out = []
    with torch.cuda.device(m.devs[c]):
        for ai in range(NL):
            raw = m.state[(c, 'idxraw')][ai].clone(); pool = torch.zeros_like(m.state[(c, 'idxpool')][ai])
            A.idx_pool(raw, pool, 0, nj, m._idx_kn[ai][c], m.theta_scale, F.EPS, AW.err)
            torch.cuda.synchronize(m.devs[c])
            d = (pool[:nj] != m.state[(c, 'idxpool')][ai][:nj]).any(dim=1).cpu().numpy()
            out.append(dict(layer=QSA_LAYERS[ai], pools=nj, differ_lt513=int(d[:513].sum()), differ_ge513=int(d[513:].sum())))
            del raw, pool
        err = int(AW.err.reshape(-1)[0].item())
    return out, err


def kstats(ref, oth, pos):
    s = G.stats(ref, oth, pos); ks = np.array(s['kld']); top = np.argsort(-ks)[:5]
    s['top5'] = [(int(pos[i]), float(ks[i])) for i in top]; s['top5_mean'] = float(ks[top].mean())
    s['boot_mean'] = G.boot(ks); s['boot_p99'] = boot_p99(ks)
    return s


def boot_p99(ks, n=10000, seed=0):
    ks = np.asarray(ks, np.float64); rng = np.random.default_rng(seed)
    v = np.percentile(ks[rng.integers(0, len(ks), (n, len(ks)))], 99, axis=1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def cmd_k3(a):
    import torch
    hooked = os.environ.get('K3_TEST_HOOKS') == HOOKS
    if a.plain == hooked:
        sys.exit(f'k3: --plain needs K3_TEST_HOOKS unset, the capture run K3_TEST_HOOKS={HOOKS} (got {os.environ.get("K3_TEST_HOOKS")!r})')
    for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)
    assert 'XTX' in torch.cuda.get_device_name(0) and 'XTX' not in torch.cuda.get_device_name(1), 'cuda:0 must be the XTX, cuda:1 the XT'
    from caminus import longctx
    import test_prefill_s6 as S6
    out = os.path.join(a.out, 'k3'); os.makedirs(out, exist_ok=True)
    mode = 'plain' if a.plain else 'hooked'
    print(f'compaction_proactiveness {comp()}; MemAvailable {memavail()}; mode {mode}', flush=True)
    R = S6.Rig(torch, a.coder, max_ctx=a.max_ctx)
    m, pf, S = R.m, R.pf, R.pf.S
    assert m.qsa_layer_bufs == hooked and (not hooked or len(m.qsa_score_l[0]) == NL), (m.qsa_layer_bufs, len(m.qsa_score_l[0]))
    jf = os.path.join(out, f'k3_{mode}.json')
    res = json.load(open(jf)) if os.path.exists(jf) else dict(mode=mode, max_ctx=m.max_ctx, depths={})   # (depths of earlier runs kept)
    res['compaction'] = comp()
    for N in a.depths:
        toks = tokens(os.path.join(LC, f'd{N}', 'ctx.ids')); lo, hi = N, N + NPOS; pos = list(range(lo, hi))
        nvmax = hi // 4; D = {}; t0 = time.perf_counter()
        print(f'== N={N}: MemAvailable {memavail()}', flush=True)

        def window(st, tag):
            cap = WindowCap(torch, m, lo, hi, nvmax, hooked)
            longctx.teacher_forced(m, st, toks, NPOS, rows=cap)
            return cap

        def save(tag, cap, scores=True):
            np.savez(os.path.join(out, f'k3_N{N}_{tag}{"" if hooked else "_plain"}.npz'), sel=cap.sel,
                     **({} if cap.score is None or not scores else dict(score=cap.score)), **({} if cap.q is None else dict(q=cap.q)))

        # dec: one-token decode from 0 (T = 4 teacher-forced passes, as G4a's reference)
        m.reset(); st = longctx.teacher_forced(m, dict(i0=0, commit=0), toks, N, rows=lambda i0, lo: None)[1]
        h_dec = idx_rows(torch, m, 0, N, N // 4)
        c_dec = window(st, 'dec'); lg_dec = c_dec.logits(); save('dec', c_dec)
        st_dec = idx_rows(torch, m, 0, hi, nvmax)
        rp_dec = repool_check(torch, m, S, hi) if not a.plain else None
        t1_ = time.perf_counter()
        # pd: the batched prefill of [0, N), then the window
        m.reset(); st = pf.run(toks, 0, 0, N)
        h_pd = idx_rows(torch, m, 0, N, N // 4)
        c_pd = window(dict(i0=st['i0'], commit=0), 'pd'); lg_pd = c_pd.logits(); save('pd', c_pd)
        r0_, r1_ = m.check_replicas()
        st_pd = idx_rows(torch, m, 0, hi, nvmax)
        rp_pd = repool_check(torch, m, S, hi) if not a.plain else None
        t2_ = time.perf_counter()
        lc_pd = np.memmap(os.path.join(S7G4, f'pd{N}_r1.bin'), dtype=np.float32, mode='r').reshape(NPOS, G.NV)
        lc_1t = np.memmap(os.path.join(S7G4, f'ref{N}.bin'), dtype=np.float32, mode='r').reshape(NPOS, G.NV)
        D['g4a'] = kstats(lg_dec, lg_pd, pos); D['g4b'] = kstats(lc_pd, lg_pd, pos)
        D['pd_vs_llama_1tok'] = kstats(lc_1t, lg_pd, pos); D['dec_vs_llama_pd'] = kstats(lc_pd, lg_dec, pos)
        D['dec_vs_llama_1tok'] = kstats(lc_1t, lg_dec, pos)
        D['sha'] = dict(dec=sha(lg_dec), pd=sha(lg_pd), sel_dec=sha(c_dec.sel), sel_pd=sha(c_pd.sel))
        D['replicas_pd'] = [bool(r0_), bool(r1_)]; D['sel_replicas'] = dict(dec=c_dec.sel_rep, pd=c_pd.sel_rep)
        for k in ('g4a', 'g4b'):
            print(f'  {k}: ' + G.line(k, D[k]) + f'; lead-g4d-ab {LEAD.get(N, {}).get(k)}; top5 {D[k]["top5"]}', flush=True)
        print(f'  sha dec {D["sha"]["dec"][:16]} pd {D["sha"]["pd"][:16]}; SEL replicas {D["sel_replicas"]}; replicas after pd {D["replicas_pd"]}; '
              f'dec {t1_ - t0:.0f} s, pd {t2_ - t1_:.0f} s', flush=True)
        if not a.plain:
            D['repool'] = dict(dec=rp_dec, pd=rp_pd)
            print(f'  re-pool self-check (XT): dec {[(r["layer"], r["differ_lt513"], r["differ_ge513"]) for r in rp_dec[0]]} err {rp_dec[1]}; '
                  f'pd {[(r["layer"], r["differ_lt513"], r["differ_ge513"]) for r in rp_pd[0]]} err {rp_pd[1]}', flush=True)
            np.savez(os.path.join(out, f'states_N{N}_dec.npz'), raw=st_dec[0], pool=st_dec[1])
            np.savez(os.path.join(out, f'states_N{N}_pd.npz'), raw=st_pd[0], pool=st_pd[1])
            # t1: pd's state, dec's indexer codes below N
            m.reset(); st = pf.run(toks, 0, 0, N); put_idx(torch, m, *h_dec)
            c = window(dict(i0=st['i0'], commit=0), 't1'); lg = c.logits(); save('t1', c, scores=False)
            D['t1_vs_dec'] = kstats(lg_dec, lg, pos); D['t1_vs_pd'] = kstats(lg_pd, lg, pos); del lg
            # t2: dec's state, pd's indexer codes below N
            m.reset(); st = longctx.teacher_forced(m, dict(i0=0, commit=0), toks, N, rows=lambda i0, lo: None)[1]; put_idx(torch, m, *h_pd)
            c = window(st, 't2'); lg = c.logits(); save('t2', c, scores=False)
            D['t2_vs_dec'] = kstats(lg_dec, lg, pos); D['t2_vs_pd'] = kstats(lg_pd, lg, pos); del lg
            for k in ('t1_vs_dec', 't2_vs_dec', 't1_vs_pd', 't2_vs_pd'):
                print(f'  {k}: ' + G.line(k, D[k]) + f'; top5 mean {D[k]["top5_mean"]:.4f} {D[k]["top5"]}', flush=True)
            # inb: the window as the last chunk of a prefill of [0, N + 256)
            cap = InbCap(S, lo, hi, nvmax)
            try:
                m.reset(); pf.run(toks, 0, 0, hi)
            finally: cap.close()
            R.sync()
            D['inb_seen'] = [int(cap.seen[..., 0].sum()), int(cap.seen[..., 1].sum())]
            np.savez(os.path.join(out, f'k3_N{N}_inb.npz'), sel=cap.sel, score=cap.score, q=cap.q)
            st_inb = idx_rows(torch, m, 0, hi, nvmax)
            np.savez(os.path.join(out, f'states_N{N}_inb.npz'), raw=st_inb[0], pool=st_inb[1])
            rp_inb = repool_check(torch, m, S, hi); D['repool']['inb'] = rp_inb
            print(f'  inb: rows captured (sel, q) {D["inb_seen"]} of {NPOS * NL}; re-pool (XT) '
                  f'{[(r["layer"], r["differ_lt513"], r["differ_ge513"]) for r in rp_inb[0]]} err {rp_inb[1]}', flush=True)
        del lg_dec, lg_pd, lc_pd, lc_1t
        res['depths'][str(N)] = {k: ({kk: vv for kk, vv in v.items() if kk != 'agree'} if isinstance(v, dict) and 'kld' in v else v)
                                 for k, v in D.items()}
        json.dump(res, open(jf, 'w'), indent=1, default=str)
        print(f'== N={N} done in {time.perf_counter() - t0:.0f} s; MemAvailable {memavail()}', flush=True)
    print('RESULT DONE', flush=True)


# ---- llama.cpp jobs -----------------------------------------------------------------------------------------------------
def cmd_jobs(a):
    d = os.path.join(a.out, 'llama'); os.makedirs(d, exist_ok=True)
    L = ','.join(map(str, QSA_LAYERS)); names = 'indexer_q,indexer_score,indexer_top_k'
    J = [f'nctx {max(a.depths) + NPOS}', 'ubatch 2048']
    for N in a.depths:
        c = os.path.join(LC, f'd{N}', 'ctx.ids'); save = 1 if N > 8192 else 0
        J.append(f'qsax dec{N} {c} {N} {NPOS} 1 {L} {names},indexer_k@last {save} {d}/dec{N}')
        J.append(f'qsax pd{N}_r1 {c} {N} {NPOS} 2048 {L} {names},indexer_k@last {save} {d}/pd{N}_r1')
        J.append(f'qsax pd{N}_r2 {c} {N} {NPOS} 2048 {L} {names} {save} {d}/pd{N}_r2')
        J.append(f'qsab inb{N} {c} {N} {NPOS} {L} {names},indexer_k {d}/inb{N}')
    jf = os.path.join(d, 'jobs_qsa.txt'); open(jf, 'w').write('\n'.join(J) + '\n'); print(jf)


# ---- comparison (CPU) ---------------------------------------------------------------------------------------------------
class Dump:
    """a qsax / qsab job's packed dumps"""
    def __init__(self, d):
        self.d = d; self.idx = {}
        for l in open(os.path.join(d, 'index.tsv')):
            r = l.rstrip('\n').split('\t')
            assert len(r) == 10, (d, r)   # (no short-write marker)
            self.idx[(int(r[0]), r[2])] = (r[3], [int(x) for x in r[4:8]], int(r[8]), int(r[9]))
        self.bin = np.memmap(os.path.join(d, 'dump.bin'), dtype=np.uint8, mode='r')

    def get(self, step, name):
        ty, ne, off, nb = self.idx[(step, name)]
        a = np.frombuffer(self.bin[off:off + nb], dtype=np.float32 if ty == 'f32' else np.int32)
        return a.reshape(ne[::-1]).squeeze()

    def steps(self):
        return [tuple(l.split('\t')[:4]) for l in open(os.path.join(self.d, 'steps.tsv')).read().splitlines() if l]


def ldir(run, N):
    """the llama job directory of a run name (cmd_jobs' names): dec -> dec{N}, pd_r1 -> pd{N}_r1, inb -> inb{N}"""
    return f'pd{N}_{run.split("_")[1]}' if run.startswith('pd_') else f'{run}{N}'


_RANKS = {}


def order_rank(s, nv):
    """rank of every pool j < nv in OP_QSA_SELECT's order (score descending, the lower index first among equal scores)"""
    k = np.asarray(s[:nv], np.float32)
    o = np.lexsort((np.arange(nv), -G_key(k)))
    r = np.empty(nv, np.int64); r[o] = np.arange(nv); return r, o


def G_key(k):
    from caminus import qsa
    return qsa.key_of(k).astype(np.int64)


def k3_sets(sel):
    """K3 SEL rows [n, 12, 516] -> (pool sets [n, 12] of sorted int arrays, nv [n])"""
    n = sel.shape[0]; S = np.empty((n, NL), object)
    for i in range(n):
        for l in range(NL):
            c = int(sel[i, l, 0]); S[i, l] = np.sort(sel[i, l, 4:4 + c])
    return S, sel[:, 0, 2].astype(np.int64)


def llama_sets(D, N, inb=False):
    """llama.cpp's indexer_top_k / indexer_score / indexer_q at the window positions -> sets, scores [n, 12, nvmax], q"""
    n = NPOS; nvmax = (N + NPOS) // 4; S = np.empty((n, NL), object)
    sc = np.full((n, NL, nvmax), -np.inf, np.float32); q = np.zeros((n, NL, QD), np.float32)
    for l, L in enumerate(QSA_LAYERS):
        if inb:
            tk = D.get(N, f'indexer_top_k-{L}'); s = D.get(N, f'indexer_score-{L}'); qq = D.get(N, f'indexer_q-{L}')
            for i in range(n):
                S[i, l] = np.sort(tk[i]); w = min(s.shape[1], nvmax); sc[i, l, :w] = s[i, :w]; q[i, l] = qq[i].reshape(-1)
        else:
            for i in range(n):
                p = N + i
                S[i, l] = np.sort(D.get(p, f'indexer_top_k-{L}')); s = D.get(p, f'indexer_score-{L}')
                w = min(s.shape[0], nvmax); sc[i, l, :w] = s[:w]; q[i, l] = D.get(p, f'indexer_q-{L}').reshape(-1)
    return S, sc, q


def pair(X, Y, ref_score, nv):
    """per (position, layer): dis = |X \\ Y|, the first-difference rank in ref_score's order (None if equal), the ranks
    of every differing pool"""
    n = X.shape[0]; dis = np.zeros((n, NL), np.int64); fdr = np.full((n, NL), -1, np.int64); ranks = []
    for i in range(n):
        for l in range(NL):
            x, y = X[i, l], Y[i, l]
            d = np.setxor1d(x, y)
            dis[i, l] = len(np.setdiff1d(x, y))
            if len(d) and ref_score is not None:
                key = (id(ref_score), i, l)
                if key not in _RANKS: _RANKS[key] = order_rank(ref_score[i, l], int(nv[i]))[0]
                r = _RANKS[key]
                rr = r[d[d < nv[i]]]; fdr[i, l] = int(rr.min()) if len(rr) else -1; ranks.extend(rr.tolist())
    return dis, fdr, np.array(ranks, np.int64)


def boundary(score, nv):
    """per (position, layer) of one run's scores: threshold (rank 511), gap to rank 512, ties at the threshold, visible
    zero scores"""
    n = score.shape[0]; out = dict(thr=np.zeros((n, NL)), gap=np.zeros((n, NL)), rel_gap=np.zeros((n, NL)),
                                   ties=np.zeros((n, NL), np.int64), zeros=np.zeros((n, NL), np.int64))
    for i in range(n):
        for l in range(NL):
            s = np.sort(score[i, l, :nv[i]])[::-1]
            thr, nxt = s[TOPK - 1], s[TOPK]
            out['thr'][i, l] = thr; out['gap'][i, l] = thr - nxt; out['rel_gap'][i, l] = (thr - nxt) / thr if thr > 0 else np.nan
            out['ties'][i, l] = int((s == thr).sum()); out['zeros'][i, l] = int((s == 0).sum())
    return out


def summ(x):
    x = np.asarray(x, np.float64)
    if x.size == 0: return dict(n=0)
    return dict(n=int(x.size), mean=float(np.nanmean(x)), median=float(np.nanmedian(x)), p99=float(np.nanpercentile(x, 99)),
                max=float(np.nanmax(x)), min=float(np.nanmin(x)))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64); rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def paired_boot(x, y, n=10000, seed=0):
    d = np.asarray(x, np.float64) - np.asarray(y, np.float64); rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def within(dO, dL):
    """the pre-registered test on per-position values dO / dL (the same positions)"""
    lo, hi = G.boot(dL); md, plo, phi = paired_boot(dO, dL)
    return dict(ours=float(np.mean(dO)), llama=float(np.mean(dL)), llama_boot95=[lo, hi], paired_diff=md, paired_ci=[plo, phi],
                a=bool(np.mean(dO) <= hi), b=bool(plo <= 0), within=bool(np.mean(dO) <= hi and plo <= 0))


def codes(pa, pb, n):
    """IDXPOOL rows [12, n, 136] of two runs: per layer the fraction of pools whose bytes differ, the mean differing q8_0
    codes (of 128) and scale halves (of 4) per differing pool"""
    out = []
    for l in range(NL):
        a, b = pa[l, :n].reshape(n, 4, 34), pb[l, :n].reshape(n, 4, 34)
        dq = (a[:, :, 2:] != b[:, :, 2:]).sum((1, 2)); dd = (a[:, :, :2] != b[:, :, :2]).any(2).sum(1)
        diff = (dq > 0) | (dd > 0)
        out.append(dict(layer=QSA_LAYERS[l], frac=float(diff.mean()), codes=float(dq[diff].mean()) if diff.any() else 0.0,
                        scales=float(dd[diff].mean()) if diff.any() else 0.0))
    return out


def host_scores(q, pool_rows, nv):
    """sum_h relu(q_h . dequant(pool_j)) / sqrt(128) on the host (numpy float32 matmul: NOT the device's order; used only to
    split a selection difference between its two inputs), q [n, 12, 512], pool rows [12, nvmax, 136] -> [n, 12, nvmax]"""
    from caminus import qsa
    n = q.shape[0]; nvmax = pool_rows.shape[1]; out = np.full((n, NL, nvmax), -np.inf, np.float32)
    for l in range(NL):
        P = qsa.dq_rows(pool_rows[l])                       # [nvmax, 128]
        d = np.einsum('nhd,jd->nhj', q[:, l].reshape(n, 4, 128), P, optimize=True)
        sc = (np.maximum(d, 0).sum(1) * np.float32(1 / np.sqrt(128.0))).astype(np.float32)
        for i in range(n): out[i, l, :nv[i]] = sc[i, :nv[i]]
    return out


def host_sets(score, nv):
    from caminus import qsa
    n = score.shape[0]; S = np.empty((n, NL), object)
    for i in range(n):
        for l in range(NL): S[i, l] = qsa.select_ref(score[i, l], int(nv[i]))[0]
    return S


def cmd_cmp(a):
    from caminus import qsa
    K = os.path.join(a.out, 'k3'); L = os.path.join(a.out, 'llama')
    kh = json.load(open(os.path.join(K, 'k3_hooked.json'))); kp = json.load(open(os.path.join(K, 'k3_plain.json'))) if os.path.exists(os.path.join(K, 'k3_plain.json')) else None
    rep = dict(depths={}); problems = []
    for N in a.depths:
        _RANKS.clear(); sN = str(N); H = kh['depths'][sN]; R = rep['depths'][sN] = {}; pos = list(range(N, N + NPOS)); nvmax = (N + NPOS) // 4
        print(f'===== N = {N}', flush=True)
        # controls: the hook changed nothing; G4 numbers as the lead's run
        if kp:
            P = kp['depths'][sN]; same = {k: H['sha'][k] == P['sha'][k] for k in H['sha']}
            R['hook_bitwise'] = same; print(f'  control: hooked == plain (logits / SEL sha256): {same}', flush=True)
            if not all(same.values()): problems.append(f'N={N}: the hooked run differs from the plain run {same}')
        R['g4'] = {k: dict(mean=H[k]['kld_mean'], p99=H[k]['kld_p99'], max=H[k]['kld_max'], top1=H[k]['top1'], top5=H[k]['top5'],
                           boot_mean=H[k]['boot_mean'], boot_p99=H[k]['boot_p99']) for k in ('g4a', 'g4b', 'pd_vs_llama_1tok', 'dec_vs_llama_pd', 'dec_vs_llama_1tok')}
        for k in ('g4a', 'g4b'):
            ok = abs(H[k]['kld_mean'] - LEAD[N][k]) < 5e-6
            print(f'  control: {k} mean {H[k]["kld_mean"]:.5f} vs lead-g4d-ab {LEAD[N][k]:.5f}: {"same" if ok else "DIFFERENT"}', flush=True)
            if not ok: problems.append(f'N={N}: {k} {H[k]["kld_mean"]:.5f} != lead {LEAD[N][k]}')
        # K3 selections
        ks = {}
        for run in ('dec', 'pd', 't1', 't2', 'inb'):
            z = np.load(os.path.join(K, f'k3_N{N}_{run}.npz'))
            S_, nv = k3_sets(z['sel']); ks[run] = dict(S=S_, nv=nv, score=z['score'] if 'score' in z else None, q=z['q'] if 'q' in z else None)
        nv = ks['dec']['nv']; assert np.array_equal(nv, np.array([(p + 1) // 4 for p in pos])), 'K3 SEL nv header'
        # llama.cpp selections
        ls = {}
        for run in ('dec', 'pd_r1', 'pd_r2', 'inb'):
            Dm = Dump(os.path.join(L, ldir(run, N)))
            S_, sc, q = llama_sets(Dm, N, inb=run == 'inb'); ls[run] = dict(S=S_, score=sc, q=q, dump=Dm)
        # llama controls and floor KLD from the dump runs: each dump run's rows vs the stored floor run of the same protocol
        fl = {}
        for run, stored in (('dec', f'ref{N}'), ('pd_r1', f'pd{N}_r1'), ('pd_r2', f'pd{N}_r2')):
            st = ls[run]['dump'].steps(); ref = G.tsv(os.path.join(S7G4, f'{stored}.tsv'))
            assert [x[0] for x in st] == [y[0] for y in ref] and [x[1] for x in st] == [y[1] for y in ref], (run, 'positions / tokens')
            fl[run] = dict(fnv=sum(1 for x, y in zip(st, ref) if x[3] != y[3]), argmax=sum(1 for x, y in zip(st, ref) if x[2] != y[2]))
        st1, st2 = ls['pd_r1']['dump'].steps(), ls['pd_r2']['dump'].steps()
        fl['pd_r2_vs_r1'] = dict(fnv=sum(1 for x, y in zip(st1, st2) if x[3] != y[3]), argmax=sum(1 for x, y in zip(st1, st2) if x[2] != y[2]))
        R['llama_fnv_vs_stored'] = fl
        print(f'  llama dump runs vs the stored floor runs (rows differing by FNV / argmax): {fl} (8K: 0 expected; 32K ub 2048 is not run-to-run '
              f'deterministic)', flush=True)
        if N == 8192 and (fl['dec']['fnv'] or fl['pd_r1']['fnv']): problems.append(f'N=8192: llama dump runs differ from the stored runs {fl}')
        if N > 8192:
            lg = {r: np.memmap(os.path.join(L, ldir(r, N), 'logits.bin'), dtype=np.float32, mode='r').reshape(NPOS, G.NV)
                  for r in ('dec', 'pd_r1', 'pd_r2')}
        else:
            lg = {'dec': np.memmap(os.path.join(S7G4, f'ref{N}.bin'), dtype=np.float32, mode='r').reshape(NPOS, G.NV),
                  'pd_r1': np.memmap(os.path.join(S7G4, f'pd{N}_r1.bin'), dtype=np.float32, mode='r').reshape(NPOS, G.NV)}
        fk = kstats(lg['dec'], lg['pd_r1'], pos); R['floor_dump_run'] = {k: fk[k] for k in ('kld_mean', 'kld_p99', 'kld_max', 'top1', 'top5', 'boot_mean', 'boot_p99')}
        if 'pd_r2' in lg:
            f2 = kstats(lg['dec'], lg['pd_r2'], pos); R['floor_dump_run_r2'] = {k: f2[k] for k in ('kld_mean', 'kld_p99', 'kld_max', 'top1', 'top5')}
            # the floor's own spread: every pairing of this campaign's and the stored S7 runs (the stored floor: S7's g4_depth.json)
            sref = np.memmap(os.path.join(S7G4, f'ref{N}.bin'), dtype=np.float32, mode='r').reshape(NPOS, G.NV)
            spd = np.memmap(os.path.join(S7G4, f'pd{N}_r1.bin'), dtype=np.float32, mode='r').reshape(NPOS, G.NV)
            R['floor_spread'] = {}
            for nm, (x, y) in (('stored_ref_vs_stored_pd', (sref, spd)), ('stored_ref_vs_new_pd_r1', (sref, lg['pd_r1'])),
                               ('stored_ref_vs_new_pd_r2', (sref, lg['pd_r2'])), ('new_dec_vs_stored_pd', (lg['dec'], spd)),
                               ('new_dec_vs_new_pd_r2', (lg['dec'], lg['pd_r2'])), ('stored_ref_vs_new_dec', (sref, lg['dec']))):
                f_ = kstats(x, y, pos); R['floor_spread'][nm] = {k: f_[k] for k in ('kld_mean', 'kld_p99', 'kld_max', 'top1', 'top5', 'boot_mean', 'boot_p99')}
                print(f'  floor spread {nm:26s}: mean {f_["kld_mean"]:.5f} p99 {f_["kld_p99"]:.4f} max {f_["kld_max"]:.4f} top-1 {f_["top1"]:.2f} '
                      f'top5 {[(p_, round(k_, 4)) for p_, k_ in f_["top5"][:3]]}', flush=True)
        print('  ' + G.line('llama floor (dump run pd_r1 vs dec)', fk) + f'; top5 {fk["top5"]}; boot p99 {fk["boot_p99"]}', flush=True)
        g4a_k = np.array(H['g4a']['kld']); fl_k = np.array(fk['kld'])
        print(f'  G4a p99 {H["g4a"]["kld_p99"]:.4f} boot {H["g4a"]["boot_p99"]}; top5 {H["g4a"]["top5"]}', flush=True)
        # pairs
        rk, rl = ks['dec']['score'], ls['dec']['score']
        PAIRS = {'ours_pd_vs_dec': (ks['pd']['S'], ks['dec']['S'], rk), 'ours_inb_vs_dec': (ks['inb']['S'], ks['dec']['S'], rk),
                 'ours_t1_vs_dec': (ks['t1']['S'], ks['dec']['S'], rk), 'ours_t2_vs_dec': (ks['t2']['S'], ks['dec']['S'], rk),
                 'ours_inb_vs_pd': (ks['inb']['S'], ks['pd']['S'], rk),
                 'llama_pd_vs_dec': (ls['pd_r1']['S'], ls['dec']['S'], rl), 'llama_pd_r2_vs_dec': (ls['pd_r2']['S'], ls['dec']['S'], rl),
                 'llama_pd_r2_vs_r1': (ls['pd_r2']['S'], ls['pd_r1']['S'], rl), 'llama_inb_vs_dec': (ls['inb']['S'], ls['dec']['S'], rl),
                 'ours_dec_vs_llama_dec': (ks['dec']['S'], ls['dec']['S'], rl), 'ours_pd_vs_llama_pd': (ks['pd']['S'], ls['pd_r1']['S'], ls['pd_r1']['score']),
                 'ours_inb_vs_llama_inb': (ks['inb']['S'], ls['inb']['S'], ls['inb']['score'])}
        PR = {}; R['pairs'] = {}
        for name, (X, Y, sc) in PAIRS.items():
            dis, fdr, ranks = pair(X, Y, sc, nv); PR[name] = (dis, fdr, ranks)
            dpos = dis.mean(1)
            R['pairs'][name] = dict(dis=summ(dis), dis_pos=summ(dpos), frac_identical=float((dis == 0).mean()), worst=int(dis.max()),
                                    worst_at=[int(pos[np.unravel_index(dis.argmax(), dis.shape)[0]]), QSA_LAYERS[np.unravel_index(dis.argmax(), dis.shape)[1]]],
                                    per_layer=[float(x) for x in dis.mean(0)], fdr=summ(fdr[fdr >= 0]), diff_ranks=summ(ranks),
                                    diff_ranks_below_448=int((ranks < 448).sum()), n_diff=int(len(ranks)))
            r = R['pairs'][name]
            print(f'  {name:24s} dis mean {r["dis"]["mean"]:.3f} (per pos max {r["dis_pos"]["max"]:.2f}), identical {100 * r["frac_identical"]:.1f} %, '
                  f'worst {r["worst"]} at {r["worst_at"]}; first-diff rank median {r["fdr"].get("median")} min {r["fdr"].get("min")}; '
                  f'differing pools\' ranks median {r["diff_ranks"].get("median")}, < 448: {r["diff_ranks_below_448"]} of {r["n_diff"]}', flush=True)
        # THE WITHIN TEST
        dO, dL = PR['ours_pd_vs_dec'][0], PR['llama_pd_vs_dec'][0]
        W = within(dO.mean(1), dL.mean(1)); W['per_layer'] = []
        for l in range(NL):
            w = within(dO[:, l], dL[:, l]); w['layer'] = QSA_LAYERS[l]; w['flag'] = bool(w['paired_ci'][0] > 0); W['per_layer'].append(w)
        W['in_batch'] = within(PR['ours_inb_vs_dec'][0].mean(1), PR['llama_inb_vs_dec'][0].mean(1))
        R['within'] = W
        print(f'  WITHIN (PD): ours mean D {W["ours"]:.3f} vs llama {W["llama"]:.3f} (boot95 {W["llama_boot95"][0]:.3f} .. {W["llama_boot95"][1]:.3f}); '
              f'paired diff {W["paired_diff"]:.3f} CI {W["paired_ci"][0]:.3f} .. {W["paired_ci"][1]:.3f}: (a) {W["a"]} (b) {W["b"]} -> within {W["within"]}', flush=True)
        print('    per layer (ours / llama / flag): ' + ' '.join(f'L{w["layer"]}:{w["ours"]:.2f}/{w["llama"]:.2f}{"!" if w["flag"] else ""}' for w in W['per_layer']), flush=True)
        print(f'  in-batch (reported): ours {W["in_batch"]["ours"]:.3f} vs llama {W["in_batch"]["llama"]:.3f}, paired CI {W["in_batch"]["paired_ci"]}', flush=True)
        # EXPLAINED: KLD vs disagreement
        Do = dO.mean(1); Dl = dL.mean(1)
        top = np.argsort(-g4a_k)[:5]; pct = [float((Do <= Do[i]).mean() * 100) for i in top]
        ftop = np.argsort(-fl_k)[:5]; fpct = [float((Dl <= Dl[i]).mean() * 100) for i in ftop]
        R['explained'] = dict(spearman_g4a=spearman(g4a_k, Do), spearman_floor=spearman(fl_k, Dl),
                              g4a_top5=[dict(pos=pos[i], kld=float(g4a_k[i]), D=float(Do[i]), D_pct=pct[k], per_layer=dO[i].tolist()) for k, i in enumerate(top)],
                              floor_top5=[dict(pos=pos[i], kld=float(fl_k[i]), D=float(Dl[i]), D_pct=fpct[k], per_layer=dL[i].tolist()) for k, i in enumerate(ftop)])
        E = R['explained']
        print(f'  EXPLAINED: Spearman(G4a KLD, D_ours) {E["spearman_g4a"]:.3f}; Spearman(floor KLD, D_llama) {E["spearman_floor"]:.3f}', flush=True)
        for t in E['g4a_top5']: print(f'    G4a top: pos {t["pos"]} KLD {t["kld"]:.4f} D_ours {t["D"]:.2f} (pct {t["D_pct"]:.0f}) per layer {t["per_layer"]}', flush=True)
        for t in E['floor_top5']: print(f'    floor top: pos {t["pos"]} KLD {t["kld"]:.4f} D_llama {t["D"]:.2f} (pct {t["D_pct"]:.0f}) per layer {t["per_layer"]}', flush=True)
        # CAUSE: the transplant
        base = H['g4a']['top5_mean']; t1m, t2m = H['t1_vs_dec']['top5_mean'], H['t2_vs_dec']['top5_mean']
        cause = 'codes' if t1m <= 0.5 * base and t2m >= 0.5 * base else ('state' if t1m >= 0.5 * base and t2m < 0.5 * base else 'mixed')
        R['cause'] = dict(pd_top5=base, t1_top5=t1m, t2_top5=t2m, t1=dict(mean=H['t1_vs_dec']['kld_mean'], p99=H['t1_vs_dec']['kld_p99'], top5=H['t1_vs_dec']['top5']),
                          t2=dict(mean=H['t2_vs_dec']['kld_mean'], p99=H['t2_vs_dec']['kld_p99'], top5=H['t2_vs_dec']['top5']), verdict=cause)
        print(f'  CAUSE (transplant, top-5 mean KLD vs dec): pd {base:.4f}, t1 (pd state + dec codes) {t1m:.4f}, t2 (dec state + pd codes) {t2m:.4f} -> {cause}', flush=True)
        # boundaries
        R['boundary'] = {}
        for name, sc in (('k3_dec', ks['dec']['score']), ('k3_pd', ks['pd']['score']), ('k3_inb', ks['inb']['score']),
                         ('llama_dec', ls['dec']['score']), ('llama_pd', ls['pd_r1']['score']), ('llama_inb', ls['inb']['score'])):
            b = boundary(sc, nv)
            R['boundary'][name] = dict(thr=summ(b['thr']), gap=summ(b['gap']), rel_gap=summ(b['rel_gap']), ties_gt1=float((b['ties'] > 1).mean()),
                                       ties_max=int(b['ties'].max()), thr_zero=float((b['thr'] == 0).mean()), zeros=summ(b['zeros']))
            r = R['boundary'][name]
            print(f'  boundary {name:9s}: thr median {r["thr"]["median"]:.4g}, gap median {r["gap"]["median"]:.3g} (rel {r["rel_gap"]["median"]:.2e}), '
                  f'ties > 1 at the threshold {100 * r["ties_gt1"]:.2f} % (max {r["ties_max"]}), threshold 0 in {100 * r["thr_zero"]:.2f} %, zeros median {r["zeros"]["median"]}', flush=True)
        # localization: codes and queries
        sd, sp, si = (np.load(os.path.join(K, f'states_N{N}_{r}.npz')) for r in ('dec', 'pd', 'inb'))
        R['codes'] = dict(pd_vs_dec=codes(sp['pool'], sd['pool'], N // 4), inb_vs_dec=codes(si['pool'], sd['pool'], N // 4),
                          raw_rows_differ_pd_vs_dec=[float((sp['raw'][l, :N] != sd['raw'][l, :N]).any(1).mean()) for l in range(NL)])
        # llama's pooled keys at the last step (dequantized q8_0) -> bytes, vs ours
        for run, k3run, step in (('dec', sd, N + NPOS - 1), ('pd_r1', sp, N + NPOS - 1)):
            Dm = ls[run]['dump']; out = []
            for l, Lr in enumerate(QSA_LAYERS):
                kk = Dm.get(step, f'indexer_k-{Lr}').reshape(-1, 128)[:N // 4]
                try: lb = qsa.requant_rows(kk)
                except AssertionError: out.append(None); continue
                out.append(float((lb != k3run['pool'][l, :N // 4]).any(1).mean()))
            R['codes'][f'ours_vs_llama_{run}'] = out
        # llama.cpp's own pd vs dec pooled codes (its indexer_k at the last step, requantized), as ours' pd_vs_dec above
        lp = {}
        for run in ('dec', 'pd_r1'):
            Dm = ls[run]['dump']; lp[run] = np.stack([qsa.requant_rows(Dm.get(N + NPOS - 1, f'indexer_k-{Lr}').reshape(-1, 128)[:N // 4]) for Lr in QSA_LAYERS])
        R['codes']['llama_pd_vs_dec'] = codes(lp['pd_r1'], lp['dec'], N // 4)
        print(f'  codes: llama pools differing pd vs dec per layer {[round(c["frac"], 3) for c in R["codes"]["llama_pd_vs_dec"]]} (codes per differing pool '
              f'{[round(c["codes"], 1) for c in R["codes"]["llama_pd_vs_dec"]]})', flush=True)
        print(f'  codes: pools differing pd vs dec per layer {[round(c["frac"], 3) for c in R["codes"]["pd_vs_dec"]]} (codes per differing pool '
              f'{[round(c["codes"], 1) for c in R["codes"]["pd_vs_dec"]]}); raw rows differing {[round(x, 3) for x in R["codes"]["raw_rows_differ_pd_vs_dec"]]}', flush=True)
        print(f'  codes: ours vs llama pools differing (dec / pd) {R["codes"]["ours_vs_llama_dec"]} / {R["codes"]["ours_vs_llama_pd_r1"]}', flush=True)
        qd = {}
        for name, x, y in (('ours_pd_vs_dec', ks['pd']['q'], ks['dec']['q']), ('ours_inb_vs_dec', ks['inb']['q'], ks['dec']['q']),
                           ('ours_dec_vs_llama_dec', ks['dec']['q'], ls['dec']['q']), ('llama_pd_vs_dec', ls['pd_r1']['q'], ls['dec']['q'])):
            rel = np.linalg.norm(x - y, axis=2) / np.maximum(np.linalg.norm(y, axis=2), 1e-30)
            qd[name] = dict(rel=summ(rel), exact=float((rel == 0).mean()))
        R['query'] = qd
        print('  query rel. L2 differences: ' + '; '.join(f'{k} median {v["rel"]["median"]:.2e} p99 {v["rel"]["p99"]:.2e} exact {100 * v["exact"]:.1f} %' for k, v in qd.items()), flush=True)
        # split ours' PD-vs-dec selection difference between its inputs (host recompute: the query rows of the run x the
        # pools of the run y; the window's own pools j >= N / 4 come from each run's decode)
        hs = {}
        for qa, pb in (('dec', 'dec'), ('pd', 'pd'), ('pd', 'dec'), ('dec', 'pd')):
            hs[(qa, pb)] = host_scores(ks[qa]['q'], {'dec': sd, 'pd': sp}[pb]['pool'], nv)
        dev_err = float(np.nanmax(np.abs(hs[('dec', 'dec')] - np.where(np.isfinite(hs[('dec', 'dec')]), ks['dec']['score'], -np.inf))[np.isfinite(hs[('dec', 'dec')])]))
        HS = {k: host_sets(v, nv) for k, v in hs.items()}
        base_h = pair(HS[('dec', 'dec')], ks['dec']['S'], None, nv)[0]
        R['split'] = dict(host_vs_device_max_abs=dev_err, host_dec_vs_device_dec=summ(base_h),
                          query_only=summ(pair(HS[('pd', 'dec')], HS[('dec', 'dec')], None, nv)[0]),
                          pools_only=summ(pair(HS[('dec', 'pd')], HS[('dec', 'dec')], None, nv)[0]),
                          both=summ(pair(HS[('pd', 'pd')], HS[('dec', 'dec')], None, nv)[0]))
        sp_ = R['split']
        print(f'  split (host recompute; max |host - device| score {dev_err:.2e}, host dec vs device dec dis mean {sp_["host_dec_vs_device_dec"]["mean"]:.3f}): '
              f'pd query + dec pools {sp_["query_only"]["mean"]:.3f}, dec query + pd pools {sp_["pools_only"]["mean"]:.3f}, both {sp_["both"]["mean"]:.3f}', flush=True)
        R['repool'] = H.get('repool')
    rep['problems'] = problems
    if a.json: json.dump(rep, open(a.json, 'w'), indent=1, default=str)
    print('problems: ' + ('; '.join(problems) if problems else 'none'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    k = sub.add_parser('k3'); k.add_argument('coder'); k.add_argument('--out', required=True)
    k.add_argument('--depths', default='8192,32768'); k.add_argument('--max-ctx', type=int, default=33024); k.add_argument('--plain', action='store_true')
    j = sub.add_parser('jobs'); j.add_argument('out'); j.add_argument('--depths', default='8192,32768')
    c = sub.add_parser('cmp'); c.add_argument('out'); c.add_argument('--depths', default='8192,32768'); c.add_argument('--json', default=None)
    a = ap.parse_args(); a.depths = [int(x) for x in a.depths.split(',')]
    {'k3': cmd_k3, 'jobs': cmd_jobs, 'cmp': cmd_cmp}[a.cmd](a)


if __name__ == '__main__':
    main()
