"""K3 on the long-context probes of tools/eval/lc_probes.py, against llama.cpp's references (gg_golden --lc).
Default: the full-attention BASELINE (dense_beyond_qsa=True, a test-only override: past 2051 positions it is not the model's
QSA). --qsa: the production engine instead (FlashNextTP with max_ctx > 2051: the model's QSA attention, caminus.engine.flashnext_tp;
results in OUT/k3_qsa.json).
  guard.sh LOG SECS python -I tools/eval/lc_baseline.py ramp [--max 32768] [--T 4]
      ms per T-token pass at depths 0 .. 32K (no fill: zero KV rows, the same work); stops before a depth whose projected pass
      passes 50 ms
  guard.sh LOG SECS python -I tools/eval/lc_baseline.py run OUT [--depths 1792,8192,32768] [--qsa] [--kv-host]
      per depth: ONE fill of the context (caminus.decode.longctx.fill: T=4 passes, progress + tok/s), snapshot; then from the
      snapshot each probe (restore before each): teacher-forced logits at D .. D + 255 vs llama.cpp's (KLD float64 over the
      full vocabulary, top-1 agreement; llama.cpp's run 1 vs run 2 as its own drift), and the six greedy answers (24 tokens)
      vs llama.cpp's (copy found, first divergence). Writes OUT/k3_baseline.json and prints a table.
Passes are policed by caminus.engine.passpolice (attached when the engine is built): the per-pass limit applies to the time net
of the KFD queue eviction provably inside the pass's launches, a bail no eviction explains stops the run, and a POLICE
summary (evicted passes, evicted ms, longest pass raw and net, compaction_proactiveness) is printed at exit."""
import atexit, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, HERE)
import numpy as np

from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
MODEL = devpaths.coder()
NV = 248320
D281_KLD, D281_TOP1 = 0.0076, 0.959   # the Coder's D-281 floor / top-1 bar at short context (M1 G4)
MAX_PASS_MS = 100.0   # stop when a pass takes longer net of evictions (launches stay well under the 200 ms wait bound) or a bail is unexplained


def opt(a, name, default, f=str):
    if name in a:
        i = a.index(name); v = f(a[i + 1]); del a[i:i + 2]; return v
    return default


def engine(max_ctx, T=4, use_qsa=False, kv_host=False):
    import torch
    from caminus.engine import passpolice
    from caminus.engine.flashnext_tp import FlashNextTP
    for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)
    t0 = time.time()
    kw = dict(kv_host=True) if kv_host else {}   # (--kv-host: the KV cache in host memory; the results must not change)
    if use_qsa: m = FlashNextTP(MODEL, ntok=T, max_ctx=max(max_ctx, 2056), qsa=True, **kw)   # the model's QSA attention
    else: m = FlashNextTP(MODEL, ntok=T, max_ctx=max_ctx, dense_beyond_qsa=True, **kw)       # TEST-ONLY: full attention past 2051
    pol = passpolice.of(m); atexit.register(lambda: print(pol.summary(), flush=True))     # before the first pass
    print(f'engine: max_ctx {m.max_ctx}, T={T}, {"QSA (modules " + str([e.module for e in m.engines]) + ")" if m.qsa else "dense_beyond_qsa (test only)"}, '
          f'loaded in {time.time() - t0:.1f} s; vram plan fixed {m.vram_plan["fixed_mib"]} + ctx {m.vram_plan["ctx_mib"]:.0f} MiB per card', flush=True)
    return m


def ramp(a):
    import torch
    mx = opt(a, '--max', 32768, int); T = opt(a, '--T', 4, int)
    m = engine(mx + 8, T)
    toks = [248044, 760, 22414, 5285, 315, 13, 11, 13] * ((mx + 16) // 8 + 1)
    pts = []
    for X in [0, 2044, 4096, 8192, 16384, 32768]:
        if X > mx: break
        if len(pts) >= 2:   # linear in depth (dense attention: a serial O(pos) walk per head task): project from the last two
            (x0, y0), (x1, y1) = pts[-2:]; proj = y1 + (y1 - y0) / (x1 - x0) * (X - x1)
            if proj > 50: print(f'depth {X}: projected {proj:.1f} ms > 50 ms per pass: stop'); break
        m.reset(); ts = []
        for r in range(6):
            torch.cuda.synchronize(0); torch.cuda.synchronize(1); t0 = time.perf_counter()
            m.verify_tokens(toks, X - X % T if X + T > mx + 8 else X, 0); ts.append((time.perf_counter() - t0) * 1e3)
        med = float(np.median(ts[1:]))
        print(f'depth {X:>6}: T={T} pass {med:.2f} ms (min {min(ts[1:]):.2f}, max {max(ts[1:]):.2f}); bails {len(m.bails)}', flush=True)
        pts.append((X, med))


def kld_top1(ref, x):
    """per row: KL(ref || x) in float64 over the full vocabulary, and argmax agreement (first index on ties)"""
    r = ref.astype(np.float64); q = x.astype(np.float64)
    lr = r - r.max(1, keepdims=True); lr -= np.log(np.exp(lr).sum(1, keepdims=True))
    lq = q - q.max(1, keepdims=True); lq -= np.log(np.exp(lq).sum(1, keepdims=True))
    kl = (np.exp(lr) * (lr - lq)).sum(1)
    return kl, (r.argmax(1) == q.argmax(1))


def run(a):
    import torch
    from caminus.decode import longctx
    from caminus.engine import passpolice
    import lc_probes as L
    use_qsa = '--qsa' in a
    if use_qsa: a.remove('--qsa')
    kv_host = '--kv-host' in a
    if kv_host: a.remove('--kv-host')
    out = a[0]; meta = json.load(open(os.path.join(out, 'probes.json')))
    want = opt(a, '--depths', None)
    res_name = 'k3_qsa.json' if use_qsa else 'k3_baseline.json'
    depths = [d for d in meta['depths'] if want is None or str(d['D']) in want.split(',')]
    need = max(max(d['D'] + meta['ntf'], max(p['P'] + meta['ngen'] for p in d['probes'])) for d in depths)
    m = engine((need + 8 + 255) // 256 * 256, use_qsa=use_qsa, kv_host=kv_host)
    w0 = [int(x) for x in open(os.path.join(out, f'd{depths[0]["D"]}', 'ctx.ids')).read().split()]
    for _ in range(3): m.verify_tokens(w0, 0, 0)   # warm-up (first-launch costs off the policed passes); fill resets
    res = dict(model=MODEL, T=m.ntok, max_ctx=m.max_ctx, qsa=m.qsa, depths=[])
    for d in depths:
        D = d['D']; ref = os.path.join(out, 'ref')
        ctx = [int(x) for x in open(os.path.join(out, f'd{D}', 'ctx.ids')).read().split()]
        print(f'== depth {D}', flush=True)
        st = longctx.fill(m, ctx, D, every=15.0, out=sys.stdout, max_pass_ms=MAX_PASS_MS)
        print(f'fill {D}: {st["seconds"]:.1f} s, {st["tok_s"]:.0f} tok/s, {st["passes"]} passes, longest pass {st["max_pass_ms"]:.1f} ms '
              f'({st["max_pass_net_ms"]:.1f} ms net of evictions; {st["evicted_passes"]} evicted passes, {st["evicted_ms"]} ms), '
              f'bails {len(m.bails)}', flush=True)
        t0 = time.perf_counter(); snap = longctx.snapshot(m, st); t_snap = time.perf_counter() - t0
        r = dict(D=D, fill_s=st['seconds'], fill_tok_s=st['tok_s'], fill_max_pass_ms=st['max_pass_ms'], fill_max_pass_net_ms=st['max_pass_net_ms'],
                 fill_evicted_passes=st['evicted_passes'], fill_evicted_ms=st['evicted_ms'], police=st['police'],
                 compaction_proactiveness=passpolice.proactiveness(), snapshot_mib=snap['bytes'] / 2**20, snapshot_s=t_snap, probes={})
        # teacher forced
        s2 = longctx.restore(m, snap)
        t0 = time.perf_counter(); lo, _ = longctx.teacher_forced(m, s2, ctx, meta['ntf'], max_pass_ms=MAX_PASS_MS); t_tf = time.perf_counter() - t0
        x = lo.numpy()
        r1 = np.fromfile(os.path.join(ref, f'tf{D}_r1.bin'), dtype=np.float32).reshape(-1, NV)
        r2 = np.fromfile(os.path.join(ref, f'tf{D}_r2.bin'), dtype=np.float32).reshape(-1, NV)
        kl, t1 = kld_top1(r1, x); kd, td = kld_top1(r1, r2)
        r['tf'] = dict(n=len(kl), kld_mean=float(kl.mean()), kld_max=float(kl.max()), top1=float(t1.mean()),
                       llama_drift_kld_mean=float(kd.mean()), llama_drift_kld_max=float(kd.max()), llama_drift_top1=float(td.mean()),
                       ms_per_pos=1e3 * t_tf / len(kl), kld_by_64=[float(kl[i:i + 64].mean()) for i in range(0, len(kl), 64)])
        print(f'tf {D}: K3 vs llama.cpp mean KLD {kl.mean():.5f} (max {kl.max():.4f}), top-1 {t1.mean():.1%}; llama.cpp r1 vs r2: '
              f'mean KLD {kd.mean():.6f}, top-1 {td.mean():.1%}; {1e3 * t_tf / len(kl):.2f} ms/position', flush=True)
        del r1, r2, x, lo
        for p in d['probes']:
            toks = [int(v) for v in open(os.path.join(out, f'd{D}', f'{p["name"]}.ids')).read().split()]
            s2 = longctx.restore(m, snap)
            nu = len(passpolice.of(m).unexplained_bails())
            t0 = time.perf_counter(); g, _ = longctx.generate(m, s2, toks, meta['ngen']); t_g = time.perf_counter() - t0
            un = passpolice.of(m).unexplained_bails()
            if len(un) > nu: raise RuntimeError(f'probe {p["name"]}: the engine bailed and no KFD eviction explains it ({un[nu:]}): stopped')
            gk = g[len(toks):]
            gl = [int(v) for v in open(os.path.join(ref, f'{p["name"]}_{D}.tokens')).read().split()]
            tk, tl = L.decode(gk), L.decode(gl)
            div = next((i for i, (u, v) in enumerate(zip(gk, gl)) if u != v), None)
            gap = None
            if div is not None:   # llama.cpp's own top-1 - top-2 logit gap where the answers part (a near-tie: a diagnostic)
                row = open(os.path.join(ref, f'{p["name"]}_{D}.tsv')).read().split('\n')[p['P'] - 1 - D + div].split('\t')
                tops = [(int(x.split(':')[0]), float(x.split(':')[1])) for x in row[3:]]
                gap = dict(gap=tops[0][1] - tops[1][1], k3_token_rank_in_llama=next((i for i, (t, _) in enumerate(tops) if t == gk[div]), None))
            r['probes'][p['name']] = dict(answer=p['answer'], item_pos=p['item_pos'], k3_found=tk.startswith(p['answer']),
                                          llama_found=tl.startswith(p['answer']), same_as_llama=gk == gl, first_divergence=div, divergence_gap=gap,
                                          k3_text=tk, llama_text=tl, seconds=t_g)
            print(f'  {p["name"]:<16} copy K3 {"yes" if tk.startswith(p["answer"]) else "NO ":<3} llama.cpp {"yes" if tl.startswith(p["answer"]) else "NO ":<3} '
                  f'== llama.cpp {gk == gl} (first divergence {div}{"" if gap is None else f", llama.cpp gap there {gap[chr(103) + "ap"]:.3f}, K3 token its #{gap[chr(107) + "3_token_rank_in_llama"]}"})  K3 {tk[:28]!r}', flush=True)
        res['depths'].append(r)
        json.dump(res, open(os.path.join(out, res_name), 'w'), indent=1)
        if D + meta['ntf'] <= 2051:   # a control depth (dense == QSA): the D-281 bar and llama.cpp's answers, or the harness is wrong
            # (greedy-identical continuations are a diagnostic, as in D-281: past the copy they can part at near-ties)
            ok = r['tf']['kld_mean'] <= D281_KLD and r['tf']['top1'] >= D281_TOP1 and all(p['k3_found'] == p['llama_found'] for p in r['probes'].values())
            print(f'control depth {D}: ' + ('PASS' if ok else 'FAIL') + f' (mean KLD <= {D281_KLD}, top-1 >= {D281_TOP1:.1%}, copy found == llama.cpp '
                  f'on every probe)', flush=True)
            if not ok: print('STOP: the control fails, deeper numbers would mean nothing'); sys.exit(1)
    print('bails', len(m.bails))
    print('SUMMARY')
    for r in res['depths']:
        pr = r['probes'].values()
        print(f'D={r["D"]:>6}: fill {r["fill_s"]:.1f} s ({r["fill_tok_s"]:.0f} tok/s); tf mean KLD {r["tf"]["kld_mean"]:.5f} top-1 '
              f'{r["tf"]["top1"]:.1%} (llama.cpp drift {r["tf"]["llama_drift_kld_mean"]:.6f} / {r["tf"]["llama_drift_top1"]:.1%}); copies K3 '
              f'{sum(p["k3_found"] for p in pr)}/6, llama.cpp {sum(p["llama_found"] for p in pr)}/6, answers == llama.cpp '
              f'{sum(p["same_as_llama"] for p in pr)}/6')


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a: print(__doc__); sys.exit(2)
    {'ramp': ramp, 'run': run}[a[0]](a[1:])
