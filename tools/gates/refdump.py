"""Bitwise reference dumps of the TP decode (FlashNextTP, ntok=1) and their comparison: the gate for any change that must
leave target decode numerics bitwise (CLAUDE.md, bitwise discipline). Teacher forced over the first N positions of a token
file; per position: the full logits (cpu float32), both cards' router ids and check_replicas().
  guard.sh LOG SECS python -I tools/gates/refdump.py save OUT.pt [--model coder|reap|PATH] [--tokens FILE] [--n 24]
                                                         [--per-wgp N] [--max-ctx N] [--qsa 0|1] [--tape dual|always]
  python -I tools/gates/refdump.py cmp A.pt B.pt      CPU only: per position logits bitwise (int32 view: NaN-safe, -0 != 0),
                                                         router ids equal, replicas equal in both; exit 0 only if all are
save refuses an OUT that exists: a stored reference is never overwritten (new dumps go to a fresh directory).
--per-wgp N sets K3_PER_WGP before the engine is built (N workgroups per WGP: the race check, 3 / 6 / 7);
--max-ctx 4096 builds the Coder's QSA engine (two tapes; K3_QSA_TAPE or --tape picks dual, the default, or always).
Without --max-ctx / --qsa / --tape the engine is FlashNextTP(model) with its defaults (max_ctx 2048: the trunk).
Ported from the job-scratch gates of 11-01 .. 11-03 (tmp/m1/s3/refdump_coder.py: the Coder trunk; tmp/ceiling/refdump.py
and tmp/m1/s6/verify/refdump_reap.py: REAP-308; tmp/qsa/sel/refdump_qsa.py: the Coder's QSA engine): the same engine
calls in the same order and the same .pt layout (plus a 'meta' entry), so old and new dumps compare with cmp.
Current references and the expected results: $ARTIFACTS/refs/MANIFEST.md; the runner is tools/gates/gates.sh."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def save(a):
    out_path = a.pop(0)
    model = C.model_path(C.opt(a, '--model', 'coder')); tok_path = C.opt(a, '--tokens', C.TOKENS)
    n = C.opt(a, '--n', 24, int); per_wgp = C.opt(a, '--per-wgp', None, int)
    kw = C.engine_kwargs(C.opt(a, '--max-ctx', None, int), C.opt(a, '--qsa', None), C.opt(a, '--tape', None))
    if a: sys.exit(f'refdump save: unknown arguments {a}')
    if os.path.exists(out_path): sys.exit(f'refdump save: {out_path} exists: refusing to overwrite (write new dumps to a fresh path)')
    if per_wgp: os.environ['K3_PER_WGP'] = str(per_wgp)
    torch = C.setup_torch()
    from caminus.flashnext_tp import FlashNextTP
    toks = C.tokens(tok_path)
    print(f'model {model}; tokens {tok_path} ({len(toks)}); n {n}; per_wgp {per_wgp}; engine kwargs {kw}; K3_QSA_TAPE '
          f'{os.environ.get("K3_QSA_TAPE")}; compaction_proactiveness {C.proactiveness()}', flush=True)
    t0 = time.time(); m = FlashNextTP(model, **kw)
    print(f'qsa {m.qsa} max_ctx {m.max_ctx} modules {[e.module for e in m.engines]}; loaded in {time.time() - t0:.1f} s; grids '
          f'{[e.grid for e in m.engines]}; segments {[len(e.segs) for e in m.engines]}', flush=True)
    name = os.path.splitext(os.path.basename(out_path))[0]
    out = dict(name=name, per_wgp=per_wgp, grids=[e.grid for e in m.engines], logits=[], ids0=[], ids1=[], replicas=[],
               meta=dict(model=model, tokens=tok_path, n=n, engine_kwargs=kw, k3_qsa_tape=os.environ.get('K3_QSA_TAPE'),
                         qsa=m.qsa, max_ctx=m.max_ctx, modules=[e.module for e in m.engines], tool='tools/gates/refdump.py'))
    for i in range(n):
        lo = m.step(toks, i)
        out['logits'].append(lo.float().cpu().clone())
        out['ids0'].append(m.S[0]['ids'].cpu().clone()); out['ids1'].append(m.S[1]['ids'].cpu().clone())
        out['replicas'].append(m.check_replicas())
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.exists(out_path): sys.exit(f'refdump save: {out_path} appeared during the run: refusing to overwrite')
    torch.save(out, out_path)
    bad = [i for i, r in enumerate(out['replicas']) if not all(r)]
    print(f'saved {out_path}: {n} positions, replicas all equal: {not bad}' + (f' (bad at {bad})' if bad else ''))
    return not bad


def cmp(a):
    import torch
    A, B = a
    x, y = (torch.load(p) for p in (A, B))
    nx, ny = len(x['logits']), len(y['logits']); n = min(nx, ny); allok = nx == ny
    if nx != ny: print(f'position counts differ: {nx} vs {ny} (only {n} compared)')
    for i in range(n):
        lx, ly = x['logits'][i], y['logits'][i]
        eq = lx.shape == ly.shape and torch.equal(lx.view(torch.int32), ly.view(torch.int32))   # bitwise (NaN-safe, -0 vs 0 counted)
        md = float((lx - ly).abs().max()) if lx.shape == ly.shape else float('nan')
        ie = torch.equal(x['ids0'][i], y['ids0'][i]) and torch.equal(x['ids1'][i], y['ids1'][i])
        rep = all(x['replicas'][i]) and all(y['replicas'][i])
        allok &= eq and ie and rep
        print(f'pos {i:2d}: logits {"EQ " if eq else "DIFF"} max|d| {md:.3e}  ids {"EQ" if ie else "DIFF"}  replicas {rep}')
    print(f'{os.path.basename(A)} vs {os.path.basename(B)}: {n} positions, ' + ('BITWISE EQUAL' if allok else 'NOT EQUAL'))
    return allok


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == 'save': sys.exit(0 if save(a[1:]) else 1)
    elif len(a) == 3 and a[0] == 'cmp': sys.exit(0 if cmp(a[1:]) else 1)
    else: print(__doc__); sys.exit(2)
