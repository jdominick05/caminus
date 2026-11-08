"""Qwen3-Coder-Next (qwen3next) in the K3 engine (caminus.engine.qwen3next_tp.Qwen3NextTP, both cards) vs llama.cpp
(ignite-llama build-hip-therock through tools/gg_golden, tools/reference/q3n_refs.sh; q8_0 KV, flash attention on):
  layers  the first n = 1 .. 4 layers alone (debug_layers=n: three GDN layers, then the first attention layer):
          the replicated residual after each of the first 8 positions of the code text vs llama.cpp's l_out of layer
          n - 1 (golden_layers), relative error per position (localises a wrong layer type before any end-to-end number)
  e2e     each text teacher-forced one token per pass (llama.cpp's own token ids, tokens.txt): per position KLD(llama.cpp ||
          engine) over the full vocabulary and top-1 agreement, a replica check after every position; llama.cpp's own
          batch-vs-decode floor from the _batch dumps. D-281 bar: mean KLD <= the floor's, top-1 >= the floor's - 1 point
  bitwise the repo's K3 invariants on the whole tape, teacher-forced over the code text's first NPOS positions (--npos,
          default 96): a one-token reference at the default grid; T = 4 verify passes (--ntok T: another width; k3_gguf_q3n_mt) == it, bitwise
          per logits row (tools/gates/tp_multi.py's rule); K3_PER_WGP = 3 / 6 / 7 one-token runs == it (the race check);
          replicas equal after every run
usage: tools/guard.sh LOG SECS python -I tests/engine/test_q3n_e2e.py REFS [layers|e2e|bitwise|all] [--model GGUF] [--max-ctx N] [--npos N] [--ntok T]
  REFS: the tools/reference/q3n_refs.sh output dir (e2e_{prose,code,shell}_{decode,batch}, golden_layers)"""
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
from caminus import devpaths  # noqa: E402  (catalog models and ARTIFACTS from the environment: CLAUDE.md, "Models and data")
import numpy as np
import torch

args = sys.argv[1:]
PATH = devpaths.catalog('qwen3-coder-next', 'Q3N'); MAXC = 512
if '--model' in args: i = args.index('--model'); PATH = args[i + 1]; del args[i:i + 2]
if '--max-ctx' in args: i = args.index('--max-ctx'); MAXC = int(args[i + 1]); del args[i:i + 2]
NPOS = 96
if '--npos' in args: i = args.index('--npos'); NPOS = int(args[i + 1]); del args[i:i + 2]
NT = 4                                    # bitwise: the verify width (--ntok: a drafter's block, e.g. 8)
if '--ntok' in args: i = args.index('--ntok'); NT = int(args[i + 1]); del args[i:i + 2]
if not args: raise SystemExit(__doc__)
REFS = args[0]; CMD = args[1] if len(args) > 1 else 'all'
if CMD not in ('layers', 'e2e', 'bitwise', 'all'): raise SystemExit(__doc__)
def engine_class(path):
    """the engine for the GGUF's architecture (the registry, caminus.arch): qwen3next -> Qwen3NextTP (both cards), qwen35moe /
    qwen35 -> Qwen35One (one card); the Coder's FlashNextTP takes other arguments: refused here"""
    from caminus import arch
    a = arch.of(path)
    if a.kind not in ('tp', 'one-card'): raise SystemExit(f'{path}: arch {arch.meta(path)["arch"]!r} has no engine here')
    return a.engine_class()


import caminus.engine.flashnext   # noqa: E402,F401  (gguf-py on the path)

RES = []


def ok(name, cond, info=''):
    RES.append((name, bool(cond))); print(f'{"ok  " if cond else "FAIL"} {name} {info}', flush=True)


def logp(x):
    x = x - x.max(); return x - np.log(np.exp(x).sum())


def kld(ref, oth):
    lp, lq = logp(ref), logp(oth); return float((np.exp(lp) * (lp - lq)).sum())


def index(GD):
    return [l.rstrip('\n').split('\t') for l in open(f'{GD}/index.tsv')]


def tensor(GD, row):
    return np.fromfile(f'{GD}/{row[8]}', dtype=np.float32)


def cmd_layers():
    GD = f'{REFS}/golden_layers'; idx = index(GD)
    toks = [int(v) for v in open(f'{GD}/tokens.txt').read().split()]
    for n in (1, 2, 3, 4):
        t0 = time.time(); m = engine_class(PATH)(PATH, max_ctx=64, debug_layers=n)
        ref = {int(r[0]): r for r in idx if r[2] == f'l_out-{n - 1}'}
        m.reset(); errs = []
        for i in range(8):
            m.step(toks, i)
            x = m.res[0].reshape(-1).cpu().numpy().astype(np.float64); y = tensor(GD, ref[i]).astype(np.float64)
            errs.append(float(np.abs(x - y).max() / max(1e-30, np.abs(y).max())))
        rep = m.check_replicas()
        kind = 'attention' if (n - 1) % m.fai == m.fai - 1 else 'GDN'
        ok(f'layers 0 .. {n - 1} ({kind} last): l_out vs llama.cpp', max(errs) < 2e-2 and rep == (True, True),
           f'(rel max over 8 positions {max(errs):.2e}, per position {" ".join(f"{e:.1e}" for e in errs)}; replicas {rep}; built in {time.time() - t0:.0f} s)')
        del m; torch.cuda.empty_cache()


def cmd_e2e():
    t0 = time.time(); m = engine_class(PATH)(PATH, max_ctx=MAXC); print(f'loaded in {time.time() - t0:.0f} s', flush=True)
    allk, alla, allf, allfa, rep_bad = [], [], [], [], 0
    for name in ('prose', 'code', 'shell'):
        GD = f'{REFS}/e2e_{name}_decode'; BD = f'{REFS}/e2e_{name}_batch'
        toks = [int(v) for v in open(f'{GD}/tokens.txt').read().split()]
        dec = {int(r[0]): r for r in index(GD) if r[2] == 'result_output'}
        br = [r for r in index(BD) if r[2] == 'result_output']; assert len(br) == 1
        batch = tensor(BD, br[0]).reshape(int(br[0][5]), -1)
        m.reset(); kls, ag, fk, fa, flips = [], [], [], [], []
        for i in range(len(toks)):
            lo = m.step(toks, i).double().cpu().numpy().reshape(-1)
            if m.check_replicas() != (True, True): rep_bad += 1
            ref = tensor(GD, dec[i]).astype(np.float64)
            kls.append(kld(ref, lo)); ag.append(int(lo.argmax() == ref.argmax()))
            if not ag[-1]: s = np.sort(ref)[::-1]; flips.append(f'{i}:{s[0] - s[1]:.2f}')
            b = batch[i].astype(np.float64); fk.append(kld(ref, b)); fa.append(int(b.argmax() == ref.argmax()))
        allk += kls; alla += ag; allf += fk; allfa += fa
        print(f'e2e_{name}: {len(toks)} positions, engine vs llama.cpp: mean KLD {np.mean(kls):.4f}, top-1 {100 * np.mean(ag):.1f}% | '
              f'llama.cpp batch vs decode: mean KLD {np.mean(fk):.4f}, top-1 {100 * np.mean(fa):.1f}%', flush=True)
        print(f'   engine top-1 flips (position: reference top-1 - top-2 logit gap): {" ".join(flips) or "none"}', flush=True)
    mk, a1, fk_, fa_ = float(np.mean(allk)), 100 * float(np.mean(alla)), float(np.mean(allf)), 100 * float(np.mean(allfa))
    print(f'replica mismatches (residual or router ids differ between the cards): {rep_bad}')
    print(f'ALL: {len(allk)} positions, engine: mean KLD {mk:.4f}, top-1 {a1:.1f}% | llama.cpp floor: mean KLD {fk_:.4f}, top-1 {fa_:.1f}%')
    ok('D-281 (mean KLD <= floor, top-1 >= floor - 1, replicas equal)', mk <= fk_ and a1 >= fa_ - 1.0 and rep_bad == 0,
       f'(KLD {mk:.4f} vs {fk_:.4f}, top-1 {a1:.1f}% vs {fa_ - 1.0:.1f}%)')


def beq(a, b):
    """bitwise: the bytes of two rows (torch.equal counts -0 == 0 and NaN != NaN)"""
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def one_token_rows(toks, n, per_wgp=None):
    """logits rows of n teacher-forced one-token steps on a fresh engine (K3_PER_WGP = per_wgp while it is built)"""
    old = os.environ.get('K3_PER_WGP')
    if per_wgp: os.environ['K3_PER_WGP'] = str(per_wgp)
    try: m = engine_class(PATH)(PATH, max_ctx=MAXC)
    finally:
        if per_wgp:
            if old is None: os.environ.pop('K3_PER_WGP')
            else: os.environ['K3_PER_WGP'] = old
    m.reset(); rows = torch.cat([m.step(toks, i)[None].cpu() for i in range(n)]); rep = m.check_replicas()
    grid = [e.grid for e in m.engines]
    del m; torch.cuda.empty_cache()
    return rows, rep, grid


def cmd_bitwise():
    toks = [int(v) for v in open(f'{REFS}/e2e_code_decode/tokens.txt').read().split()]
    n = min(NPOS, len(toks)) // NT * NT; assert n >= 2 * NT, n
    ref, rep, grid = one_token_rows(toks, n)
    ok('one-token reference: replicas equal', rep == (True, True), f'({n} positions of the code text, grid {grid})')
    m = engine_class(PATH)(PATH, max_ctx=MAXC, ntok=NT); m.reset(); out = []
    mods = [e.module for e in m.engines]
    for i0 in range(0, n, NT): out.append(m.verify(toks, i0, 0 if i0 == 0 else NT).cpu())
    out = torch.cat(out); rep4 = m.check_replicas(); del m; torch.cuda.empty_cache()
    same = [beq(out[i], ref[i]) for i in range(n)]
    ok(f'T = {NT} verify == one-token steps, bitwise', all(same) and rep4 == (True, True),
       f'({sum(same)}/{n} rows; modules {mods}; max |diff| {(out - ref).abs().max().item():.3e}; replicas {rep4})')
    for pw in (3, 6, 7):
        rows, repg, g = one_token_rows(toks, n, per_wgp=pw)
        same = [beq(rows[i], ref[i]) for i in range(n)]
        ok(f'K3_PER_WGP={pw} == the default grid, bitwise', all(same) and repg == (True, True),
           f'({sum(same)}/{n} rows; grid {g}; replicas {repg})')


if __name__ == '__main__':
    for d in range(torch.cuda.device_count()): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)   # (one visible card: device 0)
    if CMD in ('layers', 'all'): cmd_layers()
    if CMD in ('e2e', 'all'): cmd_e2e()
    if CMD in ('bitwise', 'all'): cmd_bitwise()
    nf = sum(1 for _, g in RES if not g)
    print(f'RESULT {CMD}: {len(RES) - nf} ok, {nf} FAIL')
    sys.exit(1 if nf else 0)
