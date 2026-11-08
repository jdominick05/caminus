"""K3 TP smoke benchmark: one command, a few minutes including the model loads, run through guard.sh.
  guard.sh LOG SECS python -I tools/k3_smoke.py [--model coder|reap] [--n 128] [--tf 64] [--spec dflash[,mtp]] [--out DIR]
  python -I tools/k3_smoke.py --compare A.json B.json [--tol 2]      (CPU only: per-metric drift between two runs)

Stages (each GPU stage is its own child process, one after the other, so two engines never share VRAM; this parent
never touches the GPU):
  tp1          FlashNextTP(ntok=1):
               (a) correctness spot check: teacher-forced positions 0 .. TF-1 of each e2e text (prose / code / shell)
                   against llama.cpp's one-token decode dump -> mean KLD (float64, full vocab) and top-1 agreement vs
                   the D-281 floor on the same positions (llama.cpp batch vs decode, from the _batch dumps): pass if
                   mean KLD <= floor and top-1 >= floor - 1 point; check_replicas() after every position
               (b) plain decode with the real generation loop (step_token() per token: step() + argmax with only the
                   token read back, the same tokens as m2/s7/spec_k3.py plain's step().argmax()):
                   a 16-token warm-up, then N greedy tokens after each 200-token prompt, 2 reps (rep 2 reported, rep 1
                   faults in memory-mapped rows); rep 1 == rep 2 tokens
  spec:NAME    one speculative method from DRAFTERS (dflash: caminus.decode.dflash, the DFlash 2 drafter in K3, or with
               --dflash-impl torch the torch drafter caminus.decode.dflash_torch; dflash_hostacc: the K3 drafter with the host accept;
               dflash_torch: always the torch one; mtp: caminus.decode.mtp in K3;
               T - 1 drafts per pass). Its own
               engine + drafter, a warm-up generation, then N tokens per prompt with caminus.decode.spec.generate, 2 reps:
               tok/s (generation tokens after the prompt pass / generation wall), tokens per pass, acceptance, the
               pass breakdown; gate: output identical to the plain tokens of the tp1 stage (and rep 1 == rep 2)
A method plugs in as a factory(cfg, T) -> (engine, drafter, info) in DRAFTERS; the MTP one is looked up as
caminus.decode.mtp.smoke_factory (python/caminus/decode/mtp.py) and is "pending" while that module / function does not exist.
--spec defaults to dflash,mtp when an MTP sidecar is configured (CAMINUS_MTP, or config.toml's mtp: what caminus.decode.mtp loads),
else to dflash with a printed note (the sidecar is not distributed; this is a development machine's tool).
The table puts K3 next to llama.cpp's reference numbers from tools/smoke_ref.json (each with its source and N).
CAMINUS_SMOKE_INSTALLED=1: the stages import the installed caminus package (run with that environment's python), not this
checkout's python/ (the packaging gate: the wheel's code objects and sources).
Results: OUT/<time>.json (default $ARTIFACTS/smoke). Exit 1 on any correctness failure (KLD / top-1 bar, a replica
mismatch, rep mismatch, spec != plain, a missing reference, a failed stage)."""
import os, sys, json, time, subprocess, threading, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_JSON = os.path.join(REPO, 'tools', 'smoke_ref.json')
OUT_DEFAULT = os.path.join(os.environ['ARTIFACTS'], 'smoke') if os.environ.get('ARTIFACTS') else None
TEXTS = ('prose', 'code', 'shell')
VRAM = {0: '/sys/class/drm/card1/device/mem_info_vram_used', 1: '/sys/class/drm/card2/device/mem_info_vram_used'}


def load_ref():
    """tools/smoke_ref.json with its paths' $VARIABLES expanded from the environment (CLAUDE.md, "Models and data": CODER,
    REAP, DFLASH, ARTIFACTS, REAP_DUMPS); an unset variable stays as written, so the missing-input check names it"""
    ref = json.load(open(REF_JSON))
    x = lambda p: os.path.expanduser(os.path.expandvars(p))
    for m in ref['models'].values():
        m['gguf'], m['ref_dumps'] = x(m['gguf']), x(m['ref_dumps'])
    ref['drafter'] = x(ref['drafter'])
    return ref


def guarded():
    """True if an ancestor process is guard.sh (every GPU command runs under it)"""
    pid = os.getppid()
    while pid > 1:
        try:
            cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
            if 'guard.sh' in cmd: return True
            pid = int(open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def prompt_tokens(cfg, name, n=None):
    toks = [int(v) for v in open(os.path.join(cfg['ref_dumps'].format(text=name), 'tokens.txt')).read().split()]
    return toks if n is None else toks[:n]


# -- reference dumps (CPU) -------------------------------------------------------------------------------------------
def _logp(x):
    import numpy as np
    x = x - x.max(); return x - np.log(np.exp(x).sum())


def kld(ref, oth):
    """KLD(ref || oth) in float64 over the full vocab (tests/engine/test_flashnext_e2e.py)"""
    import numpy as np
    lp, lq = _logp(ref), _logp(oth); return float((np.exp(lp) * (lp - lq)).sum())


def decode_ref(GD, i):
    """llama.cpp's one-token decode logits at position i (float64)"""
    import numpy as np
    if not hasattr(decode_ref, 'idx'): decode_ref.idx = {}
    if GD not in decode_ref.idx:
        rows = [l.rstrip('\n').split('\t') for l in open(f'{GD}/index.tsv')]
        decode_ref.idx[GD] = {int(r[0]): r for r in rows if r[2] == 'result_output'}
    return np.fromfile(f'{GD}/{decode_ref.idx[GD][i][8]}', dtype=np.float32).astype(np.float64)


def batch_rows(BD, n):
    """the first n rows of llama.cpp's batch-mode logits [positions, vocab] (memory-mapped)"""
    import numpy as np
    rows = [l.rstrip('\n').split('\t') for l in open(f'{BD}/index.tsv')]
    rs = [r for r in rows if r[2] == 'result_output']
    assert len(rs) == 1 and int(rs[0][5]) >= n, (BD, len(rs))
    return np.memmap(f'{BD}/{rs[0][8]}', dtype=np.float32, mode='r').reshape(int(rs[0][5]), -1)[:n]


def floor_subset(cfg, ntf):
    """D-281 floor on positions 0 .. ntf-1 of each text: llama.cpp batch vs llama.cpp decode"""
    import numpy as np
    ks, ag = [], []
    for name in TEXTS:
        GD = cfg['ref_dumps'].format(text=name); BD = GD.replace('_decode', '_batch')
        assert open(f'{BD}/tokens.txt').read().split() == open(f'{GD}/tokens.txt').read().split(), BD
        B = batch_rows(BD, ntf)
        for i in range(ntf):
            ref = decode_ref(GD, i); b = np.asarray(B[i], dtype=np.float64)
            ks.append(kld(ref, b)); ag.append(int(b.argmax() == ref.argmax()))
    return dict(mean_kld=float(np.mean(ks)), top1=100 * float(np.mean(ag)), positions=len(ks))


# -- GPU stages (children) -------------------------------------------------------------------------------------------
def _torch_setup():
    if os.environ.get('CAMINUS_SMOKE_INSTALLED') != '1':
        sys.path.insert(0, os.path.join(REPO, 'python'))
    import torch
    import caminus
    print(f'caminus from {os.path.dirname(caminus.__file__)}', flush=True)
    for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)
    return torch


def stage_tp1(cfg, a):
    import numpy as np
    torch = _torch_setup()
    from caminus.engine.flashnext_tp import FlashNextTP
    t0 = time.time(); m = FlashNextTP(cfg['gguf'], max_ctx=a.max_ctx); out = dict(load_s=time.time() - t0, grids=[e.grid for e in m.engines])
    print(f'tp1: loaded in {out["load_s"]:.1f} s; grids {out["grids"]}', flush=True)
    # (a) teacher forced
    tf = {}; t1 = time.time()
    for name in TEXTS:
        GD = cfg['ref_dumps'].format(text=name); toks = prompt_tokens(cfg, name)
        assert len(toks) >= a.tf, (name, len(toks), a.tf)
        m.reset(); L = []; rep_bad = []
        for i in range(a.tf):
            L.append(m.step(toks, i).float().cpu().numpy())
            if m.check_replicas() != (True, True): rep_bad.append(i)
        ks, ag, flips = [], [], []
        for i, lo in enumerate(L):
            ref = decode_ref(GD, i); assert ref.shape == lo.shape, (ref.shape, lo.shape)
            lo = lo.astype(np.float64); ks.append(kld(ref, lo)); ag.append(int(lo.argmax() == ref.argmax()))
            if not ag[-1]: s = np.sort(ref)[::-1]; flips.append(f'{i}:{s[0] - s[1]:.2f}')
        tf[name] = dict(kld=ks, top1=ag, flips=flips, replica_bad=rep_bad)
        print(f'tf {name}: {a.tf} positions, mean KLD {np.mean(ks):.4f}, top-1 {100 * np.mean(ag):.1f}%, flips {flips or "none"}, '
              f'replica mismatches {len(rep_bad)}', flush=True)
    out['tf'] = tf; out['tf_s'] = time.time() - t1
    # (b) plain decode, the real loop
    def run(p, n):
        m.reset(); seq = list(p)
        for i in range(len(p) - 1): m.step(seq, i)
        t = time.perf_counter()
        for g in range(n):
            seq.append(m.step_token(seq, len(seq) - 1))   # the greedy token only (argmax of the logits, first on ties)
        return seq[len(p):], time.perf_counter() - t
    run(prompt_tokens(cfg, 'prose', a.prompt), 16)
    plain = {name: dict(tokens=None, tok_s=[], rep_equal=True, replicas=[]) for name in TEXTS}
    for rep in (1, 2):
        for name in TEXTS:
            seq, dt = run(prompt_tokens(cfg, name, a.prompt), a.n); P = plain[name]
            if rep == 1: P['tokens'] = seq
            else: P['rep_equal'] = P['tokens'] == seq
            P['tok_s'].append(a.n / dt); P['replicas'].append(list(m.check_replicas()))
            print(f'plain {name} rep{rep}: {a.n} tokens in {dt:.3f} s = {a.n / dt:.1f} tok/s ({1e3 * dt / a.n:.2f} ms/token); '
                  f'replicas {m.check_replicas()}', flush=True)
    out['plain'] = plain; out['bails'] = len(m.bails)
    return out


DFLASH_IMPL = 'k3'   # --dflash-impl: 'k3' (caminus.decode.dflash, the drafter inside K3) or 'torch' (caminus.decode.dflash_torch, the old arm)


def _dflash_factory(cfg, T, max_ctx=2048):
    """spec:dflash: the K3 drafter with the device accept (caminus.decode.dflash.smoke_factory: its layers on the XT, the inject on
    the XTX from the target's features there, the target's head shards on both cards + an XTX copy of part of the XT's;
    spec.generate hands the generation to DFlashK3Drafter.generate, the S3 loop) unless --dflash-impl torch, or the model is
    not the GSQ-RCO Coder (the K3 drafter reads the Coder's IQ4_XS token_embd / Q6_K head). Its "draft" ms is the step
    launch's device time on the XT (the accept + inject + draft), "verify" the rest of the pass"""
    if DFLASH_IMPL == 'k3' and cfg.get('name', 'coder') == 'coder':
        from caminus.decode import dflash as dflash_k3
        return dflash_k3.smoke_factory(cfg, T, max_ctx)
    return _dflash_torch_factory(cfg, T, max_ctx)


def _dflash_hostacc_factory(cfg, T, max_ctx=2048):
    """spec:dflash_hostacc: the same K3 drafter in spec.generate's generic loop (the host accept, the inject fused into the
    next draft launch): the device accept's A/B (the same drafts pass for pass; "draft" = the draft call's host wall)"""
    from caminus.decode import dflash as dflash_k3
    m, dr, info = dflash_k3.smoke_factory(cfg, T, max_ctx); dr.device_accept = False
    return m, dr, dict(info, impl='k3 host accept')


def _dflash_torch_factory(cfg, T, max_ctx=2048):
    from caminus.engine.flashnext_tp import FlashNextTP
    from gguf import GGUFReader                       # (caminus.engine.flashnext puts gguf-py on the path)
    from caminus.decode.dflash_torch import DFlash2
    from caminus.decode import spec
    rd = GGUFReader(cfg['drafter'], 'r'); f = {x.name: x for x in rd.fields.values()}
    layers = [int(v) for v in f['dflash.target_layers'].contents()]; del rd, f
    m = FlashNextTP(cfg['gguf'], ntok=T, feature_layers=layers, max_ctx=max_ctx)
    assert m.feature_layers == layers and all(0 <= l < m.nl for l in layers), (layers, m.nl)
    df = DFlash2(cfg['drafter'], m, **({} if max_ctx <= 4096 else dict(max_pos=max_ctx))); assert df.layers_tgt == layers
    df.check_capture = True; df.warm(T - 1)           # graph capture now (live K / V checked untouched)
    return m, spec.DFlashDrafter(df), dict(feature_layers=layers, drafter=cfg['drafter'], impl='torch')


def _mtp_factory(cfg, T, max_ctx=2048):
    from caminus.decode import mtp                            # python/caminus/decode/mtp.py: smoke_factory(cfg, T) -> (engine, drafter, info)
    return mtp.smoke_factory(cfg, T) if max_ctx == 2048 else mtp.smoke_factory(cfg, T, max_ctx=max_ctx)


DRAFTERS = {'dflash': _dflash_factory, 'dflash_hostacc': _dflash_hostacc_factory, 'dflash_torch': _dflash_torch_factory, 'mtp': _mtp_factory}


def mtp_available():
    p = os.path.join(REPO, 'python', 'caminus', 'mtp.py')
    return os.path.exists(p) and 'def smoke_factory' in open(p).read()


def stage_spec(cfg, a, name):
    torch = _torch_setup()
    from caminus.decode import spec
    t0 = time.time(); m, dr, info = DRAFTERS[name](cfg, a.T, a.max_ctx); T = m.ntok
    out = dict(load_s=time.time() - t0, T=T, info=info)
    print(f'spec:{name}: T={T}, engine + drafter ready in {out["load_s"]:.1f} s; {info}', flush=True)
    plain = json.load(open(a.plain_json))['plain']
    def gen(nm, n):
        st = {}; p = prompt_tokens(cfg, nm, a.prompt)
        return spec.generate(m, p, n, dr, st)[len(p):], st
    gen('prose', 32)                                  # warm-up
    R = {nm: dict(tok_s=[], tokens_per_pass=[], acceptance=[], passes=[], ms_pass=[], ms_draft=[], ms_verify=[], ms_inject=[],
                  replicas=[], rep_equal=True, same_as_plain=None, first_diff=None) for nm in TEXTS}
    for rep in (1, 2):
        for nm in TEXTS:
            toks, st = gen(nm, a.n); r = R[nm]
            gp = st['gen_passes']; ng = st['gen_tokens'] - 1; gs = st['gen_seconds']
            r['tok_s'].append(ng / gs); r['tokens_per_pass'].append(ng / gp); r['passes'].append(gp)
            r['acceptance'].append(100 * st['accepted'] / (gp * (T - 1)))
            r['ms_pass'].append(1e3 * gs / gp); r['ms_draft'].append(1e3 * st['draft_seconds'] / gp)
            r['ms_verify'].append(1e3 * st['verify_seconds'] / gp); r['ms_inject'].append(1e3 * st['inject_seconds'] / gp)
            r['replicas'].append(list(m.check_replicas()))
            if rep == 1:
                p = plain[nm]['tokens']; r['tokens'] = toks
                r['first_diff'] = next((i for i in range(min(len(toks), len(p))) if toks[i] != p[i]), None)
                r['same_as_plain'] = r['first_diff'] is None and len(toks) == len(p)
            else: r['rep_equal'] = r['tokens'] == toks
            print(f'{name} {nm} rep{rep}: {ng / gs:.1f} tok/s; {gp} passes, {ng / gp:.2f} tokens/pass, acceptance {r["acceptance"][-1]:.1f}%; '
                  f'per pass {r["ms_pass"][-1]:.2f} ms = draft {r["ms_draft"][-1]:.2f} + verify {r["ms_verify"][-1]:.2f} + inject '
                  f'{r["ms_inject"][-1]:.2f}; replicas {r["replicas"][-1]}' + ('' if rep == 2 else
                  f'; == plain: {r["same_as_plain"]}' + ('' if r['same_as_plain'] else f' (first difference at generated token {r["first_diff"]})')), flush=True)
    out['texts'] = R; out['bails'] = len(m.bails)
    return out


# -- parent ----------------------------------------------------------------------------------------------------------
class VramPeak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.peak = {0: 0, 1: 0}; self.stop = False
    def run(self):
        while not self.stop:
            for d, p in VRAM.items():
                try: self.peak[d] = max(self.peak[d], int(open(p).read()) >> 20)
                except OSError: pass
            time.sleep(0.5)


def child(stage, a, part):
    cmd = [sys.executable, '-I', os.path.abspath(__file__), '--stage', stage, '--model', a.model, '--n', str(a.n), '--tf', str(a.tf),
           '--T', str(a.T), '--max-ctx', str(a.max_ctx), '--dflash-impl', a.dflash_impl, '--part', part] + (['--plain-json', a.plain_json] if a.plain_json else [])
    t0 = time.time(); rc = subprocess.call(cmd); dt = time.time() - t0
    print(f'== stage {stage}: rc={rc}, {dt:.1f} s', flush=True)
    res = json.load(open(part)) if rc == 0 and os.path.exists(part) else None
    if os.path.exists(part): os.remove(part)
    return rc, dt, res


def med2(x): return x[-1]                             # rep 2 (rep 1 faults memory-mapped rows in)


def report(R, ref):
    """correctness verdicts + the table; returns the list of failures"""
    import numpy as np
    fails = []; L = ref['models'][R['model']].get('llama') or {}
    tp1 = R['stages'].get('tp1')
    if tp1 is None:
        fails.append('stage tp1 failed')
    else:
        tf = tp1['tf']; ks = sum((tf[t]['kld'] for t in TEXTS), []); ag = sum((tf[t]['top1'] for t in TEXTS), [])
        mk, mt = float(np.mean(ks)), 100 * float(np.mean(ag)); fl = R['floor']
        ok = mk <= fl['mean_kld'] and mt >= fl['top1'] - 1.0
        rb = sum(len(tf[t]['replica_bad']) for t in TEXTS)
        R['correctness'] = dict(tf_mean_kld=mk, tf_top1=mt, floor=fl, d281_pass=ok, tf_replica_mismatches=rb)
        print(f'\nCORRECTNESS teacher forced ({len(ks)} positions): mean KLD {mk:.4f} vs floor {fl["mean_kld"]:.4f}, top-1 {mt:.1f}% vs '
              f'bar {fl["top1"] - 1.0:.1f}% (floor {fl["top1"]:.1f}%, {fl["source"]}) -> {"PASS" if ok else "FAIL"}; replica mismatches {rb}')
        ff = ref['models'][R['model']].get('floor_full')
        if ff: print(f'   (full D-281 floor for context: mean KLD {ff["mean_kld"]:.4f}, top-1 {ff["top1"]:.1f}% over {ff["positions"]} positions)')
        if not ok: fails.append('D-281 bar (teacher forced)')
        if rb: fails.append(f'{rb} replica mismatches (teacher forced)')
        for t in TEXTS:
            P = tp1['plain'][t]
            if not P['rep_equal']: fails.append(f'plain {t}: rep1 != rep2')
            if not all(all(x) for x in P['replicas']): fails.append(f'plain {t}: replica mismatch')
    for name in R['spec_names']:
        s = R['stages'].get(f'spec:{name}')
        if s == 'pending': continue
        if s is None: fails.append(f'stage spec:{name} failed'); continue
        for t in TEXTS:
            r = s['texts'][t]
            if not r['same_as_plain']: fails.append(f'{name} {t}: output != plain (first difference at token {r["first_diff"]})')
            if not r['rep_equal']: fails.append(f'{name} {t}: rep1 != rep2')
            if not all(all(x) for x in r['replicas']): fails.append(f'{name} {t}: replica mismatch')
    # table
    n = R['n']
    def lc(key, t, f='tok_s'):
        e = L.get(key); return e[f][t] if e and f in e and e[f] else None
    def fmt(v, w=7, p=1): return f'{v:{w}.{p}f}' if isinstance(v, (int, float)) else f'{v if v is not None else "-":>{w}}'
    print(f'\nSMOKE {R["model"]}: K3 {n} greedy tokens per 200-token prompt (rep 2); llama.cpp references at N = '
          f'{sorted({e["n"] for e in L.values() if e})} (tools/smoke_ref.json)')
    hdr = f'{"text":6s} | {"K3 plain":>8s} {"lc plain":>8s} | {"K3 DFl":>7s} {"tok/pass":>8s} {"acc%":>6s} {"=plain":>6s} {"lc DFl":>7s} {"lc acc%":>7s} | {"K3 MTP":>7s} {"acc%":>6s} {"=plain":>6s} {"lc MTP3":>7s} {"lc MTP1":>7s} | {"KLD":>7s} {"top1%":>6s}'
    print(hdr); print('-' * len(hdr))
    d = R['stages'].get('spec:dflash'); mt = R['stages'].get('spec:mtp')
    for t in TEXTS:
        kp = med2(tp1['plain'][t]['tok_s']) if tp1 else None
        kd = tpp = acc = None; same = 'FAIL' if 'dflash' in R['spec_names'] else '-'
        if isinstance(d, dict):
            r = d['texts'][t]; kd, tpp, acc, same = med2(r['tok_s']), med2(r['tokens_per_pass']), med2(r['acceptance']), ('yes' if r['same_as_plain'] else 'NO')
        km = 'pending' if mt == 'pending' else med2(mt['texts'][t]['tok_s']) if isinstance(mt, dict) else 'FAIL' if 'mtp' in R['spec_names'] else '-'
        kma = med2(mt['texts'][t]['acceptance']) if isinstance(mt, dict) else None
        kms = ('yes' if mt['texts'][t]['same_as_plain'] else 'NO') if isinstance(mt, dict) else '-'
        tk = (float(np.mean(tp1['tf'][t]['kld'])), 100 * float(np.mean(tp1['tf'][t]['top1']))) if tp1 else (None, None)
        print(f'{t:6s} | {fmt(kp, 8)} {fmt(lc("plain", t), 8)} | {fmt(kd)} {fmt(tpp, 8, 2)} {fmt(acc, 6)} {same:>6s} {fmt(lc("dflash", t))} '
              f'{fmt(lc("dflash", t, "acceptance"), 7)} | {fmt(km)} {fmt(kma, 6)} {kms:>6s} {fmt(lc("mtp3", t))} {fmt(lc("mtp1", t))} | {fmt(tk[0], 7, 4)} {fmt(tk[1], 6)}')
    dt = R['stages'].get('spec:dflash_torch'); dh = R['stages'].get('spec:dflash_hostacc')
    if isinstance(d, dict): print(f'   K3 DFl column: {d.get("info", {}).get("impl", "?")} drafter; draft ms/pass (k3: the step launch device time) ' +
                                  ' / '.join(f'{med2(d["texts"][t]["ms_draft"]):.2f}' for t in TEXTS))
    if isinstance(dh, dict):
        print('   K3 DFlash, host accept (spec:dflash_hostacc): tok/s ' + ' / '.join(f'{med2(dh["texts"][t]["tok_s"]):.1f}' for t in TEXTS) +
              ', ms/pass ' + ' / '.join(f'{med2(dh["texts"][t]["ms_pass"]):.2f}' for t in TEXTS) + ' (device accept: ' +
              ' / '.join(f'{med2(d["texts"][t]["ms_pass"]):.2f}' for t in TEXTS if isinstance(d, dict)) + '), == plain ' +
              ' / '.join('yes' if dh['texts'][t]['same_as_plain'] else 'NO' for t in TEXTS))
    if isinstance(dt, dict):
        print('   torch DFlash (spec:dflash_torch): tok/s ' + ' / '.join(f'{med2(dt["texts"][t]["tok_s"]):.1f}' for t in TEXTS) +
              ', acceptance ' + ' / '.join(f'{med2(dt["texts"][t]["acceptance"]):.1f}%' for t in TEXTS) + ', == plain ' +
              ' / '.join('yes' if dt['texts'][t]['same_as_plain'] else 'NO' for t in TEXTS) + ', draft ms/pass ' +
              ' / '.join(f'{med2(dt["texts"][t]["ms_draft"]):.2f}' for t in TEXTS))
    for k, e in L.items():
        if e: print(f'   lc {k}: {e["source"]}')
    return fails


def compare(A, B, tol):
    x, y = (json.load(open(p)) for p in (A, B)); bad = []
    def m(R):
        out = {}
        t = R['stages'].get('tp1')
        if t:
            for n in TEXTS: out[f'plain {n} tok/s'] = med2(t['plain'][n]['tok_s'])
        for k, s in R['stages'].items():
            if k.startswith('spec:') and isinstance(s, dict):
                for n in TEXTS:
                    for f in ('tok_s', 'tokens_per_pass', 'acceptance'): out[f'{k[5:]} {n} {f}'] = med2(s['texts'][n][f])
        if R.get('correctness'): out['tf mean KLD'] = R['correctness']['tf_mean_kld']; out['tf top-1'] = R['correctness']['tf_top1']
        return out
    mx, my = m(x), m(y)
    for k in mx:
        if k not in my: continue
        d = 100 * (my[k] - mx[k]) / mx[k] if mx[k] else 0.0
        flag = abs(d) > tol and 'KLD' not in k     # KLD: bitwise engine, reported, not a speed gate
        bad += [k] if flag else []
        print(f'{k:32s} {mx[k]:10.4f} {my[k]:10.4f} {d:+7.2f}%' + ('  OUT OF TOLERANCE' if flag else ''))
    print(f'runtime {x.get("wall_s", 0):.1f} s vs {y.get("wall_s", 0):.1f} s; ' + (f'{len(bad)} metrics beyond +-{tol}%: {bad}' if bad else f'all metrics within +-{tol}%'))
    return not bad


def default_spec():
    """(spec, note): dflash,mtp when an MTP sidecar is configured (caminus.config's mtp: CAMINUS_MTP, else config.toml), else
    dflash and the reason (the sidecar is not distributed)"""
    if os.environ.get('CAMINUS_SMOKE_INSTALLED') != '1' and os.path.join(REPO, 'python') not in sys.path:
        sys.path.insert(0, os.path.join(REPO, 'python'))
    try:
        from caminus import config
        side = config.mtp()
    except Exception as e:                            # (a broken config.toml: say so, run without MTP)
        return 'dflash', f'the MTP sidecar path could not be read ({e})'
    if side:
        return 'dflash,mtp', None
    return 'dflash', 'no MTP sidecar configured (CAMINUS_MTP unset, no mtp in config.toml; the sidecar is not distributed)'


def check_args(a):
    """the positions every stage touches, against --max-ctx, before anything is loaded or launched: a list of problems.
    tf: positions 0 .. tf - 1; plain: the prompt (<= a.prompt tokens) then n greedy tokens, positions up to prompt + n - 1;
    spec: caminus.decode.spec.positions_needed (a T-token pass reaches T - 1 positions past the last accepted token)"""
    bad = []
    for k in ('n', 'tf', 'T', 'max_ctx'):
        if getattr(a, k) < 1: bad.append(f'--{k.replace("_", "-")} {getattr(a, k)} < 1')
    if bad: return bad
    if a.tf > a.max_ctx: bad.append(f'--tf {a.tf} > --max-ctx {a.max_ctx}')
    if a.prompt + a.n > a.max_ctx: bad.append(f'prompt {a.prompt} + --n {a.n} = {a.prompt + a.n} positions > --max-ctx {a.max_ctx}')
    spec_need = max(a.prompt - 1, a.prompt + a.n - 2) + a.T   # = caminus.decode.spec.positions_needed (not imported: no torch here)
    if [s for s in a.spec.split(',') if s] and spec_need > a.max_ctx:
        bad.append(f'speculative stages need prompt {a.prompt} + --n {a.n} + --T {a.T} - 2 = {spec_need} positions > --max-ctx {a.max_ctx}')
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='coder', choices=('coder', 'reap'))
    ap.add_argument('--n', type=int, default=128, help='greedy tokens per prompt (plain and speculative)')
    ap.add_argument('--tf', type=int, default=64, help='teacher-forced positions per text')
    ap.add_argument('--T', type=int, default=4, help='verify tokens per pass for the speculative stages')
    ap.add_argument('--spec', default=None, help='speculative methods, comma separated (from DRAFTERS; default dflash,mtp, or dflash '
                    'when no MTP sidecar is configured: CAMINUS_MTP / config.toml)')
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--compare', nargs=2, metavar=('A', 'B')); ap.add_argument('--tol', type=float, default=2.0)
    ap.add_argument('--max-ctx', type=int, default=2048, help='KV cache positions of the engines (FlashNextTP max_ctx)')
    ap.add_argument('--dflash-impl', default='k3', choices=('k3', 'torch'),
                    help='spec:dflash runs the K3 drafter (caminus.decode.dflash) or the torch one (caminus.decode.dflash_torch; also spec:dflash_torch)')
    ap.add_argument('--stage'); ap.add_argument('--part'); ap.add_argument('--plain-json')
    a = ap.parse_args(); a.prompt = 200
    global DFLASH_IMPL; DFLASH_IMPL = a.dflash_impl
    if a.compare: sys.exit(0 if compare(*a.compare, a.tol) else 1)
    if a.spec is None:
        a.spec, why = default_spec()
        if why and not a.stage:
            print(f'k3_smoke: NOTE --spec defaults to {a.spec}: {why}; set CAMINUS_MTP (or mtp in config.toml) to the sidecar GGUF for the MTP stage', flush=True)
    bad = check_args(a)
    if bad: print(f'k3_smoke: refusing the arguments (nothing launched): {bad}'); sys.exit(2)
    ref = load_ref(); cfg = dict(ref['models'][a.model]); cfg['drafter'] = ref['drafter']; cfg['name'] = a.model
    if not guarded():
        print('k3_smoke: refusing to run outside guard.sh (every GPU command runs as guard.sh LOG SECS CMD...); see docs/dev/smoke.md'); sys.exit(3)
    if a.stage:                                       # child
        res = stage_tp1(cfg, a) if a.stage == 'tp1' else stage_spec(cfg, a, a.stage.split(':', 1)[1])
        json.dump(res, open(a.part, 'w')); return
    if a.out is None: print('k3_smoke: set ARTIFACTS or pass --out (nothing launched)'); sys.exit(2)
    t0 = time.time(); stamp = time.strftime('%Y%m%d-%H%M%S'); os.makedirs(a.out, exist_ok=True)
    names = [s for s in a.spec.split(',') if s]
    for s in names: assert s in DRAFTERS, (s, list(DRAFTERS))
    R = dict(model=a.model, gguf=cfg['gguf'], n=a.n, tf=a.tf, T=a.T, prompt_tokens=a.prompt, time=stamp, spec_names=names, stages={},
             git=subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip())
    fails = []
    missing = [p for p in [cfg['gguf'], cfg['drafter']] + [cfg['ref_dumps'].format(text=t) for t in TEXTS] if not os.path.exists(p)]
    if missing: print(f'missing references / inputs: {missing}'); sys.exit(1)
    try:                                              # the floor on the same positions (CPU), while nothing runs on the GPU
        R['floor'] = dict(floor_subset(cfg, a.tf), source=f'llama.cpp batch vs decode, positions 0..{a.tf - 1} x 3 texts, live')
    except (OSError, AssertionError) as e:
        print(f'no batch dumps for the floor ({e!r})'); fails.append('floor: batch dumps missing'); R['floor'] = dict(mean_kld=0.0, top1=100.0, source='missing')
    print(f'floor (llama.cpp batch vs decode, {R["floor"].get("positions")} positions): mean KLD {R["floor"]["mean_kld"]:.4f}, top-1 {R["floor"]["top1"]:.1f}%', flush=True)
    vp = VramPeak(); vp.start(); timing = {}
    a.plain_json = None
    part = os.path.join(a.out, f'.{stamp}.tp1.json')
    rc, timing['tp1'], res = child('tp1', a, part); R['stages']['tp1'] = res
    if res is not None:
        a.plain_json = os.path.join(a.out, f'.{stamp}.plain.json'); json.dump(dict(plain=res['plain']), open(a.plain_json, 'w'))
    for s in names:
        if s == 'mtp' and not mtp_available():
            R['stages']['spec:mtp'] = 'pending'; print('== stage spec:mtp: pending (python/caminus/decode/mtp.py smoke_factory not there yet)'); continue
        if a.plain_json is None: R['stages'][f'spec:{s}'] = None; continue
        part = os.path.join(a.out, f'.{stamp}.{s}.json')
        rc, timing[f'spec:{s}'], R['stages'][f'spec:{s}'] = child(f'spec:{s}', a, part)
    if a.plain_json and os.path.exists(a.plain_json): os.remove(a.plain_json)
    vp.stop = True; R['vram_peak_mib'] = dict(xtx=vp.peak[0], xt=vp.peak[1]); R['stage_s'] = timing
    fails += report(R, ref)
    R['wall_s'] = time.time() - t0; R['fails'] = fails; R['pass'] = not fails
    path = os.path.join(a.out, f'{stamp}.json'); json.dump(R, open(path, 'w'), indent=1)
    print(f'\nRESULT {"PASS" if not fails else "FAIL"}' + (f': {fails}' if fails else '') +
          f'; wall {R["wall_s"]:.1f} s (stages {", ".join(f"{k} {v:.0f} s" for k, v in timing.items())}); VRAM peak XTX {vp.peak[0]} MiB, '
          f'XT {vp.peak[1]} MiB; bails {sum(s["bails"] for s in R["stages"].values() if isinstance(s, dict))}; results {path}')
    sys.exit(0 if not fails else 1)


if __name__ == '__main__':
    main()
