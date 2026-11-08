"""caminus.decode.mtp.PolicyDrafter and the generic speculative loop (caminus.decode.spec.generate / spec.loop, caminus.decode.longctx) with both K3
drafters: impl:dflash-loop, the regression fix of 7d2f0a2 (it made DFlash the default drafter, but DFlash was correct only
in its own device-accept loop: the generic loop crashed on prompt lengths 2 / 3 mod 4 at T = 4, a k = 0 pass after given
tokens raised 'the inject at X is still pending', PolicyDrafter forwarded neither snapshot / restore nor generate, and the
default policy raised KeyError with only MTP loaded). Stages:
  cpu  (no GPU) the policy and the wrapper on fake drafters: the default policy with only MTP / only DFlash / neither loaded
       falls back (no KeyError; 'plain' gives filler drafts and plain greedy output), an explicit table's pick that is not
       loaded too (PolicyDrafter.fallback says so); reset / inject / prefill / flush / snapshot / restore reach every
       drafter; each pass the idle DFlash is flushed (no 'still pending') and the idle MTP caught up; device_accept /
       generate hand a generation to the active drafter's own loop (the idle drafters reset), else spec.generate's generic
       loop over the wrapper; spec.loop flushes before a k = 0 pass. (DFlashK3Drafter's own host code on a simulated device:
       tests/decode/test_dflash_ring_fuzz.py.)
  gpu  one Coder target (T = 4, max_ctx 2048) with both drafters loaded (MTPK3 + DFlashK3, the inject on the XTX), under
       guard.sh. Plain = the Repeat filler on the same engine (T = 4 passes == one-token steps bitwise, tools/gates/tp_multi.py),
       anchored to the latest smoke reference's one-token tokens at the 200-token code prompt:
   (1) the generic loop through PolicyDrafter at prompt lengths 1 .. 7, 13 .. 16, 200 .. 203 (0 / 1 / 2 / 3 mod 4), N new
       tokens: DFlash (MTP idle), DFlash alone in the wrapper and unwrapped, its device-accept loop (spec.generate's hand-off
       through the wrapper), MTP (DFlash idle), MTP alone (the default policy's fallback), neither (plain): tokens == plain,
       no exception; DFlash's drafts identical pass for pass in all four DFlash arms (generic == device accept); MTP's with
       DFlash idle == MTP alone
   (2) given tokens after a generation, then more generation (spec.loop, no caller flush), two rounds: 1 .. 9 given tokens
       (filler passes, k = 0 passes, prompt-like passes), DFlash and MTP: every segment == plain from scratch over its prefix
   (3) snapshot / restore through PolicyDrafter (longctx.fill with the wrapper, questions of any length), DFlash and MTP:
       A, restore, B, restore, A again: the same tokens, drafts call for call and acceptance, == plain resumed; a snapshot
       mid-generation (DFlash's inject pending: the snapshot flushes it), the continuation (generation, 3 given tokens = a
       k = 0 pass, generation, 2 given tokens = a filler pass, generation), restored and continued again: == the first
       continuation and == the same flow without the snapshot (tokens and drafts call for call); every segment == plain
   (4) timing (reported, not a gate; --no-bench skips it): DFlash's generic loop vs its device-accept loop, alone and in the
       wrapper with MTP idle, MTP alone vs MTP with DFlash idle: the smoke texts (200-token prompts, 128 tokens), the arms
       interleaved, --reps reps, median tok/s and acceptance; vm.compaction_proactiveness recorded
  HIP_VISIBLE_DEVICES= python -I tests/decode/test_spec_policy.py cpu
  guard.sh LOG 1800 python -I tests/decode/test_spec_policy.py gpu [--n 48] [--no-bench] [--reps 3] [--smoke SMOKE.json]"""
import os, sys, json, time, argparse, types, traceback
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools')); sys.path.insert(0, os.path.join(REPO, 'python'))   # (python/ first: tools/caminus.py would shadow the package)

ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=('cpu', 'gpu'))
ap.add_argument('--n', type=int, default=48); ap.add_argument('--reps', type=int, default=3); ap.add_argument('--no-bench', action='store_true')
ap.add_argument('--smoke', default=None, help='the smoke JSON whose tp1 plain tokens anchor the plain reference (default: the latest)')
a = ap.parse_args()
if a.stage == 'cpu':
    for v in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'): os.environ[v] = ''   # before torch
import torch
from caminus import devpaths
from caminus.decode import spec, longctx, mtp
fails = []


def check(c, what):
    print(('ok   ' if c else 'FAIL ') + what, flush=True)
    if not c: fails.append(what)
    return c


class Log:
    """a drafter wrapper recording each call's drafts (the rest of the interface forwarded)"""
    def __init__(self, d): self.d, self.log = d, []
    def __call__(self, toks, k):
        r = list(self.d(toks, k)); self.log.append(r); return r
    def __getattr__(self, n): return getattr(self.d, n)


# -- cpu: fakes ----------------------------------------------------------------------------------------------------------
V = 97


def oracle(prefix):
    """the fake target's greedy next token after a prefix (a hash, with repeats now and then so filler is accepted too)"""
    h = 1469598103
    for t in prefix: h = ((h ^ int(t)) * 1099511628211) & 0xFFFFFFFFFFFF
    return int(toks_last(prefix)) if h % 5 == 0 else h % V


def toks_last(p): return p[-1] if p else 0


class FakeTP:
    """a T-token target: verify_tokens(seq, i0, commit) -> the oracle's token after each position of the pass"""
    def __init__(self, T=4, max_ctx=4096):
        self.ntok, self.max_ctx, self.qsa_limit, self.features, self.seq, self.token = T, max_ctx, None, object(), 0, 0
    def reset(self): self.seq += 1
    def verify_tokens(self, seq, i0, commit):
        assert len(seq) >= i0 + self.ntok, (len(seq), i0)
        return [oracle(seq[:i0 + t + 1]) for t in range(self.ntok)]


def plain_cpu(m, p, n):
    out = list(p)
    while len(out) < len(p) + n: out.append(oracle(out))
    return out


class FakeDF:
    """DFlashK3Drafter's interface (one pending inject at most, as DFlashK3: a second inject before a flush / draft raises)"""
    def __init__(self, device_accept=False):
        self.calls, self.device_accept, self.pending, self.max_pos = [], device_accept, None, 3000
    def reset(self): self.calls.append('reset'); self.pending = None
    def inject(self, i0, feats):
        if self.pending is not None: raise RuntimeError(f'FakeDF.inject({i0}): the inject at {self.pending} is still pending')
        self.calls.append(('inject', i0)); self.pending = i0
    def prefill(self, toks, n): self.flush()
    def flush(self):
        if self.pending is not None: self.calls.append(('flush', self.pending))
        self.pending = None
    def snapshot(self): self.flush(); return dict(df=len(self.calls))
    def restore(self, s): self.calls.append(('restore', s['df']))
    def __call__(self, toks, k):
        self.calls.append(('draft', len(toks), k)); self.pending = None
        return [oracle(toks)] * k
    def generate(self, m, prompt, n_new, stats=None):
        self.calls.append(('generate', len(prompt), n_new))
        if stats is not None: stats['device_accept'] = True
        return plain_cpu(m, prompt, n_new)


class FakeMTPK3:
    """MTPK3's host state as MTPDrafter uses it (take_h / prefill / catch_up / draft / snapshot / restore / reset)"""
    def __init__(self, T=4):
        self.Tc, self.max_ctx, self.calls = T, 2048, []; self.reset()
    def reset(self): self.n_done = self.h_valid = 0; self.calls.append('reset')
    def take_h(self, i0, n=None): self.h_valid = i0 + self.Tc
    def prefill(self, toks, n_h):
        p_end = min(n_h + 1, len(toks) - 1)
        if p_end > self.n_done: self.catch_up(toks, p_end)
    def catch_up(self, toks, p_end, draft=False):
        assert p_end - 1 <= self.h_valid, (p_end, self.h_valid)
        self.calls.append(('catch_up', self.n_done, p_end)); self.n_done = max(self.n_done, p_end)
    def draft(self, toks, k):
        self.catch_up(toks, len(toks), True); self.calls.append(('draft', len(toks), k)); return [oracle(toks)] * k
    def snapshot(self): return dict(n_done=self.n_done, h_valid=self.h_valid)
    def restore(self, s): self.n_done, self.h_valid = s['n_done'], s['h_valid']; self.calls.append(('restore', s['n_done']))


def stage_cpu():
    assert not torch.cuda.is_available(), 'the cpu stage must run with no GPU visible'
    # choose: the default table and a caller's, against what is loaded
    for loaded, want in ((None, 'dflash'), ({'dflash'}, 'dflash'), ({'mtp'}, 'mtp'), (set(), 'plain'), ({'dflash', 'mtp'}, 'dflash')):
        for c in ('prose', 'code', 'shell', None, 'other'):
            check(mtp.choose(c, None, loaded) == want, f'choose({c!r}) default policy, loaded {loaded}: {want}')
    check(mtp.choose('shell', {'shell': 'mtp', None: 'dflash'}, {'dflash'}) == 'dflash', "explicit {'shell': 'mtp'} without MTP: falls back to dflash")
    check(mtp.choose('shell', {'shell': 'mtp', None: 'dflash'}, set()) == 'plain', 'explicit pick with nothing loaded: plain')
    check(mtp.choose('shell', {'shell': 'plain', None: 'dflash'}, {'dflash'}) == 'plain', "explicit 'plain' stays plain")
    check(mtp.choose('x', {'code': 'mtp'}, {'mtp'}) == 'plain', 'no entry and no None entry: plain')
    T = 4; m = FakeTP(T)
    prompts = [[(7 * i + j) % V for j in range(L)] for i, L in enumerate((1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 30, 31, 33))]
    # each subset of drafters, the default policy: no KeyError, the method used, spec == plain (generic loop)
    for names in (('dflash',), ('mtp',), (), ('dflash', 'mtp')):
        F, M = FakeDF(), mtp.MTPDrafter(FakeMTPK3(T))
        pd = mtp.PolicyDrafter({n: {'dflash': F, 'mtp': M}[n] for n in names})
        exp = 'dflash' if 'dflash' in names else ('mtp' if names else 'plain')
        check(pd.active == exp and pd.select('code') == exp, f'PolicyDrafter({list(names)}) default policy: {pd.active} (want {exp}; fallback {pd.fallback})')
        check(pd.fallback == (None if exp == 'dflash' else ('dflash', exp)), f'PolicyDrafter({list(names)}).fallback {pd.fallback}')
        check((pd.max_pos is None) == (not names), f'PolicyDrafter({list(names)}).max_pos {pd.max_pos}')
        bad = []
        for p in prompts:
            for n in (1, 5, 17):
                try: out = spec.generate(m, p, n, pd)
                except Exception as e: bad.append((len(p), n, f'{type(e).__name__}: {e}')); continue
                if out != plain_cpu(m, p, n): bad.append((len(p), n, 'output != plain'))
        check(not bad, f'PolicyDrafter({list(names)}): spec.generate == plain at prompt lengths {[len(p) for p in prompts]}: {bad[:3]}')
    # an explicit table naming an unloaded drafter: falls back, recorded
    pd = mtp.PolicyDrafter({'mtp': mtp.MTPDrafter(FakeMTPK3(T))}, policy={'code': 'dflash', None: 'plain'})
    check(pd.select('code') == 'mtp' and pd.fallback == ('dflash', 'mtp'), f"explicit {{'code': 'dflash'}} with only MTP: {pd.active}, fallback {pd.fallback}")
    check(pd.select('prose') == 'plain' and pd.fallback is None, 'explicit None: plain stays plain, no fallback')
    # forwarding to every drafter; the idle DFlash flushed each pass (MTP active: no 'still pending'), the idle MTP caught up
    F, Mk = FakeDF(), FakeMTPK3(T); M = mtp.MTPDrafter(Mk)
    pd = mtp.PolicyDrafter({'dflash': F, 'mtp': M}, policy={None: 'mtp'})
    check(pd.active == 'mtp', 'policy {None: mtp}: mtp active')
    p = prompts[9]; F.calls.clear(); Mk.calls.clear()
    try: out = spec.generate(m, p, 40, pd); err = None
    except Exception as e: out, err = None, f'{type(e).__name__}: {e}'
    nf = sum(1 for c in F.calls if c[0] == 'flush'); ni = sum(1 for c in F.calls if c[0] == 'inject'); nd = sum(1 for c in F.calls if c[0] == 'draft')
    check(err is None and out == plain_cpu(m, p, 40), f'MTP active, DFlash idle: spec == plain, no exception ({err})')
    check(ni > 3 and nf == ni - 1 and F.pending is not None and nd == 0, f'the idle DFlash: every inject flushed before the next ({ni} injects, {nf} flushes, '
          f'the last pass\'s still pending as the generic loop leaves it: {F.pending}; {nd} drafts)')
    pd.policy = None; pd.select('code'); F.calls.clear(); Mk.calls.clear()
    out = spec.generate(m, p, 40, pd)
    nc = sum(1 for c in Mk.calls if c[0] == 'catch_up'); nd = sum(1 for c in F.calls if c[0] == 'draft')
    check(out == plain_cpu(m, p, 40) and nc > 3 and nd > 3 and Mk.n_done >= Mk.h_valid - T, f'DFlash active, MTP idle: == plain, MTP caught up ({nc} catch-ups, n_done {Mk.n_done}, h_valid {Mk.h_valid})')
    # snapshot / restore / reset / flush forwarded
    F.pending = 5; s = pd.snapshot()
    check(set(s) == {'dflash', 'mtp'} and F.pending is None, f'snapshot: one state per drafter {sorted(s)}, the pending inject flushed')
    F.calls.clear(); Mk.calls.clear(); pd.restore(s)
    check(('restore', s['dflash']['df']) in F.calls and ('restore', s['mtp']['n_done']) in Mk.calls, 'restore: every drafter restored')
    try: pd.restore({'dflash': s['dflash']}); check(False, 'restore with a snapshot of other drafters refused')
    except RuntimeError: check(True, 'restore with a snapshot of other drafters refused')
    F.pending = 9; pd.flush(); check(F.pending is None, 'flush reaches the DFlash drafter')
    F.calls.clear(); Mk.calls.clear(); pd.reset(); check('reset' in F.calls and 'reset' in Mk.calls, 'reset reaches every drafter')
    # device_accept / generate: the active drafter's own loop, the idle ones reset; without it the generic loop
    F.device_accept = True; pd.select('code'); F.calls.clear(); Mk.calls.clear(); st = {}
    out = spec.generate(m, p, 12, pd, st)
    check(pd.device_accept and st.get('device_accept') and ('generate', len(p), 12) in F.calls and 'reset' in Mk.calls and out == plain_cpu(m, p, 12),
          'spec.generate hands the generation to the active drafter\'s device-accept loop (the idle MTP reset)')
    pd.policy = {None: 'mtp'}; pd.select('code')
    check(not pd.device_accept, 'MTP active: no device_accept (the generic loop)')
    F.calls.clear(); out = pd.generate(m, p, 12)
    check(out == plain_cpu(m, p, 12) and not any(c[0] == 'generate' for c in F.calls), 'PolicyDrafter.generate without a device loop: the generic loop')
    F.device_accept = False
    # spec.loop: given tokens after a generation (k = 0 / filler passes), no caller flush: the FakeDF raises on a lost inject
    for nm, d in (('DFlash', mtp.PolicyDrafter({'dflash': FakeDF()})), ('MTP', mtp.PolicyDrafter({'mtp': mtp.MTPDrafter(FakeMTPK3(T))}))):
        bad = []
        for g in range(1, 10):
            for L in (5, 6, 7, 8):
                p = prompts[10][:L]; m.reset(); d.reset()
                try:
                    t1, i0, c = spec.loop(m, list(p), 0, 0, L + 9, d)
                    t2 = t1 + [(3 * g + x) % V for x in range(g)]
                    t3, i0, c = spec.loop(m, list(t2), i0, c, len(t2) + 9, d)
                    if t3 != plain_cpu(m, t2, len(t3) - len(t2)) or t1 != plain_cpu(m, p, len(t1) - L): bad.append((g, L, 'output != plain'))
                except Exception as e: bad.append((g, L, f'{type(e).__name__}: {e}'))
        check(not bad, f'{nm}: spec.loop with 1 .. 9 given tokens after a generation (k = 0 passes for >= {T - 1}), no caller flush: == plain {bad[:3]}')
    # longctx.generate's state ends at the returned tokens (drafts accepted past the total are dropped from the state too),
    # so chained continuations with given tokens run on exactly the returned tokens
    for nm, mk in (('DFlash', lambda: mtp.PolicyDrafter({'dflash': FakeDF()})), ('MTP', lambda: mtp.PolicyDrafter({'mtp': mtp.MTPDrafter(FakeMTPK3(T))})),
                   ('plain', lambda: None)):
        bad = []; over = 0
        for L in (5, 6, 7, 8):
            d = mk(); m.reset()
            if d is not None: d.reset()
            st, toks = dict(i0=0, commit=0), prompts[11][:L]
            for r in range(8):
                pre = toks + [(5 * r + x) % V for x in range(r % 5)]
                s_ = {}
                try: out, st = longctx.generate(m, st, pre, 7, draft=d, stats=s_)
                except Exception as e: bad.append((L, r, f'{type(e).__name__}: {e}')); break
                over += s_['gen_tokens'] > 7
                if out != plain_cpu(m, pre, 7) or st['i0'] != len(out) - 1: bad.append((L, r, f'!= plain or i0 {st["i0"]} != {len(out) - 1}'))
                toks = out
        check(not bad and (over > 0 or nm == 'plain'), f'{nm}: chained longctx.generate continuations with 0 .. 4 given tokens == plain '
                                                       f'({over} of them overshot the total: dropped from the state) {bad[:3]}')


# -- gpu -----------------------------------------------------------------------------------------------------------------
def stage_gpu():
    import glob, statistics
    import k3_smoke as ks
    from caminus.decode import dflash as dk
    from caminus.engine import passpolice
    from caminus.engine.flashnext_tp import FlashNextTP
    from gguf import GGUFReader
    for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)
    torch.cuda.set_device(0)
    ref = ks.load_ref(); cfg = dict(ref['models']['coder']); cfg['drafter'] = ref['drafter']; cfg['name'] = 'coder'
    smoke = a.smoke or sorted(glob.glob(devpaths.need('ARTIFACTS', 'smoke', '2*.json')))[-1]
    sp = json.load(open(smoke))['stages']['tp1']['plain']
    print(f'compaction_proactiveness {passpolice.proactiveness()}; smoke reference {smoke}', flush=True)
    layers = [int(v) for v in {x.name: x for x in GGUFReader(dk.DRAFTER, 'r').fields.values()}['dflash.target_layers'].contents()]
    dg = dk.geom(dk.DRAFTER); t0 = time.time()
    m = FlashNextTP(cfg['gguf'], ntok=4, max_ctx=2048, feature_layers=layers, vram_drafters=('mtp', ('dflash', dg, None)))
    DF = dk.DFlashK3Drafter(dk.DFlashK3(m, dk.DRAFTER)); DF.device_accept = False
    MT = mtp.MTPDrafter(mtp.MTPK3(m))
    VR = {0: '/sys/class/drm/card1/device/mem_info_vram_used', 1: '/sys/class/drm/card2/device/mem_info_vram_used'}
    vram = {d: int(open(p).read()) >> 20 for d, p in VR.items()}
    print(f'target + MTP + DFlash loaded in {time.time() - t0:.0f} s; VRAM used XTX {vram[0]} / XT {vram[1]} MiB', flush=True)
    T, N = m.ntok, a.n
    PD = dict(both=mtp.PolicyDrafter({'dflash': DF, 'mtp': MT}), both_mtp=mtp.PolicyDrafter({'dflash': DF, 'mtp': MT}, policy={None: 'mtp'}),
              df=mtp.PolicyDrafter({'dflash': DF}), mt=mtp.PolicyDrafter({'mtp': MT}), none=mtp.PolicyDrafter({}))
    check(PD['both'].active == 'dflash' and PD['both_mtp'].active == 'mtp' and PD['df'].active == 'dflash', 'the wrappers\' methods')
    check(PD['mt'].active == 'mtp' and PD['mt'].fallback == ('dflash', 'mtp'), f'the default policy with only MTP loaded: {PD["mt"].active} (fallback {PD["mt"].fallback}), no KeyError')
    check(PD['none'].active == 'plain' and PD['none'].fallback == ('dflash', 'plain') and PD['none'].max_pos is None, 'the default policy with nothing loaded: plain')
    code, prose, shell = (ks.prompt_tokens(cfg, t) for t in ('code', 'prose', 'shell'))

    def plain(p, n): return spec.generate(m, list(p), n, longctx.Repeat())

    def run(d, p, n, dev=False, content='code'):
        """one spec.generate through drafter d; DF.log = DFlash's own drafts (filler not logged), Log = what d returned"""
        DF.device_accept = dev; DF.log = []; lg = Log(d); st = {}
        if hasattr(d, 'select'): d.select(content)
        try: out = spec.generate(m, list(p), n, lg, st); err = None
        except Exception as e: out, err = None, f'{type(e).__name__}: {e}'; traceback.print_exc()
        r = dict(toks=out, dlog=DF.log, plog=lg.log, st=st, err=err); DF.log = None; DF.device_accept = False
        return r

    def acc(st): return 100 * st['accepted'] / max(1, st['gen_passes'] * (T - 1)) if st else float('nan')

    # (1) any prompt length, every arm
    print('(1) the generic loop at every prompt length mod 4', flush=True)
    LENS = (1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 200, 201, 202, 203); tot = {}
    for L in LENS:
        p = code[:L]; ref_ = plain(p, N)
        if L == 200: check(ref_[L:] == sp['code']['tokens'][:N], f'(1) the plain reference at 200 == the smoke reference\'s one-token tokens ({N})')
        R = dict(dflash=run(PD['both'], p, N), dflash_dev=run(PD['both'], p, N, dev=True), dflash_alone=run(PD['df'], p, N),
                 dflash_raw=run(DF, p, N), mtp=run(PD['both_mtp'], p, N), mtp_alone=run(PD['mt'], p, N), plain=run(PD['none'], p, N))
        for k, r in R.items():
            tot.setdefault(k, []).append(r['st'])
            check(r['err'] is None and r['toks'] == ref_, f'(1) L={L} (mod 4 = {L % 4}) {k}: == plain, {N} tokens'
                  + (f' (error {r["err"]})' if r['err'] else f'; acceptance {acc(r["st"]):.1f}%, {r["st"].get("gen_passes")} passes'))
        check('step_seconds' in R['dflash_dev']['st'], f'(1) L={L}: spec.generate handed the wrapper\'s DFlash generation to the device-accept loop')
        dl = R['dflash']['dlog']
        check(all(R[k]['dlog'] == dl for k in ('dflash_dev', 'dflash_alone', 'dflash_raw')) and len(dl) > 0,
              f'(1) L={L}: DFlash drafts pass for pass: generic (MTP idle) == device accept == alone == unwrapped ({len(dl)} draft calls; '
              + ', '.join(f'{k} {len(R[k]["dlog"])}' for k in ('dflash_dev', 'dflash_alone', 'dflash_raw')) + ')')
        check(R['mtp']['plog'] == R['mtp_alone']['plog'] and len(R['mtp']['plog']) > 0, f'(1) L={L}: MTP drafts with DFlash idle == MTP alone ({len(R["mtp"]["plog"])} calls)')
    for k, sts in tot.items():
        ga = sum(s.get('accepted', 0) for s in sts); gp = sum(s.get('gen_passes', 0) for s in sts)
        print(f'    (1) {k}: acceptance {100 * ga / max(1, gp * (T - 1)):.1f}% over {len(sts)} prompts ({ga} of {gp * (T - 1)} drafts)', flush=True)

    # (2) given tokens after a generation, two rounds, no caller flush
    print('(2) given tokens after a generation (spec.loop)', flush=True)
    for meth, d in (('dflash', PD['both']), ('mtp', PD['both_mtp'])):
        bad = []; n0 = 0
        for g in range(1, 10):
            L = 37 + g % 4; p = code[:L]; q1 = prose[:g]; q2 = shell[:(3 * g) % 7 + 1]
            m.reset(); d.reset(); d.select('code'); segs = []
            try:
                t1, i0, c = spec.loop(m, list(p), 0, 0, L + 16, d); segs.append((p, t1))
                t2 = t1 + q1; t3, i0, c = spec.loop(m, list(t2), i0, c, len(t2) + 16, d); segs.append((t2, t3))
                t4 = t3 + q2; t5, i0, c = spec.loop(m, list(t4), i0, c, len(t4) + 16, d); segs.append((t4, t5))
            except Exception as e: bad.append((g, f'{type(e).__name__}: {e}')); traceback.print_exc(); continue
            n0 += (1 + g >= T) + (1 + len(q2) >= T)
            for si, (pre, out) in enumerate(segs):
                if out != plain(pre, len(out) - len(pre)): bad.append((g, f'segment {si} != plain'))
        check(not bad, f'(2) {meth}: 1 .. 9 given tokens after a generation, two rounds ({n0} rounds open with a k = 0 pass): every segment == plain {bad[:3]}')

    # (3) snapshot / restore through the wrapper
    print('(3) snapshot / restore through PolicyDrafter', flush=True)
    PDIR = devpaths.artifacts('qsa', 'lc')
    ids = lambda f: [int(v) for v in open(os.path.join(PDIR, 'd1792', f)).read().split()]
    ctx = ids('ctx.ids'); qA = ids('q_needle_10.ids')[1792:]; qB = ids('q_rare_50.ids')[1792:]
    n = 1501; A, B = ctx[:n] + qA, ctx[:n] + qB; NS = 40
    print(f'    fill {n}; A {len(A)} tokens ({len(A) % 4} mod 4), B {len(B)} ({len(B) % 4} mod 4)', flush=True)

    def cont(d, st, toks):
        """generation, 3 given tokens (a k = 0 pass), generation, 2 given tokens (a filler pass), generation: (segments, log)"""
        lg = Log(d); segs = []
        t, st = longctx.generate(m, st, toks, 12, draft=lg); segs.append((list(toks), t))
        for q in (prose[50:53], shell[60:62]):
            t2 = t + q; t, st = longctx.generate(m, st, t2, 12, draft=lg); segs.append((t2, t))
        return segs, lg.log

    for meth, d in (('dflash', PD['both']), ('mtp', PD['both_mtp'])):
        d.select('code')
        try:
            st = longctx.fill(m, ctx, n, draft=d, progress=False); snap = longctx.snapshot(m, st, d)
            L1, s1 = Log(d), {}; gA1, _ = longctx.generate(m, st, A, NS, draft=L1, stats=s1)
            s = longctx.restore(m, snap, d); gB, _ = longctx.generate(m, s, B, NS, draft=d)
            s = longctx.restore(m, snap, d); L2, s2 = Log(d), {}; gA2, _ = longctx.generate(m, s, A, NS, draft=L2, stats=s2)
            s = longctx.restore(m, snap, d); gP, _ = longctx.generate(m, s, A, NS)
            check(gA1 == gA2 and L1.log == L2.log and s1['accepted'] == s2['accepted'] and s1['passes'] == s2['passes'],
                  f'(3) {meth}: A after restore (B in between) == A after the fill: tokens, drafts ({len(L1.log)} / {len(L2.log)} calls), '
                  f'acceptance ({s1["accepted"]} / {s2["accepted"]} in {s1["passes"]} / {s2["passes"]} passes)')
            check(gA1 == gP and gB != gA1, f'(3) {meth}: A == plain resumed; B differs from A (not vacuous)')
            # mid-generation: a snapshot with DFlash's inject pending, restored and continued, vs no snapshot
            s = longctx.restore(m, snap, d); L3 = Log(d); tA, stA = longctx.generate(m, s, A, 12, draft=L3)
            pend = getattr(DF.df, 'pending', None)
            snap2 = longctx.snapshot(m, stA, d)
            c1, lg1 = cont(d, stA, tA)
            s = longctx.restore(m, snap2, d); c2, lg2 = cont(d, s, tA)
            s = longctx.restore(m, snap, d); L6 = Log(d); tA6, stA6 = longctx.generate(m, s, A, 12, draft=L6); c3, lg3 = cont(d, stA6, tA6)
            print(f'    {meth}: DFlash inject pending at the mid-generation snapshot: {pend}', flush=True)
            check(c1 == c2 and lg1 == lg2, f'(3) {meth}: the continuation after restoring the mid-generation snapshot == the first ({len(lg1)} / {len(lg2)} draft calls)')
            check(tA6 == tA and L6.log == L3.log and c3 == c1 and lg3 == lg1,
                  f'(3) {meth}: == the same flow without the snapshot: tokens and drafts ({len(lg3)} draft calls, {sum(x == y for x, y in zip(lg3, lg1))} equal)')
            pl = [(pre, out) for pre, out in [(A, tA)] + c1]
            okp = [out == plain(pre, len(out) - len(pre)) for pre, out in pl]   # (from scratch: a new sequence, so last)
            check(all(okp), f'(3) {meth}: every continuation segment == plain from scratch over its prefix {okp}')
        except Exception as e:
            traceback.print_exc(); check(False, f'(3) {meth}: {type(e).__name__}: {e}')

    # (4) timing
    if not a.no_bench:
        print(f'(4) timing: the smoke texts (200-token prompts, 128 tokens), {a.reps} reps interleaved; compaction_proactiveness '
              f'{passpolice.proactiveness()}', flush=True)
        ARMS = (('dflash device accept (alone)', DF, True), ('dflash generic (alone)', DF, False),
                ('dflash device accept (wrapper, MTP idle)', PD['both'], True), ('dflash generic (wrapper, MTP idle)', PD['both'], False),
                ('mtp generic (alone)', MT, False), ('mtp generic (wrapper, DFlash idle)', PD['both_mtp'], False))
        P = {t: ks.prompt_tokens(cfg, t, 200) for t in ks.TEXTS}
        run(DF, P['prose'], 32, dev=True); run(DF, P['prose'], 32)   # warm-up
        B_ = {(an, t): [] for an, _, _ in ARMS for t in ks.TEXTS}
        for rep in range(a.reps):
            for t in ks.TEXTS:
                for an, d, dev in ARMS:
                    r = run(d, P[t], 128, dev=dev); st = r['st']
                    if r['err'] or r['toks'][200:] != sp[t]['tokens'][:128]: check(False, f'(4) {an} {t} rep {rep + 1}: != plain ({r["err"]})')
                    B_[(an, t)].append(((st['gen_tokens'] - 1) / st['gen_seconds'], acc(st), 1e3 * st['gen_seconds'] / st['gen_passes']))
        for an, _, _ in ARMS:
            row = []
            for t in ks.TEXTS:
                v = B_[(an, t)]; row.append(f'{t} {statistics.median(x[0] for x in v):.1f} tok/s ({min(x[0] for x in v):.1f} .. {max(x[0] for x in v):.1f}), '
                                            f'acc {statistics.median(x[1] for x in v):.1f}%, {statistics.median(x[2] for x in v):.2f} ms/pass')
            print(f'    BENCH {an}: ' + '; '.join(row), flush=True)
    print(f'replicas {m.check_replicas()}; bails: target {len(m.bails)}, DFlash {len(DF.df.bails)}, MTP {len(MT.m.bails)}; DFlash stats {DF.df.stats}; '
          f'compaction_proactiveness {passpolice.proactiveness()}', flush=True)
    check(m.check_replicas() == (True, True), 'replicas equal')


stage_cpu() if a.stage == 'cpu' else stage_gpu()
print('RESULT ' + ('PASS' if not fails else f'FAIL {fails}'), flush=True)
sys.exit(0 if not fails else 1)
