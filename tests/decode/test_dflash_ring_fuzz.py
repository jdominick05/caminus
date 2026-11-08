"""A fuzzer of the DFlash K3 drafter's host code (CPU only: no GPU is touched), both loops: the REAL host code runs
(DFlashK3.inject / flush / draft / step_start / step_wait / snapshot / restore / reset / _launch / _check_args with RingSlots,
DFlashK3Drafter.__call__ / flush / generate, caminus.decode.spec.loop / spec.generate, caminus.decode.mtp.PolicyDrafter); only the
device (_start / _wait / record / drafts) and the target are simulated. From the verifier's fuzz_ring.py of
verify-dflash-ring (the 4096-row ring, 7d2f0a2), widened by impl:dflash-loop to the generic loop's whole input space:
prompts of any length (the last prompt pass forcing 1 .. T tokens: filler drafts), given-token questions of any length
after a generation with no caller flush (k = 0 passes, filler passes), half of the generic runs through PolicyDrafter
(snapshot / restore / flush through the wrapper, and a 'switch' op: plain <-> DFlash, the idle DFlash flushed every pass),
and half of the device-accept runs through spec.generate + PolicyDrafter (the hand-off to DFlashK3Drafter.generate).
- fake target: an oracle LM whose next token is a hash of the prefix; each pass records the prefix hash of every position
  it ran (its "KV row by position") and leaves its T features (= those prefix hashes) in the feature buffer;
- simulated device: every ring write gets a unique id; a ledger by position (the full ring: latest write id, kind
  inject / draft, the feature key written). At every launch the kernel's actual read rows max(0, n0 - (W - 1)) .. n0 - 1
  (computed independently of RingSlots.reads) are checked AFTER the launch's own writes:
    alias  : ring slot's write id == the ledger's write id of that position (what a full ring would hold)
    kind   : the row is an injected row (never a draft-block row)
    key    : its feature key == the target's prefix hash of that position (the injected row of THIS sequence's tokens:
             an inject flushed after the next target pass rewrote the features would fail here)
  any failure on a launch the host accepted = a stale read the slot tags let through (BLOCKING);
  any StaleRingError in a valid flow = a false refusal (BLOCKING); any other exception (e.g. 'still pending') fails too;
  every generated token must be the oracle's greedy token.
Flows: prompts, generation segments, questions (given tokens), snapshots, restores (longctx's rule: only snapshots at
or below the position restored to stay valid), resets, bails (random, force_bail), failed launches (then restore /
reset), rejections aimed at the wraps, flush + draft pairs, invalid back steps (must be refused or read cleanly), policy
switches; M6 fix-ups: 'resume' (the device-accept loop from the current state, DFlashK3Drafter.resume, with given tokens
or an exact hit: tokens and the trimmed state == the generic loop from the same state) and 'gap' (passes the drafter never
sees, then skip_inject as the prompt cache does: drafts whose window reads the gap are filler, never a false refusal). Plus the negative controls (restore without the ring; tag check off) to show the detector is not vacuous.
Before impl:dflash-loop's fix the widened generic runs fail at once (k == T - 1 asserted, 'still pending', PolicyDrafter
without snapshot, 'plain' raising).
  HIP_VISIBLE_DEVICES= python -I tests/decode/test_dflash_ring_fuzz.py [seeds]      (default 1 2: ~1.5 min; FUZZ_T=2,4,8, FUZZ_ONLY=generic|device)"""
import os, sys, random, contextlib, time
for _v in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'): os.environ[_v] = ''   # before torch
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'python'))
import numpy as np
import torch
from caminus.decode import dflash as dk, spec, mtp
from caminus.device import capacity

M64 = (1 << 64) - 1
V = 1000


def mix(h, t):
    z = (h * 0x9E3779B97F4A7C15 + t * 0xD1B54A32D192ED03 + 0x632BE59BD9B4E019) & M64
    z ^= z >> 30; z = (z * 0xBF58476D1CE4E5B9) & M64; z ^= z >> 27; z = (z * 0x94D049BB133111EB) & M64; z ^= z >> 31
    return z & 0x7FFFFFFFFFFFFFFF


H0 = 12345
def nxt(h): return (h >> 3) % V        # the oracle's greedy next token after a prefix with hash h


class Tgt:
    def __init__(self, T, max_ctx):
        self.ntok, self.max_ctx, self.qsa_limit = T, max_ctx, None
        self.ph = np.full(max_ctx + 64, -1, np.int64)
        self.features = object(); self.feat = None; self.a = None; self.bail_next = 0; self.vh = V // 2; self.bails = []
        self.passes = 0

    def reset(self): self.ph[:] = -1; self.feat = None

    def _pass(self, seq, i0):
        T = self.ntok; assert len(seq) >= i0 + T, (len(seq), i0, T)
        h = int(self.ph[i0 - 1]) if i0 > 0 else H0
        assert h >= 0, f'target pass at {i0}: position {i0 - 1} never ran'
        keys, a = [], []
        for t in range(T):
            h = mix(h, int(seq[i0 + t])); self.ph[i0 + t] = h; keys.append(h); a.append(nxt(h))
        self.feat = (i0, keys); self.a = a; self.passes += 1
        return a

    def verify_tokens(self, seq, i0, commit): return self._pass(seq, i0)
    def launch_tokens(self, seq, i0, commit): self._pass(seq, i0)

    def finish_pass(self, i0, commit):
        if self.bail_next > 0: self.bail_next -= 1; return True
        return False


class Dev:
    def __init__(self, df, tg, rng, wrap_every=4096):
        self.df, self.tg, self.rng = df, tg, rng
        n = df.max_pos + 64
        self.wid = 0
        self.wid_ = np.full(n, -1, np.int64); self.key_ = np.full(n, -1, np.int64); self.kind_ = np.zeros(n, np.int8)
        self.ring = df.cache[0]
        self.reads = 0; self.launches = 0; self.viol = []; self.bail_p = 0.0; self.fail_next = False
        self.cur = []

    def ledger(self): return (self.wid_.copy(), self.key_.copy(), self.kind_.copy())
    def rewind(self, L): self.wid_[:], self.key_[:], self.kind_[:] = L

    def put(self, lo, hi, kind, keys=None):
        r = self.ring.numpy(); mask = self.df.rmask
        for t, x in enumerate(range(lo, hi)):
            self.wid += 1; self.wid_[x] = self.wid; self.kind_[x] = kind; self.key_[x] = keys[t] if keys is not None else -2
            r[x & mask] = self.wid

    def start(self, kind, i0=None, n0=None, anchor=None, F=None, toks=None, armed=False, t_ev=False):
        df, T, tg = self.df, self.df.T, self.tg
        self.launches += 1
        if kind in ('inject', 'fused', 'step'):
            self.put(i0, i0 + T, 1, tg.feat[1])        # the feature buffer: the latest target pass's features
        if kind == 'step':
            a = list(tg.a); j = 0
            while j < T - F and int(toks[F + j]) == a[F - 1 + j]: j += 1
            n0 = i0 + F + j; anchor = a[F - 1 + j]
            df._rec = dict(a=a, j=j, anchor=anchor, pos_d=n0, flag=0)
        elif kind in ('draft', 'fused'):
            df._rec = dict(a=[0] * T, j=0, anchor=int(anchor), pos_d=n0, flag=0)
        if kind == 'inject': return
        self.put(n0, n0 + T, 2)                         # the draft block (after the inject's rows)
        lo = max(0, n0 - (df.window - 1))               # the kernel's window (k3_draft_ops.h df_attn_window), row 0
        xs = np.arange(lo, n0)
        if xs.size:
            r = self.ring.numpy()[xs & df.rmask]
            bad_alias = r != self.wid_[xs]; bad_kind = self.kind_[xs] != 1; bad_key = self.key_[xs] != tg.ph[xs]
            self.reads += xs.size
            nb = int(bad_alias.sum()), int(bad_kind.sum()), int(bad_key.sum())
            if any(nb):
                w = np.flatnonzero(bad_alias | bad_kind | bad_key)[0]
                self.viol.append((kind, i0, n0, nb, int(xs[w])))
        # the drafts: the oracle's continuation (deterministic in the prefix), ~30 % of them corrupted
        h = int(tg.ph[n0 - 1]) if n0 > 0 else H0
        h = mix(h, int(anchor)); d = []
        for _ in range(T - 1):
            t_ = nxt(h)
            if (h >> 17) % 10 < 3: t_ = (t_ + 1) % V
            d.append(t_); h = mix(h, t_)
        self.cur = d

    def wait(self, kind, armed=False):
        if self.fail_next: return ['fail']
        if armed or (self.bail_p and self.rng.random() < self.bail_p): return ['bail']
        return []


def make(T, max_pos, seed):
    tg = Tgt(T, max_pos)
    df = object.__new__(dk.DFlashK3)
    W = 2048; ring = capacity.dflash_ring(W, T)
    df.T, df.k, df.window, df.ring, df.rmask, df.max_pos, df.nl, df.dx = T, T - 1, W, ring, ring - 1, max_pos, 1, 0
    df.emb_vocab = df.vocab = V; df.max_retries, df.limit_us, df.bails = 3, 1, []
    df.engines = {k: None for k in ('inject', 'draft', 'fused', 'step')}
    df.cache = [torch.full((ring,), -1, dtype=torch.int64)]
    df.ring_tags = dk.RingSlots(ring, W, T); df._ctl = dict(no_ring_restore=False, no_tag_check=False)
    dev = Dev(df, tg, random.Random(seed))
    df._start = dev.start; df._wait = dev.wait; df.record = lambda: df._rec; df.drafts = lambda: list(dev.cur)
    df.step_ms = lambda: 0.1; df.feats = tg.features
    df.reset()
    dr = dk.DFlashK3Drafter(df, device_accept=False)
    return tg, df, dev, dr


def greedy_ok(toks, given):
    """every generated position: toks[x + 1] == the oracle's next token after toks[:x + 1]"""
    h = H0
    for x in range(len(toks) - 1):
        h = mix(h, int(toks[x]))
        if not given[x + 1] and toks[x + 1] != nxt(h): return False, x + 1
    return True, None


class Fail(Exception):
    pass


def near_wrap(lo, hi, ring=4096):
    return (hi - 1) // ring != lo // ring or lo % ring == 0


def generic_run(T, seed, max_pos=20480, events=400, log=print, wrap=False):
    """the generic loop (spec.loop, host accept) with snapshot / restore / reset / bails / failures / back steps"""
    rng = random.Random(seed)
    tg, df, dev, dr0 = make(T, max_pos, seed)
    dr = mtp.PolicyDrafter({'dflash': dr0}, policy={'shell': 'plain', None: 'dflash'}) if wrap else dr0
    st = dict(switches=0, k0_questions=0, calls=0, snaps=0, restores=0, resets=0, fails=0, back_ok=0, back_refused=0, refused_valid=0, wrapcorrupt=0,
              resumes=0, resume_exact=0, gaps=0, gap_filler=0)
    def new_seq():
        P = rng.choice([1, 2, 3, T, T + 1, 200, 2046, 2047, 2048, 2049, 4090, 4093, 4095, 4096, 4097, 8189, 8192, 8195, rng.randrange(1, 9000)])
        P = min(P, max_pos - 4 * T - 64)
        P = max(1, P + rng.choice([0, 1, 2, 3]))   # (any length: the last prompt pass forces 1 .. T tokens)
        toks = [rng.randrange(V) for _ in range(P)]
        tg.reset(); dr.reset()
        return toks, [True] * P, 0, 0
    toks, given, i0, commit = new_seq()
    snaps = []
    def corrupt_wrap(n, d):
        r = df._rec
        p = r['pos_d']
        if near_wrap(p - T, p + 2 * T): st['wrapcorrupt'] += 1; return [(x + 1) % V for x in d]
        return d
    hist = []
    for ev in range(events):
        room = max_pos - len(toks) - 2 * T - 8
        op = rng.choices(['gen', 'snap', 'restore', 'reset', 'q', 'bail', 'fail', 'back', 'flushdraft', 'wrap', 'switch', 'resume', 'gap'],
                         [30, 8, 8, 2, 6, 4, 2, 4, 3, 4, 4 if wrap else 0, 8, 3])[0]
        if room < 64: op = rng.choice(['restore', 'reset']) if snaps else 'reset'
        try:
            if op in ('gen', 'bail', 'wrap', 'q'):
                if op == 'q':
                    nq = rng.choice([1, 2, T - 1, T, T + 1, rng.randrange(1, 4 * T + 2)])   # (no flush, any length: k = 0 / filler passes)
                    q = [rng.randrange(V) for _ in range(nq)]; toks = toks + q; given = given + [True] * len(q)
                    st['k0_questions'] += (1 + nq) >= T
                if op == 'bail':
                    dev.bail_p = 0.2
                    if rng.random() < 0.5: df.force_bail(rng.choice(['fused', 'draft', 'inject']), rng.randrange(1, 6))
                if op == 'wrap':   # land the front just before a ring multiple, then reject every draft near it
                    dr0.corrupt = corrupt_wrap
                n = rng.randrange(1, min(room, 600 if op != 'wrap' else 300))
                tot = len(toks) + n
                try:
                    toks, i0, commit = spec.loop(tg, toks, i0, commit, tot, dr)
                except RuntimeError as e:   # (bails past max_retries: the launch failed; recover as the 'fail' op does)
                    if isinstance(e, dk.StaleRingError) or 'a wait gave up' not in str(e): raise
                    st['fails'] += 1; dev.bail_p = 0.0; dr0.corrupt = None; df._force_bail = None
                    if snaps:
                        P, s, t_, g_, i0_, c_, L = snaps[-1]; dr.restore(s); toks, given, i0, commit = list(t_), list(g_), i0_, c_; dev.rewind(L)
                        snaps = [x for x in snaps if x[0] <= P]
                    else:
                        toks, given, i0, commit = new_seq()
                    continue
                given = given + [False] * (len(toks) - len(given))
                dev.bail_p = 0.0; dr0.corrupt = None; df._force_bail = None
                ok, x = greedy_ok(toks, given)
                if not ok: raise Fail(f'output != greedy at {x}')
            elif op == 'snap':
                s = dr.snapshot(); snaps.append((i0, s, list(toks), list(given), i0, commit, dev.ledger())); st['snaps'] += 1
            elif op == 'restore' and snaps:
                k = rng.randrange(len(snaps)); P, s, t_, g_, i0_, c_, L = snaps[k]
                dr.restore(s); toks, given, i0, commit = list(t_), list(g_), i0_, c_; dev.rewind(L)
                snaps = [x for x in snaps if x[0] <= P]; st['restores'] += 1
            elif op == 'reset' or (op == 'restore' and not snaps):
                toks, given, i0, commit = new_seq(); snaps = []; st['resets'] += 1
            elif op == 'fail':   # a launch failing every attempt in the next generation: then restore or reset
                dev.fail_next = True
                try:
                    spec.loop(tg, list(toks), i0, commit, len(toks) + 8, dr); raise Fail('a failing launch did not raise')
                except RuntimeError as e:
                    if isinstance(e, dk.StaleRingError): raise
                finally:
                    dev.fail_next = False
                st['fails'] += 1
                if snaps:
                    P, s, t_, g_, i0_, c_, L = snaps[-1]; dr.restore(s); toks, given, i0, commit = list(t_), list(g_), i0_, c_; dev.rewind(L)
                    snaps = [x for x in snaps if x[0] <= P]
                else:
                    toks, given, i0, commit = new_seq()
            elif op == 'back' and i0 > 8:   # INVALID flow: a draft behind the front; refused, or it must read clean rows
                s = dr.snapshot(); L = dev.ledger(); nv = len(dev.viol)
                n0b = rng.choice([max(0, i0 - rng.randrange(1, 64)), max(0, i0 - rng.randrange(1, 4200)), max(0, i0 - 2048), max(0, i0 - 2047)])
                try:
                    df.draft(5, n0b); st['back_ok'] += 1
                    if len(dev.viol) > nv: raise Fail(f'back step to {n0b} (front {i0}) accepted with a stale read: {dev.viol[nv:]}')
                except dk.StaleRingError:
                    st['back_refused'] += 1
                dr.restore(s); dev.rewind(L)
            elif op == 'resume':   # the device-accept loop from this state (DFlashK3Drafter.resume) == the generic loop from it
                s = dr.snapshot(); L = dev.ledger(); S = (list(toks), list(given), i0, commit)
                kind = rng.choice(['as is', 'given', 'exact'] if commit >= 1 and len(toks) == i0 + 1 else ['as is', 'given'])
                tk = list(toks)
                if kind == 'given': q = [rng.randrange(V) for _ in range(rng.randrange(1, 3 * T + 2))]; tk = tk + q
                if kind == 'exact': tk = tk[:i0]; st['resume_exact'] += 1
                n = rng.randrange(1, min(room, 300)); tot = len(tk) + n
                outA, stA = dr0.resume(tg, dict(i0=i0, commit=commit), tk, n)
                nvA = len(dev.viol)
                dr.restore(s); dev.rewind(L)
                outB, iB, cB = spec.loop(tg, list(tk), i0, commit, tot, dr0)   # (the raw drafter: a PolicyDrafter with plain active drafts filler)
                ex = max(0, len(outB) - tot)
                if outA != outB[:tot] or (stA['i0'], stA['commit']) != (iB - ex, cB - ex) or nvA:
                    raise Fail(f'resume ({kind}, from ({i0}, {commit}), {len(tk)} tokens, {n} new): device-accept {len(outA)} tokens {stA} vs generic '
                               f'{len(outB[:tot])} ({iB - ex}, {cB - ex}); equal tokens {outA == outB[:tot]}; stale reads {nvA}')
                toks, i0, commit = outB, iB, cB
                given = (S[1] + [True] * (len(tk) - len(S[0])) if kind == 'given' else S[1][:len(tk)])
                given = given + [False] * (len(toks) - len(given))
                ok, x = greedy_ok(toks, given)
                if not ok: raise Fail(f'resume: output != greedy at {x}')
                st['resumes'] += 1
            elif op == 'gap':   # passes the drafter never sees (a generation that bypassed it), then the prompt cache's rule: skip
                dr.flush()      # (as prepare() leaves it: nothing pending)
                n = rng.randrange(1, min(room, 120))
                toks, i0, commit = spec.loop(tg, toks, i0, commit, len(toks) + n, lambda t_, k_: [int(t_[-1])] * k_)
                given = given + [False] * (len(toks) - len(given))
                dr.skip_inject(i0); st['gaps'] += 1
                ok, x = greedy_ok(toks, given)
                if not ok: raise Fail(f'gap: output != greedy at {x}')
            elif op == 'switch':   # (PolicyDrafter: plain <-> DFlash; while plain, the idle DFlash flushes each pass)
                dr.select('shell' if dr.active == 'dflash' else 'code'); st['switches'] += 1
            elif op == 'flushdraft' and len(toks) - i0 == 1:   # an explicit flush, then a plain 'draft' launch at the front
                df.flush(); dr(toks, T - 1); st['calls'] += 1
        except dk.StaleRingError as e:
            raise Fail(f'FALSE REFUSAL in a valid flow (op {op}, event {ev}): {e}')
        except AssertionError as e:
            raise Fail(f'AssertionError (op {op}, event {ev}, len(toks) {len(toks)}, i0 {i0}, commit {commit}, last ops {hist[-6:]}): {e}')
        hist.append((op, len(toks), i0, commit))
        if dev.viol: raise Fail(f'STALE READ accepted (op {op}, event {ev}): {dev.viol[:3]}')
    if dev.viol: raise Fail(f'STALE READ accepted: {dev.viol[:3]}')
    st.update(launches=dev.launches, reads=dev.reads, passes=tg.passes, front=len(toks), gap_filler=df.stats.get('gap_filler', 0))
    return st


def device_run(T, seed, P, n_new, max_pos=20480, bails=True, corrupt_wrap=True, wrap=False):
    """DFlashK3Drafter.generate (the device-accept loop) from scratch: prompt P, n_new tokens, step / target bails, wraps"""
    rng = random.Random(seed)
    tg, df, dev, dr = make(T, max_pos, seed)
    dr.device_accept = True; nc = [0]
    if corrupt_wrap:
        def cw(n, d):
            # the next pass's write span: its i0 = the draft's anchor position (pos_d) .. + 2T
            p = df._rec['pos_d']
            if near_wrap(p - T, p + 2 * T): nc[0] += 1; return [(x + 1) % V for x in d]
            return d
        dr.corrupt = cw
    if bails:
        dev.bail_p = 0.05
        orig = tg.finish_pass
        def fp(i0, commit):
            if rng.random() < 0.03: return True
            return orig(i0, commit)
        tg.finish_pass = fp
    prompt = [rng.randrange(V) for _ in range(P)]
    st = {}
    try:
        out = spec.generate(tg, prompt, n_new, mtp.PolicyDrafter({'dflash': dr}), st) if wrap else dr.generate(tg, prompt, n_new, stats=st)
        if wrap and 'step_seconds' not in st: raise Fail('PolicyDrafter: spec.generate did not hand the generation to the device-accept loop')
    except dk.StaleRingError as e:
        raise Fail(f'FALSE REFUSAL in the device-accept loop: {e}')
    if dev.viol: raise Fail(f'STALE READ accepted (device loop): {dev.viol[:3]}')
    ok, x = greedy_ok(out, [True] * P + [False] * (len(out) - P))
    if not ok: raise Fail(f'device loop output != greedy at {x}')
    return dict(passes=st['passes'], accepted=st['accepted'], launches=dev.launches, reads=dev.reads, steps=df.stats['steps'],
                bails=len(df.bails), wrapcorrupt=nc[0], front=len(out))


def controls(T, seed):
    """the detector is not vacuous: restore without the ring (refused with the tags; read stale with the tags off)"""
    os.environ['K3_TEST_HOOKS'] = dk.RING_HOOKS
    try:
        rng = random.Random(seed)
        tg, df, dev, dr = make(T, 20480, seed)
        toks = [rng.randrange(V) for _ in range(6001)]; tg.reset(); dr.reset()
        toks, i0, commit = spec.loop(tg, toks, 0, 0, len(toks) + 50, dr)
        s = dr.snapshot(); L = dev.ledger(); S = (list(toks), i0, commit)
        toks2, i02, c2 = spec.loop(tg, list(toks), i0, commit, len(toks) + 3000, dr)   # the leg: past the window's slots
        res = {}
        df.test_controls(no_ring_restore=True); dr.restore(s); dev.rewind(L)
        try: spec.loop(tg, list(S[0]), S[1], S[2], len(S[0]) + 100, dr); res['refused'] = False
        except dk.StaleRingError: res['refused'] = True
        # again with the tag check off: the stale ring is read and the ledger must see it
        dr.restore(s); dev.rewind(L)   # (n_inj only: ring and tags as the leg left them... re-run the leg to clobber)
        df.test_controls(); dr.restore(s); dev.rewind(L)
        spec.loop(tg, list(S[0]), S[1], S[2], len(S[0]) + 3000, dr)
        df.test_controls(no_ring_restore=True, no_tag_check=True); dr.restore(s); dev.rewind(L); nv = len(dev.viol)
        spec.loop(tg, list(S[0]), S[1], S[2], len(S[0]) + 100, dr)
        res['stale_reads_seen'] = len(dev.viol) - nv
        df.test_controls()
        return res
    finally:
        del os.environ['K3_TEST_HOOKS']


def slices_prop(seed, n=20000):
    """RingSlots._slices / bad / set vs a naive per-position loop, lo / hi around ring multiples"""
    rng = random.Random(seed); bad = 0
    for T in (2, 4, 8):
        rs = dk.RingSlots(4096, 2048, T)
        for _ in range(n // 3):
            k = rng.randrange(0, 6) * 4096
            lo = max(0, k + rng.choice([-1, 0, 1, rng.randrange(-4100, 4100)]))
            hi = lo + rng.choice([0, 1, 2, 2047, 2048, 2049, 4095, 4096, rng.randrange(0, 4097)])
            hi = max(lo, min(hi, lo + 4096))
            tag = np.array([rng.choice([-1, s + 4096 * rng.randrange(0, 6), rng.randrange(-1, 30000)]) for s in range(4096)], np.int64)
            rs.tag[:] = tag
            naive = next((p for p in range(lo, hi) if tag[p & 4095] != p), None)
            if rs.bad(lo, hi) != naive: bad += 1
            inj = rng.random() < 0.5
            rs.set(lo, hi, inj); t2 = tag.copy()
            for p in range(lo, hi): t2[p & 4095] = p if inj else -1
            if not np.array_equal(rs.tag, t2): bad += 1
    return bad


if __name__ == '__main__':
    only = os.environ.get('FUZZ_ONLY')   # 'generic' / 'device' / None (all)
    seeds = [int(x) for x in sys.argv[1:]] or [1, 2]
    _dev, _sync, _chk = torch.cuda.device, torch.cuda.synchronize, capacity.check
    torch.cuda.device = lambda d: contextlib.nullcontext(); torch.cuda.synchronize = lambda *a: None
    capacity.check = lambda devs, need, what: [(19594.0, 0.0, float(need[0]))]
    fails = []
    t0 = time.time()
    b = slices_prop(7); print(f'slices/bad/set vs naive: {b} mismatches', flush=True)
    if b: fails.append(f'slices {b}')
    for T in [int(x) for x in os.environ.get('FUZZ_T', '2,4,8').split(',')]:
        c = controls(T, 100 + T); print(f'controls T={T}: {c}', flush=True)
        if not (c['refused'] and c['stale_reads_seen'] > 0): fails.append(f'controls T={T} {c}')
        for P, n in (((4090, 4200), (4095, 300), (8189, 4500), (2047, 9000), (1, 2100), (12285, 200)) if only != 'generic' else ()):
            for sd in seeds:
                try:
                    r = device_run(T, sd * 1000 + P, P, n, wrap=sd % 2 == 0); print(f'device T={T} P={P} n={n} seed {sd} wrap {sd % 2 == 0}: {r}', flush=True)
                except Exception as e:
                    print(f'FAIL device T={T} P={P} seed {sd}: {type(e).__name__}: {e}', flush=True); fails.append(str(e))
        for sd in (seeds if only != 'device' else ()):
            try:
                r = generic_run(T, sd, wrap=sd % 2 == 1); print(f'generic T={T} seed {sd} wrap {sd % 2 == 1}: {r}', flush=True)
            except Exception as e:
                print(f'FAIL generic T={T} seed {sd}: {type(e).__name__}: {e}', flush=True); fails.append(str(e))
    print(f'{time.time() - t0:.0f} s')
    print('RESULT PASS' if not fails else f'RESULT FAIL: {fails[:5]}')
    sys.exit(1 if fails else 0)
