"""Host bound checks (no GPU: the process hides every device before torch loads, and the test fails if one is visible).
Every position-indexed entry point must refuse an out-of-range request with k3.ContextBoundsError BEFORE anything is
staged, read or launched: FlashNextTP.verify / step / step_token / verify_tokens / _pass (and a device refusal keeps it
refusing until reset()), FlashNextK3.step / run_layers / _launch, spec.generate (up front, against the engine's max_ctx
and the drafter's rows), DFlash2.inject / draft_graph / draft, k3.Engine's KV segments (the cache's rows go to the
device check), and tools/k3_smoke.py's arguments (refused with exit 2, nothing loaded).
  HIP_VISIBLE_DEVICES= python -I tests/test_bounds_host.py"""
import os, sys, random, subprocess, types
for v in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'): os.environ[v] = ''   # before torch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'python'))
import numpy as np
import torch
from caminus import k3, spec
from caminus.flashnext_tp import FlashNextTP
from caminus.flashnext import FlashNextK3
from caminus.dflash import DFlash2

assert not torch.cuda.is_available(), 'this test must run with no GPU visible'
E = k3.ContextBoundsError
fails = []


def refused(fn, what):
    try: fn()
    except E as e: return str(e)
    fails.append(f'{what}: not refused'); return None


def check(cond, what):
    if not cond: fails.append(what)


# -- k3.check_span / bounds_status -------------------------------------------------------------------------------------
k3.check_span(0, 1, 1); k3.check_span(7, 1, 8); k3.check_span(4, 4, 8, commit=4, ntok=4); k3.check_span(np.int64(3), 2, 8)
for args, kw, what in [((8, 1, 8), {}, 'i0 = max_ctx'), ((-1, 1, 8), {}, 'i0 = -1'), ((5, 4, 8), {}, 'straddles the end'),
                       ((0, 0, 8), {}, 'n = 0'), ((1.0, 1, 8), {}, 'float position'), ((True, 1, 8), {}, 'bool position'),
                       ((0, 4, 8), dict(commit=5, ntok=4), 'commit > ntok'), ((0, 4, 8), dict(commit=-1, ntok=4), 'commit < 0'),
                       ((0, 4, 8), dict(commit=1.5, ntok=4), 'float commit'), ((262140, 8, 262144), {}, 'past the 262K cap')]:
    refused(lambda: k3.check_span(*args, **kw), f'check_span {what}')
check(k3.bounds_status([0] * 8) is None, 'bounds_status: clean status reported')
check('pass position 2048 (+ T tokens) outside the KV cache rows [0, 2048)' in (k3.bounds_status([0, 1, 0, -1, 1, -1, 2048, 2048]) or ''), 'bounds_status position message')
check('lazy commit 5 outside [0, 4]' in (k3.bounds_status([0, 1, 0, -1, 1, -2, 5, 4]) or ''), 'bounds_status commit message')
check('clamped to 0' in (k3.bounds_status([0, 1, 0, -1, 1, -1, 9, 8]) or ''), 'bounds_status: the separate guard (bnd 1) clamps')
check('skipped its whole tape' in (k3.bounds_status([0, 1, 0, -1, 2, -1, 9, 8]) or ''), 'bounds_status: k3_run (bnd 2) skips')
check(k3.SYNC_WORDS == 2 * k3.MAXSEG + 8 and k3.STATUS_WORDS == 8, 'sync block size')


# -- k3.Engine: the KV ops register their position tensor with the cache's rows for the device guard ------------------
def engine(T):
    e = object.__new__(k3.Engine); e.segs, e.keep, e.T = [], [], T; e.guard_caps, e.guard_commit, e._glc = {}, False, None; e.guard_launch = False
    e.params = torch.zeros(16, dtype=torch.int32); return e


for q8, M in ((True, 256), (False, 256), (True, 512)):
    pb = k3.kv_pos_bytes(M, q8); check(pb == (2 * (M // 32) * 34 if q8 else 4 * M), f'kv_pos_bytes {M} {q8}')
    for T in (1, 4):
        e = engine(T); cache = torch.zeros(37 * pb, dtype=torch.uint8); pos = torch.zeros(1, dtype=torch.int32)
        k_, v_ = torch.zeros(T * M), torch.zeros(T * M); qo, att = torch.zeros(T * 12 * M), torch.zeros(T * 12 * M)
        e.kv_append(k_, v_, cache, pos, q8=q8); e.attn_dec(qo, cache, pos, att, 12, M // 256 or 1, 256 if M >= 256 else M, 0.0625, q8=q8)
        check(e.segs[0].op == k3.OP_KV_APPEND and e.segs[0].K == 0 and e.segs[0].N == M, f'kv_append segment {q8} {M} T={T}')   # tape bytes as before
        check(e.segs[1].op == k3.OP_ATTN_DEC and e.segs[1].pad2 == 0, f'attn_dec segment {q8} {M} T={T}')
        check(list(e.guard_caps) == [pos.data_ptr()] and e.guard_caps[pos.data_ptr()][1] == 37 and not e.guard_commit,
              f'guard registration {q8} {M} T={T}: {[(k, v[1]) for k, v in e.guard_caps.items()]}')
        small = torch.zeros(20 * pb, dtype=torch.uint8); e.kv_append(k_, v_, small, pos, q8=q8)
        check(e.guard_caps[pos.data_ptr()][1] == 20, 'guard: the smallest cache on a position tensor bounds it')
# the cached guard / stage launches are rebuilt when a tape gains a guarded op after them (stale bounds otherwise)
e = engine(1); e._glc = 'cached'; e.kv_append(torch.zeros(256), torch.zeros(256), torch.zeros(9 * 544, dtype=torch.uint8), torch.zeros(1, dtype=torch.int32), q8=True)
check(e._glc is None, 'guard: kv_append keeps a stale cached guard launch')
e._glc = 'cached'; e.gdn_conv(torch.zeros(64), torch.zeros(64, 4), torch.zeros(64, 3), torch.zeros(64), torch.zeros(64))
check(e._glc is None, 'guard: gdn_conv keeps a stale cached guard launch')
e = engine(2); C = 64
e.gdn_conv(torch.zeros(2 * C), torch.zeros(C, 4), torch.zeros(C, 3), torch.zeros(2 * C), torch.zeros(2 * C))
check(e.guard_commit and not e.guard_caps, 'guard: gdn_conv registers the commit check')
try:
    engine(1).kv_append(torch.zeros(256), torch.zeros(256), torch.zeros(37 * 544 + 1, dtype=torch.uint8), torch.zeros(1, dtype=torch.int32), q8=True)
    fails.append('kv_append: a cache of a fractional row count accepted')
except ValueError: pass


# -- FlashNextTP: refused before anything is staged / read / launched ----------------------------------------------------
class Counter:
    def __init__(self): self.n = 0
    def __call__(self, *a, **k): self.n += 1


def tp(ntok, max_ctx):
    m = object.__new__(FlashNextTP)
    m.ntok, m.max_ctx, m.poisoned, m.fresh, m.token, m.ple_guess, m._during = ntok, max_ctx, None, True, 0, 0, None
    m.qsa_limit = None   # (the exact-attention limit: its own cases below)
    m.max_retries, m.bails, m.state, m.devs, m.vh, m._tok = 0, [], {}, (0, 1), 3, False
    m.h_i32 = np.zeros(64, dtype=np.int32); m.h_res = np.zeros((ntok, 1, 4), np.float32); m.h_ple = np.zeros((ntok, 4), np.float32)
    m.calls = types.SimpleNamespace(start=Counter(), emb=Counter(), run=Counter(), pass_=Counter())
    m.ple_layers = {3}; m._dev = lambda c: 'cpu'
    m.ple = types.SimpleNamespace(start=lambda *a: (m.calls.start(), None)[1], embeddings=lambda h: np.zeros((ntok, 4), np.float32))
    m._emb_row = lambda tok: (m.calls.emb(), np.zeros(4, np.float32))[1]
    m.engines = [types.SimpleNamespace(run=m.calls.run, sync=None) for _ in (0, 1)]
    m.logits_half = [torch.zeros(ntok, 3), torch.zeros(ntok, 2)]
    m.h_max = [(torch.zeros(ntok), torch.zeros(ntok, dtype=torch.int64)) for _ in (0, 1)]
    orig = FlashNextTP._pass
    def stub_pass(xt, ple, i0, commit):
        m._check(None, i0, commit, 'FlashNextTP._pass'); m.calls.pass_(); m.token += 1; return []
    m._pass = stub_pass; m._orig_pass = types.MethodType(orig, m)
    return m


def untouched(m, what, fresh=True, token=0):
    c = m.calls
    check(c.start.n == c.emb.n == c.run.n == c.pass_.n == 0, f'{what}: work done before the refusal '
          f'(PLE reads {c.start.n}, embeddings {c.emb.n}, launches {c.run.n}, passes {c.pass_.n})')
    check(m.fresh == fresh and m.token == token, f'{what}: state changed (fresh {m.fresh}, token {m.token})')


for ntok, max_ctx in ((1, 8), (4, 8), (4, 262144)):
    toks = list(range(max_ctx + 8)) if max_ctx < 100 else None
    bad = [max_ctx - ntok + 1, max_ctx, -1, 10 ** 9] + ([max_ctx - 1] if ntok > 1 else [])
    for i0 in bad:
        tk = toks if toks is not None else [0] * (min(max(i0, 0), max_ctx) + ntok + 1)
        for name, fn in [('verify', lambda m: m.verify(tk, i0, 0)), ('verify_tokens', lambda m: m.verify_tokens(tk, i0, 0))] + \
                        ([('step', lambda m: m.step(tk, i0)), ('step_token', lambda m: m.step_token(tk, i0))] if ntok == 1 else []):
            m = tp(ntok, max_ctx); refused(lambda: fn(m), f'TP ntok={ntok} max_ctx={max_ctx} {name}({i0})'); untouched(m, f'TP {name}({i0})')
        m = tp(ntok, max_ctx); m.h_i32[16] = i0 if -2 ** 31 <= i0 < 2 ** 31 else 0
        refused(lambda: m._orig_pass(None, None, i0, 0), f'TP _pass({i0})'); untouched(m, f'TP _pass({i0})')
    m = tp(ntok, max_ctx); tk = [0] * (ntok + 2)
    refused(lambda: m.verify(tk, 0, ntok + 1), f'TP ntok={ntok} commit {ntok + 1}'); untouched(m, 'TP commit > ntok')
    refused(lambda: m.verify(tk, 0, -1), f'TP ntok={ntok} commit -1'); untouched(m, 'TP commit -1')
    refused(lambda: m.verify(tk[:ntok - 1] if ntok > 1 else [], 0, 0), f'TP ntok={ntok} too few tokens'); untouched(m, 'TP too few tokens')
    # in range: reaches the pass (the checks are not vacuous), at the last positions of the cache too
    m = tp(ntok, max_ctx); i_last = max_ctx - ntok; tk = [1] * (i_last + ntok)
    m.verify(tk, i_last, 0); m.verify_tokens(tk, i_last, ntok)
    check(m.calls.pass_.n == 2 and m.calls.emb.n == 2 * ntok and m.calls.start.n == 2, f'TP ntok={ntok} in range: passes {m.calls.pass_.n}')
    # a device refusal poisons the engine until reset()
    m = tp(ntok, max_ctx); m.poisoned = 'segment 9 refused value 8 (bound 8)'
    refused(lambda: m.verify([1] * ntok, 0, 0), f'TP ntok={ntok} poisoned'); untouched(m, 'TP poisoned')
    m.reset(); m.verify([1] * ntok, 0, 0); check(m.calls.pass_.n == 1 and m.poisoned is None, 'TP reset() clears the refusal')

# the exact-attention limit (capacity.QSA_EXACT = 2051) is separate from the capacity: an engine with 262,144 rows still
# refuses a pass reaching position 2051 unless built with dense_beyond_qsa=True (qsa_limit None); an engine object without
# the attribute counts as limited (fail closed)
from caminus.capacity import QSA_EXACT
for ntok in (1, 4):
    for lim_attr in (QSA_EXACT, 'absent'):
        m = tp(ntok, 262144)
        if lim_attr == 'absent': del m.qsa_limit
        else: m.qsa_limit = lim_attr
        tk = [1] * (QSA_EXACT + 8)
        msg = refused(lambda: m.verify(tk, QSA_EXACT - ntok + 1, 0), f'TP ntok={ntok} past QSA_EXACT ({lim_attr})')
        check(msg is not None and 'dense_beyond_qsa=True' in msg, f'TP past QSA_EXACT: the message names the override: {msg}')
        untouched(m, f'TP ntok={ntok} past QSA_EXACT')
        m.verify(tk, QSA_EXACT - ntok, 0); check(m.calls.pass_.n == 1, f'TP ntok={ntok} at QSA_EXACT - ntok: not run')
    m = tp(ntok, 262144); m.qsa_limit = None; tk = [1] * 40000
    m.verify(tk, 32768, 0); check(m.calls.pass_.n == 1, f'TP ntok={ntok} dense_beyond_qsa / QSA: a pass at 32768 not run')
    m = tp(ntok, 262144); m.qsa_limit = None
    refused(lambda: m.verify([1] * 262200, 262144 - ntok + 1, 0), f'TP ntok={ntok} QSA past max_ctx'); untouched(m, 'TP QSA past max_ctx')

# which engines wire QSA (FlashNextTP.qsa_mode): the Coder with the indexer caches at max_ctx > QSA_EXACT (pass limit: max_ctx);
# REAP-308, the Coder at max_ctx <= 2051, without idx_cache, or qsa=False: the dense tape, limited to QSA_EXACT unless
# dense_beyond_qsa (which never wires QSA); qsa=True only where it can be built
qm = FlashNextTP.qsa_mode
check(qm('coder', True, False, 262144) == (True, None), 'qsa_mode: Coder 262144 -> QSA')
check(qm('coder', True, False, 2052) == (True, None), 'qsa_mode: Coder 2052 -> QSA')
check(qm('coder', True, False, 2051) == (False, QSA_EXACT), 'qsa_mode: Coder 2051 -> dense, limited')
check(qm('coder', True, False, 2048, True) == (True, None), 'qsa_mode: Coder 2048 qsa=True -> QSA')
check(qm('coder', True, False, 32768, False) == (False, QSA_EXACT), 'qsa_mode: Coder qsa=False -> dense, limited')
check(qm('coder', False, False, 32768) == (False, QSA_EXACT), 'qsa_mode: Coder without idx_cache -> dense, limited')
check(qm('coder', True, True, 32768) == (False, None), 'qsa_mode: dense_beyond_qsa -> dense, unlimited (test override)')
check(qm('reap', True, False, 262144) == (False, QSA_EXACT), 'qsa_mode: REAP-308 -> dense, limited')
for bad in (('reap', True, False, 32768, True), ('coder', False, False, 32768, True), ('coder', True, True, 32768, True)):
    try: qm(*bad); check(False, f'qsa_mode{bad}: accepted')
    except ValueError: check(True, f'qsa_mode{bad} refused')

# two-tape QSA (FlashNextTP._pick_tape): the dense tape for passes ending below QSA_EXACT, the QSA tape from there; the pools
# a QSA pass needs (j < i0 / 4) are backfilled exactly once per stretch the dense tape wrote; any pass lowers the mark to
# i0 / 4 (rejected drafts, a restore below an earlier fill), and a QSA pass raises it to (i0 + T) / 4
for ntok in (1, 4):
    m = object.__new__(FlashNextTP); m.ntok, m.dual, m._pool_ok, bf = ntok, True, 0, []
    m._tapes = dict(dense='D', qsa='Q'); m._backfill = lambda j0, j1: bf.append((j0, j1))
    seq = []
    for i0 in list(range(2040, 2060, ntok)) + [1500, 1504, 2060, 2049]:
        m._pick_tape(i0); seq.append((i0, m.engines, m._pool_ok))
    q0 = QSA_EXACT - ntok + 1   # the first i0 whose pass reaches position QSA_EXACT
    check(all(e == ('Q' if i0 >= q0 else 'D') for i0, e, _ in seq), f'two-tape ntok={ntok}: tapes {[(i, e) for i, e, _ in seq]}')
    first = min(i0 for i0, e, _ in seq if e == 'Q')
    want = [(0, first // 4), (1500 // 4, 2060 // 4)]
    check(bf == want, f'two-tape ntok={ntok}: backfills {bf} != {want}')
    check(seq[-1][2] == (2049 + ntok) // 4 and seq[-3][2] == 1500 // 4, f'two-tape ntok={ntok}: marks {[(i, p) for i, _, p in seq]}')

# the real _pass's device-refusal path: a refusal in either card's status words raises (no retry), clears both sync
# blocks and poisons the engine; the other card's status shows only the wait bail its err causes
import contextlib


class FakeSync:
    def __init__(self): self.zeroed = 0
    def __getitem__(self, k): return None
    def zero_(self): self.zeroed += 1


_dev, _sync = torch.cuda.device, torch.cuda.synchronize
torch.cuda.device = lambda d: contextlib.nullcontext(); torch.cuda.synchronize = lambda *a: None
try:
    for card in (0, 1):
        m = tp(1, 8); m._pass = m._orig_pass; m.h_i32[16] = 2; m.n_ex = 2; m.hstage = None
        words = [[0, 1, 7, 3, 0, 0, 0, 0], [0, 1, 7, 3, 0, 0, 0, 0]]; words[card] = [0, 1, 0, -1, 1, -1, 8, 8]
        m.h_st = [types.SimpleNamespace(copy_=lambda *a, **k: None, tolist=lambda w=w: list(w)) for w in words]
        m.stage = [types.SimpleNamespace(copy_=lambda *a, **k: None) for _ in (0, 1)]
        sy = [FakeSync(), FakeSync()]
        m.engines = [types.SimpleNamespace(run=m.calls.run, sync=sy[c]) for c in (0, 1)]
        m.ev_done = [types.SimpleNamespace(record=lambda: None, synchronize=lambda: None) for _ in (0, 1)]
        msg = refused(lambda: m._pass(None, None, 2, 0), f'device refusal on card {card}')
        check(msg is not None and f'card {card}' in msg and 'bound guard' in msg and 'pass position 8 (+ T tokens) outside the KV cache rows [0, 8)' in msg and m.bails == [], f'device refusal card {card}: {msg}')
        check(sy[0].zeroed == 1 and sy[1].zeroed == 1, f'device refusal card {card}: sync blocks not cleared ({sy[0].zeroed}, {sy[1].zeroed})')
        check(bool(m.poisoned), f'device refusal card {card}: engine not poisoned')
        refused(lambda: m.verify([1] * 4, 1, 0), f'device refusal card {card}: next pass accepted before reset()')
finally:
    torch.cuda.device, torch.cuda.synchronize = _dev, _sync


# -- FlashNextK3 (layer split) --------------------------------------------------------------------------------------------
k = object.__new__(FlashNextK3); k.max_ctx = 8; k.pos = None; k.engines = None
for i in (8, -1, 100):
    refused(lambda: k.step([0] * 200, i), f'FlashNextK3.step({i})')
    refused(lambda: k.run_layers(None, [0] * 200, i), f'FlashNextK3.run_layers({i})')
refused(lambda: k.step([0] * 3, 5), 'FlashNextK3.step: too few tokens')
refused(lambda: k._launch(0, 2), 'FlashNextK3._launch commit 2')
# a device bound-guard trip (Engine.run raising ContextBoundsError) poisons the layer-split engine until reset()
class TripEngine:
    def __init__(self): self.params, self.runs = torch.zeros(16, dtype=torch.int32), 0
    def run(self, trace=None):
        self.runs += 1; raise k3.ContextBoundsError('k3: the device bound guard caught an out-of-range value: test')
k = object.__new__(FlashNextK3); k.max_ctx = 8; k.devs = (1,); k.engines = [TripEngine()]; k.poisoned = None; k.state = {}
refused(lambda: k._launch(0, 1), 'FlashNextK3._launch device trip')
check(bool(k.poisoned) and k.engines[0].runs == 1, f'FlashNextK3: device trip not poisoned ({k.poisoned!r})')
for name, fn in (('step', lambda: k.step([0] * 8, 2)), ('run_layers', lambda: k.run_layers(None, [0] * 8, 2)), ('_launch', lambda: k._launch(0, 0))):
    refused(fn, f'FlashNextK3.{name} while poisoned')
check(k.engines[0].runs == 1, 'FlashNextK3: a launch while poisoned')
k.reset(); check(k.poisoned is None and k.fresh, 'FlashNextK3.reset() clears the refusal')
refused(lambda: k._launch(0, 1), 'FlashNextK3._launch after reset (trips again: launched)')
check(k.engines[0].runs == 2, 'FlashNextK3: no launch after reset()')


# -- spec.generate ---------------------------------------------------------------------------------------------------------
class FakeTP:
    """the verify_tokens interface: target token after position p = (7 * p) % 11 (deterministic); records positions"""
    def __init__(self, T, max_ctx): self.ntok, self.max_ctx, self.resets, self.top, self.passes = T, max_ctx, 0, 0, 0
    def reset(self): self.resets += 1
    def verify_tokens(self, seq, i0, commit):
        k3.check_span(i0, self.ntok, self.max_ctx, commit=commit, ntok=self.ntok)
        assert len(seq) >= i0 + self.ntok
        self.top = max(self.top, i0 + self.ntok); self.passes += 1
        return [(7 * (i0 + t)) % 11 for t in range(self.ntok)]


def drafter(rng, p_ok):
    def d(toks, k):   # right with probability p_ok per draft (the target's rule), else wrong
        n = len(toks); return [(7 * (n - 1 + j)) % 11 if rng.random() < p_ok else 99 for j in range(k)]
    return d


rng = random.Random(0); hit = 0
for trial in range(400):
    T = rng.choice((1, 2, 3, 4, 8)); P = rng.randint(1, 30); N = rng.randint(1, 40); need = spec.positions_needed(P, N, T)
    m = FakeTP(T, need); out = spec.generate(m, list(range(P)), N, drafter(rng, rng.choice((0.0, 0.5, 0.9, 1.0))))
    check(len(out) == P + N and m.top <= need, f'generate T={T} P={P} N={N}: top {m.top} > positions_needed {need}')
    hit += m.top == need
    m = FakeTP(T, need - 1)
    refused(lambda: spec.generate(m, list(range(P)), N, drafter(rng, 1.0)), f'generate T={T} P={P} N={N} at max_ctx {need - 1}')
    check(m.resets == 0 and m.passes == 0, f'generate refused late (resets {m.resets}, passes {m.passes})')
check(hit > 0, 'positions_needed never reached (not tight)')
dd = drafter(rng, 1.0); dd.max_pos = 20; m = FakeTP(4, 4096)
refused(lambda: spec.generate(m, list(range(10)), 10, dd), 'generate: the drafter rows too few')
check(m.passes == 0, 'generate: drafter refusal after a pass')
check(spec.DFlashDrafter(types.SimpleNamespace(K=torch.zeros(1, 33, 1, 1))).max_pos == 33, 'DFlashDrafter.max_pos')


# -- DFlash2 -------------------------------------------------------------------------------------------------------------
df = object.__new__(DFlash2); df.K = torch.zeros(1, 8, 1, 1); df.V = torch.zeros(1, 8, 1, 1)
refused(lambda: df.inject(6, torch.zeros(1, 3, 4)), 'DFlash2.inject past max_pos')
refused(lambda: df.inject(-1, torch.zeros(1, 2, 4)), 'DFlash2.inject at -1')
refused(lambda: df.draft_graph(5, 6, 2), 'DFlash2.draft_graph past max_pos')
refused(lambda: df._draft_eager(5, 7, 1), 'DFlash2.draft past max_pos')


# -- the pass limit up front (capacity.pass_limit): spec.generate / longctx refuse before any reset or pass -----------------
from caminus import capacity, longctx
for lim_attr in (QSA_EXACT, 'absent'):
    m = FakeTP(4, 262144)
    if lim_attr != 'absent': m.qsa_limit = lim_attr
    check(capacity.pass_limit(m) == QSA_EXACT, f'pass_limit ({lim_attr}): {capacity.pass_limit(m)}')
    refused(lambda: spec.generate(m, list(range(2040)), 40, drafter(rng, 1.0)), f'generate 2040 + 40 past the pass limit ({lim_attr})')
    check(m.resets == 0 and m.passes == 0, f'generate past the pass limit ({lim_attr}): resets {m.resets}, passes {m.passes}')
m = FakeTP(4, 262144); m.qsa_limit = None
out = spec.generate(m, list(range(2040)), 40, drafter(rng, 1.0)); check(len(out) == 2080 and m.top > QSA_EXACT, 'generate with qsa_limit None: not run past 2051')
check(capacity.pass_limit(types.SimpleNamespace(max_ctx=1000, qsa_limit=None)) == 1000, 'pass_limit: max_ctx below the limit')


class FakeLC(FakeTP):
    """FakeTP with what longctx touches (no device: devs empty, no PLE)"""
    def __init__(self, T, max_ctx, lim='absent'):
        super().__init__(T, max_ctx); self.bails, self.devs, self.poisoned, self.fresh, self.ple_layers = [], (), None, True, set()
        self.state = {(0, 'gdn'): torch.zeros(3), (1, 'gdn'): torch.zeros(3), (0, 'kv'): torch.zeros(5)}
        if lim != 'absent': self.qsa_limit = lim
    def reset(self):
        super().reset(); self.seq = getattr(self, 'seq', 0) + 1
        for t in self.state.values(): t.zero_()
    def verify(self, seq, i0, commit):
        self.verify_tokens(seq, i0, commit); return torch.zeros(self.ntok, 3)


for lim in (QSA_EXACT, 'absent'):
    m = FakeLC(4, 262144, lim); toks = list(range(4000))
    refused(lambda: longctx.fill(m, toks, 2100, progress=False), f'longctx.fill 2100 past the pass limit ({lim})')
    check(m.resets == 0 and m.passes == 0, f'fill past the pass limit ({lim}): resets {m.resets}, passes {m.passes}')
    st = dict(i0=2040, commit=4)
    refused(lambda: longctx.teacher_forced(m, st, toks, 40), f'longctx.teacher_forced 2040 + 40 past the pass limit ({lim})')
    refused(lambda: longctx.generate(m, st, toks[:2040], 40), f'longctx.generate 2040 + 40 past the pass limit ({lim})')
    check(m.passes == 0, f'teacher_forced / generate past the pass limit ({lim}): {m.passes} passes')
m = FakeLC(4, 262144, None); st = longctx.fill(m, list(range(4000)), 2100, progress=False)
check(st['i0'] == 2100 and m.top == 2100 and m.resets == 1, f'fill 2100 with qsa_limit None: {st["i0"]}, top {m.top}')
m = FakeLC(4, 8192, None); dd = types.SimpleNamespace(max_pos=1000)
refused(lambda: longctx.fill(m, list(range(4000)), 1500, draft=dd, progress=False), 'longctx.fill past the drafter rows')
check(m.resets == 0 and m.passes == 0, 'fill past the drafter rows: work done')
m = FakeLC(4, 8192, None); refused(lambda: longctx.generate(m, dict(i0=990, commit=4), list(range(990)), 20, draft=dd), 'longctx.generate past the drafter rows')
check(m.passes == 0, 'longctx.generate past the drafter rows: passes run')

# -- longctx snapshots: tagged with the sequence epoch and the restore log; stale ones refused, nothing restored ------------
S = longctx.StaleSnapshotError
def stale(fn, what, needle):
    try: fn(); fails.append(f'{what}: not refused')
    except S as e: check(needle in str(e), f'{what}: message {e}')
m = FakeLC(4, 8192, None); m.reset(); m2 = FakeLC(4, 8192, None); m2.reset()
g = m.state[(0, 'gdn')]
g.fill_(1.0); A = longctx.snapshot(m, dict(i0=100, commit=4))
s = longctx.restore(m, A); check(s['i0'] == 100 and float(g[0]) == 1.0, 'restore of a fresh snapshot')
g.fill_(2.0); B = longctx.snapshot(m, dict(i0=150, commit=4))   # (after a probe from A)
longctx.restore(m, B); longctx.restore(m, B); check(float(g[0]) == 2.0, 'restore B twice')
longctx.restore(m, A); g.fill_(7.0)                               # a probe from A (rows >= 100 rewritten)
stale(lambda: longctx.restore(m, B), 'restore B after a restore to A (100 < 150)', 'before a restore to position 100')
check(float(g[0]) == 7.0, 'refused restore changed the state')
longctx.restore(m, A); check(float(g[0]) == 1.0, 'A still restores after B was refused')
C = longctx.snapshot(m, dict(i0=100, commit=4)); longctx.restore(m, A); longctx.restore(m, C)   # (same position: fine)
stale(lambda: longctx.restore(m2, A), 'restore into another engine', 'another engine')
dA, dB = types.SimpleNamespace(snapshot=lambda: 'dA', restore=lambda s: None), types.SimpleNamespace(snapshot=lambda: 'dB', restore=lambda s: None)
D_ = longctx.snapshot(m, dict(i0=100, commit=4), dA)
stale(lambda: longctx.restore(m, D_, dB), 'restore with another drafter', 'another drafter')
stale(lambda: longctx.restore(m, A, dA), 'restore a drafter-less snapshot with a drafter', 'no drafter')
longctx.restore(m, D_, dA); longctx.restore(m, D_)                # (with its own drafter, or none: fine)
m.reset()                                                          # a new fill / spec.generate / the drafter's own generate
for nm, sn in (('A', A), ('B', B), ('D', D_)): stale(lambda: longctx.restore(m, sn), f'restore {nm} after a reset', 'sequence')
stale(lambda: longctx.restore(m, {k: v for k, v in A.items() if k != 'tag'}), 'restore an untagged snapshot', 'untagged')
tp_ = object.__new__(FlashNextTP); tp_.state, tp_.poisoned = {}, None; FlashNextTP.reset(tp_); s0 = tp_.seq; FlashNextTP.reset(tp_)
check(tp_.seq == s0 + 1, 'FlashNextTP.reset grows seq')

# -- FlashNextTP.finish_pass: at most max_retries + 1 attempts (as _run) -------------------------------------------------------
for R in (0, 1, 3, 8):
    m = tp(4, 64); m.max_retries = R; n = [0]
    m._pass_wait = lambda i0: ['bail']; m._pass = lambda *a: (n.__setitem__(0, n[0] + 1), ['bail'])[1]
    try: m.finish_pass(0, 0); fails.append(f'finish_pass R={R}: no raise')
    except RuntimeError as e: check(f'in {R + 1} attempts' in str(e), f'finish_pass R={R}: {e}')
    check(n[0] == R and len(m.bails) == R + 1, f'finish_pass R={R}: {n[0] + 1} attempts, {len(m.bails)} bail records (want {R + 1})')
    m = tp(4, 64); m.max_retries = R; seq_ = [['bail']] * (R - 1) + [[]]; n = [0]   # rerun R (the last allowed) succeeds
    m._pass_wait = lambda i0: ['bail']; m._pass = lambda *a: (n.__setitem__(0, n[0] + 1), seq_[n[0] - 1])[1]
    if R: check(m.finish_pass(0, 0) is True and n[0] == R, f'finish_pass R={R}: the last allowed rerun succeeds ({n[0]})')
    m = tp(4, 64); m.max_retries = R; m._pass_wait = lambda i0: []
    check(m.finish_pass(0, 0) is False and m.bails == [], f'finish_pass R={R}: a clean pass')
    m = tp(4, 64); m.max_retries = R; n = [0]
    m._pass = lambda xt, ple, i0, c: (n.__setitem__(0, n[0] + 1), ['bail'])[1]
    try: m._run([1] * 8, 0, 0, False); fails.append(f'_run R={R}: no raise')
    except RuntimeError as e: check(n[0] == R + 1 and f'in {R + 1} attempts' in str(e), f'_run R={R}: {n[0]} attempts')

# -- DFlashK3Drafter.generate: the step rerun loop capped at max_retries + 1 failed attempts ----------------------------------
from caminus.dflash_k3 import DFlashK3Drafter


class FakeDF:
    def __init__(self, R, waits): self.T, self.k, self.max_pos, self.n_inj, self.max_retries, self.waits, self.starts = 4, 3, 4096, 0, R, list(waits), 0; self.pending = None
    def reset(self): self.n_inj = 0
    def window_gap(self, n0): return False      # (no position given up: DFlashK3.skip)
    def step_check(self, i0, F, toks): pass     # (the host refusals before the target's launch: DFlashK3.step_check)
    def step_start(self, i0, F, toks, t_ev=False): self.starts += 1; self._i = (i0, F)
    def step_wait(self): return self.waits.pop(0) if self.waits else []
    def record(self): i0, F = self._i; return dict(a=[5] * 4, j=0, anchor=5, pos_d=i0 + F, flag=0)
    def drafts(self): return [1, 2, 3]
    def step_ms(self): return 1.0
    def flush(self): pass


class FakeTgt:
    def __init__(self, retried): self.ntok, self.max_ctx, self.qsa_limit, self.retried = 4, 4096, None, retried
    def reset(self): pass
    def launch_tokens(self, seq, i0, commit): pass
    def finish_pass(self, i0, commit): return self.retried


for R, waits, retried, raises, starts in ((3, [['b']] * 9, False, True, 4), (0, [['b']], False, True, 1), (0, [], True, False, 2),
                                          (2, [['b'], ['b']], False, False, 3), (2, [['b']] * 3, False, True, 3),
                                          (1, [[], ['b']], True, False, 3), (8, [['b']] * 20, True, True, 9)):
    df_ = FakeDF(R, waits); dr_ = DFlashK3Drafter(df_)
    try: dr_.generate(FakeTgt(retried), [7, 8], 1); got = False
    except RuntimeError as e: got = 'DFlashK3: a wait gave up in the step launch' in str(e)
    check(got == raises and df_.starts == starts, f'DFlashK3Drafter step cap R={R} waits {len(waits)} retried {retried}: raised {got}, '
                                                  f'{df_.starts} step launches (want {raises}, {starts})')

# -- k3.Engine.qsa_score / qsa_select argument checks (host: the segments are only recorded) -----------------------------
for T in (1, 4):
    rows = 4 * 1000; prow = rows // 4; nch = (prow + 31) // 32
    e = engine(T); e.module = 'k3_gguf_gsq_qsa' if T == 1 else 'k3_gguf_gsq_qsa_mt'; e.caps = None
    q = torch.zeros(T * 4 * 128); pool = torch.zeros(prow * 136, dtype=torch.uint8); pos = torch.zeros(1, dtype=torch.int32)
    sc = torch.zeros(T, prow); sel = torch.zeros(T * 516, dtype=torch.int32); sel2 = torch.zeros(T * 516, dtype=torch.int32)
    def bad(fn, what):
        try: fn(); fails.append(f'{what}: accepted')
        except ValueError: pass
    for nt in (0, -1, nch + 1, 2 ** 26, True, 1.5): bad(lambda: e.qsa_score(q, pool, pos, sc, ntask=nt), f'qsa_score T={T} ntask {nt!r}')
    for mn in (513, -1, 10 ** 6, 1.0): bad(lambda: e.qsa_score(q, pool, pos, sc, min_nv=mn), f'qsa_score T={T} min_nv {mn!r}')
    bad(lambda: e.qsa_score(q, pool, pos, sc, w=float('nan')), f'qsa_score T={T} weight NaN')
    check(len(e.segs) == 0, f'qsa_score T={T}: a refused call left a segment')
    s1 = e.qsa_score(q, pool, pos, sc, ntask=1); s2 = e.qsa_score(q, pool, pos, sc, ntask=nch, min_nv=0); s3 = e.qsa_score(q, pool, pos, sc, min_nv=512)
    check(e.segs[s1].ntask == 1 and e.segs[s2].ntask == nch and e.segs[s3].ntask == min(64, nch), f'qsa_score T={T}: ntask in the segment')
    for pr in (prow - 1, prow + 1, 0, True): bad(lambda: e.qsa_select(sc, pos, sel, pr, dep=s3), f'qsa_select T={T} pool_rows {pr!r} (SCORE has {prow})')
    bad(lambda: e.qsa_select(sc, pos, sel, prow), f'qsa_select T={T}: no dep on its SCORE')
    bad(lambda: e.qsa_select(sc, pos, sel, prow, dep=s2), f'qsa_select T={T}: dep on an earlier SCORE of the buffer, not the last')
    n0 = len(e.segs); a = e.qsa_select(sc, pos, sel, prow, dep=s3)
    b = e.qsa_select(sc, pos, sel2, prow, dep=a)                     # waits for the SCORE through the first select
    i = e.idx_append(torch.zeros(T * 128), torch.zeros(rows * 136, dtype=torch.uint8), pool, pos, torch.zeros(128), 1.0, 1e-6)
    try: e.qsa_select(sc, pos, sel2, prow, dep=i); fails.append(f'qsa_select T={T}: dep on an unrelated segment accepted')
    except ValueError: pass
    e.also(i, s3); e.qsa_select(sc, pos, sel2, prow, dep=i)        # (the SCORE as a second dependency of the dep)
    check(len(e.segs) == n0 + 4, f'qsa_select T={T}: segments {len(e.segs) - n0}')
    host = torch.zeros(T, 3 * prow); e.qsa_select(host, pos, sel, 2 * prow)   # a host-written score buffer: taken as given
    bad(lambda: e.qsa_select(host, pos, sel, 3 * prow + 1), f'qsa_select T={T}: pool_rows past the row stride')

# -- the VRAM checks: an unmeasured model refused, T > 4 extrapolated, the drafters' need counted -----------------------------
check(capacity.fixed_mib('coder', 4) == (18950, 12480) and capacity.fixed_mib('coder', 4, capacity.MODEL_BYTES['coder']) == (18950, 12480),
      'fixed_mib coder 4')
check(capacity.fixed_mib('coder', 4, capacity.MODEL_BYTES['coder'] + 1) is None and capacity.fixed_mib('other', 4) is None and
      capacity.fixed_mib('reap', 4, capacity.MODEL_BYTES['coder']) is None, 'fixed_mib: an unmeasured model is not None')
f8 = capacity.fixed_mib('coder', 8); check(f8 == (18950 + 4 * capacity.EXTRA_TOKEN_MIB, 12480 + 4 * capacity.EXTRA_TOKEN_MIB), f'fixed_mib T=8 {f8}')
check(capacity.fixed_mib('reap', 2) == (22480, 14720), 'fixed_mib reap 2')
_caps = capacity.caps_mib
G = {'n_attn': 12, 'nkv': 2, 'hd': 256, 'idx_dim': 128, 'idx_heads': 4, 'ctx_train': 262144, 'D': 2560, 'hc': 4}
try:
    capacity.caps_mib = lambda devs: [(23567.0, 109.0 if d == 0 else 30.0) for d in devs]
    def tpv(tensor_bytes, model='coder', ntok=4, max_ctx=2048):
        t = object.__new__(FlashNextTP); t.model_tag, t.ntok, t.max_ctx, t.idx_cache, t.qsa, t.geom, t.devs = model, ntok, max_ctx, True, max_ctx > 2051, G, (0, 1)
        t.rd = types.SimpleNamespace(tensors=[types.SimpleNamespace(n_bytes=tensor_bytes)]); return t
    try: tpv(capacity.MODEL_BYTES['coder'] - 4096)._check_vram((0, 0)); fails.append('FlashNextTP: an unmeasured model accepted')
    except capacity.CapacityError as e: check('no measured fixed VRAM' in str(e), f'unmeasured model message: {e}')
    plan = tpv(capacity.MODEL_BYTES['coder'])._check_vram((0, 0)); check(plan['fixed_mib'] == (18950, 12480), f'plan {plan["fixed_mib"]}')
    for n_, ok_ in ((2048, True), (32768, True), (65536, True), (131072, False)):   # the target + MTP (smoke_factory's reserve)
        res = capacity.drafter_need_mib('mtp', G, n_, 4)
        for how, args in (('vram_drafters', ((0, 0), ('mtp',))), ('vram_reserve_mib', (res,))):
            try: pl = tpv(capacity.MODEL_BYTES['coder'], max_ctx=n_)._check_vram(*args); got = True
            except capacity.CapacityError: got = False
            check(got == ok_, f'target + MTP at {n_} ({how}): accepted {got} (want {ok_}; MTP needs {res[0]:.0f} MiB per card)')
    pl = tpv(capacity.MODEL_BYTES['coder'], max_ctx=32768)._check_vram((0, 0), ('mtp', ('dflash', (5, 8, 128))))
    want = [a + b for a, b in zip(capacity.drafter_need_mib('mtp', G, 32768, 4), capacity.drafter_need_mib('dflash', G, 32768, 4, dflash_geom=(5, 8, 128)))]
    check(list(pl['drafters_mib']) == want, f'vram_drafters mtp + dflash: {pl["drafters_mib"]} != {want}')
    # the drafter's own check (the target loaded: its use now), and one device named twice (a one-card build): summed
    capacity.caps_mib = lambda devs: [(23567.0, 18950.0 + 1071 + 50) for d in devs]
    try: capacity.check((0, 1), capacity.drafter_need_mib('mtp', G, 131072, 4), 'MTPK3'); fails.append('MTP at 128K accepted after the target')
    except capacity.CapacityError: pass
    capacity.caps_mib = lambda devs: [(1000.0, 0.0) for d in devs]
    capacity.check((0, 1), (600, 600), 'two devices')
    try: capacity.check((1, 1), (600, 600), 'one device twice'); fails.append('check: a device named twice not summed')
    except capacity.CapacityError: pass
finally:
    capacity.caps_mib = _caps
check(capacity.drafter_ctx_bytes('dflash', None, 2051, dflash_geom=(5, 8, 128), ring=8192)[1] == 5 * 8192 * 8 * 128 * 4, 'dflash ring bytes (own ring)')


# -- the DFlash ring (impl:dflash-ring): a fixed 4096 rows, counted the same everywhere; the host's slot tags -----------------
from caminus import dflash_k3 as dk
DG = (5, 8, 128, 2048)
for n_ in (2048, 32768, 131072, 262144):
    for T_ in (1, 2, 4, 8):
        check(capacity.dflash_ring(2048, T_) == 4096, f'dflash_ring(2048, {T_}) = {capacity.dflash_ring(2048, T_)}')
        b_ = capacity.drafter_ctx_bytes('dflash', G, n_, T_, dflash_geom=DG)
        check(b_ == (0, 5 * 4096 * 8 * 128 * 4) == capacity.drafter_ctx_bytes('dflash', G, n_, T_, dflash_geom=DG[:3]),
              f'dflash ring bytes at {n_} T={T_}: {b_} (a fixed 4096 rows, not max_ctx)')
    bud = capacity.budget(G, n_, 4, 'coder', ('dflash',), dflash_geom=DG)
    row = [r for r in bud['rows'] if r[0].startswith('dflash ctx')]
    check(len(row) == 1 and row[0][2] == 80.0 and 'ring 4096 rows' in row[0][0], f'budget dflash row at {n_}: {row}')
check(capacity.dflash_ring(4096, 4) == 8192 and capacity.dflash_ring(3000, 8) == 8192 and capacity.dflash_ring(100, 8) == 4096,
      'dflash_ring grows with the window only')
for bad_ in ((0, 4), (2048, 0), (2048, True), (2048.0, 4)):
    try: capacity.dflash_ring(*bad_); fails.append(f'dflash_ring{bad_} accepted')
    except ValueError: pass
_caps2 = capacity.caps_mib
try:
    capacity.caps_mib = lambda devs: [(23567.0, 109.0 if d == 0 else 30.0) for d in devs]
    def tpv2(max_ctx):
        t = object.__new__(FlashNextTP); t.model_tag, t.ntok, t.max_ctx, t.idx_cache, t.qsa, t.geom, t.devs = 'coder', 4, max_ctx, True, max_ctx > 2051, G, (0, 1)
        t.rd = types.SimpleNamespace(tensors=[types.SimpleNamespace(n_bytes=capacity.MODEL_BYTES['coder'])]); return t
    for n_ in (2048, 32768, 262144):
        pl = tpv2(n_)._check_vram((0, 0), (('dflash', DG),))
        check(pl['drafters_mib'] == (120.0, 960.0 + 80.0) == capacity.drafter_need_mib('dflash', G, n_, 4, dflash_geom=DG, ring=4096),
              f'vram_drafters dflash at {n_}: {pl["drafters_mib"]} (want the 4096-row ring)')
        pl = tpv2(n_)._check_vram((0, 0), (('dflash', DG, 65536),))
        check(pl['drafters_mib'][1] == 960.0 + 5 * 65536 * 8 * 128 * 4 / capacity.MIB, f'vram_drafters dflash ring 65536 at {n_}: {pl["drafters_mib"]}')
    pl = tpv2(262144)._check_vram((0, 0), (('dflash', DG),))   # 262K + DFlash fits now (the old 5,120 MiB ring did not)
    check(pl['rows'][1][2] < 19594 - 30, f'262144 + DFlash: XT need {pl["rows"][1][2]:.0f} MiB')
finally:
    capacity.caps_mib = _caps2

# RingSlots by itself: the read / write spans of each kind, and the boundary slots of every read span
W_ = 2048
for T_ in (2, 4, 8):
    rs = dk.RingSlots(4096, W_, T_)
    check(rs.reads('draft', None, 5000) == (5000 - W_ + 1, 5000) and rs.reads('fused', 4998, 5000) == (5000 - W_ + 1, 4998)
          and rs.reads('step', 4998) == (4998 + 2 - W_, 4998) and rs.reads('inject', 4998) == (0, 0) and rs.reads('draft', None, 10) == (0, 10),
          f'RingSlots.reads T={T_}')
    check(rs.writes('step', 100) == (100, 100 + 2 * T_) and rs.writes('fused', 100, 102) == (100, 102 + T_) and
          rs.writes('inject', 100) == (100, 100 + T_) and rs.writes('draft', None, 7) == (7, 7 + T_), f'RingSlots.writes T={T_}')
    for kind, i0_, n0_ in (('draft', None, 9000), ('fused', 8998, 9000), ('step', 9000, None)):
        lo_, hi_ = rs.reads(kind, i0_, n0_)
        for p_, want in ((lo_, True), (lo_ - 1, False), (hi_ - 1, True), (hi_, False)):
            rs.set(lo_ - 64, hi_ + 64, True); rs.set(p_, p_ + 1, False)
            try: rs.check(kind, i0_, n0_); got = False
            except dk.StaleRingError: got = True
            check(got == want, f'RingSlots {kind} T={T_}: slot of {p_} (reads {lo_} .. {hi_ - 1}) invalid -> refused {got}, want {want}')
        rs.set(lo_, hi_, True); rs.tag[(lo_ + 7) & rs.mask] = lo_ + 7 - 4096   # an alias: the slot holds the row one ring back
        try: rs.check(kind, i0_, n0_); fails.append(f'RingSlots {kind}: an aliased slot accepted')
        except dk.StaleRingError: pass
for bad_ in ((4095, 2048, 4), (4096, 4090, 4), (2048, 2048, 4)):
    try: dk.RingSlots(*bad_); fails.append(f'RingSlots{bad_} accepted (not a power of two, or no spare slots beyond the window)')
    except AssertionError: pass


# DFlashK3's own host code (inject / flush / draft / step_start / step_wait / snapshot / restore / _launch) on a simulated
# device: the ring's slots (df.cache[0], one write id per row) against a full ring of max_pos rows (dv.full), every row the
# kernels' attention reads (rows max(0, pos_d - (W - 1)) .. pos_d + T - 1) compared with the full ring's
class RingDev:
    def __init__(self, df, seed):
        self.df, self.rng, self.wid = df, random.Random(seed), 0
        self.full = np.full(df.max_pos + 64, -1, dtype=np.int64); self.mism = 0; self.reads = 0; self.bail = []; self.fail_all = False

    def put(self, lo, hi):
        for x in range(lo, hi):
            self.wid += 1; self.full[x] = self.wid; self.df.cache[0][x & self.df.rmask] = self.wid

    def start(self, kind, i0=None, n0=None, anchor=None, F=None, toks=None, armed=False, t_ev=False):
        df, T = self.df, self.df.T
        if kind in ('inject', 'fused', 'step'): self.put(i0, i0 + T)
        if kind == 'step':
            j = self.rng.randrange(0, T - F + 1); n0 = i0 + F + j
            a = [(int(toks[q]) + 7) % df.emb_vocab for q in range(T)]
            for jj in range(j): a[F - 1 + jj] = int(toks[F + jj])
            if F + j < T: a[F - 1 + j] = (int(toks[F + j]) + 1) % df.emb_vocab
            df._rec = dict(a=a, j=j, anchor=a[F - 1 + j], pos_d=n0, flag=0)
        elif kind in ('draft', 'fused'): df._rec = dict(a=[0] * T, j=0, anchor=int(anchor), pos_d=n0, flag=0)
        if kind != 'inject':
            self.put(n0, n0 + T)
            lo = max(0, n0 - (df.window - 1)); xs = np.arange(lo, n0 + T)
            got = df.cache[0].numpy()[xs & df.rmask]; self.reads += xs.size; self.mism += int((got != self.full[xs]).sum())

    def wait(self, kind, armed=False):
        return ['bail'] if (self.fail_all or (self.bail and self.bail.pop(0))) else []


def ring_df(T, seed, max_pos=1 << 15):
    df = object.__new__(dk.DFlashK3)
    df.T, df.k, df.window, df.ring, df.rmask, df.max_pos, df.nl, df.dx = T, T - 1, W_, 4096, 4095, max_pos, 1, 0
    df.emb_vocab, df.max_retries, df.limit_us, df.bails, df.engines = 1000, 2, 1, [], {}
    df.cache = [torch.full((4096,), -1, dtype=torch.int64)]
    df.ring_tags = dk.RingSlots(4096, W_, T); df._ctl = dict(no_ring_restore=False, no_tag_check=False)
    dv = RingDev(df, seed); df._start = dv.start; df._wait = dv.wait; df.record = lambda: df._rec; df.drafts = lambda: [1] * (T - 1)
    df.reset(); return df, dv


def dev_loop(df, dv, i0, F, passes, toks):
    """DFlashK3Drafter.generate's device-accept passes from (i0, F): step_start / step_wait, then the accepted position"""
    for _ in range(passes):
        if i0 + 2 * df.T > df.max_pos: break
        df.step_start(i0, F, [toks[(i0 + t) % len(toks)] for t in range(df.T)])
        while df.step_wait(): df.step_start(i0, F, [toks[(i0 + t) % len(toks)] for t in range(df.T)])
        i0, F = df.record()['pos_d'], 1
    return i0


def prompt(df, P):
    """generate's prompt passes (inject + flush each) over P positions; returns (i0, F) of the last prompt pass"""
    i0 = 0
    while P - i0 > df.T: df.inject(i0); df.flush(); i0 += df.T
    return i0, P - i0


def host_loop(df, dv, i0, passes):
    """spec.loop's generic passes after a whole last prompt pass at i0 (its inject pending): then per pass the draft at the
    bonus position (fused with the previous pass's inject), the verify, its inject, the accept"""
    df.inject(i0); i0 += df.T
    for _ in range(passes):
        if i0 + 2 * df.T > df.max_pos: break
        df.draft(5, i0); df.inject(i0)
        i0 += 1 + dv.rng.randrange(0, df.T)
    df.flush()
    return i0


_dev, _sync = torch.cuda.device, torch.cuda.synchronize
_chk = capacity.check
try:
    torch.cuda.device = lambda d: __import__('contextlib').nullcontext(); torch.cuda.synchronize = lambda *a: None
    capacity.check = lambda devs, need, what: [(19594.0, 0.0, float(need[0]))]
    toks_ = [random.Random(3).randrange(0, 1000) for _ in range(997)]
    for T_ in (2, 4, 8):
        # forward flows past three ring wraps: never refused, every read row the full ring's
        df, dv = ring_df(T_, 11 + T_)
        i0_, F_ = prompt(df, 5001); dev_loop(df, dv, i0_, F_, 6000, toks_)
        check(dv.mism == 0 and dv.reads > 0, f'ring T={T_} device-accept loop to 3 x 4096: {dv.mism} of {dv.reads} read rows != the full ring')
        df.reset(); dv.full[:] = -1; i0_, F_ = prompt(df, 4096); host_loop(df, dv, i0_, 6000)
        check(dv.mism == 0, f'ring T={T_} generic loop past the wraps: {dv.mism} rows != the full ring')
        # bails (a launch run again) leave the same rows
        df.reset(); dv.full[:] = -1; dv.bail = [True, False, True]; i0_, F_ = prompt(df, 2000); dev_loop(df, dv, i0_, F_, 50, toks_)
        check(dv.mism == 0, f'ring T={T_} with bailed launches: {dv.mism}')
        # snapshot at N past the ring, a long leg (clobbers the window's slots), restore, probe: the ring is put back
        df.reset(); dv.full[:] = -1; n_ = 6000; i0_, F_ = prompt(df, n_ + 1); df.flush()
        snap = df.snapshot(); full0 = dv.full.copy()
        check(snap['tags'] is not df.ring_tags.tag and torch.equal(snap['ring'][0], df.cache[0]), f'ring T={T_}: snapshot copies')
        dev_loop(df, dv, i0_, F_, 3000, toks_)                    # the leg: past n + 2049, the window's slots overwritten
        df.restore(snap); dv.full[:] = full0; m0 = dv.mism
        dev_loop(df, dv, i0_, F_, 600, toks_)
        check(dv.mism == m0, f'ring T={T_}: probe after restore read {dv.mism - m0} rows != the full ring')
        # the controls: refused without K3_TEST_HOOKS; armed, the old restore (n_inj only) is refused by the tags, and with the
        # tags off too the probe reads aliased rows (the leg's), so the tags are load-bearing
        try: df.test_controls(no_ring_restore=True); fails.append('test_controls armed without K3_TEST_HOOKS')
        except ValueError: pass
        os.environ['K3_TEST_HOOKS'] = 'ring-controls'
        try:
            df.restore(snap); dv.full[:] = full0; dev_loop(df, dv, i0_, F_, 3000, toks_)   # clobber again
            df.test_controls(no_ring_restore=True); df.restore(snap); dv.full[:] = full0
            try: dev_loop(df, dv, i0_, F_, 600, toks_); fails.append(f'ring T={T_}: restore without the ring accepted')
            except dk.StaleRingError: pass
            df.test_controls(no_ring_restore=True, no_tag_check=True); df.restore(snap); dv.full[:] = full0; m0 = dv.mism
            dev_loop(df, dv, i0_, F_, 600, toks_)
            check(dv.mism > m0, f'ring T={T_}: the stale-ring control read no aliased row (vacuous)')
        finally:
            del os.environ['K3_TEST_HOOKS']
        df.test_controls()
        # a failed launch leaves its rows invalid: a later read of them (here: the caller drops the failed inject and drafts at
        # a position below n_inj) is refused, where the old ring read them as injected
        df.reset(); dv.full[:] = -1; i0_, F_ = prompt(df, 3001); df.inject(i0_ - T_)
        dv.fail_all = True
        try: df.draft(5, i0_ - T_ + 2); fails.append('a launch failing every attempt did not raise')
        except RuntimeError: pass
        dv.fail_all = False; df.pending = None
        try: df.draft(5, i0_); fails.append(f'ring T={T_}: rows of a failed launch read')
        except dk.StaleRingError: pass
        # draft-block rows are never read as injected rows (a draft past its block without an inject between)
        df.reset(); dv.full[:] = -1; i0_, F_ = prompt(df, 3001); df.inject(i0_); df.flush(); df.draft(5, i0_ + 1)
        try: df.draft(5, i0_ + 2); fails.append(f'ring T={T_}: a draft-block row read as injected')
        except dk.StaleRingError: pass
        # a back step past the ring (positions never re-injected since the front moved on): refused
        df.reset(); dv.full[:] = -1; i0_, F_ = prompt(df, 9001)
        try: df.draft(5, 3000); fails.append(f'ring T={T_}: a draft 6000 positions behind the front accepted')
        except dk.StaleRingError: pass
    # the snapshot's ring copy is capacity-checked first: refused, nothing copied
    capacity.check = lambda devs, need, what: (_ for _ in ()).throw(capacity.CapacityError(what))
    df, dv = ring_df(4, 5); i0_, F_ = prompt(df, 100); df.flush()
    try: df.snapshot(); fails.append('DFlashK3.snapshot: no capacity check')
    except capacity.CapacityError: pass
    try: df.restore(dict(n_inj=4)); fails.append('DFlashK3.restore: a snapshot without the ring accepted')
    except RuntimeError: pass
finally:
    torch.cuda.device, torch.cuda.synchronize = _dev, _sync
    capacity.check = _chk

# -- tools/k3_smoke.py arguments --------------------------------------------------------------------------------------------
sys.path.insert(0, os.path.join(REPO, 'tools'))
import k3_smoke
ns = lambda **kw: types.SimpleNamespace(**{**dict(n=128, tf=64, T=4, max_ctx=2048, prompt=200, spec='dflash,mtp'), **kw})
check(k3_smoke.check_args(ns()) == [], f'k3_smoke defaults refused: {k3_smoke.check_args(ns())}')
check(k3_smoke.check_args(ns(n=1848, spec='')) == [] and k3_smoke.check_args(ns(n=1846, T=4)) == [] and
      k3_smoke.check_args(ns(tf=2048)) == [], 'k3_smoke: exact fit refused')   # plain 200 + 1848; spec 200 + 1846 + 4 - 2
for kw in (dict(n=1849, spec=''), dict(n=1847, T=4), dict(tf=2049), dict(max_ctx=64), dict(n=0), dict(T=0)):
    check(k3_smoke.check_args(ns(**kw)) != [], f'k3_smoke accepted {kw}')
for P, N, T in ((200, 1846, 4), (1, 1, 1), (5, 1, 8), (200, 128, 4)):
    check(max(P - 1, P + N - 2) + T == spec.positions_needed(P, N, T), f'k3_smoke spec_need != spec.positions_needed {P} {N} {T}')
r = subprocess.run([sys.executable, '-I', os.path.join(REPO, 'tools', 'k3_smoke.py'), '--n', '4000'], capture_output=True, text=True,
                   env={**os.environ, 'HIP_VISIBLE_DEVICES': ''})
check(r.returncode == 2 and 'refusing the arguments (nothing launched)' in r.stdout, f'k3_smoke --n 4000: rc {r.returncode} {r.stdout[-200:]}')

print(('FAIL: ' + '\n  '.join([''] + fails)) if fails else 'PASS: every out-of-range request refused on the host, before any work')
sys.exit(1 if fails else 0)
