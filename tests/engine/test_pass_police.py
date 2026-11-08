"""caminus.engine.passpolice on the CPU: the eviction-aware pass police's arithmetic and verdicts on synthetic timelines (a fake
clock, fake evicted_ms counters, a fake engine with FlashNextTP's launch halves and bail / retry loop). No GPU, no guard:
  HIP_VISIBLE_DEVICES= python -I tests/engine/test_pass_police.py
Covers: the largest per-node delta (not the sum: the real 230.3 ms pass of $ARTIFACTS/depth/fix/diag/d1_new_1.log), an
eviction longer than the pass (clamped), an eviction that began in the host gap before the launch (only the provable part
credited), an eviction between passes (credited to nothing), host stalls with and without an eviction, explained /
unexplained / forced bails inside and outside policed passes, the strict fallbacks (counters unreadable, no launch halves),
the test hooks' parsing and refusals, and the summary's required fields."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))
from caminus.engine import passpolice as pp

fails = []


def check(c, what):
    print(('ok   ' if c else 'FAIL ') + what, flush=True)
    if not c: fails.append(what)


def near(a, b, tol=1e-6): return abs(a - b) <= tol


class Clock:
    def __init__(self): self.t = 100.0
    def __call__(self): return self.t
    def ms(self, x): self.t += x / 1e3


class Counters:
    """fake evicted_ms: one counter per node (the real node order: XTX, XT, iGPU); broken: every sample fails"""
    def __init__(self, n=3): self.c, self.nodes, self.broken, self.failed, self.why = [0] * n, [f'stats_{i}' for i in range(n)], False, 0, None
    def sample(self):
        if self.broken: self.failed += 1; self.why = 'fake: unreadable'; return None
        return tuple(self.c)
    def fresh(self): return tuple(self.c)
    def evict(self, *ms):   # an eviction ENDS now: KFD adds its whole duration per node
        for i, v in enumerate(ms): self.c[i] += v


class Eng:
    def __init__(self): self.limit_us = 200_000


class FakeTP:
    """FlashNextTP's shape for the police: limit_us, bails, _tapes / engines, _pass_launch / _pass_wait, and verify_tokens
    = staging + _run's retry loop. plan[n] (n: the attempt number) = dict(ms, ev=(per node), bail): the attempt's time;
    an eviction that ends inside it (added just before its end); a bail. A launch with a 1 us bound bails"""
    def __init__(self, clk, cnt, stage_ms=0.5):
        self.clk, self.cnt, self.stage_ms = clk, cnt, stage_ms
        self.limit_us, self.max_retries, self.bails = 200_000, 8, []
        self.seq, self.token = 0, 0
        self._tapes = {'dense': [Eng(), Eng()], 'qsa': [Eng(), Eng()]}; self.engines = self._tapes['dense']
        self.plan, self.n_att, self.limits_seen = {}, 0, []
    def _pass_launch(self, i0, commit):
        self.limits_seen.append(sorted({e.limit_us for E in self._tapes.values() for e in E}))
        self._lim1 = any(e.limit_us == 1 for E in self._tapes.values() for e in E)
    def _pass_wait(self, i0):
        a = self.plan.get(self.n_att, {}); self.n_att += 1
        ms = a.get('ms', 20.0)
        self.clk.ms(ms * 0.99)
        if a.get('ev'): self.cnt.evict(*a['ev'])
        self.clk.ms(ms * 0.01)
        return ['card 0: segment 7 (seen 0)'] if (a.get('bail') or self._lim1) else []
    def verify_tokens(self, seq, i0, commit):
        self.clk.ms(self.stage_ms)
        for attempt in range(self.max_retries + 1):
            self._pass_launch(i0, commit); bad = self._pass_wait(i0)
            if not bad: break
            self.bails.append((i0, attempt, bad))
        else: raise RuntimeError('gave up')
        return [0, 0, 0, 0]


def setup(env=None, stage_ms=0.5, broken=False):
    clk, cnt = Clock(), Counters(); cnt.broken = broken
    m = FakeTP(clk, cnt, stage_ms)
    pol = pp.Police(source=cnt, clock=clk, sleep=lambda s: clk.ms(s * 1e3), env={} if env is None else env, out=open(os.devnull, 'w'))
    pol.attach(m)
    return clk, cnt, m, pol


def run_pass(m, pol, clk, i0, limit=150.0, gap_ms=3.0, gap_ev=None):
    """one policed pass (as longctx.fill), after gap_ms of host time (the drafter's inject in a real fill), during which an
    eviction may end (gap_ev). Returns the record, or the PassViolation"""
    clk.ms(gap_ms)
    if gap_ev: m.cnt.evict(*gap_ev)
    p = pol.begin('fill', i0)
    m.verify_tokens(None, i0, 4)
    try: return pol.end(p, limit)
    except pp.PassViolation as e: return e


# -- the arithmetic -------------------------------------------------------------------------------------------------------
check(pp.inside_ms(None, 3.0, 100.0) == 0 and pp.inside_ms(50, None, 100.0) == 0 and pp.inside_ms(0, 0.0, 9.0) == 0,
      'inside_ms: no sample / no earlier attempt / no eviction -> 0')
check(pp.inside_ms(100, 30.0, 200.0) == 70.0 and pp.inside_ms(100, 130.0, 200.0) == 0.0 and pp.inside_ms(300, 0.0, 120.0) == 120.0,
      'inside_ms: d - gap, clamped to [0, attempt time]')
check(pp.deltas((1, 2, 3), (4, 2, 10)) == (3, 0, 7) and pp.deltas(None, (1,)) is None and pp.deltas((5,), (4,)) == (0,),
      'deltas: per node, None when unsampled, never negative')

# -- clean passes, a slow pass without an eviction ------------------------------------------------------------------------
clk, cnt, m, pol = setup()
r = [run_pass(m, pol, clk, 4 * i) for i in range(3)]
check(all(isinstance(x, dict) and x['verdict'] == 'ok' and x['credit'] == 0 for x in r), 'clean passes: ok, nothing credited')
m.plan[m.n_att] = dict(ms=250.0)
r = run_pass(m, pol, clk, 12)
check(isinstance(r, pp.PassViolation) and '250.' in str(r) and '> 150.0 ms' in str(r), f'slow pass, no eviction: violation ({r})')

# -- the real pass (d1_new_1.log, i0 6772): 230.3 ms, attempt 229.8 ms, nodes 211 / 198 / 181 ms ------------------------------
clk, cnt, m, pol = setup()
run_pass(m, pol, clk, 6768)                                          # the previous pass (bounds the eviction's start)
m.plan[m.n_att] = dict(ms=229.8, ev=(211, 198, 181))
r = run_pass(m, pol, clk, 6772, gap_ms=3.0)
gap = 3.0 + 0.5                                                      # host time between the two attempts: the gap + staging
check(isinstance(r, dict) and r['verdict'] == 'forgiven', f'the real evicted pass: forgiven ({r if not isinstance(r, dict) else r["verdict"]})')
if isinstance(r, dict):
    check(r['ev'] == 211 and near(r['credit'], 211 - gap) and near(r['net'], 230.3 - (211 - gap), 1e-6),
          f'largest node (211), not the sum (590): credit {r["credit"]:.2f} = 211 - {gap} gap, net {r["net"]:.2f} ms')
    check(r['net'] > 0 and r['raw'] - min(590, r['raw']) == 0, 'the sum would have forgiven the whole pass (net 0); the max leaves the engine time')

# -- an eviction longer than the attempt (it began earlier, ended inside): clamped to the attempt ---------------------------
clk, cnt, m, pol = setup()
run_pass(m, pol, clk, 0)
m.plan[m.n_att] = dict(ms=180.0, ev=(400, 390, 380))
r = run_pass(m, pol, clk, 4, gap_ms=0.0)
check(isinstance(r, dict) and near(r['credit'], 180.0) and near(r['net'], 0.5) and r['verdict'] == 'forgiven',
      f'eviction (400 ms) longer than the attempt (180 ms): credit clamped to 180, net = staging 0.5 ({r if not isinstance(r, dict) else (r["credit"], r["net"])})')

# -- partial overlap: the eviction began in the host gap before the launch: only d - gap is provably inside -----------------
for g, want in ((30.0, 'forgiven'), (80.0, 'violation')):
    clk, cnt, m, pol = setup()
    run_pass(m, pol, clk, 0)
    m.plan[m.n_att] = dict(ms=200.0, ev=(100, 95, 90))
    r = run_pass(m, pol, clk, 4, gap_ms=g)
    got = r['verdict'] if isinstance(r, dict) else 'violation'
    cr = (100 - g - 0.5)
    check(got == want and (not isinstance(r, dict) or near(r['credit'], cr)),
          f'eviction 100 ms ending inside a 200 ms attempt after a {g:.0f} ms gap: credit {cr:.1f} ms -> {got} (want {want})')
    if not isinstance(r, dict): check('net > 150.0' in str(r).replace(' ms net >', ' net >') or '> 150.0 ms' in str(r), f'  the message: {r}')

# -- an eviction that ends between passes: credited to nothing, reported ----------------------------------------------------
clk, cnt, m, pol = setup()
run_pass(m, pol, clk, 0)
r = run_pass(m, pol, clk, 4, gap_ms=150.0, gap_ev=(150, 140, 130))
check(isinstance(r, dict) and r['ev'] == 0 and r['credit'] == 0 and r['verdict'] == 'ok', 'eviction in the gap between passes: not in the pass, nothing credited')
m.plan[m.n_att] = dict(ms=170.0)
r = run_pass(m, pol, clk, 8, gap_ms=1.0)
check(isinstance(r, pp.PassViolation), 'the next slow pass (no eviction of its own) is still a violation')
s = pol.summary()
check('evicted_ms over the run 150/140/130' in s, 'summary: the gap eviction is in the run total')

# -- K3_TEST_STALL: a host stall inside the timed span; with no eviction, and with one that ends inside the launch -----------
env = {'K3_TEST_HOOKS': 'police-controls', 'K3_TEST_STALL': '1:250'}
clk, cnt, m, pol = setup(env)
run_pass(m, pol, clk, 0)
r = run_pass(m, pol, clk, 4)
check(isinstance(r, pp.PassViolation) and 'K3_TEST_STALL 250 ms' in str(r) and 'eviction ended inside it: 0 ms' in str(r),
      f'stall 250 ms, no eviction: violation ({r})')
clk, cnt, m, pol = setup(env)
run_pass(m, pol, clk, 0)
m.plan[m.n_att] = dict(ms=20.0, ev=(150, 140, 130))                 # an eviction during the stall, ending inside the launch
r = run_pass(m, pol, clk, 4)
check(isinstance(r, pp.PassViolation) and '0.0 ms of it provably inside' in str(r),
      f'stall 250 ms + a 150 ms eviction ending in the launch: the stall is host time, nothing credited, violation')
check(pol.passes[-1]['raw'] > 270 and pol.passes[-1]['stall'] == 250.0, 'the stall is inside the timed span')

# -- bails ------------------------------------------------------------------------------------------------------------------
clk, cnt, m, pol = setup()                                           # base tree d1_base_1: i0 4636, attempts 266.8 + 29.3, 248 ms
run_pass(m, pol, clk, 4632)
m.plan[m.n_att] = dict(ms=266.8, ev=(248, 240, 232), bail=True); m.plan[m.n_att + 1] = dict(ms=29.3)
r = run_pass(m, pol, clk, 4636)
check(isinstance(r, dict) and r['verdict'] == 'forgiven' and r['bails'] == 1 and r['ok_bails'] and not pol.unexplained_bails(),
      f'bail with a 248 ms eviction inside its attempt, retry 29.3 ms: explained ({r if not isinstance(r, dict) else r["ok_bails"]})')
check(isinstance(r, dict) and near(r['net'], r['raw'] - (248 - 3.5)), f'  its pass net {r["net"]:.1f} ms (raw {r["raw"]:.1f})' if isinstance(r, dict) else 'n/a')

clk, cnt, m, pol = setup()
run_pass(m, pol, clk, 0)
m.plan[m.n_att] = dict(ms=201.0, bail=True)
r = run_pass(m, pol, clk, 4, limit=None)
check(isinstance(r, pp.PassViolation) and 'no eviction provably inside' in str(r), f'bail without an eviction: violation even with no time limit ({r})')

clk, cnt, m, pol = setup()
run_pass(m, pol, clk, 0)
m.plan[m.n_att] = dict(ms=205.0, ev=(20, 18, 15), bail=True); m.plan[m.n_att + 1] = dict(ms=25.0)
r = run_pass(m, pol, clk, 4, limit=None)
check(isinstance(r, pp.PassViolation) and '< the 200 ms wait bound' in str(r), f'bail with a 20 ms eviction (+ a 25 ms pass < 200): unexplained ({r})')

env = {'K3_TEST_HOOKS': 'police-controls', 'K3_TEST_BAIL': '1'}
clk, cnt, m, pol = setup(env)
run_pass(m, pol, clk, 0)
r = run_pass(m, pol, clk, 4)
check(isinstance(r, pp.PassViolation) and '[forced: K3_TEST_BAIL]' in str(r) and len(m.bails) == 1, f'K3_TEST_BAIL: a forced bail, no eviction: violation ({r})')
check(m.limits_seen[1] == [1] and m.limits_seen[2] == [200_000] and all(e.limit_us == 200_000 for E in m._tapes.values() for e in E),
      f'K3_TEST_BAIL: the 1 us bound on every tape for the first launch only, restored ({m.limits_seen[:3]})')

clk, cnt, m, pol = setup()                                           # bails outside policed passes (generate, the device-accept loop)
def unpoliced(i0, **a):
    """a pass outside any policed span (generate, the device-accept loop): its first attempt planned as a, then a 25 ms retry"""
    if a: m.plan[m.n_att] = a; m.plan[m.n_att + 1] = dict(ms=25.0)
    clk.ms(2.0); m.verify_tokens(None, i0, 4)
unpoliced(0, ms=230.0, ev=(232, 225, 220), bail=True)               # (the first attempt: no earlier attempt bounds the eviction)
unpoliced(4)
unpoliced(8, ms=230.0, ev=(232, 225, 220), bail=True)
unpoliced(12, ms=200.5, bail=True)
un = pol.unexplained_bails()
check(len(m.bails) == 3 and len(un) == 2 and any('at 0 ' in u for u in un) and any('at 12 ' in u for u in un),
      f'bails outside policed passes: the one after an earlier attempt explained, the very first one and the eviction-free one not ({un})')

# -- strict fallbacks -------------------------------------------------------------------------------------------------------
clk, cnt, m, pol = setup(broken=True)
run_pass(m, pol, clk, 0)
m.plan[m.n_att] = dict(ms=229.8, ev=(211, 198, 181))
r = run_pass(m, pol, clk, 4)
check(isinstance(r, pp.PassViolation) and 'STRICT' in str(r) and 'STRICT' in pol.mode(), f'counters unreadable: strict, the raw time judged ({r})')
m.plan[m.n_att] = dict(ms=230.0, ev=(232, 225, 220), bail=True)
clk.ms(2.0); m.verify_tokens(None, 8, 4)
check(len(pol.unexplained_bails()) == 1 and 'no eviction sample' in pol.unexplained_bails()[0], 'counters unreadable: every bail unexplained')


class Bare:   # an engine without launch halves (as tests/test_bounds_host's FakeLC)
    limit_us, bails = 200_000, []
    def verify_tokens(self, seq, i0, commit): clk.ms(230.0); cnt.evict(211, 198, 181); return [0]
clk, cnt = Clock(), Counters(); b = Bare(); pol = pp.Police(source=cnt, clock=clk, env={}).attach(b)
p = pol.begin('fill', 0); b.verify_tokens(None, 0, 0)
try: pol.end(p, 150.0); r = None
except pp.PassViolation as e: r = e
check(r is not None and 'no _pass_launch / _pass_wait' in str(r) and pol.mode().startswith('STRICT'), f'engine without launch halves: strict ({r})')
b.bails.append((4, 0, ['x']))
check(len(pol.unexplained_bails()) == 1, 'engine without launch halves: its bails unexplained')

k = pp.KfdEvictions(root=os.path.join(os.sep, 'nonexistent-kfd-root'))
check(k.sample() is None and k.why.startswith('no ') and k.failed == 1, f'KfdEvictions without the directory: None ({k.why})')

# -- the hooks: off by default, refused without the confirmation, malformed values refused -----------------------------------
check(pp.parse_hooks({}) == {}, 'no hook set: {}')
for env, what in (({'K3_TEST_STALL': '5:250'}, 'stall without K3_TEST_HOOKS'), ({'K3_TEST_BAIL': '5'}, 'bail without K3_TEST_HOOKS'),
                  ({'K3_TEST_STALL': '5:250', 'K3_TEST_HOOKS': '1'}, 'K3_TEST_HOOKS=1 (not the confirmation)'),
                  ({'K3_TEST_STALL': '5', 'K3_TEST_HOOKS': 'police-controls'}, "stall '5'"),
                  ({'K3_TEST_STALL': '-1:250', 'K3_TEST_HOOKS': 'police-controls'}, "stall '-1:250'"),
                  ({'K3_TEST_STALL': '5:0', 'K3_TEST_HOOKS': 'police-controls'}, "stall '5:0'"),
                  ({'K3_TEST_STALL': '5:9999', 'K3_TEST_HOOKS': 'police-controls'}, "stall '5:9999'"),
                  ({'K3_TEST_STALL': ' 5:250', 'K3_TEST_HOOKS': 'police-controls'}, "stall ' 5:250'"),
                  ({'K3_TEST_BAIL': 'x', 'K3_TEST_HOOKS': 'police-controls'}, "bail 'x'")):
    try: pp.parse_hooks(env); ok = False
    except ValueError: ok = True
    check(ok, f'hooks refused: {what}')
check(pp.parse_hooks({'K3_TEST_STALL': '5:250', 'K3_TEST_BAIL': '7', 'K3_TEST_HOOKS': 'police-controls'}) == {'stall': (5, 250.0), 'bail': 7},
      'hooks armed with the confirmation')

# -- the summary's required fields --------------------------------------------------------------------------------------------
clk, cnt, m, pol = setup()
run_pass(m, pol, clk, 0); m.plan[m.n_att] = dict(ms=229.8, ev=(211, 198, 181)); run_pass(m, pol, clk, 4)
s = pol.summary(); print(s)
for f in ('compaction_proactiveness', 'evicted passes 1', 'total evicted 211 ms', 'longest raw pass', 'longest net of evictions',
          'forgiven: fill at 4', 'test hooks: none', 'mode: eviction-aware', 'result: no unexplained'):
    check(f in s, f'summary has {f!r}')
print('PASS' if not fails else f'FAIL: {fails}')
sys.exit(1 if fails else 0)
