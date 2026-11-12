"""Forced-bail gate for FlashNextTP's pass retry: a teacher-forced schedule of passes (the code token file), run once
uninterrupted (the reference) and once with bails injected; every pass's logits must be bitwise equal and the replicas
equal at the end, and every injected pass must have bailed.
  guard.sh LOG SECS python -I tools/gates/forcebail.py T NPASS [modes] [--model coder|reap|PATH] [--tokens FILE]
                                                       [--ctx MAX_CTX --start P]
    --ctx / --start: the engine at max_ctx MAX_CTX (past 2051: the Coder's QSA tapes) and the schedule from position P,
    after a batched prefill of [0, P) in every run (passes past 2051 run the indexer's pool selection: P >= 2052)
    modes: comma list cycled over the injected passes (default tiny,tiny0,tiny1,delay1), from
    tiny   both cards' wait bound 1 us for the first attempt (bails at the first wait that has to poll: depth ~0)
    tiny0 / tiny1   only card 0 / card 1 (the other card runs on, on the bailed card's garbage partials)
    delay1 card 1 launched 30 ms late, card 0's bound 2 ms (card 0 bails at its first exchange, card 1 does not)
    stress no injection: the bound is 20 ms (env LIMIT_US) and an external compaction stressor makes the bails
  T = 1: step() per position; T > 1: verify() with a commit schedule cycling 4, 1, 3, 2 (clamped to T).
  Injected passes: every 5th pass from pass 2 on. REPS=n (env) repeats the injected run.
Every injection is the designed bail path (a time bound that expires: the kernel finishes every task and reports err;
all reads and writes stay in bounds). Not for unattended runs: NEG=1 (env), the negative control (a retry with the
original commit, a double commit: the gate must FAIL), and the stress mode (needs an external compaction stressor).
Ported from tmp/m1/s5/forcebail_coder.py (the Coder, the default) and tmp/ceiling/s11/forcebail.py (--model reap) with the
same hooks and schedule. Expected (MANIFEST.md): T=1 40 and T=4 40: 40/40 passes bitwise equal, bails 8, RESULT PASS.
  --engine dense35: the dense own line's one-card engine (caminus.engine.q35_dense.Dense35Own, --model its GGUF, the card in
  HIP_VISIBLE_DEVICES; T = 1 only, every pass a verify at the next position; modes 'tiny' only; no --ctx / --start).
  SAMERUN=1 (env): a second uninterrupted run against the reference instead (determinism without any injection)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

a = sys.argv[1:]
PATH = C.model_path(C.opt(a, '--model', 'coder')); TOK = C.opt(a, '--tokens', C.TOKENS)
ENGINE = C.opt(a, '--engine', 'flashnext')
CTX = C.opt(a, '--ctx', 2048, int); START = C.opt(a, '--start', 0, int)
if not 2 <= len(a) <= 3: print(__doc__); sys.exit(2)
T = int(a[0]); NP = int(a[1]); MODES = (a[2] if len(a) > 2 else 'tiny,tiny0,tiny1,delay1').split(',')
torch = C.setup_torch()
toks = C.tokens(TOK)
print(f'model {PATH}; engine {ENGINE}; tokens {TOK}; T {T}; passes {NP}; modes {MODES}; compaction_proactiveness {C.proactiveness()}', flush=True)
DENSE = ENGINE == 'dense35'
if ENGINE == 'dense35':
    from caminus.engine.q35_dense import Dense35Own
    if T != 1 or START or MODES != ['tiny']: print('--engine dense35: T = 1, no --start, modes tiny'); sys.exit(2)
    m = Dense35Own(PATH, max_ctx=CTX)
    m._tapes = {'one': m.engines}
else:
    from caminus.engine.flashnext_tp import FlashNextTP
    m = FlashNextTP(PATH, ntok=T, max_ctx=CTX, **(dict(vram_prefill=2048) if START else {}))
if START:
    from caminus.prefill import coder as PF
    pf = PF.Prefill(m, chunk=2048)
print(f'max_ctx {CTX}; qsa {m.qsa}; start {START}', flush=True)
LIM = m.limit_us
stress = MODES == ['stress']
if stress: LIM = int(os.environ.get('LIMIT_US', 20000))


def schedule():
    """(i0, commit) per pass"""
    out, i0, commit, cyc = [], START, 0, [4, 1, 3, 2]
    for p in range(NP):
        out.append((i0, commit))
        c = 1 if T == 1 else min(T, cyc[p % 4])
        i0 += c; commit = c
    return out


SCH = schedule(); assert SCH[-1][0] + T <= len(toks)
orig_pass = m._pass
inj = {'pass': -1, 'mode': None, 'first': False}


def wrap(e, c):
    orig = e.run
    def f(*a, **k):
        if inj['first'] and inj['mode'] == 'delay1' and c == 1: time.sleep(0.030)
        return orig(*a, **k)
    e.run = f


for tape in m._tapes.values():   # (every tape's engines: past 2051 the passes run on the QSA tape's, not m.engines at start)
    for c, e in enumerate(tape): wrap(e, c)


NEG = os.environ.get('NEG') == '1'   # negative control: retry with the original commit (double commit) -> must FAIL
last = {}


def hooked_pass(*args):
    if DENSE: i0, commit = args[0], args[1]                 # (Dense35Own._pass(i0, commit, trace=None))
    else: xt, ple, i0, commit = args
    if NEG:
        if inj['first'] or (i0 not in last): last[i0] = commit
        else: commit = last[i0]
    last[i0] = commit
    mode = inj['mode'] if inj['first'] else None
    NC = len(m.engines); lim = [LIM] * NC
    if mode == 'tiny': lim = [1] * NC
    elif mode == 'tiny0': lim = [1, LIM]
    elif mode == 'tiny1': lim = [LIM, 1]
    elif mode == 'delay1': lim = [2000, LIM]
    for c in range(NC): m.engines[c].limit_us = lim[c]
    bad = orig_pass(i0, commit) if DENSE else orig_pass(xt, ple, i0, commit)
    inj['first'] = False
    for c in range(NC): m.engines[c].limit_us = LIM
    return bad


m._pass = hooked_pass


def run(inject):
    m.reset(); out = []; nb0 = len(m.bails); got = {}
    if START:
        st = pf.run(toks, 0, 0, START)
        assert st['i0'] == START and st['commit'] == 0, st
    for p, (i0, commit) in enumerate(SCH):
        injp = inject and not stress and p >= 2 and (p - 2) % 5 == 0
        inj['mode'] = MODES[((p - 2) // 5) % len(MODES)] if injp else None; inj['first'] = injp
        nb = len(m.bails)
        lo = m.verify(toks, i0, commit) if DENSE else m.step(toks, i0) if T == 1 else m.verify(toks, i0, commit)
        out.append(lo.float().cpu().clone())
        if injp: got[p] = (inj['mode'], len(m.bails) - nb)
    return out, got, m.bails[nb0:]


if stress:
    for c in (0, 1): m.engines[c].limit_us = 200_000
ref, _, rb = run(False)
print(f'reference: {len(ref)} passes, bails {len(rb)}', flush=True)
if os.environ.get('SAMERUN') == '1':   # determinism check: a second uninterrupted run against the reference, no injection
    out2, _, rb2 = run(False)
    bad = [p for p, (x, y) in enumerate(zip(ref, out2)) if not torch.equal(x.view(torch.int32), y.view(torch.int32))]
    for p in bad[:12]:
        print(f'  differs (no injection): pass {p}: max |d logit| {float((ref[p] - out2[p]).abs().max()):.3e}', flush=True)
    print(f'SAMERUN: {len(ref) - len(bad)}/{len(ref)} passes bitwise equal without any injection; bails {len(rb2)}; replicas {m.check_replicas()}')
    print('RESULT', 'PASS' if not bad else 'FAIL'); sys.exit(0 if not bad else 1)
for e in m.engines: e.limit_us = LIM
REPS = int(os.environ.get('REPS', 1)); t0 = time.time(); same, got, bails = [], {}, []
for rep in range(REPS):
    out, g, b = run(True); got.update(g); bails += b
    eq = [torch.equal(a.view(torch.int32), b_.view(torch.int32)) for a, b_ in zip(ref, out)]
    same += eq
    for p, (x, y) in enumerate(zip(ref, out)):
        if not eq[p]:
            print(f'  differs: pass {p} (i0 {SCH[p][0]}, commit {SCH[p][1]}){" injected " + str(got.get(p)) if p in got else ""}: '
                  f'max |d logit| {float((x - y).abs().max()):.3e}, argmax {int(x.argmax())} vs {int(y.argmax())}', flush=True)
print(f'T={T}: injected {got}', flush=True)
for b in bails: print('  bail', b[0], 'attempt', b[1], '|', ' || '.join(x[:150] for x in b[2]))
miss = [p for p, (md, n) in got.items() if n == 0]
print(f'T={T} modes {MODES} x{REPS}: {sum(same)}/{len(same)} passes bitwise equal; bails {len(bails)}; injected passes without a bail {miss}; '
      f'replicas {m.check_replicas()}; {time.time() - t0:.1f} s', flush=True)
ok = all(same) and all(m.check_replicas()) and not miss and bails
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
