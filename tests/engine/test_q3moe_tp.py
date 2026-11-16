"""The qwen3moe two-card engine on two cards (caminus.engine.q3moe_tp.Q3MoETP on k3_q3m_lean_tp, torch devices 0 and 1):
bitwise against its own one-token reference, since by design it is not bitwise to the one-card engine (the K-split down,
the attn_output column split and the split count each change a summation order; its accuracy gate is
tools/gates/q3moe_gmodel.py --cards 2).
  - the reference: NPOS greedy one-token passes at each card's default grid (q3moe_one.card_per_wgp), the full logits'
    bytes kept; the replicas (the residual and every layer's router ids) equal after every pass; finite logits
  - K3_PER_WGP 3 / 6 / 7 (both cards' tapes rebuilt on the same weights and receive slots): every pass's logits the
    reference's bytes, the replicas equal after every pass
  - the served contract's pass forms at the default grid: step_token's greedy token = the reference logits' argmax;
    launch_tokens + finish_pass and verify give the reference's bytes
  - a span past max_ctx is refused on the host before any launch (step, step_token, launch_tokens), nothing poisoned
  - forced bails: on chosen passes the first attempt's wait bound is 1 us on both cards, on card 0 only and on card 1 only
    (the other card runs on, on the bailed card's garbage partial); every injected pass bailed, its retry (a fresh exchange
    epoch) gives the reference's bytes, the replicas stay equal, and the following passes too
  - every other run has 0 bails
--tiny: tests/engine/tiny_q3m.py's synthetic file with two-card shapes (a few MiB; the reference's first passes are also
compared with the float64 host forward, caminus.ref_q3moe: top-1 and KLD reported, logits checked not degenerate). Else
the GGUF given (the catalog Qwen3-30B-A3B file: about 11,900 / 6,300 MiB on the cards at max_ctx 128).
Both cards visible, card 0 first (the display card: run the tiny form first; every two-card run with CAMINUS_UPLOAD_LOG):
  CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 900 python -I tests/engine/test_q3moe_tp.py --tiny --scratch DIR
  CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 1800 python -I tests/engine/test_q3moe_tp.py GGUF --npos 96 --scratch DIR
--splits-cap N: the engine's attention splits cap (1 .. 128 on the two-card module; past 64 the merge's second load round
runs from position 4096 on: --npos 4200 --max-ctx 4300, logits then kept as digests past 512 passes).
Prints PASS / FAIL lines and RESULT PASS | FAIL."""
import argparse, hashlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, HERE)
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('gguf', nargs='?'); ap.add_argument('--tiny', action='store_true'); ap.add_argument('--scratch', default=None)
ap.add_argument('--npos', type=int, default=None); ap.add_argument('--max-ctx', type=int, default=None)
ap.add_argument('--splits-cap', type=int, default=None, help='the engine\'s splits_cap (default: q3moe_tp.SPLITS_CAP); its new\n               merge rounds run past 64 splits, i.e. past position 64 * 64: --npos 4200 --max-ctx 4300')
A = ap.parse_args()
if bool(A.tiny) == bool(A.gguf): raise SystemExit('test_q3moe_tp: give a GGUF or --tiny (one of them)')
if A.tiny and not A.scratch: raise SystemExit('test_q3moe_tp: --tiny writes its file under --scratch DIR')
if not os.environ.get('CAMINUS_GUARD'): raise SystemExit('test_q3moe_tp: run it under tools/guard.sh')
if torch.cuda.device_count() < 2: raise SystemExit(f'test_q3moe_tp: {torch.cuda.device_count()} card(s) visible: two needed')
from caminus.engine import k3
from caminus.engine.q3moe_tp import Q3MoETP, SPLITS_CAP

ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


bits = lambda t: t.detach().reshape(-1).cpu().numpy().view(np.uint32).copy()
if A.tiny:
    import tiny_q3m
    os.makedirs(A.scratch, exist_ok=True)
    P = os.path.join(A.scratch, 'tiny-q3m-tp.gguf'); tiny_q3m.write(P, tp=True)
    npos, first = A.npos or 32, [3, 17]
else:
    P, npos, first = A.gguf, A.npos or 96, [785, 6722, 9625]
max_ctx = A.max_ctx or npos + 32
if npos + 1 > max_ctx: raise SystemExit(f'--npos {npos} needs max_ctx > {npos}')
GRIDS = (3, 6, 7)
# each pass's logits: the bytes, or past 512 passes their digest, argmax and finiteness (4K passes of the 152K-row head
# would hold 2.5 GB per list)
FULL = bool(A.tiny) or npos <= 512


def keep(t):
    b = bits(t)
    if FULL: return b
    f = b.view(np.float32)
    return hashlib.sha256(b.tobytes()).digest(), int(np.argmax(f)), bool(np.isfinite(f).all())


eqb = lambda a, b: np.array_equal(a, b) if FULL else a[0] == b[0]
top = lambda r: int(np.argmax(r.view(np.float32))) if FULL else r[1]
finite = lambda r: bool(np.isfinite(r.view(np.float32)).all()) if FULL else r[2]
print(torch.cuda.get_device_name(0), '+', torch.cuda.get_device_name(1), flush=True)
# beside the projection (checked by the constructor with this): one more tape set per grid and one for the default again
eng = Q3MoETP(P, devices=(0, 1), max_ctx=max_ctx, vram_extra_mib=[(len(GRIDS) + 1) * Q3MoETP.TAPE_MIB] * 2,
              splits_cap=SPLITS_CAP if A.splits_cap is None else A.splits_cap)
print(f'{os.path.basename(P)}: expert rows {eng.exp_rows}, kv heads {eng.kvc}, head rows {eng.vh} / {eng.vocab - eng.vh}, '
      f'module {eng.module}, grids {eng.per_wgp_c}, splits {eng.splits}, {eng.vram_mib[0]:.0f} / {eng.vram_mib[1]:.0f} MiB', flush=True)
check(eng.module == 'k3_q3m_lean_tp' and eng.E[0].module == eng.E[1].module == 'k3_q3m_lean_tp', 'both cards decode on k3_q3m_lean_tp')


def run(seq_in, extend):
    """npos one-token passes from position 0: (the logits' bytes per pass, passes whose replicas differed, bails)"""
    eng.reset(); nb = len(eng.bails); out, rep = [], []
    seq = list(seq_in)
    for i in range(npos):
        lg = eng.step(seq[i]); out.append(keep(lg))
        if not all(eng.check_replicas()): rep.append(i)
        if extend and i + 1 == len(seq): seq.append(int(torch.argmax(lg)))
    return seq, out, rep, len(eng.bails) - nb


# ---- the reference at each card's default grid ---------------------------------------------------------------------------
seq, ref, rep, nb = run(first, True)
check(not rep and nb == 0 and all(finite(r) for r in ref),
      f'{npos} greedy passes at the default grid {eng.per_wgp_c}: finite logits, the replicas equal after every pass, 0 bails'
      + (f' (replicas differ at {rep[:8]})' if rep else '') + (f' ({nb} bails)' if nb else ''))
if A.tiny:
    from caminus.ref_q3moe import Q3MoERef
    rm, n_cmp, agree, klds = Q3MoERef(P), min(6, npos), 0, []
    for i in range(n_cmp):
        r, gl = rm.step(seq[i]), ref[i].view(np.float32).astype(np.float64)
        lr, lg_ = r - r.max(), gl - gl.max(); lr, lg_ = lr - np.log(np.exp(lr).sum()), lg_ - np.log(np.exp(lg_).sum())
        klds.append(float(np.sum(np.exp(lr) * (lr - lg_)))); agree += int(np.argmax(r)) == int(np.argmax(gl))
    check(agree == n_cmp and float(np.mean(klds)) < 1e-3,
          f'tiny: two cards against the float64 reference, first {n_cmp} passes: top-1 {agree} / {n_cmp}, mean KLD {np.mean(klds):.2e} '
          f'(bounds: every top-1, mean KLD < 1e-3 as test_q3moe_one; one card on this file: 6 / 6, 2.4e-07)')
    check(all(float(r.view(np.float32).std()) > 0.1 for r in ref), 'tiny: the logits are not degenerate (std > 0.1 at every pass)')

# ---- the grids ------------------------------------------------------------------------------------------------------------
for pw in GRIDS:
    eng.per_wgp = pw; eng._build_tapes()
    _, got, rep, nb = run(seq, False)
    bad = [i for i, (a, b) in enumerate(zip(ref, got)) if not eqb(a, b)]
    check(not bad and not rep and nb == 0 and eng.per_wgp_c == [pw, pw],
          f'K3_PER_WGP {pw} on both cards: {npos} passes\' logits the reference\'s bytes, replicas equal after every pass, 0 bails'
          + (f'; positions {bad[:8]} differ' if bad else '') + (f'; replicas differ at {rep[:8]}' if rep else '') + (f'; {nb} bails' if nb else ''))
eng.per_wgp = 'card'; eng._build_tapes()

# ---- the served pass forms -----------------------------------------------------------------------------------------------
eng.reset(); nb0 = len(eng.bails); good = True
for i in range(npos):
    if i % 3 == 0:
        t = eng.step_token(seq, i); good &= t == top(ref[i])
    elif i % 3 == 1:
        eng.launch_tokens(seq, i, 1); eng.finish_pass(i, 1); good &= eqb(keep(eng.logits_full()), ref[i])
    else:
        good &= eqb(keep(eng.verify(seq, i, 1)), ref[i])
    good &= all(eng.check_replicas())
check(good and len(eng.bails) == nb0, 'step_token / launch_tokens + finish_pass / verify in turn: the reference\'s greedy tokens and logits bytes, '
      'replicas equal, 0 bails')

# ---- spans past max_ctx -----------------------------------------------------------------------------------------------------
launched = []
orig_run = [e.run for e in eng.E]
for c, e in enumerate(eng.E): e.run = (lambda f: (lambda *a, **k: (launched.append(1), f(*a, **k))[1]))(orig_run[c])
res = []
toks = list(seq) + [0] * (max_ctx + 1 - len(seq))
for f in (lambda: (setattr(eng, 'pos_host', max_ctx), eng.step(seq[0])), lambda: eng.step_token(toks, max_ctx),
          lambda: eng.launch_tokens(toks, max_ctx, 1)):
    try: f(); res.append(False)
    except k3.ContextBoundsError: res.append(True)
for c, e in enumerate(eng.E): e.run = orig_run[c]
check(all(res) and not launched and eng.poisoned is None, f'a span at max_ctx {max_ctx} is refused on the host before any launch '
      '(step, step_token, launch_tokens); nothing poisoned')

# ---- forced bails -----------------------------------------------------------------------------------------------------------
orig_pass = eng._pass
inj = {'mode': None}


def hooked_pass(i0, commit, trace=None):
    mode, inj['mode'] = inj['mode'], None   # (the first attempt only)
    lim = {'both': [1, 1], 'card0': [1, eng.limit_us], 'card1': [eng.limit_us, 1]}.get(mode, [eng.limit_us] * 2)
    for c in (0, 1): eng.E[c].limit_us = lim[c]
    try: return orig_pass(i0, commit, trace)
    finally:
        for c in (0, 1): eng.E[c].limit_us = eng.limit_us


eng._pass = hooked_pass
MODES = ('both', 'card0', 'card1')
inject = {p: MODES[k % 3] for k, p in enumerate(range(2, npos, max(1, npos // 9)))}
eng.reset(); nb0 = len(eng.bails); same, rep, missed = [], [], []
for i in range(npos):
    inj['mode'] = inject.get(i); nb = len(eng.bails)
    lg = eng.step(seq[i]); same.append(eqb(keep(lg), ref[i]))
    if not all(eng.check_replicas()): rep.append(i)
    if i in inject and len(eng.bails) == nb: missed.append(i)
eng._pass = orig_pass
n_bails = len(eng.bails) - nb0
check(all(same) and not rep and not missed and n_bails >= len(inject),
      f'forced bails on {len(inject)} passes (both cards / card 0 / card 1 at 1 us): every injected pass bailed and its retry, and '
      f'every pass after it, gave the reference\'s bytes; replicas equal; {n_bails} bails'
      + (f'; passes {[i for i, s in enumerate(same) if not s][:8]} differ' if not all(same) else '') + (f'; no bail at {missed}' if missed else ''))
_, got, rep, nb = run(seq, False)
check(all(eqb(a, b) for a, b in zip(ref, got)) and not rep and nb == 0, 'a clean run after the injections: the reference\'s bytes, 0 bails')

print('RESULT', 'PASS' if ok else 'FAIL', flush=True)
sys.exit(0 if ok else 1)
