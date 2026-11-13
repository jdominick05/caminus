"""Dense35VerifyTP (the dense qwen35 line's T-token verify passes on TWO cards: the two-card verify modules
kernels/k3_q35_{swift,tcap}_tp_ver.hip with OP_XCHG exchanging T * D partials) against Dense35TP's decode steps on the
same tokens and positions, bit for bit, over a prefilled context of real text (this repository's docs) and from an empty
one:
  for T = 1 .. 4, a walk of passes with rejections: each pass accepts a of its T rows (the schedule cycles T, 1, T - 1,
  2, ...), the rows past a hold wrong tokens, and the next pass at the next position commits a. The accepted rows' logits
  equal the decode steps' at their positions; every third pass is followed by a decode step (committing the pass's a)
  and the pass after that commits 1, so the decode and verify tapes hand the lazily committed states to each other;
  the walk ends with a decode step equal to the reference.
  replicas   after verify passes and decode steps, the replicated residual is the same bits on both cards (the T * D
  exchanges sum card 0's partial first on both cards).
  T = 4 with every row accepted on 3, 6 and 7 workgroups per WGP: the same logits.
  tools/guard.sh LOG 1800 python -I tests/engine/test_q35_verify_tp.py GGUF [--depths 0,512]
      (HIP_VISIBLE_DEVICES=0,1: card 0 the XTX; new kernel code: run it on a non-display card alone first)"""
import argparse, glob, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'python'))
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('gguf'); ap.add_argument('--depths', default='0,512')
A = ap.parse_args()
from caminus import ggufio
from caminus.bpe import BPE
from caminus.engine import k3
from caminus.engine.q35_dense_tp import Dense35TP
from caminus.engine.q35_dense_verify import MAX_T
from caminus.engine.q35_dense_verify_tp import Dense35VerifyTP
from caminus.prefill.q35own_tp import Q35OwnPrefillTP, buffer_need_tp_mib, tp_shapes

depths = [int(x) for x in A.depths.split(',')]
G = 24                                                    # positions checked per depth
tk = BPE(ggufio.GGUFFile(A.gguf).meta)
text = ''.join(open(f).read() for f in sorted(glob.glob(os.path.join(ROOT, 'docs', '*.md'))) + [os.path.join(ROOT, 'AGENTS.md')])
ids_all = tk.encode(text)
if len(ids_all) < max(depths) + G + 1: raise SystemExit(f'the docs are {len(ids_all)} tokens: lower --depths')
print(torch.cuda.get_device_name(0), '+', torch.cuda.get_device_name(1), flush=True)
max_ctx = max(depths) + G + 8
CH = 64
GRIDS = (3, 6, 7)
per_card = sum(Dense35VerifyTP.need_mib(T, max_ctx=max_ctx) for T in range(1, MAX_T + 1)) \
    + len(GRIDS) * Dense35VerifyTP.need_mib(MAX_T, max_ctx=max_ctx)
bn = buffer_need_tp_mib(tp_shapes(A.gguf), CH)
eng = Dense35TP(A.gguf, max_ctx=max_ctx, spec_T=MAX_T, vram_extra_mib=(per_card + bn[0] + 16, per_card + bn[1] + 16))
pf = Q35OwnPrefillTP(eng, chunk=CH)
vers = {T: Dense35VerifyTP(eng, T) for T in range(1, MAX_T + 1)}
alt = {w: Dense35VerifyTP(eng, MAX_T, per_wgp=(w, w)) for w in GRIDS}
print(f'engine {eng.module} {eng.vram_mib[0]:.0f} / {eng.vram_mib[1]:.0f} MiB at max_ctx {max_ctx}, spec_T {eng.spec_T}; '
      f'verify tapes: ' + ', '.join(f'T {T}: {len(v.E[0].segs)} segments, {v.mib[0]:.1f} / {v.mib[1]:.1f} MiB' for T, v in vers.items()),
      flush=True)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


bits = lambda t: t.detach().reshape(-1).cpu().numpy().view(np.uint32).copy()


def replicas(ver=None):
    """the replicated residual the same bits on both cards: the decode tape's, or a verify tape's after its pass"""
    if ver is None: return eng.check_replicas()[0]
    return bool(torch.equal(ver.buf[0]['x'].cpu(), ver.buf[1]['x'].cpu()))


def start(N):
    """an empty engine at position N: a prefill of real text (N > 0)"""
    eng.reset()
    if N: pf.run(ids_all[:N], 0, 0, N)
    eng.pos_host = N


for N in depths:
    toks = ids_all[N:N + G + 1]
    full = ids_all[:N + G + 1]
    start(N)
    ref = []
    for i in range(G + 1): ref.append(bits(eng.verify(full, N + i, 0 if i == 0 else 1)))
    check(replicas(), f'depth {N}: the replicated residual is the same bits on both cards after the decode steps')
    for T, ver in vers.items():
        start(N)
        p, commit, npass, bad, rows, decs = N, 0, 0, [], 0, 0
        sched = [T, 1, max(1, T - 1), min(2, T)]
        while p + T <= N + G:
            a = sched[npass % len(sched)]
            wrong = [(full[p + r] + 7919 * (r + 1)) % eng.vocab for r in range(a, T)]
            ver.run(full[p:p + a] + wrong, p, commit)
            if not replicas(ver): bad.append(f'replicas differ after the pass at {p}')
            for r in range(a):
                rows += 1
                if not np.array_equal(bits(ver.logits(r)), ref[p + r - N]): bad.append(f'pass at {p} row {r}')
            p += a; commit = a; npass += 1
            if npass % 3 == 0 and p < N + G:   # a decode step between passes: it commits the pass's accepted rows
                decs += 1
                if not np.array_equal(bits(eng.verify(full, p, commit)), ref[p - N]): bad.append(f'decode step at {p} after a pass')
                if not replicas(): bad.append(f'replicas differ after the decode step at {p}')
                p += 1; commit = 1
        while p <= N + G:   # the rest as decode steps
            decs += 1
            if not np.array_equal(bits(eng.verify(full, p, commit)), ref[p - N]): bad.append(f'decode step at {p}')
            p += 1; commit = 1
        check(not bad, f'depth {N} T {T}: {npass} passes ({rows} accepted rows) and {decs} decode steps bitwise the decode '
                       f'steps on two cards, wrong tokens in every rejected row{"" if not bad else ": " + ", ".join(bad[:6])}')
    # passes of different T on one state: each commits what the last pass (of any T) accepted
    start(N)
    p, commit, bad, plan = N, 0, [], [(4, 3), (2, 2), (3, 1), (4, 4), (1, 1), (3, 2), (2, 1), (4, 2)]
    for T, a in plan:
        if p + T > N + G: break
        wrong = [(full[p + r] + 7919 * (r + 1)) % eng.vocab for r in range(a, T)]
        vers[T].run(full[p:p + a] + wrong, p, commit)
        bad += [f'T {T} at {p} row {r}' for r in range(a) if not np.array_equal(bits(vers[T].logits(r)), ref[p + r - N])]
        p += a; commit = a
    if not np.array_equal(bits(eng.verify(full, p, commit)), ref[p - N]): bad.append(f'decode step at {p}')
    check(not bad, f'depth {N}: passes of T 4, 2, 3, 4, 1, 3, 2, 4 on one state (a pass of 2 committing 3 of a pass of 4), '
                   f'then a decode step: bitwise{"" if not bad else ": " + ", ".join(bad[:6])}')
    # a commit past the last pass's tokens is refused before anything is written (its records are an older pass's)
    start(N)
    eng.verify(full, N, 0)
    try: vers[MAX_T].run(full[N + 1:N + 1 + MAX_T], N + 1, 3); refused = False
    except k3.ContextBoundsError: refused = True
    vers[MAX_T].run(full[N + 1:N + 1 + MAX_T], N + 1, 1)
    check(refused and eng.poisoned is None and all(np.array_equal(bits(vers[MAX_T].logits(r)), ref[1 + r]) for r in range(MAX_T)),
          f'depth {N}: a pass committing 3 after a decode step is refused; the engine stays usable and the right pass is bitwise')
    # the grids: T = 4, every row accepted
    for w, ver in alt.items():
        start(N)
        p, commit, bad = N, 0, []
        while p + MAX_T <= N + G:
            ver.run(full[p:p + MAX_T], p, commit)
            bad += [f'{p}+{r}' for r in range(MAX_T) if not np.array_equal(bits(ver.logits(r)), ref[p + r - N])]
            p += MAX_T; commit = MAX_T
        if not np.array_equal(bits(eng.verify(full, p, commit)), ref[p - N]): bad.append(f'decode step at {p}')
        check(not bad, f'depth {N} T {MAX_T} on {w} workgroups per WGP per card: every row and the decode step after them bitwise'
                       f'{"" if not bad else ": " + ", ".join(bad[:6])}')
check(not eng.bails, f'no wait gave up ({len(eng.bails)} bails)')
print('RESULT', 'PASS' if ok else 'FAIL', flush=True)
sys.exit(0 if ok else 1)
