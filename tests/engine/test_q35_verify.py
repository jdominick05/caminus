"""Dense35Verify (the dense qwen35 line, Swift 1.5: T-token verify passes on kernels/k3_q35_swift_ver.hip) against
Dense35Own's decode steps on the same tokens and positions, bit for bit, over a prefilled context of real text (this
repository's docs) and from an empty one:
  for T = 1 .. 4, a walk of passes with rejections: each pass accepts a of its T rows (the schedule cycles T, 1, T - 1,
  2, ...), the rows past a hold wrong tokens, and the next pass at the next position commits a. The accepted rows' logits
  equal the decode steps' at their positions; every third pass is followed by a decode step (committing the pass's a)
  and the pass after that commits 1, so the decode and verify tapes hand the lazily committed states to each other;
  the walk ends with a decode step equal to the reference.
  T = 4 with every row accepted on 3, 6 and 7 workgroups per WGP: the same logits.
The verify tape's feed-forward runs as three segments and the decode tape's as the fused gate | up segment: equal logits
are also that check. Prints PASS / FAIL lines and RESULT PASS | FAIL.
  tools/guard.sh LOG 1200 python -I tests/engine/test_q35_verify.py GGUF [--depths 0,512]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, glob, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'python'))
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('gguf'); ap.add_argument('--depths', default='0,512')
A = ap.parse_args()
from caminus import ggufio
from caminus.bpe import BPE
from caminus.device import capacity
from caminus.engine.q35_dense import Dense35Own
from caminus.engine.q35_dense_verify import Dense35Verify, MAX_T
from caminus.prefill.q35own import Q35OwnPrefill, buffer_need_mib, shapes_of

depths = [int(x) for x in A.depths.split(',')]
G = 24                                                    # positions checked per depth
tk = BPE(ggufio.GGUFFile(A.gguf).meta)
text = ''.join(open(f).read() for f in sorted(glob.glob(os.path.join(ROOT, 'docs', '*.md'))) + [os.path.join(ROOT, 'AGENTS.md')])
ids_all = tk.encode(text)
if len(ids_all) < max(depths) + G + 1: raise SystemExit(f'the docs are {len(ids_all)} tokens: lower --depths')
print(torch.cuda.get_device_name(0), flush=True)
max_ctx = max(depths) + G + 8
CH = 64
GRIDS = (3, 6, 7)
extra = sum(Dense35Verify.need_mib(T) for T in range(1, MAX_T + 1)) + len(GRIDS) * Dense35Verify.need_mib(MAX_T)
eng = Dense35Own(A.gguf, max_ctx=max_ctx, spec_T=MAX_T, vram_extra_mib=extra + buffer_need_mib(shapes_of(A.gguf), CH) + 16)
pf = Q35OwnPrefill(eng, chunk=CH, crossover=0)
vers = {T: Dense35Verify(eng, T) for T in range(1, MAX_T + 1)}
alt = {w: Dense35Verify(eng, MAX_T, per_wgp=w) for w in GRIDS}
print(f'engine {eng.module} {eng.vram_mib:.0f} MiB at max_ctx {max_ctx}, {eng.per_wgp} workgroups per WGP; verify tapes: ' +
      ', '.join(f'T {T}: {len(v.E.segs)} segments, {v.mib:.1f} MiB' for T, v in vers.items()), flush=True)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


bits = lambda t: t.detach().reshape(-1).cpu().numpy().view(np.uint32).copy()


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
    for T, ver in vers.items():
        start(N)
        p, commit, npass, bad, rows, decs = N, 0, 0, [], 0, 0
        sched = [T, 1, max(1, T - 1), min(2, T)]
        while p + T <= N + G:
            a = sched[npass % len(sched)]
            wrong = [(full[p + r] + 7919 * (r + 1)) % eng.vocab for r in range(a, T)]
            ver.run(full[p:p + a] + wrong, p, commit)
            for r in range(a):
                rows += 1
                if not np.array_equal(bits(ver.logits(r)), ref[p + r - N]): bad.append(f'pass at {p} row {r}')
            p += a; commit = a; npass += 1
            if npass % 3 == 0 and p < N + G:   # a decode step between passes: it commits the pass's accepted rows
                decs += 1
                if not np.array_equal(bits(eng.verify(full, p, commit)), ref[p - N]): bad.append(f'decode step at {p} after a pass')
                p += 1; commit = 1
        while p <= N + G:   # the rest as decode steps
            decs += 1
            if not np.array_equal(bits(eng.verify(full, p, commit)), ref[p - N]): bad.append(f'decode step at {p}')
            p += 1; commit = 1
        check(not bad, f'depth {N} T {T}: {npass} passes ({rows} accepted rows) and {decs} decode steps bitwise the decode '
                       f'steps, wrong tokens in every rejected row{"" if not bad else ": " + ", ".join(bad[:6])}')
    # the grids: T = 4, every row accepted
    for w, ver in alt.items():
        start(N)
        p, commit, bad = N, 0, []
        while p + MAX_T <= N + G:
            ver.run(full[p:p + MAX_T], p, commit)
            bad += [f'{p}+{r}' for r in range(MAX_T) if not np.array_equal(bits(ver.logits(r)), ref[p + r - N])]
            p += MAX_T; commit = MAX_T
        if not np.array_equal(bits(eng.verify(full, p, commit)), ref[p - N]): bad.append(f'decode step at {p}')
        check(not bad, f'depth {N} T {MAX_T} on {w} workgroups per WGP: every row and the decode step after them bitwise'
                       f'{"" if not bad else ": " + ", ".join(bad[:6])}')
check(not eng.bails, f'no wait gave up ({len(eng.bails)} bails)')
print('RESULT', 'PASS' if ok else 'FAIL', flush=True)
sys.exit(0 if ok else 1)
