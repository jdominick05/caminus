"""One-card smoke of the two-card verify modules (kernels/k3_q35_{swift,tcap}_tp_ver.hip: K3_TP + K3_Q3M_VER in one code
object) before any two-card run: on ONE card (HIP_VISIBLE_DEVICES=<the non-display card>), Dense35Verify with its engine's
module swapped to the two-card verify build, plus one OP_XCHG segment in local_only mode with a T * D payload, over the tiny
synthetic models. The passes' logits are the one-card decode steps' bit for bit (the T-token forms in the combined code
object), and the exchange's out is its partial's bits (pad1: local only, the peer is never touched). Prints PASS / FAIL
lines and RESULT PASS | FAIL.
  tools/guard.sh LOG 600 python -I tests/engine/test_q35_tp_ver_probe.py --scratch DIR"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, HERE)
import numpy as np, torch

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); A = ap.parse_args()
os.makedirs(A.scratch, exist_ok=True)
from caminus.engine import k3
from caminus.engine.q35_dense import Dense35Own
from caminus.engine.q35_dense_verify import Dense35Verify, MAX_T
from caminus.engine.tp_slots import Slots
import tiny_q35

ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


bits = lambda t: t.detach().reshape(-1).cpu().numpy().view(np.uint32).copy()
G = 12

for variant in ('swift', 'tcap'):
    P = os.path.join(A.scratch, f'tiny-probe-{variant}.gguf'); tiny_q35.write(P, variant=variant)
    eng = Dense35Own(P, max_ctx=64, spec_T=MAX_T)
    ids = [(i * 7 + 3) % eng.vocab for i in range(G + MAX_T + 1)]
    ref = [bits(eng.verify(ids, i, 0 if i == 0 else 1)) for i in range(G + 1)]
    want_mod = k3.Q35_TPV_OF[k3.Q35_TP_OF[eng.module]]   # the engine's line: one-card module -> two-card -> its verify build
    for T in range(1, MAX_T + 1):
        eng.reset(); eng.pos_host = 0
        ver = Dense35Verify(eng, T)
        ver.E.module = want_mod            # the two-card verify build of the engine's line (one-card tape, one card)
        ver.E.caps = k3.MODULE_CAPS[want_mod]
        slots = Slots(eng.dev, 1, T * eng.D)
        epoch = torch.zeros(1, dtype=torch.int32, device='cuda')
        partial = torch.randn(T * eng.D, device='cuda')
        xout = torch.zeros(T * eng.D, device='cuda')
        ver.E.xchg(partial, xout, slots.slot(0), slots.slot(0), epoch, 0, 0, local_only=True, dep=len(ver.E.segs) - 1)
        ver.E.build()
        p, commit, bad = 0, 0, []
        while p + T <= G:
            ver.run(ids[p:p + T], p, commit)
            bad += [f'{p}+{r}' for r in range(T) if not np.array_equal(bits(ver.logits(r)), ref[p + r])]
            p += T; commit = T
        if p <= G and not np.array_equal(bits(eng.verify(ids, p, commit)), ref[p]): bad.append(f'decode step at {p}')
        xeq = bool(torch.equal(xout.cpu(), partial.cpu()))
        check(not bad and xeq and not ver.bails,
              f'{variant} T {T} on module {want_mod}: {G // T + 1} passes and a decode step bitwise the decode, the local-only '
              f'exchange of T * D = {T * eng.D} floats copies its partial{"" if not bad and xeq else ": " + ", ".join(bad[:6])}')
    del eng
print('RESULT', 'PASS' if ok else 'FAIL', flush=True)
sys.exit(0 if ok else 1)
