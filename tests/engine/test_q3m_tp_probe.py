"""One-card probe of the qwen3moe two-card decode module (kernels/k3_q3m_lean_tp.hip: K3_TP + K3_Q3M_TP, push-only
OP_XCHG and OP_OWN_SLOT_SUM in the lean module's k3_run) before any two-card run: on ONE card
(HIP_VISIBLE_DEVICES=<the non-display card>), nothing sent to a peer.
  - Q3MoEOne runs STEPS greedy steps on k3_q3m_lean at its card's grid, then rebuilds the same tape on k3_q3m_lean_tp and
    replays the tokens at 3, 5, 6 and 7 workgroups per WGP: every step's logits are the lean module's bits, and no pass bailed
  - at each of those grids, one local-only OP_XCHG of D floats (pad1: the peer is never touched) writes its partial's
    bits, and OP_OWN_SLOT_SUM over the file's top-k slots equals OP_OWN_MOE_SUM into a zeroed x on k3_gguf_q3m_own, value
    for value (a -0 / +0 difference excepted: moe_sum adds the sum to +0)
--tiny: tests/engine/tiny_q3m.py's synthetic file (a few MiB; its first steps on the lean module are also compared with
the float64 host forward, caminus.ref_q3moe: top-1 and KLD reported, logits checked not degenerate); else the GGUF given (the catalog file: about 18,000 MiB on
the card at max_ctx 512). Prints PASS / FAIL lines and RESULT PASS | FAIL.
  HIP_VISIBLE_DEVICES=<card> tools/guard.sh LOG 300 python -I tests/engine/test_q3m_tp_probe.py --tiny --scratch DIR
  HIP_VISIBLE_DEVICES=<card> tools/guard.sh LOG 900 python -I tests/engine/test_q3m_tp_probe.py GGUF --scratch DIR   (after --tiny passed)
--kv q8: the Q8 KV builds: the reference on k3_q3m_lean_kvq8 (Q3MoEOne(q8=True)), the replays and the two ops on
k3_q3m_lean_tp_kvq8 (its first launch, on one card: the exchange, slot sum and pinned add_rms16 of its k3_run), the logits
bitwise the one-card Q8 module's, both error words 0.
A file whose matrices are Q4_0 (tools/eval/gptq_to_gguf.py) runs the Q4_0 family's twins of the same modules (k3.Q3M_Q40_OF:
k3_q3m_q40 and k3_q3m_q40_tp, with --kv q8 their Q8 builds)."""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, HERE)
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('gguf', nargs='?'); ap.add_argument('--tiny', action='store_true'); ap.add_argument('--scratch', required=True)
ap.add_argument('--steps', type=int, default=48); ap.add_argument('--max-ctx', type=int, default=None)
ap.add_argument('--kv', default='f16', choices=('f16', 'q8'), help='q8: the Q8 KV builds of both modules')
A = ap.parse_args()
if bool(A.tiny) == bool(A.gguf): raise SystemExit('test_q3m_tp_probe: give a GGUF or --tiny (one of them)')
if torch.cuda.device_count() != 1:
    raise SystemExit(f'test_q3m_tp_probe: {torch.cuda.device_count()} cards visible: run it on one card alone (HIP_VISIBLE_DEVICES=<the non-display card>)')
os.makedirs(A.scratch, exist_ok=True)
from caminus.engine import k3
from caminus.engine.q3moe_one import Q3MoEOne
from caminus.engine.tp_slots import Slots

ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


bits = lambda t: t.detach().reshape(-1).cpu().numpy().view(np.uint32).copy()
if A.tiny:
    import tiny_q3m
    P = os.path.join(A.scratch, 'tiny-q3m-probe.gguf'); tiny_q3m.write(P)
    max_ctx, first = A.max_ctx or 64, [3, 17]
else:
    P, max_ctx, first = A.gguf, A.max_ctx or 512, [785, 6722, 9625]
if A.steps + 1 > max_ctx: raise SystemExit(f'--steps {A.steps} needs max_ctx > {A.steps}')
Q8 = A.kv == 'q8'
GRIDS = (3, 5, 6, 7)
print(torch.cuda.get_device_name(0), flush=True)
# beside the engine's projection (checked by its constructor with this): one tape per grid on the two-card module, the
# exchange's receive slot (16 words + D floats, outside torch) and the two small checks' buffers
own_mib = (len(GRIDS) + 1) * Q3MoEOne.TAPE_MIB + 8
eng = Q3MoEOne(P, max_ctx=max_ctx, vram_extra_mib=own_mib, q8=Q8)
BASE = k3.Q3M_Q40_OF['k3_q3m_lean'] if eng.q40 else 'k3_q3m_lean'   # (a Q4_0 file: the Q4_0 family's twins)
LEAN = k3.Q3M_KV_Q8_OF[BASE] if Q8 else BASE
TP = k3.Q3M_TP_OF[LEAN]
D, KU, base_mod, pw0 = eng.D, eng.k_used, eng.module, eng.per_wgp
print(f'{os.path.basename(P)}: D {D}, top-{KU}, {eng.nkv} kv heads, module {base_mod} at {pw0} per WGP ({eng.vram_mib:.0f} + {own_mib} MiB)', flush=True)
check(base_mod == LEAN, f'the engine decodes on {LEAN} ({base_mod})')

# the reference: greedy steps on the lean module at the card's grid
seq, ref = list(first), []
eng.reset()
for i in range(A.steps):
    lg = eng.step(seq[i]); ref.append(bits(lg))
    if i + 1 == len(seq): seq.append(int(torch.argmax(lg)))
check(not eng.bails and all(np.isfinite(r.view(np.float32)).all() for r in ref), f'{A.steps} greedy steps on {LEAN}, finite logits, no bail')
if A.tiny:   # (the synthetic shapes' first launch: the lean reference itself against the float64 host forward, reported)
    from caminus.ref_q3moe import Q3MoERef
    rm, n_cmp, agree, klds = Q3MoERef(P), min(6, A.steps), 0, []
    for i in range(n_cmp):
        r, gl = rm.step(seq[i]), ref[i].view(np.float32).astype(np.float64)
        lr, lg_ = r - r.max(), gl - gl.max(); lr, lg_ = lr - np.log(np.exp(lr).sum()), lg_ - np.log(np.exp(lg_).sum())
        klds.append(float(np.sum(np.exp(lr) * (lr - lg_)))); agree += int(np.argmax(r)) == int(np.argmax(gl))
    print(f'  k3_q3m_lean vs the float64 reference, first {n_cmp} steps: top-1 {agree} / {n_cmp}, mean KLD {np.mean(klds):.2e} (reported only)', flush=True)
    check(all(float(r.view(np.float32).std()) > 0.1 for r in ref), 'tiny: the lean reference\'s logits are not degenerate (std > 0.1 at every step)')

slots = Slots(eng.dev, 1, D)
epoch = torch.zeros(1, dtype=torch.int32, device='cuda')
g = torch.Generator().manual_seed(11)
partial = torch.randn(D, generator=g); partial[:4] = torch.tensor([-0.0, 1e-40, -3e38, 65504.0])   # (a -0, a denormal, large values)
partial = partial.cuda()
d = torch.randn(KU * D, generator=g); d[::97] = 0.0; d[1::89] = -0.0; d[2::83] *= 1e-30
d = d.cuda()
w = torch.rand(KU, generator=g); w = (w / w.sum()).cuda()   # (a router's weights: positive, summing to about 1)
for pw in GRIDS:
    eng.module, eng.per_wgp = TP, pw
    with torch.cuda.device(eng.dev): eng._build_tape()
    eng.reset(); nb = len(eng.bails)
    got = [bits(eng.step(t)) for t in seq[:A.steps]]
    bad = [i for i, (a, b) in enumerate(zip(ref, got)) if not np.array_equal(a, b)]
    check(eng.E.module == TP and not bad and len(eng.bails) == nb and (not Q8 or int(eng.E.kv_error[0]) == 0),
          f'{TP} at {pw} per WGP: {A.steps} steps\' logits bitwise {LEAN}\'s at {pw0}' + (', error word 0' if Q8 else '') + (f': positions {bad[:8]} differ' if bad else '')
          + (f', {len(eng.bails) - nb} bails' if len(eng.bails) != nb else ''))
    # the two new ops alone on the two-card module, and the slot sum's oracle on the full module (OP_OWN_MOE_SUM)
    E2 = k3.Engine(module=TP, grid=k3.resident_grid(TP, pw))
    xout = torch.full((D,), float('nan'), device='cuda'); y = torch.full((D,), float('nan'), device='cuda')
    E2.xchg(partial, xout, slots.slot(0), slots.slot(0), epoch, 0, 0, local_only=True)
    E2.own_slot_sum(d, y, KU, w)
    E2.build().run()
    E3 = k3.Engine(module='k3_gguf_q3m_own', grid=k3.resident_grid('k3_gguf_q3m_own', pw))
    x = torch.zeros(D, device='cuda'); E3.own_moe_sum(d, x, slots=KU, w=w); E3.build().run()
    torch.cuda.synchronize()
    check(np.array_equal(bits(xout), bits(partial)), f'{TP} at {pw} per WGP: a local-only OP_XCHG of {D} floats writes its partial\'s bits')
    yb, xb = bits(y), bits(x)
    same = (yb == xb) | (((yb | xb) & 0x7fffffff) == 0)   # (equal bits, or both zeros of either sign)
    check(bool(same.all()), f'{TP} at {pw} per WGP: OP_OWN_SLOT_SUM over {KU} slots equals OP_OWN_MOE_SUM into a zeroed x for all {D} values'
          + ('' if same.all() else f' ({int((~same).sum())} differ)'))
eng.module, eng.per_wgp = base_mod, pw0
print('RESULT', 'PASS' if ok else 'FAIL', flush=True)
sys.exit(0 if ok else 1)
