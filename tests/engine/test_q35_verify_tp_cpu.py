"""The two-card verify tape's host side without a card (caminus.engine.q35_dense_verify_tp.Dense35VerifyTP on Dense35TP with
spec_T): the tiny synthetic models with two-card shapes (tests/engine/tiny_q35.py tp=True), on stand-ins for the torch.cuda
and HIP calls (every tensor in host memory; nothing is launched):
  - the module: k3.Q35_TPV_OF maps each two-card module to its verify build (k3_q35_{swift,tcap}_tp_ver); Dense35VerifyTP
    takes it by default and refuses any other; T outside 1 .. 4, T above the engine's spec_T, and an engine without spec_T
    are refused
  - the tapes: every op one the module dispatches (k3.Q35S_OPS + OP_XCHG), 2 exchanges per block on each card with indices
    0 .. 2 nl - 1 once each, the card's own number and N = T * D (the payload's floats), every segment's team = T, the guard's
    T = the engine's spec_T, every dependency pointing back
  - the buffers: T times the decode tape's per-token sizes, the head's shard per card, the split attention records over
    max_ctx + spec_T - 1 rows, the exchange slots spec_T * D floats; need_mib agrees with the buffers
  - spec_T = 1 keeps the served projection: the engine's bytes, KV rows, records, slots and recurrent MiB are the spec_T
    default's; commit_max is 1 on it, and on a spec_T engine 0 after a reset
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/engine/test_q35_verify_tp_cpu.py --scratch DIR"""
import argparse, contextlib, os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, HERE)
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); A = ap.parse_args()
os.makedirs(A.scratch, exist_ok=True)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


# ---- stand-ins: the engine's torch.cuda / HIP calls on host memory (no device is opened) ------------------------------
_zeros, _to = torch.zeros, torch.Tensor.to
torch.zeros = lambda *a, **k: _zeros(*a, **{x: v for x, v in k.items() if x not in ('device', 'pin_memory')})
torch.Tensor.to = lambda self, *a, **k: self if a and str(a[0]).startswith('cuda') else _to(self, *a, **k)
torch.Tensor.cuda = lambda self, *a, **k: self
torch.cuda.set_device = lambda d: None
torch.cuda.device = lambda d: contextlib.nullcontext()
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda d: types.SimpleNamespace(multi_processor_count=42, total_memory=20464 << 20)
torch.cuda.Event = lambda *a, **k: types.SimpleNamespace()

import tiny_q35
from caminus import ggufio
from caminus.engine import k3, q35_dense_tp
from caminus.engine.q35_dense_tp import Dense35TP
from caminus.engine.q35_dense_verify import MAX_T
from caminus.engine.q35_dense_verify_tp import Dense35VerifyTP
from caminus.engine.q35_dense import attn_part_elems


class FakeHip:
    def __getattr__(self, n): return lambda *a, **k: 0


class FakeSlots:
    """the receive slots' addresses only (distinct per card and slot)"""
    n_made = 0

    def __init__(self, dev, n, N):
        FakeSlots.n_made += 1; self.base = (FakeSlots.n_made << 40); self.stride = (16 + N) * 4; self.n, self.N = n, N

    def slot(self, i):
        assert 0 <= i < self.n; return self.base + i * self.stride


q35_dense_tp._hip = lambda: FakeHip()
q35_dense_tp.Slots = FakeSlots

P_S = os.path.join(A.scratch, 'tiny-tp-swift.gguf'); iS = tiny_q35.write(P_S, tp=True)
P_T = os.path.join(A.scratch, 'tiny-tp-tcap.gguf'); iT = tiny_q35.write(P_T, variant='tcap', tp=True)

# ---- the module mapping and the refusals --------------------------------------------------------------------------------
good = k3.Q35_TPV_OF == {'k3_q35_swift_tp': 'k3_q35_swift_tp_ver', 'k3_q35_tcap_tp': 'k3_q35_tcap_tp_ver'}
good &= all(m in k3.Q35V_MODULES and m in k3.ATTN_PACK_MODULES and m in k3.Q35_SPLIT_MODULES for m in k3.Q35TPV_MODULES)
good &= all(k3.MODULE_CAPS[m] for m in k3.Q35TPV_MODULES)
check(good, 'k3.Q35_TPV_OF maps each two-card module to its verify build; the verify builds are T-token, packed-split modules with caps')
eng = Dense35TP(P_S, devices=(0, 1), max_ctx=64, vram_check=False, spec_T=MAX_T)
try: Dense35VerifyTP(eng, MAX_T, module='k3_q35_swift_ver'); refused = False
except ValueError: refused = True
try: Dense35VerifyTP(eng, 0); refused0 = False
except ValueError: refused0 = True
try: Dense35VerifyTP(eng, MAX_T + 1); refusedT = False
except ValueError: refusedT = True
eng2 = Dense35TP(P_S, devices=(0, 1), max_ctx=64, vram_check=False, spec_T=2)
try: Dense35VerifyTP(eng2, 3); refusedS = False
except ValueError: refusedS = True
class Bare: pass
try: Dense35VerifyTP(Bare(), 1); refusedE = False
except ValueError: refusedE = True
check(refused and refused0 and refusedT and refusedS and refusedE,
      'a one-card verify module, T outside 1 .. 4, T above spec_T and an engine without spec_T are all refused')

# ---- the tapes ----------------------------------------------------------------------------------------------------------
for path, info, want_mod in ((P_S, iS, 'k3_q35_swift_tp_ver'), (P_T, iT, 'k3_q35_tcap_tp_ver')):
    m = Dense35TP(path, devices=(0, 1), max_ctx=64, vram_check=False, spec_T=MAX_T)
    nl = info['blocks']; D = m.D
    check(m.spec_T == MAX_T and m.commit_max() == 0, f'{os.path.basename(path)}: spec_T {m.spec_T}, commit_max 0 after a reset')
    check(all(m.slots[c].N == MAX_T * D and m.slots[c].stride == (16 + MAX_T * D) * 4 for c in (0, 1)),
          f'{os.path.basename(path)}: the exchange slots hold spec_T * D = {MAX_T * D} floats')
    for T in range(1, MAX_T + 1):
        ver = Dense35VerifyTP(m, T, vram_check=False)
        good = ver.module == want_mod and ver.T == T
        xs_all = []
        for c in (0, 1):
            E = ver.E[c]; ops = [sg.op for sg in E.segs]
            good &= E.T == T and E.guard_T == MAX_T and all(o in k3.Q35S_OPS or o == k3.OP_XCHG for o in ops)
            good &= not E.unsupported() and E.refuse is None
            good &= all(sg.team == (T if T > 1 else -1) for sg in E.segs)   # (build(): T > 1 carries T in every segment's team)
            good &= all(sg.dep < i and sg.dep2p1 - 1 < i for i, sg in enumerate(E.segs))
            xs = [sg for sg in E.segs if sg.op == k3.OP_XCHG]
            good &= all(sg.work == c and sg.N == T * D for sg in xs) and len(xs) == 2 * nl
            xs_all.append(sorted(sg.K for sg in xs))
            s = ver.buf[c]
            fc = m.f0 if c == 0 else m.ffn - m.f0
            nlog = m.vh if c == 0 else m.vocab - m.vh
            good &= s['x'].numel() == T * D and s['blkp'].numel() == T * D and s['ffpp'].numel() == T * D
            good &= s['g'].numel() == T * fc and s['logits'].numel() == T * nlog
            good &= s['part'].numel() == attn_part_elems(T, m.nh // 2, m.max_ctx + MAX_T - 1)
            good &= ver.logits2[c].shape == (T, nlog) and ver.dstage[c].numel() == 32
        check(good and xs_all[0] == xs_all[1] == list(range(2 * nl)),
              f'{os.path.basename(path)} T {T}: both tapes run only the module {want_mod}\'s ops, every segment\'s team is T, '
              f'2 exchanges per block (indices 0 .. {2 * nl - 1} once, the card\'s own number, N = T * D = {T * D}), deps point back')
        mib = Dense35VerifyTP.need_mib(T, m)
        sz = {torch.float32: 4, torch.float16: 2, torch.int32: 4}
        exact = [sum(n * sz[dt] for n, dt in sp.values()) / 2 ** 20 + 2.0 for sp in Dense35VerifyTP._spec(m, T)]
        check(all(abs(a - b) < 1e-9 for a, b in zip(mib, exact)) and ver.mib == mib,
              f'{os.path.basename(path)} T {T}: need_mib per card agrees with the tape\'s buffers ({mib[0]:.2f} / {mib[1]:.2f} MiB)')
    check(Dense35VerifyTP.need_mib(MAX_T, max_ctx=64) > 0, f'{os.path.basename(path)}: need_mib\'s upper bound without an engine')

# ---- spec_T = 1 keeps the served projection -------------------------------------------------------------------------------
p_def, p_1 = Dense35TP.projection_mib(P_S, 64), Dense35TP.projection_mib(P_S, 64, spec_T=1)
p_4 = Dense35TP.projection_mib(P_S, 64, spec_T=MAX_T)
check(p_def == p_1 and p_4[0] > p_1[0] and Dense35TP.recurrent_mib(P_S) == Dense35TP.recurrent_mib(P_S, 1)
      and Dense35TP.recurrent_mib(P_S, MAX_T) > Dense35TP.recurrent_mib(P_S, 1),
      'spec_T defaults to 1: the served projection and recurrent MiB are unchanged; spec_T 4 grows them (saved inputs, KV rows, slots, records)')
m1 = Dense35TP(P_S, devices=(0, 1), max_ctx=64, vram_check=False)
check(m1.spec_T == 1 and m1.commit_max() == 1 and m1.slots[0].N == m1.D and m1.E[0].guard_T is None,
      'a spec_T 1 engine: commit_max 1, D-float slots, no verify guard_T on the decode tape (as served)')

print('RESULT', 'PASS' if ok else 'FAIL', flush=True)
sys.exit(0 if ok else 1)
