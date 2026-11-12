"""The two-card exchange alone (the plan's first P6 gate): what OP_XCHG costs per token on this link before any model runs.
A tape of N chained exchanges of an fp32 [D] partial on each card, on the dense qwen35 two-card module (k3_q35_swift_tp:
the same OP_XCHG as every TP engine, kernels/k3_xchg.h), launched like a decode pass (a fresh epoch per pass); pass time
per card from events, median over --passes after --warm. Also the same tape with local-only exchanges (no peer: the
op's own copy cost and the tape walk) and an empty one-segment tape (launch + wait). Checks that every exchanged sum is
card 0's partial plus card 1's, the same bits on both cards.
  tools/guard.sh LOG 600 python -I benchmarks/q35_tp_xchg.py [--n 128] [--D 5120] [--passes 200]   (HIP_VISIBLE_DEVICES=0,1)
Prints one line per tape and RESULT PASS / FAIL (the sums)."""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'python'))
import numpy as np
import torch
from caminus.device import capacity
from caminus.engine import k3
from caminus.engine.tp_slots import Slots, _hip
from caminus.engine.q3moe_one import card_per_wgp


def need_mib(n, D):
    """per card, MiB one build() allocates: the partial, the n result rows, the stage block, the n receive slots (16 header
    words + D floats each, uncached) and an upper bound of the engine's tape and sync blocks (1 MiB)"""
    n = max(n, 1)
    return (4 * D + 4 * n * D + 4 * 32 + n * (16 + D) * 4) / 2 ** 20 + 1.0


def build(devs, n, D, module, local_only):
    mib = need_mib(n, D)
    capacity.check(list(devs), [mib, mib], f'q35_tp_xchg tapes of {n} exchanges of {D} floats ({mib:.1f} MiB per card)')
    out = []
    for c in (0, 1):
        with torch.cuda.device(devs[c]):
            ds = torch.zeros(32, dtype=torch.int32, device='cuda')
            E = k3.Engine(module=module, grid=k3.resident_grid(module, card_per_wgp()), T=1, limit_us=200_000)
            E.params = ds[0:16]
            part = torch.arange(D, dtype=torch.float32, device='cuda') * (1.0 + c) + 0.25
            res = [torch.zeros(D, device='cuda') for _ in range(max(n, 1))]
            out.append(dict(E=E, ds=ds, part=part, res=res, slots=Slots(devs[c], max(n, 1), D) if n else None))
    for c in (0, 1):
        with torch.cuda.device(devs[c]):
            o = out[c]; E = o['E']; prev = -1
            for x in range(n):
                prev = E.xchg(o['part'], o['res'][x], out[1 - c]['slots'].slot(x), o['slots'].slot(x), o['ds'][18:19], x, c,
                              local_only=local_only, dep=prev)
            E.build()
    return out


def time_tape(devs, T, n, passes, warm):
    hs = torch.zeros(32, dtype=torch.int32, pin_memory=True)
    ev = [[None, None] for _ in range(2)]
    for c in (0, 1):
        with torch.cuda.device(devs[c]): ev[c] = [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]
    ms, wall = [], []
    for p in range(warm + passes):
        hs[18] = p * max(n, 1)
        t0 = time.perf_counter()
        for c in (0, 1):
            with torch.cuda.device(devs[c]):
                ev[c][0].record(); T[c]['ds'].copy_(hs, non_blocking=True); T[c]['E'].run(check=False); ev[c][1].record()
        for c in (0, 1): ev[c][1].synchronize()
        t1 = time.perf_counter()
        st = [T[c]['E'].sync[2 * k3.MAXSEG:2 * k3.MAXSEG + k3.STATUS_WORDS].cpu().tolist() for c in (0, 1)]
        if any(s[1] for s in st): raise RuntimeError(f'a wait gave up at pass {p}: {st}')
        if p >= warm:
            ms.append(tuple(ev[c][0].elapsed_time(ev[c][1]) for c in (0, 1))); wall.append((t1 - t0) * 1e3)
    a = np.array(ms)
    return np.median(a[:, 0]), np.median(a[:, 1]), float(np.median(wall))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=128); ap.add_argument('--D', type=int, default=5120)
    ap.add_argument('--passes', type=int, default=200); ap.add_argument('--warm', type=int, default=20)
    ap.add_argument('--module', default='k3_q35_swift_tp')
    A = ap.parse_args()
    assert torch.cuda.device_count() >= 2, 'two cards (HIP_VISIBLE_DEVICES=0,1)'
    devs = (0, 1)
    print(torch.cuda.get_device_name(0), '+', torch.cuda.get_device_name(1), f'module {A.module}, D {A.D}', flush=True)
    hip = _hip()
    for a, b in ((0, 1), (1, 0)):
        hip.hipSetDevice(devs[a]); hip.hipDeviceEnablePeerAccess(devs[b], 0); hip.hipGetLastError()
    ok = True
    rows = []
    for n, lo, tag in ((1, False, 'one exchange'), (A.n, True, f'{A.n} exchanges, local only'), (A.n, False, f'{A.n} exchanges')):
        T = build(devs, n, A.D, A.module, lo)
        m0, m1, w = time_tape(devs, T, n, A.passes, A.warm)
        rows.append((tag, n, m0, m1, w))
        print(f'{tag:<28} card0 {m0:7.3f} ms  card1 {m1:7.3f} ms  host wall {w:7.3f} ms  ({1e3 * max(m0, m1) / max(n, 1):6.2f} us per exchange)', flush=True)
        if not lo:
            r0, r1 = T[0]['res'][n - 1].cpu(), T[1]['res'][n - 1].cpu()
            want = T[0]['part'].cpu() + T[1]['part'].cpu()
            good = bool(torch.equal(r0, r1)) and bool(torch.equal(r0, want))
            ok = ok and good
            print(('PASS' if good else 'FAIL') + f' {tag}: the sum is card 0\'s partial + card 1\'s, the same bits on both cards', flush=True)
        del T; torch.cuda.empty_cache()
    one, loc, full = rows
    print(f'per token ({A.n} exchanges): {max(full[2], full[3]) - max(one[2], one[3]):.3f} ms over a one-exchange pass '
          f'({max(full[2], full[3]) - max(loc[2], loc[3]):.3f} ms over local-only copies)', flush=True)
    print('RESULT', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
