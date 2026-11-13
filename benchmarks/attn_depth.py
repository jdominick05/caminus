"""Decode / verify attention at depth on the dense qwen35 geometry (Swift 1.5, ThinkingCap, AREX-2: 24 query heads over 4
kv heads, head dim 256), one attention layer alone: a tape of just the attention op(s) on a zeroed KV cache of P
positions (the timing does not depend on the values), T = 1 (decode) and T = 4 (a verify pass).

Arms (the same geometry, our own kernels only):
  own        the Swift modules' attention (k3_q35_swift; k3_q35_swift_ver for T = 4): the split pair since they run it
             (kernels/k3_q35_ops.h K3_Q35_ATTN_SPLIT); before that OP_ATTN_DEC, one workgroup per query head walking
             every position (kernels/k3_own_ops.h own_op_attn_dec, batched loads); f16 cache.
  part       the same modules' OP_ATTN_PART alone (the pair's COMB segment dropped from the tape; its records are
             written, not merged): own - part = what the COMB costs
  split      the split pair OP_ATTN_PART + OP_ATTN_COMB (kernels/k3_attn_split.h, flash decoding: ATTN_SS-position
             slices, one task per (slice, row, kv head) reads its K/V rows once for the 6 query heads, then a merge per
             query head) in the AREX-2 modules (k3_gguf_q35d; k3_gguf_q35d_mt for T = 4), f16 and q8_0 caches.
Every arm runs on the same grid (--per-wgp; default the dense engine's own, ~200 workgroups). Each row prints the launch median over REPS runs (host events around E.run: one persistent launch, so it includes the
launch; the 'empty' row is a tape of one no-op segment, the launch floor to subtract), the KV bytes one layer must read
(T rows reading the cache once) and the time those bytes take at --bw GB/s (default 952: the XTX's measured read peak,
docs/dev/benchmarks.md), and x floor. x16 = the 16 attention layers of a token.
  tools/guard.sh LOG SECS python -I benchmarks/attn_depth.py [--ctx 1024,4096,16384,32768,49152] [--reps 30] [--bw 952]
      [--arms own,split,split_q8]
Run it on the XT alone first."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np, torch
from caminus.engine import k3
from caminus.device import capacity
from caminus.engine.q3moe_one import card_per_wgp

ap = argparse.ArgumentParser()
ap.add_argument('--ctx', default='1024,4096,16384,32768,49152'); ap.add_argument('--reps', type=int, default=30)
ap.add_argument('--bw', type=float, default=952.0); ap.add_argument('--arms', default='own,split,split_q8')
ap.add_argument('--T', default='1,4')
ap.add_argument('--per-wgp', default='card', help="workgroups per WGP for every arm: 'card' = the dense engine's own grid (q3moe_one.card_per_wgp, ~200 workgroups), or a number")
A = ap.parse_args()
CTX = [int(v) for v in A.ctx.split(',')]; ARMS = A.arms.split(','); TS = [int(v) for v in A.T.split(',')]
H, NKV, D = 24, 4, 256
SCALE = 1.0 / 16.0
MOD = {('own', 1): 'k3_q35_swift', ('own', 4): 'k3_q35_swift_ver', ('part', 1): 'k3_q35_swift', ('part', 4): 'k3_q35_swift_ver',
       ('split', 1): 'k3_gguf_q35d', ('split', 4): 'k3_gguf_q35d_mt'}
PW = card_per_wgp() if A.per_wgp == 'card' else int(A.per_wgp)
print(torch.cuda.get_device_name(0), f'| grid {PW} per WGP ({k3.resident_grid("k3_q35_swift", PW)} workgroups) for every arm | geometry {H} q heads / {NKV} kv heads x {D} | floor at {A.bw:.0f} GB/s', flush=True)


def timed(E, reps):
    for _ in range(3): E.run(check=True)
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); E.run(check=False); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 10)), float(np.percentile(ts, 90))


def empty_launch(mod, T):
    capacity.check([0], [64], f'attn_depth empty tape on {mod} (the engine\'s sync words, params and tape: under 64 MiB)')
    E = k3.Engine(module=mod, grid=k3.resident_grid(mod, PW), T=T, limit_us=200_000)
    E.nop(1, 0)
    E.build()
    return timed(E, A.reps)[0]


for T in TS:
    for arm in ARMS:
        base, q8 = ('split', True) if arm == 'split_q8' else (arm, False)
        mod = MOD[(base, T)]
        try:
            e_us = empty_launch(mod, T)
        except Exception as ex:   # (the empty tape is only a reference row)
            e_us = float('nan'); print(f'  {mod}: empty tape not built ({type(ex).__name__}: {ex})', flush=True)
        print(f'T {T} {arm:9s} module {mod}: empty-tape launch {e_us:.1f} us', flush=True)
        for P in CTX:
            M = NKV * D
            ns = (P + k3.ATTN_SS - 1) // k3.ATTN_SS
            need = (P * k3.kv_pos_bytes(M, q8) + 2 * T * H * D * 4 + T * H * ns * k3.ATTN_RS * 4) / 2 ** 20 + 64
            capacity.check([0], [need], f'attn_depth {arm} T {T} ctx {P} ({need:.0f} MiB)')
            cache = torch.zeros(P * k3.kv_pos_bytes(M, q8), dtype=torch.uint8, device='cuda')
            if not q8: cache = cache.view(torch.float16)
            q = torch.randn(T * H * D, device='cuda'); out = torch.zeros(T * H * D, device='cuda')
            pos = torch.full((1,), P - T, dtype=torch.int32, device='cuda')
            E = k3.Engine(module=mod, grid=k3.resident_grid(mod, PW), T=T, limit_us=500_000)
            try:
                if base == 'own':
                    E.attn_dec(q, cache, pos, out, H, NKV, D, SCALE, q8=False, batched=True)
                elif base == 'part':
                    E.attn_dec(q, cache, pos, out, H, NKV, D, SCALE, q8=False, max_cells=P)
                    assert E.segs[-1].op == k3.OP_ATTN_COMB and E.segs[-2].op == k3.OP_ATTN_PART
                    E.segs.pop()                       # (the COMB: the PART's records are the tape's last output)
                else:
                    E.attn_dec(q, cache, pos, out, H, NKV, D, SCALE, q8=q8, max_cells=P)
                E.build()
                med, p10, p90 = timed(E, A.reps)
            except Exception as ex:
                print(f'  ctx {P:6d}: NOT RUN ({type(ex).__name__}: {ex})', flush=True)
                del cache, E; torch.cuda.empty_cache(); continue
            kvb = P * k3.kv_pos_bytes(M, q8)           # K and V of every position, read once for the T rows
            floor_us = kvb / (A.bw * 1e3)
            print(f'  ctx {P:6d}: {med:9.1f} us (p10 {p10:.1f}, p90 {p90:.1f}) | KV {kvb / 2**20:7.1f} MiB, floor {floor_us:7.1f} us, '
                  f'x{med / floor_us:6.1f} | x16 layers {16 * med / 1e3:7.2f} ms', flush=True)
            del cache, E; torch.cuda.empty_cache()
print('RESULT PASS (ran to the end; a measurement, not a gate)')
