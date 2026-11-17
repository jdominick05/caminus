"""Qwen3-30B-A3B batched prefill speed (caminus.prefill.q3m.Q3MPrefill): positions [0, L) of random tokens per length,
medians over --rounds, prompt tok/s = L / seconds (prefill only: the last token's K3 pass is the decode's). With --ops,
one more run per length with every launch timed by HIP events (the time per op kind: GEMM by matrix, attention, glue).
--cards 2: the two-card prefill (caminus.prefill.q3m_tp.Q3MPrefillTP on Q3MoETP, torch devices 0 and 1, card 0 first;
CAMINUS_UPLOAD_LOG on every two-card run): --micro pieces a chunk (default its MICRO), --min-piece, --mode overlap | serial |
local (local: no exchange, each card's own partials: the compute floor, not the model's numbers). Each arm in --modes is run
per length in turn (interleaved), so the exchange's exposed cost is overlap or serial minus local in the same process.
  tools/guard.sh LOG 1800 python -I benchmarks/q3moe_prefill.py GGUF [--lengths 512,4096] [--chunk 1024] [--max-ctx N] [--ops] [--prefill-module M]
  CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 1800 python -I benchmarks/q3moe_prefill.py GGUF --cards 2 [--modes overlap,serial,local] [--micro 2]
--kv q8: the engine and prefill with the KV cache as Q8_0 rows (Q3MoEOne(q8=True); with --cards 2 Q3MoETP(q8=True) and
Q3MPrefillTP's Q8 path); runs alternated with an F16 run for the A/B (both caches do not fit at depth)."""
import argparse, collections, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus.engine.q3moe_one import Q3MoEOne
from caminus.prefill.q3m import Q3MPrefill
ap = argparse.ArgumentParser()
ap.add_argument('gguf'); ap.add_argument('--lengths', default='512,4096'); ap.add_argument('--chunk', type=int, default=1024)
ap.add_argument('--max-ctx', type=int, default=None); ap.add_argument('--rounds', type=int, default=3); ap.add_argument('--ops', action='store_true')
ap.add_argument('--prefill-module', default=None, help='the prefill code object (default: by the file\'s formats; q3m_prefill_iq times a Q4_K_M file on the IQ study\'s object)')
ap.add_argument('--cards', type=int, default=1, choices=(1, 2)); ap.add_argument('--modes', default='overlap')
ap.add_argument('--micro', type=int, default=None); ap.add_argument('--min-piece', type=int, default=None)
ap.add_argument('--kv', default='f16', choices=('f16', 'q8'))
A = ap.parse_args()
Ls = [int(x) for x in A.lengths.split(',')]
if A.cards == 2:
    from caminus.engine.q3moe_tp import Q3MoETP
    from caminus.prefill import q3m_tp
    if A.ops or A.prefill_module: raise SystemExit('--cards 2: no --ops / --prefill-module (the one-card prefill\'s)')
    modes = A.modes.split(',')
    print(torch.cuda.get_device_name(0), '+', torch.cuda.get_device_name(1), flush=True)
    eng = Q3MoETP(A.gguf, max_ctx=A.max_ctx or max(Ls) + 8, vram_extra_mib=q3m_tp.need_tp_mib(q3m_tp.tp_shapes(A.gguf), A.chunk),
                  q8=A.kv == 'q8')
    pf = q3m_tp.Q3MPrefillTP(eng, chunk=A.chunk, micro=A.micro or q3m_tp.MICRO, min_piece=A.min_piece or q3m_tp.MIN_PIECE)
    print(f'two cards: max_ctx {eng.max_ctx}, KV {A.kv}, decode module {eng.module}, chunk {A.chunk}, micro {pf.micro} (pieces of a full chunk '
          f'{q3m_tp.micro_bounds(A.chunk, pf.micro, min_piece=pf.min_piece)}), expert rows {eng.exp_rows}, kv heads {eng.kvc}: '
          f'engine {eng.vram_mib[0]:.0f} / {eng.vram_mib[1]:.0f} MiB + prefill {pf.need_mib[0]:.0f} / {pf.need_mib[1]:.0f} MiB', flush=True)
    g = np.random.default_rng(0)
    sync = lambda: [torch.cuda.synchronize(d) for d in eng.devs]
    for L in Ls:
        toks = g.integers(0, 151000, L).tolist()
        ts = {md: [] for md in modes}
        for r in range(A.rounds + 1):
            for md in modes:
                pf.mode = md; eng.reset(); eng.poisoned = None; sync(); t0 = time.perf_counter()
                pf.run(toks, 0, 0, L); sync(); ts[md].append(time.perf_counter() - t0)
        eng.poisoned = None
        for md in modes:
            s_ = float(np.median(ts[md][1:]))
            print(f'  L {L:6d} {md:7s}: {s_ * 1e3:9.1f} ms = {L / s_:7.0f} tok/s prefill, two cards (rounds {min(ts[md][1:]) * 1e3:.1f} .. '
                  f'{max(ts[md][1:]) * 1e3:.1f})' + (' [no exchange: not the model\'s numbers]' if md == 'local' else ''), flush=True)
    print('RESULT DONE')
    sys.exit(0)
print(torch.cuda.get_device_name(0), flush=True)
eng = Q3MoEOne(A.gguf, max_ctx=A.max_ctx or max(Ls) + 8, q8=A.kv == 'q8')
pf = Q3MPrefill(eng, chunk=A.chunk, module=A.prefill_module)
print(f'max_ctx {eng.max_ctx}, chunk {A.chunk}, decode module {eng.module}, prefill object {pf.ops.module}: engine {eng.vram_mib:.0f} MiB + prefill buffers {pf.need_mib:.0f} MiB', flush=True)
g = np.random.default_rng(0)
for L in Ls:
    toks = g.integers(0, 151000, L).tolist()
    ts = []
    for r in range(A.rounds + 1):
        eng.reset(); torch.cuda.synchronize(); t0 = time.perf_counter()
        pf.run(toks, 0, 0, L); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    s = float(np.median(ts[1:]))
    print(f'  L {L:6d}: {s * 1e3:9.1f} ms = {L / s:7.0f} tok/s prefill (rounds {min(ts[1:]) * 1e3:.1f} .. {max(ts[1:]) * 1e3:.1f})', flush=True)
    if A.ops:
        acc = collections.defaultdict(float); evs = []
        O = pf.ops
        for name in ('mm', 'emb', 'add_rms16', 'router', 'topk', 'group', 'swiglu', 'qk_rope', 'attn'):
            f = getattr(O, name)
            def wrap(*a, _f=f, _n=name, **kw):
                key = _n
                if _n == 'mm':
                    W = a[0]; key = f'mm {W[0]} {W[1]}x{W[2]}' + (' experts' if 'work' in kw else '')
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); _f(*a, **kw); e1.record(); evs.append((key, e0, e1))
            setattr(O, name, wrap)
        eng.reset(); pf.run(toks, 0, 0, L); torch.cuda.synchronize()
        for k, e0, e1 in evs: acc[k] += e0.elapsed_time(e1)
        for name in ('mm', 'emb', 'add_rms16', 'router', 'topk', 'group', 'swiglu', 'qk_rope', 'attn'):
            delattr(O, name)
        tot = sum(acc.values())
        for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
            print(f'      {k:34s} {v:9.2f} ms  {100 * v / tot:5.1f} %', flush=True)
print('RESULT DONE')
