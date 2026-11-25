"""Qwen3-30B-A3B batched prefill speed (caminus.prefill.q3m.Q3MPrefill; a qwen35moe file, Qwen3.6-35B-A3B: caminus.prefill.
q35moe.Q35MoEPrefill on Moe35Own, one card, without --ops / --prefill-module): positions [0, L) of random tokens per length,
medians over --rounds, prompt tok/s = L / seconds (prefill only: the last token's K3 pass is the decode's). With --ops,
one more run per length with every launch timed by HIP events (the time per op kind: GEMM by matrix, attention, glue).
--cards 2: the two-card prefill (caminus.prefill.q3m_tp.Q3MPrefillTP on Q3MoETP, torch devices 0 and 1, card 0 first;
CAMINUS_UPLOAD_LOG on every two-card run): --micro pieces a chunk (default its MICRO), --min-piece, --mode overlap | serial |
local (local: no exchange, each card's own partials: the compute floor, not the model's numbers). Each arm in --modes is run
per length in turn (interleaved), so the exchange's exposed cost is overlap or serial minus local in the same process.
  tools/guard.sh LOG 1800 python -I benchmarks/q3moe_prefill.py GGUF [--lengths 512,4096] [--chunk 1024] [--max-ctx N] [--ops] [--prefill-module M]
  CAMINUS_UPLOAD_LOG=ULOG tools/guard.sh LOG 1800 python -I benchmarks/q3moe_prefill.py GGUF --cards 2 [--modes overlap,serial,local] [--micro 2]
--ab (one card): every round also on the narrow tiles (Ops.wide / gu / q8l False), interleaved, both arms' medians printed and
the last attention layer's KV rows of the last round compared bitwise (RESULT FAIL if they differ).
--kv q8: the engine and prefill with the KV cache as Q8_0 rows (Q3MoEOne(q8=True); with --cards 2 Q3MoETP(q8=True) and
Q3MPrefillTP's Q8 path); runs alternated with an F16 run for the A/B (both caches do not fit at depth).
--yarn F: the long-context mode at YaRN factor F (the 30B: rope=caminus.rope.YarnMode(F, its native context 32,768, as
caminus serve --long-context builds it): the decode module's YaRN build and pf_q3m's YaRN rope, one card or two; a qwen35moe
file: Moe35Own(yarn=F), the prefill's YaRN rope); lengths up to F x the native window. The header line names the rope."""
import argparse, collections, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus.engine.q3moe_one import Q3MoEOne
from caminus.prefill.q3m import Q3MPrefill
ap = argparse.ArgumentParser()
ap.add_argument('gguf'); ap.add_argument('--lengths', default='512,4096'); ap.add_argument('--chunk', type=int, default=1024)
ap.add_argument('--max-ctx', type=int, default=None); ap.add_argument('--rounds', type=int, default=3); ap.add_argument('--ops', action='store_true')
ap.add_argument('--prefill-module', default=None, help='the prefill code object (default: by the file\'s formats; pf_q3m_iq times a Q4_K_M file on the IQ study\'s object)')
ap.add_argument('--cards', type=int, default=1, choices=(1, 2)); ap.add_argument('--modes', default='overlap')
ap.add_argument('--micro', type=int, default=None); ap.add_argument('--min-piece', type=int, default=None)
ap.add_argument('--kv', default='f16', choices=('f16', 'q8'))
ap.add_argument('--yarn', type=float, default=None, help='the YaRN factor: the long-context mode (the 30B: YarnMode(F, 32,768); '
                'a qwen35moe file: Moe35Own(yarn=F))')
ap.add_argument('--ab', action='store_true', help='one card: each round also on the narrow tiles (Ops.wide / gu False: the '
                'pf_q3m_mm wide tiles and fused gate | up off), interleaved; the last layer\'s KV rows of both arms bitwise')
A = ap.parse_args()
Ls = [int(x) for x in A.lengths.split(',')]
from caminus.arch import QWEN3MOE_ROPE
from caminus.rope import YarnMode
rope = YarnMode(A.yarn, QWEN3MOE_ROPE.native_ctx) if A.yarn is not None else None   # (the 30B's long-context mode)
if A.cards == 2:
    from caminus.engine.q3moe_tp import Q3MoETP
    from caminus.prefill import q3m_tp
    if A.ops or A.prefill_module: raise SystemExit('--cards 2: no --ops / --prefill-module (the one-card prefill\'s)')
    modes = A.modes.split(',')
    print(torch.cuda.get_device_name(0), '+', torch.cuda.get_device_name(1), flush=True)
    eng = Q3MoETP(A.gguf, max_ctx=A.max_ctx or max(Ls) + 8, vram_extra_mib=q3m_tp.need_tp_mib(q3m_tp.tp_shapes(A.gguf), A.chunk),
                  q8=A.kv == 'q8', rope=rope)
    pf = q3m_tp.Q3MPrefillTP(eng, chunk=A.chunk, micro=A.micro or q3m_tp.MICRO, min_piece=A.min_piece or q3m_tp.MIN_PIECE)
    print(f'two cards: max_ctx {eng.max_ctx}, KV {A.kv}, rope {rope or "native"}, decode module {eng.module}, chunk {A.chunk}, micro {pf.micro} (pieces of a full chunk '
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
from caminus import ggufio
_F = ggufio.GGUFFile(A.gguf); ARCH = _F.arch; _F.close()
if ARCH == 'qwen35moe':
    from caminus.engine.q35_moe import Moe35Own
    from caminus.prefill import q35moe
    if A.ops or A.prefill_module: raise SystemExit('a qwen35moe file: no --ops / --prefill-module (the 30B\'s)')
    sh = q35moe.shapes_of(A.gguf)   # (the narrow arm's gate / up outputs projected with the engine when --ab runs it)
    eng = Moe35Own(A.gguf, max_ctx=A.max_ctx or max(Ls) + 8, q8=A.kv == 'q8', yarn=A.yarn,
                   vram_extra_mib=q35moe.buffer_need_mib(sh, A.chunk) + (q35moe.narrow_need_mib(sh, A.chunk) if A.ab else 0))
    pf = q35moe.Q35MoEPrefill(eng, chunk=A.chunk); pf.ops = pf.O
    print(f'max_ctx {eng.max_ctx}, chunk {A.chunk}, KV {A.kv}, rope {eng.rope_key or "native"} (prefill YaRN {pf.yarn}), decode module {eng.module}, prefill objects {pf.ops.module} + '
          f'{q35moe.WIDE_MODULE}: prefill buffers {pf.need_mib:.0f} MiB', flush=True)
else:
    eng = Q3MoEOne(A.gguf, max_ctx=A.max_ctx or max(Ls) + 8, q8=A.kv == 'q8', rope=rope)
    pf = Q3MPrefill(eng, chunk=A.chunk, module=A.prefill_module)
    print(f'max_ctx {eng.max_ctx}, chunk {A.chunk}, KV {A.kv}, rope {rope or "native"} (prefill YaRN {pf.yarn}), decode module {eng.module}, prefill object {pf.ops.module}: engine {eng.vram_mib:.0f} MiB + prefill buffers {pf.need_mib:.0f} MiB', flush=True)
g = np.random.default_rng(0)
arms = (True, False) if A.ab else (pf.ops.wide,)
bad = False
if A.ab and not pf.ops.wide: raise SystemExit(f'--ab: the prefill object {pf.ops.module} has no wide tiles')
for L in Ls:
    toks = g.integers(0, 151000, L).tolist()
    ts, last = {a: [] for a in arms}, {}
    for r in range(A.rounds + 1):
        for a in arms:
            pf.ops.wide = pf.ops.gu = a
            if hasattr(pf.ops, 'q8l'): pf.ops.q8l = a   # (the 35B's dense Q8_0 LDS tile: off in the narrow arm, as Ops(wide=False))
            eng.reset(); torch.cuda.synchronize(); t0 = time.perf_counter()
            pf.run(toks, 0, 0, L); torch.cuda.synchronize(); ts[a].append(time.perf_counter() - t0)
            if A.ab and r == A.rounds: last[a] = [L for L in eng.L if 'cache' in L][-1]['cache'].cpu()   # (host memory: the last attention layer's rows)
    for a in arms:
        s = float(np.median(ts[a][1:]))
        print(f'  L {L:6d}' + ((' wide  ' if a else ' narrow') if A.ab else '') + f': {s * 1e3:9.1f} ms = {L / s:7.0f} tok/s prefill '
              f'(rounds {min(ts[a][1:]) * 1e3:.1f} .. {max(ts[a][1:]) * 1e3:.1f})', flush=True)
    if A.ab:
        same = torch.equal(last[True], last[False])
        print(f'  L {L:6d}: the last layer\'s KV rows {"bitwise equal" if same else "DIFFER"} (wide vs narrow)', flush=True)
        if not same: bad = True
    pf.ops.wide = pf.ops.gu = arms[0]
    if hasattr(pf.ops, 'q8l'): pf.ops.q8l = arms[0]
    if A.ops:
        acc = collections.defaultdict(float); evs = []
        O = pf.ops
        for name in ('mm', 'mm_gu', 'emb', 'add_rms16', 'router', 'topk', 'group', 'swiglu', 'qk_rope', 'attn'):
            f = getattr(O, name)
            def wrap(*a, _f=f, _n=name, **kw):
                key = _n
                if _n == 'mm':
                    W = a[0]; key = f'mm {W[0]} {W[1]}x{W[2]}' + (' experts' if 'work' in kw else '')
                elif _n == 'mm_gu':
                    W = a[0]; key = f'mm_gu {W[0]} 2x{W[1]}x{W[2]} experts'
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); _f(*a, **kw); e1.record(); evs.append((key, e0, e1))
            setattr(O, name, wrap)
        eng.reset(); pf.run(toks, 0, 0, L); torch.cuda.synchronize()
        for k, e0, e1 in evs: acc[k] += e0.elapsed_time(e1)
        for name in ('mm', 'mm_gu', 'emb', 'add_rms16', 'router', 'topk', 'group', 'swiglu', 'qk_rope', 'attn'):
            delattr(O, name)
        tot = sum(acc.values())
        for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
            print(f'      {k:34s} {v:9.2f} ms  {100 * v / tot:5.1f} %', flush=True)
print('RESULT FAIL (the arms\' KV rows differ)' if bad else 'RESULT DONE')
sys.exit(1 if bad else 0)
