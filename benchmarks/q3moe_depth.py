"""Qwen3-30B-A3B decode speed by context depth (Q3MoEOne, engine/q3moe_one.py): ms per token at chosen positions, every
arm at the same positions in turn (interleaved), medians over --tokens steps and --rounds rounds. The position is set
directly (no prompt is processed first): decode time does not depend on the cache's values, only on how many rows the
attention reads, so a 40K-position measurement takes seconds instead of 40,960 real passes. Prints one line per depth
and arm, then RESULT DONE.
  tools/guard.sh LOG 1800 python -I benchmarks/q3moe_depth.py GGUF [--depths 8,1024,4096,16384,40900] [--max-ctx N]
      [--arms new=k3_q3m_lean old=k3_gguf_q3m_own:attn_split=0 ...]
An arm is name=module[@workgroups per WGP][:option=int ...] (engine options: attn_split, splits, lmin, gh, ...).
--cards 2: the two-card engine (engine/q3moe_tp.py Q3MoETP, torch devices 0 and 1, its default split; CAMINUS_UPLOAD_LOG on
every two-card run); --arms then names grids only (name=@workgroups per WGP, default: each card's own; :cap=N after it: at most
N attention splits per kv head, 1 .. 128, default 128 (q3moe_tp.SPLITS_CAP), e.g. c64=@card:cap=64; past 64 splits * 64 positions the arms'
logits differ in their last bits, the attention's summation order). A pass is timed
from card 0's launch to the host's return with both cards' argmax read back (the served pass's shape), and each card's
own span is printed beside it. --local-only (with --cards 2): every exchange copies the card's own partial, so each card
runs alone (its ceiling; the logits are not the model's): the two-card time minus the slower card's alone is the coupling.
The one-card engine does not fit card 0 beside the two-card one, so compare with a one-card run of this script in the
same window, runs alternated."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus.engine import k3
from caminus.engine.q3moe_one import Q3MoEOne
ap = argparse.ArgumentParser()
ap.add_argument('gguf'); ap.add_argument('--depths', default='8,1024,2048,4096,8192')
ap.add_argument('--max-ctx', type=int, default=None, help='default: the deepest position + the tokens timed')
ap.add_argument('--tokens', type=int, default=24); ap.add_argument('--rounds', type=int, default=3)
ap.add_argument('--arms', nargs='+', default=None)
ap.add_argument('--cards', type=int, default=1, choices=(1, 2)); ap.add_argument('--local-only', action='store_true')
A = ap.parse_args()
depths = [int(x) for x in A.depths.split(',')]
if A.local_only and A.cards != 2: raise SystemExit('q3moe_depth: --local-only is a two-card option (--cards 2)')
if A.cards == 2:
    from caminus.engine.q3moe_tp import Q3MoETP, SPLITS_CAP
    arms = A.arms or ['default=@card']
    print(torch.cuda.get_device_name(0), '+', torch.cuda.get_device_name(1), flush=True)
    max_ctx = A.max_ctx or max(depths) + A.tokens + 1
    caps = [int(a.split('=', 1)[1].partition(':cap=')[2] or SPLITS_CAP) for a in arms]
    if not all(1 <= c <= max(caps) <= 128 for c in caps) or min(caps) < 1:
        raise SystemExit(f'q3moe_depth: arm caps {caps}: each must be 1 .. 128 (the two-card modules\' attention merge)')
    eng = Q3MoETP(A.gguf, max_ctx=max_ctx, local_only=A.local_only, vram_extra_mib=[len(arms) * Q3MoETP.TAPE_MIB] * 2,   # (+ a tape set per arm,
                  splits_cap=max(caps))                                                                                  #  the projection at the largest cap)
    print(f'max_ctx {max_ctx}: {eng.vram_mib[0]:.0f} / {eng.vram_mib[1]:.0f} MiB; expert rows {eng.exp_rows}, kv heads {eng.kvc}, '
          f'head rows {eng.vh} / {eng.vocab - eng.vh}{"; LOCAL ONLY (each card alone, not the model)" if A.local_only else ""}', flush=True)
    eng.enable_card_timing(True)
    keys = ('E', 'buf', 'dstage', 'hstage', 'tok', 'pos', 'epoch', 'splits', 'per_wgp_c', 'seg_names', 'layer_seg', 'logits_half',
            'd_max', 'h_max', 'h_st', 'ev_done', 'engines', 'res')
    forms = {}
    for a in arms:
        name, spec = a.split('=', 1)
        spec, _, cap = spec.partition(':cap=')   # (name=@card:cap=N: at most N attention splits per kv head)
        eng.per_wgp = 'card' if spec in ('', '@card') else int(spec.lstrip('@'))
        eng.splits_cap = int(cap) if cap else SPLITS_CAP   # (within the constructor's cap: max(caps), checked and projected there)
        eng._build_tapes(); forms[name] = tuple(getattr(eng, k) for k in keys)
        print(f'  arm {name}: {eng.module}, {eng.per_wgp_c} workgroups per WGP, attention splits {eng.splits}', flush=True)
    names = list(forms)
    def use(n):
        for k, v in zip(keys, forms[n]): setattr(eng, k, v)
    res = {}
    for P in depths:
        med = {n: [] for n in names}; card = {n: [] for n in names}
        for rd in range(A.rounds):
            tm = {n: [] for n in names}; cm = {n: [] for n in names}; tok = 785
            for i in range(A.tokens):
                for n in names[i % len(names):] + names[:i % len(names)]:
                    use(n)
                    t0 = time.perf_counter(); nxt = eng._run_tok(tok, P + i, 1, True)[0]; t1 = time.perf_counter()
                    tm[n].append((t1 - t0) * 1e3); cm[n].append(eng.card_ms)
                tok = nxt
            for n in names:
                med[n].append(float(np.median(tm[n][4:]))); card[n].append(np.median(np.array(cm[n][4:]), axis=0))
        for n in names:
            m = float(np.median(med[n])); cc = np.median(np.array(card[n]), axis=0); res[(P, n)] = m
            print(f'  depth {P:6d}  {n:10s} {m:7.3f} ms/token = {1e3 / m:6.1f} tok/s (rounds {min(med[n]):.3f} .. {max(med[n]):.3f}); '
                  f'card 0 {cc[0]:.3f}, card 1 {cc[1]:.3f} ms' + (' (each alone)' if A.local_only else ''), flush=True)
    print(f'  bails {len(eng.bails)}')
    for n in names:
        x = np.array(depths, dtype=np.float64); y = np.array([res[(P, n)] for P in depths])
        if len(x) > 1:
            sl, ic = np.polyfit(x[1:] if len(x) > 2 else x, y[1:] if len(x) > 2 else y, 1)
            print(f'  {n:10s} fit (depths past the first): {ic:.3f} ms + {sl * 1e3:.4f} us per position per token')
    print('RESULT DONE')
    sys.exit(0)
A.arms = A.arms or ['default=k3_q3m_lean']
ARMS = []
for a in A.arms:
    name, spec = a.split('=', 1); spec, *kv = spec.split(':'); mod, _, g = spec.partition('@')
    if mod not in k3.MODULE_CAPS:   # (a measurement build under another name: of the module its name starts with, else the full one's ops)
        base = max((m for m in k3.Q3M_LEAN_MODULES + k3.Q3M_IQ_MODULES + tuple(k3.Q3M_RS_MODULES) if mod.startswith(m + '_')), key=len, default='k3_gguf_q3m_own')
        k3.MODULE_CAPS[mod] = k3.MODULE_CAPS[base]; k3.Q3M_MODULES = k3.Q3M_MODULES + (mod,)
        if base in k3.Q3M_LEAN_MODULES: k3.Q3M_LEAN_MODULES = k3.Q3M_LEAN_MODULES + (mod,)
        if base in k3.Q3M_IQ_MODULES: k3.Q3M_IQ_MODULES = k3.Q3M_IQ_MODULES + (mod,); k3.ROUTER_NE[mod] = k3.ROUTER_NE[base]
        if base in k3.Q3M_RS_MODULES: k3.Q3M_RS_MODULES[mod] = k3.Q3M_RS_MODULES[base]
    o = dict(module=mod)
    if g: o['per_wgp'] = int(g)
    for x in kv: k_, v_ = x.split('='); o[k_] = int(v_)
    ARMS.append((name, o))
print(torch.cuda.get_device_name(0), flush=True)
max_ctx = A.max_ctx or max(depths) + A.tokens + 1
eng = Q3MoEOne(A.gguf, max_ctx=max_ctx, vram_extra_mib=len(ARMS) * Q3MoEOne.TAPE_MIB, **ARMS[0][1])   # (+ a tape per arm, built below)
print(f'max_ctx {max_ctx}: {eng.vram_mib:.0f} MiB', flush=True)
keys = ('E', 'S', 'dstage', 'hstage', 'tok', 'pos')
forms = {}
for name, opt in ARMS:
    for k, v in opt.items(): setattr(eng, k, v)
    with torch.cuda.device(eng.dev): eng._build_tape()
    forms[name] = tuple(getattr(eng, k) for k in keys)
    print(f'  arm {name}: {eng.module}, {eng.per_wgp} workgroups per WGP ({k3.resident_grid(eng.module, eng.per_wgp)} in all)', flush=True)
def use(n):
    for k, v in zip(keys, forms[n]): setattr(eng, k, v)
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
names = list(forms)
res = {}
for P in depths:
    med = {n: [] for n in names}
    for rd in range(A.rounds):
        tm = {n: [] for n in names}; tok = 785
        for i in range(A.tokens):
            for n in names[i % len(names):] + names[:i % len(names)]:
                use(n); eng.pos_host = P + i
                ev[0].record(); lg = eng.step(tok); ev[1].record(); torch.cuda.synchronize()
                tm[n].append(ev[0].elapsed_time(ev[1]))
            tok = int(torch.argmax(lg))
        for n in names: med[n].append(float(np.median(tm[n][4:])))
    for n in names:
        m = float(np.median(med[n])); res[(P, n)] = m
        print(f'  depth {P:6d}  {n:10s} {m:7.3f} ms/token = {1e3 / m:6.1f} tok/s (rounds {min(med[n]):.3f} .. {max(med[n]):.3f})', flush=True)
for n in names:
    x = np.array(depths, dtype=np.float64); y = np.array([res[(P, n)] for P in depths])
    if len(x) > 1:
        sl, ic = np.polyfit(x[1:] if len(x) > 2 else x, y[1:] if len(x) > 2 else y, 1)
        print(f'  {n:10s} fit (depths past the first): {ic:.3f} ms + {sl * 1e3:.4f} us per position per token')
print('RESULT DONE')
