"""The split attention pair (OP_ATTN_PART + OP_ATTN_COMB, kernels/k3_attn_split.h) alone at a given context length: a tape
of just the pair on a zeroed q8_0 KV cache of P cells (the timing does not depend on the values), the shapes of the one-card
Qwen3.6 engine (16 query heads over 2 kv heads, head dim 256) or --nkv 1 (Qwen3-Next's per-card share). Prints the median
launch time over REPS runs (host events around E.run: one persistent launch, so it includes the launch) and, with the trace
build (--trace: k3_gguf_q35h_trace), each segment's span (first claim to last publish, the 100 MHz steady counter) and
its task count. Run it on the XT alone first when the kernel changed.
  tools/guard.sh LOG SECS python -I benchmarks/attn_bench.py [--ctx 2048,8192,32768] [--nkv 2] [--reps 50] [--trace] [--module M]
      [--ss 64|128: M is a scratch build with that ATTN_SS, registered for the run]
--new: the qwen35moe decode attention kernel (kernels/dec_q35m_attn.hip, caminus.engine.dec_q35m_attn) against the pair on
the qwen35moe module (default k3_q35_moe, or k3_q35_moe_kvq8 with --kv q8) at the engine's grid (k3.card_per_wgp), on
random cache rows (Q8: random codes, scale 0.01). Both arms timed the same way, interleaved per round: a queued batch
(a filler matmul holds the stream while the host queues the start event, --batch launches back to back and the stop
event), so a time is the GPU's per launch, launch gaps included and host overhead not; median over --reps rounds.
--cold rotates over enough copies of the cache that each launch's rows are not in the Infinity Cache (> 2 x 96 MB
between two reads of a copy); without it every launch rereads the same rows, as rocKE's graph replay does. --ns N
fixes the new kernel's splits (default: caminus.engine.dec_q35m_attn.splits of the context).
  tools/guard.sh LOG SECS python -I benchmarks/attn_bench.py --new [--kv f16|q8] [--ctx 4096,32768,131072] [--cold] [--ns N]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np, torch
from caminus.engine import k3

a = sys.argv[1:]
def opt(n, d, f=str):
    if n in a: i = a.index(n); v = f(a[i + 1]); del a[i:i + 2]; return v
    return d
NEW, COLD = '--new' in a, '--cold' in a
KV = opt('--kv', 'q8'); Q8 = KV == 'q8'
CTX = [int(v) for v in opt('--ctx', '4096,32768,131072' if NEW else '2048,8192,32768').split(',')]; NKV = opt('--nkv', 2, int)
REPS = opt('--reps', 15 if NEW else 50, int); BATCH = opt('--batch', 8, int); NS = opt('--ns', None, int)
TRACE = '--trace' in a
MOD = opt('--module', ('k3_q35_moe_kvq8' if Q8 else 'k3_q35_moe') if NEW else ('k3_gguf_q35h_trace' if TRACE else 'k3_gguf_q35h'))
SS_SCRATCH = opt('--ss', None, int)
if SS_SCRATCH is not None:   # a scratch build (build/<MOD>.hsaco compiled with -DATTN_SS=...): registered for this run only
    k3.MODULE_CAPS[MOD] = k3.MODULE_CAPS['k3_gguf_q35h']; k3.ATTN_G_OF[MOD] = 8; k3.ATTN_SS_OF[MOD] = SS_SCRATCH
    k3.ATTN_SPLIT_MODULES = k3.ATTN_SPLIT_MODULES + (MOD,)
D = 256; G = k3.ATTN_G_OF[MOD]; H = G * NKV
torch.cuda.set_per_process_memory_fraction(0.9, 0)


def pair(cache, q, out, pos, P):
    """an engine holding only the pair over `cache` (P cells), at the module's resident grid (--new: the engine's)"""
    SS = k3.ATTN_SS_OF.get(MOD, k3.ATTN_SS); ns = (P + SS - 1) // SS
    part = torch.zeros(H * ns * k3.ATTN_RS, device='cuda')
    grid = k3.resident_grid(MOD, k3.card_per_wgp() if NEW else None)
    E = k3.Engine(module=MOD, grid=grid, T=1, limit_us=200_000)
    E.attn_dec(q, cache, pos, out, H, NKV, D, 1.0 / 16, q8=Q8 if NEW else True, part=part, max_cells=P)
    E.build()
    return E, part, ns


if not NEW:
    for P in CTX:
        cache = torch.zeros(P * k3.kv_pos_bytes(NKV * D, True), dtype=torch.uint8, device='cuda')
        q = torch.randn(H * D, device='cuda'); out = torch.zeros(H * D, device='cuda'); pos = torch.full((1,), P - 1, dtype=torch.int32, device='cuda')
        E, part, ns = pair(cache, q, out, pos, P); nseg = len(E.segs); grid = E.grid
        for _ in range(5): E.run(check=True)
        ts = []
        for _ in range(REPS):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); E.run(check=False); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3)
        line = f'ctx {P:6d} (splits {ns:5d} x {NKV} kv heads, {H} heads): launch median {np.median(ts):8.1f} us (p10 {np.percentile(ts, 10):.1f}, p90 {np.percentile(ts, 90):.1f})'
        if TRACE:
            tr = torch.zeros(grid * nseg * 8, dtype=torch.int64, device='cuda'); spans = []
            for _ in range(10):
                tr.zero_(); E.run(trace=tr); torch.cuda.synchronize(); T = tr.view(grid, nseg, 8).cpu().numpy().astype(np.float64)
                row = []
                for s in range(nseg):
                    st, en = T[:, s, 0], T[:, s, 7]; w = en > 0
                    row.append((en[w].max() - st[w].min()) / 100.0 if w.any() else float('nan'))
                spans.append(row)
            sp = np.median(np.array(spans), axis=0)
            line += ' | spans: ' + ', '.join(f'{nm} {v:.1f} us' for nm, v in zip(('PART', 'COMB'), sp))
        print(line, flush=True)
        del cache, part, E; torch.cuda.empty_cache()
    sys.exit(0)

# ---- --new: the standalone kernel against the pair, both queued
from caminus.device import capacity
from caminus.engine.dec_q35m_attn import DecQ35mAttn, splits
assert (H, NKV) == (16, 2), 'dec_q35m_attn: 16 query heads over 2 kv heads'
att = DecQ35mAttn(Q8)
print(f'{torch.cuda.get_device_name(0)} | {KV} cache | pair on {MOD} (grid {k3.resident_grid(MOD, k3.card_per_wgp())}) | '
      f'new: dec_q35m_attn_{KV} | {"cold (rotating copies)" if COLD else "warm (the same rows each launch)"} | '
      f'batch {BATCH}, median of {REPS} rounds, interleaved', flush=True)


FILL_MIB = 3 * 3072 * 3072 * 4 / 2 ** 20                           # (108 MiB: projected before it is allocated)
capacity.check([0], [FILL_MIB], 'attn_bench --new filler')
FILL = [torch.randn(3072, 3072, device='cuda') for _ in range(3)]   # (a ~2-5 ms matmul holds the stream while the host queues)


def queued(fns):
    """GPU us per launch of each fn: a filler matmul, the start event, BATCH launches, the stop event"""
    out = []
    for fn in fns:
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.mm(FILL[0], FILL[1], out=FILL[2])
        e0.record()
        for i in range(BATCH): fn(i)
        e1.record(); e1.synchronize(); out.append(e0.elapsed_time(e1) * 1e3 / BATCH)
    return out


for P in CTX:
    pb = k3.kv_pos_bytes(NKV * D, Q8); nb = P * pb
    ncopy = max(1, -(-2 * 96 * 2 ** 20 // nb)) + 1 if COLD else 1
    capacity.check([0], [ncopy * nb / 2 ** 20 + 64], f'attn_bench --new ctx {P}')
    caches = []
    for _ in range(ncopy):
        if Q8:
            c = torch.randint(0, 256, (nb,), dtype=torch.uint8, device='cuda')
            c.view(-1, 34)[:, :2] = torch.tensor([0.01], dtype=torch.float16).view(torch.uint8).to('cuda')
        else:
            c = torch.empty(nb // 2, dtype=torch.float16, device='cuda').normal_()   # (in place: no fp32 temporary)
        caches.append(c)
    q = torch.randn(H * D, device='cuda'); out = torch.zeros(H * D, device='cuda'); pos = torch.full((1,), P - 1, dtype=torch.int32, device='cuda')
    pairs = [pair(c, q, out, pos, P) for c in caches]
    ns = NS or splits(P)
    fn_new = lambda i: att(q, caches[i % ncopy], pos, out, 1.0 / 16, ns=ns)
    fn_pair = lambda i: pairs[i % ncopy][0].run(check=False)
    for _ in range(2): fn_new(0); pairs[0][0].run(check=True)
    torch.cuda.synchronize()
    tn, tp = [], []
    for r in range(REPS):
        x = queued([fn_new, fn_pair] if r % 2 == 0 else [fn_pair, fn_new])
        tn.append(x[0] if r % 2 == 0 else x[1]); tp.append(x[1] if r % 2 == 0 else x[0])
    if att.error(): print('ERROR: the kernel\'s error word is set', flush=True)
    mb = nb / 1e6
    print(f'ctx {P:6d} ({mb:6.1f} MB of KV, {ncopy} cop{"ies" if ncopy > 1 else "y"}): new {np.median(tn):7.1f} us '
          f'(p10 {np.percentile(tn, 10):.1f}, p90 {np.percentile(tn, 90):.1f}; {ns} splits; {mb / np.median(tn) * 1e3:.0f} GB/s) | '
          f'pair {np.median(tp):7.1f} us (p10 {np.percentile(tp, 10):.1f}, p90 {np.percentile(tp, 90):.1f}) | '
          f'pair / new {np.median(tp) / np.median(tn):.2f}x', flush=True)
    del caches, pairs; torch.cuda.empty_cache()
print('RESULT PASS (ran to the end; a measurement, not a gate)', flush=True)
