"""The KV cache in host memory (caminus.device.hip.host_mapped: pinned, mapped into every card; the opt-in long-context mode,
FlashNextTP(kv_host=True)) on ONE card (run it on the XT first: HIP_VISIBLE_DEVICES=1), module k3_gguf_gsq, the TP card's
geometry (12 query heads over 1 kv head of 256, q8_0):
  alloc     host_mapped returns exactly the bytes asked, zeroed, inside the registry; bufcheck accepts it and refuses a
            plain CPU tensor
  chain     a tape of kv_append (this pass's T rows) -> the split attention over 0 .. pos (it reads the rows the same pass
            just wrote) run pass after pass from position 0 to past 2051, T = 1 and T = 4, once with the cache in VRAM and
            once in host memory: every pass's output and the final cache bytes bitwise equal, run twice (run-to-run)
  speed     the attention pair at 2051 cells: VRAM vs host (labelled: one card, synthetic)
  HIP_VISIBLE_DEVICES=1 tools/guard.sh LOG 600 python -I tests/engine/test_kv_host.py [--end 2200]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus.engine import k3
from caminus.device import hip, bufcheck

a = sys.argv[1:]
END = int(a[a.index('--end') + 1]) if '--end' in a else 2200
MOD, D, H, NKV = 'k3_gguf_gsq', 256, 12, 1
fails = []
def ok(name, cond, info=''):
    print(f'{"ok  " if cond else "FAIL"} {name} {info}', flush=True)
    if not cond: fails.append(name)

ROWS = 4096
PB = k3.kv_pos_bytes(NKV * D, True)
# alloc
h = hip.host_mapped(ROWS * PB)
ok('host_mapped: exact size, zeroed, registered', h.numel() == ROWS * PB and h.device.type == 'cpu' and int(h[:4096].sum()) == 0
   and hip.is_host_mapped(h) and hip.is_host_mapped(h[PB:2 * PB]), f'({h.numel()} B at 0x{h.data_ptr():x})')
chk = bufcheck.LaunchCheck('kv_host test')
try: chk(torch.device('cuda', 0), 'probe', dict(cache=(h, torch.uint8, h.numel())), ()); acc = True
except ValueError: acc = False
try: chk(torch.device('cuda', 0), 'probe', dict(cache=(torch.zeros(64, dtype=torch.uint8), torch.uint8, 64)), ()); rej = False
except ValueError: rej = True
ok('bufcheck: accepts the host_mapped cache, refuses a plain CPU tensor', acc and rej)

def build(T, cache):
    cu = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device='cuda')
    k, v, q, out, pos = cu(T * NKV * D), cu(T * NKV * D), cu(T * H * D), cu(T * H * D), cu(1, dt=torch.int32)
    e = k3.Engine(module=MOD, grid=k3.resident_grid(MOD), T=T, limit_us=200_000)
    d = e.kv_append(k, v, cache, pos, q8=True)
    e.attn_dec(q, cache, pos, out, H, NKV, D, 1.0 / 16, q8=True, dep=d)
    e.build()
    return e, (k, v, q, out, pos)

def chain(T, cache, seed):
    """passes at 0, T, 2T, ... < END (attention identity lists up to 2051 cells: the split modules' dense tape limit)"""
    e, (k, v, q, out, pos) = build(T, cache)
    g = torch.Generator(device='cuda'); g.manual_seed(seed); outs = []
    p = 0
    while p + T <= min(END, 2051 + 1 - T + 1):
        k.copy_(torch.randn(k.numel(), device='cuda', generator=g)); v.copy_(torch.randn(v.numel(), device='cuda', generator=g))
        q.copy_(torch.randn(q.numel(), device='cuda', generator=g)); pos.fill_(p)
        e.run(check=True); outs.append(out.view(torch.int32).cpu().clone()); p += T
    return outs, p

for T in (1, 4):
    res = {}
    for where in ('vram', 'host', 'host2'):
        cache = torch.zeros(ROWS * PB, dtype=torch.uint8, device='cuda') if where == 'vram' else h
        if where != 'vram': h.zero_()
        outs, end = chain(T, cache, 11)
        res[where] = (outs, cache.cpu().clone() if where == 'vram' else h.clone())
    o0, c0 = res['vram']; o1, c1 = res['host']; o2, c2 = res['host2']
    bad = [i for i in range(len(o0)) if not (torch.equal(o0[i], o1[i]) and torch.equal(o1[i], o2[i]))]
    ok(f'T={T}: append + attention, host cache == VRAM cache bitwise at every pass, twice', not bad,
       f'({len(o0)} passes to position {end - 1}; first bad {bad[:3]})')
    ok(f'T={T}: the cache bytes written, host == VRAM', torch.equal(c0, c1) and torch.equal(c1, c2))

# speed: the attention pair alone at 2051 cells (one card, synthetic)
for where in ('vram', 'host'):
    cache = torch.zeros(ROWS * PB, dtype=torch.uint8, device='cuda') if where == 'vram' else h
    cache.copy_(torch.randint(0, 256, (ROWS * PB,), dtype=torch.uint8).to(cache.device) if where == 'vram' else torch.randint(0, 256, (ROWS * PB,), dtype=torch.uint8))
    cu = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device='cuda')
    q, out, pos = cu(H * D), cu(H * D), cu(1, dt=torch.int32)
    e = k3.Engine(module=MOD, grid=k3.resident_grid(MOD), T=1, limit_us=200_000)
    for _ in range(12): e.attn_dec(q, cache, pos, out, H, NKV, D, 1.0 / 16, q8=True, dep=len(e.segs) - 1)   # 12 layers' worth
    e.build(); pos.fill_(2050); q.normal_()
    for _ in range(5): e.run(check=True)
    ts = []
    for _ in range(30):
        torch.cuda.synchronize(); t0 = time.perf_counter(); e.run(check=False); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    print(f'     speed (synthetic, one card): 12 x attention over 2051 cells, cache in {where}: {np.median(ts) * 1e3:.3f} ms', flush=True)
print(f'RESULT {"PASS" if not fails else "FAIL"} ({len(fails)} failed)')
