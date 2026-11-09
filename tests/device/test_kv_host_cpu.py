"""The host KV cache's CPU-side rules (FlashNextTP(kv_host=True); nothing launched, no GPU):
  capacity   check_host_kv refuses a cache that leaves less than HOST_KV_MARGIN_MIB of MemAvailable, or passes 95 % of the
             GTT page limit; accepts one that fits; reads both from /proc/meminfo and ttm's pages_limit (host_kv_room)
  bufcheck   the launch checker accepts a CPU tensor inside a registered host_mapped() buffer (and a slice of it) and refuses
             any other CPU tensor, and a registered tensor that is not contiguous
  python -I tests/device/test_kv_host_cpu.py      ends RESULT PASS"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch
from caminus.device import capacity as C, bufcheck, hip

fails = []
def check(cond, msg):
    print(('PASS ' if cond else 'FAIL ') + msg, flush=True)
    if not cond: fails.append(msg)

MIB = C.MIB
ma, lim = C.host_kv_room()
check(ma is not None and ma > 0, f'host_kv_room reads MemAvailable ({ma and round(ma)} MiB) and the GTT limit ({lim and round(lim)} MiB)')
real = C.host_kv_room
def refused(nb, room):
    C.host_kv_room = lambda: room
    try: C.check_host_kv(nb); return False
    except C.CapacityError: return True
    finally: C.host_kv_room = real
check(not refused(13_056 * MIB, (20_000.0, 15_222.0)), 'a 1M-context cache (13,056 MiB) fits 20,000 MiB available and a 15,222 MiB GTT limit')
check(refused(13_056 * MIB, (17_000.0, 15_222.0)), 'refused when it would leave less than the margin of MemAvailable')
check(refused(14_600 * MIB, (40_000.0, 15_222.0)), 'refused past 95 % of the GTT page limit')
check(not refused(6_528 * MIB, (20_000.0, None)), 'no GTT limit readable: MemAvailable alone decides')

# the projection follows the cache format FlashNextTP allocates (q8_0: 544 bytes per position per layer and card on the
# Coder; fp32: 1,024), and _check_vram hands check_host_kv that figure for both cards
g = dict(n_attn=12, nkv=2, hd=256, idx_dim=128)
n1m = 1 << 20
check(C.tp_kv_bytes(g, n1m, True) == 12 * n1m * 544 and C.tp_kv_bytes(g, n1m, False) == 12 * n1m * 1024
      and C.tp_ctx_bytes(g, n1m)['kv'] == C.tp_kv_bytes(g, n1m, True) and C.tp_ctx_bytes(g, n1m, q8=False)['kv'] == C.tp_kv_bytes(g, n1m, False),
      'tp_kv_bytes: 544 (q8_0) / 1,024 (fp32) bytes per position per layer and card; tp_ctx_bytes counts the same')
from caminus.engine.flashnext_tp import FlashNextTP
class _Stop(Exception): pass
seen, saved = [], (C.model_bytes, C.fixed_mib, C.check_host_kv)
C.model_bytes = lambda rd: 1; C.fixed_mib = lambda *a, **k: [1.0, 1.0]
def _rec(nb): seen.append(nb); raise _Stop()
C.check_host_kv = _rec
try:
    for q8 in (True, False):
        m = FlashNextTP.__new__(FlashNextTP)
        m.__dict__.update(rd=None, gdn_keys=10, moe_blocks=12, model_tag='gsq', ntok=4, geom=g, max_ctx=n1m, idx_cache=True,
                          kv_host=True, q8=q8, qsa=True)
        seen.clear()
        try: m._check_vram(0)
        except _Stop: pass
        want = 2 * C.tp_kv_bytes(g, n1m, q8)
        check(seen == [want], f'_check_vram (kv_host, q8={q8}) checks {want // MIB:,} MiB of host memory for both cards ({[x // MIB for x in seen]})')
finally:
    C.model_bytes, C.fixed_mib, C.check_host_kv = saved

# the launch checker: a fake registry entry over a real CPU buffer (no hipHostMalloc on a CPU-only runner)
buf = torch.zeros(4096, dtype=torch.uint8)
hip._HOST.append((buf.data_ptr(), buf.numel(), buf))
chk = bufcheck.LaunchCheck('kv_host cpu test')
dev = torch.device('cuda', 0)
def accepted(t):
    try: chk(dev, 'probe', dict(cache=(t, torch.uint8, t.numel())), ()); return True
    except ValueError: return False
check(accepted(buf), 'bufcheck accepts a registered host-mapped buffer')
check(accepted(buf[1024:2048]), 'and a contiguous slice of it')
check(not accepted(torch.zeros(4096, dtype=torch.uint8)), 'bufcheck refuses an unregistered CPU tensor')
check(not accepted(buf.view(64, 64).t()), 'bufcheck refuses a registered but non-contiguous view')
check(hip.host_mapped_bytes() >= 4096, 'host_mapped_bytes counts the registry')
hip._HOST.pop()
print(f'RESULT {"PASS" if not fails else "FAIL"}')
sys.exit(1 if fails else 0)
