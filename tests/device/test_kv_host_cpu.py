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
