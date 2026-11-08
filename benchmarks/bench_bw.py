"""K3 engine cost model, part 2 (kernels/bw_bench.hip): read bandwidth vs working-set size with b128 loads, normal and
nontemporal, across grid sizes; plus the decode matvec (kernels/k3_mv.hip) streaming a full-size weight set to see the
bandwidth the real compute pattern reaches.
  HIP_VISIBLE_DEVICES=0 python benchmarks/bench_bw.py"""
import ctypes, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import torch
from caminus.device.hip import Module
from caminus.engine import k3

m = Module('bw_bench')
sink = torch.zeros(1, dtype=torch.int32, device='cuda')
wgps = torch.cuda.get_device_properties(0).multi_processor_count
print(torch.cuda.get_device_name(), f'{wgps} WGPs')


def timed(fn, reps=10):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps / 1e3   # s


print(f'{"MiB":>5s} | ' + ' | '.join(f'{g} WG {"nt" if nt else "  "}' for g in (wgps * 2, wgps * 4) for nt in (0, 1)) + '   (TB/s)')
for mib in (16, 32, 48, 64, 80, 88, 92, 96, 112, 128, 256):
    n16 = mib * 2**20 // 16
    buf = torch.randint(0, 2**31, (n16 * 4,), dtype=torch.int32, device='cuda')
    passes = max(1, 2048 // mib)
    row = []
    for grid in (wgps * 2, wgps * 4):
        for nt in (0, 1):
            t = timed(lambda: m.bw_read((grid,), (256,), buf, ctypes.c_size_t(n16), ctypes.c_int(passes), ctypes.c_int(nt), sink))
            row.append(f'{mib * 2**20 * passes / t / 1e12:9.2f}')
    print(f'{mib:5d} | ' + ' | '.join(row))
    del buf
# the decode matvec over a hot-set-sized stack of S1 linears (block projections + o + down x 12 + slack)
torch.manual_seed(0)
for total_mib, (N, K) in [(40, (7680, 896)), (80, (7680, 896))]:
    reps = max(1, int(total_mib * 2**20 // (N * K // 2)))
    ws = [k3.repack_mv(torch.randint(-8, 8, (N, K), device='cuda')) for _ in range(reps)]
    gs = torch.rand(N, K // 128, device='cuda'); a = torch.randint(-8, 8, (K,), device='cuda').to(torch.int8)
    t = timed(lambda: [k3.mv_g128(w, gs, a, 1.0) for w in ws])
    byts = reps * (N * K // 2 + gs.numel() * 4)
    print(f'k3_mv_g128 (one wave per row) over {reps} x {N}x{K} = {byts / 2**20:.0f} MiB: {byts / t / 1e12:.2f} TB/s '
          f'({t / reps * 1e6:.1f} us per matrix incl. launch)')
