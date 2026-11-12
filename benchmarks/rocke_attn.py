"""Yardstick: rocKE's RDNA WMMA attention (ROCm/rocm-libraries, dnn-providers/hip-kernel-provider/rocke; MIT) on the dense
qwen35 geometry (24 query heads over 4 kv heads, head dim 256, fp16), the shapes benchmarks/attn_depth.py and
benchmarks/q35_prefill_attn.py measure for our kernels:
  decode   T = 1 and T = 4 query rows over P cache positions (T 4: bottom-right causal, a verify pass)
  prefill  a 2048-row chunk at depth P0 over P0 + 2048 positions (bottom-right causal)
rocKE's own dispatch picks the kernel (split-KV GQA-packed decode `wmma_fmha_decode` or the multi-wave prefill
`wmma_fmha_fwd_v2`) and its own benchmark harness times it: HIP graph replay, the median of REPEATS batches of
GRAPH_COUNT launches. No PyTorch in this process (one HIP runtime: rocKE's). Correctness: against rocKE's float64 CPU
reference on every decode case and on the prefill chunk without a prefix (the reference of a deep prefill chunk takes
minutes on the CPU; those rows are timed unchecked, marked so).
Needs rocKE installed in the running interpreter (its platform/ and library/ packages, editable) and its library on the
path: run it with that environment's python, never the server's.
  tools/guard.sh LOG SECS python -I benchmarks/rocke_attn.py [--ctx 1024,4096,16384,32768,49152] [--p0 0,8192,30720]
Run it on the XT alone first."""
import argparse, os, statistics, sys
from contextlib import ExitStack
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
from caminus.device import host   # (sysfs only: no torch, no HIP)
from rocke.core.arch import generic_arch_from_target_id
from rocke.runtime.hip_module import Runtime, get_device_target_id
from benchmarks.gfx1151.attention import benchmark_sdpa as bs
from benchmarks.gfx1151.attention.candidate import RockeKernels, UnsupportedCase
from benchmarks.gfx1151.attention.cases import SdpaCase, make_inputs, reference

ap = argparse.ArgumentParser()
ap.add_argument('--ctx', default='1024,4096,16384,32768,49152'); ap.add_argument('--p0', default='0,8192,30720')
ap.add_argument('--graph-count', type=int, default=16); ap.add_argument('--repeats', type=int, default=5)
A = ap.parse_args()
H, NKV, D, N = 24, 4, 256, 2048
dev = get_device_target_id(); arch = generic_arch_from_target_id(dev or '')
print(f'device {dev} (rocKE target {arch}) | {H} q heads / {NKV} kv heads x {D}, fp16', flush=True)


def run(case, check):
    """bs._case_result's timing, with the reference skipped when check is False"""
    room(case)
    inputs = make_inputs(case)
    expected = reference(case, inputs) if check else None
    res = {}
    with ExitStack() as stack:
        buffers = bs.DeviceBuffers(rt, inputs); stack.callback(buffers.close)
        graphs = bs.HipGraphs(rt); stack.callback(kernels.release, graphs.stream); stack.callback(graphs.close)
        try:
            launch, res['kernel'] = kernels.prepare(case, buffers)
            launch(graphs.stream); rt.stream_sync(graphs.stream)
            if check:
                res.update(bs._check_output(buffers, 'rocke_out', expected, case.atol))
                if res['status'] != 'passed': return res
            else:
                res['status'] = 'unchecked'
            graph = graphs.capture(launch, A.graph_count)
            res['us'] = statistics.median(graphs.measure(graph, A.graph_count) for _ in range(A.repeats))
        except UnsupportedCase as ex:
            res.update(status='unsupported', reason=str(ex))
        except Exception as ex:
            res.update(status='error', reason=f'{type(ex).__name__}: {ex}')
    return res


def show(label, res, fl=None):
    k = res.get('kernel'); kn = k.get('name', k) if isinstance(k, dict) else k
    if 'us' in res:
        tf = f', {fl / res["us"] / 1e6:5.1f} TF' if fl else ''
        print(f'{label}: {res["us"]:9.1f} us{tf} [{res["status"]}] {kn}', flush=True)
    else:
        print(f'{label}: NOT RUN [{res["status"]}] {res.get("reason", "")[:160]}', flush=True)


MIB = 1 << 20


def room(case):
    """the capacity check without torch (caminus.device.capacity imports it, and this process holds only rocKE's HIP
    runtime): the case's device bytes (q, k, v, out in fp16, plus an allowance of 4 bytes per output value for rocKE's
    split workspace, which it sizes itself) must fit under the guard's kill line on every gfx1100 card the guard
    watches (caminus.device.host: sysfs, no HIP), keeping a display card's reserve, or the case is refused before it
    allocates"""
    need = (2 * (2 * case.seqlen_q * H * D + 2 * case.seqlen_k * NKV * D) + 4 * case.seqlen_q * H * D) / MIB
    for g in host.gpus():
        if not g.get('watched') or g.get('line_mib') is None: continue
        used = int(open(os.path.join(g['dev'], 'mem_info_vram_used')).read()) / MIB
        if used + need + (g.get('reserve_mib') or 0) > g['line_mib']:
            raise SystemExit(f"rocke_attn: {case.name} needs {need:.0f} MiB; {g['label']} has {used:.0f} MiB used of its {g['line_mib']} MiB line")
    return need


rt = Runtime(); kernels = RockeKernels(rt, arch)
try:
    for T in (1, 4):
        for P in [int(v) for v in A.ctx.split(',')]:
            c = SdpaCase(name=f'dec_T{T}_P{P}', group='caminus', dtype='fp16', batch=1, seqlen_q=T, seqlen_k=P, heads_q=H,
                         heads_kv=NKV, head_dim=D, mask='none' if T == 1 else 'causal_bottomright')
            show(f'decode T {T} ctx {P:6d}', run(c, True))
    for P0 in [int(v) for v in A.p0.split(',')]:
        c = SdpaCase(name=f'pre_P{P0}', group='caminus', dtype='fp16', batch=1, seqlen_q=N, seqlen_k=P0 + N, heads_q=H,
                     heads_kv=NKV, head_dim=D, mask='causal_bottomright')
        show(f'prefill chunk {N} at P0 {P0:6d}', run(c, P0 == 0), 4.0 * N * (P0 + N / 2) * H * D)
finally:
    kernels.close()
print('RESULT PASS (ran to the end; a measurement, not a gate)')
