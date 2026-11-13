"""The dense qwen35 prefill's GEMMs on their real shapes and formats (one GDN block and one attention block of a GGUF file:
Swift 1.5 Q4_K_M by default), M tokens at a time: our own WMMA GEMMs reading the quantized blocks (kernels/q35_prefill.hip
via prefill/q35own.py Ops.mm: Q4_K, Q6_K, Q8_0, Q5_K), next to yardsticks on the SAME (M, N, K) with dense operands:
  torch      torch.matmul in fp16 (PyTorch picks rocBLAS or hipBLASLt): fp16 weights, 2 bytes per weight
  flydsl     FlyDSL's RDNA3 WMMA GEMMs (kernels/gemm/rdna3_f16_gemm_autotune.py, rdna3_int8_gemm_autotune.py from a FlyDSL
             source checkout; --flydsl-src DIR, the flydsl package installed in the running interpreter): fp16 TN with
             fp32 output, and int8 x int8 -> int32 (exact), each on its own autotuned tile
Like-for-like: ours reads 4.5-8.5 bits per weight and dequantizes inside the GEMM; the yardsticks read dense 16- or 8-bit
operands and do no dequantization, so they bound what a dense GEMM reaches on this card at the shape, not a drop-in.
Timing: events around ITERS back-to-back launches, median of REPS such groups, the arms interleaved per shape; FlyDSL's
arms replay the ITERS launches from a captured HIP graph (its own tuner's method), the others launch eagerly.
TFLOPS = 2 M N K / t. 'per chunk' = the shape's time x its count in the model (48 GDN blocks, 16 attention blocks).
  tools/guard.sh LOG SECS python -I benchmarks/q35_prefill_gemm.py [GGUF] [--M 512,2048] [--arms own,torch,flydsl]
      [--flydsl-src DIR] [--reps 7] [--iters 10] [--module q35_prefill_abl1]
Run it on the XT alone first."""
import argparse, os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np, torch
from caminus import ggufio
from caminus.device import capacity
from caminus.engine.q35_dense import Dense35Own
from caminus.prefill import q35own

ap = argparse.ArgumentParser()
ap.add_argument('gguf', nargs='?', default=os.path.expanduser(os.environ.get('SWIFT', '')))
ap.add_argument('--M', default='512,2048'); ap.add_argument('--arms', default='own,torch')
ap.add_argument('--module', default=None, help='the prefill object of the own arm (default the file\'s: q35_prefill / q35_prefill_tcap); a measurement build such as q35_prefill_abl1 times what is left of the tile without one of its parts (kernels/q3m_prefill.hip Q3P_ABL; its outputs are wrong by design)')
ap.add_argument('--flydsl-src', default=None); ap.add_argument('--reps', type=int, default=7); ap.add_argument('--iters', type=int, default=10)
A = ap.parse_args()
if not A.gguf: sys.exit('q35_prefill_gemm: give the GGUF path (or export SWIFT)')
ARMS = A.arms.split(','); MS = [int(v) for v in A.M.split(',')]
dev = torch.device('cuda:0')
F = ggufio.GGUFFile(A.gguf)
blocks = [int(n.split('.')[1]) for n in F.T if n.startswith('blk.') and n.endswith('.attn_q.weight')]
ab = blocks[0]; gb = 0 if ab != 0 else 1
NAMES = [(gb, 'attn_qkv', 48), (gb, 'attn_gate', 48), (gb, 'ssm_out', 48), (gb, 'ffn_gate', 48), (gb, 'ffn_up', 48), (gb, 'ffn_down', 48),
         (ab, 'attn_q', 16), (ab, 'attn_k', 16), (ab, 'attn_v', 16), (ab, 'attn_output', 16), (ab, 'ffn_gate', 16), (ab, 'ffn_up', 16), (ab, 'ffn_down', 16)]
fake = types.SimpleNamespace(F=F, tcap=False, module='k3_q35_swift')
fly_f16 = fly_i8 = None
if 'flydsl' in ARMS:
    if not A.flydsl_src: sys.exit('q35_prefill_gemm: --arms flydsl needs --flydsl-src (a FlyDSL source checkout matching the installed package)')
    sys.path.insert(0, os.path.abspath(A.flydsl_src))
    from kernels.gemm.rdna3_f16_gemm_autotune import rdna3_gemm_autotuned as fly_f16
    from kernels.gemm.rdna3_int8_gemm_autotune import rdna3_int8_gemm_autotuned as fly_i8
ops = q35own.Ops(dev) if A.module is None else q35own.Ops(dev, module=A.module)
print(torch.cuda.get_device_name(dev), os.path.basename(A.gguf), f'| GDN block {gb}, attention block {ab} | arms {ARMS} | own object {ops.module}', flush=True)


def bench(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize(dev); ts = []
    for _ in range(A.reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(A.iters): fn()
        e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3 / A.iters)
    return float(np.median(ts))


def bench_graph(fn):
    """FlyDSL's own timing method (its autotuner's _graph_bench): the launches captured in a HIP graph, then replayed, so
    the host launch path (measured ~30 ms per eager call through its Python JIT dispatch) does not count"""
    for _ in range(3): fn()
    torch.cuda.synchronize(dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(A.iters): fn()
    for _ in range(2): g.replay()
    torch.cuda.synchronize(dev); ts = []
    for _ in range(A.reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3 / A.iters)
    del g
    return float(np.median(ts))


AB_DIFF = []
CMP_ROWS = 256   # rows per block of the FlyDSL-vs-torch output comparison
tot = {}
for M in MS:
    for b, nm, cnt in NAMES:
        t = F.T[f'blk.{b}.{nm}.weight']; K = int(t.shape[0]); R, _ = t.rows()
        dense = R * K * 2 + M * R * 2 if set(ARMS) & {'torch', 'flydsl'} else 0          # (fp16 weights; torch's fp16 output)
        fly = (M * R * 4 + M * K + R * K + M * R * 4 + 3 * min(M, CMP_ROWS) * R * 4) if 'flydsl' in ARMS else 0   # (fp32 output; int8
        # operands, int32 output; the output comparison's row-block temporaries)
        need = (t.n_bytes + M * K * 2 + M * R * 4 + dense + fly) / 2 ** 20 + 64
        capacity.check([0], [need], f'q35_prefill_gemm {nm} M {M} ({need:.0f} MiB)')
        W = Dense35Own._w(fake, f'blk.{b}.{nm}.weight')
        x16 = torch.empty(M, K, dtype=torch.float16, device=dev).normal_(0.0, 0.1); y = torch.zeros(M, R, device=dev)   # (in place: no fp32 temporaries)
        fl = 2.0 * M * R * K; row = f'M {M:5d} blk.{b}.{nm:12s} {t.type:5s} [{R:6d} x {K:6d}]'
        res = {}
        Wd = None
        if set(ARMS) & {'torch', 'flydsl'}:
            Wd = torch.empty(R, K, dtype=torch.float16, device=dev).normal_(0.0, 0.02)   # (in place: no fp32 temporaries)
        for arm in ARMS:
          try:
            if arm == 'own':
                res['own'] = bench(lambda: ops.mm(W, x16, K, y, R, n=M))
            elif arm == 'ab':   # Q4_K / Q6_K: the wide tile vs the narrow one, interleaved rep by rep in this process, outputs bitwise
                if t.type not in ('Q4_K', 'Q6_K'): continue
                flag_attr = 'q4k_narrow' if t.type == 'Q4_K' else 'q6k_narrow'
                y2 = torch.zeros(M, R, device=dev)
                setattr(ops, flag_attr, False); ops.mm(W, x16, K, y, R, n=M)
                setattr(ops, flag_attr, True); ops.mm(W, x16, K, y2, R, n=M); torch.cuda.synchronize(dev)
                same = torch.equal(y.view(torch.int32), y2.view(torch.int32))
                tw, tn = [], []
                for rep in range(A.reps):
                    for flag, ts in ((False, tw), (True, tn)) if rep % 2 == 0 else ((True, tn), (False, tw)):
                        setattr(ops, flag_attr, flag)
                        for _ in range(3): ops.mm(W, x16, K, y, R, n=M)
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        for _ in range(A.iters): ops.mm(W, x16, K, y, R, n=M)
                        e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3 / A.iters)
                setattr(ops, flag_attr, False)
                res['wide'] = float(np.median(tw)); res['narrow'] = float(np.median(tn))
                print(f'  {"BITWISE" if same else "DIFFER"} wide vs narrow at [{R} x {K}] M {M}'
                      + ('' if same else f' ({int((y.view(torch.int32) != y2.view(torch.int32)).sum())} values)'), flush=True)
                if not same: AB_DIFF.append(row)
            elif arm == 'torch':
                yh = torch.empty(M, R, device=dev, dtype=torch.float16)
                res['torch'] = bench(lambda: torch.matmul(x16, Wd.t(), out=yh))
            elif arm == 'flydsl':
                yf = torch.empty(M, R, device=dev, dtype=torch.float32)
                res['fly_f16'] = bench_graph(lambda: fly_f16(yf, x16, Wd, in_dtype='f16', out_dtype='f32'))
                if 'torch' in res:   # (the output too: FlyDSL's fp16 GEMM against torch's on the same operands)
                    torch.matmul(x16, Wd.t(), out=yh); fly_f16(yf, x16, Wd, in_dtype='f16', out_dtype='f32'); torch.cuda.synchronize(dev)
                    dmax = amax = 0.0
                    for r0 in range(0, M, CMP_ROWS):   # (in row blocks: the comparison's temporaries stay CMP_ROWS x R x 4 B x 3)
                        a_ = yh[r0:r0 + CMP_ROWS].float(); dmax = max(dmax, float((yf[r0:r0 + CMP_ROWS] - a_).abs().max())); amax = max(amax, float(a_.abs().max()))
                        del a_
                    row += f' [fly_f16 vs torch: max rel {dmax / amax:.1e}]'
                a8 = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8); w8 = torch.randint(-127, 128, (R, K), device=dev, dtype=torch.int8)
                yi = torch.empty(M, R, device=dev, dtype=torch.int32)
                res['fly_i8'] = bench_graph(lambda: fly_i8(yi, a8, w8))
                del a8, w8, yi, yf
          except Exception as ex:   # (a yardstick that fails leaves the other arms' rows)
            res[arm + ' NOT RUN'] = float('nan'); print(f'  {arm}: NOT RUN ({type(ex).__name__}: {str(ex)[:200]})', flush=True)
        print(row + ' | ' + ' | '.join(f'{k} {v:8.1f} us {fl / v / 1e6:6.1f} TF' for k, v in res.items() if v == v), flush=True)
        for k, v in res.items():
            if v == v: tot[(M, k)] = tot.get((M, k), 0.0) + v * cnt
        del W, x16, y, Wd; torch.cuda.empty_cache()
for (M, k), v in sorted(tot.items()):
    print(f'per chunk M {M}: {k:8s} {v / 1e3:8.2f} ms of GEMMs ({M / (v / 1e6):8.0f} tok/s if GEMMs were all)', flush=True)
print('RESULT FAIL (wide vs narrow differ: ' + '; '.join(AB_DIFF) + ')' if AB_DIFF else 'RESULT PASS (ran to the end; a measurement, not a gate)')
