"""Per-shape GEMM table for the S1 step: rocBLAS (F.linear), hipBLASLt (torch's preferred BLAS switched to it) and our
WMMA GEMM per tile, on the shapes caminus.linear issues (forward x . w^T and dgrad dy . w as C = A . B^T), and the
choice per shape (ours only where it beats rocBLAS: hipBLASLt has other bits). Run on an idle card.
  python tools/bench_gemm.py [OUT.json]    (default build/gemm_table.json, read by caminus.linear)"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python'))
import torch
import torch.nn.functional as F
from caminus import gemm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'build', 'gemm_table.json')
T = 8192   # tokens per microbatch (batch 4 x 2048)
# (M, N, K) as C[M, N] = A[M, K] . B[N, K]^T; forward of a [N_out, K_in] layer is (T, N_out, K_in), its dgrad (T, K_in, N_out)
layers = {'gdn block (SHARE)': (7680, 896), 'attn block (SHARE)': (5248, 896), 'qkvg 896': (896, 896), 'mlp.gu': (4096, 896),
          'mlp.down': (896, 2048), 'attn kv': (256, 896)}
shapes = {}
for name, (n, k) in layers.items():
    shapes[(T, n, k)] = f'{name} fwd'; shapes.setdefault((T, k, n), f'{name} dgrad')
for M in (1024, 1536, 2048):   # slices (rows per section vary; these bracket them) at width 768
    shapes.setdefault((M, 1536, 896), 'slice gu fwd'); shapes.setdefault((M, 896, 1536), 'slice gu dgrad')
    shapes.setdefault((M, 896, 768), 'slice down fwd'); shapes.setdefault((M, 768, 896), 'slice down dgrad')


def bench(fn, n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n


rows = []
print(f'{"shape (M x N x K)":>22s} {"what":22s} | rocBLAS  hipBLASLt | ours: ' + '  '.join(f'{a}x{b}' for a, b in gemm.TILES) + '  (TFLOPS)')
for (M, N, K), what in shapes.items():
    a = torch.randn(M, K, device='cuda').bfloat16(); b = torch.randn(N, K, device='cuda').bfloat16()
    tf = lambda t: 2 * M * N * K / t / 1e12
    r = tf(bench(lambda: F.linear(a, b)))
    torch.backends.cuda.preferred_blas_library('cublaslt'); lt = tf(bench(lambda: F.linear(a, b)))
    torch.backends.cuda.preferred_blas_library('cublas')
    ours = {f'{t[0]}x{t[1]}': tf(bench(lambda: gemm.gemm_tn(a, b, tile=t))) for t in gemm.TILES if gemm.supported(M, N, K, t)}
    best = max(ours, key=ours.get) if ours else None
    use = best if best and ours[best] > r * 1.02 else 'rocblas'
    rows.append({'M': M, 'N': N, 'K': K, 'what': what, 'rocblas': r, 'hipblaslt': lt, 'ours': ours, 'use': use})
    print(f'{f"{M} x {N} x {K}":>22s} {what:22s} | {r:7.1f}  {lt:9.1f} | ' + '  '.join(f'{ours.get(f"{a}x{b}", 0):7.1f}' for a, b in gemm.TILES) + f'  -> {use}')
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({'device': torch.cuda.get_device_name(), 'rows': rows}, open(out, 'w'), indent=1)
print(f'wrote {out}')
