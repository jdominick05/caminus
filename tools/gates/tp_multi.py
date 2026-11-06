"""The T == one-token gate: a T-token verify pass gives logits bitwise equal to T one-token steps (teacher forced over the
first NPOS positions of a token file), replicas checked after both runs.
  guard.sh LOG SECS python -I tools/gates/tp_multi.py T [NPOS=24] [--model coder|reap|PATH] [--tokens FILE]
                                                         [--max-ctx N] [--qsa 0|1] [--tape dual|always]
One engine with ntok=1 steps through the positions (the reference), then one with ntok=T runs verify passes at 0, T, 2T, ..
(commit 0 first, then T). Equality is bitwise (the bytes of each logits row; the old scripts used torch.equal, which counts
-0 == 0 and NaN != NaN). Prints the old summary line, then RESULT PASS / FAIL (exit 0 / 1): PASS needs every row equal,
replicas equal after both runs. The pass ms is the median over the passes after the first (a timing only: it is printed
with vm.compaction_proactiveness).
Ported from tmp/m1/s3/tp_multi_coder.py (trunk) and tmp/qsa/sel/tp_multi_qsa.py (RD_MAXCTX / RD_QSA: --max-ctx / --qsa
here; K3_QSA_TAPE or --tape picks the QSA tape) with the same engine calls. Current expectations: $ARTIFACTS/refs/MANIFEST.md."""
import gc, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

a = sys.argv[1:]
model = C.model_path(C.opt(a, '--model', 'coder')); tok_path = C.opt(a, '--tokens', C.TOKENS)
kw = C.engine_kwargs(C.opt(a, '--max-ctx', None, int), C.opt(a, '--qsa', None), C.opt(a, '--tape', None))
if not 1 <= len(a) <= 2: print(__doc__); sys.exit(2)
T = int(a[0]); NPOS = int(a[1]) if len(a) > 1 else 24
torch = C.setup_torch()
import numpy as np
from caminus.flashnext_tp import FlashNextTP
toks = C.tokens(tok_path)[:NPOS]
print(f'model {model}; tokens {tok_path}; T {T}; NPOS {NPOS}; engine kwargs {kw}; K3_QSA_TAPE {os.environ.get("K3_QSA_TAPE")}; '
      f'K3_PER_WGP {os.environ.get("K3_PER_WGP")}; compaction_proactiveness {C.proactiveness()}', flush=True)
m = FlashNextTP(model, ntok=1, **kw); ref = []
print(f'qsa {m.qsa} max_ctx {m.max_ctx} modules {[e.module for e in m.engines]}', flush=True)
for i in range(NPOS): ref.append(m.step(toks, i)[None].cpu())
ref = torch.cat(ref); rep1 = m.check_replicas(); print('one-token reference done; replicas', rep1)
del m; gc.collect(); torch.cuda.empty_cache()
m = FlashNextTP(model, ntok=T, **kw); out = []; ts = []
print(f'T={T} engine: modules {[e.module for e in m.engines]}', flush=True)
for i0 in range(0, NPOS - NPOS % T, T):
    torch.cuda.synchronize(0); t0 = time.perf_counter()
    out.append(m.verify(toks, i0, 0 if i0 == 0 else T).cpu()); ts.append(time.perf_counter() - t0)
out = torch.cat(out); n = out.shape[0]
same = [bool(torch.equal(out[i].contiguous().view(torch.uint8), ref[i].contiguous().view(torch.uint8))) for i in range(n)]
rep = m.check_replicas(); pass_ms = np.median(ts[1:]) * 1e3 if len(ts) > 1 else float('nan')
print(f'T={T}: {sum(same)}/{n} positions bitwise equal; max |diff| {(out - ref[:n]).abs().max().item():.3e}; replicas {rep}; '
      f'pass {pass_ms:.2f} ms')
ok = n == NPOS - NPOS % T and n > 0 and all(same) and all(rep) and all(rep1)
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
