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
Run it on the XT alone first.
--engine (a MoE file: qwen3moe, Qwen3-30B-A3B; qwen35moe, Qwen3.6-35B-A3B): every GEMM of one real chunk. The engine and its
batched prefill load on the card, M tokens of --tokens REF.npz's 'seq' (a reference's public text: real routing) prefill as
one chunk, and every GEMM launch of it is recorded (the prefill's Ops.mm / mm_q2_0 / mm_gu calls; the expert work items copied as
they were). Then per (format, R x K, dense | experts) group: ours = the group's recorded launches replayed; the yardsticks at
the same shape: dense rows at M; experts at M = the chunk's (token, slot) pairs (one dense GEMM: grouping free) and per
expert at its own count (the middle MoE layer's routing, padded to FlyDSL's 32-row tile, the time scaled by the group's
launch count). TFLOPS of experts = 2 pairs R K / t. 'per chunk' = the sum over the recorded launches (the formats as the
file mixes them; the last layer's attention output and FFN are skipped by the prefill). Also the routing: pairs per
expert, work items, their fill. --ref-module NAME (an object in the kernel dir, e.g. a copy of the base build's
q3m_prefill.hsaco as q3m_prefill_base.hsaco): ours on both objects, interleaved rep by rep, every output compared bitwise.
  tools/guard.sh LOG SECS python -I benchmarks/q35_prefill_gemm.py GGUF --engine --tokens REF.npz [--M 2048]
      [--arms own,torch,flydsl] [--ref-module NAME] [--set A=1,..] [--ref-set A=0,..] [--only REGEX] [--tokens-from GGUF]
--set / --ref-set set Ops attributes (the A / B flags: q3m_ops.Ops wide, gu) on the own / ref arm; a ref arm
without the fused gate | up replays a fused launch as the gate GEMM, the up GEMM and q3p_swiglu.
FlyDSL picks its tiles by its heuristic, or by a search with FLYDSL_AUTOTUNE=1 (FLYDSL_AUTOTUNE_CACHE_DIR keeps them)."""
import argparse, collections, os, re, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np, torch
from caminus import ggufio
from caminus.device import capacity
from caminus.engine.q35_dense import Dense35Own
from caminus.prefill import q35own, q3m_ops

ap = argparse.ArgumentParser()
ap.add_argument('gguf', nargs='?', default=os.path.expanduser(os.environ.get('SWIFT', '')))
ap.add_argument('--M', default='512,2048'); ap.add_argument('--arms', default='own,torch')
ap.add_argument('--module', default=None, help='the prefill object of the own arm (default the file\'s: q35_prefill / q35_prefill_tcap); a measurement build such as q35_prefill_abl1 times what is left of the tile without one of its parts (kernels/q3m_prefill.hip Q3P_ABL; its outputs are wrong by design)')
ap.add_argument('--flydsl-src', default=None); ap.add_argument('--reps', type=int, default=7); ap.add_argument('--iters', type=int, default=10)
ap.add_argument('--engine', action='store_true', help='a MoE file: every GEMM of one real chunk (above)')
ap.add_argument('--tokens', default=None, help='--engine: a reference .npz whose seq is the chunk\'s text')
ap.add_argument('--tokens-from', default=None, help='--engine: the GGUF whose vocabulary --tokens is in (its text re-encoded for this file)')
ap.add_argument('--ref-module', default=None, help='--engine: ours on this object too, interleaved, outputs bitwise')
ap.add_argument('--only', default=None, help='--engine: the groups whose key matches this regex')
ap.add_argument('--set', default=None, help='--engine: NAME=INT,... attributes of the prefill\'s Ops before the chunk (e.g. wide=0,gu=0)')
ap.add_argument('--ref-set', default=None, help='--engine: the same for the ref arm (its module: --ref-module, default the own one)')
A = ap.parse_args()
if not A.gguf: sys.exit('q35_prefill_gemm: give the GGUF path (or export SWIFT)')
ARMS = A.arms.split(','); MS = [int(v) for v in A.M.split(',')]
dev = torch.device('cuda:0')
fly_f16 = fly_i8 = None
if 'flydsl' in ARMS:
    if not A.flydsl_src: sys.exit('q35_prefill_gemm: --arms flydsl needs --flydsl-src (a FlyDSL source checkout matching the installed package)')
    sys.path.insert(0, os.path.abspath(A.flydsl_src))
    from kernels.gemm.rdna3_f16_gemm_autotune import rdna3_gemm_autotuned as fly_f16
    from kernels.gemm.rdna3_int8_gemm_autotune import rdna3_int8_gemm_autotuned as fly_i8
if not A.engine:
    F = ggufio.GGUFFile(A.gguf)
    blocks = [int(n.split('.')[1]) for n in F.T if n.startswith('blk.') and n.endswith('.attn_q.weight')]
    ab = blocks[0]; gb = 0 if ab != 0 else 1
    NAMES = [(gb, 'attn_qkv', 48), (gb, 'attn_gate', 48), (gb, 'ssm_out', 48), (gb, 'ffn_gate', 48), (gb, 'ffn_up', 48), (gb, 'ffn_down', 48),
             (ab, 'attn_q', 16), (ab, 'attn_k', 16), (ab, 'attn_v', 16), (ab, 'attn_output', 16), (ab, 'ffn_gate', 16), (ab, 'ffn_up', 16), (ab, 'ffn_down', 16)]
    # (Dense35Own._w's host half and its upload: each matrix straight to the card, as the engine's layout)
    fake = types.SimpleNamespace(F=F, tcap=False, module='k3_q35_swift', one_rt=(),
                                 h2d=types.SimpleNamespace(up=lambda b, *_: torch.from_numpy(b).to(dev)))
    fake._w_host = lambda name: Dense35Own._w_host(fake, name)
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


def bench_graph(fn, iters=None):
    """FlyDSL's own timing method (its autotuner's _graph_bench): the launches captured in a HIP graph, then replayed, so
    the host launch path (measured ~30 ms per eager call through its Python JIT dispatch) does not count"""
    for _ in range(3): fn()
    torch.cuda.synchronize(dev)
    g = torch.cuda.CUDAGraph(); iters = A.iters if iters is None else iters
    with torch.cuda.graph(g):
        for _ in range(iters): fn()
    for _ in range(2): g.replay()
    torch.cuda.synchronize(dev); ts = []
    for _ in range(A.reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3 / iters)
    del g
    return float(np.median(ts))


def engine_mode():
    """--engine: every GEMM of one real chunk of a MoE file, recorded and replayed (the docstring)"""
    from caminus.engine import k3
    G = ggufio.GGUFFile(A.gguf); arch = G.arch; G.close()
    if not A.tokens: sys.exit('q35_prefill_gemm --engine: give --tokens REF.npz (a reference\'s seq: real text, real routing)')
    M = MS[0]; seq = [int(v) for v in np.load(A.tokens)['seq']]
    if A.tokens_from:   # (the seq is another vocabulary's: its text, encoded with this file's own tokenizer)
        from caminus.bpe import BPE
        S_, T_ = ggufio.GGUFFile(A.tokens_from), ggufio.GGUFFile(A.gguf)
        seq = BPE(T_.meta).encode(BPE(S_.meta).decode(seq)); S_.close(); T_.close()
    if len(seq) < M: print(f'NOTE --tokens: {len(seq)} tokens < {M}: the text repeats', flush=True)
    toks = (seq * (M // len(seq) + 1))[:M]
    if arch == 'qwen3moe':
        from caminus.engine.q3moe_one import Q3MoEOne
        from caminus.prefill import q3m
        eng = Q3MoEOne(A.gguf, max_ctx=M + 64); pf = q3m.Q3MPrefill(eng, chunk=M)
        names = {L[key][3].data_ptr(): f'blk.{il}.{key}' for il, L in enumerate(eng.L) for key in q3m.MM_KEYS}
    elif arch == 'qwen35moe':
        from caminus.engine.q35_moe import Moe35Own
        from caminus.prefill import q35moe
        eng = Moe35Own(A.gguf, max_ctx=M + 64, vram_extra_mib=q35moe.buffer_need_mib(q35moe.shapes_of(A.gguf), M))
        pf = q35moe.Q35MoEPrefill(eng, chunk=M); names = {}
    else: sys.exit(f'q35_prefill_gemm --engine: architecture {arch!r} (qwen3moe, qwen35moe)')
    O = getattr(pf, 'ops', None) or pf.O; E, k = eng.n_exp, eng.k_used
    print(torch.cuda.get_device_name(dev), os.path.basename(A.gguf), f'| {arch}, a chunk of {M} tokens of {os.path.basename(A.tokens)} '
          f'| arms {ARMS} | own object {O.module}' + (f' vs {A.ref_module}' if A.ref_module else ''), flush=True)

    def wkey(W):   # (format, R, K, name) of a launch's matrix: q3m's tuple or q35's dict
        if isinstance(W, dict): return W['fmt'], int(W['rows']), int(W['K']), W.get('name', '?')
        return W[0], int(W[1]), int(W[2]), names.get(W[3].data_ptr(), '?')
    calls, routing = [], []
    YW = {'mm': (3, None), 'mm_q2_0': (3, 5), 'mm_gu': (4, 6)}   # (the output's and the positional work's argument index)
    def rec(meth):
        f = getattr(O, meth); yi, wi = YW[meth]
        def wrap(*a, **kw):
            W = a[0]; work = kw.get('work', a[wi] if wi is not None and len(a) > wi else None)
            fmt, R, K, nm = wkey(W)
            if work is not None:   # (the items as they are now: the next layer's grouping rewrites them)
                items, nitems, bound, xs, yd = work; work = (items.clone(), nitems.clone(), bound, xs.clone(), yd.clone())
                if 'work' in kw: kw = dict(kw, work=work)
                else: a = a[:wi] + (work,) + a[wi + 1:]
            n = kw.get('n'); gu = meth == 'mm_gu'
            calls.append(dict(meth=meth, a=a, kw=kw, key=(fmt, R, K, 'gate|up' if gu else 'experts' if work is not None else 'dense'),
                              name=nm, flops=(4.0 if gu else 2.0) * (M * k if work is not None else int(n)) * R * K, y=a[yi]))
            return f(*a, **kw)
        setattr(O, meth, wrap)
    for meth in YW:
        if hasattr(O, meth): rec(meth)
    gname = 'group9' if hasattr(O, 'group9') else 'group'
    g0 = getattr(O, gname)
    def grp(ids, npairs, *a, **kw):
        routing.append(ids[:npairs].clone()); return g0(ids, npairs, *a, **kw)
    setattr(O, gname, grp)
    def setattrs(o_, spec):   # (--set / --ref-set: name=value,...; values are ints)
        for kv in filter(None, (spec or '').split(',')):
            nm, v = kv.split('='); setattr(o_, nm, int(v))
            if nm in ('wide', 'gu') and int(v) and getattr(o_, 'wmod', None) is None:
                raise SystemExit(f'--set {nm}=1: {o_.module} was built without the wide tiles (q3m_ops.Ops(wide=None) loads them with q3m_prefill)')
    setattrs(O, A.set)
    eng.reset(); pf.run(toks, 0, 0, M); torch.cuda.synchronize(dev)
    for meth in list(YW) + [gname]:
        if meth in O.__dict__: delattr(O, meth)
    # the routing: pairs per expert, the work items (64 pairs each, a partial one per expert), their fill
    cnt = np.stack([np.bincount(r.cpu().numpy(), minlength=E) for r in routing])   # [MoE layers, E]
    items = np.ceil(cnt / q3m_ops.MM_M).sum(1)
    print(f'routing ({len(routing)} MoE layers, {E} experts, top {k}, {M * k} pairs a layer): pairs per expert median '
          f'{np.median(cnt):.0f}, p10 {np.percentile(cnt, 10):.0f}, p90 {np.percentile(cnt, 90):.0f}, max {cnt.max()}, '
          f'unused {(cnt == 0).mean() * 100:.1f} %; work items per layer {items.mean():.0f} (vs {M * k / q3m_ops.MM_M:.0f} full), '
          f'fill {(M * k / (items * q3m_ops.MM_M)).mean() * 100:.1f} %, partial items {((cnt % q3m_ops.MM_M) > 0).sum(1).mean():.0f} a layer', flush=True)
    mid = cnt[len(cnt) // 2]
    groups = collections.OrderedDict()
    for i, c in enumerate(calls): groups.setdefault(c['key'], []).append(i)
    ref = type(O)(O.dev, module=A.ref_module or O.module) if (A.ref_module or A.ref_set) else None
    if ref is not None: setattrs(ref, A.ref_set)
    mods = [('own', O)] + ([('ref', ref)] if ref is not None else [])
    gu_buf = None
    def replay(ops_, idx):
        nonlocal gu_buf
        for i in idx:
            c = calls[i]
            if c['meth'] == 'mm_gu' and not ops_.gu:   # (an arm without the fused tile: gate, up and swiglu as the prefill runs them)
                Wg, Wu, x16_, ldx_, y16_, ldy_, work_ = c['a']
                if gu_buf is None:
                    need = 2 * M * k * ldy_ * 4 / 2 ** 20 + 16
                    capacity.check([dev.index or 0], [need], f'q35_prefill_gemm: gate / up outputs ({need:.0f} MiB)')
                    gu_buf = torch.empty(2, M * k * ldy_, dtype=torch.float32, device=dev)
                ops_.mm(Wg, x16_, ldx_, gu_buf[0], ldy_, work=work_); ops_.mm(Wu, x16_, ldx_, gu_buf[1], ldy_, work=work_)
                ops_.swiglu(gu_buf[0], gu_buf[1], y16_, M * k * ldy_)
                continue
            getattr(ops_, c['meth'])(*c['a'], **c['kw'])
    tot = collections.defaultdict(float); differ = []
    for key, idx in groups.items():
        fmt, R, K, kind = key; label = f'{fmt:5s} {R:5d} x {K:5d} {kind:7s}'
        mult = 2 if kind == 'gate|up' else 1   # (a fused launch is two GEMMs)
        if A.only and not re.search(A.only, label): continue
        fl = sum(calls[i]['flops'] for i in idx); res = {}
        if 'own' in ARMS:
            if ref is not None:   # (bitwise: every launch of the group on both objects, its output compared)
                nd = 0
                for i in idx:
                    y = calls[i]['y']; replay(ref, [i]); torch.cuda.synchronize(dev); y0 = y.clone()
                    replay(O, [i]); torch.cuda.synchronize(dev); nd += int((y.view(torch.int32) != y0.view(torch.int32)).sum()); del y0
                print(f'  {"BITWISE" if nd == 0 else "DIFFER"} ref ({ref.module} {A.ref_set or ""}) vs own ({O.module} {A.set or ""}): {label} ({len(idx)} launches'
                      + (f', {nd} values differ)' if nd else ')'), flush=True)
                if nd: differ.append(label)
            ts = {nm: [] for nm, _ in mods}
            for _ in range(2): [replay(o_, idx) for _, o_ in mods]
            for rep in range(A.reps):
                for nm, o_ in (mods if rep % 2 == 0 else mods[::-1]):
                    torch.cuda.synchronize(dev)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for _ in range(A.iters): replay(o_, idx)
                    e1.record(); e1.synchronize(); ts[nm].append(e0.elapsed_time(e1) * 1e3 / A.iters)
            for nm, _ in mods: res[nm] = float(np.median(ts[nm]))
        n_ = M if kind == 'dense' else M * k
        ys = yard_arms(n_, R, K, mult * len(idx))
        if kind != 'dense' and set(ARMS) & {'torch', 'flydsl'}:
            for nm, v in yard_experts(mid, R, K).items(): ys[nm] = v * mult * len(idx)
        res.update(ys)
        print(f'{label} {len(idx):3d} launches ({calls[idx[0]]["name"]} ..) | ' + ' | '.join(
            f'{nm} {v / 1e3:8.2f} ms {fl / v / 1e6:5.1f} TF' for nm, v in res.items() if v == v), flush=True)
        for nm, v in res.items():
            if v == v: tot[nm] += v
    print(f'per chunk of {M} ({len(calls)} GEMM launches' + (f', groups matching {A.only!r}' if A.only else '') + '): '
          + ' | '.join(f'{nm} {v / 1e3:.1f} ms' for nm, v in tot.items()), flush=True)
    print('RESULT FAIL (outputs differ: ' + '; '.join(differ) + ')' if differ else 'RESULT PASS (ran to the end; a measurement, not a gate)')


def yard_arms(M, R, K, count):
    """the dense yardsticks at (M, R, K), each one launch's time x count: torch fp16, FlyDSL fp16 and int8 (random operands)"""
    res = {}
    if not set(ARMS) & {'torch', 'flydsl'}: return res
    need = (R * K * 2 + M * K * 2 + M * R * 2 + (M * R * 4 * 2 + M * K + R * K if 'flydsl' in ARMS else 0)) / 2 ** 20 + 64
    capacity.check([dev.index or 0], [need], f'q35_prefill_gemm yardsticks [{M} x {R} x {K}] ({need:.0f} MiB)')
    x16 = torch.empty(M, K, dtype=torch.float16, device=dev).normal_(0.0, 0.1)
    Wd = torch.empty(R, K, dtype=torch.float16, device=dev).normal_(0.0, 0.02)
    try:
        if 'torch' in ARMS:
            yh = torch.empty(M, R, device=dev, dtype=torch.float16)
            res['torch'] = bench(lambda: torch.matmul(x16, Wd.t(), out=yh)) * count; del yh
        if 'flydsl' in ARMS:
            yf = torch.empty(M, R, device=dev, dtype=torch.float32)
            res['fly_f16'] = bench_graph(lambda: fly_f16(yf, x16, Wd, in_dtype='f16', out_dtype='f32')) * count; del yf
            a8 = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8); w8 = torch.randint(-127, 128, (R, K), device=dev, dtype=torch.int8)
            yi = torch.empty(M, R, device=dev, dtype=torch.int32)
            res['fly_i8'] = bench_graph(lambda: fly_i8(yi, a8, w8)) * count; del a8, w8, yi
    except Exception as ex:
        print(f'  yardsticks [{M} x {R} x {K}]: NOT RUN ({type(ex).__name__}: {str(ex)[:200]})', flush=True)
    del x16, Wd; torch.cuda.empty_cache()
    return res


def yard_experts(cnt, R, K):
    """the dense yardsticks per expert at its own pair count (one layer's routing cnt [E]; FlyDSL's M padded to 32 rows, its
    smallest tile): one layer's E GEMMs, replayed from a graph (torch's too: E small launches)"""
    res = {}; cs = [int(c) for c in cnt if c > 0]; mx = -(-max(cs) // 32) * 32
    need = (R * K * 2 + mx * K * 2 + mx * R * 2 + mx * R * 4) / 2 ** 20 + 64
    capacity.check([dev.index or 0], [need], f'q35_prefill_gemm per-expert yardsticks ({need:.0f} MiB)')
    x16 = torch.empty(mx, K, dtype=torch.float16, device=dev).normal_(0.0, 0.1)
    Wd = torch.empty(R, K, dtype=torch.float16, device=dev).normal_(0.0, 0.02)
    try:
        if 'torch' in ARMS:
            yh = torch.empty(mx, R, device=dev, dtype=torch.float16)
            def tl():
                for c in cs: torch.matmul(x16[:c], Wd.t(), out=yh[:c])
            res['torch/exp'] = bench_graph(tl, iters=1); del yh
        if 'flydsl' in ARMS:
            yf = torch.empty(mx, R, device=dev, dtype=torch.float32)
            def fl():
                for c in cs: c32 = -(-c // 32) * 32; fly_f16(yf[:c32], x16[:c32], Wd, in_dtype='f16', out_dtype='f32')
            res['fly/exp'] = bench_graph(fl, iters=1); del yf
    except Exception as ex:
        print(f'  per-expert yardsticks [{R} x {K}]: NOT RUN ({type(ex).__name__}: {str(ex)[:200]})', flush=True)
    del x16, Wd; torch.cuda.empty_cache()
    return res


if A.engine:
    engine_mode(); sys.exit(0)


AB_DIFF = []
CMP_ROWS = 256   # rows per block of the FlyDSL-vs-torch output comparison
tot = {}
for M in MS:
    for b, nm, cnt in NAMES:
        t = F.T[f'blk.{b}.{nm}.weight']; K = int(t.shape[0]); R, _ = t.rows()
        dense = R * K * 2 + M * R * 2 if set(ARMS) & {'torch', 'flydsl'} else 0          # (fp16 weights; torch's fp16 output)
        fly = (M * R * 4 + M * K + R * K + M * R * 4 + 3 * min(M, CMP_ROWS) * R * 4) if 'flydsl' in ARMS else 0   # (fp32 output; int8
        # operands, int32 output; the output comparison's row-block temporaries)
        need = (t.n_bytes + M * K * 2 + M * R * 4 * (2 if 'ab' in ARMS else 1) + dense + fly) / 2 ** 20 + 64   # (ab: a second output)
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
            elif arm == 'ab':   # Q4_K / Q6_K / Q8_0: the wide tile vs the narrow one, interleaved rep by rep in this process, outputs bitwise
                if t.type not in ('Q4_K', 'Q6_K', 'Q8_0'): continue
                pre = {'Q4_K': 'q4k', 'Q6_K': 'q6k', 'Q8_0': 'q8'}[t.type]
                y2 = torch.zeros(M, R, device=dev)
                routed = 'wide' if -(-M // 64) * -(-R // q35own.MMW_N) >= q35own.MMW_MIN_PER_CU * ops.cus else 'narrow'
                setattr(ops, pre + '_force_wide', True)   # (the A/B compares the two kernels on every shape; `routed` is what Ops.mm serves)
                setattr(ops, pre + '_narrow', False); ops.mm(W, x16, K, y, R, n=M)
                setattr(ops, pre + '_narrow', True); ops.mm(W, x16, K, y2, R, n=M); torch.cuda.synchronize(dev)
                same = torch.equal(y.view(torch.int32), y2.view(torch.int32))
                tw, tn = [], []
                for rep in range(A.reps):
                    for flag, ts in ((False, tw), (True, tn)) if rep % 2 == 0 else ((True, tn), (False, tw)):
                        setattr(ops, pre + '_narrow', flag)
                        for _ in range(3): ops.mm(W, x16, K, y, R, n=M)
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        for _ in range(A.iters): ops.mm(W, x16, K, y, R, n=M)
                        e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1e3 / A.iters)
                setattr(ops, pre + '_narrow', False); setattr(ops, pre + '_force_wide', False)
                res['wide'] = float(np.median(tw)); res['narrow'] = float(np.median(tn))
                print(f'  {"BITWISE" if same else "DIFFER"} wide vs narrow at [{R} x {K}] M {M} (served: {routed})'
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
