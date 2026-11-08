"""Gates of the batched prefill's own dense quantized GEMMs (caminus.prefill.dense, kernels/prefill_dense.hip): the Q6_K dense matrices
against the vendored MMQ's mul_mat_q they replace (caminus.prefill.mmq.MMQ.mm), on the same quantized activations.

  python -I tests/prefill/test_prefill_dense.py CODER STAGE... [--small] [--vds 0,1,2,3] [--json OUT] [--reps N]

Shapes: every Q6_K dense matrix ChunkStack runs, at each TP card's shard shape (FlashNextTP's split: gdn_keys 10, one kv head
per card, moe_blocks 12): attn_qkv 6400 / 3840 rows, attn_gate 3840 / 2304, ssm_out K 3840 / 2304 (the K split), attn_q 6144,
attn_k / attn_v 256, attn_output K 3072, the shared expert's gate / up 384 / 256 rows; real Q6_K rows of the Coder (the first
rows / first K blocks of a layer's tensor). Activations: the real FFN-half `mixed` rows ($ARTIFACTS/impl-moe-gemm/real) for
K = 2560, seeded normal rows for the other K, with edge columns (all zero, one huge value, tiny values).

Stages (one card, cuda:0: HIP_VISIBLE_DEVICES picks it; the XT first for a changed kernel):
  cpu    CPU only, nothing launched: dense.Dense's host refusals (short views of w / y / dst, overlaps, misaligned buffers,
         rows not a multiple of the row block, K not whole Q6_K blocks, no kernel for (VD, J)) through a launch recorder, with
         the positive controls on the same buffers
  g1     every shape, n = 1 / 3 / 17 / 777 / 1024 / 2048 (--small: 1 / 3 on two shapes, a changed kernel's first run), every VD
         of --vds and every J (n >= 777: llama.cpp's J and J 64 / 128 only): dst == mmq.mm's bitwise, under the run1 canary
         method: w, y and dst each in one allocation between 1 MiB sentinel guards, dst's whole view (rows x ncols + a
         sentinel tail) prefilled with the sentinel; after every launch the guards and the tail hold the sentinel, w and y
         equal their copies, and dst == MMQ's; a launch runs only after every check of the one before passed (the stage
         stops at the first byte outside its range)
  g2b    2048 tokens as one launch and split (1024 + 1024, 777 + 1271, 1 + 2047, 512 + 788 + 748 and the micro-chunks of m 3 / 4),
         each piece quantized and multiplied on its own: every output == the one-launch run (== MMQ), every VD
  speed  per shape at n 1024 / 2048: MMQ (mmq.mm, llama.cpp's J) vs ours per VD (dense.choose_J, and every J for the default
         VD), interleaved rounds, medians of --reps launches per round, 3 rounds; the longest single launch; the 48 layers'
         dense Q6_K ms per chunk (the Coder's per-layer counts) for chunk 2048 as 1 x 2048 and 2 x 1024; for reference the
         bf16 WMMA GEMM (gemm.hip, round_bf16) and torch's bf16 F.linear at the same shapes (no dequantization counted)
Prints PASS / FAIL per check and RESULT PASS / FAIL; exit 0 = PASS."""
import argparse, json, os, statistics, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'python')); sys.path.insert(0, HERE)
FAILS, NUMBERS = [], {}
from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
ART = devpaths.artifacts()
REAL = os.path.join(ART, 'impl-moe-gemm', 'real')
SENT = -5913691                  # the sentinel word 0xFFA5C3A5: a NaN as fp32
GUARD = 1 << 20                  # bytes of sentinel before and after every guarded allocation
# (name, gguf tensor, layer with a Q6_K copy, rows / K per card): FlashNextTP's shard shapes
def _tp_split(default=(10, 12)):
    """--tp-split GK,MB (taken out of sys.argv before argparse): the TP split whose card shapes are run (default the
    measured XTX + XT split 10 / 12; 8,10: the even split of equal pairs)"""
    if '--tp-split' not in sys.argv: return default
    i = sys.argv.index('--tp-split'); v = tuple(int(x) for x in sys.argv[i + 1].split(',')); del sys.argv[i:i + 2]
    print(f'split: gdn_keys {v[0]}, moe_blocks {v[1]}', flush=True)
    return v


GK, MB = _tp_split()
_kc, _mb = (GK, 16 - GK), (MB, 20 - MB)          # GDN key heads and 32-wide intermediate blocks per card
SHAPES = [('qkv', 'attn_qkv', 0, (640 * _kc[0], 2560), (640 * _kc[1], 2560)), ('z', 'attn_gate', 1, (384 * _kc[0], 2560), (384 * _kc[1], 2560)),
          ('ssm_out', 'ssm_out', 0, (2560, 384 * _kc[0]), (2560, 384 * _kc[1])), ('q', 'attn_q', 31, (6144, 2560), (6144, 2560)),
          ('k', 'attn_k', 3, (256, 2560), (256, 2560)), ('v', 'attn_v', 3, (256, 2560), (256, 2560)),
          ('o', 'attn_output', 7, (2560, 3072), (2560, 3072)), ('sg', 'ffn_gate_shexp', 4, (32 * _mb[0], 2560), (32 * _mb[1], 2560)),
          ('su', 'ffn_up_shexp', 2, (32 * _mb[0], 2560), (32 * _mb[1], 2560))]
COUNT = dict(qkv=22, z=14, ssm_out=29, q=5, k=11, v=10, o=11, sg=13, su=13)   # Q6_K layers of each (the Coder's GGUF)


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)
    return ok


def comp():
    try: return open('/proc/sys/vm/compaction_proactiveness').read().strip()
    except OSError: return '?'


def memavail():
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable'): return int(l.split()[1]) // 1024
    return -1


# ---- cpu ----------------------------------------------------------------------------------------------------------------
def stage_cpu():
    import torch
    from caminus.prefill import dense, mmq
    calls = []
    class Rec:
        def __init__(self, name): self.name = name
        def __call__(self, grid, block, *a, **kw): calls.append((self.name, grid, block))
    O = dense.Dense.__new__(dense.Dense); O.dev = torch.device('cpu'); O.cfg, O.k = {}, {}
    for vd in dense.VDS:
        for J in dense.JS_VD[vd]:
            O.cfg[('Q6_K', vd, J)] = dict(nthreads=128, I=64, nbytes=40000); O.k[('Q6_K', vd, J)] = Rec(f'v{vd} {J}')
    O._launch = lambda k, *a, **kw: k(*a, **kw)
    z = lambda n, dt=torch.uint8: torch.zeros(n, dtype=dt)
    R, K, n = 384, 2560, 100
    wb = R * mmq.row_bytes('Q6_K', K); w = z(wb); y = z(mmq.y_bytes(K, n)); d = z(R * n, torch.float32)
    def mmc(**over):
        b = dict(w=w, y=y, d=d, R=R, K=K, n=n, J=None, vd=None); b.update(over)
        return lambda: O.mm('Q6_K', b['w'], b['R'], b['K'], b['y'], b['n'], b['d'], J=b['J'], vd=b['vd'])
    def expect(fn, refused, what):
        n0 = len(calls)
        try: fn(); got = False
        except ValueError: got = True
        check(got == refused and (len(calls) == n0) == refused, f'cpu: {what} {"refused, nothing launched" if refused else "accepted, launched"}')
    expect(mmc(), False, 'mm on whole buffers')
    check(calls[-1][1] == (R // 64, (n + 64 - 1) // 64, 1) and calls[-1][2] == (32, 4), f'cpu: the grid / block {calls[-1][1:]} (J 64 at {R} rows)')
    for vd in dense.VDS:
        for J in dense.JS: expect(mmc(vd=vd, J=J), False, f'mm VD {vd} J {J}')
    expect(mmc(J=24), True, 'mm: a column tile no kernel has (24)')
    expect(mmc(vd=4), True, 'mm: VD 4')
    expect(mmc(w=w[:wb - 1]), True, 'mm: a short view of w')
    expect(mmc(y=y[:mmq.y_bytes(K, n) - 1]), True, 'mm: a short view of y')
    expect(mmc(d=d[:R * n - 1]), True, 'mm: a short view of dst')
    expect(mmc(R=352), True, 'mm: rows 352 (not a multiple of 64)')
    expect(mmc(K=2304 + 128), True, 'mm: K 2432 (not whole Q6_K blocks)')
    expect(mmc(K=0), True, 'mm: K 0')
    expect(mmc(n=0), True, 'mm: ncols 0')
    expect(mmc(w=w.view(torch.int8)), True, 'mm: w not uint8')
    expect(mmc(d=d.to(torch.float16)), True, 'mm: dst not float32')
    expect(mmc(w=z(wb + 2)[1:wb + 1]), True, 'mm: w at an odd address')
    expect(mmc(y=z(mmq.y_bytes(K, n) + 16)[4:]), True, 'mm: y not 16-byte aligned')
    blk = z(4 * R * n + wb); off = wb // 2 // 4 * 4
    expect(mmc(w=blk[:wb], d=blk[off:off + 4 * R * n].view(torch.float32)), True, 'mm: dst overlapping w')
    yd = z(mmq.y_bytes(K, n) + 4 * R * n)
    expect(mmc(y=yd[:mmq.y_bytes(K, n)], d=yd[256:256 + 4 * R * n].view(torch.float32)), True, 'mm: dst overlapping y')
    expect(mmc(d=d.view(R, n).t()), True, 'mm: a non-contiguous dst')
    J = dense.choose_J(2048); check(J == 128, f'cpu: choose_J(2048) == 128 ({J})')
    check(dense.choose_J(2048, 384) == 64 and dense.choose_J(2048, 6400) == 128 and dense.choose_J(17, 256) == 32, 'cpu: choose_J with rows (J 64 below 1024 rows)')
    check([dense.choose_J(c) for c in (1, 16, 17, 33, 100, 777)] == [16, 16, 32, 48, 128, 128], f'cpu: choose_J {[dense.choose_J(c) for c in (1, 16, 17, 33, 100, 777)]}')


# ---- device helpers -----------------------------------------------------------------------------------------------------
class Guarded:
    """one allocation [GUARD | n bytes | GUARD], all the sentinel word; .v the n-byte view"""
    def __init__(self, torch, dev, n, align=256):
        n = (int(n) + 3) // 4 * 4; self.n = n
        self.t = torch.empty(GUARD + n + GUARD, dtype=torch.uint8, device=dev)
        self.t.view(torch.int32).fill_(SENT)
        self.v = self.t[GUARD:GUARD + n]
    def guards_ok(self, torch):
        g = self.t.view(torch.int32)
        return bool((g[:GUARD // 4] == SENT).all().item()) and bool((g[(GUARD + self.n) // 4:] == SENT).all().item())


def load_weight(torch, dev, rd, name, il, R, K):
    from caminus.prefill import mmq
    t = {x.name: x for x in rd.tensors}[f'blk.{il}.{name}.weight']
    assert t.tensor_type.name == 'Q6_K', (name, il, t.tensor_type.name)
    Kf, Rf = int(t.shape[0]), int(t.shape[1]); rb = mmq.row_bytes('Q6_K', Kf)
    raw = np.asarray(t.data).reshape(-1).view(np.uint8).reshape(Rf, Kf // 256, 210)
    return np.ascontiguousarray(raw[:R, :K // 256]).reshape(-1)


def activations(K, n, seed, edges=True):
    """[n][K] fp32: the real mixed rows for K 2560, else seeded normal rows; edges: columns 0 / 1 / 2 (when present) all zero,
    one huge value, tiny values"""
    if K == 2560 and os.path.exists(os.path.join(REAL, 'd8192_L1_mixed.npy')):
        x = np.load(os.path.join(REAL, 'd8192_L1_mixed.npy'))
        x = np.ascontiguousarray(np.concatenate([x, x[::-1] * 0.5])[:n]).astype(np.float32)
    else:
        x = np.random.default_rng(seed).standard_normal((n, K)).astype(np.float32)
    if n >= 3 and edges:
        x[0] = 0; x[1] = 0.01; x[1, 7] = 3.0e4; x[2] = np.float32(1e-30) * np.sign(x[2] + 0.5)
    return x


class Case:
    """one shape on the device: guarded weights, MMQ's reference outputs per n"""
    def __init__(self, torch, dev, M, rd, op, name, il, R, K):
        from caminus.prefill import mmq
        self.op, self.R, self.K = op, R, K
        raw = load_weight(torch, dev, rd, name, il, R, K)
        self.gw = Guarded(torch, dev, raw.size); self.w = self.gw.v[:raw.size]
        self.w.copy_(torch.from_numpy(raw).to(dev)); self.w0 = self.w.clone()


def run_g1(torch, dev, M, Dn, cases, ns, vds, tag0, js_small):
    from caminus.prefill import mmq, dense
    for cs in cases:
        R, K = cs.R, cs.K
        for n in ns:
            x = torch.from_numpy(activations(K, n, 1000 + n)).to(dev)
            gy = Guarded(torch, dev, mmq.y_bytes(K, n)); y = gy.v[:mmq.y_bytes(K, n)]
            M.quantize(x, K, n, y, M.layout('Q6_K'))
            ref = torch.zeros(R * n, dtype=torch.float32, device=dev); Jr = M.mm('Q6_K', cs.w, R, K, y, n, ref)
            y0 = y.clone(); torch.cuda.synchronize(dev)
            tail = 4096                                                   # dst's view is rows x ncols + a sentinel tail
            gd = Guarded(torch, dev, 4 * R * n + tail); dv = gd.v.view(torch.float32)
            jl = dense.JS if n < 777 or js_small else sorted({dense.choose_J(n, R), 64, 128})
            for vd in vds:
                for J in jl:
                    tag = f'{tag0} {cs.op} R {R} K {K} n {n} VD {vd} J {J}'
                    gd.t.view(torch.int32).fill_(SENT)
                    Dn.mm('Q6_K', cs.w, R, K, y, n, dv[:R * n], J=J, vd=vd); torch.cuda.synchronize(dev)
                    ok = bool(torch.equal(dv[:R * n].view(torch.int32), ref.view(torch.int32)))
                    okt = bool((gd.v.view(torch.int32)[R * n:] == SENT).all().item()) and gd.guards_ok(torch)
                    oki = bool(torch.equal(y, y0)) and bool(torch.equal(cs.w, cs.w0)) and gy.guards_ok(torch) and cs.gw.guards_ok(torch)
                    if not (ok and okt and oki):
                        nd = int((dv[:R * n].view(torch.int32) != ref.view(torch.int32)).sum().item())
                        check(False, f'{tag}: dst == MMQ (J {Jr}) bitwise {ok} ({nd} differ), tail + guards {okt}, w / y unchanged {oki}')
                        return False
            check(True, f'{tag0} {cs.op} R {R} K {K} n {n}: every VD {list(vds)} x J {list(jl)} == MMQ (J {Jr}) bitwise; dst tail, guards, w, y unchanged')
            del gy, gd, ref, y0
    return True


def stage_g1(torch, rd, card_shapes, vds, small):
    from caminus.prefill import mmq, dense
    dev = torch.device('cuda:0'); M = mmq.MMQ(dev, formats=('Q6_K',)); Dn = dense.Dense(dev)
    ns = (1, 3) if small else (1, 3, 17, 777, 1024, 2048)
    for c in card_shapes:
        cases = [Case(torch, dev, M, rd, op, name, il, *(s0 if c == 0 else s1)) for op, name, il, s0, s1 in SHAPES
                 if not small or op in ('k', 'ssm_out')]
        if not run_g1(torch, dev, M, Dn, cases, ns, vds, f'g1 card {c} shapes:', small): return
        del cases


def stage_g2b(torch, rd, vds):
    from caminus.prefill import mmq, dense
    from caminus.prefill.tp import micro_bounds
    dev = torch.device('cuda:0'); M = mmq.MMQ(dev, formats=('Q6_K',)); Dn = dense.Dense(dev)
    N = 2048
    splits = [(1024, 1024), (777, 1271), (1, 2047), (512, 788, 748)]
    for m in (3, 4): splits.append(tuple(b - a for a, b in micro_bounds(N, m)))
    for c in (0, 1):
        for op, name, il, s0, s1 in SHAPES:
            if op not in ('qkv', 'ssm_out', 'o', 'sg'): continue
            R, K = s0 if c == 0 else s1
            cs = Case(torch, dev, M, rd, op, name, il, R, K)
            x = torch.from_numpy(activations(K, N, 7)).to(dev)
            y = torch.zeros(mmq.y_bytes(K, N), dtype=torch.uint8, device=dev); M.quantize(x, K, N, y, M.layout('Q6_K'))
            ref = torch.zeros(R * N, dtype=torch.float32, device=dev); M.mm('Q6_K', cs.w, R, K, y, N, ref)
            for vd in vds:
                one = torch.zeros(R * N, dtype=torch.float32, device=dev); Dn.mm('Q6_K', cs.w, R, K, y, N, one, vd=vd)
                ok = bool(torch.equal(one.view(torch.int32), ref.view(torch.int32)))
                for sp in splits:
                    out = torch.full((R * N,), float('nan'), dtype=torch.float32, device=dev); r0 = 0
                    for p in sp:
                        yp = torch.zeros(mmq.y_bytes(K, p), dtype=torch.uint8, device=dev); M.quantize(x[r0:r0 + p].contiguous(), K, p, yp, M.layout('Q6_K'))
                        Dn.mm('Q6_K', cs.w, R, K, yp, p, out[r0 * R:(r0 + p) * R], vd=vd); r0 += p
                    ok &= bool(torch.equal(out.view(torch.int32), ref.view(torch.int32)))
                check(ok, f'g2b card {c} {op} R {R} K {K} VD {vd}: one launch of 2048 and every split {splits} == MMQ bitwise')
            del cs


def stage_speed(torch, rd, vds, reps):
    from caminus.prefill import mmq, dense
    from caminus.device import gemm as G
    dev = torch.device('cuda:0'); M = mmq.MMQ(dev, formats=('Q6_K',)); Dn = dense.Dense(dev)
    print(f'  speed: {torch.cuda.get_device_name(0)}, compaction_proactiveness {comp()}, MemAvailable {memavail()} MiB', flush=True)
    def tm(fn):
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
        fn(); torch.cuda.synchronize(dev)
        for a, b in ev: a.record(); fn(); b.record()
        torch.cuda.synchronize(dev); t = [a.elapsed_time(b) for a, b in ev]
        return statistics.median(t), max(t)
    res, longest = {}, 0.0
    JVDS = [v for v in (dense.VD,) if v in vds]
    g = torch.Generator(device='cpu').manual_seed(0)
    for c in (0, 1):
        for op, name, il, s0, s1 in SHAPES:
            R, K = s0 if c == 0 else s1
            cs = Case(torch, dev, M, rd, op, name, il, R, K)
            wbf = torch.randn(R, K, generator=g).to(torch.bfloat16).to(dev)
            for n in (1024, 2048):
                x = torch.from_numpy(activations(K, n, 3)).to(dev)
                y = torch.zeros(mmq.y_bytes(K, n), dtype=torch.uint8, device=dev); M.quantize(x, K, n, y, M.layout('Q6_K'))
                dst = torch.zeros(R * n, dtype=torch.float32, device=dev)
                xb = x.to(torch.bfloat16); ow = torch.zeros(n, R, dtype=torch.float32, device=dev)
                variants = [('mmq', lambda: M.mm('Q6_K', cs.w, R, K, y, n, dst))]
                for vd in vds: variants.append((f'v{vd}', lambda vd=vd: Dn.mm('Q6_K', cs.w, R, K, y, n, dst, vd=vd)))
                for jv in JVDS:
                    for J in dense.JS_VD[jv]:
                        if J != dense.choose_J(n, R): variants.append((f'v{jv}J{J}', lambda J=J, jv=jv: Dn.mm('Q6_K', cs.w, R, K, y, n, dst, J=J, vd=jv)))
                variants += [('wmma-bf16', lambda: G.gemm_tn(xb, wbf, out=ow, tile=(128, 128), round_bf16=True)),
                             ('torch-bf16', lambda: torch.nn.functional.linear(xb, wbf))]
                rounds = {k: [] for k, _ in variants}
                for rnd in range(3):
                    for k, fn in variants:
                        med, mx = tm(fn); rounds[k].append(med)
                        if k.startswith('v'): longest = max(longest, mx)
                med = {k: statistics.median(v) for k, v in rounds.items()}
                res[(c, op, n)] = med
                fl = 2 * R * K * n
                line = '  '.join(f'{k} {v:.4f}' for k, v in med.items() if 'J' not in k)
                best = '; '.join('best J v%d: %s %.4f' % ((jv,) + min((v, k) for k, v in med.items() if k.startswith(f'v{jv}J') or k == f'v{jv}')[::-1])
                                 for jv in JVDS)
                print(f'  TIME card {c} {op:8s} R {R:5d} K {K:5d} n {n:5d}: {line} (ours at dense.choose_J: J {dense.choose_J(n, R)})  | {best}; '
                      f'mmq {fl / med["mmq"] / 1e9:.1f} TF, v{dense.VD} {fl / med[f"v{dense.VD}"] / 1e9:.1f} TF', flush=True)
            del cs, wbf
    keys = [k for k in res[(0, 'qkv', 2048)] if 'J' not in k]   # (the J sweeps differ per shape: the default J is not repeated)
    for c in (0, 1):
        for n, mult in ((2048, 1), (1024, 2)):
            s = {k: sum(COUNT[op] * res[(c, op, n)][k] for op in COUNT) * mult for k in keys}
            NUMBERS[f'chunk_card{c}_{mult}x{n}'] = s
            print(f'  CHUNK card {c} dense Q6_K, 48 layers, chunk 2048 as {mult} x {n}: ' + '  '.join(f'{k} {v:.1f}' for k, v in s.items())
                  + ' ms', flush=True)
            for jv in JVDS:   # with each shape's best J (per op, per n)
                b = sum(COUNT[op] * min(v for k, v in res[(c, op, n)].items() if k.startswith(f'v{jv}J') or k == f'v{jv}') for op in COUNT) * mult
                NUMBERS[f'chunk_card{c}_{mult}x{n}_v{jv}_bestJ'] = b
                print(f'  CHUNK card {c} chunk 2048 as {mult} x {n}: v{jv} with each shape\'s best J {b:.1f} ms', flush=True)
    NUMBERS['per_op'] = {f'{c} {op} {n}': v for (c, op, n), v in res.items()}
    check(longest < 50.0, f'speed: every launch of ours under 50 ms (longest {longest:.3f} ms)')


def q6k_parts(raw, R, K):
    """the Q6_K values' parts ([R][K] each, ggml's dequantize_row_q6_K layout): the block scale d (fp16 as fp64), the
    sub-block scale sc and the 6-bit value q - 32"""
    b = raw.reshape(R, K // 256, 210)
    ql = b[..., :128].astype(np.int64); qh = b[..., 128:192].astype(np.int64)
    sc = b[..., 192:208].view(np.int8).astype(np.int64); d = b[..., 208:210].copy().view(np.float16).astype(np.float64)[..., 0]
    q = np.zeros((R, K // 256, 256), dtype=np.int64); l = np.arange(32)
    for n in range(2):
        L = ql[..., 64 * n:64 * n + 64]; H = qh[..., 32 * n:32 * n + 32]; base = 128 * n
        q[..., base + l] = ((L[..., l] & 0xF) | (((H[..., l] >> 0) & 3) << 4)) - 32
        q[..., base + l + 32] = ((L[..., l + 32] & 0xF) | (((H[..., l] >> 2) & 3) << 4)) - 32
        q[..., base + l + 64] = ((L[..., l] >> 4) | (((H[..., l] >> 4) & 3) << 4)) - 32
        q[..., base + l + 96] = ((L[..., l + 32] >> 4) | (((H[..., l] >> 6) & 3) << 4)) - 32
    return (np.repeat(d[..., None], 256, -1).reshape(R, K), np.repeat(sc, 16, -1).reshape(R, K), q.reshape(R, K))


def deq64(raw, R, K):
    """the exact Q6_K values in fp64"""
    d, sc, q = q6k_parts(raw, R, K)
    return d * sc * q


def stage_tol(torch, rd, ref_stock, scratch):
    """candidate (b) (ChunkStack(q6k='wmma'): dense.deq_bf16 + the activation's bf16 copy + gemm.hip's WMMA) as a measured
    tolerance: its error vs fp64 <= llama.cpp's own batched op's (the stock build's hipBLAS path for dense Q6_K above 128
    columns: --ref-stock gg_mmq_ref), max |err| / max |ref| and mean |err| / mean |ref|, with the bf16-rounded output
    (prefill_ops.WMMA_OUT's default) and fp32; the dequantization's bytes in a guarded buffer, == fp32 d * sc * q rounded to
    bf16 (RNE); G2b: one launch == splits, bitwise. MMQ's (== caminus.prefill.dense) error reported beside them"""
    import subprocess
    from caminus.prefill import mmq, dense, glue as po
    dev = torch.device('cuda:0'); M = mmq.MMQ(dev, formats=('Q6_K',)); Dn = dense.Dense(dev); Gl = po.Glue(dev)
    os.makedirs(scratch, exist_ok=True)
    cases = [('qkv', 'attn_qkv', 0, 6400, 2560), ('ssm_out', 'ssm_out', 0, 2560, 2304), ('o', 'attn_output', 7, 2560, 3072), ('sg', 'ffn_gate_shexp', 4, 384, 2560)]
    N = 1024; jobs = []
    for op, name, il, R, K in cases:
        raw = load_weight(torch, dev, rd, name, il, R, K); x = activations(K, N, 11, edges=False)   # (the S3 method: real / normal rows, no edge columns)
        raw.tofile(os.path.join(scratch, f'tol_{op}_w.bin')); x.tofile(os.path.join(scratch, f'tol_{op}_x.bin'))
        jobs.append(f'mm Q6_K {R} {K} {N} {scratch}/tol_{op}_w.bin {scratch}/tol_{op}_x.bin {scratch}/tol_{op}_stock.bin')
    open(os.path.join(scratch, 'tol_jobs.txt'), 'w').write('\n'.join(jobs) + '\n')
    p = subprocess.run([ref_stock, os.path.join(scratch, 'tol_jobs.txt')], capture_output=True, text=True, timeout=600)
    if not check(p.returncode == 0, f'tol: llama.cpp stock gg_mmq_ref ran ({p.returncode}; {p.stdout.strip().splitlines()[:1]})'): print(p.stderr[-1500:]); return
    def err(o, r):
        e = np.abs(o.astype(np.float64) - r); return float(e.max() / np.abs(r).max()), float(e.mean() / np.abs(r).mean())
    out = {}
    for op, name, il, R, K in cases:
        raw = np.fromfile(os.path.join(scratch, f'tol_{op}_w.bin'), dtype=np.uint8); xn = np.fromfile(os.path.join(scratch, f'tol_{op}_x.bin'), dtype=np.float32).reshape(N, K)
        stock = np.fromfile(os.path.join(scratch, f'tol_{op}_stock.bin'), dtype=np.float32).reshape(N, R)
        W64 = deq64(raw, R, K); ref = xn.astype(np.float64) @ W64.T
        w = torch.from_numpy(raw).to(dev); x = torch.from_numpy(xn).to(dev)
        gw = Guarded(torch, dev, 2 * R * K + 4096); wb = gw.v[:2 * R * K].view(torch.bfloat16)
        Dn.deq_bf16('Q6_K', w, R, K, wb); torch.cuda.synchronize(dev)
        d_, sc_, q_ = q6k_parts(raw, R, K)
        host = torch.from_numpy(np.ascontiguousarray((d_.astype(np.float32) * sc_.astype(np.float32)) * q_.astype(np.float32)))   # ggml's fp32 (d * sc) * q
        hb = host.to(torch.bfloat16)
        okd = bool(torch.equal(wb.cpu().view(torch.int16), hb.reshape(-1).view(torch.int16))) and gw.guards_ok(torch) and bool((gw.v[2 * R * K:].view(torch.int32) == SENT).all().item())
        check(okd, f'tol {op} R {R} K {K}: deq_bf16 == ggml\'s fp32 d * sc * q rounded to bf16, every value; guards and the tail unchanged')
        xb = torch.zeros(po.tpad(N) * K, dtype=torch.bfloat16, device=dev); Gl.to_bf16(x, xb, N, K)
        o_b = torch.zeros(po.tpad(N) * R, dtype=torch.float32, device=dev); o_f = torch.zeros_like(o_b)
        Gl.bf16_gemm(wb, R, K, xb, o_b, N, wmma_out='bf16'); Gl.bf16_gemm(wb, R, K, xb, o_f, N, wmma_out='fp32')
        y = torch.zeros(mmq.y_bytes(K, N), dtype=torch.uint8, device=dev); M.quantize(x, K, N, y, M.layout('Q6_K'))
        o_m = torch.zeros(R * N, dtype=torch.float32, device=dev); M.mm('Q6_K', w, R, K, y, N, o_m)
        e = dict(stock=err(stock, ref), b_bf16out=err(o_b.view(-1, R)[:N].cpu().numpy(), ref), b_fp32out=err(o_f.view(-1, R)[:N].cpu().numpy(), ref),
                 mmq=err(o_m.view(N, R).cpu().numpy(), ref), stock_low13_zero=float(np.mean((stock.view(np.uint32) & 0x1FFF) == 0)))
        out[op] = e
        check(e['b_bf16out'][0] <= e['stock'][0] and e['b_bf16out'][1] <= e['stock'][1] and e['b_fp32out'][0] <= e['stock'][0] and e['b_fp32out'][1] <= e['stock'][1],
              f'tol {op} R {R} K {K} n {N}: (b) error vs fp64 <= llama.cpp\'s own (stock hipBLAS): bf16 out {e["b_bf16out"][0]:.2e} / {e["b_bf16out"][1]:.2e}, '
              f'fp32 out {e["b_fp32out"][0]:.2e} / {e["b_fp32out"][1]:.2e} vs stock {e["stock"][0]:.2e} / {e["stock"][1]:.2e} (max / mean, relative); '
              f'MMQ {e["mmq"][0]:.2e} / {e["mmq"][1]:.2e}; stock output fp16-rounded on {100 * e["stock_low13_zero"]:.1f} % (low 13 bits zero)')
        # G2b: the same rows as one launch and split
        full = o_b.view(-1, R)[:N].clone(); ok = True
        for sp in ((512, 512), (777, 247), (1, 1023), (384, 384, 256)):
            r0 = 0
            for q_ in sp:
                xq = torch.zeros(po.tpad(q_) * K, dtype=torch.bfloat16, device=dev); Gl.to_bf16(x[r0:r0 + q_].contiguous(), xq, q_, K)
                oq = torch.full((po.tpad(q_) * R,), float('nan'), dtype=torch.float32, device=dev); Gl.bf16_gemm(wb, R, K, xq, oq, q_, wmma_out='bf16')
                ok &= bool(torch.equal(oq.view(-1, R)[:q_].view(torch.int32), full[r0:r0 + q_].view(torch.int32))); r0 += q_
        check(ok, f'tol {op}: G2b, (b) on splits (512+512, 777+247, 1+1023, 384+384+256) == one launch of {N}, bitwise')
    NUMBERS['tol'] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('coder'); ap.add_argument('stages', nargs='+')
    ap.add_argument('--small', action='store_true'); ap.add_argument('--vds', default='0,1,2,3')
    ap.add_argument('--cards', default='0,1', help='the TP cards whose shard shapes g1 runs')
    ap.add_argument('--reps', type=int, default=20); ap.add_argument('--json', default=None)
    ap.add_argument('--ref-stock', default=os.path.join(ROOT, 'build', 'gg_mmq_ref'), help='tol: llama.cpp\'s stock gg_mmq_ref (hipBLAS above 128 columns)')
    ap.add_argument('--scratch', default=None, help='default $ARTIFACTS/impl-q6k-dense/scratch; ' 'tol: the job / weight / activation files (fixed names)')
    a = ap.parse_args()
    vds = tuple(int(v) for v in a.vds.split(','))
    t0 = time.time()
    print(f'start: compaction_proactiveness {comp()}, MemAvailable {memavail()} MiB', flush=True)
    if 'cpu' in a.stages: stage_cpu()
    gpu = [s for s in a.stages if s != 'cpu']
    if gpu:
        import torch
        sys.path.insert(0, os.path.expanduser('~/src/ignite-llama/gguf-py'))
        from gguf import GGUFReader
        name = torch.cuda.get_device_name(0)
        print(f'device 0: {name}', flush=True)
        rd = GGUFReader(a.coder)
        for s in gpu:
            if FAILS: break
            print(f'== stage {s}', flush=True)
            if s == 'g1': stage_g1(torch, rd, [int(c) for c in a.cards.split(',')], vds, a.small)
            elif s == 'g2b': stage_g2b(torch, rd, vds)
            elif s == 'speed': stage_speed(torch, rd, vds, a.reps)
            elif s == 'tol': stage_tol(torch, rd, a.ref_stock, a.scratch or devpaths.need('ARTIFACTS', 'impl-q6k-dense', 'scratch'))
            else: raise SystemExit(f'unknown stage {s}')
    print(f'end: compaction_proactiveness {comp()}, MemAvailable {memavail()} MiB, {time.time() - t0:.0f} s', flush=True)
    if a.json:
        with open(a.json, 'w') as f: json.dump(dict(fails=FAILS, numbers=NUMBERS), f, indent=1, default=str)
    print('RESULT ' + ('FAIL' if FAILS else 'PASS'))
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
