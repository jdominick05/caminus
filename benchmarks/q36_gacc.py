"""docs/specs/spec-own-quant.md's G-acc for the own Q2_0 tile (kernels/quant_q2_0.h, kernels/bench_own_mv_q2_0.hip) on
recorded activations, one card: Qwen3.6-35B-A3B's routed experts (gate / up K 2048, down K 512), and with --coder-acts
the GSQ-RCO Coder's Q2_0 expert down (K 640 cut 384 / 256 over its two cards), plus a standalone speed read.

--record ACTS (needs K3_TEST_HOOKS=gacc-sites): Qwen35One (the one-card engine on llama.cpp's math, every layer its own
  site rows under the hook) teacher-forced over three texts of this tree (tokenized by caminus.bpe), --positions each;
  per layer the FFN half's input (mixed: the experts' gate / up read it), each slot's h (the experts' down input), the
  router's ids and weights. Checked while recording: each fp32 row quantizes to the q8_1 copy the op read (so the rows are
  the ops' inputs). A recording is a reference: an existing ACTS is never overwritten.
--acts ACTS: per tensor (each Q2_0 expert role, its first and last layer; --layers all: every layer), on 32
  sampled rows of each routed expert with the vectors routed to it: |y - y_ref| / rms(y_ref), y_ref the float64 dot of
  caminus.own_dequant's rows with the recorded fp32 row; mean and p99 per tensor and pooled per (op, format), for
  own (fp16 activations, own_mv_q2_0_k3: the tile in K3's form) and llama.cpp (ggml-cuda's vec_dot_q2_0_q8_1 on q8_1
  activations, kernels/ref_gg_mv.hip). PASS when own's mean and p99 are both at most llama.cpp's, per pair.
--coder-acts FILE --coder GGUF: the same on the Coder's recording (tests/engine/test_k3_gguf_gsq.py --record), its
  MOE_DOWN Q2_0 sets sampled as that test's (e) samples them, per card cut; --coder-baseline JSON (its --gacc-out) is
  checked to equal the llama.cpp numbers computed here.
Speed (decode op, standalone): the grouped gate | up and down of 8 slots, one launch each, cold (layers and ids
rotated over GACC_SPEED_LAYERS whole tensors), median us over --rounds, against own_read of the same byte count (the
bandwidth ceiling); not the module's time (G-speed is the module A/B).
  K3_TEST_HOOKS=gacc-sites HIP_VISIBLE_DEVICES=<one card> tools/guard.sh LOG 600 python -I benchmarks/q36_gacc.py GGUF --record ACTS.pt
  HIP_VISIBLE_DEVICES=<one card> tools/guard.sh LOG 600 python -I benchmarks/q36_gacc.py GGUF --acts ACTS.pt
      [--coder CODER --coder-acts CODER_ACTS.pt --coder-baseline GACC.json] [--gacc-out OUT.json] [--layers ends | all]
New kernel code: run it on a non-display card alone first. Prints PASS / FAIL lines and RESULT PASS | FAIL."""
import argparse, ctypes, json, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'python')); sys.path.insert(0, os.path.join(ROOT, 'tools', 'gates'))
import numpy as np
import torch
from caminus import ggufio, own_dequant
from caminus.device import capacity
from caminus.device.hip import Module
import gacc  # noqa: E402  (tools/gates/gacc.py: G-acc's shared pieces)

ap = argparse.ArgumentParser()
ap.add_argument('gguf')
ap.add_argument('--record', default=None)
ap.add_argument('--positions', type=int, default=32, help='--record: teacher-forced positions per text (3 texts)')
ap.add_argument('--acts', default=None)
ap.add_argument('--layers', choices=('ends', 'all'), default='ends')
ap.add_argument('--coder', default=None)
ap.add_argument('--coder-acts', default=None)
ap.add_argument('--coder-baseline', default=None)
ap.add_argument('--gacc-out', default=None, help='write own and llama.cpp per tensor and per pair here (JSON; refuses an existing file)')
ap.add_argument('--rounds', type=int, default=7)
A = ap.parse_args()
TEXTS = [os.path.join(ROOT, 'docs', 'dev', 'architecture.md'), os.path.join(ROOT, 'python', 'caminus', 'arch.py'),
         os.path.join(ROOT, 'README.md')]
GACC_SPEED_LAYERS = 4         # whole tensors the speed read rotates over (Qwen3.6: 4 x 226 MB, past the Infinity Cache)
ok_all = True
dev = torch.device('cuda:0')


def check(c, what):
    global ok_all
    print(('PASS ' if c else 'FAIL ') + what, flush=True); ok_all &= bool(c)


# ---- --record ---------------------------------------------------------------------------------------------------------------
if A.record:
    from caminus.bpe import BPE
    from caminus.engine import gguf as G
    from caminus.engine.qwen35_one import Qwen35One, GACC_SITE_HOOKS
    if os.environ.get('K3_TEST_HOOKS') != GACC_SITE_HOOKS: sys.exit(f'--record: run it with K3_TEST_HOOKS={GACC_SITE_HOOKS}')
    if os.path.exists(A.record): sys.exit(f'--record: {A.record} exists: refusing to overwrite (a recording is a reference)')
    if 3 * A.positions < 64: sys.exit('--record: need at least 64 vectors per site (3 texts x --positions)')
    tk = BPE(ggufio.GGUFFile(A.gguf).meta)
    texts = [tk.encode(open(p, encoding='utf-8').read())[:A.positions] for p in TEXTS]
    t0 = time.time(); m = Qwen35One(A.gguf, max_ctx=A.positions + 16)
    assert m.gacc_sites and not m.dense and m.ntok == 1, 'the qwen35moe tape, one token, under the hook'
    print(f'Qwen35One {m.module} loaded in {time.time() - t0:.0f} s; {len(texts)} texts x {A.positions} positions', flush=True)
    rec, bad = {}, {}
    t0 = time.time()
    for toks in texts:
        m.reset()
        for i in range(A.positions):
            m.step(toks, i)
            for il in range(m.nl):
                b = m.site_bufs[(il, 'ffn')]
                for x, q in (('mixed', 'mq'), ('h', 'hq')):
                    y = G.quantize_q8_1(b[x]); bad[f'{q} = q8_1({x})'] = bad.get(f'{q} = q8_1({x})', 0) + (not torch.equal(y, b[q][:y.numel()]))
                for n in ('mixed', 'h', 'ids', 'wts'): rec.setdefault(f'L{il}.ffn.{n}', []).append(b[n].reshape(-1).cpu().clone())   # (flat rows)
    print(f'recorded in {time.time() - t0:.0f} s', flush=True)
    for kind, nb in sorted(bad.items()): check(nb == 0, f'record: {kind}: {nb} of {len(texts) * A.positions * m.nl} rows differ')
    if not ok_all: print('--record: a recording check failed: nothing saved'); print('RESULT FAIL'); sys.exit(1)
    nv = len(texts) * A.positions
    for il in range(m.nl):   # [vectors, row]: mixed [D] (the experts' gate / up input), h [k_used * ffn], ids / wts [k_used]
        for n, w in (('mixed', m.D), ('h', m.k_used * m.ffn), ('ids', m.k_used), ('wts', m.k_used)):
            assert len(rec[f'L{il}.ffn.{n}']) == nv and rec[f'L{il}.ffn.{n}'][0].shape == (w,), (il, n, rec[f'L{il}.ffn.{n}'][0].shape, w)
    out = dict(meta=dict(model=os.path.basename(A.gguf), texts=[os.path.relpath(p, ROOT) for p in TEXTS], positions=A.positions,
                         nl=int(m.nl), k_used=int(m.k_used), ffn=int(m.ffn), module=str(m.module), tool='benchmarks/q36_gacc.py --record'),
               sites={k: torch.stack(v) for k, v in rec.items()})
    if os.path.exists(A.record): sys.exit(f'--record: {A.record} appeared during the run: refusing to overwrite')
    torch.save(out, A.record)
    print(f'saved {A.record}: {len(out["sites"])} sites x {len(texts) * A.positions} vectors')
    print('RESULT', 'PASS' if ok_all else 'FAIL'); sys.exit(0 if ok_all else 1)

# ---- --acts: G-acc and speed --------------------------------------------------------------------------------------------------
if not A.acts: sys.exit('q36_gacc: --record ACTS or --acts ACTS')
if A.gacc_out and os.path.exists(A.gacc_out): sys.exit(f'--gacc-out: {A.gacc_out} exists: refusing to overwrite')
OWN = Module('bench_own_mv_q2_0'); GG = Module('ref_gg_mv'); RD = Module('bench_own_mv')   # (own_read: the plain read)
sink = torch.zeros(1, dtype=torch.int32, device=dev)


def arms(raw, K, X):
    """(own, llama) outputs [r, v] float64 of raw rows [r, rb] on each recorded fp32 row of X [v, K]"""
    r = raw.shape[0]; w = torch.from_numpy(np.ascontiguousarray(raw)).to(dev).reshape(-1)
    o = torch.empty(2, r, device=dev); own, ll = [], []
    for x in X:
        x32 = x.to(dev).float().contiguous()
        OWN.own_mv_q2_0_k3(((r + 15) // 16,), (256,), w, ctypes.c_int(r), ctypes.c_int(K), x32.half(), o[0])
        GG.gg_ref_mv_q2_0(((r + 3) // 4,), (32, 4), w, gacc.ref_q8_1(GG, x32), o[1], ctypes.c_int(r), ctypes.c_int(K))
        own.append(o[0].double().cpu().numpy()); ll.append(o[1].double().cpu().numpy())
    return np.array(own).T, np.array(ll).T


def errors(raw, K, X):
    """(own, llama) G-acc errors of one row set, and its zero-reference count (tools/gates/gacc.py rel_errors)"""
    W = own_dequant.dequant(raw.reshape(-1), 'Q2_0', raw.shape[0] * K).astype(np.float64).reshape(raw.shape[0], K)
    Y = W @ X.double().numpy().T                                     # [r, v]
    own, ll = arms(raw, K, X)
    (eo, nz), (el, _) = gacc.rel_errors(Y, own), gacc.rel_errors(Y, ll)
    return eo, el, nz


pooled, tensors = {}, []


def record_tensor(src, name, op, sets, nvec):
    """pool and print one tensor's errors: sets = [(raw rows, K, vectors [v, K], row ids)]"""
    eo, el, seen, nz = [], [], set(), 0
    for raw, K, X, rows in sets:
        a, b, z = errors(raw, K, X); eo.append(a); el.append(b); nz += z; seen.update(int(q) for q in rows)
    eo, el = np.concatenate(eo), np.concatenate(el)
    r = dict(src=src, tensor=name, op=op, fmt='Q2_0', rows=len(seen), vectors=nvec, dots=int(eo.size), own=gacc.summary(eo), llama=gacc.summary(el))
    check(nvec >= 64 and len(seen) >= 256 and bool(np.isfinite(eo).all() and np.isfinite(el).all()) and nz == 0,
          f'{src} {op} Q2_0 {name}: {nvec} vectors (>= 64), {len(seen)} distinct rows (>= 256), finite errors, {nz} zero references')
    print(f'  {src} G-acc {op:8s} Q2_0 {name:28s} {len(seen):6d} rows {eo.size:7d} dots: own mean {r["own"]["mean"]:.2e} p99 '
          f'{r["own"]["p99"]:.2e} | llama.cpp mean {r["llama"]["mean"]:.2e} p99 {r["llama"]["p99"]:.2e}', flush=True)
    pooled.setdefault((src, op), ([], []))
    pooled[(src, op)][0].append(eo); pooled[(src, op)][1].append(el); tensors.append(r)


# Qwen3.6: the recorded sites
R_ = torch.load(A.acts, weights_only=False); meta, sites = R_['meta'], R_['sites']
F = ggufio.GGUFFile(A.gguf)
KU, FFN, NL = meta['k_used'], meta['ffn'], meta['nl']
NV = sites['L0.ffn.mixed'].shape[0]; D_ = int(F.T['blk.0.ffn_gate_exps.weight'].shape[0])
for il in range(NL):
    for n, w in (('mixed', D_), ('h', KU * FFN), ('ids', KU)):
        assert tuple(sites[f'L{il}.ffn.{n}'].shape) == (NV, w), (f'L{il}.ffn.{n}', tuple(sites[f'L{il}.ffn.{n}'].shape), (NV, w))
print(f'{torch.cuda.get_device_name(0)}  {os.path.basename(A.gguf)}  recording {os.path.basename(A.acts)} ({meta["texts"]}, '
      f'{meta["positions"]} positions each)', flush=True)
ROLES = (('ffn_gate_exps', 'MOE_GU', 'mixed'), ('ffn_up_exps', 'MOE_GU', 'mixed'), ('ffn_down_exps', 'MOE_DOWN', 'h'))
for role, op, x in ROLES:
    ls = [il for il in range(NL) if F.T[f'blk.{il}.{role}.weight'].type == 'Q2_0']
    for il in (ls if A.layers == 'all' else sorted({ls[0], ls[-1]})):
        name = f'blk.{il}.{role}.weight'; t = F.T[name]; K, Rw, E = t.shape
        data = np.asarray(t.data()).reshape(E, Rw, -1)
        ids = sites[f'L{il}.ffn.ids'].numpy(); X = sites[f'L{il}.ffn.{x}']
        sets = [(raw, K, Xe, rr) for raw, Xe, rr in gacc.expert_sets(data, ids, X, op == 'MOE_DOWN')]
        record_tensor('qwen3.6', name, op, sets, X.shape[0])

# the Coder's MOE_DOWN Q2_0 sets (its recording and its (e)'s sampling)
if A.coder_acts:
    if not A.coder: sys.exit('--coder-acts: name the Coder shard with --coder')
    C_ = torch.load(A.coder_acts, weights_only=False); cm, cs, csh = C_['meta'], C_['sites'], C_['shards']
    CF = ggufio.GGUFFile(A.coder)
    ls = [il for il in range(cm['nl']) if f'blk.{il}.ffn_down_exps.weight' in CF.T and CF.T[f'blk.{il}.ffn_down_exps.weight'].type == 'Q2_0']
    for il in (ls if A.layers == 'all' else sorted({ls[0], ls[-1]})):
        name = f'blk.{il}.ffn_down_exps.weight'; t = CF.T[name]; K, Rw, E = t.shape
        data = np.asarray(t.data()).reshape(E, Rw, -1); ids = cs[f'L{il}.ffn.ids'].numpy()
        sets = []   # (per expert, card 0's shard then card 1's: the stored baseline's order)
        for c0, c1 in zip(*[gacc.expert_sets(data, ids, cs[f'L{il}.ffn.h.c{c}'], True, ranges=csh[name]['ranges'][c], bs=64, ts=18)
                            for c in (0, 1)]):
            sets += [(raw, raw.shape[1] // 18 * 64, Xe, rr) for raw, Xe, rr in (c0, c1)]
        record_tensor('coder', name, 'MOE_DOWN', sets, cs[f'L{il}.ffn.h.c0'].shape[0])
pairs = {}
for (src, op), (eo, el) in sorted(pooled.items()):
    eo, el = np.concatenate(eo), np.concatenate(el)
    p = dict(own=gacc.summary(eo), llama=gacc.summary(el), dots=int(eo.size), tensors=len(pooled[(src, op)][0]))
    pairs[f'{src} {op} Q2_0'] = p
    check(p['own']['mean'] <= p['llama']['mean'] and p['own']['p99'] <= p['llama']['p99'],
          f'G-acc {src} {op} Q2_0 pooled ({p["dots"]} dots, {p["tensors"]} tensors): own mean {p["own"]["mean"]:.2e} p99 {p["own"]["p99"]:.2e} '
          f'<= llama.cpp mean {p["llama"]["mean"]:.2e} p99 {p["llama"]["p99"]:.2e} ({p["llama"]["mean"] / p["own"]["mean"]:.0f}x / '
          f'{p["llama"]["p99"] / p["own"]["p99"]:.0f}x)')
if A.coder_baseline:   # the llama.cpp arm here against the Coder's stored baseline (same rows, vectors and kernel)
    base = json.load(open(A.coder_baseline))
    for r in (r for r in tensors if r['src'] == 'coder'):
        b = next((t for t in base['tensors'] if t['tensor'] == r['tensor'] and t['fmt'] == 'Q2_0'), None)
        check(b is not None and b['dots'] == r['dots'] and abs(b['mean'] - r['llama']['mean']) <= 1e-12 and abs(b['p99'] - r['llama']['p99']) <= 1e-12,
              f'coder {r["tensor"]}: the llama.cpp arm equals the stored baseline (mean {r["llama"]["mean"]:.6e} vs '
              f'{b["mean"] if b else float("nan"):.6e}, dots {r["dots"]} vs {b["dots"] if b else None})')

# ---- speed: the grouped ops of 8 slots, standalone, cold ---------------------------------------------------------------------
lsp = [il for il in range(NL) if F.T[f'blk.{il}.ffn_gate_exps.weight'].type == 'Q2_0']
lsp = [lsp[round(k * (len(lsp) - 1) / (GACC_SPEED_LAYERS - 1))] for k in range(GACC_SPEED_LAYERS)]
T3 = [{r: F.T[f'blk.{il}.{r}.weight'] for r, _, _ in ROLES} for il in lsp]
capacity.check([0], [sum(t.n_bytes for L in T3 for t in L.values()) / 2 ** 20 + 256], f'q36_gacc speed: {len(lsp)} layers of experts')
Wd = [{r: torch.from_numpy(np.array(t.data())).reshape(-1).to(dev) for r, t in L.items()} for L in T3]   # (flat bytes)
tg, td = T3[0]['ffn_gate_exps'], T3[0]['ffn_down_exps']
Kg, Rg, E = tg.shape; Kd, Rd, _ = td.shape
rbg, rbd = Kg // 64 * 18, Kd // 64 * 18
assert Rg % 2 == 0 and Rd % 2 == 0 and E > 0, (Rg, Rd, E)   # (the grouped kernels' row pairs and id clamp assume them)
x16 = sites[f'L{lsp[0]}.ffn.mixed'][0].to(dev).half().contiguous()
h16 = torch.from_numpy(np.random.default_rng(0).standard_normal(KU * Kd)).to(dev).half()
y16 = torch.empty(KU * Rg, dtype=torch.float16, device=dev); yd = torch.empty(KU * Rd, device=dev)
rng = np.random.default_rng(7)
IDS = [torch.from_numpy(rng.choice(E, KU, replace=False).astype(np.int32)).to(dev) for _ in range(len(lsp) * 4)]
bg, bd = KU * 2 * Rg * rbg, KU * Rd * rbd                           # bytes the two ops read (weights)


def read(w, nbytes, k):
    off = (k * 7919 % max(1, (w.numel() - nbytes) // 4096)) * 4096
    gacc.own_read(RD, w[off:off + nbytes], sink)


ops = {'gate|up': lambda L, ids, k: OWN.own_moe_gu_q2_0(((KU * Rg + 15) // 16,), (256,), L['ffn_gate_exps'], L['ffn_up_exps'], ids,
                                                         ctypes.c_int(KU), ctypes.c_int(Rg), ctypes.c_int(E), ctypes.c_int(Kg), x16, y16, ctypes.c_void_p(0), ctypes.c_void_p(0)),
       'read gate|up': lambda L, ids, k: read(L['ffn_gate_exps'], bg, k),
       'down': lambda L, ids, k: OWN.own_moe_down_q2_0(((KU * Rd + 15) // 16,), (256,), L['ffn_down_exps'], ids, ctypes.c_int(KU),
                                                       ctypes.c_int(Rd), ctypes.c_int(E), ctypes.c_int(Kd), h16, yd),
       'read down': lambda L, ids, k: read(L['ffn_down_exps'], bd, k)}
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
tm = {k: [] for k in ops}
for r in range(A.rounds + 1):
    for k, fn in ops.items():
        torch.cuda.synchronize(); ev[0].record()
        for j, ids in enumerate(IDS): fn(Wd[j % len(Wd)], ids, j + r)
        ev[1].record(); torch.cuda.synchronize()
        if r: tm[k].append(ev[0].elapsed_time(ev[1]) * 1e3 / len(IDS))
us = {k: float(np.median(v)) for k, v in tm.items()}
print(f'speed (decode op, standalone, one launch each, cold over layers {lsp}, median of {A.rounds}): gate | up {us["gate|up"]:.1f} us '
      f'({bg / us["gate|up"] / 1e3:.0f} GB/s) vs read {us["read gate|up"]:.1f} us; down {us["down"]:.1f} us ({bd / us["down"] / 1e3:.0f} GB/s) '
      f'vs read {us["read down"]:.1f} us; {bg / 2 ** 20:.2f} / {bd / 2 ** 20:.2f} MiB per op', flush=True)
if A.gacc_out:
    if os.path.exists(A.gacc_out): sys.exit(f'--gacc-out: {A.gacc_out} appeared during the run: refusing to overwrite')
    with open(A.gacc_out, 'w') as f:
        json.dump(dict(acts=os.path.basename(A.acts), acts_meta=meta, coder_acts=A.coder_acts and os.path.basename(A.coder_acts), layers=A.layers,
                       exp_rows=gacc.EXP_ROWS, arms=dict(own='build/bench_own_mv_q2_0.hsaco own_mv_q2_0_k3 (fp16)', llama='build/ref_gg_mv.hsaco (q8_1)'),
                       pairs=pairs, tensors=tensors, speed_us=us), f, indent=1)
    print(f'wrote {A.gacc_out}')
print('RESULT', 'PASS' if ok_all else 'FAIL'); sys.exit(0 if ok_all else 1)
