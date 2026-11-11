"""Dense35Own (the dense qwen35 line on one card, Caminus's own code from the GGUF to the logits; module k3_q35_swift)
against caminus.ref_q35 (the float64 host forward on the same reader and decoders).

Without a model file, on the tiny synthetic quantized model of tests/engine/tiny_q35.py (every matrix format the module
has a tile for; small inputs: the first run of new kernel code):
  - each quantized matrix's matvec on fp16 activations against the float64 product of the decoded weights
  - the residual after each block (the engine built to that depth) and the logits, position by position
  - the engine against itself: a second run after reset(), a position run again without committing the one before it,
    and the tape at 3 / 6 / 7 workgroups per WGP, all bitwise
With --model and --ref (the npz of tools/reference/q35_ref.py --trace: seq, logits, resid):
  - the device embedding rows bitwise the host decoder
  - the residual after the blocks of --depths, then the logits of --positions positions: top-1, KLD(ref || gpu),
    max |logit difference|; the grids bitwise; ms per token (reported, not gated: the grid is not measured for this model)
Gates: the matvecs within 1e-5 of the row's sum |w x|; the tiny model's residuals within 1 % of the block's largest value
and its logits' mean KLD < 1e-3; the model's mean KLD < 1e-3 and top-1 equal at all but at most 1 position in 100.
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  tools/guard.sh LOG 600 python -I tests/engine/test_q35_dense.py --scratch DIR
  tools/guard.sh LOG 1800 python -I tests/engine/test_q35_dense.py --scratch DIR --model GGUF --ref FILE.npz
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, gc, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, HERE)
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--scratch', required=True); ap.add_argument('--model'); ap.add_argument('--ref')
ap.add_argument('--positions', type=int, default=128); ap.add_argument('--depths', type=int, nargs='*', default=[1, 4])
ap.add_argument('--depth-positions', type=int, default=12); ap.add_argument('--max-ctx', type=int, default=256)
ap.add_argument('--module', default=None)
A = ap.parse_args()
if bool(A.model) != bool(A.ref): ap.error('--model and --ref go together')
os.makedirs(A.scratch, exist_ok=True)
import torch
import tiny_q35
from caminus import ggufio, own_dequant, ref_q35
from caminus.device import capacity
from caminus.engine import k3
from caminus.engine.q35_dense import MODULE, Dense35Own
MOD = A.module or MODULE
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


def lsm(v): v = v - v.max(); return v - np.log(np.exp(v).sum())


def kld(r, g): lr, lg = lsm(r), lsm(g); return float(np.sum(np.exp(lr) * (lr - lg)))


print(torch.cuda.get_device_name(0), flush=True)
DEV = torch.cuda.current_device()

# ---- the tiny model ------------------------------------------------------------------------------------------------------
P = os.path.join(A.scratch, 'tiny-q35-quant.gguf'); info = tiny_q35.write(P)
SEQ = [1, 7, 3, 60, 7, 2, 33, 9, 9, 41]
NB = info['blocks']
# this part's own buffers and its engines (one per depth, three more for the options; each a few MiB): checked once
capacity.check([DEV], [(NB + 4) * Dense35Own.projection_mib(P, 64) + 16], 'test_q35_dense: the tiny model\'s engines and op buffers')

# the matvec of every quantized matrix of the tiny file on fp16 activations, against float64
F = ggufio.GGUFFile(P); rng = np.random.default_rng(2); worst = {}
E = k3.Engine(module=MOD); outs = []
for t in F.tensors:
    if t.type not in ('Q4_K', 'Q6_K', 'Q8_0', 'Q5_K') or t.name == 'token_embd.weight': continue
    R, _ = t.rows(); K = int(t.shape[0])
    x = (rng.standard_normal(K) * 2).astype(np.float16); x16 = torch.from_numpy(x).cuda(); out = torch.zeros(R, device='cuda')
    raw = t.pread(0, t.n_bytes)
    if t.type in k3.OWN_1RT_FMT:
        from caminus.engine.own_layout import q6k_planes
        w = torch.from_numpy(np.ascontiguousarray(q6k_planes(raw, R, K) if t.type == 'Q6_K' else raw)).cuda()
        E.own_matvec_1rt(w, t.type, R, K, x16, out, rows_per_task=16)
    else:
        w = torch.from_numpy(np.ascontiguousarray(raw)).cuda()
        E.own_matvec16(w, t.type, R, K, x16, out, rows_per_task=32)
    outs.append((t, x, out, w, x16))
E.build().run(); torch.cuda.synchronize()
for t, x, out, _, _ in outs:
    R, _ = t.rows(); W = own_dequant.rows(t, 0, R).astype(np.float64); xd = x.astype(np.float64)
    rel = float((np.abs(out.double().cpu().numpy() - W @ xd) / (np.abs(W) @ np.abs(xd))).max())
    worst[t.type] = max(worst.get(t.type, 0.0), rel)
check(set(worst) == {'Q4_K', 'Q6_K', 'Q8_0', 'Q5_K'} and max(worst.values()) < 1e-5,
      f'{len(outs)} matvecs on fp16 activations against float64, worst error / sum |w x| per format: ' +
      ', '.join(f'{k} {v:.1e}' for k, v in sorted(worst.items())) + ' (< 1e-5)')
F.close(); del E, outs

ref = ref_q35.Q35Ref(P); trace = []; rl = ref.feed(SEQ, trace=trace)   # trace[b]: [N, D] after block b


def run(eng, seq, grab=None):
    eng.reset(); out = []
    for t in seq:
        lg = eng.step(t)
        out.append((eng.buf['x'] if grab == 'x' else lg).clone())
    torch.cuda.synchronize(); return out


for n in range(1, NB + 1):
    eng = Dense35Own(P, max_ctx=64, module=MOD, debug_layers=n)
    xs = torch.stack(run(eng, SEQ, 'x')).double().cpu().numpy()
    err = np.abs(xs - trace[n - 1]).max(axis=1) / np.abs(trace[n - 1]).max(axis=1)
    kind = 'DeltaNet' if info['kinds'][n - 1] else 'attention'
    check(err.max() < 1e-2 and np.isfinite(xs).all(),
          f'tiny model, the residual after block {n - 1} ({kind}; {", ".join(sorted(set(info["formats"][n - 1].values())))}): '
          f'largest error / largest value {err.max():.1e} over {len(SEQ)} positions (< 1e-2)')
    del eng
eng = Dense35Own(P, max_ctx=64, module=MOD)
base = run(eng, SEQ); g = torch.stack(base).double().cpu().numpy()
kl = [kld(rl[i], g[i]) for i in range(len(SEQ))]; top = sum(int(np.argmax(rl[i]) == np.argmax(g[i])) for i in range(len(SEQ)))
check(np.mean(kl) < 1e-3, f'tiny model, logits: mean KLD {np.mean(kl):.1e} (max {np.max(kl):.1e}) < 1e-3; top-1 equal at {top} of {len(SEQ)}; '
      f'max |dlogit| {np.abs(rl - g).max():.1e} of {np.abs(rl).max():.1f}')
check(all(torch.equal(a, b) for a, b in zip(base, run(eng, SEQ))) and not eng.bails, f'a second run after reset(): bitwise; no wait gave up ({len(eng.bails)})')
again = eng.verify(SEQ, len(SEQ) - 1, 0)
check(torch.equal(again[0], base[-1]), 'the last position run again without committing it: bitwise (the recurrent states commit lazily)')
eng.reset(); eng.step(SEQ[0])
try: eng.verify(SEQ, 64, 1); refused = False
except k3.ContextBoundsError: refused = True
check(refused and not eng.poisoned, 'a position past the KV cache is refused on the host')
def run_lf(e, seq):
    """the device sampler's pass: launch_tokens then finish_pass at each position (the previous one committed), the logits
    read after each pass (K3Session._sampled_pass's order)"""
    e.reset(); out = []
    for i in range(len(seq)):
        c = 0 if i == 0 else 1
        e.launch_tokens(seq, i, c); e.finish_pass(i, c)
        out.append(e.logits.reshape(-1).clone())
    torch.cuda.synchronize(); return out


check(all(torch.equal(a.reshape(-1), b) for a, b in zip(base, run_lf(eng, SEQ))) and not eng.bails,
      f'launch_tokens + finish_pass (the device sampler\'s pass): bitwise the step() logits at every position')
for pw in (3, 6, 7):
    e2 = Dense35Own(P, max_ctx=64, module=MOD, per_wgp=pw)
    check(all(torch.equal(a, b) for a, b in zip(base, run(e2, SEQ))), f'the tape at {pw} workgroups per WGP: bitwise the default grid ({eng.per_wgp})')
    check(all(torch.equal(a.reshape(-1), b) for a, b in zip(base, run_lf(e2, SEQ))),
          f'launch_tokens + finish_pass at {pw} workgroups per WGP: bitwise the default grid\'s step() logits')
    del e2
e3 = Dense35Own(P, max_ctx=64, module=MOD, gu=False)
d3 = max(float((a - b).abs().max()) for a, b in zip(base, run(e3, SEQ)))
print(f'  (gate | up as three segments against the fused segment: max |dlogit| {d3:.1e}; reported, not gated)', flush=True)
del e3, eng, base
gc.collect(); torch.cuda.empty_cache()

# ---- the model file ------------------------------------------------------------------------------------------------------
if A.model:
    z = np.load(A.ref); seq = [int(v) for v in z['seq']]
    REFL, RESID = z['logits'], z['resid']   # (read once: an npz member is re-read from the file at every access)
    NP = min(A.positions, len(REFL), A.max_ctx)
    for n in A.depths:
        t0 = time.time(); eng = Dense35Own(A.model, max_ctx=A.max_ctx, module=MOD, debug_layers=n)
        m = min(A.depth_positions, NP)
        xs = torch.stack(run(eng, seq[:m], 'x')).double().cpu().numpy(); r = RESID[n - 1, :m].astype(np.float64)
        err = np.abs(xs - r).max(axis=1) / np.abs(r).max(axis=1)
        check(err.max() < 1e-2 and np.isfinite(xs).all(), f'the residual after block {n - 1}: largest error / largest value {err.max():.1e} '
              f'over {m} positions (< 1e-2; {time.time() - t0:.0f} s)')
        del eng, xs; gc.collect(); torch.cuda.empty_cache()
    t0 = time.time(); eng = Dense35Own(A.model, max_ctx=A.max_ctx, module=MOD)
    print(f'loaded in {time.time() - t0:.0f} s ({eng.vram_mib:.0f} MiB projected; {eng.nl} blocks; grid {eng.per_wgp} per WGP; '
          f'{len(eng.E.segs)} segments: {sorted(eng.E.inventory().items())})', flush=True)
    # this part's own buffers: the logits of every compared position kept on the card twice, a tape per compared grid
    own = 2 * NP * eng.vocab * 4 / 2 ** 20 + 3 * Dense35Own.TAPE_MIB + 2
    capacity.check([eng.dev], [own], f'test_q35_dense: its own buffers ({NP} positions of logits x 2: {own:.0f} MiB)')
    ids_chk = [0, 1, 785, 6722, 9625, eng.vocab - 1]
    e2 = k3.Engine(module=MOD); tk_d = torch.zeros(len(ids_chk), dtype=torch.int32, device='cuda'); xs = torch.zeros(len(ids_chk), eng.D, device='cuda')
    for j, t in enumerate(ids_chk): tk_d[j] = t; e2.own_emb(eng.emb_dev, 'Q4_K', eng.vocab, eng.D, tk_d[j:j + 1], xs[j])
    e2.build().run(); torch.cuda.synchronize()
    check(all(np.array_equal(xs[j].cpu().numpy().view(np.uint32), own_dequant.rows(eng.emb, t, t + 1).reshape(-1).view(np.uint32))
              for j, t in enumerate(ids_chk)), f'device embedding rows bitwise the host decoder ({len(ids_chk)} tokens)')
    del e2
    eng.reset(); base, ms, kl, top, dmax, miss = [], [], [], 0, 0.0, []
    for i in range(NP):
        t1 = time.perf_counter(); lg = eng.step(seq[i]); torch.cuda.synchronize(); ms.append((time.perf_counter() - t1) * 1e3)
        base.append(lg.clone()); gg = lg.double().cpu().numpy(); r = REFL[i].astype(np.float64)
        kl.append(kld(r, gg)); dmax = max(dmax, float(np.abs(r - gg).max()))
        if int(np.argmax(r)) == int(np.argmax(gg)): top += 1
        else: s2 = np.sort(r)[-2:]; miss.append((i, float(s2[1] - s2[0])))
    check(np.mean(kl) < 1e-3, f'{NP} positions: mean KLD(ref || gpu) {np.mean(kl):.2e} (max {np.max(kl):.2e}) < 1e-3; max |dlogit| {dmax:.3f}')
    check(top >= NP - max(1, NP // 100), f'top-1 equal at {top} of {NP} positions' +
          (f' (the others, position: the reference\'s margin: {", ".join(f"{i}: {m:.3f}" for i, m in miss)})' if miss else ''))
    check(not eng.bails, f'no wait gave up ({len(eng.bails)})')
    bound = eng.physics_bytes()[0]
    print(f'  decode, ms per token over positions {NP // 2} .. {NP - 1}: median {np.median(ms[NP // 2:]):.2f} (each step synchronized; '
          f'reported, not a speed claim: the grid and task sizes are not measured for this model); one token reads {bound / 1e9:.2f} GB of weights',
          flush=True)
    m = min(4, NP); pw0 = eng.per_wgp
    for pw in (3, 6, 7):
        eng.per_wgp = pw
        with torch.cuda.device(eng.dev): eng._build_tape()
        check(all(torch.equal(a, b) for a, b in zip(base[:m], run(eng, seq[:m]))), f'the tape at {pw} workgroups per WGP: bitwise the default grid ({pw0}) on {m} positions')
    eng.per_wgp = pw0
print('RESULT', 'PASS' if ok else 'FAIL')
