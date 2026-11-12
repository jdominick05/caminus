"""Dense35Verify's feature export (a DFlash drafter's target layers: the residual entering each layer, llama.cpp's layer
input) on the dense qwen35 line (Swift 1.5, kernels/k3_q35_swift_ver.hip), bit for bit:
  1. the logits of a T = 4 pass with the export on equal the same pass with it off (the copies change nothing else);
  2. the own prefill's export (Q35OwnPrefill(feature_layers=)) over the same tokens is close to the verify export at the
     same positions: not bitwise, by design (the prefill's GEMMs sum in WMMA order; the copy itself is exact). Reported:
     the worst relative error and cosine per layer, checked against a loose bound;
  3. each exported row equals the residual a decode engine cut after the layer before computes at that position
     (Dense35Own(debug_layers=L): the first L blocks, then the output norm's add, so its x after a step is the residual
     entering block L), one cut engine at a time after the full engine is freed.
  4. the export on 3, 6 and 7 workgroups per WGP: the same rows, bitwise (the copy is ordered before the layer's later
     writes of x at every grid);
  5. the decode tape's own export (Dense35Own(feature_layers=): one-token passes feed a DFlash drafter too): each step's
     logits bitwise the verify pass's row (as before the export) and its exported rows bitwise the verify export's;
  6. --time-ctx P (with --time-only: alone, in its own process): the verify pass's cost with the export on vs off at a P-position context (prefilled), interleaved,
     the median of 7 each (a decode measurement; no check).
Context: decode steps from position 0 (no prefill), then the pass. Prints PASS / FAIL lines and RESULT PASS | FAIL.
  tools/guard.sh LOG 1800 python -I tests/engine/test_q35_features.py GGUF [--layers 6,20,34,48,62] [--n 16]
The export adds no device code (the module's add-rms op on a zeroed buffer); the first run still goes to a non-display card
alone (HIP_VISIBLE_DEVICES=<its index>) when the card fits the model, as the gates do."""
import argparse, gc, glob, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'python'))
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('gguf'); ap.add_argument('--layers', default='6,20,34,48,62'); ap.add_argument('--n', type=int, default=16)
ap.add_argument('--time-ctx', type=int, default=0)
ap.add_argument('--time-only', action='store_true', help='only the --time-ctx measurement (its own process: no phase before it)')
A = ap.parse_args()
from caminus import ggufio
from caminus.bpe import BPE
from caminus.engine.q35_dense import Dense35Own
from caminus.engine.q35_dense_verify import Dense35Verify, MAX_T
from caminus.prefill.q35own import Q35OwnPrefill, buffer_need_mib, shapes_of

layers = [int(x) for x in A.layers.split(',')]
N, T = A.n, MAX_T
tk = BPE(ggufio.GGUFFile(A.gguf).meta)
text = ''.join(open(f).read() for f in sorted(glob.glob(os.path.join(ROOT, 'docs', '*.md'))))
full = tk.encode(text)[:N + T]
max_ctx = N + T + 8
print(torch.cuda.get_device_name(0), flush=True)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


bits = lambda t: t.detach().reshape(-1).cpu().numpy().view(np.uint32).copy()


def context(eng):
    """decode steps over the first N tokens from an empty engine"""
    eng.reset()
    for p in range(N): eng.verify(full, p, 0 if p == 0 else 1)


nf = len(layers)
if not A.time_only:
    CH = 64
    GRIDS = (3, 6, 7)
    eng = Dense35Own(A.gguf, max_ctx=max_ctx, spec_T=T, feature_layers=layers,
                     vram_extra_mib=Dense35Verify.need_mib(T, max_ctx=max_ctx) + Dense35Verify.need_mib(T, max_ctx=max_ctx, nfeat=nf)
                     + buffer_need_mib(shapes_of(A.gguf), CH, nfeat=nf) + 16)
    plain = Dense35Verify(eng, T)
    exp = Dense35Verify(eng, T, feature_layers=layers)
    print(f'verify T {T}: {len(plain.E.segs)} segments plain, {len(exp.E.segs)} with {nf} exported layers ({exp.mib - plain.mib:.1f} MiB more)', flush=True)
    context(eng); plain.run(full[N:N + T], N, 1, argmax=False)
    lp = [bits(plain.logits(r)) for r in range(T)]
    context(eng); exp.run(full[N:N + T], N, 1, argmax=False)
    le = [bits(exp.logits(r)) for r in range(T)]
    check(all(np.array_equal(a, b) for a, b in zip(lp, le)), f'the {T} rows\' logits with the export on == off (bitwise)')
    feats = exp.features.detach().cpu().numpy().copy()
    check(np.isfinite(feats).all() and all(np.abs(feats[k]).max() > 0 for k in range(nf)), 'every exported row is finite and non-zero')
    # a second pass on the same tape: the buffer is zeroed before each pass (the copies add into it), so the rows repeat
    context(eng); exp.run(full[N:N + T], N, 1, argmax=False)
    check(np.array_equal(exp.features.detach().cpu().numpy(), feats), 'a second identical pass exports the same rows (zeroed per pass)')
    # the decode tape's export: steps at N .. N + T - 1 against the verify pass's rows (the verify pass is bitwise the steps)
    context(eng); sb, sl = [], []
    for r in range(T):
        lg = eng.verify(full, N + r, 1); sl.append(bits(lg)); sb.append(eng.features[:, 0].detach().cpu().numpy().copy())
    check(all(np.array_equal(sl[r], le[r]) for r in range(T)), f'the decode tape with its export on: each step\'s logits == the verify row (bitwise, {T} steps)')
    check(all(np.array_equal(sb[r], feats[:, r]) for r in range(T)), f'the decode tape\'s exported rows == the verify export\'s (bitwise, {T} steps)')
    pf = Q35OwnPrefill(eng, chunk=CH, crossover=1, feature_layers=layers)
    eng.reset(); pf.run(full[:N + T], 0, 0, N + T); torch.cuda.synchronize()
    pfe = pf.features[:, N:N + T].detach().cpu().numpy().copy()
    for k, L in enumerate(layers):
        a, b = pfe[k].astype(np.float64), feats[k].astype(np.float64)
        rel = float(np.abs(a - b).max() / np.abs(b).max())
        cos = float(min((a[r] @ b[r]) / np.linalg.norm(a[r]) / np.linalg.norm(b[r]) for r in range(T)))
        check(rel < 0.05 and cos > 0.999, f'layer {L}: the prefill export vs the verify export at positions {N} .. {N + T - 1}: '
                                          f'worst |diff| / max {rel:.2e}, worst cosine {cos:.6f} (WMMA GEMMs: close, not bitwise)')
    del plain, exp, pf, eng; gc.collect(); torch.cuda.empty_cache()

    for k, L in enumerate(layers):
        cut = Dense35Own(A.gguf, max_ctx=max_ctx, debug_layers=L)
        context(cut)
        bad = []
        for r in range(T):
            cut.verify(full, N + r, 1)
            if not np.array_equal(bits(cut.buf['x']), feats[k, r].view(np.uint32)): bad.append(r)
        check(not bad, f'layer {L}: the exported rows == the residual entering block {L} (a decode engine cut after {L} blocks), '
                       f'{T} positions' + (f'; rows {bad} differ' if bad else ''))
        del cut; gc.collect(); torch.cuda.empty_cache()
    # the grids last, on a fresh engine (the cut engines above needed the first one's memory back)
    eng = Dense35Own(A.gguf, max_ctx=max_ctx, spec_T=T, vram_extra_mib=len(GRIDS) * Dense35Verify.need_mib(T, max_ctx=max_ctx, nfeat=nf) + 16)
    for w in GRIDS:
        ew = Dense35Verify(eng, T, per_wgp=w, feature_layers=layers)
        context(eng); ew.run(full[N:N + T], N, 1, argmax=False)
        check(np.array_equal(ew.features.detach().cpu().numpy(), feats) and all(np.array_equal(bits(ew.logits(r)), le[r]) for r in range(T)),
              f'{w} workgroups per WGP: the same exported rows and logits, bitwise')
    del ew, eng; gc.collect(); torch.cuda.empty_cache()
if A.time_ctx:
    import time
    P = A.time_ctx
    ids = (tk.encode(text) * (P // max(1, len(tk.encode(text))) + 2))[:P + T]
    mc = P + T + 8
    eng = Dense35Own(A.gguf, max_ctx=mc, spec_T=T, vram_extra_mib=Dense35Verify.need_mib(T, max_ctx=mc)
                     + Dense35Verify.need_mib(T, max_ctx=mc, nfeat=nf) + buffer_need_mib(shapes_of(A.gguf), 1024) + 16)
    plain, exp = Dense35Verify(eng, T), Dense35Verify(eng, T, feature_layers=layers)
    pf = Q35OwnPrefill(eng, chunk=1024)
    eng.reset(); pf.run(ids[:P], 0, 0, P); torch.cuda.synchronize()
    tm = {'off': [], 'on': []}
    for i in range(16):                  # (commit 0 every pass: the states stay the prefill's; the first 2 rounds warm up)
        for name, v in (('off', plain), ('on', exp)):
            t0 = time.perf_counter(); v.run(ids[P:P + T], P, 0, argmax=False); torch.cuda.synchronize()
            if i >= 2: tm[name].append(time.perf_counter() - t0)
    med = {k: 1e3 * float(np.median(v)) for k, v in tm.items()}
    print(f'verify T {T} at {P} positions: export off {med["off"]:.3f} ms, on {med["on"]:.3f} ms (median of {len(tm["on"])} '
          f'interleaved; +{1e3 * (med["on"] - med["off"]):.0f} us)', flush=True)
print('RESULT ' + ('PASS' if ok else 'FAIL'), flush=True)
sys.exit(0 if ok else 1)
