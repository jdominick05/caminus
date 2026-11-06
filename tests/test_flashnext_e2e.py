"""Flash-Next (REAP-308) end to end in the K3 engine vs llama.cpp: for each text, the same token sequence (teacher
forced, from a tools/gg_golden decode-mode dump, q8_0 KV cache) runs through caminus.flashnext.FlashNextK3 (layer split
over the two cards); per position: KLD(llama.cpp || engine) over the full 248k vocabulary and top-1 agreement. If a
batch-mode dump of the same text exists beside it (DIR with _decode -> _batch), llama.cpp's own batch-vs-decode noise on
the same tokens is reported next to it (the floor: two llama.cpp runs that differ only in kernels and rounding). At each
top-1 disagreement the reference's top-1 / top-2 logit gap is printed (rounding noise flips only near-ties).
D-281 bar: over all texts, mean KLD <= llama.cpp's own batch-vs-decode mean KLD and top-1 >= that floor's top-1 - 1 point
(needs the batch dumps; without them the old absolute bar, mean KLD <= 0.001 and top-1 >= 99%, is applied).
  python tests/test_flashnext_e2e.py DECODE_DIR [DECODE_DIR ...] [--model model.gguf] [--split 26] [--tp]
(--tp: the two-card tensor-parallel engine, caminus.flashnext_tp, with a replica check after every position)"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus.flashnext import FlashNextK3

from caminus import devpaths  # noqa: E402
args = sys.argv[1:]; PATH = devpaths.reap(); SPLIT = 26
if '--model' in args: i = args.index('--model'); PATH = args[i + 1]; del args[i:i + 2]
if '--split' in args: i = args.index('--split'); SPLIT = int(args[i + 1]); del args[i:i + 2]
TP = '--tp' in args
if TP: args.remove('--tp')
def logp(x): x = x - x.max(); return x - np.log(np.exp(x).sum())
def kld(ref, oth):
    lp, lq = logp(ref), logp(oth); return float((np.exp(lp) * (lp - lq)).sum())
def outputs(GD):
    idx = [l.rstrip('\n').split('\t') for l in open(f'{GD}/index.tsv')]
    rs = [r for r in idx if r[2] == 'result_output']
    if len(rs) == 1 and int(rs[0][5]) > 1:      # batch mode: one tensor [positions, vocab]
        return np.fromfile(f'{GD}/{rs[0][8]}', dtype=np.float32).reshape(int(rs[0][5]), -1)
    return {int(r[0]): r for r in rs}, GD
for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.95 if d == 1 else 0.97, d)   # XTX: the fixed 0.5 GB reserve
if TP:
    from caminus.flashnext_tp import FlashNextTP
t0 = time.time(); m = FlashNextTP(PATH) if TP else FlashNextK3(PATH, split=SPLIT); print(f'loaded in {time.time() - t0:.0f} s')
rep_bad = 0
allk, alla, allf, allfa = [], [], [], []
for GD in args:
    tokens = [int(v) for v in open(f'{GD}/tokens.txt').read().split()]
    dec, _ = outputs(GD)
    BD = GD.replace('_decode', '_batch'); batch = outputs(BD) if BD != GD and os.path.isdir(BD) else None
    m.reset(); kls, ag, fk, fa, flips = [], [], [], [], []
    for i in range(len(tokens)):
        lo = m.step(tokens, i).double().cpu().numpy()
        if TP and m.check_replicas() != (True, True): rep_bad += 1
        ref = np.fromfile(f'{GD}/{dec[i][8]}', dtype=np.float32).astype(np.float64)
        kls.append(kld(ref, lo)); ag.append(int(lo.argmax() == ref.argmax()))
        if not ag[-1]: s = np.sort(ref)[::-1]; flips.append(f'{i}:{s[0] - s[1]:.2f}')
        if batch is not None:
            b = batch[i].astype(np.float64); fk.append(kld(ref, b)); fa.append(int(b.argmax() == ref.argmax()))
    allk += kls; alla += ag; allf += fk; allfa += fa
    line = f'{os.path.basename(GD)}: {len(tokens)} positions, engine vs llama.cpp: mean KLD {np.mean(kls):.4f}, top-1 {100 * np.mean(ag):.1f}%'
    if fk: line += f' | llama.cpp batch vs decode: mean KLD {np.mean(fk):.4f}, top-1 {100 * np.mean(fa):.1f}%'
    print(line); print(f'   engine top-1 flips (position: reference top-1 - top-2 logit gap): {" ".join(flips) or "none"}')
mk, ag = float(np.mean(allk)), 100 * float(np.mean(alla))
if TP: print(f'replica mismatches (residual or router ids differ between the cards): {rep_bad}')
print(f'ALL: {len(allk)} positions, engine: mean KLD {mk:.4f}, top-1 {ag:.1f}%' + (f' | llama.cpp floor: mean KLD {np.mean(allf):.4f}, top-1 {100 * np.mean(allfa):.1f}%' if allf else ''))
if allf and len(allf) == len(allk):
    fk, fa = float(np.mean(allf)), 100 * float(np.mean(allfa))
    ok = mk <= fk and ag >= fa - 1.0 and rep_bad == 0
    print(f'{"PASS" if ok else "FAIL"} (D-281: mean KLD {mk:.4f} <= floor {fk:.4f}, top-1 {ag:.1f}% >= {fa - 1.0:.1f}%)')
else:
    ok = mk <= 1e-3 and ag >= 99.0
    print(f'{"PASS" if ok else "FAIL"} (no floor dumps: absolute bar, mean KLD <= 0.001 and top-1 >= 99%)')
sys.exit(0 if ok else 1)
