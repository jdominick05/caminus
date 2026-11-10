"""Q3MoEOne (qwen3moe on one card, Caminus's own code from the GGUF to the logits) against caminus.ref_q3moe (the float64
host forward on the same reader, decoders and tokenizer): the same token sequence (the prompt, then the reference's own
greedy continuation) through both; per position top-1 agreement, KLD(ref || gpu) and max |logit difference|; then the
engine's own greedy text and its decode speed. Prints PASS / FAIL lines and RESULT PASS | FAIL.
Gate (spec-own-quant.md, first form): top-1 equal at every position, mean KLD < 1e-3.
  tools/guard.sh LOG 1800 python -I tests/engine/test_q3moe_one.py GGUF --ref-cache FILE.npz [--prompt TEXT] [--new 3]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('gguf'); ap.add_argument('--ref-cache', required=True)
ap.add_argument('--prompt', default='The capital of France is'); ap.add_argument('--new', type=int, default=3)
ap.add_argument('--ref-only', action='store_true', help='compute / check the reference cache only (CPU)')
ap.add_argument('--speed', type=int, default=64, help='decode steps timed after the check')
A = ap.parse_args()
from caminus import ggufio
from caminus.bpe import BPE
tk = BPE(ggufio.GGUFFile(A.gguf).meta)
ids = tk.encode(A.prompt)
if os.path.exists(A.ref_cache):
    z = np.load(A.ref_cache); seq, ref = [int(v) for v in z['seq']], z['logits']
    assert seq[:len(ids)] == ids, 'the reference cache is for another prompt'
else:
    from caminus.ref_q3moe import Q3MoERef
    m = Q3MoERef(A.gguf); seq, ref = list(ids), []
    t0 = time.time()
    for i in range(len(ids) + A.new - 1):
        lg = m.step(seq[i]); ref.append(lg)
        if i >= len(ids) - 1: seq.append(int(np.argmax(lg)))
    ref = np.stack(ref); np.savez(A.ref_cache, seq=np.array(seq), logits=ref)
    print(f'reference: {len(seq)} tokens in {time.time() - t0:.0f} s: {A.prompt!r} + {tk.decode(seq[len(ids):])!r}', flush=True)
if A.ref_only: sys.exit(0)

import torch
from caminus.engine.q3moe_one import Q3MoEOne
print(torch.cuda.get_device_name(0), flush=True)
t0 = time.time(); eng = Q3MoEOne(A.gguf, max_ctx=512); print(f'loaded in {time.time() - t0:.0f} s ({eng.vram_mib:.0f} MiB)', flush=True)
ok = True
# the device embedding row (OP_OWN_EMB) bitwise our host decoder, on a spread of token ids
from caminus import own_dequant
from caminus.engine import k3
ids_chk = [0, 1, 785, 6722, 9625, 151642, 151935]
e2 = k3.Engine(module=eng.module); tk_d = torch.zeros(len(ids_chk), dtype=torch.int32, device='cuda')
xs = torch.zeros(len(ids_chk), eng.D, device='cuda')
for j, t in enumerate(ids_chk): tk_d[j] = t; e2.own_emb(eng.emb_dev, 'Q4_K', eng.vocab, eng.D, tk_d[j:j + 1], xs[j])
e2.build().run(); torch.cuda.synchronize()
same = all(np.array_equal(xs[j].cpu().numpy().view(np.uint32), own_dequant.rows(eng.emb, t, t + 1).reshape(-1).view(np.uint32))
           for j, t in enumerate(ids_chk))
print(('PASS' if same else 'FAIL') + f' device embedding rows bitwise the host decoder ({len(ids_chk)} tokens)'); ok &= same
def lsm(v): v = v - v.max(); return v - np.log(np.exp(v).sum())
klds, top_ok = [], True
for i in range(len(ref)):
    g = eng.step(seq[i]).double().cpu().numpy(); r = ref[i]
    lr, lgp = lsm(r), lsm(g); kld = float(np.sum(np.exp(lr) * (lr - lgp)))
    t1r, t1g = int(np.argmax(r)), int(np.argmax(g)); top_ok &= t1r == t1g; klds.append(kld)
    print(f'  pos {i}: in {tk.decode([seq[i]])!r:12s} top-1 ref {tk.decode([t1r])!r} gpu {tk.decode([t1g])!r}  '
          f'KLD {kld:.2e}  max |dlogit| {np.abs(r - g).max():.3e}', flush=True)
print(('PASS' if top_ok else 'FAIL') + ' top-1 equal at every position'); ok &= top_ok
print(('PASS' if np.mean(klds) < 1e-3 else 'FAIL') + f' mean KLD {np.mean(klds):.2e} (max {np.max(klds):.2e}) < 1e-3'); ok &= np.mean(klds) < 1e-3
# G-self across grids: the tape at 3 / 6 / 7 workgroups per WGP gives the resident grid's logits bitwise (same weights, same
# KV caches rewritten position by position)
def run_seq():
    eng.reset(); return [eng.step(t).clone() for t in seq[:len(ref)]]
base = run_seq(); keep = (eng.E, eng.S, eng.dstage, eng.hstage, eng.tok, eng.pos); gs_ok = True
for pw in (3, 6, 7):
    eng.per_wgp = pw
    with torch.cuda.device(eng.dev): eng._build_tape()
    gs_ok &= all(torch.equal(a, b) for a, b in zip(base, run_seq()))
eng.per_wgp = None; eng.E, eng.S, eng.dstage, eng.hstage, eng.tok, eng.pos = keep
print(('PASS' if gs_ok else 'FAIL') + f' logits bitwise the resident grid\'s at 3 / 6 / 7 workgroups per WGP ({len(ref)} positions)'); ok &= gs_ok
# the engine's own greedy text, from the prompt
eng.reset()
for t in ids: lg = eng.step(t)
out = []
for _ in range(max(A.new, 12)):
    nxt = int(torch.argmax(lg)); out.append(nxt)
    if nxt == tk.eos: break
    lg = eng.step(nxt)
print(f'gpu greedy: {A.prompt!r} + {tk.decode(out)!r}', flush=True)
# decode speed: steps of its own argmax, each one launch + the host embedding row + a sync
ts = []
for _ in range(A.speed):
    t1 = time.perf_counter(); lg = eng.step(int(torch.argmax(lg))); ts.append(time.perf_counter() - t1)
ms = 1e3 * float(np.median(ts))
print(f'decode: median {ms:.3f} ms/token = {1e3 / ms:.1f} tok/s over {A.speed} steps (pos {eng.pos_host}; host embedding + launch + sync)')
print('RESULT', 'PASS' if ok else 'FAIL')
