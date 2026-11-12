"""The float64 DFlash reference (caminus.ref_dflash) against llama.cpp's own drafter tensors (tools/reference/dflash_dump.cpp
on a dense qwen35 target, one card): CPU only, reads a dump directory (index.jsonl, passes.jsonl, the .bin tensors).
  inject: llama.cpp's g rows (inp_g_embeddings) through the reference's K / V path vs its Kcur / Vcur_injected-L
  draft : for the first dumped draft calls, the reference's block (llama.cpp's injected K / V in the cache) vs its per-layer
          l_out-L, result_norm and result_output, and the drafted tokens vs llama.cpp's draft for that pass
Reports the worst relative error per tensor (max |ref - llama| / max |llama|). Not bitwise by design (llama.cpp: fp32 / fp16
math on Q8_0, the reference: float64).
  python -I tools/reference/dflash35_anchor.py DUMP_DIR DRAFTER_GGUF TARGET_GGUF [--drafts 3]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
from caminus.ref_dflash import RefDFlash

ap = argparse.ArgumentParser(); ap.add_argument('dump'); ap.add_argument('drafter'); ap.add_argument('target')
ap.add_argument('--drafts', type=int, default=3)
A = ap.parse_args()
ix = {}
for l in open(os.path.join(A.dump, 'index.jsonl')):
    r = json.loads(l); ix.setdefault((r['ctx'], r['eval'], r['name']), r)
passes = [json.loads(l) for l in open(os.path.join(A.dump, 'passes.jsonl'))]


def get(ctx, ev, name):
    r = ix[(ctx, ev, name)]; ne = [n for n in r['ne'] if n] or [1]
    return np.fromfile(os.path.join(A.dump, r['file']), dtype=np.float32).reshape(list(reversed(r['ne'])))


rel = lambda a, b: float(np.abs(np.asarray(a, np.float64) - b).max() / max(np.abs(b).max(), 1e-30))
ref = RefDFlash(A.drafter, A.target, max_pos=1024)
worst = {}
def note(k, v):
    worst[k] = max(worst.get(k, 0.0), v)

injects = sorted({ev for (c, ev, n) in ix if c == 'inj'})
lk = {}   # llama's injected K / V per position (the draft's cache)
for e in injects:
    g = get('inj', e, 'inp_g_embeddings')[0, 0]                         # [n, D]
    n = g.shape[0]
    pos0 = 0 if e == injects[0] else passes[injects.index(e) - 1]['pos']
    ref.inject_g(pos0, g)
    for il in range(ref.nl):
        kl = get('inj', e, f'Kcur_injected-{il}')[0].reshape(n, ref.nkv, ref.hd)
        vl = get('inj', e, f'Vcur_injected-{il}')[0].reshape(n, ref.nkv, ref.hd)
        note('inject K', rel(ref.K[il, pos0:pos0 + n], kl)); note('inject V', rel(ref.V[il, pos0:pos0 + n], vl))
        lk[(il, pos0)] = (kl, vl, n)
print(f'{len(injects)} inject calls: worst rel K {worst["inject K"]:.2e}, V {worst["inject V"]:.2e}', flush=True)

for (il, p0), (kl, vl, n) in lk.items():   # the draft compared on llama.cpp's own cache rows (isolates the draft path)
    ref.K[il, p0:p0 + n], ref.V[il, p0:p0 + n] = kl, vl
same = 0
for d in range(min(A.drafts, len(passes))):
    ps = passes[d]; ev = d + 2; k = len(ps["draft"])   # (draft call 1: llama.cpp's 2-token warm-up graph)
    # cache rows are rewritten in call order: restore llama.cpp's rows injected up to this pass
    for (il, p0), (kl, vl, n) in lk.items():
        if p0 <= ps['pos']: ref.K[il, p0:p0 + n], ref.V[il, p0:p0 + n] = kl, vl
    toks, logits, _ = ref.draft(ps['last'], ps['pos'], k)
    for il in range(ref.nl):
        if ('dft', ev, f'l_out-{il}') in ix: note('draft l_out', rel(ref._last_layers[il], get('dft', ev, f'l_out-{il}')[0, 0]))
    if ('dft', ev, 'result_output') in ix: note('draft logits', rel(logits, get('dft', ev, 'result_output')[0, 0]))
    ok = toks == ps['draft']; same += ok
    print(f'pass {d} at {ps["pos"]} (last {ps["last"]}): reference drafts {toks}, llama.cpp {ps["draft"]}: {"equal" if ok else "DIFFER"}', flush=True)
print('worst relative error: ' + ', '.join(f'{k} {v:.2e}' for k, v in worst.items()))
print(f'RESULT drafts equal {same}/{min(A.drafts, len(passes))}')
