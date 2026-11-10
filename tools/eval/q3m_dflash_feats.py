"""The DFlash drafter's offline acceptance, phase 1 (the target only, one card): greedy generations of six public prompts
(400 tokens each), then the batched prefill of each whole sequence with the residual x recorded entering every layer (after
the add at its attention norm): OUT_<i>.npz = seq, n_prompt, fa = x entering layers 1 / 12 / 23 / 34 / 45 (the drafter's
features), fb = entering 2 / 13 / 24 / 35 / 46 (the other convention, for the comparison). Phase 2:
tools/eval/q3m_dflash_accept.py.
  tools/guard.sh LOG 900 python -I tools/eval/q3m_dflash_feats.py GGUF OUT"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus import ggufio
from caminus.bpe import BPE
from caminus.engine.q3moe_one import Q3MoEOne
from caminus.prefill.q3m import Q3MPrefill
path = sys.argv[1]; out = sys.argv[2]
tk = BPE(ggufio.GGUFFile(path).meta)
eng = Q3MoEOne(path, max_ctx=2048); pf = Q3MPrefill(eng, chunk=1024)
prompts = ["Write a Python function that parses a CSV file of orders and returns the total revenue per customer, with error handling.",
           "Summarize the causes of the French Revolution in a few paragraphs.",
           "What is 37 * 43? Think it through step by step.",
           "Implement binary search in C and explain its complexity.",
           "Translate into French: The weather was cold, so we stayed inside and read books all afternoon.",
           "Give me three ideas for a weekend trip near the mountains, with a short plan for each."]
A, B = [1, 12, 23, 34, 45], [2, 13, 24, 35, 46]
D = eng.D
for pi, ptxt in enumerate(prompts):
    ids = tk.encode(f"<|im_start|>user\n{ptxt}<|im_end|>\n<|im_start|>assistant\n")
    eng.reset(); pf.run(ids, 0, 0, len(ids) - 1); eng.pos_host = len(ids) - 1
    gen = []; t = ids[-1]
    for i in range(400):
        t = int(torch.argmax(eng.step(t))); gen.append(t)
    seq = ids + gen; n = len(seq)
    rec = {}
    fa = pf.ops.add_rms16; cnt = [0]
    def add_rms16(x, nw, eps, h16, nn, N, d=None, w=None, S=0, h=None):
        fa(x, nw, eps, h16, nn, N, d=d, w=w, S=S, h=h)
        if h is None:                        # (the attention norm of layer cnt: x entering that layer)
            il = cnt[0]; cnt[0] += 1
            if il in A or il in B: rec[il] = x[:nn * D].view(nn, D).clone()
    pf.ops.add_rms16 = add_rms16
    eng.reset(); pf.run(seq, 0, 0, n); torch.cuda.synchronize()
    del pf.ops.add_rms16
    fa_ = torch.cat([rec[l] for l in A], 1).cpu().numpy(); fb_ = torch.cat([rec[l] for l in B], 1).cpu().numpy()
    np.savez(f'{out}_{pi}.npz', seq=np.array(seq), n_prompt=len(ids), fa=fa_, fb=fb_)
    print(f'prompt {pi}: {len(ids)} + {len(gen)}: {tk.decode(gen[:40])!r}', flush=True)
print('RESULT DONE')
