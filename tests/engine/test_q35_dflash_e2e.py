"""Swift 1.5 with its own DFlash 2 drafter (engine/q35_dflash.py) on one card, the host's accept, against plain greedy
decoding on the same engine: the drafted run's tokens must equal the plain run's, token for token (the verify pass decides
every token; the drafts only change how many each pass keeps). Reports the drafts' acceptance and both runs' speed (decode,
ENGINE level: Python loop, no serve, no prompt cache; not a served number).
The round: prefill the prompt but its last token (Q35OwnPrefill exporting the drafter's target layers), inject those
features; then per pass at position i0 (`last` = the token there): the drafter's 3 drafts, one verify pass of
[last, d1, d2, d3] (Dense35Verify T 4, exporting the features of its rows), keep d_j while it equals the greedy token after
row j - 1, inject the kept rows' features, the bonus token is the next `last`.
  tools/guard.sh LOG 1800 python -I tests/engine/test_q35_dflash_e2e.py SWIFT_GGUF DFLASH_GGUF [--n 128] [--texts code,prose,shell]
Texts: the 48-token prompts the llama.cpp yardstick ran (--prompts JSON: accept_ids.json's 'prompts'), else this repository's docs."""
import argparse, json, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'python'))
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('gguf'); ap.add_argument('drafter'); ap.add_argument('--n', type=int, default=128)
ap.add_argument('--prompts', default=None); ap.add_argument('--texts', default='code,prose,shell')
A = ap.parse_args()
from caminus import ggufio
from caminus.bpe import BPE
from caminus.engine.q35_dense import Dense35Own
from caminus.engine.q35_dense_verify import Dense35Verify
from caminus.engine.q35_dflash import Dflash35
from caminus.prefill.q35own import Q35OwnPrefill, buffer_need_mib, shapes_of

T = 4
if A.prompts: prompts = {k: v for k, v in json.load(open(A.prompts))['prompts'].items() if k in A.texts.split(',')}
else:
    tk = BPE(ggufio.GGUFFile(A.gguf).meta)
    prompts = {'docs': tk.encode(open(os.path.join(ROOT, 'docs', 'quickstart.md')).read())[:256]}
max_ctx = max(len(v) for v in prompts.values()) + A.n + 2 * T + 8
print(torch.cuda.get_device_name(0), flush=True)
dfm = Dflash35.need_mib(A.drafter, int(ggufio.GGUFFile(A.gguf).T['output.weight'].shape[1]))
layers = None
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


eng = Dense35Own(A.gguf, max_ctx=max_ctx, spec_T=T, vram_extra_mib=dfm + Dense35Verify.need_mib(T, max_ctx=max_ctx, nfeat=5)
                 + buffer_need_mib(shapes_of(A.gguf), 256, nfeat=5) + 32)
dr = Dflash35(A.drafter, eng)
layers = dr.layers_tgt
ver = Dense35Verify(eng, T, feature_layers=layers)
pf = Q35OwnPrefill(eng, chunk=256, crossover=1, feature_layers=layers)


def prefill(ids):
    """prefill ids[:-1]; returns (i0, last): the next pass's position and token"""
    eng.reset()
    n = len(ids) - 1
    assert n <= pf.chunk, 'the test prompt fits one prefill chunk (its features are the last chunk\'s)'
    pf.run(ids[:n], 0, 0, n); torch.cuda.synchronize()
    return n, ids[n]


def plain(ids):
    i0, last = prefill(ids)
    out, commit = [], 0
    t0 = time.perf_counter()
    full = list(ids)
    for _ in range(A.n):
        lg = eng.verify(full, i0, commit); nxt = int(torch.argmax(lg).item())
        out.append(nxt); full.append(nxt); i0 += 1; commit = 1
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def drafted(ids):
    i0, last = prefill(ids)
    dr.inject(0, pf.features[:, :i0])
    out, commit, passes, acc = [], 0, 0, 0
    t0 = time.perf_counter()
    while len(out) < A.n:
        d = dr.draft(last, i0)
        g = ver.run([last] + d, i0, commit)            # g[j] = the greedy token after row j
        a = 0
        while a < T - 1 and d[a] == g[a]: a += 1
        keep = g[:a + 1]                               # the a accepted drafts (== g[0 .. a-1]) and the bonus g[a]
        out += keep; passes += 1; acc += a
        dr.inject(i0, ver.features[:, :a + 1])         # rows 0 .. a hold real tokens: last, d1 .. d_a
        i0 += a + 1; last = keep[-1]; commit = a + 1
    torch.cuda.synchronize()
    return out[:A.n], time.perf_counter() - t0, passes, acc


for name, ids in prompts.items():
    ref, tp = plain(ids)
    got, td, passes, acc = drafted(ids)
    first = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), None)
    check(got == ref, f'{name}: {A.n} drafted tokens == plain greedy' + ('' if first is None else f' (first difference at {first})'))
    print(f'  {name}: {passes} passes, {acc} of {3 * passes} drafts accepted ({100 * acc / max(1, 3 * passes):.0f} %), '
          f'{A.n / passes:.2f} tokens per pass; decode (engine level) plain {A.n / tp:.1f} tok/s, drafted {A.n / td:.1f} tok/s '
          f'({tp / td:.2f}x)', flush=True)
print('RESULT ' + ('PASS' if ok else 'FAIL'), flush=True)
sys.exit(0 if ok else 1)
