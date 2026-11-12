"""Speculative decoding on Swift 1.5 with its own multi-token-prediction block, end to end on one card: greedy decoding
with the drafter (engine/q35_mtp.py, k = 3 drafts a round) and the 4-token verify pass (engine/q35_dense_verify.py),
against plain greedy decoding (Dense35Own, one token per pass) on the same prompts.

Per content class (three texts of this tree: python/caminus/arch.py code, docs/dev/architecture.md prose, build.sh
shell; the first --prompt tokens of each, then --n generated tokens):
  the gate   the speculative loop's tokens are the plain loop's, every one (greedy: a draft is kept only where the
             verify pass's own greedy token equals it, so the output cannot change)
  speed      decode tok/s of each loop over the generated tokens (the prompt not counted), drafts accepted per index,
             tokens per round, and the round's parts: the drafter's round (catch rows + chained steps) and the verify pass.
The prompt goes through 4-token verify passes (their h rows fill the drafter's KV rows), not the batched prefill.
A round: the drafter's catch rows (the last round's accepted drafts and its bonus token, from the verify pass's h rows),
k - 1 chained steps; then the verify pass of [the newest token, d1, d2, d3]; a = the leading drafts equal to the verify
pass's greedy tokens; emitted d1 .. da and the verify pass's token after row a.
  tools/guard.sh LOG 1800 python -I benchmarks/q35_mtp_spec.py GGUF [--prompt 256] [--n 128]
Prints PASS / FAIL and RESULT."""
import argparse, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'python'))
import numpy as np
import torch
from caminus import ggufio
from caminus.bpe import BPE
from caminus.engine.q35_dense import Dense35Own
from caminus.engine.q35_dense_verify import Dense35Verify
from caminus.engine.q35_mtp import Mtp35

ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--prompt', type=int, default=256)
ap.add_argument('--n', type=int, default=128); ap.add_argument('--k', type=int, default=3)
ap.add_argument('--ids', default=None, help='a JSON file {"prompts": {name: [token ids]}} in place of the texts (e.g. the inputs '
                'another engine was measured on); --prompt then cuts each to its first P ids')
A = ap.parse_args()
P, N, K = A.prompt, A.n, A.k
TEXTS = {'code': os.path.join(ROOT, 'python', 'caminus', 'arch.py'), 'prose': os.path.join(ROOT, 'docs', 'dev', 'architecture.md'),
         'shell': os.path.join(ROOT, 'build.sh')}
eng = Dense35Own(A.gguf, max_ctx=P + N + 16, spec_T=4,
                 vram_extra_mib=sum(Dense35Verify.need_mib(t, max_ctx=P + N + 16) for t in (1, 2, 3, 4)) + 400)
D = eng.D
ver = {t: Dense35Verify(eng, t) for t in (1, 2, 3, 4)}
mtp = Mtp35(eng, k=K)
tk = BPE(ggufio.GGUFFile(A.gguf).meta)
print(torch.cuda.get_device_name(eng.dev), os.path.basename(A.gguf), f'prompt {P}, generated {N}, k {K}', flush=True)
bad = 0


def sync(): torch.cuda.synchronize(eng.dev)


def plain(prompt):
    """greedy, one token per pass: the tokens after the prompt and the decode seconds"""
    eng.reset(); toks = list(prompt); t0 = None
    for i in range(P + N - 1):
        lo = eng.verify(toks, i, 0 if i == 0 else 1)
        if i >= P - 1: toks.append(int(torch.argmax(lo.reshape(-1))))
        if i == P - 1: sync(); t0 = time.perf_counter()   # (timed: the N - 1 tokens after the first, as the speculative loop)
    sync()
    return toks[P:], time.perf_counter() - t0


def prompt_pass(prompt):
    """the prompt through verify passes (4 tokens each, the rest 1 .. 3), the drafter's rows 1 .. P - 1 filled from their
    h rows; returns the first generated token and the h row of position P - 1 (a copy)"""
    eng.reset(); mtp.reset()
    a, last_T = 0, 0
    while a < P:
        T = min(4, P - a)
        top = ver[T].run(prompt[a:a + T], a, last_T)
        h = ver[T].buf['h'][:T * D].view(T, D)
        nf = min(T, P - 1 - a)                       # rows a + 1 .. a + nf (the drafter's row j: h of j - 1, token j)
        if nf > 0: mtp.fill(h[:nf], prompt[a + 1:a + 1 + nf], a + 1)
        a += T; last_T = T
    return top[-1], h[T - 1:T].clone(), last_T


def spec(prompt):
    first, h0, last_T = prompt_pass(prompt)
    sync(); t0 = time.perf_counter()
    out = [first]; p = P                             # out[-1] = the token at position p, not yet in the trunk
    hrows, rtoks, p0 = h0, [first], P                # the drafter's next catch rows: h rows, their tokens, the first position
    acc = [0] * K; offered = [0] * K; rounds = 0; t_draft = 0.0; t_ver = 0.0
    while len(out) < N:
        ta = time.perf_counter()
        d = mtp.draft(hrows, rtoks, p0, k=K)
        tb = time.perf_counter()
        top = ver[K + 1].run([out[-1]] + d, p, last_T)
        tc = time.perf_counter(); t_draft += tb - ta; t_ver += tc - tb
        a = 0
        while a < K and d[a] == top[a]: a += 1
        for i in range(K): offered[i] += i <= a; acc[i] += i < a   # (draft i counts only when every draft before it was kept)
        new = d[:a] + [top[a]]
        h = ver[K + 1].buf['h'][:(K + 1) * D].view(K + 1, D)
        hrows, rtoks, p0 = h[:a + 1], new, p + 1     # rows p + 1 .. p + a + 1: (h of p .. p + a, d1 .. da, bonus)
        out += new; p += a + 1; last_T = a + 1; rounds += 1
    sync()
    return out[:N], time.perf_counter() - t0, dict(acc=acc, offered=offered, rounds=rounds, t_draft=t_draft, t_ver=t_ver)


if A.ids:
    import json
    SRC = {k: [int(t) for t in v][:P] for k, v in json.load(open(A.ids))['prompts'].items()}
    if any(len(v) < P for v in SRC.values()): sys.exit(f'q35_mtp_spec: a prompt in {A.ids} is shorter than --prompt {P}')
else:
    SRC = {k: tk.encode(open(path, encoding='utf-8').read())[:P] for k, path in TEXTS.items()}
for name, prompt in SRC.items():
    ref, tp = plain(prompt)
    got, ts, st = spec(prompt)
    same = got == ref
    bad += not same
    gen = N - 1                                       # (timed: the tokens after the first, which the prompt's last pass gives)
    print(f'{"PASS" if same else "FAIL"} {name}: the speculative loop\'s {len(got)} tokens are the plain loop\'s'
          + ('' if same else f' (first difference at {next(i for i in range(len(got)) if got[i] != ref[i])})'), flush=True)
    print(f'  plain {gen / tp:.1f} tok/s ({1e3 * tp / gen:.2f} ms/token) | speculative {gen / ts:.1f} tok/s ({1e3 * ts / gen:.2f} ms/token, '
          f'{tp / ts:.2f}x) | {st["rounds"]} rounds, {gen / st["rounds"]:.2f} tokens/round | drafts accepted '
          + ', '.join(f'{st["acc"][i]}/{st["offered"][i]}' for i in range(K))
          + f' | per round: drafter {1e3 * st["t_draft"] / st["rounds"]:.2f} ms, verify {1e3 * st["t_ver"] / st["rounds"]:.2f} ms', flush=True)
print('RESULT', 'PASS' if bad == 0 else 'FAIL'); sys.exit(1 if bad else 0)
