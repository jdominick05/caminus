"""Q3MoEVerify (qwen3moe, T-token verify passes on kernels/k3_q3m_ver.hip) against Q3MoEOne's decode steps on the same
tokens and positions: the logits and the next decode step bit for bit, for T = 1 .. 4, on real consecutive tokens (this
repository's docs as text) over a prefilled context, and on one token repeated from an empty context (the tokens of a
pass then route alike: an expert has up to 4 member slots); a wrong token at row 2 leaves rows 0 and 1 right and the right pass after it gives every
row; tap features equal across T; the host refusals leave the cache and the stage untouched.
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  tools/guard.sh LOG 900 python -I tests/engine/test_q3moe_verify.py GGUF [--depths 512,2048]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, glob, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'python'))
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument('gguf'); ap.add_argument('--depths', default='512,2048')
A = ap.parse_args()
from caminus import ggufio
from caminus.bpe import BPE
from caminus.device import capacity
from caminus.engine import k3
from caminus.engine.q3moe_one import Q3MoEOne
from caminus.engine.q3moe_verify import Q3MoEVerify, MAX_T
from caminus.prefill.q3m import Q3MPrefill

depths = [int(x) for x in A.depths.split(',')]
G = 24                                                    # tokens checked per depth (a multiple of 1, 2, 3, 4)
tk = BPE(ggufio.GGUFFile(A.gguf).meta)
text = ''.join(open(f).read() for f in sorted(glob.glob(os.path.join(ROOT, 'docs', '*.md'))) + [os.path.join(ROOT, 'AGENTS.md')])
ids_all = tk.encode(text)
if len(ids_all) < max(depths) + G: raise SystemExit(f'the docs are {len(ids_all)} tokens: lower --depths')
print(torch.cuda.get_device_name(0), flush=True)
# beside the engine: a verify tape per T, the prefill (which projects its own buffers), and this test's reference logits
max_ctx = max(depths) + G + 8
eng = Q3MoEOne(A.gguf, max_ctx=max_ctx, vram_extra_mib=sum(Q3MoEVerify.need_mib(T) for T in range(1, MAX_T + 1)))
own_mib = (G + MAX_T) * eng.vocab * 4 / 2 ** 20 + 8
capacity.check([eng.dev], [own_mib], f'test_q3moe_verify: its own buffers ({G} positions of logits: {own_mib:.0f} MiB)')
pf = Q3MPrefill(eng, chunk=512)
vers = {T: Q3MoEVerify(eng, T) for T in range(1, MAX_T + 1)}
print(f'engine {eng.vram_mib:.0f} MiB at max_ctx {max_ctx}; verify tapes: ' +
      ', '.join(f'T {T}: {len(v.E.segs)} segments, {v.mib:.1f} MiB' for T, v in vers.items()), flush=True)
ok = True
def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)
bits = lambda t: t.detach().cpu().numpy().view(np.uint32).copy()


def start(N, toks):
    """the context for position N (a prefill of real text, or an empty cache), then the decode steps' logits"""
    eng.reset()
    if N in depths: pf.run(ids_all[:N], 0, 0, N)
    else:
        for d in eng.L: d['cache'].zero_()
    eng.pos_host = N
    return [bits(eng.step(t)) for t in toks]


for N in depths + [8]:
    real = N in depths
    toks = ids_all[N:N + G] if real else [785] * G
    ref = start(N, toks)
    nxt = bits(eng.step(toks[0]))                          # (position N + G: the step after the checked span)
    what = f'depth {N}, ' + ('real tokens' if real else 'one token repeated')
    for T, ver in vers.items():
        bad = 0; dis = []
        for b in range(G // T):
            ver.run(toks[b * T:(b + 1) * T], N + b * T); dis.append(ver.distinct())
            bad += sum(int((bits(ver.logits(t)) != ref[b * T + t]).sum()) for t in range(T))
        eng.pos_host = N + G                               # the cache rows N .. N + G - 1 are now the verify passes'
        same_next = np.array_equal(bits(eng.step(toks[0])), nxt)
        check(bad == 0 and same_next, f'{what}, T {T}: {G // T} passes bitwise the decode steps\' logits ({bad} words differ); the decode step '
              f'after them {"the same" if same_next else "DIFFERS"}; distinct experts per layer {np.mean(dis):.1f} of {8 * T}')
    if not real:
        check(np.mean(dis) < 8.5, f'{what}: the tokens of a T {MAX_T} pass route alike ({np.mean(dis):.2f} distinct experts per layer: '
              f'{MAX_T} member slots per expert)')
    # a wrong token at row 2: rows 0 and 1 stay right; the right pass after it gives every row
    ver = vers[MAX_T]; w = list(toks[:MAX_T]); w[2] = (w[2] + 1) % eng.vocab
    ver.run(w, N)
    r01 = all(np.array_equal(bits(ver.logits(t)), ref[t]) for t in (0, 1)); r2 = not np.array_equal(bits(ver.logits(2)), ref[2]) or not real
    ver.run(toks[:MAX_T], N)
    back = all(np.array_equal(bits(ver.logits(t)), ref[t]) for t in range(MAX_T))
    check(r01 and r2 and back, f'{what}: a wrong token at row 2 leaves rows 0, 1 right ({r01}), changes row 2 ({r2}); the right pass after it gives every row ({back})')
    # tap features: a row's features do not depend on the T of the pass that computed it
    f4 = [bits(ver.feats(t)) for t in range(MAX_T)]
    same = True
    for T in (1, 2):
        for b in range(MAX_T // T):
            vers[T].run(toks[b * T:(b + 1) * T], N + b * T)
            same &= all(np.array_equal(bits(vers[T].feats(t)), f4[b * T + t]) for t in range(T))
    check(same and all(np.isfinite(f.view(np.float32)).all() and f.any() for f in f4), f'{what}: tap features ({len(ver.taps)} layers) bitwise equal from T 1, 2 and {MAX_T} passes')

# host refusals: nothing is written (the last cache row of a layer and the staged block stay as they were)
ver = vers[MAX_T]; N = depths[0]; toks = ids_all[N:N + MAX_T]
def state(): return (bits(eng.L[0]['cache'][-64:].float()), bits(ver.S['stage'].float()), bits(ver.S['logits'][:64]))
def refused(fn, exc):
    s0 = state()
    try: fn()
    except exc: return all(np.array_equal(a, b) for a, b in zip(s0, state()))
    return False
check(refused(lambda: ver.run(toks, max_ctx - MAX_T + 1), k3.ContextBoundsError), f'a span past max_ctx ({max_ctx}) is refused before anything is written')
check(refused(lambda: ver.run(toks, -1), k3.ContextBoundsError), 'a negative position is refused')
check(refused(lambda: ver.run(toks[:2], N), ValueError), f'2 tokens on the T {MAX_T} tape are refused')
check(refused(lambda: ver.run([toks[0], eng.vocab, toks[2], toks[3]], N), ValueError), 'a token outside the vocabulary is refused')
eng.poisoned = 'test'
check(refused(lambda: ver.run(toks, N), k3.ContextBoundsError), 'a poisoned engine refuses until reset()')
eng.reset()
def raises(fn, exc):
    try: fn()
    except exc: return True
    return False
check(raises(lambda: Q3MoEVerify(eng, MAX_T + 1), ValueError) and raises(lambda: Q3MoEVerify(eng, 2, module='k3_q3m_lean'), ValueError),
      f'T {MAX_T + 1} and a decode module are refused at construction')
e2 = k3.Engine(module='k3_q3m_lean'); h = torch.zeros(eng.D, dtype=torch.float16, device='cuda'); o = torch.zeros(2 * eng.D, device='cuda')
w = eng.L[0]['attn_output']
check(raises(lambda: e2.own_matvec_1rt(w[3], w[0], w[1], w[2], torch.zeros(2 * eng.nh * eng.hd, dtype=torch.float16, device='cuda'), o, tokens=2,
                                       x_stride=eng.nh * eng.hd, out_stride=eng.D), ValueError),
      'the decode module refuses a T-token matvec segment')
print('bails:', eng.bails)
check(not eng.bails, 'no wait gave up')
print('RESULT', 'PASS' if ok else 'FAIL')
