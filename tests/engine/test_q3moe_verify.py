"""Q3MoEVerify (qwen3moe, T-token verify passes on kernels/k3_q3m_ver.hip) against Q3MoEOne's decode steps on the same
tokens and positions: the logits and the next decode step bit for bit, for T = 1 .. 4, on real consecutive tokens (this
repository's docs as text) over a prefilled context, and on one token repeated from an empty context (the tokens of a
pass then route alike: an expert has up to 4 member slots). Before each T's passes the cache rows of the checked span are
overwritten in every layer, so the passes' own row stores are what the later rows and the decode step after them read.
A wrong token at row 2 leaves rows 0 and 1 right and changes rows 2 and 3; the right pass after it gives every row and
the decode step after it. Tap features are equal across T; the host refusals leave the cache and the stage untouched.
A pass of 3 or 4 tokens runs on a grid of its own (WIDE_WGP): passes of 2 on that grid, and passes of 4 on the decode
engine's grid and on 3 workgroups per WGP, give the same logits.
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
from caminus.engine.q3moe_verify import Q3MoEVerify, MAX_T, WIDE_T, WIDE_WGP
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
eng = Q3MoEOne(A.gguf, max_ctx=max_ctx, vram_extra_mib=sum(Q3MoEVerify.need_mib(T) for T in list(range(1, MAX_T + 1)) + [2, MAX_T, MAX_T]))
own_mib = (G + MAX_T) * eng.vocab * 4 / 2 ** 20 + 8
capacity.check([eng.dev], [own_mib], f'test_q3moe_verify: its own buffers ({G} positions of logits: {own_mib:.0f} MiB)')
pf = Q3MPrefill(eng, chunk=512)
vers = {T: Q3MoEVerify(eng, T) for T in range(1, MAX_T + 1)}
alt = [Q3MoEVerify(eng, T, per_wgp=w) for T, w in ((2, WIDE_WGP), (MAX_T, eng.per_wgp), (MAX_T, 3))]   # the grids the defaults do not run
print(f'engine {eng.vram_mib:.0f} MiB at max_ctx {max_ctx}, {eng.per_wgp} workgroups per WGP; verify tapes: ' +
      ', '.join(f'T {T}: {len(v.E.segs)} segments, {v.mib:.1f} MiB, {v.per_wgp} per WGP' for T, v in vers.items()), flush=True)
RW = 2 * eng.nkv * eng.hd                                 # elements of one position's cache row (k | v)
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
        for d in eng.L: d['cache'][N * RW:(N + G) * RW] = 3.0   # (the decode steps' rows of the span are gone: the passes store their own)
        bad = 0; dis = []
        for b in range(G // T):
            ver.run(toks[b * T:(b + 1) * T], N + b * T); dis.append(ver.distinct())
            bad += sum(int((bits(ver.logits(t)) != ref[b * T + t]).sum()) for t in range(T))
        eng.pos_host = N + G                               # the cache rows N .. N + G - 1 are now the verify passes'
        same_next = np.array_equal(bits(eng.step(toks[0])), nxt)
        check(bad == 0 and same_next, f'{what}, T {T}: {G // T} passes over overwritten cache rows bitwise the decode steps\' logits ({bad} words '
              f'differ); the decode step after them {"the same" if same_next else "DIFFERS"}; distinct experts per layer {np.mean(dis):.1f} of {8 * T}')
    if N == depths[0]:
        check(all(v.per_wgp == (WIDE_WGP if T >= WIDE_T else eng.per_wgp) for T, v in vers.items()) and len({(v.T, v.E.grid) for v in list(vers.values()) + alt}) == MAX_T + len(alt),
              f'a pass of {WIDE_T} or more tokens takes {WIDE_WGP} workgroups per WGP, a shorter one the decode engine\'s {eng.per_wgp}; the other tapes are on other grids')
        for ver in alt:
            T = ver.T; bad = 0
            for b in range(G // T):
                ver.run(toks[b * T:(b + 1) * T], N + b * T)
                bad += sum(int((bits(ver.logits(t)) != ref[b * T + t]).sum()) for t in range(T))
            check(bad == 0, f'{what}, T {T} at {ver.per_wgp} workgroups per WGP (grid {ver.E.grid}): bitwise the decode steps\' logits ({bad} words differ)')
    if not real:
        check(np.mean(dis) < 8.5, f'{what}: the tokens of a T {MAX_T} pass route alike ({np.mean(dis):.2f} distinct experts per layer: '
              f'{MAX_T} member slots per expert)')
    # a wrong token at row 2: rows 0 and 1 stay right, rows 2 and 3 change (row 3 reads row 2's cache row); the right
    # pass after it gives every row, and its row stores give the decode step after it
    ver = vers[MAX_T]; w = list(toks[:MAX_T]); w[2] = (w[2] + 1) % eng.vocab
    ver.run(w, N)
    r01 = all(np.array_equal(bits(ver.logits(t)), ref[t]) for t in (0, 1))
    r23 = all(not np.array_equal(bits(ver.logits(t)), ref[t]) for t in (2, 3))
    ver.run(toks[:MAX_T], N)
    back = all(np.array_equal(bits(ver.logits(t)), ref[t]) for t in range(MAX_T))
    eng.pos_host = N + MAX_T; after = np.array_equal(bits(eng.step(toks[MAX_T])), ref[MAX_T])
    check(r01 and r23 and back and after, f'{what}: a wrong token at row 2 leaves rows 0, 1 right ({r01}), changes rows 2, 3 ({r23}); the right pass after it '
          f'gives every row ({back}) and the decode step after it ({after})')
    # tap features: a row's features do not depend on the T of the pass that computed it
    f4 = [bits(ver.feats(t)) for t in range(MAX_T)]
    same = True
    for T in (1, 2):
        for b in range(MAX_T // T):
            vers[T].run(toks[b * T:(b + 1) * T], N + b * T)
            same &= all(np.array_equal(bits(vers[T].feats(t)), f4[b * T + t]) for t in range(T))
    check(same and all(np.isfinite(f.view(np.float32)).all() and f.any() for f in f4), f'{what}: tap features ({len(ver.taps)} layers) bitwise equal from T 1, 2 and {MAX_T} passes')

# the link keep-alive segment (Q3MoEOne's, on its host buffers) is in every verify tape and runs in a pass
ver = vers[MAX_T]; N = depths[0]; toks = ids_all[N:N + MAX_T]
eng._ka[0].fill_(1.0); eng._ka[1].zero_(); ver.run(toks, N)
want = eng.out_norm.detach().float().cpu() / float(np.sqrt(1.0 + eng.eps))
check(all(v.E.segs[v.ka_seg].op == k3.OP_OWN_ADD_RMS16 for v in vers.values()) and bool(torch.allclose(eng._ka[1], want, rtol=1e-5, atol=1e-7)),
      'the link keep-alive segment is in every verify tape and ran in a pass (the norm of its host input, in host memory)')
eng._ka[0].zero_()
# host refusals: nothing is written (the last cache row of a layer and the staged block stay as they were)
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
