"""The dense qwen35 own prefill's tiled attention (kernels/q35_prefill.hip q35p_attn_wm through
caminus.prefill.q35own.Ops.attn_wm: WMMA tiles, head dim 256, wave pairs) on random q and f16 cache rows, no model file:
  - against float64 causal attention (the q rounded to fp16 as the kernel reads it): max / mean |difference| of the
    reference's rms, at chunks from position 0, past the 32-position staging chunks, from an odd position (the first
    8-position block partly before the chunk), deep (8K), Swift / ThinkingCap's 24 query heads over 4 kv heads and 8 per
    kv head (no idle columns)
  - the same bits at every launch split (attn_blocks with a small budget) and every chunk geometry (the chunk cut at an
    odd and an even position; chunks of 16): a row's sums depend only on absolute positions (the prompt cache's stops)
  - against the per-(row, head) kernel it replaces (q35p_attn16, fp32 q): within the float64 bounds
  - shapes it does not run are refused on the host (head dim 128, 12 query heads per kv head, a block range outside the chunk)
  - timing (--speed): a 2048-row chunk at P0 = 0 / 8192 / 30720 (the yardstick rows of docs/dev/yardsticks.md), ours new
    vs old (old at 0 and 8192 only), median of 5 after a warm-up, TFLOPS as there (4 N (P0 + N / 2) heads D); and the
    slowest single launch at ATTN_WM_TP
  tools/guard.sh LOG 900 python -I tests/prefill/test_q35_prefill_attn.py [--speed]
New kernel code: a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>). Prints PASS / FAIL and RESULT."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus.device import capacity
from caminus.prefill import q35own

ap = argparse.ArgumentParser(); ap.add_argument('--speed', action='store_true')
A = ap.parse_args()
D = 256
dev = torch.device('cuda:0')
FAILS = []


def check(c, msg):
    print(('PASS ' if c else 'FAIL ') + msg, flush=True)
    if not c: FAILS.append(msg)


def bits(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


ROWS = 32768 + 64
need = (ROWS * 2 * 8 * D * 2 + 6 * 2048 * 32 * D * 4) / 2 ** 20 + 64   # (the cache at 8 kv heads; q / out buffers)
capacity.check([0], [need], f'test_q35_prefill_attn ({need:.0f} MiB)')
print(torch.cuda.get_device_name(dev), flush=True)
O = q35own.Ops(dev)
g = torch.Generator(device='cuda').manual_seed(11)


def make(nkv, rows):
    return torch.empty(rows * 2 * nkv * D, dtype=torch.float16, device=dev).normal_(0.0, 0.5, generator=g)   # (in place)


def ref_rows(q, cache, nkv, heads, pos0, rows):
    """float64 attention of chunk rows `rows` (q rounded to fp16, as the kernel reads it): [len(rows), heads, D]"""
    G, last = heads // nkv, pos0 + max(rows) + 1
    kv = cache.view(-1, 2, nkv, D)[:last].cpu().double().numpy()   # (to the host first: no float64 copy on the card)
    qq = q.view(-1, heads, D).half().cpu().double().numpy()
    out = np.zeros((len(rows), heads, D))
    for i, r in enumerate(rows):
        P = pos0 + r + 1
        for h in range(heads):
            s = kv[:P, 0, h // G] @ qq[r, h] / np.sqrt(D); e = np.exp(s - s.max()); out[i, h] = (e @ kv[:P, 1, h // G]) / e.sum()
    return out


def rel(a, b):
    rms = np.sqrt((b * b).mean()); d = np.abs(a - b)
    return d.max() / rms, d.mean() / rms


def wm(q, cache, pos0, n, heads, nkv, out):
    for b0, b1 in O.attn_blocks(pos0, n): O.attn_wm(q, cache, pos0, n, heads, nkv, D, 1.0 / np.sqrt(D), out, ROWS, b0, b1)


# ---- accuracy against float64 -------------------------------------------------------------------------------------------
cases = [(24, 4, 0, 37), (24, 4, 600, 37), (24, 4, 601, 50), (24, 4, 8191, 64), (32, 4, 300, 41)]
worst = (0.0, 0.0)
for heads, nkv, pos0, n in cases:
    cache = make(nkv, ROWS)
    q = torch.randn(n * heads * D, generator=g, device=dev) * 0.15
    out = torch.zeros_like(q)
    wm(q, cache, pos0, n, heads, nkv, out); torch.cuda.synchronize()
    rows = list(range(n)) if n * (pos0 + n) < 400_000 else list(range(0, n, 7)) + [n - 1]
    on = out.view(n, heads, D).cpu().double().numpy()[rows]
    mx, mn = rel(on, ref_rows(q, cache, nkv, heads, pos0, rows))
    worst = (max(worst[0], mx), max(worst[1], mn))
    check(np.isfinite(on).all() and mx < 2e-2 and mn < 2e-3,
          f'{heads} q heads / {nkv} kv, rows at positions {pos0} .. {pos0 + n - 1} vs float64: max {mx:.2e}, mean {mn:.2e} of rms')
    if (heads, pos0) == (24, 600):
        old = torch.zeros_like(q)
        for r0, r1 in O.attn_rows(pos0, n): O.attn16(q, cache, pos0, n, heads, nkv, D, 1.0 / np.sqrt(D), old, ROWS, r0, r1)
        torch.cuda.synchronize()
        mo, no = rel(on, old.view(n, heads, D).cpu().double().numpy()[rows])
        check(mo < 2e-2 and no < 2e-3, f'against q35p_attn16 (the kernel it replaces) at {pos0}: max {mo:.2e}, mean {no:.2e} of rms')
    del cache

# ---- the same bits at every launch split and chunk geometry -------------------------------------------------------------
heads, nkv, pos0, n = 24, 4, 1000, 300
cache = make(nkv, ROWS)
q = torch.randn(n * heads * D, generator=g, device=dev) * 0.15
one = torch.zeros_like(q); wm(q, cache, pos0, n, heads, nkv, one)
keep = q35own.ATTN_WM_TP
q35own.ATTN_WM_TP = 8 * 8 * 1400                       # (a few blocks per launch)
spl = torch.zeros_like(q); wm(q, cache, pos0, n, heads, nkv, spl)
nl = len(O.attn_blocks(pos0, n)); q35own.ATTN_WM_TP = keep
torch.cuda.synchronize()
check(nl > 3 and bits(one, spl), f'{nl} launches instead of {len(O.attn_blocks(pos0, n))}: the same bits')
HD = heads * D
for cut in (41, 64):                                    # (position 1041: a pair split across chunks; 1064: a block edge)
    o3 = torch.zeros_like(q)
    wm(q[:cut * HD], cache, pos0, cut, heads, nkv, o3[:cut * HD])
    wm(q[cut * HD:], cache, pos0 + cut, n - cut, heads, nkv, o3[cut * HD:])
    torch.cuda.synchronize()
    check(bits(one, o3), f'the chunk cut at row {cut} (position {pos0 + cut}): the same bits')
o16 = torch.zeros_like(q)
for r in range(0, n, 16):
    m_ = min(16, n - r); wm(q[r * HD:(r + m_) * HD], cache, pos0 + r, m_, heads, nkv, o16[r * HD:(r + m_) * HD])
torch.cuda.synchronize()
check(bits(one, o16), 'chunks of 16: the same bits')

# ---- refusals -----------------------------------------------------------------------------------------------------------
for what, args in (('head dim 128', (24, 4, 128)), ('12 query heads per kv head', (24, 2, D))):
    try:
        O.attn_wm(q, cache, pos0, n, args[0], args[1], args[2], 0.0625, one, ROWS); refused = False
    except ValueError: refused = True
    check(refused, f'attn_wm refuses {what}')
B0, B1 = O._wm_span(pos0, n)
try:
    O.attn_wm(q, cache, pos0, n, heads, nkv, D, 0.0625, one, ROWS, B0, B1 + 1); refused = False
except ValueError: refused = True
check(refused, 'attn_wm refuses blocks past the chunk')
del cache

# ---- timing -------------------------------------------------------------------------------------------------------------
if A.speed:
    N, heads, nkv = 2048, 24, 4
    q = torch.randn(N * heads * D, generator=g, device=dev) * 0.15; out = torch.zeros_like(q)

    def med(fn, reps=5):
        fn(); torch.cuda.synchronize(); ts = []
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); fn(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1))
        return float(np.median(ts))
    cache = make(nkv, ROWS)
    slow = 0.0
    for P0 in (0, 8192, 30720):
        fl = 4.0 * N * (P0 + N / 2) * heads * D
        t_n = med(lambda: wm(q, cache, P0, N, heads, nkv, out))
        line = f'NOTE chunk {N} at P0 {P0:6d}: tiled {t_n:8.2f} ms ({fl / t_n / 1e9:5.1f} TF, {len(O.attn_blocks(P0, N))} launches)'
        if P0 <= 8192:
            def old_():
                for r0, r1 in O.attn_rows(P0, N): O.attn16(q, cache, P0, N, heads, nkv, D, 0.0625, out, ROWS, r0, r1)
            t_o = med(old_, 3)
            line += f' | per-(row, head) {t_o:8.2f} ms ({fl / t_o / 1e9:5.1f} TF) | {t_o / t_n:5.1f}x'
        print(line, flush=True)
        for b0, b1 in O.attn_blocks(P0, N):
            slow = max(slow, med(lambda: O.attn_wm(q, cache, P0, N, heads, nkv, D, 0.0625, out, ROWS, b0, b1), 3))
    print(f'NOTE the slowest single launch at ATTN_WM_TP {q35own.ATTN_WM_TP}: {slow:.2f} ms', flush=True)
    check(slow < 100.0, f'every launch under 100 ms (slowest {slow:.2f} ms)')
print('RESULT', 'FAIL' if FAILS else 'PASS'); sys.exit(1 if FAILS else 0)
