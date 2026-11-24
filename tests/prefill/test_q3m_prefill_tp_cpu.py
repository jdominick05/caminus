"""The qwen3moe two-card batched prefill's host side without a card (caminus.prefill.q3m_tp.Q3MPrefillTP on Q3MoETP): the
tiny synthetic file with two-card shapes (tests/engine/tiny_q3m.py tp=True), the engine and the prefill on stand-ins for the
torch.cuda and HIP calls (every tensor in host memory). The launch wrappers are q3m_ops.Ops's own (their host checks run);
only the code object is a recorder, so every kernel launch's grid and arguments are checked, and nothing is launched.
  - micro_bounds: the pieces partition the chunk, every piece but the last on the 128 grid, at most m, none below the floor
  - local shapes: each card's (its kv heads with 8 query heads, its expert rows) from the engine and from the header agree;
    the buffer projection equals the bytes the prefill allocates (+ the exchange's receive rows), and need_tp_mib adds the
    code objects and streams; capacity.check runs once on both cards with it before the prefill's first allocation
  - the launches: every GEMM reads the card's own slice in place (rows / K of attn_q, attn_k, attn_v, attn_output, the
    experts' gate / up / down per matrix, Q6_K's total rows), qk_rope and the attention run the card's 16 query / 2 kv heads
    at the chunk's absolute positions, every position's cache rows written once per layer and card, the slot sum over
    the top-k slots of D floats per token into the FFN partial
  - the exchange: 2 (nl - 1) per piece (the last layer stops after its cache rows), card 0's partial first in every post,
    the attention's partial `o` and the FFN's `ffp`; mode 'overlap' (2 pieces): every piece's sum is queued after another
    piece's compute and before its own next half on both cards; 'serial': each sum right after its post; 'local': no
    exchange, and the engine is poisoned after it; the three modes run the same compute launches
  - refusals before any launch: a span past max_ctx, a drafter, a local_only engine, an unknown mode
  - the served build (caminus.arch.Q3MoETP.build, caminus serve --cards 2): the engine at the placement's stored split,
    its capacity check counting the prefill's construction need, then the prefill at chunk 2048 checking that need
  - the imports: caminus.prefill.q3m_tp with gguf-py and tokenizers refused
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/prefill/test_q3m_prefill_tp_cpu.py --scratch DIR"""
import argparse, contextlib, os, subprocess, sys, types
os.environ['CAMINUS_H2D'] = 'pageable'   # (CPU stand-ins: the uploads are tests/engine/test_q3moe_tp_cpu.py's)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, os.path.join(HERE, '..', 'engine'))
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); A = ap.parse_args()
os.makedirs(A.scratch, exist_ok=True)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


# ---- stand-ins: torch.cuda / HIP on host memory (no device is opened) -----------------------------------------------------
EVENTS = None
_zeros, _to, _tensor = torch.zeros, torch.Tensor.to, torch.tensor


def zeros(*a, **k):
    if EVENTS is not None and str(k.get('device', '')).startswith('cuda'): EVENTS.append(('alloc', a, k.get('dtype')))
    return _zeros(*a, **{x: v for x, v in k.items() if x not in ('device', 'pin_memory')})


class FakeEv:
    def record(self, *a): pass

    def synchronize(self): pass


torch.zeros = zeros
torch.tensor = lambda *a, **k: _tensor(*a, **{x: v for x, v in k.items() if x != 'device'})
torch.Tensor.to = lambda self, *a, **k: self if a and str(a[0]).startswith('cuda') else _to(self, *a, **k)
torch.Tensor.cuda = lambda self, *a, **k: self
torch.Tensor.is_cuda = property(lambda self: True)
torch.cuda.set_device = lambda d: None
torch.cuda.device = lambda d: contextlib.nullcontext()
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda d: types.SimpleNamespace(multi_processor_count=42, total_memory=20464 << 20)
torch.cuda.Event = lambda *a, **k: FakeEv()
torch.cuda.current_stream = lambda d=None: types.SimpleNamespace(device=torch.device(str(d)) if d is not None else None)
torch.cuda.synchronize = lambda d=None: None

import tiny_q3m
from caminus.device import capacity
from caminus.engine import k3, q3moe_tp
from caminus.engine.q3moe_tp import Q3MoETP
from caminus.prefill import q3m_tp, q3m_ops
from caminus.prefill.driver import PrefillRefused
from caminus.prefill.q3m_tp import Q3MPrefillTP, micro_bounds, local_shapes, tp_shapes, layout_tp, buffer_need_tp_mib, need_tp_mib
from caminus.prefill.xchg import xchg_bytes


class FakeHip:
    def __getattr__(self, n): return lambda *a, **k: 0


class FakeSlots:
    n = 0

    def __init__(self, dev, n, N): FakeSlots.n += 1; self.base = FakeSlots.n << 40; self.stride = (16 + N) * 4

    def slot(self, i): return self.base + i * self.stride


q3moe_tp._hip = lambda: FakeHip()
q3moe_tp.Slots = FakeSlots

LOG = []   # (card, kernel, grid, args)


class Rec:
    def __init__(self, card): self.card = card

    def __getattr__(self, name):
        return lambda grid, block, *args, **kw: LOG.append((self.card, name, tuple(grid), args))


class FakeOps(q3m_ops.Ops):
    def __init__(self, dev, module='q3m_prefill', q8=False, yarn=False):
        self.dev, self.module, self.q8, self.yarn = dev, module, bool(q8), bool(yarn)
        self.mod = Rec(int(str(dev).split(':')[1]))
        if EVENTS is not None: EVENTS.append(('ops', module))


class FakeSlotSum(q3m_tp.SlotSum):
    def __init__(self, dev):
        self.dev = dev; self.mod = Rec(int(str(dev).split(':')[1]))


class FakeX:
    def __init__(self, devices, T, D, streams, check=True):
        self.devs, self.T, self.D = list(devices), T, D; self.calls = []
        if EVENTS is not None: EVENTS.append(('exchange', T, D))

    def post(self, parts, r0, r1):
        self.calls.append(('post', parts[0].data_ptr(), parts[1].data_ptr(), r0, r1)); return ('t', len(self.calls))

    def sum(self, t, mark=None): self.calls.append(('sum', t))

    def settle(self): pass


q3m_tp.q3m_ops.Ops = FakeOps; q3m_tp.SlotSum = FakeSlotSum; q3m_tp.Exchange = FakeX

# ---- micro_bounds ----------------------------------------------------------------------------------------------------------
good = True
for T in (1, 16, 127, 128, 300, 511, 512, 1000, 1024, 2047, 2048, 4096):
    for m_ in (1, 2, 3, 4):
        for mp in (1, 128, 512):
            Bd = micro_bounds(T, m_, min_piece=mp)
            good &= Bd[0][0] == 0 and Bd[-1][1] == T and all(a < b for a, b in Bd) and all(Bd[i][1] == Bd[i + 1][0] for i in range(len(Bd) - 1))
            good &= all(a % 128 == 0 for a, _ in Bd) and 1 <= len(Bd) <= max(1, min(m_, T // mp))
check(good and micro_bounds(2048, 2) == [(0, 1024), (1024, 2048)] and micro_bounds(600, 2) == [(0, 600)]
      and micro_bounds(300, 2, min_piece=128) == [(0, 256), (256, 300)],
      'micro_bounds: the pieces partition the chunk on the 128 grid, at most m, none below the floor (2048 at m 2: two of 1,024; '
      '600: one piece at the 512 floor)')

# ---- the engine and the prefill on the tiny file ----------------------------------------------------------------------------
P_T = os.path.join(A.scratch, 'tiny-q3m-tp.gguf'); info = tiny_q3m.write(P_T, tp=True)
MAXC = 640
m = Q3MoETP(P_T, devices=(0, 1), max_ctx=MAXC, vram_check=False)
nl, D, KU, E = info['layers'], info['D'], info['KU'], info['E']
ls, hs = [local_shapes(m, c) for c in (0, 1)], tp_shapes(P_T)
check(all(vars(a) == vars(b) for a, b in zip(ls, hs)) and [s.nkv for s in ls] == [2, 2] and [s.nh for s in ls] == [16, 16]
      and [s.ffn for s in ls] == [512, 256],
      f'local shapes from the engine and from the header agree: 16 query / 2 kv heads per card, expert rows 512 / 256')
calls = {}
real_check = capacity.check
capacity.check = lambda devs, need, what: (EVENTS.append(('check',)), calls.update(devs=list(devs), need=list(need)))[-1]
EVENTS = []
CH = 512
try:
    pf = Q3MPrefillTP(m, chunk=CH)
finally:
    capacity.check = real_check
ev, EVENTS = EVENTS, None
alloc = [sum(v.numel() * v.element_size() for v in pf.B[c].values()) for c in (0, 1)]
want = [sum(4 * v for v in layout_tp(s, CH)[0].values()) + sum(2 * v for v in layout_tp(s, CH)[1].values())
        + sum(4 * v for v in layout_tp(s, CH)[2].values()) for s in ls]
bn = buffer_need_tp_mib(ls, CH); nn = need_tp_mib(ls, CH)
check(alloc == want and all(abs(b * capacity.MIB - (a + xchg_bytes(CH, D))) < 1e-6 for a, b in zip(alloc, bn))
      and all(abs(n_ - b - capacity.PREFILL_CODE_MIB[c] - capacity.PREFILL_STREAMS_MIB[c]) < 1e-9 for c, (n_, b) in enumerate(zip(nn, bn))),
      f'the buffer projection is the bytes allocated per card + the exchange\'s receive rows ({bn[0]:.2f} / {bn[1]:.2f} MiB at chunk {CH}); '
      f'need_tp_mib adds the code objects and streams')
check(ev[0] == ('check',) and [e for e in ev if e[0] == 'check'] == [('check',)] and calls['devs'] == [0, 1]
      and all(abs(a - b) < 1e-9 for a, b in zip(calls['need'], nn)) and any(e[0] == 'alloc' for e in ev) and ev[-1][0] == 'exchange',
      'capacity.check runs once on both cards with need_tp_mib before the prefill\'s first allocation; the exchange is built last (check=False)')

# ---- one prefill: the launches ---------------------------------------------------------------------------------------------
names = {}
for c in (0, 1):
    for il in range(nl):
        for k_, W in pf.W[c][il].items():
            if isinstance(W, tuple): names[W[3].data_ptr()] = (c, il, k_, W)
g = np.random.default_rng(3)
toks = g.integers(0, info['V'], 600).tolist()


def run(mode, micro=2, min_piece=128, chunk=None, p1=300, stops=()):
    m.reset(); m.poisoned = None
    pf.mode, pf.micro, pf.min_piece = mode, micro, min_piece
    LOG.clear(); pf.trace = []; pf.X.calls = []
    st = []
    pf.run(toks, 0, 0, p1, chunk=chunk, stops=stops, at_stop=(lambda b, s: st.append(b)) if stops else None)
    return list(LOG), list(pf.trace), list(pf.X.calls), st


Lg, tr, xc, _ = run('overlap')
bad = []
seen_mm = set()
for c, k_, grid, a in Lg:
    if k_ in ('q3p_mm_q4k', 'q3p_mm_q6k'):
        wc, il, key, W = names[a[0].data_ptr()]
        R, K = int(a[1].value), int(a[2 if k_ == 'q3p_mm_q4k' else 3].value)
        s = ls[c]
        exp = dict(attn_q=(s.nh * 128, D), attn_k=(s.nkv * 128, D), attn_v=(s.nkv * 128, D), attn_output=(D, s.nh * 128),
                   ffn_gate_exps=(s.ffn, D), ffn_up_exps=(s.ffn, D), ffn_down_exps=(D, s.ffn))[key]
        if wc != c or (R, K) != exp or grid[1] != -(-R // 128): bad.append((c, key, R, K, exp))
        if k_ == 'q3p_mm_q6k':
            rtot = int(a[2].value)
            if rtot != (E * R if key.endswith('_exps') else R): bad.append((c, key, 'rtot', rtot))
        seen_mm.add((c, key, W[0]))
# (the tiny file: attn_v and ffn_down Q6_K in layer 0, Q4_K in layer 1; the last layer runs only q / k / v)
want_mm = {(c, k_, 'Q4_K') for c in (0, 1) for k_ in ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate_exps', 'ffn_up_exps')} \
    | {(c, k_, 'Q6_K') for c in (0, 1) for k_ in ('attn_v', 'ffn_down_exps')}
check(not bad and seen_mm == want_mm,
      f'every GEMM reads card c\'s own slice in place at its local shape (rows per matrix, K, Q6_K total rows; Q4_K and Q6_K, '
      f'{len(seen_mm)} (card, matrix, format) pairs)' + (f'; wrong {bad[:4]}' if bad else ''))
rope = [(c, int(a[5].value), grid) for c, k_, grid, a in Lg if k_ == 'q3p_qk_rope']
att = [(c, grid, int(a[2].value), int(a[3].value), (int(a[5].value), int(a[6].value))) for c, k_, grid, a in Lg if k_ == 'q3mp_attn']
cover = {(c, L): np.zeros(300, int) for c in (0, 1) for L in range(nl)}
li = {c: 0 for c in (0, 1)}
for c, pos0, grid in rope:
    n = grid[0]; cover[(c, li[c] // len(micro_bounds(300, 2, min_piece=128)))][pos0:pos0 + n] += 1; li[c] += 1
good = all(grid[1] == 16 + 2 for _, _, grid in rope) and all(v.min() == 1 == v.max() for v in cover.values())
good &= all(grid[1] == 2 and hk == (16, 2) for _, grid, _, _, hk in att) and len(att) == 2 * (nl - 1) * 2
check(good, f'qk_rope runs each card\'s 16 query + 2 kv heads and writes every position\'s cache rows once per layer and card; '
      f'the attention runs 16 query heads over 2 kv heads per card ({len(att)} launches: none on the last layer)')
ss = [(c, grid, a) for c, k_, grid, a in Lg if k_ == 'q3p_slot_sum']
good = len(ss) == 2 * (nl - 1) * 2 and all(int(a[2].value) == KU and int(a[4].value) == D for _, _, a in ss)
ffp = [pf.B[c]['ffp'] for c in (0, 1)]
good &= all(ffp[c].data_ptr() <= a[3].data_ptr() < ffp[c].data_ptr() + ffp[c].numel() * 4 for c, _, a in ss)
check(good, f'the slot sum weighs the top-{KU} slots of D = {D} floats per token into the card\'s FFN partial, once per piece and layer but the last')

# ---- the exchange order -------------------------------------------------------------------------------------------------------
Bd = micro_bounds(300, 2, min_piece=128); J = len(Bd)
posts = [x for x in xc if x[0] == 'post']; sums = [x for x in xc if x[0] == 'sum']
key_ptr = {pf.B[0][k_].data_ptr(): k_ for k_ in ('o', 'ffp')}
good = J == 2 and len(posts) == len(sums) == J * 2 * (nl - 1)
good &= all(p[1] == pf.B[0][key_ptr[p[1]]].data_ptr() and p[2] == pf.B[1][key_ptr[p[1]]].data_ptr() for p in posts)
check(good, f'{len(posts)} exchanges = {J} pieces x 2 (nl - 1) (the last layer stops after its cache rows); every post names card 0\'s '
      f'partial first, `o` after the attention and `ffp` after the FFN')
halves = pf.halves()
hidx = {h: i for i, h in enumerate(halves)}
seq = []
for e in tr:
    if e[0] in ('attn', 'ffn'):
        j = [i for i, (a, b) in enumerate(Bd) if a == e[3]][0]; seq.append(('c', e[1], hidx[(e[0], e[2])], j))
    elif e[0] == 'post': seq.append(('p', None, hidx[(e[1], e[2])], e[3]))
    elif e[0] == 'sum': seq.append(('s', None, hidx[(e[1], e[2])], e[3]))


def overlap_ok(seq):
    for i, (t, _, h, j) in enumerate(seq):
        if t != 's': continue
        ip = seq.index(('p', None, h, j))
        between = [x for x in seq[ip + 1:i] if x[0] == 'c' and (x[2], x[3]) != (h, j)]
        nxt = [k_ for k_, x in enumerate(seq) if x[0] == 'c' and x[2] == h + 1 and x[3] == j]
        if not between or (nxt and min(nxt) < i): return False
        cs = [x for x in seq[:ip] if x[0] == 'c' and x[2] == h and x[3] == j]
        if sorted(x[1] for x in cs) != [0, 1]: return False
    return True


check(overlap_ok(seq), 'overlap (2 pieces): each piece\'s exchange is posted after both cards\' compute of it, and summed after another '
      'piece\'s compute (the copy runs under it) but before its own next half on both cards')
Lg_s, tr_s, xc_s, _ = run('serial')
check(all(xc_s[i][0] == 'post' and xc_s[i + 1][0] == 'sum' for i in range(0, len(xc_s), 2)) and len(xc_s) == len(xc)
      and [x[:2] + tuple(a.value if hasattr(a, 'value') else (a.data_ptr() if hasattr(a, 'data_ptr') else a) for a in x[3])
           for x in Lg_s] == [x[:2] + tuple(a.value if hasattr(a, 'value') else (a.data_ptr() if hasattr(a, 'data_ptr') else a) for a in x[3]) for x in Lg],
      'serial: each sum right after its post; the same compute launches as overlap, argument for argument')
Lg_l, _, xc_l, _ = run('local')
check(not xc_l and len(Lg_l) == len(Lg) and m.poisoned and 'local' in m.poisoned,
      'local: no exchange, the same compute launches, and the engine is poisoned after it (its KV rows are not the model\'s)')
m.poisoned = None
Lg1, _, xc1, _ = run('overlap', micro=1)
Lg16, _, _, _ = run('overlap', chunk=16)
Lgst, _, _, st = run('overlap', stops=(16, 40, 41))
r16 = sorted((c, int(a[5].value), g_[0]) for c, k_, g_, a in Lg16 if k_ == 'q3p_qk_rope')
cov = {}
for c, p0, n in r16: cov.setdefault(c, []).append((p0, n))
good = len(xc1) == 2 * (nl - 1) * 2 and st == [16, 40, 41]
good &= all(sum(n for _, n in v) == 300 * nl for v in cov.values())
check(good, 'one piece: 2 (nl - 1) exchanges a chunk; chunks of 16 and the cache\'s stops (16, 40, 41) cover every position once a layer')

# ---- refusals ------------------------------------------------------------------------------------------------------------------
LOG.clear(); res = []
m.reset()
for f in (lambda: pf.run(toks * 2, 0, 0, MAXC + 1), lambda: pf.run(toks, 0, 0, 300, draft=types.SimpleNamespace(kind='mtp'))):
    try: f(); res.append(False)
    except PrefillRefused: res.append(True)
check(all(res) and not LOG and m.poisoned is None, 'a span past max_ctx and a drafter are refused before any launch; nothing poisoned')
res = []
m.local_only = True
try: Q3MPrefillTP(m, chunk=64, check=False); res.append(False)
except ValueError: res.append(True)
m.local_only = False
try: Q3MPrefillTP(m, chunk=64, check=False, mode='fast'); res.append(False)
except ValueError: res.append(True)
check(all(res), 'a local_only engine and an unknown mode are refused at construction')
check(pf.numerics()['tp'] == 'xchg-fp32-c0-first-slot-sum' and pf.limit == MAXC and pf.crossover == q3m_tp.CROSSOVER,
      'the numerics key names the exchange; the limit is max_ctx; the crossover is the one-card prefill\'s')

# ---- the served build: arch.Q3MoETP (caminus serve --cards 2) on the placement's split -------------------------------------
from caminus import arch
TPA = arch.ARCHS['qwen3moe-tp']
seen = []
capacity.check = lambda devs, need, what: seen.append((list(devs), [float(v) for v in need], what))
lines = []
try:
    bm, bpf, bdf, bdr = TPA.build(dict(model=P_T, max_ctx=MAXC, placement=dict(split={'exp_rows': 512, 'vocab_share': 0.532})),
                                  log=lambda *a, **k: lines.append(' '.join(map(str, a))))
finally:
    capacity.check = real_check
pfn = need_tp_mib(tp_shapes(P_T, 512), TPA.PREFILL_CHUNK)
check(type(bm) is Q3MoETP and type(bpf) is Q3MPrefillTP and bdf is None and bdr is None and bpf.m is bm
      and bm.exp_rows0 == 512 and bm.vocab_share == 0.532 and bm.max_ctx == MAXC and bpf.chunk == TPA.PREFILL_CHUNK == 2048
      and len(seen) == 2 and seen[0][0] == [0, 1] and seen[1][0] == [0, 1]
      and all(abs(a - (v + e)) < 1e-9 for a, v, e in zip(seen[0][1], bm.vram_mib, pfn))
      and all(abs(a - b) < 1e-9 for a, b in zip(seen[1][1], pfn)) and any('expert rows 512 / 256' in l for l in lines),
      f'arch.Q3MoETP.build: the engine at the placed split (512 / 256, vocab_share 0.532) with the prefill\'s own check '
      f'counted in its check ({" / ".join(f"{v:.0f}" for v in seen[0][1])} MiB), then Q3MPrefillTP at chunk 2048 checking the '
      f'same need_tp_mib; no drafter')
del bm, bpf

# ---- the kernel source ---------------------------------------------------------------------------------------------------------
src = open(os.path.join(HERE, '..', '..', 'kernels', 'q3m_prefill_tp.hip')).read()
check('__builtin_fmaf(wr[s], dr[(size_t)s * N + i], acc)' in src and 'float acc = 0.0f;' in src and '#include "' not in src,
      'kernels/q3m_prefill_tp.hip: q3p_slot_sum is OP_OWN_SLOT_SUM\'s fma chain from 0 in slot order; no local includes')

# ---- the imports ---------------------------------------------------------------------------------------------------------------
BLOCK = ("import sys\nclass B:\n    def find_spec(self, name, path=None, target=None):\n"
         "        if name.split('.')[0] in ('gguf', 'tokenizers'): raise ImportError('blocked: ' + name)\n"
         "sys.meta_path.insert(0, B()); sys.path.insert(0, %r)\n"
         "import caminus.engine.q3moe_tp, caminus.prefill.q3m_tp\nprint('IMPORTED')\n") % os.path.join(HERE, '..', '..', 'python')
r = subprocess.run([sys.executable, '-I', '-c', BLOCK], capture_output=True, text=True, timeout=120)
check(r.returncode == 0 and 'IMPORTED' in r.stdout, 'the two-card prefill imports with gguf-py and tokenizers refused'
      + ('' if r.returncode == 0 else f' ({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode})'))

print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
