"""The qwen3moe two-card engine's host side without a card (caminus.engine.q3moe_tp): the split and the two tapes it builds
for the tiny synthetic file with two-card shapes (tests/engine/tiny_q3m.py tp=True), on stand-ins for the torch.cuda and HIP
calls (every tensor in host memory; nothing is launched).
  - the split plan, on the served file's shapes and the tiny one's: every split matrix's rows / columns are a partition of
    the whole over the two cards (each index on exactly one card), the column cuts and the expert cut fall on 256-value
    blocks, each card holds whole kv heads with their 8 query heads, and a cut inside a block / a kv split with no head on
    a card is refused
  - the cut bytes: every split tensor of the tiny file (attention, every expert of gate / up / down in Q4_K and Q6_K, the
    head) as each card holds it (Q6_K: its planes) decodes (caminus.own_dequant) to exactly the full matrix's part
  - the tapes: k3_q3m_lean_tp on both cards, every op in k3.Q3M_TP_OPS, 1 + 15 per layer + 2 segments, 2 exchanges per
    layer with indices 0 .. 2 nl - 1 once each and the card's own number, the FFN exchange after an OP_OWN_SLOT_SUM over the
    top-k slots, every residual add after an exchange one row (slots 1, no weights), every dependency pointing back; each
    card's matrices have its share's shapes; the attention's splits follow the card; the tapes build again (a grid test's
    rebuild) without new receive slots; at splits_cap 128 each card takes splits_for(its kv heads, 128), and caps 0 / 129
    are refused
  - host refusals before any launch: a span past max_ctx, a token outside the vocabulary
  - projection: per card, the KV row is its kv heads' share; the terms are its kv heads' KV, the receive slots, the ATS
    scratch and the replicated tensors; the split term equals the bytes the engine uploads per card; the two cards'
    weights add up to more than the file's and less than twice it; past the file's context_length it is computed, not
    refused (and the engine refuses it); at splits_cap 128 the ATS term is that cap's scratch
  - the load: capacity.check runs (both cards, the projection) before the first allocation or upload; with CAMINUS_H2D
    unset the uploader is built with 'pinned' (upload.TWO_CARD_DEFAULT) and the upload log from the environment; every kept
    tensor's host_bytes equals its upload
  - the prompt cache on it (caminus serve --cards 2): the checkpoint layout holds each card's own KV rows (its kv heads),
    no recurrent state, a slot within the registry's per-card slot projection; the cache key carries the two-card module,
    both devices and the split, so a one-card entry on the same cache directory never matches
  - the cache round trip on it (PromptCache on the engine's own KV views, host memory): each (card, layer) filled with its
    own bytes; the tail row back from a checkpoint slot (restore() picks 'vram'), every row back from a park after the
    rows are cleared (restore() picks 'disk'), and again into a new engine and cache on the same directory (a restart),
    compared byte for byte per card and layer; negative controls: the unpark with card 0's and card 1's reads swapped on
    one layer, or one card's layer read one row late, differs in exactly those views
  - the imports: the engine imports with gguf-py and tokenizers refused
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/engine/test_q3moe_tp_cpu.py --scratch DIR"""
import argparse, contextlib, os, subprocess, sys, types
os.environ['CAMINUS_H2D'] = 'pageable'   # (CPU stand-ins: the tapes are under test here, not the uploads; the pinned default
                                         #  needs a GPU: tests/engine/test_q35_upload_cpu.py covers the uploader's arms)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, HERE)
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); A = ap.parse_args()
os.makedirs(A.scratch, exist_ok=True)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


# ---- stand-ins: the engine's torch.cuda / HIP calls on host memory (no device is opened) ------------------------------
EVENTS = None   # a list while the load-order check records (allocations and uploads)
_zeros, _to, _tensor = torch.zeros, torch.Tensor.to, torch.tensor


def zeros(*a, **k):
    if EVENTS is not None and str(k.get('device', '')).startswith('cuda'): EVENTS.append('alloc')
    return _zeros(*a, **{x: v for x, v in k.items() if x not in ('device', 'pin_memory')})


torch.zeros = zeros
torch.tensor = lambda *a, **k: _tensor(*a, **{x: v for x, v in k.items() if x != 'device'})
torch.Tensor.to = lambda self, *a, **k: self if a and str(a[0]).startswith('cuda') else _to(self, *a, **k)
torch.Tensor.cuda = lambda self, *a, **k: self
torch.Tensor.is_cuda = property(lambda self: True)
torch.cuda.set_device = lambda d: None
torch.cuda.device = lambda d: contextlib.nullcontext()
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda d: types.SimpleNamespace(multi_processor_count=42, total_memory=20464 << 20)
torch.cuda.Event = lambda *a, **k: types.SimpleNamespace()

import tiny_q3m
from caminus import ggufio, own_dequant
from caminus.device import capacity
from caminus.engine import k3, upload, q3moe_tp, q3moe_one
from caminus.engine.q3moe_tp import Q3MoETP, split_plan, projection_terms
from caminus.engine.q35_dense_tp import cut, vocab_cut


class FakeHip:
    def __getattr__(self, n): return lambda *a, **k: 0


class FakeSlots:
    """the receive slots' addresses only (distinct per card and slot)"""
    n_made = 0

    def __init__(self, dev, n, N):
        FakeSlots.n_made += 1; self.base = (FakeSlots.n_made << 40); self.stride = (16 + N) * 4; self.n = n; self.N = N
        if EVENTS is not None: EVENTS.append('slots')

    def slot(self, i):
        assert 0 <= i < self.n; return self.base + i * self.stride


q3moe_tp._hip = lambda: FakeHip()
q3moe_tp.Slots = FakeSlots


def partition(ranges, n):
    """each index 0 .. n - 1 in exactly one card's ranges"""
    hit = np.zeros(n, np.int64)
    for rr in ranges:
        for a, b in rr: hit[a:b] += 1
    return bool((hit == 1).all())


# ---- the split plan: the served file's shapes and the tiny one's --------------------------------------------------------
for tag, (nh, nkv, hd, E, ffn, D, vocab) in {'Qwen3-30B-A3B': (32, 4, 128, 128, 768, 2048, 151936),
                                              'tiny': (32, 4, 128, 8, 768, 256, 384)}.items():
    vh = vocab_cut(vocab, 0.532)
    for kv0, r0 in ((2, 512), (1, 256), (3, 512)):
        P = split_plan(nh, nkv, hd, E, ffn, D, vocab, kv0, r0, vh)
        parts = (partition(P['q'], nh * hd) and partition(P['kv'], nkv * hd) and partition(P['attn_output'], nh * hd)
                 and partition(P['exp'], E * ffn) and partition(P['exp_k'], ffn) and partition(P['head'], vocab))
        blocks = all(a % 256 == 0 and b % 256 == 0 for k in ('attn_output', 'exp_k') for rr in P[k] for a, b in rr)
        heads = (P['kv_heads'] == [kv0, nkv - kv0] and all(b - a == 8 * hd * k for (a, b), k in zip([r[0] for r in P['q']], P['kv_heads']))
                 and all(rr[0][0] % hd == 0 for rr in P['kv']) and P['exp_rows'] == [r0, ffn - r0]
                 and all(b - a == r for rr, r in zip(P['exp'], P['exp_rows']) for a, b in rr))
        check(parts and blocks and heads and vh % 128 == 0,
              f'{tag}: kv heads {kv0} / {nkv - kv0}, expert rows {r0} / {ffn - r0}, head {vh} / {vocab - vh}: every split matrix\'s rows / '
              f'columns a partition over the two cards, the column cuts on 256-value blocks, whole kv heads with 8 query heads each')
check(vocab_cut(151936, 0.532) == 80768, f'the served file\'s head cut at 0.532: {vocab_cut(151936, 0.532)} / {151936 - vocab_cut(151936, 0.532)} rows')
refused = []
for bad in (dict(exp_rows=384), dict(exp_rows=768), dict(exp_rows=0), dict(kv_heads=0), dict(kv_heads=4)):
    try: q3moe_tp._check_split(dict(hd=128, nh=32, nkv=4, ffn=768), bad.get('kv_heads', 2), bad.get('exp_rows', 512)); refused.append(False)
    except ValueError: refused.append(True)
check(all(refused), 'an expert cut off the 256-value blocks or not inside the 768 rows, and a kv split leaving a card no head, are refused')

# ---- the tiny file with two-card shapes; the cut bytes decode to the full matrices' parts -------------------------------
P_T = os.path.join(A.scratch, 'tiny-q3m-tp.gguf'); info = tiny_q3m.write(P_T, tp=True)
check(info['H'] == 32 and info['HKV'] == 4 and info['V'] == 384 and os.path.getsize(P_T) < 16 << 20,
      f'tiny_q3m tp=True: {info["H"]} q / {info["HKV"]} kv heads (2 per card), vocabulary {info["V"]}, {os.path.getsize(P_T) / 2 ** 20:.1f} MiB')
m = Q3MoETP(P_T, devices=(0, 1), max_ctx=64, vram_check=False)
from caminus.engine.own_layout import q6k_from_planes
F = ggufio.GGUFFile(P_T)
seen, bad, n_cmp = set(), [], 0
for name, c, kind, w, args in m.uploads:
    if kind != 'cut': continue
    t = F.T[name]
    if t.type == 'F32': continue
    K = int(t.shape[0]); R, rb = t.rows()
    _, _, dec = own_dequant.DECODERS[t.type]
    full = dec(np.asarray(t.data()).copy()).reshape(R, K)
    rows, cols = args
    rr, cc = rows[c] or [(0, R)], cols[c] or [(0, K)]
    want = np.concatenate([full[a:b] for a, b in rr], axis=0); want = np.concatenate([want[:, a:b] for a, b in cc], axis=1)
    raw = w.contiguous().view(-1).view(torch.uint8).numpy()
    Rl, Kl = want.shape
    if t.type == 'Q6_K': raw = q6k_from_planes(raw, Rl, Kl).reshape(-1)
    if not np.array_equal(dec(raw).reshape(Rl, Kl), want): bad.append((name, c))
    seen.add((t.type, name.split('.')[-2])); n_cmp += 1
F.close()
check(not bad and {('Q4_K', 'ffn_gate_exps'), ('Q4_K', 'ffn_up_exps'), ('Q6_K', 'ffn_down_exps'), ('Q4_K', 'ffn_down_exps'),
                   ('Q6_K', 'attn_v'), ('Q4_K', 'attn_output'), ('Q6_K', 'output')} <= seen,
      f'the {n_cmp} cut matrices on the cards (Q6_K: from their planes) decode to exactly the full matrices\' rows / columns '
      f'({sorted(seen)})' + (f'; wrong: {bad[:6]}' if bad else ''))
try:
    cut(np.zeros((2, 144), np.uint8), None, [(0, 128)], 256, 144); r = False
except ValueError:
    r = True
check(r, 'a column cut inside a block is refused')

# ---- the two tapes -------------------------------------------------------------------------------------------------------
nl = info['layers']
good, xi, why = True, [], []
for c in (0, 1):
    E_ = m.E[c]; ops = [sg.op for sg in E_.segs]
    good &= E_.module == 'k3_q3m_lean_tp' and all(o in k3.Q3M_TP_OPS for o in ops) and not E_.unsupported() and E_.refuse is None
    good &= all(sg.dep < i and sg.dep2p1 - 1 < i for i, sg in enumerate(E_.segs))
    good &= len(E_.segs) == 1 + 15 * nl + 2
    xs = [(i, sg) for i, sg in enumerate(E_.segs) if sg.op == k3.OP_XCHG]
    xi.append(sorted(sg.K for _, sg in xs)); good &= all(sg.work == c and sg.N == info['D'] and not sg.pad1 for _, sg in xs) and len(xs) == 2 * nl
    good &= all(sg.W == m.slots[1 - c].slot(sg.K) and sg.gs == m.slots[c].slot(sg.K) and sg.asum == m.epoch[c].data_ptr() for _, sg in xs)   # (push to the peer's slot, read its own, its own epoch word)
    for i, sg in xs:
        if sg.K % 2:   # (the FFN exchange: its partial is the slot sum's output)
            p = E_.segs[sg.dep]
            good &= p.op == k3.OP_OWN_SLOT_SUM and p.pad1 == info['KU'] and p.N == info['D']
    for i, sg in enumerate(E_.segs):
        if sg.op == k3.OP_OWN_ADD_RMS16 and i > 1:   # (segment 1: layer 0's norm, nothing to add)
            good &= sg.pad1 == 1 and E_.segs[sg.dep].op == k3.OP_XCHG   # (one row: the exchange's sum; no weights)
    good &= ops.count(k3.OP_OWN_ATTS) == nl and ops.count(k3.OP_ROUTER) == nl and ops.count(k3.OP_OWN_SLOT_SUM) == nl
    if not good: why.append(c)
check(good and xi[0] == xi[1] == list(range(2 * nl)),
      f'both tapes on k3_q3m_lean_tp run only Q3M_TP_OPS, {1 + 15 * nl + 2} segments (15 per layer), every dependency points back, '
      f'2 exchanges of D floats per layer (indices 0 .. {2 * nl - 1} once, the card\'s own number, push-only), the FFN one after '
      f'OP_OWN_SLOT_SUM over the top-{info["KU"]} slots, each residual add after an exchange one row' + (f' (cards {why})' if why else ''))
sh = True
for il, dd in enumerate(m.L):
    for c in (0, 1):
        d = dd[c]; rc, kvc = m.exp_rows[c], m.kvc[c]
        sh &= (d['gate']['rows'], d['up']['rows'], d['down']['K'], d['down']['rows']) == (info['E'] * rc, info['E'] * rc, rc, info['E'] * info['D'])
        sh &= (d['q']['rows'], d['k']['rows'], d['v']['rows'], d['o']['K'], d['o']['rows']) == (kvc * 8 * 128, kvc * 128, kvc * 128, kvc * 8 * 128, info['D'])
        sh &= (d['router']['rows'], d['router']['K']) == (info['E'], info['D']) and d['qk_par'].numel() == 2 * 128 + 2
        sh &= d['cache'].numel() == (64 + k3.Engine.ATS_PAD) * 2 * kvc * 128 and m.state[(c, il, 'kv')].numel() == 64 * 2 * kvc * 128
sh &= m.head[0]['rows'] == m.vh and m.head[1]['rows'] == m.vocab - m.vh and m.exp_rows == [512, 256] and m.kvc == [2, 2]
seg_ats = [[sg for sg in m.E[c].segs if sg.op == k3.OP_OWN_ATTS] for c in (0, 1)]
sh &= all(sg.ntask == m.splits[c] * m.kvc[c] for c in (0, 1) for sg in seg_ats[c]) and m.splits == [q3moe_tp.splits_for(2, q3moe_tp.SPLITS_CAP)] * 2
check(sh, f'each card\'s matrices have its share\'s shapes (expert rows {m.exp_rows}, kv heads {m.kvc}, head {m.vh} / {m.vocab - m.vh}); '
      f'its KV cache holds its kv heads at max_ctx + {k3.Engine.ATS_PAD} rows; the attention takes splits_for(its kv heads) = {m.splits}')
grids = []
n_slots = FakeSlots.n_made; slots0 = [s.base for s in m.slots]
for pw in (3, 6, 7):
    m.per_wgp = pw; m._build_tapes()
    grids.append(([e.grid for e in m.E], m.splits[:], [[(sg.op, sg.ntask, sg.dep, sg.K, sg.N) for sg in e.segs] for e in m.E]))
m.per_wgp = 'card'; m._build_tapes()
base = [[(sg.op, sg.ntask, sg.dep, sg.K, sg.N) for sg in e.segs] for e in m.E]
check(FakeSlots.n_made == n_slots and [s.base for s in m.slots] == slots0 and all(g[1] == m.splits and g[2] == base for g in grids)
      and [g[0][0] for g in grids] == [3 * 42, 6 * 42, 7 * 42],
      'the tapes build again at 3 / 6 / 7 per WGP on the same receive slots, with the same segments and attention splits (the grid alone changes)')
# the splits cap: 128 lets each card's 2 kv heads take its whole grid (the stand-in's 42 WGPs at 5 per WGP: 105 splits)
m.splits_cap = 128; m._build_tapes()
seg128 = [[sg for sg in e.segs if sg.op == k3.OP_OWN_ATTS] for e in m.E]
c128 = (m.splits == [q3moe_tp.splits_for(2, 128)] * 2 == [105, 105] and q3moe_one.card_per_wgp() == 5
        and all(sg.work == 105 and sg.ntask == 105 * 2 for c in (0, 1) for sg in seg128[c])
        and all(m.buf[c]['ats'].numel() == k3.Engine.atts_scratch_floats(2, 105) for c in (0, 1)))
m.splits_cap = q3moe_tp.SPLITS_CAP; m._build_tapes()
check(c128 and m.splits == [105, 105] and q3moe_tp.SPLITS_CAP == 128,
      f'splits_cap 128: each card takes splits_for(2, 128) = 105 splits (the 210-workgroup grid over 2 kv heads), its scratch for them; '
      f'the default cap is 128 ({m.splits})')
refused = []
for cap in (0, 129):
    try: Q3MoETP(P_T, devices=(0, 1), max_ctx=64, vram_check=False, splits_cap=cap); refused.append(False)
    except ValueError: refused.append(True)
check(all(refused) and k3.atts_max_splits('k3_q3m_lean_tp') == 128, 'Q3MoETP refuses splits_cap 0 and 129 (1 .. 128 on k3_q3m_lean_tp)')
m.reset()
check((m.seq, m.fresh, m.pos_host) == (1, True, 0) and {k[0] for k in m.state} == {0, 1} and {k[2] for k in m.state} == {'kv'}
      and m.tp_key == dict(exp_rows=512, kv_heads=2, vocab_rows=m.vh) and m.commit_max() == 1 and m.shard_devs == [0, 1],
      'reset(): a new sequence; KV states on both cards; tp_key names the split; one token per pass')

# ---- host refusals before a launch --------------------------------------------------------------------------------------
ran = []
for e in m.E: e.run = lambda *a, **k: ran.append(1)
res = []
for f in (lambda: (setattr(m, 'pos_host', 64), m.step(3)), lambda: m.step_token([3] * 70, 64), lambda: m.verify([3] * 70, 64, 1),
          lambda: m.launch_tokens([3] * 70, 64, 1)):
    try: f(); res.append(False)
    except k3.ContextBoundsError: res.append(True)
m.pos_host = 0
try: m.step(info['V']); res.append(False)
except ValueError: res.append(True)
check(all(res) and not ran and m.poisoned is None, 'a span past max_ctx (step, step_token, verify, launch_tokens) and a token outside the '
      'vocabulary are refused on the host before any launch; nothing is poisoned')

# ---- projection ---------------------------------------------------------------------------------------------------------
p0, p1 = Q3MoETP.projection_mib(P_T, 64), Q3MoETP.projection_mib(P_T, 65)
kvp = Q3MoETP.kv_pos_mib(P_T)
F = ggufio.GGUFFile(P_T); tot = sum(t.n_bytes for t in F.tensors) / 2 ** 20
rep = sum(t.n_bytes for t in F.tensors if q3moe_tp._role(t.name) == 'rep')
T = projection_terms(F, 64); T2 = projection_terms(F, 100000)
F.close()
check(all(abs((b - a) - k) < 1e-9 for a, b, k in zip(p0, p1, kvp)) and all(abs(k * 2 ** 20 - nl * 4 * 2 * 128) < 1e-6 for k in kvp),
      f'the projection grows by each card\'s KV row ({kvp[0] * 1024:.2f} / {kvp[1] * 1024:.2f} KiB: its 2 kv heads of each layer)')
up_split = [sum(w.numel() * w.element_size() for n, c_, k, w, a in m.uploads if c_ == c and k == 'cut' and q3moe_tp._role(n) != 'rep') for c in (0, 1)]
# (the uploads hold the q / k norms as one block per layer with theta_scale and eps: 8 bytes more per layer)
up_rep = [sum(w.numel() * w.element_size() for n, c_, k, w, a in m.uploads if c_ == c and q3moe_tp._role(n) == 'rep') - 8 * nl for c in (0, 1)]
good = all(t['rep'] == rep == ur for t, ur in zip(T, up_rep)) and all(abs(t['split'] - us) < 1e-6 for t, us in zip(T, up_split))
good &= all(t['slots'] == 2 * nl * (16 + info['D']) * 4 for t in T) and all(t['kv'] == nl * (64 + k3.Engine.ATS_PAD) * 4 * 2 * 128 for t in T)
good &= all(t['ats'] == 4 * k3.Engine.atts_scratch_floats(2, q3moe_tp.SPLITS_CAP) for t in T)
check(good, f'the projection\'s terms per card: its slices ({up_split[0]} / {up_split[1]} B, the bytes the engine uploads), the replicated '
      f'tensors ({rep} B on each), its kv heads\' KV at max_ctx + {k3.Engine.ATS_PAD} rows, the ATS scratch, {2 * nl} receive slots of 16 words + D floats')
w = [t['split'] / 2 ** 20 + t['rep'] / 2 ** 20 for t in T]
check(tot < w[0] + w[1] < 2 * tot, f'the two cards\' weights add up to more than the file ({tot:.2f} MiB: the replicated tensors) and less than '
      f'twice it ({w[0]:.2f} + {w[1]:.2f} MiB)')
try:
    Q3MoETP(P_T, devices=(0, 1), max_ctx=info.get('n_ctx', 4096) + 1, vram_check=False); r = False
except ValueError:
    r = True
check(r and T2[0]['kv'] > T[0]['kv'], 'past the file\'s context_length the projection is computed (arithmetic), and the engine refuses it')
F = ggufio.GGUFFile(P_T); T128 = projection_terms(F, 64, splits_cap=128); F.close()
p128 = Q3MoETP.projection_mib(P_T, 64, splits_cap=128); p0 = Q3MoETP.projection_mib(P_T, 64, splits_cap=64)
d_ats = 4 * (k3.Engine.atts_scratch_floats(2, 128) - k3.Engine.atts_scratch_floats(2, 64)) / 2 ** 20
check(all(t['ats'] == 4 * k3.Engine.atts_scratch_floats(2, 128) for t in T128) and all(abs((b - a) - d_ats) < 1e-9 for a, b in zip(p0, p128)),
      f'the projection at splits_cap 128 counts its ATS scratch: {d_ats * 1024:.0f} KiB more per card than at 64')

# ---- the load: the capacity check first, the pinned uploader by default ---------------------------------------------------
calls = {}


class FakeUploader:
    def __init__(self, devs, mode=None, log=None, what=''):
        EVENTS.append('uploader'); calls['mode'] = mode; calls['log'] = log; self.mode = mode; self.n_up = 0; self.n_bytes = 0

    def up(self, a, name, card, role):
        EVENTS.append('up'); self.n_up += 1; self.n_bytes += a.nbytes; return torch.from_numpy(a)

    def begin(self, *a): pass

    def failed(self, *a): pass

    def finish(self): pass


real_check, real_up = capacity.check, upload.Uploader
capacity.check = lambda devs, need, what: (EVENTS.append('check'), calls.update(devs=list(devs), need=list(need)))[-1]
upload.Uploader = FakeUploader
os.environ.pop('CAMINUS_H2D')
EVENTS = []
try:
    m2 = Q3MoETP(P_T, devices=(0, 1), max_ctx=64, vram_extra_mib=(5.0, 7.0))
finally:
    capacity.check, upload.Uploader = real_check, real_up
    os.environ['CAMINUS_H2D'] = 'pageable'
ev, EVENTS = EVENTS, None
pj = Q3MoETP.projection_mib(P_T, 64)
check(ev[0] == 'check' and ev.count('check') == 1 and calls['devs'] == [0, 1]
      and all(abs(a - (b + e)) < 1e-9 for a, b, e in zip(calls['need'], pj, (5.0, 7.0))) and 'alloc' in ev and 'up' in ev and 'slots' in ev,
      f'capacity.check runs once on both cards with the projection + what follows ({calls["need"][0]:.1f} / {calls["need"][1]:.1f} MiB) before the '
      f'first allocation, upload or receive slot ({ev[:3]} ...)')
check(calls['mode'] == 'pinned' == upload.TWO_CARD_DEFAULT and isinstance(calls['log'], upload.UploadLog) and m2.h2d.n_up == len(m2.uploads),
      'with CAMINUS_H2D unset the uploader is built with \'pinned\' (upload.TWO_CARD_DEFAULT) and the upload log from the environment; every '
      'upload is kept for the read-back')
host_eq = all(np.array_equal(np.ascontiguousarray(m.host_bytes(n, c, k, a)).reshape(-1).view(np.uint8),
                             w.contiguous().view(-1).view(torch.uint8).numpy()) for n, c, k, w, a in m.uploads)
kinds = {k for _, _, k, _, _ in m.uploads}
check(host_eq and kinds == {'cut', 'vec', 'qkpar', 'emb'}, f'every kept tensor\'s host_bytes equals its upload ({len(m.uploads)} tensors, kinds {sorted(kinds)})')
pb = m.physics_bytes()
check(len(pb) == 2 and pb[0] > pb[1] > 0, f'physics_bytes per card: {pb[0] / 2 ** 20:.2f} / {pb[1] / 2 ** 20:.2f} MiB a pass')

# ---- tools/gates/tp_load_only.py picks the engine by the file's architecture ------------------------------------------------
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools', 'gates'))
import tp_load_only
_empty, _sync_real, _room = torch.empty, torch.cuda.synchronize, capacity.host_kv_room
torch.empty = lambda *a, **k: _empty(*a, **{x: v for x, v in k.items() if x not in ('device', 'pin_memory')})
torch.cuda.synchronize = lambda d=None: None
capacity.host_kv_room = lambda: (65536.0, 65536.0)   # (the read-back buffer's pinned check sees a roomy host)
try:
    lo = tp_load_only.load_only(P_T, vram_check=False)   # (no card to read: the check itself is tested above)
    n, nb, badr = tp_load_only.readback(lo)
    cuts = [u for u in lo.uploads if u[2] == 'cut']   # (a cut: each card's own array; on the stand-ins a replicated upload is one array)
    name, c, kind, w, _ = cuts[len(cuts) // 2]
    w.contiguous().view(-1).view(torch.uint8)[w.numel() * w.element_size() // 2] ^= 1        # (one bit of one byte)
    n2, _, bad2 = tp_load_only.readback(lo)
finally:
    torch.empty, torch.cuda.synchronize, capacity.host_kv_room = _empty, _sync_real, _room
check(isinstance(lo, Q3MoETP) and type(lo).__name__ == 'Q3MLoadOnly' and not hasattr(lo, 'E') and lo.slots is None
      and tp_load_only.arch_of(P_T) == 'qwen3moe' and n == len(lo.uploads) and not badr and len(bad2) == 1 and bad2[0][:2] == (name, c)
      and tp_load_only.result_line(1, lo.h2d.n_up, badr, lo.h2d.mode).startswith('RESULT PASS'),
      f'tp_load_only on a qwen3moe file: Q3MoETP without tapes or receive slots; {n} tensors ({nb} B) read back equal, a planted '
      f'one-byte mismatch found')
del lo

# ---- the prompt cache on the served engine (caminus serve --cards 2): per-card KV rows, slot size, key ----------------------
from types import SimpleNamespace as _NS
from caminus.cache import prompt as _prompt
m3 = Q3MoETP(P_T, devices=(0, 1), max_ctx=64, vram_check=False)
L = _prompt.Layout(m3)
row = [k3.kv_pos_bytes(m3.kvc[c] * m3.hd, False) for c in (0, 1)]
slot_mib = [v * 1.05 + 1.0 for v in Q3MoETP.kv_pos_mib(P_T)]          # (arch.Q3MoETP.auto_need's checkpoint slot per card)
check(L.devs == [0, 1] and L.T == 1 and L.kv_rows == 64 and L.pb == row and all(len(L.kv[c]) == m3.nl for c in (0, 1))
      and all(not L.state[c] for c in (0, 1)) and L.df_card is None and not L.idx
      and L.rows_bytes(40) == [m3.nl * 40 * row[c] for c in (0, 1)]
      and all(L.nbytes[c] >= m3.nl * row[c] and L.nbytes[c] / 2 ** 20 <= slot_mib[c] for c in (0, 1)),
      f'cache layout: each card\'s own KV rows ({row} B per position per layer: its kv heads), {m3.nl} layers, no recurrent '
      f'state; a checkpoint slot ({L.nbytes} B) within the registry\'s slot projection ({[round(v, 3) for v in slot_mib]} MiB); '
      f'park / restore rows per card (rows_bytes)')
key = _prompt.engine_key(m3, _NS(numerics=lambda: dict(q='x'), crossover=8), build='fixture')
check(key['module'] == 'k3_q3m_lean_tp' and key['tp']['devices'] == [0, 1] and key['tp']['exp_rows'] == m3.exp_rows0 == 512
      and key['tp']['kv_heads'] == 2 and key['tp']['vocab_rows'] == m3.vh and key['tp']['ntok'] == 1
      and key['module'] != q3moe_one.MODULE_LEAN,
      'cache key: the two-card module, both devices and the split (exp_rows, kv_heads, vocab rows): a one-card park on the '
      'same cache directory (caminus start --cards 1 / 2 share it) is never matched')
del m3

# ---- the cache round trip on it: each card's KV rows through a checkpoint slot, a park and an unpark after a restart ---------
# (the server's own path: PromptCache.commit / extend / restore / park on the real engine's KV views. Every (card, layer)
# gets its own random bytes, and both cards' rows have the same size, so only the bytes catch a swap or a one-row offset)
RT_CTX, RT_N, RT_MORE = 256, 240, 12               # (n / PREFILL_TOK_S > a disk load's fixed 0.05 s: restore() picks the disk)
m5 = Q3MoETP(P_T, devices=(0, 1), max_ctx=RT_CTX, vram_check=False)
m6 = Q3MoETP(P_T, devices=(0, 1), max_ctx=RT_CTX, vram_check=False)   # (the engine after a restart: built now, its KV zeros)
_is_cuda = torch.Tensor.is_cuda
torch.Tensor.is_cuda = property(lambda self: False)   # (the segment copies: host tensors, no device stream or event)
try:
    rt_dir = os.path.join(A.scratch, 'cache-tp')
    rt_ids = [int(x) for x in np.random.default_rng(5).integers(0, info['V'], RT_N + RT_MORE)]

    def rt_cache(eng):
        pf = _NS(m=eng, numerics=lambda: dict(q='x'), crossover=8)
        return _prompt.PromptCache(eng, pf, None, cache_dir=rt_dir, n_slots=2, vram_check=False, side_rows=0, min_free_gb=0,
                                   key=_prompt.engine_key(eng, pf, build='fixture'))

    def rt_rows(eng, c, il, a, b):
        pb = _prompt.Layout(eng).pb[c]
        return _prompt._u8(eng.state[(c, il, 'kv')])[a * pb:b * pb]

    def rt_fill(eng, a, b, salt):
        for c in (0, 1):
            for il in range(eng.nl):
                v = rt_rows(eng, c, il, a, b)
                v.copy_(torch.from_numpy(np.random.default_rng((salt, c, il)).integers(0, 256, v.numel(), dtype=np.uint8)))

    def rt_diff(eng, ref, a, b):
        """the (card, layer) pairs whose rows [a, b) differ from ref"""
        pb = _prompt.Layout(eng).pb
        return [(c, il) for c in (0, 1) for il in range(eng.nl)
                if not torch.equal(rt_rows(eng, c, il, a, b), ref[c, il][a * pb[c]:b * pb[c]])]

    L5 = _prompt.Layout(m5)
    pc5 = rt_cache(m5)
    rt_fill(m5, 0, RT_N, 1)                                   # (as a prefill of RT_N tokens would leave the rows)
    ref = {(c, il): rt_rows(m5, c, il, 0, RT_N).clone() for c in (0, 1) for il in range(m5.nl)}
    check(L5.pb[0] == L5.pb[1] and not L5.state[0] and not L5.state[1] and len({bytes(t.numpy()) for t in ref.values()}) == len(ref),
          f'round trip: {RT_N} positions, each of the {len(ref)} (card, layer) KV views its own bytes, the same row size on both cards '
          f'({L5.pb} B), no recurrent state')
    pc5.commit(rt_ids[:RT_N], dict(i0=RT_N, commit=1), checkpoint=True)     # (a checkpoint slot: the T = 1 tail row RT_N - 1)
    rt_fill(m5, RT_N, RT_N + RT_MORE, 2)
    pc5.extend(rt_ids, dict(i0=RT_N + RT_MORE, commit=1))
    for c in (0, 1):
        for il in range(m5.nl): rt_rows(m5, c, il, RT_N - 1, RT_N).zero_()
    r = pc5.restore(rt_ids[:RT_N])
    d_slot = rt_diff(m5, ref, 0, RT_N)
    check(r['source'] == 'vram' and r['cached'] == RT_N and r['st'] == dict(i0=RT_N, commit=1) and not d_slot,
          f'checkpoint slot: the tail row {RT_N - 1} of every card and layer, cleared, comes back from the slot ({r["source"]}, '
          f'{r["cached"]} positions); rows [0, {RT_N}) equal per card and layer (differ: {d_slot})')
    nm = pc5.park()
    check(nm is not None and pc5.park_error is None, f'park: the live state ({RT_N}, 1) written ({nm})')
    for c in (0, 1):
        for il in range(m5.nl): rt_rows(m5, c, il, 0, RT_CTX).zero_()
    pc5.reset()
    r = pc5.restore(rt_ids[:RT_N])
    d_disk = rt_diff(m5, ref, 0, RT_N)
    check(r['source'] == 'disk' and r['cached'] == RT_N and not d_disk,
          f'unpark in the same process: every KV row cleared and the slots dropped, restore() reads the entry ({r["source"]}, '
          f'{r["cached"]} positions); rows [0, {RT_N}) equal per card and layer (differ: {d_disk})')
    pc5.close(); del pc5, m5
    # a restart: a new engine (its KV zeros) and a new cache on the same directory, as caminus serve --cards 2 starts again
    check(all(not rt_rows(m6, c, il, 0, RT_CTX).any() for c in (0, 1) for il in range(m6.nl)), 'the new engine\'s KV rows are zeros')
    pc6 = rt_cache(m6)
    r = pc6.restore(rt_ids[:RT_N])
    d_restart = rt_diff(m6, ref, 0, RT_N)
    check(r['source'] == 'disk' and r['cached'] == RT_N and r['st'] == dict(i0=RT_N, commit=1) and pc6.live_tokens() == rt_ids[:RT_N]
          and not d_restart,
          f'unpark after a restart: a new engine and cache on the same directory read the entry ({r["source"]}, {r["cached"]} '
          f'positions, state {r["st"]}); rows [0, {RT_N}) equal per card and layer (differ: {d_restart})')
    # negative controls: the same unpark with the disk reads misdirected (card 0 and card 1 swapped on layer 0; card 1's layer
    # 1 one row late): the comparison names exactly those views
    real_read = pc6.store.read

    def misread(how):
        def read(name, h, wanted):
            w = dict(wanted)
            if 'kv0.c0' in w and how == 'swap': w['kv0.c0'], w['kv0.c1'] = wanted['kv0.c1'], wanted['kv0.c0']
            if 'kv1.c1' in w and how == 'offset':
                pb = L5.pb[1]; n = wanted['kv1.c1'][0].numel()
                w['kv1.c1'] = [_prompt._u8(m6.state[(1, 1, 'kv')])[pb:pb + n]]
            return real_read(name, h, w)
        return read

    seen = {}
    for how, want in (('swap', [(0, 0), (1, 0)]), ('offset', [(1, 1)])):
        for c in (0, 1):
            for il in range(m6.nl): rt_rows(m6, c, il, 0, RT_CTX).zero_()
        pc6.reset(); pc6.store.read = misread(how)
        try: r = pc6.restore(rt_ids[:RT_N])
        finally: pc6.store.read = real_read
        seen[how] = (r['source'], rt_diff(m6, ref, 0, RT_N))
        check(seen[how] == ('disk', want), f'negative control ({how}): the unpark with misdirected reads differs in {seen[how][1]} (want {want})')
    pc6.close(); del pc6, m6
finally:
    torch.Tensor.is_cuda = _is_cuda

# ---- the own-code path imports no gguf-py ---------------------------------------------------------------------------------
BLOCK = ("import sys\nclass B:\n    def find_spec(self, name, path=None, target=None):\n"
         "        if name.split('.')[0] in ('gguf', 'tokenizers'): raise ImportError('blocked: ' + name)\n"
         "sys.meta_path.insert(0, B()); sys.path.insert(0, %r)\n"
         "import caminus.engine.q3moe_tp\nprint('IMPORTED')\n") % os.path.join(HERE, '..', '..', 'python')
r = subprocess.run([sys.executable, '-I', '-c', BLOCK], capture_output=True, text=True, timeout=120)
check(r.returncode == 0 and 'IMPORTED' in r.stdout, 'the two-card engine imports with gguf-py and tokenizers refused'
      + ('' if r.returncode == 0 else f' ({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode})'))

print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
