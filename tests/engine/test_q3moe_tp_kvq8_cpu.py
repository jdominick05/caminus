"""The qwen3moe two-card engine's Q8 KV option without a card (caminus.engine.q3moe_tp.Q3MoETP(q8=True), its batched prefill
caminus.prefill.q3m_tp.Q3MPrefillTP and the 'qwen3moe-tp' registry entry), on stand-ins for the torch.cuda and HIP calls
(every tensor in host memory; nothing is launched), with the tiny synthetic file with two-card shapes (tests/engine/tiny_q3m.py
tp=True: 4 kv heads, 2 per card).
  - the module tables: k3_q3m_lean_tp_kvq8 is k3_q3m_lean_tp's Q8 build (Q3M_KV_Q8_BASE) and k3_q3m_lean_kvq8's two-card
    form (Q3M_TP_OF): the two-card modules' ops whitelist and (op, format) pairs, 128 attention splits, an error word after
    Sync on its Engine; not a lean (one-card) module; in the installed runtime set (served: caminus serve --cards 2 --kv q8); its kernels/*.hip file defines the
    Q8 switch and the add_rms16 pin over the two-card module and nothing else
  - the engine: k3_q3m_lean_tp_kvq8 on both cards, each card's caches uint8 (max_ctx + ATS_PAD) rows of its 2 kv heads'
    Q8 row (544 bytes), the prompt cache's state views max_ctx rows of bytes, every OP_OWN_ATTS segment on its layer's cache
    of that card, both tapes the F16 engine's ops in order; kv_format_id = the one-card engine's KV_FORMAT_Q8; the
    projection counts Q8 rows (the engine's own check, and the F16 projection less exactly the row bytes saved per card);
    kv_pos_mib per card
  - refusals: q8 not a bool; q8 on the F16 two-card module (by either name) or the _trace build; the Q8 module without
    q8; the one-card engine on the two-card Q8 module without q8 (with q8 only set by name: the one-card probe); a float16 cache on the two-card Q8 module's own_atts (and a uint8
    one on the F16 module)
  - a refused encoding: card 1's error word set with a stalled wait on card 0, read back by _pass_launch's one status copy per
    card (9 words on a Q8 engine, 8 on F16): _pass_wait raises InvalidKVEncoding before the bail path (no bail recorded,
    the words not cleared), the engine stays broken across reset() and refuses every pass with that class; non-finite
    logits still mark it with NonFiniteLogits
  - the prefill on a Q8 engine: each card's Ops with q8 (q3m_prefill_kvq8 beside q3m_prefill), qk_rope / attn launched as
    q3p_qk_rope_q8 (with that card's KV error word) / q3p_attn_q8 over 2 kv heads, no F16 row or attention kernel, rows
    checked against the byte caches; numerics() names the cache (F16: unchanged); either card's error word set: the
    settle raises InvalidKVEncoding before the routing flags, the engine is broken, the next chunk is refused
  - the prompt cache: the checkpoint layout's row on each card is that card's Q8 row; the engine key carries the KV format
    and the Q8 module (the F16 two-card key's tp entries unchanged: no 'kv'), so neither restores into the other
  - the park round trip (caminus serve --cards 2 --kv q8; tests/engine/kv_park_rt.py): each card's and layer's Q8 byte rows
    back from a checkpoint slot, from a park in the same process and after a restart, byte for byte, rows past the restored
    positions still zero; negative controls (card 0's and card 1's layer 0 swapped, card 1's layer 1 read a row late)
    differ in exactly those views (the late one also past the positions). An F16 two-card
    park is not offered to a Q8 two-card server and a Q8 one not to an F16 one (cold, rows untouched); the park's own
    format restores it (the positive control)
  - the registry: 'qwen3moe-tp' takes opt['kv'] = 'q8' (kv_formats); auto_need counts the Q8 projection and checkpoint
    slots per card (less than F16 by exactly the rows and slots saved); build passes q8 to the engine; every entry but the
    two qwen3moe ones refuses 'q8'
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/engine/test_q3moe_tp_kvq8_cpu.py --scratch DIR"""
import argparse, contextlib, os, sys, types
os.environ['CAMINUS_H2D'] = 'pageable'   # (CPU stand-ins: the uploads are tests/engine/test_q3moe_tp_cpu.py's)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'python')); sys.path.insert(0, HERE)
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); A = ap.parse_args()
os.makedirs(A.scratch, exist_ok=True)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


def raises(fn, exc=ValueError):
    try: fn()
    except exc: return True
    except Exception as e: print('   (raised', type(e).__name__, e, ')'); return False
    return False


# ---- stand-ins: torch.cuda / HIP on host memory (no device is opened) ------------------------------------------------------
_zeros, _to, _tensor = torch.zeros, torch.Tensor.to, torch.tensor
torch.zeros = lambda *a, **k: _zeros(*a, **{x: v for x, v in k.items() if x not in ('device', 'pin_memory')})
torch.tensor = lambda *a, **k: _tensor(*a, **{x: v for x, v in k.items() if x != 'device'})
torch.Tensor.to = lambda self, *a, **k: self if a and str(a[0]).startswith('cuda') else _to(self, *a, **k)
torch.Tensor.cuda = lambda self, *a, **k: self
torch.Tensor.is_cuda = property(lambda self: True)
torch.cuda.set_device = lambda d: None
torch.cuda.device = lambda d: contextlib.nullcontext()
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda d: types.SimpleNamespace(multi_processor_count=42, total_memory=20464 << 20)
torch.cuda.get_device_name = lambda d=None: 'stand-in'
torch.cuda.current_stream = lambda d=None: types.SimpleNamespace(device=torch.device(str(d)) if d is not None else None)
torch.cuda.synchronize = lambda d=None: None


class FakeEv:
    def record(self, *a): pass

    def synchronize(self): pass


torch.cuda.Event = lambda *a, **k: FakeEv()

import tiny_q3m
from caminus import arch, errors
from caminus.cache import prompt
from caminus.device import capacity, kernels
from caminus.engine import k3, q3moe_tp, q3moe_one
from caminus.engine.q3moe_one import Q3MoEOne
from caminus.engine.q3moe_tp import Q3MoETP
from caminus.prefill import q3m_tp, q3m_ops


class FakeHip:
    def __getattr__(self, n): return lambda *a, **k: 0


class FakeSlots:
    n = 0

    def __init__(self, dev, n, N): FakeSlots.n += 1; self.base = FakeSlots.n << 40; self.stride = (16 + N) * 4

    def slot(self, i): return self.base + i * self.stride


q3moe_tp._hip = lambda: FakeHip()
q3moe_tp.Slots = FakeSlots
LOG = []   # (card, object, kernel, grid, args)


class Rec:
    def __init__(self, card, name): self.card, self.name = card, name

    def __getattr__(self, kern): return lambda grid, block, *args, **kw: LOG.append((self.card, self.name, kern, tuple(grid), args))


class FakeOps(q3m_ops.Ops):
    """q3m_ops.Ops with its code objects recorders (its launch wrappers' host checks run)"""
    def __init__(self, dev, module='q3m_prefill', q8=False):
        self.dev, self.module, self.q8 = torch.device(dev), module, bool(q8)
        card = int(str(dev).split(':')[1])
        self.mod = Rec(card, module); self.kvmod = Rec(card, q3m_ops.KVQ8_MODULE) if self.q8 else None


class FakeSlotSum(q3m_tp.SlotSum):
    def __init__(self, dev): self.dev = dev; self.mod = Rec(int(str(dev).split(':')[1]), q3m_tp.SLOT_OBJ)


class FakeX:
    def __init__(self, devices, T, D, streams, check=True): self.calls = []

    def post(self, parts, r0, r1): self.calls.append(('post', r0, r1)); return ('t', len(self.calls))

    def sum(self, t, mark=None): self.calls.append(('sum', t))

    def settle(self): pass


q3m_tp.q3m_ops.Ops = FakeOps; q3m_tp.SlotSum = FakeSlotSum; q3m_tp.Exchange = FakeX
capacity.check = lambda *a, **k: None

# ---- the module tables -------------------------------------------------------------------------------------------------------
TPQ = 'k3_q3m_lean_tp_kvq8'
check(k3.Q3M_KV_Q8_BASE == {'k3_q3m_lean_kvq8': 'k3_q3m_lean', TPQ: 'k3_q3m_lean_tp'}
      and k3.Q3M_KV_Q8_OF['k3_q3m_lean_tp'] == TPQ and k3.Q3M_TP_OF['k3_q3m_lean_kvq8'] == TPQ
      and TPQ in k3.Q3M_TP_MODULES and TPQ in k3.Q3M_KV_Q8_MODULES and TPQ in k3.Q3M_MODULES and TPQ not in k3.Q3M_LEAN_MODULES
      and k3.MODULE_CAPS[TPQ] == k3.MODULE_CAPS['k3_q3m_lean_tp'] and k3.atts_max_splits(TPQ) == 128,
      f'{TPQ}: k3_q3m_lean_tp\'s Q8 build and k3_q3m_lean_kvq8\'s two-card form: the two-card ops and pairs, 128 attention splits, '
      f'not a one-card lean module')
check(TPQ in kernels.RUNTIME_SET and 'k3_q3m_lean_kvq8' in kernels.RUNTIME_SET and 'q3m_prefill_kvq8' in kernels.RUNTIME_SET
      and 'k3_q3m_lean_tp' in kernels.RUNTIME_SET,
      'in the installed runtime set with the one-card Q8 module and q3m_prefill_kvq8 (served: --cards 2 --kv q8); the F16 two-card '
      'module stays in it')
src = [ln.strip() for ln in open(os.path.join(ROOT, 'kernels', TPQ + '.hip')) if ln.strip() and not ln.strip().startswith('//')]
check(src == ['#define K3_Q3M_KV_Q8', '#define K3_Q3M_ADD_RMS_PIN', '#include "k3_q3m_lean_tp.hip"'],
      f'kernels/{TPQ}.hip: the Q8 switch and the add_rms16 pin over the two-card module, nothing else')
E = k3.Engine(module=TPQ, grid=8, T=1)
check(E.sync.numel() == k3.SYNC_WORDS + 1 and E.kv_error is not None and E.kv_error.data_ptr() == E.sync.data_ptr() + 4 * k3.SYNC_WORDS
      and E.status_words().numel() == k3.STATUS_WORDS + 1 and k3.Engine(module='k3_q3m_lean_tp', grid=8).kv_error is None,
      'its Engine: one int32 error word after Sync, in the status words\' one copy (the F16 two-card Engine has none)')

# ---- the engine on the tiny file ------------------------------------------------------------------------------------------------
P = os.path.join(A.scratch, 'tiny-q3m-tp.gguf'); info = tiny_q3m.write(P, tp=True)
MC = 64
m = Q3MoETP(P, devices=(0, 1), max_ctx=MC, vram_check=False, q8=True)
f = Q3MoETP(P, devices=(0, 1), max_ctx=MC, vram_check=False)
rb = [k3.kv_pos_bytes(k * m.hd, True) for k in m.kvc]
check(m.module == TPQ and all(e.module == TPQ for e in m.E) and f.module == 'k3_q3m_lean_tp' and m.kv_format_id == q3moe_one.KV_FORMAT_Q8
      and not hasattr(f, 'kv_format_id') and m.kvc == [2, 2] and rb == [544, 544],
      f'Q3MoETP(q8=True) decodes on {TPQ} on both cards with kv_format_id {m.kv_format_id!r} (the one-card format); 2 kv heads per '
      f'card: a {rb[0]}-byte Q8 row (F16 1,024)')
good = True
for c in (0, 1):
    ats = [sg for sg in m.E[c].segs if sg.op == k3.OP_OWN_ATTS]
    good &= all(d[c]['cache'].dtype == torch.uint8 and d[c]['cache'].numel() == (MC + k3.Engine.ATS_PAD) * rb[c] for d in m.L)
    good &= all(m.state[(c, il, 'kv')].numel() == MC * rb[c] and m.state[(c, il, 'kv')].dtype == torch.uint8 for il in range(m.nl))
    good &= len(ats) == m.nl and [sg.W for sg in ats] == [d[c]['cache'].data_ptr() for d in m.L]
    good &= [sg.op for sg in m.E[c].segs] == [sg.op for sg in f.E[c].segs] and m.E[c].refuse is None and m.E[c].kv_error is not None
check(good, f'each card: uint8 caches of (max_ctx + {k3.Engine.ATS_PAD}) rows of {rb[0]} B, the state views max_ctx rows of bytes, '
            f'each OP_OWN_ATTS on its layer\'s cache, the tape the F16 engine\'s ops in order, an error word on its Engine')
pq_ = Q3MoETP.projection_mib(P, MC, q8=True); pf_ = Q3MoETP.projection_mib(P, MC)
saved = [m.nl * (MC + k3.Engine.ATS_PAD) * (2 * 2 * k * m.hd - b) / 2 ** 20 for k, b in zip(m.kvc, rb)]
check(all(abs(a - b) < 1e-9 for a, b in zip(m.vram_mib, pq_)) and all(abs(a - b - s) < 1e-9 for a, b, s in zip(pf_, pq_, saved))
      and all(abs(a - m.nl * b / 2 ** 20) < 1e-12 for a, b in zip(Q3MoETP.kv_pos_mib(P, q8=True), rb))
      and all(abs(a - m.nl * 4 * k * m.hd / 2 ** 20) < 1e-12 for a, k in zip(Q3MoETP.kv_pos_mib(P), m.kvc)),
      f'projection per card: the engine\'s own check = projection_mib(q8=True); F16 - Q8 = the row bytes saved '
      f'({saved[0]:.4f} / {saved[1]:.4f} MiB here); kv_pos_mib counts Q8 rows')

# ---- refusals ---------------------------------------------------------------------------------------------------------------------
mk = lambda **kw: Q3MoETP(P, devices=(0, 1), max_ctx=MC, vram_check=False, **kw)
check(raises(lambda: mk(q8=1)) and raises(lambda: mk(q8=True, module='k3_q3m_lean')) and raises(lambda: mk(q8=True, module='k3_q3m_lean_tp'))
      and raises(lambda: mk(q8=True, module='k3_q3m_lean_trace')) and raises(lambda: mk(module=TPQ)) and raises(lambda: mk(module='k3_q3m_lean_kvq8'))
      and mk(q8=True, module='k3_q3m_lean_kvq8').module == TPQ and mk(q8=True, module=TPQ).module == TPQ,
      'refused: q8 not a bool, q8 on the F16 two-card module (either name) or the _trace build, the Q8 module (either name) '
      'without q8; q8 with the Q8 module named (either name) runs it')
P1 = os.path.join(A.scratch, 'tiny-q3m.gguf'); tiny_q3m.write(P1)
mo = Q3MoEOne(P1, max_ctx=MC, vram_check=False, link_keepalive=False, q8=True)
mo._module = TPQ   # (the one-card probe's switch: tests/engine/test_q3m_tp_probe.py --kv q8 replays the tape on it)
check(raises(lambda: Q3MoEOne(P1, max_ctx=MC, vram_check=False, link_keepalive=False, module=TPQ)) and mo.module == TPQ
      and raises(lambda: Q3MoEOne(P1, max_ctx=MC, vram_check=False, link_keepalive=False, q8=True, module='k3_q3m_lean_tp')),
      'the one-card engine refuses the two-card Q8 module without q8 (and the F16 two-card module with q8); with q8 it takes it '
      'only when set by name (the one-card probe)')
E2, Ef = k3.Engine(module=TPQ, grid=8), k3.Engine(module='k3_q3m_lean_tp', grid=8)
B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt)
M2 = 2 * 128
args = lambda cache: (B(16 * 128 + 2 * M2), B(k3.Engine.atts_scratch_floats(2, 100)), B(258), cache, B(1, torch.int32),
                      B(16 * 128, torch.float16), 16, 2, 128, 0.1)
check(raises(lambda: E2.own_atts(*args(B((MC + 64) * 2 * M2, torch.float16)), splits=100))
      and raises(lambda: Ef.own_atts(*args(B((MC + 64) * 544, torch.uint8)), splits=100))
      and not raises(lambda: E2.own_atts(*args(B((MC + 64) * 544, torch.uint8)), splits=100))
      and raises(lambda: E2.own_atts(*args(B((MC + 64) * 544, torch.uint8)), splits=129)),
      'own_atts on the two-card Q8 module: a float16 cache refused (and a uint8 one on the F16 module); a Q8 cache at 100 splits '
      'builds (past the one-card 64), 129 refused')

# ---- a refused encoding --------------------------------------------------------------------------------------------------------
mm = mk(q8=True)
check([h.numel() for h in mm.h_st] == [k3.STATUS_WORDS + 1] * 2 and [h.numel() for h in f.h_st] == [k3.STATUS_WORDS] * 2,
      'the pinned status blocks: 9 words per card on the Q8 engine (the error word), 8 on the F16 one (unchanged)')
for e in mm.E: e.run = lambda check=True, trace=None: None   # (no launch: the copies below run on the host tensors)
mm._stage(1, 5); mm._tok = False
mm.E[1].kv_error.fill_(2); mm.E[0].sync[2 * k3.MAXSEG + 1] = 1; mm.E[0].sync[2 * k3.MAXSEG + 2] = 0   # (and card 0's wait gave up)
mm._pass_launch(5, 0)
mm.E[1].kv_error.zero_()   # (the live word cleared after the launch's copy: only the pinned copy still holds 2, so a live read misses it)
ITEM = torch.Tensor.item; reads = []
torch.Tensor.item = lambda self: reads.append(1) or ITEM(self)
r = raises(lambda: mm._pass_wait(5), errors.InvalidKVEncoding)
torch.Tensor.item = ITEM
check(r and mm.broken and mm._broken_error is errors.InvalidKVEncoding and not mm.bails and int(mm.E[1].kv_error[0]) == 0
      and int(mm.h_st[1][k3.STATUS_WORDS]) == 2 and int(mm.h_st[0][k3.STATUS_WORDS]) == 0 and not reads and 'card 1' in mm.broken,
      'card 1\'s refused block with card 0\'s wait given up: InvalidKVEncoding from the launch\'s one status copy per card (the live word '
      'zeroed after the copy), before the bail path; no .item() read back, no bail; the broken note names the card')
mm.reset()
toks = [1] * 8
check(mm.broken and raises(lambda: mm.step_token(toks, 0), errors.InvalidKVEncoding) and raises(lambda: mm.step(1), errors.InvalidKVEncoding)
      and raises(lambda: mm.verify(toks, 0, 0), errors.InvalidKVEncoding) and raises(lambda: mm.launch_tokens(toks, 0, 0), errors.InvalidKVEncoding),
      'reset() keeps the engine broken; step, step_token, verify and launch_tokens are refused with InvalidKVEncoding')
mn = mk(q8=True); mn.mark_broken('non-finite logits (test)')
check(mn._broken_error is errors.NonFiniteLogits and raises(lambda: mn.step_token(toks, 0), errors.NonFiniteLogits),
      'non-finite logits still mark the engine with NonFiniteLogits (mark_broken\'s default)')

# ---- the prefill on a Q8 engine ---------------------------------------------------------------------------------------------------
mp = mk(q8=True)
pf = q3m_tp.Q3MPrefillTP(mp, chunk=64, check=False)
pff = q3m_tp.Q3MPrefillTP(mk(), chunk=64, check=False)
check(pf.q8 and all(o.q8 and o.module == 'q3m_prefill' and o.kvmod.name == 'q3m_prefill_kvq8' for o in pf.O)
      and pf.numerics().get('kv') == 'q8_0' and 'kv' not in pff.numerics() and not any(o.q8 for o in pff.O)
      and all('kverr' in b for b in pf.B),
      'Q3MPrefillTP on a Q8 engine: each card\'s ops on q3m_prefill with q3m_prefill_kvq8 for the rows and attention, a KV error '
      'word per card; numerics name the cache (the F16 prefill\'s unchanged)')
LOG.clear()
seq = list(range(1, 41))
pf._chunk(seq, 0, 40)
rope = [e for e in LOG if e[2].startswith('q3p_qk_rope')]; att = [e for e in LOG if e[2].startswith('q3p_attn')]
check(rope and all(e[1] == 'q3m_prefill_kvq8' and e[2] == 'q3p_qk_rope_q8' and e[4][-1] is pf.B[e[0]]['kverr'] and e[3][1] == 16 + 2 for e in rope)
      and att and all(e[1] == 'q3m_prefill_kvq8' and e[2] == 'q3p_attn_q8' and e[3][1] == 2 for e in att)
      and sorted(e[0] for e in rope) == [0] * mp.nl + [1] * mp.nl and not any(e[2] in ('q3p_qk_rope', 'q3p_attn') for e in LOG)
      and all(e[4][7] is mp.L[il][e[0]]['cache'] for il, e in zip([i for i in range(mp.nl) for _ in (0, 1)], rope)),
      f'a chunk: {len(rope)} q3p_qk_rope_q8 (16 + 2 heads, the card\'s own KV error word, its layer\'s byte cache) and {len(att)} '
      f'q3p_attn_q8 (2 kv heads) launches; no F16 row or attention kernel')
pf.B[1]['kverr'].fill_(2); pf.B[0]['err'].fill_(1)   # (a routing flag too: the KV word is read first)
check(raises(pf._run_settle, errors.InvalidKVEncoding) and mp.broken and mp._broken_error is errors.InvalidKVEncoding and 'card 1' in mp.broken
      and int(pf.B[0]['err'].item()) == 1 and raises(lambda: pf._chunk(seq, 0, 40), errors.InvalidKVEncoding),
      'card 1\'s prefill error word set: the settle raises InvalidKVEncoding before the routing flags (left set), the engine is broken '
      'with that class, the next chunk is refused')

# ---- the prompt cache ----------------------------------------------------------------------------------------------------------------
from types import SimpleNamespace as NS
Lq, Lf = prompt.Layout(m), prompt.Layout(f)
check(Lq.pb == rb and Lf.pb == [4 * k * m.hd for k in m.kvc] and Lq.kv_rows == MC and not any(Lq.state),
      f'checkpoint layout per card: a Q8 row of {rb[0]} B per layer (F16 {Lf.pb[0]} B), no recurrent state')
pfk = NS(numerics=lambda: dict(q='x'), crossover=8)
kq, kf_ = prompt.engine_key(m, pfk, build='fixture'), prompt.engine_key(f, pfk, build='fixture')
check(kq['tp'].get('kv') == q3moe_one.KV_FORMAT_Q8 and kq['module'] == TPQ and kf_['module'] == 'k3_q3m_lean_tp'
      and set(kf_['tp']) == {'devices', 'names', 'gdn_keys', 'moe_blocks', 'vocab_share', 'ntok', 'local_only', 'exp_rows', 'kv_heads', 'vocab_rows'}
      and prompt.key_hash(kq) != prompt.key_hash(kf_),
      'engine key: the Q8 engine\'s carries its KV format and module; the F16 two-card key\'s tp entries are the ones it had (no kv), '
      'so neither restores into the other')
from caminus.cmd import replay
tpa = arch.ARCHS[arch.TWO_CARD['qwen3moe']]
check(replay.kv_of_key(kq) == 'q8' and replay.kv_of_key(kf_) == 'f16'
      and tpa.kv_check(dict(kv=replay.kv_of_key(kq))) is True and tpa.kv_check(dict(kv=replay.kv_of_key(kf_))) is False,
      'caminus replay: a two-card Q8 bundle\'s engine key builds the entry with kv q8, the F16 two-card key with f16')

# ---- the park round trip on the two-card Q8 engine (caminus serve --cards 2 --kv q8: each card's byte rows through a checkpoint
# slot, a park and an unpark after a restart), negative controls, and the format mismatch both ways (tests/engine/kv_park_rt.py) --
import kv_park_rt as RT
RT_CTX = 256
mk2 = lambda q8: (lambda: Q3MoETP(P, devices=(0, 1), max_ctx=RT_CTX, vram_check=False, q8=q8))


def rt_swap(w, eng):            # (card 0's and card 1's layer 0 read into each other's rows: the same row size)
    if 'kv0.c0' in w and 'kv0.c1' in w: w['kv0.c0'], w['kv0.c1'] = w['kv0.c1'], w['kv0.c0']
    return w


def rt_late(w, eng):            # (card 1's layer 1 read one row late)
    if 'kv1.c1' in w:
        pb = prompt.Layout(eng).pb[1]; n = w['kv1.c1'][0].numel()
        w['kv1.c1'] = [prompt._u8(eng.state[(1, 1, 'kv')])[pb:pb + n]]
    return w


RT.round_trip(mk2(True), check, os.path.join(A.scratch, 'cache-tp-q8'), info['V'], 'Q8 two cards',
              [('card 0 and card 1 swapped on layer 0', rt_swap, [(0, 0), (1, 0)], []),
               ('card 1\'s layer 1 one row late', rt_late, [(1, 1)], [(1, 1)])])
RT.mismatch(mk2(False), mk2(True), check, os.path.join(A.scratch, 'cache-tp-f16-then-q8'), info['V'], 'an F16 park, a Q8 server (two cards)')
RT.mismatch(mk2(True), mk2(False), check, os.path.join(A.scratch, 'cache-tp-q8-then-f16'), info['V'], 'a Q8 park, an F16 server (two cards)')

# ---- the registry -------------------------------------------------------------------------------------------------------------------
tp = arch.ARCHS[arch.TWO_CARD['qwen3moe']]
check(tp.kv_formats == ('f16', 'q8') and tp.kv_check({'kv': 'q8'}) is True and tp.kv_check({}) is False,
      f"'{tp.name}' takes opt['kv'] = 'q8' (kv_formats {tp.kv_formats})")
nq, dq, sq, needq = tp.auto_need({'model': P, 'kv': 'q8'}, 4)
nf, df_, sf, needf = tp.auto_need({'model': P}, 4)
sl_q = [v * 1.05 + 1.0 for v in Q3MoETP.kv_pos_mib(P, q8=True)]; sl_f = [v * 1.05 + 1.0 for v in Q3MoETP.kv_pos_mib(P)]
saved_a = [m.nl * (MC + k3.Engine.ATS_PAD) * (4 * k * m.hd - b) / 2 ** 20 for k, b in zip(m.kvc, rb)]
check(nq == nf and dq == df_ == [0, 1] and abs(sq - max(sl_q)) < 1e-12
      and all(abs((a - b) - (s + 6 * (x - y))) < 1e-9 for a, b, s, x, y in zip(needf(MC), needq(MC), saved_a, sl_f, sl_q)),
      'qwen3moe-tp auto_need with kv q8: Q8 checkpoint slots and the Q8 projection on each card (less than F16 by exactly the rows '
      'and 6 slots saved)')
got = {}


class Stop(Exception): pass


real_E = q3moe_tp.Q3MoETP


def fake_E(*a, **kw): got.update(kw); raise Stop()


q3moe_tp.Q3MoETP = fake_E
try:
    raises(lambda: tp.build({'model': P, 'max_ctx': MC, 'kv': 'q8'}), Stop); q8_seen = got.get('q8')
    got.clear(); raises(lambda: tp.build({'model': P, 'max_ctx': MC}), Stop); f16_seen = got.get('q8')
finally:
    q3moe_tp.Q3MoETP = real_E
check(q8_seen is True and f16_seen is False, f"qwen3moe-tp build passes q8 to the engine ({q8_seen} with kv q8, {f16_seen} without)")
bad = []
for name, a in arch.ARCHS.items():
    if name in ('qwen3moe', 'qwen3moe-tp'): continue
    for what, fn in (('build', lambda: a.build({'kv': 'q8'})), ('auto_need', lambda: a.auto_need({'kv': 'q8'}, 4))):
        try: fn(); bad.append((name, what, 'no error'))
        except ValueError as e:
            if "kv 'q8'" not in str(e): bad.append((name, what, str(e)[:80]))
        except Exception as e: bad.append((name, what, type(e).__name__ + ' ' + str(e)[:80]))
check(not bad, f"every other registry entry ({len(arch.ARCHS) - 2}) refuses opt['kv'] = 'q8' in build and auto_need before reading anything"
      + (f' (wrong: {bad})' if bad else ''))

print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
