"""Qwen3-30B-A3B's long-context mode (YaRN x4, caminus serve --long-context) without a card: the table, the module tables, the
engines (one card and two) and prefills on stand-ins for torch.cuda / HIP (host tensors, nothing launched) with the tiny
synthetic files (tests/engine/tiny_q3m.py), the registry's mode, caps, KV choice and refusals, the serve and start flags.
  - the table (caminus.rope.yarn_table): correction dims [23, 40] for 128 dims, base 1e6, 32,768; each pair's frequency
    within fp32 rounding of caminus.ref_q3moe's llama.cpp-statement YaRN (written separately: a wrong table fails here);
    mscale fp32(1 + 0.1 ln 4), the value llama.cpp's fp32 chain (get_mscale, the cancel, rope_yarn's multiply) gives
  - the module tables: each YaRN build has its base's memberships (lean / two-card / Q8 tables, MODULE_CAPS, the split
    cap), Q3M_YARN_OF covers exactly the four decode modules, no verify build; every YaRN object (and q3m_prefill_kvq8's
    Q8 attention) is in the installed runtime set; each module file is its define over its base, nothing else
  - the engines: rope=YarnMode picks the YaRN build (F16 / Q8, one card / two), every layer's qk_par is the native block +
    the table (323 floats), max_ctx past the file's context_length is taken in the mode and refused without it, rope_key set;
    the two-card projection counts the table; refusals: rope not a YarnMode, a YaRN module without rope, another tape form
    with rope, the verify engine on a YaRN engine, a short par block on a YaRN module (own_atts) and in the YaRN prefill
  - the prefills: q3p_qk_rope from q3m_prefill_yarn (Q8: q3p_qk_rope_q8 from q3m_prefill_kvq8_yarn), every other kernel on
    its own object (the Q8 attention on q3m_prefill_kvq8); numerics carry the rope (F16 numerics unchanged)
  - the engine key: a YaRN engine's carries tp.rope and its module; a native key has no rope entry (unchanged)
  - the registry: the mode is opt-in (--long-context, or an explicit max_ctx past 40,960), its cap 131,072 (more refused),
    auto_need's native_ctx the cap; the KV choice (F16 where the asked context fits, else Q8 where it holds more) and the
    note naming it; every other entry refuses --long-context; the Coder's needs an explicit max_ctx past 262,144
  - the CLI: caminus serve --long-context (and in options_json), caminus start passes it and refuses a model without a mode
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/engine/test_q3moe_yarn_cpu.py --scratch DIR"""
import argparse, contextlib, math, os, sys, types
os.environ['CAMINUS_H2D'] = 'pageable'   # (CPU stand-ins: the two-card uploads are tests/engine/test_q3moe_tp_cpu.py's)
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


def raises(fn, exc=ValueError):
    try: fn()
    except exc: return True
    except Exception as e: print('   (raised', type(e).__name__, e, ')'); return False
    return False


# ---- stand-ins: torch.cuda / HIP on host memory (as tests/engine/test_q3moe_kvq8_cpu.py) -----------------------------------------
_zeros, _to, _tensor = torch.zeros, torch.Tensor.to, torch.tensor
torch.zeros = lambda *a, **k: _zeros(*a, **{x: v for x, v in k.items() if x not in ('device', 'pin_memory')})
torch.tensor = lambda *a, **k: _tensor(*a, **{x: v for x, v in k.items() if x != 'device'})
torch.Tensor.to = lambda self, *a, **k: self if a and str(a[0]).startswith('cuda') else _to(self, *a, **k)
torch.Tensor.cuda = lambda self, *a, **k: self
torch.Tensor.is_cuda = property(lambda self: True)
torch.cuda.set_device = lambda d: None
torch.cuda.device = lambda d: contextlib.nullcontext()
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda d: types.SimpleNamespace(multi_processor_count=48, total_memory=24560 << 20)
torch.cuda.get_device_name = lambda d=None: 'stand-in'


class FakeEv:
    def record(self, *a): pass

    def synchronize(self): pass


torch.cuda.Event = lambda *a, **k: FakeEv()
torch.cuda.synchronize = lambda d=None: None

import tiny_q3m
from caminus import arch, rope
from caminus.cache import prompt
from caminus.device import capacity, hip, kernels
from caminus.device.context_fit import ContextFit
from caminus.engine import k3
from caminus.engine.q3moe_one import Q3MoEOne
from caminus.engine.q3moe_verify import Q3MoEVerify
from caminus.prefill import q3m, q3m_ops
from caminus.ref_q3moe import Q3MoERef

capacity.check = lambda *a, **k: None
LOG = []


class Rec:
    def __init__(self, name): self.name = name

    def __getattr__(self, kern): return lambda grid, block, *args: LOG.append((self.name, kern, tuple(grid), args))


hip.Module = lambda name: Rec(name)
Y = rope.YarnMode(4.0, 32768)

# ---- the table ---------------------------------------------------------------------------------------------------------------------
t = rope.yarn_table(128, 1e6, 4.0, 32768)
r = Q3MoERef.__new__(Q3MoERef); r.yarn, r.HD, r.base = (4.0, 32768), 128, 1e6
th, ms = r.yarn_angles(1.0)
f32 = np.float32
l4 = f32(np.log(f32(4)))
af = f32(f32(f32(f32(0.1) * l4) + f32(1)) * f32(f32(1) / f32(f32(1) + f32(f32(0.1) * l4))))   # (llama.cpp: get_mscale x the cancel)
ms_ll = f32(af * f32(f32(1) + f32(f32(0.1) * f32(np.log(f32(1) / f32(0.25))))))                  # (rope_yarn's multiply)
check(rope.corr_dims(128, 32768, 1e6) == (23.0, 40.0) and t.dtype == np.float32 and t.size == rope.TABLE_FLOATS == k3.YARN_TABLE_FLOATS
      and float(np.abs(th / t[:64].astype(np.float64) - 1).max()) < 1.2e-7 and abs(ms - float(t[64])) < 1e-7 and t[64] == ms_ll
      and Y.max_ctx == 131072 and arch.QWEN3MOE_ROPE.shipped == 'yarn' and arch.QWEN3MOE_ROPE.shipped_factor == 4.0,
      f'table: correction dims [23, 40]; frequencies within fp32 rounding of the llama.cpp-statement reference '
      f'(worst {float(np.abs(th / t[:64].astype(np.float64) - 1).max()):.1e}); mscale {t[64]!r} = llama.cpp\'s fp32 chain {ms_ll!r}; '
      f'the mode serves {Y.max_ctx} positions')
check(raises(lambda: rope.yarn_table(64, 1e6, 4.0, 32768)) and raises(lambda: rope.YarnMode(1.0, 32768)),
      'the table is for 128 dims only; a factor <= 1 is refused')

# ---- the module tables ------------------------------------------------------------------------------------------------------------
YO = k3.Q3M_YARN_OF
mem = lambda m: (m in k3.Q3M_LEAN_MODULES, m in k3.Q3M_TP_MODULES, m in k3.Q3M_KV_Q8_MODULES, m in k3.Q3M_MODULES,
                 k3.MODULE_CAPS.get(m), k3.atts_max_splits(m), m in k3.Q3M_VER_MODULES)
check(set(YO) == {'k3_q3m_lean', 'k3_q3m_lean_kvq8', 'k3_q3m_lean_tp', 'k3_q3m_lean_tp_kvq8'}
      and all(mem(y) == mem(b) for b, y in YO.items()) and all(k3.Q3M_YARN_BASE[y] == b for b, y in YO.items())
      and k3.Q3M_TP_OF[YO['k3_q3m_lean']] == YO['k3_q3m_lean_tp'] and k3.Q3M_TP_OF[YO['k3_q3m_lean_kvq8']] == YO['k3_q3m_lean_tp_kvq8']
      and k3.Q3M_KV_Q8_OF[YO['k3_q3m_lean']] == YO['k3_q3m_lean_kvq8'] and k3.Q3M_KV_Q8_OF[YO['k3_q3m_lean_tp']] == YO['k3_q3m_lean_tp_kvq8']
      and not any(y in k3.Q3M_VER_MODULES for y in YO.values()),
      'module tables: each YaRN build has its base\'s memberships (lean / two-card / Q8 / caps / split cap), the two-card and Q8 '
      'maps between YaRN builds, no verify build')
ship = ('k3_q3m_lean_yarn', 'k3_q3m_lean_kvq8_yarn', 'k3_q3m_lean_tp_yarn', 'k3_q3m_lean_tp_kvq8_yarn', 'q3m_prefill_yarn',
        'q3m_prefill_kvq8_yarn', 'q3m_prefill_kvq8')
check(all(n in kernels.RUNTIME_SET for n in ship) and 'k3_q3m_lean_kvq8' not in kernels.RUNTIME_SET
      and q3m_ops.YARN_MODULE == {False: 'q3m_prefill_yarn', True: 'q3m_prefill_kvq8_yarn'},
      'runtime set: the four YaRN decode objects, the two YaRN prefill objects and q3m_prefill_kvq8 (the Q8 attention the mode '
      'runs); the native Q8 decode module stays out')
src_ok = True
for name, (define, inc) in {'k3_q3m_lean_yarn': ('K3_Q3M_YARN', 'k3_q3m_lean.hip'), 'k3_q3m_lean_kvq8_yarn': ('K3_Q3M_YARN', 'k3_q3m_lean_kvq8.hip'),
                            'k3_q3m_lean_tp_yarn': ('K3_Q3M_YARN', 'k3_q3m_lean_tp.hip'), 'k3_q3m_lean_tp_kvq8_yarn': ('K3_Q3M_YARN', 'k3_q3m_lean_tp_kvq8.hip'),
                            'q3m_prefill_yarn': ('Q3P_YARN', 'q3m_prefill.hip'), 'q3m_prefill_kvq8_yarn': ('Q3P_YARN', 'q3m_prefill_kvq8.hip')}.items():
    src = [ln.strip() for ln in open(os.path.join(HERE, '..', '..', 'kernels', name + '.hip')) if ln.strip() and not ln.strip().startswith('//')]
    src_ok &= src == [f'#define {define}', f'#include "{inc}"']
check(src_ok, 'each YaRN module file is its define over its base, nothing else (the base objects keep their code)')

# ---- the engines (one card) -------------------------------------------------------------------------------------------------------
P = os.path.join(A.scratch, 'tiny-q3m.gguf'); tiny_q3m.write(P)
MC = 64
mf = Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False)
my = Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False, rope=Y)
mq = Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False, rope=Y, q8=True)
nat = 2 * 128 + 2
check(mf.module == 'k3_q3m_lean' and my.module == 'k3_q3m_lean_yarn' and my.E.module == 'k3_q3m_lean_yarn'
      and mq.module == 'k3_q3m_lean_kvq8_yarn' and mq.E.module == 'k3_q3m_lean_kvq8_yarn' and mq.E.kv_error is not None
      and all(d['qk_par'].numel() == nat for d in mf.L) and all(d['qk_par'].numel() == nat + 65 for d in my.L)
      and all(torch.equal(dy['qk_par'][:nat], df['qk_par']) and np.array_equal(dy['qk_par'][nat:].numpy(), my.rope_table) for dy, df in zip(my.L, mf.L))
      and np.array_equal(my.rope_table, rope.yarn_table(128, 1e6, 4.0, 32768)) and my.rope_key == Y.key() and not hasattr(mf, 'rope_key')
      and [sg.op for sg in my.E.segs] == [sg.op for sg in mf.E.segs],
      'Q3MoEOne(rope=YarnMode): k3_q3m_lean_yarn (Q8: k3_q3m_lean_kvq8_yarn, with its error word), every qk_par the native block + '
      'the table, rope_key set; the tape the native one\'s ops in order')
ats = [sg for sg in my.E.segs if sg.op == k3.OP_OWN_ATTS]
check(len(ats) == my.nl and [sg.aux for sg in ats] == [d['qk_par'].data_ptr() for d in my.L],
      'every OP_OWN_ATTS segment reads its layer\'s block with the table')
big = Q3MoEOne(P, max_ctx=8192, vram_check=False, link_keepalive=False, rope=Y)
check(big.max_ctx == 8192 and raises(lambda: Q3MoEOne(P, max_ctx=8192, vram_check=False, link_keepalive=False))
      and raises(lambda: Q3MoEOne(P, max_ctx=Y.max_ctx + 1, vram_check=False, link_keepalive=False, rope=Y)),
      'max_ctx past the file\'s context_length (4,096) is taken in the mode and refused without it; past the mode\'s 131,072 refused')
del big
check(raises(lambda: Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False, rope='yarn'))
      and raises(lambda: Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False, module='k3_q3m_lean_yarn'))
      and raises(lambda: Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False, rope=Y, attn_split=False))
      and raises(lambda: Q3MoEOne(P, max_ctx=MC, vram_check=False, link_keepalive=False, rope=Y, module='k3_gguf_q3m_own'))
      and raises(lambda: Q3MoEVerify(my, 2, vram_check=False)),
      'refused: rope not a YarnMode; a YaRN module without rope; rope with another tape form or module; the verify engine on a YaRN engine')
E2 = k3.Engine(module='k3_q3m_lean_yarn', grid=8)
B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt)
M = my.nkv * my.hd
args = lambda npar: (B(my.nh * 128 + 2 * M), B(k3.Engine.atts_scratch_floats(my.nkv, 8)), B(npar), B((MC + 64) * 2 * M, torch.float16),
                     B(1, torch.int32), B(my.nh * 128, torch.float16), my.nh, my.nkv, 128, 0.1)
check(raises(lambda: E2.own_atts(*args(nat), splits=8)) and not raises(lambda: E2.own_atts(*args(nat + 65), splits=8))
      and not raises(lambda: k3.Engine(module='k3_q3m_lean', grid=8).own_atts(*args(nat), splits=8)),
      'own_atts: a YaRN module refuses a par block without the table (the device reads past eps); the native module takes it as before')

# ---- the prefills -------------------------------------------------------------------------------------------------------------------
pf, pq, pn = q3m.Q3MPrefill(my, chunk=64, check=False), q3m.Q3MPrefill(mq, chunk=64, check=False), q3m.Q3MPrefill(mf, chunk=64, check=False)
toks = list(range(1, 41))
seen = {}
for name_, p_ in (('f16', pf), ('q8', pq), ('native', pn)):
    LOG.clear(); p_._chunk(toks, 0, 40)
    seen[name_] = ({(e[0], e[1]) for e in LOG if e[1].startswith('q3p_qk_rope')}, {(e[0], e[1]) for e in LOG if e[1].startswith('q3p_attn')},
                   {e[0] for e in LOG if not e[1].startswith(('q3p_qk_rope', 'q3p_attn'))})
check(seen['f16'] == ({('q3m_prefill_yarn', 'q3p_qk_rope')}, {('q3m_prefill', 'q3p_attn')}, {'q3m_prefill'})
      and seen['q8'] == ({('q3m_prefill_kvq8_yarn', 'q3p_qk_rope_q8')}, {('q3m_prefill_kvq8', 'q3p_attn_q8')}, {'q3m_prefill'})
      and seen['native'] == ({('q3m_prefill', 'q3p_qk_rope')}, {('q3m_prefill', 'q3p_attn')}, {'q3m_prefill'}),
      'prefill: q3p_qk_rope from q3m_prefill_yarn (Q8: q3p_qk_rope_q8 from q3m_prefill_kvq8_yarn), the attention and every other '
      'kernel on their own objects; the native prefill unchanged')
check(pf.numerics().get('rope') == Y.key() and pq.numerics().get('rope') == Y.key() and 'rope' not in pn.numerics()
      and raises(lambda: pf.ops.qk_rope(pf.qkv, 512, my.nh, my.nkv, mf.L[0]['qk_par'], 0, pf.q16, my.L[0]['cache'], 4))
      and raises(lambda: q3m.Q3MPrefill(my, chunk=64, check=False, module=q3m_ops.IQ_MODULE)),
      'prefill numerics carry the rope (native unchanged); the YaRN qk_rope refuses a par block without the table; an IQ object refused')

# ---- the engine key -------------------------------------------------------------------------------------------------------------------
from types import SimpleNamespace as NS
pfk = NS(numerics=lambda: dict(q='x'), crossover=8)
ky, kf = prompt.engine_key(my, pfk, build='fixture'), prompt.engine_key(mf, pfk, build='fixture')
check(ky['tp'].get('rope') == Y.key() and 'rope' not in kf['tp'] and ky['module'] == 'k3_q3m_lean_yarn' and prompt.key_hash(ky) != prompt.key_hash(kf),
      'engine key: a YaRN engine\'s carries tp.rope and its module; a native key has no rope entry (unchanged): no park crosses')

# ---- two cards ------------------------------------------------------------------------------------------------------------------------
from caminus.engine import q3moe_tp
from caminus.prefill import q3m_tp
PT = os.path.join(A.scratch, 'tiny-q3m-tp.gguf'); tiny_q3m.write(PT, tp=True)
torch.cuda.current_stream = lambda d=None: types.SimpleNamespace(device=torch.device(str(d)) if d is not None else None)


class FakeHip:
    def __getattr__(self, n): return lambda *a, **k: 0


class FakeSlots:   # (as tests/engine/test_q3moe_tp_kvq8_cpu.py)
    n = 0

    def __init__(self, dev, n, N): FakeSlots.n += 1; self.base = FakeSlots.n << 40; self.stride = (16 + N) * 4

    def slot(self, i): return self.base + i * self.stride


q3moe_tp._hip = lambda: FakeHip()
q3moe_tp.Slots = FakeSlots
tY = q3moe_tp.Q3MoETP(PT, devices=(0, 1), max_ctx=MC, vram_check=False, rope=Y)
tQ = q3moe_tp.Q3MoETP(PT, devices=(0, 1), max_ctx=MC, vram_check=False, rope=Y, q8=True)
tN = q3moe_tp.Q3MoETP(PT, devices=(0, 1), max_ctx=MC, vram_check=False)
tp_ok = (tY.module == 'k3_q3m_lean_tp_yarn' and tQ.module == 'k3_q3m_lean_tp_kvq8_yarn' and tN.module == 'k3_q3m_lean_tp'
         and tY._qk_par(0).size == nat + 65 and np.array_equal(tY._qk_par(0)[:nat], tN._qk_par(0))
         and np.array_equal(tY._qk_par(0)[nat:], tY.rope_table) and tY.rope_key == Y.key() and not hasattr(tN, 'rope_key')
         and all(E_.module == 'k3_q3m_lean_tp_yarn' for E_ in getattr(tY, 'E', []) if hasattr(E_, 'module')))
tp_note = 'built on stand-ins, F16 and Q8'
class FakeX:   # (the prefill's peer exchange: as tests/engine/test_q3moe_tp_kvq8_cpu.py)
    def __init__(self, devices, T, D, streams, check=True): self.calls = []

    def post(self, parts, r0, r1): self.calls.append(('post', r0, r1)); return ('t', len(self.calls))

    def sum(self, t, mark=None): self.calls.append(('sum', t))

    def settle(self): pass


q3m_tp.Exchange = FakeX
ptp = q3m_tp.Q3MPrefillTP(tY, chunk=64, check=False)
LOG.clear(); ptp._chunk(list(range(1, 41)), 0, 40)
tr = [e for e in LOG if e[1].startswith('q3p_qk_rope')]
check(ptp.yarn and len(tr) == 2 * tY.nl and all(e[0] == 'q3m_prefill_yarn' and e[1] == 'q3p_qk_rope' for e in tr)
      and all(e[3][4].numel() == nat + 65 for e in tr) and ptp.numerics().get('rope') == Y.key()
      and not any(e[1] == 'q3p_qk_rope' and e[0] != 'q3m_prefill_yarn' for e in LOG),
      f'Q3MPrefillTP on a YaRN engine: {len(tr)} q3p_qk_rope launches (both cards, every layer) from q3m_prefill_yarn on the '
      f'{nat + 65}-float block; numerics carry the rope')
F = __import__('caminus.ggufio', fromlist=['GGUFFile']).GGUFFile(PT)
tn, ty = q3moe_tp.projection_terms(F, MC), q3moe_tp.projection_terms(F, MC, yarn=True)
nl = int(F.hp('block_count'))
check(tp_ok and all(b['small'] - a['small'] == nl * 65 * 4 for a, b in zip(tn, ty))
      and raises(lambda: q3moe_tp.Q3MoETP(PT, devices=(0, 1), max_ctx=MC, vram_check=False, module='k3_q3m_lean_tp_yarn')),
      f'two cards: the YaRN two-card builds and the table in each block ({tp_note}); the projection counts 65 floats per layer; '
      f'a YaRN module without rope refused')

# ---- the registry -------------------------------------------------------------------------------------------------------------------
q, qt = arch.ARCHS['qwen3moe'], arch.ARCHS['qwen3moe-tp']
o0, o1, o2 = dict(model=P), dict(model=P, long_context=True), dict(model=P, max_ctx=50000)
check(q.long_mode(o0) is None and 'rope' not in o0 and q.rope_mode(o0) is None
      and q.long_mode(o1) and o1['rope'] == 'yarn' and isinstance(q.rope_mode(o1), rope.YarnMode)
      and q.long_mode(o2) and o2['rope'] == 'yarn' and o2['long_context'] is True
      and raises(lambda: q.long_mode(dict(model=P, max_ctx=131073))) and raises(lambda: qt.long_mode(dict(model=P, long_context=True, max_ctx=200000)))
      and q.long_mode(dict(model=P, max_ctx=4096)) is None,
      'mode: off by default; --long-context or an explicit max_ctx past the file\'s context_length turns it on (rope yarn); past '
      '131,072 refused; an explicit max_ctx within the file\'s context stays native')
n0 = q.auto_need(dict(model=P), 4)[0]; n1 = q.auto_need(dict(o1), 4)[0]
check(n0 == 4096 and n1 == 131072 and qt.auto_need(dict(model=PT, long_context=True, rope='yarn'), 4)[0] == 131072,
      'auto_need: native_ctx the file\'s context_length natively, the mode\'s 131,072 in the mode (one card and two)')
# the KV choice, on a stand-in room (F16 ~0.094 MiB per position, Q8 ~0.05: an XTX-like room of 4.25 GiB for the context)
_ARCH_FIT = arch.Arch.context_fit
arch.Arch.context_fit = lambda self, opt, cards=None: ContextFit(self._ctx_cap(opt), [0], 1.0, [4352.0],
                                                                    lambda c: [c * (0.0498 if arch.kv_q8(opt) else 0.0938)], arch.AUTO_STEP)
pa = q.context_fit(dict(o1)); pe = q.context_fit(dict(o1, max_ctx=40000)); pg = q.context_fit(dict(o1, max_ctx=70000))
pk = q.context_fit(dict(o1, kv='f16')); pn_ = q.context_fit(dict(model=P))
check(pa.kv == 'q8' and pa.maximum() == 86016 and pe.kv == 'f16' and pg.kv == 'q8' and pk.kv == 'f16' and pn_.kv == 'f16'
      and pn_.native_ctx == 4096,
      f'KV choice: auto picks Q8 where it holds more ({pa.maximum()} against F16\'s {pk.maximum()}); an explicit 40,000 stays F16, '
      f'70,000 takes Q8; an explicit kv is kept; native serves never switch')
ox = dict(o1); extra = q.placement_extra(ox, None); ox['max_ctx'] = 86016
note = q.long_context(ox)
check(extra == {'kv': 'q8'} and ox['kv'] == 'q8' and 'Q8_0' in note and 'chosen by the mode' in note and 'YaRN x4' in note
      and q.placement_extra(dict(model=P), None) == {} and q.long_context(dict(model=P, max_ctx=4096)) is None,
      'placement_extra writes the choice to opt[\'kv\'] once; the note names YaRN, the Q8 rows and why; native: nothing')
arch.Arch.context_fit = _ARCH_FIT
bad = []
for name, a in arch.ARCHS.items():
    if name in ('qwen3moe', 'qwen3moe-tp', 'qwen4exp'): continue
    try: a.long_mode(dict(model=P, long_context=True)); bad.append(name)
    except ValueError as e:
        if '--long-context' not in str(e): bad.append(name)
co = arch.ARCHS['qwen4exp']
check(not bad and raises(lambda: co.long_mode(dict(long_context=True, max_ctx='auto')))
      and raises(lambda: co.long_mode(dict(long_context=True, max_ctx=200000))) and co.long_mode(dict(long_context=True, max_ctx=1048576))
      and co.long_mode(dict(max_ctx=1048576)) is None and arch.ARCHS['qwen35-own'].long_mode(dict(model=P)) is None,
      'refusals: every other entry refuses --long-context; the Coder takes it only with an explicit --max-ctx past 262,144 (its '
      'NTK mode, chosen by that max_ctx as before)' + (f' (wrong: {bad})' if bad else ''))

# ---- the CLI ------------------------------------------------------------------------------------------------------------------------
from caminus.serve import cli
from caminus.cmd import manage
from caminus import models
opt = cli.parse(['--model', P, '--long-context'])
flags = {o['flag']: o for o in cli.options_json()}
check(opt['long_context'] is True and cli.parse(['--model', P])['long_context'] is False and flags['--long-context']['type'] == 'bool',
      'caminus serve --long-context (default off), listed in options_json')
m30 = models.BY_NAME.get('qwen3-30b-a3b')
e = dict(path=P, paths={}, present=True)
argv1 = manage.serve_argv(m30, e, 'x', 9000, long_context=True) if m30 else ['--long-context']
argv0 = manage.serve_argv(m30, e, 'x', 9000) if m30 else []
check('--long-context' in argv1 and '--long-context' not in argv0 and argv1.count('--long-context') == 1,
      'caminus start passes --long-context to serve when asked (once), never otherwise')
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
