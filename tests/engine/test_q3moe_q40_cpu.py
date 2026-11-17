"""The qwen3moe Q4_0 family without a card (k3.Q3M_Q40_OF: kernels/k3_q3m_q40*.hip, the official GPTQ-Int4 checkpoint
converted by tools/eval/gptq_to_gguf.py; its prefill object kernels/q3m_prefill_q40.hip), on stand-ins for the torch.cuda
and HIP calls (every tensor in host memory; nothing is launched), with the tiny synthetic file in its Q4_0 form
(tests/engine/tiny_q3m.py q40=True):
  - the tables: every lean, Q8 KV, YaRN, two-card and verify build has one twin; each table holds the twin exactly where
    it holds the base, paired with the twins of the base's partners; the twin's (op, format) pairs are the base's with the
    Q4_0 tile's forms for the Q4_K tile's; its kernel file is K3_Q3M_Q40 over the base's file; the serving twins and
    q3m_prefill_q40 are in the installed runtime set, the trace and verify twins are not
  - own_matvec_1rt: Q4_0 and Q6_K on a twin, Q4_K refused there (the twin compiles no Q4_K tile); Q4_0 refused on the
    shipped lean module
  - the one-card engine on the Q4_0 file: k3_q3m_q40 (q8: k3_q3m_q40_kvq8; the long-context mode: k3_q3m_q40_yarn), every
    attention and expert matrix as engine.own_layout.q4_0_planes of the file's rows on the device, OWN_MV segments of the
    Q4_0 work id with the base tape's op sequence; the shipped module named on the Q4_0 file is refused at build; a
    row-scale module is used only when named
  - the batched prefill: Q3MPrefill picks q3m_prefill_q40, its Q4_0 GEMMs launch q3p_mm_q40 with the buffer's rows; the
    Q8 cache's and the long-context mode's qk_rope / attention from their own objects
  - the two-card engine on the Q4_0 file: k3_q3m_q40_tp (q8: k3_q3m_q40_tp_kvq8), each card's cut matrices as the planes
    of the cut; Q3MPrefillTP on q3m_prefill_q40
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/engine/test_q3moe_q40_cpu.py --scratch DIR"""
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
from caminus import ggufio, rope
from caminus.device import capacity, hip, kernels
from caminus.engine import k3, q3moe_tp
from caminus.engine.own_layout import q4_0_planes
from caminus.engine.q3moe_one import Q3MoEOne
from caminus.engine.q3moe_tp import Q3MoETP
from caminus.prefill import q3m, q3m_ops, q3m_tp

capacity.check = lambda *a, **k: None
LOG = []   # (card, object, kernel, grid, args)


class Rec:
    def __init__(self, card, name): self.card, self.name = card, name

    def __getattr__(self, kern): return lambda grid, block, *args, **kw: LOG.append((self.card, self.name, kern, tuple(grid), args))


hip.Module = lambda name: Rec(0, name)


class FakeHip:
    def __getattr__(self, n): return lambda *a, **k: 0


class FakeSlots:
    n = 0

    def __init__(self, dev, n, N): FakeSlots.n += 1; self.base = FakeSlots.n << 40; self.stride = (16 + N) * 4

    def slot(self, i): return self.base + i * self.stride


q3moe_tp._hip = lambda: FakeHip()
q3moe_tp.Slots = FakeSlots


class FakeOps(q3m_ops.Ops):
    """q3m_ops.Ops with its code objects recorders (its launch wrappers' host checks run)"""
    def __init__(self, dev, module='q3m_prefill', q8=False, yarn=False):
        self.dev, self.module, self.q8, self.yarn = torch.device(dev), module, bool(q8), bool(yarn)
        card = int(str(dev).split(':')[1])
        self.mod = Rec(card, module); self.kvmod = Rec(card, q3m_ops.KVQ8_MODULE) if self.q8 else None
        self.ropemod = Rec(card, q3m_ops.YARN_MODULE[self.q8]) if self.yarn else (self.kvmod if self.q8 else self.mod)


class FakeSlotSum(q3m_tp.SlotSum):
    def __init__(self, dev): self.dev = dev; self.mod = Rec(int(str(dev).split(':')[1]), q3m_tp.SLOT_OBJ)


class FakeX:
    def __init__(self, devices, T, D, streams, check=True): self.calls = []

    def post(self, parts, r0, r1): self.calls.append(('post', r0, r1)); return ('t', len(self.calls))

    def sum(self, t, mark=None): self.calls.append(('sum', t))

    def settle(self): pass


q3m_tp.q3m_ops.Ops = FakeOps; q3m_tp.SlotSum = FakeSlotSum; q3m_tp.Exchange = FakeX

# ---- the tables -------------------------------------------------------------------------------------------------------------
OF, BASE = k3.Q3M_Q40_OF, k3.Q3M_Q40_BASE
bases = [m for m in k3.Q3M_LEAN_MODULES + k3.Q3M_TP_MODULES + k3.Q3M_VER_MODULES if m not in BASE]
check(sorted(OF) == sorted(bases) and len(set(OF.values())) == len(OF) and all(BASE[OF[b]] == b for b in OF)
      and tuple(OF.values()) == k3.Q3M_Q40_MODULES,
      f'one twin for each of the {len(OF)} lean, Q8 KV, YaRN, two-card and verify builds (k3.Q3M_Q40_OF; Q3M_Q40_BASE its inverse)')
mem = lambda m: tuple(m in t for t in (k3.Q3M_LEAN_MODULES, k3.Q3M_TP_MODULES, k3.Q3M_VER_MODULES, k3.Q3M_KV_Q8_MODULES,
                                        k3.Q3M_YARN_MODULES, k3.Q3M_MODULES, k3.Q3M_IQ_MODULES, tuple(k3.Q3M_RS_MODULES)))
pairs = lambda tbl: all((OF.get(tbl[b]) == tbl.get(OF[b])) if b in tbl else OF[b] not in tbl for b in OF)
check(all(mem(OF[b]) == mem(b) for b in OF) and pairs(k3.Q3M_KV_Q8_OF) and pairs(k3.Q3M_KV_Q8_BASE) and pairs(k3.Q3M_TP_OF)
      and pairs(k3.Q3M_YARN_OF) and pairs(k3.Q3M_YARN_BASE) and all(k3.atts_max_splits(OF[b]) == k3.atts_max_splits(b) for b in OF),
      'every table holds each twin where it holds its base (lean, two-card, verify, Q8 KV, YaRN, all qwen3moe), paired with '
      'the twins of the base\'s partners (Q8 KV, two-card, YaRN, both directions); the same attention split cap')
swap = lambda c: frozenset((op, f.replace('Q4_K/', 'Q4_0/', 1) if op == 'OWN_MV' else f) for op, f in c)
check(all(k3.MODULE_CAPS[OF[b]] == swap(k3.MODULE_CAPS[b]) and not any(f.startswith('Q4_K/') for op, f in k3.MODULE_CAPS[OF[b]] if op == 'OWN_MV')
          and ('OWN_EMB', 'Q4_K') in k3.MODULE_CAPS[OF[b]] for b in OF),
      'each twin\'s (op, format) pairs: the base\'s with Q4_0/... for every Q4_K/... matvec form (none left), the Q4_K embedding kept')
src = {t: open(os.path.join(ROOT, 'kernels', t + '.hip')).read().splitlines() for t in OF.values()}
check(all([ln for ln in src[OF[b]] if not ln.startswith('//')] == ['#define K3_Q3M_Q40', f'#include "{b}.hip"'] for b in OF),
      'each twin\'s kernel file: K3_Q3M_Q40 over its base\'s file, nothing else')
serve = [OF[b] for b in OF if b in kernels.Q3_OWN]
check(len(serve) == 8 and all(t in kernels.RUNTIME_SET for t in serve) and 'q3m_prefill_q40' in kernels.RUNTIME_SET
      and not any(OF[b] in kernels.RUNTIME_SET for b in OF if b not in kernels.Q3_OWN),
      f'the serving twins ({len(serve)}) and q3m_prefill_q40 in the installed runtime set; the trace and verify twins outside it')

# ---- own_matvec_1rt ----------------------------------------------------------------------------------------------------------
E = k3.Engine(module='k3_q3m_q40', grid=8)
B = lambda n, dt=torch.uint8: torch.zeros(n, dtype=dt)
w = B(512 * 2048 // 32 * 18); x = B(2048, torch.float16); y = B(512, torch.float32)
check(not raises(lambda: E.own_matvec_1rt(w, 'Q4_0', 512, 2048, x, y)) and E.segs[-1].work == 14
      and not raises(lambda: E.own_matvec_1rt(B(512 * 8 * 210), 'Q6_K', 512, 2048, x, y))
      and raises(lambda: E.own_matvec_1rt(B(512 * 8 * 144), 'Q4_K', 512, 2048, x, y))
      and raises(lambda: k3.Engine(module='k3_q3m_lean', grid=8).own_matvec_1rt(w, 'Q4_0', 512, 2048, x, y)),
      'own_matvec_1rt: Q4_0 (work id 14) and Q6_K on the twin, Q4_K refused there; Q4_0 refused on the shipped lean module')

# ---- the one-card engine -------------------------------------------------------------------------------------------------------
P = os.path.join(A.scratch, 'tiny-q3m-q40.gguf'); tiny_q3m.write(P, q40=True)
PK = os.path.join(A.scratch, 'tiny-q3m.gguf'); tiny_q3m.write(PK)
MC = 64
mk = lambda path=P, **kw: Q3MoEOne(path, max_ctx=MC, vram_check=False, link_keepalive=False, **kw)
m, mq, mb = mk(), mk(q8=True), mk(PK)
F = ggufio.GGUFFile(P)
lay = all(np.array_equal(m.L[il][r][3].numpy(), q4_0_planes(np.asarray(F.T[f'blk.{il}.{r}.weight'].data()), *F.T[f'blk.{il}.{r}.weight'].rows()[:1],
                                                              int(F.T[f'blk.{il}.{r}.weight'].shape[0])))
          and m.L[il][r][0] == 'Q4_0' for il in range(m.nl) for r in ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'))
mv = [sg for sg in m.E.segs if sg.op == k3.OP_OWN_MV]
check(m.module == m.E.module == 'k3_q3m_q40' and mq.module == 'k3_q3m_q40_kvq8' and lay
      and [sg.op for sg in m.E.segs] == [sg.op for sg in mb.E.segs] and {sg.work for sg in mv} == {14, 2}
      and sum(sg.work == 2 for sg in mv) == 1,
      'Q3MoEOne on the Q4_0 file: k3_q3m_q40 (q8: k3_q3m_q40_kvq8), every attention and expert matrix as q4_0_planes of its '
      'rows; the Q4_K file\'s op sequence; OWN_MV segments of the Q4_0 id (14) and one Q6_K (the head)')
my = mk(rope=rope.YarnMode(4.0, 32768))
check(my.module == 'k3_q3m_q40_yarn' and mk(q8=True, rope=rope.YarnMode(4.0, 32768)).module == 'k3_q3m_q40_kvq8_yarn',
      'the long-context mode on the Q4_0 file: k3_q3m_q40_yarn (q8: k3_q3m_q40_kvq8_yarn)')
check(raises(lambda: mk(module='k3_q3m_lean')) and raises(lambda: mk(attn_split=False)) and raises(lambda: mk(module='k3_q3m_rs4')),
      'refused: the shipped lean module named on the Q4_0 file (its build refuses the Q4_0 pairs), another tape form, and the '
      'row-scale module named (its layout refuses a row with more than one scale: the file is not a row-scale file)')

# ---- the batched prefill --------------------------------------------------------------------------------------------------------
pf, pq, py = q3m.Q3MPrefill(m, chunk=64, check=False), q3m.Q3MPrefill(mq, chunk=64, check=False), q3m.Q3MPrefill(my, chunk=64, check=False)
check(pf.ops.module == pq.ops.module == py.ops.module == 'q3m_prefill_q40' and pq.ops.kvmod.name == 'q3m_prefill_kvq8'
      and py.ops.ropemod.name == 'q3m_prefill_yarn' and q3m.Q3MPrefill(mb, chunk=64, check=False).ops.module == 'q3m_prefill',
      'Q3MPrefill on the Q4_0 engine: q3m_prefill_q40 (the Q8 rows and attention from q3m_prefill_kvq8, the YaRN qk_rope from '
      'q3m_prefill_yarn); the Q4_K engine keeps q3m_prefill')
LOG.clear(); pf._chunk(list(range(1, 41)), 0, 40)
mm40 = [e for e in LOG if e[2] == 'q3p_mm_q40']
rows = {(e[4][1].value, e[4][2].value) for e in mm40}
check(mm40 and all(e[1] == 'q3m_prefill_q40' for e in mm40) and not any(e[2] == 'q3p_mm_q4k' for e in LOG)
      and any(e[2] == 'q3p_mm_q6k' for e in LOG) is False and all(rt >= r for r, rt in rows)
      and (m.D, m.D) in rows and (m.ffn, m.ffn * m.n_exp) in rows,
      f'a prefill chunk: every matrix GEMM is q3p_mm_q40 on q3m_prefill_q40 with its buffer\'s rows (rows / buffer rows {sorted(rows)}), '
      f'no Q4_K or Q6_K GEMM (the head runs in the decode pass)')
check(raises(lambda: q3m_ops.Ops('cuda:0', 'q3m_prefill').mm(m.L[0]['attn_q'], x, 2048, y, 512, n=1)),
      'the shipped prefill object refuses a Q4_0 matrix')

# ---- the two-card engine --------------------------------------------------------------------------------------------------------
PT = os.path.join(A.scratch, 'tiny-q3m-q40-tp.gguf'); tiny_q3m.write(PT, tp=True, q40=True)
t, tq = Q3MoETP(PT, devices=(0, 1), max_ctx=MC, vram_check=False), Q3MoETP(PT, devices=(0, 1), max_ctx=MC, vram_check=False, q8=True)
FT = ggufio.GGUFFile(PT)
cut = all(any(u[0] == name and u[2] == 'cut' for u in t.uploads) for name in (f'blk.0.ffn_gate_exps.weight', f'blk.1.attn_output.weight'))
hb = [t.host_bytes(u[0], u[1], u[2], u[4]) for u in t.uploads if u[2] == 'cut' and FT.T[u[0]].type == 'Q4_0']
check(t.module == 'k3_q3m_q40_tp' and tq.module == 'k3_q3m_q40_tp_kvq8' and cut and hb
      and all(np.array_equal(np.asarray(u[3].cpu()).reshape(-1).view(np.uint8), b.reshape(-1).view(np.uint8))
              for u, b in zip([u for u in t.uploads if u[2] == 'cut' and FT.T[u[0]].type == 'Q4_0'], hb)),
      f'Q3MoETP on the Q4_0 file: k3_q3m_q40_tp (q8: k3_q3m_q40_tp_kvq8); each card\'s cut Q4_0 matrices on the device are the '
      f'planes of the cut, as host_bytes rebuilds them ({len(hb)} uploads)')
pt = q3m_tp.Q3MPrefillTP(t, chunk=64, check=False)
check(pt.module == 'q3m_prefill_q40' and all(o.module == 'q3m_prefill_q40' for o in pt.O),
      'Q3MPrefillTP on the two-card Q4_0 engine: q3m_prefill_q40 on both cards')
print('RESULT', 'PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
