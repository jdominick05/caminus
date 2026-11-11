"""The dense qwen35 own-kernel engine's host side without a card (caminus.engine.q35_dense, module k3_q35_swift): the tape
it builds for a tiny synthetic quantized model (tests/engine/tiny_q35.py), on stand-ins for the torch.cuda calls (every
tensor stays in host memory; nothing is launched).
  - the model file is one the float64 reference reads (finite logits), and the engine takes the reference's block kinds
  - the tape holds the tile each format needs (one-round-trip Q4_K / Q6_K, their fused gate | up, the rows tiles for
    Q8_0 / Q5_K, the F32 matvec) and only ops the module dispatches; every dependency points back; each DeltaNet
    block's output matvec waits for the block's save
  - the host's op list is the device's: the op names kernels/k3_q35_ops.h dispatches are k3.Q35S_OPS, and its three new
    op numbers are k3's
  - the options: gu=False has no fused segment, debug_layers builds a prefix, the projection grows by the KV row
  - refusals: the new ops on another module, a long add-and-norm row with weights, the q8 KV cache on this module
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/engine/test_q35_tape_cpu.py --scratch DIR"""
import argparse, contextlib, os, re, sys, types
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


# ---- stand-ins: the engine's torch.cuda calls on host memory (no device is opened) -------------------------------------
_zeros, _to = torch.zeros, torch.Tensor.to
torch.zeros = lambda *a, **k: _zeros(*a, **{x: v for x, v in k.items() if x not in ('device', 'pin_memory')})
torch.Tensor.to = lambda self, *a, **k: self if a and a[0] == 'cuda' else _to(self, *a, **k)
torch.Tensor.cuda = lambda self, *a, **k: self
torch.cuda.set_device = lambda d: None
torch.cuda.device = lambda d: contextlib.nullcontext()
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda d: types.SimpleNamespace(multi_processor_count=42, total_memory=20464 << 20)
torch.cuda.Event = lambda: types.SimpleNamespace()

import tiny_q35
from caminus import ggufio, ref_q35
from caminus.engine import k3, q35_dense
from caminus.engine.q35_dense import Dense35Own

P = os.path.join(A.scratch, 'tiny-q35-quant.gguf'); info = tiny_q35.write(P)
ref = ref_q35.Q35Ref(P); lg = ref.feed([1, 7, 3, 60, 7, 2])
check(lg.shape == (6, info['V']) and np.isfinite(lg).all() and np.abs(lg).max() > 1e-3, f'the reference reads the tiny quantized model (6 positions, |logit| up to {np.abs(lg).max():.2f})')
F = ggufio.GGUFFile(P)
check(q35_dense.block_kinds(F) == ref.gdn == info['kinds'] and q35_dense.trunk_blocks(F) == ref.L == info['blocks'],
      f'the engine takes the reference\'s blocks: {ref.L} trunk blocks, kinds {ref.gdn}')
F.close()

m = Dense35Own(P, max_ctx=64, vram_check=False)
E = m.E; inv = E.inventory(); ops = [sg.op for sg in E.segs]
want = ({('OWN_MV', f'{f}/a16_1rt{g}') for f in ('Q4_K', 'Q6_K') for g in ('', '_gu')} | {('OWN_MV16', 'Q8_0'), ('OWN_MV16', 'Q5_K')} |
        {('F32_MV', 'F32'), ('OWN_SWIGLU16', '-'), ('OWN_EMB', 'Q4_K'), ('OWN_ADD_RMS16', '-')})
check(set(inv) == want, f'the tape holds every tile: {sorted(inv.items())}')
check(not E.unsupported() and all(o in k3.Q35S_OPS for o in ops) and E.refuse is None, f'{len(ops)} segments, every op one the module dispatches')
check(all(sg.dep < i and sg.dep2p1 - 1 < i for i, sg in enumerate(E.segs)), 'every dependency points back')
n_gdn, n_att = sum(info['kinds']), len(info['kinds']) - sum(info['kinds'])
cnt = lambda o: ops.count(o)
check((cnt(k3.OP_GDN_CONV), cnt(k3.OP_GDN_L2), cnt(k3.OP_GDN_GATES), cnt(k3.OP_GDN_STEP), cnt(k3.OP_GDN_SAVE), cnt(k3.OP_GDN_NORM16)) == (n_gdn,) * 6
      and (cnt(k3.OP_QK_NR), cnt(k3.OP_KV_APPEND), cnt(k3.OP_ATTN_DEC), cnt(k3.OP_GATE_MUL16)) == (2 * n_att, n_att, n_att, n_att)
      and cnt(k3.OP_OWN_ADD_RMS16) == 2 * len(info['kinds']) + 1 and cnt(k3.OP_OWN_EMB) == 1,
      f'per block: the DeltaNet ops x {n_gdn}, the attention ops x {n_att}, two add-and-norms; one more before the head')
names = {v: k for k, v in m.seg_names.items()}
sv_ok = True
for il, g in enumerate(info['kinds']):
    if not g: continue
    so = E.segs[names[f'blk.{il}.ssm_out.weight']]
    sv_ok &= so.dep2p1 > 0 and E.segs[so.dep2p1 - 1].op == k3.OP_GDN_SAVE and E.segs[so.dep].op == k3.OP_GDN_NORM16
check(sv_ok, 'each DeltaNet block\'s output matvec waits for its norm and its save')
check(m.E.guard_commit and len(m.E.guard_caps) == 1 and next(iter(m.E.guard_caps.values()))[1] == 64,
      'the device guard bounds the one position word by the KV rows (64) and the commit')
fused = {il: E.segs[names[f'blk.{il}.ffn_gate.weight']].pad1 & k3.OWN_GU for il in range(len(info['kinds']))
         if info['formats'][il]['ffn_gate'] in k3.OWN_1RT_FMT}
check(all(bool(v) == (info['formats'][il]['ffn_gate'] == info['formats'][il]['ffn_up']) for il, v in fused.items()),
      'gate | up are one segment exactly where the two matrices share a one-round-trip format')

# ---- the host's op list is the device's ---------------------------------------------------------------------------------
src = open(os.path.join(HERE, '..', '..', 'kernels', 'k3_q35_ops.h')).read()
disp = src[src.index('__device__ __forceinline__ bool k3_extra_op('):]
dev_ops = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', disp))
host_ops = {k for k, v in vars(k3).items() if k.startswith('OP_') and isinstance(v, int) and v in k3.Q35S_OPS and k != 'OP_NOP'}
check(dev_ops == host_ops, f'kernels/k3_q35_ops.h dispatches k3.Q35S_OPS ({len(dev_ops)} ops beside the core\'s no-op)')
enum = dict(re.findall(r'(OP_[A-Z0-9_]+) = (\d+)', src))
check({k: int(v) for k, v in enum.items()} == {'OP_OWN_MV16': k3.OP_OWN_MV16, 'OP_GDN_NORM16': k3.OP_GDN_NORM16, 'OP_GATE_MUL16': k3.OP_GATE_MUL16},
      f'the header\'s new op numbers are k3\'s ({enum})')

# ---- options --------------------------------------------------------------------------------------------------------------
m2 = Dense35Own(P, max_ctx=64, vram_check=False, gu=False)
check(not any(k[1].endswith('_gu') for k in m2.E.inventory()) and len(m2.E.segs) > len(E.segs), 'gu=False: no fused segment')
m3 = Dense35Own(P, max_ctx=64, vram_check=False, debug_layers=2)
check(len(m3.L) == 2 and m3.layer_seg[-1][0] == 'head' and len(m3.E.segs) < len(E.segs), 'debug_layers=2: two blocks, then the norm and the head')
p0, p1 = Dense35Own.projection_mib(P, 64), Dense35Own.projection_mib(P, 65)
check(abs((p1 - p0) - Dense35Own.kv_pos_mib(P)) < 1e-9 and abs(Dense35Own.kv_pos_mib(P) * 2 ** 20 - n_att * 4 * tiny_q35.HKV * tiny_q35.HD) < 1e-6
      and abs(m.vram_mib - p0) < 1e-9, f'the projection: {p0:.1f} MiB at 64 positions, + {Dense35Own.kv_pos_mib(P) * 1024:.2f} KiB per position')
n0 = len(E.segs); m.per_wgp = 3; m._build_tape()
check(len(m.E.segs) == n0 and m.E is not E and m.S == tiny_q35.S, 'the tape builds again on the same engine (another grid): the same segments, the shapes untouched')
m.reset(); st = (m.seq, m.fresh, m.pos_host, m.staged)
check(st == (1, True, 0, 0) and sorted({k[2] for k in m.state}) == ['conv', 'convx', 'gdn', 'kv'], 'reset(): a new sequence; the states are the conv windows, the delta states and the KV caches')


# ---- refusals ---------------------------------------------------------------------------------------------------------------
def refused(fn, exc=ValueError):
    try: fn(); return False
    except exc: return True


B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt)
e_lean, e_q35 = k3.Engine(module='k3_q3m_lean'), k3.Engine(module='k3_q35_swift')
w8 = torch.zeros(8 * 8 * 34, dtype=torch.uint8)
check(refused(lambda: e_lean.own_matvec16(w8, 'Q8_0', 8, 256, B(256, torch.float16), B(8))) and
      refused(lambda: e_lean.gdn_norm16(B(256), B(128), B(256), B(256, torch.float16), 2, 128, 1e-6)) and
      refused(lambda: e_lean.gate_mul16(B(256), B(512), 256, 128, B(256, torch.float16), 128)) and
      refused(lambda: e_lean.own_add_rms16(B(5120), B(5120), B(5120), B(5120, torch.float16), 1e-6)),
      'another module refuses the new ops and a 5,120-wide add-and-norm')
check(refused(lambda: e_q35.own_matvec16(w8, 'Q4_K', 8, 256, B(256, torch.float16), B(8))) and
      refused(lambda: e_q35.own_matvec16(w8, 'Q8_0', 8, 256, B(256, torch.float16), B(8), rows_per_task=6)) and
      refused(lambda: e_q35.own_add_rms16(B(5120), B(5120), B(5120), B(5120, torch.float16), 1e-6, d=B(10240), slots=2)) and
      refused(lambda: e_q35.own_add_rms16(B(5120), B(5120), B(5120), B(5120, torch.float16), 1e-6, d=B(5120), slots=1, w=B(1))) and
      refused(lambda: e_q35.kv_append(B(128), B(128), torch.zeros(64 * 2 * 4 * 34, dtype=torch.uint8), torch.zeros(1, dtype=torch.int32), q8=True)),
      'this module refuses a format without a rows tile, an odd task size, a weighted or two-slot add, the q8 KV cache')
e_bad = k3.Engine(module='k3_q35_swift'); e_bad.own_f16(B(256), B(256, torch.float16))
check(refused(lambda: e_bad.build()), 'a tape with an op outside the module\'s dispatch does not build')
print('RESULT', 'PASS' if ok else 'FAIL')
