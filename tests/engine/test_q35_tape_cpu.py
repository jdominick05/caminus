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
torch.cuda.is_available = lambda: False
def no_gpu(*args, **kwargs):
    raise AssertionError('GPU initialization forbidden in the CPU tape suite')
torch.cuda.init = torch.cuda._lazy_init = no_gpu

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
        {('F32_MV', 'F32'), ('OWN_SWIGLU16', '-'), ('OWN_EMB', 'Q4_K'), ('OWN_ADD_RMS16', '-')} | {('ATTN', 'PART'), ('ATTN', 'COMB')})
check(set(inv) == want, f'the tape holds every tile: {sorted(inv.items())}')
check(not E.unsupported() and all(o in k3.Q35S_OPS for o in ops) and E.refuse is None, f'{len(ops)} segments, every op one the module dispatches')
check(all(sg.dep < i and sg.dep2p1 - 1 < i for i, sg in enumerate(E.segs)), 'every dependency points back')
n_gdn, n_att = sum(info['kinds']), len(info['kinds']) - sum(info['kinds'])
cnt = lambda o: ops.count(o)
check((cnt(k3.OP_GDN_CONV), cnt(k3.OP_GDN_L2), cnt(k3.OP_GDN_GATES), cnt(k3.OP_GDN_STEP), cnt(k3.OP_GDN_SAVE), cnt(k3.OP_GDN_NORM16)) == (n_gdn,) * 6
      and (cnt(k3.OP_QK_NR), cnt(k3.OP_KV_APPEND), cnt(k3.OP_ATTN_PART), cnt(k3.OP_ATTN_COMB), cnt(k3.OP_GATE_MUL16)) == (2 * n_att,) + (n_att,) * 4
      and cnt(k3.OP_ATTN_DEC) == 0
      and cnt(k3.OP_OWN_ADD_RMS16) == 2 * len(info['kinds']) + 1 and cnt(k3.OP_OWN_EMB) == 1,
      f'per block: the DeltaNet ops x {n_gdn}, the attention ops x {n_att} (the split pair, no OP_ATTN_DEC), two add-and-norms; one more before the head')
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
TPB = r'#ifdef K3_TP[^\n]*\n(.*?)#endif'   # (the two-card modules' exchange: k3.Q35_TP_MODULES only)
MTPB = r'#ifdef K3_Q35_MTP[^\n]*\n(.*?)#endif'   # (the drafter module's argmax: k3.Q35M_MODULES only)
DFB = r'#ifdef K3_Q35_DF[^\n]*\n(.*?)#endif'   # (the DFlash drafter module's ops: k3.Q35DF_MODULES only)
Q8B = r'#ifdef K3_Q35_KV_Q8[^\n]*\n.*?#else[^\n]*\n(.*?)#endif'   # retain the F16 dispatch in this suite
SPB = r'#ifdef K3_Q35_ATTN_SPLIT[^\n]*\n(.*?)#else[^\n]*\n(.*?)#endif'   # (decode attention: the split pair; #else the DFlash module's OP_ATTN_DEC)
disp = src[src.index('__device__ __forceinline__ bool k3_extra_op('):]
disp = re.sub(Q8B, lambda mm: mm.group(1), disp, flags=re.S)
tp_ops = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', ''.join(re.findall(TPB, disp, re.S))))
mtp_ops = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', ''.join(re.findall(MTPB, disp, re.S))))
df_ops = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', ''.join(re.findall(DFB, disp, re.S))))
disp = re.sub(DFB, '', disp, flags=re.S)
sp_ops = [tuple(set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', g)) for g in m) for m in re.findall(SPB, disp, re.S)]
dev_ops = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', re.sub(SPB, lambda mm: mm.group(1), re.sub(MTPB, '', re.sub(TPB, '', disp, flags=re.S), flags=re.S), flags=re.S)))
host_ops = {k for k, v in vars(k3).items() if k.startswith('OP_') and isinstance(v, int) and v in k3.Q35S_OPS and k != 'OP_NOP'}
check(dev_ops == host_ops and tp_ops == {'OP_XCHG'}, f'kernels/k3_q35_ops.h dispatches k3.Q35S_OPS ({len(dev_ops)} ops beside the core\'s no-op), and OP_XCHG under K3_TP')
df_host = {k for k, v in vars(k3).items() if k.startswith('OP_') and isinstance(v, int) and v in k3.Q35DF_OPS} - {'OP_NOP'}
check(df_ops == {'OP_DF_CONV', 'OP_DF_TOPK', 'OP_DF_SELECT', 'OP_KV_APPEND', 'OP_ATTN_DEC'} and df_ops <= df_host,
      f'under K3_Q35_DF: the DFlash drafter\'s own ops (conv / top-k / select, the ring append, its attention), all in k3.Q35DF_OPS')
check(sp_ops == [({'OP_ATTN_PART', 'OP_ATTN_COMB'}, {'OP_ATTN_DEC'})] and k3.OP_ATTN_DEC not in k3.Q35M_OPS
      and {k3.OP_ATTN_PART, k3.OP_ATTN_COMB} <= k3.Q35M_OPS and k3.OP_ATTN_DEC not in k3.Q35S_OPS and k3.OP_ATTN_DEC in k3.Q35DF_OPS,
      'decode attention: the split pair under K3_Q35_ATTN_SPLIT (k3.Q35S_OPS, the MTP drafter\'s k3.Q35M_OPS too), OP_ATTN_DEC only in its #else (the DFlash module, k3.Q35DF_OPS)')
swift_src = open(os.path.join(HERE, '..', '..', 'kernels', 'k3_q35_swift.hip')).read()
guard = re.search(r'#if ([^\n]*)\n#define K3_Q35_ATTN_SPLIT', swift_src)
check(guard is not None and guard.group(1).strip() == '!defined(K3_Q35_DF)',
      f'kernels/k3_q35_swift.hip compiles the split pair into every module but the DFlash one (guard: {guard.group(1).strip() if guard else None}): '
      'the MTP module (K3_Q35_MTP) has it, as k3.Q35M_OPS says')
enum = dict(re.findall(r'(OP_[A-Z0-9_]+) = (\d+)', src))
check({k: int(v) for k, v in enum.items()} == {'OP_OWN_MV16': k3.OP_OWN_MV16, 'OP_GDN_NORM16': k3.OP_GDN_NORM16, 'OP_GATE_MUL16': k3.OP_GATE_MUL16,
                                               'OP_OWN_ARGMAX': k3.OP_OWN_ARGMAX, 'OP_ATTN_PART': k3.OP_ATTN_PART, 'OP_ATTN_COMB': k3.OP_ATTN_COMB},
      f'the header\'s new op numbers are k3\'s ({enum})')

# ---- options --------------------------------------------------------------------------------------------------------------
m2 = Dense35Own(P, max_ctx=64, vram_check=False, gu=False)
check(not any(k[1].endswith('_gu') for k in m2.E.inventory()) and len(m2.E.segs) > len(E.segs), 'gu=False: no fused segment')
m3 = Dense35Own(P, max_ctx=64, vram_check=False, debug_layers=2)
check(len(m3.L) == 2 and m3.layer_seg[-1][0] == 'head' and len(m3.E.segs) < len(E.segs), 'debug_layers=2: two blocks, then the norm and the head')
p0, p1 = Dense35Own.projection_mib(P, 64), Dense35Own.projection_mib(P, 65)
rec = 4 * (q35_dense.attn_part_elems(1, tiny_q35.H, 65) - q35_dense.attn_part_elems(1, tiny_q35.H, 64)) / 2 ** 20   # (65 rows: one more split's records)
check(rec > 0 and abs((p1 - p0) - Dense35Own.kv_pos_mib(P) - rec) < 1e-9 and abs(Dense35Own.kv_pos_mib(P) * 2 ** 20 - n_att * 4 * tiny_q35.HKV * tiny_q35.HD) < 1e-6
      and abs(m.vram_mib - p0) < 1e-9, f'the projection: {p0:.1f} MiB at 64 positions, + {Dense35Own.kv_pos_mib(P) * 1024:.2f} KiB per position + the split records per 32 rows')
n0 = len(E.segs); m.per_wgp = 3; m._build_tape()
check(len(m.E.segs) == n0 and m.E is not E and m.S == tiny_q35.S, 'the tape builds again on the same engine (another grid): the same segments, the shapes untouched')
m.reset(); st = (m.seq, m.fresh, m.pos_host, m.staged)
check(st == (1, True, 0, 0) and sorted({k[2] for k in m.state}) == ['conv', 'convx', 'gdn', 'kv'], 'reset(): a new sequence; the states are the conv windows, the delta states and the KV caches')


# ---- refusals ---------------------------------------------------------------------------------------------------------------
def refused(fn, exc=ValueError):
    try: fn(); return False
    except exc: return True


B = lambda n, dt=torch.float32: torch.zeros(n, dtype=dt)
# ---- the drafter module (k3_q35_swift_mtp: engine/q35_mtp.py) ---------------------------------------------------------
mtp_host = {k for k, v in vars(k3).items() if k.startswith('OP_') and isinstance(v, int) and v in k3.Q35M_OPS and k != 'OP_NOP'}
check(mtp_ops == {'OP_OWN_ARGMAX'} and mtp_host <= dev_ops | mtp_ops | sp_ops[0][1] and int(re.search(r'OWN_Q4_0 = (\d+)', open(os.path.join(HERE, '..', '..', 'kernels', 'k3_own_ops.h')).read()).group(1)) == k3.OWN_MTP_FMT['Q4_0'],
      f'the drafter module: OP_OWN_ARGMAX under K3_Q35_MTP, k3.Q35M_OPS ({len(mtp_host)} ops) all dispatched, the Q4_0 format number k3\'s')
e_m = k3.Engine(module='k3_q35_swift_mtp'); e_s = k3.Engine(module='k3_q35_swift')
q4 = lambda r, kk: torch.zeros(r * kk // 256 * 144, dtype=torch.uint8)
check(refused(lambda: e_m.own_matvec_1rt(torch.zeros(8 * 144, dtype=torch.uint8), 'Q4_K', 8, 256, B(256, torch.float16), B(8))) and
      refused(lambda: e_s.own_matvec_1rt(q4(8, 256), 'Q4_0', 8, 256, B(256, torch.float16), B(8))) and
      refused(lambda: e_s.own_argmax(B(300), 300, torch.zeros(1, dtype=torch.int64))) and
      refused(lambda: e_s.own_emb(torch.zeros(64 * 144, dtype=torch.uint8), 'Q4_K', 64, 256, torch.zeros(1, dtype=torch.int64), B(256), key=True)) and
      refused(lambda: e_s.own_add_rms16(B(256), B(256), B(256), B(512, torch.float16), 1e-6, h16_stride=512)),
      'the drafter module refuses Q4_K; the Swift module refuses Q4_0, the argmax, the key embedding and the h16 stride')
e_m.own_matvec_1rt(q4(8, 256), 'Q4_0', 8, 256, B(256, torch.float16), B(8)); e_m.own_argmax(B(300), 300, torch.zeros(1, dtype=torch.int64))
e_m.build()
check(set(e_m.inventory()) == {('OWN_MV', 'Q4_0/a16_1rt'), ('OWN_ARGMAX', '-')} and not e_m.unsupported(), 'the drafter module builds a Q4_0 matvec and the argmax')
e_g = k3.Engine(module='k3_q35_swift_mtp'); e_g.gdn_l2(B(512), B(512), 2, 128, 1e-6)
check(refused(lambda: e_g.build()), 'the drafter module refuses an op of the trunk its tape never runs (a Gated DeltaNet op)')
check(k3.key_token(((0x80000000 | 1234) << 32) | (0xffffffff ^ 4321)) == 4321, 'key_token: the low half\'s complement')
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

# ---- the ThinkingCap module (k3_q35_tcap: kernels/k3_q35t_ops.h) ----------------------------------------------------------
PT = os.path.join(A.scratch, 'tiny-q35-tcap.gguf'); tiny_q35.write(PT, variant='tcap')
lt = ref_q35.Q35Ref(PT).feed([1, 7, 3, 60])
check(lt.shape == (4, info['V']) and np.isfinite(lt).all() and np.abs(lt).max() > 1e-3,
      f'the reference reads the tiny ThinkingCap-format model (IQ2_S embedding, BF16 alpha / beta; |logit| up to {np.abs(lt).max():.2f})')
mt = Dense35Own(PT, max_ctx=64, vram_check=False); Et = mt.E; it = Et.inventory()
check(mt.module == 'k3_q35_tcap' and m.module == 'k3_q35_swift', f'the module comes from the file\'s formats ({mt.module}; Swift\'s: {m.module})')
want_t = ({('OWN_MV', f'{f}/a16_1rt') for f in k3.OWN_Q35T_1RT} | {('OWN_MV16', 'Q8_0'), ('F32_MV', 'F32'), ('OWN_SWIGLU16', '-'),
          ('OWN_EMB', 'IQ2_S'), ('OWN_ADD_RMS16', '-')} | {('ATTN', 'PART'), ('ATTN', 'COMB')})
check(set(it) == want_t, f'the ThinkingCap tape holds every tile, no fused gate|up, BF16 alpha / beta as F32: {sorted(it.items())}')
check(not Et.unsupported() and all(sg.op in k3.Q35S_OPS for sg in Et.segs) and Et.refuse is None,
      f'{len(Et.segs)} segments, every op one the ThinkingCap module dispatches')
check(Et.segs[0].op == k3.OP_OWN_EMB and Et.segs[0].out_f32 == 1 | 2 | 8,
      f'the embedding carries the grids to stage: IQ3_XXS | IQ3_S | IQ2_S = 11 (got {Et.segs[0].out_f32})')
check(abs(Dense35Own.projection_mib(PT, 64) - mt.vram_mib) < 1e-9, 'the projection counts the widened BF16 and the record tails as the engine')
srct = open(os.path.join(HERE, '..', '..', 'kernels', 'k3_q35t_ops.h')).read()
dt = srct.split('k3_extra_op(')[1]
dt = re.sub(Q8B, lambda mm: mm.group(1), dt, flags=re.S)
dft_ops = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', ''.join(re.findall(DFB, dt, re.S))))
dt = re.sub(DFB, '', dt, flags=re.S)   # (ThinkingCap's DFlash drafter module's ops: k3.Q35TDF_MODULES only)
mtt_ops = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', ''.join(re.findall(MTPB, dt, re.S))))
dt = re.sub(MTPB, '', dt, flags=re.S)   # (ThinkingCap's MTP drafter module's argmax: k3.Q35TM_MODULES only)
dev_t = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', re.sub(SPB, lambda mm: mm.group(1), re.sub(TPB, '', dt, flags=re.S), flags=re.S)))
check(dev_t == host_ops and set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', ''.join(re.findall(TPB, dt, re.S)))) == {'OP_XCHG'},
      f'kernels/k3_q35t_ops.h dispatches k3.Q35S_OPS too ({len(dev_t)} ops), and OP_XCHG under K3_TP')
check(mtt_ops == {'OP_OWN_ARGMAX'}, 'under K3_Q35_MTP in k3_q35t_ops.h: the argmax (ThinkingCap\'s MTP drafter module)')
check(dft_ops == {'OP_DF_CONV', 'OP_DF_TOPK', 'OP_DF_SELECT', 'OP_KV_APPEND', 'OP_ATTN_DEC'}, 'under K3_Q35_DF in k3_q35t_ops.h: the DFlash drafter\'s own ops (ThinkingCap\'s drafter module)')
check(int(re.search(r'OWN_IQ2_S = (\d+)', srct).group(1)) == k3.OWN_Q35T_FMT['IQ2_S'], 'the IQ2_S format number is k3\'s')
e_tc = k3.Engine(module='k3_q35_tcap')
check(refused(lambda: e_tc.own_matvec_1rt(torch.zeros(8 * 144 + 16, dtype=torch.uint8), 'Q4_K', 8, 256, B(256, torch.float16), B(8))) and
      refused(lambda: e_tc.own_matvec_1rt(torch.zeros(16 * 176, dtype=torch.uint8), 'Q5_K', 8, 256, B(256, torch.float16), B(8, torch.float16),
                                          up=torch.zeros(16 * 176, dtype=torch.uint8))) and
      refused(lambda: e_q35.own_matvec_1rt(torch.zeros(8 * 82 + 16, dtype=torch.uint8), 'IQ2_S', 8, 256, B(256, torch.float16), B(8))) and
      refused(lambda: e_q35.own_emb(torch.zeros(64 * 82, dtype=torch.uint8), 'IQ2_S', 64, 256, torch.zeros(1, dtype=torch.int32), B(256))),
      'the ThinkingCap module refuses Q4_K and a fused gate|up; the Swift module refuses IQ2_S matvecs and embeddings')

# ---- the verify tape (engine/q35_dense_verify.py, module k3_q35_swift_ver) -----------------------------------------------
from caminus.engine.q35_dense_verify import Dense35Verify
mv4 = Dense35Own(P, max_ctx=64, vram_check=False, spec_T=4)
v4 = Dense35Verify(mv4, 4, vram_check=False); Ev = v4.E; iv = Ev.inventory(); ov = [sg.op for sg in Ev.segs]
check(set(iv) == {('OWN_MV', f'{f}/a16_1rt_mt') for f in ('Q4_K', 'Q6_K')} | {('OWN_MV16', 'Q8_0'), ('OWN_MV16', 'Q5_K')} |
      {('F32_MV', 'F32'), ('OWN_SWIGLU16', '-'), ('OWN_EMB', 'Q4_K'), ('OWN_ADD_RMS16', '-')} | {('ATTN', 'PART'), ('ATTN', 'COMB')},
      f'the verify tape: every matvec in its T-token form, no fused gate | up, the split attention pair ({sorted(iv)})')
check(not Ev.unsupported() and Ev.refuse is None and all(o in k3.Q35S_OPS for o in ov) and all(sg.dep < i and sg.dep2p1 - 1 < i for i, sg in enumerate(Ev.segs)),
      f'{len(ov)} segments, every op one the verify module dispatches, every dependency points back')
m16 = [sg for sg in Ev.segs if sg.op == k3.OP_OWN_MV16]
check(m16 and all(sg.pad1 >> 8 == 4 and sg.gs == sg.K | (sg.N << 32) for sg in m16) and
      all(sg.team == 4 for sg in Ev.segs) and all(sg.ntask == 4 for sg in Ev.segs if sg.op in (k3.OP_OWN_ADD_RMS16, k3.OP_OWN_EMB)),
      'the rows tiles carry 4 tokens and their strides; the glue ops one task per token; every segment the pass\'s T')
blk = mv4.L[[i for i, g in enumerate(info['kinds']) if g][0]]
check(blk['gdn'].numel() == k3.gdn_block_size(tiny_q35.NV, tiny_q35.S, tiny_q35.NK, 4) and blk['convx'].numel() == 4 * blk['conv_st'].shape[0]
      and mv4.E.guard_T == 4 and m.E.guard_T is None,
      'spec_T 4: each DeltaNet block keeps 4 tokens\' inputs; the decode tape\'s guard takes commits up to 4 (spec_T 1: the tape\'s own T)')
q1, q4 = Dense35Own.projection_mib(P, 64), Dense35Own.projection_mib(P, 64, spec_T=4)
check(q4 > q1 and abs(Dense35Own.projection_mib(P, 64, spec_T=4) - mv4.vram_mib) < 1e-9
      and Dense35Own.recurrent_mib(P, 4) > Dense35Own.recurrent_mib(P), f'the projection counts spec_T\'s records and rows ({q1:.3f} -> {q4:.3f} MiB)')
check(refused(lambda: Dense35Verify(m, 2, vram_check=False)) and refused(lambda: Dense35Verify(mv4, 5, vram_check=False)) and
      refused(lambda: Dense35Verify(mv4, 4, module='k3_q35_swift', vram_check=False)) and refused(lambda: Dense35Own(P, max_ctx=64, vram_check=False, spec_T=5)),
      'refused: a verify tape longer than the engine\'s spec_T, T 5, a decode module, spec_T 5')
ok_c = mv4.commit_max() == 0 and refused(lambda: v4.run([1, 2, 3, 4], 0, 1), k3.ContextBoundsError)
mv4.fresh, mv4.last_T = False, 1
ok_c &= mv4.commit_max() == 1 and refused(lambda: v4.run([1, 2, 3, 4], 1, 2), k3.ContextBoundsError)
ok_c &= refused(lambda: mv4.verify([1] * 8, 1, 2), k3.ContextBoundsError)
mv4.last_T = 4
ok_c &= mv4.commit_max() == 4 and m.commit_max() == 1 and Ev.guard_T == 4
mv4.reset(); ok_c &= mv4.commit_max() == 0
check(ok_c, 'the commit is bounded by the last pass on the states (0 after a reset, 1 after a decode step, T after a verify '
            'pass: refused on the host past it); the served engine keeps 1; the verify tape\'s guard takes spec_T')
# ThinkingCap's verify tape (k3_q35_tcap_ver: kernels/own_quant_tcap_mt.h's T-token tiles)
mt4 = Dense35Own(PT, max_ctx=64, vram_check=False, spec_T=4)
vt = Dense35Verify(mt4, 4, vram_check=False); Evt = vt.E; ivt = Evt.inventory()
check(vt.module == 'k3_q35_tcap_ver' and set(ivt) == {('OWN_MV', f'{f}/a16_1rt_mt') for f in k3.OWN_Q35T_1RT} |
      {('OWN_MV16', 'Q8_0'), ('F32_MV', 'F32'), ('OWN_SWIGLU16', '-'), ('OWN_EMB', 'IQ2_S'), ('OWN_ADD_RMS16', '-')} | {('ATTN', 'PART'), ('ATTN', 'COMB')},
      f'ThinkingCap\'s verify tape on k3_q35_tcap_ver: every format\'s matvec in its T-token form ({sorted(ivt)})')
check(not Evt.unsupported() and Evt.refuse is None and Evt.segs[0].op == k3.OP_OWN_EMB and Evt.segs[0].out_f32 == 1 | 2 | 8
      and all(sg.team == 4 for sg in Evt.segs) and all(sg.pad1 >> 8 == 4 for sg in Evt.segs if sg.op == k3.OP_OWN_MV),
      f'{len(Evt.segs)} segments; the embedding stages the grids (11); every matvec carries 4 tokens')
check(refused(lambda: Dense35Verify(mt4, 4, module='k3_q35_swift_ver', vram_check=False)) and
      refused(lambda: Dense35Verify(mv4, 4, module='k3_q35_tcap_ver', vram_check=False)) and
      refused(lambda: k3.Engine(module='k3_q35_tcap', T=1).own_matvec_1rt(B(4096, torch.uint8), 'IQ4_XS', 2, 256, B(4 * 256, torch.float16), B(8), tokens=4, x_stride=256, out_stride=2)),
      'refused: a verify module of the other line, and T-token matvecs on ThinkingCap\'s decode module')
check(not any(c[0] == 'OWN_MV' for c in k3.MODULE_CAPS['k3_q35_tcap_dflash']),
      'the ThinkingCap DFlash module lists no OWN_MV pair (its device code compiles OP_OWN_MV and the IQ grids out: kernels/k3_q35t_ops.h K3_Q35_DF)')
# ThinkingCap's MTP drafter (k3_q35_tcap_mtp: engine/q35_mtp.py on its Q6_K block, Q8_0 head, IQ2_S embedding)
from caminus.engine.q35_mtp import Mtp35
PM = os.path.join(A.scratch, 'tiny-q35-tcap-mtp.gguf'); tiny_q35.write(PM, variant='tcap', mtp=True)
mm4 = Dense35Own(PM, max_ctx=64, vram_check=False, spec_T=4)
mt_ = Mtp35(mm4, k=3, vram_check=False)
ivm = set().union(*(set(E_.inventory()) for E_ in mt_.E.values()))
chain = mt_.E[('chain', 1)]
check(mt_.module == 'k3_q35_tcap_mtp' and ivm <= k3.MODULE_CAPS['k3_q35_tcap_mtp'] and ('OWN_MV', 'Q6_K/a16_1rt_mt') in ivm
      and ('OWN_MV16', 'Q8_0') in ivm and ('OWN_EMB', 'IQ2_S') in ivm and all(not E_.unsupported() for E_ in mt_.E.values())
      and chain.segs[0].op == k3.OP_OWN_EMB and chain.segs[0].pad1 == 1 and any(sg.op == k3.OP_OWN_ARGMAX for sg in chain.segs),
      f'ThinkingCap\'s MTP tapes on k3_q35_tcap_mtp: the block\'s Q6_K T-token tiles, the Q8_0 head, the IQ2_S embedding from '
      f'the argmax key on a chained step ({len(mt_.E)} tapes; {sorted(ivm)})')
check(refused(lambda: Mtp35(mm4, module='k3_q35_swift_mtp', vram_check=False)) and
      refused(lambda: k3.Engine(module='k3_q35_tcap_mtp', T=4).own_matvec_1rt(B(4096, torch.uint8), 'IQ4_XS', 2, 256, B(4 * 256, torch.float16), B(8), tokens=4, x_stride=256, out_stride=2)),
      'refused: the other line\'s drafter module, and a format other than Q6_K on ThinkingCap\'s MTP module')
check(('OWN_MV', 'Q4_K/a16_1rt_gu') not in k3.MODULE_CAPS['k3_q35_swift_ver'] and ('OWN_MV', 'Q4_K/a16_1rt_gu') in k3.MODULE_CAPS['k3_q35_swift'],
      'the verify module\'s capabilities leave out the fused gate | up (no verify tape runs it)')
e_v = k3.Engine(module='k3_q35_swift_ver', T=4)
check(refused(lambda: e_q35.own_matvec16(w8, 'Q8_0', 8, 256, B(4 * 256, torch.float16), B(32), tokens=4)) and
      refused(lambda: e_v.own_matvec16(w8, 'Q8_0', 8, 256, B(4 * 256, torch.float16), B(32), tokens=2)) and
      refused(lambda: e_v.own_add_rms16(B(4 * 256), B(256), B(4 * 256), B(4 * 256, torch.float16), 1e-6, tokens=2)) and
      refused(lambda: e_q35.gdn_norm16(B(256), B(128), B(256), B(256, torch.float16), 2, 128, 1e-6) or k3.Engine(module='k3_q35_swift', T=2).gdn_norm16(B(512), B(128), B(512), B(512, torch.float16), 2, 128, 1e-6)),
      'the T-token forms: refused on the decode module and for a token count other than the tape\'s T')

# ---- staged Q8 target: allocation, tape order, verify and terminal failures -------------------------------------------
from caminus.cache.prompt import Layout, engine_key, key_hash
from caminus.prefill.q35own import Q35OwnPrefill
from caminus.prefill.q35own_tp import Q35OwnPrefillTP
from caminus.errors import InvalidKVEncoding
qm = Dense35Own(P, max_ctx=64, spec_T=4, q8=True, vram_check=False)
qt = Dense35Own(PT, max_ctx=64, spec_T=4, q8=True, vram_check=False)
for target in (qm, qt):
    kv = next(v for k, v in target.state.items() if k[-1] == 'kv')
    pb = k3.kv_pos_bytes(target.nkv * target.hd, True)
    check(kv.dtype == torch.uint8 and kv.numel() == target.kv_rows * pb and target.kv_rows == 67,
          f'{target.module}: exact byte cache allocation includes verify tail')
    check(abs(Dense35Own.projection_mib(target.path, 64, 4, q8=True) - target.vram_mib) < 1e-9,
          f'{target.module}: Q8 allocation and projection agree')
    ly = Layout(target, T=4)
    check(ly.pb == [pb], f'{target.module}: checkpoint row stride uses physical state-view rows')
    check(sum(sg.op == k3.OP_HADAMARD for sg in target.E.segs) == 4 * n_att,
          f'{target.module}: Q/K256, V/output64 transforms on every attention layer')
    app = next(i for i, sg in enumerate(target.E.segs) if sg.op == k3.OP_KV_APPEND)
    gates = next(i for i, sg in enumerate(target.E.segs) if sg.op == k3.OP_GATE_MUL16)
    sg = target.E.segs[app]
    check(target.E.segs[sg.dep].op == k3.OP_HADAMARD and target.E.segs[sg.dep2p1 - 1].op == k3.OP_HADAMARD
          and target.E.segs[target.E.segs[gates].dep].op == k3.OP_HADAMARD,
          f'{target.module}: append waits for both rotations, output rotation precedes gating')
    vv = Dense35Verify(target, 4, vram_check=False)
    check(vv.module == k3.Q35_KV_Q8_OF['k3_q35_tcap_ver' if target.tcap else 'k3_q35_swift_ver']
          and vv.module not in k3.ATTN_PACK_MODULES and not vv.E.unsupported(),
          f'{target.module}: matching unpacked verify tape constructs for T4')
    check(sum(sg.op == k3.OP_HADAMARD for sg in vv.E.segs) == 4 * n_att,
          f'{target.module}: verify carries every transform')
    check(refused(lambda: Dense35Verify(target, 4, module='k3_q35_tcap_ver' if target.tcap else 'k3_q35_swift_ver', vram_check=False)),
          f'{target.module}: F16 verify cannot reinterpret the Q8 state')
check(refused(lambda: Dense35Own(P, module='k3_q35_swift_kvq8', vram_check=False))
      and refused(lambda: q35_dense.kv_module(None, 'k3_q35_swift', q8='q8')),
      'Q8-only module and strict cache-format flag fail closed before allocation')
delta = n_att * 67 * (k3.kv_pos_bytes(qm.nkv * qm.hd, False) - k3.kv_pos_bytes(qm.nkv * qm.hd, True)) - 4
check(abs((Dense35Own.projection_mib(P, 64, 4) - qm.vram_mib) * 2 ** 20 - delta) < 1e-6,
      'Q8 memory reduction is exact row packing minus the new four-byte error word')
check(refused(lambda: Q35OwnPrefill(types.SimpleNamespace(q8=True)))
      and refused(lambda: Q35OwnPrefillTP(types.SimpleNamespace(q8=True))),
      'one- and two-card Q8 prefill require supported geometry before module load')

pf_key = types.SimpleNamespace(numerics=lambda: {'attention': 'synthetic'}, crossover=None)
f16_key = engine_key(mv4, pf_key, build={'synthetic': True})
q8_key = engine_key(qm, pf_key, build={'synthetic': True})
check('target_kv' not in f16_key['numerics'] and q8_key['numerics']['target_kv'] == qm.kv_format_id
      and key_hash(f16_key) != key_hash(q8_key), 'checkpoint key identifies Q8 codec and both rotations; F16 key stays unchanged')
for rows in (True, 66.5, 63, 69):
    qm.kv_rows = rows
    check(refused(lambda: Layout(qm, T=4)), f'checkpoint refuses invalid physical KV row count {rows!r}')
qm.kv_rows = 67

# The actual verify pass runs with launch/event CPU stand-ins. A simultaneous
# numerical error and timeout must leave no opportunity to retry or clear it.
qv = Dense35Verify(qt, 4, vram_check=False)
qv.E.run = lambda **kwargs: None
qv.ev_done = types.SimpleNamespace(record=lambda: None,
    synchronize=lambda: (qv.E.kv_error.fill_(2), qv.h_st[1].fill_(1)))
verify_passes = []
original_verify_pass = qv._pass
def fail_verify_pass(*args):
    verify_passes.append(args)
    return original_verify_pass(*args)
qv._pass = fail_verify_pass
try:
    qv.run([1, 2, 3, 4], 0, 0, argmax=False)
    verify_failed = False
except InvalidKVEncoding:
    verify_failed = True
check(verify_failed and len(verify_passes) == 1 and not qv.bails and not qt.bails
      and qt.broken and int(qv.E.kv_error[0]) == 2,
      'verify waits, reports the refused encoding before timeout recovery and marks its target terminal')
check(refused(lambda: qt.step(1), InvalidKVEncoding)
      and refused(lambda: qv.run([1, 2, 3, 4], 0, 0), InvalidKVEncoding),
      'verify failure prevents both a target decode and another verify pass')

# Error becomes visible only at the event wait; both it and a timeout are set.
# The numerical error must throw once, survive reset and never enter the retry.
qm.ev_done = types.SimpleNamespace(synchronize=lambda: (qm.E.kv_error.fill_(2), qm.h_st[1].fill_(1)))
calls = []
def fail_pass(*args):
    calls.append(args)
    return qm._pass_wait(0)
qm._pass = fail_pass
try:
    qm._run_tok(1, 0, 0, False)
    failed = False
except InvalidKVEncoding:
    failed = True
check(failed and len(calls) == 1 and not qm.bails and int(qm.E.kv_error[0]) == 2 and qm.broken,
      'numerical cache refusal wins over timeout without clearing or retrying, marks engine terminal')
qm.reset()
try:
    qm.step(1)
    refused_after_reset = False
except InvalidKVEncoding:
    refused_after_reset = True
check(refused_after_reset, 'reset cannot make a refused Q8 cache reusable')

print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
