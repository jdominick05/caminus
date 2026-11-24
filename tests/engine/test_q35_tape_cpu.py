"""The dense qwen35 own-kernel engine's host side without a card (caminus.engine.q35_dense, module dec_q35_swift): the tape
it builds for a tiny synthetic quantized model (tests/engine/tiny_q35.py), on stand-ins for the torch.cuda calls (every
tensor stays in host memory; nothing is launched).
  - the model file is one the float64 reference reads (finite logits), and the engine takes the reference's block kinds
  - the tape holds the tile each format needs (one-round-trip Q4_K / Q6_K, their fused gate | up, the rows tiles for
    Q8_0 / Q5_K, the F32 matvec) and only ops the module dispatches; every dependency points back; each DeltaNet
    block's output matvec waits for the block's save
  - the host's op list is the device's: the op names kernels/ops_q35.h dispatches are k3.Q35S_OPS, and its three new
    op numbers are k3's
  - the options: gu=False has no fused segment, debug_layers builds a prefix, the projection grows by the KV row
  - refusals: the new ops on another module, a long add-and-norm row with weights, the q8 KV cache on this module
  - OP_PREFETCH (the qwen35moe modules): one segment over its ranges, 128-byte lines from each rounded-down base
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/engine/test_q35_tape_cpu.py --scratch DIR"""
import argparse, contextlib, ctypes, os, re, sys, types
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
from caminus.device import kernels
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
src = open(os.path.join(HERE, '..', '..', 'kernels', 'ops_q35.h')).read()
TPB = r'#ifdef K3_TP[^\n]*\n(.*?)#endif'   # (the two-card modules' exchange: k3.Q35_TP_MODULES only)
MTPB = r'#ifdef K3_Q35_MTP[^\n]*\n(.*?)#endif'   # (the drafter module's argmax: k3.Q35M_MODULES only)
DFB = r'#ifdef K3_Q35_DF[^\n]*\n(.*?)#endif'   # (the DFlash drafter module's ops: k3.Q35DF_MODULES only)
MOEB = r'#ifdef K3_Q35_MOE[^\n]*\n(.*?)#else[^\n]*\n(.*?)#endif'   # (the qwen35moe module's extra ops; #else the dense dispatch)
Q8B = r'#ifdef K3_Q35_KV_Q8[^\n]*\n.*?#else[^\n]*\n(.*?)#endif'   # retain the F16 dispatch in this suite
SPB = r'#ifdef K3_Q35_ATTN_SPLIT[^\n]*\n(.*?)#else[^\n]*\n(.*?)#endif'   # (decode attention: the split pair; #else the DFlash module's OP_ATTN_DEC)
disp = src[src.index('__device__ __forceinline__ bool k3_extra_op('):]
disp = re.sub(Q8B, lambda mm: mm.group(1), disp, flags=re.S)
tp_ops = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', ''.join(re.findall(TPB, disp, re.S))))
mtp_ops = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', ''.join(re.findall(MTPB, disp, re.S))))
df_ops = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', ''.join(re.findall(DFB, disp, re.S))))
moe_m = re.findall(MOEB, disp, re.S)
moe_ops = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', moe_m[0][0])) if moe_m else set()
disp = re.sub(MOEB, lambda mm: mm.group(2), disp, flags=re.S)   # (this suite's tape is the dense line's: keep the #else)
disp = re.sub(DFB, '', disp, flags=re.S)
sp_ops = [tuple(set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', g)) for g in m) for m in re.findall(SPB, disp, re.S)]
dev_ops = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', re.sub(SPB, lambda mm: mm.group(1), re.sub(MTPB, '', re.sub(TPB, '', disp, flags=re.S), flags=re.S), flags=re.S)))
host_ops = {k for k, v in vars(k3).items() if k.startswith('OP_') and isinstance(v, int) and v in k3.Q35S_OPS and k != 'OP_NOP'}
check(dev_ops == host_ops and tp_ops == {'OP_XCHG'}, f'kernels/ops_q35.h dispatches k3.Q35S_OPS ({len(dev_ops)} ops beside the core\'s no-op), and OP_XCHG under K3_TP')
moe_host = {k for k, v in vars(k3).items() if k.startswith('OP_') and isinstance(v, int) and v in k3.Q35MOE_OPS} - {'OP_NOP'}
check(moe_ops <= moe_host and (moe_ops - host_ops) == (moe_host - host_ops) == {'OP_ROUTER', 'OP_PREFETCH'},
      f'under K3_Q35_MOE: the qwen35moe module\'s extra ops ({sorted(moe_ops - host_ops)}), all in k3.Q35MOE_OPS and none in k3.Q35S_OPS')
df_host = {k for k, v in vars(k3).items() if k.startswith('OP_') and isinstance(v, int) and v in k3.Q35DF_OPS} - {'OP_NOP'}
check(df_ops == {'OP_DF_CONV', 'OP_DF_TOPK', 'OP_DF_SELECT', 'OP_KV_APPEND', 'OP_ATTN_DEC'} and df_ops <= df_host,
      f'under K3_Q35_DF: the DFlash drafter\'s own ops (conv / top-k / select, the ring append, its attention), all in k3.Q35DF_OPS')
check(sp_ops == [({'OP_ATTN_PART', 'OP_ATTN_COMB'}, {'OP_ATTN_DEC'})] and k3.OP_ATTN_DEC not in k3.Q35M_OPS
      and {k3.OP_ATTN_PART, k3.OP_ATTN_COMB} <= k3.Q35M_OPS and k3.OP_ATTN_DEC not in k3.Q35S_OPS and k3.OP_ATTN_DEC in k3.Q35DF_OPS,
      'decode attention: the split pair under K3_Q35_ATTN_SPLIT (k3.Q35S_OPS, the MTP drafter\'s k3.Q35M_OPS too), OP_ATTN_DEC only in its #else (the DFlash module, k3.Q35DF_OPS)')
swift_src = open(os.path.join(HERE, '..', '..', 'kernels', 'dec_q35_swift.hip')).read()
guard = re.search(r'#if ([^\n]*)\n#define K3_Q35_ATTN_SPLIT', swift_src)
check(guard is not None and guard.group(1).strip() == '!defined(K3_Q35_DF)',
      f'kernels/dec_q35_swift.hip compiles the split pair into every module but the DFlash one (guard: {guard.group(1).strip() if guard else None}): '
      'the MTP module (K3_Q35_MTP) has it, as k3.Q35M_OPS says')
enum = dict(re.findall(r'(OP_[A-Z0-9_]+) = (\d+)', src))
check({k: int(v) for k, v in enum.items()} == {'OP_OWN_MV16': k3.OP_OWN_MV16, 'OP_GDN_NORM16': k3.OP_GDN_NORM16, 'OP_GATE_MUL16': k3.OP_GATE_MUL16,
                                               'OP_OWN_ARGMAX': k3.OP_OWN_ARGMAX, 'OP_ATTN_PART': k3.OP_ATTN_PART, 'OP_ATTN_COMB': k3.OP_ATTN_COMB,
                                               'OP_PREFETCH': k3.OP_PREFETCH},
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
# ---- the drafter module (dec_q35_swift_mtp: engine/q35_mtp.py) ---------------------------------------------------------
mtp_host = {k for k, v in vars(k3).items() if k.startswith('OP_') and isinstance(v, int) and v in k3.Q35M_OPS and k != 'OP_NOP'}
check(mtp_ops == {'OP_OWN_ARGMAX'} and mtp_host <= dev_ops | mtp_ops | sp_ops[0][1] and int(re.search(r'OWN_Q4_0 = (\d+)', open(os.path.join(HERE, '..', '..', 'kernels', 'ops_own.h')).read()).group(1)) == k3.OWN_MTP_FMT['Q4_0'],
      f'the drafter module: OP_OWN_ARGMAX under K3_Q35_MTP, k3.Q35M_OPS ({len(mtp_host)} ops) all dispatched, the Q4_0 format number k3\'s')
e_m = k3.Engine(module='dec_q35_swift_mtp'); e_s = k3.Engine(module='dec_q35_swift')
q4 = lambda r, kk: torch.zeros(r * kk // 256 * 144, dtype=torch.uint8)
check(refused(lambda: e_m.own_matvec_1rt(torch.zeros(8 * 144, dtype=torch.uint8), 'Q4_K', 8, 256, B(256, torch.float16), B(8))) and
      refused(lambda: e_s.own_matvec_1rt(q4(8, 256), 'Q4_0', 8, 256, B(256, torch.float16), B(8))) and
      refused(lambda: e_s.own_argmax(B(300), 300, torch.zeros(1, dtype=torch.int64))) and
      refused(lambda: e_s.own_emb(torch.zeros(64 * 144, dtype=torch.uint8), 'Q4_K', 64, 256, torch.zeros(1, dtype=torch.int64), B(256), key=True)) and
      refused(lambda: e_s.own_add_rms16(B(256), B(256), B(256), B(512, torch.float16), 1e-6, h16_stride=512)),
      'the drafter module refuses Q4_K; the Swift module refuses Q4_0, the argmax, the key embedding and the h16 stride')
# the qwen35moe module's experts tile at 4 rows per wave: the host's table matches the module file, and the fused gate | up
# refuses tasks whose gate and up rows would fall to different waves (16 rows there: NaN logits, a race on the gate sums)
rw_src = int(re.search(r'^#define OWN_K3_RW (\d+)', open(os.path.join(HERE, '..', '..', 'kernels', 'dec_q35m.hip')).read(), re.M).group(1))
e_q = k3.Engine(module='dec_q35m'); q2 = lambda r, kk: torch.zeros(r * kk // 64 * 18 + 64, dtype=torch.uint8)
ids8 = torch.zeros(8, dtype=torch.int32)
def gu_ok(rpt):   # (past the host's argument checks: a CPU stand-in then stops at the device asserts)
    try: e_q.own_matvec_1rt(q2(1024, 2048), 'Q2_0', 512, 2048, B(2048, torch.float16), B(8 * 512, torch.float16), ids=ids8,
                            total_rows=1024, up=q2(1024, 2048), rows_per_task=rpt)
    except ValueError: return False
    except AssertionError: return True
    return True
check(rw_src == k3.OWN_K3_RW_OF['dec_q35m'] == 4 and all(k3.OWN_K3_RW_OF.get(m) == 4 for m in k3.Q35MOE_MODULES)
      and gu_ok(32) and gu_ok(64) and not gu_ok(16) and not gu_ok(48),
      f'the qwen35moe modules: OWN_K3_RW {rw_src} (k3.OWN_K3_RW_OF), the fused gate | up at 32 / 64 rows per task, refused at 16 / 48')
# OP_PREFETCH: one segment over every range, its 128-byte lines from each base rounded down; the qwen35moe modules only
pw = [B(1000, torch.uint8), B(64)]
pf = e_q.prefetch(pw, 16, threads=16, dep=3); sg = e_q.segs[pf]
rec = [int(v) for v in (ctypes.c_int64 * 4).from_address(sg.W)]
lines = lambda w: (w.data_ptr() + w.numel() * w.element_size() - (w.data_ptr() & ~127) + 127) // 128
check(sg.op == k3.OP_PREFETCH and (sg.ntask, sg.dep, sg.rows_per_task, sg.K) == (16, 3, 16, 2)
      and rec == [pw[0].data_ptr() & ~127, lines(pw[0]), pw[1].data_ptr() & ~127, lines(pw[1])] and sg.N == rec[1] + rec[3]
      and e_q.prefetch([B(0)], 16) is None and refused(lambda: e_s.prefetch(pw, 16)),
      f'OP_PREFETCH: one segment, 16 tasks x 16 threads over {sg.N} lines of two ranges; nothing for no bytes; the dense module refuses it')
# k3.Engine.split (the qwen35moe sliced pass, Moe35Own._build_slices): a tape cut around its split attention pair runs as
# launches of their own, each on its own sync row; a dependency inside a launch is renumbered, one on an earlier launch
# dropped (stream order holds it); the pass parameters, the bound guard and the Q8 error word stay the tape's
e_sp = k3.Engine(module='dec_q35m_kvq8'); M_, R_ = 2 * 256, 64
cache_, pos_ = torch.zeros(R_ * k3.kv_pos_bytes(M_, True), dtype=torch.uint8), torch.zeros(1, dtype=torch.int32)
ap_ = e_sp.kv_append(B(M_), B(M_), cache_, pos_, q8=True)
cb_ = e_sp.also(e_sp.attn_dec(B(16 * 256), cache_, pos_, B(16 * 256), 16, 2, 256, 0.0625, q8=True, dep=ap_, max_cells=R_), ap_)
g1_ = e_sp.gate_mul16(B(16 * 256), B(2 * 16 * 256), 512, 256, B(16 * 256, torch.float16), 256, dep=cb_)
g2_ = e_sp.also(e_sp.gate_mul16(B(16 * 256), B(2 * 16 * 256), 512, 256, B(16 * 256, torch.float16), 256, dep=g1_), ap_)
e_sp.build()
lns, blk = e_sp.split([1, 3])
dd = [[(sg.op, sg.dep, sg.dep2p1) for sg in e.segs] for e in lns]
check(len(lns) == 3 and tuple(blk.shape) == (3, k3.SYNC_WORDS + 8) and all(e.sync.data_ptr() == blk[j].data_ptr() for j, e in enumerate(lns))
      and [e.base for e in lns] == [0, 1, 3] and dd == [[(k3.OP_KV_APPEND, -1, 0)], [(k3.OP_ATTN_PART, -1, 0), (k3.OP_ATTN_COMB, 0, 0)],
                                                        [(k3.OP_GATE_MUL16, -1, 0), (k3.OP_GATE_MUL16, 0, 0)]]
      and all(e.kv_error is e_sp.kv_error and e.params is e_sp.params and e.guard_caps == e_sp.guard_caps and e.module == e_sp.module
              and e.grid == e_sp.grid and not e.refuse for e in lns)
      and [(sg.op, sg.dep, sg.dep2p1) for sg in e_sp.segs][3:] == [(k3.OP_GATE_MUL16, 2, 0), (k3.OP_GATE_MUL16, 3, 1)]
      and refused(lambda: e_sp.split([3, 1])) and refused(lambda: e_sp.split([0])) and refused(lambda: e_sp.split([5])),
      f'k3.Engine.split: 3 launches on rows of one sync block, dependencies {dd}, the tape\'s guard, parameters and Q8 error '
      f'word; the tape itself unchanged; cuts out of order or outside (0, 5) refused')
# split(cuts, order): the launches run the tape in another order (Moe35Own: each attention block's prefetch behind its
# pair); cuts are positions in it, dependencies renumbered through it, .orig the tape indices; an order that puts a
# segment before its dependency, or is not a permutation, refused
e_so = k3.Engine(module='dec_q35m_kvq8')
ao_ = e_so.kv_append(B(M_), B(M_), cache_, pos_, q8=True)
pf_ = e_so.prefetch([B(4096)], 4, dep=ao_)
co_ = e_so.also(e_so.attn_dec(B(16 * 256), cache_, pos_, B(16 * 256), 16, 2, 256, 0.0625, q8=True, dep=ao_, max_cells=R_), ao_)
go_ = e_so.gate_mul16(B(16 * 256), B(2 * 16 * 256), 512, 256, B(16 * 256, torch.float16), 256, dep=co_)
e_so.build()
lo_, _ = e_so.split([1, 3], order=[0, 2, 3, 1, 4])
do_ = [[(sg.op, sg.dep, sg.dep2p1) for sg in e.segs] for e in lo_]
check(pf_ == 1 and [e.orig for e in lo_] == [[0], [2, 3], [1, 4]] and [e.base for e in lo_] == [0, 1, 3]
      and do_ == [[(k3.OP_KV_APPEND, -1, 0)], [(k3.OP_ATTN_PART, -1, 0), (k3.OP_ATTN_COMB, 0, 0)], [(k3.OP_PREFETCH, -1, 0), (k3.OP_GATE_MUL16, -1, 0)]]
      and refused(lambda: e_so.split([1, 3], order=[0, 3, 2, 1, 4])) and refused(lambda: e_so.split([1, 3], order=[0, 2, 3, 1, 1])),
      f'k3.Engine.split(order): the prefetch moved behind the pair, launches {do_}; an order against a dependency or with a repeat refused')
e_m.own_matvec_1rt(q4(8, 256), 'Q4_0', 8, 256, B(256, torch.float16), B(8)); e_m.own_argmax(B(300), 300, torch.zeros(1, dtype=torch.int64))
e_m.build()
check(set(e_m.inventory()) == {('OWN_MV', 'Q4_0/a16_1rt'), ('OWN_ARGMAX', '-')} and not e_m.unsupported(), 'the drafter module builds a Q4_0 matvec and the argmax')
e_g = k3.Engine(module='dec_q35_swift_mtp'); e_g.gdn_l2(B(512), B(512), 2, 128, 1e-6)
check(refused(lambda: e_g.build()), 'the drafter module refuses an op of the trunk its tape never runs (a Gated DeltaNet op)')
check(k3.key_token(((0x80000000 | 1234) << 32) | (0xffffffff ^ 4321)) == 4321, 'key_token: the low half\'s complement')
e_lean, e_q35 = k3.Engine(module='dec_q3m'), k3.Engine(module='dec_q35_swift')
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
e_bad = k3.Engine(module='dec_q35_swift'); e_bad.own_f16(B(256), B(256, torch.float16))
check(refused(lambda: e_bad.build()), 'a tape with an op outside the module\'s dispatch does not build')

# ---- the ThinkingCap module (dec_q35_tcap: kernels/ops_q35_tcap.h) ----------------------------------------------------------
PT = os.path.join(A.scratch, 'tiny-q35-tcap.gguf'); tiny_q35.write(PT, variant='tcap')
lt = ref_q35.Q35Ref(PT).feed([1, 7, 3, 60])
check(lt.shape == (4, info['V']) and np.isfinite(lt).all() and np.abs(lt).max() > 1e-3,
      f'the reference reads the tiny ThinkingCap-format model (IQ2_S embedding, BF16 alpha / beta; |logit| up to {np.abs(lt).max():.2f})')
mt = Dense35Own(PT, max_ctx=64, vram_check=False); Et = mt.E; it = Et.inventory()
check(mt.module == 'dec_q35_tcap' and m.module == 'dec_q35_swift', f'the module comes from the file\'s formats ({mt.module}; Swift\'s: {m.module})')
want_t = ({('OWN_MV', f'{f}/a16_1rt') for f in k3.OWN_Q35T_1RT} | {('OWN_MV16', 'Q8_0'), ('F32_MV', 'F32'), ('OWN_SWIGLU16', '-'),
          ('OWN_EMB', 'IQ2_S'), ('OWN_ADD_RMS16', '-')} | {('ATTN', 'PART'), ('ATTN', 'COMB')})
check(set(it) == want_t, f'the ThinkingCap tape holds every tile, no fused gate|up, BF16 alpha / beta as F32: {sorted(it.items())}')
check(not Et.unsupported() and all(sg.op in k3.Q35S_OPS for sg in Et.segs) and Et.refuse is None,
      f'{len(Et.segs)} segments, every op one the ThinkingCap module dispatches')
check(Et.segs[0].op == k3.OP_OWN_EMB and Et.segs[0].out_f32 == 1 | 2 | 8,
      f'the embedding carries the grids to stage: IQ3_XXS | IQ3_S | IQ2_S = 11 (got {Et.segs[0].out_f32})')
check(abs(Dense35Own.projection_mib(PT, 64) - mt.vram_mib) < 1e-9, 'the projection counts the widened BF16 and the record tails as the engine')
srct = open(os.path.join(HERE, '..', '..', 'kernels', 'ops_q35_tcap.h')).read()
dt = srct.split('k3_extra_op(')[1]
dt = re.sub(Q8B, lambda mm: mm.group(1), dt, flags=re.S)
dft_ops = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', ''.join(re.findall(DFB, dt, re.S))))
dt = re.sub(DFB, '', dt, flags=re.S)   # (ThinkingCap's DFlash drafter module's ops: k3.Q35TDF_MODULES only)
mtt_ops = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', ''.join(re.findall(MTPB, dt, re.S))))
dt = re.sub(MTPB, '', dt, flags=re.S)   # (ThinkingCap's MTP drafter module's argmax: k3.Q35TM_MODULES only)
dev_t = set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', re.sub(SPB, lambda mm: mm.group(1), re.sub(TPB, '', dt, flags=re.S), flags=re.S)))
check(dev_t == host_ops and set(re.findall(r'sg\.op == (OP_[A-Z0-9_]+)', ''.join(re.findall(TPB, dt, re.S)))) == {'OP_XCHG'},
      f'kernels/ops_q35_tcap.h dispatches k3.Q35S_OPS too ({len(dev_t)} ops), and OP_XCHG under K3_TP')
check(mtt_ops == {'OP_OWN_ARGMAX'}, 'under K3_Q35_MTP in ops_q35_tcap.h: the argmax (ThinkingCap\'s MTP drafter module)')
check(dft_ops == {'OP_DF_CONV', 'OP_DF_TOPK', 'OP_DF_SELECT', 'OP_KV_APPEND', 'OP_ATTN_DEC'}, 'under K3_Q35_DF in ops_q35_tcap.h: the DFlash drafter\'s own ops (ThinkingCap\'s drafter module)')
check(int(re.search(r'OWN_IQ2_S = (\d+)', srct).group(1)) == k3.OWN_Q35T_FMT['IQ2_S'], 'the IQ2_S format number is k3\'s')
e_tc = k3.Engine(module='dec_q35_tcap')
check(refused(lambda: e_tc.own_matvec_1rt(torch.zeros(8 * 144 + 16, dtype=torch.uint8), 'Q4_K', 8, 256, B(256, torch.float16), B(8))) and
      refused(lambda: e_tc.own_matvec_1rt(torch.zeros(16 * 176, dtype=torch.uint8), 'Q5_K', 8, 256, B(256, torch.float16), B(8, torch.float16),
                                          up=torch.zeros(16 * 176, dtype=torch.uint8))) and
      refused(lambda: e_q35.own_matvec_1rt(torch.zeros(8 * 82 + 16, dtype=torch.uint8), 'IQ2_S', 8, 256, B(256, torch.float16), B(8))) and
      refused(lambda: e_q35.own_emb(torch.zeros(64 * 82, dtype=torch.uint8), 'IQ2_S', 64, 256, torch.zeros(1, dtype=torch.int32), B(256))),
      'the ThinkingCap module refuses Q4_K and a fused gate|up; the Swift module refuses IQ2_S matvecs and embeddings')

# ---- the verify tape (engine/q35_dense_verify.py, module dec_q35_swift_ver) -----------------------------------------------
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
      refused(lambda: Dense35Verify(mv4, 4, module='dec_q35_swift', vram_check=False)) and refused(lambda: Dense35Own(P, max_ctx=64, vram_check=False, spec_T=5)),
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
# ThinkingCap's verify tape (dec_q35_tcap_ver: kernels/quant_tcap_ver.h's T-token tiles)
mt4 = Dense35Own(PT, max_ctx=64, vram_check=False, spec_T=4)
vt = Dense35Verify(mt4, 4, vram_check=False); Evt = vt.E; ivt = Evt.inventory()
check(vt.module == 'dec_q35_tcap_ver' and set(ivt) == {('OWN_MV', f'{f}/a16_1rt_mt') for f in k3.OWN_Q35T_1RT} |
      {('OWN_MV16', 'Q8_0'), ('F32_MV', 'F32'), ('OWN_SWIGLU16', '-'), ('OWN_EMB', 'IQ2_S'), ('OWN_ADD_RMS16', '-')} | {('ATTN', 'PART'), ('ATTN', 'COMB')},
      f'ThinkingCap\'s verify tape on dec_q35_tcap_ver: every format\'s matvec in its T-token form ({sorted(ivt)})')
check(not Evt.unsupported() and Evt.refuse is None and Evt.segs[0].op == k3.OP_OWN_EMB and Evt.segs[0].out_f32 == 1 | 2 | 8
      and all(sg.team == 4 for sg in Evt.segs) and all(sg.pad1 >> 8 == 4 for sg in Evt.segs if sg.op == k3.OP_OWN_MV),
      f'{len(Evt.segs)} segments; the embedding stages the grids (11); every matvec carries 4 tokens')
check(refused(lambda: Dense35Verify(mt4, 4, module='dec_q35_swift_ver', vram_check=False)) and
      refused(lambda: Dense35Verify(mv4, 4, module='dec_q35_tcap_ver', vram_check=False)) and
      refused(lambda: k3.Engine(module='dec_q35_tcap', T=1).own_matvec_1rt(B(4096, torch.uint8), 'IQ4_XS', 2, 256, B(4 * 256, torch.float16), B(8), tokens=4, x_stride=256, out_stride=2)),
      'refused: a verify module of the other line, and T-token matvecs on ThinkingCap\'s decode module')
# ---- the GSQ module (dec_q35_gsq) and its verify modules (dec_q35_gsq_ver + dec_q35_gsq_ver_m: three launches a pass) --------
PG = os.path.join(A.scratch, 'tiny-q35-gsq.gguf'); tiny_q35.write(PG, variant='gsq')
mg = Dense35Own(PG, max_ctx=64, vram_check=False, spec_T=4); Eg = mg.E; ig = Eg.inventory()
gfm = sorted({f for fm in tiny_q35.FORMATS_GSQ for f in fm.values()} | {'Q4_K'})   # (the blocks' formats + the Q4_K head)
check(mg.module == 'dec_q35_gsq' and {f for (o, f) in ig if o == 'OWN_MV'} == {f'{f}/a16_1rt' for f in gfm}
      and not Eg.unsupported() and Eg.segs[0].op == k3.OP_OWN_EMB and Eg.segs[0].out_f32 == 1 | 2 | 4 | 8 | 16,
      f'the GSQ file on dec_q35_gsq: every format on its one-round-trip tile; the embedding stages five grids (31; got {Eg.segs[0].out_f32})')
vg = Dense35Verify(mg, 4, vram_check=False)
lm = [e.module for e in vg.Es]; inv = [e.inventory() for e in vg.Es]
check(set(lm) <= set(kernels.RUNTIME_SET), 'the real GSQ verify tapes select shipped runtime objects, including the IQ1_M launch')
check(vg.module == 'dec_q35_gsq_ver' and lm == ['dec_q35_gsq_ver', 'dec_q35_gsq_ver_m', 'dec_q35_gsq_ver']
      and set(inv[1]) == {('OWN_MV', 'IQ1_M/a16_1rt_mt')} and len(vg.Es[1].segs) == 1,
      f'the GSQ verify pass: three launches ({lm}); the IQ1_M gate alone on dec_q35_gsq_ver_m')
mvf = set().union(*[{f for (o, f) in iv if o == 'OWN_MV'} for iv in inv])
check(mvf == {f'{f}/a16_1rt_mt' for f in gfm}, f'every format\'s matvec in its T-token form across the launches ({sorted(mvf)})')
check(all(not e.unsupported() and e.refuse is None and all(sg.dep < i and sg.dep2p1 - 1 < i for i, sg in enumerate(e.segs)) for e in vg.Es)
      and all(e.segs[0].op == k3.OP_OWN_EMB and e.segs[0].out_f32 for e in (vg.Es[0], vg.Es[2]))
      and all(e.params.data_ptr() == vg.Es[0].params.data_ptr() and e.params.numel() == 16 and e.guard_T == 4 for e in vg.Es),
      'each launch: ops its module dispatches, dependencies inside it pointing back; the grid launches start with an embedding '
      'that stages them; one pass-parameter block and guard for all three')
check(refused(lambda: Dense35Verify(mg, 4, module='dec_q35_tcap_ver', vram_check=False)) and
      refused(lambda: Dense35Verify(mt4, 4, module='dec_q35_gsq_ver', vram_check=False)) and
      refused(lambda: k3.Engine(module='dec_q35_gsq_ver', T=4).own_matvec_1rt(B(4096, torch.uint8), 'IQ1_M', 2, 256, B(4 * 256, torch.float16), B(8), tokens=4, x_stride=256, out_stride=2)) and
      refused(lambda: k3.Engine(module='dec_q35_gsq_ver_m', T=4).own_matvec_1rt(B(4096, torch.uint8), 'Q4_K', 2, 256, B(4 * 256, torch.float16), B(8), tokens=4, x_stride=256, out_stride=2)),
      'refused: another line\'s verify module on either engine, IQ1_M on dec_q35_gsq_ver, any other format on dec_q35_gsq_ver_m')
check(not any(c[0] == 'OWN_MV' for c in k3.MODULE_CAPS['dec_q35_tcap_dflash']),
      'the ThinkingCap DFlash module lists no OWN_MV pair (its device code compiles OP_OWN_MV and the IQ grids out: kernels/ops_q35_tcap.h K3_Q35_DF)')
# ThinkingCap's MTP drafter (dec_q35_tcap_mtp: engine/q35_mtp.py on its Q6_K block, Q8_0 head, IQ2_S embedding)
from caminus.engine.q35_mtp import Mtp35
PM = os.path.join(A.scratch, 'tiny-q35-tcap-mtp.gguf'); tiny_q35.write(PM, variant='tcap', mtp=True)
mm4 = Dense35Own(PM, max_ctx=64, vram_check=False, spec_T=4)
mt_ = Mtp35(mm4, k=3, vram_check=False)
ivm = set().union(*(set(E_.inventory()) for E_ in mt_.E.values()))
chain = mt_.E[('chain', 1)]
check(mt_.module == 'dec_q35_tcap_mtp' and ivm <= k3.MODULE_CAPS['dec_q35_tcap_mtp'] and ('OWN_MV', 'Q6_K/a16_1rt_mt') in ivm
      and ('OWN_MV16', 'Q8_0') in ivm and ('OWN_EMB', 'IQ2_S') in ivm and all(not E_.unsupported() for E_ in mt_.E.values())
      and chain.segs[0].op == k3.OP_OWN_EMB and chain.segs[0].pad1 == 1 and any(sg.op == k3.OP_OWN_ARGMAX for sg in chain.segs),
      f'ThinkingCap\'s MTP tapes on dec_q35_tcap_mtp: the block\'s Q6_K T-token tiles, the Q8_0 head, the IQ2_S embedding from '
      f'the argmax key on a chained step ({len(mt_.E)} tapes; {sorted(ivm)})')
check(refused(lambda: Mtp35(mm4, module='dec_q35_swift_mtp', vram_check=False)) and
      refused(lambda: k3.Engine(module='dec_q35_tcap_mtp', T=4).own_matvec_1rt(B(4096, torch.uint8), 'IQ4_XS', 2, 256, B(4 * 256, torch.float16), B(8), tokens=4, x_stride=256, out_stride=2)),
      'refused: the other line\'s drafter module, and a format other than Q6_K on ThinkingCap\'s MTP module')
# the GSQ files' MTP drafter (dec_q35_gsq_mtp: their Q6_K block as ThinkingCap's, the Q4_K head, the IQ2_S embedding)
PGM = os.path.join(A.scratch, 'tiny-q35-gsq-mtp.gguf'); tiny_q35.write(PGM, variant='gsq', mtp=True)
mg4 = Dense35Own(PGM, max_ctx=64, vram_check=False, spec_T=4)
mtg = Mtp35(mg4, k=3, vram_check=False)
check({e.module for e in mtg.E.values()} <= set(kernels.RUNTIME_SET), 'the real GSQ MTP tapes select shipped runtime objects')
ivg = set().union(*(set(E_.inventory()) for E_ in mtg.E.values()))
check(mtg.module == 'dec_q35_gsq_mtp' and ivg <= k3.MODULE_CAPS['dec_q35_gsq_mtp'] and ('OWN_MV', 'Q6_K/a16_1rt_mt') in ivg
      and ('OWN_MV', 'Q4_K/a16_1rt') in ivg and ('OWN_MV16', 'Q8_0') not in ivg and ('OWN_EMB', 'IQ2_S') in ivg
      and all(not E_.unsupported() for E_ in mtg.E.values()),
      f'the GSQ files\' MTP tapes on dec_q35_gsq_mtp: the Q6_K block\'s T-token tiles, the Q4_K head on its one-round-trip tile ({sorted(ivg)})')
check(refused(lambda: Mtp35(mg4, module='dec_q35_tcap_mtp', vram_check=False)) and refused(lambda: Mtp35(mm4, module='dec_q35_gsq_mtp', vram_check=False)) and
      refused(lambda: k3.Engine(module='dec_q35_gsq_mtp', T=4).own_matvec_1rt(B(4096, torch.uint8), 'IQ2_XS', 2, 256, B(4 * 256, torch.float16), B(8), tokens=4, x_stride=256, out_stride=2)) and
      refused(lambda: k3.Engine(module='dec_q35_tcap_mtp', T=4).own_matvec_1rt(B(4096, torch.uint8), 'Q4_K', 2, 256, B(4 * 256, torch.float16), B(8), tokens=4, x_stride=256, out_stride=2)),
      'refused: ThinkingCap\'s MTP module on the GSQ engine and back, IQ2_XS on the GSQ MTP module, Q4_K on ThinkingCap\'s')
check(set(k3.MODULE_CAPS['dec_q35_gsq_dflash']) - set(k3.MODULE_CAPS['dec_q35_tcap_dflash']) == {('OWN_MV', 'Q4_K/a16_1rt'), ('OWN_MV', 'Q4_K/a16_1rt_mt')},
      'the GSQ DFlash module: ThinkingCap\'s DFlash caps plus the Q4_K head\'s matvec, nothing else')
check(('OWN_MV', 'Q4_K/a16_1rt_gu') not in k3.MODULE_CAPS['dec_q35_swift_ver'] and ('OWN_MV', 'Q4_K/a16_1rt_gu') in k3.MODULE_CAPS['dec_q35_swift'],
      'the verify module\'s capabilities leave out the fused gate | up (no verify tape runs it)')
e_v = k3.Engine(module='dec_q35_swift_ver', T=4)
check(refused(lambda: e_q35.own_matvec16(w8, 'Q8_0', 8, 256, B(4 * 256, torch.float16), B(32), tokens=4)) and
      refused(lambda: e_v.own_matvec16(w8, 'Q8_0', 8, 256, B(4 * 256, torch.float16), B(32), tokens=2)) and
      refused(lambda: e_v.own_add_rms16(B(4 * 256), B(256), B(4 * 256), B(4 * 256, torch.float16), 1e-6, tokens=2)) and
      refused(lambda: e_q35.gdn_norm16(B(256), B(128), B(256), B(256, torch.float16), 2, 128, 1e-6) or k3.Engine(module='dec_q35_swift', T=2).gdn_norm16(B(512), B(128), B(512), B(512, torch.float16), 2, 128, 1e-6)),
      'the T-token forms: refused on the decode module and for a token count other than the tape\'s T')

# ---- staged Q8 target: allocation, tape order, verify and terminal failures -------------------------------------------
from caminus.prefill.q35own import Q35OwnPrefill
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
    check(sum(sg.op == k3.OP_HADAMARD for sg in target.E.segs) == 4 * n_att,
          f'{target.module}: Q/K256, V/output64 transforms on every attention layer')
    app = next(i for i, sg in enumerate(target.E.segs) if sg.op == k3.OP_KV_APPEND)
    gates = next(i for i, sg in enumerate(target.E.segs) if sg.op == k3.OP_GATE_MUL16)
    sg = target.E.segs[app]
    check(target.E.segs[sg.dep].op == k3.OP_HADAMARD and target.E.segs[sg.dep2p1 - 1].op == k3.OP_HADAMARD
          and target.E.segs[target.E.segs[gates].dep].op == k3.OP_HADAMARD,
          f'{target.module}: append waits for both rotations, output rotation precedes gating')
    vv = Dense35Verify(target, 4, vram_check=False)
    check(vv.module == k3.Q35_KV_Q8_OF['dec_q35_tcap_ver' if target.tcap else 'dec_q35_swift_ver']
          and vv.module not in k3.ATTN_PACK_MODULES and not vv.E.unsupported(),
          f'{target.module}: matching unpacked verify tape constructs for T4')
    check(sum(sg.op == k3.OP_HADAMARD for sg in vv.E.segs) == 4 * n_att,
          f'{target.module}: verify carries every transform')
    check(refused(lambda: Dense35Verify(target, 4, module='dec_q35_tcap_ver' if target.tcap else 'dec_q35_swift_ver', vram_check=False)),
          f'{target.module}: F16 verify cannot reinterpret the Q8 state')
check(refused(lambda: Dense35Own(P, module='dec_q35_swift_kvq8', vram_check=False))
      and refused(lambda: q35_dense.kv_module(None, 'dec_q35_swift', q8='q8')),
      'Q8-only module and strict cache-format flag fail closed before allocation')
delta = n_att * 67 * (k3.kv_pos_bytes(qm.nkv * qm.hd, False) - k3.kv_pos_bytes(qm.nkv * qm.hd, True)) - 4
check(abs((Dense35Own.projection_mib(P, 64, 4) - qm.vram_mib) * 2 ** 20 - delta) < 1e-6,
      'Q8 memory reduction is exact row packing minus the new four-byte error word')
check(refused(lambda: Q35OwnPrefill(types.SimpleNamespace(q8=True))),
      'unimplemented matching batched prefill refuses before looking up buffers or loading a module')
check(not any(m in k3.Q35_KV_Q8_MODULES for m in k3.Q35_TP_MODULES + k3.Q35TPV_MODULES),
      'no two-card Q8 object is registered on this branch')

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
qt.weight_sums = lambda: (_ for _ in ()).throw(AssertionError('a refused KV encoding must not read the weights'))
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
qm.weight_sums = qt.weight_sums
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

def refused_with(fn, want):
    try: fn(); return False
    except ValueError as e: return want in str(e)


# The GSQ line: a Q8 decode object exists, its two verify objects have no Q8 build, so verify refuses the engine.
mgq = Dense35Own(PG, max_ctx=64, spec_T=4, q8=True, vram_check=False)
check(mgq.module == 'dec_q35_gsq_kvq8' and mgq.module in k3.Q35R_MODULES and not mgq.E.unsupported()
      and any(sg.op == k3.OP_HADAMARD for sg in mgq.E.segs)
      and refused_with(lambda: Dense35Verify(mgq, 4, vram_check=False), 'no Q8 KV verify objects for the GSQ line'),
      'GSQ: the Q8 decode tape builds on dec_q35_gsq_kvq8; Dense35Verify refuses it (no Q8 verify objects)')

# A pass of several launches (the GSQ shape): a refused encoding in a LATER launch wins over a bound report and a
# timeout in the first, before any status is cleared or the pass is run again.
qb = Dense35Own(P, max_ctx=64, spec_T=4, q8=True, vram_check=False)
qb.weight_sums = qt.weight_sums
qv2 = Dense35Verify(qb, 4, vram_check=False)
e2 = k3.Engine(module=qv2.module, T=4)
qv2.E.run = e2.run = lambda **kwargs: None
qv2.Es.append(e2); qv2.h_sts.append(torch.zeros(k3.STATUS_WORDS, dtype=torch.int32))
def sync_two():
    qv2.h_sts[0][4] = 1; qv2.h_sts[0][5] = -1; qv2.h_sts[0][1] = 1   # (launch 1: a bound report and a timeout)
    e2.kv_error.fill_(2)                                                # (launch 2: a refused block)
qv2.ev_done = types.SimpleNamespace(record=lambda: None, synchronize=sync_two)
try:
    qv2.run([1, 2, 3, 4], 0, 0, argmax=False); multi = 'none'
except InvalidKVEncoding: multi = 'kv'
except k3.ContextBoundsError: multi = 'bounds'
check(multi == 'kv' and not qv2.bails and qb.broken and not qb.poisoned and int(e2.kv_error[0]) == 2,
      f'a later launch\'s refused encoding is reported before the first launch\'s bound and timeout (got {multi})')

# Q35Host.physics_bytes (the server's bandwidth floor): the routed experts count at k_used / n_exp (a token reads k_used
# of each layer's n_exp), everything else once, MTP blocks and the embedding table out (one row instead)
T_ = lambda n, b: types.SimpleNamespace(name=n, n_bytes=b)
Fm = types.SimpleNamespace(hp=lambda k: 3, tensors=[T_('token_embd.weight', 10 ** 6), T_('blk.0.attn_qkv.weight', 1000),
     T_('blk.0.ffn_gate_exps.weight', 256_000), T_('blk.0.ffn_down_exps.weight', 512_000), T_('blk.0.ffn_up_shexp.weight', 300),
     T_('blk.1.ffn_gate_exps.weight', 256_000), T_('blk.2.ffn_gate_exps.weight', 10 ** 7), T_('output.weight', 5000)])
moe_pb = q35_dense.Q35Host.physics_bytes(types.SimpleNamespace(F=Fm, nl=2, D=256, k_used=8, n_exp=256))[0]
dense_pb = q35_dense.Q35Host.physics_bytes(types.SimpleNamespace(F=Fm, nl=2, D=256))[0]
check(moe_pb == 1000 + 300 + 5000 + (256_000 + 512_000 + 256_000) * 8 / 256 + 144 and dense_pb == 1000 + 300 + 5000 + 1_024_000 + 144,
      f'physics_bytes: routed experts at k_used / n_exp ({moe_pb:.0f} bytes), all of them without n_exp ({dense_pb:.0f})')

print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
