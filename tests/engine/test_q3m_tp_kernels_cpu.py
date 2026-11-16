"""The qwen3moe decode module's two-card build (kernels/k3_q3m_lean_tp.hip, + _trace; issue #86, PR 1) without a card:
  - the device text: kernels/k3_q3m_ops.h dispatches OP_XCHG and OP_OWN_SLOT_SUM only under K3_Q3M_TP, inside its
    K3_OPS_NO_DISPATCH block (so the dense modules that take its ops, k3_q35_ops.h, never see them), with the exchange's
    warm word and claim-time store under the same macro; only the two wrappers define K3_Q3M_TP / K3_XCHG_PUSH_ONLY; the
    slot sum is OP_OWN_MOE_SUM's fma chain with the sum stored; the op number is the host's
  - the host: k3.Q3M_TP_OF maps the lean modules to the two-card ones, their .hip files exist and build.sh compiles them as
    the other own decode modules (max-ilp, no ggml include path); their capabilities are lean's plus OWN_SLOT_SUM;
    Engine.build accepts OP_XCHG / OP_OWN_SLOT_SUM on them only (the lean, verify, IQ and row-scale modules refuse both),
    refuses every other op outside the lean tape's set there, and refuses a pull-mode exchange (the module is push-only);
    own_slot_sum refuses another module; the served two-card object ships in the installed runtime set, its trace build does not
  - OP_OWN_ATTS's split cap: kernels/k3_own_atts.h sizes its merge from ATS_MAXS (128 under K3_Q3M_TP, else 64: the
    one-card text in the #else branch, the 128 form writing each weight over its m inside q + scores); Engine.own_atts
    takes up to k3.atts_max_splits(module) (128 on the two-card modules, 64 on lean, verify, IQ and row-scale) and refuses more
  - tests/engine/tiny_q3m.py: the synthetic qwen3moe file decodes on the float64 reference (caminus.ref_q3moe), and
    Q3MoEOne builds its decode tape for it on stand-ins (nothing launched): the same segments on k3_q3m_lean and on
    k3_q3m_lean_tp, every op in the lean set
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/engine/test_q3m_tp_kernels_cpu.py --scratch DIR"""
import argparse, contextlib, os, re, sys, types
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


def src(name):
    with open(os.path.join(ROOT, 'kernels', name)) as f: return f.read()


# ---- the device text ----------------------------------------------------------------------------------------------------
ops_h = src('k3_q3m_ops.h')
nd = ops_h[ops_h.index('#ifndef K3_OPS_NO_DISPATCH'):]
pre = ops_h[:ops_h.index('#ifndef K3_OPS_NO_DISPATCH')]
code = lambda s: re.sub(r'//[^\n]*', '', s)   # (comments may name the ops)
check('OP_XCHG' not in code(pre) and 'OP_OWN_SLOT_SUM' not in code(pre) and 'k3_xchg.h' not in code(pre) and 'K3_Q3M_TP' not in code(pre),
      'kernels/k3_q3m_ops.h: nothing of the two-card build before its K3_OPS_NO_DISPATCH block (the dense modules include the ops above it)')
blocks = re.findall(r'#ifdef K3_Q3M_TP[^\n]*\n(.*?)#(?:endif|else)', nd, re.S)
inside = ''.join(blocks)
outside = re.sub(r'#ifdef K3_Q3M_TP[^\n]*\n.*?#(?:endif|else)', '', nd, flags=re.S)
check(len(blocks) == 4 and '#include "k3_xchg.h"' in inside and 'own_op_slot_sum' in inside
      and re.search(r'if \(sg\.op == OP_XCHG\) \{ op_xchg\(sg, sy, limit_us, seg, tr\); return true; \}', inside)
      and re.search(r'if \(sg\.op == OP_OWN_SLOT_SUM\) \{ own_op_slot_sum\(sg, t\); return true; \}', inside)
      and 'return xchg_warm(sg);' in inside and 'xchg_prefetch(sg);' in inside,
      'k3_q3m_ops.h: the exchange header, the slot sum, both dispatch lines, the warm word and the claim-time store under K3_Q3M_TP')
check(not re.search(r'OP_XCHG|OP_OWN_SLOT_SUM|xchg_warm|xchg_prefetch|k3_xchg\.h', code(outside)),
      'k3_q3m_ops.h: no exchange or slot sum outside K3_Q3M_TP (the one-card modules\' dispatch is unchanged)')
disp = nd[nd.index('__device__ __forceinline__ bool k3_extra_op('):]; disp = disp[:disp.index('\n}\n')]
check(disp.count('OP_XCHG') == 1 and disp.count('OP_OWN_SLOT_SUM') == 1, 'k3_q3m_ops.h: k3_extra_op dispatches each new op once')
pf = nd[nd.index('__device__ __forceinline__ void k3_extra_prefetch('):]; pf = pf[:pf.index('\n}\n')]
check(pf.index('xchg_prefetch') < pf.index('if (sg.op != OP_GG_MV'), 'k3_q3m_ops.h: the claim-time store comes before the router prefetch\'s early return')
ss = inside[inside.index('own_op_slot_sum'):]
ms = src('k3_own_ops.h'); ms = ms[ms.index('void own_op_moe_sum('):]; ms = ms[:ms.index('\n}\n')]
check('acc = __builtin_fmaf(w[s], d[(size_t)s * sg.N + i], acc);' in ss and 'y[i] = acc;' in ss and 'float acc = 0.0f;' in ss
      and '__builtin_fmaf(w[s], d[(size_t)s * sg.N + i], acc)' in ms and 'x[i] += acc;' in ms,
      'own_op_slot_sum: OP_OWN_MOE_SUM\'s fma chain in slot order from 0.0f, the sum stored (moe_sum adds it)')
m = re.search(r'enum \{ OP_OWN_SLOT_SUM = (\d+) \}', ops_h)
from caminus.engine import k3
nums = [int(n) for n in re.findall(r'OP_[A-Z0-9_]+ = (\d+)', ''.join(src(f) for f in sorted(os.listdir(os.path.join(ROOT, 'kernels'))) if f.endswith('.h') or f == 'k3.hip'))]
check(m and int(m.group(1)) == k3.OP_OWN_SLOT_SUM and nums.count(k3.OP_OWN_SLOT_SUM) == 1,
      f'OP_OWN_SLOT_SUM = {k3.OP_OWN_SLOT_SUM} on the device and the host, and no other op has that number')
xh = src('k3_xchg.h')
check(xh.count('#if defined(K3_MT) || defined(K3_XCHG_PUSH_ONLY)') == 2 and '#ifdef K3_MT' not in xh,
      'kernels/k3_xchg.h: K3_XCHG_PUSH_ONLY selects the multi-token module\'s push-only form (both places)')
defs = {}
for f in sorted(os.listdir(os.path.join(ROOT, 'kernels'))):
    if f.endswith(('.hip', '.h')):
        for mac in ('K3_Q3M_TP', 'K3_XCHG_PUSH_ONLY'):
            if re.search(r'#define ' + mac + r'\b', src(f)): defs.setdefault(mac, []).append(f)
check(defs == {'K3_Q3M_TP': ['k3_q3m_lean_tp.hip'], 'K3_XCHG_PUSH_ONLY': ['k3_q3m_lean_tp.hip']},
      f'only kernels/k3_q3m_lean_tp.hip defines K3_Q3M_TP and K3_XCHG_PUSH_ONLY ({defs})')
tp, tpt = src('k3_q3m_lean_tp.hip'), src('k3_q3m_lean_tp_trace.hip')
check(all(re.search(r'#define ' + mac + r'\b', tp) for mac in ('K3_TP', 'K3_Q3M_TP', 'K3_XCHG_PUSH_ONLY'))
      and tp.rstrip().endswith('#include "k3_q3m_lean.hip"') and '#define K3_TRACE' in tpt and tpt.rstrip().endswith('#include "k3_q3m_lean_tp.hip"'),
      'the wrappers: k3_q3m_lean_tp = K3_TP + K3_Q3M_TP + K3_XCHG_PUSH_ONLY over k3_q3m_lean; _trace = K3_TRACE over it')

# ---- the host: names, build.sh, capabilities, the build check ---------------------------------------------------------------
bs = open(os.path.join(ROOT, 'build.sh')).read()
check(k3.Q3M_TP_OF == {'k3_q3m_lean': 'k3_q3m_lean_tp', 'k3_q3m_lean_trace': 'k3_q3m_lean_tp_trace'}
      and k3.Q3M_TP_MODULES == tuple(k3.Q3M_TP_OF.values())
      and all(os.path.exists(os.path.join(ROOT, 'kernels', m + '.hip')) for m in k3.Q3M_TP_MODULES)
      and 'for src in kernels/*.hip; do' in bs and re.search(r'kernels/k3_q3m\*\.hip\|[^)]*\)\s*extra=\(-mllvm -amdgpu-sched-strategy=max-ilp\); std=c\+\+17', bs),
      'k3.Q3M_TP_OF: kernels/<name>.hip for each two-card module; build.sh compiles every kernels/*.hip, k3_q3m*.hip with max-ilp and no ggml path')
check(all(m in k3.Q3M_MODULES and k3.MODULE_CAPS[m] == k3.MODULE_CAPS['k3_q3m_lean'] | {('OWN_SLOT_SUM', '-')} for m in k3.Q3M_TP_MODULES)
      and ('OWN_SLOT_SUM', '-') not in k3.MODULE_CAPS['k3_q3m_lean'] and k3.Q3M_TP_OPS == k3.Q3M_LEAN_OPS | {k3.OP_XCHG, k3.OP_OWN_SLOT_SUM},
      'capabilities: the lean module\'s plus OWN_SLOT_SUM; ops: the lean set plus OP_XCHG and OP_OWN_SLOT_SUM')
from caminus.device import kernels as dk
shipped = set(dk.K3) | set(getattr(dk, 'DENSE35_OWN', ())) | {n for g in getattr(dk, 'PREFILL', ()) for n in (g if isinstance(g, tuple) else (g,))}
check('k3_q3m_lean_tp' in dk.RUNTIME_SET and 'k3_q3m_lean_tp_trace' not in (shipped | set(dk.RUNTIME_SET)),
      'the served two-card module ships in the installed runtime set; its trace build does not')

# stand-ins: the engine's torch.cuda calls on host memory (no device is opened, nothing launches)
_zeros, _to, _tensor = torch.zeros, torch.Tensor.to, torch.tensor
torch.tensor = lambda *a, **k: _tensor(*a, **{x: v for x, v in k.items() if x != 'device'})
torch.zeros = lambda *a, **k: _zeros(*a, **{x: v for x, v in k.items() if x not in ('device', 'pin_memory')})
torch.Tensor.to = lambda self, *a, **k: self if a and str(a[0]).startswith('cuda') else _to(self, *a, **k)
torch.Tensor.cuda = lambda self, *a, **k: self
torch.Tensor.is_cuda = property(lambda self: True)
torch.cuda.set_device = lambda d: None
torch.cuda.device = lambda d: contextlib.nullcontext()
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda d: types.SimpleNamespace(multi_processor_count=42, total_memory=20464 << 20)
torch.cuda.Event = lambda *a, **k: types.SimpleNamespace()

B = lambda n: torch.zeros(n)
D, KU = 256, 8


def add_xchg(e, pull_mode=0):
    e.xchg(B(D), B(D), 1 << 40, 2 << 40, torch.zeros(1, dtype=torch.int32), 0, 0, pull=(3 << 40) if pull_mode else None, pull_mode=pull_mode)


def builds(module, add):
    e = k3.Engine(module=module); add(e)
    try:
        e.build(); return True
    except ValueError:
        return False


tpe = k3.Engine(module='k3_q3m_lean_tp'); tpe.own_slot_sum(B(KU * D), B(D), KU, B(KU)); slot_seg = tpe.segs[0]
check(slot_seg.op == k3.OP_OWN_SLOT_SUM and slot_seg.pad1 == KU and slot_seg.N == D and k3.seg_kind(slot_seg) == ('OWN_SLOT_SUM', '-'),
      'Engine.own_slot_sum: one segment (pad1 = slots, N = the row), kind OWN_SLOT_SUM')
add_slot = lambda e: e.segs.append(slot_seg)
others = [m for m in k3.Q3M_LEAN_MODULES + k3.Q3M_VER_MODULES + k3.Q3M_IQ_MODULES + tuple(k3.Q3M_RS_MODULES)]
check(all(builds(m, add_xchg) and builds(m, add_slot) for m in k3.Q3M_TP_MODULES), 'the two-card modules build tapes with OP_XCHG and OP_OWN_SLOT_SUM')
check(not any(builds(m, add_xchg) or builds(m, add_slot) for m in others),
      f'the lean, verify, IQ and row-scale modules refuse both at build ({len(others)} modules)')
try:
    k3.Engine(module='k3_q3m_lean').own_slot_sum(B(KU * D), B(D), KU, B(KU)); r = False
except ValueError:
    r = True
check(r, 'own_slot_sum refuses a module without it (k3_q3m_lean)')
check(not builds('k3_q3m_lean_tp', lambda e: add_xchg(e, pull_mode=1)) and not builds('k3_q3m_lean_tp', lambda e: add_xchg(e, pull_mode=2)),
      'the two-card modules refuse a pull-mode exchange (push-only build)')
other_ops = [lambda e: e.own_moe_sum(B(KU * D), B(D), slots=KU, w=B(KU)), lambda e: e.own_f16(B(D), torch.zeros(D, dtype=torch.float16)),
             lambda e: e.own_swiglu16(B(D), B(D), torch.zeros(D, dtype=torch.float16))]
check(not any(builds(m, f) for m in k3.Q3M_TP_MODULES for f in other_ops),
      'the two-card modules refuse ops outside the lean set (OP_OWN_MOE_SUM, OP_OWN_F16, OP_OWN_SWIGLU16)')
check(all(builds(m, add_xchg) for m in k3.Q35_TP_MODULES) and not builds('k3_q35_swift', add_slot),
      'the dense two-card modules still take OP_XCHG; a dense module refuses OP_OWN_SLOT_SUM')
check(not builds('k3', add_slot), 'the generic k3 module (no op list, no capability set) refuses a raw OP_OWN_SLOT_SUM segment')

# ---- OP_OWN_ATTS's split cap --------------------------------------------------------------------------------------------
at = src('k3_own_atts.h')
mx = re.search(r'#ifndef ATS_MAXS\n#ifdef K3_Q3M_TP\n#define ATS_MAXS (\d+)\n#else\n#define ATS_MAXS (\d+)\n#endif\n#endif', at)
mg = at[at.index('#if ATS_MAXS > 64'):]; mg = mg[:mg.index('#endif')]
two, one = mg.split('#else')
check(mx and (int(mx.group(1)), int(mx.group(2))) == (k3.ATS_MAXS_TP, k3.ATS_MAXS) == (128, 64) and at.count('#if ATS_MAXS > 64') == 1
      and 'static_assert(16 * ATS_MAXS <= ATS_RC' in two and '#define ATS_WV(s2, h) ml[(s2) * 16 + (h)]' in two and 'j += 256' in two
      and '* wv = qs + 16 * 64;' in one and '#define ATS_WV(s2, h) wv[(s2) * 8 + (h)]' in one and 'if (tid < 4 * nact) {' in one
      and 'ATS_WV(s2, h) = w;' in at and 'ATS_WV(s0 + e, h);' in at,
      'kernels/k3_own_atts.h: ATS_MAXS 128 under K3_Q3M_TP, 64 else; the 128 merge loads m | l in rounds of 256 threads and writes '
      'each weight over its m (inside q + scores); the 64 merge keeps its weights row and one load round')


def atts(module, splits):
    """one OP_OWN_ATTS segment on module (2 kv heads, one cache row past the padding): None, or the refusal"""
    nkv, hd = 2, 128; nh, M = 8 * nkv, nkv * hd
    e = k3.Engine(module=module)
    try:
        e.own_atts(B(nh * hd + 2 * M), B(k3.Engine.atts_scratch_floats(nkv, splits)), B(2 * hd + 2),
                   torch.zeros((1 + k3.Engine.ATS_PAD) * 2 * M, dtype=torch.float16), torch.zeros(1, dtype=torch.int32),
                   torch.zeros(nh * hd, dtype=torch.float16), nh, nkv, hd, 0.088, splits=splits, lmin=64)
        return None if e.segs[-1].work == splits else 'work'
    except ValueError as x:
        return str(x)


one_mods = k3.Q3M_LEAN_MODULES + k3.Q3M_VER_MODULES + k3.Q3M_IQ_MODULES + tuple(k3.Q3M_RS_MODULES)
r65 = {m: atts(m, 65) for m in one_mods}
check(all(r65.values()) and all(atts(m, 64) is None for m in one_mods) and all(k3.atts_max_splits(m) == 64 for m in one_mods),
      f'own_atts refuses 65 splits and takes 64 on the one-card modules ({len(one_mods)}: lean, verify, IQ, row-scale): {r65["k3_q3m_lean"]}')
check(all(atts(m, n) is None for m in k3.Q3M_TP_MODULES for n in (1, 64, 65, 105, 128)) and all(atts(m, 129) for m in k3.Q3M_TP_MODULES)
      and all(k3.atts_max_splits(m) == 128 for m in k3.Q3M_TP_MODULES),
      'own_atts takes 1 .. 128 splits on k3_q3m_lean_tp and its trace form (64, 65, 105, 128 tried) and refuses 129')

# ---- tests/engine/tiny_q3m.py: the reference decodes it; the engine's tape on the stand-ins ----------------------------------
import tiny_q3m
from caminus.ref_q3moe import Q3MoERef
P = os.path.join(A.scratch, 'tiny-q3m.gguf'); info = tiny_q3m.write(P)
ref = Q3MoERef(P); lg = [ref.step(t) for t in (3, 17, 200)]
check(os.path.getsize(P) < 16 << 20 and all(v.shape == (info['V'],) and np.isfinite(v).all() and v.std() > 0.1 for v in lg)
      and ref.H == 16 and ref.HKV == 2 and ref.E == 8 and ref.KU == 2 and ref.FF == 768,
      f'tiny_q3m: {os.path.getsize(P) / 2 ** 20:.1f} MiB, 16 q / 2 kv heads, 8 experts of 768, top-2; the float64 reference decodes 3 tokens (finite logits)')
from caminus import ggufio
F = ggufio.GGUFFile(P)
fm = {t.name: t.type for t in F.tensors}
check(fm['token_embd.weight'] == 'Q4_K' and fm['output.weight'] == 'Q6_K' and {fm['blk.0.ffn_down_exps.weight'], fm['blk.1.ffn_down_exps.weight']} == {'Q4_K', 'Q6_K'}
      and tuple(F.T['blk.0.ffn_gate_exps.weight'].shape) == (256, 768, 8) and tuple(F.T['blk.0.ffn_down_exps.weight'].shape) == (768, 256, 8),
      'tiny_q3m: Q4_K embedding, Q6_K head, Q6_K and Q4_K expert downs; experts [K, rows, 8] innermost first')
F.close()
from caminus.engine.q3moe_one import Q3MoEOne
tapes = {}
for mod in ('k3_q3m_lean', 'k3_q3m_lean_tp'):
    e = Q3MoEOne(P, max_ctx=64, module=mod, vram_check=False, link_keepalive=False, per_wgp=5)
    tapes[mod] = [(s.op, s.ntask, s.dep, s.rows_per_task, s.N, s.K, s.work, s.pad1, s.pad2, s.dep2p1) for s in e.E.segs]
check(tapes['k3_q3m_lean'] == tapes['k3_q3m_lean_tp'] and {s[0] for s in tapes['k3_q3m_lean']} <= k3.Q3M_LEAN_OPS and len(tapes['k3_q3m_lean']) > 20,
      f'Q3MoEOne on tiny_q3m: the same {len(tapes["k3_q3m_lean"])} segments on k3_q3m_lean and k3_q3m_lean_tp, every op in the lean set')

print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
