"""Coder target KV dispatch/refusal and rotation policy, CPU launch recorders only; no model file or GPU.

  python -I tests/prefill/test_coder_kv_formats_cpu.py
"""
import contextlib
import importlib.util
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch
from caminus.engine import k3
from caminus.prefill import attn as pa
from caminus.prefill.tp import ChunkStack
from caminus.prefill.coder import Prefill

ok = True


def check(c, msg):
    global ok
    ok &= bool(c)
    print('PASS' if c else 'FAIL', msg, flush=True)


class KernelRecorder:
    def __init__(self, name, calls): self.name, self.calls = name, calls

    def __getattr__(self, fn):
        return lambda *a: self.calls.append((self.name, fn, a))


real_device, real_module = torch.cuda.device, pa.Module
torch.cuda.device = lambda dev: contextlib.nullcontext()
try:
    calls, loads = [], []

    def module(name):
        loads.append(name)
        return KernelRecorder(name, calls)

    pa.Module = module
    A = pa.Attn('cpu', module=KernelRecorder('q8', calls))
    T, M, NH, ROWS, RPB = 4, 256, 12, 4096, 2
    k, v = torch.zeros(T * M), torch.zeros(T * M)
    q, out = torch.zeros(T * NH * M), torch.zeros(T * NH * M)
    rec = torch.zeros(RPB * NH * pa.max_splits() * pa.ATTN_RS)
    err = torch.zeros(1, dtype=torch.int32)
    c8 = torch.zeros(ROWS * k3.kv_pos_bytes(M, True), dtype=torch.uint8)
    c16 = torch.zeros(ROWS * 2 * M, dtype=torch.float16)
    sel = torch.zeros(T * pa.QSA_SELW, dtype=torch.int32)
    for q8, cache in ((True, c8), (False, c16)):
        start = len(calls)
        A.kv_append(k, M, v, M, M, cache, 0, T, err, q8=q8)
        A.attend(q, cache, 0, T, 1, .0625, out, rec, err, rows_per_block=RPB, q8=q8)
        A.attend(q, cache, pa.qsa_exact(), T, 1, .0625, out, rec, err, rows_per_block=RPB, sel=sel, q8=q8)
        emitted = calls[start:]
        want = 'q8' if q8 else 'prefill_attn_f16'
        names = [fn for _, fn, _ in emitted]
        check(all(n == want for n, _, _ in emitted) and names.count('pf_kv_append') == 1
              and names.count('pf_attn_part') == 2 and names.count('pf_attn_comb') == 2
              and names.count('pf_attn_part_sel') == 2 and names.count('pf_attn_comb_sel') == 2,
              f'{want}: append, dense and SEL attention select the same explicit cache format')
        check(emitted[0][2][-2].value == ROWS and emitted[1][2][-3].value == ROWS,
              f'{want}: append and attention derive physical row count from matching dtype and row bytes')
        if q8: check(not loads, 'default Q8 loads no F16 module')
    A.kv_append(k, M, v, M, M, c16, 4, T, err, q8=False)
    A.kv_append(k, M, v, M, M, c8, 4, T, err)   # indexer remains Q8 even after target F16 launches
    check(loads == ['prefill_attn_f16'] and calls[-1][0] == 'q8', 'F16 module loads once; later default/indexer append stays Q8')
    preloaded = pa.Attn('cpu', module=KernelRecorder('q8', calls))
    n = len(calls)
    preloaded.prepare_f16()
    preloaded.prepare_f16()
    check(loads == ['prefill_attn_f16', 'prefill_attn_f16'] and len(calls) == n,
          'F16 preparation loads once before capture without launching kernels')

    def refused(fn, msg, types=(ValueError,)):
        n = len(calls)
        try: fn()
        except types: check(len(calls) == n, msg + ': refused without launch')
        else: check(False, msg + ': accepted unsafe input')

    for q8, wrong in ((False, c8), (True, c16)):
        refused(lambda: A.kv_append(k, M, v, M, M, wrong, 0, T, err, q8=q8), 'append cache dtype mismatch')
        refused(lambda: A.attend(q, wrong, 0, T, 1, .0625, out, rec, err, rows_per_block=RPB, q8=q8), 'attention cache dtype mismatch')
    refused(lambda: A.kv_append(k, M, v, M, M, c16[:-1], 0, T, err, q8=False), 'partial F16 row')
    refused(lambda: A.kv_append(k, M, v, M, M, c16, ROWS - T + 1, T, err, q8=False), 'F16 write past last row', (k3.ContextBoundsError,))
    refused(lambda: A.attend(q, c16, ROWS - T + 1, T, 1, .0625, out, rec, err, rows_per_block=RPB, sel=sel, q8=False),
            'F16 SEL read past last row', (k3.ContextBoundsError,))
    refused(lambda: A.attend(q, c16, pa.qsa_exact() - 1, T, 1, .0625, out, rec, err, rows_per_block=RPB, q8=False),
            'F16 dense read past QSA boundary', (k3.ContextBoundsError,))
    bad_align = torch.zeros(c16.numel() + 1, dtype=torch.float16)[1:]
    refused(lambda: A.attend(q, bad_align, 0, T, 1, .0625, out, rec, err, rows_per_block=RPB, q8=False),
            'F16 vector-load cache alignment')
    refused(lambda: A.kv_append(k, M, v, M, M, c8, 0, T, err, q8='f16'), 'non-boolean cache policy')
    other = pa.Attn('cpu', name='prefill_attn_q35', module=KernelRecorder('other', calls))
    refused(lambda: other.kv_append(k, M, v, M, M, c16, 0, T, err, q8=False), 'unsupported F16 object geometry')
    refused(other.prepare_f16, 'unsupported F16 module preparation')
finally:
    torch.cuda.device, pa.Module = real_device, real_module


class CoreRecorder:
    """Record the real ChunkStack core's policy, without launching or fabricating numerical outputs."""
    def __init__(self): self.calls = []

    def __getattr__(self, fn):
        return lambda *a, **kw: self.calls.append((fn, a, kw))


for q8 in (True, False):
    for qsa, p0 in ((False, 0), (True, 0), (True, 2048)):
        n, D, hd, nh = 8, 16, 256, 12
        A, Gl = CoreRecorder(), CoreRecorder()
        AW = SimpleNamespace(ko=torch.zeros(n * hd), qo=torch.zeros(n * nh * hd), ikr=torch.zeros(n * 128),
                             iq=torch.zeros(n * pa.QSA_H * pa.QSA_D), iqo=torch.zeros(n * pa.QSA_H * pa.QSA_D),
                             err=object(), rec=object(), score=object(), sel=torch.zeros(n * pa.QSA_SELW, dtype=torch.int32),
                             srows=4, prow=1024)
        S = ChunkStack.__new__(ChunkStack)
        S.A, S.AW, S.Gl = [A], [AW], [Gl]
        S.m = SimpleNamespace(q8=q8, nrot=64, theta_scale=.5)
        S.D, S.nh, S.hd, S.p0, S.qsa, S.rpb, S._pool_from = D, nh, hd, p0, qsa, 2, 0
        S._mk = lambda *a: None
        L = dict(kn=object(), qn=object(), kv=torch.zeros((p0 + n + 1) * k3.kv_pos_bytes(hd, q8), dtype=torch.uint8),
                 raw=object(), pool=object(), ikn=object(), iqn=object(),
                 ikw={'w': object()}, iqw={'w': object()})
        S._core(0, L, 0, n, torch.zeros(n * D), torch.zeros(n * 2 * nh * hd), torch.zeros(n * hd),
                torch.zeros(n * hd), torch.zeros(n * nh * hd))
        had = [c for c in A.calls if c[0] == 'hadamard']
        app = [c for c in A.calls if c[0] == 'kv_append']
        att = [c for c in A.calls if c[0] == 'attend']
        target = [c for c in app if isinstance(c[1][5], torch.Tensor) and c[1][5].data_ptr() == L['kv'].data_ptr()]
        indexer = [c for c in app if c[1][5] is L['raw']]
        check(len(had) == (4 if q8 else 0) and len(target) == 1 and target[0][2]['q8'] == q8
              and all(c[2]['q8'] == q8 for c in att) and A.calls[-1][0] == 'gate_mul',
              f'core q8={q8}, qsa={qsa}, pos={p0}: target format and rotations agree with decode; gate always runs')
        check(target[0][1][5].dtype == (torch.uint8 if q8 else torch.float16)
              and target[0][1][5].numel() * target[0][1][5].element_size() == L['kv'].numel(),
              f'core q8={q8}, qsa={qsa}, pos={p0}: raw engine/host-mapped storage has a matching zero-copy typed view')
        check(len(indexer) == int(qsa) and all(c[2].get('q8', True) for c in indexer),
              f'core q8={q8}, qsa={qsa}, pos={p0}: indexer wire format remains Q8')
        if p0 == 2048:
            check([(c[1][2], c[1][3], c[2].get('sel') is not None) for c in att] == [(2048, 3, False), (2051, 5, True)],
                  f'core q8={q8}: a chunk crossing QSA_EXACT keeps dense/SEL boundaries')

pf = Prefill.__new__(Prefill)
pf.S = SimpleNamespace(Gl=[SimpleNamespace(wmma_out='f32', router_mode='f32')], moe='own', q6k='own')
pf.m = SimpleNamespace(q8=True)
old = dict(wmma_out='f32', router_logits='f32', moe='own', q6k='own')
check(pf.numerics() == old, 'default Q8 prefill numerical key is unchanged')
pf.m.q8 = False
check(pf.numerics() == dict(old, target_kv='f16', kv_prefill=1), 'F16 prefill key refuses parks made by the old Q8-only prefill path')

# The full-model GPU harness must hash every byte of the selected cache format and refuse mismatched old baselines.
spec = importlib.util.spec_from_file_location('coder_prefill_s6', os.path.join(os.path.dirname(__file__), 'test_prefill_s6.py'))
s6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s6)
for q8 in (True, False):
    pb = k3.kv_pos_bytes(256, q8)
    cache = torch.zeros(2 * pb, dtype=torch.uint8)
    rig = s6.Rig.__new__(s6.Rig)
    rig.m = SimpleNamespace(hd=256, q8=q8, state={(0, 3, 'kv'): cache})
    rig.pf = SimpleNamespace(S=SimpleNamespace(feat_rows=0, feat_base=0))
    before = rig.states(1)
    cache[pb - 1] = 1
    last = rig.states(1)
    cache[pb] = 1
    check(before != last and last == rig.states(1), f'full-model harness q8={q8}: hashes the full KV row and excludes the next')
for stage in s6.F16_STAGES: s6.check_kv_stages('f16', [stage])
for stage in ('g4a', 'g4d', 'long', 'speed', 'speed7', 'tstar', 'probes', 'vram7'):
    s6.check_kv_stages('q8', [stage])
    try: s6.check_kv_stages('f16', [stage])
    except ValueError: pass
    else: check(False, f'F16 accepted recorded Q8 baseline stage {stage}')
check(True, 'full-model harness refuses F16 stages tied to recorded Q8 baselines before device allocation')

print('RESULT', 'PASS' if ok else 'FAIL', flush=True)
sys.exit(0 if ok else 1)
