"""kernels/q3m_prefill_iq.hip (the IQ study's prefill object: the codebook formats' quantized GEMMs, docs/dev/iq-quant-study.md)
against float64 on an arm file's real expert rows and random activations; one card.

  per (expert role, format) in the file, at the first layer that has it:
    experts   150 tokens x 8 picks over 12 experts through q3p_group's work items (partial tiles): max and mean
              |y - ref| / rms(ref) within fp32-accumulation bounds (the weights are exact in the B operand)
    dense     expert 0 as one matrix, 77 tokens (the kernel's other entry)
  the shipped object refuses a codebook format before any launch
  tools/guard.sh LOG 600 python -I tests/prefill/test_q3m_iq_ops.py GGUF
New kernel code: a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus import ggufio, own_dequant
from caminus.device import capacity
from caminus.engine.own_layout import iq_records, IQ_REC
from caminus.prefill import q3m_ops
from caminus.prefill.q3m_ops import Ops

path = sys.argv[1]
F = ggufio.GGUFFile(path)
dev = torch.device('cuda:0')
print(torch.cuda.get_device_name(0), os.path.basename(path), flush=True)
ROLES = ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')
nl = 1 + max(int(n.split('.')[1]) for n in F.T if n.startswith('blk.'))
cases = {}
for il in range(nl):
    for role in ROLES:
        t = F.T[f'blk.{il}.{role}.weight']
        if t.type in q3m_ops.IQ_KERNEL: cases.setdefault((role, t.type), t.name)
if not cases: print('no codebook-format expert tensor in this file'); print('RESULT FAIL'); sys.exit(1)
# every tensor this test puts on the card (each stays cached by the allocator), its own buffers (the largest: the
# experts' fp32 output, 1,200 x 2,048) and the HIP context
WORK_MIB, CONTEXT_MIB = 96, 256
need_mib = sum(F.T[name].n_bytes for name in cases.values()) / 2 ** 20 + WORK_MIB + CONTEXT_MIB
capacity.check([0], [need_mib], f'test_q3m_iq_ops ({need_mib:.0f} MiB projected)')
print(f'projected {need_mib:.0f} MiB; cases {sorted(cases)}', flush=True)
O = Ops(dev, q3m_ops.IQ_MODULE)
g = np.random.default_rng(5)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


def W(name):
    t = F.T[name]; K = int(t.shape[0]); R, _ = t.rows(); b = np.asarray(t.data())
    if t.type in IQ_REC: b = iq_records(b, R, K, t.type)
    return (t.type, int(t.shape[1]), K, torch.from_numpy(np.ascontiguousarray(b)).to(dev)), t


def rel(y, ref):
    r = np.sqrt((ref ** 2).mean()); e = np.abs(y - ref) / r
    return float(e.max()), float(e.mean())


n, k, E = 150, 8, 128
use = g.choice(E, 12, replace=False)
ids_np = np.stack([g.choice(use, k, replace=False) for _ in range(n)]).astype(np.int32)
ids = torch.from_numpy(ids_np.reshape(-1)).to(dev)
bound = q3m_ops.items_bound(n * k, E)
xs, yd = (torch.zeros(n * k, dtype=torch.int32, device=dev) for _ in range(2))
items = torch.zeros(bound, 4, dtype=torch.int32, device=dev); nitems = torch.zeros(1, dtype=torch.int32, device=dev)
err = torch.zeros(1, dtype=torch.int32, device=dev)
O.group(ids, n * k, k, E, xs, yd, items, nitems, err); torch.cuda.synchronize()
check(int(err.item()) == 0 and int(nitems.item()) > 0, f'group: {int(nitems.item())} items')
refused = False
for (role, fmt), name in sorted(cases.items()):
    w, t = W(name); _, R, K, _ = w
    down = 'down' in role
    x = (g.standard_normal((n * k if down else n, K)) * 0.7).astype(np.float16)
    y = torch.full((n * k, R), np.nan, dtype=torch.float32, device=dev)
    wk = (items, nitems, bound, yd, yd) if down else (items, nitems, bound, xs, yd)
    O.mm(w, torch.from_numpy(x).to(dev), K, y, R, work=wk); torch.cuda.synchronize()
    yn = y.cpu().numpy(); ref = np.zeros_like(yn, dtype=np.float64)
    deq = {int(e): own_dequant.rows(t, int(e) * R, int(e) * R + R).astype(np.float64) for e in use}
    for p in range(n * k):
        ref[p] = deq[int(ids_np.reshape(-1)[p])] @ (x[p] if down else x[p // k]).astype(np.float64)
    mx, mn = rel(yn, ref)
    check(np.isfinite(yn).all() and mx < 2e-5 and mn < 2e-6, f'mm {fmt} experts {name} ({n} tokens x {k}): max {mx:.2e}, mean {mn:.2e} of rms')
    nd = 77
    xd = (g.standard_normal((nd, K)) * 0.7).astype(np.float16)
    y = torch.full((nd, R), np.nan, dtype=torch.float32, device=dev)
    O.mm(w, torch.from_numpy(xd).to(dev), K, y, R, n=nd); torch.cuda.synchronize()
    d0 = own_dequant.rows(t, 0, R).astype(np.float64)
    mx, mn = rel(y.cpu().numpy(), xd.astype(np.float64) @ d0.T)
    check(np.isfinite(y.cpu().numpy()).all() and mx < 2e-5 and mn < 2e-6, f'mm {fmt} dense (expert 0 of {name}) n {nd}: max {mx:.2e}, mean {mn:.2e} of rms')
    if not refused:
        refused = True
        try: Ops(dev).mm(w, torch.from_numpy(xd).to(dev), K, y, R, n=nd); r = False
        except ValueError as e: r = fmt in str(e)
        check(r, f'the shipped object refuses {fmt} before any launch')
    del w, y
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
