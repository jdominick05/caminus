"""The two-card verify tape's host side without a card (caminus.engine.q35_dense_verify_tp.Dense35VerifyTP on Dense35TP with
spec_T): the tiny synthetic models with two-card shapes (tests/engine/tiny_q35.py tp=True), on stand-ins for the torch.cuda
and HIP calls (every tensor in host memory; nothing is launched):
  - the module: k3.Q35_TPV_OF maps each two-card module to its verify build (k3_q35_{swift,tcap}_tp_ver); Dense35VerifyTP
    takes it by default and refuses any other; T outside 1 .. 4, T above the engine's spec_T, and an engine without spec_T
    are refused
  - the tapes: every op one the module dispatches (k3.Q35S_OPS + OP_XCHG), 2 exchanges per block on each card with indices
    0 .. 2 nl - 1 once each, the card's own number and N = T * D (the payload's floats), every segment's team = T, the guard's
    T = the engine's spec_T, every dependency pointing back
  - the buffers: T times the decode tape's per-token sizes, the head's shard per card, the split attention records over
    max_ctx + spec_T - 1 rows, the exchange slots spec_T * D floats; need_mib agrees with the buffers
  - spec_T = 1 keeps the served projection: the engine's bytes, KV rows, records, slots and recurrent MiB are the spec_T
    default's; commit_max is 1 on it, and on a spec_T engine 0 after a reset
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/engine/test_q35_verify_tp_cpu.py --scratch DIR"""
import argparse, contextlib, os, sys, types
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
_zeros, _to = torch.zeros, torch.Tensor.to
torch.zeros = lambda *a, **k: _zeros(*a, **{x: v for x, v in k.items() if x not in ('device', 'pin_memory')})
torch.Tensor.to = lambda self, *a, **k: self if a and str(a[0]).startswith('cuda') else _to(self, *a, **k)
torch.Tensor.cuda = lambda self, *a, **k: self
torch.cuda.set_device = lambda d: None
torch.cuda.device = lambda d: contextlib.nullcontext()
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda d: types.SimpleNamespace(multi_processor_count=42, total_memory=20464 << 20)
torch.cuda.Event = lambda *a, **k: types.SimpleNamespace()

import tiny_q35
from caminus import ggufio
from caminus.engine import k3, q35_dense_tp
from caminus.engine.q35_dense_tp import Dense35TP
from caminus.engine.q35_dense_verify import MAX_T
from caminus.engine.q35_dense_verify_tp import Dense35VerifyTP
from caminus.engine.q35_dense import attn_part_elems


class FakeHip:
    def __getattr__(self, n): return lambda *a, **k: 0


class FakeSlots:
    """the receive slots' addresses only (distinct per card and slot)"""
    n_made = 0

    def __init__(self, dev, n, N):
        FakeSlots.n_made += 1; self.base = (FakeSlots.n_made << 40); self.stride = (16 + N) * 4; self.n, self.N = n, N

    def slot(self, i):
        assert 0 <= i < self.n; return self.base + i * self.stride


q35_dense_tp._hip = lambda: FakeHip()
q35_dense_tp.Slots = FakeSlots

P_S = os.path.join(A.scratch, 'tiny-tp-swift.gguf'); iS = tiny_q35.write(P_S, tp=True)
P_T = os.path.join(A.scratch, 'tiny-tp-tcap.gguf'); iT = tiny_q35.write(P_T, variant='tcap', tp=True)

# ---- the module mapping and the refusals --------------------------------------------------------------------------------
expected = {'k3_q35_swift_tp': 'k3_q35_swift_tp_ver', 'k3_q35_tcap_tp': 'k3_q35_tcap_tp_ver'}
expected.update({k3.Q35_KV_Q8_OF[m]: k3.Q35_KV_Q8_OF[v] for m, v in tuple(expected.items())})
good = k3.Q35_TPV_OF == expected
good &= all(m in k3.Q35V_MODULES and m in k3.Q35_SPLIT_MODULES
            and (m in k3.ATTN_PACK_MODULES) == (m not in k3.Q35_KV_Q8_MODULES) for m in k3.Q35TPV_MODULES)
good &= all(k3.MODULE_CAPS[m] for m in k3.Q35TPV_MODULES)
check(good, 'k3.Q35_TPV_OF maps each cache format to its T-token verify build; packed F16 and unpacked Q8 have matching caps')
eng = Dense35TP(P_S, devices=(0, 1), max_ctx=64, vram_check=False, spec_T=MAX_T)
try: Dense35VerifyTP(eng, MAX_T, module='k3_q35_swift_ver'); refused = False
except ValueError: refused = True
try: Dense35VerifyTP(eng, 0); refused0 = False
except ValueError: refused0 = True
try: Dense35VerifyTP(eng, MAX_T + 1); refusedT = False
except ValueError: refusedT = True
eng2 = Dense35TP(P_S, devices=(0, 1), max_ctx=64, vram_check=False, spec_T=2)
try: Dense35VerifyTP(eng2, 3); refusedS = False
except ValueError: refusedS = True
class Bare: pass
try: Dense35VerifyTP(Bare(), 1); refusedE = False
except ValueError: refusedE = True
check(refused and refused0 and refusedT and refusedS and refusedE,
      'a one-card verify module, T outside 1 .. 4, T above spec_T and an engine without spec_T are all refused')

# ---- the tapes ----------------------------------------------------------------------------------------------------------
for path, info, want_mod in ((P_S, iS, 'k3_q35_swift_tp_ver'), (P_T, iT, 'k3_q35_tcap_tp_ver')):
    m = Dense35TP(path, devices=(0, 1), max_ctx=64, vram_check=False, spec_T=MAX_T)
    nl = info['blocks']; D = m.D
    check(m.spec_T == MAX_T and m.commit_max() == 0, f'{os.path.basename(path)}: spec_T {m.spec_T}, commit_max 0 after a reset')
    check(all(m.slots[c].N == MAX_T * D and m.slots[c].stride == (16 + MAX_T * D) * 4 for c in (0, 1)),
          f'{os.path.basename(path)}: the exchange slots hold spec_T * D = {MAX_T * D} floats')
    for T in range(1, MAX_T + 1):
        ver = Dense35VerifyTP(m, T, vram_check=False)
        good = ver.module == want_mod and ver.T == T
        xs_all = []
        for c in (0, 1):
            E = ver.E[c]; ops = [sg.op for sg in E.segs]
            good &= E.T == T and E.guard_T == MAX_T and all(o in k3.Q35S_OPS or o == k3.OP_XCHG for o in ops)
            good &= not E.unsupported() and E.refuse is None
            good &= all(sg.team == (T if T > 1 else -1) for sg in E.segs)   # (build(): T > 1 carries T in every segment's team)
            good &= all(sg.dep < i and sg.dep2p1 - 1 < i for i, sg in enumerate(E.segs))
            xs = [sg for sg in E.segs if sg.op == k3.OP_XCHG]
            good &= all(sg.work == c and sg.N == T * D for sg in xs) and len(xs) == 2 * nl
            xs_all.append(sorted(sg.K for sg in xs))
            s = ver.buf[c]
            fc = m.f0 if c == 0 else m.ffn - m.f0
            nlog = m.vh if c == 0 else m.vocab - m.vh
            good &= s['x'].numel() == T * D and s['blkp'].numel() == T * D and s['ffpp'].numel() == T * D
            good &= s['g'].numel() == T * fc and s['logits'].numel() == T * nlog
            good &= s['part'].numel() == attn_part_elems(T, m.nh // 2, m.max_ctx + MAX_T - 1)
            good &= ver.logits2[c].shape == (T, nlog) and ver.dstage[c].numel() == 32
        check(good and xs_all[0] == xs_all[1] == list(range(2 * nl)),
              f'{os.path.basename(path)} T {T}: both tapes run only the module {want_mod}\'s ops, every segment\'s team is T, '
              f'2 exchanges per block (indices 0 .. {2 * nl - 1} once, the card\'s own number, N = T * D = {T * D}), deps point back')
        mib = Dense35VerifyTP.need_mib(T, m)
        sz = {torch.float32: 4, torch.float16: 2, torch.int32: 4}
        exact = [sum(n * sz[dt] for n, dt in sp.values()) / 2 ** 20 + 2.0 for sp in Dense35VerifyTP._spec(m, T)]
        check(all(abs(a - b) < 1e-9 for a, b in zip(mib, exact)) and ver.mib == mib,
              f'{os.path.basename(path)} T {T}: need_mib per card agrees with the tape\'s buffers ({mib[0]:.2f} / {mib[1]:.2f} MiB)')
    check(Dense35VerifyTP.need_mib(MAX_T, max_ctx=64) > 0, f'{os.path.basename(path)}: need_mib\'s upper bound without an engine')

# ---- spec_T = 1 keeps the served projection -------------------------------------------------------------------------------
p_def, p_1 = Dense35TP.projection_mib(P_S, 64), Dense35TP.projection_mib(P_S, 64, spec_T=1)
p_4 = Dense35TP.projection_mib(P_S, 64, spec_T=MAX_T)
check(p_def == p_1 and p_4[0] > p_1[0] and Dense35TP.recurrent_mib(P_S) == Dense35TP.recurrent_mib(P_S, 1)
      and Dense35TP.recurrent_mib(P_S, MAX_T) > Dense35TP.recurrent_mib(P_S, 1),
      'spec_T defaults to 1: the served projection and recurrent MiB are unchanged; spec_T 4 grows them (saved inputs, KV rows, slots, records)')
m1 = Dense35TP(P_S, devices=(0, 1), max_ctx=64, vram_check=False)
check(m1.spec_T == 1 and m1.commit_max() == 1 and m1.slots[0].N == m1.D and m1.E[0].guard_T is None,
      'a spec_T 1 engine: commit_max 1, D-float slots, no verify guard_T on the decode tape (as served)')

# ---- the drafted session's two-card seams ------------------------------------------------------------------------------
from caminus.cache.prompt import Layout, PromptCache
from caminus.engine.q35_draft_tp import DraftTarget35TP, head_mib, pass_h, verify_shards
from caminus.engine.q35_mtp import Mtp35
from caminus.decode.mtp35 import Mtp35Drafter
from caminus.decode.dflash35 import Dflash35Drafter
from caminus.engine.q35_dflash import Dflash35
from caminus.prefill.q35own_tp import Q35OwnPrefillTP, buffer_need_tp_mib, tp_shapes

for card in (0, 1):
    m = Dense35TP(P_T, max_ctx=64, vram_check=False, spec_T=4, feature_layers=(1, 3), feature_card=card)
    ver = Dense35VerifyTP(m, 4, vram_check=False, feature_layers=(1, 3))
    good = tuple(m.features.shape) == (2, 1, m.D) and tuple(ver.features.shape) == (2, 4, m.D)
    for src in (m, ver):
        for c in (0, 1):
            copies = [i for i, sg in enumerate(src.E[c].segs)
                      if sg.op == k3.OP_OWN_ADD_RMS16 and 'feat' in src.buf[c]
                      and src.buf[c]['feat'].data_ptr() <= (sg.y or 0) < src.buf[c]['feat'].data_ptr() + src.buf[c]['feat'].numel() * 4]
            good &= len(copies) == (2 if c == card else 0)
            # The next writer of the residual must depend on the copy, which reads that residual as its addend.
            for i in copies:
                sg = src.E[c].segs[i]
                writer = next(s for s in src.E[c].segs[i + 1:] if s.op == k3.OP_OWN_ADD_RMS16 and s.y == sg.a)
                good &= writer.dep2p1 == i + 1
    base = Dense35VerifyTP.need_mib(4, m)
    diff = [a - b for a, b in zip(ver.mib, base)]
    good &= abs(diff[card] - Dense35TP.feature_mib(m.D, 4, 2)) < 1e-9 and diff[1 - card] == 0
    check(good, f'feature card {card}: decode and verify export the requested residuals only there, before the next x writer; bytes projected')
    # The KV tail allocation is longer than the legal context. Its physical row stride is used by checkpoint/park files.
    L = Layout(m, T=4)
    pb = 2 * (m.nkv // 2) * m.hd * 2
    check(L.pb == [pb, pb] and m.kv_rows == 67,
          f'feature card {card}: checkpoint KV rows use physical stride {pb} bytes at max_ctx 64, spec_T 4 (not bytes / 64)')
    # Exercise the actual checkpoint save/load offsets rather than only comparing the computed stride.
    pc = PromptCache.__new__(PromptCache)
    pc.m, pc.layout, pc.T, pc.df, pc.draft = m, L, 4, None, None
    pc._sync = lambda: None
    for k, t in m.state.items():
        if k[-1] == 'kv':
            raw = t.view(torch.uint8).view(m.kv_rows, pb)
            raw[:] = torch.arange(m.kv_rows, dtype=torch.uint8)[:, None]
        else: t.fill_(7)
    bufs = [torch.zeros(nb, dtype=torch.uint8) for nb in L.nbytes]
    pc._save(bufs, 32, 4)
    for t in m.state.values(): t.zero_()
    pc._load(bufs, types.SimpleNamespace(pos=32, commit=4, fresh=False, last_T=4, pool_ok=0, df=None), 0)
    good = m.last_T == 4
    for k, t in m.state.items():
        if k[-1] == 'kv':
            raw = t.view(torch.uint8).view(m.kv_rows, pb)
            good &= torch.equal(raw[28:32, 0], torch.tensor([28, 29, 30, 31], dtype=torch.uint8))
            good &= bool((raw[:28] == 0).all()) and bool((raw[32:] == 0).all())
        else: good &= bool((t == 7).all())
    check(good, f'feature card {card}: actual checkpoint save/load restores precisely the four KV tail rows and recurrent states')
    shards = verify_shards(ver, m.vh)
    check(shards is ver.logits2 and pass_h(ver) is ver.buf[0]['h'] and pass_h(m) is m.buf[0]['h'],
          f'feature card {card}: sampled acceptance keeps the two device shards; MTP reads the replicated h on card 0')

for path in (P_S, P_T):
    m = Dense35TP(path, max_ctx=64, vram_check=False, spec_T=4)
    for card in (0, 1):
        target = DraftTarget35TP(m, card, check=False)
        check(target.head['rows'] == m.vocab and target.head['K'] == m.D and target.emb_dev is m.emb_dev[card]
              and target.head['w'].numel() == m.F.T['output.weight'].n_bytes and target.mib == head_mib(path)
              and target.dev == m.devs[card] and target.per_wgp == m.per_wgp_c[card],
              f'{os.path.basename(path)}: drafter card {card} has the complete head, shares its embedding, and projects every head byte')

P_M = os.path.join(A.scratch, 'tiny-tp-mtp.gguf'); tiny_q35.write(P_M, variant='tcap', tp=True, mtp=True)
m = Dense35TP(P_M, max_ctx=64, vram_check=False, spec_T=4)
target = DraftTarget35TP(m, 0, check=False)
mtp = Mtp35(target, vram_check=False)
dr = Mtp35Drafter(mtp, vram_check=False)
check(dr.df.tgt is m and mtp.eng is target and mtp.eng.head is target.head,
      'MTP cache identity is the TP trunk while its tapes use the one-card full-head view')
d = Dflash35.__new__(Dflash35); d.eng = target
check(Dflash35Drafter(d).df.tgt is m, 'DFlash cache identity is the TP trunk, not the drafter compute view')
from caminus import arch
tp = arch.ARCHS['qwen35-own-tp']
_, _, _, need_off = tp.auto_need(dict(model=P_M, draft='off'), 4)
_, _, _, need_mtp = tp.auto_need(dict(model=P_M, draft='mtp'), 4)
check(all(a > b for a, b in zip(need_mtp(64), need_off(64))) and tp.default_draft(dict(model=P_M)) == 'off',
      'two-card MTP projection includes verify on both cards and head/block/window on card 0; default stays off pending served A/B')

# The prefill hooks shift h's token by one across chunks and send DFlash the selected card's launchers.
pf = Q35OwnPrefillTP.__new__(Q35OwnPrefillTP)
pf.m = types.SimpleNamespace(D=2); pf.export_h = True
pf.B = [dict(hout=torch.arange(8, dtype=torch.float32)), {}]
pf._chunk = lambda *args: None
calls = []
pf._draft = types.SimpleNamespace(kind='mtp', fill_chunk=lambda h, ts, p: calls.append((h.tolist(), ts, p)))
pf._run_chunk([10, 11, 12, 13, 14], None, None, 1, 5, None, None)
check(calls == [([0., 1., 2., 3., 4., 5., 6., 7.], [12, 13, 14, 14], 2)],
      'TP prefill feeds MTP h rows with the next-position tokens, including the clamped prompt-end row')
pf.export_h = False; pf.draft_ops = object(); calls = []
pf._draft = types.SimpleNamespace(kind='dflash35', inject_chunk=lambda src, a, n: calls.append((src, a, n)))
pf._run_chunk([10, 11, 12, 13, 14], None, None, 1, 5, None, None)
check(calls == [(pf.draft_ops, 1, 4)], 'TP prefill sends DFlash the selected card launchers and exact chunk interval')

sh = tp_shapes(P_M)
base = buffer_need_tp_mib(sh, 16)
mh = buffer_need_tp_mib(sh, 16, export_h=True)
df = buffer_need_tp_mib(sh, 16, nfeat=2, feature_card=1)
check(abs(mh[0] - base[0] - 16 * m.D * 6 / 2 ** 20) < 1e-9 and mh[1] == base[1]
      and df[0] == base[0] and abs(df[1] - base[1] - 2 * 16 * m.D * 4 / 2 ** 20) < 1e-9,
      'TP prefill projects MTP h on card 0 only and DFlash feature rows on its chosen card only')

# A header-only DFlash fixture exercises placement and window projections without reading a real drafter's weights.
import struct
P_D = os.path.join(A.scratch, 'dflash-header.gguf')
dm = {'general.architecture': 'dflash'}
dm.update({f'dflash.{k}': v for k, v in dict(embedding_length=256, feed_forward_length=512, block_count=2,
    selector_rank=32, selector_top_k=4, conv_kernel_size=4, conv_group_size=32, target_layers=[1, 3]).items()})
dm.update({f'dflash.attention.{k}': v for k, v in dict(head_count=6, head_count_kv=1, key_length=256, sliding_window=2048).items()})
with open(P_D, 'wb') as f:
    f.write(struct.pack('<4sIQQ', b'GGUF', 3, 0, len(dm)) + b''.join(tiny_q35._s(k) + tiny_q35._val(v) for k, v in dm.items()))
cards = [dict(line_mib=10000, label='XT', display=False), dict(line_mib=14000, label='XTX', display=True)]
opt = dict(model=P_T, drafter=P_D, draft='dflash')
spec, dc, layers, dfl, extra = tp._resources(opt, cards)
_, _, slots_df, need_df = tp.auto_need(opt, 4, cards)
cards_rev = cards[::-1]
_, dc_rev, _, _, _ = tp._resources(opt, cards_rev)
check(spec == 4 and dc == 1 and dc_rev == 0 and layers == [1, 3]
      and extra(64)[dc] > extra(64)[1 - dc] and slots_df > 0 and all(v > 0 for v in need_df(64)),
      'DFlash placement follows projected card headroom in either order; head, features, verify and checkpoint windows are counted')

print('RESULT', 'PASS' if ok else 'FAIL', flush=True)
sys.exit(0 if ok else 1)
