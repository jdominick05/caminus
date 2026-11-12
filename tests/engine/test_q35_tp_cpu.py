"""The dense qwen35 two-card engine's host side without a card (caminus.engine.q35_dense_tp): the split and the two tapes it
builds for the tiny synthetic models with two-card shapes (tests/engine/tiny_q35.py tp=True), on stand-ins for the torch.cuda
and HIP calls (every tensor in host memory; nothing is launched).
  - the split plan: every row-split matrix's rows and every column-split matrix's columns are a partition of the whole
    (each index on exactly one card), and the cuts fall on block boundaries; card c's value heads are the ones that read its
    key heads, in the order the device op reads them (local value head j reads local key head j % kc: global key head
    ks + j % kc = h % nk); attn_qkv's rows are its key heads' q and k, then its value heads' v; ssm_out's columns follow
    the same value-head order
  - cut(): the raw bytes it keeps decode (caminus.own_dequant) to exactly the full matrix's rows / columns, for every format
    of both tiny models, and a cut inside a block is refused
  - the tapes: the two-card module of the file's formats (k3_q35_swift_tp / k3_q35_tcap_tp), every op one that module
    dispatches (k3.Q35S_OPS + OP_XCHG), 2 exchanges per block on each card with indices 0 .. 2 nl - 1 once each and the
    card's own number, every dependency pointing back; each card's matrices have its share's shapes
  - the host op list against the device: kernels/k3_q35_ops.h and kernels/k3_q35t_ops.h dispatch OP_XCHG only under K3_TP;
    the one-card modules refuse OP_XCHG at build
  - projection: per card, the KV row is its kv heads' share; the two cards' weights add up to more than the file's (the
    replicated vectors and embedding) and less than twice it
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/engine/test_q35_tp_cpu.py --scratch DIR"""
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
from caminus import ggufio, own_dequant
from caminus.engine import k3, q35_dense_tp
from caminus.engine.q35_dense_tp import Dense35TP, split_plan, cut, gdn_split


class FakeHip:
    def __getattr__(self, n): return lambda *a, **k: 0


class FakeSlots:
    """the receive slots' addresses only (distinct per card and slot)"""
    n_made = 0

    def __init__(self, dev, n, N):
        FakeSlots.n_made += 1; self.base = (FakeSlots.n_made << 40); self.stride = (16 + N) * 4; self.n = n

    def slot(self, i):
        assert 0 <= i < self.n; return self.base + i * self.stride


q35_dense_tp._hip = lambda: FakeHip()
q35_dense_tp.Slots = FakeSlots


def partition(ranges, n):
    """each index 0 .. n - 1 in exactly one card's ranges"""
    hit = np.zeros(n, np.int64)
    for rr in ranges:
        for a, b in rr: hit[a:b] += 1
    return bool((hit == 1).all())


# ---- the split plan on the real models' shapes and the tiny ones --------------------------------------------------------
for tag, (NK, NV, S, nh, nkv, hd, ffn, vocab) in {'Qwen3.8-27B': (16, 48, 128, 24, 4, 256, 17408, 248320),
                                                    'tiny': (4, 8, 128, 4, 2, 128, 512, 256)}.items():
    f0, vh = q35_dense_tp.ffn_cut(ffn, 0.53), q35_dense_tp.vocab_cut(vocab, 0.5)
    P = split_plan(NK, NV, S, nh, nkv, hd, ffn, f0, vocab, vh)
    KD, CH = NK * S, 2 * NK * S + NV * S
    parts = (partition(P['qkv'], CH) and partition(P['gate'], NV * S) and partition(P['ab'], NV) and partition(P['ssm_out'], NV * S)
             and partition(P['q'], nh * 2 * hd) and partition(P['kv'], nkv * hd) and partition(P['attn_output'], nh * hd)
             and partition(P['ffn'], ffn) and partition(P['head'], vocab))
    check(parts, f'{tag}: every split matrix\'s rows / columns are a partition over the two cards')
    blocks = all(a % 256 == 0 and b % 256 == 0 for k in ('ssm_out', 'attn_output', 'ffn') for rr in P[k] for a, b in rr)
    check(blocks and f0 % 256 == 0 and vh % 128 == 0, f'{tag}: the column cuts (ssm_out, attn_output, ffn_down) fall on 256-value blocks; '
          f'ffn {f0} / {ffn - f0}, head {vh} / {vocab - vh}')
    good = True
    for c, (ks, kc, VH) in enumerate(P['gdn']):
        good &= len(VH) == NV // 2 and all(ks + j % kc == h % NK for j, h in enumerate(VH))
        q_ch = np.concatenate([np.arange(a, b) for a, b in P['qkv'][c]])
        want = np.concatenate([np.arange(ks * S, (ks + kc) * S), KD + np.arange(ks * S, (ks + kc) * S)] +
                              [2 * KD + np.arange(h * S, (h + 1) * S) for h in VH])
        good &= np.array_equal(q_ch, want) and np.array_equal(P['ch'][c], want)
        so = np.concatenate([np.arange(a, b) for a, b in P['ssm_out'][c]])
        good &= np.array_equal(so, np.concatenate([np.arange(h * S, (h + 1) * S) for h in VH]))
    check(good, f'{tag}: each card\'s value heads read its key heads in the device\'s order (local j -> key ks + j % kc = h % nk); '
          f'qkv rows and ssm_out columns follow that order')

# ---- cut(): the bytes kept decode to the full matrix's rows / columns --------------------------------------------------
P_S = os.path.join(A.scratch, 'tiny-tp-swift.gguf'); iS = tiny_q35.write(P_S, tp=True)
P_T = os.path.join(A.scratch, 'tiny-tp-tcap.gguf'); iT = tiny_q35.write(P_T, variant='tcap', tp=True)
seen, bad = set(), []
for path in (P_S, P_T):
    F = ggufio.GGUFFile(path)
    for t in F.tensors:
        if len(t.shape) != 2 or t.type not in own_dequant.DECODERS or t.type in seen: continue
        seen.add(t.type)
        K = int(t.shape[0]); R, rb = t.rows(); _, bs, ts = ggufio.TYPES[t.type_id]
        bs_, bb, dec = own_dequant.DECODERS[t.type]
        full = dec(np.asarray(t.data()).copy()).reshape(R, K)
        u = max(bs, 256) if K >= 512 else bs
        rr, cc = [(0, 1), (R // 2, R)], [(0, u), (K - u, K)] if K >= 2 * u else [(0, K)]
        x, Rl, Kl = cut(np.asarray(t.data()).reshape(R, rb), rr, cc, bs, ts)
        got = dec(x.reshape(-1)).reshape(Rl, Kl)
        want = np.concatenate([full[a:b] for a, b in rr], axis=0); want = np.concatenate([want[:, a:b] for a, b in cc], axis=1)
        if not np.array_equal(got, want): bad.append(t.type)
    F.close()
check(not bad and len(seen) >= 8, f'cut() keeps exactly the rows / columns asked for, for {sorted(seen)}' + (f' (wrong: {bad})' if bad else ''))
try:
    cut(np.zeros((2, 144), np.uint8), None, [(0, 128)], 256, 144); refused = False
except ValueError:
    refused = True
check(refused, 'a column cut inside a block is refused')

# ---- the two tapes ----------------------------------------------------------------------------------------------------
for path, info, want_mod in ((P_S, iS, 'k3_q35_swift_tp'), (P_T, iT, 'k3_q35_tcap_tp')):
    m = Dense35TP(path, devices=(0, 1), max_ctx=64, vram_check=False, ffn_share=0.5)
    nl = info['blocks']
    check(m.module == want_mod, f'{os.path.basename(path)}: the two-card module of its formats ({m.module})')
    good, xi = True, []
    for c in (0, 1):
        E = m.E[c]; ops = [sg.op for sg in E.segs]
        good &= all(o in k3.Q35S_OPS or o == k3.OP_XCHG for o in ops) and not E.unsupported() and E.refuse is None
        good &= all(sg.dep < i and sg.dep2p1 - 1 < i for i, sg in enumerate(E.segs))
        xs = [sg for sg in E.segs if sg.op == k3.OP_XCHG]
        xi.append(sorted(sg.K for sg in xs)); good &= all(sg.work == c for sg in xs) and len(xs) == 2 * nl
    check(good and xi[0] == xi[1] == list(range(2 * nl)),
          f'{os.path.basename(path)}: both tapes run only the module\'s ops, every dependency points back, 2 exchanges per block '
          f'(indices 0 .. {2 * nl - 1} once, the card\'s own number)')
    sh = True
    for il, dd in enumerate(m.L):
        for c in (0, 1):
            d = dd[c]; fc = m.f0 if c == 0 else m.ffn - m.f0
            sh &= (d['fg']['rows'], d['fu']['rows'], d['fd']['K']) == (fc, fc, fc)
            if d['kind'] == 'gdn':
                sh &= (d['qkv']['rows'], d['z']['rows'], d['al']['rows'], d['so']['K']) == (d['CH'], info['NV'] // 2 * 128, info['NV'] // 2, info['NV'] // 2 * 128)
                sh &= d['NK'] == info['NK'] // 2 and d['conv_w'].shape == (d['CH'], 4) and d['dt'].numel() == info['NV'] // 2
            else:
                HD = tiny_q35.HD   # (the attention head width: 256, the split pair's)
                sh &= (d['q']['rows'], d['k']['rows'], d['o']['K']) == (info['H'] // 2 * 2 * HD, info['HKV'] // 2 * HD, info['H'] // 2 * HD)
    sh &= m.head[0]['rows'] == m.vh and m.head[1]['rows'] == m.vocab - m.vh
    check(sh, f'{os.path.basename(path)}: each card\'s matrices have its share\'s shapes (ffn {m.f0} / {m.ffn - m.f0}, head {m.vh} / {m.vocab - m.vh})')
    from caminus.prefill.q35own_tp import local_shapes, tp_shapes, buffer_need_tp_mib
    ls, hs = [local_shapes(m, c) for c in (0, 1)], tp_shapes(path, 0.5)
    good = all(vars(a) == vars(b) for a, b in zip(ls, hs)) and [x.NV for x in ls] == [info['NV'] // 2] * 2 and \
        [x.ffn for x in ls] == [m.f0, m.ffn - m.f0] and [x.nkv for x in ls] == [info['HKV'] // 2] * 2
    nb = buffer_need_tp_mib(hs, 128)
    check(good and all(v > 0 for v in nb), f'{os.path.basename(path)}: the two-card prefill\'s per-card shapes from the engine and from the '
          f'header agree ({[vars(x) for x in ls][0]}); its buffers at chunk 128: {nb[0]:.2f} / {nb[1]:.2f} MiB')
    m.reset()
    check((m.seq, m.fresh, m.pos_host) == (1, True, 0) and {k[0] for k in m.state} == {0, 1}
          and sorted({k[2] for k in m.state}) == ['conv', 'convx', 'gdn', 'kv'], 'reset(): a new sequence; states on both cards')

# ---- the device's dispatch and the one-card modules ---------------------------------------------------------------------
good = True
for h in ('k3_q35_ops.h', 'k3_q35t_ops.h'):
    src = open(os.path.join(HERE, '..', '..', 'kernels', h)).read()
    disp = src[src.index('__device__ __forceinline__ bool k3_extra_op('):]
    disp = disp[:disp.index('\n}\n')]
    good &= bool(re.search(r'#ifdef K3_TP\s*(//[^\n]*)?\n\s*if \(sg\.op == OP_XCHG\)', disp)) and disp.count('OP_XCHG') == 1
check(good, 'kernels/k3_q35_ops.h and k3_q35t_ops.h dispatch OP_XCHG only under K3_TP')
B = lambda n: torch.zeros(n)
e1 = k3.Engine(module='k3_q35_swift'); e1.xchg(B(256), B(256), 1 << 40, 2 << 40, torch.zeros(1, dtype=torch.int32), 0, 0)
try:
    e1.build(); refused = False
except ValueError:
    refused = True
check(refused and set(k3.Q35_TP_OF.values()) == set(k3.Q35_TP_MODULES) and all(m in k3.Q35_MODULES for m in k3.Q35_TP_MODULES),
      'a one-card module refuses OP_XCHG at build; the two-card modules are the dense line\'s modules')

# ---- projection -------------------------------------------------------------------------------------------------------
p0, p1 = Dense35TP.projection_mib(P_S, 64), Dense35TP.projection_mib(P_S, 65)
F = ggufio.GGUFFile(P_S); tot = sum(t.n_bytes for t in F.tensors) / 2 ** 20; F.close()
kvp = Dense35TP.kv_pos_mib(P_S)
from caminus.engine.q35_dense import attn_part_elems
rec = 4 * (attn_part_elems(1, iS['H'] // 2, 65) - attn_part_elems(1, iS['H'] // 2, 64)) / 2 ** 20   # (65 rows: one more split's records)
check(rec > 0 and all(abs((b - a) - kvp - rec) < 1e-9 for a, b in zip(p0, p1)) and abs(kvp * 2 ** 20 - 1 * 4 * (iS['HKV'] // 2) * tiny_q35.HD) < 1e-6,
      f'the projection grows by each card\'s KV row ({kvp * 1024:.2f} KiB: its kv heads of each attention block) and its split records per 32 rows')
w = [v - 256 for v in p0]
check(tot < w[0] + w[1] < 2 * tot + 1, f'the two cards\' parts add up to more than the file ({tot:.2f} MiB: replicated vectors and '
      f'embedding) and less than twice it ({w[0]:.2f} + {w[1]:.2f} MiB with states, slots, KV)')

# ---- the own-code path imports no gguf-py (caminus serve --block-imports gguf,tokenizers on the card) ------------------------
import subprocess
BLOCK = ("import sys\nclass B:\n    def find_spec(self, name, path=None, target=None):\n"
         "        if name.split('.')[0] in ('gguf', 'tokenizers'): raise ImportError('blocked: ' + name)\n"
         "sys.meta_path.insert(0, B()); sys.path.insert(0, %r)\n"
         "import caminus.engine.q35_dense_tp, caminus.prefill.q35own_tp, caminus.prefill.xchg\n"
         "from caminus import arch; arch.ARCHS['qwen35-own-tp'].engine_class()\nprint('IMPORTED')\n") % os.path.join(HERE, '..', '..', 'python')
r = subprocess.run([sys.executable, '-I', '-c', BLOCK], capture_output=True, text=True, timeout=120)
check(r.returncode == 0 and 'IMPORTED' in r.stdout, 'the two-card engine, its prefill and the exchange import with gguf-py and tokenizers refused'
      + ('' if r.returncode == 0 else f' ({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode})'))

print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
