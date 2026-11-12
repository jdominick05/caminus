"""G-acc, G-self and a standalone G-speed read for the Q4_0 matvec of the dense qwen35 line's multi-token-prediction block
(Swift 1.5: blk.<n_layer>.* all Q4_0; docs/specs/spec-own-quant.md), on one card.

Activations: recorded at each Q4_0 op site of the block while it runs teacher-forced over two fixed texts from this tree
(docs/dev/architecture.md, prose and tables; python/caminus/arch.py, code): the trunk on Dense35Own (one token per pass,
its normed final residual h from buf['h']), the block in fp32 torch from the host decoders (caminus.own_dequant), at MTP
position j the pair (h_{j-1}, token_j) as llama.cpp pairs them, plus two chained steps every 8 positions (the block's own
output as the next h, the text's next token as the draft), as a drafter's accepted drafts run. The sites: eh_proj [e; h] (10240), q | k | v (5120,
the attention norm's output), attn_output (6144, the gated attention), ffn_gate | ffn_up (5120), ffn_down (17408).
The block has no reference mode in the engine yet, so its fp32 forward stands in for "the activation as the op receives it".

Per tensor, on 256 sampled rows x every recorded vector of its site:
  G-acc   |y - y_ref| / rms(y_ref), y_ref the float64 dot of our decoders' rows (bitwise gguf-py's) with the fp32
          activation; mean and p99 over rows x vectors; own (fp16 activations, kernels/own_quant_q4_0.h on the plane
          layout) against llama.cpp (ggml-cuda's vec_dot_q4_0_q8_1 on q8_1 activations, kernels/gg_mv_ref.hip). PASS when
          own's mean and p99 are both at most llama.cpp's.
  G-self  4 tokens in one launch (nt = 4) equal 4 one-token launches, bit for bit, on every row.
  speed   cold: copies of the tensor rotated past the Infinity Cache (at least 512 MiB); median us per matvec over
          --rounds: read (the same bytes, the bandwidth ceiling), llama, own at nt 1 and nt 4 (one launch).
  tools/guard.sh LOG 1800 python -I benchmarks/q35_mtp_gacc.py GGUF [--rounds 7] [--positions 48]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>). Prints PASS / FAIL and RESULT."""
import argparse, ctypes, gc, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'python'))
import numpy as np
import torch
from caminus import ggufio, own_dequant
from caminus.bpe import BPE
from caminus.device import capacity
from caminus.device.hip import Module
from caminus.engine.own_layout import q4_0_planes
from caminus.engine.q35_dense import Dense35Own

ap = argparse.ArgumentParser()
ap.add_argument('gguf')
ap.add_argument('--rounds', type=int, default=7)
ap.add_argument('--positions', type=int, default=48, help='teacher-forced MTP positions per text')
A = ap.parse_args()
rng = np.random.default_rng(0)
TEXTS = [os.path.join(ROOT, 'docs', 'dev', 'architecture.md'), os.path.join(ROOT, 'python', 'caminus', 'arch.py')]
N = A.positions
ok_all = True


def say(s): print(s, flush=True)


eng = Dense35Own(A.gguf, max_ctx=N + 64, vram_extra_mib=2400)
dev = torch.device(f'cuda:{eng.dev}'); F = eng.F
L = eng.nl; P_ = f'blk.{L}.'
D, hd, nh, nkv, eps = eng.D, eng.hd, eng.nh, eng.nkv, eng.eps
tk = BPE(ggufio.GGUFFile(A.gguf).meta)
say(f'{torch.cuda.get_device_name(eng.dev)}  {os.path.basename(A.gguf)}  the block: {P_}*')
Q4 = {'eh': 'nextn.eh_proj', 'q': 'attn_q', 'k': 'attn_k', 'v': 'attn_v', 'o': 'attn_output', 'g': 'ffn_gate', 'u': 'ffn_up',
      'd': 'ffn_down'}
for k, n in Q4.items():
    t = F.T[P_ + n + '.weight']
    if t.type != 'Q4_0': sys.exit(f'q35_mtp_gacc: {P_}{n} is {t.type}, not Q4_0 (this tool is for Swift 1.5\'s block)')
SITE = {'eh': 'eh', 'q': 'hh', 'k': 'hh', 'v': 'hh', 'o': 'att', 'g': 'a', 'u': 'a', 'd': 'hm'}


# ---- 1. the block in fp32 (dequantized by our decoders) and the recorded activations --------------------------------
capacity.check([eng.dev], [2048], 'q35_mtp_gacc: the block in fp32 (~1.7 GiB) and its KV')


def W(name):
    t = F.T[name]; R, _ = t.rows()
    out = torch.empty((R, int(t.shape[0])), dtype=torch.float32, device=dev)
    step = max(1, (1 << 24) // int(t.shape[0]))
    for a in range(0, R, step):
        out[a:min(R, a + step)] = torch.from_numpy(own_dequant.rows(t, a, min(R, a + step)).astype(np.float32)).to(dev)
    return out


def V(name): return W(name).reshape(-1)


B = {k: W(P_ + n + '.weight') for k, n in Q4.items()}
B.update(enorm=V(P_ + 'nextn.enorm.weight'), hnorm=V(P_ + 'nextn.hnorm.weight'), shn=V(P_ + 'nextn.shared_head_norm.weight'),
         an=V(P_ + 'attn_norm.weight'), pn=V(P_ + 'post_attention_norm.weight'), qn=V(P_ + 'attn_q_norm.weight'),
         kn=V(P_ + 'attn_k_norm.weight'))
emb = F.T['token_embd.weight']; base = float(F.hp('rope.freq_base')); nrot = eng.nrot


def rms(x, w): return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * w


def rope(x, pos):
    half = nrot // 2
    th = pos * base ** (-2.0 * torch.arange(half, device=dev, dtype=torch.float32) / nrot)
    c, s = torch.cos(th), torch.sin(th)
    a, b = x[..., :half], x[..., half:nrot]
    return torch.cat([a * c - b * s, a * s + b * c, x[..., nrot:]], -1)


REC = {s: [] for s in set(SITE.values())}


class Block:
    def __init__(self, n):
        self.K = torch.zeros((n, nkv, hd), device=dev); self.V = torch.zeros((n, nkv, hd), device=dev)

    def step(self, h_prev, tok, pos, rec=True):
        e = torch.from_numpy(own_dequant.rows(emb, tok, tok + 1).astype(np.float32)).to(dev).reshape(-1)
        xe = torch.cat([rms(e, B['enorm']), rms(h_prev, B['hnorm'])])
        x = B['eh'] @ xe
        hh = rms(x, B['an'])
        qf = (B['q'] @ hh).reshape(nh, 2 * hd)
        ps = torch.tensor(float(pos), device=dev)
        q = rope(rms(qf[:, :hd], B['qn']), ps); gate = qf[:, hd:]
        k = rope(rms((B['k'] @ hh).reshape(nkv, hd), B['kn']), ps); v = (B['v'] @ hh).reshape(nkv, hd)
        self.K[pos] = k; self.V[pos] = v
        Kc, Vc = self.K[:pos + 1], self.V[:pos + 1]
        G = nh // nkv; att = torch.empty((nh, hd), device=dev)
        for j in range(nkv):
            sc = (q[j * G:(j + 1) * G] @ Kc[:, j].T) / np.sqrt(hd)
            att[j * G:(j + 1) * G] = torch.softmax(sc, -1) @ Vc[:, j]
        att = (att * torch.sigmoid(gate)).reshape(-1)
        x = x + B['o'] @ att
        a = rms(x, B['pn'])
        hm = torch.nn.functional.silu(B['g'] @ a) * (B['u'] @ a)
        x = x + B['d'] @ hm
        if rec:
            for s, val in (('eh', xe), ('hh', hh), ('att', att), ('a', a), ('hm', hm)): REC[s].append(val.clone())
        return rms(x, B['shn'])


t0 = time.time(); ch = None
for path in TEXTS:
    ids = tk.encode(open(path, encoding='utf-8').read())[:N + 1]
    eng.reset(); H = []
    for i in range(len(ids)):                       # the trunk, teacher-forced: h[i] predicts token i + 1
        eng.verify(ids, i, 0 if i == 0 else 1)
        H.append(eng.buf['h'][:D].float().clone())
    blk = Block(N + 8)
    for j in range(1, len(ids)):
        hn = blk.step(H[j - 1], ids[j], j)
        if j % 8 == 0 and j + 3 < len(ids):         # two chained draft steps on a scratch copy of the block's KV: the
            ch = Block(N + 8); ch.K.copy_(blk.K); ch.V.copy_(blk.V)   # block's own output as h, the drafts right (the
            for c in range(2): hn = ch.step(hn, ids[j + 1 + c], j + 1 + c)   # text's next tokens), as accepted drafts run
say(f'recorded: ' + ', '.join(f'{s} {len(v)} x {v[0].numel()}' for s, v in REC.items()) + f' ({time.time() - t0:.0f} s)')
X = {s: torch.stack(v).cpu() for s, v in REC.items()}
DEV = eng.dev
del B, blk, ch, REC, H, eng; gc.collect(); torch.cuda.empty_cache()   # (the trunk's weights go: the timing copies need the room)

# ---- 2. per tensor: G-acc, G-self, speed -------------------------------------------------------------------------------
OWN = Module('own_mv_q35'); GG = Module('gg_mv_ref'); RD = Module('own_mv')   # (own_read: the plain read)
sink = torch.zeros(1, dtype=torch.int32, device=dev)


def q8_1(x32):
    K = x32.numel(); y = torch.zeros(K // 32 * 36, dtype=torch.uint8, device=dev)
    GG.gg_ref_quantize_q8_1(((K + 255) // 256,), (256,), x32, y, ctypes.c_int(K)); return y


def run_own(w, rows, K, x16, out, nt):
    OWN.own_mv_q4_0_k3(((rows + 15) // 16,), (256,), w, ctypes.c_int(rows), ctypes.c_int(K), x16, out, ctypes.c_int(nt))


def run_llama(w, rows, K, yq, out):
    GG.gg_ref_mv_q4_0(((rows + 3) // 4,), (32, 4), w, yq, out, ctypes.c_int(rows), ctypes.c_int(K))


def run_read(w):
    n16 = w.numel() // 16
    RD.own_read((min(4096, (n16 + 255) // 256),), (256,), w, ctypes.c_long(n16), sink)


rows_tab = []
for key, name in Q4.items():
    t = F.T[P_ + name + '.weight']; rows, _ = t.rows(); K = int(t.shape[0])
    raw = np.ascontiguousarray(t.data()).reshape(-1)
    mat_bytes = raw.size
    ncp = max(1, -(-(512 << 20) // mat_bytes))
    capacity.check([DEV], [2 * ncp * mat_bytes / 2 ** 20 + 256], f'q35_mtp_gacc {name}: {ncp} copies x 2 layouts')
    Wl = [torch.from_numpy(raw.copy()).to(dev) for _ in range(ncp)]
    pl = q4_0_planes(raw, rows, K)
    Wo = [torch.from_numpy(pl.copy()).to(dev) for _ in range(ncp)]
    xs = X[SITE[key]]                                # [n, K] fp32
    assert xs.shape[1] == K, (name, xs.shape, K)
    srows = np.sort(rng.choice(rows, min(256, rows), replace=False))
    W64 = own_dequant.dequant(raw.reshape(rows, -1)[srows].reshape(-1), 'Q4_0', len(srows) * K).astype(np.float64).reshape(len(srows), K)
    e_own, e_ll = [], []
    out = torch.zeros(4 * rows, dtype=torch.float32, device=dev)
    for x in xs:
        yref = W64 @ x.double().numpy(); sc = np.sqrt((yref ** 2).mean())
        x32 = x.to(dev).contiguous()
        out.zero_(); run_own(Wo[0], rows, K, x32.half(), out, 1); torch.cuda.synchronize()
        e_own.append(np.abs(out[:rows].double().cpu().numpy()[srows] - yref) / sc)
        out.zero_(); run_llama(Wl[0], rows, K, q8_1(x32), out); torch.cuda.synchronize()
        e_ll.append(np.abs(out[:rows].double().cpu().numpy()[srows] - yref) / sc)
    e_own, e_ll = np.concatenate(e_own), np.concatenate(e_ll)
    mo, po, ml, pll = e_own.mean(), np.percentile(e_own, 99), e_ll.mean(), np.percentile(e_ll, 99)
    acc_ok = mo <= ml and po <= pll
    # G-self: nt 4 in one launch vs 4 launches of nt 1, bitwise, every row
    x4 = xs[:4].to(dev).half().contiguous()
    o4 = torch.zeros(4 * rows, dtype=torch.float32, device=dev); o1 = torch.zeros(4 * rows, dtype=torch.float32, device=dev)
    run_own(Wo[0], rows, K, x4.reshape(-1), o4, 4)
    for j in range(4): run_own(Wo[0], rows, K, x4[j].contiguous(), o1[j * rows:], 1)
    torch.cuda.synchronize()
    self_ok = bool(torch.equal(o4, o1))
    # speed, cold, interleaved round by round
    x16 = xs[0].to(dev).half().contiguous(); yq = q8_1(xs[0].to(dev).contiguous())
    arms = {'read': (Wl, lambda w: run_read(w)), 'llama': (Wl, lambda w: run_llama(w, rows, K, yq, out)),
            'own': (Wo, lambda w: run_own(w, rows, K, x16, out, 1)), 'own T4': (Wo, lambda w: run_own(w, rows, K, x4.reshape(-1), o4, 4))}
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    tm = {k: [] for k in arms}
    for r in range(A.rounds + 1):
        for k, (Ws, fn) in arms.items():
            torch.cuda.synchronize(); ev[0].record()
            for w in Ws: fn(w)
            ev[1].record(); torch.cuda.synchronize()
            if r: tm[k].append(ev[0].elapsed_time(ev[1]) * 1e3 / len(Ws))
    us = {k: float(np.median(v)) for k, v in tm.items()}
    ok = acc_ok and self_ok; ok_all &= ok
    say(f'{"PASS" if ok else "FAIL"} {name:16s} [{rows:6d} x {K:5d}] {mat_bytes / 2**20:6.2f} MiB, {len(xs)} vectors x {len(srows)} rows | '
        f'err own {mo:.2e} / p99 {po:.2e}, llama.cpp {ml:.2e} / p99 {pll:.2e} ({ml / mo:.0f}x / {pll / po:.0f}x) | '
        f'nt 4 = 4 x nt 1 bitwise: {self_ok} | us: read {us["read"]:.1f}, llama {us["llama"]:.1f}, own {us["own"]:.1f} '
        f'({mat_bytes / us["own"] / 1e3:.0f} GB/s), own T4 {us["own T4"]:.1f} ({us["own T4"] / us["own"]:.2f}x)')
    rows_tab.append((name, us))
    del Wl, Wo; torch.cuda.empty_cache()
tot = {k: sum(u[k] for _, u in rows_tab) for k in ('read', 'llama', 'own', 'own T4')}
say(f'the block\'s Q4_0 matvecs per draft step (one launch each): read {tot["read"]:.0f} us, llama {tot["llama"]:.0f} us, '
    f'own {tot["own"]:.0f} us, own T4 {tot["own T4"]:.0f} us')
say('RESULT ' + ('PASS' if ok_all else 'FAIL'))
sys.exit(0 if ok_all else 1)
