"""GPU gate of the dense qwen35 drafter (engine/q35_mtp.py Mtp35: Swift 1.5 on kernels/k3_q35_swift_mtp.hip, ThinkingCap on
k3_q35_tcap_mtp.hip, from the file), one card.

The trunk (Dense35Own) runs teacher-forced over a fixed text of this tree (docs/dev/architecture.md), one token per pass,
keeping its h per position; the block runs on the drafter module and, as the reference, in fp32 torch from the host
decoders (caminus.own_dequant), with the logits of both through the trunk's own head tile.
  1. rows: fill() in tapes of 4 rows, of 3 + 1, and of 1 row each give the same KV rows bit for bit; a round's catch
     tape of T rows (T = 1 .. 4) gives the same KV rows, last-row hn and draft as T rounds of one row.
  2. argmax: the device key's token is the first maximum of the logits it was computed from.
  3. agreement: at every position the first draft (from the trunk's h) and the two chained drafts against the fp32
     block's: top-1 agreement and the logits' mean KLD (reference || device) of the first draft. PASS at >= 97 %
     first-draft agreement, >= 90 % for the chained drafts, mean KLD <= 1e-3.
  4. refusals: a round past the filled rows (a gap), a span past the cache, a bad token, the drafter module in the
     verify engine.
  then the time of a round (k = 3) for T = 1 .. 4 rows, median of 21.
  tools/guard.sh LOG 1800 python -I tests/engine/test_q35_mtp.py GGUF [--n 64]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>). Prints PASS / FAIL and RESULT."""
import argparse, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'python'))
import numpy as np
import torch
from caminus import ggufio, own_dequant
from caminus.bpe import BPE
from caminus.device import capacity
from caminus.engine import k3
from caminus.engine.q35_dense import Dense35Own
from caminus.engine.q35_dense_verify import Dense35Verify
from caminus.engine.q35_mtp import Mtp35

ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('--n', type=int, default=64)
A = ap.parse_args()
N = A.n
bad = 0


def check(ok, what):
    global bad
    bad += not ok
    print(('PASS' if ok else 'FAIL') + ' ' + what, flush=True)


eng = Dense35Own(A.gguf, max_ctx=N + 64, vram_extra_mib=2400)
dev = torch.device(f'cuda:{eng.dev}'); F = eng.F; D = eng.D
print(torch.cuda.get_device_name(eng.dev), os.path.basename(A.gguf), flush=True)
mtp = Mtp35(eng, k=3)
print(f'drafter: block blk.{mtp.il}, {mtp.mib:.0f} MiB; tapes ' + ', '.join(f'{t[0]} {t[1]}: {len(E.segs)} segments' for t, E in mtp.E.items()), flush=True)
tk = BPE(ggufio.GGUFFile(A.gguf).meta)
ids = tk.encode(open(os.path.join(ROOT, 'docs', 'dev', 'architecture.md'), encoding='utf-8').read())[:N + 8]

# ---- the trunk, teacher-forced: H[i] = h after position i (predicts token i + 1) -------------------------------------
eng.reset(); H = []
for i in range(len(ids)):
    eng.verify(ids, i, 0 if i == 0 else 1)
    H.append(eng.buf['h'][:D].float().clone())
Hs = torch.stack(H)

# ---- 1. rows: fill in 4s, in 3 + 1, in 1s; a round of T rows vs T rounds of one --------------------------------------
M = 2 * eng.nkv * eng.hd


def rows(a, b): return mtp.cache[a * M:b * M].clone()


L = 24
mtp.reset(); mtp.fill(Hs[0:L], ids[1:L + 1], 1); r4 = rows(1, L + 1)
mtp.reset()
for a in range(1, L + 1, 4): mtp.fill(Hs[a - 1:a + 2], ids[a:a + 3], a); mtp.fill(Hs[a + 2:a + 3], ids[a + 3:a + 4], a + 3)
r31 = rows(1, L + 1)
mtp.reset()
for a in range(1, L + 1): mtp.fill(Hs[a - 1:a], ids[a:a + 1], a)
r1 = rows(1, L + 1)
check(torch.equal(r4, r1) and torch.equal(r31, r1), f'fill: rows 1 .. {L} the same bits in tapes of 4, of 3 + 1 and of 1 row')
for T in range(1, 5):
    p0 = L + 1
    mtp.reset(); mtp.fill(Hs[0:L], ids[1:L + 1], 1)
    dT = mtp.draft(Hs[p0 - 1:p0 - 1 + T], ids[p0:p0 + T], p0, k=1)
    kvT = rows(p0, p0 + T); hnT = mtp.b['hn'][3 * D:4 * D].clone()
    mtp.reset(); mtp.fill(Hs[0:L], ids[1:L + 1], 1)
    for t in range(T): d1 = mtp.draft(Hs[p0 - 1 + t:p0 + t], ids[p0 + t:p0 + t + 1], p0 + t, k=1)
    ok = torch.equal(kvT, rows(p0, p0 + T)) and torch.equal(hnT, mtp.b['hn'][3 * D:4 * D]) and dT == d1
    check(ok, f'round of {T} rows: KV rows, last hn and draft the bits of {T} rounds of one row')

# ---- 2. argmax: the key's token is the first maximum of the logits -------------------------------------------------
mtp.reset(); mtp.fill(Hs[0:L], ids[1:L + 1], 1)
d = mtp.draft(Hs[L:L + 1], ids[L + 1:L + 2], L + 1, k=1)
lg = mtp.b['logits'].double().cpu().numpy()
check(d[0] == int(np.argmax(lg)), f'argmax: the device key\'s token {d[0]} is the first maximum of its logits ({int(np.argmax(lg))})')
# a synthetic tie: the first of the equal values wins
with torch.cuda.device(eng.dev):
    E = k3.Engine(module=mtp.module, grid=k3.resident_grid(mtp.module, eng.per_wgp), T=1, limit_us=eng.limit_us)
    z = torch.full((eng.vocab,), -3.0, device=dev); z[777] = 5.0; z[123456] = 5.0; z[5] = float('nan')
    key = torch.zeros(1, dtype=torch.int64, device=dev)
    E.own_argmax(z, eng.vocab, key); E.build(); E.run(check=True); torch.cuda.synchronize()
    check(k3.key_token(int(key.item())) == 777, f'argmax: of two equal maxima the first index wins, a NaN loses (got {k3.key_token(int(key.item()))})')

# ---- 3. agreement with the fp32 block --------------------------------------------------------------------------------
capacity.check([eng.dev], [1800], 'test_q35_mtp: the fp32 reference block')
P_ = f'blk.{mtp.il}.'


def Wf(name):
    t = F.T[name]; R, _ = t.rows()
    out = torch.empty((R, int(t.shape[0])), dtype=torch.float32, device=dev)
    step = max(1, (1 << 24) // int(t.shape[0]))
    for a in range(0, R, step):
        out[a:min(R, a + step)] = torch.from_numpy(own_dequant.rows(t, a, min(R, a + step)).astype(np.float32)).to(dev)
    return out


RB = {k: Wf(P_ + n + '.weight') for k, n in dict(eh='nextn.eh_proj', q='attn_q', k='attn_k', v='attn_v', o='attn_output',
                                                    g='ffn_gate', u='ffn_up', d='ffn_down').items()}
RN = {k: Wf(P_ + n + '.weight').reshape(-1) for k, n in dict(enorm='nextn.enorm', hnorm='nextn.hnorm', shn='nextn.shared_head_norm',
                                                               an='attn_norm', pn='post_attention_norm', qn='attn_q_norm', kn='attn_k_norm').items()}
emb = F.T['token_embd.weight']; base = float(F.hp('rope.freq_base')); nrot, hd, nh, nkv, eps = eng.nrot, eng.hd, eng.nh, eng.nkv, eng.eps


def rms(x, w): return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * w


def rope(x, pos):
    half = nrot // 2
    th = pos * base ** (-2.0 * torch.arange(half, device=dev, dtype=torch.float32) / nrot)
    c, s = torch.cos(th), torch.sin(th)
    a, b = x[..., :half], x[..., half:nrot]
    return torch.cat([a * c - b * s, a * s + b * c, x[..., nrot:]], -1)


x16 = torch.zeros(D, dtype=torch.float16, device=dev); hl = torch.zeros(eng.vocab, device=dev)
with torch.cuda.device(eng.dev):
    He = k3.Engine(module=eng.module, grid=k3.resident_grid(eng.module, eng.per_wgp), T=1)
    if eng.head['fmt'] == 'Q8_0': He.own_matvec16(eng.head['w'], 'Q8_0', eng.head['rows'], eng.head['K'], x16, hl, rows_per_task=eng.rpt16)   # (ThinkingCap's head)
    else: He.own_matvec_1rt(eng.head['w'], eng.head['fmt'], eng.head['rows'], eng.head['K'], x16, hl, rows_per_task=eng.rpt)
    He.build()


def head(hn):
    x16.copy_(hn.to(torch.float16)); He.run(check=False); torch.cuda.synchronize(); return hl.clone()


class Ref:
    def __init__(self, n): self.K = torch.zeros((n, nkv, hd), device=dev); self.V = torch.zeros((n, nkv, hd), device=dev)

    def step(self, h_prev, tok, pos):
        e = torch.from_numpy(own_dequant.rows(emb, tok, tok + 1).astype(np.float32)).to(dev).reshape(-1)
        x = RB['eh'] @ torch.cat([rms(e, RN['enorm']), rms(h_prev, RN['hnorm'])])
        hh = rms(x, RN['an'])
        qf = (RB['q'] @ hh).reshape(nh, 2 * hd); ps = torch.tensor(float(pos), device=dev)
        q = rope(rms(qf[:, :hd], RN['qn']), ps); gate = qf[:, hd:]
        k = rope(rms((RB['k'] @ hh).reshape(nkv, hd), RN['kn']), ps); v = (RB['v'] @ hh).reshape(nkv, hd)
        self.K[pos] = k; self.V[pos] = v
        G = nh // nkv; att = torch.empty((nh, hd), device=dev)
        for j in range(nkv):
            sc = (q[j * G:(j + 1) * G] @ self.K[:pos + 1, j].T) / np.sqrt(hd)
            att[j * G:(j + 1) * G] = torch.softmax(sc, -1) @ self.V[:pos + 1, j]
        x = x + RB['o'] @ (att * torch.sigmoid(gate)).reshape(-1)
        a = rms(x, RN['pn'])
        x = x + RB['d'] @ (torch.nn.functional.silu(RB['g'] @ a) * (RB['u'] @ a))
        return rms(x, RN['shn'])


ref = Ref(N + 16)
mtp.reset()
agree = [0, 0, 0]; total = [0, 0, 0]; klds = []
for j in range(1, N):
    # the device: rows 1 .. j - 1 filled, the round at j (one row), 3 drafts; the reference: row j, then two chained rows
    if j > 1: mtp.fill(Hs[j - 2:j - 1], ids[j - 1:j], j - 1)
    dd = mtp.draft(Hs[j - 1:j], ids[j:j + 1], j, k=1)
    lg_dev = mtp.b['logits'].double()
    dk = mtp.draft(Hs[j - 1:j], ids[j:j + 1], j, k=3)
    hn = ref.step(Hs[j - 1], ids[j], j)
    lr = head(hn).double()
    pr = torch.softmax(lr, -1); kl = float((pr * (torch.log_softmax(lr, -1) - torch.log_softmax(lg_dev, -1))).sum())
    klds.append(kl)
    rt = [int(torch.argmax(lr))]
    sav = (ref.K.clone(), ref.V.clone()); h2 = hn
    for c in range(2):
        h2 = ref.step(h2, rt[-1], j + 1 + c); rt.append(int(torch.argmax(head(h2))))
    ref.K, ref.V = sav
    for i in range(3):
        if i and dk[i - 1] != rt[i - 1]: break   # (a chained draft compares only while the drafts before it agree)
        total[i] += 1; agree[i] += dk[i] == rt[i]
    assert dd[0] == dk[0]
a1 = agree[0] / max(1, total[0]); a23 = (agree[1] + agree[2]) / max(1, total[1] + total[2])
print(f'agreement with the fp32 block over {N - 1} positions: draft 1 {agree[0]}/{total[0]}, draft 2 {agree[1]}/{total[1]}, '
      f'draft 3 {agree[2]}/{total[2]}; first-draft logits mean KLD {np.mean(klds):.2e} (max {np.max(klds):.2e})', flush=True)
check(a1 >= 0.97 and a23 >= 0.90 and np.mean(klds) <= 1e-3,
      f'agreement: draft 1 {100 * a1:.1f} % (>= 97), chained {100 * a23:.1f} % (>= 90), mean KLD {np.mean(klds):.1e} (<= 1e-3)')
del RB; torch.cuda.empty_cache()

# ---- 4. refusals -------------------------------------------------------------------------------------------------------
mtp.reset(); mtp.fill(Hs[0:8], ids[1:9], 1)
for what, fn in (('a round past the filled rows', lambda: mtp.draft(Hs[10:11], ids[11:12], 11)),
                 ('a span past the cache', lambda: mtp.draft(Hs[8:9], ids[9:10], eng.max_ctx + 10)),
                 ('a token outside the vocabulary', lambda: mtp.draft(Hs[8:9], [eng.vocab], 9)),
                 ('the drafter module in the verify engine', lambda: Dense35Verify(eng, 1, module=mtp.module))):
    try: fn(); check(False, f'refused: {what}')
    except (ValueError, IndexError): check(True, f'refused: {what}')

# ---- speed -------------------------------------------------------------------------------------------------------------
mtp.reset(); mtp.fill(Hs[0:N - 8], ids[1:N - 7], 1)
for T in range(1, 5):
    p0 = N - 8 - T + 1 if T > 1 else N - 7
    ts = []
    for r in range(22):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        mtp.draft(Hs[p0 - 1:p0 - 1 + T], ids[p0:p0 + T], p0, k=3)
        ts.append(time.perf_counter() - t0)
    print(f'round, {T} rows + 2 chained steps (k 3), at position {p0}: {1e3 * np.median(ts[1:]):.2f} ms', flush=True)
print('RESULT', 'PASS' if bad == 0 else 'FAIL'); sys.exit(1 if bad else 0)
