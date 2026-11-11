"""fp32 reference logits of a qwen3moe GGUF at depth (spec-own-quant's fp32 activation reference mode): every quantized weight
decoded exactly (caminus.own_dequant's formulas, ported to torch: the same float32 statements), activations, KV and sums in
fp32, RoPE angles in float64, layer-major over the whole prompt on one card (the residual [N, D] kept; per layer its
attention in query chunks and its experts one by one). Logits of the last M positions. Cross-check: at N = 128 against
tools/reference/q3moe_ref.py's float64 rows (mean KLD ~1e-7 expected).
The text: the tree's own docs and python sources (public), concatenated in sorted order, the first N tokens; or --seq.
  tools/guard.sh LOG 1800 python -I tools/reference/q3moe_ref32.py GGUF OUT.npz --n N [--last 64] [--seq SEQ.npz]
OUT.npz: seq (int64 [N]), logits (float32 [M, vocab]) for positions N - M .. N - 1. Its peak VRAM is projected from the shapes
(peak_mib), checked against the card's cap before the first device transfer, and set as the process's allocator limit:
8.1 GB at N = 40,960, --qchunk 256 (measured at 40,896: 7.8 GB)."""
import argparse, glob, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
import torch
from caminus import ggufio, iq_tables
from caminus.bpe import BPE
from caminus.device import capacity

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')


def f16(b):
    return b.contiguous().view(torch.float16).float()


def q4_k(raw):
    """caminus.own_dequant.q4_k in torch: raw uint8 [nb * 144] -> float32 [nb * 256]"""
    b = raw.view(-1, 144)
    d, dmin = f16(b[:, 0:2]).view(-1, 1), f16(b[:, 2:4]).view(-1, 1)
    s = b[:, 4:16]
    sc = torch.empty((b.shape[0], 8), dtype=torch.uint8, device=b.device); m = torch.empty_like(sc)
    sc[:, :4] = s[:, 0:4] & 63; m[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    m[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    dl = (d * sc.float()).view(-1, 8, 1); ml = (dmin * m.float()).view(-1, 8, 1)
    qs = b[:, 16:144].view(-1, 4, 32)
    q = torch.stack([qs & 0xF, qs >> 4], dim=2).view(-1, 8, 32).float()
    return (dl * q - ml).reshape(-1)


def q6_k(raw):
    b = raw.view(-1, 210)
    ql, qh = b[:, 0:128].view(-1, 2, 64), b[:, 128:192].view(-1, 2, 32)
    sc = b[:, 192:208].contiguous().view(torch.int8).float()
    d = f16(b[:, 208:210]).view(-1, 1)
    q = torch.empty((b.shape[0], 2, 4, 32), dtype=torch.int16, device=b.device)
    q[:, :, 0] = ((ql[:, :, :32] & 0xF) | (((qh >> 0) & 3) << 4)).short()
    q[:, :, 1] = ((ql[:, :, 32:] & 0xF) | (((qh >> 2) & 3) << 4)).short()
    q[:, :, 2] = ((ql[:, :, :32] >> 4) | (((qh >> 4) & 3) << 4)).short()
    q[:, :, 3] = ((ql[:, :, 32:] >> 4) | (((qh >> 6) & 3) << 4)).short()
    q = q.float() - 32.0
    ds = (d * sc).view(-1, 2, 4, 2, 1)
    return (ds * q.view(-1, 2, 4, 2, 16)).reshape(-1)


# ---- the codebook formats: caminus.own_dequant's iq4_xs / iq3_xxs / iq3_s / iq2_xxs in torch, the same float32 statements
_TAB = {}


def tab(name, dev):
    """a table of caminus.iq_tables on the device"""
    if (name, str(dev)) not in _TAB: _TAB[(name, str(dev))] = torch.from_numpy(np.ascontiguousarray(getattr(iq_tables, name))).to(dev)
    return _TAB[(name, str(dev))]


def _pm(bits):
    """uint8 [...] -> float32 [..., 8]: +1 where the bit is clear, -1 where it is set (bit i = value i)"""
    sh = torch.arange(8, device=bits.device, dtype=torch.uint8)
    one = torch.ones((), device=bits.device)
    return torch.where(((bits[..., None] >> sh) & 1) == 0, one, -one)


def _signs7(idx):
    par = idx ^ (idx >> 4); par = par ^ (par >> 2); par = par ^ (par >> 1)
    return idx | ((par & 1) << 7)


def _nib(b):
    return torch.stack([b & 0xF, b >> 4], dim=2).reshape(b.shape[0], -1)


def _su32(b):
    """uint8 [n, 32] (8 u32) -> (scale float32 [n, 8], signs float32 [n, 8, 32])"""
    u = b.contiguous().view(torch.int32).view(-1, 8).long() & 0xffffffff
    idx = ((u[:, :, None] >> torch.tensor([0, 7, 14, 21], device=b.device)) & 0x7F).to(torch.uint8)
    return (u >> 28).float(), _pm(_signs7(idx)).reshape(-1, 8, 32)


def iq4_xs(raw):
    b = raw.view(-1, 136); n = b.shape[0]
    d = f16(b[:, 0:2]).view(n, 1)
    sh = b[:, 2:4].contiguous().view(torch.int16).long().view(n, 1) & 0xffff
    ls = _nib(b[:, 4:8]).long() | (((sh >> (2 * torch.arange(8, device=b.device))) & 3) << 4)
    dl = (d * (ls - 32).float()).view(n, 8, 1)
    qs = b[:, 8:136].view(n, 8, 16)
    q = tab('KVALUES_IQ4', b.device)[torch.cat([qs & 0xF, qs >> 4], dim=2).long()].float()
    return (dl * q).reshape(-1)


def iq3_xxs(raw):
    b = raw.view(-1, 98); n = b.shape[0]
    d = f16(b[:, 0:2]).view(n, 1)
    s, sg = _su32(b[:, 66:98])
    db = ((d * (0.5 + s)) * 0.5).view(n, 8, 1)
    g = tab('GRID_IQ3_XXS', b.device)[b[:, 2:66].long()].float().view(n, 8, 32)
    return (db * g * sg).reshape(-1)


def iq3_s(raw):
    b = raw.view(-1, 110); n = b.shape[0]
    d = f16(b[:, 0:2]).view(n, 1)
    db = (d * (1 + 2 * _nib(b[:, 106:110]).long()).float()).view(n, 8, 1)
    hi = ((b[:, 66:74, None] >> torch.arange(8, device=b.device, dtype=torch.uint8)) & 1).long()
    g = tab('GRID_IQ3_S', b.device)[b[:, 2:66].reshape(n, 8, 8).long() | (hi << 8)].float().view(n, 8, 32)
    return (db * g * _pm(b[:, 74:106]).reshape(n, 8, 32)).reshape(-1)


def iq2_xxs(raw):
    b = raw.view(-1, 66); n = b.shape[0]
    d = f16(b[:, 0:2]).view(n, 1)
    w = b[:, 2:66].reshape(n, 8, 8)
    s, sg = _su32(w[:, :, 4:8].reshape(n, 32))
    db = ((d * (0.5 + s)) * 0.25).view(n, 8, 1)
    g = tab('GRID_IQ2_XXS', b.device)[w[:, :, 0:4].long()].float().view(n, 8, 32)
    return (db * g * sg).reshape(-1)


DECODERS = {'Q4_K': q4_k, 'Q6_K': q6_k, 'IQ4_XS': iq4_xs, 'IQ3_XXS': iq3_xxs, 'IQ3_S': iq3_s, 'IQ2_XXS': iq2_xxs}


class W:
    """a GGUF tensor's rows decoded on the card"""
    def __init__(self, F, dev):
        self.F, self.dev = F, dev

    def rows(self, name, r0=None, r1=None):
        t = self.F.T[name]; K = int(t.shape[0]); R, rb = t.rows()
        r0 = 0 if r0 is None else r0; r1 = R if r1 is None else r1
        if t.type == 'F32': return torch.from_numpy(np.asarray(t.data()).view(np.float32)[r0 * K:r1 * K].copy()).to(self.dev).view(r1 - r0, K)
        raw = torch.from_numpy(np.ascontiguousarray(np.asarray(t.data())[r0 * rb:r1 * rb])).to(self.dev)
        return DECODERS[t.type](raw).view(r1 - r0, K)


def rms(x, w, eps):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


SLACK_MIB = 1280   # the runtime's own memory, and blocks the allocator has freed that the driver has not returned yet
                   # (measured at N = 40,896: sysfs 713 MiB above PyTorch's reserved peak); the allocator's own limit is
                   # the projection less 1024 of it


def peak_mib(F, N, qchunk, nl, D, nh, nkv, hd, E, ff):
    """an upper bound of main()'s peak device memory in MiB as the guard reads it (sysfs), from the shapes: the largest of
    its three phases + SLACK_MIB
      attention   x, h; q before and after RoPE, att, k and v per query head; k and v twice for the rotation's
                  temporaries; a query chunk's scores and their softmax [nh, qchunk, N], its mask; the tables
      experts     x, h, y and one expert's gathered rows (at most all N); the router's logits, probabilities, top-k;
                  one expert's three matrices being decoded
      head        x and the largest tensor decoded whole (16 bytes per weight: the raw bytes, the fp32 result and the
                  decoder's live temporaries)"""
    res = N * D * 4
    whole = max(int(np.prod(t.shape)) for n, t in F.T.items() if n != 'token_embd.weight' and '_exps.' not in n)
    dec = lambda n: 16 * n
    attn = 2 * res + 5 * N * nh * hd * 4 + 4 * N * nkv * hd * 4 + 2 * nh * qchunk * N * 4 + qchunk * N + 3 * N * hd * 8 + dec(nh * hd * D)
    moe = 4 * res + 3 * N * E * 4 + 2 * N * 8 * 8 + 3 * N * ff * 4 + dec(3 * ff * D)
    head = res + dec(whole)
    emb = N * F.T['token_embd.weight'].rows()[1]   # (the prompt's raw embedding rows, on the card until they are decoded)
    return (max(attn, moe, head) + emb) / 2 ** 20 + SLACK_MIB


def tree_text():
    fs = sorted(glob.glob(os.path.join(ROOT, 'docs', '**', '*.md'), recursive=True)) + sorted(glob.glob(os.path.join(ROOT, 'python', 'caminus', '**', '*.py'), recursive=True))
    return ''.join(f'\n# file: {os.path.relpath(f, ROOT)}\n' + open(f, encoding='utf-8').read() for f in fs)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('out'); ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--last', type=int, default=64); ap.add_argument('--seq', default=None); ap.add_argument('--qchunk', type=int, default=256)
    A = ap.parse_args()
    dev = torch.device('cuda:0')
    F = ggufio.GGUFFile(A.gguf); Wt = W(F, dev)   # (W reads a tensor when it is asked for: nothing on the card yet)
    hp = lambda k: F.hp(k)
    nl, D, nh, nkv, hd = int(hp('block_count')), int(hp('embedding_length')), int(hp('attention.head_count')), int(hp('attention.head_count_kv')), int(hp('attention.key_length'))
    E, k, ff, eps, base = int(hp('expert_count')), int(hp('expert_used_count')), int(hp('expert_feed_forward_length')), float(hp('attention.layer_norm_rms_epsilon')), float(hp('rope.freq_base'))
    if A.seq: seq = [int(v) for v in np.load(A.seq)['seq']][:A.n]
    else: seq = BPE(F.meta).encode(tree_text())[:A.n]
    N = len(seq)
    if N < A.n: raise SystemExit(f'the text has {N} tokens < {A.n}')
    need = peak_mib(F, N, A.qchunk, nl, D, nh, nkv, hd, E, ff)
    capacity.check([0], [need], f'q3moe_ref32 (N {N}, --qchunk {A.qchunk}: {need:.0f} MiB projected; lower --qchunk or --n)')
    # the projection is also this process's allocator limit: PyTorch's caching allocator otherwise keeps every freed block
    # (measured at N = 16,384 without it: 19,964 MiB reserved for 3.6 GB of live tensors)
    torch.cuda.set_per_process_memory_fraction(min(1.0, (need - 1024) * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory), 0)
    print(torch.cuda.get_device_name(0), f'N {N}; projected peak {need:.0f} MiB', flush=True)
    te = F.T['token_embd.weight']; _, rb = te.rows()
    raw = torch.from_numpy(np.ascontiguousarray(np.asarray(te.data()).reshape(-1, rb)[np.array(seq)])).to(dev)
    x = q4_k(raw.view(-1)).view(N, D)
    kpos = torch.arange(N, device=dev)
    pos = torch.arange(N, device=dev, dtype=torch.float64)
    inv = base ** (-torch.arange(0, hd, 2, device=dev, dtype=torch.float64) / hd)
    ang = pos[:, None] * inv[None]; cos, sin = ang.cos().float(), ang.sin().float()     # [N, hd / 2]

    def rope(t):   # t [N, h, hd], NEOX halves
        a, b = t[..., :hd // 2], t[..., hd // 2:]
        c, s = cos[:, None], sin[:, None]
        return torch.cat([a * c - b * s, b * c + a * s], -1)
    for il in range(nl):
        p = f'blk.{il}.'
        h = rms(x, Wt.rows(p + 'attn_norm.weight')[0], eps)
        q = (h @ Wt.rows(p + 'attn_q.weight').T).view(N, nh, hd)
        kk = (h @ Wt.rows(p + 'attn_k.weight').T).view(N, nkv, hd)
        v = (h @ Wt.rows(p + 'attn_v.weight').T).view(N, nkv, hd)
        q = rope(rms(q, Wt.rows(p + 'attn_q_norm.weight')[0], eps)); kk = rope(rms(kk, Wt.rows(p + 'attn_k_norm.weight')[0], eps))
        del h
        att = torch.empty(N, nh, hd, device=dev)
        # every chunk over all N keys, the later ones masked: the same shapes in every chunk, so the allocator reuses its
        # blocks. (Scores over [0, a1) gave a new size per chunk: the freed blocks reached the driver faster than it
        # returned them, 15 GB in use by sysfs with 5 GB reserved at N = 40,896.)
        kT = kk.permute(1, 2, 0).unsqueeze(1).expand(nkv, nh // nkv, hd, N).contiguous()   # [nkv, G, hd, N]
        vq = v.permute(1, 0, 2).unsqueeze(1).expand(nkv, nh // nkv, N, hd).contiguous()    # [nkv, G, N, hd]
        del kk, v
        for a0 in range(0, N, A.qchunk):
            a1 = min(N, a0 + A.qchunk)
            qc = q[a0:a1].view(a1 - a0, nkv, nh // nkv, hd).permute(1, 2, 0, 3)          # [nkv, G, c, hd]
            s = torch.matmul(qc, kT).div_(math.sqrt(hd))                                 # [nkv, G, c, N]
            s.masked_fill_(kpos[None, None, None, :] > kpos[None, None, a0:a1, None], float('-inf'))
            pr = torch.softmax(s, -1); del s
            o = torch.matmul(pr, vq); del pr
            att[a0:a1] = o.permute(2, 0, 1, 3).reshape(a1 - a0, nh, hd)
            del o
        del q, kT, vq
        torch.cuda.empty_cache()   # (the experts' many sizes start from an empty cache, and the next layer's attention too)
        x = x + att.view(N, nh * hd) @ Wt.rows(p + 'attn_output.weight').T
        del att
        h = rms(x, Wt.rows(p + 'ffn_norm.weight')[0], eps)
        lg = h @ Wt.rows(p + 'ffn_gate_inp.weight').T
        pr = torch.softmax(lg, -1); wv, ids = torch.topk(pr, k, -1)               # (ties: the lower index first, as torch.topk)
        wv = wv / wv.sum(-1, keepdim=True)
        y = torch.zeros_like(x)
        for e in range(E):
            sel = (ids == e).nonzero()
            if not len(sel): continue
            t_, s_ = sel[:, 0], sel[:, 1]
            g_ = h[t_] @ Wt.rows(p + 'ffn_gate_exps.weight', e * ff, (e + 1) * ff).T
            u_ = h[t_] @ Wt.rows(p + 'ffn_up_exps.weight', e * ff, (e + 1) * ff).T
            dn = (g_ * torch.sigmoid(g_) * u_) @ Wt.rows(p + 'ffn_down_exps.weight', e * D, (e + 1) * D).T
            y.index_add_(0, t_, dn * wv[t_, s_][:, None])
        x = x + y
        del h, y, lg, pr, wv, ids
        torch.cuda.empty_cache()
        if il % 8 == 0 or il == nl - 1: print(f'  layer {il} done', flush=True)
    hl = rms(x[N - A.last:], Wt.rows('output_norm.weight')[0], eps).clone()
    del x; torch.cuda.empty_cache()   # (the head's decode gets one contiguous block)
    out = Wt.rows('output.weight')
    logits = (hl @ out.T).float().cpu().numpy()
    np.savez(A.out, seq=np.array(seq, dtype=np.int64), logits=logits)
    print(f'peak {torch.cuda.max_memory_reserved() / 2 ** 20:.0f} MiB reserved, {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB live '
          f'({need:.0f} projected)')
    print(f'RESULT DONE: {A.out} ({A.last} rows at positions {N - A.last} .. {N - 1})')


if __name__ == '__main__':
    main()
