"""fp32 reference logits of a qwen3moe GGUF at depth (spec-own-quant's fp32 activation reference mode): every quantized weight
decoded exactly (caminus.own_dequant's formulas, ported to torch: the same float32 statements), activations, KV and sums in
fp32, RoPE angles in float64, layer-major over the whole prompt on one card (the residual [N, D] kept; per layer its
attention in query chunks and its experts one by one). Logits of the last M positions. Cross-check: at N = 128 against
tools/reference/q3moe_ref.py's float64 rows (mean KLD ~1e-7 expected).
The text: the tree's own docs and python sources (public), concatenated in sorted order, the first N tokens; or --seq.
  tools/guard.sh LOG 1800 python -I tools/reference/q3moe_ref32.py GGUF OUT.npz --n N [--last 64] [--seq SEQ.npz]
OUT.npz: seq (int64 [N]), logits (float32 [M, vocab]) for positions N - M .. N - 1. Needs ~4 GB of VRAM at N = 40,960."""
import argparse, glob, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
import torch
from caminus import ggufio
from caminus.bpe import BPE

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


class W:
    """a GGUF tensor's rows decoded on the card"""
    def __init__(self, F, dev):
        self.F, self.dev = F, dev

    def rows(self, name, r0=None, r1=None):
        t = self.F.T[name]; K = int(t.shape[0]); R, rb = t.rows()
        r0 = 0 if r0 is None else r0; r1 = R if r1 is None else r1
        if t.type == 'F32': return torch.from_numpy(np.asarray(t.data()).view(np.float32)[r0 * K:r1 * K].copy()).to(self.dev).view(r1 - r0, K)
        raw = torch.from_numpy(np.ascontiguousarray(np.asarray(t.data())[r0 * rb:r1 * rb])).to(self.dev)
        return {'Q4_K': q4_k, 'Q6_K': q6_k}[t.type](raw).view(r1 - r0, K)


def rms(x, w, eps):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w


def tree_text():
    fs = sorted(glob.glob(os.path.join(ROOT, 'docs', '**', '*.md'), recursive=True)) + sorted(glob.glob(os.path.join(ROOT, 'python', 'caminus', '**', '*.py'), recursive=True))
    return ''.join(f'\n# file: {os.path.relpath(f, ROOT)}\n' + open(f, encoding='utf-8').read() for f in fs)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('gguf'); ap.add_argument('out'); ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--last', type=int, default=64); ap.add_argument('--seq', default=None); ap.add_argument('--qchunk', type=int, default=256)
    A = ap.parse_args()
    dev = torch.device('cuda:0')
    F = ggufio.GGUFFile(A.gguf); Wt = W(F, dev)
    hp = lambda k: F.hp(k)
    nl, D, nh, nkv, hd = int(hp('block_count')), int(hp('embedding_length')), int(hp('attention.head_count')), int(hp('attention.head_count_kv')), int(hp('attention.key_length'))
    E, k, ff, eps, base = int(hp('expert_count')), int(hp('expert_used_count')), int(hp('expert_feed_forward_length')), float(hp('attention.layer_norm_rms_epsilon')), float(hp('rope.freq_base'))
    if A.seq: seq = [int(v) for v in np.load(A.seq)['seq']][:A.n]
    else: seq = BPE(F.meta).encode(tree_text())[:A.n]
    N = len(seq)
    if N < A.n: raise SystemExit(f'the text has {N} tokens < {A.n}')
    print(torch.cuda.get_device_name(0), f'N {N}', flush=True)
    te = F.T['token_embd.weight']; _, rb = te.rows()
    raw = torch.from_numpy(np.ascontiguousarray(np.asarray(te.data()).reshape(-1, rb)[np.array(seq)])).to(dev)
    x = q4_k(raw.view(-1)).view(N, D)
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
        kq = kk.permute(1, 0, 2); vq = v.permute(1, 0, 2)        # [nkv, N, hd]
        for a0 in range(0, N, A.qchunk):
            a1 = min(N, a0 + A.qchunk)
            qc = q[a0:a1].view(a1 - a0, nkv, nh // nkv, hd).permute(1, 2, 0, 3)          # [nkv, G, c, hd]
            s = torch.einsum('gjcd,gnd->gjcn', qc, kq[:, :a1]) / math.sqrt(hd)
            s.masked_fill_(torch.arange(a1, device=dev)[None, None, None, :] > torch.arange(a0, a1, device=dev)[None, None, :, None], float('-inf'))
            o = torch.einsum('gjcn,gnd->gjcd', torch.softmax(s, -1), vq[:, :a1])
            att[a0:a1] = o.permute(2, 0, 1, 3).reshape(a1 - a0, nh, hd)
            del s, o
        del q, kk, v, kq, vq
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
        del h, y
        if il % 8 == 0 or il == nl - 1: print(f'  layer {il} done', flush=True)
    hl = rms(x[N - A.last:], Wt.rows('output_norm.weight')[0], eps)
    out = Wt.rows('output.weight')
    logits = (hl @ out.T).float().cpu().numpy()
    np.savez(A.out, seq=np.array(seq, dtype=np.int64), logits=logits)
    print(f'RESULT DONE: {A.out} ({A.last} rows at positions {N - A.last} .. {N - 1})')


if __name__ == '__main__':
    main()
