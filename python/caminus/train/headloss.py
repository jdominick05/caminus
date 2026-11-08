"""The S1 adaptive head's log q(id) (s1.py AdaptiveHead.logq) with fused row kernels (kernels/headloss.hip).

Same math as the reference: hot logits L = bf16(h @ w^T) over the H hot rows + the escape row; a hot id's log q is
L[id] - lse(L); a tail id's is (L[escape] - lse(L)) + Z[t] - lse(Z) with Z = bf16(p(h) @ e^T) over the tail rows,
computed only for rows that hold a tail id. rocBLAS still does the GEMMs (bf16 logits, as the reference rounds them);
the fp32 casts, logsumexp, gathers and their backward become one pass each. Logit blocks are sized to stay in the
96 MB Infinity Cache between the GEMM that writes them and the kernel that reads them (hot: 1024 rows x 24.6k
x 2 B = 50 MB; tail: 2048 rows x 16,384-column blocks = 67 MB, so the K=112 GEMMs keep a usable shape). Like the reference's checkpointing, nothing [rows x vocab] is
kept for backward: the logits are recomputed.
Gradients: h always; w and e when they require grad; p's parameters through a nested autograd call."""
import ctypes
import torch
from ..device.hip import Module

_mod = {}
ALL_ROWS = [False]   # AdaptiveCE: run the tail pass on every row (no nonzero() sync); rows without a tail id get zero coefficients


def _m():
    dev = torch.cuda.current_device()
    if dev not in _mod: _mod[dev] = Module('headloss')
    return _mod[dev]


def _i(v): return ctypes.c_int(int(v))


def lse_gather(L, V, cols):
    """L [R, ld] bf16 contiguous, cols [R, k] int32 (-1 = none) -> lse [R] fp32, gathered [R, k] fp32"""
    R, ld = L.shape; k = cols.shape[1]
    lse = torch.empty(R, device=L.device); g = torch.empty(R, k, device=L.device)
    if R: _m().head_lse_gather_bf16((R,), (256,), L, _i(V), _i(ld), cols, _i(k), lse, g)
    return lse, g


def softmax_grad_(L, V, lse, cols, coef, total):
    """in place: L <- bf16(sum_i coef [j == cols_i] - total * exp(L - lse)) on the first V columns"""
    R, ld = L.shape; k = cols.shape[1]
    assert k <= 16
    if R: _m().head_softmax_grad_bf16((R,), (256,), L, _i(V), _i(ld), lse.contiguous(), cols, coef.contiguous(), _i(k), total.contiguous())
    return L


def _padded_bf16(w, mult=8):
    """bf16 copy of w [V, d] with zero rows up to a multiple of `mult` (keeps every logit row 16-byte aligned)"""
    V = w.shape[0]; ld = (V + mult - 1) // mult * mult
    out = torch.zeros(ld, w.shape[1], device=w.device, dtype=torch.bfloat16)
    out[:V] = w.detach().to(torch.bfloat16)
    return out


def _block_cols(cols, c0, n):
    """column ids relative to the block [c0, c0 + n), -1 outside it"""
    rel = cols - c0
    return torch.where((cols >= 0) & (rel >= 0) & (rel < n), rel, torch.full_like(rel, -1)).to(torch.int32).contiguous()


def _tail_lse_gather(pz, eb, tcols, chunk):
    """lse over all tail logits Z = pz @ eb^T and Z at tcols, walking (row chunk x column block) tiles so each GEMM
    has a usable shape (the tail's K is only 112) and each tile stays in the Infinity Cache; block results merge
    with logaddexp, and a gathered column lives in exactly one block"""
    RC, CB = chunk; R, k = tcols.shape; T = eb.shape[0]
    lse = torch.full((R,), float('-inf'), device=pz.device); g = torch.zeros(R, k, device=pz.device)
    for s in range(0, R, RC):
        for c0 in range(0, T, CB):
            blk = eb[c0:c0 + CB]
            Z = pz[s:s + RC] @ blk.t()
            lb, gb = lse_gather(Z, blk.shape[0], _block_cols(tcols[s:s + RC], c0, blk.shape[0]))
            lse[s:s + RC] = torch.logaddexp(lse[s:s + RC], lb); g[s:s + RC] += gb
    return lse, g


class AdaptiveLogQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, ids, w, e, head, chunk, tail_chunk):
        H = head.H; Vh = H + 1
        idx = head.lut[ids.clamp(0, head.lut.numel() - 1)]; hot = idx < H
        hcols = torch.where(hot, idx, torch.full_like(idx, H)).to(torch.int32).contiguous()   # tail ids read the escape
        wp = _padded_bf16(w); hb = h.detach().to(torch.bfloat16)
        N = h.shape[0]; lse_h = torch.empty(N, device=h.device); gh = torch.empty(N, ids.shape[1], device=h.device)
        for s in range(0, N, chunk):
            L = hb[s:s + chunk] @ wp.t()
            lse_h[s:s + chunk], gh[s:s + chunk] = lse_gather(L, Vh, hcols[s:s + chunk])
        lq = gh - lse_h[:, None]
        rows = (~hot).any(1).nonzero().squeeze(1)
        lse_z = None
        if rows.numel():
            tcols = torch.where(hot[rows], torch.full_like(idx[rows], -1), idx[rows] - H - 1).to(torch.int32).contiguous()
            eb = e.detach().to(torch.bfloat16)
            with torch.no_grad(): pz = head.p(h.detach()[rows]).to(torch.bfloat16)
            lse_z, gz = _tail_lse_gather(pz, eb, tcols, tail_chunk)
            lt = lq[rows] + gz - lse_z[:, None]                     # (L[esc] - lse) + Z[t] - lse_z
            lq = lq.index_copy(0, rows, torch.where(hot[rows], lq[rows], lt))
        ctx.save_for_backward(h, hot, hcols, lse_h, rows, lse_z if lse_z is not None else lse_h.new_empty(0), w, e)
        ctx.idx_tail = (idx[rows] - H - 1) if rows.numel() else None
        ctx.head, ctx.chunk, ctx.tail_chunk = head, chunk, tail_chunk
        ctx.autocast = (torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))
        return lq

    @staticmethod
    def backward(ctx, g):
        h, hot, hcols, lse_h, rows, lse_z, w, e = ctx.saved_tensors
        head, chunk, tail_chunk = ctx.head, ctx.chunk, ctx.tail_chunk; H = head.H; Vh = H + 1
        g = g.float().contiguous(); N = h.shape[0]
        wp = _padded_bf16(w); hb = h.detach().to(torch.bfloat16)
        dh = torch.zeros(N, h.shape[1], device=h.device)
        dw = torch.zeros(Vh, w.shape[1], device=w.device) if ctx.needs_input_grad[2] else None
        tot = g.sum(1)
        for s in range(0, N, chunk):
            L = hb[s:s + chunk] @ wp.t()
            softmax_grad_(L, Vh, lse_h[s:s + chunk], hcols[s:s + chunk], g[s:s + chunk], tot[s:s + chunk])
            dh[s:s + chunk] += (L @ wp).float()
            if dw is not None: dw += (L.t() @ hb[s:s + chunk])[:Vh].float()
        de = torch.zeros_like(e, dtype=torch.float32) if ctx.needs_input_grad[3] else None
        if rows.numel():
            gt = torch.where(hot[rows], torch.zeros_like(g[rows]), g[rows]).contiguous(); tt = gt.sum(1)
            tcols = torch.where(hot[rows], torch.full_like(ctx.idx_tail, -1), ctx.idx_tail).to(torch.int32).contiguous()
            eb = e.detach().to(torch.bfloat16); T = eb.shape[0]
            hr = h.detach()[rows].requires_grad_()
            with torch.enable_grad(), torch.autocast('cuda', dtype=ctx.autocast[1], enabled=ctx.autocast[0]):
                pz = head.p(hr)
            pzb = pz.detach().to(torch.bfloat16); R = rows.numel(); RC, CB = tail_chunk
            dpz = torch.zeros(R, eb.shape[1], device=h.device)
            for s in range(0, R, RC):
                for c0 in range(0, T, CB):
                    blk = eb[c0:c0 + CB]
                    Z = pzb[s:s + RC] @ blk.t()
                    softmax_grad_(Z, blk.shape[0], lse_z[s:s + RC], _block_cols(tcols[s:s + RC], c0, blk.shape[0]),
                                  gt[s:s + RC], tt[s:s + RC])
                    dpz[s:s + RC] += (Z @ blk).float()
                    if de is not None: de[c0:c0 + CB] += (Z.t() @ pzb[s:s + RC]).float()
            torch.autograd.backward(pz, dpz.to(pz.dtype))
            dh.index_add_(0, rows, hr.grad.float())
        return (dh.to(h.dtype), None, dw.to(w.dtype) if dw is not None else None,
                de.to(e.dtype) if de is not None else None, None, None, None)


class AdaptiveCE(torch.autograd.Function):
    """loss = sum(coef * log q(ids)) for a loss that is linear in log q (soft CE vs a teacher's top-k, hard CE): coef is
    d loss / d log q, known before the head runs, so the gradient is formed in the forward pass while each hot logit
    chunk is still in the Infinity Cache, and backward only scales it (no recompute of the hot logits). A hot id
    reads its column, a tail id the escape column, so the hot pass needs only coef; the tail keeps its two passes
    over the column blocks (lse first, then the gradient). Gradients: h; w and e when they require grad; p.weight
    when it requires grad (passed explicitly so it can be returned)."""
    @staticmethod
    def forward(ctx, h, ids, coef, w, e, pw, head, chunk, tail_chunk):
        H = head.H; Vh = H + 1; N, d = h.shape
        idx = head.lut[ids.clamp(0, head.lut.numel() - 1)]; hot = idx < H
        hcols = torch.where(hot, idx, torch.full_like(idx, H)).to(torch.int32).contiguous()
        coef = coef.float().contiguous(); tot = coef.sum(1)
        wp = _padded_bf16(w); hb = h.detach().to(torch.bfloat16)
        lse_h = torch.empty(N, device=h.device); gh = torch.empty(N, ids.shape[1], device=h.device)
        dh = torch.zeros(N, d, device=h.device)
        dw = torch.zeros(Vh, d, device=h.device) if ctx.needs_input_grad[3] else None
        for s in range(0, N, chunk):
            L = hb[s:s + chunk] @ wp.t()
            lse_h[s:s + chunk], gh[s:s + chunk] = lse_gather(L, Vh, hcols[s:s + chunk])
            softmax_grad_(L, Vh, lse_h[s:s + chunk], hcols[s:s + chunk], coef[s:s + chunk], tot[s:s + chunk])
            dh[s:s + chunk] += (L @ wp).float()
            if dw is not None: dw += (L.t() @ hb[s:s + chunk])[:Vh].float()
        lq = gh - lse_h[:, None]
        de = torch.zeros_like(e, dtype=torch.float32) if ctx.needs_input_grad[4] else None
        dpw = None
        rows = torch.arange(N, device=h.device) if ALL_ROWS[0] else (~hot).any(1).nonzero().squeeze(1)
        if N if ALL_ROWS[0] else rows.numel():
            tcols = torch.where(hot[rows], torch.full_like(idx[rows], -1), idx[rows] - H - 1).to(torch.int32).contiguous()
            eb = e.detach().to(torch.bfloat16); T = eb.shape[0]
            hr = h.detach()[rows].requires_grad_()
            with torch.enable_grad(): pz = head.p(hr)
            pzb = pz.detach().to(torch.bfloat16); R = rows.numel(); RC, CB = tail_chunk
            lse_z, gz = _tail_lse_gather(pzb, eb, tcols, tail_chunk)
            lq = lq.index_copy(0, rows, torch.where(hot[rows], lq[rows], lq[rows] + gz - lse_z[:, None]))
            gt = torch.where(hot[rows], torch.zeros_like(coef[rows]), coef[rows]).contiguous(); tt = gt.sum(1)
            dpz = torch.zeros(R, eb.shape[1], device=h.device)
            for s in range(0, R, RC):
                for c0 in range(0, T, CB):
                    blk = eb[c0:c0 + CB]
                    Z = pzb[s:s + RC] @ blk.t()
                    softmax_grad_(Z, blk.shape[0], lse_z[s:s + RC], _block_cols(tcols[s:s + RC], c0, blk.shape[0]),
                                  gt[s:s + RC], tt[s:s + RC])
                    dpz[s:s + RC] += (Z @ blk).float()
                    if de is not None: de[c0:c0 + CB] += (Z.t() @ pzb[s:s + RC]).float()
            want = [hr] + ([pw] if ctx.needs_input_grad[5] else [])
            with torch.enable_grad(): gr = torch.autograd.grad(pz, want, dpz.to(pz.dtype))
            dh.index_add_(0, rows, gr[0].float())
            if ctx.needs_input_grad[5]: dpw = gr[1].float()
        ctx.save_for_backward(dh, *(t for t in (dw, de, dpw) if t is not None))
        ctx.has = (dw is not None, de is not None, dpw is not None)
        ctx.dtypes = (h.dtype, w.dtype, e.dtype, pw.dtype)
        return (coef * lq).sum()

    @staticmethod
    def backward(ctx, gl):
        dh, *rest = ctx.saved_tensors; rest = iter(rest)
        out = [None if not on else next(rest) for on in ctx.has]
        sc = lambda t, dt: (t * gl).to(dt) if t is not None else None
        return (sc(dh, ctx.dtypes[0]), None, None, sc(out[0], ctx.dtypes[1]), sc(out[1], ctx.dtypes[2]),
                sc(out[2], ctx.dtypes[3]), None, None, None)


def soft_ce(head, h, tid, tlp, chunk=1024, tail_chunk=(2048, 16384)):
    """drop-in for AdaptiveHead.soft_ce (teacher top-k soft CE), gradient formed in the forward pass"""
    from s1 import INVALID
    valid = tid != INVALID; p = torch.where(valid, tlp.exp(), torch.zeros_like(tlp)).float()
    ok = p.sum(-1) > 0; p = p / p.sum(-1, keepdim=True).clamp_min(1e-9)
    coef = -p * ok[:, None] / ok.sum().clamp_min(1)
    return AdaptiveCE.apply(h, torch.where(valid, tid, torch.zeros_like(tid)), coef, head.w, head.e, head.p.weight,
                            head, chunk, tail_chunk)


def hard_ce(head, h, target, chunk=1024, tail_chunk=(2048, 16384)):
    """drop-in for AdaptiveHead.hard_ce (mean CE of the true next token), gradient formed in the forward pass"""
    coef = torch.full((h.shape[0], 1), -1.0 / max(1, h.shape[0]), device=h.device)
    return AdaptiveCE.apply(h, target[:, None], coef, head.w, head.e, head.p.weight, head, chunk, tail_chunk)


def adaptive_logq(head, h, ids, chunk=1024, tail_chunk=(2048, 16384)):
    """drop-in for AdaptiveHead.logq(h, ids): log q(id) [N, k]"""
    return AdaptiveLogQ.apply(h, ids, head.w, head.e, head, chunk, tail_chunk)
