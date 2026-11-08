"""DFlash 2 (incoai/Qwen3.8-Flash-Next-DFlash2, Q8_0 GGUF) as a K3 drafter beside the TP target (caminus.engine.flashnext_tp.FlashNextTP),
docs/specs/spec-drafters.md 1 (S2-S3). The torch drafter (caminus.decode.dflash_torch.DFlash2) stays as the comparison arm.

The graph is llama.cpp's (ignite-llama src/models/dflash.cpp graph<false>, common/speculative.cpp draft-dflash), op for op
bitwise against its dumps except the attention (tests/decode/test_dflash_k3_ops.py: every matvec, norm, RoPE, conv, the head, the
top-k values, the embedding; the attention within 1e-3 relative: llama.cpp's flash-attention kernel computes KQ in half):
- inject, over the T rows of each verify pass: the target's features (the stream mean of the residual entering the five
  target layers, exported by the target tape's OP_HC_AVG) read as [T, 5 * 2560] (q8_1 mode PERM), fc (Q8_0, K 12800), RMS,
  then per layer the k / v matvecs and the K norm + RoPE. Two placements: features on the XTX (FlashNextTP's default
  feature_card 0; smoke_factory): the inject runs on the XTX from its own copy of those weights (63 MB) while the XT runs
  the draft, and its K / V rows (fp32) reach the XT by two exchanges (layer 0, layers 1 ..) and are appended to the ring
  (f16); features on the XT (feature_card 1): the K norm op writes K and V straight into the ring (the XT then reads the
  inject's 63 MB in the same launch as the draft: ~0.1 ms more)
- draft, the block [anchor = the last token, MASK x (T - 1)] at positions n0 .. n0 + T - 1, 5 layers: RMS, the conv
  projection, OP_DF_CONV side 0, one q | k | v matvec, q norm + RoPE, k norm + RoPE into the ring (after the inject's rows),
  non-causal attention with llama.cpp's per-query sliding window 2048 (OP_ATTN_DEC flags; split over 64-position
  sub-blocks, one wave each, then a combine that also writes the q8_1 the o matvec reads), o with OP_DF_CONV side 1 + the
  residual as its epilogue (OP_DF_MVCONV), the same around the FFN (gate / up matvecs, the swiglu q8_1 glue, down + conv +
  residual); the output norm; the head: the target's Q6_K row shards on both cards, plus an XTX copy of the first
  HEAD_SHIFT rows of the XT's shard (balance), in three candidate halves; OP_DF_TOPK per piece, the exchange, OP_DF_SELECT
  (merge of the candidates by (value desc, id asc), the selector scores, the greedy lattice walk)
- the device accept (spec-drafters S3; kind 'step'): queued behind the target's verify on the same streams, OP_DF_TOPK
  work 2 takes each card's per-row first maximum of its verify logits shard (the host rule: torch.max, card 0 on a tie, a
  NaN above all), the pairs are exchanged, OP_DF_ACCEPT finds the target's tokens, the accepted drafts j and the bonus and
  writes the draft's position and anchor; then this pass's inject and the draft at the accepted position, in the same
  launch. One host round trip per pass (DFlashK3Drafter.generate); a verify bail reruns the verify, then the step
- the anchor's embedding by OP_DF_EMBED (IQ4_XS token_embd on the XT), the mask rows once at load
- norms in llama.cpp's own layouts (op_rms_mul_ll / op_qk_nr_ll; RMS_VAR / QK_VAR: the expression variant matched to its
  build element by element)

Placement: every drafter weight on the XT (card 1) except the head, whose shards are the target's (+ the XTX copy), and the
remote inject's weights. Launch kinds: 'step' (accept + inject + draft), 'fused' (inject + draft at pos_i + F, the generic
loop), 'draft', 'inject'; each an XT launch and (except a local 'inject') an XTX launch. Segments are claimed in tape
order, so the accept and the draft's first ops come before the inject's matvecs.
The code object kernels/k3_gguf_draft.hip holds only the drafter's ops (the trunk-only ops are compiled out: k3.DRAFT_OPS).
KV: a ring of `ring` positions of f16 K / V per layer, a fixed capacity.dflash_ring(window, T) = 4096 rows whatever max_pos
(before impl:dflash-ring: the power of two >= max(4096, max_pos), 5,120 MiB at 262,144): logical position p in row p &
mask, rows rewritten by position, so a rejected draft costs nothing (its rows are re-injected / overwritten by the next
pass) and a bailed launch is simply run again (everything it writes is recomputed from unchanged inputs). The attention
walks logical positions (lo .. hi, the slot is only an address), so any ring that holds the rows it reads gives the same
bits as a ring of max_pos rows. Why 4096 is enough, by construction: a launch at inject position i0 writes i0 .. i0 + 2 T - 1
(the inject rows, then the draft block at pos_d in i0 + 1 .. i0 + T) and reads from pos_d - (W - 1) >= i0 + 2 - W, so its
W + 2 T - 2 positions never share a slot (ring >= W + 2 T, asserted); positions only move forward between launches (a
pass advances by >= 1), so a write at q evicts q - ring < (the next launch's first read) - 2 T: never a live window row.
Positions move back only on restore (caminus.decode.longctx), which is why a snapshot copies the ring. The host also proves it
per launch (RingSlots): every ring slot a launch reads from outside its own writes must hold the injected row of exactly
that position, or the launch is refused before staging (StaleRingError); draft-block rows and rows of a launch that has
not finished are tagged invalid. Exchange epochs only grow (never reset: a reset epoch let a new generation's wait pass
on the previous one's flag, reading stale candidates).
Bounds: host checks first (k3.check_span: 'fused' / 'step' write i0 .. i0 + 2 T - 1), then k3_run's prologue guard on the
one staged position word, then OP_DF_ACCEPT's clamps (F, the anchor, pos_d: a clamp flags the record and the host
refuses) and OP_DF_SELECT's id clamps. Long context: the target runs the model's QSA past 2051 positions (FlashNextTP with
max_ctx > 2051), so max_pos follows the target's max_ctx; a target limited to 2051 positions (the dense tape) limits
max_pos the same way. Snapshots (snapshot / restore) copy the whole ring (80 MiB at 4096 rows; device to device on the
XT, capacity-checked first) and its slot tags with n_inj, so a restore to an earlier position puts back exactly the
rows its window reads: a stale ring after a restore is refused by the slot tags, never read."""
import os
import sys
import numpy as np
import torch
from ..engine import k3
from ..engine.flashnext import DirectReader, quants
from ..engine.flashnext_tp import FlashNextTP, Slots
from gguf import GGUFReader                      # (caminus.engine.flashnext puts gguf-py on the path)

from .. import config as _config   # noqa: E402
DRAFTER = _config.dflash()        # the DFlash2 Q8_0 GGUF: CAMINUS_DFLASH / config.toml, else under the model directory
MODULE = os.environ.get('K3_DRAFT_MODULE', 'k3_gguf_draft')   # (k3_gguf_draft_trace: the K3_TRACE measurement build)
from ..device.capacity import QSA_EXACT   # noqa: E402  (2051)
from ..device import capacity   # noqa: E402
CHUNK = 4096            # = DF_CHUNK in kernels/k3_draft_ops.h (OP_DF_TOPK's scan)
MERGE_MAX = 640         # OP_DF_TOPK merge: pairs per row held in LDS


class StaleRingError(RuntimeError):
    """a DFlash launch whose window would read a ring slot that does not hold the injected row of its own position (an
    alias left by a restore without the ring, a draft-block row, a launch that never finished): refused before staging"""


RING_HOOKS = 'ring-controls'   # K3_TEST_HOOKS value that arms DFlashK3.test_controls (test-only negative controls)


class RingSlots:
    """the host's map of the DFlash K / V ring (dflash_k3 module notes): tag[s] = the position whose INJECTED row slot s
    holds, or -1 (never written in this sequence, a draft-block row, or a launch still running / failed). Checked before
    every launch (check: the launch's reads outside its own writes), set after every successful launch (done), invalidated
    for a launch's whole write span before it starts (begin). All numpy slices (no Python loop): a few us per launch"""
    def __init__(self, ring, window, T):
        assert ring >= 2 and ring & (ring - 1) == 0, ring
        assert 1 <= T and 1 <= window and ring >= window + 2 * T, (ring, window, T)
        self.ring, self.mask, self.window, self.T = int(ring), int(ring) - 1, int(window), int(T)
        self.tag = np.full(self.ring, -1, dtype=np.int64)
        self.ar = np.arange(self.ring, dtype=np.int64)

    def reset(self):
        self.tag.fill(-1)

    def reads(self, kind, i0=None, n0=None):
        """[lo, hi) of the positions a launch reads from the ring that it does not write itself first: the draft's rows tt
        read max(0, p - (W - 1)) .. pos_d + T - 1 (p = pos_d + tt); the launch's own rows (the inject from i0, the block
        from pos_d) are written before its attention. 'step': pos_d is set on the device (i0 + 1 .. i0 + T): the lowest"""
        W = self.window
        if kind == 'inject': return 0, 0
        if kind == 'draft': return max(0, n0 - (W - 1)), n0
        if kind == 'fused': return max(0, n0 - (W - 1)), i0
        if kind == 'step': return max(0, i0 + 1 - (W - 1)), i0
        raise ValueError(kind)

    def writes(self, kind, i0=None, n0=None):
        """[lo, hi) of every position a launch may write ('step': up to the last possible draft block, i0 + 2 T - 1)"""
        T = self.T
        if kind == 'inject': return i0, i0 + T
        if kind == 'draft': return n0, n0 + T
        if kind == 'fused': return i0, n0 + T
        if kind == 'step': return i0, i0 + 2 * T
        raise ValueError(kind)

    def _slices(self, lo, hi):
        """the slot slices of positions [lo, hi) (hi - lo <= ring): (slot0, slot1, first position) per piece, <= 2"""
        assert 0 <= lo <= hi and hi - lo <= self.ring, (lo, hi)
        s0 = lo & self.mask; n = hi - lo; a = min(n, self.ring - s0)
        return [(s0, s0 + a, lo)] + ([(0, n - a, lo + a)] if n > a else [])

    def bad(self, lo, hi):
        """the first position in [lo, hi) whose slot does not hold its injected row, or None"""
        for s0, s1, p0 in self._slices(lo, hi):
            d = self.tag[s0:s1] - self.ar[:s1 - s0]
            w = np.flatnonzero(d != p0)
            if w.size: return int(p0 + w[0])
        return None

    def valid_runs(self, lo, hi):
        """[[start, end)] runs of positions in [lo, hi) whose slot holds their injected row"""
        pos = np.arange(lo, hi, dtype=np.int64)
        ok = self.tag[pos & self.mask] == pos
        out, i, n = [], 0, len(ok)
        while i < n:
            if ok[i]:
                j = i
                while j < n and ok[j]: j += 1
                out.append([int(lo) + i, int(lo) + j]); i = j
            else: i += 1
        return out

    def set(self, lo, hi, injected):
        for s0, s1, p0 in self._slices(lo, hi):
            if injected: self.tag[s0:s1] = self.ar[:s1 - s0] + p0
            else: self.tag[s0:s1] = -1

    def check(self, kind, i0=None, n0=None, what='DFlashK3'):
        lo, hi = self.reads(kind, i0, n0)
        if hi > lo:
            b = self.bad(lo, hi)
            if b is not None:
                s_ = b & self.mask
                raise StaleRingError(f'{what} {kind} (i0 {i0}, n0 {n0}): its window reads positions {lo} .. {hi - 1}, and ring slot '
                                     f'{s_} does not hold the injected row of position {b} (it holds '
                                     f'{"no injected row" if self.tag[s_] < 0 else f"position {int(self.tag[s_])}"}): refused '
                                     f'(a restore without the ring, a back step past the ring, or rows never injected)')

    def begin(self, kind, i0=None, n0=None):
        lo, hi = self.writes(kind, i0, n0); self.set(lo, hi, False)

    def done(self, kind, i0=None, n0=None):
        """after a successful launch: the inject rows i0 .. i0 + T - 1 hold injected rows, then the draft block n0 ..
        n0 + T - 1 (written after them: it overwrites the inject's rows from n0) holds draft rows"""
        T = self.T
        if kind in ('inject', 'fused', 'step'): self.set(i0, i0 + T, True)
        if kind in ('draft', 'fused', 'step'): self.set(n0, n0 + T, False)

    def state(self):
        return self.tag.copy()

    def load(self, tag):
        assert tag.shape == self.tag.shape and tag.dtype == self.tag.dtype, (tag.shape, tag.dtype)
        self.tag[:] = tag


def _seg(E, op, ntask, dep, rpt, W=None, gs=None, a=None, asum=None, y=None, rs=0.0, N=0, K=0, out_f32=1, work=0, pad1=0, pad2=0,
         aux=None, keep=()):
    p = lambda t: None if t is None else (t if isinstance(t, int) else t.data_ptr())
    sg = k3.Seg(op, ntask, dep, rpt, p(W), p(gs), p(a), p(asum), p(y), float(rs), N, K, out_f32, work, -1, pad1, pad2)
    if aux is not None: sg.aux = p(aux)
    E.keep += [t for t in keep if t is not None and not isinstance(t, int)]
    E.segs.append(sg)
    return len(E.segs) - 1


def df_conv(E, h, dyn, base, out, side, ks, gs, res=None, q8=None, dep=-1, rpt=1024):
    """OP_DF_CONV over h [T, C]: out = build_dflash2_conv(h, dyn, base, side) (+ res) (+ q8_1 copy)"""
    T, C = E.T, h.shape[-1]
    assert h.numel() == T * C and out.numel() == T * C and dyn.numel() == T * 2 * ks * (C // gs) and base.numel() == 2 * ks * C
    assert C % gs == 0 and rpt % 256 == 0 and (T * C) % 32 == 0 and (res is None or res.numel() == T * C)
    if q8 is not None: assert q8.numel() >= T * C // 32 * 36
    return _seg(E, k3.OP_DF_CONV, (T * C + rpt - 1) // rpt, dep, rpt, W=base, gs=dyn, a=h, asum=q8, y=out, N=T * C, K=C, work=ks,
                pad1=side, pad2=gs, aux=res, keep=(h, dyn, base, out, res, q8))


def df_mvconv(E, Wi, xq, out, dyn, base, res, side, ks, gs, dep=-1, rpt=8):
    """OP_DF_MVCONV: the Q8_0 matvec Wi . x (rows = channels) with OP_DF_CONV side `side` + res as its epilogue"""
    T, C = E.T, Wi['rows']
    assert Wi['fmt'] == 'Q8_0' and Wi['cols'] // 32 > 16 and 2 <= T <= 4 and ks == 2 and xq.dtype == torch.uint8
    assert out.numel() == T * C and res.numel() == T * C and dyn.numel() == T * 2 * ks * (C // gs) and base.numel() == 2 * ks * C
    return _seg(E, k3.OP_DF_MVCONV, (C + rpt - 1) // rpt, dep, rpt, W=Wi['w'], gs=dyn, a=xq, asum=base, y=out, N=C, K=Wi['cols'],
                out_f32=k3.GG_ACT_Q8, work=ks, pad1=side, pad2=gs, aux=res, keep=(Wi['w'], xq, out, dyn, base, res))


def df_topk_scan(E, logits, rows_c, row0, out, K, id_off, dep=-1):
    """OP_DF_TOPK work 0: per row r >= row0 of logits [T, rows_c] and chunk of CHUNK ids, the top K as (key, id) int pairs"""
    T = E.T; nch = (rows_c + CHUNK - 1) // CHUNK; nr = T - row0
    assert logits.numel() == T * rows_c and out.numel() >= nr * nch * K * 2 and 1 <= K <= 16 and 0 <= row0 < T
    assert 0 <= id_off and id_off + rows_c < (1 << 24) and nch * K <= MERGE_MAX
    return _seg(E, k3.OP_DF_TOPK, nr * nch, dep, 1, a=logits, y=out, N=rows_c, K=K, work=0, pad1=nch, pad2=row0, aux=int(id_off),
                keep=(logits, out)), nch


def df_topk_merge(E, pairs, nrows, n, out, K, dep=-1, parts=4):
    """OP_DF_TOPK work 1: per row, the top K of n (key, id) pairs -> (value, float id) pairs (each row ranked by `parts`
    workgroups, a slice each)"""
    assert pairs.numel() >= nrows * n * 2 and out.numel() >= nrows * K * 2 and n <= MERGE_MAX and parts >= 1
    return _seg(E, k3.OP_DF_TOPK, nrows * parts, dep, 1, a=pairs, y=out, N=n, K=K, work=1, pad1=parts, keep=(pairs, out))


def df_select(E, cand, halves, sel_prev, sel_next, gate, anchor, out, rank, kc, row_bytes, vocab, dep=-1):
    k = E.T - 1
    assert cand.numel() >= halves * k * kc * 2 and gate.numel() == E.T * rank and out.numel() >= k + 3 * k * kc
    assert rank % 32 == 0 and rank <= 768 and kc <= 16 and halves * kc <= 48 and k <= 8 and 2 * k * kc + kc + 1 <= 512
    assert 512 + 2 * k * halves * kc <= 1280 and 512 + rank <= 1280   # (LDS scratch: the merge's pairs, then cond, from 512)
    assert sel_prev.numel() == sel_next.numel() == vocab * row_bytes and 0 < vocab < (1 << 31)
    return _seg(E, k3.OP_DF_SELECT, 1, dep, 1, W=sel_prev, gs=sel_next, a=cand, asum=anchor, y=out, N=k, K=rank, work=row_bytes,
                pad1=halves, pad2=kc, out_f32=vocab, aux=gate, keep=(cand, sel_prev, sel_next, gate, anchor, out))


def df_argmax_scan(E, logits, rows_c, out, id_off, dep=-1):
    """OP_DF_TOPK work 2: per row of logits [T, rows_c] and chunk of CHUNK ids, the first maximum by the host rule (a NaN
    above everything, +0 == -0) as a (value, float(id_off + id)) pair: out [T][nch][2]"""
    T = E.T; nch = (rows_c + CHUNK - 1) // CHUNK
    assert logits.numel() == T * rows_c and out.numel() >= T * nch * 2 and 0 <= id_off and id_off + rows_c < (1 << 24)
    return _seg(E, k3.OP_DF_TOPK, T * nch, dep, 1, a=logits, y=out, N=rows_c, K=1, work=2, pad1=nch, aux=int(id_off),
                keep=(logits, out)), nch


def df_accept(E, blk, rec, mode, vocab, cap, pairs=None, nch=0, dep=-1):
    """OP_DF_ACCEPT: mode 0 sets the draft position pos_d = pos_i + F (blk[16] + blk[20]), mode 1 the device accept from
    both cards' argmax pairs ([2][T][nch]); writes blk[18] (pos_d) / blk[19] (anchor) and rec [T + 4] (kernels/k3_draft_ops.h)"""
    T = E.T
    assert blk.dtype == torch.int32 and blk.numel() >= 21 + T and rec.dtype == torch.int32 and rec.numel() >= T + 4 and T <= 8
    assert mode in (0, 1) and (mode == 0 or (pairs is not None and pairs.numel() >= 2 * T * nch * 2 and nch >= 1))
    assert 0 < vocab and 2 * T <= cap < (1 << 31) and 4 * T * nch + T <= 1280   # (mode 1: the pairs and a in the LDS scratch)
    return _seg(E, k3.OP_DF_ACCEPT, 1, dep, 1, a=pairs, asum=blk, y=rec, N=T, K=nch, work=mode, pad1=vocab, pad2=cap,
                keep=(blk, rec, pairs))


def df_embed(E, table, ids, out, n, D, vocab, row_bytes, dep=-1):
    assert out.numel() >= n * D and ids.dtype == torch.int32 and ids.numel() >= n and D % 256 == 0
    assert table.numel() == vocab * row_bytes
    return _seg(E, k3.OP_DF_EMBED, n, dep, 1, W=table, asum=ids, y=out, N=D, pad1=vocab, pad2=row_bytes, keep=(table, ids, out))


def q8_perm(E, feats, out, L, D, dep=-1):
    """OP_GG_Q8_1 mode 3: feats [L, T, D] quantized as [T, L * D] (the inject's fc input, target-layer order per token)"""
    T = E.T; n = T * L * D
    assert feats.numel() == n and out.numel() >= n // 32 * 36 and D % 32 == 0
    return _seg(E, k3.OP_GG_Q8_1, (n // 32 + 7) // 8, dep, 0, a=feats, y=out, N=n, K=D, out_f32=0, work=3, pad1=L, keep=(feats, out))


def kv_ring(E, k, v, cache, pos, mask, dep=-1):
    """OP_KV_APPEND f16 at rows (pos + t) & mask"""
    M = k.numel() // E.T; rows = mask + 1
    assert rows & mask == 0 and cache.numel() * cache.element_size() == rows * 4 * M and v.numel() == k.numel()
    return _seg(E, k3.OP_KV_APPEND, E.T, dep, 1, W=v, a=k, asum=pos, y=cache, N=M, out_f32=1, work=0, aux=int(mask),
                keep=(k, v, cache, pos))


def attn_nsub(window, T):
    """sub-splits of 64 positions per (row, head) the split attention can write (every row's window fits)"""
    return 2 * ((window + T - 1 + 127) // 128)


def attn_df(E, q, cache, pos, out, heads, kv_heads, dim, scale, mask, window, noncausal=True, dep=-1, q8=None, part=None):
    """OP_ATTN_DEC with the drafter's flags: non-causal over the pass's rows, per-query window, ring rows (q8: also the
    output's q8_1 blocks, D = 128 only). part (D = 128, 4 query heads per kv head): the split form, a partial pass into
    part (fp32 [T * heads * attn_nsub * 132]) then the combine (two segments; returns the combine's)"""
    assert q8 is None or (dim == 128 and q8.numel() >= q.numel() // 32 * 36)
    assert dim <= 512 and q.numel() == E.T * heads * dim and out.numel() == q.numel() and 0 < window < (1 << 31)
    assert cache.numel() * cache.element_size() == (mask + 1) * 4 * kv_heads * dim
    flags = (1 if noncausal else 0) | 2
    aux = int(window) | (int(mask) << 32)
    if part is None:
        return _seg(E, k3.OP_ATTN_DEC, E.T * heads, dep, 1, W=cache, gs=q8, a=q, asum=pos, y=out, rs=scale, N=E.T * heads, K=dim,
                    out_f32=1, work=0, pad1=kv_heads, pad2=flags, aux=aux, keep=(q, cache, pos, out, q8))
    ns = attn_nsub(window, E.T)
    assert dim == 128 and heads == 4 * kv_heads and part.dtype == torch.float32 and part.numel() >= E.T * heads * ns * 132
    assert mask + 1 >= window + E.T, 'the ring holds every row a window reads'
    a = _seg(E, k3.OP_ATTN_DEC, ns // 2 * E.T * kv_heads, dep, 1, W=cache, a=q, asum=pos, y=part, rs=scale, N=E.T * heads, K=dim,
             out_f32=1, work=0, pad1=kv_heads, pad2=flags | 4, aux=aux, keep=(q, cache, pos, part))
    return _seg(E, k3.OP_ATTN_DEC, E.T * heads, a, 1, gs=q8, a=part, asum=pos, y=out, N=E.T * heads, K=dim, out_f32=1, work=0,
                pad1=kv_heads, pad2=flags | 8, aux=aux, keep=(part, pos, out, q8))


class _Eng:
    pass


class DFlashK3:
    def __init__(self, tgt=None, path=DRAFTER, T=None, target_gguf=None, devs=(0, 1), ring=None, max_pos=None, limit_us=200_000,
                 max_retries=8, vram_check=True):
        """tgt: the FlashNextTP target (ntok T, feature_layers = the drafter's target layers, feature_card=1): its head shards
        (both cards), its token_embd bytes and its features are used. tgt=None: the solo test build on the XT alone (devs[1]):
        the head as one shard read from target_gguf, no XTX launch, no exchange (features fed through self.feats).
        vram_check: before loading or allocating anything, refuse (capacity.CapacityError) when the drafter's VRAM (its
        measured fixed part + the ring, capacity.drafter_need_mib) would push a card past its cap given its use now"""
        self.solo = tgt is None
        self.inj_remote = (not self.solo) and getattr(tgt, 'feature_card', 0) == 0
        self.tgt, self.devs = tgt, (tuple(tgt.devs) if tgt is not None else tuple(devs))
        self.dx = self.devs[1]                                   # the XT: the drafter's own card
        rd = GGUFReader(path, 'r'); self.WT = {t.name: t for t in rd.tensors}
        f = {x.name: x for x in rd.fields.values()}
        hp = lambda k: f[f'dflash.{k}'].contents()
        self.nl, self.D, self.ffn = int(hp('block_count')), int(hp('embedding_length')), int(hp('feed_forward_length'))
        self.nh, self.nkv, self.hd = int(hp('attention.head_count')), int(hp('attention.head_count_kv')), int(hp('attention.key_length'))
        self.eps = float(np.float32(hp('attention.layer_norm_rms_epsilon')))
        self.base = float(hp('rope.freq_base'))
        self.block, self.ks, self.gs = int(hp('block_size')), int(hp('conv_kernel_size')), int(hp('conv_group_size'))
        self.rank, self.topk, self.window = int(hp('selector_rank')), int(hp('selector_top_k')), int(hp('attention.sliding_window'))
        self.layers_tgt = [int(v) for v in hp('target_layers')]
        assert not bool(hp('attention.causal')), 'DFlash 2: a non-causal drafter'
        self.mask_id = int(f['tokenizer.ggml.mask_token_id'].contents())
        self.L_feat = len(self.layers_tgt)
        if tgt is not None:
            assert isinstance(tgt, FlashNextTP), 'DFlashK3: the K3 TP target'
            assert list(tgt.feature_layers) == self.layers_tgt, (tgt.feature_layers, self.layers_tgt)
            # the inject runs where the features are: feature_card 0 (the XTX: its own copy of the inject's weights, the K / V
            # rows sent to the XT's ring by two exchanges; the default) or 1 (the XT: K / V written into the ring directly)
            assert getattr(tgt, 'feature_card', 0) in (0, 1)
            self.T = tgt.ntok
            max_pos = tgt.max_ctx if max_pos is None else max_pos
            assert max_pos <= tgt.max_ctx
        else:
            assert T is not None and target_gguf is not None
            self.T = T
            max_pos = 2048 if max_pos is None else max_pos
        lim = getattr(tgt, 'qsa_limit', QSA_EXACT) if tgt is not None else None
        if lim is not None and max_pos > lim:   # (a target whose passes stop at QSA_EXACT: the dense tape)
            raise ValueError(f'DFlashK3: max_pos {max_pos} > {lim}: the target is exact only up to {lim} positions')
        assert 2 <= self.T <= min(8, self.block), self.T
        if ring is None: ring = capacity.dflash_ring(self.window, self.T)   # (fixed: never max_pos; the reserve counts the same)
        # spare slots beyond the window: a launch's reads and writes span W + 2 T - 2 positions (the module notes)
        assert ring & (ring - 1) == 0 and ring >= 2 * self.window and ring >= self.window + 2 * self.T, (ring, self.window, self.T)
        self.max_pos, self.ring, self.rmask = max_pos, ring, ring - 1
        self.ring_tags = RingSlots(ring, self.window, self.T)
        self._ctl = dict(no_ring_restore=False, no_tag_check=False)   # (test_controls: off unless armed)
        if vram_check:   # (before _load / _buffers: nothing allocated yet; the solo build's two 'cards' are one device)
            need = capacity.drafter_need_mib('dflash', None, max_pos, self.T, dflash_geom=(self.nl, self.nkv, self.hd, self.window), ring=ring)
            self.vram_rows = capacity.check((self.devs[0], self.dx), (0.0 if self.solo else need[0], need[1]),
                                            f'DFlashK3(max_pos={max_pos}, ring={ring})')
        self.k = self.T - 1
        self.limit_us, self.max_retries, self.bails = limit_us, max_retries, []
        import ctypes
        libm = ctypes.CDLL('libm.so.6'); libm.powf.restype = ctypes.c_float; libm.powf.argtypes = (ctypes.c_float, ctypes.c_float)
        self.theta_scale = libm.powf(float(self.base), -2.0 / self.hd)     # as flashnext (llama.cpp: powf(base, -2 / n_rot))
        self.scale = float(np.float32(1.0) / np.sqrt(np.float32(self.hd)))
        self.dr = DirectReader(path)
        self.path = path   # (the prompt cache's key fingerprints the drafter GGUF: caminus.cache.prompt)
        self._target_gguf = target_gguf
        self._load()
        self._buffers()
        self.engines = {kd: self._build(kd) for kd in ('inject', 'draft', 'fused', 'step')}
        torch.cuda.set_device(self.devs[0] if not self.solo else self.dx)
        self.reset()

    # -- weights -----------------------------------------------------------------------------------------------------
    def _raw(self, name, dev=None):
        t = self.WT[name]
        return self.dr.read(int(t.data_offset), int(t.n_bytes), device=f'cuda:{self.dx if dev is None else dev}')

    def _mat(self, name):
        t = self.WT[name]; assert t.tensor_type.name == 'Q8_0', (name, t.tensor_type.name)
        rows, cols = int(t.shape[1]), int(t.shape[0])
        return dict(w=self._raw(name), fmt='Q8_0', rows=rows, cols=cols, rb=int(t.n_bytes) // rows)

    def _f32(self, name):
        t = self.WT[name]; assert t.tensor_type.name == 'F32', (name, t.tensor_type.name)
        return self._raw(name).view(torch.float32)

    def _load(self):
        D = self.D
        self.fc, self.enc_norm, self.out_norm = self._mat('fc.weight'), self._f32('enc.output_norm.weight'), self._f32('output_norm.weight')
        assert (self.fc['rows'], self.fc['cols']) == (D, self.L_feat * D)
        self.sel_hidden = self._mat('selector_hidden.weight')
        sp, sn = self.WT['selector_predecessor.weight'], self.WT['selector_successor.weight']
        assert sp.tensor_type.name == sn.tensor_type.name == 'Q8_0' and int(sp.shape[0]) == self.rank
        self.vocab = int(sp.shape[1]); self.sel_rb = int(sp.n_bytes) // self.vocab
        self.sel_prev, self.sel_next = self._raw('selector_predecessor.weight'), self._raw('selector_successor.weight')
        self.L = []
        for il in range(self.nl):
            m = lambda n: self._mat(f'blk.{il}.{n}')
            g = lambda n: self._f32(f'blk.{il}.{n}')
            L = dict(attn_norm=g('attn_norm.weight'), ffn_norm=g('ffn_norm.weight'), q_norm=g('attn_q_norm.weight'),
                     k_norm=g('attn_k_norm.weight'), q=m('attn_q.weight'), k=m('attn_k.weight'), v=m('attn_v.weight'),
                     o=m('attn_output.weight'), gate=m('ffn_gate.weight'), up=m('ffn_up.weight'), down=m('ffn_down.weight'),
                     ca_base=g('attn_conv_base'), ca_proj=m('attn_conv_proj.weight'), cf_base=g('ffn_conv_base'),
                     cf_proj=m('ffn_conv_proj.weight'))
            assert L['ca_proj']['rows'] == 2 * self.ks * (D // self.gs) and L['ca_base'].numel() == 2 * self.ks * D
            # q | k | v rows in one buffer (one matvec segment for the draft; k | v contiguous for the inject); the separate
            # entries are views into it
            q, k, v = L['q'], L['k'], L['v']; assert q['rb'] == k['rb'] == v['rb'] and q['cols'] == k['cols'] == v['cols']
            w = torch.cat([q['w'], k['w'], v['w']]); nq, nk = q['w'].numel(), k['w'].numel()
            L['qkv'] = dict(w=w, fmt='Q8_0', rows=q['rows'] + k['rows'] + v['rows'], cols=q['cols'], rb=q['rb'])
            L['kv'] = dict(w=w[nq:], fmt='Q8_0', rows=k['rows'] + v['rows'], cols=q['cols'], rb=q['rb'])
            L['q']['w'], L['k']['w'], L['v']['w'] = w[:nq], w[nq:nq + nk], w[nq + nk:]
            self.L.append(L)
        if self.inj_remote:   # the inject's weights on the XTX (the features' card): fc, its norm, each layer's k / v rows, k norm
            d0 = self.devs[0]; cp = lambda t: t.to(f'cuda:{d0}')
            self.IW = dict(fc=dict(self.fc, w=cp(self.fc['w'])), enc_norm=cp(self.enc_norm),
                           k=[dict(L['k'], w=cp(L['k']['w'])) for L in self.L], v=[dict(L['v'], w=cp(L['v']['w'])) for L in self.L],
                           k_norm=[cp(L['k_norm']) for L in self.L])
        # the target's token_embd (OP_DF_EMBED) and head
        if self.solo:
            trd = GGUFReader(self.target_gguf_path(), 'r'); TT = {t.name: t for t in trd.tensors}; tdr = DirectReader(self.target_gguf_path())
        else:
            TT, tdr = self.tgt.T, self.tgt.dr
        te = TT['token_embd.weight']; assert te.tensor_type.name == 'IQ4_XS' and int(te.shape[0]) == D, te.tensor_type.name
        self.emb_vocab = int(te.shape[1]); self.emb_rb = int(te.n_bytes) // self.emb_vocab
        assert self.emb_vocab == self.vocab
        self.tok_embd = tdr.read(int(te.data_offset), int(te.n_bytes), device=f'cuda:{self.dx}')
        self._te = te
        if self.solo:   # the target's two row shards (vocab share 0.5, as FlashNextTP's Coder head), both on the XT
            ot = TT['output.weight']; rows = int(ot.shape[1]); rb = int(ot.n_bytes) // rows; vh = int(round(rows * 0.5))
            w = tdr.read(int(ot.data_offset), int(ot.n_bytes), device=f'cuda:{self.dx}').view(rows, rb)
            self.head = [dict(w=w[:vh].reshape(-1), fmt=ot.tensor_type.name, rows=vh, cols=int(ot.shape[0]), rb=rb),
                         dict(w=w[vh:].reshape(-1), fmt=ot.tensor_type.name, rows=rows - vh, cols=int(ot.shape[0]), rb=rb)]
            self.head_off = [0, vh]
        else:
            self.head = self.tgt.head_w
            self.head_off = [0, self.tgt.vh]
            assert self.head[0]['rows'] + self.head[1]['rows'] == self.vocab
        for c in (0, 1): assert self.head[c]['fmt'] == 'Q6_K' and self.head[c]['cols'] == D
        assert self.head[0]['rows'] + self.head[1]['rows'] == self.vocab
        # the draft head's balance (two cards): the XTX also takes the first HEAD_SHIFT rows of the XT's shard from its own copy
        # (the XT reads ~650 GB/s, the XTX ~950: the XT's half finished last and the XTX waited at the candidate exchange). The
        # target's shards stay as they are; candidates come in three halves [XTX shard | XTX copy | XT rest], merged by (value
        # desc, id asc) in OP_DF_SELECT: the same candidates in the same order, so the same drafts
        self.hshift = 0 if self.solo else int(self.HEAD_SHIFT)
        if self.hshift:
            h1, R = self.head[1], self.hshift; rb = h1['rb']
            assert 0 < R < h1['rows'] and h1['w'].dtype == torch.uint8 and h1['w'].numel() == h1['rows'] * rb and (R * rb) % 16 == 0
            wf = h1['w'].reshape(-1)   # (a view: the shard is contiguous); the copy: its first R rows, device to device
            self.hx = dict(w=wf[:R * rb].to(f'cuda:{self.devs[0]}'), fmt='Q6_K', rows=R, cols=D, rb=rb)
            self.hxt = dict(h1, w=wf[R * rb:], rows=h1['rows'] - R)

    def target_gguf_path(self):
        return self._target_gguf

    # -- buffers -----------------------------------------------------------------------------------------------------
    def _buffers(self):
        D, T, hd = self.D, self.T, self.hd
        B = lambda *s, dt=torch.float32, dev=None: torch.zeros(*s, dtype=dt, device=f'cuda:{self.dx if dev is None else dev}')
        q8b = lambda n, dev=None: B((n + 31) // 32 * 36, dt=torch.uint8, dev=dev)
        ndyn = 2 * self.ks * (D // self.gs)
        self.cache = [B(self.ring * 4 * self.nkv * hd // 2, dt=torch.int16) for _ in range(self.nl)]   # f16 [ring][K | V]
        if self.solo: self.feats = B(self.L_feat, T, D)
        else:
            self.feats = self.tgt.features
            assert self.feats.device.index == self.devs[0 if self.inj_remote else 1] and self.feats.shape == (self.L_feat, T, D)
        M = self.nkv * hd
        if self.inj_remote:   # the XTX's inject buffers; X = [layer][K | V][T][M] fp32 (the exchanges' payload), the XT's copy Xr
            d0 = self.devs[0]
            self.SX = dict(fq=q8b(T * self.L_feat * D, dev=d0), g0=B(T, D, dev=d0), gn=B(T, D, dev=d0), gq=q8b(T * D, dev=d0),
                           kr=[B(T, M, dev=d0) for _ in range(self.nl)], X=B(self.nl, 2, T, M, dev=d0), XO=B(self.nl, 2, T, M, dev=d0))
            self.Xr, self.XZ = B(self.nl, 2, T, M), B(self.nl, 2, T, M)
        S = self.S = dict(
            fq=q8b(T * self.L_feat * D), g0=B(T, D), gn=B(T, D), gq=q8b(T * D),
            ikv=[B(T, 2 * self.nkv * hd) for _ in range(self.nl)],
            xin=B(T, D), nn=B(T, D), nnq=q8b(T * D), dyna=B(T, ndyn), nc=B(T, D), ncq=q8b(T * D),
            qkv=B(T, (self.nh + 2 * self.nkv) * hd), qo=B(T, self.nh * hd), qs=B(T, self.nh * hd), ks=B(T, self.nkv * hd),
            vs=B(T, self.nkv * hd), ko=B(T, self.nkv * hd),
            att=B(T, self.nh * hd), attq=q8b(T * self.nh * hd), ob=B(T, D), xa=B(T, D), hn=B(T, D), hnq=q8b(T * D), dynf=B(T, ndyn),
            hc=B(T, D), hcq=q8b(T * D), hh=B(T, self.ffn), hu=B(T, self.ffn), hhq=q8b(T * self.ffn), ff=B(T, D), xb=B(T, D), emb=B(T, D),
            gate=B(T, self.rank), out=B(self.k + 3 * self.k * self.topk, dt=torch.int32), rec=B(T + 4, dt=torch.int32),
            attp=B(T * self.nh * attn_nsub(self.window, T) * 132))
        # the mask rows of the block input (rows 1 .. T - 1), once: the host dequantize the target's own rows use
        mrow = torch.from_numpy(quants.dequantize(np.ascontiguousarray(np.asarray(self._te.data)[[self.mask_id]]),
                                                  self._te.tensor_type).astype(np.float32).reshape(-1))
        S['xin'][1:].copy_(mrow.to(S['xin'].device).expand(T - 1, D))
        self.mask_row = mrow
        # per card: the head piece (output norm rows exchanged in, q8_1, logits, top-k scan / merge, the candidate exchange)
        # and the device accept's argmax pairs of the verify logits ([2 cards][T][nch]: amxp = this card's half, the
        # exchange's partial; amx = both halves after the exchange)
        self.H = {}
        self.nch_v = (self.head[0]['rows'] + CHUNK - 1) // CHUNK
        assert self.nch_v == (self.head[1]['rows'] + CHUNK - 1) // CHUNK, 'the accept reads [2][T][nch]: equal chunk counts per card'
        self.halves = 3 if self.hshift else 2
        hl = self.halves * self.k * self.topk * 2
        for c in (0, 1):   # (solo: both pieces on the XT)
            dev = self.dx if self.solo else self.devs[c]; hw = self.hxt if (c == 1 and self.hshift) else self.head[c]
            nch = (hw['rows'] + CHUNK - 1) // CHUNK
            hb = dict(embx=B(T, D, dev=dev), embq=q8b(T * D, dev=dev), lg=B(T, hw['rows'], dev=dev),
                      tk0=B(self.k * nch * self.topk * 2, dt=torch.int32, dev=dev),
                      tkf=B(hl, dev=dev), tkx=B(hl, dev=dev),
                      zero=B(T, D, dev=dev), amxp=B(2 * T * self.nch_v * 2, dev=dev), amx=B(2 * T * self.nch_v * 2, dev=dev))
            # the verify logits the accept reads: the target's own [T, rows] shard (solo: a test buffer)
            hb['vlg'] = B(T, self.head[c]['rows'], dev=dev) if self.solo else self.tgt.logits_half[c]
            assert hb['vlg'].shape == (T, self.head[c]['rows']) and hb['vlg'].dtype == torch.float32, hb['vlg'].shape
            if c == 0 and self.hshift:
                nx = (self.hshift + CHUNK - 1) // CHUNK
                hb.update(lgx=B(T, self.hshift, dev=dev), tk0x=B(self.k * nx * self.topk * 2, dt=torch.int32, dev=dev))
            self.H[c] = hb
        npl = max(T * D, hl, 2 * T * self.nch_v * 2, (self.nl - 1) * 2 * T * M)
        self.slots = {} if self.solo else {c: Slots(self.devs[c], self.N_EX, npl) for c in (0, 1)}

    # -- tapes -------------------------------------------------------------------------------------------------------
    # llama.cpp's own norm layouts (kernels/k3_draft_ops.h op_rms_mul_ll / op_qk_nr_ll; Seg.aux = the expression variant)
    RMS_VAR, QK_VAR = 2, 2   # bitwise llama.cpp's dumps (df_replay.py ops): the fma chain from x x; the fused rms_norm_mul_rope rotation
    FUSE_QKV, FUSE_RING, FUSE_ATTQ, FUSE_MVCONV = 1, 1, 1, 1   # tape fusions (off: the separate segments, the same bits)
    ATTN_SPLIT = 1   # the attention as the split partial pass + combine (kernels/k3_draft_ops.h op_attn_split; 0: op_attn_dec_df128)
    RPT_FFN, RPT_D, RPT_HEAD = 8, 8, 32   # rows per task: gate / up, the 2560- / 6144-row matvecs, the head shards (measured on the XT)
    HEAD_SHIFT = int(os.environ.get('DF_HEAD_SHIFT', 24576))   # rows of the XT's head shard the XTX computes from a copy (6 chunks)

    def _rms(self, E, x, w, out, dep=-1, q8=None):
        assert x.shape[-1] % 4 == 0 and x.shape[-1] // 4 <= 1024
        s = E.rms_mul(x, w, out, self.eps, dep=dep, q8=q8); E.segs[s].aux = int(self.RMS_VAR); return s

    def _qkn(self, E, x, heads, w, pos, out, dep=-1, tok_stride=None, ring=None):
        """per head RMS * w + NeoX RoPE (llama.cpp's layout). x: [T, tok_stride] rows (the heads at the front, or at x's view
        offset); ring = (cache, v rows): K (f16) and the heads' V written into the ring instead of out (the kv append fused)"""
        hd = self.hd; assert hd < 1024 and hd % 2 == 0
        ts = heads * hd if tok_stride is None else tok_stride
        if ring is None:
            assert out.numel() == self.T * heads * hd
            s = E.qk_nr(x, hd, heads, hd, w, pos, out, hd, self.theta_scale, self.eps, dep=dep); E.segs[s].aux = int(self.QK_VAR)
        else:
            cache, v = ring
            assert cache.numel() * cache.element_size() == self.ring * 4 * heads * hd
            s = E.qk_nr(x, hd, heads, hd, w, pos, v, hd, self.theta_scale, self.eps, dep=dep)
            E.segs[s].aux = int(self.QK_VAR) | 8 | (int(self.rmask) << 32); E.segs[s].gs = cache.data_ptr(); E.keep += [cache, v]
        E.segs[s].pad1 = ts
        return s

    def _mv(self, E, Wi, x, out, dep, rpt=16):
        return E.gg_matvec(Wi['w'], Wi['fmt'], Wi['rows'], Wi['cols'], Wi['rb'], x, out, rows_per_task=rpt, dep=dep)

    # exchanges per launch (their slot index: 0 the verify argmax pairs (kind 'step' only), 1 the output norm rows, 2 the
    # candidates, 3 / 4 the remote inject's rows); epochs base = token * N_EX
    N_EX = 5   # (3, 4: the remote inject's K / V rows of layer 0, of layers 1 ..)
    # where each layer's inject (k | v matvec + K norm / RoPE into the ring) sits in the fused tapes' claim order: before
    # the draft's layer il (il >= 1: after layer il - 1's q | k | v matvec; layer 0: after the draft's first conv). Segments
    # are claimed in tape order, so an inject emitted ahead of the draft holds the grid while the draft's chain waits
    INJ_INTERLEAVE = 1

    def _new_engine(self, dev, T):
        with torch.cuda.device(dev):
            E = k3.Engine(module=MODULE, grid=k3.resident_grid(MODULE), T=T, limit_us=self.limit_us)
            blk = torch.zeros(64, dtype=torch.int32, device='cuda'); E.params = blk[0:16]
        st = _Eng(); st.E, st.blk, st.dev = E, blk, dev
        # blk: [16] pos_i (inject / verify rows), [17] epoch base, [18] pos_d (the draft block), [19] anchor, [20] F (the
        # draft offset (kind 'fused') / the forced tokens of the verify (kind 'step')), [21 .. 21 + T) the verify's tokens
        st.pos_i, st.epoch, st.pos_d, st.anchor = blk[16:17], blk[17:18], blk[18:19], blk[19:20]
        st.hstage = torch.zeros(64, dtype=torch.int32, pin_memory=True)
        st.h_st = torch.zeros(k3.STATUS_WORDS, dtype=torch.int32, pin_memory=True)
        with torch.cuda.device(dev): st.ev = torch.cuda.Event(); st.ev0 = torch.cuda.Event(enable_timing=True); st.ev1 = torch.cuda.Event(enable_timing=True)
        return st

    def _inject_front(self, st):
        """the inject's features -> fc -> RMS (q8_1 of the encoder output): the layers' k | v matvecs read it"""
        E, S, D = st.E, self.S, self.D
        q = q8_perm(E, self.feats, S['fq'], self.L_feat, D)
        a = self._mv(E, self.fc, S['fq'], S['g0'], q, rpt=self.RPT_D)
        return self._rms(E, S['g0'], self.enc_norm, S['gn'], dep=a, q8=S['gq'])

    def _inject_layer(self, st, il, r):
        E, S, M = st.E, self.S, self.nkv * self.hd; L = self.L[il]
        kv_ = self._mv(E, L['kv'], S['gq'], S['ikv'][il], r, rpt=self.RPT_D)           # [T, K | V]
        return self._qkn(E, S['ikv'][il], self.nkv, L['k_norm'], st.pos_i, None, dep=kv_, tok_stride=2 * M,
                         ring=(self.cache[il], S['ikv'][il].view(-1)[M:]))

    def _inject_segs(self, st):
        r = self._inject_front(st)
        return [self._inject_layer(st, il, r) for il in range(self.nl)]

    def _inject_xtx(self, st):
        """the remote inject on the XTX's tape: fc, RMS, per layer k / v matvecs and the K norm + RoPE (fp32) into X, then the
        exchanges of X's layer 0 (slot 3) and of layers 1 .. (slot 4) to the XT (its partial: zeros, so Xr = 0 + X)"""
        E, X, W, M = st.E, self.SX, self.IW, self.nkv * self.hd; D = self.D
        q = q8_perm(E, self.feats, X['fq'], self.L_feat, D)
        a = self._mv(E, W['fc'], X['fq'], X['g0'], q, rpt=self.RPT_D)
        r = self._rms(E, X['g0'], W['enc_norm'], X['gn'], dep=a, q8=X['gq'])
        done = []
        for il in range(self.nl):
            k_ = self._mv(E, W['k'][il], X['gq'], X['kr'][il], r, rpt=self.RPT_D)
            v_ = self._mv(E, W['v'][il], X['gq'], X['X'][il, 1], r, rpt=self.RPT_D)
            q_ = self._qkn(E, X['kr'][il], self.nkv, W['k_norm'][il], st.pos_i, X['X'][il, 0], dep=k_)
            done.append(E.also(q_, v_))
        x3 = self._xchg(st, 0, X['X'][0], X['XO'][0], 3, done[0])
        j = done[1]
        for d in done[2:]: j = E.also(E.nop(1, 0, dep=j), d)
        x4 = self._xchg(st, 0, X['X'][1:], X['XO'][1:], 4, j)
        return x3, x4

    def _inject_xt_recv(self, st, il, state):
        """the XT's side of the remote inject for layer il: the exchange (layer 0: slot 3; layer 1: slot 4, layers 1 ..), then
        the f16 ring append of the received rows (Xr = 0 + X: the same values, a -0 arrives as +0; the attention's sums are
        the same bits). Returns the append's segment (the draft's K append waits for it: WAW on overlapping rows)"""
        E = st.E
        if il == 0: state['x3'] = self._xchg(st, 1, self.XZ[0], self.Xr[0], 3, -1)
        if il == 1: state['x4'] = self._xchg(st, 1, self.XZ[1:], self.Xr[1:], 4, -1)
        return kv_ring(E, self.Xr[il, 0], self.Xr[il, 1], self.cache[il], st.pos_i, self.rmask, dep=state['x3' if il == 0 else 'x4'])

    def _draft_segs(self, st, inject, dep0=-1):
        """inject: this tape also injects the verify rows at pos_i (the inject front emitted after the draft's first conv,
        each layer's k | v + K norm / RoPE before that layer's use: INJ_INTERLEAVE; else all of it first). dep0: the
        segment that sets pos_d / the anchor (OP_DF_ACCEPT), or -1. Segments are claimed in tape order: the accept and the
        draft's first ops come first so they are not queued behind the inject's ~500 matvec tasks"""
        E, S, D, hd, T = st.E, self.S, self.D, self.hd, self.T
        injdep = [None] * self.nl; inj = None; rs = {}
        if self.inj_remote:   # the inject itself is on the XTX's tape: here the exchanges and ring appends only
            lay = lambda il: self._inject_xt_recv(st, il, rs)
            front = lambda: None
        else:
            lay = lambda il: self._inject_layer(st, il, inj)
            front = lambda: self._inject_front(st)
        if inject and not self.INJ_INTERLEAVE:
            inj = front(); injdep = [lay(il) for il in range(self.nl)]
        dep = df_embed(E, self.tok_embd, st.anchor, S['xin'], 1, D, self.emb_vocab, self.emb_rb, dep=dep0)
        src = S['xin']
        for il, L in enumerate(self.L):
            r = self._rms(E, src, L['attn_norm'], S['nn'], dep=dep, q8=S['nnq'])
            a = self._mv(E, L['ca_proj'], S['nnq'], S['dyna'], r)
            c = df_conv(E, S['nn'], S['dyna'], L['ca_base'], S['nc'], 0, self.ks, self.gs, q8=S['ncq'], dep=a)
            if inject and self.INJ_INTERLEAVE and il == 0:
                inj = front(); injdep[0] = lay(0)
            # one q | k | v matvec; q's norm + RoPE to qo; k's norm + RoPE with V straight into the ring (after the inject's rows: WAW)
            Q, M = self.nh * hd, self.nkv * hd; qkvf = S['qkv'].view(-1); TS = Q + 2 * M
            if self.FUSE_QKV:
                m_ = self._mv(E, L['qkv'], S['ncq'], S['qkv'], c, rpt=self.RPT_D); mq = mk = mv_ = m_
            else:   # three matvecs into the same [T, q | k | v] rows (strided outputs: separate buffers then the same layout)
                mq = self._mv(E, L['q'], S['ncq'], S['qs'], c, rpt=self.RPT_D); mk = self._mv(E, L['k'], S['ncq'], S['ks'], c, rpt=self.RPT_D)
                mv_ = self._mv(E, L['v'], S['ncq'], S['vs'], c, rpt=self.RPT_D)
            if inject and self.INJ_INTERLEAVE and il + 1 < self.nl: injdep[il + 1] = lay(il + 1)
            qx, kx, vx, ts_q, ts_kv = ((S['qkv'], qkvf[Q:], qkvf[Q + M:], TS, TS) if self.FUSE_QKV else (S['qs'], S['ks'], S['vs'], Q, M))
            qd = self._qkn(E, qx, self.nh, L['q_norm'], st.pos_d, S['qo'], dep=mq, tok_stride=ts_q)
            if self.FUSE_RING:
                kd = self._qkn(E, kx, self.nkv, L['k_norm'], st.pos_d, None, dep=mk, tok_stride=ts_kv, ring=(self.cache[il], vx))
                if not self.FUSE_QKV: E.also(kd, mv_)
                if injdep[il] is not None:
                    if E.segs[kd].dep2p1: kd = E.also(E.nop(1, 0, dep=kd), injdep[il])   # (a third dependency through a nop)
                    else: E.also(kd, injdep[il])
            else:
                kd = self._qkn(E, kx, self.nkv, L['k_norm'], st.pos_d, S['ko'], dep=mk, tok_stride=ts_kv)
                if injdep[il] is not None: E.also(kd, injdep[il])
                assert not self.FUSE_QKV, 'FUSE_RING off needs FUSE_QKV off (contiguous v rows for kv_ring)'
                kd = E.also(kv_ring(E, S['ko'], S['vs'], self.cache[il], st.pos_d, self.rmask, dep=kd), mv_)
            at = attn_df(E, S['qo'], self.cache[il], st.pos_d, S['att'], self.nh, self.nkv, hd, self.scale, self.rmask,
                         self.window, dep=kd, q8=S['attq'] if self.FUSE_ATTQ else None, part=S['attp'] if self.ATTN_SPLIT else None)
            E.also(at - 1 if self.ATTN_SPLIT else at, qd)   # (the split's partial pass reads q)
            if not self.FUSE_ATTQ: at = E.gg_q8_1(S['att'], S['attq'], dep=at)
            if self.FUSE_MVCONV:
                c2 = df_mvconv(E, L['o'], S['attq'], S['xa'], S['dyna'], L['ca_base'], src, 1, self.ks, self.gs, dep=at, rpt=self.RPT_D)
            else:
                o_ = self._mv(E, L['o'], S['attq'], S['ob'], at, rpt=self.RPT_D)
                c2 = df_conv(E, S['ob'], S['dyna'], L['ca_base'], S['xa'], 1, self.ks, self.gs, res=src, dep=o_)
            r2 = self._rms(E, S['xa'], L['ffn_norm'], S['hn'], dep=c2, q8=S['hnq'])
            f_ = self._mv(E, L['cf_proj'], S['hnq'], S['dynf'], r2)
            c3 = df_conv(E, S['hn'], S['dynf'], L['cf_base'], S['hc'], 0, self.ks, self.gs, q8=S['hcq'], dep=f_)
            # gate / up as two Q8_0 matvecs (the multi-token fast path) and the swiglu glue into q8_1 (gg_silu(g) * u: the same
            # values as OP_GG_GU_DENSE, so the same q8_1 bytes; GU_DENSE's row walk was 95 us per layer vs ~2 x 30)
            g_ = self._mv(E, L['gate'], S['hcq'], S['hh'], c3, rpt=self.RPT_FFN)
            u_ = self._mv(E, L['up'], S['hcq'], S['hu'], c3, rpt=self.RPT_FFN)
            hq = E.also(E.gg_q8_1(S['hh'], S['hhq'], dep=g_, mul=S['hu']), u_)
            if self.FUSE_MVCONV:
                dep = df_mvconv(E, L['down'], S['hhq'], S['xb'], S['dynf'], L['cf_base'], S['xa'], 1, self.ks, self.gs, dep=hq, rpt=self.RPT_D)
            else:
                d_ = self._mv(E, L['down'], S['hhq'], S['ff'], hq, rpt=self.RPT_D)
                dep = df_conv(E, S['ff'], S['dynf'], L['cf_base'], S['xb'], 1, self.ks, self.gs, res=S['xa'], dep=d_)
            src = S['xb']
        return self._rms(E, src, self.out_norm, S['emb'], dep=dep)

    def _argmax_segs(self, st, c):
        """the device accept's first half on card c: the argmax pairs of its verify logits shard, then the pairs' exchange
        (solo: both shards scanned on the XT into both halves, no exchange). Returns the segment after which amx is full"""
        E = st.E; n = self.T * self.nch_v * 2
        if self.solo:
            ss = []
            for cc in (0, 1):
                hb = self.H[cc]
                s_, nch = df_argmax_scan(E, hb['vlg'], self.head[cc]['rows'], self.H[1]['amx'][cc * n:(cc + 1) * n], self.head_off[cc])
                ss.append(s_)
            return E.also(ss[1], ss[0])
        hb = self.H[c]
        s_, nch = df_argmax_scan(E, hb['vlg'], self.head[c]['rows'], hb['amxp'][c * n:(c + 1) * n], self.head_off[c])
        return self._xchg(st, c, hb['amxp'], hb['amx'], 0, s_)

    def _head_segs(self, st, c, dep):
        """card c's head piece after `dep` (the output norm rows on the XT; -1 on the XTX: the exchange waits for them). Solo:
        both pieces on the XT's tape, no exchange (each merge writes its half of the candidates directly)"""
        E, hb = st.E, self.H[c]
        if self.solo:
            x = E.gg_q8_1(self.S['emb'], hb['embq'], dep=dep)
            ms = []
            for cc in (0, 1):
                hw, hc_ = self.head[cc], self.H[cc]
                lg = self._mv(E, hw, hb['embq'], hc_['lg'], x, rpt=self.RPT_HEAD)
                sc, nch = df_topk_scan(E, hc_['lg'], hw['rows'], 1, hc_['tk0'], self.topk, self.head_off[cc], dep=lg)
                half = hb['tkx'][cc * self.k * self.topk * 2:(cc + 1) * self.k * self.topk * 2]
                ms.append(df_topk_merge(E, hc_['tk0'], self.k, nch * self.topk, half, self.topk, dep=sc))
            return x, E.also(ms[1], ms[0])
        part = self.S['emb'] if c == 1 else hb['zero']   # card 0's partial is zeros: 0 + emb (the same bits on both cards)
        x = self._xchg(st, c, part, hb['embx'], 1, dep)
        x = E.gg_q8_1(hb['embx'], hb['embq'], dep=x)
        hk = self.k * self.topk * 2
        # (head piece, its logits, its scan buffer, its first vocabulary id, its candidate half)
        pieces = ([(self.head[0], hb['lg'], hb['tk0'], 0, 0)] + ([(self.hx, hb['lgx'], hb['tk0x'], self.head_off[1], 1)] if self.hshift else [])
                  if c == 0 else [(self.hxt if self.hshift else self.head[1], hb['lg'], hb['tk0'], self.head_off[1] + self.hshift, self.halves - 1)])
        ms = []
        for hw, lgb, tk0, off, hh in pieces:
            lg = self._mv(E, hw, hb['embq'], lgb, x, rpt=self.RPT_HEAD)
            sc, nch = df_topk_scan(E, lgb, hw['rows'], 1, tk0, self.topk, off, dep=lg)
            ms.append(df_topk_merge(E, tk0, self.k, nch * self.topk, hb['tkf'][hh * hk:(hh + 1) * hk], self.topk, dep=sc))
        m = ms[0] if len(ms) == 1 else E.also(ms[1], ms[0])
        m = self._xchg(st, c, hb['tkf'], hb['tkx'], 2, m)
        return x, m

    def _xchg(self, st, c, partial, out, x, dep):
        return st.E.xchg(partial, out, self.slots[1 - c].slot(x), self.slots[c].slot(x), st.epoch, x, c, dep=dep)

    def _build(self, kind):
        """kind: 'inject' (XT only), 'draft' (the draft alone at the host's pos_d / anchor), 'fused' (the pending inject at
        pos_i, then the draft at pos_i + F), 'step' (the device accept of the verify that just ran at pos_i (its logits
        shards and F forced + drafted tokens), its inject, and the draft at the accepted position). Device bound guard
        (k3_run's prologue, one position word): pos_i < max_pos - T + 1 for 'inject'; pos_d for 'draft'; pos_i <= max_pos -
        2 T for 'fused' / 'step' (every draft block pos_i + 1 .. pos_i + T fits; OP_DF_ACCEPT clamps and flags anyway)"""
        sets = {}
        st = self._new_engine(self.dx, self.T); sets[1] = st; E = st.E
        cap2 = self.max_pos - self.T
        with torch.cuda.device(self.dx):
            if kind == 'inject':
                if self.inj_remote:
                    rs = {}
                    for il in range(self.nl): self._inject_xt_recv(st, il, rs)
                else: self._inject_segs(st)
                E._guard_pos(st.pos_i, self.max_pos)
            else:
                dep0 = -1
                if kind in ('fused', 'step'):
                    if kind == 'step':
                        g = self._argmax_segs(st, 1)
                        dep0 = df_accept(E, st.blk, self.S['rec'], 1, self.emb_vocab, self.max_pos, pairs=self.H[1]['amx'],
                                         nch=self.nch_v, dep=g)
                    else:
                        dep0 = df_accept(E, st.blk, self.S['rec'], 0, self.emb_vocab, self.max_pos)
                e = self._draft_segs(st, kind in ('fused', 'step'), dep0)
                hx, m = self._head_segs(st, 1, e)
                gq = hx   # the gate reads the q8_1 copy of the (exchanged) output norm rows
                gt = self._mv(E, self.sel_hidden, self.H[1]['embq'], self.S['gate'], gq)
                sl = df_select(E, self.H[1]['tkx'], self.halves, self.sel_prev, self.sel_next, self.S['gate'], st.anchor,
                               self.S['out'], self.rank, self.topk, self.sel_rb, self.vocab, dep=m)
                E.also(sl, gt)
                if kind == 'draft': E._guard_pos(st.pos_d, self.max_pos)
                else: E._guard_pos(st.pos_i, cap2)
            E.build()
        if (kind != 'inject' or self.inj_remote) and not self.solo:
            s0 = self._new_engine(self.devs[0], self.T); sets[0] = s0
            with torch.cuda.device(self.devs[0]):
                if kind == 'step': self._argmax_segs(s0, 0)
                if kind in ('inject', 'fused', 'step') and self.inj_remote: self._inject_xtx(s0)
                if kind != 'inject': self._head_segs(s0, 0, -1)
                # the same device bound as the XT's tape (the same staged words: both skip, or neither)
                if kind == 'draft': s0.E._guard_pos(s0.pos_d, self.max_pos)
                elif kind == 'inject': s0.E._guard_pos(s0.pos_i, self.max_pos)
                else: s0.E._guard_pos(s0.pos_i, cap2)
                s0.E.build()
        return sets

    # -- state -------------------------------------------------------------------------------------------------------
    def reset(self):
        """a new sequence: nothing injected yet (the ring's rows are rewritten by position; every slot tag invalid)"""
        self.pending = None      # the last verify pass's first position, not yet injected (fused into the next draft)
        self.n_inj = 0           # positions < n_inj have been injected
        self.gaps = []           # [lo, hi) position ranges given up by skip() / give_up() (no row of theirs): gap_in / window_gap
        self.poisoned = None
        # the target's launch epoch (tgt.token: one per pass launch, a retry too) when this drafter last took in a pass's
        # features (fed: inject() queued, or a step's inject done; pending_ep: the pending inject's). A target pass after it
        # rewrote the features: followed() is False, a pending inject is stale (drop_stale(): dropped, never launched)
        self.fed = self.pending_ep = None
        self.ring_tags.reset()
        if not hasattr(self, '_force_bail'): self._force_bail = None   # (a test hook armed before a generation survives its reset)
        if not hasattr(self, 'token'): self.token = 0                   # exchange epochs only grow (never reset)
        self.stats = dict(launches=0, inject_launches=0, draft_launches=0, fused=0, steps=0, gap_filler=0, stale_pending=0)

    def ring_bytes(self):
        return sum(c.numel() * c.element_size() for c in self.cache)

    # -- the window: what a sequence's state keeps of the drafter (one mechanism: the prompt cache's checkpoints and
    # longctx's snapshots both carry it; nothing outside reads the ring's slots or tags) ------------------------------
    @property
    def row_bytes(self):
        """bytes of one ring row of one layer (the K / V row of a position)"""
        return self.cache[0].numel() * self.cache[0].element_size() // self.ring

    def _ring_u8(self, l):
        return self.cache[l].view(torch.uint8).view(self.ring, self.row_bytes)

    @property
    def queued(self):
        """the pending inject's first position (queued by inject(), launched by the next draft / flush), or None"""
        return self.pending

    def injected_to(self):
        """positions below this are injected or queued: n_inj, or the pending inject's end (pending + T) when one is queued"""
        return self.n_inj if self.pending is None else max(self.n_inj, self.pending + self.T)

    def window_state(self, lo):
        """the window of positions [lo, n_inj) as this drafter holds it (no inject pending: flush first), the meta that goes
        with its rows: dict(n_inj, lo, hi = max(lo, n_inj), valid: the runs whose slots hold their injected row, gaps: the
        given-up ranges (skip / give_up) a window from lo can still read). A state at position pos keeps lo = pos - W (every
        later launch reads at or past it); its rows: export_window. (window, the attribute, is the attention window W)"""
        if self.pending is not None: raise RuntimeError('DFlashK3.window_state: an inject is pending (flush() first)')
        lo = max(0, int(lo)); hi = max(lo, int(self.n_inj))
        if hi - lo > self.ring: raise ValueError(f'DFlashK3.window_state: {lo} .. {hi} is more than the ring\'s {self.ring} rows')
        return dict(n_inj=int(self.n_inj), lo=lo, hi=hi, valid=self.ring_tags.valid_runs(lo, hi),
                    gaps=[[int(a), int(b)] for a, b in self.gaps if b > lo])

    def export_window(self, win, w):
        """window w's (window_state) ring rows into win (uint8 [nl, >= hi - lo, row_bytes] on this drafter's card; row i = position lo + i);
        queued on the current stream"""
        for s0, s1, p0 in self.ring_tags._slices(w['lo'], w['hi']):
            for l in range(self.nl): win[l, p0 - w['lo']:p0 - w['lo'] + s1 - s0].copy_(self._ring_u8(l)[s0:s1])

    def import_window(self, win, w):
        """window w's rows (export_window's win) back into the ring, every slot tag invalidated but the window's valid runs
        (a launch that reads outside them is refused, never wrong); the device copies queued on the current stream, the
        host state after the caller's sync: set_window_state"""
        self.ring_tags.reset()
        for s0, s1, p0 in self.ring_tags._slices(w['lo'], w['hi']):
            for l in range(self.nl): self._ring_u8(l)[s0:s1].copy_(win[l, p0 - w['lo']:p0 - w['lo'] + s1 - s0])
        for r0, r1 in w['valid']: self.ring_tags.set(r0, r1, True)

    def set_window_state(self, w):
        """the host state of window w: nothing pending, its n_inj and its gaps"""
        self.pending = None; self.n_inj = int(w['n_inj']); self.gaps = [tuple(g) for g in w.get('gaps', ())]

    def drop_pending(self):
        """a pending inject of a failed pass is dropped, never launched (its features are the failed pass's)"""
        self.pending = None

    def snapshot(self):
        """caminus.decode.longctx: the state of a sequence, with no inject pending: the window of the whole ring region
        [n_inj - ring, n_inj) (window_state / export_window: the rows of every position whose slot holds its injected row, the
        same rows and tags a full ring copy held; on the XT: the ring holds fewer rows than the context, so a later probe
        overwrites slots of this position's window) in a device copy, and n_inj and the gaps. The copy's VRAM is checked
        first (capacity.CapacityError: nothing copied). The exchange epoch (token) is never saved"""
        if self.poisoned: raise k3.ContextBoundsError(f'DFlashK3 snapshot: poisoned ({self.poisoned})')
        if self.pending is not None: raise RuntimeError('DFlashK3 snapshot: an inject is pending (flush() first)')
        w = self.window_state(int(self.n_inj) - self.ring)
        nb = self.nl * (w['hi'] - w['lo']) * self.row_bytes
        capacity.check((self.dx,), (nb / capacity.MIB,), f'DFlashK3.snapshot (a window copy of {nb / capacity.MIB:.0f} MiB)')
        with torch.cuda.device(self.dx):
            rows = torch.empty(self.nl, w['hi'] - w['lo'], self.row_bytes, dtype=torch.uint8, device=self.cache[0].device)
            self.export_window(rows, w); torch.cuda.synchronize()
        return dict(window=w, rows=rows, n_inj=w['n_inj'], geom=(self.ring, self.nl, self.cache[0].numel()))

    def restore(self, s):
        """put a snapshot back: its window's rows and slot tags (import_window) and its host state (set_window_state); a
        snapshot of another layout (an older caminus, another drafter) is refused"""
        if self.poisoned: raise k3.ContextBoundsError(f'DFlashK3 restore: poisoned ({self.poisoned})')
        if 'window' not in s or s.get('geom') != (self.ring, self.nl, self.cache[0].numel()):
            raise RuntimeError('DFlashK3 restore: a snapshot without this drafter\'s window (an older caminus, or another drafter): take it again')
        self.pending = None
        if self._ctl['no_ring_restore']:   # (test control: the old restore, n_inj only; ring and tags stay as they are)
            self.n_inj = s['n_inj']; return
        with torch.cuda.device(self.dx):
            self.import_window(s['rows'], s['window']); torch.cuda.synchronize()
        self.set_window_state(s['window'])

    def test_controls(self, no_ring_restore=False, no_tag_check=False):
        """TEST-ONLY negative controls of the ring's safety (refused unless K3_TEST_HOOKS=ring-controls): no_ring_restore:
        restore() puts back n_inj only (the pre-ring behaviour: the ring and its tags stay as the last probe left them);
        no_tag_check: launches skip the slot-tag check (a stale ring is then read). Both must make tests/decode/test_longctx.py's
        lc-ring controls fail (refused, then different drafts)"""
        if (no_ring_restore or no_tag_check) and os.environ.get('K3_TEST_HOOKS') != RING_HOOKS:
            raise ValueError(f'DFlashK3.test_controls: test-only, refused without K3_TEST_HOOKS={RING_HOOKS}')
        if no_ring_restore or no_tag_check:
            print(f'DFlashK3: TEST CONTROLS ARMED no_ring_restore={no_ring_restore} no_tag_check={no_tag_check}', file=sys.stderr, flush=True)
        self._ctl = dict(no_ring_restore=bool(no_ring_restore), no_tag_check=bool(no_tag_check))

    def _test_mark_injected(self, p0, p1):
        """test-only (tests/decode/test_dflash_k3_ops.py): the caller wrote positions p0 .. p1 - 1's injected rows into the ring
        itself (llama.cpp's K / V) or wants the accept tested on whatever the ring holds: tag them and count them injected"""
        k3.check_span(p0, p1 - p0, self.max_pos, what='DFlashK3._test_mark_injected')
        assert p1 - p0 <= self.ring
        if p1 > p0: self.ring_tags.set(p0, p1, True)
        self.n_inj = max(self.n_inj, p1)

    def force_bail(self, kind='fused', nth=1):
        """test hook: the nth launch of `kind` from now bails on its first attempt (a 1 us wait bound) and is run again"""
        assert kind in self.engines and nth >= 1
        self._force_bail = [kind, nth]

    # -- launches ----------------------------------------------------------------------------------------------------
    def _check_args(self, kind, i0, n0, anchor, F, toks):
        """host checks before staging (k3.ContextBoundsError / AssertionError): the rows each kind writes inside max_pos"""
        if self.poisoned: raise k3.ContextBoundsError(f'DFlashK3: refused until reset(): {self.poisoned}')
        if kind == 'inject': k3.check_span(i0, self.T, self.max_pos, what='DFlashK3 inject (inject rows)')
        if kind == 'draft':
            k3.check_span(n0, self.T, self.max_pos, what='DFlashK3 draft (draft rows)')
            assert 0 <= int(anchor) < self.emb_vocab, anchor
        if kind in ('fused', 'step'):   # the inject rows and every draft block i0 + 1 .. i0 + T: i0 .. i0 + 2 T - 1
            k3.check_span(i0, 2 * self.T, self.max_pos, what=f'DFlashK3 {kind} (inject rows + the draft block)')
            assert isinstance(F, (int, np.integer)) and 1 <= F <= self.T, F
        if kind == 'fused': assert 0 <= int(anchor) < self.emb_vocab, anchor
        if kind == 'step':
            assert len(toks) == self.T and all(0 <= int(t) < self.emb_vocab for t in toks), toks
        if not self._ctl['no_tag_check']: self.ring_tags.check(kind, i0, n0)   # (after the bounds: the slot arithmetic)

    def _start(self, kind, i0=None, n0=None, anchor=None, F=None, toks=None, armed=False, t_ev=False):
        """stage the launch's words and launch it on every card of the kind (no wait). t_ev: time the XT launch (events)"""
        sets = self.engines[kind]
        base = self.token * self.N_EX; self.token += 1
        for c, st in sets.items():
            h = st.hstage.numpy()
            h[1] = 0; h[16] = -1 if i0 is None else i0; h[17] = base; h[18] = -1 if n0 is None else n0
            h[19] = -1 if anchor is None else int(anchor); h[20] = 0 if F is None else int(F)
            if toks is not None: h[21:21 + self.T] = np.asarray(toks, dtype=np.int32)
            if armed: st.E.limit_us = 1
            with torch.cuda.device(st.dev):
                if t_ev and c == 1: st.ev0.record()
                st.blk.copy_(st.hstage, non_blocking=True)
                st.E.run(check=False)
                if t_ev and c == 1: st.ev1.record()
        for c, st in sets.items():
            with torch.cuda.device(st.dev):
                st.h_st.copy_(st.E.sync[2 * k3.MAXSEG:2 * k3.MAXSEG + k3.STATUS_WORDS], non_blocking=True)
                if c == 1 and kind != 'inject':
                    self._h_out[:self.S['out'].numel()].copy_(self.S['out'], non_blocking=True)
                    self._h_out[self.S['out'].numel():].copy_(self.S['rec'], non_blocking=True)
                st.ev.record()
        self.stats['launches'] += 1

    def _wait(self, kind, armed=False):
        """wait for the launch; raise on a device bound report (poisons) or an accept flag; returns the bail reports"""
        sets = self.engines[kind]
        for st in sets.values(): st.ev.synchronize()
        if armed:
            for st in sets.values(): st.E.limit_us = self.limit_us
        oob = [f'card {c}: {b}' for c, st in sets.items() for b in [k3.bounds_status(st.h_st.tolist())] if b]
        if oob:
            for st in sets.values():
                with torch.cuda.device(st.dev): torch.cuda.synchronize(); st.E.sync.zero_()
            self.poisoned = f'the device bound guard caught a position at DFlashK3 {kind} launch: ' + '; '.join(oob)
            raise k3.ContextBoundsError(self.poisoned)
        bad = []
        for c, st in sets.items():
            s_ = st.h_st.tolist()
            if s_[1]:
                with torch.cuda.device(st.dev):
                    torch.cuda.synchronize(); bad.append(f'card {c}: segment {s_[2]} (seen {s_[3]}); {st.E.stall_report()}'); st.E.sync.zero_()
        if not bad and kind in ('fused', 'step'):
            r = self.record()
            if r['flag']:
                self.poisoned = f'OP_DF_ACCEPT clamped a value at a DFlashK3 {kind} launch (flag {r["flag"]}: 1 F, 2 a token id: the anchor or a row argmax out of [0, vocab) or NaN, 4 pos_d): {r}'
                raise k3.ContextBoundsError(self.poisoned)
        return bad

    def _armed(self, kind):
        fb = self._force_bail
        if fb is not None and fb[0] == kind:
            fb[1] -= 1
            if fb[1] == 0: self._force_bail = None; return True
        return False

    def _launch(self, kind, **kw):
        i0, n0 = kw.get('i0'), kw.get('n0')
        self._check_args(kind, i0, n0, kw.get('anchor'), kw.get('F'), kw.get('toks'))
        armed = self._armed(kind)
        self.ring_tags.begin(kind, i0, n0)   # (its rows invalid until it succeeds: a raise leaves them so)
        for attempt in range(self.max_retries + 1):
            self._start(kind, armed=armed, **kw)
            bad = self._wait(kind, armed=armed); armed = False
            if not bad:
                self.ring_tags.done(kind, i0, n0 if kind != 'step' else self.record()['pos_d']); return
            self.bails.append((kind, kw.get('i0'), kw.get('n0'), attempt, bad))
            print(f'DFlashK3: a wait gave up in a {kind} launch (attempt {attempt + 1}), running it again: ' + '; '.join(bad),
                  file=sys.stderr, flush=True)
        raise RuntimeError(f'DFlashK3: a wait gave up in a {kind} launch in {self.max_retries + 1} attempts')

    @property
    def _h_out(self):
        if not hasattr(self, '_hout'):
            self._hout = torch.zeros(self.S['out'].numel() + self.S['rec'].numel(), dtype=torch.int32, pin_memory=True)
        return self._hout

    def record(self):
        """the last 'fused' / 'step' launch's OP_DF_ACCEPT record (host copy)"""
        r = self._h_out.numpy()[self.S['out'].numel():]; T = self.T
        return dict(a=r[:T].tolist(), j=int(r[T]), anchor=int(r[T + 1]), pos_d=int(r[T + 2]), flag=int(r[T + 3]))

    def drafts(self):
        return self._h_out[:self.k].tolist()

    def inject(self, i0):
        """queue the inject of the verify pass at positions i0 .. i0 + T - 1 (its features in the target's feature buffer);
        it runs fused into the next draft(), or now by flush(). One pending inject at most: the next target pass rewrites
        the features, so a second inject() before a draft() / flush() is refused (it would inject the wrong features)"""
        k3.check_span(i0, self.T, self.max_pos, what='DFlashK3.inject')
        self.drop_stale()
        if self.pending is not None:
            raise RuntimeError(f'DFlashK3.inject({i0}): the inject at {self.pending} is still pending (flush() or draft() first)')
        if i0 > self.n_inj: raise RuntimeError(f'DFlashK3.inject: positions {self.n_inj} .. {i0 - 1} were never injected')
        self.pending = i0
        self.pending_ep = self.fed = self._tgt_ep()

    def _tgt_ep(self):
        """the target's launch epoch now (None without a target: the solo build and host-only fakes never see a stale inject)"""
        t = getattr(self, 'tgt', None)
        return None if t is None else int(t.token)

    def followed(self):
        """True when this drafter took in the target's latest pass (no target pass ran after its last inject / step): a
        generation reported to caminus.cache.prompt ran through this drafter. False after a pass loop that bypassed it"""
        return self.fed is not None and self.fed == self._tgt_ep()

    def drop_stale(self):
        """a pending inject whose features a later target pass rewrote (a pass loop that bypassed this drafter ran after the
        generic loop left its last inject pending) is dropped, never launched: launching it would write another pass's
        features as these positions' rows. Positions [pending, n_inj) then hold no valid row of theirs (the pass before
        injected them with its tokens, rejected drafts among them): their tags are invalidated and n_inj = pending. Returns
        the dropped inject's position, or None"""
        if self.pending is None or self.pending_ep is None or self.pending_ep == self._tgt_ep(): return None
        p = int(self.pending)
        self._untag(p, max(p, int(self.n_inj)))
        self.n_inj = min(int(self.n_inj), p); self.pending = self.pending_ep = None
        self.stats['stale_pending'] += 1
        return p

    def _untag(self, lo, hi):
        """invalidate every slot tag holding a position in [lo, hi) (slots of other positions keep theirs)"""
        if hi > lo:
            t = self.ring_tags.tag; t[(t >= lo) & (t < hi)] = -1

    def give_up(self, lo, hi):
        """positions [lo, hi) are given up: their ring rows are not their injected rows of this sequence (a pass loop that
        bypassed this drafter ran them: caminus.cache.prompt gives them up when such passes are reported). A pending inject
        is flushed first (a stale one dropped: drop_stale), every slot tag holding a position in [lo, max(n_inj, hi)) is
        invalidated (rows at or past hi are rewritten by the next pass before any read: invalid tags there only make sure),
        the range kept in `gaps` (the loops draft filler while a window reads it: window_gap), n_inj = max(n_inj, hi). A
        launch reading one of these slots would be refused by the tags, never wrong. Host only"""
        lo, hi = int(lo), int(hi)
        k3.check_span(lo, max(0, hi - lo), self.max_pos, what='DFlashK3.give_up')
        self.flush()
        if hi <= lo: return
        self._untag(lo, max(hi, int(self.n_inj)))
        # (gaps no later window can read are dropped: windows start at most W - 1 below the front, the front >= hi - T)
        self.gaps = [g for g in self.gaps if g[1] > hi - 2 * self.window] + [(lo, hi)]
        self.n_inj = max(int(self.n_inj), hi)

    def skip(self, p):
        """positions [n_inj, p) are given up, never injected (a batched prefill longer than its feature export injects only
        its last rows: caminus.prefill.coder): after the pending inject runs, n_inj = max(n_inj, p) (give_up). Their ring slots hold
        no injected row of those positions (RingSlots tags), so a launch whose window reads one is refused (StaleRingError),
        never wrong; the range is kept in `gaps` so the loops draft filler instead of launching such a draft (window_gap: the
        prefill's own skips lie below every later window)"""
        p = int(p)
        k3.check_span(0, p, self.max_pos, what='DFlashK3.skip')
        self.flush()
        if p > self.n_inj: self.give_up(self.n_inj, p)

    def gap_in(self, lo, hi):
        """True when a given-up range (skip) meets positions [lo, hi)"""
        return any(a < hi and lo < b for a, b in self.gaps)

    def window_gap(self, n0):
        """True when a draft at n0 (its window max(0, n0 - (W - 1)) .. n0 - 1, RingSlots.reads) would read a position given
        up by skip(): the drafter then gives filler drafts (the verify rejects what is wrong: output == plain) instead of a
        launch the slot tags would refuse"""
        return bool(self.gaps) and self.gap_in(max(0, int(n0) - (self.window - 1)), int(n0))

    def flush(self):
        """run the pending inject alone (before the next target pass rewrites the features; one whose features a later pass
        rewrote already is dropped instead: drop_stale)"""
        self.drop_stale()
        if self.pending is None: return
        i0 = self.pending
        self._launch('inject', i0=i0); self.stats['inject_launches'] += 1
        self.n_inj = max(self.n_inj, i0 + self.T); self.pending = None

    def draft(self, anchor, n0):
        """k = T - 1 drafts after token `anchor` at position n0 (the context before n0 injected, or pending): one 'fused'
        launch when the pending inject's i0 has 1 <= n0 - i0 <= T and i0 + 2 T <= max_pos, else flush() + 'draft' (a stale
        pending inject is dropped first: drop_stale)"""
        self.drop_stale()
        i0 = self.pending
        n_inj = self.n_inj if i0 is None else max(self.n_inj, i0 + self.T)
        if n0 > n_inj: raise RuntimeError(f'DFlashK3.draft at {n0}: positions {n_inj} .. {n0 - 1} not injected')
        if i0 is not None and not (1 <= n0 - i0 <= self.T and i0 + 2 * self.T <= self.max_pos):
            self.flush(); i0 = None
        if i0 is not None:
            self._launch('fused', i0=i0, n0=n0, anchor=anchor, F=n0 - i0)
            r = self.record(); assert r['pos_d'] == n0 and r['anchor'] == anchor, (r, n0, anchor)
            self.stats['fused'] += 1
        else:
            self._launch('draft', n0=n0, anchor=anchor); self.stats['draft_launches'] += 1
        self.n_inj, self.pending = n_inj, None
        return self.drafts()

    def step_check(self, i0, F, toks):
        """step_start's host refusals, nothing staged or launched: the bounds and arguments, the ring's slot tags of the
        step's window, no pending inject, every position below i0 injected. DFlashK3Drafter._loop runs it before the
        target's launch_tokens, so a refused step never leaves a launched target pass without its finish_pass"""
        self._check_args('step', i0, None, None, F, toks)   # (the bounds first, then the ring's slot tags)
        if self.pending is not None: raise RuntimeError('DFlashK3.step: an inject is pending (flush() first)')
        if i0 > self.n_inj: raise RuntimeError(f'DFlashK3.step at {i0}: positions {self.n_inj} .. {i0 - 1} never injected')

    def step_start(self, i0, F, toks, t_ev=False):
        """the device accept path (spec-drafters S3): after the target's verify at i0 .. i0 + T - 1 was launched (tokens toks:
        F forced, then the drafts), queue on the same streams the accept, this pass's inject and the draft at the accepted
        position (no wait: step_wait). Every draft block lies in i0 + 1 .. i0 + 2 T - 1"""
        self.step_check(i0, F, toks)
        self._step = (i0, F, list(toks)); self._step_armed = self._armed('step')
        self.ring_tags.begin('step', i0)   # (i0 .. i0 + 2 T - 1 invalid until step_wait succeeds)
        self._start('step', i0=i0, F=F, toks=toks, armed=self._step_armed, t_ev=t_ev)

    def step_wait(self):
        """wait for step_start's launch: [] (done: record() / drafts()) or its bail reports (run step_start again)"""
        bad = self._wait('step', armed=self._step_armed); self._step_armed = False
        if bad: self.bails.append(('step', self._step[0], None, 0, bad)); return bad
        i0, F, toks = self._step; r = self.record()
        # the host's own check of the device's accept rule on the device's tokens (the argmax itself is checked in the tests)
        j = 0
        while j < self.T - F and toks[F + j] == r['a'][F - 1 + j]: j += 1
        assert r['j'] == j and r['anchor'] == r['a'][F - 1 + j] and r['pos_d'] == i0 + F + j, (r, self._step)
        self.ring_tags.done('step', i0, r['pos_d'])
        self.n_inj = max(self.n_inj, i0 + self.T); self.stats['steps'] += 1
        self.fed = self._tgt_ep()      # (after the target's own retries: finish_pass ran before this wait)
        return []

    def step_ms(self):
        """device time of the last timed step launch on the XT (accept + inject + draft), ms"""
        st = self.engines['step'][1]; return st.ev0.elapsed_time(st.ev1)

    def details(self):
        """the last draft's candidates [k][topk], their logits and the walked score rows (host copies)"""
        o = self._h_out.numpy(); k, kc = self.k, self.topk
        return (o[k:k + k * kc].reshape(k, kc).copy(), o[k + k * kc:k + 2 * k * kc].view(np.float32).reshape(k, kc).copy(),
                o[k + 2 * k * kc:k + 3 * k * kc].view(np.float32).reshape(k, kc).copy())


class DFlashK3Drafter:
    """DFlashK3 with caminus.decode.spec.generate's drafter interface. device_accept (default): spec.generate hands the whole
    generation to generate() below (the S3 loop: the accept on the device, one host round trip per pass) and
    caminus.decode.longctx.generate a continuation from a state to resume() (the same loop from any (i0, commit): the prompt
    cache's states, M6 fix-ups); off, and always in caminus.decode.spec.loop: the generic loop (inject queued, fused into the next draft;
    prompt passes' injects flushed at once, and a pass that asks for no draft from this drafter flushes it: spec.loop's
    k = 0 passes, a filler pass, or another drafter's pass in caminus.decode.mtp.PolicyDrafter). corrupt(pass_no, drafts) -> drafts
    and log (a list: the drafter's own drafts per generation pass) are test hooks for both paths"""
    def __init__(self, df, device_accept=True):
        self.df = df
        self.max_pos = df.max_pos
        self.device_accept = device_accept
        self.corrupt = None; self.log = None; self._n = 0
        self.check_argmax = False; self.argmax_checks = self.argmax_mismatch = 0   # test hook: device a vs the host rule
        self.step_trace = None   # a list: (anchor position, step launch device ms) per timed pass

    def reset(self):
        self.df.reset(); self._n = 0

    def snapshot(self):
        self.df.flush()   # (the fill's prefill() calls flush already: nothing pending after a prompt pass)
        return dict(df=self.df.snapshot(), n=self._n)

    def restore(self, s):
        self.df.restore(s['df']); self._n = s['n']

    # -- the window (DFlashK3.window_state: the prompt cache's checkpoints carry it), with this wrapper's draft count -----------
    def window_state(self, lo):
        return dict(self.df.window_state(lo), n=int(self._n))

    def export_window(self, win, w):
        self.df.export_window(win, w)

    def import_window(self, win, w):
        self.df.import_window(win, w)

    def set_window_state(self, w):
        self.df.set_window_state(w); self._n = int(w.get('n', 0))

    def inject(self, i0, feats):
        assert feats is self.df.feats, 'DFlashK3Drafter: the target exports its features into the drafter-side buffer'
        self.df.inject(i0)

    def skip_inject(self, p):
        """positions below p are never injected (DFlashK3.skip: a long batched prefill injects its last rows only)"""
        self.df.skip(p)

    def prefill(self, toks, n):
        self.df.flush()

    def flush(self):
        """a pass that asks no draft of this drafter (spec.loop's k = 0 passes, caminus.decode.mtp.PolicyDrafter's idle drafter):
        the pending inject runs now, before the next target pass rewrites its features (else the next inject() is refused:
        'the inject at X is still pending')"""
        self.df.flush()

    def _hook(self, d):
        if self.log is not None: self.log.append(list(d))
        if self.corrupt is not None: d = list(self.corrupt(self._n, list(d)))
        self._n += 1
        return d

    def __call__(self, toks, k):
        """spec.loop's draft call: k drafts after toks (the anchor toks[-1] at n0 = len(toks) - 1, the next pass's first
        position i0 when the pass forces one token). The drafter's block always gives T - 1 drafts, from the injected rows of
        every position before n0, so it drafts only when k = T - 1 (one forced token: positions < n0 have run, their inject
        pending or done), n0 is injected and the block fits max_pos: generate()'s rule for its first draft. Otherwise (a
        prompt tail or given tokens of r >= 2 forced tokens: positions i0 .. n0 - 1 have not run yet, and their rows may hold
        the pending inject's rows of rejected drafts) the pending inject is flushed (the coming pass rewrites the features)
        and the drafts are filler, the last token k times (the verify rejects what is wrong: output == plain). Filler is not
        logged or counted by the test hooks (generate() does not log its filler either). (It used to assert k == T - 1: the
        generic loop crashed at T = 4 on prompt lengths 2 / 3 mod 4 once 7d2f0a2 made DFlash the default)"""
        df = self.df
        assert 0 <= k <= df.k, (k, df.k)
        if k == 0: return []
        n0 = len(toks) - 1
        n_inj = df.injected_to()
        if k == df.k and n0 <= n_inj and n0 + df.T <= df.max_pos:
            if not df.window_gap(n0): return self._hook(df.draft(int(toks[-1]), n0))
            df.stats['gap_filler'] += 1    # (the window reads positions never injected: filler until it has passed them)
        df.flush()
        return [int(toks[-1])] * k

    def generate(self, m, prompt, n_new, stats=None, prefill=None):
        """caminus.decode.spec.generate with the device accept: prompt passes as there (verify, inject flushed; prefill: a
        caminus.prefill.coder.Prefill of m takes the prompt's positions [0, P) first, spec.prefill_span: the batched prefill and
        its features' inject, then commit 0); from the last
        prompt pass on, each pass is the target's verify and the drafter's 'step' launch queued behind it on the same
        streams (the accept from the verify's logits shards, its inject, the next drafts), then one wait. Greedy: output ==
        plain greedy. A target bail reruns the verify (commit 0) and then the step; a step bail reruns the step. Passes
        whose draft blocks would leave max_pos fall back to the host accept (verify_tokens) + draft(). stats: as
        spec.generate, plus step_seconds (verify + step wall) and draft_seconds = the step launches' XT device time"""
        import time
        from . import spec
        df = self.df; T = m.ntok; assert T == df.T
        if not prompt: raise ValueError('generate: an empty prompt')
        spec.check_capacity(m, len(prompt), n_new, self)
        P = spec.prefill_span(prefill, prompt, self)      # (validated with the whole prompt before the reset)
        m.reset(); self.reset()
        toks = list(prompt); i0 = 0; t0 = time.perf_counter()
        pst = None
        if P: pst = prefill.run(toks, 0, 0, P, self, validated=True); i0 = P   # (committed state: the next pass commits 0)
        toks, _, _ = self._loop(m, toks, i0, 0, len(prompt) + n_new, stats, t0, pst)
        return toks[:len(prompt) + n_new]

    def resume(self, m, st, toks, n_new, stats=None):
        """the device-accept loop from an engine state (M6: a state caminus.cache.prompt prepared or restored, or any state
        longctx.generate continues; longctx.generate hands a device_accept drafter's continuation here): positions below
        st['i0'] have run (the last pass's first st['commit'] rows uncommitted), toks[st['i0']:] are given tokens, then n_new
        generated tokens. The drafter is in the state the sequence left it (every position below st['i0'] injected, pending,
        or given up by skip(): drafts whose window reads a given-up position are filler, window_gap). Same passes as the
        generic loop (spec.loop) from the same state: the prompt passes (k = 0, the pending inject flushed first), the last
        given token's pass with DFlash's draft when it forces one token, then verify + 'step' per pass. Zero given tokens (an
        exact cache hit) rerun the last pass's row of position len(toks) - 1 (spec.resume_point: refused on the host from
        commit 0). Returns (toks + exactly n_new tokens, the state at the last returned token: drafts accepted past the total
        dropped from it, as longctx.generate's). The inject of the last pass is flushed (generate()'s end)"""
        import time
        from . import spec
        df = self.df; T = m.ntok; assert T == df.T
        toks = [int(x) for x in toks]; total = len(toks) + int(n_new)
        spec.check_capacity(m, len(toks), n_new, self)             # up front: max_ctx, the pass limit, the drafter's rows
        if len(toks) >= total: return toks, dict(st)
        i0, commit = spec.resume_point(toks, st['i0'], st['commit'], T)   # (host only: before any launch)
        out, i0, commit = self._loop(m, toks, i0, commit, total, stats, time.perf_counter(), None)
        extra = max(0, len(out) - total)
        assert extra == 0 or (len(out) == i0 + 1 and commit - extra >= 1), (len(out), total, i0, commit)
        return out[:total], dict(st, i0=i0 - extra, commit=commit - extra)

    def step_ok(self, i0):
        """True when the pass at i0 can run with the device accept ('step'): its next draft block fits max_pos and that
        draft's window reads no given-up position (else the host accept and the generic draft path)"""
        df = self.df
        return i0 + 2 * df.T <= df.max_pos and not df.window_gap(i0 + 1)

    def device_step(self, m, seq, i0, commit, F, timed=True):
        """one pass of the device-accept loop (_loop's, and caminus.serve.k3_session's per-pass form): the target's verify at
        i0 .. i0 + T - 1 (seq[i0:i0 + T]: F forced tokens, then the drafts) and the drafter's 'step' queued behind it (the
        accept from the verify's logits shards, this pass's inject, the next drafts at the accepted position), then one wait.
        The step's host refusals run before the target launches (never a launched pass left unfinished); a target bail reruns
        the verify (finish_pass) and then the step; a step bail reruns the step (at most max_retries + 1 attempts). When the
        target's retries are exhausted (finish_pass raises), the step queued behind its first attempt is waited for (its
        result dropped) before the error propagates, so nothing is left in flight. Returns (record, next drafts, timed: the
        step launch's device time is valid for step_ms())"""
        df, T = self.df, self.df.T
        tm = bool(timed)
        df.step_check(i0, F, seq[i0:i0 + T])           # (host refusals before the target launches: never a pass left unfinished)
        m.launch_tokens(seq, i0, commit)
        df.step_start(i0, F, seq[i0:i0 + T], t_ev=tm)
        try:
            retried = m.finish_pass(i0, commit)        # waits for the target; reruns a bailed verify (commit 0)
        except BaseException:
            try: df.step_wait()                        # (the step behind the failed verify: drained, its result dropped)
            except BaseException: pass
            raise
        bad = df.step_wait(); n_bad = 1 if bad else 0
        while retried or bad:                          # the step again on the final verify (features / logits)
            # (a rerun forced by the target's rerun alone always runs; bailed steps: at most max_retries + 1 attempts)
            if n_bad > df.max_retries:
                raise RuntimeError(f'DFlashK3: a wait gave up in the step launch at {i0} in {n_bad} attempts: ' + '; '.join(bad))
            df.step_start(i0, F, seq[i0:i0 + T], t_ev=False); tm = False
            retried = False; bad = df.step_wait(); n_bad += 1 if bad else 0
        return df.record(), df.drafts(), tm

    def _loop(self, m, toks, i0, commit, total, stats, t0, pst):
        """the device-accept pass loop from the state (i0, commit) with toks[i0:] given, until len(toks) >= total: returns
        (toks, i0, commit) after the last pass (toks may pass total by its accepted drafts)"""
        import time
        df = self.df; T = m.ntok
        passes = acc = gen_passes = 0; t_gen = None; n_given = len(toks)
        t_step = t_dev = t_verify = 0.0; n_dev = 0
        while len(toks) - i0 > T:                          # prompt passes: no drafts
            df.flush()                                     # (a pending inject runs before the pass rewrites its features: spec.loop's k = 0)
            m.verify_tokens(toks, i0, commit); df.inject(i0); df.flush()
            commit = T; i0 += T; passes += 1
        F = len(toks) - i0                                 # the last prompt pass: F forced tokens (1 .. T)
        n_inj = df.injected_to()
        if F == 1 and i0 <= n_inj and i0 + T <= df.max_pos and not df.window_gap(i0): d = self._hook(df.draft(toks[-1], i0))
        else:                                              # (any guess is safe: the verify checks it)
            if F == 1 and i0 <= n_inj and i0 + T <= df.max_pos: df.stats['gap_filler'] += 1
            df.flush(); d = [int(toks[-1])] * (T - F)
        if F < T and t_gen is None: t_gen = time.perf_counter()
        while len(toks) < total:
            k = T - F; d = list(d[:k])
            seq = toks[:i0 + F] + d
            if i0 + T > m.max_ctx: raise k3.ContextBoundsError(f'generate: a pass at {i0} past max_ctx {m.max_ctx}')
            ts = time.perf_counter()
            if self.step_ok(i0):                           # (the step's next draft reads from i0 + 2 - W)
                r, nd, tm = self.device_step(m, seq, i0, commit, F, timed=True)
                j, bonus = r['j'], r['anchor']
                if self.check_argmax:   # the verify's logits shards are untouched until the next target pass
                    (v0, x0), (v1, x1) = [[t.cpu().numpy() for t in torch.max(m.logits_half[c], dim=1)] for c in (0, 1)]
                    host = [int(x0[t]) if (v0[t] >= v1[t] or v0[t] != v0[t]) else m.vh + int(x1[t]) for t in range(T)]
                    self.argmax_checks += 1; self.argmax_mismatch += host != r['a']
                if tm:
                    ms_ = df.step_ms(); t_dev += ms_ / 1e3; n_dev += 1
                    if self.step_trace is not None: self.step_trace.append((i0 + F + j, ms_))   # (the draft's anchor position, ms)
            else:                                          # near max_pos (or a window over a given-up range): the host accept and the generic draft path
                tv = time.perf_counter(); a = m.verify_tokens(seq, i0, commit); t_verify += time.perf_counter() - tv
                df.inject(i0)
                j = 0
                while j < k and d[j] == a[F - 1 + j]: j += 1
                bonus = a[F - 1 + j]; nd = None
            t_step += time.perf_counter() - ts
            gen_passes += k > 0; acc += j if k > 0 else 0
            toks = toks[:i0 + F] + d[:j] + [bonus]
            commit = F + j; i0 += F + j; passes += 1; F = 1
            if t_gen is None: t_gen = time.perf_counter()
            if len(toks) >= total: break
            if nd is None:
                if i0 + T <= df.max_pos and not df.window_gap(i0): nd = df.draft(bonus, i0)
                else:
                    if i0 + T <= df.max_pos: df.stats['gap_filler'] += 1
                    df.flush(); nd = [bonus] * (T - 1)
            else: assert r['pos_d'] == i0, (r, i0)
            d = self._hook(nd)
        df.flush()
        if stats is not None:
            t1 = time.perf_counter()
            stats.update(passes=passes, accepted=acc, seconds=t1 - t0, gen_seconds=t1 - (t_gen or t1), gen_passes=gen_passes,
                         draft_seconds=t_dev * gen_passes / max(1, n_dev), verify_seconds=t_step - t_dev * gen_passes / max(1, n_dev),
                         inject_seconds=0.0, gen_tokens=len(toks) - n_given, step_seconds=t_step, device_accept=True, prefill=pst)
        return toks, i0, commit


def geom(path=DRAFTER):
    """the drafter's (layers, kv heads, head dim, window) from its GGUF metadata (no tensor data): capacity's dflash_geom"""
    rd = GGUFReader(path, 'r'); f = {x.name: x for x in rd.fields.values()}
    return tuple(int(f[f'dflash.{k}'].contents()) for k in ('block_count', 'attention.head_count_kv', 'attention.key_length',
                                                            'attention.sliding_window'))


def smoke_factory(cfg, T, max_ctx=2048, feature_card=0, ring=None):
    """tools/k3_smoke.py's spec:dflash stage (the K3 drafter): the Coder target with ntok = T, features on the XTX (feature_card
    0, the default) or the XT. ring: the drafter's ring rows (None: capacity.dflash_ring; tests pass the old full ring as a
    reference), counted in the target's VRAM check as the drafter allocates it"""
    rd = GGUFReader(cfg['drafter'], 'r'); f = {x.name: x for x in rd.fields.values()}
    layers = [int(v) for v in f['dflash.target_layers'].contents()]; del rd, f
    dg = geom(cfg['drafter'])
    # the drafter's VRAM reserved in the target's check: a config over the caps is refused before the target loads
    m = FlashNextTP(cfg['gguf'], ntok=T, max_ctx=max_ctx, feature_layers=layers, feature_card=feature_card,
                    vram_drafters=(('dflash', dg, ring),))
    df = DFlashK3(m, cfg['drafter'], ring=ring)
    assert df.ring == (ring or capacity.dflash_ring(dg[3], T)), (df.ring, ring)   # (the reserve counted this ring)
    return m, DFlashK3Drafter(df), dict(feature_layers=layers, drafter=cfg['drafter'], impl='k3', ring=df.ring)
