"""The batched prefill of Qwen3-30B-A3B on TWO cards (caminus.engine.q3moe_tp.Q3MoETP), on Caminus's own kernels: the one-card
prefill's chunk (caminus.prefill.q3m.Q3MPrefill) run on each card at that card's local shapes, on the decode engine's
per-card weights read in place (Q3MoETP.L[il][c]: its kv heads' attn_q / attn_k / attn_v rows and attn_output columns, its
expert rows of ffn_gate_exps / ffn_up_exps and columns of ffn_down_exps; the router, norms and embedding replicated) and
writing its kv heads' cache rows (the decode's own caches: the hand-off is the KV rows on each card, as the one-card one).

Per layer and card, the one-card chunk's ops in its order, split where the decode tape splits:
  attn half  h16 = add_rms16(x (+ the previous FFN exchange's sum), attn_norm); q | k | v (its heads' rows) -> qk_rope ->
             its kv heads' cache rows; attention over its kv heads; o = attn_output (its columns): card c's PARTIAL [n, D]
  exchange   the two partials summed on both cards (caminus.prefill.xchg.Exchange: hipMemcpyPeerAsync pushes on copy
             streams, ordered by HIP events only, then p0 + p1 in place, card 0's partial first on both)
  ffn half   x += o (one row), h, h16 = norm(x, ffn_norm); the router (fp32), top-k, grouping (replicated: the same bits on
             both cards from the same inputs); gate / up over its expert rows, swiglu, the down over its columns -> the k
             slot rows per token, weighted on the card (q3p_slot_sum, kernels/q3m_prefill_tp.hip: OP_OWN_SLOT_SUM's fma
             chain) into its PARTIAL [n, D]
  exchange   as above; the next layer's attn half adds it
The last layer stops after its cache rows on each card (as the one-card prefill): 2 (nl - 1) exchanges per micro-chunk.

Micro-chunks (prefill/tp.py's overlap mode, the dense two-card prefill's): a chunk's tokens are split into m pieces on a
128-token grid (micro_bounds) and every op runs per piece. Mode 'overlap' (default) queues each card's compute stream as
compute(h, j) for every piece j of half h, each piece's exchange posted right after it, and the sum of (h, j) only before
compute(h + 1, j): piece j's exchange runs on the copy engines while pieces j + 1 .. compute. 'serial' sums each piece
right after its push (the exchange exposed); 'local' skips the exchange (each card keeps its own partial: the no-exchange
floor for a benchmark only; its KV rows are not the model's, and the engine is marked poisoned after it). Every op is per
token (the GEMMs' rows and the attention's queries are computed from their own inputs; the sums are elementwise), so
pieces, modes and the chunk split change no bit.

Not bitwise to the one-card prefill by design (the K-split down, the attn_output column split, the slot sum before the
exchange each change a summation order); G-model gated as the one-card prefill (tests/prefill/test_q3m_prefill_tp.py):
prefill then two-card decode against the float64 / fp32 references, llama.cpp as the bound. Served as
the prompt path of `caminus serve --cards 2` for Qwen3-30B-A3B (caminus.arch.Q3MoETP). Imports no gguf-py and no tokenizers (prefill/tp.py imports the vendored MMQ path: not imported here)."""
from types import SimpleNamespace
import ctypes

import numpy as np
import torch

from ..device import capacity
from ..device.hip import Module
from ..engine import k3
from . import q3m_ops
from .driver import PrefillDriver, PrefillRefused
from .q3m import layout, buffer_need_mib, shapes_of, ATTN_TP, ROUTER_FLOOR, CROSSOVER
from .xchg import Exchange, xchg_bytes

MODES = ('overlap', 'serial', 'local')
MICRO = 2            # pieces per chunk (the dense two-card prefill's measured best at chunk 2048: m 2; unmeasured here)
MIN_PIECE = 512      # no piece below it (q3p_group's 64-pair tiles: at 512 tokens x 8 slots / 128 experts, 32 pairs an expert)
GRID = 128           # piece boundaries (a multiple of the GEMMs' 64-token tiles and the attention's 16-position blocks)
SLOT_OBJ = 'q3m_prefill_tp'
I = ctypes.c_int


def micro_bounds(T, m, grid=GRID, min_piece=MIN_PIECE):
    """the pieces of a T-token chunk: [(r0, r1)], at most m, every piece but the last a multiple of `grid`, none empty, and
    no more pieces than T // min_piece (at least one): prefill/tp.py's rule plus the floor"""
    T, m = int(T), int(m)
    if T < 1 or m < 1: raise ValueError(f'micro_bounds: T {T}, m {m}')
    m = max(1, min(m, T // max(1, int(min_piece))))
    q = -(-T // m); q = -(-q // grid) * grid
    return [(a, min(a + q, T)) for a in range(0, T, q)]


def local_shapes(m, c):
    """card c's shapes of a Q3MoETP (the ones q3m.layout reads): its kv heads with 8 query heads each, its expert rows"""
    return SimpleNamespace(D=m.D, nh=m.kvc[c] * (m.nh // m.nkv), nkv=m.kvc[c], hd=m.hd, k_used=m.k_used, n_exp=m.n_exp,
                           ffn=m.exp_rows[c])


def tp_shapes(path, exp_rows=None, kv_heads=None):
    """[card 0's, card 1's] local shapes from a qwen3moe GGUF's header alone (before any load); exp_rows / kv_heads: card 0's
    (None: Q3MoETP's defaults)"""
    from ..engine.q3moe_tp import EXP_ROWS
    s = shapes_of(path)
    r0 = EXP_ROWS if exp_rows is None else int(exp_rows); kv0 = s.nkv // 2 if kv_heads is None else int(kv_heads)
    g = s.nh // s.nkv
    return [SimpleNamespace(D=s.D, nh=k * g, nkv=k, hd=s.hd, k_used=s.k_used, n_exp=s.n_exp, ffn=r)
            for k, r in ((kv0, r0), (s.nkv - kv0, s.ffn - r0))]


def layout_tp(s, T):
    """(f32, f16, i32 counts) of one card's buffers at local shapes s and chunk T: q3m.layout's + the FFN partial ffp [T, D]
    (the slot sum's output; `o` is the attention's partial)"""
    f32, f16, i32 = layout(s, T)
    f32 = dict(f32, ffp=T * s.D)
    return f32, f16, i32


def buffer_need_tp_mib(shapes, T):
    """per card: Q3MPrefillTP's buffers at chunk T (its layout at its local shapes + the exchange's receive rows)"""
    return [buffer_need_mib(s, T) + 4 * s.D * T / capacity.MIB + xchg_bytes(T, s.D) / capacity.MIB for s in shapes]


def need_tp_mib(shapes, T):
    """per card: everything Q3MPrefillTP takes at construction (checked before its first allocation): the buffers and receive
    rows (buffer_need_tp_mib), its code objects (capacity.PREFILL_CODE_MIB) and the exchange's streams
    (capacity.PREFILL_STREAMS_MIB: a stream takes device memory at its first use)"""
    return [b + capacity.PREFILL_CODE_MIB[c] + capacity.PREFILL_STREAMS_MIB[c] for c, b in enumerate(buffer_need_tp_mib(shapes, T))]


class SlotSum:
    """kernels/q3m_prefill_tp.hip's q3p_slot_sum on one device (launched on its current stream)"""

    def __init__(self, dev):
        self.dev = torch.device(dev) if not isinstance(dev, torch.device) else dev
        with torch.cuda.device(self.dev): self.mod = Module(SLOT_OBJ)

    def __call__(self, d, w, S, y, n, N):
        """y[t] = sum over s < S of w[t, s] d[t S + s] (fma in slot order from 0), t < n; rows of N floats"""
        n, S, N = int(n), int(S), int(N)
        if n < 1 or not 1 <= S <= 64 or N < 1: raise ValueError(f'q3m slot_sum: n {n}, S {S}, N {N}')
        if d.numel() < n * S * N or w.numel() < n * S or y.numel() < n * N:
            raise ValueError(f'q3m slot_sum: {n} tokens of {S} slots x {N} need d {n * S * N}, w {n * S}, y {n * N} '
                             f'(given {d.numel()}, {w.numel()}, {y.numel()})')
        self.mod.q3p_slot_sum((n,), (256,), d, w, I(S), y, I(N))


def card_weights(m, c, il):
    """layer il of card c as the one-card prefill's matrix tuples (format, rows per matrix, K, device bytes): the decode
    engine's dicts read in place (an expert tensor's rows are all its experts': rows per matrix = rows / n_exp)"""
    L = m.L[il][c]
    t = lambda W, per=1: (W['fmt'], W['rows'] // per, W['K'], W['w'])
    return dict(attn_q=t(L['q']), attn_k=t(L['k']), attn_v=t(L['v']), attn_output=t(L['o']),
                ffn_gate_exps=t(L['gate'], m.n_exp), ffn_up_exps=t(L['up'], m.n_exp), ffn_down_exps=t(L['down'], m.n_exp),
                attn_norm=L['attn_norm'], ffn_norm=L['ffn_norm'], qk_par=L['qk_par'], router=L['router']['w'], cache=L['cache'])


MM_KEYS = ('attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps')


class Q3MPrefillTP(PrefillDriver):
    """the batched prefill of a Q3MoETP engine m, chunks of up to `chunk` tokens in `micro` pieces with the exchange in
    `mode` (MODES); buffers preallocated on both cards (checked against each card's guard line first: capacity.check)"""

    def __init__(self, m, chunk=2048, crossover=None, check=True, micro=MICRO, mode='overlap', min_piece=MIN_PIECE):
        self.m, self.chunk = m, int(chunk)
        if not 16 <= self.chunk <= 4096: raise ValueError(f'Q3MPrefillTP: chunk {chunk} (16 .. 4096)')
        if mode not in MODES: raise ValueError(f'Q3MPrefillTP: mode {mode!r} (one of {MODES})')
        if int(micro) < 1: raise ValueError(f'Q3MPrefillTP: micro {micro} (>= 1)')
        if getattr(m, 'local_only', False): raise ValueError('Q3MPrefillTP: a local_only engine is refused (its numbers are not the model\'s)')
        if getattr(m, 'broken', None): raise ValueError(f'Q3MPrefillTP: the engine is broken ({m.broken})')
        if m.nh != 8 * m.nkv or m.hd != 128: raise ValueError(f'Q3MPrefillTP: {m.nh} / {m.nkv} heads of {m.hd} (8 query heads per kv head of 128: q3p_attn)')
        self.mode, self.micro, self.min_piece = mode, int(micro), int(min_piece)
        self.devs = [torch.device(f'cuda:{d}') for d in m.devs]
        self.crossover = CROSSOVER if crossover is None else int(crossover)
        self.W = [[card_weights(m, c, il) for il in range(m.nl)] for c in (0, 1)]
        bad = [f'blk.{il} {k} card {c}: {self.W[c][il][k][0]}' for c in (0, 1) for il in range(m.nl) for k in MM_KEYS
               if self.W[c][il][k][0] not in q3m_ops.formats('q3m_prefill')]
        if m.emb.type != 'Q4_K': bad.append(f'token_embd: {m.emb.type}')
        if bad: raise ValueError(f'Q3MPrefillTP: {len(bad)} matrices without a prefill kernel (first: {"; ".join(bad[:4])})')
        T = self.chunk
        self.sh = [local_shapes(m, c) for c in (0, 1)]
        self.need_mib = need_tp_mib(self.sh, T)
        if check: capacity.check(list(m.devs), self.need_mib[:], f'Q3MPrefillTP buffers, code objects and streams (chunk {T}, both cards)')
        self.O, self.S, self.B = [], [], []
        for c in (0, 1):
            dev = self.devs[c]
            f32, f16, i32 = layout_tp(self.sh[c], T)
            with torch.cuda.device(dev):
                self.O.append(q3m_ops.Ops(dev, 'q3m_prefill')); self.S.append(SlotSum(dev))
                b = {}
                for n, k in f32.items(): b[n] = torch.zeros(k, dtype=torch.float32, device=dev)
                for n, k in f16.items(): b[n] = torch.zeros(k, dtype=torch.float16, device=dev)
                for n, k in i32.items(): b[n] = torch.zeros(k, dtype=torch.int32, device=dev)
                b['items'] = b['items'].view(-1, 4)
                self.B.append(b)
        self.streams = [torch.cuda.current_stream(d) for d in self.devs]
        self.X = Exchange(m.devs, T, m.D, self.streams, check=False)          # (its rows and streams are counted in need_mib above)
        self.host_toks = torch.zeros(T, dtype=torch.int32, pin_memory=True)
        self.host_ev = [None, None]
        self.n_vocab = int(m.emb.shape[1])
        self.emb_rb = int(m.emb.rows()[1])
        self.last = None
        self.last_n = 0     # (the last chunk's tokens: check_replicas compares its residual rows)
        self.trace = None   # (a list: every launch and exchange call is appended; tests and the benchmark's op timing)

    # -- checks (the shared driver's: limit = max_ctx, check_tokens, validate, route) -------------------------------------
    def check_drafter(self, draft):
        if draft is not None: raise PrefillRefused('Q3MPrefillTP: no drafter')

    def numerics(self):
        """the engine key's prefill part: own WMMA GEMMs (fp16 x, exact weights, fp32 sums), the fp32 router, the two cards'
        partials (the FFN's weighted over its slots on each card first) summed in fp32, card 0's first"""
        return dict(wmma_out='f32', router_logits='f32', moe='own-wmma', q6k='own-wmma', tp='xchg-fp32-c0-first-slot-sum')

    # -- the halves of a layer, one piece ---------------------------------------------------------------------------------
    def _tr(self, *ev):
        if self.trace is not None: self.trace.append(ev)

    def _attn(self, c, il, r0, n, p):
        """card c, layer il, rows [r0, r0 + n) at positions p + r0 ..: the norm (+ the previous FFN exchange), q | k | v, the
        cache rows, then (all but the last layer) the attention and its attn_output partial into o. Returns False on the last
        layer (its cache rows written; nothing more)"""
        m, O, B, s, W = self.m, self.O[c], self.B[c], self.sh[c], self.W[c][il]
        D, nh, nkv, hd = s.D, s.nh, s.nkv, s.hd
        Q, M = nh * hd, nkv * hd; ldq = Q + 2 * M
        self._tr('attn', c, il, r0, n)
        x, h16 = B['x'][r0 * D:(r0 + n) * D], B['h16'][r0 * D:(r0 + n) * D]
        qkv, q16 = B['qkv'][r0 * ldq:(r0 + n) * ldq], B['q16'][r0 * Q:(r0 + n) * Q]
        with torch.cuda.device(self.devs[c]):
            if il == 0: O.add_rms16(x, W['attn_norm'], m.eps, h16, n, D)
            else: O.add_rms16(x, W['attn_norm'], m.eps, h16, n, D, d=B['ffp'][r0 * D:(r0 + n) * D], S=1)
            O.mm(W['attn_q'], h16, D, qkv, ldq, n=n)
            O.mm(W['attn_k'], h16, D, qkv[Q:], ldq, n=n)
            O.mm(W['attn_v'], h16, D, qkv[Q + M:], ldq, n=n)
            O.qk_rope(qkv, ldq, nh, nkv, W['qk_par'], p + r0, q16, W['cache'], n, rows=m.max_ctx)
            if il == m.nl - 1: return False
            att16 = B['att16'][r0 * Q:(r0 + n) * Q]
            _, nb = O.attn_blocks(p + r0, n)
            # (each launch at most the one-card prefill's work: ATTN_TP token-positions over all kv heads, so a card with
            # fewer kv heads takes proportionally more positions a launch)
            step = max(1, ATTN_TP * m.nkv // nkv // (p + r0 + n) // q3m_ops.AT_T)
            for i0 in range(0, nb, step):
                O.attn(q16, W['cache'], p + r0, n, nkv, 1.0 / float(np.sqrt(hd)), att16, i0, min(nb, i0 + step), rows=m.max_ctx)
            O.mm(W['attn_output'], att16, Q, B['o'][r0 * D:(r0 + n) * D], D, n=n)
        return True

    def _ffn(self, c, il, r0, n):
        """card c, layer il, rows [r0, r0 + n): x += o (the summed attention), the norm, the routing, its expert rows, the
        slot sum into its FFN partial ffp"""
        m, O, B, s, W = self.m, self.O[c], self.B[c], self.sh[c], self.W[c][il]
        D, k, E, ff = s.D, s.k_used, s.n_exp, s.ffn
        P = n * k
        self._tr('ffn', c, il, r0, n)
        x, h, h16 = B['x'][r0 * D:(r0 + n) * D], B['h'][r0 * D:(r0 + n) * D], B['h16'][r0 * D:(r0 + n) * D]
        ids, wts, xs, yd = B['ids'][:P], B['wts'][:P], B['xs'][:P], B['yd'][:P]
        g, u, dn, a16 = B['big'][:P * ff], B['big'][P * ff:2 * P * ff], B['big'][:P * D], B['a16'][:P * ff]
        bound = q3m_ops.items_bound(P, E)
        with torch.cuda.device(self.devs[c]):
            O.add_rms16(x, W['ffn_norm'], m.eps, h16, n, D, d=B['o'][r0 * D:(r0 + n) * D], S=1, h=h)
            O.router(h, W['router'], E, D, B['logits'], n)
            O.topk(B['logits'], E, k, ROUTER_FLOOR, ids, wts, B['err'], n)
            O.group(ids, P, k, E, xs, yd, B['items'], B['nitems'], B['err'])
            O.mm(W['ffn_gate_exps'], h16, D, g, ff, work=(B['items'], B['nitems'], bound, xs, yd))
            O.mm(W['ffn_up_exps'], h16, D, u, ff, work=(B['items'], B['nitems'], bound, xs, yd))
            O.swiglu(g, u, a16, P * ff)
            O.mm(W['ffn_down_exps'], a16, ff, dn, D, work=(B['items'], B['nitems'], bound, yd, yd))
            self.S[c](dn, wts, k, B['ffp'][r0 * D:(r0 + n) * D], n, D)

    def halves(self):
        """the chunk's halves in order: (kind, layer); the last layer has only its attention half (its cache rows)"""
        return [(kind, il) for il in range(self.m.nl) for kind in ('attn', 'ffn') if not (il == self.m.nl - 1 and kind == 'ffn')]

    # -- one chunk ----------------------------------------------------------------------------------------------------------
    def _chunk(self, toks, a, b, mode=None):
        m = self.m; n, p = b - a, a
        mode = self.mode if mode is None else mode
        if mode not in MODES: raise ValueError(f'Q3MPrefillTP: mode {mode!r} (one of {MODES})')
        k3.check_span(p, n, m.max_ctx, what='Q3MPrefillTP chunk (the KV rows)')
        for c in (0, 1):
            if self.host_ev[c] is not None: self.host_ev[c].synchronize()   # (the stage is rewritten after its last copies)
        self.host_toks[:n].numpy()[:] = np.asarray(toks[a:b], dtype=np.int32)
        for c in (0, 1):
            B, O = self.B[c], self.O[c]
            with torch.cuda.device(self.devs[c]):
                B['toks'][:n].copy_(self.host_toks[:n], non_blocking=True)
                ev = torch.cuda.Event(); ev.record(); self.host_ev[c] = ev
                self._tr('emb', c, 0, n)
                O.emb(m.emb_dev[c], self.emb_rb, self.n_vocab, B['toks'], B['x'][:n * m.D], n, m.D)
        self.last_n = n
        Bd = micro_bounds(n, self.micro, min_piece=self.min_piece)
        pend = [None] * len(Bd)
        for kind, il in self.halves():
            key = 'o' if kind == 'attn' else 'ffp'
            for j, (r0, r1) in enumerate(Bd):
                if pend[j] is not None:
                    self._tr('sum', *pend[j][1]); self.X.sum(pend[j][0]); pend[j] = None   # (the previous half's piece j)
                more = True
                for c in (0, 1):
                    if kind == 'attn': more = self._attn(c, il, r0, r1 - r0, p) and more
                    else: self._ffn(c, il, r0, r1 - r0)
                if not more or mode == 'local': continue
                t = self.X.post([self.B[0][key], self.B[1][key]], r0, r1)
                self._tr('post', kind, il, j, r0, r1)
                if mode == 'serial': self._tr('sum', kind, il, j); self.X.sum(t)
                else: pend[j] = (t, (kind, il, j))
        assert all(x is None for x in pend), 'Q3MPrefillTP: an exchange left unsummed'   # (the last layer posts none)
        if mode == 'local' and not m.poisoned:
            m.poisoned = 'Q3MPrefillTP ran a local (no exchange) chunk: its KV rows are not the model\'s'

    def check_flags(self, what='prefill'):
        """raise if a chunk set either card's flag word (1: a router logit not finite; 2: an expert id out of range); the
        driver poisons the engine"""
        for c in (0, 1):
            e = int(self.B[c]['err'].item())
            if e:
                self.B[c]['err'].zero_()
                raise k3.ContextBoundsError(f'{what}: card {c}\'s routing flag word is {e:#x} (1: a router logit not finite, 2: an id out of range)')

    def check_replicas(self):
        """after a run (both cards synchronized): the replicated state the same bits on both cards: (the residual rows of the
        last chunk entering the last layer, the last piece's routing ids of the last FFN layer)"""
        n, D = self.last_n, self.m.D
        x0, x1 = self.B[0]['x'][:n * D].cpu(), self.B[1]['x'][:n * D].cpu()
        i0, i1 = self.B[0]['ids'].cpu(), self.B[1]['ids'].cpu()
        return bool(torch.equal(x0.view(torch.int32), x1.view(torch.int32))), bool(torch.equal(i0, i1))

    # -- the prefill (run(), fill(), the hand-off: PrefillDriver's) -----------------------------------------------------------
    def _run_chunk(self, toks, k, j, a, b, segs, bounds):
        self._chunk(toks, a, b)

    def _run_settle(self):
        self._sync_devices(); self.X.settle(); self.check_flags('prefill')

    def _sync_devices(self):
        for d in self.devs: torch.cuda.synchronize(d)
