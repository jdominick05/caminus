"""The batched prefill on both TP cards (docs/specs/spec-prefill-cache.md M5 S5): the exchange (two all-reduces per layer
of a chunk's [T, 2560] fp32 partials) with micro-chunk overlap, and the 48-layer chunk that runs S2's quantized GEMMs, S3's
BF16 GEMMs and glue and S4's GDN half on K3's own weights and states (FlashNextTP.shards / .weights / .state), so the
exchange is measured inside a whole chunk. Device code: kernels/prefill_tp.hip (build/prefill_tp.hsaco).

The exchange (Exchange): each card's partial rows go to the other card by hipMemcpyPeerAsync on that card's own copy stream
(a push: the source card's copy engine), into an ordinary receive buffer on the peer; then on each card's compute stream
pf_xchg_add writes p0 + p1 IN PLACE over that card's partial, card 0's partial first on BOTH cards (OP_XCHG's rule), so the
two replicas get the same bits. Ordering is HIP events only, never a spin: the push waits the partial's event (recorded with
a system-scope release, SysEvent) and the peer's last read of the receive rows (the write-after-read wait: without it the
next push lands while the peer still reads the rows); the sum waits both pushes (the incoming rows landed, and the outgoing
push has read this card's partial, which the sum then overwrites). benchmarks/prefill_link_bench.py measured the path: 7.1 GB/s
each way, 14.0 GB/s both at once (pulls: 12.6), a 21 MB exchange 2.99 ms; ordering 200 + 50 rounds each way bitwise. Not
torch's cross-device copy_: it puts a two-way barrier between both devices' current streams around every peer copy.

Measured (11-04, tests/prefill/test_prefill_tp.py stack, chunk 2048 of real text, both cards, compaction_proactiveness 0): no
exchange 634 ms at m 2 (616 at m 1); the exchange serialized 911 ms (m 1: 96 x ~3 ms exposed); overlapped at m 2 695 ms.
Of the 61 ms left at m 2: the XTX's compute runs ~55 ms slower while the copy engines move the partials both ways (paced
pushes alone, no sums or waits, cost it 36 ms; one direction alone nothing), the sums ~27 ms on the XTX, and the XT waits
~43 ms for the XTX (coarse timelines; tests/prefill/test_prefill_tp.py's COARSE lines).

Micro-chunks (ChunkStack.run m=): the chunk's tokens are split into m pieces on the 128-token WMMA grid (micro_bounds), and
every op runs per piece; mode 'overlap' issues each card's compute stream as compute(h, j) for every piece j of half-layer
h, and only then sum(h, j) before compute(h + 1, j), so piece j's exchange runs on the copy engines while pieces j + 1 ..
compute; 'serial' sums each piece right after its push (the exchange exposed); 'local' skips the exchange (each card keeps
its own partial: the no-exchange floor, not the model's numbers). Every op is per token (the GEMMs and glue: S2 / S3 G2b;
the recurrences carry their state from piece to piece in order: S4 G2b), so the pieces and the chunk split do not change a
bit (tests/prefill/test_prefill_tp.py stack: G2b).

The attention (ChunkStack(attn=)): 'core' (default, M5 S6): the model's attention core on caminus.prefill.attn's kernels,
each matching its K3 decode op: q / k norm + RoPE at the row's absolute position, the target's cache format (Q8 uses
Hadamard rotations and q8_0 rows; F16 uses unrotated half rows),
written into K3's own KV cache (FlashNextTP.state[(card, layer, 'kv')]) and, on a QSA engine, the indexer's raw key rows
(IDXRAW: k_proj through Glue.bf16_mv, bitwise OP_BF16_MV, then q8_0 as the dense tape writes them) and its pooled keys
(IDXPOOL from token 0: prefill_attn.idx_pool, OP_IDX_APPEND's pooling; the pools a QSA-tape pass would backfill are
pooled at the chunk's first piece, so after a chunk every pool j < p1 / 4 is valid: FlashNextTP._pool_ok), dense attention
over cells 0 .. pos for rows below capacity.QSA_EXACT, and for rows at or past it (a QSA engine only, M5 S7: the QSA
prompt path) the model's QSA: the indexer query (q_proj through Glue.bf16_mv, bitwise OP_BF16_MV; q norm + RoPE), the
scores in query sub-blocks into the bounded scratch (AttnWorkspace.score), the per-row top-512 selection plus the tail,
the split pair over each row's selected cells (each bitwise its QSA-module decode op: tests/prefill/test_prefill_qsa.py), then
the output rotation and gate. A non-QSA engine refuses a chunk reaching QSA_EXACT. 'standin': S5's pf_attn_standin (each
token attends to itself only: its own v, gated), kept for the S5 stack test and its labelled 8K number (the attention
layers' GEMMs and exchange on real inputs, the core left out).
The routed experts (ChunkStack(moe=), MOE_PATHS): 'own' (the default, cleared by verify:moe-gemm 11-04): caminus.prefill.moe's kernels: ONE
grouping per layer and micro-chunk (the clamp, then mm_ids_helper's tables and the flat tile lists), quantize_scatter, the
fused gate / up + swiglu + down-input quantization, the down projection over its tile list; 'mmq': the vendored MMQ's
mul_mat_id sequence (two mm_ids_helper groupings, mul_mat_id gate / up, pf_swiglu, the per-slot quantize, mul_mat_id down),
kept for A/Bs. Both give the same bits (tests/prefill/test_prefill_moe.py g1 per op on real routed inputs; the stack's moe A/B
end to end); ChunkStack.moe may be switched between chunks. The shared expert's swiglu (and the MMQ path's routed one),
the slot-order combine and the gated shared-expert add are this module's kernels (the decode ops' expressions, gated only
for the replicas and G2b here). The dense Q6_K GEMMs (ChunkStack(q6k=), Q6K_PATHS; impl:q6k-dense): 'own' (default):
caminus.prefill.dense on the same quantized activation, bitwise MMQ (the 48 layers' dense Q6_K 112.4 -> 79.5 ms per 2K chunk at card 0's
shapes on the XTX, 2 x 1024); 'wmma': each Q6_K matrix dequantized to bf16 right before its GEMM, the activation's bf16 copy,
gemm.hip's WMMA (a measured tolerance; dense.Bf16Workspace); 'mmq' for A/Bs. The S5 stack at overlap m 2 (11-04): mmq 616.7,
own 583.4, wmma 578.1 ms per 2K chunk. The DFlash features (the hc mean of the residual entering each
of FlashNextTP.feature_layers, OP_HC_AVG) go into the attention workspace's export on the feature card: the positions
[feat_base, feat_base + feat_rows) (the driver sets feat_base per prompt: 0 for a prompt within the export, else its last
feat_rows positions, which cover the drafter's window); rows outside it are not exported. No output head (the last token goes through K3: caminus.prefill.coder), no lazy-commit replay (the driver restarts a
prefill at the last committed position). The chunk writes K3's own recurrent state (the GDN state, conv state and PLE
history in FlashNextTP.state) in place.

HIP graphs (caminus.prefill.graph, M6, opt-in: run(graphs=)): ChunkStack._body runs the 48 layers with each card's half-layer
compute either launched here (_half) or replayed as that card's captured segment graph (launch=), the exchange between them
the same calls either way; structure() is what a captured structure is keyed by, and _pos marks the position-dependent
launches (the attention core, the feature export) a replay reruns to patch their nodes.

Memory: the exchange's receive buffer (xchg_layout) and the four streams (a stream takes device memory at its first use:
capacity.PREFILL_STREAMS_MIB; used once at construction) are counted in capacity.prefill_need_mib; everything else is the S2 / S3
workspaces' (mmq.Workspace, prefill_ops.GlueWorkspace). Nothing here allocates per chunk. Every launch is refused
(ValueError) unless its buffers cover what it touches within their own views and no written buffer overlaps another
(caminus.device.bufcheck; the exact in-place forms only)."""
import ctypes
import torch
from ..device.hip import Module, hip, check
from ..device.bufcheck import ALIGN, LaunchCheck, aligned   # noqa: F401  (ALIGN: tp.ALIGN, the layouts' block)
from . import mmq, glue as po
from .xchg import (EV_DISABLE_TIMING, EV_RELEASE_TO_SYSTEM, xchg_layout, xchg_bytes, SysEvent, F32, _check, Tp,   # noqa: F401
                   _peer_copy, Exchange)          # (the exchange's gguf-free home: caminus.prefill.xchg)
from ..device import gemm as _gemm
from .gdn import Gdn
from . import attn as pa
from . import moe as _moe
from . import dense as _dense_mod

EPS = 9.999999974752427e-07                                    # the model's rms eps (flashnext.EPS)
MODES = ('overlap', 'serial', 'local')
MOE_PATHS = ('own', 'mmq')   # ChunkStack's routed experts: caminus.prefill.moe's kernels (default) or the vendored MMQ's mul_mat_id
MOE_PATH = 'own'   # verify:moe-gemm (11-04) cleared the own path: XT-alone canaries around every buffer it writes, the faulting
                   # sequence on both cards under per-launch canaries, stack x3, test_prefill_s6 g4a g2b g3 dflash long, gates, smoke, no
                   # amdgpu kernel-log line in any run. The 09:24 SDMA page fault's cause is still undetermined (it did not recur).
                   # 'mmq' stays for A/Bs
# ChunkStack's dense Q6_K GEMMs (ChunkStack(q6k=), impl:q6k-dense): 'own' caminus.prefill.dense (bitwise MMQ, the same quantized
# activation), 'wmma' Q6_K dequantized to bf16 + gemm.hip's bf16 WMMA (a measured tolerance; its buffers: dense.Bf16Workspace,
# counted by capacity when it is the default), 'mmq' the vendored MMQ (A/Bs)
Q6K_PATHS = ('own', 'wmma', 'mmq')
Q6K_PATH = 'own'
ATTN_MODES = ('core', 'standin')   # ChunkStack's attention: the model's core (S6, positions < QSA_EXACT) or S5's stand-in


def bf16_wanted(q6k=None, q6k_buffers=None):
    """whether a ChunkStack(q6k=, q6k_buffers=) allocates the 'wmma' dense path's bf16 buffers (dense.Bf16Workspace) at
    construction: q6k_buffers None = when its path (None: Q6K_PATH) is 'wmma'; True / False as given. The prefill plan's
    bf16_dense (caminus.prefill.plan)"""
    return ((Q6K_PATH if q6k is None else q6k) == 'wmma') if q6k_buffers is None else bool(q6k_buffers)


def micro_bounds(T, m, grid=po.TPAD):
    """the m micro-chunks of a T-token chunk: [(r0, r1)], every piece but the last a multiple of `grid` (the WMMA GEMMs' 128
    rows: a piece's padding rows then never reach into the next piece's rows) and none empty"""
    T, m = int(T), int(m)
    if T < 1 or m < 1: raise ValueError(f'micro_bounds: T {T}, m {m}')
    q = -(-T // m); q = -(-q // grid) * grid
    return [(a, min(a + q, T)) for a in range(0, T, q)]


class Timeline:
    """per card, a timing event at each label change on the compute stream: the time between two marks is the earlier
    label's (instrumented runs only: each event costs GPU time, ~5 us measured per event without the exchange and more with
    it, so the fine labels inflate the small ops). coarse=True: only the exchange's labels (xchg_wait, xchg_add) and the
    feed are kept apart, everything else is 'compute' (a few events per exchange)"""
    KEEP = ('xchg_wait', 'xchg_add', 'feed', 'end')

    def __init__(self, streams, coarse=False):
        self.ks = streams; self.ev = [[], []]; self.coarse = coarse

    def mark(self, c, label):
        if self.coarse and label not in self.KEEP: label = 'compute'
        if self.ev[c] and self.ev[c][-1][0] == label: return               # (the same label continues: no event)
        e = torch.cuda.Event(enable_timing=True); e.record(self.ks[c]); self.ev[c].append((label, e))

    def totals(self):
        out = []
        for c in (0, 1):
            t = {}
            for (lab, e0), (_, e1) in zip(self.ev[c], self.ev[c][1:]): t[lab] = t.get(lab, 0.0) + e0.elapsed_time(e1)
            span = self.ev[c][0][1].elapsed_time(self.ev[c][-1][1]) if len(self.ev[c]) > 1 else 0.0
            out.append(dict(span=span, parts=t))
        return out


def _v(t, r0, n, w):
    """rows [r0, r0 + n) of a buffer of rows of w elements"""
    return t.view(-1)[r0 * w:(r0 + n) * w]


def _vp(t, r0, n, w):
    """the same rows padded to the WMMA grid (a WMMA GEMM reads / writes tpad(n) rows)"""
    return t.view(-1)[r0 * w:(r0 + po.tpad(n)) * w]


class ChunkStack:
    """one chunk of the 48 layers on both TP cards: K3's weights (FlashNextTP m: .shards, .weights) and recurrent states
    (m.state, written in place), the S2 / S3 workspaces per card for chunks of up to T tokens, the two-card feed
    (feed: prefill_feed.PrefillFeed(devices=..., max_chunk >= T)), the exchange. run() queues a chunk; nothing is
    synchronized until the caller does (then check_routing). router_logits / wmma_out: prefill_ops.Glue's modes (None: its
    defaults); moe: the routed experts' path (MOE_PATHS; None: MOE_PATH; the attribute .moe may change between chunks); q6k:
    the dense Q6_K GEMMs' path (Q6K_PATHS; None: Q6K_PATH; .q6k may change between chunks, to 'wmma' only with q6k_buffers).
    plan: the prefill plan its buffers are sized by (caminus.prefill.plan.PrefillPlan: Prefill's; None: this engine's at
    chunk T, the bf16 buffers by bf16_wanted(q6k, q6k_buffers)), the same arguments the projection counts"""
    def __init__(self, m, feed, T, router_logits=None, attn='core', rows_per_block=pa.ROWS_PER_BLOCK, wmma_out=None, moe=None,
                 q6k=None, q6k_buffers=None, plan=None):
        from .plan import PrefillPlan, check_counted
        if attn not in ATTN_MODES: raise ValueError(f'ChunkStack: attn {attn!r} (one of {ATTN_MODES})')
        self.moe = MOE_PATH if moe is None else moe
        if self.moe not in MOE_PATHS: raise ValueError(f'ChunkStack: moe {self.moe!r} (one of {MOE_PATHS})')
        self.q6k = Q6K_PATH if q6k is None else q6k
        if self.q6k not in Q6K_PATHS: raise ValueError(f'ChunkStack: q6k {self.q6k!r} (one of {Q6K_PATHS})')
        # the 'wmma' dense path's buffers (caminus.prefill.dense.Bf16Workspace; q6k_buffers None: when it is this stack's path, True: at
        # construction anyway, so .q6k may switch to it; alloc_q6k_buffers() allocates them later)
        want_bf = bf16_wanted(self.q6k, q6k_buffers)
        self.m, self.feed, self.T, self.attn, self.rpb = m, feed, int(T), attn, int(rows_per_block)
        if plan is None: plan = PrefillPlan.for_engine(m, self.T, bf16_dense=want_bf, rows_per_block=self.rpb)
        mine = PrefillPlan.for_engine(m, self.T, bf16_dense=want_bf, rows_per_block=self.rpb, feed=plan.feed)
        if plan != mine: raise ValueError(f'ChunkStack: the plan given is not this stack\'s ({plan._replace(g=None)} vs {mine._replace(g=None)})')
        check_counted(m, plan, 'ChunkStack')     # (no more than the engine counted: refused before anything is allocated)
        self.plan = plan
        self.devs = [torch.device(f'cuda:{d}') for d in m.devs]
        g = m.geom; self.g = g
        self.D, self.HC, self.LR, self.nl, self.fai = m.D, m.hc, g['hc_lr'], m.nl, m.fai
        self.S = g['ssm_state']; nvk = g['ssm_heads'] // g['ssm_groups']
        self.kc = [m.gdn_keys, g['ssm_groups'] - m.gdn_keys]; self.H = [nvk * k for k in self.kc]
        self.CH = [(2 + nvk) * self.S * k for k in self.kc]
        self.nh, self.hd, self.k_used, self.n_exp = g['nh'] // g['nkv'], g['hd'], g['k_used'], g['n_exp']
        self.dims = [plan.card_dims(c) for c in (0, 1)]
        # the compute streams first: a stream takes device memory at its first kernel (capacity.PREFILL_STREAMS_MIB), so one
        # tiny launch each, before the workspaces check the cards' caps against the live use
        self.ks = [torch.cuda.Stream(device=d) for d in self.devs]
        for c in (0, 1):
            with torch.cuda.device(self.devs[c]), torch.cuda.stream(self.ks[c]): torch.zeros(1, device=self.devs[c]).add_(1)
            self.ks[c].synchronize()
        self.W = [mmq.Workspace(self.devs[c], self.T, self.dims[c]) for c in (0, 1)]
        self.WB = [None, None]
        if plan.bf16_dense: self.alloc_q6k_buffers()
        self.GW = [po.GlueWorkspace(self.devs[c], self.T, plan.glue_dims(c)) for c in (0, 1)]
        self.M, self.Gl, self.Gd, self.Dn = [], [], [], []
        self.O = []
        for d in self.devs:
            self.M.append(mmq.MMQ(d)); self.Gl.append(po.Glue(d, router_logits, wmma_out)); self.Gd.append(Gdn(d))
            self.O.append(_moe.MoE(d, self.M[-1])); self.Dn.append(_dense_mod.Dense(d))
            with torch.cuda.device(d): _gemm._m()
        self.P = [Tp(d) for d in self.devs]
        self.qsa = bool(getattr(m, 'qsa', False))
        self.feat = list(getattr(m, 'feature_layers', ())); self.fcard = int(getattr(m, 'feature_card', 0))
        self.AW, self.A = [None, None], [None, None]
        if attn == 'core':
            for c in (0, 1):
                a = plan.attn_args(c)
                self.AW[c] = pa.AttnWorkspace(self.devs[c], a['T'], a['nh'], a['hd'], a['idx_dim'], rows_per_block=a['rows_per_block'],
                                              n_feat=a['n_feat'], D=a['D'], T_pass=a['T_pass'], qsa_ctx=a['qsa_ctx'])
                self.A[c] = pa.Attn(self.devs[c])
                if not m.q8: self.A[c].prepare_f16()   # module loading stays outside the first graph capture
        self.p0 = 0
        self.feat_base, self.feat_rows = 0, pa.feat_rows(plan.T_pass)   # the feature export's positions [feat_base, + feat_rows)
        self._pool_from = 0
        assert self.D == plan.g['D'], (self.D, plan.g['D'])
        self.X = Exchange(m.devs, plan.chunk, plan.g['D'], self.ks)
        self.tl = None
        self.rec, self._il = None, 0   # (caminus.prefill.graph's recorder while a graph is captured or patched)
        self._weights()

    def alloc_q6k_buffers(self):
        """the 'wmma' dense Q6_K path's buffers on both cards (dense.Bf16Workspace; once). Refused when the engine counted a
        prefill without them (plan.check_counted: FlashNextTP(vram_prefill=dict(..., q6k_buffers=True)))"""
        from .plan import check_counted
        if not self.plan.bf16_dense:
            check_counted(self.m, self.plan._replace(bf16_dense=True), 'ChunkStack.alloc_q6k_buffers')
            self.plan = self.plan._replace(bf16_dense=True)
        for c in (0, 1):
            if self.WB[c] is None: self.WB[c] = _dense_mod.Bf16Workspace(self.devs[c], self.T, self.dims[c])

    # -- K3's weights and states ------------------------------------------------------------------------------------------
    def _weights(self):
        m = self.m; w = lambda kind, name: m.weights[(kind, name)]; sh = lambda name, c: m.shards[name]['infos'][c]
        self.L = [[], []]
        for il in range(self.nl):
            p = f'blk.{il}'
            for c in (0, 1):
                d = {h: dict(nw=w('_rep_f32', f'{p}.{h}_norm.weight')[c], dn=w('_rep', f'{p}.{h}_down.weight')[c],
                             iw=w('_rep', f'{p}.{h}_inject.weight')[c], up=w('_rep', f'{p}.{h}_up.weight')[c]) for h in ('hc_attn', 'hc_ffn')}
                for h in ('hc_attn', 'hc_ffn'):
                    assert all(d[h][k]['fmt'] == 'BF16' for k in ('dn', 'iw', 'up')), f'{p}.{h}: the prefill runs BF16 hc matrices only'
                if il in m.ple_layers:
                    d['ple'] = dict(k=w('_rep', f'{p}.ple_key.weight')[c], v=w('_rep', f'{p}.ple_value.weight')[c],
                                    nk=w('_rep_f32', f'{p}.ple_norm_key.weight')[c], nq=w('_rep_f32', f'{p}.ple_norm_query.weight')[c],
                                    nc=w('_rep_f32', f'{p}.ple_norm_conv.weight')[c], conv=w('ple_conv', f'{p}.ple_conv1d.weight')[c],
                                    hist=m.state[(c, il, 'ple')])
                    assert d['ple']['k']['fmt'] == d['ple']['v']['fmt'] == 'BF16'
                if il % self.fai != self.fai - 1:
                    H = self.H[c]
                    d['gdn'] = dict(qkv=sh(f'{p}.attn_qkv.weight', c), z=sh(f'{p}.attn_gate.weight', c), al=sh(f'{p}.ssm_alpha.weight', c),
                                    be=sh(f'{p}.ssm_beta.weight', c), dt=w('_f32_idx', f'{p}.ssm_dt.bias')[c], a=w('_f32_idx', f'{p}.ssm_a')[c],
                                    nw=w('_rep_f32', f'{p}.ssm_norm.weight')[c], conv=w('conv', f'{p}.ssm_conv1d.weight')[c],
                                    so=sh(f'{p}.ssm_out.weight', c), conv_st=m.state[(c, il, 'conv')],
                                    state=m.state[(c, il, 'gdn')][:H * self.S * self.S])
                    assert d['gdn']['al']['fmt'] == d['gdn']['be']['fmt'] == 'BF16'
                else:
                    d['attn'] = dict(q=sh(f'{p}.attn_q.weight', c), k=sh(f'{p}.attn_k.weight', c), v=sh(f'{p}.attn_v.weight', c),
                                     o=sh(f'{p}.attn_output.weight', c), qn=w('_rep_f32', f'{p}.attn_q_norm.weight')[c],
                                     kn=w('_rep_f32', f'{p}.attn_k_norm.weight')[c], kv=m.state[(c, il, 'kv')])
                    if self.qsa:   # the indexer: IDXRAW as the QSA engine's dense tape writes it, IDXPOOL, the query (S7)
                        ai = il // self.fai
                        d['attn'].update(ikw=w('_rep', f'{p}.indexer.k_proj.weight')[c], raw=m.state[(c, 'idxraw')][ai],
                                         iqw=w('_rep', f'{p}.indexer.q_proj.weight')[c], ikn=w('_rep_f32', f'{p}.indexer.k_norm.weight')[c],
                                         iqn=w('_rep_f32', f'{p}.indexer.q_norm.weight')[c], pool=m.state[(c, 'idxpool')][ai])
                        assert d['attn']['ikw']['fmt'] == 'BF16' and d['attn']['ikw']['rows'] == self.g['idx_dim'] == 2 * 64
                        assert d['attn']['iqw']['fmt'] == 'BF16' and d['attn']['iqw']['rows'] == pa.QSA_H * pa.QSA_D
                d['ffn'] = dict(router=w('_router_w', f'{p}.ffn_gate_inp.weight')[c], sgate=w('_router_w', f'{p}.ffn_gate_inp_shexp.weight')[c],
                                sg=sh(f'{p}.ffn_gate_shexp.weight', c), su=sh(f'{p}.ffn_up_shexp.weight', c), sd=sh(f'{p}.ffn_down_shexp.weight', c),
                                g=sh(f'{p}.ffn_gate_exps.weight', c), u=sh(f'{p}.ffn_up_exps.weight', c), d=sh(f'{p}.ffn_down_exps.weight', c))
                assert d['ffn']['router']['fmt'] == d['ffn']['sgate']['fmt'] == 'BF16', f'{p}: the prefill router reads BF16 logits weights'
                assert d['ffn']['g']['fmt'] == d['ffn']['u']['fmt'], f'{p}: the MoE gate / up share one quantized activation'
                self.L[c].append(d)

    # -- helpers ------------------------------------------------------------------------------------------------------------
    def _mk(self, c, label):
        if self.tl is not None: self.tl.mark(c, label)

    def _dense(self, c, sh, x, K, n, out, yc, name, reuse=True, outp=None):
        """out [n, rows] = W x for the n rows of x (fp32 [n, K]) on MMQ (a Q6_K W: on self.q6k's path). x is quantized into
        the workspace's buffer of the format's layout unless yc (this half's record of what each layout's buffer holds) says
        the buffer holds this x already (the D-wide mixed row: quantized once per layout per half); reuse=False (the K-side
        inputs) always quantizes and leaves the buffer marked as holding nothing reusable. outp: the same rows padded to the
        WMMA grid (_vp), which q6k 'wmma' writes (its bf16 copy of x is kept in yc the same way)"""
        M, W = self.M[c], self.W[c]; fmt = sh['fmt']
        key = (x.data_ptr(), x.numel(), int(K), int(n))
        if fmt == 'Q6_K' and self.q6k == 'wmma':
            WB = self.WB[c]
            if WB is None: raise ValueError('ChunkStack: q6k \'wmma\' needs its buffers (ChunkStack(q6k_buffers=True) or alloc_q6k_buffers())')
            if outp is None: raise ValueError(f'ChunkStack._dense {name}: q6k \'wmma\' writes whole WMMA row tiles (outp)')
            R, Gl = sh['rows'], self.Gl[c]
            self._mk(c, f'q6k:{name}:wmma')
            xb = WB.x_bf[:po.tpad(n) * K]
            if not (reuse and yc.get('bf') == key): Gl.to_bf16(x, xb, n, K)
            yc['bf'] = key if reuse else None
            wb = WB.w_bf[:R * K]
            self.Dn[c].deq_bf16(fmt, sh['w'], R, K, wb)
            Gl.bf16_gemm(wb, R, K, xb, outp, n)
            return
        lay = M.layout(fmt)
        yb = W.y_d4 if lay == mmq.D4 else W.y_ds4
        self._mk(c, f'mmq:{name}:{fmt}')
        if not (reuse and yc.get(lay) == key):
            M.quantize(x, K, n, yb, lay)
        yc[lay] = key if reuse else None
        if fmt == 'Q6_K' and self.q6k == 'own': self.Dn[c].mm(fmt, sh['w'], sh['rows'], K, yb, n, out)
        else: M.mm(fmt, sh['w'], sh['rows'], K, yb, n, out)

    # -- the halves of a layer, one micro-chunk ---------------------------------------------------------------------------
    def _hc_mix(self, c, hw, r0, n, post):
        G, Gl = self.GW[c], self.Gl[c]; HC, D, LR = self.HC, self.D, self.LR; HD = HC * D
        res, scale, xbf = _v(G.res, r0, n, HD), _v(G.scale, r0, n, HC), _v(G.xn_bf, r0, n, HD)
        self._mk(c, 'glue')
        if post is None: Gl.rms_mul(res, hw['nw'], None, n * HC, EPS, ybf=xbf, scale=scale)
        else: Gl.hc_post_rms(post, _v(G.inj, r0, n, HC), res, hw['nw'], n, HC, EPS, xnbf=xbf, scale=scale)
        self._mk(c, 'wmma')
        lo = _vp(G.lo, r0, n, LR)
        Gl.bf16_gemm(hw['dn']['w'], LR, HD, _vp(G.xn_bf, r0, n, HD), lo, n)
        self._mk(c, 'bf16_mv')
        Gl.bf16_mv(hw['iw']['w'], HC, HD, res, _v(G.inj, r0, n, HC), n, xscale=scale, xw=hw['nw'].view(-1))
        self._mk(c, 'glue')
        lob = _vp(G.lo_bf, r0, n, LR)
        Gl.silu_bf16(lo, lob, n, LR, 1.0 / HC)
        self._mk(c, 'wmma')
        gate = _vp(G.gate, r0, n, HD)
        Gl.bf16_gemm(hw['up']['w'], HD, LR, lob, gate, n)
        self._mk(c, 'glue')
        Gl.hc_mean(res, gate, _v(G.mixed, r0, n, D), n, HC, scale=scale, w=hw['nw'].view(-1))

    def _ple(self, c, pw, ple_in, r0, n):
        G, Gl = self.GW[c], self.Gl[c]; HC, D = self.HC, self.D; HD = HC * D
        res = _v(G.res, r0, n, HD)
        self._mk(c, 'glue')
        xb = _vp(G.xn_bf, r0, n, D)                                   # (glue_layout's alias: the PLE input's bf16 copy)
        Gl.to_bf16(ple_in, xb, n, D)
        self._mk(c, 'wmma')
        pk, pv = _vp(G.gate, r0, n, HD), _vp(G.mixed, r0, n, D)      # (aliases: the PLE key / gated value, the PLE value)
        Gl.bf16_gemm(pw['k']['w'], HD, D, xb, pk, n); Gl.bf16_gemm(pw['v']['w'], D, D, xb, pv, n)
        self._mk(c, 'glue')
        pk = _v(G.gate, r0, n, HD)
        Gl.rms_mul(pk, pw['nk'], pk, n * HC, EPS)
        sq, sc = _v(G.scale2, r0, n, HC), _v(G.scale, r0, n, HC)
        Gl.rms_mul(res, pw['nq'], None, n * HC, EPS, scale=sq)
        Gl.ple_gate(pk, res, _v(G.mixed, r0, n, D), pk, _v(G.pg, r0, n, HC), n, HC, qscale=sq, qw=pw['nq'].view(-1))
        Gl.rms_mul(pk, pw['nc'], None, n * HC, EPS, scale=sc)
        Gl.ple_conv(pk, pw['conv'], pw['hist'], HD, n, dil=3, a=pk, res=res, xscale=sc, xw=pw['nc'].view(-1))

    def _gdn(self, c, L, r0, n):
        G, W, Gl, Gd = self.GW[c], self.W[c], self.Gl[c], self.Gd[c]; D, S = self.D, self.S
        kc, H, CH = self.kc[c], self.H[c], self.CH[c]
        mixed = _v(G.mixed, r0, n, D); yc = {}
        al, be, gt = _v(G.alpha, r0, n, H), _v(G.beta, r0, n, H), _v(G.gdn_gates, r0, n, 2 * H)
        self._mk(c, 'bf16_mv')
        Gl.bf16_mv(L['al']['w'], H, D, mixed, al, n); Gl.bf16_mv(L['be']['w'], H, D, mixed, be, n)
        self._mk(c, 'glue')
        Gd.gates(al, be, L['dt'], L['a'], gt, n, H)
        oq = _v(W.o_qkv, r0, n, CH)
        self._dense(c, L['qkv'], mixed, D, n, oq, yc, 'qkv', outp=_vp(W.o_qkv, r0, n, CH))
        self._mk(c, 'glue')
        conv = _v(G.gate, r0, n, CH)                                  # (glue_layout's alias: the conv output)
        Gl.gdn_conv(oq, L['conv'], L['conv_st'], conv, CH, n)
        qk = _v(G.gdn_qk, r0, n, 2 * kc * S)
        Gd.l2(conv, qk, n, 2 * kc, EPS, ldx=CH)
        oz = _v(W.o_z, r0, n, H * S)
        self._dense(c, L['z'], mixed, D, n, oz, yc, 'z', outp=_vp(W.o_z, r0, n, H * S))
        self._mk(c, 'gdn_step')
        att = _v(G.gdn_att, r0, n, H * S)
        Gd.step(qk, conv[2 * kc * S:], gt, L['state'], att, n, H, kc, CH)
        self._mk(c, 'glue')
        xs = _v(W.x_ssm_out, r0, n, H * S)
        Gd.norm(att, L['nw'], oz, xs, n, H, EPS)
        ob = _v(W.o_blk, r0, n, D)
        self._dense(c, L['so'], xs, H * S, n, ob, yc, 'ssm_out', reuse=False, outp=_vp(W.o_blk, r0, n, D))
        return W.o_blk

    def _attn(self, c, L, r0, n):
        G, W = self.GW[c], self.W[c]; D, nh, hd = self.D, self.nh, self.hd
        mixed = _v(G.mixed, r0, n, D); yc = {}
        oq, ok_, ov = _v(W.o_q, r0, n, 2 * nh * hd), _v(W.o_k, r0, n, hd), _v(W.o_v, r0, n, hd)
        self._dense(c, L['q'], mixed, D, n, oq, yc, 'attn_q', outp=_vp(W.o_q, r0, n, 2 * nh * hd))
        self._dense(c, L['k'], mixed, D, n, ok_, yc, 'attn_k', outp=_vp(W.o_k, r0, n, hd))
        self._dense(c, L['v'], mixed, D, n, ov, yc, 'attn_v', outp=_vp(W.o_v, r0, n, hd))
        xa = _v(W.x_attn_out, r0, n, nh * hd)
        if self.attn == 'standin':
            self._mk(c, 'attn_standin')
            self.P[c].attn_standin(ov, oq, xa, n, nh, hd)
        else:
            self._pos(('core', c, self._il, r0), lambda: self._core(c, L, r0, n, mixed, oq, ok_, ov, xa))
        self._dense(c, L['o'], xa, nh * hd, n, _v(W.o_blk, r0, n, D), yc, 'attn_output', reuse=False, outp=_vp(W.o_blk, r0, n, D))
        return W.o_blk

    def _core(self, c, L, r0, n, mixed, oq, ok_, ov, xa):
        """the attention core of rows [r0, r0 + n) (positions p0 + r0 ..): FlashNextTP's attention-layer tape over many rows,
        bitwise its ops (caminus.prefill.attn); the output (after its rotation and gate) into xa"""
        A, AW, m = self.A[c], self.AW[c], self.m; nh, hd, p = self.nh, self.hd, self.p0 + r0
        self._mk(c, 'attn_core')
        q8 = bool(m.q8)
        # The engine and host-mapped KV keep raw uint8 storage in either mode. The launcher takes a typed F16 view
        # so its dtype, extent and vector-load alignment checks describe the selected representation explicitly.
        cache = L['kv'] if q8 else L['kv'].view(torch.float16)
        ko = AW.ko[:n * hd]
        A.qk_nr(ok_, hd, 1, hd, L['kn'], m.nrot, m.theta_scale, EPS, p, ko, n)
        if q8:
            A.hadamard(ko, hd, n)
            A.hadamard(ov, 64, n * hd // 64)
        A.kv_append(ko, hd, ov, hd, hd, cache, p, n, AW.err, q8=q8)
        if self.qsa:
            ik = AW.ikr[:n * 128]
            self._mk(c, 'bf16_mv'); self.Gl[c].bf16_mv(L['ikw']['w'], 128, self.D, mixed, ik, n); self._mk(c, 'attn_core')
            A.kv_append(ik, 128, ik[64:], 128, 64, L['raw'], p, n, AW.err)
            # the pools this piece completes (and, at the chunk's first piece, every pool left unpooled before it): j < (p + n) / 4
            j0, j1 = (self._pool_from if r0 == 0 else p // 4), (p + n) // 4
            if j1 > j0: A.idx_pool(L['raw'], L['pool'], j0, j1, L['ikn'], m.theta_scale, EPS, AW.err)
        qo = AW.qo[:n * nh * hd]
        A.qk_nr(oq, 2 * hd, nh, hd, L['qn'], m.nrot, m.theta_scale, EPS, p, qo, n)
        if q8: A.hadamard(qo, hd, n * nh)
        self._mk(c, 'attn')
        sc = 1.0 / (hd ** 0.5)
        b = max(p, min(p + n, pa.qsa_exact()))                 # rows [p, b) dense, [b, p + n) the model's QSA
        if b > p:
            A.attend(qo[:(b - p) * nh * hd], cache, p, b - p, 1, sc, xa[:(b - p) * nh * hd], AW.rec, AW.err, rows_per_block=self.rpb, q8=q8)
        if p + n > b:
            o, q_ = b - p, p + n - b; Q = pa.QSA_H * pa.QSA_D
            iq, iqo = AW.iq[:q_ * Q], AW.iqo[:q_ * Q]
            self._mk(c, 'bf16_mv'); self.Gl[c].bf16_mv(L['iqw']['w'], Q, self.D, mixed[o * self.D:], iq, q_); self._mk(c, 'qsa')
            A.qk_nr(iq, pa.QSA_D, pa.QSA_H, pa.QSA_D, L['iqn'], m.nrot, m.theta_scale, EPS, b, iqo, q_)
            for s0 in range(b, p + n, AW.srows):               # the scores in query sub-blocks, each row's selection
                R = min(AW.srows, p + n - s0)
                A.qsa_score(iqo[(s0 - b) * Q:], L['pool'], s0, R, AW.score, AW.prow)
                A.qsa_select(AW.score, AW.prow, s0, R, AW.prow, AW.sel[(s0 - b) * pa.QSA_SELW:])
            self._mk(c, 'attn')
            A.attend(qo[o * nh * hd:], cache, b, q_, 1, sc, xa[o * nh * hd:], AW.rec, AW.err, rows_per_block=self.rpb,
                     sel=AW.sel[:q_ * pa.QSA_SELW], q8=q8)
        self._mk(c, 'attn_core')
        if q8: A.hadamard(xa, 64, n * nh * hd // 64, 4)
        A.gate_mul(xa, oq, 2 * hd, hd, xa, n * nh * hd, hd)

    def _ffn(self, c, L, r0, n, hist):
        G, W, Gl, M, P = self.GW[c], self.W[c], self.Gl[c], self.M[c], self.P[c]
        D, k, E = self.D, self.k_used, self.n_exp; ffc = self.dims[c]['ffc']
        mixed = _v(G.mixed, r0, n, D); yc = {}
        ids, wts, sgl = _v(G.ids, r0, n, k), _v(G.wts, r0, n, k), _v(G.sgl, r0, n, 1)
        self._mk(c, 'router')
        lg = _vp(G.logits, r0, n, E)
        Gl.router_logits(L['router']['w'], E, mixed, lg, n, xbf=_vp(G.mixed_bf, r0, n, D))
        Gl.router(lg, n, E, k, ids, wts, G.err)
        self._mk(c, 'bf16_mv')
        Gl.bf16_mv(L['sgate']['w'], 1, D, mixed, sgl, n)
        if hist is not None:
            self._mk(c, 'hist')
            hist['ids'][c][hist['il']][r0:r0 + n].copy_(ids.view(n, k)); hist['wts'][c][hist['il']][r0:r0 + n].copy_(wts.view(n, k))
        osg, osu, xsd, osd = _v(W.o_sh_gate, r0, n, ffc), _v(W.o_sh_up, r0, n, ffc), _v(W.x_sh_down, r0, n, ffc), _v(W.o_sh_down, r0, n, D)
        self._dense(c, L['sg'], mixed, D, n, osg, yc, 'sh_gate', outp=_vp(W.o_sh_gate, r0, n, ffc))
        self._dense(c, L['su'], mixed, D, n, osu, yc, 'sh_up', outp=_vp(W.o_sh_up, r0, n, ffc))
        self._mk(c, 'glue')
        P.swiglu(osg, osu, xsd, n * ffc)
        self._dense(c, L['sd'], xsd, ffc, n, osd, yc, 'sh_down', reuse=False, outp=_vp(W.o_sh_down, r0, n, D))
        self._mk(c, 'moe')
        gi, ui, di = L['g'], L['u'], L['d']
        od = _v(W.o_down, r0, n, k * D)
        if self.moe == 'own':   # caminus.prefill.moe: one grouping, the fused gate / up + swiglu + down-input quantize, the down GEMM
            O = self.O[c]; jg, jd = _moe.J_GU, _moe.j_down(n)
            O.group(ids, n, k, E, W.bounds, W.ids_dst, W.ids_inv, W.moe_ta, jg, W.moe_tb, jd, W.moe_nt, W.err)
            M.quantize_scatter(mixed, D, n, W.ids_inv, k, W.y_moe, M.layout(gi['fmt']))
            O.gu(gi['fmt'], gi['w'], ui['w'], E, gi['rows'], D, W.y_moe, n * k, W.moe_ta, W.moe_nt[0:1], W.y_down, W.err, jg, di['fmt'])
            O.mm(di['fmt'], di['w'], E, di['rows'], ffc, W.y_down, n * k, W.ids_dst, W.moe_tb, W.moe_nt[1:2], od, W.err, jd)
            self._mk(c, 'glue')
            P.moe_combine(od, wts, osd, sgl, osd, n, k, D)
            return W.o_sh_down
        og, ou, xmd = _v(W.o_gate, r0, n, k * ffc), _v(W.o_up, r0, n, k * ffc), _v(W.x_moe_down, r0, n, k * ffc)
        M.group(ids, n, k, E, W.ids_inv, W.ids_dst, W.bounds, False, W.err)
        M.quantize_scatter(mixed, D, n, W.ids_inv, k, W.y_moe, M.layout(gi['fmt']))
        M.mm_id(gi['fmt'], gi['w'], E, gi['rows'], D, W.y_moe, n, k, W.ids_dst, W.bounds, og)
        M.mm_id(ui['fmt'], ui['w'], E, ui['rows'], D, W.y_moe, n, k, W.ids_dst, W.bounds, ou)
        self._mk(c, 'glue')
        P.swiglu(og, ou, xmd, n * k * ffc)
        self._mk(c, 'moe')
        M.group(ids, n, k, E, W.ids_fwd, W.ids_dst, W.bounds, True, W.err)
        M.quantize(xmd, ffc, n * k, W.y_down, M.layout(di['fmt']), ids=W.ids_fwd)
        M.mm_id(di['fmt'], di['w'], E, di['rows'], ffc, W.y_down, n, k, W.ids_dst, W.bounds, od)
        self._mk(c, 'glue')
        P.moe_combine(od, wts, osd, sgl, osd, n, k, D)
        return W.o_sh_down

    def _half(self, c, kind, il, r0, n, inputs, hist):
        """queue one half-layer of rows [r0, r0 + n) on card c (its compute stream is current); returns the partial buffer"""
        L = self.L[c][il]; G, W, D, HC = self.GW[c], self.W[c], self.D, self.HC
        self._il = il
        if kind == 'ffn':
            self._hc_mix(c, L['hc_ffn'], r0, n, _v(W.o_blk, r0, n, D))
            if hist is not None: hist['il'] = il
            return self._ffn(c, L['ffn'], r0, n, hist)
        post = _v(W.o_sh_down, r0, n, D)                              # the previous layer's FFN exchange (its sum)
        if il == 0:
            self._mk(c, 'glue')
            emb = inputs[0][r0:r0 + n]
            _v(G.res, r0, n, HC * D).view(n, HC, D).copy_(emb[:, None, :].expand(n, HC, D))   # the HC streams start as the embedding
            post = None
        elif il in self.m.ple_layers:
            self._mk(c, 'glue')
            self.Gl[c].hc_post(post, _v(G.inj, r0, n, HC), _v(G.res, r0, n, HC * D), n, HC)
            self._ple(c, L['ple'], inputs[1][r0:r0 + n], r0, n)
            post = None
        self._hc_mix(c, L['hc_attn'], r0, n, post)
        if il in self.feat and c == self.fcard and self.attn == 'core':   # the residual entering layer il (OP_HC_AVG's tap)
            self._pos(('feat', c, il, r0), lambda: self._feat(c, il, r0, n))
        return self._gdn(c, L['gdn'], r0, n) if 'gdn' in L else self._attn(c, L['attn'], r0, n)

    def _feat(self, c, il, r0, n):
        """the DFlash features of the rows [r0, r0 + n) inside the export's window [feat_base, + feat_rows) (OP_HC_AVG of the
        residual entering layer il), from self.p0 / self.feat_base at the call (a captured graph's patch reruns it)"""
        G, HC, D = self.GW[c], self.HC, self.D
        a0, a1 = max(self.p0 + r0, self.feat_base), min(self.p0 + r0 + n, self.feat_base + self.feat_rows)
        if a1 > a0:   # (the rows inside the export's window)
            self._mk(c, 'glue')
            fi = self.feat.index(il); F = self.AW[c].feats
            self.A[c].hc_avg(_v(G.res, a0 - self.p0, a1 - a0, HC * D), F[fi].view(-1)[(a0 - self.feat_base) * D:], a1 - a0, HC, D)

    def _pos(self, key, fn):
        """run fn, the launches of one position-dependent region (the attention core of a piece, its feature export): a
        captured graph's recorder (caminus.prefill.graph) records them as the nodes it patches on every replay"""
        if self.rec is None: fn()
        else: self.rec.region(key, fn)

    # -- one chunk ----------------------------------------------------------------------------------------------------------
    def run(self, hc, m=2, mode='overlap', hist=None, timeline=None, graphs=None):
        """queue the chunk of HostChunk hc (feed.host_chunk: positions [hc.p0, hc.p1), at most T tokens) on both cards, in m
        micro-chunks, the exchange in `mode` (MODES). m = 2 by default, the fastest measured (11-04, chunk 2048, warm, the
        exchange overlapped: 695 / 849 / 940 ms at m 2 / 3 / 4; serial m 1 911; no exchange 616 / 634 / 752 / 815 at m 1 .. 4:
        below 1024 tokens a piece, the MoE GEMMs at ~20 tokens per expert and the WMMA GEMMs lose more than the overlap
        hides). The recurrent states continue from what m.state holds. hist: test
        instrumentation, {'ids': [card][layer] int32 [T, k], 'wts': ...} gets every layer's routing. Returns the micro
        bounds; the caller synchronizes, then checks the routing flag words (check_routing) and the feed's (feed.check2).
        graphs: a caminus.prefill.graph.Graphs of this stack: the 48 layers (everything after the feed) replayed from a
        captured HIP graph of this chunk's structure (its position-dependent nodes patched first), captured now if the
        cache takes it, else run as usual (hist / timeline: always eager)"""
        if mode not in MODES: raise ValueError(f'ChunkStack.run: mode {mode!r} (one of {MODES})')
        T = hc.p1 - hc.p0
        if not 1 <= T <= self.T: raise ValueError(f'ChunkStack.run: a chunk of {T} tokens (1 .. {self.T})')
        if self.attn == 'core':   # positions are the KV / IDXRAW / feature rows this chunk writes: checked before anything is queued
            from ..engine import k3
            if self.m.poisoned: raise k3.ContextBoundsError(f'ChunkStack.run: the engine is poisoned ({self.m.poisoned}): reset it first')
            k3.check_span(hc.p0, T, self.m.max_ctx, what='ChunkStack.run (the KV rows)')
            if hc.p1 > pa.qsa_exact() and not self.qsa:
                raise k3.ContextBoundsError(f'ChunkStack.run: positions {hc.p0} .. {hc.p1 - 1} reach {pa.qsa_exact()}: dense attention '
                                            f"equals the model's QSA only below it, and this engine has no QSA (FlashNextTP qsa)")
            if self.qsa:   # pools j < _pool_from are valid; the chunk pools the rest up to p1 / 4 (its first piece from _pool_from)
                self._pool_from = min(self.m._pool_ok, int(hc.p0) // 4)
        self.p0 = int(hc.p0)
        B = micro_bounds(T, m)
        self.tl = timeline
        prev = [torch.cuda.current_stream(d) for d in self.devs]
        for s in self.ks: torch.cuda.set_stream(s)
        try:
            for c in (0, 1): self._mk(c, 'feed')
            outs = self.feed.device_chunks(hc)
            inputs = [None, None]
            for i, dv in enumerate(self.feed.devs2): inputs[self.devs.index(dv)] = outs[i]
            if graphs is not None and hist is None and timeline is None: graphs.run(self, B, inputs, mode)
            else: self._body(B, inputs, mode, hist, self._mk if timeline is not None else None)
            for c in (0, 1): self._mk(c, 'end')
            if self.attn == 'core' and self.qsa: self.m._pool_ok = int(hc.p1) // 4   # (queued: every pool j < p1 / 4 from the chunk's rows)
        finally:
            for s in prev: torch.cuda.set_stream(s)
            torch.cuda.set_device(self.devs[0]); self.tl = None
        return B

    def halves(self):
        return [(kind, il) for il in range(self.nl) for kind in ('blk', 'ffn')]

    def _body(self, B, inputs, mode, hist=None, mark=None, launch=None):
        """the 48 layers of the chunk (after the feed) on both cards' compute streams (current), micro-chunks B, the exchange
        in `mode`. launch(c, kind, il, j): card c's compute of half (kind, il) of piece j (kind 'post': the last hc_post) as
        a captured graph's segment (caminus.prefill.graph), returning the partial buffer; None: the launches themselves.
        The exchange between the halves is the same calls either way"""
        pend = [None] * len(B)
        for kind, il in self.halves():
            for j, (r0, r1) in enumerate(B):
                if pend[j] is not None: self.X.sum(pend[j], mark); pend[j] = None
                parts = []
                for c in (0, 1):
                    with torch.cuda.device(self.devs[c]):
                        parts.append(self._half(c, kind, il, r0, r1 - r0, inputs[c], hist) if launch is None else launch(c, kind, il, j))
                if mode == 'local': continue
                t = self.X.post(parts, r0, r1)
                if mode == 'serial': self.X.sum(t, mark)
                else: pend[j] = t
        for j, (r0, r1) in enumerate(B):
            if pend[j] is not None: self.X.sum(pend[j], mark)
            for c in (0, 1):
                with torch.cuda.device(self.devs[c]):
                    if launch is None: self._post(c, r0, r1 - r0)
                    else: launch(c, 'post', -1, j)

    def _post(self, c, r0, n):
        """the chunk's last hc_post on card c (after the last FFN exchange): the residual streams the next chunk reads"""
        self._mk(c, 'glue'); G, W = self.GW[c], self.W[c]
        self.Gl[c].hc_post(_v(W.o_sh_down, r0, n, self.D), _v(G.inj, r0, n, self.HC), _v(G.res, r0, n, self.HC * self.D), n, self.HC)

    def structure(self, B, mode):
        """the launch structure of a chunk at the current self.p0 / feat_base / _pool_from with micro-chunks B: everything
        that decides which kernels the 48 layers launch (a captured graph is replayed only for an equal structure; the
        positions, grids and feature offsets that differ are patched): per piece the dense and QSA rows of the attention
        core, whether it pools (QSA), whether it exports features"""
        q = pa.qsa_exact(); pieces = []
        for r0, r1 in B:
            p, n = self.p0 + r0, r1 - r0
            b = max(p, min(p + n, q))
            pool = self.qsa and (((p + n) // 4) > (self._pool_from if r0 == 0 else p // 4))
            feat = bool(self.feat) and max(p, self.feat_base) < min(p + n, self.feat_base + self.feat_rows)
            pieces.append((b - p, p + n - b, bool(pool), feat))
        return (tuple(B), mode, self.moe, self.q6k, self.attn, self.qsa, self.Gl[0].router_mode, self.Gl[0].wmma_out, self.rpb, tuple(pieces))

    def sync(self):
        for d in self.devs: torch.cuda.synchronize(d)
        self.X.settle()

    def check_routing(self, what='prefill chunk'):
        """raise prefill_ops.RoutingError if either card's router or id clamp flagged the chunk (waits for both)"""
        for c in (0, 1):
            po.check_routing(self.GW[c].err, self.W[c].err, what=f'{what}, card {c}')

    def check_attn(self, what='prefill chunk'):
        """raise k3.ContextBoundsError if either card's attention core refused a row (waits for both; 'core' only)"""
        for c in (0, 1):
            if self.AW[c] is not None: self.AW[c].check(f'{what}, card {c}')

    def features(self, p0, p1):
        """the DFlash features of positions [p0, p1) on the feature card: [layers, p1 - p0, D] (a view of the export, which
        holds [feat_base, feat_base + feat_rows))"""
        if self.AW[self.fcard] is None or not self.feat: raise ValueError('ChunkStack.features: no feature export (attn core with feature layers)')
        p0, p1, fb = int(p0), int(p1), self.feat_base
        if not fb <= p0 <= p1 <= fb + self.feat_rows:
            raise ValueError(f'ChunkStack.features: positions {p0} .. {p1 - 1} outside the export [{fb}, {fb + self.feat_rows})')
        return self.AW[self.fcard].feats[:, p0 - fb:p1 - fb]

    def residual(self, c, T):
        """card c's residual streams after the chunk: [T, HC, D] (a view of its glue workspace)"""
        return self.GW[c].res[:T * self.HC * self.D].view(T, self.HC, self.D)
