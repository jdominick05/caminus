"""The Coder's batched prefill plan: every argument that sizes caminus.prefill.coder.Prefill's device buffers, stated once.

The buffers are carved by layout functions of their arguments (mmq.workspace_layout, dense.bf16_layout, feed.feed_layout,
glue.glue_layout, tp.xchg_layout, attn.attn_layout). A PrefillPlan holds those arguments for one engine and one chunk, so
the projection (capacity.prefill_need_mib / budget, FlashNextTP's VRAM check, tools/k3_capacity.py) and the allocations
(ChunkStack's workspaces and exchange, PrefillFeed's two-card buffers) read the same values:

  g, chunk               the engine's geometry (capacity.geometry) and the largest chunk (tokens)
  gdn_keys, moe_blocks   the engine's TP split (FlashNextTP's: card 0's GDN key heads and expert blocks)
  T_pass                 the engine's tokens per pass (FlashNextTP.ntok): the feature export's and the records' rows
  n_feat, feature_card   the DFlash feature export's target layers and their card (FlashNextTP feature_layers / feature_card)
  qsa_ctx                a QSA engine's max_ctx (the QSA prompt path's buffers), else 0
  bf16_dense             whether the 'wmma' dense Q6_K path's bf16 buffers are allocated (dense.Bf16Workspace)
  rows_per_block         the split attention's rows per record block (attn.ROWS_PER_BLOCK)
  feed, table_card       the two-card feed's shapes (FeedShape: from the GGUF, feed.feed_shape) and the token_embd table's
                         card (engine order: capacity.PREFILL_TABLE_CARD, the XT)

Per-card values are in engine order (card 0 = m.devs[0], the XTX). Before 11-07 the projection and the allocations derived
these themselves: the feed's shapes were the Coder's literals in capacity but the GGUF's in the feed, T_pass was 4 in the
projection and m.ntok at allocation, the bf16 buffers followed the global Q6K_PATH in the projection but the constructor's
q6k / q6k_buffers at allocation, the table card was engine index 1 in one place and the feed's index 0 in the other."""
from collections import namedtuple

from ..device import capacity

# the two-card feed's shapes (caminus.prefill.feed): D (token_embd's row width), the PLE heads per position, the PLE vector
# width, the PLE and token_embd raw row bytes (IQ4_NL / IQ4_XS), the token_embd table's bytes
FeedShape = namedtuple('FeedShape', 'D ple_heads ple_dim ple_rb te_rb table_bytes')

# the GSQ-RCO Coder's (what capacity.budget counts without a GGUF; table_bytes as capacity's literal before 11-07: the file's
# table is 337,715,200 bytes, 248,320 rows x 1,360, feed.feed_shape)
CODER_FEED = FeedShape(D=2560, ple_heads=16, ple_dim=160, ple_rb=90, te_rb=1360, table_bytes=337_740_800)

_FIELDS = 'g chunk gdn_keys moe_blocks T_pass n_feat feature_card qsa_ctx bf16_dense rows_per_block feed table_card'


class PrefillPlan(namedtuple('PrefillPlan', _FIELDS)):
    __slots__ = ()

    @classmethod
    def of(cls, g, chunk, *, gdn_keys=None, moe_blocks=None, T_pass=4, n_feat=0, feature_card=None, qsa_ctx=0, bf16_dense=None,
           rows_per_block=None, feed=None, table_card=None):
        """a plan from its arguments; the split has no default (the layouts that depend on it refuse a missing one:
        capacity._split); bf16_dense None = the default dense path's (capacity.prefill_dense_bf16: prefill_tp.Q6K_PATH ==
        'wmma'); feed None = CODER_FEED with the geometry's D"""
        from . import attn as pa
        gk = None if gdn_keys is None else capacity._split(gdn_keys, 1)[0]
        mb = None if moe_blocks is None else capacity._split(1, moe_blocks)[1]
        fs = CODER_FEED._replace(D=g['D']) if feed is None else FeedShape(*feed)
        return cls(g=g, chunk=int(chunk), gdn_keys=gk, moe_blocks=mb, T_pass=int(T_pass), n_feat=int(n_feat),
                   feature_card=capacity.FEATURE_CARD if feature_card is None else int(feature_card), qsa_ctx=int(qsa_ctx),
                   bf16_dense=capacity.prefill_dense_bf16(bf16_dense), rows_per_block=pa.ROWS_PER_BLOCK if rows_per_block is None else int(rows_per_block),
                   feed=fs, table_card=capacity.PREFILL_TABLE_CARD if table_card is None else int(table_card))

    @classmethod
    def for_engine(cls, m, chunk, *, bf16_dense=None, rows_per_block=None, feed=None):
        """the plan of a FlashNextTP engine m's prefill at chunk: its geometry, split, tokens per pass, feature layers and
        card, and its max_ctx when it runs QSA"""
        return cls.of(m.geom, chunk, gdn_keys=m.gdn_keys, moe_blocks=m.moe_blocks, T_pass=m.ntok, n_feat=len(getattr(m, 'feature_layers', ()) or ()),
                      feature_card=int(getattr(m, 'feature_card', 0)), qsa_ctx=m.max_ctx if getattr(m, 'qsa', False) else 0,
                      bf16_dense=bf16_dense, rows_per_block=rows_per_block, feed=feed)

    # -- per card: the layouts' arguments --------------------------------------------------------------------------------
    def card_dims(self, c):
        from .mmq import card_dims
        return card_dims(self.g, c, *capacity._split(self.gdn_keys, self.moe_blocks))

    def glue_dims(self, c):
        from .glue import glue_dims
        return glue_dims(self.g, c, capacity._split(self.gdn_keys, 1)[0])

    def attn_args(self, c):
        """AttnWorkspace's / attn_layout's arguments on card c (the feature export on feature_card only)"""
        return dict(T=self.chunk, nh=self.g['nh'] // self.g['nkv'], hd=self.g['hd'], idx_dim=self.g['idx_dim'],
                    rows_per_block=self.rows_per_block, n_feat=self.n_feat if c == self.feature_card else 0, D=self.g['D'],
                    T_pass=self.T_pass, qsa_ctx=self.qsa_ctx)

    def feed_layout(self, c):
        """the two-card feed's buffers on engine card c (the token_embd table on table_card only)"""
        from .feed import feed_layout
        f = self.feed
        return feed_layout(self.chunk, f.D, f.ple_heads, f.ple_dim, f.ple_rb, f.te_rb, f.table_bytes if c == self.table_card else 0)

    # -- per card: bytes ---------------------------------------------------------------------------------------------------
    def workspace_bytes(self):
        """the GEMM workspace (mmq.Workspace), with the bf16 dense buffers (dense.Bf16Workspace) when bf16_dense"""
        from .mmq import workspace_bytes
        from .dense import bf16_bytes
        return tuple(workspace_bytes(self.chunk, self.card_dims(c)) + (bf16_bytes(self.chunk, self.card_dims(c)) if self.bf16_dense else 0)
                     for c in (0, 1))

    def feed_bytes(self):
        return tuple(sum(self.feed_layout(c).values()) for c in (0, 1))

    def glue_bytes(self):
        from .glue import glue_bytes
        return tuple(glue_bytes(self.chunk, self.glue_dims(c)) for c in (0, 1))

    def xchg_bytes(self):
        from .tp import xchg_bytes
        return tuple(xchg_bytes(self.chunk, self.g['D']) for c in (0, 1))

    def attn_bytes(self):
        from .attn import attn_bytes
        return tuple(attn_bytes(**self.attn_args(c)) for c in (0, 1))

    def need_mib(self, graphs=0):
        """per card MiB the batched prefill adds: GEMM workspace (+ the bf16 dense buffers) + feed + glue workspace + the
        exchange's receive buffer + the attention core's workspace + its modules (capacity.PREFILL_CODE_MIB) + its streams
        (PREFILL_STREAMS_MIB) + graphs captured chunk structures (PREFILL_GRAPH_MIB each; caminus.prefill.graph)"""
        w, f, gl, x, a = self.workspace_bytes(), self.feed_bytes(), self.glue_bytes(), self.xchg_bytes(), self.attn_bytes()
        return tuple((w[c] + f[c] + gl[c] + x[c] + a[c]) / capacity.MIB + capacity.PREFILL_CODE_MIB[c] + capacity.PREFILL_STREAMS_MIB[c]
                     + int(graphs) * capacity.PREFILL_GRAPH_MIB[c] for c in (0, 1))


def check_counted(m, plan, what):
    """refuse (ValueError), before anything is allocated, a prefill that would take more on either card than its engine
    counted before it loaded (m.prefill_plan: FlashNextTP(vram_prefill=...)); the engine's feed shapes stand in for the
    plan's (a ChunkStack built alone has none). An engine built without a planned prefill (None) counted nothing: not
    checked here (each allocation still checks the cards' caps)"""
    c = getattr(m, 'prefill_plan', None)
    if c is None: return
    need, have = plan._replace(feed=c.feed).need_mib(), c.need_mib()
    if any(n > h + 1e-6 for n, h in zip(need, have)):
        bf = ', q6k_buffers=True' if plan.bf16_dense and not c.bf16_dense else ''
        raise ValueError(f'{what}: this prefill needs {need[0]:.0f} / {need[1]:.0f} MiB, its engine counted {have[0]:.0f} / {have[1]:.0f} '
                         f'(chunk {c.chunk}{", bf16 dense" if c.bf16_dense else ""}): build the engine with '
                         f'vram_prefill=dict(chunk={max(plan.chunk, c.chunk)}{bf}) as this prefill is built')
