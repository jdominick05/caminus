"""Context capacity of the K3 TP engine (caminus.engine.flashnext_tp.FlashNextTP): the context-sized buffers per card, the VRAM
budget against the per-card caps, and the exact-attention limit.

Two separate numbers:
- max_ctx: the positions the buffers hold (KV cache rows, QSA indexer caches), up to the model's context_length (262,144).
  The device bound guard follows the allocation (k3.kv_rows).
- QSA_EXACT: full attention over the whole KV equals the model's QSA selection only while the selection covers every
  cell: the indexer's top_k 2048 tokens (512 pools of 4) + a 3-token tail = 2051 cells (ignite-llama
  src/models/qwen4exp.cpp:754). The GSQ-RCO Coder with max_ctx > QSA_EXACT runs the model's QSA (FlashNextTP qsa: the
  indexer + ATTN_SEL) up to max_ctx; the dense tape (REAP-308, max_ctx <= 2051, qsa=False) refuses passes past
  QSA_EXACT (FlashNextTP._check) unless built with dense_beyond_qsa=True: a TEST-ONLY override (the full-attention
  baseline at long context, expected to differ from the model). MTP and DFlash follow their target's limit.

Per card and position (TP: one kv head per card; the indexer replicated), q8_0 as the daily server (-ctk / -ctv q8_0):
  KV       12 attention layers x [K | V] 256 values each as q8_0 = 12 x 544 = 6,528 B
  IDXRAW   12 x the indexer's raw key (128 values, q8_0)       = 12 x 136 = 1,632 B
  IDXPOOL  12 x one pooled key per 4 positions (q8_0)          = 12 x 136 / 4 = 408 B
  => 8,568 B per position per card: 268 / 1,071 / 2,142 MiB at 32K / 128K / 262K (docs/specs/spec-qsa.md section 4).

Refusals before allocation: FlashNextTP(vram_check=True) checks the target (+ vram_drafters: the drafters' drafter_need_mib,
passed by the smoke factories so a config over the caps is refused before the target loads), and MTPK3 / DFlashK3 check
their own need against the device's use at their construction (the target loaded by then), before they allocate.
The fixed part (weights, scratch, tapes, torch's cache after the load) was measured on this machine: guard.sh's peak
device-level VRAM during a load + 3 passes, minus the per-position bytes and the idle baseline (see FIXED_MIB). The caps
are guard.sh's own kill lines (GUARD_LINE_MIB from caminus.device.machine, the same rule as the guard: 0.97 of each gfx1100 card
rounded down to 50 MiB, here XTX 23,800 MiB, XT 19,850 MiB; CAP_FRAC of a card the guard does not watch) less MARGIN_MIB; every live check reads the device's used memory the way the guard does (used_mib: sysfs
mem_info_vram_used, every process, e.g. the desktop on the XTX; hipMemGetInfo reports 163 / 353 MiB less with the Coder
loaded, so it is only the fallback where sysfs cannot be read)."""
import torch

QSA_EXACT = 2051          # cells the QSA selection covers (top_k 2048 + 3-token tail): dense == QSA up to here
QSA_BACKFILL = 2048       # positions per pool-backfill launch of the two-tape QSA engine (FlashNextTP._backfill)
CAP_FRAC = 0.97           # a card guard.sh does not watch: 0.97 of its total (guard.sh's lines were set near 0.97: 23800 / 24560, 19850 / 20464)
MARGIN_MIB = 256          # kept free below the cap (allocator rounding, HIP's own growth)
# tools/guard.sh's kill lines: MiB of sysfs mem_info_vram_used per DRM card of every gfx1100 card it watches, computed from each
# card's total by caminus.device.machine with the guard's own rule (floor(floor(total_MiB * 97 / 100) / 50) * 50): here card1 = the
# XTX 23,800 (0.969 of its 24,560 MiB, 23 MiB below CAP_FRAC x total) and card2 = the XT 19,850 (tests/packaging/test_packaging.py and
# tests/cache/test_prompt_cache.py cpu check these against `guard.sh --lines` and the script)
from . import machine as _machine   # noqa: E402
GUARD_LINE_MIB = _machine.guard_lines()
MIB = 1 << 20


def pass_limit(m):
    """the last position + 1 any pass of engine m may write: its max_ctx and, for the dense tape, the exact-attention limit
    (m.qsa_limit: None = no limit beyond max_ctx; an engine without the attribute counts as limited to QSA_EXACT, as
    FlashNextTP._check: fail closed)"""
    lim = getattr(m, 'qsa_limit', QSA_EXACT)
    return int(m.max_ctx) if lim is None else min(int(m.max_ctx), int(lim))


class CapacityError(ValueError):
    """a configuration whose projected VRAM exceeds a card's cap: refused before anything is allocated"""


def q8_bytes(n):
    """bytes of n values as q8_0 (block_q8_0: 32 values in 34 bytes)"""
    assert n % 32 == 0, n
    return n // 32 * 34


def geometry(path_or_reader):
    """the context-relevant shapes from a qwen4exp GGUF's metadata (no tensor data read)"""
    from ..engine import flashnext  # noqa: F401  (puts gguf-py on the path)
    from gguf import GGUFReader
    rd = GGUFReader(path_or_reader, 'r') if isinstance(path_or_reader, str) else path_or_reader
    f = {x.name: x for x in rd.fields.values()}
    hp = lambda k: f[f'qwen4exp.{k}'].contents()
    nl, fai = int(hp('block_count')), int(hp('full_attention_interval'))
    return dict(n_attn=sum(1 for il in range(nl) if (il + 1) % fai == 0), nkv=int(hp('attention.head_count_kv')),
                hd=int(hp('attention.key_length')), idx_dim=int(hp('attention.indexer.key_length')),
                idx_heads=int(hp('attention.indexer.head_count')),
                ctx_train=int(hp('context_length')), D=int(hp('embedding_length')), hc=int(hp('hyper_connection.count')),
                hc_lr=int(hp('hyper_connection.low_rank')),
                # the batched prefill's GEMM shapes (caminus.prefill.mmq.card_dims)
                nh=int(hp('attention.head_count')), n_exp=int(hp('expert_count')), k_used=int(hp('expert_used_count')),
                ffn=int(hp('expert_feed_forward_length')), ssm_groups=int(hp('ssm.group_count')),
                ssm_heads=int(hp('ssm.time_step_rank')), ssm_state=int(hp('ssm.state_size')))


def tp_kv_bytes(g, max_ctx, q8=True):
    """per card: the TP engine's KV cache bytes for max_ctx positions in its cache format (q8_0 K and V, else fp32: the
    bytes FlashNextTP allocates, kernels/k3_gguf_ops.h kv_pos_bytes; VRAM or, with kv_host, pinned host memory)"""
    m = g['nkv'] // 2 * g['hd']
    return g['n_attn'] * max_ctx * (2 * q8_bytes(m) if q8 else 4 * m)


def tp_ctx_bytes(g, max_ctx, idx=True, q8=True):
    """per card: the TP engine's context-sized buffers (bytes) for max_ctx positions"""
    kv = tp_kv_bytes(g, max_ctx, q8)
    raw = g['n_attn'] * q8_bytes(g['idx_dim']) * max_ctx if idx else 0
    pool = g['n_attn'] * q8_bytes(g['idx_dim']) * ((max_ctx + 3) // 4) if idx else 0
    return dict(kv=kv, idxraw=raw, idxpool=pool)


HOST_KV_MARGIN_MIB = 4096   # host RAM left for everything else while the KV cache is pinned (kv_host)


def host_kv_room():
    """(MemAvailable, the GTT page limit (ttm pages_limit: what all pinned GPU-mapped host memory may total), in MiB"""
    ma = None
    for l in open('/proc/meminfo'):
        if l.startswith('MemAvailable:'): ma = int(l.split()[1]) / 1024
    try: lim = int(open('/sys/module/ttm/parameters/pages_limit').read()) * 4096 / MIB
    except OSError: lim = None
    return ma, lim


def check_host_kv(nbytes):
    """kv_host: refuse (CapacityError) a host KV cache that would leave less than HOST_KV_MARGIN_MIB of MemAvailable or pass
    the GTT page limit, before anything is allocated"""
    need = nbytes / MIB; ma, lim = host_kv_room()
    if ma is not None and need + HOST_KV_MARGIN_MIB > ma:
        raise CapacityError(f'host KV cache {need:.0f} MiB + {HOST_KV_MARGIN_MIB} MiB margin > MemAvailable {ma:.0f} MiB')
    if lim is not None and need > 0.95 * lim:
        raise CapacityError(f'host KV cache {need:.0f} MiB > 95 % of the GTT page limit {lim:.0f} MiB (ttm pages_limit)')
    return need


def tp_qsa_bytes(g, max_ctx, ntok=4):
    """per card: what the QSA tapes add beyond the indexer caches (FlashNextTP(qsa)): the replicated indexer weights
    (k_proj [idx_dim] + q_proj [idx_heads * idx_dim] rows of D BF16, the two norms fp32) of every attention layer, the
    pool-score scratch [T][max_ctx / 4] fp32, the selection rows [layers][T][516] int32, the indexer activations and the
    two-tape backfill's fp32 rows [layers][min(QSA_BACKFILL, max_ctx)][idx_dim], and the parallel selection's scratch (the
    slices' id lists [T][P x slice] and counts, each layer's histograms [T][2][256])"""
    na, di, hi, D = g['n_attn'], g['idx_dim'], g['idx_heads'], g['D']
    w = na * ((di + hi * di) * D * 2 + 2 * di * 4)
    prow = (max_ctx + 3) // 4; P = min(64, -(-prow // 2048)); LS = P * ((-(-prow // P) + 255) // 256 * 256)
    return dict(qsa_weights=w, qsa_score=ntok * prow * 4, qsa_sel=na * ntok * 516 * 4, qsa_act=ntok * (di + 2 * hi * di) * 4,
                qsa_backfill=na * min(QSA_BACKFILL, max_ctx) * di * 4,
                qsa_psel=ntok * (LS + 2 * P) * 4 + na * ntok * 512 * 4)   # (Engine.qsa_select_par: slice lists + counts, per-layer counts)


# measured fixed VRAM per card (MiB): (XTX, XT), device-level peak during the load and 3 passes, minus the idle baseline and
# the context-sized buffers. tools/k3_capacity.py --measure (11-02, f704302 + this change, logs in
# $ARTIFACTS/qsa/logs/cap_*.log, not distributed). Keys: (model, ntok).
# The allocator reuses blocks freed during the load, so the growth with max_ctx is not exactly the buffers' bytes (Coder
# T=4, 2048 -> 32768: XTX +44, XT +282 MiB for +251 MiB of buffers): the max of the measured fixed parts, rounded up, and the
# full per-position bytes on top (conservative).
FIXED_MIB = {
    ('coder', 1): (18950, 12480),     # measured 18935 / 12441 (max_ctx 2048)
    ('coder', 4): (18950, 12480),     # measured 18939 / 12445 (2048), 18732 / 12476 (32768)
    ('reap', 1): (22480, 14720),      # measured 22469 / 14709 (2048)
    ('reap', 4): (22480, 14720),      # measured 22475 / 14713 (2048)
}
# the measured models' identity: the tensor bytes of the GGUF the engine opens (its first shard; FlashNextK3.rd). A model
# whose bytes differ (another cut or quantization behind the same module) has no measured fixed part: fixed_mib -> None and
# the runtime check refuses it (tools/k3_capacity.py measure, then add it here; or vram_check=False)
MODEL_BYTES = {'coder': 29_597_422_080, 'reap': 65_389_939_200}
MEASURED_SPLIT = (10, 12)    # (gdn_keys, moe_blocks) of the measurements above: XTX as engine card 0, XT as card 1
# T above the measured 4: the fixed part grows by the larger of the measured T 1 -> 4 slope (~1.3 MiB per token, rounded
# away in FIXED_MIB) and this per extra token (the T-sized scratch: logits shards, activations, routing; generous)
EXTRA_TOKEN_MIB = 16
# the drafters as built today (they follow the target's pass limit): fixed MiB per card (measured as the load's delta at
# max_ctx 2048) and their context-sized buffers (analytic, from their allocations: mtp.MTPK3.__init__, dflash_k3._buffers)
# measured (Coder, T=4, max_ctx 2048, after 3 passes): + MTP XTX 789 / XT 720 MiB, + DFlash XTX 4 / XT 1034 MiB, less their
# context-sized buffers at 2048 (MTP 86 / 86, DFlash ring of 4096 rows 0 / 80), rounded up. The DFlash ring is a fixed 4096 rows
# (dflash_ring) since impl:dflash-ring (before: the power of two >= max(4096, max_pos), 5,120 MiB at 262,144). DFlash's XTX
# part is its allocations there, not the measured delta: the remote inject's weight copy 59.8 MiB + the head-shift rows
# 49.2 MiB + head / inject buffers ~3 MiB = 112 MiB (+ tapes, exchange slots), rounded up to 120. The measured delta (+4 MiB
# at 2048 and 32768, 11-03) hides them in blocks the target's load freed; at 262,144 the device total came out 7 MiB above
# the old prediction (XTX 21,269 measured vs 18950 + 2193 + 10 + 110 idle = 21,263): reuse is not guaranteed. XT re-measured
# 11-03 at 2048 after a 32-token generation: +1,036 MiB = 956 + the 80 MiB ring (960 kept)
DRAFTER_FIXED_MIB = {'mtp': (710, 640), 'dflash': (120, 960)}


# the DFlash K / V ring (caminus.decode.dflash): a fixed number of rows, independent of max_ctx. The drafter's attention reads a
# per-query window of W = 2048 positions (non-causal over its block), so a launch touches at most W + 2 T - 2 positions
# (the reads from i0 + 2 - W, the inject rows i0 .. and the draft block up to i0 + 2 T - 1). DFLASH_RING_MIN = 4096 = 2 W:
# the power of two above W + 2 T for every T <= 8. No reason for 8192, by construction (the attention walks logical positions
# and reads the same 2048 rows per query whatever the ring's size; slots are only addresses), and 4096 is verified bitwise
# against a full ring (tests/decode/test_longctx.py lc-ring 8192 / 32768: every draft id equal); an 8192 ring was not run
DFLASH_RING_MIN = 4096
DFLASH_WINDOW = 2048      # the drafter GGUF's dflash.attention.sliding_window (dflash_geom's 4th entry, when given)


def dflash_ring(window=DFLASH_WINDOW, T=8):
    """the DFlash ring's rows: the power of two >= max(DFLASH_RING_MIN, 2 window, window + 2 T) (never max_ctx: the slots
    whose rows a later launch can still read are tracked on the host, dflash_k3.RingSlots, and a snapshot copies the ring)"""
    if isinstance(window, bool) or not isinstance(window, int) or window < 1: raise ValueError(f'dflash_ring: window {window!r}')
    if isinstance(T, bool) or not isinstance(T, int) or not 1 <= T <= 64: raise ValueError(f'dflash_ring: T {T!r}')
    r = DFLASH_RING_MIN
    while r < max(2 * window, window + 2 * T): r *= 2
    return r


def _dflash_window(dflash_geom):
    dg = tuple(dflash_geom or ())
    return int(dg[3]) if len(dg) > 3 else DFLASH_WINDOW


def dflash_ring_bytes(nl, nkv, hd, ring):
    """bytes of the DFlash ring: nl layers x ring rows x [K | V] x nkv x hd f16 (= dflash_k3.DFlashK3.cache)"""
    return nl * ring * 2 * nkv * hd * 2


def drafter_ctx_bytes(kind, g, max_ctx, ntok=4, dflash_geom=None, ring=None):
    """per card (XTX, XT) bytes of a drafter's context-sized buffers at max_ctx, as coded today.
    MTP: H rows (max_ctx + T + 1) x hc*D fp32 on both cards (the target's h of every verified position) + its f16 KV of one
    kv head per card (max_ctx + T rows). DFlash: the f16 K/V ring on the XT, a fixed dflash_ring(window, T) rows whatever
    max_ctx (ring: the drafter's own when given), nl layers x nkv x hd x [K | V] x 2 B per row; dflash_geom = (nl, nkv,
    hd[, window]) (window default DFLASH_WINDOW). Every other DFlash buffer is in DRAFTER_FIXED_MIB (none grows with
    max_ctx: the attention scratch follows the window, the head / exchange buffers the vocabulary)."""
    if kind == 'mtp':
        rows = max_ctx + ntok
        b = (rows + 1) * g['hc'] * g['D'] * 4 + rows * 4 * (g['nkv'] // 2 * g['hd'])
        return (b, b)
    if kind == 'dflash':
        nl, nkv, hd = tuple(dflash_geom or (5, 8, 128))[:3]
        if ring is None: ring = dflash_ring(_dflash_window(dflash_geom), int(ntok))
        return (0, dflash_ring_bytes(nl, nkv, hd, ring))
    raise ValueError(kind)


def drafter_need_mib(kind, g, max_ctx, ntok=4, dflash_geom=None, ring=None):
    """per card (XTX, XT) MiB a drafter adds: its measured fixed part + its context-sized buffers"""
    f, b = DRAFTER_FIXED_MIB[kind], drafter_ctx_bytes(kind, g, max_ctx, ntok, dflash_geom, ring)
    return tuple(f[c] + b[c] / MIB for c in (0, 1))


# the batched prefill (docs/specs/spec-prefill-cache.md M5): its VRAM per card, preallocated once for the largest chunk.
# The GEMM workspace (caminus.prefill.mmq.Workspace: quantized activations, mul_mat_id's tables, every GEMM output of a layer), the
# feed's (caminus.prefill.feed, two-card mode: the IQ4_XS token_embd table on one card, the chunk's rows and vectors on
# both) and the glue workspace (caminus.prefill.glue.GlueWorkspace, S3: the residual stream, the hc norms and their bf16
# copies, the hc gate, the router's and the thin BF16 matrices' outputs) and the exchange's receive buffer
# (caminus.prefill.tp.Exchange, S5: the peer's partial rows). All of them functions of one prefill plan
# (caminus.prefill.plan.PrefillPlan), which FlashNextTP(vram_prefill=...) builds and counts before the target allocates, and
# checked again by each allocation
PREFILL_TABLE_CARD = 1     # the token_embd table's card: the XT, which has the most headroom at 262,144 + DFlash
# the prefill's code objects on each card (the 11 prefill_mmq_* modules, prefill_glue, prefill_gdn, prefill_tp and gemm:
# ~1.1 MB of ELF; since impl:moe-gemm also the 7 prefill_moe_* modules, ~1.4 MB; since impl:q6k-dense prefill_dense_q6_k, 0.27 MB) as HIP
# loads them: measured +1.3 / +1.3 MiB device-level (impl:prefill-safety, test_prefill_mmq tp at 262,144 + DFlash,
# 11-04), the prefill_moe modules +1.5 MiB more (the XT alone, 11-04), counted with room for growth. What else the 11-03 comparison (XT 17,078 measured vs 16,772 projected) was
# missing, measured the same way: the card's own use before the process started (XT 271 / XTX 110 MiB: the desktop and
# the driver, read live by check() and subtracted by the k3_capacity table, but not in a projection's total) and the
# GEMMs' fp32 inputs, which S2's test allocated outside every count (+82 MiB on the XT; now in mmq.workspace_layout)
PREFILL_CODE_MIB = (5, 5)
# the batched prefill's own HIP streams (caminus.prefill.tp: a compute and a copy stream per card): each takes device memory
# at its first use (its queue's resources), measured device-level (impl:prefill-s5, 11-04, the loaded Coder at max_ctx 2048:
# the first kernel on each compute stream +51 / +45 MiB, the first copy on each copy stream +44 / +38 MiB, XTX / XT; nothing
# more over later chunks), counted with ~10 % room; ChunkStack / Exchange use their streams once at construction so the
# memory is taken before the chunk loop starts
PREFILL_STREAMS_MIB = (104, 96)
# one captured chunk structure of HIP graphs (caminus.prefill.graph: 194 per-card segment graphs, 388 at 2 micro-chunks) with
# HIP's packet capture off (prefill_graph.ENV; on, each graph takes a 2 MiB kernel-argument pool: ~390 MiB per structure).
# Measured 11-04: 200 graphs of 25 nodes 2.0 MiB (10 KiB each; the probe), and every structure of tests/prefill/test_prefill_graph.py
# (127 .. 2047-token chunks, 4,245 .. 10,026 nodes) below hipMemGetInfo's resolution (0.0 MiB per card); counted at 8 MiB per
# card per structure held, and Graphs refuses (drops) a structure whose measured instantiation exceeds it
PREFILL_GRAPH_MIB = (8, 8)
# the prompt cache's VRAM tier (caminus.cache.prompt, M6): sized at its construction from each card's measured headroom below the
# guard's own line (GUARD_LINE_MIB: XTX 23,800 / XT 19,850 MiB; M6 fix-ups: not CAP_FRAC x total, 23,823 on the XTX) less
# CACHE_RESERVE_MIB (the XTX's 0.5 GB reserve, which the tier must never breach: the desktop's own VRAM grows; the XT: the usual
# MARGIN_MIB) less CACHE_WORK_MIB (in-run growth after the measurement: M6 unit 1 measured +21 .. +37 MiB on the XT in-process),
# the use read as the guard reads it (used_mib), then counted like any allocation (cache_tier_mib, checked by check() before the
# slots are allocated)
CACHE_RESERVE_MIB = (512, 256)   # this machine's values by engine order (XTX, XT): cache_reserve_mib() derives them per card
DISPLAY_RESERVE_MIB = _machine.DISPLAY_RESERVE_MIB   # a card that drives a display (caminus.device.machine: a connected connector or boot_vga)
CACHE_WORK_MIB = (128, 128)


def is_display(d):
    """True when torch device d's DRM card drives a display (caminus.device.machine), so it keeps DISPLAY_RESERVE_MIB below its line"""
    c = _machine.by_card(drm_card(d))
    return bool(c and c['display'])


def cache_reserve_mib(devs):
    """per device: the prompt cache tier's reserve below the guard line: DISPLAY_RESERVE_MIB (512) on a display-driving card
    (the desktop's own VRAM grows), MARGIN_MIB on the others; here (512, 256) for (XTX, XT), CACHE_RESERVE_MIB"""
    return tuple(DISPLAY_RESERVE_MIB if is_display(d) else MARGIN_MIB for d in devs)


def cache_tier_mib(slot_bytes, n_slots, side_bytes=(0, 0), extra_slots=1):
    """per card (XTX, XT) MiB of the prompt cache's VRAM tier: n_slots checkpoint slots of slot_bytes[c] each, extra_slots more
    (the io slot that disk loads and parks go through, and the side slot's checkpoint of the main state when there is a side
    slot), and the side slot's row area side_bytes[c] (caminus.cache.prompt.Layout: one slot = the card's recurrent states + the
    tail rows (T positions of every KV / indexer layer) + on the drafter's card the DFlash window rows)"""
    return tuple((int(slot_bytes[c]) * (int(n_slots) + int(extra_slots)) + int(side_bytes[c])) / MIB for c in range(len(slot_bytes)))


def _split(gdn_keys, moe_blocks):
    """the TP split as given (FlashNextTP's gdn_keys / moe_blocks): no default, so a projection never silently assumes
    the default split for an engine built with another one (card 0's share moves MiB between the cards)"""
    for n, v in (('gdn_keys', gdn_keys), ('moe_blocks', moe_blocks)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1: raise ValueError(f'capacity: {n} {v!r} (the engine\'s TP split)')
    return int(gdn_keys), int(moe_blocks)


def prefill_dense_bf16(bf16_dense=None):
    """whether the prefill counts the bf16 dense path's buffers (caminus.prefill.dense.Bf16Workspace): None = when the default dense
    Q6_K path (prefill_tp.Q6K_PATH) is 'wmma', as ChunkStack allocates them"""
    if bf16_dense is not None: return bool(bf16_dense)
    from ..prefill import tp as prefill_tp
    return prefill_tp.Q6K_PATH == 'wmma'


def prefill_workspace_bytes(g, chunk, *, gdn_keys, moe_blocks, bf16_dense=None):
    """per card (XTX, XT): the prefill GEMM workspace bytes for chunks of up to `chunk` tokens, at the engine's split, with the
    bf16 dense path's buffers (caminus.prefill.dense.Bf16Workspace) when prefill_dense_bf16(bf16_dense)
    (caminus.prefill.plan.PrefillPlan.workspace_bytes)"""
    from ..prefill.plan import PrefillPlan
    return PrefillPlan.of(g, chunk, gdn_keys=gdn_keys, moe_blocks=moe_blocks, bf16_dense=bf16_dense).workspace_bytes()


def prefill_feed_bytes(g, chunk, feed=None):
    """per card (XTX, XT): the two-card feed's buffers (caminus.prefill.feed.feed_layout) for chunks of up to `chunk`
    tokens, the token_embd table on PREFILL_TABLE_CARD (feed: its shapes, caminus.prefill.plan.FeedShape; None: the
    GSQ-RCO Coder's, plan.CODER_FEED)"""
    from ..prefill.plan import PrefillPlan
    return PrefillPlan.of(g, chunk, feed=feed).feed_bytes()


def prefill_glue_bytes(g, chunk, *, gdn_keys):
    """per card (XTX, XT): the prefill glue workspace bytes (caminus.prefill.glue.glue_layout) for chunks of up to `chunk`
    tokens, at the engine's split"""
    from ..prefill.plan import PrefillPlan
    _split(gdn_keys, 1)
    return PrefillPlan.of(g, chunk, gdn_keys=gdn_keys).glue_bytes()


def prefill_xchg_bytes(g, chunk):
    """per card (XTX, XT): the prefill exchange's buffers (caminus.prefill.tp.xchg_layout: the receive rows for the peer's
    partial, [chunk, D] fp32) for chunks of up to `chunk` tokens"""
    from ..prefill.plan import PrefillPlan
    return PrefillPlan.of(g, chunk).xchg_bytes()


# the DFlash drafter's target layers (dflash.target_layers of the DFlash2 GGUF: 5): the prefill's feature export holds that
# many rows per position on the feature card (FlashNextTP feature_card, 0 by default)
DFLASH_FEATURE_LAYERS = 5
FEATURE_CARD = 0


def prefill_attn_bytes(g, chunk, n_feat=0, feature_card=FEATURE_CARD, T_pass=4, qsa_ctx=0):
    """per card (XTX, XT): the attention core's workspace (caminus.prefill.attn.attn_layout: the rotated q / k rows, one
    record block of the split attention, the indexer's raw key rows) for chunks of up to `chunk` tokens, on feature_card
    the DFlash feature export of n_feat layers (the last feat_rows positions a prefill writes, ~21 MiB per layer), and on a
    QSA engine of max_ctx qsa_ctx (> 0) the QSA prompt path's buffers (the indexer query rows, the SEL rows and the score
    scratch: query sub-blocks of prefill_attn.score_rows rows of (qsa_ctx + 3) / 4 pools, <= prefill_attn.SCORE_MIB)"""
    from ..prefill.plan import PrefillPlan
    return PrefillPlan.of(g, chunk, n_feat=n_feat, feature_card=feature_card, T_pass=T_pass, qsa_ctx=qsa_ctx).attn_bytes()


def prefill_need_mib(g, chunk, *, gdn_keys, moe_blocks, n_feat=0, feature_card=FEATURE_CARD, bf16_dense=None, qsa_ctx=0, graphs=0):
    """per card (XTX, XT) MiB the batched prefill adds at the engine's split (gdn_keys / moe_blocks, FlashNextTP's): GEMM
    workspace (with the bf16 dense path's buffers when prefill_dense_bf16(bf16_dense)) + feed + glue workspace + the
    exchange's receive buffer + the attention core's workspace (with the DFlash feature export of n_feat target layers on
    feature_card, and the QSA prompt path's buffers of a QSA engine of max_ctx qsa_ctx > 0) + the prefill modules
    (PREFILL_CODE_MIB) + the prefill's streams (PREFILL_STREAMS_MIB) + graphs captured chunk structures (PREFILL_GRAPH_MIB each;
    caminus.prefill.graph). The plan it counts: caminus.prefill.plan.PrefillPlan (the one the allocations read)"""
    from ..prefill.plan import PrefillPlan
    return PrefillPlan.of(g, chunk, gdn_keys=gdn_keys, moe_blocks=moe_blocks, n_feat=n_feat, feature_card=feature_card,
                          bf16_dense=bf16_dense, qsa_ctx=qsa_ctx).need_mib(graphs)


def model_bytes(rd):
    """the tensor bytes of an opened GGUF (its reader: the first shard of a split one), the measured models' identity"""
    return sum(int(t.n_bytes) for t in rd.tensors)


# The TP split moves weights between the cards (D-292): FlashNextTP shards the GDN layers' tensors by key head (card 0 gets
# gdn_keys of the ssm.group_count key heads: attn_qkv, the conv, attn_gate (z), alpha / beta, dt / a, the ssm_out K split)
# and every expert's intermediate in blocks of 32 (card 0 gets moe_blocks of expert_feed_forward_length / 32: the routed and
# the shared expert's gate / up rows and down K split); everything else is per kv head, by vocab_share or replicated, the
# same at every split. Another split's fixed part is the measured one plus what those tensors move, each card's shard
# rounded as torch's allocator rounds a large block (SPLIT_ROUND), plus SPLIT_MARGIN_MIB for what the model leaves out (the
# T-sized scratch per card follows the split; allocator reuse of the load's freed blocks).
SPLIT_GDN = ('attn_qkv.weight', 'ssm_conv1d.weight', 'attn_gate.weight', 'ssm_alpha.weight', 'ssm_beta.weight', 'ssm_dt.bias',
             'ssm_a', 'ssm_out.weight')
SPLIT_MOE = ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight', 'ffn_gate_shexp.weight', 'ffn_up_shexp.weight',
             'ffn_down_shexp.weight')
SPLIT_ROUND = 2 * MIB          # torch's caching allocator: a block of >= 10 MiB is rounded up to 2 MiB (kRoundLarge)
SPLIT_MARGIN_MIB = 256
SLOT_PAD_MIB = 2.0         # a projected checkpoint slot's own rounding (arch.Coder.SLOT_MIB, measured at the tuned split)


def split_tensors(rd):
    """the split-dependent tensors of an opened Coder GGUF (its reader): dict(gdn=[bytes], moe=[bytes], keys=key heads,
    blocks=intermediate blocks of 32), from the header alone"""
    f = {x.name: x for x in rd.fields.values()}
    a = f['general.architecture'].contents()
    out = dict(gdn=[], moe=[], keys=int(f[f'{a}.ssm.group_count'].contents()),
               blocks=int(f[f'{a}.expert_feed_forward_length'].contents()) // 32)
    for t in rd.tensors:
        n = t.name.split('.', 2)[-1] if t.name.startswith('blk.') else None
        if n in SPLIT_GDN: out['gdn'].append(int(t.n_bytes))
        elif n in SPLIT_MOE: out['moe'].append(int(t.n_bytes))
    return out


def split_weight_mib(st, split):
    """per card MiB of the split-dependent tensors at split (gdn_keys, moe_blocks), each shard as the allocator holds it"""
    gk, mb = _split(*split)
    if not (0 < gk < st['keys'] and 0 < mb < st['blocks']): raise ValueError(f'capacity: split {gk} / {mb} outside {st["keys"]} / {st["blocks"]}')
    fr = dict(gdn=(gk / st['keys'], 1 - gk / st['keys']), moe=(mb / st['blocks'], 1 - mb / st['blocks']))
    rnd = lambda b: -(-b // SPLIT_ROUND) * SPLIT_ROUND if b >= 10 * MIB else b
    return tuple(sum(rnd(int(b * fr[k][c])) for k in ('gdn', 'moe') for b in st[k]) / MIB for c in (0, 1))


def fixed_mib(model, ntok, weight_bytes=None, split=None, tensors=None):
    """the fixed MiB per card for (model, ntok) at the TP split (gdn_keys, moe_blocks): the measured parts (FIXED_MIB, at
    MEASURED_SPLIT); ntok between the measured 1 and 4 interpolated linearly, above 4 extrapolated by the larger of the
    measured slope and EXTRA_TOKEN_MIB per token. Another split moves the split-dependent tensors' bytes between the cards
    (tensors: split_tensors of the model's GGUF; split_weight_mib) + SPLIT_MARGIN_MIB per card. None when unknown: no
    entry, weight_bytes given and not the measured model's MODEL_BYTES, or another split without tensors"""
    a, b = FIXED_MIB.get((model, 1)), FIXED_MIB.get((model, 4))
    if a is None or b is None or None in a or None in b: return None
    if weight_bytes is not None and MODEL_BYTES.get(model) != int(weight_bytes): return None
    if isinstance(ntok, bool) or not isinstance(ntok, int) or ntok < 1: raise ValueError(f'fixed_mib: ntok {ntok!r}')
    if ntok <= 4: fx = tuple(a[c] + (b[c] - a[c]) * (ntok - 1) / 3 for c in (0, 1))
    else: fx = tuple(b[c] + max((b[c] - a[c]) / 3, EXTRA_TOKEN_MIB) * (ntok - 4) for c in (0, 1))
    if split is None or tuple(int(x) for x in split) == MEASURED_SPLIT: return fx
    if tensors is None: return None
    w, w0 = split_weight_mib(tensors, split), split_weight_mib(tensors, MEASURED_SPLIT)
    return tuple(fx[c] + w[c] - w0[c] + SPLIT_MARGIN_MIB for c in (0, 1))


def budget(g, max_ctx, ntok=4, model='coder', drafters=(), idx=True, dflash_geom=None, qsa=None, prefill=None, split=None, cache=None,
           feed=None):
    """the projected VRAM per card (MiB) for one configuration: dict(rows=[(item, xtx, xt)], total=(xtx, xt)); the fixed
    part None (unmeasured model) counts 0 and sets 'fixed_known' False. prefill: the batched prefill's chunk (its
    workspace, feed, glue workspace and modules, prefill_need_mib), None = not counted; split: (gdn_keys, moe_blocks),
    the engine's TP split, required with prefill; the prefill's plan (caminus.prefill.plan.PrefillPlan) at this ntok, with the
    feed's shapes feed (caminus.prefill.feed.feed_shape of the GGUF; None: the Coder's, plan.CODER_FEED). qsa None: as FlashNextTP's auto mode (the Coder with the indexer caches
    past QSA_EXACT), whose QSA tapes add tp_qsa_bytes (counted by FlashNextTP._check_vram too).
    The total is this process's own need: the device-level use when it runs adds what the card already held before it
    started (the desktop, the driver, other processes; XT ~270 MiB, XTX ~110 MiB on 11-03 / 11-04), which the runtime
    checks read live (check: used now + need) and the k3_capacity table subtracts as "in use now"."""
    rows = []
    fx = fixed_mib(model, ntok)
    rows.append((f'engine fixed ({model}, T={ntok})', *(fx if fx else (0, 0))))
    cb = tp_ctx_bytes(g, max_ctx, idx)
    if qsa is None: qsa = model == 'coder' and idx and max_ctx > QSA_EXACT
    if qsa: cb.update(tp_qsa_bytes(g, max_ctx, ntok))
    for k in cb:
        rows.append((f'{k} @ {max_ctx}', cb[k] / MIB, cb[k] / MIB))
    for d in drafters:
        f = DRAFTER_FIXED_MIB[d]; rows.append((f'{d} fixed', *f))
        b = drafter_ctx_bytes(d, g, max_ctx, ntok, dflash_geom)
        tag = f' (ring {dflash_ring(_dflash_window(dflash_geom), ntok)} rows)' if d == 'dflash' else ''
        rows.append((f'{d} ctx buffers @ {max_ctx}{tag}', b[0] / MIB, b[1] / MIB))
    if prefill:
        if split is None: raise ValueError('capacity.budget: prefill needs split=(gdn_keys, moe_blocks), the engine\'s TP split')
        from ..prefill.plan import PrefillPlan
        gk, mb = _split(*split)
        nf = DFLASH_FEATURE_LAYERS if 'dflash' in drafters else 0
        pl = PrefillPlan.of(g, prefill, gdn_keys=gk, moe_blocks=mb, T_pass=ntok, n_feat=nf, qsa_ctx=max_ctx if qsa else 0, feed=feed)
        w, f = pl.workspace_bytes(), pl.feed_bytes()
        rows.append((f'prefill GEMM workspace (chunk {prefill}, split {gk} / {mb})', w[0] / MIB, w[1] / MIB))
        rows.append((f'prefill feed (chunk {prefill}; token_embd on card {pl.table_card})', f[0] / MIB, f[1] / MIB))
        gl = pl.glue_bytes()
        rows.append((f'prefill glue workspace (chunk {prefill})', gl[0] / MIB, gl[1] / MIB))
        xb = pl.xchg_bytes()
        rows.append((f'prefill exchange receive buffer (chunk {prefill})', xb[0] / MIB, xb[1] / MIB))
        ab = pl.attn_bytes()
        rows.append((f'prefill attention workspace (chunk {prefill}' + (f'; DFlash features x {nf} on card {FEATURE_CARD}' if nf else '')
                     + (f'; QSA prompt path at {max_ctx}' if qsa else '') + ')', ab[0] / MIB, ab[1] / MIB))
        rows.append(('prefill modules (code objects, HIP module state)', *PREFILL_CODE_MIB))
        rows.append(('prefill streams (a compute and a copy stream per card)', *PREFILL_STREAMS_MIB))
    if cache:   # (the prompt cache's VRAM tier: cache = dict(slot_bytes, n_slots, side_bytes, extra_slots), promptcache.PromptCache.tier)
        rows.append((f'prompt cache VRAM tier ({cache["n_slots"]} + {cache.get("extra_slots", 1)} slots)',
                     *cache_tier_mib(cache['slot_bytes'], cache['n_slots'], cache.get('side_bytes', (0, 0)), cache.get('extra_slots', 1))))
    tot = tuple(sum(r[1 + c] for r in rows) for c in (0, 1))
    return dict(rows=rows, total=tot, fixed_known=fx is not None)


def caps_mib(devs):
    """per device: (cap MiB = its guard line (guard_line_mib) - MARGIN_MIB, used MiB now device-wide as the guard reads it
    (used_mib: all processes))"""
    return [(guard_line_mib(d) - MARGIN_MIB, used_mib(d)) for d in devs]


def guard_line_mib(d):
    """tools/guard.sh's kill line of torch device d (GUARD_LINE_MIB by its DRM card, caminus.device.machine: here the XTX 23,800, the XT
    19,850 MiB), or CAP_FRAC of its total for a card the guard does not watch (never above CAP_FRAC x total)"""
    import os
    c = drm_card(d); total = torch.cuda.get_device_properties(d).total_memory / MIB
    line = GUARD_LINE_MIB.get(os.path.basename(os.path.dirname(c))) if c is not None else None
    return min(float(line), CAP_FRAC * total) if line is not None else CAP_FRAC * total


def used_mib(d):
    """the device-level used VRAM of torch device d as the guard reads it (sysfs mem_info_vram_used, every process): the one
    reading every capacity check uses (device_used_mib's first value; hipMemGetInfo, 163 / 353 MiB lower with the Coder
    loaded, only where sysfs cannot be read)"""
    s, h = device_used_mib([d])[0]
    return s if s is not None else h


_DRM = {}


def drm_card(d):
    """the /sys/class/drm/cardN/device directory of torch device d (by its PCI address when torch gives one, else by its VRAM
    size among the cards; None when not found)"""
    if d in _DRM: return _DRM[d]
    import glob, os
    p = torch.cuda.get_device_properties(d)
    want = None
    if all(hasattr(p, a) for a in ('pci_domain_id', 'pci_bus_id', 'pci_device_id')):
        want = f'{int(p.pci_domain_id):04x}:{int(p.pci_bus_id):02x}:{int(p.pci_device_id):02x}.'
    found = None
    for c in sorted(glob.glob('/sys/class/drm/card[0-9]*/device')):
        try:
            if want is not None:
                if os.path.basename(os.path.realpath(c)).startswith(want): found = c; break
            elif abs(int(open(os.path.join(c, 'mem_info_vram_total')).read()) - int(p.total_memory)) < (64 << 20): found = c; break
        except OSError: continue
    _DRM[d] = found
    return found


def device_used_mib(devs):
    """per device: the device-level used VRAM as the guard reads it (sysfs mem_info_vram_used, every process) and as HIP reports
    it (hipMemGetInfo), (sysfs or None, hip). Measured 11-04 with the Coder at 33,024 + DFlash + prefill loaded: sysfs 20,205 /
    15,513 MiB vs hipMemGetInfo 20,042 / 15,160 (XTX / XT): HIP reports 163 / 353 MiB less than the guard sees, so a budget
    that must keep the guard's line (the prompt cache's tier, sized to the last MiB) reads the larger of the two"""
    out = []
    for d in devs:
        free, total = torch.cuda.mem_get_info(d)
        c = drm_card(d); s = None
        if c is not None:
            try: s = int(open(c + '/mem_info_vram_used').read()) / MIB
            except OSError: s = None
        out.append((s, (total - free) / MIB))
    return out


def release_cache(devs):
    """return the caching allocator's free blocks to the driver (torch.cuda.empty_cache: every device's): torch keeps the
    blocks of freed tensors reserved (a dropped snapshot, a constructor's staging buffers), and the device-level use
    counts them while no projection does. Measured 11-04 (M6 unit 1, XT, max_ctx 33024 + DFlash + prefill): the reserved-
    but-free cache 62.5 MiB after the load, 126.5 after test_prefill_s6 g3q (its three snapshots' GDN blocks in 20 MiB
    segments, 72.5 MiB; the harness's IDXRAW hashing copies, 34 MiB), which put the XT 11-21 MiB past idle + projection.
    The defined points: the end of FlashNextTP / DFlashK3 / Prefill construction (their staging buffers) and
    longctx.release (a snapshot dropped). The caches are never counted in a projection instead: a live reading there would
    make the check idle + projection >= measured hold by construction. Before anything is released, every address a live
    K3 tape holds (k3.tape_pointers) is checked against the allocator's blocks: one inside a FREE block (a tape built on
    a tensor that was dropped since) raises RuntimeError and nothing is released (fail closed: after the release the tape
    would page-fault instead of reading stale bytes). Returns per device in devs the MiB released (torch's reserved
    before - after)"""
    devs = [int(d) for d in devs]
    for d in devs: torch.cuda.synchronize(d)
    stale = stale_tape_pointers()
    if stale:
        raise RuntimeError(f'capacity.release_cache: {len(stale)} K3 tape addresses lie in blocks torch has freed (first: '
                           f'{stale[:3]}): nothing released')
    before = [torch.cuda.memory_reserved(d) for d in devs]
    torch.cuda.empty_cache()
    out = [(b - torch.cuda.memory_reserved(d)) / MIB for b, d in zip(before, devs)]
    LAST_RELEASE.update(released_mib=out)
    return out


LAST_RELEASE = {}   # the last release_cache's check and result: engines and tape addresses scanned, free blocks, MiB released


def stale_tape_pointers():
    """the K3 tape addresses (k3.tape_pointers) inside a block torch's caching allocator holds FREE: [(device, address,
    where)] (empty: every tape address is in a live block or outside torch's pool, e.g. pinned host memory, which
    empty_cache never touches). LAST_RELEASE records how many engines / addresses / free blocks were checked"""
    from ..engine import k3
    ptrs = k3.tape_pointers()
    LAST_RELEASE.clear(); LAST_RELEASE.update(engines=k3.LAST_SCAN.get('engines', 0), addresses=len(ptrs))
    if not ptrs: return []
    import bisect
    free = {}
    for sg in torch.cuda.memory_snapshot():
        a = int(sg['address'])
        for b in sg['blocks']:
            a = int(b.get('address', a))
            if b['state'] == 'inactive': free.setdefault(int(sg['device']), []).append((a, a + int(b['size'])))
            a += int(b['size'])
    # (one sorted list over every device: the cards share one virtual address space, and a tape may hold a peer card's
    # address, OP_XCHG's pull slot)
    iv = sorted(x for v in free.values() for x in v); starts = [lo for lo, _ in iv]
    LAST_RELEASE.update(free_blocks=len(iv))
    out = []
    for d, p, w in ptrs:
        i = bisect.bisect_right(starts, p) - 1
        if i >= 0 and p < iv[i][1]: out.append((d, p, w))
    return out


def check(devs, need_mib, what):
    """refuse (CapacityError) when need_mib[c] more MiB would push device devs[c] past its cap (a device named twice, e.g.
    a one-card test build: its needs summed); returns the per-device (cap, used now, need) rows in first-seen order"""
    assert len(devs) == len(need_mib), (devs, need_mib)
    tot = {}
    for d, need in zip(devs, need_mib): tot[d] = tot.get(d, 0.0) + float(need)
    rows = []
    for d, (cap, used) in zip(tot, caps_mib(list(tot))):
        need = tot[d]; rows.append((cap, used, need))
        if used + need > cap:
            raise CapacityError(f'{what}: device {d} would use {used:.0f} + {need:.0f} = {used + need:.0f} MiB > its cap '
                                f'{cap:.0f} MiB (the guard\'s line - {MARGIN_MIB} MiB margin; use as the guard reads it); lower max_ctx or drop a drafter')
    return rows
