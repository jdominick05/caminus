"""K3 decode engine helpers (kernels/k3_*.hip)."""
import ctypes, os
import numpy as np
import torch
from ..device.hip import Module

_mod = {}


def _m(name):
    dev = torch.cuda.current_device()
    if (dev, name) not in _mod: _mod[(dev, name)] = Module(name)
    return _mod[(dev, name)]


def repack_mv(q):
    """int4 weight codes [N, K] (int, -8..7) -> the decode matvec layout [N, K / 2] uint8: per 8 consecutive k, byte i
    holds u[k+i] in the low nibble and u[k+4+i] in the high nibble, u = code + 8"""
    N, K = q.shape
    u = (q.to(torch.int16) + 8).reshape(N, K // 8, 2, 4)        # [.., half (k+0..3 | k+4..7), i]
    return (u[:, :, 0, :] | (u[:, :, 1, :] << 4)).to(torch.uint8).reshape(N, K // 2).contiguous()


def unpack_codes(c):
    """packed int4 codes (caminus.device.gemm.pack_int4 / actq layout) [.., K/2] -> int8 [.., K]"""
    lo = (c & 0xF).to(torch.int16); hi = (c >> 4).to(torch.int16)
    q = torch.stack([lo, hi], -1).reshape(*c.shape[:-1], -1)
    return torch.where(q > 7, q - 16, q).to(torch.int8)


def mv_g128(w_mv, gs, a, rs, out_dtype=torch.bfloat16):
    """y [N] = the decode matvec (kernels/k3_mv.hip): w_mv from repack_mv, gs [N, K / 128] fp32, a int8 codes [K], rs
    the row scale (python float)"""
    N, Kh = w_mv.shape; K = 2 * Kh; G = K // 128
    asum = a.to(torch.int32).reshape(G, 128).sum(-1).to(torch.int32).contiguous()
    y = torch.empty(N, device=w_mv.device, dtype=out_dtype)
    _m('k3_mv').k3_mv_g128(((N + 7) // 8,), (256,), w_mv, gs, a.contiguous(), asum, ctypes.c_float(rs), y,
                          ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(int(out_dtype == torch.float32)))
    return y


# ---- the engine (kernels/k3.hip) ----
OP_NOP, OP_MATVEC_G128, OP_GG_Q8_1, OP_GG_MV, OP_GG_MOE_GU, OP_GG_MOE_DOWN = 0, 1, 2, 3, 4, 5
OP_F32_MV, OP_ROUTER, OP_SHEXP_ADD, OP_RMS_MUL, OP_HC_MEAN, OP_HC_POST = 6, 7, 8, 9, 10, 11
OP_GDN_CONV, OP_GDN_L2, OP_GDN_GATES, OP_GDN_STEP, OP_GDN_NORM = 12, 13, 14, 15, 16
OP_QK_NR, OP_KV_APPEND, OP_ATTN_DEC, OP_GATE_MUL = 17, 18, 19, 20
OP_PLE_GATE, OP_PLE_CONV, OP_ADD_SUM, OP_HADAMARD = 21, 22, 23, 24
OP_HC_POST_RMS, OP_XCHG, OP_GDN_SAVE, OP_HC_AVG, OP_GG_GU_DENSE = 25, 26, 27, 28, 29
OP_BF16_MV = 30   # the k3_gguf_gsq modules only (docs/specs/spec-gsq-port.md 4c, S2): llama.cpp's mul_mat_vec_f order
# the DFlash 2 drafter's module only (kernels/k3_gguf_draft.hip, kernels/k3_draft_ops.h; python/caminus/decode/dflash.py)
OP_DF_CONV, OP_DF_TOPK, OP_DF_SELECT, OP_DF_EMBED, OP_DF_MVCONV, OP_DF_ACCEPT = 31, 32, 33, 34, 35, 36
DF_OPS = {OP_DF_CONV: 'CONV', OP_DF_TOPK: 'TOPK', OP_DF_SELECT: 'SELECT', OP_DF_EMBED: 'EMBED', OP_DF_MVCONV: 'MVCONV',
          OP_DF_ACCEPT: 'ACCEPT'}
# the QSA indexer ops (kernels/k3_qsa_ops.h; docs/specs/spec-qsa.md S2-S4): compiled only into the GSQ-RCO Coder's QSA
# modules k3_gguf_gsq_qsa (T = 1) / k3_gguf_gsq_qsa_mt (T > 1) = the trunk modules plus these ops (inlined into the trunk's own
# k3_gguf_gsq* they cost its tapes ~0.7 ms per token, so those stay byte for byte the same). A tape built on a trunk module
# moves to its QSA module when a QSA op is added (Engine.idx_append / qsa_score / qsa_select); any other module would run
# them as nops, so Engine.build refuses them there. env K3_QSA_ENTRY=1: every k3_gguf_gsq{,_mt} tape on the QSA module
# (measures that module's cost on the trunk's tapes)
OP_IDX_APPEND, OP_QSA_SCORE, OP_QSA_SELECT, OP_ATTN_SEL = 37, 38, 39, 40
QSA_OPS = {OP_IDX_APPEND: 'IDX_APPEND', OP_QSA_SCORE: 'SCORE', OP_QSA_SELECT: 'SELECT', OP_ATTN_SEL: 'ATTN_SEL'}
QSA_MODULES = ('k3_gguf_gsq_qsa', 'k3_gguf_gsq_qsa_mt')
QSA_OF = {'k3_gguf_gsq': 'k3_gguf_gsq_qsa', 'k3_gguf_gsq_mt': 'k3_gguf_gsq_qsa_mt'}
QSA_D, QSA_H, QSA_ROWB, QSA_TOPK, QSA_SELW = 128, 4, 136, 512, 516   # = kernels/k3_qsa_ops.h (row: 4 block_q8_0)
# split attention (flash decoding, kernels/k3_attn_split.h): the GSQ-RCO Coder's trunk and QSA modules run their attention
# only as OP_ATTN_PART (partial softmax states of ATTN_SS-cell slices of each row's list, tasks (split, row, kv head)) then
# OP_ATTN_COMB (their merge per query head); Engine.attn_dec / attn_sel emit that pair there (q8_0 or f16 cache, head dim
# 256, 12 query heads per kv head only) and build() refuses OP_ATTN_DEC / OP_ATTN_SEL on these modules (no device code: a nop)
OP_ATTN_PART, OP_ATTN_COMB = 41, 42
ATTN_SPLIT_OPS = {OP_ATTN_PART: 'PART', OP_ATTN_COMB: 'COMB'}
ATTN_SS, ATTN_G, ATTN_D, ATTN_RS = 32, 12, 256, 260   # = kernels/k3_attn_split.h (cells per split, ..., record floats)
# Qwen3-Coder-Next (qwen3next, python/caminus/engine/qwen3next_tp.py): the Coder's engine compiled with K3_Q3N (kernels/k3_gguf_q3n.hip):
# grouped GDN heads (key head = h / (H / nk), qwen3next's repeat-interleave), the SiLU GDN output gate, and split attention
# with 8 query heads per kv head (16 / 2); its own format whitelist. The Coder's modules compile the same source as before
Q3N_MODULES = ('k3_gguf_q3n', 'k3_gguf_q3n_mt')
# Qwen3.6-35B-A3B (qwen35moe) on ONE card (python/caminus/engine/qwen35_one.py), HauhauCS Q3_K_P formats (kernels/k3_gguf_q35h.hip):
# K3_GDN_SILU, tiled GDN heads (the default), split attention with 8 query heads per kv head
Q35H_MODULES = ('k3_gguf_q35h', 'k3_gguf_q35h_mt', 'k3_gguf_q35h_trace')
Q35D_MODULES = ('k3_gguf_q35d', 'k3_gguf_q35d_mt')   # the dense qwen35 line (AREX-2): one card; _mt: T > 1 (verify passes)   # (_trace: K3_TRACE timestamps, measurement only)
Q35G_MODULES = ('k3_gguf_q35g', 'k3_gguf_q35g_mt')   # the 2-bit GSQ checkpoint (kernels/k3_gguf_q35g.hip)
ATTN_SPLIT_MODULES = ('k3_gguf_gsq', 'k3_gguf_gsq_trace', 'k3_gguf_gsq_mt', 'k3_gguf_gsq_mt_trace') + QSA_MODULES + Q3N_MODULES + Q35H_MODULES + Q35G_MODULES + Q35D_MODULES
ATTN_G_OF = {m: (8 if m in Q3N_MODULES + Q35H_MODULES + Q35G_MODULES else 6 if m in Q35D_MODULES else ATTN_G) for m in ATTN_SPLIT_MODULES}   # query heads per kv head (= the module's ATTN_G)
# cells per split (= the module's ATTN_SS): the Coder's modules 32; the qwen3next / qwen35moe modules set their own (kernels/k3_gguf_q3n.hip ...)
ATTN_SS_OF = {m: ATTN_SS for m in ATTN_SPLIT_MODULES}
# GGUF block formats (Seg.work of the GG ops), = the enum in kernels/k3_gguf_ops.h. 4-9 (the GSQ-RCO Coder's) are
# instantiated only in the k3_gguf_gsq modules, per op (the REAP-308 module's dispatch would decode them as Q8_0): each
# module's capability set below lists its pairs, and Engine.build / run refuse a tape outside it
GG_FMT = {'IQ3_XXS': 0, 'IQ4_NL': 1, 'Q6_K': 2, 'Q8_0': 3, 'Q4_K': 4, 'Q5_K': 5, 'IQ4_XS': 6, 'IQ2_S': 7, 'IQ3_S': 8, 'Q2_0': 9, 'Q3_K': 10}   # Q3_K: the qwen35moe HauhauCS modules only
GG_FMT_NAME = {v: k for k, v in GG_FMT.items()}
Q8_1_MODE = {0: 'PLAIN', 1: 'SILU', 2: 'SWIGLU'}   # OP_GG_Q8_1 Seg.work: x, silu(x * rs), silu(x) * Seg.gs[i] (SWIGLU: the
                                                  # k3_gguf_gsq modules and the REAP-308 test build k3_gguf_swiglu only)

# What each code object implements, as (op, format) pairs: the formatted ops (GG matvec / MoE / dense gate-up by GGUF
# format, F32 / BF16 matvecs, the q8_1 glue modes). A tape with a pair outside its module's set fails at build (or, with
# build(strict=False), at run), never silently on the device (unknown formats fall to Q8_0 there, unknown ops to a nop).
_REAP_FMTS = ('IQ3_XXS', 'IQ4_NL', 'Q6_K', 'Q8_0')
_Q8_1 = {('Q8_1', m) for m in Q8_1_MODE.values()}
# REAP-308's modules have no SWIGLU glue (K3_Q8_SWIGLU: its never-taken branch cost their k3_run ~0.04-0.07 ms per token);
# the test build k3_gguf_swiglu (K3_TEST_BUILDS=1 ./build.sh) has it, for the forced-unfused shared expert on REAP-308
CAPS_REAP = frozenset({(op, f) for op in ('MV', 'MOE_GU', 'MOE_DOWN', 'GU_DENSE') for f in _REAP_FMTS} | {('F32_MV', 'F32')} |
                      (_Q8_1 - {('Q8_1', 'SWIGLU')}))
# the GSQ-RCO Coder module's per-op whitelist (spec-gsq-port.md 4b, from tools/gg_inventory.py = kernels/k3_gguf_gsq.hip's
# K3_FMTS_*): 12 format instances + the BF16 matvec; no GU_DENSE (the shared expert runs as two MVs + the SWIGLU glue)
CAPS_GSQ = frozenset({('MV', f) for f in ('Q4_K', 'Q5_K', 'Q6_K', 'IQ4_XS', 'IQ4_NL', 'Q8_0')} |
                     {('MOE_GU', f) for f in ('IQ2_S', 'IQ3_XXS', 'IQ3_S', 'IQ4_XS')} | {('MOE_DOWN', f) for f in ('IQ4_NL', 'Q2_0')} |
                     {('BF16_MV', 'BF16')} | _Q8_1)
MODULE_CAPS = {m: CAPS_REAP for m in ('k3_gguf', 'k3_gguf_trace', 'k3_gguf_mt', 'k3_gguf_mt_trace')}
MODULE_CAPS.update({m: CAPS_REAP | {('Q8_1', 'SWIGLU')} for m in ('k3_gguf_swiglu', 'k3_gguf_mt_swiglu')})
MODULE_CAPS.update({m: CAPS_GSQ for m in ('k3_gguf_gsq', 'k3_gguf_gsq_trace', 'k3_gguf_gsq_mt', 'k3_gguf_gsq_mt_trace')})
# the MTP block's module pair (kernels/k3_gguf_gsq_mtp*.hip, python/caminus/decode/mtp.py): the Coder's formats, the F32 router of the
# sidecar (op_f32_mv is compiled into every GGUF module) and OP_BF16_MV's fp32 swiglu stage (work 2, bf16_mv(swiglu_mul=))
MTP_MODULES = ('k3_gguf_gsq_mtp', 'k3_gguf_gsq_mtp_mt')
MODULE_CAPS.update({m: CAPS_GSQ | {('F32_MV', 'F32')} for m in MTP_MODULES})
# the DFlash 2 drafter's module (kernels/k3_gguf_draft.hip): MV Q8_0 (its layers, selector gate) and Q6_K (the target's head
# shards), the Q8_0 gate/up pair as GU_DENSE, the q8_1 glue incl. mode 3 (PERM: the inject's [L][T][D] features read as
# [T][L * D]) and the drafter ops; no MoE. The flagged ATTN_DEC / KV_APPEND (window, non-causal, ring) are this module's too:
# other modules would run them unflagged, so dflash_k3 builds its tapes only with this module
Q8_1_MODE[3] = 'PERM'
DRAFT_MODULES = ('k3_gguf_draft', 'k3_gguf_draft_trace')
# the ops the drafter's module compiles in (kernels/k3_gguf_ops.h #ifndef K3_DRAFT): any other op would run as a nop there;
# Engine.build refuses it. RMS_MUL / QK_NR / KV_APPEND / ATTN_DEC only in their flagged forms (Seg.aux / pad2 != 0)
DRAFT_OPS = {OP_NOP, OP_XCHG, OP_GG_Q8_1, OP_GG_MV, OP_RMS_MUL, OP_QK_NR, OP_KV_APPEND, OP_ATTN_DEC} | set(DF_OPS)
MODULE_CAPS.update({m: frozenset({('MV', 'Q6_K'), ('MV', 'Q8_0'), ('Q8_1', 'PERM')} | _Q8_1 |
                                 {('DF', n) for n in DF_OPS.values()}) for m in DRAFT_MODULES})
MODULE_CAPS.update({m: CAPS_GSQ | {('QSA', n) for n in QSA_OPS.values()} for m in QSA_MODULES})
# qwen3next (kernels/k3_gguf_q3n.hip's K3_FMTS_*): Q6_K / IQ3_S matvecs, IQ2_S gate / up, IQ3_S down, the F32 router and
# shared-expert gate, the q8_1 glue with SWIGLU (the shared expert as two MVs)
CAPS_Q3N = frozenset({('MV', f) for f in ('Q6_K', 'IQ3_S')} | {('MOE_GU', 'IQ2_S')} | {('MOE_DOWN', 'IQ3_S')} | {('F32_MV', 'F32')} | _Q8_1)
MODULE_CAPS.update({m: CAPS_Q3N for m in Q3N_MODULES})
CAPS_Q35H = frozenset({('MV', f) for f in ('Q3_K', 'Q4_K', 'Q5_K', 'Q6_K')} | {('MOE_GU', f) for f in ('Q3_K', 'Q5_K')} |
                      {('MOE_DOWN', f) for f in ('Q4_K', 'Q5_K')} | {('F32_MV', 'F32')} | _Q8_1)
MODULE_CAPS.update({m: CAPS_Q35H for m in Q35H_MODULES})
CAPS_Q35G = frozenset({('MV', 'Q8_0'), ('MOE_GU', 'Q2_0'), ('MOE_DOWN', 'Q2_0'), ('F32_MV', 'F32'), ('BF16_MV', 'BF16')} | _Q8_1)
MODULE_CAPS.update({m: CAPS_Q35G for m in Q35G_MODULES})
CAPS_Q35D = frozenset({('MV', f) for f in ('Q4_K', 'Q5_K', 'Q6_K', 'Q8_0')} | {('F32_MV', 'F32')} | _Q8_1)   # (AREX-2 Q4_K_M)
MODULE_CAPS.update({m: CAPS_Q35D for m in Q35D_MODULES})
for _m_ in ATTN_SPLIT_MODULES: MODULE_CAPS[_m_] = MODULE_CAPS[_m_] | {('ATTN', n) for n in ATTN_SPLIT_OPS.values()}
# whitelisted pairs whose device code is not built yet: tapes using them build, runs refuse (an unknown op is a nop on the
# device). Empty since S2 (OP_BF16_MV)
PENDING = {}
_OP_KIND = {OP_GG_MV: 'MV', OP_GG_MOE_GU: 'MOE_GU', OP_GG_MOE_DOWN: 'MOE_DOWN', OP_GG_GU_DENSE: 'GU_DENSE'}


def seg_kind(sg):
    """(op, format) of a formatted segment, else None"""
    if sg.op in _OP_KIND: return (_OP_KIND[sg.op], GG_FMT_NAME.get(sg.work, f'?{sg.work}'))
    if sg.op == OP_F32_MV: return ('F32_MV', 'F32')
    if sg.op == OP_BF16_MV: return ('BF16_MV', 'BF16')
    if sg.op == OP_GG_Q8_1: return ('Q8_1', Q8_1_MODE.get(sg.work, f'?{sg.work}'))
    if sg.op in DF_OPS: return ('DF', DF_OPS[sg.op])
    if sg.op in QSA_OPS: return ('QSA', QSA_OPS[sg.op])
    if sg.op in ATTN_SPLIT_OPS: return ('ATTN', ATTN_SPLIT_OPS[sg.op])
    return None


class Seg(ctypes.Structure):
    _fields_ = [('op', ctypes.c_int), ('ntask', ctypes.c_int), ('dep', ctypes.c_int), ('rows_per_task', ctypes.c_int),
                ('W', ctypes.c_void_p), ('gs', ctypes.c_void_p), ('a', ctypes.c_void_p), ('asum', ctypes.c_void_p),
                ('y', ctypes.c_void_p), ('rs', ctypes.c_float), ('N', ctypes.c_int), ('K', ctypes.c_int),
                ('out_f32', ctypes.c_int), ('work', ctypes.c_int), ('team', ctypes.c_int), ('pad1', ctypes.c_int),
                ('pad2', ctypes.c_int), ('dep2p1', ctypes.c_int),   # dep2p1: a second dependency + 1 (0: none; Engine.also)
                ('aux', ctypes.c_void_p)]                           # one more pointer for fused ops


MAXSEG = 4096                       # = K3_MAXSEG in kernels/k3.hip
# the sync block (kernels/k3.hip struct Sync): claim [MAXSEG], done [MAXSEG], then the status words at 2 * MAXSEG: exit, err,
# err_seg, err_seen (a wait gave up), then bnd, bnd_what, bnd_val, bnd_cap (written by the bound guard only,
# k3_run's prologue or kernels/k3_guard.hip: a pass position or lazy commit out of range, refused on the device)
SYNC_WORDS = MAXSEG * 2 + 8
STATUS_WORDS = 8


class ContextBoundsError(IndexError):
    """a position (or the lazy-commit count) outside what the engine's buffers hold: refused on the host before any
    launch, or (defense in depth) caught on the device by the bound guard: k3_run's prologue (kernels/k3.hip) skips the
    whole tape, nothing is written and the launch ends at once; the separate guard launch (kernels/k3_guard.hip: k3_run2
    tapes, or tapes with more than one position tensor) clamps the values to 0 instead, so that launch still runs its
    tasks once (every access inside its buffer, its waits giving up at once) and leaves the KV rows 0 .. T - 1 and the
    states garbage. Never retried; the engines refuse passes until reset()"""


def check_span(i0, n, max_ctx, commit=None, ntok=None, what='pass'):
    """host check before a launch: positions i0 .. i0 + n - 1 must lie in [0, max_ctx) (the KV rows a pass writes and
    attends over), and the lazy commit (when given) in [0, ntok] (it indexes the previous pass's ntok inputs / records)"""
    if isinstance(i0, bool) or not isinstance(i0, (int, np.integer)):
        raise ContextBoundsError(f'k3 {what}: position {i0!r} is not an int')
    i0, n = int(i0), int(n)
    if i0 < 0 or n < 1 or i0 + n > max_ctx:
        raise ContextBoundsError(f'k3 {what}: positions {i0} .. {i0 + n - 1} outside the KV cache [0, {max_ctx}) '
                                 f'(max_ctx {max_ctx}; build the engine with a larger max_ctx)')
    if commit is not None:
        if isinstance(commit, bool) or not isinstance(commit, (int, np.integer)) or not 0 <= int(commit) <= ntok:
            raise ContextBoundsError(f'k3 {what}: commit {commit!r} outside [0, {ntok}] (tokens of the previous pass)')


def bounds_status(st):
    """the bound guard's report from the status words (st = sync[2 * MAXSEG:2 * MAXSEG + 8]): None, or a message"""
    if len(st) < 8 or not st[4]: return None
    how = ('k3_run skipped its whole tape (nothing written)' if st[4] == 2 else
           'clamped to 0 on the device (no access past the buffers; the KV rows 0 .. T - 1 and the states are garbage)')
    if st[5] == -1:
        return f'pass position {st[6]} (+ T tokens) outside the KV cache rows [0, {st[7]}): {how}; refused until reset()'
    if st[5] == -2:
        return f'lazy commit {st[6]} outside [0, {st[7]}]: {how}; refused until reset()'
    return f'bound guard report {st[4:8]}'


def resident_grid(module, per_wgp=None):
    """workgroups that are all resident at once on the current device: k3_run in the GGUF builds is capped at 96 VGPRs
    (16 waves per SIMD, K3_OPAQUE_TID) with ~13 KB of LDS, so 8 fit per WGP (a -DK3_NO_OPAQUE_TID build is capped at
    144: 5 per WGP); extra workgroups only start after the first ones exit and then walk the whole tape (K3_PER_WGP
    overrides)"""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    per_wgp = per_wgp or int(os.environ.get('K3_PER_WGP', 0)) or (8 if module.startswith('k3_gguf') else 4)
    return per_wgp * props.multi_processor_count


def module_for(module, T):
    """the code object for T tokens per pass: k3_gguf{,_trace} -> k3_gguf_mt{,_trace} for T > 1 (kernels/k3_gguf_mt.hip:
    the multi-token fast paths live only there, so the one-token k3_run keeps its code); env K3_MT=0 keeps the plain one.
    env K3_QSA_ENTRY=1: the GSQ trunk modules' QSA variants (QSA_OF).
    k3_gguf_gsq at T > 1 -> k3_gguf_gsq_qsa_mt (the trunk's multi-token QSA module: the same trunk ops, the same bits): with
    the split attention (kernels/k3_attn_split.h) the trunk's own k3_gguf_gsq_mt spills its SGPRs to scratch memory
    instead of VGPR lanes (~450 reloads, T=4 pass 16.4 -> 23 ms); the QSA module compiles the same source cleanly (T=4 pass
    at depth 0 ~+0.5 ms vs the old trunk module, the cost of its indexer ops). k3_gguf_gsq_mt stays built (tests name it)"""
    m = module
    if T > 1 and module in ('k3_gguf', 'k3_gguf_trace', 'k3_gguf_swiglu') and os.environ.get('K3_MT', '1') != '0':
        m = module.replace('k3_gguf', 'k3_gguf_mt', 1)
    elif T > 1 and module == 'k3_gguf_gsq' and os.environ.get('K3_MT', '1') != '0':
        m = 'k3_gguf_gsq_qsa_mt'
    elif T > 1 and module == 'k3_gguf_gsq_trace' and os.environ.get('K3_MT', '1') != '0':
        m = 'k3_gguf_gsq_mt_trace'
    elif T > 1 and module == 'k3_gguf_gsq_mtp': m = 'k3_gguf_gsq_mtp_mt'   # (always: the MTP pair has no K3_MT=0 variant)
    elif T > 1 and module == 'k3_gguf_gsq_qsa' and os.environ.get('K3_MT', '1') != '0': m = 'k3_gguf_gsq_qsa_mt'
    elif T > 1 and module == 'k3_gguf_q3n' and os.environ.get('K3_MT', '1') != '0': m = 'k3_gguf_q3n_mt'
    elif T > 1 and module == 'k3_gguf_q35h' and os.environ.get('K3_MT', '1') != '0': m = 'k3_gguf_q35h_mt'
    elif T > 1 and module == 'k3_gguf_q35g' and os.environ.get('K3_MT', '1') != '0': m = 'k3_gguf_q35g_mt'
    elif T > 1 and module == 'k3_gguf_q35d' and os.environ.get('K3_MT', '1') != '0': m = 'k3_gguf_q35d_mt'
    if os.environ.get('K3_QSA_ENTRY') == '1' and m in QSA_OF: m = QSA_OF[m]
    return m


def mmvf_warps(ncols, warp=32, max_block=256):
    """llama.cpp's mul_mat_vec_f block size for rows of ncols, in warps (launch_mul_mat_vec_f_cuda: from 32 threads, the
    first block size up to 256 in steps of 32 with strictly fewer iterations ceil(ncols / (2 * block)); RDNA3 allows 256):
    8 for K 2560 / 10240, 5 for K 320"""
    best, nbest = warp, (ncols + 2 * warp - 1) // (2 * warp)
    for bs in range(2 * warp, max_block + 1, warp):
        n = (ncols + 2 * bs - 1) // (2 * bs)
        if n < nbest: best, nbest = bs, n
    return best // warp


GG_ACT_Q8, GG_ACT_F32, GG_ACT_F32_SILU = 1, 2, 3   # activation source of the GG matvec / MoE ops (Seg.out_f32)
AQ_BLOCKS = 80


def gg_act(x, n, silu_scale=None):
    """(mode, rs) for a GG op's activation: q8_1 blocks (uint8, from gg_q8_1) or fp32, which each workgroup quantizes
    into LDS itself (optionally silu(x * silu_scale) first)"""
    if x.dtype == torch.uint8:
        assert silu_scale is None; return GG_ACT_Q8, 0.0
    assert x.dtype == torch.float32 and x.numel() == n and n % 32 == 0 and n <= 32 * AQ_BLOCKS, (x.dtype, x.numel(), n, 'fp32 activations (per-workgroup quantization) only with one token per pass')
    return (GG_ACT_F32, 0.0) if silu_scale is None else (GG_ACT_F32_SILU, float(silu_scale))


def kv_pos_bytes(M, q8):
    """bytes per position of a KV cache with M values each of K and V (kernels/k3_gguf_ops.h kv_pos_bytes)"""
    return 2 * (M // 32) * 34 if q8 else 4 * M


def kv_rows(cache, M, q8):
    """positions a KV cache holds, from its allocation (the device-side bound of OP_KV_APPEND / OP_ATTN_DEC)"""
    pb = kv_pos_bytes(M, q8); nb = cache.numel() * cache.element_size()
    if q8 and M % 32: raise ValueError(f'k3 KV cache: {M} values per position is not a whole number of q8_0 blocks')
    if nb % pb or not 0 < nb // pb < 2 ** 31: raise ValueError(f'k3 KV cache: {nb} bytes is not a whole number of {pb}-byte positions')
    return nb // pb


def gdn_block_size(H, S, nk, T):
    """floats in a GDN layer's block: the state, then T saved step-input records (k, v, gates) for the lazy commit"""
    return H * S * S + T * (nk * S + H * S + 2 * H)


SEG_PTRS = ('W', 'gs', 'a', 'asum', 'y', 'aux')   # Seg's pointer fields (aux also carries small flags: values < 2**32 skipped)


def tape_pointers():
    """(device, address, where) of every device address a live Engine's tape holds (its segments' pointer fields and its
    guard's position tensors), found through the garbage collector: the tapes keep raw addresses, not the tensors, so
    capacity.release_cache checks that none lies in a block torch's allocator has freed before it returns the free blocks
    to the driver (a tape reading a freed block would page-fault after the release, instead of reading stale bytes)"""
    import gc
    out = []; ne = 0
    for e in gc.get_objects():
        if not issubclass(type(e), Engine): continue   # (type(): isinstance would read __class__ of lazy module proxies)
        try: dev = e.sync.device.index
        except Exception: continue
        ne += 1
        draft = e.module in DRAFT_MODULES
        for i, sg in enumerate(e.segs):
            for f in SEG_PTRS:
                if f == 'aux' and draft and sg.op in (OP_RMS_MUL, OP_QK_NR): continue   # (the drafter's expression flags + ring mask)
                v = getattr(sg, f)
                if v and v >= 1 << 32: out.append((dev, int(v), f'{e.module} segment {i} op {sg.op} {f}'))
        for k in e.guard_caps: out.append((dev, int(k), f'{e.module} guard position'))
    LAST_SCAN.update(engines=ne, addresses=len(out))
    return out


LAST_SCAN = {}   # tape_pointers' last scan: live engines and addresses found


class Engine:
    """a tape of segments run by one persistent kernel launch; status checked after every run"""
    def __init__(self, grid=None, limit_us=20_000, module='k3', teams=None, T=1):
        """teams = None: k3_run; a list of up to 4 team sizes summing to the grid: k3_run2 (segments carry team -1 or an index)"""
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        self.grid = grid or 4 * props.multi_processor_count   # multiProcessorCount = WGPs on gfx11
        self.limit_us = limit_us   # a stuck wait gives up after this (keeps the XTX's desktop compositor alive)
        self.module = module_for(module, T); self.mv_u = 16 if module == 'k3_u16' else 8; self.teams = teams
        self.sync = torch.zeros(SYNC_WORDS, dtype=torch.int32, device='cuda')
        self.segs, self.keep = [], []
        self.T = T            # tokens per pass (speculative verification); every segment's team field carries it
        self.params = torch.zeros(16, dtype=torch.int32, device='cuda')   # per pass: [1] = previous pass's tokens to commit
        self.caps = MODULE_CAPS.get(self.module)   # None: a module without the formatted ops (no check)
        self.refuse = None    # why run() must not launch this tape (build(strict=False) of an unsupported tape)
        # the device bound guard of a tape with position-indexed ops: the position tensors with their KV caches' smallest
        # row count, and whether an op replays params[1] (the lazy commit). k3_run checks them in its prologue (one
        # position tensor: the kernel arguments gpos / gparams / gcap / gT, kernels/k3.hip); k3_run2 tapes and tapes with
        # more than one position tensor get the separate guard launch first (kernels/k3_guard.hip, _guard)
        self.guard_caps, self.guard_commit, self._glc = {}, False, None
        self.guard_launch = False   # True: the separate guard launch even ahead of k3_run (tests of its clamp path)
        self.team_fixed = set()     # segments whose team (token count) build() keeps: bf16_mv(ntok=) rows != T

    def _guard_pos(self, pos, rows):
        assert pos.dtype == torch.int32 and pos.numel() >= 1, (pos.dtype, pos.numel())
        k = pos.data_ptr(); self.guard_caps[k] = (pos, min(rows, self.guard_caps.get(k, (pos, rows))[1]))
        self._glc = None   # the cached guard launch carries the bounds: rebuilt on the next run

    def _guard_replay(self):
        """an op replays params[1] (the lazy commit) rows / records of the previous pass: the guard bounds it to [0, T]"""
        self.guard_commit = True; self._glc = None

    def _guard(self, stream):
        """the bound guard's launches (one per position tensor; the first also checks the commit) on `stream`, ahead of
        k3_run: same stream, so it reads the words the tape will read and k3_run starts after it (argument arrays cached)"""
        if not self.guard_caps and not self.guard_commit: return
        from ..device.hip import hip, check as hip_check
        key = (torch.cuda.current_device(), stream, self.params.data_ptr(), self.sync.data_ptr(), self.T, self.guard_commit,
               tuple((k, c) for k, (_, c) in self.guard_caps.items()))   # (the bounds too: a tape changed after a run)
        if self._glc is None or self._glc[0] != key:
            st = self.sync[2 * MAXSEG:]
            items = [(pos.data_ptr(), cap) for pos, cap in self.guard_caps.values()] or [(0, 0)]
            calls = []
            for j, (pp, cap) in enumerate(items):
                vals = [ctypes.c_void_p(pp), ctypes.c_int(cap), ctypes.c_int(self.T),
                        ctypes.c_void_p(self.params.data_ptr() if self.guard_commit and j == 0 else 0), ctypes.c_void_p(st.data_ptr())]
                calls.append((vals, (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])))
            self._glc = (key, _m('k3_guard').k3_guard.fn, calls, ctypes.c_void_p(stream))
        _, fn, calls, s = self._glc
        for vals, params in calls: hip_check(hip().hipModuleLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, s, params, None))

    def matvec(self, w_mv, gs, a, rs, y, rows_per_task=32, dep=-1, team=-1):
        N, Kh = w_mv.shape; K = 2 * Kh
        P = (K + 1023) // 1024
        assert K % 128 == 0 and K <= 8192 and rows_per_task <= self.mv_u * (8 // P) and rows_per_task * K // 128 <= 1024, \
            f'k3 matvec limits: K {K}, rows_per_task {rows_per_task}'
        self.keep += [w_mv, gs, a, y]
        self.segs.append(Seg(OP_MATVEC_G128, (N + rows_per_task - 1) // rows_per_task, dep, rows_per_task,
                             w_mv.data_ptr(), gs.data_ptr(), a.data_ptr(), None, y.data_ptr(), rs, N, K,
                             int(y.dtype == torch.float32), 0, team, 0, 0))
        return len(self.segs) - 1

    def gg_q8_1(self, x, yq, dep=-1, team=-1, silu_scale=None, mul=None):
        """GGUF op (module 'k3_gguf'): fp32 x [n] -> q8_1 blocks yq (uint8 [ceil(n / 32) * 36]); silu_scale: quantize
        silu(x * silu_scale) instead (ggml_scale then ggml_silu, the hyper-connection down output); mul (fp32 [n]):
        quantize silu(x) * mul (llama.cpp's unfused swiglu: the shared expert's gate x and up mul, mode SWIGLU)"""
        n = x.numel(); nb = (n + 31) // 32
        assert mul is None or (silu_scale is None and mul.dtype == torch.float32 and mul.numel() == n)
        self.keep += [x, yq] + ([] if mul is None else [mul])
        work = 2 if mul is not None else 0 if silu_scale is None else 1
        self.segs.append(Seg(OP_GG_Q8_1, (nb + 7) // 8, dep, 0, None, None if mul is None else mul.data_ptr(), x.data_ptr(), None,
                             yq.data_ptr(), 0.0 if silu_scale is None else silu_scale, n, 0, 0, work, team, 0, 0))
        return len(self.segs) - 1

    def gg_matvec(self, w, fmt, nrows, ncols, row_bytes, yq, out, rows_per_task=16, dep=-1, team=-1, silu_scale=None, ntok=None):
        """GGUF op (module 'k3_gguf'): raw blocks w of nrows contiguous rows (format name in GG_FMT) times the vector yq
        (q8_1 blocks, or fp32 quantized per workgroup: gg_act) -> fp32 out [nrows]; row_bytes = bytes per row (for the
        pre-wait prefetch). ntok = 1 (q8_1 activations only): one token's row in a T > 1 tape (the MTP catch-up's head on
        its last row: yq / out point at that row); the segment keeps that count through build()"""
        mode, rs = gg_act(yq, ncols, silu_scale)
        if ntok is not None:
            assert ntok == 1 and mode == GG_ACT_Q8 and team == -1 and self.teams is None, 'gg_matvec ntok: 1 token of q8_1 rows only'
            assert yq.numel() >= ncols // 32 * 36 and out.numel() >= nrows, (yq.numel(), out.numel())
        self.keep += [w, yq, out]
        self.segs.append(Seg(OP_GG_MV, (nrows + rows_per_task - 1) // rows_per_task, dep, rows_per_task, w.data_ptr(),
                             None, yq.data_ptr(), None, out.data_ptr(), rs, nrows, ncols, mode, GG_FMT[fmt], team if ntok is None else 1, 0,
                             row_bytes))
        if ntok is not None: self.team_fixed.add(len(self.segs) - 1)
        return len(self.segs) - 1

    def gg_moe_gu(self, gate, up, fmt, slab_bytes, ffn, d, ids, xq, h, rows_per_task=32, dep=-1, team=-1, h_q8=False, up_fmt=None):
        """GGUF op: h [k * ffn] = silu(gate_e . x) * (up_e . x) for the k experts e = ids (int32, on the device). One
        format decodes both gate and up: up_fmt (the up tensor's own format; None = the caller vouches) must equal fmt"""
        assert up_fmt is None or up_fmt == fmt, f'gg_moe_gu: gate {fmt} and up {up_fmt} differ (one format decodes both)'
        k = ids.numel(); self.keep += [gate, up, ids, xq, h]
        mode, _ = gg_act(xq, d)
        if h_q8: assert rows_per_task == 32 and h.dtype == torch.uint8   # h = q8_1 blocks of the k * ffn values
        self.segs.append(Seg(OP_GG_MOE_GU, (k * ffn + rows_per_task - 1) // rows_per_task, dep, rows_per_task,
                             gate.data_ptr(), up.data_ptr(), xq.data_ptr(), ids.data_ptr(), h.data_ptr(), 1.0 if h_q8 else 0.0, k * ffn, d,
                             mode, GG_FMT[fmt], team, ffn, slab_bytes))
        return len(self.segs) - 1

    def gg_moe_down(self, down, fmt, slab_bytes, ffn, d, ids, wts, hq, out, rows_per_task=16, dep=-1, team=-1):
        """GGUF op: out [d] = sum_j w_j * (down_e . h_j) for e = ids[j], in order (llama.cpp's mul then add chain)"""
        k = ids.numel() // self.T; self.keep += [down, ids, wts, hq, out]
        mode, rs = gg_act(hq, k * ffn)
        self.segs.append(Seg(OP_GG_MOE_DOWN, (self.T * d + rows_per_task - 1) // rows_per_task, dep, rows_per_task,
                             down.data_ptr(), wts.data_ptr(), hq.data_ptr(), ids.data_ptr(), out.data_ptr(), rs, self.T * d,
                             ffn, mode, GG_FMT[fmt], team, k, slab_bytes))
        return len(self.segs) - 1

    def bf16_mv(self, w, nrows, ncols, x, out, rows_per_task=8, dep=-1, team=-1, silu_scale=None, ntok=None, swiglu_mul=None):
        """BF16 weights w [nrows, ncols] (row-major, raw bytes) times fp32 x [T, ncols] -> fp32 out [T, nrows], bitwise
        llama.cpp's mul_mat_vec_f<nv_bfloat16> (kernels/k3_gguf_ops.h OP_BF16_MV; spec-gsq-port.md 4c). A task = rows_per_task
        weight rows for all T tokens. silu_scale: the activation is silu(silu_scale * x + 0) (the hc up: ignite's fused
        scale + silu), staged in LDS per workgroup: T * ncols <= 320 (k3_gguf_gsq) / 1280 (the multi-token module).
        ntok: x / out hold ntok rows instead of T (each row's result is the same whatever the count: the MTP eh_proj's 4 hc
        streams per token); its segment keeps that count through build(). swiglu_mul (the MTP modules only): the activation
        is silu(x) * swiglu_mul (fp32 [rows, ncols]; ggml's swiglu), staged in LDS like silu_scale's (Seg.work 2)"""
        rows = self.T if ntok is None else int(ntok)
        assert w.numel() * w.element_size() == nrows * ncols * 2, (w.numel(), nrows, ncols)
        assert x.dtype == torch.float32 and x.numel() == rows * ncols and out.numel() == rows * nrows, (x.numel(), out.numel(), rows, nrows, ncols)
        assert ncols % 2 == 0 and x.data_ptr() % 8 == 0 and w.data_ptr() % 4 == 0, 'bf16 pairs: even K, 8-byte aligned x'
        assert rows_per_task >= 1 and nrows >= 1 and rows >= 1, (rows_per_task, nrows, rows)
        assert ntok is None or (team == -1 and self.teams is None and silu_scale is None and swiglu_mul is None), 'bf16_mv ntok: plain k3_run segments only'
        assert silu_scale is None or swiglu_mul is None
        if swiglu_mul is not None:
            assert self.module in MTP_MODULES, f'bf16_mv swiglu_mul: the MTP modules only (not {self.module})'
            assert swiglu_mul.dtype == torch.float32 and swiglu_mul.numel() == x.numel(), (swiglu_mul.dtype, swiglu_mul.numel(), x.numel())
        assert (silu_scale is None and swiglu_mul is None) or rows * ncols <= (1280 if '_mt' in self.module else 320), f'staged activation of {rows} x {ncols} floats > the LDS stage of {self.module}'
        self.keep += [w, x, out] + ([] if swiglu_mul is None else [swiglu_mul])
        work = 2 if swiglu_mul is not None else 0 if silu_scale is None else 1
        self.segs.append(Seg(OP_BF16_MV, (nrows + rows_per_task - 1) // rows_per_task, dep, rows_per_task, w.data_ptr(),
                             None if swiglu_mul is None else swiglu_mul.data_ptr(),
                             x.data_ptr(), None, out.data_ptr(), 0.0 if silu_scale is None else float(silu_scale), nrows, ncols,
                             1, work, team if ntok is None else rows, mmvf_warps(ncols), ncols * 2))   # pad1: W, pad2: row bytes
        if ntok is not None: self.team_fixed.add(len(self.segs) - 1)
        return len(self.segs) - 1

    def f32_mv(self, w, x, out, rows_per_task=8, dep=-1, team=-1):
        """F32 weights w [N, K] (row-major) times fp32 x [K] -> out [N]"""
        N, K = w.shape; N = self.T * N; assert K % 4 == 0; self.keep += [w, x, out]
        self.segs.append(Seg(OP_F32_MV, (N + rows_per_task - 1) // rows_per_task, dep, rows_per_task, w.data_ptr(), None,
                             x.data_ptr(), None, out.data_ptr(), 0.0, N, K, 1, 0, team, 0, K * 4))   # pad2: row bytes (prefetch)
        return len(self.segs) - 1

    def router(self, logits, k, ids, wts, clamp=6.103515625e-5, dep=-1, team=-1):
        """softmax gating, top-k ids (int32) and normalised weights (llama.cpp build_moe_ffn, norm_w)"""
        E = logits.numel() // self.T; assert E <= 1024 and k <= 64; self.keep += [logits, ids, wts]
        self.segs.append(Seg(OP_ROUTER, self.T, dep, 0, None, wts.data_ptr(), logits.data_ptr(), None, ids.data_ptr(), clamp,
                             E, 0, 1, 0, team, k, 0))
        return len(self.segs) - 1

    def shexp_add(self, moe, shexp, gate_logit, out, rows_per_task=1024, dep=-1, team=-1):
        """out = moe + shexp * sigmoid(gate_logit[0])"""
        n = out.numel(); self.keep += [moe, shexp, gate_logit, out]
        self.segs.append(Seg(OP_SHEXP_ADD, (n + rows_per_task - 1) // rows_per_task, dep, rows_per_task, shexp.data_ptr(),
                             gate_logit.data_ptr(), moe.data_ptr(), None, out.data_ptr(), 0.0, n, 0, 1, 0, team, 0, 0))
        return len(self.segs) - 1

    def rms_mul(self, x, w, out, eps, dep=-1, team=-1, q8=None):
        """out[r] = rms_norm(x[r]) * w[r] for each row r of x [R, K] (w [R, K] fp32)"""
        K = x.shape[-1]; R = x.numel() // K; wr = w.numel() // K; self.keep += [x, w, out]
        assert R % wr == 0
        if q8 is not None: assert K % 32 == 0; self.keep.append(q8)
        self.segs.append(Seg(OP_RMS_MUL, R, dep, 1, w.data_ptr(), None, x.data_ptr(), None if q8 is None else q8.data_ptr(), out.data_ptr(), eps, R, K, 1, 0,
                             team, wr if wr < R else 0, 0))
        return len(self.segs) - 1

    def hc_mean(self, xn, gate, out, rows_per_task=256, dep=-1, team=-1, q8=None):
        """out[i] = (1 / hc) sum_s xn[s, i] * sigmoid(gate[s, i]); xn, gate [hc, n]"""
        n = xn.shape[-1]; hc = xn.numel() // n // self.T; N = self.T * n; self.keep += [xn, gate, out]
        assert n % rows_per_task == 0 or self.T == 1
        if q8 is not None: assert n % 32 == 0 and rows_per_task % 32 == 0; self.keep.append(q8)
        self.segs.append(Seg(OP_HC_MEAN, (N + rows_per_task - 1) // rows_per_task, dep, rows_per_task, gate.data_ptr(), None,
                             xn.data_ptr(), None if q8 is None else q8.data_ptr(), out.data_ptr(), 1.0 / hc, N, hc, 1, 0, team,
                             n if self.T > 1 else 0, 0))
        return len(self.segs) - 1

    def hc_post(self, block, inject, res_in, res_out, rows_per_task=512, dep=-1, team=-1):
        """res_out[s, i] = block[i] * 2 sigmoid(inject[s] / hc) + res_in[s, i]"""
        n = res_in.shape[-1]; hc = res_in.numel() // n // self.T; N = self.T * n; self.keep += [block, inject, res_in, res_out]
        assert n % rows_per_task == 0 or self.T == 1
        self.segs.append(Seg(OP_HC_POST, (N + rows_per_task - 1) // rows_per_task, dep, rows_per_task, res_in.data_ptr(),
                             inject.data_ptr(), block.data_ptr(), None, res_out.data_ptr(), 1.0 / hc, N, hc, 1, 0, team,
                             n if self.T > 1 else 0, 0))
        return len(self.segs) - 1

    def gdn_conv(self, x, w, state, out, xprev, rows_per_task=256, dep=-1, team=-1):
        """GDN short conv: out = silu(conv4([state, x])) per channel; state [C, 3] shifts in place; w [C, 4]"""
        C = x.numel() // self.T; assert xprev.numel() >= x.numel(); self.keep += [x, w, state, out, xprev]
        self._guard_replay()   # the replay reads params[1] rows of the previous pass's inputs (bounded by the guard)
        sg = Seg(OP_GDN_CONV, (C + rows_per_task - 1) // rows_per_task, dep, rows_per_task, w.data_ptr(),
                 state.data_ptr(), x.data_ptr(), self.params.data_ptr(), out.data_ptr(), 0.0, C, 0, 1, 0, team, 0, 0)
        sg.aux = xprev.data_ptr(); self.segs.append(sg)
        return len(self.segs) - 1

    def gdn_l2(self, x, out, heads, dim, eps, x_stride=0, dep=-1, team=-1):
        """per-head L2 norm (rms_norm with eps / dim, then * 1 / sqrt(dim)); tasks (token, head), x_stride = floats between
        tokens in x (0: heads * dim)"""
        self.keep += [x, out]
        self.segs.append(Seg(OP_GDN_L2, self.T * heads, dep, 1, None, None, x.data_ptr(), None, out.data_ptr(), eps / dim,
                             self.T * heads, dim, 1, 0, team, x_stride, 0))
        return len(self.segs) - 1

    def gdn_gates(self, alpha, beta, dt, a, out, dep=-1, team=-1):
        """out[:H] = softplus(alpha + dt) * a; out[H:] = sigmoid(beta)"""
        H = alpha.numel() // self.T; self.keep += [alpha, beta, dt, a, out]
        self.segs.append(Seg(OP_GDN_GATES, 1, dep, 1, beta.data_ptr(), dt.data_ptr(), alpha.data_ptr(), a.data_ptr(),
                             out.data_ptr(), 0.0, H, 0, 1, 0, team, 0, 0))
        return len(self.segs) - 1

    def gdn_step(self, qk, v, gates, block, out, H, S, nk, v_stride, dep=-1, team=-1):
        """the gated delta rule per value head for the pass's T tokens, with lazy commit. block: the layer's [state H*S*S |
        saved step inputs T x (nk*S + H*S + 2H)] (gdn_block_size); v_stride: floats between tokens in v"""
        assert S == 128, 'op_gdn_step is specialised for 128-wide heads'
        assert block.numel() >= gdn_block_size(H, S, nk, self.T)
        self.keep += [qk, v, gates, block, out]
        self._guard_replay()   # the replay reads params[1] saved records (bounded by the guard)
        sg = Seg(OP_GDN_STEP, H * (S // 32), dep, 32, v.data_ptr(), gates.data_ptr(), qk.data_ptr(), self.params.data_ptr(),
                 out.data_ptr(), 0.0, H, S, 1, 0, team, nk, v_stride)
        sg.aux = block.data_ptr(); self.segs.append(sg)
        return len(self.segs) - 1

    def gdn_save(self, qk, v, gates, block, H, S, nk, v_stride, dep=-1):
        """after the step: this pass's step inputs into the layer block's saved-input records (one task per token)"""
        self.keep += [qk, v, gates, block]
        sg = Seg(OP_GDN_SAVE, self.T, dep, 1, v.data_ptr(), gates.data_ptr(), qk.data_ptr(), None, None, 0.0, H, S, 1, 0, -1,
                 nk, v_stride)
        sg.aux = block.data_ptr(); self.segs.append(sg)
        return len(self.segs) - 1

    def gdn_norm(self, o, w, z, out, H, S, eps, dep=-1, team=-1, q8=None):
        """per head: rms_norm(o) * w * sigmoid(z)"""
        self.keep += [o, w, z, out]; H = self.T * H
        if q8 is not None: assert S % 32 == 0; self.keep.append(q8)
        self.segs.append(Seg(OP_GDN_NORM, H, dep, 1, w.data_ptr(), z.data_ptr(), o.data_ptr(), None if q8 is None else q8.data_ptr(), out.data_ptr(), eps, H,
                             S, 1, 0, team, 0, 0))
        return len(self.segs) - 1

    def qk_nr(self, x, head_stride, heads, dim, w, pos, out, n_rot, theta_scale, eps, dep=-1, team=-1):
        """per head: rms_norm(x) * w, then NEOX RoPE on the first n_rot dims at position pos (int32 [1] on the device)"""
        self.keep += [x, w, pos, out]
        eb = int(np.array([eps], dtype=np.float32).view(np.int32)[0])
        self.segs.append(Seg(OP_QK_NR, self.T * heads, dep, 1, w.data_ptr(), None, x.data_ptr(), pos.data_ptr(), out.data_ptr(),
                             theta_scale, self.T * heads, dim, n_rot, 0, team, head_stride, eb))
        return len(self.segs) - 1

    def kv_append(self, k, v, cache, pos, q8=False, dep=-1, team=-1):
        """cache[pos] = (k, v) in f16 or q8_0 (block_q8_0, as llama.cpp -ctk/-ctv q8_0); per position [K | V]. The cache's
        rows (from its allocation) bound pos .. pos + T - 1 in the device guard of every launch (k3_run's prologue)"""
        M = k.numel() // self.T; self._guard_pos(pos, kv_rows(cache, M, q8))
        self.keep += [k, v, cache, pos]
        self.segs.append(Seg(OP_KV_APPEND, self.T, dep, 1, v.data_ptr(), None, k.data_ptr(), pos.data_ptr(), cache.data_ptr(),
                             0.0, M, 0, 1, int(q8), team, 0, 0))
        return len(self.segs) - 1

    def split_attn(self):
        """True on the modules whose attention is the split pair (ATTN_SPLIT_MODULES)"""
        return getattr(self, "module", None) in ATTN_SPLIT_MODULES

    def attn_dec(self, q, cache, pos, out, heads, kv_heads, dim, scale, q8=False, dep=-1, team=-1, max_cells=None, part=None):
        """decode attention over positions 0..pos (f16 or q8_0 cache), grouped kv heads. On the split modules
        (split_attn()): the OP_ATTN_PART / OP_ATTN_COMB pair (_attn_split; max_cells / part there), the COMB's index returned"""
        if self.split_attn():
            assert team == -1, 'the split attention pair: k3_run segments only'
            return self._attn_split(q, cache, pos, None, out, heads, kv_heads, dim, scale, q8, dep, max_cells, part)
        assert max_cells is None and part is None, 'max_cells / part: the split attention modules only'
        assert dim <= 512, 'op_attn_dec holds at most 2 accumulators per thread'
        self._guard_pos(pos, kv_rows(cache, kv_heads * dim, q8))   # (the guard bounds the rows it reads)
        self.keep += [q, cache, pos, out]
        self.segs.append(Seg(OP_ATTN_DEC, self.T * heads, dep, 1, cache.data_ptr(), None, q.data_ptr(), pos.data_ptr(), out.data_ptr(),
                             scale, self.T * heads, dim, 1, int(q8), team, kv_heads, 0))
        return len(self.segs) - 1

    def gate_mul(self, a, gate, head_stride, offset, out, dim, rows_per_task=256, dep=-1, team=-1, q8=None):
        """out = a * sigmoid(gate[h * head_stride + offset + d])"""
        n = a.numel(); self.keep += [a, gate, out]
        if q8 is not None: assert n % 32 == 0 and rows_per_task % 32 == 0; self.keep.append(q8)
        self.segs.append(Seg(OP_GATE_MUL, (n + rows_per_task - 1) // rows_per_task, dep, rows_per_task, gate.data_ptr(), None,
                             a.data_ptr(), None if q8 is None else q8.data_ptr(), out.data_ptr(), 0.0, n, dim, 1, 0, team, head_stride, offset))
        return len(self.segs) - 1

    def ple_gate(self, keyn, queryn, value, gated, gate, dep=-1, team=-1):
        """PLE gate per hc stream and gated value: gated[st] = value * sigmoid(sgn(s) sqrt(|s|)), s = keyn . queryn / sqrt(D)"""
        D = keyn.shape[-1]; hc = keyn.numel() // D // self.T; self.keep += [keyn, queryn, value, gated, gate]
        self.segs.append(Seg(OP_PLE_GATE, self.T, dep, 1, queryn.data_ptr(), value.data_ptr(), keyn.data_ptr(), gate.data_ptr(),
                             gated.data_ptr(), 0.0, D, hc, 1, 0, team, 0, 0))
        return len(self.segs) - 1

    def ple_conv(self, x, w_f16, hist, out, xprev, dil=3, rows_per_task=1024, dep=-1, team=-1):
        """PLE dilated causal conv (4 taps, dilation dil) + silu; hist [C, 3 dil] shifts in place"""
        C = x.numel() // self.T; assert xprev.numel() >= x.numel(); self.keep += [x, w_f16, hist, out, xprev]
        self._guard_replay()   # the replay reads params[1] rows of the previous pass's inputs (bounded by the guard)
        sg = Seg(OP_PLE_CONV, (C + rows_per_task - 1) // rows_per_task, dep, rows_per_task, w_f16.data_ptr(),
                 hist.data_ptr(), x.data_ptr(), self.params.data_ptr(), out.data_ptr(), 0.0, C, 0, 1, 0, team, dil, 0)
        sg.aux = xprev.data_ptr(); self.segs.append(sg)
        return len(self.segs) - 1

    def add_sum(self, a, b, y, rows_per_task=1024, dep=-1, team=-1):
        """y += (a + b)"""
        n = y.numel(); self.keep += [a, b, y]
        self.segs.append(Seg(OP_ADD_SUM, (n + rows_per_task - 1) // rows_per_task, dep, rows_per_task, b.data_ptr(), None,
                             a.data_ptr(), None, y.data_ptr(), 0.0, n, 0, 1, 0, team, 0, 0))
        return len(self.segs) - 1

    def hadamard(self, x, n, rows_per_task=1, dep=-1, team=-1):
        """in place: the orthonormal Walsh-Hadamard transform of each contiguous chunk of n values (64 <= n <= 256)"""
        assert n in (64, 128, 256) and x.numel() % n == 0
        C = x.numel() // n; self.keep.append(x)
        self.segs.append(Seg(OP_HADAMARD, (C + rows_per_task - 1) // rows_per_task, dep, rows_per_task, None, None, None, None,
                             x.data_ptr(), 0.0, C, n, 1, 0, team, 0, 0))
        return len(self.segs) - 1

    def hc_post_rms(self, block, inject, res, norm_w, xn, eps, q8=None, inject_scale=0.25, dep=-1):
        """res[h] += block * 2 sigmoid(inject[h] * inject_scale), then xn[h] = rms_norm(res[h]) * norm_w[h] (+ q8_1 copy);
        one task per stream (bitwise op_hc_post then op_rms_mul)"""
        N = res.shape[-1]; hc = norm_w.numel() // N; assert N % 32 == 0
        self.keep += [block, inject, res, norm_w, xn] + ([q8] if q8 is not None else [])
        sg = Seg(OP_HC_POST_RMS, self.T * hc, dep, 1, res.data_ptr(), inject.data_ptr(), block.data_ptr(),
                 None if q8 is None else q8.data_ptr(), xn.data_ptr(), inject_scale, N, hc, 1, 0, -1, 0,
                 int(np.array(eps, dtype=np.float32).view(np.int32)))
        sg.aux = norm_w.data_ptr(); self.segs.append(sg)
        return len(self.segs) - 1

    def xchg(self, partial, out, peer_slot, my_slot, epoch_base, index, card, local_only=False, dep=-1, pull=None, pull_mode=0):
        """two-card all-reduce: out = card 0's partial + card 1's (same bits on both cards). peer_slot / my_slot: device
        addresses (int) of the receive slots (uncached VRAM, 16 header words + N floats) on the peer / this card;
        epoch_base: int32 [1] on this card (token * exchanges per token); index: this exchange's number in the token"""
        N = partial.numel(); assert N % 4 == 0; self.keep += [partial, out, epoch_base]
        self.segs.append(Seg(OP_XCHG, 1, dep, 1, peer_slot, my_slot, partial.data_ptr(), epoch_base.data_ptr(), out.data_ptr(),
                             0.0, N, index, 1, card, -1, int(local_only), int(pull_mode)))
        if pull_mode: self.segs[-1].aux = pull   # payload address in the pulling card's slot (k3_gguf_ops.h op_xchg)
        return len(self.segs) - 1

    def stall_report(self):
        """after a bail (before the sync block is cleared): the lowest segment that did not finish, with its claim / done
        counts and its dependencies' (the reported wait is only the first to expire; this is where the chain stopped)"""
        sy = self.sync.cpu().numpy(); claim, done = sy[:MAXSEG], sy[MAXSEG:2 * MAXSEG]
        names = {v: k for k, v in globals().items() if k.startswith('OP_') and isinstance(v, int)}
        for s, sg in enumerate(self.segs):
            if done[s] < sg.ntask:
                deps = [d for d in (sg.dep, sg.dep2p1 - 1) if d >= 0]
                return (f'first unfinished: segment {s} {names.get(sg.op)} claimed {claim[s]} done {done[s]} of {sg.ntask}; deps ' +
                        ', '.join(f'{d} {names.get(self.segs[d].op)} done {done[d]}/{self.segs[d].ntask}' for d in deps))
        return 'every segment finished'

    def gg_gu_dense(self, gate, up, fmt, nrows, ncols, row_bytes, xq, h, rows_per_task=8, dep=-1, up_fmt=None):
        """h [T, nrows] = silu(gate . x) * (up . x) for every token (the shared expert as a dense pair). One format decodes
        both: up_fmt (the up tensor's own; None = the caller vouches) must equal fmt, else run two gg_matvec + gg_q8_1(mul=)"""
        assert up_fmt is None or up_fmt == fmt, f'gg_gu_dense: gate {fmt} and up {up_fmt} differ (one format decodes both)'
        self.keep += [gate, up, xq, h]
        self.segs.append(Seg(OP_GG_GU_DENSE, (nrows + rows_per_task - 1) // rows_per_task, dep, rows_per_task, gate.data_ptr(),
                             up.data_ptr(), xq.data_ptr(), None, h.data_ptr(), 0.0, nrows, ncols, GG_ACT_Q8, GG_FMT[fmt], -1, 0,
                             row_bytes))
        return len(self.segs) - 1

    def hc_avg(self, res, out, rows_per_task=1024, dep=-1):
        """out [T, D] = the mean over the hc streams of res [T, hc, D] (llama.cpp build_hc_mean)"""
        D = out.shape[-1]; hc = res.numel() // D // self.T; N = self.T * D; self.keep += [res, out]
        self.segs.append(Seg(OP_HC_AVG, (N + rows_per_task - 1) // rows_per_task, dep, rows_per_task, None, None,
                             res.data_ptr(), None, out.data_ptr(), 0.0, N, hc, 1, 0, -1, D, 0))
        return len(self.segs) - 1

    # ---- QSA indexer ops (kernels/k3_qsa_ops.h). Buffers per QSA layer: IDXRAW uint8 [rows][136] (q8_0 raw keys, rows =
    # the KV positions), IDXPOOL uint8 [(rows + 3) / 4][136] (q8_0 pooled keys), SCORE fp32 [T][>= pool rows], SEL int32
    # [T][QSA_SELW]. Positions come from the pass's position word (pos: the same int32 the KV ops use, so k3_run's prologue
    # guard covers them with one word); the device clamps every position-indexed access to these rows as well.
    def _qsa_module(self):
        """a QSA op on a trunk tape: the tape runs on the trunk's QSA module (the same ops plus these; its caps too)"""
        if self.module in QSA_OF: self.module = QSA_OF[self.module]; self.caps = MODULE_CAPS.get(self.module)
        if self.module not in QSA_MODULES: raise ValueError(f'k3: module {self.module} has no QSA ops ({QSA_MODULES} only)')

    @staticmethod
    def _qsa_rows(buf, what):
        assert buf.dtype == torch.uint8 and buf.is_contiguous() and buf.numel() % QSA_ROWB == 0, (what, buf.dtype, buf.numel())
        n = buf.numel() // QSA_ROWB
        assert 0 < n < 2 ** 31, (what, n)
        return n

    def idx_append(self, k_raw, raw, pool, pos, k_norm, theta_scale, eps, n_rot=64, dep=-1, pooled=None, rows=None):
        """IDXRAW[pos + t] = q8_0(k_raw[t]) for the T pass tokens, and for each t with (pos + t) % 4 == 3 the pooled key of
        pool (pos + t) / 4 -> IDXPOOL (mean of its 4 dequantized raw rows, RMSNorm * k_norm, NEOX RoPE of n_rot = 64 dims at
        the pool's first position, q8_0: llama.cpp's build_qsa_sel). k_raw fp32 [T, 128]; pooled: fp32 [T, 128] receives the
        pooled keys before quantization (rows that end a pool; tests) or None.
        rows (a T = 1 engine only): k_raw holds `rows` positions pos .. pos + rows - 1, one task each (FlashNextTP._backfill:
        the pools of positions the dense tape wrote, from their IDXRAW rows dequantized; q8_0 of a dequantized q8_0 row is
        the same bytes, so this rewrites those rows unchanged). The device clamps every position to the IDXRAW rows; the
        caller checks the span on the host"""
        self._qsa_module()
        nt = self.T if rows is None else int(rows)
        assert rows is None or (self.T == 1 and self.teams is None and 1 <= nt < 2 ** 31), (self.T, rows)
        rows, prow = self._qsa_rows(raw, 'IDXRAW'), self._qsa_rows(pool, 'IDXPOOL')
        assert k_raw.dtype == torch.float32 and k_raw.is_contiguous() and k_raw.numel() == nt * QSA_D, (k_raw.dtype, k_raw.numel())
        assert k_norm.dtype == torch.float32 and k_norm.numel() == QSA_D and n_rot == 64, (k_norm.numel(), n_rot)
        assert prow == (rows + 3) // 4, f'IDXPOOL rows {prow} != (IDXRAW rows {rows} + 3) / 4'
        assert pooled is None or (pooled.dtype == torch.float32 and pooled.numel() == nt * QSA_D)
        self._guard_pos(pos, rows)
        self.keep += [k_raw, raw, pool, pos, k_norm] + ([] if pooled is None else [pooled])
        eb = int(np.array([eps], dtype=np.float32).view(np.int32)[0])
        self.segs.append(Seg(OP_IDX_APPEND, nt, dep, 1, k_norm.data_ptr(), None if pooled is None else pooled.data_ptr(),
                             k_raw.data_ptr(), pos.data_ptr(), raw.data_ptr(), theta_scale, rows, prow, n_rot, 0, -1, 0, eb, 0,
                             pool.data_ptr()))
        return len(self.segs) - 1

    def qsa_score(self, q, pool, pos, score, w=None, min_nv=QSA_TOPK, ntask=None, dep=-1):
        """SCORE[t][j] = sum_h relu(q[t, h] . dequant(IDXPOOL[j])) * w for the pools j < (pos + t + 1) / 4 (llama.cpp's vec
        lightning indexer order, every T). q fp32 [T, 4, 128] (normed, RoPE'd); score fp32 [T, K] with K >= the pool rows.
        Nothing runs while the last row's visible pools are <= min_nv (the selection is the identity there; 0: always):
        0 <= min_nv <= QSA_TOPK, else SELECT would read stale scores for nv in (QSA_TOPK, min_nv].
        ntask: tasks of 32-pool chunks, 1 <= ntask <= ceil(pool rows / 32) (default min(64, chunks)): the kernel strides
        its chunk loop by ntask * 32 (a negative or huge ntask would index the pools out of bounds).
        The score buffer is registered with its IDXPOOL rows: a qsa_select of it must use the same pool_rows"""
        self._qsa_module()
        prow = self._qsa_rows(pool, 'IDXPOOL')
        assert q.dtype == torch.float32 and q.is_contiguous() and q.numel() == self.T * QSA_H * QSA_D, (q.dtype, q.numel())
        assert score.dtype == torch.float32 and score.is_contiguous() and score.dim() == 2 and score.shape[0] == self.T, score.shape
        K = int(score.shape[1]); assert K >= prow, f'SCORE row stride {K} < IDXPOOL rows {prow}'
        nch = (prow + 31) // 32
        if ntask is None: ntask = min(64, nch)
        if isinstance(ntask, bool) or not isinstance(ntask, (int, np.integer)) or not 1 <= ntask <= nch:
            raise ValueError(f'qsa_score: ntask {ntask!r} outside [1, {nch}] (32-pool chunks of {prow} IDXPOOL rows)')
        if isinstance(min_nv, bool) or not isinstance(min_nv, (int, np.integer)) or not 0 <= min_nv <= QSA_TOPK:
            raise ValueError(f'qsa_score: min_nv {min_nv!r} outside [0, {QSA_TOPK}] (SELECT reads every score once nv > {QSA_TOPK})')
        w = float(np.float32(1.0) / np.sqrt(np.float32(QSA_D))) if w is None else float(w)
        if not np.isfinite(w): raise ValueError(f'qsa_score: weight {w} not finite')
        self._guard_pos(pos, 4 * prow)
        self.keep += [q, pool, pos, score]
        if not hasattr(self, '_qsa_scores'): self._qsa_scores = {}
        self._qsa_scores[score.data_ptr()] = (prow, len(self.segs))   # (IDXPOOL rows, this segment's index)
        self.segs.append(Seg(OP_QSA_SCORE, ntask, dep, 1, pool.data_ptr(), None, q.data_ptr(), pos.data_ptr(), score.data_ptr(),
                             w, prow, K, 0, 0, -1, int(min_nv), 0))
        return len(self.segs) - 1

    def qsa_select(self, score, pos, sel, pool_rows, dep=-1):
        """SEL[t] = the pools pass row t attends to: [0] count, [1] tail cells (pos + t + 1) % 4, [2] nv, [3] threshold key,
        then the pool ids ascending (all nv when nv <= 512, else the top 512 by score, lowest index first among ties).
        A score buffer written by a qsa_score of this tape: pool_rows must be that SCORE's IDXPOOL rows (more would read
        score entries SCORE never writes) and the select must wait for that SCORE (dep, directly or through other segments).
        A buffer no SCORE of this tape writes (tests: scores from the host) is taken as given"""
        self._qsa_module()
        assert score.dtype == torch.float32 and score.is_contiguous() and score.dim() == 2 and score.shape[0] == self.T, score.shape
        assert sel.dtype == torch.int32 and sel.is_contiguous() and sel.numel() == self.T * QSA_SELW, (sel.dtype, sel.numel())
        K = int(score.shape[1])
        if isinstance(pool_rows, bool) or not isinstance(pool_rows, (int, np.integer)) or not 0 < pool_rows <= K:
            raise ValueError(f'qsa_select: pool_rows {pool_rows!r} outside [1, {K}] (the SCORE row stride)')
        reg = getattr(self, '_qsa_scores', {}).get(score.data_ptr())
        if reg is not None:
            if pool_rows != reg[0]:
                raise ValueError(f'qsa_select: pool_rows {pool_rows} != {reg[0]}, the IDXPOOL rows of the SCORE (segment {reg[1]}) '
                                 f'that writes this score buffer: it would read scores that SCORE never writes')
            if not self._waits_for(dep, reg[1]):
                raise ValueError(f'qsa_select: dep {dep} does not wait for the SCORE (segment {reg[1]}) that writes its scores')
        self._guard_pos(pos, 4 * pool_rows)
        self.keep += [score, pos, sel]
        self.segs.append(Seg(OP_QSA_SELECT, self.T, dep, 1, None, None, score.data_ptr(), pos.data_ptr(), sel.data_ptr(),
                             0.0, int(pool_rows), K, 0, 0, -1, 0, 0))
        return len(self.segs) - 1

    def _waits_for(self, dep, seg):
        """True when a segment depending on `dep` starts after segment `seg` finished: seg reachable from dep through the
        dep / dep2p1 links (a segment's tasks start after its dependencies finished, so the order is transitive)"""
        todo, seen = [dep], set()
        while todo:
            d = todo.pop()
            if d == seg: return True
            if d < seg or d in seen: continue   # (links only point to earlier segments: below seg it cannot be reached)
            seen.add(d); sg = self.segs[d]
            todo.append(sg.dep)
            if sg.dep2p1: todo.append(sg.dep2p1 - 1)
        return False

    def attn_sel(self, q, cache, pos, sel, out, heads, kv_heads, dim, scale, q8=False, dep=-1, max_cells=None, part=None):
        """attn_dec over the cells QSA_SELECT kept for each pass row (sel int32 [T][QSA_SELW]: its selected pools' cells
        ascending, then the tail (pos + 1) % 4 cells): list position k in place of cell k, the same arithmetic, so the
        identity selection (nv <= 512) is attn_dec bit for bit. The device clamps the count and pool ids to the row's
        visible pools (every cell <= its position). On the split modules (the QSA modules are): the split pair over the
        same list (_attn_split), the COMB's index returned"""
        self._qsa_module()
        assert sel.dtype == torch.int32 and sel.is_contiguous() and sel.numel() == self.T * QSA_SELW, (sel.dtype, sel.numel())
        if self.split_attn(): return self._attn_split(q, cache, pos, sel, out, heads, kv_heads, dim, scale, q8, dep, max_cells, part)
        assert max_cells is None and part is None, 'max_cells / part: the split attention modules only'
        assert dim <= 512, 'op_attn_sel holds at most 2 accumulators per thread'
        self._guard_pos(pos, kv_rows(cache, kv_heads * dim, q8))
        self.keep += [q, cache, pos, sel, out]
        self.segs.append(Seg(OP_ATTN_SEL, self.T * heads, dep, 1, cache.data_ptr(), sel.data_ptr(), q.data_ptr(), pos.data_ptr(),
                             out.data_ptr(), scale, self.T * heads, dim, 1, int(q8), -1, kv_heads, 0))
        return len(self.segs) - 1

    def _attn_split(self, q, cache, pos, sel, out, heads, kv_heads, dim, scale, q8, dep, max_cells, part):
        """the split attention pair (kernels/k3_attn_split.h): OP_ATTN_PART, tasks (split, row, kv head) over each pass
        row's list (sel None: the identity 0 .. pos; else the QSA selection rows) in slices of ATTN_SS cells, records
        [T][heads][NS][ATTN_RS] floats in `part`; then OP_ATTN_COMB (task per row and query head, waits for PART). NS =
        ceil(max_cells / ATTN_SS) splits per row: max_cells = the longest list a pass row can have (default: the cache's
        rows, or min(rows, 4 * QSA_TOPK + 3) with sel; a caller that bounds its positions lower may pass less); the device
        clamps every list to NS * ATTN_SS cells (a longer one would lose cells, never write outside `part`). part: a
        float32 buffer of at least T * heads * NS * ATTN_RS (shared by attention segments that run one after another:
        the caller's dependency chain must order them), None: allocated here. also(COMB, d) puts d on the PART"""
        assert self.teams is None, 'the split attention pair: k3_run tapes only'
        G = ATTN_G_OF.get(self.module, ATTN_G)
        if dim != ATTN_D or kv_heads < 1 or heads != G * kv_heads:
            raise ValueError(f'k3 {self.module}: attention runs only as the split pair here (head dim {ATTN_D}, {G} '
                             f'query heads per kv head); got dim {dim}, heads {heads}, kv_heads {kv_heads}')
        rows = kv_rows(cache, kv_heads * dim, q8)
        longest = min(rows, 4 * QSA_TOPK + 3) if sel is not None else rows
        mc = longest if max_cells is None else max_cells
        if isinstance(mc, bool) or not isinstance(mc, (int, np.integer)) or not 1 <= int(mc) <= rows:
            raise ValueError(f'attn split: max_cells {max_cells!r} outside [1, {rows}] (the cache rows)')
        SS = ATTN_SS_OF.get(self.module, ATTN_SS)
        NS = (int(mc) + SS - 1) // SS
        need = self.T * heads * NS * ATTN_RS
        assert q.dtype == torch.float32 and q.is_contiguous() and q.numel() == self.T * heads * dim, (q.dtype, q.numel())
        assert out.dtype == torch.float32 and out.is_contiguous() and out.numel() == self.T * heads * dim, (out.dtype, out.numel())
        if part is None: part = torch.zeros(need, dtype=torch.float32, device=q.device)
        assert part.dtype == torch.float32 and part.is_contiguous() and part.numel() >= need, \
            f'attn split: the record buffer holds {part.numel()} floats < {need} (T {self.T} x {heads} heads x {NS} splits x {ATTN_RS})'
        self._guard_pos(pos, rows)   # (the guard bounds the rows it reads: every list cell <= pos)
        self.keep += [q, cache, pos, out, part] + ([] if sel is None else [sel])
        sp = None if sel is None else sel.data_ptr()
        self.segs.append(Seg(OP_ATTN_PART, NS * self.T * kv_heads, dep, 1, cache.data_ptr(), sp, q.data_ptr(), pos.data_ptr(),
                             part.data_ptr(), scale, self.T * heads, dim, 1, int(bool(q8)), -1, kv_heads, NS))
        pi = len(self.segs) - 1
        self.segs.append(Seg(OP_ATTN_COMB, self.T * heads, pi, 1, None, sp, part.data_ptr(), pos.data_ptr(), out.data_ptr(),
                             0.0, self.T * heads, dim, 1, int(bool(q8)), -1, kv_heads, NS))
        if not hasattr(self, '_split_part'): self._split_part = {}
        self._split_part[pi + 1] = pi
        return pi + 1

    def also(self, seg, dep2):
        """segment seg also waits for segment dep2 (k3_run: up to two dependencies per segment); returns seg. seg = a split
        attention pair's COMB: the PART gets it (it reads the inputs; the COMB waits for the PART)"""
        tgt = getattr(self, '_split_part', {}).get(seg, seg)
        assert 0 <= dep2 < tgt, (dep2, tgt)
        assert tgt == seg or self.segs[tgt].dep2p1 == 0, f'also: the split PART {tgt} waits for {self.segs[tgt].dep2p1 - 1} already'
        self.segs[tgt].dep2p1 = dep2 + 1
        return seg

    def nop(self, ntask, work, dep=-1, team=-1):
        sink = torch.zeros(1, device='cuda'); self.keep.append(sink)
        self.segs.append(Seg(OP_NOP, ntask, dep, 0, None, None, None, None, sink.data_ptr(), 0.0, 0, 0, 0, work, team, 0, 0))
        return len(self.segs) - 1

    def inventory(self):
        """{(op, format): segment count} of the formatted segments"""
        inv = {}
        for sg in self.segs:
            kd = seg_kind(sg)
            if kd: inv[kd] = inv.get(kd, 0) + 1
        return inv

    def unsupported(self):
        """the tape's (op, format) pairs its module does not implement (empty for modules without a capability set)"""
        return set() if self.caps is None else set(self.inventory()) - self.caps

    def pending(self):
        """the tape's whitelisted pairs its module has no device code for yet (PENDING)"""
        return set(self.inventory()) & PENDING.get(self.module, frozenset())

    def build(self, strict=True):
        """strict: refuse (ValueError) a tape with (op, format) pairs its module does not implement; strict=False builds it
        anyway (to inspect: inventory()) and run() refuses it instead"""
        assert len(self.segs) <= MAXSEG, f'k3: {len(self.segs)} segments > MAXSEG {MAXSEG} (the sync arrays would overflow)'
        if self.module in DRAFT_MODULES:   # (only the drafter's ops are compiled into its module; trunk ops need their flags)
            for i, sg in enumerate(self.segs):
                flagged = {OP_RMS_MUL: sg.aux, OP_QK_NR: sg.aux, OP_KV_APPEND: sg.aux, OP_ATTN_DEC: sg.pad2}.get(sg.op, 1)
                if sg.op not in DRAFT_OPS or not flagged:
                    raise ValueError(f'k3: module {self.module} has no op {sg.op} (segment {i}{"" if flagged else ", unflagged"})')
        q = [i for i, sg in enumerate(self.segs) if sg.op in QSA_OPS]
        if q and (self.module not in QSA_MODULES or self.teams is not None):   # (k3_run elsewhere runs them as nops)
            raise ValueError(f'k3: module {self.module} has no QSA ops (segments {q[:8]}; k3_run in {QSA_MODULES} only)')
        sa = [i for i, sg in enumerate(self.segs) if sg.op in ATTN_SPLIT_OPS]
        if sa and (self.module not in ATTN_SPLIT_MODULES or self.teams is not None):
            raise ValueError(f'k3: module {self.module} has no split attention ops (segments {sa[:8]}; k3_run in {ATTN_SPLIT_MODULES} only)')
        da = [i for i, sg in enumerate(self.segs) if sg.op in (OP_ATTN_DEC, OP_ATTN_SEL)]
        if da and self.module in ATTN_SPLIT_MODULES:
            raise ValueError(f'k3: module {self.module} runs attention only as the split pair (segments {da[:8]} are '
                             f'OP_ATTN_DEC / OP_ATTN_SEL: no device code there; Engine.attn_dec / attn_sel emit the pair)')
        bad, pend = self.unsupported(), self.pending()
        self.refuse = (f'k3: module {self.module} does not implement {sorted(bad)}' if bad else
                       f'k3: module {self.module} has no device code yet for {sorted(pend)}' if pend else None)
        if bad and strict: raise ValueError(self.refuse)
        if self.T > 1:
            assert self.teams is None, 'T > 1 uses the team field (k3_run only)'
            for i, sg in enumerate(self.segs):
                if i not in self.team_fixed: sg.team = self.T
        raw = bytes(b''.join(bytes(s) for s in self.segs))
        self.tape = torch.frombuffer(bytearray(raw), dtype=torch.uint8).cuda()
        return self

    def run(self, check=True, trace=None):
        if self.refuse: raise RuntimeError(self.refuse + ' (refusing to launch)')
        gk = (self.guard_commit, tuple((k, c) for k, (_, c) in self.guard_caps.items()))   # the guard's bounds now
        if self.teams is not None or len(self.guard_caps) > 1 or self.guard_launch:
            self._guard(torch.cuda.current_stream().cuda_stream)   # the separate guard launch first, on the launch's stream
        # k3_run's own guard (its prologue): the one position word with its cache's rows, params when the commit is replayed
        g1 = next(iter(self.guard_caps.values()), None)
        garg = (g1[0].data_ptr() if g1 else 0, self.params.data_ptr() if self.guard_commit else 0, g1[1] if g1 else 0, self.T)
        if self.teams is None and trace is None:
            # the same launch as below with its argument array built once (rebuilt when any argument or the stream changes)
            from ..device.hip import hip, check as hip_check
            s = torch.cuda.current_stream().cuda_stream
            key = (torch.cuda.current_device(), s, self.tape.data_ptr(), len(self.segs), self.sync.data_ptr(), self.limit_us, self.grid, gk,
                   garg)
            lc = getattr(self, '_lc', None)
            if lc is None or lc[0] != key:
                vals = [ctypes.c_void_p(self.tape.data_ptr()), ctypes.c_int(len(self.segs)), ctypes.c_void_p(self.sync.data_ptr()),
                        ctypes.c_int(self.limit_us), ctypes.c_void_p(0), ctypes.c_void_p(garg[0]), ctypes.c_void_p(garg[1]),
                        ctypes.c_int(garg[2]), ctypes.c_int(garg[3])]
                params = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])
                lc = self._lc = (key, _m(self.module).k3_run.fn, vals, params, ctypes.c_void_p(s))
            hip_check(hip().hipModuleLaunchKernel(lc[1], self.grid, 1, 1, 256, 1, 1, 0, lc[4], lc[3], None))
        elif self.teams is None:
            _m(self.module).k3_run((self.grid,), (256,), self.tape, ctypes.c_int(len(self.segs)), self.sync, ctypes.c_int(self.limit_us),
                                   ctypes.c_void_p(0 if trace is None else trace.data_ptr()), ctypes.c_void_p(garg[0]),
                                   ctypes.c_void_p(garg[1]), ctypes.c_int(garg[2]), ctypes.c_int(garg[3]))
        else:
            tb, acc = [], 0
            for n in self.teams[:-1]: acc += n; tb.append(acc)
            tb = (tb + [self.grid] * 3)[:3]
            _m(self.module).k3_run2((self.grid,), (256,), self.tape, ctypes.c_int(len(self.segs)), self.sync, ctypes.c_int(self.limit_us),
                                    *(ctypes.c_int(v) for v in tb))
        if check:
            torch.cuda.synchronize()
            st = self.sync[2 * MAXSEG:].tolist()      # exit, err, err_seg, err_seen, bnd, bnd_what, bnd_val, bnd_cap
            b = bounds_status(st)
            if b:   # the bound guard refused a position / commit: never retried (the host checks should have caught it)
                self.sync.zero_()
                raise ContextBoundsError(f'k3: the device bound guard caught an out-of-range value: {b}')
            if st[1]:
                diag = self.stall_report()
                self.sync.zero_()   # err would otherwise stay set and fail every later launch at once
                raise RuntimeError(f'k3: a wait gave up at segment {st[2]} (done count seen {st[3]}); {diag}')
