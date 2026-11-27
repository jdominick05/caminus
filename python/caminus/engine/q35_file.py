"""The dense qwen35 line's file-level facts, pure (no torch / numpy / HIP needed to import): which decode module a file's
formats need (module_of), its block layout and shapes, the VRAM projection's terms, the bytes a tensor occupies on a card
at a format (upload_bytes), the tape's gate | up fusion rule (gu_fused) and the trunk's matrix roles (mat_tensors).
Moved here from engine/q35_dense.py so the CPU planner (engine/q35_plan.py, tools/q35_format_plan.py) reads them from a
GGUF header alone; q35_dense.py re-exports them, so every existing import keeps working."""
from .. import ggufio
from . import k3
from .own_layout import IQ_PAD, IQ_REC

MODULE = 'dec_q35_swift'
MODULE_TCAP = 'dec_q35_tcap'   # (ThinkingCap's formats: kernels/ops_q35_tcap.h)
MODULE_GSQ = 'dec_q35_gsq'     # (the Qwen3.8-27B GSQ-RCO files: ThinkingCap's formats + IQ2_XS, Q2_K, IQ1_M, IQ2_XXS, Q4_K)
GSQ_ONLY_FMT = frozenset({'IQ2_XS', 'Q2_K', 'IQ1_M', 'IQ2_XXS'})   # (a trunk matrix in one of these: the GSQ module)
SWIFT_FMT = frozenset({'Q4_K', 'Q6_K', 'Q8_0', 'Q5_K', 'F32'})   # (the matrices the Swift module's tiles read)
RECURRENT = ('conv', 'convx', 'gdn')   # the state kinds reset() zeroes (the KV rows are written before any pass reads them)

_BLOCK = {v[0]: (v[1], v[2]) for v in ggufio.TYPES.values()}   # format name -> (values per block, bytes per block)


def module_of(F):
    """the decode module a file's formats need: the Swift module when every trunk matrix is Q4_K / Q6_K / Q8_0 / Q5_K (or
    F32) and the embedding Q4_K; the GSQ module when a trunk matrix is IQ2_XS / Q2_K / IQ1_M / IQ2_XXS (the Qwen3.8-27B
    GSQ-RCO files: ThinkingCap's formats plus those and Q4_K); else the ThinkingCap module (its matrices IQ4_XS / IQ3_XXS /
    IQ3_S / IQ2_S / Q5_K / Q6_K / Q8_0, BF16 alpha / beta, an IQ2_S or Q4_K embedding)"""
    nl = trunk_blocks(F); mtp = tuple(f'blk.{i}.' for i in range(nl, int(F.hp('block_count'))))
    fm = {t.type for t in F.tensors if not t.name.startswith(mtp) and len(t.shape) == 2 and t.name != 'token_embd.weight'}
    if fm <= SWIFT_FMT and F.T['token_embd.weight'].type == 'Q4_K': return MODULE
    return MODULE_GSQ if fm & GSQ_ONLY_FMT else MODULE_TCAP


def kv_module(F, module=None, q8=False):
    """Resolve staged cache-specialized objects separately from weight-format selection."""
    if type(q8) is not bool:
        raise ValueError('Dense35Own: q8 must be a bool')
    selected = module or module_of(F)
    if q8:
        if selected in k3.Q35_KV_Q8_MODULES:
            return selected
        if selected not in k3.Q35_KV_Q8_OF:
            raise ValueError(f'Dense35Own: no Q8 KV specialization for {selected}')
        return k3.Q35_KV_Q8_OF[selected]
    if selected in k3.Q35_KV_Q8_MODULES:
        raise ValueError(f'Dense35Own: {selected} requires q8=True')
    return selected


def trunk_blocks(F):
    """the number of blocks of the model's forward: block_count less the multi-token-prediction blocks after them"""
    return int(F.hp('block_count')) - int(F.hp('nextn_predict_layers') or 0)


def block_kinds(F):
    """per trunk block: True = Gated DeltaNet, False = gated attention (attention.recurrent_layers where the file has
    it; without that key every block but each full_attention_interval-th)"""
    rec = F.hp('attention.recurrent_layers'); fai = int(F.hp('full_attention_interval', 4))
    return [bool(rec[i]) if rec is not None else (i + 1) % fai != 0 for i in range(trunk_blocks(F))]


def _shapes(F):
    """the header's shapes the engine and its projection read"""
    hp = F.hp
    return dict(nl=trunk_blocks(F), D=int(hp('embedding_length')), ffn=int(hp('feed_forward_length')), nh=int(hp('attention.head_count')),
                nkv=int(hp('attention.head_count_kv')), hd=int(hp('attention.key_length')), nrot=int(hp('rope.dimension_count')),
                S=int(hp('ssm.state_size')), NK=int(hp('ssm.group_count')), NV=int(hp('ssm.time_step_rank')),
                d_conv=int(hp('ssm.conv_kernel')), d_inner=int(hp('ssm.inner_size')))


def attn_part_elems(T, heads, cells):
    """floats of the split attention pair's records (kernels/ops_attn_split.h, the own split modules: k3.Q35_SPLIT_MODULES)
    for T pass rows of `heads` query heads over lists of up to `cells` cache rows: one record of k3.ATTN_RS floats per
    (row, head, ATTN_SS-cell split); one buffer shared by a tape's attention blocks (they run one after another)"""
    return int(T) * int(heads) * ((int(cells) + k3.ATTN_SS - 1) // k3.ATTN_SS) * k3.ATTN_RS


def upload_bytes(t, fmt=None):
    """bytes the card holds for tensor t in format fmt (its own when None): the GGUF bytes at that format, an F32 / BF16
    tensor widened to float32, the record layouts' tail (own_layout.IQ_PAD per matrix) and IQ1_M's records (66 bytes per
    56-byte block) — the weight term of _projection_bytes, per tensor, so a planner can recompute it at a chosen format"""
    fmt = fmt or t.type
    if fmt == 'F32':
        return 4 * t.n_values
    bs, ts = _BLOCK[fmt]
    b = t.n_values // bs * ts
    if fmt in IQ_REC or fmt in ('Q2_K', 'IQ1_M'):
        b += IQ_PAD
    if fmt == 'IQ1_M':
        b += t.n_values // 256 * 10
    return b


def _projection_terms(F, max_ctx, spec_T=1, q8=False):
    """the VRAM projection's terms separately (q35_dense._projection_bytes sums them): weights (upload_bytes of every trunk
    tensor at its own format, the embedding and the head included), the KV cache of every attention block, the Gated
    DeltaNet states (with spec_T tokens' saved inputs), the decode tape's split attention records, the Q8 KV flag word"""
    c = _shapes(F); nl = c['nl']
    mtp = tuple(f'blk.{i}.' for i in range(nl, int(F.hp('block_count'))))
    w = sum(upload_bytes(t) for t in F.tensors if not t.name.startswith(mtp))
    n_attn = sum(1 for g in block_kinds(F) if not g)
    CH = 2 * c['S'] * c['NK'] + c['S'] * c['NV']
    gdn = (nl - n_attn) * (k3.gdn_block_size(c['NV'], c['S'], c['NK'], spec_T) + CH * 3 + spec_T * CH) * 4
    part = 4 * attn_part_elems(1, c['nh'], int(max_ctx) + spec_T - 1)
    kv = n_attn * (int(max_ctx) + spec_T - 1) * k3.kv_pos_bytes(c['nkv'] * c['hd'], q8)
    return dict(weights=w, kv=kv, gdn=gdn, part=part, extra=(4 if q8 else 0))


def _projection_bytes(F, max_ctx, spec_T=1, q8=False):
    """bytes the engine allocates on its card: every trunk tensor (the embedding and the head too), the selected KV cache of
    every attention block, the Gated DeltaNet states (with spec_T tokens' saved inputs: Dense35Own's spec_T), the decode
    tape's split attention records (attn_part_elems)"""
    t = _projection_terms(F, max_ctx, spec_T, q8)
    return t['weights'] + t['kv'] + t['gdn'] + t['part'] + t['extra']


def gu_fused(module, fg_fmt, fu_fmt, gu=True, rpt=16):
    """whether Dense35Own's tape runs the feed-forward's gate | up as one segment (q35_dense.py _build_tape's fusion test):
    the module allows the fused tile (off on the ThinkingCap / GSQ modules: Dense35Own.__init__'s self.gu = gu and not
    tcap), both matrices share one one-round-trip format (k3.OWN_1RT_FMT), and the task size is the served one (rpt 16)"""
    return bool(gu) and module not in k3.Q35T_MODULES and fg_fmt == fu_fmt and fg_fmt in k3.OWN_1RT_FMT and rpt % 16 == 0 and rpt <= 64


# the trunk matrix roles of one decode token, per block kind (the header map benchmarks/own_mv_census.py times):
# a Gated DeltaNet block reads attn_qkv, attn_gate (z), ssm_out (+ the F32 alpha / beta); a gated-attention block reads
# attn_q, attn_k, attn_v, attn_output; every block reads ffn_gate, ffn_up, ffn_down; the head and the embedding once a token
GDN_MAT_ROLES = ('attn_qkv', 'attn_gate', 'ssm_out')
ATTN_MAT_ROLES = ('attn_q', 'attn_k', 'attn_v', 'attn_output')
FFN_MAT_ROLES = ('ffn_gate', 'ffn_up', 'ffn_down')
CENSUS_DENSE_ROLES = ('attn_qkv', 'attn_gate', 'ssm_out', 'attn_q', 'attn_k', 'attn_v', 'attn_output',
                      'ffn_gate', 'ffn_up', 'ffn_down', 'head')   # (benchmarks/own_mv_census.py's DENSE role list)


def mat_tensors(F, role):
    """[(tensor, None)] of one matrix role over the trunk blocks: 'head' is output.weight, else blk.<i>.<role>.weight of
    every trunk block that has it (the shape map benchmarks/own_mv_census.py groups by)"""
    if role == 'head': return [(F.T['output.weight'], None)]
    return [(F.T[f'blk.{il}.{role}.weight'], None) for il in range(trunk_blocks(F)) if f'blk.{il}.{role}.weight' in F.T]
