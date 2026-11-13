"""A tiny synthetic dense qwen35 GGUF with quantized matrices, written here (no model file): the shapes the own-kernel
engine reads (caminus.engine.q35_dense: residual 256, 128-wide DeltaNet heads, the real attention geometry: 256-wide heads,
6 query heads per kv head, as the split attention pair of the own modules takes it), four trunk
blocks (three Gated DeltaNet, one attention) and a declared multi-token-prediction block whose tensors are not in the
file. Between them the blocks hold every matrix format the engine has a tile for (Q4_K, Q6_K, Q8_0, Q5_K), the
feed-forward's gate and up in one format (the fused segment) and in two (the three-segment form), and F32 alpha / beta.
The quantized blocks are random valid blocks with small scales: caminus.ref_q35 decodes the same bytes exactly, so the
reference and the engine see one model.

  write(path, seed=5) -> dict(path, D, V, blocks, kinds, formats)"""
import struct

import numpy as np

D, FF, V, S, NK, NV, H, HKV, HD, NROT, DC = 256, 512, 64, 128, 1, 2, 6, 1, 256, 64, 4
CH = 2 * NK * S + NV * S
KINDS = [True, True, True, False]
TYPE_ID = {'F32': 0, 'Q8_0': 8, 'Q2_K': 10, 'Q4_K': 12, 'Q5_K': 13, 'Q6_K': 14, 'IQ2_XXS': 16, 'IQ2_XS': 17, 'IQ3_XXS': 18, 'IQ3_S': 21,
           'IQ2_S': 22, 'IQ4_XS': 23, 'IQ1_M': 29, 'BF16': 30}
# per block: the format of each matrix (ffn gate, up, down; then the block's own)
FORMATS = [
    dict(ffn_gate='Q4_K', ffn_up='Q4_K', ffn_down='Q4_K', attn_qkv='Q8_0', attn_gate='Q8_0', ssm_out='Q8_0'),
    dict(ffn_gate='Q4_K', ffn_up='Q6_K', ffn_down='Q5_K', attn_qkv='Q4_K', attn_gate='Q6_K', ssm_out='Q6_K'),
    dict(ffn_gate='Q6_K', ffn_up='Q6_K', ffn_down='Q6_K', attn_qkv='Q6_K', attn_gate='Q4_K', ssm_out='Q5_K'),
    dict(ffn_gate='Q8_0', ffn_up='Q8_0', ffn_down='Q8_0', attn_q='Q4_K', attn_k='Q8_0', attn_v='Q6_K', attn_output='Q8_0'),
]
# variant 'tcap': ThinkingCap's formats (the ThinkingCap module, kernels/k3_q35t_ops.h): every one-round-trip format it
# reads in some block, Q8_0 on the rows tile, an IQ2_S embedding, a Q8_0 head, BF16 alpha / beta
FORMATS_TCAP = [
    dict(ffn_gate='IQ4_XS', ffn_up='IQ3_XXS', ffn_down='IQ3_S', attn_qkv='IQ2_S', attn_gate='Q5_K', ssm_out='Q6_K'),
    dict(ffn_gate='IQ2_S', ffn_up='IQ4_XS', ffn_down='Q5_K', attn_qkv='IQ3_S', attn_gate='IQ3_XXS', ssm_out='IQ4_XS'),
    dict(ffn_gate='Q6_K', ffn_up='Q5_K', ffn_down='IQ2_S', attn_qkv='Q8_0', attn_gate='IQ4_XS', ssm_out='IQ3_XXS'),
    dict(ffn_gate='IQ3_S', ffn_up='IQ3_S', ffn_down='IQ4_XS', attn_q='IQ4_XS', attn_k='IQ2_S', attn_v='Q8_0', attn_output='IQ3_S'),
]
# variant 'gsq': the Qwen3.8-27B GSQ-RCO files' formats (the GSQ module, kernels/k3_q35t_ops.h under K3_Q35_GSQ): its five
# formats ThinkingCap's files do not use (IQ2_XS, Q2_K, IQ1_M, IQ2_XXS, Q4_K) beside IQ3_XXS / IQ3_S / IQ2_S / IQ4_XS, an IQ2_S
# embedding, a Q4_K head, BF16 alpha / beta
FORMATS_GSQ = (
    dict(ffn_gate='IQ2_XS', ffn_up='IQ2_XXS', ffn_down='Q2_K', attn_qkv='IQ4_XS', attn_gate='Q2_K', ssm_out='IQ3_S'),
    dict(ffn_gate='IQ1_M', ffn_up='IQ3_XXS', ffn_down='IQ2_XS', attn_qkv='Q2_K', attn_gate='IQ4_XS', ssm_out='Q4_K'),
    dict(ffn_gate='Q4_K', ffn_up='IQ2_S', ffn_down='IQ2_XXS', attn_qkv='IQ2_XS', attn_gate='IQ3_S', ssm_out='IQ1_M'),
    dict(ffn_gate='IQ3_S', ffn_up='Q2_K', ffn_down='IQ4_XS', attn_q='Q2_K', attn_k='IQ2_XS', attn_v='IQ2_XXS', attn_output='Q4_K'),
)
# the codebook formats: block bytes and a d giving typical |weight| 0.03 .. 0.1 (d times the largest scale x grid entry)
IQ_BLOCK = {'IQ4_XS': (136, 1e-4), 'IQ3_XXS': (98, 1.5e-4), 'IQ3_S': (110, 1.5e-4), 'IQ2_S': (82, 4e-4), 'IQ2_XS': (74, 4e-4),
            'IQ2_XXS': (66, 4e-4)}
U32, I32, F32V, BOOL, STR, ARR = 4, 5, 6, 7, 8, 9


def _s(x):
    b = x.encode(); return struct.pack('<Q', len(b)) + b


def _val(v):
    if isinstance(v, bool): return struct.pack('<I?', BOOL, v)
    if isinstance(v, int): return struct.pack('<II', U32, v)
    if isinstance(v, float): return struct.pack('<If', F32V, v)
    if isinstance(v, str): return struct.pack('<I', STR) + _s(v)
    et = BOOL if isinstance(v[0], bool) else I32
    return struct.pack('<IIQ', ARR, et, len(v)) + b''.join(struct.pack('<?' if et == BOOL else '<i', e) for e in v)


def _f16(rng, shape, base):
    """fp16 bit patterns of base * U(0.7, 1.3): normal fp16 numbers (base >= 1e-4)"""
    return (base * rng.uniform(0.7, 1.3, shape)).astype(np.float16).view(np.uint16)


def quant(rng, fmt, rows, K):
    """random valid blocks of a [rows, K] matrix in format fmt: uint8 [rows * row bytes] (typical |weight| 0.03 .. 0.1)"""
    rnd = lambda *s: rng.integers(0, 256, s, dtype=np.uint8)
    if fmt == 'Q8_0':
        nb = K // 32; b = np.zeros((rows, nb, 34), np.uint8)
        b[:, :, 0:2] = _f16(rng, (rows, nb), 1e-3).view(np.uint8).reshape(rows, nb, 2); b[:, :, 2:] = rnd(rows, nb, 32)
        return b.reshape(-1)
    nb = K // 256
    if fmt == 'Q4_K':      # d, dmin, scales[12], qs[128]: w = d * sc * q - dmin * m
        b = rnd(rows, nb, 144)
        b[:, :, 0:2] = _f16(rng, (rows, nb), 1.5e-4).view(np.uint8).reshape(rows, nb, 2)
        b[:, :, 2:4] = _f16(rng, (rows, nb), 1.5e-4 * 7.5).view(np.uint8).reshape(rows, nb, 2)
    elif fmt == 'Q5_K':    # d, dmin, scales[12], qh[32], qs[128]
        b = rnd(rows, nb, 176)
        b[:, :, 0:2] = _f16(rng, (rows, nb), 1e-4).view(np.uint8).reshape(rows, nb, 2)
        b[:, :, 2:4] = _f16(rng, (rows, nb), 1e-4 * 15.5).view(np.uint8).reshape(rows, nb, 2)
    elif fmt == 'Q6_K':    # ql[128], qh[64], int8 scales[16], d: w = d * scale * (q - 32)
        b = rnd(rows, nb, 210)
        b[:, :, 208:210] = _f16(rng, (rows, nb), 1e-4).view(np.uint8).reshape(rows, nb, 2)
    elif fmt == 'Q2_K':    # scales[16], qs[64], d, dmin: w = d * sc * q - dmin * m (sc, m <= 15, q <= 3)
        b = rnd(rows, nb, 84)
        b[:, :, 80:82] = _f16(rng, (rows, nb), 1.5e-3).view(np.uint8).reshape(rows, nb, 2)
        b[:, :, 82:84] = _f16(rng, (rows, nb), 1.5e-3).view(np.uint8).reshape(rows, nb, 2)
    elif fmt == 'IQ1_M':   # qs[32], qh[16], 4 scale words: d split over their top nibbles; w = d (2 s + 1) (g +- 1/8)
        b = rnd(rows, nb, 56)
        d = _f16(rng, (rows, nb), 4e-3)
        w = b[:, :, 48:56].copy().view(np.uint16).reshape(rows, nb, 4) & np.uint16(0x0FFF)
        for k in range(4): w[..., k] |= ((d >> np.uint16(4 * k)) & np.uint16(0xF)) << np.uint16(12)
        b[:, :, 48:56] = w.view(np.uint8).reshape(rows, nb, 8)
    elif fmt in IQ_BLOCK:  # d first, then any bytes (every index, sign and scale pattern is a valid block)
        bs, sc = IQ_BLOCK[fmt]
        b = rnd(rows, nb, bs)
        b[:, :, 0:2] = _f16(rng, (rows, nb), sc).view(np.uint8).reshape(rows, nb, 2)
    else: raise ValueError(fmt)
    return b.reshape(-1)


def write(path, seed=5, variant='swift', tp=False, mtp=False):
    """variant 'swift': Swift's formats (Q4_K / Q6_K / Q8_0 / Q5_K, F32 alpha / beta); 'tcap': ThinkingCap's. tp: shapes the
    two-card split cuts on block boundaries (engine/q35_dense_tp.py): 4 key heads / 8 value heads (half a card's value
    heads' columns = 512), 12 query heads / 2 kv heads, a 256-token vocabulary (128 head rows per card). mtp (the 'tcap'
    variant): its multi-token-prediction block blk.<blocks> in Q6_K (ThinkingCap's: engine/q35_mtp.py, k3_q35_tcap_mtp)"""
    rng = np.random.default_rng(seed)
    NK, NV, H, HKV, V = (4, 8, 12, 2, 256) if tp else (globals()['NK'], globals()['NV'], globals()['H'], globals()['HKV'], globals()['V'])
    CH = 2 * NK * S + NV * S
    tc = variant in ('tcap', 'gsq')   # (the ThinkingCap module family: IQ2_S embedding, BF16 alpha / beta)
    FMTS = FORMATS_GSQ if variant == 'gsq' else FORMATS_TCAP if tc else FORMATS
    f32 = lambda *shape, sc=0.3: (rng.standard_normal(shape) * sc).astype(np.float32)
    T = {}   # name -> (format, bytes, [rows, K] or [n])

    def mat(name, fmt, rows, K): T[name] = (fmt, quant(rng, fmt, rows, K), (rows, K))

    def vec(name, a): T[name] = ('F32', np.ascontiguousarray(a, np.float32).view(np.uint8).reshape(-1), a.shape)

    def bvec(name, a):   # (BF16: the float32's high half, truncated)
        T[name] = ('BF16', (np.ascontiguousarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16).view(np.uint8).reshape(-1), a.shape)

    mat('token_embd.weight', 'IQ2_S' if tc else 'Q4_K', V, D); vec('output_norm.weight', 1 + f32(D, sc=0.1))
    mat('output.weight', 'Q4_K' if variant == 'gsq' else 'Q8_0' if tc else 'Q6_K', V, D)
    for il, (gdn, fm) in enumerate(zip(KINDS, FMTS)):
        p = f'blk.{il}.'
        vec(p + 'attn_norm.weight', 1 + f32(D, sc=0.1)); vec(p + 'post_attention_norm.weight', 1 + f32(D, sc=0.1))
        mat(p + 'ffn_gate.weight', fm['ffn_gate'], FF, D); mat(p + 'ffn_up.weight', fm['ffn_up'], FF, D)
        mat(p + 'ffn_down.weight', fm['ffn_down'], D, FF)
        if gdn:
            mat(p + 'attn_qkv.weight', fm['attn_qkv'], CH, D); mat(p + 'attn_gate.weight', fm['attn_gate'], NV * S, D)
            mat(p + 'ssm_out.weight', fm['ssm_out'], D, NV * S)
            vec(p + 'ssm_conv1d.weight', f32(CH, DC, sc=0.5)); vec(p + 'ssm_dt.bias', f32(NV)); vec(p + 'ssm_a', -np.abs(f32(NV)))
            (bvec if tc else vec)(p + 'ssm_alpha.weight', f32(NV, D, sc=0.05)); (bvec if tc else vec)(p + 'ssm_beta.weight', f32(NV, D, sc=0.05))
            vec(p + 'ssm_norm.weight', 1 + f32(S, sc=0.1))
        else:
            mat(p + 'attn_q.weight', fm['attn_q'], H * 2 * HD, D); mat(p + 'attn_k.weight', fm['attn_k'], HKV * HD, D)
            mat(p + 'attn_v.weight', fm['attn_v'], HKV * HD, D); mat(p + 'attn_output.weight', fm['attn_output'], D, H * HD)
            vec(p + 'attn_q_norm.weight', 1 + f32(HD, sc=0.1)); vec(p + 'attn_k_norm.weight', 1 + f32(HD, sc=0.1))
    if mtp:
        if not tc: raise ValueError('tiny_q35.write: mtp=True writes ThinkingCap\'s Q6_K block (variant tcap)')
        p = f'blk.{len(KINDS)}.'
        mat(p + 'nextn.eh_proj.weight', 'Q6_K', D, 2 * D)
        mat(p + 'attn_q.weight', 'Q6_K', H * 2 * HD, D); mat(p + 'attn_k.weight', 'Q6_K', HKV * HD, D)
        mat(p + 'attn_v.weight', 'Q6_K', HKV * HD, D); mat(p + 'attn_output.weight', 'Q6_K', D, H * HD)
        mat(p + 'ffn_gate.weight', 'Q6_K', FF, D); mat(p + 'ffn_up.weight', 'Q6_K', FF, D); mat(p + 'ffn_down.weight', 'Q6_K', D, FF)
        for n_ in ('nextn.enorm', 'nextn.hnorm', 'nextn.shared_head_norm', 'attn_norm', 'post_attention_norm'):
            vec(p + n_ + '.weight', 1 + f32(D, sc=0.1))
        vec(p + 'attn_q_norm.weight', 1 + f32(HD, sc=0.1)); vec(p + 'attn_k_norm.weight', 1 + f32(HD, sc=0.1))
    meta = {'general.architecture': 'qwen35', 'qwen35.block_count': len(KINDS) + 1, 'qwen35.nextn_predict_layers': 1,
            'qwen35.context_length': 4096, 'qwen35.embedding_length': D, 'qwen35.feed_forward_length': FF,
            'qwen35.attention.head_count': H, 'qwen35.attention.head_count_kv': HKV, 'qwen35.attention.key_length': HD,
            'qwen35.attention.value_length': HD, 'qwen35.attention.layer_norm_rms_epsilon': 1e-6, 'qwen35.rope.freq_base': 1e7,
            'qwen35.rope.dimension_count': NROT, 'qwen35.rope.dimension_sections': [16, 16, 0, 0],
            'qwen35.ssm.conv_kernel': DC, 'qwen35.ssm.group_count': NK, 'qwen35.ssm.inner_size': NV * S,
            'qwen35.ssm.state_size': S, 'qwen35.ssm.time_step_rank': NV, 'qwen35.full_attention_interval': 4,
            'qwen35.attention.recurrent_layers': KINDS + [False]}
    head = struct.pack('<4sIQQ', b'GGUF', 3, len(T), len(meta)) + b''.join(_s(k) + _val(v) for k, v in meta.items())
    info, data = b'', b''
    for name, (fmt, raw, shape) in T.items():
        dims = shape[::-1]
        info += _s(name) + struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<IQ', TYPE_ID[fmt], len(data))
        data += raw.tobytes(); data += b'\0' * (-len(data) % 32)
    blob = head + info
    with open(path, 'wb') as f:
        f.write(blob + b'\0' * (-len(blob) % 32) + data)
    return dict(path=path, D=D, V=V, blocks=len(KINDS), kinds=list(KINDS), formats=FMTS, NK=NK, NV=NV, H=H, HKV=HKV)
