"""A tiny synthetic qwen3moe GGUF with quantized matrices, written here (no model file): the shapes the own-kernel engine
reads (caminus.engine.q3moe_one: residual 256, head dim 128, 8 query heads per kv head as OP_OWN_ATTS takes them) with
16 query / 2 kv heads (the two-card split's per-card attention), 8 experts of FFN 768 (= 3 x 256: the 512 / 256 cut of
the two-card split falls on whole blocks), top-2, two layers. The formats are the served file's: Q4_K everywhere, Q6_K
for attn_v and ffn_down in layer 0 (Q4_K in layer 1), a Q4_K embedding and a Q6_K head, F32 router and norms. The
quantized blocks are random valid blocks with small scales (tiny_q35.quant): caminus.ref_q3moe decodes the same bytes,
so the reference and the engine see one model. A few MiB.
q40=True: the Q4_0 family's formats (the GPTQ-Int4 checkpoint converted: tools/eval/gptq_to_gguf.py): every attention and
expert matrix Q4_0 (one scale per 128 values), the Q4_K embedding and the Q6_K head as above.
tp=True: the two-card engine's shapes (caminus.engine.q3moe_tp, the served split's per card): 32 query / 4 kv heads (2 kv
heads and their 16 query heads per card, as the served file's 4 kv heads split), and a vocabulary of 384 rows (the head's
cut at the bandwidth share falls at 256 / 128 rows: unequal halves).

  write(path, seed=7, n_ctx=4096, tp=False, q40=False) -> dict(path, D, V, layers, H, HKV, HD, E, KU, FF)"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tiny_q35 import TYPE_ID, _s, _val, quant   # noqa: E402 (the block writers and the GGUF header helpers)

D, V, H, HKV, HD, E, KU, FF, NL = 256, 256, 16, 2, 128, 8, 2, 768, 2
TP_SHAPES = dict(H=32, HKV=4, V=384)   # (tp=True: two kv heads per card)
# per layer: attn_v's and ffn_down's format (the served file alternates them between Q6_K and Q4_K)
FORMATS = [dict(attn_v='Q6_K', ffn_down='Q6_K'), dict(attn_v='Q4_K', ffn_down='Q4_K')]


def write(path, seed=7, n_ctx=4096, tp=False, q40=False):
    """the file at path; n_ctx: its context_length (the most max_ctx an engine on it takes); tp: the two-card shapes"""
    H, HKV, V = (TP_SHAPES['H'], TP_SHAPES['HKV'], TP_SHAPES['V']) if tp else (globals()['H'], globals()['HKV'], globals()['V'])
    rng = np.random.default_rng(seed)
    f32 = lambda *shape, sc=0.3: (rng.standard_normal(shape) * sc).astype(np.float32)
    T = {}   # name -> (format, bytes, dims innermost first)

    def mat(name, fmt, rows, K, n=1):   # n > 1: n matrices of [rows, K] (the experts: dims K, rows, n)
        if q40 and name.startswith('blk.'): fmt = 'Q4_0'
        T[name] = (fmt, quant(rng, fmt, rows * n, K), (K, rows) + ((n,) if n > 1 else ()))

    def vec(name, a):
        a = np.ascontiguousarray(a, np.float32); T[name] = ('F32', a.view(np.uint8).reshape(-1), a.shape[::-1])

    mat('token_embd.weight', 'Q4_K', V, D); vec('output_norm.weight', 1 + f32(D, sc=0.1))
    mat('output.weight', 'Q6_K', V, D)
    for il, fm in enumerate(FORMATS[:NL]):
        p = f'blk.{il}.'
        vec(p + 'attn_norm.weight', 1 + f32(D, sc=0.1)); vec(p + 'ffn_norm.weight', 1 + f32(D, sc=0.1))
        mat(p + 'attn_q.weight', 'Q4_K', H * HD, D); mat(p + 'attn_k.weight', 'Q4_K', HKV * HD, D)
        mat(p + 'attn_v.weight', fm['attn_v'], HKV * HD, D); mat(p + 'attn_output.weight', 'Q4_K', D, H * HD)
        vec(p + 'attn_q_norm.weight', 1 + f32(HD, sc=0.1)); vec(p + 'attn_k_norm.weight', 1 + f32(HD, sc=0.1))
        vec(p + 'ffn_gate_inp.weight', f32(E, D, sc=0.2))
        mat(p + 'ffn_gate_exps.weight', 'Q4_K', FF, D, E); mat(p + 'ffn_up_exps.weight', 'Q4_K', FF, D, E)
        mat(p + 'ffn_down_exps.weight', fm['ffn_down'], D, FF, E)
    meta = {'general.architecture': 'qwen3moe', 'qwen3moe.block_count': NL, 'qwen3moe.context_length': int(n_ctx),
            'qwen3moe.embedding_length': D, 'qwen3moe.feed_forward_length': 4 * D,
            'qwen3moe.attention.head_count': H, 'qwen3moe.attention.head_count_kv': HKV,
            'qwen3moe.attention.key_length': HD, 'qwen3moe.attention.value_length': HD,
            'qwen3moe.attention.layer_norm_rms_epsilon': 1e-6, 'qwen3moe.rope.freq_base': 1e6,
            'qwen3moe.expert_count': E, 'qwen3moe.expert_used_count': KU, 'qwen3moe.expert_feed_forward_length': FF}
    head = struct.pack('<4sIQQ', b'GGUF', 3, len(T), len(meta)) + b''.join(_s(k) + _val(v) for k, v in meta.items())
    info, data = b'', b''
    for name, (fmt, raw, dims) in T.items():
        info += _s(name) + struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<IQ', TYPE_ID[fmt], len(data))
        data += raw.tobytes(); data += b'\0' * (-len(data) % 32)
    blob = head + info
    with open(path, 'wb') as f:
        f.write(blob + b'\0' * (-len(blob) % 32) + data)
    return dict(path=path, D=D, V=V, layers=NL, H=H, HKV=HKV, HD=HD, E=E, KU=KU, FF=FF)
