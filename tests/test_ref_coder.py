"""caminus.ref_coder (the Coder's float64 reference) on a tiny synthetic qwen4exp GGUF written here (random F32 weights;
4 layers: Gated DeltaNet, Gated DeltaNet with the PLE embedding, Gated DeltaNet, attention; 4 hyper-connection streams; a
MoE FFN of 6 experts with 2 used and a gated shared expert; an indexer budget that makes dense attention exact up to 11
positions), CPU only, no model file:
  - the state carries: a text fed in one chunk, token by token and in uneven pieces gives the same logits (the conv
    window, the delta state, the KV cache, the PLE conv's 9 rows of history and the n-gram context across feeds)
  - causality: a position's logits do not depend on later tokens; reset() returns to the start
  - the same model written as two shards (the PLE table alone in the second, as the Coder's file) gives the same bits
  - the hyper-connection mix and combine, and the PLE parts over 6 tokens (an EOS among them: the n-gram cut), against
    their definitions written out per token, per stream and per tap here (the n-gram hash in Python integers)
  - dense attention is the model's only up to top_k + ratio - 1 positions: a feed past that is refused, one up to it is not
  - a qwen35 file and a file with two PLE layers are refused
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/test_ref_coder.py --scratch DIR"""
import argparse, os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
from caminus import ref_coder

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); A = ap.parse_args()
os.makedirs(A.scratch, exist_ok=True)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


# ---- a minimal GGUF v3 writer (F32 tensors; the value types this file needs) -------------------------------------------
U32, I32, F32, BOOL, STR, ARR, U64 = 4, 5, 6, 7, 8, 9, 10


def _s(x):
    b = x.encode(); return struct.pack('<Q', len(b)) + b


def _val(v):
    if isinstance(v, bool): return struct.pack('<I?', BOOL, v)
    if isinstance(v, int): return struct.pack('<II', U32, v)
    if isinstance(v, float): return struct.pack('<If', F32, v)
    if isinstance(v, str): return struct.pack('<I', STR) + _s(v)
    if isinstance(v, tuple):   # (a uint64 array)
        return struct.pack('<IIQ', ARR, U64, len(v)) + b''.join(struct.pack('<Q', e) for e in v)
    et = BOOL if isinstance(v[0], bool) else I32
    return struct.pack('<IIQ', ARR, et, len(v)) + b''.join(struct.pack('<?' if et == BOOL else '<i', e) for e in v)


def write_gguf(path, meta, tensors):
    """tensors: name -> float32 array [rows, K] (written with dims innermost first) or [n]"""
    head = struct.pack('<4sIQQ', b'GGUF', 3, len(tensors), len(meta)) + b''.join(_s(k) + _val(v) for k, v in meta.items())
    info, data = b'', b''
    for name, a in tensors.items():
        a = np.ascontiguousarray(a, np.float32); dims = a.shape[::-1]
        info += _s(name) + struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<IQ', 0, len(data))
        data += a.tobytes(); data += b'\0' * (-len(data) % 32)
    blob = head + info
    with open(path, 'wb') as f:
        f.write(blob + b'\0' * (-len(blob) % 32) + data)


D, V, S, NK, NV, H, HKV, HD, NROT, DC = 16, 50, 4, 2, 4, 4, 2, 8, 4, 4
HC, HLR, E, KU, FF = 4, 8, 6, 2, 8
NG, PG, PDIM, PK, EOS = 3, 2, 4, 4, 7                 # n-gram size, heads per n-gram, PLE head dim, conv taps, PLE EOS
NH = (NG - 1) * PG                                    # PLE heads (4 x 4 = D)
MULT, VOC = (23703573157769, 20109073645365, 8052911324071), (11, 13, 17, 19)
OFF = tuple(int(v) for v in np.cumsum((0,) + VOC[:-1]))
CH = 2 * NK * S + NV * S
KINDS = [True, True, True, False]                     # layer 3 attention
rng = np.random.default_rng(5)
w = lambda *shape, sc=0.3: (rng.standard_normal(shape) * sc).astype(np.float32)
T = {'token_embd.weight': w(V, D, sc=1.0), 'output.weight': w(V, D), 'output_hc_norm.weight': 1 + w(HC * D, sc=0.1),
     'output_hc_down.weight': w(HLR, HC * D), 'output_hc_up.weight': w(HC * D, HLR),
     'per_layer_token_embd.weight': w(sum(VOC), PDIM, sc=1.0)}
for il, gdn in enumerate(KINDS):
    p = f'blk.{il}.'
    for k in ('attn', 'ffn'):
        T.update({p + f'hc_{k}_norm.weight': 1 + w(HC * D, sc=0.1), p + f'hc_{k}_down.weight': w(HLR, HC * D),
                  p + f'hc_{k}_up.weight': w(HC * D, HLR), p + f'hc_{k}_inject.weight': w(HC, HC * D)})
    T.update({p + 'ffn_gate_inp.weight': w(E, D), p + 'ffn_gate_exps.weight': w(E * FF, D), p + 'ffn_up_exps.weight': w(E * FF, D),
              p + 'ffn_down_exps.weight': w(E * D, FF), p + 'ffn_gate_inp_shexp.weight': w(D), p + 'ffn_gate_shexp.weight': w(FF, D),
              p + 'ffn_up_shexp.weight': w(FF, D), p + 'ffn_down_shexp.weight': w(D, FF)})
    if gdn:
        T.update({p + 'attn_qkv.weight': w(CH, D), p + 'attn_gate.weight': w(NV * S, D), p + 'ssm_conv1d.weight': w(CH, DC, sc=0.5),
                  p + 'ssm_dt.bias': w(NV), p + 'ssm_a': -np.abs(w(NV)), p + 'ssm_beta.weight': w(NV, D), p + 'ssm_alpha.weight': w(NV, D),
                  p + 'ssm_norm.weight': 1 + w(S, sc=0.1), p + 'ssm_out.weight': w(D, NV * S)})
    else:
        T.update({p + 'attn_q.weight': w(H * 2 * HD, D), p + 'attn_k.weight': w(HKV * HD, D), p + 'attn_v.weight': w(HKV * HD, D),
                  p + 'attn_output.weight': w(D, H * HD), p + 'attn_q_norm.weight': 1 + w(HD, sc=0.1), p + 'attn_k_norm.weight': 1 + w(HD, sc=0.1)})
T.update({'blk.1.ple_key.weight': w(HC * D, D), 'blk.1.ple_value.weight': w(D, D), 'blk.1.ple_norm_key.weight': 1 + w(HC * D, sc=0.1),
          'blk.1.ple_norm_query.weight': 1 + w(HC * D, sc=0.1), 'blk.1.ple_norm_conv.weight': 1 + w(HC * D, sc=0.1),
          'blk.1.ple_conv1d.weight': w(HC * D, PK, sc=0.5)})
A_ = 'qwen4exp.'
META = {'general.architecture': 'qwen4exp', A_ + 'block_count': len(KINDS), A_ + 'embedding_length': D,
        A_ + 'attention.head_count': H, A_ + 'attention.head_count_kv': HKV, A_ + 'attention.key_length': HD,
        A_ + 'attention.value_length': HD, A_ + 'attention.layer_norm_rms_epsilon': 1e-6, A_ + 'rope.freq_base': 1e7,
        A_ + 'rope.dimension_count': NROT, A_ + 'rope.dimension_sections': [1, 1, 0, 0], A_ + 'expert_count': E,
        A_ + 'expert_used_count': KU, A_ + 'expert_feed_forward_length': FF, A_ + 'expert_shared_feed_forward_length': FF,
        A_ + 'ssm.conv_kernel': DC, A_ + 'ssm.state_size': S, A_ + 'ssm.group_count': NK, A_ + 'ssm.time_step_rank': NV,
        A_ + 'ssm.inner_size': NV * S, A_ + 'full_attention_interval': 4, A_ + 'hyper_connection.count': HC,
        A_ + 'hyper_connection.low_rank': HLR, A_ + 'attention.indexer.head_count': 2, A_ + 'attention.indexer.key_length': 8,
        A_ + 'attention.indexer.top_k': 8, A_ + 'attention.compress_ratios': [0, 0, 0, 4], A_ + 'ple.layers': [1],
        A_ + 'ple.ngram_size': NG, A_ + 'ple.heads_per_ngram': PG, A_ + 'ple.conv_kernel': PK, A_ + 'ple.eos_token_id': EOS,
        A_ + 'embedding_length_per_layer_input': PDIM, A_ + 'ple.layer_multipliers': MULT, A_ + 'ple.head_offsets': OFF,
        A_ + 'ple.head_vocab_sizes': VOC}
P = os.path.join(A.scratch, 'tiny-coder.gguf'); write_gguf(P, META, T)
m = ref_coder.CoderRef(P, threads=2)
check(m.L == 4 and m.gdn == KINDS and m.ple_layers == [1] and m.limit == 11 and m.HC == HC,
      f'header: {m.L} layers, kinds {m.gdn}, PLE layers {m.ple_layers}, dense-exact limit {m.limit} (top_k 8 + ratio 4 - 1)')

ids = [3, 9, EOS, 12, 30, 9, 3, 44, 2, 17, 5]           # 11 positions: the limit; an EOS at position 2 (the n-gram cut)
rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
tr, ro = [], []; full = m.feed(ids, trace=tr, routes=ro)
check(full.shape == (len(ids), V) and len(tr) == m.L and tr[0].shape == (len(ids), HC, D) and len(ro) == m.L and np.isfinite(full).all(),
      f'feed: logits {full.shape}, the streams [N, {HC}, D] and the routes of each layer, finite')
m.reset(); one = np.concatenate([m.feed([t]) for t in ids])
m.reset(); cuts = [0, 1, 4, 5, 10, 11]; pieces = np.concatenate([m.feed(ids[a:b]) for a, b in zip(cuts, cuts[1:])])
check(rel(one, full) < 1e-9 and rel(pieces, full) < 1e-9, f'one chunk == token by token == uneven pieces (rel {rel(one, full):.1e}, {rel(pieces, full):.1e})')
m.reset(); again = m.feed(ids); check(np.array_equal(again, full), 'reset() returns to the start (the same bits)')
m.reset(); other = m.feed(ids[:6] + [(t + 1) % V for t in ids[6:]])
check(rel(other[:6], full[:6]) < 1e-12 and rel(other[6:], full[6:]) > 1e-3, 'a position does not see later tokens, and sees its own')
m.reset(); last = m.feed(ids, last=3); check(last.shape == (3, V) and rel(last, full[-3:]) < 1e-12, 'last=3 gives the last three rows')

# ---- two shards: the PLE table alone in the second ----
P1, P2 = (os.path.join(A.scratch, f'tiny-coder-0000{k}-of-00002.gguf') for k in (1, 2))
tab = 'per_layer_token_embd.weight'
write_gguf(P1, dict(META, **{'split.no': 0, 'split.count': 2, 'split.tensors.count': len(T)}), {k: v for k, v in T.items() if k != tab})
write_gguf(P2, {'split.no': 1, 'split.count': 2, 'split.tensors.count': len(T)}, {tab: T[tab]})
ms = ref_coder.CoderRef(P1, threads=2)
check(np.array_equal(ms.feed(ids), full), 'the same model as two shards (the PLE table in the second, read with pread): the same bits')
try: ref_coder.CoderRef(P2); refused = False
except ValueError: refused = True
check(refused, 'the second shard given as the model is refused (give the first)')

# ---- the hyper-connection mix / combine and the PLE parts against their definitions --------------------------------------
eps = float(np.float32(1e-6))
silu = lambda y: y / (1 + np.exp(-y)); sig = lambda y: 1 / (1 + np.exp(-y))
Wd = lambda n: T[n].astype(np.float64)
x = rng.standard_normal((6, HC, D)); blk = rng.standard_normal((6, D))
got, gi = m.hc_mix(x, 'blk.2.hc_ffn')
want, wi = np.empty((6, D)), np.empty((6, HC))
for n in range(6):
    xn = np.stack([x[n, s] / np.sqrt(np.mean(x[n, s] ** 2) + eps) * Wd('blk.2.hc_ffn_norm.weight')[s * D:(s + 1) * D] for s in range(HC)])
    f = xn.reshape(-1); lo = silu(Wd('blk.2.hc_ffn_down.weight') @ f / HC); g = Wd('blk.2.hc_ffn_up.weight') @ lo
    want[n] = sum(xn[s] * sig(g[s * D:(s + 1) * D]) for s in range(HC)) / HC; wi[n] = Wd('blk.2.hc_ffn_inject.weight') @ f
check(rel(got, want) < 1e-12 and rel(gi, wi) < 1e-12, f'the hyper-connection mix == its definition per token and stream ({rel(got, want):.1e}, {rel(gi, wi):.1e})')
comb = m.hc_combine(x, blk, gi)
wc = np.stack([[x[n, s] + blk[n] * 2 * sig(gi[n, s] / HC) for s in range(HC)] for n in range(6)])
check(rel(comb, wc) < 1e-14, f'the combine == streams + block * 2 sigmoid(inject / {HC}) ({rel(comb, wc):.1e})')

seq = [3, 9, EOS, 12, 30, 9]
m.reset(); m.toks = list(seq); gv, co = m.ple_parts(1, x, 0)
hist = [np.zeros(HC * D)] * ((PK - 1) * NG); wgv, wco = np.empty_like(gv), np.empty_like(co)
for n in range(6):
    ctx, cut = [seq[n]], False
    for s in range(1, NG):
        t = seq[n - s] if n - s >= 0 else -1
        cut = cut or t < 0 or t == EOS; ctx.append(EOS if cut else t)
    rows = []
    for g in range(2, NG + 1):
        mixed = 0
        for j in range(g): mixed ^= (ctx[j] * MULT[j]) % (1 << 64)
        rows += [mixed % VOC[(g - 2) * PG + k] + OFF[(g - 2) * PG + k] for k in range(PG)]
    emb = np.concatenate([Wd(tab)[r] for r in rows])
    key = Wd('blk.1.ple_key.weight') @ emb; val = Wd('blk.1.ple_value.weight') @ emb
    gn = lambda y, wn: np.concatenate([y[s * D:(s + 1) * D] / np.sqrt(np.mean(y[s * D:(s + 1) * D] ** 2) + eps) for s in range(HC)]) * Wd(wn)
    kq = gn(key, 'blk.1.ple_norm_key.weight') * gn(x[n].reshape(-1), 'blk.1.ple_norm_query.weight')
    gate = [sig(np.sign(v) * np.sqrt(max(abs(v), 1e-6))) for v in (kq[s * D:(s + 1) * D].sum() / np.sqrt(D) for s in range(HC))]
    gated = np.concatenate([val * gate[s] for s in range(HC)]); c = gn(gated, 'blk.1.ple_norm_conv.weight')
    hist = hist + [c]
    conv = sum(Wd('blk.1.ple_conv1d.weight')[:, k] * hist[len(hist) - 1 - (PK - 1 - k) * NG] for k in range(PK))
    wgv[n] = gated.reshape(HC, D); wco[n] = silu(conv).reshape(HC, D)
check(rel(gv, wgv) < 1e-12 and rel(co, wco) < 1e-12,
      f'the PLE gated value and conv (taps {NG} apart, an EOS cut) == their definitions per token ({rel(gv, wgv):.1e}, {rel(co, wco):.1e})')

# ---- the dense-exact limit and the refusals --------------------------------------------------------------------------
m.reset(); m.feed(ids)
try: m.feed([4]); refused = False
except ValueError as e: refused = 'depth reference' in str(e)
check(refused, f'position {m.limit} (past top_k + ratio - 1) is refused, naming the depth reference; positions 0 .. {m.limit - 1} were not')
P3 = os.path.join(A.scratch, 'tiny-qwen35.gguf'); write_gguf(P3, dict(META, **{'general.architecture': 'qwen35'}), T)
P4 = os.path.join(A.scratch, 'tiny-coder-2ple.gguf'); write_gguf(P4, dict(META, **{A_ + 'ple.layers': [1, 2]}), T)
for path, what in ((P3, 'a qwen35 file'), (P4, 'a file with two PLE layers')):
    try: ref_coder.CoderRef(path); refused = False
    except ValueError: refused = True
    check(refused, f'{what} is refused')
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
