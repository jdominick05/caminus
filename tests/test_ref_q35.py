"""caminus.ref_q35 (the float64 reference of the dense qwen35 line) on a tiny synthetic qwen35 GGUF written here (random F32
weights; 4 trunk blocks: 3 Gated DeltaNet + 1 attention, and a declared multi-token-prediction block whose tensors are not
in the file), CPU only, no model file:
  - the state carries: a text fed in one chunk, token by token, and in uneven pieces gives the same logits (the conv window,
    the delta state, the KV cache and the causal mask across feeds)
  - causality: a position's logits do not depend on later tokens; reset() returns to the start
  - the block kinds come from attention.recurrent_layers, and from full_attention_interval without it (the same logits)
  - the declared multi-token-prediction block is not read (the file has no tensors for it), and a qwen35 file with a
    router tensor (the MoE line) is refused
  - one Gated DeltaNet step and one attention step against the block's definition written out per head and per value here
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/test_ref_q35.py --scratch DIR"""
import argparse, os, struct, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
from caminus import ref_q35

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); A = ap.parse_args()
os.makedirs(A.scratch, exist_ok=True)
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


# ---- a minimal GGUF v3 writer (F32 tensors; the value types this file needs) -------------------------------------------
U32, I32, F32, BOOL, STR, ARR = 4, 5, 6, 7, 8, 9


def _s(x):
    b = x.encode(); return struct.pack('<Q', len(b)) + b


def _val(v):
    if isinstance(v, bool): return struct.pack('<I?', BOOL, v)
    if isinstance(v, int): return struct.pack('<II', U32, v)
    if isinstance(v, float): return struct.pack('<If', F32, v)
    if isinstance(v, str): return struct.pack('<I', STR) + _s(v)
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


D, FF, V, S, NK, NV, H, HKV, HD, NROT, DC = 32, 40, 50, 4, 2, 4, 4, 2, 8, 4, 4
CH = 2 * NK * S + NV * S
KINDS = [True, True, False, True]                 # blocks 0, 1, 3 Gated DeltaNet; block 2 attention
rng = np.random.default_rng(3)
w = lambda *shape, sc=0.3: (rng.standard_normal(shape) * sc).astype(np.float32)
T = {'token_embd.weight': w(V, D, sc=1.0), 'output_norm.weight': 1 + w(D, sc=0.1), 'output.weight': w(V, D)}
for il, gdn in enumerate(KINDS):
    p = f'blk.{il}.'
    T.update({p + 'attn_norm.weight': 1 + w(D, sc=0.1), p + 'post_attention_norm.weight': 1 + w(D, sc=0.1),
              p + 'ffn_gate.weight': w(FF, D), p + 'ffn_up.weight': w(FF, D), p + 'ffn_down.weight': w(D, FF)})
    if gdn:
        T.update({p + 'attn_qkv.weight': w(CH, D), p + 'attn_gate.weight': w(NV * S, D), p + 'ssm_conv1d.weight': w(CH, DC, sc=0.5),
                  p + 'ssm_dt.bias': w(NV), p + 'ssm_a': -np.abs(w(NV)), p + 'ssm_beta.weight': w(NV, D), p + 'ssm_alpha.weight': w(NV, D),
                  p + 'ssm_norm.weight': 1 + w(S, sc=0.1), p + 'ssm_out.weight': w(D, NV * S)})
    else:
        T.update({p + 'attn_q.weight': w(H * 2 * HD, D), p + 'attn_k.weight': w(HKV * HD, D), p + 'attn_v.weight': w(HKV * HD, D),
                  p + 'attn_output.weight': w(D, H * HD), p + 'attn_q_norm.weight': 1 + w(HD, sc=0.1), p + 'attn_k_norm.weight': 1 + w(HD, sc=0.1)})
META = {'general.architecture': 'qwen35', 'qwen35.block_count': len(KINDS) + 1, 'qwen35.nextn_predict_layers': 1,
        'qwen35.embedding_length': D, 'qwen35.feed_forward_length': FF, 'qwen35.attention.head_count': H,
        'qwen35.attention.head_count_kv': HKV, 'qwen35.attention.key_length': HD, 'qwen35.attention.value_length': HD,
        'qwen35.attention.layer_norm_rms_epsilon': 1e-6, 'qwen35.rope.freq_base': 1e7, 'qwen35.rope.dimension_count': NROT,
        'qwen35.rope.dimension_sections': [1, 1, 0, 0], 'qwen35.ssm.conv_kernel': DC, 'qwen35.ssm.group_count': NK,
        'qwen35.ssm.inner_size': NV * S, 'qwen35.ssm.state_size': S, 'qwen35.ssm.time_step_rank': NV,
        'qwen35.full_attention_interval': 3, 'qwen35.attention.recurrent_layers': KINDS + [False]}
P = os.path.join(A.scratch, 'tiny-q35.gguf'); write_gguf(P, META, T)
m = ref_q35.Q35Ref(P, threads=2)
check(m.L == len(KINDS) and m.gdn == KINDS, f'the trunk is {m.L} blocks of the declared {len(KINDS) + 1} (the multi-token-prediction block is not read), kinds {m.gdn}')

ids = [int(t) for t in rng.integers(0, V, 13)]
rel = lambda a, b: float(np.abs(a - b).max() / np.abs(b).max())
tr = []; full = m.feed(ids, trace=tr)
check(full.shape == (len(ids), V) and len(tr) == m.L and np.isfinite(full).all(), f'feed: logits {full.shape}, one residual per block, finite')
m.reset(); one = np.concatenate([m.feed([t]) for t in ids])
m.reset(); cuts = [0, 1, 4, 5, 11, 13]; pieces = np.concatenate([m.feed(ids[a:b]) for a, b in zip(cuts, cuts[1:])])
check(rel(one, full) < 1e-9 and rel(pieces, full) < 1e-9, f'one chunk == token by token == uneven pieces (rel {rel(one, full):.1e}, {rel(pieces, full):.1e})')
m.reset(); again = m.feed(ids); check(np.array_equal(again, full), 'reset() returns to the start (the same bits)')
m.reset(); other = m.feed(ids[:7] + [(t + 1) % V for t in ids[7:]])
check(rel(other[:7], full[:7]) < 1e-12 and rel(other[7:], full[7:]) > 1e-3, 'a position does not see later tokens, and sees its own')
m.reset(); last = m.feed(ids, last=3); check(last.shape == (3, V) and rel(last, full[-3:]) < 1e-12, 'last=3 gives the last three rows')

meta2 = {k: v for k, v in META.items() if k != 'qwen35.attention.recurrent_layers'}
P2 = os.path.join(A.scratch, 'tiny-q35-fai.gguf'); write_gguf(P2, meta2, T)
m2 = ref_q35.Q35Ref(P2, threads=2)
check(m2.gdn == KINDS and np.array_equal(m2.feed(ids), full), 'without recurrent_layers the kinds come from full_attention_interval (the same logits)')
P3 = os.path.join(A.scratch, 'tiny-q35-moe.gguf'); write_gguf(P3, META, dict(T, **{'blk.0.ffn_gate_inp.weight': w(4, D)}))
try: ref_q35.Q35Ref(P3); refused = False
except ValueError: refused = True
check(refused, 'a qwen35 file with a router tensor is refused')

# ---- the two block kinds against their definitions, one token at a time, per head and per value -------------------------
eps = float(np.float32(1e-6))     # (the file holds the epsilon as a float32; the reference reads that value)
silu = lambda y: y / (1 + np.exp(-y)); sig = lambda y: 1 / (1 + np.exp(-y))
rms = lambda y, g: y / np.sqrt(np.mean(y * y) + eps) * g
Wd = lambda n: T[n].astype(np.float64)
m.reset(); hs = rng.standard_normal((5, D))
got = m._gdn(0, hs)
want = np.empty_like(got); hist = [np.zeros(CH)] * (DC - 1); St = np.zeros((NV, S, S))
for n in range(5):
    h = hs[n]; mix = Wd('blk.0.attn_qkv.weight') @ h; win = hist + [mix]; hist = win[1:]
    c = silu(np.array([sum(win[j][ch] * Wd('blk.0.ssm_conv1d.weight')[ch, j] for j in range(DC)) for ch in range(CH)]))
    beta = sig(Wd('blk.0.ssm_beta.weight') @ h)
    g = np.log1p(np.exp(Wd('blk.0.ssm_alpha.weight') @ h + Wd('blk.0.ssm_dt.bias'))) * Wd('blk.0.ssm_a')
    z = Wd('blk.0.attn_gate.weight') @ h; fin = np.empty(NV * S)
    for j in range(NV):
        kh = j % NK
        q = c[kh * S:(kh + 1) * S]; k = c[NK * S + kh * S:NK * S + (kh + 1) * S]; v = c[2 * NK * S + j * S:2 * NK * S + (j + 1) * S]
        q = q / np.sqrt(np.sum(q * q) + eps) / np.sqrt(S); k = k / np.sqrt(np.sum(k * k) + eps)
        St[j] = St[j] * np.exp(g[j])
        St[j] = St[j] + np.outer((v - St[j] @ k) * beta[j], k)
        fin[j * S:(j + 1) * S] = rms(St[j] @ q, Wd('blk.0.ssm_norm.weight')) * silu(z[j * S:(j + 1) * S])
    want[n] = Wd('blk.0.ssm_out.weight') @ fin
check(rel(got, want) < 1e-12, f'a Gated DeltaNet block over 5 tokens == its definition per value head (rel {rel(got, want):.1e})')

m.reset(); pos = np.arange(5, dtype=np.float64)
got = m._attn(2, hs, pos)
want = np.empty_like(got); Ks, Vs = [], []


def rope(y, p):
    out = y.copy()
    for i in range(NROT // 2):
        th = p * 1e7 ** (-2.0 * i / NROT); a, b = y[i], y[i + NROT // 2]
        out[i] = a * np.cos(th) - b * np.sin(th); out[i + NROT // 2] = a * np.sin(th) + b * np.cos(th)
    return out


for n in range(5):
    h = hs[n]; qf = (Wd('blk.2.attn_q.weight') @ h).reshape(H, 2 * HD)
    kk = (Wd('blk.2.attn_k.weight') @ h).reshape(HKV, HD); vv = (Wd('blk.2.attn_v.weight') @ h).reshape(HKV, HD)
    Ks.append([rope(rms(kk[x], Wd('blk.2.attn_k_norm.weight')), n) for x in range(HKV)]); Vs.append(vv)
    out = np.empty((H, HD))
    for hh in range(H):
        q = rope(rms(qf[hh, :HD], Wd('blk.2.attn_q_norm.weight')), n); kv = hh // (H // HKV)
        sc = np.array([Ks[t][kv] @ q / np.sqrt(HD) for t in range(n + 1)]); pr = np.exp(sc - sc.max()); pr /= pr.sum()
        out[hh] = sum(pr[t] * Vs[t][kv] for t in range(n + 1)) * sig(qf[hh, HD:])
    want[n] = Wd('blk.2.attn_output.weight') @ out.reshape(-1)
check(rel(got, want) < 1e-12, f'an attention block over 5 tokens == its definition per head (rel {rel(got, want):.1e})')
print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
