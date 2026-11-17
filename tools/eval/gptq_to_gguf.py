"""Convert a qwen3moe GPTQ checkpoint (Hugging Face safetensors, 4 bits, symmetric, desc_act false, group size a multiple
of 32, checkpoint_format "gptq": qzeros store the zero point minus 1) to a GGUF with BASE's metadata and tensor table, every
GPTQ matrix as Q4_0, bit for bit the checkpoint's weights.

The mapping is exact: GPTQ decodes w = s_g (q - (z + 1)) with q the 4-bit code, s_g the fp16 scale of the input group g
and z the stored zero; symmetric checkpoints store z = 7 everywhere, so w = s_g (q - 8), Q4_0's w = d (q - 8) with
d = s_g repeated over the group's 32-value blocks and the codes copied unchanged. In fp32 both products are exact (an
fp16 scale times an integer in -8 .. 7), so the decoded weights are the same bits. Anything else is refused before a
byte is written: a zero word other than 0x77777777, a g_idx other than input // group, a non-finite scale, a shape
that is not BASE's.

BASE is a GGUF of the same model (the official Q4_K_M): the output takes its header and metadata bytes (the tokenizer, the
chat template, the hyperparameters; general.file_type keeps naming BASE's quantization, as tools/eval/gguf_splice.py's
study files do) and its tensor names, order and shapes. Matrices without GPTQ codes keep the checkpoint's fp16
(token_embd, output); norms and the router are widened to F32 (exact), the types BASE stores them in.

After writing, the output is opened again and checked against the checkpoint on the CPU: metadata BASE's; every tensor
decoded by caminus.own_dequant against the checkpoint decoded straight from its int32 words by the GPTQ formula above
(its g_idx and zeros, not the converter's unpacking), compared as fp32 bits: the norms, router, embedding and head
whole, attention whole in every layer, the experts whole in --full layers and --experts of them in every other layer.
Bit equality cannot see a transposed or swapped matrix that is the same bits, so each checked matrix is also compared
with BASE's own decoding of the same name: a relative difference above --max-rel (two quantizations of one model sit
well below it; a wrong matrix sits near 1.4) fails.
  python -I tools/eval/gptq_to_gguf.py GPTQ_DIR BASE OUT [--full 0,23,47] [--experts 0,1,64,127] [--max-rel 0.5]
GPTQ_DIR holds config.json and model.safetensors. Prints RESULT PASS | FAIL (exit 1). CPU only; a few GB of RAM."""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
from caminus import ggufio, own_dequant
from caminus.decode.dflash_q3m_torch import safetensors_header

ap = argparse.ArgumentParser(); ap.add_argument('gptq'); ap.add_argument('base'); ap.add_argument('out')
ap.add_argument('--full', default='0,23,47', help='layers whose every expert is checked')
ap.add_argument('--experts', default='0,1,64,127', help='experts checked in every other layer')
ap.add_argument('--max-rel', type=float, default=0.5, help='largest |w - base| / |base| a checked matrix may have')
A = ap.parse_args()
if os.path.exists(A.out): sys.exit(f'{A.out} exists')
cfg = json.load(open(os.path.join(A.gptq, 'config.json')))
qc = cfg.get('quantization_config') or {}
want = {'quant_method': 'gptq', 'bits': 4, 'sym': True, 'desc_act': False, 'checkpoint_format': 'gptq'}
if any(qc.get(k) != v for k, v in want.items()): sys.exit(f'quantization_config {qc}: needs {want}')
G = int(qc['group_size'])
if G % 32: sys.exit(f'group_size {G}: not a multiple of 32 (Q4_0 blocks)')
ST = os.path.join(A.gptq, 'model.safetensors')
H, BASE_OFF = safetensors_header(ST); H.pop('__metadata__', None)
FD = os.open(ST, os.O_RDONLY)
DT = {'F16': np.float16, 'F32': np.float32, 'BF16': np.uint16, 'I32': np.int32}
B = ggufio.GGUFFile(A.base)
if B.arch != 'qwen3moe': sys.exit(f'{A.base}: {B.arch}, not qwen3moe')
NE = int(B.hp('expert_count'))
TID = {name: tid for tid, (name, _, _) in ggufio.TYPES.items()}


def raw(name):
    """a checkpoint tensor as a numpy array in ordinary memory (pread, not a mapping)"""
    e = H[name]; a, b = e['data_offsets']
    buf = bytearray(b - a); mv = memoryview(buf); n = 0
    while n < b - a:
        k = os.preadv(FD, [mv[n:]], BASE_OFF + a + n)
        if k <= 0: sys.exit(f'{name}: short read')
        n += k
    os.posix_fadvise(FD, BASE_OFF + a, b - a, os.POSIX_FADV_DONTNEED)
    return np.frombuffer(buf, dtype=DT[e['dtype']]).reshape(e['shape'])


def hf_names(gname):
    """GGUF tensor name -> (kind, checkpoint prefix(es)): kind 'q' (GPTQ matrix), 'qx' (GPTQ, one per expert), 'f' (float)"""
    fixed = {'token_embd.weight': 'model.embed_tokens.weight', 'output.weight': 'lm_head.weight', 'output_norm.weight': 'model.norm.weight'}
    if gname in fixed: return 'f', fixed[gname]
    m = re.fullmatch(r'blk\.(\d+)\.(\w+)\.weight', gname)
    if not m: sys.exit(f'{gname}: no checkpoint name')
    L, k = f'model.layers.{m[1]}.', m[2]
    f = {'attn_norm': 'input_layernorm', 'ffn_norm': 'post_attention_layernorm', 'attn_q_norm': 'self_attn.q_norm',
         'attn_k_norm': 'self_attn.k_norm', 'ffn_gate_inp': 'mlp.gate'}
    if k in f: return 'f', L + f[k] + '.weight'
    q = {'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj', 'attn_v': 'self_attn.v_proj', 'attn_output': 'self_attn.o_proj'}
    if k in q: return 'q', L + q[k]
    x = {'ffn_gate_exps': 'gate_proj', 'ffn_up_exps': 'up_proj', 'ffn_down_exps': 'down_proj'}
    if k in x: return 'qx', [f'{L}mlp.experts.{e}.{x[k]}' for e in range(NE)]
    sys.exit(f'{gname}: no checkpoint name')


def refuse(p):
    """exit unless GPTQ matrix p is what the mapping assumes (the docstring's refusals; reads its small tensors only)"""
    K, N = H[p + '.qweight']['shape'][0] * 8, H[p + '.qweight']['shape'][1]
    sc, qz, gi = raw(p + '.scales'), raw(p + '.qzeros'), raw(p + '.g_idx')
    if not np.all(qz.view(np.uint32) == 0x77777777): sys.exit(f'{p}: qzeros not all 0x77777777 (zero 8): not a symmetric v1 checkpoint')
    if not np.array_equal(gi, np.arange(K, dtype=np.int32) // G): sys.exit(f'{p}: g_idx is not input // {G} (desc_act)')
    if sc.shape != (K // G, N) or not np.all(np.isfinite(sc)): sys.exit(f'{p}: scales {sc.shape} or not finite')


def q4_0(p):
    """GPTQ matrix p as Q4_0 rows: [N, K / 32, 18] uint8 (row n = output n, blocks along the input)"""
    qw, sc = raw(p + '.qweight'), raw(p + '.scales'); N = qw.shape[1]; K = qw.shape[0] * 8
    codes = ((qw.view(np.uint32)[:, None, :] >> (4 * np.arange(8, dtype=np.uint32))[None, :, None]) & 15).reshape(K, N)
    codes = codes.T.astype(np.uint8).reshape(N, K // 32, 32)      # [N, blocks, 32]: input k of block b = 32 b + k
    out = np.empty((N, K // 32, 18), np.uint8)
    out[:, :, 0:2] = np.repeat(sc.T, G // 32, axis=1).view(np.uint8).reshape(N, K // 32, 2)
    out[:, :, 2:] = codes[:, :, :16] | (codes[:, :, 16:] << 4)      # byte k: value k low, value 16 + k high
    return out


def float_bytes(name, tid):
    a = raw(name)
    if a.dtype != np.float16: sys.exit(f'{name}: {a.dtype} (fp16 expected)')
    return (a.astype(np.float32) if tid == TID['F32'] else a).reshape(-1)


entries, srcs = [], {}
for t in B.tensors:
    kind, p = hf_names(t.name)
    if kind == 'f':
        tid = TID['F32'] if t.type == 'F32' else TID['F16']
        shape = H[p]['shape']
        if list(reversed(shape)) != list(t.shape): sys.exit(f'{t.name}: checkpoint {shape}, BASE {list(t.shape)}')
        n = int(np.prod(shape)) * (4 if tid == TID['F32'] else 2)
        entries.append((t.name, t.shape, tid, n, (lambda p=p, tid=tid: iter([float_bytes(p, tid)]))))
    else:
        ps = [p] if kind == 'q' else p
        K, N = H[ps[0] + '.qweight']['shape'][0] * 8, H[ps[0] + '.qweight']['shape'][1]
        shape = (K, N) if kind == 'q' else (K, N, len(ps))
        if tuple(shape) != t.shape: sys.exit(f'{t.name}: checkpoint {shape}, BASE {list(t.shape)}')
        for x in ps: refuse(x)
        entries.append((t.name, t.shape, TID['Q4_0'], N * K // 32 * 18 * len(ps), (lambda ps=ps: (q4_0(x).reshape(-1) for x in ps))))
    srcs[t.name] = (kind, p)
print(f'{sum(len(p) if k == "qx" else 1 for k, p in srcs.values() if k != "f")} GPTQ matrices: zeros 8, g_idx input // {G}, scales finite', flush=True)
print(f'writing {len(entries)} tensors, {sum(e[3] for e in entries) / 1e9:.3f} GB', flush=True)
B.write_like(A.out, entries, drop_cache=True)

# -- the checks ---------------------------------------------------------------------------------------------------------
O = ggufio.GGUFFile(A.out); bad = 0; full = {int(x) for x in A.full.split(',') if x}; exps = [int(x) for x in A.experts.split(',') if x]


def gptq_dequant(p):
    """fp32 [N, K] of GPTQ matrix p by the checkpoint's own formula: w[n, k] = s[g_idx[k], n] (q[k, n] - (z[g_idx[k], n] + 1))"""
    qw, sc, qz, gi = raw(p + '.qweight'), raw(p + '.scales'), raw(p + '.qzeros'), raw(p + '.g_idx')
    K, N = qw.shape[0] * 8, qw.shape[1]; k = np.arange(K); n = np.arange(N)
    q = (qw.view(np.uint32)[k // 8, :] >> (4 * (k % 8)).astype(np.uint32)[:, None]) & 15                 # [K, N]
    z = (qz.view(np.uint32)[gi, :][:, n // 8] >> (4 * (n % 8)).astype(np.uint32)[None, :]) & 15          # [K, N]
    return (sc[gi, :].astype(np.float32) * (q.astype(np.float32) - (z.astype(np.float32) + 1))).T


def bits_eq(mine, ref):
    return mine.shape == ref.shape and np.array_equal(mine.view(np.uint32), ref.view(np.uint32))


def meta_eq(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) if isinstance(a[k], np.ndarray) else a[k] == b[k] for k in a)


if not meta_eq(O.meta, B.meta): bad += 1; print('FAIL the metadata differs from BASE')
if [(t.name, t.shape) for t in O.tensors] != [(t.name, t.shape) for t in B.tensors]: bad += 1; print('FAIL the tensor table differs from BASE')
nmat = nval = 0; worst = {}; CR = 8192   # (rows per check chunk: the embedding and the head decoded whole would take 4 GB)
for o in O.tensors:
    kind, p = srcs[o.name]; bt = B.T[o.name]; role = re.sub(r'^blk\.\d+\.', '', o.name)
    li = int(o.name.split('.')[1]) if o.name.startswith('blk.') else -1
    if kind == 'f':
        a = raw(p).astype(np.float32); a = a.reshape(-1, a.shape[-1]); R = a.shape[0]
        eq, d2, b2 = True, 0.0, 0.0
        for r0 in range(0, R, CR):
            r1 = min(R, r0 + CR); mine = own_dequant.rows(o, r0, r1); base = own_dequant.rows(bt, r0, r1)
            eq &= bits_eq(mine, a[r0:r1]); d2 += float(np.square(mine - base, dtype=np.float64).sum()); b2 += float(np.square(base, dtype=np.float64).sum())
        rel = (d2 / max(b2, 1e-300)) ** 0.5; todo = []; nval += o.n_values; worst[role] = max(worst.get(role, 0.0), rel)
        if not (eq and rel <= A.max_rel): bad += 1; print(f'FAIL {o.name}: bits equal {eq}, relative difference to BASE {rel:.4f}')
    elif kind == 'q':
        todo = [(0, p)]
    else:
        todo = [(e, p[e]) for e in (range(NE) if li in full else exps)]
    for e, pe in todo:
        N = o.shape[1]
        mine = own_dequant.rows(o, e * N, (e + 1) * N); base = own_dequant.rows(bt, e * N, (e + 1) * N)
        eq = bits_eq(mine, gptq_dequant(pe)); rel = float(np.linalg.norm(mine - base) / max(np.linalg.norm(base), 1e-30))
        nmat += 1; nval += mine.size; worst[role] = max(worst.get(role, 0.0), rel)
        if not (eq and rel <= A.max_rel): bad += 1; print(f'FAIL {o.name} expert {e}: bits equal {eq}, relative difference to BASE {rel:.4f}')
print(f'checked {nmat} GPTQ matrices + {sum(1 for o in O.tensors if srcs[o.name][0] == "f")} float tensors, {nval / 1e9:.2f} G values as fp32 bits; '
      f'{os.path.getsize(A.out) / 1e9:.3f} GB (BASE {os.path.getsize(A.base) / 1e9:.3f} GB)')
for role, r in sorted(worst.items()): print(f'  {role}: largest relative difference to BASE {r:.4f}')
print('RESULT', 'PASS' if bad == 0 else 'FAIL'); sys.exit(1 if bad else 0)
