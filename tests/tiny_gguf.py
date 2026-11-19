"""Tiny synthetic GGUF files for the CPU tests (tests/test_arch.py, tests/test_ref_q35.py, tests/test_ref_coder.py), and the
Gated DeltaNet block's definition written out per token and per value head (the float64 references' tests check
caminus.ref_q35's _gdn against it). Not a test itself: the tests import it (python -I keeps the script's directory out of
sys.path, so each adds tests/ first)."""
import struct
import numpy as np

U32, I32, F32, BOOL, STR, ARR, U64 = 4, 5, 6, 7, 8, 9, 10


def _s(x):
    b = x.encode(); return struct.pack('<Q', len(b)) + b


def _val(v):
    """a metadata value: bool, int (u32), float (f32), str, a tuple (an array of u64) or a list (of i32, or of bool)"""
    if isinstance(v, bool): return struct.pack('<I?', BOOL, v)
    if isinstance(v, int): return struct.pack('<II', U32, v)
    if isinstance(v, float): return struct.pack('<If', F32, v)
    if isinstance(v, str): return struct.pack('<I', STR) + _s(v)
    if isinstance(v, tuple): return struct.pack('<IIQ', ARR, U64, len(v)) + b''.join(struct.pack('<Q', e) for e in v)
    et = BOOL if isinstance(v[0], bool) else I32
    return struct.pack('<IIQ', ARR, et, len(v)) + b''.join(struct.pack('<?' if et == BOOL else '<i', e) for e in v)


def write_gguf(path, meta, tensors):
    """a GGUF v3 file: meta {key: value (see _val)}, tensors {name: float32 array} (written F32, dims innermost first: an
    array [rows, K] is a K x rows tensor), data aligned to 32 bytes"""
    head = struct.pack('<4sIQQ', b'GGUF', 3, len(tensors), len(meta)) + b''.join(_s(k) + _val(v) for k, v in meta.items())
    info, data = b'', b''
    for name, a in tensors.items():
        a = np.ascontiguousarray(a, np.float32); dims = a.shape[::-1]
        info += _s(name) + struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<IQ', 0, len(data))
        data += a.tobytes(); data += b'\0' * (-len(data) % 32)
    blob = head + info
    with open(path, 'wb') as f:
        f.write(blob + b'\0' * (-len(blob) % 32) + data)


def gdn_definition(Wd, p, hs, NK, NV, S, DC, eps, gate):
    """the Gated DeltaNet block (tensors p + '...') over the tokens hs [N, D] from a fresh state, one token, value head and
    conv tap at a time (Wd(name): the float64 weights; gate: the output gate on z, silu for qwen35 / sigmoid for the Coder)"""
    silu = lambda y: y / (1 + np.exp(-y)); sig = lambda y: 1 / (1 + np.exp(-y))
    CH = 2 * NK * S + NV * S
    out = []; hist = [np.zeros(CH)] * (DC - 1); St = np.zeros((NV, S, S))
    for h in hs:
        mix = Wd(p + 'attn_qkv.weight') @ h; win = hist + [mix]; hist = win[1:]
        c = silu(np.array([sum(win[j][ch] * Wd(p + 'ssm_conv1d.weight')[ch, j] for j in range(DC)) for ch in range(CH)]))
        beta = sig(Wd(p + 'ssm_beta.weight') @ h)
        g = np.log1p(np.exp(Wd(p + 'ssm_alpha.weight') @ h + Wd(p + 'ssm_dt.bias'))) * Wd(p + 'ssm_a')
        z = Wd(p + 'attn_gate.weight') @ h; fin = np.empty(NV * S)
        for j in range(NV):
            kh = j % NK
            q = c[kh * S:(kh + 1) * S]; k = c[NK * S + kh * S:NK * S + (kh + 1) * S]; v = c[2 * NK * S + j * S:2 * NK * S + (j + 1) * S]
            q = q / np.sqrt(np.sum(q * q) + eps) / np.sqrt(S); k = k / np.sqrt(np.sum(k * k) + eps)
            St[j] = St[j] * np.exp(g[j])
            St[j] = St[j] + np.outer((v - St[j] @ k) * beta[j], k)
            o = St[j] @ q
            fin[j * S:(j + 1) * S] = o / np.sqrt(np.mean(o * o) + eps) * Wd(p + 'ssm_norm.weight') * gate(z[j * S:(j + 1) * S])
        out.append(Wd(p + 'ssm_out.weight') @ fin)
    return np.array(out)
