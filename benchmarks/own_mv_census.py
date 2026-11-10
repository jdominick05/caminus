"""Where a Qwen3-30B-A3B decode token's matvec time goes, per shape, on one card: our own kernels vs llama.cpp's vs a plain
read of the same bytes (docs/specs/spec-own-quant.md; the decode-floor paper's token: 48 layers x (q, k, v, o, router,
8 x gate, 8 x up, 8 x down) + the vocab head = 1,393 matvecs). The GGUF is read with caminus.ggufio (our reader).

Per distinct (role, format, rows, K):
  accuracy  256 sampled rows of layer 0's tensor (expert 0 for the experts): |y - y_ref| / rms(y_ref), y_ref the fp64 dot
            of gguf-py's dequantized rows (the test oracle) with the fp32 activation (heavy-tailed synthetic vectors)
  speed     cold: every arm cycles through all 48 layers' copies (experts: 8 random experts per layer, rotated each
            round), so nothing stays in the Infinity Cache; one launch per matvec; median per-matvec time over --rounds
            arms: read (own_read of the same bytes: the bandwidth ceiling), llama (gg_mv_*, q8_1 activations),
                  own (own_mv_q4_k_1rt / own_mv_q6_k_pl: fp16 activations; Q6_K in our plane layout)
Then the token: the sum over the 1,393 matvecs per arm (launch gaps included as measured: one launch each), and the
bytes / bandwidth floor. The router (F32, 128 x 2048) is listed apart (no quantized kernel).
  tools/guard.sh LOG 1800 python -I benchmarks/own_mv_census.py GGUF [--rounds 5] [--layers 48]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, ctypes, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus import ggufio
from caminus.device import capacity
from caminus.device.hip import Module
from caminus.engine import gguf as G            # (llama.cpp's gg_mv_* launchers: the yardstick arm; puts gguf-py on the path)
from caminus.engine.own_layout import q6k_planes
from gguf import quants, GGMLQuantizationType   # (the test oracle only)

ap = argparse.ArgumentParser()
ap.add_argument('gguf')
ap.add_argument('--rounds', type=int, default=5)
ap.add_argument('--layers', type=int, default=48)
ap.add_argument('--roles', default=None, help='comma-separated roles (default: all), e.g. ffn_gate_exps,ffn_down_exps')
ap.add_argument('--log', default=None, help='append the result lines to this file too')
A = ap.parse_args()
dev = torch.device('cuda:0')
OWN = Module('own_mv')
rng = np.random.default_rng(0)
F = ggufio.GGUFFile(A.gguf)
NL = min(A.layers, int(F.hp('block_count')))
KU, NE = int(F.hp('expert_used_count')), int(F.hp('expert_count'))
OUT = []


def say(s):
    print(s, flush=True); OUT.append(s)


def mats(role):
    """[(tensor, expert or None)] of one role over the layers (experts: all of them; the timing picks 8 per layer)"""
    if role == 'head': return [(F.T['output.weight'], None)]
    return [(F.T[f'blk.{il}.{role}.weight'], None) for il in range(NL)]


def shape_of(t):
    """(rows per matrix, K, experts or 1)"""
    if len(t.shape) == 3: return int(t.shape[1]), int(t.shape[0]), int(t.shape[2])
    return int(t.shape[1]), int(t.shape[0]), 1


def slice_bytes(t, e, rows, K):
    n = t.n_bytes // (shape_of(t)[2])
    return t.data()[e * n:(e + 1) * n] if e is not None else t.data()


def own_kernel(fmt, K):
    nm = {'Q4_K': f'own_mv_q4_k_1rt_k{K}', 'Q6_K': f'own_mv_q6_k_pl_k{K}', 'IQ4_XS': f'own_mv_iq4xs_1rt_k{K}_r2',
          'IQ3_XXS': f'own_mv_iq3xxs_1rt_k{K}_r2', 'IQ3_S': f'own_mv_iq3s_1rt_k{K}_r2',
          'IQ2_XXS': f'own_mv_iq2xxs_1rt_k{K}_r2'}.get(fmt)
    return getattr(OWN, nm) if nm and hasattr(OWN, nm) else None


def run_read(w, out_sink):
    n16 = w.numel() // 16
    OWN.own_read((min(4096, (n16 + 255) // 256),), (256,), w, ctypes.c_long(n16), out_sink)


def to_dev(raw, fmt, rows, K, own):
    b = np.ascontiguousarray(raw)
    if own and fmt == 'Q6_K': b = q6k_planes(b, rows, K)
    return torch.from_numpy(b.copy()).to(dev)


ROLES = ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps', 'head']
if A.roles: ROLES = [r for r in ROLES if r in A.roles.split(',')]
PER_TOKEN = {'attn_q': NL, 'attn_k': NL, 'attn_v': NL, 'attn_output': NL, 'ffn_gate_exps': NL * KU, 'ffn_up_exps': NL * KU,
             'ffn_down_exps': NL * KU, 'head': 1}
say(f'{torch.cuda.get_device_name(0)}  {os.path.basename(A.gguf)}  layers {NL}, experts {KU} of {NE}')
sink = torch.zeros(1, dtype=torch.int32, device=dev)
token = {'read': 0.0, 'llama': 0.0, 'own': 0.0}; token_bytes = 0; missing = []
for role in ROLES:
    groups = {}
    for t, _ in mats(role):
        rows, K, ne = shape_of(t)
        groups.setdefault((t.type, rows, K, ne), []).append(t)
    for (fmt, rows, K, ne), ts in groups.items():
        rb = ts[0].n_bytes // ne // rows
        per = PER_TOKEN[role] * len(ts) // sum(len(v) for v in groups.values())   # this group's share of the role's matvecs
        mat_bytes = rows * rb
        token_bytes += per * mat_bytes
        kown = own_kernel(fmt, K)
        qt = GGMLQuantizationType[fmt]
        has_llama = qt in G.MV
        # copies for the cold rotation: one matrix per layer (experts: 8 per layer, rotated)
        sel = [(t, (rng.choice(ne, KU, replace=False) if ne > 1 else [None])) for t in ts]
        copies = [(t, e) for t, es in sel for e in es][:max(1, min(len(ts) * (KU if ne > 1 else 1), 400))]
        need = 2 * len(copies) * mat_bytes / 2 ** 20 + 64
        capacity.check([0], [need], f'census {role} {fmt}')
        Wl = [to_dev(slice_bytes(t, e, rows, K), fmt, rows, K, False) for t, e in copies]
        Wo = [to_dev(slice_bytes(t, e, rows, K), fmt, rows, K, True) for t, e in copies] if kown else []
        v = rng.standard_t(3, K); v[rng.choice(K, 8, replace=False)] *= 20; v = v / np.sqrt((v ** 2).mean())
        x32 = torch.tensor(v, dtype=torch.float32, device=dev); x16 = x32.half(); yq = G.quantize_q8_1(x32)
        out = torch.zeros(rows, dtype=torch.float32, device=dev)
        # accuracy on the first copy's sampled rows
        t0, e0 = copies[0]
        srows = np.sort(rng.choice(rows, min(256, rows), replace=False))
        raw = np.asarray(slice_bytes(t0, e0, rows, K)).reshape(rows, rb)[srows]
        W64 = quants.dequantize(raw, qt).astype(np.float64).reshape(len(srows), K)
        yref = W64 @ x32.double().cpu().numpy(); sc = np.sqrt((yref ** 2).mean())
        acc = {}
        if has_llama:
            G._m('k3_gguf_q35h' if qt in G.MV_Q35H else 'k3_gguf_gsq' if qt in G.MV_GSQ else 'k3_gguf')
            ml = G._m('k3_gguf_q35h' if qt in G.MV_Q35H else 'k3_gguf_gsq' if qt in G.MV_GSQ else 'k3_gguf')
            kll = getattr(ml, G.MV[qt])
            run_llama = lambda w: kll(((rows + 3) // 4,), (32, 4), w, yq, out, ctypes.c_int(rows), ctypes.c_int(K))
            out.zero_(); run_llama(Wl[0]); torch.cuda.synchronize()
            e = np.abs(out.double().cpu().numpy()[srows] - yref) / sc; acc['llama'] = (e.mean(), np.percentile(e, 99))
        if kown:
            run_own = lambda w: kown(((rows + 15) // 16,), (256,), w, ctypes.c_int(rows), x16, out)
            out.zero_(); run_own(Wo[0]); torch.cuda.synchronize()
            e = np.abs(out.double().cpu().numpy()[srows] - yref) / sc; acc['own'] = (e.mean(), np.percentile(e, 99))
        arms = {'read': (Wl, lambda w: run_read(w, sink))}
        if has_llama: arms['llama'] = (Wl, run_llama)
        if kown: arms['own'] = (Wo, run_own)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        tm = {k: [] for k in arms}
        for r in range(A.rounds + 1):
            for k, (Ws, fn) in arms.items():
                torch.cuda.synchronize(); ev[0].record()
                for w in Ws: fn(w)
                ev[1].record(); torch.cuda.synchronize()
                if r: tm[k].append(ev[0].elapsed_time(ev[1]) * 1e3 / len(Ws))   # us per matvec
        line = f'{role:14s} {fmt:5s} [{rows:6d} x {K:5d}] {mat_bytes / 2**20:7.3f} MiB x {per:5d}/token |'
        for k in ('read', 'llama', 'own'):
            if k in tm:
                us = float(np.median(tm[k])); token[k] += per * us
                line += f' {k} {us:8.2f} us {mat_bytes / us / 1e3:6.1f} GB/s |'
            else:
                line += f' {k} {"-":>8s}    {"":>6s}      |'; token[k] = float('nan') if k != 'read' else token[k]
                if k == 'own': missing.append(f'{role} {fmt} K {K}')
        for k, (m, p) in acc.items(): line += f' err {k} {m:.1e}/{p:.1e}'
        say(line)
        del Wl, Wo; torch.cuda.empty_cache()
say(f'token (1,393 matvecs, one launch each): read {token["read"] / 1e3:.3f} ms | llama {token["llama"] / 1e3:.3f} ms | '
    f'own {token["own"] / 1e3:.3f} ms;  bytes per token {token_bytes / 2**30:.3f} GiB (router excluded)')
if missing: say('no own kernel yet for: ' + '; '.join(missing))
if A.log:
    with open(A.log, 'a') as f: f.write('\n'.join(OUT) + '\n')
print('RESULT DONE')
