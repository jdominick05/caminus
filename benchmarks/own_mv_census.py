"""Where a decode token's matvec time goes, per shape, on one card: our own kernels vs llama.cpp's vs a plain read of the
same bytes (docs/specs/spec-own-quant.md). Qwen3-30B-A3B (qwen3moe): the decode-floor paper's token, 48 layers x (q, k,
v, o, router, 8 x gate, 8 x up, 8 x down) + the vocab head = 1,393 matvecs. The dense qwen35 line (Qwen3.8-27B: Swift,
ThinkingCap): every trunk tensor once per token (the 48 Gated DeltaNet layers' qkv, gate and out, the 16 attention
layers' q, k, v and output, 64 x the feed-forward's gate, up and down, the head; the multi-token-prediction block and the
F32 / BF16 alpha / beta excluded), on kernels/bench_own_mv_q35.hip's tiles in K3's form. The GGUF is read with caminus.ggufio
(our reader).

Per distinct (role, format, rows, K):
  accuracy  256 sampled rows of layer 0's tensor (expert 0 for the experts): |y - y_ref| / rms(y_ref), y_ref the fp64 dot
            of the rows our decoders dequantize (caminus.own_dequant, bitwise gguf-py's) with the fp32 activation (heavy-tailed synthetic vectors)
  speed     cold: every arm cycles through all 48 layers' copies (experts: 8 random experts per layer, rotated each
            round), so nothing stays in the Infinity Cache; one launch per matvec; median per-matvec time over --rounds
            arms: read (own_read of the same bytes: the bandwidth ceiling), llama (gg_mv_*, q8_1 activations),
                  own (own_mv_q4_k_1rt / own_mv_q6_k_pl: fp16 activations; Q6_K in our plane layout)
Then the token: the sum over the 1,393 matvecs per arm (launch gaps included as measured: one launch each), and the
bytes / bandwidth floor. The router (F32, 128 x 2048) is listed apart (no quantized kernel).
--q2_0 (the dense line): also the own Q2_0 tile (kernels/bench_own_mv_q2_0.hip own_mv_q2_0_k3), which no dense decode
module holds, so STANDALONE only: at every distinct matrix shape of the file, on synthetic Q2_0 blocks of that shape
(random fp16 scales and 2-bit codes: no file of the line is Q2_0), cold over as many copies as the file reads that shape
a token (at least 512 MiB in all), against own_read of the same bytes; error vs fp64 as above. Not in the token's sums.
--formats Q2_0 --q2_0 times it alone.
--min-set-mib N: every group's rotation padded with clones of its copies to at least N MiB (a group of a few small
matrices otherwise fits the Infinity Cache and times warm); each line prints its set's size.
  tools/guard.sh LOG 1800 python -I benchmarks/own_mv_census.py GGUF [--rounds 5] [--layers N] [--q2_0]
New kernel code: run it on a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>)."""
import argparse, ctypes, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus import ggufio
from caminus.device import capacity
from caminus.device.hip import Module
from caminus.engine import gguf as G            # (llama.cpp's gg_mv_* launchers: the yardstick arm; puts gguf-py on the path)
from caminus.engine.own_layout import q6k_planes, iq_records, IQ_REC, q2k_records, iq1m_records
from caminus.engine.q35_file import CENSUS_DENSE_ROLES, mat_tensors   # (the dense line's header->role map, from engine/q35_file.py)
from caminus import own_dequant                 # (the float64 oracle: our decoders, bitwise gguf-py's: tests/io/test_own_dequant.py)

ap = argparse.ArgumentParser()
ap.add_argument('gguf')
ap.add_argument('--rounds', type=int, default=5)
ap.add_argument('--layers', type=int, default=None, help='the first N layers (default: all)')
ap.add_argument('--roles', default=None, help='comma-separated roles (default: all), e.g. ffn_gate_exps,ffn_down_exps')
ap.add_argument('--q5k', choices=('1rt', 'general'), default='1rt', help='dense: the Q5_K tile (own_rows_q5k_1rt or the general rows tile)')
ap.add_argument('--q8', choices=('gen', 'u2', 'u4', 'u8'), default='gen', help='dense: the Q8_0 tile (own_rows_q8_0, or own_rows_q8_0_1rt with UP 2 / 4 / 8 block steps per pass; the 1rt arms are checked bitwise against the general tile)')
ap.add_argument('--formats', default=None, help='comma-separated formats to time (default: all)')
ap.add_argument('--log', default=None, help='append the result lines to this file too')
ap.add_argument('--min-set-mib', type=int, default=0, help='pad each group\'s cold rotation with clones to at least this many MiB '
                '(default 0: one copy per tensor, as before; a small group otherwise reads from the Infinity Cache)')
ap.add_argument('--q2_0', action='store_true', help='dense: also the standalone own Q2_0 tile on synthetic blocks at each shape')
A = ap.parse_args()
dev = torch.device('cuda:0')
OWN = Module('bench_own_mv')
rng = np.random.default_rng(0)
F = ggufio.GGUFFile(A.gguf)
NL = min(A.layers or 10 ** 6, int(F.hp('block_count')))
DENSE = F.meta.get('general.architecture') == 'qwen35' and F.hp('expert_count') is None
KU, NE = (1, 1) if DENSE else (int(F.hp('expert_used_count')), int(F.hp('expert_count')))
if DENSE: NL = min(A.layers or 10 ** 6, int(F.hp('block_count')) - int(F.hp('nextn_predict_layers', 0)))   # (the trunk)
OWNQ = Module('bench_own_mv_q35') if DENSE else None
GEN = ('Q8_0',) if A.q5k == '1rt' else ('Q5_K', 'Q8_0')   # (the general rows tiles: 4 rows per wave)
OUT = []
Q8_SAME = []


def say(s):
    print(s, flush=True); OUT.append(s)


def mats(role):
    """[(tensor, expert or None)] of one role over the layers (experts: all of them; the timing picks 8 per layer)"""
    if DENSE: return mat_tensors(F, role)     # (the shared header map: engine/q35_file.py mat_tensors)
    if role == 'head': return [(F.T['output.weight'], None)]
    return [(F.T[f'blk.{il}.{role}.weight'], None) for il in range(NL) if f'blk.{il}.{role}.weight' in F.T]


def shape_of(t):
    """(rows per matrix, K, experts or 1)"""
    if len(t.shape) == 3: return int(t.shape[1]), int(t.shape[0]), int(t.shape[2])
    return int(t.shape[1]), int(t.shape[0]), 1


def slice_bytes(t, e, rows, K):
    n = t.n_bytes // (shape_of(t)[2])
    return t.data()[e * n:(e + 1) * n] if e is not None else t.data()


def own_kernel(fmt, K):
    if DENSE:   # (K3's form: runtime K; Q5_K / Q8_0 the general rows tiles OP_OWN_MV16 runs, 4 rows per wave)
        nm = {'Q4_K': 'own_mv_q4_k_k3', 'IQ4_XS': 'own_mv_iq4xs_k3', 'IQ3_XXS': 'own_mv_iq3xxs_k3', 'IQ3_S': 'own_mv_iq3s_k3', 'IQ2_S': 'own_mv_iq2s_k3', 'IQ2_XS': 'own_mv_iq2xs_k3', 'IQ2_XXS': 'own_mv_iq2xxs_k3', 'Q2_K': 'own_mv_q2_k_k3', 'IQ1_M': 'own_mv_iq1m_k3',
              'Q6_K': 'own_mv_q6_k_pl_k3', 'Q5_K': 'own_mv_q5_k_k3' if A.q5k == '1rt' else 'own_mv_q5_k_a16',
              'Q8_0': 'own_mv_q8_0_a16' if A.q8 == 'gen' else f'own_mv_q8_0_1rt_{A.q8}'}.get(fmt)
        return getattr(OWNQ, nm) if nm else None
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
    if own and fmt in IQ_REC: b = iq_records(b, rows, K, fmt)
    if own and fmt == 'Q2_K': b = q2k_records(b, rows, K)
    if own and fmt == 'IQ1_M': b = iq1m_records(b, rows, K)
    return torch.from_numpy(b.copy()).to(dev)


ROLES = (list(CENSUS_DENSE_ROLES) if DENSE else
         ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps', 'head'])
if A.roles: ROLES = [r for r in ROLES if r in A.roles.split(',')]
PER_TOKEN = ({r: len(mats(r)) for r in ROLES} if DENSE else
             {'attn_q': NL, 'attn_k': NL, 'attn_v': NL, 'attn_output': NL, 'ffn_gate_exps': NL * KU, 'ffn_up_exps': NL * KU,
              'ffn_down_exps': NL * KU, 'head': 1})
say(f'{torch.cuda.get_device_name(0)}  {os.path.basename(A.gguf)}  layers {NL}, ' + ('dense' if DENSE else f'experts {KU} of {NE}'))
sink = torch.zeros(1, dtype=torch.int32, device=dev)
token = {'read': 0.0, 'llama': 0.0, 'own': 0.0}; token_bytes = 0; missing = []
for role in ROLES:
    groups = {}
    for t, _ in mats(role):
        rows, K, ne = shape_of(t)
        groups.setdefault((t.type, rows, K, ne), []).append(t)
    for (fmt, rows, K, ne), ts in groups.items():
        if A.formats and fmt not in A.formats.split(','): continue
        rb = ts[0].n_bytes // ne // rows
        per = PER_TOKEN[role] * len(ts) // sum(len(v) for v in groups.values())   # this group's share of the role's matvecs
        mat_bytes = rows * rb
        token_bytes += per * mat_bytes
        kown = own_kernel(fmt, K)
        qt = G.QT[fmt]                             # (the llama.cpp arm's key only)
        has_llama = qt in G.MV
        # copies for the cold rotation: one matrix per layer (experts: 8 per layer, rotated)
        sel = [(t, (rng.choice(ne, KU, replace=False) if ne > 1 else [None])) for t in ts]
        copies = [(t, e) for t, es in sel for e in es][:max(1, min(len(ts) * (KU if ne > 1 else 1), 400))]
        nset = min(400, max(len(copies), -(-A.min_set_mib * 2 ** 20 // mat_bytes)))   # (--min-set-mib: clones past the Infinity Cache)
        need = 2 * nset * mat_bytes / 2 ** 20 + 64
        capacity.check([0], [need], f'census {role} {fmt}')
        Wl = [to_dev(slice_bytes(t, e, rows, K), fmt, rows, K, False) for t, e in copies]
        Wo = [to_dev(slice_bytes(t, e, rows, K), fmt, rows, K, True) for t, e in copies] if kown else []
        n0 = len(Wl)   # (the padding: the same bytes at other addresses, so every timed read comes from memory)
        Wl += [Wl[i % n0].clone() for i in range(n0, nset)]
        if Wo: Wo += [Wo[i % n0].clone() for i in range(n0, nset)]
        v = rng.standard_t(3, K); v[rng.choice(K, 8, replace=False)] *= 20; v = v / np.sqrt((v ** 2).mean())
        x32 = torch.tensor(v, dtype=torch.float32, device=dev); x16 = x32.half(); yq = G.quantize_q8_1(x32)
        out = torch.zeros(rows, dtype=torch.float32, device=dev)
        # accuracy on the first copy's sampled rows
        t0, e0 = copies[0]
        srows = np.sort(rng.choice(rows, min(256, rows), replace=False))
        raw = np.asarray(slice_bytes(t0, e0, rows, K)).reshape(rows, rb)[srows]
        W64 = own_dequant.dequant(np.ascontiguousarray(raw).reshape(-1), fmt, len(srows) * K).astype(np.float64).reshape(len(srows), K)
        yref = W64 @ x32.double().cpu().numpy(); sc = np.sqrt((yref ** 2).mean())
        acc = {}
        if has_llama:
            G._m('dec_q35m_hh' if qt in G.MV_Q35H else 'dec_q4x_gsq' if qt in G.MV_GSQ else 'dec_q4x_gg')
            ml = G._m('dec_q35m_hh' if qt in G.MV_Q35H else 'dec_q4x_gsq' if qt in G.MV_GSQ else 'dec_q4x_gg')
            kll = getattr(ml, G.MV[qt])
            run_llama = lambda w: kll(((rows + 3) // 4,), (32, 4), w, yq, out, ctypes.c_int(rows), ctypes.c_int(K))
            out.zero_(); run_llama(Wl[0]); torch.cuda.synchronize()
            e = np.abs(out.double().cpu().numpy()[srows] - yref) / sc; acc['llama'] = (e.mean(), np.percentile(e, 99))
        if kown:
            if not DENSE: run_own = lambda w: kown(((rows + 15) // 16,), (256,), w, ctypes.c_int(rows), x16, out)
            else: run_own = lambda w: kown(((rows + (31 if fmt in GEN else 15)) // (32 if fmt in GEN else 16),),
                                           (256,), w, ctypes.c_int(rows), ctypes.c_int(K), x16, out)
            out.zero_(); run_own(Wo[0]); torch.cuda.synchronize()
            e = np.abs(out.double().cpu().numpy()[srows] - yref) / sc; acc['own'] = (e.mean(), np.percentile(e, 99))
            if DENSE and fmt == 'Q8_0' and A.q8 != 'gen':   # (the 1rt tile against the general one: bitwise)
                yo = out.clone(); out.zero_()
                OWNQ.own_mv_q8_0_a16(((rows + 31) // 32,), (256,), Wo[0], ctypes.c_int(rows), ctypes.c_int(K), x16, out)
                torch.cuda.synchronize()
                same = torch.equal(yo.view(torch.int32), out.view(torch.int32))
                Q8_SAME.append(same)
                say(f'  Q8_0 {role} [{rows} x {K}]: own_mv_q8_0_1rt_{A.q8} {"BITWISE" if same else "DIFFERS FROM"} own_mv_q8_0_a16')
                out.copy_(yo)
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
        line = f'{role:14s} {fmt:5s} [{rows:6d} x {K:5d}] {mat_bytes / 2**20:7.3f} MiB x {per:5d}/token, set {len(Wl)} = {len(Wl) * mat_bytes / 2**20:6.0f} MiB |'
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
say(f'token ({sum(PER_TOKEN.values()) if DENSE else 1393:,} matvecs, one launch each): read {token["read"] / 1e3:.3f} ms | llama {token["llama"] / 1e3:.3f} ms | '
    f'own {token["own"] / 1e3:.3f} ms;  bytes per token {token_bytes / 2**30:.3f} GiB (router excluded)')
if missing: say('no own kernel yet for: ' + '; '.join(missing))
if A.q2_0:   # (no dense module holds this tile: standalone only, on synthetic blocks of each shape the file reads)
    if not DENSE: raise SystemExit('own_mv_census: --q2_0 is for the dense qwen35 line')
    Q2 = Module('bench_own_mv_q2_0')
    shp = {}
    for role in ROLES:
        for t, _ in mats(role):
            rows, K, _ = shape_of(t); shp.setdefault((rows, K), [role, 0])[1] += 1
    say('Q2_0 own_mv_q2_0_k3, STANDALONE (in no dense decode module), synthetic blocks; per shape (first role reading it):')
    for (rows, K), (role, per) in shp.items():
        if K % 64: say(f'  {role} [{rows} x {K}]: K % 64 != 0, skipped'); continue
        nb = rows * K // 64; mat_bytes = nb * 18
        blk = np.empty((nb, 18), dtype=np.uint8)
        blk[:, 0:2] = (rng.standard_normal(nb) * 1e-2).astype(np.float16).view(np.uint8).reshape(nb, 2)
        blk[:, 2:] = rng.integers(0, 256, (nb, 16), dtype=np.uint8)
        raw = blk.reshape(-1)
        ncop = min(400, max(per, -(-512 * 2 ** 20 // mat_bytes)))
        capacity.check([0], [ncop * mat_bytes / 2 ** 20 + 64], f'census Q2_0 [{rows} x {K}] x {ncop}')
        w0 = torch.from_numpy(raw.copy()).to(dev); Ws = [w0] + [w0.clone() for _ in range(ncop - 1)]
        v = rng.standard_t(3, K); v[rng.choice(K, 8, replace=False)] *= 20; v = v / np.sqrt((v ** 2).mean())
        x16 = torch.tensor(v, dtype=torch.float32, device=dev).half(); out = torch.zeros(rows, dtype=torch.float32, device=dev)
        run_q2 = lambda w: Q2.own_mv_q2_0_k3(((rows + 15) // 16,), (256,), w, ctypes.c_int(rows), ctypes.c_int(K), x16, out)
        srows = np.sort(rng.choice(rows, min(256, rows), replace=False)); rb = K // 64 * 18
        W64 = own_dequant.dequant(np.ascontiguousarray(raw.reshape(rows, rb)[srows]).reshape(-1), 'Q2_0', len(srows) * K).astype(np.float64).reshape(len(srows), K)
        yref = W64 @ x16.double().cpu().numpy(); sc = np.sqrt((yref ** 2).mean())
        out.zero_(); run_q2(Ws[0]); torch.cuda.synchronize()
        e = np.abs(out.double().cpu().numpy()[srows] - yref) / sc
        arms = {'read': lambda w: run_read(w, sink), 'own': run_q2}
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]; tm = {k: [] for k in arms}
        for r in range(A.rounds + 1):
            for k, fn in arms.items():
                torch.cuda.synchronize(); ev[0].record()
                for w in Ws: fn(w)
                ev[1].record(); torch.cuda.synchronize()
                if r: tm[k].append(ev[0].elapsed_time(ev[1]) * 1e3 / len(Ws))
        line = f'{role:14s} Q2_0  [{rows:6d} x {K:5d}] {mat_bytes / 2**20:7.3f} MiB x {per:5d}/token ({ncop} copies) |'
        for k in ('read', 'own'):
            us = float(np.median(tm[k])); line += f' {k} {us:8.2f} us {mat_bytes / us / 1e3:6.1f} GB/s |'
        say(line + f' err own {e.mean():.1e}/{np.percentile(e, 99):.1e} (STANDALONE, synthetic)')
        del Ws, w0; torch.cuda.empty_cache()
if A.log:
    with open(A.log, 'a') as f: f.write('\n'.join(OUT) + '\n')
print(('RESULT PASS (the Q8_0 1rt tile bitwise the general tile on every shape)' if all(Q8_SAME) else 'RESULT FAIL (the Q8_0 1rt tile differs from the general tile)') if Q8_SAME else 'RESULT DONE')
