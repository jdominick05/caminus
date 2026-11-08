"""The qwen3next module's own behaviours (kernels/k3_gguf_q3n.hip, K3_Q3N) on ONE card (run it on the XT alone:
HIP_VISIBLE_DEVICES=1), a TP card's geometry:
  gdn   the GDN chain FlashNextTP / Qwen3NextTP build (l2 -> gates -> step + save -> norm, lazy commit) on k3_gguf_q3n
        (one token per pass) and k3_gguf_q3n_mt (T = 4) with 8 key heads and 16 value heads: the step's output and
        state and the norm's output vs a float64 reference of the gated delta rule with GROUPED heads (value head h reads
        key head h / 2, qwen3next's repeat-interleave) and the SiLU output gate, fed K3's own l2 / gates outputs (the
        unchanged ops); the tiled mapping (h % nk, Flash-Next's) and the sigmoid gate must NOT match (the defines are
        live); T = 4 passes == one-token passes bitwise (outputs and committed state).
  attn  the split attention pair with 8 query heads per kv head (ATTN_G 8) on both modules: identity lists vs float64
        (q8_0 and f16 caches) out to 20,000 cells (dense attention past the Coder's 2051: qwen3next has no QSA), T = 4
        rows == one-token passes bitwise, canaries after the records and the output intact, the same bits at per_wgp
        3 / 6 / 7 and run to run.
usage: HIP_VISIBLE_DEVICES=1 tools/guard.sh LOG SECS python -I tests/engine/test_q3n_ops.py [all|gdn|attn] [--small]"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from caminus.engine import k3
from caminus.engine.flashnext import EPS
from test_attn_split import kv_rand, ref_attn, cells_of, beq   # noqa: E402  (helpers only: its MOD / Pair are the Coder's)

M1, MT = 'k3_gguf_q3n', 'k3_gguf_q3n_mt'
S, KC = 128, 8                    # the state size; key heads per card (Qwen3NextTP's default gdn_keys 8 / 8)
H, CH = 2 * KC, 4 * S * KC        # value heads per card (2 per key head); qkv channels: q, k (S each per key head), 2 v
D = 256
RES, MODS = [], set()
CANARY = -12345.678
SMALL = '--small' in sys.argv


def ok(name, cond, info=''):
    RES.append((name, bool(cond)))
    print(f'{"ok  " if cond else "FAIL"} {name} {info}', flush=True)


def rel(a, b):
    return float(np.abs(np.asarray(a, np.float64) - b).max() / max(1e-30, np.abs(b).max()))


# ---- gdn ----------------------------------------------------------------------------------------------------------------
class GdnTape:
    """one card's GDN chain (Qwen3NextTP's segments) on `module`, ntok tokens per pass; the lazy commit as the engine runs
    it (params[1] = the previous pass's ntok after the first)"""
    def __init__(self, module, ntok, dt, a, wn, st0):
        z = lambda *s: torch.zeros(*s, device='cuda')
        self.ntok = ntok
        self.e = k3.Engine(module=module, grid=k3.resident_grid(module), T=ntok, limit_us=200_000)
        self.x, self.al, self.be, self.zg = z(ntok, CH), z(ntok, H), z(ntok, H), z(ntok, H * S)
        self.qk, self.gt, self.att, self.fin = z(ntok, 2 * KC * S), z(ntok, 2 * H), z(ntok, H * S), z(ntok, H * S)
        self.blk = z(k3.gdn_block_size(H, S, KC, ntok))
        e = self.e
        l_ = e.gdn_l2(self.x, self.qk, 2 * KC, S, EPS, x_stride=CH)
        g_ = e.gdn_gates(self.al, self.be, dt, a, self.gt)
        st = e.also(e.gdn_step(self.qk, self.x.reshape(-1)[2 * S * KC:], self.gt, self.blk, self.att, H, S, KC, CH, dep=l_), g_)
        e.gdn_save(self.qk, self.x.reshape(-1)[2 * S * KC:], self.gt, self.blk, H, S, KC, CH, dep=st)
        e.gdn_norm(self.att, wn, self.zg, self.fin, H, S, EPS, dep=st)
        e.build(); MODS.add(e.module)
        self.blk[:H * S * S].copy_(st0.reshape(-1))

    def run(self, cs, T0):
        """all T0 tokens, then one empty pass that commits the last ones: per token qk, gates, att, fin; the committed state"""
        out = {k: np.zeros((T0,) + s, np.float32) for k, s in (('qk', (2 * KC * S,)), ('gates', (2 * H,)), ('att', (H * S,)), ('fin', (H * S,)))}
        n = self.ntok
        for p in range(T0 // n + 1):
            self.e.params[1] = 0 if p == 0 else n
            if p < T0 // n:
                sl = slice(p * n, (p + 1) * n)
                for b, src in ((self.x, cs['x']), (self.al, cs['al']), (self.be, cs['be']), (self.zg, cs['z'])): b.copy_(torch.from_numpy(src[sl]))
            else:
                for b in (self.x, self.al, self.be, self.zg): b.zero_()
            self.e.run()
            if p < T0 // n:
                for k, b in (('qk', self.qk), ('gates', self.gt), ('att', self.att), ('fin', self.fin)): out[k][sl] = b.cpu().numpy()
        torch.cuda.synchronize()
        return out, self.blk[:H * S * S].cpu().numpy().reshape(H, S, S)


def gdn_ref(st0, qk, gates, v, z, wn, grouped=True, silu=True):
    """float64: the gated delta rule per value head on K3's l2 / gates outputs; key head h // (H / KC) (grouped) or h % KC
    (tiled); state M[h][col][i]; att = (M q) / sqrt(S); then the norm (rms * w) gated by silu(z) or sigmoid(z)"""
    T0 = qk.shape[0]; M = st0.astype(np.float64).copy(); att = np.zeros((T0, H, S)); fin = np.zeros((T0, H, S))
    for t in range(T0):
        q = qk[t, :KC * S].reshape(KC, S).astype(np.float64); k = qk[t, KC * S:].reshape(KC, S).astype(np.float64)
        for h in range(H):
            kh = h // (H // KC) if grouped else h % KC
            g, beta = np.exp(np.float64(gates[t, h])), np.float64(gates[t, H + h])
            kv = M[h] @ k[kh]; delta = (v[t, h].astype(np.float64) - g * kv) * beta
            M[h] = g * M[h] + np.outer(delta, k[kh]); att[t, h] = (M[h] @ q[kh]) / np.sqrt(S)
            o = att[t, h]; nrm = o / np.sqrt((o * o).mean() + EPS) * wn
            zz = z[t, h].astype(np.float64); sg = 1.0 / (1.0 + np.exp(-zz))
            fin[t, h] = nrm * (zz * sg if silu else sg)
    return att.reshape(T0, H * S), fin.reshape(T0, H * S), M


def cmd_gdn():
    rng = np.random.default_rng(31); T0 = 12 if SMALL else 64
    put = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).cuda()
    dt = put(rng.standard_normal(H) * 0.5); a = put(-np.exp(rng.standard_normal(H) * 0.5)); wn = put(1.0 + 0.1 * rng.standard_normal(S))
    st0 = (rng.standard_normal((H, S, S)) * 0.05).astype(np.float32)
    cs = dict(x=(rng.standard_normal((T0, CH)) * 0.5).astype(np.float32), al=(rng.standard_normal((T0, H)) * 2).astype(np.float32),
              be=(rng.standard_normal((T0, H)) * 2).astype(np.float32), z=(rng.standard_normal((T0, H * S))).astype(np.float32))
    one, st1 = GdnTape(M1, 1, dt, a, wn, put(st0)).run(cs, T0)
    four, st4 = GdnTape(MT, 4, dt, a, wn, put(st0)).run(cs, T0)
    v = cs['x'][:, 2 * S * KC:].reshape(T0, H, S)
    att_r, fin_r, M_r = gdn_ref(st0, one['qk'], one['gates'], v, cs['z'].reshape(T0, H, S), wn.cpu().numpy())
    TOL = 1e-4
    e_att, e_fin, e_st = rel(one['att'], att_r), rel(one['fin'], fin_r), rel(st1, M_r)
    ok(f'{M1}: step att vs float64 (grouped heads)', e_att < TOL, f'(rel {e_att:.2e}, {T0} tokens, {H} value heads over {KC} key heads)')
    ok(f'{M1}: norm out vs float64 (SiLU gate)', e_fin < TOL, f'(rel {e_fin:.2e})')
    ok(f'{M1}: committed state vs float64', e_st < TOL, f'(rel {e_st:.2e})')
    att_t, _, _ = gdn_ref(st0, one['qk'], one['gates'], v, cs['z'].reshape(T0, H, S), wn.cpu().numpy(), grouped=False)
    _, fin_s, _ = gdn_ref(st0, one['qk'], one['gates'], v, cs['z'].reshape(T0, H, S), wn.cpu().numpy(), silu=False)
    ok(f'{M1}: the tiled mapping (h % nk) does NOT match (K3_Q3N grouped heads live)', rel(one['att'], att_t) > 1e-2, f'(rel {rel(one["att"], att_t):.2e})')
    ok(f'{M1}: the sigmoid gate does NOT match (K3_Q3N SiLU gate live)', rel(one['fin'], fin_s) > 1e-2, f'(rel {rel(one["fin"], fin_s):.2e})')
    for k in ('qk', 'gates', 'att', 'fin'):
        ok(f'{MT} T=4 {k} == {M1} one-token passes bitwise', beq(four[k], one[k]))
    ok(f'{MT} T=4 committed state == one-token passes bitwise', beq(st4, st1))


# ---- attn ---------------------------------------------------------------------------------------------------------------
class Pair:
    """one engine with one split attention pair (identity lists, 8 query heads per kv head); canaries after the record
    and output buffers"""
    def __init__(self, module, T, rows, q8, nkv=1, per_wgp=None):
        cu = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device='cuda')
        G = k3.ATTN_G_OF[module]; assert G == 8, (module, G)
        self.T, self.rows, self.nkv, self.H = T, rows, nkv, G * nkv
        self.cache = cu(rows * k3.kv_pos_bytes(nkv * D, q8), dt=torch.uint8); self.q = cu(T * self.H * D)
        self.out_all = cu(T * self.H * D + 64).fill_(CANARY); self.out = self.out_all[:T * self.H * D]
        self.pos = cu(1, dt=torch.int32)
        ss = k3.ATTN_SS_OF[module]; ns = (rows + ss - 1) // ss; need = T * self.H * ns * k3.ATTN_RS
        self.part_all = cu(need + 4096).fill_(CANARY); self.part = self.part_all[:need]; self.need = need
        self.E = k3.Engine(module=module, grid=k3.resident_grid(module, per_wgp), T=T, limit_us=200_000)
        self.E.attn_dec(self.q, self.cache, self.pos, self.out, self.H, nkv, D, 1.0 / 16, q8=q8, part=self.part)
        ops = [sg.op for sg in self.E.segs]; assert ops == [k3.OP_ATTN_PART, k3.OP_ATTN_COMB], ops
        self.E.build(); MODS.add(self.E.module)

    def load(self, cache):
        self.cache.copy_(torch.from_numpy(np.ascontiguousarray(cache).reshape(-1)))

    def run(self, p, q):
        k3.check_span(p, self.T, self.rows, what='q3n attn test pass')
        self.q.copy_(torch.from_numpy(np.ascontiguousarray(q, np.float32).reshape(-1))); self.pos.fill_(p); self.E.run(check=True)
        return self.out.cpu().numpy().reshape(self.T, self.H, D).copy()

    def canaries(self):
        return bool((self.out_all[self.out.numel():] == CANARY).all().item() and (self.part_all[self.need:] == CANARY).all().item())


def cmd_attn():
    g = np.random.default_rng(7); R = 4200 if SMALL else 20480
    TOL = 4e-6   # relative to the output's max: fp32 sums of <= 20,000 terms in 32-cell slices
    pts_all = [0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 1000, 2050, 2051, 2052, 4095, 4096, 4111]
    pts_all += [] if SMALL else [8191, 8192, 12345, 16383, 16384, 20000]
    for q8 in (True, False):
        tag = 'q8_0' if q8 else 'f16'
        for nkv in ((1, 2) if q8 else (1,)):
            cache, kv = kv_rand(g, R, nkv, q8); G = 8; Hh = G * nkv
            one = Pair(M1, 1, R, q8, nkv); one.load(cache)
            for T, mod in ((1, M1), (4, MT)):
                dn = one if T == 1 else Pair(mod, T, R, q8, nkv)
                if T > 1: dn.load(cache)
                pts = sorted({min(p, R - T) for p in pts_all})
                worst, bad, tbad = 0.0, [], []
                for p in pts:
                    q = g.standard_normal((T, Hh, D)).astype(np.float32); o = dn.run(p, q)
                    for r in range(T):
                        e = rel(o[r], ref_attn(q[r], kv, cells_of(None, p + r), nkv, 1.0 / 16)); worst = max(worst, e)
                        if e > TOL: bad.append((p + r, e))
                        if T > 1 and not beq(o[r], one.run(p + r, q[r:r + 1])[0]): tbad.append(p + r)
                ok(f'{mod}: identity lists vs float64 ({tag}, nkv {nkv}, {Hh} heads, T={T})', not bad,
                   f'({len(pts)} passes up to {pts[-1] + T} cells; worst rel {worst:.2e}; bad {bad[:3]})')
                if T > 1: ok(f'{mod}: T={T} rows == one-token passes bitwise ({tag}, nkv {nkv})', not tbad, f'(bad {tbad[:4]})')
                ok(f'{mod}: canaries intact ({tag}, nkv {nkv}, T={T})', dn.canaries() and one.canaries())
    cache, _ = kv_rand(g, R, 1, True)
    qs = {p: np.random.default_rng(p).standard_normal((4, 8, D)).astype(np.float32) for p in (100, 2047, 2051, min(5000, R - 4), R - 4)}
    outs = {}
    for w in (8, 3, 6, 7, 8):
        dn = Pair(MT, 4, R, True, per_wgp=w); dn.load(cache)
        outs.setdefault(w, []).append(np.stack([dn.run(p, qs[p]) for p in qs]))
    ok(f'{MT}: run to run (per_wgp 8 twice) bitwise', beq(outs[8][0], outs[8][1]))
    for w in (3, 6, 7): ok(f'{MT}: per_wgp {w} == per_wgp 8 bitwise', beq(outs[w][0], outs[8][0]))


if __name__ == '__main__':
    torch.cuda.set_per_process_memory_fraction(0.9, 0)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]; cmd = args[0] if args else 'all'
    if cmd not in ('all', 'gdn', 'attn'): raise SystemExit(__doc__)
    if cmd in ('all', 'gdn'): cmd_gdn()
    if cmd in ('all', 'attn'): cmd_attn()
    nf = sum(1 for _, g in RES if not g)
    print(f'RESULT {cmd}: {len(RES) - nf} ok, {nf} FAIL (modules {sorted(MODS)})')
    sys.exit(1 if nf else 0)
