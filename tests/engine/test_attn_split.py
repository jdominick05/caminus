"""The split attention pair (kernels/k3_attn_split.h: OP_ATTN_PART + OP_ATTN_COMB, caminus.engine.k3.Engine.attn_dec / attn_sel on
the GSQ-RCO Coder's modules) on ONE card (run it on the XT alone: HIP_VISIBLE_DEVICES=1), a TP card's geometry (12 query
heads over 1 kv head of 256; also 24 over 2, the layer split's):
  (1) vs a float64 reference over the same cell list (dequantized K / V as kv_get): identity lists of every length
      around the split / block edges (1 .. 2051 cells) and SEL lists past 2051 (random selections, adversarial rows:
      clamped, in bounds; an empty list -> zeros), q8_0 and f16, T = 1 and 4;
  (2) T = 4 rows == the one-token pass at the same position, bitwise (the slices depend on the row's list only);
  (3) the identity through a SEL row (QSA module) == the trunk module's dense identity, bitwise (one body);
  (4) canaries after the record buffer and the output stay intact; a list longer than the records hold (max_cells
      below it) is cut to the first NS * ATTN_SS cells, never written past;
  (5) the same bits at other grids (per_wgp 3 / 6 / 7) and run to run.
usage: HIP_VISIBLE_DEVICES=1 guard.sh LOG SECS python -I tests/engine/test_attn_split.py [all|ref|grids]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
import torch
from caminus.engine import k3, qsa

MOD = 'k3_gguf_gsq'
D = 256
RES, MODS = [], set()
CANARY = -12345.678


def ok(name, cond, info=''):
    RES.append((name, bool(cond)))
    print(f'{"ok  " if cond else "FAIL"} {name} {info}', flush=True)


def beq(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def kv_rand(g, rows, nkv, q8):
    """a random cache [rows][K nkv*256 | V nkv*256]: q8_0 (block_q8_0 rows) or f16; and its dequantized values [rows][2][nkv*256]"""
    M = nkv * D
    x = (g.standard_normal((rows, 2 * M)) * 1.5).astype(np.float32)
    if q8:
        b = qsa.q8_0_rows(x.reshape(-1, 128)).reshape(rows, -1)
        v = qsa.dq_rows(b.reshape(-1, 136)).reshape(rows, 2, M)
    else:
        b = x.astype(np.float16).view(np.uint8).reshape(rows, -1); v = x.astype(np.float16).astype(np.float32).reshape(rows, 2, M)
    return b, v.astype(np.float64)


def sel_row(ids, pos):
    r = np.zeros(qsa.SELW, np.int32); r[0] = len(ids); r[1] = qsa.ntail_of(pos); r[2] = qsa.nv_of(pos); r[4:4 + len(ids)] = ids
    return r


def cells_of(row, pos):
    """the list the device walks for a SEL row (attn_list's clamps); None row: the identity"""
    if row is None: return np.arange(pos + 1, dtype=np.int64)
    nv = qsa.nv_of(pos); n = min(max(int(row[0]), 0), min(nv, qsa.TOPK))
    if n == nv: return np.arange(pos + 1, dtype=np.int64)
    ids = np.clip(row[4:4 + n].astype(np.int64), 0, max(nv - 1, 0))
    return np.concatenate([(4 * ids[:, None] + np.arange(4)[None]).reshape(-1), np.arange(4 * nv, pos + 1)]).astype(np.int64)


def ref_attn(q, kv, cl, nkv, scale):
    """float64: per query head h (kv head h // (heads / nkv)) softmax(scale q . K[cl]) . V[cl]; empty list: zeros"""
    H = q.shape[0]; G = H // nkv; out = np.zeros((H, D))
    if len(cl) == 0: return out
    for h in range(H):
        g = h // G; K = kv[cl, 0, g * D:(g + 1) * D]; V = kv[cl, 1, g * D:(g + 1) * D]
        s = (K @ q[h].astype(np.float64)) * scale; p = np.exp(s - s.max()); out[h] = (p @ V) / p.sum()
    return out


class Pair:
    """one engine with one split attention pair: sel=False the trunk module's attn_dec (identity), True the QSA module's
    attn_sel; record / output buffers with canaries after them"""
    def __init__(self, T, rows, q8, nkv=1, sel=False, per_wgp=None, max_cells=None):
        cu = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device='cuda')
        self.T, self.rows, self.nkv, self.H = T, rows, nkv, 12 * nkv
        self.cache = cu(rows * k3.kv_pos_bytes(nkv * D, q8), dt=torch.uint8); self.q = cu(T * self.H * D)
        self.out_all = cu(T * self.H * D + 64).fill_(CANARY); self.out = self.out_all[:T * self.H * D]
        self.pos = cu(1, dt=torch.int32); self.sel = cu(T * qsa.SELW, dt=torch.int32)
        mc = max_cells or (min(rows, 2051) if sel else rows)
        ns = (mc + k3.ATTN_SS - 1) // k3.ATTN_SS; need = T * self.H * ns * k3.ATTN_RS
        self.part_all = cu(need + 4096).fill_(CANARY); self.part = self.part_all[:need]; self.need = need
        self.E = k3.Engine(module=MOD, grid=k3.resident_grid(MOD, per_wgp), T=T, limit_us=200_000)
        a = (self.q, self.cache, self.pos)
        if sel: self.E.attn_sel(*a, self.sel, self.out, self.H, nkv, D, 1.0 / 16, q8=q8, max_cells=max_cells, part=self.part)
        else: self.E.attn_dec(*a, self.out, self.H, nkv, D, 1.0 / 16, q8=q8, max_cells=max_cells, part=self.part)
        ops = [sg.op for sg in self.E.segs]
        assert ops == [k3.OP_ATTN_PART, k3.OP_ATTN_COMB], ops
        self.E.build(); MODS.add(self.E.module)

    def load(self, cache):
        self.cache.copy_(torch.from_numpy(np.ascontiguousarray(cache).reshape(-1)))

    def run(self, p, q, rows=None):
        k3.check_span(p, self.T, self.rows, what='attn split test pass')
        if rows is not None: self.sel.copy_(torch.from_numpy(np.ascontiguousarray(rows, np.int32).reshape(-1)))
        self.q.copy_(torch.from_numpy(np.ascontiguousarray(q, np.float32).reshape(-1))); self.pos.fill_(p); self.E.run(check=True)
        return self.out.cpu().numpy().reshape(self.T, self.H, D).copy()

    def canaries(self):
        return bool((self.out_all[self.out.numel():] == CANARY).all().item() and (self.part_all[self.need:] == CANARY).all().item())


def err(a, b):
    return float(np.abs(a - b).max() / max(1e-30, np.abs(b).max()))


def cmd_ref():
    g = np.random.default_rng(7)
    R = 8200
    TOL = 2e-6   # relative to the output's max: fp32 sums of <= 2051 terms
    for q8 in (True, False):
        tag = 'q8_0' if q8 else 'f16'
        for nkv in ((1, 2) if q8 else (1,)):
            cache, kv = kv_rand(g, R, nkv, q8); H = 12 * nkv
            one = Pair(1, R, q8, nkv); one.load(cache)
            for T in (1, 4):
                dn = Pair(T, R, q8, nkv); dn.load(cache)
                pts = sorted({0, 1, 2, 30, 31, 32, 33, 62, 63, 64, 65, 66, 126, 127, 128, 129, 511, 512, 513, 1000, 2046, 2050 - T + 1})
                worst, bad, tbad = 0.0, [], []
                for p in pts:
                    q = g.standard_normal((T, H, D)).astype(np.float32)
                    o = dn.run(p, q)
                    for r in range(T):
                        e = err(o[r], ref_attn(q[r], kv, cells_of(None, p + r), nkv, 1.0 / 16)); worst = max(worst, e)
                        if e > TOL: bad.append((p + r, e))
                        if T > 1 and not beq(o[r], one.run(p + r, q[r:r + 1])[0]): tbad.append(p + r)
                ok(f'identity lists vs float64 ({tag}, nkv {nkv}, T={T})', not bad, f'({len(pts)} passes up to {pts[-1] + T} cells; worst rel {worst:.2e}; bad {bad[:3]})')
                if T > 1: ok(f'identity T={T} rows == one-token passes bitwise ({tag}, nkv {nkv})', not tbad, f'(bad {tbad[:4]})')
                ok(f'canaries intact (identity, {tag}, nkv {nkv}, T={T})', dn.canaries() and one.canaries())
            if nkv != 1: continue
            for T in (1, 4):
                sl = Pair(T, R, q8, sel=True); sl.load(cache); one_s = Pair(1, R, q8, sel=True); one_s.load(cache)
                dn = Pair(T, R, q8); dn.load(cache)
                # (3) identity through SEL rows (nv <= 512) == the trunk's dense identity
                bad = []
                for p in (0, 3, 64, 511, 1000, 2047 - T + 1, 2050 - T + 1):
                    q = g.standard_normal((T, H, D)).astype(np.float32)
                    rows = np.stack([sel_row(np.arange(qsa.nv_of(p + r)), p + r) for r in range(T)])
                    if not beq(sl.run(p, q, rows), dn.run(p, q)): bad.append(p)
                ok(f'SEL identity (QSA module) == dense identity (trunk) bitwise ({tag}, T={T})', not bad, f'(bad {bad})')
                # random selections past 2051, then adversarial rows
                worst, bad, tbad = 0.0, [], []
                for p in (2051, 2052, 2053, 2054, 3000, 4095, 4096, 6143, 8191 - T + 1):
                    q = g.standard_normal((T, H, D)).astype(np.float32)
                    rows = np.stack([sel_row(np.sort(g.choice(qsa.nv_of(p + r), qsa.TOPK, replace=False)), p + r) for r in range(T)])
                    o = sl.run(p, q, rows)
                    for r in range(T):
                        e = err(o[r], ref_attn(q[r], kv, cells_of(rows[r], p + r), 1, 1.0 / 16)); worst = max(worst, e)
                        if e > TOL: bad.append((p + r, e))
                        if T > 1 and not beq(o[r], one_s.run(p + r, q[r:r + 1], rows[r:r + 1])[0]): tbad.append(p + r)
                ok(f'random SEL lists vs float64 ({tag}, T={T})', not bad, f'(worst rel {worst:.2e}; bad {bad[:3]})')
                if T > 1: ok(f'SEL T={T} rows == one-token passes bitwise ({tag})', not tbad, f'(bad {tbad[:4]})')
                bad = []
                for case in range(6):
                    p = [2051, 4095, 8191 - T + 1, 2051, 4099, 2054][case]
                    q = g.standard_normal((T, H, D)).astype(np.float32)
                    rows = np.stack([sel_row(np.sort(g.choice(qsa.nv_of(p + r), qsa.TOPK, replace=False)), p + r) for r in range(T)])
                    if case == 0: rows[:, 0] = -5
                    if case == 1: rows[:, 0] = 10 ** 9
                    if case == 2: rows[:, 4:10] = [-7, -1, 2 ** 31 - 1, 99999, -2 ** 31, 5]
                    if case == 3: rows[:, 4:] = 2 ** 30
                    if case == 4: rows[:, 0] = 0          # row 0 at 4099: (pos + 1) % 4 == 0 and no pools: an empty list
                    if case == 5: rows[:, 4:] = -(2 ** 30)
                    o = sl.run(p, q, rows)
                    for r in range(T):
                        cl = cells_of(rows[r], p + r)
                        assert len(cl) == 0 or (cl.min() >= 0 and cl.max() <= p + r < R), (case, cl)
                        if len(cl) == 0:
                            if not (o[r] == 0).all(): bad.append((case, r, 'nonzero'))
                            continue
                        e = err(o[r], ref_attn(q[r], kv, cl, 1, 1.0 / 16))
                        if e > TOL: bad.append((case, r, e))
                ok(f'adversarial SEL rows == clamped list / zeros ({tag}, T={T})', not bad, f'(6 cases; bad {bad[:4]})')
                ok(f'canaries intact (SEL, {tag}, T={T})', sl.canaries() and one_s.canaries() and dn.canaries())
    # (4) records sized below the list: cut to the first NS * ATTN_SS cells, nothing written past the buffer
    cache, kv = kv_rand(g, 4200, 1, True)
    for T in (1, 4):
        sm = Pair(T, 4200, True, max_cells=300); sm.load(cache)   # NS = 5: 320 cells
        bad = []
        for p in (100, 319 - T + 1, 320, 1000, 4100):
            q = g.standard_normal((T, 12, D)).astype(np.float32); o = sm.run(p, q)
            for r in range(T):
                n = min(p + r + 1, 320)
                if err(o[r], ref_attn(q[r], kv, np.arange(n), 1, 1.0 / 16)) > 2e-6: bad.append(p + r)
        ok(f'max_cells 300 (NS 5): lists cut to 320 cells, canaries intact (T={T})', not bad and sm.canaries(), f'(bad {bad})')


def cmd_grids():
    g = np.random.default_rng(3); R = 8200
    cache, _ = kv_rand(g, R, 1, True)
    qs = {p: np.random.default_rng(p).standard_normal((4, 12, D)).astype(np.float32) for p in (100, 2047, 2051, 5000, 8188)}
    rws = {p: np.stack([sel_row(np.sort(np.random.default_rng(p + r).choice(qsa.nv_of(p + r), qsa.TOPK, replace=False)), p + r)
                        for r in range(4)]) for p in (2051, 5000, 8188)}
    outs = {}
    for w in (8, 3, 6, 7, 8):
        dn = Pair(4, R, True, per_wgp=w); sl = Pair(4, R, True, sel=True, per_wgp=w); dn.load(cache); sl.load(cache)
        o = [dn.run(p, qs[p]) for p in (100, 2047)] + [sl.run(p, qs[p], rws[p]) for p in (2051, 5000, 8188)]
        outs.setdefault(w, []).append(np.stack(o))
    base = outs[8][0]
    ok('run to run (per_wgp 8 twice) bitwise', beq(outs[8][0], outs[8][1]))
    for w in (3, 6, 7): ok(f'per_wgp {w} == per_wgp 8 bitwise', beq(outs[w][0], base))


if __name__ == '__main__':
    torch.cuda.set_per_process_memory_fraction(0.9, 0)
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd in ('all', 'ref'): cmd_ref()
    if cmd in ('all', 'grids'): cmd_grids()
    if cmd not in ('all', 'ref', 'grids'): raise SystemExit(__doc__)
    nf = sum(1 for _, g in RES if not g)
    print(f'RESULT {cmd}: {len(RES) - nf} ok, {nf} FAIL (modules {sorted(MODS)})')
    sys.exit(1 if nf else 0)
