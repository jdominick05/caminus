"""OP_QSA_PSEL (kernels/k3_qsa_ops.h: OP_QSA_SELECT's selection spread over slices of the scores; Engine.qsa_select_par)
on ONE card (run it on the XT alone: HIP_VISIBLE_DEVICES=1), the GSQ-RCO Coder's QSA modules (T = 1 and 4), host-given
scores (no SCORE op):
  cases   random, indexer-like (sums of relu'd dots, many zeros), ties at the threshold, all equal, all zero, -0 / inf /
          NaN / denormal / negative, narrow ranges (thousands of keys sharing the top 16 bits), nv 513 .. 262,143, slice
          counts 1 / 3 / 64 / the default: every SEL row (header words included) bitwise == OP_QSA_SELECT's on the same
          buffer, and the ids == caminus.engine.qsa.select_ref
  chain   three layers in one tape sharing the slice lists, different scores each: every layer == OP_QSA_SELECT's; the
          same tape run again: bitwise (ZERO clears the counts every pass); per_wgp 3 / 6 / 7: bitwise
  time    12 chained layers, OP_QSA_SELECT vs OP_QSA_PSEL at nv 16,384 / 65,536 / 262,144 (64K / 262K / 1M contexts),
          indexer-like scores, T = 1 and 4 (wall per pass, median of 25)
usage: HIP_VISIBLE_DEVICES=1 tools/guard.sh LOG 900 python -I tests/engine/test_qsa_psel.py [all|cases|chain|time]
ends RESULT PASS"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np
import torch
from caminus.engine import k3, qsa

MOD = 'k3_gguf_gsq'
fails = []
MODS = set()


def ok(name, cond, info=''):
    print(f'{"ok  " if cond else "FAIL"} {name} {info}', flush=True)
    if not cond: fails.append(name)


class Sel:
    """one tape of `layers` chained selections over [T][prow] host-given score buffers: OP_QSA_SELECT (par False) or
    OP_QSA_PSEL (par True, `slices`)"""
    def __init__(self, T, prow, par, slices=None, layers=1, per_wgp=None):
        cu = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device='cuda')
        self.T, self.prow = T, prow
        self.score = [cu(T, prow) for _ in range(layers)]
        self.sel = [cu(T, k3.QSA_SELW, dt=torch.int32) for _ in range(layers)]
        self.pos = cu(1, dt=torch.int32)
        self.E = k3.Engine(module=MOD, grid=k3.resident_grid(MOD, per_wgp), T=T, limit_us=200_000)
        d = -1
        for l in range(layers):
            d = (self.E.qsa_select_par(self.score[l], self.pos, self.sel[l], prow, dep=d, slices=slices) if par else
                 self.E.qsa_select(self.score[l], self.pos, self.sel[l], prow, dep=d))
        self.E.build(); MODS.add(self.E.module)

    def load(self, scores):   # scores: [layers] of [T, prow] float32
        for b, s in zip(self.score, scores): b.copy_(torch.from_numpy(np.ascontiguousarray(s, np.float32)))

    def run(self, pos):
        k3.check_span(pos, self.T, 4 * self.prow, what='psel test pass')
        self.pos.fill_(pos); self.E.run(check=True)
        return [s.cpu().numpy().copy() for s in self.sel]


def indexer_like(g, n):
    """sum over 4 heads of relu(q . k) / sqrt(128): non-negative, a share of exact zeros"""
    return (np.maximum(g.standard_normal((4, n)), 0).sum(0) / np.sqrt(128)).astype(np.float32)


def cases():
    g = np.random.default_rng(11)
    out = []
    def add(name, s): out.append((name, np.asarray(s, np.float32)))
    add('513 zeros', np.zeros(513)); add('1000 three levels', g.integers(0, 3, 1000) * 0.5)
    add('514 descending', np.arange(514, 0, -1)); add('2048 ascending', np.arange(2048))
    add('5000 indexer-like', indexer_like(g, 5000)); add('65536 uniform', g.random(65536))
    add('65536 all equal', np.full(65536, 1.25))
    t = np.zeros(65536, np.float32); t[g.choice(65536, 400, replace=False)] = 3.0; t[g.choice(np.where(t == 0)[0], 600, replace=False)] = 2.0
    add('65536 400 top + 600 tied at the threshold', t)
    m = g.random(4099).astype(np.float32); m[::7] = -0.0; m[::11] = 0.0; m[5] = np.inf; m[9] = 1e-45; m[13] = -1.0; m[17] = np.nan
    add('4099 -0 / +0 / inf / NaN / denormal / negative', m)
    add('100001 narrow [1, 1 + 1e-3)', 1.0 + 1e-3 * g.random(100001))
    add('100001 narrow [1, 1.05)', 1.0 + 0.05 * g.random(100001))
    add('100001 indexer-like', indexer_like(g, 100001)); add('262143 indexer-like', indexer_like(g, 262143))
    return out


def cmd_cases():
    allok, n = True, 0
    for T in (1, 4):
        for name, s in cases():
            nv = len(s); prow = nv + 37                   # (more pool rows than visible: the slices follow nv, not N)
            pos = 4 * nv - 1 - (T - 1)                    # the last row sees nv pools
            sc = np.zeros((T, prow), np.float32); sc[:, :nv] = s
            ref = Sel(T, prow, False); ref.load([sc]); want = ref.run(pos)[0]
            for P in (None, 1, 3, 64):
                ps = Sel(T, prow, True, P); ps.load([sc]); got = ps.run(pos)[0]
                good = np.array_equal(got, want)
                for r in range(T):
                    ids, thr = qsa.select_ref(s, qsa.nv_of(pos + r))
                    good = good and got[r, 0] == len(ids) and np.array_equal(got[r, 4:4 + got[r, 0]], ids)
                n += 1
                if not good:
                    allok = False
                    bad = [r for r in range(T) if not np.array_equal(got[r], want[r])]
                    print(f'  {name} T={T} slices {P}: rows differing {bad}; count {got[bad[0], 0] if bad else "-"} vs '
                          f'{want[bad[0], 0] if bad else "-"}; first differing word '
                          f'{int(np.argmax(got[bad[0]] != want[bad[0]])) if bad else "-"}', flush=True)
    ok('PSEL == SELECT bitwise (SEL rows with header) and == select_ref', allok, f'({n} case x T x slices runs)')


def cmd_chain():
    g = np.random.default_rng(5)
    for T in (1, 4):
        nv = 70001; prow = nv; pos = 4 * nv - 1 - (T - 1)
        sc = [np.stack([indexer_like(g, prow) for _ in range(T)]) for _ in range(3)]
        sc[1][:, ::3] = 0.0                               # (layer 1: more zeros; layer 2 below: a tie at the threshold)
        ref = Sel(T, prow, False, layers=3); ref.load(sc); want = ref.run(pos)
        ps = Sel(T, prow, True, layers=3); ps.load(sc); got1 = ps.run(pos); got2 = ps.run(pos)
        ok(f'chain of 3 layers sharing the slice lists == SELECT (T={T})', all(np.array_equal(a, b) for a, b in zip(got1, want)))
        ok(f'the same tape run twice: bitwise (T={T})', all(np.array_equal(a, b) for a, b in zip(got1, got2)))
        same = True
        for w in (3, 6, 7):
            pw = Sel(T, prow, True, layers=3, per_wgp=w); pw.load(sc); same = same and all(np.array_equal(a, b) for a, b in zip(pw.run(pos), want))
        ok(f'grids per_wgp 3 / 6 / 7 == SELECT (T={T})', same)


def bench(E, n=25):
    for _ in range(5): E.run(check=True)
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t0 = time.perf_counter(); E.run(check=False); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def cmd_time():
    g = np.random.default_rng(3)
    for nv in (16384, 65536, 262144):
        for T in (1, 4):
            sc = [np.stack([indexer_like(g, nv) for _ in range(T)]) for _ in range(12)]
            pos = 4 * nv - 1 - (T - 1)
            a = Sel(T, nv, False, layers=12); a.load(sc); wa = a.run(pos); ta = bench(a.E)
            b = Sel(T, nv, True, layers=12); b.load(sc); wb = b.run(pos); tb = bench(b.E)
            same = all(np.array_equal(x, y) for x, y in zip(wa, wb))
            print(f'     time nv {nv:7,d} (context {4 * nv:,}) T={T}, 12 layers: SELECT {ta:.3f} ms, PSEL {tb:.3f} ms '
                  f'(x{ta / tb:.1f}){"" if same else " SEL DIFFERS"}', flush=True)
            ok(f'time run nv {nv} T={T}: PSEL == SELECT', same)


what = sys.argv[1] if len(sys.argv) > 1 else 'all'
print(f'device {torch.cuda.get_device_name(0)}', flush=True)
if what in ('all', 'cases'): cmd_cases()
if what in ('all', 'chain'): cmd_chain()
if what in ('all', 'time'): cmd_time()
print(f'modules {sorted(MODS)}')
print(f'RESULT {"PASS" if not fails else "FAIL"} ({len(fails)} failed)')
sys.exit(1 if fails else 0)
