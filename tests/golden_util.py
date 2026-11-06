"""Shared helpers of the tests/test_golden_*.py checks (K3 engine vs llama.cpp's own tensors from tools/gg_golden), for
both Flash-Next models: REAP-308 (Q8_0 hc, F32 router, module k3_gguf) and the GSQ-RCO Coder (BF16 hc / router / alpha /
beta / PLE key-value, the K-quant and IQ formats of docs/agentic/spec-gsq-port.md, module k3_gguf_gsq).

Gate G3 of the GSQ port (spec S4): an op whose kernel reproduces llama.cpp's summation order must be BITWISE equal to
llama.cpp's tensor when fed llama.cpp's own input; every other op within 1e-6 relative. Bitwise is expected for
  - BF16 matvecs (OP_BF16_MV = mul_mat_vec_f<nv_bfloat16> at ne11 = 1),
  - the GGUF formats llama.cpp's RDNA3 MMVQ runs with one warp per row at ncols = 1 (mmvq-impl.cuh calc_nwarps):
    Q4_K, Q5_K, IQ4_XS, IQ2_S, IQ3_S, Q2_0 (K3's one-wave gg_row has the same order).
Q6_K (2 warps), IQ4_NL / Q8_0 (8 warps) and IQ3_XXS stay tolerance-only (bitwise reported when reached).
A chained check that crosses a q8_1 quantization of the engine's own float (Report.q8x / Report.chain) gates the q8_1
bytes first (equal, or rounding flips the float difference explains), then the op after it bitwise against the same op
fed llama.cpp's float quantized; its error vs llama.cpp's tensor is a diagnostic (a flip moves it far past 1e-6)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import numpy as np
import torch
from caminus import gguf as G, k3
from caminus.flashnext import pick_module

from caminus import devpaths  # noqa: E402  (CODER / REAP / ARTIFACTS: CLAUDE.md, "Models and data")
REAP = devpaths.reap()
BITWISE_FMTS = frozenset({'BF16', 'Q4_K', 'Q5_K', 'IQ4_XS', 'IQ2_S', 'IQ3_S', 'Q2_0'})
TOL = 1e-6


class Golden:
    """a tools/gg_golden dump: golden(step, name, nth) = the nth tensor of that name at that step (graph order)"""
    def __init__(self, gd):
        self.gd = gd
        self.idx = [l.rstrip('\n').split('\t') for l in open(f'{gd}/index.tsv')]
        self.steps = sorted({int(r[0]) for r in self.idx})

    def __call__(self, step, name, nth=0):
        r = [r for r in self.idx if int(r[0]) == step and r[2] == name][nth]; ne = [int(v) for v in r[4:8]]
        return np.fromfile(f'{self.gd}/{r[8]}', dtype=np.float32 if r[3] == 'f32' else np.int32).reshape(ne[::-1]).squeeze()

    def tokens(self):
        return [int(v) for v in open(f'{self.gd}/tokens.txt').read().split()]


class Model:
    """the GGUF's tensors (memory-mapped; a split GGUF's first shard holds every tensor the engine reads) and the K3 module
    its formats need (caminus.flashnext.pick_module: k3_gguf for REAP-308, k3_gguf_gsq for the Coder)"""
    def __init__(self, path):
        self.path = path
        self.rd = G.GGUFReader(path, 'r'); self.T = {t.name: t for t in self.rd.tensors}
        self.module = pick_module(self.T)

    def fmt(self, name): return self.T[name].tensor_type.name

    def raw(self, name):
        """dict(w = raw bytes on the device, fmt, rows, cols, rb = bytes per row, slab = bytes per expert (3-D))"""
        t = self.T[name]; a = np.asarray(t.data)
        return dict(name=name, w=torch.from_numpy(np.array(a)).cuda().reshape(-1), fmt=t.tensor_type.name,
                    rows=int(t.shape[1]) if len(t.shape) > 1 else 1, cols=int(t.shape[0]), rb=a.shape[-1] if a.ndim >= 2 else a.size,
                    slab=a.shape[-2] * a.shape[-1] if a.ndim == 3 else a.size)

    def f32(self, name, shape=None):
        t = self.T[name]; assert t.tensor_type.name == 'F32', (name, t.tensor_type.name)
        w = torch.from_numpy(np.asarray(t.data, dtype=np.float32).copy()).cuda()
        return w.reshape(shape) if shape else w.reshape(-1)

    def engine(self, **kw): return k3.Engine(module=self.module, **kw)


def mv(e, W, out, dep=-1, xq=None, xf=None, rpt=16, silu_scale=None):
    """W . x -> out [rows], as the tapes run it (caminus.flashnext FlashNextK3._mv): a GGUF-format W reads xq (q8_1 blocks,
    or fp32 quantized per workgroup), a BF16 W the fp32 xf (OP_BF16_MV; silu_scale: silu(xf * silu_scale) staged)"""
    if W['fmt'] == 'BF16':
        assert xf is not None, f"{W['name']}: BF16 needs the fp32 activation"
        return e.bf16_mv(W['w'], W['rows'], W['cols'], xf, out, rows_per_task=min(rpt, 8), dep=dep, silu_scale=silu_scale)
    return e.gg_matvec(W['w'], W['fmt'], W['rows'], W['cols'], W['rb'], xq if xq is not None else xf, out, rows_per_task=rpt,
                       dep=dep, silu_scale=silu_scale)


def small_mv(m, e, name, rows, x, out, dep=-1, rpt=8):
    """router logits / shared-expert gate logit read by fp32 x: F32 (REAP-308, OP_F32_MV) or BF16 (the Coder, OP_BF16_MV;
    the 1-row shared gate is stored 1-D)"""
    f = m.fmt(name)
    if f == 'F32': return e.f32_mv(m.f32(name, (rows, -1)), x, out, rows_per_task=rpt, dep=dep)
    assert f == 'BF16', (name, f)
    W = m.raw(name); assert W['rows'] == rows, (name, W['rows'], rows)
    return e.bf16_mv(W['w'], rows, W['cols'], x, out, rows_per_task=rpt, dep=dep)


def cu(a): return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).cuda()
def q8b(n): return torch.empty((n + 31) // 32 * 36, dtype=torch.uint8, device='cuda')
def buf(*s, dt=torch.float32): return torch.empty(*s, dtype=dt, device='cuda')


def _np(t): return t.detach().reshape(-1).cpu().numpy() if torch.is_tensor(t) else np.asarray(t).reshape(-1)


def q8_split(q8):
    """q8_1 blocks (36 bytes: half d, half s, int8 qs[32]) -> d f16 [nb], s f16 [nb], qs int8 [nb, 32]"""
    b = _np(q8).view(np.uint8).reshape(-1, 36)
    return b[:, 0:2].copy().view(np.float16).reshape(-1), b[:, 2:4].copy().view(np.float16).reshape(-1), b[:, 4:].view(np.int8)


def q8_host(x):
    """llama.cpp's quantize_q8_1 of fp32 x on the host, in the device's float32 arithmetic (d = amax / 127.0f,
    q = roundf(x / d), d stored as half): (d f32 [nb], d f16 [nb], v = x / d f32 [nb, 32], q int8 [nb, 32])"""
    x = _np(x).astype(np.float32); n = x.size; xb = np.zeros((n + 31) // 32 * 32, dtype=np.float32); xb[:n] = x
    xb = xb.reshape(-1, 32); amax = np.abs(xb).max(1); d = (amax / np.float32(127.0)).astype(np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.where(d[:, None] > 0, xb / np.where(d > 0, d, 1)[:, None], 0).astype(np.float32)
    q = (np.sign(v) * np.floor(np.abs(v.astype(np.float64)) + 0.5)).astype(np.int8)          # roundf: half away from zero
    return d, d.astype(np.float16), v, q


class Report:
    """per-tensor comparison: max relative error (vs max |ref|) and bitwise equality. check(name, ours, ref, fmt=...):
    fmt in BITWISE_FMTS (or bitwise=True) makes the tensor a bitwise gate, else the gate is rel <= tol; diag=True only
    reports (no gate). q8x(...): the gate of a chain crossing a q8_1 quantization (see there)."""
    def __init__(self):
        self.ok, self.line, self.stats = True, [], {}

    def check(self, name, ours, ref, fmt=None, bitwise=None, tol=TOL, diag=False):
        a, b = _np(ours), _np(ref)
        bit = a.shape == b.shape and np.array_equal(a.view(np.uint32) if a.dtype == np.float32 else a, b.view(np.uint32) if b.dtype == np.float32 else b)
        rel = 0.0 if bit else float(np.abs(a.astype(np.float64) - b).max() / (np.abs(b).max() + 1e-30))
        want_bit = bitwise if bitwise is not None else (fmt in BITWISE_FMTS)
        good = True if diag else bit if want_bit else rel <= tol
        self.ok &= good
        st = self.stats.setdefault(name, dict(fmt=fmt, want_bit=want_bit, tol=tol, bit=0, n=0, worst=0.0, ndiag=0)); st['n'] += 1; st['bit'] += bit; st['worst'] = max(st['worst'], rel)
        st['ndiag'] += diag
        tag = f'{name}{"[" + fmt + "]" if fmt else ""} ' + ('bitwise' if bit else f'{rel:.1e}') + (' (diag)' if diag else '' if good else (' FAIL(bitwise expected)' if want_bit else ' FAIL'))
        self.line.append(tag)
        return good

    def q8x(self, name, ours_q8, ref_q8, ours_x, ref_x, exact=True, eps=1e-4, tol=TOL):
        """A chain that crosses a q8_1 quantization: the engine quantized its own float ours_x into ours_q8; ref_q8 is the
        q8_1 of llama.cpp's float ref_x (K3's gg_q8_1, bitwise llama.cpp's quantize_q8_1). ours_x is within 1e-6 of
        ref_x (gated here as well: rel <= tol), yet a code whose x / d sits at a .5 boundary may round the other way, and every op after
        it then differs by far more than 1e-6 (text-dependent: 1.6e-4 on the gdn out of a code text). Gate:
          - exact (ours_x is the float the engine quantized): ours_q8's d and codes are the host's quantize_q8_1 of
            ours_x (the fused / producer-side quantizer = the standalone one, bit for bit); same for ref_q8 and ref_x;
          - the bytes the vec_dots read (d, qs) equal, or every difference explained by the float difference: a code
            off by 1 whose two values x / d (each from its own float) lie on the two sides of the .5 boundary between
            the codes (eps code units of slack only when ours_x is recomputed on the host, exact=False: the shared
            expert's swiglu glue), and a d off by at most one half ulp.
        Returns True when d and qs are equal: then the caller gates the chained output BITWISE against the same op fed
        ref_q8 (stricter than the old 1e-6), else the chained output is only a diagnostic. The block sums (s) are
        reported, not gated (none of the vec_dots of these formats reads them; a bitwise downstream gate would show it)."""
        d_o, s_o, q_o = q8_split(ours_q8); d_r, s_r, q_r = q8_split(ref_q8)
        nb = min(d_o.size, (_np(ours_x).size + 31) // 32); d_o, s_o, q_o, d_r, s_r, q_r = d_o[:nb], s_o[:nb], q_o[:nb], d_r[:nb], s_r[:nb], q_r[:nb]
        _, hd_o, v_o, hq_o = q8_host(ours_x); _, hd_r, v_r, hq_r = q8_host(ref_x)
        bad = []
        xo, xr = _np(ours_x).astype(np.float64), _np(ref_x).astype(np.float64)   # the precondition, gated here too: a flip
        xrel = float(np.abs(xo - xr).max() / (np.abs(xr).max() + 1e-30))           # is only explained by a float within tol
        if xrel > tol: bad.append(f'float in {xrel:.1e} > {tol:.0e}')
        if exact and not (np.array_equal(hd_o.view(np.uint16), d_o.view(np.uint16)) and np.array_equal(hq_o, q_o)):
            bad.append('ours != quantize_q8_1(ours float)')
        if not (np.array_equal(hd_r.view(np.uint16), d_r.view(np.uint16)) and np.array_equal(hq_r, q_r)):
            bad.append('ref != quantize_q8_1(ref float)')
        dq = q_o.astype(np.int32) - q_r.astype(np.int32); diff = dq != 0
        eq = not diff.any() and np.array_equal(d_o.view(np.uint16), d_r.view(np.uint16))
        nflip, dd = int(diff.sum()), int(np.abs(d_o.view(np.int16).astype(np.int32) - d_r.view(np.int16).astype(np.int32)).max(initial=0))
        if diff.any():
            if np.abs(dq).max() > 1: bad.append(f'|dq| {int(np.abs(dq).max())} > 1')
            bnd = np.minimum(q_o, q_r)[diff].astype(np.float64) + 0.5   # the .5 boundary between the codes lo and lo + 1 (roundf)
            a_, b_ = v_o[diff].astype(np.float64), v_r[diff].astype(np.float64)
            ok_ = (np.minimum(a_, b_) - eps * (not exact) <= bnd) & (bnd <= np.maximum(a_, b_) + eps * (not exact))
            if not ok_.all(): bad.append(f'{int((~ok_).sum())} of {nflip} code differences not at a rounding boundary')
        if dd > 1: bad.append(f'd differs by {dd} half ulps')
        good = not bad; self.ok &= good
        st = self.stats.setdefault(name, dict(fmt='q8_1', want_bit=None, tol=None, bit=0, n=0, worst=0.0, ndiag=0, q8=True, flips=0, maxflip=0))
        st['n'] += 1; st['bit'] += eq; st['flips'] += nflip; st['maxflip'] = max(st['maxflip'], nflip)
        sdiff = int((s_o.view(np.uint16) != s_r.view(np.uint16)).sum())
        self.line.append(f'{name}[q8_1] ' + ('equal' if eq else f'{nflip} flips (d {dd} ulp)') + (f' s {sdiff}' if sdiff else '') +
                         ('' if good else ' FAIL(' + '; '.join(bad) + ')'))
        return eq

    def flush(self, head):
        print(head + '  ' + '  '.join(self.line)); self.line = []

    def summary(self):
        print('summary (tensor [format]: bitwise steps / steps, worst rel; gate):')
        for k, s in self.stats.items():
            if s.get('q8'):
                print(f'  {k}[q8_1]: {s["bit"]}/{s["n"]} equal, {s["flips"]} rounding flips (max {s["maxflip"]} in a step); '
                      'gate equal or explained flips (then the chained op bitwise its isolated twin, else a diagnostic)')
                continue
            print(f'  {k}{"[" + s["fmt"] + "]" if s["fmt"] else ""}: {s["bit"]}/{s["n"]} bitwise, worst {s["worst"]:.1e}; '
                  'gate ' + ('none (diagnostic)' if s['ndiag'] == s['n'] else ('bitwise' if s['want_bit'] else f"<= {s['tol']:.0e}") +
                             (f" ({s['ndiag']} steps after a q8_1 flip: diagnostic)" if s['ndiag'] else '')))
        print('PASS' if self.ok else 'FAIL')
        return 0 if self.ok else 1

    def chain(self, name, eq, ours, iso, ref, fmt=None):
        """the op after a q8_1 crossing (q8x returned eq): bitwise its isolated twin iso (the same op fed ref_q8) when the
        q8_1 bytes are equal, and always the error vs llama.cpp's tensor as a diagnostic (the isolated twin carries
        the gate vs llama.cpp)"""
        if eq: self.check(f'{name}=iso', ours, _np(iso), bitwise=True)
        else: self.line.append(f'{name}=iso skipped (q8_1 flips)')
        self.check(name, ours, ref, fmt=fmt, diag=True)
