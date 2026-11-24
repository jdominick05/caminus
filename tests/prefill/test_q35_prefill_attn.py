"""The qwen35 lines' own prefill tiled attention, on random q and cache rows, no model file. Kernels (--kernel, default all):
  dense   kernels/q35_prefill.hip q35p_attn_wm through caminus.prefill.q35own.Ops.attn_wm (the dense line: F16 KV)
  moe     kernels/q35m_prefill.hip q35mp_attn through caminus.prefill.q35moe.Ops.attn_wm (Qwen3.6-35B-A3B, F16 KV)
  moe-q8  q35mp_attn_q8 from the Q8 object (kernels/q35m_prefill_kvq8.hip), over Q8_0 rows (ref_kv's encode of random
          rows; the reference reads the decoded values)
Each runs:
  - against float64 causal attention (the q rounded to fp16 as the kernel reads it): max / mean |difference| of the
    reference's rms, at chunks from position 0, past the 32-position staging tiles, from an odd position (the first
    8-position block partly before the chunk), deep (8K), its line's geometry (24 query heads over 4 kv heads; the 35B's
    16 over 2) and the other (6 or 8 per kv head)
  - the same bits at every launch split (attn_blocks with a small budget) and every chunk geometry (the chunk cut at an
    odd position, an 8-position block edge and a 32-position tile edge; chunks of 16): a row's sums depend only on
    absolute positions (the prompt cache's stops)
  - a peaked softmax (two planted keys take nearly all the weight: the rescale path at depth) against float64
  - shapes it does not run are refused on the host (head dim 128, 12 query heads per kv head, a block range outside the chunk)
  - moe / moe-q8: an exactly sized cache (its own hipMalloc of rows x row bytes, past torch's caching allocator, wrapped
    through DLPack; hipMemGetAddressRange must report that size): chunks whose last row is the allocation's last row, so
    a read past it faults here, against float64
  - dense: against the per-(row, head) kernel it replaced (q35p_attn16, fp32 q), within the float64 bounds
  - timing (--speed): a 2048-row chunk at P0 = 0 / 8192 / 30720 (the yardstick rows of docs/dev/yardsticks.md) in the
    kernel's line geometry, median of 5 after a warm-up, TFLOPS as there (4 N (P0 + N / 2) heads D); moe / moe-q8 also run
    the dense line's q35p_attn_wm (over the same rows decoded to fp16 for moe-q8: the kernel they replace) at the 35B's
    geometry; the slowest single launch at ATTN_WM_TP
  tools/guard.sh LOG 900 python -I tests/prefill/test_q35_prefill_attn.py [--kernel dense,moe,moe-q8] [--speed]
New kernel code: a non-display card alone first (HIP_VISIBLE_DEVICES=<its index>). Prints PASS / FAIL and RESULT."""
import argparse, ctypes, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import numpy as np, torch
from caminus import ref_kv
from caminus.device import capacity, hip as _hip
from caminus.engine import k3
from caminus.prefill import q35moe, q35own

ap = argparse.ArgumentParser(); ap.add_argument('--speed', action='store_true'); ap.add_argument('--kernel', default='dense,moe,moe-q8')
A = ap.parse_args()
D = 256
dev = torch.device('cuda:0')
FAILS = []


def check(c, msg):
    print(('PASS ' if c else 'FAIL ') + msg, flush=True)
    if not c: FAILS.append(msg)


def bits(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


ROWS = 32768 + 64                                      # (F16 caches; Q8 ones hold Q8_ROWS: the deepest case is at 8K)
Q8_ROWS = 8192 + 128
need = (ROWS * 2 * 8 * D * 2 + 6 * 2048 * 32 * D * 4) / 2 ** 20 + 64   # (the cache at 8 kv heads; q / out buffers)
capacity.check([0], [need], f'test_q35_prefill_attn ({need:.0f} MiB)')
print(torch.cuda.get_device_name(dev), flush=True)
g = torch.Generator(device='cuda').manual_seed(11)
rng = np.random.default_rng(11)


class Cache:
    """a cache of `rows` positions for nkv kv heads: dev (the kernel's: fp16 or Q8_0 uint8) and host (float64
    [rows][K | V][nkv][D]: the values the kernel reads, Q8 decoded)"""

    def __init__(self, nkv, rows, q8, dev_t=None):
        self.nkv, self.rows, self.q8 = nkv, rows, q8
        if q8:
            lay = ref_kv.Q8KVLayout(nkv, D)
            raw = np.empty((rows, lay.row_bytes), dtype=np.uint8)
            for r0 in range(0, rows, 4096):
                r1 = min(rows, r0 + 4096)
                raw[r0:r1] = ref_kv.encode_q8_kv(rng.normal(0.0, 0.5, (r1 - r0, nkv, D)).astype(np.float32),
                                                 rng.normal(0.0, 0.5, (r1 - r0, nkv, D)).astype(np.float32), lay)
            self.raw, self.lay = raw, lay
            self.dev = torch.from_numpy(raw.reshape(-1)).to(dev) if dev_t is None else dev_t.copy_(torch.from_numpy(raw.reshape(-1)))
        else:
            self.dev = torch.empty(rows * 2 * nkv * D, dtype=torch.float16, device=dev) if dev_t is None else dev_t.view(torch.float16)
            self.dev.normal_(0.0, 0.5, generator=g)   # (in place)

    def host(self, last):
        if self.q8:
            kd, vd = ref_kv.decode_q8_kv(self.raw[:last], self.lay)
            return np.stack((kd, vd), axis=1).astype(np.float64)
        return self.dev.view(-1, 2, self.nkv, D)[:last].cpu().double().numpy()   # (to the host first: no float64 copy on the card)

    def plant_k(self, p, u):
        """position p's K = 2 u in every kv head (u: +-1 per dim; exact in both formats)"""
        if self.q8:
            k = np.tile((2.0 * u.cpu().numpy()).astype(np.float32), (1, self.nkv, 1))
            v = ref_kv.decode_q8_kv(self.raw[p:p + 1], self.lay)[1].astype(np.float32)
            self.raw[p] = ref_kv.encode_q8_kv(k, v, self.lay)[0]
            self.dev.view(self.rows, -1)[p].copy_(torch.from_numpy(self.raw[p]))
        else:
            self.dev.view(-1, 2, self.nkv, D)[p, 0] = (2.0 * u).half()

    def f16(self):
        """the same rows as an fp16 cache (the dense kernel's input: Q8 decoded and rounded as the Q8 kernel stages them)"""
        if not self.q8: return self.dev
        out = torch.empty(self.rows * 2 * self.nkv * D, dtype=torch.float16, device=dev)
        for r0 in range(0, self.rows, 4096):
            r1 = min(self.rows, r0 + 4096)
            kd, vd = ref_kv.decode_q8_kv(self.raw[r0:r1], self.lay)
            out[r0 * 2 * self.nkv * D:r1 * 2 * self.nkv * D] = torch.from_numpy(np.stack((kd, vd), axis=1).astype(np.float16).reshape(-1)).to(dev)
        return out


def ref_rows(q, kv, nkv, heads, pos0, rows):
    """float64 attention of chunk rows `rows` (q rounded to fp16, as the kernel reads it): [len(rows), heads, D]"""
    G = heads // nkv
    qq = q.view(-1, heads, D).cpu().half().double().numpy()   # (rounded on the host: no temporaries on the card)
    out = np.zeros((len(rows), heads, D))
    for i, r in enumerate(rows):
        P = pos0 + r + 1
        for h in range(heads):
            s = kv[:P, 0, h // G] @ qq[r, h] / np.sqrt(D); e = np.exp(s - s.max()); out[i, h] = (e @ kv[:P, 1, h // G]) / e.sum()
    return out


def rel(a, b):
    rms = np.sqrt((b * b).mean()); d = np.abs(a - b)
    return d.max() / rms, d.mean() / rms


# ---- exactly sized allocations (a read past the last row's last byte faults) ------------------------------------------
class _DLDevice(ctypes.Structure):
    _fields_ = [('device_type', ctypes.c_int32), ('device_id', ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [('code', ctypes.c_uint8), ('bits', ctypes.c_uint8), ('lanes', ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p), ('device', _DLDevice), ('ndim', ctypes.c_int32), ('dtype', _DLDataType),
                ('shape', ctypes.POINTER(ctypes.c_int64)), ('strides', ctypes.POINTER(ctypes.c_int64)),
                ('byte_offset', ctypes.c_uint64)]


class _DLManaged(ctypes.Structure):
    pass


_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManaged))
_DLManaged._fields_ = [('dl_tensor', _DLTensor), ('manager_ctx', ctypes.c_void_p), ('deleter', _DELETER)]
_EXACT = {}     # pointer -> (the DLPack structs, kept alive until torch's deleter runs)


@_DELETER
def _exact_free(m):
    p = m.contents.dl_tensor.data
    _EXACT.pop(p, None)
    _hip.check(_hip.hip().hipFree(ctypes.c_void_p(p)))


def exact_alloc(nbytes):
    """a uint8 device tensor that is its own hipMalloc of exactly nbytes (freed with the tensor), and the (base, size)
    hipMemGetAddressRange reports for its first byte"""
    h = _hip.hip(); p = ctypes.c_void_p(); base = ctypes.c_void_p(); size = ctypes.c_size_t()
    h.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    h.hipFree.argtypes = [ctypes.c_void_p]
    h.hipMemGetAddressRange.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
    _hip.check(h.hipMalloc(ctypes.byref(p), ctypes.c_size_t(int(nbytes))))
    _hip.check(h.hipMemGetAddressRange(ctypes.byref(base), ctypes.byref(size), p))
    shape = (ctypes.c_int64 * 1)(int(nbytes))
    m = _DLManaged()
    m.dl_tensor = _DLTensor(p.value, _DLDevice(10, torch.cuda.current_device()), 1, _DLDataType(1, 8, 1),  # kDLROCM, uint8
                            shape, None, 0)
    m.manager_ctx = None; m.deleter = _exact_free
    _EXACT[p.value] = (m, shape)
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    t = torch.from_dlpack(PyCapsule_New(ctypes.addressof(m), b'dltensor', None))
    return t, (base.value, size.value)


def suite(name):
    q8 = name == 'moe-q8'
    O = q35own.Ops(dev) if name == 'dense' else q35moe.Ops(dev, q8=q8)
    rows_ = Q8_ROWS if q8 else ROWS
    geo, other = ((24, 4), (32, 4)) if name == 'dense' else ((16, 2), (24, 4))
    print(f'---- {name}: {O.WM_KERNEL[q8]} ({"Q8_0" if q8 else "F16"} KV)', flush=True)

    def wm(q, cache, pos0, n, heads, nkv, out):
        for b0, b1 in O.attn_blocks(pos0, n): O.attn_wm(q, cache, pos0, n, heads, nkv, D, 1.0 / np.sqrt(D), out, cache_rows(cache, nkv), b0, b1)

    def cache_rows(cache, nkv):
        return k3.kv_rows(cache, nkv * D, q8)

    # ---- accuracy against float64 ---------------------------------------------------------------------------------------
    (H, NKV), (H2, NKV2) = geo, other
    cases = [(H, NKV, 0, 37), (H, NKV, 600, 37), (H, NKV, 601, 50), (H, NKV, 8191, 64), (H2, NKV2, 300, 41)]
    for heads, nkv, pos0, n in cases:
        C = Cache(nkv, rows_, q8)
        q = torch.randn(n * heads * D, generator=g, device=dev) * 0.15
        out = torch.zeros_like(q)
        wm(q, C.dev, pos0, n, heads, nkv, out); torch.cuda.synchronize()
        rows = list(range(n)) if n * (pos0 + n) < 400_000 else list(range(0, n, 7)) + [n - 1]
        on = out.view(n, heads, D).cpu().double().numpy()[rows]
        mx, mn = rel(on, ref_rows(q, C.host(pos0 + n), nkv, heads, pos0, rows))
        check(np.isfinite(on).all() and mx < 2e-2 and mn < 2e-3,
              f'{name}: {heads} q heads / {nkv} kv, rows at positions {pos0} .. {pos0 + n - 1} vs float64: max {mx:.2e}, mean {mn:.2e} of rms')
        if name == 'dense' and pos0 == 600:
            old = torch.zeros_like(q)
            for r0, r1 in O.attn_rows(pos0, n): O.attn16(q, C.dev, pos0, n, heads, nkv, D, 1.0 / np.sqrt(D), old, rows_, r0, r1)
            torch.cuda.synchronize()
            mo, no = rel(on, old.view(n, heads, D).cpu().double().numpy()[rows])
            check(mo < 2e-2 and no < 2e-3, f'{name}: against q35p_attn16 (the kernel it replaced) at {pos0}: max {mo:.2e}, mean {no:.2e} of rms')
        del C

    # ---- a peaked softmax: q along a direction u, the keys at two planted positions (one early, one inside the chunk)
    # 2 u, every score at most ~0.5 elsewhere: those keys take nearly all the weight (the online softmax's rescales at depth)
    heads, nkv, pos0, n = H, NKV, 5000, 40
    C = Cache(nkv, rows_, q8)
    u = torch.where(torch.rand(D, generator=g, device=dev) < 0.5, -1.0, 1.0)
    q = (u.repeat(n * heads) * 0.25 + torch.randn(n * heads * D, generator=g, device=dev) * 0.02).contiguous()
    for p_ in (7, pos0 + 20): C.plant_k(p_, u)
    out = torch.zeros_like(q); wm(q, C.dev, pos0, n, heads, nkv, out); torch.cuda.synchronize()
    rows = list(range(n)); on = out.view(n, heads, D).cpu().double().numpy()
    mx, mn = rel(on, ref_rows(q, C.host(pos0 + n), nkv, heads, pos0, rows))
    check(np.isfinite(on).all() and mx < 2e-2 and mn < 2e-3, f'{name}: peaked softmax (planted keys at 7 and {pos0 + 20}, rows from {pos0}) vs float64: max {mx:.2e}, mean {mn:.2e} of rms')
    del C

    # ---- the same bits at every launch split and chunk geometry ---------------------------------------------------------
    heads, nkv, pos0, n = H, NKV, 1000, 300
    C = Cache(nkv, rows_, q8); cache = C.dev
    q = torch.randn(n * heads * D, generator=g, device=dev) * 0.15
    one = torch.zeros_like(q); wm(q, cache, pos0, n, heads, nkv, one)
    keep = q35own.ATTN_WM_TP
    q35own.ATTN_WM_TP = 8 * 8 * 1400                   # (a few blocks per launch)
    spl = torch.zeros_like(q); wm(q, cache, pos0, n, heads, nkv, spl)
    nl = len(O.attn_blocks(pos0, n)); q35own.ATTN_WM_TP = keep
    torch.cuda.synchronize()
    check(nl > 3 and bits(one, spl), f'{name}: {nl} launches instead of {len(O.attn_blocks(pos0, n))}: the same bits')
    HD = heads * D
    for cut in (41, 64, 24):                            # (position 1041: a pair split across chunks; 1064: a block edge;
        o3 = torch.zeros_like(q)                        #  1024: a 32-position tile edge)
        wm(q[:cut * HD], cache, pos0, cut, heads, nkv, o3[:cut * HD])
        wm(q[cut * HD:], cache, pos0 + cut, n - cut, heads, nkv, o3[cut * HD:])
        torch.cuda.synchronize()
        check(bits(one, o3), f'{name}: the chunk cut at row {cut} (position {pos0 + cut}): the same bits')
    o16 = torch.zeros_like(q)
    for r in range(0, n, 16):
        m_ = min(16, n - r); wm(q[r * HD:(r + m_) * HD], cache, pos0 + r, m_, heads, nkv, o16[r * HD:(r + m_) * HD])
    torch.cuda.synchronize()
    check(bits(one, o16), f'{name}: chunks of 16: the same bits')

    # ---- refusals -------------------------------------------------------------------------------------------------------
    for what, args in (('head dim 128', (heads, nkv, 128)), ('12 query heads per kv head', (24, 2, D))):
        try:
            O.attn_wm(q, cache, pos0, n, args[0], args[1], args[2], 0.0625, one, cache_rows(cache, args[1]) if args[2] == D else 1); refused = False
        except ValueError: refused = True
        check(refused, f'{name}: attn_wm refuses {what}')
    B0, B1 = O._wm_span(pos0, n)
    try:
        O.attn_wm(q, cache, pos0, n, heads, nkv, D, 0.0625, one, cache_rows(cache, nkv), B0, B1 + 1); refused = False
    except ValueError: refused = True
    check(refused, f'{name}: attn_wm refuses blocks past the chunk')
    del C, cache

    # ---- an exactly sized cache: the chunk's last row is the allocation's last ------------------------------------------
    if name != 'dense':
        for rx, pos0, n in ((37, 0, 37), (3001, 2901, 100)):
            pb = k3.kv_pos_bytes(NKV * D, q8)
            raw, (base, size) = exact_alloc(rx * pb)
            exact = base == raw.data_ptr() and size == rx * pb == raw.numel() and raw.is_cuda
            C = Cache(NKV, rx, q8, dev_t=raw)
            q = torch.randn(n * H * D, generator=g, device=dev) * 0.15
            out = torch.zeros_like(q); wm(q, C.dev, pos0, n, H, NKV, out); torch.cuda.synchronize()
            rows = list(range(n)); on = out.view(n, H, D).cpu().double().numpy()
            mx, mn = rel(on, ref_rows(q, C.host(rx), NKV, H, pos0, rows))
            check(exact and np.isfinite(on).all() and mx < 2e-2 and mn < 2e-3,
                  f'{name}: an exactly sized cache ({rx} rows x {pb} B, one allocation of that size: {exact}), rows {pos0} .. {rx - 1}: '
                  f'max {mx:.2e}, mean {mn:.2e} of rms')
            del C, raw

    # ---- timing ---------------------------------------------------------------------------------------------------------
    if A.speed:
        N = 2048
        q = torch.randn(N * H * D, generator=g, device=dev) * 0.15; out = torch.zeros_like(q)

        def med(fn, reps=5):
            fn(); torch.cuda.synchronize(); ts = []
            for _ in range(reps):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); fn(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1))
            return float(np.median(ts))
        C = Cache(NKV, 30720 + N, q8); cache = C.dev
        Od = q35own.Ops(dev) if name != 'dense' else None
        c16 = C.f16() if Od is not None else None
        slow = 0.0
        for P0 in (0, 8192, 30720):
            fl = 4.0 * N * (P0 + N / 2) * H * D
            t_n = med(lambda: wm(q, cache, P0, N, H, NKV, out))
            line = f'NOTE {name} chunk {N} at P0 {P0:6d}: {O.WM_KERNEL[q8]} {t_n:8.2f} ms ({fl / t_n / 1e9:5.1f} TF, {len(O.attn_blocks(P0, N))} launches)'
            if Od is not None:
                def old_():
                    for b0, b1 in Od.attn_blocks(P0, N): Od.attn_wm(q, c16, P0, N, H, NKV, D, 0.0625, out, C.rows, b0, b1)
                t_o = med(old_)
                line += f' | q35p_attn_wm {t_o:8.2f} ms ({fl / t_o / 1e9:5.1f} TF) | {t_o / t_n:5.2f}x'
            print(line, flush=True)
            for b0, b1 in O.attn_blocks(P0, N):
                slow = max(slow, med(lambda: O.attn_wm(q, cache, P0, N, H, NKV, D, 0.0625, out, C.rows, b0, b1), 3))
        print(f'NOTE {name}: the slowest single launch at ATTN_WM_TP {q35own.ATTN_WM_TP}: {slow:.2f} ms', flush=True)
        check(slow < 100.0, f'{name}: every launch under 100 ms (slowest {slow:.2f} ms)')
        del C, cache, c16


for k in A.kernel.split(','):
    suite(k)
    torch.cuda.empty_cache()
print('RESULT', 'FAIL' if FAILS else 'PASS'); sys.exit(1 if FAILS else 0)
