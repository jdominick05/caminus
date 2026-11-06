"""The PCIe link between the two TP cards, as the batched prefill's exchange uses it (docs/agentic/spec-prefill-cache.md
M5 S5): hipMemcpyPeerAsync on dedicated streams (never torch's cross-device copy_, which puts a two-way barrier between
both devices' CURRENT streams around every peer copy and so would serialize the overlap away).

  guard.sh LOG SECS python -I tools/prefill_link_bench.py [--tokens 2048] [--reps 10] [--json OUT]

Measured (median of reps, host wall around the copies with both cards synchronized; the cards otherwise idle unless said):
  one way   XTX -> XT and XT -> XTX, pushed (the stream on the source card) and pulled (on the destination card), into
            ordinary and uncached VRAM (hipDeviceMallocUncached, what the exchange receives into), for the exchange's
            payload (tokens x 2560 fp32: 21 MB at 2048) and its 1/2, 1/3, 1/4 micro-chunk pieces
  duplex    both directions at once (each card's partial to the other: one all-reduce), pushed and pulled
  gemm      the duplex exchange while an MMQ GEMM (Q8_0 [6144 x 2560], synthetic weights with finite scales) loops on each
            card's compute stream: GEMM ms alone / with the copies, copies alone / with the GEMM (does the copy cost CU
            time, does the GEMM slow the link)
  feed      the two-card feed's P2P copy (tokens x 1360 B of raw token_embd rows, XT -> XTX) alone and beside a duplex
            exchange: the feed and the exchange share the link
  order     cross-device ordering and visibility, a new value each round: card A fills its buffer (a kernel on A's
            compute stream, after the previous round's copy read it), the copy stream waits A's event (and the previous
            round's read of B: the write-after-read wait the exchange needs too) and copies to card B (push: the stream on
            A; pull: on B; into ordinary or uncached VRAM), B's compute stream waits the copy's event and reads B (the
            destination's lines were read on B the round before, so a stale L2 line would show), with HIP events recorded
            with a system-scope release (hipEventReleaseToSystem) and with torch's default events. The exchange's path
            (system-release events, push into ordinary VRAM: 200 rounds of 256 KB and 50 of the full payload each way)
            and every other variant (20 rounds each) must see its own value every round (RESULT PASS). On 11-04 a first
            version without the write-after-read waits read later rounds' values in most rounds of every variant (the
            next round's copy landed before the read): the waits are what the exchange must keep
Needs both cards (cuda:0 = XTX, cuda:1 = XT). Nothing here launches a new kernel: torch fills / sums and the vendored MMQ
(caminus.mmq, every launch checked) only."""
import argparse, ctypes, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'python'))
import torch   # noqa: E402
from caminus import mmq   # noqa: E402
from caminus.hip import hip, check   # noqa: E402

D = 2560
TE_RB = 1360          # raw IQ4_XS token_embd row bytes (the two-card feed's P2P payload per token)


class Uncached:
    """n bytes of uncached VRAM on device dev (hipExtMallocWithFlags(..., hipDeviceMallocUncached = 3), as flashnext_tp.Slots)"""
    def __init__(self, dev, n):
        h = hip(); p = ctypes.c_void_p(); check(h.hipSetDevice(dev))
        check(h.hipExtMallocWithFlags(ctypes.byref(p), ctypes.c_size_t(n), ctypes.c_uint(3)))
        check(h.hipMemset(p, 0, ctypes.c_size_t(n))); check(h.hipDeviceSynchronize())
        self.ptr, self.n, self.dev = p.value, n, dev

    def free(self):
        h = hip(); h.hipSetDevice(self.dev); check(h.hipFree(ctypes.c_void_p(self.ptr))); self.ptr = None


EV_DISABLE_TIMING, EV_RELEASE_TO_SYSTEM = 0x2, 0x80000000     # hip_runtime_api.h hipEventDisableTiming / hipEventReleaseToSystem


class SysEvent:
    """a HIP event recorded with a SYSTEM-scope release (hipEventReleaseToSystem): the work before it is written back past
    the card's L2 when it completes, so another card's copy engine reading over PCIe sees it. torch.cuda.Event's default
    marker is not (hip_runtime_api.h: the default system fence is for the host's event inspection; the release-to-system
    flag exists to ask for one), and the 'order' check below reads stale data through it"""
    def __init__(self, dev):
        h = hip(); e = ctypes.c_void_p(); check(h.hipSetDevice(dev))
        check(h.hipEventCreateWithFlags(ctypes.byref(e), ctypes.c_uint(EV_DISABLE_TIMING | EV_RELEASE_TO_SYSTEM)))
        self.e, self.dev = e, dev

    def record(self, stream):
        h = hip(); check(h.hipSetDevice(self.dev)); check(h.hipEventRecord(self.e, ctypes.c_void_p(stream.cuda_stream)))

    def wait(self, stream):
        """stream (any card) waits for this event"""
        check(hip().hipStreamWaitEvent(ctypes.c_void_p(stream.cuda_stream), self.e, ctypes.c_uint(0)))

    def __del__(self):
        try: hip().hipEventDestroy(self.e)
        except Exception: pass


def peer_copy(dst, ddev, src, sdev, nbytes, stream):
    h = hip()
    h.hipMemcpyPeerAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
    check(h.hipMemcpyPeerAsync(ctypes.c_void_p(dst), ddev, ctypes.c_void_p(src), sdev, ctypes.c_size_t(nbytes), ctypes.c_void_p(stream.cuda_stream)))


def sync():
    for d in (0, 1): torch.cuda.synchronize(d)


def wall(fn, reps):
    fn(); sync(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.min(ts))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tokens', type=int, default=2048); ap.add_argument('--reps', type=int, default=10)
    ap.add_argument('--json', default=None)
    a = ap.parse_args(); T, reps = a.tokens, a.reps
    comp = open('/proc/sys/vm/compaction_proactiveness').read().strip()
    print('compaction_proactiveness', comp, flush=True)
    assert 'XTX' in torch.cuda.get_device_name(0) and 'XTX' not in torch.cuda.get_device_name(1), 'cuda:0 must be the XTX, cuda:1 the XT'
    h = hip()
    for x, y in ((0, 1), (1, 0)):
        h.hipSetDevice(x); h.hipDeviceEnablePeerAccess(y, 0); h.hipGetLastError()   # 'already enabled' is fine
    torch.cuda.set_device(0)
    NB = T * D * 4
    src = [torch.randn(T * D, device=f'cuda:{c}') for c in (0, 1)]
    dst = [torch.zeros(T * D, device=f'cuda:{c}') for c in (0, 1)]
    unc = [Uncached(c, NB) for c in (0, 1)]
    cs = [torch.cuda.Stream(device=c) for c in (0, 1)]                 # copy streams
    ks = [torch.cuda.Stream(device=c) for c in (0, 1)]                 # compute streams
    out = dict(compaction=comp, tokens=T, payload_mb=NB / 1e6)
    gbs = lambda nb, ms: nb / ms / 1e6

    # -- one way / duplex, push / pull, ordinary / uncached ---------------------------------------------------------------
    def one(sdev, ddev, nb, how, kind):
        dp = unc[ddev].ptr if kind == 'uncached' else dst[ddev].data_ptr()
        st = cs[sdev] if how == 'push' else cs[ddev]
        return lambda: peer_copy(dp, ddev, src[sdev].data_ptr(), sdev, nb, st)
    res = {}
    for kind in ('ordinary', 'uncached'):
        for how in ('push', 'pull'):
            for parts in (1, 2, 3, 4):
                nb = NB // parts // 256 * 256
                for sdev, ddev, nm in ((0, 1, 'XTX->XT'), (1, 0, 'XT->XTX')):
                    ms = wall(one(sdev, ddev, nb, how, kind), reps)[0]
                    res[f'{nm} {how} {kind} 1/{parts}'] = (ms, gbs(nb, ms))
                f0, f1 = one(0, 1, nb, how, kind), one(1, 0, nb, how, kind)
                ms = wall(lambda: (f0(), f1()), reps)[0]
                res[f'duplex {how} {kind} 1/{parts}'] = (ms, gbs(2 * nb, ms))
    for k, (ms, g) in res.items(): print(f'  LINK {k:32s} {ms:7.3f} ms  {g:5.2f} GB/s', flush=True)
    out['copies'] = res

    # -- duplex beside an MMQ GEMM on each card -----------------------------------------------------------------------------
    rows, K, fmt = 6144, 2560, 'Q8_0'
    M, W, Y, O, X = [], [], [], [], []
    for c in (0, 1):
        with torch.cuda.device(c):
            M.append(mmq.MMQ(f'cuda:{c}', formats=(fmt,)))
            w = torch.randint(-127, 128, (rows * K // 32, 34), dtype=torch.int8, device=f'cuda:{c}').view(torch.uint8)
            w[:, 0], w[:, 1] = 0x00, 0x2C                                 # each block's f16 scale = 0.0625 (finite)
            W.append(w.reshape(-1).contiguous()); x = torch.randn(T, K, device=f'cuda:{c}'); X.append(x)
            Y.append(torch.zeros(mmq.y_bytes(K, T), dtype=torch.uint8, device=f'cuda:{c}')); O.append(torch.zeros(T * rows, device=f'cuda:{c}'))
            M[c].quantize(x, K, T, Y[c], M[c].layout(fmt))
    sync()
    NG = 8
    def gemms():
        for c in (0, 1):
            with torch.cuda.device(c), torch.cuda.stream(ks[c]):
                for _ in range(NG): M[c].mm(fmt, W[c], rows, K, Y[c], T, O[c])
    f0, f1 = one(0, 1, NB, 'push', 'uncached'), one(1, 0, NB, 'push', 'uncached')
    NX = 4
    def xch():
        for _ in range(NX): f0(); f1()
    g_alone = wall(gemms, reps)[0]; x_alone = wall(xch, reps)[0]; both = wall(lambda: (gemms(), xch()), reps)[0]
    def gemm_stream_alone(c):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(ks[c])
        with torch.cuda.device(c), torch.cuda.stream(ks[c]):
            for _ in range(NG): M[c].mm(fmt, W[c], rows, K, Y[c], T, O[c])
        e1.record(ks[c]); sync(); return e0.elapsed_time(e1)
    ga = [float(np.median([gemm_stream_alone(c) for _ in range(reps)])) for c in (0, 1)]
    # each side's own time while the other runs: events on its stream
    def timed_streams():
        e = {k: torch.cuda.Event(enable_timing=True) for k in ('g0s', 'g0e', 'g1s', 'g1e', 'x0s', 'x0e')}
        for c in (0, 1): e[f'g{c}s'].record(ks[c])
        e['x0s'].record(cs[0])
        gemms(); xch()
        for c in (0, 1): e[f'g{c}e'].record(ks[c])
        e['x0e'].record(cs[0]); sync()
        return e['g0s'].elapsed_time(e['g0e']), e['g1s'].elapsed_time(e['g1e']), e['x0s'].elapsed_time(e['x0e'])
    tl = [timed_streams() for _ in range(reps)]
    g0w, g1w, xw = (float(np.median([t[i] for t in tl])) for i in range(3))
    out['gemm'] = dict(gemm_xtx_alone_ms=ga[0], gemm_xt_alone_ms=ga[1], gemm_alone_ms=g_alone, xchg_alone_ms=x_alone, together_ms=both, gemm_xtx_with_ms=g0w, gemm_xt_with_ms=g1w,
                       xchg_push_stream_with_ms=xw, n_gemm=NG, n_xchg=NX)
    print(f'  GEMM {NG} x MMQ {fmt} [{rows} x {K}] x {T} per card alone {g_alone:.2f} ms; {NX} duplex exchanges alone {x_alone:.2f} ms; '
          f'together {both:.2f} ms (GEMM streams alone: XTX {ga[0]:.2f} / XT {ga[1]:.2f} ms; with the copies: XTX {g0w:.2f} / XT {g1w:.2f} ms; '
          f'the XTX push stream {xw:.2f} ms)', flush=True)

    # -- the feed's P2P copy beside the exchange ---------------------------------------------------------------------------
    fb = T * TE_RB // 256 * 256
    fsrc = torch.zeros(fb, dtype=torch.uint8, device='cuda:1'); fdst = torch.zeros(fb, dtype=torch.uint8, device='cuda:0')
    fs = torch.cuda.Stream(device=1)
    feed = lambda: peer_copy(fdst.data_ptr(), 0, fsrc.data_ptr(), 1, fb, fs)
    fe_alone = wall(feed, reps)[0]
    def feed_timed(with_x):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if with_x: f0(); f1()
        e0.record(fs); feed(); e1.record(fs)
        sync(); return e0.elapsed_time(e1)
    fe_with = float(np.median([feed_timed(True) for _ in range(reps)]))
    x1 = wall(lambda: (f0(), f1()), reps)[0]; x1f = wall(lambda: (f0(), f1(), feed()), reps)[0]
    out['feed'] = dict(feed_mb=fb / 1e6, feed_alone_ms=fe_alone, feed_with_xchg_ms=fe_with, xchg_ms=x1, xchg_plus_feed_ms=x1f)
    print(f'  FEED {fb / 1e6:.2f} MB XT -> XTX alone {fe_alone:.3f} ms, issued beside a duplex exchange {fe_with:.3f} ms (its own stream\'s span); '
          f'one duplex exchange {x1:.3f} ms, with the feed copy {x1f:.3f} ms', flush=True)

    # -- ordering and visibility across the cards ---------------------------------------------------------------------------
    n = 1 << 16                                                       # 256 KB: B's L2 keeps it between rounds
    A = [torch.zeros(n, device=f'cuda:{c}') for c in (0, 1)]; Bn = [torch.zeros(n, device=f'cuda:{c}') for c in (0, 1)]
    Bu = [Uncached(c, 4 * n) for c in (0, 1)]
    order = {}
    def ev_new(kind, dev):
        return SysEvent(dev) if kind == 'sys' else torch.cuda.Event()
    def ev_rec(kind, e, st):
        if kind == 'sys': e.record(st)
        else: e.record(st)
    def ev_wait(kind, e, st):
        if kind == 'sys': e.wait(st)
        else: st.wait_event(e)
    h.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    nfull = NB // 4
    A += [torch.zeros(nfull, device=f'cuda:{c}') for c in (0, 1)]; Bn += [torch.zeros(nfull, device=f'cuda:{c}') for c in (0, 1)]
    Bu += [Uncached(c, NB) for c in (0, 1)]
    variants = [('sys', 'push', 'ordinary', 0, 200), ('sys', 'push', 'ordinary', 1, 50)] + \
               [(e, hw, k, 0, 20) for e in ('sys', 'torch') for hw in ('push', 'pull') for k in ('ordinary', 'uncached')
                if (e, hw, k) != ('sys', 'push', 'ordinary')]
    for evk, how, kind, big, rounds in variants:
            n = nfull if big else (1 << 16)
            for sdev, ddev in ((0, 1), (1, 0)):
                sums = []; ia, ib = sdev + 2 * big, ddev + 2 * big
                st = cs[sdev] if how == 'push' else cs[ddev]
                tmp = torch.empty(n, device=f'cuda:{ddev}')
                prev_copy = prev_read = None
                for r in range(rounds):
                    val = float(r % 64 + 1) + 0.25 * sdev + (0.125 if (r // 64) % 2 else 0.0)
                    if prev_copy is not None: ev_wait(evk, prev_copy, ks[sdev])     # the fill after the last copy read A
                    with torch.cuda.device(sdev), torch.cuda.stream(ks[sdev]): A[ia].fill_(val)
                    e_fill = ev_new(evk, sdev); ev_rec(evk, e_fill, ks[sdev]); ev_wait(evk, e_fill, st)
                    if prev_read is not None: ev_wait(evk, prev_read, st)           # the copy after the last read of B
                    dp = Bu[ib].ptr if kind == 'uncached' else Bn[ib].data_ptr()
                    peer_copy(dp, ddev, A[ia].data_ptr(), sdev, 4 * n, st)
                    e_copy = ev_new(evk, st.device.index); ev_rec(evk, e_copy, st); ev_wait(evk, e_copy, ks[ddev])
                    with torch.cuda.device(ddev), torch.cuda.stream(ks[ddev]):
                        if kind == 'ordinary': s = (Bn[ib] != val).sum()         # a kernel reads B on its card (its L2 too)
                        else:                                                       # (no torch tensor over the uncached block)
                            check(h.hipMemcpyAsync(ctypes.c_void_p(tmp.data_ptr()), ctypes.c_void_p(Bu[ib].ptr), ctypes.c_size_t(4 * n), 3,
                                                   ctypes.c_void_p(ks[ddev].cuda_stream)))
                            s = (tmp != val).sum()
                    sums.append(s)
                    e_read = ev_new(evk, ddev); ev_rec(evk, e_read, ks[ddev])
                    prev_copy, prev_read = e_copy, e_read
                sync()
                bad = sum(1 for s in sums if int(s.item()) != 0)
                key = f'{evk} {how} {kind} {4 * n / 1e6:.2f}MB {sdev}->{ddev}'; order[key] = (bad, rounds)
                print(f'  ORDER {evk:5s} events {how:4s} {kind:8s} {4 * n / 1e6:6.2f} MB card {sdev} -> {ddev}: {rounds - bad} / {rounds} rounds saw their own value'
                      + ('  (the exchange\'s path)' if (evk, how, kind) == ('sys', 'push', 'ordinary') else ''), flush=True)
    out['order_bad_rounds'] = order
    ok = all(v[0] == 0 for v in order.values())
    for u in unc + Bu: u.free()
    if a.json: json.dump(out, open(a.json, 'w'), indent=1)
    print('RESULT ' + ('PASS' if ok else 'FAIL (a round read a stale or missing value)'), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
